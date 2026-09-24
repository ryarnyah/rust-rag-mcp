//! HNSW Vector Database with Memory-Mapped Storage
//!
//! A high-performance, thread-safe hierarchical navigable small world (HNSW) index
//! for approximate nearest neighbor search on large-scale vector datasets.
//!
//! # Features
//! - Memory-mapped storage for efficient I/O and scalability
//! - Soft-delete support with optional compaction
//! - Thread-safe async operations with Tokio
//! - Cosine distance metric with proper NaN handling
//! - Configurable HNSW parameters (m, ef_construction, max_level)
//!
//! # Example
//! ```no_run
//! use rust_rag_mcp::r_vector::{Config, AsyncVectorDb};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let cfg = Config::new(128).with_m(8).with_ef_construction(200);
//!     let db = AsyncVectorDb::open("vectors.db", cfg).await?;
//!
//!     // Insert vector
//!     let id = db.insert(&vec![0.1; 128], None).await?;
//!
//!     // Search
//!     let results = db.search(&vec![0.1; 128], 10, 200).await?;
//!     for hit in results {
//!         println!("id={}, score={:.4}", hit.id, hit.score);
//!     }
//!
//!     Ok(())
//! }
//! ```

use fs2::FileExt;
use memmap2::{MmapMut, MmapOptions};
use ordered_float::OrderedFloat;
use std::cell::UnsafeCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

use crate::wal::{WalOpType, WalRecord, WriteAheadLog};

// ============================================================================
//  mmap helpers
// ============================================================================

/// System page size in bytes.
///
/// Used to page-align the vector row area so rows don't straddle page
/// boundaries (which would turn every row access into two page faults).
fn page_size() -> usize {
    static PAGE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *PAGE.get_or_init(sys_page_size)
}

#[cfg(unix)]
fn sys_page_size() -> usize {
    let sz = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if sz > 0 { sz as usize } else { 4096 }
}

#[cfg(not(unix))]
fn sys_page_size() -> usize {
    4096
}

/// Round `x` up to a multiple of `align` (which must be a power of two).
const fn align_up(x: usize, align: usize) -> usize {
    (x + align - 1) & !(align - 1)
}

/// Map `file` read-write and apply access hints for the database workload:
/// every consumer reads by id (row lookup, graph hop, metadata record), so
/// access is effectively random and kernel read-ahead only wastes I/O.
fn map_file(file: &File) -> Result<MmapMut> {
    let m = unsafe { MmapOptions::new().map_mut(file)? };
    #[cfg(unix)]
    if let Err(e) = m.advise(memmap2::Advice::Random) {
        // Advisory only: never fail a mapping because madvise was refused.
        tracing::debug!(error = %e, "madvise(RANDOM) failed");
    }
    Ok(m)
}

// ============================================================================
//  Error Types
// ============================================================================

/// Custom error type for vector database operations
#[derive(Error, Debug)]
pub enum VectorDbError {
    /// I/O error during file operations
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Vector dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    /// ID is out of bounds
    #[error("ID {id} out of bounds, capacity is {capacity}")]
    IdOutOfBounds { id: u32, capacity: u32 },

    /// Attempted to update a deleted vector
    #[error("Cannot update tombstoned vector; insert a new one instead")]
    UpdateDeletedVector,

    /// File corruption detected
    #[error("Database file corruption: {0}")]
    Corruption(String),

    /// Configuration mismatch with existing file
    #[error("Configuration mismatch: {0}")]
    ConfigMismatch(String),

    #[error("cosine distance is undefined for a zero vector")]
    ZeroVector,

    /// Distance computation produced NaN
    #[error("Distance computation produced NaN for vectors")]
    NaNDistance,
}

pub type Result<T> = std::result::Result<T, VectorDbError>;

// ============================================================================
//  Configuration
// ============================================================================

/// Configuration for HNSW index
///
/// All parameters can be customized via builder methods. Sensible defaults
/// are provided for small to medium datasets.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Vector dimensionality (must be > 0)
    pub dim: usize,
    /// Maximum number of bidirectional connections per node (layers 1+)
    pub m: usize,
    /// Maximum connections for layer 0 (default: 1.5*m, or 2*m when set via with_m)
    pub m0: usize,
    /// Maximum layer level for new nodes
    pub max_level: usize,
    /// Size of dynamic candidate list during insertion
    pub ef_construction: usize,
    /// Random seed for reproducible level assignment
    pub seed: u64,
    /// Initial capacity (number of vectors)
    pub initial_capacity: usize,
    /// Maximum WAL segment file size in bytes before rotation (0 = no rotation)
    pub max_wal_segment_size: u64,
    /// Maximum total WAL size across all segments in bytes before cleanup (0 = no limit)
    pub max_total_wal_size: u64,
    /// Maximum number of WAL segments before cleanup (0 = no limit)
    pub max_wal_segments: u32,
}

impl Config {
    /// Create new config with given dimension, all else at defaults
    ///
    /// # Defaults (HNSW parameters):
    /// - `m`: 20 (connections per layer)
    /// - `m0`: 30 (max connections for layer 0, ~1.5*m)
    /// - `max_level`: 7 (maximum tree depth)
    /// - `ef_construction`: 150 (search expansion during construction)
    /// - `max_wal_segment_size`: 64 MB (rotate WAL when segment exceeds this)
    /// - `max_total_wal_size`: 256 MB (cleanup old segments when total exceeds this)
    ///
    /// # Panics
    /// If `dim == 0`
    ///
    /// # Example
    /// ```
    /// use rust_rag_mcp::r_vector::Config;
    /// let cfg = Config::new(128);
    /// assert_eq!(cfg.dim, 128);
    /// assert_eq!(cfg.m, 20);  // default
    /// ```
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "dimension must be > 0");
        // Generate random seed from system time
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| {
                let nanos = d.subsec_nanos() as u64;
                let secs = d.as_secs();
                secs.wrapping_mul(0x9E37_79B9) ^ nanos.wrapping_mul(0x7F4A_7C15)
            })
            .unwrap_or(0x9E37_79B9_7F4A_7C15);

        Self {
            dim,
            m: 20,
            m0: 30,
            max_level: 7,
            ef_construction: 150,
            seed,
            initial_capacity: 1024,
            max_wal_segment_size: 64 * 1024 * 1024, // 64 MB per segment
            max_total_wal_size: 256 * 1024 * 1024,  // 256 MB total across all segments
            max_wal_segments: 16,                   // max 16 WAL segments
        }
    }

    /// Set M (connections per layer 1+) and adjust m0 accordingly
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self.m0 = m * 2;
        self
    }

    /// Set maximum layer level for new nodes
    pub fn with_max_level(mut self, l: usize) -> Self {
        self.max_level = l;
        self
    }

    /// Set ef_construction (larger = better quality, slower inserts)
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set initial vector capacity
    pub fn with_capacity(mut self, c: usize) -> Self {
        self.initial_capacity = c;
        self
    }

    /// Set random seed for reproducible node level assignment
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Set maximum WAL segment file size in bytes before rotation
    ///
    /// When a WAL segment exceeds this size, it is rotated to a numbered segment
    /// and a new active WAL file is created. Set to 0 to disable rotation.
    pub fn with_max_wal_segment_size(mut self, max_wal_segment_size: u64) -> Self {
        self.max_wal_segment_size = max_wal_segment_size;
        self
    }

    /// Set maximum total WAL size across all segments in bytes before cleanup
    ///
    /// When total size of all WAL segments (including active) exceeds this limit,
    /// the oldest checkpointed segments are deleted. Set to 0 to disable cleanup.
    pub fn with_max_total_wal_size(mut self, max_total_wal_size: u64) -> Self {
        self.max_total_wal_size = max_total_wal_size;
        self
    }

    /// Set maximum number of WAL segments before cleanup
    ///
    /// When the number of segments exceeds this limit, the oldest checkpointed
    /// segments are deleted. Set to 0 to disable.
    pub fn with_max_wal_segments(mut self, max_wal_segments: u32) -> Self {
        self.max_wal_segments = max_wal_segments;
        self
    }
}

// ============================================================================
//  Numeric Helpers
// ============================================================================

/// Dot product of two vectors
///
/// Accumulates in `f64` for numerical stability, without allocating: the
/// inputs are widened element-wise and the reduction is auto-vectorized by
/// LLVM (AVX2/AVX-512 when available). Widening to `f64` first means no
/// intermediate rounding and no underflow for squared `f32` magnitudes.
///
/// # Panics
/// Panics if `a.len() != b.len()`.
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "vector dimension mismatch in dot product: {} vs {}",
        a.len(),
        b.len()
    );

    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64) * (y as f64))
        .sum()
}

/// Euclidean norm (L2) of a vector
#[inline]
fn norm(a: &[f32]) -> f64 {
    dot(a, a).sqrt()
}

/// True for an all-zero (or all-±0.0) row: cosine distance to it is
/// undefined, so such rows are treated as **placeholder rows** — stored with
/// their metadata (used by `rag.rs` for document metadata) but never entered
/// into the HNSW graph, and therefore never returned by a search.
///
/// Equivalent to `norm(v) == 0.0`: the sum of squares is computed in `f64`,
/// which cannot underflow squared `f32` values, so it is zero only if every
/// element is zero.
#[inline]
fn is_placeholder_vector(v: &[f32]) -> bool {
    v.iter().all(|&x| x == 0.0)
}

/// Cosine distance with proper NaN handling
///
/// Returns distance in [0, 2]: 0 = identical, 2 = opposite
///
/// Dot product and norms are accumulated in `f64` and the result is clamped
/// to `[-1, 1]` before conversion, so floating-point rounding can never push
/// the distance outside its documented range.
///
/// Every degenerate case is reported as an error — no numeric value is ever
/// returned for an input the formula cannot answer for.
///
/// # Errors
/// - `VectorDbError::DimensionMismatch` if `a.len() != b.len()`
/// - `VectorDbError::ZeroVector` if either vector has zero magnitude
///   (cosine similarity is undefined there — callers must not receive a
///   fabricated distance such as 0.0 or 1.0)
/// - `VectorDbError::NaNDistance` if either vector contains a non-finite
///   value (NaN/±infinity) or the computed distance is not finite
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(VectorDbError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if !a.iter().chain(b.iter()).all(|x| x.is_finite()) {
        return Err(VectorDbError::NaNDistance);
    }

    let dot_prod = dot(a, b);
    let norm_a = norm(a);
    let norm_b = norm(b);

    // Exact comparison is sound here: inputs are finite `f32`, squared in
    // `f64`, so the only way the norm is 0.0 is an all-zero (or all-±0.0)
    // vector — `f64` cannot underflow a squared `f32`.
    if norm_a == 0.0 || norm_b == 0.0 {
        return Err(VectorDbError::ZeroVector);
    }

    // Divide sequentially to avoid overflow in norm_a * norm_b.
    let similarity = (dot_prod / norm_a / norm_b).clamp(-1.0, 1.0);
    let dist = 1.0 - similarity;

    if !dist.is_finite() {
        return Err(VectorDbError::NaNDistance);
    }
    Ok(dist as f32)
}

/// Cosine distance from many candidate rows to one fixed vector.
///
/// In a graph search the query is compared against every visited row, but
/// the query itself never changes: this type computes the query's
/// finiteness check and norm **once** at construction instead of once per
/// candidate, while producing results bit-identical to
/// [`cosine_distance`] (same `f64` operations in the same order — IEEE
/// multiplication is commutative, so `dot(row, query)` equals
/// `dot(query, row)` exactly).
struct QueryCosine<'q> {
    query: &'q [f32],
    norm_query: f64,
}

impl<'q> QueryCosine<'q> {
    /// Validate the query and precompute its norm.
    ///
    /// # Errors
    /// - `VectorDbError::NaNDistance` if the query contains a non-finite value
    /// - `VectorDbError::ZeroVector` if the query has zero magnitude
    fn new(query: &'q [f32]) -> Result<Self> {
        if !query.iter().all(|x| x.is_finite()) {
            return Err(VectorDbError::NaNDistance);
        }
        let norm_query = norm(query);
        if norm_query == 0.0 {
            return Err(VectorDbError::ZeroVector);
        }
        Ok(Self { query, norm_query })
    }

    /// Distance from `other` to the fixed query vector, in [0, 2].
    ///
    /// Mirrors [`cosine_distance`]'s error contract exactly.
    #[inline]
    fn distance(&self, other: &[f32]) -> Result<f32> {
        if other.len() != self.query.len() {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.query.len(),
                got: other.len(),
            });
        }
        if !other.iter().all(|x| x.is_finite()) {
            return Err(VectorDbError::NaNDistance);
        }

        let norm_other = norm(other);
        if norm_other == 0.0 {
            return Err(VectorDbError::ZeroVector);
        }

        // Divide sequentially to avoid overflow (mirrors cosine_distance).
        let similarity = (dot(other, self.query) / norm_other / self.norm_query).clamp(-1.0, 1.0);
        let dist = 1.0 - similarity;
        if !dist.is_finite() {
            return Err(VectorDbError::NaNDistance);
        }
        Ok(dist as f32)
    }
}

/// Compare two distances, handling NaN safely
///
/// NaN is treated as equal to all distances (preserves sort stability)
#[inline]
fn distance_cmp(a: f32, b: f32) -> std::cmp::Ordering {
    if a.is_nan() || b.is_nan() {
        return std::cmp::Ordering::Equal;
    }
    a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
}

// ============================================================================
//  Pseudorandom Number Generator (Xorshift)
// ============================================================================

/// Fast pseudorandom number generator for reproducible level assignment
#[derive(Debug, Clone)]
struct Rng(u64);

impl Rng {
    /// Create new RNG with given seed (odd values guaranteed)
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    /// Generate next 64-bit unsigned integer
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Generate next float in [0, 1)
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

// ============================================================================
//  Metadata Store (append-only log)
// ============================================================================

pub const META_FLAG_DELETED: u32 = 1;
const META_MAGIC: u64 = 0x4D45_5441_0000_0002;
const META_HEADER: usize = 16; // magic(8) + count(4) + reserved(4)

/// Single metadata record descriptor (in-memory index, data stays on disk via mmap)
#[derive(Debug, Clone)]
struct MetaRecord {
    flags: u32,
    offset: u64,
    len: u32,
}

/// Metadata storage: mmap-backed append-only file with record count in header.
///
/// # File layout
/// ```text
/// [0..8)      magic: u64 LE
/// [8..12)     count: u32 LE  (flushed record count, updated on flush())
/// [12..16)    reserved: u32 LE
/// [16..)      records: [ id:u32 | flags:u32 | dlen:u32 | pad:u32 | data:dlen ] × count
/// ```
///
/// The in-memory HashMap only stores (id → offset, len, flags) — the actual
/// metadata bytes live in the mmap and are read on demand via `get()`.
/// The header count is only updated on `flush()`, so on crash recovery
/// `open()` reads exactly the flushed records and replays the WAL for the rest.
struct MetadataStore {
    file: File,
    mmap: Option<MmapMut>,
    index: HashMap<u32, MetaRecord>,
    record_count: u32,
    file_len: usize,
}

impl MetadataStore {
    async fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let exists = tokio::fs::try_exists(path)
            .await
            .map_err(VectorDbError::Io)?;

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .map_err(VectorDbError::Io)?;

        let mut store = Self {
            file,
            mmap: None,
            index: HashMap::new(),
            record_count: 0,
            file_len: 0,
        };

        if !exists {
            let total = META_HEADER;
            store
                .file
                .set_len(total as u64)
                .map_err(VectorDbError::Io)?;
            let mut m = map_file(&store.file)?;
            m[0..8].copy_from_slice(&META_MAGIC.to_le_bytes());
            m[8..12].copy_from_slice(&0u32.to_le_bytes()); // count = 0
            m.flush().map_err(VectorDbError::Io)?;
            store.mmap = Some(m);
            store.file_len = META_HEADER;
        } else {
            let len = store.file.metadata().map_err(VectorDbError::Io)?.len() as usize;
            if len < META_HEADER {
                return Err(VectorDbError::Corruption("metadata file too small".into()));
            }
            let m = map_file(&store.file)?;

            let magic = u64::from_le_bytes(
                m[0..8]
                    .try_into()
                    .map_err(|_| VectorDbError::Corruption("metadata file too small".into()))?,
            );
            if magic != META_MAGIC {
                return Err(VectorDbError::Corruption("invalid metadata magic".into()));
            }

            let count = u32::from_le_bytes(m[8..12].try_into().unwrap());
            store.record_count = count;

            let mut pos = META_HEADER;
            // The record scan reads the file front-to-back exactly once.
            #[cfg(unix)]
            let _ = m.advise(memmap2::Advice::Sequential);
            for _ in 0..count as usize {
                if pos + 16 > len {
                    return Err(VectorDbError::Corruption(format!(
                        "metadata truncated: expected {} records but only {} bytes available",
                        count,
                        len - META_HEADER
                    )));
                }
                let id = u32::from_le_bytes(m[pos..pos + 4].try_into().unwrap());
                let flags = u32::from_le_bytes(m[pos + 4..pos + 8].try_into().unwrap());
                let dlen = u32::from_le_bytes(m[pos + 8..pos + 12].try_into().unwrap()) as usize;

                let data_off = pos + 16;
                if data_off + dlen > len {
                    return Err(VectorDbError::Corruption(
                        "metadata record data extends beyond file".into(),
                    ));
                }

                store.index.insert(
                    id,
                    MetaRecord {
                        flags,
                        offset: data_off as u64,
                        len: dlen as u32,
                    },
                );
                pos = data_off + dlen;
            }
            // Back to random access: after open, records are read by id.
            #[cfg(unix)]
            let _ = m.advise(memmap2::Advice::Random);

            // Truncate any bytes beyond the flushed records (leftover from grow()
            // that were never committed via flush())
            if pos < len {
                drop(m);
                store.file.set_len(pos as u64).map_err(VectorDbError::Io)?;
                let m = map_file(&store.file)?;
                store.mmap = Some(m);
            } else {
                store.mmap = Some(m);
            }
            store.file_len = pos;
        }
        Ok(store)
    }

    fn mmap(&self) -> &MmapMut {
        self.mmap.as_ref().expect("metadata mmap present")
    }

    fn mmap_mut(&mut self) -> &mut MmapMut {
        self.mmap.as_mut().expect("metadata mmap present")
    }

    fn is_deleted(&self, id: u32) -> bool {
        self.index
            .get(&id)
            .map(|r| r.flags & META_FLAG_DELETED != 0)
            .unwrap_or(true)
    }

    fn get(&self, id: u32) -> Option<&[u8]> {
        let r = self.index.get(&id)?;
        if r.flags & META_FLAG_DELETED != 0 {
            return None;
        }
        let off = r.offset as usize;
        Some(&self.mmap()[off..off + r.len as usize])
    }

    fn put(&mut self, id: u32, flags: u32, data: &[u8]) -> Result<()> {
        let need = self.file_len + 16 + data.len();
        if need > self.mmap().len() {
            self.grow(need)?;
        }
        let off = self.file_len;
        {
            let m = self.mmap_mut();
            m[off..off + 4].copy_from_slice(&id.to_le_bytes());
            m[off + 4..off + 8].copy_from_slice(&flags.to_le_bytes());
            m[off + 8..off + 12].copy_from_slice(&(data.len() as u32).to_le_bytes());
            m[off + 12..off + 16].copy_from_slice(&0u32.to_le_bytes());
            m[off + 16..off + 16 + data.len()].copy_from_slice(data);
        }
        self.index.insert(
            id,
            MetaRecord {
                flags,
                offset: (off + 16) as u64,
                len: data.len() as u32,
            },
        );
        self.record_count += 1;
        self.file_len = off + 16 + data.len();
        Ok(())
    }

    fn grow(&mut self, need: usize) -> Result<()> {
        let new_size = (need * 2).max(1024);
        if let Some(m) = self.mmap.as_ref() {
            // Persist records written so far before re-mapping the file.
            m.flush_range(0, self.file_len).map_err(VectorDbError::Io)?;
        }
        self.mmap = None;
        self.file
            .set_len(new_size as u64)
            .map_err(VectorDbError::Io)?;
        self.mmap = Some(map_file(&self.file)?);
        Ok(())
    }

    /// Persist the count header and sync written records to disk.
    ///
    /// Only `[0, file_len)` is synced: everything beyond is untouched
    /// capacity from `grow()`. The file is intentionally **not** truncated
    /// here (as earlier versions did): keeping mapping size and file size in
    /// lockstep avoids a munmap/mmap round-trip per flush, and `open()` trims
    /// any slack left behind by a crash.
    fn flush(&mut self) -> Result<()> {
        let count = self.record_count;
        self.mmap_mut()[8..12].copy_from_slice(&count.to_le_bytes());
        self.mmap()
            .flush_range(0, self.file_len)
            .map_err(VectorDbError::Io)?;
        Ok(())
    }
}

// ============================================================================
//  HNSW Index
// ============================================================================

/// Format v3: variable-length node blocks sized by their level (see
/// [`HnswIndex::node_bytes`]) placed in an append-only arena, addressed
/// through an in-memory offset directory instead of `id * fixed_size`.
const HNSW_MAGIC: u64 = 0x484E_5357_0000_0003;
const HNSW_HEADER: usize = 128;
const HNSW_FLAG_DELETED: u8 = 1;
const NONE: u32 = u32::MAX;

/// Single layer metadata for a node
#[derive(Debug)]
struct HnswIndex {
    file: File,
    mmap: Option<MmapMut>,
    cfg: Config,
    rng: Rng,
    /// Offset directory: `dir[id]` is the byte offset of node `id`'s block.
    ///
    /// v3 stores each node in a variable-length block sized for *its* level
    /// (levels are sampled once at insertion and never change), so blocks are
    /// not at a fixed stride and need an index. The directory is rebuilt from
    /// the file at open time (blocks are self-describing) and only ever grows
    /// by appending, matching the append-only arena layout.
    dir: Vec<u64>,
}

thread_local! {
    /// Per-thread traversal scratch, reused across HNSW operations to avoid
    /// ~50 heap allocations per insert/search.
    ///
    /// Thread locality is a safety requirement, not just an allocation
    /// optimisation: concurrent searches share `&HnswIndex` through the
    /// database read lock, so candidate/visited/result heaps stored inside
    /// the index would be mutated by several threads at once (data race).
    /// Per-thread buffers make each traversal independent without taking a
    /// lock around the whole search.
    static SCRATCH: UnsafeCell<SearchBuffers> = UnsafeCell::new(SearchBuffers::new());
}

/// Pre-allocated scratch buffers reused across HNSW operations.
/// Eliminates ~50 heap allocations per insert/search by clearing
/// instead of dropping and recreating.
struct SearchBuffers {
    visited: HashSet<u32>,
    candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, u32)>>,
    results: BinaryHeap<(OrderedFloat<f32>, u32)>,
    search_output: Vec<(f32, u32)>,
    select_result: Vec<u32>,
    select_seen: HashSet<u32>,
    cross_cache: HashMap<(u32, u32), f32>,
    dists: Vec<(u32, f32)>,
}

impl SearchBuffers {
    /// Default capacity hints (matching the default `ef_construction` /
    /// `m0`). The buffers grow on demand if a configuration needs more.
    fn new() -> Self {
        Self {
            visited: HashSet::with_capacity(300),
            candidates: BinaryHeap::with_capacity(150),
            results: BinaryHeap::with_capacity(151),
            search_output: Vec::with_capacity(150),
            select_result: Vec::with_capacity(64),
            select_seen: HashSet::with_capacity(128),
            cross_cache: HashMap::new(),
            dists: Vec::with_capacity(150),
        }
    }
}

impl HnswIndex {
    /// Bytes for one node block at a given level.
    ///
    /// A block is sized for exactly the levels the node has (the level is
    /// sampled once at insertion and never changes), instead of reserving
    /// space for `cfg.max_level` layers on every node. Since ~1 - 1/m of all
    /// nodes live at layer 0, the average block shrinks from
    /// `node_bytes(cfg, max_level)` (760 B at defaults) to ~176 B.
    ///
    /// Block layout (all little-endian):
    /// `[level u8][flags u8][size u32][pad u16]` then, per layer 0..=level:
    /// `[count u32][count × neighbor id u32]`.
    fn node_bytes(cfg: &Config, level: usize) -> usize {
        8 + Self::layer_bytes(cfg, 0) + level * Self::layer_bytes(cfg, 1)
    }

    /// Calculate bytes for a single layer within a node
    fn layer_bytes(cfg: &Config, layer: usize) -> usize {
        if layer == 0 {
            4 + cfg.m0 * 4
        } else {
            4 + cfg.m * 4
        }
    }

    /// Capacity (max connections) for a specific layer
    fn layer_capacity(cfg: &Config, layer: usize) -> usize {
        if layer == 0 { cfg.m0 } else { cfg.m }
    }

    /// Open or create HNSW index
    async fn open<P: AsRef<Path>>(path: P, cfg: &Config) -> Result<Self> {
        let path = path.as_ref();

        // Check if file exists using tokio
        let exists = tokio::fs::try_exists(path)
            .await
            .map_err(VectorDbError::Io)?;

        // Open file using std for mmap compatibility
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .map_err(VectorDbError::Io)?;

        if !exists {
            // Initialize new index: fixed header plus an arena pre-sized for
            // a handful of level-0 blocks; the arena doubles as nodes arrive.
            let initial = cfg.initial_capacity.max(16);
            let size = HNSW_HEADER + initial * Self::node_bytes(cfg, 0);
            file.set_len(size as u64).map_err(VectorDbError::Io)?;
            let mut m = map_file(&file)?;
            m[0..8].copy_from_slice(&HNSW_MAGIC.to_le_bytes());
            m[8..16].copy_from_slice(&0u64.to_le_bytes()); // count
            m[16..20].copy_from_slice(&NONE.to_le_bytes()); // entry point
            m[20] = 0; // max level so far
            // Arena end: first byte after the last node block.
            m[24..32].copy_from_slice(&(HNSW_HEADER as u64).to_le_bytes());
            m[32..40].copy_from_slice(&(cfg.m as u64).to_le_bytes());
            m[40..48].copy_from_slice(&(cfg.m0 as u64).to_le_bytes());
            m[48..56].copy_from_slice(&(cfg.max_level as u64).to_le_bytes());
            m[56..64].copy_from_slice(&(cfg.ef_construction as u64).to_le_bytes());
            m[64..72].copy_from_slice(&cfg.seed.to_le_bytes());
            // Only the header page has been written on a fresh file.
            m.flush_range(0, HNSW_HEADER).map_err(VectorDbError::Io)?;
            Ok(Self {
                file,
                mmap: Some(m),
                cfg: *cfg,
                rng: Rng::new(cfg.seed),
                dir: Vec::with_capacity(initial),
            })
        } else {
            // Load existing index
            let mut m = map_file(&file)?;

            // Validate magic
            let magic = u64::from_le_bytes(m[0..8].try_into().map_err(|_| {
                VectorDbError::Corruption("file too small for HNSW header".to_string())
            })?);
            if magic != HNSW_MAGIC {
                return Err(VectorDbError::Corruption("invalid HNSW magic".to_string()));
            }
            if m.len() < HNSW_HEADER {
                return Err(VectorDbError::Corruption(
                    "HNSW file smaller than its header".to_string(),
                ));
            }

            // Validate config compatibility
            let on_disk = Config {
                dim: cfg.dim,
                m: u64::from_le_bytes(
                    m[32..40]
                        .try_into()
                        .map_err(|_| VectorDbError::Corruption("invalid m value".to_string()))?,
                ) as usize,
                m0: u64::from_le_bytes(
                    m[40..48]
                        .try_into()
                        .map_err(|_| VectorDbError::Corruption("invalid m0 value".to_string()))?,
                ) as usize,
                max_level: u64::from_le_bytes(
                    m[48..56]
                        .try_into()
                        .map_err(|_| VectorDbError::Corruption("invalid max_level".to_string()))?,
                ) as usize,
                ef_construction: u64::from_le_bytes(m[56..64].try_into().map_err(|_| {
                    VectorDbError::Corruption("invalid ef_construction".to_string())
                })?) as usize,
                seed: u64::from_le_bytes(
                    m[64..72]
                        .try_into()
                        .map_err(|_| VectorDbError::Corruption("invalid seed".to_string()))?,
                ),
                initial_capacity: cfg.initial_capacity,
                max_wal_segment_size: cfg.max_wal_segment_size,
                max_total_wal_size: cfg.max_total_wal_size,
                max_wal_segments: cfg.max_wal_segments,
            };

            if on_disk.m != cfg.m || on_disk.m0 != cfg.m0 || on_disk.max_level != cfg.max_level {
                return Err(VectorDbError::ConfigMismatch(format!(
                    "file has M={} M0={} max_level={}, requested M={} M0={} max_level={}",
                    on_disk.m, on_disk.m0, on_disk.max_level, cfg.m, cfg.m0, cfg.max_level,
                )));
            }

            let count = u64::from_le_bytes(
                m[8..16]
                    .try_into()
                    .map_err(|_| VectorDbError::Corruption("invalid count".to_string()))?,
            );
            let seed = on_disk.seed ^ (count.wrapping_mul(0x9E37_79B9_7F4A_7C15));

            // Rebuild the offset directory by walking the self-describing
            // node blocks. Every block is validated (level bounds, size
            // consistent with the level, fits in the file) as it is visited,
            // so a corrupt directory fails at open instead of panicking on
            // first graph access.
            let mut dir = Vec::with_capacity(count as usize);
            let mut off = HNSW_HEADER;
            for i in 0..count {
                if off + 8 > m.len() {
                    return Err(VectorDbError::Corruption(format!(
                        "node {i} block header truncated at offset {off}"
                    )));
                }
                let level = m[off];
                if level as usize > on_disk.max_level {
                    return Err(VectorDbError::Corruption(format!(
                        "node {i} has level {level} > configured maximum {}",
                        on_disk.max_level
                    )));
                }
                let size = u32::from_le_bytes(m[off + 2..off + 6].try_into().map_err(|_| {
                    VectorDbError::Corruption("node block header truncated".to_string())
                })?) as usize;
                if size != Self::node_bytes(&on_disk, level as usize) {
                    return Err(VectorDbError::Corruption(format!(
                        "node {i} block size {size} inconsistent with level {level}"
                    )));
                }
                if off + size > m.len() {
                    return Err(VectorDbError::Corruption(format!(
                        "node {i} block extends past end of file"
                    )));
                }
                dir.push(off as u64);
                off += size;
            }

            // A crash between a block write and the header update can leave
            // arena_end lagging behind (or ahead of) the last complete block;
            // the walk is the source of truth, so heal the header.
            let stored_arena_end = u64::from_le_bytes(
                m[24..32]
                    .try_into()
                    .map_err(|_| VectorDbError::Corruption("invalid arena end".to_string()))?,
            ) as usize;
            if stored_arena_end != off {
                tracing::debug!(
                    stored_arena_end,
                    walked_end = off,
                    "HNSW arena_end healed from node walk"
                );
                m[24..32].copy_from_slice(&(off as u64).to_le_bytes());
            }

            Ok(Self {
                file,
                mmap: Some(m),
                cfg: on_disk,
                rng: Rng::new(seed),
                dir,
            })
        }
    }

    fn mmap(&self) -> &MmapMut {
        self.mmap.as_ref().expect("hnsw mmap present")
    }

    fn mmap_mut(&mut self) -> &mut MmapMut {
        self.mmap.as_mut().expect("hnsw mmap present")
    }

    /// Raw pointer to *this thread's* traversal scratch buffers.
    ///
    /// Returning a raw pointer keeps the existing call structure (borrow the
    /// buffers for a scoped block, then release) without fighting the
    /// borrow checker over a thread-local. Safety: the buffer lives in
    /// thread-local storage, so no other thread can observe it; callers must
    /// still scope their borrows so a nested operation that also touches the
    /// scratch (e.g. `search_layer` called from `insert`) never runs while a
    /// previous borrow is live.
    #[inline]
    fn scratch(&self) -> *mut SearchBuffers {
        SCRATCH.with(|c| c.get())
    }

    fn count(&self) -> usize {
        u64::from_le_bytes(self.mmap()[8..16].try_into().unwrap()) as usize
    }

    /// First byte after the last node block, persisted in the header.
    ///
    /// Blocks are only ever appended (a node's block size never changes
    /// after creation), so this single value bounds every possibly-dirtied
    /// byte and doubles as the flush range end.
    fn arena_end(&self) -> usize {
        u64::from_le_bytes(self.mmap()[24..32].try_into().unwrap()) as usize
    }

    fn set_arena_end(&mut self, end: usize) {
        self.mmap_mut()[24..32].copy_from_slice(&(end as u64).to_le_bytes());
    }

    fn entry_point(&self) -> u32 {
        u32::from_le_bytes(self.mmap()[16..20].try_into().unwrap())
    }

    fn max_level(&self) -> u8 {
        self.mmap()[20]
    }

    fn set_count(&mut self, n: usize) {
        self.mmap_mut()[8..16].copy_from_slice(&(n as u64).to_le_bytes());
    }

    fn set_entry_point(&mut self, p: u32) {
        self.mmap_mut()[16..20].copy_from_slice(&p.to_le_bytes());
    }

    fn set_max_level(&mut self, l: u8) {
        self.mmap_mut()[20] = l;
    }

    fn node_offset(&self, id: u32) -> usize {
        // Panics on an unregistered id — the same failure mode the previous
        // fixed-stride arithmetic had for out-of-range nodes. Callers that
        // can observe a missing node (integrity checks, recovery) test the
        // directory first.
        self.dir[id as usize] as usize
    }

    /// Validate every structural invariant the graph layout relies on.
    ///
    /// Checks, for each node: the sampled level is within bounds, each used
    /// layer's neighbor count is within that layer's capacity, and every
    /// neighbor is an in-range id that is neither a self-link nor a duplicate.
    /// Also validates the header (count vs capacity, entry point range,
    /// max level).
    ///
    /// Layout-agnostic: it only asserts *semantics*, so it stays valid across
    /// on-disk layout refactors. O(N) over the whole graph — intended for
    /// tests, post-compaction verification, and manual diagnostics, not hot
    /// paths. The first violation is returned as `VectorDbError::Corruption`.
    fn integrity_check(&self) -> Result<()> {
        let count = self.count();
        if self.dir.len() < count {
            return Err(VectorDbError::Corruption(format!(
                "directory holds {} node blocks but count is {count}",
                self.dir.len()
            )));
        }

        // Directory / arena consistency: every registered block sits inside
        // the file, is ordered (no overlaps), and its stored size matches
        // the level it declares.
        let file_len = self.mmap().len();
        let mut walked_end = HNSW_HEADER;
        for id in 0..count {
            let off = self.dir[id] as usize;
            if off + 8 > file_len {
                return Err(VectorDbError::Corruption(format!(
                    "node {id} block header lies past end of file"
                )));
            }
            if off < walked_end {
                return Err(VectorDbError::Corruption(format!(
                    "node {id} block overlaps its predecessor"
                )));
            }
            let size = u32::from_le_bytes(
                self.mmap()[off + 2..off + 6]
                    .try_into()
                    .map_err(|_| VectorDbError::Corruption("block header truncated".into()))?,
            ) as usize;
            if off + size > file_len {
                return Err(VectorDbError::Corruption(format!(
                    "node {id} block extends past end of file"
                )));
            }
            let level = self.mmap()[off] as usize;
            if size != Self::node_bytes(&self.cfg, level) {
                return Err(VectorDbError::Corruption(format!(
                    "node {id} block size {size} inconsistent with level {level}"
                )));
            }
            walked_end = off + size;
        }
        let arena_end = self.arena_end();
        if arena_end < walked_end || arena_end > file_len {
            return Err(VectorDbError::Corruption(format!(
                "arena_end {arena_end} inconsistent with last block end {walked_end} (file {file_len})"
            )));
        }

        let ep = self.entry_point();
        if ep != NONE && (ep as usize) >= count {
            return Err(VectorDbError::Corruption(format!(
                "entry point {ep} is outside node range 0..{count}"
            )));
        }

        let max_level = self.max_level() as usize;
        if max_level > self.cfg.max_level {
            return Err(VectorDbError::Corruption(format!(
                "header max_level {max_level} exceeds configured maximum {}",
                self.cfg.max_level
            )));
        }

        for id in 0..count as u32 {
            let level = self.node_level(id) as usize;
            if level > self.cfg.max_level {
                return Err(VectorDbError::Corruption(format!(
                    "node {id} has level {level} > configured maximum {}",
                    self.cfg.max_level
                )));
            }

            for layer in 0..=level {
                let n = self.layer_count(id, layer);
                let layer_cap = Self::layer_capacity(&self.cfg, layer);
                if n > layer_cap {
                    return Err(VectorDbError::Corruption(format!(
                        "node {id} layer {layer} has {n} neighbors > capacity {layer_cap}"
                    )));
                }

                let mut seen: HashSet<u32> = HashSet::with_capacity(n);
                for i in 0..n {
                    let neighbor = self.layer_neighbor(id, layer, i);
                    if (neighbor as usize) >= count {
                        return Err(VectorDbError::Corruption(format!(
                            "node {id} layer {layer} links to {neighbor}, outside 0..{count}"
                        )));
                    }
                    if neighbor == id {
                        return Err(VectorDbError::Corruption(format!(
                            "node {id} layer {layer} links to itself"
                        )));
                    }
                    if !seen.insert(neighbor) {
                        return Err(VectorDbError::Corruption(format!(
                            "node {id} layer {layer} lists neighbor {neighbor} twice"
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    fn node_level(&self, id: u32) -> u8 {
        self.mmap()[self.node_offset(id)]
    }

    // Note: there is deliberately no `set_level` — a node's level (and with
    // it its block size) is fixed at creation in `write_node_block`.

    fn set_node_deleted(&mut self, id: u32) {
        let off = self.node_offset(id);
        self.mmap_mut()[off + 1] |= HNSW_FLAG_DELETED;
    }

    fn layer_region(&self, id: u32, layer: usize) -> usize {
        let base = self.node_offset(id) + 8;
        if layer == 0 {
            base
        } else {
            base + Self::layer_bytes(&self.cfg, 0) + (layer - 1) * Self::layer_bytes(&self.cfg, 1)
        }
    }

    fn layer_count(&self, id: u32, layer: usize) -> usize {
        let off = self.layer_region(id, layer);
        u32::from_le_bytes(self.mmap()[off..off + 4].try_into().unwrap()) as usize
    }

    fn set_layer_count(&mut self, id: u32, layer: usize, n: usize) {
        let off = self.layer_region(id, layer);
        self.mmap_mut()[off..off + 4].copy_from_slice(&(n as u32).to_le_bytes());
    }

    fn layer_neighbor(&self, id: u32, layer: usize, i: usize) -> u32 {
        let off = self.layer_region(id, layer) + 4 + i * 4;
        u32::from_le_bytes(self.mmap()[off..off + 4].try_into().unwrap())
    }

    fn set_layer_neighbor(&mut self, id: u32, layer: usize, i: usize, n: u32) {
        let off = self.layer_region(id, layer) + 4 + i * 4;
        self.mmap_mut()[off..off + 4].copy_from_slice(&n.to_le_bytes());
    }

    #[inline]
    fn layer_neighbors(&self, id: u32, layer: usize) -> Vec<u32> {
        let c = self.layer_count(id, layer);
        (0..c).map(|i| self.layer_neighbor(id, layer, i)).collect()
    }

    /// P6: Iterator version to avoid Vec allocation when just iterating
    #[inline]
    fn layer_neighbors_iter(&self, id: u32, layer: usize) -> impl Iterator<Item = u32> + '_ {
        let c = self.layer_count(id, layer);
        (0..c).map(move |i| self.layer_neighbor(id, layer, i))
    }

    /// Extend the file (by doubling) until the arena covers `end` bytes.
    fn ensure_arena(&mut self, end: usize) -> Result<()> {
        if end <= self.mmap().len() {
            return Ok(());
        }
        let mut new_len = self.mmap().len().max(HNSW_HEADER + 1024);
        while new_len < end {
            new_len = new_len
                .checked_mul(2)
                .ok_or_else(|| VectorDbError::Corruption("HNSW file size overflow".to_string()))?;
        }
        self.mmap = None;
        self.file
            .set_len(new_len as u64)
            .map_err(VectorDbError::Io)?;
        self.mmap = Some(map_file(&self.file)?);
        Ok(())
    }

    /// Append an empty block for a node at `level` and register it as the
    /// next directory entry.
    ///
    /// The block header carries `level`, flags (zero: freshly created) and
    /// the block size; per-layer counts are zeroed because the arena may
    /// hold leftover bytes from a crashed (unregistered) insertion.
    fn write_node_block(&mut self, level: u8) -> Result<()> {
        let size = Self::node_bytes(&self.cfg, level as usize);
        let off = self.arena_end();
        self.ensure_arena(off + size)?;
        {
            let m = self.mmap_mut();
            m[off] = level;
            m[off + 1] = 0; // flags
            m[off + 2..off + 6].copy_from_slice(&(size as u32).to_le_bytes());
            m[off + 6..off + 8].copy_from_slice(&0u16.to_le_bytes());
        }
        self.dir.push(off as u64);
        let id = (self.dir.len() - 1) as u32;
        for l in 0..=level as usize {
            self.set_layer_count(id, l, 0);
        }
        self.set_arena_end(off + size);
        Ok(())
    }

    /// Register node `id`, allocating its block (and any missing blocks for
    /// ids between the current count and `id`, which a crash can leave as
    /// gaps). Returns the node's level: freshly sampled for new blocks, or
    /// the previously sampled one when `id` is already registered.
    ///
    /// Does **not** bump `count` for `id` itself — insertion completes the
    /// node by calling [`Self::set_count`] once it is fully linked, so a
    /// crash mid-insert never leaves a counted but unlinkable node.
    fn alloc_node(&mut self, id: u32) -> Result<u8> {
        // A previous insertion may have aborted after allocating its block
        // but before counting it. Drop such orphans (reclaiming their arena
        // space) so directory and count stay in lockstep before this
        // allocation appends anything.
        while self.dir.len() > self.count() {
            let orphan = self.dir.pop().expect("directory non-empty above count");
            self.set_arena_end(orphan as usize);
        }
        // Crash gaps: rows exist for these ids but their node block was
        // never written. Register them as complete (but unlinked) nodes.
        while (self.count() as u32) < id {
            let level = self.sample_level() as u8;
            self.write_node_block(level)?;
            self.set_count(self.count() + 1);
        }
        if (self.count() as u32) > id {
            // Already registered: the block (and its level) exists.
            return Ok(self.node_level(id));
        }
        let level = self.sample_level() as u8;
        self.write_node_block(level)?;
        Ok(level)
    }

    fn sample_level(&mut self) -> usize {
        let ml = 1.0 / (self.cfg.m as f64).ln();
        let r = self.rng.next_f64().max(1e-20);
        ((-r.ln() * ml) as usize).min(self.cfg.max_level)
    }

    /// Select M best neighbors from precomputed distances using heuristic.
    ///
    /// `dists` must be pre-sorted by distance to the reference node (ascending).
    /// Each entry is `(neighbor_id, distance_to_ref)`.
    /// When `scratch` is provided, reuses pre-allocated buffers.
    fn select_neighbors_heuristic(
        dists: &[(u32, f32)],
        m: usize,
        cross_dist: impl Fn(u32, u32) -> Result<f32>,
        scratch: Option<&mut SearchBuffers>,
    ) -> Result<Vec<u32>> {
        if dists.len() <= m {
            return Ok(dists.iter().map(|(n, _)| *n).collect());
        }

        if let Some(scratch) = scratch {
            scratch.select_result.clear();
            scratch.select_seen.clear();
            scratch.cross_cache.clear();

            for &(neighbor_id, d_ref_n) in dists {
                if scratch.select_result.len() >= m {
                    break;
                }
                let mut keep = true;
                for &r in &scratch.select_result {
                    let d = if let Some(&cached) = scratch.cross_cache.get(&(neighbor_id, r)) {
                        cached
                    } else {
                        let d = cross_dist(neighbor_id, r)?;
                        scratch.cross_cache.insert((neighbor_id, r), d);
                        d
                    };
                    if d < d_ref_n {
                        keep = false;
                        break;
                    }
                }
                if keep && scratch.select_seen.insert(neighbor_id) {
                    scratch.select_result.push(neighbor_id);
                }
            }

            if scratch.select_result.len() < m {
                for &(neighbor_id, _) in dists {
                    if scratch.select_result.len() >= m {
                        break;
                    }
                    if scratch.select_seen.insert(neighbor_id) {
                        scratch.select_result.push(neighbor_id);
                    }
                }
            }

            Ok(std::mem::take(&mut scratch.select_result))
        } else {
            let mut result: Vec<u32> = Vec::with_capacity(m);
            let mut seen: HashSet<u32> = HashSet::new();
            let mut cross_cache: HashMap<(u32, u32), f32> = HashMap::new();

            for &(neighbor_id, d_ref_n) in dists {
                if result.len() >= m {
                    break;
                }
                let mut keep = true;
                for &r in &result {
                    let d = if let Some(&cached) = cross_cache.get(&(neighbor_id, r)) {
                        cached
                    } else {
                        let d = cross_dist(neighbor_id, r)?;
                        cross_cache.insert((neighbor_id, r), d);
                        d
                    };
                    if d < d_ref_n {
                        keep = false;
                        break;
                    }
                }
                if keep && seen.insert(neighbor_id) {
                    result.push(neighbor_id);
                }
            }

            if result.len() < m {
                for &(neighbor_id, _) in dists {
                    if result.len() >= m {
                        break;
                    }
                    if seen.insert(neighbor_id) {
                        result.push(neighbor_id);
                    }
                }
            }

            Ok(result)
        }
    }

    /// Add bidirectional link between two nodes at given layer
    fn add_link<F: Fn(u32, u32) -> Result<f32>>(
        &mut self,
        id: u32,
        layer: usize,
        new: u32,
        dist: &F,
    ) -> Result<()> {
        if new == id {
            // A node can surface as its own candidate while its edges are
            // being rebuilt (it is still reachable through other nodes'
            // lists). A self-link would waste a neighbor slot and break the
            // no-self-loops invariant of the graph.
            return Ok(());
        }
        let cap = Self::layer_capacity(&self.cfg, layer);
        let mut neighbors = self.layer_neighbors(id, layer);
        if !neighbors.contains(&new) {
            neighbors.push(new);
        }

        let mut dists: Vec<(u32, f32)> = neighbors
            .iter()
            .filter_map(|&n| match dist(id, n) {
                Ok(d) => Some(Ok((n, d))),
                // Placeholder row in a legacy neighbor list: drop the link
                // instead of scoring it. Other errors propagate.
                Err(VectorDbError::ZeroVector) => None,
                Err(e) => Some(Err(e)),
            })
            .collect::<Result<Vec<_>>>()?;
        dists.sort_by(|a, b| distance_cmp(a.1, b.1));

        let selected = Self::select_neighbors_heuristic(&dists, cap, dist, None)?;

        self.set_layer_count(id, layer, selected.len());
        for (i, n) in selected.iter().enumerate() {
            self.set_layer_neighbor(id, layer, i, *n);
        }
        Ok(())
    }

    /// Search layer for candidates similar to entry node.
    /// Results are written into scratch.search_output (sorted by distance).
    /// Caller must read from scratch before the next search_layer call.
    ///
    /// Error management: `ZeroVector` means the row is a placeholder (or the
    /// entry point is one) — the node is traversable but never reported as a
    /// result, and never contributes a fabricated distance. Any other error
    /// (out-of-bounds id, non-finite vector, ...) is propagated: a search
    /// never invents numbers to cover a failure.
    fn search_layer<F, D>(
        &self,
        entry: u32,
        layer: usize,
        ef: usize,
        dist: &F,
        is_deleted: &D,
    ) -> Result<()>
    where
        F: Fn(u32) -> Result<f32>,
        D: Fn(u32) -> bool,
    {
        let scratch = unsafe { &mut *self.scratch() };
        scratch.visited.clear();
        scratch.visited.insert(entry);

        // A placeholder entry point cannot be ranked, but its links (legacy
        // databases) let the walk hop off it onto real vectors.
        let (d0, rankable) = match dist(entry) {
            Ok(d) => (d, true),
            Err(VectorDbError::ZeroVector) => (f32::INFINITY, false),
            Err(e) => return Err(e),
        };

        scratch.candidates.clear();
        scratch.candidates.push(Reverse((OrderedFloat(d0), entry)));

        scratch.results.clear();
        if rankable && !is_deleted(entry) {
            scratch.results.push((OrderedFloat(d0), entry));
        }

        while let Some(Reverse((OrderedFloat(cd), c))) = scratch.candidates.pop() {
            let worst_result = scratch
                .results
                .peek()
                .map(|(d, _)| d.into_inner())
                .unwrap_or(f32::NEG_INFINITY);

            if scratch.results.len() >= ef && cd > worst_result {
                break;
            }

            for n in self.layer_neighbors_iter(c, layer) {
                if !scratch.visited.insert(n) {
                    continue;
                }
                let d = match dist(n) {
                    Ok(d) => d,
                    // Placeholder row: skip it entirely instead of scoring it.
                    Err(VectorDbError::ZeroVector) => continue,
                    Err(e) => return Err(e),
                };
                let worst = scratch
                    .results
                    .peek()
                    .map(|(d, _)| d.into_inner())
                    .unwrap_or(f32::NEG_INFINITY);

                if scratch.results.len() < ef || d < worst {
                    scratch.candidates.push(Reverse((OrderedFloat(d), n)));
                    if !is_deleted(n) {
                        scratch.results.push((OrderedFloat(d), n));
                        if scratch.results.len() > ef {
                            scratch.results.pop();
                        }
                    }
                }
            }
        }

        scratch.search_output.clear();
        scratch
            .search_output
            .extend(scratch.results.drain().map(|(d, id)| (d.into_inner(), id)));
        scratch.search_output.sort_by(|a, b| distance_cmp(a.0, b.0));
        Ok(())
    }

    /// Register a node's slot without linking it into the graph.
    ///
    /// Used for placeholder rows (all-zero vectors): they need level/count
    /// bookkeeping so later ids address valid slots and deletions are safe,
    /// but they are never distance-compared, never become the entry point,
    /// and never receive or hold graph edges.
    fn register_unlinked(&mut self, new_id: u32) -> Result<()> {
        if (new_id as usize) < self.count() {
            // Already registered (e.g. recovery replaying into a database
            // where the crash already completed this node).
            return Ok(());
        }
        self.alloc_node(new_id)?;
        self.set_count((new_id as usize + 1).max(self.count()));
        Ok(())
    }

    /// Insert new node into index (must be called with proper distance function)
    fn insert<F, D>(&mut self, new_id: u32, dist: F, is_deleted: &D) -> Result<()>
    where
        F: Fn(u32, u32) -> Result<f32> + Copy,
        D: Fn(u32) -> bool,
    {
        // Allocate the block for a new node (or find the existing one for a
        // re-insert — edge rebuilds keep the block, whose size was fixed at
        // creation together with the node's level).
        let level = self.alloc_node(new_id)? as usize;
        // (Re-)zero the per-layer counts: a fresh block may overlap crash
        // leftovers, and a rebuild starts from cleared edges.
        for l in 0..=level {
            self.set_layer_count(new_id, l, 0);
        }

        let entry = self.entry_point();
        if entry == NONE {
            self.set_entry_point(new_id);
            self.set_max_level(level as u8);
            self.set_count((new_id as usize + 1).max(self.count()));
            return Ok(());
        }

        let mut ep = entry;
        // The entry point may be a legacy placeholder row: use it only to
        // traverse (the greedy walk hops off it onto a real vector on the
        // first finite distance), never as a ranked distance.
        let mut ep_dist = match dist(new_id, ep) {
            Ok(d) => d,
            Err(VectorDbError::ZeroVector) => f32::INFINITY,
            Err(e) => return Err(e),
        };
        let cur_max = self.max_level() as usize;

        // 1. Greedy descent from top layer down to `level + 1`
        let mut l = cur_max;
        while l > level {
            let mut changed = true;
            while changed {
                changed = false;
                // P6: Use iterator to avoid Vec allocation
                for n in self.layer_neighbors_iter(ep, l) {
                    if is_deleted(n) {
                        continue;
                    }
                    let d = match dist(new_id, n) {
                        Ok(d) => d,
                        Err(VectorDbError::ZeroVector) => continue, // placeholder neighbor
                        Err(e) => return Err(e),
                    };
                    if d < ep_dist {
                        ep_dist = d;
                        ep = n;
                        changed = true;
                    }
                }
            }
            l = l.saturating_sub(1);
        }

        // 2. Link at every layer from min(level, cur_max) down to 0
        let start_layer = level.min(cur_max);
        for layer in (0..=start_layer).rev() {
            let to_new = |n: u32| dist(new_id, n);
            self.search_layer(ep, layer, self.cfg.ef_construction, &to_new, is_deleted)?;

            let cap = Self::layer_capacity(&self.cfg, layer);
            let selected = {
                let scratch = unsafe { &mut *self.scratch() };
                scratch.dists.clear();
                scratch
                    .dists
                    .extend(scratch.search_output.iter().map(|&(d, id)| (id, d)));
                scratch.dists.sort_by(|a, b| distance_cmp(a.1, b.1));
                let (dists_ptr, scratch_ptr) = {
                    let d = &scratch.dists as *const Vec<(u32, f32)>;
                    let s = scratch as *mut SearchBuffers;
                    (d, s)
                };
                Self::select_neighbors_heuristic(
                    unsafe { &*dists_ptr },
                    cap,
                    dist,
                    Some(unsafe { &mut *scratch_ptr }),
                )?
            };

            for &n in &selected {
                self.add_link(new_id, layer, n, &dist)?;
                self.add_link(n, layer, new_id, &dist)?;
            }

            let scratch = unsafe { &*self.scratch() };
            if let Some(&(_, e)) = scratch.search_output.first() {
                ep = e;
            }
        }

        if level > cur_max {
            self.set_entry_point(new_id);
            self.set_max_level(level as u8);
        }
        self.set_count((new_id as usize + 1).max(self.count()));
        Ok(())
    }

    fn mark_deleted(&mut self, id: u32) {
        if (id as usize) < self.count() {
            self.set_node_deleted(id);
        }
    }

    /// Search for k nearest neighbors
    fn search<F, D>(&self, k: usize, ef: usize, dist: F, is_deleted: &D) -> Result<Vec<(u32, f32)>>
    where
        F: Fn(u32) -> Result<f32> + Copy,
        D: Fn(u32) -> bool,
    {
        let entry = self.entry_point();
        if entry == NONE {
            return Ok(Vec::new());
        }

        let mut ep = entry;
        // `dist(ep)` may be `ZeroVector` if the entry point is a legacy
        // placeholder row — traverse through it, never rank it. Other errors
        // (bad query, corruption) must surface, not be swallowed.
        let mut ep_dist = match dist(ep) {
            Ok(d) => d,
            Err(VectorDbError::ZeroVector) => f32::INFINITY,
            Err(e) => return Err(e),
        };
        let cur_max = self.max_level() as usize;

        for layer in (1..=cur_max).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                // P6: Use iterator to avoid Vec allocation
                for n in self.layer_neighbors_iter(ep, layer) {
                    let d = match dist(n) {
                        Ok(d) => d,
                        Err(VectorDbError::ZeroVector) => continue, // placeholder neighbor
                        Err(e) => return Err(e),
                    };
                    if d < ep_dist && !is_deleted(n) {
                        ep_dist = d;
                        ep = n;
                        changed = true;
                    }
                }
            }
        }

        self.search_layer(ep, 0, ef.max(k), &dist, is_deleted)?;
        let scratch = unsafe { &*self.scratch() };
        scratch
            .search_output
            .iter()
            .take(k)
            .map(|&(d, id)| Ok((id, d)))
            .collect()
    }

    /// Sync written graph bytes: the fixed header plus the whole node arena.
    ///
    /// Blocks are appended-only and `arena_end` is bumped as they are
    /// written, so `[0, arena_end)` covers every possibly-dirtied byte and
    /// skips only file slack beyond the arena.
    fn flush(&self) -> Result<()> {
        let m = self.mmap();
        let end = self.arena_end().min(m.len());
        m.flush_range(0, end).map_err(VectorDbError::Io)
    }
}

// ============================================================================
//  Vector Storage
// ============================================================================

/// Format v3: the row area starts at a page-aligned offset (see
/// [`stored_data_start`]) instead of directly after the fixed header.
const VEC_MAGIC: u64 = 0x5645_4354_4F52_0003;
/// Fixed header layout: magic(8) | dim(8) | len(8) | capacity(8) | data_start(8).
const VEC_HEADER: usize = 40;
/// Offset of the persisted row-area start within the fixed header.
const VEC_DATA_START_OFF: usize = 32;

/// Read the persisted row-area start from a vector-file header.
#[inline]
fn stored_data_start(header: &[u8]) -> usize {
    u64::from_le_bytes(header[VEC_DATA_START_OFF..VEC_HEADER].try_into().unwrap()) as usize
}

/// View into vector data within mmap with bounds checking
struct VectorView {
    ptr: *const u8,
    dim: usize,
    file_size: usize,
    /// Row-area start (page-aligned), read from the file header.
    base: usize,
}

impl VectorView {
    /// Get vector at given ID with bounds checking
    ///
    /// # Errors
    /// Returns None if ID is out of bounds
    fn get(&self, id: u32) -> Option<&[f32]> {
        let start = self.base + (id as usize).checked_mul(self.dim)?.checked_mul(4)?;
        let end = start.checked_add(self.dim.checked_mul(4)?)?;
        if end > self.file_size {
            return None;
        }
        unsafe {
            Some(std::slice::from_raw_parts(
                self.ptr.add(start) as *const f32,
                self.dim,
            ))
        }
    }

    /// Cosine distance between two vectors with error handling
    fn distance(&self, a: u32, b: u32) -> Result<f32> {
        let va = self.get(a).ok_or(VectorDbError::IdOutOfBounds {
            id: a,
            capacity: (self.file_size as u32),
        })?;
        let vb = self.get(b).ok_or(VectorDbError::IdOutOfBounds {
            id: b,
            capacity: (self.file_size as u32),
        })?;
        cosine_distance(va, vb)
    }
}

/// Main thread-safe vector database with async support
pub struct VectorDb {
    path: PathBuf,
    cfg: Config,
    file: File,
    mmap: Option<MmapMut>,
    index: HnswIndex,
    meta: MetadataStore,
    deleted_count: usize,
    wal: WriteAheadLog,
    /// Advisory file lock to prevent concurrent access from multiple processes
    _lock_file: File,
}

impl VectorDb {
    /// Open or create vector database at path
    pub async fn open<P: AsRef<Path>>(path: P, cfg: Config) -> Result<Self> {
        assert!(cfg.dim > 0, "dim must be > 0");
        let path = path.as_ref().to_path_buf();

        // Check if file exists using tokio async
        let exists = tokio::fs::try_exists(&path)
            .await
            .map_err(VectorDbError::Io)?;

        // Open file using std for mmap compatibility
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(VectorDbError::Io)?;

        let mmap = if !exists {
            // Row area starts on the next page boundary after the fixed
            // header, so rows that fit in a page never straddle one.
            let data_start = align_up(VEC_HEADER, page_size());
            let total = data_start + cfg.initial_capacity * cfg.dim * 4;
            file.set_len(total as u64).map_err(VectorDbError::Io)?;
            let mut m = map_file(&file)?;
            m[0..8].copy_from_slice(&VEC_MAGIC.to_le_bytes());
            m[8..16].copy_from_slice(&(cfg.dim as u64).to_le_bytes());
            m[16..24].copy_from_slice(&0u64.to_le_bytes());
            m[24..32].copy_from_slice(&(cfg.initial_capacity as u64).to_le_bytes());
            m[32..40].copy_from_slice(&(data_start as u64).to_le_bytes());
            // Only the header has been written on a fresh file.
            m.flush_range(0, VEC_HEADER).map_err(VectorDbError::Io)?;
            m
        } else {
            let m = map_file(&file)?;
            let magic = u64::from_le_bytes(
                m[0..8]
                    .try_into()
                    .map_err(|_| VectorDbError::Corruption("vec file too small".to_string()))?,
            );
            if magic != VEC_MAGIC {
                return Err(VectorDbError::Corruption(
                    "invalid vector magic".to_string(),
                ));
            }
            let stored = u64::from_le_bytes(
                m[8..16]
                    .try_into()
                    .map_err(|_| VectorDbError::Corruption("invalid dim".to_string()))?,
            ) as usize;
            if stored != cfg.dim {
                return Err(VectorDbError::DimensionMismatch {
                    expected: cfg.dim,
                    got: stored,
                });
            }
            // Validate the persisted layout: the row area must be sane and
            // all `len` rows must fit inside the file, otherwise a corrupted
            // header would make row access panic instead of erroring.
            let data_start = stored_data_start(&m);
            if !(VEC_HEADER..=(1 << 20)).contains(&data_start) {
                return Err(VectorDbError::Corruption(format!(
                    "invalid row-area start {data_start}"
                )));
            }
            let len = u64::from_le_bytes(
                m[16..24]
                    .try_into()
                    .map_err(|_| VectorDbError::Corruption("invalid len".to_string()))?,
            ) as usize;
            let need = len
                .checked_mul(stored)
                .and_then(|b| b.checked_mul(4))
                .and_then(|b| b.checked_add(data_start))
                .ok_or_else(|| VectorDbError::Corruption("vector file size overflow".into()))?;
            if need > m.len() {
                return Err(VectorDbError::Corruption(format!(
                    "vector mapping holds {} bytes but {len} rows need {need}",
                    m.len()
                )));
            }
            m
        };

        let index = HnswIndex::open(hnsw_path(&path), &cfg).await?;
        let meta = MetadataStore::open(meta_path(&path)).await?;
        let wal = WriteAheadLog::new(
            &path,
            cfg.max_wal_segment_size,
            cfg.max_total_wal_size,
            cfg.max_wal_segments,
        )
        .await
        .map_err(VectorDbError::Io)?;

        // Prod 1 fix: Acquire exclusive advisory lock to prevent multi-process corruption
        let lock_path = path.with_extension("lock");
        let lock_file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(VectorDbError::Io)?;
        lock_file.try_lock_exclusive().map_err(|_e| {
            VectorDbError::Io(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!(
                    "Cannot open database: another process has an exclusive lock on {:?}",
                    lock_path
                ),
            ))
        })?;

        let mut db = Self {
            path: path.clone(),
            cfg,
            file,
            mmap: Some(mmap),
            index,
            meta,
            deleted_count: 0,
            wal,
            _lock_file: lock_file,
        };

        for id in 0..db.len() as u32 {
            if db.meta.is_deleted(id) {
                db.deleted_count += 1;
            }
        }

        // Recover from WAL if needed
        db.recover_from_wal(&path).await?;

        Ok(db)
    }

    /// Recover from Write-Ahead Log after crash
    ///
    /// Recovery is idempotent: records that were already applied to data files
    /// before the crash are safely skipped. The flush() call at the end writes
    /// a new checkpoint, so the WAL is truncated after successful recovery.
    async fn recover_from_wal(&mut self, db_path: &Path) -> Result<()> {
        let records = WriteAheadLog::load_for_recovery(db_path)
            .await
            .map_err(VectorDbError::Io)?;

        if records.is_empty() {
            return Ok(());
        }

        tracing::info!("WAL recovery: replaying {} records", records.len());

        let mut recovered_count = 0;
        for record in records {
            match record.op_type {
                WalOpType::Insert => {
                    if (record.vector_id as usize) >= self.len() {
                        // Vector not yet in file, full insert
                        self.insert_from_wal(&record)?;
                        recovered_count += 1;
                    } else if self.meta.get(record.vector_id).is_none() {
                        // Vector data exists but metadata missing (data flushed
                        // before crash, metadata not yet flushed).
                        // Restore metadata from WAL record.
                        self.meta.put(record.vector_id, 0, &record.metadata)?;
                        // The row was counted as *deleted* during open, because
                        // missing metadata reads as deleted (see
                        // `MetadataStore::is_deleted`). With its metadata back,
                        // the row is live again — keep `deleted_count` in sync,
                        // otherwise `live_len()`/compaction triggers drift.
                        // (If a Delete record follows, it re-increments below.)
                        self.deleted_count = self.deleted_count.saturating_sub(1);
                        recovered_count += 1;
                    }
                }
                WalOpType::Delete => {
                    // Skip if already deleted or out of bounds (idempotent)
                    if (record.vector_id as usize) < self.len()
                        && !self.meta.is_deleted(record.vector_id)
                    {
                        let prev = self
                            .meta
                            .get(record.vector_id)
                            .map(|b| b.to_vec())
                            .unwrap_or_default();
                        self.meta.put(record.vector_id, META_FLAG_DELETED, &prev)?;
                        self.index.mark_deleted(record.vector_id);
                        self.deleted_count += 1;
                        recovered_count += 1;
                    }
                }
                WalOpType::Update => {
                    // Skip if deleted or out of bounds (idempotent)
                    if (record.vector_id as usize) < self.len()
                        && !self.meta.is_deleted(record.vector_id)
                    {
                        self.row_mut(record.vector_id as usize)
                            .copy_from_slice(&record.vector_data);
                        self.meta.put(record.vector_id, 0, &record.metadata)?;

                        // Rebuild HNSW edges for this node with the new vector data
                        self.rebuild_hnsw_node(record.vector_id)?;

                        recovered_count += 1;
                    }
                }
                WalOpType::Checkpoint => {
                    // Checkpoint marker: all prior ops are safe on disk
                }
            }
        }

        tracing::info!("WAL recovery: recovered {} records", recovered_count);

        // Always checkpoint + clear once a non-empty WAL has been replayed (we
        // only reach here when the WAL contained records, thanks to the early
        // return above). Even if every record was idempotently skipped, leaving
        // them in the file would let their stale LSNs collide with future writes:
        // `header.current_lsn` is only refreshed at checkpoint/truncate time and
        // can lag the records actually present in the file.
        self.flush().await?;

        Ok(())
    }

    /// Rebuild HNSW edges for a single node after vector data changed
    fn rebuild_hnsw_node(&mut self, id: u32) -> Result<()> {
        let view = self.vector_view();
        let dist = |a: u32, b: u32| view.distance(a, b);
        let is_deleted = |id: u32| self.meta.is_deleted(id);

        // The row exists (id < len), but its node block may be missing if a
        // crash landed between the row write and graph registration.
        if (id as usize) >= self.index.count() {
            self.index.register_unlinked(id)?;
        }

        // Re-insert the node to rebuild its edges at all layers
        // First clear existing edges
        let node_level = self.index.node_level(id);
        for layer in 0..=node_level as usize {
            self.index.set_layer_count(id, layer, 0);
        }

        let entry = self.index.entry_point();
        let placeholder = is_placeholder_vector(self.row(id as usize));

        if placeholder {
            // The row is (now) a placeholder: keep it out of the graph, and
            // make sure it does not stay the entry point.
            if entry == id {
                match self.find_indexable_row(id) {
                    Some(other) => {
                        self.index.set_entry_point(other);
                        let level = self.index.node_level(other);
                        self.index.set_max_level(level);
                    }
                    None => {
                        self.index.set_entry_point(NONE);
                        self.index.set_max_level(0);
                    }
                }
            }
            return Ok(());
        }

        if entry == NONE {
            // Empty graph: registering as entry point needs no distances.
            return self.index.insert(id, dist, &is_deleted);
        }

        // Find a seed entry point for re-linking
        let mut ep;
        let mut ep_dist;
        if entry == id {
            // This node is the entry point and its links were just cleared:
            // re-link against another indexable row instead of itself.
            match self.find_indexable_row(id) {
                Some(other) => {
                    ep = other;
                    ep_dist = dist(id, other)?;
                }
                // This is the only row in the graph: nothing to link to.
                None => return Ok(()),
            }
        } else {
            ep = entry;
            // The entry point may be a legacy placeholder row: it is used
            // only to traverse, never ranked.
            ep_dist = match dist(id, ep) {
                Ok(d) => d,
                Err(VectorDbError::ZeroVector) => f32::INFINITY,
                Err(e) => return Err(e),
            };
        }

        // Greedy descent to find closest entry point
        let cur_max = self.index.max_level() as usize;
        for layer in (1..=cur_max).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                for n in self.index.layer_neighbors_iter(ep, layer) {
                    if is_deleted(n) || n == id {
                        continue;
                    }
                    let d = match dist(id, n) {
                        Ok(d) => d,
                        Err(VectorDbError::ZeroVector) => continue, // placeholder neighbor
                        Err(e) => return Err(e),
                    };
                    if d < ep_dist {
                        ep_dist = d;
                        ep = n;
                        changed = true;
                    }
                }
            }
        }

        // Re-link at each layer of this node
        for layer in (0..=node_level as usize).rev() {
            let to_new = |n: u32| dist(id, n);
            self.index
                .search_layer(ep, layer, self.cfg.ef_construction, &to_new, &is_deleted)?;
            let cap = HnswIndex::layer_capacity(&self.cfg, layer);
            {
                let scratch = unsafe { &mut *self.index.scratch() };
                scratch.dists.clear();
                scratch
                    .dists
                    .extend(scratch.search_output.iter().map(|&(d, id)| (id, d)));
                // The rebuilt node is still reachable through other nodes'
                // links, so it can rank itself at distance 0 here. Drop it:
                // it must not consume a neighbor slot of its own list.
                scratch.dists.retain(|&(n, _)| n != id);
                scratch.dists.sort_by(|a, b| distance_cmp(a.1, b.1));
                let (dists_ptr, scratch_ptr) = {
                    let d = &scratch.dists as *const Vec<(u32, f32)>;
                    let s = scratch as *mut SearchBuffers;
                    (d, s)
                };
                let selected = HnswIndex::select_neighbors_heuristic(
                    unsafe { &*dists_ptr },
                    cap,
                    dist,
                    Some(unsafe { &mut *scratch_ptr }),
                )?;
                for &n in &selected {
                    self.index.add_link(id, layer, n, &dist)?;
                    self.index.add_link(n, layer, id, &dist)?;
                }
            }
            let scratch = unsafe { &*self.index.scratch() };
            if let Some(&(_, e)) = scratch.search_output.first() {
                ep = e;
            }
        }

        Ok(())
    }

    /// Insert from WAL record (used during recovery)
    fn insert_from_wal(&mut self, record: &WalRecord) -> Result<u32> {
        self.insert_raw(&record.vector_data, &record.metadata)
    }

    /// Find a row that can be distance-compared (non-deleted, non-placeholder),
    /// excluding `exclude`. Used to (re)pick a graph entry point.
    fn find_indexable_row(&self, exclude: u32) -> Option<u32> {
        (0..self.len() as u32).find(|&id| {
            id != exclude
                && !self.meta.is_deleted(id)
                && !is_placeholder_vector(self.row(id as usize))
        })
    }

    /// Insert vector without WAL logging (used for compaction and recovery)
    fn insert_raw(&mut self, v: &[f32], metadata: &[u8]) -> Result<u32> {
        let n = self.len();
        let id = n as u32;

        if n == self.capacity() {
            self.grow()?;
        }
        self.row_mut(n).copy_from_slice(v);
        self.set_len_field(n + 1);
        self.meta.put(id, 0, metadata)?;

        let view = self.vector_view();
        let dist = |a: u32, b: u32| view.distance(a, b);
        let is_deleted = |_id: u32| false;
        if is_placeholder_vector(v) {
            // Placeholder row: stored, never indexed — no distance to it exists.
            self.index.register_unlinked(id)?;
        } else {
            self.index.insert(id, dist, &is_deleted)?;
        }
        Ok(id)
    }

    fn mmap(&self) -> &MmapMut {
        self.mmap.as_ref().expect("vec mmap present")
    }

    fn mmap_mut(&mut self) -> &mut MmapMut {
        self.mmap.as_mut().expect("vec mmap present")
    }

    fn vector_view(&self) -> VectorView {
        VectorView {
            ptr: self.mmap().as_ptr(),
            dim: self.cfg.dim,
            file_size: self.mmap().len(),
            base: stored_data_start(self.mmap()),
        }
    }

    /// Get vector dimensionality
    pub fn dim(&self) -> usize {
        self.cfg.dim
    }

    /// Get total number of vectors (including deleted)
    pub fn len(&self) -> usize {
        u64::from_le_bytes(self.mmap()[16..24].try_into().unwrap()) as usize
    }

    /// Check if database is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get number of live (non-deleted) vectors
    pub fn live_len(&self) -> usize {
        self.len() - self.deleted_count
    }

    /// Get number of deleted vectors
    pub fn deleted_count(&self) -> usize {
        self.deleted_count
    }

    /// Get deletion statistics
    ///
    /// Returns (deleted_count, total_count, deletion_ratio)
    /// where deletion_ratio = deleted_count / total_count (or 0.0 if empty)
    pub fn deletion_stats(&self) -> (usize, usize, f32) {
        let total = self.len();
        let deleted = self.deleted_count;
        let ratio = if total > 0 {
            deleted as f32 / total as f32
        } else {
            0.0
        };
        (deleted, total, ratio)
    }

    /// Check if compaction should be triggered (deletion ratio >= 30%)
    pub fn should_compact(&self) -> bool {
        let (_deleted, total, ratio) = self.deletion_stats();
        total > 0 && ratio > 0.30
    }

    /// Validate the structural integrity of all three on-disk structures.
    ///
    /// Verifies the vector file (row count vs capacity, row bytes fit in the
    /// mapping), metadata coverage (every row has a record, `deleted_count`
    /// matches the tombstoned rows), and the HNSW graph (per-node levels,
    /// neighbor counts/bounds, entry point — see `HnswIndex::integrity_check`).
    ///
    /// All checks are layout-agnostic invariants, so they remain meaningful
    /// across storage-layout refactors. O(N) — use after compaction, in
    /// tests, or for diagnostics; not on hot paths.
    pub fn integrity_check(&self) -> Result<()> {
        let len = self.len();
        let cap = self.capacity();
        if len > cap {
            return Err(VectorDbError::Corruption(format!(
                "vector count {len} exceeds capacity {cap}"
            )));
        }

        let row_bytes = self
            .cfg
            .dim
            .checked_mul(4)
            .ok_or_else(|| VectorDbError::Corruption("row size overflow".into()))?;
        let needed = len
            .checked_mul(row_bytes)
            .and_then(|b| b.checked_add(stored_data_start(self.mmap())))
            .ok_or_else(|| VectorDbError::Corruption("file size overflow".into()))?;
        if needed > self.mmap().len() {
            return Err(VectorDbError::Corruption(format!(
                "vector mapping holds {} bytes but {len} rows need {needed}",
                self.mmap().len()
            )));
        }

        if self.index.count() > len {
            return Err(VectorDbError::Corruption(format!(
                "graph has {} nodes but the vector file holds {len} rows",
                self.index.count()
            )));
        }

        // Every row (live or tombstoned) must carry a metadata record, and
        // `deleted_count` must agree with what the metadata store reports.
        let mut tombstoned = 0usize;
        for id in 0..len as u32 {
            if !self.meta.index.contains_key(&id) {
                return Err(VectorDbError::Corruption(format!(
                    "row {id} has no metadata record"
                )));
            }
            if self.meta.is_deleted(id) {
                tombstoned += 1;
            }
        }
        if tombstoned != self.deleted_count {
            return Err(VectorDbError::Corruption(format!(
                "deleted_count is {} but metadata reports {tombstoned} tombstoned rows",
                self.deleted_count
            )));
        }

        self.index.integrity_check()
    }

    fn capacity(&self) -> usize {
        u64::from_le_bytes(self.mmap()[24..32].try_into().unwrap()) as usize
    }

    fn set_len_field(&mut self, n: usize) {
        self.mmap_mut()[16..24].copy_from_slice(&(n as u64).to_le_bytes());
    }

    /// Byte offset of row `i` within the vector file mapping.
    ///
    /// The row area begins at the page-aligned `data_start` persisted in the
    /// file header (v3 format), so rows never straddle page boundaries when
    /// the row size divides the page size.
    #[inline]
    fn row_offset(&self, i: usize) -> usize {
        stored_data_start(self.mmap()) + i * self.cfg.dim * 4
    }

    fn row(&self, i: usize) -> &[f32] {
        let dim = self.cfg.dim;
        let start = self.row_offset(i);
        let bytes = &self.mmap()[start..start + dim * 4];
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, dim) }
    }

    fn row_mut(&mut self, i: usize) -> &mut [f32] {
        let dim = self.cfg.dim;
        let start = self.row_offset(i);
        let bytes = &mut self.mmap_mut()[start..start + dim * 4];
        unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, dim) }
    }

    fn grow(&mut self) -> Result<()> {
        let new_cap = (self.capacity() * 2).max(16);
        let new_len = stored_data_start(self.mmap()) + new_cap * self.cfg.dim * 4;
        self.mmap = None;
        self.file
            .set_len(new_len as u64)
            .map_err(VectorDbError::Io)?;
        let mut m = map_file(&self.file)?;
        m[24..32].copy_from_slice(&(new_cap as u64).to_le_bytes());
        self.mmap = Some(m);
        Ok(())
    }

    /// Check if vector is deleted
    pub fn is_deleted(&self, id: u32) -> bool {
        self.meta.is_deleted(id)
    }

    /// Get vector data (returns None if deleted or out of bounds)
    pub fn get(&self, id: u32) -> Result<Option<Vec<f32>>> {
        if (id as usize) >= self.len() || self.meta.is_deleted(id) {
            return Ok(None);
        }
        Ok(Some(self.row(id as usize).to_vec()))
    }

    /// Get metadata for vector
    pub fn get_meta(&self, id: u32) -> Result<Option<&[u8]>> {
        if (id as usize) >= self.len() {
            return Err(VectorDbError::IdOutOfBounds {
                id,
                capacity: self.len() as u32,
            });
        }
        Ok(self.meta.get(id))
    }

    /// Insert new vector, returns its ID
    ///
    /// An all-zero vector is accepted as a **placeholder row**: it is stored
    /// with its metadata (readable, deletable, compactable) but never enters
    /// the HNSW graph, because cosine distance to a zero vector is undefined.
    /// Searches never return it.
    pub fn insert(&mut self, v: &[f32], metadata: Option<&[u8]>) -> Result<u32> {
        if v.len() != self.cfg.dim {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.cfg.dim,
                got: v.len(),
            });
        }

        let n = self.len();
        let id = n as u32;
        let meta_bytes = metadata.unwrap_or(&[]);

        // Log to WAL before applying changes
        self.wal
            .log_insert(id, v, meta_bytes)
            .map_err(VectorDbError::Io)?;

        // Now apply to database
        if n == self.capacity() {
            self.grow()?;
        }
        self.row_mut(n).copy_from_slice(v);
        self.set_len_field(n + 1);
        self.meta.put(id, 0, meta_bytes)?;

        let view = self.vector_view();
        let dist = |a: u32, b: u32| view.distance(a, b);
        let is_deleted = |_id: u32| false;
        if is_placeholder_vector(v) {
            // Placeholder row: stored, never indexed — no distance to it exists.
            self.index.register_unlinked(id)?;
        } else {
            self.index.insert(id, dist, &is_deleted)?;
        }
        Ok(id)
    }

    /// Update existing vector (cannot update deleted vectors)
    pub fn update(&mut self, id: u32, v: &[f32], metadata: Option<&[u8]>) -> Result<()> {
        if v.len() != self.cfg.dim {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.cfg.dim,
                got: v.len(),
            });
        }
        if (id as usize) >= self.len() {
            return Err(VectorDbError::IdOutOfBounds {
                id,
                capacity: self.len() as u32,
            });
        }
        if self.meta.is_deleted(id) {
            return Err(VectorDbError::UpdateDeletedVector);
        }

        let meta_bytes = metadata.unwrap_or(&[]);

        // Log to WAL before applying changes
        self.wal
            .log_update(id, v, meta_bytes)
            .map_err(VectorDbError::Io)?;

        // Now apply to database
        self.row_mut(id as usize).copy_from_slice(v);
        self.meta.put(id, 0, meta_bytes)?;

        // Rebuild HNSW edges for this node with the new vector data
        self.rebuild_hnsw_node(id)?;

        Ok(())
    }

    /// Soft-delete vector (marks as tombstone)
    pub fn delete(&mut self, id: u32) -> Result<bool> {
        if (id as usize) >= self.len() || self.meta.is_deleted(id) {
            return Ok(false);
        }

        // Log to WAL before applying deletion
        let prev = self.meta.get(id).map(|b| b.to_vec()).unwrap_or_default();
        self.wal.log_delete(id, &prev).map_err(VectorDbError::Io)?;

        self.meta.put(id, META_FLAG_DELETED, &prev)?;
        self.index.mark_deleted(id);
        self.deleted_count += 1;

        Ok(true)
    }

    /// Search for k nearest neighbors
    ///
    /// # Errors
    /// - `DimensionMismatch` if the query dimension differs from the db's
    /// - `ZeroVector` if the query is all-zero (undefined for cosine distance)
    /// - any distance/storage error while traversing the graph
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchHit<'_>>> {
        if query.len() != self.cfg.dim {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.cfg.dim,
                got: query.len(),
            });
        }
        if is_placeholder_vector(query) {
            // No direction to compare: report it, never return made-up scores.
            return Err(VectorDbError::ZeroVector);
        }

        let view = self.vector_view();
        // Query finiteness and norm are validated once here rather than
        // re-checked for every visited candidate.
        let metric = QueryCosine::new(query)?;
        let dist = |id: u32| -> Result<f32> {
            let v = view.get(id).ok_or(VectorDbError::IdOutOfBounds {
                id,
                capacity: view.file_size as u32,
            })?;
            metric.distance(v)
        };
        let is_deleted = |id: u32| self.meta.is_deleted(id);

        let ef_actual = if k <= 100 {
            // For small-to-moderate k, reduce ef inflation
            ef.max(k * 2) + self.deleted_count.min(32)
        } else {
            // For very large k, maintain original formula
            ef.max(k).max(k * 4) + self.deleted_count.min(64)
        };

        self.index
            .search(k, ef_actual, dist, &is_deleted)?
            .into_iter()
            .map(|(id, d)| {
                Ok(SearchHit {
                    id,
                    score: (1.0 - d).clamp(0.0, 1.0),
                    metadata: self.meta.get(id).unwrap_or(&[]),
                })
            })
            .collect()
    }

    /// Search for k nearest neighbors with source filtering (pre-search filter)
    ///
    /// Filters results by source metadata before returning to caller.
    /// This avoids wasting work searching vectors that will be filtered out.
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of results to return
    /// * `ef` - Expansion factor for HNSW search
    /// * `source_filter` - Filter on source field
    ///
    /// # Returns
    /// Results matching both k-NN and source filter. May return fewer than k results
    /// if filtered results are exhausted before reaching k.
    pub fn search_with_source_filter(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        source_filter: &str,
    ) -> Result<Vec<SearchHit<'_>>> {
        if query.len() != self.cfg.dim {
            return Err(VectorDbError::DimensionMismatch {
                expected: self.cfg.dim,
                got: query.len(),
            });
        }
        if is_placeholder_vector(query) {
            // No direction to compare: report it, never return made-up scores.
            return Err(VectorDbError::ZeroVector);
        }

        let view = self.vector_view();
        // Query finiteness and norm are validated once here rather than
        // re-checked for every visited candidate.
        let metric = QueryCosine::new(query)?;
        let dist = |id: u32| -> Result<f32> {
            let v = view.get(id).ok_or(VectorDbError::IdOutOfBounds {
                id,
                capacity: view.file_size as u32,
            })?;
            metric.distance(v)
        };
        let is_deleted = |id: u32| self.meta.is_deleted(id);

        let ef_actual = if k <= 100 {
            ef.max(k * 2) + self.deleted_count.min(32)
        } else {
            ef.max(k).max(k * 4) + self.deleted_count.min(64)
        };

        // Search with expanded k to account for filtering
        let expanded_k = (k * 4).max(k + 100);
        let raw_results = self
            .index
            .search(expanded_k, ef_actual, dist, &is_deleted)?;

        let mut filtered = Vec::new();
        for (id, d) in raw_results {
            let metadata = self.meta.get(id).unwrap_or(&[]);
            // Try to extract source field from metadata JSON for filtering
            if let Ok(json_obj) = serde_json::from_slice::<serde_json::Value>(metadata)
                && let Some(source) = json_obj.get("source").and_then(|v| v.as_str())
                && source == source_filter
            {
                filtered.push(SearchHit {
                    id,
                    score: (1.0 - d).clamp(0.0, 1.0),
                    metadata,
                });
                if filtered.len() >= k {
                    break;
                }
            }
        }

        Ok(filtered)
    }

    /// Compact database by removing deleted vectors
    pub async fn compact(&mut self) -> Result<()> {
        if self.deleted_count == 0 {
            return Ok(());
        }

        let tmp_vec = with_suffix(&self.path, ".compact.tmp");
        let tmp_hnsw = hnsw_path(&tmp_vec);
        let tmp_meta = meta_path(&tmp_vec);

        let tmp_wal = with_suffix(&tmp_vec, ".wal");
        let _ = tokio::fs::remove_file(&tmp_vec).await;
        let _ = tokio::fs::remove_file(&tmp_hnsw).await;
        let _ = tokio::fs::remove_file(&tmp_meta).await;
        let _ = tokio::fs::remove_file(&tmp_wal).await;

        let live = self.live_len().max(1);
        let tmp_cfg = Config {
            initial_capacity: live.next_power_of_two(),
            ..self.cfg
        };

        {
            let mut fresh = VectorDb::open(&tmp_vec, tmp_cfg).await?;
            // Compaction reads every source row front-to-back once; hint the
            // kernel so it can read ahead aggressively for this pass.
            #[cfg(unix)]
            let _ = self.mmap().advise(memmap2::Advice::Sequential);
            for id in 0..self.len() as u32 {
                if self.meta.is_deleted(id) {
                    continue;
                }
                let v = self.row(id as usize).to_vec();
                let meta = self.meta.get(id).map(|b| b.to_vec()).unwrap_or_default();
                // Prod 4 fix: Use insert_raw to bypass WAL logging during compaction
                fresh.insert_raw(&v, &meta)?;
            }
            // The scan is over; normal access from here is random again.
            #[cfg(unix)]
            let _ = self.mmap().advise(memmap2::Advice::Random);
            fresh.flush().await?;
        }

        // The temp DB's WAL only ever held its checkpoint header (insert_raw
        // bypasses WAL logging), and its data has been flushed. Remove it so
        // compaction doesn't leave a stray temp WAL file behind.
        let _ = tokio::fs::remove_file(&tmp_wal).await;

        // Unmap before renaming
        self.mmap = None;
        self.index.mmap = None;

        tokio::fs::rename(&tmp_vec, &self.path)
            .await
            .map_err(VectorDbError::Io)?;
        tokio::fs::rename(&tmp_hnsw, hnsw_path(&self.path))
            .await
            .map_err(VectorDbError::Io)?;
        tokio::fs::rename(&tmp_meta, meta_path(&self.path))
            .await
            .map_err(VectorDbError::Io)?;

        // The compacted data files are complete and flushed, so every pending WAL
        // record is now both redundant and stale (their vector IDs predate the
        // compaction renumbering). Truncate the WAL *after* the rename succeeds so
        // a mid-compact failure still leaves the original WAL intact, and so
        // re-opening cannot replay pre-compaction records into the new files.
        self.wal.clear().await.map_err(VectorDbError::Io)?;

        // Shut down old WAL writer before re-opening (releases the .wal lock).
        self.wal.shutdown().await.map_err(VectorDbError::Io)?;

        // Release the advisory lock before re-opening the same path.
        // Replace with a harmless dummy file handle so the old lock is dropped.
        let dummy_lock = open_null_file().map_err(VectorDbError::Io)?;
        let _old_lock = std::mem::replace(&mut self._lock_file, dummy_lock);
        drop(_old_lock);

        let reopened = VectorDb::open(&self.path, self.cfg).await?;
        self.file = reopened.file;
        self.mmap = reopened.mmap;
        self.index = reopened.index;
        self.meta = reopened.meta;
        // Adopt the re-opened DB's live WAL writer and held lock. Keeping the old
        // (shut-down) writer or the dummy handle here would leave the database
        // without a working WAL or an advisory lock after compaction.
        self.wal = reopened.wal;
        self._lock_file = reopened._lock_file;
        self.deleted_count = 0;
        // Compaction rewrote every row and renumbered ids; verify the
        // rebuilt files satisfy all structural invariants before the
        // database is handed back to callers.
        self.integrity_check()?;
        Ok(())
    }

    /// Flush all pending changes to disk
    ///
    /// Order is critical for crash safety:
    /// 1. Sync data files first (so data is durable on disk)
    /// 2. Write WAL checkpoint marker + fsync (marks data as safe)
    /// 3. Clear WAL (remove old records, safe since data is persisted)
    ///
    /// If a crash occurs between step 1 and step 2, recovery will replay
    /// WAL records into already-flushed data files. The replay is idempotent:
    /// - Inserts: skipped if vector_id < len (already applied)
    /// - Deletes: skipped if already tombstoned
    /// - Updates: re-applied (idempotent, same data written again)
    pub async fn flush(&mut self) -> Result<()> {
        // Step 1: Sync all data files to disk BEFORE checkpoint.
        // Only written prefixes are msynced: rows beyond `len` and unused
        // capacity have never been dirtied, so syncing them would walk page
        // tables for nothing. The fixed header always falls inside the range
        // (it starts at offset 0), so len/capacity updates are covered too.
        self.mmap()
            .flush_range(0, self.row_offset(self.len()))
            .map_err(VectorDbError::Io)?;
        self.index.flush()?;
        self.meta.flush()?;

        // Step 2: Write checkpoint to WAL and wait for fsync confirmation
        // This marks all prior records as safely persisted
        self.wal.checkpoint().await.map_err(VectorDbError::Io)?;

        // Step 3: Clear WAL after successful checkpoint
        self.wal.clear().await.map_err(VectorDbError::Io)?;

        Ok(())
    }

    /// Properly shut down the database, waiting for WAL writer to finish
    /// and release all file locks
    pub async fn close(&mut self) -> Result<()> {
        self.wal.shutdown().await.map_err(VectorDbError::Io)?;
        Ok(())
    }
}

/// Search result with vector ID, similarity score, and metadata
#[derive(Debug)]
pub struct SearchHit<'a> {
    /// Vector ID
    pub id: u32,
    /// Cosine similarity score [0, 1]
    pub score: f32,
    /// Associated metadata bytes (empty if none)
    pub metadata: &'a [u8],
}

fn hnsw_path(p: &Path) -> PathBuf {
    with_suffix(p, ".hnsw")
}

fn meta_path(p: &Path) -> PathBuf {
    with_suffix(p, ".meta")
}

fn with_suffix(p: &Path, suffix: &str) -> PathBuf {
    let mut s = p.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

/// Open a platform-appropriate null file handle (used as a dummy for mem::replace)
fn open_null_file() -> io::Result<File> {
    #[cfg(unix)]
    {
        std::fs::OpenOptions::new().read(true).open("/dev/null")
    }
    #[cfg(windows)]
    {
        std::fs::OpenOptions::new().read(true).open("NUL")
    }
    #[cfg(not(any(unix, windows)))]
    {
        // Fallback: create a temporary file and immediately return it
        std::fs::File::open(std::env::temp_dir().join(".null_dummy"))
    }
}

// ============================================================================
//  Async Thread-Safe Wrapper
// ============================================================================

/// Thread-safe async wrapper around VectorDb using RwLock
///
/// Allows concurrent reads and exclusive writes, compatible with Tokio.
/// All operations are async-friendly and non-blocking.
///
/// # Example
/// ```no_run
/// use rust_rag_mcp::r_vector::{Config, AsyncVectorDb};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let cfg = Config::new(128);
///     let db = AsyncVectorDb::open("db.vec", cfg).await?;
///
///     // Concurrent inserts from multiple tasks
///     let mut handles = vec![];
///     for i in 0..10 {
///         let db = db.clone();
///         let handle = tokio::spawn(async move {
///             let vec = vec![0.1; 128];
///             db.insert(&vec, None).await
///         });
///         handles.push(handle);
///     }
///
///     for handle in handles {
///         let _ = handle.await;
///     }
///
///     Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct AsyncVectorDb {
    db: Arc<RwLock<VectorDb>>,
}

impl AsyncVectorDb {
    /// Open database with async support
    pub async fn open<P: AsRef<Path>>(path: P, cfg: Config) -> Result<Self> {
        let db = VectorDb::open(path, cfg).await?;
        Ok(Self {
            db: Arc::new(RwLock::new(db)),
        })
    }

    /// Get vector dimensionality
    pub async fn dim(&self) -> usize {
        self.db.read().await.dim()
    }

    /// Get total vector count (including deleted)
    pub async fn len(&self) -> usize {
        self.db.read().await.len()
    }

    /// Check if database is empty
    pub async fn is_empty(&self) -> bool {
        self.db.read().await.is_empty()
    }

    /// Get live vector count (excluding deleted)
    pub async fn live_len(&self) -> usize {
        self.db.read().await.live_len()
    }

    /// Get deletion statistics (async read)
    ///
    /// Returns (deleted_count, total_count, deletion_ratio)
    pub async fn deletion_stats(&self) -> (usize, usize, f32) {
        self.db.read().await.deletion_stats()
    }

    /// Validate structural integrity of the whole database (async read).
    ///
    /// See [`VectorDb::integrity_check`]: layout-agnostic invariants over
    /// the vector file, metadata coverage, and the HNSW graph. O(N) over
    /// mmap'd pages — runs on the blocking pool so its page faults don't
    /// stall async workers. For tests and diagnostics, not hot paths.
    pub async fn integrity_check(&self) -> Result<()> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || db.blocking_read().integrity_check())
            .await
            .map_err(|e| VectorDbError::Io(io::Error::other(e)))?
    }

    /// Check if compaction should be triggered (async read)
    pub async fn should_compact(&self) -> bool {
        self.db.read().await.should_compact()
    }

    /// Check if vector is deleted
    pub async fn is_deleted(&self, id: u32) -> bool {
        self.db.read().await.is_deleted(id)
    }

    /// Get vector (async read)
    pub async fn get(&self, id: u32) -> Result<Option<Vec<f32>>> {
        self.db.read().await.get(id)
    }

    /// Get metadata (async read)
    pub async fn get_meta(&self, id: u32) -> Result<Option<Vec<u8>>> {
        let db = self.db.read().await;
        Ok(db.get_meta(id)?.map(|m| m.to_vec()))
    }

    /// Insert vector (async write, exclusive lock)
    pub async fn insert(&self, v: &[f32], metadata: Option<&[u8]>) -> Result<u32> {
        self.db.write().await.insert(v, metadata)
    }

    /// Update vector (async write, exclusive lock)
    pub async fn update(&self, id: u32, v: &[f32], metadata: Option<&[u8]>) -> Result<()> {
        self.db.write().await.update(id, v, metadata)
    }

    /// Delete vector (async write, exclusive lock)
    pub async fn delete(&self, id: u32) -> Result<bool> {
        self.db.write().await.delete(id)
    }

    /// Search for k neighbors (async read)
    ///
    /// The traversal runs on the blocking pool: it synchronously page-faults
    /// mmap'd vector and graph pages in, which on the async pool would stall
    /// the worker — and with it every other task sharing it — for the whole
    /// duration of the faults. The `Arc` handle is cloned (cheap) and the
    /// read guard is acquired on the blocking thread itself.
    pub async fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchHitOwned>> {
        let db = self.db.clone();
        let query = query.to_vec();
        let hits = tokio::task::spawn_blocking(move || -> Result<Vec<SearchHitOwned>> {
            let guard = db.blocking_read();
            let hits = guard.search(&query, k, ef)?;
            Ok(hits
                .into_iter()
                .map(|h| SearchHitOwned {
                    id: h.id,
                    score: h.score,
                    metadata: h.metadata.to_vec(),
                })
                .collect())
        })
        .await
        .map_err(|e| VectorDbError::Io(io::Error::other(e)))??;
        Ok(hits)
    }

    /// Compact database (async write, exclusive lock)
    pub async fn compact(&self) -> Result<()> {
        self.db.write().await.compact().await
    }

    /// Flush changes to disk (async write, exclusive lock)
    pub async fn flush(&self) -> Result<()> {
        self.db.write().await.flush().await
    }

    /// Properly shut down the database, waiting for WAL writer to finish
    /// and release all file locks
    pub async fn close(&self) -> Result<()> {
        let mut db = self.db.write().await;
        db.close().await
    }
}

/// Owned version of SearchHit for async API
#[derive(Debug)]
pub struct SearchHitOwned {
    /// Vector ID
    pub id: u32,
    /// Cosine similarity score [0, 1]
    pub score: f32,
    /// Associated metadata bytes (owned)
    pub metadata: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_distance_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let dist = cosine_distance(&a, &b).unwrap();
        assert!((dist - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let dist = cosine_distance(&a, &b).unwrap();
        assert!((dist - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let dist = cosine_distance(&a, &b).unwrap();
        assert!((dist - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance_zero_vector() {
        // Undefined input → error, never a fabricated number.
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        let err = cosine_distance(&a, &b).unwrap_err();
        assert!(matches!(err, VectorDbError::ZeroVector));
    }

    #[test]
    fn test_cosine_distance_zero_vector_self_is_error() {
        // Regression: the old 1e-20 epsilon norm floor made
        // cosine_distance(zero, zero) return 1.0 — a number for an
        // undefined input, violating both the "0 = identical" contract
        // and error propagation. It must error instead.
        let a = vec![0.0, -0.0, 0.0];
        let err = cosine_distance(&a, &a).unwrap_err();
        assert!(matches!(err, VectorDbError::ZeroVector));
    }

    #[test]
    fn test_cosine_distance_mismatched_lengths() {
        // The fallible API returns an error instead of panicking (the old
        // `#[should_panic]` assertion predates the length check).
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let err = cosine_distance(&a, &b).unwrap_err();
        assert!(
            matches!(
                err,
                VectorDbError::DimensionMismatch {
                    expected: 2,
                    got: 3
                }
            ),
            "unexpected error: {err:?}"
        );
    }

    /// The query-side hoisting in `QueryCosine` must be a pure optimization:
    /// results are bit-identical to `cosine_distance` (same `f64` ops;
    /// IEEE multiplication is commutative, so operand order cannot change
    /// any bit), so search ranking cannot shift.
    #[test]
    fn query_cosine_matches_cosine_distance_bit_exactly() {
        let mut rng = Rng::new(7);
        let query = random_nonzero(&mut rng, 64);
        let metric = QueryCosine::new(&query).unwrap();
        for _ in 0..100 {
            let row = random_nonzero(&mut rng, 64);
            let expected = cosine_distance(&row, &query).unwrap();
            let got = metric.distance(&row).unwrap();
            assert_eq!(got, expected, "hoisted query norm changed the distance");
            // Operand order must not matter either.
            assert_eq!(got, cosine_distance(&query, &row).unwrap());
        }
    }

    /// `QueryCosine` mirrors `cosine_distance`'s error contract exactly,
    /// but reports query-side problems eagerly at construction (before any
    /// graph traversal starts).
    #[test]
    fn query_cosine_error_contracts() {
        assert!(matches!(
            QueryCosine::new(&[f32::NAN, 1.0]),
            Err(VectorDbError::NaNDistance)
        ));
        assert!(matches!(
            QueryCosine::new(&[0.0, -0.0]),
            Err(VectorDbError::ZeroVector)
        ));

        let query = [1.0, 0.0];
        let metric = QueryCosine::new(&query).unwrap();
        assert!(matches!(
            metric.distance(&[f32::NAN, 1.0]),
            Err(VectorDbError::NaNDistance)
        ));
        assert!(matches!(
            metric.distance(&[0.0, 0.0]),
            Err(VectorDbError::ZeroVector)
        ));
        assert!(matches!(
            metric.distance(&[1.0, 0.0, 0.0]),
            Err(VectorDbError::DimensionMismatch {
                expected: 2,
                got: 3
            })
        ));
    }

    #[test]
    fn test_config_defaults() {
        let cfg = Config::new(128);
        assert_eq!(cfg.dim, 128);
        assert_eq!(cfg.m, 20);
        assert_eq!(cfg.m0, 30);
        assert_eq!(cfg.ef_construction, 150);
        assert_eq!(cfg.max_level, 7);
    }

    #[test]
    fn test_config_builder() {
        let cfg = Config::new(256)
            .with_m(40)
            .with_ef_construction(300)
            .with_capacity(2048)
            .with_max_level(10);
        assert_eq!(cfg.dim, 256);
        assert_eq!(cfg.m, 40);
        assert_eq!(cfg.m0, 80); // m * 2
        assert_eq!(cfg.ef_construction, 300);
        assert_eq!(cfg.initial_capacity, 2048);
        assert_eq!(cfg.max_level, 10);
    }

    #[test]
    fn test_rng_reproducibility() {
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_rng_values_in_range() {
        let mut rng = Rng::new(1);
        for _ in 0..1000 {
            let f = rng.next_f64();
            assert!(f >= 0.0);
            assert!(f < 1.0);
        }
    }

    // -------------------------------------------------------------------
    //  Structural integrity safety net
    // -------------------------------------------------------------------

    /// Deterministic non-zero vector (an all-zero vector would be stored as
    /// an unlinked placeholder row instead of entering the graph).
    fn random_nonzero(rng: &mut Rng, dim: usize) -> Vec<f32> {
        loop {
            let v: Vec<f32> = (0..dim)
                .map(|_| (rng.next_f64() * 2.0 - 1.0) as f32)
                .collect();
            if !is_placeholder_vector(&v) {
                return v;
            }
        }
    }

    /// Structured insert/delete/update sequence: every phase (including a
    /// flush + reopen cycle) must leave all three files structurally sound.
    #[tokio::test]
    async fn integrity_check_after_random_operations() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db");
        let cfg = Config::new(16).with_capacity(32).with_seed(0xA11CE);

        {
            let mut db = VectorDb::open(&path, cfg).await.unwrap();
            let mut rng = Rng::new(4242);
            let mut ids = Vec::new();
            for i in 0..200u32 {
                let v = random_nonzero(&mut rng, 16);
                let id = db.insert(&v, Some(format!("row-{i}").as_bytes())).unwrap();
                ids.push(id);
            }
            // A placeholder row exercises the unlinked-node bookkeeping path.
            db.insert(&[0.0; 16], Some(b"placeholder")).unwrap();
            db.integrity_check().unwrap();

            // Tombstone a third of the rows (deleted nodes keep their edges).
            for &id in ids.iter().step_by(3) {
                assert!(db.delete(id).unwrap());
            }
            db.integrity_check().unwrap();

            // Update every third surviving row (rebuilds its edges in place).
            for &id in ids.iter().skip(1).step_by(3) {
                if !db.is_deleted(id) {
                    let v = random_nonzero(&mut rng, 16);
                    db.update(id, &v, None).unwrap();
                }
            }
            db.integrity_check().unwrap();
            db.flush().await.unwrap();
        }

        // Reopen: the on-disk structures must satisfy the same invariants.
        {
            let db = VectorDb::open(&path, cfg).await.unwrap();
            db.integrity_check().unwrap();
            assert_eq!(db.len(), 201, "200 real rows + 1 placeholder");
        }
    }

    /// The checker must actually detect violations — a test-only checker that
    /// silently passes corrupt data would give false confidence.
    #[tokio::test]
    async fn integrity_check_detects_corrupt_graph() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db");
        let cfg = Config::new(8).with_capacity(64).with_seed(7);

        let mut db = VectorDb::open(&path, cfg).await.unwrap();
        let mut rng = Rng::new(99);
        for _ in 0..30 {
            let v = random_nonzero(&mut rng, 8);
            db.insert(&v, None).unwrap();
        }
        db.integrity_check().unwrap();

        // Find a node that actually has layer-0 neighbors to corrupt.
        let count = db.index.count() as u32;
        let linked = (0..count).find(|&id| db.index.layer_count(id, 0) > 0);
        let linked = linked.expect("graph with 30 nodes must have some edges");

        // 1. Out-of-range neighbor.
        let saved = db.index.layer_neighbor(linked, 0, 0);
        db.index.set_layer_neighbor(linked, 0, 0, count + 1000);
        assert!(
            db.integrity_check().is_err(),
            "out-of-range neighbor must be detected"
        );
        db.index.set_layer_neighbor(linked, 0, 0, saved);

        // 2. Self-link.
        db.index.set_layer_neighbor(linked, 0, 0, linked);
        assert!(db.integrity_check().is_err(), "self-link must be detected");
        db.index.set_layer_neighbor(linked, 0, 0, saved);

        // Restored graph is clean again.
        db.integrity_check().unwrap();
    }

    /// Row layout guard: rows must occupy a uniform stride inside the vector
    /// mapping and round-trip exactly. (A layout change that makes rows
    /// page-aligned keeps the stride uniform — only its value changes.)
    /// Runs for a row size that divides the page (must never straddle) and
    /// one that doesn't (uniformity must still hold).
    #[tokio::test]
    async fn vector_rows_use_uniform_stride() {
        let page = page_size();
        for dim in [16usize, 24] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("db");
            let cfg = Config::new(dim).with_capacity(8);

            let mut db = VectorDb::open(&path, cfg).await.unwrap();
            let mut rng = Rng::new(3);
            let mut stored = Vec::new();
            for _ in 0..50 {
                let v = random_nonzero(&mut rng, dim);
                db.insert(&v, None).unwrap();
                stored.push(v);
            }

            let stride = db.row_offset(1) - db.row_offset(0);
            assert_eq!(stride, dim * 4, "stride must equal the row size");
            // v3 format: the row area starts on a page boundary, so rows that
            // fit inside one page and divide it never straddle — one row
            // access, one page fault.
            assert_eq!(
                db.row_offset(0) % page,
                0,
                "row area must start at a page boundary"
            );
            if stride <= page && page.is_multiple_of(stride) {
                for i in 0..stored.len() {
                    assert!(
                        db.row_offset(i) % page + stride <= page,
                        "row {i} straddles a page boundary"
                    );
                }
            }
            for (i, v) in stored.iter().enumerate() {
                assert_eq!(db.row_offset(i + 1) - db.row_offset(i), stride, "row {i}");
                assert_eq!(
                    db.get(i as u32).unwrap(),
                    Some(v.clone()),
                    "row {i} must round-trip exactly"
                );
            }
            assert!(
                db.row_offset(50) <= db.mmap().len(),
                "rows must fit inside the mapping"
            );
            db.integrity_check().unwrap();
        }
    }

    /// The row-area alignment changed the on-disk format; files written by the
    /// previous version must be rejected loudly instead of being misread
    /// (the old data area at offset 32 would silently look like garbage rows).
    #[tokio::test]
    async fn open_rejects_previous_format_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db");
        // Craft a v2 header: magic ...0002, dim 4, len 0, capacity 8.
        let mut bytes = vec![0u8; 32 + 8 * 4 * 4];
        bytes[0..8].copy_from_slice(&0x5645_4354_4F52_0002u64.to_le_bytes());
        bytes[8..16].copy_from_slice(&4u64.to_le_bytes());
        bytes[24..32].copy_from_slice(&8u64.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        let cfg = Config::new(4).with_capacity(8);
        let err = match VectorDb::open(&path, cfg).await {
            Err(e) => e,
            Ok(_) => panic!("opening a v2-format file must fail"),
        };
        assert!(
            matches!(&err, VectorDbError::Corruption(m) if m.contains("invalid vector magic")),
            "expected magic rejection, got {err:?}"
        );
    }

    /// v3 sizes each node block for its *own* level instead of reserving
    /// `max_level` layers on every node. With defaults, ~95% of nodes live at
    /// layer 0, so the arena must come out far smaller than the old fixed
    /// `node_bytes(cfg, max_level)` layout — and blocks must never move
    /// (updates rewrite in place, so `arena_end` is stable).
    #[tokio::test]
    async fn hnsw_nodes_use_compact_blocks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db");
        let cfg = Config::new(16).with_capacity(8);
        let (count, arena_before) = {
            let mut rng = Rng::new(11);
            let mut db = VectorDb::open(&path, cfg).await.unwrap();
            for _ in 0..500 {
                let v = random_nonzero(&mut rng, 16);
                db.insert(&v, None).unwrap();
            }
            let count = db.index.count();
            assert_eq!(count, 500);
            db.integrity_check().unwrap();

            let arena = db.index.arena_end() - HNSW_HEADER;
            let avg = arena / count;
            let worst_fixed = HnswIndex::node_bytes(&db.cfg, db.cfg.max_level);
            assert!(
                avg < 300,
                "average node block is {avg} B — not compact (old fixed layout was \
                 {worst_fixed} B/node)"
            );
            assert!(avg < worst_fixed, "compact layout must beat fixed blocks");

            // Updates rebuild edges but the level (hence block size) is fixed
            // at creation: no block may be relocated, so the arena is stable.
            let arena_before = db.index.arena_end();
            for id in [0u32, 100, 250, 499] {
                let v = random_nonzero(&mut rng, 16);
                db.update(id, &v, None).unwrap();
            }
            assert_eq!(
                db.index.arena_end(),
                arena_before,
                "updates must not move node blocks"
            );
            db.integrity_check().unwrap();
            db.close().await.unwrap();
            (count, arena_before)
        };

        // Reopen rebuilds the directory by walking the blocks: counts and
        // arena_end must reconstruct exactly.
        let db = VectorDb::open(&path, cfg).await.unwrap();
        assert_eq!(db.index.count(), count);
        assert_eq!(
            db.index.arena_end(),
            arena_before,
            "walk must reproduce arena"
        );
        db.integrity_check().unwrap();
    }

    /// The v3 node layout changed with variable-length blocks; files written
    /// by the previous fixed-stride format must be rejected, not misread.
    #[tokio::test]
    async fn open_rejects_previous_hnsw_format() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.hnsw");
        let mut bytes = vec![0u8; HNSW_HEADER];
        bytes[0..8].copy_from_slice(&0x484E_5357_0000_0002u64.to_le_bytes()); // v2 magic
        std::fs::write(&path, &bytes).unwrap();

        let cfg = Config::new(4).with_capacity(8);
        let err = match HnswIndex::open(&path, &cfg).await {
            Err(e) => e,
            Ok(_) => panic!("opening a v2-format HNSW file must fail"),
        };
        assert!(
            matches!(&err, VectorDbError::Corruption(m) if m.contains("invalid HNSW magic")),
            "expected magic rejection, got {err:?}"
        );
    }
}
