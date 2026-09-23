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
            let mut m = unsafe { MmapOptions::new().map_mut(&store.file)? };
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
            let m = unsafe { MmapOptions::new().map_mut(&store.file)? };

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

            // Truncate any bytes beyond the flushed records (leftover from grow()
            // that were never committed via flush())
            if pos < len {
                drop(m);
                store.file.set_len(pos as u64).map_err(VectorDbError::Io)?;
                let m = unsafe { MmapOptions::new().map_mut(&store.file)? };
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
            m.flush().map_err(VectorDbError::Io)?;
        }
        self.mmap = None;
        self.file
            .set_len(new_size as u64)
            .map_err(VectorDbError::Io)?;
        self.mmap = Some(unsafe { MmapOptions::new().map_mut(&self.file)? });
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        let count = self.record_count;
        self.mmap_mut()[8..12].copy_from_slice(&count.to_le_bytes());
        self.mmap().flush().map_err(VectorDbError::Io)?;
        self.file
            .set_len(self.file_len as u64)
            .map_err(VectorDbError::Io)?;
        self.mmap = Some(unsafe { MmapOptions::new().map_mut(&self.file)? });
        Ok(())
    }
}

// ============================================================================
//  HNSW Index
// ============================================================================

const HNSW_MAGIC: u64 = 0x484E_5357_0000_0002;
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
    /// Reusable scratch buffers for search_layer and insert to avoid per-call allocations.
    /// Safety: only accessed within &self or &mut self methods, never across threads.
    /// VectorDb is behind RwLock, so concurrent access is prevented externally.
    scratch: UnsafeCell<SearchBuffers>,
}

// SAFETY: HnswIndex is only accessed through VectorDb which is behind RwLock,
// preventing concurrent access to scratch buffers.
unsafe impl Sync for HnswIndex {}

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
    fn new(cfg: &Config) -> Self {
        let ef = cfg.ef_construction;
        Self {
            visited: HashSet::with_capacity(ef * 2),
            candidates: BinaryHeap::with_capacity(ef),
            results: BinaryHeap::with_capacity(ef + 1),
            search_output: Vec::with_capacity(ef),
            select_result: Vec::with_capacity(cfg.m0),
            select_seen: HashSet::with_capacity(cfg.m0 * 2),
            cross_cache: HashMap::new(),
            dists: Vec::with_capacity(ef),
        }
    }
}

impl HnswIndex {
    /// Calculate bytes per node (all layers combined)
    fn node_bytes(cfg: &Config) -> usize {
        // 8 bytes: level (u8) + flags (u8) + padding (6)
        // Layer 0: 4 (count) + m0*4 (neighbors)
        // Layers 1..max: (4 + m*4) each
        8 + (4 + cfg.m0 * 4) + cfg.max_level * (4 + cfg.m * 4)
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
            // Initialize new index
            let cap = cfg.initial_capacity.max(16);
            let size = HNSW_HEADER + cap * Self::node_bytes(cfg);
            file.set_len(size as u64).map_err(VectorDbError::Io)?;
            let mut m = unsafe { MmapOptions::new().map_mut(&file)? };
            m[0..8].copy_from_slice(&HNSW_MAGIC.to_le_bytes());
            m[8..16].copy_from_slice(&0u64.to_le_bytes()); // count
            m[16..20].copy_from_slice(&NONE.to_le_bytes()); // entry point
            m[20] = 0; // max level so far
            m[24..32].copy_from_slice(&(cap as u64).to_le_bytes());
            m[32..40].copy_from_slice(&(cfg.m as u64).to_le_bytes());
            m[40..48].copy_from_slice(&(cfg.m0 as u64).to_le_bytes());
            m[48..56].copy_from_slice(&(cfg.max_level as u64).to_le_bytes());
            m[56..64].copy_from_slice(&(cfg.ef_construction as u64).to_le_bytes());
            m[64..72].copy_from_slice(&cfg.seed.to_le_bytes());
            m.flush().map_err(VectorDbError::Io)?;
            Ok(Self {
                file,
                mmap: Some(m),
                cfg: *cfg,
                rng: Rng::new(cfg.seed),
                scratch: UnsafeCell::new(SearchBuffers::new(cfg)),
            })
        } else {
            // Load existing index
            let m = unsafe { MmapOptions::new().map_mut(&file)? };

            // Validate magic
            let magic = u64::from_le_bytes(m[0..8].try_into().map_err(|_| {
                VectorDbError::Corruption("file too small for HNSW header".to_string())
            })?);
            if magic != HNSW_MAGIC {
                return Err(VectorDbError::Corruption("invalid HNSW magic".to_string()));
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
            Ok(Self {
                file,
                mmap: Some(m),
                cfg: on_disk,
                rng: Rng::new(seed),
                scratch: UnsafeCell::new(SearchBuffers::new(&on_disk)),
            })
        }
    }

    fn mmap(&self) -> &MmapMut {
        self.mmap.as_ref().expect("hnsw mmap present")
    }

    fn mmap_mut(&mut self) -> &mut MmapMut {
        self.mmap.as_mut().expect("hnsw mmap present")
    }

    fn count(&self) -> usize {
        u64::from_le_bytes(self.mmap()[8..16].try_into().unwrap()) as usize
    }

    fn capacity(&self) -> usize {
        u64::from_le_bytes(self.mmap()[24..32].try_into().unwrap()) as usize
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
        HNSW_HEADER + id as usize * Self::node_bytes(&self.cfg)
    }

    fn node_level(&self, id: u32) -> u8 {
        self.mmap()[self.node_offset(id)]
    }

    fn set_level(&mut self, id: u32, l: u8) {
        let off = self.node_offset(id);
        self.mmap_mut()[off] = l;
    }

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

    fn grow(&mut self) -> Result<()> {
        let new_cap = (self.capacity() * 2).max(16);
        let new_size = HNSW_HEADER + new_cap * Self::node_bytes(&self.cfg);
        self.mmap = None;
        self.file
            .set_len(new_size as u64)
            .map_err(VectorDbError::Io)?;
        let mut m = unsafe { MmapOptions::new().map_mut(&self.file)? };
        m[24..32].copy_from_slice(&(new_cap as u64).to_le_bytes());
        self.mmap = Some(m);
        Ok(())
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
        let scratch = unsafe { &mut *self.scratch.get() };
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
        while (new_id as usize) >= self.capacity() {
            self.grow()?;
        }
        let level = self.sample_level();
        self.set_level(new_id, level as u8);
        for l in 0..=level {
            self.set_layer_count(new_id, l, 0);
        }
        self.set_count((new_id as usize + 1).max(self.count()));
        Ok(())
    }

    /// Insert new node into index (must be called with proper distance function)
    fn insert<F, D>(&mut self, new_id: u32, dist: F, is_deleted: &D) -> Result<()>
    where
        F: Fn(u32, u32) -> Result<f32> + Copy,
        D: Fn(u32) -> bool,
    {
        while (new_id as usize) >= self.capacity() {
            self.grow()?;
        }

        let level = self.sample_level();
        self.set_level(new_id, level as u8);
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
                let scratch = unsafe { &mut *self.scratch.get() };
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

            let scratch = unsafe { &*self.scratch.get() };
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
        if (id as usize) < self.capacity() {
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
        let scratch = unsafe { &*self.scratch.get() };
        scratch
            .search_output
            .iter()
            .take(k)
            .map(|&(d, id)| Ok((id, d)))
            .collect()
    }

    fn flush(&self) -> Result<()> {
        self.mmap().flush().map_err(VectorDbError::Io)
    }
}

// ============================================================================
//  Vector Storage
// ============================================================================

const VEC_MAGIC: u64 = 0x5645_4354_4F52_0002;
const VEC_HEADER: usize = 32;

/// View into vector data within mmap with bounds checking
struct VectorView {
    ptr: *const u8,
    dim: usize,
    file_size: usize,
}

impl VectorView {
    /// Get vector at given ID with bounds checking
    ///
    /// # Errors
    /// Returns None if ID is out of bounds
    fn get(&self, id: u32) -> Option<&[f32]> {
        let start = VEC_HEADER + (id as usize).checked_mul(self.dim)?.checked_mul(4)?;
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
            let total = VEC_HEADER + cfg.initial_capacity * cfg.dim * 4;
            file.set_len(total as u64).map_err(VectorDbError::Io)?;
            let mut m = unsafe { MmapOptions::new().map_mut(&file)? };
            m[0..8].copy_from_slice(&VEC_MAGIC.to_le_bytes());
            m[8..16].copy_from_slice(&(cfg.dim as u64).to_le_bytes());
            m[16..24].copy_from_slice(&0u64.to_le_bytes());
            m[24..32].copy_from_slice(&(cfg.initial_capacity as u64).to_le_bytes());
            m.flush().map_err(VectorDbError::Io)?;
            m
        } else {
            let m = unsafe { MmapOptions::new().map_mut(&file)? };
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
                        // before crash, metadata/metadata not yet flushed).
                        // Restore metadata from WAL record.
                        self.meta.put(record.vector_id, 0, &record.metadata)?;
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

        if recovered_count > 0 {
            self.flush().await?;
        }

        Ok(())
    }

    /// Rebuild HNSW edges for a single node after vector data changed
    fn rebuild_hnsw_node(&mut self, id: u32) -> Result<()> {
        let view = self.vector_view();
        let dist = |a: u32, b: u32| view.distance(a, b);
        let is_deleted = |id: u32| self.meta.is_deleted(id);

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
                let scratch = unsafe { &mut *self.index.scratch.get() };
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
            let scratch = unsafe { &*self.index.scratch.get() };
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

    fn capacity(&self) -> usize {
        u64::from_le_bytes(self.mmap()[24..32].try_into().unwrap()) as usize
    }

    fn set_len_field(&mut self, n: usize) {
        self.mmap_mut()[16..24].copy_from_slice(&(n as u64).to_le_bytes());
    }

    fn row(&self, i: usize) -> &[f32] {
        let dim = self.cfg.dim;
        let start = VEC_HEADER + i * dim * 4;
        let bytes = &self.mmap()[start..start + dim * 4];
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, dim) }
    }

    fn row_mut(&mut self, i: usize) -> &mut [f32] {
        let dim = self.cfg.dim;
        let start = VEC_HEADER + i * dim * 4;
        let bytes = &mut self.mmap_mut()[start..start + dim * 4];
        unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, dim) }
    }

    fn grow(&mut self) -> Result<()> {
        let new_cap = (self.capacity() * 2).max(16);
        let new_len = VEC_HEADER + new_cap * self.cfg.dim * 4;
        self.mmap = None;
        self.file
            .set_len(new_len as u64)
            .map_err(VectorDbError::Io)?;
        let mut m = unsafe { MmapOptions::new().map_mut(&self.file)? };
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
        let dist = |id: u32| -> Result<f32> {
            let v = view.get(id).ok_or(VectorDbError::IdOutOfBounds {
                id,
                capacity: view.file_size as u32,
            })?;
            cosine_distance(v, query)
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
        let dist = |id: u32| -> Result<f32> {
            let v = view.get(id).ok_or(VectorDbError::IdOutOfBounds {
                id,
                capacity: view.file_size as u32,
            })?;
            cosine_distance(v, query)
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

        let _ = tokio::fs::remove_file(&tmp_vec).await;
        let _ = tokio::fs::remove_file(&tmp_hnsw).await;
        let _ = tokio::fs::remove_file(&tmp_meta).await;

        let live = self.live_len().max(1);
        let tmp_cfg = Config {
            initial_capacity: live.next_power_of_two(),
            ..self.cfg
        };

        {
            let mut fresh = VectorDb::open(&tmp_vec, tmp_cfg).await?;
            for id in 0..self.len() as u32 {
                if self.meta.is_deleted(id) {
                    continue;
                }
                let v = self.row(id as usize).to_vec();
                let meta = self.meta.get(id).map(|b| b.to_vec()).unwrap_or_default();
                // Prod 4 fix: Use insert_raw to bypass WAL logging during compaction
                fresh.insert_raw(&v, &meta)?;
            }
            fresh.flush().await?;
        }

        // Unmap before renaming
        self.mmap = None;
        self.index.mmap = None;

        // Shut down old WAL writer before re-opening
        self.wal.shutdown().await.map_err(VectorDbError::Io)?;

        // Release the advisory lock before re-opening the same path.
        // Replace with a harmless dummy file handle so the old lock is dropped.
        let dummy_lock = open_null_file().map_err(VectorDbError::Io)?;
        let _old_lock = std::mem::replace(&mut self._lock_file, dummy_lock);
        drop(_old_lock);

        tokio::fs::rename(&tmp_vec, &self.path)
            .await
            .map_err(VectorDbError::Io)?;
        tokio::fs::rename(&tmp_hnsw, hnsw_path(&self.path))
            .await
            .map_err(VectorDbError::Io)?;
        tokio::fs::rename(&tmp_meta, meta_path(&self.path))
            .await
            .map_err(VectorDbError::Io)?;

        let reopened = VectorDb::open(&self.path, self.cfg).await?;
        self.file = reopened.file;
        self.mmap = reopened.mmap;
        self.index = reopened.index;
        self.meta = reopened.meta;
        self.deleted_count = 0;
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
        // Step 1: Sync all data files to disk BEFORE checkpoint
        self.mmap().flush().map_err(VectorDbError::Io)?;
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
    pub async fn search(&self, query: &[f32], k: usize, ef: usize) -> Result<Vec<SearchHitOwned>> {
        let db = self.db.read().await;
        let hits = db.search(query, k, ef)?;
        Ok(hits
            .into_iter()
            .map(|h| SearchHitOwned {
                id: h.id,
                score: h.score,
                metadata: h.metadata.to_vec(),
            })
            .collect())
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
}
