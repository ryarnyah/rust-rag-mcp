//! Write-Ahead Logging (WAL) - PostgreSQL style with async writer thread
//!
//! Architecture:
//! - Main DB thread: Sends log requests via channel to WAL writer
//! - WAL writer thread: Batches, writes, and fsyncs to disk
//! - Recovery: On startup, replays WAL from last checkpoint
//!
//! All operations follow: LOG-BEFORE-APPLY pattern
//! 1. Send log record to WAL via channel
//! 2. Wait for WAL confirmation (fsync'd to disk)
//! 3. Apply to data files
//! 4. Return to user

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{mpsc, oneshot};
use crc32fast::Hasher as Crc32Hasher;
use fs2::FileExt;
// Note: crc32fast::hash() is used for one-shot CRC32 computation in WalRecord::to_bytes()

/// WAL operation statistics for monitoring
#[derive(Debug, Default)]
pub struct WalMetrics {
    /// Total records written to WAL
    pub records_written: AtomicU64,
    /// Total bytes written to WAL (including headers/footers)
    pub bytes_written: AtomicU64,
    /// Total checkpoints performed
    pub checkpoints: AtomicU64,
    /// Total segment rotations
    pub segment_rotations: AtomicU64,
    /// Total segments cleaned up
    pub segments_cleaned: AtomicU64,
}

/// Log Sequence Number - unique position in WAL stream
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Lsn(u64);

impl Lsn {
    /// Get the raw u64 value
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// WAL operation types
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WalOpType {
    Insert = 1,
    Delete = 2,
    Update = 3,
    Checkpoint = 4,
}

impl WalOpType {
    fn from_u8(b: u8) -> io::Result<Self> {
        match b {
            1 => Ok(WalOpType::Insert),
            2 => Ok(WalOpType::Delete),
            3 => Ok(WalOpType::Update),
            4 => Ok(WalOpType::Checkpoint),
            _ => Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid WAL op type")),
        }
    }
}

/// WAL record - fixed 34-byte header + variable payload + 4-byte footer
///
/// Format:
/// [0..8]   LSN (u64)
/// [8..16]  Timestamp (u64) - milliseconds since UNIX epoch
/// [16]     Op type (u8)
/// [17]     Padding (u8)
/// [18..22] Vector ID (u32)
/// [22..26] Vector len in floats (u32)
/// [26..30] Metadata len (u32)
/// [30..34] CRC32 of payload (u32)
/// [34+]    Payload: vector data (4*vector_len bytes) + metadata (variable)
/// [..+4]   Footer: 0xDEADBEEF (u32)
#[derive(Debug, Clone)]
pub struct WalRecord {
    pub lsn: Lsn,
    pub timestamp: u64,
    pub op_type: WalOpType,
    pub vector_id: u32,
    pub vector_data: Vec<f32>,
    pub metadata: Vec<u8>,
}

impl WalRecord {
    /// Serialize to a new byte vector
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.write_to(&mut buf);
        buf
    }

    /// Serialize directly into an existing buffer (avoids allocation)
    pub fn write_to(&self, buf: &mut Vec<u8>) {
        let vector_bytes = self.vector_data.len() * 4;
        let metadata_len = self.metadata.len();
        let payload_len = vector_bytes + metadata_len;
        let total = 34 + payload_len + 4;

        buf.reserve(total - buf.len());
        buf.clear();

        // Header (34 bytes fixed)
        buf.extend_from_slice(&self.lsn.0.to_le_bytes());           // [0..8]
        buf.extend_from_slice(&self.timestamp.to_le_bytes());       // [8..16] timestamp
        buf.push(self.op_type as u8);                               // [16]
        buf.push(0);                                                // [17] padding
        buf.extend_from_slice(&self.vector_id.to_le_bytes());       // [18..22]
        buf.extend_from_slice(&(self.vector_data.len() as u32).to_le_bytes()); // [22..26]
        buf.extend_from_slice(&(metadata_len as u32).to_le_bytes()); // [26..30]
        // CRC32 placeholder (will fill after payload)
        buf.extend_from_slice(&0u32.to_le_bytes());                 // [30..34]

        // Payload: vector data as contiguous bytes
        let payload_start = buf.len();
        for f in &self.vector_data {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        buf.extend_from_slice(&self.metadata);

        // Compute CRC32 over contiguous payload
        let crc32 = crc32fast::hash(&buf[payload_start..buf.len()]);
        buf[30..34].copy_from_slice(&crc32.to_le_bytes());

        // Footer
        buf.extend_from_slice(&0xDEADBEEFu32.to_le_bytes());
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> io::Result<(Self, usize)> {
        if data.len() < 36 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Incomplete WAL record",
            ));
        }

        // Parse header (34 bytes)
        let lsn = Lsn(u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]));
        let timestamp = u64::from_le_bytes([
            data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
        ]);
        let op_type = WalOpType::from_u8(data[16])?;
        let vector_id = u32::from_le_bytes([data[18], data[19], data[20], data[21]]);
        let vector_len = u32::from_le_bytes([data[22], data[23], data[24], data[25]]) as usize;
        let metadata_len = u32::from_le_bytes([data[26], data[27], data[28], data[29]]) as usize;
        let expected_crc32 = u32::from_le_bytes([data[30], data[31], data[32], data[33]]);

        let vector_bytes = vector_len * 4;
        let payload_len = vector_bytes + metadata_len;
        let total_len = 34 + payload_len + 4; // header + payload + footer

        if data.len() < total_len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Incomplete WAL record payload",
            ));
        }

        // Verify footer
        let footer_offset = 34 + payload_len;
        let footer = u32::from_le_bytes([
            data[footer_offset],
            data[footer_offset + 1],
            data[footer_offset + 2],
            data[footer_offset + 3],
        ]);
        if footer != 0xDEADBEEF {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid WAL record footer",
            ));
        }

        // Verify CRC32 of payload
        let payload = &data[34..34 + payload_len];
        let mut hasher = Crc32Hasher::new();
        hasher.update(payload);
        let computed_crc32 = hasher.finalize();

        if computed_crc32 != expected_crc32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "WAL record CRC32 mismatch",
            ));
        }

        // Parse vector data
        let mut vector_data = Vec::with_capacity(vector_len);
        for i in 0..vector_len {
            let offset = i * 4;
            let f = f32::from_le_bytes([
                payload[offset],
                payload[offset + 1],
                payload[offset + 2],
                payload[offset + 3],
            ]);
            vector_data.push(f);
        }

        // Parse metadata
        let metadata = payload[vector_bytes..].to_vec();

        Ok((
            WalRecord {
                lsn,
                timestamp,
                op_type,
                vector_id,
                vector_data,
                metadata,
            },
            total_len,
        ))
    }
}

/// Message sent to WAL writer thread
#[derive(Debug)]
pub enum WalMessage {
    Record(WalRecord),
    Checkpoint {
        generation: u64,
        /// Channel to signal fsync completion back to the caller
        ack: oneshot::Sender<io::Result<()>>,
    },
    /// Full truncation after flush - clears entire WAL
    Truncate {
        ack: oneshot::Sender<io::Result<()>>,
    },
    /// Shutdown signal - drain remaining records, fsync, and stop
    Shutdown {
        ack: oneshot::Sender<io::Result<()>>,
    },
}

/// WAL file header - 40 bytes
///
/// Format:
/// [0..8]   Magic number (u64) - includes version in lower 32 bits
/// [8..16]  Last checkpoint LSN (u64)
/// [16..24] Current LSN (u64)
/// [24..32] Last checkpoint generation (u64)
/// [32..40] CRC32 of fields [0..32] (u32) + padding (u32)
#[derive(Debug)]
struct WalHeader {
    magic: u64,              // 0x5741_4C31_0000_0001 (WAL1 + version 1)
    last_checkpoint_lsn: u64,
    current_lsn: u64,
    last_checkpoint_generation: u64,  // Generation of last safe checkpoint
}

const WAL_MAGIC: u64 = 0x5741_4C31_0000_0001;
const WAL_FORMAT_VERSION: u32 = 1;
const WAL_HEADER_SIZE: usize = 40; // 5 * u64

impl WalHeader {
    fn to_bytes(&self) -> [u8; WAL_HEADER_SIZE] {
        let mut buf = [0u8; WAL_HEADER_SIZE];
        buf[0..8].copy_from_slice(&self.magic.to_le_bytes());
        buf[8..16].copy_from_slice(&self.last_checkpoint_lsn.to_le_bytes());
        buf[16..24].copy_from_slice(&self.current_lsn.to_le_bytes());
        buf[24..32].copy_from_slice(&self.last_checkpoint_generation.to_le_bytes());
        // CRC32 of first 32 bytes
        let mut hasher = Crc32Hasher::new();
        hasher.update(&buf[0..32]);
        let crc = hasher.finalize();
        buf[32..36].copy_from_slice(&crc.to_le_bytes());
        buf[36..40].copy_from_slice(&0u32.to_le_bytes()); // padding
        buf
    }

    fn from_bytes(buf: &[u8; WAL_HEADER_SIZE]) -> io::Result<Self> {
        // Verify CRC32 of first 32 bytes
        let mut hasher = Crc32Hasher::new();
        hasher.update(&buf[0..32]);
        let computed_crc = hasher.finalize();
        let stored_crc = u32::from_le_bytes([buf[32], buf[33], buf[34], buf[35]]);

        if computed_crc != stored_crc {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "WAL header CRC32 mismatch - header corrupted",
            ));
        }

        let magic = u64::from_le_bytes([buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]]);

        // Validate magic number to ensure this is actually a WAL file
        if magic != WAL_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid WAL magic number: expected {:#x}, got {:#x}", WAL_MAGIC, magic),
            ));
        }

        // Validate format version (lower 32 bits of magic contain version)
        let version = (magic & 0xFFFFFFFF) as u32;
        if version != WAL_FORMAT_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported WAL format version: expected {}, got {}", WAL_FORMAT_VERSION, version),
            ));
        }

        Ok(WalHeader {
            magic,
            last_checkpoint_lsn: u64::from_le_bytes([buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15]]),
            current_lsn: u64::from_le_bytes([buf[16], buf[17], buf[18], buf[19], buf[20], buf[21], buf[22], buf[23]]),
            last_checkpoint_generation: u64::from_le_bytes([buf[24], buf[25], buf[26], buf[27], buf[28], buf[29], buf[30], buf[31]]),
        })
    }
}

/// Write-Ahead Log - PostgreSQL style
/// 
/// Uses async channel to separate log writing from data application.
/// Main thread sends records via channel, WAL writer persists them.
/// Each checkpoint has a generation number for safe recovery.
pub struct WriteAheadLog {
    /// Next LSN to allocate (atomic, used by callers via fetch_add)
    next_lsn: Arc<AtomicU64>,
    /// Highest LSN written to disk (updated by writer after each write)
    last_written_lsn: Arc<AtomicU64>,
    /// Channel to WAL writer thread (bounded for backpressure)
    tx: mpsc::Sender<WalMessage>,
    /// Current checkpoint generation (auto-incremented)
    checkpoint_generation: Arc<AtomicU64>,
    /// WAL operation metrics
    metrics: Arc<WalMetrics>,
}

/// Internal WAL writer - runs in separate tokio task
pub struct WalWriter {
    file: File,
    /// Path to the active WAL file (.db.wal)
    wal_path: PathBuf,
    /// Base path prefix for constructing segment names (e.g., "foo.db")
    db_path_prefix: String,
    /// Highest LSN written to disk (updated after each write, shared with WriteAheadLog)
    last_written_lsn: Arc<AtomicU64>,
    last_checkpoint_lsn: Arc<AtomicU64>,
    last_checkpoint_generation: Arc<AtomicU64>,
    current_generation: u64,  // Local counter, incremented with each checkpoint
    rx: mpsc::Receiver<WalMessage>,
    /// Maximum WAL segment file size in bytes before rotation (0 = no rotation)
    max_wal_segment_size: u64,
    /// Maximum total WAL size across all segments in bytes before cleanup (0 = no limit)
    max_total_wal_size: u64,
    /// Maximum number of WAL segments before cleanup (0 = no limit)
    max_wal_segments: u32,
    /// Current WAL file size in bytes
    current_file_size: u64,
    /// Next segment ID for naming rotated segments
    next_segment_id: u64,
    /// Reusable buffer for serializing records (avoids per-record allocation)
    write_buf: Vec<u8>,
    /// WAL operation metrics
    metrics: Arc<WalMetrics>,
}

impl WriteAheadLog {
    /// Create WAL and start writer thread.
    /// Acquires an exclusive file lock to prevent concurrent access.
    pub async fn new(db_path: &Path, max_wal_segment_size: u64, max_total_wal_size: u64, max_wal_segments: u32) -> io::Result<Self> {
        let wal_path = Self::wal_path(db_path);
        let exists = wal_path.exists();

        let mut file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&wal_path)?;

        // Acquire exclusive file lock to prevent concurrent WAL access
        file.try_lock_exclusive().map_err(|e| {
            io::Error::new(
                io::ErrorKind::WouldBlock,
                format!("WAL file is locked by another process: {}", e),
            )
        })?;

        // Initialize header if new file
        if !exists {
            let header = WalHeader {
                magic: WAL_MAGIC,
                last_checkpoint_lsn: 0,
                current_lsn: 0,
                last_checkpoint_generation: 0,
        };
        file.write_all(&header.to_bytes())?;
        file.flush()?;
        }

        // Load header
        let mut header_buf = [0u8; WAL_HEADER_SIZE];
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut header_buf)?;
        let header = WalHeader::from_bytes(&header_buf)?;

        let next_lsn = Arc::new(AtomicU64::new(header.current_lsn));
        let last_written_lsn = Arc::new(AtomicU64::new(header.current_lsn));
        let metrics = Arc::new(WalMetrics::default());

        // Get current file size
        let current_file_size = file.metadata()?.len();

        // Scan for existing segments to determine next segment ID
        let db_path_prefix = Self::db_path_prefix(db_path);
        let next_segment_id = Self::find_max_segment_id(db_path);

        let (tx, rx) = mpsc::channel(1024);

        // Start WAL writer thread
        let writer = WalWriter {
            file,
            wal_path,
            db_path_prefix,
            last_written_lsn: last_written_lsn.clone(),
            last_checkpoint_lsn: Arc::new(AtomicU64::new(header.last_checkpoint_lsn)),
            last_checkpoint_generation: Arc::new(AtomicU64::new(header.last_checkpoint_generation)),
            current_generation: header.last_checkpoint_generation,
            rx,
            max_wal_segment_size,
            max_total_wal_size,
            max_wal_segments,
            current_file_size,
            next_segment_id,
            write_buf: Vec::with_capacity(4096),
            metrics: metrics.clone(),
        };

        tokio::spawn(async move {
            if let Err(e) = writer.run().await {
                tracing::error!("WAL writer error: {}", e);
            }
        });

        Ok(WriteAheadLog {
            next_lsn,
            last_written_lsn: last_written_lsn.clone(),
            tx,
            checkpoint_generation: Arc::new(AtomicU64::new(header.last_checkpoint_generation)),
            metrics,
        })
    }

    /// Allocate and return the next LSN atomically (caller-side allocation).
    /// This prevents duplicate LSNs under concurrent access.
    ///
    /// Note: LSNs are consumed via fetch_add before the record is sent to the
    /// WAL channel. If the send fails (channel full or closed), the LSN is
    /// wasted, creating a gap in the sequence. This is harmless for recovery
    /// since the replay logic handles non-sequential LSNs correctly.
    fn allocate_lsn(&self) -> Lsn {
        Lsn(self.next_lsn.fetch_add(1, Ordering::Relaxed))
    }

    /// Get WAL operation metrics for monitoring
    pub fn metrics(&self) -> &WalMetrics {
        &self.metrics
    }

    /// Get the next LSN to be allocated (in-memory, not yet on disk)
    pub fn next_lsn(&self) -> Lsn {
        Lsn(self.next_lsn.load(Ordering::Relaxed))
    }

    /// Get the highest LSN that has been written to disk
    pub fn last_written_lsn(&self) -> Lsn {
        Lsn(self.last_written_lsn.load(Ordering::Acquire))
    }

    /// Log INSERT - returns LSN when record is buffered in channel.
    /// The record is guaranteed durable only after a subsequent checkpoint/fsync.
    pub fn log_insert(&self, id: u32, vector: &[f32], metadata: &[u8]) -> io::Result<Lsn> {
        let lsn = self.allocate_lsn();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let record = WalRecord {
            lsn,
            timestamp,
            op_type: WalOpType::Insert,
            vector_id: id,
            vector_data: vector.to_vec(),
            metadata: metadata.to_vec(),
        };
        self.tx.try_send(WalMessage::Record(record)).map_err(|e| {
            match e {
                mpsc::error::TrySendError::Full(_) => io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "WAL channel full, backpressure needed",
                ),
                mpsc::error::TrySendError::Closed(_) => io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "WAL writer shut down",
                ),
            }
        })?;
        Ok(lsn)
    }

    /// Log INSERT (async) - blocks if channel is full, providing natural backpressure.
    pub async fn log_insert_async(&self, id: u32, vector: &[f32], metadata: &[u8]) -> io::Result<Lsn> {
        let lsn = self.allocate_lsn();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let record = WalRecord {
            lsn,
            timestamp,
            op_type: WalOpType::Insert,
            vector_id: id,
            vector_data: vector.to_vec(),
            metadata: metadata.to_vec(),
        };
        self.tx.send(WalMessage::Record(record))
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "WAL writer shut down"))?;
        Ok(lsn)
    }

    /// Log DELETE (non-async wrapper)
    pub fn log_delete(&self, id: u32, metadata: &[u8]) -> io::Result<Lsn> {
        let lsn = self.allocate_lsn();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let record = WalRecord {
            lsn,
            timestamp,
            op_type: WalOpType::Delete,
            vector_id: id,
            vector_data: Vec::new(),
            metadata: metadata.to_vec(),
        };
        self.tx.try_send(WalMessage::Record(record)).map_err(|e| {
            match e {
                mpsc::error::TrySendError::Full(_) => io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "WAL channel full, backpressure needed",
                ),
                mpsc::error::TrySendError::Closed(_) => io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "WAL writer shut down",
                ),
            }
        })?;
        Ok(lsn)
    }

    /// Log DELETE (async) - blocks if channel is full, providing natural backpressure.
    pub async fn log_delete_async(&self, id: u32, metadata: &[u8]) -> io::Result<Lsn> {
        let lsn = self.allocate_lsn();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let record = WalRecord {
            lsn,
            timestamp,
            op_type: WalOpType::Delete,
            vector_id: id,
            vector_data: Vec::new(),
            metadata: metadata.to_vec(),
        };
        self.tx.send(WalMessage::Record(record))
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "WAL writer shut down"))?;
        Ok(lsn)
    }

    /// Log UPDATE (non-async wrapper)
    pub fn log_update(&self, id: u32, vector: &[f32], metadata: &[u8]) -> io::Result<Lsn> {
        let lsn = self.allocate_lsn();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let record = WalRecord {
            lsn,
            timestamp,
            op_type: WalOpType::Update,
            vector_id: id,
            vector_data: vector.to_vec(),
            metadata: metadata.to_vec(),
        };
        self.tx.try_send(WalMessage::Record(record)).map_err(|e| {
            match e {
                mpsc::error::TrySendError::Full(_) => io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "WAL channel full, backpressure needed",
                ),
                mpsc::error::TrySendError::Closed(_) => io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "WAL writer shut down",
                ),
            }
        })?;
        Ok(lsn)
    }

    /// Log UPDATE (async) - blocks if channel is full, providing natural backpressure.
    pub async fn log_update_async(&self, id: u32, vector: &[f32], metadata: &[u8]) -> io::Result<Lsn> {
        let lsn = self.allocate_lsn();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let record = WalRecord {
            lsn,
            timestamp,
            op_type: WalOpType::Update,
            vector_id: id,
            vector_data: vector.to_vec(),
            metadata: metadata.to_vec(),
        };
        self.tx.send(WalMessage::Record(record))
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "WAL writer shut down"))?;
        Ok(lsn)
    }

    /// Request checkpoint and wait for fsync confirmation (async).
    /// Returns Ok(()) only after the WAL writer has fsynced the checkpoint to disk.
    /// Auto-increments the generation number.
    pub async fn checkpoint(&self) -> io::Result<()> {
        let generation = self.checkpoint_generation.fetch_add(1, Ordering::Relaxed) + 1;
        let (ack_tx, ack_rx) = oneshot::channel();
        self.tx
            .send(WalMessage::Checkpoint {
                generation,
                ack: ack_tx,
            })
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "WAL writer shut down"))?;
        // Await the WAL writer's fsync confirmation
        ack_rx
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "WAL writer dropped ack channel"))?
    }

    /// Clear WAL after checkpoint (waits for completion)
    pub async fn clear(&self) -> io::Result<()> {
        let (ack_tx, ack_rx) = oneshot::channel();
        self.tx
            .send(WalMessage::Truncate { ack: ack_tx })
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "WAL writer shut down"))?;
        ack_rx
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "WAL writer dropped ack channel"))?
    }

    /// Gracefully shutdown the WAL writer - drains pending records, fsyncs, and stops
    pub async fn shutdown(&self) -> io::Result<()> {
        let (ack_tx, ack_rx) = oneshot::channel();
        let _ = self.tx.send(WalMessage::Shutdown { ack: ack_tx }).await;
        let _ = ack_rx.await;
        Ok(())
    }

    pub fn wal_path(db_path: &Path) -> PathBuf {
        let mut s = db_path.as_os_str().to_owned();
        s.push(".wal");
        PathBuf::from(s)
    }

    /// Extract the DB path prefix from a WAL path (strip ".wal" suffix)
    fn db_path_prefix(db_path: &Path) -> String {
        db_path.to_string_lossy().to_string()
    }

    /// Find the maximum segment ID among existing .db.wal.N files
    fn find_max_segment_id(db_path: &Path) -> u64 {
        let prefix = Self::wal_segment_prefix(db_path);
        let mut max_id = 0u64;
        if let Ok(entries) = fs::read_dir(db_path.parent().unwrap_or(Path::new("."))) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if let Some(suffix) = name_str.strip_prefix(&prefix) {
                    if let Ok(id) = suffix.parse::<u64>() {
                        if id >= max_id {
                            max_id = id + 1;
                        }
                    }
                }
            }
        }
        max_id
    }

    /// Get the prefix for segment file names: "<filename>.wal."
    fn wal_segment_prefix(db_path: &Path) -> String {
        let wal_path = Self::wal_path(db_path);
        format!("{}.", wal_path.file_name().unwrap().to_string_lossy())
    }

    /// Construct path for a numbered segment file: <db_path>.wal.<id>
    fn segment_path(db_path: &Path, segment_id: u64) -> PathBuf {
        let mut s = db_path.as_os_str().to_owned();
        s.push(".wal.");
        s.push(segment_id.to_string());
        PathBuf::from(s)
    }

    /// List all segment files (numbered .db.wal.N files) sorted by ID ascending
    fn list_segment_files(db_path: &Path) -> Vec<(u64, PathBuf)> {
        let mut segments = Vec::new();
        let dir = db_path.parent().unwrap_or(Path::new("."));
        let prefix = Self::wal_segment_prefix(db_path);

        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if let Some(suffix) = name_str.strip_prefix(&prefix) {
                    if let Ok(id) = suffix.parse::<u64>() {
                        segments.push((id, entry.path()));
                    }
                }
            }
        }
        segments.sort_by_key(|(id, _)| *id);
        segments
    }

    /// Load all records for recovery
    pub async fn load_for_recovery(db_path: &Path) -> io::Result<Vec<WalRecord>> {
        let mut all_records = Vec::new();
        let mut global_checkpoint_lsn = 0u64;

        // Load from numbered segments (oldest first)
        let segments = Self::list_segment_files(db_path);
        for (_id, segment_path) in &segments {
            let records = Self::load_segment_for_recovery(segment_path, &mut global_checkpoint_lsn)?;
            all_records.extend(records);
        }

        // Load from active WAL file (last)
        let wal_path = Self::wal_path(db_path);
        if wal_path.exists() {
            let records = Self::load_segment_for_recovery(&wal_path, &mut global_checkpoint_lsn)?;
            all_records.extend(records);
        }

        // Filter to only records at or after the global checkpoint LSN.
        // Records AT the checkpoint LSN were not applied before the checkpoint
        // was written, so they must be replayed.
        let filtered: Vec<WalRecord> = all_records
            .into_iter()
            .filter(|r| r.lsn.0 >= global_checkpoint_lsn && r.op_type != WalOpType::Checkpoint)
            .collect();

        Ok(filtered)
    }

    /// Load records from a single WAL segment file, updating the global checkpoint LSN
    fn load_segment_for_recovery(
        segment_path: &Path,
        global_checkpoint_lsn: &mut u64,
    ) -> io::Result<Vec<WalRecord>> {
        let mut file = File::open(segment_path)?;

        // Read header
        let mut header_buf = [0u8; WAL_HEADER_SIZE];
        file.read_exact(&mut header_buf)?;
        let header = WalHeader::from_bytes(&header_buf)?;

        // Track the highest checkpoint LSN seen across all segments
        if header.last_checkpoint_lsn > *global_checkpoint_lsn {
            *global_checkpoint_lsn = header.last_checkpoint_lsn;
        }

        // Read all data
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        let mut records = Vec::new();
        let mut offset = 0;
        let mut last_lsn = 0u64;

        while offset < data.len() {
            match WalRecord::from_bytes(&data[offset..]) {
                Ok((record, consumed)) => {
                    // LSN ordering validation (Prod 2 fix)
                    if record.lsn.0 != 0 && record.lsn.0 <= last_lsn {
                        tracing::warn!(
                            "WAL recovery: out-of-order LSN {} <= {} at offset {} in {:?}, stopping",
                            record.lsn.0, last_lsn, offset, segment_path
                        );
                        break;
                    }
                    last_lsn = record.lsn.0;
                    records.push(record);
                    offset += consumed;
                }
                Err(_) => {
                    // Partial record at end (crash)
                    break;
                }
            }
        }

        Ok(records)
    }
}

impl Drop for WriteAheadLog {
    fn drop(&mut self) {
        // Attempt graceful shutdown: send Shutdown message to drain pending records.
        // If the channel is full or the writer is already gone, the writer task
        // will exit when the channel closes (tx dropped), but won't drain records.
        // Callers should explicitly call shutdown() before drop for guaranteed durability.
        let _ = self.tx.try_send(WalMessage::Shutdown {
            ack: oneshot::channel().0,
        });
        // When `self` is dropped, `tx` is dropped, closing the channel.
        // The writer task's `rx.recv()` will return None, causing it to exit.
    }
}

impl WalWriter {
    /// Main writer loop - runs in tokio task
    async fn run(mut self) -> io::Result<()> {
        while let Some(msg) = self.rx.recv().await {
            match msg {
                WalMessage::Record(record) => {
                    self.write_record_to_file(&record)?;
                }
                WalMessage::Checkpoint { generation, ack } => {
                    let result = self.handle_checkpoint(generation);
                    let _ = ack.send(result);
                }
                WalMessage::Truncate { ack } => {
                    let result = self.handle_truncate();
                    let _ = ack.send(result);
                }
                WalMessage::Shutdown { ack } => {
                    // Drain remaining records from channel
                    self.drain_remaining()?;
                    // Final fsync before shutdown
                    let _ = self.file.flush();
                    let _ = self.file.sync_all();
                    let _ = ack.send(Ok(()));
                    return Ok(());
                }
            }
        }

        // Channel closed (tx dropped) without explicit Shutdown.
        // Drain any remaining records and fsync before exiting.
        self.drain_remaining()?;
        let _ = self.file.flush();
        let _ = self.file.sync_all();
        Ok(())
    }

    /// Write a single WAL record to the file using the reusable buffer
    fn write_record_to_file(&mut self, record: &WalRecord) -> io::Result<()> {
        self.write_buf.clear();
        record.write_to(&mut self.write_buf);
        let written = self.write_buf.len() as u64;
        self.file.write_all(&self.write_buf)?;
        self.current_file_size += written;
        let next_lsn = record.lsn.0 + 1;
        self.last_written_lsn.fetch_max(next_lsn, Ordering::Release);

        // Update metrics
        self.metrics.records_written.fetch_add(1, Ordering::Relaxed);
        self.metrics.bytes_written.fetch_add(written, Ordering::Relaxed);

        // Rotate to new segment if WAL exceeds max segment size
        if self.max_wal_segment_size > 0
            && self.current_file_size > self.max_wal_segment_size
        {
            self.rotate_to_new_segment()?;
        }
        Ok(())
    }

    /// Drain all remaining records from the channel without blocking
    fn drain_remaining(&mut self) -> io::Result<()> {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                WalMessage::Record(record) => {
                    self.write_record_to_file(&record)?;
                }
                WalMessage::Checkpoint { generation, ack: ck_ack } => {
                    let _ = ck_ack.send(self.handle_checkpoint(generation));
                }
                WalMessage::Truncate { ack: t_ack } => {
                    let _ = t_ack.send(self.handle_truncate());
                }
                WalMessage::Shutdown { .. } => {}
            }
        }
        Ok(())
    }

    /// Handle truncation: delete segments, reset WAL to header-only
    fn handle_truncate(&mut self) -> io::Result<()> {
        self.delete_all_segments()?;

        // Reset active WAL to header-only
        self.file.set_len(WAL_HEADER_SIZE as u64)?;
        self.file.seek(SeekFrom::Start(0))?;

        // Preserve the current LSN in the header so that recovery after a crash
        // between truncation and the next checkpoint will correctly identify records.
        let current_lsn = self.last_written_lsn.load(Ordering::Acquire);
        let header = WalHeader {
            magic: WAL_MAGIC,
            last_checkpoint_lsn: current_lsn,
            current_lsn,
            last_checkpoint_generation: self.last_checkpoint_generation.load(Ordering::Relaxed),
        };
        self.file.write_all(&header.to_bytes())?;
        self.file.flush()?;
        self.file.sync_all()?;
        self.current_file_size = WAL_HEADER_SIZE as u64;

        self.last_checkpoint_lsn.store(current_lsn, Ordering::Release);

        Ok(())
    }

    /// Handle checkpoint: write record + update header atomically, then fsync.
    ///
    /// To ensure atomicity, we:
    /// 1. Write the checkpoint record
    /// 2. Seek back to position 0 and write the updated header
    /// 3. fsync once (covers both writes)
    ///
    /// This ensures that if a crash occurs, either:
    /// - Both the checkpoint record and header are on disk (correct state), OR
    /// - Neither is on disk (old state, recovery replays from previous checkpoint)
    fn handle_checkpoint(&mut self, generation: u64) -> io::Result<()> {
        // Load the latest LSN from the atomic (which includes records
        // already written by the writer). The atomic is updated via fetch_max
        // after each record write, so it reflects the highest LSN + 1 that has
        // been written to disk.
        let lsn = Lsn(self.last_written_lsn.load(Ordering::Acquire));

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let checkpoint = WalRecord {
            lsn,
            timestamp,
            op_type: WalOpType::Checkpoint,
            vector_id: 0,
            vector_data: Vec::new(),
            metadata: Vec::new(),
        };
        let bytes = checkpoint.to_bytes();
        let written = bytes.len() as u64;
        self.file.write_all(&bytes)?;
        self.current_file_size += written;

        // Update local generation
        self.current_generation = generation;

        // Prepare updated header
        let header = WalHeader {
            magic: WAL_MAGIC,
            last_checkpoint_lsn: lsn.0,
            current_lsn: lsn.0 + 1,
            last_checkpoint_generation: generation,
        };

        // Seek back to header position and write updated header
        // This must happen before fsync to ensure header and checkpoint are atomic
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&header.to_bytes())?;

        // Seek back to end so subsequent writes append correctly
        self.file.seek(SeekFrom::Start(self.current_file_size))?;

        // Single fsync covers both the checkpoint record and header update
        self.file.flush()?;
        self.file.sync_all()?;

        // Update shared atomics
        self.last_checkpoint_lsn.store(lsn.0, Ordering::Release);
        self.last_checkpoint_generation.store(generation, Ordering::Relaxed);

        // Update metrics
        self.metrics.checkpoints.fetch_add(1, Ordering::Relaxed);

        // Cleanup old segments if total size exceeds limit
        if self.max_total_wal_size > 0 {
            self.cleanup_old_segments()?;
        }

        Ok(())
    }

    /// Rotate current WAL to a numbered segment and create a fresh active WAL
    fn rotate_to_new_segment(&mut self) -> io::Result<()> {
        // Flush and sync current file before rotation
        self.file.flush()?;
        self.file.sync_all()?;

        // Explicitly drop the file handle to release it before rename.
        // Replace with a temporary file in the same directory to ensure
        // the rename target and source are on the same filesystem.
        let old_path = self.wal_path.clone();
        let tmp_path = self.wal_path.with_extension("wal.tmp");
        let tmp_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&tmp_path)?;
        drop(std::mem::replace(&mut self.file, tmp_file));

        // Rename current WAL to numbered segment
        let segment_path = WriteAheadLog::segment_path(
            Path::new(&self.db_path_prefix),
            self.next_segment_id,
        );
        fs::rename(&old_path, &segment_path)?;
        // Clean up the temporary file handle's backing file
        let _ = fs::remove_file(&tmp_path);
        self.next_segment_id += 1;

        // Create new active WAL file with fresh header
        let mut new_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&self.wal_path)?;

        // Preserve the current checkpoint LSN and generation from the previous segment.
        // This ensures recovery can correctly identify which records have been checkpointed
        // if a crash occurs immediately after rotation.
        let header = WalHeader {
            magic: WAL_MAGIC,
            last_checkpoint_lsn: self.last_checkpoint_lsn.load(Ordering::Acquire),
            current_lsn: self.last_written_lsn.load(Ordering::Acquire),
            last_checkpoint_generation: self.last_checkpoint_generation.load(Ordering::Relaxed),
        };
        new_file.write_all(&header.to_bytes())?;
        new_file.flush()?;
        new_file.sync_all()?;

        self.file = new_file;
        self.current_file_size = WAL_HEADER_SIZE as u64;

        // Update metrics
        self.metrics.segment_rotations.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Delete old WAL segments when total size or segment count exceeds limit
    /// Only deletes segments that have been checkpointed (safe to discard)
    fn cleanup_old_segments(&mut self) -> io::Result<()> {
        let db_path = Path::new(&self.db_path_prefix);
        let segments = WriteAheadLog::list_segment_files(db_path);

        if segments.is_empty() {
            return Ok(());
        }

        // Calculate total size including active WAL
        let active_size = self.current_file_size;
        let mut total_size = active_size;
        let mut segment_sizes: Vec<(u64, u64, PathBuf)> = Vec::new(); // (id, size, path)

        for (id, path) in &segments {
            if let Ok(metadata) = fs::metadata(path) {
                let size = metadata.len();
                total_size += size;
                segment_sizes.push((*id, size, path.clone()));
            }
        }

        let segment_count = segments.len();

        // Delete oldest checkpointed segments until under both limits
        for (_id, size, path) in &segment_sizes {
            let over_size_limit = self.max_total_wal_size > 0 && total_size > self.max_total_wal_size;
            let over_count_limit = self.max_wal_segments > 0 && segment_count as u32 > self.max_wal_segments;
            if !over_size_limit && !over_count_limit {
                break;
            }

            // Read segment header to check if it has a checkpoint
            if let Ok(mut file) = File::open(path) {
                let mut header_buf = [0u8; WAL_HEADER_SIZE];
                if file.read_exact(&mut header_buf).is_ok() {
                    if let Ok(header) = WalHeader::from_bytes(&header_buf) {
                        // Only delete if segment has been checkpointed
                        if header.last_checkpoint_lsn > 0 {
                            fs::remove_file(path)?;
                            total_size -= size;
                            self.metrics.segments_cleaned.fetch_add(1, Ordering::Relaxed);
                            tracing::debug!("Cleaned up WAL segment {:?} ({} bytes)", path, size);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Delete all numbered WAL segments
    fn delete_all_segments(&self) -> io::Result<()> {
        let db_path = Path::new(&self.db_path_prefix);
        let segments = WriteAheadLog::list_segment_files(db_path);

        for (_id, path) in segments {
            fs::remove_file(&path)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wal_record_roundtrip() {
        let record = WalRecord {
            lsn: Lsn(1),
            timestamp: 1234567890,
            op_type: WalOpType::Insert,
            vector_id: 42,
            vector_data: vec![0.1, 0.2, 0.3],
            metadata: b"test_meta".to_vec(),
        };

        let bytes = record.to_bytes();
        let (decoded, consumed) = WalRecord::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.lsn, Lsn(1));
        assert_eq!(decoded.timestamp, 1234567890);
        assert_eq!(decoded.op_type, WalOpType::Insert);
        assert_eq!(decoded.vector_id, 42);
        assert_eq!(decoded.vector_data, vec![0.1, 0.2, 0.3]);
        assert_eq!(decoded.metadata, b"test_meta".to_vec());
        assert_eq!(consumed, bytes.len());
    }

    #[test]
    fn test_wal_crc32_validation() {
        let record = WalRecord {
            lsn: Lsn(1),
            timestamp: 1234567890,
            op_type: WalOpType::Insert,
            vector_id: 1,
            vector_data: vec![0.5, 0.6],
            metadata: b"data".to_vec(),
        };

        let mut bytes = record.to_bytes();

        // Corrupt the payload
        bytes[40] ^= 0xFF;

        let result = WalRecord::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_wal_delete_operation() {
        let record = WalRecord {
            lsn: Lsn(5),
            timestamp: 1234567890,
            op_type: WalOpType::Delete,
            vector_id: 99,
            vector_data: Vec::new(),
            metadata: b"old_meta".to_vec(),
        };

        let bytes = record.to_bytes();
        let (decoded, _) = WalRecord::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.op_type, WalOpType::Delete);
        assert_eq!(decoded.vector_id, 99);
        assert_eq!(decoded.metadata, b"old_meta".to_vec());
    }

    #[test]
    fn test_wal_header_crc32_roundtrip() {
        let header = WalHeader {
            magic: WAL_MAGIC,
            last_checkpoint_lsn: 100,
            current_lsn: 200,
            last_checkpoint_generation: 5,

        };

        let bytes = header.to_bytes();
        let decoded = WalHeader::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.magic, WAL_MAGIC);
        assert_eq!(decoded.last_checkpoint_lsn, 100);
        assert_eq!(decoded.current_lsn, 200);
        assert_eq!(decoded.last_checkpoint_generation, 5);
    }

    #[test]
    fn test_wal_header_crc32_corruption_detected() {
        let header = WalHeader {
            magic: WAL_MAGIC,
            last_checkpoint_lsn: 100,
            current_lsn: 200,
            last_checkpoint_generation: 5,

        };

        let mut bytes = header.to_bytes();
        // Corrupt a field in the first 32 bytes
        bytes[10] ^= 0xFF;

        let result = WalHeader::from_bytes(&bytes);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("CRC32 mismatch"));
    }
}
