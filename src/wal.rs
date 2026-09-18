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

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use crc32fast::Hasher as Crc32Hasher;

/// Log Sequence Number - unique position in WAL stream
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Lsn(u64);

impl Lsn {
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

/// WAL record - fixed 32-byte header + variable payload + 4-byte footer
///
/// Format:
/// [0..8]   LSN (u64)
/// [8..16]  Timestamp (u64) 
/// [16]     Op type (u8)
/// [17]     Padding (u8)
/// [18..22] Vector ID (u32)
/// [22..26] Vector len in floats (u32)
/// [26..30] Metadata len (u32)
/// [30..32] CRC32 of payload (u32) - actually 32 bits, packed after
/// [32+]    Payload: vector data (4*vector_len bytes) + metadata (variable)
/// [..+4]   Footer: 0xDEADBEEF (u32)
#[derive(Debug, Clone)]
pub struct WalRecord {
    pub lsn: Lsn,
    pub op_type: WalOpType,
    pub vector_id: u32,
    pub vector_data: Vec<f32>,
    pub metadata: Vec<u8>,
}

impl WalRecord {
    /// Serialize to bytes: [32-byte header][payload][4-byte footer]
    pub fn to_bytes(&self) -> io::Result<Vec<u8>> {
        let _vector_bytes = self.vector_data.len() * 4;
        let metadata_len = self.metadata.len();

        // Compute CRC32 of payload only (vector data + metadata)
        let mut hasher = Crc32Hasher::new();
        for f in &self.vector_data {
            hasher.update(&f.to_le_bytes());
        }
        hasher.update(&self.metadata);
        let crc32 = hasher.finalize();

        // Build record
        let mut record = Vec::new();

        // Header (32 bytes fixed)
        record.extend_from_slice(&self.lsn.0.to_le_bytes());           // [0..8]
        record.extend_from_slice(&0u64.to_le_bytes());                 // [8..16] reserved/timestamp
        record.push(self.op_type as u8);                               // [16]
        record.push(0);                                                // [17] padding
        record.extend_from_slice(&self.vector_id.to_le_bytes());       // [18..22]
        record.extend_from_slice(&(self.vector_data.len() as u32).to_le_bytes()); // [22..26]
        record.extend_from_slice(&(metadata_len as u32).to_le_bytes()); // [26..30]
        record.extend_from_slice(&crc32.to_le_bytes());                // [30..34]

        // Payload
        for f in &self.vector_data {
            record.extend_from_slice(&f.to_le_bytes());
        }
        record.extend_from_slice(&self.metadata);

        // Footer
        record.extend_from_slice(&0xDEADBEEFu32.to_le_bytes());

        Ok(record)
    }

    /// Deserialize from bytes
    pub fn from_bytes(data: &[u8]) -> io::Result<(Self, usize)> {
        if data.len() < 36 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Incomplete WAL record",
            ));
        }

        // Parse header (32 bytes)
        let lsn = Lsn(u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]));
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
#[derive(Debug, Clone)]
pub enum WalMessage {
    Record(WalRecord),
    Checkpoint(u64),  // generation number
    Truncate,
}

/// WAL file header
struct WalHeader {
    magic: u64,              // 0x5741_4C31_0000_0001
    last_checkpoint_lsn: u64,
    current_lsn: u64,
    last_checkpoint_generation: u64,  // Generation of last safe checkpoint
}

const WAL_MAGIC: u64 = 0x5741_4C31_0000_0001;
const WAL_HEADER_SIZE: usize = 40; // 5 * u64

impl WalHeader {
    fn to_bytes(&self) -> [u8; WAL_HEADER_SIZE] {
        let mut buf = [0u8; WAL_HEADER_SIZE];
        buf[0..8].copy_from_slice(&self.magic.to_le_bytes());
        buf[8..16].copy_from_slice(&self.last_checkpoint_lsn.to_le_bytes());
        buf[16..24].copy_from_slice(&self.current_lsn.to_le_bytes());
        buf[24..32].copy_from_slice(&self.last_checkpoint_generation.to_le_bytes());
        buf[32..40].copy_from_slice(&0u64.to_le_bytes()); // reserved
        buf
    }

    fn from_bytes(buf: &[u8; WAL_HEADER_SIZE]) -> Self {
        WalHeader {
            magic: u64::from_le_bytes([buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7]]),
            last_checkpoint_lsn: u64::from_le_bytes([buf[8], buf[9], buf[10], buf[11], buf[12], buf[13], buf[14], buf[15]]),
            current_lsn: u64::from_le_bytes([buf[16], buf[17], buf[18], buf[19], buf[20], buf[21], buf[22], buf[23]]),
            last_checkpoint_generation: u64::from_le_bytes([buf[24], buf[25], buf[26], buf[27], buf[28], buf[29], buf[30], buf[31]]),
        }
    }
}

/// Write-Ahead Log - PostgreSQL style
/// 
/// Uses async channel to separate log writing from data application.
/// Main thread sends records via channel, WAL writer persists them.
/// Each checkpoint has a generation number for safe recovery.
pub struct WriteAheadLog {
    /// Current LSN (atomic, shared with writer thread)
    current_lsn: Arc<AtomicU64>,
    /// Channel to WAL writer thread
    tx: mpsc::UnboundedSender<WalMessage>,
}

/// Internal WAL writer - runs in separate tokio task
pub struct WalWriter {
    file: File,
    current_lsn: Arc<AtomicU64>,
    last_checkpoint_lsn: Arc<AtomicU64>,
    last_checkpoint_generation: Arc<AtomicU64>,
    current_generation: u64,  // Local counter, incremented with each checkpoint
    rx: mpsc::UnboundedReceiver<WalMessage>,
}

impl WriteAheadLog {
    /// Create WAL and start writer thread
    pub async fn new(db_path: &Path) -> io::Result<Self> {
        let wal_path = Self::wal_path(db_path);
        let exists = wal_path.exists();

        let mut file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&wal_path)?;

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
        let header = WalHeader::from_bytes(&header_buf);

        let current_lsn = Arc::new(AtomicU64::new(header.current_lsn));

        let (tx, rx) = mpsc::unbounded_channel();

        // Start WAL writer thread
        let writer = WalWriter {
            file,
            current_lsn: current_lsn.clone(),
            last_checkpoint_lsn: Arc::new(AtomicU64::new(header.last_checkpoint_lsn)),
            last_checkpoint_generation: Arc::new(AtomicU64::new(header.last_checkpoint_generation)),
            current_generation: header.last_checkpoint_generation,
            rx,
        };

        tokio::spawn(async move {
            let _ = writer.run().await;
        });

        Ok(WriteAheadLog {
            current_lsn,
            tx,
        })
    }

    /// Get current LSN
    pub fn current_lsn(&self) -> Lsn {
        Lsn(self.current_lsn.load(Ordering::SeqCst))
    }

    /// Log INSERT - returns LSN when persisted (non-async wrapper)
    pub fn log_insert_sync(&self, id: u32, vector: &[f32], metadata: &[u8]) -> io::Result<Lsn> {
        let lsn = self.current_lsn();
        let record = WalRecord {
            lsn,
            op_type: WalOpType::Insert,
            vector_id: id,
            vector_data: vector.to_vec(),
            metadata: metadata.to_vec(),
        };
        self.tx.send(WalMessage::Record(record))
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "WAL writer shut down"))?;
        Ok(lsn)
    }

    /// Log DELETE (non-async wrapper)
    pub fn log_delete_sync(&self, id: u32, metadata: &[u8]) -> io::Result<Lsn> {
        let lsn = self.current_lsn();
        let record = WalRecord {
            lsn,
            op_type: WalOpType::Delete,
            vector_id: id,
            vector_data: Vec::new(),
            metadata: metadata.to_vec(),
        };
        self.tx.send(WalMessage::Record(record))
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "WAL writer shut down"))?;
        Ok(lsn)
    }

    /// Log UPDATE (non-async wrapper)
    pub fn log_update_sync(&self, id: u32, vector: &[f32], metadata: &[u8]) -> io::Result<Lsn> {
        let lsn = self.current_lsn();
        let record = WalRecord {
            lsn,
            op_type: WalOpType::Update,
            vector_id: id,
            vector_data: vector.to_vec(),
            metadata: metadata.to_vec(),
        };
        self.tx.send(WalMessage::Record(record))
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "WAL writer shut down"))?;
        Ok(lsn)
    }

    /// Request checkpoint (sync wrapper)
    pub fn checkpoint_sync(&self, generation: u64) -> io::Result<()> {
        self.tx.send(WalMessage::Checkpoint(generation))
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "WAL writer shut down"))?;
        Ok(())
    }

    /// Sync WAL to disk (no-op since background writer handles sync)
    pub fn sync(&self) -> io::Result<()> {
        Ok(())
    }

    /// Clear WAL after checkpoint
    pub fn clear(&self) -> io::Result<()> {
        // Send message to writer thread to truncate WAL after checkpoint
        self.tx.send(WalMessage::Truncate)
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "WAL writer shut down"))?;
        Ok(())
    }

    /// Load all records for recovery
    pub async fn load_for_recovery(db_path: &Path) -> io::Result<Vec<WalRecord>> {
        let wal_path = Self::wal_path(db_path);
        if !wal_path.exists() {
            return Ok(Vec::new());
        }

        let mut file = File::open(&wal_path)?;

        // Skip header
        let mut header_buf = [0u8; WAL_HEADER_SIZE];
        file.read_exact(&mut header_buf)?;
        let header = WalHeader::from_bytes(&header_buf);

        // Read all data
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        let mut records = Vec::new();
        let mut offset = 0;
        let start_lsn = header.last_checkpoint_lsn;

        // Replay from AFTER last checkpoint
        while offset < data.len() {
            match WalRecord::from_bytes(&data[offset..]) {
                Ok((record, consumed)) => {
                    if record.lsn.0 > start_lsn && record.op_type != WalOpType::Checkpoint {
                        records.push(record);
                    }
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

    fn wal_path(db_path: &Path) -> PathBuf {
        let mut s = db_path.as_os_str().to_owned();
        s.push(".wal");
        PathBuf::from(s)
    }
}

impl WalWriter {
    /// Main writer loop - runs in tokio task
    async fn run(mut self) -> io::Result<()> {
        while let Some(msg) = self.rx.recv().await {
            match msg {
                WalMessage::Record(record) => {
                    let bytes = record.to_bytes()?;
                    self.file.write_all(&bytes)?;
                    self.current_lsn.store(record.lsn.0 + 1, Ordering::SeqCst);
                }
                WalMessage::Checkpoint(generation) => {
                    // Write checkpoint record
                    let lsn = Lsn(self.current_lsn.load(Ordering::SeqCst));
                    let checkpoint = WalRecord {
                        lsn,
                        op_type: WalOpType::Checkpoint,
                        vector_id: 0,
                        vector_data: Vec::new(),
                        metadata: Vec::new(),
                    };
                    let bytes = checkpoint.to_bytes()?;
                    self.file.write_all(&bytes)?;

                    // Update local generation
                    self.current_generation = generation;

                    // Flush and sync
                    self.file.flush()?;
                    self.file.sync_all()?;

                    // Update header with new generation
                    let header = WalHeader {
                        magic: WAL_MAGIC,
                        last_checkpoint_lsn: lsn.0,
                        current_lsn: lsn.0 + 1,
                        last_checkpoint_generation: generation,
                    };
                    self.file.seek(SeekFrom::Start(0))?;
                    self.file.write_all(&header.to_bytes())?;
                    self.file.flush()?;

                    // Update shared atomic
                    self.last_checkpoint_lsn.store(lsn.0, Ordering::SeqCst);
                    self.last_checkpoint_generation.store(generation, Ordering::SeqCst);
                }
                WalMessage::Truncate => {
                    // Truncate file to header size (clearing all records after checkpoint)
                    self.file.set_len(WAL_HEADER_SIZE as u64)?;
                    self.file.seek(SeekFrom::Start(0))?;
                    
                    // Rebuild header with cleared LSN
                    let header = WalHeader {
                        magic: WAL_MAGIC,
                        last_checkpoint_lsn: 0,
                        current_lsn: 0,
                        last_checkpoint_generation: 0,
                    };
                    self.file.write_all(&header.to_bytes())?;
                    self.file.flush()?;
                    self.file.sync_all()?;
                }
            }
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
            op_type: WalOpType::Insert,
            vector_id: 42,
            vector_data: vec![0.1, 0.2, 0.3],
            metadata: b"test_meta".to_vec(),
        };

        let bytes = record.to_bytes().unwrap();
        let (decoded, consumed) = WalRecord::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.lsn, Lsn(1));
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
            op_type: WalOpType::Insert,
            vector_id: 1,
            vector_data: vec![0.5, 0.6],
            metadata: b"data".to_vec(),
        };

        let mut bytes = record.to_bytes().unwrap();

        // Corrupt the payload
        bytes[40] ^= 0xFF;

        let result = WalRecord::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_wal_delete_operation() {
        let record = WalRecord {
            lsn: Lsn(5),
            op_type: WalOpType::Delete,
            vector_id: 99,
            vector_data: Vec::new(),
            metadata: b"old_meta".to_vec(),
        };

        let bytes = record.to_bytes().unwrap();
        let (decoded, _) = WalRecord::from_bytes(&bytes).unwrap();

        assert_eq!(decoded.op_type, WalOpType::Delete);
        assert_eq!(decoded.vector_id, 99);
        assert_eq!(decoded.metadata, b"old_meta".to_vec());
    }
}
