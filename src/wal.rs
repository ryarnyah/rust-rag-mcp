//! Write-Ahead Logging (WAL) - PostgreSQL style
//!
//! LOG-BEFORE-APPLY: record sent to WAL before data applied to files.
//! On crash, replay WAL from last checkpoint to recover.

use fs2::FileExt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::oneshot;

// ---------------------------------------------------------------------------
// LSN
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Lsn(pub u64);

impl Lsn {
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Op types
// ---------------------------------------------------------------------------

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
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid WAL op type",
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Record format
//
// [0..8]   LSN (u64 LE)
// [8..16]  Timestamp ms (u64 LE)
// [16]     Op type (u8)
// [17]     Padding
// [18..22] Vector ID (u32 LE)
// [22..26] Vector len (u32 LE)
// [26..30] Metadata len (u32 LE)
// [30..34] CRC32 of payload (u32 LE)
// [34..]   Payload: vector_f32s + metadata
// [..+4]   Footer: 0xDEADBEEF
// ---------------------------------------------------------------------------

const RECORD_HEADER_LEN: usize = 34;
const FOOTER_LEN: usize = 4;
const FOOTER_MAGIC: u32 = 0xDEADBEEF;

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
    /// Serialize record to bytes (for benchmarking)
    #[doc(hidden)]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.write_to(&mut buf);
        buf
    }

    fn write_to(&self, buf: &mut Vec<u8>) {
        let payload_len = self.vector_data.len() * 4 + self.metadata.len();
        buf.clear();
        buf.reserve(RECORD_HEADER_LEN + payload_len + FOOTER_LEN);

        buf.extend_from_slice(&self.lsn.0.to_le_bytes());
        buf.extend_from_slice(&self.timestamp.to_le_bytes());
        buf.push(self.op_type as u8);
        buf.push(0); // padding
        buf.extend_from_slice(&self.vector_id.to_le_bytes());
        buf.extend_from_slice(&(self.vector_data.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(self.metadata.len() as u32).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // CRC placeholder

        let payload_start = buf.len();
        for f in &self.vector_data {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        buf.extend_from_slice(&self.metadata);

        let crc = crc32fast::hash(&buf[payload_start..buf.len()]);
        buf[30..34].copy_from_slice(&crc.to_le_bytes());
        buf.extend_from_slice(&FOOTER_MAGIC.to_le_bytes());
    }

    fn from_bytes(data: &[u8]) -> io::Result<(Self, usize)> {
        if data.len() < RECORD_HEADER_LEN + FOOTER_LEN {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Incomplete record",
            ));
        }

        let lsn = Lsn(u64::from_le_bytes(data[0..8].try_into().unwrap()));
        let timestamp = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let op_type = WalOpType::from_u8(data[16])?;
        let vector_id = u32::from_le_bytes(data[18..22].try_into().unwrap());
        let vector_len = u32::from_le_bytes(data[22..26].try_into().unwrap()) as usize;
        let metadata_len = u32::from_le_bytes(data[26..30].try_into().unwrap()) as usize;
        let expected_crc = u32::from_le_bytes(data[30..34].try_into().unwrap());

        let vector_bytes = vector_len.checked_mul(4).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Vector length overflow in WAL record",
            )
        })?;
        let payload_len = vector_bytes.checked_add(metadata_len).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Payload length overflow in WAL record",
            )
        })?;
        let total = RECORD_HEADER_LEN + payload_len + FOOTER_LEN;

        if data.len() < total {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Incomplete payload",
            ));
        }

        let footer = u32::from_le_bytes(data[total - 4..total].try_into().unwrap());
        if footer != FOOTER_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad footer"));
        }

        let payload = &data[RECORD_HEADER_LEN..RECORD_HEADER_LEN + payload_len];
        if crc32fast::hash(payload) != expected_crc {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "CRC mismatch"));
        }

        let vector_data = (0..vector_len)
            .map(|i| f32::from_le_bytes(payload[i * 4..(i + 1) * 4].try_into().unwrap()))
            .collect();
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
            total,
        ))
    }
}

// ---------------------------------------------------------------------------
// Messages to writer
// ---------------------------------------------------------------------------

/// Acknowledgement channel for a single record write.
///
/// The synchronous log API blocks on a std channel; the async API awaits a
/// tokio oneshot. Both are signalled by the writer thread once the record has
/// actually been written to the WAL file, which is what upholds the
/// LOG-BEFORE-APPLY guarantee.
enum RecordAck {
    Blocking(std_mpsc::Sender<io::Result<Lsn>>),
    Async(oneshot::Sender<io::Result<Lsn>>),
}

impl RecordAck {
    fn send(self, result: io::Result<Lsn>) {
        match self {
            RecordAck::Blocking(s) => {
                let _ = s.send(result);
            }
            RecordAck::Async(s) => {
                let _ = s.send(result);
            }
        }
    }
}

enum WalMessage {
    Record(WalRecord, RecordAck),
    Checkpoint {
        generation: u64,
        ack: oneshot::Sender<io::Result<()>>,
    },
    Truncate {
        ack: oneshot::Sender<io::Result<()>>,
    },
    Shutdown {
        ack: oneshot::Sender<io::Result<()>>,
    },
}

// ---------------------------------------------------------------------------
// WAL header (40 bytes) - stores checkpoint state for recovery
// ---------------------------------------------------------------------------

const WAL_MAGIC: u64 = 0x5741_4C31_0000_0001;
const WAL_HEADER_SIZE: usize = 40;

#[derive(Debug)]
struct WalHeader {
    last_checkpoint_lsn: u64,
    current_lsn: u64,
    last_checkpoint_generation: u64,
}

impl WalHeader {
    fn to_bytes(&self) -> [u8; WAL_HEADER_SIZE] {
        let mut buf = [0u8; WAL_HEADER_SIZE];
        buf[0..8].copy_from_slice(&WAL_MAGIC.to_le_bytes());
        buf[8..16].copy_from_slice(&self.last_checkpoint_lsn.to_le_bytes());
        buf[16..24].copy_from_slice(&self.current_lsn.to_le_bytes());
        buf[24..32].copy_from_slice(&self.last_checkpoint_generation.to_le_bytes());
        let crc = crc32fast::hash(&buf[0..32]);
        buf[32..36].copy_from_slice(&crc.to_le_bytes());
        buf
    }

    fn from_bytes(buf: &[u8; WAL_HEADER_SIZE]) -> io::Result<Self> {
        let crc = crc32fast::hash(&buf[0..32]);
        if crc != u32::from_le_bytes(buf[32..36].try_into().unwrap()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Header CRC mismatch",
            ));
        }
        let magic = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        if magic != WAL_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Bad magic"));
        }
        Ok(WalHeader {
            last_checkpoint_lsn: u64::from_le_bytes(buf[8..16].try_into().unwrap()),
            current_lsn: u64::from_le_bytes(buf[16..24].try_into().unwrap()),
            last_checkpoint_generation: u64::from_le_bytes(buf[24..32].try_into().unwrap()),
        })
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub struct WriteAheadLog {
    next_lsn: Arc<AtomicU64>,
    last_written_lsn: Arc<AtomicU64>,
    tx: std_mpsc::Sender<WalMessage>,
    checkpoint_generation: Arc<AtomicU64>,
    writer_thread: Mutex<Option<thread::JoinHandle<()>>>,
}

impl WriteAheadLog {
    pub async fn new(
        db_path: &Path,
        max_wal_segment_size: u64,
        max_total_wal_size: u64,
        max_wal_segments: u32,
    ) -> io::Result<Self> {
        let wal_path = Self::wal_path(db_path);
        let exists = wal_path.exists();

        let mut file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&wal_path)?;
        file.try_lock_exclusive()
            .map_err(|e| io::Error::new(io::ErrorKind::WouldBlock, format!("WAL locked: {}", e)))?;

        if !exists {
            let header = WalHeader {
                last_checkpoint_lsn: 0,
                current_lsn: 0,
                last_checkpoint_generation: 0,
            };
            file.write_all(&header.to_bytes())?;
            file.flush()?;
        }

        let mut hdr = [0u8; WAL_HEADER_SIZE];
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut hdr)?;
        let header = WalHeader::from_bytes(&hdr)?;

        let next_lsn = Arc::new(AtomicU64::new(header.current_lsn));
        let last_written_lsn = Arc::new(AtomicU64::new(header.current_lsn));
        let current_file_size = file.metadata()?.len();
        let next_segment_id = Self::find_max_segment_id(db_path);
        let db_path_prefix = db_path.to_string_lossy().to_string();

        let (tx, rx) = std_mpsc::channel();

        let writer = WalWriter {
            file,
            wal_path,
            db_path_prefix,
            last_written_lsn: last_written_lsn.clone(),
            next_lsn: next_lsn.clone(),
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
        };

        // Run the writer on a dedicated OS thread. This lets synchronous log
        // calls block on an acknowledgement without stalling (or dead-locking)
        // the tokio runtime, and lets Drop join the thread to guarantee a flush.
        let writer_thread = thread::spawn(move || writer.run());

        Ok(WriteAheadLog {
            next_lsn,
            last_written_lsn,
            tx,
            checkpoint_generation: Arc::new(AtomicU64::new(header.last_checkpoint_generation)),
            writer_thread: Mutex::new(Some(writer_thread)),
        })
    }

    pub fn last_written_lsn(&self) -> Lsn {
        Lsn(self.last_written_lsn.load(Ordering::Acquire))
    }

    // -- Log methods: sync (try_send) and async (send) ----------------------

    fn make_record(&self, op: WalOpType, id: u32, vector: &[f32], metadata: &[u8]) -> WalRecord {
        let lsn = Lsn(self.next_lsn.fetch_add(1, Ordering::Relaxed));
        WalRecord {
            lsn,
            timestamp: now_ms(),
            op_type: op,
            vector_id: id,
            vector_data: vector.to_vec(),
            metadata: metadata.to_vec(),
        }
    }

    fn send_record(&self, record: WalRecord, ack: RecordAck) -> io::Result<()> {
        self.tx
            .send(WalMessage::Record(record, ack))
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "WAL writer shut down"))
    }

    /// Queue a record and block until the writer thread has written it to the
    /// WAL file. This upholds LOG-BEFORE-APPLY: the caller only mutates the data
    /// files once the log entry is on disk (in the OS page cache), so a process
    /// crash cannot lose the log entry while keeping the applied change.
    fn log_blocking(&self, record: WalRecord) -> io::Result<Lsn> {
        let (ack_tx, ack_rx) = std_mpsc::channel();
        self.send_record(record, RecordAck::Blocking(ack_tx))?;
        match ack_rx.recv() {
            Ok(inner) => inner,
            Err(_) => Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "WAL writer shut down",
            )),
        }
    }

    /// Queue a record and await the writer thread's acknowledgement.
    async fn log_async(&self, record: WalRecord) -> io::Result<Lsn> {
        let (ack_tx, ack_rx) = oneshot::channel();
        self.send_record(record, RecordAck::Async(ack_tx))?;
        match ack_rx.await {
            Ok(inner) => inner,
            Err(_) => Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "WAL writer shut down",
            )),
        }
    }

    // Sync variants (used by VectorDb which is not async)
    pub fn log_insert(&self, id: u32, vector: &[f32], metadata: &[u8]) -> io::Result<Lsn> {
        self.log_blocking(self.make_record(WalOpType::Insert, id, vector, metadata))
    }
    pub fn log_delete(&self, id: u32, metadata: &[u8]) -> io::Result<Lsn> {
        self.log_blocking(self.make_record(WalOpType::Delete, id, &[], metadata))
    }
    pub fn log_update(&self, id: u32, vector: &[f32], metadata: &[u8]) -> io::Result<Lsn> {
        self.log_blocking(self.make_record(WalOpType::Update, id, vector, metadata))
    }

    // Async variants
    pub async fn log_insert_async(
        &self,
        id: u32,
        vector: &[f32],
        metadata: &[u8],
    ) -> io::Result<Lsn> {
        self.log_async(self.make_record(WalOpType::Insert, id, vector, metadata))
            .await
    }
    pub async fn log_delete_async(&self, id: u32, metadata: &[u8]) -> io::Result<Lsn> {
        self.log_async(self.make_record(WalOpType::Delete, id, &[], metadata))
            .await
    }
    pub async fn log_update_async(
        &self,
        id: u32,
        vector: &[f32],
        metadata: &[u8],
    ) -> io::Result<Lsn> {
        self.log_async(self.make_record(WalOpType::Update, id, vector, metadata))
            .await
    }

    pub async fn checkpoint(&self) -> io::Result<()> {
        let r#gen = self.checkpoint_generation.fetch_add(1, Ordering::Relaxed) + 1;
        let (ack_tx, ack_rx) = oneshot::channel();
        self.tx
            .send(WalMessage::Checkpoint {
                generation: r#gen,
                ack: ack_tx,
            })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "WAL writer shut down"))?;
        ack_rx
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "ack dropped"))?
    }

    pub async fn clear(&self) -> io::Result<()> {
        let (ack_tx, ack_rx) = oneshot::channel();
        self.tx
            .send(WalMessage::Truncate { ack: ack_tx })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "WAL writer shut down"))?;
        ack_rx
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "ack dropped"))?
    }

    pub async fn shutdown(&self) -> io::Result<()> {
        let (ack_tx, ack_rx) = oneshot::channel();
        self.tx
            .send(WalMessage::Shutdown { ack: ack_tx })
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "WAL writer shut down"))?;
        let result = ack_rx
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "ack dropped"))?;
        // Wait for the writer thread to exit so the WAL is fully flushed and the
        // file lock released before returning.
        self.join_writer();
        result
    }

    /// Block until the writer thread has exited (no-op if it already has).
    fn join_writer(&self) {
        if let Some(handle) = self.writer_thread.lock().unwrap().take() {
            let _ = handle.join();
        }
    }

    // -- Recovery ------------------------------------------------------------

    pub async fn load_for_recovery(db_path: &Path) -> io::Result<Vec<WalRecord>> {
        let mut records = Vec::new();
        let mut checkpoint_lsn = 0u64;

        for (_, path) in Self::list_segments(db_path) {
            Self::load_segment(&path, &mut checkpoint_lsn, &mut records)?;
        }
        let wal = Self::wal_path(db_path);
        if wal.exists() {
            Self::load_segment(&wal, &mut checkpoint_lsn, &mut records)?;
        }

        Ok(records
            .into_iter()
            .filter(|r| r.lsn.0 >= checkpoint_lsn && r.op_type != WalOpType::Checkpoint)
            .collect())
    }

    fn load_segment(
        path: &Path,
        checkpoint_lsn: &mut u64,
        out: &mut Vec<WalRecord>,
    ) -> io::Result<()> {
        let mut file = File::open(path)?;
        let mut hdr = [0u8; WAL_HEADER_SIZE];
        file.read_exact(&mut hdr)?;
        let header = WalHeader::from_bytes(&hdr)?;
        if header.last_checkpoint_lsn > *checkpoint_lsn {
            *checkpoint_lsn = header.last_checkpoint_lsn;
        }

        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        let mut offset = 0;
        let mut last_lsn = 0u64;
        while offset < data.len() {
            match WalRecord::from_bytes(&data[offset..]) {
                Ok((rec, n)) => {
                    if rec.lsn.0 != 0 && rec.lsn.0 <= last_lsn {
                        break;
                    }
                    last_lsn = rec.lsn.0;
                    out.push(rec);
                    offset += n;
                }
                Err(_) => break,
            }
        }
        Ok(())
    }

    // -- Path helpers --------------------------------------------------------

    fn wal_path(db_path: &Path) -> PathBuf {
        let mut s = db_path.as_os_str().to_owned();
        s.push(".wal");
        PathBuf::from(s)
    }

    fn segment_path(db_path: &Path, id: u64) -> PathBuf {
        let mut s = db_path.as_os_str().to_owned();
        s.push(".wal.");
        s.push(id.to_string());
        PathBuf::from(s)
    }

    fn segment_prefix(db_path: &Path) -> String {
        format!(
            "{}.wal.",
            db_path
                .file_name()
                .map_or_else(|| "wal".into(), |f| f.to_string_lossy())
        )
    }

    fn find_max_segment_id(db_path: &Path) -> u64 {
        let prefix = Self::segment_prefix(db_path);
        let dir = db_path.parent().unwrap_or(Path::new("."));
        let mut max = 0u64;
        if let Ok(entries) = fs::read_dir(dir) {
            for e in entries.flatten() {
                if let Some(s) = e.file_name().to_string_lossy().strip_prefix(&prefix)
                    && let Ok(id) = s.parse::<u64>()
                    && id >= max
                {
                    max = id + 1;
                }
            }
        }
        max
    }

    fn list_segments(db_path: &Path) -> Vec<(u64, PathBuf)> {
        let prefix = Self::segment_prefix(db_path);
        let dir = db_path.parent().unwrap_or(Path::new("."));
        let mut segs: Vec<(u64, PathBuf)> = fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let name = e.file_name();
                let s = name.to_string_lossy();
                let id = s.strip_prefix(&prefix)?.parse::<u64>().ok()?;
                Some((id, e.path()))
            })
            .collect();
        segs.sort_by_key(|(id, _)| *id);
        segs
    }
}

impl Drop for WriteAheadLog {
    fn drop(&mut self) {
        // Signal the writer to flush and exit, then join it so the WAL is durable
        // even if the caller never invoked shutdown().
        let (ack_tx, _ack_rx) = oneshot::channel();
        let _ = self.tx.send(WalMessage::Shutdown { ack: ack_tx });
        self.join_writer();
    }
}

// ---------------------------------------------------------------------------
// Writer task - single background task that serializes all disk writes
// ---------------------------------------------------------------------------

struct WalWriter {
    file: File,
    wal_path: PathBuf,
    db_path_prefix: String,
    last_written_lsn: Arc<AtomicU64>,
    next_lsn: Arc<AtomicU64>,
    last_checkpoint_lsn: Arc<AtomicU64>,
    last_checkpoint_generation: Arc<AtomicU64>,
    current_generation: u64,
    rx: std_mpsc::Receiver<WalMessage>,
    max_wal_segment_size: u64,
    max_total_wal_size: u64,
    max_wal_segments: u32,
    current_file_size: u64,
    next_segment_id: u64,
    write_buf: Vec<u8>,
}

impl WalWriter {
    fn run(mut self) {
        while let Ok(msg) = self.rx.recv() {
            match msg {
                WalMessage::Record(rec, ack) => {
                    let lsn = rec.lsn;
                    match self.write_record(&rec) {
                        Ok(()) => ack.send(Ok(lsn)),
                        Err(e) => {
                            // Report the failure to this caller and keep the writer
                            // alive: exiting here would silently drop every record
                            // already queued behind this one.
                            tracing::error!("WAL write failed: {}", e);
                            ack.send(Err(e));
                        }
                    }
                }
                WalMessage::Checkpoint { generation, ack } => {
                    let _ = ack.send(self.do_checkpoint(generation));
                }
                WalMessage::Truncate { ack } => {
                    let _ = ack.send(self.do_truncate());
                }
                WalMessage::Shutdown { ack } => {
                    self.drain();
                    let _ = self.file.flush();
                    let _ = self.file.sync_all();
                    let _ = ack.send(Ok(()));
                    return;
                }
            }
        }
        // Channel closed without shutdown - drain and fsync
        self.drain();
        let _ = self.file.flush();
        let _ = self.file.sync_all();
    }

    fn drain(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                WalMessage::Record(rec, ack) => {
                    let lsn = rec.lsn;
                    match self.write_record(&rec) {
                        Ok(()) => ack.send(Ok(lsn)),
                        Err(e) => {
                            tracing::error!("WAL drain write failed: {}", e);
                            ack.send(Err(e));
                        }
                    }
                }
                WalMessage::Checkpoint { generation, ack } => {
                    let _ = ack.send(self.do_checkpoint(generation));
                }
                WalMessage::Truncate { ack } => {
                    let _ = ack.send(self.do_truncate());
                }
                WalMessage::Shutdown { .. } => {}
            }
        }
    }

    fn write_record(&mut self, rec: &WalRecord) -> io::Result<()> {
        self.write_buf.clear();
        rec.write_to(&mut self.write_buf);
        let n = self.write_buf.len() as u64;
        self.file.write_all(&self.write_buf)?;
        self.current_file_size += n;
        self.last_written_lsn
            .fetch_max(rec.lsn.0 + 1, Ordering::Release);

        if self.max_wal_segment_size > 0 && self.current_file_size > self.max_wal_segment_size {
            self.rotate()?;
        }
        Ok(())
    }

    fn do_checkpoint(&mut self, generation: u64) -> io::Result<()> {
        let lsn = Lsn(self.last_written_lsn.load(Ordering::Acquire));

        // Write checkpoint record
        let rec = WalRecord {
            lsn,
            timestamp: now_ms(),
            op_type: WalOpType::Checkpoint,
            vector_id: 0,
            vector_data: Vec::new(),
            metadata: Vec::new(),
        };
        self.write_buf.clear();
        rec.write_to(&mut self.write_buf);
        self.file.write_all(&self.write_buf)?;
        self.current_file_size += self.write_buf.len() as u64;

        // Update header: seek back, write, seek forward, fsync
        let hdr = WalHeader {
            last_checkpoint_lsn: lsn.0,
            current_lsn: lsn.0 + 1,
            last_checkpoint_generation: generation,
        };
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&hdr.to_bytes())?;
        self.file.seek(SeekFrom::Start(self.current_file_size))?;
        self.file.flush()?;
        self.file.sync_all()?;

        self.current_generation = generation;
        self.last_checkpoint_lsn.store(lsn.0, Ordering::Release);
        self.last_checkpoint_generation
            .store(generation, Ordering::Relaxed);

        // The checkpoint record occupies LSN `lsn`; advance the counter so the
        // next data record cannot reuse it (e.g. if clear() fails and the
        // checkpoint record remains in the file).
        self.next_lsn.fetch_max(lsn.0 + 1, Ordering::Relaxed);

        if self.max_total_wal_size > 0 {
            self.cleanup();
        }
        Ok(())
    }

    fn do_truncate(&mut self) -> io::Result<()> {
        for (_, p) in WriteAheadLog::list_segments(Path::new(&self.db_path_prefix)) {
            let _ = fs::remove_file(&p);
        }
        self.file.set_len(WAL_HEADER_SIZE as u64)?;
        self.file.seek(SeekFrom::Start(0))?;
        let lsn = self.last_written_lsn.load(Ordering::Acquire);
        let hdr = WalHeader {
            last_checkpoint_lsn: lsn,
            current_lsn: lsn,
            last_checkpoint_generation: self.last_checkpoint_generation.load(Ordering::Relaxed),
        };
        self.file.write_all(&hdr.to_bytes())?;
        self.file.flush()?;
        self.file.sync_all()?;
        self.current_file_size = WAL_HEADER_SIZE as u64;
        self.last_checkpoint_lsn.store(lsn, Ordering::Release);
        Ok(())
    }

    fn rotate(&mut self) -> io::Result<()> {
        self.file.flush()?;
        self.file.sync_all()?;

        // Drop old handle, rename to segment, create new WAL
        drop(std::mem::replace(&mut self.file, open_null_file()?));

        let seg_path =
            WriteAheadLog::segment_path(Path::new(&self.db_path_prefix), self.next_segment_id);
        fs::rename(&self.wal_path, &seg_path)?;
        self.next_segment_id += 1;

        let new_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&self.wal_path)?;
        self.file = new_file;
        // The exclusive lock lives on the old file descriptor, which was just
        // dropped. Re-acquire it on the new active WAL so the lock survives
        // rotation.
        if let Err(e) = self.file.try_lock_exclusive() {
            tracing::warn!("WAL re-lock after rotation failed: {}", e);
        }

        // Write header to new WAL with current checkpoint state
        let hdr = WalHeader {
            last_checkpoint_lsn: self.last_checkpoint_lsn.load(Ordering::Acquire),
            current_lsn: self.last_written_lsn.load(Ordering::Acquire),
            last_checkpoint_generation: self.last_checkpoint_generation.load(Ordering::Relaxed),
        };
        self.file.write_all(&hdr.to_bytes())?;
        self.file.flush()?;
        self.file.sync_all()?;
        self.current_file_size = WAL_HEADER_SIZE as u64;

        Ok(())
    }

    fn cleanup(&mut self) {
        let segs = WriteAheadLog::list_segments(Path::new(&self.db_path_prefix));
        let mut total = self.current_file_size;
        let sizes: Vec<(u64, PathBuf)> = segs
            .iter()
            .filter_map(|(_, p)| {
                let size = fs::metadata(p).ok()?.len();
                total += size;
                Some((size, p.clone()))
            })
            .collect();

        // Track how many segments remain so the count limit is re-evaluated as we
        // delete. Using the initial `segs.len()` throughout would keep `over_count`
        // stuck true and delete far more segments than configured.
        let mut remaining = segs.len() as u32;

        for (size, path) in &sizes {
            let over_size = self.max_total_wal_size > 0 && total > self.max_total_wal_size;
            let over_count = self.max_wal_segments > 0 && remaining > self.max_wal_segments;
            if !over_size && !over_count {
                break;
            }

            if let Ok(mut f) = File::open(path) {
                let mut hdr = [0u8; WAL_HEADER_SIZE];
                if f.read_exact(&mut hdr).is_ok()
                    && let Ok(h) = WalHeader::from_bytes(&hdr)
                    && h.last_checkpoint_lsn > 0
                {
                    let _ = fs::remove_file(path);
                    total -= size;
                    remaining -= 1;
                }
            }
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Open a platform-appropriate null file handle (used as a dummy for mem::replace)
fn open_null_file() -> io::Result<File> {
    #[cfg(unix)]
    {
        OpenOptions::new().read(true).open("/dev/null")
    }
    #[cfg(windows)]
    {
        OpenOptions::new().read(true).open("NUL")
    }
    #[cfg(not(any(unix, windows)))]
    {
        std::fs::File::open(std::env::temp_dir().join(".null_dummy"))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_roundtrip() {
        let rec = WalRecord {
            lsn: Lsn(1),
            timestamp: 1234567890,
            op_type: WalOpType::Insert,
            vector_id: 42,
            vector_data: vec![0.1, 0.2, 0.3],
            metadata: b"test".to_vec(),
        };
        let mut buf = Vec::new();
        rec.write_to(&mut buf);
        let (decoded, n) = WalRecord::from_bytes(&buf).unwrap();
        assert_eq!(n, buf.len());
        assert_eq!(decoded.lsn, Lsn(1));
        assert_eq!(decoded.vector_id, 42);
        assert_eq!(decoded.vector_data, vec![0.1, 0.2, 0.3]);
        assert_eq!(decoded.metadata, b"test".to_vec());
    }

    #[test]
    fn test_crc_corruption_detected() {
        let rec = WalRecord {
            lsn: Lsn(1),
            timestamp: 0,
            op_type: WalOpType::Insert,
            vector_id: 1,
            vector_data: vec![0.5, 0.6],
            metadata: b"data".to_vec(),
        };
        let mut buf = Vec::new();
        rec.write_to(&mut buf);
        buf[40] ^= 0xFF; // corrupt payload
        assert!(WalRecord::from_bytes(&buf).is_err());
    }

    #[test]
    fn test_delete_record() {
        let rec = WalRecord {
            lsn: Lsn(5),
            timestamp: 0,
            op_type: WalOpType::Delete,
            vector_id: 99,
            vector_data: Vec::new(),
            metadata: b"old".to_vec(),
        };
        let mut buf = Vec::new();
        rec.write_to(&mut buf);
        let (decoded, _) = WalRecord::from_bytes(&buf).unwrap();
        assert_eq!(decoded.op_type, WalOpType::Delete);
        assert_eq!(decoded.vector_id, 99);
    }

    #[test]
    fn test_header_roundtrip() {
        let hdr = WalHeader {
            last_checkpoint_lsn: 100,
            current_lsn: 200,
            last_checkpoint_generation: 5,
        };
        let buf = hdr.to_bytes();
        let decoded = WalHeader::from_bytes(&buf).unwrap();
        assert_eq!(decoded.last_checkpoint_lsn, 100);
        assert_eq!(decoded.current_lsn, 200);
        assert_eq!(decoded.last_checkpoint_generation, 5);
    }

    #[test]
    fn test_header_corruption_detected() {
        let hdr = WalHeader {
            last_checkpoint_lsn: 100,
            current_lsn: 200,
            last_checkpoint_generation: 5,
        };
        let mut buf = hdr.to_bytes();
        buf[10] ^= 0xFF;
        assert!(WalHeader::from_bytes(&buf).is_err());
    }

    #[test]
    fn test_empty_payload_record() {
        let rec = WalRecord {
            lsn: Lsn(99),
            timestamp: 9999,
            op_type: WalOpType::Delete,
            vector_id: 0,
            vector_data: Vec::new(),
            metadata: Vec::new(),
        };
        let mut buf = Vec::new();
        rec.write_to(&mut buf);
        let (decoded, n) = WalRecord::from_bytes(&buf).unwrap();
        assert_eq!(n, buf.len());
        assert_eq!(decoded.lsn, Lsn(99));
        assert!(decoded.vector_data.is_empty());
        assert!(decoded.metadata.is_empty());
    }

    #[test]
    fn test_record_header_too_short() {
        let buf = vec![0u8; 10];
        assert!(WalRecord::from_bytes(&buf).is_err());
    }

    #[test]
    fn test_record_bad_footer() {
        let rec = WalRecord {
            lsn: Lsn(1),
            timestamp: 0,
            op_type: WalOpType::Insert,
            vector_id: 1,
            vector_data: vec![1.0],
            metadata: Vec::new(),
        };
        let mut buf = Vec::new();
        rec.write_to(&mut buf);
        // Corrupt the footer (last 4 bytes)
        let last = buf.len() - 1;
        buf[last] ^= 0xFF;
        assert!(WalRecord::from_bytes(&buf).is_err());
    }

    #[test]
    fn test_header_bad_magic() {
        let hdr = WalHeader {
            last_checkpoint_lsn: 0,
            current_lsn: 0,
            last_checkpoint_generation: 0,
        };
        let mut buf = hdr.to_bytes();
        buf[0] = 0xFF; // corrupt magic
        assert!(WalHeader::from_bytes(&buf).is_err());
    }

    #[test]
    fn test_header_too_short() {
        let buf = [0u8; WAL_HEADER_SIZE];
        // Manually corrupt to simulate truncated header scenario
        // from_bytes checks CRC first, so a zero buffer will fail CRC
        assert!(WalHeader::from_bytes(&buf).is_err());
    }

    #[test]
    fn test_lsn_ordering() {
        assert!(Lsn(1) < Lsn(2));
        assert!(Lsn(100) > Lsn(50));
        assert_eq!(Lsn(42), Lsn(42));
    }

    #[test]
    fn test_op_type_roundtrip() {
        assert_eq!(WalOpType::from_u8(1).unwrap(), WalOpType::Insert);
        assert_eq!(WalOpType::from_u8(2).unwrap(), WalOpType::Delete);
        assert_eq!(WalOpType::from_u8(3).unwrap(), WalOpType::Update);
        assert_eq!(WalOpType::from_u8(4).unwrap(), WalOpType::Checkpoint);
        assert!(WalOpType::from_u8(0).is_err());
        assert!(WalOpType::from_u8(5).is_err());
    }

    fn temp_db_path(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("wal_{}_{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir.join("test.db")
    }

    /// Parse every record in an active WAL file (skipping the 40-byte header).
    fn read_records(path: &Path) -> Vec<WalRecord> {
        let data = fs::read(path).unwrap();
        let mut out = Vec::new();
        let mut offset = WAL_HEADER_SIZE;
        while offset < data.len() {
            match WalRecord::from_bytes(&data[offset..]) {
                Ok((rec, n)) => {
                    offset += n;
                    out.push(rec);
                }
                Err(_) => break,
            }
        }
        out
    }

    /// Regression for "ack at enqueue": the record must already be on disk when
    /// log_insert returns, not merely queued for a background writer.
    #[tokio::test]
    async fn test_record_on_disk_before_log_returns() {
        let db_path = temp_db_path("durability");
        let wal = WriteAheadLog::new(&db_path, 0, 0, 0).await.unwrap();

        wal.log_insert(7, &[1.0, 2.0], b"hello").unwrap();

        // Read the WAL back through an independent handle: the record that was
        // just logged must already be present (LOG-BEFORE-APPLY).
        let recs = read_records(&WriteAheadLog::wal_path(&db_path));
        assert_eq!(recs.len(), 1, "record must be written before log_insert returns");
        assert_eq!(recs[0].vector_id, 7);
        assert_eq!(recs[0].metadata, b"hello");

        wal.shutdown().await.unwrap();
        let _ = fs::remove_dir_all(db_path.parent().unwrap());
    }

    /// Regression for the checkpoint LSN collision: a checkpoint record must not
    /// reuse the LSN of the next data record, even if clear() is never called.
    #[tokio::test]
    async fn test_checkpoint_does_not_cause_lsn_reuse() {
        let db_path = temp_db_path("ckpt_lsn");
        let wal = WriteAheadLog::new(&db_path, 0, 0, 0).await.unwrap();

        wal.log_insert(1, &[1.0], b"a").unwrap(); // lsn 0
        wal.checkpoint().await.unwrap(); // checkpoint record, no clear()
        wal.log_insert(2, &[2.0], b"b").unwrap(); // must NOT reuse the checkpoint LSN

        let lsns: Vec<u64> = read_records(&WriteAheadLog::wal_path(&db_path))
            .iter()
            .map(|r| r.lsn.0)
            .collect();
        assert!(
            lsns.windows(2).all(|w| w[1] > w[0]),
            "LSNs must be strictly increasing (no reuse), got {:?}",
            lsns
        );

        wal.shutdown().await.unwrap();
        let _ = fs::remove_dir_all(db_path.parent().unwrap());
    }
}
