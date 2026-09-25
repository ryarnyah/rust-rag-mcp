use crate::chunker::Chunker;
use crate::docs;
use crate::embeddings::EmbeddingService;
use crate::r_vector::{AsyncVectorDb, Config as VectorDbConfig};
use crate::syntax_chunker::{SyntaxChunker, language_for_extension};
use crate::{DocumentChunk, DocumentStatus, IndexResult, SearchResult};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

/// P3: Metadata index for O(1) source lookups
/// Maps source path to vector IDs, avoiding full table scans
///
/// # Persistence: the `.srcidx` sidecar
///
/// The maps are mirrored into a sidecar file next to the vector files
/// (`db.srcidx`) so startup can skip the O(n) metadata scan. It is a
/// cache of derived state — never a source of truth — and it is allowed
/// to be absent, stale, or corrupt; the scan is always correct.
///
/// - *Validity*: the header carries `(vec_len, meta_record_count)` read
///   from the freshly opened database (WAL replay included); the sidecar
///   is used only on an exact match. Both tokens move on every mutation
///   that can change the maps — an insert appends a row *and* a
///   metadata record, a delete appends a tombstone record — so a
///   sidecar missing any insert/delete since it was written cannot
///   match. Compaction rewrites the files with renumbered ids and a
///   fresh record count, so it mismatches too and forces the correct
///   rebuild. (Compaction is not reachable through `RagCore`; wiring it
///   up would need the maps renumbered in lockstep, sidecar included.)
/// - *When it is written*: only at quiescent points — after a startup
///   rebuild, and in `close()` with the `ops` mutex held, which every
///   mutator also holds, so tokens can never race the maps. A crash
///   before `close()` leaves the previous sidecar, whose tokens then
///   mismatch the replayed database → rebuild. A failed write is
///   warned about, never fatal: the next open rebuilds.
/// - *The write itself*: temp file + `sync_all` + rename. Losing the
///   rename in a crash leaves the old file or neither file, both
///   handled above (no directory fsync needed: a lost rename means a
///   rebuild, which is safe by construction).
///
/// Memory cost is unchanged — the maps live in RAM either way. Startup
/// trades the O(n) scan for a sidecar read on clean reopens; a db with
/// no sidecar (or one that fails validation) pays the scan exactly as
/// before. The alternative of not indexing at all still makes
/// `list_sources`/`delete_source` O(n) metadata re-reads, which is the
/// cost this index exists to remove.
#[derive(Clone)]
struct MetadataIndex {
    source_to_ids: Arc<RwLock<HashMap<String, Vec<u32>>>>,
    doc_metadata_ids: Arc<RwLock<HashMap<String, u32>>>, // source -> doc metadata vector ID
}

impl MetadataIndex {
    fn new() -> Self {
        Self {
            source_to_ids: Arc::new(RwLock::new(HashMap::new())),
            doc_metadata_ids: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn add_chunk(&self, source: String, id: u32) {
        let mut map = self.source_to_ids.write().await;
        map.entry(source).or_insert_with(Vec::new).push(id);
    }

    async fn add_doc_metadata(&self, source: String, id: u32) {
        let mut map = self.doc_metadata_ids.write().await;
        map.insert(source, id);
    }

    async fn get_chunk_ids(&self, source: &str) -> Vec<u32> {
        let map = self.source_to_ids.read().await;
        map.get(source).cloned().unwrap_or_default()
    }

    async fn get_doc_metadata_id(&self, source: &str) -> Option<u32> {
        let map = self.doc_metadata_ids.read().await;
        map.get(source).copied()
    }

    async fn remove_source(&self, source: &str) {
        let mut chunk_map = self.source_to_ids.write().await;
        chunk_map.remove(source);

        let mut doc_map = self.doc_metadata_ids.write().await;
        doc_map.remove(source);
    }

    async fn get_all_sources(&self) -> Vec<String> {
        let map = self.source_to_ids.read().await;
        let mut sources: Vec<String> = map.keys().cloned().collect();
        sources.sort();
        sources
    }

    /// Sorted snapshot of both maps for the sidecar writer. Sorting makes
    /// the serialized form deterministic for a given logical state, so a
    /// rewrite with no changes produces an identical file.
    async fn snapshot(&self) -> (Vec<(String, Vec<u32>)>, Vec<(String, u32)>) {
        let chunks = {
            let map = self.source_to_ids.read().await;
            let mut entries: Vec<(String, Vec<u32>)> =
                map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            entries.sort();
            entries
        };
        let docs = {
            let map = self.doc_metadata_ids.read().await;
            let mut entries: Vec<(String, u32)> =
                map.iter().map(|(k, &v)| (k.clone(), v)).collect();
            entries.sort();
            entries
        };
        (chunks, docs)
    }

    /// Replace both maps wholesale — used when a validated sidecar is
    /// loaded at startup instead of scanning metadata.
    async fn replace(&self, chunks: Vec<(String, Vec<u32>)>, docs: Vec<(String, u32)>) {
        *self.source_to_ids.write().await = chunks.into_iter().collect();
        *self.doc_metadata_ids.write().await = docs.into_iter().collect();
    }
}

/// Sidecar magic: ASCII "SRCIDX" + format version. Any other value —
/// another file, a truncated header, a future version — means "unusable,
/// rebuild": the sidecar is a cache, so rejection is always safe.
const SRCIDX_MAGIC: u64 = 0x5352_4349_4458_0001;

/// Fixed sidecar header: magic, `vec_len`, `meta_record_count`,
/// chunk-source count, doc-source count.
const SRCIDX_HEADER: usize = 32;

/// A parsed sidecar payload: `(chunk sources, doc sources)`.
type SourceIndexPayload = (Vec<(String, Vec<u32>)>, Vec<(String, u32)>);

/// Serialize the source index for `db.srcidx`.
///
/// Layout (all little-endian): the 32-byte header, then the chunk
/// sources (name as `u32` length + bytes, then `u32` id count + ids),
/// then the doc sources (name, then the single `u32` id). Entries are
/// written in the order given — callers pass the sorted snapshot.
fn serialize_source_index(
    vec_len: u64,
    meta_record_count: u64,
    chunks: &[(String, Vec<u32>)],
    docs: &[(String, u32)],
) -> Result<Vec<u8>> {
    let name_bytes: usize = chunks.iter().map(|(s, _)| s.len()).sum::<usize>()
        + docs.iter().map(|(s, _)| s.len()).sum::<usize>();
    let body = chunks
        .iter()
        .map(|(_, ids)| 8 + ids.len() * 4)
        .sum::<usize>()
        + docs.len() * 8;
    let mut buf = Vec::with_capacity(SRCIDX_HEADER + name_bytes + body);

    buf.extend_from_slice(&SRCIDX_MAGIC.to_le_bytes());
    buf.extend_from_slice(&vec_len.to_le_bytes());
    buf.extend_from_slice(&meta_record_count.to_le_bytes());
    buf.extend_from_slice(
        &u32::try_from(chunks.len())
            .context("too many chunk sources")?
            .to_le_bytes(),
    );
    buf.extend_from_slice(
        &u32::try_from(docs.len())
            .context("too many doc sources")?
            .to_le_bytes(),
    );
    for (source, ids) in chunks {
        buf.extend_from_slice(
            &u32::try_from(source.len())
                .context("source name too long")?
                .to_le_bytes(),
        );
        buf.extend_from_slice(source.as_bytes());
        buf.extend_from_slice(
            &u32::try_from(ids.len())
                .context("too many ids for one source")?
                .to_le_bytes(),
        );
        for &id in ids {
            buf.extend_from_slice(&id.to_le_bytes());
        }
    }
    for (source, id) in docs {
        buf.extend_from_slice(
            &u32::try_from(source.len())
                .context("source name too long")?
                .to_le_bytes(),
        );
        buf.extend_from_slice(source.as_bytes());
        buf.extend_from_slice(&id.to_le_bytes());
    }
    Ok(buf)
}

/// Read a `u32` at `*cur`, advancing past it; `None` on any overrun.
fn take_u32(bytes: &[u8], cur: &mut usize) -> Option<u32> {
    let end = cur.checked_add(4)?;
    let slice = bytes.get(*cur..end)?;
    *cur = end;
    Some(u32::from_le_bytes(slice.try_into().ok()?))
}

/// Read a length-prefixed UTF-8 name at `*cur`; `None` on any overrun
/// or invalid UTF-8.
fn take_name(bytes: &[u8], cur: &mut usize) -> Option<String> {
    let len = take_u32(bytes, cur)? as usize;
    let end = cur.checked_add(len)?;
    let slice = bytes.get(*cur..end)?;
    *cur = end;
    std::str::from_utf8(slice).ok().map(str::to_owned)
}

/// Parse and validate `db.srcidx`. Returns `None` — meaning "rebuild
/// from metadata" — for *any* deviation: missing/foreign file, short
/// read, bad magic, token mismatch, invalid UTF-8, an id outside
/// `vec_len`, truncated entries, or trailing bytes. A sidecar is only
/// ever an optimization, so nothing here needs to explain itself to the
/// caller; every rejection path falls back to the always-correct scan.
fn parse_source_index(
    bytes: &[u8],
    vec_len: u64,
    meta_record_count: u64,
) -> Option<SourceIndexPayload> {
    if bytes.len() < SRCIDX_HEADER {
        return None;
    }
    let magic = u64::from_le_bytes(bytes[0..8].try_into().ok()?);
    if magic != SRCIDX_MAGIC {
        return None;
    }
    let got_vec_len = u64::from_le_bytes(bytes[8..16].try_into().ok()?);
    let got_record_count = u64::from_le_bytes(bytes[16..24].try_into().ok()?);
    if got_vec_len != vec_len || got_record_count != meta_record_count {
        return None;
    }
    let n_chunk_sources = u32::from_le_bytes(bytes[24..28].try_into().ok()?) as usize;
    let n_doc_sources = u32::from_le_bytes(bytes[28..32].try_into().ok()?) as usize;

    // Every entry costs at least 8 bytes (name length + id/count), so
    // this bounds both loops — and their allocations — by the file size
    // before trusting the counts from a potentially corrupt header.
    let body = bytes.len() - SRCIDX_HEADER;
    if n_chunk_sources
        .saturating_mul(8)
        .saturating_add(n_doc_sources.saturating_mul(8))
        > body
    {
        return None;
    }

    let mut cur = SRCIDX_HEADER;
    let mut chunks = Vec::with_capacity(n_chunk_sources);
    for _ in 0..n_chunk_sources {
        let source = take_name(bytes, &mut cur)?;
        let n_ids = take_u32(bytes, &mut cur)? as usize;
        if n_ids.saturating_mul(4) > bytes.len() - cur {
            return None;
        }
        let mut ids = Vec::with_capacity(n_ids);
        for _ in 0..n_ids {
            let id = take_u32(bytes, &mut cur)? as u64;
            if id >= vec_len {
                return None;
            }
            ids.push(id as u32);
        }
        chunks.push((source, ids));
    }

    let mut docs = Vec::with_capacity(n_doc_sources);
    for _ in 0..n_doc_sources {
        let source = take_name(bytes, &mut cur)?;
        let id = take_u32(bytes, &mut cur)? as u64;
        if id >= vec_len {
            return None;
        }
        docs.push((source, id as u32));
    }

    // Trailing bytes mean a torn or tampered body: reject rather than
    // trust a prefix.
    if cur != bytes.len() {
        return None;
    }
    Some((chunks, docs))
}

/// Write `bytes` to `path` atomically: temp file in the same directory,
/// `sync_all`, then rename over the destination. The temp file is
/// removed if any step fails (a leftover `.tmp` is harmless anyway —
/// the next write truncates it).
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut tmp_name = path.as_os_str().to_owned();
    tmp_name.push(".tmp");
    let tmp = PathBuf::from(tmp_name);

    let written = (|| -> Result<()> {
        let mut file =
            std::fs::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        file.write_all(bytes)
            .with_context(|| format!("writing {}", tmp.display()))?;
        file.sync_all()
            .with_context(|| format!("syncing {}", tmp.display()))?;
        Ok(())
    })();
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e).with_context(|| format!("renaming {} -> {}", tmp.display(), path.display()));
    }
    Ok(())
}

pub struct RagCore {
    vectors_db: AsyncVectorDb,
    embedding: EmbeddingService,
    chunker: Chunker,
    syntax_chunker: SyntaxChunker,
    metadata_index: MetadataIndex,
    /// Serializes every mutating operation — `index_file`, `index_text`,
    /// `delete_source`, `close` — so the `.srcidx` sidecar is written
    /// only at quiescent points (its tokens and the maps are then read
    /// atomically, and it also serializes indexing end-to-end, which
    /// used to interleave at `AsyncVectorDb` lock granularity).
    /// Read-only calls do not take it: the sidecar is written in
    /// `new()` before any other handle exists, and in `close()`.
    ops: Arc<Mutex<()>>,
    /// Where the source-index sidecar lives: `<db>/db.srcidx`.
    source_index_path: PathBuf,
}

/// Serialized chunk data with full metadata - stored in vector metadata field
#[derive(Serialize, Deserialize, Debug)]
struct ChunkMetadata {
    id: String,
    text: String,
    source: String,
    chunk_index: u32,
    start_offset: u64,
    end_offset: u64,
}

/// Serialized document metadata - also stored in vector metadata field
#[derive(Serialize, Deserialize, Debug)]
struct DocumentMetadataEntry {
    source_path: String,
    content_hash: String,
    indexed_at: u64,
    chunk_count: u32,
}

impl RagCore {
    /// Creates a new instance of RagCore with the specified database path, cache directory,
    /// model name, chunk size, overlap, and ef_construction. Initializes the embedding service, chunker,
    /// syntax chunker, and sets up the necessary vector database.
    pub async fn new(
        db_path: &Path,
        cache_dir: &Path,
        model_name: &str,
        chunk_size: usize,
        overlap: usize,
        ef_construction: usize,
    ) -> Result<Self> {
        let embedding = EmbeddingService::new(model_name, &cache_dir.to_string_lossy())?;
        let ndims = embedding.dimensions();

        // Ensure db_path points to a directory and create db files inside it
        tokio::fs::create_dir_all(db_path).await?;
        let db_file_path = db_path.join("db");

        // Create vector database for chunks and metadata
        let vectors_cfg = VectorDbConfig::new(ndims)
            .with_m(20)
            .with_ef_construction(ef_construction)
            .with_capacity(1024);
        let vectors_db = AsyncVectorDb::open(db_file_path, vectors_cfg).await?;

        // P3: Create metadata index. The `.srcidx` sidecar is valid only
        // if its generation tokens match the database exactly as it comes
        // out of open() (WAL replay included); anything else — a mutation
        // since the last persist, a compaction, corruption — falls back to
        // the scan, which is always correct.
        let metadata_index = MetadataIndex::new();
        let source_index_path = db_path.join("db.srcidx");

        let loaded = match Self::load_source_index(&source_index_path, &vectors_db).await {
            Some((chunks, docs)) => {
                metadata_index.replace(chunks, docs).await;
                true
            }
            None => false,
        };
        if !loaded {
            let total = vectors_db.len().await;
            for id in 0..total as u32 {
                if vectors_db.is_deleted(id).await {
                    continue;
                }
                if let Ok(Some(metadata_bytes)) = vectors_db.get_meta(id).await {
                    if let Ok(chunk_meta) = serde_json::from_slice::<ChunkMetadata>(&metadata_bytes)
                    {
                        metadata_index.add_chunk(chunk_meta.source, id).await;
                    } else if let Ok(doc_meta) =
                        serde_json::from_slice::<DocumentMetadataEntry>(&metadata_bytes)
                    {
                        metadata_index
                            .add_doc_metadata(doc_meta.source_path, id)
                            .await;
                    }
                }
            }
            // Startup is quiescent — no other handle exists yet — so the
            // rebuilt index can be persisted immediately. A failed write
            // only means the scan runs again next time.
            Self::persist_source_index(&source_index_path, &vectors_db, &metadata_index).await;
        }

        Ok(Self {
            vectors_db,
            embedding,
            chunker: Chunker::new(chunk_size, overlap),
            syntax_chunker: SyntaxChunker::new(chunk_size, overlap),
            metadata_index,
            ops: Arc::new(Mutex::new(())),
            source_index_path,
        })
    }

    /// Load the `.srcidx` sidecar if it exists *and* validates against
    /// the current database state; `None` (rebuild) covers a missing
    /// file, a token mismatch, and every form of corruption — the
    /// sidecar is a cache, so rejection never needs a reason beyond a
    /// debug log.
    async fn load_source_index(
        path: &Path,
        vectors_db: &AsyncVectorDb,
    ) -> Option<SourceIndexPayload> {
        let bytes = tokio::fs::read(path).await.ok()?;
        let vec_len = vectors_db.len().await as u64;
        let meta_record_count = vectors_db.meta_record_count().await as u64;
        let parsed = parse_source_index(&bytes, vec_len, meta_record_count)?;
        tracing::debug!(path = %path.display(), "loaded source index sidecar");
        Some(parsed)
    }

    /// Persist the sidecar — only ever called at a quiescent point
    /// (startup before any other handle exists, or `close()` with `ops`
    /// held). Failure is warned about, never propagated: a lost sidecar
    /// degrades to the startup scan.
    async fn persist_source_index(
        path: &Path,
        vectors_db: &AsyncVectorDb,
        metadata_index: &MetadataIndex,
    ) {
        if let Err(e) = Self::write_source_index(path, vectors_db, metadata_index).await {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "failed to persist source index sidecar; it will be rebuilt at next startup"
            );
        }
    }

    async fn write_source_index(
        path: &Path,
        vectors_db: &AsyncVectorDb,
        metadata_index: &MetadataIndex,
    ) -> Result<()> {
        let vec_len = vectors_db.len().await as u64;
        let meta_record_count = vectors_db.meta_record_count().await as u64;
        let (chunks, docs) = metadata_index.snapshot().await;
        let bytes = serialize_source_index(vec_len, meta_record_count, &chunks, &docs)?;
        write_atomic(path, &bytes)
    }

    /// Computes the SHA-256 hash of the given text synchronously.
    fn compute_text_hash(text: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Reads a file once and returns both its content hash and text extraction.
    /// This avoids TOCTOU races between hashing and extraction.
    async fn read_file_for_indexing(path: &Path) -> Result<(String, String)> {
        let bytes = tokio::fs::read(path)
            .await
            .with_context(|| format!("reading {}", path.display()))?;

        let content_hash = {
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            hex::encode(hasher.finalize())
        };

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let text = match ext.as_str() {
            "pdf" => docs::extract_text(path).await?,
            "docx" | "xlsx" | "pptx" => docs::extract_text(path).await?,
            _ => {
                // For text files, use the already-read bytes
                if let Ok(text) = std::str::from_utf8(&bytes) {
                    text.to_string()
                } else {
                    docs::extract_text(path).await?
                }
            }
        };

        Ok((content_hash, text))
    }

    /// Indexes the specified file by extracting its text, chunking it, and storing the chunks and metadata in the database.
    pub async fn index_file(&self, path: &Path) -> Result<IndexResult> {
        let _ops = self.ops.lock().await;
        let source_path = path
            .canonicalize()
            .map_err(|e| anyhow::anyhow!("Failed to canonicalize path {:?}: {}", path, e))?
            .to_string_lossy()
            .to_string();

        // Read file once to avoid TOCTOU between hash check and text extraction
        let (content_hash, text) = Self::read_file_for_indexing(path).await?;

        if let Some(status) = self.document_status(&source_path).await? {
            if status.content_hash == content_hash {
                return Ok(IndexResult::Skipped);
            }
            self.delete_source_inner(&source_path).await?;
        }

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let count = {
            let chunks = if language_for_extension(&ext).is_some() {
                self.syntax_chunker.chunk_text(&text, &source_path)
            } else {
                self.chunker.chunk_text(&text, &source_path)
            };
            let count = chunks.len();
            self.index_chunks(chunks).await?;
            count
        };

        let now = current_timestamp()?;
        self.upsert_metadata(&source_path, &content_hash, now, count as u32)
            .await?;

        Ok(IndexResult::Indexed(count))
    }

    /// Indexes the given text by chunking it and storing the chunks and metadata in the database.
    pub async fn index_text(&self, text: &str, source: &str) -> Result<IndexResult> {
        let _ops = self.ops.lock().await;
        let content_hash = Self::compute_text_hash(text);

        if let Some(status) = self.document_status(source).await? {
            if status.content_hash == content_hash {
                return Ok(IndexResult::Skipped);
            }
            self.delete_source_inner(source).await?;
        }

        let ext = source.rsplit('.').next().unwrap_or("").to_lowercase();

        let chunks = if language_for_extension(&ext).is_some() {
            self.syntax_chunker.chunk_text(text, source)
        } else {
            self.chunker.chunk_text(text, source)
        };
        let count = chunks.len();
        self.index_chunks(chunks).await?;

        let now = current_timestamp()?;
        self.upsert_metadata(source, &content_hash, now, count as u32)
            .await?;

        Ok(IndexResult::Indexed(count))
    }

    /// Indexes the given chunks by generating embeddings and storing them in the database.
    async fn index_chunks(&self, chunks: Vec<DocumentChunk>) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        const BATCH_SIZE: usize = 64;

        for batch in chunks.chunks(BATCH_SIZE) {
            let embeddings = self.embedding.embed_chunks(batch).await?;
            self.insert_batch(batch, &embeddings).await?;
        }

        self.vectors_db.flush().await?;

        Ok(())
    }

    async fn insert_batch(&self, batch: &[DocumentChunk], embeddings: &[Vec<f32>]) -> Result<()> {
        for (chunk, embedding) in batch.iter().zip(embeddings.iter()) {
            let chunk_meta = ChunkMetadata {
                id: chunk.id.clone(),
                text: chunk.text.clone(),
                source: chunk.source.clone(),
                chunk_index: chunk.chunk_index,
                start_offset: chunk.start_offset as u64,
                end_offset: chunk.end_offset as u64,
            };
            let metadata_bytes = serde_json::to_vec(&chunk_meta)?;

            let vec_id = self
                .vectors_db
                .insert(embedding, Some(&metadata_bytes))
                .await?;

            self.metadata_index
                .add_chunk(chunk.source.clone(), vec_id)
                .await;
        }
        Ok(())
    }

    /// Performs a semantic search for the given query string, returning the top_k most relevant results.
    /// Optionally filters results by the specified source.
    pub async fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        let query_chunk = vec![DocumentChunk {
            id: "query".to_string(),
            text: query.to_string(),
            source: "query".to_string(),
            chunk_index: 0,
            start_offset: 0,
            end_offset: 0,
        }];
        let query_embedding_vec = self.embedding.embed_chunks(&query_chunk).await?;

        if query_embedding_vec.is_empty() || query_embedding_vec[0].is_empty() {
            return Err(anyhow::anyhow!("Failed to generate query embedding"));
        }

        let query_embedding = &query_embedding_vec[0];

        // P5: Adaptive ef_search based on k
        // Small k: use lower ef (faster), large k: use higher ef (more thorough)
        let ef = (top_k as u32 * 4).clamp(40, 200);

        // Search vectors with optional source filter
        let search_results = self
            .vectors_db
            .search(query_embedding, top_k, ef as usize)
            .await?;

        let mut results = Vec::new();
        for hit in search_results {
            // Deserialize chunk metadata
            match serde_json::from_slice::<ChunkMetadata>(&hit.metadata) {
                Ok(chunk_meta) => {
                    let chunk = DocumentChunk {
                        id: chunk_meta.id,
                        text: chunk_meta.text,
                        source: chunk_meta.source,
                        chunk_index: chunk_meta.chunk_index,
                        start_offset: chunk_meta.start_offset as usize,
                        end_offset: chunk_meta.end_offset as usize,
                    };
                    results.push(SearchResult {
                        score: hit.score as f64,
                        chunk,
                    });
                }
                Err(e) => {
                    tracing::warn!(
                        vector_id = hit.id,
                        error = %e,
                        "Failed to deserialize chunk metadata, skipping result"
                    );
                }
            }
        }

        Ok(results)
    }

    /// Returns the total number of chunks stored in the database.
    pub async fn chunk_count(&self) -> Result<usize> {
        Ok(self.vectors_db.len().await)
    }

    /// Returns a list of all unique sources present in the database.
    pub async fn list_sources(&self) -> Result<Vec<String>> {
        // P3: Use metadata index for O(1) instead of O(n) full table scan
        Ok(self.metadata_index.get_all_sources().await)
    }

    /// Deletes all chunks and metadata associated with the specified source path from the database.
    pub async fn delete_source(&self, source_path: &str) -> Result<()> {
        let _ops = self.ops.lock().await;
        self.delete_source_inner(source_path).await
    }

    /// Body of [`Self::delete_source`] without taking `ops`: callers that
    /// already hold the lock (the indexing entry points, which delete
    /// before re-indexing) must not re-lock it — `tokio::sync::Mutex` is
    /// not reentrant.
    async fn delete_source_inner(&self, source_path: &str) -> Result<()> {
        // P3: Use metadata index for O(k) instead of O(n) full table scan
        // k = number of chunks for this source (much smaller than total vectors)
        let chunk_ids = self.metadata_index.get_chunk_ids(source_path).await;
        for id in chunk_ids {
            if let Err(e) = self.vectors_db.delete(id).await {
                tracing::warn!(vector_id = id, error = %e, "Failed to delete vector");
            }
        }

        // Also delete doc metadata if exists
        if let Some(doc_id) = self.metadata_index.get_doc_metadata_id(source_path).await
            && let Err(e) = self.vectors_db.delete(doc_id).await
        {
            tracing::warn!(vector_id = doc_id, error = %e, "Failed to delete doc metadata vector");
        }

        // Remove from index
        self.metadata_index.remove_source(source_path).await;

        Ok(())
    }

    pub async fn flush(&self) -> Result<()> {
        self.vectors_db
            .flush()
            .await
            .map_err(|e| anyhow::anyhow!("Flush failed: {}", e))
    }

    /// Properly shut down the database, waiting for WAL writer to finish
    /// and release all file locks
    /// Properly shut down the database: persist the `.srcidx` sidecar
    /// first (this is the quiescent point the design allows — `ops` is
    /// held, so the tokens and maps are read with no mutator in
    /// flight), then wait for the WAL writer to finish and release its
    /// file locks.
    pub async fn close(&self) -> Result<()> {
        let _ops = self.ops.lock().await;
        Self::persist_source_index(
            &self.source_index_path,
            &self.vectors_db,
            &self.metadata_index,
        )
        .await;
        self.vectors_db
            .close()
            .await
            .map_err(|e| anyhow::anyhow!("Close failed: {}", e))
    }

    /// Retrieves the status of the document associated with the specified source path.
    /// Returns None if the document is not found.
    pub async fn document_status(&self, source_path: &str) -> Result<Option<DocumentStatus>> {
        // P3: Use metadata index for O(1) lookup instead of O(n) full table scan
        if let Some(doc_id) = self.metadata_index.get_doc_metadata_id(source_path).await
            && let Ok(Some(metadata_bytes)) = self.vectors_db.get_meta(doc_id).await
            && let Ok(doc_meta) = serde_json::from_slice::<DocumentMetadataEntry>(&metadata_bytes)
        {
            return Ok(Some(DocumentStatus {
                source_path: doc_meta.source_path,
                content_hash: doc_meta.content_hash,
                indexed_at: doc_meta.indexed_at,
                chunk_count: doc_meta.chunk_count,
            }));
        }
        Ok(None)
    }

    /// Upserts the metadata for a document, replacing any existing entry with the same source path.
    async fn upsert_metadata(
        &self,
        source_path: &str,
        content_hash: &str,
        indexed_at: u64,
        chunk_count: u32,
    ) -> Result<()> {
        // P3: Use metadata index for O(1) lookup instead of O(n) full table scan
        // Delete existing entry if present
        if let Some(old_id) = self.metadata_index.get_doc_metadata_id(source_path).await
            && let Err(e) = self.vectors_db.delete(old_id).await
        {
            tracing::warn!(vector_id = old_id, error = %e, "Failed to delete old metadata vector");
        }

        // Insert new metadata entry with a dummy embedding vector
        let doc_meta = DocumentMetadataEntry {
            source_path: source_path.to_string(),
            content_hash: content_hash.to_string(),
            indexed_at,
            chunk_count,
        };
        let metadata_bytes = serde_json::to_vec(&doc_meta)?;

        // Create a dummy embedding vector for metadata storage
        let dummy_embedding = vec![0.0; self.embedding.dimensions()];
        let doc_id = self
            .vectors_db
            .insert(&dummy_embedding, Some(&metadata_bytes))
            .await?;

        // Update index
        self.metadata_index
            .add_doc_metadata(source_path.to_string(), doc_id)
            .await;

        Ok(())
    }

    pub fn list_embedding_models() -> Vec<String> {
        EmbeddingService::list_models()
    }
}

/// Returns the current time in seconds since the UNIX epoch.
/// Returns an error if the system clock is before the UNIX epoch.
fn current_timestamp() -> Result<u64> {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .map_err(|e| anyhow::anyhow!("System clock before UNIX epoch: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip: the parser returns exactly what the serializer wrote,
    /// for both payload kinds.
    #[test]
    fn source_index_roundtrip() {
        let chunks = vec![
            ("a.rs".to_string(), vec![0u32, 1, 2]),
            ("dir/b.md".to_string(), vec![3]),
            ("empty".to_string(), vec![]),
        ];
        let docs = vec![("a.rs".to_string(), 4u32)];
        let bytes = serialize_source_index(5, 7, &chunks, &docs).unwrap();
        let (got_chunks, got_docs) = parse_source_index(&bytes, 5, 7).unwrap();
        assert_eq!(got_chunks, chunks);
        assert_eq!(got_docs, docs);
    }

    /// An empty database still round-trips (a fresh dir persists an empty
    /// sidecar at startup, which the next open must accept).
    #[test]
    fn source_index_empty_roundtrip() {
        let bytes = serialize_source_index(0, 0, &[], &[]).unwrap();
        let (chunks, docs) = parse_source_index(&bytes, 0, 0).unwrap();
        assert!(chunks.is_empty() && docs.is_empty());
    }

    /// The whole safety argument of the sidecar: an exact generation
    /// match is required. Any insert or delete since the persist moves
    /// `(vec_len, meta_record_count)` and invalidates the file — which
    /// is what makes trusting it on a match sound.
    #[test]
    fn source_index_rejects_any_token_mismatch() {
        let bytes = serialize_source_index(10, 20, &[("s".to_string(), vec![1])], &[]).unwrap();
        assert!(
            parse_source_index(&bytes, 10, 20).is_some(),
            "exact generation must validate"
        );
        // A row append (insert) or a tombstone (delete) moves a token.
        assert!(parse_source_index(&bytes, 11, 20).is_none());
        assert!(parse_source_index(&bytes, 10, 21).is_none());
        // Compaction rewrites both.
        assert!(parse_source_index(&bytes, 9, 19).is_none());
    }

    /// Foreign file, short read, or mid-body truncation at *any* offset
    /// must be rejected outright — never partially trusted.
    #[test]
    fn source_index_rejects_foreign_and_truncated_bytes() {
        let bytes = serialize_source_index(
            3,
            1,
            &[("s".to_string(), vec![0, 1])],
            &[("d".to_string(), 2)],
        )
        .unwrap();

        let mut foreign = bytes.clone();
        foreign[0] = 0xFF; // e.g. an unrelated file named db.srcidx
        assert!(parse_source_index(&foreign, 3, 1).is_none());

        for cut in 0..bytes.len() {
            assert!(
                parse_source_index(&bytes[..cut], 3, 1).is_none(),
                "prefix of {cut} bytes must not parse"
            );
        }
    }

    /// Trailing bytes, an id outside the token-promised row range, an
    /// invalid UTF-8 name, and header counts that outgrow the body all
    /// mean "rebuild", not "use what parsed".
    #[test]
    fn source_index_rejects_tampered_body() {
        let base = serialize_source_index(
            4,
            4,
            &[("ok".to_string(), vec![1])],
            &[("x".to_string(), 0)],
        )
        .unwrap();

        // Trailing garbage after an otherwise valid body.
        let mut trailing = base.clone();
        trailing.extend_from_slice(&[0u8; 4]);
        assert!(parse_source_index(&trailing, 4, 4).is_none());

        // Id equal to vec_len: one past the last addressable row.
        let oob = serialize_source_index(4, 4, &[("s".to_string(), vec![4])], &[]).unwrap();
        assert!(parse_source_index(&oob, 4, 4).is_none());

        // Corrupt the doc name's single byte into invalid UTF-8.
        // Layout: 32-byte header, then name len (4 B), name, id (4 B).
        let mut bad_utf8 = base.clone();
        bad_utf8[36] = 0xFF;
        assert!(parse_source_index(&bad_utf8, 4, 4).is_none());

        // Header counts that could never fit the body (allocation bound).
        let mut absurd = serialize_source_index(4, 4, &[], &[]).unwrap();
        absurd[24..28].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(parse_source_index(&absurd, 4, 4).is_none());
    }
}
