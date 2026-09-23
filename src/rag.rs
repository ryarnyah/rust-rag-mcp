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
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

/// P3: Metadata index for O(1) source lookups
/// Maps source path to vector IDs, avoiding full table scans
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
}

pub struct RagCore {
    vectors_db: AsyncVectorDb,
    embedding: EmbeddingService,
    chunker: Chunker,
    syntax_chunker: SyntaxChunker,
    metadata_index: MetadataIndex,
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

        // P3: Create metadata index and populate from existing vectors
        let metadata_index = MetadataIndex::new();

        // Rebuild index from existing vectors on startup
        let total = vectors_db.len().await;
        for id in 0..total as u32 {
            if vectors_db.is_deleted(id).await {
                continue;
            }
            if let Ok(Some(metadata_bytes)) = vectors_db.get_meta(id).await {
                if let Ok(chunk_meta) = serde_json::from_slice::<ChunkMetadata>(&metadata_bytes) {
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

        Ok(Self {
            vectors_db,
            embedding,
            chunker: Chunker::new(chunk_size, overlap),
            syntax_chunker: SyntaxChunker::new(chunk_size, overlap),
            metadata_index,
        })
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
            self.delete_source(&source_path).await?;
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
        let content_hash = Self::compute_text_hash(text);

        if let Some(status) = self.document_status(source).await? {
            if status.content_hash == content_hash {
                return Ok(IndexResult::Skipped);
            }
            self.delete_source(source).await?;
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
    pub async fn close(&self) -> Result<()> {
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
