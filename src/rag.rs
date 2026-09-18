use crate::chunker::Chunker;
use crate::docs;
use crate::embeddings::EmbeddingService;
use crate::r_vector::{AsyncVectorDb, Config as VectorDbConfig};
use crate::syntax_chunker::{language_for_extension, SyntaxChunker};
use crate::{DocumentChunk, DocumentStatus, IndexResult, SearchResult};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::collections::HashSet;

pub struct RagCore {
    vectors_db: AsyncVectorDb,
    embedding: EmbeddingService,
    chunker: Chunker,
    syntax_chunker: SyntaxChunker,
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

    /**
     * Creates a new instance of RagCore with the specified database path, cache directory,
     * model name, chunk size, and overlap. Initializes the embedding service, chunker,
     * syntax chunker, and sets up the necessary vector database.
     */
    pub async fn new(
        db_path: &str,
        cache_dir: &str,
        model_name: &str,
        chunk_size: usize,
        overlap: usize,
    ) -> Result<Self> {
        let embedding = EmbeddingService::new(
            model_name,
            cache_dir
        )?;
        let ndims = embedding.dimensions();

        // Create vector database for chunks and metadata
        let vectors_cfg = VectorDbConfig::new(ndims)
            .with_m(20)
            .with_ef_construction(200)
            .with_capacity(1024);
        let vectors_db = AsyncVectorDb::open(db_path, vectors_cfg).await?;

        Ok(Self {
            vectors_db,
            embedding,
            chunker: Chunker::new(chunk_size, overlap),
            syntax_chunker: SyntaxChunker::new(chunk_size, overlap),
        })
    }

    /**
     * Computes the SHA-256 hash of the contents of the specified file asynchronously.
     */
    async fn compute_file_hash(path: &Path) -> Result<String> {
        let bytes = tokio::fs::read(path).await?;
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        Ok(hex::encode(hasher.finalize()))
    }

    /**
     * Computes the SHA-256 hash of the given text synchronously.
     */
    fn compute_text_hash(text: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        hex::encode(hasher.finalize())
    }

    /**
     * Indexes the specified file by extracting its text, chunking it, and storing the chunks and metadata in the database.
     */
    pub async fn index_file(&self, path: &Path) -> Result<IndexResult> {
        let source_path = path
            .canonicalize()
            .unwrap_or_else(|_| path.to_path_buf())
            .to_string_lossy()
            .to_string();

        let content_hash = Self::compute_file_hash(path).await?;

        if let Some(status) = self.document_status(&source_path).await? {
            if status.content_hash == content_hash {
                return Ok(IndexResult::Skipped);
            }
            self.delete_source(&source_path).await?;
        }

        let text = docs::extract_text(path).await?;

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let chunks = if language_for_extension(&ext).is_some() {
            self.syntax_chunker.chunk_text(&text, &source_path)
        } else {
            self.chunker.chunk_text(&text, &source_path)
        };
        let count = chunks.len();
        self.index_chunks(chunks).await?;

        let now = chrono_free_timestamp();
        self.upsert_metadata(&source_path, &content_hash, now, count as u32)
            .await?;

        Ok(IndexResult::Indexed(count))
    }

    /**
     * Indexes the given text by chunking it and storing the chunks and metadata in the database.
     */
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

        let now = chrono_free_timestamp();
        self.upsert_metadata(source, &content_hash, now, count as u32)
            .await?;

        Ok(IndexResult::Indexed(count))
    }

    /**
     * Indexes the given chunks by generating embeddings and storing them in the database.
     */
    async fn index_chunks(&self, chunks: Vec<DocumentChunk>) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        let embeddings_result = self.embedding.embed_chunks(chunks.clone()).await?;

        // Insert each chunk with its embedding
        for (chunk, embedding) in chunks.iter().zip(embeddings_result.iter()) {
            // Serialize chunk metadata
            let chunk_meta = ChunkMetadata {
                id: chunk.id.clone(),
                text: chunk.text.clone(),
                source: chunk.source.clone(),
                chunk_index: chunk.chunk_index,
                start_offset: chunk.start_offset as u64,
                end_offset: chunk.end_offset as u64,
            };
            let metadata_bytes = serde_json::to_vec(&chunk_meta)?;

            // Insert vector with metadata
            let _ = self.vectors_db
                .insert(embedding, Some(&metadata_bytes))
                .await?;
        }

        Ok(())
    }

    /**
     * Performs a semantic search for the given query string, returning the top_k most relevant results.
     * Optionally filters results by the specified source.
     */
    pub async fn search(
        &self,
        query: &str,
        top_k: usize,
        source_filter: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let query_embedding_vec = self
            .embedding
            .embed_chunks(vec![DocumentChunk {
                id: "query".to_string(),
                text: query.to_string(),
                source: "query".to_string(),
                chunk_index: 0,
                start_offset: 0,
                end_offset: 0,
            }])
            .await?;

        if query_embedding_vec.is_empty() || query_embedding_vec[0].is_empty() {
            return Err(anyhow::anyhow!("Failed to generate query embedding"));
        }

        let query_embedding = &query_embedding_vec[0];

        // Search vectors
        let search_results = self.vectors_db
            .search(query_embedding, top_k, 200)
            .await?;

        let mut results = Vec::new();
        for hit in search_results {
            // Deserialize chunk metadata
            if let Ok(chunk_meta) = serde_json::from_slice::<ChunkMetadata>(&hit.metadata) {
                // Apply source filter if needed
                if let Some(filter) = source_filter {
                    if chunk_meta.source != filter {
                        continue;
                    }
                }

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
        }

        Ok(results)
    }

    /**
     * Returns the total number of chunks stored in the database.
     */
    pub async fn chunk_count(&self) -> Result<usize> {
        Ok(self.vectors_db.len().await)
    }

    /**
     * Returns a list of all unique sources present in the database.
     */
    pub async fn list_sources(&self) -> Result<Vec<String>> {
        let mut sources = HashSet::new();

        // Iterate through all vectors and extract sources from metadata
        let total = self.vectors_db.len().await;
        for id in 0..total as u32 {
            if let Ok(Some(metadata_bytes)) = self.vectors_db.get_meta(id).await {
                if let Ok(chunk_meta) = serde_json::from_slice::<ChunkMetadata>(&metadata_bytes) {
                    sources.insert(chunk_meta.source);
                }
            }
        }

        let mut v: Vec<String> = sources.into_iter().collect();
        v.sort();
        Ok(v)
    }

    /**
     * Deletes all chunks and metadata associated with the specified source path from the database.
     */
    pub async fn delete_source(&self, source_path: &str) -> Result<()> {
        // Find all vectors with matching source and delete them
        let total = self.vectors_db.len().await;
        for id in 0..total as u32 {
            if self.vectors_db.is_deleted(id).await {
                continue;
            }
            if let Ok(Some(metadata_bytes)) = self.vectors_db.get_meta(id).await {
                if let Ok(chunk_meta) = serde_json::from_slice::<ChunkMetadata>(&metadata_bytes) {
                    if chunk_meta.source == source_path {
                        let _ = self.vectors_db.delete(id).await;
                    }
                }
            }
        }

        Ok(())
    }

    /**
     * Retrieves the status of the document associated with the specified source path.
     * Returns None if the document is not found.
     */
    pub async fn document_status(&self, source_path: &str) -> Result<Option<DocumentStatus>> {
        let total = self.vectors_db.len().await;
        for id in 0..total as u32 {
            if self.vectors_db.is_deleted(id).await {
                continue;
            }
            if let Ok(Some(metadata_bytes)) = self.vectors_db.get_meta(id).await {
                // Try to find document metadata entry for this source
                if let Ok(doc_meta) = serde_json::from_slice::<DocumentMetadataEntry>(&metadata_bytes) {
                    if doc_meta.source_path == source_path {
                        return Ok(Some(DocumentStatus {
                            source_path: doc_meta.source_path,
                            content_hash: doc_meta.content_hash,
                            indexed_at: doc_meta.indexed_at,
                            chunk_count: doc_meta.chunk_count,
                        }));
                    }
                }
            }
        }
        Ok(None)
    }

    /**
     * Upserts the metadata for a document, replacing any existing entry with the same source path.
     */
    async fn upsert_metadata(
        &self,
        source_path: &str,
        content_hash: &str,
        indexed_at: u64,
        chunk_count: u32,
    ) -> Result<()> {
        // Find and delete existing document metadata entry for this source
        let total = self.vectors_db.len().await;
        for id in 0..total as u32 {
            if self.vectors_db.is_deleted(id).await {
                continue;
            }
            if let Ok(Some(metadata_bytes)) = self.vectors_db.get_meta(id).await {
                if let Ok(doc_meta) = serde_json::from_slice::<DocumentMetadataEntry>(&metadata_bytes) {
                    if doc_meta.source_path == source_path {
                        let _ = self.vectors_db.delete(id).await;
                    }
                }
            }
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
        let _ = self.vectors_db
            .insert(&dummy_embedding, Some(&metadata_bytes))
            .await?;

        Ok(())
    }

    pub fn list_embedding_models() -> Vec<String> {
        EmbeddingService::list_models()
    }
}

/**
 * Generates a timestamp string representing the current time in seconds since the UNIX epoch.
 */
fn chrono_free_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_else(|_| 0)
}
