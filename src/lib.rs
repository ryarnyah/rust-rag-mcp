pub mod chunker;
pub mod docs;
pub mod embeddings;
pub mod mcp;
pub mod rag;
pub mod syntax_chunker;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: String,
    pub text: String,
    pub source: String,
    pub chunk_index: u32,
    pub start_offset: usize,
    pub end_offset: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub score: f64,
    pub chunk: DocumentChunk,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndexResult {
    Indexed(usize),
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentStatus {
    pub source_path: String,
    pub content_hash: String,
    pub indexed_at: u64,
    pub chunk_count: u32,
}
