pub mod chunker;
pub mod docs;
pub mod embeddings;
pub mod mcp;
pub mod rag;

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
