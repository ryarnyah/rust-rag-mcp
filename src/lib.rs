pub mod bm25;
pub mod chunker;
pub mod docs;
pub mod embeddings;
pub mod mcp;
pub mod r_vector;
pub mod rag;
pub mod schemar_ext;
pub mod syntax_chunker;
pub mod wal;

use rmcp::schemars;
use serde::{Deserialize, Serialize};

/// How a query is executed against the knowledge base.
///
/// Shared by the MCP `search` tool and the CLI `search` command so both
/// expose exactly the same modes.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
    clap::ValueEnum,
)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// Dense HNSW retrieval + BM25 lexical retrieval, fused with
    /// Reciprocal Rank Fusion (default). Surfaces documents that rank
    /// well under *either* retriever without comparing their
    /// incommensurable score scales.
    #[default]
    Hybrid,
    /// Dense HNSW only; scores are cosine similarity in `[0, 1]`.
    Semantic,
    /// BM25 lexical only; scores are unbounded BM25 weights. Skips
    /// query embedding entirely (no model invocation).
    Lexical,
}

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
