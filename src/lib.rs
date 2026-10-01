pub mod bm25;
pub mod chunker;
pub mod docs;
pub mod embeddings;
pub mod mcp;
pub mod r_vector;
pub mod rag;
pub mod schemar_ext;
mod sidecar;
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

/// Parameters for whole-document matching, shared by the CLI `match`
/// command and the MCP `match_document` tool so both behave identically.
#[derive(Debug, Clone, Copy)]
pub struct MatchOptions {
    /// How many matched documents to return.
    pub top_k: usize,
    /// Retrieval mode used for each per-chunk search — the same modes
    /// `search` exposes, with the same score scales.
    pub mode: SearchMode,
    /// HNSW `ef` override for the dense side; `None` = adaptive scaling
    /// (see `search_with_mode_ef`).
    pub ef_search: Option<usize>,
    /// Candidate pool depth per query chunk (per retriever). Each chunk
    /// of the query document retrieves this many chunks before hits are
    /// grouped by document, so it bounds how many distinct documents a
    /// single chunk can contribute to.
    pub candidates_per_chunk: usize,
}

impl Default for MatchOptions {
    fn default() -> Self {
        Self {
            top_k: 5,
            mode: SearchMode::default(),
            ef_search: None,
            // Same depth as the hybrid pool: one chunk sees the same
            // candidates it would see in a `search` call.
            candidates_per_chunk: 50,
        }
    }
}

/// The whole document to match against the index: either side of the
/// comparison can be a *new* document (raw text or a file) or one that
/// is already indexed.
#[derive(Debug, Clone, Copy)]
pub enum MatchDocument<'a> {
    /// Raw text that is not (yet) indexed. `ext` selects syntax-aware
    /// chunking (e.g. `"rs"`); `""` uses the generic word chunker.
    Text { text: &'a str, ext: &'a str },
    /// A file on disk — it does not need to be indexed. Read and
    /// chunked exactly as `index` would, so the comparison uses the
    /// chunks the file would produce.
    File { path: &'a std::path::Path },
    /// A document already in the index, queried with its stored chunks
    /// (no re-chunking, no re-embedding of the index side).
    Indexed { source: &'a str },
}

/// One indexed document's aggregate match against a whole query document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMatch {
    /// Source key of the matched indexed document.
    pub source: String,
    /// Aggregate score: the mean over the query document's chunks of the
    /// best chunk-level score against this document, with a query chunk
    /// that never hit it contributing `0.0`. Documents matching more of
    /// the query therefore pull ahead. The scale is the one of
    /// [`MatchOptions::mode`] (`SearchResult::score`), never comparable
    /// across modes.
    pub score: f64,
    /// How many of the query document's chunks matched this document.
    pub matched_chunks: u32,
    /// Total chunks in the query document — the denominator of `score`.
    pub query_chunks: u32,
    /// Strongest single chunk-level score against this document.
    pub best_score: f64,
    /// Index of the query-document chunk behind `best_score`.
    pub best_query_chunk: u32,
    /// Chunk index (within the matched document) behind `best_score`.
    pub best_match_chunk: u32,
    /// Text of the matched chunk behind `best_score` — the passage that
    /// ties the two documents together.
    pub best_match_text: String,
}

/// Outcome of `RagCore::match_document`: the ranked matches plus what
/// the query document resolved to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchDocumentResult {
    /// Source key the query document resolved to (canonicalized file
    /// path or indexed source); `None` for raw text, which has no key.
    /// The same key is excluded from `matches` — a document is never
    /// its own match.
    pub query_source: Option<String>,
    /// Number of chunks the query document was split into — the
    /// denominator behind every [`DocumentMatch::score`].
    pub query_chunks: u32,
    /// Best-matching indexed documents, sorted by `score` descending.
    pub matches: Vec<DocumentMatch>,
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

/// One retriever's contribution to a hybrid (RRF) score: the raw score
/// it would have produced on its own and the 1-based rank that fed the
/// fusion. Because [`crate::bm25::rrf_fuse`] consumes only ranks, a
/// caller can reproduce the fused score exactly as
/// `Σ 1 / (60 + rank)` over the sides present.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrieverScore {
    /// Raw per-retriever score: cosine similarity in `[0, 1]` for the
    /// dense side, unbounded BM25 weight for the lexical side.
    pub score: f64,
    /// 1-based rank in that retriever's candidate pool.
    pub rank: u32,
}

/// Per-retriever components of a hybrid result's RRF score. A `None`
/// side means the document never entered that retriever's top-pool, so
/// it contributed nothing to the fused score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridScoreComponents {
    /// Dense HNSW contribution (cosine + rank); `None` if the document
    /// was outside the dense pool.
    pub dense: Option<RetrieverScore>,
    /// BM25 lexical contribution (weight + rank); `None` if no query
    /// token matched the document.
    pub lexical: Option<RetrieverScore>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub score: f64,
    pub chunk: DocumentChunk,
    /// Present only for hybrid results: the per-retriever scores and
    /// ranks that add up to `score`. Omitted from serialized output
    /// when absent, so semantic/lexical payloads are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub components: Option<HybridScoreComponents>,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk() -> DocumentChunk {
        DocumentChunk {
            id: "doc:0".into(),
            text: "text".into(),
            source: "a.rs".into(),
            chunk_index: 0,
            start_offset: 0,
            end_offset: 4,
        }
    }

    /// Non-hybrid results must serialize exactly as before this field
    /// existed, and legacy payloads (no `components` key) must still
    /// deserialize.
    #[test]
    fn search_result_omits_absent_components() {
        let result = SearchResult {
            score: 0.5,
            chunk: chunk(),
            components: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(
            !json.contains("components"),
            "None components must stay out of the payload: {json}"
        );

        let parsed: SearchResult = serde_json::from_str(&json).unwrap();
        assert!(parsed.components.is_none());
    }

    #[test]
    fn search_result_roundtrips_present_components() {
        let result = SearchResult {
            score: 1.0 / 61.0 + 1.0 / 62.0,
            chunk: chunk(),
            components: Some(HybridScoreComponents {
                dense: Some(RetrieverScore {
                    score: 0.84,
                    rank: 1,
                }),
                lexical: Some(RetrieverScore {
                    score: 7.13,
                    rank: 2,
                }),
            }),
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"components\""), "{json}");

        let parsed: SearchResult = serde_json::from_str(&json).unwrap();
        let components = parsed.components.expect("components survive roundtrip");
        assert_eq!(components, result.components.unwrap());

        // Components reproduce the fused score: Σ 1/(60 + rank).
        let recomputed: f64 = [components.dense.as_ref(), components.lexical.as_ref()]
            .iter()
            .flatten()
            .map(|side| 1.0 / (crate::bm25::RRF_K + side.rank as f64))
            .sum();
        assert!((recomputed - result.score).abs() < 1e-12);
    }
}
