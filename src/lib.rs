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
