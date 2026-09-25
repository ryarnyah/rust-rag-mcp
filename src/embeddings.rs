use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::DocumentChunk;

pub struct EmbeddingService {
    model: Arc<Mutex<TextEmbedding>>,
    model_name: String,
    dimensions: usize,
    /// Instruction prepended to search queries (see [`prefixes_for`]).
    query_prefix: Option<&'static str>,
    /// Instruction prepended to indexed document text (see [`prefixes_for`]).
    passage_prefix: Option<&'static str>,
}

/// Task prefixes required by asymmetric embedding models, verified against the
/// official model cards (2026-09):
///
/// | Family | Query | Passage |
/// |---|---|---|
/// | E5 (`intfloat/multilingual-e5-*`) | `query: ` | `passage: ` |
/// | BGE en/zh v1.5 | instruction sentence (retrieval only) | *none* |
/// | BGE m3 | *none* | *none* |
/// | Nomic v1/v1.5 | `search_query: ` | `search_document: ` |
/// | ModernBERT-embed | `search_query: ` | `search_document: ` |
/// | EmbeddingGemma | `task: search result \| query: ` | `title: none \| text: ` |
/// | mxbai, Snowflake Arctic | `Represent this sentence for searching relevant passages: ` | *none* |
/// | everything else (MiniLM, MPNet, GTE, CLIP, Jina, ...) | *none* | *none* |
///
/// Symmetric models must be embedded verbatim: adding prefixes to a model not
/// trained on them injects out-of-distribution tokens and hurts retrieval.
///
/// Note for BGE v1.5: the card states passages *never* get an instruction, and
/// that queries work nearly as well without one ("slight degradation"), but it
/// recommends the instruction for the short-query→long-document retrieval
/// pattern, which is exactly what [`crate::rag::RagCore::search`] does.
///
/// NOTE: changing any of these strings changes the embedding space for that
/// model on the passage side (E5/Nomic/ModernBERT/EmbeddingGemma) or the query
/// side (all query-side-only families). Databases indexed under a different
/// prefix policy are incompatible; re-index from scratch after upgrading.
/// There is deliberately no automatic migration (documented decision).
fn prefixes_for(model: &EmbeddingModel) -> (Option<&'static str>, Option<&'static str>) {
    const BGE_EN_INSTRUCTION: &str = "Represent this sentence for searching relevant passages: ";
    const BGE_ZH_INSTRUCTION: &str = "为这个句子生成表示以用于检索相关文章：";
    match model {
        EmbeddingModel::MultilingualE5Small
        | EmbeddingModel::MultilingualE5Base
        | EmbeddingModel::MultilingualE5Large => (Some("query: "), Some("passage: ")),
        EmbeddingModel::NomicEmbedTextV1
        | EmbeddingModel::NomicEmbedTextV15
        | EmbeddingModel::NomicEmbedTextV15Q
        | EmbeddingModel::ModernBertEmbedLarge => {
            (Some("search_query: "), Some("search_document: "))
        }
        EmbeddingModel::EmbeddingGemma300M
        | EmbeddingModel::EmbeddingGemma300MQ
        | EmbeddingModel::EmbeddingGemma300MQ4 => (
            Some("task: search result | query: "),
            Some("title: none | text: "),
        ),
        EmbeddingModel::BGEBaseENV15
        | EmbeddingModel::BGEBaseENV15Q
        | EmbeddingModel::BGELargeENV15
        | EmbeddingModel::BGELargeENV15Q
        | EmbeddingModel::BGESmallENV15
        | EmbeddingModel::BGESmallENV15Q
        | EmbeddingModel::MxbaiEmbedLargeV1
        | EmbeddingModel::MxbaiEmbedLargeV1Q
        | EmbeddingModel::SnowflakeArcticEmbedXS
        | EmbeddingModel::SnowflakeArcticEmbedXSQ
        | EmbeddingModel::SnowflakeArcticEmbedS
        | EmbeddingModel::SnowflakeArcticEmbedSQ
        | EmbeddingModel::SnowflakeArcticEmbedM
        | EmbeddingModel::SnowflakeArcticEmbedMQ
        | EmbeddingModel::SnowflakeArcticEmbedMLong
        | EmbeddingModel::SnowflakeArcticEmbedMLongQ
        | EmbeddingModel::SnowflakeArcticEmbedL
        | EmbeddingModel::SnowflakeArcticEmbedLQ => (Some(BGE_EN_INSTRUCTION), None),
        EmbeddingModel::BGESmallZHV15 | EmbeddingModel::BGELargeZHV15 => {
            (Some(BGE_ZH_INSTRUCTION), None)
        }
        // Symmetric models (MiniLM, MPNet, paraphrase-*, GTE, CLIP, Jina v2)
        // plus BGE-m3 (no query instruction per FlagEmbedding) get no prefixes.
        // This arm also covers models fastembed adds in the future: defaulting
        // to unprefixed is the safe choice, since unknown-name models fall back
        // to AllMiniLML6V2 (symmetric) anyway.
        _ => (None, None),
    }
}

fn apply_prefix(prefix: Option<&str>, text: &str) -> String {
    match prefix {
        Some(p) => format!("{p}{text}"),
        None => text.to_string(),
    }
}

impl EmbeddingService {
    pub fn new(model_name: &str, cache_dir: &str) -> anyhow::Result<Self> {
        // Map model name to EmbeddingModel enum
        let embedding_model = match model_name {
            "sentence-transformers/all-MiniLM-L6-v2" => EmbeddingModel::AllMiniLML6V2,
            "Quantized sentence-transformers/all-MiniLM-L6-v2" => EmbeddingModel::AllMiniLML6V2Q,
            "sentence-transformers/all-MiniLM-L12-v2" => EmbeddingModel::AllMiniLML12V2,
            "Quantized sentence-transformers/all-MiniLM-L12-v2" => EmbeddingModel::AllMiniLML12V2Q,
            "sentence-transformers/all-mpnet-base-v2" => EmbeddingModel::AllMpnetBaseV2,
            "BAAI/bge-base-en-v1.5" => EmbeddingModel::BGEBaseENV15,
            "Quantized BAAI/bge-base-en-v1.5" => EmbeddingModel::BGEBaseENV15Q,
            "BAAI/bge-large-en-v1.5" => EmbeddingModel::BGELargeENV15,
            "Quantized BAAI/bge-large-en-v1.5" => EmbeddingModel::BGELargeENV15Q,
            "BAAI/bge-small-en-v1.5 - Default" => EmbeddingModel::BGESmallENV15,
            "Xenova/bge-small-en-v1.5" => EmbeddingModel::BGESmallENV15,
            "Quantized BAAI/bge-small-en-v1.5" => EmbeddingModel::BGESmallENV15Q,
            "nomic-ai/nomic-embed-text-v1" => EmbeddingModel::NomicEmbedTextV1,
            "nomic-ai/nomic-embed-text-v1.5" => EmbeddingModel::NomicEmbedTextV15,
            "Quantized v1.5 nomic-ai/nomic-embed-text-v1.5" => EmbeddingModel::NomicEmbedTextV15Q,
            "sentence-transformers/paraphrase-MiniLM-L6-v2" => {
                EmbeddingModel::ParaphraseMLMiniLML12V2
            }
            "Quantized sentence-transformers/paraphrase-MiniLM-L6-v2" => {
                EmbeddingModel::ParaphraseMLMiniLML12V2Q
            }
            "sentence-transformers/paraphrase-mpnet-base-v2" => {
                EmbeddingModel::ParaphraseMLMpnetBaseV2
            }
            "BAAI/bge-small-zh-v1.5" => EmbeddingModel::BGESmallZHV15,
            "BAAI/bge-large-zh-v1.5" => EmbeddingModel::BGELargeZHV15,
            "BAAI/bge-m3" => EmbeddingModel::BGEM3,
            "lightonai/modernbert-embed-large" => EmbeddingModel::ModernBertEmbedLarge,
            "intfloat/multilingual-e5-small" => EmbeddingModel::MultilingualE5Small,
            "intfloat/multilingual-e5-base" => EmbeddingModel::MultilingualE5Base,
            "intfloat/multilingual-e5-large" => EmbeddingModel::MultilingualE5Large,
            "mixedbread-ai/mxbai-embed-large-v1" => EmbeddingModel::MxbaiEmbedLargeV1,
            "Quantized mixedbread-ai/mxbai-embed-large-v1" => EmbeddingModel::MxbaiEmbedLargeV1Q,
            "Alibaba-NLP/gte-base-en-v1.5" => EmbeddingModel::GTEBaseENV15,
            "Quantized Alibaba-NLP/gte-base-en-v1.5" => EmbeddingModel::GTEBaseENV15Q,
            "Alibaba-NLP/gte-large-en-v1.5" => EmbeddingModel::GTELargeENV15,
            "Quantized Alibaba-NLP/gte-large-en-v1.5" => EmbeddingModel::GTELargeENV15Q,
            "Qdrant/clip-ViT-B-32-text" => EmbeddingModel::ClipVitB32,
            "jinaai/jina-embeddings-v2-base-code" => EmbeddingModel::JinaEmbeddingsV2BaseCode,
            "jinaai/jina-embeddings-v2-base-en" => EmbeddingModel::JinaEmbeddingsV2BaseEN,
            "onnx-community/embeddinggemma-300m-ONNX" => EmbeddingModel::EmbeddingGemma300M,
            "Quantized (4-bit) onnx-community/embeddinggemma-300m-ONNX" => {
                EmbeddingModel::EmbeddingGemma300MQ4
            }
            "Quantized onnx-community/embeddinggemma-300m-ONNX" => {
                EmbeddingModel::EmbeddingGemma300MQ
            }
            "snowflake/snowflake-arctic-embed-xs" => EmbeddingModel::SnowflakeArcticEmbedXS,
            "Quantized snowflake/snowflake-arctic-embed-xs" => {
                EmbeddingModel::SnowflakeArcticEmbedXSQ
            }
            "snowflake/snowflake-arctic-embed-s" => EmbeddingModel::SnowflakeArcticEmbedS,
            "Quantized snowflake/snowflake-arctic-embed-s" => {
                EmbeddingModel::SnowflakeArcticEmbedSQ
            }
            "snowflake/snowflake-arctic-embed-m" => EmbeddingModel::SnowflakeArcticEmbedM,
            "Quantized snowflake/snowflake-arctic-embed-m" => {
                EmbeddingModel::SnowflakeArcticEmbedMQ
            }
            "snowflake/snowflake-arctic-embed-m-long" => EmbeddingModel::SnowflakeArcticEmbedMLong,
            "Quantized snowflake/snowflake-arctic-embed-m-long" => {
                EmbeddingModel::SnowflakeArcticEmbedMLongQ
            }
            "snowflake/snowflake-arctic-embed-l" => EmbeddingModel::SnowflakeArcticEmbedL,
            "Quantized snowflake/snowflake-arctic-embed-l" => {
                EmbeddingModel::SnowflakeArcticEmbedLQ
            }
            _ => {
                // Default to AllMiniLML6V2 if model not found
                tracing::warn!("Unknown model: {}, using AllMiniLML6V2", model_name);
                EmbeddingModel::AllMiniLML6V2
            }
        };

        let (query_prefix, passage_prefix) = prefixes_for(&embedding_model);

        let mut model = TextEmbedding::try_new(
            TextInitOptions::new(embedding_model).with_cache_dir(cache_dir.into()),
        )?;

        // Get dimensions by embedding a dummy text
        let test_embedding = model.embed(vec!["test"], None)?;
        let dimensions = if !test_embedding.is_empty() {
            test_embedding[0].len()
        } else {
            384 // default
        };

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            model_name: model_name.to_string(),
            dimensions,
            query_prefix,
            passage_prefix,
        })
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Instruction prepended to search queries for this model, if the model's
    /// training requires one (see [`prefixes_for`]).
    pub fn query_prefix(&self) -> Option<&'static str> {
        self.query_prefix
    }

    /// Instruction prepended to indexed document text for this model, if the
    /// model's training requires one (see [`prefixes_for`]).
    pub fn passage_prefix(&self) -> Option<&'static str> {
        self.passage_prefix
    }

    pub fn list_models() -> Vec<String> {
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "Quantized sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L12-v2",
            "Quantized sentence-transformers/all-MiniLM-L12-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "BAAI/bge-base-en-v1.5",
            "Quantized BAAI/bge-base-en-v1.5",
            "BAAI/bge-large-en-v1.5",
            "Quantized BAAI/bge-large-en-v1.5",
            "BAAI/bge-small-en-v1.5 - Default",
            "Quantized BAAI/bge-small-en-v1.5",
            "nomic-ai/nomic-embed-text-v1",
            "nomic-ai/nomic-embed-text-v1.5",
            "Quantized v1.5 nomic-ai/nomic-embed-text-v1.5",
            "sentence-transformers/paraphrase-MiniLM-L6-v2",
            "Quantized sentence-transformers/paraphrase-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-mpnet-base-v2",
            "BAAI/bge-small-zh-v1.5",
            "BAAI/bge-large-zh-v1.5",
            "BAAI/bge-m3",
            "lightonai/modernbert-embed-large",
            "intfloat/multilingual-e5-small",
            "intfloat/multilingual-e5-base",
            "intfloat/multilingual-e5-large",
            "mixedbread-ai/mxbai-embed-large-v1",
            "Quantized mixedbread-ai/mxbai-embed-large-v1",
            "Alibaba-NLP/gte-base-en-v1.5",
            "Quantized Alibaba-NLP/gte-base-en-v1.5",
            "Alibaba-NLP/gte-large-en-v1.5",
            "Quantized Alibaba-NLP/gte-large-en-v1.5",
            "Qdrant/clip-ViT-B-32-text",
            "jinaai/jina-embeddings-v2-base-code",
            "jinaai/jina-embeddings-v2-base-en",
            "onnx-community/embeddinggemma-300m-ONNX",
            "Quantized (4-bit) onnx-community/embeddinggemma-300m-ONNX",
            "Quantized onnx-community/embeddinggemma-300m-ONNX",
            "snowflake/snowflake-arctic-embed-xs",
            "Quantized snowflake/snowflake-arctic-embed-xs",
            "snowflake/snowflake-arctic-embed-s",
            "Quantized snowflake/snowflake-arctic-embed-s",
            "snowflake/snowflake-arctic-embed-m",
            "Quantized snowflake/snowflake-arctic-embed-m",
            "snowflake/snowflake-arctic-embed-m-long",
            "Quantized snowflake/snowflake-arctic-embed-m-long",
            "snowflake/snowflake-arctic-embed-l",
            "Quantized snowflake/snowflake-arctic-embed-l",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Embeds document chunks for indexing. Applies the model's passage-side
    /// prefix, if any (see [`prefixes_for`]).
    pub async fn embed_chunks(&self, chunks: &[DocumentChunk]) -> anyhow::Result<Vec<Vec<f32>>> {
        let texts: Vec<String> = chunks
            .iter()
            .map(|c| apply_prefix(self.passage_prefix, &c.text))
            .collect();
        let model = self.model.clone();

        let embeddings_data = tokio::task::spawn_blocking(move || {
            let mut guard = model.blocking_lock();
            guard.embed(texts, None)
        })
        .await??;

        Ok(embeddings_data)
    }

    /// Embeds a search query. Applies the model's query-side instruction, if
    /// any (see [`prefixes_for`]) — asymmetric models (E5, BGE, Nomic, ...)
    /// embed queries differently from passages, so the raw query must not go
    /// through [`Self::embed_chunks`].
    pub async fn embed_query(&self, query: &str) -> anyhow::Result<Vec<f32>> {
        let text = apply_prefix(self.query_prefix, query);
        let model = self.model.clone();

        let mut embeddings_data = tokio::task::spawn_blocking(move || {
            let mut guard = model.blocking_lock();
            guard.embed(vec![text], None)
        })
        .await??;

        embeddings_data
            .pop()
            .ok_or_else(|| anyhow::anyhow!("Failed to generate query embedding"))
    }
}

impl std::fmt::Debug for EmbeddingService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingService")
            .field("model_name", &self.model_name)
            .field("dimensions", &self.dimensions)
            .field("query_prefix", &self.query_prefix)
            .field("passage_prefix", &self.passage_prefix)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn test_list_models() {
        let models = EmbeddingService::list_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"BAAI/bge-small-en-v1.5 - Default".to_string()));
    }

    #[test]
    #[serial]
    fn test_new_invalid_model() {
        // fastembed uses default model regardless of input
        let svc = EmbeddingService::new("Xenova/bge-small-en-v1.5", DEFAULT_CACHE_DIR).unwrap();
        assert!(svc.dimensions() > 0);
    }

    #[test]
    #[serial]
    fn test_new_valid_model() {
        let svc = EmbeddingService::new("Xenova/bge-small-en-v1.5", DEFAULT_CACHE_DIR).unwrap();
        assert!(svc.dimensions() > 0);
    }

    #[tokio::test]
    #[serial]
    async fn test_embed_chunks() {
        let svc = EmbeddingService::new("Xenova/bge-small-en-v1.5", DEFAULT_CACHE_DIR).unwrap();
        let chunks = vec![DocumentChunk {
            id: "1".to_string(),
            text: "hello world".to_string(),
            source: "test.txt".to_string(),
            chunk_index: 0,
            start_offset: 0,
            end_offset: 11,
        }];
        let embeddings = svc.embed_chunks(&chunks).await.unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), svc.dimensions());
    }

    const DEFAULT_CACHE_DIR: &str = ".fastembed_cache";

    const BGE_EN_INSTRUCTION: &str = "Represent this sentence for searching relevant passages: ";
    const BGE_ZH_INSTRUCTION: &str = "为这个句子生成表示以用于检索相关文章：";

    /// The prefix table is a pure function of the resolved model, so every
    /// family can be asserted without downloading a model.
    #[test]
    fn prefix_table_asymmetric_families() {
        use EmbeddingModel as M;

        // E5: canonical query/passage prefixes.
        for m in [
            M::MultilingualE5Small,
            M::MultilingualE5Base,
            M::MultilingualE5Large,
        ] {
            assert_eq!(
                prefixes_for(&m),
                (Some("query: "), Some("passage: ")),
                "E5 variant {m:?}"
            );
        }

        // Nomic + ModernBERT-embed: trained like Nomic (card says REQUIRED).
        for m in [
            M::NomicEmbedTextV1,
            M::NomicEmbedTextV15,
            M::NomicEmbedTextV15Q,
            M::ModernBertEmbedLarge,
        ] {
            assert_eq!(
                prefixes_for(&m),
                (Some("search_query: "), Some("search_document: ")),
                "Nomic-style variant {m:?}"
            );
        }

        // EmbeddingGemma: task/title prompt pair.
        for m in [
            M::EmbeddingGemma300M,
            M::EmbeddingGemma300MQ,
            M::EmbeddingGemma300MQ4,
        ] {
            assert_eq!(
                prefixes_for(&m),
                (
                    Some("task: search result | query: "),
                    Some("title: none | text: ")
                ),
                "EmbeddingGemma variant {m:?}"
            );
        }

        // BGE en v1.5 (+quantized), mxbai, Snowflake arctic: query-side
        // instruction only, passages are never prefixed.
        for m in [
            M::BGEBaseENV15,
            M::BGEBaseENV15Q,
            M::BGELargeENV15,
            M::BGELargeENV15Q,
            M::BGESmallENV15,
            M::BGESmallENV15Q,
            M::MxbaiEmbedLargeV1,
            M::MxbaiEmbedLargeV1Q,
            M::SnowflakeArcticEmbedXS,
            M::SnowflakeArcticEmbedXSQ,
            M::SnowflakeArcticEmbedS,
            M::SnowflakeArcticEmbedSQ,
            M::SnowflakeArcticEmbedM,
            M::SnowflakeArcticEmbedMQ,
            M::SnowflakeArcticEmbedMLong,
            M::SnowflakeArcticEmbedMLongQ,
            M::SnowflakeArcticEmbedL,
            M::SnowflakeArcticEmbedLQ,
        ] {
            assert_eq!(
                prefixes_for(&m),
                (Some(BGE_EN_INSTRUCTION), None),
                "BGE-instruction variant {m:?}"
            );
        }

        // BGE zh v1.5: Chinese instruction, passages never prefixed.
        for m in [M::BGESmallZHV15, M::BGELargeZHV15] {
            assert_eq!(
                prefixes_for(&m),
                (Some(BGE_ZH_INSTRUCTION), None),
                "BGE-zh variant {m:?}"
            );
        }
    }

    #[test]
    fn prefix_table_symmetric_families_get_nothing() {
        use EmbeddingModel as M;

        for m in [
            M::AllMiniLML6V2,
            M::AllMiniLML6V2Q,
            M::AllMiniLML12V2,
            M::AllMiniLML12V2Q,
            M::AllMpnetBaseV2,
            M::ParaphraseMLMiniLML12V2,
            M::ParaphraseMLMiniLML12V2Q,
            M::ParaphraseMLMpnetBaseV2,
            M::BGEM3, // no query instruction per FlagEmbedding docs
            M::GTEBaseENV15,
            M::GTEBaseENV15Q,
            M::GTELargeENV15,
            M::GTELargeENV15Q,
            M::ClipVitB32,
            M::JinaEmbeddingsV2BaseEN,
            M::JinaEmbeddingsV2BaseCode,
        ] {
            assert_eq!(prefixes_for(&m), (None, None), "symmetric variant {m:?}");
        }
    }

    #[test]
    fn apply_prefix_semantics() {
        assert_eq!(apply_prefix(None, "hello"), "hello");
        assert_eq!(
            apply_prefix(Some("query: "), "hello"),
            "query: hello",
            "prefix must be prepended verbatim"
        );
        assert_eq!(apply_prefix(Some("p: "), ""), "p: ");
    }

    #[tokio::test]
    #[serial]
    async fn test_embed_query_applies_model_instruction() {
        let svc = EmbeddingService::new("Xenova/bge-small-en-v1.5", DEFAULT_CACHE_DIR).unwrap();

        // BGE en: query gets the instruction, passages stay bare.
        assert_eq!(svc.query_prefix(), Some(BGE_EN_INSTRUCTION));
        assert_eq!(svc.passage_prefix(), None);

        // embed_query(x) must equal embedding the manually-instructed string
        // through the (unprefixed) document path — proving the instruction is
        // actually applied, not just stored.
        let via_query_api = svc.embed_query("hello world").await.unwrap();
        let chunks = vec![DocumentChunk {
            id: "1".to_string(),
            text: format!("{BGE_EN_INSTRUCTION}hello world"),
            source: "test.txt".to_string(),
            chunk_index: 0,
            start_offset: 0,
            end_offset: 0,
        }];
        let via_document_api = svc.embed_chunks(&chunks).await.unwrap();

        assert_eq!(via_query_api.len(), svc.dimensions());
        assert_eq!(via_query_api, via_document_api[0]);
    }

    #[tokio::test]
    #[serial]
    async fn test_embed_query_empty_string_ok() {
        let svc = EmbeddingService::new("Xenova/bge-small-en-v1.5", DEFAULT_CACHE_DIR).unwrap();
        let embedding = svc.embed_query("").await.unwrap();
        assert_eq!(embedding.len(), svc.dimensions());
        assert!(embedding.iter().all(|v| v.is_finite()));
    }
}
