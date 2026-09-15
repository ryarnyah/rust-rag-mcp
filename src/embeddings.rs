use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use tokio::sync::Mutex;

use crate::DocumentChunk;

pub struct EmbeddingService {
    model: Mutex<TextEmbedding>,
    model_name: String,
    dimensions: usize,
}

impl EmbeddingService {
    pub fn new(model_name: &str) -> anyhow::Result<Self> {
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

        let mut model = TextEmbedding::try_new(TextInitOptions::new(embedding_model))?;

        // Get dimensions by embedding a dummy text
        let test_embedding = model.embed(vec!["test"], None)?;
        let dimensions = if !test_embedding.is_empty() {
            test_embedding[0].len()
        } else {
            384 // default
        };

        Ok(Self {
            model: Mutex::new(model),
            model_name: model_name.to_string(),
            dimensions,
        })
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn get_model(&self) -> &Mutex<TextEmbedding> {
        &self.model
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

    pub async fn embed_chunks(&self, chunks: Vec<DocumentChunk>) -> anyhow::Result<Vec<Vec<f32>>> {
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();

        let embeddings_data = self.model.lock().await.embed(texts, None)?;
        Ok(embeddings_data)
    }
}

impl std::fmt::Debug for EmbeddingService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingService")
            .field("model_name", &self.model_name)
            .field("dimensions", &self.dimensions)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_models() {
        let models = EmbeddingService::list_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"Xenova/bge-small-en-v1.5".to_string()));
    }

    #[test]
    fn test_new_invalid_model() {
        // fastembed uses default model regardless of input
        let svc = EmbeddingService::new("Xenova/bge-small-en-v1.5").unwrap();
        assert!(svc.dimensions() > 0);
    }

    #[test]
    fn test_new_valid_model() {
        let svc = EmbeddingService::new("Xenova/bge-small-en-v1.5").unwrap();
        assert!(svc.dimensions() > 0);
    }

    #[test]
    fn test_embed_chunks() {
        let svc = EmbeddingService::new("Xenova/bge-small-en-v1.5").unwrap();
        let chunks = vec![DocumentChunk {
            id: "1".to_string(),
            text: "hello world".to_string(),
            source: "test.txt".to_string(),
            chunk_index: 0,
            start_offset: 0,
            end_offset: 11,
        }];
        let embeddings = svc.embed_chunks(chunks).unwrap();
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), svc.dimensions());
    }
}
