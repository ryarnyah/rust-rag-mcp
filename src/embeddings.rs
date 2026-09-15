use fastembed::TextEmbedding;
use tokio::sync::Mutex;

use crate::DocumentChunk;

pub struct EmbeddingService {
    model: Mutex<TextEmbedding>,
    model_name: String,
    dimensions: usize,
}

impl EmbeddingService {
    pub fn new(model_name: &str) -> anyhow::Result<Self> {
        let mut model = TextEmbedding::try_new(Default::default())?;
        
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
            "Xenova/all-MiniLM-L6-v2",
            "Qdrant/all-MiniLM-L6-v2-onnx",
            "Xenova/all-MiniLM-L12-v2",
            "Xenova/all-mpnet-base-v2",
            "Xenova/bge-base-en-v1.5",
            "Qdrant/bge-base-en-v1.5-onnx-Q",
            "Xenova/bge-large-en-v1.5",
            "Qdrant/bge-large-en-v1.5-onnx-Q",
            "Xenova/bge-small-en-v1.5",
            "Qdrant/bge-small-en-v1.5-onnx-Q",
            "nomic-ai/nomic-embed-text-v1",
            "nomic-ai/nomic-embed-text-v1.5",
            "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
            "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q",
            "Xenova/paraphrase-multilingual-mpnet-base-v2",
            "Xenova/bge-small-zh-v1.5",
            "Xenova/bge-large-zh-v1.5",
            "lightonai/modernbert-embed-large",
            "intfloat/multilingual-e5-small",
            "intfloat/multilingual-e5-base",
            "Qdrant/multilingual-e5-large-onnx",
            "mixedbread-ai/mxbai-embed-large-v1",
            "Alibaba-NLP/gte-base-en-v1.5",
            "Alibaba-NLP/gte-large-en-v1.5",
            "Qdrant/clip-ViT-B-32-text",
            "jinaai/jina-embeddings-v2-base-code",
            "jinaai/jina-embeddings-v2-base-en",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    pub async fn embed_chunks(
        &self,
        chunks: Vec<DocumentChunk>,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
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
