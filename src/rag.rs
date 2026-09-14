use crate::chunker::Chunker;
use crate::docs;
use crate::embeddings::EmbeddingService;
use crate::{DocumentChunk, SearchResult};
use anyhow::Result;
use arrow_array::{types::Float32Type, Array, FixedSizeListArray, RecordBatch, StringArray};
use arrow_schema::{DataType, Field};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use rig::lancedb::{LanceDBFilter, LanceDbVectorIndex, SearchParams};
use rig::vector_store::request::SearchFilter;
use rig::vector_store::VectorStoreIndex;
use std::path::Path;
use std::sync::Arc;

pub struct RagCore {
    index: LanceDbVectorIndex<rig_fastembed::EmbeddingModel>,
    table: lancedb::Table,
    embedding: EmbeddingService,
    chunker: Chunker,
}

impl RagCore {
    pub async fn new(
        db_path: &str,
        model_name: &str,
        chunk_size: usize,
        overlap: usize,
    ) -> Result<Self> {
        let embedding = EmbeddingService::new(model_name)?;
        let ndims = embedding.dimensions() as i32;

        let db = lancedb::connect(db_path).execute().await?;
        let table_name = "chunks";

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("id", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("text", arrow_schema::DataType::Utf8, true),
            arrow_schema::Field::new("source", arrow_schema::DataType::Utf8, true),
            arrow_schema::Field::new("chunk_index", arrow_schema::DataType::UInt32, true),
            arrow_schema::Field::new("start_offset", arrow_schema::DataType::UInt64, true),
            arrow_schema::Field::new("end_offset", arrow_schema::DataType::UInt64, true),
            arrow_schema::Field::new(
                "vector",
                arrow_schema::DataType::FixedSizeList(
                    Arc::new(arrow_schema::Field::new(
                        "item",
                        arrow_schema::DataType::Float32,
                        true,
                    )),
                    ndims,
                ),
                true,
            ),
        ]));

        let table = match db.open_table(table_name).execute().await {
            Ok(t) => t,
            Err(_) => db.create_empty_table(table_name, schema).execute().await?,
        };

        let search_params = SearchParams::default()
            .distance_type(lancedb::DistanceType::Cosine)
            .column("vector");

        let table_clone = table.clone();
        let index =
            LanceDbVectorIndex::new(table, embedding.rig_model().clone(), "id", search_params)
                .await?;

        Ok(Self {
            index,
            table: table_clone,
            embedding,
            chunker: Chunker::new(chunk_size, overlap),
        })
    }

    pub async fn index_file(&self, path: &Path) -> Result<usize> {
        let text = docs::extract_text(path)?;
        let source = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| path.display().to_string());

        let chunks = self.chunker.chunk_text(&text, &source);
        let count = chunks.len();
        self.index_chunks(chunks).await?;
        Ok(count)
    }

    pub async fn index_text(&self, text: &str, source: &str) -> Result<usize> {
        let chunks = self.chunker.chunk_text(text, source);
        let count = chunks.len();
        self.index_chunks(chunks).await?;
        Ok(count)
    }

    async fn index_chunks(&self, chunks: Vec<DocumentChunk>) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        let embeddings_result = self.embedding.embed_chunks(chunks).await?;
        let ndims = self.embedding.dimensions() as i32;

        let ids: Vec<&str> = embeddings_result
            .iter()
            .map(|(c, _)| c.id.as_str())
            .collect();
        let texts: Vec<&str> = embeddings_result
            .iter()
            .map(|(c, _)| c.text.as_str())
            .collect();
        let sources: Vec<&str> = embeddings_result
            .iter()
            .map(|(c, _)| c.source.as_str())
            .collect();
        let chunk_indices: Vec<u32> = embeddings_result
            .iter()
            .map(|(c, _)| c.chunk_index)
            .collect();
        let start_offsets: Vec<u64> = embeddings_result
            .iter()
            .map(|(c, _)| c.start_offset as u64)
            .collect();
        let end_offsets: Vec<u64> = embeddings_result
            .iter()
            .map(|(c, _)| c.end_offset as u64)
            .collect();

        let vectors = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            embeddings_result.iter().map(|(_, es)| {
                es.first()
                    .map(|e| e.vec.iter().map(|v| Some(*v as f32)).collect::<Vec<_>>())
            }),
            ndims,
        );

        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema::Schema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new("text", DataType::Utf8, true),
                Field::new("source", DataType::Utf8, true),
                Field::new("chunk_index", DataType::UInt32, true),
                Field::new("start_offset", DataType::UInt64, true),
                Field::new("end_offset", DataType::UInt64, true),
                Field::new(
                    "vector",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        ndims,
                    ),
                    true,
                ),
            ])),
            vec![
                Arc::new(StringArray::from(ids)),
                Arc::new(StringArray::from(texts)),
                Arc::new(StringArray::from(sources)),
                Arc::new(arrow_array::UInt32Array::from(chunk_indices)),
                Arc::new(arrow_array::UInt64Array::from(start_offsets)),
                Arc::new(arrow_array::UInt64Array::from(end_offsets)),
                Arc::new(vectors),
            ],
        )?;

        self.table.add(batch).execute().await?;
        Ok(())
    }

    pub async fn search(
        &self,
        query: &str,
        top_k: usize,
        source_filter: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let req = if let Some(filter) = source_filter {
            rig::vector_store::request::VectorSearchRequest::builder()
                .query(query)
                .samples(top_k as u64)
                .filter(LanceDBFilter::eq(
                    "source",
                    serde_json::Value::String(filter.to_string()),
                ))
                .build()
        } else {
            rig::vector_store::request::VectorSearchRequest::builder()
                .query(query)
                .samples(top_k as u64)
                .build()
        };

        let results: Vec<(f64, String, DocumentChunk)> =
            self.index.top_n::<DocumentChunk>(req).await?;
        Ok(results
            .into_iter()
            .map(|(score, _id, chunk)| SearchResult { score, chunk })
            .collect())
    }

    pub async fn chunk_count(&self) -> Result<usize> {
        Ok(self.table.count_rows(None).await? as usize)
    }

    pub async fn list_sources(&self) -> Result<Vec<String>> {
        let results: Vec<RecordBatch> = self
            .table
            .query()
            .select(lancedb::query::Select::Columns(vec!["source".to_string()]))
            .execute()
            .await?
            .try_collect()
            .await?;

        let mut sources = std::collections::HashSet::new();
        for batch in &results {
            if let Some(col) = batch.column_by_name("source") {
                if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
                    for i in 0..arr.len() {
                        sources.insert(arr.value(i).to_string());
                    }
                }
            }
        }
        let mut v: Vec<String> = sources.into_iter().collect();
        v.sort();
        Ok(v)
    }

    pub fn list_embedding_models() -> Vec<String> {
        EmbeddingService::list_models()
    }
}
