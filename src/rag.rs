use crate::chunker::Chunker;
use crate::docs;
use crate::embeddings::EmbeddingService;
use crate::{DocumentChunk, SearchResult};
use anyhow::Result;
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use std::path::Path;

// Import arrow types from lancedb's re-export
use lancedb::arrow::arrow_array::{Array, RecordBatch, StringArray, UInt32Array, UInt64Array};
use lancedb::arrow::arrow_schema::{DataType, Field};
use std::sync::Arc;

pub struct RagCore {
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

        let schema = Arc::new(lancedb::arrow::arrow_schema::Schema::new(vec![
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
        ]));

        let table = match db.open_table(table_name).execute().await {
            Ok(t) => t,
            Err(_) => db.create_empty_table(table_name, schema).execute().await?,
        };

        Ok(Self {
            table,
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

        let embeddings_result = self.embedding.embed_chunks(chunks.clone()).await?;
        let ndims = self.embedding.dimensions() as i32;

        let ids: Vec<&str> = chunks.iter().map(|c| c.id.as_str()).collect();
        let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
        let sources: Vec<&str> = chunks.iter().map(|c| c.source.as_str()).collect();
        let chunk_indices: Vec<u32> = chunks.iter().map(|c| c.chunk_index).collect();
        let start_offsets: Vec<u64> = chunks.iter().map(|c| c.start_offset as u64).collect();
        let end_offsets: Vec<u64> = chunks.iter().map(|c| c.end_offset as u64).collect();

        // Build vectors using from_iter_primitive
        use lancedb::arrow::arrow_array::types::Float32Type;

        let vectors = lancedb::arrow::arrow_array::FixedSizeListArray::from_iter_primitive::<
            Float32Type,
            _,
            _,
        >(
            embeddings_result.iter().map(|embedding| {
                Some(
                    embedding
                        .iter()
                        .map(|v| Some(*v as f32))
                        .collect::<Vec<_>>(),
                )
            }),
            ndims,
        );

        let batch = RecordBatch::try_new(
            Arc::new(lancedb::arrow::arrow_schema::Schema::new(vec![
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
                Arc::new(UInt32Array::from(chunk_indices)),
                Arc::new(UInt64Array::from(start_offsets)),
                Arc::new(UInt64Array::from(end_offsets)),
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
        // Generate query embedding
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

        let query_embedding = query_embedding_vec[0].clone();

        // Perform vector search using lancedb's nearest_to
        let mut vector_query = self.table.query().nearest_to(query_embedding)?;

        // Apply source filter if provided
        if let Some(filter) = source_filter {
            vector_query = vector_query.only_if(format!("source = '{}'", filter));
        }

        // Limit results
        vector_query = vector_query.limit(top_k);

        // Execute the query
        let results: Vec<RecordBatch> = vector_query.execute().await?.try_collect().await?;

        let mut search_results = Vec::new();
        for batch in results {
            let ids = batch
                .column_by_name("id")
                .and_then(|col| col.as_any().downcast_ref::<StringArray>());

            let texts = batch
                .column_by_name("text")
                .and_then(|col| col.as_any().downcast_ref::<StringArray>());

            let sources = batch
                .column_by_name("source")
                .and_then(|col| col.as_any().downcast_ref::<StringArray>());

            let chunk_indices = batch
                .column_by_name("chunk_index")
                .and_then(|col| col.as_any().downcast_ref::<arrow_array::UInt32Array>());

            let start_offsets = batch
                .column_by_name("start_offset")
                .and_then(|col| col.as_any().downcast_ref::<arrow_array::UInt64Array>());

            let end_offsets = batch
                .column_by_name("end_offset")
                .and_then(|col| col.as_any().downcast_ref::<arrow_array::UInt64Array>());

            // Extract similarity scores from _distance column
            let distances = batch
                .column_by_name("_distance")
                .and_then(|col| col.as_any().downcast_ref::<arrow_array::Float32Array>());

            if let (
                Some(ids),
                Some(texts),
                Some(sources),
                Some(indices),
                Some(starts),
                Some(ends),
            ) = (
                ids,
                texts,
                sources,
                chunk_indices,
                start_offsets,
                end_offsets,
            ) {
                for i in 0..ids.len() {
                    // Convert distance to similarity (1 / (1 + distance) for cosine)
                    let score = if let Some(dists) = distances {
                        let distance = dists.value(i) as f64;
                        // For cosine distance, convert to similarity
                        1.0 / (1.0 + distance)
                    } else {
                        0.5 // Default if no distance available
                    };

                    let chunk = DocumentChunk {
                        id: ids.value(i).to_string(),
                        text: texts.value(i).to_string(),
                        source: sources.value(i).to_string(),
                        chunk_index: indices.value(i),
                        start_offset: starts.value(i) as usize,
                        end_offset: ends.value(i) as usize,
                    };
                    search_results.push(SearchResult { score, chunk });
                }
            }
        }

        Ok(search_results)
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
