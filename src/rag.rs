use crate::chunker::Chunker;
use crate::docs;
use crate::embeddings::EmbeddingService;
use crate::{DocumentChunk, DocumentStatus, IndexResult, SearchResult};
use anyhow::Result;
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use sha2::{Digest, Sha256};
use std::path::Path;

use lancedb::arrow::arrow_array::{Array, RecordBatch, StringArray, UInt32Array, UInt64Array, Float32Array};
use lancedb::arrow::arrow_schema::{DataType, Field};
use std::sync::Arc;

pub struct RagCore {
    table: lancedb::Table,
    metadata_table: lancedb::Table,
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

        let metadata_schema = Arc::new(lancedb::arrow::arrow_schema::Schema::new(vec![
            Field::new("source_path", DataType::Utf8, false),
            Field::new("content_hash", DataType::Utf8, true),
            Field::new("indexed_at", DataType::Utf8, true),
            Field::new("chunk_count", DataType::UInt32, true),
        ]));

        let metadata_table_name = "document_metadata";
        let metadata_table = match db.open_table(metadata_table_name).execute().await {
            Ok(t) => t,
            Err(_) => {
                db.create_empty_table(metadata_table_name, metadata_schema)
                    .execute()
                    .await?
            }
        };

        Ok(Self {
            table,
            metadata_table,
            embedding,
            chunker: Chunker::new(chunk_size, overlap),
        })
    }

    async fn compute_file_hash(path: &Path) -> Result<String> {
        let bytes = tokio::fs::read(path).await?;
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        Ok(hex::encode(hasher.finalize()))
    }

    fn compute_text_hash(text: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        hex::encode(hasher.finalize())
    }

    pub async fn index_file(&self, path: &Path) -> Result<IndexResult> {
        let source_path = path
            .canonicalize()
            .unwrap_or_else(|_| path.to_path_buf())
            .to_string_lossy()
            .to_string();

        let content_hash = Self::compute_file_hash(path).await?;

        if let Some(status) = self.document_status(&source_path).await? {
            if status.content_hash == content_hash {
                return Ok(IndexResult::Skipped);
            }
            self.delete_source(&source_path).await?;
        }

        let text = docs::extract_text(path).await?;
        let chunks = self.chunker.chunk_text(&text, &source_path);
        let count = chunks.len();
        self.index_chunks(chunks).await?;

        let now = chrono_free_timestamp();
        self.upsert_metadata(&source_path, &content_hash, &now, count as u32)
            .await?;

        Ok(IndexResult::Indexed(count))
    }

    pub async fn index_text(&self, text: &str, source: &str) -> Result<IndexResult> {
        let content_hash = Self::compute_text_hash(text);

        if let Some(status) = self.document_status(source).await? {
            if status.content_hash == content_hash {
                return Ok(IndexResult::Skipped);
            }
            self.delete_source(source).await?;
        }

        let chunks = self.chunker.chunk_text(text, source);
        let count = chunks.len();
        self.index_chunks(chunks).await?;

        let now = chrono_free_timestamp();
        self.upsert_metadata(source, &content_hash, &now, count as u32)
            .await?;

        Ok(IndexResult::Indexed(count))
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

        use lancedb::arrow::arrow_array::types::Float32Type;

        let vectors = lancedb::arrow::arrow_array::FixedSizeListArray::from_iter_primitive::<
            Float32Type,
            _,
            _,
        >(
            embeddings_result
                .iter()
                .map(|embedding| Some(embedding.iter().map(|v| Some(*v)).collect::<Vec<_>>())),
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

        let mut vector_query = self.table.query().nearest_to(query_embedding)?;

        if let Some(filter) = source_filter {
            vector_query = vector_query.only_if(format!("source = '{}'", filter));
        }

        vector_query = vector_query.limit(top_k);

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
                .and_then(|col| col.as_any().downcast_ref::<UInt32Array>());

            let start_offsets = batch
                .column_by_name("start_offset")
                .and_then(|col| col.as_any().downcast_ref::<UInt64Array>());

            let end_offsets = batch
                .column_by_name("end_offset")
                .and_then(|col| col.as_any().downcast_ref::<UInt64Array>());

            let distances = batch
                .column_by_name("_distance")
                .and_then(|col| col.as_any().downcast_ref::<Float32Array>());

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
                    let score = if let Some(dists) = distances {
                        let distance = dists.value(i) as f64;
                        1.0 / (1.0 + distance)
                    } else {
                        0.5
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

    pub async fn delete_source(&self, source_path: &str) -> Result<()> {
        self.table
            .delete(&format!("source = '{}'", source_path))
            .await?;
        self.metadata_table
            .delete(&format!("source_path = '{}'", source_path))
            .await?;
        Ok(())
    }

    pub async fn document_status(&self, source_path: &str) -> Result<Option<DocumentStatus>> {
        let results: Vec<RecordBatch> = self
            .metadata_table
            .query()
            .only_if(format!("source_path = '{}'", source_path))
            .execute()
            .await?
            .try_collect()
            .await?;

        for batch in &results {
            let source_paths = batch
                .column_by_name("source_path")
                .and_then(|col| col.as_any().downcast_ref::<StringArray>());
            let hashes = batch
                .column_by_name("content_hash")
                .and_then(|col| col.as_any().downcast_ref::<StringArray>());
            let timestamps = batch
                .column_by_name("indexed_at")
                .and_then(|col| col.as_any().downcast_ref::<StringArray>());
            let counts = batch
                .column_by_name("chunk_count")
                .and_then(|col| col.as_any().downcast_ref::<UInt32Array>());

            if let (Some(sp), Some(h), Some(t), Some(c)) =
                (source_paths, hashes, timestamps, counts)
            {
                if sp.len() > 0 {
                    return Ok(Some(DocumentStatus {
                        source_path: sp.value(0).to_string(),
                        content_hash: h.value(0).to_string(),
                        indexed_at: t.value(0).to_string(),
                        chunk_count: c.value(0),
                    }));
                }
            }
        }
        Ok(None)
    }

    async fn upsert_metadata(
        &self,
        source_path: &str,
        content_hash: &str,
        indexed_at: &str,
        chunk_count: u32,
    ) -> Result<()> {
        self.metadata_table
            .delete(&format!("source_path = '{}'", source_path))
            .await?;

        let batch = RecordBatch::try_new(
            Arc::new(lancedb::arrow::arrow_schema::Schema::new(vec![
                Field::new("source_path", DataType::Utf8, false),
                Field::new("content_hash", DataType::Utf8, true),
                Field::new("indexed_at", DataType::Utf8, true),
                Field::new("chunk_count", DataType::UInt32, true),
            ])),
            vec![
                Arc::new(StringArray::from(vec![source_path])),
                Arc::new(StringArray::from(vec![content_hash])),
                Arc::new(StringArray::from(vec![indexed_at])),
                Arc::new(UInt32Array::from(vec![chunk_count])),
            ],
        )?;

        self.metadata_table.add(batch).execute().await?;
        Ok(())
    }

    pub fn list_embedding_models() -> Vec<String> {
        EmbeddingService::list_models()
    }
}

fn chrono_free_timestamp() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| format!("{}", d.as_secs()))
        .unwrap_or_else(|_| "0".to_string())
}
