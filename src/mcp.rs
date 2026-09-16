use crate::docs;
use crate::rag::RagCore;
use crate::IndexResult;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::{schemars, tool, tool_handler, tool_router, ServerHandler};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct RagServer {
    #[allow(dead_code)]
    core: Arc<RwLock<RagCore>>,
}

impl RagServer {
    pub async fn new(
        db_path: &str,
        model_name: &str,
        chunk_size: usize,
        overlap: usize,
    ) -> anyhow::Result<Self> {
        let core = RagCore::new(db_path, model_name, chunk_size, overlap).await?;
        Ok(Self {
            core: Arc::new(RwLock::new(core)),
        })
    }
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct IndexPathRequest {
    #[schemars(description = "Absolute or relative path to a file or directory to index. If a directory is given, all supported files within it (recursively) are indexed. Supported formats: PDF, DOCX, XLSX, PPTX, TXT, MD, RS, PY, JS, TS, GO, JAVA, C, CPP, H, JSON, YAML, YML, TOML, XML, CSV, HTML, CSS. Files unchanged since last index are skipped automatically.")]
    pub path: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct IndexTextRequest {
    #[schemars(description = "Raw text content to index. The text is chunked, embedded, and stored. If the same source was previously indexed with identical content, indexing is skipped.")]
    pub text: String,
    #[schemars(description = "A unique identifier for this text (e.g. 'docs/api.md', 'clipboard', or any logical name). Used as the key for deduplication — re-indexing the same source with the same text is a no-op.")]
    pub source: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchRequest {
    #[schemars(description = "Natural language search query. The query is embedded and compared against stored document chunks using cosine similarity.")]
    pub query: String,
    #[schemars(description = "Maximum number of results to return. Higher values return more candidates but take longer. Default: 5.")]
    pub top_k: Option<usize>,
    #[schemars(description = "Optional filter to restrict results to a specific source. Must match the exact source path used during indexing. Useful when searching within a single document.")]
    pub source_filter: Option<String>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct DeleteSourceRequest {
    #[schemars(description = "The full source path of the document to remove from the index. Must match the exact path used during indexing. All chunks and metadata for this source will be deleted.")]
    pub source_path: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct DocumentStatusRequest {
    #[schemars(description = "The full source path of the document to check. Must match the exact path used during indexing.")]
    pub source_path: String,
}

#[tool_router]
impl RagServer {
    #[tool(
        description = "Index a file or directory into the RAG knowledge base. If a file is given, it is read, split into chunks, embedded using a local model, and stored for semantic search. If a directory is given, all supported files within it are indexed recursively. Supported formats: PDF, DOCX, XLSX, PPTX, TXT, MD, RS, PY, JS, TS, GO, JAVA, C, CPP, H, JSON, YAML, YML, TOML, XML, CSV, HTML, CSS. Files unchanged since last index are skipped automatically. Returns a summary of indexed, skipped, and failed files."
    )]
    async fn index_path(
        &self,
        Parameters(req): Parameters<IndexPathRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let path = std::path::Path::new(&req.path);
        if !path.exists() {
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Path not found: {}",
                req.path
            ))]));
        }

        let mut stack = vec![path.to_path_buf()];
        let mut indexed = 0usize;
        let mut skipped = 0usize;
        let mut errors = Vec::new();

        while let Some(current) = stack.pop() {
            if current.is_dir() {
                if let Ok(mut entries) = tokio::fs::read_dir(&current).await {
                    while let Ok(Some(entry)) = entries.next_entry().await {
                        stack.push(entry.path());
                    }
                }
            } else if current.is_file() && docs::supported_extension(&current) {
                let result = {
                    let core = self.core.read().await;
                    core.index_file(&current).await
                };
                match result {
                    Ok(IndexResult::Indexed(count)) => {
                        indexed += 1;
                        errors.push(format!("Indexed {} — {} chunks", current.display(), count));
                    }
                    Ok(IndexResult::Skipped) => {
                        skipped += 1;
                    }
                    Err(e) => {
                        errors.push(format!("Failed {}: {}", current.display(), e));
                    }
                }
            }
        }

        let mut output = format!(
            "Done. Indexed: {}, Skipped (unchanged): {}",
            indexed, skipped
        );
        if !errors.is_empty() {
            output.push('\n');
            output.push_str(&errors.join("\n"));
        }
        Ok(CallToolResult::success(vec![ContentBlock::text(output)]))
    }

    #[tool(
        description = "Index raw text content into the RAG knowledge base. The text is split into chunks, embedded, and stored for semantic search. Use this for content that is not in a file (e.g. clipboard text, generated content, API responses). The source parameter serves as a unique key — re-indexing the same source with identical text is automatically skipped (deduplication via content hash). Returns the number of chunks created, or a skip message."
    )]
    async fn index_text(
        &self,
        Parameters(req): Parameters<IndexTextRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let result = {
            let core = self.core.read().await;
            core.index_text(&req.text, &req.source).await
        };
        match result {
            Ok(IndexResult::Indexed(count)) => {
                Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                    "Indexed '{}' — {} chunks created",
                    req.source, count
                ))]))
            }
            Ok(IndexResult::Skipped) => {
                Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                    "Skipped '{}' — content unchanged since last index",
                    req.source
                ))]))
            }
            Err(e) => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Error indexing text '{}': {}",
                req.source, e
            ))])),
        }
    }

    #[tool(
        description = "Search the RAG knowledge base using semantic similarity. The query is embedded into a vector and compared against all stored document chunks using cosine distance. Results are ranked by similarity score (0.0 to 1.0, where 1.0 is a perfect match). Each result includes the matching text chunk, its source document, chunk index, and similarity score. Use source_filter to restrict searches to a specific document."
    )]
    async fn search(
        &self,
        Parameters(req): Parameters<SearchRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let top_k = req.top_k.unwrap_or(5);
        let result = {
            let core = self.core.read().await;
            core.search(&req.query, top_k, req.source_filter.as_deref())
                .await
        };
        match result {
            Ok(results) => {
                if results.is_empty() {
                    Ok(CallToolResult::success(vec![ContentBlock::text(
                        "No results found.".to_string(),
                    )]))
                } else {
                    let mut output = String::new();
                    for (i, result) in results.iter().enumerate() {
                        output.push_str(&format!(
                            "[{}] (score: {:.4}) [{}:{}] {}\n\n",
                            i + 1,
                            result.score,
                            result.chunk.source,
                            result.chunk.chunk_index,
                            result.chunk.text,
                        ));
                    }
                    Ok(CallToolResult::success(vec![ContentBlock::text(output)]))
                }
            }
            Err(e) => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Error searching: {}",
                e
            ))])),
        }
    }

    #[tool(
        description = "List all source documents currently indexed in the knowledge base. Returns one source path per line. These are the full absolute paths of files or logical source names used during indexing. Use this to see what is available for search or to get source paths for the document_status or delete_source tools."
    )]
    async fn list_sources(&self) -> Result<CallToolResult, rmcp::ErrorData> {
        let result = {
            let core = self.core.read().await;
            core.list_sources().await
        };
        match result {
            Ok(sources) => {
                if sources.is_empty() {
                    Ok(CallToolResult::success(vec![ContentBlock::text(
                        "No sources indexed.".to_string(),
                    )]))
                } else {
                    Ok(CallToolResult::success(vec![ContentBlock::text(
                        sources.join("\n"),
                    )]))
                }
            }
            Err(e) => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Error listing sources: {}",
                e
            ))])),
        }
    }

    #[tool(
        description = "Get the total number of indexed chunks across all documents. Each document is split into multiple chunks during indexing. This count reflects the total number of searchable units in the knowledge base."
    )]
    async fn chunk_count(&self) -> Result<CallToolResult, rmcp::ErrorData> {
        let result = {
            let core = self.core.read().await;
            core.chunk_count().await
        };
        match result {
            Ok(count) => Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "{}",
                count
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Error getting chunk count: {}",
                e
            ))])),
        }
    }

    #[tool(
        description = "List all available embedding models that can be used with this server. Models are local (no API keys needed) and run via ONNX. Examples: Xenova/bge-small-en-v1.5 (default, fast, good quality), sentence-transformers/all-MiniLM-L6-v2 (fastest), Xenova/bge-large-en-v1.5 (higher quality, slower). The model is set at server startup via the --model flag."
    )]
    async fn list_models(&self) -> Result<CallToolResult, rmcp::ErrorData> {
        let models = RagCore::list_embedding_models();
        Ok(CallToolResult::success(vec![ContentBlock::text(
            models.join("\n"),
        )]))
    }

    #[tool(
        description = "Permanently remove all indexed chunks and metadata for a specific source document from the knowledge base. After deletion, the document will no longer appear in search results. The source_path must match exactly the path used during indexing (use list_sources to see current source paths). This operation cannot be undone — re-index the file to restore it."
    )]
    async fn delete_source(
        &self,
        Parameters(req): Parameters<DeleteSourceRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let result = {
            let core = self.core.read().await;
            core.delete_source(&req.source_path).await
        };
        match result {
            Ok(()) => Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "Deleted all chunks and metadata for '{}'",
                req.source_path
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Error deleting source '{}': {}",
                req.source_path, e
            ))])),
        }
    }

    #[tool(
        description = "Check the indexing status of a specific document. Returns the content hash (SHA-256), the timestamp when it was last indexed, and the number of chunks it was split into. Useful for verifying whether a document is up-to-date or needs re-indexing. Returns 'not found' if the source has not been indexed."
    )]
    async fn document_status(
        &self,
        Parameters(req): Parameters<DocumentStatusRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let result = {
            let core = self.core.read().await;
            core.document_status(&req.source_path).await
        };
        match result {
            Ok(Some(status)) => Ok(CallToolResult::success(vec![ContentBlock::text(
                format!(
                    "Source: {}\nContent hash: {}\nIndexed at: {}\nChunks: {}",
                    status.source_path, status.content_hash, status.indexed_at, status.chunk_count
                ),
            )])),
            Ok(None) => Ok(CallToolResult::success(vec![ContentBlock::text(
                format!("'{}' has not been indexed", req.source_path),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Error checking status for '{}': {}",
                req.source_path, e
            ))])),
        }
    }
}

#[tool_handler]
impl ServerHandler for RagServer {
    fn get_info(&self) -> rmcp::model::InitializeResult {
        rmcp::model::InitializeResult::new(
            rmcp::model::ServerCapabilities::builder()
                .enable_tools()
                .build(),
        )
        .with_server_info(rmcp::model::Implementation::from_build_env())
    }
}
