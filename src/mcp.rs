use crate::IndexResult;
use crate::docs;
use crate::rag::RagCore;
use crate::schemar_ext;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolResult, ContentBlock, NumberOrString, ProgressNotificationParam, ProgressToken,
};
use rmcp::service::RequestContext;
use rmcp::{RoleServer, ServerHandler, schemars, tool, tool_handler, tool_router};
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
        cache_path: &str,
        model_name: &str,
        chunk_size: usize,
        overlap: usize,
    ) -> anyhow::Result<Self> {
        let core = RagCore::new(db_path, cache_path, model_name, chunk_size, overlap).await?;
        Ok(Self {
            core: Arc::new(RwLock::new(core)),
        })
    }
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct IndexPathRequest {
    #[schemars(
        description = "Absolute or relative path to a file or directory to index recursively. Supports PDF, DOCX, XLSX, PPTX, TXT, MD, RS, PY, JS, TS, GO, JAVA, C, CPP, H, JSON, YAML, YML, TOML, XML, CSV, HTML, CSS. Unchanged files are skipped."
    )]
    pub path: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct IndexTextRequest {
    #[schemars(
        description = "Raw text content to index. Chunked, embedded, and stored for semantic search."
    )]
    pub text: String,
    #[schemars(
        description = "Unique identifier for this text (e.g. 'docs/api.md', 'clipboard'). Used for deduplication — identical content with same source is skipped."
    )]
    pub source: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchRequest {
    #[schemars(
        description = "Natural language query. Embedded and compared against stored chunks via cosine similarity."
    )]
    pub query: String,
    #[schemars(
        description = "Maximum results to return (default: 5). Higher values return more candidates but take longer."
    )]
    pub top_k: schemar_ext::Nullable<usize>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct DeleteSourceRequest {
    #[schemars(
        description = "Exact source path used during indexing. All chunks and metadata for this source are permanently deleted."
    )]
    pub source_path: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct DocumentStatusRequest {
    #[schemars(
        description = "Exact source path used during indexing. Returns content hash, last indexed timestamp, and chunk count."
    )]
    pub source_path: String,
}

#[tool_router]
impl RagServer {
    #[tool(description = "\
        Index a file or directory into the RAG knowledge base. \
        Reads, chunks, embeds, and stores for semantic search. \
        Supports PDF, DOCX, XLSX, PPTX, TXT, MD, RS, PY, JS, TS, GO, JAVA, C, CPP, H, JSON, YAML, YML, TOML, XML, CSV, HTML, CSS. \
        Unchanged files skipped. Returns summary of indexed/skipped/failed.")]
    async fn index_path(
        &self,
        Parameters(req): Parameters<IndexPathRequest>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let path = std::path::Path::new(&req.path);
        if !path.exists() {
            return Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Path not found: {}",
                req.path
            ))]));
        }

        let progress_token = ctx
            .meta
            .get_key_value("progressToken")
            .and_then(|(_, v)| serde_json::from_value::<NumberOrString>(v.clone()).ok())
            .map(ProgressToken);

        let mut stack = vec![path.to_path_buf()];
        let mut indexed = 0usize;
        let mut skipped = 0usize;
        let mut errors = Vec::new();
        let mut progress = 0u64;

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
                progress += 1;
                if let Some(ref token) = progress_token {
                    let _ = ctx
                        .peer
                        .notify_progress(
                            ProgressNotificationParam::new(token.clone(), progress as f64)
                                .with_message(current.display().to_string()),
                        )
                        .await;
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

    #[tool(description = "\
        Index raw text into the RAG knowledge base. \
        Text is chunked, embedded, and stored for semantic search. \
        Use for non-file content (clipboard, generated text, API responses). \
        Source parameter is a unique key — identical content with same source is skipped.")]
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

    #[tool(description = "\
        Search the RAG knowledge base using semantic similarity. \
        Query is embedded and compared against stored chunks via cosine similarity. \
        Results ranked by score (0.0-1.0) with matching text, source, chunk index, and score. \
        Use source_filter to restrict to a specific document.")]
    async fn search(
        &self,
        Parameters(req): Parameters<SearchRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let top_k = req.top_k.unwrap_or(5);
        let result = {
            let core = self.core.read().await;
            core.search(&req.query, top_k).await
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

    #[tool(description = "\
        List all indexed source documents. \
        Returns one full source path per line (file paths or logical names). \
        Use to see available documents for search or get source paths for document_status/delete_source.")]
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

    #[tool(description = "\
        Get total number of indexed chunks across all documents. \
        Each document is split into multiple chunks during indexing. \
        This count reflects total searchable units in the knowledge base.")]
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

    #[tool(description = "\
        Permanently delete all chunks and metadata for a source document. \
        Source path must match exactly (use list_sources to see current paths). \
        Irreversible — re-index to restore.")]
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

    #[tool(description = "\
        Check indexing status of a document. \
        Returns content hash (SHA-256), last indexed timestamp, and chunk count. \
        Use to verify if document is up-to-date or needs re-indexing. \
        Returns 'not found' if never indexed.")]
    async fn document_status(
        &self,
        Parameters(req): Parameters<DocumentStatusRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let result = {
            let core = self.core.read().await;
            core.document_status(&req.source_path).await
        };
        match result {
            Ok(Some(status)) => Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "Source: {}\nContent hash: {}\nIndexed at: {}\nChunks: {}",
                status.source_path, status.content_hash, status.indexed_at, status.chunk_count
            ))])),
            Ok(None) => Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "'{}' has not been indexed",
                req.source_path
            ))])),
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
