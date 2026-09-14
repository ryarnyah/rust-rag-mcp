use crate::rag::RagCore;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock};
use rmcp::{schemars, tool, tool_router, ServerHandler};
use std::path::PathBuf;
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
pub struct IndexFileRequest {
    #[schemars(description = "Path to the file to index")]
    pub path: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct IndexTextRequest {
    #[schemars(description = "Text content to index")]
    pub text: String,
    #[schemars(description = "Source identifier for the text")]
    pub source: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchRequest {
    #[schemars(description = "Search query")]
    pub query: String,
    #[schemars(description = "Number of results to return (default: 5)")]
    pub top_k: Option<usize>,
    #[schemars(description = "Optional source filter")]
    pub source_filter: Option<String>,
}

#[tool_router]
impl RagServer {
    #[tool(
        description = "Index a file (PDF, DOCX, XLSX, PPTX, TXT, etc.) into the RAG knowledge base"
    )]
    async fn index_file(
        &self,
        Parameters(req): Parameters<IndexFileRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let result = {
            let core = self.core.read().await;
            core.index_file(PathBuf::from(req.path).as_path()).await
        };
        match result {
            Ok(count) => Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "Indexed {} chunks",
                count
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Error indexing file: {}",
                e
            ))])),
        }
    }

    #[tool(description = "Index raw text content into the RAG knowledge base")]
    async fn index_text(
        &self,
        Parameters(req): Parameters<IndexTextRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let result = {
            let core = self.core.read().await;
            core.index_text(&req.text, &req.source).await
        };
        match result {
            Ok(count) => Ok(CallToolResult::success(vec![ContentBlock::text(format!(
                "Indexed {} chunks from '{}'",
                count, req.source
            ))])),
            Err(e) => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Error indexing text: {}",
                e
            ))])),
        }
    }

    #[tool(description = "Search the RAG knowledge base using semantic similarity")]
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

    #[tool(description = "List all indexed source files")]
    async fn list_sources(&self) -> Result<CallToolResult, rmcp::ErrorData> {
        let result = {
            let core = self.core.read().await;
            core.list_sources().await
        };
        match result {
            Ok(sources) => Ok(CallToolResult::success(vec![ContentBlock::text(
                sources.join("\n"),
            )])),
            Err(e) => Ok(CallToolResult::error(vec![ContentBlock::text(format!(
                "Error listing sources: {}",
                e
            ))])),
        }
    }

    #[tool(description = "Get the total number of indexed chunks")]
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

    #[tool(description = "List all available embedding models")]
    async fn list_models(&self) -> Result<CallToolResult, rmcp::ErrorData> {
        let models = RagCore::list_embedding_models();
        Ok(CallToolResult::success(vec![ContentBlock::text(
            models.join("\n"),
        )]))
    }
}

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
