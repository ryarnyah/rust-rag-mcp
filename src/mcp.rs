use crate::IndexResult;
use crate::docs;
use crate::rag::RagCore;
use crate::schemar_ext;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolResult, ContentBlock, ErrorData, NumberOrString, ProgressNotificationParam,
    ProgressToken,
};
use rmcp::service::RequestContext;
use rmcp::{RoleServer, ServerHandler, schemars, tool, tool_handler, tool_router};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::RwLock;

#[derive(Debug, serde::Serialize)]
struct IndexPathResponse {
    status: &'static str,
    indexed: usize,
    skipped: usize,
    details: Vec<String>,
}

#[derive(Debug, serde::Serialize)]
struct IndexTextResponse {
    status: &'static str,
    source: String,
    chunks: Option<usize>,
}

#[derive(Debug, serde::Serialize)]
struct SearchResponse {
    mode: crate::SearchMode,
    count: usize,
    results: Vec<SearchResultItem>,
}

#[derive(Debug, serde::Serialize)]
struct SearchResultItem {
    rank: usize,
    score: f64,
    /// Hybrid only: the per-retriever (dense cosine / BM25) scores and
    /// ranks behind the RRF weight. Omitted for other modes, whose
    /// `score` already is the raw value.
    #[serde(skip_serializing_if = "Option::is_none")]
    components: Option<crate::HybridScoreComponents>,
    source: String,
    chunk_index: u32,
    text: String,
}

#[derive(Debug, serde::Serialize)]
struct DeleteSourceResponse {
    status: &'static str,
    source: String,
}

#[derive(Debug, serde::Serialize)]
struct DocumentStatusResponse {
    found: bool,
    source: String,
    content_hash: Option<String>,
    indexed_at: Option<u64>,
    chunk_count: Option<u32>,
}

#[derive(Clone)]
pub struct RagServer {
    core: Arc<RwLock<RagCore>>,
}

/// Serialize `resp` as the single JSON content block of a successful
/// `tools/call` result.
fn ok_json<T: serde::Serialize>(resp: T) -> Result<CallToolResult, ErrorData> {
    Ok(CallToolResult::success(vec![ContentBlock::json(resp)?]))
}

/// Build the `internal error` returned when a `RagCore` operation fails,
/// prefixing the failure with `context` for the client log.
fn internal_err(context: &str, e: impl std::fmt::Display) -> ErrorData {
    ErrorData::internal_error(format!("{context}: {e}"), None)
}

impl RagServer {
    pub async fn new(
        db_path: &std::path::Path,
        cache_path: &std::path::Path,
        model_name: &str,
        chunk_size: usize,
        overlap: usize,
        ef_construction: usize,
    ) -> anyhow::Result<Self> {
        let core = RagCore::new(
            db_path,
            cache_path,
            model_name,
            chunk_size,
            overlap,
            ef_construction,
        )
        .await?;
        Ok(Self {
            core: Arc::new(RwLock::new(core)),
        })
    }

    /// Shut down the server's core: takes the write lock (draining any
    /// in-flight tool handler, which hold read locks), persists the
    /// `.srcidx` and `.bm25` sidecars at this quiescent point, and
    /// releases the database file locks.
    pub async fn close(&self) -> anyhow::Result<()> {
        let core = self.core.write().await;
        core.close().await
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
        description = "Natural language query. In hybrid mode it is both embedded (cosine over HNSW) and tokenized for BM25; in semantic mode it is embedded only; in lexical mode it is tokenized only (no model invocation)."
    )]
    pub query: String,
    #[schemars(
        description = "Maximum results to return (default: 5). Higher values return more candidates but take longer."
    )]
    pub top_k: schemar_ext::Nullable<usize>,
    #[serde(default)]
    #[schemars(
        description = "Retrieval mode (optional; omit or null = hybrid). `hybrid`: HNSW dense + BM25 lexical ranked lists fused with Reciprocal Rank Fusion — scores are RRF weights in (0, ~0.033], monotonic in fused rank, not similarities. `semantic`: dense only, scores are cosine similarity in [0,1]. `lexical`: BM25 only, scores are unbounded BM25 weights, and the query is never embedded."
    )]
    pub mode: schemar_ext::Nullable<crate::SearchMode>,
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
        Unchanged files skipped. Returns summary of indexed/skipped/failed. \
        When the request carries a `progressToken`, a progress notification is sent for every chunk indexed.")]
    async fn index_path(
        &self,
        Parameters(req): Parameters<IndexPathRequest>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let path = std::path::Path::new(&req.path);
        if !path.exists() {
            return Err(ErrorData::invalid_params(
                format!("Path not found: {}", req.path),
                None,
            ));
        }

        let progress_token = ctx
            .meta
            .get_key_value("progressToken")
            .and_then(|(_, v)| serde_json::from_value::<NumberOrString>(v.clone()).ok())
            .map(ProgressToken);

        let mut indexed = 0usize;
        let mut skipped = 0usize;
        let mut errors = Vec::new();
        // Monotonic progress for the client's token: one unit per chunk
        // stored, plus one per file that produced no chunk (skipped or
        // failed), so every notification still moves it forward. Shared
        // with the per-chunk closure below, which clones everything it
        // needs — a closure *borrowing* these locals would drag their
        // references through the indexing future and break the tool
        // handler's `Send` bound.
        let progress = Arc::new(AtomicU64::new(0));

        for current in docs::walk_supported_files(path).await {
            let file = current.display().to_string();
            let peer = ctx.peer.clone();
            let token = progress_token.clone();
            let counter = progress.clone();
            let notify_file = file.clone();
            let result = {
                let core = self.core.read().await;
                // One notification per chunk, as it lands in the index.
                core.index_file_with_progress(&current, move |done, total| {
                    let peer = peer.clone();
                    let token = token.clone();
                    let file = notify_file.clone();
                    let counter = counter.clone();
                    async move {
                        let Some(token) = token else { return };
                        let value = counter.fetch_add(1, Ordering::SeqCst) + 1;
                        let _ = peer
                            .notify_progress(
                                ProgressNotificationParam::new(token, value as f64)
                                    .with_message(format!("{file}: chunk {done}/{total}")),
                            )
                            .await;
                    }
                })
                .await
            };
            // Files with no chunk still get one notification, so the
            // client sees the walk finish them.
            let outcome = match &result {
                Ok(IndexResult::Indexed(count)) => {
                    indexed += 1;
                    errors.push(format!("Indexed {} — {} chunks", file, count));
                    if *count == 0 {
                        Some(format!("{file} — no chunks"))
                    } else {
                        None
                    }
                }
                Ok(IndexResult::Skipped) => {
                    skipped += 1;
                    Some(format!("{file} — unchanged"))
                }
                Err(e) => {
                    errors.push(format!("Failed {}: {}", file, e));
                    Some(format!("{file} — failed: {e}"))
                }
            };
            if let Some(message) = outcome
                && let Some(token) = progress_token.clone()
            {
                let value = progress.fetch_add(1, Ordering::SeqCst) + 1;
                let _ = ctx
                    .peer
                    .notify_progress(
                        ProgressNotificationParam::new(token, value as f64).with_message(message),
                    )
                    .await;
            }
        }

        let resp = IndexPathResponse {
            status: "ok",
            indexed,
            skipped,
            details: errors,
        };
        ok_json(resp)
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
            Ok(IndexResult::Indexed(count)) => ok_json(IndexTextResponse {
                status: "indexed",
                source: req.source,
                chunks: Some(count),
            }),
            Ok(IndexResult::Skipped) => ok_json(IndexTextResponse {
                status: "skipped",
                source: req.source,
                chunks: None,
            }),
            Err(e) => Err(internal_err(
                &format!("Error indexing text '{}'", req.source),
                e,
            )),
        }
    }

    #[tool(description = "\
        Search the RAG knowledge base. By default runs hybrid search: dense HNSW retrieval and BM25 lexical retrieval are fused with Reciprocal Rank Fusion (RRF), so exact identifiers and rare terms (lexical strength) and paraphrase (semantic strength) both surface. \
        Optional `mode`: `hybrid` (default), `semantic` (embedded cosine similarity only), `lexical` (BM25 only — no embedding, best for exact identifiers). \
        Score meaning depends on mode: RRF weight (0, ~0.033] for hybrid, cosine [0,1] for semantic, unbounded BM25 weight for lexical. In hybrid mode each result also carries `components` — the dense cosine score/rank and BM25 weight/rank that were fused (a side is null when the document missed that retriever's top-pool). Results carry matching text, source, chunk index, and score.")]
    async fn search(
        &self,
        Parameters(req): Parameters<SearchRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let top_k = req.top_k.unwrap_or(5);
        let mode = req.mode.0.unwrap_or_default();
        let result = {
            let core = self.core.read().await;
            core.search_with_mode(&req.query, top_k, mode).await
        };
        match result {
            Ok(results) => {
                let items: Vec<SearchResultItem> = results
                    .iter()
                    .enumerate()
                    .map(|(i, r)| SearchResultItem {
                        rank: i + 1,
                        score: r.score,
                        components: r.components.clone(),
                        source: r.chunk.source.clone(),
                        chunk_index: r.chunk.chunk_index,
                        text: r.chunk.text.clone(),
                    })
                    .collect();
                ok_json(SearchResponse {
                    mode,
                    count: items.len(),
                    results: items,
                })
            }
            Err(e) => Err(internal_err("Error searching", e)),
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
            Ok(()) => ok_json(DeleteSourceResponse {
                status: "deleted",
                source: req.source_path,
            }),
            Err(e) => Err(internal_err(
                &format!("Error deleting source '{}'", req.source_path),
                e,
            )),
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
            Ok(Some(status)) => ok_json(DocumentStatusResponse {
                found: true,
                source: status.source_path,
                content_hash: Some(status.content_hash),
                indexed_at: Some(status.indexed_at),
                chunk_count: Some(status.chunk_count),
            }),
            Ok(None) => ok_json(DocumentStatusResponse {
                found: false,
                source: req.source_path,
                content_hash: None,
                indexed_at: None,
                chunk_count: None,
            }),
            Err(e) => Err(internal_err(
                &format!("Error checking status for '{}'", req.source_path),
                e,
            )),
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
