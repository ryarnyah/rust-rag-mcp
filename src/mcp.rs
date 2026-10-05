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

#[derive(Debug, serde::Serialize)]
struct ListSourcesResponse {
    /// Total number of indexed sources — not just this page.
    total: usize,
    /// The `offset` this page was fetched with.
    offset: usize,
    /// Entries in this page (≤ `limit`).
    count: usize,
    /// Pass back as `offset` to fetch the page after this one; `null`
    /// when the listing has reached the end.
    next_offset: Option<usize>,
    sources: Vec<IndexedSource>,
}

/// One indexed source on a page, with the document status stored at
/// indexing time. `content_hash`/`indexed_at`/`chunk_count` are `null`
/// only when the source has no document-metadata row (it still has
/// chunks) — see `RagCore::document_status`.
#[derive(Debug, serde::Serialize)]
struct IndexedSource {
    source: String,
    content_hash: Option<String>,
    indexed_at: Option<u64>,
    chunk_count: Option<u32>,
}

#[derive(Debug, serde::Serialize)]
struct MatchDocumentResponse {
    mode: crate::SearchMode,
    query: MatchQueryInfo,
    count: usize,
    results: Vec<MatchResultItem>,
}

/// What the query document resolved to: which input kind supplied it,
/// the source key it is filed under (canonical path or indexed source;
/// `null` for raw text), and how many chunks it was split into.
#[derive(Debug, serde::Serialize)]
struct MatchQueryInfo {
    kind: &'static str,
    source: Option<String>,
    chunks: u32,
}

#[derive(Debug, serde::Serialize)]
struct MatchResultItem {
    rank: usize,
    source: String,
    /// Mean per-query-chunk score (see `DocumentMatch::score`): the
    /// scale is the one of `mode`, never comparable across modes.
    score: f64,
    matched_chunks: u32,
    query_chunks: u32,
    /// Strongest single chunk-level match behind `score`.
    best_score: f64,
    best_query_chunk: u32,
    best_match_chunk: u32,
    best_match_text: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchDocumentsRequest {
    #[schemars(
        description = "Search query for document source paths/names. Supports partial matching and case-insensitive search. Examples: 'api', 'docs/api.md', 'README', '*.txt'"
    )]
    pub query: String,

    #[schemars(
        description = "Maximum results to return (default: 10). Returns up to N matching document sources."
    )]
    pub top_k: schemar_ext::Nullable<usize>,
}

#[derive(Debug, serde::Serialize)]
struct SearchDocumentsResponse {
    query: String,
    total_indexed: usize,
    matched_count: usize,
    results: Vec<DocumentMatch>,
}

#[derive(Debug, serde::Serialize)]
struct DocumentMatch {
    rank: usize,
    source: String,
    /// Relevance score: 1.0 = exact match, <1.0 = partial/fuzzy match
    score: f64,
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
    #[serde(default)]
    #[schemars(
        description = "HNSW search expansion factor (optional; default: adaptive based on k). Controls search breadth during HNSW traversal: higher ef = thorough search but slower, lower ef = faster but may miss neighbors. Typical range: [20, 300]. Ignored in lexical mode."
    )]
    pub ef_search: schemar_ext::Nullable<usize>,
    #[serde(default)]
    #[schemars(
        description = "P6: Filter results to only include chunks from this source path (optional). Speeds up searches when filtering by source via pre-indexed source→vec_id mapping. Example: 'docs/api.md' returns only chunks indexed with that exact source."
    )]
    pub source_filter: Option<String>,
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

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct ListSourcesRequest {
    #[serde(default)]
    #[schemars(
        description = "Number of sources to skip before the first entry of this page (optional; default: 0). Pass the `next_offset` returned by the previous response to fetch the page after it."
    )]
    pub offset: schemar_ext::Nullable<usize>,
    #[serde(default)]
    #[schemars(
        description = "Maximum number of sources per page (optional; default: 50). The final page may contain fewer entries, and `next_offset` is null once the listing has reached the end. An `offset` past the end returns an empty page (with `total` still set)."
    )]
    pub limit: schemar_ext::Nullable<usize>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct MatchDocumentRequest {
    #[schemars(
        description = "Raw text of a new document to match against the index (it does not need to be indexed). Provide exactly one of `text`, `path`, or `source`."
    )]
    pub text: Option<String>,
    #[schemars(
        description = "Path to a document file to match against the index — supports the same formats as `index_path` (PDF, DOCX, XLSX, PPTX, TXT, MD, code, ...). The file does not need to be indexed; it is read and chunked exactly as `index_path` would. Provide exactly one of `text`, `path`, or `source`."
    )]
    pub path: Option<String>,
    #[schemars(
        description = "Source key of a document already in the index: its stored chunks are used as the query document. Provide exactly one of `text`, `path`, or `source`."
    )]
    pub source: Option<String>,
    #[serde(default)]
    #[schemars(
        description = "Maximum matched documents to return (default: 5). Higher values return more candidates but take longer."
    )]
    pub top_k: schemar_ext::Nullable<usize>,
    #[serde(default)]
    #[schemars(
        description = "Retrieval mode per query chunk (optional; omit or null = hybrid), with the same semantics and score scales as `search`: `hybrid` (RRF weights), `semantic` (cosine in [0,1]), `lexical` (raw BM25 weights, no embedding). Scores are comparable only within one call, never across modes."
    )]
    pub mode: schemar_ext::Nullable<crate::SearchMode>,
    #[serde(default)]
    #[schemars(
        description = "HNSW search expansion factor (optional; default: adaptive based on the per-chunk candidate pool). Ignored in lexical mode. See `search.ef_search`."
    )]
    pub ef_search: schemar_ext::Nullable<usize>,
    #[serde(default)]
    #[schemars(
        description = "Candidates retrieved per query chunk before hits are grouped by document (default: 50). Larger values let a single chunk of the query document reach more distinct documents; smaller values are faster."
    )]
    pub candidates_per_chunk: schemar_ext::Nullable<usize>,
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
        let ef_search = req.ef_search.0;
        let source_filter = req.source_filter.clone();

        let result = {
            let core = self.core.read().await;
            core.search_with_mode_ef(&req.query, top_k, mode, ef_search)
                .await
        };

        match result {
            Ok(results) => {
                // P6: Use source filter if provided. Pre-computed source→vec_id index
                // in RagCore allows efficient filtering without O(k) JSON parsing.
                // This is effective when filtering by document source.
                let filtered_results: Vec<_> = if let Some(source) = source_filter {
                    results
                        .into_iter()
                        .filter(|r| r.chunk.source == source)
                        .collect()
                } else {
                    results
                };

                let items: Vec<SearchResultItem> = filtered_results
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
    #[tool(description = "\
        List the indexed sources (files and text documents) with pagination. \
        Entries come back sorted by source name — stable page boundaries while the index does not change — and each carries its content hash, last indexed timestamp, and chunk count. \
        `total` is the full number of indexed sources, not just this page's; keep passing the returned `next_offset` back as `offset` until it is null. \
        Defaults: offset 0, limit 50. Use this to discover the exact source paths expected by `search.source_filter`, `match_document.source`, and `delete_source`.")]
    async fn list_sources(
        &self,
        Parameters(req): Parameters<ListSourcesRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let offset = req.offset.unwrap_or(0);
        let limit = req.limit.unwrap_or(50);

        // A single read lock spans the slice and the per-source statuses,
        // so the page cannot interleave with an index/delete and show a
        // half-updated view.
        let core = self.core.read().await;
        let (page, total) = match core.list_sources_page(offset, limit).await {
            Ok(page) => page,
            Err(e) => return Err(internal_err("Error listing sources", e)),
        };

        let mut sources = Vec::with_capacity(page.len());
        for source in &page {
            // `document_status` is an O(1) metadata-index lookup, so
            // enriching a page never scans the vector table.
            let status = match core.document_status(source).await {
                Ok(status) => status,
                Err(e) => {
                    return Err(internal_err(
                        &format!("Error reading status for '{source}'"),
                        e,
                    ));
                }
            };
            sources.push(IndexedSource {
                source: source.clone(),
                content_hash: status.as_ref().map(|s| s.content_hash.clone()),
                indexed_at: status.as_ref().map(|s| s.indexed_at),
                chunk_count: status.as_ref().map(|s| s.chunk_count),
            });
        }

        let count = sources.len();
        // A page that stops short of `total` still has a next one;
        // `offset + count` is exactly where that page must start.
        let next_offset = (offset + count < total).then_some(offset + count);
        ok_json(ListSourcesResponse {
            total,
            offset,
            count,
            next_offset,
            sources,
        })
    }

    #[tool(description = "\
        Match a whole document against the index and rank the indexed documents that best match it — document-level similarity, not isolated chunk hits. \
        The query document is one of: raw `text`, a file `path` (PDF/DOCX/TXT/code; it does not need to be indexed), or the `source` key of an already-indexed document — provide exactly one. \
        Every chunk of the query document is searched (same `mode` semantics and score scale as `search`), hits are grouped by the matched document's source, and each document scores the mean of its best per-chunk score with query chunks that never hit it contributing 0 — so documents matching more of the query rank higher. \
        The query document's own source is excluded, so a document is never its own match. \
        Each result carries the matched source, the aggregate score, how many query chunks matched, and the strongest single passage (`best_match_text`) tying the two documents together.")]
    async fn match_document(
        &self,
        Parameters(req): Parameters<MatchDocumentRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let selected = [req.text.is_some(), req.path.is_some(), req.source.is_some()]
            .into_iter()
            .filter(|&chosen| chosen)
            .count();
        if selected != 1 {
            return Err(ErrorData::invalid_params(
                "Provide exactly one of: `text`, `path`, or `source`",
                None,
            ));
        }
        if let Some(path) = &req.path
            && !std::path::Path::new(path).exists()
        {
            return Err(ErrorData::invalid_params(
                format!("Path not found: {path}"),
                None,
            ));
        }
        if let Some(source) = &req.source {
            let status = {
                let core = self.core.read().await;
                core.document_status(source).await
            };
            match status {
                Ok(Some(_)) => {}
                Ok(None) => {
                    return Err(ErrorData::invalid_params(
                        format!("Source not indexed: {source}"),
                        None,
                    ));
                }
                Err(e) => return Err(internal_err("Error checking source", e)),
            }
        }

        let defaults = crate::MatchOptions::default();
        let opts = crate::MatchOptions {
            top_k: req.top_k.unwrap_or(defaults.top_k),
            mode: req.mode.0.unwrap_or(defaults.mode),
            ef_search: req.ef_search.0,
            candidates_per_chunk: req
                .candidates_per_chunk
                .unwrap_or(defaults.candidates_per_chunk),
        };
        let (document, kind) = if let Some(text) = &req.text {
            (crate::MatchDocument::Text { text, ext: "" }, "text")
        } else if let Some(path) = &req.path {
            (
                crate::MatchDocument::File {
                    path: std::path::Path::new(path),
                },
                "file",
            )
        } else {
            (
                crate::MatchDocument::Indexed {
                    source: req.source.as_deref().unwrap(),
                },
                "indexed",
            )
        };

        let outcome = {
            let core = self.core.read().await;
            core.match_document(&document, &opts).await
        };
        match outcome {
            Ok(result) => {
                let results: Vec<MatchResultItem> = result
                    .matches
                    .into_iter()
                    .enumerate()
                    .map(|(i, m)| MatchResultItem {
                        rank: i + 1,
                        source: m.source,
                        score: m.score,
                        matched_chunks: m.matched_chunks,
                        query_chunks: m.query_chunks,
                        best_score: m.best_score,
                        best_query_chunk: m.best_query_chunk,
                        best_match_chunk: m.best_match_chunk,
                        best_match_text: m.best_match_text,
                    })
                    .collect();
                ok_json(MatchDocumentResponse {
                    mode: opts.mode,
                    query: MatchQueryInfo {
                        kind,
                        source: result.query_source,
                        chunks: result.query_chunks,
                    },
                    count: results.len(),
                    results,
                })
            }
            Err(e) => Err(internal_err("Error matching document", e)),
        }
    }

    #[tool(description = "\
        Search for indexed documents by name, path, or partial match. \
        Only searches document source paths (not content). \
        Supports case-insensitive partial matching and glob patterns. \
        Returns up to top_k matching document sources with relevance scores (1.0 = exact match).")]
    async fn search_documents(
        &self,
        Parameters(req): Parameters<SearchDocumentsRequest>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let top_k = req.top_k.unwrap_or(10);
        let query_lower = req.query.to_lowercase();

        let sources = {
            let core = self.core.read().await;
            core.list_sources().await
        };

        match sources {
            Ok(all_sources) => {
                let total_indexed = all_sources.len();

                // Score and filter documents based on source path matching
                let mut matches: Vec<(String, f64)> = all_sources
                    .into_iter()
                    .filter_map(|source| {
                        let source_lower = source.to_lowercase();
                        let score = calculate_match_score(&source_lower, &query_lower);

                        if score > 0.0 {
                            Some((source, score))
                        } else {
                            None
                        }
                    })
                    .collect();

                // Sort by score descending
                matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                let matched_count = matches.len();
                let results: Vec<DocumentMatch> = matches
                    .into_iter()
                    .take(top_k)
                    .enumerate()
                    .map(|(i, (source, score))| DocumentMatch {
                        rank: i + 1,
                        source,
                        score,
                    })
                    .collect();

                ok_json(SearchDocumentsResponse {
                    query: req.query,
                    total_indexed,
                    matched_count,
                    results,
                })
            }
            Err(e) => Err(internal_err("Error listing indexed sources", e)),
        }
    }
}

/// Calculate relevance score for document source path match (0.0 = no match, 1.0 = exact match)
fn calculate_match_score(source_lower: &str, query_lower: &str) -> f64 {
    if source_lower == query_lower {
        return 1.0; // Exact match
    }

    if source_lower.contains(&query_lower) {
        // Substring match - score higher if match is at path boundaries
        if source_lower.starts_with(&query_lower)
            || source_lower.contains(&format!("/{}", query_lower))
            || source_lower.ends_with(&query_lower)
        {
            return 0.95; // Strong substring match
        }
        return 0.7; // Weak substring match (match in middle)
    }

    // Check for glob-style matching (* wildcard)
    if query_lower.contains('*') {
        if glob_match(source_lower, &query_lower) {
            return 0.85; // Glob pattern match
        }
    }

    // Fuzzy matching: check if all query chars appear in order in source
    if fuzzy_match(source_lower, &query_lower) {
        return 0.6; // Fuzzy match
    }

    0.0 // No match
}

/// Simple glob pattern matching (* = any sequence of chars)
fn glob_match(source: &str, pattern: &str) -> bool {
    let pattern_parts: Vec<&str> = pattern.split('*').collect();
    let mut remaining = source;

    for (i, part) in pattern_parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }

        if i == 0 && !remaining.starts_with(part) {
            return false;
        } else if i == pattern_parts.len() - 1 && !remaining.ends_with(part) {
            return false;
        } else if let Some(pos) = remaining.find(part) {
            remaining = &remaining[pos + part.len()..];
        } else {
            return false;
        }
    }

    true
}

/// Fuzzy matching: check if all chars in query appear in source in order
fn fuzzy_match(source: &str, query: &str) -> bool {
    let mut source_chars = source.chars();
    for query_char in query.chars() {
        if !source_chars.any(|c| c == query_char) {
            return false;
        }
    }
    true
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
