//! End-to-end tests for the MCP tool surface.
//!
//! A real rmcp client talks to a real [`RagServer`] over an in-process
//! duplex transport, so the handlers are exercised exactly the way an MCP
//! client sees them: through `tools/list`, `tools/call`, JSON-RPC error
//! codes, and `notifications/progress`.

mod fixtures;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use rmcp::{
    ClientHandler, RoleClient, ServiceError, ServiceExt,
    model::{
        CallToolRequestParams, CallToolResult, ClientRequest, ContentBlock, ErrorCode,
        ProgressNotificationParam, Request, ServerResult,
    },
    service::{NotificationContext, PeerRequestOptions, RunningService},
};
use rust_rag_mcp::mcp::RagServer;
use serde_json::{Value, json};
use tempfile::TempDir;

/// Client that records every progress notification it receives, so tests
/// can assert on the sequence (value + message) the server emitted.
#[derive(Clone, Default)]
struct RecordingClient {
    progress: Arc<Mutex<Vec<ProgressUpdate>>>,
}

#[derive(Clone, Debug)]
struct ProgressUpdate {
    value: f64,
    message: Option<String>,
}

impl ClientHandler for RecordingClient {
    async fn on_progress(
        &self,
        params: ProgressNotificationParam,
        _context: NotificationContext<RoleClient>,
    ) {
        self.progress.lock().unwrap().push(ProgressUpdate {
            value: params.progress,
            message: params.message,
        });
    }
}

struct Harness {
    client: RunningService<RoleClient, RecordingClient>,
    /// Kept alive for the lifetime of the test so the temp database and
    /// its file locks are not torn down mid-request.
    _dir: TempDir,
    progress: Arc<Mutex<Vec<ProgressUpdate>>>,
}

impl Harness {
    async fn start() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let server = RagServer::new(
            dir.path(),
            &fixtures::model_cache_dir(),
            "Xenova/bge-small-en-v1.5",
            512,
            64,
            150,
        )
        .await
        .expect("RagServer::new");

        let (server_transport, client_transport) = tokio::io::duplex(1 << 20);
        tokio::spawn(async move {
            let service = server.serve(server_transport).await.expect("serve");
            let _ = service.waiting().await;
        });

        let client = RecordingClient::default();
        let progress = client.progress.clone();
        let client_service = client.serve(client_transport).await.expect("client");
        Harness {
            client: client_service,
            _dir: dir,
            progress,
        }
    }

    /// Call a tool and decode the JSON payload the handler put in its
    /// text content block. Panics if the tool returned an error.
    async fn call(&self, name: &str, args: Value) -> Value {
        match self.try_call(name, args).await {
            Ok(value) => value,
            Err(e) => panic!("`{name}` returned an error: {e:?}"),
        }
    }

    async fn try_call(&self, name: &str, args: Value) -> Result<Value, rmcp::ErrorData> {
        let params = CallToolRequestParams::new(name.to_string())
            .with_arguments(args.as_object().cloned().unwrap_or_default());
        let result = self
            .client
            .peer()
            .call_tool(params)
            .await
            .map_err(as_error_data)?;
        Ok(json_payload(result))
    }

    /// Call a tool with a progress token attached, returning the token so
    /// the test can correlate it with the recorded notifications.
    async fn call_with_progress(
        &self,
        name: &str,
        args: Value,
    ) -> (Value, rmcp::model::ProgressToken) {
        let handle = self
            .client
            .send_cancellable_request(
                ClientRequest::CallToolRequest(Request::new(
                    CallToolRequestParams::new(name.to_string())
                        .with_arguments(args.as_object().cloned().unwrap_or_default()),
                )),
                PeerRequestOptions::no_options(),
            )
            .await
            .expect("send request");
        let token = handle.progress_token.clone();
        let result = handle.await_response().await.expect("response");
        (server_json(result), token)
    }

    fn progress_updates(&self) -> Vec<ProgressUpdate> {
        self.progress.lock().unwrap().clone()
    }

    /// Progress values in arrival order.
    fn progress_values(&self) -> Vec<f64> {
        self.progress_updates()
            .into_iter()
            .map(|u| u.value)
            .collect()
    }

    /// Progress messages in arrival order, `None` where the server
    /// emitted a value with no message.
    fn progress_messages(&self) -> Vec<Option<String>> {
        self.progress_updates()
            .into_iter()
            .map(|u| u.message)
            .collect()
    }
}

/// The handlers serialize their response into a JSON *string* inside a
/// text content block (see `ContentBlock::json`), so decode that.
fn json_payload(result: CallToolResult) -> Value {
    let text = result
        .content
        .into_iter()
        .find_map(|block| match block {
            ContentBlock::Text(text) => Some(text.text),
            _ => None,
        })
        .expect("tool result carries a text block");
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("text block is JSON ({e}): {text}"))
}

fn server_json(result: ServerResult) -> Value {
    match result {
        ServerResult::CallToolResult(result) => json_payload(result),
        other => panic!("expected a tool result, got {other:?}"),
    }
}

/// A tool handler's `ErrorData` arrives wrapped in the transport error
/// enum; anything else is a broken test harness, not a server bug.
fn as_error_data(error: ServiceError) -> rmcp::ErrorData {
    match error {
        ServiceError::McpError(error) => error,
        other => panic!("expected a JSON-RPC error, got {other:?}"),
    }
}

const TOOL_NAMES: [&str; 5] = [
    "delete_source",
    "document_status",
    "index_path",
    "index_text",
    "search",
];

#[tokio::test]
async fn lists_every_tool_the_server_implements() {
    let h = Harness::start().await;
    let listed = h.client.peer().list_tools(None).await.unwrap();
    let mut names: Vec<String> = listed
        .tools
        .iter()
        .map(|tool| tool.name.to_string())
        .collect();
    names.sort();
    let mut expected: Vec<String> = TOOL_NAMES.iter().map(|name| name.to_string()).collect();
    expected.sort();
    assert_eq!(names, expected);
}

#[tokio::test]
async fn server_info_advertises_tool_capability() {
    let h = Harness::start().await;
    let info = h.client.peer().list_tools(None).await.unwrap();
    // Every advertised tool must carry a description and an input schema,
    // otherwise MCP clients cannot prompt for it.
    for tool in &info.tools {
        assert!(
            tool.description.is_some(),
            "{} has no description",
            tool.name
        );
        assert!(
            tool.input_schema.get("type").is_some(),
            "{} has no input schema",
            tool.name
        );
    }
}

/// index → dedup-skip → search → status → delete → status, the full
/// lifecycle a client drives, all through the wire.
#[tokio::test]
async fn index_search_status_delete_roundtrip() {
    let h = Harness::start().await;
    let text = "Rust is a systems programming language focused on safety.";
    let source = "docs/roundtrip.rs";

    let first = h
        .call("index_text", json!({ "text": text, "source": source }))
        .await;
    assert_eq!(first["status"], "indexed", "{first}");
    assert_eq!(first["source"], source);
    assert!(first["chunks"].as_u64().unwrap() > 0, "{first}");

    let again = h
        .call("index_text", json!({ "text": text, "source": source }))
        .await;
    assert_eq!(again["status"], "skipped", "{again}");
    assert!(
        again["chunks"].is_null(),
        "skipped carries no count: {again}"
    );

    let found = h
        .call("document_status", json!({ "source_path": source }))
        .await;
    assert_eq!(found["found"], true, "{found}");
    assert_eq!(found["source"], source);
    assert_eq!(
        found["chunk_count"].as_u64(),
        first["chunks"].as_u64(),
        "{found}"
    );
    assert!(
        found["content_hash"].as_str().unwrap().len() == 64,
        "{found}"
    );
    assert!(found["indexed_at"].as_u64().unwrap() > 0, "{found}");

    let search = h
        .call("search", json!({ "query": "systems programming" }))
        .await;
    assert_eq!(search["mode"], "hybrid", "default mode is hybrid: {search}");
    assert!(search["count"].as_u64().unwrap() > 0, "{search}");
    let top = &search["results"][0];
    assert_eq!(top["rank"], 1);
    assert_eq!(top["source"], source);
    assert!(top["score"].as_f64().unwrap() > 0.0, "{top}");
    assert!(top["text"].as_str().unwrap().contains("Rust"), "{top}");
    // Hybrid results explain their RRF weight with both retriever sides.
    let components = top["components"].as_object().expect("components");
    assert!(components["dense"]["rank"].as_u64().unwrap() >= 1, "{top}");
    assert!(
        components["lexical"]["rank"].as_u64().unwrap() >= 1,
        "{top}"
    );

    let deleted = h
        .call("delete_source", json!({ "source_path": source }))
        .await;
    assert_eq!(deleted["status"], "deleted", "{deleted}");

    let gone = h
        .call("document_status", json!({ "source_path": source }))
        .await;
    assert_eq!(gone["found"], false, "{gone}");
    assert!(gone["content_hash"].is_null(), "{gone}");
    assert!(gone["chunk_count"].is_null(), "{gone}");

    let after = h
        .call("search", json!({ "query": "systems programming" }))
        .await;
    assert_eq!(after["count"], 0, "deletion must remove the hit: {after}");
}

/// Lexical mode never embeds the query, and its raw BM25 score is not an
/// RRF weight — so it must not carry `components`.
#[tokio::test]
async fn lexical_mode_omits_hybrid_components() {
    let h = Harness::start().await;
    h.call(
        "index_text",
        json!({ "text": "the quick brown fox jumps", "source": "fox.txt" }),
    )
    .await;

    let search = h
        .call(
            "search",
            json!({ "query": "brown fox", "mode": "lexical", "top_k": 3 }),
        )
        .await;
    assert_eq!(search["mode"], "lexical", "{search}");
    assert!(search["count"].as_u64().unwrap() > 0, "{search}");
    assert!(
        search["results"][0].get("components").is_none(),
        "non-hybrid results must not carry components: {}",
        search["results"][0]
    );
    // `top_k` defaults and explicit values are both honoured.
    assert!(
        search["count"].as_u64().unwrap() <= 3,
        "top_k ignored: {search}"
    );
}

/// A missing path is a client mistake (`invalid_params`), not a crash.
#[tokio::test]
async fn index_path_rejects_missing_path() {
    let h = Harness::start().await;
    let err = h
        .try_call("index_path", json!({ "path": "/definitely/not/here" }))
        .await
        .expect_err("missing path must be rejected");
    assert_eq!(err.code, ErrorCode::INVALID_PARAMS);
    assert!(
        err.message.contains("Path not found"),
        "unexpected message: {}",
        err.message
    );
}

/// A file with a supported extension that cannot be extracted is reported
/// per-file rather than aborting the walk: `indexed` still counts the
/// files that succeeded and `details` explains the failure.
#[tokio::test]
async fn index_path_reports_failing_files_without_aborting() {
    let h = Harness::start().await;
    let tree = h.dir_path().join("broken");
    tokio::fs::create_dir_all(&tree).await.unwrap();
    tokio::fs::write(tree.join("good.txt"), "hello world")
        .await
        .unwrap();
    tokio::fs::write(tree.join("bad.pdf"), b"this is not a pdf")
        .await
        .unwrap();

    let resp = h
        .call("index_path", json!({ "path": tree.to_string_lossy() }))
        .await;
    assert_eq!(resp["status"], "ok", "{resp}");
    assert_eq!(resp["indexed"], 1, "{resp}");

    let details: Vec<String> = resp["details"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d.as_str().unwrap().to_string())
        .collect();
    assert!(
        details
            .iter()
            .any(|d| d.starts_with("Failed ") && d.contains("bad.pdf")),
        "failure not reported: {details:?}"
    );
    assert!(
        details.iter().any(|d| d.contains("good.txt")),
        "success not reported: {details:?}"
    );
}

/// The walk must be idempotent: a second pass over the same tree skips
/// every unchanged file. Skipped files are deliberately *not* added to
/// `details` (that array is a per-indexed-file report); the client learns
/// about them through a progress notification instead.
#[tokio::test]
async fn index_path_skips_unchanged_files_on_rerun() {
    let h = Harness::start().await;
    let tree = h.dir_path().join("rerun");
    tokio::fs::create_dir_all(&tree).await.unwrap();
    tokio::fs::write(tree.join("a.txt"), "one two three")
        .await
        .unwrap();
    tokio::fs::write(tree.join("b.rs"), "fn f() {}")
        .await
        .unwrap();

    let first = h
        .call("index_path", json!({ "path": tree.to_string_lossy() }))
        .await;
    assert_eq!(first["indexed"], 2, "{first}");
    assert_eq!(first["skipped"], 0, "{first}");

    let (second, _token) = h
        .call_with_progress("index_path", json!({ "path": tree.to_string_lossy() }))
        .await;
    assert_eq!(second["indexed"], 0, "{second}");
    assert_eq!(second["skipped"], 2, "{second}");
    assert!(
        second["details"].as_array().unwrap().is_empty(),
        "skipped files must not be reported as indexed detail entries: {second}"
    );

    // The walk still acknowledges both files — through progress, so the
    // client can see them being finished.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let unchanged: Vec<String> = h
        .progress_messages()
        .into_iter()
        .flatten()
        .filter(|m| m.contains("unchanged"))
        .collect();
    assert_eq!(unchanged.len(), 2, "got {unchanged:?} from {second}");
}

/// One notification per stored chunk (plus one for every file that
/// produced none), all strictly increasing under the caller's token.
#[tokio::test]
async fn index_path_emits_monotonic_progress_per_chunk() {
    let h = Harness::start().await;
    let tree = h.dir_path().join("progress");
    tokio::fs::create_dir_all(tree.join("sub")).await.unwrap();
    // 512-word chunks with 64 words of overlap: a few thousand words
    // guarantee more than one chunk, hence more than one notification.
    let big = (0..3000)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    tokio::fs::write(tree.join("big.txt"), &big).await.unwrap();
    tokio::fs::write(tree.join("sub").join("small.rs"), "fn main() {}")
        .await
        .unwrap();
    // Not a supported extension: walked, never indexed.
    tokio::fs::write(tree.join("noise.bin"), b"\x00\x01\x02")
        .await
        .unwrap();

    let (resp, token) = h
        .call_with_progress("index_path", json!({ "path": tree.to_string_lossy() }))
        .await;
    assert_eq!(resp["indexed"], 2, "{resp}");

    // Give any trailing notification a beat to be dispatched.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let values = h.progress_values();
    assert!(
        !values.is_empty(),
        "no progress notifications under {token:?}"
    );
    assert!(
        values.windows(2).all(|w| w[0] < w[1]),
        "progress must strictly increase: {values:?}"
    );
    // At least one notification per chunk of the 3000-word file, each
    // naming the file and its `done/total` position.
    assert!(
        values.len() >= 3,
        "expected a notification per chunk, got {values:?}"
    );
    let messages: Vec<String> = h.progress_messages().into_iter().flatten().collect();
    assert!(
        messages
            .iter()
            .any(|m| m.contains("big.txt") && m.contains("chunk")),
        "per-chunk messages missing: {messages:?}"
    );
    assert!(
        messages.iter().any(|m| m.contains("small.rs")),
        "small.rs never acknowledged: {messages:?}"
    );
}

/// A plain `tools/call` must work exactly as well as one the caller
/// opted into progress for. (rmcp always attaches a token of its own, so
/// what this really pins is that opting in or out of *observing* the
/// progress never changes whether the walk succeeds.)
#[tokio::test]
async fn index_path_works_for_a_plain_tools_call() {
    let h = Harness::start().await;
    let tree = h.dir_path().join("silent");
    tokio::fs::create_dir_all(&tree).await.unwrap();
    tokio::fs::write(tree.join("a.txt"), "quiet content")
        .await
        .unwrap();

    let resp = h
        .call("index_path", json!({ "path": tree.to_string_lossy() }))
        .await;
    assert_eq!(resp["status"], "ok", "{resp}");
    assert_eq!(resp["indexed"], 1, "{resp}");
}

impl Harness {
    fn dir_path(&self) -> std::path::PathBuf {
        self._dir.path().to_path_buf()
    }
}
