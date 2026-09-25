//! End-to-end tests for the `rust-rag-mcp` binary.
//!
//! `main.rs` is only reachable by running the real executable, so these
//! tests spawn it and assert on stdout — which doubles as a check that
//! the CLI wiring (flags, defaults, exit codes) works for a user.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use tempfile::TempDir;

mod fixtures;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_rust-rag-mcp"))
}

struct Cli {
    /// Temp directory holding the database; kept alive across runs.
    _dir: TempDir,
    db: String,
    cache: String,
}

impl Cli {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("db");
        Cli {
            db: db.to_string_lossy().into_owned(),
            // The shared `.fastembed_cache` (see `fixtures::model_cache_dir`)
            // keeps these tests offline and fast.
            cache: fixtures::model_cache_dir().to_string_lossy().into_owned(),
            _dir: dir,
        }
    }

    /// Run the binary and capture its output.
    fn run(&self, args: &[&str]) -> std::process::Output {
        bin()
            .args(args)
            .args(["--db-path", &self.db, "--cache-path", &self.cache])
            .output()
            .expect("spawn rust-rag-mcp")
    }

    fn stdout(&self, args: &[&str]) -> String {
        let output = self.run(args);
        assert!(
            output.status.success(),
            "`{args:?}` failed\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).expect("stdout is utf8")
    }
}

#[test]
fn help_lists_every_subcommand() {
    let output = bin().arg("--help").output().unwrap();
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    for subcommand in [
        "serve", "index", "search", "sources", "stats", "delete", "models",
    ] {
        assert!(help.contains(subcommand), "`{subcommand}` missing:\n{help}");
    }
}

#[test]
fn models_prints_the_catalog() {
    let output = bin().arg("models").output().unwrap();
    assert!(output.status.success());
    let listed = String::from_utf8(output.stdout).unwrap();
    assert!(listed.contains("Available embedding models:"), "{listed}");
    // The default model must be advertised, or `--model` has no visible
    // way to discover what the server runs out of the box.
    assert!(
        listed.contains("Xenova/bge-small-en-v1.5"),
        "default model missing:\n{listed}"
    );
    assert!(listed.contains("BAAI/bge-base-en-v1.5"), "{listed}");
}

#[test]
fn unknown_subcommand_exits_with_a_usage_error() {
    let output = bin().arg("frobnicate").output().unwrap();
    assert!(!output.status.success());
    let err = String::from_utf8_lossy(&output.stderr);
    assert!(err.contains("frobnicate"), "{err}");
}

/// The whole lifecycle a user drives from a shell: index a file, search
/// it in every mode, list sources and stats, then delete it.
#[test]
fn index_search_sources_stats_delete_roundtrip() {
    let cli = Cli::new();
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("note.txt");
    std::fs::write(
        &file,
        "Rust is a systems programming language focused on safety and performance.\n",
    )
    .unwrap();
    let file_arg = file.to_string_lossy().into_owned();

    let indexed = cli.stdout(&["index", &file_arg]);
    assert!(indexed.contains("Indexed"), "{indexed}");
    assert!(indexed.contains("1 chunks"), "{indexed}");

    // Re-running must report the unchanged file rather than re-indexing.
    let rerun = cli.stdout(&["index", &file_arg]);
    assert!(rerun.contains("Skipped"), "{rerun}");

    // Hybrid (default): the line carries the RRF weight plus both
    // retriever components behind it.
    let hybrid = cli.stdout(&["search", "programming language"]);
    assert!(hybrid.contains("[1]"), "{hybrid}");
    assert!(hybrid.contains("(rrf,"), "{hybrid}");
    assert!(hybrid.contains("dense:"), "{hybrid}");
    assert!(hybrid.contains("bm25:"), "{hybrid}");

    // Semantic: cosine score, no component suffix at all.
    let semantic = cli.stdout(&["search", "programming language", "--mode", "semantic"]);
    assert!(semantic.contains("[1]"), "{semantic}");
    assert!(semantic.contains("(cosine,"), "{semantic}");
    assert!(!semantic.contains("dense:"), "{semantic}");

    // Lexical: raw BM25 weight, no model invoked for the query itself.
    let lexical = cli.stdout(&["search", "programming language", "--mode", "lexical"]);
    assert!(lexical.contains("[1]"), "{lexical}");
    assert!(lexical.contains("(bm25,"), "{lexical}");
    assert!(!lexical.contains("dense:"), "{lexical}");

    let sources = cli.stdout(&["sources"]);
    let source = sources
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with("No indexed"))
        .expect("a source is listed")
        .to_string();
    assert_eq!(
        std::path::Path::new(&source),
        file.canonicalize().unwrap().as_path(),
        "listed source must be the canonical path: {sources}"
    );

    let stats = cli.stdout(&["stats"]);
    assert!(stats.contains("Indexed chunks: "), "{stats}");
    assert!(stats.contains("Indexed sources: 1"), "{stats}");

    let deleted = cli.stdout(&["delete", &source]);
    assert!(deleted.contains("Deleted:"), "{deleted}");

    // Everything gone: an empty result set prints the empty-state line.
    let empty = cli.stdout(&["search", "programming language"]);
    assert!(empty.contains("No results found."), "{empty}");

    // Deleting again is not an error — the source simply is not there.
    let again = cli.stdout(&["delete", &source]);
    assert!(again.contains("Deleted:"), "{again}");
}

/// A path that does not exist is reported on stderr and skipped rather
/// than aborting: `index` still processes the rest of its arguments.
#[test]
fn index_missing_path_is_reported_and_not_fatal() {
    let cli = Cli::new();
    let src_dir = tempfile::tempdir().unwrap();
    let good = src_dir.path().join("note.md");
    std::fs::write(&good, "Rust ownership keeps memory safe without a GC.\n").unwrap();

    let output = cli.run(&["index", "/no/such/path", good.to_str().unwrap()]);
    assert!(
        output.status.success(),
        "status={:?} stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let err = String::from_utf8_lossy(&output.stderr);
    assert!(err.contains("Path not found"), "{err}");
    let out = String::from_utf8_lossy(&output.stdout);
    assert!(
        out.contains("Indexed"),
        "the real file must still run: {out}"
    );
}

/// Deleting a source that was never indexed is a successful no-op:
/// `delete_source` removes whatever is registered under the name and
/// reports success either way, so the CLI never fails the command.
#[test]
fn delete_unknown_source_is_a_successful_noop() {
    let cli = Cli::new();
    let output = cli.run(&["delete", "/never/indexed/at/all"]);
    assert!(
        output.status.success(),
        "status={:?} stderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty(), "nothing should fail here");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Deleted:"), "{stdout}");
}

/// The `serve` subcommand must complete an MCP handshake over stdio and
/// then shut down cleanly — closing the sidecars — when stdin ends.
#[test]
fn serve_handshakes_over_stdio_and_shuts_down_on_eof() {
    let cli = Cli::new();
    let mut child = bin()
        .args(["serve", "--db-path", &cli.db, "--cache-path", &cli.cache])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn serve");

    // Drain stderr on its own thread so a chatty server can never fill
    // the pipe and deadlock the handshake below.
    let stderr_task = std::thread::spawn({
        let mut stderr = child.stderr.take().expect("stderr piped");
        move || {
            let mut buf = String::new();
            let _ = std::io::Read::read_to_string(&mut stderr, &mut buf);
            buf
        }
    });

    {
        let mut stdin = child.stdin.take().unwrap();
        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"2025-11-25","capabilities":{{}},"clientInfo":{{"name":"cli-test","version":"0.0.1"}}}}}}"#
        )
        .unwrap();
        writeln!(
            stdin,
            r#"{{"jsonrpc":"2.0","method":"notifications/initialized"}}"#
        )
        .unwrap();
        stdin.flush().unwrap();

        let stdout = child.stdout.take().unwrap();
        let mut line = String::new();
        BufReader::new(stdout)
            .read_line(&mut line)
            .expect("read initialize response");
        let response: serde_json::Value =
            serde_json::from_str(&line).unwrap_or_else(|e| panic!("bad response ({e}): {line}"));
        assert_eq!(response["id"], 1, "{response}");
        assert!(response["result"]["serverInfo"].is_object(), "{response}");
        assert!(
            response["result"]["capabilities"]["tools"].is_object(),
            "tools capability not advertised: {response}"
        );
        // stdin drops here: EOF is what tells the transport to finish.
    }

    let deadline = Instant::now() + Duration::from_secs(60);
    let status = loop {
        match child.try_wait().expect("try_wait") {
            Some(status) => break status,
            None if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(50));
            }
            None => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("`serve` did not exit within 60s of stdin closing");
            }
        }
    };
    let stderr = stderr_task.join().unwrap_or_default();
    assert!(
        status.success(),
        "`serve` exited with {status}; stderr: {stderr}"
    );
}
