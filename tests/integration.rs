mod fixtures;

use rust_rag_mcp::chunker::Chunker;
use rust_rag_mcp::docs;
use rust_rag_mcp::mcp::RagServer;
use rust_rag_mcp::rag::RagCore;
use tempfile::tempdir;

const TEST_FIXTURES_DIR: &str = "../../test-fixtures";

/// Helper to create a RagCore for testing (uses small model, fast settings)
async fn test_rag_core(dir: &std::path::Path) -> RagCore {
    RagCore::new(
        dir,
        &dir.join("cache"),
        "Xenova/bge-small-en-v1.5",
        512,
        64,
        150,
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn test_rag_server_instantiation() {
    let dir = tempdir().unwrap();
    let server = RagServer::new(
        dir.path(),
        &dir.path().join("cache"),
        "Xenova/bge-small-en-v1.5",
        512,
        64,
        150,
    )
    .await;
    assert!(
        server.is_ok(),
        "RagServer should instantiate: {:?}",
        server.err()
    );
}

#[tokio::test]
async fn test_rag_index_text_and_search() {
    let dir = tempdir().unwrap();
    let core = test_rag_core(dir.path()).await;

    // Index some text
    let result = core
        .index_text(
            "Rust is a systems programming language focused on safety and performance.",
            "test.rs",
        )
        .await
        .unwrap();
    match result {
        rust_rag_mcp::IndexResult::Indexed(count) => assert!(count > 0),
        rust_rag_mcp::IndexResult::Skipped => panic!("Should not skip first index"),
    }

    // Search for it
    let results = core.search("programming language", 5).await.unwrap();
    assert!(!results.is_empty(), "Should find at least one result");
    assert!(
        results[0].score > 0.0,
        "Score should be positive, got {}",
        results[0].score
    );
    assert!(
        results[0].chunk.text.contains("Rust"),
        "Result should contain 'Rust'"
    );

    core.close().await.unwrap();
}

#[tokio::test]
async fn test_rag_index_text_dedup() {
    let dir = tempdir().unwrap();
    let core = test_rag_core(dir.path()).await;

    let text = "This is a duplicate content test.";

    // First index
    let r1 = core.index_text(text, "dedup.txt").await.unwrap();
    assert!(matches!(r1, rust_rag_mcp::IndexResult::Indexed(_)));

    // Same content again — should skip
    let r2 = core.index_text(text, "dedup.txt").await.unwrap();
    assert!(matches!(r2, rust_rag_mcp::IndexResult::Skipped));

    // Different content, same source — should re-index
    let r3 = core
        .index_text("This is completely different content.", "dedup.txt")
        .await
        .unwrap();
    assert!(matches!(r3, rust_rag_mcp::IndexResult::Indexed(_)));

    core.close().await.unwrap();
}

#[tokio::test]
async fn test_rag_delete_source() {
    let dir = tempdir().unwrap();
    let core = test_rag_core(dir.path()).await;

    core.index_text("Some content to delete.", "delete_me.txt")
        .await
        .unwrap();

    let sources = core.list_sources().await.unwrap();
    assert!(sources.contains(&"delete_me.txt".to_string()));

    core.delete_source("delete_me.txt").await.unwrap();

    let sources = core.list_sources().await.unwrap();
    assert!(!sources.contains(&"delete_me.txt".to_string()));

    core.close().await.unwrap();
}

#[tokio::test]
async fn test_rag_document_status() {
    let dir = tempdir().unwrap();
    let core = test_rag_core(dir.path()).await;

    // Before indexing — should be None
    let status = core.document_status("status_test.txt").await.unwrap();
    assert!(status.is_none());

    // After indexing — should have status
    core.index_text("Status check content.", "status_test.txt")
        .await
        .unwrap();
    let status = core.document_status("status_test.txt").await.unwrap();
    assert!(status.is_some());
    let s = status.unwrap();
    assert_eq!(s.source_path, "status_test.txt");
    assert!(!s.content_hash.is_empty());
    assert!(s.chunk_count > 0);

    core.close().await.unwrap();
}

#[tokio::test]
async fn test_rag_chunk_count() {
    let dir = tempdir().unwrap();
    let core = test_rag_core(dir.path()).await;

    let before = core.chunk_count().await.unwrap();
    assert_eq!(before, 0);

    core.index_text("First document.", "a.txt").await.unwrap();
    core.index_text("Second document.", "b.txt").await.unwrap();

    let after = core.chunk_count().await.unwrap();
    assert!(after >= 2, "Should have at least 2 chunks, got {after}");

    core.close().await.unwrap();
}

/// Full `.srcidx` sidecar lifecycle: close persists it, a reopen trusts
/// it, and a sidecar whose generation tokens no longer match the
/// database (stale from an older run, or plain garbage) is ignored in
/// favor of the metadata scan — never believed, never fatal.
#[tokio::test]
async fn test_rag_source_index_sidecar_reopen() {
    let dir = tempdir().unwrap();
    let sidecar = dir.path().join("db.srcidx");

    // Generation 1: two sources, re-indexed once so the tokens cover an
    // insert *and* a delete (tombstone) mutation.
    let gen1_sources = {
        let core = test_rag_core(dir.path()).await;
        core.index_text("Rust is a systems programming language.", "a.rs")
            .await
            .unwrap();
        core.index_text("Ownership and borrowing make Rust unique.", "a.rs")
            .await
            .unwrap();
        core.index_text("Memory safety without a garbage collector.", "b.md")
            .await
            .unwrap();
        let sources = core.list_sources().await.unwrap();
        assert_eq!(sources, vec!["a.rs".to_string(), "b.md".to_string()]);
        core.close().await.unwrap();
        sources
    }; // core dropped here: the db file locks release for the reopen

    assert!(sidecar.exists(), "close must persist the sidecar");
    let gen1 = std::fs::read(&sidecar).unwrap();

    // Reopen on the same generation: the sidecar validates and must
    // reproduce the exact same view (this is the skip-the-scan path).
    {
        let core = test_rag_core(dir.path()).await;
        assert_eq!(core.list_sources().await.unwrap(), gen1_sources);
        core.close().await.unwrap();
    }

    // Generation 2: one more source. The sidecar on disk now describes
    // a newer database than gen1.
    {
        let core = test_rag_core(dir.path()).await;
        core.index_text("Second generation content.", "c.txt")
            .await
            .unwrap();
        core.close().await.unwrap();
    }
    let all_sources = vec!["a.rs".to_string(), "b.md".to_string(), "c.txt".to_string()];

    // Stale sidecar (gen1 against a gen2 database): tokens mismatch ->
    // rebuild from metadata -> the *new* truth, not the sidecar's.
    std::fs::write(&sidecar, &gen1).unwrap();
    {
        let core = test_rag_core(dir.path()).await;
        assert_eq!(
            core.list_sources().await.unwrap(),
            all_sources,
            "a stale sidecar must be rejected in favor of the scan"
        );
        core.close().await.unwrap();
    }

    // Corrupt/foreign sidecar: same fallback, no error.
    std::fs::write(&sidecar, b"not a sidecar at all").unwrap();
    {
        let core = test_rag_core(dir.path()).await;
        assert_eq!(core.list_sources().await.unwrap(), all_sources);
        core.close().await.unwrap();
    }
}

/// Hybrid search end-to-end: lexical mode finds exact identifiers,
/// hybrid fuses both retrievers, and the RRF score contract holds
/// (positive, rank-monotonic, capped at one contribution per list).
#[tokio::test]
async fn test_rag_hybrid_and_lexical_search() {
    let dir = tempdir().unwrap();
    let core = test_rag_core(dir.path()).await;

    core.index_text(
        "Rust is a systems programming language focused on safety.",
        "rust.md",
    )
    .await
    .unwrap();
    core.index_text(
        "The parse_source_index function validates sidecar tokens.",
        "sidecar.rs",
    )
    .await
    .unwrap();
    core.index_text("Banana bread recipes for beginners.", "cooking.txt")
        .await
        .unwrap();

    // Lexical mode: exact identifier, ranked first — the query never
    // touches the embedding model, and the tokenizer splits the
    // identifier the same way on both sides (underscore boundaries).
    let lexical = core
        .search_with_mode("parse_source_index", 5, rust_rag_mcp::SearchMode::Lexical)
        .await
        .unwrap();
    assert!(!lexical.is_empty(), "exact identifier must match lexically");
    assert_eq!(lexical[0].chunk.source, "sidecar.rs");
    assert!(lexical[0].score > 0.0);

    // Semantic mode keeps the dense-only contract.
    let semantic = core
        .search_with_mode(
            "systems programming safety",
            5,
            rust_rag_mcp::SearchMode::Semantic,
        )
        .await
        .unwrap();
    assert!(!semantic.is_empty());
    assert!(semantic[0].score > 0.0);

    // Hybrid: every document appears in both retrievers' pools (the
    // corpus is smaller than the pool), so the top fused score is
    // exactly two RRF contributions — 2/(60+1) — and never more, and
    // every score is positive and rank-monotonic.
    let hybrid = core
        .search_with_mode(
            "parse_source_index safety",
            5,
            rust_rag_mcp::SearchMode::Hybrid,
        )
        .await
        .unwrap();
    assert!(!hybrid.is_empty());
    assert!(
        hybrid[0].score <= 2.0 / 61.0 + 1e-9,
        "top RRF score must be one contribution per list, got {}",
        hybrid[0].score
    );
    for pair in hybrid.windows(2) {
        assert!(
            pair[0].score >= pair[1].score,
            "hybrid results must be sorted by fused score"
        );
    }
    assert!(
        hybrid.iter().any(|r| r.chunk.source == "sidecar.rs"),
        "fused results must contain the lexically matched document"
    );

    // top_k is respected in every mode.
    for mode in [
        rust_rag_mcp::SearchMode::Hybrid,
        rust_rag_mcp::SearchMode::Semantic,
        rust_rag_mcp::SearchMode::Lexical,
    ] {
        let results = core.search_with_mode("rust", 2, mode).await.unwrap();
        assert!(results.len() <= 2, "{mode:?} exceeded top_k");
    }

    core.close().await.unwrap();
}

/// Deleting a source must drop it from the lexical index too — both
/// via the in-process `remove_document` wiring and (as backstop) the
/// query-time liveness guard — while leaving other sources intact.
#[tokio::test]
async fn test_rag_delete_source_removes_from_lexical_index() {
    let dir = tempdir().unwrap();
    let core = test_rag_core(dir.path()).await;

    core.index_text(
        "The flibbertigibbet token appears only in the doomed document.",
        "doomed.txt",
    )
    .await
    .unwrap();
    core.index_text("Eternal content about giraffes.", "keeper.txt")
        .await
        .unwrap();

    let before = core.search_lexical("flibbertigibbet", 5).await.unwrap();
    assert_eq!(before.len(), 1, "token indexed exactly once");

    core.delete_source("doomed.txt").await.unwrap();

    assert!(
        core.search_lexical("flibbertigibbet", 5)
            .await
            .unwrap()
            .is_empty(),
        "deleted content must not surface lexically"
    );
    // Hybrid can never be *empty* for any query — the dense side
    // returns the nearest live neighbours even for gibberish — so the
    // contract here is narrower: the dead document must not appear.
    let hybrid = core.search_hybrid("flibbertigibbet", 5).await.unwrap();
    assert!(
        hybrid.iter().all(|r| r.chunk.source != "doomed.txt"),
        "deleted content must not surface in hybrid results: {hybrid:?}"
    );
    assert!(
        !core.search_lexical("giraffes", 5).await.unwrap().is_empty(),
        "surviving source stays searchable"
    );

    core.close().await.unwrap();
}

/// Full `.bm25` sidecar lifecycle, mirroring `.srcidx`: close
/// persists it, a reopen trusts it, and any deviation — stale
/// generation tokens (including an insert that happened after the last
/// persist and was never closed), plain garbage — forces the rebuild
/// that makes lexical search correct again.
#[tokio::test]
async fn test_rag_bm25_sidecar_reopen() {
    let dir = tempdir().unwrap();
    let sidecar = dir.path().join("db.bm25");

    // Generation 1, clean close: the sidecar lands on disk.
    {
        let core = test_rag_core(dir.path()).await;
        core.index_text("Alpha content about herons.", "a.txt")
            .await
            .unwrap();
        core.index_text("Beta content about egrets.", "b.txt")
            .await
            .unwrap();
        core.close().await.unwrap();
    }
    assert!(sidecar.exists(), "close must persist the bm25 sidecar");
    let gen1 = std::fs::read(&sidecar).unwrap();

    // Reopen on the same generation: sidecar validates; lexical search
    // must reproduce the corpus (the skip-the-scan path).
    {
        let core = test_rag_core(dir.path()).await;
        assert!(!core.search_lexical("herons", 5).await.unwrap().is_empty());
        assert!(!core.search_lexical("egrets", 5).await.unwrap().is_empty());
        core.close().await.unwrap();
    }

    // Generation 2: a new source, closed — the on-disk sidecar now
    // describes a newer database than gen1.
    {
        let core = test_rag_core(dir.path()).await;
        core.index_text("Xylophone content in the newer generation.", "c.txt")
            .await
            .unwrap();
        core.close().await.unwrap();
    }

    // Stale sidecar (gen1 against gen2): tokens mismatch -> rebuild —
    // trusting the stale file would silently lose "xylophone".
    std::fs::write(&sidecar, &gen1).unwrap();
    {
        let core = test_rag_core(dir.path()).await;
        assert!(
            !core
                .search_lexical("xylophone", 5)
                .await
                .unwrap()
                .is_empty(),
            "a stale sidecar must be rejected in favor of the scan"
        );
        core.close().await.unwrap();
    }

    // Corrupt/foreign sidecar: same fallback, no error.
    std::fs::write(&sidecar, b"not a sidecar at all").unwrap();
    {
        let core = test_rag_core(dir.path()).await;
        assert!(!core.search_lexical("herons", 5).await.unwrap().is_empty());
        core.close().await.unwrap();
    }

    // Crash simulation: an insert after the last persist, never
    // closed. The sidecar on disk still carries the pre-insert tokens,
    // so the next open must reject it and see the un-persisted insert.
    {
        let core = test_rag_core(dir.path()).await;
        core.index_text("Wombat content appears after the crash point.", "d.txt")
            .await
            .unwrap();
        // deliberately no close: drop == crash from the sidecar's view
    }
    {
        let core = test_rag_core(dir.path()).await;
        assert!(
            !core.search_lexical("wombat", 5).await.unwrap().is_empty(),
            "an insert invalidates the previously persisted sidecar tokens"
        );
        // Hybrid runs off the rebuilt index too.
        assert!(!core.search_hybrid("wombat", 5).await.unwrap().is_empty());
        core.close().await.unwrap();
    }
}

/// Deletes are WAL-only (`delete_source` never flushes), so both sidecars
/// must stay correct across a crash whose recovery runs entirely through
/// WAL replay: the replayed tombstones move `meta_record_count`, the old
/// generation tokens no longer match, and the rebuild from the scan sees
/// the post-replay state. The crash state is pinned by rolling the data
/// files back to the pre-delete bytes while leaving the WAL and both
/// sidecars in place — without replay, the restored files would present
/// the deleted source as live and validate the stale sidecars.
#[tokio::test]
async fn test_rag_wal_replayed_deletes_invalidate_sidecars() {
    let dir = tempdir().unwrap();
    let srcidx = dir.path().join("db.srcidx");
    let bm25_sidecar = dir.path().join("db.bm25");
    let data_files = ["db", "db.hnsw", "db.meta"].map(|n| dir.path().join(n));

    // Generation 1: both sources indexed, closed — sidecars on disk.
    let chunks_before = {
        let core = test_rag_core(dir.path()).await;
        core.index_text(
            "The flibbertigibbet token appears only in the doomed document.",
            "doomed.txt",
        )
        .await
        .unwrap();
        core.index_text("Eternal content about giraffes.", "keeper.txt")
            .await
            .unwrap();
        assert!(
            !core
                .search_lexical("flibbertigibbet", 5)
                .await
                .unwrap()
                .is_empty(),
            "baseline: the doomed token is indexed"
        );
        let n = core.chunk_count().await.unwrap();
        core.close().await.unwrap();
        n
    };
    let srcidx_gen1 = std::fs::read(&srcidx).unwrap();
    let bm25_gen1 = std::fs::read(&bm25_sidecar).unwrap();
    let backups: Vec<Vec<u8>> = data_files
        .iter()
        .map(|p| std::fs::read(p).unwrap())
        .collect();

    // The crash window: reopen, delete without ever flushing, then drop.
    // Tombstones live only in the WAL (fsynced by the writer's shutdown)
    // and in dirty pages the restore below discards.
    {
        let core = test_rag_core(dir.path()).await;
        core.delete_source("doomed.txt").await.unwrap();
        // deliberately no close: drop == crash
    }
    for (p, bytes) in data_files.iter().zip(&backups) {
        std::fs::write(p, bytes).unwrap();
    }

    // Recovery: replay must apply the tombstones, which moves the
    // generation tokens, which rejects both pre-delete sidecars.
    {
        let core = test_rag_core(dir.path()).await;

        assert_eq!(
            core.list_sources().await.unwrap(),
            vec!["keeper.txt".to_string()],
            "a stale .srcidx trusted through replay would list the phantom source"
        );
        assert!(
            core.search_lexical("flibbertigibbet", 5)
                .await
                .unwrap()
                .is_empty(),
            "without replay the restored rows are live and the stale sidecar would match"
        );
        assert!(
            !core.search_lexical("giraffes", 5).await.unwrap().is_empty(),
            "the surviving source stays lexically searchable"
        );
        let hybrid = core.search_hybrid("flibbertigibbet", 5).await.unwrap();
        assert!(
            hybrid.iter().all(|r| r.chunk.source != "doomed.txt"),
            "dead content must not surface in hybrid results: {hybrid:?}"
        );
        assert_eq!(
            core.chunk_count().await.unwrap(),
            chunks_before,
            "tombstones never shrink the row count"
        );
        core.close().await.unwrap();
    }

    // A rebuilt payload is strictly smaller than the generation it
    // replaced; trusting and re-persisting the stale file would keep its
    // bytes (only the token header would differ).
    assert!(
        std::fs::read(&srcidx).unwrap().len() < srcidx_gen1.len(),
        "the rebuilt .srcidx must have dropped the deleted source"
    );
    assert!(
        std::fs::read(&bm25_sidecar).unwrap().len() < bm25_gen1.len(),
        "the rebuilt .bm25 must have dropped the deleted postings"
    );
}

/// The MCP server never reached `close()` before (only the CLI commands
/// did), which would have left the sidecar stale on every server
/// shutdown. Pin both persistence points of the server path — startup
/// rebuild and `RagServer::close` — for both sidecars (`.srcidx` and
/// `.bm25`).
#[tokio::test]
async fn test_rag_server_close_persists_sidecar() {
    let dir = tempdir().unwrap();
    let server = RagServer::new(
        dir.path(),
        &dir.path().join("cache"),
        "Xenova/bge-small-en-v1.5",
        512,
        64,
        150,
    )
    .await
    .unwrap();

    let sidecar = dir.path().join("db.srcidx");
    let bm25_sidecar = dir.path().join("db.bm25");
    assert!(sidecar.exists(), "startup must persist the sidecar");
    assert!(
        bm25_sidecar.exists(),
        "startup must persist the bm25 sidecar"
    );

    // Remove both so only close() can bring them back — this pins the
    // shutdown wiring in main.rs (close after the transport ends).
    std::fs::remove_file(&sidecar).unwrap();
    std::fs::remove_file(&bm25_sidecar).unwrap();
    server.close().await.unwrap();
    assert!(sidecar.exists(), "close must re-persist the sidecar");
    assert!(
        bm25_sidecar.exists(),
        "close must re-persist the bm25 sidecar"
    );
}

#[tokio::test]
async fn test_extract_sample_pdf() {
    let pdf_path = std::path::Path::new(TEST_FIXTURES_DIR).join("sample.pdf");
    if !pdf_path.exists() {
        eprintln!("Skipping - sample.pdf not found");
        return;
    }
    let text = docs::extract_text(&pdf_path).await.unwrap();
    assert!(!text.is_empty(), "Should extract text from real PDF");
    assert!(
        text.len() > 100,
        "Should extract meaningful content: {} chars",
        text.len()
    );
}

#[tokio::test]
async fn test_extract_sample_docx() {
    let docx_path = std::path::Path::new(TEST_FIXTURES_DIR).join("sample.docx");
    if !docx_path.exists() {
        eprintln!("Skipping - sample.docx not found");
        return;
    }
    let text = docs::extract_text(&docx_path).await.unwrap();
    assert!(!text.is_empty(), "Should extract text from real DOCX");
    assert!(
        text.len() > 50,
        "Should extract meaningful content: {} chars",
        text.len()
    );
}

#[tokio::test]
async fn test_extract_sample_xlsx() {
    let xlsx_path = std::path::Path::new(TEST_FIXTURES_DIR).join("sample.xlsx");
    if !xlsx_path.exists() {
        eprintln!("Skipping - sample.xlsx not found");
        return;
    }
    let text = docs::extract_text(&xlsx_path).await.unwrap();
    assert!(!text.is_empty(), "Should extract text from real XLSX");
    assert!(
        text.len() > 10,
        "Should extract meaningful content: {} chars",
        text.len()
    );
}

#[tokio::test]
async fn test_extract_sample_ppt() {
    let ppt_path = std::path::Path::new(TEST_FIXTURES_DIR).join("sample.pptx");
    if !ppt_path.exists() {
        eprintln!("Skipping - sample.pptx not found");
        return;
    }
    let text = docs::extract_text(&ppt_path).await.unwrap();
    assert!(!text.is_empty(), "Should extract text from real PPT");
    assert!(
        text.len() > 10,
        "Should extract meaningful content: {} chars",
        text.len()
    );
}

#[tokio::test]
async fn test_extract_txt_file() {
    let dir = tempdir().unwrap();
    let txt_path = dir.path().join("test.txt");
    let sample = fixtures::sample_text();
    tokio::fs::write(&txt_path, &sample).await.unwrap();

    let text = docs::extract_text(&txt_path).await.unwrap();
    assert!(text.contains("Rust is a systems programming language"));
    assert!(text.contains("tokio runtime"));
}

#[test]
fn test_chunker_preserves_content() {
    let chunker = Chunker::new(20, 5);
    let text = "The quick brown fox jumps over the lazy dog. A second sentence for testing.";
    let chunks = chunker.chunk_text(text, "test.txt");

    let combined: String = chunks
        .iter()
        .map(|c| c.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(combined.contains("quick brown fox"));
    assert!(combined.contains("second sentence"));
}

#[test]
fn test_supported_extensions() {
    assert!(docs::supported_extension(std::path::Path::new("doc.pdf")));
    assert!(docs::supported_extension(std::path::Path::new("doc.docx")));
    assert!(docs::supported_extension(std::path::Path::new("doc.xlsx")));
    assert!(docs::supported_extension(std::path::Path::new("doc.pptx")));
    assert!(docs::supported_extension(std::path::Path::new(
        "readme.txt"
    )));
    assert!(docs::supported_extension(std::path::Path::new("code.rs")));
    assert!(docs::supported_extension(std::path::Path::new("data.json")));
    assert!(!docs::supported_extension(std::path::Path::new(
        "image.png"
    )));
    assert!(!docs::supported_extension(std::path::Path::new(
        "binary.exe"
    )));
}

#[tokio::test]
async fn test_extract_unsupported_returns_error() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.xyz");
    tokio::fs::write(&path, "data").await.unwrap();
    assert!(docs::extract_text(&path).await.is_err());
}

#[test]
fn test_chunker_empty_and_whitespace() {
    let chunker = Chunker::default();
    assert!(chunker.chunk_text("", "src.txt").is_empty());
    assert!(chunker.chunk_text("  \n\t  ", "src.txt").is_empty());
}

#[test]
fn test_chunker_single_word() {
    let chunker = Chunker::new(100, 10);
    let chunks = chunker.chunk_text("hello", "test.txt");
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, "hello");
    assert_eq!(chunks[0].source, "test.txt");
}

#[test]
fn test_chunker_offset_accuracy() {
    let chunker = Chunker::new(5, 0);
    let text = "alpha bravo charlie delta echo";
    let chunks = chunker.chunk_text(text, "test.txt");

    for chunk in &chunks {
        let reconstructed = &text[chunk.start_offset..chunk.end_offset];
        assert_eq!(
            reconstructed, chunk.text,
            "offset mismatch for chunk {}",
            chunk.chunk_index
        );
    }
}
