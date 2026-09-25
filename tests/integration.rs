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

/// The MCP server never reached `close()` before (only the CLI commands
/// did), which would have left the sidecar stale on every server
/// shutdown. Pin both persistence points of the server path: startup
/// rebuild and `RagServer::close`.
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
    assert!(sidecar.exists(), "startup must persist the sidecar");

    // Remove it so only close() can bring it back — this pins the
    // shutdown wiring in main.rs (close after the transport ends).
    std::fs::remove_file(&sidecar).unwrap();
    server.close().await.unwrap();
    assert!(sidecar.exists(), "close must re-persist the sidecar");
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
