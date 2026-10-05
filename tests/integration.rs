mod fixtures;

use rust_rag_mcp::chunker::Chunker;
use rust_rag_mcp::docs;
use rust_rag_mcp::mcp::RagServer;
use rust_rag_mcp::rag::RagCore;
use tempfile::tempdir;

/// Fixture directory, anchored on the manifest dir so it resolves no matter
/// what the runner's working directory is. (The old relative
/// `"../../test-fixtures"` pointed *outside* the repo from the package root,
/// so every fixture extraction test passed by silently skipping.)
fn fixtures_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-fixtures")
}

/// Helper to create a RagCore for testing (uses small model, fast settings)
async fn test_rag_core(dir: &std::path::Path) -> RagCore {
    RagCore::new(
        dir,
        &fixtures::model_cache_dir(),
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
        &fixtures::model_cache_dir(),
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

/// Whole-document matching: a document (new text, new file, or an
/// already-indexed source) is compared against the index as a whole and
/// ranked per *document* — the related source first, the query document
/// itself never, and every match carrying the shared query-chunk
/// denominator behind its score.
#[tokio::test]
async fn test_rag_match_document_ranks_whole_documents() {
    let dir = tempdir().unwrap();
    let core = test_rag_core(dir.path()).await;

    core.index_text(
        "Rust is a systems programming language focused on safety and performance.",
        "rust.txt",
    )
    .await
    .unwrap();
    core.index_text(
        "Gardening advice: tomatoes thrive in warm soil with regular watering.",
        "garden.txt",
    )
    .await
    .unwrap();

    // Raw, unindexed text: the related document ranks first.
    let query = rust_rag_mcp::MatchDocument::Text {
        text: "systems programming language safety",
        ext: "",
    };
    let res = core
        .match_document(&query, &rust_rag_mcp::MatchOptions::default())
        .await
        .unwrap();
    assert!(res.query_source.is_none(), "{res:?}");
    assert!(res.query_chunks >= 1, "{res:?}");
    assert!(!res.matches.is_empty(), "{res:?}");
    assert_eq!(res.matches[0].source, "rust.txt", "{res:?}");
    assert!(res.matches[0].matched_chunks >= 1, "{res:?}");
    assert!(res.matches[0].score > 0.0, "{res:?}");
    // The mean never exceeds the best single-chunk hit it is built from.
    assert!(
        res.matches[0].best_score + 1e-9 >= res.matches[0].score,
        "{res:?}"
    );
    assert!(
        res.matches
            .iter()
            .all(|m| m.query_chunks == res.query_chunks),
        "{res:?}"
    );

    // `top_k` truncates the ranking.
    let res = core
        .match_document(
            &query,
            &rust_rag_mcp::MatchOptions {
                top_k: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap();
    assert_eq!(res.matches.len(), 1, "{res:?}");

    // An indexed source is queried with its own stored chunks and must
    // never rank against itself.
    let res = core
        .match_document(
            &rust_rag_mcp::MatchDocument::Indexed { source: "rust.txt" },
            &rust_rag_mcp::MatchOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(res.query_source.as_deref(), Some("rust.txt"), "{res:?}");
    assert!(res.query_chunks >= 1, "{res:?}");
    assert!(
        res.matches.iter().all(|m| m.source != "rust.txt"),
        "a document must not match itself: {res:?}"
    );
    assert!(
        res.matches.iter().any(|m| m.source == "garden.txt"),
        "{res:?}"
    );

    // An unknown source is an error, not an empty ranking.
    let err = core
        .match_document(
            &rust_rag_mcp::MatchDocument::Indexed {
                source: "never.txt",
            },
            &rust_rag_mcp::MatchOptions::default(),
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not indexed"), "{err}");

    // A new file resolves to its canonical path — the same key indexing
    // would file it under, and therefore the source excluded from its
    // own matches.
    let file_dir = tempdir().unwrap();
    let new_doc = file_dir.path().join("borrow-checker.txt");
    std::fs::write(
        &new_doc,
        "The Rust borrow checker enforces memory safety at compile time.\n",
    )
    .unwrap();
    let res = core
        .match_document(
            &rust_rag_mcp::MatchDocument::File {
                path: new_doc.as_path(),
            },
            &rust_rag_mcp::MatchOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        res.query_source.as_deref(),
        Some(new_doc.canonicalize().unwrap().to_str().unwrap()),
        "{res:?}"
    );
    assert!(!res.matches.is_empty(), "{res:?}");
    assert_eq!(res.matches[0].source, "rust.txt", "{res:?}");

    // A code file must be split exactly as `index` splits it: the same
    // syntax-aware boundaries, so the query-chunk count equals the
    // stored chunk count. (A chunking label without the extension would
    // silently fall back to the generic word chunker and mismatch.)
    let code_dir = tempdir().unwrap();
    let code_doc = code_dir.path().join("sample.rs");
    std::fs::write(
        &code_doc,
        "fn alpha() {}\npub struct Beta;\npub fn gamma() {}\n",
    )
    .unwrap();
    let stored = match core.index_file(&code_doc).await.unwrap() {
        rust_rag_mcp::IndexResult::Indexed(count) => count,
        rust_rag_mcp::IndexResult::Skipped => panic!("first index must store chunks"),
    };
    let res = core
        .match_document(
            &rust_rag_mcp::MatchDocument::File {
                path: code_doc.as_path(),
            },
            &rust_rag_mcp::MatchOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        res.query_chunks as usize, stored,
        "matching must chunk the file like indexing does: {res:?}"
    );
    assert!(
        res.matches
            .iter()
            .all(|m| m.source != res.query_source.as_deref().unwrap()),
        "a document must not match itself: {res:?}"
    );

    core.close().await.unwrap();
}

/// `list_sources_page` walks the sorted source list in stable pages:
/// the page is a contiguous slice of `list_sources`, `total` always
/// reports the full count, and offsets past the end (or a zero limit)
/// yield empty pages rather than errors.
#[tokio::test]
async fn test_rag_list_sources_page_pagination() {
    let dir = tempdir().unwrap();
    let core = test_rag_core(dir.path()).await;

    for i in 0..5 {
        core.index_text(
            &format!("content of document number {i}"),
            &format!("doc{i}.txt"),
        )
        .await
        .unwrap();
    }

    let (page, total) = core.list_sources_page(0, 2).await.unwrap();
    assert_eq!(total, 5, "total counts every source, not the page");
    assert_eq!(page, vec!["doc0.txt", "doc1.txt"], "sorted slice");

    let (page, _) = core.list_sources_page(2, 2).await.unwrap();
    assert_eq!(page, vec!["doc2.txt", "doc3.txt"]);

    let (page, _) = core.list_sources_page(4, 2).await.unwrap();
    assert_eq!(page, vec!["doc4.txt"], "the last page may be short");

    let (page, total) = core.list_sources_page(5, 2).await.unwrap();
    assert!(
        page.is_empty(),
        "offset past the end is empty, not an error"
    );
    assert_eq!(total, 5);

    let (page, _) = core.list_sources_page(0, 0).await.unwrap();
    assert!(page.is_empty(), "limit 0 yields an empty page");

    // Pages are exactly a partition of the full listing.
    let (first, _) = core.list_sources_page(0, 5).await.unwrap();
    assert_eq!(first, core.list_sources().await.unwrap());

    core.close().await.unwrap();
}

/// Chunk progress must fire once per stored chunk, counting `1..=total` in
/// insertion order — each report is what the MCP layer forwards as a
/// `notifications/progress` message. A skipped (unchanged) file produces no
/// chunk, so it must produce no report either.
#[tokio::test]
async fn test_index_file_reports_progress_per_chunk() {
    use std::sync::Arc;

    let dir = tempdir().unwrap();
    let core = test_rag_core(dir.path()).await;

    // Chunks hold 512 words with 64 words of overlap, so a few thousand
    // words guarantee more than one chunk.
    let text = (0..3000)
        .map(|i| format!("word{i}"))
        .collect::<Vec<_>>()
        .join(" ");
    let path = dir.path().join("progress.txt");
    tokio::fs::write(&path, &text).await.unwrap();

    let reports = Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = reports.clone();
    let result = core
        .index_file_with_progress(&path, move |done, total| {
            let sink = sink.clone();
            async move {
                sink.lock().unwrap().push((done, total));
            }
        })
        .await
        .unwrap();
    let count = match result {
        rust_rag_mcp::IndexResult::Indexed(count) => count,
        rust_rag_mcp::IndexResult::Skipped => panic!("Fresh file must be indexed"),
    };
    assert!(count > 1, "Test needs several chunks, got {count}");
    {
        let reports = reports.lock().unwrap();
        assert_eq!(reports.len(), count, "Exactly one report per chunk");
        for (i, &(done, total)) in reports.iter().enumerate() {
            assert_eq!(
                (done, total),
                (i + 1, count),
                "Reports must run 1..={count} in order"
            );
        }
    }

    // Unchanged content is skipped: no chunk, no report.
    let skipped_reports = Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = skipped_reports.clone();
    let result = core
        .index_file_with_progress(&path, move |done, total| {
            let sink = sink.clone();
            async move {
                sink.lock().unwrap().push((done, total));
            }
        })
        .await
        .unwrap();
    assert!(matches!(result, rust_rag_mcp::IndexResult::Skipped));
    assert!(
        skipped_reports.lock().unwrap().is_empty(),
        "A skipped file must not report chunk progress"
    );

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

/// Hybrid results carry per-retriever components (cosine + BM25, each
/// with its 1-based pool rank) that reproduce the fused RRF score
/// exactly — `Σ 1/(RRF_K + rank)` over the sides present — while the
/// other modes omit components because their score already is the raw
/// value. A query matching only one document lexically forces both
/// "in both pools" and "dense-only" component shapes.
#[tokio::test]
async fn test_rag_hybrid_components_explain_rrf_score() {
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

    // The query's tokens exist only in sidecar.rs, so the lexical list
    // holds exactly one document while the dense pool (top-50 over a
    // 3-document corpus) holds everything.
    let hybrid = core
        .search_with_mode("parse_source_index", 5, rust_rag_mcp::SearchMode::Hybrid)
        .await
        .unwrap();
    assert!(!hybrid.is_empty());

    let mut saw_both = false;
    let mut saw_dense_only = false;
    for result in &hybrid {
        let components = result
            .components
            .as_ref()
            .expect("hybrid results carry components");

        // The components must explain the fused score bit-for-bit:
        // the same rank values rrf_fuse consumed.
        let recomputed: f64 = [components.dense.as_ref(), components.lexical.as_ref()]
            .iter()
            .flatten()
            .map(|side| 1.0 / (rust_rag_mcp::bm25::RRF_K + side.rank as f64))
            .sum();
        assert!(
            (recomputed - result.score).abs() < 1e-12,
            "components {components:?} must reproduce score {}",
            result.score
        );

        match (&components.dense, &components.lexical) {
            (Some(dense), Some(lexical)) => {
                assert!(
                    (0.0..=1.0).contains(&dense.score),
                    "dense component is cosine similarity, got {}",
                    dense.score
                );
                assert!(
                    lexical.score > 0.0,
                    "lexical component is a raw BM25 weight, got {}",
                    lexical.score
                );
                assert!(dense.rank >= 1 && lexical.rank >= 1, "ranks are 1-based");
                saw_both = true;
            }
            (Some(dense), None) => {
                assert!(
                    (0.0..=1.0).contains(&dense.score),
                    "dense component is cosine similarity, got {}",
                    dense.score
                );
                assert!(dense.rank >= 1, "ranks are 1-based");
                saw_dense_only = true;
            }
            (None, _) => panic!(
                "dense pool covers the whole corpus; {} cannot be lexical-only",
                result.chunk.source
            ),
        }
    }
    assert!(
        saw_both,
        "sidecar.rs matched both retrievers and must show both components"
    );
    assert!(
        saw_dense_only,
        "documents without query-token overlap must show dense-only components"
    );

    // Non-hybrid modes: score is already interpretable, no components.
    for mode in [
        rust_rag_mcp::SearchMode::Semantic,
        rust_rag_mcp::SearchMode::Lexical,
    ] {
        let results = core
            .search_with_mode("parse_source_index", 5, mode)
            .await
            .unwrap();
        assert!(!results.is_empty());
        assert!(
            results.iter().all(|r| r.components.is_none()),
            "{mode:?} must not carry hybrid components"
        );
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
        &fixtures::model_cache_dir(),
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
    let pdf_path = fixtures_dir().join("sample.pdf");
    assert!(pdf_path.exists(), "committed fixture sample.pdf is missing");
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
    let docx_path = fixtures_dir().join("sample.docx");
    assert!(
        docx_path.exists(),
        "committed fixture sample.docx is missing"
    );
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
    let xlsx_path = fixtures_dir().join("sample.xlsx");
    assert!(
        xlsx_path.exists(),
        "committed fixture sample.xlsx is missing"
    );
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
    let ppt_path = fixtures_dir().join("sample.pptx");
    assert!(
        ppt_path.exists(),
        "committed fixture sample.pptx is missing"
    );
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

#[tokio::test]
#[ignore]
async fn bench_metadata_snapshot() {
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    #[derive(Clone)]
    struct MetadataIndex {
        source_to_ids: Arc<RwLock<HashMap<String, Vec<u32>>>>,
    }

    let idx = MetadataIndex {
        source_to_ids: Arc::new(RwLock::new(HashMap::new())),
    };

    // Setup: 1000 sources with 100 IDs each
    {
        let mut map = idx.source_to_ids.write().await;
        for s in 0..1000 {
            let source = format!("source_{}", s);
            let ids: Vec<u32> = (0..100).collect();
            map.insert(source, ids);
        }
    }

    // Benchmark: 100 snapshots
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let map = idx.source_to_ids.read().await;
        let mut entries: Vec<(String, Vec<u32>)> =
            map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        entries.sort();
        let _ = entries;
    }
    let elapsed = start.elapsed();
    println!(
        "Baseline (100 snapshots of 1000 sources): {:.2} ms",
        elapsed.as_secs_f64() * 1000.0
    );
}

#[tokio::test]
#[ignore]
async fn bench_chunker_performance() {
    use rust_rag_mcp::chunker::Chunker;
    use std::time::Instant;

    let chunker = Chunker::new(512, 64);

    // Generate large text document
    let words: Vec<String> = (0..10000).map(|i| format!("word_{}", i)).collect();
    let large_text = words.join(" ");

    // Benchmark: 100 iterations of chunking
    let start = Instant::now();
    for _ in 0..100 {
        let _ = chunker.chunk_text(&large_text, "test.txt");
    }
    let elapsed = start.elapsed();

    let avg_ms = elapsed.as_secs_f64() / 100.0 * 1000.0;
    println!("Chunker (10k words, 100 iterations): {:?}", elapsed);
    println!("Per-iteration: {:.4} ms", avg_ms);
}

#[tokio::test]
#[ignore]
async fn bench_search_ef_scaling() {
    use rust_rag_mcp::r_vector::{AsyncVectorDb, Config};
    use std::time::Instant;

    let cfg = Config::new(384).with_m(16).with_ef_construction(200);
    let db = AsyncVectorDb::open("test_ef_scale.db", cfg).await.unwrap();

    // Insert 1000 vectors
    for i in 0..1000 {
        let v: Vec<f32> = (0..384)
            .map(|j| ((i * 13 + j * 17) % 1000) as f32 / 1000.0)
            .collect();
        let _ = db.insert(&v, Some(format!("doc_{}", i).as_bytes())).await;
    }

    let query: Vec<f32> = (0..384).map(|_| 0.5).collect();

    // Benchmark different k values with adaptive ef_search
    let k_values = vec![1, 10, 50, 100, 500];
    for k in k_values {
        let start = Instant::now();
        for _ in 0..10 {
            let _ = db.search(&query, k, 200).await; // static ef for comparison
        }
        let elapsed = start.elapsed();
        println!(
            "Search k={:3}: 10x avg {:.3} ms",
            k,
            elapsed.as_secs_f64() / 10.0 * 1000.0
        );
    }

    let _ = std::fs::remove_file("test_ef_scale.db");
    let _ = std::fs::remove_file("test_ef_scale.db.hnsw");
    let _ = std::fs::remove_file("test_ef_scale.db.meta");
    let _ = std::fs::remove_file("test_ef_scale.db.wal");
}
