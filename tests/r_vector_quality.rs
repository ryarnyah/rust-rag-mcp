//! Behavioural safety net for the vector store.
//!
//! These tests assert *semantics* — recall against a brute-force oracle,
//! durability across flush/reopen, content preservation through compaction,
//! concurrent-search correctness, and edge cases — and deliberately avoid
//! pinning the on-disk layout, so storage-layout refactors stay verifiable:
//! if a change breaks graph quality, durability, or concurrency, these fail.

use rust_rag_mcp::r_vector::Result;
use rust_rag_mcp::r_vector::{AsyncVectorDb, Config, VectorDb, VectorDbError, cosine_distance};
use std::collections::HashSet;
use std::fs;
use std::sync::Arc;
use std::time::Duration;

fn cleanup(path: &str) {
    let base = std::path::Path::new(path);
    let _ = fs::remove_file(base);
    let _ = fs::remove_file(format!("{path}.hnsw"));
    let _ = fs::remove_file(format!("{path}.meta"));
    let _ = fs::remove_file(format!("{path}.wal"));
    let _ = fs::remove_file(base.with_extension("lock"));
    let dir = base.parent().unwrap_or(std::path::Path::new("."));
    let prefix = format!(
        "{}.wal.",
        base.file_name().unwrap_or_default().to_string_lossy()
    );
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            if entry.file_name().to_string_lossy().starts_with(&prefix) {
                let _ = fs::remove_file(entry.path());
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  Deterministic helpers
// ---------------------------------------------------------------------------

/// Xorshift RNG mirroring the in-crate one so test data is reproducible.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Float in [-1, 1).
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// Non-zero random vector (all-zero rows would be stored as placeholders
/// that never enter the graph and are never searchable).
fn random_vector(rng: &mut Rng, dim: usize) -> Vec<f32> {
    loop {
        let v: Vec<f32> = (0..dim).map(|_| rng.next_f32()).collect();
        if v.iter().any(|&x| x != 0.0) {
            return v;
        }
    }
}

/// All live, graph-searchable rows as (id, vector) pairs.
fn live_pool(db: &VectorDb) -> Vec<(u32, Vec<f32>)> {
    let mut pool = Vec::new();
    for id in 0..db.len() as u32 {
        if let Ok(Some(v)) = db.get(id)
            && v.iter().any(|&x| x != 0.0)
        {
            pool.push((id, v));
        }
    }
    pool
}

/// Exact top-k by cosine distance — the oracle every recall test compares
/// against. Uses the same `cosine_distance` as the index, so ordering ties
/// resolve identically.
fn brute_force(pool: &[(u32, Vec<f32>)], query: &[f32], k: usize) -> Vec<u32> {
    let mut scored: Vec<(u32, f32)> = pool
        .iter()
        .filter_map(|(id, v)| cosine_distance(query, v).ok().map(|d| (*id, d)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

/// |returned ∩ exact| / |exact|.
fn recall(returned: &[u32], exact: &[u32]) -> f64 {
    if exact.is_empty() {
        return 1.0;
    }
    let exact_set: HashSet<u32> = exact.iter().copied().collect();
    let hits = returned.iter().filter(|id| exact_set.contains(id)).count();
    hits as f64 / exact.len() as f64
}

/// Minimal view over both hit types (`VectorDb::search` returns borrowed
/// hits, `AsyncVectorDb::search` returns owned ones).
trait Hit {
    fn hit_id(&self) -> u32;
    fn hit_score(&self) -> f32;
}

impl Hit for rust_rag_mcp::r_vector::SearchHit<'_> {
    fn hit_id(&self) -> u32 {
        self.id
    }
    fn hit_score(&self) -> f32 {
        self.score
    }
}

impl Hit for rust_rag_mcp::r_vector::SearchHitOwned {
    fn hit_id(&self) -> u32 {
        self.id
    }
    fn hit_score(&self) -> f32 {
        self.score
    }
}

/// Shared structural assertions for search results: bounded unique ids,
/// non-increasing scores, score range, nothing deleted, nothing unsearchable.
fn assert_hit_shape(db: &VectorDb, hits: &[impl Hit], k: usize) {
    assert!(hits.len() <= k, "more than k hits: {} > {k}", hits.len());
    let mut seen = HashSet::new();
    for h in hits {
        let id = h.hit_id();
        let score = h.hit_score();
        assert!(seen.insert(id), "duplicate id {id} in results");
        assert!((0.0..=1.0).contains(&score), "score {score} outside [0,1]");
        assert!(!db.is_deleted(id), "deleted row {id} returned");
        if let Ok(Some(v)) = db.get(id) {
            assert!(v.iter().any(|&x| x != 0.0), "placeholder row {id} returned");
        }
    }
    for w in hits.windows(2) {
        assert!(
            w[0].hit_score() >= w[1].hit_score(),
            "scores not sorted descending: {} < {}",
            w[0].hit_score(),
            w[1].hit_score()
        );
    }
}

// ---------------------------------------------------------------------------
//  Recall against the brute-force oracle
// ---------------------------------------------------------------------------

/// A layout change that damages graph quality (wrong offsets, broken edge
/// writes, lost nodes) must show up here as a recall drop.
#[tokio::test]
async fn test_recall_matches_brute_force() -> Result<()> {
    cleanup("test_recall.db");
    {
        let cfg = Config::new(32).with_capacity(64).with_seed(7);
        let mut db = VectorDb::open("test_recall.db", cfg).await?;

        let mut rng = Rng::new(0xC0FFEE);
        let data: Vec<Vec<f32>> = (0..800).map(|_| random_vector(&mut rng, 32)).collect();
        for v in &data {
            db.insert(v, None)?;
        }
        db.integrity_check()?;

        let pool = live_pool(&db);
        assert_eq!(pool.len(), 800);

        let mut total_recall = 0.0;
        let queries = 15;
        for _ in 0..queries {
            let query = random_vector(&mut rng, 32);
            let hits = db.search(&query, 10, 128)?;
            assert_hit_shape(&db, &hits, 10);
            let returned: Vec<u32> = hits.iter().map(|h| h.id).collect();
            let exact = brute_force(&pool, &query, 10);
            total_recall += recall(&returned, &exact);
        }
        let avg = total_recall / queries as f64;
        assert!(
            avg >= 0.9,
            "recall@10 = {avg:.3} below 0.9 — graph quality regressed"
        );
    }
    cleanup("test_recall.db");
    Ok(())
}

/// Recall must survive a flush + close + reopen cycle, tombstones included:
/// the on-disk graph and row data are read back through the mmap path.
#[tokio::test]
async fn test_recall_survives_flush_and_reopen() -> Result<()> {
    cleanup("test_recall_reopen.db");
    let metas_before: Vec<Vec<u8>>;
    {
        let cfg = Config::new(24).with_capacity(64).with_seed(13);
        let mut db = VectorDb::open("test_recall_reopen.db", cfg).await?;

        let mut rng = Rng::new(99);
        for i in 0..400 {
            let v = random_vector(&mut rng, 24);
            db.insert(&v, Some(format!("meta-{i}").as_bytes()))?;
        }
        for id in (0..400u32).step_by(4) {
            assert!(db.delete(id)?);
        }
        db.integrity_check()?;
        db.flush().await?;
        db.close().await?;

        metas_before = (0..400u32)
            .filter_map(|id| db.get_meta(id).ok().flatten().map(|m| m.to_vec()))
            .collect();
    }
    {
        let cfg = Config::new(24).with_capacity(64).with_seed(13);
        let db = VectorDb::open("test_recall_reopen.db", cfg).await?;

        assert_eq!(db.len(), 400, "all rows must survive reopen");
        assert_eq!(db.deleted_count(), 100, "tombstones must survive reopen");
        db.integrity_check()?;

        // Metadata of live rows must be byte-identical after reopen.
        let metas_after: Vec<Vec<u8>> = (0..400u32)
            .filter_map(|id| db.get_meta(id).ok().flatten().map(|m| m.to_vec()))
            .collect();
        assert_eq!(metas_before, metas_after, "metadata changed across reopen");

        // Recall against the live-only oracle (tombstones excluded).
        let pool = live_pool(&db);
        assert_eq!(pool.len(), 300);

        let mut rng = Rng::new(1000);
        let mut total_recall = 0.0;
        let queries = 12;
        for _ in 0..queries {
            let query = random_vector(&mut rng, 24);
            let hits = db.search(&query, 10, 128)?;
            assert_hit_shape(&db, &hits, 10);
            let returned: Vec<u32> = hits.iter().map(|h| h.id).collect();
            let exact = brute_force(&pool, &query, 10);
            total_recall += recall(&returned, &exact);
        }
        let avg = total_recall / queries as f64;
        assert!(avg >= 0.85, "recall@10 after reopen = {avg:.3} < 0.85");
    }
    cleanup("test_recall_reopen.db");
    Ok(())
}

// ---------------------------------------------------------------------------
//  Growth
// ---------------------------------------------------------------------------

/// Inserting far past the initial capacity exercises every grow()/remap of
/// the vector, graph, and metadata files; search must stay correct and the
/// grown files must reopen cleanly.
#[tokio::test]
async fn test_capacity_growth_preserves_search() -> Result<()> {
    cleanup("test_growth.db");
    {
        let cfg = Config::new(8).with_capacity(16).with_seed(3);
        let mut db = VectorDb::open("test_growth.db", cfg).await?;

        let mut rng = Rng::new(555);
        let data: Vec<Vec<f32>> = (0..1000).map(|_| random_vector(&mut rng, 8)).collect();
        for v in &data {
            db.insert(v, None)?;
        }
        assert_eq!(db.len(), 1000);
        db.integrity_check()?;

        let pool = live_pool(&db);
        let mut total_recall = 0.0;
        let queries = 8;
        for _ in 0..queries {
            let query = random_vector(&mut rng, 8);
            let hits = db.search(&query, 10, 128)?;
            assert_hit_shape(&db, &hits, 10);
            let returned: Vec<u32> = hits.iter().map(|h| h.id).collect();
            total_recall += recall(&returned, &brute_force(&pool, &query, 10));
        }
        assert!(
            total_recall / queries as f64 >= 0.85,
            "recall regressed after growth"
        );

        db.flush().await?;
        db.close().await?;
    }
    {
        let cfg = Config::new(8).with_capacity(16).with_seed(3);
        let db = VectorDb::open("test_growth.db", cfg).await?;
        assert_eq!(db.len(), 1000);
        db.integrity_check()?;
    }
    cleanup("test_growth.db");
    Ok(())
}

// ---------------------------------------------------------------------------
//  Compaction
// ---------------------------------------------------------------------------

/// Compaction renumbers every live row; the surviving content (metadata and
/// search behaviour) must be preserved exactly, and the rebuilt files must
/// pass structural checks both immediately and after a reopen.
#[tokio::test]
async fn test_compaction_preserves_live_content() -> Result<()> {
    cleanup("test_compact_content.db");
    let expected_live: HashSet<String> = (0..200u32)
        .filter(|i| i % 2 == 1)
        .map(|i| format!("src-{}:{i}", i % 8))
        .collect();
    {
        let cfg = Config::new(16).with_capacity(64).with_seed(21);
        let mut db = VectorDb::open("test_compact_content.db", cfg).await?;

        let mut rng = Rng::new(777);
        for i in 0..200u32 {
            let v = random_vector(&mut rng, 16);
            db.insert(&v, Some(format!("src-{}:{i}", i % 8).as_bytes()))?;
        }
        for id in (0..200u32).step_by(2) {
            assert!(db.delete(id)?);
        }
        assert!(db.should_compact());
        db.flush().await?;

        db.compact().await?;
        assert_eq!(db.live_len(), 100, "compaction must keep the 100 live rows");
        assert_eq!(db.deleted_count(), 0);
        db.integrity_check()?;

        // Content set equality (ids are renumbered, content is not).
        let live_metas: HashSet<String> = (0..db.len() as u32)
            .filter_map(|id| db.get_meta(id).ok().flatten())
            .map(|m| String::from_utf8_lossy(m).into_owned())
            .collect();
        assert_eq!(
            live_metas, expected_live,
            "compaction lost or altered content"
        );

        // Search over the rebuilt graph still tracks the brute-force oracle.
        let pool = live_pool(&db);
        assert_eq!(pool.len(), 100);
        let mut total_recall = 0.0;
        let queries = 8;
        for _ in 0..queries {
            let query = random_vector(&mut rng, 16);
            let hits = db.search(&query, 10, 128)?;
            assert_hit_shape(&db, &hits, 10);
            let returned: Vec<u32> = hits.iter().map(|h| h.id).collect();
            total_recall += recall(&returned, &brute_force(&pool, &query, 10));
        }
        assert!(total_recall / queries as f64 >= 0.9);

        db.flush().await?;
        db.close().await?;
    }
    {
        let cfg = Config::new(16).with_capacity(64).with_seed(21);
        let db = VectorDb::open("test_compact_content.db", cfg).await?;
        assert_eq!(db.live_len(), 100);
        db.integrity_check()?;
        let live_metas: HashSet<String> = (0..db.len() as u32)
            .filter_map(|id| db.get_meta(id).ok().flatten())
            .map(|m| String::from_utf8_lossy(m).into_owned())
            .collect();
        assert_eq!(live_metas, expected_live, "compacted content must persist");
    }
    cleanup("test_compact_content.db");
    Ok(())
}

// ---------------------------------------------------------------------------
//  Concurrency
// ---------------------------------------------------------------------------

/// Concurrent readers share the index; every search must return the same
/// results a single-threaded search would. (This is the regression guard for
/// shared scratch-buffer state during traversal.)
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_concurrent_searches_agree_with_brute_force() -> Result<()> {
    cleanup("test_conc_search.db");
    {
        let cfg = Config::new(16).with_capacity(64).with_seed(11);
        let db = AsyncVectorDb::open("test_conc_search.db", cfg).await?;

        let mut rng = Rng::new(2024);
        let data: Arc<Vec<Vec<f32>>> =
            Arc::new((0..400).map(|_| random_vector(&mut rng, 16)).collect());
        for v in data.iter() {
            db.insert(v, None).await?;
        }
        db.integrity_check().await?;

        let pool: Arc<Vec<(u32, Vec<f32>)>> = Arc::new(
            data.iter()
                .enumerate()
                .map(|(i, v)| (i as u32, v.clone()))
                .collect(),
        );

        let mut handles = Vec::new();
        for task in 0..8u64 {
            let db = db.clone();
            let pool = pool.clone();
            handles.push(tokio::spawn(async move {
                let mut rng = Rng::new(1000 + task);
                for _ in 0..25 {
                    let qidx = (rng.next_u64() as usize) % pool.len();
                    let query = &pool[qidx].1;
                    let hits = db.search(query, 5, 128).await?;
                    assert!(!hits.is_empty(), "stored vector must be findable");

                    let ids: Vec<u32> = hits.iter().map(|h| h.id).collect();
                    assert_eq!(
                        ids[0] as usize, qidx,
                        "a stored vector is its own nearest neighbour"
                    );
                    assert!(hits[0].score > 0.999, "self-match score {}", hits[0].score);
                    assert_eq!(
                        ids.iter().copied().collect::<HashSet<_>>().len(),
                        ids.len(),
                        "duplicate ids in results: {ids:?}"
                    );
                    for w in hits.windows(2) {
                        assert!(w[0].score >= w[1].score, "results not sorted");
                    }
                    let exact = brute_force(&pool, query, 5);
                    let r = recall(&ids, &exact);
                    assert!(r >= 0.8, "concurrent search recall {r:.3} < 0.8");
                }
                Ok::<(), VectorDbError>(())
            }));
        }
        for h in handles {
            h.await.expect("search task panicked")?;
        }
        db.integrity_check().await?;
    }
    cleanup("test_conc_search.db");
    Ok(())
}

/// Mixed readers and writers on separate tasks: must not deadlock, and the
/// database must be structurally sound afterwards.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_concurrent_insert_and_search() -> Result<()> {
    cleanup("test_conc_mixed.db");
    {
        let cfg = Config::new(12).with_capacity(64).with_seed(17);
        let db = AsyncVectorDb::open("test_conc_mixed.db", cfg).await?;

        let mut rng = Rng::new(31337);
        let seed_data: Arc<Vec<Vec<f32>>> =
            Arc::new((0..200).map(|_| random_vector(&mut rng, 12)).collect());
        for v in seed_data.iter() {
            db.insert(v, None).await?;
        }

        let mut handles = Vec::new();
        for task in 0..4u64 {
            let reader_db = db.clone();
            let pool = seed_data.clone();
            handles.push(tokio::spawn(async move {
                let mut rng = Rng::new(500 + task);
                for _ in 0..20 {
                    let qidx = (rng.next_u64() as usize) % pool.len();
                    let hits = reader_db.search(&pool[qidx], 3, 64).await?;
                    assert!(!hits.is_empty());
                    assert_eq!(hits[0].id as usize, qidx);
                }
                Ok::<(), VectorDbError>(())
            }));
            let writer_db = db.clone();
            handles.push(tokio::spawn(async move {
                let mut rng = Rng::new(700 + task);
                for _ in 0..20 {
                    let v = random_vector(&mut rng, 12);
                    writer_db.insert(&v, None).await?;
                }
                Ok::<(), VectorDbError>(())
            }));
        }

        let joined = async {
            for h in handles {
                h.await.expect("task panicked")?;
            }
            Ok::<(), VectorDbError>(())
        };
        tokio::time::timeout(Duration::from_secs(60), joined)
            .await
            .expect("concurrent insert/search deadlocked")?;

        assert_eq!(db.len().await, 280, "200 seed + 80 concurrent inserts");
        db.integrity_check().await?;
    }
    cleanup("test_conc_mixed.db");
    Ok(())
}

// ---------------------------------------------------------------------------
//  Update
// ---------------------------------------------------------------------------

/// `update()` rewrites a row and rebuilds its edges; the new position must be
/// findable and the graph must stay sound.
#[tokio::test]
async fn test_update_moves_vector_in_search() -> Result<()> {
    cleanup("test_update_move.db");
    {
        let cfg = Config::new(12).with_capacity(64).with_seed(9);
        let mut db = VectorDb::open("test_update_move.db", cfg).await?;

        let mut rng = Rng::new(4242);
        for _ in 0..150 {
            let v = random_vector(&mut rng, 12);
            db.insert(&v, None)?;
        }

        let new_v = random_vector(&mut rng, 12);
        db.update(7, &new_v, None)?;
        db.integrity_check()?;

        let hits = db.search(&new_v, 1, 128)?;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 7, "updated vector must sit at its new position");
        assert!(hits[0].score > 0.999, "self-match score {}", hits[0].score);
    }
    cleanup("test_update_move.db");
    Ok(())
}

/// Many `update()`s rebuild edges in place (clearing and relinking each
/// node). After a batch of rebuilds the graph must still track the oracle:
/// every moved node findable at its new position, overall recall intact.
#[tokio::test]
async fn test_batch_updates_preserve_recall() -> Result<()> {
    cleanup("test_batch_update.db");
    {
        let cfg = Config::new(16).with_capacity(64).with_seed(41);
        let mut db = VectorDb::open("test_batch_update.db", cfg).await?;

        let mut rng = Rng::new(1717);
        for _ in 0..200 {
            let v = random_vector(&mut rng, 16);
            db.insert(&v, None)?;
        }

        // Move every 4th node to a fresh random position.
        let mut moved = Vec::new();
        for id in (0..200u32).step_by(4) {
            let v = random_vector(&mut rng, 16);
            db.update(id, &v, None)?;
            moved.push((id, v));
        }
        db.integrity_check()?;

        // Each moved node must be reachable at its new position.
        for (id, v) in &moved {
            let hits = db.search(v, 1, 128)?;
            assert_eq!(hits.len(), 1, "no hit for moved node {id}");
            assert_eq!(
                hits[0].id, *id,
                "moved node {id} not found at its new position"
            );
            assert!(hits[0].score > 0.999, "self-match score {}", hits[0].score);
        }

        // Overall recall against the oracle must survive the rebuilds.
        let pool = live_pool(&db);
        let mut total_recall = 0.0;
        let queries = 10;
        for _ in 0..queries {
            let query = random_vector(&mut rng, 16);
            let hits = db.search(&query, 10, 128)?;
            assert_hit_shape(&db, &hits, 10);
            let returned: Vec<u32> = hits.iter().map(|h| h.id).collect();
            total_recall += recall(&returned, &brute_force(&pool, &query, 10));
        }
        let avg = total_recall / queries as f64;
        assert!(avg >= 0.9, "recall@10 after batch updates = {avg:.3} < 0.9");
    }
    cleanup("test_batch_update.db");
    Ok(())
}

// ---------------------------------------------------------------------------
//  Crash/WAL recovery + graph soundness
// ---------------------------------------------------------------------------

/// Records logged but never flushed must replay into data files that satisfy
/// every structural invariant, and search must still track the oracle.
#[tokio::test]
async fn test_wal_replay_keeps_graph_sound() -> Result<()> {
    cleanup("test_wal_replay.db");
    {
        let cfg = Config::new(16).with_capacity(64).with_seed(31);
        let mut db = VectorDb::open("test_wal_replay.db", cfg).await?;
        let mut rng = Rng::new(606);
        for i in 0..200u32 {
            let v = random_vector(&mut rng, 16);
            db.insert(&v, Some(format!("w{i}").as_bytes()))?;
        }
        // No flush: dropping closes the WAL writer (fsyncs) but leaves the
        // data files un-msynced, so reopen must replay the log.
        drop(db);
    }
    {
        let cfg = Config::new(16).with_capacity(64).with_seed(31);
        let db = VectorDb::open("test_wal_replay.db", cfg).await?;
        assert_eq!(db.len(), 200, "WAL replay must restore every insert");
        db.integrity_check()?;

        let pool = live_pool(&db);
        assert_eq!(pool.len(), 200);
        let mut rng = Rng::new(607);
        let mut total_recall = 0.0;
        let queries = 8;
        for _ in 0..queries {
            let query = random_vector(&mut rng, 16);
            let hits = db.search(&query, 10, 128)?;
            assert_hit_shape(&db, &hits, 10);
            let returned: Vec<u32> = hits.iter().map(|h| h.id).collect();
            total_recall += recall(&returned, &brute_force(&pool, &query, 10));
        }
        assert!(total_recall / queries as f64 >= 0.9);
    }
    cleanup("test_wal_replay.db");
    Ok(())
}

// ---------------------------------------------------------------------------
//  Edge cases
// ---------------------------------------------------------------------------

/// Degenerate k/ef sizes, empty databases, and fully-tombstoned databases
/// must return well-formed (possibly empty) results without panicking.
#[tokio::test]
async fn test_search_edge_cases() -> Result<()> {
    cleanup("test_edge_cases.db");
    {
        let cfg = Config::new(4).with_capacity(16).with_seed(5);
        let mut db = VectorDb::open("test_edge_cases.db", cfg).await?;
        let query = [1.0, 0.0, 0.0, 0.0];

        // Empty database.
        assert!(db.search(&query, 5, 32)?.is_empty());
        assert!(db.search(&query, 0, 32)?.is_empty());

        let mut rng = Rng::new(808);
        for _ in 0..5 {
            let v = random_vector(&mut rng, 4);
            db.insert(&v, None)?;
        }

        // k = 0 → no hits; k > len → at most len hits.
        assert!(db.search(&query, 0, 32)?.is_empty());
        assert_eq!(db.search(&query, 100, 32)?.len(), 5);

        // ef far below k: the store inflates it internally, still k hits.
        assert_eq!(db.search(&query, 5, 1)?.len(), 5);

        // Fully tombstoned database → empty results, no error, sound files.
        for id in 0..5u32 {
            assert!(db.delete(id)?);
        }
        assert!(db.search(&query, 5, 32)?.is_empty());
        assert_eq!(db.live_len(), 0);
        db.integrity_check()?;
    }
    cleanup("test_edge_cases.db");
    Ok(())
}
