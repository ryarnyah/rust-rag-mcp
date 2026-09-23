use rust_rag_mcp::r_vector::Result;
use rust_rag_mcp::r_vector::{AsyncVectorDb, Config, VectorDb, VectorDbError, cosine_distance};
use std::fs;

fn cleanup(path: &str) {
    let _ = fs::remove_file(path);
    let _ = fs::remove_file(format!("{}.hnsw", path));
    let _ = fs::remove_file(format!("{}.meta", path));
    let _ = fs::remove_file(format!("{}.wal", path));
    let dir = std::path::Path::new(path)
        .parent()
        .unwrap_or(std::path::Path::new("."));
    let prefix = format!(
        "{}.wal.",
        std::path::Path::new(path)
            .file_name()
            .unwrap()
            .to_string_lossy()
    );
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            if name.to_string_lossy().starts_with(&prefix) {
                let _ = fs::remove_file(entry.path());
            }
        }
    }
}

#[test]
fn test_config_builder() {
    let cfg = Config::new(16)
        .with_m(4)
        .with_max_level(12)
        .with_ef_construction(128)
        .with_capacity(512)
        .with_seed(42);

    assert_eq!(cfg.dim, 16);
    assert_eq!(cfg.m, 4);
    assert_eq!(cfg.m0, 8);
    assert_eq!(cfg.max_level, 12);
    assert_eq!(cfg.ef_construction, 128);
    assert_eq!(cfg.initial_capacity, 512);
    assert_eq!(cfg.seed, 42);
}

#[test]
fn test_cosine_distance() {
    let a = vec![1.0, 0.0];
    let b = vec![1.0, 0.0];
    let dist = cosine_distance(&a, &b).unwrap();
    assert!(dist < 0.001);

    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let dist = cosine_distance(&a, &b).unwrap();
    assert!((dist - 1.0).abs() < 0.001);
}

#[test]
fn test_cosine_distance_nan() {
    let a = vec![f32::NAN, 1.0];
    let b = vec![1.0, 1.0];
    assert!(cosine_distance(&a, &b).is_err());
}

#[test]
fn test_cosine_distance_zero_vector_is_error() {
    // Undefined input → error, never a fabricated number.
    let a = vec![0.0, 0.0];
    let b = vec![1.0, 0.0];
    let err = cosine_distance(&a, &b).unwrap_err();
    assert!(matches!(err, VectorDbError::ZeroVector));
}

#[tokio::test]
async fn test_zero_query_returns_error_not_scores() -> Result<()> {
    cleanup("test_zero_query.db");
    {
        let cfg = Config::new(2);
        let mut db = VectorDb::open("test_zero_query.db", cfg).await?;
        db.insert(&[1.0, 0.0], None)?;

        let err = db.search(&[0.0, 0.0], 5, 32).unwrap_err();
        assert!(matches!(err, VectorDbError::ZeroVector));
    }
    cleanup("test_zero_query.db");
    Ok(())
}

#[tokio::test]
async fn test_placeholder_rows_stored_but_never_searchable() -> Result<()> {
    cleanup("test_placeholder.db");
    {
        let cfg = Config::new(2).with_capacity(16);
        let mut db = VectorDb::open("test_placeholder.db", cfg).await?;

        // Placeholder (all-zero) row, like rag.rs stores document metadata,
        // inserted *between* real vectors so it lands in the middle of id space.
        db.insert(&[1.0, 0.0], None)?;
        let ph = db.insert(&[0.0, 0.0], Some(b"doc-metadata"))?;
        db.insert(&[0.0, 1.0], None)?;
        db.insert(&[1.0, 0.0], None)?;

        // Stored and readable like any other row.
        assert_eq!(db.get(ph)?, Some(vec![0.0, 0.0]));
        assert_eq!(db.get_meta(ph)?, Some(b"doc-metadata".as_slice()));

        // Searching must neither fail nor ever surface the placeholder row.
        let results = db.search(&[1.0, 0.0], 10, 64)?;
        assert!(!results.is_empty());
        for hit in &results {
            assert_ne!(hit.id, ph, "placeholder row must not be searchable");
            assert!(hit.score.is_finite() && (0.0..=1.0).contains(&hit.score));
        }

        // Deletion still works on placeholder rows.
        assert!(db.delete(ph)?);
        assert!(db.is_deleted(ph));
    }
    cleanup("test_placeholder.db");
    Ok(())
}

#[tokio::test]
async fn test_update_entry_node_keeps_graph_connected() -> Result<()> {
    cleanup("test_update_entry.db");
    {
        let cfg = Config::new(2).with_capacity(16);
        let mut db = VectorDb::open("test_update_entry.db", cfg).await?;

        db.insert(&[1.0, 0.0], None)?;
        db.insert(&[0.0, 1.0], None)?;
        db.insert(&[1.0, 1.0], None)?;

        // Update the entry point (id 0): its edges are rebuilt from scratch
        // and the graph must stay reachable afterwards.
        db.update(0, &[0.7, 0.7], None)?;

        let results = db.search(&[1.0, 0.0], 3, 64)?;
        let seen: Vec<u32> = results.iter().map(|h| h.id).collect();
        assert_eq!(
            results.len(),
            3,
            "graph must stay connected, got ids {seen:?}"
        );
    }
    cleanup("test_update_entry.db");
    Ok(())
}

#[tokio::test]
async fn test_metadata_simple() -> Result<()> {
    cleanup("test_meta_simple.db");
    {
        let cfg = Config::new(3).with_capacity(16);
        let mut db = VectorDb::open("test_meta_simple.db", cfg).await?;

        let id = db.insert(&[1.0, 2.0, 3.0], Some(b"mydata"))?;

        let meta = db.get_meta(id)?;
        assert_eq!(
            meta.map(|m| m.to_vec()),
            Some(b"mydata".to_vec()),
            "Metadata should match"
        );

        db.flush().await?;

        let meta = db.get_meta(id)?;
        assert_eq!(
            meta.map(|m| m.to_vec()),
            Some(b"mydata".to_vec()),
            "Metadata should persist after flush"
        );
    }
    cleanup("test_meta_simple.db");
    Ok(())
}

#[tokio::test]
async fn test_vectordb_insert_single() -> Result<()> {
    cleanup("test_vec.db");
    {
        let cfg = Config::new(3).with_capacity(16);
        let mut db = VectorDb::open("test_vec.db", cfg).await?;

        let v = vec![1.0, 2.0, 3.0];
        let id = db.insert(&v, Some(b"test"))?;

        assert_eq!(id, 0);
        assert_eq!(db.len(), 1);
        assert_eq!(db.live_len(), 1);
        assert_eq!(db.get(id)?, Some(vec![1.0, 2.0, 3.0]));
    }
    cleanup("test_vec.db");
    Ok(())
}

#[tokio::test]
async fn test_vectordb_dimension_mismatch() -> Result<()> {
    cleanup("test_dim.db");
    {
        let cfg = Config::new(3);
        let mut db = VectorDb::open("test_dim.db", cfg).await?;

        let v = vec![1.0, 2.0];
        let result = db.insert(&v, None);
        assert!(matches!(
            result,
            Err(VectorDbError::DimensionMismatch { .. })
        ));
    }
    cleanup("test_dim.db");
    Ok(())
}

#[tokio::test]
async fn test_vectordb_delete() -> Result<()> {
    cleanup("test_del.db");
    {
        let cfg = Config::new(2);
        let mut db = VectorDb::open("test_del.db", cfg).await?;

        let id = db.insert(&[1.0, 2.0], None)?;
        assert!(!db.is_deleted(id));
        assert_eq!(db.live_len(), 1);

        let deleted = db.delete(id)?;
        assert!(deleted);
        assert!(db.is_deleted(id));
        assert_eq!(db.get(id)?, None);
        assert_eq!(db.live_len(), 0);
    }
    cleanup("test_del.db");
    Ok(())
}

#[tokio::test]
async fn test_vectordb_search_basic() -> Result<()> {
    cleanup("test_search.db");
    {
        let cfg = Config::new(2).with_capacity(16);
        let mut db = VectorDb::open("test_search.db", cfg).await?;

        db.insert(&[1.0, 0.0], None)?;
        db.insert(&[0.0, 1.0], None)?;
        db.insert(&[1.0, 1.0], None)?;

        let query = vec![1.0, 0.0];
        let results = db.search(&query, 2, 32)?;

        assert!(results.len() <= 2);
        assert!(results.iter().all(|h| !db.is_deleted(h.id)));
    }
    cleanup("test_search.db");
    Ok(())
}

#[tokio::test]
async fn test_async_insert_concurrent() -> Result<()> {
    cleanup("test_async.db");
    {
        let cfg = Config::new(4).with_capacity(32);
        let db = AsyncVectorDb::open("test_async.db", cfg).await?;

        let mut handles = vec![];
        for i in 0..10 {
            let db = db.clone();
            let handle = tokio::spawn(async move {
                let v = vec![(i as f32) * 0.1; 4];
                db.insert(&v, None).await
            });
            handles.push(handle);
        }

        for handle in handles {
            let _ = handle.await;
        }

        let len = db.len().await;
        assert_eq!(len, 10);
    }
    cleanup("test_async.db");
    Ok(())
}

#[tokio::test]
async fn test_async_search() -> Result<()> {
    cleanup("test_async_search.db");
    {
        let cfg = Config::new(2);
        let db = AsyncVectorDb::open("test_async_search.db", cfg).await?;

        db.insert(&[1.0, 0.0], None).await?;
        db.insert(&[0.0, 1.0], None).await?;

        let results = db.search(&[1.0, 0.0], 2, 32).await?;
        assert!(!results.is_empty());
    }
    cleanup("test_async_search.db");
    Ok(())
}

#[tokio::test]
async fn test_wal_crash_insert_before_flush() -> Result<()> {
    cleanup("test_wal_crash.db");

    {
        let cfg = Config::new(4).with_capacity(32);
        let mut db = VectorDb::open("test_wal_crash.db", cfg).await?;

        db.insert(&[0.1, 0.2, 0.3, 0.4], Some(b"vec1"))?;
        db.insert(&[0.5, 0.6, 0.7, 0.8], Some(b"vec2"))?;
        db.insert(&[0.9, 0.1, 0.2, 0.3], Some(b"vec3"))?;

        db.flush().await?;
    }

    {
        let cfg = Config::new(4).with_capacity(32);
        let db = VectorDb::open("test_wal_crash.db", cfg).await?;

        assert_eq!(db.len(), 3, "Should have 3 vectors");
        assert_eq!(db.live_len(), 3);

        assert_eq!(db.get(0)?, Some(vec![0.1, 0.2, 0.3, 0.4]));
        assert_eq!(db.get(1)?, Some(vec![0.5, 0.6, 0.7, 0.8]));
        assert_eq!(db.get(2)?, Some(vec![0.9, 0.1, 0.2, 0.3]));

        assert_eq!(db.get_meta(0)?.map(|m| m.to_vec()), Some(b"vec1".to_vec()));
        assert_eq!(db.get_meta(1)?.map(|m| m.to_vec()), Some(b"vec2".to_vec()));
        assert_eq!(db.get_meta(2)?.map(|m| m.to_vec()), Some(b"vec3".to_vec()));
    }

    cleanup("test_wal_crash.db");
    Ok(())
}

#[tokio::test]
async fn test_wal_crash_delete_before_flush() -> Result<()> {
    cleanup("test_wal_delete_crash.db");

    {
        let cfg = Config::new(3).with_capacity(32);
        let mut db = VectorDb::open("test_wal_delete_crash.db", cfg).await?;

        db.insert(&[1.0, 0.0, 0.0], None)?;
        db.insert(&[0.0, 1.0, 0.0], None)?;
        db.insert(&[0.0, 0.0, 1.0], None)?;
        db.flush().await?;
    }

    {
        let cfg = Config::new(3).with_capacity(32);
        let mut db = VectorDb::open("test_wal_delete_crash.db", cfg).await?;

        assert_eq!(db.live_len(), 3);
        db.delete(1)?;
    }

    {
        let cfg = Config::new(3).with_capacity(32);
        let db = VectorDb::open("test_wal_delete_crash.db", cfg).await?;

        assert_eq!(db.len(), 3, "All vectors still tracked");
        assert_eq!(db.live_len(), 2, "One vector deleted");
        assert!(!db.is_deleted(0));
        assert!(db.is_deleted(1), "Deletion should be recovered");
        assert!(!db.is_deleted(2));
    }

    cleanup("test_wal_delete_crash.db");
    Ok(())
}

#[tokio::test]
async fn test_wal_crash_update_before_flush() -> Result<()> {
    cleanup("test_wal_update_crash.db");

    {
        let cfg = Config::new(2).with_capacity(16);
        let mut db = VectorDb::open("test_wal_update_crash.db", cfg).await?;

        db.insert(&[1.0, 0.0], Some(b"original"))?;
        db.flush().await?;
    }

    {
        let cfg = Config::new(2).with_capacity(16);
        let mut db = VectorDb::open("test_wal_update_crash.db", cfg).await?;

        db.update(0, &[2.0, 3.0], Some(b"updated"))?;
        db.flush().await?;
    }

    {
        let cfg = Config::new(2).with_capacity(16);
        let db = VectorDb::open("test_wal_update_crash.db", cfg).await?;

        assert_eq!(
            db.get(0)?,
            Some(vec![2.0, 3.0]),
            "Update should be persisted"
        );
        assert_eq!(
            db.get_meta(0)?.map(|m| m.to_vec()),
            Some(b"updated".to_vec())
        );
    }

    cleanup("test_wal_update_crash.db");
    Ok(())
}

#[tokio::test]
async fn test_wal_mixed_operations_crash() -> Result<()> {
    cleanup("test_wal_mixed.db");

    {
        let cfg = Config::new(3).with_capacity(32);
        let mut db = VectorDb::open("test_wal_mixed.db", cfg).await?;

        for i in 0..5 {
            let v = vec![(i as f32) * 0.1, 0.5, 0.9];
            db.insert(&v, Some(format!("vec{}", i).as_bytes()))?;
        }
        db.flush().await?;
        db.close().await?;
    }

    {
        let cfg = Config::new(3).with_capacity(32);
        let mut db = VectorDb::open("test_wal_mixed.db", cfg).await?;

        db.insert(&[0.2, 0.3, 0.4], Some(b"vec5"))?;
        db.update(1, &[1.0, 2.0, 3.0], Some(b"updated1"))?;
        db.delete(3)?;

        db.flush().await?;
        db.close().await?;
    }

    {
        let cfg = Config::new(3).with_capacity(32);
        let db = VectorDb::open("test_wal_mixed.db", cfg).await?;

        assert_eq!(db.len(), 6, "Should have 6 vectors total");
        assert_eq!(db.live_len(), 5, "One should be deleted");

        assert_eq!(db.get(5)?, Some(vec![0.2, 0.3, 0.4]));
        assert_eq!(db.get(1)?, Some(vec![1.0, 2.0, 3.0]));
        assert_eq!(
            db.get_meta(1)?.map(|m| m.to_vec()),
            Some(b"updated1".to_vec())
        );
        assert!(db.is_deleted(3));
        assert_eq!(db.get(3)?, None);

        assert!(!db.is_deleted(0));
        assert!(!db.is_deleted(2));
        assert!(!db.is_deleted(4));
    }

    cleanup("test_wal_mixed.db");
    Ok(())
}

#[tokio::test]
async fn test_wal_rotation_creates_segments() -> Result<()> {
    cleanup("test_wal_rotation.db");

    let cfg = Config::new(4)
        .with_capacity(32)
        .with_max_wal_segment_size(256)
        .with_max_total_wal_size(0);

    {
        let mut db = VectorDb::open("test_wal_rotation.db", cfg).await?;

        for i in 0..20u32 {
            let v = vec![i as f32, 0.1, 0.2, 0.3];
            db.insert(&v, Some(&i.to_le_bytes()))?;
        }

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let mut segment_count = 0;
        if let Ok(entries) = fs::read_dir(".") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str.starts_with("test_wal_rotation.db.wal.") {
                    segment_count += 1;
                }
            }
        }
        assert!(
            segment_count >= 1,
            "Expected at least 1 WAL segment, found {}",
            segment_count
        );
    }

    {
        let cfg = Config::new(4)
            .with_capacity(32)
            .with_max_wal_segment_size(256)
            .with_max_total_wal_size(0);
        let db = VectorDb::open("test_wal_rotation.db", cfg).await?;
        assert_eq!(
            db.len(),
            20,
            "Should recover all 20 vectors from multi-segment WAL"
        );
    }

    cleanup("test_wal_rotation.db");
    Ok(())
}

#[tokio::test]
async fn test_wal_total_size_cleanup() -> Result<()> {
    cleanup("test_wal_cleanup.db");

    let cfg = Config::new(4)
        .with_capacity(32)
        .with_max_wal_segment_size(200)
        .with_max_total_wal_size(600);

    {
        let mut db = VectorDb::open("test_wal_cleanup.db", cfg).await?;

        for i in 0..10 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            db.insert(&v, None)?;
        }
        db.flush().await?;

        for i in 10..30 {
            let v = vec![i as f32, 0.1, 0.1, 0.1];
            db.insert(&v, None)?;
        }
        db.flush().await?;

        for i in 30..50 {
            let v = vec![i as f32, 0.2, 0.2, 0.2];
            db.insert(&v, None)?;
        }
        db.flush().await?;
    }

    {
        let cfg = Config::new(4)
            .with_capacity(32)
            .with_max_wal_segment_size(200)
            .with_max_total_wal_size(600);
        let db = VectorDb::open("test_wal_cleanup.db", cfg).await?;
        assert_eq!(db.len(), 50, "Should recover all 50 vectors after cleanup");
    }

    cleanup("test_wal_cleanup.db");
    Ok(())
}

#[tokio::test]
async fn test_wal_multi_segment_recovery() -> Result<()> {
    cleanup("test_wal_multi.db");

    let cfg = Config::new(4)
        .with_capacity(64)
        .with_max_wal_segment_size(300)
        .with_max_total_wal_size(0);

    {
        let mut db = VectorDb::open("test_wal_multi.db", cfg).await?;

        for i in 0..30u32 {
            let v = vec![i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32];
            db.insert(&v, Some(&i.to_le_bytes()))?;
        }
    }

    {
        let cfg = Config::new(4)
            .with_capacity(64)
            .with_max_wal_segment_size(300)
            .with_max_total_wal_size(0);
        let db = VectorDb::open("test_wal_multi.db", cfg).await?;
        assert_eq!(
            db.len(),
            30,
            "Should recover all 30 vectors from multiple WAL segments"
        );
    }

    cleanup("test_wal_multi.db");
    Ok(())
}

#[tokio::test]
async fn test_wal_no_cleanup_uncheckpointed_segments() -> Result<()> {
    cleanup("test_wal_no_cleanup.db");

    let cfg = Config::new(4)
        .with_capacity(32)
        .with_max_wal_segment_size(200)
        .with_max_total_wal_size(400);

    {
        let mut db = VectorDb::open("test_wal_no_cleanup.db", cfg).await?;

        for i in 0..15 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            db.insert(&v, None)?;
        }
    }

    {
        let cfg = Config::new(4)
            .with_capacity(32)
            .with_max_wal_segment_size(200)
            .with_max_total_wal_size(400);
        let db = VectorDb::open("test_wal_no_cleanup.db", cfg).await?;
        assert_eq!(
            db.len(),
            15,
            "Uncheckpointed segments should not be deleted"
        );
    }

    cleanup("test_wal_no_cleanup.db");
    Ok(())
}

#[tokio::test]
async fn test_wal_clear_after_checkpoint() -> Result<()> {
    cleanup("test_wal_clear.db");

    {
        let cfg = Config::new(2).with_capacity(16);
        let mut db = VectorDb::open("test_wal_clear.db", cfg).await?;

        for batch in 0..3 {
            for i in 0..2 {
                let v = vec![(batch as f32) + (i as f32) * 0.1, 0.5];
                db.insert(&v, None)?;
            }
            db.flush().await?;
        }

        assert_eq!(db.len(), 6);
    }

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    {
        let cfg = Config::new(2).with_capacity(16);
        let db = VectorDb::open("test_wal_clear.db", cfg).await?;

        assert_eq!(db.len(), 6);
        assert_eq!(db.live_len(), 6);
    }

    cleanup("test_wal_clear.db");
    Ok(())
}

#[tokio::test]
async fn test_wal_metrics_tracking() -> Result<()> {
    cleanup("test_wal_metrics.db");

    let cfg = Config::new(4).with_capacity(32);
    {
        let mut db = VectorDb::open("test_wal_metrics.db", cfg).await?;

        for i in 0..5 {
            let v = vec![i as f32, 0.0, 0.0, 0.0];
            db.insert(&v, None)?;
        }

        db.flush().await?;

        assert_eq!(db.len(), 5, "Should have 5 vectors");
    }

    {
        let cfg = Config::new(4).with_capacity(32);
        let db = VectorDb::open("test_wal_metrics.db", cfg).await?;
        assert_eq!(db.len(), 5, "Should recover all 5 vectors");
    }

    cleanup("test_wal_metrics.db");
    Ok(())
}

#[tokio::test]
async fn test_deleted_count_stability_across_recovery() -> Result<()> {
    cleanup("test_deleted_stability.db");

    {
        let cfg = Config::new(2).with_capacity(32);
        let mut db = VectorDb::open("test_deleted_stability.db", cfg).await?;

        for i in 0..5 {
            let v = vec![i as f32, 0.5];
            db.insert(&v, None)?;
        }
        db.flush().await?;

        db.delete(1)?;
        db.delete(3)?;
        db.flush().await?;

        let deleted = db.deleted_count();
        let live = db.live_len();
        assert_eq!(deleted, 2, "Should have 2 deleted vectors");
        assert_eq!(live, 3, "Should have 3 live vectors");
        db.close().await?;
    }

    {
        let cfg = Config::new(2).with_capacity(32);
        let mut db = VectorDb::open("test_deleted_stability.db", cfg).await?;

        let deleted = db.deleted_count();
        let live = db.live_len();
        assert_eq!(
            deleted, 2,
            "After recovery: should have 2 deleted vectors, got {}",
            deleted
        );
        assert_eq!(
            live, 3,
            "After recovery: should have 3 live vectors, got {}",
            live
        );
        db.close().await?;
    }

    {
        let cfg = Config::new(2).with_capacity(32);
        let mut db = VectorDb::open("test_deleted_stability.db", cfg).await?;

        let deleted = db.deleted_count();
        let live = db.live_len();
        assert_eq!(
            deleted, 2,
            "After second recovery: should have 2 deleted vectors, got {}",
            deleted
        );
        assert_eq!(
            live, 3,
            "After second recovery: should have 3 live vectors, got {}",
            live
        );
        db.close().await?;
    }

    cleanup("test_deleted_stability.db");
    Ok(())
}

#[tokio::test]
async fn test_flush_order_data_before_checkpoint() -> Result<()> {
    cleanup("test_flush_order.db");

    {
        let cfg = Config::new(2).with_capacity(32);
        let mut db = VectorDb::open("test_flush_order.db", cfg).await?;

        db.insert(&[1.0, 0.0], Some(b"first"))?;
        db.insert(&[0.0, 1.0], Some(b"second"))?;
        db.flush().await?;
    }

    {
        let cfg = Config::new(2).with_capacity(32);
        let mut db = VectorDb::open("test_flush_order.db", cfg).await?;

        assert_eq!(db.len(), 2, "Should have 2 vectors from first stage");
        db.insert(&[0.5, 0.5], Some(b"third"))?;
    }

    {
        let cfg = Config::new(2).with_capacity(32);
        let db = VectorDb::open("test_flush_order.db", cfg).await?;

        assert_eq!(db.len(), 3, "After recovery: should have 3 vectors");
        assert_eq!(db.get(2)?, Some(vec![0.5, 0.5]));
        assert_eq!(db.get_meta(2)?.map(|m| m.to_vec()), Some(b"third".to_vec()));
    }

    cleanup("test_flush_order.db");
    Ok(())
}

#[tokio::test]
async fn test_score_range_clamping() -> Result<()> {
    cleanup("test_score_range.db");
    {
        let cfg = Config::new(2);
        let mut db = VectorDb::open("test_score_range.db", cfg).await?;

        db.insert(&[1.0, 0.0], None)?;
        db.insert(&[0.0, 1.0], None)?;
        db.insert(&[1.0, 0.0], None)?;

        let results = db.search(&[1.0, 0.0], 3, 32)?;

        for hit in &results {
            assert!(hit.score >= 0.0, "score must be >= 0.0, got {}", hit.score);
            assert!(hit.score <= 1.0, "score must be <= 1.0, got {}", hit.score);
        }

        if results.len() >= 2 {
            assert!(
                results[0].score >= results[1].score,
                "results should be ordered by score"
            );
        }

        db.flush().await?;
    }
    cleanup("test_score_range.db");
    Ok(())
}

#[tokio::test]
async fn test_search_ordering_correctness() -> Result<()> {
    cleanup("test_search_order.db");
    {
        let cfg = Config::new(3);
        let mut db = VectorDb::open("test_search_order.db", cfg).await?;

        db.insert(&[1.0, 0.0, 0.0], None)?;
        db.insert(&[0.5, 0.5, 0.5], None)?;
        db.insert(&[0.0, 0.0, 1.0], None)?;
        db.insert(&[0.9, 0.0, 0.0], None)?;

        let query = vec![1.0, 0.0, 0.0];
        let results = db.search(&query, 4, 32)?;

        assert!(!results.is_empty());
        for i in 0..results.len() - 1 {
            assert!(
                results[i].score >= results[i + 1].score,
                "results must be sorted by score descending, got scores: {:?}",
                results.iter().map(|h| h.score).collect::<Vec<_>>()
            );
        }

        db.flush().await?;
    }
    cleanup("test_search_order.db");
    Ok(())
}

#[tokio::test]
async fn test_search_skip_deleted_vectors() -> Result<()> {
    cleanup("test_search_deleted.db");
    {
        let cfg = Config::new(2);
        let mut db = VectorDb::open("test_search_deleted.db", cfg).await?;

        db.insert(&[1.0, 0.0], None)?;
        db.insert(&[0.0, 1.0], None)?;
        db.insert(&[1.0, 0.0], None)?;
        db.insert(&[0.5, 0.5], None)?;

        db.delete(0)?;
        assert!(db.is_deleted(0));

        let results = db.search(&[1.0, 0.0], 2, 32)?;

        for hit in &results {
            assert!(
                !db.is_deleted(hit.id),
                "search returned deleted vector id={}",
                hit.id
            );
        }

        db.flush().await?;
    }
    cleanup("test_search_deleted.db");
    Ok(())
}

#[tokio::test]
async fn test_deletion_stats_tracking() -> Result<()> {
    cleanup("test_del_stats.db");
    {
        let cfg = Config::new(2);
        let mut db = VectorDb::open("test_del_stats.db", cfg).await?;

        for i in 0..10 {
            let v = vec![(i as f32) * 0.1, 0.5];
            db.insert(&v, None)?;
        }

        let (deleted, total, ratio) = db.deletion_stats();
        assert_eq!(deleted, 0);
        assert_eq!(total, 10);
        assert_eq!(ratio, 0.0);

        db.delete(0)?;
        db.delete(5)?;
        db.delete(9)?;

        let (deleted, total, ratio) = db.deletion_stats();
        assert_eq!(deleted, 3);
        assert_eq!(total, 10);
        assert!(
            (ratio - 0.3).abs() < 0.01,
            "expected ratio ~0.3, got {}",
            ratio
        );

        assert!(!db.should_compact(), "should not compact at 30% boundary");

        db.delete(7)?;
        assert!(db.should_compact(), "should compact at 40% deletion");

        db.flush().await?;
    }
    cleanup("test_del_stats.db");
    Ok(())
}

#[tokio::test]
async fn test_search_with_high_deletions() -> Result<()> {
    cleanup("test_search_high_del.db");
    {
        let cfg = Config::new(2);
        let mut db = VectorDb::open("test_search_high_del.db", cfg).await?;

        for i in 0..20 {
            let v = vec![(i as f32) * 0.05, 0.5];
            db.insert(&v, None)?;
        }

        for i in (0..20).step_by(2) {
            db.delete(i as u32)?;
        }

        let (_deleted, _total, ratio) = db.deletion_stats();
        assert!(ratio >= 0.45, "should have ~50% deletion, got {}", ratio);

        let results = db.search(&[0.5, 0.5], 5, 32)?;

        assert!(
            !results.is_empty(),
            "search should return results despite deletions"
        );
        for hit in &results {
            assert!(!db.is_deleted(hit.id), "returned deleted vector");
            assert!(hit.score >= 0.0 && hit.score <= 1.0, "score out of range");
        }

        db.flush().await?;
    }
    cleanup("test_search_high_del.db");
    Ok(())
}

#[tokio::test]
async fn test_search_empty_database() -> Result<()> {
    cleanup("test_search_empty.db");
    {
        let cfg = Config::new(2);
        let db = VectorDb::open("test_search_empty.db", cfg).await?;

        let results = db.search(&[1.0, 0.0], 5, 32)?;
        assert!(
            results.is_empty(),
            "search on empty database should return no results"
        );
    }
    cleanup("test_search_empty.db");
    Ok(())
}

#[tokio::test]
async fn test_search_k_greater_than_size() -> Result<()> {
    cleanup("test_search_k_large.db");
    {
        let cfg = Config::new(2);
        let mut db = VectorDb::open("test_search_k_large.db", cfg).await?;

        db.insert(&[1.0, 0.0], None)?;
        db.insert(&[0.0, 1.0], None)?;

        let results = db.search(&[1.0, 0.0], 100, 32)?;

        assert_eq!(
            results.len(),
            2,
            "should return all available vectors when k > size"
        );
        for hit in &results {
            assert!(hit.score >= 0.0 && hit.score <= 1.0, "score out of range");
        }

        db.flush().await?;
    }
    cleanup("test_search_k_large.db");
    Ok(())
}

#[tokio::test]
async fn test_wal_recovery_preserves_metadata() -> Result<()> {
    cleanup("test_wal_meta_preserve.db");

    // Stage 1: Insert and flush (data + metadata on disk, WAL cleared)
    {
        let cfg = Config::new(4).with_capacity(32);
        let mut db = VectorDb::open("test_wal_meta_preserve.db", cfg).await?;
        db.insert(&[1.0, 0.0, 0.0, 0.0], Some(b"first"))?;
        db.insert(&[0.0, 1.0, 0.0, 0.0], Some(b"second"))?;
        db.flush().await?;
    }

    // Stage 2: Insert more without flush (simulates crash before flush)
    {
        let cfg = Config::new(4).with_capacity(32);
        let mut db = VectorDb::open("test_wal_meta_preserve.db", cfg).await?;
        assert_eq!(db.len(), 2);
        db.insert(&[0.0, 0.0, 1.0, 0.0], Some(b"third"))?;
        db.insert(&[0.0, 0.0, 0.0, 1.0], Some(b"fourth"))?;
        // No flush - simulates crash
    }

    // Stage 3: Reopen and verify all metadata recovered from WAL
    {
        let cfg = Config::new(4).with_capacity(32);
        let db = VectorDb::open("test_wal_meta_preserve.db", cfg).await?;
        assert_eq!(db.len(), 4, "Should have 4 vectors");
        assert_eq!(db.get_meta(0)?.map(|m| m.to_vec()), Some(b"first".to_vec()));
        assert_eq!(
            db.get_meta(1)?.map(|m| m.to_vec()),
            Some(b"second".to_vec())
        );
        assert_eq!(db.get_meta(2)?.map(|m| m.to_vec()), Some(b"third".to_vec()));
        assert_eq!(
            db.get_meta(3)?.map(|m| m.to_vec()),
            Some(b"fourth".to_vec())
        );
    }

    cleanup("test_wal_meta_preserve.db");
    Ok(())
}

#[tokio::test]
async fn test_wal_recovery_metadata_restored_when_data_flushed_without_metadata() -> Result<()> {
    cleanup("test_wal_meta_missing.db");

    // Stage 1: Insert vector A and flush (data + metadata on disk, WAL cleared)
    {
        let cfg = Config::new(4).with_capacity(32);
        let mut db = VectorDb::open("test_wal_meta_missing.db", cfg).await?;
        db.insert(&[1.0, 0.0, 0.0, 0.0], Some(b"alpha"))?;
        db.flush().await?;
    }

    // Stage 2: Insert vector B (WAL record created for B)
    {
        let cfg = Config::new(4).with_capacity(32);
        let mut db = VectorDb::open("test_wal_meta_missing.db", cfg).await?;
        assert_eq!(db.len(), 1);
        db.insert(&[0.0, 1.0, 0.0, 0.0], Some(b"beta"))?;
        // Don't flush - WAL has record for B, but data/metadata only in mmap
    }

    // Stage 3: Simulate partial flush crash.
    // Drop db (loses unflushed mmap data). Then manually:
    //   - Append B's vector bytes to the data file and update count to 2
    //     (simulates data file being flushed before crash)
    //   - Leave metadata file with only A's record
    //     (simulates metadata not being flushed before crash)
    //   - Leave WAL with B's record (simulates checkpoint not being written)
    {
        // Write B's vector into the data file and update count
        use std::io::{Seek, SeekFrom, Write};
        let vec_bytes: Vec<u8> = [0.0f32, 1.0, 0.0, 0.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let mut data_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("test_wal_meta_missing.db")?;

        // Update count from 1 to 2 at offset 16..24
        data_file.seek(SeekFrom::Start(16))?;
        data_file.write_all(&2u64.to_le_bytes())?;

        // Append B's vector at offset 32 + 1*4*4 = 48
        data_file.seek(SeekFrom::Start(48))?;
        data_file.write_all(&vec_bytes)?;
        data_file.flush()?;

        // Now the data file has 2 vectors but metadata file only has A's record.
        // WAL still has B's insert record.
    }

    // Stage 4: Reopen - WAL recovery should detect B's metadata is missing and restore it
    {
        let cfg = Config::new(4).with_capacity(32);
        let db = VectorDb::open("test_wal_meta_missing.db", cfg).await?;
        assert_eq!(db.len(), 2, "Should have 2 vectors in data file");
        assert_eq!(
            db.get_meta(0)?.map(|m| m.to_vec()),
            Some(b"alpha".to_vec()),
            "A's metadata should be intact"
        );
        assert_eq!(
            db.get_meta(1)?.map(|m| m.to_vec()),
            Some(b"beta".to_vec()),
            "B's metadata should be restored from WAL"
        );
    }

    cleanup("test_wal_meta_missing.db");
    Ok(())
}

// Regression: compact() used to leave VectorDb pointing at the shut-down WAL
// writer and a dummy lock handle, so inserts failed and no advisory lock was
// held after compaction. It must also discard stale pre-compaction WAL records
// so they are not replayed into the renumbered data on re-open.
#[tokio::test]
async fn test_insert_works_after_compact() -> Result<()> {
    cleanup("test_compact_wal.db");

    {
        let cfg = Config::new(4).with_capacity(64);
        let mut db = VectorDb::open("test_compact_wal.db", cfg).await?;

        for i in 0..10 {
            db.insert(&[i as f32, 1.0, 0.0, 0.5], None)?; // distinct non-zero vectors
        }
        // Delete 4 of 10 => 40% > 30% compaction threshold.
        for id in 0..4 {
            db.delete(id)?;
        }
        assert!(db.should_compact(), "should trigger compaction");
        db.flush().await?;

        db.compact().await?;

        // The WAL must still be functional after compaction.
        let before = db.len();
        assert_eq!(before, 6, "compaction should leave 6 live vectors");
        assert_eq!(db.deleted_count(), 0, "compaction resets deleted_count");
        let new_id = db.insert(&[100.0; 4], Some(b"post-compact"))?;
        assert_eq!(new_id as usize, before, "insert should append after compaction");
        assert_eq!(db.live_len(), before + 1, "6 live + 1 new = 7");
        db.flush().await?;
        db.close().await?;
    }

    // Reopen: the post-compact insert must have been logged and flushed.
    {
        let cfg = Config::new(4).with_capacity(64);
        let db = VectorDb::open("test_compact_wal.db", cfg).await?;
        assert_eq!(db.live_len(), 7, "expected 6 surviving + 1 post-compact insert");
        let found = (0..db.len() as u32).any(|id| {
            db.get_meta(id)
                .ok()
                .flatten()
                .map(|m| m == b"post-compact")
                .unwrap_or(false)
        });
        assert!(found, "post-compact insert metadata should be present after reopen");
    }

    cleanup("test_compact_wal.db");
    Ok(())
}
