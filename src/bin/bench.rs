use rust_rag_mcp::r_vector::{Config, VectorDb, cosine_distance};
use rust_rag_mcp::wal::{Lsn, WalOpType, WalRecord};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::Instant;

// ── Allocation counter ───────────────────────────────────────────────────────
//
// Counts bytes allocated/freed globally. Peak tracks the high-water mark.
// Uses AtomicI64 for `current` to handle the case where deallocs from other
// threads (e.g. WAL writer) free memory allocated before reset_counters().
// When current goes negative, we skip the peak update since it's meaningless.
struct CountingAllocator {
    inner: System,
    allocated: AtomicU64,
    freed: AtomicU64,
    peak: AtomicU64,
    current: AtomicI64,
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe {
            let ptr = self.inner.alloc(layout);
            if !ptr.is_null() {
                let size = layout.size() as u64;
                self.allocated.fetch_add(size, Ordering::Relaxed);
                let prev = self.current.fetch_add(size as i64, Ordering::Relaxed);
                let cur = prev + size as i64;
                if cur > 0 {
                    let cur = cur as u64;
                    loop {
                        let old = self.peak.load(Ordering::Relaxed);
                        if cur <= old {
                            break;
                        }
                        if self
                            .peak
                            .compare_exchange_weak(old, cur, Ordering::Relaxed, Ordering::Relaxed)
                            .is_ok()
                        {
                            break;
                        }
                    }
                }
            }
            ptr
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe {
            self.inner.dealloc(ptr, layout);
            let size = layout.size() as u64;
            self.freed.fetch_add(size, Ordering::Relaxed);
            self.current.fetch_sub(size as i64, Ordering::Relaxed);
        }
    }
}

#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator {
    inner: System,
    allocated: AtomicU64::new(0),
    freed: AtomicU64::new(0),
    peak: AtomicU64::new(0),
    current: AtomicI64::new(0),
};

fn reset_counters() {
    ALLOC.allocated.store(0, Ordering::Relaxed);
    ALLOC.freed.store(0, Ordering::Relaxed);
    ALLOC.peak.store(0, Ordering::Relaxed);
    ALLOC.current.store(0, Ordering::Relaxed);
}

fn snapshot() -> (u64, u64) {
    (
        ALLOC.allocated.load(Ordering::Relaxed),
        ALLOC.peak.load(Ordering::Relaxed),
    )
}

// ── Helpers ──────────────────────────────────────────────────────────────────
fn random_vector(dim: usize, seed: &mut u64) -> Vec<f32> {
    let mut v = Vec::with_capacity(dim);
    for _ in 0..dim {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        v.push((*seed as f32 / u64::MAX as f32) * 2.0 - 1.0);
    }
    v
}

fn human_bytes(b: u64) -> String {
    if b >= 1 << 30 {
        format!("{:.1} GB", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.1} MB", b as f64 / (1u64 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.1} KiB", b as f64 / (1u64 << 10) as f64)
    } else {
        format!("{} B", b)
    }
}

fn ns_per_op(elapsed: std::time::Duration, ops: usize) -> f64 {
    elapsed.as_nanos() as f64 / ops as f64
}

fn print_row(name: &str, ops: usize, elapsed: std::time::Duration, alloc: u64, peak: u64) {
    let ns = ns_per_op(elapsed, ops);
    let alloc_per_op = alloc / ops as u64;
    let peak_str = human_bytes(peak);
    println!(
        "{:<44} {:>8}  {:>10.0} ns/op  {:>10} alloc  {:>10} peak",
        name,
        ops,
        ns,
        human_bytes(alloc_per_op),
        peak_str,
    );
}

// ── Benchmarks ───────────────────────────────────────────────────────────────

fn main() {
    let dim = 128;
    let mut seed: u64 = 0xDEAD_BEEF_CAFE_1234;

    println!("=== RustRAG MCP Hot Path Benchmarks ===");
    println!("dim={}  |  allocation counting via global allocator\n", dim);
    println!("{:-<100}", "");

    // ── 1. cosine_distance: pure computation, no alloc expected ──
    {
        let a = random_vector(dim, &mut seed);
        let b = random_vector(dim, &mut seed);
        reset_counters();
        let n = 2_000_000;
        let start = Instant::now();
        let mut sink = 0.0f32;
        let mut errors = 0usize;
        for _ in 0..n {
            // On error count it — never substitute a fake 0.0 distance.
            match cosine_distance(&a, &b) {
                Ok(d) => sink += d,
                Err(e) => {
                    errors += 1;
                    if errors == 1 {
                        eprintln!("cosine_distance failed: {e}");
                    }
                }
            }
        }
        let elapsed = start.elapsed();
        let (alloc, peak) = snapshot();
        let _ = sink;
        assert_eq!(errors, 0, "cosine_distance errored {errors} times");
        print_row("cosine_distance (128d)", n, elapsed, alloc, peak);
    }

    println!();

    // ── 2. WalRecord allocation: vector.to_vec() + metadata.to_vec() ──
    {
        let v = random_vector(dim, &mut seed);
        let meta = b"test-metadata" as &[u8];
        let n = 100_000;
        reset_counters();
        let start = Instant::now();
        for i in 0..n {
            let record = WalRecord {
                lsn: Lsn(i as u64),
                timestamp: 0,
                op_type: WalOpType::Insert,
                vector_id: i as u32,
                vector_data: v.clone(),
                metadata: meta.to_vec(),
            };
            std::hint::black_box(&record);
        }
        let elapsed = start.elapsed();
        let (alloc, peak) = snapshot();
        print_row("WalRecord alloc (128d, clone)", n, elapsed, alloc, peak);
    }

    // ── 3. WalRecord serialization (write_to) ──
    {
        let v = random_vector(dim, &mut seed);
        let record = WalRecord {
            lsn: Lsn(0),
            timestamp: 0,
            op_type: WalOpType::Insert,
            vector_id: 0,
            vector_data: v,
            metadata: b"test-metadata".to_vec(),
        };
        let n: usize = 100_000;
        reset_counters();
        let start = Instant::now();
        for i in 0..n as u64 {
            let mut r = record.clone();
            r.lsn = Lsn(i);
            let bytes = r.to_bytes();
            std::hint::black_box(&bytes);
        }
        let elapsed = start.elapsed();
        let (alloc, peak) = snapshot();
        print_row("WalRecord::to_bytes (128d)", n, elapsed, alloc, peak);
    }

    println!();

    // ── 4. Full insert path (WAL + HNSW + mmap) ──
    {
        let counts = [100, 500, 1_000, 5_000];
        let rt = tokio::runtime::Runtime::new().unwrap();

        for &n in &counts {
            let tmp =
                std::env::temp_dir().join(format!("bench_insert_{}_{}", std::process::id(), n));
            let _ = std::fs::remove_dir_all(&tmp);
            std::fs::create_dir_all(&tmp).unwrap();
            let db_path = tmp.join("vectors.db");

            let cfg = Config::new(dim).with_capacity(n + 100);
            let mut db = rt.block_on(VectorDb::open(&db_path, cfg)).unwrap();
            let vectors: Vec<Vec<f32>> = (0..n).map(|_| random_vector(dim, &mut seed)).collect();

            reset_counters();
            let start = Instant::now();
            for v in &vectors {
                db.insert(v, Some(b"test-metadata")).unwrap();
            }
            let elapsed = start.elapsed();
            let (alloc, peak) = snapshot();
            print_row(&format!("insert (n={}, 128d)", n), n, elapsed, alloc, peak);

            drop(db);
            let _ = std::fs::remove_dir_all(&tmp);
        }
    }

    println!();

    // ── 5. Search path (read-only) ──
    {
        let counts = [(500, 500), (2_000, 2_000), (10_000, 1_000), (50_000, 500)];
        for &(preload, n_queries) in &counts {
            let tmp = std::env::temp_dir().join(format!(
                "bench_search_{}_{}",
                preload,
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&tmp);
            std::fs::create_dir_all(&tmp).unwrap();
            let db_path = tmp.join("vectors.db");
            let rt = tokio::runtime::Runtime::new().unwrap();
            let cfg = Config::new(dim).with_capacity(preload + 100);
            let mut db = rt.block_on(VectorDb::open(&db_path, cfg)).unwrap();
            for _ in 0..preload {
                let v = random_vector(dim, &mut seed);
                db.insert(&v, None).unwrap();
            }
            let queries: Vec<Vec<f32>> = (0..n_queries)
                .map(|_| random_vector(dim, &mut seed))
                .collect();

            reset_counters();
            let start = Instant::now();
            for q in &queries {
                let _ = db.search(q, 10, 200).unwrap();
            }
            let elapsed = start.elapsed();
            let (alloc, peak) = snapshot();
            print_row(
                &format!("search k=10 ef=200 (loaded={}, q={})", preload, n_queries),
                n_queries,
                elapsed,
                alloc,
                peak,
            );
            let _ = std::fs::remove_dir_all(&tmp);
        }
    }

    println!("\n{:-<100}", "");
}
