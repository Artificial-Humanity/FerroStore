//! The code-index vector workload, timed: a cold build of 11,247 vectors
//! (dim 256, `f32`, cosine, HNSW `m = 16`, `ef_construction = 200`) into a
//! real `.fls` file, then open time and warm search latency.
//!
//! Ignored by default. Run it in a release build:
//!
//! ```text
//! cargo test --release -p fl-store --test vector_workload_bench -- --ignored --nocapture
//! ```
//!
//! The files go under Cargo's target directory (on disk, not the `/tmp`
//! tmpfs) and are removed afterwards.

use fl_store::{FerroStore, VectorMetric, VectorStorage};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const VECTORS: usize = 11_247;
const DIM: usize = 256;
const M: u32 = 16;
const EF_CONSTRUCTION: u32 = 200;
const EF_SEARCH: u32 = 64;
const QUERIES: usize = 200;
const K: usize = 10;
const SINGLES: usize = 1_000;

/// Deterministic pseudo-random components in `[-0.5, 0.5)`.
fn values(seed: u64) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..DIM)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

struct Corpus {
    items: Vec<(String, Vec<f32>, Value)>,
}

impl Corpus {
    fn new() -> Self {
        let items = (0..VECTORS)
            .map(|i| {
                (
                    format!("src/file_{:04}.rs#chunk{}", i / 8, i % 8),
                    values(i as u64 + 1),
                    json!({"path": format!("src/file_{:04}.rs", i / 8), "chunk": i % 8}),
                )
            })
            .collect();
        Self { items }
    }

    fn batch(&self, range: std::ops::Range<usize>) -> Vec<(&str, &[f32], Option<&Value>)> {
        self.items[range]
            .iter()
            .map(|(id, values, metadata)| (id.as_str(), values.as_slice(), Some(metadata)))
            .collect()
    }
}

fn create(store: &FerroStore) {
    store
        .create_vector_collection(
            "code",
            DIM as u32,
            VectorStorage::F32,
            VectorMetric::Cosine,
            M,
            EF_CONSTRUCTION,
            EF_SEARCH,
        )
        .unwrap();
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// Cold-build every vector in batches of `batch`; the total, and the slowest
/// and last batch.
fn build(store: &FerroStore, corpus: &Corpus, batch: usize) -> Duration {
    let start = Instant::now();
    let mut slowest = Duration::ZERO;
    let mut last = Duration::ZERO;
    let mut at = 0;
    while at < VECTORS {
        let end = (at + batch).min(VECTORS);
        let items = corpus.batch(at..end);
        let t = Instant::now();
        store.upsert_vectors("code", &items).unwrap();
        last = t.elapsed();
        slowest = slowest.max(last);
        at = end;
    }
    let total = start.elapsed();
    println!(
        "batch {batch:>4}: cold build {VECTORS} vectors in {:.2} s = {:.3} ms/vector \
         (slowest batch {:.0} ms, last batch {:.0} ms)",
        total.as_secs_f64(),
        ms(total) / VECTORS as f64,
        ms(slowest),
        ms(last),
    );
    total
}

/// p50 and p99 of `QUERIES` searches, after 10 unmeasured ones.
fn search_latency(store: &FerroStore, label: &str) {
    for q in 0..10 {
        store
            .search_vectors("code", &values(1_000_000 + q), K, None)
            .unwrap();
    }
    let mut times: Vec<Duration> = (0..QUERIES)
        .map(|q| {
            let query = values(2_000_000 + q as u64);
            let t = Instant::now();
            let hits = store.search_vectors("code", &query, K, None).unwrap();
            let took = t.elapsed();
            assert_eq!(hits.len(), K);
            took
        })
        .collect();
    times.sort();
    let p = |q: f64| ms(times[((times.len() as f64 * q) as usize).min(times.len() - 1)]);
    println!(
        "{label}: warm search k={K} ef_search={EF_SEARCH}: p50 {:.2} ms, p99 {:.2} ms, max {:.2} ms",
        p(0.50),
        p(0.99),
        ms(*times.last().unwrap())
    );
}

fn open_timed(path: &Path, label: &str) -> FerroStore {
    let t = Instant::now();
    let store = FerroStore::open(path).unwrap();
    // The open loads the state; the first read only checks the slots.
    assert!(store.has_vector_collection("code").unwrap());
    println!("{label}: open {:.0} ms", ms(t.elapsed()));
    store
}

/// A scratch directory under the target dir, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("vector-workload-bench-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
#[ignore = "workload benchmark: run in release with --ignored --nocapture"]
fn code_index_vector_workload() {
    let scratch = Scratch::new();
    let corpus = Corpus::new();
    println!(
        "workload: {VECTORS} vectors, dim {DIM} f32 cosine, m {M}, ef_construction \
         {EF_CONSTRUCTION}, ef_search {EF_SEARCH}, file under {}",
        scratch.0.display()
    );

    // Single upserts, first 1,000: one commit and one fsync each.
    {
        let store = FerroStore::open(scratch.0.join("single.fls")).unwrap();
        create(&store);
        let start = Instant::now();
        for (id, values, metadata) in &corpus.items[..SINGLES] {
            store
                .upsert_vector("code", id, values, Some(metadata))
                .unwrap();
        }
        let total = start.elapsed();
        println!(
            "single upsert_vector: first {SINGLES} vectors in {:.2} s = {:.3} ms/vector",
            total.as_secs_f64(),
            ms(total) / SINGLES as f64
        );
    }

    // Batch 512, for comparison.
    {
        let store = FerroStore::open(scratch.0.join("b512.fls")).unwrap();
        create(&store);
        build(&store, &corpus, 512);
    }

    // Batch 256: the build the rest of the run measures.
    let path = scratch.0.join("b256.fls");
    {
        let store = FerroStore::open(&path).unwrap();
        create(&store);
        build(&store, &corpus, 256);
        // The same handle, warm.
        search_latency(&store, "same handle");
    }
    println!(
        "file: {:.1} MiB",
        std::fs::metadata(&path).unwrap().len() as f64 / (1 << 20) as f64
    );

    // Reopen: the snapshot plus a WAL of every batch, replayed.
    {
        let store = open_timed(&path, "reopen (WAL replay)");
        search_latency(&store, "after WAL-replay open");
        let t = Instant::now();
        store.checkpoint().unwrap();
        println!("checkpoint: {:.0} ms", ms(t.elapsed()));
    }

    // Reopen from the checkpoint's snapshot.
    {
        let store = open_timed(&path, "reopen (snapshot)");
        search_latency(&store, "after snapshot open");
    }
    let mib = |bytes: u64| bytes as f64 / (1 << 20) as f64;
    println!(
        "file after checkpoint: {:.1} MiB",
        mib(std::fs::metadata(&path).unwrap().len())
    );

    // Compaction (spec §5): one snapshot in a new file, renamed over the old.
    {
        let store = FerroStore::open(&path).unwrap();
        let before = store.stats().unwrap();
        let t = Instant::now();
        store.compact().unwrap();
        let took = t.elapsed();
        let after = store.stats().unwrap();
        println!(
            "compact: {:.0} ms; file {:.1} MiB ({:.1} MiB live, {:.1} MiB dead) -> {:.1} MiB \
             ({:.1} MiB dead)",
            ms(took),
            mib(before.file_bytes),
            mib(before.live_bytes),
            mib(before.dead_bytes),
            mib(after.file_bytes),
            mib(after.dead_bytes)
        );
        assert_eq!(after.dead_bytes, 0);
        assert_eq!(after.file_bytes, std::fs::metadata(&path).unwrap().len());
    }
    {
        let store = open_timed(&path, "reopen (compacted)");
        search_latency(&store, "after compacted open");
    }
}
