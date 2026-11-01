//! `FerroStore::open` of a large vector sidecar, timed and split: 21,461
//! vectors (dim 256, cosine, HNSW `m = 16`, `ef_construction = 200`), one
//! `f32` and one `f16` store, each checkpointed so the open decodes one
//! snapshot and replays an empty WAL.
//!
//! Ignored by default. Run it in a release build:
//!
//! ```text
//! cargo test --release -p fl-store --lib open_bench -- --ignored --nocapture
//! ```
//!
//! The files go under Cargo's target directory (on disk, not the `/tmp`
//! tmpfs) and are removed afterwards.

use super::super::container_io::{ContainerFs, RealFs};
use super::super::{FerroStore, latest_superblock, read_range, vector};
use super::{crc32, crc32_bitwise, decode_snapshot, decode_snapshot_with};
use crate::{VectorMetric, VectorStorage};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const VECTORS: usize = 21_461;
const DIM: usize = 256;
const M: u32 = 16;
const EF_CONSTRUCTION: u32 = 200;
const EF_SEARCH: u32 = 64;
const BATCH: usize = 512;
const RUNS: usize = 7;

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

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// The median of `RUNS` timings of `f`.
fn median(mut f: impl FnMut() -> Duration) -> Duration {
    let mut times: Vec<Duration> = (0..RUNS).map(|_| f()).collect();
    times.sort();
    times[RUNS / 2]
}

/// A scratch directory under the target dir, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        // `<target>/<profile>/deps/<test binary>`: `CARGO_TARGET_TMPDIR` is only
        // set for integration tests, so take `<target>/tmp` from the binary's path.
        let exe = std::env::current_exe().unwrap();
        let target = exe.ancestors().nth(3).unwrap();
        let dir = target
            .join("tmp")
            .join(format!("ferrostore-open-bench-{}", std::process::id()));
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

fn build(path: &Path, storage: VectorStorage) {
    let store = FerroStore::open(path).unwrap();
    store
        .create_vector_collection(
            "code",
            DIM as u32,
            storage,
            VectorMetric::Cosine,
            M,
            EF_CONSTRUCTION,
            EF_SEARCH,
        )
        .unwrap();
    let items: Vec<(String, Vec<f32>, Value)> = (0..VECTORS)
        .map(|i| {
            (
                format!("src/file_{:04}.rs#chunk{}", i / 8, i % 8),
                values(i as u64 + 1),
                json!({"path": format!("src/file_{:04}.rs", i / 8), "chunk": i % 8}),
            )
        })
        .collect();
    let t = Instant::now();
    for chunk in items.chunks(BATCH) {
        let batch: Vec<(&str, &[f32], Option<&Value>)> = chunk
            .iter()
            .map(|(id, values, metadata)| (id.as_str(), values.as_slice(), Some(metadata)))
            .collect();
        store.upsert_vectors("code", &batch).unwrap();
    }
    let built = t.elapsed();
    let t = Instant::now();
    store.checkpoint().unwrap();
    println!(
        "{storage:?}: built {VECTORS} vectors in {:.1} s, checkpoint {:.0} ms",
        built.as_secs_f64(),
        ms(t.elapsed())
    );
}

fn read_snapshot(path: &Path) -> Vec<u8> {
    let file = RealFs.open_ro(path).unwrap();
    let (_, sb) = latest_superblock(&*file).unwrap();
    assert_eq!(sb.wal_length, 0, "the open replays an empty WAL");
    read_range(&*file, sb.snapshot_offset, sb.snapshot_length).unwrap()
}

fn measure(path: &Path, storage: VectorStorage) {
    let file_mib = std::fs::metadata(path).unwrap().len() as f64 / f64::from(1 << 20);
    let open = median(|| {
        let t = Instant::now();
        let store = FerroStore::open(path).unwrap();
        let took = t.elapsed();
        assert!(store.has_vector_collection("code").unwrap());
        took
    });
    let read = median(|| {
        let t = Instant::now();
        let bytes = read_snapshot(path);
        let took = t.elapsed();
        assert!(!bytes.is_empty());
        took
    });
    let bytes = read_snapshot(path);
    let body = &bytes[..bytes.len() - 4];
    let crc = median(|| {
        let t = Instant::now();
        let crc = crc32(body);
        let took = t.elapsed();
        assert_ne!(crc, 0);
        took
    });
    let crc_bitwise = median(|| {
        let t = Instant::now();
        let crc = crc32_bitwise(body);
        let took = t.elapsed();
        assert_eq!(crc, crc32(body));
        took
    });
    let decode = median(|| {
        let t = Instant::now();
        let snap = decode_snapshot_with(&bytes, |_, _| Ok(())).unwrap();
        let took = t.elapsed();
        assert_eq!(snap.vectors["code"].vectors.len(), VECTORS);
        took
    });
    let snap = decode_snapshot(&bytes).unwrap();
    let col = &snap.vectors["code"];
    let structure = median(|| {
        let t = Instant::now();
        vector::validate_graph("code", col).unwrap();
        t.elapsed()
    });
    let full = median(|| {
        let t = Instant::now();
        vector::validate_graph_full("code", col).unwrap();
        t.elapsed()
    });
    // Pieces of the structural check, each timed on its own.
    let lookups = median(|| {
        let t = Instant::now();
        let index: HashMap<&str, u32> = col
            .vectors
            .keys()
            .enumerate()
            .map(|(at, id)| (id.as_str(), at as u32))
            .collect();
        let mut sum = 0u64;
        for rec in col.vectors.values() {
            for nids in &rec.neighbors {
                for nid in nids {
                    sum += u64::from(index[nid.as_str()]);
                }
            }
        }
        let took = t.elapsed();
        assert_ne!(sum, 0);
        took
    });
    let levels = median(|| {
        let t = Instant::now();
        for (id, rec) in &col.vectors {
            assert_eq!(vector::vector_level(id, col.m), rec.level);
        }
        t.elapsed()
    });
    let stored = median(|| {
        let t = Instant::now();
        for rec in col.vectors.values() {
            vector::check_stored(col.storage, col.dim, &rec.bytes).unwrap();
        }
        t.elapsed()
    });
    let links: usize = col
        .vectors
        .values()
        .flat_map(|rec| rec.neighbors.iter().map(Vec::len))
        .sum();
    println!(
        "{storage:?}: {links} links; inside the structural check: id->index map + one lookup \
         per link {:.0} ms, SHA-256 level per id {:.0} ms, stored-bytes check {:.0} ms",
        ms(lookups),
        ms(levels),
        ms(stored),
    );
    println!(
        "{storage:?}: file {file_mib:.1} MiB, snapshot {:.1} MiB; median of {RUNS}:\n  \
         FerroStore::open {:.0} ms\n  \
         file read (superblock + snapshot extent) {:.1} ms\n  \
         snapshot decode without graph validation {:.0} ms (of it, crc32 {:.0} ms; \
         the bit-at-a-time crc32 takes {:.0} ms)\n  \
         structural validation (open path) {:.0} ms\n  \
         distance-order validation (validate_graph_full - validate_graph, not on open) {:.0} ms",
        bytes.len() as f64 / f64::from(1 << 20),
        ms(open),
        ms(read),
        ms(decode),
        ms(crc),
        ms(crc_bitwise),
        ms(structure),
        ms(full.saturating_sub(structure)),
    );
}

#[test]
#[ignore = "open benchmark: run in release with --ignored --nocapture"]
fn open_bench_large_vector_sidecar() {
    let scratch = Scratch::new();
    for (storage, name) in [
        (VectorStorage::F32, "f32.fls"),
        (VectorStorage::F16, "f16.fls"),
    ] {
        let path = scratch.0.join(name);
        build(&path, storage);
        measure(&path, storage);
    }
}
