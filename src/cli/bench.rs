//! `bench`: put, get, scan and vector-search workloads against FerroStore and
//! redb, each timed operation by operation, reported as p50, p99 and
//! throughput per engine and workload.
//!
//! By default every engine gets a fresh store in a private directory the
//! caller provides ([`Scratch`]), removed before this returns: no existing
//! store is opened. [`BenchTarget::Existing`] runs against a store a person
//! names, only when they also confirm it, and then only on FerroStore and
//! only the document workloads, in a collection of its own that it empties
//! again.
//!
//! The inputs are fixed: document `i` is `doc-<i>` with the same fields on
//! every run, and the vectors come from a fixed-seed generator. Only the
//! timings vary.

use super::{Error, Output, Result, Scratch, Wording, line};
use crate::{BackendKind, FerroStore, VectorMetric, VectorStorage};
use redb::{ReadableDatabase, ReadableTable, TableDefinition};
use serde::Serialize;
use serde_json::{Value, json};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// The default number of operations per workload.
pub const DEFAULT_N: usize = 1000;
/// How many times `scan` reads the whole collection, at most.
const SCANS: usize = 20;
/// The vector workload's collection parameters.
const DIM: u32 = 16;
const K: usize = 10;

/// One workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Workload {
    /// `n` single-document writes, each its own commit.
    Put,
    /// `n` single-document reads.
    Get,
    /// Up to 20 reads of the whole `n`-document collection.
    Scan,
    /// `n` nearest-neighbour searches (k = 10) over `n` vectors (dim 16).
    Vector,
}

impl Workload {
    pub const ALL: [Self; 4] = [Self::Put, Self::Get, Self::Scan, Self::Vector];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Put => "put",
            Self::Get => "get",
            Self::Scan => "scan",
            Self::Vector => "vector",
        }
    }

    /// `put`, `get`, `scan`, `vector`, a comma-separated list of them, or
    /// `all`.
    pub fn parse_list(raw: &str) -> Result<Vec<Self>> {
        let mut out = Vec::new();
        for word in raw.split(',').map(str::trim) {
            let picked: &[Self] = match word {
                "all" => &Self::ALL,
                "put" => &[Self::Put],
                "get" => &[Self::Get],
                "scan" => &[Self::Scan],
                "vector" => &[Self::Vector],
                other => {
                    refuse!("unknown workload `{other}`: expected put, get, scan, vector or all")
                }
            };
            for w in picked {
                if !out.contains(w) {
                    out.push(*w);
                }
            }
        }
        Ok(out)
    }
}

/// `ferro`, `redb`, or both, comma-separated.
pub fn parse_engines(raw: &str) -> Result<Vec<BackendKind>> {
    let mut out = Vec::new();
    for word in raw.split(',').map(str::trim) {
        let kind = match word {
            "ferro" | "ferrostore" => BackendKind::Ferro,
            "redb" => BackendKind::Redb,
            other => refuse!("unknown engine `{other}`: expected ferro or redb"),
        };
        if !out.contains(&kind) {
            out.push(kind);
        }
    }
    Ok(out)
}

/// Which stores a bench writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchTarget {
    /// A fresh store per engine in the scratch directory (the default).
    Fresh,
    /// The existing FerroStore `path` names. Refused unless `confirmed`.
    Existing { path: PathBuf, confirmed: bool },
}

/// What to run.
#[derive(Debug, Clone)]
pub struct BenchOptions {
    pub workloads: Vec<Workload>,
    /// Operations per workload (at least 1).
    pub n: usize,
    pub engines: Vec<BackendKind>,
    pub target: BenchTarget,
}

/// One workload on one engine. Times are in microseconds.
#[derive(Debug, Clone, Serialize)]
pub struct BenchResult {
    pub engine: &'static str,
    pub workload: &'static str,
    /// Operations timed.
    pub ops: u64,
    /// Documents or vectors each operation touches.
    pub items_per_op: u64,
    pub p50_us: f64,
    pub p99_us: f64,
    pub mean_us: f64,
    pub total_ms: f64,
    pub ops_per_sec: f64,
}

/// A workload an engine does not run, and why.
#[derive(Debug, Clone, Serialize)]
pub struct Skipped {
    pub engine: &'static str,
    pub workload: &'static str,
    pub reason: &'static str,
}

/// What `bench` measured. JSON: `command`, `n`, `target` (`fresh`, or the
/// store's path), `results` and `skipped`.
#[derive(Debug, Clone, Serialize)]
pub struct BenchReport {
    pub command: &'static str,
    pub n: u64,
    pub target: String,
    pub results: Vec<BenchResult>,
    pub skipped: Vec<Skipped>,
}

impl BenchReport {
    /// A table, one row per engine and workload, then the skipped ones.
    pub fn text(&self) -> Output {
        let mut out = Output::default();
        let o = &mut out.stdout;
        line(o, format!("bench: n = {}, on {}", self.n, self.target));
        line(
            o,
            format!(
                "{:<6}  {:<8}  {:>7}  {:>12}  {:>12}  {:>12}",
                "engine", "workload", "ops", "p50 us", "p99 us", "ops/s"
            ),
        );
        for r in &self.results {
            line(
                o,
                format!(
                    "{:<6}  {:<8}  {:>7}  {:>12.1}  {:>12.1}  {:>12.0}",
                    r.engine, r.workload, r.ops, r.p50_us, r.p99_us, r.ops_per_sec
                ),
            );
        }
        for s in &self.skipped {
            line(
                o,
                format!("skipped: {} {}: {}", s.engine, s.workload, s.reason),
            );
        }
        out
    }
}

/// Run `options`. Fresh stores are made in the directory `scratch` returns
/// (called only for [`BenchTarget::Fresh`]) and removed with it.
pub fn bench<S: Scratch>(
    options: &BenchOptions,
    scratch: impl FnOnce() -> Result<S>,
    w: &Wording,
) -> Result<BenchReport> {
    if options.n == 0 {
        refuse!("--n must be at least 1");
    }
    if options.workloads.is_empty() || options.engines.is_empty() {
        refuse!("nothing to run: name at least one workload and one engine");
    }
    let n = options.n;
    let mut results = Vec::new();
    let mut skipped = Vec::new();
    let target = match &options.target {
        BenchTarget::Existing { path, confirmed } => {
            if !confirmed {
                refuse!(
                    "`{} bench --path` writes to the store {} (and removes what it wrote): add \
                     --i-know-it-writes to run on it, or leave --path out to bench fresh stores",
                    w.command,
                    path.display()
                );
            }
            if options.engines != [BackendKind::Ferro] {
                refuse!(
                    "`{} bench --path` runs on FerroStore only: pass --engines ferro",
                    w.command
                );
            }
            if options.workloads.contains(&Workload::Vector) {
                refuse!(
                    "`{} bench --path` runs the document workloads only (put, get, scan): a \
                     vector collection cannot be removed again",
                    w.command
                );
            }
            let store = super::data::open_local(path, super::data::Create::Never, "bench", w)?;
            for warning in sweep_stale_collections(&store, path) {
                let _ = writeln!(std::io::stderr(), "{warning}");
            }
            let collection = format!(
                "{BENCH_PREFIX}{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos())
            );
            let engine = Ferro {
                store,
                collection: collection.clone(),
            };
            let run = run_engine(&engine, "ferro", options, &mut results, &mut skipped);
            // Remove what was written, whatever happened.
            let cleaned = (0..n).try_for_each(|i| engine.store.delete_doc(&collection, &doc_id(i)));
            run?;
            cleaned.map_err(|e| {
                Error::context(
                    format!("could not remove the bench collection `{collection}`"),
                    e,
                )
            })?;
            path.display().to_string()
        }
        BenchTarget::Fresh => {
            let dir = scratch()?;
            let ran = run_fresh(dir.path(), options, &mut results, &mut skipped);
            dir.close()
                .map_err(|e| Error::context("could not remove the bench stores", e))?;
            ran?;
            "fresh".to_string()
        }
    };
    Ok(BenchReport {
        command: "bench",
        n: n as u64,
        target,
        results,
        skipped,
    })
}

/// The name of the collection a `bench --path` run writes, before its
/// `<pid>-<n>`.
const BENCH_PREFIX: &str = "ferrostore-bench-";

/// Empty every `ferrostore-bench-<pid>-<n>` collection of `store` (the store
/// `path` names) whose process is gone: what a `bench --path` that was killed
/// before it could remove its documents left. A live process's run is left
/// alone. What could not be emptied is returned, one warning line each.
fn sweep_stale_collections(store: &FerroStore, path: &Path) -> Vec<String> {
    let physical = crate::physical_path(path, BackendKind::Ferro);
    let names: Vec<String> = match crate::ferrostore::inspect_at(&physical) {
        Ok(report) => report
            .contents
            .map(|c| c.collections.into_iter().map(|c| c.name).collect())
            .unwrap_or_default(),
        Err(e) => {
            return vec![format!(
                "warning: could not look for stale bench collections in {}: {e}",
                physical.display()
            )];
        }
    };
    let mut warnings = Vec::new();
    for name in names {
        let Some(pid) = name.strip_prefix(BENCH_PREFIX).and_then(super::pid_of_tail) else {
            continue;
        };
        if super::pid_alive(pid) {
            continue;
        }
        let emptied = store.transaction(|txn| {
            for (id, _) in txn.scan(&name) {
                txn.delete_doc(&name, &id)?;
            }
            Ok(())
        });
        if let Err(e) = emptied {
            warnings.push(format!(
                "warning: could not empty the stale bench collection `{name}` (left by process \
                 {pid}, which is gone): {e}"
            ));
        }
    }
    warnings
}

fn run_fresh(
    dir: &Path,
    options: &BenchOptions,
    results: &mut Vec<BenchResult>,
    skipped: &mut Vec<Skipped>,
) -> Result<()> {
    for kind in &options.engines {
        match kind {
            BackendKind::Ferro => {
                let engine = Ferro {
                    store: FerroStore::open(dir.join("bench.fls"))?,
                    collection: "bench".to_string(),
                };
                run_engine(&engine, "ferro", options, results, skipped)?;
            }
            BackendKind::Redb => {
                let db = redb::Database::create(dir.join("bench.redb"))
                    .map_err(|e| Error::context("could not create the redb bench store", e))?;
                run_engine(&Redb { db }, "redb", options, results, skipped)?;
            }
        }
    }
    Ok(())
}

/// The operations a workload times, on one engine.
trait Engine {
    fn put(&self, id: &str, doc: &Value) -> Result<()>;
    /// Every document in one write, untimed: the data `get` and `scan` read
    /// when `put` did not run first.
    fn load(&self, n: usize) -> Result<()>;
    fn get(&self, id: &str) -> Result<Option<Value>>;
    /// Read every document; how many there were.
    fn scan(&self) -> Result<usize>;
    /// Make the vector collection and fill it, untimed; `None` when the
    /// engine has no vector search.
    fn vectors(&self, n: usize) -> Result<Option<()>>;
    fn search(&self, values: &[f32]) -> Result<usize>;
}

fn doc_id(i: usize) -> String {
    format!("doc-{i:06}")
}

fn doc(i: usize) -> Value {
    json!({
        "i": i,
        "group": format!("g{}", i % 10),
        "text": format!("bench document {i}, word{} and word{}", i % 97, i % 13),
    })
}

/// A fixed-seed generator (64-bit LCG), so the vectors are the same on
/// every run.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn vector(&mut self) -> Vec<f32> {
        (0..DIM).map(|_| self.next()).collect()
    }
}

fn run_engine(
    engine: &dyn Engine,
    name: &'static str,
    options: &BenchOptions,
    results: &mut Vec<BenchResult>,
    skipped: &mut Vec<Skipped>,
) -> Result<()> {
    let n = options.n;
    let mut loaded = false;
    for workload in &options.workloads {
        let mut times = Vec::new();
        let mut items = 1u64;
        match workload {
            Workload::Put => {
                for i in 0..n {
                    let d = doc(i);
                    let id = doc_id(i);
                    let start = Instant::now();
                    engine.put(&id, &d)?;
                    times.push(start.elapsed());
                }
                loaded = true;
            }
            Workload::Get => {
                if !loaded {
                    engine.load(n)?;
                    loaded = true;
                }
                for i in 0..n {
                    let id = doc_id(i);
                    let start = Instant::now();
                    let got = engine.get(&id)?;
                    times.push(start.elapsed());
                    if got.is_none() {
                        refuse!("bench: {name} lost document {id}");
                    }
                }
            }
            Workload::Scan => {
                if !loaded {
                    engine.load(n)?;
                    loaded = true;
                }
                items = n as u64;
                for _ in 0..n.clamp(1, SCANS) {
                    let start = Instant::now();
                    let seen = engine.scan()?;
                    times.push(start.elapsed());
                    if seen != n {
                        refuse!("bench: {name} scanned {seen} documents, expected {n}");
                    }
                }
            }
            Workload::Vector => {
                if engine.vectors(n)?.is_none() {
                    skipped.push(Skipped {
                        engine: name,
                        workload: workload.as_str(),
                        reason: "the engine has no vector search",
                    });
                    continue;
                }
                // A seed apart from the data's.
                let mut queries = Lcg(0x5eed);
                items = n as u64;
                for _ in 0..n {
                    let q = queries.vector();
                    let start = Instant::now();
                    engine.search(&q)?;
                    times.push(start.elapsed());
                }
            }
        }
        results.push(summarise(name, workload.as_str(), items, &times));
    }
    Ok(())
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    // Nearest rank: the smallest value with at least `p` of them at or below it.
    let rank = ((p * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1]
}

fn summarise(
    engine: &'static str,
    workload: &'static str,
    items: u64,
    times: &[Duration],
) -> BenchResult {
    let mut sorted = times.to_vec();
    sorted.sort();
    let total: Duration = sorted.iter().sum();
    let us = |d: Duration| d.as_secs_f64() * 1e6;
    let ops = sorted.len() as f64;
    BenchResult {
        engine,
        workload,
        ops: sorted.len() as u64,
        items_per_op: items,
        p50_us: us(percentile(&sorted, 0.50)),
        p99_us: us(percentile(&sorted, 0.99)),
        mean_us: us(total) / ops,
        total_ms: total.as_secs_f64() * 1e3,
        ops_per_sec: if total.is_zero() {
            0.0
        } else {
            ops / total.as_secs_f64()
        },
    }
}

struct Ferro {
    store: FerroStore,
    collection: String,
}

const VECTORS: &str = "bench-vectors";

impl Engine for Ferro {
    fn put(&self, id: &str, doc: &Value) -> Result<()> {
        Ok(self.store.put_doc(&self.collection, id, doc)?)
    }

    fn load(&self, n: usize) -> Result<()> {
        Ok(self.store.transaction(|txn| {
            for i in 0..n {
                txn.put_doc(&self.collection, &doc_id(i), &doc(i))?;
            }
            Ok(())
        })?)
    }

    fn get(&self, id: &str) -> Result<Option<Value>> {
        Ok(self.store.get_doc(&self.collection, id)?)
    }

    fn scan(&self) -> Result<usize> {
        Ok(self
            .store
            .transaction(|txn| Ok(txn.scan(&self.collection).len()))?)
    }

    fn vectors(&self, n: usize) -> Result<Option<()>> {
        self.store.create_vector_collection(
            VECTORS,
            DIM,
            VectorStorage::F32,
            VectorMetric::Cosine,
            16,
            64,
            64,
        )?;
        let mut data = Lcg(0xda7a);
        let ids: Vec<String> = (0..n).map(doc_id).collect();
        let values: Vec<Vec<f32>> = (0..n).map(|_| data.vector()).collect();
        let items: Vec<(&str, &[f32], Option<&Value>)> = ids
            .iter()
            .zip(&values)
            .map(|(id, v)| (id.as_str(), v.as_slice(), None))
            .collect();
        self.store.upsert_vectors(VECTORS, &items)?;
        Ok(Some(()))
    }

    fn search(&self, values: &[f32]) -> Result<usize> {
        Ok(self.store.search_vectors(VECTORS, values, K, None)?.len())
    }
}

struct Redb {
    db: redb::Database,
}

const TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("bench");

fn redb_err(e: impl std::error::Error + Send + Sync + 'static) -> Error {
    Error::context("redb", e)
}

impl Redb {
    fn write(&self, docs: impl Iterator<Item = (String, Value)>) -> Result<()> {
        let tx = self.db.begin_write().map_err(redb_err)?;
        {
            let mut table = tx.open_table(TABLE).map_err(redb_err)?;
            for (id, doc) in docs {
                let bytes = serde_json::to_vec(&doc).map_err(Error::Json)?;
                table
                    .insert(id.as_str(), bytes.as_slice())
                    .map_err(redb_err)?;
            }
        }
        tx.commit().map_err(redb_err)
    }
}

impl Engine for Redb {
    fn put(&self, id: &str, doc: &Value) -> Result<()> {
        self.write(std::iter::once((id.to_string(), doc.clone())))
    }

    fn load(&self, n: usize) -> Result<()> {
        self.write((0..n).map(|i| (doc_id(i), doc(i))))
    }

    fn get(&self, id: &str) -> Result<Option<Value>> {
        let tx = self.db.begin_read().map_err(redb_err)?;
        let table = tx.open_table(TABLE).map_err(redb_err)?;
        match table.get(id).map_err(redb_err)? {
            Some(bytes) => Ok(Some(
                serde_json::from_slice(bytes.value()).map_err(Error::Json)?,
            )),
            None => Ok(None),
        }
    }

    fn scan(&self) -> Result<usize> {
        let tx = self.db.begin_read().map_err(redb_err)?;
        let table = tx.open_table(TABLE).map_err(redb_err)?;
        let mut seen = 0;
        for entry in table.iter().map_err(redb_err)? {
            let (_, bytes) = entry.map_err(redb_err)?;
            let _: Value = serde_json::from_slice(bytes.value()).map_err(Error::Json)?;
            seen += 1;
        }
        Ok(seen)
    }

    fn vectors(&self, _n: usize) -> Result<Option<()>> {
        Ok(None)
    }

    fn search(&self, _values: &[f32]) -> Result<usize> {
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workloads_and_engines_parse_from_lists() {
        assert_eq!(Workload::parse_list("all").unwrap(), Workload::ALL);
        assert_eq!(
            Workload::parse_list("get,put,get").unwrap(),
            [Workload::Get, Workload::Put]
        );
        assert!(Workload::parse_list("delete").is_err());
        assert_eq!(
            parse_engines("ferro,redb").unwrap(),
            [BackendKind::Ferro, BackendKind::Redb]
        );
        assert!(parse_engines("sqlite").is_err());
    }

    #[test]
    fn the_percentiles_are_nearest_rank() {
        let ms = |v: u64| Duration::from_millis(v);
        let sorted: Vec<Duration> = (1..=100).map(ms).collect();
        assert_eq!(percentile(&sorted, 0.50), ms(50));
        assert_eq!(percentile(&sorted, 0.99), ms(99));
        assert_eq!(percentile(&[ms(7)], 0.99), ms(7));
    }

    #[test]
    fn the_inputs_are_the_same_on_every_run() {
        assert_eq!(Lcg(1).vector(), Lcg(1).vector());
        assert!(Lcg(1).vector().iter().all(|v| (-1.0..=1.0).contains(v)));
        assert_eq!(doc(3), doc(3));
    }
}
