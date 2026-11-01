//! The public multi-op transaction, the batch vector upsert, the
//! vector-collection lookup, and the error of a durable but undelivered
//! `WriteThrough` commit.

use super::{
    FerroStore, FerroStoreServer, SyncAdapter, SyncBatch, SyncMode, Transaction, VectorMetric,
    VectorStorage, vector,
};
use fl_core::StoreError;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

/// A store of each mode. The remote one is served from an embedded disk
/// store over a Unix socket; the guard keeps the server and the files alive.
struct Modes {
    _dir: tempfile::TempDir,
    _server: FerroStoreServer,
    stores: Vec<(&'static str, FerroStore)>,
    disk_path: PathBuf,
    sock: PathBuf,
}

fn modes() -> Modes {
    let dir = tempfile::tempdir().unwrap();
    let disk_path = dir.path().join("disk.fls");
    let served = dir.path().join("served.fls");
    let sock = dir.path().join("served.sock");
    let server = FerroStoreServer::bind_unix(&sock, FerroStore::open(&served).unwrap()).unwrap();
    let remote = FerroStore::connect_uri(&format!("unix://{}", sock.display())).unwrap();
    Modes {
        stores: vec![
            ("memory", FerroStore::in_memory()),
            ("disk", FerroStore::open(&disk_path).unwrap()),
            ("remote", remote),
        ],
        disk_path,
        sock,
        _dir: dir,
        _server: server,
    }
}

fn last_seq(store: &FerroStore) -> u64 {
    store.last_seq().unwrap()
}

fn make_vectors(store: &FerroStore, name: &str, dim: u32) {
    store
        .create_vector_collection(
            name,
            dim,
            VectorStorage::F32,
            VectorMetric::Cosine,
            4,
            16,
            16,
        )
        .unwrap();
}

/// Deterministic pseudo-random vectors.
fn values(seed: u64, dim: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..dim)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn counter(txn: &Transaction<'_>) -> u64 {
    txn.get_doc("counters", "c")
        .and_then(|doc| doc["n"].as_u64())
        .unwrap_or(0)
}

#[test]
fn a_transaction_commits_every_op_as_one_sequence() {
    let modes = modes();
    for (mode, store) in &modes.stores {
        make_vectors(store, "v", 3);
        let before = last_seq(store);
        let out = store
            .transaction(|txn| {
                txn.put_doc("docs", "a", &json!({"n": 1}))?;
                txn.put_doc("docs", "b", &json!({"n": 2}))?;
                txn.delete_doc("docs", "b")?;
                txn.upsert_vector("v", "x", &[1.0, 0.0, 0.0], Some(&json!({"k": 1})))?;
                Ok(7)
            })
            .unwrap();
        assert_eq!(out, 7, "{mode}");
        assert_eq!(last_seq(store), before + 1, "{mode}: one commit");
        assert_eq!(
            store.get_doc("docs", "a").unwrap(),
            Some(json!({"n": 1})),
            "{mode}"
        );
        assert_eq!(store.get_doc("docs", "b").unwrap(), None, "{mode}");
        assert!(store.get_vector("v", "x").unwrap().is_some(), "{mode}");
    }
}

#[test]
fn a_failing_op_mid_body_applies_nothing() {
    let modes = modes();
    for (mode, store) in &modes.stores {
        make_vectors(store, "v", 3);
        store.put_doc("docs", "kept", &json!(0)).unwrap();
        let before = last_seq(store);
        let err = store
            .transaction(|txn| {
                txn.put_doc("docs", "a", &json!({"n": 1}))?;
                txn.delete_doc("docs", "kept")?;
                txn.upsert_vector("v", "x", &[1.0, 0.0, 0.0], None)?;
                // Wrong length: the op is refused and the body returns it.
                txn.upsert_vector("v", "y", &[1.0], None)?;
                txn.put_doc("docs", "never", &json!(1))?;
                Ok(())
            })
            .unwrap_err();
        assert!(err.to_string().contains("length 1"), "{mode}: {err}");
        assert_eq!(last_seq(store), before, "{mode}: nothing committed");
        assert_eq!(store.get_doc("docs", "a").unwrap(), None, "{mode}");
        assert_eq!(store.get_doc("docs", "kept").unwrap(), Some(json!(0)));
        assert_eq!(store.get_vector("v", "x").unwrap(), None, "{mode}");
    }
}

#[test]
fn a_body_that_ignores_a_failed_staged_op_still_cannot_commit() {
    let store = FerroStore::in_memory();
    make_vectors(&store, "v", 2);
    let before = last_seq(&store);
    // A refused write, a reserved put and a second create of the same
    // collection, each ignored by the body.
    let err = store
        .transaction(|txn| {
            txn.put_doc("docs", "a", &json!(1))?;
            let _ = txn.put_doc("__meta", "x", &json!(1));
            Ok(())
        })
        .unwrap_err();
    assert!(err.to_string().contains("cannot commit"), "{err}");
    let err = store
        .transaction(|txn| {
            txn.put_doc("docs", "a", &json!(1))?;
            let _ = txn.create_vector_collection(
                "v",
                2,
                VectorStorage::F32,
                VectorMetric::L2,
                4,
                16,
                16,
            );
            Ok(())
        })
        .unwrap_err();
    assert!(err.to_string().contains("cannot commit"), "{err}");
    assert_eq!(last_seq(&store), before);
    assert_eq!(store.get_doc("docs", "a").unwrap(), None);
}

#[test]
fn a_transaction_reads_its_own_writes() {
    let modes = modes();
    for (mode, store) in &modes.stores {
        store.put_doc("docs", "old", &json!({"tag": "x"})).unwrap();
        store
            .transaction(|txn| {
                assert!(!txn.has_vector_collection("v"), "{mode}");
                txn.create_vector_collection(
                    "v",
                    2,
                    VectorStorage::F32,
                    VectorMetric::L2,
                    4,
                    16,
                    16,
                )?;
                assert!(txn.has_vector_collection("v"), "{mode}");
                txn.upsert_vector("v", "p", &[1.0, 2.0], Some(&json!("m")))?;
                // A remote transaction refuses to read its own vector writes
                // back (spec §6.1): the server applies them.
                if *mode == "remote" {
                    let err = txn.get_vector("v", "p").unwrap_err();
                    assert!(
                        err.to_string()
                            .contains("cannot read back its own vector writes"),
                        "{err}"
                    );
                } else {
                    assert_eq!(
                        txn.get_vector("v", "p")?,
                        Some((vec![1.0, 2.0], Some(json!("m")))),
                        "{mode}"
                    );
                }
                txn.delete_vector("v", "p")?;
                if *mode != "remote" {
                    assert_eq!(txn.get_vector("v", "p")?, None, "{mode}");
                }

                txn.put_doc("docs", "new", &json!({"tag": "x"}))?;
                assert_eq!(txn.get_doc("docs", "new"), Some(json!({"tag": "x"})));
                assert_eq!(
                    txn.query_by_field("docs", "tag", "x"),
                    vec!["new".to_string(), "old".to_string()],
                    "{mode}"
                );
                txn.delete_doc("docs", "old")?;
                assert_eq!(txn.get_doc("docs", "old"), None, "{mode}");
                assert_eq!(
                    txn.scan("docs"),
                    vec![("new".to_string(), json!({"tag": "x"}))],
                    "{mode}"
                );

                txn.create_column_table(
                    "t",
                    &[super::ColumnDef {
                        name: "n".into(),
                        ty: super::ColumnType::Int64,
                    }],
                )?;
                txn.append_column_rows("t", &[vec![super::ColumnValue::Int64(5)]])?;
                Ok(())
            })
            .unwrap();
        assert!(store.has_vector_collection("v").unwrap(), "{mode}");
        assert_eq!(store.get_vector("v", "p").unwrap(), None, "{mode}");
        assert_eq!(store.get_doc("docs", "old").unwrap(), None, "{mode}");
        let scan = store.scan_column_table("t", &[]).unwrap();
        assert_eq!(
            scan.rows,
            vec![vec![super::ColumnValue::Int64(5)]],
            "{mode}"
        );
    }
}

#[test]
fn a_transaction_refuses_reserved_collections() {
    let modes = modes();
    for (mode, store) in &modes.stores {
        let before = last_seq(store);
        for write in [
            &(|txn: &mut Transaction<'_>| txn.put_doc("__meta", "x", &json!(1)))
                as &dyn Fn(&mut Transaction<'_>) -> Result<(), StoreError>,
            &|txn: &mut Transaction<'_>| txn.delete_doc("__meta", "schema_version"),
        ] {
            let err = store.transaction(write).unwrap_err();
            assert!(err.to_string().contains("reserved"), "{mode}: {err}");
        }
        assert_eq!(last_seq(store), before, "{mode}");
        // A read of a reserved collection is allowed, as `get_doc` allows it.
        let version = store
            .transaction(|txn| Ok(txn.get_doc("__meta", "schema_version")))
            .unwrap();
        assert!(version.is_some(), "{mode}");
    }
}

#[test]
fn a_body_that_writes_nothing_commits_nothing() {
    let modes = modes();
    for (mode, store) in &modes.stores {
        let before = last_seq(store);
        assert_eq!(store.transaction(|txn| Ok(counter(txn))).unwrap(), 0);
        assert_eq!(last_seq(store), before, "{mode}");
    }
}

#[test]
fn a_disk_transaction_runs_its_body_once() {
    let dir = tempfile::tempdir().unwrap();
    let store = FerroStore::open(dir.path().join("s.fls")).unwrap();
    let runs = AtomicUsize::new(0);
    store
        .transaction(|txn| {
            runs.fetch_add(1, Ordering::SeqCst);
            let n = counter(txn);
            txn.put_doc("counters", "c", &json!({"n": n + 1}))
        })
        .unwrap();
    assert_eq!(runs.load(Ordering::SeqCst), 1);
}

const THREADS: u64 = 8;
const PER_THREAD: u64 = 25;

/// Retries of one increment that lost every compare-and-swap attempt. A
/// finite cap, so a real livelock fails the test instead of hanging it.
const MAX_CONFLICT_RETRIES: u32 = 1_000;

/// Increments the counter once. A remote handle may return
/// `StoreError::Conflict` when its bounded compare-and-swap loop is exhausted
/// (spec §4.5); that write committed nothing, so the caller retries, and the
/// retry cannot count twice. Any other error panics. Returns the number of
/// conflicts seen.
fn increment(store: &FerroStore) -> u32 {
    let mut conflicts = 0;
    loop {
        let result = store.transaction(|txn| {
            let n = counter(txn);
            txn.put_doc("counters", "c", &json!({"n": n + 1}))
        });
        match result {
            Ok(_) => return conflicts,
            Err(StoreError::Conflict { .. }) => {
                conflicts += 1;
                assert!(
                    conflicts <= MAX_CONFLICT_RETRIES,
                    "increment still conflicting after {MAX_CONFLICT_RETRIES} retries"
                );
            }
            Err(err) => panic!("increment failed: {err}"),
        }
    }
}

/// Runs the contending threads, one handle each, and returns the conflicts
/// they retried through.
fn hammer(make: impl Fn() -> FerroStore + Sync) -> u32 {
    thread::scope(|scope| {
        let threads: Vec<_> = (0..THREADS)
            .map(|_| {
                let store = make();
                scope.spawn(move || (0..PER_THREAD).map(|_| increment(&store)).sum::<u32>())
            })
            .collect();
        threads.into_iter().map(|t| t.join().unwrap()).sum()
    })
}

fn count(store: &FerroStore) -> u64 {
    store.get_doc("counters", "c").unwrap().unwrap()["n"]
        .as_u64()
        .unwrap()
}

#[test]
fn concurrent_read_modify_write_transactions_lose_no_update() {
    let modes = modes();
    // One handle shared by every thread, in each mode.
    for (mode, store) in &modes.stores {
        thread::scope(|scope| {
            for _ in 0..THREADS {
                scope.spawn(|| {
                    for _ in 0..PER_THREAD {
                        // Only a remote handle may conflict.
                        let conflicts = increment(store);
                        assert!(
                            conflicts == 0 || *mode == "remote",
                            "{mode}: unexpected conflict"
                        );
                    }
                });
            }
        });
        assert_eq!(count(store), THREADS * PER_THREAD, "{mode}");
    }
    // A handle per thread: disk handles serialize on the flock, remote
    // clients on the server's compare-and-swap.
    let disk = modes.disk_path.clone();
    assert_eq!(
        hammer(|| FerroStore::open(&disk).unwrap()),
        0,
        "disk conflicts"
    );
    assert_eq!(
        count(&FerroStore::open(&disk).unwrap()),
        2 * THREADS * PER_THREAD
    );
    let uri = format!("unix://{}", modes.sock.display());
    let conflicts = hammer(|| FerroStore::connect_uri(&uri).unwrap());
    eprintln!("remote hammer: {conflicts} conflicts retried");
    assert_eq!(
        count(&FerroStore::connect_uri(&uri).unwrap()),
        2 * THREADS * PER_THREAD
    );
}

#[test]
fn a_get_then_put_without_a_transaction_can_lose_an_update() {
    // Two writers read the same value before either writes: the second put
    // overwrites the first. `transaction` runs the read and the write under
    // one write lock, which is what the test above relies on.
    let store = FerroStore::in_memory();
    store.put_doc("counters", "c", &json!({"n": 0})).unwrap();
    let first = count(&store);
    let second = count(&store);
    store
        .put_doc("counters", "c", &json!({"n": first + 1}))
        .unwrap();
    store
        .put_doc("counters", "c", &json!({"n": second + 1}))
        .unwrap();
    assert_eq!(count(&store), 1, "two increments, one survives");
}

#[test]
fn has_vector_collection_reads_and_never_creates() {
    let modes = modes();
    for (mode, store) in &modes.stores {
        let before = last_seq(store);
        assert!(!store.has_vector_collection("v").unwrap(), "{mode}");
        assert!(!store.has_vector_collection("v").unwrap(), "{mode}");
        assert_eq!(last_seq(store), before, "{mode}: a lookup commits nothing");
        // A document collection of that name is not a vector collection.
        store.put_doc("v", "doc", &json!(1)).unwrap();
        assert!(!store.has_vector_collection("v").unwrap(), "{mode}");
        make_vectors(store, "v", 2);
        assert!(store.has_vector_collection("v").unwrap(), "{mode}");
        assert!(!store.has_vector_collection("w").unwrap(), "{mode}");
    }
}

/// Every item through `upsert_vectors` on `batch`, and one at a time on
/// `single`: the two collections are the same, graph and entry included.
fn assert_batch_matches_singles(items: &[(String, Vec<f32>, Option<Value>)]) {
    let batch = FerroStore::in_memory();
    let single = FerroStore::in_memory();
    for store in [&batch, &single] {
        store
            .create_vector_collection("v", 8, VectorStorage::F32, VectorMetric::L2, 4, 16, 16)
            .unwrap();
    }
    for (id, values, metadata) in items {
        single
            .upsert_vector("v", id, values, metadata.as_ref())
            .unwrap();
    }
    let borrowed: Vec<(&str, &[f32], Option<&Value>)> = items
        .iter()
        .map(|(id, values, metadata)| (id.as_str(), values.as_slice(), metadata.as_ref()))
        .collect();
    let before = last_seq(&batch);
    let seq = batch.upsert_vectors("v", &borrowed).unwrap();
    assert_eq!(seq, Some(before + 1), "one sequence for the whole batch");
    assert_eq!(last_seq(&batch), before + 1);

    let batch_state = batch.read_state().unwrap();
    let single_state = single.read_state().unwrap();
    let (b, s) = (&batch_state.vectors["v"], &single_state.vectors["v"]);
    assert_eq!(**b, **s, "the same graph as one upsert at a time");
    assert_eq!(
        b.entry,
        vector::canonical_entry(&b.vectors),
        "the canonical entry"
    );
    vector::validate_graph_full("v", b).unwrap();
    for probe in 0..5 {
        let query = values(1000 + probe, 8);
        assert_eq!(
            batch.search_vectors("v", &query, 5, None).unwrap(),
            single.search_vectors("v", &query, 5, None).unwrap()
        );
    }
}

#[test]
fn upsert_vectors_equals_one_upsert_at_a_time() {
    let items: Vec<(String, Vec<f32>, Option<Value>)> = (0..60)
        .map(|i| (format!("id-{i:03}"), values(i, 8), Some(json!({"i": i}))))
        .collect();
    assert_batch_matches_singles(&items);
}

#[test]
fn upsert_vectors_with_a_repeated_id_keeps_the_last() {
    let mut items: Vec<(String, Vec<f32>, Option<Value>)> = (0..20)
        .map(|i| (format!("id-{i}"), values(i, 8), None))
        .collect();
    items.push(("id-3".into(), values(99, 8), Some(json!("last"))));
    items.push(("id-7".into(), values(98, 8), None));
    assert_batch_matches_singles(&items);

    let store = FerroStore::in_memory();
    store
        .create_vector_collection("v", 8, VectorStorage::F32, VectorMetric::L2, 4, 16, 16)
        .unwrap();
    let first = values(1, 8);
    let last = values(2, 8);
    store
        .upsert_vectors(
            "v",
            &[
                ("a", &first, Some(&json!("first"))),
                ("b", &first, None),
                ("a", &last, Some(&json!("last"))),
            ],
        )
        .unwrap();
    let (got, metadata) = store.get_vector("v", "a").unwrap().unwrap();
    assert_eq!(got, last);
    assert_eq!(metadata, Some(json!("last")));
}

#[test]
fn upsert_vectors_is_all_or_nothing_in_every_mode() {
    let modes = modes();
    for (mode, store) in &modes.stores {
        make_vectors(store, "v", 3);
        let before = last_seq(store);
        let good = [1.0, 0.0, 0.0];
        let bad = [1.0, 0.0];
        let nan = [f32::NAN, 0.0, 0.0];
        for items in [
            vec![("a", &good[..], None), ("b", &bad[..], None)],
            vec![("a", &good[..], None), ("b", &nan[..], None)],
            vec![("a", &good[..], None), ("", &good[..], None)],
        ] {
            assert!(store.upsert_vectors("v", &items).is_err(), "{mode}");
            assert_eq!(last_seq(store), before, "{mode}: nothing committed");
            assert_eq!(store.get_vector("v", "a").unwrap(), None, "{mode}");
        }
        assert!(
            store
                .upsert_vectors("missing", &[("a", &good[..], None)])
                .is_err(),
            "{mode}"
        );
        assert_eq!(store.upsert_vectors("v", &[]).unwrap(), None, "{mode}");
        assert!(store.upsert_vectors("missing", &[]).is_err(), "{mode}");
        assert_eq!(
            last_seq(store),
            before,
            "{mode}: an empty batch commits nothing"
        );
        let seq = store
            .upsert_vectors("v", &[("a", &good[..], None), ("b", &good[..], None)])
            .unwrap();
        assert_eq!(seq, Some(before + 1), "{mode}");
        assert!(store.get_vector("v", "b").unwrap().is_some(), "{mode}");
    }
}

#[test]
fn upsert_vectors_on_disk_survives_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("s.fls");
    let items: Vec<(String, Vec<f32>)> = (0..40).map(|i| (format!("v{i}"), values(i, 4))).collect();
    let borrowed: Vec<(&str, &[f32], Option<&Value>)> = items
        .iter()
        .map(|(id, values)| (id.as_str(), values.as_slice(), None))
        .collect();
    let expected = {
        let store = FerroStore::open(&path).unwrap();
        make_vectors(&store, "v", 4);
        store.upsert_vectors("v", &borrowed).unwrap();
        store.read_state().unwrap().vectors["v"].clone()
    };
    let reopened = FerroStore::open(&path).unwrap();
    assert_eq!(*reopened.read_state().unwrap().vectors["v"], *expected);
    reopened.checkpoint().unwrap();
    let again = FerroStore::open(&path).unwrap();
    assert_eq!(*again.read_state().unwrap().vectors["v"], *expected);
}

/// A `WriteThrough` adapter that refuses while `fail` is set.
#[derive(Default)]
struct Flaky {
    fail: AtomicBool,
    sent: Mutex<Vec<u64>>,
}

impl SyncAdapter for Flaky {
    fn name(&self) -> &str {
        "flaky"
    }

    fn on_commit(&self, batch: &SyncBatch) -> Result<(), StoreError> {
        if self.fail.load(Ordering::SeqCst) {
            return Err(StoreError::Backend("adapter is down".into()));
        }
        self.sent.lock().unwrap().push(batch.seq);
        Ok(())
    }
}

#[test]
fn a_durable_commit_whose_batch_was_not_sent_says_so() {
    let dir = tempfile::tempdir().unwrap();
    let stores = [
        ("memory", FerroStore::in_memory()),
        ("disk", FerroStore::open(dir.path().join("s.fls")).unwrap()),
    ];
    for (mode, store) in stores {
        let flaky = Arc::new(Flaky::default());
        store
            .attach_sync_adapter(SyncMode::WriteThrough, Arc::clone(&flaky) as _)
            .unwrap();
        let base = last_seq(&store);
        flaky.fail.store(true, Ordering::SeqCst);

        // Owned notify: this commit's own batch failed.
        let err = store.put_doc("docs", "a", &json!(1)).unwrap_err();
        let StoreError::Undelivered { seq, cause } = &err else {
            panic!("{mode}: {err:?}");
        };
        assert_eq!(*seq, base + 1, "{mode}");
        assert!(matches!(**cause, StoreError::Backend(_)), "{mode}");
        assert_eq!(
            err.to_string(),
            format!(
                "committed at seq {} (durable), not yet delivered to the sync adapter: \
                 backend failure: adapter is down",
                base + 1
            ),
            "{mode}"
        );
        assert_eq!(store.get_doc("docs", "a").unwrap(), Some(json!(1)));

        // Queued: an older batch is still pending, and its drain failed.
        let err = store
            .transaction(|txn| txn.put_doc("docs", "b", &json!(2)))
            .unwrap_err();
        assert!(
            matches!(err, StoreError::Undelivered { seq, .. } if seq == base + 2),
            "{mode}: {err:?}"
        );

        // Not committed: a refused write is not `Undelivered`.
        let err = store.put_doc("__meta", "x", &json!(1)).unwrap_err();
        assert!(matches!(err, StoreError::Backend(_)), "{mode}: {err:?}");
        make_vectors_ignoring_delivery(&store);
        let err = store.upsert_vector("v", "x", &[1.0], None).unwrap_err();
        assert!(matches!(err, StoreError::Backend(_)), "{mode}: {err:?}");
        assert_eq!(last_seq(&store), base + 3, "{mode}");

        flaky.fail.store(false, Ordering::SeqCst);
        assert_eq!(store.sync_flush().unwrap(), base + 3, "{mode}");
        assert_eq!(
            *flaky.sent.lock().unwrap(),
            vec![base + 1, base + 2, base + 3]
        );
    }
}

/// Create collection `v` (dim 3) while the adapter refuses: the create is
/// durable and reports `Undelivered`.
fn make_vectors_ignoring_delivery(store: &FerroStore) {
    match store.create_vector_collection("v", 3, VectorStorage::F32, VectorMetric::L2, 4, 16, 16) {
        Ok(()) | Err(StoreError::Undelivered { .. }) => {}
        Err(err) => panic!("{err:?}"),
    }
    assert!(store.has_vector_collection("v").unwrap());
}

/// A served store whose own `WriteThrough` adapter refuses: the remote
/// client gets the server's `Undelivered`, with its sequence, and the commit
/// is durable on the server.
#[test]
fn a_remote_client_gets_the_servers_undelivered_with_its_sequence() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("s.sock");
    let server =
        FerroStoreServer::bind_unix(&sock, FerroStore::open(dir.path().join("s.fls")).unwrap())
            .unwrap();
    let flaky = Arc::new(Flaky::default());
    server
        .store()
        .attach_sync_adapter(SyncMode::WriteThrough, Arc::clone(&flaky) as _)
        .unwrap();
    flaky.fail.store(true, Ordering::SeqCst);
    let client = FerroStore::connect_uri(&format!("unix://{}", sock.display())).unwrap();
    let base = last_seq(&client);

    let err = client.put_doc("docs", "a", &json!(1)).unwrap_err();
    let StoreError::Undelivered { seq, cause } = &err else {
        panic!("put_doc: {err:?}");
    };
    assert_eq!(*seq, base + 1);
    assert!(matches!(**cause, StoreError::Backend(_)), "{cause:?}");
    assert_eq!(
        err.to_string(),
        format!(
            "committed at seq {} (durable), not yet delivered to the sync adapter: \
             backend failure: adapter is down",
            base + 1
        ),
        "the same message as on the server"
    );
    assert_eq!(client.get_doc("docs", "a").unwrap(), Some(json!(1)));

    let err = client
        .transaction(|txn| txn.put_doc("docs", "b", &json!(2)))
        .unwrap_err();
    assert!(
        matches!(err, StoreError::Undelivered { seq, .. } if seq == base + 2),
        "transaction: {err:?}"
    );
    assert_eq!(client.get_doc("docs", "b").unwrap(), Some(json!(2)));
    assert_eq!(last_seq(&client), base + 2, "each write applied once");

    // Every other server error still arrives as the server's message.
    let err = client.scan_column_table("missing", &[]).unwrap_err();
    assert!(
        matches!(&err, StoreError::Backend(msg) if msg == "backend failure: no column table `missing`"),
        "{err:?}"
    );

    flaky.fail.store(false, Ordering::SeqCst);
    assert_eq!(server.store().sync_flush().unwrap(), base + 2);
}

/// What the refusing adapter below returns: not a `Backend` error.
fn adapter_conflict() -> StoreError {
    StoreError::Conflict {
        id: fl_core::iri::Iri::parse("urn:adapter:peer").unwrap(),
        detail: "the peer holds a newer copy".into(),
    }
}

/// A `WriteThrough` adapter that refuses every batch with
/// [`adapter_conflict`].
struct Refusing;

impl SyncAdapter for Refusing {
    fn name(&self) -> &str {
        "refusing"
    }

    fn on_commit(&self, _batch: &SyncBatch) -> Result<(), StoreError> {
        Err(adapter_conflict())
    }
}

/// The wire form of an `Undelivered` rebuilds the server's error word for
/// word, whatever kind its cause is; a `Backend` cause stays `Backend`.
#[test]
fn an_undelivered_over_the_wire_reads_as_the_servers_error() {
    let causes = [
        StoreError::Backend("adapter is down".into()),
        adapter_conflict(),
        StoreError::Relayed("already relayed once".into()),
    ];
    for cause in causes {
        let backend = matches!(cause, StoreError::Backend(_));
        let server_err = StoreError::Undelivered {
            seq: 7,
            cause: Box::new(cause),
        };
        let client_err = super::server::response_error(&super::server::error_value(1, &server_err));
        assert_eq!(client_err.to_string(), server_err.to_string());
        let StoreError::Undelivered { seq, cause } = &client_err else {
            panic!("{client_err:?}");
        };
        assert_eq!(*seq, 7);
        assert_eq!(
            matches!(**cause, StoreError::Backend(_)),
            backend,
            "{cause:?}"
        );
    }
}

/// One remote write path.
type Write<'a> = Box<dyn Fn() -> Result<(), StoreError> + 'a>;

/// A served store whose adapter refuses, and a client with its own
/// `WriteThrough` adapter: every remote write path returns the server's
/// `Undelivered`, with the server's message, and the client's adapter has
/// still been sent the batch (the commit is durable).
#[test]
fn a_servers_undelivered_still_reaches_the_clients_own_adapter() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("s.sock");
    let server =
        FerroStoreServer::bind_unix(&sock, FerroStore::open(dir.path().join("s.fls")).unwrap())
            .unwrap();
    server
        .store()
        .attach_sync_adapter(SyncMode::WriteThrough, Arc::new(Refusing) as _)
        .unwrap();
    let client = FerroStore::connect_uri(&format!("unix://{}", sock.display())).unwrap();
    let own = Arc::new(Flaky::default());
    client
        .attach_sync_adapter(SyncMode::WriteThrough, Arc::clone(&own) as _)
        .unwrap();
    let base = last_seq(&client);

    // A remote handle has no sequence compare-and-swap (spec §6.1): its
    // writes are a blind `commit_txn` and a validated one.
    let writes: [(&str, Write<'_>); 3] = [
        (
            "put_doc",
            Box::new(|| client.put_doc("docs", "a", &json!(1))),
        ),
        (
            "transaction",
            Box::new(|| client.transaction(|txn| txn.put_doc("docs", "b", &json!(2)))),
        ),
        (
            "read-modify-write transaction",
            Box::new(|| {
                client.transaction(|txn| {
                    let n = txn.get_doc("docs", "b").and_then(|doc| doc.as_u64());
                    txn.put_doc("docs", "c", &json!(n.unwrap_or(0) + 1))
                })
            }),
        ),
    ];
    for (k, (path, write)) in writes.iter().enumerate() {
        let seq = base + 1 + k as u64;
        let err = write().unwrap_err();
        let server_err = StoreError::Undelivered {
            seq,
            cause: Box::new(adapter_conflict()),
        };
        assert_eq!(err.to_string(), server_err.to_string(), "{path}");
        assert!(
            matches!(err, StoreError::Undelivered { seq: at, .. } if at == seq),
            "{path}: {err:?}"
        );
        assert!(
            own.sent.lock().unwrap().contains(&seq),
            "{path}: the client's adapter never got batch {seq}"
        );
    }
    assert_eq!(last_seq(&client), base + 3, "each write applied once");
    assert!(
        !client.has_pending_sync(),
        "the client's own batches were sent"
    );
}

/// A `WriteThrough` publish whose batch the adapter refuses is durable:
/// `publish_mq` returns `Undelivered` and the topic's subscribers still get
/// the message, on every publish of the outage. A served store's own
/// subscribers get a remote client's publish the same way.
#[test]
fn an_undelivered_publish_still_reaches_the_subscribers() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("s.sock");
    let server =
        FerroStoreServer::bind_unix(&sock, FerroStore::open(dir.path().join("s.fls")).unwrap())
            .unwrap();
    let client = FerroStore::connect_uri(&format!("unix://{}", sock.display())).unwrap();
    let local = [
        ("memory", FerroStore::in_memory()),
        ("disk", FerroStore::open(dir.path().join("d.fls")).unwrap()),
    ];
    let cases = local
        .iter()
        .map(|(mode, store)| (*mode, store, store))
        .chain([("served", &client, server.store())]);
    for (mode, publisher, owner) in cases {
        let flaky = Arc::new(Flaky::default());
        owner
            .attach_sync_adapter(SyncMode::WriteThrough, Arc::clone(&flaky) as _)
            .unwrap();
        let rx = owner.subscribe_mq("t");
        flaky.fail.store(true, Ordering::SeqCst);
        for n in 1..=2u64 {
            let base = last_seq(owner);
            let err = publisher.publish_mq("t", "k", &json!(n)).unwrap_err();
            assert!(
                matches!(err, StoreError::Undelivered { seq, .. } if seq == base + 1),
                "{mode}: publish {n}: {err:?}"
            );
            let got = rx
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap_or_else(|_| panic!("{mode}: publish {n} reached no subscriber"));
            assert_eq!((got.seq, got.payload), (n, json!(n)), "{mode}");
        }
        // Each publish reached the subscriber once.
        assert!(rx.try_recv().is_err(), "{mode}");
        flaky.fail.store(false, Ordering::SeqCst);
        owner.sync_flush().unwrap();
        assert_eq!(flaky.sent.lock().unwrap().len(), 2, "{mode}");
        owner.detach_sync_adapter().unwrap();
    }
}
