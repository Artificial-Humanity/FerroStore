//! `SyncAdapter` stress: concurrent multi-thread push/pull, failure recovery, the
//! `LocalCache` and `CloudPreCache` fill paths, and the search-offload fallback.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use fl_core::StoreError;
use fl_store::{
    FerroStore, Fusion, SyncAdapter, SyncBatch, SyncMode, SyncOp, VectorHit, VectorMetric,
    VectorStorage,
};
use serde_json::{Value, json};

fn err(msg: &str) -> StoreError {
    StoreError::Backend(msg.into())
}

/// A remote peer that records what it was given and can be told to misbehave.
#[derive(Default)]
struct Peer {
    commits: Mutex<Vec<SyncBatch>>,
    fail_commit: AtomicBool,
    /// Fail the next N commits, then recover.
    fail_next: AtomicUsize,
    cloud_docs: Mutex<BTreeMap<(String, String), Value>>,
    fetches: AtomicUsize,
    fail_fetch: AtomicBool,
    inbound: Mutex<Vec<SyncBatch>>,
    prewarm: Mutex<Vec<SyncBatch>>,
    fail_pull: AtomicBool,
    vector_hits: Mutex<Option<Vec<VectorHit>>>,
    hybrid_hits: Mutex<Option<Vec<VectorHit>>>,
    fail_search: AtomicBool,
    vector_calls: AtomicUsize,
    hybrid_calls: AtomicUsize,
}

impl Peer {
    fn commits(&self) -> Vec<SyncBatch> {
        self.commits.lock().unwrap().clone()
    }

    /// Atomically consume one pending injected failure, if any remain.
    fn take_flake(&self) -> bool {
        let mut current = self.fail_next.load(Ordering::SeqCst);
        loop {
            let Some(next) = current.checked_sub(1) else {
                return false;
            };
            match self.fail_next.compare_exchange_weak(
                current,
                next,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return true,
                Err(observed) => current = observed,
            }
        }
    }
}

impl SyncAdapter for Peer {
    fn name(&self) -> &str {
        "peer"
    }

    fn on_commit(&self, batch: &SyncBatch) -> Result<(), StoreError> {
        if self.fail_commit.load(Ordering::SeqCst) {
            return Err(err("peer down"));
        }
        if self.take_flake() {
            return Err(err("peer flaked"));
        }
        self.commits.lock().unwrap().push(batch.clone());
        Ok(())
    }

    fn pull_since(&self, after_seq: u64) -> Result<Vec<SyncBatch>, StoreError> {
        if self.fail_pull.load(Ordering::SeqCst) {
            return Err(err("pull refused"));
        }
        Ok(self
            .inbound
            .lock()
            .unwrap()
            .iter()
            .filter(|b| b.seq > after_seq)
            .cloned()
            .collect())
    }

    fn fetch_document(&self, collection: &str, id: &str) -> Result<Option<Value>, StoreError> {
        self.fetches.fetch_add(1, Ordering::SeqCst);
        if self.fail_fetch.load(Ordering::SeqCst) {
            return Err(err("fetch refused"));
        }
        Ok(self
            .cloud_docs
            .lock()
            .unwrap()
            .get(&(collection.to_string(), id.to_string()))
            .cloned())
    }

    fn prewarm(&self, _: &[&str]) -> Result<Vec<SyncBatch>, StoreError> {
        Ok(self.prewarm.lock().unwrap().clone())
    }

    fn search_vectors(
        &self,
        _: &str,
        _: &[f32],
        _: usize,
        _: Option<u32>,
    ) -> Result<Option<Vec<VectorHit>>, StoreError> {
        self.vector_calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_search.load(Ordering::SeqCst) {
            return Err(err("search offload down"));
        }
        Ok(self.vector_hits.lock().unwrap().clone())
    }

    fn search_hybrid(
        &self,
        _: &str,
        _: &str,
        _: &str,
        _: &[f32],
        _: usize,
        _: Fusion,
    ) -> Result<Option<Vec<VectorHit>>, StoreError> {
        self.hybrid_calls.fetch_add(1, Ordering::SeqCst);
        if self.fail_search.load(Ordering::SeqCst) {
            return Err(err("search offload down"));
        }
        Ok(self.hybrid_hits.lock().unwrap().clone())
    }
}

fn attach(mode: SyncMode) -> (Arc<FerroStore>, Arc<Peer>) {
    let peer = Arc::new(Peer::default());
    let store = FerroStore::in_memory()
        .with_sync_adapter(mode, Arc::clone(&peer) as Arc<dyn SyncAdapter>)
        .unwrap();
    (Arc::new(store), peer)
}

fn seqs(batches: &[SyncBatch]) -> Vec<u64> {
    batches.iter().map(|b| b.seq).collect()
}

fn doc_batch(seq: u64, collection: &str, id: &str, doc: Value) -> SyncBatch {
    SyncBatch {
        seq,
        ops: vec![SyncOp::PutDoc {
            collection: collection.into(),
            id: id.into(),
            doc,
        }],
    }
}

const THREADS: usize = 8;
const PER_THREAD: usize = 60;

// ---- concurrent push ------------------------------------------------------------------

#[test]
fn write_through_from_many_threads_delivers_every_commit_once_and_in_order() {
    let (store, peer) = attach(SyncMode::WriteThrough);
    thread::scope(|scope| {
        for t in 0..THREADS {
            let store = &store;
            scope.spawn(move || {
                for i in 0..PER_THREAD {
                    store
                        .put_doc("docs", &format!("t{t}-{i}"), &json!({"t": t, "i": i}))
                        .unwrap();
                }
            });
        }
    });
    store.sync_flush().unwrap();
    let total = (THREADS * PER_THREAD) as u64;
    let commits = peer.commits();
    assert_eq!(commits.len() as u64, total);
    assert_eq!(
        seqs(&commits),
        (1..=total).collect::<Vec<_>>(),
        "gap, dup or reorder"
    );
    assert_eq!(store.synced_seq(), total);
    assert!(!store.has_pending_sync());
    for t in 0..THREADS {
        for i in 0..PER_THREAD {
            assert_eq!(
                store.get_doc("docs", &format!("t{t}-{i}")).unwrap(),
                Some(json!({"t": t, "i": i}))
            );
        }
    }
}

#[test]
fn write_behind_with_concurrent_flushers_sends_each_batch_exactly_once() {
    let (store, peer) = attach(SyncMode::WriteBehind);
    let done = AtomicBool::new(false);
    thread::scope(|scope| {
        let writers: Vec<_> = (0..THREADS)
            .map(|t| {
                let store = &store;
                scope.spawn(move || {
                    for i in 0..PER_THREAD {
                        store
                            .put_doc("docs", &format!("t{t}-{i}"), &json!(i))
                            .unwrap();
                    }
                })
            })
            .collect();
        for _ in 0..3 {
            let store = &store;
            let done = &done;
            scope.spawn(move || {
                while !done.load(Ordering::SeqCst) {
                    // A flusher may lose the race to another; none may duplicate a batch.
                    let _ = store.sync_flush();
                    thread::yield_now();
                }
            });
        }
        // Join without unwrapping: a panicking writer must not leave the flushers spinning
        // (thread::scope would wait on them forever). Stop them first, then assert.
        let results: Vec<_> = writers.into_iter().map(|writer| writer.join()).collect();
        done.store(true, Ordering::SeqCst);
        assert!(
            results.iter().all(Result::is_ok),
            "a writer thread panicked"
        );
    });
    store.sync_flush().unwrap();
    let total = (THREADS * PER_THREAD) as u64;
    assert_eq!(seqs(&peer.commits()), (1..=total).collect::<Vec<_>>());
    assert_eq!(store.synced_seq(), total);
    assert!(!store.has_pending_sync());
}

#[test]
fn a_flaky_peer_never_loses_or_reorders_a_batch() {
    let (store, peer) = attach(SyncMode::WriteThrough);
    let total = (THREADS * PER_THREAD) as u64;
    let errors = AtomicUsize::new(0);
    thread::scope(|scope| {
        for t in 0..THREADS {
            let (store, peer, errors) = (&store, &peer, &errors);
            scope.spawn(move || {
                for i in 0..PER_THREAD {
                    if (t * PER_THREAD + i).is_multiple_of(7) {
                        peer.fail_next.fetch_add(1, Ordering::SeqCst);
                    }
                    if store
                        .put_doc("docs", &format!("t{t}-{i}"), &json!(i))
                        .is_err()
                    {
                        errors.fetch_add(1, Ordering::SeqCst);
                    }
                }
            });
        }
    });
    // The peer recovers; draining must deliver everything the store still owes.
    peer.fail_next.store(0, Ordering::SeqCst);
    store.sync_flush().unwrap();
    assert!(errors.load(Ordering::SeqCst) > 0, "the flake never fired");
    let got = seqs(&peer.commits());
    assert_eq!(
        got,
        (1..=total).collect::<Vec<_>>(),
        "lost, duplicated or reordered"
    );
    assert_eq!(store.synced_seq(), total);
    assert!(!store.has_pending_sync());
}

// ---- failure recovery -----------------------------------------------------------------

#[test]
fn a_failed_commit_stays_pending_blocks_reconfiguration_and_recovers() {
    let (store, peer) = attach(SyncMode::WriteThrough);
    store.put_doc("d", "a", &json!(1)).unwrap();
    peer.fail_commit.store(true, Ordering::SeqCst);
    assert!(store.put_doc("d", "b", &json!(2)).is_err());
    assert!(store.has_pending_sync());
    assert_eq!(store.synced_seq(), 1);
    // The local write is durable even though replication failed.
    assert_eq!(store.get_doc("d", "b").unwrap(), Some(json!(2)));

    let other: Arc<dyn SyncAdapter> = Arc::new(Peer::default());
    assert!(
        store
            .attach_sync_adapter(SyncMode::WriteBehind, Arc::clone(&other))
            .is_err()
    );
    assert!(store.detach_sync_adapter().is_err());
    assert_eq!(
        store.sync_mode(),
        Some(SyncMode::WriteThrough),
        "mode changed on refusal"
    );

    // A later commit queues behind the failed one instead of overtaking it.
    assert!(store.put_doc("d", "c", &json!(3)).is_ok() || store.has_pending_sync());
    assert!(store.sync_flush().is_err(), "still down");
    peer.fail_commit.store(false, Ordering::SeqCst);
    assert_eq!(store.sync_flush().unwrap(), 3);
    assert_eq!(seqs(&peer.commits()), vec![1, 2, 3]);
    assert!(!store.has_pending_sync());

    store.detach_sync_adapter().unwrap();
    assert_eq!(store.sync_mode(), None);
    assert_eq!(store.synced_seq(), 3, "detach keeps synced_seq");
    store
        .attach_sync_adapter(SyncMode::WriteBehind, other)
        .unwrap();
}

#[test]
fn flush_without_an_adapter_is_an_error_and_without_work_is_a_no_op() {
    let bare = FerroStore::in_memory();
    assert!(bare.sync_flush().is_err());
    assert!(bare.sync_pull().is_err());
    assert!(bare.sync_prewarm(&["x"]).is_err());
    assert_eq!(bare.synced_seq(), 0);
    bare.put_doc("d", "a", &json!(1)).unwrap();
    assert!(!bare.has_pending_sync(), "no adapter queues nothing");

    let (store, peer) = attach(SyncMode::WriteBehind);
    assert_eq!(store.sync_flush().unwrap(), 0);
    assert!(peer.commits().is_empty());
}

#[test]
fn write_behind_accumulates_until_flushed_and_a_partial_flush_resumes() {
    let (store, peer) = attach(SyncMode::WriteBehind);
    for i in 0..10 {
        store.put_doc("d", &format!("k{i}"), &json!(i)).unwrap();
    }
    assert!(peer.commits().is_empty());
    assert!(store.has_pending_sync());
    assert_eq!(store.synced_seq(), 0);

    peer.fail_next.store(1, Ordering::SeqCst);
    assert!(store.sync_flush().is_err(), "first batch flaked");
    assert_eq!(store.synced_seq(), 0);
    assert_eq!(store.sync_flush().unwrap(), 10);
    assert_eq!(seqs(&peer.commits()), (1..=10).collect::<Vec<_>>());
}

#[test]
fn a_multi_op_batch_arrives_whole_in_a_single_on_commit() {
    let (store, peer) = attach(SyncMode::WriteThrough);
    store
        .create_vector_collection("v", 2, VectorStorage::F32, VectorMetric::L2, 4, 16, 8)
        .unwrap();
    store
        .upsert_vector("v", "a", &[1.0, 2.0], Some(&json!({"k": 1})))
        .unwrap();
    store.delete_vector("v", "a").unwrap();
    let commits = peer.commits();
    assert_eq!(commits.len(), 3);
    assert!(matches!(
        commits[0].ops[0],
        SyncOp::CreateVectorCollection { .. }
    ));
    assert!(matches!(commits[1].ops[0], SyncOp::UpsertVector { .. }));
    assert!(matches!(commits[2].ops[0], SyncOp::DeleteVector { .. }));
}

#[test]
fn a_rejected_local_write_is_never_replicated() {
    let (store, peer) = attach(SyncMode::WriteThrough);
    assert!(store.upsert_vector("missing", "a", &[1.0], None).is_err());
    assert!(store.append_column_rows("missing", &[vec![]]).is_err());
    assert!(peer.commits().is_empty());
    assert_eq!(store.synced_seq(), 0);
    assert!(!store.has_pending_sync());
}

// ---- pull and prewarm -----------------------------------------------------------------

#[test]
fn pull_applies_inbound_batches_silently_and_advances_the_cursor() {
    let (store, peer) = attach(SyncMode::WriteBehind);
    *peer.inbound.lock().unwrap() = (1..=5)
        .map(|s| doc_batch(s, "d", &format!("k{s}"), json!(s)))
        .collect();
    assert_eq!(store.sync_pull().unwrap(), 5);
    assert_eq!(store.synced_seq(), 5);
    for s in 1..=5u64 {
        assert_eq!(
            store.get_doc("d", &format!("k{s}")).unwrap(),
            Some(json!(s))
        );
    }
    assert!(
        peer.commits().is_empty(),
        "inbound batches were echoed back"
    );
    assert!(!store.has_pending_sync());
    // Idempotent: nothing newer, nothing applied.
    assert_eq!(store.sync_pull().unwrap(), 5);
    // New inbound data continues from the cursor.
    peer.inbound
        .lock()
        .unwrap()
        .push(doc_batch(6, "d", "k6", json!(6)));
    assert_eq!(store.sync_pull().unwrap(), 6);
    // Local commits continue the same sequence.
    store.put_doc("d", "local", &json!("x")).unwrap();
    store.sync_flush().unwrap();
    assert_eq!(seqs(&peer.commits()), vec![7]);
}

#[test]
fn pull_failure_leaves_the_store_untouched() {
    let (store, peer) = attach(SyncMode::WriteBehind);
    *peer.inbound.lock().unwrap() = vec![doc_batch(1, "d", "k", json!(1))];
    peer.fail_pull.store(true, Ordering::SeqCst);
    assert!(store.sync_pull().is_err());
    assert_eq!(store.get_doc("d", "k").unwrap(), None);
    assert_eq!(store.synced_seq(), 0);
    peer.fail_pull.store(false, Ordering::SeqCst);
    assert_eq!(store.sync_pull().unwrap(), 1);
}

#[test]
fn a_bad_inbound_batch_applies_none_of_the_batches() {
    let (store, peer) = attach(SyncMode::WriteBehind);
    *peer.inbound.lock().unwrap() = vec![
        doc_batch(1, "d", "good", json!(1)),
        // Sequence gap: seq 3 cannot follow seq 1.
        doc_batch(3, "d", "gap", json!(3)),
    ];
    assert!(store.sync_pull().is_err());
    assert_eq!(store.get_doc("d", "good").unwrap(), None, "partial apply");
    assert_eq!(store.synced_seq(), 0);

    *peer.inbound.lock().unwrap() = vec![
        doc_batch(1, "d", "good", json!(1)),
        SyncBatch {
            seq: 2,
            ops: vec![SyncOp::UpsertVector {
                collection: "nope".into(),
                id: "x".into(),
                bytes: vec![0; 4],
                metadata: None,
            }],
        },
    ];
    assert!(store.sync_pull().is_err());
    assert_eq!(store.get_doc("d", "good").unwrap(), None);
}

#[test]
fn pull_and_prewarm_are_refused_while_commits_are_pending() {
    let (store, peer) = attach(SyncMode::WriteBehind);
    store.put_doc("d", "a", &json!(1)).unwrap();
    *peer.inbound.lock().unwrap() = vec![doc_batch(2, "d", "b", json!(2))];
    *peer.prewarm.lock().unwrap() = vec![doc_batch(2, "d", "b", json!(2))];
    assert!(store.sync_pull().is_err());
    assert!(store.sync_prewarm(&["d"]).is_err());
    store.sync_flush().unwrap();
    assert_eq!(store.sync_pull().unwrap(), 2);
}

#[test]
fn concurrent_pulls_apply_each_batch_at_most_once() {
    let (store, peer) = attach(SyncMode::WriteBehind);
    *peer.inbound.lock().unwrap() = (1..=50)
        .map(|s| doc_batch(s, "d", &format!("k{s}"), json!(s)))
        .collect();
    thread::scope(|scope| {
        for _ in 0..6 {
            let store = &store;
            scope.spawn(move || {
                // Racing pulls may lose with a sequence error; none may corrupt state.
                let _ = store.sync_pull();
            });
        }
    });
    assert_eq!(store.sync_pull().unwrap(), 50);
    for s in 1..=50u64 {
        assert_eq!(
            store.get_doc("d", &format!("k{s}")).unwrap(),
            Some(json!(s))
        );
    }
}

/// The store and the peer share one sequence space, and `pull_since(after_seq)` is given the
/// store's last committed sequence. A pull that follows `k` local commits therefore gets
/// `k+1..=20`, and the store applies exactly that: the remote documents that land are a
/// SUFFIX of the peer's log, not a prefix. The earlier `k` sequences are local commits.
#[test]
fn a_pull_after_local_commits_lands_only_the_remote_suffix() {
    let (store, peer) = attach(SyncMode::WriteBehind);
    *peer.inbound.lock().unwrap() = (1..=20)
        .map(|s| doc_batch(s, "remote", &format!("r{s}"), json!(s)))
        .collect();
    for i in 0..4 {
        store.put_doc("local", &format!("l{i}"), &json!(i)).unwrap();
    }
    // A pull is refused while commits are pending, so drain them first.
    store.sync_flush().unwrap();
    assert_eq!(store.sync_pull().unwrap(), 20);
    for s in 1..=4u64 {
        assert_eq!(store.get_doc("remote", &format!("r{s}")).unwrap(), None);
    }
    for s in 5..=20u64 {
        assert_eq!(
            store.get_doc("remote", &format!("r{s}")).unwrap(),
            Some(json!(s))
        );
    }
    assert_eq!(seqs(&peer.commits()), vec![1, 2, 3, 4]);
}

#[test]
fn pull_and_local_pushes_interleave_without_losing_either_side() {
    const LOCAL: u64 = 100;
    const REMOTE: u64 = 20;
    let (store, peer) = attach(SyncMode::WriteBehind);
    *peer.inbound.lock().unwrap() = (1..=REMOTE)
        .map(|s| doc_batch(s, "remote", &format!("r{s}"), json!(s)))
        .collect();
    thread::scope(|scope| {
        let store1 = &store;
        scope.spawn(move || {
            for i in 0..LOCAL {
                store1
                    .put_doc("local", &format!("l{i}"), &json!(i))
                    .unwrap();
            }
        });
        let store2 = &store;
        scope.spawn(move || {
            for _ in 0..50 {
                // A pull is refused while a local commit is pending: that is allowed.
                let _ = store2.sync_pull();
                store2.sync_flush().unwrap();
            }
        });
    });
    // Both threads have joined, so this flush settles everything still queued.
    store.sync_flush().unwrap();
    assert!(!store.has_pending_sync());

    // Nothing local was lost.
    for i in 0..LOCAL {
        assert_eq!(
            store.get_doc("local", &format!("l{i}")).unwrap(),
            Some(json!(i))
        );
    }

    // Local commits and pulled batches share one sequence space, so a pull taken after `k`
    // local commits applies `k+1..=REMOTE` (all or nothing) and the landed remote documents
    // are a SUFFIX of the peer's log. A pull taken once the store is already past `REMOTE`
    // applies nothing, which is the `k == REMOTE` case.
    let landed: Vec<bool> = (1..=REMOTE)
        .map(|s| store.get_doc("remote", &format!("r{s}")).unwrap().is_some())
        .collect();
    let k = landed.iter().take_while(|present| !**present).count();
    assert!(landed[k..].iter().all(|present| *present), "{landed:?}");
    for s in (k as u64 + 1)..=REMOTE {
        assert_eq!(
            store.get_doc("remote", &format!("r{s}")).unwrap(),
            Some(json!(s))
        );
    }
    let remote_seqs: BTreeSet<u64> = ((k as u64 + 1)..=REMOTE).collect();
    let total = LOCAL + remote_seqs.len() as u64;
    assert_eq!(store.synced_seq(), total, "{landed:?}");

    // The peer got every local commit exactly once and in order, and never an echo of a
    // pulled batch: the commits are exactly the sequences the pull did not take.
    let local_seqs: Vec<u64> = (1..=total).filter(|s| !remote_seqs.contains(s)).collect();
    let expected: Vec<SyncBatch> = (0..LOCAL)
        .zip(&local_seqs)
        .map(|(i, seq)| doc_batch(*seq, "local", &format!("l{i}"), json!(i)))
        .collect();
    assert_eq!(peer.commits(), expected, "{landed:?}");
}

// ---- local cache and cloud pre-cache --------------------------------------------------

#[test]
fn local_cache_fills_a_miss_once_and_serves_it_locally_afterwards() {
    let (store, peer) = attach(SyncMode::LocalCache);
    peer.cloud_docs
        .lock()
        .unwrap()
        .insert(("d".into(), "a".into()), json!({"from": "cloud"}));
    assert_eq!(
        store.get_doc("d", "a").unwrap(),
        Some(json!({"from": "cloud"}))
    );
    assert_eq!(
        store.get_doc("d", "a").unwrap(),
        Some(json!({"from": "cloud"}))
    );
    assert_eq!(
        peer.fetches.load(Ordering::SeqCst),
        1,
        "second read refetched"
    );
    // A cache fill is not a local write: nothing queues, nothing is echoed.
    assert!(!store.has_pending_sync());
    assert!(peer.commits().is_empty());
}

#[test]
fn local_cache_does_not_cache_a_cloud_miss() {
    let (store, peer) = attach(SyncMode::LocalCache);
    assert_eq!(store.get_doc("d", "ghost").unwrap(), None);
    assert_eq!(store.get_doc("d", "ghost").unwrap(), None);
    assert_eq!(
        peer.fetches.load(Ordering::SeqCst),
        2,
        "a miss must stay retryable"
    );
    peer.cloud_docs
        .lock()
        .unwrap()
        .insert(("d".into(), "ghost".into()), json!(1));
    assert_eq!(store.get_doc("d", "ghost").unwrap(), Some(json!(1)));
}

#[test]
fn local_cache_surfaces_a_cloud_failure_and_recovers() {
    let (store, peer) = attach(SyncMode::LocalCache);
    peer.cloud_docs
        .lock()
        .unwrap()
        .insert(("d".into(), "a".into()), json!(1));
    peer.fail_fetch.store(true, Ordering::SeqCst);
    assert!(store.get_doc("d", "a").is_err());
    // A document already held locally never needs the cloud.
    store.put_doc("d", "held", &json!("local")).unwrap();
    store.sync_flush().unwrap();
    assert_eq!(store.get_doc("d", "held").unwrap(), Some(json!("local")));
    peer.fail_fetch.store(false, Ordering::SeqCst);
    assert_eq!(store.get_doc("d", "a").unwrap(), Some(json!(1)));
}

#[test]
fn only_cache_modes_fetch_on_a_miss() {
    for mode in [SyncMode::WriteThrough, SyncMode::WriteBehind] {
        let (store, peer) = attach(mode);
        peer.cloud_docs
            .lock()
            .unwrap()
            .insert(("d".into(), "a".into()), json!(1));
        assert_eq!(store.get_doc("d", "a").unwrap(), None, "{mode:?} fetched");
        assert_eq!(peer.fetches.load(Ordering::SeqCst), 0, "{mode:?}");
    }
    let (store, peer) = attach(SyncMode::CloudPreCache);
    peer.cloud_docs
        .lock()
        .unwrap()
        .insert(("d".into(), "a".into()), json!(1));
    assert_eq!(store.get_doc("d", "a").unwrap(), Some(json!(1)));
}

#[test]
fn a_local_delete_evicts_and_the_cache_refills_only_once_the_delete_is_flushed() {
    let (store, peer) = attach(SyncMode::LocalCache);
    peer.cloud_docs
        .lock()
        .unwrap()
        .insert(("d".into(), "a".into()), json!("cloud"));
    store.put_doc("d", "a", &json!("local")).unwrap();
    store.delete_doc("d", "a").unwrap();
    // The delete is still queued: a fill now could resurrect data the user removed.
    assert!(store.has_pending_sync());
    assert_eq!(store.get_doc("d", "a").unwrap(), None);
    assert_eq!(
        peer.fetches.load(Ordering::SeqCst),
        0,
        "fetched with a delete pending"
    );
    store.sync_flush().unwrap();
    // The cloud still holds its copy, so the cache fallback now refills the evicted entry.
    assert_eq!(store.get_doc("d", "a").unwrap(), Some(json!("cloud")));
    assert_eq!(peer.fetches.load(Ordering::SeqCst), 1);
}

#[test]
fn a_local_write_is_never_overwritten_by_a_stale_cloud_copy() {
    let (store, peer) = attach(SyncMode::LocalCache);
    peer.cloud_docs
        .lock()
        .unwrap()
        .insert(("d".into(), "a".into()), json!("stale"));
    store.put_doc("d", "a", &json!("fresh")).unwrap();
    assert_eq!(store.get_doc("d", "a").unwrap(), Some(json!("fresh")));
    store.sync_flush().unwrap();
    assert_eq!(store.get_doc("d", "a").unwrap(), Some(json!("fresh")));
    assert_eq!(peer.fetches.load(Ordering::SeqCst), 0);
}

#[test]
fn concurrent_misses_on_one_key_converge_on_the_cloud_value() {
    let (store, peer) = attach(SyncMode::LocalCache);
    peer.cloud_docs
        .lock()
        .unwrap()
        .insert(("d".into(), "hot".into()), json!({"v": 1}));
    thread::scope(|scope| {
        for _ in 0..16 {
            let store = &store;
            scope.spawn(move || {
                for _ in 0..50 {
                    assert_eq!(store.get_doc("d", "hot").unwrap(), Some(json!({"v": 1})));
                }
            });
        }
    });
    let fetches = peer.fetches.load(Ordering::SeqCst);
    assert!(
        (1..=16).contains(&fetches),
        "{fetches} fetches for 16 racing readers"
    );
    assert!(!store.has_pending_sync());
    assert!(peer.commits().is_empty(), "cache fills must not replicate");
}

#[test]
fn concurrent_misses_on_many_keys_fill_each_key() {
    let (store, peer) = attach(SyncMode::CloudPreCache);
    for k in 0..200 {
        peer.cloud_docs
            .lock()
            .unwrap()
            .insert(("d".into(), format!("k{k}")), json!(k));
    }
    thread::scope(|scope| {
        for t in 0..THREADS {
            let store = &store;
            scope.spawn(move || {
                for k in (0..200).skip(t).step_by(1) {
                    assert_eq!(
                        store.get_doc("d", &format!("k{k}")).unwrap(),
                        Some(json!(k))
                    );
                }
            });
        }
    });
    assert!(peer.fetches.load(Ordering::SeqCst) >= 200);
}

#[test]
fn cloud_pre_cache_prewarm_installs_batches_without_echoing_them() {
    let (store, peer) = attach(SyncMode::CloudPreCache);
    *peer.prewarm.lock().unwrap() = (1..=3)
        .map(|s| doc_batch(s, "hot", &format!("k{s}"), json!(s)))
        .collect();
    assert_eq!(store.sync_prewarm(&["hot"]).unwrap(), 3);
    assert_eq!(store.synced_seq(), 3);
    for s in 1..=3u64 {
        assert_eq!(
            store.get_doc("hot", &format!("k{s}")).unwrap(),
            Some(json!(s))
        );
    }
    assert_eq!(
        peer.fetches.load(Ordering::SeqCst),
        0,
        "prewarmed reads hit the cache"
    );
    assert!(peer.commits().is_empty());
    // An empty prewarm is a no-op that reports the cursor.
    *peer.prewarm.lock().unwrap() = Vec::new();
    assert_eq!(store.sync_prewarm(&["hot"]).unwrap(), 3);
}

#[test]
fn a_prewarm_that_does_not_continue_the_sequence_is_refused_whole() {
    let (store, peer) = attach(SyncMode::CloudPreCache);
    store.put_doc("d", "local", &json!(1)).unwrap();
    store.sync_flush().unwrap();
    *peer.prewarm.lock().unwrap() = vec![doc_batch(1, "hot", "stale", json!(0))];
    assert!(store.sync_prewarm(&["hot"]).is_err());
    assert_eq!(store.get_doc("hot", "stale").unwrap(), None);
}

// ---- search offload fallback ----------------------------------------------------------

fn seeded_vector_store(mode: SyncMode) -> (Arc<FerroStore>, Arc<Peer>) {
    let (store, peer) = attach(mode);
    store
        .create_vector_collection("v", 2, VectorStorage::F32, VectorMetric::L2, 8, 32, 16)
        .unwrap();
    for (id, x) in [("near", 0.0f32), ("mid", 5.0), ("far", 50.0)] {
        store.upsert_vector("v", id, &[x, 0.0], None).unwrap();
        store
            .put_doc("docs", id, &json!({"text": format!("{id} lorem ipsum")}))
            .unwrap();
    }
    store.sync_flush().unwrap();
    (store, peer)
}

fn local_ids(store: &FerroStore) -> Vec<String> {
    store
        .search_vectors("v", &[0.0, 0.0], 3, None)
        .unwrap()
        .into_iter()
        .map(|(id, _)| id)
        .collect()
}

#[test]
fn vector_search_uses_the_remote_answer_when_there_is_one() {
    let (store, peer) = seeded_vector_store(SyncMode::WriteBehind);
    *peer.vector_hits.lock().unwrap() = Some(vec![VectorHit {
        id: "remote-only".into(),
        score: 9.5,
    }]);
    let hits = store.search_vectors("v", &[0.0, 0.0], 3, None).unwrap();
    assert_eq!(hits, vec![("remote-only".to_string(), 9.5)]);
}

#[test]
fn vector_search_falls_back_to_local_on_ok_none_and_on_error() {
    let (store, peer) = seeded_vector_store(SyncMode::WriteBehind);
    let baseline = local_ids(&store);
    assert_eq!(baseline.first().map(String::as_str), Some("near"));

    *peer.vector_hits.lock().unwrap() = None;
    assert_eq!(local_ids(&store), baseline, "Ok(None) must search locally");

    peer.fail_search.store(true, Ordering::SeqCst);
    assert_eq!(
        local_ids(&store),
        baseline,
        "Err must search locally, not fail"
    );
    assert!(peer.vector_calls.load(Ordering::SeqCst) >= 3);
}

#[test]
fn an_empty_remote_answer_is_authoritative_not_a_fallback_trigger() {
    let (store, peer) = seeded_vector_store(SyncMode::WriteBehind);
    *peer.vector_hits.lock().unwrap() = Some(Vec::new());
    assert!(
        store
            .search_vectors("v", &[0.0, 0.0], 3, None)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn hybrid_search_offload_falls_back_when_the_remote_adapter_fails() {
    let (store, peer) = seeded_vector_store(SyncMode::CloudPreCache);
    let local = |store: &FerroStore| {
        store
            .search_hybrid("docs", "v", "near", &[0.0, 0.0], 3, Fusion::Rrf { k: 60 })
            .unwrap()
    };
    peer.fail_search.store(true, Ordering::SeqCst);
    let fallback = local(&store);
    assert_eq!(fallback.len(), 3, "{fallback:?}");
    assert_eq!(fallback[0].0, "near");

    peer.fail_search.store(false, Ordering::SeqCst);
    *peer.hybrid_hits.lock().unwrap() = None;
    assert_eq!(
        local(&store),
        fallback,
        "Ok(None) must match the local result"
    );

    *peer.hybrid_hits.lock().unwrap() = Some(vec![VectorHit {
        id: "cloud".into(),
        score: 1.0,
    }]);
    assert_eq!(local(&store), vec![("cloud".to_string(), 1.0)]);
    assert!(peer.hybrid_calls.load(Ordering::SeqCst) >= 3);
}

#[test]
fn search_errors_from_the_store_itself_are_not_masked_by_a_failing_remote() {
    let (store, peer) = seeded_vector_store(SyncMode::WriteBehind);
    peer.fail_search.store(true, Ordering::SeqCst);
    // The remote failed and so does the local fallback: the local error is what surfaces.
    assert!(
        store
            .search_vectors("no-such", &[0.0, 0.0], 3, None)
            .is_err()
    );
    assert!(
        store
            .search_hybrid(
                "docs",
                "no-such",
                "x",
                &[0.0, 0.0],
                3,
                Fusion::Rrf { k: 60 }
            )
            .is_err()
    );
}

#[test]
fn searching_while_the_remote_flaps_and_writers_run_never_errors() {
    let (store, peer) = seeded_vector_store(SyncMode::WriteBehind);
    let stop = AtomicBool::new(false);
    thread::scope(|scope| {
        let (store_w, peer_f) = (&store, &peer);
        let stop_ref = &stop;
        scope.spawn(move || {
            let mut on = false;
            while !stop_ref.load(Ordering::SeqCst) {
                on = !on;
                peer_f.fail_search.store(on, Ordering::SeqCst);
                *peer_f.vector_hits.lock().unwrap() = if on {
                    None
                } else {
                    Some(vec![VectorHit {
                        id: "r".into(),
                        score: 1.0,
                    }])
                };
                thread::yield_now();
            }
        });
        let writers: Vec<_> = (0..4)
            .map(|t| {
                scope.spawn(move || {
                    for i in 0..40 {
                        store_w
                            .upsert_vector("v", &format!("w{t}-{i}"), &[i as f32, t as f32], None)
                            .unwrap();
                    }
                })
            })
            .collect();
        let searchers: Vec<_> = (0..4)
            .map(|_| {
                let store = &store;
                scope.spawn(move || {
                    for _ in 0..100 {
                        let hits = store.search_vectors("v", &[0.0, 0.0], 5, None).unwrap();
                        assert!(!hits.is_empty() && hits.len() <= 5);
                    }
                })
            })
            .collect();
        // Join without unwrapping: a panicking worker must not leave the flapper spinning
        // (thread::scope would wait on it forever). Stop it first, then assert.
        let results: Vec<_> = writers
            .into_iter()
            .chain(searchers)
            .map(|h| h.join())
            .collect();
        stop.store(true, Ordering::SeqCst);
        assert!(
            results.iter().all(Result::is_ok),
            "a worker thread panicked"
        );
    });
}
