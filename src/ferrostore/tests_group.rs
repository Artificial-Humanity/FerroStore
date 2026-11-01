//! §4.3 of the v2 cutover spec: `fdatasync` and group commit.
//!
//! A test forms a group on purpose by holding the store's commit gate (the
//! leader takes it first) until its writers are queued, in the order it
//! queues them: [`one_group`].

use super::container_io::{ContainerFs, RealFs};
use super::crash_sim::{IoOp, SimFs};
use super::layout::{Op, SLOT_LEN, backend};
use super::{FerroStore, SyncAdapter, SyncBatch, SyncMode, SyncOp, lock_mutex};
use fl_core::StoreError;
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Barrier, Mutex};

const STORE: &str = "/sim/store.fls";

fn open(fs: &SimFs) -> FerroStore {
    FerroStore::open_with(fs, Path::new(STORE)).unwrap()
}

/// `(sync_data, sync_all)` calls in `trace`.
fn syncs(trace: &[IoOp]) -> (usize, usize) {
    let data = trace
        .iter()
        .filter(|op| matches!(op, IoOp::SyncData { .. }))
        .count();
    let all = trace
        .iter()
        .filter(|op| matches!(op, IoOp::SyncAll { .. }))
        .count();
    (data, all)
}

fn is_slot_write(op: &IoOp) -> bool {
    matches!(op, IoOp::Write { offset, len, .. }
        if *len == SLOT_LEN as u64 && (*offset == 0 || *offset == SLOT_LEN as u64))
}

/// The frame writes of each publish in `trace`, in order: the lengths of
/// the data writes before each slot write.
fn publishes(trace: &[IoOp]) -> Vec<Vec<u64>> {
    let mut out = Vec::new();
    let mut frames = Vec::new();
    for op in trace {
        match op {
            op if is_slot_write(op) => out.push(std::mem::take(&mut frames)),
            IoOp::Write { len, .. } => frames.push(*len),
            _ => {}
        }
    }
    out
}

type Writer<'s, R> = Box<dyn FnOnce() -> R + Send + 's>;

/// Run `writers` as one group, queued in the order given: the commit gate
/// is held until each in turn is queued, so the first leads and takes them
/// all. Each writer runs on its own thread; its join result is returned.
fn one_group<'s, R: Send + 's>(
    store: &'s FerroStore,
    writers: Vec<Writer<'s, R>>,
) -> Vec<std::thread::Result<R>> {
    std::thread::scope(|scope| {
        let gate = lock_mutex(&store.commit_gate);
        let mut handles = Vec::new();
        for (queued, writer) in writers.into_iter().enumerate() {
            handles.push(scope.spawn(writer));
            store.wait_until_queued(queued + 1);
        }
        drop(gate);
        handles.into_iter().map(|handle| handle.join()).collect()
    })
}

fn put(store: &FerroStore, collection: &str, id: &str, doc: serde_json::Value) -> u64 {
    store
        .with_write_outcome(|txn| {
            txn.put(collection, id, doc.clone());
            Ok(())
        })
        .unwrap()
        .1
        .unwrap()
}

/// Not timing-asserted: the syncs a commit costs, printed for the PR. A
/// commit never calls `sync_all`; one writer pays two `sync_data`; eight
/// concurrent writers pay at most that, and eight queued as one group pay
/// two between them.
#[test]
fn sync_calls_per_commit_for_the_record() {
    const COMMITS: usize = 40;
    let fs = SimFs::default();
    let store = open(&fs);
    let mark = fs.trace().len();
    for i in 0..COMMITS {
        store.put_doc("one", &format!("k{i}"), &json!(i)).unwrap();
    }
    let (data, all) = syncs(&fs.trace()[mark..]);
    println!(
        "one writer: {COMMITS} commits, {data} sync_data + {all} sync_all = {:.2} syncs/commit",
        (data + all) as f64 / COMMITS as f64
    );
    assert_eq!(all, 0, "a commit called sync_all");
    assert_eq!(
        data,
        2 * COMMITS,
        "one writer: one frame sync and one slot sync"
    );

    const WRITERS: usize = 8;
    const EACH: usize = 20;
    let fs = SimFs::default();
    let store = open(&fs);
    let mark = fs.trace().len();
    let start = Barrier::new(WRITERS);
    std::thread::scope(|scope| {
        for writer in 0..WRITERS {
            let (store, start) = (&store, &start);
            scope.spawn(move || {
                start.wait();
                for i in 0..EACH {
                    store
                        .put_doc("eight", &format!("w{writer}-{i}"), &json!(i))
                        .unwrap();
                }
            });
        }
    });
    let commits = WRITERS * EACH;
    let (data, all) = syncs(&fs.trace()[mark..]);
    println!(
        "{WRITERS} writers: {commits} commits, {data} sync_data + {all} sync_all = {:.2} \
         syncs/commit",
        (data + all) as f64 / commits as f64
    );
    assert_eq!(all, 0, "a concurrent commit called sync_all");
    assert!(
        data <= 2 * commits,
        "{data} sync_data for {commits} commits"
    );

    let mark = fs.trace().len();
    let writers: Vec<Writer<'_, ()>> = (0..WRITERS)
        .map(|writer| {
            let store = &store;
            Box::new(move || {
                store
                    .put_doc("grouped", &format!("w{writer}"), &json!(writer))
                    .unwrap();
            }) as Writer<'_, ()>
        })
        .collect();
    for joined in one_group(&store, writers) {
        joined.unwrap();
    }
    let (data, all) = syncs(&fs.trace()[mark..]);
    println!("{WRITERS} writers queued as one group: {data} sync_data + {all} sync_all");
    assert_eq!(
        (data, all),
        (2, 0),
        "one group: one frame sync and one slot sync"
    );
    assert_eq!(publishes(&fs.trace()[mark..]).len(), 1);
}

/// Records every batch it is sent. With `durable`, also checks that the
/// batch was already durable: the simulated disk's durable image alone
/// opens at its sequence or later.
struct Recorder {
    durable: Option<SimFs>,
    batches: Mutex<Vec<SyncBatch>>,
    early: Mutex<Vec<u64>>,
}

impl Recorder {
    fn new(durable: Option<SimFs>) -> Arc<Self> {
        Arc::new(Self {
            durable,
            batches: Mutex::new(Vec::new()),
            early: Mutex::new(Vec::new()),
        })
    }
}

impl SyncAdapter for Recorder {
    fn name(&self) -> &str {
        "recorder"
    }

    fn on_commit(&self, batch: &SyncBatch) -> Result<(), StoreError> {
        if let Some(fs) = &self.durable {
            let durable = SimFs::new(&fs.durable_image(), None);
            let seq = FerroStore::open_with(&durable, Path::new(STORE))
                .and_then(|store| store.read_state())
                .map_or(0, |state| state.last_seq);
            if seq < batch.seq {
                lock_mutex(&self.early).push(batch.seq);
            }
        }
        lock_mutex(&self.batches).push(batch.clone());
        Ok(())
    }
}

/// The sequences `recorder` has been sent, in order.
fn sent(recorder: &Recorder) -> Vec<u64> {
    lock_mutex(&recorder.batches)
        .iter()
        .map(|batch| batch.seq)
        .collect()
}

/// A store on a fresh simulated disk with a recorder attached in `mode`.
fn recorded_store(mode: SyncMode) -> (SimFs, FerroStore, Arc<Recorder>) {
    let fs = SimFs::default();
    let store = open(&fs);
    let recorder = Recorder::new(Some(fs.clone()));
    store
        .attach_sync_adapter(mode, Arc::clone(&recorder) as Arc<dyn SyncAdapter>)
        .unwrap();
    (fs, store, recorder)
}

/// Three writers queued as one group. Each returns its sequence and the
/// sequences the recorder had been sent when its commit returned.
fn three_member_group(store: &FerroStore, recorder: &Recorder) -> Vec<(u64, Vec<u64>)> {
    let writers: Vec<Writer<'_, (u64, Vec<u64>)>> = (0..3)
        .map(|n| {
            Box::new(move || {
                let seq = put(store, "wt", &format!("{n}"), json!(n));
                (seq, sent(recorder))
            }) as Writer<'_, (u64, Vec<u64>)>
        })
        .collect();
    one_group(store, writers)
        .into_iter()
        .map(|joined| joined.unwrap())
        .collect()
}

#[test]
fn every_write_through_member_of_a_group_returns_with_its_own_batch_sent() {
    let (fs, store, recorder) = recorded_store(SyncMode::WriteThrough);
    let base = store.read_state().unwrap().last_seq;
    let mark = fs.trace().len();
    let members = three_member_group(&store, &recorder);
    assert_eq!(publishes(&fs.trace()[mark..]).len(), 1, "one group");
    for (seq, seen) in &members {
        assert!(
            seen.contains(seq),
            "member {seq} returned before its batch was sent: {seen:?}"
        );
    }
    assert!(!store.has_pending_sync(), "a batch is still queued");
    assert_eq!(sent(&recorder), vec![base + 1, base + 2, base + 3]);
    assert_eq!(store.synced_seq(), base + 3);
    assert_eq!(*lock_mutex(&recorder.early), Vec::<u64>::new());
}

#[test]
fn a_write_through_commit_after_a_group_is_sent_before_it_returns() {
    let (_fs, store, recorder) = recorded_store(SyncMode::WriteThrough);
    let base = store.read_state().unwrap().last_seq;
    three_member_group(&store, &recorder);
    let seq = put(&store, "wt", "later", json!("later"));
    assert_eq!(seq, base + 4);
    assert_eq!(
        sent(&recorder),
        vec![base + 1, base + 2, base + 3, base + 4],
        "the commit after the group returned before its batch was sent"
    );
    assert!(!store.has_pending_sync());
    assert_eq!(store.synced_seq(), base + 4);
}

#[test]
fn write_behind_members_of_a_group_stay_queued_until_a_flush() {
    let (_fs, store, recorder) = recorded_store(SyncMode::WriteBehind);
    let base = store.read_state().unwrap().last_seq;
    for (_, seen) in three_member_group(&store, &recorder) {
        assert_eq!(seen, Vec::<u64>::new(), "WriteBehind sent on commit");
    }
    assert!(store.has_pending_sync());
    assert_eq!(store.sync_flush().unwrap(), base + 3);
    assert_eq!(sent(&recorder), vec![base + 1, base + 2, base + 3]);
    assert!(!store.has_pending_sync());
}

/// 8 threads x 25 commits on one handle, with a `WriteThrough` recorder,
/// and 25 commits on a second handle in parallel.
fn hammer(fs: &dyn ContainerFs, path: &Path, durable: Option<SimFs>) {
    const THREADS: usize = 8;
    const EACH: usize = 25;
    const OTHER: usize = 25;
    let one = FerroStore::open_with(fs, path).unwrap();
    let two = FerroStore::open_with(fs, path).unwrap();
    let base = one.read_state().unwrap().last_seq;
    let recorder = Recorder::new(durable);
    one.attach_sync_adapter(
        SyncMode::WriteThrough,
        Arc::clone(&recorder) as Arc<dyn SyncAdapter>,
    )
    .unwrap();
    let start = Barrier::new(THREADS + 1);
    let (ones, twos) = std::thread::scope(|scope| {
        let ones: Vec<_> = (0..THREADS)
            .map(|t| {
                let (one, start, recorder) = (&one, &start, &recorder);
                scope.spawn(move || {
                    start.wait();
                    (0..EACH)
                        .map(|i| {
                            let id = format!("{t}-{i}");
                            let seq = put(one, "one", &id, json!({"t": t, "i": i}));
                            assert!(
                                sent(recorder).contains(&seq),
                                "WriteThrough commit {seq} returned before its batch was sent"
                            );
                            (seq, id)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let twos = {
            let (two, start) = (&two, &start);
            scope.spawn(move || {
                start.wait();
                (0..OTHER)
                    .map(|i| {
                        let id = format!("x-{i}");
                        (put(two, "two", &id, json!(i)), id)
                    })
                    .collect::<Vec<_>>()
            })
        };
        let ones: Vec<(u64, String)> = ones
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect();
        (ones, twos.join().unwrap())
    });
    let total = (THREADS * EACH + OTHER) as u64;
    let mut seqs: Vec<u64> = ones.iter().chain(&twos).map(|(seq, _)| *seq).collect();
    seqs.sort_unstable();
    assert_eq!(
        seqs,
        (base + 1..=base + total).collect::<Vec<_>>(),
        "every commit took its own sequence, with no gap"
    );

    // The stream: every commit of handle one, once, in sequence order, each
    // its own member's op, all sent before their commits returned: no flush.
    assert!(!one.has_pending_sync());
    let batches = lock_mutex(&recorder.batches).clone();
    let sent: Vec<u64> = batches.iter().map(|batch| batch.seq).collect();
    let mut want: Vec<u64> = ones.iter().map(|(seq, _)| *seq).collect();
    want.sort_unstable();
    assert_eq!(
        sent, want,
        "handle one's stream is its commits, once each, in order"
    );
    let ids: HashMap<u64, &String> = ones.iter().map(|(seq, id)| (*seq, id)).collect();
    for batch in &batches {
        match &batch.ops[..] {
            [SyncOp::PutDoc { collection, id, .. }] => {
                assert_eq!((collection.as_str(), id), ("one", ids[&batch.seq]));
            }
            other => panic!("batch {} is not one put: {other:?}", batch.seq),
        }
    }
    assert_eq!(
        *lock_mutex(&recorder.early),
        Vec::<u64>::new(),
        "on_commit ran before these batches were durable"
    );
    assert_eq!(one.synced_seq(), *want.last().unwrap());

    drop((one, two));
    let again = FerroStore::open_with(fs, path).unwrap();
    assert_eq!(again.read_state().unwrap().last_seq, base + total);
    for t in 0..THREADS {
        for i in 0..EACH {
            assert_eq!(
                again.get_doc("one", &format!("{t}-{i}")).unwrap(),
                Some(json!({"t": t, "i": i}))
            );
        }
    }
    for i in 0..OTHER {
        assert_eq!(
            again.get_doc("two", &format!("x-{i}")).unwrap(),
            Some(json!(i))
        );
    }
}

#[test]
fn concurrent_writers_on_two_handles_keep_every_commit_and_sync_in_order_on_the_sim() {
    let fs = SimFs::default();
    hammer(&fs, Path::new(STORE), Some(fs.clone()));
}

#[test]
fn concurrent_writers_on_two_handles_keep_every_commit_and_sync_in_order_on_a_real_file() {
    let dir = tempfile::tempdir().unwrap();
    hammer(&RealFs, &dir.path().join("hammer.fls"), None);
}

#[test]
fn a_body_that_fails_in_a_group_drops_only_its_own_changes() {
    let fs = SimFs::default();
    let store = open(&fs);
    let base = store.read_state().unwrap().last_seq;
    let mark = fs.trace().len();
    let s = &store;
    let results = one_group(
        s,
        vec![
            Box::new(move || {
                s.with_write_outcome(|txn| {
                    txn.put("g", "a", json!(1));
                    Ok(None)
                })
            }),
            Box::new(move || {
                s.with_write_outcome(|txn| {
                    // Sees the first member's commit.
                    let a = txn.get("g", "a");
                    txn.put("g", "b", json!(2));
                    txn.delete("g", "a");
                    Err(backend(format!("refused after reading {a:?}")))
                })
            }),
            Box::new(move || {
                s.with_write_outcome(|txn| {
                    // Sees the first member's commit and nothing of the second.
                    let seen = (txn.get("g", "a"), txn.get("g", "b"));
                    txn.put("g", "c", json!(3));
                    Ok(Some(seen))
                })
            }),
        ],
    );
    let mut results = results.into_iter().map(|joined| joined.unwrap());
    let (_, first) = results.next().unwrap().unwrap();
    let failed = results.next().unwrap().unwrap_err();
    let (seen, third) = results.next().unwrap().unwrap();
    assert_eq!(first, Some(base + 1));
    assert_eq!(
        failed.to_string(),
        "backend failure: refused after reading Some(Number(1))",
        "the failing member gets its own error"
    );
    assert_eq!(seen, Some((Some(json!(1)), None)));
    assert_eq!(third, Some(base + 2), "the failed member took no sequence");
    assert_eq!(publishes(&fs.trace()[mark..]).len(), 1, "one group");
    for store in [&store, &open(&fs)] {
        assert_eq!(store.get_doc("g", "a").unwrap(), Some(json!(1)));
        assert_eq!(store.get_doc("g", "b").unwrap(), None);
        assert_eq!(store.get_doc("g", "c").unwrap(), Some(json!(3)));
        assert_eq!(store.read_state().unwrap().last_seq, base + 2);
    }
}

#[test]
fn a_body_that_panics_in_a_group_unwinds_only_its_own_thread() {
    let fs = SimFs::default();
    let store = open(&fs);
    let s = &store;
    let results = one_group(
        s,
        vec![
            Box::new(move || s.put_doc("p", "a", &json!(1)).unwrap()),
            Box::new(move || {
                s.with_write(|txn| {
                    txn.put("p", "b", json!(2));
                    if txn.get("p", "a").is_some() {
                        panic!("a body panics");
                    }
                    Ok(())
                })
                .unwrap()
            }),
            Box::new(move || s.put_doc("p", "c", &json!(3)).unwrap()),
        ],
    );
    assert!(results[0].is_ok());
    assert!(results[1].is_err(), "the panic reached its own thread");
    assert!(results[2].is_ok());
    let again = open(&fs);
    assert_eq!(again.get_doc("p", "a").unwrap(), Some(json!(1)));
    assert_eq!(again.get_doc("p", "b").unwrap(), None);
    assert_eq!(again.get_doc("p", "c").unwrap(), Some(json!(3)));
    // The group handed on: the store still takes commits.
    store.put_doc("p", "d", &json!(4)).unwrap();
}

#[test]
fn a_compare_and_swap_whose_sequence_an_earlier_member_took_loses() {
    let fs = SimFs::default();
    let store = open(&fs);
    let base = store.read_state().unwrap().last_seq;
    let mark = fs.trace().len();
    let op = |id: &str| Op::Put {
        collection: "cas".to_string(),
        id: id.to_string(),
        doc: json!(id),
    };
    let s = &store;
    let (x, y, z) = (op("x"), op("y"), op("z"));
    let results = one_group(
        s,
        vec![
            Box::new(move || s.commit_if_seq(base, &[x])),
            // Expected `base`, but the first member moved it to `base + 1`.
            Box::new(move || s.commit_if_seq(base, &[y])),
            // Expects the group's running sequence: wins.
            Box::new(move || s.commit_if_seq(base + 1, &[z])),
        ],
    );
    let won: Vec<bool> = results
        .into_iter()
        .map(|joined| joined.unwrap().unwrap())
        .collect();
    assert_eq!(won, vec![true, false, true]);
    assert_eq!(publishes(&fs.trace()[mark..]).len(), 1, "one group");
    for store in [&store, &open(&fs)] {
        assert_eq!(store.get_doc("cas", "x").unwrap(), Some(json!("x")));
        assert_eq!(store.get_doc("cas", "y").unwrap(), None, "the loser wrote");
        assert_eq!(store.get_doc("cas", "z").unwrap(), Some(json!("z")));
        assert_eq!(store.read_state().unwrap().last_seq, base + 2);
    }
}

#[test]
fn sixty_five_queued_writers_take_two_groups() {
    let fs = SimFs::default();
    let store = open(&fs);
    let base = store.read_state().unwrap().last_seq;
    let mark = fs.trace().len();
    let writers: Vec<Writer<'_, u64>> = (0..65)
        .map(|n| {
            let store = &store;
            Box::new(move || put(store, "many", &format!("{n}"), json!(n))) as Writer<'_, u64>
        })
        .collect();
    let mut seqs: Vec<u64> = one_group(&store, writers)
        .into_iter()
        .map(|joined| joined.unwrap())
        .collect();
    let groups = publishes(&fs.trace()[mark..]);
    let sizes: Vec<usize> = groups.iter().map(Vec::len).collect();
    assert_eq!(sizes, vec![64, 1], "at most 64 members in a group");
    seqs.sort_unstable();
    assert_eq!(seqs, (base + 1..=base + 65).collect::<Vec<_>>());
    let again = open(&fs);
    for n in 0..65 {
        assert_eq!(
            again.get_doc("many", &format!("{n}")).unwrap(),
            Some(json!(n))
        );
    }
}

#[test]
fn one_publish_carries_at_most_4_mib_of_frames_and_a_larger_frame_goes_alone() {
    const MIB: usize = 1 << 20;
    let fs = SimFs::default();
    let store = open(&fs);
    // 8 MiB of WAL would pass the auto-checkpoint threshold (spec §4.4,
    // `tests_checkpoint`): off here, so the trace is the publishes alone.
    store.set_auto_checkpoint_floor(u64::MAX);
    let mark = fs.trace().len();
    let sized = |bytes: usize| json!("x".repeat(bytes));
    let s = &store;
    let results = one_group(
        s,
        vec![
            Box::new(move || s.put_doc("big", "a", &sized(3 * MIB / 2))),
            Box::new(move || s.put_doc("big", "b", &sized(3 * MIB / 2))),
            // a + b + c would pass 4 MiB: a and b are published first.
            Box::new(move || s.put_doc("big", "c", &sized(5 * MIB))),
            // c alone is past the cap: d goes in the next publish.
            Box::new(move || s.put_doc("big", "d", &json!("small"))),
        ],
    );
    for joined in results {
        joined.unwrap().unwrap();
    }
    let sizes: Vec<Vec<u64>> = publishes(&fs.trace()[mark..]);
    let counts: Vec<usize> = sizes.iter().map(Vec::len).collect();
    assert_eq!(counts, vec![2, 1, 1], "publishes: {sizes:?}");
    assert!(sizes[0].iter().sum::<u64>() <= (4 * MIB) as u64);
    assert!(sizes[1][0] > (4 * MIB) as u64);
    let (data, all) = syncs(&fs.trace()[mark..]);
    assert_eq!((data, all), (6, 0), "two syncs per publish");
    let again = open(&fs);
    for (id, len) in [("a", 3 * MIB / 2), ("b", 3 * MIB / 2), ("c", 5 * MIB)] {
        assert_eq!(again.get_doc("big", id).unwrap(), Some(sized(len)));
    }
    assert_eq!(again.get_doc("big", "d").unwrap(), Some(json!("small")));
}

const POISONED: &str =
    "FerroStore handle poisoned after a failed fsync at /sim/store.fls: reopen the store";

/// Every write and checkpoint on `store` is refused as poisoned, before any
/// write reaches the disk. Reads still answer, from the cache.
fn assert_poisoned(fs: &SimFs, store: &FerroStore) {
    let mark = fs.trace().len();
    let refused = [
        store.put_doc("f", "refused", &json!(1)).unwrap_err(),
        store
            .with_write(|txn| {
                txn.put("f", "refused", json!(2));
                Ok(())
            })
            .unwrap_err(),
        store.checkpoint().unwrap_err(),
    ];
    for err in refused {
        assert!(err.to_string().contains(POISONED), "{err}");
    }
    assert!(
        !fs.trace()[mark..]
            .iter()
            .any(|op| matches!(op, IoOp::Write { .. } | IoOp::SyncData { .. })),
        "a poisoned handle wrote"
    );
    assert_eq!(store.get_doc("f", "before").unwrap(), Some(json!(0)));
    assert_eq!(store.get_doc("f", "refused").unwrap(), None);
}

/// A fresh open of `fs` works: it reads `before` and takes a commit and a
/// checkpoint. Returns the docs `0..n` it reads and its sequence.
fn reopen_works(fs: &SimFs, n: usize) -> (Vec<Option<serde_json::Value>>, u64) {
    let reopened = open(fs);
    let docs = (0..n)
        .map(|i| reopened.get_doc("f", &format!("{i}")).unwrap())
        .collect::<Vec<_>>();
    let seq = reopened.read_state().unwrap().last_seq;
    assert_eq!(reopened.get_doc("f", "before").unwrap(), Some(json!(0)));
    reopened.put_doc("f", "after", &json!(9)).unwrap();
    reopened.checkpoint().unwrap();
    let after = open(fs);
    assert_eq!(after.read_state().unwrap().last_seq, seq + 1);
    assert_eq!(after.get_doc("f", "after").unwrap(), Some(json!(9)));
    (docs, seq)
}

/// A group whose publish fails at the `nth` `sync_data` after the group
/// starts: every member gets the error and the handle is poisoned. Returns
/// what a reopen reads, and the sequences before the group and after it.
fn failed_group(nth: usize) -> (Vec<Option<serde_json::Value>>, u64, u64) {
    let fs = SimFs::default();
    let store = open(&fs);
    store.put_doc("f", "before", &json!(0)).unwrap();
    let base = store.read_state().unwrap().last_seq;
    let mut seen = 0;
    fs.fail_next(move |op| {
        if matches!(op, IoOp::SyncData { .. }) {
            seen += 1;
        }
        seen == nth
    });
    let s = &store;
    let results = one_group(
        s,
        (0..3)
            .map(|n| {
                Box::new(move || s.put_doc("f", &format!("{n}"), &json!(n)))
                    as Writer<'_, Result<(), StoreError>>
            })
            .collect(),
    );
    for joined in results {
        let err = joined.unwrap().unwrap_err();
        assert!(err.to_string().contains("simulated I/O error"), "{err}");
    }
    assert_poisoned(&fs, &store);
    // The handle still reads the last state it acknowledged.
    assert_eq!(store.read_state().unwrap().last_seq, base);
    let (docs, seq) = reopen_works(&fs, 3);
    let none = vec![None, None, None];
    let all = vec![Some(json!(0)), Some(json!(1)), Some(json!(2))];
    assert!(
        (docs == none && seq == base) || (docs == all && seq == base + 3),
        "a reopen read part of the group: {docs:?} at {seq}"
    );
    (docs, base, seq)
}

#[test]
fn a_group_whose_frame_sync_fails_fails_every_member_and_leaves_the_state_before_it() {
    let (docs, base, seq) = failed_group(1);
    assert_eq!(docs, vec![None, None, None]);
    assert_eq!(seq, base);
}

#[test]
fn a_group_whose_slot_sync_fails_fails_every_member_and_leaves_all_of_it_or_none() {
    // The slot write was issued: the file names the whole group (the
    // simulated page cache kept it), though no member was acknowledged.
    let (docs, base, seq) = failed_group(2);
    assert_eq!(docs, vec![Some(json!(0)), Some(json!(1)), Some(json!(2))]);
    assert_eq!(seq, base + 3);
}

/// A checkpoint whose `nth` `sync_data` fails (1: the snapshot, 2: the slot)
/// errors and poisons the handle; a reopen reads every commit.
fn failed_checkpoint(nth: usize) {
    let fs = SimFs::default();
    let store = open(&fs);
    store.put_doc("f", "before", &json!(0)).unwrap();
    for i in 0..3 {
        store.put_doc("f", &format!("{i}"), &json!(i)).unwrap();
    }
    let base = store.read_state().unwrap().last_seq;
    let mut seen = 0;
    fs.fail_next(move |op| {
        if matches!(op, IoOp::SyncData { .. }) {
            seen += 1;
        }
        seen == nth
    });
    let err = store.checkpoint().unwrap_err();
    assert!(err.to_string().contains("simulated I/O error"), "{err}");
    assert_poisoned(&fs, &store);
    let (docs, seq) = reopen_works(&fs, 3);
    assert_eq!(docs, vec![Some(json!(0)), Some(json!(1)), Some(json!(2))]);
    assert_eq!(seq, base);
}

#[test]
fn a_checkpoint_whose_snapshot_sync_fails_poisons_the_handle() {
    failed_checkpoint(1);
}

#[test]
fn a_checkpoint_whose_slot_sync_fails_poisons_the_handle() {
    failed_checkpoint(2);
}

/// A lock-free read that loads the slots after `publish`'s slot write and
/// before that slot's sync fails reads the unacknowledged epoch for its own
/// call, but the handle keeps none of it: once poisoned, every read serves
/// the last state whose syncs all returned. `publish` writes `f/new = 1` at
/// the next sequence, with one frame sync and one slot sync.
fn a_read_between_a_failed_slot_sync_is_not_kept(
    publish: impl FnOnce(&FerroStore, u64) -> Result<(), StoreError>,
) {
    let fs = SimFs::default();
    let store = Arc::new(open(&fs));
    store.put_doc("f", "before", &json!(0)).unwrap();
    let base = store.read_state().unwrap().last_seq;
    let mid = Arc::new(Mutex::new(None));
    {
        let (reader, sim, mid) = (Arc::clone(&store), fs.clone(), Arc::clone(&mid));
        let mut seen = 0;
        fs.before_next_sync(
            move |op| {
                if matches!(op, IoOp::SyncData { .. }) {
                    seen += 1;
                }
                // The publish's second sync: the one after its slot write.
                seen == 2
            },
            move || {
                let read = std::thread::spawn(move || {
                    reader
                        .read_state()
                        .map(|state| (state.last_seq, state.docs.get("f", "new").cloned()))
                        .map_err(|err| err.to_string())
                })
                .join()
                .unwrap();
                *lock_mutex(&mid) = Some(read);
                sim.fail_next(|op| matches!(op, IoOp::SyncData { .. }));
            },
        );
    }
    let err = publish(&store, base + 1).unwrap_err();
    assert!(err.to_string().contains("simulated I/O error"), "{err}");
    let read = lock_mutex(&mid)
        .take()
        .expect("the read between did not run");
    assert_eq!(
        read,
        Ok((base + 1, Some(json!(1)))),
        "the read between did not load the slot the publish wrote"
    );
    assert_eq!(
        store.read_state().unwrap().last_seq,
        base,
        "a poisoned handle served an epoch whose sync never returned"
    );
    assert_eq!(store.get_doc("f", "new").unwrap(), None);
    assert_poisoned(&fs, &store);
}

#[test]
fn a_read_between_the_slot_write_and_its_failed_sync_is_not_kept_once_poisoned() {
    a_read_between_a_failed_slot_sync_is_not_kept(|store, _| store.put_doc("f", "new", &json!(1)));
}

/// The same through an inbound sync batch, which publishes on its own path
/// (not in a group).
#[test]
fn a_read_between_a_sync_batch_slot_write_and_its_failed_sync_is_not_kept_once_poisoned() {
    a_read_between_a_failed_slot_sync_is_not_kept(|store, seq| {
        store
            .apply_inbound(vec![SyncBatch {
                seq,
                ops: vec![SyncOp::PutDoc {
                    collection: "f".into(),
                    id: "new".into(),
                    doc: json!(1),
                }],
            }])
            .map(drop)
    });
}

/// A `LocalCache` adapter that holds every document it is asked for.
struct Peer;

impl SyncAdapter for Peer {
    fn name(&self) -> &str {
        "peer"
    }

    fn on_commit(&self, _batch: &SyncBatch) -> Result<(), StoreError> {
        Ok(())
    }

    fn fetch_document(&self, _: &str, id: &str) -> Result<Option<serde_json::Value>, StoreError> {
        Ok(Some(json!({ "peer": id })))
    }
}

/// Reads go on on a poisoned handle: a cache miss in `LocalCache` returns the
/// adapter's document, and stores nothing.
#[test]
fn a_poisoned_handle_returns_a_fetched_document_without_storing_it() {
    let fs = SimFs::default();
    let store = open(&fs);
    store.put_doc("f", "before", &json!(0)).unwrap();
    fs.fail_next(|op| matches!(op, IoOp::SyncData { .. }));
    let err = store.put_doc("f", "lost", &json!(1)).unwrap_err();
    assert!(err.to_string().contains("simulated I/O error"), "{err}");
    store
        .attach_sync_adapter(SyncMode::LocalCache, Arc::new(Peer))
        .unwrap();
    let mark = fs.trace().len();
    for _ in 0..2 {
        assert_eq!(
            store.get_doc("f", "miss").unwrap(),
            Some(json!({ "peer": "miss" }))
        );
    }
    assert!(
        !fs.trace()[mark..]
            .iter()
            .any(|op| matches!(op, IoOp::Write { .. } | IoOp::SyncData { .. })),
        "a poisoned handle stored the fetched document"
    );
    assert_eq!(store.get_doc("f", "before").unwrap(), Some(json!(0)));
    assert!(!store.has_pending_sync());
    store.detach_sync_adapter().unwrap();
    assert_eq!(
        store.get_doc("f", "miss").unwrap(),
        None,
        "the fill was kept"
    );
    assert_poisoned(&fs, &store);
}
