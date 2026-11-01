//! Spec §4.4: the auto-checkpoint after a commit, in a lock window of its
//! own once the commit's locks are released.

use super::crash_sim::{IoOp, SimFs};
use super::layout::{SLOT_LEN, SlotRead, Superblock, decode_wal_prefix, parse_superblock};
use super::{
    FerroStore, SyncAdapter, SyncBatch, SyncMode, SyncOp, auto_checkpoint_warnings, lock_mutex,
};
use fl_core::StoreError;
use serde_json::{Value, json};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

const STORE: &str = "/sim/store.fls";
const SNAPSHOT_MAGIC: &[u8] = b"FERRSNAP";

fn open(fs: &SimFs) -> FerroStore {
    FerroStore::open_with(fs, Path::new(STORE)).unwrap()
}

/// The live superblock of the simulated store, as the page cache holds it.
fn live_sb(fs: &SimFs) -> Superblock {
    let image = fs.live_image();
    let bytes = image.file(Path::new(STORE)).unwrap();
    (0..2)
        .filter_map(|index| {
            match parse_superblock(&bytes[index * SLOT_LEN..(index + 1) * SLOT_LEN]) {
                SlotRead::Valid(sb) => Some(sb),
                _ => None,
            }
        })
        .max_by_key(|sb| sb.epoch)
        .unwrap()
}

/// The frames of the live WAL.
fn live_frames(fs: &SimFs) -> usize {
    let sb = live_sb(fs);
    let image = fs.live_image();
    let bytes = image.file(Path::new(STORE)).unwrap();
    let wal = &bytes[sb.wal_offset as usize..(sb.wal_offset + sb.wal_length) as usize];
    decode_wal_prefix(wal).unwrap().0.len()
}

fn is_slot_write(op: &IoOp) -> bool {
    matches!(op, IoOp::Write { offset, len, .. }
        if *len == SLOT_LEN as u64 && (*offset == 0 || *offset == SLOT_LEN as u64))
}

fn text(bytes: usize) -> Value {
    json!("x".repeat(bytes))
}

#[test]
fn each_commit_that_crosses_the_threshold_checkpoints_exactly_once() {
    const FLOOR: u64 = 16 << 10;
    let fs = SimFs::default();
    let store = open(&fs);
    store.set_auto_checkpoint_floor(FLOOR);
    let (mut crossings, mut quiet) = (0, 0);
    for i in 0..60 {
        let before = live_sb(&fs);
        let mark = fs.trace().len();
        store
            .put_doc("bulk", &format!("{i:02}"), &text(1500 + 37 * i))
            .unwrap();
        let trace = fs.trace();
        let frame = trace[mark..]
            .iter()
            .find_map(|op| match op {
                IoOp::Write { len, .. } if !is_slot_write(op) => Some(*len),
                _ => None,
            })
            .unwrap();
        let crossed = before.wal_length + frame > before.snapshot_length.max(FLOOR);
        let snapshots = fs
            .writes_starting_with(SNAPSHOT_MAGIC)
            .into_iter()
            .filter(|at| *at >= mark)
            .count();
        assert_eq!(
            snapshots,
            usize::from(crossed),
            "commit {i}: WAL {} + frame {frame} against max(snapshot {}, {FLOOR})",
            before.wal_length,
            before.snapshot_length
        );
        let after = live_sb(&fs);
        if crossed {
            crossings += 1;
            assert_eq!(after.wal_length, 0, "commit {i}: the WAL restarts");
            assert_eq!(after.snapshot_seq, before.snapshot_seq.max(i as u64 + 1));
        } else {
            quiet += 1;
            assert_eq!(after.wal_length, before.wal_length + frame, "commit {i}");
        }
    }
    // The snapshot grows past the floor, so later crossings follow it.
    assert!(
        crossings >= 3 && quiet >= 20,
        "{crossings} crossings, {quiet} not"
    );
    assert!(live_sb(&fs).snapshot_length > FLOOR);
    let reopened = open(&fs);
    for i in 0..60 {
        assert_eq!(
            reopened.get_doc("bulk", &format!("{i:02}")).unwrap(),
            Some(text(1500 + 37 * i)),
            "doc {i}"
        );
    }
}

#[test]
fn a_reopen_after_an_auto_checkpoint_replays_only_the_wal_tail() {
    // The same writes with auto-checkpoint off, then on.
    let run = |floor: u64| {
        let fs = SimFs::default();
        let store = open(&fs);
        store.set_auto_checkpoint_floor(floor);
        for i in 0..20 {
            store.put_doc("bulk", &format!("{i}"), &text(4000)).unwrap();
        }
        for i in 0..3 {
            store.put_doc("tail", &format!("{i}"), &json!(i)).unwrap();
        }
        drop(store);
        let reopened = open(&fs);
        for i in 0..20 {
            assert_eq!(
                reopened.get_doc("bulk", &format!("{i}")).unwrap(),
                Some(text(4000))
            );
        }
        (live_frames(&fs), live_sb(&fs))
    };
    let (all, _) = run(u64::MAX);
    assert_eq!(
        all, 23,
        "with no checkpoint the reopen replays every commit"
    );
    let (tail, sb) = run(64 << 10);
    assert!(tail < 23, "a checkpoint ran: {tail} frames");
    assert!(
        sb.wal_length <= sb.snapshot_length.max(64 << 10),
        "the WAL a reopen replays is bounded: {sb:?}"
    );
    // 20 x 4 KB crosses 64 KiB once, at commit 16 or 17: what is left is
    // the commits after it.
    assert!(
        (3..=7).contains(&tail),
        "{tail} frames after the checkpoint"
    );
}

/// The writers of one group, queued in the order given (see `tests_group`).
fn one_group<'s, R: Send + 's>(
    store: &'s FerroStore,
    writers: Vec<Box<dyn FnOnce() -> R + Send + 's>>,
) -> Vec<R> {
    std::thread::scope(|scope| {
        let gate = lock_mutex(&store.commit_gate);
        let mut handles = Vec::new();
        for (queued, writer) in writers.into_iter().enumerate() {
            handles.push(scope.spawn(writer));
            store.wait_until_queued(queued + 1);
        }
        drop(gate);
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    })
}

/// Hold the auto-checkpoint at its first op (its `len`, after the group's
/// two syncs and the checkpoint's own lock and refresh) until `members`
/// writers have said they returned, or for 10 s. The flags: the hold
/// happened, and every member had returned when it ended.
fn hold_the_checkpoint_for(
    fs: &SimFs,
    members: usize,
) -> (mpsc::Sender<()>, Arc<AtomicBool>, Arc<AtomicBool>) {
    let (returned_tx, returned_rx) = mpsc::channel::<()>();
    let returned_rx = Mutex::new(returned_rx);
    let members_first = Arc::new(AtomicBool::new(false));
    let held = Arc::new(AtomicBool::new(false));
    let (first, was_held) = (Arc::clone(&members_first), Arc::clone(&held));
    let mut syncs = 0;
    fs.fail_next(move |op| {
        match op {
            IoOp::SyncData { .. } => syncs += 1,
            IoOp::Len { .. } if syncs == 2 && !was_held.swap(true, Ordering::SeqCst) => {
                let rx = returned_rx.lock().unwrap();
                let all = (0..members).all(|_| rx.recv_timeout(Duration::from_secs(10)).is_ok());
                first.store(all, Ordering::SeqCst);
            }
            _ => {}
        }
        false
    });
    (returned_tx, held, members_first)
}

/// A group whose publish crosses the threshold: the second member returns
/// before the leader starts the checkpoint.
#[test]
fn a_crossing_group_acknowledges_its_members_before_the_checkpoint() {
    let fs = SimFs::default();
    let store = open(&fs);
    store.set_auto_checkpoint_floor(1);
    let (returned_tx, held, member_first) = hold_the_checkpoint_for(&fs, 1);
    let mark = fs.trace().len();
    let s = &store;
    let results = one_group(
        s,
        vec![
            Box::new(move || s.put_doc("g", "leader", &text(1000))),
            Box::new(move || {
                let result = s.put_doc("g", "member", &text(1000));
                returned_tx.send(()).unwrap();
                result
            }),
        ],
    );
    for result in results {
        result.unwrap();
    }
    assert!(held.load(Ordering::SeqCst), "the checkpoint never started");
    assert!(
        member_first.load(Ordering::SeqCst),
        "the member was not acknowledged before the checkpoint"
    );
    let snapshots = fs
        .writes_starting_with(SNAPSHOT_MAGIC)
        .into_iter()
        .filter(|at| *at >= mark)
        .count();
    assert_eq!(snapshots, 1, "one checkpoint after the group");
    assert_eq!(live_sb(&fs).wal_length, 0);
    let reopened = open(&fs);
    assert_eq!(reopened.get_doc("g", "member").unwrap(), Some(text(1000)));
    assert_eq!(reopened.get_doc("g", "leader").unwrap(), Some(text(1000)));
}

/// Records every batch it is sent. No I/O: it runs while the checkpoint is
/// held inside the simulated disk.
#[derive(Default)]
struct Recorder {
    sent: Mutex<Vec<u64>>,
}

impl SyncAdapter for Recorder {
    fn name(&self) -> &str {
        "recorder"
    }

    fn on_commit(&self, batch: &SyncBatch) -> Result<(), StoreError> {
        lock_mutex(&self.sent).push(batch.seq);
        Ok(())
    }
}

/// `WriteThrough`: the leader owns the first batch's notify, and every later
/// member drains behind it. Each later member returns, with its own batch
/// sent, while the leader's checkpoint is still held: the leader delivered
/// its own batch and released the locks before it checkpointed.
#[test]
fn a_crossing_write_through_group_delivers_its_members_before_the_checkpoint() {
    const MEMBERS: usize = 3;
    let fs = SimFs::default();
    let store = open(&fs);
    let recorder = Arc::new(Recorder::default());
    store
        .attach_sync_adapter(
            SyncMode::WriteThrough,
            Arc::clone(&recorder) as Arc<dyn SyncAdapter>,
        )
        .unwrap();
    store.set_auto_checkpoint_floor(1);
    let base = store.read_state().unwrap().last_seq;
    let (returned_tx, held, members_first) = hold_the_checkpoint_for(&fs, MEMBERS);
    let mark = fs.trace().len();
    let (s, r) = (&store, &recorder);
    let mut writers: Vec<Box<dyn FnOnce() -> Result<bool, StoreError> + Send + '_>> =
        vec![Box::new(move || {
            s.put_doc("g", "leader", &text(1000))?;
            Ok(lock_mutex(&r.sent).contains(&(base + 1)))
        })];
    for n in 0..MEMBERS {
        let tx = returned_tx.clone();
        writers.push(Box::new(move || {
            let result = s.put_doc("g", &format!("member-{n}"), &text(1000));
            // The member's own sequence: base + 2 + n, in queue order.
            let delivered = lock_mutex(&r.sent).contains(&(base + 2 + n as u64));
            tx.send(()).unwrap();
            result.map(|()| delivered)
        }));
    }
    drop(returned_tx);
    let results = one_group(s, writers);
    for (n, result) in results.into_iter().enumerate() {
        assert!(
            result.unwrap(),
            "writer {n} returned before its batch was sent"
        );
    }
    assert!(held.load(Ordering::SeqCst), "the checkpoint never started");
    assert!(
        members_first.load(Ordering::SeqCst),
        "a member's commit waited for the leader's checkpoint"
    );
    let snapshots = fs
        .writes_starting_with(SNAPSHOT_MAGIC)
        .into_iter()
        .filter(|at| *at >= mark)
        .count();
    assert_eq!(snapshots, 1, "one checkpoint after the group");
    assert_eq!(live_sb(&fs).wal_length, 0);
    let mut sent = lock_mutex(&recorder.sent).clone();
    sent.sort_unstable();
    let want: Vec<u64> = (1..=MEMBERS as u64 + 1).map(|k| base + k).collect();
    assert_eq!(sent, want, "every batch sent once");
    assert!(!store.has_pending_sync());
}

const POISONED: &str = "FerroStore handle poisoned after a failed fsync";

/// A commit whose auto-checkpoint meets a failed `sync_data`: the `n`th of
/// the checkpoint (1: the snapshot, 2: the slot). Returns the commit's
/// result and the next write's.
fn failed_auto_checkpoint_sync(
    n: usize,
) -> (Result<(), StoreError>, Result<(), StoreError>, SimFs) {
    let fs = SimFs::default();
    let store = open(&fs);
    store.put_doc("f", "before", &json!(0)).unwrap();
    store.set_auto_checkpoint_floor(1);
    // The commit's own frame sync and slot sync come first.
    let mut syncs = 0;
    fs.fail_next(move |op| {
        let sync = matches!(op, IoOp::SyncData { .. });
        if syncs < 2 {
            syncs += usize::from(sync);
            return false;
        }
        syncs += usize::from(sync);
        sync && syncs == 2 + n
    });
    let commit = store.put_doc("f", "big", &text(2000));
    let next = store.put_doc("f", "next", &json!(1));
    (commit, next, fs)
}

#[test]
fn a_failed_auto_checkpoint_sync_keeps_the_commit_and_poisons_the_handle() {
    for fault in [1, 2] {
        let (commit, next, fs) = failed_auto_checkpoint_sync(fault);
        assert!(
            commit.is_ok(),
            "{fault:?}: the durable commit failed: {commit:?}"
        );
        let err = next.unwrap_err().to_string();
        assert!(err.contains(POISONED), "{fault:?}: {err}");
        let reopened = open(&fs);
        assert_eq!(
            reopened.get_doc("f", "big").unwrap(),
            Some(text(2000)),
            "{fault:?}"
        );
        assert_eq!(reopened.get_doc("f", "next").unwrap(), None, "{fault:?}");
    }
}

/// Fail the next snapshot write: the first write that follows a `len`
/// (the checkpoint's extent allocation). No commit writes after a `len`.
fn fail_the_next_snapshot_write(fs: &SimFs) {
    let mut after_len = false;
    fs.fail_next(move |op| {
        let fail = after_len && matches!(op, IoOp::Write { .. });
        after_len = matches!(op, IoOp::Len { .. });
        fail
    });
}

/// `flock` acquisitions since `mark` (a lock-event count): one per commit,
/// and one more for each auto-checkpoint it tried.
fn locks_since(fs: &SimFs, mark: usize) -> usize {
    fs.lock_events()[mark..]
        .iter()
        .filter(|(_, op)| matches!(op, IoOp::Lock { .. }))
        .count()
}

/// A failed write is not a failed fsync: the handle is not poisoned. A
/// failure that persists (ENOSPC, an encode error) is tried again only once
/// the WAL has grown by another threshold, and only the first failure of the
/// run is reported; a checkpoint that succeeds ends the run.
#[test]
fn a_failing_auto_checkpoint_write_is_retried_only_after_another_threshold() {
    let fs = SimFs::default();
    let store = open(&fs);
    store.put_doc("f", "before", &json!(0)).unwrap();
    store.set_auto_checkpoint_floor(1);
    let warnings = auto_checkpoint_warnings();
    let threshold = live_sb(&fs).snapshot_length;

    // 1: the commit crosses, its checkpoint fails, the commit stands.
    fail_the_next_snapshot_write(&fs);
    let mark = fs.lock_events().len();
    store.put_doc("f", "big", &text(2000)).unwrap();
    assert_eq!(locks_since(&fs, mark), 2, "the commit and its checkpoint");
    let failed_at = live_sb(&fs).wal_length;
    assert!(failed_at > threshold, "the commit crossed: {failed_at}");
    assert_eq!(
        auto_checkpoint_warnings(),
        warnings + 1,
        "the first failure warns"
    );

    // 2: the next commits are past the threshold, but the WAL has not grown
    // by another threshold since the failure: no attempt, and the armed
    // failure is still armed.
    fail_the_next_snapshot_write(&fs);
    let mut n = 0;
    loop {
        let before = live_sb(&fs).wal_length;
        let mark = fs.lock_events().len();
        store
            .put_doc("f", &format!("small-{n}"), &json!(n))
            .unwrap();
        n += 1;
        let after = live_sb(&fs).wal_length;
        if after - failed_at > threshold {
            // 3: this one has grown past it: the second attempt, which
            // fails again, silently.
            assert!(before - failed_at <= threshold);
            assert_eq!(locks_since(&fs, mark), 2, "commit {n} retried");
            break;
        }
        assert_eq!(locks_since(&fs, mark), 1, "commit {n} retried too early");
    }
    assert!(n > 1, "a small commit came before the retry");
    assert!(
        live_sb(&fs).wal_length > failed_at,
        "the second attempt failed too"
    );
    assert_eq!(
        auto_checkpoint_warnings(),
        warnings + 1,
        "a repeated failure does not warn again"
    );

    // 4: past one more threshold, with the disk well, it checkpoints.
    let second = live_sb(&fs).wal_length;
    while live_sb(&fs).wal_length != 0 {
        assert!(
            live_sb(&fs).wal_length <= second + threshold + 2000,
            "no third attempt"
        );
        store
            .put_doc("f", &format!("small-{n}"), &json!(n))
            .unwrap();
        n += 1;
    }

    // 5: that ended the run: the next failure warns again.
    let threshold = live_sb(&fs).snapshot_length;
    fail_the_next_snapshot_write(&fs);
    while live_sb(&fs).wal_length <= threshold {
        store
            .put_doc("f", &format!("small-{n}"), &text(500))
            .unwrap();
        n += 1;
    }
    assert_eq!(auto_checkpoint_warnings(), warnings + 2, "a new run warns");

    let reopened = open(&fs);
    assert_eq!(reopened.get_doc("f", "big").unwrap(), Some(text(2000)));
    for k in 0..n {
        assert!(
            reopened
                .get_doc("f", &format!("small-{k}"))
                .unwrap()
                .is_some(),
            "small-{k}"
        );
    }
}

/// The other two publishers, a sync-batch apply and a cache fill, also
/// checkpoint once their own locks are released: the WAL restarts, and a
/// reopen reads everything back.
#[test]
fn a_sync_apply_and_a_cache_fill_past_the_threshold_checkpoint() {
    let fs = SimFs::default();
    let store = open(&fs);
    store.set_auto_checkpoint_floor(1);
    let seq = store.read_state().unwrap().last_seq + 1;
    let inbound = vec![SyncBatch {
        seq,
        ops: vec![SyncOp::PutDoc {
            collection: "in".into(),
            id: "big".into(),
            doc: text(2000),
        }],
    }];
    let mark = fs.trace().len();
    assert_eq!(store.apply_inbound(inbound).unwrap(), seq);
    let snapshots = |from: usize| {
        fs.writes_starting_with(SNAPSHOT_MAGIC)
            .into_iter()
            .filter(|at| *at >= from)
            .count()
    };
    assert_eq!(snapshots(mark), 1, "the apply checkpointed");
    assert_eq!(live_sb(&fs).wal_length, 0);

    let mark = fs.trace().len();
    let filled = store
        .install_fetched_doc("fill", "big", text(4000))
        .unwrap();
    assert_eq!(filled, Some(text(4000)));
    assert_eq!(snapshots(mark), 1, "the fill checkpointed");
    assert_eq!(live_sb(&fs).wal_length, 0);

    let reopened = open(&fs);
    assert_eq!(reopened.get_doc("in", "big").unwrap(), Some(text(2000)));
    assert_eq!(reopened.get_doc("fill", "big").unwrap(), Some(text(4000)));
}
