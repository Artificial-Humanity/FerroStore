//! §4.1 of the v2 cutover spec: the epoch-cached handle and WAL tail replay.
//!
//! The simulated disk (`crash_sim`) records every container operation, so
//! these tests count what a read or a write actually reads.

use super::FerroStore;
use super::container_io::ContainerFs;
use super::crash_sim::{IoOp, SimFs};
use super::layout::{SLOT_LEN, SlotRead, Superblock, parse_superblock};
use serde_json::json;
use std::path::Path;

const STORE: &str = "/sim/store.fls";

fn open(fs: &SimFs) -> FerroStore {
    FerroStore::open_with(fs, Path::new(STORE)).unwrap()
}

/// The live superblock, read from the simulated disk without an operation.
fn live_sb(fs: &SimFs) -> Superblock {
    let image = fs.live_image();
    let bytes = image.file(Path::new(STORE)).unwrap();
    let mut best: Option<Superblock> = None;
    for index in 0..2 {
        let slot = &bytes[index * SLOT_LEN..(index + 1) * SLOT_LEN];
        if let SlotRead::Valid(sb) = parse_superblock(slot)
            && best.is_none_or(|held| sb.epoch > held.epoch)
        {
            best = Some(sb);
        }
    }
    best.unwrap()
}

/// The live bytes of superblock slot `index`, read without an operation.
fn slot_bytes(fs: &SimFs, index: usize) -> Vec<u8> {
    let image = fs.live_image();
    let bytes = image.file(Path::new(STORE)).unwrap();
    bytes[index * SLOT_LEN..(index + 1) * SLOT_LEN].to_vec()
}

fn slot_epoch(fs: &SimFs, index: usize) -> Option<u64> {
    match parse_superblock(&slot_bytes(fs, index)) {
        SlotRead::Valid(sb) => Some(sb.epoch),
        _ => None,
    }
}

/// Every `(offset, len)` read since trace position `mark`.
fn reads_since(fs: &SimFs, mark: usize) -> Vec<(u64, u64)> {
    fs.trace()[mark..]
        .iter()
        .filter_map(|op| match op {
            IoOp::Read { offset, len, .. } => Some((*offset, *len)),
            _ => None,
        })
        .collect()
}

fn is_slot_read(&(offset, len): &(u64, u64)) -> bool {
    len == SLOT_LEN as u64 && (offset == 0 || offset == SLOT_LEN as u64)
}

/// The reads since `mark` that are not one of the two superblock slots.
fn extent_reads_since(fs: &SimFs, mark: usize) -> Vec<(u64, u64)> {
    reads_since(fs, mark)
        .into_iter()
        .filter(|read| !is_slot_read(read))
        .collect()
}

fn bytes(reads: &[(u64, u64)]) -> u64 {
    reads.iter().map(|(_, len)| len).sum()
}

/// Not timing-asserted: the bytes a read costs, printed for the PR, with the
/// floor each case must meet.
#[test]
fn read_cost_in_bytes_for_the_record() {
    let fs = SimFs::default();
    let writer = open(&fs);
    writer
        .with_write(|txn| {
            for i in 0..5000 {
                txn.put(
                    "docs",
                    format!("doc-{i:05}"),
                    json!({"i": i, "title": format!("document number {i}")}),
                );
            }
            Ok(())
        })
        .unwrap();
    writer.checkpoint().unwrap();
    writer
        .put_doc("docs", "wal-only", &json!("in the wal"))
        .unwrap();
    let sb = live_sb(&fs);
    let reader = open(&fs);
    assert_eq!(
        reader.get_doc("docs", "doc-00042").unwrap().unwrap()["i"],
        42
    );

    let mark = fs.trace().len();
    for _ in 0..1000 {
        assert!(reader.get_doc("docs", "doc-04999").unwrap().is_some());
    }
    let idle = reads_since(&fs, mark);
    let idle_bytes = bytes(&idle);

    let before = live_sb(&fs);
    writer.put_doc("docs", "small", &json!(1)).unwrap();
    let after = live_sb(&fs);
    let mark = fs.trace().len();
    assert_eq!(reader.get_doc("docs", "small").unwrap(), Some(json!(1)));
    let one = reads_since(&fs, mark);
    let one_bytes = bytes(&one);
    let frame = after.wal_length - before.wal_length;
    println!(
        "§4.1 read cost: snapshot {} B, wal {} B (5001 docs); \
         1000 reads, no commit: {idle_bytes} B in {} reads; \
         1 read after 1 small commit ({frame} B frame): {one_bytes} B in {} reads",
        sb.snapshot_length,
        sb.wal_length,
        idle.len(),
        one.len(),
    );
    assert_eq!(
        idle_bytes,
        1000 * 2 * SLOT_LEN as u64,
        "idle reads, first few: {:?}",
        &idle[..idle.len().min(8)]
    );
    assert_eq!(one_bytes, 2 * SLOT_LEN as u64 + frame, "tail read: {one:?}");
}

// ---- the refresh branches -------------------------------------------------------

#[test]
fn a_read_with_no_new_commit_reads_only_the_two_slots() {
    let fs = SimFs::default();
    let store = open(&fs);
    store.put_doc("c", "a", &json!(1)).unwrap();
    store.checkpoint().unwrap();
    store.put_doc("c", "b", &json!(2)).unwrap();
    // The first read after this handle's own commit is already cached: the
    // commit installed its state.
    let mark = fs.trace().len();
    for _ in 0..50 {
        assert_eq!(store.get_doc("c", "b").unwrap(), Some(json!(2)));
        assert_eq!(store.get_doc("c", "a").unwrap(), Some(json!(1)));
    }
    for op in &fs.trace()[mark..] {
        match op {
            IoOp::Read { offset, len, .. } => assert!(
                is_slot_read(&(*offset, *len)),
                "a read with no new commit read an extent: {op}"
            ),
            // The §5 swap check: `stat` of the path.
            IoOp::Len { .. } | IoOp::StatId { .. } => {}
            other => panic!("a read did more than read the slots: {other}"),
        }
    }
    assert_eq!(bytes(&reads_since(&fs, mark)), 100 * 2 * SLOT_LEN as u64);
}

#[test]
fn another_handles_commits_are_read_as_their_tail_only() {
    let fs = SimFs::default();
    let a = open(&fs);
    let b = open(&fs);
    a.put_doc("c", "base", &json!(0)).unwrap();
    assert_eq!(b.get_doc("c", "base").unwrap(), Some(json!(0)));
    for i in 0..25 {
        let before = live_sb(&fs);
        b.put_doc("c", &format!("k{i}"), &json!(i)).unwrap();
        let after = live_sb(&fs);
        let mark = fs.trace().len();
        assert_eq!(a.get_doc("c", &format!("k{i}")).unwrap(), Some(json!(i)));
        assert_eq!(
            extent_reads_since(&fs, mark),
            vec![(
                before.wal_offset + before.wal_length,
                after.wal_length - before.wal_length
            )],
            "commit {i}: the reader read more than the new frame"
        );
        // Nothing new: the next read is the cached state.
        let mark = fs.trace().len();
        assert_eq!(a.get_doc("c", "base").unwrap(), Some(json!(0)));
        assert!(extent_reads_since(&fs, mark).is_empty());
    }
    let state = a.read_state().unwrap();
    assert_eq!(state.last_seq, 26);
    assert_eq!(state.docs.in_collection("c").count(), 26);
}

#[test]
fn a_write_reads_only_the_slots_and_the_tail_another_handle_added() {
    let fs = SimFs::default();
    let a = open(&fs);
    let b = open(&fs);
    a.put_doc("c", "a1", &json!(1)).unwrap();
    assert_eq!(b.get_doc("c", "a1").unwrap(), Some(json!(1)));
    let seen_by_b = live_sb(&fs);
    // `a` wrote last: its next write starts from its own installed state.
    let mark = fs.trace().len();
    a.put_doc("c", "a2", &json!(2)).unwrap();
    assert!(
        extent_reads_since(&fs, mark).is_empty(),
        "a write after this handle's own commit re-read the file"
    );
    // `b` last loaded after `a1`: its write reads only the `a2` frame.
    let before = live_sb(&fs);
    let mark = fs.trace().len();
    b.with_write(|txn| {
        assert_eq!(
            txn.get("c", "a2"),
            Some(json!(2)),
            "the write missed a commit"
        );
        txn.put("c", "b1", json!(3));
        Ok(())
    })
    .unwrap();
    assert_eq!(
        extent_reads_since(&fs, mark),
        vec![(
            seen_by_b.wal_offset + seen_by_b.wal_length,
            before.wal_length - seen_by_b.wal_length
        )],
        "the write read more than the frame it had not seen"
    );
    assert_eq!(a.get_doc("c", "b1").unwrap(), Some(json!(3)));
    assert_eq!(a.get_doc("c", "a2").unwrap(), Some(json!(2)));
    assert_eq!(b.read_state().unwrap().last_seq, 3);
}

#[test]
fn another_handles_checkpoint_is_a_full_reload() {
    let fs = SimFs::default();
    let a = open(&fs);
    let b = open(&fs);
    for i in 0..5 {
        a.put_doc("c", &format!("k{i}"), &json!(i)).unwrap();
    }
    assert_eq!(b.get_doc("c", "k4").unwrap(), Some(json!(4)));
    b.delete_doc("c", "k0").unwrap();
    b.checkpoint().unwrap();
    let sb = live_sb(&fs);
    assert_eq!(sb.wal_length, 0);
    let mark = fs.trace().len();
    let state = a.read_state().unwrap();
    assert_eq!(
        extent_reads_since(&fs, mark),
        vec![(sb.snapshot_offset, sb.snapshot_length)],
        "a checkpoint is a new snapshot extent: a full load"
    );
    assert_eq!(state.last_seq, 6);
    assert_eq!(state.docs.get("c", "k0"), None);
    assert_eq!(state.docs.get("c", "k4"), Some(&json!(4)));
    // After the reload `a` tails again.
    b.put_doc("c", "after", &json!("x")).unwrap();
    let mark = fs.trace().len();
    assert_eq!(a.get_doc("c", "after").unwrap(), Some(json!("x")));
    let reads = extent_reads_since(&fs, mark);
    assert_eq!(reads.len(), 1);
    assert_eq!(reads[0].0, sb.wal_offset);
}

#[test]
fn a_checkpoint_installs_its_result_without_a_reread() {
    let fs = SimFs::default();
    let store = open(&fs);
    store.put_doc("c", "a", &json!(1)).unwrap();
    store.checkpoint().unwrap();
    let mark = fs.trace().len();
    assert_eq!(store.get_doc("c", "a").unwrap(), Some(json!(1)));
    assert!(extent_reads_since(&fs, mark).is_empty());
}

#[test]
fn a_cache_read_through_another_file_is_not_trusted() {
    use super::Inner;
    use super::crash_sim::Image;
    use std::collections::BTreeMap;
    // Two inodes with the same bytes: the same epoch in both.
    let seed = SimFs::default();
    let first = open(&seed);
    first.put_doc("c", "a", &json!(1)).unwrap();
    first.checkpoint().unwrap();
    first.put_doc("c", "b", &json!(2)).unwrap();
    drop(first);
    let bytes = seed.live_image().file(Path::new(STORE)).unwrap().to_vec();
    let copy = Path::new("/sim/copy.fls");
    let image = Image {
        names: BTreeMap::from([(Path::new(STORE).to_path_buf(), 0), (copy.to_path_buf(), 1)]),
        inodes: vec![bytes.clone(), bytes],
    };
    let fs = SimFs::new(&image, None);
    let store = open(&fs);
    assert_eq!(store.get_doc("c", "b").unwrap(), Some(json!(2)));
    // A swap under the path, as §5's rename makes: the path now names
    // another file at the same epoch. The cache was filled through inode 0,
    // and the read's `stat` check reopens the path on inode 1.
    fs.rename(copy, Path::new(STORE)).unwrap();
    let Inner::Disk(disk) = &store.inner else {
        panic!("a disk store");
    };
    assert_eq!(
        crate::ferrostore::lock_mutex(&disk.cache)
            .as_ref()
            .unwrap()
            .file_id
            .1,
        0
    );
    let sb = live_sb(&fs);
    let mark = fs.trace().len();
    assert_eq!(store.get_doc("c", "b").unwrap(), Some(json!(2)));
    let loaded: Vec<(usize, u64, u64)> = fs.trace()[mark..]
        .iter()
        .filter_map(|op| match op {
            IoOp::Read { ino, offset, len } if !is_slot_read(&(*offset, *len)) => {
                Some((*ino, *offset, *len))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        loaded,
        vec![
            (1, sb.snapshot_offset, sb.snapshot_length),
            (1, sb.wal_offset, sb.wal_length)
        ],
        "same epoch through another file must be a full load"
    );
    // The cache now belongs to inode 1: the next read is a hit.
    let mark = fs.trace().len();
    assert_eq!(store.get_doc("c", "a").unwrap(), Some(json!(1)));
    assert!(extent_reads_since(&fs, mark).is_empty());
}

/// The interleaving a lock-free reader can meet: it reads slot 0 (`E - 1`);
/// another handle publishes `E + 1` into slot 0 and starts `E + 2` into slot
/// 1; the reader's slot 1 read is torn, so the slots it read name `E - 1`.
/// The reader's cache already holds `E`, its own commit: it must return that,
/// not go back to `E - 1`, and must not load the file for it.
#[test]
fn a_reader_whose_second_slot_read_is_torn_keeps_its_newer_cached_state() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    let fs = SimFs::default();
    let reader = open(&fs);
    reader.put_doc("c", "k", &json!(1)).unwrap();
    let held = live_sb(&fs);
    assert_eq!(slot_epoch(&fs, 1), Some(held.epoch), "E is in slot 1");
    assert_eq!(
        slot_epoch(&fs, 0),
        Some(held.epoch - 1),
        "E - 1 is in slot 0"
    );
    let want = reader.read_state().unwrap().last_seq;
    let other = Arc::new(open(&fs));
    let fired = Arc::new(AtomicBool::new(false));
    let finished = Arc::new(std::sync::Mutex::new(Vec::new()));
    {
        let (sim, other, fired, finished) = (
            fs.clone(),
            Arc::clone(&other),
            Arc::clone(&fired),
            Arc::clone(&finished),
        );
        fs.after_next_read(
            |op| matches!(op, IoOp::Read { offset: 0, len, .. } if *len == SLOT_LEN as u64),
            move || {
                let old = slot_bytes(&sim, 1);
                other.put_doc("c", "k", &json!(2)).unwrap();
                assert_eq!(slot_epoch(&sim, 0), Some(held.epoch + 1));
                other.put_doc("c", "k", &json!(3)).unwrap();
                let new = slot_bytes(&sim, 1);
                assert_eq!(slot_epoch(&sim, 1), Some(held.epoch + 2));
                // The slot 1 write, partly done: what a concurrent read sees.
                // The new header and checksum, the old fields after them.
                let mut torn = old.clone();
                torn[..24].copy_from_slice(&new[..24]);
                assert!(matches!(parse_superblock(&torn), SlotRead::Invalid));
                let raw = sim.open_rw(Path::new(STORE)).unwrap();
                raw.write_at(&torn, SLOT_LEN as u64).unwrap();
                *finished.lock().unwrap() = new;
                fired.store(true, Ordering::SeqCst);
            },
        );
    }
    let mark = fs.trace().len();
    let state = reader.read_state().unwrap();
    assert!(
        fired.load(Ordering::SeqCst),
        "the interleaving did not happen"
    );
    assert_eq!(
        state.last_seq, want,
        "the reader went back past the commit it made"
    );
    assert_eq!(state.docs.get("c", "k"), Some(&json!(1)));
    assert_eq!(
        extent_reads_since(&fs, mark),
        Vec::<(u64, u64)>::new(),
        "the reader loaded the older state"
    );
    // The slot write completes: the next read moves on to `E + 2`.
    let new = finished.lock().unwrap().clone();
    fs.open_rw(Path::new(STORE))
        .unwrap()
        .write_at(&new, SLOT_LEN as u64)
        .unwrap();
    let state = reader.read_state().unwrap();
    assert_eq!(state.last_seq, want + 2);
    assert_eq!(state.docs.get("c", "k"), Some(&json!(3)));
}

// ---- concurrency on real files ---------------------------------------------------

fn counter(store: &FerroStore) -> u64 {
    store
        .get_doc("counter", "n")
        .unwrap()
        .and_then(|v| v.as_u64())
        .unwrap_or(0)
}

/// One read-modify-write commit: a lost update shows as a short counter.
fn bump(store: &FerroStore, key: &str, value: u64) -> u64 {
    let (_, seq) = store
        .with_write_outcome(|txn| {
            let n = txn
                .get("counter", "n")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            txn.put("counter", "n", json!(n + 1));
            txn.put("hammer", key, json!(value));
            Ok(())
        })
        .unwrap();
    seq.expect("a commit")
}

#[test]
fn writes_through_two_handles_lose_no_update() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("two.fls");
    let a = FerroStore::open(&path).unwrap();
    let b = FerroStore::open(&path).unwrap();
    let start = a.read_state().unwrap().last_seq;
    let mut seqs = Vec::new();
    for i in 0..40u64 {
        let store = if i % 2 == 0 { &a } else { &b };
        seqs.push(bump(store, &format!("alt-{i}"), i));
    }
    assert_eq!(counter(&a), 40);
    assert_eq!(counter(&b), 40);

    const PER_THREAD: u64 = 60;
    let threaded = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for (h, store) in [&a, &b].into_iter().enumerate() {
            for t in 0..2 {
                let threaded = &threaded;
                scope.spawn(move || {
                    for i in 0..PER_THREAD {
                        let seq = bump(store, &format!("{h}-{t}-{i}"), i);
                        threaded.lock().unwrap().push(seq);
                    }
                });
            }
        }
    });
    seqs.extend(threaded.into_inner().unwrap());
    let total = 40 + 4 * PER_THREAD;
    seqs.sort_unstable();
    assert_eq!(
        seqs,
        (start + 1..=start + total).collect::<Vec<_>>(),
        "every commit took the next sequence number, none twice"
    );
    let fresh = FerroStore::open(&path).unwrap();
    for store in [&a, &b, &fresh] {
        assert_eq!(counter(store), total, "an update was lost");
        let state = store.read_state().unwrap();
        assert_eq!(state.last_seq, start + total);
        for h in 0..2 {
            for t in 0..2 {
                for i in 0..PER_THREAD {
                    assert_eq!(
                        state.docs.get("hammer", &format!("{h}-{t}-{i}")),
                        Some(&json!(i))
                    );
                }
            }
        }
        for i in 0..40u64 {
            assert_eq!(
                state.docs.get("hammer", &format!("alt-{i}")),
                Some(&json!(i))
            );
        }
    }
}

#[test]
fn a_reader_never_sees_a_mixed_state_while_writers_commit() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pairs.fls");
    let writer = FerroStore::open(&path).unwrap();
    let other = FerroStore::open(&path).unwrap();
    let done = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let readers: Vec<_> = [&writer, &other, &other]
            .into_iter()
            .map(|store| {
                let done = &done;
                scope.spawn(move || {
                    let mut last = (0u64, 0u64);
                    let mut reads = 0u64;
                    while !done.load(Ordering::SeqCst) || reads < 10 {
                        let seen = store
                            .with_read(|txn| {
                                let a = txn.get("pair", "a").and_then(|v| v.as_u64());
                                let b = txn.get("pair", "b").and_then(|v| v.as_u64());
                                assert_eq!(
                                    a,
                                    b,
                                    "a mixed state at seq {}",
                                    txn.local_base().unwrap().last_seq
                                );
                                Ok((txn.local_base().unwrap().last_seq, a.unwrap_or(0)))
                            })
                            .unwrap();
                        assert!(seen.0 >= last.0, "last_seq went back: {last:?} -> {seen:?}");
                        assert!(seen.1 >= last.1, "a pair went back: {last:?} -> {seen:?}");
                        last = seen;
                        reads += 1;
                    }
                    last
                })
            })
            .collect();
        for i in 1..=300u64 {
            // Alternate the writing handle; checkpoint now and then, so
            // readers also take the full-load branch under load.
            let store = if i % 3 == 0 { &other } else { &writer };
            store
                .with_write(|txn| {
                    txn.put("pair", "a", json!(i));
                    txn.put("pair", "b", json!(i));
                    Ok(())
                })
                .unwrap();
            if i % 50 == 0 {
                store.checkpoint().unwrap();
            }
        }
        done.store(true, Ordering::SeqCst);
        for reader in readers {
            let (seq, _) = reader.join().unwrap();
            assert!(seq > 0);
        }
    });
    for store in [&writer, &other] {
        assert_eq!(store.get_doc("pair", "a").unwrap(), Some(json!(300)));
        assert_eq!(store.get_doc("pair", "b").unwrap(), Some(json!(300)));
    }
}
