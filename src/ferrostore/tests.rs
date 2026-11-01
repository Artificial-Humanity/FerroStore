use super::layout::{self, Op, Superblock};
use super::tests_auth::psk_auth;
use super::{
    AttemptRollup, ColumnDef, ColumnFilter, ColumnPredicate, ColumnType, ColumnValue, FerroStore,
    FerroStoreServer, Fusion, GateRunRollup, KnowledgeHit, VectorMetric, VectorStorage,
    default_socket_path_for_store, file_uri_path, latest_superblock, write_slot,
};
use fl_core::StoreError;
use fl_core::ids::{GateId, ProjectId, RecordId, seq_iri};
use fl_core::log::{Attempt, AttemptStatus, GateRun, PathsTouched};
use fl_core::store::Ledger;
use fl_core::verdict::Verdict;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::mpsc::RecvTimeoutError;
use std::thread;
use std::time::Duration;

fn temp_fls() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.fls");
    (dir, path)
}

fn slots(path: &Path) -> [Option<Superblock>; 2] {
    let file = File::open(path).unwrap();
    let mut out = [None, None];
    for (index, slot) in out.iter_mut().enumerate() {
        let mut buf = [0u8; layout::SLOT_LEN];
        file.read_exact_at(&mut buf, index as u64 * layout::SLOT_LEN as u64)
            .unwrap();
        *slot = layout::parse_superblock(&buf).valid();
    }
    out
}

fn live(path: &Path) -> Superblock {
    let file = File::open(path).unwrap();
    latest_superblock(&file).unwrap().1
}

fn extend_wal(path: &Path, extra: &[u8]) {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let (index, sb) = latest_superblock(&file).unwrap();
    let at = sb.wal_offset + sb.wal_length;
    file.write_all_at(extra, at).unwrap();
    let mut next = sb;
    next.epoch += 1;
    next.wal_length += u64::try_from(extra.len()).unwrap();
    write_slot(&file, index ^ 1, &next).unwrap();
    file.sync_all().unwrap();
}

fn corrupt_live_crc(path: &Path) {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let (index, _) = latest_superblock(&file).unwrap();
    let offset = index as u64 * layout::SLOT_LEN as u64;
    let mut buf = [0u8; layout::SLOT_LEN];
    file.read_exact_at(&mut buf, offset).unwrap();
    buf[20] ^= 0xff;
    file.write_all_at(&buf, offset).unwrap();
    file.sync_all().unwrap();
}

fn seed(path: &Path) {
    let store = FerroStore::open(path).unwrap();
    store
        .put_doc("docs", "kept", &json!({ "ok": true }))
        .unwrap();
}

/// The table CRC equals the bit-at-a-time one at every length 0..=64 (each tail length after
/// whole 8-byte steps) and every offset into the buffer, and on a large buffer.
#[test]
fn the_table_crc32_equals_the_bitwise_crc32() {
    let mut x = 0x2545_f491_4f6c_dd1d_u64;
    let data: Vec<u8> = (0..1 << 16)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 32) as u8
        })
        .collect();
    for start in 0..8 {
        for len in 0..=64 {
            let part = &data[start..start + len];
            assert_eq!(
                layout::crc32(part),
                layout::crc32_bitwise(part),
                "start {start}, len {len}"
            );
        }
    }
    assert_eq!(layout::crc32(&data), layout::crc32_bitwise(&data));
    assert_eq!(layout::crc32(&[]), 0);
}

#[test]
fn crc32_iso_hdlc_and_superblock_round_trip() {
    assert_eq!(layout::crc32(b"123456789"), 0xCBF4_3926);
    let sb = Superblock {
        epoch: 7,
        snapshot_offset: 8192,
        snapshot_length: 3,
        wal_offset: 9000,
        wal_length: 4,
        snapshot_seq: 5,
    };
    let bytes = layout::encode_superblock(&sb);
    assert_eq!(layout::parse_superblock(&bytes).valid(), Some(sb));
    let mut torn = bytes;
    torn[20] ^= 0xff;
    assert!(layout::parse_superblock(&torn).valid().is_none());
    torn = bytes;
    torn[64] = 1;
    assert!(layout::parse_superblock(&torn).valid().is_none());
}

#[test]
fn superblock_double_buffer_falls_back_when_the_live_slot_is_torn() {
    let (_dir, path) = temp_fls();
    {
        let store = FerroStore::open(&path).unwrap();
        drop(store);
    }
    let fresh = slots(&path);
    assert_eq!(fresh[0].unwrap().epoch, 1);
    assert!(fresh[1].is_none());
    assert_eq!(fresh[0].unwrap().wal_length, 0);
    assert_eq!(fresh[0].unwrap().snapshot_offset, 8192);
    assert_eq!(fresh[0].unwrap().snapshot_seq, 0);

    {
        let store = FerroStore::open(&path).unwrap();
        store.put_doc("docs", "a", &json!({ "n": 1 })).unwrap();
        store.put_doc("docs", "b", &json!({ "n": 2 })).unwrap();
        store.put_doc("docs", "c", &json!({ "n": 3 })).unwrap();
    }
    let both = slots(&path);
    assert_eq!(both[0].map(|sb| sb.epoch), Some(3));
    assert_eq!(both[1].map(|sb| sb.epoch), Some(4));
    assert!(both[1].unwrap().wal_length > both[0].unwrap().wal_length);

    {
        let store = FerroStore::open(&path).unwrap();
        assert!(store.get_doc("docs", "c").unwrap().is_some());
    }
    corrupt_live_crc(&path);
    let after = slots(&path);
    assert_eq!(after[0].map(|sb| sb.epoch), Some(3));
    assert!(after[1].is_none());

    let store = FerroStore::open(&path).unwrap();
    assert_eq!(store.get_doc("docs", "a").unwrap().unwrap()["n"], 1);
    assert_eq!(store.get_doc("docs", "b").unwrap().unwrap()["n"], 2);
    assert!(store.get_doc("docs", "c").unwrap().is_none());
}

#[test]
fn wal_truncated_tail_is_ignored_and_a_bad_crc_in_the_middle_is_corrupt() {
    let (_dir, path) = temp_fls();
    seed(&path);
    let mut header_only = Vec::new();
    header_only.extend_from_slice(&64u32.to_le_bytes());
    header_only.extend_from_slice(&0u32.to_le_bytes());
    extend_wal(&path, &header_only);
    {
        let store = FerroStore::open(&path).unwrap();
        assert_eq!(store.get_doc("docs", "kept").unwrap().unwrap()["ok"], true);
    }

    let (_dir, path) = temp_fls();
    seed(&path);
    let mut short = Vec::new();
    short.extend_from_slice(&32u32.to_le_bytes());
    short.extend_from_slice(&0u32.to_le_bytes());
    short.extend_from_slice(&[1, 2, 3]);
    extend_wal(&path, &short);
    {
        let store = FerroStore::open(&path).unwrap();
        assert!(store.get_doc("docs", "kept").unwrap().is_some());
    }

    let (_dir, path) = temp_fls();
    seed(&path);
    extend_wal(&path, &[0x01, 0x02, 0x03]);
    {
        let store = FerroStore::open(&path).unwrap();
        assert!(store.get_doc("docs", "kept").unwrap().is_some());
    }

    let (_dir, path) = temp_fls();
    seed(&path);
    let mut ghost = layout::encode_frame(
        2,
        &[Op::Put {
            collection: "docs".into(),
            id: "ghost".into(),
            doc: json!({ "no": true }),
        }],
    )
    .unwrap();
    ghost[4] ^= 0xff;
    extend_wal(&path, &ghost);
    {
        let store = FerroStore::open(&path).unwrap();
        assert!(store.get_doc("docs", "kept").unwrap().is_some());
        assert!(store.get_doc("docs", "ghost").unwrap().is_none());
    }

    let (_dir, path) = temp_fls();
    seed(&path);
    let mut middle = layout::encode_frame(
        2,
        &[Op::Put {
            collection: "docs".into(),
            id: "middle".into(),
            doc: json!({ "n": 2 }),
        }],
    )
    .unwrap();
    middle[4] ^= 0xff;
    let tail = layout::encode_frame(
        3,
        &[Op::Put {
            collection: "docs".into(),
            id: "tail".into(),
            doc: json!({ "n": 3 }),
        }],
    )
    .unwrap();
    middle.extend(tail);
    extend_wal(&path, &middle);
    let err = match FerroStore::open(&path) {
        Ok(_) => panic!("open of a corrupt wal succeeded"),
        Err(err) => err,
    };
    let msg = err.to_string();
    assert!(
        matches!(err, StoreError::Backend(_)) && msg.contains("corrupt"),
        "{msg}"
    );
}

#[test]
fn zeroed_header_recovers_and_a_longer_headerless_file_does_not() {
    assert_eq!(layout::HEADER_BYTES, 8192);
    let (_dir, path) = temp_fls();
    {
        let file = File::create(&path).unwrap();
        file.set_len(8192).unwrap();
        file.sync_all().unwrap();
    }
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 8192);
    assert!(slots(&path).iter().all(Option::is_none));
    {
        let store = FerroStore::open(&path).unwrap();
        store.put_doc("docs", "a", &json!({ "n": 1 })).unwrap();
    }
    let store = FerroStore::open(&path).unwrap();
    assert_eq!(store.get_doc("docs", "a").unwrap().unwrap()["n"], 1);

    let (_dir, path) = temp_fls();
    let marker = vec![0xA5u8; 8193];
    {
        let file = File::create(&path).unwrap();
        file.write_all_at(&marker, 0).unwrap();
        file.sync_all().unwrap();
    }
    let err = match FerroStore::open(&path) {
        Ok(_) => panic!("open rewrote a long headerless file into a store"),
        Err(err) => err,
    };
    let msg = err.to_string();
    assert!(
        matches!(err, StoreError::Backend(_)) && msg.contains("no valid FerroStore superblock"),
        "{msg}"
    );
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 8193);
    let file = File::open(&path).unwrap();
    let mut buf = vec![0u8; marker.len()];
    file.read_exact_at(&mut buf, 0).unwrap();
    assert_eq!(buf, marker);
}

#[test]
fn checkpoint_compacts_the_wal_and_later_puts_survive_reopen() {
    let (_dir, path) = temp_fls();
    {
        let store = FerroStore::open(&path).unwrap();
        store.put_doc("docs", "a", &json!({ "n": 1 })).unwrap();
        store.put_doc("docs", "b", &json!({ "n": 2 })).unwrap();
        store.checkpoint().unwrap();
    }
    let sb = live(&path);
    assert_eq!(sb.wal_length, 0);
    assert_eq!(sb.snapshot_seq, 2);
    assert!(sb.snapshot_length > 0);

    {
        let store = FerroStore::open(&path).unwrap();
        assert_eq!(store.get_doc("docs", "a").unwrap().unwrap()["n"], 1);
        assert_eq!(store.get_doc("docs", "b").unwrap().unwrap()["n"], 2);
        store.put_doc("docs", "c", &json!({ "n": 3 })).unwrap();
    }
    let store = FerroStore::open(&path).unwrap();
    assert_eq!(store.get_doc("docs", "a").unwrap().unwrap()["n"], 1);
    assert_eq!(store.get_doc("docs", "c").unwrap().unwrap()["n"], 3);
    assert!(live(&path).wal_length > 0);
}

#[test]
fn unpublished_snapshot_past_the_live_extent_keeps_committed_documents() {
    let (_dir, path) = temp_fls();
    {
        let store = FerroStore::open(&path).unwrap();
        store.put_doc("docs", "kept", &json!({ "n": 1 })).unwrap();
        store.put_doc("docs", "also", &json!({ "n": 2 })).unwrap();
    }
    let before = live(&path);
    let mut docs = BTreeMap::new();
    docs.insert(("docs".into(), "kept".into()), json!({ "n": 99 }));
    docs.insert(("docs".into(), "ghost".into()), json!({ "n": 7 }));
    let blob = layout::encode_snapshot(99, &docs.into()).unwrap();
    let snap = layout::decode_snapshot(&blob).unwrap();
    assert_eq!(snap.seq, 99);
    assert_eq!(snap.docs.get("docs", "ghost").unwrap()["n"], 7);

    {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let (_, sb) = latest_superblock(&file).unwrap();
        let live_end = sb
            .snapshot_offset
            .saturating_add(sb.snapshot_length)
            .max(sb.wal_offset.saturating_add(sb.wal_length))
            .max(layout::HEADER_BYTES);
        let offset = file.metadata().unwrap().len().max(live_end);
        assert!(offset >= live_end);
        file.write_all_at(&blob, offset).unwrap();
        file.sync_all().unwrap();
        assert!(file.metadata().unwrap().len() > live_end);
    }
    assert_eq!(live(&path), before);

    let store = FerroStore::open(&path).unwrap();
    assert_eq!(store.get_doc("docs", "kept").unwrap().unwrap()["n"], 1);
    assert_eq!(store.get_doc("docs", "also").unwrap().unwrap()["n"], 2);
    assert!(store.get_doc("docs", "ghost").unwrap().is_none());
    assert_eq!(live(&path), before);
}

#[test]
fn two_handles_commit_and_a_reader_sees_the_put_without_unreachable() {
    let (_dir, path) = temp_fls();
    let writer = FerroStore::open(&path).unwrap();
    let reader = FerroStore::open(&path).unwrap();
    match FerroStore::open(&path) {
        Ok(_) => {}
        Err(StoreError::Unreachable { .. }) => panic!("another open returned Unreachable"),
        Err(err) => panic!("another open failed: {err}"),
    }

    let worker = std::thread::spawn(move || {
        writer
            .put_doc("docs", "k", &json!({ "v": "seen" }))
            .unwrap();
    });
    let start = std::time::Instant::now();
    let seen = loop {
        match reader.get_doc("docs", "k") {
            Ok(Some(value)) => {
                assert_eq!(value, json!({ "v": "seen" }));
                break true;
            }
            Ok(None) => {
                if start.elapsed() > Duration::from_secs(5) {
                    break false;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(err) => panic!("reader: {err}"),
        }
    };
    worker.join().expect("writer panicked");
    assert!(seen, "reader did not observe the committed document");

    let left = FerroStore::open(&path).unwrap();
    let right = FerroStore::open(&path).unwrap();
    let left_task = std::thread::spawn(move || {
        for i in 0..10 {
            left.put_doc("c", &format!("l{i}"), &json!({ "n": i }))
                .unwrap();
        }
    });
    let right_task = std::thread::spawn(move || {
        for i in 0..10 {
            right
                .put_doc("c", &format!("r{i}"), &json!({ "n": i }))
                .unwrap();
        }
    });
    left_task.join().unwrap();
    right_task.join().unwrap();
    let store = FerroStore::open(&path).unwrap();
    for i in 0..10 {
        assert_eq!(
            store.get_doc("c", &format!("l{i}")).unwrap().unwrap()["n"],
            i
        );
        assert_eq!(
            store.get_doc("c", &format!("r{i}")).unwrap().unwrap()["n"],
            i
        );
    }
    store.checkpoint().unwrap();
    // The earlier single put plus 20 concurrent puts.
    assert_eq!(live(&path).snapshot_seq, 21);
    assert_eq!(live(&path).wal_length, 0);
}

#[test]
fn documents_round_trip_field_queries_and_bm25() {
    let (_dir, path) = temp_fls();
    {
        let store = FerroStore::open(&path).unwrap();
        store
            .put_doc(
                "docs",
                "c",
                &json!({ "name": "ada", "n": 1, "ok": true, "skip": null, "tags": ["red"], "nest": { "name": "nope" } }),
            )
            .unwrap();
        store
            .put_doc("docs", "a", &json!({ "name": "ada", "n": 1, "ok": false }))
            .unwrap();
        store
            .put_doc("docs", "b", &json!({ "name": "bea", "n": 2 }))
            .unwrap();
        store
            .put_doc("docs", "f", &json!({ "x": 1.5, "neg": -2 }))
            .unwrap();
        assert_eq!(store.get_doc("docs", "b").unwrap().unwrap()["name"], "bea");
        store
            .put_doc("docs", "b", &json!({ "name": "bea2", "n": 2 }))
            .unwrap();
        assert_eq!(store.get_doc("docs", "b").unwrap().unwrap()["name"], "bea2");

        assert_eq!(
            store.query_by_field("docs", "name", "ada").unwrap(),
            vec!["a".to_string(), "c".to_string()]
        );
        assert_eq!(
            store.query_by_field("docs", "n", "1").unwrap(),
            vec!["a".to_string(), "c".to_string()]
        );
        assert_eq!(
            store.query_by_field("docs", "n", "2").unwrap(),
            vec!["b".to_string()]
        );
        assert_eq!(
            store.query_by_field("docs", "x", "1.5").unwrap(),
            vec!["f".to_string()]
        );
        assert_eq!(
            store.query_by_field("docs", "neg", "-2").unwrap(),
            vec!["f".to_string()]
        );
        assert_eq!(
            store.query_by_field("docs", "ok", "true").unwrap(),
            vec!["c".to_string()]
        );
        assert_eq!(
            store.query_by_field("docs", "ok", "false").unwrap(),
            vec!["a".to_string()]
        );
        assert!(
            store
                .query_by_field("docs", "tags", "red")
                .unwrap()
                .is_empty()
        );
        assert!(
            store
                .query_by_field("docs", "skip", "null")
                .unwrap()
                .is_empty()
        );
        assert!(
            store
                .query_by_field("docs", "name", "nope")
                .unwrap()
                .is_empty()
        );

        store.delete_doc("docs", "a").unwrap();
        assert!(store.get_doc("docs", "a").unwrap().is_none());
        assert_eq!(
            store.query_by_field("docs", "name", "ada").unwrap(),
            vec!["c".to_string()]
        );
        store.delete_doc("docs", "missing").unwrap();

        store
            .put_doc("bm25", "r", &json!({ "text": "cat cat" }))
            .unwrap();
        store
            .put_doc("bm25", "o", &json!({ "text": "cat dog" }))
            .unwrap();
        store
            .put_doc("bm25", "n", &json!({ "text": "bird bird" }))
            .unwrap();
        store
            .put_doc(
                "bm25",
                "nest",
                &json!({ "outer": { "inner": "zebra zebra" } }),
            )
            .unwrap();
        store
            .put_doc("bm25", "keys", &json!({ "cat": "hound" }))
            .unwrap();
        let hits = store.search_bm25("bm25", "cat", 10).unwrap();
        assert_eq!(hits[0].0, "r");
        assert_eq!(hits[1].0, "o");
        assert!(hits[0].1 > hits[1].1);
        assert!(hits.iter().all(|(id, _)| id != "n" && id != "keys"));
        let zebra = store.search_bm25("bm25", "ZeBra!", 10).unwrap();
        assert_eq!(zebra.len(), 1);
        assert_eq!(zebra[0].0, "nest");
        assert!(
            store
                .search_bm25("bm25", "hound", 10)
                .unwrap()
                .iter()
                .any(|(id, _)| id == "keys")
        );
        assert!(store.search_bm25("bm25", "", 10).unwrap().is_empty());
        assert!(store.search_bm25("bm25", "cat", 0).unwrap().is_empty());
        assert!(store.search_bm25("bm25", "!!!", 5).unwrap().is_empty());
    }

    let store = FerroStore::open(&path).unwrap();
    assert!(store.get_doc("docs", "a").unwrap().is_none());
    assert_eq!(store.get_doc("docs", "b").unwrap().unwrap()["name"], "bea2");
    assert_eq!(
        store.query_by_field("docs", "name", "ada").unwrap(),
        vec!["c".to_string()]
    );
    let hits = store.search_bm25("bm25", "cat", 10).unwrap();
    assert_eq!(hits[0].0, "r");
    assert!(hits[0].1 > hits[1].1);
    assert!(hits.iter().all(|(id, _)| id != "n"));
}

#[test]
fn ferromq_redelivers_until_ack_and_subscribers_see_the_next_publish() {
    let (_dir, path) = temp_fls();
    let store = FerroStore::open(&path).unwrap();
    assert!(store.poll_mq_group("orders", "g", 0).unwrap().is_empty());
    let seq1 = store.publish_mq("orders", "k", &json!({ "a": 1 })).unwrap();
    let seq2 = store.publish_mq("orders", "k2", &json!(2)).unwrap();
    assert_eq!((seq1, seq2), (1, 2));
    assert_eq!(
        store.publish_mq("other", "k", &json!("x")).unwrap(),
        1,
        "sequences are per topic"
    );
    store.ack_mq("orders", "g", 99).unwrap();

    let first = store.poll_mq_group("orders", "g", 1).unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].seq, 1);
    let both = store.poll_mq_group("orders", "g", 10).unwrap();
    assert_eq!(both.len(), 2);
    assert_eq!(both[0].seq, 1);
    assert_eq!(both[0].key, "k");
    assert_eq!(both[0].payload, json!({ "a": 1 }));
    assert_eq!(both[1].seq, 2);
    assert_eq!(both[1].payload, json!(2));

    let stored = store.get_doc("mq:orders", "1").unwrap().unwrap();
    assert_eq!(stored["seq"], 1);
    assert_eq!(stored["key"], "k");
    assert_eq!(stored["payload"]["a"], 1);

    store.ack_mq("orders", "g", 1).unwrap();
    store.ack_mq("orders", "g", 1).unwrap();
    let ack = store.get_doc("mq_ack:orders:g", "1").unwrap().unwrap();
    assert_eq!(ack["seq"], 1);
    let after = store.poll_mq_group("orders", "g", 10).unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].seq, 2);
    let again = store.poll_mq_group("orders", "g", 10).unwrap();
    assert_eq!(again[0].seq, 2);
    assert_eq!(
        store.poll_mq_group("orders", "other", 10).unwrap().len(),
        2,
        "another group has its own acks"
    );

    store.ack_mq("orders", "g", 2).unwrap();
    assert!(store.poll_mq_group("orders", "g", 10).unwrap().is_empty());

    let rx = store.subscribe_mq("orders");
    assert!(rx.try_recv().is_err(), "subscribe does not replay history");
    let seq3 = store
        .publish_mq("orders", "k3", &json!({ "n": 3 }))
        .unwrap();
    let rec = rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(rec.seq, seq3);
    assert_eq!(rec.key, "k3");
    assert_eq!(rec.payload, json!({ "n": 3 }));
    drop(store);

    let store = FerroStore::open(&path).unwrap();
    let pending = store.poll_mq_group("orders", "g", 10).unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].seq, 3);
    assert_eq!(store.publish_mq("orders", "k4", &json!(4)).unwrap(), 4);
}

/// A `ws://` client under the test PSK of [`psk_auth`].
fn connect_psk(uri: &str) -> Result<FerroStore, StoreError> {
    FerroStore::connect_with(uri, &super::tests_auth::psk_options())
}

fn must_not_unreachable<T>(result: Result<T, StoreError>) -> Result<T, String> {
    match result {
        Ok(value) => Ok(value),
        Err(err) => {
            assert!(!matches!(err, StoreError::Unreachable { .. }), "{err}");
            Err(err.to_string())
        }
    }
}

#[test]
fn unix_socket_serves_concurrent_clients_without_opening_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("store.fls");
    let sock = dir.path().join("store.sock");
    let server = FerroStoreServer::bind_unix(&sock, FerroStore::open(&file).unwrap()).unwrap();
    let uri = format!("unix://{}", sock.display());

    let mut handles = Vec::new();
    for i in 0..4 {
        let uri = uri.clone();
        handles.push(thread::spawn(move || {
            let client = must_not_unreachable(FerroStore::connect_uri(&uri))?;
            let id = format!("id-{i}");
            must_not_unreachable(client.put_doc(
                "docs",
                &id,
                &json!({"n": i, "name": format!("name{i}")}),
            ))?;
            let doc = must_not_unreachable(client.get_doc("docs", &id))?
                .ok_or_else(|| format!("missing {id}"))?;
            if doc["n"] != i {
                return Err(format!("doc {id} was {doc}"));
            }
            Ok(())
        }));
    }
    for handle in handles {
        handle.join().unwrap().unwrap();
    }

    let host_uri = format!("unix://localhost{}", sock.display());
    let client = FerroStore::connect_uri(&host_uri).unwrap();
    assert_eq!(
        client.query_by_field("docs", "name", "name1").unwrap(),
        vec!["id-1".to_string()]
    );
    let hits = client.search_bm25("docs", "name2", 10).unwrap();
    assert_eq!(hits.len(), 1, "{hits:?}");
    assert_eq!(hits[0].0, "id-2");
    client.delete_doc("docs", "id-3").unwrap();
    assert!(client.get_doc("docs", "id-3").unwrap().is_none());
    client.checkpoint().unwrap();
    client.with_read(|_| Ok(())).unwrap();
    client.with_write(|_| Ok(())).unwrap();

    assert_eq!(
        server.store().get_doc("docs", "id-0").unwrap().unwrap()["n"],
        0
    );
    assert_eq!(
        server.store().get_doc("docs", "id-1").unwrap().unwrap()["name"],
        "name1"
    );
    assert!(server.store().get_doc("docs", "id-3").unwrap().is_none());
    drop(client);
    drop(server);

    let opened = FerroStore::open(&file).unwrap();
    assert_eq!(opened.get_doc("docs", "id-0").unwrap().unwrap()["n"], 0);
    assert_eq!(
        opened.get_doc("docs", "id-1").unwrap().unwrap()["name"],
        "name1"
    );
    assert_eq!(opened.get_doc("docs", "id-2").unwrap().unwrap()["n"], 2);
    assert!(opened.get_doc("docs", "id-3").unwrap().is_none());
}

#[test]
fn unix_socket_mq_subscribe_does_not_replay_and_ack_matches_local() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("store.fls");
    let sock = dir.path().join("mq.sock");
    let server = FerroStoreServer::bind_unix(&sock, FerroStore::open(&file).unwrap()).unwrap();
    let uri = format!("unix://{}", sock.display());

    let publisher = FerroStore::connect_uri(&uri).unwrap();
    let early = publisher
        .publish_mq("orders", "before", &json!({"n": 0}))
        .unwrap();
    assert_eq!(early, 1);

    let client_a = FerroStore::connect_uri(&uri).unwrap();
    let rx = client_a.subscribe_mq("orders");
    assert!(
        rx.recv_timeout(Duration::from_millis(200)).is_err(),
        "subscribe does not replay history"
    );

    let client_b = FerroStore::connect_uri(&uri).unwrap();
    let seq = client_b
        .publish_mq("orders", "k", &json!({"n": 1}))
        .unwrap();
    let rec = rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(rec.seq, seq);
    assert_eq!(rec.key, "k");
    assert_eq!(rec.payload, json!({"n": 1}));
    assert!(rx.recv_timeout(Duration::from_millis(200)).is_err());

    let polled = client_a.poll_mq_group("orders", "g", 10).unwrap();
    assert_eq!(polled.len(), 2);
    assert_eq!(polled[0].seq, 1);
    assert_eq!(polled[0].key, "before");
    assert_eq!(polled[0].payload, json!({"n": 0}));
    assert_eq!(polled[1].seq, seq);
    assert_eq!(polled[1].key, "k");
    assert_eq!(polled[1].payload, json!({"n": 1}));
    assert_eq!(client_b.poll_mq_group("orders", "g", 10).unwrap().len(), 2);
    client_a.ack_mq("orders", "g", 1).unwrap();
    let after = client_a.poll_mq_group("orders", "g", 10).unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].seq, seq);
    assert_eq!(
        client_a.poll_mq_group("orders", "other", 10).unwrap().len(),
        2
    );
    assert!(client_a.poll_mq_group("orders", "g", 0).unwrap().is_empty());
    client_a.ack_mq("orders", "g", seq).unwrap();
    assert!(
        client_a
            .poll_mq_group("orders", "g", 10)
            .unwrap()
            .is_empty()
    );
    client_a.ack_mq("orders", "g", 99).unwrap();
    assert!(
        client_a
            .poll_mq_group("orders", "g", 10)
            .unwrap()
            .is_empty()
    );

    drop(client_a);
    drop(client_b);
    drop(publisher);
    drop(server);
    let opened = FerroStore::open(&file).unwrap();
    assert!(opened.poll_mq_group("orders", "g", 10).unwrap().is_empty());
    assert_eq!(
        opened.poll_mq_group("orders", "other", 10).unwrap().len(),
        2
    );
}

#[test]
fn bind_unix_refuses_a_non_socket_and_drop_unlinks_its_socket() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("not-a-socket");
    std::fs::write(&file, b"file").unwrap();
    let err = match FerroStoreServer::bind_unix(&file, FerroStore::in_memory()) {
        Ok(_) => panic!("non-socket path was bound"),
        Err(err) => err,
    };
    assert!(matches!(err, StoreError::Backend(_)), "{err}");
    assert!(file.is_file(), "a non-socket path is left in place");

    let sock = dir.path().join("s.sock");
    let stale = std::os::unix::net::UnixListener::bind(&sock).unwrap();
    drop(stale);
    assert!(sock.exists());
    let server = FerroStoreServer::bind_unix(&sock, FerroStore::in_memory()).unwrap();
    let client = FerroStore::connect_uri(&format!("unix://{}", sock.display())).unwrap();
    client.put_doc("c", "a", &json!(1)).unwrap();
    assert_eq!(server.store().get_doc("c", "a").unwrap().unwrap(), json!(1));
    drop(client);
    drop(server);
    assert!(!sock.exists());
}

#[test]
fn unix_socket_get_doc_returns_stored_null_and_missing_is_none() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("null.sock");
    let server = FerroStoreServer::bind_unix(&sock, FerroStore::in_memory()).unwrap();
    let client = FerroStore::connect_uri(&format!("unix://{}", sock.display())).unwrap();
    client.put_doc("docs", "empty", &json!(null)).unwrap();
    let stored = client.get_doc("docs", "empty");
    assert!(
        matches!(stored, Ok(Some(ref value)) if value.is_null()),
        "{stored:?}"
    );
    let missing = client.get_doc("docs", "missing");
    assert!(matches!(missing, Ok(None)), "{missing:?}");
    drop(client);
    drop(server);
}

#[test]
fn unix_socket_subscribe_after_server_drop_is_disconnected() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("gone.sock");
    let server = FerroStoreServer::bind_unix(&sock, FerroStore::in_memory()).unwrap();
    let client = FerroStore::connect_uri(&format!("unix://{}", sock.display())).unwrap();
    drop(server);
    let rx = client.subscribe_mq("orders");
    assert_eq!(
        rx.recv_timeout(Duration::from_millis(400)).unwrap_err(),
        RecvTimeoutError::Disconnected
    );
}

#[test]
fn websocket_serves_concurrent_clients_without_opening_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("store.fls");
    let server =
        FerroStoreServer::bind_tcp("127.0.0.1:0", FerroStore::open(&file).unwrap(), psk_auth())
            .unwrap();
    assert_ne!(server.local_addr().port(), 0);
    let uri = format!("ws://{}", server.local_addr());

    let mut handles = Vec::new();
    for i in 0..4 {
        let uri = uri.clone();
        handles.push(thread::spawn(move || {
            let client = must_not_unreachable(connect_psk(&uri))?;
            let id = format!("id-{i}");
            must_not_unreachable(client.put_doc(
                "docs",
                &id,
                &json!({"n": i, "name": format!("name{i}")}),
            ))?;
            let doc = must_not_unreachable(client.get_doc("docs", &id))?
                .ok_or_else(|| format!("missing {id}"))?;
            if doc["n"] != i {
                return Err(format!("doc {id} was {doc}"));
            }
            Ok(())
        }));
    }
    for handle in handles {
        handle.join().unwrap().unwrap();
    }

    for i in 0..4 {
        assert_eq!(
            server
                .store()
                .get_doc("docs", &format!("id-{i}"))
                .unwrap()
                .unwrap()["n"],
            i
        );
    }
    drop(server);

    let opened = FerroStore::open(&file).unwrap();
    for i in 0..4 {
        assert_eq!(
            opened.get_doc("docs", &format!("id-{i}")).unwrap().unwrap()["n"],
            i
        );
    }
}

#[test]
fn websocket_mq_subscribe_does_not_replay_and_ack_matches_local() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("store.fls");
    let server =
        FerroStoreServer::bind_tcp("127.0.0.1:0", FerroStore::open(&file).unwrap(), psk_auth())
            .unwrap();
    let uri = format!("ws://{}", server.local_addr());

    let publisher = connect_psk(&uri).unwrap();
    let early = publisher
        .publish_mq("orders", "before", &json!({"n": 0}))
        .unwrap();
    assert_eq!(early, 1);

    let client_a = connect_psk(&uri).unwrap();
    let rx = client_a.subscribe_mq("orders");
    assert!(
        rx.recv_timeout(Duration::from_millis(200)).is_err(),
        "subscribe does not replay history"
    );

    let client_b = connect_psk(&format!("{uri}/store")).unwrap();
    let seq = client_b
        .publish_mq("orders", "k", &json!({"n": 1}))
        .unwrap();
    let rec = rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(rec.seq, seq);
    assert_eq!(rec.key, "k");
    assert_eq!(rec.payload, json!({"n": 1}));
    assert!(rx.recv_timeout(Duration::from_millis(200)).is_err());

    let polled = client_a.poll_mq_group("orders", "g", 10).unwrap();
    assert_eq!(polled.len(), 2);
    assert_eq!(polled[0].seq, 1);
    assert_eq!(polled[0].key, "before");
    assert_eq!(polled[0].payload, json!({"n": 0}));
    assert_eq!(polled[1].seq, seq);
    assert_eq!(polled[1].key, "k");
    assert_eq!(polled[1].payload, json!({"n": 1}));
    assert_eq!(client_b.poll_mq_group("orders", "g", 10).unwrap().len(), 2);
    client_a.ack_mq("orders", "g", 1).unwrap();
    let after = client_a.poll_mq_group("orders", "g", 10).unwrap();
    assert_eq!(after.len(), 1);
    assert_eq!(after[0].seq, seq);
    assert_eq!(
        client_a.poll_mq_group("orders", "other", 10).unwrap().len(),
        2
    );
    assert!(client_a.poll_mq_group("orders", "g", 0).unwrap().is_empty());
    client_a.ack_mq("orders", "g", seq).unwrap();
    assert!(
        client_a
            .poll_mq_group("orders", "g", 10)
            .unwrap()
            .is_empty()
    );
    client_a.ack_mq("orders", "g", 99).unwrap();
    assert!(
        client_a
            .poll_mq_group("orders", "g", 10)
            .unwrap()
            .is_empty()
    );

    drop(client_a);
    drop(client_b);
    drop(publisher);
    drop(server);
    let opened = FerroStore::open(&file).unwrap();
    assert!(opened.poll_mq_group("orders", "g", 10).unwrap().is_empty());
    assert_eq!(
        opened.poll_mq_group("orders", "other", 10).unwrap().len(),
        2
    );
}

#[test]
fn websocket_get_doc_returns_stored_null() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("null.fls");
    let server =
        FerroStoreServer::bind_tcp("127.0.0.1:0", FerroStore::open(&file).unwrap(), psk_auth())
            .unwrap();
    let client = connect_psk(&format!("ws://{}/store", server.local_addr())).unwrap();
    client.put_doc("docs", "empty", &json!(null)).unwrap();
    let stored = client.get_doc("docs", "empty");
    assert!(
        matches!(stored, Ok(Some(ref value)) if value.is_null()),
        "{stored:?}"
    );
    let missing = client.get_doc("docs", "missing");
    assert!(matches!(missing, Ok(None)), "{missing:?}");
    assert!(
        server
            .store()
            .get_doc("docs", "empty")
            .unwrap()
            .unwrap()
            .is_null()
    );
    drop(client);
    drop(server);
}

#[test]
fn websocket_subscribe_disconnects_when_the_server_drops() {
    let server =
        FerroStoreServer::bind_tcp("127.0.0.1:0", FerroStore::in_memory(), psk_auth()).unwrap();
    let client = connect_psk(&format!("ws://{}", server.local_addr())).unwrap();
    let rx = client.subscribe_mq("orders");
    drop(server);
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(2)).unwrap_err(),
        RecvTimeoutError::Disconnected
    );
}

#[test]
fn unix_socket_subscribe_disconnects_when_the_server_drops() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("live.sock");
    let server = FerroStoreServer::bind_unix(&sock, FerroStore::in_memory()).unwrap();
    let client = FerroStore::connect_uri(&format!("unix://{}", sock.display())).unwrap();
    let rx = client.subscribe_mq("orders");
    drop(server);
    assert_eq!(
        rx.recv_timeout(Duration::from_millis(500)).unwrap_err(),
        RecvTimeoutError::Disconnected
    );
}

fn committed_frames(path: &Path) -> Vec<(u64, Vec<Op>)> {
    let file = File::open(path).unwrap();
    let sb = live(path);
    let mut buf = vec![0u8; usize::try_from(sb.wal_length).unwrap()];
    if !buf.is_empty() {
        file.read_exact_at(&mut buf, sb.wal_offset).unwrap();
    }
    layout::decode_wal(&buf).unwrap()
}

#[test]
fn ferromq_sequence_survives_acked_delete_checkpoint_and_reopen() {
    let (_dir, path) = temp_fls();
    {
        let store = FerroStore::open(&path).unwrap();
        assert_eq!(
            store.publish_mq("orders", "k", &json!({ "a": 1 })).unwrap(),
            1
        );
        let frames = committed_frames(&path);
        assert_eq!(frames.len(), 1, "message and high-water share one frame");
        let (seq, ops) = &frames[0];
        assert_eq!(*seq, 1);
        assert_eq!(ops.len(), 2);
        match &ops[..] {
            [
                Op::Put {
                    collection: message_collection,
                    id: message_id,
                    doc: message,
                },
                Op::Put {
                    collection: hw_collection,
                    id: hw_id,
                    doc: hw,
                },
            ] => {
                assert_eq!(message_collection, "mq:orders");
                assert_eq!(message_id, "1");
                assert_eq!(message["seq"], 1);
                assert_eq!(hw_collection, "mq_hw:orders");
                assert_eq!(hw_id, "high_water");
                assert_eq!(hw["seq"], 1);
            }
            other => panic!("expected message and high-water puts, got {other:?}"),
        }

        store.ack_mq("orders", "g", 1).unwrap();
        store.delete_doc("mq:orders", "1").unwrap();
        assert!(store.get_doc("mq:orders", "1").unwrap().is_none());
        assert_eq!(
            store
                .get_doc("mq_hw:orders", "high_water")
                .unwrap()
                .unwrap()["seq"],
            1
        );
        store.checkpoint().unwrap();
    }

    let store = FerroStore::open(&path).unwrap();
    assert!(store.get_doc("mq:orders", "1").unwrap().is_none());
    assert_eq!(
        store
            .get_doc("mq_hw:orders", "high_water")
            .unwrap()
            .unwrap()["seq"],
        1
    );
    let seq = store.publish_mq("orders", "k2", &json!(2)).unwrap();
    assert_eq!(seq, 2);
    let pending = store.poll_mq_group("orders", "g", 10).unwrap();
    assert_eq!(pending.len(), 1);
    assert_eq!(pending[0].seq, 2);
    assert_eq!(pending[0].key, "k2");
    assert_eq!(pending[0].payload, json!(2));

    // A file that already has messages and no high-water document.
    store
        .put_doc(
            "mq:legacy",
            "4",
            &json!({ "seq": 4, "key": "old", "payload": true }),
        )
        .unwrap();
    assert_eq!(store.publish_mq("legacy", "n", &json!(1)).unwrap(), 5);
    store.delete_doc("mq:legacy", "4").unwrap();
    store.delete_doc("mq:legacy", "5").unwrap();
    assert_eq!(store.publish_mq("legacy", "n2", &json!(2)).unwrap(), 6);
    let legacy = store.poll_mq_group("legacy", "g", 10).unwrap();
    assert_eq!(legacy.len(), 1);
    assert_eq!(legacy[0].seq, 6);
}

#[test]
fn connect_uri_mem_and_file_and_refuses_other_schemes() {
    let mem = FerroStore::connect_uri("mem://").unwrap();
    mem.checkpoint().unwrap();
    mem.put_doc("c", "a", &json!({ "n": 1 })).unwrap();
    assert_eq!(mem.get_doc("c", "a").unwrap().unwrap()["n"], 1);
    let other = FerroStore::connect_uri("mem:///").unwrap();
    other.checkpoint().unwrap();
    assert!(other.get_doc("c", "a").unwrap().is_none());

    assert_eq!(file_uri_path("/tmp/x.fls"), "/tmp/x.fls");
    assert_eq!(file_uri_path("localhost/tmp/x.fls"), "/tmp/x.fls");
    assert_eq!(file_uri_path("rel/x.fls"), "rel/x.fls");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("x.fls");
    let absolute = format!("file://{}", path.display());
    {
        let store = FerroStore::connect_uri(&absolute).unwrap();
        store.put_doc("c", "a", &json!({ "n": 7 })).unwrap();
    }
    let store = FerroStore::connect_uri(&absolute).unwrap();
    assert_eq!(store.get_doc("c", "a").unwrap().unwrap()["n"], 7);
    let via_host = format!("file://localhost{}", path.display());
    let store = FerroStore::connect_uri(&via_host).unwrap();
    assert_eq!(store.get_doc("c", "a").unwrap().unwrap()["n"], 7);
    drop(store);
    let store = FerroStore::open(&path).unwrap();
    assert_eq!(store.get_doc("c", "a").unwrap().unwrap()["n"], 7);

    let name = format!("ferrostore-rel-{}", std::process::id());
    let _cleanup = RemoveDir(PathBuf::from(&name));
    std::fs::create_dir_all(&name).unwrap();
    let uri = format!("file://{name}/t.fls");
    {
        let store = FerroStore::connect_uri(&uri).unwrap();
        store.put_doc("c", "rel", &json!({ "ok": true })).unwrap();
    }
    let reopened = FerroStore::connect_uri(&uri).unwrap();
    assert_eq!(reopened.get_doc("c", "rel").unwrap().unwrap()["ok"], true);

    for uri in ["ws://127.0.0.1:9", "ws://127.0.0.1:9/store"] {
        let err = match FerroStore::connect_uri(uri) {
            Ok(_) => panic!("closed ws port {uri} was accepted"),
            Err(err) => err,
        };
        let msg = err.to_string();
        assert!(matches!(err, StoreError::Backend(_)), "{uri}: {msg}");
        assert!(!msg.contains("unknown"), "{uri}: {msg}");
        assert!(!msg.contains("Phase 2"), "{uri}: {msg}");
        assert!(!msg.contains("deferred"), "{uri}: {msg}");
    }
    let wss = match FerroStore::connect_uri("wss://example.test/store") {
        Ok(_) => panic!("wss uri was accepted"),
        Err(err) => err,
    };
    let msg = wss.to_string();
    assert!(matches!(wss, StoreError::Backend(_)), "{msg}");
    assert!(msg.contains("wss"), "{msg}");
    assert!(!msg.contains("unknown"), "{msg}");
    assert!(!msg.contains("Phase 2"), "{msg}");
    assert!(!msg.contains("deferred"), "{msg}");
    let missing = match FerroStore::connect_uri("unix:///tmp/ferro.sock") {
        Ok(_) => panic!("missing unix socket was accepted"),
        Err(err) => err,
    };
    let msg = missing.to_string();
    assert!(matches!(missing, StoreError::Backend(_)), "{msg}");
    assert!(!msg.contains("unknown"), "{msg}");
    assert!(!msg.contains("Phase 2"), "{msg}");
    assert!(!msg.contains("deferred"), "{msg}");
    let via_host = match FerroStore::connect_uri("unix://localhost/tmp/ferro.sock") {
        Ok(_) => panic!("missing unix socket was accepted"),
        Err(err) => err,
    };
    let msg = via_host.to_string();
    assert!(msg.contains("/tmp/ferro.sock"), "{msg}");
    assert!(!msg.contains("unknown"), "{msg}");
    assert!(!msg.contains("Phase 2"), "{msg}");
    let rel = format!("unix://ferrostore-missing-{}/x.sock", std::process::id());
    let missing_rel = match FerroStore::connect_uri(&rel) {
        Ok(_) => panic!("missing unix socket was accepted"),
        Err(err) => err,
    };
    let msg = missing_rel.to_string();
    assert!(
        msg.contains(&format!("ferrostore-missing-{}/x.sock", std::process::id())),
        "{msg}"
    );
    assert!(!msg.contains("unknown"), "{msg}");
    assert!(!msg.contains("Phase 2"), "{msg}");
    for uri in ["", "http://example.test", "bogus", "mem://other"] {
        let err = match FerroStore::connect_uri(uri) {
            Ok(_) => panic!("uri {uri} was accepted"),
            Err(err) => err,
        };
        let msg = err.to_string();
        assert!(msg.contains("unknown"), "{uri}: {msg}");
        assert!(!msg.contains("Phase 2"), "{uri}: {msg}");
        assert!(!msg.contains("deferred"), "{uri}: {msg}");
    }
}

#[test]
fn with_write_is_one_frame_and_a_failure_or_empty_body_writes_nothing() {
    let (_dir, path) = temp_fls();
    let store = FerroStore::open(&path).unwrap();
    store
        .with_write(|txn| {
            txn.put("c", "a", json!(1));
            assert_eq!(txn.get("c", "a").unwrap(), json!(1));
            txn.put("c", "a", json!(2));
            assert_eq!(txn.get("c", "a").unwrap(), json!(2));
            txn.put("c", "b", json!(3));
            txn.delete("c", "b");
            assert!(txn.get("c", "b").is_none());
            txn.put("d", "z", json!(4));
            let ids: Vec<_> = txn.scan("c").into_iter().map(|(id, _)| id).collect();
            assert_eq!(ids, vec!["a".to_string()]);
            assert!(txn.scan("missing").is_empty());
            Ok(())
        })
        .unwrap();
    assert_eq!(store.get_doc("c", "a").unwrap().unwrap(), json!(2));
    assert!(store.get_doc("c", "b").unwrap().is_none());
    assert_eq!(store.get_doc("d", "z").unwrap().unwrap(), json!(4));

    let err = store
        .with_write(|txn| {
            txn.put("c", "nope", json!(9));
            Err::<(), _>(layout::backend("rollback"))
        })
        .unwrap_err();
    assert!(err.to_string().contains("rollback"), "{err}");
    assert!(store.get_doc("c", "nope").unwrap().is_none());

    store
        .with_write(|txn| {
            assert_eq!(txn.get("c", "a").unwrap(), json!(2));
            Ok(())
        })
        .unwrap();

    store.checkpoint().unwrap();
    assert_eq!(live(&path).snapshot_seq, 1);
    assert_eq!(live(&path).wal_length, 0);

    let mem = FerroStore::in_memory();
    mem.with_write(|txn| {
        txn.put("c", "a", json!("mem"));
        assert_eq!(txn.scan("c").len(), 1);
        txn.delete("c", "a");
        assert!(txn.scan("c").is_empty());
        txn.put("c", "b", json!(1));
        Ok(())
    })
    .unwrap();
    assert!(mem.get_doc("c", "a").unwrap().is_none());
    assert_eq!(mem.get_doc("c", "b").unwrap().unwrap(), json!(1));
    assert!(
        mem.with_write(|txn| {
            txn.put("c", "gone", json!(1));
            Err::<(), _>(layout::backend("rollback"))
        })
        .is_err()
    );
    assert!(mem.get_doc("c", "gone").unwrap().is_none());
}

#[test]
fn ferrostore_meets_the_store_contracts() {
    use fl_core::conformance::{Single, SplitOver};
    let fresh = || (FerroStore::in_memory(), ());
    let single = || {
        let (s, g) = fresh();
        Single(s, g)
    };
    let split = || {
        let (s, g) = fresh();
        SplitOver(s, g)
    };
    fl_core::conformance::catalog(fresh);
    fl_core::conformance::tracker(single);
    fl_core::conformance::ledger(single);
    fl_core::conformance::ledger(split);
    fl_core::conformance::split_ledger(split);
    fl_core::conformance::all_roles(single);
    fl_core::conformance::local_handles(fresh);
    fl_core::conformance::ledger_cache(fresh);
    fl_core::conformance::agent_registry(fresh);
    fl_core::conformance::state_store(fresh);
    fl_core::conformance::message_store(fresh);
    fl_core::conformance::workflow_store(fresh);
    fl_core::conformance::escalations(|| {
        let (s, g) = fresh();
        Single(s, g)
    });
}

#[test]
fn project_record_and_state_survive_reopen() {
    use fl_core::model::State;
    use fl_core::store::{Catalog, Tracker};
    let (_dir, path) = temp_fls();
    let (project, record) = {
        let store = FerroStore::open(&path).unwrap();
        let project = store.add_project("/work/app").unwrap();
        let record = store.add_record(&project, "keep this").unwrap();
        store.set_record_state(&record, State::Doing).unwrap();
        (project, record)
    };
    let store = FerroStore::open(&path).unwrap();
    let loaded = store.get_project(&project).unwrap().expect("project");
    assert_eq!(loaded.id, project);
    assert_eq!(loaded.root, "/work/app");
    let records = store.list_records(&project).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].id, record);
    assert_eq!(records[0].title, "keep this");
    assert_eq!(records[0].state, State::Doing);
}

struct RemoveDir(PathBuf);

impl Drop for RemoveDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn gate_kind() -> fl_core::model::GateKind {
    fl_core::model::GateKind::Command(fl_core::model::CommandSpec {
        program: "true".into(),
        args: vec![],
        delivery: fl_core::model::PopulationDelivery::Args,
        timeout_secs: 5,
        pass_codes: vec![0],
    })
}

fn gate_selector() -> fl_core::model::Selector {
    fl_core::model::Selector::Glob {
        pattern: "**/*.rs".into(),
    }
}

fn redb_format(store: &crate::RedbStore) -> u64 {
    let tx = store.db.begin_read().unwrap();
    let meta = tx.open_table(crate::META).unwrap();
    meta.get(crate::FORMAT_KEY)
        .unwrap()
        .expect("format version")
        .value()
}

#[test]
fn a_redb_store_migrates_into_ferrostore_without_reminting_ids() {
    use fl_core::agent::{AgentRole, AreaRecord, RegisterAgentArgs};
    use fl_core::at::At;
    use fl_core::conformance::{entry_iri, sample_attempt, sample_record_run};
    use fl_core::ferrowire::{MessageKind, MessageUrgency, SendWireArgs, WireAddress, WireFilter};
    use fl_core::finding::Finding;
    use fl_core::ids::{Kind, RecordId, WorkflowId, WorkflowRunId, seq_iri};
    use fl_core::model::{Regret, State, Transition};
    use fl_core::split::{CachedSegment, LedgerCache};
    use fl_core::state::{
        HandoffEntry, SensitivityPatch, StatePayload, StateScope, UpsertStateArgs,
    };
    use fl_core::store::{
        AgentRegistry, Bindings, Catalog, Handles, Ledger, MessageStore, StateStore, Tracker,
        WorkflowStore,
    };
    use fl_core::workflow::{
        AssigneeRule, DEFAULT_MAX_TOTAL_STEPS, DEFAULT_STEP_MAX_ATTEMPTS, DEFAULT_STEP_MAX_VISITS,
        END_STEP, EdgeCondition, EdgeDef, JoinPolicy, StepCompletion, StepDef, WorkflowDef,
        WorkflowRun,
    };
    use std::collections::BTreeMap;

    use crate::RedbStore;
    use crate::migrate_redb_to_ferrostore;

    let dir = tempfile::tempdir().unwrap();
    let redb = RedbStore::open(&dir.path().join("source.redb")).unwrap();
    let project = redb.add_project("/work/app").unwrap();
    let gate = redb
        .add_gate(&project, "fmt", gate_kind(), gate_selector(), 1, "c1", "o")
        .unwrap();
    redb.add_transition(Transition {
        project: project.clone(),
        name: "ship".into(),
        from: State::Review,
        to: State::Done,
        regret: Regret::High,
        gates: vec![gate.clone()],
    })
    .unwrap();
    let record = redb.add_record(&project, "keep this").unwrap();
    redb.set_record_state(&record, State::Doing).unwrap();
    let finding = redb
        .add_finding(Finding::raise(
            project.clone(),
            record.clone(),
            "reviewer",
            "wrong on empty",
        ))
        .unwrap();
    let alias = entry_iri(90);
    redb.add_alias(record.iri(), alias.clone()).unwrap();
    redb.append_gate_run(sample_record_run(1, &gate, Some(&record)))
        .unwrap();
    redb.append_attempt(sample_attempt(2, &project, &record))
        .unwrap();

    let area = AreaRecord::new(
        "ferroloop",
        Some("FerroLoop/".into()),
        "Agent orchestration",
        At::from_unix_millis(1),
        "owner",
    )
    .unwrap();
    redb.upsert_area(area).unwrap();
    let agent = redb
        .register_agent(RegisterAgentArgs {
            display_name: "bob".into(),
            email: None,
            project: "ferroloop".into(),
            role: AgentRole::Lead,
            extra_areas: vec![],
            create_area: false,
            replace: false,
            at: At::from_unix_millis(2),
            by: "owner".into(),
        })
        .unwrap();
    let state = redb
        .upsert_state(UpsertStateArgs {
            scope: StateScope::Workspace,
            slug: "rule".into(),
            sensitivity: SensitivityPatch::Inherit,
            payload: StatePayload::Handoff(HandoffEntry {
                summary: "v6".into(),
                shipped: vec![],
                in_flight: vec![],
                blockers: vec![],
                next_steps: vec![],
                notes: None,
            }),
            at: At::from_unix_millis(10),
            by: "carol".into(),
        })
        .unwrap();

    let workflow = redb
        .upsert_workflow_def(WorkflowDef {
            id: WorkflowId(seq_iri(801)),
            project: project.clone(),
            slug: "ci-flow".into(),
            title: "CI Flow".into(),
            start_step: "build".into(),
            steps: vec![StepDef {
                step_id: "build".into(),
                title: "Build".into(),
                prompt: "Run build.".into(),
                assignee: AssigneeRule::Agent {
                    name: "alice".into(),
                },
                reviews_step: None,
                require_different_vendor: false,
                completion: StepCompletion::Ungated,
                join_policy: JoinPolicy::All,
                max_attempts: DEFAULT_STEP_MAX_ATTEMPTS,
                max_visits: DEFAULT_STEP_MAX_VISITS,
                budget_usd_micros: None,
                max_tokens: None,
                isolate_worktree: false,
            }],
            edges: vec![EdgeDef {
                from_step: "build".into(),
                to_step: END_STEP.into(),
                on: EdgeCondition::Pass,
            }],
            max_total_steps: DEFAULT_MAX_TOTAL_STEPS,
            budget_usd_micros: None,
            max_tokens: None,
            declared_keys: BTreeMap::new(),
            authored_at_commit: "0123456789abcdef0123456789abcdef01234567".into(),
            authored_by: "bob".into(),
        })
        .unwrap();
    let workflow_run = redb
        .create_workflow_run(
            WorkflowRun::new(
                WorkflowRunId(seq_iri(802)),
                &workflow,
                "CI Run 1",
                At::from_unix_millis(100),
                "bob",
            )
            .unwrap(),
        )
        .unwrap();

    let (message, _) = redb
        .send_wire(SendWireArgs {
            id: None,
            sender_id: None,
            sender_name: "carol".into(),
            sender_host: Some("build-host".into()),
            to: WireAddress::AgentName("bob".into()),
            kind: MessageKind::Mail,
            urgency: MessageUrgency::Normal,
            durable: false,
            subject: "Format 7 test".into(),
            body: "Raise to 7".into(),
            reply_to: None,
            thread_id: None,
            deadline_at: None,
            at: At::from_unix_millis(10),
        })
        .unwrap();

    let root = "0123456789abcdef0123456789abcdef01234567";
    redb.set_ledger_root("R_1", root).unwrap();
    let segment = CachedSegment {
        oid: "o1".into(),
        bytes: vec![0xff, 0xfe, 0xfd],
        closed: false,
    };
    redb.remember(
        "R_1",
        "c1",
        &[("runs/aa/1.jsonl".to_string(), segment.clone())],
    )
    .unwrap();

    let version = redb_format(&redb);
    let fls = dir.path().join("migrated.fls");
    {
        let ferro = FerroStore::open(&fls).unwrap();
        migrate_redb_to_ferrostore(&redb, &ferro).unwrap();
    }
    let ferro = FerroStore::open(&fls).unwrap();

    assert_eq!(
        ferro.get_project(&project).unwrap(),
        redb.get_project(&project).unwrap()
    );
    assert_eq!(
        ferro.get_gate(&gate).unwrap(),
        redb.get_gate(&gate).unwrap()
    );
    assert_eq!(
        ferro.get_transition(&project, "ship").unwrap(),
        redb.get_transition(&project, "ship").unwrap()
    );
    assert_eq!(
        ferro.get_record(&record).unwrap(),
        redb.get_record(&record).unwrap()
    );
    assert_eq!(
        ferro.get_record(&record).unwrap().unwrap().state,
        State::Doing
    );
    assert_eq!(
        ferro
            .get_record(&RecordId(alias.clone()))
            .unwrap()
            .unwrap()
            .id,
        record
    );
    assert_eq!(
        ferro.get_finding(&finding).unwrap(),
        redb.get_finding(&finding).unwrap()
    );
    assert_eq!(
        ferro.gate_runs(&gate).unwrap(),
        redb.gate_runs(&gate).unwrap()
    );
    assert_eq!(
        ferro.attempts(&project).unwrap(),
        redb.attempts(&project).unwrap()
    );
    assert_eq!(
        ferro.get_area("ferroloop").unwrap(),
        redb.get_area("ferroloop").unwrap()
    );
    assert_eq!(
        ferro.get_agent(&agent.id).unwrap(),
        redb.get_agent(&agent.id).unwrap()
    );
    assert_eq!(
        ferro.get_state(&state.id).unwrap(),
        redb.get_state(&state.id).unwrap()
    );
    assert_eq!(
        ferro.get_workflow_def(&workflow.id).unwrap(),
        redb.get_workflow_def(&workflow.id).unwrap()
    );
    assert_eq!(
        ferro.get_workflow_run(&workflow_run.id).unwrap(),
        redb.get_workflow_run(&workflow_run.id).unwrap()
    );
    assert_eq!(
        ferro.get_wire_message(&message.id).unwrap(),
        redb.get_wire_message(&message.id).unwrap()
    );
    let filter = WireFilter {
        recipient: Some("bob".into()),
        ..WireFilter::default()
    };
    assert_eq!(
        ferro.list_wire_inbox(&filter).unwrap(),
        redb.list_wire_inbox(&filter).unwrap()
    );
    assert!(!ferro.list_wire_inbox(&filter).unwrap().is_empty());
    assert_eq!(
        ferro.cached("R_1", "runs/aa/1.jsonl").unwrap(),
        Some(segment)
    );
    assert_eq!(ferro.ledger_root("R_1").unwrap().as_deref(), Some(root));
    for (kind, id) in [
        (Kind::Project, project.iri()),
        (Kind::Record, record.iri()),
        (Kind::Agent, agent.id.iri()),
    ] {
        let handle = redb.handle_of(kind, id).unwrap();
        assert!(handle.is_some(), "{kind:?} has no handle");
        assert_eq!(ferro.handle_of(kind, id).unwrap(), handle);
    }
    // The schema version is the source's, kept once, in `__meta`.
    assert_eq!(ferro.format_version().unwrap(), Some(version));
    assert_eq!(
        ferro.get_doc("__meta", "schema_version").unwrap(),
        Some(json!(version))
    );
    assert_eq!(ferro.get_doc("meta", "format_version").unwrap(), None);
}

#[test]
fn migration_refuses_a_non_empty_store_and_an_empty_redb_stays_empty() {
    use fl_core::store::{Catalog, StoreError};

    use crate::RedbStore;
    use crate::migrate_redb_to_ferrostore;

    let dir = tempfile::tempdir().unwrap();
    let redb = RedbStore::open(&dir.path().join("empty.redb")).unwrap();
    let mem = FerroStore::in_memory();
    migrate_redb_to_ferrostore(&redb, &mem).unwrap();
    assert!(mem.list_projects().unwrap().is_empty());
    let project = mem.add_project("/p").unwrap();
    assert_eq!(mem.get_project(&project).unwrap().unwrap().root, "/p");
    assert_eq!(mem.get_project(&project).unwrap().unwrap().id, project);

    let occupied = FerroStore::in_memory();
    occupied
        .put_doc("docs", "kept", &json!({ "ok": true }))
        .unwrap();
    let err = migrate_redb_to_ferrostore(&redb, &occupied).unwrap_err();
    assert!(matches!(err, StoreError::Backend(_)), "{err}");
    assert!(err.to_string().contains("column table"), "{err}");
    let tabular = FerroStore::in_memory();
    tabular.create_column_table("t", &int_column()).unwrap();
    let err = migrate_redb_to_ferrostore(&redb, &tabular).unwrap_err();
    assert!(err.to_string().contains("column table"), "{err}");
    let vectors = FerroStore::in_memory();
    vectors
        .create_vector_collection("emb", 2, VectorStorage::F32, VectorMetric::Cosine, 4, 8, 8)
        .unwrap();
    let err = migrate_redb_to_ferrostore(&redb, &vectors).unwrap_err();
    assert!(err.to_string().contains("vector collection"), "{err}");
    assert_eq!(
        occupied.get_doc("docs", "kept").unwrap(),
        Some(json!({ "ok": true }))
    );
    assert!(occupied.list_projects().unwrap().is_empty());
    assert!(
        occupied
            .get_doc("meta", "format_version")
            .unwrap()
            .is_none()
    );
}

#[test]
fn nonzero_short_file_is_not_initialized() {
    let (_dir, path) = temp_fls();
    {
        let file = File::create(&path).unwrap();
        file.write_all_at(&[0x01], 0).unwrap();
        file.sync_all().unwrap();
    }
    let err = match FerroStore::open(&path) {
        Ok(_) => panic!("open initialized a one-byte non-zero file"),
        Err(err) => err,
    };
    let msg = err.to_string();
    assert!(
        matches!(err, StoreError::Backend(_)) && msg.contains("no valid FerroStore superblock"),
        "{msg}"
    );
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 1);
    let file = File::open(&path).unwrap();
    let mut buf = [0u8; 1];
    file.read_exact_at(&mut buf, 0).unwrap();
    assert_eq!(buf, [0x01]);

    let (_dir, path) = temp_fls();
    {
        FerroStore::open(&path).unwrap();
    }
    let created = slots(&path);
    assert!(created[0].is_some());
    assert!(created[1].is_none());
    // The header, then the snapshot that holds the new store's schema version.
    let first = created[0].unwrap();
    assert_eq!(first.snapshot_offset, 8192);
    assert!(first.snapshot_length > 0);
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        8192 + first.snapshot_length
    );
    corrupt_live_crc(&path);
    assert!(slots(&path).iter().all(Option::is_none));
    let file = File::open(&path).unwrap();
    let mut before = [0u8; 8192];
    file.read_exact_at(&mut before, 0).unwrap();
    drop(file);
    let err = match FerroStore::open(&path) {
        Ok(_) => panic!("open rewrote a header whose live crc was torn"),
        Err(err) => err,
    };
    let msg = err.to_string();
    assert!(
        matches!(err, StoreError::Backend(_)) && msg.contains("no valid FerroStore superblock"),
        "{msg}"
    );
    assert_eq!(
        std::fs::metadata(&path).unwrap().len(),
        8192 + first.snapshot_length
    );
    let file = File::open(&path).unwrap();
    let mut after = [0u8; 8192];
    file.read_exact_at(&mut after, 0).unwrap();
    assert_eq!(after, before);

    let (_dir, path) = temp_fls();
    {
        let file = File::create(&path).unwrap();
        file.write_all_at(&[0x00], 0).unwrap();
        file.sync_all().unwrap();
    }
    FerroStore::open(&path).unwrap();
    assert_eq!(slots(&path)[0].unwrap().epoch, 1);
    assert!(slots(&path)[1].is_none());
}

#[test]
fn parse_superblock_rejects_epoch_zero_and_illegal_extents() {
    let legal = Superblock {
        epoch: 1,
        snapshot_offset: 8192,
        snapshot_length: 0,
        wal_offset: 8192,
        wal_length: 0,
        snapshot_seq: 0,
    };
    assert_eq!(
        layout::parse_superblock(&layout::encode_superblock(&legal)).valid(),
        Some(legal)
    );

    let mut epoch_zero = legal;
    epoch_zero.epoch = 0;
    assert!(
        layout::parse_superblock(&layout::encode_superblock(&epoch_zero))
            .valid()
            .is_none()
    );

    let mut snapshot_short = legal;
    snapshot_short.snapshot_offset = 8191;
    assert!(
        layout::parse_superblock(&layout::encode_superblock(&snapshot_short))
            .valid()
            .is_none()
    );

    let mut wal_inside_header = legal;
    wal_inside_header.wal_offset = 0;
    wal_inside_header.wal_length = 0;
    assert!(
        layout::parse_superblock(&layout::encode_superblock(&wal_inside_header))
            .valid()
            .is_none()
    );

    let mut snapshot_wraps = legal;
    snapshot_wraps.snapshot_offset = 8192;
    snapshot_wraps.snapshot_length = u64::MAX;
    assert!(
        layout::parse_superblock(&layout::encode_superblock(&snapshot_wraps))
            .valid()
            .is_none()
    );

    let mut wal_wraps = legal;
    wal_wraps.wal_offset = 8192;
    wal_wraps.wal_length = u64::MAX;
    assert!(
        layout::parse_superblock(&layout::encode_superblock(&wal_wraps))
            .valid()
            .is_none()
    );
}

#[test]
fn apply_frames_skips_snapshot_prefix_and_rejects_bad_seqs() {
    let put = |id: &str| Op::Put {
        collection: "docs".into(),
        id: id.into(),
        doc: json!(id),
    };
    let assert_corrupt = |err: StoreError| {
        let msg = err.to_string();
        assert!(
            matches!(err, StoreError::Backend(_)) && msg.contains("corrupt"),
            "{msg}"
        );
    };

    let mut state = super::state::DocState::at_seq(0);
    let err = super::state::apply_frames(&mut state, 0, &[(0, vec![put("zero")])]).unwrap_err();
    assert_corrupt(err);
    assert!(state.docs.is_empty());
    assert_eq!(state.last_seq, 0);

    let mut state = super::state::DocState::at_seq(10);
    state
        .docs
        .insert("docs".into(), "kept".into(), json!("kept").into());
    let err = super::state::apply_frames(
        &mut state,
        10,
        &[(9, vec![put("down")]), (8, vec![put("lower")])],
    )
    .unwrap_err();
    assert_corrupt(err);
    assert_eq!(state.docs.get("docs", "kept"), Some(&json!("kept")));
    assert_eq!(state.docs.len(), 1);
    assert_eq!(state.last_seq, 10);

    let mut state = super::state::DocState::at_seq(10);
    state
        .docs
        .insert("docs".into(), "kept".into(), json!("kept").into());
    let err = super::state::apply_frames(
        &mut state,
        10,
        &[(8, vec![put("dup")]), (8, vec![put("again")])],
    )
    .unwrap_err();
    assert_corrupt(err);
    assert_eq!(state.docs.get("docs", "kept"), Some(&json!("kept")));
    assert_eq!(state.docs.len(), 1);
    assert_eq!(state.last_seq, 10);

    let mut state = super::state::DocState::at_seq(2);
    state
        .docs
        .insert("docs".into(), "kept".into(), json!("kept").into());
    super::state::apply_frames(
        &mut state,
        2,
        &[
            (1, vec![put("skip-a")]),
            (2, vec![put("skip-b")]),
            (3, vec![put("new")]),
        ],
    )
    .unwrap();
    assert_eq!(state.docs.get("docs", "kept"), Some(&json!("kept")));
    assert_eq!(state.docs.get("docs", "new"), Some(&json!("new")));
    assert!(!state.docs.contains("docs", "skip-a"));
    assert!(!state.docs.contains("docs", "skip-b"));
    assert_eq!(state.last_seq, 3);

    let mut state = super::state::DocState::at_seq(5);
    super::state::apply_frames(
        &mut state,
        5,
        &[(3, vec![put("old")]), (4, vec![put("older")])],
    )
    .unwrap();
    assert!(state.docs.is_empty());
    assert_eq!(state.last_seq, 5);

    let mut state = super::state::DocState::at_seq(5);
    let err = super::state::apply_frames(
        &mut state,
        5,
        &[(4, vec![put("skip")]), (7, vec![put("gap")])],
    )
    .unwrap_err();
    assert_corrupt(err);
    assert!(state.docs.is_empty());
    assert_eq!(state.last_seq, 5);

    let mut state = super::state::DocState::at_seq(0);
    super::state::apply_frames(
        &mut state,
        0,
        &[(1, vec![put("one")]), (2, vec![put("two")])],
    )
    .unwrap();
    assert_eq!(state.docs.get("docs", "one"), Some(&json!("one")));
    assert_eq!(state.docs.get("docs", "two"), Some(&json!("two")));
    assert_eq!(state.last_seq, 2);
}

#[test]
fn default_socket_path_for_store_names_the_socket_beside_the_file() {
    let path = default_socket_path_for_store("dir/store.fls");
    assert_eq!(path, PathBuf::from("dir/store.fls.sock"));
    assert!(path.ends_with("store.fls.sock"));
    assert_eq!(path.parent(), Some(Path::new("dir")));
    assert_eq!(
        default_socket_path_for_store("/"),
        PathBuf::from("/ferrostore.sock")
    );
    assert_eq!(
        default_socket_path_for_store(".."),
        PathBuf::from("../ferrostore.sock")
    );
}

#[test]
fn auto_uses_the_listening_socket_and_otherwise_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.fls");

    {
        let store = FerroStore::auto(&path).unwrap();
        store.put_doc("docs", "local", &json!({"n": 1})).unwrap();
    }
    let opened = FerroStore::open(&path).unwrap();
    assert_eq!(opened.get_doc("docs", "local").unwrap().unwrap()["n"], 1);
    drop(opened);

    // A socket file with nobody listening is a refused connect, not an error.
    let sock = default_socket_path_for_store(&path);
    drop(std::os::unix::net::UnixListener::bind(&sock).unwrap());
    {
        let store = FerroStore::auto(&path).unwrap();
        store.put_doc("docs", "stale", &json!({"n": 4})).unwrap();
    }
    let opened = FerroStore::open(&path).unwrap();
    assert_eq!(opened.get_doc("docs", "stale").unwrap().unwrap()["n"], 4);
    drop(opened);

    let server = FerroStoreServer::bind_unix(&sock, FerroStore::open(&path).unwrap()).unwrap();
    let client = FerroStore::auto(&path).unwrap();
    client.with_read(|_| Ok(())).unwrap();
    client.put_doc("docs", "remote", &json!({"n": 2})).unwrap();
    assert_eq!(
        server.store().get_doc("docs", "remote").unwrap().unwrap()["n"],
        2
    );
    drop(client);
    drop(server);

    let client = FerroStore::auto(&path).unwrap();
    assert_eq!(client.get_doc("docs", "remote").unwrap().unwrap()["n"], 2);
    client.put_doc("docs", "again", &json!({"n": 3})).unwrap();
    drop(client);
    let opened = FerroStore::open(&path).unwrap();
    assert_eq!(opened.get_doc("docs", "again").unwrap().unwrap()["n"], 3);
    assert_eq!(opened.get_doc("docs", "remote").unwrap().unwrap()["n"], 2);
}

fn exercise_store_traits(store: &FerroStore, tag: &str) {
    use fl_core::agent::{AgentRole, AreaRecord, RegisterAgentArgs};
    use fl_core::at::At;
    use fl_core::ferrowire::{MessageKind, MessageUrgency, SendWireArgs, WireAddress};
    use fl_core::ids::{WorkflowId, WorkflowRunId, seq_iri};
    use fl_core::model::State;
    use fl_core::state::{
        HandoffEntry, SensitivityPatch, StatePayload, StateScope, UpsertStateArgs,
    };
    use fl_core::store::{
        AgentRegistry, Catalog, MessageStore, StateStore, Tracker, WorkflowStore,
    };
    use fl_core::workflow::{
        AssigneeRule, DEFAULT_MAX_TOTAL_STEPS, DEFAULT_STEP_MAX_ATTEMPTS, DEFAULT_STEP_MAX_VISITS,
        END_STEP, EdgeCondition, EdgeDef, JoinPolicy, StepCompletion, StepDef, WorkflowDef,
        WorkflowRun,
    };
    use std::collections::BTreeMap;

    let root = format!("/work/{tag}");
    let project = store.add_project(&root).unwrap();
    let loaded = store.get_project(&project).unwrap().expect("project");
    assert_eq!(loaded.id, project);
    assert_eq!(loaded.root, root);

    let title = format!("title {tag}");
    let record = store.add_record(&project, &title).unwrap();
    store.set_record_state(&record, State::Doing).unwrap();
    let got = store.get_record(&record).unwrap().expect("record");
    assert_eq!(got.id, record);
    assert_eq!(got.state, State::Doing);
    assert_eq!(got.title, title);

    let area_name = format!("area-{tag}");
    store
        .upsert_area(
            AreaRecord::new(
                &area_name,
                Some(format!("{tag}/")),
                "Agent orchestration",
                At::from_unix_millis(1),
                "owner",
            )
            .unwrap(),
        )
        .unwrap();
    let agent_name = format!("agent-{tag}");
    let agent = store
        .register_agent(RegisterAgentArgs {
            display_name: agent_name.clone(),
            email: None,
            project: area_name.clone(),
            role: AgentRole::Lead,
            extra_areas: vec![],
            create_area: false,
            replace: false,
            at: At::from_unix_millis(2),
            by: "owner".into(),
        })
        .unwrap();
    let fetched = store.get_agent(&agent.id).unwrap().expect("agent");
    assert_eq!(fetched.id, agent.id);
    assert_eq!(fetched.display_name, agent_name);
    assert_eq!(fetched.project, area_name);

    let slug = format!("rule-{tag}");
    let state = store
        .upsert_state(UpsertStateArgs {
            scope: StateScope::Workspace,
            slug: slug.clone(),
            sensitivity: SensitivityPatch::Inherit,
            payload: StatePayload::Handoff(HandoffEntry {
                summary: "v6".into(),
                shipped: vec![],
                in_flight: vec![],
                blockers: vec![],
                next_steps: vec![],
                notes: None,
            }),
            at: At::from_unix_millis(10),
            by: "carol".into(),
        })
        .unwrap();
    let got_state = store.get_state(&state.id).unwrap().expect("state");
    assert_eq!(got_state.id, state.id);
    assert_eq!(got_state.slug, slug);

    let flow = format!("flow-{tag}");
    let workflow = store
        .upsert_workflow_def(WorkflowDef {
            id: WorkflowId(seq_iri(900)),
            project: project.clone(),
            slug: flow.clone(),
            title: format!("CI {tag}"),
            start_step: "build".into(),
            steps: vec![StepDef {
                step_id: "build".into(),
                title: "Build".into(),
                prompt: "Run build.".into(),
                assignee: AssigneeRule::Agent {
                    name: "alice".into(),
                },
                reviews_step: None,
                require_different_vendor: false,
                completion: StepCompletion::Ungated,
                join_policy: JoinPolicy::All,
                max_attempts: DEFAULT_STEP_MAX_ATTEMPTS,
                max_visits: DEFAULT_STEP_MAX_VISITS,
                budget_usd_micros: None,
                max_tokens: None,
                isolate_worktree: false,
            }],
            edges: vec![EdgeDef {
                from_step: "build".into(),
                to_step: END_STEP.into(),
                on: EdgeCondition::Pass,
            }],
            max_total_steps: DEFAULT_MAX_TOTAL_STEPS,
            budget_usd_micros: None,
            max_tokens: None,
            declared_keys: BTreeMap::new(),
            authored_at_commit: "0123456789abcdef0123456789abcdef01234567".into(),
            authored_by: "bob".into(),
        })
        .unwrap();
    let got_workflow = store
        .get_workflow_def(&workflow.id)
        .unwrap()
        .expect("workflow");
    assert_eq!(got_workflow.slug, flow);
    assert_eq!(got_workflow.project, project);

    let run = store
        .create_workflow_run(
            WorkflowRun::new(
                WorkflowRunId(seq_iri(901)),
                &got_workflow,
                format!("CI run {tag}"),
                At::from_unix_millis(20),
                "bob",
            )
            .unwrap(),
        )
        .unwrap();
    let got_run = store.get_workflow_run(&run.id).unwrap().expect("run");
    assert_eq!(got_run.id, run.id);
    assert_eq!(got_run.workflow_id, got_workflow.id);

    let import_def = got_workflow.clone();
    let import_run = got_run.clone();
    store.import_workflow_def(import_def).unwrap();
    store.import_workflow_run(import_run).unwrap();
    assert_eq!(
        store.get_workflow_def(&got_workflow.id).unwrap().as_ref(),
        Some(&got_workflow)
    );
    assert_eq!(
        store.get_workflow_run(&got_run.id).unwrap().as_ref(),
        Some(&got_run)
    );

    let body = format!("Body {tag}");
    let (message, _) = store
        .send_wire(SendWireArgs {
            id: None,
            sender_id: None,
            sender_name: "carol".into(),
            sender_host: Some("build-host".into()),
            to: WireAddress::AgentName(agent_name),
            kind: MessageKind::Mail,
            urgency: MessageUrgency::Normal,
            durable: false,
            subject: format!("Hello {tag}"),
            body: body.clone(),
            reply_to: None,
            thread_id: None,
            deadline_at: None,
            at: At::from_unix_millis(10),
        })
        .unwrap();
    let got_message = store
        .get_wire_message(&message.id)
        .unwrap()
        .expect("wire message");
    assert_eq!(got_message.id, message.id);
    assert_eq!(got_message.body, body);
}

fn assert_concurrent_increments(connect: impl Fn() -> FerroStore + Sync, path: &Path) {
    const THREADS: u64 = 8;
    thread::scope(|scope| {
        for i in 0..THREADS {
            let connect = &connect;
            scope.spawn(move || {
                let client = connect();
                client
                    .with_write(|txn| {
                        let n = txn
                            .get("counter", "n")
                            .and_then(|doc| doc.as_u64())
                            .unwrap_or(0);
                        txn.put("counter", "n", json!(n + 1));
                        txn.put("private", format!("t{i}"), json!(i));
                        Ok(())
                    })
                    .unwrap();
            });
        }
    });

    let client = connect();
    assert_eq!(
        client.get_doc("counter", "n").unwrap().unwrap(),
        json!(THREADS)
    );
    for i in 0..THREADS {
        assert_eq!(
            client
                .get_doc("private", &format!("t{i}"))
                .unwrap()
                .unwrap(),
            json!(i)
        );
    }

    let frames = committed_frames(path);
    assert_eq!(
        frames.len(),
        usize::try_from(THREADS).unwrap(),
        "{frames:?}"
    );
    let mut privates = BTreeMap::new();
    for (i, (seq, ops)) in frames.iter().enumerate() {
        let expect = i as u64 + 1;
        assert_eq!(*seq, expect);
        assert_eq!(ops.len(), 2, "seq {seq} was not one frame: {ops:?}");
        match &ops[..] {
            [
                Op::Put {
                    collection: counter_collection,
                    id: counter_id,
                    doc: counter,
                },
                Op::Put {
                    collection: private_collection,
                    id: private_id,
                    doc: private,
                },
            ] => {
                assert_eq!(counter_collection, "counter");
                assert_eq!(counter_id, "n");
                assert_eq!(counter, &json!(expect));
                assert_eq!(private_collection, "private");
                assert_eq!(private, &json!(private.as_u64().unwrap()));
                assert!(
                    privates
                        .insert(private_id.clone(), private.clone())
                        .is_none(),
                    "duplicate {private_id}"
                );
            }
            other => panic!("expected counter and private puts, got {other:?}"),
        }
    }
    assert_eq!(privates.len(), usize::try_from(THREADS).unwrap());
}

#[test]
fn store_traits_run_over_unix_ws_and_auto() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("traits.fls");
    let sock = dir.path().join("traits.sock");
    let server = FerroStoreServer::bind_unix(&sock, FerroStore::open(&file).unwrap()).unwrap();
    let client = FerroStore::connect_uri(&format!("unix://{}", sock.display())).unwrap();
    exercise_store_traits(&client, "unix");
    drop(client);
    drop(server);

    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("traits.fls");
    let server =
        FerroStoreServer::bind_tcp("127.0.0.1:0", FerroStore::open(&file).unwrap(), psk_auth())
            .unwrap();
    let port = server.local_addr().port();
    let client = connect_psk(&format!("ws://127.0.0.1:{port}")).unwrap();
    exercise_store_traits(&client, "ws");
    drop(client);
    drop(server);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("traits.fls");
    let server = FerroStoreServer::bind_unix(
        default_socket_path_for_store(&path),
        FerroStore::open(&path).unwrap(),
    )
    .unwrap();
    let client = FerroStore::auto(&path).unwrap();
    exercise_store_traits(&client, "auto");
    drop(client);
    drop(server);
}

#[test]
fn remote_with_write_commits_both_ops_in_one_frame() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("cas.fls");
    let sock = dir.path().join("cas.sock");
    let server = FerroStoreServer::bind_unix(&sock, FerroStore::open(&file).unwrap()).unwrap();
    let uri = format!("unix://{}", sock.display());
    assert_concurrent_increments(|| FerroStore::connect_uri(&uri).unwrap(), &file);
    drop(server);

    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("cas.fls");
    let server =
        FerroStoreServer::bind_tcp("127.0.0.1:0", FerroStore::open(&file).unwrap(), psk_auth())
            .unwrap();
    let port = server.local_addr().port();
    let uri = format!("ws://127.0.0.1:{port}");
    assert_concurrent_increments(|| connect_psk(&uri).unwrap(), &file);
    drop(server);

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cas.fls");
    let server = FerroStoreServer::bind_unix(
        default_socket_path_for_store(&path),
        FerroStore::open(&path).unwrap(),
    )
    .unwrap();
    assert_concurrent_increments(|| FerroStore::auto(&path).unwrap(), &path);
    drop(server);
}

fn int_column() -> Vec<ColumnDef> {
    vec![ColumnDef {
        name: "n".into(),
        ty: ColumnType::Int64,
    }]
}

fn eq_int(value: i64) -> ColumnFilter {
    ColumnFilter {
        column: "n".into(),
        predicate: ColumnPredicate::Eq(ColumnValue::Int64(value)),
    }
}

fn ints(values: &[i64]) -> Vec<Vec<ColumnValue>> {
    values
        .iter()
        .copied()
        .map(|value| vec![ColumnValue::Int64(value)])
        .collect()
}

fn snapshot_version(path: &Path) -> u32 {
    let sb = live(path);
    assert!(sb.snapshot_length > 0);
    let file = File::open(path).unwrap();
    let mut buf = [0u8; 4];
    file.read_exact_at(&mut buf, sb.snapshot_offset + 8)
        .unwrap();
    u32::from_le_bytes(buf)
}

#[test]
fn scan_skips_a_row_group_only_when_the_zone_map_excludes_it() {
    let store = FerroStore::in_memory();
    store.create_column_table("t", &int_column()).unwrap();
    store.append_column_rows("t", &ints(&[1, 2, 3])).unwrap();
    store.append_column_rows("t", &ints(&[10, 11, 12])).unwrap();

    let hit = store.scan_column_table("t", &[eq_int(10)]).unwrap();
    assert_eq!(hit.row_groups_total, 2);
    assert_eq!(hit.row_groups_skipped, 1);
    assert_eq!(hit.rows, ints(&[10]));

    let miss = store.scan_column_table("t", &[eq_int(100)]).unwrap();
    assert_eq!(miss.row_groups_total, 2);
    assert_eq!(miss.row_groups_skipped, 2);
    assert!(miss.rows.is_empty());

    let overlap = FerroStore::in_memory();
    overlap.create_column_table("t", &int_column()).unwrap();
    overlap.append_column_rows("t", &ints(&[1, 100])).unwrap();
    let inside = overlap.scan_column_table("t", &[eq_int(50)]).unwrap();
    assert_eq!(inside.row_groups_total, 1);
    assert_eq!(inside.row_groups_skipped, 0);
    assert!(inside.rows.is_empty());

    let err = store
        .scan_column_table(
            "t",
            &[ColumnFilter {
                column: "missing".into(),
                predicate: ColumnPredicate::Eq(ColumnValue::Int64(1)),
            }],
        )
        .unwrap_err();
    assert!(matches!(err, StoreError::Backend(_)), "{err}");
    let err = store
        .scan_column_table(
            "t",
            &[ColumnFilter {
                column: "n".into(),
                predicate: ColumnPredicate::Eq(ColumnValue::TimestampMs(1)),
            }],
        )
        .unwrap_err();
    assert!(matches!(err, StoreError::Backend(_)), "{err}");
}

#[test]
fn bad_width_and_a_second_create_write_nothing() {
    let (_dir, path) = temp_fls();
    let store = FerroStore::open(&path).unwrap();
    store.create_column_table("t", &int_column()).unwrap();
    store.append_column_rows("t", &ints(&[7])).unwrap();
    let frames = committed_frames(&path).len();
    assert_eq!(frames, 2);

    let err = store
        .append_column_rows("t", &[vec![ColumnValue::Int64(1), ColumnValue::Int64(2)]])
        .unwrap_err();
    assert!(matches!(err, StoreError::Backend(_)), "{err}");
    let err = store
        .append_column_rows("t", &[vec![ColumnValue::Utf8("no".into())]])
        .unwrap_err();
    assert!(matches!(err, StoreError::Backend(_)), "{err}");
    store.append_column_rows("t", &[]).unwrap();
    let err = store.create_column_table("t", &int_column()).unwrap_err();
    assert!(matches!(err, StoreError::Backend(_)), "{err}");
    let err = store
        .append_column_rows("missing", &ints(&[1]))
        .unwrap_err();
    assert!(matches!(err, StoreError::Backend(_)), "{err}");

    assert_eq!(committed_frames(&path).len(), frames);
    assert_eq!(store.scan_column_table("t", &[]).unwrap().rows, ints(&[7]));
}

#[test]
fn a_column_frame_that_does_not_apply_leaves_the_document_and_seq() {
    let duplicate = || {
        let create = Op::CreateColumnTable {
            name: "t".into(),
            columns: int_column(),
        };
        vec![create.clone(), create]
    };
    let exercise = |store: &FerroStore| {
        store
            .put_doc("docs", "a", &json!({ "kept": true }))
            .unwrap();
        let seq = store.read_state().unwrap().last_seq;
        let err = store.commit_if_seq(seq, &duplicate()).unwrap_err();
        assert!(matches!(err, StoreError::Backend(_)), "{err}");
        assert_eq!(
            store.get_doc("docs", "a").unwrap(),
            Some(json!({ "kept": true }))
        );
        assert_eq!(store.read_state().unwrap().last_seq, seq);
        let empty = Op::CreateColumnTable {
            name: String::new(),
            columns: int_column(),
        };
        let err = store.commit_if_seq(seq, &[empty]).unwrap_err();
        assert!(err.to_string().contains("empty"), "{err}");
        assert_eq!(store.read_state().unwrap().last_seq, seq);
        assert_eq!(
            store.get_doc("docs", "a").unwrap(),
            Some(json!({ "kept": true }))
        );
        store.create_column_table("t", &int_column()).unwrap();
        assert_eq!(store.read_state().unwrap().last_seq, seq + 1);
        assert!(store.scan_column_table("t", &[]).unwrap().rows.is_empty());
    };

    exercise(&FerroStore::in_memory());

    let (_dir, path) = temp_fls();
    let seq = {
        let store = FerroStore::open(&path).unwrap();
        store
            .put_doc("docs", "a", &json!({ "kept": true }))
            .unwrap();
        let seq = store.read_state().unwrap().last_seq;
        let frames = committed_frames(&path).len();
        let err = store.commit_if_seq(seq, &duplicate()).unwrap_err();
        assert!(matches!(err, StoreError::Backend(_)), "{err}");
        assert_eq!(
            store.get_doc("docs", "a").unwrap(),
            Some(json!({ "kept": true }))
        );
        assert_eq!(store.read_state().unwrap().last_seq, seq);
        assert_eq!(committed_frames(&path).len(), frames);
        seq
    };
    {
        let store = FerroStore::open(&path).unwrap();
        assert_eq!(
            store.get_doc("docs", "a").unwrap(),
            Some(json!({ "kept": true }))
        );
        assert_eq!(store.read_state().unwrap().last_seq, seq);
        store.create_column_table("t", &int_column()).unwrap();
        assert!(store.scan_column_table("t", &[]).unwrap().rows.is_empty());
    }
    let store = FerroStore::open(&path).unwrap();
    assert_eq!(
        store.get_doc("docs", "a").unwrap(),
        Some(json!({ "kept": true }))
    );
    assert!(store.scan_column_table("t", &[]).unwrap().rows.is_empty());
}

#[test]
fn column_rows_survive_reopen_and_a_version_2_checkpoint() {
    let (_dir, path) = temp_fls();
    {
        let store = FerroStore::open(&path).unwrap();
        store.put_doc("docs", "a", &json!(1)).unwrap();
        store.checkpoint().unwrap();
        assert_eq!(snapshot_version(&path), 1);
        store.create_column_table("t", &int_column()).unwrap();
        store.append_column_rows("t", &ints(&[1, 2, 3])).unwrap();
        store.append_column_rows("t", &ints(&[10])).unwrap();
    }
    {
        let store = FerroStore::open(&path).unwrap();
        assert_eq!(store.get_doc("docs", "a").unwrap(), Some(json!(1)));
        let scan = store.scan_column_table("t", &[eq_int(10)]).unwrap();
        assert_eq!(scan.rows, ints(&[10]));
        assert_eq!(scan.row_groups_total, 2);
        assert_eq!(scan.row_groups_skipped, 1);
        store.checkpoint().unwrap();
        assert_eq!(snapshot_version(&path), 2);
        assert_eq!(live(&path).wal_length, 0);
    }
    let store = FerroStore::open(&path).unwrap();
    assert_eq!(store.get_doc("docs", "a").unwrap(), Some(json!(1)));
    assert_eq!(
        store.scan_column_table("t", &[]).unwrap().rows,
        ints(&[1, 2, 3, 10])
    );
}

fn exercise_served_columns(store: &FerroStore) {
    store.create_column_table("t", &int_column()).unwrap();
    store.append_column_rows("t", &ints(&[1, 2])).unwrap();
    store.append_column_rows("t", &ints(&[9, 10])).unwrap();
    let scan = store.scan_column_table("t", &[eq_int(9)]).unwrap();
    assert_eq!(scan.row_groups_total, 2);
    assert_eq!(scan.row_groups_skipped, 1);
    assert_eq!(scan.rows, ints(&[9]));
    let miss = store.scan_column_table("t", &[eq_int(100)]).unwrap();
    assert_eq!(miss.row_groups_skipped, 2);
    assert!(miss.rows.is_empty());
    store.put_doc("docs", "kept", &json!(null)).unwrap();
    assert_eq!(store.get_doc("docs", "kept").unwrap(), Some(json!(null)));
    assert!(store.get_doc("docs", "missing").unwrap().is_none());
}

#[test]
fn served_column_scan_works_over_unix_and_ws_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("cols.fls");
    let sock = dir.path().join("cols.sock");
    let server = FerroStoreServer::bind_unix(&sock, FerroStore::open(&file).unwrap()).unwrap();
    let client = FerroStore::connect_uri(&format!("unix://{}", sock.display())).unwrap();
    exercise_served_columns(&client);
    drop(client);
    drop(server);
    let reopened = FerroStore::open(&file).unwrap();
    assert_eq!(
        reopened.scan_column_table("t", &[eq_int(9)]).unwrap().rows,
        ints(&[9])
    );
    assert_eq!(reopened.get_doc("docs", "kept").unwrap(), Some(json!(null)));

    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("cols.fls");
    let server =
        FerroStoreServer::bind_tcp("127.0.0.1:0", FerroStore::open(&file).unwrap(), psk_auth())
            .unwrap();
    let port = server.local_addr().port();
    let client = connect_psk(&format!("ws://127.0.0.1:{port}")).unwrap();
    exercise_served_columns(&client);
    drop(client);
    drop(server);
    let reopened = FerroStore::open(&file).unwrap();
    let scan = reopened.scan_column_table("t", &[eq_int(9)]).unwrap();
    assert_eq!(scan.row_groups_total, 2);
    assert_eq!(scan.row_groups_skipped, 1);
    assert_eq!(scan.rows, ints(&[9]));
}

fn sample_attempt(
    project: ProjectId,
    cost: u64,
    tokens_in: u64,
    tokens_out: u64,
    duration_ms: u64,
) -> Attempt {
    Attempt {
        id: None,
        at: None,
        project,
        record: RecordId(seq_iri(9)),
        adapter: "claude".into(),
        status: AttemptStatus::Completed,
        duration_ms,
        tokens_in,
        tokens_out,
        cost_usd_micros: cost,
        paths_touched: PathsTouched::Listed(vec![]),
        output_excerpt: None,
    }
}

fn sample_error_run(gate: GateId, duration_ms: u64, cost_usd_micros: u64) -> GateRun {
    GateRun {
        id: None,
        at: None,
        gate,
        record: None,
        commit: "abc".into(),
        verdict: Verdict::error("boom"),
        population: 1,
        output_excerpt: None,
        duration_ms,
        cost_usd_micros,
    }
}

fn sample_run(gate: GateId, pass: bool, duration_ms: u64, cost_usd_micros: u64) -> GateRun {
    GateRun {
        id: None,
        at: None,
        gate,
        record: None,
        commit: "abc".into(),
        verdict: Verdict::from_predicate(pass, 1),
        population: 1,
        output_excerpt: None,
        duration_ms,
        cost_usd_micros,
    }
}

fn assert_rollups(store: &FerroStore) {
    let project_a = ProjectId(seq_iri(1));
    let project_b = ProjectId(seq_iri(2));
    let gate_a = GateId(seq_iri(3));
    let gate_b = GateId(seq_iri(4));
    store
        .append_attempt(sample_attempt(project_a.clone(), 10, 1, 2, 3))
        .unwrap();
    store
        .append_attempt(sample_attempt(project_b.clone(), 100, 4, 5, 6))
        .unwrap();
    store
        .append_gate_run(sample_run(gate_a.clone(), true, 7, 8))
        .unwrap();
    store
        .append_gate_run(sample_run(gate_b.clone(), false, 9, 10))
        .unwrap();
    store
        .append_gate_run(sample_error_run(gate_a.clone(), 4, 5))
        .unwrap();

    assert_eq!(
        store.attempt_rollup(None).unwrap(),
        AttemptRollup {
            count: 2,
            cost_usd_micros: 110,
            tokens_in: 5,
            tokens_out: 7,
            duration_ms: 9,
        }
    );
    assert_eq!(
        store.attempt_rollup(Some(&project_a)).unwrap(),
        AttemptRollup {
            count: 1,
            cost_usd_micros: 10,
            tokens_in: 1,
            tokens_out: 2,
            duration_ms: 3,
        }
    );
    assert_eq!(store.attempt_rollup(Some(&project_b)).unwrap().count, 1);
    assert_eq!(
        store.gate_run_rollup(None).unwrap(),
        GateRunRollup {
            pass: 1,
            fail: 1,
            error: 1,
            duration_ms: 20,
            cost_usd_micros: 23,
        }
    );
    assert_eq!(
        store.gate_run_rollup(Some(&gate_a)).unwrap(),
        GateRunRollup {
            pass: 1,
            fail: 0,
            error: 1,
            duration_ms: 11,
            cost_usd_micros: 13,
        }
    );
    assert_eq!(store.gate_run_rollup(Some(&gate_b)).unwrap().fail, 1);
    assert_eq!(store.gate_run_rollup(Some(&gate_b)).unwrap().pass, 0);
}

#[test]
fn attempt_and_gate_rollups_match_hand_sums_locally_and_over_unix() {
    assert_rollups(&FerroStore::in_memory());

    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("roll.fls");
    let sock = dir.path().join("roll.sock");
    let server = FerroStoreServer::bind_unix(&sock, FerroStore::open(&file).unwrap()).unwrap();
    let client = FerroStore::connect_uri(&format!("unix://{}", sock.display())).unwrap();
    assert_rollups(&client);
    drop(client);
    drop(server);

    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("roll.fls");
    let server =
        FerroStoreServer::bind_tcp("127.0.0.1:0", FerroStore::open(&file).unwrap(), psk_auth())
            .unwrap();
    let client = connect_psk(&format!("ws://{}", server.local_addr())).unwrap();
    assert_rollups(&client);
    drop(client);
    drop(server);
}

fn float_column() -> Vec<ColumnDef> {
    vec![ColumnDef {
        name: "f".into(),
        ty: ColumnType::Float64,
    }]
}

fn float_bits(value: &ColumnValue) -> u64 {
    match value {
        ColumnValue::Float64(number) => number.to_bits(),
        other => panic!("expected f64, got {other:?}"),
    }
}

fn float_filter(predicate: ColumnPredicate) -> ColumnFilter {
    ColumnFilter {
        column: "f".into(),
        predicate,
    }
}

fn assert_non_finite_round_trip(store: &FerroStore) {
    let payload = f64::from_bits(0x7ff8_0000_0000_0001);
    assert_ne!(payload.to_bits(), f64::NAN.to_bits());
    let rows = vec![
        vec![ColumnValue::Float64(payload)],
        vec![ColumnValue::Float64(f64::INFINITY)],
        vec![ColumnValue::Float64(f64::NEG_INFINITY)],
        vec![ColumnValue::Float64(1.5)],
    ];
    store.create_column_table("f", &float_column()).unwrap();
    store.append_column_rows("f", &rows).unwrap();
    let scan = store.scan_column_table("f", &[]).unwrap();
    let got = scan
        .rows
        .iter()
        .map(|row| float_bits(&row[0]))
        .collect::<Vec<_>>();
    assert_eq!(
        got,
        vec![
            payload.to_bits(),
            f64::INFINITY.to_bits(),
            f64::NEG_INFINITY.to_bits(),
            1.5f64.to_bits(),
        ]
    );
    let nan = store
        .scan_column_table(
            "f",
            &[float_filter(ColumnPredicate::Eq(ColumnValue::Float64(
                f64::NAN,
            )))],
        )
        .unwrap();
    assert_eq!(nan.row_groups_total, 1);
    assert_eq!(nan.row_groups_skipped, 1);
    assert!(nan.rows.is_empty());
    let inverted = store
        .scan_column_table(
            "f",
            &[float_filter(ColumnPredicate::Between {
                low: ColumnValue::Float64(2.0),
                high: ColumnValue::Float64(1.0),
            })],
        )
        .unwrap();
    assert_eq!(inverted.row_groups_skipped, 1);
    assert!(inverted.rows.is_empty());
    let finite = store
        .scan_column_table(
            "f",
            &[float_filter(ColumnPredicate::Eq(ColumnValue::Float64(1.5)))],
        )
        .unwrap();
    assert_eq!(finite.row_groups_skipped, 0);
    assert_eq!(finite.rows.len(), 1);
    assert_eq!(float_bits(&finite.rows[0][0]), 1.5f64.to_bits());
}

#[test]
fn negative_zero_compares_equal_and_keeps_its_bits_across_reopen() {
    let payload = f64::from_bits(0x7ff8_0000_0000_0001);
    assert_ne!(payload.to_bits(), f64::NAN.to_bits());
    let neg = -0.0f64;
    let (_dir, path) = temp_fls();
    {
        let store = FerroStore::open(&path).unwrap();
        store.create_column_table("f", &float_column()).unwrap();
        store
            .append_column_rows(
                "f",
                &[
                    vec![ColumnValue::Float64(neg)],
                    vec![ColumnValue::Float64(payload)],
                ],
            )
            .unwrap();
    }
    let store = FerroStore::open(&path).unwrap();
    let scan = store.scan_column_table("f", &[]).unwrap();
    assert_eq!(float_bits(&scan.rows[0][0]), neg.to_bits());
    assert_eq!(float_bits(&scan.rows[1][0]), payload.to_bits());
    for predicate in [
        ColumnPredicate::Eq(ColumnValue::Float64(0.0)),
        ColumnPredicate::Le(ColumnValue::Float64(0.0)),
        ColumnPredicate::Ge(ColumnValue::Float64(0.0)),
    ] {
        let hit = store
            .scan_column_table("f", &[float_filter(predicate)])
            .unwrap();
        assert_eq!(hit.row_groups_skipped, 0);
        assert_eq!(hit.rows.len(), 1);
        assert_eq!(float_bits(&hit.rows[0][0]), neg.to_bits());
    }
}

#[test]
fn non_finite_floats_round_trip_locally_and_over_unix_and_ws() {
    assert_non_finite_round_trip(&FerroStore::in_memory());

    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("floats.fls");
    let sock = dir.path().join("floats.sock");
    let server = FerroStoreServer::bind_unix(&sock, FerroStore::open(&file).unwrap()).unwrap();
    let client = FerroStore::connect_uri(&format!("unix://{}", sock.display())).unwrap();
    assert_non_finite_round_trip(&client);
    drop(client);
    drop(server);

    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("floats.fls");
    let server =
        FerroStoreServer::bind_tcp("127.0.0.1:0", FerroStore::open(&file).unwrap(), psk_auth())
            .unwrap();
    let client = connect_psk(&format!("ws://{}", server.local_addr())).unwrap();
    assert_non_finite_round_trip(&client);
    drop(client);
    drop(server);
}

#[test]
fn le_and_ge_together_skip_a_group_only_when_a_zone_excludes_it() {
    let store = FerroStore::in_memory();
    store.create_column_table("t", &int_column()).unwrap();
    store.append_column_rows("t", &ints(&[1, 2, 3])).unwrap();
    store.append_column_rows("t", &ints(&[10, 20])).unwrap();
    let scan = store
        .scan_column_table(
            "t",
            &[
                ColumnFilter {
                    column: "n".into(),
                    predicate: ColumnPredicate::Le(ColumnValue::Int64(5)),
                },
                ColumnFilter {
                    column: "n".into(),
                    predicate: ColumnPredicate::Ge(ColumnValue::Int64(2)),
                },
            ],
        )
        .unwrap();
    assert_eq!(scan.row_groups_total, 2);
    assert_eq!(scan.row_groups_skipped, 1);
    assert_eq!(scan.rows, ints(&[2, 3]));

    let overlap = FerroStore::in_memory();
    overlap.create_column_table("t", &int_column()).unwrap();
    overlap.append_column_rows("t", &ints(&[1, 10])).unwrap();
    let scan = overlap
        .scan_column_table(
            "t",
            &[
                ColumnFilter {
                    column: "n".into(),
                    predicate: ColumnPredicate::Le(ColumnValue::Int64(2)),
                },
                ColumnFilter {
                    column: "n".into(),
                    predicate: ColumnPredicate::Ge(ColumnValue::Int64(9)),
                },
            ],
        )
        .unwrap();
    assert_eq!(scan.row_groups_total, 1);
    assert_eq!(scan.row_groups_skipped, 0);
    assert!(scan.rows.is_empty());
}

#[test]
fn a_row_group_that_does_not_decode_is_not_written() {
    let store = FerroStore::in_memory();
    store
        .put_doc("docs", "a", &json!({ "kept": true }))
        .unwrap();
    store.create_column_table("t", &int_column()).unwrap();
    let seq = store.read_state().unwrap().last_seq;
    let err = store
        .commit_if_seq(
            seq,
            &[Op::AppendColumnGroup {
                name: "t".into(),
                bytes: vec![0, 0, 0, 0],
            }],
        )
        .unwrap_err();
    assert!(err.to_string().contains("corrupt"), "{err}");
    assert_eq!(store.read_state().unwrap().last_seq, seq);
    assert!(store.scan_column_table("t", &[]).unwrap().rows.is_empty());
    assert_eq!(
        store.get_doc("docs", "a").unwrap(),
        Some(json!({ "kept": true }))
    );
}

#[test]
fn a_version_2_snapshot_with_no_tables_or_a_bad_group_is_corrupt() {
    let docs = super::state::Docs::default();
    let tables = super::state::Tables::new();
    let blob = layout::encode_snapshot_v2(1, &docs, &tables).unwrap();
    let err = match layout::decode_snapshot(&blob) {
        Ok(_) => panic!("version-2 snapshot with no tables opened"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("no column tables"), "{err}");

    let mut tables = super::state::Tables::new();
    tables.insert(
        "t".into(),
        super::columnar::ColumnTable {
            columns: int_column(),
            groups: vec![vec![0, 0, 0, 0]],
        }
        .into(),
    );
    let blob = layout::encode_snapshot_v2(1, &docs, &tables).unwrap();
    let err = match layout::decode_snapshot(&blob) {
        Ok(_) => panic!("version-2 snapshot with a bad row group opened"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("corrupt"), "{err}");

    let v1 = layout::encode_snapshot(1, &docs).unwrap();
    let decoded = layout::decode_snapshot(&v1).unwrap();
    assert!(decoded.tables.is_empty());
    assert!(decoded.vectors.is_empty());
}

fn snapshot_bytes(path: &Path) -> Vec<u8> {
    let sb = live(path);
    let file = File::open(path).unwrap();
    let mut buf = vec![0u8; usize::try_from(sb.snapshot_length).unwrap()];
    if !buf.is_empty() {
        file.read_exact_at(&mut buf, sb.snapshot_offset).unwrap();
    }
    buf
}

fn nn_vector(index: usize) -> Vec<f32> {
    let center = index % 4;
    let mut values = vec![0.0; 8];
    values[center] = 1.0;
    values[4 + (index % 4)] = 0.05 + 0.01 * (index as f32);
    values
}

fn cosine_score(left: &[f32], right: &[f32]) -> f64 {
    let dot: f64 = left
        .iter()
        .zip(right)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum();
    let left_norm = left
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    let right_norm = right
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        .sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot / (left_norm * right_norm)
    }
}

fn brute_cosine(stored: &[(String, Vec<f32>)], query: &[f32], limit: usize) -> Vec<(String, f64)> {
    let mut scored: Vec<(String, f64)> = stored
        .iter()
        .map(|(id, values)| (id.clone(), cosine_score(values, query)))
        .collect();
    scored.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    scored.truncate(limit);
    scored
}

fn rrf_scores(bm25: &[(String, f64)], vectors: &[(String, f64)], k: u32) -> Vec<(String, f64)> {
    let mut scores: BTreeMap<String, f64> = BTreeMap::new();
    for (rank, (id, _)) in bm25.iter().enumerate() {
        *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (f64::from(k) + rank as f64 + 1.0);
    }
    for (rank, (id, _)) in vectors.iter().enumerate() {
        *scores.entry(id.clone()).or_insert(0.0) += 1.0 / (f64::from(k) + rank as f64 + 1.0);
    }
    let mut fused: Vec<(String, f64)> = scores.into_iter().collect();
    fused.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    fused
}

fn normalize_hits(hits: &[(String, f64)]) -> BTreeMap<String, f64> {
    if hits.is_empty() {
        return BTreeMap::new();
    }
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for (_, score) in hits {
        min = min.min(*score);
        max = max.max(*score);
    }
    let mut out = BTreeMap::new();
    for (id, score) in hits {
        let norm = if max == min {
            1.0
        } else {
            (score - min) / (max - min)
        };
        out.insert(id.clone(), norm);
    }
    out
}

fn weighted_scores(
    bm25: &[(String, f64)],
    vectors: &[(String, f64)],
    bm25_weight: f64,
    vector_weight: f64,
) -> Vec<(String, f64)> {
    let left = normalize_hits(bm25);
    let right = normalize_hits(vectors);
    let mut ids = BTreeSet::new();
    ids.extend(left.keys().cloned());
    ids.extend(right.keys().cloned());
    let mut fused: Vec<(String, f64)> = ids
        .into_iter()
        .map(|id| {
            let score = bm25_weight * left.get(&id).copied().unwrap_or(0.0)
                + vector_weight * right.get(&id).copied().unwrap_or(0.0);
            (id, score)
        })
        .collect();
    fused.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    fused
}

fn nan_f32_pair() -> Vec<u8> {
    let mut bytes = 1.0f32.to_le_bytes().to_vec();
    bytes.extend_from_slice(&f32::NAN.to_le_bytes());
    bytes
}

fn assert_vector_rejects_write_nothing(store: &FerroStore) {
    store
        .create_vector_collection("emb", 2, VectorStorage::F32, VectorMetric::Cosine, 4, 8, 8)
        .unwrap();
    store
        .create_vector_collection("half", 2, VectorStorage::F16, VectorMetric::L2, 4, 8, 8)
        .unwrap();
    store.upsert_vector("emb", "ok", &[1.0, 0.0], None).unwrap();
    let seq = store.read_state().unwrap().last_seq;
    let err = store.upsert_vector("emb", "bad", &[1.0], None).unwrap_err();
    assert!(matches!(err, StoreError::Backend(_)), "{err}");
    let err = store
        .upsert_vector("emb", "bad", &[f32::NAN, 0.0], None)
        .unwrap_err();
    assert!(err.to_string().contains("not finite"), "{err}");
    let err = store
        .upsert_vector("emb", "bad", &[f32::INFINITY, 0.0], None)
        .unwrap_err();
    assert!(matches!(err, StoreError::Backend(_)), "{err}");
    let err = store
        .upsert_vector("emb", "", &[1.0, 0.0], None)
        .unwrap_err();
    assert!(err.to_string().contains("vector id is empty"), "{err}");
    let err = store
        .upsert_vector("emb", "a\0b", &[1.0, 0.0], None)
        .unwrap_err();
    assert!(err.to_string().contains("invalid vector id"), "{err}");
    let err = store
        .create_vector_collection("emb", 2, VectorStorage::F32, VectorMetric::Cosine, 4, 8, 8)
        .unwrap_err();
    assert!(err.to_string().contains("already exists"), "{err}");
    let err = store
        .create_vector_collection("tiny", 2, VectorStorage::F32, VectorMetric::Cosine, 1, 8, 8)
        .unwrap_err();
    assert!(err.to_string().contains("M is out of range"), "{err}");
    let err = store
        .create_vector_collection("", 2, VectorStorage::F32, VectorMetric::Cosine, 4, 8, 8)
        .unwrap_err();
    assert!(
        err.to_string().contains("vector collection name is empty"),
        "{err}"
    );
    let err = store
        .create_vector_collection("a/b", 2, VectorStorage::F32, VectorMetric::Cosine, 4, 8, 8)
        .unwrap_err();
    assert!(
        err.to_string().contains("invalid vector collection name"),
        "{err}"
    );
    let err = store
        .upsert_vector("half", "wide", &[100_000.0, 0.0], None)
        .unwrap_err();
    assert!(err.to_string().contains("out of range"), "{err}");
    let err = store
        .upsert_vector("missing", "a", &[1.0, 0.0], None)
        .unwrap_err();
    assert!(err.to_string().contains("no vector collection"), "{err}");
    let err = store.delete_vector("missing", "a").unwrap_err();
    assert!(err.to_string().contains("no vector collection"), "{err}");
    assert_eq!(store.read_state().unwrap().last_seq, seq);
    assert_eq!(
        store.get_vector("emb", "ok").unwrap().unwrap().0,
        vec![1.0, 0.0]
    );
    assert!(store.get_vector("emb", "bad").unwrap().is_none());
    assert!(!store.read_state().unwrap().vectors.contains_key("tiny"));
}

fn exercise_delete(store: &FerroStore) {
    store
        .create_vector_collection("emb", 2, VectorStorage::F32, VectorMetric::Cosine, 4, 8, 8)
        .unwrap();
    store
        .upsert_vector("emb", "keep", &[1.0, 0.0], None)
        .unwrap();
    store
        .upsert_vector("emb", "drop", &[0.0, 1.0], None)
        .unwrap();
    let before = store.search_vectors("emb", &[0.0, 1.0], 5, None).unwrap();
    assert_eq!(before[0].0, "drop");
    store.delete_vector("emb", "drop").unwrap();
    let after = store.search_vectors("emb", &[0.0, 1.0], 5, None).unwrap();
    assert!(after.iter().all(|(id, _)| id != "drop"));
    assert_eq!(after[0].0, "keep");
    let seq = store.read_state().unwrap().last_seq;
    store.delete_vector("emb", "missing").unwrap();
    assert_eq!(store.read_state().unwrap().last_seq, seq + 1);
    store
        .upsert_vector("emb", "drop", &[0.0, 1.0], None)
        .unwrap();
    let back = store.search_vectors("emb", &[0.0, 1.0], 5, None).unwrap();
    assert_eq!(back[0].0, "drop");
    assert!(back.iter().all(|(_, score)| score.is_finite()));
}

fn exercise_bad_batch(store: &FerroStore) -> u64 {
    store
        .put_doc("docs", "earlier", &json!({ "kept": true }))
        .unwrap();
    store
        .create_vector_collection("emb", 2, VectorStorage::F32, VectorMetric::Cosine, 4, 8, 8)
        .unwrap();
    let seq = store.read_state().unwrap().last_seq;
    let err = store
        .commit_if_seq(
            seq,
            &[
                Op::Put {
                    collection: "docs".into(),
                    id: "later".into(),
                    doc: json!({ "no": true }),
                },
                Op::UpsertVector {
                    collection: "emb".into(),
                    id: "bad".into(),
                    bytes: nan_f32_pair(),
                    metadata: None,
                },
            ],
        )
        .unwrap_err();
    assert!(matches!(err, StoreError::Backend(_)), "{err}");
    assert_eq!(store.read_state().unwrap().last_seq, seq);
    assert_eq!(
        store.get_doc("docs", "earlier").unwrap(),
        Some(json!({ "kept": true }))
    );
    assert!(store.get_doc("docs", "later").unwrap().is_none());
    assert!(store.get_vector("emb", "bad").unwrap().is_none());
    seq
}

fn exercise_vector_rpc(client: &FerroStore, file: &Path) {
    client
        .create_vector_collection(
            "emb",
            2,
            VectorStorage::F32,
            VectorMetric::Cosine,
            4,
            16,
            16,
        )
        .unwrap();
    client
        .put_doc("notes", "both", &json!({ "text": "alpha widget" }))
        .unwrap();
    client
        .upsert_vector("emb", "both", &[1.0, 0.0], Some(&json!({ "k": 1 })))
        .unwrap();
    client
        .upsert_vector("emb", "only", &[0.2, 0.8], None)
        .unwrap();
    client
        .upsert_vector(
            "emb",
            "null-meta",
            &[0.0, 1.0],
            Some(&serde_json::Value::Null),
        )
        .unwrap();
    let (values, metadata) = client.get_vector("emb", "both").unwrap().unwrap();
    assert_eq!(values, vec![1.0, 0.0]);
    assert_eq!(metadata, Some(json!({ "k": 1 })));
    assert!(client.get_vector("emb", "missing").unwrap().is_none());
    assert!(
        client
            .get_vector("emb", "only")
            .unwrap()
            .unwrap()
            .1
            .is_none()
    );
    assert_eq!(
        client.get_vector("emb", "null-meta").unwrap().unwrap().1,
        Some(serde_json::Value::Null)
    );
    let found = client.search_vectors("emb", &[1.0, 0.0], 3, None).unwrap();
    assert_eq!(found[0].0, "both");
    assert_eq!(found[0].1, 1.0);
    assert!(found.iter().all(|(_, score)| score.is_finite()));
    assert!(
        client
            .search_vectors("emb", &[1.0, 0.0], 0, None)
            .unwrap()
            .is_empty()
    );
    assert!(
        client
            .search_vectors("emb", &[1.0, 0.0], 3, Some(0))
            .unwrap()
            .is_empty()
    );
    let missing = client
        .search_vectors("missing", &[1.0, 0.0], 0, None)
        .unwrap_err();
    assert!(matches!(missing, StoreError::Backend(_)), "{missing}");
    let missing_ef = client
        .search_vectors("missing", &[1.0, 0.0], 3, Some(0))
        .unwrap_err();
    assert!(matches!(missing_ef, StoreError::Backend(_)), "{missing_ef}");
    let hybrid = client
        .search_hybrid(
            "notes",
            "emb",
            "alpha",
            &[1.0, 0.0],
            5,
            Fusion::Rrf { k: 60 },
        )
        .unwrap();
    assert_eq!(hybrid[0].0, "both");
    let knowledge = client
        .search_knowledge(
            "notes",
            "emb",
            "alpha",
            &[1.0, 0.0],
            5,
            Fusion::Rrf { k: 60 },
        )
        .unwrap();
    let both = knowledge.iter().find(|hit| hit.id == "both").unwrap();
    assert_eq!(both.metadata, Some(json!({ "k": 1 })));
    assert_eq!(both.doc, Some(json!({ "text": "alpha widget" })));
    let only = knowledge.iter().find(|hit| hit.id == "only").unwrap();
    assert!(only.metadata.is_none());
    assert!(only.doc.is_none());
    assert!(knowledge.iter().all(|hit| hit.score.is_finite()));
    let rrf = Fusion::Rrf { k: 60 };
    assert!(
        client
            .search_hybrid("notes", "emb", "alpha", &[1.0, 0.0], 0, rrf)
            .unwrap()
            .is_empty()
    );
    assert!(
        client
            .search_knowledge("notes", "emb", "alpha", &[1.0, 0.0], 0, rrf)
            .unwrap()
            .is_empty()
    );
    assert!(
        client
            .search_hybrid("no-such-docs", "emb", "alpha", &[1.0, 0.0], 0, rrf)
            .unwrap()
            .is_empty()
    );
    for search in [
        client
            .search_hybrid("notes", "missing", "alpha", &[1.0, 0.0], 0, rrf)
            .unwrap_err(),
        client
            .search_knowledge("notes", "missing", "alpha", &[1.0, 0.0], 0, rrf)
            .unwrap_err(),
        client
            .search_hybrid("notes", "emb", "alpha", &[1.0], 0, rrf)
            .unwrap_err(),
        client
            .search_knowledge("notes", "emb", "alpha", &[1.0], 0, rrf)
            .unwrap_err(),
        client
            .search_hybrid(
                "notes",
                "emb",
                "alpha",
                &[1.0, 0.0],
                0,
                Fusion::Rrf { k: 0 },
            )
            .unwrap_err(),
        client
            .search_knowledge(
                "notes",
                "emb",
                "alpha",
                &[1.0, 0.0],
                0,
                Fusion::Rrf { k: 0 },
            )
            .unwrap_err(),
    ] {
        assert!(matches!(search, StoreError::Backend(_)), "{search}");
    }
    // The client never receives the whole state (spec §6.1); the file the
    // server wrote holds the graph its upserts built.
    assert!(client.read_state().is_err());
    let disk = FerroStore::open(file).unwrap().read_state().unwrap();
    assert_eq!(client.last_seq().unwrap(), disk.last_seq);
    assert!(disk.vectors["emb"].entry.is_some());
    assert!(
        disk.vectors["emb"]
            .vectors
            .values()
            .any(|record| record.neighbors.iter().any(|layer| !layer.is_empty()))
    );
}

#[test]
fn f32_f16_and_i8_vectors_round_trip_search_and_share_a_document_name() {
    let store = FerroStore::in_memory();
    store
        .create_vector_collection("f32", 4, VectorStorage::F32, VectorMetric::Cosine, 4, 8, 8)
        .unwrap();
    store
        .upsert_vector("f32", "near", &[1.0, 0.0, 0.0, 0.0], None)
        .unwrap();
    store
        .upsert_vector("f32", "far", &[0.0, 1.0, 0.0, 0.0], None)
        .unwrap();
    store
        .upsert_vector("f32", "a/b", &[0.0, 0.0, 1.0, 0.0], None)
        .unwrap();
    store
        .put_doc("f32", "near", &json!({ "from": "doc" }))
        .unwrap();
    assert_eq!(
        store.get_doc("f32", "near").unwrap(),
        Some(json!({ "from": "doc" }))
    );
    assert_eq!(
        store.get_vector("f32", "near").unwrap().unwrap().0,
        vec![1.0, 0.0, 0.0, 0.0]
    );
    assert!(store.get_vector("f32", "missing").unwrap().is_none());
    let hits = store
        .search_vectors("f32", &[1.0, 0.0, 0.0, 0.0], 3, None)
        .unwrap();
    assert_eq!(hits[0].0, "near");
    assert_eq!(hits[0].1, 1.0);
    assert!(hits.iter().any(|(id, score)| id == "far" && *score == 0.0));
    assert!(hits.iter().all(|(_, score)| score.is_finite()));

    store
        .create_vector_collection("f16", 3, VectorStorage::F16, VectorMetric::L2, 4, 8, 8)
        .unwrap();
    store
        .upsert_vector("f16", "near", &[1.0, 0.5, -2.0], None)
        .unwrap();
    store
        .upsert_vector("f16", "far", &[0.5, 0.5, 0.5], None)
        .unwrap();
    assert_eq!(
        store.get_vector("f16", "near").unwrap().unwrap().0,
        vec![1.0, 0.5, -2.0]
    );
    assert_eq!(
        store.get_vector("f16", "far").unwrap().unwrap().0,
        vec![0.5, 0.5, 0.5]
    );
    let hits = store
        .search_vectors("f16", &[1.0, 0.5, -2.0], 2, None)
        .unwrap();
    assert_eq!(hits[0], ("near".into(), 1.0));
    assert!(hits[1].1 < 1.0);

    store
        .create_vector_collection(
            "i8",
            4,
            VectorStorage::I8,
            VectorMetric::DotProduct,
            4,
            8,
            8,
        )
        .unwrap();
    store
        .upsert_vector("i8", "near", &[10.0, 0.0, 0.0, 0.0], None)
        .unwrap();
    store
        .upsert_vector("i8", "round", &[1.5, -1.5, 200.0, -200.0], None)
        .unwrap();
    store
        .upsert_vector("i8", "bounds", &[-128.0, 127.0, 0.0, -1.0], None)
        .unwrap();
    assert_eq!(
        store.get_vector("i8", "near").unwrap().unwrap().0,
        vec![10.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        store.get_vector("i8", "round").unwrap().unwrap().0,
        vec![2.0, -2.0, 127.0, -128.0]
    );
    assert_eq!(
        store.get_vector("i8", "bounds").unwrap().unwrap().0,
        vec![-128.0, 127.0, 0.0, -1.0]
    );
    let hits = store
        .search_vectors("i8", &[10.0, 0.0, 0.0, 0.0], 3, None)
        .unwrap();
    assert_eq!(hits[0].0, "near");
    assert_eq!(hits[0].1, 100.0);
    assert!(hits.iter().all(|(_, score)| score.is_finite()));
}

#[test]
fn i8_quantize_rounds_half_away_from_zero_and_saturates() {
    let bytes =
        super::vector::quantize(VectorStorage::I8, &[0.49999997, -0.49999997, 127.5, -128.5])
            .unwrap();
    assert_eq!(
        super::vector::dequantize(VectorStorage::I8, &bytes).unwrap(),
        vec![0.0, 0.0, 127.0, -128.0]
    );
}

#[test]
fn hnsw_top5_matches_brute_force_cosine() {
    let store = FerroStore::in_memory();
    store
        .create_vector_collection("nn", 8, VectorStorage::F32, VectorMetric::Cosine, 8, 32, 20)
        .unwrap();
    let mut stored = Vec::new();
    for index in 0..20 {
        let id = format!("v{index:02}");
        let values = nn_vector(index);
        store.upsert_vector("nn", &id, &values, None).unwrap();
        stored.push((id, values));
    }
    let query = nn_vector(0);
    let hits = store.search_vectors("nn", &query, 5, None).unwrap();
    let brute = brute_cosine(&stored, &query, 5);
    assert_eq!(hits, brute);
    assert_eq!(hits.len(), 5);
    assert_eq!(hits[0].0, "v00");
    assert_eq!(hits[0].1, 1.0);
    assert!(hits.iter().all(|(_, score)| score.is_finite()));
    assert_eq!(
        store.search_vectors("nn", &query, 5, Some(20)).unwrap(),
        hits
    );
}

#[test]
fn bad_vector_ops_write_nothing_on_memory_and_disk() {
    let mem = FerroStore::in_memory();
    assert_vector_rejects_write_nothing(&mem);
    assert_eq!(mem.read_state().unwrap().last_seq, 3);

    let (_dir, path) = temp_fls();
    {
        let store = FerroStore::open(&path).unwrap();
        assert_vector_rejects_write_nothing(&store);
        assert_eq!(store.read_state().unwrap().last_seq, 3);
        assert_eq!(committed_frames(&path).len(), 3);
    }
    let store = FerroStore::open(&path).unwrap();
    assert_eq!(
        store.get_vector("emb", "ok").unwrap().unwrap().0,
        vec![1.0, 0.0]
    );
    assert!(store.get_vector("half", "wide").unwrap().is_none());
    assert!(store.get_vector("emb", "bad").unwrap().is_none());
    assert_eq!(store.read_state().unwrap().last_seq, 3);
}

#[test]
fn delete_of_a_missing_vector_commits_and_search_omits_it() {
    exercise_delete(&FerroStore::in_memory());

    let (_dir, path) = temp_fls();
    {
        let store = FerroStore::open(&path).unwrap();
        exercise_delete(&store);
    }
    let store = FerroStore::open(&path).unwrap();
    let hits = store.search_vectors("emb", &[0.0, 1.0], 5, None).unwrap();
    assert_eq!(hits[0].0, "drop");
    assert!(hits.iter().all(|(id, _)| id != "missing"));
}

#[test]
fn vector_metadata_absent_and_json_null_round_trip() {
    let store = FerroStore::in_memory();
    store
        .create_vector_collection(
            "emb",
            2,
            VectorStorage::F32,
            VectorMetric::DotProduct,
            4,
            8,
            8,
        )
        .unwrap();
    store
        .upsert_vector("emb", "absent", &[1.0, 0.0], None)
        .unwrap();
    store
        .upsert_vector("emb", "null", &[0.0, 1.0], Some(&serde_json::Value::Null))
        .unwrap();
    assert_eq!(store.get_vector("emb", "absent").unwrap().unwrap().1, None);
    assert_eq!(
        store.get_vector("emb", "null").unwrap().unwrap().1,
        Some(serde_json::Value::Null)
    );

    let (_dir, path) = temp_fls();
    {
        let store = FerroStore::open(&path).unwrap();
        store
            .create_vector_collection(
                "emb",
                2,
                VectorStorage::F32,
                VectorMetric::DotProduct,
                4,
                8,
                8,
            )
            .unwrap();
        store
            .upsert_vector("emb", "absent", &[1.0, 0.0], None)
            .unwrap();
        store
            .upsert_vector("emb", "null", &[0.0, 1.0], Some(&serde_json::Value::Null))
            .unwrap();
        let frames = committed_frames(&path);
        let upserts: Vec<&Op> = frames
            .iter()
            .flat_map(|(_, ops)| ops)
            .filter(|op| matches!(op, Op::UpsertVector { .. }))
            .collect();
        assert!(upserts.iter().any(|op| matches!(
            op,
            Op::UpsertVector {
                id,
                metadata: None,
                ..
            } if id == "absent"
        )));
        assert!(upserts.iter().any(|op| matches!(
            op,
            Op::UpsertVector {
                id,
                metadata: Some(serde_json::Value::Null),
                ..
            } if id == "null"
        )));
    }
    let store = FerroStore::open(&path).unwrap();
    assert_eq!(store.get_vector("emb", "absent").unwrap().unwrap().1, None);
    assert_eq!(
        store.get_vector("emb", "null").unwrap().unwrap().1,
        Some(serde_json::Value::Null)
    );
}

#[test]
fn vector_checkpoint_is_version_3_and_older_snapshots_still_open() {
    let (_dir, path) = temp_fls();
    {
        let store = FerroStore::open(&path).unwrap();
        store
            .put_doc("docs", "a", &json!({ "text": "alpha" }))
            .unwrap();
        store.checkpoint().unwrap();
        assert_eq!(snapshot_version(&path), 1);
        let state = store.read_state().unwrap();
        assert!(state.tables.is_empty());
        assert!(state.vectors.is_empty());
        assert_eq!(
            snapshot_bytes(&path),
            layout::encode_snapshot(state.last_seq, &state.docs).unwrap()
        );
    }
    {
        let store = FerroStore::open(&path).unwrap();
        assert_eq!(
            store.get_doc("docs", "a").unwrap(),
            Some(json!({ "text": "alpha" }))
        );
        assert_eq!(snapshot_version(&path), 1);
        store.create_column_table("t", &int_column()).unwrap();
        store.append_column_rows("t", &ints(&[4, 5])).unwrap();
        store.checkpoint().unwrap();
        assert_eq!(snapshot_version(&path), 2);
    }
    let hits = {
        let store = FerroStore::open(&path).unwrap();
        assert_eq!(
            store.scan_column_table("t", &[]).unwrap().rows,
            ints(&[4, 5])
        );
        store
            .create_vector_collection(
                "empty",
                2,
                VectorStorage::F32,
                VectorMetric::Cosine,
                4,
                8,
                8,
            )
            .unwrap();
        store
            .create_vector_collection("emb", 2, VectorStorage::F32, VectorMetric::Cosine, 4, 8, 8)
            .unwrap();
        store
            .put_doc("emb", "doc-only", &json!({ "n": 1 }))
            .unwrap();
        store
            .upsert_vector("emb", "near", &[1.0, 0.0], Some(&json!({ "m": true })))
            .unwrap();
        store
            .upsert_vector("emb", "far", &[0.0, 1.0], Some(&serde_json::Value::Null))
            .unwrap();
        let hits = store.search_vectors("emb", &[1.0, 0.0], 2, None).unwrap();
        assert_eq!(hits[0].0, "near");
        store.checkpoint().unwrap();
        assert_eq!(snapshot_version(&path), 3);
        assert_eq!(live(&path).wal_length, 0);
        hits
    };
    let store = FerroStore::open(&path).unwrap();
    assert_eq!(
        store.get_doc("docs", "a").unwrap(),
        Some(json!({ "text": "alpha" }))
    );
    assert_eq!(
        store.get_doc("emb", "doc-only").unwrap(),
        Some(json!({ "n": 1 }))
    );
    assert_eq!(
        store.scan_column_table("t", &[]).unwrap().rows,
        ints(&[4, 5])
    );
    assert_eq!(
        store.search_vectors("emb", &[1.0, 0.0], 2, None).unwrap(),
        hits
    );
    assert!(
        store
            .search_vectors("empty", &[1.0, 0.0], 2, None)
            .unwrap()
            .is_empty()
    );
    let (values, metadata) = store.get_vector("emb", "near").unwrap().unwrap();
    assert_eq!(values, vec![1.0, 0.0]);
    assert_eq!(metadata, Some(json!({ "m": true })));
    assert_eq!(
        store.get_vector("emb", "far").unwrap().unwrap().1,
        Some(serde_json::Value::Null)
    );
    let state = store.read_state().unwrap();
    assert!(state.vectors["empty"].entry.is_none());
    assert!(state.vectors["empty"].vectors.is_empty());
    assert!(state.vectors["emb"].entry.is_some());
}

#[test]
fn a_bad_vector_op_in_a_batch_does_not_publish() {
    exercise_bad_batch(&FerroStore::in_memory());

    let (_dir, path) = temp_fls();
    let seq = {
        let store = FerroStore::open(&path).unwrap();
        let seq = exercise_bad_batch(&store);
        assert_eq!(committed_frames(&path).len(), 2);
        assert_eq!(store.read_state().unwrap().last_seq, seq);
        seq
    };
    let store = FerroStore::open(&path).unwrap();
    assert_eq!(
        store.get_doc("docs", "earlier").unwrap(),
        Some(json!({ "kept": true }))
    );
    assert!(store.get_doc("docs", "later").unwrap().is_none());
    assert!(store.get_vector("emb", "bad").unwrap().is_none());
    assert_eq!(store.read_state().unwrap().last_seq, seq);
}

#[test]
fn vector_rpc_works_over_unix_and_ws() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("vecs.fls");
    let sock = dir.path().join("vecs.sock");
    let server = FerroStoreServer::bind_unix(&sock, FerroStore::open(&file).unwrap()).unwrap();
    let client = FerroStore::connect_uri(&format!("unix://{}", sock.display())).unwrap();
    exercise_vector_rpc(&client, &file);

    let raw = super::server::connect_unix(&sock.display().to_string()).unwrap();
    let base = json!({
        "doc_collection": "notes",
        "vector_collection": "emb",
        "query": "alpha",
        "values": [1.0, 0.0],
        "limit": 5,
    });
    let mut with_default = base.clone();
    with_default["fusion"] = json!({ "method": "rrf" });
    let mut with_k = base.clone();
    with_k["fusion"] = json!({ "method": "rrf", "k": 60 });
    assert_eq!(
        raw.call("search_hybrid", with_default).unwrap(),
        raw.call("search_hybrid", with_k).unwrap()
    );
    let mut bad_k = base;
    bad_k["fusion"] = json!({ "method": "rrf", "k": 0 });
    let err = raw.call("search_hybrid", bad_k).unwrap_err();
    assert!(err.to_string().contains("k"), "{err}");

    // Vector ops in a `commit_txn` are checked and applied on the server,
    // HNSW and all, as a local write applies them.
    let seq = client.last_seq().unwrap();
    let err = raw
        .call(
            "commit_txn",
            json!({
                "reads": [],
                "ops": [{
                    "op": "upsert_vector",
                    "collection": "missing",
                    "id": "a",
                    "bytes": [0, 0, 128, 63]
                }]
            }),
        )
        .unwrap_err();
    assert!(matches!(err, StoreError::Backend(_)), "{err}");
    assert_eq!(client.last_seq().unwrap(), seq);

    let bytes: Vec<u64> = 0.5f32
        .to_le_bytes()
        .into_iter()
        .chain(0.25f32.to_le_bytes())
        .map(u64::from)
        .collect();
    assert_eq!(
        raw.call(
            "commit_txn",
            json!({
                "reads": [],
                "ops": [
                    {
                        "op": "upsert_vector",
                        "collection": "emb",
                        "id": "from-bytes",
                        "bytes": bytes,
                        "metadata": null
                    },
                    {
                        "op": "delete_vector",
                        "collection": "emb",
                        "id": "does-not-exist"
                    }
                ]
            }),
        )
        .unwrap(),
        json!({"committed": true, "seq": seq + 1})
    );
    let (values, metadata) = client.get_vector("emb", "from-bytes").unwrap().unwrap();
    assert_eq!(values, vec![0.5, 0.25]);
    assert_eq!(metadata, Some(serde_json::Value::Null));
    let seq = client.last_seq().unwrap();
    assert_eq!(
        raw.call(
            "commit_txn",
            json!({
                "reads": [],
                "ops": [{
                    "op": "create_vector_collection",
                    "name": "extra",
                    "dim": 2,
                    "storage": "i8",
                    "metric": "dot",
                    "m": 4,
                    "ef_construction": 8,
                    "ef_search": 8
                }]
            }),
        )
        .unwrap(),
        json!({"committed": true, "seq": seq + 1})
    );
    client
        .upsert_vector("extra", "q", &[1.0, -1.0], None)
        .unwrap();
    assert_eq!(
        client.get_vector("extra", "q").unwrap().unwrap().0,
        vec![1.0, -1.0]
    );
    drop(client);
    drop(raw);
    drop(server);

    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("vecs.fls");
    let server =
        FerroStoreServer::bind_tcp("127.0.0.1:0", FerroStore::open(&file).unwrap(), psk_auth())
            .unwrap();
    let port = server.local_addr().port();
    let client = connect_psk(&format!("ws://127.0.0.1:{port}")).unwrap();
    exercise_vector_rpc(&client, &file);
    let knowledge: Vec<KnowledgeHit> = client
        .search_knowledge(
            "notes",
            "emb",
            "---",
            &[1.0, 0.0],
            5,
            Fusion::Weighted {
                bm25: 0.0,
                vector: 1.0,
            },
        )
        .unwrap();
    assert!(
        knowledge
            .iter()
            .any(|hit| hit.id == "both" && hit.doc.is_some())
    );
    assert!(
        knowledge
            .iter()
            .any(|hit| hit.id == "only" && hit.doc.is_none())
    );
    drop(client);
    drop(server);
}

#[test]
fn hybrid_and_knowledge_fusion_orders_one_sided_ids() {
    let store = FerroStore::in_memory();
    store
        .create_vector_collection(
            "notes",
            2,
            VectorStorage::F32,
            VectorMetric::Cosine,
            4,
            16,
            16,
        )
        .unwrap();
    store
        .put_doc("notes", "both", &json!({ "text": "alpha alpha" }))
        .unwrap();
    store
        .put_doc("notes", "alpha-doc", &json!({ "text": "alpha" }))
        .unwrap();
    store
        .put_doc("notes", "other", &json!({ "text": "zzzz" }))
        .unwrap();
    store
        .upsert_vector(
            "notes",
            "both",
            &[1.0, 0.0],
            Some(&json!({ "side": "both" })),
        )
        .unwrap();
    store
        .upsert_vector(
            "notes",
            "vec-only",
            &[0.9, 0.1],
            Some(&json!({ "side": "vec" })),
        )
        .unwrap();
    store
        .upsert_vector("notes", "far", &[0.0, 1.0], None)
        .unwrap();

    let bm25 = store.search_bm25("notes", "alpha", 10).unwrap();
    assert_eq!(
        bm25.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
        vec!["both", "alpha-doc"]
    );
    let vectors = store
        .search_vectors("notes", &[1.0, 0.0], 10, None)
        .unwrap();
    assert_eq!(
        vectors
            .iter()
            .map(|(id, _)| id.as_str())
            .collect::<Vec<_>>(),
        vec!["both", "vec-only", "far"]
    );
    let rrf = Fusion::Rrf { k: 60 };
    let hybrid = store
        .search_hybrid("notes", "notes", "alpha", &[1.0, 0.0], 10, rrf)
        .unwrap();
    assert_eq!(hybrid, rrf_scores(&bm25, &vectors, 60));
    assert_eq!(
        hybrid.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(),
        vec!["both", "alpha-doc", "vec-only", "far"]
    );
    let weighted = Fusion::Weighted {
        bm25: 1.0,
        vector: 1.0,
    };
    assert_eq!(
        store
            .search_hybrid("notes", "notes", "alpha", &[1.0, 0.0], 10, weighted)
            .unwrap(),
        weighted_scores(&bm25, &vectors, 1.0, 1.0)
    );
    let vector_only = Fusion::Weighted {
        bm25: 0.0,
        vector: 1.0,
    };
    let vector_order = store
        .search_hybrid("notes", "notes", "alpha", &[1.0, 0.0], 10, vector_only)
        .unwrap();
    assert_eq!(vector_order, weighted_scores(&bm25, &vectors, 0.0, 1.0));
    assert!(vector_order.iter().any(|(id, _)| id == "alpha-doc"));
    assert!(vector_order.iter().any(|(id, _)| id == "vec-only"));

    let zbm = store.search_bm25("notes", "zzzz", 10).unwrap();
    assert_eq!(zbm.len(), 1);
    let zhybrid = store
        .search_hybrid(
            "notes",
            "notes",
            "zzzz",
            &[1.0, 0.0],
            10,
            Fusion::Weighted {
                bm25: 1.0,
                vector: 0.0,
            },
        )
        .unwrap();
    assert_eq!(zhybrid, weighted_scores(&zbm, &vectors, 1.0, 0.0));
    assert_eq!(zhybrid[0].0, "other");
    assert_eq!(zhybrid[0].1, 1.0);

    assert_eq!(
        store
            .search_hybrid("notes", "notes", "---", &[1.0, 0.0], 10, rrf)
            .unwrap(),
        rrf_scores(&[], &vectors, 60)
    );
    assert!(
        store
            .search_hybrid("notes", "notes", "", &[1.0, 0.0], 10, rrf)
            .unwrap()
            .iter()
            .any(|(id, _)| id == "both")
    );
    assert!(
        store
            .search_hybrid(
                "notes",
                "notes",
                "alpha",
                &[1.0, 0.0],
                0,
                Fusion::Rrf { k: 60 },
            )
            .unwrap()
            .is_empty()
    );
    assert!(
        store
            .search_hybrid(
                "no-such-docs",
                "notes",
                "alpha",
                &[1.0, 0.0],
                0,
                Fusion::Rrf { k: 60 },
            )
            .unwrap()
            .is_empty()
    );
    assert!(
        store
            .search_knowledge(
                "notes",
                "notes",
                "alpha",
                &[1.0, 0.0],
                0,
                Fusion::Rrf { k: 60 },
            )
            .unwrap()
            .is_empty()
    );
    let err = store
        .search_hybrid(
            "notes",
            "missing",
            "alpha",
            &[1.0, 0.0],
            0,
            Fusion::Rrf { k: 60 },
        )
        .unwrap_err();
    assert!(matches!(err, StoreError::Backend(_)), "{err}");
    let err = store
        .search_hybrid("notes", "notes", "alpha", &[1.0], 0, Fusion::Rrf { k: 60 })
        .unwrap_err();
    assert!(matches!(err, StoreError::Backend(_)), "{err}");
    let err = store
        .search_hybrid(
            "notes",
            "notes",
            "alpha",
            &[1.0, 0.0],
            0,
            Fusion::Rrf { k: 0 },
        )
        .unwrap_err();
    assert!(matches!(err, StoreError::Backend(_)), "{err}");
    let err = store
        .search_knowledge(
            "notes",
            "notes",
            "alpha",
            &[1.0, 0.0],
            0,
            Fusion::Rrf { k: 0 },
        )
        .unwrap_err();
    assert!(matches!(err, StoreError::Backend(_)), "{err}");
    let err = store
        .search_hybrid(
            "notes",
            "notes",
            "alpha",
            &[1.0, 0.0],
            5,
            Fusion::Rrf { k: 0 },
        )
        .unwrap_err();
    assert!(err.to_string().contains("k"), "{err}");
    let err = store
        .search_hybrid(
            "notes",
            "notes",
            "alpha",
            &[1.0, 0.0],
            5,
            Fusion::Weighted {
                bm25: f64::MAX,
                vector: f64::MAX,
            },
        )
        .unwrap_err();
    assert!(matches!(err, StoreError::Backend(_)), "{err}");
    assert_eq!(
        store
            .search_hybrid(
                "notes",
                "notes",
                "alpha",
                &[1.0, 0.0],
                10,
                Fusion::Weighted {
                    bm25: f64::MAX,
                    vector: 0.0,
                },
            )
            .unwrap(),
        weighted_scores(&bm25, &vectors, f64::MAX, 0.0)
    );
    assert_eq!(
        store
            .search_hybrid(
                "notes",
                "notes",
                "alpha",
                &[1.0, 0.0],
                10,
                Fusion::Weighted {
                    bm25: 0.0,
                    vector: f64::MAX,
                },
            )
            .unwrap(),
        weighted_scores(&bm25, &vectors, 0.0, f64::MAX)
    );
    let err = store
        .search_hybrid(
            "notes",
            "notes",
            "alpha",
            &[1.0, 0.0],
            5,
            Fusion::Weighted {
                bm25: 0.0,
                vector: 0.0,
            },
        )
        .unwrap_err();
    assert!(matches!(err, StoreError::Backend(_)), "{err}");

    let knowledge = store
        .search_knowledge("notes", "notes", "alpha", &[1.0, 0.0], 10, rrf)
        .unwrap();
    assert_eq!(
        knowledge
            .iter()
            .map(|hit| hit.id.as_str())
            .collect::<Vec<_>>(),
        vec!["both", "alpha-doc", "vec-only", "far"]
    );
    let both = knowledge.iter().find(|hit| hit.id == "both").unwrap();
    assert_eq!(both.doc, Some(json!({ "text": "alpha alpha" })));
    assert_eq!(both.metadata, Some(json!({ "side": "both" })));
    assert_eq!(both.score, hybrid[0].1);
    let alpha = knowledge.iter().find(|hit| hit.id == "alpha-doc").unwrap();
    assert!(alpha.metadata.is_none());
    assert_eq!(alpha.doc, Some(json!({ "text": "alpha" })));
    let only = knowledge.iter().find(|hit| hit.id == "vec-only").unwrap();
    assert!(only.doc.is_none());
    assert_eq!(only.metadata, Some(json!({ "side": "vec" })));
    let far = knowledge.iter().find(|hit| hit.id == "far").unwrap();
    assert!(far.doc.is_none());
    assert!(far.metadata.is_none());
    assert!(knowledge.iter().all(|hit| hit.score.is_finite()));
}

#[test]
fn zero_vector_cosine_is_zero_and_empty_limits_do_not_error() {
    let store = FerroStore::in_memory();
    store
        .create_vector_collection("z", 2, VectorStorage::F32, VectorMetric::Cosine, 4, 8, 8)
        .unwrap();
    store
        .upsert_vector("z", "origin", &[0.0, 0.0], None)
        .unwrap();
    store.upsert_vector("z", "axis", &[1.0, 0.0], None).unwrap();
    assert_eq!(
        store.search_vectors("z", &[0.0, 0.0], 5, None).unwrap(),
        vec![("axis".into(), 0.0), ("origin".into(), 0.0)]
    );
    let hits = store.search_vectors("z", &[1.0, 0.0], 5, None).unwrap();
    assert_eq!(hits.iter().find(|(id, _)| id == "origin").unwrap().1, 0.0);
    assert_eq!(hits[0].0, "axis");
    assert!(
        store
            .search_vectors("z", &[0.0, 0.0], 0, None)
            .unwrap()
            .is_empty()
    );
    assert!(
        store
            .search_vectors("z", &[1.0, 0.0], 5, Some(0))
            .unwrap()
            .is_empty()
    );
    store
        .create_vector_collection(
            "empty",
            2,
            VectorStorage::F32,
            VectorMetric::Cosine,
            4,
            8,
            8,
        )
        .unwrap();
    assert!(
        store
            .search_vectors("empty", &[1.0, 0.0], 0, None)
            .unwrap()
            .is_empty()
    );
    assert!(
        store
            .search_vectors("empty", &[1.0, 0.0], 5, Some(0))
            .unwrap()
            .is_empty()
    );
    assert!(
        store
            .search_vectors("empty", &[1.0, 0.0], 5, None)
            .unwrap()
            .is_empty()
    );
    for (limit, ef) in [(0_usize, None), (5, Some(0))] {
        let wrong = store.search_vectors("z", &[1.0], limit, ef).unwrap_err();
        assert!(matches!(wrong, StoreError::Backend(_)), "{wrong}");
        let non_finite = store
            .search_vectors("z", &[f32::NAN, 0.0], limit, ef)
            .unwrap_err();
        assert!(matches!(non_finite, StoreError::Backend(_)), "{non_finite}");
        let empty_wrong = store
            .search_vectors("empty", &[1.0], limit, ef)
            .unwrap_err();
        assert!(
            matches!(empty_wrong, StoreError::Backend(_)),
            "{empty_wrong}"
        );
    }
    let err = store
        .search_vectors("missing", &[0.0, 0.0], 1, None)
        .unwrap_err();
    assert!(err.to_string().contains("no vector collection"), "{err}");
    let err = store
        .search_vectors("missing", &[0.0, 0.0], 0, None)
        .unwrap_err();
    assert!(matches!(err, StoreError::Backend(_)), "{err}");
    assert!(err.to_string().contains("no vector collection"), "{err}");
    let err = store
        .search_vectors("missing", &[1.0, 0.0], 5, Some(0))
        .unwrap_err();
    assert!(matches!(err, StoreError::Backend(_)), "{err}");
    assert!(err.to_string().contains("no vector collection"), "{err}");
}

fn hnsw_record(id: &str, m: u32, values: &[f32], layer0: &[&str]) -> super::vector::VectorRecord {
    let level = super::vector::vector_level(id, m);
    let mut neighbors = vec![
        layer0
            .iter()
            .map(|nid| (*nid).to_string())
            .collect::<Vec<_>>(),
    ];
    neighbors.resize(usize::from(level) + 1, Vec::new());
    super::vector::VectorRecord {
        bytes: super::vector::quantize(VectorStorage::F32, values).unwrap(),
        metadata: None,
        level,
        neighbors,
    }
}

fn canonical_id(vectors: &BTreeMap<String, super::vector::VectorRecord>) -> String {
    vectors
        .iter()
        .max_by(|(id_a, rec_a), (id_b, rec_b)| {
            rec_a.level.cmp(&rec_b.level).then_with(|| id_b.cmp(id_a))
        })
        .map(|(id, _)| id.clone())
        .unwrap()
}

#[test]
fn validate_graph_rejects_a_live_entry_that_is_not_canonical() {
    let m = 4;
    let mut vectors = BTreeMap::new();
    for (id, values) in [("a", [0.0_f32, 0.0]), ("b", [1.0, 0.0])] {
        vectors.insert(id.to_string(), hnsw_record(id, m, &values, &[]));
    }
    let canonical = canonical_id(&vectors);
    let other = vectors.keys().find(|id| *id != &canonical).unwrap().clone();
    let mut col = super::vector::VectorCollection {
        dim: 2,
        storage: VectorStorage::F32,
        metric: VectorMetric::L2,
        m,
        ef_construction: 8,
        ef_search: 8,
        entry: Some(other),
        vectors,
    };
    let err = super::vector::validate_graph("emb", &col).unwrap_err();
    assert!(matches!(err, StoreError::Backend(_)), "{err}");
    assert!(err.to_string().contains("corrupt"), "{err}");
    col.entry = Some(canonical);
    super::vector::validate_graph("emb", &col).unwrap();
}

fn l2_graph(order: &[&str]) -> super::vector::VectorCollection {
    let m = 4;
    // `b` and `m` are equally far from `o`; `a` is farther. Distance order is
    // `b`, `m`, `a` (equal distance breaks ties by id).
    let points = BTreeMap::from([
        ("o", [0.0_f32, 0.0]),
        ("a", [5.0, 0.0]),
        ("b", [0.0, 1.0]),
        ("m", [1.0, 0.0]),
    ]);
    let links: BTreeMap<&str, &[&str]> = BTreeMap::from([
        ("o", order),
        ("a", &["o"][..]),
        ("b", &["o"][..]),
        ("m", &["o"][..]),
    ]);
    let mut vectors = BTreeMap::new();
    for id in ["o", "a", "b", "m"] {
        vectors.insert(id.to_string(), hnsw_record(id, m, &points[id], links[id]));
    }
    let entry = Some(canonical_id(&vectors));
    super::vector::VectorCollection {
        dim: 2,
        storage: VectorStorage::F32,
        metric: VectorMetric::L2,
        m,
        ef_construction: 8,
        ef_search: 8,
        entry,
        vectors,
    }
}

/// The neighbor order is the full check's, not open's: `validate_graph` (what a snapshot
/// decode runs) accepts a list out of (distance, id) order, `validate_graph_full` refuses it.
#[test]
fn the_full_check_rejects_neighbors_that_are_not_distance_then_id_ordered() {
    for order in [&["m", "b", "a"][..], &["a", "b", "m"]] {
        let graph = l2_graph(order);
        super::vector::validate_graph("emb", &graph).unwrap();
        let err = super::vector::validate_graph_full("emb", &graph).unwrap_err();
        assert!(matches!(err, StoreError::Backend(_)), "{err}");
        assert!(err.to_string().contains("corrupt"), "{err}");
        assert!(err.to_string().contains("not ordered by distance"), "{err}");
    }
    super::vector::validate_graph_full("emb", &l2_graph(&["b", "m", "a"])).unwrap();
}

/// A CRC-valid snapshot whose only fault is a neighbor list out of canonical order decodes
/// (open does not re-derive the order), the full check names the fault, and the encoder
/// refuses to write it again.
#[test]
fn a_snapshot_decodes_a_misordered_neighbor_list_the_full_check_and_the_encoder_refuse_it() {
    let vectors: super::state::Vectors = BTreeMap::from([(
        "emb".to_string(),
        std::sync::Arc::new(l2_graph(&["b", "m", "a"])),
    )]);
    let docs = super::state::Docs::default();
    let tables = super::state::Tables::new();
    let mut blob = layout::encode_snapshot_v3(1, &docs, &tables, &vectors).unwrap();
    // `o`'s layer-0 list: a count of 3, then `b`, `m`, `a`, each length-prefixed.
    let lp = |id: u8| [1, 0, 0, 0, id];
    let list = |ids: [u8; 3]| -> Vec<u8> {
        let mut out = vec![3, 0, 0, 0];
        for id in ids {
            out.extend_from_slice(&lp(id));
        }
        out
    };
    let canonical = list(*b"bma");
    let at: Vec<usize> = (0..blob.len() - canonical.len())
        .filter(|at| blob[*at..].starts_with(&canonical))
        .collect();
    assert_eq!(at.len(), 1, "one canonical list of three in the snapshot");
    blob[at[0]..at[0] + canonical.len()].copy_from_slice(&list(*b"mba"));
    let body = blob.len() - 4;
    let crc = layout::crc32(&blob[..body]);
    blob[body..].copy_from_slice(&crc.to_le_bytes());

    let snap = layout::decode_snapshot(&blob).unwrap();
    let col = &snap.vectors["emb"];
    assert_eq!(col.vectors["o"].neighbors[0], ["m", "b", "a"]);
    let err = super::vector::validate_graph_full("emb", col).unwrap_err();
    assert!(err.to_string().contains("not ordered by distance"), "{err}");
    let err = layout::encode_snapshot_v3(2, &docs, &tables, &snap.vectors).unwrap_err();
    assert!(err.to_string().contains("not ordered by distance"), "{err}");
}

fn put_frame(seq: u64, id: &str) -> Vec<u8> {
    layout::encode_frame(
        seq,
        &[Op::Put {
            collection: "docs".into(),
            id: id.into(),
            doc: json!({ "n": seq }),
        }],
    )
    .unwrap()
}

/// Extra coverage from the 2026-10-30 audit, tied to no reported defect (the audit's S-4 is the
/// HNSW search re-sort): these pin boundaries of WAL decoding that recovery relies on. `decode_wal` is
/// unchanged by the audit; see O-9 for the one open question about its tail.
#[test]
fn wal_decode_handles_every_truncation_point_of_a_tail_frame() {
    let first = put_frame(1, "a");
    let second = put_frame(2, "b");
    let mut full = first.clone();
    full.extend_from_slice(&second);
    // Cutting the second frame anywhere short of whole drops exactly that frame.
    for cut in first.len()..full.len() {
        let frames = layout::decode_wal(&full[..cut]).unwrap();
        assert_eq!(frames.len(), 1, "cut at {cut}");
        assert_eq!(frames[0].0, 1);
    }
    assert_eq!(layout::decode_wal(&full).unwrap().len(), 2);
    assert!(layout::decode_wal(&[]).unwrap().is_empty());
    assert!(layout::decode_wal(&first[..7]).unwrap().is_empty());
}

#[test]
fn wal_decode_never_allocates_for_a_length_it_does_not_have() {
    // A header that claims 4 GiB and a good CRC field is a torn tail, not an allocation.
    let mut buf = put_frame(1, "a");
    buf.extend_from_slice(&u32::MAX.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&[1, 2, 3]);
    assert_eq!(layout::decode_wal(&buf).unwrap().len(), 1);
}

#[test]
fn wal_decode_flags_a_corrupt_middle_frame_but_not_a_corrupt_last_one() {
    let mut a = put_frame(1, "a");
    let b = put_frame(2, "b");
    let c = put_frame(3, "c");
    // Corrupt the last frame's payload: it is dropped as a torn tail.
    let mut torn_last = a.clone();
    torn_last.extend_from_slice(&b);
    let n = torn_last.len();
    torn_last[n - 1] ^= 0xff;
    assert_eq!(layout::decode_wal(&torn_last).unwrap().len(), 1);
    // Corrupt a frame that a whole frame follows: that is damage, and must not be skipped.
    a[10] ^= 0xff;
    a.extend_from_slice(&b);
    a.extend_from_slice(&c);
    assert!(layout::decode_wal(&a).is_err());
}

#[test]
fn wal_decode_rejects_well_formed_frames_with_bad_contents() {
    // An empty payload has a valid CRC (0) but no sequence number: corrupt, not a tail.
    assert!(layout::decode_wal(&[0u8; 8]).is_err());
    // An unknown op tag behind a valid CRC.
    let mut payload = 1u64.to_le_bytes().to_vec();
    payload.extend_from_slice(&1u32.to_le_bytes());
    payload.push(0xEE);
    let mut frame = (payload.len() as u32).to_le_bytes().to_vec();
    frame.extend_from_slice(&layout::crc32(&payload).to_le_bytes());
    frame.extend_from_slice(&payload);
    assert!(layout::decode_wal(&frame).is_err());
    // An op count larger than the bytes that could hold it.
    let mut payload = 1u64.to_le_bytes().to_vec();
    payload.extend_from_slice(&u32::MAX.to_le_bytes());
    let mut frame = (payload.len() as u32).to_le_bytes().to_vec();
    frame.extend_from_slice(&layout::crc32(&payload).to_le_bytes());
    frame.extend_from_slice(&payload);
    assert!(layout::decode_wal(&frame).is_err());
    // Trailing bytes after the last op inside the payload.
    let ok = put_frame(1, "a");
    let mut payload = ok[8..].to_vec();
    payload.push(0);
    let mut frame = (payload.len() as u32).to_le_bytes().to_vec();
    frame.extend_from_slice(&layout::crc32(&payload).to_le_bytes());
    frame.extend_from_slice(&payload);
    assert!(layout::decode_wal(&frame).is_err());
}
