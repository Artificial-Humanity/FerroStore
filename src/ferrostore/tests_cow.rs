//! §4.2 of the v2 cutover spec: copy-on-write state and overlay transactions.
//!
//! The byte-stability test pins the on-disk bytes of a populated store. Its
//! constants were recorded from the engine before the state became two levels
//! of `Arc`, so a change to the in-memory shape cannot change the file.

use super::layout::Superblock;
use super::state::DocState;
use super::{
    ColumnDef, ColumnType, ColumnValue, FerroStore, FerroStoreServer, VectorMetric, VectorStorage,
    latest_superblock,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn live(path: &Path) -> Superblock {
    latest_superblock(&File::open(path).unwrap()).unwrap().1
}

/// `(sha256 of the live snapshot extent, sha256 of the whole file)`.
fn file_hashes(path: &Path) -> (String, String) {
    let bytes = std::fs::read(path).unwrap();
    let sb = live(path);
    let start = usize::try_from(sb.snapshot_offset).unwrap();
    let end = start + usize::try_from(sb.snapshot_length).unwrap();
    (sha256_hex(&bytes[start..end]), sha256_hex(&bytes))
}

fn columns() -> Vec<ColumnDef> {
    vec![
        ColumnDef {
            name: "n".to_string(),
            ty: ColumnType::Int64,
        },
        ColumnDef {
            name: "s".to_string(),
            ty: ColumnType::Utf8,
        },
    ]
}

fn row(n: i64, s: &str) -> Vec<ColumnValue> {
    vec![ColumnValue::Int64(n), ColumnValue::Utf8(s.to_string())]
}

/// Documents in several collections (out of order, a delete, a replace, a
/// null, nested JSON, non-ASCII), an MQ publish and an ack.
fn populate_docs(store: &FerroStore) {
    store
        .put_doc("beta", "z", &json!({"k": [1, 2, 3]}))
        .unwrap();
    store.put_doc("alpha", "b", &json!("two")).unwrap();
    store
        .put_doc("alpha", "a", &json!({"x": 1, "y": {"z": null}}))
        .unwrap();
    store.put_doc("alpha", "c", &json!(null)).unwrap();
    store.put_doc("gamma", "only", &json!(3.5)).unwrap();
    store.put_doc("alpha", "b", &json!("two, again")).unwrap();
    store.delete_doc("gamma", "only").unwrap();
    store
        .put_doc("delta", "ünï", &json!({"text": "héllo wörld"}))
        .unwrap();
    store.publish_mq("topic", "k1", &json!({"p": 1})).unwrap();
    store.publish_mq("topic", "k2", &json!({"p": 2})).unwrap();
    store.ack_mq("topic", "g", 1).unwrap();
}

fn populate_table(store: &FerroStore) {
    store.create_column_table("t", &columns()).unwrap();
    store
        .append_column_rows("t", &[row(1, "one"), row(2, "two")])
        .unwrap();
    store.append_column_rows("t", &[row(30, "thirty")]).unwrap();
    store.create_column_table("u", &columns()).unwrap();
}

fn populate_vectors(store: &FerroStore) {
    store
        .create_vector_collection("emb", 3, VectorStorage::F32, VectorMetric::Cosine, 4, 8, 8)
        .unwrap();
    for (i, id) in ["v0", "v1", "v2", "v3", "v4", "v5"].iter().enumerate() {
        let f = i as f32;
        let meta = json!({"i": i});
        store
            .upsert_vector("emb", id, &[1.0 + f, 2.0 - f, 0.5 * f], Some(&meta))
            .unwrap();
    }
    store
        .upsert_vector("emb", "v2", &[9.0, 1.0, 1.0], None)
        .unwrap();
    store.delete_vector("emb", "v4").unwrap();
    store
        .create_vector_collection("i8s", 2, VectorStorage::I8, VectorMetric::L2, 4, 8, 8)
        .unwrap();
    store
        .upsert_vector("i8s", "a", &[0.25, -0.5], None)
        .unwrap();
}

/// Build a store with `populate`, hash its WAL-only file, checkpoint, and hash
/// the snapshot extent and the whole file again.
fn hashes_for(populate: impl Fn(&FerroStore)) -> [String; 3] {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("fixture.fls");
    let store = FerroStore::open(&path).unwrap();
    populate(&store);
    let (_, wal_file) = file_hashes(&path);
    store.checkpoint().unwrap();
    let (snapshot, whole) = file_hashes(&path);
    [wal_file, snapshot, whole]
}

#[test]
fn snapshot_and_wal_bytes_are_unchanged_by_the_state_shape() {
    // Snapshot version 1 (documents only).
    let v1 = hashes_for(populate_docs);
    // Snapshot version 2 (documents and column tables).
    let v2 = hashes_for(|store| {
        populate_docs(store);
        populate_table(store);
    });
    // Snapshot version 3 (documents, column tables and vector collections).
    let v3 = hashes_for(|store| {
        populate_docs(store);
        populate_table(store);
        populate_vectors(store);
    });
    let got = [v1, v2, v3];
    // Recorded from origin/main 96577b8, before the state shape changed.
    let want: [[&str; 3]; 3] = [
        [
            "31d1e5be32fe754236359eb7c3fd5ffefbfdfe10a2179fa7725489608869d324",
            "8dd7d0f96f7f251c7039e72aa8313dd173ab7ce21927fe4e32d8064f7d8e0807",
            "fa0c5aa7979d0d709bd4c56a8fccd4fc74962690f845e387de7803bb794a9d48",
        ],
        [
            "fdf430a5e8e47623f84db7657c88a37ed837dbcd82b638e381790e8bc0ee8b8f",
            "9a65e03fbc32e605edd3cd4e51bfb062f191d30aec479e5667ddcf8a9fa02a4c",
            "0f9c09fc2f7175aca5b5cde54396f07a2a3b9286f53387a82f780cbb205f0894",
        ],
        [
            "de571b6a5130ed42844c7322d23dcaee3b7effc666bf5cc72379c011b3c11d79",
            "74817dc09d9c8edbd1a9d19eb3b78123b0e68bb8d4b2637267b34d40420351b9",
            "d2a48dbf58a5f51bbbf13200ec20c58f7bb03acd34a7aa1eb722f4e37cb3626d",
        ],
    ];
    assert_eq!(
        got.iter()
            .map(|row| row.iter().map(String::as_str).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        want.iter().map(|row| row.to_vec()).collect::<Vec<_>>(),
        "the .fls bytes for the same logical state changed"
    );
}

// ---- overlay semantics ------------------------------------------------------

/// A memory store and a disk store (with its directory kept alive).
fn stores() -> Vec<(FerroStore, Option<tempfile::TempDir>)> {
    let tmp = tempfile::tempdir().unwrap();
    let disk = FerroStore::open(tmp.path().join("overlay.fls")).unwrap();
    vec![(FerroStore::in_memory(), None), (disk, Some(tmp))]
}

fn ids(scan: &[(String, serde_json::Value)]) -> Vec<&str> {
    scan.iter().map(|(id, _)| id.as_str()).collect()
}

#[test]
fn a_put_then_a_delete_in_one_transaction_leaves_nothing() {
    for (store, _dir) in stores() {
        store.put_doc("c", "keep", &json!(1)).unwrap();
        store
            .with_write(|txn| {
                txn.put("c", "x", json!("new"));
                assert_eq!(txn.get("c", "x"), Some(json!("new")));
                txn.delete("c", "x");
                assert_eq!(txn.get("c", "x"), None);
                assert_eq!(ids(&txn.scan("c")), vec!["keep"]);
                Ok(())
            })
            .unwrap();
        assert_eq!(store.get_doc("c", "x").unwrap(), None);
        assert_eq!(store.get_doc("c", "keep").unwrap(), Some(json!(1)));
    }
}

#[test]
fn a_delete_then_a_put_in_one_transaction_is_the_put() {
    for (store, _dir) in stores() {
        store.put_doc("c", "x", &json!("old")).unwrap();
        store
            .with_write(|txn| {
                txn.delete("c", "x");
                assert_eq!(txn.get("c", "x"), None);
                assert!(txn.scan("c").is_empty());
                txn.put("c", "x", json!("new"));
                assert_eq!(txn.get("c", "x"), Some(json!("new")));
                assert_eq!(txn.scan("c"), vec![("x".to_string(), json!("new"))]);
                Ok(())
            })
            .unwrap();
        assert_eq!(store.get_doc("c", "x").unwrap(), Some(json!("new")));
    }
}

#[test]
fn a_delete_of_a_missing_id_commits_and_changes_no_document() {
    for (store, _dir) in stores() {
        store.put_doc("c", "a", &json!(1)).unwrap();
        let before = store.read_state().unwrap().last_seq;
        store
            .with_write(|txn| {
                txn.delete("c", "missing");
                txn.delete("nowhere", "missing");
                assert_eq!(txn.get("c", "missing"), None);
                assert!(txn.scan("nowhere").is_empty());
                assert_eq!(ids(&txn.scan("c")), vec!["a"]);
                Ok(())
            })
            .unwrap();
        let state = store.read_state().unwrap();
        assert_eq!(state.last_seq, before + 1, "the delete is still a commit");
        assert_eq!(store.get_doc("c", "a").unwrap(), Some(json!(1)));
        assert_eq!(store.get_doc("c", "missing").unwrap(), None);
    }
}

#[test]
fn a_scan_merges_overlay_puts_and_deletes_into_document_id_order() {
    for (store, _dir) in stores() {
        for id in ["a", "c", "e", "g"] {
            store.put_doc("c", id, &json!(id)).unwrap();
        }
        // Neighbouring collection names must not bleed into the scan.
        store.put_doc("b", "zz", &json!(0)).unwrap();
        store.put_doc("cc", "0", &json!(0)).unwrap();
        store.put_doc("c\u{0}", "0", &json!(0)).unwrap();
        store
            .with_write(|txn| {
                txn.put("c", "d", json!("d"));
                txn.put("c", "0", json!("0"));
                txn.put("c", "z", json!("z"));
                txn.delete("c", "c");
                txn.put("c", "e", json!("e2"));
                txn.delete("c", "g");
                txn.put("cc", "1", json!(1));
                txn.put("b", "zzz", json!(1));
                let scan = txn.scan("c");
                assert_eq!(ids(&scan), vec!["0", "a", "d", "e", "z"]);
                assert_eq!(scan[3].1, json!("e2"));
                assert_eq!(ids(&txn.scan("cc")), vec!["0", "1"]);
                assert_eq!(ids(&txn.scan("b")), vec!["zz", "zzz"]);
                assert!(txn.scan("").is_empty());
                Ok(())
            })
            .unwrap();
        let scanned = store.with_read(|txn| Ok(txn.scan("c"))).unwrap();
        assert_eq!(
            scanned,
            vec![
                ("0".to_string(), json!("0")),
                ("a".to_string(), json!("a")),
                ("d".to_string(), json!("d")),
                ("e".to_string(), json!("e2")),
                ("z".to_string(), json!("z")),
            ]
        );
    }
}

#[test]
fn a_failed_body_leaves_the_base_untouched() {
    for (store, _dir) in stores() {
        store.put_doc("c", "a", &json!(1)).unwrap();
        store.create_column_table("t", &columns()).unwrap();
        let before = store.read_state().unwrap().last_seq;
        let err = store
            .with_write(|txn| {
                txn.put("c", "a", json!("changed"));
                txn.put("c", "b", json!("added"));
                txn.delete("c", "a");
                txn.put("other", "x", json!(1));
                Err::<(), _>(super::backend("body refused"))
            })
            .unwrap_err();
        assert!(err.to_string().contains("body refused"), "{err}");
        // A body that stages a table op and then fails writes nothing either.
        store
            .with_write(|txn| {
                txn.stage(super::Op::AppendColumnGroup {
                    name: "t".to_string(),
                    bytes: super::columnar::encode_row_group(&columns(), &[row(1, "x")])?,
                })?;
                Err::<(), _>(super::backend("after the table op"))
            })
            .unwrap_err();
        assert_eq!(store.read_state().unwrap().last_seq, before);
        assert_eq!(store.get_doc("c", "a").unwrap(), Some(json!(1)));
        assert_eq!(store.get_doc("c", "b").unwrap(), None);
        assert_eq!(store.get_doc("other", "x").unwrap(), None);
        assert!(store.scan_column_table("t", &[]).unwrap().rows.is_empty());
    }
}

// ---- copy-on-write: what a commit copies ------------------------------------

/// Documents in `a`, `b` and `c`, column tables `t1` and `t2`, vector
/// collections `v1` and `v2`.
fn cow_fixture(store: &FerroStore) {
    for collection in ["a", "b", "c"] {
        for i in 0..3 {
            store
                .put_doc(collection, &format!("id{i}"), &json!({"i": i}))
                .unwrap();
        }
    }
    for table in ["t1", "t2"] {
        store.create_column_table(table, &columns()).unwrap();
        store
            .append_column_rows(table, &[row(1, "one"), row(2, "two")])
            .unwrap();
    }
    for name in ["v1", "v2"] {
        store
            .create_vector_collection(name, 2, VectorStorage::F32, VectorMetric::L2, 4, 8, 8)
            .unwrap();
        store.upsert_vector(name, "p", &[1.0, 2.0], None).unwrap();
        store.upsert_vector(name, "q", &[3.0, 1.0], None).unwrap();
    }
}

/// Every docs collection, column table and vector collection not named in
/// `touched` is the same `Arc` in `before` and `after`; every named one that
/// is in both is a different `Arc`.
fn assert_copied_only(before: &DocState, after: &DocState, touched: &[&str]) {
    let names: BTreeSet<&String> = before
        .docs
        .collections()
        .chain(after.docs.collections())
        .collect();
    assert!(names.len() >= 3, "fixture collections missing: {names:?}");
    for name in names {
        let (Some(old), Some(new)) = (
            before.docs.collection_arc(name),
            after.docs.collection_arc(name),
        ) else {
            assert!(
                touched.contains(&name.as_str()),
                "collection {name} appeared or vanished"
            );
            continue;
        };
        assert_eq!(
            !Arc::ptr_eq(old, new),
            touched.contains(&name.as_str()),
            "docs collection {name}"
        );
    }
    assert_eq!(before.tables.len(), 2);
    for (name, old) in &before.tables {
        let new = &after.tables[name];
        assert_eq!(
            !Arc::ptr_eq(old, new),
            touched.contains(&name.as_str()),
            "column table {name}"
        );
    }
    assert_eq!(before.vectors.len(), 2);
    for (name, old) in &before.vectors {
        let new = &after.vectors[name];
        assert_eq!(
            !Arc::ptr_eq(old, new),
            touched.contains(&name.as_str()),
            "vector collection {name}"
        );
    }
}

#[test]
fn a_write_copies_only_what_it_touches_and_a_held_state_never_changes() {
    let store = FerroStore::in_memory();
    cow_fixture(&store);

    // A document put: only collection `a`.
    let before = store.read_state().unwrap();
    store.put_doc("a", "new", &json!("n")).unwrap();
    let after = store.read_state().unwrap();
    assert!(!Arc::ptr_eq(&before, &after));
    assert_copied_only(&before, &after, &["a"]);
    assert_eq!(before.docs.get("a", "new"), None, "the held state changed");
    assert_eq!(before.docs.in_collection("a").count(), 3);
    assert_eq!(after.docs.get("a", "new"), Some(&json!("n")));
    assert_eq!(after.last_seq, before.last_seq + 1);

    // A column-table append: only `t1`.
    let before = store.read_state().unwrap();
    store.append_column_rows("t1", &[row(3, "three")]).unwrap();
    let after = store.read_state().unwrap();
    assert_copied_only(&before, &after, &["t1"]);
    assert_eq!(before.tables["t1"].groups.len(), 1);
    assert_eq!(after.tables["t1"].groups.len(), 2);

    // A vector upsert: only `v1`.
    let before = store.read_state().unwrap();
    store.upsert_vector("v1", "r", &[0.5, 0.5], None).unwrap();
    let after = store.read_state().unwrap();
    assert_copied_only(&before, &after, &["v1"]);
    assert!(!before.vectors["v1"].vectors.contains_key("r"));
    assert!(after.vectors["v1"].vectors.contains_key("r"));

    // One transaction over two collections and a delete of a missing id.
    let before = store.read_state().unwrap();
    store
        .with_write(|txn| {
            txn.put("a", "id0", json!("replaced"));
            txn.delete("b", "id1");
            txn.delete("c", "missing");
            Ok(())
        })
        .unwrap();
    let after = store.read_state().unwrap();
    assert_copied_only(&before, &after, &["a", "b"]);
    assert_eq!(before.docs.get("a", "id0"), Some(&json!({"i": 0})));
    assert_eq!(before.docs.get("b", "id1"), Some(&json!({"i": 1})));
    assert_eq!(after.docs.get("b", "id1"), None);

    // Deleting the last document drops the collection; the reader keeps it.
    let before = store.read_state().unwrap();
    store
        .with_write(|txn| {
            for (id, _) in txn.scan("c") {
                txn.delete("c", id);
            }
            Ok(())
        })
        .unwrap();
    let after = store.read_state().unwrap();
    assert!(after.docs.collection_arc("c").is_none());
    assert_eq!(before.docs.in_collection("c").count(), 3);
    assert_copied_only(&before, &after, &["c"]);
}

#[test]
fn a_commit_nobody_else_holds_changes_the_state_in_place() {
    let store = FerroStore::in_memory();
    cow_fixture(&store);
    // Addresses only: no `Arc` is held across the write. A copy would be
    // allocated while the original is still alive, so it cannot reuse the
    // original's address.
    let addresses = |store: &FerroStore| {
        let state = store.read_state().unwrap();
        (
            Arc::as_ptr(&state) as usize,
            Arc::as_ptr(state.docs.collection_arc("a").unwrap()) as usize,
            Arc::as_ptr(&state.tables["t1"]) as usize,
        )
    };
    let before = addresses(&store);
    store.put_doc("a", "new", &json!(1)).unwrap();
    store.append_column_rows("t1", &[row(9, "nine")]).unwrap();
    store
        .with_write(|txn| {
            txn.put("a", "id1", json!("again"));
            Ok(())
        })
        .unwrap();
    assert_eq!(addresses(&store), before, "an unshared commit copied");
}

#[test]
fn a_failed_commit_leaves_the_live_state_the_same_arc() {
    let store = FerroStore::in_memory();
    cow_fixture(&store);
    let before = store.read_state().unwrap();
    store
        .with_write(|txn| {
            txn.put("a", "x", json!(1));
            Err::<(), _>(super::backend("refused"))
        })
        .unwrap_err();
    // A vector op that fails in staging: a duplicate create.
    store
        .create_vector_collection("v1", 2, VectorStorage::F32, VectorMetric::L2, 4, 8, 8)
        .unwrap_err();
    let after = store.read_state().unwrap();
    assert!(
        Arc::ptr_eq(&before, &after),
        "a failed write replaced the state"
    );
}

/// A body that sees a staged op fail and carries on regardless returns `Ok`:
/// only the transaction's own `failed` guard keeps it from committing what it
/// staged before the failure.
#[test]
fn a_body_that_ignores_a_failed_stage_cannot_commit() {
    use super::Op;
    for (store, _dir) in stores() {
        cow_fixture(&store);
        let bad_ops = [
            // A duplicate create: `v1` exists.
            Op::CreateVectorCollection {
                name: "v1".to_string(),
                dim: 2,
                storage: VectorStorage::F32,
                metric: VectorMetric::L2,
                m: 4,
                ef_construction: 8,
                ef_search: 8,
            },
            // An append to a column table that does not exist.
            Op::AppendColumnGroup {
                name: "missing".to_string(),
                bytes: super::columnar::encode_row_group(&columns(), &[row(1, "x")]).unwrap(),
            },
        ];
        for bad in bad_ops {
            let before = store.read_state().unwrap();
            let err = store
                .with_write(|txn| {
                    txn.put("a", "staged-first", json!(1));
                    let _ = txn.stage(bad.clone());
                    Ok(())
                })
                .unwrap_err();
            assert!(err.to_string().contains("staged op failed"), "{err}");
            let after = store.read_state().unwrap();
            assert!(
                Arc::ptr_eq(&before, &after),
                "a body that ignored a failed stage replaced the state"
            );
            assert_eq!(store.get_doc("a", "staged-first").unwrap(), None);
        }
    }
}

#[test]
fn a_served_store_copies_only_what_a_client_write_touches() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("cow.sock");
    let server = FerroStoreServer::bind_unix(&sock, FerroStore::in_memory()).unwrap();
    let client = FerroStore::connect_uri(&format!("unix://{}", sock.display())).unwrap();
    cow_fixture(&client);

    let before = server.store().read_state().unwrap();
    client.put_doc("b", "new", &json!(1)).unwrap();
    let after = server.store().read_state().unwrap();
    assert_copied_only(&before, &after, &["b"]);
    assert_eq!(before.docs.get("b", "new"), None);

    // The client's multi-op transaction is a `commit_ops` on the server.
    let before = server.store().read_state().unwrap();
    client
        .with_write(|txn| {
            txn.put("c", "x", json!(1));
            txn.delete("a", "id2");
            Ok(())
        })
        .unwrap();
    let after = server.store().read_state().unwrap();
    assert_copied_only(&before, &after, &["a", "c"]);
    drop(client);
    drop(server);
}

#[test]
fn applied_sync_batches_copy_only_what_they_touch() {
    use super::{SyncBatch, SyncOp};
    let store = FerroStore::in_memory();
    cow_fixture(&store);
    let before = store.read_state().unwrap();
    let next = before.last_seq + 1;
    store
        .apply_inbound(vec![
            SyncBatch {
                seq: next,
                ops: vec![SyncOp::PutDoc {
                    collection: "a".into(),
                    id: "s1".into(),
                    doc: json!(1),
                }],
            },
            SyncBatch {
                seq: next + 1,
                ops: vec![SyncOp::DeleteDoc {
                    collection: "a".into(),
                    id: "s1".into(),
                }],
            },
        ])
        .unwrap();
    let after = store.read_state().unwrap();
    assert_eq!(after.last_seq, next + 1);
    assert_eq!(after.docs.get("a", "s1"), None);
    // The put and the delete cancel: no collection changed, none was copied.
    assert_copied_only(&before, &after, &[]);

    let before = after;
    store
        .apply_inbound(vec![SyncBatch {
            seq: before.last_seq + 1,
            ops: vec![
                SyncOp::PutDoc {
                    collection: "b".into(),
                    id: "s2".into(),
                    doc: json!(2),
                },
                SyncOp::DeleteDoc {
                    collection: "c".into(),
                    id: "id0".into(),
                },
            ],
        }])
        .unwrap();
    let after = store.read_state().unwrap();
    assert_copied_only(&before, &after, &["b", "c"]);
    assert_eq!(before.docs.get("b", "s2"), None);
    assert_eq!(after.docs.get("b", "s2"), Some(&json!(2)));
}
