//! A write body that calls back into its own store handle: every call is
//! refused with an error, and nothing hangs. A call from another thread is
//! never refused.

use super::{
    FerroStore, FerroStoreServer, SyncAdapter, SyncBatch, SyncMode, VectorMetric, VectorStorage,
    lock_mutex,
};
use fl_core::StoreError;
use fl_core::store::Catalog;
use serde_json::json;
use std::sync::{Arc, mpsc};
use std::time::Duration;

const REFUSED: &str = "backend failure: FerroStore called from inside its own transaction body";

/// A store of each mode, shareable with a thread. The guard keeps the
/// directory and the server.
type Guard = (tempfile::TempDir, FerroStoreServer);

fn stores() -> (Vec<(&'static str, Arc<FerroStore>)>, Guard) {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("s.sock");
    let server = FerroStoreServer::bind_unix(&sock, FerroStore::in_memory()).unwrap();
    let remote = FerroStore::connect_uri(&format!("unix://{}", sock.display())).unwrap();
    let disk = FerroStore::open(dir.path().join("s.fls")).unwrap();
    (
        vec![
            ("memory", Arc::new(FerroStore::in_memory())),
            ("disk", Arc::new(disk)),
            ("remote", Arc::new(remote)),
        ],
        (dir, server),
    )
}

/// `f` on its own thread, which must answer within 20 s: a call that
/// deadlocks fails the test instead of hanging it.
fn within<T: Send + 'static>(what: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(Duration::from_secs(20))
        .unwrap_or_else(|_| panic!("{what}: no answer after 20 s (a nested call hung)"))
}

type Nested = fn(&FerroStore) -> Result<(), StoreError>;

/// An adapter a nested call tries to attach.
struct Quiet;

impl SyncAdapter for Quiet {
    fn name(&self) -> &str {
        "quiet"
    }

    fn on_commit(&self, _batch: &SyncBatch) -> Result<(), StoreError> {
        Ok(())
    }
}

/// Calls a body might make on its own store: reads and writes, the public
/// API and the traits.
const NESTED: &[(&str, Nested)] = &[
    ("get_doc", |s| s.get_doc("c", "x").map(drop)),
    ("query_by_field", |s| {
        s.query_by_field("c", "f", "v").map(drop)
    }),
    ("search_bm25", |s| s.search_bm25("c", "q", 3).map(drop)),
    ("format_version", |s| s.format_version().map(drop)),
    ("has_vector_collection", |s| {
        s.has_vector_collection("v").map(drop)
    }),
    ("poll_mq_group", |s| s.poll_mq_group("t", "g", 3).map(drop)),
    ("list_projects", |s| s.list_projects().map(drop)),
    ("put_doc", |s| s.put_doc("c", "nested", &json!(1))),
    ("delete_doc", |s| s.delete_doc("c", "x")),
    ("ack_mq", |s| s.ack_mq("t", "g", 1)),
    ("publish_mq", |s| {
        s.publish_mq("t", "k", &json!(1)).map(drop)
    }),
    ("create_vector_collection", |s| {
        s.create_vector_collection("v", 2, VectorStorage::F32, VectorMetric::L2, 4, 16, 16)
    }),
    ("add_project", |s| s.add_project("/nested").map(drop)),
    ("transaction", |s| s.transaction(|_| Ok(()))),
    ("checkpoint", |s| s.checkpoint()),
    ("sync_flush", |s| s.sync_flush().map(drop)),
    ("attach_sync_adapter", |s| {
        s.attach_sync_adapter(SyncMode::WriteBehind, Arc::new(Quiet))
    }),
    ("detach_sync_adapter", |s| s.detach_sync_adapter()),
];

#[test]
fn every_call_from_inside_a_body_is_refused_and_the_outer_transaction_goes_on() {
    let (stores, _guard) = stores();
    for (mode, store) in stores {
        let inner = Arc::clone(&store);
        let errors = within(mode, move || {
            inner
                .transaction(|txn| {
                    let mut errors = Vec::new();
                    for (name, call) in NESTED {
                        let got = call(&inner);
                        errors.push((*name, got.map_err(|err| err.to_string())));
                    }
                    // The body goes on after the refusals, and commits.
                    txn.put_doc("c", "outer", &json!(1))?;
                    Ok(errors)
                })
                .unwrap()
        });
        for (name, got) in errors {
            assert_eq!(got, Err(REFUSED.to_string()), "{mode}: nested {name}");
        }
        assert_eq!(
            store.get_doc("c", "outer").unwrap(),
            Some(json!(1)),
            "{mode}"
        );
        assert_eq!(store.get_doc("c", "nested").unwrap(), None, "{mode}");
        assert!(store.list_projects().unwrap().is_empty(), "{mode}");
        assert!(!store.has_vector_collection("v").unwrap(), "{mode}");
        assert_eq!(
            store.sync_mode(),
            None,
            "{mode}: a nested attach took effect"
        );
        // The handle works normally once the body is done.
        store.put_doc("c", "after", &json!(2)).unwrap();
        assert_eq!(
            store.get_doc("c", "after").unwrap(),
            Some(json!(2)),
            "{mode}"
        );
    }
}

#[test]
fn a_body_that_returns_the_refusal_commits_nothing() {
    let (stores, _guard) = stores();
    for (mode, store) in stores {
        let inner = Arc::clone(&store);
        let (err, after) = within(mode, move || {
            let err = inner
                .transaction(|txn| {
                    txn.put_doc("c", "staged", &json!(1))?;
                    inner.put_doc("c", "nested", &json!(2))?;
                    Ok(())
                })
                .unwrap_err()
                .to_string();
            // The same thread, out of the body: no longer refused.
            let after = inner
                .put_doc("c", "after", &json!(3))
                .map_err(|e| e.to_string());
            (err, after)
        });
        assert_eq!(err, REFUSED, "{mode}");
        assert_eq!(after, Ok(()), "{mode}");
        assert_eq!(store.get_doc("c", "staged").unwrap(), None, "{mode}");
        assert_eq!(store.get_doc("c", "nested").unwrap(), None, "{mode}");
    }
}

#[test]
fn a_body_that_panics_leaves_the_handle_usable() {
    let (stores, _guard) = stores();
    for (mode, store) in stores {
        let inner = Arc::clone(&store);
        let (panicked, write, read) = within(mode, move || {
            let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = inner.transaction(|_| -> Result<(), StoreError> { panic!("body panics") });
            }))
            .is_err();
            // The same thread, after the unwind: no longer in a body.
            let write = inner
                .put_doc("c", "after", &json!(1))
                .map_err(|e| e.to_string());
            let read = inner.get_doc("c", "after").map_err(|e| e.to_string());
            (panicked, write, read)
        });
        assert!(panicked, "{mode}");
        assert_eq!(write, Ok(()), "{mode}");
        assert_eq!(read, Ok(Some(json!(1))), "{mode}");
    }
}

/// A group commit on disk: each member's body runs on the member's own
/// thread, and its nested call is refused. A reader on another thread,
/// while a body runs, is not refused.
#[test]
fn in_a_disk_group_each_member_is_refused_and_another_thread_is_not() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(FerroStore::open(dir.path().join("g.fls")).unwrap());
    store.put_doc("c", "base", &json!(0)).unwrap();
    let (reader_go, reader_wait) = mpsc::channel::<()>();
    let (reader_done, reader_result) = mpsc::channel();
    // The outside reader: runs while member 0's body waits for it.
    let reader_store = Arc::clone(&store);
    let reader = std::thread::spawn(move || {
        reader_wait.recv().unwrap();
        let got = reader_store.get_doc("c", "base").map_err(|e| e.to_string());
        reader_done.send(got).unwrap();
    });
    let reader_result = Arc::new(std::sync::Mutex::new(Some(reader_result)));
    let outcomes = within("group", move || {
        let gate = lock_mutex(&store.commit_gate);
        let mut handles = Vec::new();
        for member in 0..3 {
            let member_store = Arc::clone(&store);
            let go = reader_go.clone();
            let result = Arc::clone(&reader_result);
            handles.push(std::thread::spawn(move || {
                let store = member_store;
                store.transaction(|txn| {
                    if member == 0 {
                        go.send(()).unwrap();
                        let rx = result.lock().unwrap().take().unwrap();
                        let outside = rx.recv_timeout(Duration::from_secs(10)).unwrap();
                        assert_eq!(
                            outside,
                            Ok(Some(json!(0))),
                            "the outside reader was refused"
                        );
                    }
                    let nested = store.get_doc("c", "base").unwrap_err().to_string();
                    txn.put_doc("c", &format!("m{member}"), &json!(member))?;
                    Ok(nested)
                })
            }));
            store.wait_until_queued(member + 1);
        }
        drop(gate);
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap().map_err(|e| e.to_string()))
            .collect::<Vec<_>>()
    });
    reader.join().unwrap();
    for outcome in outcomes {
        assert_eq!(outcome, Ok(REFUSED.to_string()));
    }
}
