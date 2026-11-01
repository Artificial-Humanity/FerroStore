//! Spec §4.5: the bounded retry of a remote commit, and write bodies free
//! of side effects.
//!
//! The retry hook (`retry::hook`) sees each remote commit attempt before it
//! is sent and may lose it, as a commit by another writer would.

use super::retry::{BACKOFF_CAP, BACKOFF_FIRST, Backoff, Cas, REMOTE_ATTEMPTS, hook};
use super::{FerroStore, FerroStoreServer};
use fl_core::StoreError;
use serde_json::json;
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::rc::Rc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// A served in-memory store; the guard keeps the server and its socket.
pub(super) struct Served {
    server: FerroStoreServer,
    uri: String,
    _dir: tempfile::TempDir,
}

fn served() -> Served {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("s.sock");
    let server = FerroStoreServer::bind_unix(&sock, FerroStore::in_memory()).unwrap();
    Served {
        server,
        uri: format!("unix://{}", sock.display()),
        _dir: dir,
    }
}

fn client(served: &Served) -> FerroStore {
    FerroStore::connect_uri(&served.uri).unwrap()
}

/// Run `f` on its own thread; panic if it gives no answer in `secs`, so a
/// retry that never ends fails the test instead of hanging it.
fn within<T: Send + 'static>(secs: u64, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(Duration::from_secs(secs))
        .unwrap_or_else(|_| panic!("no answer after {secs} s: the retry is not bounded"))
}

#[test]
fn backoff_grows_from_5_ms_to_500_ms_with_jitter() {
    let mut firsts = BTreeSet::new();
    for _ in 0..64 {
        let mut backoff = Backoff::new();
        for attempt in 1..REMOTE_ATTEMPTS {
            let base = BACKOFF_FIRST
                .saturating_mul(1 << (attempt - 1))
                .min(BACKOFF_CAP);
            let top = base.saturating_mul(2).min(BACKOFF_CAP);
            let delay = backoff.delay(attempt);
            assert!(
                delay >= base && delay <= top,
                "attempt {attempt}: {delay:?} is outside {base:?}..={top:?}"
            );
            if attempt == 1 {
                firsts.insert(delay);
            }
        }
        assert_eq!(backoff.delay(40), BACKOFF_CAP, "capped at 500 ms");
    }
    assert!(firsts.len() > 8, "the jitter spreads writers: {firsts:?}");
}

/// What one write that loses every attempt did.
struct Exhausted {
    error: String,
    conflict: bool,
    attempts: Vec<u32>,
    elapsed: Duration,
}

type Write = fn(&FerroStore) -> Result<(), StoreError>;

fn lose_every_attempt(uri: String, write: Write) -> Exhausted {
    within(30, move || {
        let client = FerroStore::connect_uri(&uri).unwrap();
        let attempts = Rc::new(RefCell::new(Vec::new()));
        let seen = Rc::clone(&attempts);
        let _hook = hook::install(
            move |cas: &Cas<'_>| {
                seen.borrow_mut().push(cas.attempt);
                true
            },
            true,
        );
        let start = Instant::now();
        let err = write(&client).unwrap_err();
        let elapsed = start.elapsed();
        let attempts = attempts.borrow().clone();
        Exhausted {
            conflict: matches!(err, StoreError::Conflict { .. }),
            error: err.to_string(),
            attempts,
            elapsed,
        }
    })
}

#[test]
fn a_remote_write_that_loses_every_attempt_is_a_conflict_after_eight() {
    let served = served();
    let writes: [(&str, Write); 2] = [
        ("transaction", |store| {
            store.transaction(|txn| txn.put_doc("d", "a", &json!(1)))
        }),
        ("put_doc", |store| store.put_doc("d", "b", &json!(2))),
    ];
    for (name, write) in writes {
        let got = lose_every_attempt(served.uri.clone(), write);
        assert!(got.conflict, "{name}: {}", got.error);
        assert!(
            got.error.contains("8 commit attempts"),
            "{name}: {}",
            got.error
        );
        assert_eq!(
            got.attempts,
            (1..=REMOTE_ATTEMPTS).collect::<Vec<_>>(),
            "{name}"
        );
        // Seven waits: at least 5 + 10 + ... + 320 ms.
        assert!(
            got.elapsed >= Duration::from_millis(635),
            "{name}: {:?}",
            got.elapsed
        );
        assert!(
            got.elapsed < Duration::from_secs(10),
            "{name}: {:?}",
            got.elapsed
        );
    }
    let store = served.server.store();
    assert_eq!(store.read_state().unwrap().last_seq, 0, "nothing committed");
    assert_eq!(store.get_doc("d", "a").unwrap(), None);
    assert_eq!(store.get_doc("d", "b").unwrap(), None);
}

#[test]
fn a_lost_compare_and_swap_runs_the_body_again_on_the_new_state() {
    let served = served();
    let uri = served.uri.clone();
    let (runs, value) = within(30, move || {
        let client = FerroStore::connect_uri(&uri).unwrap();
        let other = FerroStore::connect_uri(&uri).unwrap();
        // Another writer commits between the body and its swap.
        let fired = Cell::new(false);
        let _hook = hook::install(
            move |cas: &Cas<'_>| {
                if cas.attempt == 1 && !fired.replace(true) {
                    other.put_doc("counters", "c", &json!({"n": 10})).unwrap();
                }
                false
            },
            true,
        );
        let runs = Cell::new(0);
        client
            .transaction(|txn| {
                runs.set(runs.get() + 1);
                let n = txn
                    .get_doc("counters", "c")
                    .and_then(|doc| doc["n"].as_u64())
                    .unwrap_or(0);
                txn.put_doc("counters", "c", &json!({"n": n + 1}))
            })
            .unwrap();
        (runs.get(), client.get_doc("counters", "c").unwrap())
    });
    assert_eq!(runs, 2, "the body ran again after the lost swap");
    assert_eq!(
        value,
        Some(json!({"n": 11})),
        "the second run saw the commit"
    );
}

// ---- every write body twice --------------------------------------------------

/// The sources whose write call sites the test enumerates.
const SOURCES: &[(&str, &str)] = &[
    ("mod.rs", include_str!("mod.rs")),
    ("transaction.rs", include_str!("transaction.rs")),
    ("manifest_ops.rs", include_str!("manifest_ops.rs")),
    ("migrate.rs", include_str!("migrate.rs")),
    ("traits_catalog.rs", include_str!("traits_catalog.rs")),
    ("traits_state.rs", include_str!("traits_state.rs")),
    ("traits_wire.rs", include_str!("traits_wire.rs")),
    ("traits_workflow.rs", include_str!("traits_workflow.rs")),
];

/// Call sites that never reach a remote compare-and-swap, and why.
const NOT_REMOTE: &[(&str, &str, &str)] = &[
    (
        "mod.rs",
        "Ok(self.with_write_outcome(body)?.0)",
        "`with_write` itself: it is #[track_caller], so its caller is the site",
    ),
    (
        "mod.rs",
        "let outcome = self.with_write_outcome(move |txn| {",
        "`publish_mq`: a remote handle sends one `publish_mq` RPC, which the server runs once",
    ),
];

/// The fewest write call sites the enumeration may find: it may grow, and a
/// shrink must be deliberate.
const WRITE_SITES_FLOOR: usize = 50;

/// The source files of `ferrostore` that are not scanned for write call
/// sites, though a line in them matches [`starts_a_write`], and why.
const NOT_SCANNED: &[(&str, &str)] = &[
    (
        "compat.rs",
        "its one write call is in its own #[cfg(test)] module, on an in-memory store",
    ),
    ("tests.rs", "test code, as every tests_*.rs file is"),
];

/// A line that starts a write: a call of `with_write`, `with_write_outcome`
/// or `commit`, outside comments.
fn starts_a_write(line: &str) -> bool {
    let code = line.trim_start();
    !code.starts_with("//")
        && (code.contains(".with_write(")
            || code.contains(".with_write_outcome(")
            || code.contains("self.commit("))
}

/// Every `(file, line)` that starts a write ([`starts_a_write`]).
fn write_sites() -> BTreeSet<(&'static str, u32)> {
    let mut sites = BTreeSet::new();
    let mut excluded = vec![0; NOT_REMOTE.len()];
    for (file, text) in SOURCES {
        for (index, line) in text.lines().enumerate() {
            let code = line.trim_start();
            if !starts_a_write(code) {
                continue;
            }
            if let Some(at) = NOT_REMOTE
                .iter()
                .position(|(f, needle, _)| f == file && code.contains(needle))
            {
                excluded[at] += 1;
                continue;
            }
            sites.insert((*file, u32::try_from(index + 1).unwrap()));
        }
    }
    for ((file, needle, why), hits) in NOT_REMOTE.iter().zip(excluded) {
        assert_eq!(
            hits, 1,
            "exclusion `{needle}` in {file} ({why}) matched {hits} lines"
        );
    }
    sites
}

/// [`SOURCES`] is hand-listed: every source file of `ferrostore` (not a
/// `tests_*.rs` file) with a line that starts a write is in it, or in
/// [`NOT_SCANNED`] with its reason. A new file of write call sites cannot
/// go unscanned.
#[test]
fn every_source_file_with_a_write_call_site_is_scanned() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/ferrostore");
    let mut with_writes = BTreeSet::new();
    let mut files = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap().to_string();
        if !name.ends_with(".rs") || name.starts_with("tests_") {
            continue;
        }
        files += 1;
        let text = std::fs::read_to_string(&path).unwrap();
        if text.lines().any(starts_a_write) {
            with_writes.insert(name);
        }
    }
    assert!(
        files >= 20,
        "only {files} source files in {}",
        dir.display()
    );
    for (name, why) in NOT_SCANNED {
        assert!(
            with_writes.remove(*name),
            "{name} is excluded ({why}) but has no write call site, or is gone"
        );
        assert!(
            SOURCES.iter().all(|(file, _)| file != name),
            "{name} is both scanned and excluded"
        );
    }
    let scanned: BTreeSet<String> = SOURCES.iter().map(|(file, _)| file.to_string()).collect();
    let missing: Vec<&String> = with_writes.difference(&scanned).collect();
    assert!(
        missing.is_empty(),
        "files with write call sites missing from SOURCES: {missing:?}"
    );
}

/// `text` with every `urn:uuid:` id that `known` does not hold renamed by
/// order of first use: a body that mints a fresh id mints another when it
/// runs again, and that is not a side effect. An id from the state is kept.
fn fresh_ids_renamed(text: &str, known: &str) -> String {
    const PREFIX: &str = "urn:uuid:";
    let mut out = String::with_capacity(text.len());
    let mut names: Vec<String> = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find(PREFIX) {
        let end = (at + PREFIX.len() + 36).min(rest.len());
        let id = rest.get(at..end).unwrap_or(&rest[at..]);
        out.push_str(&rest[..at]);
        if known.contains(id) {
            out.push_str(id);
        } else {
            let k = match names.iter().position(|name| name == id) {
                Some(k) => k,
                None => {
                    names.push(id.to_string());
                    names.len() - 1
                }
            };
            out.push_str(&format!("<fresh id {k}>"));
        }
        rest = &rest[at + id.len()..];
    }
    out.push_str(rest);
    out
}

/// What the hook saw over a run of writes.
#[derive(Default)]
struct Twice {
    /// The first attempt of the write in flight: its site and its ops.
    first: Option<((&'static str, u32), String)>,
    /// Sites whose second run staged the same ops as the first.
    same: BTreeMap<(&'static str, u32), usize>,
    mismatches: Vec<String>,
}

fn site_of(cas: &Cas<'_>) -> (&'static str, u32) {
    let file = SOURCES
        .iter()
        .map(|(name, _)| *name)
        .find(|name| cas.caller.file().ends_with(&format!("ferrostore/{name}")))
        .unwrap_or_else(|| {
            panic!(
                "a remote write from outside the enumerated sources: {}",
                cas.caller
            )
        });
    (file, cas.caller.line())
}

/// Every remote write on this thread loses its first attempt, so its body
/// runs twice against the same state; the second run must stage the same
/// ops, up to fresh ids.
fn every_body_twice() -> (hook::Installed, Rc<RefCell<Twice>>) {
    let twice = Rc::new(RefCell::new(Twice::default()));
    let seen = Rc::clone(&twice);
    let installed = hook::install(
        move |cas: &Cas<'_>| {
            let site = site_of(cas);
            let known = format!("{:?}", cas.known);
            let ops = fresh_ids_renamed(&format!("{:?}", cas.ops), &known);
            let mut seen = seen.borrow_mut();
            match cas.attempt {
                1 => {
                    seen.first = Some((site, ops));
                    true
                }
                2 => {
                    match seen.first.take() {
                        Some((first_site, first)) if first_site == site && first == ops => {
                            *seen.same.entry(site).or_default() += 1;
                        }
                        Some((first_site, first)) => seen.mismatches.push(format!(
                            "{}:{}: the second run staged other ops\n  first ({}:{}): {first}\n  \
                             second: {ops}",
                            site.0, site.1, first_site.0, first_site.1
                        )),
                        None => seen.mismatches.push(format!(
                            "{}:{}: a second attempt with no first",
                            site.0, site.1
                        )),
                    }
                    false
                }
                n => {
                    seen.mismatches.push(format!(
                        "{}:{}: attempt {n} with no other writer",
                        site.0, site.1
                    ));
                    false
                }
            }
        },
        false,
    );
    (installed, twice)
}

/// A fresh served store for one conformance case: the client, then the
/// guard that keeps its server.
pub(super) fn fresh_remote() -> (FerroStore, Served) {
    let served = served();
    (client(&served), served)
}

/// The store contracts over a remote handle (a Unix socket).
pub(super) fn run_contracts() {
    run_contracts_over(fresh_remote);
}

/// The store contracts over the remote handles `fresh` makes (each with
/// the guard that keeps its server).
pub(super) fn run_contracts_over<G>(fresh: impl Fn() -> (FerroStore, G) + Copy) {
    use fl_core::conformance::{Single, SplitOver};
    let single = move || {
        let (s, g) = fresh();
        Single(s, g)
    };
    let split = move || {
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
    fl_core::conformance::escalations(single);
    crate::manifest_conformance::manifest_contract(fresh);
}

/// The public writes of `FerroStore` the contracts do not reach.
pub(super) fn run_public_writes() {
    run_public_writes_over(fresh_remote);
}

/// [`run_public_writes`] over the remote handles `fresh` makes.
pub(super) fn run_public_writes_over<G>(fresh: impl Fn() -> (FerroStore, G)) {
    use super::{ColumnDef, ColumnType, ColumnValue, VectorMetric, VectorStorage};
    let (store, _guard) = fresh();
    store.put_doc("docs", "a", &json!({"n": 1})).unwrap();
    store.delete_doc("docs", "a").unwrap();
    store.publish_mq("topic", "k", &json!(1)).unwrap();
    store.ack_mq("topic", "g", 1).unwrap();
    let columns = [ColumnDef {
        name: "n".into(),
        ty: ColumnType::Int64,
    }];
    store.create_column_table("t", &columns).unwrap();
    store
        .append_column_rows("t", &[vec![ColumnValue::Int64(1)]])
        .unwrap();
    store
        .create_vector_collection("v", 2, VectorStorage::F32, VectorMetric::L2, 4, 16, 16)
        .unwrap();
    store.upsert_vector("v", "x", &[1.0, 0.0], None).unwrap();
    store
        .upsert_vectors("v", &[("y", &[0.0, 1.0], Some(&json!("m")))])
        .unwrap();
    store.delete_vector("v", "x").unwrap();
    store
        .transaction(|txn| txn.put_doc("docs", "b", &json!(2)))
        .unwrap();

    // Trait writes the contracts leave out.
    {
        use fl_core::finding::Finding;
        use fl_core::ids::RecordId;
        use fl_core::iri::Iri;
        use fl_core::routing::{ForeignRecord, Tier};
        use fl_core::state::StateSyncCursor;
        use fl_core::store::{Bindings, Catalog, StateStore, Tracker};
        let project = store.add_project("/remote").unwrap();
        let elsewhere =
            RecordId(Iri::parse("urn:uuid:00000000-0000-7000-8000-000000000042").unwrap());
        store
            .add_finding_checked(
                Finding::raise(project, elsewhere.clone(), "rev", "the claim"),
                ForeignRecord::for_tests(elsewhere, "t", Tier::Github),
            )
            .unwrap();
        store.bind_node_id("acme/widgets", "R_node").unwrap();
        store
            .set_sync_cursor(StateSyncCursor {
                remote: "origin".into(),
                branch: "fl/state".into(),
                last_pushed_commit: None,
                last_revision_seq: 1,
                last_registry_seq: 2,
                last_wire_msg_seq: 3,
                last_wire_rcpt_seq: 4,
            })
            .unwrap();
    }

    let dir = tempfile::tempdir().unwrap();
    let redb = crate::RedbStore::open(&dir.path().join("src.redb")).unwrap();
    fl_core::store::Catalog::add_project(&redb, "/migrated").unwrap();
    let (target, _target_guard) = fresh();
    super::migrate_redb_to_ferrostore(&redb, &target).unwrap();
}

#[test]
fn every_remote_write_body_runs_twice_to_the_same_ops() {
    let sites = write_sites();
    assert!(
        sites.len() >= WRITE_SITES_FLOOR,
        "only {} write call sites found: {sites:?}",
        sites.len()
    );
    let (_hook, twice) = every_body_twice();
    run_contracts();
    run_public_writes();
    let twice = twice.borrow();
    assert!(
        twice.mismatches.is_empty(),
        "{} write bodies did not run twice to the same ops:\n{}",
        twice.mismatches.len(),
        twice.mismatches.join("\n")
    );
    let missing: Vec<_> = sites
        .iter()
        .filter(|site| !twice.same.contains_key(site))
        .collect();
    assert!(
        missing.is_empty(),
        "{} of {} write call sites never ran twice: {missing:?}",
        missing.len(),
        sites.len()
    );
    let runs: usize = twice.same.values().sum();
    println!(
        "{} write call sites, {runs} writes, each body run twice to the same ops",
        sites.len()
    );
}
