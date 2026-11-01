//! Spec §6.1: a remote `Txn` reads through a session pinned at one snapshot
//! and commits with `commit_txn`, validated by value; `req` makes a write
//! exactly-once.

use super::auth::Principal;
use super::retry::{Cas, hook};
use super::server::{self, RemoteClient};
use super::session::{
    self, Claim, DEDUP_ENTRIES, DEDUP_FULL, DEDUP_MIN_AGE, DEDUP_PER_PRINCIPAL, DEDUP_QUOTA,
    DEDUP_TTL, Dedup, SESSION_EXPIRED, SESSIONS_PER_CONN, SESSIONS_PER_PRINCIPAL,
    SESSIONS_PER_SERVER,
};
use super::{FerroStore, FerroStoreServer, Inner, VectorMetric, VectorStorage};
use fl_core::StoreError;
use fl_core::store::Catalog;
use serde_json::{Value, json};
use std::cell::Cell;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

struct Served {
    server: FerroStoreServer,
    sock: PathBuf,
    _dir: tempfile::TempDir,
}

impl Served {
    fn uri(&self) -> String {
        format!("unix://{}", self.sock.display())
    }

    fn client(&self) -> FerroStore {
        FerroStore::connect_uri(&self.uri()).unwrap()
    }

    fn raw(&self) -> RemoteClient {
        server::connect_unix(&self.sock.display().to_string()).unwrap()
    }

    /// Wait until the server holds `n` open sessions (a close is sent
    /// without waiting for its reply).
    fn wait_sessions(&self, n: usize) {
        let deadline = Instant::now() + Duration::from_secs(5);
        while self.server.open_sessions() != n {
            assert!(
                Instant::now() < deadline,
                "the server holds {} sessions, not {n}",
                self.server.open_sessions()
            );
            thread::sleep(Duration::from_millis(5));
        }
    }
}

fn serve(store: FerroStore) -> Served {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("s.sock");
    let server = FerroStoreServer::bind_unix(&sock, store).unwrap();
    Served {
        server,
        sock,
        _dir: dir,
    }
}

fn remote(store: &FerroStore) -> &RemoteClient {
    match &store.inner {
        Inner::Remote(remote) => remote,
        _ => panic!("not a remote handle"),
    }
}

/// Run `f` on its own thread; panic if it gives no answer in `secs`.
fn within<T: Send + 'static>(secs: u64, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(Duration::from_secs(secs))
        .unwrap_or_else(|_| panic!("no answer after {secs} s"))
}

/// Retry `write` through `Conflict` (its 8 attempts all lost): that write
/// committed nothing, so a retry cannot apply it twice.
fn until_committed<T>(mut write: impl FnMut() -> Result<T, StoreError>) -> T {
    for _ in 0..1_000 {
        match write() {
            Ok(done) => return done,
            Err(StoreError::Conflict { .. }) => {}
            Err(err) => panic!("write failed: {err}"),
        }
    }
    panic!("still conflicting after 1000 retries");
}

// ---- no whole state over the wire ----------------------------------------------

#[test]
fn the_remote_conformance_run_never_asks_for_the_whole_state() {
    assert_eq!(server::legacy_calls(), 0, "before the run");
    super::tests_retry::run_contracts();
    super::tests_retry::run_public_writes();
    assert_eq!(
        server::legacy_calls(),
        0,
        "a client asked for read_state or commit_ops"
    );
    // The counter counts: a raw request for the removed method is refused.
    let (store, _guard) = super::tests_retry::fresh_remote();
    let err = remote(&store).call("read_state", json!({})).unwrap_err();
    assert!(err.to_string().contains("no such method"), "{err}");
    assert_eq!(server::legacy_calls(), 1);
}

#[test]
fn a_trait_read_of_one_record_moves_kilobytes_from_a_20_mb_store() {
    let store = FerroStore::in_memory();
    store
        .transaction(|txn| {
            let pad = "x".repeat(10_000);
            for i in 0..2_000 {
                txn.put_doc("bulk", &format!("{i:05}"), &json!({ "pad": pad }))?;
            }
            Ok(())
        })
        .unwrap();
    let served = serve(store);
    let client = served.client();
    let project = Catalog::add_project(&client, "/bytes").unwrap();
    let counter = remote(&client).counter();
    counter.reset();
    let got = Catalog::get_project(&client, &project).unwrap().unwrap();
    assert_eq!(got.root, "/bytes");
    let moved = counter.total();
    println!("one trait read moved {moved} frame bytes");
    assert!(moved > 0, "the counter counted nothing");
    assert!(moved < 64 * 1024, "one trait read moved {moved} bytes");
    // The read-only body closed its session.
    served.wait_sessions(0);
}

// ---- serializability ------------------------------------------------------------

/// Put `mine` on call when neither `a` nor `b` is: the write-skew pair.
fn take_call(
    txn: &mut super::Transaction<'_>,
    a: &str,
    b: &str,
    mine: &str,
) -> Result<(), StoreError> {
    let taken = txn.get_doc("oncall", a).is_some() || txn.get_doc("oncall", b).is_some();
    if !taken {
        txn.put_doc("oncall", mine, &json!(true))?;
    }
    Ok(())
}

#[test]
fn write_skew_commits_exactly_one_transaction_of_each_pair() {
    let served = serve(FerroStore::in_memory());
    let uri = served.uri();
    within(60, move || {
        let one = FerroStore::connect_uri(&uri).unwrap();
        for round in 0..5 {
            let a = format!("a{round}");
            let b = format!("b{round}");
            // The other transaction reads both and commits between this
            // one's reads and its commit.
            let two = FerroStore::connect_uri(&uri).unwrap();
            let fired = Cell::new(false);
            let (ha, hb) = (a.clone(), b.clone());
            let _hook = hook::install(
                move |cas: &Cas<'_>| {
                    if cas.attempt == 1 && !fired.replace(true) {
                        two.transaction(|txn| take_call(txn, &ha, &hb, &ha))
                            .unwrap();
                    }
                    false
                },
                false,
            );
            one.transaction(|txn| take_call(txn, &a, &b, &b)).unwrap();
            let on_call = [&a, &b]
                .into_iter()
                .filter(|key| one.get_doc("oncall", key).unwrap().is_some())
                .count();
            assert_eq!(on_call, 1, "round {round}: both transactions committed");
        }
    });
}

/// Add a person named `name` with id `id` unless one is already there:
/// a read of the whole collection (`scan`) or of a query.
fn add_unless_named(
    txn: &mut super::Transaction<'_>,
    name: &str,
    id: &str,
    by_query: bool,
) -> Result<(), StoreError> {
    let taken = if by_query {
        !txn.query_by_field("people", "name", name).is_empty()
    } else {
        txn.scan("people")
            .iter()
            .any(|(_, doc)| doc["name"] == name)
    };
    if !taken {
        txn.put_doc("people", id, &json!({ "name": name }))?;
    }
    Ok(())
}

fn named(store: &FerroStore, name: &str) -> usize {
    store.query_by_field("people", "name", name).unwrap().len()
}

#[test]
fn a_phantom_insert_between_a_scan_and_its_commit_is_caught() {
    let served = serve(FerroStore::in_memory());
    let uri = served.uri();
    within(60, move || {
        let one = FerroStore::connect_uri(&uri).unwrap();
        for (round, by_query) in [(0, false), (1, true)] {
            let name = format!("n{round}");
            let two = FerroStore::connect_uri(&uri).unwrap();
            let fired = Cell::new(false);
            let hname = name.clone();
            let _hook = hook::install(
                move |cas: &Cas<'_>| {
                    if cas.attempt == 1 && !fired.replace(true) {
                        two.transaction(|txn| add_unless_named(txn, &hname, "two", by_query))
                            .unwrap();
                    }
                    false
                },
                false,
            );
            let id = format!("one{round}");
            one.transaction(|txn| add_unless_named(txn, &name, &id, by_query))
                .unwrap();
            assert_eq!(named(&one, &name), 1, "by_query {by_query}: a duplicate");
        }
    });
}

#[test]
fn concurrent_inserters_never_add_a_duplicate() {
    const THREADS: usize = 8;
    let served = serve(FerroStore::in_memory());
    let uri = served.uri();
    within(120, move || {
        for round in 0..4 {
            let name = format!("p{round}");
            thread::scope(|scope| {
                for t in 0..THREADS {
                    let (uri, name) = (&uri, &name);
                    scope.spawn(move || {
                        let store = FerroStore::connect_uri(uri).unwrap();
                        let id = format!("{name}-{t}");
                        until_committed(|| {
                            store.transaction(|txn| add_unless_named(txn, name, &id, t % 2 == 0))
                        });
                    });
                }
            });
            let store = FerroStore::connect_uri(&uri).unwrap();
            assert_eq!(named(&store, &name), 1, "round {round}");
        }
    });
}

#[test]
fn a_body_reads_one_snapshot_while_another_client_commits() {
    let store = FerroStore::in_memory();
    store.put_doc("d", "a", &json!({"n": 0})).unwrap();
    store.put_doc("d", "b", &json!({"n": 0})).unwrap();
    let served = serve(store);
    let one = served.client();
    let other = served.client();
    let fired = Cell::new(false);
    let (a, b) = one
        .transaction(|txn| {
            let a = txn.get_doc("d", "a");
            if !fired.replace(true) {
                other
                    .transaction(|t| {
                        t.put_doc("d", "a", &json!({"n": 1}))?;
                        t.put_doc("d", "b", &json!({"n": 1}))
                    })
                    .unwrap();
            }
            let b = txn.get_doc("d", "b");
            Ok((a, b))
        })
        .unwrap();
    assert_eq!(a, Some(json!({"n": 0})));
    assert_eq!(
        b,
        Some(json!({"n": 0})),
        "the second read saw a later commit"
    );
    assert_eq!(one.get_doc("d", "b").unwrap(), Some(json!({"n": 1})));
}

#[test]
fn a_lost_commit_retries_over_a_new_session_and_leaks_none() {
    let store = FerroStore::in_memory();
    store.put_doc("c", "n", &json!(0)).unwrap();
    let served = serve(store);
    let uri = served.uri();
    let runs = within(30, move || {
        let one = FerroStore::connect_uri(&uri).unwrap();
        let two = FerroStore::connect_uri(&uri).unwrap();
        let fired = Cell::new(false);
        let _hook = hook::install(
            move |cas: &Cas<'_>| {
                if cas.attempt == 1 && !fired.replace(true) {
                    two.put_doc("c", "n", &json!(10)).unwrap();
                }
                false
            },
            false,
        );
        let runs = Cell::new(0);
        one.transaction(|txn| {
            runs.set(runs.get() + 1);
            let n = txn.get_doc("c", "n").and_then(|v| v.as_u64()).unwrap_or(0);
            txn.put_doc("c", "n", &json!(n + 1))
        })
        .unwrap();
        assert_eq!(one.get_doc("c", "n").unwrap(), Some(json!(11)));
        runs.get()
    });
    assert_eq!(runs, 2);
    served.wait_sessions(0);
}

// ---- sessions -------------------------------------------------------------------

fn open(raw: &RemoteClient) -> Result<u64, StoreError> {
    let opened = raw.call("open_session", json!({}))?;
    Ok(opened["session"].as_u64().unwrap())
}

fn read_in(raw: &RemoteClient, session: u64) -> Result<Value, StoreError> {
    raw.call(
        "txn_read",
        json!({"session": session, "read": {"p": "get", "collection": "d", "id": "a"}}),
    )
}

#[test]
fn a_connection_holds_at_most_64_sessions_and_closing_it_drops_them() {
    let served = serve(FerroStore::in_memory());
    let raw = served.raw();
    for _ in 0..SESSIONS_PER_CONN {
        open(&raw).unwrap();
    }
    let err = open(&raw).unwrap_err();
    assert!(
        err.to_string()
            .contains("too many open read sessions: this connection holds 64"),
        "{err}"
    );
    let other = served.raw();
    open(&other).unwrap();
    assert_eq!(served.server.open_sessions(), SESSIONS_PER_CONN + 1);
    drop(raw);
    served.wait_sessions(1);
    // A closed session is gone; a session of another connection is unknown.
    let mine = open(&other).unwrap();
    other
        .call("close_session", json!({"session": mine}))
        .unwrap();
    let err = read_in(&other, mine).unwrap_err();
    assert!(err.to_string().contains("no open read session"), "{err}");
}

#[test]
fn a_server_holds_at_most_1024_sessions() {
    // One credential holds a quarter of them, so four fill the server.
    let principals = SESSIONS_PER_SERVER / SESSIONS_PER_PRINCIPAL;
    let (server, tokens, _dir) = super::tests_auth::token_principals(principals + 1);
    let addr = server.local_addr();
    let conns: Vec<RemoteClient> = tokens[..principals]
        .iter()
        .flat_map(|token| {
            (0..SESSIONS_PER_PRINCIPAL / SESSIONS_PER_CONN)
                .map(move |_| super::tests_auth::token_raw(addr, token))
        })
        .collect();
    for raw in &conns {
        for _ in 0..SESSIONS_PER_CONN {
            open(raw).unwrap();
        }
    }
    assert_eq!(server.open_sessions(), SESSIONS_PER_SERVER);
    let late = super::tests_auth::token_raw(addr, &tokens[principals]);
    let err = open(&late).unwrap_err();
    assert!(
        err.to_string()
            .contains("too many open read sessions: this server holds 1024"),
        "{err}"
    );
    // The sessions already open are kept.
    assert_eq!(server.open_sessions(), SESSIONS_PER_SERVER);
}

/// Every Unix client of a server is the owner's own `unix:<uid>`: the per-credential
/// quota would cap them all together, so only the connection and server limits bind.
#[test]
fn unix_clients_of_one_uid_share_no_per_credential_session_quota() {
    let served = serve(FerroStore::in_memory());
    let conns: Vec<RemoteClient> = (0..SESSIONS_PER_SERVER / SESSIONS_PER_CONN)
        .map(|_| served.raw())
        .collect();
    for raw in &conns {
        for _ in 0..SESSIONS_PER_CONN {
            open(raw).unwrap();
        }
    }
    const { assert!(SESSIONS_PER_SERVER > SESSIONS_PER_PRINCIPAL) };
    assert_eq!(served.server.open_sessions(), SESSIONS_PER_SERVER);
    // The server-wide cap still holds.
    let late = served.raw();
    let err = open(&late).unwrap_err();
    assert!(
        err.to_string()
            .contains("too many open read sessions: this server holds 1024"),
        "{err}"
    );
}

#[test]
fn an_idle_session_expires_and_frees_its_place() {
    let served = serve(FerroStore::in_memory());
    served.server.set_session_idle(Duration::from_millis(100));
    let raw = served.raw();
    let session = open(&raw).unwrap();
    read_in(&raw, session).unwrap();
    for _ in 1..SESSIONS_PER_CONN {
        open(&raw).unwrap();
    }
    thread::sleep(Duration::from_millis(250));
    let err = read_in(&raw, session).unwrap_err();
    assert!(err.to_string().contains("expired"), "{err}");
    // The expired sessions are swept before the limit is checked.
    open(&raw).unwrap();
    assert_eq!(served.server.open_sessions(), 1);
}

#[test]
fn a_session_read_often_still_expires_at_its_lifetime() {
    let served = serve(FerroStore::in_memory());
    served
        .server
        .set_session_lifetime(Duration::from_millis(300));
    let raw = served.raw();
    let session = open(&raw).unwrap();
    let opened = Instant::now();
    // Read well inside the idle limit, until the lifetime has passed.
    let err = loop {
        match read_in(&raw, session) {
            Ok(_) => assert!(
                opened.elapsed() < Duration::from_secs(5),
                "the session outlived its lifetime"
            ),
            Err(err) => break err,
        }
        thread::sleep(Duration::from_millis(40));
    };
    assert!(opened.elapsed() >= Duration::from_millis(300));
    assert!(err.to_string().contains(SESSION_EXPIRED), "{err}");
    assert!(err.to_string().contains("lifetime"), "{err}");
    assert_eq!(served.server.open_sessions(), 0);
}

#[test]
fn closing_a_session_sweeps_the_expired_ones() {
    let served = serve(FerroStore::in_memory());
    served
        .server
        .set_session_lifetime(Duration::from_millis(100));
    let raw = served.raw();
    let first = open(&raw).unwrap();
    let other = served.raw();
    open(&other).unwrap();
    thread::sleep(Duration::from_millis(250));
    assert_eq!(served.server.open_sessions(), 2, "nothing swept yet");
    // Closing a session that is not open still sweeps.
    raw.call("close_session", json!({"session": u64::MAX}))
        .unwrap();
    assert_eq!(served.server.open_sessions(), 0);
    // A read in a swept session says it expired, so the client retries it.
    let err = read_in(&raw, first).unwrap_err();
    assert!(err.to_string().contains(SESSION_EXPIRED), "{err}");
}

#[test]
fn a_transaction_whose_session_expires_runs_again_over_a_new_one() {
    let store = FerroStore::in_memory();
    store.put_doc("d", "a", &json!(1)).unwrap();
    store.put_doc("d", "b", &json!(2)).unwrap();
    let served = serve(store);
    served
        .server
        .set_session_lifetime(Duration::from_millis(200));
    let client = served.client();
    let runs = Cell::new(0);
    let sum = client
        .transaction(|txn| {
            runs.set(runs.get() + 1);
            let a = txn.get_doc("d", "a").and_then(|v| v.as_u64()).unwrap_or(0);
            if runs.get() == 1 {
                // The session reaches its lifetime between two reads.
                thread::sleep(Duration::from_millis(300));
            }
            let b = txn.get_doc("d", "b").and_then(|v| v.as_u64()).unwrap_or(0);
            txn.put_doc("d", "sum", &json!(a + b))?;
            Ok(a + b)
        })
        .unwrap();
    assert_eq!(sum, 3);
    assert_eq!(runs.get(), 2, "the expired attempt ran again");
    assert_eq!(
        served.server.store().get_doc("d", "sum").unwrap(),
        Some(json!(3))
    );
    served.wait_sessions(0);
}

// ---- exactly-once ------------------------------------------------------------------

#[test]
fn a_repeated_request_id_is_applied_once_and_answered_the_same() {
    let served = serve(FerroStore::in_memory());
    let raw = served.raw();
    let store = served.server.store();
    let base = store.last_seq().unwrap();

    let commit = json!({
        "reads": [],
        "ops": [{"op": "put", "collection": "d", "id": "x", "doc": 1}],
        "req": format!("{:032x}", 1u128),
    });
    let first = raw.call("commit_txn", commit.clone()).unwrap();
    let again = raw.call("commit_txn", commit).unwrap();
    assert_eq!(first, json!({"committed": true, "seq": base + 1}));
    assert_eq!(again, first, "a replay answers the first reply");
    assert_eq!(
        store.last_seq().unwrap(),
        base + 1,
        "commit_txn applied twice"
    );

    let put = json!({"collection": "d", "id": "y", "doc": 2, "req": format!("{:032x}", 2u128)});
    let first = raw.call("put_doc", put.clone()).unwrap();
    // From another connection of the same credential too: the table is the
    // server's, keyed by principal.
    let again = served.raw().call("put_doc", put).unwrap();
    assert_eq!(again, first);
    assert_eq!(store.last_seq().unwrap(), base + 2, "put_doc applied twice");

    let publish = json!({"topic": "t", "key": "k", "payload": 1, "req": format!("{:032x}", 3u128)});
    let first = raw.call("publish_mq", publish.clone()).unwrap();
    let again = raw.call("publish_mq", publish).unwrap();
    assert_eq!(again, first);
    assert_eq!(
        store.last_seq().unwrap(),
        base + 3,
        "publish_mq applied twice"
    );

    // An error is answered the same way, and a request without `req` is not
    // remembered.
    let bad = json!({"reads": [], "ops": [{"op": "put", "collection": "__x", "id": "a", "doc": 1}],
        "req": format!("{:032x}", 4u128)});
    let first = raw.call("commit_txn", bad.clone()).unwrap_err().to_string();
    assert_eq!(raw.call("commit_txn", bad).unwrap_err().to_string(), first);
    let plain = json!({"collection": "d", "id": "z", "doc": 3});
    raw.call("put_doc", plain.clone()).unwrap();
    raw.call("put_doc", plain).unwrap();
    assert_eq!(store.last_seq().unwrap(), base + 5);

    let err = raw
        .call(
            "put_doc",
            json!({"collection": "d", "id": "w", "doc": 1, "req": "nope"}),
        )
        .unwrap_err();
    assert!(err.to_string().contains("32 hex digits"), "{err}");
}

#[test]
fn a_remote_write_sends_a_fresh_request_id_per_attempt() {
    let served = serve(FerroStore::in_memory());
    let client = served.client();
    client.put_doc("d", "a", &json!(1)).unwrap();
    client.put_doc("d", "a", &json!(1)).unwrap();
    client
        .transaction(|txn| txn.put_doc("d", "b", &json!(2)))
        .unwrap();
    // Three logical writes, three commits: no id was reused.
    assert_eq!(served.server.store().last_seq().unwrap(), 3);
}

/// The principal of filler id `req`: four of them share the table, so
/// together they can fill it (each holds at most a quarter).
fn who(req: u128) -> Principal {
    Principal::from(format!("filler:{}", req % 4))
}

/// A fifth principal, holding nothing.
fn newcomer() -> Principal {
    Principal::from("newcomer")
}

/// Claim `principal`'s `req` at `at` on `dedup`, which must run it.
fn run_as<'a>(
    dedup: &'a Dedup,
    principal: &Principal,
    req: u128,
    at: Instant,
) -> session::Running<'a> {
    match dedup.claim_at(principal, "put_doc", req, || at) {
        Ok(Claim::Run(running)) => running,
        Ok(Claim::Replay(reply)) => panic!("request {req} replayed {reply}"),
        Err(err) => panic!("request {req} refused: {err}"),
    }
}

/// Claim filler `req` at `at` on `dedup`, which must run it.
fn run_at(dedup: &Dedup, req: u128, at: Instant) -> session::Running<'_> {
    run_as(dedup, &who(req), req, at)
}

/// `dedup` holds filler `req`'s reply at `at`.
fn replays(dedup: &Dedup, req: u128, at: Instant) -> Option<Value> {
    match dedup.claim_at(&who(req), "put_doc", req, || at) {
        Ok(Claim::Replay(reply)) => Some(reply),
        Ok(Claim::Run(_)) => None,
        Err(err) => panic!("request {req} refused: {err}"),
    }
}

/// The reply `dedup` keeps for filler `req`.
fn reply_of(dedup: &Dedup, req: u128) -> Option<Value> {
    dedup.reply_of(&who(req), req)
}

#[test]
fn a_running_request_id_survives_eviction_pressure() {
    let dedup = Dedup::default();
    let base = Instant::now();
    let first = run_at(&dedup, 0, base);
    for req in 1..DEDUP_ENTRIES as u128 {
        run_at(&dedup, req, base).finish(&json!(req));
    }
    // Full, and every id is old enough to forget by count: the oldest
    // answered one goes, not the running one.
    let later = base + DEDUP_MIN_AGE + Duration::from_secs(1);
    run_as(&dedup, &newcomer(), u128::MAX, later).finish(&json!("new"));
    assert_eq!(reply_of(&dedup, 1), None, "the oldest answered id stays");
    // Past the TTL every answered id goes; the running one is kept.
    let late = base + DEDUP_TTL + Duration::from_secs(1);
    run_as(&dedup, &newcomer(), u128::MAX - 1, late).finish(&json!("newer"));
    assert_eq!(reply_of(&dedup, 2), None, "an id past the TTL stays");
    first.finish(&json!("first"));
    assert_eq!(
        reply_of(&dedup, 0),
        Some(json!("first")),
        "the running id was forgotten: a resend of its write would apply it again"
    );
}

#[test]
fn an_answered_request_id_younger_than_two_call_timeouts_is_kept_when_full() {
    let dedup = Dedup::default();
    let base = Instant::now();
    for req in 0..DEDUP_ENTRIES as u128 {
        run_at(&dedup, req, base).finish(&json!(req));
    }
    let young = base + DEDUP_MIN_AGE;
    let err = match dedup.claim_at(&newcomer(), "put_doc", u128::MAX, || young) {
        Err(err) => err,
        Ok(_) => panic!("a full table took a new request id"),
    };
    assert!(err.to_string().contains(DEDUP_FULL), "{err}");
    assert_eq!(
        replays(&dedup, 0, young),
        Some(json!(0)),
        "evicted by count"
    );
    // Older than that, the oldest goes by count and the new id runs.
    let old = young + Duration::from_secs(1);
    run_as(&dedup, &newcomer(), u128::MAX, old).finish(&json!("new"));
    assert_eq!(reply_of(&dedup, 0), None);
    assert_eq!(reply_of(&dedup, 1), Some(json!(1)));
}

#[test]
fn a_table_full_of_running_request_ids_refuses_a_new_write() {
    let dedup = Dedup::default();
    let base = Instant::now();
    let running: Vec<_> = (0..DEDUP_ENTRIES as u128)
        .map(|req| run_at(&dedup, req, base))
        .collect();
    let late = base + DEDUP_TTL + Duration::from_secs(1);
    let err = match dedup.claim_at(&newcomer(), "put_doc", u128::MAX, || late) {
        Err(err) => err,
        Ok(_) => panic!("a running id was dropped for a new one"),
    };
    assert!(err.to_string().contains(DEDUP_FULL), "{err}");
    // Released (each dropped unanswered), the ids make room again.
    drop(running);
    run_as(&dedup, &newcomer(), u128::MAX, late).finish(&json!("new"));
}

#[test]
fn one_credential_holds_at_most_a_quarter_of_the_dedup_table() {
    assert_eq!(DEDUP_PER_PRINCIPAL, DEDUP_ENTRIES / 4);
    let dedup = Dedup::default();
    let hog = Principal::from("hog");
    let base = Instant::now();
    let running: Vec<_> = (0..DEDUP_PER_PRINCIPAL as u128)
        .map(|req| run_as(&dedup, &hog, req, base))
        .collect();
    // At its quota, with the table three quarters empty.
    let err = match dedup.claim_at(&hog, "put_doc", u128::MAX, || base) {
        Err(err) => err,
        Ok(_) => panic!("a credential took more than its quota"),
    };
    assert!(err.to_string().contains(DEDUP_QUOTA), "{err}");
    assert!(session::is_dedup_full(&err), "a quota refusal ran nothing");
    // Another credential is not refused.
    run_as(&dedup, &newcomer(), 1, base).finish(&json!("other"));
    // Answered ids younger than two call timeouts are still kept.
    for (req, claim) in running.into_iter().enumerate() {
        claim.finish(&json!(req));
    }
    let young = base + DEDUP_MIN_AGE;
    let err = match dedup.claim_at(&hog, "put_doc", u128::MAX, || young) {
        Err(err) => err,
        Ok(_) => panic!("a young answered id was forgotten for the quota"),
    };
    assert!(err.to_string().contains(DEDUP_QUOTA), "{err}");
    // Older, the credential's own oldest goes, and nobody else's.
    let old = young + Duration::from_secs(1);
    run_as(&dedup, &hog, u128::MAX, old).finish(&json!("new"));
    assert_eq!(dedup.reply_of(&hog, 0), None);
    assert_eq!(dedup.reply_of(&hog, 1), Some(json!(1)));
    assert_eq!(dedup.reply_of(&newcomer(), 1), Some(json!("other")));
}

/// A Unix principal has no per-credential dedup quota: it can hold more than a
/// quarter of the table, up to the table's own limit.
#[test]
fn a_unix_principal_is_not_held_to_the_dedup_quota() {
    let dedup = Dedup::default();
    let unix = Principal::from("unix:1000");
    let base = Instant::now();
    let running: Vec<_> = (0..(DEDUP_PER_PRINCIPAL + 500) as u128)
        .map(|req| run_as(&dedup, &unix, req, base))
        .collect();
    assert!(running.len() > DEDUP_PER_PRINCIPAL);
    // Still bounded by the table as a whole.
    let more: Vec<_> = (running.len() as u128..DEDUP_ENTRIES as u128)
        .map(|req| run_as(&dedup, &unix, req, base))
        .collect();
    let err = match dedup.claim_at(&unix, "put_doc", u128::MAX, || base) {
        Err(err) => err,
        Ok(_) => panic!("the table took more than its limit"),
    };
    assert!(err.to_string().contains(DEDUP_FULL), "{err}");
    assert!(!err.to_string().contains(DEDUP_QUOTA), "{err}");
    drop((running, more));
}

#[test]
fn a_request_id_is_kept_per_credential_and_per_method() {
    let dedup = Dedup::default();
    let (a, b) = (Principal::from("a"), Principal::from("b"));
    let now = Instant::now();
    run_as(&dedup, &a, 7, now).finish(&json!("a's reply"));
    // B's equal id is B's own: it runs.
    run_as(&dedup, &b, 7, now).finish(&json!("b's reply"));
    match dedup.claim_at(&a, "put_doc", 7, || now) {
        Ok(Claim::Replay(reply)) => assert_eq!(reply, json!("a's reply")),
        _ => panic!("a's replay was not answered with a's reply"),
    }
    match dedup.claim_at(&a, "publish_mq", 7, || now) {
        Err(err) => assert!(
            err.to_string().contains(session::DEDUP_OTHER_METHOD),
            "{err}"
        ),
        Ok(_) => panic!("an id was replayed or run for another method"),
    }
}

/// Fill `served`'s exactly-once table with running ids, run `write` while
/// they are held, and free them after a moment: `write` must wait it out.
fn while_the_dedup_table_is_full(served: &Served, write: impl FnOnce() + Send) {
    let dedup = served.server.dedup();
    let refused = served.server.dedup_refusals();
    thread::scope(|scope| {
        // Claim fresh ids until the table refuses one: it is full. Four
        // other credentials fill it, a quarter each.
        let mut held = Vec::new();
        for req in 1..=DEDUP_ENTRIES as u128 + 1 {
            match dedup.claim(&who(req), "put_doc", req) {
                Ok(Claim::Run(running)) => held.push(running),
                Ok(Claim::Replay(_)) => panic!("a fresh id replayed"),
                Err(err) => {
                    assert!(err.to_string().contains(DEDUP_FULL), "{err}");
                    break;
                }
            }
        }
        assert!(held.len() > DEDUP_ENTRIES / 2, "the table was full already");
        scope.spawn(move || {
            thread::sleep(Duration::from_millis(40));
            drop(held);
        });
        write();
    });
    assert!(
        served.server.dedup_refusals() > refused,
        "the table was never full for the write"
    );
}

#[test]
fn a_full_dedup_table_is_a_lost_attempt_retried_until_it_has_room() {
    let served = serve(FerroStore::in_memory());
    let client = served.client();
    // `commit_txn`, in a transaction that also read.
    while_the_dedup_table_is_full(&served, || {
        client
            .transaction(|txn| {
                let seen = txn.get_doc("docs", "a");
                txn.put_doc("docs", "a", &json!({ "seen": seen }))
            })
            .unwrap();
    });
    assert_eq!(
        client.get_doc("docs", "a").unwrap(),
        Some(json!({ "seen": null }))
    );
    // `publish_mq`, a write RPC of its own.
    while_the_dedup_table_is_full(&served, || {
        assert_eq!(client.publish_mq("t", "k", &json!({})).unwrap(), 1);
    });
    assert_eq!(client.poll_mq_group("t", "g", 10).unwrap().len(), 1);
}

// ---- commit validation of ops ---------------------------------------------------------

#[test]
fn commit_txn_refuses_reserved_ops_but_the_schema_version() {
    let served = serve(FerroStore::in_memory());
    let raw = served.raw();
    for (collection, id, doc) in [
        ("__meta", "other", json!(1)),
        ("__private", "x", json!(1)),
        ("__meta", "schema_version", json!(999)),
        ("__meta", "schema_version", json!("8")),
    ] {
        let err = raw
            .call(
                "commit_txn",
                json!({"reads": [], "ops": [{"op": "put", "collection": collection, "id": id, "doc": doc}]}),
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("reserved"),
            "{collection}/{id}: {err}"
        );
    }
    let err = raw
        .call(
            "commit_txn",
            json!({"reads": [], "ops": [{"op": "delete", "collection": "__meta", "id": "x"}]}),
        )
        .unwrap_err();
    assert!(err.to_string().contains("reserved"), "{err}");
    let version = crate::FORMAT_WITH_WORKFLOW;
    raw.call(
        "commit_txn",
        json!({"reads": [], "ops": [{"op": "put", "collection": "__meta", "id": "schema_version", "doc": version}]}),
    )
    .unwrap();
    assert_eq!(served.server.store().last_seq().unwrap(), 1);
}

/// One `commit_txn` of `ops` with an empty read set.
fn commit_ops(raw: &RemoteClient, ops: Value) -> Result<Value, StoreError> {
    raw.call("commit_txn", json!({"reads": [], "ops": ops}))
}

fn put_schema(version: Value) -> Value {
    json!({"op": "put", "collection": "__meta", "id": "schema_version", "doc": version})
}

#[test]
fn a_remote_commit_only_raises_the_schema_version_of_a_store_with_data() {
    let served = serve(FerroStore::in_memory());
    let raw = served.raw();
    let store = served.server.store();
    let delete_schema = json!({"op": "delete", "collection": "__meta", "id": "schema_version"});
    // An empty store: a lower version is accepted (what a migration writes).
    commit_ops(&raw, json!([put_schema(json!(crate::FORMAT_WITH_STATE))])).unwrap();
    assert_eq!(
        store.format_version().unwrap(),
        Some(crate::FORMAT_WITH_STATE)
    );
    commit_ops(
        &raw,
        json!([{"op": "put", "collection": "d", "id": "a", "doc": 1}]),
    )
    .unwrap();
    // A raise is accepted.
    commit_ops(&raw, json!([put_schema(json!(crate::FORMAT_WITH_WIRE))])).unwrap();
    assert_eq!(
        store.format_version().unwrap(),
        Some(crate::FORMAT_WITH_WIRE)
    );
    // A lower or equal version, or a delete, is refused whole on a store with data.
    let before = store.last_seq().unwrap();
    for (op, says) in [
        (put_schema(json!(crate::FORMAT_WITH_STATE)), "from 7 to 6"),
        (put_schema(json!(crate::FORMAT_WITH_WIRE)), "from 7 to 7"),
        (delete_schema.clone(), "delete the schema version 7"),
    ] {
        let err = commit_ops(
            &raw,
            json!([{"op": "put", "collection": "d", "id": "b", "doc": 2}, op]),
        )
        .unwrap_err();
        assert!(err.to_string().contains("commit_txn refused"), "{err}");
        assert!(err.to_string().contains(says), "{err}");
    }
    assert_eq!(store.last_seq().unwrap(), before, "a refused commit wrote");
    assert_eq!(store.get_doc("d", "b").unwrap(), None);
    assert_eq!(
        store.format_version().unwrap(),
        Some(crate::FORMAT_WITH_WIRE)
    );
    // Off the ladder, or another reserved op: refused as reserved.
    for op in [
        put_schema(json!(999)),
        json!({"op": "put", "collection": "__other", "id": "x", "doc": 1}),
        json!({"op": "delete", "collection": "__meta", "id": "other"}),
    ] {
        let err = commit_ops(&raw, json!([op])).unwrap_err();
        assert!(err.to_string().contains("reserved"), "{err}");
    }
    // A store whose only data is a vector collection holds data too.
    let vectors = with_vectors();
    let err = commit_ops(&vectors.raw(), json!([delete_schema.clone()])).unwrap_err();
    assert!(err.to_string().contains("commit_txn refused"), "{err}");
    // Emptied again, the store takes the delete.
    commit_ops(
        &raw,
        json!([{"op": "delete", "collection": "d", "id": "a"}]),
    )
    .unwrap();
    commit_ops(&raw, json!([delete_schema])).unwrap();
    assert_eq!(store.format_version().unwrap(), None);
}

#[test]
fn a_migration_into_an_empty_remote_store_copies_the_schema_version() {
    let dir = tempfile::tempdir().unwrap();
    let redb = crate::RedbStore::open(&dir.path().join("src.redb")).unwrap();
    let project = Catalog::add_project(&redb, "/migrated").unwrap();
    let served = serve(FerroStore::in_memory());
    let client = served.client();
    super::migrate_redb_to_ferrostore(&redb, &client).unwrap();
    let store = served.server.store();
    assert_eq!(
        store.format_version().unwrap(),
        redb.format_version().unwrap()
    );
    assert_eq!(
        store.get_project(&project).unwrap().unwrap().root,
        "/migrated"
    );
    // Into a store that now holds data, a second migration is refused.
    assert!(super::migrate_redb_to_ferrostore(&redb, &client).is_err());
}

#[test]
fn a_read_whose_hash_changed_loses_and_writes_nothing() {
    let store = FerroStore::in_memory();
    store.put_doc("d", "a", &json!(1)).unwrap();
    let served = serve(store);
    let raw = served.raw();
    let read = json!({"p": "get", "collection": "d", "id": "a"});
    let reply = raw.call("txn_read", json!({"read": read})).unwrap();
    assert_eq!(reply["result"], json!({"doc": 1}));
    let hash = reply["hash"].as_str().unwrap().to_string();
    assert_eq!(hash.len(), 64);
    let session = reply["session"].as_u64().unwrap();
    assert_eq!(reply["base_seq"], json!(1));
    served.server.store().put_doc("d", "a", &json!(2)).unwrap();
    let lost = raw
        .call(
            "commit_txn",
            json!({
                "session": session,
                "base_seq": 1,
                "reads": [{"read": read, "hash": hash}],
                "ops": [{"op": "put", "collection": "d", "id": "b", "doc": 1}],
            }),
        )
        .unwrap();
    assert_eq!(lost, json!({"committed": false, "seq": null}));
    assert_eq!(served.server.store().get_doc("d", "b").unwrap(), None);
    // The commit ended the session.
    assert_eq!(served.server.open_sessions(), 0);
}

// ---- vectors ----------------------------------------------------------------------------

fn with_vectors() -> Served {
    let store = FerroStore::in_memory();
    store
        .create_vector_collection("v", 2, VectorStorage::F32, VectorMetric::L2, 4, 16, 16)
        .unwrap();
    store.upsert_vector("v", "base", &[0.0, 1.0], None).unwrap();
    serve(store)
}

#[test]
fn a_remote_transaction_refuses_to_read_its_own_vector_writes() {
    let served = with_vectors();
    let client = served.client();
    let err = client
        .transaction(|txn| {
            txn.upsert_vector("v", "a", &[1.0, 0.0], None)?;
            txn.get_vector("v", "base")
        })
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("remote transaction cannot read back its own vector writes"),
        "{err}"
    );
    let err = client
        .transaction(|txn| {
            txn.create_vector_collection("w", 2, VectorStorage::F32, VectorMetric::L2, 4, 16, 16)?;
            assert!(txn.has_vector_collection("w"));
            txn.upsert_vector("w", "a", &[1.0, 0.0], None)?;
            txn.get_vector("w", "a")
        })
        .unwrap_err();
    assert!(err.to_string().contains("cannot read back"), "{err}");
    let store = served.server.store();
    assert_eq!(
        store.get_vector("v", "a").unwrap(),
        None,
        "nothing committed"
    );
    assert!(!store.has_vector_collection("w").unwrap());
}

#[test]
fn a_vector_upsert_in_a_remote_transaction_is_applied_and_validated_on_the_server() {
    let served = with_vectors();
    let uri = served.uri();
    let runs = within(30, move || {
        let one = FerroStore::connect_uri(&uri).unwrap();
        let two = FerroStore::connect_uri(&uri).unwrap();
        let fired = Cell::new(false);
        let _hook = hook::install(
            move |cas: &Cas<'_>| {
                if cas.attempt == 1 && !fired.replace(true) {
                    two.upsert_vector("v", "b", &[0.5, 0.5], None).unwrap();
                }
                false
            },
            false,
        );
        let runs = Cell::new(0);
        one.transaction(|txn| {
            runs.set(runs.get() + 1);
            // A read of a vector the body did not write is validated.
            let seen = txn.get_vector("v", "b")?;
            let meta = json!({"saw_b": seen.is_some()});
            txn.upsert_vector("v", "a", &[1.0, 0.0], Some(&meta))?;
            txn.upsert_vector("v", "a2", &[0.9, 0.1], None)
        })
        .unwrap();
        runs.get()
    });
    assert_eq!(runs, 2, "the lost commit ran the body again");
    let store = served.server.store();
    let (values, meta) = store.get_vector("v", "a").unwrap().unwrap();
    assert_eq!(values, vec![1.0, 0.0]);
    assert_eq!(meta, Some(json!({"saw_b": true})));
    // The server linked the new vectors into its graph.
    let state = store.read_state().unwrap();
    let col = &state.vectors["v"];
    assert!(col.entry.is_some());
    assert!(
        col.vectors["a"]
            .neighbors
            .iter()
            .any(|layer| !layer.is_empty()),
        "the server did not link `a`"
    );
    let hits = store.search_vectors("v", &[1.0, 0.0], 1, None).unwrap();
    assert_eq!(hits[0].0, "a");
    // A wrong length is refused before anything is sent.
    let client = served.client();
    let before = store.last_seq().unwrap();
    assert!(
        client
            .transaction(|txn| txn.upsert_vector("v", "bad", &[1.0], None))
            .is_err()
    );
    assert_eq!(store.last_seq().unwrap(), before);
}

// ---- result size ----------------------------------------------------------------------------

#[test]
fn a_read_larger_than_a_frame_is_an_error_naming_the_primitive_and_collection() {
    let store = FerroStore::in_memory();
    let pad = "y".repeat(1 << 20);
    store
        .transaction(|txn| {
            for i in 0..17 {
                txn.put_doc("big", &format!("{i:02}"), &json!(pad))?;
            }
            Ok(())
        })
        .unwrap();
    let served = serve(store);
    let client = served.client();
    // The body ignores the failed read; the transaction still fails with it.
    let err = client
        .transaction(|txn| {
            let _ = txn.scan("big");
            txn.put_doc("d", "after", &json!(1))
        })
        .unwrap_err();
    let text = err.to_string();
    assert!(text.contains("scan of `big`"), "{text}");
    assert!(text.contains("16 MiB"), "{text}");
    assert_eq!(served.server.store().get_doc("d", "after").unwrap(), None);
    // A read that opens a session and fails leaves none open: the client
    // never learned its id.
    let err = served
        .raw()
        .call(
            "txn_read",
            json!({"read": {"p": "scan", "collection": "big"}}),
        )
        .unwrap_err();
    assert!(err.to_string().contains("16 MiB"), "{err}");
    assert_eq!(
        served.server.open_sessions(),
        0,
        "the failed read pinned a session"
    );
    // One document of it is still read.
    let one = client
        .transaction(|txn| Ok(txn.get_doc("big", "03")))
        .unwrap();
    assert_eq!(one, Some(json!(pad)));
}
