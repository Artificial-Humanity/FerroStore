//! Remote transactions on the server (spec §6.1): read sessions pinned at
//! one snapshot, the read primitives a remote `Txn` asks for, the commit
//! that validates a read set by value, and the table of recent request ids
//! that makes a write exactly-once.
//!
//! A read primitive's answer is canonical JSON. Its hash is SHA-256 of that
//! JSON's bytes, computed here on the server both when the read is served
//! and when the commit validates it, so the client never re-encodes a value
//! to hash it.

use super::auth::Principal;
use super::columnar;
use super::doc;
use super::layout::{Op, backend};
use super::state::{self, DocState};
use super::vector::{self, VectorParams};
use super::{FerroStore, lock_mutex, wait_mutex};
use fl_core::StoreError;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// A session no read or commit has used for this long is gone.
pub(super) const SESSION_IDLE: Duration = Duration::from_secs(30);
/// A session open this long is gone, however often it is read: a reader
/// cannot pin one version forever.
pub(super) const SESSION_LIFETIME: Duration = Duration::from_secs(300);
/// The most sessions one connection holds open.
pub(super) const SESSIONS_PER_CONN: usize = 64;
/// The most sessions one server holds open.
pub(super) const SESSIONS_PER_SERVER: usize = 1024;
/// The most sessions one TCP principal (`token:*`, `psk`) holds open, over all its
/// connections: a quarter of the server's, so one credential cannot take them all.
/// A `unix:<uid>` principal has no such quota: every Unix client of a server is the
/// owner's own uid (`SO_PEERCRED`), so only the connection and server limits bound it.
pub(super) const SESSIONS_PER_PRINCIPAL: usize = SESSIONS_PER_SERVER / 4;
/// The start of the error a principal at [`SESSIONS_PER_PRINCIPAL`] gets.
pub(super) const SESSION_QUOTA: &str = "session quota for this credential reached";
/// The most request ids the exactly-once table remembers.
pub(super) const DEDUP_ENTRIES: usize = 10_000;
/// The most request ids one TCP principal (`token:*`, `psk`) holds in the table: a
/// quarter of it. A `unix:<uid>` principal has no such quota, only the table's own limit.
pub(super) const DEDUP_PER_PRINCIPAL: usize = DEDUP_ENTRIES / 4;

/// `true` when the per-principal quotas apply to `principal`: a TCP credential
/// (`token:*`, `psk`). A `unix:<uid>` principal is the server's own user, shared by
/// every local client, so a per-credential quota would cap all of them together.
pub(super) fn quota_applies(principal: &str) -> bool {
    !principal.starts_with("unix:")
}
/// How long the exactly-once table remembers a request id.
pub(super) const DEDUP_TTL: Duration = Duration::from_secs(600);
/// An answered request id younger than this is kept even when the table is
/// full: the client resends a write once, after its call timeout.
pub(super) const DEDUP_MIN_AGE: Duration =
    Duration::from_secs(2 * super::server::CALL_TIMEOUT.as_secs());
/// The start of the error a new write gets while the exactly-once table is
/// full of ids it may not forget yet.
pub(super) const DEDUP_FULL: &str = "dedup table full, retry";
/// The start of the error a new write gets while its principal holds
/// [`DEDUP_PER_PRINCIPAL`] ids the table may not forget yet. Like
/// [`DEDUP_FULL`] it ran nothing.
pub(super) const DEDUP_QUOTA: &str = "dedup quota for this credential reached, retry";
/// The start of the error a write gets when its principal already used its
/// request id for another method: it runs nothing, and is not answered with
/// that method's reply.
pub(super) const DEDUP_OTHER_METHOD: &str = "request id already used for another method";
/// Room a read's reply keeps in its frame for the envelope around the result.
const ENVELOPE_BYTES: usize = 4096;

/// A transaction's reads, each with its answer's hash, in the order read.
pub(super) type ReadSet = Vec<(ReadKey, String)>;

/// An open read session: its id and the `last_seq` of the state it pinned.
pub(super) type Pinned = (u64, u64);

/// One read primitive of a remote `Txn` (spec §6.1). Its JSON form is the
/// `read` of `txn_read` and of each entry of `commit_txn`'s `reads`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "p", rename_all = "snake_case")]
pub(super) enum ReadKey {
    /// `null`, or `{"doc": value}`.
    Get { collection: String, id: String },
    /// `[[id, doc], ...]` in id order.
    Scan { collection: String },
    /// The ids whose top-level `field` matches `value`, ascending.
    Query {
        collection: String,
        field: String,
        value: String,
    },
    /// `null`, or the column table's columns.
    Columns { name: String },
    /// `null`, or the vector collection's parameters.
    VectorParams { name: String },
    /// `{"err": message}`, `null` for a missing id, or `{"values", "metadata"?}`.
    GetVector { collection: String, id: String },
    /// `{"collections", "tables", "vectors"}`: the names that hold anything.
    Inventory,
}

impl ReadKey {
    /// The primitive's name, as an error names it.
    pub(super) fn primitive(&self) -> &'static str {
        match self {
            Self::Get { .. } => "get",
            Self::Scan { .. } => "scan",
            Self::Query { .. } => "query",
            Self::Columns { .. } => "table_columns",
            Self::VectorParams { .. } => "vector",
            Self::GetVector { .. } => "get_vector",
            Self::Inventory => "inventory",
        }
    }

    /// The collection, table or vector collection the primitive reads.
    pub(super) fn subject(&self) -> &str {
        match self {
            Self::Get { collection, .. }
            | Self::Scan { collection }
            | Self::Query { collection, .. }
            | Self::GetVector { collection, .. } => collection,
            Self::Columns { name } | Self::VectorParams { name } => name,
            Self::Inventory => "*",
        }
    }
}

/// The canonical answer to `key` at `state`. An error that depends on the
/// state (a missing vector collection) is part of the answer, so a commit
/// validates it as it validates a value.
pub(super) fn eval(state: &DocState, key: &ReadKey) -> Value {
    match key {
        ReadKey::Get { collection, id } => match state.docs.get(collection, id) {
            Some(doc) => json!({ "doc": doc }),
            None => Value::Null,
        },
        ReadKey::Scan { collection } => Value::Array(
            state
                .docs
                .in_collection(collection)
                .map(|(id, doc)| json!([id, doc]))
                .collect(),
        ),
        ReadKey::Query {
            collection,
            field,
            value,
        } => Value::Array(
            state::query_by_field(state, collection, field, value)
                .into_iter()
                .map(Value::String)
                .collect(),
        ),
        ReadKey::Columns { name } => match state.tables.get(name) {
            Some(table) => Value::Array(
                table
                    .columns
                    .iter()
                    .map(columnar::column_def_json)
                    .collect(),
            ),
            None => Value::Null,
        },
        ReadKey::VectorParams { name } => match state.vectors.get(name) {
            Some(col) => VectorParams::of(col).to_json(),
            None => Value::Null,
        },
        ReadKey::GetVector { collection, id } => match stored_vector(state, collection, id) {
            Ok(value) => value,
            Err(err) => json!({ "err": err.to_string() }),
        },
        ReadKey::Inventory => json!({
            "collections": state.docs.collections().collect::<Vec<_>>(),
            "tables": state.tables.keys().collect::<Vec<_>>(),
            "vectors": state.vectors.keys().collect::<Vec<_>>(),
        }),
    }
}

fn stored_vector(state: &DocState, collection: &str, id: &str) -> Result<Value, StoreError> {
    let Some(col) = state.vectors.get(collection) else {
        return Err(backend(format!("no vector collection `{collection}`")));
    };
    vector::check_vector_id(id)?;
    let Some(record) = col.vectors.get(id) else {
        return Ok(Value::Null);
    };
    let values = vector::dequantize(col.storage, &record.bytes)?;
    vector::vector_get_json(&values, &record.metadata)
}

/// SHA-256 of `bytes`, as 64 lower-case hex digits.
fn hex_digest(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn encode(result: &Value) -> Result<Vec<u8>, StoreError> {
    serde_json::to_vec(result).map_err(|err| backend(format!("encode read result: {err}")))
}

/// The answer to `key` at `state` and its hash, for a reply. An answer
/// larger than a frame holds is an error that names the primitive and the
/// collection (spec §6.1).
pub(super) fn answer(state: &DocState, key: &ReadKey) -> Result<(Value, String), StoreError> {
    let result = eval(state, key);
    let bytes = encode(&result)?;
    let cap = super::server::MAX_FRAME_BYTES as usize - ENVELOPE_BYTES;
    if bytes.len() > cap {
        return Err(backend(format!(
            "{} of `{}`: the result is {} bytes, more than one 16 MiB frame holds \
             (a paginated read is a known limit)",
            key.primitive(),
            key.subject(),
            bytes.len()
        )));
    }
    Ok((result, hex_digest(&bytes)))
}

/// The hash of `key`'s answer at `state`, as a commit recomputes it.
fn fingerprint(state: &DocState, key: &ReadKey) -> Result<String, StoreError> {
    Ok(hex_digest(&encode(&eval(state, key))?))
}

/// What a remote commit's op does to the schema version.
#[derive(Clone, Copy)]
enum SchemaOp {
    /// A put of this version, which is on this build's ladder.
    Put(u64),
    Delete,
}

/// A reserved op a remote commit may carry: the schema version, which only
/// the store's own bodies write (a raise to a version on this build's
/// ladder, or the migration's copy or removal of it). Any other reserved op
/// is refused, as public `put_doc` and `delete_doc` refuse it. `Some` names
/// a schema-version op, which [`admit_schema_op`] then checks against the
/// committed state.
fn admit_remote_op(op: &Op) -> Result<Option<SchemaOp>, StoreError> {
    let (collection, id, version) = match op {
        Op::Put {
            collection,
            id,
            doc,
        } => (collection, id, Some(doc.as_u64())),
        Op::Delete { collection, id } => (collection, id, None),
        _ => return Ok(None),
    };
    if !doc::is_reserved(collection) {
        return Ok(None);
    }
    if collection == doc::SCHEMA_COLLECTION && id == doc::SCHEMA_KEY {
        match version {
            None => return Ok(Some(SchemaOp::Delete)),
            Some(Some(to)) if crate::check_format(Some(to)).is_ok() => {
                return Ok(Some(SchemaOp::Put(to)));
            }
            Some(_) => {}
        }
    }
    doc::refuse_reserved(collection, "commit_txn").map(|()| None)
}

/// `true` when `state` holds nothing outside `__meta`: no other document
/// collection, no column table, no vector collection. That is the target
/// `migrate_redb_to_ferrostore` copies into, the one write that lowers or
/// deletes the schema version.
fn holds_only_meta(state: &DocState) -> bool {
    state
        .docs
        .collections()
        .all(|collection| collection == doc::SCHEMA_COLLECTION)
        && state.tables.is_empty()
        && state.vectors.is_empty()
}

/// `Err` unless a remote commit may apply `op` to the schema version of the
/// committed state `current` (before any of the commit's ops). A put that
/// raises the version is applied, as `raise_format` raises it. A put that
/// does not raise it, or a delete, is applied only to a store that holds
/// nothing outside `__meta` ([`holds_only_meta`]): lowering the version of
/// a store with data would let an older build open data it cannot read.
fn admit_schema_op(current: &DocState, op: SchemaOp) -> Result<(), StoreError> {
    let now = doc::schema_version_in(|collection, id| current.docs.get(collection, id).cloned())?
        .unwrap_or(crate::FORMAT_VERSION);
    if let SchemaOp::Put(to) = op
        && to > now
    {
        return Ok(());
    }
    if holds_only_meta(current) {
        return Ok(());
    }
    let what = match op {
        SchemaOp::Put(to) => format!("set the schema version from {now} to {to}"),
        SchemaOp::Delete => format!("delete the schema version {now}"),
    };
    Err(backend(format!(
        "commit_txn refused: it would {what} on a store that holds data; a remote commit \
         only raises the schema version, except into an empty store"
    )))
}

/// `commit_txn` (spec §6.1): inside the write lock, as one write body of
/// the group, recompute every read against the committed state. All equal:
/// stage `ops`, each checked as a local write checks it, and commit. Any
/// different: write nothing and reply `committed: false` (lost).
pub(super) fn commit_txn(
    store: &FerroStore,
    sessions: &Sessions,
    conn: u64,
    params: &Value,
) -> Result<Value, StoreError> {
    // The session ends with its commit, lost or not.
    if let Some(session) = optional_u64(params, "session")? {
        let pinned = sessions.close(conn, session);
        if let (Some(pinned), Some(base_seq)) = (pinned, optional_u64(params, "base_seq")?)
            && pinned.last_seq != base_seq
        {
            return Err(backend(format!(
                "commit_txn base_seq {base_seq} is not session {session}'s snapshot {}",
                pinned.last_seq
            )));
        }
    }
    let reads = parse_reads(params.get("reads"))?;
    let ops = super::server::parse_ops(
        params
            .get("ops")
            .ok_or_else(|| backend("missing param `ops`"))?,
    )?;
    let silent = match params.get("silent") {
        None => false,
        Some(Value::Bool(silent)) => *silent,
        Some(_) => return Err(backend("param `silent` is not a bool")),
    };
    let mut schema_ops = Vec::new();
    for op in &ops {
        schema_ops.extend(admit_remote_op(op)?);
    }
    let (committed, seq) = store.write_local(silent, |txn| {
        let current = txn.local_arc()?;
        for (key, hash) in &reads {
            if fingerprint(&current, key)? != *hash {
                return Ok(false);
            }
        }
        // Checked against the committed state before any op of this commit.
        for op in &schema_ops {
            admit_schema_op(&current, *op)?;
        }
        drop(current);
        for op in &ops {
            txn.stage(op.clone())?;
        }
        Ok(true)
    })?;
    Ok(json!({ "committed": committed, "seq": seq }))
}

fn parse_reads(value: Option<&Value>) -> Result<Vec<(ReadKey, String)>, StoreError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let items = value
        .as_array()
        .ok_or_else(|| backend("param `reads` is not an array"))?;
    let mut reads = Vec::with_capacity(items.len());
    for item in items {
        let key = parse_key(item.get("read"))?;
        let hash = item
            .get("hash")
            .and_then(Value::as_str)
            .ok_or_else(|| backend("a read is missing its hash"))?;
        reads.push((key, hash.to_string()));
    }
    Ok(reads)
}

pub(super) fn parse_key(value: Option<&Value>) -> Result<ReadKey, StoreError> {
    let value = value.ok_or_else(|| backend("missing param `read`"))?;
    ReadKey::deserialize(value).map_err(|err| backend(format!("read primitive: {err}")))
}

pub(super) fn optional_u64(params: &Value, key: &str) -> Result<Option<u64>, StoreError> {
    match params.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| backend(format!("param `{key}` is not an integer"))),
    }
}

// ---- sessions ----------------------------------------------------------------

/// The start of the error a read in a session that is no longer open gets:
/// it expired, idle or at its lifetime, or was swept. The client treats it
/// as a lost attempt and runs the body again over a new session.
pub(super) const SESSION_EXPIRED: &str = "read session expired, retry";

/// `true` for the error [`SESSION_EXPIRED`] starts, also as the client
/// rebuilds it from the server's message.
pub(super) fn is_session_expired(err: &StoreError) -> bool {
    matches!(err, StoreError::Backend(detail) if detail.contains(SESSION_EXPIRED))
}

/// `true` for the refusal [`DEDUP_FULL`] or [`DEDUP_QUOTA`] starts, also as
/// the client rebuilds it from the server's message: the write ran
/// nothing, so the client may send it again.
pub(super) fn is_dedup_full(err: &StoreError) -> bool {
    matches!(err, StoreError::Backend(detail)
        if detail.contains(DEDUP_FULL) || detail.contains(DEDUP_QUOTA))
}

fn expired(detail: impl std::fmt::Display) -> StoreError {
    backend(format!("{SESSION_EXPIRED}: {detail}"))
}

/// A `txn_read` sweeps expired sessions once per this many reads.
const SWEEP_EVERY_READS: u64 = 64;
/// The most swept session ids remembered, so a read in one says it expired.
const SWEPT_REMEMBERED: usize = 2 * SESSIONS_PER_SERVER;

struct Session {
    conn: u64,
    principal: Principal,
    state: Arc<DocState>,
    opened: Instant,
    last_used: Instant,
}

#[derive(Default)]
struct Open {
    sessions: HashMap<u64, Session>,
    /// `(session, conn)` of the sessions a sweep dropped, newest last.
    swept: VecDeque<(u64, u64)>,
}

/// Every open read session of one server, keyed by a server-wide id.
pub(super) struct Sessions {
    open: Mutex<Open>,
    next: AtomicU64,
    /// [`SESSION_IDLE`] in milliseconds; a test lowers it.
    idle_ms: AtomicU64,
    /// [`SESSION_LIFETIME`] in milliseconds; a test lowers it.
    lifetime_ms: AtomicU64,
    /// Reads served, for [`SWEEP_EVERY_READS`].
    reads: AtomicU64,
}

fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

impl Sessions {
    pub(super) fn new() -> Self {
        Self {
            open: Mutex::new(Open::default()),
            next: AtomicU64::new(1),
            idle_ms: AtomicU64::new(millis(SESSION_IDLE)),
            lifetime_ms: AtomicU64::new(millis(SESSION_LIFETIME)),
            reads: AtomicU64::new(0),
        }
    }

    fn idle(&self) -> Duration {
        Duration::from_millis(self.idle_ms.load(Ordering::Relaxed))
    }

    fn lifetime(&self) -> Duration {
        Duration::from_millis(self.lifetime_ms.load(Ordering::Relaxed))
    }

    #[cfg(test)]
    pub(super) fn set_idle(&self, idle: Duration) {
        self.idle_ms.store(millis(idle), Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(super) fn set_lifetime(&self, lifetime: Duration) {
        self.lifetime_ms.store(millis(lifetime), Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(super) fn count(&self) -> usize {
        lock_mutex(&self.open).sessions.len()
    }

    /// Why `session` is no longer usable at `now`: idle past the limit, or
    /// open past its lifetime. `None` while it is usable.
    fn expiry(&self, session: &Session, now: Instant) -> Option<String> {
        let idle = self.idle();
        let lifetime = self.lifetime();
        if now.duration_since(session.last_used) > idle {
            Some(format!("idle more than {} ms", idle.as_millis()))
        } else if now.duration_since(session.opened) > lifetime {
            Some(format!(
                "open more than its {} ms lifetime",
                lifetime.as_millis()
            ))
        } else {
            None
        }
    }

    /// Drop every expired session, remembering its id.
    fn sweep(&self, open: &mut Open, now: Instant) {
        let gone: Vec<(u64, u64)> = open
            .sessions
            .iter()
            .filter(|(_, session)| self.expiry(session, now).is_some())
            .map(|(id, session)| (*id, session.conn))
            .collect();
        for (id, conn) in gone {
            open.sessions.remove(&id);
            open.swept.push_back((id, conn));
        }
        while open.swept.len() > SWEPT_REMEMBERED {
            open.swept.pop_front();
        }
    }

    /// Pin `state` for `conn` of `principal`. Expired sessions are dropped
    /// first; then a connection, a TCP principal ([`SESSIONS_PER_PRINCIPAL`];
    /// not a Unix one, see [`quota_applies`]) or a server still at its limit refuses the new session, and the
    /// sessions already open are kept.
    pub(super) fn open(
        &self,
        conn: u64,
        principal: &Principal,
        state: Arc<DocState>,
    ) -> Result<u64, StoreError> {
        let now = Instant::now();
        let mut guard = lock_mutex(&self.open);
        self.sweep(&mut guard, now);
        let open = &mut guard.sessions;
        let mine = open.values().filter(|session| session.conn == conn).count();
        if mine >= SESSIONS_PER_CONN {
            return Err(backend(format!(
                "too many open read sessions: this connection holds {SESSIONS_PER_CONN}"
            )));
        }
        let theirs = if quota_applies(principal) {
            open.values()
                .filter(|session| session.principal == *principal)
                .count()
        } else {
            0
        };
        if theirs >= SESSIONS_PER_PRINCIPAL {
            return Err(backend(format!(
                "{SESSION_QUOTA}: its connections hold {SESSIONS_PER_PRINCIPAL} of this \
                 server's {SESSIONS_PER_SERVER} read sessions"
            )));
        }
        if open.len() >= SESSIONS_PER_SERVER {
            return Err(backend(format!(
                "too many open read sessions: this server holds {SESSIONS_PER_SERVER}"
            )));
        }
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        open.insert(
            id,
            Session {
                conn,
                principal: Arc::clone(principal),
                state,
                opened: now,
                last_used: now,
            },
        );
        Ok(id)
    }

    /// The state `session` pinned, which is used again from now. A session
    /// of another connection is unknown here. A session that expired, or
    /// that a sweep dropped, is a [`SESSION_EXPIRED`] error. Every
    /// [`SWEEP_EVERY_READS`]th read also sweeps the expired sessions.
    pub(super) fn state(&self, conn: u64, session: u64) -> Result<Arc<DocState>, StoreError> {
        let now = Instant::now();
        let mut guard = lock_mutex(&self.open);
        if self.reads.fetch_add(1, Ordering::Relaxed) % SWEEP_EVERY_READS == SWEEP_EVERY_READS - 1 {
            self.sweep(&mut guard, now);
        }
        let open = &mut *guard;
        match open.sessions.get_mut(&session) {
            Some(held) if held.conn == conn => {
                if let Some(why) = self.expiry(held, now) {
                    open.sessions.remove(&session);
                    return Err(expired(format!("session {session} was {why}")));
                }
                held.last_used = now;
                Ok(Arc::clone(&held.state))
            }
            _ if open.swept.contains(&(session, conn)) => Err(expired(format!(
                "session {session} expired and was dropped"
            ))),
            _ => Err(backend(format!(
                "no open read session {session} on this connection"
            ))),
        }
    }

    /// End `session`; its pinned state, when it was open on `conn`. Expired
    /// sessions are swept too.
    pub(super) fn close(&self, conn: u64, session: u64) -> Option<Arc<DocState>> {
        let now = Instant::now();
        let mut guard = lock_mutex(&self.open);
        let state = match guard.sessions.get(&session) {
            Some(held) if held.conn == conn => {
                guard.sessions.remove(&session).map(|held| held.state)
            }
            _ => None,
        };
        self.sweep(&mut guard, now);
        state
    }

    /// End every session of a connection that closed.
    pub(super) fn drop_conn(&self, conn: u64) {
        let mut guard = lock_mutex(&self.open);
        guard.sessions.retain(|_, session| session.conn != conn);
        guard.swept.retain(|(_, of)| *of != conn);
    }
}

// ---- exactly-once --------------------------------------------------------------

/// A request id in the table: ids are per principal, so one principal's id
/// never answers another's request.
type Key = (Principal, u128);

struct Entry {
    /// When it was claimed.
    at: Instant,
    /// The method it was claimed for: the same id for another method is
    /// refused.
    method: &'static str,
    /// Its reply, without the frame id; `None` while the first request with
    /// this id is running.
    reply: Option<Value>,
}

#[derive(Default)]
struct Table {
    entries: HashMap<Key, Entry>,
    /// Ids in arrival order, with the time each was claimed.
    order: VecDeque<(Instant, Key)>,
    /// How many entries each principal holds.
    held: HashMap<Principal, usize>,
}

impl Table {
    fn held_by(&self, principal: &str) -> usize {
        self.held.get(principal).copied().unwrap_or(0)
    }

    fn insert(&mut self, key: Key, entry: Entry) {
        *self.held.entry(Arc::clone(&key.0)).or_default() += 1;
        self.order.push_back((entry.at, key.clone()));
        self.entries.insert(key, entry);
    }

    fn remove(&mut self, key: &Key) {
        if self.entries.remove(key).is_none() {
            return;
        }
        if let Some(count) = self.held.get_mut(&*key.0) {
            *count -= 1;
            if *count == 0 {
                self.held.remove(&*key.0);
            }
        }
    }

    /// The entry an `order` item stands for, unless it was released or
    /// claimed again since.
    fn live(&self, at: Instant, key: &Key) -> Option<&Entry> {
        self.entries.get(key).filter(|entry| entry.at == at)
    }

    /// Forget answered ids older than [`DEDUP_TTL`], then, while the table
    /// holds [`DEDUP_ENTRIES`] or more, the oldest answered ids older than
    /// [`DEDUP_MIN_AGE`]. A running id is never forgotten: its reply is
    /// still to be kept. Neither is one answered so recently that a client
    /// may still resend it after a call timeout.
    fn evict(&mut self, now: Instant) {
        // Running ids at the front, set aside and put back in order.
        let mut running = Vec::new();
        while let Some((at, key)) = self.order.front().cloned() {
            let Some(entry) = self.live(at, &key) else {
                // Released, or claimed again since: nothing to forget.
                self.order.pop_front();
                continue;
            };
            if entry.reply.is_none() {
                running.extend(self.order.pop_front());
                continue;
            }
            let age = now.duration_since(at);
            let full = self.entries.len() >= DEDUP_ENTRIES;
            if age <= DEDUP_TTL && !(full && age > DEDUP_MIN_AGE) {
                // Every id behind it is younger still.
                break;
            }
            self.order.pop_front();
            self.remove(&key);
        }
        for item in running.into_iter().rev() {
            self.order.push_front(item);
        }
    }

    /// For `principal` at its quota: forget its own oldest answered ids
    /// older than [`DEDUP_MIN_AGE`] until it is below the quota, under the
    /// same rule as [`Self::evict`] for a full table.
    fn evict_own(&mut self, principal: &str, now: Instant) {
        let excess = (self.held_by(principal) + 1).saturating_sub(DEDUP_PER_PRINCIPAL);
        let mut gone = Vec::new();
        for (at, key) in &self.order {
            if gone.len() >= excess || now.duration_since(*at) <= DEDUP_MIN_AGE {
                // Every id behind it is younger still.
                break;
            }
            if *key.0 == *principal
                && self
                    .live(*at, key)
                    .is_some_and(|entry| entry.reply.is_some())
            {
                gone.push(key.clone());
            }
        }
        for key in &gone {
            self.remove(key);
        }
        // Items of forgotten ids stay in `order` until they reach its front;
        // keep it from growing past twice the table.
        if self.order.len() > 2 * DEDUP_ENTRIES {
            let entries = &self.entries;
            self.order
                .retain(|(at, key)| entries.get(key).is_some_and(|entry| entry.at == *at));
        }
    }
}

/// The server's recent request ids and their replies (spec §6.1).
#[derive(Default)]
pub(super) struct Dedup {
    table: Mutex<Table>,
    cv: Condvar,
}

/// What a request with an id does.
pub(super) enum Claim<'a> {
    /// The first with this id: run it and [`Running::finish`] with its reply.
    Run(Running<'a>),
    /// Seen before: send this reply and apply nothing.
    Replay(Value),
}

/// The claim of a request that runs. Dropped without a reply (an unwind),
/// it releases the id, so a later request with it runs.
pub(super) struct Running<'a> {
    dedup: &'a Dedup,
    key: Key,
    at: Instant,
    done: bool,
}

impl Running<'_> {
    pub(super) fn finish(mut self, reply: &Value) {
        let mut table = lock_mutex(&self.dedup.table);
        if let Some(entry) = table.entries.get_mut(&self.key)
            && entry.at == self.at
        {
            entry.reply = Some(reply.clone());
        }
        self.done = true;
        self.dedup.cv.notify_all();
    }
}

impl Drop for Running<'_> {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        let mut table = lock_mutex(&self.dedup.table);
        if table.live(self.at, &self.key).is_some() {
            table.remove(&self.key);
        }
        self.dedup.cv.notify_all();
    }
}

impl Dedup {
    /// Claim `principal`'s `req` for `method`. A request with the same id
    /// still running elsewhere is waited for, then replayed; one with the
    /// same id for another method is refused with [`DEDUP_OTHER_METHOD`]. A
    /// new id is refused, and nothing runs (the client may send it again),
    /// with [`DEDUP_FULL`] when the table holds [`DEDUP_ENTRIES`] ids it may
    /// not forget yet, and with [`DEDUP_QUOTA`] when `principal` holds
    /// [`DEDUP_PER_PRINCIPAL`] of them.
    pub(super) fn claim(
        &self,
        principal: &Principal,
        method: &'static str,
        req: u128,
    ) -> Result<Claim<'_>, StoreError> {
        self.claim_at(principal, method, req, Instant::now)
    }

    /// The reply the table keeps for `principal`'s `req`, without evicting
    /// anything.
    #[cfg(test)]
    pub(super) fn reply_of(&self, principal: &Principal, req: u128) -> Option<Value> {
        lock_mutex(&self.table)
            .entries
            .get(&(Arc::clone(principal), req))
            .and_then(|entry| entry.reply.clone())
    }

    /// [`Self::claim`] with the clock `now`, which a test sets.
    pub(super) fn claim_at(
        &self,
        principal: &Principal,
        method: &'static str,
        req: u128,
        now: impl Fn() -> Instant,
    ) -> Result<Claim<'_>, StoreError> {
        let key = (Arc::clone(principal), req);
        let mut table = lock_mutex(&self.table);
        loop {
            let now = now();
            table.evict(now);
            match table.entries.get(&key) {
                Some(entry) if entry.method != method => {
                    return Err(backend(format!(
                        "{DEDUP_OTHER_METHOD}: this credential sent request id {req:032x} \
                         for `{}`, and this request is `{method}`; nothing ran",
                        entry.method
                    )));
                }
                Some(Entry {
                    reply: Some(reply), ..
                }) => return Ok(Claim::Replay(reply.clone())),
                Some(_) => table = wait_mutex(&self.cv, table),
                None => {
                    let capped = quota_applies(principal);
                    if capped && table.held_by(principal) >= DEDUP_PER_PRINCIPAL {
                        table.evict_own(principal, now);
                    }
                    if table.entries.len() >= DEDUP_ENTRIES {
                        return Err(backend(format!(
                            "{DEDUP_FULL}: the server holds {DEDUP_ENTRIES} recent writes it \
                             cannot forget yet"
                        )));
                    }
                    if capped && table.held_by(principal) >= DEDUP_PER_PRINCIPAL {
                        return Err(backend(format!(
                            "{DEDUP_QUOTA}: it holds {DEDUP_PER_PRINCIPAL} of the server's \
                             {DEDUP_ENTRIES} recent writes, which cannot be forgotten yet"
                        )));
                    }
                    table.insert(
                        key.clone(),
                        Entry {
                            at: now,
                            method,
                            reply: None,
                        },
                    );
                    return Ok(Claim::Run(Running {
                        dedup: self,
                        key,
                        at: now,
                        done: false,
                    }));
                }
            }
        }
    }
}

/// The request id of `params`: 32 hex digits, or absent.
pub(super) fn request_id(params: &Value) -> Result<Option<u128>, StoreError> {
    match params.get("req") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) if text.len() == 32 => u128::from_str_radix(text, 16)
            .map(Some)
            .map_err(|_| backend("param `req` is not 32 hex digits")),
        Some(_) => Err(backend("param `req` is not 32 hex digits")),
    }
}

/// A new request id for one logical attempt of a write: 32 hex digits.
pub(super) fn new_request_id() -> String {
    format!("{:032x}", uuid::Uuid::now_v7().as_u128())
}
