//! One `FerroStore` serving many clients over a Unix socket or `ws://`.
//!
//! The server process is the only opener of the `.fls` file. Unix clients
//! speak length-prefixed JSON. WebSocket clients speak one JSON value per
//! frame. Neither takes `flock`.

use super::columnar;
use super::layout::{Op, backend};
use super::state::DocState;
use super::vector;
use super::{FerroStore, MqRecord, SyncBatch, lock_mutex};
use fl_core::StoreError;
use fl_core::ids::{GateId, ProjectId};
use fl_core::iri::Iri;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{self, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::os::unix::fs::{FileTypeExt, MetadataExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use super::ws;

const MAX_FRAME_BYTES: u32 = 16 * 1024 * 1024;
const CALL_TIMEOUT: Duration = Duration::from_secs(10);
/// Accept is nonblocking. Drop waits at most one pause to observe the flag.
const ACCEPT_PAUSE: Duration = Duration::from_millis(20);
/// Forwarders wake on this interval so a closed connection can join them.
const FORWARD_PAUSE: Duration = Duration::from_millis(50);

struct ServerState {
    store: Arc<FerroStore>,
    shutdown: AtomicBool,
    conns: Mutex<HashMap<u64, Box<dyn JsonIo>>>,
    next_conn: AtomicU64,
}

struct ConnReg {
    state: Arc<ServerState>,
    id: u64,
}

impl Drop for ConnReg {
    fn drop(&mut self) {
        lock_mutex(&self.state.conns).remove(&self.id);
    }
}

enum Bound {
    Unix { path: PathBuf, inode: u64 },
    Tcp { addr: SocketAddr },
}

/// Accept thread, shutdown flag, and the socket this server bound.
pub struct FerroStoreServer {
    bound: Bound,
    state: Arc<ServerState>,
    accept: Option<JoinHandle<()>>,
}

impl FerroStoreServer {
    /// Bind a Unix socket and serve `store`.
    ///
    /// Moves `store` in and does not open its path again. If `path` exists and
    /// is a socket, it is unlinked first. If it exists and is not a socket,
    /// this returns [`StoreError::Backend`].
    pub fn bind_unix(path: impl AsRef<Path>, store: FerroStore) -> Result<Self, StoreError> {
        let path = path.as_ref();
        prepare_bind_path(path)?;
        let listener = UnixListener::bind(path)
            .map_err(|err| backend(format!("bind {}: {err}", path.display())))?;
        if let Err(err) = listener.set_nonblocking(true) {
            drop(listener);
            let _ = fs::remove_file(path);
            return Err(backend(format!("set socket nonblocking: {err}")));
        }
        let inode = match fs::symlink_metadata(path) {
            Ok(meta) => meta.ino(),
            Err(err) => {
                drop(listener);
                let _ = fs::remove_file(path);
                return Err(backend(format!("stat {}: {err}", path.display())));
            }
        };
        let state = new_state(store);
        let accept_state = Arc::clone(&state);
        let path_buf = path.to_path_buf();
        let accept = match thread::Builder::new()
            .name("fls-accept".to_string())
            .spawn(move || accept_loop(move || poll_unix(&listener), accept_state))
        {
            Ok(handle) => handle,
            Err(err) => {
                unlink_bound(&path_buf, inode);
                return Err(backend(format!("start accept thread: {err}")));
            }
        };
        Ok(Self {
            bound: Bound::Unix {
                path: path_buf,
                inode,
            },
            state,
            accept: Some(accept),
        })
    }

    /// Listen on TCP, speak RFC 6455, and serve `store`. `127.0.0.1:0` must bind an ephemeral port.
    pub fn bind_tcp(addr: impl ToSocketAddrs, store: FerroStore) -> Result<Self, StoreError> {
        let listener =
            TcpListener::bind(addr).map_err(|err| backend(format!("bind tcp: {err}")))?;
        if let Err(err) = listener.set_nonblocking(true) {
            return Err(backend(format!("set socket nonblocking: {err}")));
        }
        let local = listener
            .local_addr()
            .map_err(|err| backend(format!("tcp local address: {err}")))?;
        let state = new_state(store);
        let accept_state = Arc::clone(&state);
        let accept = thread::Builder::new()
            .name("fls-accept".to_string())
            .spawn(move || accept_loop(move || poll_tcp(&listener), accept_state))
            .map_err(|err| backend(format!("start accept thread: {err}")))?;
        Ok(Self {
            bound: Bound::Tcp { addr: local },
            state,
            accept: Some(accept),
        })
    }

    /// The address actually bound (`127.0.0.1:<port>` for an ephemeral bind).
    pub fn local_addr(&self) -> SocketAddr {
        match &self.bound {
            Bound::Tcp { addr } => *addr,
            Bound::Unix { .. } => SocketAddr::from((std::net::Ipv4Addr::LOCALHOST, 0)),
        }
    }
}

fn new_state(store: FerroStore) -> Arc<ServerState> {
    Arc::new(ServerState {
        store: Arc::new(store),
        shutdown: AtomicBool::new(false),
        conns: Mutex::new(HashMap::new()),
        next_conn: AtomicU64::new(1),
    })
}

#[cfg(test)]
impl FerroStoreServer {
    pub(super) fn store(&self) -> &FerroStore {
        &self.state.store
    }
}

impl Drop for FerroStoreServer {
    fn drop(&mut self) {
        // Stop accept and connection readers before joining. Unlink only the
        // inode a unix server bound, so a replacement socket is left alone.
        // A tcp server has no filesystem path.
        self.state.shutdown.store(true, Ordering::SeqCst);
        shutdown_conns(&self.state);
        match &self.bound {
            Bound::Unix { path, .. } => {
                let _ = UnixStream::connect(path);
            }
            Bound::Tcp { addr } => {
                let _ = TcpStream::connect_timeout(addr, Duration::from_millis(200));
            }
        }
        if let Some(handle) = self.accept.take() {
            let _ = handle.join();
        }
        if let Bound::Unix { path, inode } = &self.bound {
            unlink_bound(path, *inode);
        }
    }
}

fn prepare_bind_path(path: &Path) -> Result<(), StoreError> {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_socket() => fs::remove_file(path)
            .map_err(|err| backend(format!("unlink {}: {err}", path.display()))),
        Ok(_) => Err(backend(format!(
            "{} exists and is not a socket",
            path.display()
        ))),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(backend(format!("stat {}: {err}", path.display()))),
    }
}

fn unlink_bound(path: &Path, inode: u64) {
    let Ok(meta) = fs::symlink_metadata(path) else {
        return;
    };
    if meta.ino() == inode && meta.file_type().is_socket() {
        let _ = fs::remove_file(path);
    }
}

fn shutdown_conns(state: &ServerState) {
    let conns = std::mem::take(&mut *lock_mutex(&state.conns));
    for (_, conn) in conns {
        conn.shutdown_both();
    }
}

enum AcceptPoll {
    Conn(Accepted),
    Retry,
    Wait,
    Stop,
}

struct Accepted {
    read: Box<dyn JsonIo>,
    write: Box<dyn JsonIo>,
    watch: Box<dyn JsonIo>,
}

fn accept_loop(mut poll: impl FnMut() -> AcceptPoll, state: Arc<ServerState>) {
    let mut handlers = Vec::new();
    loop {
        reap(&mut handlers);
        if state.shutdown.load(Ordering::SeqCst) {
            break;
        }
        match poll() {
            AcceptPoll::Conn(accepted) => {
                if state.shutdown.load(Ordering::SeqCst) {
                    accepted.watch.shutdown_both();
                    break;
                }
                let id = state.next_conn.fetch_add(1, Ordering::Relaxed);
                lock_mutex(&state.conns).insert(id, accepted.watch);
                let conn_state = Arc::clone(&state);
                let read = accepted.read;
                let write = Arc::new(Mutex::new(accepted.write));
                match thread::Builder::new()
                    .name("fls-conn".to_string())
                    .spawn(move || serve_connection(read, write, conn_state, id))
                {
                    Ok(handle) => handlers.push(handle),
                    Err(_) => {
                        lock_mutex(&state.conns).remove(&id);
                    }
                }
            }
            AcceptPoll::Retry => {}
            AcceptPoll::Wait => thread::sleep(ACCEPT_PAUSE),
            AcceptPoll::Stop => break,
        }
    }
    state.shutdown.store(true, Ordering::SeqCst);
    shutdown_conns(&state);
    for handle in handlers {
        let _ = handle.join();
    }
}

fn poll_unix(listener: &UnixListener) -> AcceptPoll {
    match listener.accept() {
        Ok((stream, _)) => match unix_accepted(stream) {
            Ok(accepted) => AcceptPoll::Conn(accepted),
            Err(_) => AcceptPoll::Retry,
        },
        Err(err) if err.kind() == io::ErrorKind::WouldBlock => AcceptPoll::Wait,
        Err(err) if accept_again(&err) => AcceptPoll::Retry,
        Err(_) => AcceptPoll::Stop,
    }
}

fn poll_tcp(listener: &TcpListener) -> AcceptPoll {
    match listener.accept() {
        Ok((stream, _)) => match tcp_accepted(stream) {
            Ok(accepted) => AcceptPoll::Conn(accepted),
            Err(_) => AcceptPoll::Retry,
        },
        Err(err) if err.kind() == io::ErrorKind::WouldBlock => AcceptPoll::Wait,
        Err(err) if accept_again(&err) => AcceptPoll::Retry,
        Err(_) => AcceptPoll::Stop,
    }
}

fn unix_accepted(stream: UnixStream) -> Result<Accepted, StoreError> {
    let write = stream
        .try_clone()
        .map_err(|err| backend(format!("clone unix stream: {err}")))?;
    let watch = stream
        .try_clone()
        .map_err(|err| backend(format!("clone unix stream: {err}")))?;
    Ok(Accepted {
        read: Box::new(UnixIo(stream)),
        write: Box::new(UnixIo(write)),
        watch: Box::new(UnixIo(watch)),
    })
}

fn tcp_accepted(stream: TcpStream) -> Result<Accepted, StoreError> {
    stream
        .set_nodelay(true)
        .map_err(|err| backend(format!("set nodelay: {err}")))?;
    // `O_NONBLOCK` is per file description and is shared by `try_clone`.
    // Accepted sockets must block in the frame reader.
    stream
        .set_nonblocking(false)
        .map_err(|err| backend(format!("set socket blocking: {err}")))?;
    let write = stream
        .try_clone()
        .map_err(|err| backend(format!("clone tcp stream: {err}")))?;
    let watch = stream
        .try_clone()
        .map_err(|err| backend(format!("clone tcp stream: {err}")))?;
    Ok(Accepted {
        read: Box::new(WsIo::server(stream, true)),
        write: Box::new(WsIo::server(write, false)),
        watch: Box::new(WsIo::server(watch, false)),
    })
}

fn accept_again(err: &io::Error) -> bool {
    matches!(
        err.kind(),
        io::ErrorKind::Interrupted | io::ErrorKind::ConnectionAborted
    )
}

fn reap(handlers: &mut Vec<JoinHandle<()>>) {
    let pending = std::mem::take(handlers);
    for handle in pending {
        if handle.is_finished() {
            let _ = handle.join();
        } else {
            handlers.push(handle);
        }
    }
}

fn serve_connection(
    mut read: Box<dyn JsonIo>,
    write: Arc<Mutex<Box<dyn JsonIo>>>,
    state: Arc<ServerState>,
    conn_id: u64,
) {
    let _reg = ConnReg {
        state: Arc::clone(&state),
        id: conn_id,
    };
    if read.prepare().is_err() {
        return;
    }
    let alive = Arc::new(AtomicBool::new(true));
    let mut subscribed = HashSet::new();
    let mut forwarders = Vec::new();
    loop {
        if state.shutdown.load(Ordering::SeqCst) {
            break;
        }
        let incoming = match read.read_value() {
            Ok(incoming) => incoming,
            Err(_) => break,
        };
        match incoming {
            Incoming::Ping(payload) => {
                let mut stream = lock_mutex(&write);
                if stream.write_pong(&payload).is_err() {
                    break;
                }
            }
            Incoming::Close => break,
            Incoming::Value(frame) => {
                let Some(id) = frame.get("id").and_then(Value::as_u64) else {
                    break;
                };
                let method = frame.get("method").and_then(Value::as_str).unwrap_or("");
                let params = frame.get("params").cloned().unwrap_or_else(|| json!({}));
                let result = if method == "subscribe_mq" {
                    subscribe_topic(
                        &state.store,
                        &params,
                        &write,
                        &alive,
                        &mut subscribed,
                        &mut forwarders,
                    )
                    .map(|()| Value::Null)
                } else {
                    dispatch(&state.store, method, &params)
                };
                // An unknown method is an error response. The connection stays up.
                if send_result(&write, id, result).is_err() {
                    break;
                }
            }
        }
    }
    alive.store(false, Ordering::SeqCst);
    for handle in forwarders {
        let _ = handle.join();
    }
}

fn subscribe_topic(
    store: &FerroStore,
    params: &Value,
    write: &Arc<Mutex<Box<dyn JsonIo>>>,
    alive: &Arc<AtomicBool>,
    subscribed: &mut HashSet<String>,
    forwarders: &mut Vec<JoinHandle<()>>,
) -> Result<(), StoreError> {
    let topic = param_str(params, "topic")?.to_string();
    if subscribed.contains(&topic) {
        return Ok(());
    }
    // One in-process subscription per topic. It does not replay history.
    let rx = store.subscribe_mq(&topic);
    let topic_for_thread = topic.clone();
    let write = Arc::clone(write);
    let alive = Arc::clone(alive);
    let handle = thread::Builder::new()
        .name("fls-mq".to_string())
        .spawn(move || forward_mq(rx, topic_for_thread, write, alive))
        .map_err(|err| backend(format!("start mq forwarder: {err}")))?;
    subscribed.insert(topic);
    forwarders.push(handle);
    Ok(())
}

fn forward_mq(
    rx: Receiver<MqRecord>,
    topic: String,
    write: Arc<Mutex<Box<dyn JsonIo>>>,
    alive: Arc<AtomicBool>,
) {
    loop {
        if !alive.load(Ordering::SeqCst) {
            break;
        }
        match rx.recv_timeout(FORWARD_PAUSE) {
            Ok(record) => {
                let frame = json!({
                    "method": "mq",
                    "params": {
                        "topic": topic,
                        "record": record_value(record),
                    }
                });
                let mut stream = lock_mutex(&write);
                if stream.write_value(&frame).is_err() {
                    break;
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
}

fn dispatch(store: &FerroStore, method: &str, params: &Value) -> Result<Value, StoreError> {
    match method {
        "put_doc" => {
            let collection = param_str(params, "collection")?;
            let id = param_str(params, "id")?;
            let doc = param_value(params, "doc")?;
            let silent = optional_bool(params, "silent")?;
            if optional_bool(params, "if_absent")? {
                let placed = store.put_if_absent(collection, id, &doc, silent)?;
                let mut body = serde_json::Map::new();
                body.insert("wrote".into(), Value::Bool(placed.wrote));
                body.insert("doc".into(), placed.doc);
                if let Some(seq) = placed.store_seq {
                    body.insert("store_seq".into(), json!(seq));
                }
                Ok(Value::Object(body))
            } else {
                store.put_doc_inner(collection, id, &doc, silent)?;
                Ok(Value::Null)
            }
        }
        "get_doc" => {
            // Null is a missing id. A stored value, including JSON null, is wrapped
            // so the client can tell the two apart.
            match store.get_doc(param_str(params, "collection")?, param_str(params, "id")?)? {
                Some(doc) => Ok(json!({"doc": doc})),
                None => Ok(Value::Null),
            }
        }
        "delete_doc" => {
            store.delete_doc(param_str(params, "collection")?, param_str(params, "id")?)?;
            Ok(Value::Null)
        }
        "query_by_field" => {
            let ids = store.query_by_field(
                param_str(params, "collection")?,
                param_str(params, "field")?,
                param_str(params, "value")?,
            )?;
            Ok(Value::Array(ids.into_iter().map(Value::String).collect()))
        }
        "search_bm25" => {
            let hits = store.search_bm25(
                param_str(params, "collection")?,
                param_str(params, "query")?,
                param_usize(params, "limit")?,
            )?;
            Ok(Value::Array(
                hits.into_iter()
                    .map(|(id, score)| json!([id, score]))
                    .collect(),
            ))
        }
        "publish_mq" => {
            let (seq, store_seq) = store.publish_mq_detail(
                param_str(params, "topic")?,
                param_str(params, "key")?,
                &param_value(params, "payload")?,
            )?;
            Ok(json!({"seq": seq, "store_seq": store_seq}))
        }
        "poll_mq_group" => {
            let records = store.poll_mq_group(
                param_str(params, "topic")?,
                param_str(params, "group")?,
                param_usize(params, "limit")?,
            )?;
            Ok(Value::Array(
                records.into_iter().map(record_value).collect(),
            ))
        }
        "ack_mq" => {
            store.ack_mq(
                param_str(params, "topic")?,
                param_str(params, "group")?,
                param_u64(params, "seq")?,
            )?;
            Ok(Value::Null)
        }
        "checkpoint" => {
            store.checkpoint()?;
            Ok(Value::Null)
        }
        "create_column_table" => {
            let columns = columnar::parse_columns(
                params
                    .get("columns")
                    .ok_or_else(|| backend("missing param `columns`"))?,
            )?;
            store.create_column_table(param_str(params, "name")?, &columns)?;
            Ok(Value::Null)
        }
        "append_column_rows" => {
            let rows = columnar::parse_rows(
                params
                    .get("rows")
                    .ok_or_else(|| backend("missing param `rows`"))?,
            )?;
            store.append_column_rows(param_str(params, "name")?, &rows)?;
            Ok(Value::Null)
        }
        "scan_column_table" => {
            let filters = columnar::parse_filters(
                params
                    .get("filters")
                    .ok_or_else(|| backend("missing param `filters`"))?,
            )?;
            columnar::scan_json(&store.scan_column_table(param_str(params, "name")?, &filters)?)
        }
        "attempt_rollup" => {
            let project = optional_project(params)?;
            Ok(columnar::attempt_rollup_json(
                &store.attempt_rollup(project.as_ref())?,
            ))
        }
        "gate_run_rollup" => {
            let gate = optional_gate(params)?;
            Ok(columnar::gate_rollup_json(
                &store.gate_run_rollup(gate.as_ref())?,
            ))
        }
        "create_vector_collection" => {
            store.create_vector_collection(
                param_str(params, "name")?,
                param_u32(params, "dim")?,
                vector::storage_from_name(param_str(params, "storage")?)?,
                vector::metric_from_name(param_str(params, "metric")?)?,
                param_u32(params, "m")?,
                param_u32(params, "ef_construction")?,
                param_u32(params, "ef_search")?,
            )?;
            Ok(Value::Null)
        }
        "upsert_vector" => {
            let values = vector::parse_values(
                params
                    .get("values")
                    .ok_or_else(|| backend("missing param `values`"))?,
            )?;
            let metadata = params.get("metadata");
            store.upsert_vector(
                param_str(params, "collection")?,
                param_str(params, "id")?,
                &values,
                metadata,
            )?;
            Ok(Value::Null)
        }
        "delete_vector" => {
            store.delete_vector(param_str(params, "collection")?, param_str(params, "id")?)?;
            Ok(Value::Null)
        }
        "get_vector" => {
            match store.get_vector(param_str(params, "collection")?, param_str(params, "id")?)? {
                Some((values, metadata)) => vector::vector_get_json(&values, &metadata),
                None => Ok(Value::Null),
            }
        }
        "search_vectors" => vector::hits_json(
            &store.search_vectors(
                param_str(params, "collection")?,
                &vector::parse_values(
                    params
                        .get("values")
                        .ok_or_else(|| backend("missing param `values`"))?,
                )?,
                param_usize(params, "limit")?,
                optional_u32(params, "ef")?,
            )?,
        ),
        "search_hybrid" => vector::hits_json(
            &store.search_hybrid(
                param_str(params, "doc_collection")?,
                param_str(params, "vector_collection")?,
                param_str(params, "query")?,
                &vector::parse_values(
                    params
                        .get("values")
                        .ok_or_else(|| backend("missing param `values`"))?,
                )?,
                param_usize(params, "limit")?,
                vector::fusion_from_json(
                    params
                        .get("fusion")
                        .ok_or_else(|| backend("missing param `fusion`"))?,
                )?,
            )?,
        ),
        "search_knowledge" => vector::knowledge_json(
            &store.search_knowledge(
                param_str(params, "doc_collection")?,
                param_str(params, "vector_collection")?,
                param_str(params, "query")?,
                &vector::parse_values(
                    params
                        .get("values")
                        .ok_or_else(|| backend("missing param `values`"))?,
                )?,
                param_usize(params, "limit")?,
                vector::fusion_from_json(
                    params
                        .get("fusion")
                        .ok_or_else(|| backend("missing param `fusion`"))?,
                )?,
            )?,
        ),
        "read_state" => Ok(doc_state_value(&store.read_state()?)),
        "commit_ops" => {
            let expected_seq = param_u64(params, "expected_seq")?;
            let ops = parse_ops(
                params
                    .get("ops")
                    .ok_or_else(|| backend("missing param `ops`"))?,
            )?;
            let silent = optional_bool(params, "silent")?;
            // `false` is a sequence mismatch, not an error response.
            Ok(Value::Bool(store.commit_if_seq_inner(
                expected_seq,
                &ops,
                silent,
            )?))
        }
        "apply_sync_batches" => {
            let batches = parse_sync_batches(
                params
                    .get("batches")
                    .ok_or_else(|| backend("missing param `batches`"))?,
            )?;
            let silent = optional_bool(params, "silent")?;
            Ok(Value::from(store.apply_sync_batches(batches, silent)?))
        }
        _ => Err(backend(format!("no such method `{method}`"))),
    }
}

fn send_result(
    write: &Mutex<Box<dyn JsonIo>>,
    id: u64,
    result: Result<Value, StoreError>,
) -> Result<(), StoreError> {
    let body = match result {
        Ok(result) => json!({"id": id, "ok": true, "result": result}),
        Err(err) => json!({"id": id, "ok": false, "error": err.to_string()}),
    };
    let mut stream = lock_mutex(write);
    stream.write_value(&body)
}

fn record_value(record: MqRecord) -> Value {
    json!({
        "seq": record.seq,
        "key": record.key,
        "payload": record.payload,
    })
}

fn param_str<'a>(params: &'a Value, key: &str) -> Result<&'a str, StoreError> {
    params
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| backend(format!("missing string param `{key}`")))
}

fn param_u64(params: &Value, key: &str) -> Result<u64, StoreError> {
    params
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| backend(format!("missing integer param `{key}`")))
}

fn param_usize(params: &Value, key: &str) -> Result<usize, StoreError> {
    usize::try_from(param_u64(params, key)?)
        .map_err(|_| backend(format!("param `{key}` does not fit")))
}

fn param_u32(params: &Value, key: &str) -> Result<u32, StoreError> {
    u32::try_from(param_u64(params, key)?)
        .map_err(|_| backend(format!("param `{key}` does not fit")))
}

fn optional_bool(params: &Value, key: &str) -> Result<bool, StoreError> {
    match params.get(key) {
        None => Ok(false),
        Some(Value::Bool(value)) => Ok(*value),
        Some(_) => Err(backend(format!("param `{key}` is not a bool"))),
    }
}

fn parse_sync_batches(value: &Value) -> Result<Vec<SyncBatch>, StoreError> {
    serde_json::from_value(value.clone()).map_err(|err| backend(format!("sync batches: {err}")))
}

fn optional_u32(params: &Value, key: &str) -> Result<Option<u32>, StoreError> {
    match params.get(key) {
        None => Ok(None),
        Some(value) => {
            let number = value
                .as_u64()
                .ok_or_else(|| backend(format!("missing integer param `{key}`")))?;
            let number = u32::try_from(number)
                .map_err(|_| backend(format!("param `{key}` does not fit")))?;
            Ok(Some(number))
        }
    }
}

fn param_value(params: &Value, key: &str) -> Result<Value, StoreError> {
    params
        .get(key)
        .cloned()
        .ok_or_else(|| backend(format!("missing param `{key}`")))
}

fn doc_state_value(state: &DocState) -> Value {
    let docs = state
        .docs
        .iter()
        .map(|((collection, id), doc)| {
            json!({
                "collection": collection,
                "id": id,
                "doc": doc,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "last_seq": state.last_seq,
        "docs": docs,
        "tables": columnar::tables_json(&state.tables),
        "vectors": vector::vectors_json(&state.vectors),
    })
}

fn parse_doc_state(value: &Value) -> Result<DocState, StoreError> {
    let last_seq = value
        .get("last_seq")
        .and_then(Value::as_u64)
        .ok_or_else(|| backend("read_state result is missing last_seq"))?;
    let items = value
        .get("docs")
        .and_then(Value::as_array)
        .ok_or_else(|| backend("read_state result is missing docs"))?;
    let mut docs = BTreeMap::new();
    for item in items {
        let collection = param_str(item, "collection")?.to_string();
        let id = param_str(item, "id")?.to_string();
        let doc = item
            .get("doc")
            .cloned()
            .ok_or_else(|| backend("read_state doc is missing doc"))?;
        docs.insert((collection, id), doc);
    }
    let tables = match value.get("tables") {
        None => BTreeMap::new(),
        Some(tables) => columnar::parse_tables(tables)?,
    };
    let vectors = match value.get("vectors") {
        None => BTreeMap::new(),
        Some(vectors) => vector::parse_vectors(vectors)?,
    };
    Ok(DocState {
        docs,
        tables,
        vectors,
        last_seq,
    })
}

fn optional_project(params: &Value) -> Result<Option<ProjectId>, StoreError> {
    match params.get("project") {
        Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => {
            let iri = Iri::parse(text).map_err(backend)?;
            Ok(Some(ProjectId(iri)))
        }
        _ => Err(backend("missing project param")),
    }
}

fn optional_gate(params: &Value) -> Result<Option<GateId>, StoreError> {
    match params.get("gate") {
        Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => {
            let iri = Iri::parse(text).map_err(backend)?;
            Ok(Some(GateId(iri)))
        }
        _ => Err(backend("missing gate param")),
    }
}

fn op_value(op: &Op) -> Value {
    match op {
        Op::Put {
            collection,
            id,
            doc,
        } => json!({
            "op": "put",
            "collection": collection,
            "id": id,
            "doc": doc,
        }),
        Op::Delete { collection, id } => json!({
            "op": "delete",
            "collection": collection,
            "id": id,
        }),
        Op::CreateColumnTable { name, columns } => json!({
            "op": "create_column_table",
            "name": name,
            "columns": columns.iter().map(columnar::column_def_json).collect::<Vec<_>>(),
        }),
        Op::AppendColumnGroup { name, bytes } => json!({
            "op": "append_column_group",
            "name": name,
            "bytes": bytes.iter().copied().map(Value::from).collect::<Vec<_>>(),
        }),
        Op::CreateVectorCollection {
            name,
            dim,
            storage,
            metric,
            m,
            ef_construction,
            ef_search,
        } => json!({
            "op": "create_vector_collection",
            "name": name,
            "dim": dim,
            "storage": vector::storage_name(*storage),
            "metric": vector::metric_name(*metric),
            "m": m,
            "ef_construction": ef_construction,
            "ef_search": ef_search,
        }),
        Op::UpsertVector {
            collection,
            id,
            bytes,
            metadata,
        } => {
            let mut item = serde_json::Map::new();
            item.insert("op".into(), json!("upsert_vector"));
            item.insert("collection".into(), json!(collection));
            item.insert("id".into(), json!(id));
            item.insert(
                "bytes".into(),
                Value::Array(bytes.iter().copied().map(Value::from).collect()),
            );
            if let Some(metadata) = metadata {
                item.insert("metadata".into(), metadata.clone());
            }
            Value::Object(item)
        }
        Op::DeleteVector { collection, id } => json!({
            "op": "delete_vector",
            "collection": collection,
            "id": id,
        }),
    }
}

fn parse_ops(value: &Value) -> Result<Vec<Op>, StoreError> {
    let items = value
        .as_array()
        .ok_or_else(|| backend("param `ops` is not an array"))?;
    items.iter().map(parse_op).collect()
}

fn parse_op(value: &Value) -> Result<Op, StoreError> {
    let kind = value
        .get("op")
        .and_then(Value::as_str)
        .ok_or_else(|| backend("op is missing op"))?;
    match kind {
        "put" => {
            let doc = value
                .get("doc")
                .cloned()
                .ok_or_else(|| backend("put op is missing doc"))?;
            Ok(Op::Put {
                collection: param_str(value, "collection")?.to_string(),
                id: param_str(value, "id")?.to_string(),
                doc,
            })
        }
        "delete" => Ok(Op::Delete {
            collection: param_str(value, "collection")?.to_string(),
            id: param_str(value, "id")?.to_string(),
        }),
        "create_column_table" => {
            let columns = columnar::parse_columns(
                value
                    .get("columns")
                    .ok_or_else(|| backend("create_column_table op is missing columns"))?,
            )?;
            Ok(Op::CreateColumnTable {
                name: param_str(value, "name")?.to_string(),
                columns,
            })
        }
        "append_column_group" => Ok(Op::AppendColumnGroup {
            name: param_str(value, "name")?.to_string(),
            bytes: columnar::parse_bytes(
                value
                    .get("bytes")
                    .ok_or_else(|| backend("append_column_group op is missing bytes"))?,
            )?,
        }),
        "create_vector_collection" => Ok(Op::CreateVectorCollection {
            name: param_str(value, "name")?.to_string(),
            dim: param_u32(value, "dim")?,
            storage: vector::storage_from_name(param_str(value, "storage")?)?,
            metric: vector::metric_from_name(param_str(value, "metric")?)?,
            m: param_u32(value, "m")?,
            ef_construction: param_u32(value, "ef_construction")?,
            ef_search: param_u32(value, "ef_search")?,
        }),
        "upsert_vector" => {
            let metadata = value.get("metadata").cloned();
            Ok(Op::UpsertVector {
                collection: param_str(value, "collection")?.to_string(),
                id: param_str(value, "id")?.to_string(),
                bytes: columnar::parse_bytes(
                    value
                        .get("bytes")
                        .ok_or_else(|| backend("upsert_vector op is missing bytes"))?,
                )?,
                metadata,
            })
        }
        "delete_vector" => Ok(Op::DeleteVector {
            collection: param_str(value, "collection")?.to_string(),
            id: param_str(value, "id")?.to_string(),
        }),
        other => Err(backend(format!("unknown op `{other}`"))),
    }
}

enum Incoming {
    Value(Value),
    Ping(Vec<u8>),
    Close,
}

trait JsonIo: Send {
    fn prepare(&mut self) -> Result<(), StoreError> {
        Ok(())
    }

    fn read_value(&mut self) -> Result<Incoming, StoreError>;

    fn write_value(&mut self, value: &Value) -> Result<(), StoreError>;

    fn write_pong(&mut self, _payload: &[u8]) -> Result<(), StoreError> {
        Ok(())
    }

    fn shutdown_both(&self);
}

struct UnixIo(UnixStream);

impl JsonIo for UnixIo {
    fn read_value(&mut self) -> Result<Incoming, StoreError> {
        Ok(Incoming::Value(read_frame(&mut self.0)?))
    }

    fn write_value(&mut self, value: &Value) -> Result<(), StoreError> {
        write_frame(&mut self.0, value)
    }

    fn shutdown_both(&self) {
        let _ = self.0.shutdown(Shutdown::Both);
    }
}

struct WsIo {
    stream: TcpStream,
    client: bool,
    handshake: bool,
}

impl WsIo {
    fn client(stream: TcpStream) -> Self {
        Self {
            stream,
            client: true,
            handshake: false,
        }
    }

    fn server(stream: TcpStream, handshake: bool) -> Self {
        Self {
            stream,
            client: false,
            handshake,
        }
    }
}

impl JsonIo for WsIo {
    fn prepare(&mut self) -> Result<(), StoreError> {
        if !self.handshake {
            return Ok(());
        }
        self.handshake = false;
        ws::accept(&mut self.stream)
    }

    fn read_value(&mut self) -> Result<Incoming, StoreError> {
        match ws::read_event(&mut self.stream, self.client)? {
            ws::WsEvent::Data(payload) => {
                if payload.is_empty() {
                    return Err(backend("frame length is outside 1..=16MiB"));
                }
                let text =
                    std::str::from_utf8(&payload).map_err(|_| backend("frame is not utf-8"))?;
                let value = serde_json::from_str(text).map_err(|_| backend("frame is not json"))?;
                Ok(Incoming::Value(value))
            }
            ws::WsEvent::Ping(payload) => Ok(Incoming::Ping(payload)),
            ws::WsEvent::Close => Ok(Incoming::Close),
        }
    }

    fn write_value(&mut self, value: &Value) -> Result<(), StoreError> {
        let bytes =
            serde_json::to_vec(value).map_err(|err| backend(format!("encode frame: {err}")))?;
        if bytes.is_empty() || bytes.len() > ws::MAX_PAYLOAD {
            return Err(backend("frame length is outside 1..=16MiB"));
        }
        ws::write_frame(&mut self.stream, self.client, 0x1, &bytes)
    }

    fn write_pong(&mut self, payload: &[u8]) -> Result<(), StoreError> {
        ws::write_frame(&mut self.stream, self.client, 0xA, payload)
    }

    fn shutdown_both(&self) {
        let _ = self.stream.shutdown(Shutdown::Both);
    }
}

fn write_frame(stream: &mut UnixStream, value: &Value) -> Result<(), StoreError> {
    let bytes = serde_json::to_vec(value).map_err(|err| backend(format!("encode frame: {err}")))?;
    let len = u32::try_from(bytes.len()).map_err(|_| backend("frame length exceeds u32"))?;
    if len == 0 || len > MAX_FRAME_BYTES {
        return Err(backend("frame length is outside 1..=16MiB"));
    }
    stream.write_all(&len.to_be_bytes()).map_err(backend)?;
    stream.write_all(&bytes).map_err(backend)?;
    Ok(())
}

fn read_frame(stream: &mut UnixStream) -> Result<Value, StoreError> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).map_err(backend)?;
    let len = u32::from_be_bytes(len_buf);
    if len == 0 || len > MAX_FRAME_BYTES {
        return Err(backend("frame length is outside 1..=16MiB"));
    }
    let mut buf = vec![0u8; len as usize];
    stream.read_exact(&mut buf).map_err(backend)?;
    let text = std::str::from_utf8(&buf).map_err(|_| backend("frame is not utf-8"))?;
    serde_json::from_str(text).map_err(|_| backend("frame is not json"))
}

struct RemoteShared {
    write: Mutex<Box<dyn JsonIo>>,
    pending: Mutex<HashMap<u64, Sender<Result<Value, String>>>>,
    next_id: AtomicU64,
    subs: Mutex<HashMap<String, Vec<Arc<Sender<MqRecord>>>>>,
}

pub(super) struct RemoteClient {
    shared: Arc<RemoteShared>,
    reader: Mutex<Option<JoinHandle<()>>>,
}

impl RemoteClient {
    pub(super) fn call(&self, method: &str, params: Value) -> Result<Value, StoreError> {
        let id = self.shared.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = mpsc::channel();
        lock_mutex(&self.shared.pending).insert(id, tx);
        let request = json!({
            "id": id,
            "method": method,
            "params": params,
        });
        // Release the write lock before waiting. The reader must run, and a
        // response must not need this task to still hold the mutex.
        {
            let mut stream = lock_mutex(&self.shared.write);
            if let Err(err) = stream.write_value(&request) {
                lock_mutex(&self.shared.pending).remove(&id);
                return Err(err);
            }
        }
        match rx.recv_timeout(CALL_TIMEOUT) {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(message)) => Err(backend(message)),
            Err(RecvTimeoutError::Timeout) => {
                lock_mutex(&self.shared.pending).remove(&id);
                Err(backend("timed out waiting for the FerroStore server"))
            }
            Err(RecvTimeoutError::Disconnected) => Err(backend("FerroStore connection closed")),
        }
    }

    pub(super) fn call_unit(&self, method: &str, params: Value) -> Result<(), StoreError> {
        let _ = self.call(method, params)?;
        Ok(())
    }

    /// JSON null is a missing document. A present document, including JSON null,
    /// is `{"doc": value}`.
    pub(super) fn call_option(
        &self,
        method: &str,
        params: Value,
    ) -> Result<Option<Value>, StoreError> {
        let value = self.call(method, params)?;
        if value.is_null() {
            return Ok(None);
        }
        let doc = value
            .get("doc")
            .cloned()
            .ok_or_else(|| backend("result is neither null nor a document"))?;
        Ok(Some(doc))
    }

    pub(super) fn call_strings(
        &self,
        method: &str,
        params: Value,
    ) -> Result<Vec<String>, StoreError> {
        let value = self.call(method, params)?;
        let items = value
            .as_array()
            .ok_or_else(|| backend("result is not an array"))?;
        items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(ToString::to_string)
                    .ok_or_else(|| backend("result id is not a string"))
            })
            .collect()
    }

    pub(super) fn call_pairs(
        &self,
        method: &str,
        params: Value,
    ) -> Result<Vec<(String, f64)>, StoreError> {
        let value = self.call(method, params)?;
        let items = value
            .as_array()
            .ok_or_else(|| backend("result is not an array"))?;
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            let pair = item
                .as_array()
                .ok_or_else(|| backend("search hit is not a pair"))?;
            if pair.len() != 2 {
                return Err(backend("search hit is not a pair"));
            }
            let id = pair[0]
                .as_str()
                .ok_or_else(|| backend("search hit id is not a string"))?
                .to_string();
            let score = pair[1]
                .as_f64()
                .ok_or_else(|| backend("search hit score is not a number"))?;
            out.push((id, score));
        }
        Ok(out)
    }

    pub(super) fn call_records(
        &self,
        method: &str,
        params: Value,
    ) -> Result<Vec<MqRecord>, StoreError> {
        let value = self.call(method, params)?;
        let items = value
            .as_array()
            .ok_or_else(|| backend("result is not an array"))?;
        items.iter().map(parse_record).collect()
    }

    pub(super) fn read_state(&self) -> Result<DocState, StoreError> {
        parse_doc_state(&self.call("read_state", json!({}))?)
    }

    pub(super) fn commit_ops(
        &self,
        expected_seq: u64,
        ops: &[Op],
        silent: bool,
    ) -> Result<bool, StoreError> {
        let mut params = serde_json::Map::new();
        params.insert("expected_seq".into(), json!(expected_seq));
        params.insert(
            "ops".into(),
            Value::Array(ops.iter().map(op_value).collect()),
        );
        if silent {
            params.insert("silent".into(), Value::Bool(true));
        }
        let value = self.call("commit_ops", Value::Object(params))?;
        value
            .as_bool()
            .ok_or_else(|| backend("commit_ops result is not a bool"))
    }

    pub(super) fn apply_sync_batches(
        &self,
        batches: &[SyncBatch],
        silent: bool,
    ) -> Result<u64, StoreError> {
        let encoded = serde_json::to_value(batches)
            .map_err(|err| backend(format!("encode sync batches: {err}")))?;
        let mut params = serde_json::Map::new();
        params.insert("batches".into(), encoded);
        if silent {
            params.insert("silent".into(), Value::Bool(true));
        }
        let value = self.call("apply_sync_batches", Value::Object(params))?;
        value
            .as_u64()
            .ok_or_else(|| backend("apply_sync_batches result is not a u64"))
    }

    pub(super) fn subscribe_topic(&self, topic: &str) -> Receiver<MqRecord> {
        let (tx, rx) = mpsc::channel();
        let tx = Arc::new(tx);
        lock_mutex(&self.shared.subs)
            .entry(topic.to_string())
            .or_default()
            .push(Arc::clone(&tx));
        // Registered first so a push that arrives with the response is
        // delivered. A failed RPC drops this sender, so recv is Disconnected.
        if self.call("subscribe_mq", json!({"topic": topic})).is_err() {
            remove_subscriber(&self.shared, topic, &tx);
        }
        rx
    }
}

fn remove_subscriber(shared: &RemoteShared, topic: &str, tx: &Arc<Sender<MqRecord>>) {
    let mut subs = lock_mutex(&shared.subs);
    let Some(senders) = subs.get_mut(topic) else {
        return;
    };
    senders.retain(|sender| !Arc::ptr_eq(sender, tx));
    if senders.is_empty() {
        subs.remove(topic);
    }
}

impl Drop for RemoteClient {
    fn drop(&mut self) {
        let stream = lock_mutex(&self.shared.write);
        stream.shutdown_both();
        drop(stream);
        if let Some(handle) = lock_mutex(&self.reader).take() {
            let _ = handle.join();
        }
    }
}

pub(super) fn connect_unix(path: &str) -> Result<RemoteClient, StoreError> {
    let stream =
        UnixStream::connect(path).map_err(|err| backend(format!("connect {path}: {err}")))?;
    client_from_unix(stream)
}

pub(super) fn client_from_unix(stream: UnixStream) -> Result<RemoteClient, StoreError> {
    let writer = stream
        .try_clone()
        .map_err(|err| backend(format!("clone unix stream: {err}")))?;
    finish_client(Box::new(UnixIo(stream)), Box::new(UnixIo(writer)))
}

pub(super) fn connect_ws(rest: &str) -> Result<RemoteClient, StoreError> {
    let stream = ws::connect(rest)?;
    let writer = stream
        .try_clone()
        .map_err(|err| backend(format!("clone tcp stream: {err}")))?;
    finish_client(
        Box::new(WsIo::client(stream)),
        Box::new(WsIo::client(writer)),
    )
}

fn finish_client(
    read: Box<dyn JsonIo>,
    write: Box<dyn JsonIo>,
) -> Result<RemoteClient, StoreError> {
    let shared = Arc::new(RemoteShared {
        write: Mutex::new(write),
        pending: Mutex::new(HashMap::new()),
        next_id: AtomicU64::new(1),
        subs: Mutex::new(HashMap::new()),
    });
    let reader_shared = Arc::clone(&shared);
    let reader = thread::Builder::new()
        .name("fls-client".to_string())
        .spawn(move || reader_loop(read, reader_shared))
        .map_err(|err| backend(format!("start client reader: {err}")))?;
    Ok(RemoteClient {
        shared,
        reader: Mutex::new(Some(reader)),
    })
}

fn reader_loop(mut stream: Box<dyn JsonIo>, shared: Arc<RemoteShared>) {
    loop {
        match stream.read_value() {
            Ok(Incoming::Value(frame)) => {
                if frame.get("id").is_none()
                    && frame.get("method").and_then(Value::as_str) == Some("mq")
                {
                    deliver_push(&shared, &frame);
                    continue;
                }
                let Some(id) = frame.get("id").and_then(Value::as_u64) else {
                    continue;
                };
                let outcome = if frame.get("ok").and_then(Value::as_bool) == Some(true) {
                    Ok(frame.get("result").cloned().unwrap_or(Value::Null))
                } else {
                    let message = frame
                        .get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("FerroStore request failed");
                    Err(message.to_string())
                };
                if let Some(sender) = lock_mutex(&shared.pending).remove(&id) {
                    let _ = sender.send(outcome);
                }
            }
            Ok(Incoming::Ping(payload)) => {
                let mut write = lock_mutex(&shared.write);
                if write.write_pong(&payload).is_err() {
                    break;
                }
            }
            Ok(Incoming::Close) | Err(_) => break,
        }
    }
    fail_pending(&shared, "FerroStore connection closed");
    // Drop live subscription senders so an already-open recv sees Disconnected.
    let _ = std::mem::take(&mut *lock_mutex(&shared.subs));
}

fn deliver_push(shared: &RemoteShared, frame: &Value) {
    let Some(params) = frame.get("params") else {
        return;
    };
    let Some(topic) = params.get("topic").and_then(Value::as_str) else {
        return;
    };
    let Some(record_value) = params.get("record") else {
        return;
    };
    let Ok(record) = parse_record(record_value) else {
        return;
    };
    let mut subs = lock_mutex(&shared.subs);
    let Some(senders) = subs.get_mut(topic) else {
        return;
    };
    senders.retain(|sender| sender.send(record.clone()).is_ok());
    if senders.is_empty() {
        subs.remove(topic);
    }
}

#[cfg(test)]
mod empty_frame {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn empty_websocket_data_frame_is_outside_the_json_limit_and_empty_ping_is_a_ping() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        let (server, _) = listener.accept().unwrap();
        let mut io = WsIo::server(server, false);

        ws::write_frame(&mut client, true, 0x1, &[]).unwrap();
        let msg = match io.read_value() {
            Err(err) => err.to_string(),
            Ok(Incoming::Ping(_)) => panic!("empty data frame was a ping"),
            Ok(Incoming::Value(_)) => panic!("empty data frame was accepted"),
            Ok(Incoming::Close) => panic!("empty data frame was a close"),
        };
        assert!(msg.contains("frame length is outside 1..=16MiB"), "{msg}");
        assert!(!msg.contains("frame is not json"), "{msg}");

        ws::write_frame(&mut client, true, 0xA, &[]).unwrap();
        ws::write_frame(&mut client, true, 0x9, &[]).unwrap();
        match io.read_value().unwrap() {
            Incoming::Ping(payload) => assert!(payload.is_empty()),
            Incoming::Value(_) => panic!("empty ping was parsed as json"),
            Incoming::Close => panic!("empty ping was a close"),
        }
    }
}

fn parse_record(value: &Value) -> Result<MqRecord, StoreError> {
    let seq = value
        .get("seq")
        .and_then(Value::as_u64)
        .ok_or_else(|| backend("mq record is missing seq"))?;
    let key = value
        .get("key")
        .and_then(Value::as_str)
        .ok_or_else(|| backend("mq record is missing key"))?
        .to_string();
    let payload = value.get("payload").cloned().unwrap_or(Value::Null);
    Ok(MqRecord { seq, key, payload })
}

fn fail_pending(shared: &RemoteShared, message: &str) {
    let pending = std::mem::take(&mut *lock_mutex(&shared.pending));
    for (_, sender) in pending {
        let _ = sender.send(Err(message.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_method_errors_and_the_connection_stays_up() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("store.sock");
        let server = FerroStoreServer::bind_unix(&sock, FerroStore::in_memory()).unwrap();
        let client = connect_unix(&sock.display().to_string()).unwrap();
        let err = client.call("nope", json!({})).unwrap_err();
        assert!(
            matches!(err, StoreError::Backend(ref message) if message.contains("nope")),
            "{err}"
        );
        client
            .call(
                "put_doc",
                json!({"collection": "docs", "id": "a", "doc": {"n": 1}}),
            )
            .unwrap();
        assert_eq!(
            client
                .call("get_doc", json!({"collection": "docs", "id": "a"}))
                .unwrap(),
            json!({"doc": {"n": 1}})
        );
        drop(client);
        drop(server);
    }

    #[test]
    fn read_state_and_ops_round_trip_column_tables_and_a_null_document() {
        use super::columnar::{self, ColumnDef, ColumnTable, ColumnType, ColumnValue};
        let columns = vec![ColumnDef {
            name: "n".into(),
            ty: ColumnType::Int64,
        }];
        let blob = columnar::encode_row_group(
            &columns,
            &[
                vec![ColumnValue::Int64(3)],
                vec![ColumnValue::Null],
                vec![ColumnValue::Int64(4)],
            ],
        )
        .unwrap();
        let mut tables = BTreeMap::new();
        tables.insert(
            "t".into(),
            ColumnTable {
                columns: columns.clone(),
                groups: vec![blob.clone()],
            },
        );
        let mut docs = BTreeMap::new();
        docs.insert(("c".into(), "id".into()), Value::Null);
        let state = DocState {
            docs,
            tables,
            vectors: BTreeMap::new(),
            last_seq: 4,
        };
        let parsed = parse_doc_state(&doc_state_value(&state)).unwrap();
        assert_eq!(parsed.last_seq, 4);
        assert_eq!(
            parsed.docs.get(&("c".into(), "id".into())),
            Some(&Value::Null)
        );
        assert_eq!(parsed.tables.get("t").unwrap().groups, vec![blob.clone()]);
        assert_eq!(parsed.tables.get("t").unwrap().columns, columns);

        let ops = [
            Op::CreateColumnTable {
                name: "t".into(),
                columns: columns.clone(),
            },
            Op::AppendColumnGroup {
                name: "t".into(),
                bytes: blob,
            },
            Op::Put {
                collection: "c".into(),
                id: "id".into(),
                doc: Value::Null,
            },
            Op::Delete {
                collection: "c".into(),
                id: "gone".into(),
            },
        ];
        let encoded = Value::Array(ops.iter().map(op_value).collect());
        let again = parse_ops(&encoded).unwrap();
        match &again[..] {
            [
                Op::CreateColumnTable {
                    name,
                    columns: cols,
                },
                Op::AppendColumnGroup {
                    name: appended,
                    bytes,
                },
                Op::Put { doc, .. },
                Op::Delete { id, .. },
            ] => {
                assert_eq!(name, "t");
                assert_eq!(cols, &columns);
                assert_eq!(appended, "t");
                assert_eq!(bytes, &state.tables.get("t").unwrap().groups[0]);
                assert_eq!(doc, &Value::Null);
                assert_eq!(id, "gone");
            }
            other => panic!("ops did not round-trip: {other:?}"),
        }
    }

    #[test]
    fn vector_ops_round_trip_absent_and_null_metadata() {
        let ops = [
            Op::CreateVectorCollection {
                name: "emb".into(),
                dim: 2,
                storage: vector::VectorStorage::F16,
                metric: vector::VectorMetric::L2,
                m: 4,
                ef_construction: 8,
                ef_search: 9,
            },
            Op::UpsertVector {
                collection: "emb".into(),
                id: "a".into(),
                bytes: vec![0, 60],
                metadata: None,
            },
            Op::UpsertVector {
                collection: "emb".into(),
                id: "b".into(),
                bytes: vec![1, 2],
                metadata: Some(Value::Null),
            },
            Op::DeleteVector {
                collection: "emb".into(),
                id: "a".into(),
            },
        ];
        let encoded = Value::Array(ops.iter().map(op_value).collect());
        let again = parse_ops(&encoded).unwrap();
        match &again[..] {
            [
                Op::CreateVectorCollection {
                    name,
                    dim,
                    storage,
                    metric,
                    m,
                    ef_construction,
                    ef_search,
                },
                Op::UpsertVector {
                    id,
                    bytes: absent_bytes,
                    metadata: absent,
                    ..
                },
                Op::UpsertVector {
                    id: present,
                    metadata: present_meta,
                    ..
                },
                Op::DeleteVector { id: deleted, .. },
            ] => {
                assert_eq!(name, "emb");
                assert_eq!(*dim, 2);
                assert_eq!(*storage, vector::VectorStorage::F16);
                assert_eq!(*metric, vector::VectorMetric::L2);
                assert_eq!(*m, 4);
                assert_eq!(*ef_construction, 8);
                assert_eq!(*ef_search, 9);
                assert_eq!(id, "a");
                assert!(absent.is_none());
                assert_eq!(absent_bytes, &[0, 60]);
                assert_eq!(present, "b");
                assert_eq!(present_meta, &Some(Value::Null));
                assert_eq!(deleted, "a");
            }
            other => panic!("vector ops did not round-trip: {other:?}"),
        }
        let absent_meta = encoded
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["id"] == "a")
            .unwrap();
        assert!(absent_meta.get("metadata").is_none());
        let present_meta = encoded
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["id"] == "b")
            .unwrap();
        assert!(present_meta.get("metadata").unwrap().is_null());
    }
}
