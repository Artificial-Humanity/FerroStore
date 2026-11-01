//! Document store and FerroMQ log in a `.fls` file, and a socket server.
//! See `docs/ferrostore-spec.md`.

mod columnar;
mod doc;
mod layout;
mod migrate;
mod server;
mod state;
mod sync;
mod traits_catalog;
mod traits_state;
mod traits_wire;
mod traits_workflow;
mod vector;
mod ws;

#[cfg(test)]
mod tests;

use fl_core::StoreError;
use fl_core::ids::{GateId, ProjectId};
pub use layout::Op;
use layout::{
    HEADER_BYTES, SLOT_LEN, Superblock, backend, corrupt, decode_snapshot, decode_wal,
    encode_frame, encode_snapshot, encode_snapshot_v2, encode_snapshot_v3, encode_superblock,
    parse_superblock,
};
use serde_json::{Value, json};
use state::{
    DocState, MQ_HIGH_WATER_ID, apply_commit, apply_frames, commit_ops, mq_hw_collection,
    mq_message_collection, next_mq_seq, stage_ops, unread_mq,
};

pub use columnar::{
    AttemptRollup, ColumnDef, ColumnFilter, ColumnPredicate, ColumnScan, ColumnType, ColumnValue,
    GateRunRollup, ZoneMap,
};
pub use migrate::migrate_redb_to_ferrostore;
pub use server::FerroStoreServer;
pub use state::MqRecord;
use std::collections::{BTreeMap, HashMap};
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::ThreadId;
pub use sync::{EnvelopeKind, RecordEnvelope, SyncAdapter, SyncBatch, SyncMode, SyncOp};
pub use vector::{Fusion, KnowledgeHit, VectorHit, VectorMetric, VectorStorage};

const LOCK_EX: i32 = 2;
const LOCK_UN: i32 = 8;

// libc `flock`. No extra crate: the process already links libc.
unsafe extern "C" {
    fn flock(fd: i32, op: i32) -> i32;
}

struct FlockGuard {
    fd: i32,
}

impl FlockGuard {
    fn acquire(fd: i32) -> Result<Self, StoreError> {
        // SAFETY: `fd` is an open file descriptor owned by the caller. `LOCK_EX`
        // waits until this description holds the advisory lock, or returns -1.
        let rc = unsafe { flock(fd, LOCK_EX) };
        if rc != 0 {
            let err = io::Error::last_os_error();
            return Err(backend(format!("exclusive flock failed: {err}")));
        }
        Ok(Self { fd })
    }
}

impl Drop for FlockGuard {
    fn drop(&mut self) {
        // SAFETY: the descriptor stays open until after this guard is dropped.
        // `LOCK_UN` releases the advisory lock held by that description.
        unsafe {
            flock(self.fd, LOCK_UN);
        }
    }
}

enum Inner {
    Memory(Mutex<DocState>),
    Disk(Disk),
    Remote(server::RemoteClient),
}

#[derive(Default)]
struct SyncSlot {
    mode: Option<SyncMode>,
    adapter: Option<Arc<dyn SyncAdapter>>,
    synced_seq: u64,
    pending: Vec<SyncBatch>,
    /// `on_commit` is running. Another thread's `sync_flush` waits until this is clear.
    notify_in_flight: bool,
    /// Thread that set `notify_in_flight`. Cleared with the flag.
    notify_owner: Option<ThreadId>,
}

impl SyncSlot {
    /// No queued batch and no `on_commit` running. A batch that a notify owns is
    /// in neither `pending` nor `synced_seq` until the notify returns, so a check
    /// of `pending` alone would call that window idle.
    fn is_idle(&self) -> bool {
        self.pending.is_empty() && !self.notify_in_flight
    }
}

struct Committed {
    seq: u64,
    ops: Vec<Op>,
}

enum SeqCommit {
    Lost,
    Same,
    Wrote(Committed),
}

struct Disk {
    write: Mutex<File>,
    /// Separate `O_RDONLY` description. Reads never flock it.
    read: File,
}

/// One transaction: a clone of the document map plus the ops that produced it.
/// `get` / `put` / `delete` / `scan` all see earlier writes in this transaction.
struct Txn {
    docs: BTreeMap<(String, String), Value>,
    tables: BTreeMap<String, columnar::ColumnTable>,
    vectors: BTreeMap<String, vector::VectorCollection>,
    ops: Vec<Op>,
}

impl Txn {
    fn get(&self, collection: &str, id: &str) -> Option<Value> {
        self.docs
            .get(&(collection.to_string(), id.to_string()))
            .cloned()
    }

    fn put(&mut self, collection: impl Into<String>, id: impl Into<String>, doc: Value) {
        let collection = collection.into();
        let id = id.into();
        self.docs
            .insert((collection.clone(), id.clone()), doc.clone());
        self.ops.push(Op::Put {
            collection,
            id,
            doc,
        });
    }

    fn delete(&mut self, collection: impl Into<String>, id: impl Into<String>) {
        let collection = collection.into();
        let id = id.into();
        self.docs.remove(&(collection.clone(), id.clone()));
        self.ops.push(Op::Delete { collection, id });
    }

    /// Every document in `collection`, in document-id order.
    fn scan(&self, collection: &str) -> Vec<(String, Value)> {
        let start = (collection.to_string(), String::new());
        let mut out = Vec::new();
        for ((coll, id), doc) in self.docs.range(start..) {
            if coll != collection {
                break;
            }
            out.push((id.clone(), doc.clone()));
        }
        out
    }
}

/// Document store and FerroMQ log.
///
/// `mem://` keeps the map in this process. `file://` is a `.fls` container:
/// two superblock slots, a copy-on-write snapshot, and a CRC32-framed WAL.
/// `unix://` and `ws://` are clients of a [`FerroStoreServer`] and do not open the file.
/// [`Self::auto`] is that client when the default socket is listening, and [`Self::open`] otherwise.
pub struct FerroStore {
    inner: Inner,
    subs: Mutex<HashMap<String, Vec<Sender<MqRecord>>>>,
    /// Named in `NotOwned`: where a lookup looked.
    label: String,
    /// Process memory only. Not part of the `.fls` file.
    sync: Mutex<SyncSlot>,
    /// Taken before `last_seq` advances. Held until that batch is registered.
    commit_gate: Mutex<()>,
    /// Paired with `sync`. Wakes `sync_flush` when `notify_in_flight` clears.
    sync_cv: Condvar,
}

impl FerroStore {
    fn assemble(inner: Inner, label: String) -> Self {
        Self {
            inner,
            subs: Mutex::new(HashMap::new()),
            label,
            sync: Mutex::new(SyncSlot::default()),
            commit_gate: Mutex::new(()),
            sync_cv: Condvar::new(),
        }
    }

    /// Empty in-memory store. `checkpoint` succeeds and writes nothing.
    pub fn in_memory() -> Self {
        Self::assemble(
            Inner::Memory(Mutex::new(DocState::default())),
            "mem://".to_string(),
        )
    }

    /// Open or create a `.fls` file. A second open of the same path succeeds.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let path = path.as_ref();
        let write = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .map_err(|err| backend(format!("open {}: {err}", path.display())))?;
        ensure_superblock(&write)?;
        read_consistent(&write)?;
        // `OpenOptions::read` without write is `O_RDONLY`.
        let read = OpenOptions::new()
            .read(true)
            .open(path)
            .map_err(|err| backend(format!("open {}: {err}", path.display())))?;
        Ok(Self::assemble(
            Inner::Disk(Disk {
                write: Mutex::new(write),
                read,
            }),
            path.display().to_string(),
        ))
    }

    /// If `default_socket_path_for_store(path)` accepts a Unix connection, return a client of that server.
    /// Otherwise `open(path)` in embedded mode.
    ///
    /// The successful `UnixStream` is the client. It is not dropped in favor of a second connect.
    /// A refused connect (nothing listening, or no socket file) is the embedded fallback, not an error.
    /// `open` errors are still returned.
    pub fn auto(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let path = path.as_ref();
        let sock = default_socket_path_for_store(path);
        match UnixStream::connect(&sock) {
            Ok(stream) => {
                let remote = server::client_from_unix(stream)?;
                Ok(Self::assemble(
                    Inner::Remote(remote),
                    sock.display().to_string(),
                ))
            }
            // Nothing listening, or the socket file is missing. Other connect
            // errors stay errors. `open` errors are not swallowed.
            Err(err) if refused_connect(&err) => Self::open(path),
            Err(err) => Err(backend(format!("connect {}: {err}", sock.display()))),
        }
    }

    /// `mem://`, `file://`, `unix://`, and `ws://`. `wss://` is recognized and refused.
    pub fn connect_uri(uri: &str) -> Result<Self, StoreError> {
        if uri == "mem://" || uri == "mem:///" {
            return Ok(Self::in_memory());
        }
        if let Some(rest) = uri.strip_prefix("file://") {
            return Self::open(file_uri_path(rest));
        }
        if let Some(rest) = uri.strip_prefix("unix://") {
            let path = file_uri_path(rest);
            let remote = server::connect_unix(&path)?;
            return Ok(Self::assemble(Inner::Remote(remote), path));
        }
        if uri.starts_with("wss://") {
            return Err(backend("wss is not implemented"));
        }
        if let Some(rest) = uri.strip_prefix("ws://") {
            let remote = server::connect_ws(rest)?;
            return Ok(Self::assemble(Inner::Remote(remote), uri.to_string()));
        }
        Err(backend(format!("unknown FerroStore URI scheme in `{uri}`")))
    }

    pub fn checkpoint(&self) -> Result<(), StoreError> {
        match &self.inner {
            Inner::Remote(remote) => remote.call_unit("checkpoint", json!({})),
            Inner::Memory(_) => Ok(()),
            Inner::Disk(disk) => checkpoint_disk(disk),
        }
    }

    pub fn put_doc(&self, collection: &str, id: &str, doc: &Value) -> Result<(), StoreError> {
        self.put_doc_inner(collection, id, doc, false)
    }

    fn put_doc_inner(
        &self,
        collection: &str,
        id: &str,
        doc: &Value,
        silent: bool,
    ) -> Result<(), StoreError> {
        let collection = collection.to_string();
        let id = id.to_string();
        let doc = doc.clone();
        self.commit(
            move |_| {
                Ok((
                    Op::Put {
                        collection,
                        id,
                        doc,
                    },
                    (),
                ))
            },
            silent,
        )
    }

    pub fn get_doc(&self, collection: &str, id: &str) -> Result<Option<Value>, StoreError> {
        let remote_miss = if let Inner::Remote(remote) = &self.inner {
            match remote.call_option("get_doc", json!({"collection": collection, "id": id}))? {
                Some(doc) => return Ok(Some(doc)),
                None => true,
            }
        } else {
            false
        };
        if remote_miss {
            return self.fill_doc_miss(collection, id);
        }
        let state = self.read_state()?;
        if let Some(doc) = state
            .docs
            .get(&(collection.to_string(), id.to_string()))
            .cloned()
        {
            return Ok(Some(doc));
        }
        self.fill_doc_miss(collection, id)
    }

    pub fn delete_doc(&self, collection: &str, id: &str) -> Result<(), StoreError> {
        let collection = collection.to_string();
        let id = id.to_string();
        self.commit(move |_| Ok((Op::Delete { collection, id }, ())), false)
    }

    pub fn query_by_field(
        &self,
        collection: &str,
        field: &str,
        value: &str,
    ) -> Result<Vec<String>, StoreError> {
        if let Inner::Remote(remote) = &self.inner {
            return remote.call_strings(
                "query_by_field",
                json!({"collection": collection, "field": field, "value": value}),
            );
        }
        let state = self.read_state()?;
        Ok(state::query_by_field(&state, collection, field, value))
    }

    pub fn search_bm25(
        &self,
        collection: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(String, f64)>, StoreError> {
        if let Inner::Remote(remote) = &self.inner {
            return remote.call_pairs(
                "search_bm25",
                json!({"collection": collection, "query": query, "limit": limit}),
            );
        }
        let state = self.read_state()?;
        Ok(state::search_bm25(&state, collection, query, limit))
    }

    /// Create a column table. An existing name is an error and writes nothing.
    pub fn create_column_table(&self, name: &str, columns: &[ColumnDef]) -> Result<(), StoreError> {
        let name = name.to_string();
        let columns = columns.to_vec();
        self.with_write(move |txn| {
            columnar::check_new_table(&txn.tables, &name, &columns)?;
            txn.tables.insert(
                name.clone(),
                columnar::ColumnTable {
                    columns: columns.clone(),
                    groups: Vec::new(),
                },
            );
            txn.ops.push(Op::CreateColumnTable {
                name: name.clone(),
                columns: columns.clone(),
            });
            Ok(())
        })
    }

    /// Append one row group. An empty `rows` writes nothing. A bad row writes nothing.
    pub fn append_column_rows(
        &self,
        name: &str,
        rows: &[Vec<ColumnValue>],
    ) -> Result<(), StoreError> {
        let name = name.to_string();
        let rows = rows.to_vec();
        self.with_write(move |txn| {
            let Some(table) = txn.tables.get(&name) else {
                return Err(backend(format!("no column table `{name}`")));
            };
            if rows.is_empty() {
                return Ok(());
            }
            let bytes = columnar::encode_row_group(&table.columns, &rows)?;
            let Some(table) = txn.tables.get_mut(&name) else {
                return Err(backend(format!("no column table `{name}`")));
            };
            table.groups.push(bytes.clone());
            txn.ops.push(Op::AppendColumnGroup {
                name: name.clone(),
                bytes,
            });
            Ok(())
        })
    }

    /// Scan `name`. A row group whose zone map excludes any filter is skipped
    /// before its values are decoded. Filters are AND.
    pub fn scan_column_table(
        &self,
        name: &str,
        filters: &[ColumnFilter],
    ) -> Result<ColumnScan, StoreError> {
        if let Inner::Remote(remote) = &self.inner {
            let mut encoded = Vec::with_capacity(filters.len());
            for filter in filters {
                encoded.push(columnar::filter_json(filter)?);
            }
            let value = remote.call(
                "scan_column_table",
                json!({"name": name, "filters": encoded}),
            )?;
            return columnar::parse_scan(&value);
        }
        let name = name.to_string();
        let filters = filters.to_vec();
        self.with_read(move |txn| {
            let Some(table) = txn.tables.get(&name) else {
                return Err(backend(format!("no column table `{name}`")));
            };
            columnar::scan_groups(&table.columns, &table.groups, &filters)
        })
    }

    /// Sum attempt ledger rows through an in-memory columnar scan. `None` is every attempt.
    pub fn attempt_rollup(&self, project: Option<&ProjectId>) -> Result<AttemptRollup, StoreError> {
        if let Inner::Remote(remote) = &self.inner {
            let project = match project {
                Some(id) => Value::String(id.as_str().to_string()),
                None => Value::Null,
            };
            let value = remote.call("attempt_rollup", json!({"project": project}))?;
            return columnar::parse_attempt_rollup(&value);
        }
        let project = project.cloned();
        self.with_read(move |txn| columnar::attempt_rollup(txn, project.as_ref()))
    }

    /// Sum gate-run ledger rows through an in-memory columnar scan. `None` is every run.
    pub fn gate_run_rollup(&self, gate: Option<&GateId>) -> Result<GateRunRollup, StoreError> {
        if let Inner::Remote(remote) = &self.inner {
            let gate = match gate {
                Some(id) => Value::String(id.as_str().to_string()),
                None => Value::Null,
            };
            let value = remote.call("gate_run_rollup", json!({"gate": gate}))?;
            return columnar::parse_gate_rollup(&value);
        }
        let gate = gate.cloned();
        self.with_read(move |txn| columnar::gate_run_rollup(txn, gate.as_ref()))
    }

    /// Create an empty vector collection. An existing name is an error and writes nothing.
    /// A document collection may use the same name; the two maps are distinct.
    #[allow(clippy::too_many_arguments)]
    pub fn create_vector_collection(
        &self,
        name: &str,
        dim: u32,
        storage: VectorStorage,
        metric: VectorMetric,
        m: u32,
        ef_construction: u32,
        ef_search: u32,
    ) -> Result<(), StoreError> {
        let name = name.to_string();
        self.with_write(move |txn| {
            vector::check_create(&name, dim, storage, m, ef_construction, ef_search)?;
            if txn.vectors.contains_key(&name) {
                return Err(backend(format!(
                    "vector collection `{name}` already exists"
                )));
            }
            txn.ops.push(Op::CreateVectorCollection {
                name: name.clone(),
                dim,
                storage,
                metric,
                m,
                ef_construction,
                ef_search,
            });
            Ok(())
        })
    }

    /// Insert or replace one vector. `None` metadata is absent. `Some(Null)` stores JSON null.
    pub fn upsert_vector(
        &self,
        collection: &str,
        id: &str,
        values: &[f32],
        metadata: Option<&Value>,
    ) -> Result<(), StoreError> {
        let collection = collection.to_string();
        let id = id.to_string();
        let values = values.to_vec();
        let metadata = metadata.cloned();
        self.with_write(move |txn| {
            let Some(col) = txn.vectors.get(&collection) else {
                return Err(backend(format!("no vector collection `{collection}`")));
            };
            let dim = col.dim;
            let storage = col.storage;
            vector::check_vector_id(&id)?;
            if u32::try_from(values.len()).ok() != Some(dim) {
                return Err(backend(format!(
                    "vector values have length {}, expected {dim}",
                    values.len()
                )));
            }
            let bytes = vector::quantize(storage, &values)?;
            txn.ops.push(Op::UpsertVector {
                collection: collection.clone(),
                id: id.clone(),
                bytes,
                metadata: metadata.clone(),
            });
            Ok(())
        })
    }

    /// Delete one vector. A missing id still commits. A missing collection writes nothing.
    pub fn delete_vector(&self, collection: &str, id: &str) -> Result<(), StoreError> {
        let collection = collection.to_string();
        let id = id.to_string();
        self.with_write(move |txn| {
            if !txn.vectors.contains_key(&collection) {
                return Err(backend(format!("no vector collection `{collection}`")));
            }
            vector::check_vector_id(&id)?;
            txn.ops.push(Op::DeleteVector {
                collection: collection.clone(),
                id: id.clone(),
            });
            Ok(())
        })
    }

    /// Dequantized values and metadata. `Ok(None)` when the id is missing.
    #[allow(clippy::type_complexity)]
    pub fn get_vector(
        &self,
        collection: &str,
        id: &str,
    ) -> Result<Option<(Vec<f32>, Option<Value>)>, StoreError> {
        if let Inner::Remote(remote) = &self.inner {
            return vector::parse_get(
                &remote.call("get_vector", json!({"collection": collection, "id": id}))?,
            );
        }
        let collection = collection.to_string();
        let id = id.to_string();
        self.with_read(move |txn| {
            let Some(col) = txn.vectors.get(&collection) else {
                return Err(backend(format!("no vector collection `{collection}`")));
            };
            vector::check_vector_id(&id)?;
            let Some(record) = col.vectors.get(&id) else {
                return Ok(None);
            };
            let values = vector::dequantize(col.storage, &record.bytes)?;
            Ok(Some((values, record.metadata.clone())))
        })
    }

    /// Higher-better scores, at most `limit`. `ef` of `None` uses the collection `ef_search`.
    /// `limit == 0` or `ef == Some(0)` is an empty vec.
    pub fn search_vectors(
        &self,
        collection: &str,
        values: &[f32],
        limit: usize,
        ef: Option<u32>,
    ) -> Result<Vec<(String, f64)>, StoreError> {
        if let Some(adapter) = self.attached_adapter()
            && let Some(hits) = cloud_hits(adapter.search_vectors(collection, values, limit, ef))
        {
            return Ok(hits);
        }
        if let Inner::Remote(remote) = &self.inner {
            let mut params = serde_json::Map::new();
            params.insert("collection".into(), json!(collection));
            params.insert("values".into(), vector::values_json(values)?);
            params.insert("limit".into(), json!(limit));
            if let Some(ef) = ef {
                params.insert("ef".into(), json!(ef));
            }
            return vector::parse_hits(&remote.call("search_vectors", Value::Object(params))?);
        }
        let collection = collection.to_string();
        let values = values.to_vec();
        self.with_read(move |txn| {
            let Some(col) = txn.vectors.get(&collection) else {
                return Err(backend(format!("no vector collection `{collection}`")));
            };
            vector::search(col, &values, limit, ef)
        })
    }

    /// Fuse at most `limit` BM25 hits with at most `limit` vector hits, then keep `limit`.
    pub fn search_hybrid(
        &self,
        doc_collection: &str,
        vector_collection: &str,
        query: &str,
        values: &[f32],
        limit: usize,
        fusion: Fusion,
    ) -> Result<Vec<(String, f64)>, StoreError> {
        if let Some(adapter) = self.attached_adapter()
            && let Some(hits) = cloud_hits(adapter.search_hybrid(
                doc_collection,
                vector_collection,
                query,
                values,
                limit,
                fusion,
            ))
        {
            return Ok(hits);
        }
        if let Inner::Remote(remote) = &self.inner {
            return vector::parse_hits(&remote.call(
                "search_hybrid",
                hybrid_params(
                    doc_collection,
                    vector_collection,
                    query,
                    values,
                    limit,
                    fusion,
                )?,
            )?);
        }
        let doc_collection = doc_collection.to_string();
        let vector_collection = vector_collection.to_string();
        let query = query.to_string();
        let values = values.to_vec();
        self.with_read(move |txn| {
            state::search_hybrid(
                &txn.docs,
                &txn.vectors,
                &doc_collection,
                &vector_collection,
                &query,
                &values,
                limit,
                fusion,
            )
        })
    }

    /// Same fusion as [`Self::search_hybrid`]. `doc` is the document at that id, if any.
    /// `metadata` is the vector metadata when the id is in the vector collection.
    pub fn search_knowledge(
        &self,
        doc_collection: &str,
        vector_collection: &str,
        query: &str,
        values: &[f32],
        limit: usize,
        fusion: Fusion,
    ) -> Result<Vec<KnowledgeHit>, StoreError> {
        if let Inner::Remote(remote) = &self.inner {
            return vector::parse_knowledge(&remote.call(
                "search_knowledge",
                hybrid_params(
                    doc_collection,
                    vector_collection,
                    query,
                    values,
                    limit,
                    fusion,
                )?,
            )?);
        }
        let doc_collection = doc_collection.to_string();
        let vector_collection = vector_collection.to_string();
        let query = query.to_string();
        let values = values.to_vec();
        self.with_read(move |txn| {
            state::search_knowledge(
                &txn.docs,
                &txn.vectors,
                &doc_collection,
                &vector_collection,
                &query,
                &values,
                limit,
                fusion,
            )
        })
    }

    /// Append one message. The returned sequence starts at 1 per topic and is
    /// not reused after that message document is deleted.
    pub fn publish_mq(&self, topic: &str, key: &str, payload: &Value) -> Result<u64, StoreError> {
        Ok(self.publish_mq_detail(topic, key, payload)?.0)
    }

    /// Message sequence and the store sequence captured under the document lock.
    pub(super) fn publish_mq_detail(
        &self,
        topic: &str,
        key: &str,
        payload: &Value,
    ) -> Result<(u64, u64), StoreError> {
        if let Inner::Remote(remote) = &self.inner {
            let registered = {
                let _gate = lock_mutex(&self.commit_gate);
                let value = remote.call(
                    "publish_mq",
                    json!({"topic": topic, "key": key, "payload": payload}),
                )?;
                let (mq_seq, store_seq) = parse_publish_result(&value)?;
                let registered = self.register_committed(
                    &Committed {
                        seq: store_seq,
                        ops: mq_publish_ops(topic, key, payload, mq_seq),
                    },
                    false,
                );
                (mq_seq, store_seq, registered)
            };
            self.deliver_owned(registered.2)?;
            return Ok((registered.0, registered.1));
        }
        let topic_owned = topic.to_string();
        let key_owned = key.to_string();
        let payload = payload.clone();
        // One `with_write`: the message and the high-water share a WAL frame.
        // A crash must not persist either document alone.
        let (record, store_seq) = self.with_write_outcome(move |txn| {
            let seq = next_mq_seq(&txn.docs, &topic_owned)?;
            let record = MqRecord {
                seq,
                key: key_owned.clone(),
                payload: payload.clone(),
            };
            txn.put(
                mq_message_collection(&topic_owned),
                seq.to_string(),
                serde_json::json!({
                    "seq": seq,
                    "key": key_owned,
                    "payload": payload,
                }),
            );
            txn.put(
                mq_hw_collection(&topic_owned),
                MQ_HIGH_WATER_ID,
                serde_json::json!({ "seq": seq }),
            );
            Ok(record)
        })?;
        let store_seq = store_seq.ok_or_else(|| backend("publish_mq did not commit"))?;
        self.notify(topic, &record);
        Ok((record.seq, store_seq))
    }

    /// Oldest messages for `topic` that `group` has not acked, without acking them.
    pub fn poll_mq_group(
        &self,
        topic: &str,
        group: &str,
        limit: usize,
    ) -> Result<Vec<MqRecord>, StoreError> {
        if let Inner::Remote(remote) = &self.inner {
            return remote.call_records(
                "poll_mq_group",
                json!({"topic": topic, "group": group, "limit": limit}),
            );
        }
        let state = self.read_state()?;
        unread_mq(&state, topic, group, limit)
    }

    /// Record an ack. An unknown sequence is still stored.
    pub fn ack_mq(&self, topic: &str, group: &str, seq: u64) -> Result<(), StoreError> {
        let topic = topic.to_string();
        let group = group.to_string();
        self.commit(
            move |_| {
                Ok((
                    Op::Put {
                        collection: format!("mq_ack:{topic}:{group}"),
                        id: seq.to_string(),
                        doc: serde_json::json!({ "seq": seq }),
                    },
                    (),
                ))
            },
            false,
        )
    }

    /// Later publishes on this handle. History is not replayed.
    pub fn subscribe_mq(&self, topic: &str) -> Receiver<MqRecord> {
        if let Inner::Remote(remote) = &self.inner {
            return remote.subscribe_topic(topic);
        }
        let (tx, rx) = mpsc::channel();
        lock_mutex(&self.subs)
            .entry(topic.to_string())
            .or_default()
            .push(tx);
        rx
    }

    /// Attach `adapter` in `mode`. A pending batch keeps the current adapter.
    pub fn attach_sync_adapter(
        &self,
        mode: SyncMode,
        adapter: Arc<dyn SyncAdapter>,
    ) -> Result<(), StoreError> {
        let mut slot = lock_mutex(&self.sync);
        if !slot.is_idle() {
            return Err(backend(
                "cannot attach a sync adapter while batches are pending",
            ));
        }
        slot.mode = Some(mode);
        slot.adapter = Some(adapter);
        Ok(())
    }

    /// Set `mode` and `adapter` when the pending buffer is empty.
    /// Pending batches keep the current adapter.
    pub fn with_sync_adapter(
        self,
        mode: SyncMode,
        adapter: Arc<dyn SyncAdapter>,
    ) -> Result<Self, StoreError> {
        self.attach_sync_adapter(mode, adapter)?;
        Ok(self)
    }

    /// Drop the adapter. A pending batch is an error and the adapter stays.
    /// `synced_seq` is kept.
    pub fn detach_sync_adapter(&self) -> Result<(), StoreError> {
        let mut slot = lock_mutex(&self.sync);
        if !slot.is_idle() {
            return Err(backend(
                "cannot detach a sync adapter while batches are pending",
            ));
        }
        slot.mode = None;
        slot.adapter = None;
        Ok(())
    }

    pub fn sync_mode(&self) -> Option<SyncMode> {
        lock_mutex(&self.sync).mode
    }

    pub fn synced_seq(&self) -> u64 {
        lock_mutex(&self.sync).synced_seq
    }

    /// `true` while a batch is queued or an `on_commit` call is running.
    pub fn has_pending_sync(&self) -> bool {
        !lock_mutex(&self.sync).is_idle()
    }

    /// The attached adapter, cloned so no lock is held across a call into it.
    fn attached_adapter(&self) -> Option<Arc<dyn SyncAdapter>> {
        lock_mutex(&self.sync).adapter.clone()
    }

    /// Call `on_commit` for each pending batch, oldest first.
    ///
    /// Waits while another thread's notify is in flight, then sends only the
    /// front batch. The thread that owns that notify returns `synced_seq` and
    /// does not send. A later batch stays pending until this loop reaches it.
    pub fn sync_flush(&self) -> Result<u64, StoreError> {
        let adapter = {
            let slot = lock_mutex(&self.sync);
            slot.adapter
                .clone()
                .ok_or_else(|| backend("no sync adapter is attached"))?
        };
        loop {
            let batch = {
                let mut slot = lock_mutex(&self.sync);
                while slot.notify_in_flight {
                    if slot.notify_owner == Some(std::thread::current().id()) {
                        return Ok(slot.synced_seq);
                    }
                    slot = wait_mutex(&self.sync_cv, slot);
                }
                match slot.pending.first() {
                    Some(batch) => {
                        let batch = batch.clone();
                        slot.notify_in_flight = true;
                        slot.notify_owner = Some(std::thread::current().id());
                        batch
                    }
                    None => return Ok(slot.synced_seq),
                }
            };
            let result = self.call_on_commit(&adapter, &batch);
            let mut slot = lock_mutex(&self.sync);
            slot.notify_in_flight = false;
            slot.notify_owner = None;
            let failed = match result {
                Ok(()) => {
                    if slot
                        .pending
                        .first()
                        .is_some_and(|pending| pending.seq == batch.seq)
                    {
                        slot.pending.remove(0);
                        if batch.seq > slot.synced_seq {
                            slot.synced_seq = batch.seq;
                        }
                    }
                    None
                }
                Err(err) => Some(err),
            };
            self.sync_cv.notify_all();
            if let Some(err) = failed {
                return Err(err);
            }
        }
    }

    /// Apply `pull_since(last_seq)` with sync suppressed.
    pub fn sync_pull(&self) -> Result<u64, StoreError> {
        let adapter = self.require_sync_adapter()?;
        self.require_sync_idle()?;
        let last = self.read_state()?.last_seq;
        let batches = adapter.pull_since(last)?;
        self.apply_inbound(batches)
    }

    /// Apply `prewarm` batches with sync suppressed.
    pub fn sync_prewarm(&self, collections: &[&str]) -> Result<u64, StoreError> {
        let adapter = self.require_sync_adapter()?;
        self.require_sync_idle()?;
        let batches = adapter.prewarm(collections)?;
        self.apply_inbound(batches)
    }

    fn with_read<T>(
        &self,
        body: impl FnOnce(&Txn) -> Result<T, StoreError>,
    ) -> Result<T, StoreError> {
        let state = self.read_state()?;
        let txn = Txn {
            docs: state.docs,
            tables: state.tables,
            vectors: state.vectors,
            ops: Vec::new(),
        };
        body(&txn)
    }

    fn read_state(&self) -> Result<DocState, StoreError> {
        match &self.inner {
            Inner::Memory(state) => Ok(lock_mutex(state).clone()),
            Inner::Disk(disk) => Ok(read_consistent(&disk.read)?.2),
            Inner::Remote(remote) => remote.read_state(),
        }
    }

    /// Compare-and-swap `ops` onto a store whose `last_seq` is still `expected_seq`.
    ///
    /// `Ok(false)` means the sequence moved and nothing was written. An empty `ops`
    /// list that matches returns `Ok(true)` and does not advance `last_seq`. A
    /// non-empty disk success is one WAL frame at `expected_seq + 1`. Ops that do
    /// not apply return `Err` and write nothing. A remote handle sends `commit_ops`.
    #[cfg(test)]
    fn commit_if_seq(&self, expected_seq: u64, ops: &[Op]) -> Result<bool, StoreError> {
        self.commit_if_seq_inner(expected_seq, ops, false)
    }

    fn commit_if_seq_inner(
        &self,
        expected_seq: u64,
        ops: &[Op],
        silent: bool,
    ) -> Result<bool, StoreError> {
        match &self.inner {
            Inner::Memory(slot) => {
                let registered = {
                    let _gate = lock_mutex(&self.commit_gate);
                    let committed = {
                        let mut state = lock_mutex(slot);
                        if state.last_seq != expected_seq {
                            return Ok(false);
                        }
                        if ops.is_empty() {
                            return Ok(true);
                        }
                        let owned = ops.to_vec();
                        commit_ops(&mut state, &owned)?;
                        Committed {
                            seq: state.last_seq,
                            ops: owned,
                        }
                    };
                    self.register_committed(&committed, silent)
                };
                self.deliver_owned(registered)?;
                Ok(true)
            }
            Inner::Disk(disk) => {
                let registered = {
                    let _gate = lock_mutex(&self.commit_gate);
                    match commit_if_seq_disk(disk, expected_seq, ops)? {
                        SeqCommit::Lost => return Ok(false),
                        SeqCommit::Same => return Ok(true),
                        SeqCommit::Wrote(committed) => self.register_committed(&committed, silent),
                    }
                };
                self.deliver_owned(registered)?;
                Ok(true)
            }
            Inner::Remote(remote) => {
                if ops.is_empty() {
                    return Ok(remote.read_state()?.last_seq == expected_seq);
                }
                let registered = {
                    let _gate = lock_mutex(&self.commit_gate);
                    if !remote.commit_ops(expected_seq, ops, silent)? {
                        return Ok(false);
                    }
                    let seq = expected_seq
                        .checked_add(1)
                        .ok_or_else(|| backend("transaction sequence overflow"))?;
                    self.register_committed(
                        &Committed {
                            seq,
                            ops: ops.to_vec(),
                        },
                        silent,
                    )
                };
                self.deliver_owned(registered)?;
                Ok(true)
            }
        }
    }

    /// Run `body` against a cloned document map.
    ///
    /// Memory holds the document mutex for the whole call. Disk takes `LOCK_EX`,
    /// re-reads a consistent snapshot, then builds the transaction from that
    /// state. `Ok` with at least one op appends one WAL frame (`seq = last_seq + 1`)
    /// and publishes one superblock. `Err` or an empty op list writes nothing.
    /// A remote handle compare-and-swaps those ops and runs `body` again when
    /// `last_seq` changed. The body may run more than once only on a remote handle.
    /// The sync adapter runs after that lock and the commit gate are released.
    fn with_write<T>(
        &self,
        body: impl FnMut(&mut Txn) -> Result<T, StoreError>,
    ) -> Result<T, StoreError> {
        Ok(self.with_write_outcome(body)?.0)
    }

    /// Like [`Self::with_write`]. The sequence was taken before the document
    /// mutex or disk flock was released. `None` means the body wrote no ops.
    fn with_write_outcome<T>(
        &self,
        mut body: impl FnMut(&mut Txn) -> Result<T, StoreError>,
    ) -> Result<(T, Option<u64>), StoreError> {
        match &self.inner {
            Inner::Remote(remote) => loop {
                let state = remote.read_state()?;
                let seq_base = state.last_seq;
                let mut txn = Txn {
                    docs: state.docs,
                    tables: state.tables,
                    vectors: state.vectors,
                    ops: Vec::new(),
                };
                let result = body(&mut txn)?;
                if txn.ops.is_empty() {
                    return Ok((result, None));
                }
                let committed = {
                    let _gate = lock_mutex(&self.commit_gate);
                    if !remote.commit_ops(seq_base, &txn.ops, false)? {
                        None
                    } else {
                        let seq = seq_base
                            .checked_add(1)
                            .ok_or_else(|| backend("transaction sequence overflow"))?;
                        let committed = Committed { seq, ops: txn.ops };
                        let registered = self.register_committed(&committed, false);
                        Some((result, seq, registered))
                    }
                };
                if let Some((result, seq, registered)) = committed {
                    self.deliver_owned(registered)?;
                    return Ok((result, Some(seq)));
                }
            },
            Inner::Memory(slot) => {
                let (result, seq, registered) = {
                    let _gate = lock_mutex(&self.commit_gate);
                    let (result, committed) = {
                        let mut state = lock_mutex(slot);
                        let mut txn = Txn {
                            docs: state.docs.clone(),
                            tables: state.tables.clone(),
                            vectors: state.vectors.clone(),
                            ops: Vec::new(),
                        };
                        let result = body(&mut txn)?;
                        let ops = std::mem::take(&mut txn.ops);
                        commit_ops(&mut state, &ops)?;
                        let committed = if ops.is_empty() {
                            None
                        } else {
                            Some(Committed {
                                seq: state.last_seq,
                                ops,
                            })
                        };
                        (result, committed)
                    };
                    let seq = committed.as_ref().map(|committed| committed.seq);
                    let registered = committed
                        .as_ref()
                        .and_then(|committed| self.register_committed(committed, false));
                    (result, seq, registered)
                };
                self.deliver_owned(registered)?;
                Ok((result, seq))
            }
            Inner::Disk(disk) => {
                let (result, seq, registered) = {
                    let _gate = lock_mutex(&self.commit_gate);
                    let (result, committed) = with_write_disk(disk, body)?;
                    let seq = committed.as_ref().map(|committed| committed.seq);
                    let registered = committed
                        .as_ref()
                        .and_then(|committed| self.register_committed(committed, false));
                    (result, seq, registered)
                };
                self.deliver_owned(registered)?;
                Ok((result, seq))
            }
        }
    }

    fn commit<T>(
        &self,
        build: impl FnOnce(&DocState) -> Result<(Op, T), StoreError>,
        silent: bool,
    ) -> Result<T, StoreError> {
        match &self.inner {
            Inner::Remote(remote) => {
                let state = remote.read_state()?;
                let (op, extra) = build(&state)?;
                let mut expected = state.last_seq;
                loop {
                    let registered = {
                        let _gate = lock_mutex(&self.commit_gate);
                        if !remote.commit_ops(expected, std::slice::from_ref(&op), silent)? {
                            None
                        } else {
                            let seq = expected
                                .checked_add(1)
                                .ok_or_else(|| backend("transaction sequence overflow"))?;
                            Some(self.register_committed(
                                &Committed {
                                    seq,
                                    ops: vec![op.clone()],
                                },
                                silent,
                            ))
                        }
                    };
                    if let Some(registered) = registered {
                        self.deliver_owned(registered)?;
                        return Ok(extra);
                    }
                    expected = remote.read_state()?.last_seq;
                }
            }
            Inner::Memory(slot) => {
                let (extra, registered) = {
                    let _gate = lock_mutex(&self.commit_gate);
                    let (extra, committed) = {
                        let mut state = lock_mutex(slot);
                        let (op, extra) = build(&state)?;
                        apply_commit(&mut state, op.clone())?;
                        (
                            extra,
                            Committed {
                                seq: state.last_seq,
                                ops: vec![op],
                            },
                        )
                    };
                    (extra, self.register_committed(&committed, silent))
                };
                self.deliver_owned(registered)?;
                Ok(extra)
            }
            Inner::Disk(disk) => {
                let (extra, registered) = {
                    let _gate = lock_mutex(&self.commit_gate);
                    let (extra, committed) = commit_disk(disk, build)?;
                    (extra, self.register_committed(&committed, silent))
                };
                self.deliver_owned(registered)?;
                Ok(extra)
            }
        }
    }

    fn fill_doc_miss(&self, collection: &str, id: &str) -> Result<Option<Value>, StoreError> {
        let adapter = {
            let slot = lock_mutex(&self.sync);
            match slot.mode {
                Some(SyncMode::LocalCache | SyncMode::CloudPreCache) if slot.is_idle() => {
                    slot.adapter.clone()
                }
                _ => None,
            }
        };
        let Some(adapter) = adapter else {
            return Ok(None);
        };
        match adapter.fetch_document(collection, id)? {
            None => Ok(None),
            Some(doc) => self.install_fetched_doc(collection, id, doc),
        }
    }

    /// Silent put of a cache fill. Does not call `on_commit` or queue the batch.
    ///
    /// Holds the commit gate, then the document lock, then the sync mutex.
    /// A commit releases the document lock before taking the sync mutex.
    fn install_fetched_doc(
        &self,
        collection: &str,
        id: &str,
        fetched: Value,
    ) -> Result<Option<Value>, StoreError> {
        let key = (collection.to_string(), id.to_string());
        let _gate = lock_mutex(&self.commit_gate);
        if let Inner::Remote(remote) = &self.inner {
            if !lock_mutex(&self.sync).is_idle() {
                drop(_gate);
                return remote.call_option("get_doc", json!({"collection": collection, "id": id}));
            }
            let value = remote.call(
                "put_doc",
                json!({
                    "collection": collection,
                    "id": id,
                    "doc": fetched,
                    "silent": true,
                    "if_absent": true,
                }),
            )?;
            let (wrote, doc, store_seq) = parse_if_absent(&value)?;
            if let Some(store_seq) = store_seq {
                let mut slot = lock_mutex(&self.sync);
                if wrote && slot.is_idle() && store_seq > slot.synced_seq {
                    slot.synced_seq = store_seq;
                }
            }
            return Ok(Some(doc));
        }
        match &self.inner {
            Inner::Remote(_) => unreachable!("remote fill returned above"),
            Inner::Memory(slot) => {
                let mut state = lock_mutex(slot);
                let mut sync = lock_mutex(&self.sync);
                if !sync.is_idle() {
                    return Ok(state.docs.get(&key).cloned());
                }
                if let Some(existing) = state.docs.get(&key).cloned() {
                    return Ok(Some(existing));
                }
                let op = Op::Put {
                    collection: key.0.clone(),
                    id: key.1.clone(),
                    doc: fetched.clone(),
                };
                apply_commit(&mut state, op)?;
                let seq = state.last_seq;
                if sync.is_idle() && seq > sync.synced_seq {
                    sync.synced_seq = seq;
                }
                Ok(Some(fetched))
            }
            Inner::Disk(disk) => {
                let file = lock_mutex(&disk.write);
                let _locked = FlockGuard::acquire(file.as_raw_fd())?;
                let (index, sb, state) = read_consistent(&file)?;
                let mut sync = lock_mutex(&self.sync);
                if !sync.is_idle() {
                    return Ok(state.docs.get(&key).cloned());
                }
                if let Some(existing) = state.docs.get(&key).cloned() {
                    return Ok(Some(existing));
                }
                let op = Op::Put {
                    collection: key.0.clone(),
                    id: key.1.clone(),
                    doc: fetched.clone(),
                };
                let seq = stage_ops(state, std::slice::from_ref(&op))?.last_seq;
                let frame = encode_frame(seq, std::slice::from_ref(&op))?;
                append_frame(&file, index, &sb, &frame)?;
                if sync.is_idle() && seq > sync.synced_seq {
                    sync.synced_seq = seq;
                }
                Ok(Some(fetched))
            }
        }
    }

    /// Conditional put used by a remote cache fill. The document lock covers
    /// the absence check and the write. `store_seq` is taken before that lock
    /// is released.
    fn put_if_absent(
        &self,
        collection: &str,
        id: &str,
        doc: &Value,
        silent: bool,
    ) -> Result<IfAbsent, StoreError> {
        let key = (collection.to_string(), id.to_string());
        match &self.inner {
            Inner::Remote(_) => Err(backend(
                "if_absent put is local to the store that owns the file",
            )),
            Inner::Memory(slot) => {
                let (seq, registered) = {
                    let _gate = lock_mutex(&self.commit_gate);
                    let committed = {
                        let mut state = lock_mutex(slot);
                        if let Some(existing) = state.docs.get(&key).cloned() {
                            return Ok(IfAbsent {
                                wrote: false,
                                doc: existing,
                                store_seq: None,
                            });
                        }
                        let op = Op::Put {
                            collection: key.0.clone(),
                            id: key.1.clone(),
                            doc: doc.clone(),
                        };
                        apply_commit(&mut state, op.clone())?;
                        (state.last_seq, op)
                    };
                    (
                        committed.0,
                        self.register_committed(
                            &Committed {
                                seq: committed.0,
                                ops: vec![committed.1],
                            },
                            silent,
                        ),
                    )
                };
                self.deliver_owned(registered)?;
                Ok(IfAbsent {
                    wrote: true,
                    doc: doc.clone(),
                    store_seq: Some(seq),
                })
            }
            Inner::Disk(disk) => {
                let (seq, registered) = {
                    let _gate = lock_mutex(&self.commit_gate);
                    let committed = {
                        let file = lock_mutex(&disk.write);
                        let _locked = FlockGuard::acquire(file.as_raw_fd())?;
                        let (index, sb, state) = read_consistent(&file)?;
                        if let Some(existing) = state.docs.get(&key).cloned() {
                            return Ok(IfAbsent {
                                wrote: false,
                                doc: existing,
                                store_seq: None,
                            });
                        }
                        let op = Op::Put {
                            collection: key.0,
                            id: key.1,
                            doc: doc.clone(),
                        };
                        let seq = stage_ops(state, std::slice::from_ref(&op))?.last_seq;
                        let frame = encode_frame(seq, std::slice::from_ref(&op))?;
                        append_frame(&file, index, &sb, &frame)?;
                        (seq, op)
                    };
                    (
                        committed.0,
                        self.register_committed(
                            &Committed {
                                seq: committed.0,
                                ops: vec![committed.1],
                            },
                            silent,
                        ),
                    )
                };
                self.deliver_owned(registered)?;
                Ok(IfAbsent {
                    wrote: true,
                    doc: doc.clone(),
                    store_seq: Some(seq),
                })
            }
        }
    }

    fn require_sync_adapter(&self) -> Result<Arc<dyn SyncAdapter>, StoreError> {
        lock_mutex(&self.sync)
            .adapter
            .clone()
            .ok_or_else(|| backend("no sync adapter is attached"))
    }

    fn require_sync_idle(&self) -> Result<(), StoreError> {
        if lock_mutex(&self.sync).is_idle() {
            Ok(())
        } else {
            Err(backend(
                "cannot apply sync batches while commits are pending",
            ))
        }
    }

    fn apply_inbound(&self, batches: Vec<SyncBatch>) -> Result<u64, StoreError> {
        self.apply_sync_batches(batches, true)
    }

    /// Dry-run every batch, then publish them in one critical section.
    /// `silent` does not call `on_commit`. An empty list returns `synced_seq`.
    ///
    /// A loud apply queues every prepared batch. Only `WriteThrough` then calls
    /// `sync_flush`. The other modes leave the batches pending.
    /// The commit gate covers the idle check, the publish, and that registration.
    fn apply_sync_batches(&self, batches: Vec<SyncBatch>, silent: bool) -> Result<u64, StoreError> {
        if batches.is_empty() {
            return Ok(self.synced_seq());
        }
        if let Inner::Remote(remote) = &self.inner {
            let seq = {
                let _gate = lock_mutex(&self.commit_gate);
                self.require_sync_idle()?;
                let seq = remote.apply_sync_batches(&batches, silent)?;
                let mut slot = lock_mutex(&self.sync);
                if slot.is_idle() && seq > slot.synced_seq {
                    slot.synced_seq = seq;
                }
                seq
            };
            return Ok(seq);
        }
        let (seq, flush) = {
            let _gate = lock_mutex(&self.commit_gate);
            if !lock_mutex(&self.sync).is_idle() {
                return Err(backend(
                    "cannot apply sync batches while commits are pending",
                ));
            }
            let (seq, prepared) = publish_sync_batches(&self.inner, &batches)?;
            if silent {
                let mut slot = lock_mutex(&self.sync);
                if slot.is_idle() && seq > slot.synced_seq {
                    slot.synced_seq = seq;
                }
                return Ok(seq);
            }
            let mut slot = lock_mutex(&self.sync);
            let queue = slot.adapter.is_some() && slot.mode.is_some();
            let flush = queue && slot.mode == Some(SyncMode::WriteThrough);
            if queue {
                for batch in prepared {
                    slot.pending.push(SyncBatch {
                        seq: batch.seq,
                        ops: batch.ops.iter().map(SyncOp::from_op).collect(),
                    });
                }
            }
            (seq, flush)
        };
        if flush {
            self.sync_flush()?;
        }
        Ok(seq)
    }

    /// Caller holds the commit gate and no document lock. Does not call `on_commit`.
    fn register_committed(
        &self,
        committed: &Committed,
        silent: bool,
    ) -> Option<(Arc<dyn SyncAdapter>, SyncBatch)> {
        if committed.ops.is_empty() {
            return None;
        }
        let batch = SyncBatch {
            seq: committed.seq,
            ops: committed.ops.iter().map(SyncOp::from_op).collect(),
        };
        let mut slot = lock_mutex(&self.sync);
        if silent {
            if slot.is_idle() && batch.seq > slot.synced_seq {
                slot.synced_seq = batch.seq;
            }
            return None;
        }
        let adapter = slot.adapter.clone()?;
        match slot.mode {
            Some(SyncMode::WriteThrough) => {
                if !slot.is_idle() {
                    slot.pending.push(batch);
                    None
                } else {
                    slot.notify_in_flight = true;
                    slot.notify_owner = Some(std::thread::current().id());
                    Some((adapter, batch))
                }
            }
            Some(SyncMode::WriteBehind | SyncMode::LocalCache | SyncMode::CloudPreCache) => {
                slot.pending.push(batch);
                None
            }
            None => None,
        }
    }

    /// Run `adapter.on_commit(batch)` for a caller that has set `notify_in_flight`.
    ///
    /// If the adapter panics, the flag, the owner and the queue are put right before the panic
    /// goes on: the batch stays queued, so a later `sync_flush` sends it. Without this, the flag
    /// would stay set for good and every other thread's `sync_flush` would wait on `sync_cv`
    /// forever, while `attach`, `detach` and `sync_pull` would refuse for good.
    fn call_on_commit(
        &self,
        adapter: &Arc<dyn SyncAdapter>,
        batch: &SyncBatch,
    ) -> Result<(), StoreError> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| adapter.on_commit(batch))) {
            Ok(result) => result,
            Err(payload) => {
                let mut slot = lock_mutex(&self.sync);
                slot.notify_in_flight = false;
                slot.notify_owner = None;
                if !slot.pending.iter().any(|p| p.seq == batch.seq) {
                    slot.pending.insert(0, batch.clone());
                }
                self.sync_cv.notify_all();
                drop(slot);
                std::panic::resume_unwind(payload)
            }
        }
    }

    /// Call `on_commit` after the commit gate has been released.
    /// Later pending batches stay queued for `sync_flush`.
    fn deliver_owned(
        &self,
        owned: Option<(Arc<dyn SyncAdapter>, SyncBatch)>,
    ) -> Result<(), StoreError> {
        let Some((adapter, batch)) = owned else {
            return Ok(());
        };
        let result = self.call_on_commit(&adapter, &batch);
        let mut slot = lock_mutex(&self.sync);
        slot.notify_in_flight = false;
        slot.notify_owner = None;
        match result {
            Ok(()) => {
                if batch.seq > slot.synced_seq {
                    slot.synced_seq = batch.seq;
                }
            }
            Err(err) => {
                if !slot.pending.iter().any(|pending| pending.seq == batch.seq) {
                    slot.pending.insert(0, batch);
                }
                self.sync_cv.notify_all();
                return Err(err);
            }
        }
        self.sync_cv.notify_all();
        Ok(())
    }

    fn notify(&self, topic: &str, record: &MqRecord) {
        let mut subs = lock_mutex(&self.subs);
        let Some(senders) = subs.get_mut(topic) else {
            return;
        };
        senders.retain(|sender| sender.send(record.clone()).is_ok());
        if senders.is_empty() {
            subs.remove(topic);
        }
    }
}

/// Cloud offload decision for a search. `Some` is the answer. `None` means search locally.
///
/// `Ok(None)` is the adapter declining. `Err(_)` also falls back to the local
/// index, so a cloud outage degrades a search and does not fail it. To make a
/// cloud error fail the search instead, return it from here (change the return
/// type to `Result<Option<..>, StoreError>` and use `?` at the two call sites).
fn cloud_hits(
    result: Result<Option<Vec<vector::VectorHit>>, StoreError>,
) -> Option<Vec<(String, f64)>> {
    match result {
        Ok(Some(hits)) => Some(hits.into_iter().map(|hit| (hit.id, hit.score)).collect()),
        Ok(None) | Err(_) => None,
    }
}

struct IfAbsent {
    wrote: bool,
    doc: Value,
    store_seq: Option<u64>,
}

fn parse_publish_result(value: &Value) -> Result<(u64, u64), StoreError> {
    let obj = value
        .as_object()
        .ok_or_else(|| backend("publish_mq result is not an object"))?;
    let seq = obj
        .get("seq")
        .and_then(Value::as_u64)
        .ok_or_else(|| backend("publish_mq result is missing seq"))?;
    let store_seq = obj
        .get("store_seq")
        .and_then(Value::as_u64)
        .ok_or_else(|| backend("publish_mq result is missing store_seq"))?;
    Ok((seq, store_seq))
}

fn parse_if_absent(value: &Value) -> Result<(bool, Value, Option<u64>), StoreError> {
    let obj = value
        .as_object()
        .ok_or_else(|| backend("put_doc if_absent result is not an object"))?;
    let wrote = obj
        .get("wrote")
        .and_then(Value::as_bool)
        .ok_or_else(|| backend("put_doc if_absent result is missing wrote"))?;
    let doc = obj
        .get("doc")
        .cloned()
        .ok_or_else(|| backend("put_doc if_absent result is missing doc"))?;
    if !wrote {
        return Ok((false, doc, None));
    }
    let store_seq = obj
        .get("store_seq")
        .and_then(Value::as_u64)
        .ok_or_else(|| backend("put_doc if_absent result is missing store_seq"))?;
    Ok((true, doc, Some(store_seq)))
}

fn mq_publish_ops(topic: &str, key: &str, payload: &Value, mq_seq: u64) -> Vec<Op> {
    vec![
        Op::Put {
            collection: mq_message_collection(topic),
            id: mq_seq.to_string(),
            doc: json!({"seq": mq_seq, "key": key, "payload": payload}),
        },
        Op::Put {
            collection: mq_hw_collection(topic),
            id: MQ_HIGH_WATER_ID.to_string(),
            doc: json!({"seq": mq_seq}),
        },
    ]
}

struct PreparedBatch {
    seq: u64,
    ops: Vec<Op>,
}

fn prepare_sync_batches(
    state: &DocState,
    batches: &[SyncBatch],
) -> Result<(DocState, Vec<PreparedBatch>), StoreError> {
    let mut scratch = state.clone();
    let mut expect = state.last_seq;
    let mut prepared = Vec::with_capacity(batches.len());
    for batch in batches {
        let next = expect
            .checked_add(1)
            .ok_or_else(|| backend("transaction sequence overflow"))?;
        if batch.seq != next || batch.ops.is_empty() {
            return Err(backend(format!(
                "sync batch sequence {} is not the next commit {next}",
                batch.seq
            )));
        }
        let ops = batch
            .ops
            .iter()
            .cloned()
            .map(SyncOp::into_op)
            .collect::<Vec<_>>();
        scratch = stage_ops(scratch, &ops)?;
        expect = next;
        prepared.push(PreparedBatch {
            seq: batch.seq,
            ops,
        });
    }
    Ok((scratch, prepared))
}

fn publish_sync_batches(
    inner: &Inner,
    batches: &[SyncBatch],
) -> Result<(u64, Vec<PreparedBatch>), StoreError> {
    match inner {
        Inner::Remote(_) => Err(backend(
            "a remote FerroStore does not publish sync batches locally",
        )),
        Inner::Memory(slot) => {
            let mut state = lock_mutex(slot);
            let (next, prepared) = prepare_sync_batches(&state, batches)?;
            let seq = next.last_seq;
            *state = next;
            Ok((seq, prepared))
        }
        Inner::Disk(disk) => {
            let file = lock_mutex(&disk.write);
            let _locked = FlockGuard::acquire(file.as_raw_fd())?;
            let (index, sb, state) = read_consistent(&file)?;
            let (next, prepared) = prepare_sync_batches(&state, batches)?;
            let mut frames = Vec::with_capacity(prepared.len());
            for batch in &prepared {
                frames.push(encode_frame(batch.seq, &batch.ops)?);
            }
            publish_frames(&file, index, &sb, &frames)?;
            Ok((next.last_seq, prepared))
        }
    }
}

fn hybrid_params(
    doc_collection: &str,
    vector_collection: &str,
    query: &str,
    values: &[f32],
    limit: usize,
    fusion: Fusion,
) -> Result<Value, StoreError> {
    vector::validate_fusion(fusion)?;
    let fusion = match fusion {
        Fusion::Rrf { k } => json!({"method": "rrf", "k": k}),
        Fusion::Weighted { bm25, vector } => {
            json!({
                "method": "weighted",
                "bm25": bm25,
                "vector": vector,
            })
        }
    };
    Ok(json!({
        "doc_collection": doc_collection,
        "vector_collection": vector_collection,
        "query": query,
        "values": vector::values_json(values)?,
        "limit": limit,
        "fusion": fusion,
    }))
}

/// `dir/store.fls` -> `dir/store.fls.sock`.
/// A path with no file name (`/` or `..`) -> that path joined with `ferrostore.sock`.
pub fn default_socket_path_for_store(path: impl AsRef<Path>) -> PathBuf {
    let path = path.as_ref();
    match path.file_name() {
        Some(name) => {
            let mut sock_name = name.to_os_string();
            sock_name.push(".sock");
            path.with_file_name(sock_name)
        }
        None => path.join("ferrostore.sock"),
    }
}

fn refused_connect(err: &io::Error) -> bool {
    matches!(
        err.kind(),
        io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
    )
}

fn file_uri_path(rest: &str) -> String {
    if rest.starts_with('/') {
        return rest.to_string();
    }
    if let Some(path) = rest.strip_prefix("localhost/") {
        return format!("/{path}");
    }
    if rest == "localhost" {
        return "/".to_string();
    }
    rest.to_string()
}

fn commit_if_seq_disk(disk: &Disk, expected_seq: u64, ops: &[Op]) -> Result<SeqCommit, StoreError> {
    let file = lock_mutex(&disk.write);
    let _locked = FlockGuard::acquire(file.as_raw_fd())?;
    let (index, sb, state) = read_consistent(&file)?;
    if state.last_seq != expected_seq {
        return Ok(SeqCommit::Lost);
    }
    if ops.is_empty() {
        return Ok(SeqCommit::Same);
    }
    let seq = stage_ops(state, ops)?.last_seq;
    let frame = encode_frame(seq, ops)?;
    append_frame(&file, index, &sb, &frame)?;
    Ok(SeqCommit::Wrote(Committed {
        seq,
        ops: ops.to_vec(),
    }))
}

fn with_write_disk<T>(
    disk: &Disk,
    mut body: impl FnMut(&mut Txn) -> Result<T, StoreError>,
) -> Result<(T, Option<Committed>), StoreError> {
    let file = lock_mutex(&disk.write);
    let _locked = FlockGuard::acquire(file.as_raw_fd())?;
    let (index, sb, state) = read_consistent(&file)?;
    let baseline = state.clone();
    let mut txn = Txn {
        docs: state.docs,
        tables: state.tables,
        vectors: state.vectors,
        ops: Vec::new(),
    };
    let result = body(&mut txn)?;
    if txn.ops.is_empty() {
        return Ok((result, None));
    }
    let seq = stage_ops(baseline, &txn.ops)?.last_seq;
    let frame = encode_frame(seq, &txn.ops)?;
    append_frame(&file, index, &sb, &frame)?;
    Ok((result, Some(Committed { seq, ops: txn.ops })))
}

fn commit_disk<T>(
    disk: &Disk,
    build: impl FnOnce(&DocState) -> Result<(Op, T), StoreError>,
) -> Result<(T, Committed), StoreError> {
    let file = lock_mutex(&disk.write);
    let _locked = FlockGuard::acquire(file.as_raw_fd())?;
    let (index, sb, state) = read_consistent(&file)?;
    let (op, extra) = build(&state)?;
    let seq = stage_ops(state, std::slice::from_ref(&op))?.last_seq;
    let frame = encode_frame(seq, std::slice::from_ref(&op))?;
    append_frame(&file, index, &sb, &frame)?;
    Ok((extra, Committed { seq, ops: vec![op] }))
}

/// Write every frame at the current WAL end, then publish one superblock.
/// A failure before `write_slot` leaves the previous superblock current.
fn publish_frames(
    file: &File,
    index: usize,
    sb: &Superblock,
    frames: &[Vec<u8>],
) -> Result<(), StoreError> {
    if frames.is_empty() {
        return Ok(());
    }
    let mut bytes = Vec::new();
    for frame in frames {
        bytes.extend_from_slice(frame);
    }
    let added = u64::try_from(bytes.len()).map_err(|_| backend("wal frame exceeds u64"))?;
    let at = sb
        .wal_offset
        .checked_add(sb.wal_length)
        .ok_or_else(|| backend("wal offset overflow"))?;
    if at.checked_add(added).is_none() {
        return Err(backend("wal offset overflow"));
    }
    let wal_length = sb
        .wal_length
        .checked_add(added)
        .ok_or_else(|| backend("wal length overflow"))?;
    let epoch = sb
        .epoch
        .checked_add(1)
        .ok_or_else(|| backend("epoch overflow"))?;
    file.write_all_at(&bytes, at).map_err(io_err)?;
    file.sync_all().map_err(io_err)?;
    let new_sb = Superblock {
        epoch,
        snapshot_offset: sb.snapshot_offset,
        snapshot_length: sb.snapshot_length,
        wal_offset: sb.wal_offset,
        wal_length,
        snapshot_seq: sb.snapshot_seq,
    };
    write_slot(file, index ^ 1, &new_sb)?;
    file.sync_all().map_err(io_err)?;
    Ok(())
}

fn checkpoint_disk(disk: &Disk) -> Result<(), StoreError> {
    let file = lock_mutex(&disk.write);
    let _locked = FlockGuard::acquire(file.as_raw_fd())?;
    let (index, sb, state) = read_consistent(&file)?;
    let blob = if !state.vectors.is_empty() {
        encode_snapshot_v3(state.last_seq, &state.docs, &state.tables, &state.vectors)?
    } else if state.tables.is_empty() {
        encode_snapshot(state.last_seq, &state.docs)?
    } else {
        encode_snapshot_v2(state.last_seq, &state.docs, &state.tables)?
    };
    let blob_len = u64::try_from(blob.len()).map_err(|_| backend("snapshot exceeds u64"))?;
    let offset = allocate_extent(&file, &sb, blob_len)?;
    file.write_all_at(&blob, offset).map_err(io_err)?;
    file.sync_all().map_err(io_err)?;
    let new_sb = Superblock {
        epoch: sb
            .epoch
            .checked_add(1)
            .ok_or_else(|| backend("epoch overflow"))?,
        snapshot_offset: offset,
        snapshot_length: blob_len,
        wal_offset: offset
            .checked_add(blob_len)
            .ok_or_else(|| backend("wal offset overflow"))?,
        wal_length: 0,
        snapshot_seq: state.last_seq,
    };
    write_slot(&file, index ^ 1, &new_sb)?;
    file.sync_all().map_err(io_err)?;
    Ok(())
}

fn allocate_extent(file: &File, sb: &Superblock, len: u64) -> Result<u64, StoreError> {
    let file_len = file.metadata().map_err(io_err)?.len();
    let live_end = sb
        .snapshot_offset
        .saturating_add(sb.snapshot_length)
        .max(sb.wal_offset.saturating_add(sb.wal_length))
        .max(HEADER_BYTES);
    let offset = file_len.max(live_end);
    if offset.checked_add(len).is_none() {
        return Err(backend("extent offset overflow"));
    }
    Ok(offset)
}

fn append_frame(
    file: &File,
    index: usize,
    sb: &Superblock,
    frame: &[u8],
) -> Result<Superblock, StoreError> {
    let frame_len = u64::try_from(frame.len()).map_err(|_| backend("wal frame exceeds u64"))?;
    let at = sb
        .wal_offset
        .checked_add(sb.wal_length)
        .ok_or_else(|| backend("wal offset overflow"))?;
    file.write_all_at(frame, at).map_err(io_err)?;
    file.sync_all().map_err(io_err)?;
    let new_sb = Superblock {
        epoch: sb
            .epoch
            .checked_add(1)
            .ok_or_else(|| backend("epoch overflow"))?,
        snapshot_offset: sb.snapshot_offset,
        snapshot_length: sb.snapshot_length,
        wal_offset: sb.wal_offset,
        wal_length: sb
            .wal_length
            .checked_add(frame_len)
            .ok_or_else(|| backend("wal length overflow"))?,
        snapshot_seq: sb.snapshot_seq,
    };
    write_slot(file, index ^ 1, &new_sb)?;
    file.sync_all().map_err(io_err)?;
    Ok(new_sb)
}

fn init_empty(file: &File) -> Result<(), StoreError> {
    file.set_len(HEADER_BYTES).map_err(io_err)?;
    let sb = Superblock {
        epoch: 1,
        snapshot_offset: HEADER_BYTES,
        snapshot_length: 0,
        wal_offset: HEADER_BYTES,
        wal_length: 0,
        snapshot_seq: 0,
    };
    write_slot(file, 0, &sb)?;
    file.sync_all().map_err(io_err)?;
    Ok(())
}

// No valid slot yet. Publish slot 0 only for an empty file, or when every byte
// already in a file no longer than the header is zero. A non-zero short file
// is not rewritten, and neither is a longer file.
fn ensure_superblock(file: &File) -> Result<(), StoreError> {
    if parsed_superblock(file)?.is_some() {
        return Ok(());
    }
    let _locked = FlockGuard::acquire(file.as_raw_fd())?;
    if parsed_superblock(file)?.is_some() {
        return Ok(());
    }
    let len = file.metadata().map_err(io_err)?.len();
    if len > HEADER_BYTES {
        return Ok(());
    }
    if len == 0 {
        return init_empty(file);
    }
    let n = usize::try_from(len).map_err(|_| backend("short file length exceeds usize"))?;
    let mut buf = vec![0u8; n];
    file.read_exact_at(&mut buf, 0).map_err(io_err)?;
    if buf.iter().any(|byte| *byte != 0) {
        return Ok(());
    }
    init_empty(file)
}

fn read_consistent(file: &File) -> Result<(usize, Superblock, DocState), StoreError> {
    let (index, sb) = latest_superblock(file)?;
    let mut state = if sb.snapshot_length == 0 {
        DocState::at_seq(sb.snapshot_seq)
    } else {
        let bytes = read_range(file, sb.snapshot_offset, sb.snapshot_length)?;
        let snap = decode_snapshot(&bytes)?;
        if snap.seq != sb.snapshot_seq {
            return Err(corrupt("snapshot sequence does not match the superblock"));
        }
        DocState {
            docs: snap.docs,
            tables: snap.tables,
            vectors: snap.vectors,
            last_seq: snap.seq,
        }
    };
    let wal = read_range(file, sb.wal_offset, sb.wal_length)?;
    let frames = decode_wal(&wal)?;
    apply_frames(&mut state, sb.snapshot_seq, &frames)?;
    Ok((index, sb, state))
}

fn parsed_superblock(file: &File) -> Result<Option<(usize, Superblock)>, StoreError> {
    let file_len = file.metadata().map_err(io_err)?.len();
    let mut best: Option<(usize, Superblock)> = None;
    for index in 0..2 {
        let offset = slot_offset(index);
        if file_len < offset + SLOT_LEN as u64 {
            continue;
        }
        let mut buf = [0u8; SLOT_LEN];
        file.read_exact_at(&mut buf, offset).map_err(io_err)?;
        if let Some(sb) = parse_superblock(&buf) {
            match &best {
                Some((_, current)) if sb.epoch < current.epoch => {}
                Some((_, current)) if sb.epoch == current.epoch => {
                    return Err(corrupt("two valid superblocks share an epoch"));
                }
                _ => best = Some((index, sb)),
            }
        }
    }
    Ok(best)
}

fn latest_superblock(file: &File) -> Result<(usize, Superblock), StoreError> {
    parsed_superblock(file)?.ok_or_else(|| backend("no valid FerroStore superblock"))
}

fn write_slot(file: &File, index: usize, sb: &Superblock) -> Result<(), StoreError> {
    if index > 1 {
        return Err(backend("superblock slot out of range"));
    }
    let bytes = encode_superblock(sb);
    file.write_all_at(&bytes, slot_offset(index))
        .map_err(io_err)?;
    Ok(())
}

fn read_range(file: &File, offset: u64, length: u64) -> Result<Vec<u8>, StoreError> {
    if length == 0 {
        return Ok(Vec::new());
    }
    let len =
        usize::try_from(length).map_err(|_| corrupt("extent length does not fit in memory"))?;
    let file_len = file.metadata().map_err(io_err)?.len();
    let end = offset
        .checked_add(length)
        .ok_or_else(|| corrupt("extent range overflows"))?;
    if end > file_len {
        return Err(corrupt("extent extends past the end of the file"));
    }
    let mut buf = vec![0u8; len];
    file.read_exact_at(&mut buf, offset).map_err(io_err)?;
    Ok(buf)
}

fn slot_offset(index: usize) -> u64 {
    index as u64 * SLOT_LEN as u64
}

fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn wait_mutex<'a, T>(condvar: &Condvar, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    match condvar.wait(guard) {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn io_err(err: io::Error) -> StoreError {
    backend(err)
}
