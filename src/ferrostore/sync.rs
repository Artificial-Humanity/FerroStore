//! Process-local sync socket. Nothing here is written to the `.fls` file.

use super::columnar::{self, ColumnDef};
use super::layout::{Op, backend};
use super::vector::{self, Fusion, VectorHit, VectorMetric, VectorStorage};
use fl_core::StoreError;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

/// One committed mutation, in the shape of an [`Op`] a commit carries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum SyncOp {
    #[serde(rename = "put")]
    PutDoc {
        collection: String,
        id: String,
        doc: Value,
    },
    #[serde(rename = "delete")]
    DeleteDoc {
        collection: String,
        id: String,
    },
    CreateColumnTable {
        name: String,
        #[serde(serialize_with = "ser_columns", deserialize_with = "de_columns")]
        columns: Vec<ColumnDef>,
    },
    AppendColumnGroup {
        name: String,
        bytes: Vec<u8>,
    },
    CreateVectorCollection {
        name: String,
        dim: u32,
        #[serde(serialize_with = "ser_storage", deserialize_with = "de_storage")]
        storage: VectorStorage,
        #[serde(serialize_with = "ser_metric", deserialize_with = "de_metric")]
        metric: VectorMetric,
        m: u32,
        ef_construction: u32,
        ef_search: u32,
    },
    UpsertVector {
        collection: String,
        id: String,
        bytes: Vec<u8>,
        /// Absent is `None`. JSON null is `Some(Null)`, not a missing field.
        #[serde(
            default,
            skip_serializing_if = "Option::is_none",
            deserialize_with = "de_metadata"
        )]
        metadata: Option<Value>,
    },
    DeleteVector {
        collection: String,
        id: String,
    },
}

impl SyncOp {
    /// Lossless copy of a committed [`Op`].
    pub fn from_op(op: &Op) -> Self {
        match op {
            Op::Put {
                collection,
                id,
                doc,
            } => Self::PutDoc {
                collection: collection.clone(),
                id: id.clone(),
                doc: doc.clone(),
            },
            Op::Delete { collection, id } => Self::DeleteDoc {
                collection: collection.clone(),
                id: id.clone(),
            },
            Op::CreateColumnTable { name, columns } => Self::CreateColumnTable {
                name: name.clone(),
                columns: columns.clone(),
            },
            Op::AppendColumnGroup { name, bytes } => Self::AppendColumnGroup {
                name: name.clone(),
                bytes: bytes.clone(),
            },
            Op::CreateVectorCollection {
                name,
                dim,
                storage,
                metric,
                m,
                ef_construction,
                ef_search,
            } => Self::CreateVectorCollection {
                name: name.clone(),
                dim: *dim,
                storage: *storage,
                metric: *metric,
                m: *m,
                ef_construction: *ef_construction,
                ef_search: *ef_search,
            },
            Op::UpsertVector {
                collection,
                id,
                bytes,
                metadata,
            } => Self::UpsertVector {
                collection: collection.clone(),
                id: id.clone(),
                bytes: bytes.clone(),
                metadata: metadata.clone(),
            },
            Op::DeleteVector { collection, id } => Self::DeleteVector {
                collection: collection.clone(),
                id: id.clone(),
            },
        }
    }

    /// Inverse of [`Self::from_op`].
    pub fn into_op(self) -> Op {
        match self {
            Self::PutDoc {
                collection,
                id,
                doc,
            } => Op::Put {
                collection,
                id,
                doc,
            },
            Self::DeleteDoc { collection, id } => Op::Delete { collection, id },
            Self::CreateColumnTable { name, columns } => Op::CreateColumnTable { name, columns },
            Self::AppendColumnGroup { name, bytes } => Op::AppendColumnGroup { name, bytes },
            Self::CreateVectorCollection {
                name,
                dim,
                storage,
                metric,
                m,
                ef_construction,
                ef_search,
            } => Op::CreateVectorCollection {
                name,
                dim,
                storage,
                metric,
                m,
                ef_construction,
                ef_search,
            },
            Self::UpsertVector {
                collection,
                id,
                bytes,
                metadata,
            } => Op::UpsertVector {
                collection,
                id,
                bytes,
                metadata,
            },
            Self::DeleteVector { collection, id } => Op::DeleteVector { collection, id },
        }
    }
}

/// One committed sequence and the ops that advanced it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyncBatch {
    pub seq: u64,
    pub ops: Vec<SyncOp>,
}

/// When the store calls [`SyncAdapter::on_commit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMode {
    /// Call `on_commit` after each commit. A failure stays pending.
    WriteThrough,
    /// Queue commits until [`super::FerroStore::sync_flush`].
    WriteBehind,
    /// Queue commits, and fill a document miss from `fetch_document`.
    LocalCache,
    /// Queue commits, fill misses, and accept `sync_prewarm` batches.
    CloudPreCache,
}

/// Cloud or peer replication. The store does not link a cloud SDK.
pub trait SyncAdapter: Send + Sync {
    fn name(&self) -> &str;

    /// Receive one committed batch. In `WriteThrough` the commit is durable
    /// before this runs, and the committer waits for it: an `Err` makes the
    /// write return [`StoreError::Undelivered`] and leaves the batch pending.
    ///
    /// ⚠ An `on_commit` that blocks waiting for a commit made on another
    /// thread through the same handle deadlocks under `WriteThrough`: that
    /// commit waits for this call to return before its own batch can be
    /// sent. For example, an HTTP adapter pointed back at a server that
    /// writes to the same handle. A commit made on this call's own thread
    /// does not wait: it stays queued.
    fn on_commit(&self, batch: &SyncBatch) -> Result<(), StoreError>;

    fn pull_since(&self, _after_seq: u64) -> Result<Vec<SyncBatch>, StoreError> {
        Ok(Vec::new())
    }

    fn fetch_document(&self, _: &str, _: &str) -> Result<Option<Value>, StoreError> {
        Ok(None)
    }

    fn prewarm(&self, _: &[&str]) -> Result<Vec<SyncBatch>, StoreError> {
        Ok(Vec::new())
    }

    /// Remote vector hits. `Ok(None)` means the store should search locally.
    fn search_vectors(
        &self,
        _collection: &str,
        _query: &[f32],
        _k: usize,
        _ef: Option<u32>,
    ) -> Result<Option<Vec<VectorHit>>, StoreError> {
        Ok(None)
    }

    /// Remote hybrid hits. `Ok(None)` means the store should search locally.
    fn search_hybrid(
        &self,
        _doc_collection: &str,
        _vector_collection: &str,
        _query_text: &str,
        _query_vector: &[f32],
        _k: usize,
        _fusion: Fusion,
    ) -> Result<Option<Vec<VectorHit>>, StoreError> {
        Ok(None)
    }
}

fn ser_columns<S: Serializer>(columns: &[ColumnDef], serializer: S) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeSeq;
    let mut seq = serializer.serialize_seq(Some(columns.len()))?;
    for column in columns {
        seq.serialize_element(&columnar::column_def_json(column))?;
    }
    seq.end()
}

fn de_columns<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<ColumnDef>, D::Error> {
    let value = Value::deserialize(deserializer)?;
    columnar::parse_columns(&value).map_err(D::Error::custom)
}

fn ser_storage<S: Serializer>(storage: &VectorStorage, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(vector::storage_name(*storage))
}

fn de_storage<'de, D: Deserializer<'de>>(deserializer: D) -> Result<VectorStorage, D::Error> {
    let name = String::deserialize(deserializer)?;
    vector::storage_from_name(&name).map_err(D::Error::custom)
}

fn ser_metric<S: Serializer>(metric: &VectorMetric, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_str(vector::metric_name(*metric))
}

fn de_metric<'de, D: Deserializer<'de>>(deserializer: D) -> Result<VectorMetric, D::Error> {
    let name = String::deserialize(deserializer)?;
    vector::metric_from_name(&name).map_err(D::Error::custom)
}

/// JSON null stays `Some(Null)`. A missing field uses `#[serde(default)]` and
/// does not call this function.
fn de_metadata<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Value>, D::Error> {
    Ok(Some(Value::deserialize(deserializer)?))
}

const ENVELOPE_FRAME_VERSION: u16 = 1;
const ENVELOPE_SCHEMA_VERSION: u16 = 1;

/// One replicated record. The tag is the on-wire kind byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvelopeKind {
    Document = 0,
    DeleteDocument = 1,
    ColumnTableSchema = 2,
    ColumnRowGroup = 3,
    VectorCollectionSchema = 4,
    VectorUpsert = 5,
    VectorDelete = 6,
}

impl EnvelopeKind {
    fn tag(self) -> u8 {
        self as u8
    }

    fn from_tag(tag: u8) -> Result<Self, StoreError> {
        match tag {
            0 => Ok(Self::Document),
            1 => Ok(Self::DeleteDocument),
            2 => Ok(Self::ColumnTableSchema),
            3 => Ok(Self::ColumnRowGroup),
            4 => Ok(Self::VectorCollectionSchema),
            5 => Ok(Self::VectorUpsert),
            6 => Ok(Self::VectorDelete),
            other => Err(backend(format!("unknown record envelope kind {other}"))),
        }
    }
}

/// Lossless copy of one [`SyncOp`] inside a [`SyncBatch`], plus optional vector floats.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordEnvelope {
    pub seq: u64,
    pub op_index: u32,
    pub kind: EnvelopeKind,
    pub namespace: String,
    pub record_id: String,
    pub schema_version: u16,
    pub payload_bytes: Vec<u8>,
    pub metadata: Option<Value>,
    pub vector_f32: Option<Vec<f32>>,
}

impl SyncBatch {
    /// One envelope per op. `schema_version` is 1. `op_index` is the index in `ops`.
    pub fn to_envelopes(&self) -> Result<Vec<RecordEnvelope>, StoreError> {
        let mut envelopes = Vec::with_capacity(self.ops.len());
        for (index, op) in self.ops.iter().enumerate() {
            let op_index = u32::try_from(index)
                .map_err(|_| backend(format!("sync op index {index} does not fit in u32")))?;
            envelopes.push(RecordEnvelope::from_op(self.seq, op_index, op)?);
        }
        Ok(envelopes)
    }

    /// Rebuild a batch. Every `seq` must match, and `op_index` must be exactly `0..len`.
    pub fn from_envelopes(seq: u64, envelopes: &[RecordEnvelope]) -> Result<Self, StoreError> {
        if envelopes.is_empty() {
            return Ok(Self {
                seq,
                ops: Vec::new(),
            });
        }
        let mut ordered = vec![None; envelopes.len()];
        for envelope in envelopes {
            if envelope.seq != seq {
                return Err(backend(format!(
                    "record envelope seq {} does not match batch seq {seq}",
                    envelope.seq
                )));
            }
            let index = usize::try_from(envelope.op_index)
                .map_err(|_| backend("record envelope op_index does not fit"))?;
            if index >= ordered.len() || ordered[index].is_some() {
                return Err(backend(format!(
                    "record envelope op_index {index} is not the dense range 0..{}",
                    ordered.len()
                )));
            }
            ordered[index] = Some(envelope);
        }
        let mut ops = Vec::with_capacity(envelopes.len());
        for (index, slot) in ordered.into_iter().enumerate() {
            let Some(envelope) = slot else {
                return Err(backend(format!("record envelope op_index gap at {index}")));
            };
            ops.push(sync_op_from_envelope(envelope)?);
        }
        Ok(Self { seq, ops })
    }
}

impl RecordEnvelope {
    fn from_op(seq: u64, op_index: u32, op: &SyncOp) -> Result<Self, StoreError> {
        let mut envelope = Self {
            seq,
            op_index,
            kind: EnvelopeKind::Document,
            namespace: String::new(),
            record_id: String::new(),
            schema_version: ENVELOPE_SCHEMA_VERSION,
            payload_bytes: Vec::new(),
            metadata: None,
            vector_f32: None,
        };
        match op {
            SyncOp::PutDoc {
                collection,
                id,
                doc,
            } => {
                envelope.kind = EnvelopeKind::Document;
                envelope.namespace = collection.clone();
                envelope.record_id = id.clone();
                envelope.payload_bytes = json_bytes(doc)?;
            }
            SyncOp::DeleteDoc { collection, id } => {
                envelope.kind = EnvelopeKind::DeleteDocument;
                envelope.namespace = collection.clone();
                envelope.record_id = id.clone();
            }
            SyncOp::CreateColumnTable { name, columns } => {
                envelope.kind = EnvelopeKind::ColumnTableSchema;
                envelope.namespace = name.clone();
                let columns: Vec<Value> = columns.iter().map(columnar::column_def_json).collect();
                envelope.payload_bytes = json_bytes(&Value::Array(columns))?;
            }
            SyncOp::AppendColumnGroup { name, bytes } => {
                envelope.kind = EnvelopeKind::ColumnRowGroup;
                envelope.namespace = name.clone();
                envelope.payload_bytes = bytes.clone();
            }
            SyncOp::CreateVectorCollection {
                name,
                dim,
                storage,
                metric,
                m,
                ef_construction,
                ef_search,
            } => {
                envelope.kind = EnvelopeKind::VectorCollectionSchema;
                envelope.namespace = name.clone();
                envelope.payload_bytes = json_bytes(&serde_json::json!({
                    "dim": dim,
                    "storage": vector::storage_name(*storage),
                    "metric": vector::metric_name(*metric),
                    "m": m,
                    "ef_construction": ef_construction,
                    "ef_search": ef_search,
                }))?;
            }
            SyncOp::UpsertVector {
                collection,
                id,
                bytes,
                metadata,
            } => {
                envelope.kind = EnvelopeKind::VectorUpsert;
                envelope.namespace = collection.clone();
                envelope.record_id = id.clone();
                envelope.payload_bytes = bytes.clone();
                envelope.metadata = metadata.clone();
            }
            SyncOp::DeleteVector { collection, id } => {
                envelope.kind = EnvelopeKind::VectorDelete;
                envelope.namespace = collection.clone();
                envelope.record_id = id.clone();
            }
        }
        Ok(envelope)
    }

    /// Little-endian `FLEN` frame. No trailing bytes.
    ///
    /// Every variable field carries a `u32` length prefix, so a field of 4 GiB or
    /// more is an `Err`, never a panic.
    pub fn encode(&self) -> Result<Vec<u8>, StoreError> {
        let mut out = Vec::new();
        out.extend_from_slice(b"FLEN");
        out.extend_from_slice(&ENVELOPE_FRAME_VERSION.to_le_bytes());
        out.extend_from_slice(&self.seq.to_le_bytes());
        out.extend_from_slice(&self.op_index.to_le_bytes());
        out.push(self.kind.tag());
        out.extend_from_slice(&self.schema_version.to_le_bytes());
        push_counted(&mut out, self.namespace.as_bytes())?;
        push_counted(&mut out, self.record_id.as_bytes())?;
        push_counted(&mut out, &self.payload_bytes)?;
        match &self.metadata {
            None => out.push(0),
            Some(value) => {
                out.push(1);
                push_counted(&mut out, &json_bytes(value)?)?;
            }
        }
        match &self.vector_f32 {
            None => out.push(0),
            Some(values) => {
                out.push(1);
                let count = u32::try_from(values.len()).map_err(|_| {
                    backend("record envelope vector float count does not fit in u32")
                })?;
                out.extend_from_slice(&count.to_le_bytes());
                for value in values {
                    out.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        Ok(out)
    }

    /// Inverse of [`Self::encode`]. Truncated, corrupt or over-long input is an
    /// `Err`. A length prefix never allocates more than the bytes that remain.
    pub fn decode(bytes: &[u8]) -> Result<Self, StoreError> {
        let mut frame = Frame::new(bytes);
        let magic = frame.take(4)?;
        if magic != b"FLEN" {
            return Err(backend("record envelope magic is not FLEN"));
        }
        let version = frame.u16()?;
        if version != ENVELOPE_FRAME_VERSION {
            return Err(backend(format!(
                "unknown record envelope version {version}"
            )));
        }
        let seq = frame.u64()?;
        let op_index = frame.u32()?;
        let kind = EnvelopeKind::from_tag(frame.u8()?)?;
        let schema_version = frame.u16()?;
        let namespace = frame.string()?;
        let record_id = frame.string()?;
        let payload_bytes = frame.counted()?;
        let metadata = match frame.u8()? {
            0 => None,
            1 => {
                let raw = frame.counted()?;
                Some(serde_json::from_slice(&raw).map_err(|err| {
                    backend(format!("record envelope metadata is not json: {err}"))
                })?)
            }
            flag => {
                return Err(backend(format!(
                    "unknown record envelope metadata flag {flag}"
                )));
            }
        };
        let vector_f32 = match frame.u8()? {
            0 => None,
            1 => {
                let count = usize_len(frame.u32()?)?;
                let byte_len = count
                    .checked_mul(4)
                    .ok_or_else(|| backend("truncated record envelope"))?;
                let raw = frame.take(byte_len)?;
                let (chunks, _) = raw.as_chunks::<4>();
                Some(
                    chunks
                        .iter()
                        .map(|chunk| f32::from_le_bytes(*chunk))
                        .collect(),
                )
            }
            flag => {
                return Err(backend(format!(
                    "unknown record envelope vector flag {flag}"
                )));
            }
        };
        frame.finish()?;
        Ok(Self {
            seq,
            op_index,
            kind,
            namespace,
            record_id,
            schema_version,
            payload_bytes,
            metadata,
            vector_f32,
        })
    }
}

fn sync_op_from_envelope(envelope: &RecordEnvelope) -> Result<SyncOp, StoreError> {
    if envelope.schema_version != ENVELOPE_SCHEMA_VERSION {
        return Err(backend(format!(
            "unsupported record envelope schema version {}",
            envelope.schema_version
        )));
    }
    match envelope.kind {
        EnvelopeKind::Document => Ok(SyncOp::PutDoc {
            collection: envelope.namespace.clone(),
            id: envelope.record_id.clone(),
            doc: json_value(&envelope.payload_bytes, "document")?,
        }),
        EnvelopeKind::DeleteDocument => Ok(SyncOp::DeleteDoc {
            collection: envelope.namespace.clone(),
            id: envelope.record_id.clone(),
        }),
        EnvelopeKind::ColumnTableSchema => {
            let value = json_value(&envelope.payload_bytes, "columns")?;
            Ok(SyncOp::CreateColumnTable {
                name: envelope.namespace.clone(),
                columns: columnar::parse_columns(&value)?,
            })
        }
        EnvelopeKind::ColumnRowGroup => Ok(SyncOp::AppendColumnGroup {
            name: envelope.namespace.clone(),
            bytes: envelope.payload_bytes.clone(),
        }),
        EnvelopeKind::VectorCollectionSchema => {
            let value = json_value(&envelope.payload_bytes, "vector collection")?;
            let obj = value
                .as_object()
                .ok_or_else(|| backend("vector collection schema is not an object"))?;
            Ok(SyncOp::CreateVectorCollection {
                name: envelope.namespace.clone(),
                dim: json_u32(obj, "dim")?,
                storage: vector::storage_from_name(json_str(obj, "storage")?)?,
                metric: vector::metric_from_name(json_str(obj, "metric")?)?,
                m: json_u32(obj, "m")?,
                ef_construction: json_u32(obj, "ef_construction")?,
                ef_search: json_u32(obj, "ef_search")?,
            })
        }
        EnvelopeKind::VectorUpsert => Ok(SyncOp::UpsertVector {
            collection: envelope.namespace.clone(),
            id: envelope.record_id.clone(),
            bytes: upsert_bytes(envelope)?,
            metadata: envelope.metadata.clone(),
        }),
        EnvelopeKind::VectorDelete => Ok(SyncOp::DeleteVector {
            collection: envelope.namespace.clone(),
            id: envelope.record_id.clone(),
        }),
    }
}

fn upsert_bytes(envelope: &RecordEnvelope) -> Result<Vec<u8>, StoreError> {
    if !envelope.payload_bytes.is_empty() {
        return Ok(envelope.payload_bytes.clone());
    }
    let Some(values) = &envelope.vector_f32 else {
        return Err(backend("vector upsert envelope has no payload"));
    };
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(4));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

fn json_bytes(value: &Value) -> Result<Vec<u8>, StoreError> {
    serde_json::to_vec(value).map_err(|err| backend(format!("record envelope json: {err}")))
}

fn json_value(bytes: &[u8], what: &str) -> Result<Value, StoreError> {
    serde_json::from_slice(bytes)
        .map_err(|err| backend(format!("record envelope {what} is not json: {err}")))
}

fn json_u32(obj: &serde_json::Map<String, Value>, field: &str) -> Result<u32, StoreError> {
    let number = obj
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| backend(format!("vector collection schema is missing {field}")))?;
    u32::try_from(number).map_err(|_| {
        backend(format!(
            "vector collection schema {field} does not fit in u32"
        ))
    })
}

fn json_str<'a>(
    obj: &'a serde_json::Map<String, Value>,
    field: &str,
) -> Result<&'a str, StoreError> {
    obj.get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| backend(format!("vector collection schema is missing {field}")))
}

fn push_counted(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), StoreError> {
    let len = u32::try_from(bytes.len())
        .map_err(|_| backend("record envelope field does not fit a u32 length"))?;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

fn usize_len(len: u32) -> Result<usize, StoreError> {
    usize::try_from(len).map_err(|_| backend("record envelope length does not fit"))
}

struct Frame<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Frame<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], StoreError> {
        let Some(end) = self.pos.checked_add(n) else {
            return Err(backend("truncated record envelope"));
        };
        if end > self.buf.len() {
            return Err(backend("truncated record envelope"));
        }
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, StoreError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, StoreError> {
        let raw = self.take(2)?;
        Ok(u16::from_le_bytes([raw[0], raw[1]]))
    }

    fn u32(&mut self) -> Result<u32, StoreError> {
        let raw = self.take(4)?;
        Ok(u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
    }

    fn u64(&mut self) -> Result<u64, StoreError> {
        let raw = self.take(8)?;
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(raw);
        Ok(u64::from_le_bytes(bytes))
    }

    fn counted(&mut self) -> Result<Vec<u8>, StoreError> {
        let len = usize_len(self.u32()?)?;
        Ok(self.take(len)?.to_vec())
    }

    fn string(&mut self) -> Result<String, StoreError> {
        let bytes = self.counted()?;
        String::from_utf8(bytes).map_err(|_| backend("record envelope string is not utf-8"))
    }

    fn finish(self) -> Result<(), StoreError> {
        if self.pos != self.buf.len() {
            return Err(backend("trailing bytes after record envelope"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ferrostore::{FerroStore, FerroStoreServer};
    use serde_json::json;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex};
    use std::thread;

    /// Records the largest single allocation per thread, so a test can prove a
    /// corrupt length prefix did not drive an allocation.
    struct TrackingAlloc;

    thread_local! {
        static LARGEST_ALLOC: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    // SAFETY: every call forwards to the system allocator unchanged.
    unsafe impl std::alloc::GlobalAlloc for TrackingAlloc {
        unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
            let _ = LARGEST_ALLOC.try_with(|largest| largest.set(largest.get().max(layout.size())));
            unsafe { std::alloc::System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
            unsafe { std::alloc::System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new: usize) -> *mut u8 {
            let _ = LARGEST_ALLOC.try_with(|largest| largest.set(largest.get().max(new)));
            unsafe { std::alloc::System.realloc(ptr, layout, new) }
        }
    }

    #[global_allocator]
    static ALLOC: TrackingAlloc = TrackingAlloc;

    fn assert_roundtrip(op: Op) {
        let sync = SyncOp::from_op(&op);
        let value = serde_json::to_value(&sync).unwrap();
        let decoded: SyncOp = serde_json::from_value(value).unwrap();
        assert_eq!(decoded, sync);
        assert_eq!(SyncOp::from_op(&decoded.into_op()), sync);
    }

    #[test]
    fn sync_op_json_round_trip_covers_every_variant() {
        assert_roundtrip(Op::Put {
            collection: "docs".into(),
            id: "a".into(),
            doc: json!({"n": 1, "ok": true}),
        });
        assert_roundtrip(Op::Delete {
            collection: "docs".into(),
            id: "a".into(),
        });
        assert_roundtrip(Op::CreateColumnTable {
            name: "t".into(),
            columns: vec![
                ColumnDef {
                    name: "n".into(),
                    ty: columnar::ColumnType::Int64,
                },
                ColumnDef {
                    name: "s".into(),
                    ty: columnar::ColumnType::Utf8,
                },
            ],
        });
        assert_roundtrip(Op::AppendColumnGroup {
            name: "t".into(),
            bytes: vec![0, 255],
        });
        assert_roundtrip(Op::CreateVectorCollection {
            name: "v".into(),
            dim: 2,
            storage: VectorStorage::F16,
            metric: VectorMetric::DotProduct,
            m: 4,
            ef_construction: 8,
            ef_search: 3,
        });
        assert_roundtrip(Op::UpsertVector {
            collection: "v".into(),
            id: "a".into(),
            bytes: vec![0, 255],
            metadata: None,
        });
        assert_roundtrip(Op::UpsertVector {
            collection: "v".into(),
            id: "b".into(),
            bytes: vec![1, 2, 3],
            metadata: Some(Value::Null),
        });
        assert_roundtrip(Op::DeleteVector {
            collection: "v".into(),
            id: "a".into(),
        });

        let absent = SyncOp::from_op(&Op::UpsertVector {
            collection: "v".into(),
            id: "a".into(),
            bytes: vec![0],
            metadata: None,
        });
        let absent_json = serde_json::to_value(&absent).unwrap();
        assert_eq!(absent_json["op"], "upsert_vector");
        assert!(absent_json.get("metadata").is_none());
        assert_eq!(absent_json["bytes"], json!([0]));

        let null_meta = SyncOp::from_op(&Op::UpsertVector {
            collection: "v".into(),
            id: "b".into(),
            bytes: vec![9],
            metadata: Some(Value::Null),
        });
        let null_json = serde_json::to_value(&null_meta).unwrap();
        assert!(null_json["metadata"].is_null());
        let back: SyncOp = serde_json::from_value(null_json).unwrap();
        assert_eq!(back, null_meta);

        let table = SyncOp::from_op(&Op::CreateColumnTable {
            name: "t".into(),
            columns: vec![ColumnDef {
                name: "n".into(),
                ty: columnar::ColumnType::TimestampMs,
            }],
        });
        assert_eq!(
            serde_json::to_value(&table).unwrap(),
            json!({
                "op": "create_column_table",
                "name": "t",
                "columns": [{"name": "n", "ty": "timestamp_ms"}]
            })
        );
        let vector = SyncOp::from_op(&Op::CreateVectorCollection {
            name: "v".into(),
            dim: 2,
            storage: VectorStorage::F32,
            metric: VectorMetric::Cosine,
            m: 2,
            ef_construction: 4,
            ef_search: 1,
        });
        assert_eq!(
            serde_json::to_value(&vector).unwrap(),
            json!({
                "op": "create_vector_collection",
                "name": "v",
                "dim": 2,
                "storage": "f32",
                "metric": "cosine",
                "m": 2,
                "ef_construction": 4,
                "ef_search": 1
            })
        );
        let batch = SyncBatch {
            seq: 3,
            ops: vec![SyncOp::DeleteDoc {
                collection: "docs".into(),
                id: "a".into(),
            }],
        };
        assert_eq!(
            serde_json::to_value(&batch).unwrap(),
            json!({"seq": 3, "ops": [{"op": "delete", "collection": "docs", "id": "a"}]})
        );
        let put = SyncOp::PutDoc {
            collection: "docs".into(),
            id: "a".into(),
            doc: json!({"n": 1}),
        };
        assert_eq!(serde_json::to_value(&put).unwrap()["op"], "put");
    }

    struct FakeSync {
        commits: Mutex<Vec<SyncBatch>>,
        fail: AtomicBool,
        fetches: AtomicUsize,
        fetch: Mutex<Option<Value>>,
        pull: Mutex<Vec<SyncBatch>>,
        prewarm: Mutex<Vec<SyncBatch>>,
        pull_after: Mutex<Vec<u64>>,
    }

    impl FakeSync {
        fn new() -> Self {
            Self {
                commits: Mutex::new(Vec::new()),
                fail: AtomicBool::new(false),
                fetches: AtomicUsize::new(0),
                fetch: Mutex::new(None),
                pull: Mutex::new(Vec::new()),
                prewarm: Mutex::new(Vec::new()),
                pull_after: Mutex::new(Vec::new()),
            }
        }

        fn commits(&self) -> Vec<SyncBatch> {
            self.commits.lock().unwrap().clone()
        }
    }

    impl SyncAdapter for FakeSync {
        fn name(&self) -> &str {
            "fake"
        }

        fn on_commit(&self, batch: &SyncBatch) -> Result<(), StoreError> {
            if self.fail.load(Ordering::SeqCst) {
                return Err(StoreError::Backend(
                    "sync adapter rejected the batch".into(),
                ));
            }
            self.commits.lock().unwrap().push(batch.clone());
            Ok(())
        }

        fn pull_since(&self, after_seq: u64) -> Result<Vec<SyncBatch>, StoreError> {
            self.pull_after.lock().unwrap().push(after_seq);
            Ok(self.pull.lock().unwrap().clone())
        }

        fn fetch_document(&self, _: &str, _: &str) -> Result<Option<Value>, StoreError> {
            self.fetches.fetch_add(1, Ordering::SeqCst);
            Ok(self.fetch.lock().unwrap().clone())
        }

        fn prewarm(&self, _: &[&str]) -> Result<Vec<SyncBatch>, StoreError> {
            Ok(self.prewarm.lock().unwrap().clone())
        }
    }

    fn attach(mode: SyncMode) -> (FerroStore, Arc<FakeSync>) {
        let fake = Arc::new(FakeSync::new());
        let store = FerroStore::in_memory()
            .with_sync_adapter(mode, Arc::clone(&fake) as Arc<dyn SyncAdapter>)
            .unwrap();
        (store, fake)
    }

    #[test]
    fn write_through_notifies_each_commit() {
        let (store, fake) = attach(SyncMode::WriteThrough);
        store.put_doc("docs", "a", &json!({"n": 1})).unwrap();
        store.put_doc("docs", "b", &json!({"n": 2})).unwrap();
        let commits = fake.commits();
        assert_eq!(commits.len(), 2);
        assert_eq!(commits[0].seq, 1);
        assert_eq!(
            commits[0].ops,
            vec![SyncOp::PutDoc {
                collection: "docs".into(),
                id: "a".into(),
                doc: json!({"n": 1}),
            }]
        );
        assert_eq!(commits[1].seq, 2);
        assert_eq!(
            commits[1].ops,
            vec![SyncOp::PutDoc {
                collection: "docs".into(),
                id: "b".into(),
                doc: json!({"n": 2}),
            }]
        );
        assert_eq!(store.synced_seq(), 2);
        assert_eq!(store.sync_mode(), Some(SyncMode::WriteThrough));
    }

    #[test]
    fn write_behind_flushes_in_commit_order() {
        let (store, fake) = attach(SyncMode::WriteBehind);
        store.put_doc("docs", "a", &json!({"n": 1})).unwrap();
        store.put_doc("docs", "b", &json!({"n": 2})).unwrap();
        assert!(fake.commits().is_empty());
        assert_eq!(store.synced_seq(), 0);
        assert_eq!(store.sync_flush().unwrap(), 2);
        let commits = fake.commits();
        assert_eq!(
            commits.iter().map(|batch| batch.seq).collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(store.synced_seq(), 2);
        assert_eq!(store.sync_flush().unwrap(), 2);
        assert_eq!(fake.commits().len(), 2);
    }

    #[test]
    fn failed_write_through_stays_pending_until_flush() {
        let (store, fake) = attach(SyncMode::WriteThrough);
        fake.fail.store(true, Ordering::SeqCst);
        let err = store.put_doc("docs", "a", &json!({"n": 1})).unwrap_err();
        assert!(matches!(err, StoreError::Undelivered { seq: 1, .. }));
        assert_eq!(store.get_doc("docs", "a").unwrap(), Some(json!({"n": 1})));
        assert!(fake.commits().is_empty());
        assert_eq!(store.synced_seq(), 0);
        fake.fail.store(false, Ordering::SeqCst);
        assert_eq!(store.sync_flush().unwrap(), 1);
        assert_eq!(fake.commits().len(), 1);
        assert_eq!(fake.commits()[0].seq, 1);
        assert_eq!(store.synced_seq(), 1);
    }

    #[test]
    fn local_cache_fills_a_miss_once_and_keeps_json_null() {
        let (store, fake) = attach(SyncMode::LocalCache);
        *fake.fetch.lock().unwrap() = Some(json!({"cached": true}));
        assert_eq!(
            store.get_doc("docs", "a").unwrap(),
            Some(json!({"cached": true}))
        );
        assert!(fake.commits().is_empty());
        assert_eq!(fake.fetches.load(Ordering::SeqCst), 1);
        assert_eq!(
            store.get_doc("docs", "a").unwrap(),
            Some(json!({"cached": true}))
        );
        assert_eq!(fake.fetches.load(Ordering::SeqCst), 1);

        let store = FerroStore::in_memory();
        store.put_doc("docs", "n", &Value::Null).unwrap();
        store
            .attach_sync_adapter(
                SyncMode::LocalCache,
                Arc::clone(&fake) as Arc<dyn SyncAdapter>,
            )
            .unwrap();
        let before = fake.fetches.load(Ordering::SeqCst);
        assert_eq!(store.get_doc("docs", "n").unwrap(), Some(Value::Null));
        assert_eq!(fake.fetches.load(Ordering::SeqCst), before);
    }

    #[test]
    fn write_through_miss_does_not_fetch() {
        let (store, fake) = attach(SyncMode::WriteThrough);
        *fake.fetch.lock().unwrap() = Some(json!({"cached": true}));
        assert_eq!(store.get_doc("docs", "missing").unwrap(), None);
        assert_eq!(fake.fetches.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn cloud_pre_cache_prewarm_applies_without_on_commit() {
        let (store, fake) = attach(SyncMode::CloudPreCache);
        *fake.prewarm.lock().unwrap() = vec![SyncBatch {
            seq: 1,
            ops: vec![SyncOp::PutDoc {
                collection: "docs".into(),
                id: "warm".into(),
                doc: json!({"ready": true}),
            }],
        }];
        assert_eq!(store.sync_prewarm(&["docs"]).unwrap(), 1);
        assert_eq!(
            store.get_doc("docs", "warm").unwrap(),
            Some(json!({"ready": true}))
        );
        assert!(fake.commits().is_empty());
        assert_eq!(store.synced_seq(), 1);
    }

    #[test]
    fn sync_pull_applies_the_next_batch_and_rejects_a_gap() {
        let (store, fake) = attach(SyncMode::WriteBehind);
        *fake.pull.lock().unwrap() = vec![SyncBatch {
            seq: 1,
            ops: vec![SyncOp::PutDoc {
                collection: "docs".into(),
                id: "a".into(),
                doc: json!({"n": 1}),
            }],
        }];
        assert_eq!(store.sync_pull().unwrap(), 1);
        assert_eq!(store.get_doc("docs", "a").unwrap(), Some(json!({"n": 1})));
        assert!(fake.commits().is_empty());
        assert_eq!(fake.pull_after.lock().unwrap().as_slice(), &[0]);

        *fake.pull.lock().unwrap() = vec![SyncBatch {
            seq: 3,
            ops: vec![SyncOp::PutDoc {
                collection: "docs".into(),
                id: "gap".into(),
                doc: json!({"n": 3}),
            }],
        }];
        assert!(matches!(
            store.sync_pull().unwrap_err(),
            StoreError::Backend(_)
        ));
        assert_eq!(store.get_doc("docs", "gap").unwrap(), None);
        assert_eq!(store.get_doc("docs", "a").unwrap(), Some(json!({"n": 1})));
        assert_eq!(store.synced_seq(), 1);
    }

    #[test]
    fn detach_refuses_a_pending_batch_and_clears_mode_after_flush() {
        let (store, _fake) = attach(SyncMode::WriteBehind);
        store.put_doc("docs", "a", &json!({"n": 1})).unwrap();
        assert!(matches!(
            store.detach_sync_adapter().unwrap_err(),
            StoreError::Backend(_)
        ));
        assert_eq!(store.sync_mode(), Some(SyncMode::WriteBehind));
        assert_eq!(store.sync_flush().unwrap(), 1);
        store.detach_sync_adapter().unwrap();
        assert_eq!(store.sync_mode(), None);
        assert_eq!(store.synced_seq(), 1);
        assert!(matches!(
            store.sync_flush().unwrap_err(),
            StoreError::Backend(_)
        ));
    }

    #[test]
    fn write_through_disk_commit_survives_reopen_without_an_adapter() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("store.fls");
        let fake = Arc::new(FakeSync::new());
        {
            let store = FerroStore::open(&path).unwrap();
            store
                .attach_sync_adapter(
                    SyncMode::WriteThrough,
                    Arc::clone(&fake) as Arc<dyn SyncAdapter>,
                )
                .unwrap();
            store.put_doc("docs", "kept", &json!({"ok": true})).unwrap();
            assert_eq!(fake.commits().len(), 1);
            assert_eq!(store.synced_seq(), 1);
        }
        let store = FerroStore::open(&path).unwrap();
        assert_eq!(
            store.get_doc("docs", "kept").unwrap(),
            Some(json!({"ok": true}))
        );
        assert_eq!(store.sync_mode(), None);
        assert_eq!(store.synced_seq(), 0);
    }

    struct NameOnly;

    impl SyncAdapter for NameOnly {
        fn name(&self) -> &str {
            "name-only"
        }

        fn on_commit(&self, _batch: &SyncBatch) -> Result<(), StoreError> {
            Ok(())
        }
    }

    #[test]
    fn default_adapter_methods_pull_prewarm_and_miss() {
        let store = FerroStore::in_memory()
            .with_sync_adapter(
                SyncMode::LocalCache,
                Arc::new(NameOnly) as Arc<dyn SyncAdapter>,
            )
            .unwrap();
        assert_eq!(store.sync_pull().unwrap(), store.synced_seq());
        assert_eq!(store.sync_prewarm(&["docs"]).unwrap(), store.synced_seq());
        assert_eq!(store.synced_seq(), 0);
        assert_eq!(store.get_doc("docs", "missing").unwrap(), None);
    }

    #[test]
    fn a_failed_write_through_batch_is_sent_first_by_the_next_commit() {
        let (store, fake) = attach(SyncMode::WriteThrough);
        fake.fail.store(true, Ordering::SeqCst);
        let err = store.put_doc("docs", "a", &json!({"n": 1})).unwrap_err();
        assert!(
            matches!(err, StoreError::Undelivered { seq: 1, .. }),
            "{err}"
        );
        assert_eq!(store.get_doc("docs", "a").unwrap(), Some(json!({"n": 1})));
        assert!(fake.commits().is_empty());
        assert_eq!(store.synced_seq(), 0);
        assert!(store.has_pending_sync(), "the failed batch stays queued");

        // While the adapter still fails, the next commit is durable and
        // returns the adapter's error: its batch is queued behind the first.
        let err = store.put_doc("docs", "b", &json!({"n": 2})).unwrap_err();
        assert!(
            matches!(err, StoreError::Undelivered { seq: 2, .. }),
            "{err}"
        );
        assert_eq!(store.get_doc("docs", "b").unwrap(), Some(json!({"n": 2})));
        assert!(fake.commits().is_empty());

        // The next commit sends the queue, oldest first, then its own batch.
        fake.fail.store(false, Ordering::SeqCst);
        store.put_doc("docs", "b2", &json!({"n": 3})).unwrap();
        assert_eq!(
            fake.commits()
                .iter()
                .map(|batch| batch.seq)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(store.synced_seq(), 3);
        assert!(!store.has_pending_sync());
        assert_eq!(store.sync_flush().unwrap(), 3);

        store.put_doc("docs", "c", &json!({"n": 4})).unwrap();
        assert_eq!(fake.commits().len(), 4);
        assert_eq!(fake.commits()[3].seq, 4);
        assert_eq!(store.synced_seq(), 4);
    }

    struct HoldFirst {
        entered: Arc<(Mutex<bool>, Condvar)>,
        release: Arc<(Mutex<bool>, Condvar)>,
        commits: Mutex<Vec<u64>>,
    }

    impl SyncAdapter for HoldFirst {
        fn name(&self) -> &str {
            "hold-first"
        }

        fn on_commit(&self, batch: &SyncBatch) -> Result<(), StoreError> {
            if batch.seq == 1 {
                {
                    let (lock, cv) = &*self.entered;
                    *lock.lock().unwrap() = true;
                    cv.notify_all();
                }
                let (lock, cv) = &*self.release;
                let mut guard = lock.lock().unwrap();
                while !*guard {
                    guard = cv.wait(guard).unwrap();
                }
            }
            self.commits.lock().unwrap().push(batch.seq);
            Ok(())
        }
    }

    #[test]
    fn write_through_in_flight_holds_the_next_commit_until_its_batch_is_sent() {
        let entered = Arc::new((Mutex::new(false), Condvar::new()));
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let adapter = Arc::new(HoldFirst {
            entered: Arc::clone(&entered),
            release: Arc::clone(&release),
            commits: Mutex::new(Vec::new()),
        });
        let store = Arc::new(
            FerroStore::in_memory()
                .with_sync_adapter(
                    SyncMode::WriteThrough,
                    Arc::clone(&adapter) as Arc<dyn SyncAdapter>,
                )
                .unwrap(),
        );
        let worker = {
            let store = Arc::clone(&store);
            thread::spawn(move || store.put_doc("docs", "a", &json!({"n": 1})).unwrap())
        };
        {
            let (lock, cv) = &*entered;
            let mut guard = lock.lock().unwrap();
            while !*guard {
                guard = cv.wait(guard).unwrap();
            }
        }
        let second = {
            let store = Arc::clone(&store);
            thread::spawn(move || store.put_doc("docs", "b", &json!({"n": 2})).unwrap())
        };
        // The second commit is applied but waits out the first notify.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while store.read_state().unwrap().last_seq < 2 {
            assert!(
                std::time::Instant::now() < deadline,
                "the second commit did not apply"
            );
            thread::yield_now();
        }
        thread::sleep(std::time::Duration::from_millis(50));
        assert!(!second.is_finished(), "returned before its batch was sent");
        assert!(adapter.commits.lock().unwrap().is_empty());
        {
            let (lock, cv) = &*release;
            *lock.lock().unwrap() = true;
            cv.notify_all();
        }
        worker.join().unwrap();
        second.join().unwrap();
        assert_eq!(adapter.commits.lock().unwrap().as_slice(), &[1, 2]);
        assert_eq!(store.synced_seq(), 2);
        assert!(!store.has_pending_sync());
    }

    struct BlockingNotify {
        calls: Mutex<Vec<u64>>,
        entered: Arc<(Mutex<bool>, Condvar)>,
        release: Arc<(Mutex<bool>, Condvar)>,
        early: Arc<(Mutex<bool>, Condvar)>,
        fail_first: bool,
        saw_first: AtomicBool,
    }

    impl SyncAdapter for BlockingNotify {
        fn name(&self) -> &str {
            "blocking-notify"
        }

        fn on_commit(&self, batch: &SyncBatch) -> Result<(), StoreError> {
            let first = batch.seq == 1
                && self
                    .saw_first
                    .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok();
            if !first {
                let (lock, _) = &*self.release;
                if !*lock.lock().unwrap() {
                    let (lock, cv) = &*self.early;
                    *lock.lock().unwrap() = true;
                    cv.notify_all();
                }
            }
            self.calls.lock().unwrap().push(batch.seq);
            if first {
                {
                    let (lock, cv) = &*self.entered;
                    *lock.lock().unwrap() = true;
                    cv.notify_all();
                }
                let (lock, cv) = &*self.release;
                let mut guard = lock.lock().unwrap();
                while !*guard {
                    guard = cv.wait(guard).unwrap();
                }
                if self.fail_first {
                    return Err(StoreError::Backend(
                        "sync adapter rejected the batch".into(),
                    ));
                }
            }
            Ok(())
        }
    }

    struct BlockingRig {
        adapter: Arc<BlockingNotify>,
        entered: Arc<(Mutex<bool>, Condvar)>,
        release: Arc<(Mutex<bool>, Condvar)>,
    }

    fn blocking_adapter(fail_first: bool) -> BlockingRig {
        let entered = Arc::new((Mutex::new(false), Condvar::new()));
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let adapter = Arc::new(BlockingNotify {
            calls: Mutex::new(Vec::new()),
            entered: Arc::clone(&entered),
            release: Arc::clone(&release),
            early: Arc::new((Mutex::new(false), Condvar::new())),
            fail_first,
            saw_first: AtomicBool::new(false),
        });
        BlockingRig {
            adapter,
            entered,
            release,
        }
    }

    fn wait_flag(pair: &Arc<(Mutex<bool>, Condvar)>) {
        let (lock, cv) = &**pair;
        let mut guard = lock.lock().unwrap();
        while !*guard {
            guard = cv.wait(guard).unwrap();
        }
    }

    #[test]
    fn write_through_second_put_returns_only_after_the_first_notify_and_its_own() {
        let BlockingRig {
            adapter,
            entered,
            release,
        } = blocking_adapter(false);
        let store = Arc::new(
            FerroStore::in_memory()
                .with_sync_adapter(
                    SyncMode::WriteThrough,
                    Arc::clone(&adapter) as Arc<dyn SyncAdapter>,
                )
                .unwrap(),
        );
        let first = {
            let store = Arc::clone(&store);
            thread::spawn(move || store.put_doc("docs", "a", &json!({"n": 1})).unwrap())
        };
        wait_flag(&entered);
        let early = Arc::clone(&adapter.early);
        let second = {
            let store = Arc::clone(&store);
            thread::spawn(move || store.put_doc("docs", "b", &json!({"n": 2})).unwrap())
        };
        {
            let (lock, cv) = &*early;
            let guard = lock.lock().unwrap();
            let (guard, status) = cv
                .wait_timeout(guard, std::time::Duration::from_millis(250))
                .unwrap();
            assert!(
                status.timed_out() && !*guard,
                "the second commit called on_commit while the first was in flight"
            );
        }
        assert!(!second.is_finished(), "returned before its batch was sent");
        assert_eq!(adapter.calls.lock().unwrap().as_slice(), &[1]);
        {
            let (lock, cv) = &*release;
            *lock.lock().unwrap() = true;
            cv.notify_all();
        }
        first.join().unwrap();
        second.join().unwrap();
        assert_eq!(adapter.calls.lock().unwrap().as_slice(), &[1, 2]);
        assert_eq!(store.synced_seq(), 2);
        assert!(!store.has_pending_sync());
    }

    #[test]
    fn a_queued_commit_waits_out_a_blocked_notify_and_then_sends_the_failed_batch_first() {
        let BlockingRig {
            adapter,
            entered,
            release,
        } = blocking_adapter(true);
        let early = Arc::clone(&adapter.early);
        let store = Arc::new(
            FerroStore::in_memory()
                .with_sync_adapter(
                    SyncMode::WriteThrough,
                    Arc::clone(&adapter) as Arc<dyn SyncAdapter>,
                )
                .unwrap(),
        );
        let first = {
            let store = Arc::clone(&store);
            thread::spawn(move || store.put_doc("docs", "a", &json!({"n": 1})))
        };
        wait_flag(&entered);
        // The second commit queues behind the in-flight notify and drains.
        let second = {
            let store = Arc::clone(&store);
            thread::spawn(move || store.put_doc("docs", "b", &json!({"n": 2})).unwrap())
        };
        let flushed = {
            let store = Arc::clone(&store);
            thread::spawn(move || store.sync_flush())
        };
        {
            let (lock, cv) = &*early;
            let guard = lock.lock().unwrap();
            let (guard, status) = cv
                .wait_timeout(guard, std::time::Duration::from_millis(250))
                .unwrap();
            assert!(
                status.timed_out() && !*guard,
                "a drain or flush called on_commit while an older notify was still in flight"
            );
        }
        assert!(!second.is_finished(), "returned before its batch was sent");
        {
            let (lock, cv) = &*release;
            *lock.lock().unwrap() = true;
            cv.notify_all();
        }
        let err = first.join().unwrap().unwrap_err();
        assert!(
            matches!(err, StoreError::Undelivered { seq: 1, .. }),
            "{err}"
        );
        second.join().unwrap();
        assert_eq!(flushed.join().unwrap().unwrap(), 2);
        assert_eq!(adapter.calls.lock().unwrap().as_slice(), &[1, 1, 2]);
        assert_eq!(store.synced_seq(), 2);
        assert_eq!(store.get_doc("docs", "b").unwrap(), Some(json!({"n": 2})));
    }

    fn bad_second_batch() -> Vec<SyncBatch> {
        vec![
            SyncBatch {
                seq: 1,
                ops: vec![SyncOp::PutDoc {
                    collection: "docs".into(),
                    id: "a".into(),
                    doc: json!({"n": 1}),
                }],
            },
            SyncBatch {
                seq: 2,
                ops: vec![SyncOp::DeleteVector {
                    collection: "missing".into(),
                    id: "x".into(),
                }],
            },
        ]
    }

    #[test]
    fn multi_batch_pull_with_a_bad_later_op_writes_nothing() {
        let (store, fake) = attach(SyncMode::WriteBehind);
        *fake.pull.lock().unwrap() = bad_second_batch();
        assert!(matches!(
            store.sync_pull().unwrap_err(),
            StoreError::Backend(_)
        ));
        assert_eq!(store.get_doc("docs", "a").unwrap(), None);
        assert_eq!(store.synced_seq(), 0);
        assert_eq!(store.read_state().unwrap().last_seq, 0);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pull.fls");
        let store = FerroStore::open(&path).unwrap();
        store
            .attach_sync_adapter(
                SyncMode::WriteBehind,
                Arc::clone(&fake) as Arc<dyn SyncAdapter>,
            )
            .unwrap();
        assert!(matches!(
            store.sync_pull().unwrap_err(),
            StoreError::Backend(_)
        ));
        assert_eq!(store.synced_seq(), 0);
        drop(store);
        let store = FerroStore::open(&path).unwrap();
        assert_eq!(store.get_doc("docs", "a").unwrap(), None);
        assert_eq!(store.read_state().unwrap().last_seq, 0);
    }

    #[test]
    fn silent_two_batch_disk_apply_publishes_one_superblock() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("epoch.fls");
        let store = FerroStore::open(&path).unwrap();
        let before = {
            let file = std::fs::File::open(&path).unwrap();
            super::super::latest_superblock(&file).unwrap().1.epoch
        };
        let fake = Arc::new(FakeSync::new());
        store
            .attach_sync_adapter(
                SyncMode::WriteBehind,
                Arc::clone(&fake) as Arc<dyn SyncAdapter>,
            )
            .unwrap();
        *fake.pull.lock().unwrap() = vec![
            SyncBatch {
                seq: 1,
                ops: vec![SyncOp::PutDoc {
                    collection: "docs".into(),
                    id: "a".into(),
                    doc: json!({"n": 1}),
                }],
            },
            SyncBatch {
                seq: 2,
                ops: vec![SyncOp::PutDoc {
                    collection: "docs".into(),
                    id: "b".into(),
                    doc: json!({"n": 2}),
                }],
            },
        ];
        assert_eq!(store.sync_pull().unwrap(), 2);
        let after = {
            let file = std::fs::File::open(&path).unwrap();
            super::super::latest_superblock(&file).unwrap().1.epoch
        };
        assert_eq!(after, before + 1);
        drop(store);
        let store = FerroStore::open(&path).unwrap();
        assert_eq!(store.get_doc("docs", "a").unwrap(), Some(json!({"n": 1})));
        assert_eq!(store.get_doc("docs", "b").unwrap(), Some(json!({"n": 2})));
        assert_eq!(store.read_state().unwrap().last_seq, 2);
        assert!(fake.commits().is_empty());
    }

    #[test]
    fn with_sync_adapter_while_pending_keeps_the_original_adapter() {
        let original = Arc::new(FakeSync::new());
        let store = FerroStore::in_memory()
            .with_sync_adapter(
                SyncMode::WriteBehind,
                Arc::clone(&original) as Arc<dyn SyncAdapter>,
            )
            .unwrap();
        store.put_doc("docs", "a", &json!({"n": 1})).unwrap();
        let replacement = Arc::new(FakeSync::new());
        let err = store
            .attach_sync_adapter(
                SyncMode::WriteThrough,
                Arc::clone(&replacement) as Arc<dyn SyncAdapter>,
            )
            .unwrap_err();
        assert!(
            matches!(err, StoreError::Backend(ref message) if message == "cannot attach a sync adapter while batches are pending"),
            "{err}"
        );
        assert_eq!(store.sync_mode(), Some(SyncMode::WriteBehind));
        assert_eq!(store.synced_seq(), 0);
        assert_eq!(store.sync_flush().unwrap(), 1);
        assert_eq!(original.commits().len(), 1);
        assert_eq!(original.commits()[0].seq, 1);
        assert!(replacement.commits().is_empty());

        let original = Arc::new(FakeSync::new());
        let store = FerroStore::in_memory()
            .with_sync_adapter(
                SyncMode::WriteBehind,
                Arc::clone(&original) as Arc<dyn SyncAdapter>,
            )
            .unwrap();
        store.put_doc("docs", "b", &json!({"n": 2})).unwrap();
        let failed = store.with_sync_adapter(
            SyncMode::CloudPreCache,
            Arc::new(FakeSync::new()) as Arc<dyn SyncAdapter>,
        );
        let err = match failed {
            Ok(_) => panic!("attached a sync adapter while batches are pending"),
            Err(err) => err,
        };
        assert!(
            matches!(err, StoreError::Backend(ref message) if message == "cannot attach a sync adapter while batches are pending"),
            "{err}"
        );
    }

    struct FetchThenPut {
        put: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
        fetches: AtomicUsize,
    }

    impl SyncAdapter for FetchThenPut {
        fn name(&self) -> &str {
            "fetch-then-put"
        }

        fn on_commit(&self, _batch: &SyncBatch) -> Result<(), StoreError> {
            Ok(())
        }

        fn fetch_document(&self, _: &str, _: &str) -> Result<Option<Value>, StoreError> {
            self.fetches.fetch_add(1, Ordering::SeqCst);
            if let Some(put) = self.put.lock().unwrap().as_ref() {
                put();
            }
            Ok(Some(json!({"stale": true})))
        }
    }

    fn assert_fill_keeps_the_callback_put(store: FerroStore) {
        let fetch = Arc::new(FetchThenPut {
            put: Mutex::new(None),
            fetches: AtomicUsize::new(0),
        });
        let store = store
            .with_sync_adapter(
                SyncMode::LocalCache,
                Arc::clone(&fetch) as Arc<dyn SyncAdapter>,
            )
            .unwrap();
        let addr = &store as *const FerroStore as usize;
        // `get_doc` calls `fetch_document` on this stack, before `store` moves.
        *fetch.put.lock().unwrap() = Some(Box::new(move || {
            let store = unsafe { &*(addr as *const FerroStore) };
            store.put_doc("docs", "a", &json!({"fresh": true})).unwrap();
        }));
        assert_eq!(
            store.get_doc("docs", "a").unwrap(),
            Some(json!({"fresh": true}))
        );
        assert_eq!(
            store.get_doc("docs", "a").unwrap(),
            Some(json!({"fresh": true}))
        );
        assert_eq!(fetch.fetches.load(Ordering::SeqCst), 1);
        store.put_doc("docs", "n", &Value::Null).unwrap();
        let before = fetch.fetches.load(Ordering::SeqCst);
        assert_eq!(store.get_doc("docs", "n").unwrap(), Some(Value::Null));
        assert_eq!(fetch.fetches.load(Ordering::SeqCst), before);
    }

    #[test]
    fn cache_fill_does_not_overwrite_a_document_inserted_during_fetch() {
        assert_fill_keeps_the_callback_put(FerroStore::in_memory());
        let dir = tempfile::tempdir().unwrap();
        assert_fill_keeps_the_callback_put(FerroStore::open(dir.path().join("fill.fls")).unwrap());
    }

    #[test]
    fn remote_put_doc_reports_the_commit_seq_not_a_later_read() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("seq.sock");
        let server = FerroStoreServer::bind_unix(&sock, FerroStore::in_memory()).unwrap();
        let uri = format!("unix://{}", sock.display());
        let client = FerroStore::connect_uri(&uri).unwrap();
        let fake = Arc::new(FakeSync::new());
        client
            .attach_sync_adapter(
                SyncMode::WriteThrough,
                Arc::clone(&fake) as Arc<dyn SyncAdapter>,
            )
            .unwrap();
        client.put_doc("docs", "a", &json!({"n": 1})).unwrap();
        assert_eq!(fake.commits()[0].seq, 1);
        assert_eq!(server.store().read_state().unwrap().last_seq, 1);

        let other = FerroStore::connect_uri(&uri).unwrap();
        other.put_doc("docs", "c", &json!({"n": 9})).unwrap();
        client.put_doc("docs", "b", &json!({"n": 2})).unwrap();
        assert_eq!(
            fake.commits()
                .iter()
                .map(|batch| batch.seq)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
        assert_eq!(server.store().read_state().unwrap().last_seq, 3);
        drop(other);
        drop(client);
        drop(server);
    }

    #[test]
    fn remote_silent_fill_pull_and_prewarm_skip_the_server_adapter() {
        let dir = tempfile::tempdir().unwrap();
        let sock = dir.path().join("silent.sock");
        let server_fake = Arc::new(FakeSync::new());
        let server_store = FerroStore::in_memory();
        server_store
            .attach_sync_adapter(
                SyncMode::WriteThrough,
                Arc::clone(&server_fake) as Arc<dyn SyncAdapter>,
            )
            .unwrap();
        let server = FerroStoreServer::bind_unix(&sock, server_store).unwrap();
        let uri = format!("unix://{}", sock.display());
        let client = FerroStore::connect_uri(&uri).unwrap();
        let client_fake = Arc::new(FakeSync::new());
        client
            .attach_sync_adapter(
                SyncMode::WriteThrough,
                Arc::clone(&client_fake) as Arc<dyn SyncAdapter>,
            )
            .unwrap();
        client.put_doc("docs", "loud", &json!({"n": 1})).unwrap();
        assert_eq!(server_fake.commits().len(), 1);
        assert_eq!(server_fake.commits()[0].seq, 1);
        assert_eq!(client_fake.commits().len(), 1);
        client.detach_sync_adapter().unwrap();
        client
            .attach_sync_adapter(
                SyncMode::LocalCache,
                Arc::clone(&client_fake) as Arc<dyn SyncAdapter>,
            )
            .unwrap();

        *client_fake.fetch.lock().unwrap() = Some(json!({"cached": true}));
        let server_commits = server_fake.commits().len();
        assert_eq!(
            client.get_doc("docs", "cached").unwrap(),
            Some(json!({"cached": true}))
        );
        assert_eq!(server_fake.commits().len(), server_commits);
        assert_eq!(
            server.store().get_doc("docs", "cached").unwrap(),
            Some(json!({"cached": true}))
        );
        assert_eq!(client_fake.commits().len(), 1);

        let next = server.store().read_state().unwrap().last_seq + 1;
        *client_fake.pull.lock().unwrap() = vec![SyncBatch {
            seq: next,
            ops: vec![SyncOp::PutDoc {
                collection: "docs".into(),
                id: "pulled".into(),
                doc: json!({"from": "pull"}),
            }],
        }];
        assert_eq!(client.sync_pull().unwrap(), next);
        assert_eq!(server_fake.commits().len(), server_commits);
        assert_eq!(
            server.store().get_doc("docs", "pulled").unwrap(),
            Some(json!({"from": "pull"}))
        );
        assert_eq!(client.synced_seq(), next);

        let warm = next + 1;
        *client_fake.prewarm.lock().unwrap() = vec![SyncBatch {
            seq: warm,
            ops: vec![SyncOp::PutDoc {
                collection: "docs".into(),
                id: "warm".into(),
                doc: json!({"from": "prewarm"}),
            }],
        }];
        assert_eq!(client.sync_prewarm(&["docs"]).unwrap(), warm);
        assert_eq!(server_fake.commits().len(), server_commits);
        assert_eq!(
            server.store().get_doc("docs", "warm").unwrap(),
            Some(json!({"from": "prewarm"}))
        );
        assert_eq!(client.synced_seq(), warm);
        assert_eq!(client_fake.commits().len(), 1);
        drop(client);
        drop(server);
    }

    fn two_doc_batches() -> Vec<SyncBatch> {
        vec![
            SyncBatch {
                seq: 1,
                ops: vec![SyncOp::PutDoc {
                    collection: "docs".into(),
                    id: "a".into(),
                    doc: json!({"n": 1}),
                }],
            },
            SyncBatch {
                seq: 2,
                ops: vec![SyncOp::PutDoc {
                    collection: "docs".into(),
                    id: "b".into(),
                    doc: json!({"n": 2}),
                }],
            },
        ]
    }

    #[test]
    fn loud_apply_flushes_only_in_write_through() {
        for mode in [
            SyncMode::WriteBehind,
            SyncMode::LocalCache,
            SyncMode::CloudPreCache,
        ] {
            let (store, fake) = attach(mode);
            assert_eq!(
                store.apply_sync_batches(two_doc_batches(), false).unwrap(),
                2
            );
            assert!(
                fake.commits().is_empty(),
                "{mode:?} called on_commit from a loud apply"
            );
            assert_eq!(store.synced_seq(), 0);
            assert_eq!(store.get_doc("docs", "a").unwrap(), Some(json!({"n": 1})));
            assert_eq!(store.get_doc("docs", "b").unwrap(), Some(json!({"n": 2})));
            assert_eq!(store.sync_flush().unwrap(), 2);
            assert_eq!(
                fake.commits()
                    .iter()
                    .map(|batch| batch.seq)
                    .collect::<Vec<_>>(),
                vec![1, 2]
            );
            assert_eq!(store.synced_seq(), 2);
        }

        let (store, fake) = attach(SyncMode::WriteThrough);
        assert_eq!(
            store.apply_sync_batches(two_doc_batches(), false).unwrap(),
            2
        );
        assert_eq!(
            fake.commits()
                .iter()
                .map(|batch| batch.seq)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(store.synced_seq(), 2);
        assert_eq!(store.get_doc("docs", "b").unwrap(), Some(json!({"n": 2})));
        assert_eq!(store.sync_flush().unwrap(), 2);
        assert_eq!(fake.commits().len(), 2);
    }

    struct FlushOnCommit {
        store: Mutex<Option<Arc<FerroStore>>>,
        seqs: Mutex<Vec<u64>>,
    }

    impl SyncAdapter for FlushOnCommit {
        fn name(&self) -> &str {
            "flush-on-commit"
        }

        fn on_commit(&self, batch: &SyncBatch) -> Result<(), StoreError> {
            self.seqs.lock().unwrap().push(batch.seq);
            let store = self.store.lock().unwrap().clone().unwrap();
            store.sync_flush()?;
            Ok(())
        }
    }

    #[test]
    fn sync_flush_from_in_flight_on_commit_returns() {
        let adapter = Arc::new(FlushOnCommit {
            store: Mutex::new(None),
            seqs: Mutex::new(Vec::new()),
        });
        let store = Arc::new(
            FerroStore::in_memory()
                .with_sync_adapter(
                    SyncMode::WriteThrough,
                    Arc::clone(&adapter) as Arc<dyn SyncAdapter>,
                )
                .unwrap(),
        );
        *adapter.store.lock().unwrap() = Some(Arc::clone(&store));
        store.put_doc("docs", "a", &json!({"n": 1})).unwrap();
        assert_eq!(adapter.seqs.lock().unwrap().as_slice(), &[1]);
        assert_eq!(store.synced_seq(), 1);
        assert_eq!(store.get_doc("docs", "a").unwrap(), Some(json!({"n": 1})));

        let behind = Arc::new(
            FerroStore::in_memory()
                .with_sync_adapter(
                    SyncMode::WriteBehind,
                    Arc::clone(&adapter) as Arc<dyn SyncAdapter>,
                )
                .unwrap(),
        );
        behind.put_doc("docs", "a", &json!({"n": 1})).unwrap();
        behind.put_doc("docs", "b", &json!({"n": 2})).unwrap();
        assert_eq!(adapter.seqs.lock().unwrap().as_slice(), &[1]);
        *adapter.store.lock().unwrap() = Some(Arc::clone(&behind));
        adapter.seqs.lock().unwrap().clear();
        assert_eq!(behind.sync_flush().unwrap(), 2);
        assert_eq!(adapter.seqs.lock().unwrap().as_slice(), &[1, 2]);
        assert_eq!(behind.synced_seq(), 2);
        *adapter.store.lock().unwrap() = None;
    }

    fn every_kind_ops() -> Vec<SyncOp> {
        vec![
            SyncOp::PutDoc {
                collection: "docs".into(),
                id: "a".into(),
                doc: json!({"n": 1, "nested": {"ok": [true, null]}}),
            },
            SyncOp::DeleteDoc {
                collection: "docs".into(),
                id: "gone".into(),
            },
            SyncOp::CreateColumnTable {
                name: "t".into(),
                columns: vec![
                    ColumnDef {
                        name: "n".into(),
                        ty: columnar::ColumnType::Int64,
                    },
                    ColumnDef {
                        name: "s".into(),
                        ty: columnar::ColumnType::Utf8,
                    },
                ],
            },
            SyncOp::AppendColumnGroup {
                name: "t".into(),
                bytes: vec![0, 1, 2, 255],
            },
            SyncOp::CreateVectorCollection {
                name: "v".into(),
                dim: 2,
                storage: VectorStorage::F16,
                metric: VectorMetric::DotProduct,
                m: 4,
                ef_construction: 8,
                ef_search: 3,
            },
            SyncOp::UpsertVector {
                collection: "v".into(),
                id: "a".into(),
                bytes: vec![9, 8, 7, 6],
                metadata: Some(Value::Null),
            },
            SyncOp::DeleteVector {
                collection: "v".into(),
                id: "a".into(),
            },
        ]
    }

    const ALL_KINDS: [EnvelopeKind; 7] = [
        EnvelopeKind::Document,
        EnvelopeKind::DeleteDocument,
        EnvelopeKind::ColumnTableSchema,
        EnvelopeKind::ColumnRowGroup,
        EnvelopeKind::VectorCollectionSchema,
        EnvelopeKind::VectorUpsert,
        EnvelopeKind::VectorDelete,
    ];

    #[test]
    fn envelopes_round_trip_every_kind() {
        let batch = SyncBatch {
            seq: 41,
            ops: every_kind_ops(),
        };
        let envelopes = batch.to_envelopes().unwrap();
        assert_eq!(
            envelopes.iter().map(|e| e.kind).collect::<Vec<_>>(),
            ALL_KINDS
        );
        for (index, envelope) in envelopes.iter().enumerate() {
            assert_eq!(envelope.seq, 41);
            assert_eq!(envelope.op_index as usize, index);
            assert_eq!(envelope.schema_version, 1);
            assert_eq!(envelope.kind.tag() as usize, index);
            let decoded = RecordEnvelope::decode(&envelope.encode().unwrap()).unwrap();
            assert_eq!(&decoded, envelope);
        }
        let rebuilt = SyncBatch::from_envelopes(41, &envelopes).unwrap();
        assert_eq!(rebuilt, batch);
        assert_eq!(rebuilt.to_envelopes().unwrap(), envelopes);
    }

    #[test]
    fn envelopes_of_an_empty_batch_round_trip() {
        let batch = SyncBatch {
            seq: 3,
            ops: Vec::new(),
        };
        assert!(batch.to_envelopes().unwrap().is_empty());
        assert_eq!(SyncBatch::from_envelopes(3, &[]).unwrap(), batch);
    }

    #[test]
    fn envelope_metadata_distinguishes_absent_from_null() {
        let ops = |metadata| SyncOp::UpsertVector {
            collection: "v".into(),
            id: "a".into(),
            bytes: vec![1],
            metadata,
        };
        for metadata in [None, Some(Value::Null), Some(json!({"k": [1, 2]}))] {
            let batch = SyncBatch {
                seq: 1,
                ops: vec![ops(metadata)],
            };
            let envelopes = batch.to_envelopes().unwrap();
            let decoded = RecordEnvelope::decode(&envelopes[0].encode().unwrap()).unwrap();
            assert_eq!(SyncBatch::from_envelopes(1, &[decoded]).unwrap(), batch);
        }
    }

    fn sample_frames() -> Vec<Vec<u8>> {
        let mut frames: Vec<Vec<u8>> = SyncBatch {
            seq: 7,
            ops: every_kind_ops(),
        }
        .to_envelopes()
        .unwrap()
        .iter()
        .map(|envelope| envelope.encode().unwrap())
        .collect();
        frames.push(
            RecordEnvelope {
                seq: 7,
                op_index: 9,
                kind: EnvelopeKind::VectorUpsert,
                namespace: "v".into(),
                record_id: "f".into(),
                schema_version: 1,
                payload_bytes: Vec::new(),
                metadata: Some(json!({"k": 1})),
                vector_f32: Some(vec![1.5, -2.0, 0.25]),
            }
            .encode()
            .unwrap(),
        );
        frames
    }

    #[test]
    fn envelope_frame_with_floats_round_trips() {
        let frame = sample_frames().pop().unwrap();
        let decoded = RecordEnvelope::decode(&frame).unwrap();
        assert_eq!(decoded.vector_f32, Some(vec![1.5, -2.0, 0.25]));
        assert_eq!(decoded.encode().unwrap(), frame);
        let batch = SyncBatch::from_envelopes(
            7,
            &[RecordEnvelope {
                op_index: 0,
                ..decoded
            }],
        )
        .unwrap();
        let SyncOp::UpsertVector { bytes, .. } = &batch.ops[0] else {
            panic!("not an upsert");
        };
        let expected: Vec<u8> = [1.5f32, -2.0, 0.25]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert_eq!(bytes, &expected);
    }

    #[test]
    fn every_truncated_envelope_frame_is_rejected() {
        for frame in sample_frames() {
            for len in 0..frame.len() {
                let err = RecordEnvelope::decode(&frame[..len]).unwrap_err();
                assert!(matches!(err, StoreError::Backend(_)), "len {len}: {err}");
            }
            RecordEnvelope::decode(&frame).unwrap();
        }
    }

    #[test]
    fn envelope_frame_with_trailing_bytes_is_rejected() {
        for frame in sample_frames() {
            let mut long = frame.clone();
            long.push(0);
            let err = RecordEnvelope::decode(&long).unwrap_err();
            assert!(
                matches!(&err, StoreError::Backend(m) if m.contains("trailing")),
                "{err}"
            );
        }
    }

    // Offsets in a frame whose namespace and record id are short.
    const KIND_OFFSET: usize = 4 + 2 + 8 + 4;

    #[test]
    fn envelope_frame_with_a_bad_header_is_rejected() {
        let frame = sample_frames().remove(0);
        let mut bad_magic = frame.clone();
        bad_magic[0] = b'X';
        let mut bad_version = frame.clone();
        bad_version[4] = 9;
        for kind in 7..=255u8 {
            let mut bad_kind = frame.clone();
            bad_kind[KIND_OFFSET] = kind;
            let err = RecordEnvelope::decode(&bad_kind).unwrap_err();
            assert!(
                matches!(&err, StoreError::Backend(m) if m.contains("kind")),
                "{err}"
            );
        }
        for bad in [bad_magic, bad_version] {
            assert!(matches!(
                RecordEnvelope::decode(&bad).unwrap_err(),
                StoreError::Backend(_)
            ));
        }
    }

    #[test]
    fn envelope_frame_with_a_bad_length_prefix_is_rejected_without_allocating() {
        let frame = RecordEnvelope {
            seq: 1,
            op_index: 0,
            kind: EnvelopeKind::Document,
            namespace: "ns".into(),
            record_id: "id".into(),
            schema_version: 1,
            payload_bytes: vec![1, 2, 3],
            metadata: None,
            vector_f32: None,
        }
        .encode()
        .unwrap();
        let namespace_len = KIND_OFFSET + 1 + 2;
        for bad in [u32::MAX, u32::MAX / 2, 3, 1 << 20] {
            let mut huge = frame.clone();
            huge[namespace_len..namespace_len + 4].copy_from_slice(&bad.to_le_bytes());
            assert!(RecordEnvelope::decode(&huge).is_err(), "namespace {bad}");
        }
        let payload_len = namespace_len + 4 + 2 + 4 + 2;
        assert_eq!(
            u32::from_le_bytes(frame[payload_len..payload_len + 4].try_into().unwrap()),
            3
        );
        for bad in [u32::MAX, 4, 1 << 30] {
            let mut huge = frame.clone();
            huge[payload_len..payload_len + 4].copy_from_slice(&bad.to_le_bytes());
            assert!(RecordEnvelope::decode(&huge).is_err(), "payload {bad}");
        }
        // The float count is the last u32 before the floats.
        let floats = RecordEnvelope {
            vector_f32: Some(vec![1.0, 2.0]),
            ..RecordEnvelope::decode(&frame).unwrap()
        }
        .encode()
        .unwrap();
        let count_at = floats.len() - 2 * 4 - 4;
        for bad in [u32::MAX, u32::MAX / 4, u32::MAX / 4 + 1, 3, 1 << 28] {
            let mut huge = floats.clone();
            huge[count_at..count_at + 4].copy_from_slice(&bad.to_le_bytes());
            let err = RecordEnvelope::decode(&huge).unwrap_err();
            assert!(matches!(err, StoreError::Backend(_)), "floats {bad}: {err}");
        }
        LARGEST_ALLOC.with(|largest| largest.set(0));
        for bad in [u32::MAX, u32::MAX / 4, 1 << 28] {
            let mut huge = floats.clone();
            huge[count_at..count_at + 4].copy_from_slice(&bad.to_le_bytes());
            let _ = RecordEnvelope::decode(&huge);
        }
        let largest = LARGEST_ALLOC.with(std::cell::Cell::get);
        assert!(
            largest < 1 << 16,
            "a corrupt length prefix drove a {largest} byte allocation"
        );
        let mut flag = frame.clone();
        let metadata_flag = frame.len() - 2;
        flag[metadata_flag] = 7;
        assert!(RecordEnvelope::decode(&flag).is_err());
        let mut flag = frame;
        let vector_flag = flag.len() - 1;
        flag[vector_flag] = 7;
        assert!(RecordEnvelope::decode(&flag).is_err());
    }

    #[test]
    fn envelope_frame_with_non_utf8_or_non_json_fields_is_rejected() {
        let mut frame = RecordEnvelope {
            seq: 1,
            op_index: 0,
            kind: EnvelopeKind::Document,
            namespace: "ns".into(),
            record_id: "id".into(),
            schema_version: 1,
            payload_bytes: Vec::new(),
            metadata: None,
            vector_f32: None,
        }
        .encode()
        .unwrap();
        let namespace_at = KIND_OFFSET + 1 + 2 + 4;
        frame[namespace_at] = 0xff;
        assert!(RecordEnvelope::decode(&frame).is_err());
        let bad_json = RecordEnvelope {
            kind: EnvelopeKind::Document,
            payload_bytes: b"{not json".to_vec(),
            ..RecordEnvelope::decode(&sample_frames().remove(0)).unwrap()
        };
        let decoded = RecordEnvelope::decode(&bad_json.encode().unwrap()).unwrap();
        assert!(
            SyncBatch::from_envelopes(
                7,
                &[RecordEnvelope {
                    op_index: 0,
                    ..decoded
                }]
            )
            .is_err()
        );
    }

    #[test]
    fn from_envelopes_accepts_any_order_but_not_gaps_dupes_or_mixed_seq() {
        let batch = SyncBatch {
            seq: 5,
            ops: every_kind_ops(),
        };
        let mut shuffled = batch.to_envelopes().unwrap();
        shuffled.reverse();
        shuffled.swap(1, 4);
        assert_eq!(SyncBatch::from_envelopes(5, &shuffled).unwrap(), batch);

        let envelopes = batch.to_envelopes().unwrap();
        let mut mixed = envelopes.clone();
        mixed[3].seq = 6;
        assert!(SyncBatch::from_envelopes(5, &mixed).is_err());
        assert!(SyncBatch::from_envelopes(6, &envelopes).is_err());

        let mut gap = envelopes.clone();
        gap.remove(2);
        assert!(SyncBatch::from_envelopes(5, &gap).is_err());

        let mut dupe = envelopes.clone();
        dupe[4].op_index = 3;
        assert!(SyncBatch::from_envelopes(5, &dupe).is_err());

        let mut far = envelopes.clone();
        far[0].op_index = u32::MAX;
        assert!(SyncBatch::from_envelopes(5, &far).is_err());

        let mut version = envelopes;
        version[0].schema_version = 2;
        let err = SyncBatch::from_envelopes(5, &version).unwrap_err();
        assert!(
            matches!(&err, StoreError::Backend(m) if m.contains("schema version")),
            "{err}"
        );
    }

    #[test]
    fn from_envelopes_rejects_malformed_payloads() {
        let envelopes = SyncBatch {
            seq: 1,
            ops: every_kind_ops(),
        }
        .to_envelopes()
        .unwrap();
        let one = |index: usize, edit: &dyn Fn(&mut RecordEnvelope)| {
            let mut envelope = envelopes[index].clone();
            envelope.op_index = 0;
            edit(&mut envelope);
            SyncBatch::from_envelopes(1, &[envelope])
        };
        assert!(one(0, &|e| e.payload_bytes = b"nope".to_vec()).is_err());
        assert!(one(2, &|e| e.payload_bytes = b"{}".to_vec()).is_err());
        assert!(one(4, &|e| e.payload_bytes = b"[]".to_vec()).is_err());
        assert!(one(4, &|e| e.payload_bytes = b"{\"dim\":2}".to_vec()).is_err());
        assert!(one(4, &|e| e.payload_bytes = json_bytes(&json!({"dim": 4294967296u64, "storage": "f32", "metric": "cosine", "m": 1, "ef_construction": 1, "ef_search": 1})).unwrap()).is_err());
        assert!(one(4, &|e| e.payload_bytes = json_bytes(&json!({"dim": 2, "storage": "f64", "metric": "cosine", "m": 1, "ef_construction": 1, "ef_search": 1})).unwrap()).is_err());
        assert!(
            one(5, &|e| {
                e.payload_bytes.clear();
                e.vector_f32 = None;
            })
            .is_err()
        );
    }

    #[test]
    fn envelopes_of_a_committed_store_replay_into_a_second_store() {
        let (store, fake) = attach(SyncMode::WriteThrough);
        store.put_doc("docs", "a", &json!({"n": 1})).unwrap();
        store
            .create_vector_collection("v", 2, VectorStorage::F32, VectorMetric::Cosine, 4, 16, 16)
            .unwrap();
        store
            .upsert_vector("v", "x", &[1.0, 0.0], Some(&json!({"k": 1})))
            .unwrap();
        let replica = FerroStore::in_memory();
        let mut wire = Vec::new();
        for batch in fake.commits() {
            for envelope in batch.to_envelopes().unwrap() {
                wire.push((batch.seq, envelope.encode().unwrap()));
            }
        }
        for seq in 1..=3u64 {
            let envelopes: Vec<RecordEnvelope> = wire
                .iter()
                .filter(|(s, _)| *s == seq)
                .map(|(_, frame)| RecordEnvelope::decode(frame).unwrap())
                .collect();
            let batch = SyncBatch::from_envelopes(seq, &envelopes).unwrap();
            replica.apply_sync_batches(vec![batch], true).unwrap();
        }
        assert_eq!(replica.get_doc("docs", "a").unwrap(), Some(json!({"n": 1})));
        assert_eq!(
            replica.get_vector("v", "x").unwrap(),
            Some((vec![1.0, 0.0], Some(json!({"k": 1}))))
        );
    }

    type VectorCall = (String, Vec<f32>, usize, Option<u32>);
    type HybridCall = (String, String, String, Vec<f32>, usize, Fusion);

    #[derive(Default)]
    struct SearchSync {
        reply: Mutex<Option<Result<Option<Vec<VectorHit>>, String>>>,
        vector_calls: Mutex<Vec<VectorCall>>,
        hybrid_calls: Mutex<Vec<HybridCall>>,
        slot_free: Mutex<Vec<bool>>,
        store: Mutex<Option<Arc<FerroStore>>>,
    }

    impl SearchSync {
        fn reply(&self) -> Result<Option<Vec<VectorHit>>, StoreError> {
            if let Some(store) = &*self.store.lock().unwrap() {
                // A caller that held the sync slot across this call fails try_lock.
                self.slot_free
                    .lock()
                    .unwrap()
                    .push(store.sync.try_lock().is_ok());
            }
            match self.reply.lock().unwrap().clone() {
                None => Ok(None),
                Some(Ok(hits)) => Ok(hits),
                Some(Err(message)) => Err(StoreError::Backend(message)),
            }
        }
    }

    impl SyncAdapter for SearchSync {
        fn name(&self) -> &str {
            "search-sync"
        }

        fn on_commit(&self, _: &SyncBatch) -> Result<(), StoreError> {
            Ok(())
        }

        fn search_vectors(
            &self,
            collection: &str,
            query: &[f32],
            k: usize,
            ef: Option<u32>,
        ) -> Result<Option<Vec<VectorHit>>, StoreError> {
            self.vector_calls
                .lock()
                .unwrap()
                .push((collection.into(), query.to_vec(), k, ef));
            self.reply()
        }

        fn search_hybrid(
            &self,
            doc_collection: &str,
            vector_collection: &str,
            query_text: &str,
            query_vector: &[f32],
            k: usize,
            fusion: Fusion,
        ) -> Result<Option<Vec<VectorHit>>, StoreError> {
            self.hybrid_calls.lock().unwrap().push((
                doc_collection.into(),
                vector_collection.into(),
                query_text.into(),
                query_vector.to_vec(),
                k,
                fusion,
            ));
            self.reply()
        }
    }

    fn search_store() -> FerroStore {
        let store = FerroStore::in_memory();
        store
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
        store
            .put_doc("notes", "near", &json!({"text": "alpha widget"}))
            .unwrap();
        store
            .upsert_vector("emb", "near", &[1.0, 0.0], None)
            .unwrap();
        store
            .upsert_vector("emb", "far", &[0.0, 1.0], None)
            .unwrap();
        store
    }

    fn cloud_hit(id: &str, score: f64) -> VectorHit {
        VectorHit {
            id: id.into(),
            score,
        }
    }

    const RRF: Fusion = Fusion::Rrf { k: 60 };

    #[test]
    fn search_without_an_adapter_is_local() {
        let store = search_store();
        let hits = store.search_vectors("emb", &[1.0, 0.0], 2, None).unwrap();
        assert_eq!(hits[0].0, "near");
        let hybrid = store
            .search_hybrid("notes", "emb", "alpha", &[1.0, 0.0], 2, RRF)
            .unwrap();
        assert_eq!(hybrid[0].0, "near");
    }

    #[test]
    fn search_returns_cloud_hits_and_passes_the_exact_arguments() {
        let store = Arc::new(search_store());
        let adapter = Arc::new(SearchSync::default());
        *adapter.store.lock().unwrap() = Some(Arc::clone(&store));
        store
            .attach_sync_adapter(
                SyncMode::LocalCache,
                Arc::clone(&adapter) as Arc<dyn SyncAdapter>,
            )
            .unwrap();
        *adapter.reply.lock().unwrap() = Some(Ok(Some(vec![
            cloud_hit("cloud-b", 0.9),
            cloud_hit("cloud-a", 0.1),
        ])));
        let hits = store
            .search_vectors("emb", &[0.5, 0.25], 7, Some(33))
            .unwrap();
        assert_eq!(
            hits,
            vec![("cloud-b".to_string(), 0.9), ("cloud-a".to_string(), 0.1)]
        );
        assert_eq!(
            adapter.vector_calls.lock().unwrap().as_slice(),
            &[("emb".to_string(), vec![0.5, 0.25], 7, Some(33))]
        );
        let fusion = Fusion::Weighted {
            bm25: 0.3,
            vector: 0.7,
        };
        let hybrid = store
            .search_hybrid("notes", "emb", "beta", &[0.5, 0.25], 4, fusion)
            .unwrap();
        assert_eq!(
            hybrid,
            vec![("cloud-b".to_string(), 0.9), ("cloud-a".to_string(), 0.1)]
        );
        assert_eq!(
            adapter.hybrid_calls.lock().unwrap().as_slice(),
            &[(
                "notes".to_string(),
                "emb".to_string(),
                "beta".to_string(),
                vec![0.5, 0.25],
                4,
                fusion
            )]
        );
        // The adapter ran without the sync slot held.
        assert_eq!(adapter.slot_free.lock().unwrap().as_slice(), &[true, true]);
        *adapter.store.lock().unwrap() = None;
    }

    #[test]
    fn search_falls_back_to_local_on_ok_none_and_on_err() {
        let store = search_store();
        let adapter = Arc::new(SearchSync::default());
        store
            .attach_sync_adapter(
                SyncMode::LocalCache,
                Arc::clone(&adapter) as Arc<dyn SyncAdapter>,
            )
            .unwrap();
        for reply in [None, Some(Ok(None)), Some(Err("cloud down".to_string()))] {
            *adapter.reply.lock().unwrap() = reply;
            let hits = store.search_vectors("emb", &[1.0, 0.0], 2, None).unwrap();
            assert_eq!(hits[0].0, "near");
            assert_eq!(hits.len(), 2);
            let hybrid = store
                .search_hybrid("notes", "emb", "alpha", &[1.0, 0.0], 2, RRF)
                .unwrap();
            assert_eq!(hybrid[0].0, "near");
        }
        assert_eq!(adapter.vector_calls.lock().unwrap().len(), 3);
        assert_eq!(adapter.hybrid_calls.lock().unwrap().len(), 3);
    }

    #[test]
    fn default_adapter_search_methods_decline() {
        let adapter = NameOnly;
        assert!(
            adapter
                .search_vectors("c", &[1.0], 1, None)
                .unwrap()
                .is_none()
        );
        assert!(
            adapter
                .search_hybrid("d", "c", "q", &[1.0], 1, RRF)
                .unwrap()
                .is_none()
        );
    }

    fn within(pair: &Arc<(Mutex<bool>, Condvar)>, what: &str) {
        let (lock, cv) = &**pair;
        let guard = lock.lock().unwrap();
        let (guard, status) = cv
            .wait_timeout_while(guard, std::time::Duration::from_secs(10), |set| !*set)
            .unwrap();
        assert!(
            !status.timed_out() && *guard,
            "timed out waiting for {what}"
        );
    }

    fn open(pair: &Arc<(Mutex<bool>, Condvar)>) {
        let (lock, cv) = &**pair;
        *lock.lock().unwrap() = true;
        cv.notify_all();
    }

    fn bounded_in_flight_rig() -> (Arc<FerroStore>, BlockingRig) {
        let rig = blocking_adapter(false);
        let store = Arc::new(
            FerroStore::in_memory()
                .with_sync_adapter(
                    SyncMode::WriteThrough,
                    Arc::clone(&rig.adapter) as Arc<dyn SyncAdapter>,
                )
                .unwrap(),
        );
        (store, rig)
    }

    #[test]
    fn has_pending_sync_is_true_while_a_notify_is_in_flight() {
        let (store, rig) = bounded_in_flight_rig();
        assert!(!store.has_pending_sync());
        let writer = {
            let store = Arc::clone(&store);
            thread::spawn(move || store.put_doc("docs", "a", &json!({"n": 1})).unwrap())
        };
        within(&rig.entered, "on_commit to start");
        // The batch is neither pending nor synced yet: only the flag shows it.
        assert!(store.has_pending_sync());
        assert_eq!(store.synced_seq(), 0);
        open(&rig.release);
        writer.join().unwrap();
        assert!(!store.has_pending_sync());
        assert_eq!(store.synced_seq(), 1);
    }

    #[test]
    fn attach_and_detach_refuse_while_a_notify_is_in_flight() {
        let (store, rig) = bounded_in_flight_rig();
        let writer = {
            let store = Arc::clone(&store);
            thread::spawn(move || store.put_doc("docs", "a", &json!({"n": 1})).unwrap())
        };
        within(&rig.entered, "on_commit to start");
        let other: Arc<dyn SyncAdapter> = Arc::new(FakeSync::new());
        let err = store
            .attach_sync_adapter(SyncMode::WriteBehind, other)
            .unwrap_err();
        assert!(
            matches!(&err, StoreError::Backend(m) if m == "cannot attach a sync adapter while batches are pending"),
            "{err}"
        );
        let err = store.detach_sync_adapter().unwrap_err();
        assert!(
            matches!(&err, StoreError::Backend(m) if m == "cannot detach a sync adapter while batches are pending"),
            "{err}"
        );
        assert_eq!(store.sync_mode(), Some(SyncMode::WriteThrough));
        open(&rig.release);
        writer.join().unwrap();
        store.detach_sync_adapter().unwrap();
        assert_eq!(store.sync_mode(), None);
    }

    #[test]
    fn inbound_apply_and_cache_fill_refuse_while_a_notify_is_in_flight() {
        let (store, rig) = bounded_in_flight_rig();
        let writer = {
            let store = Arc::clone(&store);
            thread::spawn(move || store.put_doc("docs", "a", &json!({"n": 1})).unwrap())
        };
        within(&rig.entered, "on_commit to start");
        let inbound = vec![SyncBatch {
            seq: 3,
            ops: vec![SyncOp::PutDoc {
                collection: "docs".into(),
                id: "in".into(),
                doc: json!({"in": true}),
            }],
        }];
        let err = store.apply_inbound(inbound.clone()).unwrap_err();
        assert!(
            matches!(&err, StoreError::Backend(m) if m.contains("commits are pending")),
            "{err}"
        );
        let err = store.require_sync_idle().unwrap_err();
        assert!(matches!(err, StoreError::Backend(_)));
        // A silent commit and a silent fill must not move synced_seq past the
        // unacknowledged batch.
        assert!(
            store
                .commit_if_seq_inner(
                    1,
                    &[Op::Put {
                        collection: "docs".into(),
                        id: "silent".into(),
                        doc: json!({"silent": true}),
                    }],
                    true,
                )
                .unwrap()
        );
        assert_eq!(store.synced_seq(), 0);
        let filled = store
            .install_fetched_doc("docs", "fill", json!({"fill": true}))
            .unwrap();
        assert_eq!(filled, None);
        assert_eq!(store.synced_seq(), 0);
        open(&rig.release);
        writer.join().unwrap();
        assert_eq!(store.synced_seq(), 1);
        store.apply_inbound(inbound).unwrap();
        assert_eq!(
            store.get_doc("docs", "in").unwrap(),
            Some(json!({"in": true}))
        );
    }

    /// An adapter whose first `on_commit` panics; later calls record the batch.
    struct PanicsOnce {
        armed: AtomicBool,
        commits: Mutex<Vec<u64>>,
    }

    impl PanicsOnce {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                armed: AtomicBool::new(true),
                commits: Mutex::new(Vec::new()),
            })
        }
    }

    impl SyncAdapter for PanicsOnce {
        fn name(&self) -> &str {
            "panics-once"
        }

        fn on_commit(&self, batch: &SyncBatch) -> Result<(), StoreError> {
            if self.armed.swap(false, Ordering::SeqCst) {
                panic!("adapter blew up");
            }
            self.commits.lock().unwrap().push(batch.seq);
            Ok(())
        }
    }

    /// Run `f` on a fresh thread and fail the test when it does not finish in time (a wedged
    /// store would block it for good).
    fn finishes_within<T: Send + 'static>(
        secs: u64,
        f: impl FnOnce() -> T + Send + 'static,
    ) -> Option<T> {
        let (tx, rx) = std::sync::mpsc::channel();
        thread::spawn(move || {
            let _ = tx.send(f());
        });
        rx.recv_timeout(std::time::Duration::from_secs(secs)).ok()
    }

    #[test]
    fn panicking_write_through_adapter_does_not_wedge_the_store() {
        let adapter = PanicsOnce::new();
        let store = Arc::new(
            FerroStore::in_memory()
                .with_sync_adapter(
                    SyncMode::WriteThrough,
                    adapter.clone() as Arc<dyn SyncAdapter>,
                )
                .unwrap(),
        );
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = store.put_doc("docs", "a", &json!({"n": 1}));
        }));
        assert!(unwound.is_err(), "the adapter panic must still propagate");
        // The batch is still queued, and another thread can flush it instead of blocking.
        assert!(store.has_pending_sync());
        let flusher = Arc::clone(&store);
        let flushed = finishes_within(5, move || flusher.sync_flush());
        assert_eq!(flushed.expect("sync_flush wedged").unwrap(), 1);
        assert_eq!(*adapter.commits.lock().unwrap(), vec![1]);
        assert!(!store.has_pending_sync());
        // The write itself was committed locally, and later writes still go through.
        assert_eq!(store.get_doc("docs", "a").unwrap(), Some(json!({"n": 1})));
        store.put_doc("docs", "b", &json!({"n": 2})).unwrap();
        assert_eq!(store.synced_seq(), 2);
    }

    #[test]
    fn panicking_flush_keeps_the_batch_and_releases_waiters() {
        let adapter = PanicsOnce::new();
        let store = Arc::new(
            FerroStore::in_memory()
                .with_sync_adapter(
                    SyncMode::WriteBehind,
                    adapter.clone() as Arc<dyn SyncAdapter>,
                )
                .unwrap(),
        );
        store.put_doc("docs", "a", &json!({"n": 1})).unwrap();
        let panicker = Arc::clone(&store);
        let joined = thread::spawn(move || panicker.sync_flush()).join();
        assert!(joined.is_err(), "the adapter panic must still propagate");
        let flusher = Arc::clone(&store);
        let flushed = finishes_within(5, move || flusher.sync_flush());
        assert_eq!(flushed.expect("sync_flush wedged").unwrap(), 1);
        assert_eq!(*adapter.commits.lock().unwrap(), vec![1]);
    }
}
