//! Little-endian `.fls` layout: superblocks, copy-on-write snapshots, WAL frames.

use super::columnar::{
    ColumnDef, ColumnTable, check_new_table, type_from_tag, type_tag, validate_row_group,
};
use super::vector::{
    self, VectorCollection, VectorMetric, VectorRecord, VectorStorage, metric_from_tag, metric_tag,
    storage_from_tag, storage_tag,
};
use fl_core::StoreError;
use serde_json::Value;
use std::collections::BTreeMap;

pub(super) const SLOT_LEN: usize = 4096;
pub(super) const HEADER_BYTES: u64 = 8192;
const SUPER_MAGIC: &[u8; 8] = b"FERRSTR1";
const SNAP_MAGIC: &[u8; 8] = b"FERRSNAP";
const FORMAT_VERSION: u32 = 1;
const SNAPSHOT_VERSION_DOCS: u32 = 1;
const SNAPSHOT_VERSION_COLUMNAR: u32 = 2;
const SNAPSHOT_VERSION_VECTOR: u32 = 3;
const TAG_PUT: u8 = 1;
const TAG_DELETE: u8 = 2;
const TAG_CREATE_COLUMN_TABLE: u8 = 3;
const TAG_APPEND_COLUMN_GROUP: u8 = 4;
const TAG_CREATE_VECTOR_COLLECTION: u8 = 5;
const TAG_UPSERT_VECTOR: u8 = 6;
const TAG_DELETE_VECTOR: u8 = 7;

pub(super) fn backend(err: impl std::fmt::Display) -> StoreError {
    StoreError::Backend(err.to_string())
}

pub(super) fn corrupt(detail: impl std::fmt::Display) -> StoreError {
    StoreError::Backend(format!("corrupt ferrostore data: {detail}"))
}

/// CRC-32/ISO-HDLC (poly `0xEDB88320`, init and xorout `0xFFFFFFFF`).
pub(super) fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if crc & 1 == 1 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Superblock {
    pub(super) epoch: u64,
    pub(super) snapshot_offset: u64,
    pub(super) snapshot_length: u64,
    pub(super) wal_offset: u64,
    pub(super) wal_length: u64,
    pub(super) snapshot_seq: u64,
}

pub(super) fn encode_superblock(sb: &Superblock) -> [u8; SLOT_LEN] {
    let mut buf = [0u8; SLOT_LEN];
    buf[0..8].copy_from_slice(SUPER_MAGIC);
    buf[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    buf[12..20].copy_from_slice(&sb.epoch.to_le_bytes());
    buf[24..32].copy_from_slice(&sb.snapshot_offset.to_le_bytes());
    buf[32..40].copy_from_slice(&sb.snapshot_length.to_le_bytes());
    buf[40..48].copy_from_slice(&sb.wal_offset.to_le_bytes());
    buf[48..56].copy_from_slice(&sb.wal_length.to_le_bytes());
    buf[56..64].copy_from_slice(&sb.snapshot_seq.to_le_bytes());
    let crc = crc32(&buf);
    buf[20..24].copy_from_slice(&crc.to_le_bytes());
    buf
}

pub(super) fn parse_superblock(buf: &[u8]) -> Option<Superblock> {
    if buf.len() != SLOT_LEN || buf.get(..8) != Some(SUPER_MAGIC.as_slice()) {
        return None;
    }
    let version = u32_at(buf, 8)?;
    if version != FORMAT_VERSION {
        return None;
    }
    let stored = u32_at(buf, 20)?;
    let mut zeroed = [0u8; SLOT_LEN];
    zeroed.copy_from_slice(buf);
    zeroed[20..24].fill(0);
    if crc32(&zeroed) != stored {
        return None;
    }
    if buf[64..].iter().any(|byte| *byte != 0) {
        return None;
    }
    let epoch = u64_at(buf, 12)?;
    let snapshot_offset = u64_at(buf, 24)?;
    let snapshot_length = u64_at(buf, 32)?;
    let wal_offset = u64_at(buf, 40)?;
    let wal_length = u64_at(buf, 48)?;
    if epoch == 0
        || snapshot_offset < HEADER_BYTES
        || wal_offset < HEADER_BYTES
        || snapshot_offset.checked_add(snapshot_length).is_none()
        || wal_offset.checked_add(wal_length).is_none()
    {
        return None;
    }
    Some(Superblock {
        epoch,
        snapshot_offset,
        snapshot_length,
        wal_offset,
        wal_length,
        snapshot_seq: u64_at(buf, 56)?,
    })
}

#[derive(Debug, Clone)]
pub enum Op {
    Put {
        collection: String,
        id: String,
        doc: Value,
    },
    Delete {
        collection: String,
        id: String,
    },
    CreateColumnTable {
        name: String,
        columns: Vec<ColumnDef>,
    },
    AppendColumnGroup {
        name: String,
        bytes: Vec<u8>,
    },
    CreateVectorCollection {
        name: String,
        dim: u32,
        storage: VectorStorage,
        metric: VectorMetric,
        m: u32,
        ef_construction: u32,
        ef_search: u32,
    },
    UpsertVector {
        collection: String,
        id: String,
        bytes: Vec<u8>,
        metadata: Option<Value>,
    },
    DeleteVector {
        collection: String,
        id: String,
    },
}

pub(super) fn encode_frame(seq: u64, ops: &[Op]) -> Result<Vec<u8>, StoreError> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&seq.to_le_bytes());
    let count =
        u32::try_from(ops.len()).map_err(|_| backend("too many operations in one frame"))?;
    payload.extend_from_slice(&count.to_le_bytes());
    for op in ops {
        encode_op(&mut payload, op)?;
    }
    let len = u32::try_from(payload.len()).map_err(|_| backend("wal payload exceeds 4 GiB"))?;
    let mut frame = Vec::with_capacity(8 + payload.len());
    frame.extend_from_slice(&len.to_le_bytes());
    frame.extend_from_slice(&crc32(&payload).to_le_bytes());
    frame.extend_from_slice(&payload);
    Ok(frame)
}

fn encode_op(buf: &mut Vec<u8>, op: &Op) -> Result<(), StoreError> {
    match op {
        Op::Put {
            collection,
            id,
            doc,
        } => {
            buf.push(TAG_PUT);
            write_lp(buf, collection.as_bytes())?;
            write_lp(buf, id.as_bytes())?;
            let raw =
                serde_json::to_vec(doc).map_err(|err| backend(format!("json encode: {err}")))?;
            write_lp(buf, &raw)?;
        }
        Op::Delete { collection, id } => {
            buf.push(TAG_DELETE);
            write_lp(buf, collection.as_bytes())?;
            write_lp(buf, id.as_bytes())?;
        }
        Op::CreateColumnTable { name, columns } => {
            buf.push(TAG_CREATE_COLUMN_TABLE);
            write_lp(buf, name.as_bytes())?;
            let count = u32::try_from(columns.len()).map_err(|_| backend("too many columns"))?;
            buf.extend_from_slice(&count.to_le_bytes());
            for column in columns {
                write_lp(buf, column.name.as_bytes())?;
                buf.push(type_tag(column.ty));
            }
        }
        Op::AppendColumnGroup { name, bytes } => {
            buf.push(TAG_APPEND_COLUMN_GROUP);
            write_lp(buf, name.as_bytes())?;
            write_lp(buf, bytes)?;
        }
        Op::CreateVectorCollection {
            name,
            dim,
            storage,
            metric,
            m,
            ef_construction,
            ef_search,
        } => {
            buf.push(TAG_CREATE_VECTOR_COLLECTION);
            write_lp(buf, name.as_bytes())?;
            buf.extend_from_slice(&dim.to_le_bytes());
            buf.push(storage_tag(*storage));
            buf.push(metric_tag(*metric));
            buf.extend_from_slice(&m.to_le_bytes());
            buf.extend_from_slice(&ef_construction.to_le_bytes());
            buf.extend_from_slice(&ef_search.to_le_bytes());
        }
        Op::UpsertVector {
            collection,
            id,
            bytes,
            metadata,
        } => {
            buf.push(TAG_UPSERT_VECTOR);
            write_lp(buf, collection.as_bytes())?;
            write_lp(buf, id.as_bytes())?;
            write_lp(buf, bytes)?;
            write_metadata(buf, metadata)?;
        }
        Op::DeleteVector { collection, id } => {
            buf.push(TAG_DELETE_VECTOR);
            write_lp(buf, collection.as_bytes())?;
            write_lp(buf, id.as_bytes())?;
        }
    }
    Ok(())
}

fn write_metadata(buf: &mut Vec<u8>, metadata: &Option<Value>) -> Result<(), StoreError> {
    match metadata {
        None => buf.push(0),
        Some(value) => {
            buf.push(1);
            let raw =
                serde_json::to_vec(value).map_err(|err| backend(format!("json encode: {err}")))?;
            write_lp(buf, &raw)?;
        }
    }
    Ok(())
}

fn write_lp(buf: &mut Vec<u8>, bytes: &[u8]) -> Result<(), StoreError> {
    let len = u32::try_from(bytes.len()).map_err(|_| backend("length does not fit in u32"))?;
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(bytes);
    Ok(())
}

pub(super) struct Snapshot {
    pub(super) seq: u64,
    pub(super) docs: BTreeMap<(String, String), Value>,
    pub(super) tables: BTreeMap<String, ColumnTable>,
    pub(super) vectors: BTreeMap<String, VectorCollection>,
}

pub(super) fn encode_snapshot(
    seq: u64,
    docs: &BTreeMap<(String, String), Value>,
) -> Result<Vec<u8>, StoreError> {
    let mut body = Vec::new();
    body.extend_from_slice(SNAP_MAGIC);
    body.extend_from_slice(&SNAPSHOT_VERSION_DOCS.to_le_bytes());
    body.extend_from_slice(&seq.to_le_bytes());
    let count =
        u32::try_from(docs.len()).map_err(|_| backend("snapshot has too many documents"))?;
    body.extend_from_slice(&count.to_le_bytes());
    for ((collection, id), doc) in docs {
        write_lp(&mut body, collection.as_bytes())?;
        write_lp(&mut body, id.as_bytes())?;
        let raw = serde_json::to_vec(doc).map_err(|err| backend(format!("json encode: {err}")))?;
        write_lp(&mut body, &raw)?;
    }
    let crc = crc32(&body);
    body.extend_from_slice(&crc.to_le_bytes());
    Ok(body)
}

/// Version 2: the version-1 document section, then column tables in name order.
/// Callers use this only when `tables` is non-empty. An empty map stays version 1
/// via [`encode_snapshot`], which is byte-identical to a document-only snapshot.
pub(super) fn encode_snapshot_v2(
    seq: u64,
    docs: &BTreeMap<(String, String), Value>,
    tables: &BTreeMap<String, ColumnTable>,
) -> Result<Vec<u8>, StoreError> {
    let mut body = Vec::new();
    body.extend_from_slice(SNAP_MAGIC);
    body.extend_from_slice(&SNAPSHOT_VERSION_COLUMNAR.to_le_bytes());
    body.extend_from_slice(&seq.to_le_bytes());
    let count =
        u32::try_from(docs.len()).map_err(|_| backend("snapshot has too many documents"))?;
    body.extend_from_slice(&count.to_le_bytes());
    for ((collection, id), doc) in docs {
        write_lp(&mut body, collection.as_bytes())?;
        write_lp(&mut body, id.as_bytes())?;
        let raw = serde_json::to_vec(doc).map_err(|err| backend(format!("json encode: {err}")))?;
        write_lp(&mut body, &raw)?;
    }
    let table_count =
        u32::try_from(tables.len()).map_err(|_| backend("snapshot has too many column tables"))?;
    body.extend_from_slice(&table_count.to_le_bytes());
    for (name, table) in tables {
        write_lp(&mut body, name.as_bytes())?;
        let col_count =
            u32::try_from(table.columns.len()).map_err(|_| backend("too many columns"))?;
        body.extend_from_slice(&col_count.to_le_bytes());
        for column in &table.columns {
            write_lp(&mut body, column.name.as_bytes())?;
            body.push(type_tag(column.ty));
        }
        let group_count =
            u32::try_from(table.groups.len()).map_err(|_| backend("too many row groups"))?;
        body.extend_from_slice(&group_count.to_le_bytes());
        for group in &table.groups {
            write_lp(&mut body, group)?;
        }
    }
    let crc = crc32(&body);
    body.extend_from_slice(&crc.to_le_bytes());
    Ok(body)
}

/// Version 3: the version-1 document section, the version-2 table section
/// (`table_count` may be 0), then one or more vector collections.
pub(super) fn encode_snapshot_v3(
    seq: u64,
    docs: &BTreeMap<(String, String), Value>,
    tables: &BTreeMap<String, ColumnTable>,
    vectors: &BTreeMap<String, VectorCollection>,
) -> Result<Vec<u8>, StoreError> {
    if vectors.is_empty() {
        return Err(backend("snapshot has no vector collections"));
    }
    for (name, collection) in vectors {
        vector::validate_graph(name, collection)?;
    }
    let mut body = Vec::new();
    body.extend_from_slice(SNAP_MAGIC);
    body.extend_from_slice(&SNAPSHOT_VERSION_VECTOR.to_le_bytes());
    body.extend_from_slice(&seq.to_le_bytes());
    let count =
        u32::try_from(docs.len()).map_err(|_| backend("snapshot has too many documents"))?;
    body.extend_from_slice(&count.to_le_bytes());
    for ((collection, id), doc) in docs {
        write_lp(&mut body, collection.as_bytes())?;
        write_lp(&mut body, id.as_bytes())?;
        let raw = serde_json::to_vec(doc).map_err(|err| backend(format!("json encode: {err}")))?;
        write_lp(&mut body, &raw)?;
    }
    let table_count =
        u32::try_from(tables.len()).map_err(|_| backend("snapshot has too many column tables"))?;
    body.extend_from_slice(&table_count.to_le_bytes());
    for (name, table) in tables {
        write_column_table(&mut body, name, table)?;
    }
    let collection_count = u32::try_from(vectors.len())
        .map_err(|_| backend("snapshot has too many vector collections"))?;
    body.extend_from_slice(&collection_count.to_le_bytes());
    for (name, collection) in vectors {
        write_vector_collection(&mut body, name, collection)?;
    }
    let crc = crc32(&body);
    body.extend_from_slice(&crc.to_le_bytes());
    Ok(body)
}

fn write_column_table(
    body: &mut Vec<u8>,
    name: &str,
    table: &ColumnTable,
) -> Result<(), StoreError> {
    write_lp(body, name.as_bytes())?;
    let col_count = u32::try_from(table.columns.len()).map_err(|_| backend("too many columns"))?;
    body.extend_from_slice(&col_count.to_le_bytes());
    for column in &table.columns {
        write_lp(body, column.name.as_bytes())?;
        body.push(type_tag(column.ty));
    }
    let group_count =
        u32::try_from(table.groups.len()).map_err(|_| backend("too many row groups"))?;
    body.extend_from_slice(&group_count.to_le_bytes());
    for group in &table.groups {
        write_lp(body, group)?;
    }
    Ok(())
}

fn write_vector_collection(
    body: &mut Vec<u8>,
    name: &str,
    collection: &VectorCollection,
) -> Result<(), StoreError> {
    write_lp(body, name.as_bytes())?;
    body.extend_from_slice(&collection.dim.to_le_bytes());
    body.push(storage_tag(collection.storage));
    body.push(metric_tag(collection.metric));
    body.extend_from_slice(&collection.m.to_le_bytes());
    body.extend_from_slice(&collection.ef_construction.to_le_bytes());
    body.extend_from_slice(&collection.ef_search.to_le_bytes());
    let count = u32::try_from(collection.vectors.len())
        .map_err(|_| backend("too many vectors in one collection"))?;
    body.extend_from_slice(&count.to_le_bytes());
    for (id, record) in &collection.vectors {
        write_lp(body, id.as_bytes())?;
        write_lp(body, &record.bytes)?;
        write_metadata(body, &record.metadata)?;
    }
    match &collection.entry {
        None => body.push(0),
        Some(id) => {
            body.push(1);
            write_lp(body, id.as_bytes())?;
        }
    }
    for record in collection.vectors.values() {
        body.push(record.level);
        for neighbors in &record.neighbors {
            let ncount =
                u32::try_from(neighbors.len()).map_err(|_| backend("too many vector neighbors"))?;
            body.extend_from_slice(&ncount.to_le_bytes());
            for neighbor in neighbors {
                write_lp(body, neighbor.as_bytes())?;
            }
        }
    }
    Ok(())
}

pub(super) fn decode_snapshot(buf: &[u8]) -> Result<Snapshot, StoreError> {
    if buf.len() < 4 {
        return Err(corrupt("snapshot is shorter than its crc"));
    }
    let split = buf.len() - 4;
    let (body, crc_bytes) = buf.split_at(split);
    let mut stored_bytes = [0u8; 4];
    stored_bytes.copy_from_slice(crc_bytes);
    if crc32(body) != u32::from_le_bytes(stored_bytes) {
        return Err(corrupt("snapshot crc mismatch"));
    }
    if body.len() < 24 {
        return Err(corrupt("snapshot header is truncated"));
    }
    let mut cur = Cur::new(body);
    let magic = cur.take(8)?;
    if magic != SNAP_MAGIC {
        return Err(corrupt("snapshot magic mismatch"));
    }
    let version = cur.u32()?;
    if version != SNAPSHOT_VERSION_DOCS
        && version != SNAPSHOT_VERSION_COLUMNAR
        && version != SNAPSHOT_VERSION_VECTOR
    {
        return Err(corrupt(format!(
            "snapshot version {version} is not supported"
        )));
    }
    let seq = cur.u64()?;
    let count = cur.u32()?;
    if usize::try_from(count).unwrap_or(usize::MAX) > cur.remaining() {
        return Err(corrupt("snapshot count exceeds the blob"));
    }
    let mut docs = BTreeMap::new();
    for _ in 0..count {
        let collection = cur.lp_str()?;
        let id = cur.lp_str()?;
        let raw = cur.lp()?;
        let doc =
            serde_json::from_slice(raw).map_err(|_| corrupt("snapshot document is not json"))?;
        if docs.insert((collection, id), doc).is_some() {
            return Err(corrupt("snapshot repeats a document id"));
        }
    }
    let mut tables = BTreeMap::new();
    if version == SNAPSHOT_VERSION_COLUMNAR || version == SNAPSHOT_VERSION_VECTOR {
        tables = decode_column_tables(&mut cur, version == SNAPSHOT_VERSION_COLUMNAR)?;
    }
    let mut vectors = BTreeMap::new();
    if version == SNAPSHOT_VERSION_VECTOR {
        vectors = decode_vector_collections(&mut cur)?;
    }
    cur.finish()?;
    Ok(Snapshot {
        seq,
        docs,
        tables,
        vectors,
    })
}

fn decode_column_tables(
    cur: &mut Cur<'_>,
    require_nonempty: bool,
) -> Result<BTreeMap<String, ColumnTable>, StoreError> {
    let table_count = cur.u32()?;
    if require_nonempty && table_count == 0 {
        return Err(corrupt("snapshot has no column tables"));
    }
    if usize::try_from(table_count).unwrap_or(usize::MAX) > cur.remaining() {
        return Err(corrupt("snapshot table count exceeds the blob"));
    }
    let mut tables = BTreeMap::new();
    for _ in 0..table_count {
        let name = cur.lp_str()?;
        let col_count = cur.u32()?;
        if usize::try_from(col_count).unwrap_or(usize::MAX) > cur.remaining() {
            return Err(corrupt("snapshot column count exceeds the blob"));
        }
        let mut columns = Vec::with_capacity(usize::try_from(col_count).unwrap_or(0));
        for _ in 0..col_count {
            let column_name = cur.lp_str()?;
            let ty = type_from_tag(cur.u8()?)?;
            columns.push(ColumnDef {
                name: column_name,
                ty,
            });
        }
        let group_count = cur.u32()?;
        if usize::try_from(group_count).unwrap_or(usize::MAX) > cur.remaining() {
            return Err(corrupt("snapshot row-group count exceeds the blob"));
        }
        let mut groups = Vec::with_capacity(usize::try_from(group_count).unwrap_or(0));
        for _ in 0..group_count {
            groups.push(cur.lp()?.to_vec());
        }
        if tables.contains_key(&name) {
            return Err(corrupt(format!("snapshot repeats column table `{name}`")));
        }
        check_new_table(&tables, &name, &columns)?;
        for group in &groups {
            validate_row_group(&columns, group)?;
        }
        tables.insert(name, ColumnTable { columns, groups });
    }
    Ok(tables)
}

fn decode_vector_collections(
    cur: &mut Cur<'_>,
) -> Result<BTreeMap<String, VectorCollection>, StoreError> {
    let collection_count = cur.u32()?;
    if collection_count == 0 {
        return Err(corrupt("snapshot has no vector collections"));
    }
    if usize::try_from(collection_count).unwrap_or(usize::MAX) > cur.remaining() {
        return Err(corrupt("snapshot vector count exceeds the blob"));
    }
    let mut collections = BTreeMap::new();
    for _ in 0..collection_count {
        let name = cur.lp_str()?;
        let dim = cur.u32()?;
        let storage = storage_from_tag(cur.u8()?)?;
        let metric = metric_from_tag(cur.u8()?)?;
        let m = cur.u32()?;
        let ef_construction = cur.u32()?;
        let ef_search = cur.u32()?;
        let vector_count = cur.u32()?;
        if usize::try_from(vector_count).unwrap_or(usize::MAX) > cur.remaining() {
            return Err(corrupt("snapshot vector count exceeds the blob"));
        }
        let mut vectors = BTreeMap::new();
        for _ in 0..vector_count {
            let id = cur.lp_str()?;
            let bytes = cur.lp()?.to_vec();
            let metadata = read_metadata(cur)?;
            if vectors
                .insert(
                    id.clone(),
                    VectorRecord {
                        bytes,
                        metadata,
                        level: 0,
                        neighbors: Vec::new(),
                    },
                )
                .is_some()
            {
                return Err(corrupt(format!("snapshot repeats vector id `{id}`")));
            }
        }
        let entry_flag = cur.u8()?;
        let entry = match entry_flag {
            0 => None,
            1 => Some(cur.lp_str()?),
            _ => return Err(corrupt(format!("unknown vector entry flag {entry_flag}"))),
        };
        for record in vectors.values_mut() {
            let level = cur.u8()?;
            if usize::from(level) > cur.remaining() {
                return Err(corrupt("snapshot vector level exceeds the blob"));
            }
            let mut neighbors = Vec::with_capacity(usize::from(level) + 1);
            for _layer in 0..=level {
                let ncount = cur.u32()?;
                if usize::try_from(ncount).unwrap_or(usize::MAX) > cur.remaining() {
                    return Err(corrupt("snapshot neighbor count exceeds the blob"));
                }
                let mut nids = Vec::with_capacity(usize::try_from(ncount).unwrap_or(0));
                for _ in 0..ncount {
                    nids.push(cur.lp_str()?);
                }
                neighbors.push(nids);
            }
            record.level = level;
            record.neighbors = neighbors;
        }
        if collections.contains_key(&name) {
            return Err(corrupt(format!(
                "snapshot repeats vector collection `{name}`"
            )));
        }
        let collection = VectorCollection {
            dim,
            storage,
            metric,
            m,
            ef_construction,
            ef_search,
            vectors,
            entry,
        };
        vector::validate_graph(&name, &collection)?;
        collections.insert(name, collection);
    }
    Ok(collections)
}

fn read_metadata(cur: &mut Cur<'_>) -> Result<Option<Value>, StoreError> {
    let flag = cur.u8()?;
    match flag {
        0 => Ok(None),
        1 => {
            let raw = cur.lp()?;
            let value =
                serde_json::from_slice(raw).map_err(|_| corrupt("vector metadata is not json"))?;
            Ok(Some(value))
        }
        _ => Err(corrupt(format!("unknown vector metadata flag {flag}"))),
    }
}

/// Decode committed WAL bytes. A torn tail is dropped; a bad CRC with a
/// following complete frame is corruption.
pub(super) fn decode_wal(buf: &[u8]) -> Result<Vec<(u64, Vec<Op>)>, StoreError> {
    let mut offset = 0usize;
    let mut frames = Vec::new();
    while offset < buf.len() {
        if buf.len() - offset < 8 {
            break;
        }
        let Some(len) = u32_at(buf, offset) else {
            break;
        };
        let Some(stored) = u32_at(buf, offset + 4) else {
            break;
        };
        let len = len as usize;
        let Some(payload_end) = offset
            .checked_add(8)
            .and_then(|start| start.checked_add(len))
        else {
            break;
        };
        if payload_end > buf.len() {
            break;
        }
        let payload = &buf[offset + 8..payload_end];
        if crc32(payload) != stored {
            if payload_end == buf.len() || !frame_fits(buf, payload_end) {
                break;
            }
            return Err(corrupt(
                "wal frame crc does not match and another frame follows",
            ));
        }
        frames.push(decode_payload(payload)?);
        offset = payload_end;
    }
    Ok(frames)
}

fn frame_fits(buf: &[u8], offset: usize) -> bool {
    if buf.len().saturating_sub(offset) < 8 {
        return false;
    }
    let Some(len) = u32_at(buf, offset) else {
        return false;
    };
    match offset
        .checked_add(8)
        .and_then(|start| start.checked_add(len as usize))
    {
        Some(end) => end <= buf.len(),
        None => false,
    }
}

fn decode_payload(buf: &[u8]) -> Result<(u64, Vec<Op>), StoreError> {
    let mut cur = Cur::new(buf);
    let seq = cur.u64()?;
    let count = cur.u32()?;
    if usize::try_from(count).unwrap_or(usize::MAX) > cur.remaining() {
        return Err(corrupt("op count exceeds the wal payload"));
    }
    let mut ops = Vec::new();
    for _ in 0..count {
        let tag = cur.u8()?;
        let op = match tag {
            TAG_PUT => {
                let collection = cur.lp_str()?;
                let id = cur.lp_str()?;
                let raw = cur.lp()?;
                let doc =
                    serde_json::from_slice(raw).map_err(|_| corrupt("document is not json"))?;
                Op::Put {
                    collection,
                    id,
                    doc,
                }
            }
            TAG_DELETE => {
                let collection = cur.lp_str()?;
                let id = cur.lp_str()?;
                Op::Delete { collection, id }
            }
            TAG_CREATE_COLUMN_TABLE => {
                let name = cur.lp_str()?;
                let col_count = cur.u32()?;
                if usize::try_from(col_count).unwrap_or(usize::MAX) > cur.remaining() {
                    return Err(corrupt("column count exceeds the wal payload"));
                }
                let mut columns = Vec::with_capacity(usize::try_from(col_count).unwrap_or(0));
                for _ in 0..col_count {
                    let column_name = cur.lp_str()?;
                    let ty = type_from_tag(cur.u8()?)?;
                    columns.push(ColumnDef {
                        name: column_name,
                        ty,
                    });
                }
                Op::CreateColumnTable { name, columns }
            }
            TAG_APPEND_COLUMN_GROUP => {
                let name = cur.lp_str()?;
                let bytes = cur.lp()?.to_vec();
                Op::AppendColumnGroup { name, bytes }
            }
            TAG_CREATE_VECTOR_COLLECTION => {
                let name = cur.lp_str()?;
                let dim = cur.u32()?;
                let storage = storage_from_tag(cur.u8()?)?;
                let metric = metric_from_tag(cur.u8()?)?;
                let m = cur.u32()?;
                let ef_construction = cur.u32()?;
                let ef_search = cur.u32()?;
                Op::CreateVectorCollection {
                    name,
                    dim,
                    storage,
                    metric,
                    m,
                    ef_construction,
                    ef_search,
                }
            }
            TAG_UPSERT_VECTOR => {
                let collection = cur.lp_str()?;
                let id = cur.lp_str()?;
                let bytes = cur.lp()?.to_vec();
                let metadata = read_metadata(&mut cur)?;
                Op::UpsertVector {
                    collection,
                    id,
                    bytes,
                    metadata,
                }
            }
            TAG_DELETE_VECTOR => {
                let collection = cur.lp_str()?;
                let id = cur.lp_str()?;
                Op::DeleteVector { collection, id }
            }
            _ => return Err(corrupt(format!("unknown wal op tag {tag}"))),
        };
        ops.push(op);
    }
    cur.finish()?;
    Ok((seq, ops))
}

struct Cur<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Cur<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, at: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.at)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], StoreError> {
        let end = self
            .at
            .checked_add(n)
            .ok_or_else(|| corrupt("length overflow"))?;
        if end > self.buf.len() {
            return Err(corrupt("truncated record"));
        }
        let bytes = &self.buf[self.at..end];
        self.at = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, StoreError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, StoreError> {
        let bytes = self.take(4)?;
        let mut arr = [0u8; 4];
        arr.copy_from_slice(bytes);
        Ok(u32::from_le_bytes(arr))
    }

    fn u64(&mut self) -> Result<u64, StoreError> {
        let bytes = self.take(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(arr))
    }

    fn lp(&mut self) -> Result<&'a [u8], StoreError> {
        let n = usize::try_from(self.u32()?).map_err(|_| corrupt("length does not fit"))?;
        if n > self.remaining() {
            return Err(corrupt("length prefix exceeds the record"));
        }
        self.take(n)
    }

    fn lp_str(&mut self) -> Result<String, StoreError> {
        let bytes = self.lp()?;
        String::from_utf8(bytes.to_vec()).map_err(|_| corrupt("text is not utf-8"))
    }

    fn finish(self) -> Result<(), StoreError> {
        if self.at != self.buf.len() {
            return Err(corrupt("trailing bytes inside a committed record"));
        }
        Ok(())
    }
}

fn u32_at(buf: &[u8], offset: usize) -> Option<u32> {
    let bytes = buf.get(offset..offset + 4)?;
    let mut arr = [0u8; 4];
    arr.copy_from_slice(bytes);
    Some(u32::from_le_bytes(arr))
}

fn u64_at(buf: &[u8], offset: usize) -> Option<u64> {
    let bytes = buf.get(offset..offset + 8)?;
    let mut arr = [0u8; 8];
    arr.copy_from_slice(bytes);
    Some(u64::from_le_bytes(arr))
}
