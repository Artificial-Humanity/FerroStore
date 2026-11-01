//! Document map, top-level field index, Okapi BM25, and FerroMQ views.
//! Indexes are derived from documents; they are not part of the log.

use super::columnar::{ColumnTable, check_new_table, validate_row_group};
use super::layout::{Op, backend, corrupt};
use super::vector::{self, Fusion, KnowledgeHit, VectorCollection};
use fl_core::StoreError;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};

const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;

#[derive(Clone, Debug, Default)]
pub(super) struct DocState {
    pub(super) docs: BTreeMap<(String, String), Value>,
    pub(super) tables: BTreeMap<String, ColumnTable>,
    pub(super) vectors: BTreeMap<String, VectorCollection>,
    pub(super) last_seq: u64,
}

impl DocState {
    pub(super) fn at_seq(seq: u64) -> Self {
        Self {
            last_seq: seq,
            ..Self::default()
        }
    }
}

/// One durable FerroMQ message.
#[derive(Debug, Clone, PartialEq)]
pub struct MqRecord {
    pub seq: u64,
    pub key: String,
    pub payload: Value,
}

pub(super) fn apply_commit(state: &mut DocState, op: Op) -> Result<(), StoreError> {
    commit_ops(state, &[op])
}

/// Apply every op, then advance `last_seq` by one. An empty list writes nothing.
/// `state` is replaced only when every op applies.
pub(super) fn commit_ops(state: &mut DocState, ops: &[Op]) -> Result<(), StoreError> {
    if ops.is_empty() {
        return Ok(());
    }
    *state = stage_ops(state.clone(), ops)?;
    Ok(())
}

/// Owned copy of `state` after every op applies. `last_seq` is one past `state`.
pub(super) fn stage_ops(mut state: DocState, ops: &[Op]) -> Result<DocState, StoreError> {
    let seq = state
        .last_seq
        .checked_add(1)
        .ok_or_else(|| backend("transaction sequence overflow"))?;
    for op in ops {
        apply_op(&mut state, op)?;
    }
    state.last_seq = seq;
    Ok(state)
}

pub(super) fn apply_frames(
    state: &mut DocState,
    snapshot_seq: u64,
    frames: &[(u64, Vec<Op>)],
) -> Result<(), StoreError> {
    let mut previous = None;
    let mut expect = snapshot_seq;
    for (seq, _) in frames {
        if *seq == 0 {
            return Err(corrupt("wal sequence is zero"));
        }
        if let Some(prev) = previous
            && *seq <= prev
        {
            return Err(corrupt(format!(
                "wal sequence is not strictly increasing: previous {prev}, found {seq}"
            )));
        }
        previous = Some(*seq);
        if *seq <= snapshot_seq {
            continue;
        }
        expect = expect
            .checked_add(1)
            .ok_or_else(|| corrupt("transaction sequence overflow"))?;
        if *seq != expect {
            return Err(corrupt(format!(
                "wal sequence gap or duplicate: expected {expect}, found {seq}"
            )));
        }
    }
    for (seq, ops) in frames {
        if *seq <= snapshot_seq {
            continue;
        }
        for op in ops {
            apply_op(state, op)?;
        }
        state.last_seq = *seq;
    }
    Ok(())
}

fn apply_op(state: &mut DocState, op: &Op) -> Result<(), StoreError> {
    match op {
        Op::Put {
            collection,
            id,
            doc,
        } => {
            state
                .docs
                .insert((collection.clone(), id.clone()), doc.clone());
        }
        Op::Delete { collection, id } => {
            state.docs.remove(&(collection.clone(), id.clone()));
        }
        Op::CreateColumnTable { name, columns } => {
            check_new_table(&state.tables, name, columns)?;
            state.tables.insert(
                name.clone(),
                ColumnTable {
                    columns: columns.clone(),
                    groups: Vec::new(),
                },
            );
        }
        Op::AppendColumnGroup { name, bytes } => {
            let Some(table) = state.tables.get_mut(name) else {
                return Err(corrupt(format!("append to missing column table `{name}`")));
            };
            validate_row_group(&table.columns, bytes)?;
            table.groups.push(bytes.clone());
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
            vector::check_create(name, *dim, *storage, *m, *ef_construction, *ef_search)?;
            if state.vectors.contains_key(name) {
                return Err(backend(format!(
                    "vector collection `{name}` already exists"
                )));
            }
            state.vectors.insert(
                name.clone(),
                VectorCollection::empty(*dim, *storage, *metric, *m, *ef_construction, *ef_search),
            );
        }
        Op::UpsertVector {
            collection,
            id,
            bytes,
            metadata,
        } => {
            let Some(col) = state.vectors.get_mut(collection) else {
                return Err(backend(format!("no vector collection `{collection}`")));
            };
            col.upsert(id.clone(), bytes.clone(), metadata.clone())?;
        }
        Op::DeleteVector { collection, id } => {
            let Some(col) = state.vectors.get_mut(collection) else {
                return Err(backend(format!("no vector collection `{collection}`")));
            };
            col.delete_id(id)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn search_hybrid(
    docs: &BTreeMap<(String, String), Value>,
    vectors: &BTreeMap<String, VectorCollection>,
    doc_collection: &str,
    vector_collection: &str,
    query: &str,
    values: &[f32],
    limit: usize,
    fusion: Fusion,
) -> Result<Vec<(String, f64)>, StoreError> {
    vector::validate_fusion(fusion)?;
    let Some(col) = vectors.get(vector_collection) else {
        return Err(backend(format!(
            "no vector collection `{vector_collection}`"
        )));
    };
    vector::prepare_query(col, values)?;
    if limit == 0 {
        return Ok(Vec::new());
    }
    let bm25 = search_bm25_docs(docs, doc_collection, query, limit);
    let found = vector::search(col, values, limit, None)?;
    vector::fuse(&bm25, &found, fusion, limit)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn search_knowledge(
    docs: &BTreeMap<(String, String), Value>,
    vectors: &BTreeMap<String, VectorCollection>,
    doc_collection: &str,
    vector_collection: &str,
    query: &str,
    values: &[f32],
    limit: usize,
    fusion: Fusion,
) -> Result<Vec<KnowledgeHit>, StoreError> {
    let fused = search_hybrid(
        docs,
        vectors,
        doc_collection,
        vector_collection,
        query,
        values,
        limit,
        fusion,
    )?;
    let mut hits = Vec::with_capacity(fused.len());
    for (id, score) in fused {
        let metadata = vectors
            .get(vector_collection)
            .and_then(|col| col.vectors.get(&id))
            .and_then(|record| record.metadata.clone());
        let doc = docs.get(&(doc_collection.to_string(), id.clone())).cloned();
        hits.push(KnowledgeHit {
            id,
            score,
            metadata,
            doc,
        });
    }
    Ok(hits)
}

pub(super) fn query_by_field(
    state: &DocState,
    collection: &str,
    field: &str,
    value: &str,
) -> Vec<String> {
    let mut ids = Vec::new();
    for ((coll, id), doc) in &state.docs {
        if coll != collection {
            continue;
        }
        let Some(object) = doc.as_object() else {
            continue;
        };
        let Some(field_value) = object.get(field) else {
            continue;
        };
        if scalar_key(field_value).as_deref() == Some(value) {
            ids.push(id.clone());
        }
    }
    ids.sort();
    ids
}

fn scalar_key(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(true) => Some("true".to_string()),
        Value::Bool(false) => Some("false".to_string()),
        _ => None,
    }
}

pub(super) fn search_bm25(
    state: &DocState,
    collection: &str,
    query: &str,
    limit: usize,
) -> Vec<(String, f64)> {
    search_bm25_docs(&state.docs, collection, query, limit)
}

fn search_bm25_docs(
    stored: &BTreeMap<(String, String), Value>,
    collection: &str,
    query: &str,
    limit: usize,
) -> Vec<(String, f64)> {
    if limit == 0 || query.is_empty() {
        return Vec::new();
    }
    let query_terms = tokenize(query);
    if query_terms.is_empty() {
        return Vec::new();
    }

    struct RankedDoc {
        id: String,
        tf: HashMap<String, u32>,
        len: usize,
    }

    let mut docs = Vec::new();
    let mut df: HashMap<String, u32> = HashMap::new();
    let mut total_tokens = 0usize;
    for ((coll, id), doc) in stored {
        if coll != collection {
            continue;
        }
        let tokens = doc_tokens(doc);
        total_tokens += tokens.len();
        let mut tf: HashMap<String, u32> = HashMap::new();
        let mut seen = BTreeSet::new();
        for token in tokens {
            let count = tf.entry(token.clone()).or_insert(0);
            *count = count.saturating_add(1);
            seen.insert(token);
        }
        for token in seen {
            let docs_with: &mut u32 = df.entry(token).or_insert(0);
            *docs_with = docs_with.saturating_add(1);
        }
        let len = tf_len(&tf);
        docs.push(RankedDoc {
            id: id.clone(),
            tf,
            len,
        });
    }
    if docs.is_empty() {
        return Vec::new();
    }
    let n = docs.len() as f64;
    let avgdl = total_tokens as f64 / n;
    if avgdl == 0.0 {
        return Vec::new();
    }

    let mut scored = Vec::new();
    for doc in docs {
        let mut score = 0.0;
        let mut hit = false;
        for term in &query_terms {
            let tf = f64::from(*doc.tf.get(term).unwrap_or(&0));
            if tf > 0.0 {
                hit = true;
            }
            let df = f64::from(*df.get(term).unwrap_or(&0));
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            let denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * doc.len as f64 / avgdl);
            score += idf * (tf * (BM25_K1 + 1.0)) / denom;
        }
        if hit {
            scored.push((doc.id, score));
        }
    }
    scored.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    scored.truncate(limit);
    scored
}

fn tf_len(tf: &HashMap<String, u32>) -> usize {
    tf.values().map(|count| *count as usize).sum()
}

fn doc_tokens(value: &Value) -> Vec<String> {
    let mut tokens = Vec::new();
    walk_strings(value, &mut tokens);
    tokens
}

fn walk_strings(value: &Value, tokens: &mut Vec<String>) {
    match value {
        Value::String(text) => tokens.extend(tokenize(text)),
        Value::Array(items) => {
            for item in items {
                walk_strings(item, tokens);
            }
        }
        Value::Object(map) => {
            for child in map.values() {
                walk_strings(child, tokens);
            }
        }
        _ => {}
    }
}

pub(super) fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for lower in ch.to_lowercase() {
                current.push(lower);
            }
        } else if !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// Document id of the per-topic FerroMQ high-water in `mq_hw:{topic}`.
///
/// Checkpoint keeps every live document and drops puts whose rows were
/// deleted. The high-water is a separate document so deleting `mq:{topic}/{id}`
/// cannot reuse a sequence.
pub(super) const MQ_HIGH_WATER_ID: &str = "high_water";

pub(super) fn mq_message_collection(topic: &str) -> String {
    format!("mq:{topic}")
}

pub(super) fn mq_hw_collection(topic: &str) -> String {
    format!("mq_hw:{topic}")
}

pub(super) fn next_mq_seq(
    docs: &BTreeMap<(String, String), Value>,
    topic: &str,
) -> Result<u64, StoreError> {
    // Absent on files written before the high-water existed: one past the
    // highest message still stored. A present high-water is not lowered by
    // deletes, so it wins over the remaining messages.
    let floor = match mq_high_water(docs, topic)? {
        Some(seq) => seq,
        None => max_stored_mq_seq(docs, topic),
    };
    floor
        .checked_add(1)
        .ok_or_else(|| backend("mq sequence overflow"))
}

fn mq_high_water(
    docs: &BTreeMap<(String, String), Value>,
    topic: &str,
) -> Result<Option<u64>, StoreError> {
    let Some(doc) = docs.get(&(mq_hw_collection(topic), MQ_HIGH_WATER_ID.to_string())) else {
        return Ok(None);
    };
    let Some(seq) = doc.get("seq").and_then(Value::as_u64) else {
        return Err(corrupt(format!("mq high-water for {topic} has no seq")));
    };
    Ok(Some(seq))
}

fn max_stored_mq_seq(docs: &BTreeMap<(String, String), Value>, topic: &str) -> u64 {
    let collection = mq_message_collection(topic);
    let mut max = 0u64;
    for ((coll, id), doc) in docs {
        if coll != &collection {
            continue;
        }
        let seq = doc
            .get("seq")
            .and_then(Value::as_u64)
            .or_else(|| id.parse().ok());
        if let Some(seq) = seq {
            max = max.max(seq);
        }
    }
    max
}

pub(super) fn unread_mq(
    state: &DocState,
    topic: &str,
    group: &str,
    limit: usize,
) -> Result<Vec<MqRecord>, StoreError> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let acked = acked_seqs(state, topic, group);
    let mut records = load_mq(state, topic)?;
    records.retain(|record| !acked.contains(&record.seq));
    records.truncate(limit);
    Ok(records)
}

fn load_mq(state: &DocState, topic: &str) -> Result<Vec<MqRecord>, StoreError> {
    let collection = mq_message_collection(topic);
    let mut records = Vec::new();
    for ((coll, id), doc) in &state.docs {
        if coll != &collection {
            continue;
        }
        let Some(object) = doc.as_object() else {
            return Err(corrupt(format!("mq document {id} is not an object")));
        };
        let seq = object
            .get("seq")
            .and_then(Value::as_u64)
            .ok_or_else(|| corrupt(format!("mq document {id} has no seq")))?;
        let key = object
            .get("key")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let payload = object.get("payload").cloned().unwrap_or(Value::Null);
        records.push(MqRecord { seq, key, payload });
    }
    records.sort_by(|left, right| {
        left.seq
            .cmp(&right.seq)
            .then_with(|| left.key.cmp(&right.key))
    });
    Ok(records)
}

fn acked_seqs(state: &DocState, topic: &str, group: &str) -> BTreeSet<u64> {
    let collection = format!("mq_ack:{topic}:{group}");
    let mut seqs = BTreeSet::new();
    for ((coll, id), doc) in &state.docs {
        if coll != &collection {
            continue;
        }
        if let Some(seq) = doc.get("seq").and_then(Value::as_u64) {
            seqs.insert(seq);
        } else if let Ok(seq) = id.parse::<u64>() {
            seqs.insert(seq);
        }
    }
    seqs
}
