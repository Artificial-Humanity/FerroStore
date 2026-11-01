//! Document map, top-level field index, Okapi BM25, and FerroMQ views.
//! Indexes are derived from documents; they are not part of the log.

use super::Txn;
use super::columnar::{ColumnDef, ColumnTable, check_new_table, validate_row_group};
use super::layout::{Op, backend, corrupt};
use super::vector::{self, Fusion, KnowledgeHit, VectorCollection};
use fl_core::StoreError;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;

/// One document collection: id -> document.
pub(super) type Collection = BTreeMap<String, Arc<Value>>;
/// One collection's staged puts (`Some`) and deletes (`None`), in id order.
type DocChanges = Vec<(String, Option<Arc<Value>>)>;
/// Column tables by name.
pub(super) type Tables = BTreeMap<String, Arc<ColumnTable>>;
/// Vector collections by name.
pub(super) type Vectors = BTreeMap<String, Arc<VectorCollection>>;

/// Every document, as two levels of `Arc`: collection -> id -> document.
///
/// A commit copies the outer map only when another holder shares it, and then
/// copies only the collections it touches. A collection with no document is
/// not kept, so two states with the same documents compare equal. Iteration is
/// in `(collection, id)` order, the order the snapshot writes.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Docs(BTreeMap<String, Arc<Collection>>);

impl Docs {
    pub(super) fn get(&self, collection: &str, id: &str) -> Option<&Value> {
        self.0.get(collection)?.get(id).map(|doc| &**doc)
    }

    #[cfg(test)]
    pub(super) fn contains(&self, collection: &str, id: &str) -> bool {
        self.get(collection, id).is_some()
    }

    /// `(id, document)` in `collection`, in id order.
    pub(super) fn in_collection<'a>(
        &'a self,
        collection: &str,
    ) -> impl Iterator<Item = (&'a String, &'a Value)> + 'a {
        self.0
            .get(collection)
            .into_iter()
            .flat_map(|docs| docs.iter().map(|(id, doc)| (id, &**doc)))
    }

    /// Every `(collection, id, document)`, in `(collection, id)` order.
    pub(super) fn iter(&self) -> impl Iterator<Item = (&String, &String, &Value)> {
        self.0.iter().flat_map(|(collection, docs)| {
            docs.iter().map(move |(id, doc)| (collection, id, &**doc))
        })
    }

    /// Names of the collections that hold at least one document.
    pub(super) fn collections(&self) -> impl Iterator<Item = &String> {
        self.0.keys()
    }

    pub(super) fn len(&self) -> usize {
        self.0.values().map(|docs| docs.len()).sum()
    }

    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Insert or replace one document. Copies the collection only when it is shared.
    pub(super) fn insert(&mut self, collection: String, id: String, doc: Arc<Value>) {
        Arc::make_mut(self.0.entry(collection).or_default()).insert(id, doc);
    }

    /// Remove one document. A missing one copies nothing.
    pub(super) fn remove(&mut self, collection: &str, id: &str) {
        let Some(docs) = self.0.get_mut(collection) else {
            return;
        };
        if !docs.contains_key(id) {
            return;
        }
        let docs = Arc::make_mut(docs);
        docs.remove(id);
        if docs.is_empty() {
            self.0.remove(collection);
        }
    }

    /// Apply one collection's puts (`Some`) and deletes (`None`), in id order.
    /// The collection is copied once, and only when it is shared and a change
    /// lands; a delete of a missing id is no change.
    fn apply_collection(&mut self, collection: String, changes: DocChanges) {
        let held = self.0.get(&collection);
        let changes_something = changes
            .iter()
            .any(|(id, doc)| doc.is_some() || held.is_some_and(|docs| docs.contains_key(id)));
        if !changes_something {
            return;
        }
        let docs = Arc::make_mut(self.0.entry(collection.clone()).or_default());
        for (id, doc) in changes {
            match doc {
                Some(doc) => {
                    docs.insert(id, doc);
                }
                None => {
                    docs.remove(&id);
                }
            }
        }
        if docs.is_empty() {
            self.0.remove(&collection);
        }
    }

    /// The `Arc` that holds one collection, for tests that check what a commit copied.
    #[cfg(test)]
    pub(super) fn collection_arc(&self, collection: &str) -> Option<&Arc<Collection>> {
        self.0.get(collection)
    }
}

impl Docs {
    /// From collections built in place. An empty collection is dropped.
    pub(super) fn from_nested(nested: BTreeMap<String, Collection>) -> Self {
        Self(
            nested
                .into_iter()
                .filter(|(_, docs)| !docs.is_empty())
                .map(|(collection, docs)| (collection, Arc::new(docs)))
                .collect(),
        )
    }
}

impl FromIterator<((String, String), Value)> for Docs {
    fn from_iter<I: IntoIterator<Item = ((String, String), Value)>>(iter: I) -> Self {
        let mut nested: BTreeMap<String, Collection> = BTreeMap::new();
        for ((collection, id), doc) in iter {
            nested
                .entry(collection)
                .or_default()
                .insert(id, Arc::new(doc));
        }
        Self::from_nested(nested)
    }
}

impl From<BTreeMap<(String, String), Value>> for Docs {
    fn from(flat: BTreeMap<(String, String), Value>) -> Self {
        flat.into_iter().collect()
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct DocState {
    pub(super) docs: Docs,
    pub(super) tables: Tables,
    pub(super) vectors: Vectors,
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

/// Changes staged over a base [`DocState`] that the base does not see.
///
/// Reads through an overlay see its changes merged over the base. Staging an op
/// checks it against that merged view, exactly as applying it to the base would,
/// and never changes the base. [`Overlay::apply`] then cannot fail.
#[derive(Default)]
pub(super) struct Overlay {
    /// Pending puts (`Some`) and deletes (`None`), last write wins.
    docs: BTreeMap<(String, String), Option<Arc<Value>>>,
    tables: BTreeMap<String, TableStage>,
    /// The whole collection after this overlay's ops: a vector op re-links the
    /// graph, so a touched collection is copied once and changed here.
    vectors: Vectors,
}

/// One column table's staged change.
struct TableStage {
    /// `Some` when this overlay creates the table.
    created: Option<Vec<ColumnDef>>,
    /// Row groups appended after the base's, in order.
    groups: Vec<Vec<u8>>,
}

impl Overlay {
    /// The document at `(collection, id)` seen through this overlay.
    pub(super) fn doc<'a>(
        &'a self,
        base: &'a DocState,
        collection: &str,
        id: &str,
    ) -> Option<&'a Value> {
        if self.docs.is_empty() {
            return base.docs.get(collection, id);
        }
        match self.docs.get(&(collection.to_string(), id.to_string())) {
            Some(staged) => staged.as_deref(),
            None => base.docs.get(collection, id),
        }
    }

    /// The staged put (`Some(Some)`) or delete (`Some(None)`) at
    /// `(collection, id)`; `None` when this overlay has not touched it.
    pub(super) fn staged_doc(&self, collection: &str, id: &str) -> Option<Option<&Value>> {
        if self.docs.is_empty() {
            return None;
        }
        self.docs
            .get(&(collection.to_string(), id.to_string()))
            .map(|staged| staged.as_deref())
    }

    /// Every staged put (`Some`) and delete (`None`) in `collection`, in id order.
    pub(super) fn staged_in<'a>(
        &'a self,
        collection: &'a str,
    ) -> impl Iterator<Item = (&'a str, Option<&'a Value>)> + 'a {
        self.docs
            .range((collection.to_string(), String::new())..)
            .take_while(move |((held, _), _)| held == collection)
            .map(|((_, id), doc)| (id.as_str(), doc.as_deref()))
    }

    /// Every document in `collection` seen through this overlay, in id order:
    /// a staged put replaces or adds, a staged delete hides.
    pub(super) fn visit(&self, base: &DocState, collection: &str, f: impl FnMut(&str, &Value)) {
        self.visit_over(
            collection,
            base.docs
                .in_collection(collection)
                .map(|(id, doc)| (id.as_str(), doc)),
            f,
        );
    }

    /// [`Self::visit`] over `held`, the base's documents of `collection` in
    /// id order, wherever they came from.
    pub(super) fn visit_over<'b>(
        &self,
        collection: &str,
        held: impl Iterator<Item = (&'b str, &'b Value)>,
        mut f: impl FnMut(&str, &Value),
    ) {
        let mut staged = self.staged_in(collection).peekable();
        let mut held = held.peekable();
        loop {
            let take_staged = match (held.peek(), staged.peek()) {
                (None, None) => break,
                (Some(_), None) => false,
                (None, Some(_)) => true,
                (Some((base_id, _)), Some((staged_id, _))) => staged_id <= base_id,
            };
            if take_staged {
                let Some((id, doc)) = staged.next() else {
                    break;
                };
                if held.peek().is_some_and(|(base_id, _)| *base_id == id) {
                    held.next();
                }
                if let Some(doc) = doc {
                    f(id, doc);
                }
            } else if let Some((id, doc)) = held.next() {
                f(id, doc);
            }
        }
    }

    pub(super) fn put(&mut self, collection: String, id: String, doc: Arc<Value>) {
        self.docs.insert((collection, id), Some(doc));
    }

    pub(super) fn delete(&mut self, collection: String, id: String) {
        self.docs.insert((collection, id), None);
    }

    /// The columns of `name` seen through this overlay.
    pub(super) fn table_columns<'a>(
        &'a self,
        base: &'a DocState,
        name: &str,
    ) -> Option<&'a [ColumnDef]> {
        if let Some(created) = self.tables.get(name).and_then(|t| t.created.as_ref()) {
            return Some(created);
        }
        base.tables.get(name).map(|table| table.columns.as_slice())
    }

    /// The columns of `name` when this overlay creates it.
    pub(super) fn created_columns(&self, name: &str) -> Option<&[ColumnDef]> {
        self.tables.get(name).and_then(|t| t.created.as_deref())
    }

    /// `true` when this overlay creates a column table.
    pub(super) fn creates_table(&self) -> bool {
        self.tables.values().any(|t| t.created.is_some())
    }

    /// Record a create the caller has checked, for a remote transaction,
    /// whose overlay holds no base to check it against.
    pub(super) fn record_table_create(&mut self, name: &str, columns: &[ColumnDef]) {
        self.tables.insert(
            name.to_string(),
            TableStage {
                created: Some(columns.to_vec()),
                groups: Vec::new(),
            },
        );
    }

    /// Record an append the caller has checked (see [`Self::record_table_create`]).
    pub(super) fn record_table_append(&mut self, name: &str, bytes: &[u8]) {
        self.tables
            .entry(name.to_string())
            .or_insert_with(|| TableStage {
                created: None,
                groups: Vec::new(),
            })
            .groups
            .push(bytes.to_vec());
    }

    /// The vector collection `name` seen through this overlay.
    pub(super) fn vector<'a>(
        &'a self,
        base: &'a DocState,
        name: &str,
    ) -> Option<&'a VectorCollection> {
        self.vectors
            .get(name)
            .or_else(|| base.vectors.get(name))
            .map(|col| &**col)
    }

    pub(super) fn has_tables(&self, base: &DocState) -> bool {
        !base.tables.is_empty() || self.tables.values().any(|t| t.created.is_some())
    }

    pub(super) fn has_vectors(&self, base: &DocState) -> bool {
        !base.vectors.is_empty() || !self.vectors.is_empty()
    }

    /// Stage one op over `base` and this overlay. `Err` is the error applying
    /// `op` would give. A failed vector op may leave its staged collection
    /// partly changed, so the caller drops the whole overlay on `Err`.
    pub(super) fn stage(&mut self, base: &DocState, op: &Op) -> Result<(), StoreError> {
        match op {
            Op::Put {
                collection,
                id,
                doc,
            } => self.put(collection.clone(), id.clone(), Arc::new(doc.clone())),
            Op::Delete { collection, id } => self.delete(collection.clone(), id.clone()),
            Op::CreateColumnTable { name, columns } => {
                let taken = self.table_columns(base, name).is_some();
                check_new_table(taken, name, columns)?;
                self.tables.insert(
                    name.clone(),
                    TableStage {
                        created: Some(columns.clone()),
                        groups: Vec::new(),
                    },
                );
            }
            Op::AppendColumnGroup { name, bytes } => {
                let Some(columns) = self.table_columns(base, name) else {
                    return Err(corrupt(format!("append to missing column table `{name}`")));
                };
                validate_row_group(columns, bytes)?;
                self.tables
                    .entry(name.clone())
                    .or_insert_with(|| TableStage {
                        created: None,
                        groups: Vec::new(),
                    })
                    .groups
                    .push(bytes.clone());
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
                if self.vector(base, name).is_some() {
                    return Err(backend(format!(
                        "vector collection `{name}` already exists"
                    )));
                }
                self.vectors.insert(
                    name.clone(),
                    Arc::new(VectorCollection::empty(
                        *dim,
                        *storage,
                        *metric,
                        *m,
                        *ef_construction,
                        *ef_search,
                    )),
                );
            }
            Op::UpsertVector {
                collection,
                id,
                bytes,
                metadata,
            } => {
                self.staged_vectors(base, collection)?.upsert(
                    id.clone(),
                    bytes.clone(),
                    metadata.clone(),
                )?;
            }
            Op::DeleteVector { collection, id } => {
                self.staged_vectors(base, collection)?.delete_id(id)?;
            }
        }
        Ok(())
    }

    /// This overlay's own copy of the vector collection, made on first touch.
    fn staged_vectors(
        &mut self,
        base: &DocState,
        collection: &str,
    ) -> Result<&mut VectorCollection, StoreError> {
        if !self.vectors.contains_key(collection) {
            let Some(held) = base.vectors.get(collection) else {
                return Err(backend(format!("no vector collection `{collection}`")));
            };
            self.vectors
                .insert(collection.to_string(), Arc::clone(held));
        }
        let Some(staged) = self.vectors.get_mut(collection) else {
            return Err(backend(format!("no vector collection `{collection}`")));
        };
        Ok(Arc::make_mut(staged))
    }

    /// Write every staged change into `state`, which must be the base this
    /// overlay was staged over. Each touched collection, table or vector
    /// collection is copied only when it is shared; the rest are not touched.
    pub(super) fn apply(self, state: &mut DocState) {
        let mut pending: Option<(String, DocChanges)> = None;
        for ((collection, id), doc) in self.docs {
            match &mut pending {
                Some((current, changes)) if *current == collection => changes.push((id, doc)),
                _ => {
                    if let Some((current, changes)) = pending.take() {
                        state.docs.apply_collection(current, changes);
                    }
                    pending = Some((collection, vec![(id, doc)]));
                }
            }
        }
        if let Some((current, changes)) = pending {
            state.docs.apply_collection(current, changes);
        }
        for (name, stage) in self.tables {
            match stage.created {
                Some(columns) => {
                    state.tables.insert(
                        name,
                        Arc::new(ColumnTable {
                            columns,
                            groups: stage.groups,
                        }),
                    );
                }
                None => {
                    if let Some(table) = state.tables.get_mut(&name) {
                        Arc::make_mut(table).groups.extend(stage.groups);
                    }
                }
            }
        }
        for (name, col) in self.vectors {
            state.vectors.insert(name, col);
        }
    }
}

pub(super) fn apply_commit(state: &mut Arc<DocState>, op: Op) -> Result<(), StoreError> {
    commit_ops(state, &[op])
}

/// Apply every op, then advance `last_seq` by one. An empty list writes nothing.
/// `state` is changed only when every op applies.
pub(super) fn commit_ops(state: &mut Arc<DocState>, ops: &[Op]) -> Result<(), StoreError> {
    if ops.is_empty() {
        return Ok(());
    }
    let (overlay, seq) = stage_ops(state, ops)?;
    publish_overlay(state, overlay, seq);
    Ok(())
}

/// Every op staged over `state`, and the sequence one past `state`'s.
/// `state` is not changed.
pub(super) fn stage_ops(state: &DocState, ops: &[Op]) -> Result<(Overlay, u64), StoreError> {
    let seq = state
        .last_seq
        .checked_add(1)
        .ok_or_else(|| backend("transaction sequence overflow"))?;
    let mut overlay = Overlay::default();
    for op in ops {
        overlay.stage(state, op)?;
    }
    Ok((overlay, seq))
}

/// Apply `overlay` to `state` and set `last_seq` to `seq`. With no other holder
/// of `state` this changes it in place; with one, the outer maps are copied and
/// then only the touched collections.
pub(super) fn publish_overlay(state: &mut Arc<DocState>, overlay: Overlay, seq: u64) {
    let next = Arc::make_mut(state);
    overlay.apply(next);
    next.last_seq = seq;
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

/// Apply one replayed op in place. Replay owns the state it builds, so nothing
/// is copied; an error fails the whole load. The checks are those of
/// [`Overlay::stage`].
fn apply_op(state: &mut DocState, op: &Op) -> Result<(), StoreError> {
    match op {
        Op::Put {
            collection,
            id,
            doc,
        } => {
            state
                .docs
                .insert(collection.clone(), id.clone(), Arc::new(doc.clone()));
        }
        Op::Delete { collection, id } => {
            state.docs.remove(collection, id);
        }
        Op::CreateColumnTable { name, columns } => {
            check_new_table(state.tables.contains_key(name), name, columns)?;
            state.tables.insert(
                name.clone(),
                Arc::new(ColumnTable {
                    columns: columns.clone(),
                    groups: Vec::new(),
                }),
            );
        }
        Op::AppendColumnGroup { name, bytes } => {
            let Some(table) = state.tables.get_mut(name) else {
                return Err(corrupt(format!("append to missing column table `{name}`")));
            };
            validate_row_group(&table.columns, bytes)?;
            Arc::make_mut(table).groups.push(bytes.clone());
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
                Arc::new(VectorCollection::empty(
                    *dim,
                    *storage,
                    *metric,
                    *m,
                    *ef_construction,
                    *ef_search,
                )),
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
            Arc::make_mut(col).upsert(id.clone(), bytes.clone(), metadata.clone())?;
        }
        Op::DeleteVector { collection, id } => {
            let Some(col) = state.vectors.get_mut(collection) else {
                return Err(backend(format!("no vector collection `{collection}`")));
            };
            Arc::make_mut(col).delete_id(id)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn search_hybrid(
    docs: &Docs,
    vectors: &Vectors,
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
    docs: &Docs,
    vectors: &Vectors,
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
        let doc = docs.get(doc_collection, &id).cloned();
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
    for (id, doc) in state.docs.in_collection(collection) {
        if field_matches(doc, field, value) {
            ids.push(id.clone());
        }
    }
    ids.sort();
    ids
}

/// `doc` is an object whose top-level `field` is a scalar whose key is `value`.
pub(super) fn field_matches(doc: &Value, field: &str, value: &str) -> bool {
    doc.as_object()
        .and_then(|object| object.get(field))
        .is_some_and(|field_value| scalar_key(field_value).as_deref() == Some(value))
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
    stored: &Docs,
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
    for (id, doc) in stored.in_collection(collection) {
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

/// The next message sequence for `topic`, read through `txn`.
pub(super) fn next_mq_seq(txn: &Txn, topic: &str) -> Result<u64, StoreError> {
    // Absent on files written before the high-water existed: one past the
    // highest message still stored. A present high-water is not lowered by
    // deletes, so it wins over the remaining messages.
    let floor = match mq_high_water(txn, topic)? {
        Some(seq) => seq,
        None => max_stored_mq_seq(txn, topic),
    };
    floor
        .checked_add(1)
        .ok_or_else(|| backend("mq sequence overflow"))
}

fn mq_high_water(txn: &Txn, topic: &str) -> Result<Option<u64>, StoreError> {
    let Some(doc) = txn.get(&mq_hw_collection(topic), MQ_HIGH_WATER_ID) else {
        return Ok(None);
    };
    let Some(seq) = doc.get("seq").and_then(Value::as_u64) else {
        return Err(corrupt(format!("mq high-water for {topic} has no seq")));
    };
    Ok(Some(seq))
}

fn max_stored_mq_seq(txn: &Txn, topic: &str) -> u64 {
    let mut max = 0u64;
    txn.visit(&mq_message_collection(topic), |id, doc| {
        let seq = doc
            .get("seq")
            .and_then(Value::as_u64)
            .or_else(|| id.parse().ok());
        if let Some(seq) = seq {
            max = max.max(seq);
        }
    });
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
    for (id, doc) in state.docs.in_collection(&collection) {
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
    for (id, doc) in state.docs.in_collection(&collection) {
        if let Some(seq) = doc.get("seq").and_then(Value::as_u64) {
            seqs.insert(seq);
        } else if let Ok(seq) = id.parse::<u64>() {
            seqs.insert(seq);
        }
    }
    seqs
}
