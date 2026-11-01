//! The client side of a remote transaction (spec §6.1): a `Txn` whose base
//! is a read session on the server, not a state in this process.
//!
//! Each read primitive is one `txn_read` against the session's pinned
//! snapshot; the first read opens the session. An answer is cached for the
//! rest of the body, and recorded once in the read set with the hash the
//! server computed. A read the overlay answers alone (the body's own put or
//! delete of that document, its own create of a column table or a vector
//! collection) asks nothing and is not recorded; a scan or a query merges the
//! overlay over the base, so the base's answer is fetched and recorded whole.
//!
//! A read whose RPC fails records its error, answers as if empty, and every
//! later read answers the same without asking: the transaction then returns
//! that error, whatever the body returned.

use super::layout::backend;
use super::server::RemoteShared;
use super::session::{Pinned, ReadKey, ReadSet};
use super::vector::VectorParams;
use fl_core::StoreError;
use serde_json::{Value, json};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

/// A remote transaction's base: its session, its cache and its read set.
pub(super) struct RemoteBase {
    shared: Arc<RemoteShared>,
    /// `(session, base_seq)` once the first read opened it.
    session: Cell<Option<Pinned>>,
    cache: RefCell<HashMap<ReadKey, Arc<Value>>>,
    /// Each distinct read and its answer's hash, in the order first read.
    reads: RefCell<ReadSet>,
    error: RefCell<Option<StoreError>>,
    /// Vector collections this transaction creates: their parameters.
    pub(super) created_vectors: BTreeMap<String, VectorParams>,
    /// Vector collections this transaction creates or changes. A read of
    /// one of their vectors is refused: the server applies the change.
    pub(super) written_vectors: BTreeSet<String>,
}

impl RemoteBase {
    pub(super) fn new(shared: Arc<RemoteShared>) -> Self {
        Self {
            shared,
            session: Cell::new(None),
            cache: RefCell::new(HashMap::new()),
            reads: RefCell::new(Vec::new()),
            error: RefCell::new(None),
            created_vectors: BTreeMap::new(),
            written_vectors: BTreeSet::new(),
        }
    }

    /// `key`'s answer at the session's snapshot. `None` once a read failed.
    pub(super) fn read(&self, key: ReadKey) -> Option<Arc<Value>> {
        if self.error.borrow().is_some() {
            return None;
        }
        if let Some(hit) = self.cache.borrow().get(&key) {
            return Some(Arc::clone(hit));
        }
        match self.fetch(&key) {
            Ok((result, hash)) => {
                let result = Arc::new(result);
                self.cache
                    .borrow_mut()
                    .insert(key.clone(), Arc::clone(&result));
                self.reads.borrow_mut().push((key, hash));
                Some(result)
            }
            Err(err) => {
                self.fail(err);
                None
            }
        }
    }

    fn fetch(&self, key: &ReadKey) -> Result<(Value, String), StoreError> {
        let mut params = serde_json::Map::new();
        params.insert(
            "read".into(),
            serde_json::to_value(key).map_err(|err| backend(format!("encode read: {err}")))?,
        );
        let opened = self.session.get();
        if let Some((session, _)) = opened {
            params.insert("session".into(), json!(session));
        }
        let mut reply = self.shared.call("txn_read", Value::Object(params))?;
        if opened.is_none() {
            let session = reply.get("session").and_then(Value::as_u64);
            let base_seq = reply.get("base_seq").and_then(Value::as_u64);
            let (Some(session), Some(base_seq)) = (session, base_seq) else {
                return Err(backend("txn_read opened no session"));
            };
            self.session.set(Some((session, base_seq)));
        }
        let hash = match reply.get("hash").and_then(Value::as_str) {
            Some(hash) if hash.len() == 64 => hash.to_string(),
            _ => return Err(backend("txn_read reply has no 32-byte hash")),
        };
        let result = reply
            .get_mut("result")
            .map(Value::take)
            .ok_or_else(|| backend("txn_read reply has no result"))?;
        check_shape(key, &result)?;
        Ok((result, hash))
    }

    /// Record `err` as the transaction's, unless one is already recorded.
    pub(super) fn fail(&self, err: StoreError) {
        let mut slot = self.error.borrow_mut();
        if slot.is_none() {
            *slot = Some(err);
        }
    }

    /// The error a read recorded, which the transaction returns.
    pub(super) fn take_error(&self) -> Option<StoreError> {
        self.error.borrow_mut().take()
    }

    /// The session and the read set, for `commit_txn`, which ends the
    /// session on the server: it is not closed again here.
    pub(super) fn take_for_commit(&self) -> (Option<Pinned>, ReadSet) {
        (
            self.session.take(),
            std::mem::take(&mut *self.reads.borrow_mut()),
        )
    }

    /// The answers read so far (not the keys asked for), formatted only when
    /// the retry test hook formats them: what it knows of the state, so an id
    /// in it is not fresh. A build without tests never formats them.
    pub(super) fn known(&self) -> Known<'_> {
        Known(self)
    }
}

/// [`RemoteBase::known`]: its `Debug` lists the cached answers.
pub(super) struct Known<'a>(&'a RemoteBase);

impl std::fmt::Debug for Known<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(self.0.cache.borrow().values())
            .finish()
    }
}

impl Drop for RemoteBase {
    /// A session not handed to `commit_txn` is closed: a read-only body, an
    /// error, a lost attempt the client gave up, or an unwind.
    fn drop(&mut self) {
        if let Some((session, _)) = self.session.take() {
            self.shared
                .send_only("close_session", json!({ "session": session }));
        }
    }
}

/// `Err` unless `result` has the form `key`'s primitive answers with.
fn check_shape(key: &ReadKey, result: &Value) -> Result<(), StoreError> {
    let ok = match key {
        ReadKey::Get { .. } => result.is_null() || result.get("doc").is_some(),
        ReadKey::Scan { .. } => result.as_array().is_some_and(|items| {
            items.iter().all(|pair| {
                pair.as_array()
                    .is_some_and(|pair| pair.len() == 2 && pair[0].is_string())
            })
        }),
        ReadKey::Query { .. } => result
            .as_array()
            .is_some_and(|ids| ids.iter().all(Value::is_string)),
        ReadKey::Columns { .. } => result.is_null() || result.is_array(),
        ReadKey::VectorParams { .. } => result.is_null() || result.is_object(),
        ReadKey::GetVector { .. } => result.is_null() || result.is_object(),
        ReadKey::Inventory => ["collections", "tables", "vectors"]
            .iter()
            .all(|field| result.get(field).is_some_and(Value::is_array)),
    };
    if ok {
        Ok(())
    } else {
        Err(backend(format!(
            "txn_read {} of `{}` answered in the wrong form",
            key.primitive(),
            key.subject()
        )))
    }
}

/// The names under `field` of an inventory answer.
pub(super) fn inventory_names(result: &Value, field: &str) -> Vec<String> {
    result
        .get(field)
        .and_then(Value::as_array)
        .map(|names| {
            names
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}
