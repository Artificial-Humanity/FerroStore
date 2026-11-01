//! Document keys and the JSON rows the trait impls share.
//! Collection names are the redb table names. A string key is the document
//! id, a `u64` key is its decimal, and a pair joins with U+001F.

use super::Txn;
use crate::{backend, decode, mint};
use fl_core::escalation::{Mark, Tombstone, escalate_command};
use fl_core::ids::Kind;
use fl_core::iri::Iri;
use fl_core::store::StoreError;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::collections::BTreeSet;

pub(super) const PROJECTS: &str = "projects";
pub(super) const GATES: &str = "gates";
pub(super) const TRANSITIONS: &str = "transitions";
pub(super) const RECORDS: &str = "records";
pub(super) const FINDINGS: &str = "findings";
pub(super) const GATE_RUNS: &str = "gate_runs";
pub(super) const ATTEMPTS: &str = "attempts";
pub(super) const IMPORTS: &str = "imports";
pub(super) const GITHUB_BINDINGS: &str = "github_bindings";
pub(super) const LEDGER_PUBLISHED: &str = "ledger_published";
pub(super) const LEDGER_CUTOVERS: &str = "ledger_cutovers";
pub(super) const CANDIDATE_RUNS: &str = "ledger_candidate_runs";
pub(super) const CANDIDATE_ATTEMPTS: &str = "ledger_candidate_attempts";
pub(super) const LEDGER_ROOTS: &str = "ledger_roots";
pub(super) const LEDGER_HEADS: &str = "ledger_heads";
pub(super) const LEDGER_SEGMENTS: &str = "ledger_segments";
pub(super) const ROUTING: &str = "routing";
pub(super) const ESCALATING: &str = "escalating";
pub(super) const TOMBSTONES: &str = "tombstones";
pub(super) const META: &str = "meta";
pub(super) const IDS: &str = "ids";
pub(super) const ALIASES: &str = "aliases";
pub(super) const HANDLES: &str = "handles";
pub(super) const HANDLE_OF: &str = "handle_of";
pub(super) const STATE_ENTRIES: &str = "state_entries";
pub(super) const STATE_KEYS: &str = "state_keys";
pub(super) const STATE_REVISIONS: &str = "state_revisions";
pub(super) const STATE_REVISION_IDS: &str = "state_revision_ids";
pub(super) const STATE_CONFLICTS: &str = "state_conflicts";
pub(super) const STATE_SYNC_CURSORS: &str = "state_sync_cursors";
pub(super) const AGENTS: &str = "agents";
pub(super) const AGENT_BY_NAME: &str = "agent_by_name";
pub(super) const AREAS: &str = "areas";
pub(super) const AGENT_EVENTS: &str = "agent_events";
pub(super) const AGENT_EVENT_IDS: &str = "agent_event_ids";
pub(super) const WIRE_MESSAGES: &str = "wire_messages";
pub(super) const WIRE_MSG_LOG: &str = "wire_msg_log";
pub(super) const WIRE_MSG_SEQ: &str = "wire_msg_seq";
pub(super) const WIRE_INBOX: &str = "wire_inbox";
pub(super) const WIRE_RCPT_LOG: &str = "wire_rcpt_log";
pub(super) const WIRE_THREADS: &str = "wire_threads";
pub(super) const WIRE_REPLIES: &str = "wire_replies";
pub(super) const WORKFLOW_DEFS: &str = "workflow_defs";
pub(super) const WORKFLOW_SLUGS: &str = "workflow_slugs";
pub(super) const WORKFLOW_RUNS: &str = "workflow_runs";

/// The store's own bookkeeping. Not a redb table: no trait lists it, the
/// migration writes it from redb's `meta.format_version`, and the verifier
/// checks it as its own entry.
pub(super) const SCHEMA_COLLECTION: &str = "__meta";
/// A collection whose name starts with this belongs to the store: only its own
/// transactions write it. Public `put_doc` / `delete_doc` refuse it. Outbound
/// sync batches carry its ops like any other; an inbound sync batch applies
/// one only as a schema raise (`admit_inbound` in `mod.rs`) and drops the rest.
pub(super) const RESERVED_PREFIX: &str = "__";

/// `true` for a collection only the store itself writes ([`RESERVED_PREFIX`]).
pub(super) fn is_reserved(collection: &str) -> bool {
    collection.starts_with(RESERVED_PREFIX)
}

/// `Err` naming `collection` when it is reserved; `what` is the refused call.
pub(super) fn refuse_reserved(collection: &str, what: &str) -> Result<(), StoreError> {
    if is_reserved(collection) {
        return Err(backend(format!(
            "{what} refused: collection `{collection}` is reserved \
             (a `{RESERVED_PREFIX}` collection is written only by the store itself)"
        )));
    }
    Ok(())
}
/// `(SCHEMA_COLLECTION, SCHEMA_KEY)` holds the schema version: the same ladder
/// as `RedbStore` (`FORMAT_VERSION` … `FORMAT_WITH_WORKFLOW`).
pub(super) const SCHEMA_KEY: &str = "schema_version";
/// redb's key for the schema version in `meta`. A FerroStore written before
/// `__meta` existed may hold it in the `meta` collection.
pub(super) const FORMAT_KEY: &str = "format_version";
pub(super) const NEXT_RUN: &str = "next_run";
pub(super) const NEXT_ATTEMPT: &str = "next_attempt";
pub(super) const NEXT_STATE_REV: &str = "next_state_rev";
pub(super) const NEXT_AGENT_EVENT: &str = "next_agent_event";
pub(super) const NEXT_WIRE_MSG_SEQ: &str = "next_wire_msg_seq";
pub(super) const NEXT_WIRE_RCPT_SEQ: &str = "next_wire_rcpt_seq";

const SEP: char = '\u{1f}';

pub(super) fn pair(a: &str, b: &str) -> String {
    format!("{a}{SEP}{b}")
}

pub(super) fn pair_u64(a: &str, n: u64) -> String {
    format!("{a}{SEP}{n}")
}

pub(super) fn triple(a: &str, b: &str, c: &str) -> String {
    format!("{a}{SEP}{b}{SEP}{c}")
}

pub(super) fn key_u64(n: u64) -> String {
    n.to_string()
}

pub(super) fn split_pair(key: &str) -> Option<(&str, &str)> {
    key.split_once(SEP)
}

pub(super) fn put_json<T: Serialize>(
    txn: &mut Txn,
    collection: &str,
    id: &str,
    value: &T,
) -> Result<(), StoreError> {
    let doc = serde_json::to_value(value).map_err(backend)?;
    txn.put(collection, id, doc);
    Ok(())
}

pub(super) fn get_json<T: DeserializeOwned>(
    txn: &Txn,
    collection: &str,
    id: &str,
) -> Result<Option<T>, StoreError> {
    match txn.get(collection, id) {
        Some(doc) => Ok(Some(serde_json::from_value(doc).map_err(decode)?)),
        None => Ok(None),
    }
}

pub(super) fn scan_json<T: DeserializeOwned>(
    txn: &Txn,
    collection: &str,
) -> Result<Vec<T>, StoreError> {
    let mut out = Vec::new();
    for (_id, doc) in txn.scan(collection) {
        out.push(serde_json::from_value(doc).map_err(decode)?);
    }
    Ok(out)
}

/// `(seq, value)` in numeric sequence order. Decimal keys are not numeric order.
pub(super) fn scan_u64<T: DeserializeOwned>(
    txn: &Txn,
    collection: &str,
) -> Result<Vec<(u64, T)>, StoreError> {
    let mut out = Vec::new();
    for (id, doc) in txn.scan(collection) {
        let seq = id
            .parse::<u64>()
            .map_err(|_| decode(format!("sequence key `{id}` is not a u64")))?;
        out.push((seq, serde_json::from_value(doc).map_err(decode)?));
    }
    out.sort_by_key(|(seq, _)| *seq);
    Ok(out)
}

pub(super) fn bump_meta(txn: &mut Txn, key: &str) -> Result<u64, StoreError> {
    let next = get_json::<u64>(txn, META, key)?
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| backend(format!("counter `{key}` overflow")))?;
    put_json(txn, META, key, &next)?;
    Ok(next)
}

pub(super) fn not_owned(label: &str, id: &Iri) -> StoreError {
    StoreError::NotOwned {
        id: id.clone(),
        searched: vec![label.to_string()],
    }
}

fn alias_primary(txn: &Txn, id: &Iri) -> Result<(Iri, String), StoreError> {
    let Some(primary) = get_json::<String>(txn, ALIASES, id.as_str())? else {
        return Err(decode(format!("alias `{id}` has no primary recorded")));
    };
    let primary = Iri::parse(&primary).map_err(decode)?;
    let Some(wire) = get_json::<String>(txn, IDS, primary.as_str())? else {
        return Err(decode(format!(
            "alias `{id}` names {primary}, which this store does not hold"
        )));
    };
    Ok((primary, wire))
}

pub(super) fn locate(txn: &Txn, label: &str, id: &Iri) -> Result<(Iri, Kind), StoreError> {
    let Some(wire) = get_json::<String>(txn, IDS, id.as_str())? else {
        return Err(not_owned(label, id));
    };
    let (primary, wire) = if wire == "alias" {
        alias_primary(txn, id)?
    } else {
        (id.clone(), wire)
    };
    let kind = Kind::from_wire(&wire)
        .ok_or_else(|| decode(format!("unknown kind `{wire}` for {primary}")))?;
    Ok((primary, kind))
}

pub(super) fn check_kind(
    txn: &Txn,
    label: &str,
    id: &Iri,
    expected: Kind,
) -> Result<(), StoreError> {
    let (_primary, found) = locate(txn, label, id)?;
    if found != expected {
        return Err(StoreError::WrongKind {
            id: id.clone(),
            expected,
            found,
        });
    }
    Ok(())
}

pub(super) fn owns(txn: &Txn, label: &str, id: &Iri) -> Result<bool, StoreError> {
    match locate(txn, label, id) {
        Ok(_) => Ok(true),
        Err(StoreError::NotOwned { .. }) => Ok(false),
        Err(err) => Err(err),
    }
}

/// Primary, or `id` itself when this store does not hold it.
pub(super) fn resolve_iri(txn: &Txn, label: &str, id: &Iri) -> Result<Iri, StoreError> {
    match locate(txn, label, id) {
        Ok((primary, _)) => Ok(primary),
        Err(StoreError::NotOwned { .. }) => Ok(id.clone()),
        Err(err) => Err(err),
    }
}

pub(super) fn resolve_if_kind(
    txn: &Txn,
    label: &str,
    id: &Iri,
    expected: Kind,
) -> Result<Iri, StoreError> {
    match locate(txn, label, id) {
        Ok((_, found)) if found != expected => Err(StoreError::WrongKind {
            id: id.clone(),
            expected,
            found,
        }),
        Ok((primary, _)) => Ok(primary),
        Err(StoreError::NotOwned { .. }) => Ok(id.clone()),
        Err(err) => Err(err),
    }
}

pub(super) fn index_new(txn: &mut Txn, id: &Iri, kind: Kind) -> Result<(), StoreError> {
    if get_json::<String>(txn, IDS, id.as_str())?.is_some() {
        return Err(StoreError::AlreadyExists(id.clone()));
    }
    put_json(txn, IDS, id.as_str(), &kind.as_wire())?;
    let key = format!("next_handle:{}", kind.as_wire());
    let n = bump_meta(txn, &key)?;
    put_json(txn, HANDLES, &pair_u64(kind.as_wire(), n), &id.as_str())?;
    put_json(txn, HANDLE_OF, id.as_str(), &n)?;
    Ok(())
}

pub(super) fn index_imported(txn: &mut Txn, kind: Kind, id: &Iri) -> Result<(), StoreError> {
    if let Some(wire) = get_json::<String>(txn, IDS, id.as_str())? {
        if wire == "alias" {
            return Ok(());
        }
        let found = Kind::from_wire(&wire)
            .ok_or_else(|| decode(format!("unknown kind `{wire}` for {id}")))?;
        if found != kind {
            return Err(StoreError::WrongKind {
                id: id.clone(),
                expected: kind,
                found,
            });
        }
        return Ok(());
    }
    index_new(txn, id, kind)
}

/// Point `old_id` at `new_id` and retarget every alias that named `old_id`.
/// The handle moves with the id; a new id with none is given the next one.
pub(super) fn alias_unpublished(
    txn: &mut Txn,
    kind: Kind,
    old_id: &Iri,
    new_id: &Iri,
) -> Result<(), StoreError> {
    if old_id == new_id {
        return Ok(());
    }
    put_json(txn, IDS, old_id.as_str(), &"alias")?;
    put_json(txn, ALIASES, old_id.as_str(), &new_id.as_str())?;
    let retarget: Vec<String> = txn
        .scan(ALIASES)
        .into_iter()
        .filter_map(|(key, doc)| {
            let target = doc.as_str()?;
            if target == old_id.as_str() && key != old_id.as_str() {
                Some(key)
            } else {
                None
            }
        })
        .collect();
    for alias_key in retarget {
        put_json(txn, ALIASES, &alias_key, &new_id.as_str())?;
    }
    put_json(txn, IDS, new_id.as_str(), &kind.as_wire())?;

    let old_handle = get_json::<u64>(txn, HANDLE_OF, old_id.as_str())?;
    if let Some(handle) = old_handle {
        txn.delete(HANDLE_OF, old_id.as_str());
        put_json(
            txn,
            HANDLES,
            &pair_u64(kind.as_wire(), handle),
            &new_id.as_str(),
        )?;
        put_json(txn, HANDLE_OF, new_id.as_str(), &handle)?;
    } else if get_json::<u64>(txn, HANDLE_OF, new_id.as_str())?.is_none() {
        let key = format!("next_handle:{}", kind.as_wire());
        let n = bump_meta(txn, &key)?;
        put_json(txn, HANDLES, &pair_u64(kind.as_wire(), n), &new_id.as_str())?;
        put_json(txn, HANDLE_OF, new_id.as_str(), &n)?;
    }
    Ok(())
}

/// The schema version `txn` records: `("__meta", "schema_version")`, else the
/// `("meta", "format_version")` a store from before `__meta` raised, else `None`.
pub(super) fn schema_version(txn: &Txn) -> Result<Option<u64>, StoreError> {
    schema_version_in(|collection, id| txn.get(collection, id))
}

/// [`schema_version`] over any document lookup.
pub(super) fn schema_version_in(
    get: impl Fn(&str, &str) -> Option<Value>,
) -> Result<Option<u64>, StoreError> {
    for (collection, id) in [(SCHEMA_COLLECTION, SCHEMA_KEY), (META, FORMAT_KEY)] {
        if let Some(doc) = get(collection, id) {
            return doc.as_u64().map(Some).ok_or_else(|| {
                decode(format!(
                    "the schema version at `{collection}`/`{id}` is not an unsigned integer: {doc}"
                ))
            });
        }
    }
    Ok(None)
}

pub(super) fn raise_format(txn: &mut Txn, to: u64) -> Result<(), StoreError> {
    let now = schema_version(txn)?.unwrap_or(crate::FORMAT_VERSION);
    if now < to {
        put_json(txn, SCHEMA_COLLECTION, SCHEMA_KEY, &to)?;
    }
    Ok(())
}

pub(super) fn refuse_escalated(txn: &Txn, primary: &Iri) -> Result<(), StoreError> {
    if let Some(tomb) = get_json::<Tombstone>(txn, TOMBSTONES, primary.as_str())? {
        return Err(StoreError::Escalated {
            from: primary.clone(),
            to: tomb.to,
        });
    }
    Ok(())
}

pub(super) fn refuse_unwritable(txn: &Txn, primary: &Iri, kind: Kind) -> Result<(), StoreError> {
    refuse_escalated(txn, primary)?;
    if get_json::<Mark>(txn, ESCALATING, primary.as_str())?.is_some() {
        return Err(StoreError::Escalating {
            id: primary.clone(),
            to_finish: escalate_command(kind, primary),
        });
    }
    Ok(())
}

pub(super) fn tombstoned(txn: &Txn) -> Result<BTreeSet<String>, StoreError> {
    Ok(txn.scan(TOMBSTONES).into_iter().map(|(id, _)| id).collect())
}

pub(super) fn routed_store_is_taken() -> StoreError {
    backend(
        "this store holds a routed project, which needs a store of its own: handles are \
         numbered per store. Give the new project its own store — a config entry with its own \
         `store`",
    )
}

pub(super) fn holds_routing(txn: &Txn) -> bool {
    !txn.scan(ROUTING).is_empty()
}

/// Entries in `index` with an id greater than `after` that `repo` has not marked.
pub(super) fn waiting<T: DeserializeOwned>(
    txn: &Txn,
    index: &str,
    log: &str,
    repo: &str,
    after: &Iri,
) -> Result<Vec<T>, StoreError> {
    let mut rows = Vec::new();
    for (id, seq) in txn.scan(index) {
        if id.as_str() <= after.as_str() {
            continue;
        }
        let seq: u64 = serde_json::from_value(seq).map_err(decode)?;
        if txn.get(LEDGER_PUBLISHED, &pair(repo, &id)).is_some() {
            continue;
        }
        let Some(row) = get_json::<T>(txn, log, &key_u64(seq))? else {
            return Err(decode(format!(
                "the ledger's candidate index names {id}, whose row is missing"
            )));
        };
        rows.push((id, row));
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(rows.into_iter().map(|(_, row)| row).collect())
}

pub(super) fn drop_candidates(txn: &mut Txn, ids: &[Iri]) {
    for index in [CANDIDATE_RUNS, CANDIDATE_ATTEMPTS] {
        for id in ids {
            txn.delete(index, id.as_str());
        }
    }
}

pub(super) fn refuse_a_changed_root(
    node_id: &str,
    held: Option<String>,
    commit: &str,
) -> Result<(), StoreError> {
    match held {
        Some(held) if held != commit => Err(StoreError::LedgerRootChanged {
            node_id: node_id.to_string(),
            held,
            found: commit.to_string(),
        }),
        _ => Ok(()),
    }
}

pub(super) fn record_ledger_root(
    txn: &mut Txn,
    node_id: &str,
    commit: &str,
) -> Result<(), StoreError> {
    put_json(txn, LEDGER_ROOTS, node_id, &commit)?;
    raise_format(txn, crate::FORMAT_WITH_LEDGER_ROOT)
}

pub(super) fn insert_in<T: Serialize>(
    txn: &mut Txn,
    id: Iri,
    kind: Kind,
    table: &str,
    raise: Option<u64>,
    build: impl FnOnce(&mut Txn, Iri) -> Result<T, StoreError>,
) -> Result<Iri, StoreError> {
    let value = build(txn, id.clone())?;
    index_new(txn, &id, kind)?;
    put_json(txn, table, id.as_str(), &value)?;
    if let Some(to) = raise {
        raise_format(txn, to)?;
    }
    Ok(id)
}

pub(super) fn insert_new<T: Serialize>(
    txn: &mut Txn,
    kind: Kind,
    table: &str,
    raise: Option<u64>,
    build: impl FnOnce(Iri) -> T,
) -> Result<Iri, StoreError> {
    insert_in(txn, mint()?, kind, table, raise, |_, id| Ok(build(id)))
}

pub(super) fn get_owned<T: DeserializeOwned>(
    txn: &Txn,
    label: &str,
    table: &str,
    key: &Iri,
) -> Result<Option<T>, StoreError> {
    let (target, _kind) = locate(txn, label, key)?;
    if let Some(tomb) = get_json::<Tombstone>(txn, TOMBSTONES, target.as_str())? {
        return Err(StoreError::Escalated {
            from: target,
            to: tomb.to,
        });
    }
    get_json(txn, table, target.as_str())
}

pub(super) fn rewrite<T, F>(
    txn: &mut Txn,
    label: &str,
    table: &str,
    id: &Iri,
    change: F,
) -> Result<bool, StoreError>
where
    T: Serialize + DeserializeOwned,
    F: FnOnce(T) -> T,
{
    let (primary, kind) = locate(txn, label, id)?;
    refuse_unwritable(txn, &primary, kind)?;
    let Some(old) = get_json::<T>(txn, table, primary.as_str())? else {
        return Ok(false);
    };
    put_json(txn, table, primary.as_str(), &change(old))?;
    Ok(true)
}

pub(super) fn put_unit(txn: &mut Txn, collection: &str, id: &str) {
    txn.put(collection, id, Value::Null);
}

pub(super) fn take_json<T: DeserializeOwned>(
    txn: &mut Txn,
    collection: &str,
    id: &str,
) -> Result<Option<T>, StoreError> {
    let Some(doc) = txn.get(collection, id) else {
        return Ok(None);
    };
    let parsed = serde_json::from_value(doc).map_err(decode)?;
    txn.delete(collection, id);
    Ok(Some(parsed))
}
