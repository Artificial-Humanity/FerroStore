//! Copy a redb database into an empty FerroStore in one write.
//!
//! A value redb stored with `serde_json::to_string` is parsed back to that
//! value. A raw `&str` is stored as a JSON string and is not parsed:
//! `from_str("1")` would turn a numeric token into a number.

use super::doc::{self, key_u64, pair, pair_u64, triple};
use super::{FerroStore, Txn};
use crate::{RedbStore, backend, decode};
use fl_core::StoreError;
use redb::{ReadableDatabase, ReadableTable, TableDefinition};
use serde_json::Value;

macro_rules! copy_rows {
    ($tx:expr, $def:expr, |$key:ident, $value:ident| $body:expr) => {{
        if let Some(table) = read_table(&$tx, $def)? {
            for row in table.iter().map_err(backend)? {
                let (key_guard, value_guard) = row.map_err(backend)?;
                let $key = key_guard.value();
                let $value = value_guard.value();
                let result: Result<(), StoreError> = $body;
                result?;
            }
        }
    }};
}

pub fn migrate_redb_to_ferrostore(redb: &RedbStore, ferro: &FerroStore) -> Result<(), StoreError> {
    ferro.with_write(|txn| {
        if !txn.docs.is_empty() || !txn.tables.is_empty() || !txn.vectors.is_empty() {
            return Err(backend(
                "refusing to migrate into a FerroStore that already holds a document, a column table, or a vector collection",
            ));
        }
        copy_tables(redb, txn)
    })
}

fn copy_tables(redb: &RedbStore, txn: &mut Txn) -> Result<(), StoreError> {
    let tx = redb.db.begin_read().map_err(backend)?;

    copy_rows!(
        tx,
        TableDefinition::<&str, u64>::new(doc::META),
        |key, value| { doc::put_json(txn, doc::META, key, &value) }
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::IDS),
        |key, value| put_raw(txn, doc::IDS, key, value)
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::ALIASES),
        |key, value| put_raw(txn, doc::ALIASES, key, value)
    );
    copy_rows!(
        tx,
        TableDefinition::<(&str, u64), &str>::new(doc::HANDLES),
        |key, value| {
            let (kind, n) = key;
            put_raw(txn, doc::HANDLES, &pair_u64(kind, n), value)
        }
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, u64>::new(doc::HANDLE_OF),
        |key, value| doc::put_json(txn, doc::HANDLE_OF, key, &value)
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::PROJECTS),
        |key, value| put_json_text(txn, doc::PROJECTS, key, value)
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::GATES),
        |key, value| put_json_text(txn, doc::GATES, key, value)
    );
    copy_rows!(
        tx,
        TableDefinition::<(&str, &str), &str>::new(doc::TRANSITIONS),
        |key, value| {
            let (project, name) = key;
            put_json_text(txn, doc::TRANSITIONS, &pair(project, name), value)
        }
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::RECORDS),
        |key, value| put_json_text(txn, doc::RECORDS, key, value)
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::FINDINGS),
        |key, value| put_json_text(txn, doc::FINDINGS, key, value)
    );
    copy_rows!(
        tx,
        TableDefinition::<u64, &str>::new(doc::GATE_RUNS),
        |key, value| put_json_text(txn, doc::GATE_RUNS, &key_u64(key), value)
    );
    copy_rows!(
        tx,
        TableDefinition::<u64, &str>::new(doc::ATTEMPTS),
        |key, value| put_json_text(txn, doc::ATTEMPTS, &key_u64(key), value)
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::IMPORTS),
        |key, value| put_raw(txn, doc::IMPORTS, key, value)
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::GITHUB_BINDINGS),
        |key, value| put_raw(txn, doc::GITHUB_BINDINGS, key, value)
    );
    copy_rows!(
        tx,
        TableDefinition::<(&str, &str), bool>::new(doc::LEDGER_PUBLISHED),
        |key, value| {
            let (repo, id) = key;
            doc::put_json(txn, doc::LEDGER_PUBLISHED, &pair(repo, id), &value)
        }
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::LEDGER_CUTOVERS),
        |key, value| put_raw(txn, doc::LEDGER_CUTOVERS, key, value)
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, u64>::new(doc::CANDIDATE_RUNS),
        |key, value| doc::put_json(txn, doc::CANDIDATE_RUNS, key, &value)
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, u64>::new(doc::CANDIDATE_ATTEMPTS),
        |key, value| doc::put_json(txn, doc::CANDIDATE_ATTEMPTS, key, &value)
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::LEDGER_ROOTS),
        |key, value| put_raw(txn, doc::LEDGER_ROOTS, key, value)
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::LEDGER_HEADS),
        |key, value| put_raw(txn, doc::LEDGER_HEADS, key, value)
    );
    copy_rows!(
        tx,
        TableDefinition::<(&str, &str), &str>::new(doc::LEDGER_SEGMENTS),
        |key, value| {
            let (repo, path) = key;
            put_json_text(txn, doc::LEDGER_SEGMENTS, &pair(repo, path), value)
        }
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::ROUTING),
        |key, value| put_json_text(txn, doc::ROUTING, key, value)
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::ESCALATING),
        |key, value| put_json_text(txn, doc::ESCALATING, key, value)
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::TOMBSTONES),
        |key, value| put_json_text(txn, doc::TOMBSTONES, key, value)
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::STATE_ENTRIES),
        |key, value| put_json_text(txn, doc::STATE_ENTRIES, key, value)
    );
    copy_rows!(
        tx,
        TableDefinition::<(&str, &str, &str), &str>::new(doc::STATE_KEYS),
        |key, value| {
            let (scope, concern, slug) = key;
            put_raw(txn, doc::STATE_KEYS, &triple(scope, concern, slug), value)
        }
    );
    copy_rows!(
        tx,
        TableDefinition::<u64, &str>::new(doc::STATE_REVISIONS),
        |key, value| put_json_text(txn, doc::STATE_REVISIONS, &key_u64(key), value)
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, u64>::new(doc::STATE_REVISION_IDS),
        |key, value| doc::put_json(txn, doc::STATE_REVISION_IDS, key, &value)
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::STATE_CONFLICTS),
        |key, value| put_json_text(txn, doc::STATE_CONFLICTS, key, value)
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::STATE_SYNC_CURSORS),
        |key, value| put_json_text(txn, doc::STATE_SYNC_CURSORS, key, value)
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::AGENTS),
        |key, value| put_json_text(txn, doc::AGENTS, key, value)
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::AGENT_BY_NAME),
        |key, value| put_raw(txn, doc::AGENT_BY_NAME, key, value)
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::AREAS),
        |key, value| put_json_text(txn, doc::AREAS, key, value)
    );
    copy_rows!(
        tx,
        TableDefinition::<u64, &str>::new(doc::AGENT_EVENTS),
        |key, value| put_json_text(txn, doc::AGENT_EVENTS, &key_u64(key), value)
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, u64>::new(doc::AGENT_EVENT_IDS),
        |key, value| doc::put_json(txn, doc::AGENT_EVENT_IDS, key, &value)
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::WIRE_MESSAGES),
        |key, value| put_json_text(txn, doc::WIRE_MESSAGES, key, value)
    );
    copy_rows!(
        tx,
        TableDefinition::<u64, &str>::new(doc::WIRE_MSG_LOG),
        |key, value| put_raw(txn, doc::WIRE_MSG_LOG, &key_u64(key), value)
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, u64>::new(doc::WIRE_MSG_SEQ),
        |key, value| doc::put_json(txn, doc::WIRE_MSG_SEQ, key, &value)
    );
    copy_rows!(
        tx,
        TableDefinition::<(&str, &str), &str>::new(doc::WIRE_INBOX),
        |key, value| {
            let (recipient, message) = key;
            put_json_text(txn, doc::WIRE_INBOX, &pair(recipient, message), value)
        }
    );
    copy_rows!(
        tx,
        TableDefinition::<u64, &str>::new(doc::WIRE_RCPT_LOG),
        |key, value| put_json_text(txn, doc::WIRE_RCPT_LOG, &key_u64(key), value)
    );
    copy_rows!(
        tx,
        TableDefinition::<(&str, &str), ()>::new(doc::WIRE_THREADS),
        |key, _value| {
            let (thread, message) = key;
            doc::put_unit(txn, doc::WIRE_THREADS, &pair(thread, message));
            Ok(())
        }
    );
    copy_rows!(
        tx,
        TableDefinition::<(&str, &str), ()>::new(doc::WIRE_REPLIES),
        |key, _value| {
            let (parent, message) = key;
            doc::put_unit(txn, doc::WIRE_REPLIES, &pair(parent, message));
            Ok(())
        }
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::WORKFLOW_DEFS),
        |key, value| put_json_text(txn, doc::WORKFLOW_DEFS, key, value)
    );
    copy_rows!(
        tx,
        TableDefinition::<(&str, &str), &str>::new(doc::WORKFLOW_SLUGS),
        |key, value| {
            let (project, slug) = key;
            put_raw(txn, doc::WORKFLOW_SLUGS, &pair(project, slug), value)
        }
    );
    copy_rows!(
        tx,
        TableDefinition::<&str, &str>::new(doc::WORKFLOW_RUNS),
        |key, value| put_json_text(txn, doc::WORKFLOW_RUNS, key, value)
    );
    Ok(())
}

/// `text` is the JSON `serde_json::to_string` wrote, not a raw token.
fn put_json_text(txn: &mut Txn, collection: &str, id: &str, text: &str) -> Result<(), StoreError> {
    let value: Value = serde_json::from_str(text).map_err(|err| {
        decode(format!(
            "collection `{collection}` id `{id}` is not JSON: {err}"
        ))
    })?;
    doc::put_json(txn, collection, id, &value)
}

fn put_raw(txn: &mut Txn, collection: &str, id: &str, text: &str) -> Result<(), StoreError> {
    doc::put_json(txn, collection, id, &text)
}

fn read_table<K, V>(
    tx: &redb::ReadTransaction,
    def: TableDefinition<K, V>,
) -> Result<Option<redb::ReadOnlyTable<K, V>>, StoreError>
where
    K: redb::Key + 'static,
    V: redb::Value + 'static,
{
    match tx.open_table(def) {
        Ok(table) => Ok(Some(table)),
        Err(redb::TableError::TableDoesNotExist(_)) => Ok(None),
        Err(err) => Err(backend(err)),
    }
}
