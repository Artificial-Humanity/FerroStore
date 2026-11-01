//! redb persistence for the `fl-core` store roles.

use fl_core::escalation::{EscalationFault, Escalations, Mark, Tombstone, escalate_command};
use fl_core::finding::{Finding, FindingState};
use fl_core::ids::{FindingId, GateId, Kind, ProjectId, RecordId};
use fl_core::iri::Iri;
use fl_core::log::{Attempt, GateRun};
use fl_core::model::{GateDef, GateKind, Project, Record, Selector, State, Transition};
use fl_core::routing::{ForeignRecord, Routes, RoutingMap, check_foreign_for_local};
use fl_core::split::{CachedSegment, LedgerCache, Outbox, Pending};
use fl_core::store::{Bindings, Catalog, Handles, Ledger, StoreError, Tracker};
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition, TableHandle};
use std::collections::BTreeSet;
use std::path::Path;

pub mod any;
pub mod cli;
pub mod code_knowledge;
pub mod ferrostore;
pub mod ferrowire;
pub mod knowledge;
pub mod manifest;
pub mod migrate;
pub mod private;
pub mod rollup;
pub mod state;
pub mod workflow;

#[cfg(test)]
mod manifest_conformance;

pub use any::{
    AnyStats, AnyStore, BACKEND_ENV, BackendKind, COMPACT_FERRO_ONLY, DEFAULT_BACKEND,
    MigrationLock, StoreOptions, Unmigrated, choose_backend, ferro_stats_at, forget_fallback,
    is_redb_busy, migrate_locked, open_redb, open_store, open_store_for_scan, open_store_with,
    physical_path, store_exists, store_identity,
};
pub use code_knowledge::{
    CodeSession, CodeSidecar, CodeSidecarCache, DEFAULT_CODE_SYNC_BUDGET, RankedCode, search_code,
    search_code_with, sidecar_path,
};
pub use ferrostore::{
    AttemptRollup, ColumnDef, ColumnFilter, ColumnPredicate, ColumnScan, ColumnType, ColumnValue,
    ConnectOptions, EnvelopeKind, FerroStore, FerroStoreServer, Fusion, GateRunRollup,
    KnowledgeHit, RecordEnvelope, ServerAuth, SyncAdapter, SyncBatch, SyncMode, SyncOp,
    Transaction, VectorHit, VectorMetric, VectorStorage, ZoneMap, default_socket_path_for_store,
    migrate_redb_to_ferrostore,
};
pub use knowledge::{
    DEFAULT_SYNC_BUDGET, IndexPolicy, RankedHits, SearchMode, rank_bm25, rank_state_entries,
    rank_state_entries_read_only, rank_state_entries_with,
};
pub use rollup::{
    AttemptTotals, GateRunTotals, TotalsSource, attempt_totals, attempt_totals_from_rows,
    gate_run_totals, gate_run_totals_from_rows,
};

use crate::manifest::{LedgerRoot, Manifest, ManifestError};

/// Format 2: ids are IRIs, with an ownership index and per-kind handles.
/// Format 1 keyed every table by a `u64` id from one shared counter.
pub const FORMAT_VERSION: u64 = 2;
/// What a store that imported a project cannot do to its routing map.
pub const CHANGE_ROUTING: &str = "change the routing map of";
const FORMAT_KEY: &str = "format_version";

pub(crate) const META: TableDefinition<&str, u64> = TableDefinition::new("meta");
/// Every id this store ever minted → its kind's wire name, OR the literal
/// `"alias"` for an id minted by `add_alias`. `"alias"` is not a `Kind`: it
/// is an index marker, so `check` must handle it before `Kind::from_wire`.
pub(crate) const IDS: TableDefinition<&str, &str> = TableDefinition::new("ids");
/// alias → primary, for every id `add_alias` minted. Never chains: `primary`
/// is always resolved to a true primary before it is written here.
pub(crate) const ALIASES: TableDefinition<&str, &str> = TableDefinition::new("aliases");
pub(crate) const HANDLES: TableDefinition<(&str, u64), &str> = TableDefinition::new("handles");
pub(crate) const HANDLE_OF: TableDefinition<&str, u64> = TableDefinition::new("handle_of");
const PROJECTS: TableDefinition<&str, &str> = TableDefinition::new("projects");
const GATES: TableDefinition<&str, &str> = TableDefinition::new("gates");
const TRANSITIONS: TableDefinition<(&str, &str), &str> = TableDefinition::new("transitions");
const RECORDS: TableDefinition<&str, &str> = TableDefinition::new("records");
const FINDINGS: TableDefinition<&str, &str> = TableDefinition::new("findings");
/// Ledger rows keep internal sequence keys. They are not ids and never leave
/// the store.
const GATE_RUNS: TableDefinition<u64, &str> = TableDefinition::new("gate_runs");
const ATTEMPTS: TableDefinition<u64, &str> = TableDefinition::new("attempts");
/// project → the `content_sha256` of the manifest it was imported from.
/// Created by the first import; a store without it has imported nothing
/// (GitHub tracker spec §4.2).
const IMPORTS: TableDefinition<&str, &str> = TableDefinition::new("imports");
/// configured `owner/repo` (lowercase) → the repository's `node_id`.
/// Additive like `imports`: created by the first bind, and a store without
/// it has bound nothing. An older fl ignores it, and cannot use GitHub mode.
const GITHUB_BINDINGS: TableDefinition<&str, &str> = TableDefinition::new("github_bindings");
/// (repository `node_id`, entry id) → published (GitHub ledger spec §3.2
/// step 6). Additive, like every table below: created by the first write,
/// and a store without it has none. An older fl ignores them all: it
/// publishes nothing.
const LEDGER_PUBLISHED: TableDefinition<(&str, &str), bool> =
    TableDefinition::new("ledger_published");
/// repository `node_id` → the id after which entries are publishable (spec
/// §2.1).
const LEDGER_CUTOVERS: TableDefinition<&str, &str> = TableDefinition::new("ledger_cutovers");
/// Entry id → row key in `gate_runs`, for every run that carries an id AND
/// a record — the only runs a flush may publish (spec §1.3, §2.1). Written
/// in the append's own transaction. A plain `fl check` run, and every run
/// from before ids, never enters it, so a flush never scans them.
const CANDIDATE_RUNS: TableDefinition<&str, u64> = TableDefinition::new("ledger_candidate_runs");
/// Entry id → row key in `attempts`, for every attempt that carries an id.
const CANDIDATE_ATTEMPTS: TableDefinition<&str, u64> =
    TableDefinition::new("ledger_candidate_attempts");

/// ⚠ The format of a store that holds an import. The first import raises
/// the store from 2 to 3 in the same transaction, so an older fl — which
/// knows nothing of imports and would let a person edit an imported gate —
/// REFUSES the store with `FormatVersion` instead of ignoring the mark.
/// This build opens both. A store that never imports stays 2 and still
/// opens in older builds.
pub const FORMAT_WITH_IMPORTS: u64 = 3;

/// ⚠ The format of a store that records a ledger root. An older fl opens a
/// store at 2 or 3 and would export its manifest WITHOUT the root — the
/// anchor every other machine checks the ledger against — so recording one
/// raises the store to 4, which an older fl refuses. This build opens 2, 3
/// and 4. A format is only ever raised (`raise_format`).
pub const FORMAT_WITH_LEDGER_ROOT: u64 = 4;

/// ⚠ The format of a store that holds a routing map, an item with an area,
/// or an escalation's mark or tombstone (routing spec §1.4, §3.6). An older
/// fl would read such an item and drop its area, route nothing, or ignore a
/// mark and write to the item it guards — so the first such write raises
/// the store to 5 in the same transaction, and an older fl refuses it. A
/// tombstone is written only over a mark, so its store is 5 already. This
/// build opens 2 to 5. A store that never routes stays where it was.
pub const FORMAT_WITH_ROUTING: u64 = 5;

/// ⚠ The format of a store that holds state entries or agent registry rows.
/// The first state or agent write raises the store to 6 in the same
/// transaction, so an older fl refuses the store with `FormatVersion` rather
/// than ignoring its state or agent tables. This build opens 2 to 6.
pub const FORMAT_WITH_STATE: u64 = 6;

/// ⚠ The format of a store that holds FerroWire messages or delivery receipts.
/// The first FerroWire write raises the store to 7 in the same transaction, so
/// an older fl refuses the store with `FormatVersion` rather than ignoring its
/// FerroWire tables. This build opens 2 to 7.
pub const FORMAT_WITH_WIRE: u64 = 7;

/// ⚠ The format of a store that holds workflow definitions or workflow runs.
/// The first workflow write raises the store to 8 in the same transaction, so
/// an older fl refuses the store with `FormatVersion` rather than ignoring its
/// workflow tables. This build opens 2 to 8.
pub const FORMAT_WITH_WORKFLOW: u64 = 8;

/// Every schema version this build opens, oldest first. Both `RedbStore` and
/// `FerroStore` use this one ladder, so `format_version()` means the same on
/// both; a new schema step adds a constant here and an upgrade for both.
pub(crate) const FORMAT_LADDER: [u64; 7] = [
    FORMAT_VERSION,
    FORMAT_WITH_IMPORTS,
    FORMAT_WITH_LEDGER_ROOT,
    FORMAT_WITH_ROUTING,
    FORMAT_WITH_STATE,
    FORMAT_WITH_WIRE,
    FORMAT_WITH_WORKFLOW,
];

/// `FormatVersion` unless `found` is on [`FORMAT_LADDER`].
pub(crate) fn check_format(found: Option<u64>) -> Result<(), StoreError> {
    match found {
        Some(v) if FORMAT_LADDER.contains(&v) => Ok(()),
        _ => Err(StoreError::FormatVersion {
            found,
            oldest: FORMAT_VERSION,
            newest: FORMAT_WITH_WORKFLOW,
        }),
    }
}

/// repository `node_id` → the first commit of its `fl/ledger` branch
/// (GitHub ledger spec §6.1 step 4). Created by the first root recorded.
const LEDGER_ROOTS: TableDefinition<&str, &str> = TableDefinition::new("ledger_roots");
/// repository `node_id` → the last head of its GitHub ledger this machine
/// checked (GitHub ledger spec §3.2 step 6). Additive.
const LEDGER_HEADS: TableDefinition<&str, &str> = TableDefinition::new("ledger_heads");
/// (repository `node_id`, path on the branch) → that file as last read, as
/// JSON (`CachedSegment`). Additive.
const LEDGER_SEGMENTS: TableDefinition<(&str, &str), &str> =
    TableDefinition::new("ledger_segments");

/// project → its routing map, as JSON (routing spec §1.2). Created by the
/// first map written; writing one raises the store to
/// [`FORMAT_WITH_ROUTING`] in the same transaction.
const ROUTING: TableDefinition<&str, &str> = TableDefinition::new("routing");
/// primary IRI → the mark of an escalation under way, as JSON (`Mark`;
/// routing spec §3.3 step 1). Created by the first mark, which raises the
/// store to [`FORMAT_WITH_ROUTING`] in the same transaction. While a mark
/// stands, every write to its item is refused.
const ESCALATING: TableDefinition<&str, &str> = TableDefinition::new("escalating");
/// primary IRI → the tombstone of an escalated item, as JSON (`Tombstone`;
/// routing spec §3.3 step 3). Created by the first tombstone. The item's
/// row, id, handle and aliases stay: its id still chooses this store, and
/// its handle still resolves (§2.3).
const TOMBSTONES: TableDefinition<&str, &str> = TableDefinition::new("tombstones");

const NEXT_RUN: &str = "next_run";
const NEXT_ATTEMPT: &str = "next_attempt";

pub struct RedbStore {
    pub(crate) db: Db,
    pub(crate) label: String,
}

/// The redb database behind a [`RedbStore`]: opened for writing, or, for a
/// store only read (see [`RedbStore::open_read_only`]), read-only. A read-only
/// open does not rewrite the file (opening it for writing does: its pages and
/// its mtime change), and refuses every write.
pub(crate) enum Db {
    Write(Database),
    Read(redb::ReadOnlyDatabase),
}

/// Why a write transaction could not begin.
#[derive(Debug)]
pub(crate) enum BeginWriteError {
    Redb(redb::TransactionError),
    ReadOnly,
}

impl std::fmt::Display for BeginWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Redb(e) => e.fmt(f),
            Self::ReadOnly => f.write_str("the store was opened read-only"),
        }
    }
}

impl Db {
    pub(crate) fn begin_read(&self) -> Result<redb::ReadTransaction, redb::TransactionError> {
        match self {
            Self::Write(db) => db.begin_read(),
            Self::Read(db) => db.begin_read(),
        }
    }

    pub(crate) fn begin_write(&self) -> Result<redb::WriteTransaction, BeginWriteError> {
        match self {
            Self::Write(db) => db.begin_write().map_err(BeginWriteError::Redb),
            Self::Read(_) => Err(BeginWriteError::ReadOnly),
        }
    }
}

/// What an import did, for the person who ran it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportReport {
    pub project: ProjectId,
    pub gates_added: usize,
    pub gates_changed: usize,
    pub gates_unchanged: usize,
    pub transitions: usize,
    /// Names of transitions this store held that the manifest no longer
    /// lists, and that the import therefore removed.
    pub transitions_removed: Vec<String>,
    /// `(old, new)` when a re-import came from another checkout. The
    /// project's gates now run over the new root, and the CLI says so.
    pub root_moved: Option<(String, String)>,
    /// How many areas the imported routing map declares, when it has one.
    pub areas: Option<usize>,
}

pub(crate) fn backend(e: impl std::fmt::Display) -> StoreError {
    StoreError::Backend(e.to_string())
}

/// Reading a stored value back into its type. Separate from [`backend`]
/// because the remedy is different: a decode failure is a version mismatch,
/// not a broken database.
pub(crate) fn decode(e: impl std::fmt::Display) -> StoreError {
    StoreError::Decode(e.to_string())
}

/// One alias hop, given already-open `IDS` and `ALIASES` tables. Shared by
/// [`RedbStore::locate`] (inside a read transaction) and `add_alias` (inside
/// a write transaction) — both `Table` and `ReadOnlyTable` implement
/// `ReadableTable`, so the same function serves either. `id` is already
/// known to name an alias (its own `IDS` entry reads `"alias"`); this
/// returns the primary it names and that primary's own kind, as a wire
/// string.
///
/// A damaged store — an alias with no `ALIASES` row, or one naming a primary
/// `IDS` no longer holds — is `Decode`, never a panic: on-disk corruption is
/// not a bug this build can rule out by construction, so it must be reported
/// the same way any other unreadable stored value is.
fn alias_primary(
    id: &Iri,
    ids: &impl ReadableTable<&'static str, &'static str>,
    aliases: &impl ReadableTable<&'static str, &'static str>,
) -> Result<(Iri, String), StoreError> {
    let Some(p) = aliases.get(id.as_str()).map_err(backend)? else {
        return Err(decode(format!("alias `{id}` has no primary recorded")));
    };
    let primary = Iri::parse(p.value()).map_err(decode)?;
    let Some(pk) = ids.get(primary.as_str()).map_err(backend)? else {
        return Err(decode(format!(
            "alias `{id}` names {primary}, which this store does not hold"
        )));
    };
    Ok((primary, pk.value().to_string()))
}

/// The primary id and kind behind `id`, read from `ids` — `IDS`, open in
/// any transaction — following one alias hop through the `ALIASES` table
/// `aliases` opens in the same one. `"alias"` is an index marker in `IDS`,
/// not a `Kind`, so it is handled here before `Kind::from_wire` sees it.
pub(crate) fn locate_in<A: ReadableTable<&'static str, &'static str>>(
    label: &str,
    id: &Iri,
    ids: &impl ReadableTable<&'static str, &'static str>,
    aliases: impl FnOnce() -> Result<A, StoreError>,
) -> Result<(Iri, Kind), StoreError> {
    let Some(v) = ids.get(id.as_str()).map_err(backend)? else {
        return Err(StoreError::NotOwned {
            id: id.clone(),
            searched: vec![label.to_string()],
        });
    };
    let (primary, wire) = if v.value() == "alias" {
        alias_primary(id, ids, &aliases()?)?
    } else {
        (id.clone(), v.value().to_string())
    };
    let kind = Kind::from_wire(&wire)
        .ok_or_else(|| decode(format!("unknown kind `{wire}` for {primary}")))?;
    Ok((primary, kind))
}

/// The JSON row `key` names in `table`, if it has one.
fn json_at<T: serde::de::DeserializeOwned>(
    table: &impl ReadableTable<&'static str, &'static str>,
    key: &Iri,
) -> Result<Option<T>, StoreError> {
    match table.get(key.as_str()).map_err(backend)? {
        Some(v) => Ok(Some(serde_json::from_str(v.value()).map_err(decode)?)),
        None => Ok(None),
    }
}

/// The JSON row `key` names in an additive `table`, read inside `tx`. A
/// table no write has created yet holds nothing.
pub(crate) fn additive<T: serde::de::DeserializeOwned>(
    tx: &redb::ReadTransaction,
    table: TableDefinition<&str, &str>,
    key: &Iri,
) -> Result<Option<T>, StoreError> {
    match tx.open_table(table) {
        Ok(t) => json_at(&t, key),
        Err(redb::TableError::TableDoesNotExist(_)) => Ok(None),
        Err(e) => Err(backend(e)),
    }
}

/// [`additive`], inside a write transaction. ⚠ A write transaction's
/// `open_table` creates the table, so this asks whether it exists first: a
/// write that only looks for a mark leaves a store that never marked
/// without the table.
fn additive_in_write<T: serde::de::DeserializeOwned>(
    tx: &redb::WriteTransaction,
    table: TableDefinition<&str, &str>,
    key: &Iri,
) -> Result<Option<T>, StoreError> {
    let exists = tx
        .list_tables()
        .map_err(backend)?
        .any(|t| t.name() == table.name());
    if !exists {
        return Ok(None);
    }
    json_at(&tx.open_table(table).map_err(backend)?, key)
}

/// `Escalated` when `primary` has a tombstone, read inside the write
/// transaction `tx` that is about to write (routing spec §3.6).
fn refuse_escalated(tx: &redb::WriteTransaction, primary: &Iri) -> Result<(), StoreError> {
    if let Some(t) = additive_in_write::<Tombstone>(tx, TOMBSTONES, primary)? {
        return Err(StoreError::Escalated {
            from: primary.clone(),
            to: t.to,
        });
    }
    Ok(())
}

/// Refuse a write to `primary`, a `kind`, inside the write transaction `tx`
/// that would make it: a tombstoned item is `Escalated`, and a marked one is
/// `Escalating` — a write would change the copy the escalation is moving
/// (routing spec §3.3 step 1, §3.6).
fn refuse_unwritable(
    tx: &redb::WriteTransaction,
    primary: &Iri,
    kind: Kind,
) -> Result<(), StoreError> {
    refuse_escalated(tx, primary)?;
    if additive_in_write::<Mark>(tx, ESCALATING, primary)?.is_some() {
        return Err(StoreError::Escalating {
            id: primary.clone(),
            to_finish: escalate_command(kind, primary),
        });
    }
    Ok(())
}

/// A new item's id.
pub(crate) fn mint() -> Result<Iri, StoreError> {
    Iri::parse(&format!("urn:uuid:{}", uuid::Uuid::now_v7())).map_err(backend)
}

/// Index `id` as a new item of `kind` and give it the next handle, inside a
/// write transaction the caller already holds. Refuses an id already
/// indexed: an insert never overwrites. Shared by `insert_new_with_id` and
/// `import_manifest`, so a minted id and an imported id are indexed the same
/// way.
pub(crate) fn index_new(
    tx: &redb::WriteTransaction,
    id: &Iri,
    kind: Kind,
) -> Result<(), StoreError> {
    let mut ids = tx.open_table(IDS).map_err(backend)?;
    if ids.get(id.as_str()).map_err(backend)?.is_some() {
        return Err(StoreError::AlreadyExists(id.clone()));
    }
    ids.insert(id.as_str(), kind.as_wire()).map_err(backend)?;

    let key = format!("next_handle:{}", kind.as_wire());
    let mut meta = tx.open_table(META).map_err(backend)?;
    let n = meta
        .get(key.as_str())
        .map_err(backend)?
        .map(|v| v.value())
        .unwrap_or(0)
        + 1;
    meta.insert(key.as_str(), n).map_err(backend)?;

    tx.open_table(HANDLES)
        .map_err(backend)?
        .insert((kind.as_wire(), n), id.as_str())
        .map_err(backend)?;
    tx.open_table(HANDLE_OF)
        .map_err(backend)?
        .insert(id.as_str(), n)
        .map_err(backend)?;
    Ok(())
}

/// Raise the store's format to at least `to`, inside `tx`. Never lowers it:
/// an import into a store that records a ledger root must leave it at 4.
pub(crate) fn raise_format(tx: &redb::WriteTransaction, to: u64) -> Result<(), StoreError> {
    let mut meta = tx.open_table(META).map_err(backend)?;
    let now = meta
        .get(FORMAT_KEY)
        .map_err(backend)?
        .map(|v| v.value())
        .unwrap_or(FORMAT_VERSION);
    if now < to {
        meta.insert(FORMAT_KEY, to).map_err(backend)?;
    }
    Ok(())
}

/// ⚠ An anchor never changes (spec §3.5): a different root for a
/// `node_id` that has one is refused. Asked BEFORE any write, by both
/// callers of [`record_ledger_root`].
fn refuse_a_changed_root(
    node_id: &str,
    held: Option<String>,
    commit: &str,
) -> Result<(), StoreError> {
    match held {
        Some(h) if h != commit => Err(StoreError::LedgerRootChanged {
            node_id: node_id.to_string(),
            held: h,
            found: commit.to_string(),
        }),
        _ => Ok(()),
    }
}

/// Record `commit` as `node_id`'s ledger root inside `tx`, and raise the
/// store to [`FORMAT_WITH_LEDGER_ROOT`]. The caller has already refused a
/// changed root.
fn record_ledger_root(
    tx: &redb::WriteTransaction,
    node_id: &str,
    commit: &str,
) -> Result<(), StoreError> {
    tx.open_table(LEDGER_ROOTS)
        .map_err(backend)?
        .insert(node_id, commit)
        .map_err(backend)?;
    raise_format(tx, FORMAT_WITH_LEDGER_ROOT)
}

/// The entries `index` lists with an id after `after` that `repo` has not
/// marked, read from `log` by their row key, in id order.
fn waiting<T: serde::de::DeserializeOwned>(
    tx: &redb::ReadTransaction,
    index: TableDefinition<&str, u64>,
    log: TableDefinition<u64, &str>,
    repo: &str,
    after: &Iri,
) -> Result<Vec<T>, StoreError> {
    let candidates = match tx.open_table(index) {
        Ok(t) => t,
        Err(redb::TableError::TableDoesNotExist(_)) => return Ok(vec![]),
        Err(e) => return Err(backend(e)),
    };
    let marks = match tx.open_table(LEDGER_PUBLISHED) {
        Ok(t) => Some(t),
        Err(redb::TableError::TableDoesNotExist(_)) => None,
        Err(e) => return Err(backend(e)),
    };
    let rows = tx.open_table(log).map_err(backend)?;
    let mut out = Vec::new();
    for entry in candidates.range(after.as_str()..).map_err(backend)? {
        let (id, seq) = entry.map_err(backend)?;
        let id = id.value();
        if id == after.as_str() {
            continue;
        }
        if let Some(m) = &marks
            && m.get((repo, id)).map_err(backend)?.is_some()
        {
            continue;
        }
        let Some(row) = rows.get(seq.value()).map_err(backend)? else {
            return Err(decode(format!(
                "the ledger's candidate index names {id}, whose row is missing"
            )));
        };
        out.push(serde_json::from_str(row.value()).map_err(decode)?);
    }
    Ok(out)
}

/// Drop `ids` from the candidate index inside `tx`: published or set aside,
/// they wait no more, and no flush scans them again (spec §2.1).
fn drop_candidates(tx: &redb::WriteTransaction, ids: &[Iri]) -> Result<(), StoreError> {
    for index in [CANDIDATE_RUNS, CANDIDATE_ATTEMPTS] {
        let mut table = tx.open_table(index).map_err(backend)?;
        for id in ids {
            table.remove(id.as_str()).map_err(backend)?;
        }
    }
    Ok(())
}

/// Routing spec decision 20: a store whose project is routed takes no
/// other project.
fn routed_store_is_taken() -> StoreError {
    backend(
        "this store holds a routed project, which needs a store of its own: handles are \
         numbered per store. Give the new project its own store — a config entry with its own \
         `store`",
    )
}

/// Why `path` could not be opened, when a FerroStore migration made it
/// read-only: the open was refused for permission and the file has no owner
/// write bit, the mode `migrate_verified` leaves a migrated `.redb` in.
/// `None` for any other failure.
fn frozen_by_migration(path: &Path, err: &redb::DatabaseError) -> Option<String> {
    use std::os::unix::fs::PermissionsExt;
    let redb::DatabaseError::Storage(redb::StorageError::Io(io)) = err else {
        return None;
    };
    if io.kind() != std::io::ErrorKind::PermissionDenied {
        return None;
    }
    let mode = std::fs::metadata(path).ok()?.permissions().mode();
    if mode & 0o200 != 0 {
        return None;
    }
    let fls = physical_path(path, BackendKind::Ferro);
    let fls_shown = private::shown_path(&fls);
    let found = if fls.exists() {
        "it exists"
    } else {
        "it does not exist"
    };
    let rollback = match private::quoted_path(path) {
        Some(quoted) => format!("run: chmod 600 {quoted}"),
        None => "run chmod 600 on it yourself; its name contains characters that cannot be \
                 pasted safely"
            .to_string(),
    };
    Some(format!(
        "{io}: {redb} is read-only (mode {mode:o}), as a FerroStore migration leaves it once it \
         has moved the store to {fls_shown} ({found}). Use the FerroStore engine: open \
         {fls_shown}, or set {BACKEND_ENV}=ferro. To roll back to redb instead, {rollback}",
        mode = mode & 0o7777,
        redb = private::shown_path(path),
    ))
}

/// A shared lock on a redb file, held while this value lives. It reads the file
/// and writes nothing: the file may be 0400, and its bytes and mtime stay as
/// they are (opening a [`RedbStore`] rewrites internal pages). It is refused
/// while any process holds the file open for write, as a [`RedbStore`] open
/// does, so [`is_redb_busy`] recognises that refusal. A caller that copies the
/// file holds this across the copy and gets a copy no writer is changing.
pub struct RedbSharedLock {
    _db: redb::ReadOnlyDatabase,
}

impl RedbSharedLock {
    pub fn acquire(path: &Path) -> Result<Self, StoreError> {
        redb::ReadOnlyDatabase::open(path)
            .map(|db| Self { _db: db })
            .map_err(|e| StoreError::Unreachable {
                store: path.display().to_string(),
                cause: e.to_string(),
            })
    }
}

impl RedbStore {
    pub fn open(path: &Path) -> Result<Self, StoreError> {
        Self::open_with_retries(path, 200)
    }

    /// [`RedbStore::open`] with a bound on how long it waits for another holder of the file:
    /// it retries an "already open" file `retries` times, 15 ms apart, then fails. `open` is
    /// this with 200 retries (about 3 s).
    pub fn open_with_retries(path: &Path, retries: u32) -> Result<Self, StoreError> {
        let label = path.display().to_string();
        // A new store file is private from its first byte: create it empty with mode 0600 so
        // `Database::create` finds a file it already accepts as a new database. An existing file
        // is never touched, and any error here is left to `Database::create` to report.
        let _ = private::create_private_file(path);
        let mut attempts = 0u32;
        let db = loop {
            match Database::create(path) {
                Ok(db) => break db,
                Err(e)
                    if attempts < retries
                        && (matches!(e, redb::DatabaseError::DatabaseAlreadyOpen)
                            || e.to_string().contains("Database already open")
                            || e.to_string().contains("Cannot acquire lock")) =>
                {
                    attempts += 1;
                    std::thread::sleep(std::time::Duration::from_millis(15));
                }
                Err(e) => {
                    let cause = frozen_by_migration(path, &e).unwrap_or_else(|| e.to_string());
                    return Err(StoreError::Unreachable {
                        store: label.clone(),
                        cause,
                    });
                }
            }
        };

        // ⚠ Read before writing. A missing version key must not read as
        // "fresh": that would make an old store look empty — a vacuous pass.
        let found = {
            let tx = db.begin_read().map_err(backend)?;
            match tx.open_table(META) {
                Ok(meta) => Some(meta.get(FORMAT_KEY).map_err(backend)?.map(|v| v.value())),
                Err(redb::TableError::TableDoesNotExist(_)) => None,
                Err(e) => return Err(backend(e)),
            }
        };
        match found {
            // No META table at all: a brand-new file.
            None => Self::create_tables(&db)?,
            Some(v) => check_format(v)?,
        }
        // Only a report, and only after the open worked: a store others can reach is the owner's
        // to fix, and a stat error here is no reason to fail the open.
        private::warn_if_loose(path);
        Ok(Self {
            db: Db::Write(db),
            label,
        })
    }

    /// Open the redb file at `path` for reading only, waiting up to `retries`
    /// times, 15 ms apart, for a process that holds it for writing. Nothing is
    /// created, written or touched (not its bytes, not its mtime); every
    /// write through the store fails. A file that is not an initialised store
    /// of a known format, or that needs repair, is an error here (the
    /// ordinary [`RedbStore::open`] decides what to do with it).
    pub fn open_read_only(path: &Path, retries: u32) -> Result<Self, StoreError> {
        let label = path.display().to_string();
        let mut attempts = 0u32;
        let db = loop {
            match redb::ReadOnlyDatabase::open(path) {
                Ok(db) => break db,
                Err(e)
                    if attempts < retries
                        && (e.to_string().contains("Database already open")
                            || e.to_string().contains("Cannot acquire lock")) =>
                {
                    attempts += 1;
                    std::thread::sleep(std::time::Duration::from_millis(15));
                }
                Err(e) => {
                    return Err(StoreError::Unreachable {
                        store: label,
                        cause: e.to_string(),
                    });
                }
            }
        };
        let found = {
            let tx = db.begin_read().map_err(backend)?;
            match tx.open_table(META) {
                Ok(meta) => meta.get(FORMAT_KEY).map_err(backend)?.map(|v| v.value()),
                Err(redb::TableError::TableDoesNotExist(_)) => {
                    return Err(backend(format!("{label} is not an initialised store")));
                }
                Err(e) => return Err(backend(e)),
            }
        };
        check_format(found)?;
        Ok(Self {
            db: Db::Read(db),
            label,
        })
    }

    fn create_tables(db: &Database) -> Result<(), StoreError> {
        let tx = db.begin_write().map_err(backend)?;
        {
            let mut meta = tx.open_table(META).map_err(backend)?;
            meta.insert(FORMAT_KEY, FORMAT_VERSION).map_err(backend)?;
        }
        {
            tx.open_table(IDS).map_err(backend)?;
            tx.open_table(ALIASES).map_err(backend)?;
            tx.open_table(HANDLES).map_err(backend)?;
            tx.open_table(HANDLE_OF).map_err(backend)?;
            tx.open_table(PROJECTS).map_err(backend)?;
            tx.open_table(GATES).map_err(backend)?;
            tx.open_table(TRANSITIONS).map_err(backend)?;
            tx.open_table(RECORDS).map_err(backend)?;
            tx.open_table(FINDINGS).map_err(backend)?;
            tx.open_table(GATE_RUNS).map_err(backend)?;
            tx.open_table(ATTEMPTS).map_err(backend)?;
        }
        tx.commit().map_err(backend)
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    fn insert_new<T: serde::Serialize>(
        &self,
        kind: Kind,
        table: TableDefinition<&str, &str>,
        raise: Option<u64>,
        build: impl FnOnce(Iri) -> T,
    ) -> Result<Iri, StoreError> {
        self.insert_new_with_id(mint()?, kind, table, raise, build)
    }

    /// `insert_new`, with `build` given the write transaction: what the new
    /// row depends on is read — and refused — in the write that adds it.
    fn insert_new_in<T: serde::Serialize>(
        &self,
        kind: Kind,
        table: TableDefinition<&str, &str>,
        raise: Option<u64>,
        build: impl FnOnce(&redb::WriteTransaction, Iri) -> Result<T, StoreError>,
    ) -> Result<Iri, StoreError> {
        self.insert_in(mint()?, kind, table, raise, build)
    }

    /// Mint, index, hand out a handle, and write the row — in ONE write
    /// transaction, so they land together or not at all (the lesson of the old
    /// `bump_and_put`). `pub(crate)` so a test can force a collision.
    ///
    /// That lesson: an earlier version of this store bumped its id counter in
    /// its own transaction (committing immediately) and then wrote the row in
    /// a second one. If the row write failed in between, the counter's commit
    /// had already landed — an id burned with no row behind it, a silent side
    /// effect riding along on a call that reported failure. Here an error
    /// anywhere before `commit()` drops the transaction (`redb`'s
    /// `WriteTransaction` aborts on `Drop` when it was not completed), which
    /// discards the `IDS` entry and the handle bump along with everything
    /// else.
    pub(crate) fn insert_new_with_id<T: serde::Serialize>(
        &self,
        id: Iri,
        kind: Kind,
        table: TableDefinition<&str, &str>,
        raise: Option<u64>,
        build: impl FnOnce(Iri) -> T,
    ) -> Result<Iri, StoreError> {
        self.insert_in(id, kind, table, raise, |_, id| Ok(build(id)))
    }

    /// `insert_new_with_id`'s one write transaction, with `build` run inside
    /// it: an error from `build` drops the transaction like any other.
    fn insert_in<T: serde::Serialize>(
        &self,
        id: Iri,
        kind: Kind,
        table: TableDefinition<&str, &str>,
        raise: Option<u64>,
        build: impl FnOnce(&redb::WriteTransaction, Iri) -> Result<T, StoreError>,
    ) -> Result<Iri, StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        let json = serde_json::to_string(&build(&tx, id.clone())?).map_err(backend)?;
        index_new(&tx, &id, kind)?;
        tx.open_table(table)
            .map_err(backend)?
            .insert(id.as_str(), json.as_str())
            .map_err(backend)?;
        if let Some(to) = raise {
            raise_format(&tx, to)?;
        }
        tx.commit().map_err(backend)?;
        Ok(id)
    }

    /// The primary id and kind behind `id`, following one alias hop if `id`
    /// names an alias rather than a primary. `"alias"` is an index marker in
    /// `IDS`, not a `Kind`, so it is handled here before `Kind::from_wire`
    /// ever sees it.
    ///
    /// ⚠ Every method that takes an id asks this (via [`Self::check`] or
    /// directly) first — list methods too. A list over a project this store
    /// never held is "didn't look", and an empty list would say "looked,
    /// found nothing".
    pub(crate) fn locate(&self, id: &Iri) -> Result<(Iri, Kind), StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let ids = tx.open_table(IDS).map_err(backend)?;
        locate_in(&self.label, id, &ids, || {
            tx.open_table(ALIASES).map_err(backend)
        })
    }

    /// [`Self::locate`], inside the write transaction `tx` that is about to
    /// write: a check made here holds when the write lands.
    pub(crate) fn locate_for_write(
        &self,
        tx: &redb::WriteTransaction,
        id: &Iri,
    ) -> Result<(Iri, Kind), StoreError> {
        let ids = tx.open_table(IDS).map_err(backend)?;
        locate_in(&self.label, id, &ids, || {
            tx.open_table(ALIASES).map_err(backend)
        })
    }

    /// The primary `id` names, or `None` when this store does not hold it.
    fn held(&self, id: &Iri) -> Result<Option<Iri>, StoreError> {
        match self.locate(id) {
            Ok((primary, _)) => Ok(Some(primary)),
            Err(StoreError::NotOwned { .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// The primary IRIs this store holds a tombstone for: what every list
    /// leaves out (routing spec §2.4).
    fn tombstoned(&self) -> Result<BTreeSet<String>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let table = match tx.open_table(TOMBSTONES) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(BTreeSet::new()),
            Err(e) => return Err(backend(e)),
        };
        let mut out = BTreeSet::new();
        for entry in table.iter().map_err(backend)? {
            let (k, _) = entry.map_err(backend)?;
            out.insert(k.value().to_string());
        }
        Ok(out)
    }

    /// Change the row `id` names in `table` with `change` — in ONE write
    /// transaction that first refuses an escalated or escalating item
    /// (routing spec §3.3 step 1, §3.6), so nothing lands between the check
    /// and the write. The row is keyed by the primary, whatever name `id`
    /// is. `false`, writing nothing, when `table` holds no row for it: an id
    /// held under another kind.
    fn rewrite<T: serde::Serialize + serde::de::DeserializeOwned>(
        &self,
        table: TableDefinition<&str, &str>,
        id: &Iri,
        change: impl FnOnce(T) -> T,
    ) -> Result<bool, StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        let (primary, kind) = self.locate_for_write(&tx, id)?;
        refuse_unwritable(&tx, &primary, kind)?;
        {
            let mut t = tx.open_table(table).map_err(backend)?;
            let Some(old) = json_at::<T>(&t, &primary)? else {
                return Ok(false);
            };
            let json = serde_json::to_string(&change(old)).map_err(backend)?;
            t.insert(primary.as_str(), json.as_str()).map_err(backend)?;
        }
        tx.commit().map_err(backend)?;
        Ok(true)
    }

    /// The kind this store holds `id` under, or `NotOwned` naming this store.
    /// An alias is owned too: this follows it to its primary's kind.
    fn check(&self, id: &Iri) -> Result<Kind, StoreError> {
        self.locate(id).map(|(_, kind)| kind)
    }

    /// `check`, and then refuse an id held under any kind but `expected`
    /// with `WrongKind`. Every method that takes a project (and
    /// `add_finding`'s record) asks this rather than `check`: answering an
    /// empty list for a gate's IRI would say "looked, found nothing" about
    /// an item that was never a project.
    fn check_kind(&self, id: &Iri, expected: Kind) -> Result<(), StoreError> {
        let found = self.check(id)?;
        if found != expected {
            return Err(StoreError::WrongKind {
                id: id.clone(),
                expected,
                found,
            });
        }
        Ok(())
    }

    /// Whether this store holds `id`. An I/O failure is an error, never
    /// `false`: "could not look" is not "not mine".
    pub fn owns(&self, id: &Iri) -> Result<bool, StoreError> {
        match self.check(id) {
            Ok(_) => Ok(true),
            Err(StoreError::NotOwned { .. }) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Log rows are keyed by an internal sequence, not by an id: they are
    /// never addressed from outside the store. A `candidate` — an index and
    /// the entry's id — is written in the same transaction as the row.
    fn append_json<T: serde::Serialize>(
        &self,
        counter: &str,
        table: TableDefinition<u64, &str>,
        value: &T,
        candidate: Option<(TableDefinition<&str, u64>, &Iri)>,
    ) -> Result<(), StoreError> {
        let json = serde_json::to_string(value).map_err(backend)?;
        let tx = self.db.begin_write().map_err(backend)?;
        {
            let mut meta = tx.open_table(META).map_err(backend)?;
            let seq = meta
                .get(counter)
                .map_err(backend)?
                .map(|v| v.value())
                .unwrap_or(0)
                + 1;
            meta.insert(counter, seq).map_err(backend)?;
            tx.open_table(table)
                .map_err(backend)?
                .insert(seq, json.as_str())
                .map_err(backend)?;
            if let Some((index, id)) = candidate {
                tx.open_table(index)
                    .map_err(backend)?
                    .insert(id.as_str(), seq)
                    .map_err(backend)?;
            }
        }
        tx.commit().map_err(backend)
    }

    fn put_json<T: serde::Serialize>(
        &self,
        table: TableDefinition<&str, &str>,
        key: &Iri,
        value: &T,
    ) -> Result<(), StoreError> {
        let json = serde_json::to_string(value).map_err(backend)?;
        let tx = self.db.begin_write().map_err(backend)?;
        {
            let mut t = tx.open_table(table).map_err(backend)?;
            t.insert(key.as_str(), json.as_str()).map_err(backend)?;
        }
        tx.commit().map_err(backend)?;
        Ok(())
    }

    /// Checks ownership first, so an id this store never held is `NotOwned`,
    /// and an id it holds under another kind is `Ok(None)`. `key` may be an
    /// alias: `locate` resolves it to the primary the row is stored under,
    /// which is what this reads — a lookup by alias answers as the primary
    /// would (spec §2.5).
    fn get_json<T: serde::de::DeserializeOwned>(
        &self,
        table: TableDefinition<&str, &str>,
        key: &Iri,
    ) -> Result<Option<T>, StoreError> {
        let (target, _kind) = self.locate(key)?;
        let tx = self.db.begin_read().map_err(backend)?;
        // An escalated item reads as `Escalated`, under every name it has
        // (routing spec §3.6).
        if let Some(t) = additive::<Tombstone>(&tx, TOMBSTONES, &target)? {
            return Err(StoreError::Escalated {
                from: target,
                to: t.to,
            });
        }
        let t = tx.open_table(table).map_err(backend)?;
        // `ReadOnlyTable::get_owned` (unlike `Table::get`) keeps the read
        // transaction alive via a reference-counted guard, so the returned
        // value can outlive the local borrow of `t`.
        let Some(v) = t.get_owned(target.as_str()).map_err(backend)? else {
            return Ok(None);
        };
        let parsed = serde_json::from_str(v.value()).map_err(decode)?;
        Ok(Some(parsed))
    }

    fn all_json<T: serde::de::DeserializeOwned>(
        &self,
        table: TableDefinition<&str, &str>,
    ) -> Result<Vec<T>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let t = tx.open_table(table).map_err(backend)?;
        let mut out = Vec::new();
        for entry in t.iter().map_err(backend)? {
            let (_k, v) = entry.map_err(backend)?;
            out.push(serde_json::from_str(v.value()).map_err(decode)?);
        }
        Ok(out)
    }

    fn all_log<T: serde::de::DeserializeOwned>(
        &self,
        table: TableDefinition<u64, &str>,
    ) -> Result<Vec<T>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let t = tx.open_table(table).map_err(backend)?;
        let mut out = Vec::new();
        for entry in t.iter().map_err(backend)? {
            let (_k, v) = entry.map_err(backend)?;
            out.push(serde_json::from_str(v.value()).map_err(decode)?);
        }
        Ok(out)
    }

    /// The hash of the manifest `project` was imported from, or `None` if
    /// this store authors it. A project this store never held is `NotOwned`:
    /// `None` would read as "authored here".
    pub fn imported_hash(&self, project: &ProjectId) -> Result<Option<String>, StoreError> {
        self.check_kind(project.iri(), Kind::Project)?;
        let tx = self.db.begin_read().map_err(backend)?;
        let table = match tx.open_table(IMPORTS) {
            Ok(t) => t,
            // The first import creates the table. Its absence is "nothing
            // imported", read from a table that does not exist yet — not a
            // failure to look.
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(e) => return Err(backend(e)),
        };
        let hash = table
            .get(project.iri().as_str())
            .map_err(backend)?
            .map(|v| v.value().to_string());
        Ok(hash)
    }

    /// `Imported`, naming `action`, when this store imported `project`
    /// from a manifest: its definitions are authored elsewhere.
    pub fn refuse_if_imported(
        &self,
        project: &ProjectId,
        action: &'static str,
    ) -> Result<(), StoreError> {
        if self.imported_hash(project)?.is_some() {
            return Err(StoreError::Imported {
                id: project.iri().clone(),
                action,
            });
        }
        Ok(())
    }

    /// Export `project`, which this store must author. `repository_node_id`
    /// is the repository the project's ledger is in, when its binding names
    /// one: the export carries that repository's ledger root, if this store
    /// records one (GitHub ledger spec §6.1 step 4).
    pub fn export_manifest(
        &self,
        project: &ProjectId,
        commit: &str,
        exported_at_unix: u64,
        repository_node_id: Option<&str>,
    ) -> Result<Manifest, ManifestError> {
        self.check_kind(project.iri(), Kind::Project)?;
        if self.imported_hash(project)?.is_some() {
            return Err(ManifestError::NotAuthoring(project.clone()));
        }
        let ledger_root = match repository_node_id {
            None => None,
            Some(node) => self.ledger_root(node)?.map(|commit| LedgerRoot {
                repository_node_id: node.to_string(),
                commit,
            }),
        };
        manifest::export(
            self,
            project,
            commit,
            exported_at_unix,
            ledger_root,
            self.routes(project)?,
        )
    }

    /// The format version the store file records, `None` for a file with no version key.
    pub fn format_version(&self) -> Result<Option<u64>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let meta = match tx.open_table(META) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(e) => return Err(backend(e)),
        };
        let version = meta.get(FORMAT_KEY).map_err(backend)?.map(|v| v.value());
        Ok(version)
    }

    /// Whether this store records any ledger root. An export that cannot
    /// tell which repository its project's ledger is in asks this before it
    /// writes a manifest without one.
    pub fn holds_a_ledger_root(&self) -> Result<bool, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let table = match tx.open_table(LEDGER_ROOTS) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(false),
            Err(e) => return Err(backend(e)),
        };
        let any = table.iter().map_err(backend)?.next().is_some();
        Ok(any)
    }

    /// Write `project`'s routing map (routing spec §1.2) — which this store
    /// must author, and which must be its only project (decision 20) — and
    /// raise the store to format 5, in one transaction.
    pub fn set_routes(&self, project: &ProjectId, map: &RoutingMap) -> Result<(), StoreError> {
        self.check_kind(project.iri(), Kind::Project)?;
        self.refuse_if_imported(project, CHANGE_ROUTING)?;
        self.refuse_a_second_project(project)?;
        map.check()
            .map_err(|why| backend(format!("the routing map is not valid: {why}")))?;
        let json = serde_json::to_string(map).map_err(backend)?;
        let tx = self.db.begin_write().map_err(backend)?;
        tx.open_table(ROUTING)
            .map_err(backend)?
            .insert(project.iri().as_str(), json.as_str())
            .map_err(backend)?;
        raise_format(&tx, FORMAT_WITH_ROUTING)?;
        tx.commit().map_err(backend)
    }

    /// Routing spec decision 20: a routed store holds exactly one project,
    /// because handles are numbered per store. `Err` names the other
    /// project and the remedy.
    pub(crate) fn refuse_a_second_project(&self, project: &ProjectId) -> Result<(), StoreError> {
        match self.list_projects()?.into_iter().find(|p| p.id != *project) {
            Some(other) => Err(backend(format!(
                "this store also holds project {} (root {}), and a routed project needs a store \
                 of its own: handles are numbered per store, so one number would name items of \
                 both. Give the project its own store — a config entry with its own `store` — \
                 and route it there",
                other.id, other.root
            ))),
            None => Ok(()),
        }
    }

    /// Whether any project in this store has a routing map: the store is
    /// then routed (routing spec §1.3), whatever the map holds.
    pub fn holds_routing(&self) -> Result<bool, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let table = match tx.open_table(ROUTING) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(false),
            Err(e) => return Err(backend(e)),
        };
        let any = table.iter().map_err(backend)?.next().is_some();
        Ok(any)
    }

    /// Write the manifest's project, gates and transitions under their own
    /// IRIs, and mark the project imported — in ONE write transaction.
    /// Everything that can refuse is decided before anything is written.
    pub fn import_manifest(&self, m: &Manifest, root: &str) -> Result<ImportReport, ManifestError> {
        // A `Manifest` can be built by hand; the store checks it itself.
        m.verify()?;
        let body = &m.body;
        let project = &body.project;

        if let Some(root) = &body.ledger_root {
            refuse_a_changed_root(
                &root.repository_node_id,
                self.ledger_root(&root.repository_node_id)?,
                &root.commit,
            )?;
        }

        let held_project = match self.locate(project.iri()) {
            Ok((_, Kind::Project)) => true,
            Ok((_, found)) => {
                return Err(StoreError::WrongKind {
                    id: project.iri().clone(),
                    expected: Kind::Project,
                    found,
                }
                .into());
            }
            Err(StoreError::NotOwned { .. }) => false,
            Err(e) => return Err(e.into()),
        };
        if held_project && self.imported_hash(project)?.is_none() {
            return Err(ManifestError::AuthoringStore(project.clone()));
        }
        // ⚠ An older checked-out manifest would un-route this machine alone
        // (routing spec §1.2: one routing rule per project, not per machine).
        if held_project && body.routing.is_none() && self.routes(project)?.is_some() {
            return Err(ManifestError::WouldDropRouting(project.clone()));
        }
        // Routing spec decision 20: a routed store holds one project — a
        // routed manifest goes only into a store holding no other, and no
        // other project goes into a routed store.
        if body.routing.is_some() {
            self.refuse_a_second_project(project)?;
        } else if !held_project && self.holds_routing()? {
            return Err(routed_store_is_taken().into());
        }
        // Another project on the same root would give one checkout two
        // sets of gates and two handles, with nothing said.
        for other in self.list_projects()? {
            if other.id != *project && other.root == root {
                return Err(ManifestError::RootTaken {
                    root: root.to_string(),
                    other: other.id,
                });
            }
        }
        let root_moved = match held_project {
            true => {
                let held = self
                    .get_project(project)?
                    .ok_or_else(|| ManifestError::Inconsistent(format!("{project} vanished")))?;
                (held.root != root).then(|| (held.root, root.to_string()))
            }
            false => None,
        };

        let held_gates = if held_project {
            self.list_gates(project)?
        } else {
            vec![]
        };
        for g in &held_gates {
            if !body.gates.iter().any(|m| m.id == g.id) {
                return Err(ManifestError::WouldRemoveGate {
                    id: g.id.clone(),
                    name: g.name.clone(),
                });
            }
        }
        // Transitions mirror the manifest: one it no longer lists is
        // removed. (Neither side has a command that removes one today.)
        let stale_transitions: Vec<String> = if held_project {
            self.list_transitions(project)?
                .into_iter()
                .filter(|t| !body.transitions.iter().any(|m| m.name == t.name))
                .map(|t| t.name)
                .collect()
        } else {
            vec![]
        };

        let mut report = ImportReport {
            project: project.clone(),
            gates_added: 0,
            gates_changed: 0,
            gates_unchanged: 0,
            transitions: body.transitions.len(),
            transitions_removed: stale_transitions.clone(),
            root_moved,
            areas: body.routing.as_ref().map(|m| m.areas.len()),
        };
        // (definition to write, whether it is new to this store)
        let mut writes: Vec<(&GateDef, bool)> = Vec::new();
        for g in &body.gates {
            match self.locate(g.id.iri()) {
                Err(StoreError::NotOwned { .. }) => {
                    writes.push((g, true));
                    report.gates_added += 1;
                }
                Ok((_, Kind::Gate)) => {
                    let Some(held) = held_gates.iter().find(|h| h.id == g.id) else {
                        return Err(ManifestError::Inconsistent(format!(
                            "gate {} is held by this store under another project",
                            g.id
                        )));
                    };
                    let mut bare = held.clone();
                    bare.last_pass_commit = None;
                    if bare == *g {
                        report.gates_unchanged += 1;
                    } else {
                        writes.push((g, false));
                        report.gates_changed += 1;
                    }
                }
                Ok((_, found)) => {
                    return Err(StoreError::WrongKind {
                        id: g.id.iri().clone(),
                        expected: Kind::Gate,
                        found,
                    }
                    .into());
                }
                Err(e) => return Err(e.into()),
            }
        }

        let tx = self.db.begin_write().map_err(backend)?;
        if !held_project || report.root_moved.is_some() {
            if !held_project {
                index_new(&tx, project.iri(), Kind::Project)?;
            }
            let json = serde_json::to_string(&Project {
                id: project.clone(),
                root: root.to_string(),
            })
            .map_err(backend)?;
            tx.open_table(PROJECTS)
                .map_err(backend)?
                .insert(project.iri().as_str(), json.as_str())
                .map_err(backend)?;
        }
        for (g, is_new) in &writes {
            if *is_new {
                index_new(&tx, g.id.iri(), Kind::Gate)?;
            }
            let json = serde_json::to_string(g).map_err(backend)?;
            tx.open_table(GATES)
                .map_err(backend)?
                .insert(g.id.iri().as_str(), json.as_str())
                .map_err(backend)?;
        }
        {
            let mut table = tx.open_table(TRANSITIONS).map_err(backend)?;
            for name in &stale_transitions {
                table
                    .remove((project.iri().as_str(), name.as_str()))
                    .map_err(backend)?;
            }
            for t in &body.transitions {
                let json = serde_json::to_string(t).map_err(backend)?;
                table
                    .insert((t.project.iri().as_str(), t.name.as_str()), json.as_str())
                    .map_err(backend)?;
            }
        }
        tx.open_table(IMPORTS)
            .map_err(backend)?
            .insert(project.iri().as_str(), m.content_sha256.as_str())
            .map_err(backend)?;
        if let Some(map) = &body.routing {
            let json = serde_json::to_string(map).map_err(backend)?;
            tx.open_table(ROUTING)
                .map_err(backend)?
                .insert(project.iri().as_str(), json.as_str())
                .map_err(backend)?;
            raise_format(&tx, FORMAT_WITH_ROUTING)?;
        }
        if let Some(root) = &body.ledger_root {
            record_ledger_root(&tx, &root.repository_node_id, &root.commit)?;
        }
        raise_format(&tx, FORMAT_WITH_IMPORTS)?;
        tx.commit().map_err(backend)?;
        Ok(report)
    }
}

impl Catalog for RedbStore {
    fn add_project(&self, root: &str) -> Result<ProjectId, StoreError> {
        // Routing spec decision 20: a routed store holds one project.
        if self.holds_routing()? {
            return Err(routed_store_is_taken());
        }
        let id = self.insert_new(Kind::Project, PROJECTS, None, |id| Project {
            id: ProjectId(id),
            root: root.to_string(),
        })?;
        Ok(ProjectId(id))
    }

    fn get_project(&self, id: &ProjectId) -> Result<Option<Project>, StoreError> {
        self.get_json(PROJECTS, id.iri())
    }

    fn list_projects(&self) -> Result<Vec<Project>, StoreError> {
        self.all_json(PROJECTS)
    }

    #[allow(clippy::too_many_arguments)]
    fn add_gate(
        &self,
        project: &ProjectId,
        name: &str,
        kind: GateKind,
        selector: Selector,
        min_population: u64,
        authored_at_commit: &str,
        authored_by: &str,
    ) -> Result<GateId, StoreError> {
        self.check_kind(project.iri(), Kind::Project)?;
        self.refuse_if_imported(project, "add a gate to")?;
        let id = self.insert_new(Kind::Gate, GATES, None, |id| GateDef {
            id: GateId(id),
            project: project.clone(),
            name: name.to_string(),
            kind,
            selector,
            min_population,
            authored_at_commit: authored_at_commit.to_string(),
            authored_by: authored_by.to_string(),
            last_pass_commit: None,
        })?;
        Ok(GateId(id))
    }

    fn get_gate(&self, id: &GateId) -> Result<Option<GateDef>, StoreError> {
        self.get_json(GATES, id.iri())
    }

    fn list_gates(&self, project: &ProjectId) -> Result<Vec<GateDef>, StoreError> {
        self.check_kind(project.iri(), Kind::Project)?;
        let all: Vec<GateDef> = self.all_json(GATES)?;
        Ok(all.into_iter().filter(|g| g.project == *project).collect())
    }

    fn update_gate(&self, def: &GateDef) -> Result<(), StoreError> {
        let Some(held) = self.get_gate(&def.id)? else {
            return Err(StoreError::NoSuchGate(def.id.clone()));
        };
        // An imported gate may earn a local pass mark and nothing else
        // (GitHub tracker spec §4.2).
        if self.imported_hash(&held.project)?.is_some() {
            let (mut before, mut after) = (held, def.clone());
            before.last_pass_commit = None;
            after.last_pass_commit = None;
            if before != after {
                return Err(StoreError::Imported {
                    id: def.id.iri().clone(),
                    action: "change the definition of",
                });
            }
        }
        self.put_json(GATES, def.id.iri(), def)
    }

    fn add_transition(&self, t: Transition) -> Result<(), StoreError> {
        self.check_kind(t.project.iri(), Kind::Project)?;
        self.refuse_if_imported(&t.project, "add a transition to")?;
        let json = serde_json::to_string(&t).map_err(backend)?;
        let tx = self.db.begin_write().map_err(backend)?;
        {
            let mut table = tx.open_table(TRANSITIONS).map_err(backend)?;
            table
                .insert((t.project.iri().as_str(), t.name.as_str()), json.as_str())
                .map_err(backend)?;
        }
        tx.commit().map_err(backend)?;
        Ok(())
    }

    fn get_transition(
        &self,
        project: &ProjectId,
        name: &str,
    ) -> Result<Option<Transition>, StoreError> {
        self.check_kind(project.iri(), Kind::Project)?;
        let tx = self.db.begin_read().map_err(backend)?;
        let table = tx.open_table(TRANSITIONS).map_err(backend)?;
        let Some(v) = table
            .get_owned((project.iri().as_str(), name))
            .map_err(backend)?
        else {
            return Ok(None);
        };
        Ok(Some(serde_json::from_str(v.value()).map_err(decode)?))
    }

    fn list_transitions(&self, project: &ProjectId) -> Result<Vec<Transition>, StoreError> {
        self.check_kind(project.iri(), Kind::Project)?;
        let tx = self.db.begin_read().map_err(backend)?;
        let table = tx.open_table(TRANSITIONS).map_err(backend)?;
        let mut out = Vec::new();
        for row in table.iter().map_err(backend)? {
            let (_, v) = row.map_err(backend)?;
            let t: Transition = serde_json::from_str(v.value()).map_err(decode)?;
            // Filtered on the stored field rather than on the key's project
            // half: the key is a storage convention, the field is the fact,
            // and only one of the two is checked by the type system.
            if t.project == *project {
                out.push(t);
            }
        }
        Ok(out)
    }

    fn kind_of(&self, id: &Iri) -> Result<Kind, StoreError> {
        self.check(id)
    }
}

impl Routes for RedbStore {
    fn routes(&self, project: &ProjectId) -> Result<Option<RoutingMap>, StoreError> {
        self.check_kind(project.iri(), Kind::Project)?;
        let tx = self.db.begin_read().map_err(backend)?;
        let table = match tx.open_table(ROUTING) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(e) => return Err(backend(e)),
        };
        let Some(v) = table.get(project.iri().as_str()).map_err(backend)? else {
            return Ok(None);
        };
        let map: RoutingMap = serde_json::from_str(v.value()).map_err(decode)?;
        // ⚠ A stored map routes nothing until it is one fl writes.
        map.check().map_err(decode)?;
        Ok(Some(map))
    }
}

impl Tracker for RedbStore {
    fn add_record_with_area(
        &self,
        project: &ProjectId,
        title: &str,
        area: Option<&str>,
    ) -> Result<RecordId, StoreError> {
        self.check_kind(project.iri(), Kind::Project)?;
        let raise = area.map(|_| FORMAT_WITH_ROUTING);
        let id = self.insert_new(Kind::Record, RECORDS, raise, |id| Record {
            id: RecordId(id),
            project: project.clone(),
            title: title.to_string(),
            state: State::Todo,
            also_known_as: vec![],
            area: area.map(str::to_string),
        })?;
        Ok(RecordId(id))
    }

    fn get_record(&self, id: &RecordId) -> Result<Option<Record>, StoreError> {
        self.get_json(RECORDS, id.iri())
    }

    fn list_records(&self, project: &ProjectId) -> Result<Vec<Record>, StoreError> {
        self.check_kind(project.iri(), Kind::Project)?;
        let all: Vec<Record> = self.all_json(RECORDS)?;
        let gone = self.tombstoned()?;
        Ok(all
            .into_iter()
            .filter(|r| r.project == *project && !gone.contains(r.id.iri().as_str()))
            .collect())
    }

    fn set_record_state(&self, id: &RecordId, state: State) -> Result<(), StoreError> {
        // `rewrite` writes under the primary, not `id`: `id` may be an
        // alias, and writing under an alias key would leave a stray row
        // behind instead of updating the one that exists.
        let changed = self.rewrite(RECORDS, id.iri(), |mut rec: Record| {
            rec.state = state;
            rec
        })?;
        if !changed {
            return Err(StoreError::NoSuchRecord(id.clone()));
        }
        Ok(())
    }

    fn add_finding(&self, finding: Finding) -> Result<FindingId, StoreError> {
        self.check_kind(finding.project.iri(), Kind::Project)?;
        let raise = finding.area.as_ref().map(|_| FORMAT_WITH_ROUTING);
        let id = self.insert_new_in(Kind::Finding, FINDINGS, raise, |tx, id| {
            // `record` may be given as an alias (e.g. the CLI stores
            // whatever the caller typed): resolve to the primary, so two
            // findings raised against the same record always agree on which
            // IRI names it.
            let (record_primary, record_kind) = self.locate_for_write(tx, finding.record.iri())?;
            if record_kind != Kind::Record {
                return Err(StoreError::WrongKind {
                    id: finding.record.iri().clone(),
                    expected: Kind::Record,
                    found: record_kind,
                });
            }
            // A marked record takes findings (routing spec §3.3 step 1); an
            // escalated one is `Escalated`, which the router follows.
            refuse_escalated(tx, &record_primary)?;
            let mut finding = finding;
            finding.id = FindingId(id);
            finding.record = RecordId(record_primary);
            Ok(finding)
        })?;
        Ok(FindingId(id))
    }

    fn add_finding_checked(
        &self,
        finding: Finding,
        record: ForeignRecord,
    ) -> Result<FindingId, StoreError> {
        self.check_kind(finding.project.iri(), Kind::Project)?;
        check_foreign_for_local(&record, self.owns(record.id().iri())?)?;
        let raise = finding.area.as_ref().map(|_| FORMAT_WITH_ROUTING);
        let id = self.insert_new(Kind::Finding, FINDINGS, raise, |id| {
            let mut finding = finding;
            finding.id = FindingId(id);
            finding.record = record.id().clone();
            finding
        })?;
        Ok(FindingId(id))
    }

    fn get_finding(&self, id: &FindingId) -> Result<Option<Finding>, StoreError> {
        self.get_json(FINDINGS, id.iri())
    }

    fn update_finding(&self, finding: &Finding) -> Result<(), StoreError> {
        let changed = self.rewrite(FINDINGS, finding.id.iri(), |kept: Finding| {
            // `kept.id` is always the primary: the row is keyed by it. Take
            // every other field from the caller's version, but keep the id
            // pinned to the primary — even if `finding.id` (what the caller
            // passed) is an alias — so an update through an alias still
            // lands on, and stays keyed by, the primary, rather than writing
            // a second row under the alias.
            //
            // The stored `also_known_as` is kept and the caller's ignored
            // (see the trait): only `add_alias` adds a name.
            let mut stored = finding.clone();
            stored.id = kept.id;
            stored.also_known_as = kept.also_known_as;
            // Fixed when the finding is raised (routing spec §1.1; GitHub
            // tracker spec §6): the caller's copy never changes them.
            stored.record = kept.record;
            stored.raised_by = kept.raised_by;
            stored.security = kept.security;
            stored.area = kept.area;
            stored
        })?;
        if !changed {
            return Err(StoreError::NoSuchFinding(finding.id.clone()));
        }
        Ok(())
    }

    fn list_findings(&self, project: &ProjectId) -> Result<Vec<Finding>, StoreError> {
        self.check_kind(project.iri(), Kind::Project)?;
        let all: Vec<Finding> = self.all_json(FINDINGS)?;
        let gone = self.tombstoned()?;
        Ok(all
            .into_iter()
            .filter(|f| f.project == *project && !gone.contains(f.id.iri().as_str()))
            .collect())
    }

    /// An escalated finding is counted by the tier it lives in now.
    fn withdrawals_by(&self, actor: &str) -> Result<u64, StoreError> {
        let all: Vec<Finding> = self.all_json(FINDINGS)?;
        let gone = self.tombstoned()?;
        Ok(all
            .into_iter()
            .filter(|f| f.raised_by == actor && f.state == FindingState::Withdrawn)
            .filter(|f| !gone.contains(f.id.iri().as_str()))
            .count() as u64)
    }

    /// Mint an alias, index it, and append it to the item's `also_known_as`
    /// row — in ONE write transaction, for the same reason as
    /// `insert_new_with_id`: a failure partway must leave neither the index
    /// entry nor the row change behind.
    fn add_alias(&self, primary: &Iri, alias: Iri) -> Result<(), StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;

        let (resolved, kind) = {
            let ids = tx.open_table(IDS).map_err(backend)?;
            // `alias` must be unused, whether as a primary id or as another
            // alias — both live in this one table.
            if ids.get(alias.as_str()).map_err(backend)?.is_some() {
                return Err(StoreError::AlreadyExists(alias));
            }
            let Some(pv) = ids.get(primary.as_str()).map_err(backend)? else {
                return Err(StoreError::NotOwned {
                    id: primary.clone(),
                    searched: vec![self.label.clone()],
                });
            };
            if pv.value() == "alias" {
                // `primary` is itself an alias: resolve one more hop so
                // `ALIASES` never chains. Shares `locate`'s own alias-hop
                // logic rather than re-implementing it.
                let aliases = tx.open_table(ALIASES).map_err(backend)?;
                let (resolved, wire) = alias_primary(primary, &ids, &aliases)?;
                let kind = Kind::from_wire(&wire)
                    .ok_or_else(|| decode(format!("unknown kind `{wire}` for {resolved}")))?;
                (resolved, kind)
            } else {
                let wire = pv.value().to_string();
                let kind = Kind::from_wire(&wire)
                    .ok_or_else(|| decode(format!("unknown kind `{wire}` for {primary}")))?;
                (primary.clone(), kind)
            }
        };

        // An alias is a write to the item it names (routing spec §3.3 step
        // 1, §3.6), checked in this transaction.
        refuse_unwritable(&tx, &resolved, kind)?;

        let table = match kind {
            Kind::Record => RECORDS,
            Kind::Finding => FINDINGS,
            other => {
                return Err(StoreError::Backend(format!(
                    "{resolved} is a {}, and only a record or finding can carry an alias",
                    other.as_wire()
                )));
            }
        };
        {
            let mut t = tx.open_table(table).map_err(backend)?;
            // `IDS` says this id is a record or finding, but that does not
            // put its row here: a damaged store can hold the index entry
            // without the row. That is `Decode`, like `alias_primary`'s
            // damaged cases — never a panic.
            let json = match t.get(resolved.as_str()).map_err(backend)? {
                Some(v) => v.value().to_string(),
                None => {
                    return Err(decode(format!(
                        "{resolved} is indexed as a {}, but this store holds no row for it",
                        kind.as_wire()
                    )));
                }
            };
            let updated = match kind {
                Kind::Record => {
                    let mut r: Record = serde_json::from_str(&json).map_err(decode)?;
                    r.also_known_as.push(alias.clone());
                    serde_json::to_string(&r).map_err(backend)?
                }
                Kind::Finding => {
                    let mut f: Finding = serde_json::from_str(&json).map_err(decode)?;
                    f.also_known_as.push(alias.clone());
                    serde_json::to_string(&f).map_err(backend)?
                }
                _ => unreachable!("checked above"),
            };
            t.insert(resolved.as_str(), updated.as_str())
                .map_err(backend)?;
        }
        {
            let mut ids = tx.open_table(IDS).map_err(backend)?;
            ids.insert(alias.as_str(), "alias").map_err(backend)?;
        }
        {
            let mut aliases = tx.open_table(ALIASES).map_err(backend)?;
            aliases
                .insert(alias.as_str(), resolved.as_str())
                .map_err(backend)?;
        }

        tx.commit().map_err(backend)?;
        Ok(())
    }
}

/// Each method is one write transaction, or one read: a check made in it
/// holds when its write lands (routing spec §2.7: the store is opened by
/// one process at a time).
impl Escalations for RedbStore {
    fn mark(&self, id: &Iri, mark: &Mark) -> Result<(), StoreError> {
        let json = serde_json::to_string(mark).map_err(backend)?;
        let tx = self.db.begin_write().map_err(backend)?;
        let (primary, kind) = self.locate_for_write(&tx, id)?;
        refuse_escalated(&tx, &primary)?;
        if !matches!(kind, Kind::Record | Kind::Finding) {
            return Err(StoreError::WrongKind {
                id: id.clone(),
                expected: Kind::Record,
                found: kind,
            });
        }
        if additive_in_write::<Mark>(&tx, ESCALATING, &primary)?.is_some() {
            return Err(EscalationFault::AlreadyMarked { id: primary, kind }.into());
        }
        tx.open_table(ESCALATING)
            .map_err(backend)?
            .insert(primary.as_str(), json.as_str())
            .map_err(backend)?;
        raise_format(&tx, FORMAT_WITH_ROUTING)?;
        tx.commit().map_err(backend)
    }

    fn mark_of(&self, id: &Iri) -> Result<Option<Mark>, StoreError> {
        let Some(primary) = self.held(id)? else {
            return Ok(None);
        };
        let tx = self.db.begin_read().map_err(backend)?;
        additive(&tx, ESCALATING, &primary)
    }

    fn unmark(&self, id: &Iri) -> Result<(), StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        let (primary, _) = self.locate_for_write(&tx, id)?;
        let removed = tx
            .open_table(ESCALATING)
            .map_err(backend)?
            .remove(primary.as_str())
            .map_err(backend)?
            .is_some();
        if !removed {
            return Err(EscalationFault::NotMarked { id: primary }.into());
        }
        tx.commit().map_err(backend)
    }

    fn tombstone(&self, id: &Iri, to: &Iri) -> Result<Tombstone, StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        let (primary, _) = self.locate_for_write(&tx, id)?;
        // The mark is removed and the tombstone written in this one
        // transaction: the item is never both, and never neither.
        let mark: Mark = {
            let mut t = tx.open_table(ESCALATING).map_err(backend)?;
            let Some(v) = t.remove(primary.as_str()).map_err(backend)? else {
                return Err(EscalationFault::NotMarked { id: primary }.into());
            };
            serde_json::from_str(v.value()).map_err(decode)?
        };
        let tomb = Tombstone {
            from: primary.clone(),
            to: to.clone(),
            by: mark.by,
            reason: mark.reason,
            at_ms: mark.at_ms,
        };
        let json = serde_json::to_string(&tomb).map_err(backend)?;
        tx.open_table(TOMBSTONES)
            .map_err(backend)?
            .insert(primary.as_str(), json.as_str())
            .map_err(backend)?;
        tx.commit().map_err(backend)?;
        Ok(tomb)
    }

    fn tombstone_of(&self, id: &Iri) -> Result<Option<Tombstone>, StoreError> {
        let Some(primary) = self.held(id)? else {
            return Ok(None);
        };
        let tx = self.db.begin_read().map_err(backend)?;
        additive(&tx, TOMBSTONES, &primary)
    }
}

impl Ledger for RedbStore {
    // `append_*` checks nothing (spec §3.4): the engine already resolved the
    // references it is recording.
    fn append_gate_run(&self, run: GateRun) -> Result<(), StoreError> {
        let candidate = match (&run.id, &run.record) {
            (Some(id), Some(_)) => Some((CANDIDATE_RUNS, id)),
            _ => None,
        };
        self.append_json(NEXT_RUN, GATE_RUNS, &run, candidate)
    }

    fn append_attempt(&self, attempt: Attempt) -> Result<(), StoreError> {
        let candidate = attempt.id.as_ref().map(|id| (CANDIDATE_ATTEMPTS, id));
        self.append_json(NEXT_ATTEMPT, ATTEMPTS, &attempt, candidate)
    }

    fn gate_runs(&self, gate: &GateId) -> Result<Vec<GateRun>, StoreError> {
        self.check(gate.iri())?;
        let all: Vec<GateRun> = self.all_log(GATE_RUNS)?;
        Ok(all.into_iter().filter(|r| r.gate == *gate).collect())
    }

    fn attempts(&self, project: &ProjectId) -> Result<Vec<Attempt>, StoreError> {
        self.check_kind(project.iri(), Kind::Project)?;
        let all: Vec<Attempt> = self.all_log(ATTEMPTS)?;
        Ok(all.into_iter().filter(|a| a.project == *project).collect())
    }
}

impl Outbox for RedbStore {
    fn unpublished(&self, repo: &str, after: &Iri) -> Result<Pending, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        Ok(Pending {
            runs: waiting(&tx, CANDIDATE_RUNS, GATE_RUNS, repo, after)?,
            attempts: waiting(&tx, CANDIDATE_ATTEMPTS, ATTEMPTS, repo, after)?,
        })
    }

    fn is_published(&self, repo: &str, id: &Iri) -> Result<bool, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let table = match tx.open_table(LEDGER_PUBLISHED) {
            Ok(t) => t,
            // The first mark creates the table: "nothing marked".
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(false),
            Err(e) => return Err(backend(e)),
        };
        let found = table.get((repo, id.as_str())).map_err(backend)?.is_some();
        Ok(found)
    }

    fn mark_published(&self, repo: &str, ids: &[Iri]) -> Result<(), StoreError> {
        self.settle(repo, ids, &[])
    }

    fn settle(&self, repo: &str, published: &[Iri], set_aside: &[Iri]) -> Result<(), StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        {
            let mut table = tx.open_table(LEDGER_PUBLISHED).map_err(backend)?;
            for id in published {
                table.insert((repo, id.as_str()), true).map_err(backend)?;
            }
        }
        // ⚠ The same write: a published or set-aside entry waits no more.
        drop_candidates(&tx, published)?;
        drop_candidates(&tx, set_aside)?;
        tx.commit().map_err(backend)
    }

    fn cutover(&self, repo: &str) -> Result<Option<Iri>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let table = match tx.open_table(LEDGER_CUTOVERS) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(e) => return Err(backend(e)),
        };
        let Some(v) = table.get(repo).map_err(backend)? else {
            return Ok(None);
        };
        Iri::parse(v.value()).map(Some).map_err(decode)
    }

    fn set_cutover(&self, repo: &str, id: &Iri) -> Result<(), StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        {
            let mut table = tx.open_table(LEDGER_CUTOVERS).map_err(backend)?;
            let held = table
                .get(repo)
                .map_err(backend)?
                .map(|v| v.value().to_string());
            match held {
                // Dropping `tx` uncommitted aborts it: nothing was written.
                Some(h) if h == id.as_str() => return Ok(()),
                Some(h) => {
                    return Err(StoreError::CutoverChanged {
                        node_id: repo.to_string(),
                        held: Iri::parse(&h).map_err(decode)?,
                        found: id.clone(),
                    });
                }
                None => {
                    table.insert(repo, id.as_str()).map_err(backend)?;
                }
            }
        }
        tx.commit().map_err(backend)
    }
}

impl Handles for RedbStore {
    /// `None` for an id this store does not hold under `kind`: handles are
    /// per kind, so a gate's IRI has no project handle.
    fn handle_of(&self, kind: Kind, id: &Iri) -> Result<Option<u64>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let ids = tx.open_table(IDS).map_err(backend)?;
        match ids.get(id.as_str()).map_err(backend)? {
            Some(k) if k.value() == kind.as_wire() => {}
            _ => return Ok(None),
        }
        let t = tx.open_table(HANDLE_OF).map_err(backend)?;
        Ok(t.get(id.as_str()).map_err(backend)?.map(|v| v.value()))
    }

    fn resolve_handle(&self, kind: Kind, handle: u64) -> Result<Option<Iri>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let t = tx.open_table(HANDLES).map_err(backend)?;
        let Some(v) = t.get((kind.as_wire(), handle)).map_err(backend)? else {
            return Ok(None);
        };
        Iri::parse(v.value()).map(Some).map_err(decode)
    }
}

impl Bindings for RedbStore {
    fn bound_node_id(&self, repo: &str) -> Result<Option<String>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let table = match tx.open_table(GITHUB_BINDINGS) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(e) => return Err(backend(e)),
        };
        let found = table
            .get(repo.to_ascii_lowercase().as_str())
            .map_err(backend)?
            .map(|v| v.value().to_string());
        Ok(found)
    }
    fn bind_node_id(&self, repo: &str, node_id: &str) -> Result<(), StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        tx.open_table(GITHUB_BINDINGS)
            .map_err(backend)?
            .insert(repo.to_ascii_lowercase().as_str(), node_id)
            .map_err(backend)?;
        tx.commit().map_err(backend)
    }
    fn ledger_root(&self, node_id: &str) -> Result<Option<String>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let table = match tx.open_table(LEDGER_ROOTS) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(e) => return Err(backend(e)),
        };
        let found = table
            .get(node_id)
            .map_err(backend)?
            .map(|v| v.value().to_string());
        Ok(found)
    }
    fn set_ledger_root(&self, node_id: &str, commit: &str) -> Result<(), StoreError> {
        refuse_a_changed_root(node_id, self.ledger_root(node_id)?, commit)?;
        let tx = self.db.begin_write().map_err(backend)?;
        record_ledger_root(&tx, node_id, commit)?;
        tx.commit().map_err(backend)
    }
}

impl LedgerCache for RedbStore {
    fn last_head(&self, repo: &str) -> Result<Option<String>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let table = match tx.open_table(LEDGER_HEADS) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(e) => return Err(backend(e)),
        };
        let found = table
            .get(repo)
            .map_err(backend)?
            .map(|v| v.value().to_string());
        Ok(found)
    }

    fn cached(&self, repo: &str, path: &str) -> Result<Option<CachedSegment>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let table = match tx.open_table(LEDGER_SEGMENTS) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(e) => return Err(backend(e)),
        };
        let found = match table.get((repo, path)).map_err(backend)? {
            Some(v) => Some(serde_json::from_str(v.value()).map_err(decode)?),
            None => None,
        };
        Ok(found)
    }

    fn cached_under(
        &self,
        repo: &str,
        dir: &str,
    ) -> Result<Vec<(String, CachedSegment)>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let table = match tx.open_table(LEDGER_SEGMENTS) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(vec![]),
            Err(e) => return Err(backend(e)),
        };
        let prefix = format!("{dir}/");
        let mut out = Vec::new();
        for entry in table.range((repo, prefix.as_str())..).map_err(backend)? {
            let (k, v) = entry.map_err(backend)?;
            let (r, p) = k.value();
            if r != repo || !p.starts_with(&prefix) {
                break;
            }
            out.push((
                p.to_string(),
                serde_json::from_str(v.value()).map_err(decode)?,
            ));
        }
        Ok(out)
    }

    /// One write transaction: the head and every segment land together, or
    /// (on any error before `commit()`) none of them do — `redb`'s
    /// `WriteTransaction` aborts on drop when it was never completed.
    fn remember(
        &self,
        repo: &str,
        head: &str,
        segments: &[(String, CachedSegment)],
    ) -> Result<(), StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        tx.open_table(LEDGER_HEADS)
            .map_err(backend)?
            .insert(repo, head)
            .map_err(backend)?;
        {
            let mut table = tx.open_table(LEDGER_SEGMENTS).map_err(backend)?;
            for (path, segment) in segments {
                let json = serde_json::to_string(segment).map_err(backend)?;
                table
                    .insert((repo, path.as_str()), json.as_str())
                    .map_err(backend)?;
            }
        }
        tx.commit().map_err(backend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fl_core::conformance::{entry_iri, sample_attempt, sample_record_run};
    use fl_core::ids::seq_iri;
    use fl_core::log::GateRun;
    use fl_core::model::State;
    use fl_core::verdict::Verdict;
    use redb::ReadableTableMetadata;

    /// A store written before format versioning: tables, and no version key.
    /// The table definitions are local and frozen: this fixture must keep
    /// writing the OLD shape after Task 4 changes the live ones.
    fn legacy_store(path: &std::path::Path) {
        const OLD_META: TableDefinition<&str, u64> = TableDefinition::new("meta");
        const OLD_PROJECTS: TableDefinition<u64, &str> = TableDefinition::new("projects");
        let db = redb::Database::create(path).unwrap();
        let tx = db.begin_write().unwrap();
        {
            tx.open_table(OLD_META)
                .unwrap()
                .insert("next_id", 3u64)
                .unwrap();
            tx.open_table(OLD_PROJECTS)
                .unwrap()
                .insert(1u64, "{}")
                .unwrap();
        }
        tx.commit().unwrap();
    }

    #[test]
    fn a_cursor_at_u64_max_reads_nothing_instead_of_overflowing() {
        let (s, _dir) = fresh();
        fl_core::conformance::a_cursor_at_u64_max_reads_nothing(&s);
    }

    #[test]
    fn a_project_survives_a_close_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.redb");

        let id = {
            let s = RedbStore::open(&path).unwrap();
            s.add_project("/tmp/p").unwrap()
        };

        let s = RedbStore::open(&path).unwrap();
        assert_eq!(s.get_project(&id).unwrap().unwrap().root, "/tmp/p");
    }

    #[test]
    fn two_independent_stores_mint_different_ids() {
        let (a, _da) = fresh();
        let (b, _db) = fresh();
        let pa = a.add_project("/p").unwrap();
        let pb = b.add_project("/p").unwrap();
        let ga = a
            .add_gate(&pa, "g", kind(), selector(), 1, "abc", "o")
            .unwrap();
        let gb = b
            .add_gate(&pb, "g", kind(), selector(), 1, "abc", "o")
            .unwrap();
        assert_ne!(
            ga, gb,
            "two installs must not both have `gate 1` as their id"
        );
        assert!(ga.iri().as_str().starts_with("urn:uuid:"));
    }

    #[test]
    fn a_handle_is_never_reused_after_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.redb");
        {
            let s = RedbStore::open(&path).unwrap();
            s.add_project("/a").unwrap();
        }
        let s = RedbStore::open(&path).unwrap();
        let b = s.add_project("/b").unwrap();
        assert_eq!(s.handle_of(Kind::Project, b.iri()).unwrap(), Some(2));
    }

    #[test]
    fn a_minted_id_that_already_exists_is_refused_and_nothing_is_overwritten() {
        let (s, _d) = fresh();
        let id = Iri::parse("urn:uuid:0190a1b2-c3d4-7e5f-8a6b-7c8d9e0f1a2b").unwrap();
        s.insert_new_with_id(id.clone(), Kind::Project, PROJECTS, None, |i| Project {
            id: ProjectId(i),
            root: "/first".into(),
        })
        .unwrap();
        let err = s
            .insert_new_with_id(id.clone(), Kind::Project, PROJECTS, None, |i| Project {
                id: ProjectId(i),
                root: "/second".into(),
            })
            .unwrap_err();
        assert!(
            matches!(err, StoreError::AlreadyExists(ref i) if *i == id),
            "{err:?}"
        );
        assert_eq!(
            s.get_project(&ProjectId(id)).unwrap().unwrap().root,
            "/first"
        );
    }

    #[test]
    fn a_format_1_store_is_refused_by_format_2() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v1.redb");
        {
            let db = redb::Database::create(&path).unwrap();
            let tx = db.begin_write().unwrap();
            tx.open_table(META)
                .unwrap()
                .insert("format_version", 1u64)
                .unwrap();
            tx.commit().unwrap();
        }
        let err = RedbStore::open(&path).err().unwrap();
        assert!(
            matches!(
                err,
                StoreError::FormatVersion {
                    found: Some(1),
                    oldest: 2,
                    newest: FORMAT_WITH_WORKFLOW
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn a_record_state_change_is_durable() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.redb");
        let (p, r) = {
            let s = RedbStore::open(&path).unwrap();
            let p = s.add_project("/p").unwrap();
            let r = s.add_record(&p, "t").unwrap();
            s.set_record_state(&r, State::Review).unwrap();
            (p, r)
        };
        let s = RedbStore::open(&path).unwrap();
        assert_eq!(s.get_record(&r).unwrap().unwrap().state, State::Review);
        assert_eq!(s.get_project(&p).unwrap().unwrap().id, p);
    }

    /// Beyond the brief: `Verdict`'s `Deserialize` is hand-written and rejects
    /// a zero-population `Pass`. A `GateRun` carries a `Verdict` through this
    /// store as JSON, so round-tripping it is the one place this task can
    /// silently corrupt a passing gate into something else (or fail to
    /// deserialize at all). Pin that the verdict and its population survive
    /// a close and reopen unchanged.
    #[test]
    fn a_gate_run_verdict_survives_a_close_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.redb");

        let verdict = Verdict::from_predicate(true, 3);
        let g = {
            let s = RedbStore::open(&path).unwrap();
            let p = s.add_project("/p").unwrap();
            let g = s
                .add_gate(
                    &p,
                    "fmt",
                    fl_core::model::GateKind::Command(fl_core::model::CommandSpec {
                        program: "true".into(),
                        args: vec![],
                        delivery: fl_core::model::PopulationDelivery::Args,
                        timeout_secs: 5,
                        pass_codes: vec![0],
                    }),
                    fl_core::model::Selector::Glob {
                        pattern: "**/*.rs".into(),
                    },
                    1,
                    "abc",
                    "owner",
                )
                .unwrap();
            s.append_gate_run(GateRun {
                id: None,
                at: None,
                gate: g.clone(),
                record: None,
                commit: "abc".into(),
                verdict: verdict.clone(),
                population: 3,
                output_excerpt: Some(String::new()),
                duration_ms: 1,
                cost_usd_micros: 0,
            })
            .unwrap();
            g
        };

        let s = RedbStore::open(&path).unwrap();
        let runs = s.gate_runs(&g).unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].verdict, verdict);
        assert_eq!(runs[0].verdict.population(), Some(3));
    }

    /// Pins that minting, indexing,
    /// the handle bump and the row write inside `insert_new_with_id` commit
    /// as one unit. If they didn't, a failed insert would still leave the
    /// handle counter advanced and the id in `IDS` — a burned handle, and an
    /// id this store "owns" with no row behind it.
    ///
    /// There is no seam in the public role API to make the *last* step of a
    /// normal insert fail deterministically and cheaply: every concrete type
    /// this store serializes always serializes via `serde_json` without
    /// error, and redb's own size ceiling (`MAX_VALUE_LENGTH`, 3 GiB) is too
    /// large to hit in a fast test. So this test reaches for the same
    /// private field `RedbStore::open` itself would build (`db`, visible to
    /// this module) and pre-corrupts the on-disk `gates` table with a
    /// mismatched value type *before* constructing the store — bypassing
    /// `RedbStore::open`, which eagerly creates every table itself. This
    /// makes `add_gate`'s `IDS` insert, its `next_handle:gate` bump and its
    /// handle rows succeed, and its final `open_table(GATES)` fail with a
    /// genuine `redb::TableTypeMismatch`, inside the same still-uncommitted
    /// write transaction — the same commit boundary a real row-write failure
    /// would cross.
    #[test]
    fn a_failed_insert_does_not_advance_the_shared_id_counter() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.redb");

        let db = redb::Database::create(&path).unwrap();
        {
            // Same table name as `GATES`, wrong key and value types. Opening
            // it later under the real `GATES` definition will fail with
            // `TableTypeMismatch`, not silently coerce.
            const WRONG_GATES: TableDefinition<u64, u64> = TableDefinition::new("gates");
            let tx = db.begin_write().unwrap();
            {
                tx.open_table(WRONG_GATES).unwrap();
            }
            tx.commit().unwrap();
        }

        // Bypasses `RedbStore::open` deliberately: it would try to create
        // the real `GATES` definition itself and fail right there, before we
        // ever get to call `add_gate`.
        let store = RedbStore {
            db: Db::Write(db),
            label: path.display().to_string(),
        };

        // The gate needs a project this store owns, or `add_gate` would be
        // refused by the ownership check before it reached the write.
        let p = store.add_project("/p").unwrap();

        let failed = store.add_gate(
            &p,
            "fmt",
            GateKind::Command(fl_core::model::CommandSpec {
                program: "true".into(),
                args: vec![],
                delivery: fl_core::model::PopulationDelivery::Args,
                timeout_secs: 5,
                pass_codes: vec![0],
            }),
            Selector::Glob {
                pattern: "**/*.rs".into(),
            },
            1,
            "abc",
            "owner",
        );
        assert!(
            failed.is_err(),
            "expected the mismatched `gates` table to reject the write"
        );

        let tx = store.db.begin_read().unwrap();
        let meta = tx.open_table(META).unwrap();
        assert_eq!(
            meta.get("next_handle:gate").unwrap().map(|v| v.value()),
            None,
            "a failed insert must not burn a handle"
        );
        let ids = tx.open_table(IDS).unwrap();
        assert_eq!(
            ids.len().unwrap(),
            1,
            "a failed insert must leave no `IDS` entry: only the project is owned"
        );
        assert!(ids.get(p.iri().as_str()).unwrap().is_some());
    }

    #[test]
    fn a_finding_survives_a_close_and_reopen_with_a_real_id() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.redb");

        let id = {
            let s = RedbStore::open(&path).unwrap();
            let p = s.add_project("/p").unwrap();
            let r = s.add_record(&p, "t").unwrap();
            s.add_finding(Finding::raise(p, r, "reviewer", "wrong on empty"))
                .unwrap()
        };
        assert_ne!(
            id,
            FindingId(seq_iri(0)),
            "the store must replace the placeholder id"
        );

        let s = RedbStore::open(&path).unwrap();
        let back = s.get_finding(&id).unwrap().unwrap();
        assert_eq!(back.id, id);
        assert_eq!(back.state, FindingState::Raised);
    }

    #[test]
    fn withdrawals_are_counted_against_whoever_raised_the_finding_and_survive_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.redb");

        {
            let s = RedbStore::open(&path).unwrap();
            let p = s.add_project("/p").unwrap();
            let r = s.add_record(&p, "t").unwrap();

            for claim in ["a", "b"] {
                let id = s
                    .add_finding(Finding::raise(p.clone(), r.clone(), "hasty", claim))
                    .unwrap();
                let mut f = s.get_finding(&id).unwrap().unwrap();
                f.withdraw("not concrete").unwrap();
                s.update_finding(&f).unwrap();
            }
            let id = s
                .add_finding(Finding::raise(p.clone(), r.clone(), "careful", "c"))
                .unwrap();
            let mut f = s.get_finding(&id).unwrap().unwrap();
            f.attach_reproduction(GateId(seq_iri(1))).unwrap();
            s.update_finding(&f).unwrap();
        }

        let s = RedbStore::open(&path).unwrap();
        assert_eq!(s.withdrawals_by("hasty").unwrap(), 2);
        assert_eq!(s.withdrawals_by("careful").unwrap(), 0);
        assert_eq!(s.withdrawals_by("nobody").unwrap(), 0);
    }

    #[test]
    fn a_record_written_in_an_older_wire_format_is_refused_with_a_remedy() {
        // What a wire-format change looks like from the other side. The
        // refusal must name the cause AND what to do, not just repeat what
        // serde said. Written as raw JSON because the point is a value this
        // build can no longer produce.
        let d = tempfile::tempdir().unwrap();
        let path = d.path().join("old.redb");
        let s = RedbStore::open(&path).expect("open");
        // A record this store owns, so the read reaches the decode step; its
        // row is then overwritten with what an older build wrote (a
        // PascalCase state).
        let p = s.add_project("/p").expect("project");
        let r = s.add_record(&p, "t").expect("record");
        {
            let tx = s.db.begin_write().expect("write tx");
            {
                let mut t = tx.open_table(RECORDS).expect("table");
                let old = format!(
                    r#"{{"id":"{}","project":"{}","title":"t","state":"Todo"}}"#,
                    r.iri(),
                    p.iri()
                );
                t.insert(r.iri().as_str(), old.as_str()).expect("insert");
            }
            tx.commit().expect("commit");
        }
        let err = s.get_record(&r).expect_err("must refuse");
        let msg = err.to_string();
        assert!(msg.contains("unknown variant"), "got {msg}");
        assert!(msg.contains("there is no migration"), "got {msg}");
        assert!(msg.contains("restore the file from a backup"), "got {msg}");
        assert!(
            matches!(err, StoreError::Decode(_)),
            "a decode failure must be Decode, not merely not-Backend: {err:?}"
        );
    }

    // `IDS` naming an id as a record whose row is
    // missing is on-disk damage, and `add_alias` must report it as `Decode`
    // — as `alias_primary` reports its own damaged cases — never panic.
    #[test]
    fn add_alias_on_an_indexed_record_whose_row_is_missing_is_decode_not_a_panic() {
        let (s, _d) = fresh();
        let p = s.add_project("/p").unwrap();
        let r = s.add_record(&p, "t").unwrap();
        {
            let tx = s.db.begin_write().unwrap();
            tx.open_table(RECORDS)
                .unwrap()
                .remove(r.iri().as_str())
                .unwrap();
            tx.commit().unwrap();
        }
        let alias = Iri::parse("https://github.com/o/r/issues/15").unwrap();
        let err = s.add_alias(r.iri(), alias.clone()).unwrap_err();
        assert!(matches!(err, StoreError::Decode(_)), "{err:?}");
        assert!(err.to_string().contains(r.iri().as_str()), "{err}");
        assert!(
            !s.owns(&alias).unwrap(),
            "a refused add_alias must not index the alias"
        );
    }

    fn fresh() -> (RedbStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let s = RedbStore::open(&dir.path().join("t.redb")).unwrap();
        (s, dir)
    }

    /// A store holding one run and one attempt in the shape every store held
    /// before entry ids: raw rows, as the current fl writes them. The row
    /// counters move with them, so a later append cannot overwrite them.
    fn with_legacy_entries() -> (RedbStore, tempfile::TempDir, ProjectId, GateId) {
        let (s, d) = fresh();
        let p = s.add_project("/p").unwrap();
        let g = s
            .add_gate(&p, "fmt", kind(), selector(), 1, "abc", "o")
            .unwrap();
        let r = s.add_record(&p, "t").unwrap();
        let run = format!(
            r#"{{"gate":"{}","record":"{}","commit":"abc","verdict":{{"pass":{{"population":1}}}},"population":1,"output_excerpt":"ok","duration_ms":1,"cost_usd_micros":0}}"#,
            g.iri(),
            r.iri()
        );
        let attempt = format!(
            r#"{{"project":"{}","record":"{}","adapter":"claude","status":"completed","duration_ms":1,"tokens_in":0,"tokens_out":0,"cost_usd_micros":0,"paths_touched":["a.rs"],"output_excerpt":""}}"#,
            p.iri(),
            r.iri()
        );
        {
            let tx = s.db.begin_write().unwrap();
            tx.open_table(GATE_RUNS)
                .unwrap()
                .insert(1u64, run.as_str())
                .unwrap();
            tx.open_table(ATTEMPTS)
                .unwrap()
                .insert(1u64, attempt.as_str())
                .unwrap();
            {
                let mut meta = tx.open_table(META).unwrap();
                meta.insert(NEXT_RUN, 1u64).unwrap();
                meta.insert(NEXT_ATTEMPT, 1u64).unwrap();
            }
            tx.commit().unwrap();
        }
        (s, d, p, g)
    }

    #[test]
    fn a_run_and_an_attempt_stored_before_entry_ids_still_read() {
        let (s, _d, p, g) = with_legacy_entries();
        let runs = s.gate_runs(&g).unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!((runs[0].id.as_ref(), runs[0].at.as_ref()), (None, None));
        assert_eq!(runs[0].output_excerpt.as_deref(), Some("ok"));
        let attempts = s.attempts(&p).unwrap();
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].id, None);
        assert_eq!(
            attempts[0].paths_touched,
            fl_core::log::PathsTouched::Listed(vec!["a.rs".into()])
        );
    }

    fn kind() -> GateKind {
        GateKind::Command(fl_core::model::CommandSpec {
            program: "true".into(),
            args: vec![],
            delivery: fl_core::model::PopulationDelivery::Args,
            timeout_secs: 5,
            pass_codes: vec![0],
        })
    }

    fn selector() -> Selector {
        Selector::Glob {
            pattern: "**/*.rs".into(),
        }
    }

    #[test]
    fn a_store_from_before_format_versioning_is_refused_with_a_remedy() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("old.redb");
        legacy_store(&path);
        let err = RedbStore::open(&path)
            .err()
            .expect("an unversioned store must be refused");
        assert!(
            matches!(
                err,
                StoreError::FormatVersion {
                    found: None,
                    oldest: FORMAT_VERSION,
                    ..
                }
            ),
            "{err:?}"
        );
        let msg = err.to_string();
        assert!(msg.contains("start a new store"), "no remedy: {msg}");
    }

    // Separate from the refusal above, so that "refuses everything" cannot pass
    // as "refuses the old store".
    #[test]
    fn a_new_store_is_accepted_and_accepted_again_on_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new.redb");
        RedbStore::open(&path).unwrap();
        RedbStore::open(&path).unwrap();
    }

    #[test]
    fn a_store_already_open_is_unreachable_and_names_its_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.redb");
        let _held = RedbStore::open(&path).unwrap();
        let err = RedbStore::open(&path)
            .err()
            .expect("a second open of a held store must fail");
        assert!(matches!(err, StoreError::Unreachable { .. }), "{err:?}");
        assert!(
            err.to_string().contains(&path.display().to_string()),
            "{err}"
        );
    }

    #[test]
    fn redb_store_meets_every_role_contract() {
        use fl_core::conformance::{Single, SplitOver};
        let single = || {
            let (s, g) = fresh();
            Single(s, g)
        };
        let split = || {
            let (s, g) = fresh();
            SplitOver(s, g)
        };
        fl_core::conformance::catalog(fresh);
        fl_core::conformance::tracker(single);
        fl_core::conformance::ledger(single);
        fl_core::conformance::ledger(split);
        fl_core::conformance::split_ledger(split);
        fl_core::conformance::all_roles(single);
        fl_core::conformance::local_handles(fresh);
        fl_core::conformance::ledger_cache(fresh);
        fl_core::conformance::agent_registry(fresh);
        fl_core::conformance::state_store(fresh);
        fl_core::conformance::message_store(fresh);
    }

    #[test]
    fn redb_store_meets_the_escalation_contract() {
        use fl_core::conformance::Single;
        fl_core::conformance::escalations(|| {
            let (s, g) = fresh();
            Single(s, g)
        });
    }

    // Routing spec §3.3 step 1: the shared case, run on its own by name.
    #[test]
    fn a_marked_item_refuses_every_write_and_a_finding_about_it_is_allowed() {
        use fl_core::conformance::{EscalationFixture, Single};
        let (s, g) = fresh();
        Single(s, g).with_escalations(&mut |b| {
            fl_core::conformance::a_marked_item_refuses_every_write_and_a_finding_about_it_is_allowed(
                b,
            )
        });
    }

    /// A project with one gate and one record, in `s`.
    fn gate_and_record(s: &RedbStore) -> (ProjectId, GateId, RecordId) {
        let p = s.add_project("/p").unwrap();
        let g = s
            .add_gate(&p, "fmt", kind(), selector(), 1, "abc", "o")
            .unwrap();
        let r = s.add_record(&p, "t").unwrap();
        (p, g, r)
    }

    // Spec §3.2 step 6: marks are keyed by the repository's node_id, and
    // survive a reopen.
    #[test]
    fn published_marks_are_per_repository_and_survive_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.redb");
        let id = {
            let s = RedbStore::open(&path).unwrap();
            let (_p, g, r) = gate_and_record(&s);
            let run = sample_record_run(1, &g, Some(&r));
            let id = run.id.clone().unwrap();
            s.append_gate_run(run).unwrap();
            // No mark exists yet, and no table: "nothing marked", not a failure.
            assert!(!s.is_published("R_1", &id).unwrap());
            assert_eq!(s.unpublished("R_1", &entry_iri(0)).unwrap().runs.len(), 1);
            s.mark_published("R_1", std::slice::from_ref(&id)).unwrap();
            s.mark_published("R_1", std::slice::from_ref(&id)).unwrap();
            id
        };
        let s = RedbStore::open(&path).unwrap();
        assert!(s.is_published("R_1", &id).unwrap());
        assert!(
            !s.is_published("R_2", &id).unwrap(),
            "a mark is per repository"
        );
        assert!(s.unpublished("R_1", &entry_iri(0)).unwrap().runs.is_empty());
        assert!(
            s.unpublished("R_2", &entry_iri(0)).unwrap().runs.is_empty(),
            "the waiting set is the store's: published anywhere, it waits nowhere"
        );
    }

    #[test]
    fn an_entry_stored_before_ids_is_never_waiting_to_be_published() {
        let (s, _d, _p, _g) = with_legacy_entries();
        let pending = s.unpublished("R_1", &entry_iri(0)).unwrap();
        assert!(pending.runs.is_empty(), "{:?}", pending.runs);
        assert!(pending.attempts.is_empty(), "{:?}", pending.attempts);
    }

    // ⚠ Spec §2.1: runs with no record are never candidates, so a plain
    // `fl check` never lengthens the scan a flush makes.
    #[test]
    fn a_run_with_no_record_never_enters_the_candidate_index() {
        let (s, _d) = fresh();
        let (_p, g, r) = gate_and_record(&s);
        s.append_gate_run(sample_record_run(1, &g, None)).unwrap();
        s.append_gate_run(sample_record_run(2, &g, Some(&r)))
            .unwrap();
        let tx = s.db.begin_read().unwrap();
        let index = tx.open_table(CANDIDATE_RUNS).unwrap();
        let ids: Vec<String> = index
            .iter()
            .unwrap()
            .map(|e| e.unwrap().0.value().to_string())
            .collect();
        assert_eq!(ids, vec![entry_iri(2).to_string()]);
    }

    // ⚠ Spec §2.1: the set a flush scans stays bounded by what is waiting —
    // a published or set-aside entry leaves the candidate index in the same
    // write that marks it.
    #[test]
    fn a_published_or_set_aside_entry_leaves_the_candidate_index() {
        let (s, _d) = fresh();
        let (p, g, r) = gate_and_record(&s);
        let published = sample_record_run(1, &g, Some(&r));
        let aside = sample_record_run(2, &g, Some(&r));
        let waiting = sample_record_run(3, &g, Some(&r));
        for run in [&published, &aside, &waiting] {
            s.append_gate_run(run.clone()).unwrap();
        }
        let attempt = sample_attempt(4, &p, &r);
        s.append_attempt(attempt.clone()).unwrap();
        s.settle(
            "R_1",
            &[published.id.clone().unwrap(), attempt.id.clone().unwrap()],
            &[aside.id.clone().unwrap()],
        )
        .unwrap();

        let tx = s.db.begin_read().unwrap();
        let runs: Vec<String> = tx
            .open_table(CANDIDATE_RUNS)
            .unwrap()
            .iter()
            .unwrap()
            .map(|e| e.unwrap().0.value().to_string())
            .collect();
        assert_eq!(runs, vec![entry_iri(3).to_string()]);
        let attempts = tx
            .open_table(CANDIDATE_ATTEMPTS)
            .unwrap()
            .iter()
            .unwrap()
            .count();
        assert_eq!(attempts, 0);
        drop(tx);
        assert!(!s.is_published("R_1", aside.id.as_ref().unwrap()).unwrap());
        assert_eq!(
            s.unpublished("R_1", &entry_iri(0)).unwrap().runs,
            vec![waiting]
        );
    }

    // Spec §2.1: entries recorded before the cut-over stay local.
    #[test]
    fn only_entries_after_the_cut_over_are_waiting() {
        let (s, _d) = fresh();
        let (_p, g, r) = gate_and_record(&s);
        for n in [2, 3, 4] {
            s.append_gate_run(sample_record_run(n, &g, Some(&r)))
                .unwrap();
        }
        let ids: Vec<Option<fl_core::Iri>> = s
            .unpublished("R_1", &entry_iri(3))
            .unwrap()
            .runs
            .into_iter()
            .map(|r| r.id)
            .collect();
        assert_eq!(ids, vec![Some(entry_iri(4))]);
    }

    #[test]
    fn a_cut_over_is_recorded_once_and_survives_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.redb");
        {
            let s = RedbStore::open(&path).unwrap();
            assert_eq!(s.cutover("R_1").unwrap(), None);
            s.set_cutover("R_1", &entry_iri(3)).unwrap();
            s.set_cutover("R_1", &entry_iri(3)).unwrap();
            let err = s.set_cutover("R_1", &entry_iri(4)).unwrap_err();
            assert!(matches!(err, StoreError::CutoverChanged { .. }), "{err:?}");
        }
        let s = RedbStore::open(&path).unwrap();
        assert_eq!(s.cutover("R_1").unwrap(), Some(entry_iri(3)));
        assert_eq!(s.cutover("R_2").unwrap(), None);
    }

    #[test]
    fn the_ledger_cache_survives_a_reopen() {
        use fl_core::split::{CachedSegment, LedgerCache};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.redb");
        let seg = CachedSegment {
            oid: "o1".into(),
            bytes: b"{}\n".to_vec(),
            closed: false,
        };
        {
            let s = RedbStore::open(&path).unwrap();
            s.remember("R_1", "c1", &[("runs/aa/1.jsonl".to_string(), seg.clone())])
                .unwrap();
        }
        let s = RedbStore::open(&path).unwrap();
        assert_eq!(s.last_head("R_1").unwrap().as_deref(), Some("c1"));
        assert_eq!(s.cached("R_1", "runs/aa/1.jsonl").unwrap(), Some(seg));
    }

    // ⚠ A ledger cache written before `CachedSegment` held bytes (its
    // `text` a JSON string) reads back unchanged: the change is additive,
    // nothing to migrate or download again.
    #[test]
    fn a_ledger_cache_row_written_as_text_still_reads() {
        use fl_core::split::{CachedSegment, LedgerCache};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.redb");
        {
            let s = RedbStore::open(&path).unwrap();
            let tx = s.db.begin_write().unwrap();
            tx.open_table(LEDGER_SEGMENTS)
                .unwrap()
                .insert(
                    ("R_1", "runs/aa/1.jsonl"),
                    r#"{"oid":"o1","text":"{}\n","closed":true}"#,
                )
                .unwrap();
            tx.commit().unwrap();
        }
        let s = RedbStore::open(&path).unwrap();
        assert_eq!(
            s.cached("R_1", "runs/aa/1.jsonl").unwrap(),
            Some(CachedSegment {
                oid: "o1".into(),
                bytes: b"{}\n".to_vec(),
                closed: true,
            })
        );
    }

    use fl_core::model::{Regret, Transition};

    // The export/import contract runs against both stores in
    // `manifest_conformance`; what stays here is redb's own file format.

    /// An authoring store with one project: gates `fmt` and `lint`, and a
    /// transition over both. Returns the store's guard too.
    fn authoring() -> (RedbStore, tempfile::TempDir, ProjectId, GateId, GateId) {
        let (s, dir) = fresh();
        let p = s.add_project("/author").unwrap();
        let g1 = s
            .add_gate(&p, "fmt", kind(), selector(), 1, "c1", "o")
            .unwrap();
        let g2 = s
            .add_gate(&p, "lint", kind(), selector(), 1, "c1", "o")
            .unwrap();
        s.add_transition(Transition {
            project: p.clone(),
            name: "ship".into(),
            from: State::Review,
            to: State::Done,
            regret: Regret::High,
            gates: vec![g1.clone(), g2.clone()],
        })
        .unwrap();
        (s, dir, p, g1, g2)
    }

    #[test]
    fn a_store_that_imported_is_format_3_and_reopens() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("b.redb");
        let (a, _ga, p, _, _) = authoring();
        {
            let b = RedbStore::open(&path).unwrap();
            b.import_manifest(&a.export_manifest(&p, "c1", 7, None).unwrap(), "/x")
                .unwrap();
        }
        let db = redb::Database::open(&path).unwrap();
        let tx = db.begin_read().unwrap();
        let meta = tx.open_table(META).unwrap();
        assert_eq!(
            meta.get(FORMAT_KEY).unwrap().map(|v| v.value()),
            Some(FORMAT_WITH_IMPORTS),
            "an older fl must refuse this store rather than ignoring its imports"
        );
        drop(meta);
        drop(tx);
        drop(db);
        let b = RedbStore::open(&path).unwrap();
        assert!(b.imported_hash(&p).unwrap().is_some());
    }

    #[test]
    fn a_binding_survives_a_reopen_and_a_fresh_store_has_none() {
        use fl_core::store::Bindings;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("b.redb");
        {
            let s = RedbStore::open(&path).unwrap();
            assert_eq!(s.bound_node_id("acme/widgets").unwrap(), None);
            s.bind_node_id("Acme/Widgets", "R_1").unwrap();
        }
        let s = RedbStore::open(&path).unwrap();
        assert_eq!(
            s.bound_node_id("ACME/widgets").unwrap().as_deref(),
            Some("R_1")
        );
    }

    // Ruling 12: a store a newer fl wrote says to upgrade, not to start over.
    #[test]
    fn a_store_from_a_newer_fl_says_to_upgrade_not_to_start_a_new_store() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new.redb");
        {
            let db = redb::Database::create(&path).unwrap();
            let tx = db.begin_write().unwrap();
            tx.open_table(META)
                .unwrap()
                .insert(FORMAT_KEY, FORMAT_WITH_WORKFLOW + 1)
                .unwrap();
            tx.commit().unwrap();
        }
        let msg = RedbStore::open(&path).err().unwrap().to_string();
        assert!(msg.contains("Upgrade fl"), "{msg}");
        assert!(!msg.contains("start a new store"), "{msg}");
    }

    /// The format a store at `path` records.
    fn format_at(path: &std::path::Path) -> Option<u64> {
        let db = redb::Database::open(path).unwrap();
        let tx = db.begin_read().unwrap();
        let meta = tx.open_table(META).unwrap();
        meta.get(FORMAT_KEY).unwrap().map(|v| v.value())
    }

    // Routing spec §1.4: an older fl would read an item and drop its area,
    // so the first item with one raises the store to 5 — in the same write.
    // A store that never routes stays where it was.
    #[test]
    fn an_item_with_an_area_raises_the_store_to_format_5_and_one_without_does_not() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.redb");
        let r = {
            let s = RedbStore::open(&path).unwrap();
            let p = s.add_project("/p").unwrap();
            let plain = s.add_record(&p, "t").unwrap();
            s.add_finding(Finding::raise(p.clone(), plain, "a", "c"))
                .unwrap();
            drop(s);
            assert_eq!(format_at(&path), Some(FORMAT_VERSION));
            let s = RedbStore::open(&path).unwrap();
            s.add_record_with_area(&p, "u", Some("code")).unwrap()
        };
        assert_eq!(format_at(&path), Some(FORMAT_WITH_ROUTING));
        let s = RedbStore::open(&path).unwrap();
        assert_eq!(
            s.get_record(&r).unwrap().unwrap().area.as_deref(),
            Some("code")
        );

        let path = dir.path().join("b.redb");
        {
            let s = RedbStore::open(&path).unwrap();
            let p = s.add_project("/p").unwrap();
            let rec = s.add_record(&p, "t").unwrap();
            let mut f = Finding::raise(p, rec, "a", "c");
            f.area = Some("design".into());
            s.add_finding(f).unwrap();
        }
        assert_eq!(
            format_at(&path),
            Some(FORMAT_WITH_ROUTING),
            "a finding raises it too"
        );
    }

    // ⚠ An older fl would open a store at format 2 or 3 and export its
    // manifest WITHOUT the root; format 4 makes it refuse the store instead.
    #[test]
    fn a_ledger_root_is_recorded_once_and_raises_the_store_to_format_4() {
        use fl_core::store::Bindings;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.redb");
        {
            let s = RedbStore::open(&path).unwrap();
            assert!(!s.holds_a_ledger_root().unwrap());
            assert_eq!(s.ledger_root("R_1").unwrap(), None);
            s.set_ledger_root("R_1", "abc").unwrap();
            s.set_ledger_root("R_1", "abc").unwrap();
            let err = s.set_ledger_root("R_1", "def").unwrap_err();
            assert!(
                matches!(err, StoreError::LedgerRootChanged { ref held, ref found, .. } if held == "abc" && found == "def"),
                "{err:?}"
            );
            assert!(s.holds_a_ledger_root().unwrap());
        }
        let db = redb::Database::open(&path).unwrap();
        let tx = db.begin_read().unwrap();
        let meta = tx.open_table(META).unwrap();
        assert_eq!(
            meta.get(FORMAT_KEY).unwrap().map(|v| v.value()),
            Some(FORMAT_WITH_LEDGER_ROOT),
            "an older fl must refuse this store rather than export its manifest without the root"
        );
        drop(meta);
        drop(tx);
        drop(db);
        let s = RedbStore::open(&path).unwrap();
        assert_eq!(s.ledger_root("R_1").unwrap().as_deref(), Some("abc"));
        assert_eq!(s.ledger_root("R_2").unwrap(), None);
    }

    // Routing spec §1.2, §1.4: the map is kept, raising the store to 5 in
    // the same write; a store holds routing once its project has a map.
    #[test]
    fn a_routing_map_is_kept_and_raises_the_store_to_format_5() {
        use fl_core::routing::{Routes, RoutingMap};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("r.redb");
        let p = {
            let s = RedbStore::open(&path).unwrap();
            let p = s.add_project("/p").unwrap();
            assert!(!s.holds_routing().unwrap());
            assert_eq!(s.routes(&p).unwrap(), None, "no table yet: no map");
            s.set_routes(&p, &RoutingMap::starting()).unwrap();
            assert!(s.holds_routing().unwrap());
            p
        };
        assert_eq!(format_at(&path), Some(FORMAT_WITH_ROUTING));
        let s = RedbStore::open(&path).unwrap();
        assert_eq!(s.routes(&p).unwrap(), Some(RoutingMap::starting()));
        let err = s.routes(&ProjectId(fl_core::ids::seq_iri(99))).unwrap_err();
        assert!(matches!(err, StoreError::NotOwned { .. }), "{err:?}");
    }

    // Routing spec decision 20: handles are numbered per store, so a routed
    // store holds exactly one project — no map while another project is
    // there, and no other project once there is one.
    #[test]
    fn a_routed_store_holds_exactly_one_project() {
        use fl_core::routing::{Routes, RoutingMap};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("two.redb");
        {
            let s = RedbStore::open(&path).unwrap();
            let p = s.add_project("/p").unwrap();
            let q = s.add_project("/q").unwrap();
            let err = s
                .set_routes(&p, &RoutingMap::starting())
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("needs a store of its own")
                    && err.contains(q.iri().as_str())
                    && err.contains("its own `store`"),
                "{err}"
            );
            assert_eq!(s.routes(&p).unwrap(), None, "nothing written");
            assert!(!s.holds_routing().unwrap());
        }
        assert_eq!(format_at(&path), Some(FORMAT_VERSION));
        let (s, _d) = fresh();
        let p = s.add_project("/p").unwrap();
        s.set_routes(&p, &RoutingMap::starting()).unwrap();
        let err = s.add_project("/q").unwrap_err().to_string();
        assert!(err.contains("needs a store of its own"), "{err}");
        assert_eq!(s.list_projects().unwrap().len(), 1);
    }

    #[test]
    fn a_map_that_is_not_valid_is_refused_before_anything_is_written() {
        use fl_core::routing::{Routes, RoutingMap};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("r.redb");
        {
            let s = RedbStore::open(&path).unwrap();
            let p = s.add_project("/p").unwrap();
            let mut bad = RoutingMap::starting();
            bad.areas[0].area = "Code".into();
            let err = s.set_routes(&p, &bad).unwrap_err().to_string();
            assert!(err.contains("is not an area name"), "{err}");
            assert_eq!(s.routes(&p).unwrap(), None);
        }
        assert_eq!(format_at(&path), Some(FORMAT_VERSION));
    }

    #[test]
    fn the_local_store_keeps_a_checked_reference_to_a_github_record() {
        use fl_core::routing::{ForeignRecord, Tier};
        let (s, _d) = fresh();
        let p = s.add_project("/p").unwrap();
        let url = RecordId(Iri::parse("https://github.com/acme/widgets/issues/7").unwrap());
        let err = s
            .add_finding(Finding::raise(p.clone(), url.clone(), "rev", "claim"))
            .unwrap_err();
        assert!(
            matches!(err, StoreError::NotOwned { .. }),
            "unchecked stays refused: {err:?}"
        );
        // Raised against the placeholder: the stored reference must come
        // from the proof.
        let f = Finding::raise(
            p.clone(),
            RecordId(fl_core::ids::seq_iri(0)),
            "rev",
            "claim",
        );
        let id = s
            .add_finding_checked(
                f.clone(),
                ForeignRecord::for_tests(url.clone(), "t", Tier::Github),
            )
            .unwrap();
        assert_eq!(s.get_finding(&id).unwrap().unwrap().record, url);
        let err = s
            .add_finding_checked(f.clone(), ForeignRecord::for_tests(url, "t", Tier::Local))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("is a record in the local tier, so a finding about it"),
            "{err}"
        );
        let held = s.add_record(&p, "t").unwrap();
        let err = s
            .add_finding_checked(f, ForeignRecord::for_tests(held, "t", Tier::Github))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("is held by this store, so a finding about it"),
            "{err}"
        );
    }

    fn sample_mark() -> Mark {
        Mark {
            by: "alice".into(),
            reason: "needs a design review".into(),
            at_ms: 1_000,
        }
    }

    fn issue(n: u64) -> Iri {
        Iri::parse(&format!("https://github.com/acme/widgets/issues/{n}")).unwrap()
    }

    // Routing spec §3.3: "running the command again resumes from where it
    // stopped" — so the mark and the tombstone are the store's, and a new
    // process reads them back.
    #[test]
    fn marks_and_tombstones_survive_a_close_and_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.redb");
        let (r, f, tomb) = {
            let s = RedbStore::open(&path).unwrap();
            let p = s.add_project("/p").unwrap();
            let r = s.add_record(&p, "t").unwrap();
            let f = s
                .add_finding(Finding::raise(p, r.clone(), "rev", "c"))
                .unwrap();
            s.mark(r.iri(), &sample_mark()).unwrap();
            s.mark(f.iri(), &sample_mark()).unwrap();
            let tomb = s.tombstone(f.iri(), &issue(7)).unwrap();
            (r, f, tomb)
        };
        let s = RedbStore::open(&path).unwrap();
        assert_eq!(s.mark_of(r.iri()).unwrap(), Some(sample_mark()));
        assert_eq!(s.tombstone_of(r.iri()).unwrap(), None);
        assert_eq!(s.tombstone_of(f.iri()).unwrap(), Some(tomb));
        assert_eq!(s.mark_of(f.iri()).unwrap(), None);
        let err = s.set_record_state(&r, State::Doing).unwrap_err();
        assert!(matches!(err, StoreError::Escalating { .. }), "{err:?}");
        let err = s.get_finding(&f).unwrap_err();
        assert!(matches!(err, StoreError::Escalated { .. }), "{err:?}");
    }

    // Routing spec §1.4, §3.6: an older fl would ignore a mark and let a
    // write through, so the first mark raises the store to 5. A store that
    // never marks stays where it was, and holds neither table: a write that
    // only looks for a mark or a tombstone creates nothing.
    #[test]
    fn the_first_mark_raises_the_store_to_format_5_and_a_store_that_never_marks_stays() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.redb");
        let r = {
            let s = RedbStore::open(&path).unwrap();
            let p = s.add_project("/p").unwrap();
            let r = s.add_record(&p, "t").unwrap();
            let f = s
                .add_finding(Finding::raise(p, r.clone(), "rev", "c"))
                .unwrap();
            s.set_record_state(&r, State::Doing).unwrap();
            let mut held = s.get_finding(&f).unwrap().unwrap();
            held.withdraw("not concrete").unwrap();
            s.update_finding(&held).unwrap();
            s.add_alias(r.iri(), issue(41)).unwrap();
            assert_eq!(s.mark_of(r.iri()).unwrap(), None);
            r
        };
        assert_eq!(format_at(&path), Some(FORMAT_VERSION));
        {
            let db = redb::Database::open(&path).unwrap();
            let tx = db.begin_read().unwrap();
            for table in [ESCALATING, TOMBSTONES] {
                assert!(
                    matches!(
                        tx.open_table(table),
                        Err(redb::TableError::TableDoesNotExist(_))
                    ),
                    "{} was created",
                    table.name()
                );
            }
        }
        RedbStore::open(&path)
            .unwrap()
            .mark(r.iri(), &sample_mark())
            .unwrap();
        assert_eq!(format_at(&path), Some(FORMAT_WITH_ROUTING));

        let path = dir.path().join("b.redb");
        let r = {
            let s = RedbStore::open(&path).unwrap();
            let p = s.add_project("/p").unwrap();
            s.set_ledger_root("node", "abc").unwrap();
            s.add_record(&p, "t").unwrap()
        };
        assert_eq!(format_at(&path), Some(FORMAT_WITH_LEDGER_ROOT));
        RedbStore::open(&path)
            .unwrap()
            .mark(r.iri(), &sample_mark())
            .unwrap();
        assert_eq!(format_at(&path), Some(FORMAT_WITH_ROUTING));
    }

    // Routing spec §2.3: a local handle of an escalated item resolves
    // through its tombstone, so its id, its handle and its row stay, and the
    // id still chooses this store.
    #[test]
    fn an_escalated_items_handle_still_resolves_and_the_store_still_owns_it() {
        let (s, _d) = fresh();
        let p = s.add_project("/p").unwrap();
        let r = s.add_record(&p, "t").unwrap();
        s.mark(r.iri(), &sample_mark()).unwrap();
        s.tombstone(r.iri(), &issue(7)).unwrap();
        assert_eq!(
            s.resolve_handle(Kind::Record, 1).unwrap(),
            Some(r.0.clone())
        );
        assert_eq!(s.handle_of(Kind::Record, r.iri()).unwrap(), Some(1));
        assert!(s.owns(r.iri()).unwrap());
        assert_eq!(s.kind_of(r.iri()).unwrap(), Kind::Record);
    }

    #[test]
    fn redb_store_meets_the_agent_registry_contract() {
        fl_core::conformance::agent_registry(fresh);
    }

    #[test]
    fn redb_store_meets_the_state_store_contract() {
        fl_core::conformance::state_store(fresh);
    }

    #[test]
    fn state_or_agent_write_raises_the_store_to_format_6() {
        use fl_core::at::At;
        use fl_core::state::{
            HandoffEntry, SensitivityPatch, StatePayload, StateScope, UpsertStateArgs,
        };
        use fl_core::store::StateStore;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.redb");
        let st = {
            let s = RedbStore::open(&path).unwrap();
            s.upsert_state(UpsertStateArgs {
                scope: StateScope::Workspace,
                slug: "rule".into(),
                sensitivity: SensitivityPatch::Inherit,
                payload: StatePayload::Handoff(HandoffEntry {
                    summary: "v6".into(),
                    shipped: vec![],
                    in_flight: vec![],
                    blockers: vec![],
                    next_steps: vec![],
                    notes: None,
                }),
                at: At::from_unix_millis(10),
                by: "carol".into(),
            })
            .unwrap()
        };
        assert_eq!(format_at(&path), Some(FORMAT_WITH_STATE));
        let reopened = RedbStore::open(&path).unwrap();
        assert_eq!(reopened.get_state(&st.id).unwrap(), Some(st));
    }

    #[test]
    fn redb_store_meets_the_message_store_contract() {
        fl_core::conformance::message_store(fresh);
    }

    #[test]
    fn wire_write_raises_the_store_to_format_7_and_verifies_invariants() {
        use fl_core::at::At;
        use fl_core::ferrowire::{
            DeliveryStatus, MessageKind, MessageUrgency, SendWireArgs, WireAddress,
        };
        use fl_core::store::MessageStore;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wire.redb");
        {
            let _s = RedbStore::open(&path).unwrap();
        }
        assert_eq!(format_at(&path), Some(FORMAT_VERSION));
        let msg = {
            let s = RedbStore::open(&path).unwrap();
            let (msg, _) = s
                .send_wire(SendWireArgs {
                    id: None,
                    sender_id: None,
                    sender_name: "carol".into(),
                    sender_host: Some("build-host".into()),
                    to: WireAddress::AgentName("bob".into()),
                    kind: MessageKind::Mail,
                    urgency: MessageUrgency::Normal,
                    durable: false,
                    subject: "Format 7 test".into(),
                    body: "Raise to 7".into(),
                    reply_to: None,
                    thread_id: None,
                    deadline_at: None,
                    at: At::from_unix_millis(10),
                })
                .unwrap();
            s.verify_wire_invariants().unwrap();
            msg
        };
        assert_eq!(format_at(&path), Some(FORMAT_WITH_WIRE));
        let reopened = RedbStore::open(&path).unwrap();
        reopened.verify_wire_invariants().unwrap();
        assert_eq!(
            reopened.get_wire_message(&msg.id).unwrap(),
            Some(msg.clone())
        );

        reopened
            .update_wire_receipt(
                &msg.id,
                "bob",
                DeliveryStatus::Acked,
                false,
                At::from_unix_millis(15),
            )
            .unwrap();
        assert_eq!(reopened.prune_wire(At::from_unix_millis(20)).unwrap(), 1);
        reopened.verify_wire_invariants().unwrap();
    }

    #[test]
    fn redb_store_meets_the_workflow_store_contract() {
        fl_core::conformance::workflow_store(fresh);
    }

    #[test]
    fn workflow_write_raises_the_store_to_format_8_and_survives_reopen() {
        use fl_core::at::At;
        use fl_core::ids::{WorkflowId, WorkflowRunId, seq_iri};
        use fl_core::store::WorkflowStore;
        use fl_core::workflow::{
            AssigneeRule, DEFAULT_MAX_TOTAL_STEPS, DEFAULT_STEP_MAX_ATTEMPTS,
            DEFAULT_STEP_MAX_VISITS, END_STEP, EdgeCondition, EdgeDef, JoinPolicy, StepCompletion,
            StepDef, WorkflowDef, WorkflowRun,
        };
        use std::collections::BTreeMap;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("workflow.redb");
        let p = {
            let s = RedbStore::open(&path).unwrap();
            s.add_project("/p").unwrap()
        };
        assert_eq!(format_at(&path), Some(FORMAT_VERSION));

        let (wf, run) = {
            let s = RedbStore::open(&path).unwrap();
            let wf = s
                .upsert_workflow_def(WorkflowDef {
                    id: WorkflowId(seq_iri(801)),
                    project: p,
                    slug: "ci-flow".into(),
                    title: "CI Flow".into(),
                    start_step: "build".into(),
                    steps: vec![StepDef {
                        step_id: "build".into(),
                        title: "Build".into(),
                        prompt: "Run build.".into(),
                        assignee: AssigneeRule::Agent {
                            name: "alice".into(),
                        },
                        reviews_step: None,
                        require_different_vendor: false,
                        completion: StepCompletion::Ungated,
                        join_policy: JoinPolicy::All,
                        max_attempts: DEFAULT_STEP_MAX_ATTEMPTS,
                        max_visits: DEFAULT_STEP_MAX_VISITS,
                        budget_usd_micros: None,
                        max_tokens: None,
                        isolate_worktree: false,
                    }],
                    edges: vec![EdgeDef {
                        from_step: "build".into(),
                        to_step: END_STEP.into(),
                        on: EdgeCondition::Pass,
                    }],
                    max_total_steps: DEFAULT_MAX_TOTAL_STEPS,
                    budget_usd_micros: None,
                    max_tokens: None,
                    declared_keys: BTreeMap::new(),
                    authored_at_commit: "0123456789abcdef0123456789abcdef01234567".into(),
                    authored_by: "bob".into(),
                })
                .unwrap();
            let run = s
                .create_workflow_run(
                    WorkflowRun::new(
                        WorkflowRunId(seq_iri(802)),
                        &wf,
                        "CI Run 1",
                        At::from_unix_millis(100),
                        "bob",
                    )
                    .unwrap(),
                )
                .unwrap();
            (wf, run)
        };
        assert_eq!(format_at(&path), Some(FORMAT_WITH_WORKFLOW));

        let reopened = RedbStore::open(&path).unwrap();
        assert_eq!(reopened.get_workflow_def(&wf.id).unwrap(), Some(wf.clone()));
        assert_eq!(
            reopened.get_workflow_run(&run.id).unwrap(),
            Some(run.clone())
        );
        assert_eq!(
            reopened.handle_of(Kind::Workflow, wf.id.iri()).unwrap(),
            Some(1)
        );
        assert_eq!(
            reopened.handle_of(Kind::WorkflowRun, run.id.iri()).unwrap(),
            Some(1)
        );
    }

    #[test]
    fn a_read_only_open_reads_without_touching_the_file_and_refuses_writes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ro.redb");
        let project = {
            let s = RedbStore::open(&path).unwrap();
            s.add_project("/repo").unwrap()
        };
        let bytes = std::fs::read(&path).unwrap();
        let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        {
            let ro = RedbStore::open_read_only(&path, 0).unwrap();
            assert!(ro.owns(project.iri()).unwrap());
            assert_eq!(ro.get_project(&project).unwrap().unwrap().root, "/repo");
            let err = ro.add_project("/other").expect_err("refused");
            assert!(err.to_string().contains("read-only"), "{err}");
            // Shared: a second reader is fine; a writer is not.
            drop(RedbStore::open_read_only(&path, 0).unwrap());
            let busy = RedbStore::open_with_retries(&path, 0).err().expect("held");
            assert!(crate::is_redb_busy(&busy), "{busy}");
        }
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        assert_eq!(std::fs::metadata(&path).unwrap().modified().unwrap(), mtime);
        // Held for writing: a read-only open is busy, not a different error.
        let writer = RedbStore::open(&path).unwrap();
        let busy = RedbStore::open_read_only(&path, 0).err().expect("held");
        assert!(crate::is_redb_busy(&busy), "{busy}");
        drop(writer);
        // Not a store: an error, and nothing created.
        let missing = dir.path().join("missing.redb");
        assert!(RedbStore::open_read_only(&missing, 0).is_err());
        assert!(!missing.exists());
        let empty = dir.path().join("empty.redb");
        std::fs::write(&empty, b"").unwrap();
        assert!(RedbStore::open_read_only(&empty, 0).is_err());
        assert_eq!(std::fs::metadata(&empty).unwrap().len(), 0);
    }
}
