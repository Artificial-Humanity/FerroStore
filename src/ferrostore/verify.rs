//! Parity between a redb store and the FerroStore it was migrated into, and the
//! verified cutover that uses it.
//!
//! **Tables.** One [`TableParity`] for each redb table `migrate.rs` copies,
//! named after it. The redb side is read here, row by row, and turned into the
//! document id and JSON value the migration is meant to store; it does not call
//! the migration's code, so a wrong mapping shows as a mismatch. A raw `&str`
//! is hashed as the JSON string the migration stores, a JSON-text `&str` as
//! the value it parses to, a `u64` as a number, a `bool` as a boolean and `()`
//! as `null`. redb's `meta.format_version` is checked as the `__meta` entry,
//! against the FerroStore schema document, and is left out of `meta`. Two more
//! entries catch what the copy list misses: `redb:<name>` for any redb table
//! the list does not name (FerroStore holds none of it), and
//! `ferrostore:unlisted` for any FerroStore document, column table or vector
//! collection outside the list (redb holds none of them).
//!
//! **Hash.** SHA-256 over the rows sorted by key, each as
//! `len(key) ‖ key ‖ len(value) ‖ canonical-JSON(value)`, lengths as `u64`
//! little-endian. Canonical JSON has sorted object keys and no whitespace.
//!
//! **Trait views.** Every list-style trait method, run on both stores with the
//! same arguments (taken from the redb side), its items hashed the same way
//! with each item's canonical JSON as its key: the comparison is of what the
//! method returns, not of its order. A failure on the FerroStore side is a
//! mismatch whose one row is the error; a failure on the redb side is `Err`.
//! Methods that take one message, thread or state id run for at most
//! [`PER_ITEM_SAMPLE`] ids, the lowest; the `*_after(0)` and unfiltered views
//! cover every item. FerroStore views run on one in-memory copy of the store's
//! decoded state, so they cost one decode, not one per call.

use super::doc::{self, SCHEMA_COLLECTION, SCHEMA_KEY};
use super::state::DocState;
use super::{FerroStore, Inner};
use crate::{RedbStore, backend, decode};
use fl_core::ferrowire::{ThreadId, WireFilter};
use fl_core::ids::{GateId, MessageId, ProjectId, StateId};
use fl_core::iri::Iri;
use fl_core::split::{LedgerCache, Outbox};
use fl_core::store::{
    AgentRegistry, Catalog, Ledger, MessageStore, StateStore, StoreError, Tracker, WorkflowStore,
};
use redb::{
    MultimapTableHandle, ReadableTable, ReadableTableMetadata, TableDefinition, TableHandle,
};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Mutex;

/// The most ids a per-item view (one thread, one message, one state) runs for.
pub const PER_ITEM_SAMPLE: usize = 32;

/// One table or one trait view, on both sides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableParity {
    pub table: String,
    pub redb_rows: u64,
    pub ferro_rows: u64,
    pub redb_hash: [u8; 32],
    pub ferro_hash: [u8; 32],
}

impl TableParity {
    /// Same row count and same hash.
    pub fn is_clean(&self) -> bool {
        self.redb_rows == self.ferro_rows && self.redb_hash == self.ferro_hash
    }
}

/// Every table and every trait view the verifier compared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParityReport {
    pub tables: Vec<TableParity>,
    pub trait_views: Vec<TableParity>,
}

impl ParityReport {
    /// No table and no view differs.
    pub fn is_clean(&self) -> bool {
        self.mismatches().is_empty()
    }

    /// The tables, then the views, that differ.
    pub fn mismatches(&self) -> Vec<&TableParity> {
        self.tables
            .iter()
            .chain(&self.trait_views)
            .filter(|parity| !parity.is_clean())
            .collect()
    }
}

/// Compare `src` with `dst` table by table and view by view. `Ok` even when
/// they differ; `Err` only when a store cannot be read or decoded.
pub fn verify_redb_vs_ferrostore(
    src: &RedbStore,
    dst: &FerroStore,
) -> Result<ParityReport, StoreError> {
    let state = dst.read_state()?;
    let tables = table_parity(src, &state)?;
    let copy = FerroStore::assemble(
        Inner::Memory(Mutex::new(state)),
        format!("{} (verify copy)", dst.label),
    );
    let trait_views = view_parity(src, &copy)?;
    Ok(ParityReport {
        tables,
        trait_views,
    })
}

/// Migrate the redb store at `redb_path` into a new FerroStore at `fls_path`,
/// verified, as one switch:
///
/// 1. migrate into a new sibling `<name>.migrating-<pid>-<n>-<nonce>` (for
///    `fl.fls`, `fl.fls.migrating-…`), created 0600 with `O_EXCL`: a name
///    already taken is an error, never a file to reuse;
/// 2. verify; a report that is not clean deletes that file, creates no
///    `fls_path`, and is returned as it is, so the caller keeps running on redb;
/// 3. `fsync` the file; 4. move it to `fls_path`; 5. `fsync` the directory;
/// 6. `chmod 0400` the `.redb`, which is kept.
///
/// The `.redb` is held open, and so keeps redb's exclusive lock, from before
/// the copy until after the `chmod`: no other process can write it once it is
/// copied, so nothing written to it is missing from the published `.fls`.
///
/// First, it removes this target's stale temp files: regular files beside
/// `fls_path`, not symlinks, whose names have the step 1 form for this
/// `fls_path`, owned by this user and not modified for 24 hours (a crashed
/// migration's leftovers). A sweep that fails is a warning on stderr.
///
/// Refuses when `fls_path` already exists or `redb_path` does not. The move is
/// `link` then `unlink`, not `rename`: a `rename` would replace an `fls_path`
/// another process created after the check, and `link` refuses to. The file
/// linked is the one this call created and verified: its device and inode are
/// taken when it is created and checked again before and after the link, and a
/// mismatch is refused. This call only ever removes a temp file it created.
///
/// Once the link is made, `fls_path` is the store and the result is
/// `Ok(report)`. A failure after that point is a warning on stderr, not an
/// `Err`: a temp file that cannot be removed is left behind; a directory that
/// cannot be synced leaves the new entry not yet durable; a `.redb` that cannot
/// be made 0400 stays writable.
pub fn migrate_verified(redb_path: &Path, fls_path: &Path) -> Result<ParityReport, StoreError> {
    migrate_verified_with(redb_path, fls_path, |_| Ok(()))
}

/// The device and inode of a file: what a path names at one moment.
type FileId = (u64, u64);

fn file_id(meta: &std::fs::Metadata) -> FileId {
    use std::os::unix::fs::MetadataExt;
    (meta.dev(), meta.ino())
}

/// What `path` names now, without following a symlink. `None` when nothing.
fn id_at(path: &Path) -> Option<FileId> {
    std::fs::symlink_metadata(path)
        .ok()
        .map(|meta| file_id(&meta))
}

/// A temp name no other call uses: the process id, a per-process counter, and
/// a nonce from the clock and a stack address (another process that reuses
/// the pid still differs).
fn temp_path(fls_path: &Path) -> Result<std::path::PathBuf, StoreError> {
    unique_temp_path(fls_path, TEMP_MARK)
}

/// [`temp_path`] with another `mark` between the file name and the numbers:
/// `<name><mark><pid>-<n>-<nonce>`. A compaction's sibling uses
/// `.compact-` (spec §5).
pub(super) fn unique_temp_path(
    fls_path: &Path,
    mark: &str,
) -> Result<std::path::PathBuf, StoreError> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let name = fls_path
        .file_name()
        .ok_or_else(|| backend(format!("{} has no file name", fls_path.display())))?;
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let marker = 0u8;
    let addr = std::ptr::addr_of!(marker) as usize as u64;
    let nonce = (nanos ^ addr.rotate_left(32)) & 0xffff_ffff;
    let mut temp_name = name.to_os_string();
    temp_name.push(format!("{mark}{}-{n}-{nonce:08x}", std::process::id()));
    Ok(fls_path.with_file_name(temp_name))
}

/// What [`temp_path`] puts between the target's file name and its numbers.
const TEMP_MARK: &str = ".migrating-";

/// A temp file left this long without a write is a crashed migration's (or
/// a crashed compaction's, spec §5).
const STALE_TEMP_AGE: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);

/// `candidate` is a name [`unique_temp_path`] makes for a target named
/// `target` with `mark`: `<target><mark><pid>-<n>-<nonce>`, decimal pid and
/// counter, and an eight-digit lowercase hex nonce.
fn is_temp_name_for(candidate: &std::ffi::OsStr, target: &std::ffi::OsStr, mark: &str) -> bool {
    let (Some(candidate), Some(target)) = (candidate.to_str(), target.to_str()) else {
        return false;
    };
    let Some(numbers) = candidate
        .strip_prefix(target)
        .and_then(|rest| rest.strip_prefix(mark))
    else {
        return false;
    };
    let decimal = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let mut parts = numbers.split('-');
    match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some(pid), Some(n), Some(nonce), None) => {
            decimal(pid)
                && decimal(n)
                && nonce.len() == 8
                && nonce
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        }
        _ => false,
    }
}

/// Remove the stale temp files of a migration into `fls_path` (see
/// [`migrate_verified`]). Never an error: what cannot be read or removed is a
/// warning, and the migration goes on.
fn sweep_stale_temps(fls_path: &Path) {
    sweep_stale_temps_marked(fls_path, TEMP_MARK, "migration");
}

/// [`sweep_stale_temps`] for the temps [`unique_temp_path`] makes with
/// `mark`: regular files beside `fls_path`, not symlinks, whose names have
/// that form for this `fls_path`, owned by this user and not modified for
/// [`STALE_TEMP_AGE`]. `what` names the operation in a warning. A
/// compaction (spec §5) sweeps its own `.compact-` temps this way.
pub(super) fn sweep_stale_temps_marked(fls_path: &Path, mark: &str, what: &str) {
    use std::os::unix::fs::MetadataExt;
    let Some(target) = fls_path.file_name() else {
        return;
    };
    let dir = super::parent_dir(fls_path);
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        // No directory, no temp files in it; the migration reports it.
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return,
        Err(err) => {
            warn(&format!(
                "warning: stale {what} temp files in {} were not swept: {err}",
                dir.display()
            ));
            return;
        }
    };
    // SAFETY: `geteuid` takes no arguments, cannot fail and touches no memory.
    let uid = unsafe { libc::geteuid() };
    let now = std::time::SystemTime::now();
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(err) => {
                warn(&format!(
                    "warning: stale {what} temp files in {} were not all swept: {err}",
                    dir.display()
                ));
                return;
            }
        };
        if !is_temp_name_for(&entry.file_name(), target, mark) {
            continue;
        }
        let path = entry.path();
        // `symlink_metadata`: a symlink is itself, never the file it names.
        let Ok(meta) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        let stale = meta
            .modified()
            .ok()
            .and_then(|mtime| now.duration_since(mtime).ok())
            .is_some_and(|age| age > STALE_TEMP_AGE);
        if !meta.file_type().is_file() || meta.uid() != uid || !stale {
            continue;
        }
        if let Err(err) = std::fs::remove_file(&path) {
            warn(&format!(
                "warning: the stale {what} temp file {} was not removed: {err}",
                path.display()
            ));
        }
    }
}

/// Remove `temp` only while it is still the file this call created.
fn remove_own_temp(temp: &Path, ours: FileId) -> std::io::Result<()> {
    if id_at(temp) == Some(ours) {
        std::fs::remove_file(temp)
    } else {
        Ok(())
    }
}

/// [`migrate_verified`] with `after_copy` run on the new store between the
/// copy and the verify. Production passes a no-op; tests use it to make the
/// two stores differ.
pub(super) fn migrate_verified_with(
    redb_path: &Path,
    fls_path: &Path,
    after_copy: impl FnOnce(&FerroStore) -> Result<(), StoreError>,
) -> Result<ParityReport, StoreError> {
    migrate_verified_hooked(redb_path, fls_path, after_copy, || {})
}

/// [`migrate_verified_with`] with `before_chmod` also run after the link and
/// the directory `fsync`, just before the `chmod`: tests use it to look at the
/// `.redb` while the migration still holds it.
#[cfg(test)]
pub(super) fn migrate_verified_with_before_chmod(
    redb_path: &Path,
    fls_path: &Path,
    after_copy: impl FnOnce(&FerroStore) -> Result<(), StoreError>,
    before_chmod: impl FnOnce(),
) -> Result<ParityReport, StoreError> {
    migrate_verified_hooked(redb_path, fls_path, after_copy, before_chmod)
}

fn migrate_verified_hooked(
    redb_path: &Path,
    fls_path: &Path,
    after_copy: impl FnOnce(&FerroStore) -> Result<(), StoreError>,
    before_chmod: impl FnOnce(),
) -> Result<ParityReport, StoreError> {
    sweep_stale_temps(fls_path);
    if std::fs::symlink_metadata(fls_path).is_ok() {
        return Err(backend(format!(
            "{} already exists: a verified migration only creates a new store",
            fls_path.display()
        )));
    }
    if !redb_path.is_file() {
        return Err(backend(format!(
            "{} is not a redb store file: nothing to migrate",
            redb_path.display()
        )));
    }
    let temp = temp_path(fls_path)?;
    migrate_into(redb_path, fls_path, &temp, after_copy, before_chmod)
}

/// [`migrate_verified`] through the temp file `temp`, for a test that must
/// name it.
#[cfg(test)]
pub(super) fn migrate_verified_with_temp(
    redb_path: &Path,
    fls_path: &Path,
    temp: &Path,
) -> Result<ParityReport, StoreError> {
    migrate_into(redb_path, fls_path, temp, |_| Ok(()), || {})
}

/// Steps 1 to 6 of [`migrate_verified`], through `temp`, once the checks on
/// `redb_path` and `fls_path` have passed.
fn migrate_into(
    redb_path: &Path,
    fls_path: &Path,
    temp: &Path,
    after_copy: impl FnOnce(&FerroStore) -> Result<(), StoreError>,
    before_chmod: impl FnOnce(),
) -> Result<ParityReport, StoreError> {
    // `create_new` is `O_EXCL`: a taken name fails here and is not touched.
    let held = {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(temp)
            .map_err(|err| backend(format!("create {}: {err}", temp.display())))?
    };
    let ours = file_id(
        &held
            .metadata()
            .map_err(|err| backend(format!("stat {}: {err}", temp.display())))?,
    );
    let cleanup = |err: StoreError| -> StoreError {
        let _ = remove_own_temp(temp, ours);
        err
    };

    // Invariant: `src` holds redb's exclusive lock on the `.redb` from before
    // the copy until after the `chmod`, and is dropped only once the steps
    // below (and any temp cleanup on their error paths) are done, so no other
    // process can write the `.redb` after it was copied.
    let src = RedbStore::open(redb_path).map_err(cleanup)?;
    let result = (|| {
        let report = {
            let dst = FerroStore::open(temp).map_err(cleanup)?;
            let outcome = super::migrate_redb_to_ferrostore(&src, &dst)
                .and_then(|()| after_copy(&dst))
                .and_then(|()| verify_redb_vs_ferrostore(&src, &dst));
            drop(dst);
            match outcome {
                Ok(report) if report.is_clean() => report,
                Ok(report) => {
                    remove_own_temp(temp, ours)
                        .map_err(|err| backend(format!("remove {}: {err}", temp.display())))?;
                    return Ok(report);
                }
                Err(err) => return Err(cleanup(err)),
            }
        };

        // Publish the inode that was migrated and verified, and no other.
        let replaced = |when: &str| {
            backend(format!(
                "{} is no longer the file this migration verified ({when}): not published",
                temp.display()
            ))
        };
        if id_at(temp) != Some(ours) {
            return Err(cleanup(replaced("before the link")));
        }
        held.sync_all()
            .map_err(|err| cleanup(backend(format!("sync {}: {err}", temp.display()))))?;
        std::fs::hard_link(temp, fls_path).map_err(|err| {
            cleanup(backend(format!(
                "move {} to {}: {err}",
                temp.display(),
                fls_path.display()
            )))
        })?;
        let linked = id_at(fls_path);
        if linked != Some(ours) {
            // The temp name was swapped between the check and the link, so the
            // link this call made names that other file: take it back. An
            // `fls_path` that is not the temp's file is not this call's link and
            // is left alone.
            if linked.is_some() && linked == id_at(temp) {
                let _ = std::fs::remove_file(fls_path);
            }
            return Err(cleanup(replaced("after the link")));
        }
        drop(held);

        // `fls_path` is the store from here on: what fails now is a warning.
        if let Err(err) = remove_own_temp(temp, ours) {
            warn(&format!(
                "warning: {} is published, but its temp file {} was not removed: {err}",
                fls_path.display(),
                temp.display()
            ));
        }
        if let Err(err) = super::sync_dir(&super::parent_dir(fls_path)) {
            warn(&format!(
                "warning: {} is published, but its directory was not synced, so a crash \
                 may still lose the new entry: {err}",
                fls_path.display()
            ));
        }
        before_chmod();
        {
            use std::os::unix::fs::PermissionsExt;
            if let Err(err) =
                std::fs::set_permissions(redb_path, std::fs::Permissions::from_mode(0o400))
            {
                warn(&format!(
                    "warning: {} is published, but chmod 0400 {} failed, so the redb store \
                     remains writable: {err}",
                    fls_path.display(),
                    redb_path.display()
                ));
            }
        }
        Ok(report)
    })();
    // The lock is released here, last, on every path.
    drop(src);
    result
}

/// One line on stderr, as the store's other warnings print: a closed stderr is
/// not a panic.
fn warn(text: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "{text}");
}

// ---- hashing ----------------------------------------------------------------

/// Rows of one side: `(key, value)`.
type Rows = Vec<(String, Value)>;

fn canonical(value: &Value, out: &mut Vec<u8>) -> Result<(), StoreError> {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push(b'{');
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                serde_json::to_writer(&mut *out, key).map_err(backend)?;
                out.push(b':');
                canonical(&map[key], out)?;
            }
            out.push(b'}');
        }
        Value::Array(items) => {
            out.push(b'[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                canonical(item, out)?;
            }
            out.push(b']');
        }
        scalar => serde_json::to_writer(&mut *out, scalar).map_err(backend)?,
    }
    Ok(())
}

fn canonical_bytes(value: &Value) -> Result<Vec<u8>, StoreError> {
    let mut out = Vec::new();
    canonical(value, &mut out)?;
    Ok(out)
}

/// Row count and hash of `rows`, sorted by key (then value, for equal keys).
fn digest(rows: Rows) -> Result<(u64, [u8; 32]), StoreError> {
    let mut encoded = Vec::with_capacity(rows.len());
    for (key, value) in rows {
        encoded.push((key.into_bytes(), canonical_bytes(&value)?));
    }
    encoded.sort();
    let mut hasher = Sha256::new();
    for (key, value) in &encoded {
        hasher.update((key.len() as u64).to_le_bytes());
        hasher.update(key);
        hasher.update((value.len() as u64).to_le_bytes());
        hasher.update(value);
    }
    let count = u64::try_from(encoded.len()).map_err(|_| backend("row count exceeds u64"))?;
    Ok((count, hasher.finalize().into()))
}

fn parity(table: impl Into<String>, redb: Rows, ferro: Rows) -> Result<TableParity, StoreError> {
    let (redb_rows, redb_hash) = digest(redb)?;
    let (ferro_rows, ferro_hash) = digest(ferro)?;
    Ok(TableParity {
        table: table.into(),
        redb_rows,
        ferro_rows,
        redb_hash,
        ferro_hash,
    })
}

// ---- tables -----------------------------------------------------------------

const SEP: char = '\u{1f}';

fn pair(a: &str, b: &str) -> String {
    format!("{a}{SEP}{b}")
}

fn raw(text: &str) -> Value {
    Value::String(text.to_string())
}

fn json_text(table: &str, key: &str, text: &str) -> Result<Value, StoreError> {
    serde_json::from_str(text).map_err(|err| {
        decode(format!(
            "redb table `{table}` key `{key}` is not JSON: {err}"
        ))
    })
}

/// Every row of one redb table, mapped by `row`. A table redb never created
/// has no rows.
macro_rules! redb_rows {
    ($tx:expr, $name:expr, $k:ty, $v:ty, |$key:ident, $value:ident| $row:expr) => {{
        let mut rows: Rows = Vec::new();
        match $tx.open_table(TableDefinition::<$k, $v>::new($name)) {
            Ok(table) => {
                for entry in table.iter().map_err(backend)? {
                    let (key_guard, value_guard) = entry.map_err(backend)?;
                    let $key = key_guard.value();
                    let $value = value_guard.value();
                    let row: Result<(String, Value), StoreError> = $row;
                    rows.push(row?);
                }
            }
            Err(redb::TableError::TableDoesNotExist(_)) => {}
            Err(err) => return Err(backend(err)),
        }
        rows
    }};
}

/// `name`'s rows as the migration should have stored them. `None` for a name
/// the migration does not copy.
fn redb_table(tx: &redb::ReadTransaction, name: &str) -> Result<Option<Rows>, StoreError> {
    let rows = match name {
        doc::META => {
            let mut rows = redb_rows!(tx, name, &str, u64, |k, v| Ok((
                k.to_string(),
                Value::from(v)
            )));
            rows.retain(|(key, _)| key != doc::FORMAT_KEY);
            rows
        }
        doc::IDS
        | doc::ALIASES
        | doc::IMPORTS
        | doc::GITHUB_BINDINGS
        | doc::LEDGER_CUTOVERS
        | doc::LEDGER_ROOTS
        | doc::LEDGER_HEADS
        | doc::AGENT_BY_NAME => {
            redb_rows!(tx, name, &str, &str, |k, v| Ok((k.to_string(), raw(v))))
        }
        doc::PROJECTS
        | doc::GATES
        | doc::RECORDS
        | doc::FINDINGS
        | doc::ROUTING
        | doc::ESCALATING
        | doc::TOMBSTONES
        | doc::STATE_ENTRIES
        | doc::STATE_CONFLICTS
        | doc::STATE_SYNC_CURSORS
        | doc::AGENTS
        | doc::AREAS
        | doc::WIRE_MESSAGES
        | doc::WORKFLOW_DEFS
        | doc::WORKFLOW_RUNS => {
            redb_rows!(tx, name, &str, &str, |k, v| Ok((
                k.to_string(),
                json_text(name, k, v)?
            )))
        }
        doc::HANDLE_OF
        | doc::CANDIDATE_RUNS
        | doc::CANDIDATE_ATTEMPTS
        | doc::STATE_REVISION_IDS
        | doc::AGENT_EVENT_IDS
        | doc::WIRE_MSG_SEQ => {
            redb_rows!(tx, name, &str, u64, |k, v| Ok((
                k.to_string(),
                Value::from(v)
            )))
        }
        doc::HANDLES => redb_rows!(tx, name, (&str, u64), &str, |k, v| Ok((
            format!("{}{SEP}{}", k.0, k.1),
            raw(v)
        ))),
        doc::TRANSITIONS | doc::LEDGER_SEGMENTS | doc::WIRE_INBOX => {
            redb_rows!(tx, name, (&str, &str), &str, |k, v| {
                let key = pair(k.0, k.1);
                let value = json_text(name, &key, v)?;
                Ok((key, value))
            })
        }
        doc::WORKFLOW_SLUGS => redb_rows!(tx, name, (&str, &str), &str, |k, v| Ok((
            pair(k.0, k.1),
            raw(v)
        ))),
        doc::LEDGER_PUBLISHED => redb_rows!(tx, name, (&str, &str), bool, |k, v| Ok((
            pair(k.0, k.1),
            Value::Bool(v)
        ))),
        doc::WIRE_THREADS | doc::WIRE_REPLIES => {
            redb_rows!(tx, name, (&str, &str), (), |k, _v| Ok((
                pair(k.0, k.1),
                Value::Null
            )))
        }
        doc::STATE_KEYS => redb_rows!(tx, name, (&str, &str, &str), &str, |k, v| Ok((
            format!("{}{SEP}{}{SEP}{}", k.0, k.1, k.2),
            raw(v)
        ))),
        doc::GATE_RUNS
        | doc::ATTEMPTS
        | doc::STATE_REVISIONS
        | doc::AGENT_EVENTS
        | doc::WIRE_RCPT_LOG => redb_rows!(tx, name, u64, &str, |k, v| {
            let key = k.to_string();
            let value = json_text(name, &key, v)?;
            Ok((key, value))
        }),
        doc::WIRE_MSG_LOG => {
            redb_rows!(tx, name, u64, &str, |k, v| Ok((k.to_string(), raw(v))))
        }
        _ => return Ok(None),
    };
    Ok(Some(rows))
}

/// Every redb table `migrate.rs` copies, in its order.
const COPIED: [&str; 45] = [
    doc::META,
    doc::IDS,
    doc::ALIASES,
    doc::HANDLES,
    doc::HANDLE_OF,
    doc::PROJECTS,
    doc::GATES,
    doc::TRANSITIONS,
    doc::RECORDS,
    doc::FINDINGS,
    doc::GATE_RUNS,
    doc::ATTEMPTS,
    doc::IMPORTS,
    doc::GITHUB_BINDINGS,
    doc::LEDGER_PUBLISHED,
    doc::LEDGER_CUTOVERS,
    doc::CANDIDATE_RUNS,
    doc::CANDIDATE_ATTEMPTS,
    doc::LEDGER_ROOTS,
    doc::LEDGER_HEADS,
    doc::LEDGER_SEGMENTS,
    doc::ROUTING,
    doc::ESCALATING,
    doc::TOMBSTONES,
    doc::STATE_ENTRIES,
    doc::STATE_KEYS,
    doc::STATE_REVISIONS,
    doc::STATE_REVISION_IDS,
    doc::STATE_CONFLICTS,
    doc::STATE_SYNC_CURSORS,
    doc::AGENTS,
    doc::AGENT_BY_NAME,
    doc::AREAS,
    doc::AGENT_EVENTS,
    doc::AGENT_EVENT_IDS,
    doc::WIRE_MESSAGES,
    doc::WIRE_MSG_LOG,
    doc::WIRE_MSG_SEQ,
    doc::WIRE_INBOX,
    doc::WIRE_RCPT_LOG,
    doc::WIRE_THREADS,
    doc::WIRE_REPLIES,
    doc::WORKFLOW_DEFS,
    doc::WORKFLOW_SLUGS,
    doc::WORKFLOW_RUNS,
];

fn ferro_collection(state: &DocState, collection: &str) -> Rows {
    state
        .docs
        .in_collection(collection)
        .map(|(id, doc)| (id.clone(), doc.clone()))
        .collect()
}

fn table_parity(src: &RedbStore, state: &DocState) -> Result<Vec<TableParity>, StoreError> {
    let tx = src.db.begin_read().map_err(backend)?;
    let mut out = Vec::with_capacity(COPIED.len() + 2);
    for name in COPIED {
        let redb = redb_table(&tx, name)?
            .ok_or_else(|| backend(format!("the verifier has no mapping for `{name}`")))?;
        out.push(parity(name, redb, ferro_collection(state, name))?);
    }

    // The schema version: redb's `meta.format_version`, FerroStore's `__meta`.
    // Only a missing `meta` table means "no schema version"; any other open
    // error is a read failure.
    let mut schema = Vec::new();
    match tx.open_table(TableDefinition::<&str, u64>::new(doc::META)) {
        Ok(meta) => {
            if let Some(version) = meta.get(doc::FORMAT_KEY).map_err(backend)? {
                schema.push((SCHEMA_KEY.to_string(), Value::from(version.value())));
            }
        }
        Err(redb::TableError::TableDoesNotExist(_)) => {}
        Err(err) => return Err(backend(err)),
    }
    out.push(parity(
        SCHEMA_COLLECTION,
        schema,
        ferro_collection(state, SCHEMA_COLLECTION),
    )?);

    // A redb table the copy list does not name: none of it reached FerroStore.
    let listed: BTreeSet<&str> = COPIED.iter().copied().collect();
    for handle in tx.list_tables().map_err(backend)? {
        let name = handle.name().to_string();
        if listed.contains(name.as_str()) {
            continue;
        }
        let table = tx.open_untyped_table(handle).map_err(backend)?;
        let len = table.len().map_err(backend)?;
        let (_, empty) = digest(Vec::new())?;
        out.push(TableParity {
            table: format!("redb:{name}"),
            redb_rows: len,
            ferro_rows: 0,
            redb_hash: empty,
            ferro_hash: empty,
        });
    }
    for handle in tx.list_multimap_tables().map_err(backend)? {
        let name = handle.name().to_string();
        let table = tx.open_untyped_multimap_table(handle).map_err(backend)?;
        let len = table.len().map_err(backend)?;
        let (_, empty) = digest(Vec::new())?;
        out.push(TableParity {
            table: format!("redb:{name}"),
            redb_rows: len,
            ferro_rows: 0,
            redb_hash: empty,
            ferro_hash: empty,
        });
    }

    // FerroStore data outside the copy list: nothing in redb produced it.
    let mut unlisted: Rows = state
        .docs
        .iter()
        .filter(|(collection, _, _)| {
            *collection != SCHEMA_COLLECTION && !listed.contains(collection.as_str())
        })
        .map(|(collection, id, doc)| (pair(collection, id), doc.clone()))
        .collect();
    unlisted.extend(
        state
            .tables
            .keys()
            .map(|name| (pair("column table", name), Value::Null)),
    );
    unlisted.extend(
        state
            .vectors
            .keys()
            .map(|name| (pair("vector collection", name), Value::Null)),
    );
    out.push(parity("ferrostore:unlisted", Vec::new(), unlisted)?);
    Ok(out)
}

// ---- trait views ------------------------------------------------------------

/// Every store trait a view calls.
trait Viewed:
    Catalog
    + Tracker
    + Ledger
    + Outbox
    + LedgerCache
    + AgentRegistry
    + StateStore
    + MessageStore
    + WorkflowStore
{
}

impl<T> Viewed for T where
    T: Catalog
        + Tracker
        + Ledger
        + Outbox
        + LedgerCache
        + AgentRegistry
        + StateStore
        + MessageStore
        + WorkflowStore
{
}

/// The arguments the views run with, all read from the redb side.
struct Plan {
    projects: Vec<ProjectId>,
    gates: Vec<GateId>,
    /// `(repository node id, cut-over)` for every repository with one.
    cutovers: Vec<(String, Iri)>,
    /// `(repository node id, top directory)` of every cached ledger segment.
    segment_dirs: Vec<(String, String)>,
    threads: Vec<ThreadId>,
    messages: Vec<MessageId>,
    states: Vec<StateId>,
}

fn items<T: Serialize>(list: Result<Vec<T>, StoreError>) -> Result<Rows, StoreError> {
    let mut rows = Vec::new();
    for item in list? {
        let value = serde_json::to_value(item).map_err(backend)?;
        let key = String::from_utf8(canonical_bytes(&value)?).map_err(backend)?;
        rows.push((key, value));
    }
    Ok(rows)
}

/// `(view name, rows)` for every view `plan` names, on `store`.
fn run_views<S: Viewed>(store: &S, plan: &Plan) -> Vec<(String, Result<Rows, StoreError>)> {
    let mut out: Vec<(String, Result<Rows, StoreError>)> = Vec::new();
    out.push((
        "Catalog::list_projects".into(),
        items(store.list_projects()),
    ));
    for p in &plan.projects {
        out.push((
            format!("Catalog::list_gates({p})"),
            items(store.list_gates(p)),
        ));
        out.push((
            format!("Catalog::list_transitions({p})"),
            items(store.list_transitions(p)),
        ));
        out.push((
            format!("Tracker::list_records({p})"),
            items(store.list_records(p)),
        ));
        out.push((
            format!("Tracker::list_findings({p})"),
            items(store.list_findings(p)),
        ));
        out.push((format!("Ledger::attempts({p})"), items(store.attempts(p))));
        out.push((
            format!("WorkflowStore::list_workflow_defs({p})"),
            items(store.list_workflow_defs(Some(p))),
        ));
        out.push((
            format!("WorkflowStore::list_workflow_runs({p})"),
            items(store.list_workflow_runs(Some(p), None)),
        ));
    }
    for g in &plan.gates {
        out.push((format!("Ledger::gate_runs({g})"), items(store.gate_runs(g))));
    }
    for (repo, after) in &plan.cutovers {
        let pending = store.unpublished(repo, after);
        let (runs, attempts) = match pending {
            Ok(pending) => (Ok(pending.runs), Ok(pending.attempts)),
            Err(err) => {
                let text = err.to_string();
                (Err(err), Err(backend(text)))
            }
        };
        out.push((
            format!("Outbox::unpublished({repo}, {after}).runs"),
            items(runs),
        ));
        out.push((
            format!("Outbox::unpublished({repo}, {after}).attempts"),
            items(attempts),
        ));
    }
    for (repo, dir) in &plan.segment_dirs {
        out.push((
            format!("LedgerCache::cached_under({repo}, {dir})"),
            items(store.cached_under(repo, dir)),
        ));
    }
    out.push((
        "AgentRegistry::list_areas".into(),
        items(store.list_areas()),
    ));
    out.push((
        "AgentRegistry::list_agents(all)".into(),
        items(store.list_agents(None, true)),
    ));
    out.push((
        "AgentRegistry::registry_events_after(0)".into(),
        items(store.registry_events_after(0)),
    ));
    out.push((
        "StateStore::list_state(all)".into(),
        items(store.list_state(None, None)),
    ));
    out.push((
        "StateStore::list_conflicts".into(),
        items(store.list_conflicts()),
    ));
    out.push((
        "StateStore::state_revisions_after(0)".into(),
        items(store.state_revisions_after(0)),
    ));
    for id in &plan.states {
        out.push((
            format!("StateStore::state_revisions({id})"),
            items(store.state_revisions(id)),
        ));
    }
    out.push((
        "MessageStore::list_wire_inbox(all)".into(),
        items(store.list_wire_inbox(&WireFilter::default())),
    ));
    out.push((
        "MessageStore::wire_messages_after(0)".into(),
        items(store.wire_messages_after(0)),
    ));
    out.push((
        "MessageStore::wire_receipts_after(0)".into(),
        items(store.wire_receipts_after(0)),
    ));
    for thread in &plan.threads {
        out.push((
            format!("MessageStore::list_wire_thread({thread})"),
            items(store.list_wire_thread(thread)),
        ));
    }
    for id in &plan.messages {
        out.push((
            format!("MessageStore::list_wire_replies({id})"),
            items(store.list_wire_replies(id)),
        ));
        out.push((
            format!("MessageStore::list_wire_receipts({id})"),
            items(store.list_wire_receipts(id)),
        ));
    }
    out.push((
        "WorkflowStore::list_workflow_defs(all)".into(),
        items(store.list_workflow_defs(None)),
    ));
    out.push((
        "WorkflowStore::list_workflow_runs(all)".into(),
        items(store.list_workflow_runs(None, None)),
    ));
    out
}

fn plan_views(src: &RedbStore) -> Result<Plan, StoreError> {
    let projects: Vec<ProjectId> = src.list_projects()?.into_iter().map(|p| p.id).collect();
    let mut gates = Vec::new();
    for p in &projects {
        gates.extend(src.list_gates(p)?.into_iter().map(|g| g.id));
    }
    let tx = src.db.begin_read().map_err(backend)?;
    let mut cutovers = Vec::new();
    for (repo, _) in redb_table(&tx, doc::LEDGER_CUTOVERS)?.unwrap_or_default() {
        if let Some(after) = src.cutover(&repo)? {
            cutovers.push((repo, after));
        }
    }
    let mut dirs = BTreeSet::new();
    for (key, _) in redb_table(&tx, doc::LEDGER_SEGMENTS)?.unwrap_or_default() {
        if let Some((repo, path)) = key.split_once(SEP)
            && let Some((top, _)) = path.split_once('/')
        {
            dirs.insert((repo.to_string(), top.to_string()));
        }
    }
    drop(tx);
    let messages = src.wire_messages_after(0)?;
    let threads: BTreeSet<ThreadId> = messages
        .iter()
        .map(|(_, message)| message.thread_id.clone())
        .collect();
    let mut message_ids: Vec<MessageId> = messages.into_iter().map(|(_, m)| m.id).collect();
    message_ids.sort();
    message_ids.truncate(PER_ITEM_SAMPLE);
    let mut states: Vec<StateId> = src
        .list_state(None, None)?
        .into_iter()
        .map(|entry| entry.id)
        .collect();
    states.sort();
    states.truncate(PER_ITEM_SAMPLE);
    Ok(Plan {
        projects,
        gates,
        cutovers,
        segment_dirs: dirs.into_iter().collect(),
        threads: threads.into_iter().take(PER_ITEM_SAMPLE).collect(),
        messages: message_ids,
        states,
    })
}

fn view_parity(src: &RedbStore, dst: &FerroStore) -> Result<Vec<TableParity>, StoreError> {
    let plan = plan_views(src)?;
    let redb = run_views(src, &plan);
    let ferro: BTreeMap<String, Result<Rows, StoreError>> =
        run_views(dst, &plan).into_iter().collect();
    let mut ferro = ferro;
    let mut out = Vec::with_capacity(redb.len());
    for (name, redb_rows) in redb {
        let redb_rows = redb_rows?;
        let ferro_rows = match ferro.remove(&name) {
            Some(Ok(rows)) => rows,
            Some(Err(err)) => vec![(format!("error: {err}"), Value::Null)],
            None => vec![("error: the view did not run".to_string(), Value::Null)],
        };
        out.push(parity(name, redb_rows, ferro_rows)?);
    }
    Ok(out)
}
