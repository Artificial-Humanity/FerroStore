//! One handle for either store engine.
//!
//! Every command, the hub and the wire daemon open their store through
//! [`open_store`] and hold an [`AnyStore`], so the engine behind a store is
//! chosen in one place. `AnyStore` implements all thirteen store traits by
//! delegation and carries the inherent methods callers use beyond them.
//!
//! * The engine is chosen by [`choose_backend`]: a `.fls` path is always a
//!   [`FerroStore`]; any other path takes `FL_STORE_BACKEND` (`redb` or
//!   `ferro`), then [`DEFAULT_BACKEND`], which is FerroStore.
//!   `FL_STORE_BACKEND=redb` keeps a store on redb.
//! * A logical path such as `fl.redb` names the same store under either
//!   engine: [`physical_path`] maps it to the sibling `fl.fls` when the engine
//!   is FerroStore, so a config entry never changes at cutover.
//! * A scan that only asks [`AnyStore::owns`] ([`Unmigrated::UseRedb`]) opens
//!   a lone `.redb` read-only first and falls back to today's ordinary open
//!   only when the read-only open fails for a non-busy reason (so, as before
//!   this module existed, that fallback can repair a crash-damaged redb or
//!   initialise an empty file). It never migrates.

use crate::manifest::{Manifest, ManifestError};
use crate::{FerroStore, ImportReport, RedbStore, backend};
use fl_core::{
    AgentId, AgentRecord, AgentRegistry, AreaRecord, At, Attempt, Bindings, CachedSegment, Catalog,
    Decision, DeliveryReceipt, DeliveryStatus, Escalations, Finding, FindingId, Flushed,
    ForeignRecord, GateDef, GateId, GateKind, GateRun, Handles, Iri, Kind, Ledger, LedgerCache,
    Mark, MemoryLifecycle, MessageId, MessageStore, Outbox, Pending, Project, ProjectId, Record,
    RecordId, RegisterAgentArgs, RegistryEvent, RevisionId, Routes, RoutingMap, Selector,
    SendWireArgs, State, StateConcern, StateConflict, StateEntry, StateId, StateRevision,
    StateScope, StateStore, StateSyncCursor, StoreError, ThreadId, Tombstone, Tracker, Transition,
    UpsertStateArgs, WireFilter, WireImportReport, WireInboxItem, WireMessage, WorkflowDef,
    WorkflowId, WorkflowRun, WorkflowRunId, WorkflowStore,
};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// The engine behind a store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// redb: single-process, one `.redb` file.
    Redb,
    /// FerroStore: multi-process embedded engine, one `.fls` file.
    Ferro,
}

impl BackendKind {
    /// Parse a `FL_STORE_BACKEND` value.
    pub fn parse(raw: &str) -> Result<Self, StoreError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "redb" => Ok(Self::Redb),
            "ferro" | "ferrostore" | "fls" => Ok(Self::Ferro),
            other => Err(backend(format!(
                "unknown store backend `{other}` in {BACKEND_ENV}: expected `redb` or `ferro`"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Redb => "redb",
            Self::Ferro => "ferro",
        }
    }
}

/// The environment variable that selects the engine for a path that does not
/// name one by its extension.
pub const BACKEND_ENV: &str = "FL_STORE_BACKEND";

/// The engine a path takes when neither the path nor the environment names one:
/// FerroStore. `FL_STORE_BACKEND=redb` selects redb instead.
pub const DEFAULT_BACKEND: BackendKind = BackendKind::Ferro;

/// The extension of a FerroStore file.
pub const FERRO_EXTENSION: &str = "fls";

/// Which engine opens `path`: a `.fls` path is always FerroStore, the explicit
/// path winning over `env`; any other path takes `env` (the value of
/// [`BACKEND_ENV`]), then [`DEFAULT_BACKEND`].
pub fn choose_backend(path: &Path, env: Option<&str>) -> Result<BackendKind, StoreError> {
    if path.extension().and_then(|e| e.to_str()) == Some(FERRO_EXTENSION) {
        return Ok(BackendKind::Ferro);
    }
    match env.map(str::trim).filter(|v| !v.is_empty()) {
        Some(raw) => BackendKind::parse(raw),
        None => Ok(DEFAULT_BACKEND),
    }
}

/// The file `kind` keeps the store named `path` in. Redb keeps it at `path`;
/// FerroStore keeps it at the sibling `.fls`, so a store's configured path
/// is the same before and after a cutover.
pub fn physical_path(path: &Path, kind: BackendKind) -> PathBuf {
    match kind {
        BackendKind::Redb => path.to_path_buf(),
        BackendKind::Ferro => path.with_extension(FERRO_EXTENSION),
    }
}

/// What a ferro selection does with a lone `.redb` (no `.fls` beside it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Unmigrated {
    /// Migrate it, verified (see [`crate::migrate::migrate_verified`]), and open
    /// the new `.fls`. When the migration cannot run or does not verify, the
    /// store opens on the `.redb` with a warning, so that every process that
    /// cannot migrate keeps using one engine. For the store a command works
    /// on, and for `fl hub`.
    #[default]
    Migrate,
    /// Open the `.redb` with the redb engine and migrate nothing. For the scans
    /// that open other configured stores only to ask [`AnyStore::owns`]: a
    /// read-only lookup must not migrate (and `chmod`) every project store.
    /// The redb is opened read-only first (writes through it fail; its bytes
    /// and mtime stay as they were), and no file is created and no lock file
    /// that does not exist already. Only when that read-only open fails for a
    /// reason other than "held by another process" (the file needs repair, or
    /// is not initialised) does the scan fall back to today's ordinary
    /// writable open, which can repair a crash-damaged redb or initialise an
    /// empty file, exactly as before auto-migration existed.
    UseRedb,
    /// Refuse, naming `fl store migrate`.
    Refuse,
}

/// How to open a store.
#[derive(Debug, Clone)]
pub struct StoreOptions {
    /// Force an engine; `None` runs [`choose_backend`] on the path and
    /// [`BACKEND_ENV`].
    pub backend: Option<BackendKind>,
    /// Times to retry a redb file another process holds, 15 ms apart.
    /// FerroStore is multi-process safe and ignores it.
    pub retries: u32,
    /// What a ferro selection beside a lone `.redb` does. Default
    /// [`Unmigrated::Migrate`].
    pub unmigrated: Unmigrated,
}

impl Default for StoreOptions {
    fn default() -> Self {
        Self {
            backend: None,
            retries: 200,
            unmigrated: Unmigrated::Migrate,
        }
    }
}

/// Open the store named `path` with the default options.
pub fn open_store(path: &Path) -> Result<AnyStore, StoreError> {
    open_store_with(path, &StoreOptions::default())
}

/// Open the store named `path` only to look at it (to ask [`AnyStore::owns`]):
/// a lone `.redb` under a ferro selection is read as redb, never migrated, and
/// is opened read-only, so writing through the returned store fails. Open the
/// store with [`open_store`] to work on it. See [`Unmigrated::UseRedb`] and
/// [`store_exists`].
pub fn open_store_for_scan(path: &Path) -> Result<AnyStore, StoreError> {
    open_store_with(
        path,
        &StoreOptions {
            unmigrated: Unmigrated::UseRedb,
            ..StoreOptions::default()
        },
    )
}

/// Open the store named `path`.
pub fn open_store_with(path: &Path, options: &StoreOptions) -> Result<AnyStore, StoreError> {
    open_store_sink(path, options, &warn)
}

/// [`open_store_with`] with the stderr lines it prints (warnings, the
/// migration notice) handed to `sink` instead, so a test can read them.
pub(crate) fn open_store_sink(
    path: &Path,
    options: &StoreOptions,
    sink: &dyn Fn(&str),
) -> Result<AnyStore, StoreError> {
    open_store_sink_env(
        path,
        options,
        std::env::var(BACKEND_ENV).ok().as_deref(),
        sink,
    )
}

/// [`open_store_sink`] with the value of [`BACKEND_ENV`] passed in, so a test
/// can name the environment without changing the process's.
pub(crate) fn open_store_sink_env(
    path: &Path,
    options: &StoreOptions,
    env: Option<&str>,
    sink: &dyn Fn(&str),
) -> Result<AnyStore, StoreError> {
    let kind = match options.backend {
        Some(kind) => kind,
        None => choose_backend(path, env)?,
    };
    let physical = physical_path(path, kind);
    let (backend_store, label) = match kind {
        BackendKind::Redb => {
            warn_if_fls_beside(&physical, sink);
            (
                Backend::Redb(RedbStore::open_with_retries(&physical, options.retries)?),
                physical.display().to_string(),
            )
        }
        BackendKind::Ferro => open_ferro(path, &physical, options, sink)?,
    };
    // The file of a disk FerroStore, as a path and not as the lossy label.
    let file = matches!(backend_store, Backend::Ferro(_)).then_some(physical);
    Ok(AnyStore {
        backend: backend_store,
        label,
        file,
    })
}

/// Whether `err` is redb saying another holder has the file: the case an open
/// retries, and the case a migration cannot run in. (FerroStore is
/// multi-process safe and never reports it.)
pub fn is_redb_busy(err: &StoreError) -> bool {
    let StoreError::Unreachable { cause, .. } = err else {
        return false;
    };
    cause.contains("Database already open") || cause.contains("Cannot acquire lock")
}

/// One line on stderr; a closed stderr is not a panic.
fn warn(text: &str) {
    use std::io::Write;
    let _ = writeln!(std::io::stderr(), "{text}");
}

// ---- what this process remembers --------------------------------------------

/// What this process has already said or learned about a store, so a retry
/// loop (the hub opens a store up to 200 times) neither repeats a full
/// copy-and-verify nor repeats a warning.
#[derive(Default)]
struct Memory {
    /// Stores whose migration ran and did not verify: they stay on redb for the
    /// life of the process, and are not copied again.
    unclean: HashSet<PathBuf>,
    /// Warnings already printed, by kind and store.
    said: HashSet<(&'static str, PathBuf)>,
}

fn memory() -> std::sync::MutexGuard<'static, Memory> {
    static MEMORY: OnceLock<Mutex<Memory>> = OnceLock::new();
    MEMORY
        .get_or_init(|| Mutex::new(Memory::default()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// The store named by the logical `path`, as the process memory keys it: the
/// identity of its `.fls` (symlinks and an aliased directory resolved).
fn memory_key(path: &Path) -> PathBuf {
    store_identity_with(&physical_path(path, BackendKind::Ferro), None)
}

/// Whether this is the first time this process says `what` about `key`.
fn first_time(what: &'static str, key: &Path) -> bool {
    memory().said.insert((what, key.to_path_buf()))
}

/// Forget that this process saw the store named `path` fall back to redb, and
/// every warning it printed about it. `fl store migrate` calls it: an explicit
/// migration ignores what an earlier automatic one learned.
pub fn forget_fallback(path: &Path) {
    let key = memory_key(path);
    let mut m = memory();
    m.unclean.remove(&key);
    m.said.retain(|(_, k)| *k != key);
}

/// Warn, once per process per store, that a `.fls` sits beside a `.redb` that
/// is being opened with the redb engine. After a migration the redb is kept
/// read-only; if it was made writable again and written, the `.fls` no longer
/// matches it. When the `.fls` was written after the `.redb`, the warning says
/// the `.fls` is newer: this process is then working on the older copy, and
/// any process on the ferro engine still works on the newer one.
fn warn_if_fls_beside(redb: &Path, sink: &dyn Fn(&str)) {
    let fls = redb.with_extension(FERRO_EXTENSION);
    if fls == redb || std::fs::symlink_metadata(&fls).is_err() {
        return;
    }
    if first_time("fls-beside-redb", &memory_key(redb)) {
        let fls_is_newer = match (
            std::fs::metadata(&fls).and_then(|m| m.modified()),
            std::fs::metadata(redb).and_then(|m| m.modified()),
        ) {
            (Ok(fls_time), Ok(redb_time)) => fls_time > redb_time,
            _ => false,
        };
        let differ = if fls_is_newer {
            "the .fls is NEWER than the .redb, so the two differ: writes made on the ferro \
             engine are not in the redb"
        } else {
            "if the redb was made writable again after a migration, the two may differ"
        };
        sink(&format!(
            "warning: {} exists beside {}, which is opened with the redb engine: {differ}; run \
             `fl store verify` to compare them",
            fls.display(),
            redb.display()
        ));
    }
}

// ---- the migration lock -----------------------------------------------------

/// The advisory lock that serialises every ferro-engine process that meets a
/// lone `.redb`: `<physical .fls path>.migrate.lock` (an exclusive `flock`).
/// Whoever holds it is the only ferro-engine process that may probe, migrate
/// or open the `.redb`, so none can take the `.redb` in the instant a
/// migration has finished copying it and not yet linked the `.fls`.
///
/// The (empty, 0600) lock file is left in place: removing it would let a
/// waiter and a newcomer lock two different files. It is not a store file:
/// `store_exists`, `fl store status`, and the hub's per-project scan (which
/// matches `*.redb` / `*.fls` exactly) do not see it. A process on the redb
/// engine does not take it.
pub struct MigrationLock {
    _file: std::fs::File,
}

impl MigrationLock {
    /// The lock file of the store whose `.fls` is `physical`.
    pub fn path_for(physical: &Path) -> PathBuf {
        let mut name = physical.as_os_str().to_os_string();
        name.push(".migrate.lock");
        PathBuf::from(name)
    }

    /// Take the lock for the store whose `.fls` is `physical`, creating the lock
    /// file if need be, and wait about `wait` for its holder (a non-zero wait can
    /// overshoot by up to one poll interval, 25 ms; `Duration::ZERO` is exactly
    /// one attempt). The lock is held until the returned value is dropped.
    ///
    /// For a caller outside this module that creates a `.fls` (the
    /// `ferrostore` binary's `put` and `serve`): it excludes a migration of the
    /// same store exactly as the engine's own open does. A holder that stays
    /// past `wait` gives `ErrorKind::TimedOut`; a symlink at the lock path
    /// gives a hard error (`O_NOFOLLOW`). The lock file is left in place.
    pub fn acquire_within(physical: &Path, wait: std::time::Duration) -> std::io::Result<Self> {
        match Self::acquire_for(physical, true, wait)? {
            Some(lock) => Ok(lock),
            // `create` is true, so the file is made and there is always a lock.
            None => Err(std::io::Error::other(
                "the migration lock file was not created",
            )),
        }
    }

    /// Take the lock, waiting up to [`LOCK_WAIT`] for its holder. With `create`
    /// false, a lock file that does not exist is not created: `Ok(None)`, and
    /// nobody can be holding it.
    fn acquire(physical: &Path, create: bool) -> std::io::Result<Option<Self>> {
        Self::acquire_for(physical, create, LOCK_WAIT)
    }

    /// [`acquire`](Self::acquire) with its own deadline. The lock is polled
    /// (`try_lock`), not waited on, so a holder that hangs or is stopped cannot
    /// block every ferro process for ever: after `wait` the error is
    /// `TimedOut`, naming the lock file.
    fn acquire_for(
        physical: &Path,
        create: bool,
        wait: std::time::Duration,
    ) -> std::io::Result<Option<Self>> {
        use std::fs::TryLockError;
        use std::os::unix::fs::OpenOptionsExt;
        let path = Self::path_for(physical);
        let mut options = std::fs::OpenOptions::new();
        // O_NOFOLLOW: a symlink at the lock path (live or dangling) is refused
        // with ELOOP, a hard error, instead of being followed by O_CREAT to lock
        // or create whatever it points at.
        options
            .read(true)
            .write(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW);
        if create {
            options.create(true).truncate(false);
        }
        let file = loop {
            match options.open(&path) {
                Ok(file) => break file,
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(err) if !create && err.kind() == std::io::ErrorKind::NotFound => {
                    return Ok(None);
                }
                Err(err) => return Err(err),
            }
        };
        let deadline = std::time::Instant::now() + wait;
        loop {
            match file.try_lock() {
                Ok(()) => return Ok(Some(Self { _file: file })),
                Err(TryLockError::Error(err)) if err.kind() == std::io::ErrorKind::Interrupted => {}
                Err(TryLockError::Error(err)) => return Err(err),
                Err(TryLockError::WouldBlock) => {
                    if std::time::Instant::now() >= deadline {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::TimedOut,
                            format!(
                                "another process is migrating this store (or holds its lock): \
                                 {} stayed locked for {} s",
                                path.display(),
                                wait.as_secs_f32()
                            ),
                        ));
                    }
                    std::thread::sleep(LOCK_POLL);
                }
            }
        }
    }
}

/// How long a process waits for the [`MigrationLock`] before it gives up with an
/// error. A migration of a large store takes seconds; this is long past that.
const LOCK_WAIT: std::time::Duration = std::time::Duration::from_secs(60);

/// How often a waiting process tries the [`MigrationLock`] again.
const LOCK_POLL: std::time::Duration = std::time::Duration::from_millis(25);

/// Whether a failure to take the [`MigrationLock`] is one where no other process
/// can be holding it either: the user may not write the directory or the lock
/// file (`PermissionDenied`), the filesystem is read-only, or the directory is
/// gone. Only then may the redb be opened without the lock. Any other failure
/// (`ENOLCK`, an unexpected I/O error, a timeout: someone IS holding it) says
/// nothing about whether another process holds the lock, so it is an error.
fn lock_error_is_benign(err: &std::io::Error) -> bool {
    use std::io::ErrorKind;
    const EROFS: i32 = 30;
    matches!(
        err.kind(),
        ErrorKind::PermissionDenied | ErrorKind::ReadOnlyFilesystem | ErrorKind::NotFound
    ) || err.raw_os_error() == Some(EROFS)
}

/// The error for a lock that could not be taken and may not be skipped.
fn lock_failed(physical: &Path, err: &std::io::Error) -> StoreError {
    backend(format!(
        "could not take the migration lock {}: {err}",
        MigrationLock::path_for(physical).display()
    ))
}

/// Migrate the redb store `redb` into the new `fls` under the [`MigrationLock`]
/// (what `fl store migrate` runs; [`crate::migrate::migrate_verified`] itself
/// takes no lock). Forgets an earlier automatic fallback of this store.
pub fn migrate_locked(redb: &Path, fls: &Path) -> Result<crate::migrate::ParityReport, StoreError> {
    let _lock = MigrationLock::acquire(fls, true)
        .map_err(|err| {
            backend(format!(
                "could not take the migration lock {}: {err}",
                MigrationLock::path_for(fls).display()
            ))
        })?
        .expect("created");
    forget_fallback(fls);
    crate::migrate::migrate_verified(redb, fls)
}

// ---- opening a ferro selection ----------------------------------------------

/// Whether `path` names anything, a dangling symlink included.
fn present(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok()
}

fn open_fls(physical: &Path) -> Result<(Backend, String), StoreError> {
    Ok((
        Backend::Ferro(Box::new(FerroStore::open(physical)?)),
        physical.display().to_string(),
    ))
}

/// A redb open failed with `err`: if the `.fls` exists now (another process
/// migrated, and left the `.redb` read-only), the `.fls` is the store.
fn fls_if_present(physical: &Path, err: StoreError) -> Result<(Backend, String), StoreError> {
    if present(physical) {
        open_fls(physical)
    } else {
        Err(err)
    }
}

/// How [`redb_engine`] is used.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Access {
    /// The store a command works on: opened for writing, and the
    /// [`MigrationLock`] file created if need be.
    Use,
    /// A store only asked `owns`: opened read-only first (the file is not
    /// rewritten, not even its mtime) and, if that fails for a non-busy reason,
    /// with the ordinary open; the lock is taken only if its file exists already.
    Scan,
}

/// Open the redb at `path`: for [`Access::Scan`] read-only first, falling back
/// to today's ordinary (writable) open only when the read-only open fails for a
/// non-busy reason (the file needs repair, or is not initialised yet), so a scan
/// never sees less than an ordinary open does. That fallback is an ordinary
/// open: it can repair a crash-damaged redb or initialise an empty file. A busy
/// redb is reported, not opened the other way.
fn open_redb_engine(path: &Path, retries: u32, access: Access) -> Result<RedbStore, StoreError> {
    if access == Access::Scan {
        match RedbStore::open_read_only(path, retries) {
            Ok(store) => return Ok(store),
            Err(err) if is_redb_busy(&err) => return Err(err),
            Err(_) => {}
        }
    }
    RedbStore::open_with_retries(path, retries)
}

/// Open the lone `.redb` at `path` with the redb engine, in a way that cannot
/// split the store: the redb is taken only while holding the [`MigrationLock`]
/// and with no `.fls` there. So it is never taken while a migration (which
/// holds the lock until the `.fls` is linked and the redb made read-only) is
/// running; a `.fls` that exists once the lock is held is opened instead. Waits
/// up to `retries` times, 15 ms apart, for a process on the redb engine to
/// release the file, holding the lock only for each try. `held` is a lock the
/// caller already has.
fn redb_engine(
    path: &Path,
    physical: &Path,
    retries: u32,
    mut held: Option<MigrationLock>,
    access: Access,
) -> Result<(Backend, String), StoreError> {
    let label = path.display().to_string();
    let mut attempts = 0;
    loop {
        let lock = match held.take() {
            Some(lock) => Some(lock),
            // A lock that cannot be taken because no one else can hold it
            // either (a directory or file this user may not write, a read-only
            // filesystem) is treated as absent: the same directory refuses a
            // migration. Any other failure is an error, never an unlocked open.
            None => match MigrationLock::acquire(physical, access == Access::Use) {
                Ok(lock) => lock,
                Err(err) if lock_error_is_benign(&err) => None,
                Err(err) => return Err(lock_failed(physical, &err)),
            },
        };
        let Some(lock) = lock else {
            return match open_redb_engine(path, retries, access) {
                Ok(store) => Ok((Backend::Redb(store), label)),
                Err(err) => fls_if_present(physical, err),
            };
        };
        if present(physical) {
            drop(lock);
            return open_fls(physical);
        }
        match open_redb_engine(path, 0, access) {
            Ok(store) => return Ok((Backend::Redb(store), label)),
            Err(err) if is_redb_busy(&err) && attempts < retries => {
                drop(lock);
                attempts += 1;
                std::thread::sleep(std::time::Duration::from_millis(15));
            }
            Err(err) => return fls_if_present(physical, err),
        }
    }
}

/// Open the ferro store at `physical`, the file of the logical `path`.
///
/// A lone `.redb` beside it (`physical` absent, `path` a different file that
/// exists) is handled as `options.unmigrated` says. `Migrate` runs the
/// verified migration under the [`MigrationLock`]:
///
/// 1. the lock is taken (waiting for any other process that holds it);
/// 2. a `.fls` that exists now is opened (someone else migrated);
/// 3. the redb is probed, once: held by a process outside this lock (a hub still
///    on redb) -> the store opens on the redb engine instead, and no `.fls` is
///    made, so that every process that cannot migrate agrees on one engine;
/// 4. otherwise `migrate_verified` runs; clean -> the `.fls` is opened, after
///    the lock is released, which is after the `.fls` is linked and the redb
///    made read-only; not clean -> redb engine, and this process does not copy
///    that store again.
///
/// A held redb is not remembered (the holder may stop), only its warning.
///
/// Limit: the lock orders ferro-engine processes. A process on the redb
/// engine (`FL_STORE_BACKEND=redb`) does not take it, but it takes the redb's
/// own file lock: the migration holds that across the chmod to 0400, so such a
/// process cannot write the redb between the copy and the chmod. The migration
/// lock additionally orders ferro-engine processes before the redb is opened.
fn open_ferro(
    path: &Path,
    physical: &Path,
    options: &StoreOptions,
    sink: &dyn Fn(&str),
) -> Result<(Backend, String), StoreError> {
    let lone_redb = !present(physical) && path != physical && path.exists();
    if !lone_redb {
        return open_fls(physical);
    }
    if options.unmigrated == Unmigrated::Refuse || !path.is_file() {
        return Err(backend(format!(
            "{} holds a redb store and {} does not exist yet: migrate it first \
             (`fl store migrate`), or set {BACKEND_ENV}=redb to keep using the .redb file",
            path.display(),
            physical.display()
        )));
    }
    if options.unmigrated == Unmigrated::UseRedb {
        return redb_engine(path, physical, options.retries, None, Access::Scan);
    }

    let key = memory_key(path);
    let stay = |held: Option<MigrationLock>| {
        redb_engine(path, physical, options.retries, held, Access::Use)
    };
    if memory().unclean.contains(&key) {
        return stay(None);
    }
    let lock = match MigrationLock::acquire(physical, true) {
        Ok(lock) => lock.expect("created"),
        // A lock that cannot be taken, and may not be skipped: an error.
        Err(err) if !lock_error_is_benign(&err) => return Err(lock_failed(physical, &err)),
        Err(err) => {
            // Without the lock a migration could split the store: do not run one.
            if first_time("no-lock", &key) {
                sink(&format!(
                    "warning: {} stays on redb: could not take the migration lock {}: {err}; \
                     no {} was created",
                    path.display(),
                    MigrationLock::path_for(physical).display(),
                    physical.display()
                ));
            }
            return redb_engine_unlocked(path, physical, options.retries);
        }
    };
    // Another thread of this process may have run the migration, and recorded
    // that it did not verify, while this one waited for the lock.
    if memory().unclean.contains(&key) {
        return stay(Some(lock));
    }
    let busy = |lock: MigrationLock| {
        if first_time("busy", &key) {
            sink(&format!(
                "warning: {} stays on redb: another process still holds the redb store, so it \
                 cannot be migrated yet; no {} was created",
                path.display(),
                physical.display()
            ));
        }
        stay(Some(lock))
    };

    // Someone migrated while this process waited for the lock.
    if present(physical) {
        drop(lock);
        return open_fls(physical);
    }
    // Ask once: a process still holding the redb must not wait out a
    // migration's own 3 s of retries. (Dropped at once; the migration opens it
    // again, with this lock still held.)
    match RedbStore::open_with_retries(path, 0) {
        Ok(store) => drop(store),
        Err(err) if is_redb_busy(&err) => return busy(lock),
        Err(err) => {
            drop(lock);
            return fls_if_present(physical, err);
        }
    }
    #[cfg(test)]
    tests::note_migration_attempt(&key);
    match crate::migrate::migrate_verified(path, physical) {
        Ok(report) if report.is_clean() => {
            drop(lock);
            sink(&format!(
                "migrated {} -> {}; the redb is kept read-only (restore with chmod u+w and \
                 {BACKEND_ENV}=redb)",
                path.display(),
                physical.display()
            ));
            open_fls(physical)
        }
        Ok(report) => {
            let why = match report.mismatches().into_iter().next() {
                Some(t) => format!(
                    "the migration did not verify (first mismatch: `{}`, redb {} rows, ferro {} \
                     rows); run `fl store migrate` for the full report",
                    t.table, t.redb_rows, t.ferro_rows
                ),
                None => "the migration did not verify".to_string(),
            };
            memory().unclean.insert(key.clone());
            if first_time("unclean", &key) {
                sink(&format!(
                    "warning: {} stays on redb: {why}; no {} was created",
                    path.display(),
                    physical.display()
                ));
            }
            stay(Some(lock))
        }
        Err(err) if is_redb_busy(&err) => busy(lock),
        // Another process migrated first (one that took no lock).
        Err(err) => {
            drop(lock);
            fls_if_present(physical, err)
        }
    }
}

/// [`redb_engine`] with no lock at all: the migration lock could not be taken.
fn redb_engine_unlocked(
    path: &Path,
    physical: &Path,
    retries: u32,
) -> Result<(Backend, String), StoreError> {
    match RedbStore::open_with_retries(path, retries) {
        Ok(store) => Ok((Backend::Redb(store), path.display().to_string())),
        Err(err) => fls_if_present(physical, err),
    }
}

/// Open the redb store at `path` whatever the environment says. For tests and
/// for tools that must read a `.redb` file regardless of `FL_STORE_BACKEND`.
pub fn open_redb(path: &Path) -> Result<AnyStore, StoreError> {
    open_store_with(
        path,
        &StoreOptions {
            backend: Some(BackendKind::Redb),
            ..StoreOptions::default()
        },
    )
}

/// Whether the store named by the logical `path` exists under the engine that
/// [`choose_backend`] selects for it (the process environment's
/// [`BACKEND_ENV`] included): the physical file of that engine, or, under a
/// ferro selection, the lone `.redb` that [`open_store`] migrates (or, when it
/// cannot, keeps opening as redb). Opening either never creates an empty
/// store. An unusable environment value counts as "check the path itself".
///
/// The rule this gives callers: a lone `.redb` counts as existing under ferro,
/// so a loop that visits every configured store with `store_exists` and opens
/// each one only to ask `owns` must open it with [`open_store_for_scan`]
/// ([`Unmigrated::UseRedb`]), never [`open_store`]: `open_store` migrates, and
/// would migrate and `chmod` every project store on a read-only lookup. The
/// store a command works on is opened with [`open_store`].
pub fn store_exists(path: &Path) -> bool {
    store_exists_with(path, std::env::var(BACKEND_ENV).ok().as_deref())
}

fn store_exists_with(path: &Path, env: Option<&str>) -> bool {
    match choose_backend(path, env) {
        Ok(kind) => {
            let physical = physical_path(path, kind);
            physical.exists() || (kind == BackendKind::Ferro && path != physical && path.is_file())
        }
        Err(_) => path.exists(),
    }
}

/// The identity of the store named `path` under the process environment: the
/// physical file of the selected engine, with symlinks resolved. Two paths
/// name the same store exactly when their identities are equal, so a
/// configured `x.redb` and an explicit `x.fls` agree under FerroStore, and a
/// symlinked alias agrees with its target. A file that does not exist yet has
/// its directory resolved instead. An unusable engine selection falls back to
/// the path itself, as [`store_exists`] does.
pub fn store_identity(path: &Path) -> PathBuf {
    store_identity_with(path, std::env::var(BACKEND_ENV).ok().as_deref())
}

fn store_identity_with(path: &Path, env: Option<&str>) -> PathBuf {
    let physical = match choose_backend(path, env) {
        Ok(kind) => physical_path(path, kind),
        Err(_) => path.to_path_buf(),
    };
    if let Ok(resolved) = physical.canonicalize() {
        return resolved;
    }
    // Absent file: resolve the directory it would be created in.
    match (physical.parent(), physical.file_name()) {
        (Some(dir), Some(name)) => {
            let dir = if dir.as_os_str().is_empty() {
                Path::new(".")
            } else {
                dir
            };
            match dir.canonicalize() {
                Ok(resolved) => resolved.join(name),
                Err(_) => physical,
            }
        }
        _ => physical,
    }
}

enum Backend {
    Ferro(Box<FerroStore>),
    Redb(RedbStore),
}

/// The size accounting of a store file, engine-neutral (see
/// [`AnyStore::stats`]).
///
/// `file_bytes` is the file's length under either engine. `live_bytes` and
/// `dead_bytes` are `Some` only for FerroStore, which knows which bytes its
/// live superblock names; they are `None` for redb, which does not report
/// them, so a redb store is never shown as having zero dead bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnyStats {
    pub engine: BackendKind,
    pub file_bytes: u64,
    /// Bytes the live superblock names. `None` on redb.
    pub live_bytes: Option<u64>,
    /// Bytes only a compaction gives back. `None` on redb.
    pub dead_bytes: Option<u64>,
}

/// Why compaction does nothing on redb. The one wording, shared by
/// [`AnyStore::compact`]'s refusal and `fl store compact`'s report.
pub const COMPACT_FERRO_ONLY: &str =
    "compaction applies to the FerroStore (.fls) engine only; redb is not compacted";

/// The size accounting of the FerroStore file `path`, read without opening a
/// store (see [`FerroStore::stats_at`]): read-only, no lock, nothing created
/// or initialised, so it is safe beside a live store and on a 0400 file. The
/// three byte counts come from one open file and agree with each other.
pub fn ferro_stats_at(path: &Path) -> Result<AnyStats, StoreError> {
    let st = FerroStore::stats_at(path)?;
    Ok(AnyStats {
        engine: BackendKind::Ferro,
        file_bytes: st.file_bytes,
        live_bytes: Some(st.live_bytes),
        dead_bytes: Some(st.dead_bytes),
    })
}

/// A store on either engine. See the [module documentation](self).
pub struct AnyStore {
    backend: Backend,
    label: String,
    /// The real path of a FerroStore opened from a file, kept as a `Path`
    /// because [`label`](Self::label) is lossy for a name that is not UTF-8.
    file: Option<PathBuf>,
}

impl From<RedbStore> for AnyStore {
    fn from(store: RedbStore) -> Self {
        let label = store.label().to_string();
        Self {
            backend: Backend::Redb(store),
            label,
            file: None,
        }
    }
}

impl From<FerroStore> for AnyStore {
    fn from(store: FerroStore) -> Self {
        let label = store.label().to_string();
        Self {
            backend: Backend::Ferro(Box::new(store)),
            label,
            file: None,
        }
    }
}

impl AnyStore {
    /// An in-memory FerroStore, for tests.
    pub fn in_memory() -> Self {
        FerroStore::in_memory().into()
    }

    pub fn kind(&self) -> BackendKind {
        match self.backend {
            Backend::Redb(_) => BackendKind::Redb,
            Backend::Ferro(_) => BackendKind::Ferro,
        }
    }

    /// The store's name in messages: the file it was opened from.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The file a FerroStore was opened from, as the path it was opened with.
    /// `None` for a redb store and for a FerroStore with no file of its own
    /// (in memory, or converted with `From<FerroStore>`).
    pub fn ferro_file(&self) -> Option<&Path> {
        self.file.as_deref()
    }

    pub fn as_redb(&self) -> Option<&RedbStore> {
        match &self.backend {
            Backend::Redb(s) => Some(s),
            Backend::Ferro(_) => None,
        }
    }

    pub fn as_ferro(&self) -> Option<&FerroStore> {
        match &self.backend {
            Backend::Ferro(s) => Some(s),
            Backend::Redb(_) => None,
        }
    }

    /// Whether this store holds `id`. An I/O failure is an error, never `false`.
    pub fn owns(&self, id: &Iri) -> Result<bool, StoreError> {
        match &self.backend {
            Backend::Ferro(s) => s.owns(id),
            Backend::Redb(s) => s.owns(id),
        }
    }

    /// The hash of the manifest `project` was imported from, or `None` if this
    /// store authors it.
    pub fn imported_hash(&self, project: &ProjectId) -> Result<Option<String>, StoreError> {
        match &self.backend {
            Backend::Ferro(s) => s.imported_hash(project),
            Backend::Redb(s) => s.imported_hash(project),
        }
    }

    /// `Imported`, naming `action`, when this store imported `project`.
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

    /// Whether any project in this store has a routing map.
    pub fn holds_routing(&self) -> Result<bool, StoreError> {
        match &self.backend {
            Backend::Ferro(s) => s.holds_routing(),
            Backend::Redb(s) => s.holds_routing(),
        }
    }

    /// Whether this store records any ledger root.
    pub fn holds_a_ledger_root(&self) -> Result<bool, StoreError> {
        match &self.backend {
            Backend::Ferro(s) => s.holds_a_ledger_root(),
            Backend::Redb(s) => s.holds_a_ledger_root(),
        }
    }

    /// Write `project`'s routing map.
    pub fn set_routes(&self, project: &ProjectId, map: &RoutingMap) -> Result<(), StoreError> {
        match &self.backend {
            Backend::Ferro(s) => s.set_routes(project, map),
            Backend::Redb(s) => s.set_routes(project, map),
        }
    }

    /// Rewrite a FerroStore as one snapshot and an empty WAL, giving back
    /// every extent earlier checkpoints left dead (see
    /// [`FerroStore::compact`]).
    ///
    /// Compaction is a FerroStore operation. A redb store is refused with a
    /// clear error rather than reported as done: redb has no dead extents of
    /// this kind to give back, so an `Ok(())` would claim a file was
    /// compacted that was not touched.
    pub fn compact(&self) -> Result<(), StoreError> {
        match &self.backend {
            Backend::Ferro(s) => s.compact(),
            Backend::Redb(_) => Err(backend(format!("{}: {COMPACT_FERRO_ONLY}", self.label))),
        }
    }

    /// The size accounting of the store file (see [`AnyStats`]). FerroStore
    /// reports live and dead bytes; redb reports the file's length only, and
    /// `None` for both, because redb does not expose them.
    pub fn stats(&self) -> Result<AnyStats, StoreError> {
        match &self.backend {
            Backend::Ferro(s) => {
                let st = s.stats()?;
                Ok(AnyStats {
                    engine: BackendKind::Ferro,
                    file_bytes: st.file_bytes,
                    live_bytes: Some(st.live_bytes),
                    dead_bytes: Some(st.dead_bytes),
                })
            }
            Backend::Redb(_) => {
                let file_bytes = std::fs::metadata(&self.label)
                    .map_err(|e| {
                        backend(format!("could not read the size of {}: {e}", self.label))
                    })?
                    .len();
                Ok(AnyStats {
                    engine: BackendKind::Redb,
                    file_bytes,
                    live_bytes: None,
                    dead_bytes: None,
                })
            }
        }
    }

    /// The format version the store file records.
    pub fn format_version(&self) -> Result<Option<u64>, StoreError> {
        match &self.backend {
            Backend::Ferro(s) => s.format_version(),
            Backend::Redb(s) => s.format_version(),
        }
    }

    /// Export `project`, which this store must author.
    pub fn export_manifest(
        &self,
        project: &ProjectId,
        commit: &str,
        exported_at_unix: u64,
        repository_node_id: Option<&str>,
    ) -> Result<Manifest, ManifestError> {
        match &self.backend {
            Backend::Ferro(s) => {
                s.export_manifest(project, commit, exported_at_unix, repository_node_id)
            }
            Backend::Redb(s) => {
                s.export_manifest(project, commit, exported_at_unix, repository_node_id)
            }
        }
    }

    /// Write the manifest's project, gates and transitions, atomically.
    pub fn import_manifest(&self, m: &Manifest, root: &str) -> Result<ImportReport, ManifestError> {
        match &self.backend {
            Backend::Ferro(s) => s.import_manifest(m, root),
            Backend::Redb(s) => s.import_manifest(m, root),
        }
    }
}

/// Implement each store trait on [`AnyStore`] by calling the engine's own
/// implementation (always by trait path, so an inherent method of the same
/// name can never be picked instead). Listing every method, defaulted ones
/// included, keeps an engine's override of a default in force.
macro_rules! delegate {
    ($($trait:ident {
        $(fn $name:ident(&self $(, $arg:ident : $ty:ty)* $(,)?) -> $ret:ty;)*
    })*) => {$(
        impl $trait for AnyStore {
            $(fn $name(&self $(, $arg: $ty)*) -> $ret {
                match &self.backend {
                    Backend::Ferro(s) => $trait::$name(&**s $(, $arg)*),
                    Backend::Redb(s) => $trait::$name(s $(, $arg)*),
                }
            })*
        }
    )*};
}

delegate! {
    Catalog {
        fn add_project(&self, root: &str) -> Result<ProjectId, StoreError>;
        fn get_project(&self, id: &ProjectId) -> Result<Option<Project>, StoreError>;
        fn list_projects(&self) -> Result<Vec<Project>, StoreError>;
        fn add_gate(&self, project: &ProjectId, name: &str, kind: GateKind, selector: Selector, min_population: u64, authored_at_commit: &str, authored_by: &str) -> Result<GateId, StoreError>;
        fn get_gate(&self, id: &GateId) -> Result<Option<GateDef>, StoreError>;
        fn list_gates(&self, project: &ProjectId) -> Result<Vec<GateDef>, StoreError>;
        fn update_gate(&self, def: &GateDef) -> Result<(), StoreError>;
        fn add_transition(&self, t: Transition) -> Result<(), StoreError>;
        fn get_transition(&self, project: &ProjectId, name: &str) -> Result<Option<Transition>, StoreError>;
        fn list_transitions(&self, project: &ProjectId) -> Result<Vec<Transition>, StoreError>;
        fn kind_of(&self, id: &Iri) -> Result<Kind, StoreError>;
    }
    Routes {
        fn routes(&self, project: &ProjectId) -> Result<Option<RoutingMap>, StoreError>;
    }
    Tracker {
        fn add_record(&self, project: &ProjectId, title: &str) -> Result<RecordId, StoreError>;
        fn add_record_with_area(&self, project: &ProjectId, title: &str, area: Option<&str>) -> Result<RecordId, StoreError>;
        fn get_record(&self, id: &RecordId) -> Result<Option<Record>, StoreError>;
        fn list_records(&self, project: &ProjectId) -> Result<Vec<Record>, StoreError>;
        fn set_record_state(&self, id: &RecordId, state: State) -> Result<(), StoreError>;
        fn add_finding(&self, finding: Finding) -> Result<FindingId, StoreError>;
        fn add_finding_checked(&self, finding: Finding, record: ForeignRecord) -> Result<FindingId, StoreError>;
        fn get_finding(&self, id: &FindingId) -> Result<Option<Finding>, StoreError>;
        fn update_finding(&self, finding: &Finding) -> Result<(), StoreError>;
        fn list_findings(&self, project: &ProjectId) -> Result<Vec<Finding>, StoreError>;
        fn withdrawals_by(&self, actor: &str) -> Result<u64, StoreError>;
        fn add_alias(&self, primary: &Iri, alias: Iri) -> Result<(), StoreError>;
    }
    Escalations {
        fn mark(&self, id: &Iri, mark: &Mark) -> Result<(), StoreError>;
        fn mark_of(&self, id: &Iri) -> Result<Option<Mark>, StoreError>;
        fn unmark(&self, id: &Iri) -> Result<(), StoreError>;
        fn tombstone(&self, id: &Iri, to: &Iri) -> Result<Tombstone, StoreError>;
        fn tombstone_of(&self, id: &Iri) -> Result<Option<Tombstone>, StoreError>;
    }
    Ledger {
        fn append_gate_run(&self, run: GateRun) -> Result<(), StoreError>;
        fn append_attempt(&self, attempt: Attempt) -> Result<(), StoreError>;
        fn gate_runs(&self, gate: &GateId) -> Result<Vec<GateRun>, StoreError>;
        fn attempts(&self, project: &ProjectId) -> Result<Vec<Attempt>, StoreError>;
        fn flush(&self, _decision: Decision) -> Result<Flushed, StoreError>;
    }
    Outbox {
        fn unpublished(&self, repo: &str, after: &Iri) -> Result<Pending, StoreError>;
        fn is_published(&self, repo: &str, id: &Iri) -> Result<bool, StoreError>;
        fn mark_published(&self, repo: &str, ids: &[Iri]) -> Result<(), StoreError>;
        fn settle(&self, repo: &str, published: &[Iri], set_aside: &[Iri]) -> Result<(), StoreError>;
        fn cutover(&self, repo: &str) -> Result<Option<Iri>, StoreError>;
        fn set_cutover(&self, repo: &str, id: &Iri) -> Result<(), StoreError>;
    }
    Handles {
        fn handle_of(&self, kind: Kind, id: &Iri) -> Result<Option<u64>, StoreError>;
        fn resolve_handle(&self, kind: Kind, handle: u64) -> Result<Option<Iri>, StoreError>;
    }
    Bindings {
        fn bound_node_id(&self, repo: &str) -> Result<Option<String>, StoreError>;
        fn bind_node_id(&self, repo: &str, node_id: &str) -> Result<(), StoreError>;
        fn ledger_root(&self, node_id: &str) -> Result<Option<String>, StoreError>;
        fn set_ledger_root(&self, node_id: &str, commit: &str) -> Result<(), StoreError>;
    }
    LedgerCache {
        fn last_head(&self, repo: &str) -> Result<Option<String>, StoreError>;
        fn cached(&self, repo: &str, path: &str) -> Result<Option<CachedSegment>, StoreError>;
        fn cached_under(&self, repo: &str, dir: &str) -> Result<Vec<(String, CachedSegment)>, StoreError>;
        fn remember(&self, repo: &str, head: &str, segments: &[(String, CachedSegment)]) -> Result<(), StoreError>;
    }
    AgentRegistry {
        fn upsert_area(&self, area: AreaRecord) -> Result<AreaRecord, StoreError>;
        fn get_area(&self, area_key: &str) -> Result<Option<AreaRecord>, StoreError>;
        fn list_areas(&self) -> Result<Vec<AreaRecord>, StoreError>;
        fn register_agent(&self, args: RegisterAgentArgs) -> Result<AgentRecord, StoreError>;
        fn retire_agent(&self, id: &AgentId, at: At, by: &str) -> Result<AgentRecord, StoreError>;
        fn get_agent(&self, id: &AgentId) -> Result<Option<AgentRecord>, StoreError>;
        fn agent_by_name(&self, name_key: &str) -> Result<Option<AgentRecord>, StoreError>;
        fn list_agents(&self, project: Option<&str>, include_retired: bool) -> Result<Vec<AgentRecord>, StoreError>;
        fn registry_events_after(&self, after_seq: u64) -> Result<Vec<(u64, RegistryEvent)>, StoreError>;
        fn import_registry_events(&self, events: &[RegistryEvent]) -> Result<(), StoreError>;
    }
    StateStore {
        fn upsert_state(&self, args: UpsertStateArgs) -> Result<StateEntry, StoreError>;
        fn retire_memory(&self, scope: &StateScope, slug: &str, lifecycle: MemoryLifecycle, superseded_by: Option<StateId>, at: At, by: &str) -> Result<StateEntry, StoreError>;
        fn get_state(&self, id: &StateId) -> Result<Option<StateEntry>, StoreError>;
        fn get_state_by_key(&self, scope: &StateScope, concern: StateConcern, slug: &str) -> Result<Option<StateEntry>, StoreError>;
        fn list_state(&self, scope: Option<&StateScope>, concern: Option<StateConcern>) -> Result<Vec<StateEntry>, StoreError>;
        fn state_revisions(&self, id: &StateId) -> Result<Vec<StateRevision>, StoreError>;
        fn list_conflicts(&self) -> Result<Vec<StateConflict>, StoreError>;
        fn resolve_conflict(&self, id: &StateId, keep: &RevisionId, at: At, by: &str) -> Result<StateEntry, StoreError>;
        fn state_revisions_after(&self, after_seq: u64) -> Result<Vec<(u64, StateRevision)>, StoreError>;
        fn import_state_revisions(&self, revisions: &[StateRevision], merge_at: At) -> Result<(), StoreError>;
        fn sync_cursor(&self, remote: &str) -> Result<Option<StateSyncCursor>, StoreError>;
        fn set_sync_cursor(&self, cursor: StateSyncCursor) -> Result<(), StoreError>;
    }
    MessageStore {
        fn send_wire(&self, args: SendWireArgs) -> Result<(WireMessage, Vec<DeliveryReceipt>), StoreError>;
        fn get_wire_message(&self, id: &MessageId) -> Result<Option<WireMessage>, StoreError>;
        fn list_wire_inbox(&self, filter: &WireFilter) -> Result<Vec<WireInboxItem>, StoreError>;
        fn list_wire_thread(&self, thread_id: &ThreadId) -> Result<Vec<WireMessage>, StoreError>;
        fn list_wire_replies(&self, correlation_id: &MessageId) -> Result<Vec<WireMessage>, StoreError>;
        fn list_wire_receipts(&self, message_id: &MessageId) -> Result<Vec<DeliveryReceipt>, StoreError>;
        fn update_wire_receipt(&self, message_id: &MessageId, recipient: &str, status: DeliveryStatus, increment_attempts: bool, at: At) -> Result<Option<DeliveryReceipt>, StoreError>;
        fn wire_messages_after(&self, after_seq: u64) -> Result<Vec<(u64, WireMessage)>, StoreError>;
        fn wire_receipts_after(&self, after_seq: u64) -> Result<Vec<(u64, DeliveryReceipt)>, StoreError>;
        fn import_wire_batch(&self, messages: &[WireMessage], receipts: &[DeliveryReceipt]) -> Result<WireImportReport, StoreError>;
        fn prune_wire(&self, before: At) -> Result<usize, StoreError>;
    }
    WorkflowStore {
        fn upsert_workflow_def(&self, def: WorkflowDef) -> Result<WorkflowDef, StoreError>;
        fn get_workflow_def(&self, id: &WorkflowId) -> Result<Option<WorkflowDef>, StoreError>;
        fn get_workflow_def_by_slug(&self, project: &ProjectId, slug: &str) -> Result<Option<WorkflowDef>, StoreError>;
        fn list_workflow_defs(&self, project: Option<&ProjectId>) -> Result<Vec<WorkflowDef>, StoreError>;
        fn create_workflow_run(&self, run: WorkflowRun) -> Result<WorkflowRun, StoreError>;
        fn update_workflow_run(&self, run: WorkflowRun) -> Result<WorkflowRun, StoreError>;
        fn get_workflow_run(&self, id: &WorkflowRunId) -> Result<Option<WorkflowRun>, StoreError>;
        fn list_workflow_runs(&self, project: Option<&ProjectId>, workflow_id: Option<&WorkflowId>) -> Result<Vec<WorkflowRun>, StoreError>;
        fn import_workflow_def(&self, def: WorkflowDef) -> Result<(), StoreError>;
        fn import_workflow_run(&self, run: WorkflowRun) -> Result<(), StoreError>;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fl_core::conformance::{Single, SplitOver};
    use fl_core::{
        CommandSpec, HandoffEntry, MessageKind, MessageUrgency, PopulationDelivery,
        SensitivityPatch, StatePayload, WireAddress,
    };
    use tempfile::TempDir;

    /// A store of each engine, with the directory it lives in kept alive.
    fn on(kind: BackendKind) -> (AnyStore, TempDir) {
        let dir = TempDir::new().unwrap();
        let store = open_store_with(
            &dir.path().join("s.redb"),
            &StoreOptions {
                backend: Some(kind),
                ..StoreOptions::default()
            },
        )
        .unwrap();
        (store, dir)
    }

    fn both() -> [BackendKind; 2] {
        [BackendKind::Redb, BackendKind::Ferro]
    }

    #[test]
    fn stats_and_compact_on_ferro_report_dead_bytes_and_give_them_back() {
        let (store, _dir) = on(BackendKind::Ferro);
        let ferro = store.as_ferro().unwrap();
        for round in 0..4 {
            for i in 0..20 {
                ferro
                    .put_doc(
                        "docs",
                        &format!("r{round}-{i}"),
                        &serde_json::json!({"round": round, "i": i}),
                    )
                    .unwrap();
            }
            ferro.checkpoint().unwrap();
        }
        let before = store.stats().unwrap();
        assert_eq!(before.engine, BackendKind::Ferro);
        let (live, dead) = (before.live_bytes.unwrap(), before.dead_bytes.unwrap());
        assert!(dead > 0, "{before:?}");
        assert_eq!(before.file_bytes, live + dead, "{before:?}");

        store.compact().unwrap();

        let after = store.stats().unwrap();
        assert_eq!(after.dead_bytes, Some(0), "{after:?}");
        assert_eq!(Some(after.file_bytes), after.live_bytes);
        assert!(after.file_bytes < before.file_bytes);
        let on_disk = std::fs::metadata(store.label()).unwrap().len();
        assert_eq!(after.file_bytes, on_disk);
    }

    /// Every name in `dir`, sorted.
    fn names(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    }

    #[test]
    fn ferro_stats_at_agrees_with_an_open_handle_and_works_on_a_0400_file() {
        use std::os::unix::fs::PermissionsExt;
        let (store, dir) = on(BackendKind::Ferro);
        let ferro = store.as_ferro().unwrap();
        for round in 0..4 {
            for i in 0..20 {
                ferro
                    .put_doc(
                        "docs",
                        &format!("r{round}-{i}"),
                        &serde_json::json!({"i": i}),
                    )
                    .unwrap();
            }
            ferro.checkpoint().unwrap();
        }
        let path = PathBuf::from(store.label());
        let open = store.stats().unwrap();
        let at = ferro_stats_at(&path).unwrap();
        assert_eq!(at, open);
        assert!(at.dead_bytes.unwrap() > 0);
        assert_eq!(
            at.file_bytes,
            at.live_bytes.unwrap() + at.dead_bytes.unwrap()
        );
        assert_eq!(FerroStore::stats_at(&path).unwrap(), ferro.stats().unwrap());

        let before = names(dir.path());
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o400)).unwrap();
        assert_eq!(ferro_stats_at(&path).unwrap(), open);
        assert_eq!(names(dir.path()), before);
    }

    #[test]
    fn ferro_stats_at_never_initialises_or_writes_a_file_that_is_not_a_store() {
        let dir = TempDir::new().unwrap();
        let cases: [(&str, Vec<u8>); 4] = [
            ("empty.fls", Vec::new()),
            ("short.fls", vec![0u8; 100]),
            ("zeroed.fls", vec![0u8; 8192 + 64]),
            (
                "garbage.fls",
                (0..20_000u32).map(|i| (i % 251) as u8).collect(),
            ),
        ];
        for (name, bytes) in cases {
            let path = dir.path().join(name);
            std::fs::write(&path, &bytes).unwrap();
            let before = names(dir.path());
            let err = ferro_stats_at(&path).unwrap_err().to_string();
            assert!(!err.is_empty());
            assert_eq!(std::fs::read(&path).unwrap(), bytes, "{name} was modified");
            assert_eq!(names(dir.path()), before, "{name} made a sibling");
        }
        let missing = dir.path().join("missing.fls");
        assert!(ferro_stats_at(&missing).is_err());
        assert!(!missing.exists());
    }

    #[test]
    fn compact_leaves_no_lock_or_sibling_file_beside_the_store() {
        let (store, dir) = on(BackendKind::Ferro);
        let ferro = store.as_ferro().unwrap();
        for round in 0..3 {
            ferro
                .put_doc(
                    "docs",
                    &format!("r{round}"),
                    &serde_json::json!({"r": round}),
                )
                .unwrap();
            ferro.checkpoint().unwrap();
        }
        let before = names(dir.path());
        store.compact().unwrap();
        assert_eq!(names(dir.path()), before);
    }

    #[test]
    fn stats_on_redb_is_the_file_length_with_no_live_or_dead_figure() {
        let (store, _dir) = on(BackendKind::Redb);
        let stats = store.stats().unwrap();
        assert_eq!(stats.engine, BackendKind::Redb);
        assert_eq!(
            stats.file_bytes,
            std::fs::metadata(store.label()).unwrap().len()
        );
        assert!(stats.file_bytes > 0);
        assert_eq!((stats.live_bytes, stats.dead_bytes), (None, None));
    }

    #[test]
    fn compact_on_redb_is_refused_in_words_and_leaves_the_file_alone() {
        let (store, _dir) = on(BackendKind::Redb);
        let before = std::fs::read(store.label()).unwrap();
        let err = store.compact().unwrap_err().to_string();
        assert!(err.contains(COMPACT_FERRO_ONLY), "{err}");
        assert_eq!(std::fs::read(store.label()).unwrap(), before);
    }

    #[test]
    fn stats_and_compact_on_an_in_memory_store_are_empty_and_ok() {
        let store = AnyStore::in_memory();
        store.compact().unwrap();
        let stats = store.stats().unwrap();
        assert_eq!(
            (stats.file_bytes, stats.live_bytes, stats.dead_bytes),
            (0, Some(0), Some(0))
        );
    }

    #[test]
    fn any_store_meets_every_store_contract_on_both_engines() {
        for kind in both() {
            let fresh = || on(kind);
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
            fl_core::conformance::workflow_store(fresh);
            fl_core::conformance::escalations(single);
        }
    }

    #[test]
    fn the_default_engine_is_ferrostore_and_redb_is_the_environments_rollback() {
        assert_eq!(DEFAULT_BACKEND, BackendKind::Ferro);
        // No engine named: a `.redb` name opens a FerroStore at the sibling `.fls`.
        let dir = TempDir::new().unwrap();
        let logical = dir.path().join("fl.redb");
        let store = open_store_sink_env(&logical, &StoreOptions::default(), None, &|_| {}).unwrap();
        assert_eq!(store.kind(), BackendKind::Ferro);
        drop(store);
        assert!(dir.path().join("fl.fls").is_file());
        assert!(!logical.exists(), "no redb was created");

        // `FL_STORE_BACKEND=redb` keeps the store on redb, at the path itself.
        let dir = TempDir::new().unwrap();
        let logical = dir.path().join("fl.redb");
        let store =
            open_store_sink_env(&logical, &StoreOptions::default(), Some("redb"), &|_| {}).unwrap();
        assert_eq!(store.kind(), BackendKind::Redb);
        drop(store);
        assert!(logical.is_file());
        assert!(
            !dir.path().join("fl.fls").exists(),
            "no ferro file was created"
        );

        // A name with no extension and an empty value take the default too.
        let dir = TempDir::new().unwrap();
        let bare = dir.path().join("store");
        let store =
            open_store_sink_env(&bare, &StoreOptions::default(), Some(""), &|_| {}).unwrap();
        assert_eq!(store.kind(), BackendKind::Ferro);
        assert!(dir.path().join("store.fls").is_file());
    }

    #[test]
    fn a_fls_path_is_ferrostore_and_other_paths_follow_the_environment() {
        let fls = Path::new("/x/fl.fls");
        let redb = Path::new("/x/fl.redb");
        // An explicit .fls path wins over the environment.
        assert_eq!(choose_backend(fls, None).unwrap(), BackendKind::Ferro);
        assert_eq!(
            choose_backend(fls, Some("redb")).unwrap(),
            BackendKind::Ferro
        );
        assert_eq!(choose_backend(redb, None).unwrap(), DEFAULT_BACKEND);
        assert_eq!(
            choose_backend(redb, Some("ferro")).unwrap(),
            BackendKind::Ferro
        );
        assert_eq!(
            choose_backend(redb, Some(" Redb ")).unwrap(),
            BackendKind::Redb
        );
        assert_eq!(choose_backend(redb, Some("")).unwrap(), DEFAULT_BACKEND);
        let err = choose_backend(redb, Some("sqlite")).unwrap_err();
        assert!(err.to_string().contains("sqlite"), "{err}");
    }

    #[test]
    fn a_store_keeps_one_logical_path_across_engines() {
        let logical = Path::new("/x/fl.redb");
        assert_eq!(physical_path(logical, BackendKind::Redb), logical);
        assert_eq!(
            physical_path(logical, BackendKind::Ferro),
            Path::new("/x/fl.fls")
        );
        assert_eq!(
            physical_path(Path::new("/x/fl.fls"), BackendKind::Ferro),
            Path::new("/x/fl.fls")
        );
    }

    #[test]
    fn store_identity_is_the_logical_path_on_redb() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let redb = root.join("a.redb");
        // Absent: the directory is resolved, the name kept.
        assert_eq!(store_identity_with(&redb, Some("redb")), redb);
        // A sibling .fls is a different store under redb.
        assert_ne!(
            store_identity_with(&redb, Some("redb")),
            store_identity_with(&root.join("a.fls"), None)
        );
        std::fs::write(&redb, b"x").unwrap();
        assert_eq!(store_identity_with(&redb, Some("redb")), redb);
        // With no engine named, the default (FerroStore) maps it to the sibling .fls.
        assert_eq!(
            store_identity_with(&redb, None),
            store_identity_with(&root.join("a.fls"), None)
        );
    }

    #[test]
    fn store_identity_is_the_physical_file_on_ferro() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let redb = root.join("a.redb");
        let fls = root.join("a.fls");
        // Both absent: the raw physical path.
        assert_eq!(store_identity_with(&redb, Some("ferro")), fls);
        assert_eq!(store_identity_with(&fls, None), fls);
        assert_eq!(
            store_identity_with(&redb, Some("ferro")),
            store_identity_with(&fls, None)
        );
        // A stale .redb beside it does not change the answer.
        std::fs::write(&redb, b"stale").unwrap();
        assert_eq!(store_identity_with(&redb, Some("ferro")), fls);
        std::fs::write(&fls, b"x").unwrap();
        assert_eq!(store_identity_with(&redb, Some("ferro")), fls);
    }

    #[cfg(unix)]
    #[test]
    fn store_identity_sees_through_symlinks() {
        let dir = TempDir::new().unwrap();
        let root = dir.path().canonicalize().unwrap();
        let real = root.join("real");
        std::fs::create_dir(&real).unwrap();
        let alias_dir = root.join("alias");
        std::os::unix::fs::symlink(&real, &alias_dir).unwrap();
        // An absent file in an aliased directory.
        assert_eq!(
            store_identity_with(&alias_dir.join("a.redb"), Some("ferro")),
            store_identity_with(&real.join("a.fls"), None)
        );
        // An existing file reached through a file symlink.
        std::fs::write(real.join("a.fls"), b"x").unwrap();
        std::os::unix::fs::symlink(real.join("a.fls"), root.join("link.fls")).unwrap();
        assert_eq!(
            store_identity_with(&root.join("link.fls"), None),
            real.join("a.fls")
        );
        assert_eq!(
            store_identity_with(&alias_dir.join("a.redb"), Some("ferro")),
            store_identity_with(&root.join("link.fls"), None)
        );
        // An unusable engine selection falls back to the path itself.
        assert_eq!(
            store_identity_with(&real.join("b.redb"), Some("sqlite")),
            real.join("b.redb")
        );
    }

    #[test]
    fn each_engine_persists_across_reopen_through_open_store() {
        for kind in both() {
            let dir = TempDir::new().unwrap();
            let path = dir.path().join("s.redb");
            let opts = StoreOptions {
                backend: Some(kind),
                ..StoreOptions::default()
            };
            let project = {
                let store = open_store_with(&path, &opts).unwrap();
                assert_eq!(store.kind(), kind);
                store.add_project("/repo").unwrap()
            };
            let store = open_store_with(&path, &opts).unwrap();
            assert_eq!(store.get_project(&project).unwrap().unwrap().root, "/repo");
            assert!(
                physical_path(&path, kind).exists(),
                "{kind:?} wrote its file"
            );
        }
    }

    #[test]
    fn opening_a_fls_store_beside_an_unmigrated_redb_asks_for_a_migration() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("s.redb");
        drop(RedbStore::open(&path).unwrap());
        let err = open_store_with(
            &path,
            &StoreOptions {
                backend: Some(BackendKind::Ferro),
                unmigrated: Unmigrated::Refuse,
                ..StoreOptions::default()
            },
        )
        .err()
        .expect("must refuse");
        assert!(err.to_string().contains("fl store migrate"), "{err}");
        assert!(
            err.to_string().contains("FL_STORE_BACKEND=redb"),
            "names the remedy: {err}"
        );
        assert!(!dir.path().join("s.fls").exists(), "nothing was created");
    }

    fn handoff(summary: &str) -> StatePayload {
        StatePayload::Handoff(HandoffEntry {
            summary: summary.into(),
            shipped: vec![],
            in_flight: vec![],
            blockers: vec![],
            next_steps: vec![],
            notes: None,
        })
    }

    /// A project, a gate, a record, a state entry and a wire message, written
    /// through the traits.
    fn populate(store: &RedbStore) -> (ProjectId, GateId, RecordId) {
        let project = store.add_project("/work/app").unwrap();
        let gate = store
            .add_gate(
                &project,
                "fmt",
                GateKind::Command(CommandSpec {
                    program: "true".into(),
                    args: vec![],
                    delivery: PopulationDelivery::Args,
                    timeout_secs: 5,
                    pass_codes: vec![0],
                }),
                Selector::Glob {
                    pattern: "**/*.rs".into(),
                },
                1,
                "c1",
                "o",
            )
            .unwrap();
        let record = store.add_record(&project, "keep this").unwrap();
        store
            .upsert_state(UpsertStateArgs {
                scope: StateScope::Workspace,
                slug: "rule".into(),
                sensitivity: SensitivityPatch::Inherit,
                payload: handoff("v1"),
                at: At::from_unix_millis(10),
                by: "carol".into(),
            })
            .unwrap();
        store
            .send_wire(SendWireArgs {
                id: None,
                sender_id: None,
                sender_name: "carol".into(),
                sender_host: None,
                to: WireAddress::AgentName("bob".into()),
                kind: MessageKind::Mail,
                urgency: MessageUrgency::Normal,
                durable: false,
                subject: "first".into(),
                body: "hello".into(),
                reply_to: None,
                thread_id: None,
                deadline_at: None,
                at: At::from_unix_millis(10),
            })
            .unwrap();
        (project, gate, record)
    }

    fn assert_populated(store: &AnyStore, project: &ProjectId, gate: &GateId, record: &RecordId) {
        assert_eq!(
            store.get_project(project).unwrap().unwrap().root,
            "/work/app"
        );
        assert_eq!(store.get_gate(gate).unwrap().unwrap().name, "fmt");
        assert!(store.get_record(record).unwrap().is_some());
        let entry = store
            .get_state_by_key(&StateScope::Workspace, StateConcern::Handoff, "rule")
            .unwrap()
            .expect("the state entry");
        assert_eq!(entry.slug, "rule");
        let inbox = store
            .list_wire_inbox(&WireFilter {
                recipient: Some("bob".into()),
                ..WireFilter::default()
            })
            .unwrap();
        assert_eq!(inbox.len(), 1);
        assert_eq!(inbox[0].message.subject, "first");
    }

    fn ferro_opts(unmigrated: Unmigrated) -> StoreOptions {
        StoreOptions {
            backend: Some(BackendKind::Ferro),
            unmigrated,
            retries: 0,
        }
    }

    #[cfg(unix)]
    fn mode(path: &Path) -> u32 {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    #[cfg(unix)]
    #[test]
    fn a_lone_redb_is_migrated_once_and_kept_read_only() {
        use std::os::unix::fs::MetadataExt;
        let dir = TempDir::new().unwrap();
        let redb = dir.path().join("s.redb");
        let fls = dir.path().join("s.fls");
        let (project, gate, record) = {
            let src = RedbStore::open(&redb).unwrap();
            populate(&src)
        };

        let store = open_store_with(&redb, &ferro_opts(Unmigrated::Migrate)).unwrap();
        assert_eq!(store.kind(), BackendKind::Ferro);
        assert_populated(&store, &project, &gate, &record);
        drop(store);
        assert_eq!(mode(&fls), 0o600);
        assert_eq!(mode(&redb), 0o400);
        // Opening a redb file rewrites its internal pages, so the bytes before
        // the migration are not comparable; what the migration leaves is.
        let kept = std::fs::read(&redb).unwrap();
        let kept_mtime = std::fs::metadata(&redb).unwrap().modified().unwrap();
        // Its content is still all there: read a writable copy of it.
        let copy = dir.path().join("copy.redb");
        std::fs::copy(&redb, &copy).unwrap();
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&copy, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        assert_populated(&open_redb(&copy).unwrap(), &project, &gate, &record);
        std::fs::remove_file(&copy).unwrap();
        assert!(
            std::fs::read_dir(dir.path()).unwrap().all(|e| !e
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains("migrating")),
            "no temp file is left"
        );

        // A second open finds the .fls: it does not migrate again.
        let meta = std::fs::metadata(&fls).unwrap();
        let again = open_store_with(&redb, &ferro_opts(Unmigrated::Migrate)).unwrap();
        assert_eq!(again.kind(), BackendKind::Ferro);
        assert_populated(&again, &project, &gate, &record);
        drop(again);
        let after = std::fs::metadata(&fls).unwrap();
        assert_eq!(after.ino(), meta.ino());
        assert_eq!(after.mtime(), meta.mtime());
        assert_eq!(after.mtime_nsec(), meta.mtime_nsec());
        assert_eq!(std::fs::read(&redb).unwrap(), kept);
        assert_eq!(
            std::fs::metadata(&redb).unwrap().modified().unwrap(),
            kept_mtime
        );
        assert_eq!(mode(&redb), 0o400);
    }

    #[cfg(unix)]
    #[test]
    fn a_read_only_redb_beside_its_fls_is_never_opened_for_write() {
        let dir = TempDir::new().unwrap();
        let redb = dir.path().join("s.redb");
        drop(RedbStore::open(&redb).unwrap());
        drop(open_store_with(&redb, &ferro_opts(Unmigrated::Migrate)).unwrap());
        assert_eq!(mode(&redb), 0o400);
        let before = std::fs::metadata(&redb).unwrap().modified().unwrap();
        drop(open_store_with(&redb, &ferro_opts(Unmigrated::Migrate)).unwrap());
        drop(open_store_with(&redb, &ferro_opts(Unmigrated::Refuse)).unwrap());
        assert_eq!(
            std::fs::metadata(&redb).unwrap().modified().unwrap(),
            before
        );
        assert_eq!(mode(&redb), 0o400);
    }

    #[test]
    fn a_redb_another_store_holds_stays_on_redb_and_creates_no_fls() {
        let dir = TempDir::new().unwrap();
        let redb = dir.path().join("s.redb");
        let fls = dir.path().join("s.fls");
        drop(RedbStore::open(&redb).unwrap());
        // redb takes an exclusive file lock per open, so a second open in this
        // process is refused exactly as another process's would be.
        let holder = RedbStore::open(&redb).unwrap();
        let refused = RedbStore::open_with_retries(&redb, 0).err().expect("held");
        assert!(is_redb_busy(&refused), "{refused}");
        let err = open_store_with(&redb, &ferro_opts(Unmigrated::Migrate))
            .err()
            .expect("the holder still has the file");
        assert!(is_redb_busy(&err), "{err}");
        assert!(!fls.exists(), "no .fls was created");
        drop(holder);

        // Held for a moment, then released: the migration finds it busy and
        // does not run; the fallback waits its retries and opens the redb.
        let (held_tx, held_rx) = std::sync::mpsc::channel();
        let path = redb.clone();
        let holder = std::thread::spawn(move || {
            let store = RedbStore::open(&path).unwrap();
            held_tx.send(()).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(300));
            drop(store);
        });
        held_rx.recv().unwrap();
        let store = open_store_with(
            &redb,
            &StoreOptions {
                retries: 200,
                ..ferro_opts(Unmigrated::Migrate)
            },
        )
        .unwrap();
        assert_eq!(store.kind(), BackendKind::Redb);
        assert!(store.label().ends_with("s.redb"), "{}", store.label());
        assert!(!fls.exists(), "no .fls was created");
        holder.join().unwrap();
        drop(store);

        // Free again: the same call migrates.
        let store = open_store_with(&redb, &ferro_opts(Unmigrated::Migrate)).unwrap();
        assert_eq!(store.kind(), BackendKind::Ferro);
        assert!(fls.exists());
    }

    /// The lines a store open prints, collected instead of written to stderr.
    #[derive(Clone, Default)]
    struct Lines(std::sync::Arc<Mutex<Vec<String>>>);

    impl Lines {
        fn sink(&self) -> impl Fn(&str) + '_ {
            |line: &str| self.0.lock().unwrap().push(line.to_string())
        }
        fn all(&self) -> Vec<String> {
            self.0.lock().unwrap().clone()
        }
        fn count_containing(&self, needle: &str) -> usize {
            self.all().iter().filter(|l| l.contains(needle)).count()
        }
    }

    fn leftovers(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains("migrating"))
            .collect();
        names.sort();
        names
    }

    /// A table the copy does not know: the verifier reports it, so a migration
    /// of this redb is not clean.
    fn add_unknown_table(redb: &Path) {
        use redb::TableDefinition;
        let db = redb::Database::create(redb).unwrap();
        let tx = db.begin_write().unwrap();
        {
            let mut t = tx
                .open_table(TableDefinition::<&str, &str>::new("zz_unknown"))
                .unwrap();
            t.insert("k", "v").unwrap();
        }
        tx.commit().unwrap();
    }

    fn drop_unknown_table(redb: &Path) {
        use redb::TableDefinition;
        let db = redb::Database::create(redb).unwrap();
        let tx = db.begin_write().unwrap();
        assert!(
            tx.delete_table(TableDefinition::<&str, &str>::new("zz_unknown"))
                .unwrap()
        );
        tx.commit().unwrap();
    }

    /// Migration attempts per store key, counted where `migrate_verified` is
    /// about to run (see `open_ferro`).
    fn attempts() -> std::sync::MutexGuard<'static, std::collections::HashMap<PathBuf, usize>> {
        static ATTEMPTS: OnceLock<Mutex<std::collections::HashMap<PathBuf, usize>>> =
            OnceLock::new();
        ATTEMPTS
            .get_or_init(|| Mutex::new(std::collections::HashMap::new()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    pub(super) fn note_migration_attempt(key: &Path) {
        *attempts().entry(key.to_path_buf()).or_default() += 1;
    }

    #[test]
    fn two_threads_racing_on_an_unclean_store_migrate_once_and_warn_once() {
        let dir = TempDir::new().unwrap();
        let redb = dir.path().join("s.redb");
        let fls = dir.path().join("s.fls");
        {
            let src = RedbStore::open(&redb).unwrap();
            populate(&src);
        }
        add_unknown_table(&redb);
        let key = memory_key(&redb);

        // Hold the lock so that both threads get past the early check of the
        // process memory and queue on the lock; then let them go.
        let holder = MigrationLock::acquire(&fls, true).unwrap().unwrap();
        let said = Lines::default();
        let opened: Vec<BackendKind> = std::thread::scope(|scope| {
            let threads: Vec<_> = (0..2)
                .map(|_| {
                    scope.spawn(|| {
                        // Both threads fall back to the same redb file, which
                        // opens once per process: the retries make the second
                        // wait for the first to drop its store.
                        let options = StoreOptions {
                            backend: Some(BackendKind::Ferro),
                            unmigrated: Unmigrated::Migrate,
                            retries: 200,
                        };
                        let store = open_store_sink(&redb, &options, &said.sink()).unwrap();
                        store.kind()
                    })
                })
                .collect();
            std::thread::sleep(std::time::Duration::from_millis(300));
            drop(holder);
            threads.into_iter().map(|t| t.join().unwrap()).collect()
        });
        assert_eq!(opened, vec![BackendKind::Redb, BackendKind::Redb]);
        assert_eq!(attempts().get(&key).copied(), Some(1), "one migration");
        assert_eq!(said.all().len(), 1, "{:?}", said.all());
        assert_eq!(said.count_containing("did not verify"), 1);
        assert!(!fls.exists());
    }

    #[test]
    fn a_migration_that_does_not_verify_stays_on_redb_warns_once_and_is_not_repeated() {
        let dir = TempDir::new().unwrap();
        let redb = dir.path().join("s.redb");
        let fls = dir.path().join("s.fls");
        let (project, _gate, _record) = {
            let src = RedbStore::open(&redb).unwrap();
            populate(&src)
        };
        add_unknown_table(&redb);

        let said = Lines::default();
        let store = open_store_sink(&redb, &ferro_opts(Unmigrated::Migrate), &said.sink()).unwrap();
        assert_eq!(store.kind(), BackendKind::Redb);
        assert_eq!(
            store.get_project(&project).unwrap().unwrap().root,
            "/work/app"
        );
        assert!(!fls.exists(), "no .fls was created");
        drop(store);
        assert_eq!(leftovers(dir.path()), Vec::<String>::new(), "no temp file");
        #[cfg(unix)]
        assert_eq!(mode(&redb), 0o600, "the redb stays writable");
        let lines = said.all();
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(lines[0].contains("stays on redb"), "{lines:?}");
        assert!(lines[0].contains("did not verify"), "{lines:?}");
        assert!(
            lines[0].contains("zz_unknown"),
            "names the table: {lines:?}"
        );
        assert!(lines[0].contains("fl store migrate"), "{lines:?}");

        // Remembered for the life of the process: with the cause removed, the
        // store is still not copied again, and says nothing more.
        drop_unknown_table(&redb);
        for _ in 0..3 {
            let store =
                open_store_sink(&redb, &ferro_opts(Unmigrated::Migrate), &said.sink()).unwrap();
            assert_eq!(store.kind(), BackendKind::Redb);
        }
        assert!(!fls.exists(), "no migration ran again");
        assert_eq!(said.all().len(), 1, "no second warning: {:?}", said.all());

        // An explicit migration clears the memory: the next open migrates.
        forget_fallback(&redb);
        let store = open_store_sink(&redb, &ferro_opts(Unmigrated::Migrate), &said.sink()).unwrap();
        assert_eq!(store.kind(), BackendKind::Ferro);
        assert!(fls.exists());
    }

    #[test]
    fn a_held_redb_warns_once_per_process_but_is_probed_again_every_open() {
        let dir = TempDir::new().unwrap();
        let redb = dir.path().join("s.redb");
        let fls = dir.path().join("s.fls");
        drop(RedbStore::open(&redb).unwrap());
        let said = Lines::default();

        let holder = RedbStore::open(&redb).unwrap();
        for _ in 0..4 {
            let err = open_store_sink(&redb, &ferro_opts(Unmigrated::Migrate), &said.sink())
                .err()
                .expect("the holder still has the file");
            assert!(is_redb_busy(&err), "{err}");
        }
        assert_eq!(
            said.count_containing("still holds the redb"),
            1,
            "{:?}",
            said.all()
        );
        assert!(!fls.exists());
        drop(holder);

        // The holder stopped: the busy case was never cached, so this migrates.
        let store = open_store_sink(&redb, &ferro_opts(Unmigrated::Migrate), &said.sink()).unwrap();
        assert_eq!(store.kind(), BackendKind::Ferro);
        assert!(fls.exists());
        assert_eq!(said.count_containing("still holds the redb"), 1);
    }

    #[test]
    fn every_automatic_migration_says_so_on_one_line() {
        let dir = TempDir::new().unwrap();
        let redb = dir.path().join("s.redb");
        let fls = dir.path().join("s.fls");
        drop(RedbStore::open(&redb).unwrap());
        let said = Lines::default();
        drop(open_store_sink(&redb, &ferro_opts(Unmigrated::Migrate), &said.sink()).unwrap());
        assert_eq!(
            said.all(),
            vec![format!(
                "migrated {} -> {}; the redb is kept read-only (restore with chmod u+w and \
                 FL_STORE_BACKEND=redb)",
                redb.display(),
                fls.display()
            )]
        );
        // A second open finds the .fls: nothing more is said.
        drop(open_store_sink(&redb, &ferro_opts(Unmigrated::Migrate), &said.sink()).unwrap());
        assert_eq!(said.all().len(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn a_shared_lock_reads_a_read_only_redb_without_touching_it_and_sees_a_holder() {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempDir::new().unwrap();
        let redb = dir.path().join("s.redb");
        let holder = RedbStore::open(&redb).unwrap();
        let err = crate::RedbSharedLock::acquire(&redb)
            .err()
            .expect("a writer holds it");
        assert!(is_redb_busy(&err), "{err}");
        drop(holder);

        std::fs::set_permissions(&redb, std::fs::Permissions::from_mode(0o400)).unwrap();
        let bytes = std::fs::read(&redb).unwrap();
        let mtime = std::fs::metadata(&redb).unwrap().modified().unwrap();
        let lock = crate::RedbSharedLock::acquire(&redb).unwrap();
        // Shared: a second reader is fine.
        drop(crate::RedbSharedLock::acquire(&redb).unwrap());
        drop(lock);
        assert_eq!(std::fs::read(&redb).unwrap(), bytes);
        assert_eq!(std::fs::metadata(&redb).unwrap().modified().unwrap(), mtime);
    }

    #[test]
    fn the_busy_error_is_classified_by_cause() {
        let busy = |cause: &str| StoreError::Unreachable {
            store: "s".into(),
            cause: cause.into(),
        };
        assert!(is_redb_busy(&busy(
            "Database already open. Cannot acquire lock."
        )));
        assert!(is_redb_busy(&busy("Cannot acquire lock")));
        assert!(!is_redb_busy(&busy("Permission denied")));
        assert!(!is_redb_busy(&backend("Database already open")));
    }

    #[test]
    fn a_fls_that_appears_first_is_opened_not_migrated_over() {
        let dir = TempDir::new().unwrap();
        let redb = dir.path().join("s.redb");
        let fls = dir.path().join("s.fls");
        let (project, _gate, _record) = {
            let src = RedbStore::open(&redb).unwrap();
            populate(&src)
        };
        // Another process migrated first: its .fls (here with nothing from
        // the redb in it) is the store. This is the plain `.fls`-exists
        // branch; the lone-redb branch's own re-check is tested below.
        drop(FerroStore::open(&fls).unwrap());
        let store = open_store_with(&redb, &ferro_opts(Unmigrated::Migrate)).unwrap();
        assert_eq!(store.kind(), BackendKind::Ferro);
        assert!(store.get_project(&project).is_err());
    }

    /// A file that is no redb database: opening it as one fails, not busy.
    fn write_garbage_redb(path: &Path) {
        std::fs::write(path, vec![0x5a_u8; 8192]).unwrap();
    }

    #[test]
    fn a_redb_open_that_fails_is_replaced_by_the_fls_when_one_exists() {
        // The `Err` arm of the fallback: the redb cannot be opened (here it is
        // not a database, which fails the same way a 0400 file does for a
        // user, and does not depend on who runs the test) and a `.fls` exists.
        let dir = TempDir::new().unwrap();
        let redb = dir.path().join("s.redb");
        let fls = dir.path().join("s.fls");
        write_garbage_redb(&redb);
        let err = RedbStore::open_with_retries(&redb, 0)
            .err()
            .expect("not a redb file");
        assert!(!is_redb_busy(&err), "{err}");

        // No .fls: the error is the caller's.
        for locking in [Access::Use, Access::Scan] {
            let err = redb_engine(&redb, &fls, 0, None, locking)
                .err()
                .expect("no .fls to fall back to");
            assert!(!is_redb_busy(&err), "{err}");
        }
        let err = redb_engine_unlocked(&redb, &fls, 0)
            .err()
            .expect("no .fls to fall back to");
        assert!(!is_redb_busy(&err), "{err}");
        assert!(!fls.exists());

        // A .fls: it is opened, whichever way the redb was reached. (The lock
        // is created by the first call, so `IfExists` takes it too.)
        drop(FerroStore::open(&fls).unwrap());
        for locking in [Access::Use, Access::Scan] {
            let (backend, label) =
                redb_engine(&redb, &fls, 0, None, locking).expect("the .fls is the store");
            assert!(matches!(backend, Backend::Ferro(_)));
            assert!(label.ends_with("s.fls"), "{label}");
        }
        assert!(matches!(
            redb_engine_unlocked(&redb, &fls, 0).ok().unwrap().0,
            Backend::Ferro(_)
        ));
    }

    #[test]
    fn a_fls_that_appears_while_a_process_waits_for_the_lock_is_opened() {
        // The lone-redb branch itself: the redb exists, no .fls yet; this
        // process waits for the migration lock, and the .fls is there by the
        // time it has it. It must open the .fls, not probe or copy the redb.
        let dir = TempDir::new().unwrap();
        let redb = dir.path().join("s.redb");
        let fls = dir.path().join("s.fls");
        let (project, gate, record) = {
            let src = RedbStore::open(&redb).unwrap();
            populate(&src)
        };
        let holder = MigrationLock::acquire(&fls, true).unwrap().unwrap();
        let (go_tx, go_rx) = std::sync::mpsc::channel();
        let waiter = {
            let (redb, said) = (redb.clone(), Lines::default());
            std::thread::spawn(move || {
                go_tx.send(()).unwrap();
                let store =
                    open_store_sink(&redb, &ferro_opts(Unmigrated::Migrate), &said.sink()).unwrap();
                (store.kind(), said.all())
            })
        };
        go_rx.recv().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(150));
        // The lock holder "migrated": a real migration, then released the lock.
        crate::migrate::migrate_verified(&redb, &fls).unwrap();
        drop(holder);
        let (kind, lines) = waiter.join().unwrap();
        assert_eq!(kind, BackendKind::Ferro);
        assert!(lines.is_empty(), "the waiter migrated nothing: {lines:?}");
        assert_eq!(leftovers(dir.path()), Vec::<String>::new());
        let store = open_store_with(&redb, &ferro_opts(Unmigrated::Migrate)).unwrap();
        assert_populated(&store, &project, &gate, &record);
    }

    #[test]
    fn the_three_unmigrated_choices_differ_only_in_what_they_do() {
        let populated = |name: &str| {
            let dir = TempDir::new().unwrap();
            let redb = dir.path().join(name);
            let ids = {
                let src = RedbStore::open(&redb).unwrap();
                populate(&src)
            };
            (dir, redb, ids)
        };

        // Refuse: an error naming the remedy, and nothing changes.
        let (dir, redb, _) = populated("a.redb");
        let before = std::fs::read(&redb).unwrap();
        let err = open_store_with(&redb, &ferro_opts(Unmigrated::Refuse))
            .err()
            .expect("refused");
        assert!(err.to_string().contains("fl store migrate"), "{err}");
        assert_eq!(std::fs::read(&redb).unwrap(), before);
        assert!(!dir.path().join("a.fls").exists());
        assert!(!MigrationLock::path_for(&dir.path().join("a.fls")).exists());

        // UseRedb: opens the redb engine, creates no file, migrates nothing,
        // leaves the mode alone (and prints nothing).
        let (dir, redb, (project, gate, record)) = populated("b.redb");
        let names = |d: &Path| {
            let mut n: Vec<String> = std::fs::read_dir(d)
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            n.sort();
            n
        };
        let listing = names(dir.path());
        let bytes = std::fs::read(&redb).unwrap();
        let mtime = std::fs::metadata(&redb).unwrap().modified().unwrap();
        let said = Lines::default();
        let store = open_store_sink(&redb, &ferro_opts(Unmigrated::UseRedb), &said.sink()).unwrap();
        assert_eq!(store.kind(), BackendKind::Redb);
        assert_populated(&store, &project, &gate, &record);
        assert!(store.owns(project.iri()).unwrap());
        // Read-only: it cannot be written through.
        let err = store.add_project("/not-written").expect_err("read-only");
        assert!(err.to_string().contains("read-only"), "{err}");
        drop(store);
        assert_eq!(names(dir.path()), listing, "no file was created");
        assert!(said.all().is_empty(), "{:?}", said.all());
        assert_eq!(std::fs::read(&redb).unwrap(), bytes, "bytes untouched");
        assert_eq!(
            std::fs::metadata(&redb).unwrap().modified().unwrap(),
            mtime,
            "mtime untouched"
        );
        #[cfg(unix)]
        assert_eq!(mode(&redb), 0o600);

        // A redb that cannot be read read-only (here: no tables yet) is
        // opened the ordinary way, as before.
        let empty = dir.path().join("empty.redb");
        std::fs::write(&empty, b"").unwrap();
        let store = open_store_sink(&empty, &ferro_opts(Unmigrated::UseRedb), &|_| {}).unwrap();
        assert_eq!(store.kind(), BackendKind::Redb);
        assert!(store.list_projects().unwrap().is_empty());
        drop(store);
        assert!(!dir.path().join("empty.fls").exists());

        // The convenience: UseRedb under the environment's engine. A lone
        // redb is read as redb whichever engine is selected, and never migrated.
        let store = open_store_for_scan(&redb).unwrap();
        assert_eq!(store.kind(), BackendKind::Redb);

        // Migrate: a .fls, a read-only redb, and a notice.
        let (dir, redb, (project, gate, record)) = populated("c.redb");
        let said = Lines::default();
        let store = open_store_sink(&redb, &ferro_opts(Unmigrated::Migrate), &said.sink()).unwrap();
        assert_eq!(store.kind(), BackendKind::Ferro);
        assert_populated(&store, &project, &gate, &record);
        assert!(dir.path().join("c.fls").exists());
        assert_eq!(said.all().len(), 1);
        #[cfg(unix)]
        assert_eq!(mode(&redb), 0o400);
    }

    #[test]
    fn a_scan_open_takes_an_existing_migration_lock_and_creates_none() {
        let dir = TempDir::new().unwrap();
        let redb = dir.path().join("s.redb");
        let fls = dir.path().join("s.fls");
        let lock_path = MigrationLock::path_for(&fls);
        drop(RedbStore::open(&redb).unwrap());
        drop(open_store_sink(&redb, &ferro_opts(Unmigrated::UseRedb), &|_| {}).unwrap());
        assert!(!lock_path.exists(), "a scan created the lock file");

        // With the lock file there and held, the scan waits for it.
        let holder = MigrationLock::acquire(&fls, true).unwrap().unwrap();
        let started = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let scanner = {
            let (redb, started, done) = (redb.clone(), started.clone(), done.clone());
            std::thread::spawn(move || {
                started.store(true, std::sync::atomic::Ordering::SeqCst);
                let store =
                    open_store_sink(&redb, &ferro_opts(Unmigrated::UseRedb), &|_| {}).unwrap();
                done.store(true, std::sync::atomic::Ordering::SeqCst);
                store.kind()
            })
        };
        while !started.load(std::sync::atomic::Ordering::SeqCst) {
            std::thread::yield_now();
        }
        std::thread::sleep(std::time::Duration::from_millis(150));
        assert!(
            !done.load(std::sync::atomic::Ordering::SeqCst),
            "the scan opened the redb while the lock was held"
        );
        drop(holder);
        assert_eq!(scanner.join().unwrap(), BackendKind::Redb);
    }

    #[test]
    fn a_lock_that_stays_held_times_out_with_an_error_naming_the_lock_file() {
        let dir = TempDir::new().unwrap();
        let fls = dir.path().join("s.fls");
        let lock_path = MigrationLock::path_for(&fls);
        let holder = MigrationLock::acquire(&fls, true).unwrap().unwrap();

        for create in [true, false] {
            let started = std::time::Instant::now();
            let err =
                MigrationLock::acquire_for(&fls, create, std::time::Duration::from_millis(80))
                    .err()
                    .expect("the holder still has the lock");
            assert!(started.elapsed() >= std::time::Duration::from_millis(80));
            assert!(started.elapsed() < std::time::Duration::from_secs(10));
            assert_eq!(err.kind(), std::io::ErrorKind::TimedOut, "{err}");
            let text = err.to_string();
            assert!(text.contains(&lock_path.display().to_string()), "{text}");
            assert!(text.contains("another process is migrating"), "{text}");
            // Waiting out a holder is not benign: no unlocked open.
            assert!(!lock_error_is_benign(&err));
        }

        // Released: the same call takes it at once.
        drop(holder);
        assert!(
            MigrationLock::acquire_for(&fls, true, std::time::Duration::from_millis(80))
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn acquire_within_creates_the_lock_file_and_excludes_every_other_taker() {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempDir::new().unwrap();
        let fls = dir.path().join("s.fls");
        let lock_path = MigrationLock::path_for(&fls);
        assert!(!lock_path.exists());

        // Free: taken at once with no wait, the file made 0600.
        let held = MigrationLock::acquire_within(&fls, std::time::Duration::ZERO).unwrap();
        let mode = std::fs::metadata(&lock_path).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);

        // Held: a second taker with no wait is refused as `TimedOut`, naming the file,
        // and so is the engine's own acquire (the same lock).
        let err = MigrationLock::acquire_within(&fls, std::time::Duration::ZERO)
            .err()
            .expect("the lock is held");
        assert_eq!(err.kind(), std::io::ErrorKind::TimedOut, "{err}");
        assert!(err.to_string().contains(&lock_path.display().to_string()));
        let engine_err = MigrationLock::acquire_for(&fls, false, std::time::Duration::ZERO)
            .err()
            .expect("the engine's own acquire must see this holder");
        assert_eq!(
            engine_err.kind(),
            std::io::ErrorKind::TimedOut,
            "{engine_err}"
        );

        // Released: taken again, and the file is left in place.
        drop(held);
        let again = MigrationLock::acquire_within(&fls, std::time::Duration::ZERO).unwrap();
        drop(again);
        assert!(lock_path.exists());
    }

    #[test]
    fn acquire_within_refuses_a_symlink_at_the_lock_path() {
        let dir = TempDir::new().unwrap();
        let fls = dir.path().join("s.fls");
        let target = dir.path().join("elsewhere");
        std::fs::write(&target, b"x").unwrap();
        std::os::unix::fs::symlink(&target, MigrationLock::path_for(&fls)).unwrap();
        let err = MigrationLock::acquire_within(&fls, std::time::Duration::ZERO)
            .err()
            .expect("a symlinked lock path is refused");
        assert_ne!(err.kind(), std::io::ErrorKind::TimedOut, "{err}");
        assert_eq!(std::fs::read(&target).unwrap(), b"x");
    }

    #[test]
    fn only_a_lock_error_that_rules_out_a_holder_is_benign() {
        use std::io::{Error, ErrorKind};
        for kind in [
            ErrorKind::PermissionDenied,
            ErrorKind::ReadOnlyFilesystem,
            ErrorKind::NotFound,
        ] {
            assert!(lock_error_is_benign(&Error::from(kind)), "{kind:?}");
        }
        // EROFS by number, whatever kind std gives it.
        assert!(lock_error_is_benign(&Error::from_raw_os_error(30)));
        assert!(lock_error_is_benign(&Error::from_raw_os_error(13))); // EACCES
        assert!(lock_error_is_benign(&Error::from_raw_os_error(2))); // ENOENT
        for err in [
            Error::from_raw_os_error(37), // ENOLCK
            Error::from_raw_os_error(5),  // EIO
            Error::from_raw_os_error(21), // EISDIR
            Error::from_raw_os_error(40), // ELOOP: a symlink at the lock path
            Error::from(ErrorKind::TimedOut),
            Error::from(ErrorKind::WouldBlock),
            Error::from(ErrorKind::Interrupted),
            Error::from(ErrorKind::Other),
            Error::other("unexpected"),
        ] {
            assert!(!lock_error_is_benign(&err), "{err:?}");
        }
    }

    /// A symlink at the lock path, live or dangling, is never followed: the
    /// open is refused (ELOOP), the target is neither locked, written, nor
    /// created, and the redb is neither opened nor migrated.
    #[cfg(unix)]
    #[test]
    fn a_symlink_at_the_lock_path_is_a_hard_error_and_is_never_followed() {
        const ELOOP: i32 = 40;
        for live in [true, false] {
            let dir = TempDir::new().unwrap();
            let redb = dir.path().join("s.redb");
            let fls = dir.path().join("s.fls");
            let target = dir.path().join("lock-target");
            {
                let src = RedbStore::open(&redb).unwrap();
                populate(&src);
            }
            if live {
                std::fs::write(&target, b"untouched").unwrap();
            }
            std::os::unix::fs::symlink(&target, MigrationLock::path_for(&fls)).unwrap();
            let before = std::fs::read(&redb).unwrap();

            // The probe (no create) and the creating open both fail hard; the
            // probe's NotFound => Ok(None) must not swallow a dangling symlink.
            for create in [false, true] {
                let err = MigrationLock::acquire(&fls, create)
                    .err()
                    .unwrap_or_else(|| panic!("live={live} create={create}: lock taken"));
                assert_eq!(err.raw_os_error(), Some(ELOOP), "{err:?}");
                assert!(!lock_error_is_benign(&err), "{err:?}");
            }

            for unmigrated in [Unmigrated::Migrate, Unmigrated::UseRedb] {
                let said = Lines::default();
                let err = open_store_sink(&redb, &ferro_opts(unmigrated), &said.sink())
                    .err()
                    .unwrap_or_else(|| panic!("live={live} {unmigrated:?}: opened"));
                let text = err.to_string();
                assert!(text.contains("could not take the migration lock"), "{text}");
                assert!(said.all().is_empty(), "{:?}", said.all());
            }
            assert!(!fls.exists(), "nothing was migrated");
            assert_eq!(std::fs::read(&redb).unwrap(), before, "redb untouched");
            assert_eq!(mode(&redb), 0o600);
            if live {
                assert_eq!(std::fs::read(&target).unwrap(), b"untouched");
            } else {
                assert!(!target.exists(), "the dangling target was not created");
            }
            assert!(
                std::fs::symlink_metadata(MigrationLock::path_for(&fls))
                    .unwrap()
                    .file_type()
                    .is_symlink()
            );
        }
    }

    #[test]
    fn a_lock_error_that_is_not_benign_is_an_error_not_an_unlocked_open() {
        // A directory where the lock file belongs: opening it for writing fails
        // with EISDIR, which says nothing about another holder.
        let dir = TempDir::new().unwrap();
        let redb = dir.path().join("s.redb");
        let fls = dir.path().join("s.fls");
        let (project, gate, record) = {
            let src = RedbStore::open(&redb).unwrap();
            populate(&src)
        };
        std::fs::create_dir(MigrationLock::path_for(&fls)).unwrap();
        let before = std::fs::read(&redb).unwrap();

        for unmigrated in [Unmigrated::Migrate, Unmigrated::UseRedb] {
            let said = Lines::default();
            let err = open_store_sink(&redb, &ferro_opts(unmigrated), &said.sink())
                .err()
                .unwrap_or_else(|| panic!("{unmigrated:?} opened the redb with no lock"));
            let text = err.to_string();
            assert!(text.contains("could not take the migration lock"), "{text}");
            assert!(text.contains("s.fls.migrate.lock"), "{text}");
            assert!(said.all().is_empty(), "{:?}", said.all());
        }
        assert!(!fls.exists());
        assert_eq!(std::fs::read(&redb).unwrap(), before);

        // The directory gone, the same redb opens and migrates normally.
        std::fs::remove_dir(MigrationLock::path_for(&fls)).unwrap();
        let store = open_store_with(&redb, &ferro_opts(Unmigrated::Migrate)).unwrap();
        assert_eq!(store.kind(), BackendKind::Ferro);
        assert_populated(&store, &project, &gate, &record);
    }

    #[test]
    #[cfg(unix)]
    fn a_directory_that_refuses_the_lock_file_keeps_the_redb_engine_and_says_so() {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("ro");
        std::fs::create_dir(&sub).unwrap();
        let redb = sub.join("s.redb");
        let fls = sub.join("s.fls");
        let (project, gate, record) = {
            let src = RedbStore::open(&redb).unwrap();
            populate(&src)
        };
        // Put the directory back so the TempDir can clean up, however this ends.
        struct Restore(PathBuf);
        impl Drop for Restore {
            fn drop(&mut self) {
                let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(0o755));
            }
        }
        let _restore = Restore(sub.clone());
        std::fs::set_permissions(&sub, std::fs::Permissions::from_mode(0o555)).unwrap();
        if std::fs::File::create(sub.join("probe")).is_ok() {
            eprintln!(
                "SKIPPED a_directory_that_refuses_the_lock_file_keeps_the_redb_engine_and_says_so: \
                 the directory is writable despite mode 0555 (running as root)"
            );
            return;
        }
        let before = std::fs::read(&redb).unwrap();

        let said = Lines::default();
        let store = open_store_sink(&redb, &ferro_opts(Unmigrated::Migrate), &said.sink()).unwrap();
        assert_eq!(store.kind(), BackendKind::Redb);
        assert!(store.label().ends_with("s.redb"), "{}", store.label());
        assert_populated(&store, &project, &gate, &record);
        assert_eq!(
            said.count_containing("could not take the migration lock"),
            1
        );
        assert_eq!(said.count_containing("stays on redb"), 1);
        assert_eq!(said.all().len(), 1, "{:?}", said.all());
        // No .fls and no lock file (or anything else) was made.
        assert!(!fls.exists());
        assert!(!MigrationLock::path_for(&fls).exists());
        let names: Vec<_> = std::fs::read_dir(&sub)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["s.redb".to_string()]);
        // The redb is opened by the ordinary (writable) open on this path, so
        // it is a live store. Observed below: whether its bytes changed.
        drop(store);
        let unchanged = std::fs::read(&redb).unwrap() == before;
        eprintln!("redb bytes unchanged by the unlocked open: {unchanged}");
        // Opened again: the warning is given once per store.
        let again = Lines::default();
        let store =
            open_store_sink(&redb, &ferro_opts(Unmigrated::Migrate), &again.sink()).unwrap();
        assert_eq!(store.kind(), BackendKind::Redb);
        assert!(again.all().is_empty(), "{:?}", again.all());
    }

    #[test]
    fn the_migration_lock_file_is_not_a_store() {
        let dir = TempDir::new().unwrap();
        let redb = dir.path().join("s.redb");
        let fls = dir.path().join("s.fls");
        let lock_path = MigrationLock::path_for(&fls);
        assert_eq!(lock_path, dir.path().join("s.fls.migrate.lock"));
        drop(RedbStore::open(&redb).unwrap());
        drop(open_store_sink(&redb, &ferro_opts(Unmigrated::Migrate), &|_| {}).unwrap());
        // Left behind, empty and private.
        assert_eq!(std::fs::metadata(&lock_path).unwrap().len(), 0);
        #[cfg(unix)]
        assert_eq!(mode(&lock_path), 0o600);
        // Not seen as the store, under either engine, or as a `.redb`/`.fls`.
        assert_eq!(lock_path.extension().and_then(|e| e.to_str()), Some("lock"));
        let only_lock = dir.path().join("t.redb");
        let only_fls = dir.path().join("t.fls");
        std::fs::write(MigrationLock::path_for(&only_fls), b"").unwrap();
        assert!(!store_exists_with(&only_lock, Some("ferro")));
        assert!(!store_exists_with(&only_lock, None));
        assert!(!store_exists_with(&only_fls, None));
        assert!(store_exists_with(&redb, Some("ferro")));
    }

    #[test]
    fn a_redb_opened_on_the_redb_engine_beside_its_fls_warns_once() {
        let dir = TempDir::new().unwrap();
        let redb = dir.path().join("s.redb");
        let redb_opts = StoreOptions {
            backend: Some(BackendKind::Redb),
            ..StoreOptions::default()
        };
        let said = Lines::default();
        // No .fls beside it: nothing to say.
        drop(open_store_sink(&redb, &redb_opts, &said.sink()).unwrap());
        assert!(said.all().is_empty());
        drop(FerroStore::open(dir.path().join("s.fls")).unwrap());
        for _ in 0..3 {
            drop(open_store_sink(&redb, &redb_opts, &said.sink()).unwrap());
        }
        let lines = said.all();
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert!(lines[0].contains("fl store verify"), "{lines:?}");
        assert!(lines[0].contains("s.fls"), "{lines:?}");
        // A ferro open of the same store is not a redb-engine open.
        drop(open_store_sink(&redb, &ferro_opts(Unmigrated::Migrate), &said.sink()).unwrap());
        assert_eq!(said.all().len(), 1);
    }

    #[test]
    fn the_redb_engine_warning_says_the_fls_is_newer_when_it_is() {
        let t0 = std::time::SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000_000);
        let t1 = t0 + std::time::Duration::from_secs(3600);
        let redb_opts = StoreOptions {
            backend: Some(BackendKind::Redb),
            ..StoreOptions::default()
        };
        let set_mtime = |path: &Path, time: std::time::SystemTime| {
            std::fs::OpenOptions::new()
                .write(true)
                .open(path)
                .unwrap()
                .set_modified(time)
                .unwrap();
        };
        // (mtime of the .redb, mtime of the .fls, whether the warning says NEWER)
        for (redb_time, fls_time, newer) in [(t0, t1, true), (t1, t0, false)] {
            let dir = TempDir::new().unwrap();
            let redb = dir.path().join("s.redb");
            let fls = dir.path().join("s.fls");
            drop(RedbStore::open(&redb).unwrap());
            drop(FerroStore::open(&fls).unwrap());
            set_mtime(&redb, redb_time);
            set_mtime(&fls, fls_time);
            let said = Lines::default();
            drop(open_store_sink(&redb, &redb_opts, &said.sink()).unwrap());
            let lines = said.all();
            assert_eq!(lines.len(), 1, "{lines:?}");
            assert_eq!(lines[0].contains("NEWER"), newer, "{lines:?}");
            assert!(lines[0].contains("s.fls"), "{lines:?}");
            assert!(lines[0].contains("fl store verify"), "{lines:?}");
            if !newer {
                assert!(lines[0].contains("may differ"), "{lines:?}");
            }
        }
    }

    #[test]
    fn migrate_locked_creates_the_fls_and_refuses_a_second_one() {
        let dir = TempDir::new().unwrap();
        let redb = dir.path().join("s.redb");
        let fls = dir.path().join("s.fls");
        let (project, gate, record) = {
            let src = RedbStore::open(&redb).unwrap();
            populate(&src)
        };
        assert!(migrate_locked(&redb, &fls).unwrap().is_clean());
        assert!(MigrationLock::path_for(&fls).exists());
        let store = open_store_with(&redb, &ferro_opts(Unmigrated::Migrate)).unwrap();
        assert_populated(&store, &project, &gate, &record);
        let err = migrate_locked(&redb, &fls).unwrap_err();
        assert!(err.to_string().contains("already exists"), "{err}");
    }

    /// One round of the race: a real migration in one thread, an ordinary
    /// ferro-engine open in the other, started `delay` apart.
    #[cfg(unix)]
    fn race_round(
        round: u32,
        a_is_explicit: bool,
        a_delay: std::time::Duration,
        b_delay: std::time::Duration,
    ) {
        use std::sync::{Arc, Barrier};
        let dir = TempDir::new().unwrap();
        let redb = dir.path().join("s.redb");
        let fls = dir.path().join("s.fls");
        let (project, gate, record) = {
            let src = RedbStore::open(&redb).unwrap();
            let ids = populate(&src);
            // Enough rows that the copy takes a moment.
            for i in 0..120 {
                src.add_record(&ids.0, &format!("filler {i}")).unwrap();
            }
            ids
        };
        let barrier = Arc::new(Barrier::new(2));
        let a = {
            let (redb, fls, barrier) = (redb.clone(), fls.clone(), barrier.clone());
            std::thread::spawn(move || {
                barrier.wait();
                std::thread::sleep(a_delay);
                if a_is_explicit {
                    // `fl store migrate`: an "already exists" refusal is the
                    // other thread having won, which is allowed.
                    match migrate_locked(&redb, &fls) {
                        Ok(report) => assert!(report.is_clean()),
                        Err(err) => assert!(err.to_string().contains("already exists"), "{err}"),
                    }
                } else {
                    let store =
                        open_store_sink(&redb, &ferro_opts(Unmigrated::Migrate), &|_| {}).unwrap();
                    assert_eq!(store.kind(), BackendKind::Ferro);
                }
            })
        };
        let b = {
            let (redb, barrier) = (redb.clone(), barrier.clone());
            std::thread::spawn(move || {
                barrier.wait();
                std::thread::sleep(b_delay);
                // The retries are the ordinary ones, so a busy redb is waited for.
                let store = open_store_sink(
                    &redb,
                    &StoreOptions {
                        retries: 200,
                        ..ferro_opts(Unmigrated::Migrate)
                    },
                    &|_| {},
                )
                .unwrap();
                assert_eq!(store.kind(), BackendKind::Ferro, "round {round}");
                // A write by this process: it must land in the .fls.
                store.add_project("/b-wrote").unwrap();
            })
        };
        a.join().unwrap();
        b.join().unwrap();

        // Exactly one .fls, no temp file, and the redb is the read-only copy
        // it was before any writer could reach it.
        let mut fls_files: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".fls"))
            .collect();
        fls_files.sort();
        assert_eq!(fls_files, vec!["s.fls".to_string()], "round {round}");
        assert_eq!(leftovers(dir.path()), Vec::<String>::new(), "round {round}");
        assert_eq!(mode(&redb), 0o400, "round {round}");
        // The .fls holds the original data and the write that came after.
        let store = open_store_with(&redb, &ferro_opts(Unmigrated::Migrate)).unwrap();
        assert_eq!(store.kind(), BackendKind::Ferro);
        assert_populated(&store, &project, &gate, &record);
        assert!(
            store
                .list_projects()
                .unwrap()
                .iter()
                .any(|p| p.root == "/b-wrote"),
            "round {round}: the write made after the open is in the .fls"
        );
        // The redb never saw that write: a writable copy of it still has
        // only the original project.
        let copy = dir.path().join("check.redb");
        std::fs::copy(&redb, &copy).unwrap();
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&copy, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        let old = open_redb(&copy).unwrap();
        let roots: Vec<String> = old
            .list_projects()
            .unwrap()
            .into_iter()
            .map(|p| p.root)
            .collect();
        assert_eq!(roots, vec!["/work/app".to_string()], "round {round}");
    }

    #[cfg(unix)]
    #[test]
    fn a_second_ferro_process_never_takes_the_redb_while_a_migration_runs() {
        // Twenty rounds, the second thread starting 0 to 28 ms before or
        // after the first, so the open meets the migration at every phase:
        // before it, during the copy, in the instants after the copy releases
        // the redb and before the .fls is linked, and after.
        for round in 0..20u32 {
            let lag = std::time::Duration::from_micros(u64::from(round) * 1500);
            let none = std::time::Duration::ZERO;
            let (a_delay, b_delay) = if round % 4 < 2 {
                (none, lag)
            } else {
                (lag, none)
            };
            race_round(round, round % 2 == 0, a_delay, b_delay);
        }
    }

    #[test]
    fn turning_auto_migration_off_refuses_and_leaves_the_redb_alone() {
        let dir = TempDir::new().unwrap();
        let redb = dir.path().join("s.redb");
        drop(RedbStore::open(&redb).unwrap());
        let before = std::fs::read(&redb).unwrap();
        let err = open_store_with(&redb, &ferro_opts(Unmigrated::Refuse))
            .err()
            .expect("must refuse");
        assert!(err.to_string().contains("fl store migrate"), "{err}");
        assert!(!dir.path().join("s.fls").exists());
        assert_eq!(std::fs::read(&redb).unwrap(), before);
    }

    #[test]
    fn the_label_names_the_file_the_store_was_opened_from() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("s.redb");
        let store = open_store_with(
            &path,
            &StoreOptions {
                backend: Some(BackendKind::Ferro),
                ..StoreOptions::default()
            },
        )
        .unwrap();
        assert!(store.label().ends_with("s.fls"), "{}", store.label());
        let (redb, _g) = on(BackendKind::Redb);
        assert!(redb.label().ends_with("s.redb"), "{}", redb.label());
    }

    #[test]
    fn an_in_memory_any_store_takes_its_label_from_the_ferrostore() {
        assert_eq!(
            AnyStore::in_memory().label(),
            FerroStore::in_memory().label()
        );
    }

    #[test]
    fn set_routes_makes_the_store_hold_routing_on_both_engines() {
        for kind in both() {
            let (store, _g) = on(kind);
            let project = store.add_project("/repo").unwrap();
            assert!(!store.holds_routing().unwrap(), "{kind:?}");
            store.set_routes(&project, &RoutingMap::starting()).unwrap();
            assert!(store.holds_routing().unwrap(), "{kind:?}");
            assert_eq!(
                store.routes(&project).unwrap(),
                Some(RoutingMap::starting()),
                "{kind:?}"
            );
        }
    }

    #[test]
    fn an_imported_project_is_refused_by_name() {
        let (author, _g) = on(BackendKind::Redb);
        let project = author.add_project("/repo").unwrap();
        let manifest = author.export_manifest(&project, "c0ffee", 1, None).unwrap();
        let (other, _g2) = on(BackendKind::Redb);
        other.import_manifest(&manifest, "/elsewhere").unwrap();
        let err = other.refuse_if_imported(&project, "x").unwrap_err();
        assert!(
            matches!(err, StoreError::Imported { action: "x", .. }),
            "{err}"
        );
    }

    #[test]
    fn the_inherent_questions_agree_on_both_engines() {
        for kind in both() {
            let (store, _g) = on(kind);
            let project = store.add_project("/repo").unwrap();
            assert!(store.owns(project.iri()).unwrap(), "{kind:?}");
            let stranger = Iri::parse("urn:uuid:00000000-0000-7000-8000-000000000001").unwrap();
            assert!(!store.owns(&stranger).unwrap(), "{kind:?}");
            assert_eq!(store.imported_hash(&project).unwrap(), None, "{kind:?}");
            store.refuse_if_imported(&project, "do a thing").unwrap();
            assert!(!store.holds_routing().unwrap(), "{kind:?}");
            assert!(!store.holds_a_ledger_root().unwrap(), "{kind:?}");
            store.set_ledger_root("R_1", "abc123").unwrap();
            assert!(store.holds_a_ledger_root().unwrap(), "{kind:?}");
            let missing = ProjectId(stranger.clone());
            assert!(
                matches!(
                    store.imported_hash(&missing),
                    Err(StoreError::NotOwned { .. })
                ),
                "{kind:?}"
            );
        }
    }

    #[test]
    fn manifests_and_versions_are_served_on_both_engines() {
        for kind in both() {
            let (store, _g) = on(kind);
            assert!(store.format_version().unwrap().is_some(), "{kind:?}");
            let project = store.add_project("/repo").unwrap();
            let manifest = store.export_manifest(&project, "c0ffee", 1, None).unwrap();
            for target in both() {
                let (other, _g2) = on(target);
                let report = other.import_manifest(&manifest, "/elsewhere").unwrap();
                assert_eq!(report.project, project, "{kind:?} -> {target:?}");
                assert!(
                    other.imported_hash(&project).unwrap().is_some(),
                    "{kind:?} -> {target:?}"
                );
                assert!(
                    other.format_version().unwrap() >= Some(crate::FORMAT_WITH_IMPORTS),
                    "{kind:?} -> {target:?}"
                );
            }
        }
    }

    #[test]
    fn the_redb_arm_still_serves_manifests_and_versions() {
        let (store, _g) = on(BackendKind::Redb);
        assert!(store.format_version().unwrap().is_some());
        let project = store.add_project("/repo").unwrap();
        let manifest = store.export_manifest(&project, "c0ffee", 1, None).unwrap();
        let (other, _g2) = on(BackendKind::Redb);
        let report = other.import_manifest(&manifest, "/elsewhere").unwrap();
        assert_eq!(report.project, project);
        assert!(other.imported_hash(&project).unwrap().is_some());
    }

    #[test]
    fn open_redb_ignores_the_environment_and_the_path_extension_rules() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("s.redb");
        let store = open_redb(&path).unwrap();
        assert_eq!(store.kind(), BackendKind::Redb);
        assert!(path.exists());
        assert!(!dir.path().join("s.fls").exists());
    }

    #[test]
    fn store_exists_checks_the_chosen_engines_physical_file() {
        let dir = TempDir::new().unwrap();
        let logical = dir.path().join("s.redb");
        // Nothing yet, under either engine.
        assert!(!store_exists_with(&logical, None));
        assert!(!store_exists_with(&logical, Some("ferro")));
        // A redb file exists for redb; under a ferro selection it is the store
        // that open_store migrates, so it exists there too.
        drop(open_redb(&logical).unwrap());
        assert!(store_exists_with(&logical, None));
        assert!(store_exists_with(&logical, Some("redb")));
        assert!(store_exists_with(&logical, Some("ferro")));
        // A directory named like the redb is not a store.
        let dir_named = dir.path().join("d.redb");
        std::fs::create_dir(&dir_named).unwrap();
        assert!(store_exists_with(&dir_named, Some("redb")));
        assert!(!store_exists_with(&dir_named, Some("ferro")));
        let ferro = StoreOptions {
            backend: Some(BackendKind::Ferro),
            unmigrated: Unmigrated::Refuse,
            ..StoreOptions::default()
        };
        // Without auto-migration the lone .redb is refused, not created over.
        assert!(open_store_with(&logical, &ferro).is_err());
        assert!(!dir.path().join("s.fls").exists());
        let ferro = StoreOptions {
            unmigrated: Unmigrated::Migrate,
            ..ferro
        };
        // A ferro store of its own: the logical `.redb` name maps to the
        // sibling .fls, and an explicit .fls path needs no environment.
        let fresh = dir.path().join("f.redb");
        assert!(!store_exists_with(&fresh, Some("ferro")));
        drop(open_store_with(&fresh, &ferro).unwrap());
        let physical = dir.path().join("f.fls");
        assert!(physical.exists());
        assert!(store_exists_with(&fresh, Some("ferro")));
        assert!(
            store_exists_with(&fresh, None),
            "the default engine is ferro"
        );
        assert!(!store_exists_with(&fresh, Some("redb")));
        assert!(store_exists_with(&physical, None));
        // A bad environment value falls back to the path itself.
        assert!(store_exists_with(&logical, Some("bogus")));
        assert!(!store_exists_with(
            &dir.path().join("none.redb"),
            Some("bogus")
        ));
    }
}
