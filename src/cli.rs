//! The engine-independent core of the store tool commands, shared by `fl
//! store` and the `ferrostore` binary so the two never drift: `status`,
//! `checkpoint`, `compact`, `migrate` (with its dry run) and `verify` here;
//! `get`, `put` and `query` in [`data`]; `serve` in [`serve`](mod@serve);
//! `inspect` in [`inspect`](mod@inspect); `bench` in [`bench`](mod@bench).
//!
//! Each command is a function that takes explicit inputs (the store path, the
//! engine already chosen, the caller's [`Wording`]) and returns a typed report.
//! A report renders itself as text ([`StatusReport::text`] and the like) and
//! serialises as the command's `--json` object ([`to_json`]), so every program
//! built on this module (`fl store`, the `ferrostore` binary) prints the same
//! thing. Argument parsing, and how a program finds its store, stay with the
//! program; the arguments of the commands both programs share are declared
//! once, in `args` (feature `cli`).
//!
//! Only `put` and `serve` create a store (a new `.fls`, mode 0600), as the
//! caller's [`Create`] allows (`ferrostore`: only with `--create`, in a
//! directory that exists; `fl store`: the bound store, as every `fl` command
//! makes it), only when nothing is at its path, and only while holding the
//! store's migration lock: a `.redb` beside the missing `.fls` is refused,
//! never migrated, whether the path names the `.redb` or the `.fls`. A
//! path that names nothing is reported ([`status`]) or refused (every other
//! command), never opened. [`migrate`] creates the one `.fls` it is asked to; a
//! dry run and [`verify`] work on a private copy of the `.redb` and never write
//! the real files (opening a redb file rewrites its internal pages, so they do
//! not open the real one). `inspect` only reads. `bench` works on fresh stores
//! of its own unless it is told, twice, to use an existing one.
//!
//! Cost of the private copy: it is made in the [`Scratch`] directory the
//! caller provides (for `fl store`, the system temp directory: `$TMPDIR`, else
//! `/tmp`, which is often RAM-backed; for `ferrostore`, a private directory
//! beside the store, [`SiblingScratch`]), and it is made while this process
//! holds a shared lock on the `.redb`, which stops any writer (a hub or daemon
//! on redb) until the copy is done. `verify` makes one copy of the redb; a dry
//! run makes the copy and then a `.fls` of the same size. Each command first
//! checks that the filesystem has room (the size of each file plus 10%) and
//! refuses, with the scratch directory's [`Scratch::room_hint`], when it does
//! not. A real `migrate` copies nothing: it writes the new `.fls` beside the
//! `.redb` through [`migrate_locked`], which takes the same lock as an
//! automatic migration, so a ferro process that opens the store meanwhile
//! waits for it.
//!
//! `status` never opens a store that can write or migrate: FerroStore sizes are
//! read with [`ferro_stats_at`]. A non-empty `.redb` is opened read-only
//! (`RedbStore::open_read_only`, no retries): it takes a shared lock, writes and
//! creates nothing, and works on a 0400 file. A file that needs repair, or has
//! no store tables, is reported ("could not read it"), never repaired or
//! initialised.

use crate::migrate::{ParityReport, TableParity, migrate_verified, verify_redb_vs_ferrostore};
use crate::{
    AnyStats, BackendKind, COMPACT_FERRO_ONLY, FerroStore, RedbSharedLock, RedbStore, StoreOptions,
    Unmigrated, ferro_stats_at, is_redb_busy, migrate_locked, physical_path,
};
use fl_core::store::StoreError;
use serde::Serialize;
use serde::ser::SerializeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

// ---- errors ----

/// Why a command did not run, or did not finish. Its [`Display`](std::fmt::Display)
/// is the outermost message; [`source`](std::error::Error::source) is what
/// caused it, so a caller that prints the chain (`{:#}` in `anyhow`) prints
/// `context: cause: …`.
#[derive(Debug)]
pub enum Error {
    /// A refusal or failure stated in words, with nothing underneath.
    Refused(String),
    /// `context`, caused by `source`.
    Context {
        context: String,
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },
    /// A store error, as it is.
    Store(StoreError),
    /// An I/O error, as it is.
    Io(std::io::Error),
    /// A report that could not be written as JSON.
    Json(serde_json::Error),
}

impl Error {
    fn context(
        context: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Context {
            context: context.into(),
            source: Box::new(source),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused(msg) => f.write_str(msg),
            Self::Context { context, .. } => f.write_str(context),
            Self::Store(e) => e.fmt(f),
            Self::Io(e) => e.fmt(f),
            Self::Json(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Refused(_) => None,
            Self::Context { source, .. } => Some(source.as_ref()),
            Self::Store(e) => e.source(),
            Self::Io(e) => e.source(),
            Self::Json(e) => e.source(),
        }
    }
}

impl From<StoreError> for Error {
    fn from(e: StoreError) -> Self {
        Self::Store(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

macro_rules! refuse {
    ($($arg:tt)*) => {
        return Err($crate::cli::Error::Refused(format!($($arg)*)))
    };
}

// Declared after `refuse!`, which they use.
#[cfg(feature = "cli")]
pub mod args;
pub mod bench;
pub mod data;
pub mod inspect;
pub mod serve;

pub use bench::{BenchOptions, BenchReport, BenchTarget, Workload, bench};
pub use data::{
    Create, Credentials, GetReport, PutReport, Query, QueryReport, Target, get, open_target, put,
    query, read_doc,
};
pub use inspect::inspect;
pub use serve::{ServeOptions, Serving, serve};

// ---- what the caller supplies ----

/// How the calling program names itself in what a command prints.
#[derive(Debug, Clone, Copy)]
pub struct Wording<'a> {
    /// What a person types before a subcommand: `fl store`, `ferrostore`.
    pub command: &'a str,
    /// Where the `.redb` path comes from, named when a `.fls` path is given
    /// instead: for `fl store`, `the bound store, or --db <file>.redb`.
    pub redb_path_hint: &'a str,
}

/// A private, empty directory a command makes its copies in, and removes
/// with [`Scratch::close`]. The caller makes it, so it decides where (and the
/// module needs no temp-directory crate).
pub trait Scratch {
    fn path(&self) -> &Path;
    /// Remove the directory and everything in it.
    fn close(self) -> std::io::Result<()>;
    /// What a refusal for want of room tells the person to do: by default,
    /// to point `$TMPDIR` elsewhere (a directory in the system temp
    /// directory).
    fn room_hint(&self) -> &str {
        TMPDIR_HINT
    }
}

/// The default [`Scratch::room_hint`].
pub const TMPDIR_HINT: &str = "point $TMPDIR at a bigger filesystem";

/// A private (0700) directory beside a store file, for a program that keeps
/// its private copies on the store's own filesystem rather than in the
/// system temp directory (often RAM-backed). Removed by [`Scratch::close`],
/// or on drop when a command fails first.
#[derive(Debug)]
pub struct SiblingScratch {
    path: PathBuf,
    removed: bool,
}

impl SiblingScratch {
    /// A new directory `.<file name>.scratch-<pid>-<n>` in the directory of
    /// `store`, made mode 0700 and refused when the name exists.
    pub fn beside(store: &Path) -> Result<Self> {
        let (dir, name) = Self::place(store);
        Self::make(dir, &name)
    }

    /// The directory and name [`beside`](Self::beside) uses for `store`.
    fn place(store: &Path) -> (&Path, String) {
        let dir = match store.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => Path::new("."),
        };
        let name = store
            .file_name()
            .map_or_else(|| "store".into(), |n| n.to_string_lossy().into_owned());
        (dir, name)
    }

    /// Remove the directories [`beside`](Self::beside) made for `store` and
    /// that a process left behind because it was killed (SIGKILL, the OOM
    /// killer, a SIGINT during the copy) before it could remove them: every
    /// `.<file name>.scratch-<pid>-<n>` in the store's directory that is a
    /// directory (a symlink is never followed or removed), is owned by this
    /// user, and whose `<pid>` names no live process. Run at the start of
    /// `migrate` and `verify`. Nothing here stops the command: what could not
    /// be removed is returned, one warning line each.
    pub fn sweep_beside(store: &Path) -> Vec<String> {
        let (dir, name) = Self::place(store);
        sweep_stale(dir, &format!(".{name}.scratch-"))
    }

    /// A new directory `.ferrostore.scratch-<pid>-<n>` inside `dir`, made
    /// mode 0700 and refused when the name exists.
    pub fn within(dir: &Path) -> Result<Self> {
        Self::make(dir, "ferrostore")
    }

    fn make(dir: &Path, name: &str) -> Result<Self> {
        use std::os::unix::fs::DirBuilderExt;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos());
        let path = dir.join(format!(".{name}.scratch-{}-{nanos}", std::process::id()));
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&path)
            .map_err(|e| {
                Error::context(
                    format!("could not make a private directory in {}", dir.display()),
                    e,
                )
            })?;
        Ok(Self {
            path,
            removed: false,
        })
    }
}

/// Whether `pid` names a live process (`kill(pid, 0)` other than `ESRCH`; a
/// process of another user is alive too). A pid that cannot be one (0, or
/// past `i32`) is taken as alive, so nothing is removed for it.
pub(crate) fn pid_alive(pid: u32) -> bool {
    let Ok(pid) = libc::pid_t::try_from(pid) else {
        return true;
    };
    if pid <= 0 {
        return true;
    }
    // SAFETY: signal 0 sends nothing; it only checks that `pid` exists.
    if unsafe { libc::kill(pid, 0) } == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

/// The pid in `<pid>-<digits>`, the tail of a name made with
/// `<prefix><pid>-<n>`.
pub(crate) fn pid_of_tail(tail: &str) -> Option<u32> {
    let (pid, n) = tail.split_once('-')?;
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    (digits(pid) && digits(n)).then(|| pid.parse().ok())?
}

/// Remove each directory in `dir` named `<prefix><pid>-<n>`, owned by this
/// user, whose pid is not alive. See [`SiblingScratch::sweep_beside`].
fn sweep_stale(dir: &Path, prefix: &str) -> Vec<String> {
    use std::os::unix::fs::MetadataExt;
    let mut warnings = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            warnings.push(format!(
                "warning: could not look for stale scratch directories in {}: {e}",
                dir.display()
            ));
            return warnings;
        }
    };
    // SAFETY: `getuid` has no preconditions and cannot fail.
    let uid = unsafe { libc::getuid() };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(tail) = name.to_str().and_then(|n| n.strip_prefix(prefix)) else {
            continue;
        };
        let Some(pid) = pid_of_tail(tail) else {
            continue;
        };
        let path = entry.path();
        // Not followed: a symlink is not a directory here.
        let Ok(meta) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if !meta.file_type().is_dir() || meta.uid() != uid || pid_alive(pid) {
            continue;
        }
        if let Err(e) = std::fs::remove_dir_all(&path) {
            warnings.push(format!(
                "warning: could not remove the stale scratch directory {} (left by process \
                 {pid}, which is gone): {e}",
                path.display()
            ));
        }
    }
    warnings
}

impl Scratch for SiblingScratch {
    fn path(&self) -> &Path {
        &self.path
    }

    fn close(mut self) -> std::io::Result<()> {
        self.removed = true;
        std::fs::remove_dir_all(&self.path)
    }

    fn room_hint(&self) -> &str {
        "free space on the store's filesystem (the copy is made beside the store)"
    }
}

impl Drop for SiblingScratch {
    fn drop(&mut self) {
        if !self.removed {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

/// `report` as the command's pretty-printed JSON object, keys sorted.
pub fn to_json<T: Serialize>(report: &T) -> Result<String> {
    let value = serde_json::to_value(report).map_err(Error::Json)?;
    serde_json::to_string_pretty(&value).map_err(Error::Json)
}

/// What a command prints: `stdout` first, then `stderr`. Each holds whole
/// lines, each ending in a newline.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Output {
    pub stdout: String,
    pub stderr: String,
}

fn line(out: &mut String, text: impl AsRef<str>) {
    out.push_str(text.as_ref());
    out.push('\n');
}

fn shown<S: serde::Serializer>(path: &Path, s: S) -> std::result::Result<S::Ok, S::Error> {
    s.serialize_str(&path.display().to_string())
}

fn engine_name<S: serde::Serializer>(
    kind: &BackendKind,
    s: S,
) -> std::result::Result<S::Ok, S::Error> {
    s.serialize_str(kind.as_str())
}

// ---- status ----

/// The format version `status` reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatVersion {
    /// The version the file records.
    Known(u64),
    /// The engine has none to read yet (FerroStore): JSON `"n/a"`.
    NotAvailable,
    /// None read: no file, an empty file, none recorded, or unreadable. JSON `null`.
    Unknown,
}

impl Serialize for FormatVersion {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::Known(v) => s.serialize_u64(*v),
            Self::NotAvailable => s.serialize_str("n/a"),
            Self::Unknown => s.serialize_none(),
        }
    }
}

/// What `status` found. The JSON object has every field below except
/// `format_note`; `redb_beside_ferro` only when true, `sizes_note` only when set.
#[derive(Debug, Clone, Serialize)]
pub struct StatusReport {
    #[serde(serialize_with = "engine_name")]
    pub engine: BackendKind,
    #[serde(serialize_with = "shown")]
    pub logical_path: PathBuf,
    #[serde(serialize_with = "shown")]
    pub physical_path: PathBuf,
    pub exists: bool,
    pub size_bytes: Option<u64>,
    pub format_version: FormatVersion,
    /// Why the format version is what it is, when that needs saying. Text only.
    #[serde(skip)]
    pub format_note: Option<String>,
    pub live_bytes: Option<u64>,
    pub dead_bytes: Option<u64>,
    /// A lone `.redb` beside a ferro selection: the next open migrates it.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub redb_beside_ferro: bool,
    /// Why live and dead bytes are unknown for an existing `.fls`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sizes_note: Option<String>,
}

/// How every command here opens a store: on the engine `kind` the caller
/// chose, never on the environment (this library reads none), and refusing
/// rather than migrating a lone `.redb`.
pub(crate) fn engine_options(kind: BackendKind) -> StoreOptions {
    StoreOptions {
        backend: Some(kind),
        unmigrated: Unmigrated::Refuse,
        ..StoreOptions::default()
    }
}

/// The engine, files and format version of the store `path` names, on the
/// engine `kind`. Read-only: see the [module documentation](self). Because it
/// opens the redb read-only, `status` does not print the stderr warnings an
/// ordinary redb open prints (a `.fls` beside the `.redb`, loose file
/// permissions); it must change and say nothing beyond its report.
pub fn status(path: &Path, kind: BackendKind) -> StatusReport {
    let physical = physical_path(path, kind);
    let meta = std::fs::metadata(&physical).ok();
    let exists = meta.is_some();
    let mut size = meta.as_ref().map(std::fs::Metadata::len);
    // A lone `.redb` beside a ferro selection: `open_store` migrates it, so say so.
    let redb_beside = kind == BackendKind::Ferro && !exists && path != physical && path.exists();

    // FerroStore has no format version to read yet; redb's is read from the
    // file, which is opened only when it exists.
    let (format_version, format_note) = match (kind, exists) {
        (BackendKind::Ferro, _) => (
            FormatVersion::NotAvailable,
            Some("not available on ferro yet".into()),
        ),
        (BackendKind::Redb, false) => (FormatVersion::Unknown, None),
        // A 0-byte file is not a redb database yet; opening it would
        // initialise it, so report it without opening.
        (BackendKind::Redb, true) if size == Some(0) => {
            (FormatVersion::Unknown, Some("empty file".into()))
        }
        (BackendKind::Redb, true) => {
            // Opened for reading only (`open_read_only`): an ordinary open
            // rewrites redb's internal pages and may repair or initialise the
            // file, and status must change nothing, not even on a 0400 file.
            // No retries: a store another process holds is reported, not
            // waited on. A file that needs repair or is not an initialised
            // store is reported as unreadable; `fl` repairs it on its next
            // ordinary open.
            match RedbStore::open_read_only(path, 0).and_then(|s| s.format_version()) {
                Ok(Some(v)) => (FormatVersion::Known(v), None),
                Ok(None) => (FormatVersion::Unknown, Some("the file records none".into())),
                Err(e) => (
                    FormatVersion::Unknown,
                    Some(format!("could not read it: {e}")),
                ),
            }
        }
    };

    // FerroStore knows which bytes of its file are live and which an earlier
    // checkpoint left dead; redb does not say, and a missing store has none.
    // Read with `ferro_stats_at`: read-only, no lock, nothing created or
    // initialised, so status is safe beside a live store and on a 0400 file,
    // and a file that is not a store yet is reported, not turned into one.
    // The size comes from the same read, so the three figures agree.
    let (live_bytes, dead_bytes, sizes_note) = match (kind, exists) {
        (BackendKind::Ferro, true) => match ferro_stats_at(&physical) {
            Ok(st) => {
                size = Some(st.file_bytes);
                (st.live_bytes, st.dead_bytes, None)
            }
            Err(e) => (None, None, Some(format!("could not read them: {e}"))),
        },
        _ => (None, None, None),
    };

    StatusReport {
        engine: kind,
        logical_path: path.to_path_buf(),
        physical_path: physical,
        exists,
        size_bytes: size,
        format_version,
        format_note,
        live_bytes,
        dead_bytes,
        redb_beside_ferro: redb_beside,
        sizes_note,
    }
}

impl StatusReport {
    pub fn text(&self, w: &Wording) -> String {
        let mut out = String::new();
        line(
            &mut out,
            format!("engine:         {}", self.engine.as_str()),
        );
        line(
            &mut out,
            format!("logical path:   {}", self.logical_path.display()),
        );
        line(
            &mut out,
            format!("physical path:  {}", self.physical_path.display()),
        );
        line(
            &mut out,
            format!("exists:         {}", if self.exists { "yes" } else { "no" }),
        );
        match self.size_bytes {
            Some(n) => line(&mut out, format!("size:           {n} bytes")),
            None => line(&mut out, "size:           -"),
        }
        if self.engine == BackendKind::Ferro && self.exists {
            match (self.live_bytes, self.dead_bytes) {
                (Some(live), Some(dead)) => {
                    line(&mut out, format!("live:           {live} bytes"));
                    line(
                        &mut out,
                        format!(
                            "dead:           {dead} bytes (`{} compact` gives them back)",
                            w.command
                        ),
                    );
                }
                _ => line(
                    &mut out,
                    format!(
                        "live/dead:      - ({})",
                        self.sizes_note.as_deref().unwrap_or("unavailable")
                    ),
                ),
            }
        }
        let shown = match &self.format_version {
            FormatVersion::Known(n) => n.to_string(),
            FormatVersion::NotAvailable => "n/a".to_string(),
            FormatVersion::Unknown => "-".to_string(),
        };
        match &self.format_note {
            Some(note) => line(&mut out, format!("format version: {shown} ({note})")),
            None => line(&mut out, format!("format version: {shown}")),
        }
        if self.redb_beside_ferro {
            line(
                &mut out,
                format!(
                    "note: {} holds a redb store and the selected engine is ferro; \
                     the next command that opens it migrates it (or run `{} migrate`)",
                    self.logical_path.display(),
                    w.command
                ),
            );
        }
        out
    }
}

// ---- checkpoint ----

/// What `checkpoint` did.
#[derive(Debug, Clone, Serialize)]
pub struct CheckpointReport {
    #[serde(serialize_with = "engine_name")]
    pub engine: BackendKind,
    #[serde(serialize_with = "shown")]
    pub physical_path: PathBuf,
    /// False on redb, which has no write-ahead log to fold.
    pub checkpointed: bool,
}

/// Fold a FerroStore's write-ahead log into its file. redb has none: reported,
/// and left alone. A missing `.fls` is refused, never created.
pub fn checkpoint(path: &Path, kind: BackendKind, w: &Wording) -> Result<CheckpointReport> {
    let physical = physical_path(path, kind);
    if kind == BackendKind::Redb {
        return Ok(CheckpointReport {
            engine: kind,
            physical_path: physical,
            checkpointed: false,
        });
    }
    // The physical file, not `store_exists`: a lone `.redb` would be migrated
    // by the open below, and a checkpoint must not do that.
    if !physical.exists() {
        refuse!(
            "no store at {}; `{} checkpoint` does not create one",
            physical.display(),
            w.command
        );
    }
    let store = crate::open_store_with(path, &engine_options(kind))?;
    let Some(ferro) = store.as_ferro() else {
        refuse!("the store at {} is not a FerroStore", physical.display());
    };
    ferro.checkpoint()?;
    Ok(CheckpointReport {
        engine: kind,
        physical_path: physical,
        checkpointed: true,
    })
}

impl CheckpointReport {
    pub fn text(&self) -> String {
        if self.checkpointed {
            format!("checkpointed {}\n", self.physical_path.display())
        } else {
            "redb has no checkpoint; nothing to do\n".to_string()
        }
    }
}

// ---- compact ----

/// The size accounting of a store file, as `compact` reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Sizes {
    pub file_bytes: u64,
    pub live_bytes: Option<u64>,
    pub dead_bytes: Option<u64>,
}

impl From<&AnyStats> for Sizes {
    fn from(s: &AnyStats) -> Self {
        Self {
            file_bytes: s.file_bytes,
            live_bytes: s.live_bytes,
            dead_bytes: s.dead_bytes,
        }
    }
}

/// What `compact` did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompactReport {
    /// The engine is not compacted (redb); `reason` says so, built on
    /// [`COMPACT_FERRO_ONLY`].
    NotCompacted {
        engine: BackendKind,
        physical_path: PathBuf,
        reason: String,
    },
    /// The store was compacted. `after` is `None` when the reading after it
    /// failed: a failed second reading is not a failed compaction.
    Compacted {
        physical_path: PathBuf,
        before: Sizes,
        after: Option<Sizes>,
    },
}

/// JSON: `command`, `engine`, `physical_path`, `compacted`, then `reason`
/// (not compacted) or `before` and `after` (`null` when unknown).
impl Serialize for CompactReport {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::NotCompacted {
                engine,
                physical_path,
                reason,
            } => {
                let mut m = s.serialize_map(Some(5))?;
                m.serialize_entry("command", "compact")?;
                m.serialize_entry("compacted", &false)?;
                m.serialize_entry("engine", engine.as_str())?;
                m.serialize_entry("physical_path", &physical_path.display().to_string())?;
                m.serialize_entry("reason", reason)?;
                m.end()
            }
            Self::Compacted {
                physical_path,
                before,
                after,
            } => {
                let mut m = s.serialize_map(Some(6))?;
                m.serialize_entry("after", after)?;
                m.serialize_entry("before", before)?;
                m.serialize_entry("command", "compact")?;
                m.serialize_entry("compacted", &true)?;
                m.serialize_entry("engine", BackendKind::Ferro.as_str())?;
                m.serialize_entry("physical_path", &physical_path.display().to_string())?;
                m.end()
            }
        }
    }
}

/// Rewrite a FerroStore as one snapshot and an empty log. Mirrors
/// [`checkpoint`]: redb is reported and left alone, a missing `.fls` is
/// refused (a lone `.redb` is not migrated just to be compacted), and the
/// physical file is the one opened.
pub fn compact(path: &Path, kind: BackendKind, w: &Wording) -> Result<CompactReport> {
    let physical = physical_path(path, kind);
    if kind == BackendKind::Redb {
        return Ok(CompactReport::NotCompacted {
            engine: kind,
            physical_path: physical,
            reason: format!("{COMPACT_FERRO_ONLY}, nothing to do"),
        });
    }
    if !physical.exists() {
        refuse!(
            "no store at {}; `{} compact` does not create one",
            physical.display(),
            w.command
        );
    }
    let store = crate::open_store_with(path, &engine_options(kind))?;
    if store.as_ferro().is_none() {
        refuse!("the store at {} is not a FerroStore", physical.display());
    }
    let before = store.stats()?;
    store.compact()?;
    // The store is compacted from here on. A failed second reading is not a
    // failed compaction: report the compaction and leave `after` unknown.
    let after = store.stats().ok();
    Ok(CompactReport::Compacted {
        physical_path: physical,
        before: Sizes::from(&before),
        after: after.as_ref().map(Sizes::from),
    })
}

impl CompactReport {
    /// The text report; `-` where `after` is unknown.
    pub fn text(&self) -> String {
        match self {
            Self::NotCompacted { reason, .. } => format!("{reason}\n"),
            Self::Compacted {
                physical_path,
                before,
                after,
            } => {
                let dash = || "-".to_string();
                let dead = |s: &Sizes| s.dead_bytes.map_or_else(dash, |d| d.to_string());
                format!(
                    "compacted {}\nfile bytes: {} -> {}\ndead bytes: {} -> {}\n",
                    physical_path.display(),
                    before.file_bytes,
                    after.map_or_else(dash, |a| a.file_bytes.to_string()),
                    dead(before),
                    after.as_ref().map_or_else(dash, dead),
                )
            }
        }
    }
}

// ---- migrate and verify ----

/// The `.redb` and `.fls` of one store. Made only by [`migrate_pair`] and
/// [`verify_pair`], which check the files; the fields are read through
/// [`Pair::redb`] and [`Pair::fls`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pair {
    redb: PathBuf,
    fls: PathBuf,
}

impl Pair {
    /// The `.redb` file.
    pub fn redb(&self) -> &Path {
        &self.redb
    }

    /// The `.fls` file.
    pub fn fls(&self) -> &Path {
        &self.fls
    }
}

/// The `.redb` and `.fls` of the store named `path`, which must name the
/// `.redb`: a path that is itself a `.fls` names no redb to migrate or compare.
fn pair(path: &Path, what: &str, w: &Wording) -> Result<Pair> {
    let fls = physical_path(path, BackendKind::Ferro);
    if fls == path {
        refuse!(
            "{} names a .fls file; `{} {what}` takes the store's .redb path ({})",
            path.display(),
            w.command,
            w.redb_path_hint
        );
    }
    Ok(Pair {
        redb: path.to_path_buf(),
        fls,
    })
}

/// A file or directory that exists, symlinks not followed: a dangling link is
/// something too.
fn present(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok()
}

const BUSY: &str = "is held by another process (a hub or daemon still running on redb?); stop it \
                    and run this again";

/// Refuse in words when another process holds the redb; any other failure to
/// read it is reported as it is. The lock is released when the guard drops.
fn lock_redb(redb: &Path) -> Result<RedbSharedLock> {
    RedbSharedLock::acquire(redb).map_err(|e| {
        if is_redb_busy(&e) {
            Error::Refused(format!("{} {BUSY}", redb.display()))
        } else {
            Error::context(format!("could not read {}", redb.display()), e)
        }
    })
}

/// The line printed when a clean migration could not make the kept redb read-only.
fn still_writable_notice(fls: &Path, redb: &Path) -> String {
    format!(
        "clean: {} is the store; {} is kept but is still writable (its mode could not be \
         set to 0400): make it read-only with chmod 0400, or a write to it is lost",
        fls.display(),
        redb.display()
    )
}

/// Refuse, in words ending with `hint`, when `free` bytes cannot hold
/// `copies` files of `size` bytes plus 10% each.
fn ensure_room(dir: &Path, free: u64, size: u64, copies: u64, hint: &str) -> Result<()> {
    let need = copies.saturating_mul(size.saturating_add(size / 10));
    if free < need {
        refuse!(
            "not enough free space in {} for the private copy: {free} bytes free, {need} needed \
             ({copies} file(s) of {size} bytes, plus 10%); {hint}",
            dir.display()
        );
    }
    Ok(())
}

/// A writable copy of `redb` as `<dir>/<name>`, made while this process holds
/// a shared lock on the original, so no writer changes it mid-copy (a hub or
/// daemon on redb waits for the copy). The original is only read. `copies` is
/// how many files of that size the command will keep in `dir` in all; the
/// filesystem must have room for them, plus 10%, or nothing is copied (the
/// refusal ends with `hint`).
fn copy_redb(redb: &Path, dir: &Path, copies: u64, hint: &str) -> Result<PathBuf> {
    use std::os::unix::fs::PermissionsExt;
    let _lock = lock_redb(redb)?;
    let size = std::fs::metadata(redb)
        .map_err(|e| Error::context(format!("could not read {}", redb.display()), e))?
        .len();
    // A filesystem that cannot report its free space is not a reason to refuse.
    if let Ok(free) = crate::private::available_bytes(dir) {
        ensure_room(dir, free, size, copies, hint)?;
    }
    let Some(name) = redb.file_name() else {
        refuse!("{} has no file name", redb.display());
    };
    let copy = dir.join(name);
    std::fs::copy(redb, &copy).map_err(|e| {
        Error::context(
            format!("could not copy {} to {}", redb.display(), copy.display()),
            e,
        )
    })?;
    std::fs::set_permissions(&copy, std::fs::Permissions::from_mode(0o600))?;
    Ok(copy)
}

/// Whether nobody may write `path` by its mode bits (what `stat` says; root
/// ignores the bits, which is why the mode is read, not assumed).
fn mode_is_read_only(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path).is_ok_and(|m| m.permissions().mode() & 0o222 == 0)
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// One table or view of a parity report, as JSON.
#[derive(Serialize)]
struct ParityRow<'a> {
    name: &'a str,
    redb_rows: u64,
    ferro_rows: u64,
    redb_hash: String,
    ferro_hash: String,
    clean: bool,
}

impl<'a> From<&'a TableParity> for ParityRow<'a> {
    fn from(p: &'a TableParity) -> Self {
        Self {
            name: &p.table,
            redb_rows: p.redb_rows,
            ferro_rows: p.ferro_rows,
            redb_hash: hex(&p.redb_hash),
            ferro_hash: hex(&p.ferro_hash),
            clean: p.is_clean(),
        }
    }
}

/// The keys a parity report contributes to a `migrate` or `verify` object.
fn parity_entries<M: SerializeMap>(
    m: &mut M,
    report: &ParityReport,
    elapsed: Duration,
) -> std::result::Result<(), M::Error> {
    m.serialize_entry("clean", &report.is_clean())?;
    m.serialize_entry("elapsed_ms", &(elapsed.as_millis() as u64))?;
    m.serialize_entry(
        "mismatches",
        &report
            .mismatches()
            .iter()
            .map(|p| p.table.as_str())
            .collect::<Vec<_>>(),
    )?;
    m.serialize_entry(
        "tables",
        &report
            .tables
            .iter()
            .map(ParityRow::from)
            .collect::<Vec<_>>(),
    )?;
    m.serialize_entry(
        "trait_views",
        &report
            .trait_views
            .iter()
            .map(ParityRow::from)
            .collect::<Vec<_>>(),
    )?;
    Ok(())
}

/// Per-table rows, then a line for the views.
fn report_text(out: &mut String, report: &ParityReport) {
    let width = report
        .tables
        .iter()
        .map(|p| p.table.len())
        .max()
        .unwrap_or(5)
        .max(5);
    line(
        out,
        format!(
            "{:<width$}  {:>10}  {:>10}  status",
            "table", "redb", "ferro"
        ),
    );
    for p in &report.tables {
        line(
            out,
            format!(
                "{:<width$}  {:>10}  {:>10}  {}",
                p.table,
                p.redb_rows,
                p.ferro_rows,
                if p.is_clean() { "ok" } else { "MISMATCH" }
            ),
        );
    }
    let bad = report.trait_views.iter().filter(|p| !p.is_clean()).count();
    line(
        out,
        format!(
            "views: {} compared, {} differ",
            report.trait_views.len(),
            bad
        ),
    );
}

fn mismatches_text(out: &mut String, report: &ParityReport) {
    line(out, "mismatches:");
    for p in report.mismatches() {
        line(
            out,
            format!(
                "  {}: redb {} rows {}, ferro {} rows {}",
                p.table,
                p.redb_rows,
                &hex(&p.redb_hash)[..12],
                p.ferro_rows,
                &hex(&p.ferro_hash)[..12],
            ),
        );
    }
}

/// The pair `migrate` would write, checked: `path` names an existing `.redb`
/// file and no `.fls` (or anything else) is beside it.
pub fn migrate_pair(path: &Path, w: &Wording) -> Result<Pair> {
    let pair = pair(path, "migrate", w)?;
    if !pair.redb.is_file() {
        refuse!(
            "no redb store at {}; `{} migrate` does not create one",
            pair.redb.display(),
            w.command
        );
    }
    if present(&pair.fls) {
        refuse!(
            "{} already exists: this store is already migrated (`{} verify` compares the \
             pair); `{} migrate` only creates a new .fls",
            pair.fls.display(),
            w.command,
            w.command
        );
    }
    Ok(pair)
}

/// What `migrate` did.
#[derive(Debug, Clone)]
pub struct MigrateReport {
    pub pair: Pair,
    pub dry_run: bool,
    pub parity: ParityReport,
    pub elapsed: Duration,
    /// The kept `.redb`'s real mode is read-only: false on a dry run, on a
    /// migration that did not verify, and when the chmod failed.
    pub redb_kept_read_only: bool,
}

/// JSON: the parity keys (`clean`, `elapsed_ms`, `mismatches`, `tables`,
/// `trait_views`), then `command`, `dry_run`, `redb`, `fls` (`null` unless a
/// real migration verified) and `redb_kept_read_only`.
impl Serialize for MigrateReport {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let clean = self.parity.is_clean();
        let mut m = s.serialize_map(Some(10))?;
        parity_entries(&mut m, &self.parity, self.elapsed)?;
        m.serialize_entry("command", "migrate")?;
        m.serialize_entry("dry_run", &self.dry_run)?;
        m.serialize_entry(
            "fls",
            &(!self.dry_run && clean).then(|| self.pair.fls.display().to_string()),
        )?;
        m.serialize_entry("redb", &self.pair.redb.display().to_string())?;
        m.serialize_entry("redb_kept_read_only", &self.redb_kept_read_only)?;
        m.end()
    }
}

/// Copy the `.redb` of `pair` (from [`migrate_pair`]) into its new `.fls`,
/// verified, through [`migrate_locked`]. With `dry_run`, migrate a private copy
/// made in the directory `scratch` returns, report, and remove the copy; the
/// real files are not written. `scratch` is called only for a dry run.
pub fn migrate<S: Scratch>(
    pair: &Pair,
    dry_run: bool,
    scratch: impl FnOnce() -> Result<S>,
) -> Result<MigrateReport> {
    let Pair { redb, fls } = pair;
    let started = Instant::now();
    let (report, scratch) = if dry_run {
        let dir = scratch()?;
        let copy = copy_redb(redb, dir.path(), 2, dir.room_hint())?;
        let copy_fls = physical_path(&copy, BackendKind::Ferro);
        let report = migrate_verified(&copy, &copy_fls)
            .map_err(|e| Error::context(format!("dry run on a copy of {}", redb.display()), e))?;
        (report, Some(dir))
    } else {
        // Ask once, so a held redb is refused now and not after the
        // migration's own wait. Only "held" is refused here; a file that
        // cannot be read for another reason is the migration's to report (it
        // also repairs a file left by a crash). The guard is released before
        // the migration opens the file.
        if let Err(e) = RedbSharedLock::acquire(redb)
            && is_redb_busy(&e)
        {
            refuse!("{} {BUSY}", redb.display());
        }
        let report = migrate_locked(redb, fls).map_err(|e| {
            if is_redb_busy(&e) {
                Error::Refused(format!("{} {BUSY}", redb.display()))
            } else {
                Error::context(format!("could not migrate {}", redb.display()), e)
            }
        })?;
        (report, None)
    };
    let elapsed = started.elapsed();
    // Delete the scratch copy before reporting.
    if let Some(dir) = scratch {
        dir.close()
            .map_err(|e| Error::context("could not remove the dry-run copy", e))?;
    }
    // The file's real mode: a chmod that failed leaves it writable.
    let redb_kept_read_only = !dry_run && report.is_clean() && mode_is_read_only(redb);
    Ok(MigrateReport {
        pair: pair.clone(),
        dry_run,
        parity: report,
        elapsed,
        redb_kept_read_only,
    })
}

impl MigrateReport {
    pub fn is_clean(&self) -> bool {
        self.parity.is_clean()
    }

    /// 0 when the migration verified, 1 when it did not.
    pub fn exit_code(&self) -> i32 {
        i32::from(!self.is_clean())
    }

    pub fn text(&self, w: &Wording) -> Output {
        let Pair { redb, fls } = &self.pair;
        let mut out = Output::default();
        let stdout = &mut out.stdout;
        if self.dry_run {
            line(
                stdout,
                format!(
                    "dry run: migrated a private copy of {}; no real file was written",
                    redb.display()
                ),
            );
        } else {
            line(
                stdout,
                format!("migrate {} -> {}", redb.display(), fls.display()),
            );
        }
        report_text(stdout, &self.parity);
        line(stdout, format!("time: {:.2}s", self.elapsed.as_secs_f64()));
        if self.is_clean() {
            if self.dry_run {
                line(
                    stdout,
                    format!(
                        "clean: a real `{} migrate` would succeed; the copy was deleted",
                        w.command
                    ),
                );
            } else if self.redb_kept_read_only {
                line(
                    stdout,
                    format!(
                        "clean: {} is the store; {} is kept, read-only (0400)",
                        fls.display(),
                        redb.display()
                    ),
                );
            } else {
                line(stdout, still_writable_notice(fls, redb));
            }
            return out;
        }
        mismatches_text(&mut out.stderr, &self.parity);
        line(
            &mut out.stderr,
            format!(
                "not clean: no {} was created and {} was not made read-only",
                fls.display(),
                redb.display()
            ),
        );
        out
    }
}

/// The pair `verify` compares, checked: an existing `.redb` file and an
/// existing `.fls` file.
pub fn verify_pair(path: &Path, w: &Wording) -> Result<Pair> {
    let pair = pair(path, "verify", w)?;
    if !pair.redb.is_file() {
        refuse!(
            "no redb store at {}; `{} verify` compares a .redb with its .fls",
            pair.redb.display(),
            w.command
        );
    }
    if !pair.fls.is_file() {
        refuse!(
            "no FerroStore at {}; `{} verify` compares a .redb with its .fls \
             (`{} migrate` makes one)",
            pair.fls.display(),
            w.command,
            w.command
        );
    }
    Ok(pair)
}

/// What `verify` found.
#[derive(Debug, Clone)]
pub struct VerifyReport {
    pub pair: Pair,
    pub parity: ParityReport,
    pub elapsed: Duration,
}

/// JSON: the parity keys, then `command`, `redb` and `fls`.
impl Serialize for VerifyReport {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(8))?;
        parity_entries(&mut m, &self.parity, self.elapsed)?;
        m.serialize_entry("command", "verify")?;
        m.serialize_entry("fls", &self.pair.fls.display().to_string())?;
        m.serialize_entry("redb", &self.pair.redb.display().to_string())?;
        m.end()
    }
}

/// Compare the `.redb` and `.fls` of `pair` (from [`verify_pair`]), table by
/// table and view by view. The redb is compared as a private copy made in the
/// directory `scratch` returns, removed before this returns: opening it for
/// the comparison would rewrite the real file, which a migration left
/// read-only.
pub fn verify<S: Scratch>(
    pair: &Pair,
    scratch: impl FnOnce() -> Result<S>,
) -> Result<VerifyReport> {
    let Pair { redb, fls } = pair;
    let started = Instant::now();
    let dir = scratch()?;
    let copy = copy_redb(redb, dir.path(), 1, dir.room_hint())?;
    let report = {
        let src = RedbStore::open_with_retries(&copy, 0).map_err(|e| {
            Error::context(format!("could not open a copy of {}", redb.display()), e)
        })?;
        let dst = FerroStore::open(fls)
            .map_err(|e| Error::context(format!("could not open {}", fls.display()), e))?;
        verify_redb_vs_ferrostore(&src, &dst).map_err(|e| {
            Error::context(
                format!(
                    "could not compare {} with {}",
                    redb.display(),
                    fls.display()
                ),
                e,
            )
        })?
    };
    dir.close()
        .map_err(|e| Error::context("could not remove the verify copy", e))?;
    let elapsed = started.elapsed();
    Ok(VerifyReport {
        pair: pair.clone(),
        parity: report,
        elapsed,
    })
}

impl VerifyReport {
    pub fn is_clean(&self) -> bool {
        self.parity.is_clean()
    }

    /// 0 when every table and view matches, 1 when one differs.
    pub fn exit_code(&self) -> i32 {
        i32::from(!self.is_clean())
    }

    pub fn text(&self) -> Output {
        let Pair { redb, fls } = &self.pair;
        let mut out = Output::default();
        line(
            &mut out.stdout,
            format!("verify {} against {}", redb.display(), fls.display()),
        );
        report_text(&mut out.stdout, &self.parity);
        line(
            &mut out.stdout,
            format!("time: {:.2}s", self.elapsed.as_secs_f64()),
        );
        if self.is_clean() {
            line(&mut out.stdout, "clean: every table and view matches");
            return out;
        }
        mismatches_text(&mut out.stderr, &self.parity);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    /// The pid of a child that has exited and been reaped: no live process.
    pub(crate) fn dead_pid() -> u32 {
        let mut child = std::process::Command::new("true").spawn().unwrap();
        let pid = child.id();
        child.wait().unwrap();
        assert!(!pid_alive(pid), "pid {pid} was reused");
        pid
    }

    #[test]
    fn a_killed_commands_scratch_directory_is_swept_and_a_live_ones_is_not() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("s.redb");
        let dead = dead_pid();
        let me = std::process::id();
        assert!(pid_alive(me));
        let stale = dir.path().join(format!(".s.redb.scratch-{dead}-1"));
        let live = dir.path().join(format!(".s.redb.scratch-{me}-2"));
        let other_store = dir.path().join(format!(".t.redb.scratch-{dead}-3"));
        for d in [&stale, &live, &other_store] {
            std::fs::create_dir(d).unwrap();
        }
        std::fs::write(stale.join("copy.redb"), b"x").unwrap();
        // A symlink and a file with a stale name are not removed, nor what
        // the link points at.
        let target = dir.path().join("kept");
        std::fs::create_dir(&target).unwrap();
        std::fs::write(target.join("f"), b"y").unwrap();
        let link = dir.path().join(format!(".s.redb.scratch-{dead}-4"));
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let file = dir.path().join(format!(".s.redb.scratch-{dead}-5"));
        std::fs::write(&file, b"z").unwrap();

        assert!(SiblingScratch::sweep_beside(&store).is_empty());
        assert!(!stale.exists(), "the dead process's copy was kept");
        assert!(live.is_dir(), "a live process's copy was removed");
        assert!(other_store.is_dir(), "another store's copy was removed");
        assert!(std::fs::symlink_metadata(&link).is_ok());
        assert!(target.join("f").exists());
        assert!(file.exists());
        // A real scratch directory of this process is left alone too.
        let mine = SiblingScratch::beside(&store).unwrap();
        assert!(SiblingScratch::sweep_beside(&store).is_empty());
        assert!(mine.path().is_dir());
        mine.close().unwrap();
    }

    #[test]
    fn a_pid_is_read_from_a_scratch_name_tail() {
        assert_eq!(pid_of_tail("123-456"), Some(123));
        assert_eq!(pid_of_tail("123-"), None);
        assert_eq!(pid_of_tail("x-1"), None);
        assert_eq!(pid_of_tail("1-2-3"), None);
        assert!(pid_alive(0), "pid 0 is never swept");
    }

    #[test]
    fn ensure_room_needs_the_size_plus_ten_percent_per_copy() {
        let dir = Path::new("/scratch");
        // 1000 bytes + 10% = 1100 per copy.
        assert!(ensure_room(dir, 1100, 1000, 1, TMPDIR_HINT).is_ok());
        assert!(ensure_room(dir, 2200, 1000, 2, TMPDIR_HINT).is_ok());
        let err = ensure_room(dir, 1099, 1000, 1, TMPDIR_HINT)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not enough free space in /scratch"), "{err}");
        assert!(err.contains("1100 needed"), "{err}");
        assert!(err.contains("$TMPDIR"), "{err}");
        assert!(!err.contains("  "), "a run of spaces in the message: {err}");
        assert!(
            err.contains("1100 needed (1 file(s) of 1000 bytes, plus 10%)"),
            "{err}"
        );
        let err = ensure_room(dir, 2199, 1000, 2, TMPDIR_HINT)
            .unwrap_err()
            .to_string();
        assert!(err.contains("2200 needed"), "{err}");
        // An empty file needs nothing; sizes near u64::MAX do not wrap.
        assert!(ensure_room(dir, 0, 0, 2, TMPDIR_HINT).is_ok());
        assert!(ensure_room(dir, u64::MAX - 1, u64::MAX, 2, TMPDIR_HINT).is_err());
    }

    #[test]
    fn the_still_writable_notice_has_no_runs_of_spaces() {
        let text = still_writable_notice(Path::new("/d/s.fls"), Path::new("/d/s.redb"));
        assert!(
            !text.contains("  "),
            "a run of spaces in the message: {text}"
        );
        assert!(
            text.contains("(its mode could not be set to 0400): make it read-only"),
            "{text}"
        );
        assert!(
            text.starts_with("clean: /d/s.fls is the store; /d/s.redb is kept"),
            "{text}"
        );
    }

    #[test]
    fn copy_redb_checks_the_scratch_space_before_it_copies() {
        let src = tempfile::tempdir().unwrap();
        let redb = src.path().join("s.redb");
        drop(RedbStore::open(&redb).unwrap());
        let scratch = tempfile::tempdir().unwrap();
        // Room enough on a test machine for one copy: it is made, writable.
        let copy = copy_redb(&redb, scratch.path(), 1, TMPDIR_HINT).unwrap();
        assert_eq!(
            std::fs::metadata(&copy).unwrap().permissions().mode() & 0o777,
            0o600
        );
        // More copies than any filesystem has room for: refused, nothing copied.
        let other = tempfile::tempdir().unwrap();
        let err = copy_redb(&redb, other.path(), u64::MAX / 4, TMPDIR_HINT)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not enough free space"), "{err}");
        assert_eq!(std::fs::read_dir(other.path()).unwrap().count(), 0);
    }

    #[test]
    fn the_kept_redb_is_read_only_by_its_real_mode() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("s.redb");
        std::fs::write(&file, b"x").unwrap();
        for (mode, read_only) in [(0o600, false), (0o644, false), (0o400, true), (0o444, true)] {
            std::fs::set_permissions(&file, std::fs::Permissions::from_mode(mode)).unwrap();
            assert_eq!(mode_is_read_only(&file), read_only, "{mode:o}");
        }
        assert!(!mode_is_read_only(&dir.path().join("missing")));
    }

    fn sizes(file: u64, dead: u64) -> Sizes {
        Sizes {
            file_bytes: file,
            live_bytes: Some(file - dead),
            dead_bytes: Some(dead),
        }
    }

    #[test]
    fn a_compaction_whose_second_reading_failed_is_reported_with_after_unknown() {
        let compacted = |after| CompactReport::Compacted {
            physical_path: PathBuf::from("/s/fl.fls"),
            before: sizes(1000, 600),
            after,
        };
        let report = compacted(None);
        let v = serde_json::to_value(&report).unwrap();
        assert_eq!(v["compacted"], true);
        assert_eq!(v["before"]["dead_bytes"], 600);
        assert_eq!(v["after"], serde_json::Value::Null);
        let text = report.text();
        assert!(text.contains("file bytes: 1000 -> -"), "{text}");
        assert!(text.contains("dead bytes: 600 -> -"), "{text}");
        let report = compacted(Some(sizes(400, 0)));
        let v = serde_json::to_value(&report).unwrap();
        assert_eq!(v["after"]["file_bytes"], 400);
        assert!(report.text().contains("dead bytes: 600 -> 0"));
    }

    /// The tests below only tell an engine chosen by `kind` from one chosen by
    /// the environment when `FL_STORE_BACKEND` is unset (the default is then
    /// ferro). They never set it; a harness that did makes them skip.
    fn env_is_unset() -> bool {
        let value = std::env::var_os(crate::BACKEND_ENV);
        if value.is_some() {
            eprintln!("skipped: {} is set in the test process", crate::BACKEND_ENV);
        }
        value.is_none()
    }

    #[test]
    fn every_command_opens_on_the_callers_engine_and_refuses_to_migrate() {
        for kind in [BackendKind::Redb, BackendKind::Ferro] {
            let options = engine_options(kind);
            assert_eq!(options.backend, Some(kind));
            assert_eq!(options.unmigrated, Unmigrated::Refuse);
        }
    }

    #[test]
    fn status_on_redb_neither_migrates_nor_chmods_with_the_environment_unset() {
        if !env_is_unset() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let redb = dir.path().join("s.redb");
        drop(RedbStore::open(&redb).unwrap());
        std::fs::set_permissions(&redb, std::fs::Permissions::from_mode(0o600)).unwrap();
        let report = status(&redb, BackendKind::Redb);
        assert_eq!(report.engine, BackendKind::Redb);
        assert!(report.exists);
        assert!(
            matches!(report.format_version, FormatVersion::Known(_)),
            "{:?} {:?}",
            report.format_version,
            report.format_note
        );
        assert!(!dir.path().join("s.fls").exists(), "status made a .fls");
        assert_eq!(
            std::fs::metadata(&redb).unwrap().permissions().mode() & 0o777,
            0o600,
            "status changed the .redb's mode"
        );
    }

    #[test]
    fn status_on_redb_changes_no_byte_and_no_mtime_and_works_on_a_0400_file() {
        if !env_is_unset() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let redb = dir.path().join("s.redb");
        drop(RedbStore::open(&redb).unwrap());
        std::fs::set_permissions(&redb, std::fs::Permissions::from_mode(0o400)).unwrap();
        let before = (
            std::fs::read(&redb).unwrap(),
            std::fs::metadata(&redb).unwrap().modified().unwrap(),
        );
        let report = status(&redb, BackendKind::Redb);
        assert!(
            matches!(report.format_version, FormatVersion::Known(_)),
            "{:?} {:?}",
            report.format_version,
            report.format_note
        );
        let after = (
            std::fs::read(&redb).unwrap(),
            std::fs::metadata(&redb).unwrap().modified().unwrap(),
        );
        assert!(before.0 == after.0, "status changed the .redb's bytes");
        assert_eq!(before.1, after.1, "status changed the .redb's mtime");
        assert_eq!(
            std::fs::metadata(&redb).unwrap().permissions().mode() & 0o777,
            0o400
        );
    }

    #[test]
    fn status_on_a_redb_without_the_store_tables_reports_it_and_does_not_initialise_it() {
        if !env_is_unset() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let redb = dir.path().join("s.redb");
        // A valid redb database that is not a store: no tables at all.
        drop(redb::Database::create(&redb).unwrap());
        let before = std::fs::read(&redb).unwrap();
        let report = status(&redb, BackendKind::Redb);
        assert!(
            matches!(report.format_version, FormatVersion::Unknown),
            "{:?}",
            report.format_version
        );
        let note = report.format_note.expect("the report says why");
        assert!(note.contains("could not read it"), "{note}");
        assert!(
            std::fs::read(&redb).unwrap() == before,
            "status wrote the file"
        );
    }

    #[test]
    fn checkpoint_and_compact_open_the_ferro_file_the_caller_named() {
        if !env_is_unset() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let redb = dir.path().join("s.redb");
        let fls = dir.path().join("s.fls");
        drop(RedbStore::open(&redb).unwrap());
        drop(FerroStore::open(&fls).unwrap());
        std::fs::set_permissions(&redb, std::fs::Permissions::from_mode(0o600)).unwrap();
        let w = Wording {
            command: "t",
            redb_path_hint: "h",
        };
        assert!(
            checkpoint(&redb, BackendKind::Ferro, &w)
                .unwrap()
                .checkpointed
        );
        assert!(matches!(
            compact(&redb, BackendKind::Ferro, &w).unwrap(),
            CompactReport::Compacted { .. }
        ));
        assert_eq!(
            std::fs::metadata(&redb).unwrap().permissions().mode() & 0o777,
            0o600
        );
        // A lone `.redb` is refused, not migrated, whatever the environment.
        let lone = dir.path().join("lone.redb");
        drop(RedbStore::open(&lone).unwrap());
        assert!(checkpoint(&lone, BackendKind::Ferro, &w).is_err());
        assert!(compact(&lone, BackendKind::Ferro, &w).is_err());
        assert!(!dir.path().join("lone.fls").exists());
    }

    #[test]
    fn a_pair_is_read_through_its_getters() {
        let w = Wording {
            command: "t",
            redb_path_hint: "h",
        };
        let err = migrate_pair(Path::new("/d/s.fls"), &w)
            .unwrap_err()
            .to_string();
        assert!(err.contains("names a .fls file"), "{err}");
        let p = pair(Path::new("/d/s.redb"), "migrate", &w).unwrap();
        assert_eq!(p.redb(), Path::new("/d/s.redb"));
        assert_eq!(p.fls(), Path::new("/d/s.fls"));
    }

    #[test]
    fn an_error_keeps_its_cause_in_the_chain() {
        let io = std::io::Error::other("disk on fire");
        let e = Error::context("could not remove the verify copy", io);
        assert_eq!(e.to_string(), "could not remove the verify copy");
        let cause = std::error::Error::source(&e).unwrap().to_string();
        assert_eq!(cause, "disk on fire");
        assert!(std::error::Error::source(&Error::Refused("no".into())).is_none());
    }
}
