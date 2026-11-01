//! Private permissions for a store: what `fl` itself creates is owner-only, and what it did not
//! create is never changed.
//!
//! A store holds hub pairing codes, session hashes and other records, so the file `fl` creates is
//! mode 0600 and a store directory `fl` creates is mode 0700, whatever the umask says. The rule
//! has two halves, and the second matters as much as the first:
//!
//! * [`create_private_dir_all`] and the file creation in [`RedbStore::open`](crate::RedbStore::open)
//!   give owner-only modes to what they create, and to nothing else. An existing directory or file
//!   (`~/.local/share`, a project directory, `/tmp`, a store made by an older `fl`) keeps its mode.
//! * An existing store that others can reach is only reported: [`warn_if_loose`] prints the path,
//!   its mode and the `chmod` that fixes it, once per path per process. `fl` never runs that
//!   `chmod` for you: the store may be shared on purpose, and a surprise mode change is worse than
//!   a reminder.
//!
//! The report is for a person, so it is quiet by default: it prints only when stderr is a
//! terminal, which keeps hooks, MCP servers, scripts, CI and tests free of it. `fl hub serve`
//! calls [`warn_always`] once at start, so the service log still records a loose store.
//! `FL_QUIET_STORE_PERMS=1` (any non-empty value except `0`) silences both. The file is reported
//! when it has a group or other bit; its directory only when the file is loose too, or when
//! others can write in it (and it is not sticky), because a `0600` file is not readable through a
//! `0755` directory.
//!
//! Non-unix targets have no mode bits to set or read, so they behave as before.

use std::collections::BTreeSet;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(unix)]
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};

/// The mode of a store file `fl` creates.
#[cfg(unix)]
const FILE_MODE: u32 = 0o600;
/// The mode of a store directory `fl` creates.
#[cfg(unix)]
const DIR_MODE: u32 = 0o700;
/// The most paths the once-per-path warning remembers. A process opens a handful of stores; the
/// bound only stops a pathological caller from growing the set without end. Past it, no new path
/// is warned about.
#[cfg_attr(not(unix), allow(dead_code))]
const MAX_WARNED: usize = 1024;

/// Create `path` and any missing parents; the directory `path` itself gets mode 0700 when this
/// call creates it. Intermediate directories this call creates use the default mode (the umask):
/// `fl` owns the leaf that holds the store, not the `~/.local/share` or `~/project` above it, and
/// a private mode on a shared parent could lock other users or tools out of a path `fl` does not
/// own. A directory that already exists is left exactly as it is, mode included. A path that
/// exists and is not a directory is an error, as with [`std::fs::create_dir_all`].
pub fn create_private_dir_all(path: &Path) -> io::Result<()> {
    if path.as_os_str().is_empty() || path.is_dir() {
        return Ok(());
    }
    #[cfg(unix)]
    {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        match std::fs::DirBuilder::new().mode(DIR_MODE).create(path) {
            Ok(()) => Ok(()),
            // Another process made it between the check and the create: its mode stands.
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists && path.is_dir() => Ok(()),
            Err(e) => Err(e),
        }
    }
    #[cfg(not(unix))]
    {
        std::fs::create_dir_all(path)
    }
}

/// Create the empty store file at `path` with mode 0600, unless something is already there.
/// `Ok(true)` when this call made it, `Ok(false)` when the path existed (a file, a symlink, a
/// dangling symlink: the exclusive create refuses all of them and follows none) or another
/// process created it first. The empty file it leaves is what `redb::Database::create` accepts
/// as a new database, so the caller opens the path as before. Any other error is returned for
/// the caller to ignore: the open that follows reports a real fault in its own words.
pub(crate) fn create_private_file(path: &Path) -> io::Result<bool> {
    #[cfg(unix)]
    {
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create_new(true).mode(FILE_MODE);
        match opts.open(path) {
            Ok(_) => Ok(true),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => Ok(false),
            Err(e) => Err(e),
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(false)
    }
}

/// The bytes of the secret file at `path` (a PSK, a token file), at most `max` of them. Refused,
/// with the `chmod` that fixes it, when the file is not a regular file or its group or others have
/// any permission bit: a secret others can read is no secret, and one they can write is not ours.
/// Refused too when its owner is neither this process's effective uid nor root (a root-owned
/// secret mount works): another user's file is not ours to trust, whatever its mode.
/// The mode and the owner are read from the opened file, so the bytes read are the file that was
/// checked.
///
/// A symlink is followed and its TARGET is checked: the checks run on the opened descriptor
/// (`fstat`), so they apply to the final file with no window between check and read, and a
/// symlinked secret (a Kubernetes secret volume, say) works. The open does not block
/// (`O_NONBLOCK`): a FIFO or a device, also one a symlink points to, is opened without waiting
/// for a writer and refused as not a regular file. On a regular file `O_NONBLOCK` changes
/// nothing.
pub(crate) fn read_secret_file(path: &Path, max: u64) -> io::Result<Vec<u8>> {
    use std::io::Read;
    #[cfg(unix)]
    let file = {
        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(path)?
    };
    #[cfg(not(unix))]
    let file = std::fs::File::open(path)?;
    let meta = file.metadata()?;
    if !meta.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "not a regular file",
        ));
    }
    #[cfg(unix)]
    {
        // SAFETY: `geteuid` has no preconditions and cannot fail.
        let euid = unsafe { libc::geteuid() };
        owner_allowed(meta.uid(), euid, path)?;
        let mode = meta.mode() & 0o777;
        if mode & 0o077 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                format!(
                    "it is mode {mode:03o}, so other users can reach it; refused: {}",
                    fix_clause("chmod 600", path, "")
                ),
            ));
        }
    }
    if meta.len() > max {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("it is larger than {max} bytes"),
        ));
    }
    let mut bytes = Vec::new();
    file.take(max.saturating_add(1)).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("it is larger than {max} bytes"),
        ));
    }
    Ok(bytes)
}

/// Whether a secret file owned by `owner` may be read by a process whose effective uid is `euid`:
/// its own files and root's. The error never holds the file's contents.
#[cfg(unix)]
fn owner_allowed(owner: u32, euid: u32, path: &Path) -> io::Result<()> {
    if owner == euid || owner == 0 {
        return Ok(());
    }
    Err(io::Error::new(
        io::ErrorKind::PermissionDenied,
        format!(
            "{} is owned by uid {owner}, not by this process's uid {euid} or root; refused",
            shown_path(path)
        ),
    ))
}

/// Paths already warned about in this process.
static WARNED: Mutex<BTreeSet<PathBuf>> = Mutex::new(BTreeSet::new());

/// Set by [`warn_always`]: the process is a service whose stderr is a log, not a terminal.
static ALWAYS: AtomicBool = AtomicBool::new(false);

/// The environment variable that silences the warning, `fl hub serve`'s included.
pub const QUIET_ENV: &str = "FL_QUIET_STORE_PERMS";

/// Warnings printed so far, kept only under test so a test can count them without parsing stderr.
#[cfg(test)]
pub(crate) static EMITTED: Mutex<Vec<String>> = Mutex::new(Vec::new());

#[cfg(test)]
thread_local! {
    /// What a test thread says stderr is: the real stream is a terminal when a person runs the
    /// tests, and a test must not depend on that.
    pub(crate) static TEST_TTY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Print the store warning on every start of this process, terminal or not, until the process
/// ends. `fl hub serve` calls it before it opens a store, so the service log records a loose
/// store; the once-per-path dedupe still applies, so it is once per start. `FL_QUIET_STORE_PERMS`
/// still silences it.
pub fn warn_always() {
    ALWAYS.store(true, Ordering::Relaxed);
}

/// Whether `value` of `FL_QUIET_STORE_PERMS` asks for quiet: any non-empty value except `0`.
fn quiet_value(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|v| !v.is_empty() && v != "0")
}

/// The `quiet` field of the policy: a test build is never quiet by environment, so a test's
/// outcome does not depend on the developer's own `FL_QUIET_STORE_PERMS`. A pure function so the
/// release branch is testable from a test build.
fn quiet_from(test_build: bool, value: Option<&std::ffi::OsStr>) -> bool {
    !test_build && quiet_value(value)
}

/// When a warning may print, and where the process stands. The decision is data so a test can
/// state it without a real terminal, environment or working directory.
#[derive(Debug, Clone)]
pub(crate) struct Policy {
    /// [`warn_always`] was called.
    pub always: bool,
    /// Stderr is a terminal.
    pub stderr_is_terminal: bool,
    /// `FL_QUIET_STORE_PERMS` asks for quiet.
    pub quiet: bool,
    /// The current directory, when known; a directory advice never names it or an ancestor.
    pub cwd: Option<PathBuf>,
}

impl Policy {
    /// The policy of this process now.
    pub(crate) fn current() -> Self {
        #[cfg(test)]
        let tty = TEST_TTY.with(|t| t.get());
        #[cfg(not(test))]
        let tty = std::io::IsTerminal::is_terminal(&std::io::stderr());
        Policy {
            always: ALWAYS.load(Ordering::Relaxed),
            stderr_is_terminal: tty,
            quiet: quiet_from(cfg!(test), std::env::var_os(QUIET_ENV).as_deref()),
            cwd: std::env::current_dir().ok(),
        }
    }

    /// Whether any warning may print: never when quiet; otherwise when `always` or stderr is a
    /// terminal.
    pub(crate) fn emits(&self) -> bool {
        !self.quiet && (self.always || self.stderr_is_terminal)
    }
}

/// Claim `path` in `seen`: true the first time, false after, and false once the set is full.
/// Poison-tolerant: a panic elsewhere must not turn into a missing warning or a second panic.
#[cfg_attr(not(unix), allow(dead_code))]
fn claim(seen: &Mutex<BTreeSet<PathBuf>>, path: &Path) -> bool {
    let mut seen = seen.lock().unwrap_or_else(|e| e.into_inner());
    if seen.len() >= MAX_WARNED || seen.contains(path) {
        return false;
    }
    seen.insert(path.to_path_buf());
    true
}

/// `path` with `.` and `..` components removed without touching the file system: a `..` removes
/// the component before it, and one at the root is dropped. It is wrong across a symlink, so it is
/// only the fallback for a path that cannot be canonicalised.
#[cfg_attr(not(unix), allow(dead_code))]
fn lexical(path: &Path) -> PathBuf {
    use std::path::Component;
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() && !path.has_root() {
                    out.push("..");
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// The one name for the directory `dir`: canonical (symlinks and `..` resolved) when it exists,
/// else lexically normalised.
#[cfg_attr(not(unix), allow(dead_code))]
fn resolved(dir: &Path) -> PathBuf {
    std::fs::canonicalize(dir).unwrap_or_else(|_| lexical(dir))
}

/// `store` with its directory resolved by [`resolved`] and its own name kept as given (a symlink
/// at the store path stays a symlink: `chmod` follows it). The path an advice prints: absolute,
/// without `.` or `..`.
#[cfg_attr(not(unix), allow(dead_code))]
fn clean_store_path(store: &Path) -> PathBuf {
    match (store.parent(), store.file_name()) {
        (Some(parent), Some(name)) if !parent.as_os_str().is_empty() => resolved(parent).join(name),
        _ => lexical(store),
    }
}

/// The directory that holds `store`, as a path that names one directory from anywhere: canonical
/// when it exists, else lexically normalised. `None` when the directory is relative and cannot be
/// resolved, or when it is the working directory or one of its ancestors, compared in every
/// resolved form of both (lexical and canonical, so `..` and symlinks cannot hide an ancestor):
/// `chmod 700 .` or a chmod of a directory the person is working under is not advice to print.
#[cfg_attr(not(unix), allow(dead_code))]
fn nameable_dir(store: &Path, cwd: Option<&Path>) -> Option<PathBuf> {
    let parent = match store.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let canon = std::fs::canonicalize(parent).ok();
    let dir = match (&canon, parent.is_absolute()) {
        (Some(c), _) => c.clone(),
        (None, true) => lexical(parent),
        (None, false) => return None,
    };
    if let Some(cwd) = cwd {
        let cwd_forms = [lexical(cwd), resolved(cwd)];
        let mut dir_forms = vec![dir.clone()];
        if parent.is_absolute() {
            dir_forms.push(lexical(parent));
        }
        if cwd_forms
            .iter()
            .any(|c| dir_forms.iter().any(|d| c.starts_with(d)))
        {
            return None;
        }
    }
    Some(dir)
}

/// Whether `c` must never reach a terminal as part of a pasted command: a control character
/// (newline, ESC, DEL, ...), or a bidirectional override that makes text read differently from
/// what it is.
fn unsafe_char(c: char) -> bool {
    c.is_control()
        || matches!(c, '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
}

/// `s` as one POSIX shell word: in single quotes, each `'` written `'\''`. Always quoted, so the
/// output never depends on which characters are special.
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// `path` as a shell word, or `None` when it cannot be pasted safely: not valid UTF-8 (the text
/// would name a different path), or holding a character [`unsafe_char`] rejects.
pub(crate) fn quoted_path(path: &Path) -> Option<String> {
    let s = path.to_str()?;
    (!s.chars().any(unsafe_char)).then(|| shell_quote(s))
}

/// The end of an advice, after `lead`: the runnable command for `path` when it is safe to paste,
/// else the instruction without one. `chmod` is the command and mode, e.g. `chmod 600`.
fn fix_clause(chmod: &str, path: &Path, lead: &str) -> String {
    match quoted_path(path) {
        Some(q) => format!("{lead}fix it with: {chmod} {q}"),
        None => format!(
            "{lead}run {chmod} on it yourself; its name contains characters that cannot be pasted safely"
        ),
    }
}

/// `path` for the text of a warning: as is when it is safe to print, else debug-escaped (quoted,
/// control characters and bytes that are not UTF-8 written as escapes).
pub(crate) fn shown_path(path: &Path) -> String {
    match quoted_path(path) {
        Some(_) => path.display().to_string(),
        None => format!("{path:?}"),
    }
}

/// The warnings an existing store at `store` earns under `policy`, each one only for a path `seen`
/// has not claimed yet (a policy that does not emit claims nothing). A path that cannot be read
/// earns none: a stat error is not a finding. The text names the path, its mode and the fix; it
/// never holds store contents. Pure but for the stat calls: it prints nothing.
///
/// * The file is reported when it has a group or other bit.
/// * Its directory is reported only when the file is loose too, or when the directory is group or
///   other writable, and never when it is sticky (`/tmp`, shared by design) or cannot be named
///   (see `nameable_dir`).
pub(crate) fn loose_warnings(
    store: &Path,
    seen: &Mutex<BTreeSet<PathBuf>>,
    policy: &Policy,
) -> Vec<String> {
    #[cfg(unix)]
    {
        if !policy.emits() {
            return Vec::new();
        }
        let mut out = Vec::new();
        // A relative store path is relative to the working directory: name it in full, so the
        // advice (and a service log) reads the same from anywhere.
        let joined = match (store.is_absolute(), policy.cwd.as_deref()) {
            (false, Some(cwd)) => cwd.join(store),
            _ => store.to_path_buf(),
        };
        // `metadata` follows a symlink: the mode that matters is the file that holds the data.
        // It reads the path as the kernel does; what is printed is the clean form of it.
        let Ok(file) = std::fs::metadata(&joined) else {
            return out;
        };
        if !file.is_file() {
            return out;
        }
        let shown = clean_store_path(&joined);
        let file_loose = file.mode() & 0o077 != 0;
        if file_loose && claim(seen, &shown) {
            out.push(format!(
                "warning: the store {} is mode {:03o}, so other users can read it; {}",
                shown_path(&shown),
                file.mode() & 0o777,
                fix_clause("chmod 600", &shown, "")
            ));
        }
        let Some(dir) = nameable_dir(&shown, policy.cwd.as_deref()) else {
            return out;
        };
        let Ok(meta) = std::fs::metadata(&dir) else {
            return out;
        };
        let reachable = meta.mode() & 0o077 != 0;
        let writable = meta.mode() & 0o022 != 0;
        // A sticky directory (`/tmp`) is shared by design: its owner is not this store's.
        if meta.is_dir()
            && meta.mode() & 0o1000 == 0
            && ((file_loose && reachable) || writable)
            && claim(seen, &dir)
        {
            let why = if file_loose {
                "so other users can reach it"
            } else {
                "so other users can replace the store in it"
            };
            out.push(format!(
                "warning: the directory {} holds a store and is mode {:03o}, {why}; {}",
                shown_path(&dir),
                meta.mode() & 0o777,
                fix_clause("chmod 700", &dir, "if only your own data lives there, ")
            ));
        }
        out
    }
    #[cfg(not(unix))]
    {
        let _ = (store, seen, policy);
        Vec::new()
    }
}

/// Bytes a process that is not root can still write on the filesystem that holds `dir`
/// (`statvfs`'s available blocks). For a command that is about to make a large copy there.
// The `statvfs` fields are `u64` here and narrower on other targets.
#[allow(clippy::useless_conversion)]
pub fn available_bytes(dir: &Path) -> io::Result<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(dir.as_os_str().as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path holds a NUL byte"))?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: `c` is a NUL-terminated path and `stat` a writable, correctly sized buffer.
    if unsafe { libc::statvfs(c.as_ptr(), &mut stat) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(u64::from(stat.f_bavail).saturating_mul(u64::from(stat.f_frsize)))
}

/// Write `text` and a newline to `w`, ignoring any error: a closed pipe (EPIPE) or a full disk on
/// stderr must not turn a reminder into a panic or a failed command.
fn write_line(w: &mut dyn Write, text: &str) {
    let _ = writeln!(w, "{text}");
}

/// Print, to stderr, one warning per path per process for a store at `store` that other users can
/// reach, as `loose_warnings` decides: only when stderr is a terminal, or always after
/// [`warn_always`], and never under `FL_QUIET_STORE_PERMS`. Never changes a mode, never fails and
/// never panics, even when stderr is a closed pipe.
pub fn warn_if_loose(store: &Path) {
    for text in loose_warnings(store, &WARNED, &Policy::current()) {
        // Under test the line is recorded, not written, so a run in a terminal stays clean.
        #[cfg(test)]
        {
            write_line(&mut std::io::sink(), &text);
            EMITTED.lock().unwrap_or_else(|e| e.into_inner()).push(text);
        }
        #[cfg(not(test))]
        write_line(&mut std::io::stderr(), &text);
    }
}

#[cfg(all(test, unix))]
mod secret_file_tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::mpsc;
    use std::time::Duration;

    /// `read_secret_file` of `path` on its own thread: its answer, or a
    /// panic when it gives none within 5 s (an open that blocked).
    fn read_within(path: PathBuf) -> io::Result<Vec<u8>> {
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(read_secret_file(&path, 1024));
        });
        rx.recv_timeout(Duration::from_secs(5))
            .expect("read_secret_file blocked on the path")
    }

    #[test]
    fn a_fifo_is_refused_without_waiting_for_a_writer() {
        let dir = tempfile::tempdir().unwrap();
        let fifo = dir.path().join("psk");
        let c_path = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
        // SAFETY: `c_path` is a valid NUL-terminated path.
        assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
        let err = read_within(fifo).unwrap_err();
        assert!(err.to_string().contains("not a regular file"), "{err}");
    }

    fn secret(dir: &Path, name: &str, mode: u32) -> PathBuf {
        let file = dir.join(name);
        std::fs::write(&file, b"secret").unwrap();
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(mode)).unwrap();
        file
    }

    fn fifo_at(path: &Path) {
        let c_path = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
        // SAFETY: `c_path` is a valid NUL-terminated path.
        assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
    }

    #[test]
    fn owner_decision_accepts_own_and_root_and_refuses_others() {
        let path = Path::new("/run/secrets/psk");
        assert!(owner_allowed(1000, 1000, path).is_ok());
        assert!(owner_allowed(0, 1000, path).is_ok(), "root-owned mount");
        assert!(owner_allowed(0, 0, path).is_ok());
        assert!(
            owner_allowed(1000, 0, path).is_err(),
            "root trusts only its own"
        );
        let err = owner_allowed(1001, 1000, path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::PermissionDenied);
        let text = err.to_string();
        assert!(text.contains("/run/secrets/psk"), "{text}");
        assert!(
            text.contains("uid 1001") && text.contains("uid 1000"),
            "{text}"
        );
    }

    #[test]
    fn the_current_users_private_file_is_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let file = secret(dir.path(), "psk", 0o600);
        assert_eq!(read_within(file).unwrap(), b"secret");
    }

    #[test]
    fn a_symlink_to_a_private_secret_is_followed_and_read() {
        let dir = tempfile::tempdir().unwrap();
        let file = secret(dir.path(), "psk", 0o600);
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&file, &link).unwrap();
        assert_eq!(read_within(link).unwrap(), b"secret");
        assert_eq!(read_within(file).unwrap(), b"secret");
    }

    #[test]
    fn a_symlink_to_a_loose_secret_is_refused_for_its_target_mode() {
        let dir = tempfile::tempdir().unwrap();
        let file = secret(dir.path(), "psk", 0o644);
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&file, &link).unwrap();
        let err = read_within(link).unwrap_err();
        assert!(err.to_string().contains("mode 644"), "{err}");
    }

    #[test]
    fn a_symlink_to_a_fifo_is_refused_without_waiting_for_a_writer() {
        let dir = tempfile::tempdir().unwrap();
        let fifo = dir.path().join("fifo");
        fifo_at(&fifo);
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&fifo, &link).unwrap();
        let err = read_within(link).unwrap_err();
        assert!(err.to_string().contains("not a regular file"), "{err}");
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::RedbStore;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::Arc;
    use std::sync::Barrier;
    use tempfile::TempDir;

    fn mode(path: &Path) -> u32 {
        std::fs::metadata(path).unwrap().permissions().mode() & 0o7777
    }

    fn set_mode(path: &Path, mode: u32) {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
    }

    /// The warnings the open of `store` printed, found by the unique temp path they name.
    fn emitted_for(dir: &Path) -> Vec<String> {
        let needle = dir.display().to_string();
        EMITTED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|w| w.contains(&needle))
            .cloned()
            .collect()
    }

    /// A policy that prints: a terminal, no opt-out, no working directory.
    fn tty() -> Policy {
        Policy {
            always: false,
            stderr_is_terminal: true,
            quiet: false,
            cwd: None,
        }
    }

    fn fresh() -> Mutex<BTreeSet<PathBuf>> {
        Mutex::new(BTreeSet::new())
    }

    /// A store file at `dir/name` with `file_mode`, in a directory made `dir_mode`.
    fn store_at(tmp: &Path, dir_mode: u32, file_mode: u32) -> PathBuf {
        let dir = tmp.join("d");
        std::fs::create_dir(&dir).unwrap();
        set_mode(&dir, dir_mode);
        let path = dir.join("s.redb");
        std::fs::write(&path, b"").unwrap();
        set_mode(&path, file_mode);
        path
    }

    /// The work of the umask test. It runs in a child process that `sh` started under `umask 0`,
    /// because the umask is per process and a test thread must not change it for the others.
    fn new_store_in_a_fresh_dir() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("fl");
        let store = dir.join("fl.redb");
        create_private_dir_all(&dir).unwrap();
        drop(RedbStore::open(&store).unwrap());
        assert_eq!(mode(&dir), 0o700, "the directory fl created");
        assert_eq!(mode(&store), 0o600, "the store file fl created");
    }

    #[test]
    fn a_new_store_and_its_directory_are_private() {
        new_store_in_a_fresh_dir();
    }

    #[test]
    fn the_modes_do_not_depend_on_the_umask() {
        const CHILD: &str = "FL_STORE_PRIVATE_UMASK_CHILD";
        if std::env::var_os(CHILD).is_some() {
            // Proof the child really runs permissive: a plain create would be 0666 here.
            let probe_dir = TempDir::new().unwrap();
            let probe = probe_dir.path().join("probe");
            drop(std::fs::File::create(&probe).unwrap());
            assert_eq!(mode(&probe), 0o666, "the child must run under umask 0");
            drop(probe_dir);
            new_store_in_a_fresh_dir();
            return;
        }
        let exe = std::env::current_exe().unwrap();
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg("umask 0; exec \"$0\" --exact private::tests::the_modes_do_not_depend_on_the_umask --test-threads=1")
            .arg(exe)
            .env(CHILD, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "child failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        // A filter that matched no test also exits 0: the child must have run exactly this one.
        assert!(
            stdout.contains("1 passed"),
            "the child did not run the test:\n{stdout}"
        );
    }

    #[test]
    fn only_the_leaf_directory_fl_creates_is_private_and_existing_ones_are_untouched() {
        let tmp = TempDir::new().unwrap();
        let existing = tmp.path().join("shared");
        std::fs::create_dir(&existing).unwrap();
        set_mode(&existing, 0o755);

        // An existing directory keeps its mode, whatever it is.
        create_private_dir_all(&existing).unwrap();
        assert_eq!(mode(&existing), 0o755);
        let open_dir = tmp.path().join("open");
        std::fs::create_dir(&open_dir).unwrap();
        set_mode(&open_dir, 0o777);
        create_private_dir_all(&open_dir).unwrap();
        assert_eq!(mode(&open_dir), 0o777);

        // Under an existing directory: only the new leaf is 0700, the parent is as it was.
        let leaf = existing.join("fl");
        create_private_dir_all(&leaf).unwrap();
        assert_eq!(mode(&leaf), 0o700);
        assert_eq!(mode(&existing), 0o755);

        // Several levels: intermediates take the default mode, the leaf is 0700.
        let deep = existing.join("a").join("b").join("c");
        create_private_dir_all(&deep).unwrap();
        assert_eq!(mode(&deep), 0o700);
        assert_ne!(
            mode(&existing.join("a")),
            0o700,
            "intermediates are not private"
        );
        assert_ne!(
            mode(&existing.join("a/b")),
            0o700,
            "intermediates are not private"
        );

        // A second call, and a path that is a file, behave like `create_dir_all`.
        create_private_dir_all(&deep).unwrap();
        assert_eq!(mode(&deep), 0o700);
        let file = tmp.path().join("file");
        std::fs::write(&file, b"x").unwrap();
        assert!(create_private_dir_all(&file.join("sub")).is_err());
    }

    #[test]
    fn an_existing_loose_store_is_warned_about_once_and_never_chmodded() {
        TEST_TTY.with(|t| t.set(true));
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("old");
        std::fs::create_dir(&dir).unwrap();
        set_mode(&dir, 0o755);
        let path = dir.join("fl.redb");
        // Made the way an older fl made it: by redb, with the default mode.
        drop(redb::Database::create(&path).unwrap());
        set_mode(&path, 0o644);

        drop(RedbStore::open(&path).unwrap());
        assert_eq!(mode(&path), 0o644, "the file is not chmodded");
        assert_eq!(mode(&dir), 0o755, "the directory is not chmodded");
        let first = emitted_for(&dir);
        assert_eq!(
            first.len(),
            2,
            "one for the file, one for the dir: {first:?}"
        );
        let file_warning = first.iter().find(|w| w.contains("chmod 600")).unwrap();
        assert!(file_warning.contains(&path.display().to_string()));
        assert!(file_warning.contains("644"));
        let dir_warning = first.iter().find(|w| w.contains("chmod 700")).unwrap();
        assert!(dir_warning.contains("755"));

        // A second open of the same path is quiet.
        drop(RedbStore::open(&path).unwrap());
        assert_eq!(
            emitted_for(&dir),
            first,
            "no second warning for the same path"
        );
    }

    #[test]
    fn nothing_is_printed_when_stderr_is_not_a_terminal() {
        // The open path, with the thread's stderr a pipe or file (the default under test).
        TEST_TTY.with(|t| t.set(false));
        let tmp = TempDir::new().unwrap();
        let path = store_at(tmp.path(), 0o755, 0o644);
        drop(redb::Database::create(&path).unwrap());
        set_mode(&path, 0o644);
        drop(RedbStore::open(&path).unwrap());
        assert!(emitted_for(tmp.path()).is_empty());

        // The pure function: not a terminal, nothing; a terminal, the same store warns. The silent
        // call claims nothing, so it does not eat the later warning.
        let seen = fresh();
        let mut pol = tty();
        pol.stderr_is_terminal = false;
        assert!(loose_warnings(&path, &seen, &pol).is_empty());
        assert_eq!(loose_warnings(&path, &seen, &tty()).len(), 2);
    }

    #[test]
    fn the_hub_serve_mode_prints_without_a_terminal_once_per_path() {
        let tmp = TempDir::new().unwrap();
        let path = store_at(tmp.path(), 0o755, 0o644);
        let seen = fresh();
        let mut pol = tty();
        pol.stderr_is_terminal = false;
        assert!(loose_warnings(&path, &seen, &pol).is_empty());
        pol.always = true;
        assert_eq!(loose_warnings(&path, &seen, &pol).len(), 2);
        assert!(
            loose_warnings(&path, &seen, &pol).is_empty(),
            "once per path"
        );
    }

    #[test]
    fn the_opt_out_silences_both_modes() {
        let tmp = TempDir::new().unwrap();
        let path = store_at(tmp.path(), 0o755, 0o644);
        let mut pol = tty();
        pol.quiet = true;
        assert!(loose_warnings(&path, &fresh(), &pol).is_empty());
        pol.always = true;
        assert!(loose_warnings(&path, &fresh(), &pol).is_empty());

        // Any non-empty value except 0.
        use std::ffi::OsStr;
        assert!(quiet_value(Some(OsStr::new("1"))));
        assert!(quiet_value(Some(OsStr::new("yes"))));
        assert!(!quiet_value(Some(OsStr::new("0"))));
        assert!(!quiet_value(Some(OsStr::new(""))));
        assert!(!quiet_value(None));
    }

    #[test]
    fn the_release_policy_reads_the_environment_and_a_test_build_does_not() {
        use std::ffi::OsStr;
        // The release branch (`test_build` false): unset, empty and `0` are loud; the rest quiet.
        assert!(!quiet_from(false, None));
        assert!(!quiet_from(false, Some(OsStr::new(""))));
        assert!(!quiet_from(false, Some(OsStr::new("0"))));
        assert!(quiet_from(false, Some(OsStr::new("1"))));
        assert!(quiet_from(false, Some(OsStr::new("yes"))));
        // A test build ignores the developer's environment entirely.
        assert!(!quiet_from(true, Some(OsStr::new("1"))));
        assert!(!quiet_from(true, Some(OsStr::new("yes"))));
        // And `Policy::current` in this build is never quiet, whatever the variable says.
        assert!(!Policy::current().quiet);
    }

    #[test]
    fn a_loose_file_alone_warns_about_the_file_only() {
        let tmp = TempDir::new().unwrap();
        let path = store_at(tmp.path(), 0o700, 0o640);
        let w = loose_warnings(&path, &fresh(), &tty());
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("chmod 600"), "{w:?}");
    }

    #[test]
    fn an_open_directory_around_a_private_file_is_silent() {
        let tmp = TempDir::new().unwrap();
        let path = store_at(tmp.path(), 0o755, 0o600);
        assert!(loose_warnings(&path, &fresh(), &tty()).is_empty());
    }

    #[test]
    fn a_directory_others_can_write_in_warns_even_around_a_private_file() {
        let tmp = TempDir::new().unwrap();
        for dir_mode in [0o775, 0o757, 0o777] {
            let sub = tempfile::tempdir_in(tmp.path()).unwrap();
            let path = store_at(sub.path(), dir_mode, 0o600);
            let w = loose_warnings(&path, &fresh(), &tty());
            assert_eq!(w.len(), 1, "{dir_mode:o}: {w:?}");
            assert!(w[0].contains("chmod 700"), "{w:?}");
            assert!(w[0].contains("replace the store"), "{w:?}");
        }
    }

    #[test]
    fn a_sticky_directory_is_never_called_loose() {
        let tmp = TempDir::new().unwrap();
        // Sticky and world-writable, like /tmp, around a private file and around a loose one.
        let path = store_at(tmp.path(), 0o1777, 0o600);
        assert!(loose_warnings(&path, &fresh(), &tty()).is_empty());
        let sub = tempfile::tempdir_in(tmp.path()).unwrap();
        let loose = store_at(sub.path(), 0o1777, 0o644);
        let w = loose_warnings(&loose, &fresh(), &tty());
        assert_eq!(w.len(), 1, "only the file: {w:?}");
        assert!(w[0].contains("chmod 600"));
    }

    #[test]
    fn a_private_store_earns_no_warning() {
        TEST_TTY.with(|t| t.set(true));
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path().join("fl");
        create_private_dir_all(&dir).unwrap();
        let path = dir.join("fl.redb");
        drop(RedbStore::open(&path).unwrap());
        drop(RedbStore::open(&path).unwrap());
        assert_eq!(emitted_for(&dir), Vec::<String>::new());
    }

    #[test]
    fn a_relative_store_path_never_earns_chmod_700_dot() {
        let tmp = TempDir::new().unwrap();
        let cwd = tmp.path().join("work");
        std::fs::create_dir(&cwd).unwrap();
        set_mode(&cwd, 0o755);
        let file = cwd.join("store.redb");
        std::fs::write(&file, b"").unwrap();
        set_mode(&file, 0o644);
        let mut pol = tty();
        pol.cwd = Some(cwd.clone());

        // `--db store.redb`: the file is reported by its absolute path; the directory is the
        // working directory, so there is no directory advice at all.
        let w = loose_warnings(Path::new("store.redb"), &fresh(), &pol);
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(
            w[0].contains(&format!("chmod 600 '{}'", file.display())),
            "{w:?}"
        );
        assert!(w.iter().all(|t| !t.contains("chmod 700")));

        // An ancestor of the working directory is not advised either.
        let deep = cwd.join("a").join("b");
        std::fs::create_dir_all(&deep).unwrap();
        pol.cwd = Some(deep);
        let w = loose_warnings(&file, &fresh(), &pol);
        assert!(w.iter().all(|t| !t.contains("chmod 700")), "{w:?}");

        // A relative store in a subdirectory of the working directory (here the absolute form,
        // since resolving a relative one reads the process's own directory) names the directory
        // in full, never `.`.
        let sub = cwd.join("data");
        std::fs::create_dir(&sub).unwrap();
        set_mode(&sub, 0o755);
        let inner = sub.join("s.redb");
        std::fs::write(&inner, b"").unwrap();
        set_mode(&inner, 0o644);
        let mut pol = tty();
        pol.cwd = Some(cwd);
        let w = loose_warnings(&inner, &fresh(), &pol);
        let dir_w = w.iter().find(|t| t.contains("chmod 700")).unwrap();
        assert!(dir_w.contains(&format!("chmod 700 '{}'", sub.display())));
        assert!(!dir_w.contains("chmod 700 ."), "{dir_w}");
    }

    #[test]
    fn a_dir_that_cannot_be_named_gets_no_advice() {
        // A relative parent that does not exist cannot be resolved: no `chmod 700 sub`.
        assert_eq!(
            nameable_dir(Path::new("no-such-dir-xyz/s.redb"), None),
            None
        );
        // The working directory and its ancestors are not named; a sibling is.
        let store = Path::new("/x/y/s.redb");
        assert_eq!(nameable_dir(store, Some(Path::new("/x/y"))), None);
        assert_eq!(nameable_dir(store, Some(Path::new("/x/y/z"))), None);
        assert_eq!(
            nameable_dir(store, Some(Path::new("/x/q"))),
            Some(PathBuf::from("/x/y"))
        );
        // `..` is resolved before the comparison, in both paths.
        assert_eq!(
            nameable_dir(Path::new("/x/y/z/../s.redb"), Some(Path::new("/x/y/z"))),
            None
        );
        assert_eq!(
            nameable_dir(Path::new("/x/y/s.redb"), Some(Path::new("/x/y/z/.."))),
            None
        );
        assert_eq!(
            nameable_dir(Path::new("/x/y/../q/s.redb"), Some(Path::new("/x/y"))),
            Some(PathBuf::from("/x/q"))
        );
        assert_eq!(
            nameable_dir(store, None),
            Some(PathBuf::from("/x/y")),
            "no known working directory: named as given"
        );
    }

    /// A canonical temp root, so a printed path equals the one the test built.
    fn root(tmp: &TempDir) -> PathBuf {
        std::fs::canonicalize(tmp.path()).unwrap()
    }

    fn dir_mode(path: &Path, m: u32) -> PathBuf {
        std::fs::create_dir_all(path).unwrap();
        set_mode(path, m);
        path.to_path_buf()
    }

    fn loose_file(path: &Path) {
        std::fs::write(path, b"").unwrap();
        set_mode(path, 0o644);
    }

    fn pol_in(cwd: &Path) -> Policy {
        let mut pol = tty();
        pol.cwd = Some(cwd.to_path_buf());
        pol
    }

    /// No warning holds a `..` component.
    fn no_dotdot(w: &[String]) {
        for t in w {
            assert!(!t.contains(".."), "a `..` in the advice: {t}");
        }
    }

    #[test]
    fn a_store_in_the_parent_of_the_working_directory_earns_no_directory_advice() {
        let tmp = TempDir::new().unwrap();
        let root = root(&tmp);
        let u = dir_mode(&root.join("u"), 0o755);
        let proj = dir_mode(&u.join("proj"), 0o755);
        loose_file(&u.join("fl.redb"));

        // `--db ../fl.redb`: the directory is the cwd's parent. The file is reported by its clean
        // absolute path; the parent of the working directory is never advised.
        let w = loose_warnings(Path::new("../fl.redb"), &fresh(), &pol_in(&proj));
        assert_eq!(
            w,
            vec![format!(
                "warning: the store {0}/fl.redb is mode 644, so other users can read it; \
                 fix it with: chmod 600 '{0}/fl.redb'",
                u.display()
            )]
        );
        no_dotdot(&w);
        // The same through an absolute path that holds `..`, as `FL_DB` would give.
        let abs = proj.join("..").join("fl.redb");
        let w = loose_warnings(&abs, &fresh(), &pol_in(&proj));
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w.iter().all(|t| !t.contains("chmod 700")), "{w:?}");
        no_dotdot(&w);
    }

    #[test]
    fn dot_and_dotdot_inside_a_store_path_are_normalised() {
        let tmp = TempDir::new().unwrap();
        let root = root(&tmp);
        let proj = dir_mode(&root.join("proj"), 0o755);
        dir_mode(&proj.join("x"), 0o755);
        loose_file(&proj.join("y.redb"));

        // `./x/../y.redb` lands in the working directory itself: no directory advice.
        let w = loose_warnings(Path::new("./x/../y.redb"), &fresh(), &pol_in(&proj));
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains(&format!("chmod 600 '{}/y.redb'", proj.display())));
        no_dotdot(&w);
        assert!(!w[0].contains("/./"), "{w:?}");

        // From a directory beside it, the same file's directory is advised, in clean form.
        let side = dir_mode(&root.join("side"), 0o755);
        let rel = Path::new("../proj/./x/../y.redb");
        let w = loose_warnings(rel, &fresh(), &pol_in(&side));
        assert_eq!(w.len(), 2, "{w:?}");
        assert!(
            w[1].contains(&format!("chmod 700 '{}'", proj.display())),
            "{w:?}"
        );
        no_dotdot(&w);
        assert!(w.iter().all(|t| !t.contains("/./")), "{w:?}");
    }

    #[test]
    fn a_symlink_to_an_ancestor_of_the_working_directory_earns_no_directory_advice() {
        let tmp = TempDir::new().unwrap();
        let root = root(&tmp);
        let u = dir_mode(&root.join("u"), 0o755);
        let proj = dir_mode(&u.join("proj"), 0o755);
        loose_file(&u.join("fl.redb"));
        let link = root.join("link");
        std::os::unix::fs::symlink(&u, &link).unwrap();

        // The store is named through the link; the link's target is the cwd's parent.
        let w = loose_warnings(&link.join("fl.redb"), &fresh(), &pol_in(&proj));
        assert_eq!(w.len(), 1, "{w:?}");
        assert!(w[0].contains("chmod 600"), "{w:?}");
        assert!(w.iter().all(|t| !t.contains("chmod 700")), "{w:?}");
        // The working directory reached through the link is the same directory.
        let w = loose_warnings(&u.join("fl.redb"), &fresh(), &pol_in(&link.join("proj")));
        assert!(w.iter().all(|t| !t.contains("chmod 700")), "{w:?}");

        // A directory beside the working directory is still advised, by a clean absolute path.
        let sib = dir_mode(&root.join("sib"), 0o755);
        loose_file(&sib.join("s.redb"));
        let w = loose_warnings(Path::new("../../sib/s.redb"), &fresh(), &pol_in(&proj));
        assert_eq!(w.len(), 2, "{w:?}");
        assert!(
            w[1].contains(&format!("chmod 700 '{}'", sib.display())),
            "{w:?}"
        );
        no_dotdot(&w);
    }

    /// `shell_quote` against known vectors, then against a real `sh`: the quoted word is one
    /// argument that equals the original.
    #[test]
    fn shell_quote_round_trips_through_a_real_shell() {
        let q = '\'';
        assert_eq!(shell_quote(""), "''");
        assert_eq!(shell_quote("plain"), "'plain'");
        assert_eq!(shell_quote("it's"), r"'it'\''s'");
        // A string of only quotes: each becomes `'\''`, inside the outer pair.
        assert_eq!(shell_quote("'"), format!("{q}{q}\\{q}{q}{q}"));
        assert_eq!(shell_quote("''"), format!("{q}{q}\\{q}{q}{q}\\{q}{q}{q}"));
        let vectors = [
            "",
            "plain",
            "with space",
            "x; rm -rf ~",
            "$(touch pwned)",
            "`touch pwned`",
            "it's",
            "'",
            "''",
            "a'b'c",
            "back\\slash",
            "\"dq\" $HOME *?[a]",
            "new\nline",
            "-rf",
            "tab\tend",
        ];
        for v in vectors {
            let script = format!("printf '%s' {}", shell_quote(v));
            let out = std::process::Command::new("sh")
                .arg("-c")
                .arg(&script)
                .output()
                .unwrap();
            assert!(out.status.success(), "{script}");
            assert_eq!(String::from_utf8(out.stdout).unwrap(), v, "{script}");
        }
    }

    /// Directory names that need quoting, and the quoted form of each inside a full path.
    const QUOTED: [(&str, &str); 4] = [
        ("x; rm -rf ~", "x; rm -rf ~"),
        ("$(touch pwned)", "$(touch pwned)"),
        ("it's", r"it'\''s"),
        ("with some spaces", "with some spaces"),
    ];

    #[test]
    fn a_path_that_needs_quoting_is_quoted_in_both_advice_strings() {
        let tmp = TempDir::new().unwrap();
        let root = root(&tmp);
        let base = root.display().to_string();
        for (name, escaped) in QUOTED {
            let dir = dir_mode(&root.join(name), 0o755);
            let store = dir.join("s.redb");
            loose_file(&store);
            let w = loose_warnings(&store, &fresh(), &tty());
            assert_eq!(
                w,
                vec![
                    format!(
                        "warning: the store {base}/{name}/s.redb is mode 644, so other users \
                         can read it; fix it with: chmod 600 '{base}/{escaped}/s.redb'"
                    ),
                    format!(
                        "warning: the directory {base}/{name} holds a store and is mode 755, \
                         so other users can reach it; if only your own data lives there, fix \
                         it with: chmod 700 '{base}/{escaped}'"
                    ),
                ],
                "{name}"
            );
            // The printed command, run by a real shell with `chmod` replaced by `printf`, is
            // one argument that equals the path.
            for (text, want) in [(&w[0], &store), (&w[1], &dir)] {
                let cmd = text.rsplit_once("fix it with: chmod ").unwrap().1;
                let (_mode, arg) = cmd.split_once(' ').unwrap();
                let out = std::process::Command::new("sh")
                    .arg("-c")
                    .arg(format!("printf '%s\\n' {arg}"))
                    .output()
                    .unwrap();
                assert_eq!(
                    String::from_utf8(out.stdout).unwrap(),
                    format!("{}\n", want.display()),
                    "{name}"
                );
            }
        }
    }

    const UNSAFE_NOTE: &str = "its name contains characters that cannot be pasted safely";

    #[test]
    fn a_path_that_cannot_be_pasted_safely_gets_no_command() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        let tmp = TempDir::new().unwrap();
        let root = root(&tmp);
        let names: [&OsStr; 3] = [
            OsStr::new("new\nline"),
            OsStr::new("esc\x1b[31mred"),
            OsStr::from_bytes(b"bad\xffbytes"),
        ];
        for name in names {
            let dir = dir_mode(&root.join(name), 0o755);
            let store = dir.join("s.redb");
            loose_file(&store);
            let w = loose_warnings(&store, &fresh(), &tty());
            let store_dbg = format!("{store:?}");
            let dir_dbg = format!("{dir:?}");
            assert_eq!(
                w,
                vec![
                    format!(
                        "warning: the store {store_dbg} is mode 644, so other users can read \
                         it; run chmod 600 on it yourself; {UNSAFE_NOTE}"
                    ),
                    format!(
                        "warning: the directory {dir_dbg} holds a store and is mode 755, so \
                         other users can reach it; if only your own data lives there, run \
                         chmod 700 on it yourself; {UNSAFE_NOTE}"
                    ),
                ],
                "{name:?}"
            );
            for t in &w {
                assert!(!t.contains("fix it with"), "{t}");
                assert!(
                    !t.contains("chmod 600 '") && !t.contains("chmod 700 '"),
                    "{t}"
                );
                assert!(!t.chars().any(char::is_control), "raw control char: {t:?}");
                assert!(!t.contains('\u{fffd}'), "a replacement character: {t:?}");
            }
        }
        // The quoting helper agrees.
        assert_eq!(quoted_path(Path::new("/a\nb")), None);
        assert_eq!(quoted_path(Path::new("/a\x1bb")), None);
        assert_eq!(quoted_path(Path::new("/a\x7fb")), None);
        assert_eq!(quoted_path(Path::new("/a\u{202e}b")), None);
        assert_eq!(quoted_path(Path::new(OsStr::from_bytes(b"/a\xffb"))), None);
        assert_eq!(quoted_path(Path::new("/a b")), Some("'/a b'".to_string()));
    }

    #[test]
    fn the_dedupe_stays_bounded_and_survives_poison() {
        let tmp = TempDir::new().unwrap();
        let path = store_at(tmp.path(), 0o755, 0o664);
        let seen = fresh();

        assert_eq!(loose_warnings(&path, &seen, &tty()).len(), 2);
        assert_eq!(
            loose_warnings(&path, &seen, &tty()).len(),
            0,
            "one warning per path"
        );
        // A second store in the same directory: the directory is not repeated.
        let other = path.with_file_name("t.redb");
        std::fs::write(&other, b"").unwrap();
        set_mode(&other, 0o600);
        assert_eq!(loose_warnings(&other, &seen, &tty()).len(), 0);

        // A stat error is no warning, and no panic.
        assert!(loose_warnings(&tmp.path().join("missing/x.redb"), &seen, &tty()).is_empty());

        // The set is bounded: a full one warns about nothing new.
        let full = Mutex::new(
            (0..MAX_WARNED)
                .map(|i| PathBuf::from(format!("/p/{i}")))
                .collect(),
        );
        assert!(loose_warnings(&path, &full, &tty()).is_empty());

        // A poisoned set still works.
        let poisoned = Arc::new(Mutex::new(BTreeSet::new()));
        let p2 = Arc::clone(&poisoned);
        let _ = std::thread::spawn(move || {
            let _held = p2.lock().unwrap();
            panic!("poison the set");
        })
        .join();
        assert!(poisoned.is_poisoned());
        assert_eq!(loose_warnings(&path, &poisoned, &tty()).len(), 2);
    }

    #[test]
    fn a_closed_stderr_pipe_cannot_panic_the_warning() {
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::Error::from(io::ErrorKind::BrokenPipe))
            }
            fn flush(&mut self) -> io::Result<()> {
                Err(io::Error::from(io::ErrorKind::BrokenPipe))
            }
        }
        write_line(&mut Broken, "warning: x");
        let mut ok = Vec::new();
        write_line(&mut ok, "warning: x");
        assert_eq!(ok, b"warning: x\n");
    }

    #[test]
    fn warn_always_makes_the_open_path_print_without_a_terminal() {
        // The setter is process-global, so it runs in a child process like the umask test.
        const CHILD: &str = "FL_STORE_PRIVATE_ALWAYS_CHILD";
        if std::env::var_os(CHILD).is_some() {
            TEST_TTY.with(|t| t.set(false));
            let tmp = TempDir::new().unwrap();
            let path = store_at(tmp.path(), 0o755, 0o644);
            drop(redb::Database::create(&path).unwrap());
            set_mode(&path, 0o644);
            drop(RedbStore::open(&path).unwrap());
            assert!(emitted_for(tmp.path()).is_empty(), "silent before the call");
            warn_always();
            let two = tmp.path().join("two");
            std::fs::create_dir(&two).unwrap();
            let path2 = store_at(&two, 0o755, 0o644);
            drop(redb::Database::create(&path2).unwrap());
            set_mode(&path2, 0o644);
            drop(RedbStore::open(&path2).unwrap());
            drop(RedbStore::open(&path2).unwrap());
            assert_eq!(emitted_for(tmp.path()).len(), 2, "file and dir, once");
            return;
        }
        let exe = std::env::current_exe().unwrap();
        let out = std::process::Command::new(exe)
            .args([
                "--exact",
                "private::tests::warn_always_makes_the_open_path_print_without_a_terminal",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "child failed:\n{stdout}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(
            stdout.contains("1 passed"),
            "the child did not run:\n{stdout}"
        );
    }

    #[test]
    fn an_existing_empty_file_is_accepted_as_a_new_store() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("empty.redb");
        std::fs::write(&path, b"").unwrap();
        set_mode(&path, 0o640);
        let store = RedbStore::open(&path).unwrap();
        drop(store);
        assert!(
            std::fs::metadata(&path).unwrap().len() > 0,
            "redb initialised it"
        );
        assert_eq!(mode(&path), 0o640, "an existing file keeps its mode");
        drop(RedbStore::open(&path).unwrap());
    }

    #[test]
    fn two_first_opens_at_once_both_succeed() {
        for round in 0..5 {
            let tmp = TempDir::new().unwrap();
            let path = tmp.path().join(format!("race{round}.redb"));
            let barrier = Arc::new(Barrier::new(2));
            let threads: Vec<_> = (0..2)
                .map(|_| {
                    let path = path.clone();
                    let barrier = Arc::clone(&barrier);
                    std::thread::spawn(move || {
                        barrier.wait();
                        drop(RedbStore::open(&path).expect("a first open must not fail"));
                    })
                })
                .collect();
            for t in threads {
                t.join().unwrap();
            }
            assert_eq!(mode(&path), 0o600);
        }
        // The helper itself: a lost create race is not an error.
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("x.redb");
        assert!(create_private_file(&path).unwrap());
        assert!(!create_private_file(&path).unwrap());
    }

    #[test]
    fn a_symlink_at_the_store_path_is_followed_by_redb_and_chmodded_by_nobody() {
        TEST_TTY.with(|t| t.set(true));
        let tmp = TempDir::new().unwrap();
        let real = tmp.path().join("real.redb");
        drop(redb::Database::create(&real).unwrap());
        set_mode(&real, 0o644);
        let link = tmp.path().join("link.redb");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        drop(RedbStore::open(&link).unwrap());
        assert!(
            std::fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink(),
            "the link is not replaced"
        );
        assert_eq!(mode(&real), 0o644, "the target is not chmodded");
        // The mode that is warned about is the target's: the file that holds the data.
        assert!(
            emitted_for(tmp.path())
                .iter()
                .any(|w| w.contains(&link.display().to_string()) && w.contains("644"))
        );

        // A dangling link: the exclusive create refuses it, so nothing is created through it
        // by fl's own code, and the link stays a link.
        let dangling = tmp.path().join("dangling.redb");
        std::os::unix::fs::symlink(tmp.path().join("nowhere.redb"), &dangling).unwrap();
        assert!(!create_private_file(&dangling).unwrap());
        assert!(!tmp.path().join("nowhere.redb").exists());
    }

    #[test]
    fn available_bytes_reports_the_free_space_of_a_directory_and_fails_on_a_missing_one() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(available_bytes(tmp.path()).unwrap() > 0);
        assert!(available_bytes(&tmp.path().join("missing")).is_err());
    }
}
