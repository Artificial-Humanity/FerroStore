//! A simulated disk for the crash-consistency harness (spec §3.5).
//!
//! [`SimFs`] implements [`ContainerFs`] and its files implement
//! [`ContainerIo`], so a [`super::FerroStore`] opened through it does all of
//! its container I/O here. Every operation is recorded in a trace. A disk
//! made with a budget of `N` cuts power after its `N`th operation: every
//! later operation fails with an I/O error, on every handle.
//!
//! # The fault model
//!
//! Each file has a *durable* image and a list of *pending* operations made
//! since its last sync. The running process always sees the *live* image (the
//! durable one with every pending operation applied), as it would through the
//! page cache. After a cut, [`SimFs::pending`] lists every pending operation
//! and how it can survive, and [`SimFs::survivor`] builds the image that one
//! choice per operation leaves:
//!
//! - **`write_at`** not yet synced: kept, dropped, or torn. A torn write keeps
//!   the part before one 512-byte sector boundary inside the write (sector
//!   boundaries are absolute file offsets) and drops the rest, which keeps
//!   whatever was there before.
//! - **`set_len`** not yet synced: kept or dropped.
//! - **`sync_data` and `sync_all`** make every pending write and `set_len` of
//!   that file durable. They are modelled the same: Linux `fdatasync` also
//!   makes a change of size durable. Neither makes a directory entry durable.
//! - **Directory entries** (create, `create_new`, `rename`, `link`,
//!   `unlink`) not yet followed by a `sync_dir` of their directory: each
//!   kept or dropped.
//! - **`stat`** of a name (`stat_id`) and `fstat` (`file_id`, `len`) read
//!   the live names and files: recorded, refused after a cut, never pending.
//! - **Temp names** ([`ContainerFs::temp_path`]) are made without an
//!   operation, deterministically: the same workload names the same temps,
//!   so a cut run's trace is a prefix of the reference run's. The sweep of
//!   stale temps does nothing: the simulated disk has no clock, so no temp
//!   is stale (the real sweep is tested on a real directory).
//! - **`flock`** is an exclusive lock per file, held by one description at a
//!   time: [`ContainerIo::lock_exclusive`] waits for it, as `flock(LOCK_EX)`
//!   does across processes. A lock is not on disk: it is no cut point and a
//!   cut leaves nothing of it. Taking and releasing it are logged apart from
//!   the trace ([`SimFs::lock_events`]), each with its place in the trace, so
//!   a test can check which operations ran under it.
//!
//! Each pending operation's fate is independent of the others, so the model
//! includes a later write surviving while an earlier one is lost: there is no
//! order among the operations between two syncs. A sync is a barrier: nothing
//! from before it is lost and nothing is reordered across it.
//!
//! # What the model does not cover
//!
//! - Corruption of synced data (bit rot, misdirected or lost writes after a
//!   sync, a device that lies about a flush).
//! - A torn write that keeps a later sector but not an earlier one, or that
//!   tears inside one sector. A write is torn only at a sector boundary and
//!   only as a prefix.
//! - A file whose size grew but whose new bytes read as zeros or stale data
//!   (`data=writeback`): a dropped write leaves no trace in the size.
//! - Locks that a crashed process still holds: every lock is gone at a cut.
//! - Directories themselves. Every directory exists; `create_store_file`
//!   does not make or sync a missing parent.
//! - Open handles across a cut. A survivor image is reopened from its names;
//!   an inode with no surviving name is gone.

use super::container_io::{ContainerFs, ContainerIo};
use super::layout::backend;
use fl_core::StoreError;
use std::collections::BTreeMap;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

/// The sector size a write can tear at.
pub(super) const SECTOR: u64 = 512;

/// The device number every simulated file reports: the inode number alone
/// tells two simulated files apart.
pub(super) const SIM_DEV: u64 = 0x51_4D;

/// One recorded operation. A new kind of I/O op gets a variant here, a case
/// in its `ContainerIo`/`ContainerFs` body, and a fate in [`SimFs::pending`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum IoOp {
    Create { path: PathBuf },
    CreateNew { path: PathBuf, mode: u32 },
    StatId { path: PathBuf },
    OpenRw { path: PathBuf },
    OpenRo { path: PathBuf },
    Read { ino: usize, offset: u64, len: u64 },
    Write { ino: usize, offset: u64, len: u64 },
    Len { ino: usize },
    FileId { ino: usize },
    SetLen { ino: usize, len: u64 },
    SyncData { ino: usize },
    SyncAll { ino: usize },
    SyncDir { dir: PathBuf },
    Lock { ino: usize },
    Unlock { ino: usize },
    Rename { from: PathBuf, to: PathBuf },
    Link { from: PathBuf, to: PathBuf },
    Unlink { path: PathBuf },
}

impl fmt::Display for IoOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IoOp::Create { path } => write!(f, "create {}", path.display()),
            IoOp::CreateNew { path, mode } => {
                write!(f, "create_new {} {mode:o}", path.display())
            }
            IoOp::StatId { path } => write!(f, "stat {}", path.display()),
            IoOp::OpenRw { path } => write!(f, "open-rw {}", path.display()),
            IoOp::OpenRo { path } => write!(f, "open-ro {}", path.display()),
            IoOp::Read { ino, offset, len } => write!(f, "read ino{ino} @{offset}+{len}"),
            IoOp::Write { ino, offset, len } => write!(f, "write ino{ino} @{offset}+{len}"),
            IoOp::Len { ino } => write!(f, "len ino{ino}"),
            IoOp::FileId { ino } => write!(f, "file_id ino{ino}"),
            IoOp::SetLen { ino, len } => write!(f, "set_len ino{ino} {len}"),
            IoOp::SyncData { ino } => write!(f, "sync_data ino{ino}"),
            IoOp::SyncAll { ino } => write!(f, "sync_all ino{ino}"),
            IoOp::SyncDir { dir } => write!(f, "sync_dir {}", dir.display()),
            IoOp::Lock { ino } => write!(f, "lock ino{ino}"),
            IoOp::Unlock { ino } => write!(f, "unlock ino{ino}"),
            IoOp::Rename { from, to } => write!(f, "rename {} -> {}", from.display(), to.display()),
            IoOp::Link { from, to } => write!(f, "link {} -> {}", from.display(), to.display()),
            IoOp::Unlink { path } => write!(f, "unlink {}", path.display()),
        }
    }
}

#[derive(Clone, Debug)]
enum DataOp {
    Write { offset: u64, bytes: Vec<u8> },
    SetLen(u64),
}

#[derive(Clone, Debug)]
enum DirOp {
    Create { path: PathBuf, ino: usize },
    Link { path: PathBuf, ino: usize },
    Unlink { path: PathBuf },
    Rename { from: PathBuf, to: PathBuf },
}

impl DirOp {
    fn touches(&self, dir: &Path) -> bool {
        let in_dir = |path: &Path| path.parent() == Some(dir);
        match self {
            DirOp::Create { path, .. } | DirOp::Link { path, .. } | DirOp::Unlink { path } => {
                in_dir(path)
            }
            DirOp::Rename { from, to } => in_dir(from) || in_dir(to),
        }
    }

    fn apply(&self, names: &mut BTreeMap<PathBuf, usize>) {
        match self {
            DirOp::Create { path, ino } | DirOp::Link { path, ino } => {
                names.insert(path.clone(), *ino);
            }
            DirOp::Unlink { path } => {
                names.remove(path);
            }
            DirOp::Rename { from, to } => {
                if let Some(ino) = names.remove(from) {
                    names.insert(to.clone(), ino);
                }
            }
        }
    }
}

impl fmt::Display for DirOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DirOp::Create { path, .. } => write!(f, "create {}", path.display()),
            DirOp::Link { path, .. } => write!(f, "link {}", path.display()),
            DirOp::Unlink { path } => write!(f, "unlink {}", path.display()),
            DirOp::Rename { from, to } => {
                write!(f, "rename {} -> {}", from.display(), to.display())
            }
        }
    }
}

fn apply_data(bytes: &mut Vec<u8>, op: &DataOp) {
    match op {
        DataOp::Write {
            offset,
            bytes: data,
        } => write_into(bytes, *offset, data),
        DataOp::SetLen(len) => bytes.resize(*len as usize, 0),
    }
}

fn write_into(bytes: &mut Vec<u8>, offset: u64, data: &[u8]) {
    let start = offset as usize;
    let end = start + data.len();
    if bytes.len() < end {
        bytes.resize(end, 0);
    }
    bytes[start..end].copy_from_slice(data);
}

/// The absolute sector boundaries strictly inside `[offset, offset + len)`.
fn tear_points(offset: u64, len: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let mut at = (offset / SECTOR + 1) * SECTOR;
    while at < offset + len {
        out.push(at);
        at += SECTOR;
    }
    out
}

#[derive(Clone, Debug, Default)]
struct Inode {
    live: Vec<u8>,
    durable: Vec<u8>,
    pending: Vec<DataOp>,
}

#[derive(Default)]
struct Sim {
    /// Names as the running process sees them.
    names: BTreeMap<PathBuf, usize>,
    /// Names as they are on disk.
    durable_names: BTreeMap<PathBuf, usize>,
    pending_dir: Vec<DirOp>,
    inodes: Vec<Inode>,
    trace: Vec<IoOp>,
    /// `(trace index, first bytes)` of every write, in order: what a write
    /// was (a snapshot starts `FERRSNAP`), which [`IoOp::Write`] does not say.
    write_heads: Vec<(usize, Vec<u8>)>,
    /// `(trace length then, IoOp::Lock or IoOp::Unlock)`, in order.
    lock_events: Vec<(usize, IoOp)>,
    /// Operations allowed before the cut. `None` never cuts.
    budget: Option<usize>,
    cut: bool,
    /// The next operation this matches fails once, with nothing applied and
    /// nothing recorded, and power stays on: a transient I/O error.
    fail_next: Option<FailWhen>,
    /// The next read this matches runs the action once it is done, on the
    /// reading thread, with the disk unlocked: another handle's I/O can land
    /// between two reads of one refresh.
    after_read: Option<(FailWhen, AfterRead)>,
    /// The next sync this matches runs the action first, on the syncing
    /// thread, with the disk unlocked: a lock-free reader can read what the
    /// writes before the sync left in the page cache.
    before_sync: Option<(FailWhen, AfterRead)>,
}

type FailWhen = Box<dyn FnMut(&IoOp) -> bool + Send>;
type AfterRead = Box<dyn FnOnce() + Send>;

impl Sim {
    /// Record `op`, or fail it when power is (now) cut.
    fn step(&mut self, op: IoOp) -> io::Result<()> {
        if !self.cut && self.budget.is_some_and(|budget| self.trace.len() >= budget) {
            self.cut = true;
        }
        if self.cut {
            return Err(io::Error::other(format!(
                "simulated power cut (refused {op})"
            )));
        }
        if let Some(when) = self.fail_next.as_mut()
            && when(&op)
        {
            self.fail_next = None;
            return Err(io::Error::other(format!("simulated I/O error ({op})")));
        }
        self.trace.push(op);
        Ok(())
    }
}

/// Which description holds each file's lock.
#[derive(Default)]
struct Locks {
    held: Mutex<BTreeMap<usize, usize>>,
    freed: Condvar,
    /// How many descriptions wait for each file's lock, for a test that
    /// must know a writer is blocked.
    waiting: Mutex<BTreeMap<usize, usize>>,
}

/// The files and names a cut leaves, or that a test starts from. Inode
/// numbers are kept so two names for one file stay one file.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Image {
    pub(super) names: BTreeMap<PathBuf, usize>,
    pub(super) inodes: Vec<Vec<u8>>,
}

impl Image {
    /// One file at `path` holding `bytes`.
    pub(super) fn single(path: &Path, bytes: Vec<u8>) -> Self {
        Self {
            names: BTreeMap::from([(path.to_path_buf(), 0)]),
            inodes: vec![bytes],
        }
    }

    pub(super) fn file(&self, path: &Path) -> Option<&[u8]> {
        self.names.get(path).map(|ino| self.inodes[*ino].as_slice())
    }
}

/// One pending operation and the fates it can have.
#[derive(Clone, Debug)]
pub(super) struct Pending {
    /// What the operation is, for a failure message.
    pub(super) what: String,
    /// Fate 0 is "dropped", 1 is "kept", `2 + k` is "torn at the `k`th sector
    /// boundary" (writes only).
    pub(super) fates: usize,
    /// For a write: its offset and the boundaries it can tear at.
    tears: Option<(u64, Vec<u64>)>,
    target: PendingTarget,
}

#[derive(Clone, Debug)]
enum PendingTarget {
    Data { ino: usize, index: usize },
    Dir { index: usize },
}

impl Pending {
    pub(super) fn describe(&self, fate: usize) -> String {
        let how = match fate {
            0 => "dropped".to_string(),
            1 => "kept".to_string(),
            k => match &self.tears {
                Some((offset, points)) => {
                    let at = points[k - 2];
                    format!("torn at {at} (first {} bytes kept)", at - offset)
                }
                None => format!("fate {k}"),
            },
        };
        format!("{} {how}", self.what)
    }
}

/// A simulated disk. Clones share one disk.
#[derive(Clone, Default)]
pub(super) struct SimFs {
    sim: Arc<Mutex<Sim>>,
    locks: Arc<Locks>,
}

/// Numbers each simulated description, for its lock.
static NEXT_DESCRIPTION: AtomicUsize = AtomicUsize::new(0);

impl SimFs {
    /// A disk holding `image`, all of it durable, that cuts power after
    /// `budget` operations (`None`: never).
    pub(super) fn new(image: &Image, budget: Option<usize>) -> Self {
        let inodes = image
            .inodes
            .iter()
            .map(|bytes| Inode {
                live: bytes.clone(),
                durable: bytes.clone(),
                pending: Vec::new(),
            })
            .collect();
        let sim = Sim {
            names: image.names.clone(),
            durable_names: image.names.clone(),
            inodes,
            budget,
            ..Sim::default()
        };
        Self {
            sim: Arc::new(Mutex::new(sim)),
            locks: Arc::default(),
        }
    }

    /// The next operation `when` matches fails once (see `Sim::fail_next`).
    pub(super) fn fail_next(&self, when: impl FnMut(&IoOp) -> bool + Send + 'static) {
        self.lock().fail_next = Some(Box::new(when));
    }

    /// Once the next read `when` matches is done, run `then` on the reading
    /// thread before that read returns (see `Sim::after_read`).
    pub(super) fn after_next_read(
        &self,
        when: impl FnMut(&IoOp) -> bool + Send + 'static,
        then: impl FnOnce() + Send + 'static,
    ) {
        self.lock().after_read = Some((Box::new(when), Box::new(then)));
    }

    /// Before the next sync `when` matches, run `then` on the syncing thread,
    /// with the disk unlocked (see `Sim::before_sync`).
    pub(super) fn before_next_sync(
        &self,
        when: impl FnMut(&IoOp) -> bool + Send + 'static,
        then: impl FnOnce() + Send + 'static,
    ) {
        self.lock().before_sync = Some((Box::new(when), Box::new(then)));
    }

    /// What a cut now would leave if every pending operation were lost: the
    /// durable image alone.
    pub(super) fn durable_image(&self) -> Image {
        let sim = self.lock();
        Image {
            names: sim.durable_names.clone(),
            inodes: sim
                .inodes
                .iter()
                .map(|inode| inode.durable.clone())
                .collect(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Sim> {
        self.sim
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Every operation done so far, in order.
    pub(super) fn trace(&self) -> Vec<IoOp> {
        self.lock().trace.clone()
    }

    /// The trace index of every write so far whose bytes start with
    /// `prefix` (at most 8 bytes).
    pub(super) fn writes_starting_with(&self, prefix: &[u8]) -> Vec<usize> {
        self.lock()
            .write_heads
            .iter()
            .filter(|(_, head)| head.starts_with(prefix))
            .map(|(at, _)| *at)
            .collect()
    }

    /// Every lock taken and released so far, each with the number of trace
    /// operations before it.
    pub(super) fn lock_events(&self) -> Vec<(usize, IoOp)> {
        self.lock().lock_events.clone()
    }

    fn log_lock(&self, event: IoOp) {
        let mut sim = self.lock();
        let at = sim.trace.len();
        sim.lock_events.push((at, event));
    }

    /// How many descriptions are waiting for the lock of `ino` now.
    pub(super) fn lock_waiters(&self, ino: usize) -> usize {
        self.locks
            .waiting
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&ino)
            .copied()
            .unwrap_or(0)
    }

    /// Whether power has been cut.
    pub(super) fn is_cut(&self) -> bool {
        self.lock().cut
    }

    /// What the running process sees, as an image: what a clean shutdown
    /// would leave.
    pub(super) fn live_image(&self) -> Image {
        let sim = self.lock();
        Image {
            names: sim.names.clone(),
            inodes: sim.inodes.iter().map(|inode| inode.live.clone()).collect(),
        }
    }

    /// Every operation not yet durable, in the order it was made.
    pub(super) fn pending(&self) -> Vec<Pending> {
        let sim = self.lock();
        let mut out = Vec::new();
        for (ino, inode) in sim.inodes.iter().enumerate() {
            for (index, op) in inode.pending.iter().enumerate() {
                let (what, fates, tears) = match op {
                    DataOp::Write { offset, bytes } => {
                        let len = bytes.len() as u64;
                        let points = tear_points(*offset, len);
                        (
                            format!("write ino{ino} @{offset}+{len}"),
                            2 + points.len(),
                            Some((*offset, points)),
                        )
                    }
                    DataOp::SetLen(len) => (format!("set_len ino{ino} {len}"), 2, None),
                };
                out.push(Pending {
                    what,
                    fates,
                    tears,
                    target: PendingTarget::Data { ino, index },
                });
            }
        }
        for (index, op) in sim.pending_dir.iter().enumerate() {
            out.push(Pending {
                what: format!("dir entry {op}"),
                fates: 2,
                tears: None,
                target: PendingTarget::Dir { index },
            });
        }
        out
    }

    /// The image left when each pending operation `pending[i]` has fate
    /// `fates[i]`.
    pub(super) fn survivor(&self, pending: &[Pending], fates: &[usize]) -> Image {
        assert_eq!(pending.len(), fates.len(), "one fate per pending op");
        let sim = self.lock();
        let mut inodes: Vec<Vec<u8>> = sim.inodes.iter().map(|i| i.durable.clone()).collect();
        let mut names = sim.durable_names.clone();
        // Apply in the order each file (and the directory) saw them.
        for (item, fate) in pending.iter().zip(fates) {
            match item.target {
                PendingTarget::Data { ino, index } => {
                    let op = &sim.inodes[ino].pending[index];
                    match (*fate, op) {
                        (0, _) => {}
                        (1, op) => apply_data(&mut inodes[ino], op),
                        (k, DataOp::Write { offset, bytes }) => {
                            let points = tear_points(*offset, bytes.len() as u64);
                            let cut = points[k - 2];
                            let keep = (cut - offset) as usize;
                            write_into(&mut inodes[ino], *offset, &bytes[..keep]);
                        }
                        (k, DataOp::SetLen(_)) => panic!("set_len has no fate {k}"),
                    }
                }
                PendingTarget::Dir { index } => {
                    if *fate == 1 {
                        sim.pending_dir[index].apply(&mut names);
                    }
                }
            }
        }
        Image { names, inodes }
    }

    fn inode_of(&self, path: &Path) -> io::Result<usize> {
        self.lock().names.get(path).copied().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("{}: no such file", path.display()),
            )
        })
    }

    fn handle(&self, ino: usize, writable: bool) -> Box<dyn ContainerIo> {
        Box::new(SimFile {
            fs: self.clone(),
            ino,
            writable,
            description: NEXT_DESCRIPTION.fetch_add(1, Ordering::Relaxed),
        })
    }
}

impl ContainerFs for SimFs {
    fn shared(&self) -> Arc<dyn ContainerFs> {
        Arc::new(self.clone())
    }

    fn create_new(&self, path: &Path, mode: u32) -> io::Result<Box<dyn ContainerIo>> {
        let ino = {
            let mut sim = self.lock();
            sim.step(IoOp::CreateNew {
                path: path.to_path_buf(),
                mode,
            })?;
            if sim.names.contains_key(path) {
                return Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!("{}: exists", path.display()),
                ));
            }
            let ino = sim.inodes.len();
            sim.inodes.push(Inode::default());
            sim.names.insert(path.to_path_buf(), ino);
            sim.pending_dir.push(DirOp::Create {
                path: path.to_path_buf(),
                ino,
            });
            ino
        };
        Ok(self.handle(ino, true))
    }

    fn stat_id(&self, path: &Path) -> io::Result<(u64, u64)> {
        self.lock().step(IoOp::StatId {
            path: path.to_path_buf(),
        })?;
        Ok((SIM_DEV, self.inode_of(path)? as u64))
    }

    /// The simulated disk has no symlinks: every path is its own real path.
    /// No operation: resolving a name is not a cut point.
    fn resolve(&self, path: &Path) -> Result<PathBuf, StoreError> {
        Ok(path.to_path_buf())
    }

    fn temp_path(&self, target: &Path, mark: &str) -> Result<PathBuf, StoreError> {
        let name = target
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| backend(format!("{} has no file name", target.display())))?;
        let sim = self.lock();
        // The first free name: a leftover temp a survivor image holds is not
        // reused, and the same workload makes the same names.
        let path = (0u64..)
            .map(|n| target.with_file_name(format!("{name}{mark}0-{n}-00000000")))
            .find(|path| !sim.names.contains_key(path))
            .expect("a free temp name");
        Ok(path)
    }

    fn sweep_stale_temps(&self, _target: &Path, _mark: &str, _what: &str) {}

    fn create_store_file(&self, path: &Path) -> Result<bool, StoreError> {
        let mut sim = self.lock();
        sim.step(IoOp::Create {
            path: path.to_path_buf(),
        })
        .map_err(backend)?;
        if sim.names.contains_key(path) {
            return Ok(false);
        }
        let ino = sim.inodes.len();
        sim.inodes.push(Inode::default());
        sim.names.insert(path.to_path_buf(), ino);
        sim.pending_dir.push(DirOp::Create {
            path: path.to_path_buf(),
            ino,
        });
        Ok(true)
    }

    fn open_rw(&self, path: &Path) -> io::Result<Box<dyn ContainerIo>> {
        self.lock().step(IoOp::OpenRw {
            path: path.to_path_buf(),
        })?;
        Ok(self.handle(self.inode_of(path)?, true))
    }

    fn open_ro(&self, path: &Path) -> io::Result<Box<dyn ContainerIo>> {
        self.lock().step(IoOp::OpenRo {
            path: path.to_path_buf(),
        })?;
        Ok(self.handle(self.inode_of(path)?, false))
    }

    fn sync_dir(&self, dir: &Path) -> Result<(), StoreError> {
        let mut sim = self.lock();
        sim.step(IoOp::SyncDir {
            dir: dir.to_path_buf(),
        })
        .map_err(backend)?;
        let (now, later): (Vec<DirOp>, Vec<DirOp>) = std::mem::take(&mut sim.pending_dir)
            .into_iter()
            .partition(|op| op.touches(dir));
        for op in &now {
            op.apply(&mut sim.durable_names);
        }
        sim.pending_dir = later;
        Ok(())
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let mut sim = self.lock();
        sim.step(IoOp::Rename {
            from: from.to_path_buf(),
            to: to.to_path_buf(),
        })?;
        let ino = sim.names.remove(from).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("{}: no such file", from.display()),
            )
        })?;
        sim.names.insert(to.to_path_buf(), ino);
        sim.pending_dir.push(DirOp::Rename {
            from: from.to_path_buf(),
            to: to.to_path_buf(),
        });
        Ok(())
    }

    fn link(&self, from: &Path, to: &Path) -> io::Result<()> {
        let mut sim = self.lock();
        sim.step(IoOp::Link {
            from: from.to_path_buf(),
            to: to.to_path_buf(),
        })?;
        let ino = *sim.names.get(from).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("{}: no such file", from.display()),
            )
        })?;
        if sim.names.contains_key(to) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("{}: exists", to.display()),
            ));
        }
        sim.names.insert(to.to_path_buf(), ino);
        sim.pending_dir.push(DirOp::Link {
            path: to.to_path_buf(),
            ino,
        });
        Ok(())
    }

    fn unlink(&self, path: &Path) -> io::Result<()> {
        let mut sim = self.lock();
        sim.step(IoOp::Unlink {
            path: path.to_path_buf(),
        })?;
        if sim.names.remove(path).is_none() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("{}: no such file", path.display()),
            ));
        }
        sim.pending_dir.push(DirOp::Unlink {
            path: path.to_path_buf(),
        });
        Ok(())
    }

    fn warn_if_loose(&self, _path: &Path) {}
}

/// One open description of a simulated file.
struct SimFile {
    fs: SimFs,
    ino: usize,
    writable: bool,
    description: usize,
}

impl SimFile {
    fn writable(&self) -> io::Result<()> {
        if self.writable {
            Ok(())
        } else {
            Err(io::Error::other("write through a read-only description"))
        }
    }

    fn sync(&self, op: IoOp) -> io::Result<()> {
        let before = {
            let mut sim = self.fs.lock();
            let fire = sim.before_sync.as_mut().is_some_and(|(when, _)| when(&op));
            if fire {
                sim.before_sync.take().map(|(_, then)| then)
            } else {
                None
            }
        };
        if let Some(then) = before {
            then();
        }
        let mut sim = self.fs.lock();
        sim.step(op)?;
        let inode = &mut sim.inodes[self.ino];
        inode.durable = inode.live.clone();
        inode.pending.clear();
        Ok(())
    }
}

impl ContainerIo for SimFile {
    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<()> {
        let mut sim = self.fs.lock();
        sim.step(IoOp::Write {
            ino: self.ino,
            offset,
            len: buf.len() as u64,
        })?;
        self.writable()?;
        let at = sim.trace.len() - 1;
        sim.write_heads.push((at, buf[..buf.len().min(8)].to_vec()));
        let op = DataOp::Write {
            offset,
            bytes: buf.to_vec(),
        };
        let inode = &mut sim.inodes[self.ino];
        apply_data(&mut inode.live, &op);
        inode.pending.push(op);
        Ok(())
    }

    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        let op = IoOp::Read {
            ino: self.ino,
            offset,
            len: buf.len() as u64,
        };
        let then = {
            let mut sim = self.fs.lock();
            sim.step(op.clone())?;
            let live = &sim.inodes[self.ino].live;
            let start = offset as usize;
            let end = start + buf.len();
            if end > live.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                ));
            }
            buf.copy_from_slice(&live[start..end]);
            let fire = sim.after_read.as_mut().is_some_and(|(when, _)| when(&op));
            if fire {
                sim.after_read.take().map(|(_, then)| then)
            } else {
                None
            }
        };
        if let Some(then) = then {
            then();
        }
        Ok(())
    }

    fn sync_data(&self) -> io::Result<()> {
        self.sync(IoOp::SyncData { ino: self.ino })
    }

    fn sync_all(&self) -> io::Result<()> {
        self.sync(IoOp::SyncAll { ino: self.ino })
    }

    fn len(&self) -> io::Result<u64> {
        let mut sim = self.fs.lock();
        sim.step(IoOp::Len { ino: self.ino })?;
        Ok(sim.inodes[self.ino].live.len() as u64)
    }

    fn file_id(&self) -> io::Result<(u64, u64)> {
        // An `fstat`: recorded and refused after a cut, and never pending.
        self.fs.lock().step(IoOp::FileId { ino: self.ino })?;
        Ok((SIM_DEV, self.ino as u64))
    }

    fn set_len(&self, len: u64) -> io::Result<()> {
        let mut sim = self.fs.lock();
        sim.step(IoOp::SetLen { ino: self.ino, len })?;
        self.writable()?;
        let op = DataOp::SetLen(len);
        let inode = &mut sim.inodes[self.ino];
        apply_data(&mut inode.live, &op);
        inode.pending.push(op);
        Ok(())
    }

    fn lock_exclusive(&self) -> io::Result<()> {
        let locks = &self.fs.locks;
        let mut held = locks
            .held
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let busy = |held: &BTreeMap<usize, usize>| {
            held.get(&self.ino)
                .is_some_and(|holder| *holder != self.description)
        };
        if busy(&held) {
            let count = |delta: isize| {
                let mut waiting = locks
                    .waiting
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let entry = waiting.entry(self.ino).or_default();
                *entry = entry.saturating_add_signed(delta);
            };
            count(1);
            while busy(&held) {
                held = locks
                    .freed
                    .wait(held)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
            count(-1);
        }
        // Logged once held, in the order the locks were taken.
        held.insert(self.ino, self.description);
        self.fs.log_lock(IoOp::Lock { ino: self.ino });
        Ok(())
    }

    fn unlock(&self) {
        let locks = &self.fs.locks;
        let mut held = locks
            .held
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if held.get(&self.ino) != Some(&self.description) {
            return;
        }
        self.fs.log_lock(IoOp::Unlock { ino: self.ino });
        held.remove(&self.ino);
        locks.freed.notify_all();
    }
}

/// SplitMix64: a small, fixed PRNG, so a sampled crash reproduces from its seed.
pub(super) struct SplitMix64(pub(super) u64);

impl SplitMix64 {
    pub(super) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(super) fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// The fate vectors to try for `pending`: every combination when there are
/// at most `cap`, otherwise all-kept, all-dropped and `cap - 2` combinations
/// drawn from `SplitMix64(seed)`. The `bool` is whether it sampled.
pub(super) fn fate_vectors(pending: &[Pending], cap: usize, seed: u64) -> (Vec<Vec<usize>>, bool) {
    let total = pending
        .iter()
        .try_fold(1usize, |acc, item| acc.checked_mul(item.fates));
    match total {
        Some(total) if total <= cap => {
            let mut out = Vec::with_capacity(total);
            let mut fates = vec![0usize; pending.len()];
            for _ in 0..total {
                out.push(fates.clone());
                // Mixed-radix increment.
                for (digit, item) in fates.iter_mut().zip(pending) {
                    *digit += 1;
                    if *digit < item.fates {
                        break;
                    }
                    *digit = 0;
                }
            }
            (out, false)
        }
        _ => {
            let mut rng = SplitMix64(seed);
            let mut out = vec![vec![1; pending.len()], vec![0; pending.len()]];
            while out.len() < cap.max(2) {
                out.push(pending.iter().map(|item| rng.below(item.fates)).collect());
            }
            (out, true)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(name: &str) -> PathBuf {
        PathBuf::from("/sim").join(name)
    }

    #[test]
    fn a_synced_write_survives_and_an_unsynced_one_has_three_kinds_of_fate() {
        let fs = SimFs::default();
        assert!(fs.create_store_file(&path("f")).unwrap());
        fs.sync_dir(Path::new("/sim")).unwrap();
        let file = fs.open_rw(&path("f")).unwrap();
        file.write_at(&[1; 600], 0).unwrap();
        file.sync_data().unwrap();
        // 1000..2100 crosses the boundaries at 1024, 1536 and 2048.
        file.write_at(&[2; 1100], 1000).unwrap();
        let pending = fs.pending();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].fates, 2 + 3);
        let (fates, sampled) = fate_vectors(&pending, 64, 0);
        assert!(!sampled);
        assert_eq!(fates.len(), 5);
        assert_eq!(
            pending[0].describe(2),
            "write ino0 @1000+1100 torn at 1024 (first 24 bytes kept)"
        );
        let at = |fate: usize| {
            fs.survivor(&pending, &[fate])
                .file(&path("f"))
                .unwrap()
                .to_vec()
        };
        assert_eq!(at(0), vec![1; 600], "dropped");
        assert_eq!(at(1).len(), 2100, "kept");
        let torn = at(2);
        assert_eq!(torn.len(), 1024, "torn at the first boundary");
        assert_eq!(&torn[1000..], &[2; 24]);
        assert_eq!(&torn[600..1000], &[0; 400][..]);
    }

    #[test]
    fn a_name_is_durable_only_after_its_directory_is_synced() {
        let fs = SimFs::default();
        fs.create_store_file(&path("a")).unwrap();
        let file = fs.open_rw(&path("a")).unwrap();
        file.write_at(b"x", 0).unwrap();
        file.sync_all().unwrap();
        let pending = fs.pending();
        assert_eq!(pending.len(), 1, "only the entry is pending: {pending:?}");
        assert!(fs.survivor(&pending, &[0]).file(&path("a")).is_none());
        assert_eq!(
            fs.survivor(&pending, &[1]).file(&path("a")),
            Some(&b"x"[..])
        );

        fs.sync_dir(Path::new("/sim")).unwrap();
        fs.link(&path("a"), &path("b")).unwrap();
        fs.unlink(&path("a")).unwrap();
        fs.rename(&path("b"), &path("c")).unwrap();
        assert!(
            fs.link(&path("c"), &path("c")).is_err(),
            "link refuses an existing name"
        );
        let pending = fs.pending();
        assert_eq!(pending.len(), 3);
        let all = fs.survivor(&pending, &[1, 1, 1]);
        assert_eq!(all.names.keys().collect::<Vec<_>>(), vec![&path("c")]);
        let none = fs.survivor(&pending, &[0, 0, 0]);
        assert_eq!(none.names.keys().collect::<Vec<_>>(), vec![&path("a")]);
        // The link kept, the unlink and rename lost: both names, one inode.
        let linked = fs.survivor(&pending, &[1, 0, 0]);
        assert_eq!(linked.names[&path("a")], linked.names[&path("b")]);
    }

    #[test]
    fn power_stays_cut_for_every_handle_and_set_len_can_be_lost() {
        let image = Image::single(&path("f"), vec![7; 100]);
        let fs = SimFs::new(&image, Some(3));
        let rw = fs.open_rw(&path("f")).unwrap();
        let ro = fs.open_ro(&path("f")).unwrap();
        rw.set_len(4096).unwrap();
        assert!(rw.sync_all().is_err(), "the fourth op is past the budget");
        assert!(ro.len().is_err(), "the cut holds for the other description");
        assert!(fs.is_cut());
        assert_eq!(fs.trace().len(), 3);
        let pending = fs.pending();
        assert_eq!(pending[0].fates, 2);
        assert_eq!(
            fs.survivor(&pending, &[0]).file(&path("f")).unwrap().len(),
            100
        );
        assert_eq!(
            fs.survivor(&pending, &[1]).file(&path("f")).unwrap().len(),
            4096
        );
        assert!(ro.write_at(b"x", 0).is_err());
    }

    #[test]
    fn past_the_cap_the_fates_are_a_seeded_sample_that_repeats() {
        let fs = SimFs::default();
        fs.create_store_file(&path("f")).unwrap();
        let file = fs.open_rw(&path("f")).unwrap();
        for i in 0..8u64 {
            file.write_at(&[i as u8; 1024], i * 4096).unwrap();
        }
        let pending = fs.pending();
        // 2 entry fates x 3^8 write fates.
        let (first, sampled) = fate_vectors(&pending, 32, 42);
        assert!(sampled);
        assert_eq!(first.len(), 32);
        assert_eq!(first[0], vec![1; pending.len()]);
        assert_eq!(first[1], vec![0; pending.len()]);
        assert_eq!(
            fate_vectors(&pending, 32, 42).0,
            first,
            "same seed, same sample"
        );
        assert_ne!(fate_vectors(&pending, 32, 43).0, first);
    }
}
