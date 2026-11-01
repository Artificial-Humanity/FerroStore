//! Document store and FerroMQ log in a `.fls` file, and a socket server.
//! See `docs/ferrostore-spec.md`.

mod auth;
mod bind;
mod columnar;
mod compact;
mod compat;
mod container_io;
mod doc;
mod group;
mod inspect;
mod layout;
mod manifest_ops;
mod migrate;
mod remote_txn;
mod retry;
mod server;
mod session;
mod state;
mod sync;
mod tls;
mod traits_catalog;
mod traits_state;
mod traits_wire;
mod traits_workflow;
mod transaction;
mod vector;
mod verify;
mod ws;

#[cfg(test)]
mod crash_sim;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_auth;
#[cfg(test)]
mod tests_cache;
#[cfg(test)]
mod tests_checkpoint;
#[cfg(test)]
mod tests_compact;
#[cfg(test)]
mod tests_cow;
#[cfg(test)]
mod tests_crash;
#[cfg(test)]
mod tests_group;
#[cfg(test)]
mod tests_reentry;
#[cfg(test)]
mod tests_retry;
#[cfg(test)]
mod tests_session;
#[cfg(test)]
mod tests_tls;
#[cfg(test)]
mod tests_track0;
#[cfg(test)]
mod tests_txn;
#[cfg(test)]
mod tests_verify;

use container_io::{ContainerFs, ContainerIo, RealFs};
use fl_core::StoreError;
use fl_core::ids::{GateId, ProjectId};
use fl_core::iri::Iri;
use group::Staged;
pub use layout::Op;
use layout::{
    HEADER_BYTES, KNOWN_INCOMPAT, SLOT_LEN, SlotRead, Superblock, backend, corrupt,
    decode_snapshot, decode_wal_prefix, encode_frame, encode_snapshot, encode_snapshot_v2,
    encode_snapshot_v3, encode_superblock, parse_superblock,
};
use serde_json::{Value, json};
use state::{
    DocState, Docs, MQ_HIGH_WATER_ID, Overlay, apply_commit, apply_frames, mq_hw_collection,
    mq_message_collection, next_mq_seq, publish_overlay, stage_ops, unread_mq,
};

pub use auth::{ConnectOptions, ServerAuth};
pub use columnar::{
    AttemptRollup, ColumnDef, ColumnFilter, ColumnPredicate, ColumnScan, ColumnType, ColumnValue,
    GateRunRollup, ZoneMap,
};
pub use compact::{AUTO_COMPACT_BYTES, StoreStats};
pub use inspect::{
    CollectionReport, ColumnTableReport, Contents, Inspection, SlotReport, SuperblockReport,
    VectorCollectionReport, inspect_at,
};
pub use migrate::migrate_redb_to_ferrostore;
pub use server::FerroStoreServer;
pub use state::MqRecord;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::ThreadId;
pub use sync::{EnvelopeKind, RecordEnvelope, SyncAdapter, SyncBatch, SyncMode, SyncOp};
pub use transaction::Transaction;
pub use vector::{Fusion, KnowledgeHit, VectorHit, VectorMetric, VectorStorage};
pub use verify::{
    PER_ITEM_SAMPLE, ParityReport, TableParity, migrate_verified, verify_redb_vs_ferrostore,
};

/// Holds the exclusive `flock` of one container description until dropped.
struct FlockGuard<'a> {
    file: &'a dyn ContainerIo,
}

impl<'a> FlockGuard<'a> {
    fn acquire(file: &'a dyn ContainerIo) -> Result<Self, StoreError> {
        file.lock_exclusive()
            .map_err(|err| backend(format!("exclusive flock failed: {err}")))?;
        Ok(Self { file })
    }
}

impl Drop for FlockGuard<'_> {
    fn drop(&mut self) {
        self.file.unlock();
    }
}

/// `(dev, ino)`: which file a description reads, or a path names.
type FileId = (u64, u64);

/// One open description of the store file and the file it reads.
#[derive(Clone)]
struct Descriptor {
    file: Arc<dyn ContainerIo>,
    /// `(dev, ino)` of `file`, read once (`fstat`) when it was opened.
    id: FileId,
    /// The write description only: this handle made the directory entry
    /// that names `file` durable (spec §5). `false` for a description opened
    /// on a file that already existed (`open`, or a reopen after a swap):
    /// another handle may have renamed it into place and failed its
    /// directory sync, so after a crash the path could name the file before
    /// it again. [`Disk::lock_current`] syncs the directory before the first
    /// write through such a description. A read description never needs it.
    name_durable: bool,
}

impl Descriptor {
    fn new(file: Box<dyn ContainerIo>, path: &Path) -> Result<Self, StoreError> {
        let file: Arc<dyn ContainerIo> = Arc::from(file);
        let id = file
            .file_id()
            .map_err(|err| backend(format!("fstat {}: {err}", path.display())))?;
        Ok(Self {
            file,
            id,
            name_durable: false,
        })
    }
}

/// The exclusive `flock` of one description, held until dropped. It owns
/// the description, so the slot it came from can be swapped to another file
/// (spec §5) while the lock is still held.
struct HeldFlock {
    file: Arc<dyn ContainerIo>,
    /// `(dev, ino)` of `file`, which the path named once the lock was taken.
    id: FileId,
}

impl HeldFlock {
    fn acquire(descriptor: &Descriptor) -> Result<Self, StoreError> {
        descriptor
            .file
            .lock_exclusive()
            .map_err(|err| backend(format!("exclusive flock failed: {err}")))?;
        Ok(Self {
            file: Arc::clone(&descriptor.file),
            id: descriptor.id,
        })
    }

    fn file(&self) -> &dyn ContainerIo {
        &*self.file
    }
}

impl Drop for HeldFlock {
    fn drop(&mut self) {
        self.file.unlock();
    }
}

enum Inner {
    /// The live state. A reader clones the `Arc`; a commit replaces it
    /// copy-on-write, so a reader's view never changes under it.
    Memory(Mutex<Arc<DocState>>),
    /// Boxed: a disk handle's descriptions, cache and group are much larger
    /// than the other variants.
    Disk(Box<Disk>),
    Remote(server::RemoteClient),
}

#[derive(Default)]
struct SyncSlot {
    mode: Option<SyncMode>,
    adapter: Option<Arc<dyn SyncAdapter>>,
    synced_seq: u64,
    pending: Vec<SyncBatch>,
    /// `on_commit` is running. Another thread's `sync_flush` waits until this is clear.
    notify_in_flight: bool,
    /// Thread that set `notify_in_flight`. Cleared with the flag.
    notify_owner: Option<ThreadId>,
}

impl SyncSlot {
    /// No queued batch and no `on_commit` running. A batch that a notify owns is
    /// in neither `pending` nor `synced_seq` until the notify returns, so a check
    /// of `pending` alone would call that window idle.
    fn is_idle(&self) -> bool {
        self.pending.is_empty() && !self.notify_in_flight
    }
}

struct Committed {
    seq: u64,
    ops: Vec<Op>,
}

/// What registering a commit with the sync adapter leaves its committer to
/// do once every store lock is released ([`FerroStore::deliver_owned`]).
#[derive(Default)]
enum Registered {
    /// Nothing: a silent commit, no adapter, or a mode that leaves the batch
    /// queued for `sync_flush`.
    #[default]
    Nothing,
    /// `WriteThrough` with nothing queued or in flight: the committer owns
    /// the notify (`notify_in_flight` is set for its thread) and sends the
    /// batch itself.
    Owned(Arc<dyn SyncAdapter>, SyncBatch),
    /// `WriteThrough` behind a queued batch or a notify in flight: the batch
    /// was queued at this sequence, and the committer drains the queue until
    /// it is sent.
    Queued(u64),
}

/// A remote commit the server made
/// ([`FerroStore::commit_remote`]).
struct RemoteCommit {
    seq: u64,
    /// This handle's own batch for the commit.
    registered: Registered,
    /// The server's `Undelivered`: its own adapter refused the batch.
    server: Option<StoreError>,
}

/// How a compare-and-swap commit went.
#[cfg(test)]
enum SeqCommit {
    Lost,
    Same,
    Wrote,
}

/// The two descriptions of one `.fls` file. Production holds `File`s; the
/// crash harness holds simulated files (`container_io`, `crash_sim`).
///
/// Both descriptions are swappable (spec §5): a compaction renames a new
/// file over the path, and every handle, this one or another process's,
/// moves to it once it sees `stat(path)` differ from its description
/// ([`Self::lock_current`] for a writer, [`Self::current_read`] for a
/// reader). A description that is swapped out is closed only when its last
/// user drops it, so a reader already reading the old file finishes on it.
struct Disk {
    /// The file system the store lives on, to reopen the path after a swap.
    fs: Arc<dyn ContainerFs>,
    path: PathBuf,
    /// The read-write description every commit and checkpoint goes through.
    /// Swapped only with this mutex held, which every writer holds anyway.
    write: Mutex<Descriptor>,
    /// Separate `O_RDONLY` description. Reads never flock it. The mutex is
    /// held only to clone or swap the `Arc`, never across file I/O, so a read
    /// stays free of any lock a writer holds.
    read: Mutex<Descriptor>,
    /// The last state this handle loaded or committed (spec §4.1). The mutex
    /// is held only to clone, take or swap the `Arc`, never across file I/O.
    cache: Mutex<Option<Arc<DiskCache>>>,
    /// Writers queued for group commit (spec §4.3).
    group: group::Group,
    /// Set by a failed sync of a publish or a checkpoint.
    poison: Poison,
    /// The WAL length at which the last auto-checkpoint failed, or 0 when the
    /// last checkpoint (auto or manual) succeeded or none has failed. An
    /// auto-checkpoint is tried again only once the WAL has grown past it by
    /// another threshold ([`checkpoint_due`]). Written only with the write
    /// mutex held.
    checkpoint_failed_at: AtomicU64,
    /// The dead bytes at which the last auto-compaction failed, or 0: the
    /// same back-off as [`Self::checkpoint_failed_at`], for compaction
    /// (spec §5, [`compact::compact_due`]). Written only with the write
    /// mutex held.
    compact_failed_at: AtomicU64,
    /// [`AUTO_CHECKPOINT_BYTES`], lowered by a test so a small workload
    /// crosses it.
    #[cfg(test)]
    auto_checkpoint_floor: std::sync::atomic::AtomicU64,
    /// [`AUTO_COMPACT_BYTES`], lowered by a test.
    #[cfg(test)]
    auto_compact_floor: std::sync::atomic::AtomicU64,
}

/// The WAL length a commit's auto-checkpoint (spec §4.4) never fires at or
/// below: once a commit has published and released its locks, the writer
/// checkpoints when the WAL is longer than `max(snapshot_length, this)`. The
/// snapshot term keeps the work of checkpoints linear in the bytes written:
/// a checkpoint, which costs O(state), runs only after at least as many
/// WAL bytes as the snapshot holds. The floor keeps a small store from
/// checkpointing every few commits. A reopen replays at most that much WAL.
pub(super) const AUTO_CHECKPOINT_BYTES: u64 = 4 << 20;

/// Set once a `sync_data` of a publish or a checkpoint failed on this handle.
///
/// After a failed `fsync` Linux may mark the dirty pages clean without having
/// written them (fsyncgate): the page cache can then show the other slot at
/// the new epoch while the disk still holds the slot before it, and the next
/// publish would overwrite the only durable copy of the last acknowledged
/// epoch. So every later write and checkpoint on the handle is refused until
/// the store is reopened. Reads go on, from the cache only: it holds the last
/// state whose syncs all returned, and the file's page cache is no longer
/// trusted. A lock-free read that loaded the slots between the slot write
/// and the failed sync may return that unacknowledged state for its own
/// call, as any read that finished before the failure did; it is not kept
/// ([`Disk::settle_failed_publish`]), so no read after the poison serves it.
struct Poison {
    set: AtomicBool,
    path: String,
}

impl Poison {
    fn new(path: &Path) -> Self {
        Self {
            set: AtomicBool::new(false),
            path: path.display().to_string(),
        }
    }

    fn is_set(&self) -> bool {
        self.set.load(Ordering::Acquire)
    }

    /// `Err` once a sync has failed: the write must not start.
    fn check(&self) -> Result<(), StoreError> {
        if self.is_set() {
            return Err(backend(format!(
                "FerroStore handle poisoned after a failed fsync at {}: reopen the store",
                self.path
            )));
        }
        Ok(())
    }

    /// `file.sync_data()`, setting the poison when it fails. The failing call
    /// itself reports the I/O error.
    fn sync_data(&self, file: &dyn ContainerIo) -> Result<(), StoreError> {
        file.sync_data().map_err(|err| {
            self.mark();
            io_err(err)
        })
    }

    /// [`Self::sync_data`] with `sync_all`, for a file that was just created
    /// (a compaction's sibling, spec §5).
    fn sync_all(&self, file: &dyn ContainerIo) -> Result<(), StoreError> {
        file.sync_all().map_err(|err| {
            self.mark();
            io_err(err)
        })
    }

    /// Poison the handle: a sync failed. A compaction's directory sync is
    /// one (spec §5).
    fn mark(&self) {
        self.set.store(true, Ordering::Release);
    }
}

/// What a publish changed, for [`Disk::install_commit`].
enum Change {
    /// One commit's overlay, staged over the base state, at this sequence.
    Overlay(Overlay, u64),
    /// The base state with every change of the publish already applied.
    State(Arc<DocState>),
}

/// One loaded state and the superblock it is the state of. Never changed in
/// place once shared: a refresh or a commit makes the next one.
#[derive(Clone)]
struct DiskCache {
    /// `(dev, ino)` of the description `state` was read or written through.
    file_id: (u64, u64),
    sb: Superblock,
    state: Arc<DocState>,
    /// WAL bytes, from `sb.wal_offset`, folded into `state`: always the end
    /// of a whole frame. Equal to `sb.wal_length` unless that WAL ends torn.
    wal_applied: u64,
}

impl Disk {
    /// The current state through `file` (spec §4.1), and the index of the
    /// slot it was published in, for a writer: the caller holds the `flock`,
    /// so the slots cannot change under the read and name the newest state.
    /// Reads the two slots, then [`Self::advance`]s the cache to them.
    ///
    /// The caller took the `flock` through [`Self::lock_current`], which
    /// checked that the path still names `file` (spec §5): a compaction
    /// cannot swap the file while the lock is held, so `file` is the store.
    fn refresh(
        &self,
        file: &dyn ContainerIo,
        file_id: FileId,
    ) -> Result<(usize, Arc<DiskCache>), StoreError> {
        let cached = lock_mutex(&self.cache).clone();
        let (index, sb) = latest_superblock(file)?;
        debug_assert!(
            cached
                .as_ref()
                .is_none_or(|held| held.file_id != file_id || held.sb.epoch <= sb.epoch),
            "under the flock the slots named an epoch older than the cache"
        );
        Ok((index, self.advance(file, file_id, cached, sb)?))
    }

    /// The current state for a reader, which takes no lock (spec §4.1).
    ///
    /// The two slots are read one after the other, so a slot write can land
    /// between them: the reader reads slot 0, another handle publishes into
    /// slot 0 and starts a write of slot 1, and the reader's slot 1 read is
    /// torn. It then falls back to the slot 0 it read first, which can be
    /// older than a state this handle already holds. That older state is
    /// never returned: when the cache holds a state at a later epoch, the
    /// cache is the answer. It is a whole published state, and its extents
    /// are still valid: no extent of a file is ever reused, and a §5
    /// compaction writes a new file instead, which a handle that still holds
    /// the old one's description goes on reading until it drops it.
    ///
    /// So a handle's reads never go back: each returns the cache it started
    /// from or a later state, and the cache only moves forward
    /// ([`install_newer`]). The one exception is documented on [`Poison`]:
    /// once a failed publish poisons the handle, the cache is set back to the
    /// last acknowledged epoch E ([`Self::settle_failed_publish`]), so a
    /// reader that once returned the unacknowledged E+1 may afterwards see E.
    ///
    /// A poisoned handle (see [`Poison`]) reads only its cache: after a failed
    /// sync the page cache may not match the disk.
    ///
    /// Before the slots are read, the read description is checked against
    /// the path ([`Self::current_read`], spec §5): after a compaction the
    /// path names a new file, and the read moves to it. A cached state of
    /// the file just read at a later epoch is still the answer; a cached
    /// state of another file is not, whatever its epoch: the file the path
    /// names is read and replaces it, so a file put in place at a lower
    /// epoch is seen at once ([`install_newer`]). A compaction's new file
    /// starts at the old file's last epoch + 1, so across a compaction the
    /// epoch rises too.
    fn refresh_read(&self) -> Result<Arc<DiskCache>, StoreError> {
        Ok(self.refresh_read_on()?.0)
    }

    /// [`Self::refresh_read`], and the read description it read through.
    fn refresh_read_on(&self) -> Result<(Arc<DiskCache>, Descriptor), StoreError> {
        let cached = lock_mutex(&self.cache).clone();
        if self.poison.is_set()
            && let Some(cached) = cached
        {
            let read = lock_mutex(&self.read).clone();
            return Ok((cached, read));
        }
        let read = self.current_read()?;
        let (_, sb) = latest_superblock(&*read.file)?;
        // A later epoch of the same file: the slot read fell back past a torn
        // slot. A state of another file is no answer here: the path names
        // the file just read ([`install_newer`]).
        if let Some(held) = &cached
            && held.file_id == read.id
            && held.sb.epoch > sb.epoch
        {
            return Ok((Arc::clone(held), read));
        }
        let fresh = self.advance(&*read.file, read.id, cached, sb)?;
        Ok((fresh, read))
    }

    /// The read description, moved to the file the path names now when a
    /// compaction swapped it (spec §5): `stat(path)` is compared with the
    /// description's `(dev, ino)`, and on a difference the path is opened
    /// again. The swap is made only while the slot still holds the
    /// description this call started from; a reader still using the old
    /// description finishes on it.
    ///
    /// Nothing at the path is no swap: a compaction's rename replaces the
    /// name in one step, so there is no newer file to move to, and the read
    /// goes on through the file the description holds. A writer refuses it
    /// ([`Self::lock_current`]): a commit into an unlinked file is lost.
    fn current_read(&self) -> Result<Descriptor, StoreError> {
        let held = lock_mutex(&self.read).clone();
        let at_path = match self.fs.stat_id(&self.path) {
            Ok(id) => id,
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(held),
            Err(err) => {
                return Err(backend(format!("stat {}: {err}", self.path.display())));
            }
        };
        if at_path == held.id {
            return Ok(held);
        }
        let opened = self
            .fs
            .open_ro(&self.path)
            .map_err(|err| backend(format!("open {}: {err}", self.path.display())))?;
        let fresh = Descriptor::new(opened, &self.path)?;
        let mut slot = lock_mutex(&self.read);
        if Arc::ptr_eq(&slot.file, &held.file) {
            *slot = fresh.clone();
        }
        Ok(fresh)
    }

    /// `(dev, ino)` of the file the store's path names now.
    fn stat_path(&self) -> Result<FileId, StoreError> {
        self.fs
            .stat_id(&self.path)
            .map_err(|err| backend(format!("stat {}: {err}", self.path.display())))
    }

    /// Take the `flock` on the file the path names now (spec §5): lock
    /// `write`'s description, then compare `stat(path)` with it. When a
    /// compaction swapped the file, unlock, reopen the path for both the
    /// write and the read description, and try again. Without this, a
    /// writer that waited on the old file's lock through the rename would
    /// commit into the unlinked file and its write would be lost.
    ///
    /// The caller holds the write mutex, whose `write` this is. Once this
    /// returns, no compaction can swap the file until the lock is dropped:
    /// a compactor takes the same lock, on the same file, first.
    ///
    /// Every caller is a writer, so this is also where a handle makes the
    /// name of the file it writes durable (spec §5): a handle never commits
    /// into a file whose directory entry it has not itself made durable. A
    /// compactor whose directory sync failed after its rename leaves the
    /// path naming a file whose name a crash can still take back (to the
    /// old, unlinked file, without every commit made into the new one). So
    /// before the first write through a description that is not
    /// [`Descriptor::name_durable`], under the lock and once the path is
    /// known to name it, the directory is synced. A failure refuses the
    /// write and poisons the handle, as every failed sync does ([`Poison`]):
    /// after a failed `fsync` a later one can return success without the
    /// entry having reached the disk, so no retry is trusted.
    fn lock_current(&self, write: &mut Descriptor) -> Result<HeldFlock, StoreError> {
        loop {
            let held = HeldFlock::acquire(write)?;
            if self.stat_path()? == write.id {
                if !write.name_durable {
                    self.fs.sync_dir(&parent_dir(&self.path)).map_err(|err| {
                        self.poison.mark();
                        backend(format!(
                            "{} was not written: its directory entry could not be made \
                                 durable before the first write through this handle, so a \
                                 crash could lose the write: {err}",
                            self.path.display()
                        ))
                    })?;
                    write.name_durable = true;
                }
                return Ok(held);
            }
            drop(held);
            self.reopen(write)?;
        }
    }

    /// Open the path again for writes (into `write`) and reads (into the read
    /// slot), after a compaction swapped the file under it. The new write
    /// description is not [`Descriptor::name_durable`]: another handle made
    /// the name.
    fn reopen(&self, write: &mut Descriptor) -> Result<(), StoreError> {
        let open_err = |err: io::Error| backend(format!("open {}: {err}", self.path.display()));
        let rw = self.fs.open_rw(&self.path).map_err(open_err)?;
        *write = Descriptor::new(rw, &self.path)?;
        let ro = self.fs.open_ro(&self.path).map_err(open_err)?;
        *lock_mutex(&self.read) = Descriptor::new(ro, &self.path)?;
        Ok(())
    }

    /// The state `sb`, read through `file`, from `cached`:
    ///
    /// - the same file and the same superblock as the cache: the cached
    ///   state, with no other I/O;
    /// - the same snapshot extent and WAL offset with a longer WAL: only the
    ///   new WAL bytes are read and applied, copy-on-write, to the cached
    ///   state;
    /// - anything else: a full load of the snapshot and the WAL.
    ///
    /// Every byte read lies inside the extents of `sb`, so a reader needs no
    /// lock. The result is never newer than `sb`.
    fn advance(
        &self,
        file: &dyn ContainerIo,
        file_id: (u64, u64),
        cached: Option<Arc<DiskCache>>,
        sb: Superblock,
    ) -> Result<Arc<DiskCache>, StoreError> {
        let fresh = match cached {
            Some(cached) if cached.file_id == file_id && cached.sb == sb => {
                return Ok(cached);
            }
            Some(cached) if cached.file_id == file_id && tail_follows(&cached, &sb) => {
                // Any doubt about the tail is settled by the full load, which
                // reports the same error a tail would have, or none.
                match replay_tail(file, &cached, sb) {
                    Ok(Some(next)) => next,
                    Ok(None) | Err(_) => load_full(file, file_id, sb)?,
                }
            }
            _ => load_full(file, file_id, sb)?,
        };
        let fresh = Arc::new(fresh);
        let mut slot = lock_mutex(&self.cache);
        // A failed publish sets the poison, then takes this mutex
        // ([`Self::settle_failed_publish`]): either that runs first, and this
        // read sees the poison, or this install runs first, and that takes it
        // back out. So a poisoned handle keeps no state it read around the
        // failed sync: its slots may name an epoch whose sync never returned.
        // This one call still returns the cache, not what it read.
        if self.poison.is_set()
            && let Some(held) = slot.as_ref()
        {
            return Ok(Arc::clone(held));
        }
        // The file the path names, as the last swap check found it. Taken
        // under the cache mutex (never the other way round), so the
        // compaction that moved it has either installed its state already
        // or installs after this one.
        let current = lock_mutex(&self.read).id;
        install_newer(&mut slot, Arc::clone(&fresh), current);
        Ok(fresh)
    }

    /// After a publish or a checkpoint through `write` failed: once the
    /// failure poisoned the handle, hold the cache at `acknowledged`, the
    /// state the writer published from, whose syncs all returned. A lock-free
    /// reader that loaded the slots between the slot write and its failed
    /// sync may have installed the newer epoch: it is taken back out here, so
    /// every later read of the poisoned handle serves `acknowledged` (see
    /// [`Poison`]). The caller still holds the `flock`, so no other handle
    /// published past `acknowledged` meanwhile. A failure that did not
    /// poison leaves the cache as it is.
    fn settle_failed_publish(&self, acknowledged: &DiskCache) {
        if !self.poison.is_set() {
            return;
        }
        // The poison was set before this lock: a reader's install that takes
        // the mutex after it sees the poison ([`Self::advance`]).
        let mut slot = lock_mutex(&self.cache);
        let newer = slot.as_ref().is_none_or(|held| {
            held.file_id == acknowledged.file_id && held.sb.epoch > acknowledged.sb.epoch
        });
        if newer {
            *slot = Some(Arc::new(acknowledged.clone()));
        }
    }

    /// Every publish of frames through `write`, from `base` (the state the
    /// slot `index` names) into the other slot: the one way a commit, a
    /// group chunk, a cache fill or an applied sync batch reaches the file.
    /// On any failure it runs [`Self::settle_failed_publish`] with `base`, so
    /// no call site can leave a reader's unacknowledged epoch in the cache of
    /// a poisoned handle (see [`Poison`]). The caller holds the write mutex
    /// and the `flock`.
    fn publish_or_settle(
        &self,
        file: &dyn ContainerIo,
        index: usize,
        base: &DiskCache,
        frames: &[Vec<u8>],
    ) -> Result<Option<Superblock>, StoreError> {
        /// Write every frame at the current WAL end, one `write_at` each, make them
        /// durable with one `sync_data`, then write the other slot and `sync_data`
        /// it (spec §4.3). Returns the superblock now published. No frames: nothing
        /// is written and `None`. A failure before the slot write leaves the previous
        /// superblock current; from the slot write on, either may be current.
        /// The caller holds the `flock` for the whole call. A poisoned handle
        /// writes nothing; a failed sync poisons it (see [`Poison`]). Nested
        /// here so that no publish can skip the settle around it.
        fn publish_frames(
            file: &dyn ContainerIo,
            poison: &Poison,
            index: usize,
            sb: &Superblock,
            frames: &[Vec<u8>],
        ) -> Result<Option<Superblock>, StoreError> {
            if frames.is_empty() {
                return Ok(None);
            }
            poison.check()?;
            let mut at = sb
                .wal_offset
                .checked_add(sb.wal_length)
                .ok_or_else(|| backend("wal offset overflow"))?;
            let mut wal_length = sb.wal_length;
            let mut writes = Vec::with_capacity(frames.len());
            for frame in frames {
                let len =
                    u64::try_from(frame.len()).map_err(|_| backend("wal frame exceeds u64"))?;
                writes.push((frame, at));
                at = at
                    .checked_add(len)
                    .ok_or_else(|| backend("wal offset overflow"))?;
                wal_length = wal_length
                    .checked_add(len)
                    .ok_or_else(|| backend("wal length overflow"))?;
            }
            let epoch = sb
                .epoch
                .checked_add(1)
                .ok_or_else(|| backend("epoch overflow"))?;
            for (frame, offset) in writes {
                file.write_at(frame, offset).map_err(io_err)?;
            }
            // `fdatasync` also makes the longer file size durable.
            poison.sync_data(file)?;
            let new_sb = Superblock {
                epoch,
                snapshot_offset: sb.snapshot_offset,
                snapshot_length: sb.snapshot_length,
                wal_offset: sb.wal_offset,
                wal_length,
                snapshot_seq: sb.snapshot_seq,
            };
            write_slot(file, index ^ 1, &new_sb)?;
            poison.sync_data(file)?;
            Ok(Some(new_sb))
        }

        publish_frames(file, &self.poison, index, &base.sb, frames)
            .inspect_err(|_| self.settle_failed_publish(base))
    }

    /// Install the state a publish through `write` just made as `new_sb`:
    /// `base` (what the writer refreshed to, or its last publish) with
    /// `change`. Nothing is read back. Called with the write mutex and the
    /// flock still held, so the next writer in this process starts from it.
    /// Returns the entry, which a group goes on from (spec §4.3).
    ///
    /// When the cache still holds `base`, the entry is taken out under the
    /// cache mutex and changed in place: with no reader holding its state,
    /// an overlay applies in O(ops) (spec §4.2). Readers wait on the mutex
    /// for that apply, never for I/O, and never see an empty cache.
    fn install_commit(
        &self,
        base: Arc<DiskCache>,
        new_sb: Superblock,
        change: Change,
    ) -> Arc<DiskCache> {
        // The base WAL ended torn (published bytes no full frame covers): the
        // new frames follow those bytes, so the cache is left for the next
        // refresh, which loads the file and reports it.
        let torn = base.wal_applied != base.sb.wal_length;
        let wal_applied = if torn {
            base.wal_applied
        } else {
            new_sb.wal_length
        };
        // The file the publish went to: `base` was read or written through
        // the same locked description.
        let file_id = base.file_id;
        let mut slot = lock_mutex(&self.cache);
        if !torn && slot.as_ref().is_some_and(|held| Arc::ptr_eq(held, &base)) {
            *slot = None;
        }
        let state = match change {
            Change::Overlay(overlay, seq) => {
                let mut state = match Arc::try_unwrap(base) {
                    Ok(entry) => entry.state,
                    Err(shared) => Arc::clone(&shared.state),
                };
                publish_overlay(&mut state, overlay, seq);
                state
            }
            Change::State(state) => state,
        };
        let entry = Arc::new(DiskCache {
            file_id,
            sb: new_sb,
            state,
            wal_applied,
        });
        if !torn {
            // The writer checked under the `flock` that the path names it.
            install_newer(&mut slot, Arc::clone(&entry), file_id);
        }
        entry
    }

    /// The WAL length an auto-checkpoint never fires at or below
    /// ([`AUTO_CHECKPOINT_BYTES`]; a test may lower it).
    fn auto_checkpoint_floor(&self) -> u64 {
        #[cfg(test)]
        let floor = self.auto_checkpoint_floor.load(Ordering::Relaxed);
        #[cfg(not(test))]
        let floor = AUTO_CHECKPOINT_BYTES;
        floor
    }

    /// [`AUTO_COMPACT_BYTES`]: the dead bytes an auto-compaction never fires
    /// at or below (a test may lower it).
    fn auto_compact_floor(&self) -> u64 {
        #[cfg(test)]
        let floor = self.auto_compact_floor.load(Ordering::Relaxed);
        #[cfg(not(test))]
        let floor = AUTO_COMPACT_BYTES;
        floor
    }

    /// Install a checkpoint's result: the same state, a new superblock.
    fn install_checkpoint(&self, base: &DiskCache, new_sb: Superblock) {
        let entry = DiskCache {
            file_id: base.file_id,
            sb: new_sb,
            state: Arc::clone(&base.state),
            wal_applied: 0,
        };
        let current = entry.file_id;
        install_newer(&mut lock_mutex(&self.cache), Arc::new(entry), current);
    }
}

/// Put `fresh` in the cache when it is newer than what the cache holds.
/// `current` is `(dev, ino)` of the file the path names, as this handle
/// last found it (its read description, which every swap check moves).
///
/// - The same file: the later epoch wins, so a slow refresh never replaces
///   a newer install.
/// - Another file: the file the path names wins, whatever the epochs. A
///   state of the current file replaces a state of any other file, so a
///   file put in place at a *lower* epoch (a restore by rename, a file
///   replaced by hand) is read as soon as a refresh finds it. A state of a
///   file the path no longer names (a slow read through a description
///   cloned before the swap) never replaces a state of the current one.
///   When neither is the current file, the later epoch wins: along a
///   compacted path's history the epoch only rises (spec §5).
fn install_newer(slot: &mut Option<Arc<DiskCache>>, fresh: Arc<DiskCache>, current: FileId) {
    let replace = match slot.as_ref() {
        None => true,
        Some(held) if held.file_id == fresh.file_id => fresh.sb.epoch > held.sb.epoch,
        Some(_) if fresh.file_id == current => true,
        Some(held) if held.file_id == current => false,
        Some(held) => fresh.sb.epoch > held.sb.epoch,
    };
    if replace {
        *slot = Some(fresh);
    }
}

/// `sb` is `cached.sb` with only frames appended: the same snapshot, the same
/// WAL start, a longer WAL, a later epoch, and the cache applied all of its
/// WAL. A checkpoint (a new snapshot extent), a shorter WAL or anything else
/// is not a tail.
fn tail_follows(cached: &DiskCache, sb: &Superblock) -> bool {
    let held = &cached.sb;
    cached.wal_applied == held.wal_length
        && sb.epoch > held.epoch
        && sb.snapshot_offset == held.snapshot_offset
        && sb.snapshot_length == held.snapshot_length
        && sb.snapshot_seq == held.snapshot_seq
        && sb.wal_offset == held.wal_offset
        && sb.wal_length > held.wal_length
}

/// Read `[wal_offset + wal_applied, wal_offset + wal_length)` of `sb`, decode
/// those frames and apply them, copy-on-write, to the cached state. `None`
/// when a frame is not the next sequence number: the full load decides.
fn replay_tail(
    file: &dyn ContainerIo,
    cached: &DiskCache,
    sb: Superblock,
) -> Result<Option<DiskCache>, StoreError> {
    let start = sb
        .wal_offset
        .checked_add(cached.wal_applied)
        .ok_or_else(|| corrupt("wal offset overflow"))?;
    let bytes = read_range(file, start, sb.wal_length - cached.wal_applied)?;
    let (frames, consumed) = decode_wal_prefix(&bytes)?;
    let mut state = Arc::clone(&cached.state);
    for (seq, ops) in frames {
        if state.last_seq.checked_add(1) != Some(seq) {
            return Ok(None);
        }
        let (overlay, next) = stage_ops(&state, &ops)?;
        publish_overlay(&mut state, overlay, next);
    }
    Ok(Some(DiskCache {
        file_id: cached.file_id,
        sb,
        state,
        wal_applied: cached.wal_applied + consumed,
    }))
}

/// Decode the snapshot and the whole WAL that `sb` names.
fn load_full(
    file: &dyn ContainerIo,
    file_id: (u64, u64),
    sb: Superblock,
) -> Result<DiskCache, StoreError> {
    let (state, wal_applied) = load_state(file, &sb)?;
    Ok(DiskCache {
        file_id,
        sb,
        state: Arc::new(state),
        wal_applied,
    })
}

/// One transaction: an overlay of pending changes on a shared base state,
/// plus the ops that produced them. The base is never changed.
///
/// `get` / `put` / `delete` / `scan` all see earlier writes in this
/// transaction. Every write is staged through [`Overlay::stage`], which checks
/// it as applying it would, so a commit applies the overlay as it is.
struct Txn {
    base: Base,
    overlay: Overlay,
    ops: Vec<Op>,
    /// A staged op failed part way. The body saw that error; a body that
    /// carries on regardless still cannot commit.
    failed: bool,
}

/// What a transaction's overlay sits on.
enum Base {
    /// A state in this process (`mem://`, `file://`, and the server's side
    /// of a remote commit).
    Local(Arc<DocState>),
    /// A read session on a server (spec §6.1). Boxed: it is much larger
    /// than an `Arc`, and a local transaction should not carry its size.
    Remote(Box<remote_txn::RemoteBase>),
}

impl Txn {
    fn new(base: Arc<DocState>) -> Self {
        Self::over(Base::Local(base))
    }

    /// A transaction whose reads go to `shared`'s server.
    fn remote(shared: Arc<server::RemoteShared>) -> Self {
        Self::over(Base::Remote(Box::new(remote_txn::RemoteBase::new(shared))))
    }

    fn over(base: Base) -> Self {
        Self {
            base,
            overlay: Overlay::default(),
            ops: Vec::new(),
            failed: false,
        }
    }

    /// The state a local transaction started from, without its own changes.
    /// A remote transaction holds none.
    fn local_base(&self) -> Result<&DocState, StoreError> {
        match &self.base {
            Base::Local(state) => Ok(state),
            Base::Remote(_) => Err(backend(
                "a remote transaction holds no state: it reads through its session",
            )),
        }
    }

    /// [`Self::local_base`], shared.
    fn local_arc(&self) -> Result<Arc<DocState>, StoreError> {
        match &self.base {
            Base::Local(state) => Ok(Arc::clone(state)),
            Base::Remote(_) => Err(backend(
                "a remote transaction holds no state: it reads through its session",
            )),
        }
    }

    fn remote_base(&self) -> Option<&remote_txn::RemoteBase> {
        match &self.base {
            Base::Remote(remote) => Some(remote),
            Base::Local(_) => None,
        }
    }

    /// The error a remote read recorded: the transaction's result, whatever
    /// the body returned.
    fn take_read_error(&self) -> Option<StoreError> {
        self.remote_base()
            .and_then(remote_txn::RemoteBase::take_error)
    }

    fn get(&self, collection: &str, id: &str) -> Option<Value> {
        match &self.base {
            Base::Local(state) => self.overlay.doc(state, collection, id).cloned(),
            Base::Remote(remote) => {
                if let Some(staged) = self.overlay.staged_doc(collection, id) {
                    return staged.cloned();
                }
                remote
                    .read(session::ReadKey::Get {
                        collection: collection.to_string(),
                        id: id.to_string(),
                    })?
                    .get("doc")
                    .cloned()
            }
        }
    }

    fn put(&mut self, collection: impl Into<String>, id: impl Into<String>, doc: Value) {
        let collection = collection.into();
        let id = id.into();
        self.overlay
            .put(collection.clone(), id.clone(), Arc::new(doc.clone()));
        self.ops.push(Op::Put {
            collection,
            id,
            doc,
        });
    }

    fn delete(&mut self, collection: impl Into<String>, id: impl Into<String>) {
        let collection = collection.into();
        let id = id.into();
        self.overlay.delete(collection.clone(), id.clone());
        self.ops.push(Op::Delete { collection, id });
    }

    /// Every document in `collection`, borrowed, in document-id order.
    fn visit(&self, collection: &str, f: impl FnMut(&str, &Value)) {
        match &self.base {
            Base::Local(state) => self.overlay.visit(state, collection, f),
            Base::Remote(remote) => {
                let Some(held) = remote.read(session::ReadKey::Scan {
                    collection: collection.to_string(),
                }) else {
                    return;
                };
                // The form was checked when the answer came in.
                let pairs = held.as_array().into_iter().flatten().filter_map(|pair| {
                    let id = pair.get(0)?.as_str()?;
                    Some((id, pair.get(1)?))
                });
                self.overlay.visit_over(collection, pairs, f);
            }
        }
    }

    /// Ids in `collection` whose top-level `field` matches `value`, in
    /// ascending order, with this transaction's writes.
    fn query_by_field(&self, collection: &str, field: &str, value: &str) -> Vec<String> {
        let Base::Remote(remote) = &self.base else {
            let mut ids = Vec::new();
            self.visit(collection, |id, doc| {
                if state::field_matches(doc, field, value) {
                    ids.push(id.to_string());
                }
            });
            return ids;
        };
        let Some(held) = remote.read(session::ReadKey::Query {
            collection: collection.to_string(),
            field: field.to_string(),
            value: value.to_string(),
        }) else {
            return Vec::new();
        };
        let staged: BTreeMap<&str, Option<&Value>> = self.overlay.staged_in(collection).collect();
        let mut ids: Vec<String> = held
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .filter(|id| !staged.contains_key(id))
            .map(str::to_string)
            .collect();
        ids.extend(staged.iter().filter_map(|(id, doc)| {
            doc.filter(|doc| state::field_matches(doc, field, value))
                .map(|_| (*id).to_string())
        }));
        ids.sort();
        ids
    }

    /// The names of the base's document collections that hold a document,
    /// without this transaction's writes.
    fn collections(&self) -> Vec<String> {
        match &self.base {
            Base::Local(state) => state.docs.collections().cloned().collect(),
            Base::Remote(remote) => remote
                .read(session::ReadKey::Inventory)
                .map(|held| remote_txn::inventory_names(&held, "collections"))
                .unwrap_or_default(),
        }
    }

    /// Every document in `collection`, in document-id order.
    fn scan(&self, collection: &str) -> Vec<(String, Value)> {
        let mut out = Vec::new();
        self.visit(collection, |id, doc| {
            out.push((id.to_string(), doc.clone()))
        });
        out
    }

    /// Stage a column-table or vector op (or a document op). `Err` stages
    /// nothing the commit could use: the transaction will not commit.
    fn stage(&mut self, op: Op) -> Result<(), StoreError> {
        let staged = match &self.base {
            Base::Local(state) => self.overlay.stage(state, &op),
            Base::Remote(_) => self.stage_remote(&op),
        };
        if let Err(err) = staged {
            self.failed = true;
            return Err(err);
        }
        self.ops.push(op);
        Ok(())
    }

    /// Check `op` as [`Overlay::stage`] would, against the session's reads,
    /// and record it. A vector op is not applied here: the server applies
    /// it, HNSW and all, when it commits (spec §6.1).
    fn stage_remote(&mut self, op: &Op) -> Result<(), StoreError> {
        match op {
            Op::Put {
                collection,
                id,
                doc,
            } => self
                .overlay
                .put(collection.clone(), id.clone(), Arc::new(doc.clone())),
            Op::Delete { collection, id } => self.overlay.delete(collection.clone(), id.clone()),
            Op::CreateColumnTable { name, columns } => {
                let taken = self.table_columns(name).is_some();
                columnar::check_new_table(taken, name, columns)?;
                self.overlay.record_table_create(name, columns);
            }
            Op::AppendColumnGroup { name, bytes } => {
                let Some(columns) = self.table_columns(name) else {
                    return Err(layout::corrupt(format!(
                        "append to missing column table `{name}`"
                    )));
                };
                columnar::validate_row_group(&columns, bytes)?;
                self.overlay.record_table_append(name, bytes);
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
                if self.vector_params(name).is_some() {
                    return Err(backend(format!(
                        "vector collection `{name}` already exists"
                    )));
                }
                let params = vector::VectorParams {
                    dim: *dim,
                    storage: *storage,
                    metric: *metric,
                    m: *m,
                    ef_construction: *ef_construction,
                    ef_search: *ef_search,
                };
                let Base::Remote(remote) = &mut self.base else {
                    return Err(backend("a local transaction staged as a remote one"));
                };
                remote.created_vectors.insert(name.clone(), params);
                remote.written_vectors.insert(name.clone());
            }
            Op::UpsertVector { collection, .. } | Op::DeleteVector { collection, .. } => {
                if self.vector_params(collection).is_none() {
                    return Err(backend(format!("no vector collection `{collection}`")));
                }
                let Base::Remote(remote) = &mut self.base else {
                    return Err(backend("a local transaction staged as a remote one"));
                };
                remote.written_vectors.insert(collection.clone());
            }
        }
        Ok(())
    }

    /// The columns of column table `name`, with this transaction's creates.
    fn table_columns(&self, name: &str) -> Option<Vec<ColumnDef>> {
        match &self.base {
            Base::Local(state) => self.overlay.table_columns(state, name).map(<[_]>::to_vec),
            Base::Remote(remote) => {
                if let Some(created) = self.overlay.created_columns(name) {
                    return Some(created.to_vec());
                }
                let held = remote.read(session::ReadKey::Columns {
                    name: name.to_string(),
                })?;
                if held.is_null() {
                    return None;
                }
                match columnar::parse_columns(&held) {
                    Ok(columns) => Some(columns),
                    Err(err) => {
                        remote.fail(err);
                        None
                    }
                }
            }
        }
    }

    /// The parameters of vector collection `name`, with this transaction's
    /// creates. A read of its parameters, not of its vectors: a remote
    /// transaction may make it after writing the collection.
    fn vector_params(&self, name: &str) -> Option<vector::VectorParams> {
        match &self.base {
            Base::Local(state) => self
                .overlay
                .vector(state, name)
                .map(vector::VectorParams::of),
            Base::Remote(remote) => {
                if let Some(created) = remote.created_vectors.get(name) {
                    return Some(*created);
                }
                let held = remote.read(session::ReadKey::VectorParams {
                    name: name.to_string(),
                })?;
                if held.is_null() {
                    return None;
                }
                match vector::VectorParams::parse(&held) {
                    Ok(params) => Some(params),
                    Err(err) => {
                        remote.fail(err);
                        None
                    }
                }
            }
        }
    }

    /// Vector collection `name` of a local transaction, with its changes.
    fn local_vector(&self, name: &str) -> Result<Option<&vector::VectorCollection>, StoreError> {
        Ok(self.overlay.vector(self.local_base()?, name))
    }

    fn has_tables(&self) -> bool {
        match &self.base {
            Base::Local(state) => self.overlay.has_tables(state),
            Base::Remote(remote) => {
                self.overlay.creates_table()
                    || remote
                        .read(session::ReadKey::Inventory)
                        .is_some_and(|held| {
                            !remote_txn::inventory_names(&held, "tables").is_empty()
                        })
            }
        }
    }

    fn has_vectors(&self) -> bool {
        match &self.base {
            Base::Local(state) => self.overlay.has_vectors(state),
            Base::Remote(remote) => {
                !remote.created_vectors.is_empty()
                    || remote
                        .read(session::ReadKey::Inventory)
                        .is_some_and(|held| {
                            !remote_txn::inventory_names(&held, "vectors").is_empty()
                        })
            }
        }
    }

    /// The ops and the overlay to commit. Drops this transaction's hold on
    /// the base, so a base nobody else holds is changed in place.
    fn finish(self) -> Result<(Vec<Op>, Overlay), StoreError> {
        if self.failed {
            return Err(backend(
                "a transaction whose staged op failed cannot commit",
            ));
        }
        Ok((self.ops, self.overlay))
    }
}

/// Document store and FerroMQ log.
///
/// `mem://` keeps the map in this process. `file://` is a `.fls` container:
/// two superblock slots, a copy-on-write snapshot, and a CRC32-framed WAL.
/// `unix://` and `ws://` are clients of a [`FerroStoreServer`] and do not open the file.
/// [`Self::auto`] is that client when the default socket is listening, and [`Self::open`] otherwise.
pub struct FerroStore {
    inner: Inner,
    subs: Mutex<HashMap<String, Vec<Sender<MqRecord>>>>,
    /// Named in `NotOwned`: where a lookup looked.
    label: String,
    /// Process memory only. Not part of the `.fls` file.
    sync: Mutex<SyncSlot>,
    /// Taken before `last_seq` advances. Held until that batch is registered.
    /// A group commit's leader holds it for the whole group and registers its
    /// members one by one, in sequence order (spec §4.3).
    commit_gate: Mutex<()>,
    /// Paired with `sync`. Wakes `sync_flush` when `notify_in_flight` clears.
    sync_cv: Condvar,
    /// The threads running a write body on this handle right now: a call
    /// from one of them is refused ([`Self::refuse_reentry`]). More than
    /// one only on a remote handle, whose bodies take no lock.
    body_threads: Mutex<Vec<ThreadId>>,
    /// `body_threads.len()`, read without the mutex: a thread only ever
    /// needs to see its own entries, which its own writes make visible.
    bodies: AtomicUsize,
}

/// Marks the current thread as running a write body on one handle until
/// dropped: on return, on `Err` and on unwind alike.
struct BodyGuard<'a> {
    store: &'a FerroStore,
}

impl Drop for BodyGuard<'_> {
    fn drop(&mut self) {
        let me = std::thread::current().id();
        let mut threads = lock_mutex(&self.store.body_threads);
        if let Some(at) = threads.iter().position(|thread| *thread == me) {
            threads.swap_remove(at);
            self.store.bodies.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl FerroStore {
    fn assemble(inner: Inner, label: String) -> Self {
        Self {
            inner,
            subs: Mutex::new(HashMap::new()),
            label,
            sync: Mutex::new(SyncSlot::default()),
            commit_gate: Mutex::new(()),
            sync_cv: Condvar::new(),
            body_threads: Mutex::new(Vec::new()),
            bodies: AtomicUsize::new(0),
        }
    }

    /// The current thread runs a write body on this handle until the guard
    /// is dropped.
    fn enter_body(&self) -> BodyGuard<'_> {
        lock_mutex(&self.body_threads).push(std::thread::current().id());
        self.bodies.fetch_add(1, Ordering::Relaxed);
        BodyGuard { store: self }
    }

    /// `Err` when the current thread is running a write body on this
    /// handle. Such a call would wait for the lock its own body holds
    /// (`mem://`: any call; `file://`: a write), read the committed state
    /// without the body's own writes (`file://` and remote reads), or commit
    /// apart from the body (a remote write). Every public entry point checks
    /// this before it takes a lock. Another thread is never refused.
    fn refuse_reentry(&self) -> Result<(), StoreError> {
        if self.bodies.load(Ordering::Relaxed) == 0 {
            return Ok(());
        }
        let me = std::thread::current().id();
        if lock_mutex(&self.body_threads).contains(&me) {
            return Err(backend(
                "FerroStore called from inside its own transaction body",
            ));
        }
        Ok(())
    }

    /// Empty in-memory store at the newest schema version. `checkpoint`
    /// succeeds and writes nothing.
    pub fn in_memory() -> Self {
        let state = DocState {
            docs: new_store_docs(),
            ..DocState::default()
        };
        Self::assemble(
            Inner::Memory(Mutex::new(Arc::new(state))),
            "mem://".to_string(),
        )
    }

    /// Open or create a `.fls` file. A second open of the same path succeeds.
    ///
    /// A new file is created mode 0600 from its first byte, in a parent made
    /// 0700 when the parent is missing too, and starts at the newest schema
    /// version; its directory is synced once the first superblock is down. An
    /// existing file keeps its mode and is only reported when it is loose
    /// (`private::warn_if_loose`). A container a newer fl wrote is
    /// [`StoreError::ContainerFormat`]; a schema version off the ladder is
    /// [`StoreError::FormatVersion`]. A file from before schema versioning
    /// (no version document) opens, and [`Self::format_version`] reads `None`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        Self::open_with(&RealFs, path.as_ref())
    }

    /// [`Self::open`] through `fs`. Production passes [`RealFs`]; the crash
    /// harness passes its simulated disk.
    ///
    /// `path` is resolved once, after the file exists ([`ContainerFs::resolve`]):
    /// every later operation on a name (the swap check, a compaction's
    /// sibling, its rename and directory sync, the sweep of stale temps) is
    /// made on the file's real path, in its own directory, so a store opened
    /// through a symlink compacts in place and the link goes on naming it.
    /// The handle's label stays `path` as given.
    fn open_with(fs: &dyn ContainerFs, given: &Path) -> Result<Self, StoreError> {
        let created = fs.create_store_file(given)?;
        let resolved = fs.resolve(given)?;
        let path = resolved.as_path();
        let write = fs
            .open_rw(path)
            .map_err(|err| backend(format!("open {}: {err}", path.display())))?;
        let initialized = ensure_superblock(&*write).map_err(|err| name_store(err, given))?;
        let (_, sb) = latest_superblock(&*write).map_err(|err| name_store(err, given))?;
        let mut write = Descriptor::new(write, path)?;
        let seed = load_full(&*write.file, write.id, sb).map_err(|err| name_store(err, given))?;
        let state = &seed.state;
        // Also when an earlier open created the file and stopped before its
        // first superblock: the entry is only durable once the directory is.
        // Any other open did not make the name, so its first write syncs the
        // directory ([`Disk::lock_current`]).
        if created || initialized {
            fs.sync_dir(&parent_dir(path))?;
            write.name_durable = true;
        }
        if let Some(found) =
            doc::schema_version_in(|collection, id| state.docs.get(collection, id).cloned())?
        {
            crate::check_format(Some(found))?;
        }
        fs.warn_if_loose(path);
        let read = fs
            .open_ro(path)
            .map_err(|err| backend(format!("open {}: {err}", path.display())))?;
        let read = Descriptor::new(read, path)?;
        Ok(Self::assemble(
            Inner::Disk(Box::new(Disk {
                fs: fs.shared(),
                path: resolved.clone(),
                write: Mutex::new(write),
                read: Mutex::new(read),
                cache: Mutex::new(Some(Arc::new(seed))),
                group: group::Group::default(),
                poison: Poison::new(given),
                checkpoint_failed_at: AtomicU64::new(0),
                compact_failed_at: AtomicU64::new(0),
                #[cfg(test)]
                auto_checkpoint_floor: std::sync::atomic::AtomicU64::new(AUTO_CHECKPOINT_BYTES),
                #[cfg(test)]
                auto_compact_floor: std::sync::atomic::AtomicU64::new(AUTO_COMPACT_BYTES),
            })),
            given.display().to_string(),
        ))
    }

    /// If `default_socket_path_for_store(path)` accepts a Unix connection, return a client of that server.
    /// Otherwise `open(path)` in embedded mode.
    ///
    /// The successful `UnixStream` is the client. It is not dropped in favor of a second connect.
    /// A refused connect (nothing listening, or no socket file) is the embedded fallback, not an error.
    /// `open` errors are still returned.
    pub fn auto(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let path = path.as_ref();
        let sock = default_socket_path_for_store(path);
        match UnixStream::connect(&sock) {
            Ok(stream) => {
                let remote = server::client_from_unix(stream, &auth::ClientCred::None)?;
                Ok(Self::assemble(
                    Inner::Remote(remote),
                    sock.display().to_string(),
                ))
            }
            // Nothing listening, or the socket file is missing. Other connect
            // errors stay errors. `open` errors are not swallowed.
            Err(err) if refused_connect(&err) => Self::open(path),
            Err(err) => Err(backend(format!("connect {}: {err}", sock.display()))),
        }
    }

    /// `mem://`, `file://`, `unix://`, `ws://` and `wss://`.
    ///
    /// A `wss://` client checks the server's certificate against the bundled
    /// Mozilla root certificates (`webpki-roots`, compiled into the binary;
    /// not the OS trust store). See [`ConnectOptions::ca_file`] (an
    /// enterprise or private CA) and
    /// [`ConnectOptions::pin_spki_sha256`] for the other ways, which only
    /// options set, never a URI.
    ///
    /// A `ws://` or `wss://` URI may name its credential in its query (spec §6.2), only
    /// by a `FERROSTORE_*` environment variable:
    /// `?psk_env=FERROSTORE_PSK&server_id=<name>` or
    /// `?token_env=FERROSTORE_TOKEN`. A file, another variable or a secret
    /// itself is never accepted in a URI; name a file in [`ConnectOptions`].
    /// A `unix://` URI takes none: its server checks the peer's uid; use
    /// [`Self::connect_with`] for a PSK there.
    pub fn connect_uri(uri: &str) -> Result<Self, StoreError> {
        Self::connect_with(uri, &ConnectOptions::new())
    }

    /// [`Self::connect_uri`], authenticating with `options` (spec §6.2).
    /// `options` apply to `unix://` and `ws://` (and `wss://`) alone, and a
    /// `ws://` or `wss://` URI that also names a credential in its query is
    /// refused. Over plain `ws://` a credential goes only to a loopback
    /// address, unless `options` set [`ConnectOptions::allow_plaintext`];
    /// over `wss://` it goes to any host, inside TLS (spec §6.3). The TLS
    /// options ([`ConnectOptions::ca_file`],
    /// [`ConnectOptions::pin_spki_sha256`]) apply to `wss://` alone. The
    /// handle is returned only after the server accepted the connection
    /// and, for a PSK, proved that it holds the same PSK under the expected
    /// `server_id`.
    pub fn connect_with(uri: &str, options: &ConnectOptions) -> Result<Self, StoreError> {
        let local_only = || {
            if options.has_auth() {
                Err(backend(
                    "authentication options apply to unix:// and ws:// (or wss://) only",
                ))
            } else {
                Ok(())
            }
        };
        let tls_options = options.has_tls_options();
        if tls_options && !uri.starts_with("wss://") {
            return Err(backend(
                "the TLS options (ca_file, pin_spki_sha256) apply to wss:// only",
            ));
        }
        if uri == "mem://" || uri == "mem:///" {
            local_only()?;
            return Ok(Self::in_memory());
        }
        if let Some(rest) = uri.strip_prefix("file://") {
            local_only()?;
            return Self::open(file_uri_path(rest));
        }
        if let Some(rest) = uri.strip_prefix("unix://") {
            let path = file_uri_path(rest);
            let remote = server::connect_unix_with(&path, &options.resolve()?)?;
            return Ok(Self::assemble(Inner::Remote(remote), path));
        }
        let tcp = match (uri.strip_prefix("wss://"), uri.strip_prefix("ws://")) {
            (Some(rest), _) => Some((rest, Some(options.verify()?))),
            (None, Some(rest)) => Some((rest, None)),
            (None, None) => None,
        };
        if let Some((rest, verify)) = tcp {
            let (target, from_uri) = auth::split_uri_query(rest)?;
            let outer_plaintext = options.plaintext_allowed();
            let options = match (options.has_auth(), from_uri.has_auth()) {
                (true, true) => {
                    return Err(backend(
                        "the credential is given twice: in the ws:// URI and in the options",
                    ));
                }
                (false, true) => &from_uri,
                _ => options,
            };
            let remote = server::connect_tcp(
                &target,
                &options.resolve()?,
                // Only the caller's options allow plaintext, never a URI.
                outer_plaintext,
                // Only the caller's options name the TLS check, never a URI.
                verify.as_ref(),
            )?;
            return Ok(Self::assemble(Inner::Remote(remote), uri.to_string()));
        }
        Err(backend(format!("unknown FerroStore URI scheme in `{uri}`")))
    }

    /// The schema version the store records, `None` for a store with no
    /// version document (a FerroStore written before schema versioning), the
    /// same meaning as [`crate::RedbStore::format_version`]. The value is on
    /// the shared ladder `FORMAT_VERSION` … `FORMAT_WITH_WORKFLOW`.
    pub fn format_version(&self) -> Result<Option<u64>, StoreError> {
        self.with_read(doc::schema_version)
    }

    /// Lower (or raise) this disk handle's auto-checkpoint floor
    /// ([`AUTO_CHECKPOINT_BYTES`]), so a small test workload crosses it.
    /// `u64::MAX` turns auto-checkpoints off.
    #[cfg(test)]
    pub(super) fn set_auto_checkpoint_floor(&self, bytes: u64) {
        let Inner::Disk(disk) = &self.inner else {
            panic!("auto-checkpoint is a disk handle's");
        };
        disk.auto_checkpoint_floor.store(bytes, Ordering::Relaxed);
    }

    /// Lower (or raise) this disk handle's auto-compaction floor
    /// ([`AUTO_COMPACT_BYTES`]). `u64::MAX` turns auto-compaction off.
    #[cfg(test)]
    pub(super) fn set_auto_compact_floor(&self, bytes: u64) {
        let Inner::Disk(disk) = &self.inner else {
            panic!("auto-compaction is a disk handle's");
        };
        disk.auto_compact_floor.store(bytes, Ordering::Relaxed);
    }

    /// Write a snapshot of the current state and start an empty WAL (spec
    /// §4.4), so a later open replays nothing. A commit also does this on
    /// its own once the WAL is longer than `max(snapshot_length, 4 MiB)`.
    /// A checkpoint that leaves the file more than half dead and the dead
    /// bytes over [`AUTO_COMPACT_BYTES`] is followed by a compaction
    /// ([`Self::compact`]), in a lock window of its own. `mem://` writes
    /// nothing.
    pub fn checkpoint(&self) -> Result<(), StoreError> {
        self.refuse_reentry()?;
        match &self.inner {
            Inner::Remote(remote) => remote.call_unit("checkpoint", json!({})),
            Inner::Memory(_) => Ok(()),
            Inner::Disk(disk) => checkpoint_disk(disk),
        }
    }

    /// Rewrite the store as a new file holding one snapshot of the current
    /// state and an empty WAL, and rename it over the old one (spec §5):
    /// every extent an earlier checkpoint left dead is gone. Other handles,
    /// in this process or another, move to the new file at their next
    /// commit or read; a read already reading the old file finishes on it.
    /// The new file's epoch is the old file's current epoch + 1, and
    /// `last_seq` is kept. A failure before the rename leaves the old file
    /// as it was and removes the new one; a failed `fsync` poisons the
    /// handle (see [`Self::checkpoint`]). `mem://` is a no-op; a remote
    /// handle asks its server.
    pub fn compact(&self) -> Result<(), StoreError> {
        self.refuse_reentry()?;
        match &self.inner {
            Inner::Remote(remote) => remote.call_unit("compact", json!({})),
            Inner::Memory(_) => Ok(()),
            Inner::Disk(disk) => compact::compact_disk(disk),
        }
    }

    /// The size accounting of the store file at `path` (see [`StoreStats`]),
    /// read without opening a store: the file is opened read-only, takes no
    /// lock, and nothing is created, initialised, repaired or written, so it is
    /// safe beside a live store and works on a 0400 file. The figures are of
    /// one open file, so `file_bytes`, `live_bytes` and `dead_bytes` agree
    /// with each other even while a writer publishes.
    ///
    /// A file that holds no valid superblock (missing, shorter than the
    /// header, zeroed, or not a FerroStore) is an error; [`Self::open`] would
    /// initialise some of those, and this never does.
    pub fn stats_at(path: impl AsRef<Path>) -> Result<StoreStats, StoreError> {
        Self::stats_at_with(&RealFs, path.as_ref())
    }

    fn stats_at_with(fs: &dyn ContainerFs, given: &Path) -> Result<StoreStats, StoreError> {
        let file = fs
            .open_ro(given)
            .map_err(|err| backend(format!("open {}: {err}", given.display())))?;
        let id = file.file_id().map_err(io_err)?;
        let (_, sb) = latest_superblock(&*file).map_err(|err| name_store(err, given))?;
        let cache = load_full(&*file, id, sb).map_err(|err| name_store(err, given))?;
        let file_bytes = file.len().map_err(io_err)?;
        Ok(compact::store_stats(file_bytes, &cache))
    }

    /// The container's size accounting (see [`StoreStats`]), read without a
    /// lock, as a read is. `mem://` has no file: every byte count is 0 and
    /// the epoch is 0. A remote handle does not serve it.
    pub fn stats(&self) -> Result<StoreStats, StoreError> {
        self.refuse_reentry()?;
        match &self.inner {
            Inner::Remote(_) => Err(backend(
                "a remote FerroStore does not serve stats: read them on the server",
            )),
            Inner::Memory(state) => Ok(StoreStats {
                file_bytes: 0,
                live_bytes: 0,
                dead_bytes: 0,
                epoch: 0,
                last_seq: lock_mutex(state).last_seq,
            }),
            Inner::Disk(disk) => compact::stats(disk),
        }
    }

    /// Put one document. A reserved collection (a name starting `__`, such as
    /// the schema version's `__meta`) is refused: only the store's own
    /// transactions write it.
    pub fn put_doc(&self, collection: &str, id: &str, doc: &Value) -> Result<(), StoreError> {
        self.put_doc_inner(collection, id, doc, false)
    }

    fn put_doc_inner(
        &self,
        collection: &str,
        id: &str,
        doc: &Value,
        silent: bool,
    ) -> Result<(), StoreError> {
        doc::refuse_reserved(collection, "put_doc")?;
        self.put_doc_unchecked(collection, id, doc, silent)
    }

    /// `put_doc` without the reserved-collection check: for tests that must
    /// plant a store's own bookkeeping.
    fn put_doc_unchecked(
        &self,
        collection: &str,
        id: &str,
        doc: &Value,
        silent: bool,
    ) -> Result<(), StoreError> {
        let collection = collection.to_string();
        let id = id.to_string();
        let doc = doc.clone();
        self.commit(
            move || {
                Ok((
                    Op::Put {
                        collection,
                        id,
                        doc,
                    },
                    (),
                ))
            },
            silent,
        )
    }

    pub fn get_doc(&self, collection: &str, id: &str) -> Result<Option<Value>, StoreError> {
        self.refuse_reentry()?;
        let remote_miss = if let Inner::Remote(remote) = &self.inner {
            match remote.call_option("get_doc", json!({"collection": collection, "id": id}))? {
                Some(doc) => return Ok(Some(doc)),
                None => true,
            }
        } else {
            false
        };
        if remote_miss {
            return self.fill_doc_miss(collection, id);
        }
        let state = self.read_state()?;
        if let Some(doc) = state.docs.get(collection, id).cloned() {
            return Ok(Some(doc));
        }
        self.fill_doc_miss(collection, id)
    }

    /// Delete one document. A reserved collection is refused, as in
    /// [`Self::put_doc`].
    pub fn delete_doc(&self, collection: &str, id: &str) -> Result<(), StoreError> {
        doc::refuse_reserved(collection, "delete_doc")?;
        self.delete_doc_unchecked(collection, id)
    }

    /// `delete_doc` without the reserved-collection check, for tests.
    fn delete_doc_unchecked(&self, collection: &str, id: &str) -> Result<(), StoreError> {
        let collection = collection.to_string();
        let id = id.to_string();
        self.commit(move || Ok((Op::Delete { collection, id }, ())), false)
    }

    pub fn query_by_field(
        &self,
        collection: &str,
        field: &str,
        value: &str,
    ) -> Result<Vec<String>, StoreError> {
        self.refuse_reentry()?;
        if let Inner::Remote(remote) = &self.inner {
            return remote.call_strings(
                "query_by_field",
                json!({"collection": collection, "field": field, "value": value}),
            );
        }
        let state = self.read_state()?;
        Ok(state::query_by_field(&state, collection, field, value))
    }

    pub fn search_bm25(
        &self,
        collection: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<(String, f64)>, StoreError> {
        self.refuse_reentry()?;
        if let Inner::Remote(remote) = &self.inner {
            return remote.call_pairs(
                "search_bm25",
                json!({"collection": collection, "query": query, "limit": limit}),
            );
        }
        let state = self.read_state()?;
        Ok(state::search_bm25(&state, collection, query, limit))
    }

    /// Create a column table. An existing name is an error and writes nothing.
    pub fn create_column_table(&self, name: &str, columns: &[ColumnDef]) -> Result<(), StoreError> {
        self.with_write(|txn| txn.stage_create_column_table(name, columns))
    }

    /// Append one row group. An empty `rows` writes nothing. A bad row writes nothing.
    pub fn append_column_rows(
        &self,
        name: &str,
        rows: &[Vec<ColumnValue>],
    ) -> Result<(), StoreError> {
        self.with_write(|txn| txn.stage_append_column_rows(name, rows))
    }

    /// Scan `name`. A row group whose zone map excludes any filter is skipped
    /// before its values are decoded. Filters are AND.
    pub fn scan_column_table(
        &self,
        name: &str,
        filters: &[ColumnFilter],
    ) -> Result<ColumnScan, StoreError> {
        self.refuse_reentry()?;
        if let Inner::Remote(remote) = &self.inner {
            let mut encoded = Vec::with_capacity(filters.len());
            for filter in filters {
                encoded.push(columnar::filter_json(filter)?);
            }
            let value = remote.call(
                "scan_column_table",
                json!({"name": name, "filters": encoded}),
            )?;
            return columnar::parse_scan(&value);
        }
        let state = self.read_state()?;
        let Some(table) = state.tables.get(name) else {
            return Err(backend(format!("no column table `{name}`")));
        };
        columnar::scan_groups(&table.columns, &table.groups, filters)
    }

    /// Sum attempt ledger rows through an in-memory columnar scan. `None` is every attempt.
    pub fn attempt_rollup(&self, project: Option<&ProjectId>) -> Result<AttemptRollup, StoreError> {
        self.refuse_reentry()?;
        if let Inner::Remote(remote) = &self.inner {
            let project = match project {
                Some(id) => Value::String(id.as_str().to_string()),
                None => Value::Null,
            };
            let value = remote.call("attempt_rollup", json!({"project": project}))?;
            return columnar::parse_attempt_rollup(&value);
        }
        let project = project.cloned();
        self.with_read(move |txn| columnar::attempt_rollup(txn, project.as_ref()))
    }

    /// Sum gate-run ledger rows through an in-memory columnar scan. `None` is every run.
    pub fn gate_run_rollup(&self, gate: Option<&GateId>) -> Result<GateRunRollup, StoreError> {
        self.refuse_reentry()?;
        if let Inner::Remote(remote) = &self.inner {
            let gate = match gate {
                Some(id) => Value::String(id.as_str().to_string()),
                None => Value::Null,
            };
            let value = remote.call("gate_run_rollup", json!({"gate": gate}))?;
            return columnar::parse_gate_rollup(&value);
        }
        let gate = gate.cloned();
        self.with_read(move |txn| columnar::gate_run_rollup(txn, gate.as_ref()))
    }

    /// Create an empty vector collection. An existing name is an error and writes nothing.
    /// A document collection may use the same name; the two maps are distinct.
    #[allow(clippy::too_many_arguments)]
    pub fn create_vector_collection(
        &self,
        name: &str,
        dim: u32,
        storage: VectorStorage,
        metric: VectorMetric,
        m: u32,
        ef_construction: u32,
        ef_search: u32,
    ) -> Result<(), StoreError> {
        self.with_write(|txn| {
            txn.stage_create_vector_collection(
                name,
                dim,
                storage,
                metric,
                m,
                ef_construction,
                ef_search,
            )
        })
    }

    /// `true` when vector collection `name` exists. A read: it never creates
    /// the collection and commits nothing. A document collection of the same
    /// name does not count.
    pub fn has_vector_collection(&self, name: &str) -> Result<bool, StoreError> {
        self.refuse_reentry()?;
        if let Inner::Remote(remote) = &self.inner {
            return remote
                .call("has_vector_collection", json!({"name": name}))?
                .as_bool()
                .ok_or_else(|| backend("has_vector_collection result is not a bool"));
        }
        Ok(self.read_state()?.vectors.contains_key(name))
    }

    /// Insert or replace one vector. `None` metadata is absent. `Some(Null)` stores JSON null.
    pub fn upsert_vector(
        &self,
        collection: &str,
        id: &str,
        values: &[f32],
        metadata: Option<&Value>,
    ) -> Result<(), StoreError> {
        self.with_write(|txn| txn.stage_upsert_vectors(collection, &[(id, values, metadata)]))
    }

    /// Insert or replace every `(id, values, metadata)` item in one commit:
    /// one sequence, one WAL frame, and one copy of the collection for the
    /// whole batch. The result is the same as [`Self::upsert_vector`] for
    /// each item in order, so an id given twice ends as its last item.
    ///
    /// Every item is checked (id, length, finite components) before any is
    /// applied: one bad item, or a missing collection, writes nothing. `Ok`
    /// is `Some` of the sequence the batch committed at. An empty `items`
    /// commits nothing and is `Ok(None)` (a missing collection is still an
    /// error).
    pub fn upsert_vectors(
        &self,
        collection: &str,
        items: &[(&str, &[f32], Option<&Value>)],
    ) -> Result<Option<u64>, StoreError> {
        if items.is_empty() {
            if !self.has_vector_collection(collection)? {
                return Err(backend(format!("no vector collection `{collection}`")));
            }
            return Ok(None);
        }
        let ((), seq) =
            self.with_write_outcome(|txn| txn.stage_upsert_vectors(collection, items))?;
        Ok(seq)
    }

    /// Delete one vector. A missing id still commits. A missing collection writes nothing.
    pub fn delete_vector(&self, collection: &str, id: &str) -> Result<(), StoreError> {
        self.with_write(|txn| txn.stage_delete_vector(collection, id))
    }

    /// Dequantized values and metadata. `Ok(None)` when the id is missing.
    #[allow(clippy::type_complexity)]
    pub fn get_vector(
        &self,
        collection: &str,
        id: &str,
    ) -> Result<Option<(Vec<f32>, Option<Value>)>, StoreError> {
        self.refuse_reentry()?;
        if let Inner::Remote(remote) = &self.inner {
            return vector::parse_get(
                &remote.call("get_vector", json!({"collection": collection, "id": id}))?,
            );
        }
        self.with_read(|txn| txn.read_vector(collection, id))
    }

    /// Higher-better scores, at most `limit`. `ef` of `None` uses the collection `ef_search`.
    /// `limit == 0` or `ef == Some(0)` is an empty vec.
    pub fn search_vectors(
        &self,
        collection: &str,
        values: &[f32],
        limit: usize,
        ef: Option<u32>,
    ) -> Result<Vec<(String, f64)>, StoreError> {
        self.refuse_reentry()?;
        if let Some(adapter) = self.attached_adapter()
            && let Some(hits) = cloud_hits(adapter.search_vectors(collection, values, limit, ef))
        {
            return Ok(hits);
        }
        if let Inner::Remote(remote) = &self.inner {
            let mut params = serde_json::Map::new();
            params.insert("collection".into(), json!(collection));
            params.insert("values".into(), vector::values_json(values)?);
            params.insert("limit".into(), json!(limit));
            if let Some(ef) = ef {
                params.insert("ef".into(), json!(ef));
            }
            return vector::parse_hits(&remote.call("search_vectors", Value::Object(params))?);
        }
        let state = self.read_state()?;
        let Some(col) = state.vectors.get(collection) else {
            return Err(backend(format!("no vector collection `{collection}`")));
        };
        vector::search(col, values, limit, ef)
    }

    /// Fuse at most `limit` BM25 hits with at most `limit` vector hits, then keep `limit`.
    pub fn search_hybrid(
        &self,
        doc_collection: &str,
        vector_collection: &str,
        query: &str,
        values: &[f32],
        limit: usize,
        fusion: Fusion,
    ) -> Result<Vec<(String, f64)>, StoreError> {
        self.refuse_reentry()?;
        if let Some(adapter) = self.attached_adapter()
            && let Some(hits) = cloud_hits(adapter.search_hybrid(
                doc_collection,
                vector_collection,
                query,
                values,
                limit,
                fusion,
            ))
        {
            return Ok(hits);
        }
        if let Inner::Remote(remote) = &self.inner {
            return vector::parse_hits(&remote.call(
                "search_hybrid",
                hybrid_params(
                    doc_collection,
                    vector_collection,
                    query,
                    values,
                    limit,
                    fusion,
                )?,
            )?);
        }
        let state = self.read_state()?;
        state::search_hybrid(
            &state.docs,
            &state.vectors,
            doc_collection,
            vector_collection,
            query,
            values,
            limit,
            fusion,
        )
    }

    /// Same fusion as [`Self::search_hybrid`]. `doc` is the document at that id, if any.
    /// `metadata` is the vector metadata when the id is in the vector collection.
    pub fn search_knowledge(
        &self,
        doc_collection: &str,
        vector_collection: &str,
        query: &str,
        values: &[f32],
        limit: usize,
        fusion: Fusion,
    ) -> Result<Vec<KnowledgeHit>, StoreError> {
        self.refuse_reentry()?;
        if let Inner::Remote(remote) = &self.inner {
            return vector::parse_knowledge(&remote.call(
                "search_knowledge",
                hybrid_params(
                    doc_collection,
                    vector_collection,
                    query,
                    values,
                    limit,
                    fusion,
                )?,
            )?);
        }
        let state = self.read_state()?;
        state::search_knowledge(
            &state.docs,
            &state.vectors,
            doc_collection,
            vector_collection,
            query,
            values,
            limit,
            fusion,
        )
    }

    /// Append one message. The returned sequence starts at 1 per topic and is
    /// not reused after that message document is deleted.
    ///
    /// [`StoreError::Undelivered`] means the message is durable and its batch
    /// was not sent to the `WriteThrough` adapter: this handle's subscribers
    /// ([`Self::subscribe_mq`]) still receive it. The error carries the store
    /// sequence, not the message sequence. On a remote handle, when the
    /// server returns `Undelivered`, the message is durable on the server and
    /// its subscribers receive it, but this handle registers no batch with its
    /// own adapter: the commit's ops are not known here without the message
    /// sequence, which the error does not carry.
    pub fn publish_mq(&self, topic: &str, key: &str, payload: &Value) -> Result<u64, StoreError> {
        Ok(self.publish_mq_detail(topic, key, payload)?.0)
    }

    /// Message sequence and the store sequence captured under the document lock.
    pub(super) fn publish_mq_detail(
        &self,
        topic: &str,
        key: &str,
        payload: &Value,
    ) -> Result<(u64, u64), StoreError> {
        self.refuse_reentry()?;
        if let Inner::Remote(remote) = &self.inner {
            let registered = {
                let _gate = lock_mutex(&self.commit_gate);
                let value = remote.call_req(
                    "publish_mq",
                    json!({"topic": topic, "key": key, "payload": payload}),
                )?;
                let (mq_seq, store_seq) = parse_publish_result(&value)?;
                let registered = self.register_committed(
                    &Committed {
                        seq: store_seq,
                        ops: mq_publish_ops(topic, key, payload, mq_seq),
                    },
                    false,
                );
                (mq_seq, store_seq, registered)
            };
            self.deliver_owned(registered.2)?;
            return Ok((registered.0, registered.1));
        }
        let topic_owned = topic.to_string();
        let key_owned = key.to_string();
        let payload = payload.clone();
        // The record the body staged, for subscribers when the commit is
        // durable but its batch was not delivered.
        let mut staged: Option<MqRecord> = None;
        let staged_slot = &mut staged;
        // One `with_write`: the message and the high-water share a WAL frame.
        // A crash must not persist either document alone.
        let outcome = self.with_write_outcome(move |txn| {
            let seq = next_mq_seq(txn, &topic_owned)?;
            let record = MqRecord {
                seq,
                key: key_owned.clone(),
                payload: payload.clone(),
            };
            *staged_slot = Some(record.clone());
            txn.put(
                mq_message_collection(&topic_owned),
                seq.to_string(),
                serde_json::json!({
                    "seq": seq,
                    "key": key_owned,
                    "payload": payload,
                }),
            );
            txn.put(
                mq_hw_collection(&topic_owned),
                MQ_HIGH_WATER_ID,
                serde_json::json!({ "seq": seq }),
            );
            Ok(record)
        });
        let (record, store_seq) = match outcome {
            Ok(done) => done,
            // Durable, not yet sent to the adapter: the slot sync is done, so
            // subscribers get the message all the same (spec §4.3). Every
            // other error committed nothing, and no subscriber hears of it.
            Err(err @ StoreError::Undelivered { .. }) => {
                if let Some(record) = staged.take() {
                    self.notify(topic, &record);
                }
                return Err(err);
            }
            Err(err) => return Err(err),
        };
        let store_seq = store_seq.ok_or_else(|| backend("publish_mq did not commit"))?;
        self.notify(topic, &record);
        Ok((record.seq, store_seq))
    }

    /// Oldest messages for `topic` that `group` has not acked, without acking them.
    pub fn poll_mq_group(
        &self,
        topic: &str,
        group: &str,
        limit: usize,
    ) -> Result<Vec<MqRecord>, StoreError> {
        self.refuse_reentry()?;
        if let Inner::Remote(remote) = &self.inner {
            return remote.call_records(
                "poll_mq_group",
                json!({"topic": topic, "group": group, "limit": limit}),
            );
        }
        let state = self.read_state()?;
        unread_mq(&state, topic, group, limit)
    }

    /// Record an ack. An unknown sequence is still stored.
    pub fn ack_mq(&self, topic: &str, group: &str, seq: u64) -> Result<(), StoreError> {
        let topic = topic.to_string();
        let group = group.to_string();
        self.commit(
            move || {
                Ok((
                    Op::Put {
                        collection: format!("mq_ack:{topic}:{group}"),
                        id: seq.to_string(),
                        doc: serde_json::json!({ "seq": seq }),
                    },
                    (),
                ))
            },
            false,
        )
    }

    /// Later publishes on this handle. History is not replayed.
    pub fn subscribe_mq(&self, topic: &str) -> Receiver<MqRecord> {
        if let Inner::Remote(remote) = &self.inner {
            return remote.subscribe_topic(topic);
        }
        let (tx, rx) = mpsc::channel();
        lock_mutex(&self.subs)
            .entry(topic.to_string())
            .or_default()
            .push(tx);
        rx
    }

    /// Attach `adapter` in `mode`. A pending batch keeps the current adapter.
    pub fn attach_sync_adapter(
        &self,
        mode: SyncMode,
        adapter: Arc<dyn SyncAdapter>,
    ) -> Result<(), StoreError> {
        self.refuse_reentry()?;
        let mut slot = lock_mutex(&self.sync);
        if !slot.is_idle() {
            return Err(backend(
                "cannot attach a sync adapter while batches are pending",
            ));
        }
        slot.mode = Some(mode);
        slot.adapter = Some(adapter);
        Ok(())
    }

    /// Set `mode` and `adapter` when the pending buffer is empty.
    /// Pending batches keep the current adapter.
    pub fn with_sync_adapter(
        self,
        mode: SyncMode,
        adapter: Arc<dyn SyncAdapter>,
    ) -> Result<Self, StoreError> {
        self.refuse_reentry()?;
        self.attach_sync_adapter(mode, adapter)?;
        Ok(self)
    }

    /// Drop the adapter. A pending batch is an error and the adapter stays.
    /// `synced_seq` is kept.
    pub fn detach_sync_adapter(&self) -> Result<(), StoreError> {
        self.refuse_reentry()?;
        let mut slot = lock_mutex(&self.sync);
        if !slot.is_idle() {
            return Err(backend(
                "cannot detach a sync adapter while batches are pending",
            ));
        }
        slot.mode = None;
        slot.adapter = None;
        Ok(())
    }

    pub fn sync_mode(&self) -> Option<SyncMode> {
        lock_mutex(&self.sync).mode
    }

    pub fn synced_seq(&self) -> u64 {
        lock_mutex(&self.sync).synced_seq
    }

    /// `true` while a batch is queued or an `on_commit` call is running.
    pub fn has_pending_sync(&self) -> bool {
        !lock_mutex(&self.sync).is_idle()
    }

    /// The attached adapter, cloned so no lock is held across a call into it.
    fn attached_adapter(&self) -> Option<Arc<dyn SyncAdapter>> {
        lock_mutex(&self.sync).adapter.clone()
    }

    /// Call `on_commit` for each pending batch, oldest first.
    ///
    /// Waits while another thread's notify is in flight, then sends only the
    /// front batch. The thread that owns that notify returns `synced_seq` and
    /// does not send. A later batch stays pending until this loop reaches it.
    ///
    /// Refused from inside a transaction body on this handle
    /// ([`Self::refuse_reentry`]); a call from inside `on_commit`, which runs
    /// outside every body, is not.
    pub fn sync_flush(&self) -> Result<u64, StoreError> {
        self.refuse_reentry()?;
        let adapter = {
            let slot = lock_mutex(&self.sync);
            slot.adapter
                .clone()
                .ok_or_else(|| backend("no sync adapter is attached"))?
        };
        loop {
            let batch = {
                let mut slot = lock_mutex(&self.sync);
                while slot.notify_in_flight {
                    if slot.notify_owner == Some(std::thread::current().id()) {
                        return Ok(slot.synced_seq);
                    }
                    slot = wait_mutex(&self.sync_cv, slot);
                }
                match slot.pending.first() {
                    Some(batch) => {
                        let batch = batch.clone();
                        slot.notify_in_flight = true;
                        slot.notify_owner = Some(std::thread::current().id());
                        batch
                    }
                    None => return Ok(slot.synced_seq),
                }
            };
            let result = self.call_on_commit(&adapter, &batch);
            let mut slot = lock_mutex(&self.sync);
            slot.notify_in_flight = false;
            slot.notify_owner = None;
            let failed = match result {
                Ok(()) => {
                    if slot
                        .pending
                        .first()
                        .is_some_and(|pending| pending.seq == batch.seq)
                    {
                        slot.pending.remove(0);
                        if batch.seq > slot.synced_seq {
                            slot.synced_seq = batch.seq;
                        }
                    }
                    None
                }
                Err(err) => Some(err),
            };
            self.sync_cv.notify_all();
            if let Some(err) = failed {
                return Err(err);
            }
        }
    }

    /// Apply `pull_since(last_seq)` with sync suppressed.
    pub fn sync_pull(&self) -> Result<u64, StoreError> {
        self.refuse_reentry()?;
        let adapter = self.require_sync_adapter()?;
        self.require_sync_idle()?;
        let last = self.last_seq()?;
        let batches = adapter.pull_since(last)?;
        self.apply_inbound(batches)
    }

    /// Apply `prewarm` batches with sync suppressed.
    pub fn sync_prewarm(&self, collections: &[&str]) -> Result<u64, StoreError> {
        self.refuse_reentry()?;
        let adapter = self.require_sync_adapter()?;
        self.require_sync_idle()?;
        let batches = adapter.prewarm(collections)?;
        self.apply_inbound(batches)
    }

    /// Run a read-only `body` over the current state. A remote handle runs it
    /// over a read session (spec §6.1): every read is one RPC against the
    /// session's snapshot, and the session is closed when `body` returns.
    fn with_read<T>(
        &self,
        body: impl FnOnce(&Txn) -> Result<T, StoreError>,
    ) -> Result<T, StoreError> {
        self.refuse_reentry()?;
        let txn = match &self.inner {
            Inner::Remote(remote) => Txn::remote(Arc::clone(remote.shared())),
            Inner::Memory(_) | Inner::Disk(_) => Txn::new(self.read_state()?),
        };
        let result = body(&txn);
        match txn.take_read_error() {
            Some(err) => Err(err),
            None => result,
        }
    }

    /// The current state. Memory shares the live `Arc`, and Disk the cached
    /// one, refreshed without a lock (spec §4.1): a later commit replaces it
    /// and leaves this one as it was. A remote handle never receives the
    /// whole state (spec §6.1): it reads through a session.
    fn read_state(&self) -> Result<Arc<DocState>, StoreError> {
        self.refuse_reentry()?;
        match &self.inner {
            Inner::Memory(state) => Ok(Arc::clone(&lock_mutex(state))),
            Inner::Disk(disk) => Ok(Arc::clone(&disk.refresh_read()?.state)),
            Inner::Remote(_) => Err(backend(
                "a remote FerroStore does not receive the whole state: it reads through a \
                 session",
            )),
        }
    }

    /// The last committed sequence. A remote handle opens a session for its
    /// `base_seq` and closes it.
    fn last_seq(&self) -> Result<u64, StoreError> {
        match &self.inner {
            Inner::Remote(remote) => {
                self.refuse_reentry()?;
                remote.last_seq()
            }
            Inner::Memory(_) | Inner::Disk(_) => Ok(self.read_state()?.last_seq),
        }
    }

    /// Compare-and-swap `ops` onto a store whose `last_seq` is still `expected_seq`.
    ///
    /// `Ok(false)` means the sequence moved and nothing was written. An empty `ops`
    /// list that matches returns `Ok(true)` and does not advance `last_seq`. A
    /// non-empty disk success is one WAL frame at `expected_seq + 1`. Ops that do
    /// not apply return `Err` and write nothing. A remote handle has no such
    /// swap: it commits through a validated transaction (spec §6.1).
    #[cfg(test)]
    fn commit_if_seq(&self, expected_seq: u64, ops: &[Op]) -> Result<bool, StoreError> {
        self.commit_if_seq_inner(expected_seq, ops, false)
    }

    #[cfg(test)]
    fn commit_if_seq_inner(
        &self,
        expected_seq: u64,
        ops: &[Op],
        silent: bool,
    ) -> Result<bool, StoreError> {
        self.refuse_reentry()?;
        match &self.inner {
            Inner::Memory(slot) => {
                let registered = {
                    let _gate = lock_mutex(&self.commit_gate);
                    let committed = {
                        let mut state = lock_mutex(slot);
                        if state.last_seq != expected_seq {
                            return Ok(false);
                        }
                        if ops.is_empty() {
                            return Ok(true);
                        }
                        let owned = ops.to_vec();
                        state::commit_ops(&mut state, &owned)?;
                        Committed {
                            seq: state.last_seq,
                            ops: owned,
                        }
                    };
                    self.register_committed(&committed, silent)
                };
                self.deliver_owned(registered)?;
                Ok(true)
            }
            Inner::Disk(disk) => {
                // The expected sequence is checked against the group's
                // running state at this member's turn (spec §4.3).
                let (outcome, _) = self.group_write(disk, silent, |state| {
                    if state.last_seq != expected_seq {
                        return Ok((SeqCommit::Lost, None));
                    }
                    if ops.is_empty() {
                        return Ok((SeqCommit::Same, None));
                    }
                    let (overlay, _) = stage_ops(&state, ops)?;
                    let ops = ops.to_vec();
                    Ok((SeqCommit::Wrote, Some(Staged { ops, overlay })))
                })?;
                Ok(!matches!(outcome, SeqCommit::Lost))
            }
            Inner::Remote(_) => Err(backend(
                "a remote FerroStore has no compare-and-swap on the sequence: it commits \
                 through a validated transaction",
            )),
        }
    }

    /// Run `body` in a transaction: an overlay on the current state.
    ///
    /// Memory holds the document mutex for the whole call. Disk joins a group
    /// commit (spec §4.3): its leader takes `LOCK_EX` and refreshes the cached
    /// state (spec §4.1: the slots, then only WAL frames another handle
    /// added), and `body` runs, on this thread, over that state with the
    /// commits of the group's earlier members. `Ok` with at least one op
    /// appends one WAL frame (`seq = last_seq + 1`), published with the
    /// group's superblock. `Err` or an empty op list writes nothing.
    ///
    /// A remote handle runs `body` over a read session (spec §6.1) and sends
    /// its ops with its read set as `commit_txn`; the server recomputes every
    /// read inside its write lock and commits only when each still has the
    /// value `body` saw. A lost commit runs `body` again over a new session,
    /// at most [`retry::REMOTE_ATTEMPTS`] times in all, with a jittered
    /// backoff between attempts; then it is [`StoreError::Conflict`] (spec
    /// §4.5). The body may run more than once only on a remote handle, so it
    /// must be free of side effects. On disk, a commit that leaves the WAL
    /// past the threshold auto-checkpoints (spec §4.4, [`AUTO_CHECKPOINT_BYTES`]).
    /// The sync adapter runs after the slot is synced, on the committing
    /// thread, with none of this thread's locks or the commit gate held.
    ///
    /// While `body` runs, every call this thread makes on this handle is
    /// refused ([`Self::refuse_reentry`]).
    #[track_caller]
    fn with_write<T>(
        &self,
        body: impl FnMut(&mut Txn) -> Result<T, StoreError>,
    ) -> Result<T, StoreError> {
        Ok(self.with_write_outcome(body)?.0)
    }

    /// Like [`Self::with_write`]. The sequence was taken before the document
    /// mutex or disk flock was released. `None` means the body wrote no ops.
    #[track_caller]
    fn with_write_outcome<T>(
        &self,
        mut body: impl FnMut(&mut Txn) -> Result<T, StoreError>,
    ) -> Result<(T, Option<u64>), StoreError> {
        self.refuse_reentry()?;
        let caller = std::panic::Location::caller();
        match &self.inner {
            Inner::Remote(remote) => {
                let mut backoff = retry::Backoff::new();
                for attempt in 1..=retry::REMOTE_ATTEMPTS {
                    if attempt > 1 {
                        backoff.wait(attempt - 1);
                    }
                    if let Some(done) = self.remote_attempt(remote, caller, attempt, &mut body)? {
                        return Ok(done);
                    }
                }
                Err(self.remote_conflict())
            }
            Inner::Memory(_) | Inner::Disk(_) => self.write_local(false, body),
        }
    }

    /// The body of a `mem://` or `file://` write ([`Self::with_write_outcome`]),
    /// and of a remote client's `commit_txn` on the server. `silent` commits
    /// without the sync adapter.
    fn write_local<T>(
        &self,
        silent: bool,
        mut body: impl FnMut(&mut Txn) -> Result<T, StoreError>,
    ) -> Result<(T, Option<u64>), StoreError> {
        match &self.inner {
            Inner::Remote(_) => Err(backend(
                "a remote FerroStore cannot run a write body in this process",
            )),
            Inner::Memory(slot) => {
                let (result, seq, registered) = {
                    let _gate = lock_mutex(&self.commit_gate);
                    let (result, committed) = {
                        let mut state = lock_mutex(slot);
                        let mut txn = Txn::new(Arc::clone(&state));
                        let result = {
                            let _body = self.enter_body();
                            body(&mut txn)
                        }?;
                        // Ends the transaction's hold on the base: with no
                        // reader holding it either, the commit is in place.
                        let (ops, overlay) = txn.finish()?;
                        let committed = if ops.is_empty() {
                            None
                        } else {
                            let seq = state
                                .last_seq
                                .checked_add(1)
                                .ok_or_else(|| backend("transaction sequence overflow"))?;
                            publish_overlay(&mut state, overlay, seq);
                            Some(Committed { seq, ops })
                        };
                        (result, committed)
                    };
                    let seq = committed.as_ref().map(|committed| committed.seq);
                    let registered = committed.as_ref().map_or(Registered::Nothing, |committed| {
                        self.register_committed(committed, silent)
                    });
                    (result, seq, registered)
                };
                self.deliver_owned(registered)?;
                Ok((result, seq))
            }
            Inner::Disk(disk) => self.group_write(disk, silent, |state| {
                let mut txn = Txn::new(state);
                // On the thread running the body: in a group, the member's
                // own thread at its turn (spec §4.3).
                let result = {
                    let _body = self.enter_body();
                    body(&mut txn)
                }?;
                // The overlay was checked op by op as it was staged: once the
                // frame is published it is applied to the cached state, and
                // nothing is re-read. `finish` drops the hold on the state.
                let (ops, overlay) = txn.finish()?;
                let staged = (!ops.is_empty()).then_some(Staged { ops, overlay });
                Ok((result, staged))
            }),
        }
    }

    /// One attempt of a remote write (spec §4.5, §6.1): run `body` over a
    /// new read session and send its ops with its read set. `None` when
    /// another writer changed what `body` read, or the session expired under
    /// a read: nothing was written and the caller may try again. `Err` from the body, from a read, or from the
    /// server, ends the write. A body that wrote nothing commits nothing and
    /// closes its session.
    fn remote_attempt<T>(
        &self,
        remote: &server::RemoteClient,
        caller: &'static std::panic::Location<'static>,
        attempt: u32,
        body: &mut impl FnMut(&mut Txn) -> Result<T, StoreError>,
    ) -> Result<Option<(T, Option<u64>)>, StoreError> {
        let mut txn = Txn::remote(Arc::clone(remote.shared()));
        let result = {
            let _body = self.enter_body();
            body(&mut txn)
        };
        if let Some(err) = txn.take_read_error() {
            // The session expired under the body (idle, or at its lifetime):
            // nothing was sent, so the attempt is lost and runs again.
            if session::is_session_expired(&err) {
                return Ok(None);
            }
            return Err(err);
        }
        let result = result?;
        if txn.failed {
            return Err(backend(
                "a transaction whose staged op failed cannot commit",
            ));
        }
        if txn.ops.is_empty() {
            return Ok(Some((result, None)));
        }
        let Some(session) = txn.remote_base() else {
            return Err(backend("a remote write ran over a local state"));
        };
        let known = session.known();
        let forced = retry::forced_loss(&retry::Cas {
            caller,
            attempt,
            known: &known,
            ops: &txn.ops,
        });
        if forced {
            return Ok(None);
        }
        let (pinned, reads) = session.take_for_commit();
        let ops = std::mem::take(&mut txn.ops);
        let committed = {
            let _gate = lock_mutex(&self.commit_gate);
            self.commit_remote(remote, pinned, &reads, ops, false)?
        };
        match committed {
            Some(committed) => {
                let seq = self.deliver_remote(committed)?;
                Ok(Some((result, Some(seq))))
            }
            None => Ok(None),
        }
    }

    /// Send `ops` with the read set `reads` (taken at `session`) as one
    /// `commit_txn`, under a new request id, and, when the server committed
    /// them, register this handle's own batch for them. The caller holds the
    /// commit gate. `None`: a read no longer has the value it had, and
    /// nothing was written.
    ///
    /// The server's [`StoreError::Undelivered`] (its own adapter refused) is
    /// a commit too: the batch is registered here all the same, and the
    /// error is returned by [`Self::deliver_remote`] once this handle's own
    /// adapter has had the batch.
    fn commit_remote(
        &self,
        remote: &server::RemoteClient,
        session: Option<(u64, u64)>,
        reads: &[(session::ReadKey, String)],
        ops: Vec<Op>,
        silent: bool,
    ) -> Result<Option<RemoteCommit>, StoreError> {
        let (seq, server) = match remote.commit_txn(session, reads, &ops, silent) {
            Ok(None) => return Ok(None),
            Ok(Some(seq)) => (seq, None),
            Err(err @ StoreError::Undelivered { .. }) => {
                let StoreError::Undelivered { seq, .. } = &err else {
                    return Err(err);
                };
                (*seq, Some(err))
            }
            Err(err) => return Err(err),
        };
        let registered = self.register_committed(&Committed { seq, ops }, silent);
        Ok(Some(RemoteCommit {
            seq,
            registered,
            server,
        }))
    }

    /// Send what [`Self::commit_remote`] registered, once the commit gate is
    /// released, and return the commit's sequence, or the server's
    /// `Undelivered` when its adapter refused (this handle's own delivery
    /// was still tried: the batch is pending here if it failed too).
    fn deliver_remote(&self, committed: RemoteCommit) -> Result<u64, StoreError> {
        let delivered = self.deliver_owned(committed.registered);
        match committed.server {
            Some(err) => Err(err),
            None => delivered.map(|()| committed.seq),
        }
    }

    /// The error of a remote write whose every attempt lost (spec §4.5).
    fn remote_conflict(&self) -> StoreError {
        let id = Iri::parse(&self.label)
            .or_else(|_| Iri::parse(&format!("unix://{}", self.label)))
            .unwrap_or_else(|_| Iri::parse("urn:ferrostore:remote").expect("a valid IRI"));
        StoreError::Conflict {
            id,
            detail: format!(
                "{} commit attempts each lost to a commit by another writer that changed what \
                 this write read, and this write committed nothing",
                retry::REMOTE_ATTEMPTS
            ),
        }
    }

    /// Commit the one op `build` makes, without reading anything. A remote
    /// handle sends it as a `commit_txn` with an empty read set, which no
    /// other writer can make lose.
    #[track_caller]
    fn commit<T>(
        &self,
        build: impl FnOnce() -> Result<(Op, T), StoreError>,
        silent: bool,
    ) -> Result<T, StoreError> {
        self.refuse_reentry()?;
        let caller = std::panic::Location::caller();
        match &self.inner {
            Inner::Remote(remote) => {
                let (op, extra) = build()?;
                let mut backoff = retry::Backoff::new();
                for attempt in 1..=retry::REMOTE_ATTEMPTS {
                    if attempt > 1 {
                        backoff.wait(attempt - 1);
                    }
                    let forced = retry::forced_loss(&retry::Cas {
                        caller,
                        attempt,
                        known: &"",
                        ops: std::slice::from_ref(&op),
                    });
                    let committed = if forced {
                        None
                    } else {
                        let _gate = lock_mutex(&self.commit_gate);
                        self.commit_remote(remote, None, &[], vec![op.clone()], silent)?
                    };
                    if let Some(committed) = committed {
                        self.deliver_remote(committed)?;
                        return Ok(extra);
                    }
                }
                Err(self.remote_conflict())
            }
            Inner::Memory(slot) => {
                let (extra, registered) = {
                    let _gate = lock_mutex(&self.commit_gate);
                    let (extra, committed) = {
                        let mut state = lock_mutex(slot);
                        let (op, extra) = build()?;
                        apply_commit(&mut state, op.clone())?;
                        (
                            extra,
                            Committed {
                                seq: state.last_seq,
                                ops: vec![op],
                            },
                        )
                    };
                    (extra, self.register_committed(&committed, silent))
                };
                self.deliver_owned(registered)?;
                Ok(extra)
            }
            Inner::Disk(disk) => {
                let (extra, _) = self.group_write(disk, silent, |state| {
                    let (op, extra) = build()?;
                    let (overlay, _) = stage_ops(&state, std::slice::from_ref(&op))?;
                    let ops = vec![op];
                    Ok((extra, Some(Staged { ops, overlay })))
                })?;
                Ok(extra)
            }
        }
    }

    fn fill_doc_miss(&self, collection: &str, id: &str) -> Result<Option<Value>, StoreError> {
        // The store's own bookkeeping is local: never filled from a peer.
        if doc::is_reserved(collection) {
            return Ok(None);
        }
        let adapter = {
            let slot = lock_mutex(&self.sync);
            match slot.mode {
                Some(SyncMode::LocalCache | SyncMode::CloudPreCache) if slot.is_idle() => {
                    slot.adapter.clone()
                }
                _ => None,
            }
        };
        let Some(adapter) = adapter else {
            return Ok(None);
        };
        match adapter.fetch_document(collection, id)? {
            None => Ok(None),
            Some(doc) => self.install_fetched_doc(collection, id, doc),
        }
    }

    /// Silent put of a cache fill. Does not call `on_commit` or queue the batch.
    ///
    /// Holds the commit gate, then the document lock, then the sync mutex.
    /// A commit releases the document lock before taking the sync mutex.
    ///
    /// On disk this does not join a group (spec §4.3): its sync-idle check
    /// and its `synced_seq` move are one step with its commit, under the sync
    /// mutex, and a group registers its earlier members only after its slot
    /// sync, so inside a group the check would read idle too early. The
    /// commit gate keeps it apart from every group.
    fn install_fetched_doc(
        &self,
        collection: &str,
        id: &str,
        fetched: Value,
    ) -> Result<Option<Value>, StoreError> {
        self.refuse_reentry()?;
        let key = (collection.to_string(), id.to_string());
        let _gate = lock_mutex(&self.commit_gate);
        if let Inner::Remote(remote) = &self.inner {
            if !lock_mutex(&self.sync).is_idle() {
                drop(_gate);
                return remote.call_option("get_doc", json!({"collection": collection, "id": id}));
            }
            let value = remote.call_req(
                "put_doc",
                json!({
                    "collection": collection,
                    "id": id,
                    "doc": fetched,
                    "silent": true,
                    "if_absent": true,
                }),
            )?;
            let (wrote, doc, store_seq) = parse_if_absent(&value)?;
            if let Some(store_seq) = store_seq {
                let mut slot = lock_mutex(&self.sync);
                if wrote && slot.is_idle() && store_seq > slot.synced_seq {
                    slot.synced_seq = store_seq;
                }
            }
            return Ok(Some(doc));
        }
        match &self.inner {
            Inner::Remote(_) => unreachable!("remote fill returned above"),
            Inner::Memory(slot) => {
                let mut state = lock_mutex(slot);
                let mut sync = lock_mutex(&self.sync);
                if !sync.is_idle() {
                    return Ok(state.docs.get(&key.0, &key.1).cloned());
                }
                if let Some(existing) = state.docs.get(&key.0, &key.1).cloned() {
                    return Ok(Some(existing));
                }
                let op = Op::Put {
                    collection: key.0.clone(),
                    id: key.1.clone(),
                    doc: fetched.clone(),
                };
                apply_commit(&mut state, op)?;
                let seq = state.last_seq;
                if sync.is_idle() && seq > sync.synced_seq {
                    sync.synced_seq = seq;
                }
                Ok(Some(fetched))
            }
            Inner::Disk(disk) => {
                let due = {
                    let mut write = lock_mutex(&disk.write);
                    // Reads go on on a poisoned handle (see [`Poison`]): the
                    // adapter's document is returned, and not stored. Every
                    // sync is made under the write mutex, so this is exact.
                    if disk.poison.is_set() {
                        return Ok(Some(fetched));
                    }
                    let locked = disk.lock_current(&mut write)?;
                    let file = locked.file();
                    let (index, base) = disk.refresh(file, locked.id)?;
                    let state = &base.state;
                    let mut sync = lock_mutex(&self.sync);
                    if !sync.is_idle() {
                        return Ok(state.docs.get(&key.0, &key.1).cloned());
                    }
                    if let Some(existing) = state.docs.get(&key.0, &key.1).cloned() {
                        return Ok(Some(existing));
                    }
                    let op = Op::Put {
                        collection: key.0.clone(),
                        id: key.1.clone(),
                        doc: fetched.clone(),
                    };
                    let (overlay, seq) = stage_ops(state, std::slice::from_ref(&op))?;
                    let frame = encode_frame(seq, std::slice::from_ref(&op))?;
                    let published = disk.publish_or_settle(file, index, &base, &[frame])?;
                    let due = match published {
                        Some(new_sb) => {
                            let done =
                                disk.install_commit(base, new_sb, Change::Overlay(overlay, seq));
                            checkpoint_due(disk, &done)
                        }
                        None => false,
                    };
                    if sync.is_idle() && seq > sync.synced_seq {
                        sync.synced_seq = seq;
                    }
                    due
                };
                // Spec §4.4: once the gate, the write mutex, the `flock` and
                // the sync mutex are released.
                drop(_gate);
                if due {
                    auto_checkpoint(disk);
                }
                Ok(Some(fetched))
            }
        }
    }

    /// Conditional put used by a remote cache fill. The document lock covers
    /// the absence check and the write. `store_seq` is taken before that lock
    /// is released.
    fn put_if_absent(
        &self,
        collection: &str,
        id: &str,
        doc: &Value,
        silent: bool,
    ) -> Result<IfAbsent, StoreError> {
        self.refuse_reentry()?;
        doc::refuse_reserved(collection, "put_doc")?;
        let key = (collection.to_string(), id.to_string());
        match &self.inner {
            Inner::Remote(_) => Err(backend(
                "if_absent put is local to the store that owns the file",
            )),
            Inner::Memory(slot) => {
                let (seq, registered) = {
                    let _gate = lock_mutex(&self.commit_gate);
                    let committed = {
                        let mut state = lock_mutex(slot);
                        if let Some(existing) = state.docs.get(&key.0, &key.1).cloned() {
                            return Ok(IfAbsent {
                                wrote: false,
                                doc: existing,
                                store_seq: None,
                            });
                        }
                        let op = Op::Put {
                            collection: key.0.clone(),
                            id: key.1.clone(),
                            doc: doc.clone(),
                        };
                        apply_commit(&mut state, op.clone())?;
                        (state.last_seq, op)
                    };
                    (
                        committed.0,
                        self.register_committed(
                            &Committed {
                                seq: committed.0,
                                ops: vec![committed.1],
                            },
                            silent,
                        ),
                    )
                };
                self.deliver_owned(registered)?;
                Ok(IfAbsent {
                    wrote: true,
                    doc: doc.clone(),
                    store_seq: Some(seq),
                })
            }
            Inner::Disk(disk) => {
                let (existing, seq) = self.group_write(disk, silent, |state| {
                    if let Some(existing) = state.docs.get(&key.0, &key.1).cloned() {
                        return Ok((Some(existing), None));
                    }
                    let op = Op::Put {
                        collection: key.0,
                        id: key.1,
                        doc: doc.clone(),
                    };
                    let (overlay, _) = stage_ops(&state, std::slice::from_ref(&op))?;
                    let ops = vec![op];
                    Ok((None, Some(Staged { ops, overlay })))
                })?;
                Ok(match existing {
                    Some(existing) => IfAbsent {
                        wrote: false,
                        doc: existing,
                        store_seq: None,
                    },
                    None => IfAbsent {
                        wrote: true,
                        doc: doc.clone(),
                        store_seq: seq,
                    },
                })
            }
        }
    }

    fn require_sync_adapter(&self) -> Result<Arc<dyn SyncAdapter>, StoreError> {
        lock_mutex(&self.sync)
            .adapter
            .clone()
            .ok_or_else(|| backend("no sync adapter is attached"))
    }

    fn require_sync_idle(&self) -> Result<(), StoreError> {
        if lock_mutex(&self.sync).is_idle() {
            Ok(())
        } else {
            Err(backend(
                "cannot apply sync batches while commits are pending",
            ))
        }
    }

    fn apply_inbound(&self, batches: Vec<SyncBatch>) -> Result<u64, StoreError> {
        self.apply_sync_batches(batches, true)
    }

    /// Dry-run every batch, then publish them in one critical section.
    /// `silent` does not call `on_commit`. An empty list returns `synced_seq`.
    ///
    /// A loud apply queues every prepared batch. Only `WriteThrough` then calls
    /// `sync_flush`. The other modes leave the batches pending.
    /// The commit gate covers the idle check, the publish, and that registration.
    fn apply_sync_batches(&self, batches: Vec<SyncBatch>, silent: bool) -> Result<u64, StoreError> {
        self.refuse_reentry()?;
        if batches.is_empty() {
            return Ok(self.synced_seq());
        }
        if let Inner::Remote(remote) = &self.inner {
            let seq = {
                let _gate = lock_mutex(&self.commit_gate);
                self.require_sync_idle()?;
                let seq = remote.apply_sync_batches(&batches, silent)?;
                let mut slot = lock_mutex(&self.sync);
                if slot.is_idle() && seq > slot.synced_seq {
                    slot.synced_seq = seq;
                }
                seq
            };
            return Ok(seq);
        }
        let (seq, flush, due) = {
            let _gate = lock_mutex(&self.commit_gate);
            if !lock_mutex(&self.sync).is_idle() {
                return Err(backend(
                    "cannot apply sync batches while commits are pending",
                ));
            }
            let (seq, prepared, due) = publish_sync_batches(&self.inner, &batches)?;
            let mut slot = lock_mutex(&self.sync);
            if silent {
                if slot.is_idle() && seq > slot.synced_seq {
                    slot.synced_seq = seq;
                }
                (seq, false, due)
            } else {
                let queue = slot.adapter.is_some() && slot.mode.is_some();
                let flush = queue && slot.mode == Some(SyncMode::WriteThrough);
                if queue {
                    // The batches as they came in, not as applied: a reserved
                    // op this store dropped is still the next store's to
                    // judge, and a batch whose every op was dropped must not
                    // go out empty.
                    debug_assert_eq!(prepared.len(), batches.len());
                    slot.pending.extend(batches);
                }
                (seq, flush, due)
            }
        };
        let flushed = if flush {
            self.sync_flush()
                .map(|_| ())
                .map_err(|cause| undelivered(seq, cause))
        } else {
            Ok(())
        };
        // Spec §4.4: once the gate is released and the batches are sent.
        if due && let Inner::Disk(disk) = &self.inner {
            auto_checkpoint(disk);
        }
        flushed?;
        Ok(seq)
    }

    /// Caller holds the commit gate and no document lock. Does not call `on_commit`.
    fn register_committed(&self, committed: &Committed, silent: bool) -> Registered {
        self.register_committed_for(committed, silent, std::thread::current().id())
    }

    /// [`Self::register_committed`] for a commit whose `on_commit` runs on
    /// `owner`: a group's leader registers each member's commit, and the
    /// member sends it (spec §4.3).
    fn register_committed_for(
        &self,
        committed: &Committed,
        silent: bool,
        owner: ThreadId,
    ) -> Registered {
        if committed.ops.is_empty() {
            return Registered::Nothing;
        }
        {
            let mut slot = lock_mutex(&self.sync);
            if silent {
                if slot.is_idle() && committed.seq > slot.synced_seq {
                    slot.synced_seq = committed.seq;
                }
                return Registered::Nothing;
            }
            if slot.adapter.is_none() || slot.mode.is_none() {
                return Registered::Nothing;
            }
        }
        // Only a batch that is queued or sent is built: it clones every op,
        // vector and column bytes included, and a group's leader builds its
        // members' under the gate, the write mutex and the `flock`. It is
        // built with the sync mutex released; the slot as the second lock
        // finds it decides where the batch goes.
        let batch = SyncBatch {
            seq: committed.seq,
            ops: committed.ops.iter().map(SyncOp::from_op).collect(),
        };
        let mut slot = lock_mutex(&self.sync);
        let (Some(adapter), Some(mode)) = (slot.adapter.clone(), slot.mode) else {
            return Registered::Nothing;
        };
        match mode {
            SyncMode::WriteThrough => {
                if !slot.is_idle() {
                    let seq = batch.seq;
                    slot.pending.push(batch);
                    Registered::Queued(seq)
                } else {
                    slot.notify_in_flight = true;
                    slot.notify_owner = Some(owner);
                    Registered::Owned(adapter, batch)
                }
            }
            SyncMode::WriteBehind | SyncMode::LocalCache | SyncMode::CloudPreCache => {
                slot.pending.push(batch);
                Registered::Nothing
            }
        }
    }

    /// Run `adapter.on_commit(batch)` for a caller that has set `notify_in_flight`.
    ///
    /// If the adapter panics, the flag, the owner and the queue are put right before the panic
    /// goes on: the batch stays queued, so a later `sync_flush` sends it. Without this, the flag
    /// would stay set for good and every other thread's `sync_flush` would wait on `sync_cv`
    /// forever, while `attach`, `detach` and `sync_pull` would refuse for good.
    fn call_on_commit(
        &self,
        adapter: &Arc<dyn SyncAdapter>,
        batch: &SyncBatch,
    ) -> Result<(), StoreError> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| adapter.on_commit(batch))) {
            Ok(result) => result,
            Err(payload) => {
                let mut slot = lock_mutex(&self.sync);
                slot.notify_in_flight = false;
                slot.notify_owner = None;
                if !slot.pending.iter().any(|p| p.seq == batch.seq) {
                    slot.pending.insert(0, batch.clone());
                }
                self.sync_cv.notify_all();
                drop(slot);
                std::panic::resume_unwind(payload)
            }
        }
    }

    /// Send what [`Self::register_committed`] left, once the commit gate and
    /// every other store lock are released. In `WriteThrough` this returns
    /// `Ok` only once the commit's own batch is sent: `on_commit` returned
    /// `Ok` for it and `synced_seq` is at least its sequence.
    ///
    /// An owned batch is sent here. A queued one is drained to: see
    /// [`Self::drain_through`]. A failed `on_commit` returns the adapter's
    /// error and leaves the batch at the front of the queue: the commit is
    /// durable and not yet sent. Later pending batches of other commits stay
    /// queued for their own committers or `sync_flush`.
    ///
    /// Every error here is [`StoreError::Undelivered`] at the commit's own
    /// sequence, so a caller tells "durable, not yet sent" from "not
    /// committed".
    fn deliver_owned(&self, registered: Registered) -> Result<(), StoreError> {
        let seq = match &registered {
            Registered::Nothing => return Ok(()),
            Registered::Queued(seq) => *seq,
            Registered::Owned(_, batch) => batch.seq,
        };
        self.deliver_registered(registered)
            .map_err(|cause| undelivered(seq, cause))
    }

    /// [`Self::deliver_owned`] with the adapter's own error.
    fn deliver_registered(&self, registered: Registered) -> Result<(), StoreError> {
        let (adapter, batch) = match registered {
            Registered::Nothing => return Ok(()),
            Registered::Queued(seq) => return self.drain_through(seq),
            Registered::Owned(adapter, batch) => (adapter, batch),
        };
        let result = self.call_on_commit(&adapter, &batch);
        let mut slot = lock_mutex(&self.sync);
        slot.notify_in_flight = false;
        slot.notify_owner = None;
        match result {
            Ok(()) => {
                if batch.seq > slot.synced_seq {
                    slot.synced_seq = batch.seq;
                }
            }
            Err(err) => {
                if !slot.pending.iter().any(|pending| pending.seq == batch.seq) {
                    slot.pending.insert(0, batch);
                }
                self.sync_cv.notify_all();
                return Err(err);
            }
        }
        self.sync_cv.notify_all();
        Ok(())
    }

    /// Send queued batches, oldest first, with [`Self::sync_flush`] (which
    /// waits out another thread's notify), until `synced_seq` reaches `seq`.
    ///
    /// A commit made inside this thread's own `on_commit` cannot be sent
    /// until that call returns: it stays queued and this returns `Ok`, as it
    /// did before group commit, and as `sync_flush` returns there.
    fn drain_through(&self, seq: u64) -> Result<(), StoreError> {
        loop {
            {
                let slot = lock_mutex(&self.sync);
                if slot.synced_seq >= seq {
                    return Ok(());
                }
                if slot.notify_in_flight && slot.notify_owner == Some(std::thread::current().id()) {
                    return Ok(());
                }
            }
            self.sync_flush()?;
        }
    }

    fn notify(&self, topic: &str, record: &MqRecord) {
        let mut subs = lock_mutex(&self.subs);
        let Some(senders) = subs.get_mut(topic) else {
            return;
        };
        senders.retain(|sender| sender.send(record.clone()).is_ok());
        if senders.is_empty() {
            subs.remove(topic);
        }
    }
}

/// A commit durable at `seq` whose batch the adapter has not received.
fn undelivered(seq: u64, cause: StoreError) -> StoreError {
    StoreError::Undelivered {
        seq,
        cause: Box::new(cause),
    }
}

/// Cloud offload decision for a search. `Some` is the answer. `None` means search locally.
///
/// `Ok(None)` is the adapter declining. `Err(_)` also falls back to the local
/// index, so a cloud outage degrades a search and does not fail it. To make a
/// cloud error fail the search instead, return it from here (change the return
/// type to `Result<Option<..>, StoreError>` and use `?` at the two call sites).
fn cloud_hits(
    result: Result<Option<Vec<vector::VectorHit>>, StoreError>,
) -> Option<Vec<(String, f64)>> {
    match result {
        Ok(Some(hits)) => Some(hits.into_iter().map(|hit| (hit.id, hit.score)).collect()),
        Ok(None) | Err(_) => None,
    }
}

struct IfAbsent {
    wrote: bool,
    doc: Value,
    store_seq: Option<u64>,
}

fn parse_publish_result(value: &Value) -> Result<(u64, u64), StoreError> {
    let obj = value
        .as_object()
        .ok_or_else(|| backend("publish_mq result is not an object"))?;
    let seq = obj
        .get("seq")
        .and_then(Value::as_u64)
        .ok_or_else(|| backend("publish_mq result is missing seq"))?;
    let store_seq = obj
        .get("store_seq")
        .and_then(Value::as_u64)
        .ok_or_else(|| backend("publish_mq result is missing store_seq"))?;
    Ok((seq, store_seq))
}

fn parse_if_absent(value: &Value) -> Result<(bool, Value, Option<u64>), StoreError> {
    let obj = value
        .as_object()
        .ok_or_else(|| backend("put_doc if_absent result is not an object"))?;
    let wrote = obj
        .get("wrote")
        .and_then(Value::as_bool)
        .ok_or_else(|| backend("put_doc if_absent result is missing wrote"))?;
    let doc = obj
        .get("doc")
        .cloned()
        .ok_or_else(|| backend("put_doc if_absent result is missing doc"))?;
    if !wrote {
        return Ok((false, doc, None));
    }
    let store_seq = obj
        .get("store_seq")
        .and_then(Value::as_u64)
        .ok_or_else(|| backend("put_doc if_absent result is missing store_seq"))?;
    Ok((true, doc, Some(store_seq)))
}

fn mq_publish_ops(topic: &str, key: &str, payload: &Value, mq_seq: u64) -> Vec<Op> {
    vec![
        Op::Put {
            collection: mq_message_collection(topic),
            id: mq_seq.to_string(),
            doc: json!({"seq": mq_seq, "key": key, "payload": payload}),
        },
        Op::Put {
            collection: mq_hw_collection(topic),
            id: MQ_HIGH_WATER_ID.to_string(),
            doc: json!({"seq": mq_seq}),
        },
    ]
}

struct PreparedBatch {
    seq: u64,
    ops: Vec<Op>,
}

/// Every batch staged in one overlay over `state`, the sequence after the last
/// batch, and the ops of each batch as applied. `state` is not changed.
fn prepare_sync_batches(
    state: &DocState,
    batches: &[SyncBatch],
) -> Result<(Overlay, u64, Vec<PreparedBatch>), StoreError> {
    let mut overlay = Overlay::default();
    let mut expect = state.last_seq;
    let mut prepared = Vec::with_capacity(batches.len());
    for batch in batches {
        let next = expect
            .checked_add(1)
            .ok_or_else(|| backend("transaction sequence overflow"))?;
        if batch.seq != next || batch.ops.is_empty() {
            return Err(backend(format!(
                "sync batch sequence {} is not the next commit {next}",
                batch.seq
            )));
        }
        let mut version =
            doc::schema_version_in(|collection, id| overlay.doc(state, collection, id).cloned())?;
        let mut ops = Vec::with_capacity(batch.ops.len());
        for op in &batch.ops {
            if let Some(op) = admit_inbound(op, &mut version)? {
                ops.push(op.clone().into_op());
            }
        }
        // Every op may have been dropped: the sequence still advances, as an
        // empty frame, so the next batch is still the next commit.
        for op in &ops {
            overlay.stage(state, op)?;
        }
        expect = next;
        prepared.push(PreparedBatch {
            seq: batch.seq,
            ops,
        });
    }
    Ok((overlay, expect, prepared))
}

/// Whether an inbound sync op is applied. `version` is the schema version the
/// batch has reached so far, and is raised when `op` raises it.
///
/// An op on an ordinary collection is applied as it is. An op on a reserved
/// collection is the peer's own bookkeeping and is never applied as it is:
/// `("__meta", "schema_version")` put to a version on the ladder above
/// `version` raises it; a version newer than this build is
/// [`StoreError::FormatVersion`], the same refusal `open` gives, so the batch
/// is refused whole; any other reserved op (an older, equal or off-ladder
/// version, another `__meta` id, a delete, another reserved collection) is
/// dropped.
fn admit_inbound<'a>(
    op: &'a SyncOp,
    version: &mut Option<u64>,
) -> Result<Option<&'a SyncOp>, StoreError> {
    let (collection, raise) = match op {
        SyncOp::PutDoc {
            collection,
            id,
            doc,
        } => (
            collection,
            (collection == doc::SCHEMA_COLLECTION && id == doc::SCHEMA_KEY)
                .then(|| doc.as_u64())
                .flatten(),
        ),
        SyncOp::DeleteDoc { collection, .. } => (collection, None),
        _ => return Ok(Some(op)),
    };
    if !doc::is_reserved(collection) {
        return Ok(Some(op));
    }
    let Some(to) = raise else {
        return Ok(None);
    };
    if crate::FORMAT_LADDER.iter().all(|known| to > *known) {
        crate::check_format(Some(to))?;
    }
    if crate::check_format(Some(to)).is_err() || version.is_some_and(|now| to <= now) {
        return Ok(None);
    }
    *version = Some(to);
    Ok(Some(op))
}

/// Disk: alone, not in a group (spec §4.3). The peer fixes every batch's
/// sequence, and the batches publish as one unit (one `sync_data` for all
/// their frames, one slot), checked with the sync-idle test under the commit
/// gate, which keeps them apart from every group. The `bool` is `true` when
/// the published state is due an auto-checkpoint, which the caller runs once
/// the gate is released ([`auto_checkpoint`]).
fn publish_sync_batches(
    inner: &Inner,
    batches: &[SyncBatch],
) -> Result<(u64, Vec<PreparedBatch>, bool), StoreError> {
    match inner {
        Inner::Remote(_) => Err(backend(
            "a remote FerroStore does not publish sync batches locally",
        )),
        Inner::Memory(slot) => {
            let mut state = lock_mutex(slot);
            let (overlay, seq, prepared) = prepare_sync_batches(&state, batches)?;
            publish_overlay(&mut state, overlay, seq);
            Ok((seq, prepared, false))
        }
        Inner::Disk(disk) => {
            let mut write = lock_mutex(&disk.write);
            disk.poison.check()?;
            let locked = disk.lock_current(&mut write)?;
            let file = locked.file();
            let (index, base) = disk.refresh(file, locked.id)?;
            let (overlay, seq, prepared) = prepare_sync_batches(&base.state, batches)?;
            let mut frames = Vec::with_capacity(prepared.len());
            for batch in &prepared {
                frames.push(encode_frame(batch.seq, &batch.ops)?);
            }
            let due = match disk.publish_or_settle(file, index, &base, &frames)? {
                Some(new_sb) => {
                    let done = disk.install_commit(base, new_sb, Change::Overlay(overlay, seq));
                    checkpoint_due(disk, &done)
                }
                None => false,
            };
            Ok((seq, prepared, due))
        }
    }
}

fn hybrid_params(
    doc_collection: &str,
    vector_collection: &str,
    query: &str,
    values: &[f32],
    limit: usize,
    fusion: Fusion,
) -> Result<Value, StoreError> {
    vector::validate_fusion(fusion)?;
    let fusion = match fusion {
        Fusion::Rrf { k } => json!({"method": "rrf", "k": k}),
        Fusion::Weighted { bm25, vector } => {
            json!({
                "method": "weighted",
                "bm25": bm25,
                "vector": vector,
            })
        }
    };
    Ok(json!({
        "doc_collection": doc_collection,
        "vector_collection": vector_collection,
        "query": query,
        "values": vector::values_json(values)?,
        "limit": limit,
        "fusion": fusion,
    }))
}

/// `dir/store.fls` -> `dir/store.fls.sock`.
/// A path with no file name (`/` or `..`) -> that path joined with `ferrostore.sock`.
pub fn default_socket_path_for_store(path: impl AsRef<Path>) -> PathBuf {
    let path = path.as_ref();
    match path.file_name() {
        Some(name) => {
            let mut sock_name = name.to_os_string();
            sock_name.push(".sock");
            path.with_file_name(sock_name)
        }
        None => path.join("ferrostore.sock"),
    }
}

fn refused_connect(err: &io::Error) -> bool {
    matches!(
        err.kind(),
        io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
    )
}

fn file_uri_path(rest: &str) -> String {
    if rest.starts_with('/') {
        return rest.to_string();
    }
    if let Some(path) = rest.strip_prefix("localhost/") {
        return format!("/{path}");
    }
    if rest == "localhost" {
        return "/".to_string();
    }
    rest.to_string()
}

/// Runs alone, not in a group (spec §4.3): it writes a new snapshot and
/// starts an empty WAL, so there is nothing to batch. The write mutex keeps it
/// apart from every group. A compaction the checkpoint makes due runs once
/// its locks are released ([`compact::auto_compact`]).
fn checkpoint_disk(disk: &Disk) -> Result<(), StoreError> {
    let compact_due = {
        let mut write = lock_mutex(&disk.write);
        disk.poison.check()?;
        let locked = disk.lock_current(&mut write)?;
        let (index, base) = disk.refresh(locked.file(), locked.id)?;
        checkpoint_locked(disk, locked.file(), index, &base)?
    };
    if compact_due {
        compact::auto_compact(disk);
    }
    Ok(())
}

/// Spec §4.4: whether the state `base` is due an auto-checkpoint: its WAL is
/// longer than `threshold = max(snapshot_length, AUTO_CHECKPOINT_BYTES)`,
/// and does not end torn (that is left to the next refresh, which reports
/// it). After a failed auto-checkpoint at WAL length `f`
/// ([`Disk::checkpoint_failed_at`]) it is due again only once the WAL is
/// longer than `f + threshold`, so a failure that persists (a full disk, an
/// encode error) costs one O(state) attempt per threshold of WAL bytes, not
/// one per commit. A WAL shorter than `f` was checkpointed by another handle
/// since: the failure no longer holds it back.
fn checkpoint_due(disk: &Disk, base: &DiskCache) -> bool {
    let wal = base.sb.wal_length;
    let threshold = base.sb.snapshot_length.max(disk.auto_checkpoint_floor());
    if base.wal_applied != wal || wal <= threshold {
        return false;
    }
    let failed_at = disk.checkpoint_failed_at.load(Ordering::Acquire);
    failed_at == 0 || wal < failed_at || wal - failed_at > threshold
}

/// Spec §4.4: the auto-checkpoint of a commit whose published state was
/// [`checkpoint_due`]. Called once the commit has released every lock it
/// published under (the commit gate, the write mutex, the `flock`) and, in
/// a group, once every member has its result: the checkpoint delays no
/// other commit's result, only its own caller's return. It takes a lock
/// window of its own, as [`checkpoint_disk`] does (the write mutex, then the
/// `flock`), refreshes, and checkpoints only when the refreshed state is
/// still due: another handle may have checkpointed meanwhile.
///
/// The commit is already durable and acknowledged, so a failed checkpoint
/// does not fail it: the commit's caller gets its own result. A failed
/// `sync_data` poisons the handle, as any failed sync does ([`Poison`]); any
/// other failure leaves the file and the cache as they were, and is tried
/// again once the WAL has grown by another threshold. Only the first failure
/// of a run of failures is reported on stderr; a checkpoint that succeeds
/// ends the run.
///
/// A checkpoint that leaves the file due a compaction (spec §5) is followed
/// by one, in yet another lock window ([`compact::auto_compact`]).
fn auto_checkpoint(disk: &Disk) {
    let (compact_due, warning) = {
        let mut write = lock_mutex(&disk.write);
        if disk.poison.is_set() {
            return;
        }
        let mut wal = None;
        let result = (|| -> Result<bool, StoreError> {
            let locked = disk.lock_current(&mut write)?;
            let (index, base) = disk.refresh(locked.file(), locked.id)?;
            if !checkpoint_due(disk, &base) {
                return Ok(false);
            }
            wal = Some(base.sb.wal_length);
            checkpoint_locked(disk, locked.file(), index, &base)
        })();
        match result {
            Ok(due) => (due, None),
            // A refresh that failed before the WAL was read: the length the
            // commit left is not known here, and the next due commit tries
            // again.
            Err(err) => match wal {
                None => return,
                Some(wal) => {
                    let first = disk.checkpoint_failed_at.load(Ordering::Acquire) == 0;
                    disk.checkpoint_failed_at
                        .store(wal.max(1), Ordering::Release);
                    (false, first.then_some(err))
                }
            },
        }
    };
    if let Some(err) = warning {
        warn_auto_checkpoint(disk, &err);
    }
    if compact_due {
        compact::auto_compact(disk);
    }
}

#[cfg(not(test))]
fn warn_auto_checkpoint(disk: &Disk, err: &StoreError) {
    eprintln!(
        "warning: FerroStore auto-checkpoint of {} failed, and the commit before it is durable \
         (tried again once the WAL grows by another threshold): {err}",
        disk.poison.path
    );
}

#[cfg(test)]
thread_local! {
    /// Auto-checkpoint warnings this thread would have printed.
    static AUTO_CHECKPOINT_WARNINGS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn warn_auto_checkpoint(_disk: &Disk, _err: &StoreError) {
    AUTO_CHECKPOINT_WARNINGS.with(|count| count.set(count.get() + 1));
}

/// The auto-checkpoint warnings this thread would have printed.
#[cfg(test)]
pub(super) fn auto_checkpoint_warnings() -> usize {
    AUTO_CHECKPOINT_WARNINGS.with(std::cell::Cell::get)
}

/// Write a snapshot of `base` and publish it into the slot after `index`.
/// The caller holds the write mutex and the `flock`, and `base` is the
/// state the slot `index` names. Success clears an auto-checkpoint failure
/// ([`Disk::checkpoint_failed_at`]). `true` when the file it leaves is due a
/// compaction (spec §5, [`compact::compact_due`]), which the caller runs
/// once it has released its locks.
fn checkpoint_locked(
    disk: &Disk,
    file: &dyn ContainerIo,
    index: usize,
    base: &DiskCache,
) -> Result<bool, StoreError> {
    disk.poison.check()?;
    let new_sb = write_checkpoint(disk, file, index, base)
        .inspect_err(|_| disk.settle_failed_publish(base))?;
    disk.install_checkpoint(base, new_sb);
    disk.checkpoint_failed_at.store(0, Ordering::Release);
    // The new snapshot was written at the end of the file: the file now ends
    // where it does.
    let file_len = new_sb
        .snapshot_offset
        .saturating_add(new_sb.snapshot_length);
    Ok(compact::compact_due(disk, file_len, &new_sb))
}

/// [`checkpoint_locked`]'s writes and syncs: the superblock it published.
fn write_checkpoint(
    disk: &Disk,
    file: &dyn ContainerIo,
    index: usize,
    base: &DiskCache,
) -> Result<Superblock, StoreError> {
    let (sb, state) = (base.sb, &*base.state);
    let blob = encode_state_snapshot(state)?;
    let blob_len = u64::try_from(blob.len()).map_err(|_| backend("snapshot exceeds u64"))?;
    let offset = allocate_extent(file, &sb, blob_len)?;
    file.write_at(&blob, offset).map_err(io_err)?;
    // Data and the file size: `fdatasync` (spec §4.3).
    disk.poison.sync_data(file)?;
    let new_sb = Superblock {
        epoch: sb
            .epoch
            .checked_add(1)
            .ok_or_else(|| backend("epoch overflow"))?,
        snapshot_offset: offset,
        snapshot_length: blob_len,
        wal_offset: offset
            .checked_add(blob_len)
            .ok_or_else(|| backend("wal offset overflow"))?,
        wal_length: 0,
        snapshot_seq: state.last_seq,
    };
    write_slot(file, index ^ 1, &new_sb)?;
    disk.poison.sync_data(file)?;
    Ok(new_sb)
}

/// One snapshot blob of `state`, in the oldest snapshot version that holds
/// it: `1` for documents only, `2` with column tables, `3` with any vector
/// collection. A checkpoint and a compaction (spec §5) write the same blob.
fn encode_state_snapshot(state: &DocState) -> Result<Vec<u8>, StoreError> {
    if !state.vectors.is_empty() {
        encode_snapshot_v3(state.last_seq, &state.docs, &state.tables, &state.vectors)
    } else if state.tables.is_empty() {
        encode_snapshot(state.last_seq, &state.docs)
    } else {
        encode_snapshot_v2(state.last_seq, &state.docs, &state.tables)
    }
}

fn allocate_extent(file: &dyn ContainerIo, sb: &Superblock, len: u64) -> Result<u64, StoreError> {
    let file_len = file.len().map_err(io_err)?;
    let live_end = sb
        .snapshot_offset
        .saturating_add(sb.snapshot_length)
        .max(sb.wal_offset.saturating_add(sb.wal_length))
        .max(HEADER_BYTES);
    let offset = file_len.max(live_end);
    if offset.checked_add(len).is_none() {
        return Err(backend("extent offset overflow"));
    }
    Ok(offset)
}

/// The documents a new store starts with: its schema version, the newest on
/// the ladder. They sit in the snapshot at sequence 0, so a new store still
/// has `last_seq` 0 and no WAL frame.
fn new_store_docs() -> Docs {
    let mut docs = Docs::default();
    docs.insert(
        doc::SCHEMA_COLLECTION.to_string(),
        doc::SCHEMA_KEY.to_string(),
        Arc::new(json!(crate::FORMAT_WITH_WORKFLOW)),
    );
    docs
}

fn init_empty(file: &dyn ContainerIo) -> Result<(), StoreError> {
    let blob = encode_snapshot(0, &new_store_docs())?;
    let blob_len = u64::try_from(blob.len()).map_err(|_| backend("snapshot exceeds u64"))?;
    file.set_len(HEADER_BYTES).map_err(io_err)?;
    file.write_at(&blob, HEADER_BYTES).map_err(io_err)?;
    // First use of a new file, after a `set_len`: the whole inode, `fsync`.
    file.sync_all().map_err(io_err)?;
    let sb = Superblock {
        epoch: 1,
        snapshot_offset: HEADER_BYTES,
        snapshot_length: blob_len,
        wal_offset: HEADER_BYTES + blob_len,
        wal_length: 0,
        snapshot_seq: 0,
    };
    write_slot(file, 0, &sb)?;
    file.sync_all().map_err(io_err)?;
    Ok(())
}

/// Make the store file 0600 when it is missing, and its parent 0700 when that
/// is missing too. `true` when this call created the file. An error from the
/// file create is left to the open that follows, which reports it in its own
/// words (the same rule as `RedbStore::open`).
fn create_store_file(path: &Path) -> Result<bool, StoreError> {
    let parent = parent_dir(path);
    if !parent.exists() {
        crate::private::create_private_dir_all(&parent)
            .map_err(|err| backend(format!("create directory {}: {err}", parent.display())))?;
        sync_dir(&parent_dir(&parent))?;
    }
    Ok(crate::private::create_private_file(path).unwrap_or(false))
}

/// The directory that holds `path`: its parent, or `.` for a bare name.
fn parent_dir(path: &Path) -> PathBuf {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

/// `fsync` a directory so an entry made in it is durable. A file system that
/// cannot sync a directory (`EINVAL`) has nothing more to give.
pub(crate) fn sync_dir(dir: &Path) -> Result<(), StoreError> {
    let handle =
        File::open(dir).map_err(|err| backend(format!("open {}: {err}", dir.display())))?;
    match handle.sync_all() {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::InvalidInput => Ok(()),
        Err(err) => Err(backend(format!("sync {}: {err}", dir.display()))),
    }
}

/// Name `path` in a container-format error that was raised without one.
fn name_store(err: StoreError, path: &Path) -> StoreError {
    match err {
        StoreError::ContainerFormat {
            store,
            found,
            engine,
            unknown_incompat,
            newest,
        } if store.is_empty() => StoreError::ContainerFormat {
            store: path.display().to_string(),
            found,
            engine,
            unknown_incompat,
            newest,
        },
        other => other,
    }
}

// No valid slot yet. Publish slot 0 only for a file that can hold nothing but
// an `init_empty` cut short: every header byte present is zero, and the bytes
// after the header are none, or a prefix (up to all) of the snapshot
// `init_empty` writes. Any other file is not rewritten: it could hold real
// data, and the load in `open` reports it. `true` when this call ran
// `init_empty`.
fn ensure_superblock(file: &dyn ContainerIo) -> Result<bool, StoreError> {
    if parsed_superblock(file)?.is_some() {
        return Ok(false);
    }
    let _locked = FlockGuard::acquire(file)?;
    if parsed_superblock(file)?.is_some() {
        return Ok(false);
    }
    let len = file.len().map_err(io_err)?;
    if len == 0 {
        init_empty(file)?;
        return Ok(true);
    }
    let header_len = len.min(HEADER_BYTES);
    let n = usize::try_from(header_len).map_err(|_| backend("header length exceeds usize"))?;
    let mut buf = vec![0u8; n];
    file.read_at(&mut buf, 0).map_err(io_err)?;
    if buf.iter().any(|byte| *byte != 0) {
        return Ok(false);
    }
    if len > HEADER_BYTES {
        let blob = encode_snapshot(0, &new_store_docs())?;
        let tail_len = len - HEADER_BYTES;
        let Ok(tail) = usize::try_from(tail_len) else {
            return Ok(false);
        };
        if tail > blob.len() {
            return Ok(false);
        }
        let mut held = vec![0u8; tail];
        file.read_at(&mut held, HEADER_BYTES).map_err(io_err)?;
        if held != blob[..tail] {
            return Ok(false);
        }
    }
    init_empty(file)?;
    Ok(true)
}

/// The live slot and the state it names, decoded in full. The engine goes
/// through [`Disk::refresh`]; tests use this to read a file afresh.
#[cfg(test)]
fn read_consistent(file: &dyn ContainerIo) -> Result<(usize, Superblock, DocState), StoreError> {
    let (index, sb) = latest_superblock(file)?;
    Ok((index, sb, load_state(file, &sb)?.0))
}

/// The state `sb` names: its snapshot, then its whole WAL replayed. Also the
/// WAL bytes the replayed frames cover (see [`DiskCache::wal_applied`]).
fn load_state(file: &dyn ContainerIo, sb: &Superblock) -> Result<(DocState, u64), StoreError> {
    let mut state = if sb.snapshot_length == 0 {
        DocState::at_seq(sb.snapshot_seq)
    } else {
        let bytes = read_range(file, sb.snapshot_offset, sb.snapshot_length)?;
        let snap = decode_snapshot(&bytes)?;
        if snap.seq != sb.snapshot_seq {
            return Err(corrupt("snapshot sequence does not match the superblock"));
        }
        DocState {
            docs: snap.docs,
            tables: snap.tables,
            vectors: snap.vectors,
            last_seq: snap.seq,
        }
    };
    let wal = read_range(file, sb.wal_offset, sb.wal_length)?;
    let (frames, consumed) = decode_wal_prefix(&wal)?;
    apply_frames(&mut state, sb.snapshot_seq, &frames)?;
    Ok((state, consumed))
}

fn parsed_superblock(file: &dyn ContainerIo) -> Result<Option<(usize, Superblock)>, StoreError> {
    let file_len = file.len().map_err(io_err)?;
    let mut best: Option<(usize, Superblock)> = None;
    for index in 0..2 {
        let offset = slot_offset(index);
        if file_len < offset + SLOT_LEN as u64 {
            continue;
        }
        let mut buf = [0u8; SLOT_LEN];
        file.read_at(&mut buf, offset).map_err(io_err)?;
        match parse_superblock(&buf) {
            SlotRead::Valid(sb) => match &best {
                Some((_, current)) if sb.epoch < current.epoch => {}
                Some((_, current)) if sb.epoch == current.epoch => {
                    return Err(corrupt("two valid superblocks share an epoch"));
                }
                _ => best = Some((index, sb)),
            },
            SlotRead::Invalid => {}
            // Either slot from a newer build stops the read: falling back to
            // the other slot would read, and then overwrite, a stale state.
            SlotRead::Newer {
                version,
                engine,
                incompat,
            } => {
                return Err(StoreError::ContainerFormat {
                    store: String::new(),
                    found: version,
                    engine,
                    unknown_incompat: incompat & !KNOWN_INCOMPAT,
                    newest: layout::FORMAT_VERSION,
                });
            }
        }
    }
    Ok(best)
}

fn latest_superblock(file: &dyn ContainerIo) -> Result<(usize, Superblock), StoreError> {
    parsed_superblock(file)?.ok_or_else(|| backend("no valid FerroStore superblock"))
}

/// Bytes in the live WAL of the store file at `path`, from its newest
/// superblock: `0` right after a checkpoint. For tests in other modules that
/// must tell a folded log from an unfolded one.
#[cfg(test)]
pub(crate) fn wal_length_on_disk(path: &Path) -> u64 {
    let file = RealFs.open_ro(path).unwrap();
    latest_superblock(&*file).unwrap().1.wal_length
}

fn write_slot(file: &dyn ContainerIo, index: usize, sb: &Superblock) -> Result<(), StoreError> {
    if index > 1 {
        return Err(backend("superblock slot out of range"));
    }
    let bytes = encode_superblock(sb);
    file.write_at(&bytes, slot_offset(index)).map_err(io_err)?;
    Ok(())
}

fn read_range(file: &dyn ContainerIo, offset: u64, length: u64) -> Result<Vec<u8>, StoreError> {
    if length == 0 {
        return Ok(Vec::new());
    }
    let len =
        usize::try_from(length).map_err(|_| corrupt("extent length does not fit in memory"))?;
    let file_len = file.len().map_err(io_err)?;
    let end = offset
        .checked_add(length)
        .ok_or_else(|| corrupt("extent range overflows"))?;
    if end > file_len {
        return Err(corrupt("extent extends past the end of the file"));
    }
    let mut buf = vec![0u8; len];
    file.read_at(&mut buf, offset).map_err(io_err)?;
    Ok(buf)
}

fn slot_offset(index: usize) -> u64 {
    index as u64 * SLOT_LEN as u64
}

fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn wait_mutex<'a, T>(condvar: &Condvar, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    match condvar.wait(guard) {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn io_err(err: io::Error) -> StoreError {
    backend(err)
}
