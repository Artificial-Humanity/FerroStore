//! Compaction by sibling inode (spec §5).
//!
//! A checkpoint appends its snapshot at the end of the file (no extent is
//! reused while a reader may still read it), so every checkpoint leaves the
//! snapshot and the WAL before it dead. A compaction writes the current
//! state into a new sibling file and renames it over the store:
//!
//! 1. holding the write mutex and the `flock` of the file the path names now
//!    ([`Disk::lock_current`]), refresh;
//! 2. create `<name>.fls.compact-<pid>-<n>-<nonce>` with `O_EXCL`, mode 0600,
//!    and take its own `flock`;
//! 3. write a v2 header with one slot, one snapshot of the state and an
//!    empty WAL; the slot's epoch is the old file's current epoch + 1, and
//!    `snapshot_seq` is the state's `last_seq`;
//! 4. `fsync` it (`sync_all`: the file was just created);
//! 5. rename it over `<name>.fls`;
//! 6. `fsync` the directory;
//! 7. install the state in this handle's cache and move this handle's
//!    descriptions to the new file;
//! 8. only then release the new file's lock and the old file's lock.
//!
//! A failure before the rename removes the sibling and leaves the old file
//! as it was. Every other handle, in this process or another, finds the swap
//! the next time it takes the lock or refreshes a read: it compares
//! `stat(path)` with its description and reopens the path on a difference.
//!
//! The new file's lock is held until the directory sync is done, so a
//! writer that opened the new file after the rename cannot commit into it
//! before its name is durable: a crash before the directory sync can still
//! bring the old name back, and a commit already acknowledged in the new
//! file would then be lost. A failed directory sync poisons this handle and
//! releases that lock with the name still not durable; every other handle
//! then reaches the new file through a description that is not
//! `name_durable`, and syncs the directory itself before its first write
//! there ([`Disk::lock_current`]). So no handle ever commits into a file
//! whose name it has not itself made durable.

use super::layout::{HEADER_BYTES, Superblock, backend};
use super::{
    Descriptor, Disk, DiskCache, HeldFlock, encode_state_snapshot, install_newer, io_err,
    lock_mutex, parent_dir, write_slot,
};
use fl_core::StoreError;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// The dead bytes an auto-compaction never fires at or below: a checkpoint
/// that leaves more dead bytes than live ones (the file more than half
/// dead) *and* more than this is followed by a compaction (spec §5). The
/// half keeps the work linear in the bytes written, as the auto-checkpoint's
/// threshold does: a compaction costs O(live bytes) and runs only once at
/// least as many dead bytes have piled up. The floor keeps a small store
/// from being rewritten every few checkpoints; below it, the space is not
/// worth a new file. [`super::FerroStore::compact`] runs one on demand.
pub const AUTO_COMPACT_BYTES: u64 = 64 << 20;

/// What a compaction's sibling name puts between the store's file name and
/// its numbers.
pub(super) const COMPACT_MARK: &str = ".compact-";

/// The mode a compaction's sibling is created with, from its first byte.
const SIBLING_MODE: u32 = 0o600;

/// The size accounting of one store file ([`super::FerroStore::stats`]),
/// for `inspect`-style use. Read without a lock: the counts are of the
/// state the read saw, and another handle may have published since.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoreStats {
    /// The file's length.
    pub file_bytes: u64,
    /// The bytes the live superblock names: the 8 KiB header (both slots),
    /// the live snapshot and the live WAL.
    pub live_bytes: u64,
    /// `file_bytes - live_bytes`: the snapshots and WALs earlier
    /// checkpoints left behind, which only a compaction gives back.
    pub dead_bytes: u64,
    /// The live superblock's epoch. It rises with every publish, a
    /// compaction's included: the new file starts one past the old one.
    pub epoch: u64,
    /// The last committed sequence.
    pub last_seq: u64,
}

/// The bytes `sb` keeps live: the header, its snapshot and its WAL.
fn live_bytes(sb: &Superblock) -> u64 {
    HEADER_BYTES
        .saturating_add(sb.snapshot_length)
        .saturating_add(sb.wal_length)
}

/// The bytes of a `file_len`-byte file that `sb` does not name.
fn dead_bytes(file_len: u64, sb: &Superblock) -> u64 {
    file_len.saturating_sub(live_bytes(sb))
}

/// Spec §5: whether a file of `file_len` bytes whose live superblock is `sb`
/// is due an auto-compaction: its dead bytes are more than
/// `threshold = max(live bytes, floor)` (more than half the file, and more
/// than [`AUTO_COMPACT_BYTES`]). After a failed auto-compaction at `f` dead
/// bytes ([`Disk::compact_failed_at`]) it is due again only once the dead
/// bytes pass `f + threshold`, so a failure that persists costs one attempt
/// per threshold of dead bytes, not one per checkpoint. Fewer dead bytes
/// than `f`: another handle compacted since, and the failure no longer
/// holds it back.
pub(super) fn compact_due(disk: &Disk, file_len: u64, sb: &Superblock) -> bool {
    let dead = dead_bytes(file_len, sb);
    let threshold = live_bytes(sb).max(disk.auto_compact_floor());
    if dead <= threshold {
        return false;
    }
    let failed_at = disk.compact_failed_at.load(Ordering::Acquire);
    failed_at == 0 || dead < failed_at || dead - failed_at > threshold
}

/// The accounting of a `file_bytes`-byte file whose live superblock and
/// state `cache` holds. The one place a [`StoreStats`] is built, so a handle's
/// [`super::FerroStore::stats`] and the read-only [`super::FerroStore::stats_at`]
/// cannot count differently.
pub(super) fn store_stats(file_bytes: u64, cache: &DiskCache) -> StoreStats {
    StoreStats {
        file_bytes,
        live_bytes: live_bytes(&cache.sb),
        dead_bytes: dead_bytes(file_bytes, &cache.sb),
        epoch: cache.sb.epoch,
        last_seq: cache.state.last_seq,
    }
}

/// [`super::FerroStore::stats`] of a disk handle.
pub(super) fn stats(disk: &Disk) -> Result<StoreStats, StoreError> {
    let (cache, read) = disk.refresh_read_on()?;
    let file_bytes = read.file.len().map_err(io_err)?;
    Ok(store_stats(file_bytes, &cache))
}

/// [`super::FerroStore::compact`] of a disk handle: compact now, due or not.
/// Runs alone, as a checkpoint does: the write mutex keeps it apart from
/// every group.
pub(super) fn compact_disk(disk: &Disk) -> Result<(), StoreError> {
    disk.fs
        .sweep_stale_temps(&disk.path, COMPACT_MARK, "compaction");
    let mut write = lock_mutex(&disk.write);
    disk.poison.check()?;
    compact_window(disk, &mut write, None)
}

/// Spec §5: the auto-compaction a checkpoint made due ([`compact_due`]).
/// Called once the checkpoint has released every lock it held, as the
/// auto-checkpoint is (spec §4.4): it takes a lock window of its own,
/// refreshes, and compacts only when the refreshed file is still due
/// (another handle may have compacted meanwhile).
///
/// The checkpoint (and the commit before it) is already durable, so a
/// failed compaction does not fail it. A failed `fsync` poisons the handle,
/// as any failed sync does; any other failure leaves the old file as the
/// store, removes the sibling, and is tried again once the dead bytes have
/// grown by another threshold. Only the first failure of a run is reported
/// on stderr; a compaction that succeeds ends the run.
pub(super) fn auto_compact(disk: &Disk) {
    disk.fs
        .sweep_stale_temps(&disk.path, COMPACT_MARK, "compaction");
    let warning = {
        let mut write = lock_mutex(&disk.write);
        if disk.poison.is_set() {
            return;
        }
        let mut dead = None;
        let Err(err) = compact_window(disk, &mut write, Some(&mut dead)) else {
            return;
        };
        // A failure before the file was measured: the next due checkpoint
        // tries again.
        let Some(dead) = dead else {
            return;
        };
        let first = disk.compact_failed_at.load(Ordering::Acquire) == 0;
        disk.compact_failed_at.store(dead.max(1), Ordering::Release);
        first.then_some(err)
    };
    if let Some(err) = warning {
        warn_auto_compact(disk, &err);
    }
}

/// One compaction window, with the write mutex (whose description `write`
/// is) held. `due`: an auto-compaction, which compacts only when the file
/// the refresh reads is still [`compact_due`], and records the dead bytes it
/// measured there.
fn compact_window(
    disk: &Disk,
    write: &mut Descriptor,
    due: Option<&mut Option<u64>>,
) -> Result<(), StoreError> {
    let old = disk.lock_current(write)?;
    let (_, base) = disk.refresh(old.file(), old.id)?;
    let file_len = old.file().len().map_err(io_err)?;
    if let Some(dead) = due {
        if !compact_due(disk, file_len, &base.sb) {
            return Ok(());
        }
        *dead = Some(dead_bytes(file_len, &base.sb));
    }
    let sibling = write_sibling(disk, &base)?;
    publish_sibling(disk, write, sibling, &base)?;
    disk.compact_failed_at.store(0, Ordering::Release);
    // The old file's lock goes last: after the rename and the directory
    // sync. A writer that waited on it then finds the path swapped.
    drop(old);
    Ok(())
}

/// The new file, written and synced, not yet named as the store.
struct Sibling {
    path: PathBuf,
    write: Descriptor,
    read: Descriptor,
    /// The new file's own lock, held until it is published.
    flock: HeldFlock,
    sb: Superblock,
}

/// Steps 2 to 4: create the sibling, write the header, one snapshot of
/// `base` and an empty WAL, and `fsync` it. On any failure the sibling is
/// removed; the store is not touched.
fn write_sibling(disk: &Disk, base: &DiskCache) -> Result<Sibling, StoreError> {
    let path = disk.fs.temp_path(&disk.path, COMPACT_MARK)?;
    let created = disk
        .fs
        .create_new(&path, SIBLING_MODE)
        .map_err(|err| backend(format!("create {}: {err}", path.display())))?;
    let built = (|| -> Result<Sibling, StoreError> {
        let write = Descriptor::new(created, &path)?;
        let flock = HeldFlock::acquire(&write)?;
        let opened = disk
            .fs
            .open_ro(&path)
            .map_err(|err| backend(format!("open {}: {err}", path.display())))?;
        let read = Descriptor::new(opened, &path)?;
        if read.id != write.id {
            return Err(backend(format!(
                "{} is no longer the file this compaction created",
                path.display()
            )));
        }
        let blob = encode_state_snapshot(&base.state)?;
        let blob_len = u64::try_from(blob.len()).map_err(|_| backend("snapshot exceeds u64"))?;
        let sb = Superblock {
            epoch: base
                .sb
                .epoch
                .checked_add(1)
                .ok_or_else(|| backend("epoch overflow"))?,
            snapshot_offset: HEADER_BYTES,
            snapshot_length: blob_len,
            wal_offset: HEADER_BYTES
                .checked_add(blob_len)
                .ok_or_else(|| backend("wal offset overflow"))?,
            wal_length: 0,
            snapshot_seq: base.state.last_seq,
        };
        // Slot 1 stays zeros: the gap before the snapshot reads as zeros.
        write.file.write_at(&blob, HEADER_BYTES).map_err(io_err)?;
        write_slot(&*write.file, 0, &sb)?;
        // A file that was just created: `fsync`, not `fdatasync` (spec
        // §4.3). A failure poisons the handle, as every failed sync does.
        disk.poison.sync_all(&*write.file)?;
        Ok(Sibling {
            path: path.clone(),
            write,
            read,
            flock,
            sb,
        })
    })();
    if built.is_err() {
        let _ = disk.fs.unlink(&path);
    }
    built
}

/// Steps 5 to 7: rename the sibling over the store, `fsync` the directory,
/// and move this handle to the new file. The caller still holds the old
/// file's lock. A failed rename removes the sibling. A failed directory
/// sync poisons the handle: the rename stands in the running system, and
/// its name may not be durable. The next writer of any other handle syncs
/// the directory before it writes ([`Disk::lock_current`]).
fn publish_sibling(
    disk: &Disk,
    write: &mut Descriptor,
    sibling: Sibling,
    base: &DiskCache,
) -> Result<(), StoreError> {
    let Sibling {
        path,
        write: new_write,
        read: new_read,
        flock,
        sb,
    } = sibling;
    if let Err(err) = disk.fs.rename(&path, &disk.path) {
        let _ = disk.fs.unlink(&path);
        return Err(backend(format!(
            "rename {} over {}: {err}",
            path.display(),
            disk.path.display()
        )));
    }
    // From here the path names the new file, and the old one is never
    // published again: every writer checks the path under the old file's
    // lock, which this call still holds.
    if let Err(err) = disk.fs.sync_dir(&parent_dir(&disk.path)) {
        disk.poison.mark();
        return Err(err);
    }
    let entry = DiskCache {
        file_id: new_write.id,
        sb,
        state: Arc::clone(&base.state),
        wal_applied: 0,
    };
    // This handle made the new name durable: its writes need no other sync.
    *write = Descriptor {
        name_durable: true,
        ..new_write
    };
    *lock_mutex(&disk.read) = new_read;
    // The path names the new file, and no writer moves it while the old
    // file's lock is held.
    let current = entry.file_id;
    install_newer(&mut lock_mutex(&disk.cache), Arc::new(entry), current);
    drop(flock);
    Ok(())
}

#[cfg(not(test))]
fn warn_auto_compact(disk: &Disk, err: &StoreError) {
    eprintln!(
        "warning: FerroStore auto-compaction of {} failed, and the store is as it was \
         (tried again once the dead bytes grow by another threshold): {err}",
        disk.poison.path
    );
}

#[cfg(test)]
thread_local! {
    /// Auto-compaction warnings this thread would have printed.
    static AUTO_COMPACT_WARNINGS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn warn_auto_compact(_disk: &Disk, _err: &StoreError) {
    AUTO_COMPACT_WARNINGS.with(|count| count.set(count.get() + 1));
}

/// The auto-compaction warnings this thread would have printed.
#[cfg(test)]
pub(super) fn auto_compact_warnings() -> usize {
    AUTO_COMPACT_WARNINGS.with(std::cell::Cell::get)
}
