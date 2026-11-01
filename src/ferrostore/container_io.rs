//! The container I/O seam (spec §3.5).
//!
//! Every byte of `.fls` container I/O goes through [`ContainerIo`] (one open
//! file) and [`ContainerFs`] (the names in a directory). Production uses
//! [`File`] and [`RealFs`], which make the same system calls in the same
//! order as direct `std::fs` calls. The crash harness (`crash_sim`) puts a
//! simulated disk behind the same traits, so it can cut power after any one
//! operation.
//!
//! A new kind of container operation is added here first: a method on the
//! trait, the `File` or `RealFs` body, and the simulated body and its fault
//! model in `crash_sim`.

use fl_core::StoreError;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// One open container file.
pub(super) trait ContainerIo: Send + Sync {
    /// Write all of `buf` at `offset` (`pwrite`, repeated until done).
    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<()>;
    /// Fill all of `buf` from `offset` (`pread`, repeated until done).
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()>;
    /// `fdatasync`: the bytes written and the file size. Frames, snapshots
    /// and slot writes use it (spec §4.3).
    fn sync_data(&self) -> io::Result<()>;
    /// `fsync`: also the rest of the inode's metadata. First-use init, which
    /// sets the length of a new file, and a compaction's new sibling file
    /// (spec §5), which was just created, use it.
    fn sync_all(&self) -> io::Result<()>;
    /// The file's length (`fstat`).
    fn len(&self) -> io::Result<u64>;
    /// `ftruncate`.
    fn set_len(&self, len: u64) -> io::Result<()>;
    /// `(dev, ino)` of the open file (`fstat`): which file this description
    /// reads, so a cache filled through it is not trusted for another file.
    fn file_id(&self) -> io::Result<(u64, u64)>;
    /// `flock(LOCK_EX)`: wait until this description holds the file's
    /// exclusive advisory lock.
    fn lock_exclusive(&self) -> io::Result<()>;
    /// `flock(LOCK_UN)`. Releasing cannot usefully fail: a lock this
    /// description does not hold is not held.
    fn unlock(&self);
}

const LOCK_EX: i32 = 2;
const LOCK_UN: i32 = 8;

// libc `flock`. No extra crate: the process already links libc.
unsafe extern "C" {
    fn flock(fd: i32, op: i32) -> i32;
}

impl ContainerIo for File {
    fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<()> {
        FileExt::write_all_at(self, buf, offset)
    }

    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        FileExt::read_exact_at(self, buf, offset)
    }

    fn sync_data(&self) -> io::Result<()> {
        File::sync_data(self)
    }

    fn sync_all(&self) -> io::Result<()> {
        File::sync_all(self)
    }

    fn len(&self) -> io::Result<u64> {
        Ok(self.metadata()?.len())
    }

    fn set_len(&self, len: u64) -> io::Result<()> {
        File::set_len(self, len)
    }

    fn file_id(&self) -> io::Result<(u64, u64)> {
        let meta = self.metadata()?;
        Ok((meta.dev(), meta.ino()))
    }

    fn lock_exclusive(&self) -> io::Result<()> {
        let fd: RawFd = self.as_raw_fd();
        // SAFETY: `fd` is this open file's descriptor. `LOCK_EX` waits until
        // this description holds the advisory lock, or returns -1.
        if unsafe { flock(fd, LOCK_EX) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    fn unlock(&self) {
        // SAFETY: the descriptor is open for the life of `self`. `LOCK_UN`
        // releases the advisory lock held by this description.
        unsafe {
            flock(self.as_raw_fd(), LOCK_UN);
        }
    }
}

/// The directory side: making, opening, naming and syncing container files.
pub(super) trait ContainerFs: Send + Sync {
    /// The same file system as a shared handle, which a [`super::Disk`]
    /// keeps: compaction (spec §5) reopens its path after the file under it
    /// is swapped.
    fn shared(&self) -> Arc<dyn ContainerFs>;
    /// Make the store file 0600 when it is missing (and its parent 0700 when
    /// that is missing too). `true` when this call created the file.
    fn create_store_file(&self, path: &Path) -> Result<bool, StoreError>;
    /// Create a new file read-write with `mode`, refusing a name that exists
    /// (`O_CREAT | O_EXCL`). A compaction's sibling (spec §5).
    fn create_new(&self, path: &Path, mode: u32) -> io::Result<Box<dyn ContainerIo>>;
    /// Open an existing file read-write.
    fn open_rw(&self, path: &Path) -> io::Result<Box<dyn ContainerIo>>;
    /// Open an existing file `O_RDONLY`.
    fn open_ro(&self, path: &Path) -> io::Result<Box<dyn ContainerIo>>;
    /// `(dev, ino)` of the file `path` names now (`stat`, following a
    /// symlink as `open` does). Compared with [`ContainerIo::file_id`] of a
    /// description, it tells whether a compaction swapped the file under
    /// the path (spec §5).
    fn stat_id(&self, path: &Path) -> io::Result<(u64, u64)>;
    /// `fsync` a directory, so the entries made in it are durable.
    fn sync_dir(&self, dir: &Path) -> Result<(), StoreError>;
    /// `rename(2)`: replaces `to`. Compaction (spec §5) renames its sibling
    /// over the store.
    fn rename(&self, from: &Path, to: &Path) -> io::Result<()>;
    /// `link(2)`: a second name for `from`'s file. Refuses an existing `to`.
    #[cfg_attr(not(test), allow(dead_code, reason = "only the crash harness links"))]
    fn link(&self, from: &Path, to: &Path) -> io::Result<()>;
    /// `unlink(2)`.
    fn unlink(&self, path: &Path) -> io::Result<()>;
    /// The real path of the existing file `path` names (`realpath`: every
    /// symlink and `.`/`..` resolved). A store is opened through it, so a
    /// compaction's sibling, rename and directory sync (spec §5) are made
    /// beside the file itself, not beside a link to it.
    fn resolve(&self, path: &Path) -> Result<PathBuf, StoreError>;
    /// A sibling name for `target` that no other call uses:
    /// `<target><mark><pid>-<n>-<nonce>` (spec §5). `target` is a store's
    /// resolved path ([`Self::resolve`]).
    fn temp_path(&self, target: &Path, mark: &str) -> Result<PathBuf, StoreError>;
    /// Remove the stale temps [`Self::temp_path`] made for `target` with
    /// `mark` that a crash left behind. Never an error: see
    /// `verify::sweep_stale_temps`.
    fn sweep_stale_temps(&self, target: &Path, mark: &str, what: &str);
    /// Report a store file whose mode is looser than 0600.
    fn warn_if_loose(&self, path: &Path);
}

/// The real file system.
pub(super) struct RealFs;

impl ContainerFs for RealFs {
    fn shared(&self) -> Arc<dyn ContainerFs> {
        Arc::new(RealFs)
    }

    fn create_new(&self, path: &Path, mode: u32) -> io::Result<Box<dyn ContainerIo>> {
        // `create_new` is `O_EXCL`: a taken name fails and is not touched.
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(mode)
            .open(path)?;
        Ok(Box::new(file))
    }

    fn stat_id(&self, path: &Path) -> io::Result<(u64, u64)> {
        let meta = std::fs::metadata(path)?;
        Ok((meta.dev(), meta.ino()))
    }

    fn resolve(&self, path: &Path) -> Result<PathBuf, StoreError> {
        std::fs::canonicalize(path)
            .map_err(|err| super::layout::backend(format!("resolve {}: {err}", path.display())))
    }

    fn temp_path(&self, target: &Path, mark: &str) -> Result<PathBuf, StoreError> {
        super::verify::unique_temp_path(target, mark)
    }

    fn sweep_stale_temps(&self, target: &Path, mark: &str, what: &str) {
        super::verify::sweep_stale_temps_marked(target, mark, what);
    }

    fn create_store_file(&self, path: &Path) -> Result<bool, StoreError> {
        super::create_store_file(path)
    }

    fn open_rw(&self, path: &Path) -> io::Result<Box<dyn ContainerIo>> {
        // Not `create(true)`: a file this open makes must come from
        // `create_private_file`, so it is never made with the umask's mode.
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        Ok(Box::new(file))
    }

    fn open_ro(&self, path: &Path) -> io::Result<Box<dyn ContainerIo>> {
        // `OpenOptions::read` without write is `O_RDONLY`.
        let file = OpenOptions::new().read(true).open(path)?;
        Ok(Box::new(file))
    }

    fn sync_dir(&self, dir: &Path) -> Result<(), StoreError> {
        super::sync_dir(dir)
    }

    fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        std::fs::rename(from, to)
    }

    fn link(&self, from: &Path, to: &Path) -> io::Result<()> {
        std::fs::hard_link(from, to)
    }

    fn unlink(&self, path: &Path) -> io::Result<()> {
        std::fs::remove_file(path)
    }

    fn warn_if_loose(&self, path: &Path) {
        crate::private::warn_if_loose(path);
    }
}
