//! Group commit (spec §4.3).
//!
//! Writers on one [`Disk`] queue up. The writer that finds no leader leads:
//! it takes the commit gate, the write mutex and the `flock`, refreshes, and
//! then takes what is queued at that moment, itself first, as its group (at
//! most [`GROUP_MEMBERS`]). Nobody waits to batch: a writer that comes later
//! queues for the next group.
//!
//! Each member's body then runs in turn, on the member's own thread, against
//! the group's running state, so it sees every earlier member's committed
//! changes. A body that fails, or that panics, drops only its own overlay. A
//! committed member takes the next sequence number, so the frames are
//! contiguous and in member order.
//!
//! The leader publishes the frames with one `write_at` each, one `sync_data`,
//! one slot write and one `sync_data`, with the `flock` held throughout. One
//! publish carries at most [`GROUP_BYTES`] of frames: when the next frame
//! would pass that, the frames so far are published first and the next frame
//! starts the next publish, so a frame larger than the cap is published alone.
//! Only once a publish's slot sync has returned are its members registered
//! with the sync adapter, under the commit gate and in sequence order. The
//! leader then releases the `flock`, the write mutex and the gate, wakes
//! every member with its own result, and hands the lead on: the next queued
//! writer, if any, leads. Each member calls `on_commit` on its own thread.
//! In `WriteThrough` only the first member's batch finds the adapter idle;
//! every later member drains the queue, oldest first, until its own batch
//! is sent, so each member returns only once its own batch is delivered.
//!
//! **Auto-checkpoint (spec §4.4).** When the group published and its WAL is
//! now longer than `max(snapshot_length, AUTO_CHECKPOINT_BYTES)`, the leader
//! checkpoints only after all of the above, and after it has delivered its
//! own batch: it takes the write mutex and the `flock` again, refreshes,
//! and checkpoints if the refreshed state is still past the threshold
//! (another handle may have checkpointed meanwhile). So no member waits for
//! the checkpoint, nor for the leader's `on_commit` behind it; only the
//! leader's own call returns after the checkpoint, whose failure does not
//! change its result: a failed sync in it poisons the handle, and any other
//! failure is reported on stderr and tried again later.
//!
//! A sync that fails poisons the handle (`Poison`): the group fails, and
//! every later write and checkpoint on the handle is refused.
//!
//! **A publish that fails** (any write or sync) fails every member it carries
//! and every member not yet run, and nothing of it is acknowledged or put in
//! the cache. Before the slot write is issued the file still names the state
//! before the publish. From the slot write on, the file names either that
//! state or the state with the whole publish, never a part of it: the next
//! refresh reads whichever slot is there. The crash harness checks both.

use super::container_io::ContainerIo;
use super::layout::{Op, backend, encode_frame};
use super::state::{DocState, Overlay, publish_overlay};
use super::{
    Change, Committed, Disk, DiskCache, FerroStore, Registered, auto_checkpoint, checkpoint_due,
    lock_mutex, wait_mutex,
};
use fl_core::StoreError;
use std::any::Any;
use std::collections::VecDeque;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::ThreadId;

/// The most frame bytes one publish carries, unless one frame is larger.
pub(super) const GROUP_BYTES: usize = 4 << 20;
/// The most members one group takes from the queue.
pub(super) const GROUP_MEMBERS: usize = 64;

/// What one member's body staged over the state it ran against.
pub(super) struct Staged {
    pub(super) ops: Vec<Op>,
    pub(super) overlay: Overlay,
}

/// A member's result once its group is done.
struct Acked {
    /// `None` when the body wrote nothing.
    seq: Option<u64>,
    /// What the member sends, or drains to, on its own thread.
    delivery: Registered,
}

/// Where one member is. The member and the leader both wait on the member's
/// condvar, for different states.
enum Slot {
    Queued,
    /// The previous leader is done and this member leads next.
    Lead,
    /// Run the body against this state.
    Turn(Arc<DocState>),
    /// The body ran. The leader takes this.
    Ran(Result<Option<Staged>, StoreError>),
    Done(Result<Acked, StoreError>),
    /// Taken by the member or the leader; the other side's next state follows.
    Empty,
}

struct Member {
    silent: bool,
    /// The member's own thread: it owns the `on_commit` its registration
    /// starts (see `SyncSlot::notify_owner`).
    thread: ThreadId,
    slot: Mutex<Slot>,
    cv: Condvar,
}

impl Member {
    fn set(&self, next: Slot) {
        *lock_mutex(&self.slot) = next;
        self.cv.notify_all();
    }

    /// Give the member its turn and wait for its body's outcome.
    fn turn(&self, state: Arc<DocState>) -> Result<Option<Staged>, StoreError> {
        self.set(Slot::Turn(state));
        let mut slot = lock_mutex(&self.slot);
        loop {
            if matches!(*slot, Slot::Ran(_))
                && let Slot::Ran(ran) = std::mem::replace(&mut *slot, Slot::Empty)
            {
                return ran;
            }
            slot = wait_mutex(&self.cv, slot);
        }
    }
}

/// The writers queued on one `Disk`.
#[derive(Default)]
pub(super) struct Group {
    queue: Mutex<Queue>,
}

#[derive(Default)]
struct Queue {
    /// Writers not yet taken into a group, in arrival order. When a leader
    /// is active it is at the front until it takes its group.
    waiting: VecDeque<Arc<Member>>,
    leading: bool,
}

impl Group {
    /// Queue `member`. `true` when no leader is active: it leads now.
    fn join(&self, member: &Arc<Member>) -> bool {
        let mut queue = lock_mutex(&self.queue);
        queue.waiting.push_back(Arc::clone(member));
        !std::mem::replace(&mut queue.leading, true)
    }

    /// The group: what is queued now, at most [`GROUP_MEMBERS`].
    fn take(&self) -> Vec<Arc<Member>> {
        let mut queue = lock_mutex(&self.queue);
        let n = queue.waiting.len().min(GROUP_MEMBERS);
        queue.waiting.drain(..n).collect()
    }

    /// The next queued writer leads, or no leader is active.
    fn hand_off(&self) {
        let mut queue = lock_mutex(&self.queue);
        match queue.waiting.front() {
            Some(next) => next.set(Slot::Lead),
            None => queue.leading = false,
        }
    }

    /// Writers queued and not yet taken into a group.
    #[cfg(test)]
    pub(super) fn queued(&self) -> usize {
        lock_mutex(&self.queue).waiting.len()
    }
}

/// One leader's group. Dropped last in [`FerroStore::lead_group`], after
/// the locks: it wakes every member and hands the lead on, also when the
/// leader unwinds.
struct Leadership<'a> {
    group: &'a Group,
    members: Vec<Arc<Member>>,
    results: Vec<Option<Result<Acked, StoreError>>>,
}

impl Leadership<'_> {
    /// Every member with no result yet gets `err`.
    fn fail_rest(&mut self, err: &StoreError) {
        for result in &mut self.results {
            if result.is_none() {
                *result = Some(Err(copy_error(err)));
            }
        }
    }
}

impl Drop for Leadership<'_> {
    fn drop(&mut self) {
        for (member, result) in self.members.iter().zip(&mut self.results) {
            let result = result.take().unwrap_or_else(|| {
                Err(backend(
                    "the group's leader stopped before this commit was published",
                ))
            });
            member.set(Slot::Done(result));
        }
        self.group.hand_off();
    }
}

/// The group's state between publishes: what the last publish (or the
/// refresh) left, and every change committed since.
struct Running {
    index: usize,
    base: Arc<DiskCache>,
    /// A publish of this group wrote frames: `base` is its state.
    published: bool,
    /// The one committed overlay not yet applied to a state: a group of one
    /// committing member applies it to the cache in place (spec §4.2).
    first: Option<(Overlay, u64)>,
    /// `base.state` with every committed change since, once a later member
    /// had to run against it.
    state: Option<Arc<DocState>>,
}

impl Running {
    fn last_seq(&self) -> u64 {
        match (&self.first, &self.state) {
            (Some((_, seq)), _) => *seq,
            (None, Some(state)) => state.last_seq,
            (None, None) => self.base.state.last_seq,
        }
    }

    /// The state the next member runs against.
    fn turn_state(&mut self) -> Arc<DocState> {
        if let Some((overlay, seq)) = self.first.take() {
            let mut state = Arc::clone(&self.base.state);
            publish_overlay(&mut state, overlay, seq);
            self.state = Some(state);
        }
        Arc::clone(self.state.as_ref().unwrap_or(&self.base.state))
    }

    /// A member committed `overlay` at `seq`, staged over [`Self::turn_state`].
    fn add(&mut self, overlay: Overlay, seq: u64) {
        match &mut self.state {
            // The member has dropped its hold: this is in place.
            Some(state) => publish_overlay(state, overlay, seq),
            None => self.first = Some((overlay, seq)),
        }
    }

    fn take_change(&mut self) -> Option<Change> {
        match (self.first.take(), self.state.take()) {
            (Some((overlay, seq)), _) => Some(Change::Overlay(overlay, seq)),
            (None, Some(state)) => Some(Change::State(state)),
            (None, None) => None,
        }
    }
}

/// The frames of the publish being gathered, and whose they are.
#[derive(Default)]
struct Pending {
    frames: Vec<Vec<u8>>,
    bytes: usize,
    /// `(member position, its commit)`, in sequence order.
    committed: Vec<(usize, Committed)>,
    /// Members whose bodies wrote nothing. They may have read the changes in
    /// `frames`, so they too wait for this publish.
    readers: Vec<usize>,
}

impl FerroStore {
    /// Run `body` as one member of a group on `disk` (spec §4.3) and wait
    /// for the group. `body` gets the state to run against and returns its
    /// value and what it staged; `None` writes nothing. `Ok` carries the
    /// value and the sequence the commit took. A commit is registered with
    /// the sync adapter after its slot sync; its `on_commit` runs on this
    /// thread, after the group's locks are released, and in `WriteThrough`
    /// this returns only once that batch is sent (`deliver_owned`). A leader
    /// whose group left the WAL past the auto-checkpoint threshold
    /// checkpoints after that (spec §4.4), so its own call returns after the
    /// checkpoint.
    pub(super) fn group_write<T>(
        &self,
        disk: &Disk,
        silent: bool,
        body: impl FnOnce(Arc<DocState>) -> Result<(T, Option<Staged>), StoreError>,
    ) -> Result<(T, Option<u64>), StoreError> {
        let me = Arc::new(Member {
            silent,
            thread: std::thread::current().id(),
            slot: Mutex::new(Slot::Queued),
            cv: Condvar::new(),
        });
        let mut body = Some(body);
        let mut value: Option<T> = None;
        let mut panicked: Option<Box<dyn Any + Send>> = None;
        let mut checkpoint = false;
        let result = {
            let mut run = |state: Arc<DocState>| -> Result<Option<Staged>, StoreError> {
                let Some(body) = body.take() else {
                    return Err(backend("a group member's body ran twice"));
                };
                match panic::catch_unwind(AssertUnwindSafe(|| body(state))) {
                    Ok(Ok((out, staged))) => {
                        value = Some(out);
                        Ok(staged)
                    }
                    Ok(Err(err)) => Err(err),
                    Err(payload) => {
                        panicked = Some(payload);
                        Err(backend("a write body panicked"))
                    }
                }
            };
            if disk.group.join(&me) {
                checkpoint |= self.lead_group(disk, &me, &mut run);
            }
            loop {
                let mut slot = lock_mutex(&me.slot);
                while matches!(*slot, Slot::Queued | Slot::Ran(_) | Slot::Empty) {
                    slot = wait_mutex(&me.cv, slot);
                }
                match std::mem::replace(&mut *slot, Slot::Empty) {
                    Slot::Lead => {
                        drop(slot);
                        checkpoint |= self.lead_group(disk, &me, &mut run);
                    }
                    Slot::Turn(state) => {
                        drop(slot);
                        let ran = run(state);
                        me.set(Slot::Ran(ran));
                    }
                    Slot::Done(result) => break result,
                    Slot::Queued | Slot::Ran(_) | Slot::Empty => {
                        unreachable!("the wait above passes only Lead, Turn and Done")
                    }
                }
            }
        };
        let delivered = result.and_then(|acked| {
            self.deliver_owned(acked.delivery)?;
            Ok(acked.seq)
        });
        // Spec §4.4: every member has its result and the leader's own batch
        // is delivered; the checkpoint takes a lock window of its own.
        if checkpoint {
            auto_checkpoint(disk);
        }
        if let Some(payload) = panicked {
            panic::resume_unwind(payload);
        }
        let seq = delivered?;
        let value = value.ok_or_else(|| backend("a group member returned no value"))?;
        Ok((value, seq))
    }

    /// Lead one group. `me` is at the front of the queue; `own` runs its
    /// body. `true` when the group published a state that is due an
    /// auto-checkpoint, which the caller runs once it has its own result
    /// and has delivered its own batch ([`auto_checkpoint`]).
    fn lead_group(
        &self,
        disk: &Disk,
        me: &Arc<Member>,
        own: &mut dyn FnMut(Arc<DocState>) -> Result<Option<Staged>, StoreError>,
    ) -> bool {
        // Declared first, so dropped last: after the flock, the write mutex
        // and the gate.
        let mut lead = Leadership {
            group: &disk.group,
            members: Vec::new(),
            results: Vec::new(),
        };
        let _gate = lock_mutex(&self.commit_gate);
        let mut write = lock_mutex(&disk.write);
        // A poisoned handle runs no body: the whole group is refused. The
        // `flock` is taken on the file the path names now (spec §5).
        let locked = match disk.poison.check() {
            Ok(()) => disk.lock_current(&mut write).and_then(|flock| {
                let refreshed = disk.refresh(flock.file(), flock.id)?;
                Ok((flock, refreshed))
            }),
            Err(err) => Err(err),
        };
        lead.members = disk.group.take();
        lead.results = lead.members.iter().map(|_| None).collect();
        debug_assert!(
            lead.members
                .first()
                .is_some_and(|first| Arc::ptr_eq(first, me)),
            "a leader is the front of its group"
        );
        match locked {
            Err(err) => {
                lead.fail_rest(&err);
                false
            }
            Ok((flock, (index, base))) => {
                let running = Running {
                    index,
                    base,
                    published: false,
                    first: None,
                    state: None,
                };
                // Spec §4.4: only whether the checkpoint is due is decided
                // here. It runs outside this lock window, so no member's
                // result waits for it.
                self.run_group(disk, flock.file(), &mut lead, me, own, running)
                    .is_some_and(|done| done.published && checkpoint_due(disk, &done.base))
            }
        }
    }

    /// Every member's body in turn, then the publishes. The state the last
    /// publish left, or `None` when a publish failed.
    fn run_group(
        &self,
        disk: &Disk,
        file: &dyn ContainerIo,
        lead: &mut Leadership<'_>,
        me: &Arc<Member>,
        own: &mut dyn FnMut(Arc<DocState>) -> Result<Option<Staged>, StoreError>,
        mut running: Running,
    ) -> Option<Running> {
        let mut pending = Pending::default();
        for pos in 0..lead.members.len() {
            let member = Arc::clone(&lead.members[pos]);
            let state = running.turn_state();
            let ran = if Arc::ptr_eq(&member, me) {
                own(state)
            } else {
                member.turn(state)
            };
            let staged = match ran {
                Ok(Some(staged)) => staged,
                Ok(None) => {
                    pending.readers.push(pos);
                    continue;
                }
                Err(err) => {
                    lead.results[pos] = Some(Err(err));
                    continue;
                }
            };
            let framed = running
                .last_seq()
                .checked_add(1)
                .ok_or_else(|| backend("transaction sequence overflow"))
                .and_then(|seq| Ok((seq, encode_frame(seq, &staged.ops)?)));
            let (seq, frame) = match framed {
                Ok(framed) => framed,
                Err(err) => {
                    lead.results[pos] = Some(Err(err));
                    continue;
                }
            };
            if !pending.frames.is_empty() && pending.bytes + frame.len() > GROUP_BYTES {
                running = match self.publish(disk, file, running, &mut pending, lead) {
                    Ok(next) => next,
                    Err(err) => {
                        lead.fail_rest(&err);
                        return None;
                    }
                };
            }
            running.add(staged.overlay, seq);
            pending.bytes += frame.len();
            pending.frames.push(frame);
            pending.committed.push((
                pos,
                Committed {
                    seq,
                    ops: staged.ops,
                },
            ));
        }
        match self.publish(disk, file, running, &mut pending, lead) {
            Ok(done) => Some(done),
            Err(err) => {
                lead.fail_rest(&err);
                None
            }
        }
    }

    /// Publish `pending`, install the result in the cache and register its
    /// members, in sequence order. Returns the state to go on from.
    fn publish(
        &self,
        disk: &Disk,
        file: &dyn ContainerIo,
        mut running: Running,
        pending: &mut Pending,
        lead: &mut Leadership<'_>,
    ) -> Result<Running, StoreError> {
        let published =
            disk.publish_or_settle(file, running.index, &running.base, &pending.frames)?;
        if let Some(new_sb) = published {
            let change = running
                .take_change()
                .ok_or_else(|| backend("a publish with frames has no change"))?;
            running.base = disk.install_commit(running.base, new_sb, change);
            running.index ^= 1;
            running.published = true;
        }
        for (pos, committed) in pending.committed.drain(..) {
            let member = &lead.members[pos];
            let delivery = self.register_committed_for(&committed, member.silent, member.thread);
            lead.results[pos] = Some(Ok(Acked {
                seq: Some(committed.seq),
                delivery,
            }));
        }
        for pos in pending.readers.drain(..) {
            lead.results[pos] = Some(Ok(Acked {
                seq: None,
                delivery: Registered::Nothing,
            }));
        }
        pending.frames.clear();
        pending.bytes = 0;
        Ok(running)
    }

    /// Hold until `n` writers are queued on this store's disk. A test holds
    /// the commit gate meanwhile, so the `n` form one group.
    #[cfg(test)]
    pub(super) fn wait_until_queued(&self, n: usize) {
        let super::Inner::Disk(disk) = &self.inner else {
            panic!("group commit is a disk handle's");
        };
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while disk.group.queued() < n {
            assert!(
                std::time::Instant::now() < deadline,
                "{} of {n} writers queued after 30 s",
                disk.group.queued()
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}

/// The same error for another member. `StoreError` is not `Clone`; the
/// kinds a publish or refresh returns are copied as they are.
fn copy_error(err: &StoreError) -> StoreError {
    match err {
        StoreError::Backend(detail) => StoreError::Backend(detail.clone()),
        StoreError::ContainerFormat {
            store,
            found,
            engine,
            unknown_incompat,
            newest,
        } => StoreError::ContainerFormat {
            store: store.clone(),
            found: *found,
            engine: *engine,
            unknown_incompat: *unknown_incompat,
            newest: *newest,
        },
        StoreError::FormatVersion {
            found,
            oldest,
            newest,
        } => StoreError::FormatVersion {
            found: *found,
            oldest: *oldest,
            newest: *newest,
        },
        other => StoreError::Backend(other.to_string()),
    }
}
