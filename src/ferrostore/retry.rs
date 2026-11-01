//! Bounded retry of a remote validated commit (spec §4.5, §6.1).
//!
//! A remote handle runs a write body over a read session on the server, then
//! sends its ops with its read set (spec §6.1). When another writer changed
//! what the body read, the commit is lost and the body runs again, over a
//! new session. One write makes at most [`REMOTE_ATTEMPTS`] attempts. After
//! each lost attempt but the last it waits a jittered, exponentially growing
//! time ([`Backoff`]). When every attempt is lost the write is
//! `StoreError::Conflict` and has committed nothing.

use super::layout::Op;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::panic::Location;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// The most commit attempts of one remote write.
pub(super) const REMOTE_ATTEMPTS: u32 = 8;
/// The shortest wait after a lost attempt.
pub(super) const BACKOFF_FIRST: Duration = Duration::from_millis(5);
/// The longest wait after a lost attempt.
pub(super) const BACKOFF_CAP: Duration = Duration::from_millis(500);

/// Calls of [`Backoff::new`] in this process: two writes on one thread get
/// different jitter.
static CALLS: AtomicU64 = AtomicU64::new(0);

/// The waits of one remote write.
///
/// After lost attempt `n` (from 1) the wait is uniform in `[b, min(2b, cap)]`
/// with `b = min(5 ms * 2^(n-1), 500 ms)`: 5–10 ms, 10–20 ms, … 320–500 ms.
/// The jitter is an xorshift seeded from the thread id and a per-process call
/// count, so concurrent writers spread out without a clock or a dependency.
pub(super) struct Backoff {
    rng: u64,
}

impl Backoff {
    pub(super) fn new() -> Self {
        let mut hasher = DefaultHasher::new();
        std::thread::current().id().hash(&mut hasher);
        CALLS.fetch_add(1, Ordering::Relaxed).hash(&mut hasher);
        Self {
            rng: hasher.finish() | 1,
        }
    }

    /// The wait after lost attempt `attempt` (from 1).
    pub(super) fn delay(&mut self, attempt: u32) -> Duration {
        let shift = attempt.saturating_sub(1).min(16);
        let base = BACKOFF_FIRST.saturating_mul(1u32 << shift).min(BACKOFF_CAP);
        let top = base.saturating_mul(2).min(BACKOFF_CAP);
        // xorshift64*: the top 53 bits as a fraction in [0, 1).
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let draw = self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        let frac = draw as f64 / (1u64 << 53) as f64;
        base + (top - base).mul_f64(frac)
    }

    /// Sleep [`Self::delay`] of `attempt`.
    pub(super) fn wait(&mut self, attempt: u32) {
        let delay = self.delay(attempt);
        #[cfg(test)]
        if !hook::waits() {
            return;
        }
        std::thread::sleep(delay);
    }
}

/// One remote commit attempt, as it is about to be sent: the write's call
/// site, the attempt number (from 1), what the body read of the state (the
/// session's answers), and the ops it staged. A test hook sees it
/// (`cfg(test)` only) and may make the attempt lose, as if another writer
/// had changed what the body read.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) struct Cas<'a> {
    pub(super) caller: &'static Location<'static>,
    pub(super) attempt: u32,
    pub(super) known: &'a dyn std::fmt::Debug,
    pub(super) ops: &'a [Op],
}

/// `true` when a test makes this attempt lose without sending it. Never in
/// a build without tests.
#[cfg(not(test))]
pub(super) fn forced_loss(_: &Cas<'_>) -> bool {
    false
}

#[cfg(test)]
pub(super) fn forced_loss(cas: &Cas<'_>) -> bool {
    hook::lose(cas)
}

#[cfg(test)]
pub(super) mod hook {
    use super::Cas;
    use std::cell::{Cell, RefCell};

    type Hook = Box<dyn FnMut(&Cas<'_>) -> bool>;

    thread_local! {
        static HOOK: RefCell<Option<Hook>> = const { RefCell::new(None) };
        static WAITS: Cell<bool> = const { Cell::new(true) };
    }

    /// Removes the hook when dropped, also on unwind.
    pub(in super::super) struct Installed;

    impl Drop for Installed {
        fn drop(&mut self) {
            HOOK.with(|hook| hook.borrow_mut().take());
            WAITS.with(|waits| waits.set(true));
        }
    }

    /// Run `hook` before every remote commit attempt this thread
    /// makes; `true` loses the attempt. `waits` false skips the backoff
    /// sleeps.
    pub(in super::super) fn install(
        hook: impl FnMut(&Cas<'_>) -> bool + 'static,
        waits: bool,
    ) -> Installed {
        HOOK.with(|slot| *slot.borrow_mut() = Some(Box::new(hook)));
        WAITS.with(|slot| slot.set(waits));
        Installed
    }

    /// The hook's verdict. It is taken out while it runs, so a write the hook
    /// itself makes is not hooked.
    pub(super) fn lose(cas: &Cas<'_>) -> bool {
        let Some(mut hook) = HOOK.with(|slot| slot.borrow_mut().take()) else {
            return false;
        };
        let lost = hook(cas);
        HOOK.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.is_none() {
                *slot = Some(hook);
            }
        });
        lost
    }

    pub(super) fn waits() -> bool {
        WAITS.with(Cell::get)
    }
}
