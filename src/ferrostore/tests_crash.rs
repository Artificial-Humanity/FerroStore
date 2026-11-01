//! Crash-consistency harness (spec §3.5).
//!
//! # How it runs
//!
//! 1. **Reference run.** The scenario's workload runs once on a [`SimFs`]
//!    that never cuts. That gives the full op trace, and the logical state
//!    after every acknowledged operation (`states[k]` is the state after `k`
//!    acknowledged operations; `states[0]` is the state before the first).
//! 2. **Cut runs.** For every `N` from 0 to the trace length, the workload runs
//!    again from the same start on a disk that cuts power after op `N`. Its
//!    trace must be the reference trace's first `N` ops.
//! 3. **Survivors.** For each cut, every combination of fates of the ops not
//!    yet durable (kept, dropped, torn: see `crash_sim`) is one case, up to
//!    [`FATE_CAP`] combinations; past that, a sample seeded from [`SEED`]
//!    and the cut. One cut of today's workload reaches the cap: inside the
//!    compaction of step 6, the sibling's snapshot and slot writes are
//!    pending together before its sync (a file no name yet makes the store).
//! 4. **Checks** ([`check_recovery`]), on each survivor image, reopened
//!    through the normal `FerroStore::open` path on the simulated disk:
//!    - no two valid slots share an epoch, and no slot is from a newer build;
//!    - a store file whose creation was acknowledged still exists; a cut
//!      during first-use init leaves the file absent, empty, re-initialisable
//!      (zero header and a prefix of the first snapshot) or initialised;
//!    - `open` succeeds and `format_version()` is on the ladder;
//!    - `last_seq` is not below the acknowledged prefix's;
//!    - the state is exactly `states[j]` for some `j` from the acknowledged
//!      count to the started count: every acknowledged operation is present
//!      with its exact content, and an unacknowledged one is all there or
//!      not there at all;
//!    - a state past the starting one is published in a format-2 slot;
//!    - one more commit after recovery lands, and reads back after a reopen;
//!    - a cut inside the group commit of step 5 leaves the whole group or
//!      none of it, and acknowledges all three writers or none;
//!    - a cut inside the compaction of step 6 leaves exactly one file as the
//!      store, the old one or the new one: no other name (a leftover
//!      `.compact-` temp included) names the store's inode, and a leftover
//!      temp is never what `open` reads (the reopened state is checked
//!      above, through the store's own name).
//! 5. **Reference-run checks:** every container write and sync runs under the
//!    file's `flock`, which is not released between a write and the slot sync
//!    that publishes it ([`check_locks`]); and when an operation is
//!    acknowledged, nothing it wrote is still unsynced.
//!
//! # Extending it
//!
//! - **A new workload step:** a [`Step`] in [`WORKLOAD`] (or a scenario's own
//!   list). Each acknowledged call goes through [`Ctx::op`]; nothing else is
//!   needed. Compaction (§5) is step 6 of [`WORKLOAD`].
//! - **A new kind of I/O op:** a method on `ContainerIo`/`ContainerFs`, an
//!   [`IoOp`] variant, and its fates in `SimFs::pending` (`crash_sim`).
//! - **Group commit (§4.3):** [`three_writers`] makes one real group.

use super::container_io::{ContainerFs, RealFs};
use super::crash_sim::{Image, IoOp, SimFs, fate_vectors};
use super::layout::{
    self, FORMAT_V1, FORMAT_V2, HEADER_BYTES, Op, SLOT_LEN, SlotRead, Superblock, backend,
    encode_frame, encode_snapshot, encode_superblock_as, parse_superblock,
};
use super::state::DocState;
use super::vector::VectorCollection;
use super::{
    ColumnDef, ColumnType, ColumnValue, FerroStore, VectorMetric, VectorStorage, lock_mutex,
    new_store_docs,
};
use fl_core::StoreError;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Barrier, Mutex};
use std::time::{Duration, Instant};

/// Where the simulated store lives.
const STORE: &str = "/sim/store.fls";
/// The most fate combinations tried exhaustively at one cut.
const FATE_CAP: usize = 64;
/// The base seed of a sampled cut. A failure message prints the seed it used.
const SEED: u64 = 0x5EED_0003_0005_F1F0;
/// Spec §3.5: the harness stays under 30 s in a debug `cargo test`.
const TIME_BUDGET: Duration = Duration::from_secs(30);

// ---- logical state -----------------------------------------------------------

/// What a reader can observe: everything but the physical layout.
#[derive(Clone, Debug, PartialEq)]
struct Logical {
    last_seq: u64,
    docs: BTreeMap<(String, String), Value>,
    tables: BTreeMap<String, (Vec<ColumnDef>, Vec<Vec<u8>>)>,
    vectors: BTreeMap<String, VectorCollection>,
}

impl Logical {
    fn of(state: &DocState) -> Self {
        Self {
            last_seq: state.last_seq,
            docs: state
                .docs
                .iter()
                .map(|(collection, id, doc)| ((collection.clone(), id.clone()), doc.clone()))
                .collect(),
            tables: state
                .tables
                .iter()
                .map(|(name, table)| (name.clone(), (table.columns.clone(), table.groups.clone())))
                .collect(),
            vectors: state
                .vectors
                .iter()
                .map(|(name, col)| (name.clone(), (**col).clone()))
                .collect(),
        }
    }

    fn read(store: &FerroStore) -> Result<Self, String> {
        store
            .read_state()
            .map(|state| Self::of(&state))
            .map_err(|err| format!("read_state failed: {err}"))
    }

    fn summary(&self) -> String {
        let docs: Vec<String> = self
            .docs
            .keys()
            .map(|(collection, id)| format!("{collection}/{id}"))
            .collect();
        let tables: Vec<String> = self
            .tables
            .iter()
            .map(|(name, (_, groups))| format!("{name}:{} groups", groups.len()))
            .collect();
        let vectors: Vec<String> = self
            .vectors
            .iter()
            .map(|(name, col)| format!("{name}:{:?}", col.vectors.keys().collect::<Vec<_>>()))
            .collect();
        format!(
            "last_seq {} docs [{}] tables [{}] vectors [{}]",
            self.last_seq,
            docs.join(", "),
            tables.join(", "),
            vectors.join(", ")
        )
    }
}

// ---- the workload -------------------------------------------------------------

/// Counts the operations a workload starts and has acknowledged.
struct Ctx<'a> {
    /// Set on the reference run: each acknowledged op records the live image.
    record: Option<&'a SimFs>,
    /// The disk and path the workload runs on, for a step that opens a
    /// second handle (another process, as far as the store can tell).
    fs: &'a dyn ContainerFs,
    path: &'a Path,
    images: Mutex<Vec<Image>>,
    /// Reference run: operations acknowledged while some of their writes
    /// were not yet durable.
    unsynced_acks: Mutex<Vec<String>>,
    labels: Mutex<Vec<String>>,
    started: AtomicUsize,
    acked: AtomicUsize,
    failed: AtomicBool,
}

impl<'a> Ctx<'a> {
    fn new(record: Option<&'a SimFs>, fs: &'a dyn ContainerFs, path: &'a Path) -> Self {
        Self {
            record,
            fs,
            path,
            images: Mutex::new(Vec::new()),
            unsynced_acks: Mutex::new(Vec::new()),
            labels: Mutex::new(Vec::new()),
            started: AtomicUsize::new(0),
            acked: AtomicUsize::new(0),
            failed: AtomicBool::new(false),
        }
    }

    /// One acknowledged operation: `Ok` from `call` is the acknowledgement.
    /// After any failure no further op starts.
    fn op<T>(
        &self,
        label: &str,
        call: impl FnOnce() -> Result<T, StoreError>,
    ) -> Result<T, StoreError> {
        if self.failed.load(Ordering::SeqCst) {
            return Err(backend("an earlier operation failed"));
        }
        self.started.fetch_add(1, Ordering::SeqCst);
        lock(&self.labels).push(label.to_string());
        match call() {
            Ok(value) => {
                self.acked.fetch_add(1, Ordering::SeqCst);
                if let Some(fs) = self.record {
                    let live = fs.live_image();
                    if fs.durable_image() != live {
                        lock(&self.unsynced_acks).push(label.to_string());
                    }
                    lock(&self.images).push(live);
                }
                Ok(value)
            }
            Err(err) => {
                self.failed.store(true, Ordering::SeqCst);
                Err(err)
            }
        }
    }

    /// A second, independent handle on the workload's store.
    fn open_other(&self) -> Result<FerroStore, StoreError> {
        FerroStore::open_with(self.fs, self.path)
    }

    fn started(&self) -> usize {
        self.started.load(Ordering::SeqCst)
    }

    fn acked(&self) -> usize {
        self.acked.load(Ordering::SeqCst)
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

type StepFn = fn(&Ctx<'_>, &FerroStore) -> Result<(), StoreError>;

/// One scripted step of a workload. It may make several acknowledged calls.
struct Step {
    name: &'static str,
    run: StepFn,
}

/// The spec §3.5 workload, after the first-use `open`.
const WORKLOAD: &[Step] = &[
    Step {
        name: "1 puts and deletes",
        run: puts_and_deletes,
    },
    Step {
        name: "2 column-table append",
        run: column_table_append,
    },
    Step {
        name: "3 vector upsert",
        run: vector_upsert,
    },
    Step {
        name: "4 checkpoint",
        run: checkpoint,
    },
    Step {
        name: "5 three writers",
        run: three_writers,
    },
    Step {
        name: "6 compaction",
        run: compaction,
    },
    Step {
        name: "7 commits after the checkpoint",
        run: tail,
    },
    Step {
        name: "8 a commit past the auto-checkpoint threshold",
        run: auto_checkpoint,
    },
];

/// Spec §4.1: a handle that refreshes after another handle's commits (a
/// WAL tail) and after its checkpoint (a full load), on reads and writes.
const TWO_HANDLE_WORKLOAD: &[Step] = &[
    Step {
        name: "1 puts and deletes",
        run: puts_and_deletes,
    },
    Step {
        name: "2 a second handle",
        run: second_handle,
    },
];

/// The first write to a format-1 file publishes its first format-2 slot.
const V1_WORKLOAD: &[Step] = &[
    Step {
        name: "first write upgrades the slot",
        run: |ctx, store| {
            ctx.op("put notes/new", || {
                store.put_doc("notes", "new", &json!({"v": 2}))
            })
        },
    },
    Step {
        name: "delete a v1 document",
        run: |ctx, store| ctx.op("delete notes/old", || store.delete_doc("notes", "old")),
    },
    Step {
        name: "checkpoint",
        run: checkpoint,
    },
    Step {
        name: "put after the checkpoint",
        run: |ctx, store| {
            ctx.op("put notes/after", || {
                store.put_doc("notes", "after", &json!({"v": 3}))
            })
        },
    },
];

fn puts_and_deletes(ctx: &Ctx<'_>, store: &FerroStore) -> Result<(), StoreError> {
    ctx.op("put notes/a", || {
        store.put_doc("notes", "a", &json!({"n": 1}))
    })?;
    ctx.op("put notes/b", || {
        store.put_doc("notes", "b", &json!({"n": 2}))
    })?;
    ctx.op("delete notes/a", || store.delete_doc("notes", "a"))?;
    // One commit of three ops: all three or none.
    ctx.op("txn put c, delete b, put a", || {
        store.with_write(|txn| {
            txn.put("notes", "c", json!({"n": 3}));
            txn.delete("notes", "b");
            txn.put("notes", "a", json!({"n": 4}));
            Ok(())
        })
    })?;
    // A frame that spans several sectors, so it can tear.
    let big = "x".repeat(1500);
    ctx.op("put notes/big", || {
        store.put_doc("notes", "big", &json!({"text": big}))
    })
}

fn column_table_append(ctx: &Ctx<'_>, store: &FerroStore) -> Result<(), StoreError> {
    let columns = [
        ColumnDef {
            name: "n".to_string(),
            ty: ColumnType::Int64,
        },
        ColumnDef {
            name: "label".to_string(),
            ty: ColumnType::Utf8,
        },
    ];
    ctx.op("create table runs", || {
        store.create_column_table("runs", &columns)
    })?;
    let row = |n: i64, label: &str| vec![ColumnValue::Int64(n), ColumnValue::Utf8(label.into())];
    ctx.op("append runs group 1", || {
        store.append_column_rows("runs", &[row(1, "a"), row(2, "b")])
    })?;
    ctx.op("append runs group 2", || {
        store.append_column_rows("runs", &[row(3, "c")])
    })
}

fn vector_upsert(ctx: &Ctx<'_>, store: &FerroStore) -> Result<(), StoreError> {
    ctx.op("create vectors emb", || {
        store.create_vector_collection("emb", 4, VectorStorage::F32, VectorMetric::Cosine, 4, 8, 8)
    })?;
    ctx.op("upsert emb/v1", || {
        store.upsert_vector("emb", "v1", &[1.0, 0.0, 0.0, 0.0], None)
    })?;
    ctx.op("upsert emb/v2", || {
        store.upsert_vector("emb", "v2", &[0.0, 1.0, 0.0, 0.0], Some(&json!({"k": 2})))
    })?;
    ctx.op("upsert emb/v1 again", || {
        store.upsert_vector("emb", "v1", &[0.5, 0.5, 0.0, 0.0], None)
    })
}

fn checkpoint(ctx: &Ctx<'_>, store: &FerroStore) -> Result<(), StoreError> {
    ctx.op("checkpoint", || store.checkpoint())
}

/// Spec §3.5 step 5, "a group commit of three writers" (§4.3). Three
/// threads are released together by a barrier while this thread holds the
/// commit gate, so the first to queue leads and waits on the gate; the gate
/// is let go once all three are queued, and the leader takes the three as
/// one group.
///
/// Which thread queues first, and so the member order, still varies from
/// run to run. The cut runs need the reference run's exact op trace, and get
/// it: the three frames have the same length (`w0`/`w1`/`w2`, one digit
/// each), and the leader does every container op, so the trace does not
/// depend on the order; the trace-prefix check is not relaxed. The checks
/// need no change either: the three acknowledgements all come after the
/// slot sync and all record the same image, so `states` holds the whole
/// group three times and any part of it matches no state.
fn three_writers(ctx: &Ctx<'_>, store: &FerroStore) -> Result<(), StoreError> {
    let start = Barrier::new(3);
    std::thread::scope(|scope| {
        let gate = lock_mutex(&store.commit_gate);
        let writers: Vec<_> = (0..3)
            .map(|writer| {
                let start = &start;
                scope.spawn(move || {
                    start.wait();
                    ctx.op(&format!("writer {writer}"), || {
                        store.put_doc("writers", &format!("w{writer}"), &json!({"writer": writer}))
                    })
                })
            })
            .collect();
        store.wait_until_queued(3);
        drop(gate);
        let mut first = Ok(());
        for writer in writers {
            let result = writer
                .join()
                .map_err(|_| backend("writer thread panicked"))?;
            if first.is_ok() {
                first = result;
            }
        }
        first
    })
}

/// `Err` unless `store` reads `want` at `collection/id`: a read that missed
/// another handle's commit fails the reference run.
fn expect_doc(
    store: &FerroStore,
    collection: &str,
    id: &str,
    want: Value,
) -> Result<(), StoreError> {
    match store.get_doc(collection, id)? {
        Some(got) if got == want => Ok(()),
        got => Err(backend(format!(
            "{collection}/{id} reads {got:?}, not {want}"
        ))),
    }
}

fn second_handle(ctx: &Ctx<'_>, store: &FerroStore) -> Result<(), StoreError> {
    let other = ctx.op("open a second handle", || ctx.open_other())?;
    ctx.op("second handle puts", || {
        other.put_doc("other", "x", &json!(1))
    })?;
    ctx.op("first handle reads the tail", || {
        expect_doc(store, "other", "x", json!(1))
    })?;
    ctx.op("first handle writes on the tail", || {
        store.with_write(|txn| {
            let x = txn.get("other", "x").ok_or_else(|| backend("x missing"))?;
            txn.put("other", "y", json!({"after": x}));
            Ok(())
        })
    })?;
    ctx.op("second handle writes on the tail", || {
        expect_doc(&other, "other", "y", json!({"after": 1}))?;
        other.put_doc("other", "z", &json!(3))
    })?;
    ctx.op("second handle checkpoints", || other.checkpoint())?;
    ctx.op("first handle writes after the checkpoint", || {
        expect_doc(store, "other", "z", json!(3))?;
        store.put_doc("other", "w", &json!(4))
    })?;
    ctx.op("second handle reads the first's commit", || {
        expect_doc(&other, "other", "w", json!(4))
    })
}

/// Spec §3.5 step 6, a compaction (§5): the sibling's create, its writes
/// and sync, the rename over the store and the directory sync are each a cut
/// point. A handle opened before the compaction (another process, as far as
/// the store can tell) then writes: its swap check must move it to the new
/// inode, or its acknowledged commit lands in the unlinked file.
fn compaction(ctx: &Ctx<'_>, store: &FerroStore) -> Result<(), StoreError> {
    let other = ctx.op("open a handle before the compaction", || ctx.open_other())?;
    ctx.op("6 compaction", || store.compact())?;
    ctx.op("the older handle writes after the swap", || {
        other.put_doc("notes", "swapped", &json!({"by": "the older handle"}))
    })?;
    ctx.op("the compactor reads it", || {
        expect_doc(store, "notes", "swapped", json!({"by": "the older handle"}))
    })
}

fn tail(ctx: &Ctx<'_>, store: &FerroStore) -> Result<(), StoreError> {
    ctx.op("put writers/w0 again", || {
        store.put_doc("writers", "w0", &json!({"writer": "again"}))
    })?;
    ctx.op("delete notes/c", || store.delete_doc("notes", "c"))?;
    ctx.op("second checkpoint", || store.checkpoint())?;
    ctx.op("put notes/last", || {
        store.put_doc("notes", "last", &json!({"n": 5}))
    })
}

/// Spec §4.4: a commit whose WAL passes `max(snapshot_length, floor)`
/// checkpoints before its call returns, in a lock window of its own after
/// the commit's (so a cut can land between the two). The floor is lowered to 0, so the
/// threshold is the snapshot's length, and the commit is sized to pass it.
/// Then one more commit, on the new snapshot: a cut inside the checkpoint
/// leaves the first commit acknowledged, and this one refused.
fn auto_checkpoint(ctx: &Ctx<'_>, store: &FerroStore) -> Result<(), StoreError> {
    let snapshot = live_sb(store).snapshot_length;
    store.set_auto_checkpoint_floor(0);
    let big = "y".repeat(usize::try_from(snapshot).unwrap_or(0) + 600);
    ctx.op("put notes/auto (checkpoints)", || {
        store.put_doc("notes", "auto", &json!({"text": big}))
    })?;
    let sb = live_sb(store);
    if sb.wal_length != 0 || sb.snapshot_length <= snapshot {
        return Err(backend(format!(
            "the commit past the threshold did not checkpoint: {sb:?}"
        )));
    }
    ctx.op("put notes/after-auto", || {
        store.put_doc("notes", "after-auto", &json!({"n": 6}))
    })?;
    store.set_auto_checkpoint_floor(super::AUTO_CHECKPOINT_BYTES);
    Ok(())
}

/// The superblock of the state `store`'s cache holds.
fn live_sb(store: &FerroStore) -> Superblock {
    let super::Inner::Disk(disk) = &store.inner else {
        panic!("the crash workload runs on disk");
    };
    lock_mutex(&disk.cache).as_ref().expect("a cached state").sb
}

// ---- scenarios ------------------------------------------------------------------

struct Scenario {
    name: &'static str,
    /// What is on disk before the workload. Empty: a new store.
    initial: Image,
    steps: &'static [Step],
}

impl Scenario {
    fn new_file(&self) -> bool {
        self.initial.file(Path::new(STORE)).is_none()
    }
}

/// `open`, then every step. Stops at the first failure.
fn run_workload(
    scenario: &Scenario,
    fs: &dyn ContainerFs,
    path: &Path,
    ctx: &Ctx<'_>,
) -> Result<(), StoreError> {
    let store = ctx.op("open", || FerroStore::open_with(fs, path))?;
    for step in scenario.steps {
        (step.run)(ctx, &store).map_err(|err| backend(format!("{}: {err}", step.name)))?;
    }
    Ok(())
}

/// The snapshot a new store's `init_empty` writes after the header.
fn init_blob() -> Vec<u8> {
    encode_snapshot(0, &new_store_docs()).unwrap()
}

/// A format-1 container: slot 0 only, a snapshot and one WAL frame.
fn v1_image() -> Image {
    let mut docs = new_store_docs();
    docs.insert(
        "notes".to_string(),
        "old".to_string(),
        json!({"from": "v1"}).into(),
    );
    let blob = encode_snapshot(0, &docs).unwrap();
    let frame = encode_frame(
        1,
        &[Op::Put {
            collection: "notes".to_string(),
            id: "v1-wal".to_string(),
            doc: json!({"from": "v1 wal"}),
        }],
    )
    .unwrap();
    let snapshot_length = blob.len() as u64;
    let sb = Superblock {
        epoch: 1,
        snapshot_offset: HEADER_BYTES,
        snapshot_length,
        wal_offset: HEADER_BYTES + snapshot_length,
        wal_length: frame.len() as u64,
        snapshot_seq: 0,
    };
    let mut bytes = vec![0u8; HEADER_BYTES as usize];
    bytes[..SLOT_LEN].copy_from_slice(&encode_superblock_as(&sb, FORMAT_V1, 0, 0));
    bytes.extend_from_slice(&blob);
    bytes.extend_from_slice(&frame);
    Image::single(Path::new(STORE), bytes)
}

// ---- the harness ----------------------------------------------------------------

struct Reference {
    trace: Vec<IoOp>,
    labels: Vec<String>,
    /// `states[k]`: the state after `k` acknowledged operations.
    states: Vec<Logical>,
    /// `g` when step 5 runs: the three writers are operations `g..g + 3`.
    group: Option<usize>,
}

/// Open `image` on a fresh simulated disk and read its state.
fn decode(image: &Image) -> Logical {
    let fs = SimFs::new(image, None);
    let store = FerroStore::open_with(&fs, Path::new(STORE)).expect("reference image opens");
    Logical::read(&store).expect("reference image reads")
}

fn reference(scenario: &Scenario) -> Reference {
    let fs = SimFs::new(&scenario.initial, None);
    let ctx = Ctx::new(Some(&fs), &fs, Path::new(STORE));
    run_workload(scenario, &fs, Path::new(STORE), &ctx).expect("the uncut workload succeeds");
    let images = lock(&ctx.images).clone();
    assert_eq!(images.len(), ctx.acked());
    let mut states: Vec<Logical> = Vec::with_capacity(images.len() + 1);
    // Before anything: a new file reopens as the store `open` initialises.
    states.push(decode(if scenario.new_file() {
        &images[0]
    } else {
        &scenario.initial
    }));
    states.extend(images.iter().map(decode));
    for pair in states.windows(2) {
        assert!(
            pair[1].last_seq >= pair[0].last_seq,
            "{}: last_seq went backwards in the reference run",
            scenario.name
        );
    }
    let unsynced = lock(&ctx.unsynced_acks).clone();
    assert!(
        unsynced.is_empty(),
        "{}: acknowledged with writes not yet durable: {unsynced:?}",
        scenario.name
    );
    let trace = fs.trace();
    if let Err(why) = check_locks(&trace, &fs.lock_events()) {
        panic!("{}: {why}", scenario.name);
    }
    let labels = lock(&ctx.labels).clone();
    let group = labels.iter().position(|label| label.starts_with("writer "));
    if let Some(g) = group {
        let (before, whole) = (&states[g], &states[g + 3]);
        assert!(
            states[g + 1] == *whole && states[g + 2] == *whole && whole != before,
            "{}: the three writers were not acknowledged together, as one group",
            scenario.name
        );
    }
    Reference {
        trace,
        labels,
        states,
        group,
    }
}

/// A write to one of the two superblock slots.
fn is_slot_write(op: &IoOp) -> bool {
    matches!(op, IoOp::Write { offset, len, .. }
        if *len == SLOT_LEN as u64 && (*offset == 0 || *offset == SLOT_LEN as u64))
}

/// Spec §4.3: every write and sync of a container runs under the file's
/// `flock`, and the lock is not released between a write and the slot sync
/// that publishes it (a frame, a snapshot or a slot written but not yet
/// published). `events` are the trace positions of each lock and unlock.
fn check_locks(trace: &[IoOp], events: &[(usize, IoOp)]) -> Result<(), String> {
    #[derive(Default)]
    struct File {
        held: bool,
        /// A data write since the last published slot.
        unpublished: bool,
        /// A slot write not yet synced.
        slot: bool,
    }
    let mut files: BTreeMap<usize, File> = BTreeMap::new();
    let mut events = events.iter().peekable();
    for at in 0..=trace.len() {
        while let Some((_, event)) = events.next_if(|(when, _)| *when == at) {
            match event {
                IoOp::Lock { ino } => files.entry(*ino).or_default().held = true,
                IoOp::Unlock { ino } => {
                    let file = files.entry(*ino).or_default();
                    if file.unpublished || file.slot {
                        return Err(format!(
                            "the flock of ino{ino} was released at op #{at} with a write not \
                             yet published by a synced slot"
                        ));
                    }
                    file.held = false;
                }
                other => return Err(format!("{other} is not a lock event")),
            }
        }
        let Some(op) = trace.get(at) else { break };
        let ino = match op {
            IoOp::Write { ino, .. }
            | IoOp::SetLen { ino, .. }
            | IoOp::SyncData { ino }
            | IoOp::SyncAll { ino } => *ino,
            _ => continue,
        };
        let file = files.entry(ino).or_default();
        if !file.held {
            return Err(format!("op #{at} `{op}` ran without the file's flock"));
        }
        match op {
            IoOp::SyncData { .. } | IoOp::SyncAll { .. } if file.slot => {
                file.slot = false;
                file.unpublished = false;
            }
            _ if is_slot_write(op) => file.slot = true,
            IoOp::Write { .. } | IoOp::SetLen { .. } => file.unpublished = true,
            _ => {}
        }
    }
    Ok(())
}

/// Where step 5's publish is in `trace`: three frame writes, their
/// `sync_data`, the slot write and its `sync_data`. The index of the first.
fn group_publish(trace: &[IoOp]) -> Option<usize> {
    trace.windows(6).position(|ops| {
        ops[..3]
            .iter()
            .all(|op| matches!(op, IoOp::Write { .. }) && !is_slot_write(op))
            && matches!(ops[3], IoOp::SyncData { .. })
            && is_slot_write(&ops[4])
            && matches!(ops[5], IoOp::SyncData { .. })
    })
}

/// The two slots of a container image, where present.
fn slots(bytes: &[u8]) -> Vec<(usize, &[u8])> {
    (0..2)
        .filter(|index| bytes.len() >= (index + 1) * SLOT_LEN)
        .map(|index| (index, &bytes[index * SLOT_LEN..(index + 1) * SLOT_LEN]))
        .collect()
}

/// No two valid slots share an epoch; no slot is from a newer build.
/// Returns the live slot's container version.
fn check_slots(bytes: &[u8]) -> Result<Option<u32>, String> {
    let mut valid: Vec<(Superblock, &[u8])> = Vec::new();
    for (index, buf) in slots(bytes) {
        match parse_superblock(buf) {
            SlotRead::Valid(sb) => valid.push((sb, buf)),
            SlotRead::Invalid => {}
            SlotRead::Newer { version, .. } => {
                return Err(format!(
                    "slot {index} reads as from a newer build (v{version})"
                ));
            }
        }
    }
    if let [(a, _), (b, _)] = valid.as_slice()
        && a.epoch == b.epoch
    {
        return Err(format!("both slots hold epoch {}", a.epoch));
    }
    Ok(valid
        .iter()
        .max_by_key(|(sb, _)| sb.epoch)
        .and_then(|(_, buf)| layout::slot_version(buf)))
}

/// A cut during first-use init, before `open` returned. The file may be:
/// empty; re-initialisable (every header byte present is zero and the rest
/// is a prefix of the first snapshot: `ensure_superblock` re-runs
/// `init_empty`); or initialised (slot 0 valid at epoch 1 and slot 1 not).
fn check_init_cut(bytes: &[u8]) -> Result<&'static str, String> {
    if bytes.is_empty() {
        return Ok("empty");
    }
    let header = bytes.len().min(HEADER_BYTES as usize);
    let blob = init_blob();
    let tail = bytes.get(HEADER_BYTES as usize..).unwrap_or(&[]);
    if bytes[..header].iter().all(|byte| *byte == 0) && blob.starts_with(tail) {
        return Ok("re-initialisable");
    }
    let slot = |index: usize| {
        bytes
            .get(index * SLOT_LEN..(index + 1) * SLOT_LEN)
            .map(parse_superblock)
    };
    if let Some(SlotRead::Valid(sb)) = slot(0)
        && sb.epoch == 1
        && !matches!(slot(1), Some(SlotRead::Valid(_)))
        && tail.starts_with(&blob)
    {
        return Ok("initialised");
    }
    Err(format!(
        "a cut during first-use init left a {}-byte file that is not empty, \
         re-initialisable, or initialised",
        bytes.len()
    ))
}

/// Every check of spec §3.5 on one survivor image.
fn check_recovery(
    scenario: &Scenario,
    reference: &Reference,
    acked: usize,
    started: usize,
    image: &Image,
) -> Result<usize, String> {
    let path = Path::new(STORE);
    let first_use = scenario.new_file() && acked == 0;
    match image.file(path) {
        None if first_use => {}
        None => return Err("the store file is gone after `open` was acknowledged".into()),
        Some(bytes) => {
            check_slots(bytes)?;
            if first_use {
                check_init_cut(bytes)?;
            }
        }
    }
    check_one_store(image)?;
    let fs = SimFs::new(image, None);
    let store = FerroStore::open_with(&fs, path).map_err(|err| format!("reopen failed: {err}"))?;
    let version = store
        .format_version()
        .map_err(|err| format!("format_version failed: {err}"))?;
    if version.is_none() {
        return Err("format_version is None".into());
    }
    crate::check_format(version).map_err(|err| format!("format_version {version:?}: {err}"))?;
    let state = Logical::read(&store)?;
    let floor = &reference.states[acked];
    if state.last_seq < floor.last_seq {
        return Err(format!(
            "last_seq {} is below the acknowledged {}",
            state.last_seq, floor.last_seq
        ));
    }
    let Some(matched) = (acked..=started).find(|k| reference.states[*k] == state) else {
        return Err(format!(
            "the recovered state is no prefix from {acked} acknowledged to {started} started \
             operations (a lost acknowledged commit, or a partly applied one)\n      \
             recovered:    {}\n      acknowledged: {}",
            state.summary(),
            floor.summary()
        ));
    };
    let live = fs.live_image();
    let bytes = live.file(path).ok_or("no store file after reopen")?;
    let live_version = check_slots(bytes)?;
    if state.last_seq > reference.states[0].last_seq && live_version != Some(FORMAT_V2) {
        return Err(format!(
            "state past the start is published in a v{live_version:?} slot, not v{FORMAT_V2}"
        ));
    }
    // The recovered store takes a commit, and it reads back after a reopen.
    store
        .put_doc("probe", "after-crash", &json!(matched))
        .map_err(|err| format!("a commit after recovery failed: {err}"))?;
    drop(store);
    let again =
        FerroStore::open_with(&fs, path).map_err(|err| format!("reopen after probe: {err}"))?;
    let after = Logical::read(&again)?;
    let mut want = state;
    want.last_seq += 1;
    want.docs.insert(
        ("probe".to_string(), "after-crash".to_string()),
        json!(matched),
    );
    if after != want {
        return Err(format!(
            "the commit after recovery did not land cleanly\n      got:  {}\n      want: {}",
            after.summary(),
            want.summary()
        ));
    }
    Ok(matched)
}

/// Spec §5: after a cut at any step of a compaction, exactly one file is the
/// store. No name but the store's own names its inode (the sibling is either
/// still under its temp name, or renamed over the store and gone from the
/// temp name), and every leftover name is a `.compact-` temp of this store.
fn check_one_store(image: &Image) -> Result<(), String> {
    let path = Path::new(STORE);
    let Some(store_ino) = image.names.get(path) else {
        return Ok(());
    };
    for (name, ino) in &image.names {
        if name == path {
            continue;
        }
        if !is_compact_temp(name) {
            return Err(format!("an unexpected name {} survived", name.display()));
        }
        if ino == store_ino {
            return Err(format!(
                "the leftover temp {} names the store's inode: two names, one store",
                name.display()
            ));
        }
    }
    Ok(())
}

/// A name a compaction of [`STORE`] gives its sibling.
fn is_compact_temp(name: &Path) -> bool {
    name.parent() == Path::new(STORE).parent()
        && name
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with("store.fls.compact-"))
}

/// A cut inside step 5's group: all three writers acknowledged or none, and
/// the state the whole group or none of it.
fn check_group(
    reference: &Reference,
    acked: usize,
    started: usize,
    matched: usize,
) -> Result<(), String> {
    let Some(g) = reference.group else {
        return Ok(());
    };
    if started <= g || started > g + 3 {
        return Ok(());
    }
    if acked != g && acked != g + 3 {
        return Err(format!(
            "{} of the group's 3 writers were acknowledged at a cut inside the group",
            acked - g
        ));
    }
    let state = &reference.states[matched];
    if *state != reference.states[g] && *state != reference.states[g + 3] {
        return Err("a cut inside the group left part of the group".into());
    }
    if acked == g + 3 && *state != reference.states[g + 3] {
        return Err("the group was acknowledged and is not all there".into());
    }
    Ok(())
}

#[derive(Debug, Default)]
struct Report {
    cuts: usize,
    cases: usize,
    /// Cases with at least one torn write.
    torn_cases: usize,
    /// The most ops pending at one cut.
    max_pending: usize,
    sampled_cuts: usize,
    /// Cases left with a `.compact-` temp beside the store (§5).
    leftover_temps: usize,
    /// Cases whose store is a compaction's new inode, not the first file.
    swapped_stores: usize,
    failures: Vec<String>,
    failed_cases: usize,
}

fn run_scenario(scenario: &Scenario) -> Report {
    let reference = reference(scenario);
    let total = reference.trace.len();
    let path = PathBuf::from(STORE);
    let mut report = Report::default();
    for cut in 0..=total {
        let fs = SimFs::new(&scenario.initial, Some(cut));
        let ctx = Ctx::new(None, &fs, &path);
        let result = run_workload(scenario, &fs, &path, &ctx);
        let trace = fs.trace();
        assert_eq!(
            trace[..],
            reference.trace[..cut],
            "{}: the workload is not deterministic up to cut {cut}",
            scenario.name
        );
        if cut == total {
            assert!(
                result.is_ok(),
                "{}: the full budget run failed",
                scenario.name
            );
        } else {
            assert!(
                result.is_err() && fs.is_cut(),
                "{}: the workload ran past the cut at {cut}: {result:?}",
                scenario.name
            );
        }
        let (acked, started) = (ctx.acked(), ctx.started());
        let pending = fs.pending();
        let seed = SEED ^ (cut as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let (fates, sampled) = fate_vectors(&pending, FATE_CAP, seed);
        report.cuts += 1;
        report.sampled_cuts += usize::from(sampled);
        report.max_pending = report.max_pending.max(pending.len());
        for fate in &fates {
            report.cases += 1;
            report.torn_cases += usize::from(fate.iter().any(|fate| *fate >= 2));
            let image = fs.survivor(&pending, fate);
            report.leftover_temps +=
                usize::from(image.names.keys().any(|name| is_compact_temp(name)));
            report.swapped_stores += usize::from(
                image
                    .names
                    .get(Path::new(STORE))
                    .is_some_and(|ino| *ino != 0),
            );
            let checked = check_recovery(scenario, &reference, acked, started, &image)
                .and_then(|matched| check_group(&reference, acked, started, matched));
            if let Err(why) = checked {
                report.failed_cases += 1;
                if report.failures.len() < 6 {
                    report.failures.push(describe_failure(
                        scenario, &reference, cut, acked, &pending, fate, sampled, seed, &why,
                    ));
                }
            }
        }
    }
    report
}

#[allow(clippy::too_many_arguments)]
fn describe_failure(
    scenario: &Scenario,
    reference: &Reference,
    cut: usize,
    acked: usize,
    pending: &[super::crash_sim::Pending],
    fate: &[usize],
    sampled: bool,
    seed: u64,
    why: &str,
) -> String {
    let done = cut
        .checked_sub(1)
        .map(|last| format!("#{last} {}", reference.trace[last]))
        .unwrap_or_else(|| "nothing".into());
    let refused = reference
        .trace
        .get(cut)
        .map(|op| format!("#{cut} {op}"))
        .unwrap_or_else(|| "nothing (the workload finished)".into());
    let acked_label = acked
        .checked_sub(1)
        .and_then(|last| reference.labels.get(last))
        .map_or("nothing", String::as_str);
    let in_flight = reference
        .labels
        .get(acked)
        .map_or("nothing", String::as_str);
    let fates: Vec<String> = pending
        .iter()
        .zip(fate)
        .map(|(item, fate)| item.describe(*fate))
        .collect();
    let sample = if sampled {
        format!(" (sampled, seed {seed:#x})")
    } else {
        String::new()
    };
    format!(
        "[{}] power cut after {done}; refused {refused}\n    \
         acknowledged through `{acked_label}`, in flight `{in_flight}`\n    \
         unsynced ops{sample}: [{}]\n    FAILED: {why}",
        scenario.name,
        fates.join("; ")
    )
}

fn assert_clean(scenario: &Scenario, report: &Report, elapsed: Duration) {
    println!(
        "crash harness [{}]: {} cut points, {} cases ({} with a torn write; at most {} ops \
         pending at a cut; {} cuts sampled; {} with a leftover compaction temp; {} on a \
         compacted inode), {:.2?}",
        scenario.name,
        report.cuts,
        report.cases,
        report.torn_cases,
        report.max_pending,
        report.sampled_cuts,
        report.leftover_temps,
        report.swapped_stores,
        elapsed
    );
    assert!(
        report.failures.is_empty(),
        "crash harness [{}]: {} of {} cases failed. First failures:\n\n{}\n",
        scenario.name,
        report.failed_cases,
        report.cases,
        report.failures.join("\n\n")
    );
    assert!(
        elapsed < TIME_BUDGET,
        "crash harness [{}] took {elapsed:.2?}, over the {TIME_BUDGET:?} budget of spec §3.5",
        scenario.name
    );
}

/// Guards against a vacuous pass: if writes became durable at once, every cut
/// would yield one all-kept case and the harness would prove nothing.
fn assert_floors(report: &Report, min_cuts: usize) {
    assert!(
        report.cuts > min_cuts,
        "the workload is too small: {} cuts, need more than {min_cuts}: {report:?}",
        report.cuts
    );
    assert!(
        report.cases > report.cuts,
        "cuts are not forking into several cases: {} cases for {} cuts: {report:?}",
        report.cases,
        report.cuts
    );
    assert!(
        report.torn_cases > 0,
        "no case had a torn write: {} torn of {} cases: {report:?}",
        report.torn_cases,
        report.cases
    );
    assert!(
        report.max_pending >= 1,
        "no cut had an unsynced op: max pending {}: {report:?}",
        report.max_pending
    );
}

#[test]
fn every_cut_of_the_new_store_workload_recovers_to_an_acknowledged_prefix() {
    let start = Instant::now();
    let scenario = Scenario {
        name: "new store",
        initial: Image::default(),
        steps: WORKLOAD,
    };
    let report = run_scenario(&scenario);
    assert_floors(&report, 100);
    assert_clean(&scenario, &report, start.elapsed());
    // Step 6 is no vacuous pass: some cuts leave the sibling under its temp
    // name, and some leave it as the store.
    assert!(
        report.leftover_temps > 0 && report.swapped_stores > 0,
        "no cut exercised the compaction's rename: {report:?}"
    );
    // The cuts inside step 5 that §4.3 names exist, so the loop above tried
    // each: between the frames, after their sync, between the slot write and
    // its sync, and after it.
    let trace = reference(&scenario).trace;
    let at = group_publish(&trace).expect("step 5 publishes three frames with one slot");
    println!(
        "group publish at ops #{at}..#{}: cuts after frame 1 (#{}), frame 2 (#{}), the \
         frames' sync (#{}), the slot write (#{}), the slot sync (#{})",
        at + 5,
        at + 1,
        at + 2,
        at + 4,
        at + 5,
        at + 6
    );
}

#[test]
fn every_cut_of_a_v1_file_upgrade_recovers_to_an_acknowledged_prefix() {
    let start = Instant::now();
    let scenario = Scenario {
        name: "v1 upgrade",
        initial: v1_image(),
        steps: V1_WORKLOAD,
    };
    let report = run_scenario(&scenario);
    assert_floors(&report, 40);
    assert_clean(&scenario, &report, start.elapsed());
}

#[test]
fn every_cut_of_a_two_handle_workload_recovers_to_an_acknowledged_prefix() {
    let start = Instant::now();
    let scenario = Scenario {
        name: "two handles",
        initial: Image::default(),
        steps: TWO_HANDLE_WORKLOAD,
    };
    let report = run_scenario(&scenario);
    assert_floors(&report, 60);
    assert_clean(&scenario, &report, start.elapsed());
}

/// The scripted workload also runs on a real file through [`RealFs`], so the
/// harness never drifts from what production does.
#[test]
fn the_workload_runs_on_a_real_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.fls");
    let scenario = Scenario {
        name: "real file",
        initial: Image::default(),
        steps: WORKLOAD,
    };
    let ctx = Ctx::new(None, &RealFs, &path);
    run_workload(&scenario, &RealFs, &path, &ctx).unwrap();
    let store = FerroStore::open(&path).unwrap();
    let real = Logical::read(&store).unwrap();
    let simulated = reference(&scenario);
    assert_eq!(&real, simulated.states.last().unwrap());
}
