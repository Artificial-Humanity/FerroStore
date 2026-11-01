//! Deleting vectors must not strand the vectors that were only reachable through them
//! (audit 2026-10-30, finding S-2).

use fl_store::{FerroStore, VectorMetric, VectorStorage};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }
}

fn point(rng: &mut Rng) -> Vec<f32> {
    (0..6).map(|_| rng.unit()).collect()
}

type Points = Vec<(String, Vec<f32>)>;

fn points(total: usize) -> Points {
    seeded_points(1, total)
}

fn seeded_points(seed: u64, total: usize) -> Points {
    let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
    (0..total)
        .map(|i| (format!("v{i:04}"), point(&mut rng)))
        .collect()
}

fn insert_all(store: &FerroStore, points: &Points, m: u32) {
    store
        .create_vector_collection("c", 6, VectorStorage::F32, VectorMetric::L2, m, 32, 16)
        .unwrap();
    for (id, values) in points {
        store.upsert_vector("c", id, values, None).unwrap();
    }
}

/// Count the vectors that a search for their own value does not return first. The search width
/// exceeds the collection, so a miss means the vector is unreachable in the graph, not a recall
/// shortfall.
fn unreachable_in(store: &FerroStore, kept: &Points) -> usize {
    kept.iter()
        .filter(|(id, values)| {
            let hits = store.search_vectors("c", values, 1, Some(1000)).unwrap();
            hits.first().map(|h| h.0.as_str()) != Some(id.as_str())
        })
        .count()
}

/// Insert `total` vectors, delete every `delete_every`th (starting with the first), and return
/// `(unreachable survivors, survivors)`.
fn unreachable_after_deleting(
    store: &FerroStore,
    total: usize,
    delete_every: usize,
    m: u32,
) -> (usize, usize) {
    unreachable_after_deleting_seeded(store, 1, total, delete_every, m)
}

fn unreachable_after_deleting_seeded(
    store: &FerroStore,
    seed: u64,
    total: usize,
    delete_every: usize,
    m: u32,
) -> (usize, usize) {
    let all = seeded_points(seed, total);
    insert_all(store, &all, m);
    let mut kept = Vec::new();
    for (i, (id, values)) in all.into_iter().enumerate() {
        if i % delete_every == 0 {
            store.delete_vector("c", &id).unwrap();
        } else {
            kept.push((id, values));
        }
    }
    (unreachable_in(store, &kept), kept.len())
}

/// The control: a collection built from the survivors alone, in the same order, with no delete
/// anywhere in its history.
fn unreachable_in_a_fresh_build(total: usize, delete_every: usize, m: u32) -> (usize, usize) {
    let kept: Points = points(total)
        .into_iter()
        .enumerate()
        .filter(|(i, _)| i % delete_every != 0)
        .map(|(_, p)| p)
        .collect();
    let store = FerroStore::in_memory();
    insert_all(&store, &kept, m);
    (unreachable_in(&store, &kept), kept.len())
}

/// The parameter cases for the delete tests: `(inserted, delete every nth, m)`.
const CASES: [(usize, usize, u32); 5] = [
    (300, 2, 4),
    (300, 3, 4),
    (300, 2, 8),
    (500, 2, 2),
    (800, 4, 4),
];

/// Dropping a deleted node's links without reconnecting its neighbours stranded the vectors that
/// were only reachable through it: at m=2, deleting every second of 500 vectors left 243 of the
/// 250 survivors unfindable. Reachability after deleting must stay close to that of a collection
/// built fresh from the very same survivors (which, at small `m`, is itself not perfect).
#[test]
fn vectors_stay_findable_after_neighbours_are_deleted() {
    for (total, every, m) in CASES {
        let (missing, kept) = unreachable_after_deleting(&FerroStore::in_memory(), total, every, m);
        let (control, control_kept) = unreachable_in_a_fresh_build(total, every, m);
        assert_eq!(
            kept, control_kept,
            "the control must hold the same survivors"
        );
        let allowed = control + kept / 20;
        assert!(
            missing <= allowed,
            "{missing} of {kept} survivors are unreachable after deletes (a fresh build of the \
             same survivors loses {control}, allowed {allowed}); n={total}, delete every \
             {every}th, m={m}"
        );
    }
}

/// The repair must not strand a node that its own re-linking cut off: an orphan used to drop
/// its links to everything outside its closest `max_links`, including a neighbour whose only
/// link it was. Each case below `(seed, inserted, delete every nth, m)` is one where that
/// happened; the number is how many survivors were still unreachable once an orphan stopped
/// dropping a link that would leave the other end with fewer than two. (Without that guard the repair
/// left 1, 3, 1, 2, 3 and 3 of them in these cases.)
#[test]
fn repairing_a_delete_does_not_cut_off_the_neighbour_it_relinks() {
    for (seed, total, every, m, allowed) in [
        (1u64, 60usize, 2usize, 2u32, 0usize),
        (1, 120, 2, 2, 0),
        (4, 120, 3, 3, 0),
        (6, 120, 2, 2, 1),
        (7, 60, 2, 2, 1),
        (3, 120, 2, 2, 2),
    ] {
        let (missing, kept) =
            unreachable_after_deleting_seeded(&FerroStore::in_memory(), seed, total, every, m);
        assert!(
            missing <= allowed,
            "{missing} of {kept} survivors unreachable (at most {allowed} expected); seed={seed}, \
             n={total}, delete every {every}th, m={m}"
        );
    }
}

/// The repaired graph must still satisfy every invariant a checkpoint re-validates on load
/// (symmetric links, at most `max_links`, sorted neighbours, a canonical entry point).
#[test]
fn a_collection_with_deletes_checkpoints_and_reopens() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.fls");
    let (missing_before, kept) = {
        let store = FerroStore::open(&path).unwrap();
        let out = unreachable_after_deleting(&store, 60, 2, 3);
        store.checkpoint().unwrap();
        out
    };
    let reopened =
        FerroStore::open(&path).expect("a graph repaired after deletes failed validation");
    let mut rng = Rng(7);
    let hits = reopened
        .search_vectors("c", &point(&mut rng), 5, Some(1000))
        .unwrap();
    assert_eq!(hits.len(), 5);
    assert!(missing_before <= kept / 5, "{missing_before} of {kept}");
}

#[test]
fn deleting_everything_then_reinserting_works() {
    let store = FerroStore::in_memory();
    store
        .create_vector_collection("c", 6, VectorStorage::F32, VectorMetric::Cosine, 4, 32, 16)
        .unwrap();
    let mut rng = Rng(11);
    let values: Vec<Vec<f32>> = (0..40).map(|_| point(&mut rng)).collect();
    for round in 0..2 {
        for (i, v) in values.iter().enumerate() {
            store
                .upsert_vector("c", &format!("v{i:02}"), v, None)
                .unwrap();
        }
        for i in (0..values.len()).rev() {
            store.delete_vector("c", &format!("v{i:02}")).unwrap();
            // The collection stays searchable at every size, down to empty.
            let hits = store.search_vectors("c", &values[0], 3, None).unwrap();
            // `i` vectors are left, so a search for 3 must return exactly min(i, 3) of them.
            assert_eq!(hits.len(), i.min(3), "round {round}, {i} left, {hits:?}");
        }
        assert!(
            store
                .search_vectors("c", &values[0], 3, None)
                .unwrap()
                .is_empty()
        );
    }
}
