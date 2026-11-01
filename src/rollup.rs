//! Ledger totals that do not hand the rows to the caller when the engine can sum them natively.
//! The engine still scans the whole table (every project, even for one project's totals), so
//! the gain is CPU and memory in the caller, not I/O.
//!
//! FerroStore has a columnar scan over its attempt and gate-run tables
//! ([`FerroStore::attempt_rollup`], [`FerroStore::gate_run_rollup`]). A caller that only needs
//! totals asks here: the ferro engine answers from the scan ([`TotalsSource::Columnar`]), and
//! any other engine, or a ferro scan that errors, sums the rows through the store traits
//! ([`TotalsSource::Rows`]). Both give the same numbers for the same data.
//!
//! What counts, for both: every attempt, whatever its status, adds one to `count` and its
//! duration, tokens and cost to the sums. A gate run counts under its verdict (`pass`, `fail`
//! or `error`), and adds its duration and cost whatever the verdict.
//!
//! One difference remains, and only for a corrupt store: with no scope, the columnar scan counts
//! an entry whose project (or gate) the store does not hold, and the row path, which walks the
//! project list, cannot see it. No `fl` code path appends such an entry.
//!
//! ⚠ The columnar scan sees only this store's own ledger tables. It cannot see entries that live
//! in a GitHub ledger, so a caller that reads a split ledger must not use it for totals.

use fl_core::ids::{GateId, Kind, ProjectId};
use fl_core::iri::Iri;
use fl_core::store::{Catalog, Ledger, StoreError};
use fl_core::verdict::Verdict;

use crate::AnyStore;

/// Sums over attempt ledger rows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AttemptTotals {
    pub count: u64,
    pub cost_usd_micros: u64,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub duration_ms: u64,
}

/// Counts and sums over gate-run ledger rows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GateRunTotals {
    pub pass: u64,
    pub fail: u64,
    pub error: u64,
    pub duration_ms: u64,
    pub cost_usd_micros: u64,
}

/// How a total was computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TotalsSource {
    /// The ferro engine's columnar scan. No row was loaded into the caller.
    Columnar,
    /// The rows were read through the store traits and summed here.
    Rows,
}

impl TotalsSource {
    /// The name the CLI and the Hub report.
    pub fn as_str(self) -> &'static str {
        match self {
            TotalsSource::Columnar => "columnar",
            TotalsSource::Rows => "rows",
        }
    }
}

/// Attempt totals for `project`, or for every project when `None`.
///
/// A ferro scan that fails (an unreadable table, an overflow of the scan's own checked sums, a
/// value that does not fit its i64 column) falls back to the rows; it is not an error here.
pub fn attempt_totals(
    store: &AnyStore,
    project: Option<&ProjectId>,
) -> Result<(AttemptTotals, TotalsSource), StoreError> {
    if let Some(ferro) = store.as_ferro()
        && scope_is(store, project.map(ProjectId::iri), Kind::Project)
        && let Ok(r) = ferro.attempt_rollup(project)
    {
        let totals = AttemptTotals {
            count: r.count,
            cost_usd_micros: r.cost_usd_micros,
            tokens_in: r.tokens_in,
            tokens_out: r.tokens_out,
            duration_ms: r.duration_ms,
        };
        return Ok((totals, TotalsSource::Columnar));
    }
    Ok((
        attempt_totals_from_rows(store, project)?,
        TotalsSource::Rows,
    ))
}

/// Gate-run totals for `gate`, or for every gate of every project when `None`.
///
/// Falls back to the rows exactly as [`attempt_totals`] does.
pub fn gate_run_totals(
    store: &AnyStore,
    gate: Option<&GateId>,
) -> Result<(GateRunTotals, TotalsSource), StoreError> {
    if let Some(ferro) = store.as_ferro()
        && scope_is(store, gate.map(GateId::iri), Kind::Gate)
        && let Ok(r) = ferro.gate_run_rollup(gate)
    {
        let totals = GateRunTotals {
            pass: r.pass,
            fail: r.fail,
            error: r.error,
            duration_ms: r.duration_ms,
            cost_usd_micros: r.cost_usd_micros,
        };
        return Ok((totals, TotalsSource::Columnar));
    }
    Ok((gate_run_totals_from_rows(store, gate)?, TotalsSource::Rows))
}

/// Whether the named scope is an id of `kind` this store holds (`None` is every row, so always).
///
/// The native rollup filters by the id's text and answers zero for an id the store never minted,
/// where the row path refuses it with the store's own error. A scope that is not known to be
/// valid therefore goes to the rows, so both engines give the same answer, error included.
fn scope_is(store: &AnyStore, id: Option<&Iri>, kind: Kind) -> bool {
    match id {
        None => true,
        Some(id) => matches!(Catalog::kind_of(store, id), Ok(k) if k == kind),
    }
}

/// The row path of [`attempt_totals`]: read the attempts through [`Ledger`] and sum them with
/// saturating arithmetic. `None` walks the project list, so an attempt whose project the store
/// does not list is not in the sum.
pub fn attempt_totals_from_rows<S: Ledger + Catalog>(
    store: &S,
    project: Option<&ProjectId>,
) -> Result<AttemptTotals, StoreError> {
    let mut totals = AttemptTotals::default();
    match project {
        Some(project) => add_attempts(&mut totals, store, project)?,
        None => {
            for p in store.list_projects()? {
                add_attempts(&mut totals, store, &p.id)?;
            }
        }
    }
    Ok(totals)
}

/// The row path of [`gate_run_totals`]: read the gate runs through [`Ledger`] and sum them with
/// saturating arithmetic. `None` walks every project's gates.
pub fn gate_run_totals_from_rows<S: Ledger + Catalog>(
    store: &S,
    gate: Option<&GateId>,
) -> Result<GateRunTotals, StoreError> {
    let mut totals = GateRunTotals::default();
    match gate {
        Some(gate) => add_gate_runs(&mut totals, store, gate)?,
        None => {
            for p in store.list_projects()? {
                for g in store.list_gates(&p.id)? {
                    add_gate_runs(&mut totals, store, &g.id)?;
                }
            }
        }
    }
    Ok(totals)
}

fn add_attempts<S: Ledger>(
    totals: &mut AttemptTotals,
    store: &S,
    project: &ProjectId,
) -> Result<(), StoreError> {
    for a in store.attempts(project)? {
        totals.count = totals.count.saturating_add(1);
        totals.cost_usd_micros = totals.cost_usd_micros.saturating_add(a.cost_usd_micros);
        totals.tokens_in = totals.tokens_in.saturating_add(a.tokens_in);
        totals.tokens_out = totals.tokens_out.saturating_add(a.tokens_out);
        totals.duration_ms = totals.duration_ms.saturating_add(a.duration_ms);
    }
    Ok(())
}

fn add_gate_runs<S: Ledger>(
    totals: &mut GateRunTotals,
    store: &S,
    gate: &GateId,
) -> Result<(), StoreError> {
    for run in store.gate_runs(gate)? {
        let slot = match run.verdict {
            Verdict::Pass { .. } => &mut totals.pass,
            Verdict::Fail { .. } => &mut totals.fail,
            Verdict::Error { .. } => &mut totals.error,
        };
        *slot = slot.saturating_add(1);
        totals.duration_ms = totals.duration_ms.saturating_add(run.duration_ms);
        totals.cost_usd_micros = totals.cost_usd_micros.saturating_add(run.cost_usd_micros);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BackendKind, StoreOptions, open_store_with};
    use fl_core::ids::{RecordId, seq_iri};
    use fl_core::log::{Attempt, AttemptStatus, GateRun, PathsTouched};
    use fl_core::model::{CommandSpec, GateKind, PopulationDelivery, Selector};
    use fl_core::verdict::FailReason;

    struct Engines {
        _dir: tempfile::TempDir,
        redb: AnyStore,
        ferro: AnyStore,
    }

    fn open(dir: &tempfile::TempDir, name: &str, kind: BackendKind) -> AnyStore {
        open_store_with(
            &dir.path().join(name),
            &StoreOptions {
                backend: Some(kind),
                ..StoreOptions::default()
            },
        )
        .unwrap()
    }

    fn engines() -> Engines {
        let dir = tempfile::tempdir().unwrap();
        let redb = open(&dir, "a", BackendKind::Redb);
        let ferro = open(&dir, "b", BackendKind::Ferro);
        assert_eq!(redb.kind(), BackendKind::Redb);
        assert_eq!(ferro.kind(), BackendKind::Ferro);
        Engines {
            _dir: dir,
            redb,
            ferro,
        }
    }

    fn gate_kind() -> GateKind {
        GateKind::Command(CommandSpec {
            program: "true".into(),
            args: vec![],
            delivery: PopulationDelivery::Args,
            timeout_secs: 5,
            pass_codes: vec![0],
        })
    }

    fn attempt(
        project: &ProjectId,
        status: AttemptStatus,
        cost: u64,
        tin: u64,
        tout: u64,
        ms: u64,
    ) -> Attempt {
        Attempt {
            id: None,
            at: None,
            project: project.clone(),
            record: RecordId(seq_iri(9)),
            adapter: "claude".into(),
            status,
            duration_ms: ms,
            tokens_in: tin,
            tokens_out: tout,
            cost_usd_micros: cost,
            paths_touched: PathsTouched::Counted(0),
            output_excerpt: None,
        }
    }

    fn run(gate: &GateId, verdict: Verdict, ms: u64, cost: u64) -> GateRun {
        GateRun {
            id: None,
            at: None,
            gate: gate.clone(),
            record: None,
            commit: "c1".into(),
            verdict,
            population: 1,
            output_excerpt: None,
            duration_ms: ms,
            cost_usd_micros: cost,
        }
    }

    struct Seeded {
        p1: ProjectId,
        p2: ProjectId,
        g1: GateId,
        g2: GateId,
    }

    /// The same data, through the same calls, in either engine.
    fn seed(store: &AnyStore) -> Seeded {
        let p1 = store.add_project("root-one").unwrap();
        let p2 = store.add_project("root-two").unwrap();
        let sel = || Selector::Glob {
            pattern: "**/*.rs".into(),
        };
        let g1 = store
            .add_gate(&p1, "fmt", gate_kind(), sel(), 1, "c1", "o")
            .unwrap();
        let g2 = store
            .add_gate(&p2, "lint", gate_kind(), sel(), 1, "c1", "o")
            .unwrap();
        // Every status contributes: a refused attempt counts and sums like the others.
        for (project, status, cost, tin, tout, ms) in [
            (&p1, AttemptStatus::Completed, 1_000, 10, 20, 100),
            (&p1, AttemptStatus::Timeout, 0, 5, 0, 900),
            (&p1, AttemptStatus::Refused, 7, 0, 0, 0),
            (&p2, AttemptStatus::Crashed, 250, 1, 2, 30),
            (&p2, AttemptStatus::Completed, 4, 3, 4, 5),
        ] {
            store
                .append_attempt(attempt(project, status, cost, tin, tout, ms))
                .unwrap();
        }
        for (gate, verdict, ms, cost) in [
            (&g1, Verdict::from_predicate(true, 3), 10, 1),
            (&g1, Verdict::from_predicate(true, 2), 20, 2),
            (&g1, Verdict::from_predicate(false, 2), 30, 3),
            (&g1, Verdict::error("boom"), 40, 4),
            // An empty population is a fail, not a pass.
            (&g2, Verdict::fail_for(FailReason::EmptyPopulation, 0), 5, 0),
            (&g2, Verdict::fail_for(FailReason::Stale, 1), 6, 0),
            (&g2, Verdict::from_predicate(true, 1), 7, 9),
        ] {
            store.append_gate_run(run(gate, verdict, ms, cost)).unwrap();
        }
        Seeded { p1, p2, g1, g2 }
    }

    #[test]
    fn identical_attempts_sum_identically_on_both_engines() {
        let e = engines();
        let a = seed(&e.redb);
        let b = seed(&e.ferro);
        for (pa, pb) in [
            (Some(&a.p1), Some(&b.p1)),
            (Some(&a.p2), Some(&b.p2)),
            (None, None),
        ] {
            let (ra, sa) = attempt_totals(&e.redb, pa).unwrap();
            let (rb, sb) = attempt_totals(&e.ferro, pb).unwrap();
            assert_eq!(ra, rb);
            assert_eq!((sa, sb), (TotalsSource::Rows, TotalsSource::Columnar));
        }
        let (all, _) = attempt_totals(&e.ferro, None).unwrap();
        assert_eq!(
            all,
            AttemptTotals {
                count: 5,
                cost_usd_micros: 1_261,
                tokens_in: 19,
                tokens_out: 26,
                duration_ms: 1_035,
            }
        );
        let (one, _) = attempt_totals(&e.redb, Some(&a.p1)).unwrap();
        assert_eq!(one.count, 3);
        assert_eq!(one.duration_ms, 1_000);
    }

    #[test]
    fn identical_gate_runs_sum_identically_on_both_engines() {
        let e = engines();
        let a = seed(&e.redb);
        let b = seed(&e.ferro);
        for (ga, gb) in [
            (Some(&a.g1), Some(&b.g1)),
            (Some(&a.g2), Some(&b.g2)),
            (None, None),
        ] {
            let (ra, sa) = gate_run_totals(&e.redb, ga).unwrap();
            let (rb, sb) = gate_run_totals(&e.ferro, gb).unwrap();
            assert_eq!(ra, rb);
            assert_eq!((sa, sb), (TotalsSource::Rows, TotalsSource::Columnar));
        }
        let (all, _) = gate_run_totals(&e.redb, None).unwrap();
        assert_eq!(
            all,
            GateRunTotals {
                pass: 3,
                fail: 3,
                error: 1,
                duration_ms: 118,
                cost_usd_micros: 19,
            }
        );
    }

    #[test]
    fn an_empty_store_totals_zero_on_both_engines() {
        let e = engines();
        for (store, source) in [
            (&e.redb, TotalsSource::Rows),
            (&e.ferro, TotalsSource::Columnar),
        ] {
            assert_eq!(
                attempt_totals(store, None).unwrap(),
                (AttemptTotals::default(), source)
            );
            assert_eq!(
                gate_run_totals(store, None).unwrap(),
                (GateRunTotals::default(), source)
            );
        }
    }

    #[test]
    fn saturating_sums_agree_and_a_ferro_scan_error_falls_back_to_rows() {
        let e = engines();
        let a = e.redb.add_project("r").unwrap();
        let b = e.ferro.add_project("r").unwrap();
        // u64::MAX does not fit the scan's i64 columns, so the native rollup refuses it.
        for (store, p) in [(&e.redb, &a), (&e.ferro, &b)] {
            for _ in 0..2 {
                store
                    .append_attempt(attempt(
                        p,
                        AttemptStatus::Completed,
                        u64::MAX,
                        1,
                        1,
                        u64::MAX,
                    ))
                    .unwrap();
            }
        }
        assert!(e.ferro.as_ferro().unwrap().attempt_rollup(None).is_err());
        let (ra, sa) = attempt_totals(&e.redb, None).unwrap();
        let (rb, sb) = attempt_totals(&e.ferro, None).unwrap();
        assert_eq!(ra, rb);
        assert_eq!((sa, sb), (TotalsSource::Rows, TotalsSource::Rows));
        assert_eq!(ra.count, 2);
        assert_eq!(ra.cost_usd_micros, u64::MAX);
        assert_eq!(ra.duration_ms, u64::MAX);
    }

    #[test]
    fn a_sum_that_overflows_inside_the_scan_falls_back_to_a_saturated_row_sum() {
        let e = engines();
        let p = e.ferro.add_project("r").unwrap();
        // Each cell fits an i64; three of them overflow a u64 sum.
        for _ in 0..3 {
            e.ferro
                .append_attempt(attempt(
                    &p,
                    AttemptStatus::Completed,
                    i64::MAX as u64,
                    0,
                    0,
                    0,
                ))
                .unwrap();
        }
        assert!(e.ferro.as_ferro().unwrap().attempt_rollup(None).is_err());
        let (t, source) = attempt_totals(&e.ferro, None).unwrap();
        assert_eq!(source, TotalsSource::Rows);
        assert_eq!((t.count, t.cost_usd_micros), (3, u64::MAX));
    }

    #[test]
    fn an_unknown_scope_is_the_same_error_on_both_engines() {
        let e = engines();
        let ghost_project = ProjectId(seq_iri(77));
        let ghost_gate = GateId(seq_iri(78));
        let mut wrongs = Vec::new();
        for store in [&e.redb, &e.ferro] {
            let p = store.add_project("r").unwrap();
            let a = attempt_totals(store, Some(&ghost_project)).unwrap_err();
            let g = gate_run_totals(store, Some(&ghost_gate)).unwrap_err();
            // A project id in a gate scope: whatever the row path answers, both engines answer it.
            let wrong = gate_run_totals(store, Some(&GateId(p.iri().clone()))).map(|(t, _)| t);
            wrongs.push(format!("{wrong:?}"));
            assert!(matches!(a, StoreError::NotOwned { .. }), "{a:?}");
            assert!(matches!(g, StoreError::NotOwned { .. }), "{g:?}");
        }
        assert_eq!(wrongs[0], wrongs[1]);
    }

    #[test]
    fn an_attempt_of_a_project_the_store_does_not_hold_is_the_known_difference() {
        let e = engines();
        let ghost = ProjectId(seq_iri(77));
        for store in [&e.redb, &e.ferro] {
            let p = store.add_project("r").unwrap();
            store
                .append_attempt(attempt(&p, AttemptStatus::Completed, 1, 1, 1, 1))
                .unwrap();
            store
                .append_attempt(attempt(&ghost, AttemptStatus::Completed, 10, 10, 10, 10))
                .unwrap();
        }
        // The scan counts the stray entry; the walk over the project list cannot see it.
        assert_eq!(attempt_totals(&e.redb, None).unwrap().0.count, 1);
        assert_eq!(attempt_totals(&e.ferro, None).unwrap().0.count, 2);
        // The row path on the ferro engine agrees with redb.
        let rows = attempt_totals_from_rows(&e.ferro, None).unwrap();
        assert_eq!(rows, attempt_totals(&e.redb, None).unwrap().0);
    }
}
