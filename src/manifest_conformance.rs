//! The manifest contract, run against `RedbStore` and `FerroStore`: the
//! export/import tests that were redb's, written once over [`ManifestSubject`].
//!
//! A schema version is checked as a raise from where the store started, not
//! as a number: a new redb store starts at `FORMAT_VERSION` and a new
//! FerroStore at the newest constant, and a raise only ever raises.

use crate::manifest::{LedgerRoot, Manifest, ManifestError, content_sha256};
use crate::{
    FORMAT_WITH_IMPORTS, FORMAT_WITH_LEDGER_ROOT, FORMAT_WITH_ROUTING, FerroStore, ImportReport,
    RedbStore,
};
use fl_core::ids::{GateId, Kind, ProjectId, seq_iri};
use fl_core::iri::Iri;
use fl_core::model::{GateKind, Regret, Selector, State, Transition};
use fl_core::routing::{Routes, RoutingMap};
use fl_core::store::{Bindings, Catalog, Handles, StoreError};

/// What the suite needs from a store beyond the traits.
pub(crate) trait ManifestSubject: Catalog + Handles + Bindings + Routes {
    fn export(
        &self,
        project: &ProjectId,
        commit: &str,
        at: u64,
        node: Option<&str>,
    ) -> Result<Manifest, ManifestError>;
    fn import(&self, m: &Manifest, root: &str) -> Result<ImportReport, ManifestError>;
    fn imported(&self, project: &ProjectId) -> Result<Option<String>, StoreError>;
    fn route(&self, project: &ProjectId, map: &RoutingMap) -> Result<(), StoreError>;
    fn routed(&self) -> Result<bool, StoreError>;
    fn holds(&self, id: &Iri) -> Result<bool, StoreError>;
    fn schema(&self) -> Result<Option<u64>, StoreError>;
}

impl ManifestSubject for RedbStore {
    fn export(
        &self,
        project: &ProjectId,
        commit: &str,
        at: u64,
        node: Option<&str>,
    ) -> Result<Manifest, ManifestError> {
        self.export_manifest(project, commit, at, node)
    }
    fn import(&self, m: &Manifest, root: &str) -> Result<ImportReport, ManifestError> {
        self.import_manifest(m, root)
    }
    fn imported(&self, project: &ProjectId) -> Result<Option<String>, StoreError> {
        self.imported_hash(project)
    }
    fn route(&self, project: &ProjectId, map: &RoutingMap) -> Result<(), StoreError> {
        self.set_routes(project, map)
    }
    fn routed(&self) -> Result<bool, StoreError> {
        self.holds_routing()
    }
    fn holds(&self, id: &Iri) -> Result<bool, StoreError> {
        self.owns(id)
    }
    fn schema(&self) -> Result<Option<u64>, StoreError> {
        self.format_version()
    }
}

impl ManifestSubject for FerroStore {
    fn export(
        &self,
        project: &ProjectId,
        commit: &str,
        at: u64,
        node: Option<&str>,
    ) -> Result<Manifest, ManifestError> {
        self.export_manifest(project, commit, at, node)
    }
    fn import(&self, m: &Manifest, root: &str) -> Result<ImportReport, ManifestError> {
        self.import_manifest(m, root)
    }
    fn imported(&self, project: &ProjectId) -> Result<Option<String>, StoreError> {
        self.imported_hash(project)
    }
    fn route(&self, project: &ProjectId, map: &RoutingMap) -> Result<(), StoreError> {
        self.set_routes(project, map)
    }
    fn routed(&self) -> Result<bool, StoreError> {
        self.holds_routing()
    }
    fn holds(&self, id: &Iri) -> Result<bool, StoreError> {
        self.owns(id)
    }
    fn schema(&self) -> Result<Option<u64>, StoreError> {
        self.format_version()
    }
}

/// A well-formed ledger root (ruling 20).
const ROOT_A: &str = "0123456789abcdef0123456789abcdef01234567";

fn kind() -> GateKind {
    GateKind::Command(fl_core::model::CommandSpec {
        program: "true".into(),
        args: vec![],
        delivery: fl_core::model::PopulationDelivery::Args,
        timeout_secs: 5,
        pass_codes: vec![0],
    })
}

fn selector() -> Selector {
    Selector::Glob {
        pattern: "**/*.rs".into(),
    }
}

/// An authoring store with one project: gates `fmt` and `lint`, and a
/// transition over both.
fn authoring<S: ManifestSubject, G>(
    fresh: &impl Fn() -> (S, G),
) -> (S, G, ProjectId, GateId, GateId) {
    let (s, guard) = fresh();
    let p = s.add_project("/author").unwrap();
    let g1 = s
        .add_gate(&p, "fmt", kind(), selector(), 1, "c1", "o")
        .unwrap();
    let g2 = s
        .add_gate(&p, "lint", kind(), selector(), 1, "c1", "o")
        .unwrap();
    s.add_transition(Transition {
        project: p.clone(),
        name: "ship".into(),
        from: State::Review,
        to: State::Done,
        regret: Regret::High,
        gates: vec![g1.clone(), g2.clone()],
    })
    .unwrap();
    (s, guard, p, g1, g2)
}

fn stamp(s: &impl ManifestSubject, g: &GateId, commit: &str) {
    let mut def = s.get_gate(g).unwrap().unwrap();
    def.last_pass_commit = Some(commit.into());
    s.update_gate(&def).unwrap();
}

/// The schema version `before` becomes when it is raised to `to`.
fn raised(before: Option<u64>, to: u64) -> Option<u64> {
    Some(before.unwrap_or(crate::FORMAT_VERSION).max(to))
}

/// Every case below, against stores from `fresh`.
pub(crate) fn manifest_contract<S: ManifestSubject, G>(fresh: impl Fn() -> (S, G)) {
    an_import_writes_every_item_under_its_own_iri_and_marks_the_project(&fresh);
    a_store_that_never_imported_reads_as_nothing_imported(&fresh);
    the_authoring_store_refuses_to_import_its_own_project(&fresh);
    an_importing_store_refuses_to_export(&fresh);
    an_imported_gate_can_earn_a_pass_mark_but_cannot_be_edited(&fresh);
    a_reimport_keeps_the_mark_of_an_unchanged_gate_and_drops_a_changed_ones(&fresh);
    an_import_refuses_a_manifest_it_did_not_verify(&fresh);
    a_fresh_import_onto_a_root_another_project_uses_is_refused(&fresh);
    a_reimport_from_another_checkout_moves_the_root_and_says_so(&fresh);
    a_reimport_mirrors_the_manifests_transitions(&fresh);
    an_export_carries_the_map_and_an_import_writes_it_raising_the_store_to_5(&fresh);
    an_import_that_would_drop_the_routing_map_is_refused(&fresh);
    an_import_never_puts_a_routed_project_beside_another(&fresh);
    imported_hash_of_a_project_this_store_never_held_is_not_owned(&fresh);
    an_import_raises_the_store_to_format_3(&fresh);
    a_reimport_that_removes_a_gate_is_refused_and_changes_nothing(&fresh);
    a_gate_held_locally_under_another_project_is_refused_and_writes_nothing(&fresh);
    an_import_never_lowers_the_format_of_a_store_that_records_a_ledger_root(&fresh);
    an_export_carries_the_root_of_the_repository_it_is_told_and_no_other(&fresh);
    an_import_records_the_manifests_ledger_root(&fresh);
    an_import_naming_another_root_for_a_known_repository_is_refused_and_writes_nothing(&fresh);
    an_imported_projects_routing_map_cannot_be_changed_here(&fresh);
    exporting_or_importing_against_the_wrong_kind_of_id_is_refused(&fresh);
}

fn an_import_writes_every_item_under_its_own_iri_and_marks_the_project<S: ManifestSubject, G>(
    fresh: &impl Fn() -> (S, G),
) {
    let (a, _ga, p, g1, g2) = authoring(fresh);
    let m = a.export(&p, "c1", 7, None).unwrap();
    let (b, _gb) = fresh();
    let report = b.import(&m, "/elsewhere").unwrap();
    assert_eq!((report.gates_added, report.transitions), (2, 1));
    assert_eq!(b.get_project(&p).unwrap().unwrap().root, "/elsewhere");
    assert_eq!(b.get_gate(&g1).unwrap(), a.get_gate(&g1).unwrap());
    assert!(b.holds(g2.iri()).unwrap());
    assert!(b.get_transition(&p, "ship").unwrap().is_some());
    assert_eq!(b.imported(&p).unwrap(), Some(m.content_sha256.clone()));
    assert!(b.handle_of(Kind::Gate, g1.iri()).unwrap().is_some());
}

fn a_store_that_never_imported_reads_as_nothing_imported<S: ManifestSubject, G>(
    fresh: &impl Fn() -> (S, G),
) {
    let (s, _g) = fresh();
    let p = s.add_project("/p").unwrap();
    assert_eq!(s.imported(&p).unwrap(), None);
}

fn the_authoring_store_refuses_to_import_its_own_project<S: ManifestSubject, G>(
    fresh: &impl Fn() -> (S, G),
) {
    let (a, _g, p, _, _) = authoring(fresh);
    let m = a.export(&p, "c1", 7, None).unwrap();
    let err = a.import(&m, "/author").unwrap_err();
    assert!(matches!(err, ManifestError::AuthoringStore(_)), "{err}");
}

fn an_importing_store_refuses_to_export<S: ManifestSubject, G>(fresh: &impl Fn() -> (S, G)) {
    let (a, _ga, p, _, _) = authoring(fresh);
    let (b, _gb) = fresh();
    b.import(&a.export(&p, "c1", 7, None).unwrap(), "/x")
        .unwrap();
    let err = b.export(&p, "c1", 8, None).unwrap_err();
    assert!(matches!(err, ManifestError::NotAuthoring(_)), "{err}");
}

fn an_imported_gate_can_earn_a_pass_mark_but_cannot_be_edited<S: ManifestSubject, G>(
    fresh: &impl Fn() -> (S, G),
) {
    let (a, _ga, p, g1, _) = authoring(fresh);
    let (b, _gb) = fresh();
    b.import(&a.export(&p, "c1", 7, None).unwrap(), "/x")
        .unwrap();
    stamp(&b, &g1, "c1");
    assert_eq!(
        b.get_gate(&g1)
            .unwrap()
            .unwrap()
            .last_pass_commit
            .as_deref(),
        Some("c1")
    );

    let mut edited = b.get_gate(&g1).unwrap().unwrap();
    edited.name = "renamed".into();
    let err = b.update_gate(&edited).unwrap_err();
    assert!(matches!(err, StoreError::Imported { .. }), "{err}");

    let err = b
        .add_gate(&p, "new", kind(), selector(), 1, "c1", "o")
        .unwrap_err();
    assert!(matches!(err, StoreError::Imported { .. }), "{err}");

    let err = b
        .add_transition(Transition {
            project: p.clone(),
            name: "other".into(),
            from: State::Todo,
            to: State::Doing,
            regret: Regret::Low,
            gates: vec![],
        })
        .unwrap_err();
    assert!(matches!(err, StoreError::Imported { .. }), "{err}");
}

fn a_reimport_keeps_the_mark_of_an_unchanged_gate_and_drops_a_changed_ones<
    S: ManifestSubject,
    G,
>(
    fresh: &impl Fn() -> (S, G),
) {
    let (a, _ga, p, g1, g2) = authoring(fresh);
    let (b, _gb) = fresh();
    b.import(&a.export(&p, "c1", 7, None).unwrap(), "/x")
        .unwrap();
    stamp(&b, &g1, "c1");
    stamp(&b, &g2, "c1");

    let mut changed = a.get_gate(&g2).unwrap().unwrap();
    changed.authored_at_commit = "c2".into();
    a.update_gate(&changed).unwrap();
    let g3 = a
        .add_gate(&p, "new", kind(), selector(), 1, "c2", "o")
        .unwrap();

    let report = b
        .import(&a.export(&p, "c2", 8, None).unwrap(), "/x")
        .unwrap();
    assert_eq!(
        (
            report.gates_added,
            report.gates_changed,
            report.gates_unchanged
        ),
        (1, 1, 1)
    );
    assert_eq!(
        b.get_gate(&g1)
            .unwrap()
            .unwrap()
            .last_pass_commit
            .as_deref(),
        Some("c1")
    );
    assert_eq!(b.get_gate(&g2).unwrap().unwrap().last_pass_commit, None);
    assert_eq!(b.get_gate(&g2).unwrap().unwrap().authored_at_commit, "c2");
    assert!(b.handle_of(Kind::Gate, g3.iri()).unwrap().is_some());
}

fn an_import_refuses_a_manifest_it_did_not_verify<S: ManifestSubject, G>(
    fresh: &impl Fn() -> (S, G),
) {
    let (a, _ga, p, _, _) = authoring(fresh);
    let mut m = a.export(&p, "c1", 7, None).unwrap();
    m.body.gates[0].name = "edited".into();
    let (b, _gb) = fresh();
    let err = b.import(&m, "/x").unwrap_err();
    assert!(matches!(err, ManifestError::HandEdited { .. }), "{err}");
    assert!(!b.holds(p.iri()).unwrap(), "nothing may be written");
}

fn a_fresh_import_onto_a_root_another_project_uses_is_refused<S: ManifestSubject, G>(
    fresh: &impl Fn() -> (S, G),
) {
    let (a, _ga, p, _, _) = authoring(fresh);
    let (b, _gb) = fresh();
    let local = b.add_project("/x").unwrap();
    let err = b
        .import(&a.export(&p, "c1", 7, None).unwrap(), "/x")
        .unwrap_err();
    assert!(
        matches!(err, ManifestError::RootTaken { ref other, .. } if *other == local),
        "{err}"
    );
    assert!(!b.holds(p.iri()).unwrap(), "nothing may be written");
}

fn a_reimport_from_another_checkout_moves_the_root_and_says_so<S: ManifestSubject, G>(
    fresh: &impl Fn() -> (S, G),
) {
    let (a, _ga, p, _, _) = authoring(fresh);
    let (b, _gb) = fresh();
    let m = a.export(&p, "c1", 7, None).unwrap();
    assert_eq!(b.import(&m, "/one").unwrap().root_moved, None);
    let report = b.import(&m, "/two").unwrap();
    assert_eq!(report.root_moved, Some(("/one".into(), "/two".into())));
    assert_eq!(b.get_project(&p).unwrap().unwrap().root, "/two");
}

fn a_reimport_mirrors_the_manifests_transitions<S: ManifestSubject, G>(
    fresh: &impl Fn() -> (S, G),
) {
    let (a, _ga, p, _, _) = authoring(fresh);
    let (b, _gb) = fresh();
    let first = a.export(&p, "c1", 7, None).unwrap();
    b.import(&first, "/x").unwrap();
    let mut body = first.body.clone();
    body.transitions.clear();
    let without = Manifest {
        content_sha256: content_sha256(&body).unwrap(),
        body,
    };
    let report = b.import(&without, "/x").unwrap();
    assert!(b.list_transitions(&p).unwrap().is_empty());
    assert_eq!(report.transitions_removed, vec!["ship".to_string()]);
}

// Routing spec §1.2: the importing machine routes as the authoring one.
fn an_export_carries_the_map_and_an_import_writes_it_raising_the_store_to_5<
    S: ManifestSubject,
    G,
>(
    fresh: &impl Fn() -> (S, G),
) {
    let (a, _ga, p, _, _) = authoring(fresh);
    a.route(&p, &RoutingMap::starting()).unwrap();
    let m = a.export(&p, "c1", 7, None).unwrap();
    assert_eq!(m.body.format_version, 3);
    assert_eq!(m.body.routing, Some(RoutingMap::starting()));
    let (b, _gb) = fresh();
    let before = b.schema().unwrap();
    let report = b.import(&m, "/x").unwrap();
    assert_eq!(report.areas, Some(5));
    assert_eq!(b.routes(&p).unwrap(), Some(RoutingMap::starting()));
    assert!(b.routed().unwrap());
    assert_eq!(b.schema().unwrap(), raised(before, FORMAT_WITH_ROUTING));
}

// Routing spec §1.2: an older checked-out manifest must not un-route one
// machine.
fn an_import_that_would_drop_the_routing_map_is_refused<S: ManifestSubject, G>(
    fresh: &impl Fn() -> (S, G),
) {
    let (a, _ga, p, _, _) = authoring(fresh);
    let unrouted = a.export(&p, "c1", 7, None).unwrap();
    a.route(&p, &RoutingMap::starting()).unwrap();
    let routed = a.export(&p, "c2", 8, None).unwrap();
    let (b, _gb) = fresh();
    b.import(&routed, "/x").unwrap();
    let err = b.import(&unrouted, "/x").unwrap_err();
    assert!(matches!(err, ManifestError::WouldDropRouting(_)), "{err}");
    assert!(
        err.to_string().contains("would un-route the project"),
        "{err}"
    );
    assert_eq!(
        b.routes(&p).unwrap(),
        Some(RoutingMap::starting()),
        "nothing written"
    );
    // A map that changed is imported: areas come and go (decision 15).
    a.route(&p, &RoutingMap::starting().without("design"))
        .unwrap();
    b.import(&a.export(&p, "c3", 9, None).unwrap(), "/x")
        .unwrap();
    assert_eq!(b.routes(&p).unwrap().unwrap().declared().len(), 4);
}

// Routing spec decision 20: an import keeps a routed store to one project,
// both ways round.
fn an_import_never_puts_a_routed_project_beside_another<S: ManifestSubject, G>(
    fresh: &impl Fn() -> (S, G),
) {
    let (a, _ga, p, _, _) = authoring(fresh);
    a.route(&p, &RoutingMap::starting()).unwrap();
    let routed = a.export(&p, "c1", 7, None).unwrap();
    let (b, _gb) = fresh();
    b.add_project("/elsewhere").unwrap();
    let err = b.import(&routed, "/x").unwrap_err().to_string();
    assert!(err.contains("needs a store of its own"), "{err}");
    assert!(!b.routed().unwrap(), "nothing written");
    let (c, _gc) = fresh();
    c.import(&routed, "/x").unwrap();
    let (o, _go, op, _, _) = authoring(fresh);
    let unrouted = o.export(&op, "c1", 7, None).unwrap();
    let err = c.import(&unrouted, "/y").unwrap_err().to_string();
    assert!(err.contains("needs a store of its own"), "{err}");
    assert_eq!(c.list_projects().unwrap().len(), 1);
}

fn imported_hash_of_a_project_this_store_never_held_is_not_owned<S: ManifestSubject, G>(
    fresh: &impl Fn() -> (S, G),
) {
    let (s, _g) = fresh();
    let err = s.imported(&ProjectId(seq_iri(42))).unwrap_err();
    assert!(matches!(err, StoreError::NotOwned { .. }), "{err:?}");
}

// ⚠ An older fl must refuse a store that holds an import rather than ignore it.
fn an_import_raises_the_store_to_format_3<S: ManifestSubject, G>(fresh: &impl Fn() -> (S, G)) {
    let (a, _ga, p, _, _) = authoring(fresh);
    let (b, _gb) = fresh();
    let before = b.schema().unwrap();
    b.import(&a.export(&p, "c1", 7, None).unwrap(), "/x")
        .unwrap();
    assert_eq!(b.schema().unwrap(), raised(before, FORMAT_WITH_IMPORTS));
    assert!(b.imported(&p).unwrap().is_some());
}

fn a_reimport_that_removes_a_gate_is_refused_and_changes_nothing<S: ManifestSubject, G>(
    fresh: &impl Fn() -> (S, G),
) {
    let (a, _ga, p, g1, g2) = authoring(fresh);
    let (b, _gb) = fresh();
    let first = a.export(&p, "c1", 7, None).unwrap();
    b.import(&first, "/x").unwrap();

    let mut body = first.body.clone();
    body.gates.retain(|g| g.id != g2);
    for t in &mut body.transitions {
        t.gates.retain(|g| *g != g2);
    }
    let shrunk = Manifest {
        content_sha256: content_sha256(&body).unwrap(),
        body,
    };
    let err = b.import(&shrunk, "/x").unwrap_err();
    assert!(
        matches!(err, ManifestError::WouldRemoveGate { ref id, ref name } if *id == g2 && name == "lint"),
        "{err}"
    );
    assert_eq!(b.imported(&p).unwrap(), Some(first.content_sha256));
    assert!(b.get_gate(&g1).unwrap().is_some());
}

fn a_gate_held_locally_under_another_project_is_refused_and_writes_nothing<
    S: ManifestSubject,
    G,
>(
    fresh: &impl Fn() -> (S, G),
) {
    let (a, _ga, p, _g1, _g2) = authoring(fresh);
    let m = a.export(&p, "c1", 7, None).unwrap();

    // `b` authors its own project locally and holds a gate under it; the
    // manifest below is rewritten to claim that same gate id for a project
    // `b` has never held.
    let (b, _gb) = fresh();
    let q = b.add_project("/local").unwrap();
    let local_gate = b
        .add_gate(&q, "local", kind(), selector(), 1, "c0", "o")
        .unwrap();

    let old_id = m.body.gates[0].id.clone();
    let mut body = m.body.clone();
    body.gates[0].id = local_gate.clone();
    for t in &mut body.transitions {
        for gid in &mut t.gates {
            if *gid == old_id {
                *gid = local_gate.clone();
            }
        }
    }
    let swapped = Manifest {
        content_sha256: content_sha256(&body).unwrap(),
        body,
    };

    let err = b.import(&swapped, "/x").unwrap_err();
    assert!(matches!(err, ManifestError::Inconsistent(_)), "{err}");
    assert!(!b.holds(p.iri()).unwrap(), "nothing may be written");
    assert_eq!(b.get_gate(&local_gate).unwrap().unwrap().project, q);
}

fn an_import_never_lowers_the_format_of_a_store_that_records_a_ledger_root<
    S: ManifestSubject,
    G,
>(
    fresh: &impl Fn() -> (S, G),
) {
    let (a, _ga, p, _, _) = authoring(fresh);
    let (b, _gb) = fresh();
    let before = b.schema().unwrap();
    b.set_ledger_root("R_9", "abc").unwrap();
    b.import(&a.export(&p, "c1", 7, None).unwrap(), "/x")
        .unwrap();
    assert_eq!(b.schema().unwrap(), raised(before, FORMAT_WITH_LEDGER_ROOT));
}

fn an_export_carries_the_root_of_the_repository_it_is_told_and_no_other<S: ManifestSubject, G>(
    fresh: &impl Fn() -> (S, G),
) {
    let (a, _ga, p, _, _) = authoring(fresh);
    a.set_ledger_root("R_1", ROOT_A).unwrap();
    let m = a.export(&p, "c1", 7, Some("R_1")).unwrap();
    assert_eq!(m.body.format_version, 2);
    assert_eq!(
        m.body.ledger_root,
        Some(LedgerRoot {
            repository_node_id: "R_1".into(),
            commit: ROOT_A.into(),
        })
    );
    for other in [None, Some("R_2")] {
        let m = a.export(&p, "c1", 7, other).unwrap();
        assert_eq!(
            (m.body.format_version, m.body.ledger_root),
            (1, None),
            "{other:?}"
        );
    }
}

fn an_import_records_the_manifests_ledger_root<S: ManifestSubject, G>(fresh: &impl Fn() -> (S, G)) {
    let (a, _ga, p, _, _) = authoring(fresh);
    a.set_ledger_root("R_1", ROOT_A).unwrap();
    let (b, _gb) = fresh();
    let before = b.schema().unwrap();
    b.import(&a.export(&p, "c1", 7, Some("R_1")).unwrap(), "/x")
        .unwrap();
    assert_eq!(b.ledger_root("R_1").unwrap().as_deref(), Some(ROOT_A));
    // ⚠ An import that records a root makes the store one an older fl
    // refuses, not one it opens and exports without the root.
    assert_eq!(b.schema().unwrap(), raised(before, FORMAT_WITH_LEDGER_ROOT));
}

fn an_import_naming_another_root_for_a_known_repository_is_refused_and_writes_nothing<
    S: ManifestSubject,
    G,
>(
    fresh: &impl Fn() -> (S, G),
) {
    let (a, _ga, p, _, _) = authoring(fresh);
    a.set_ledger_root("R_1", ROOT_A).unwrap();
    let (b, _gb) = fresh();
    b.set_ledger_root("R_1", "other").unwrap();
    let err = b
        .import(&a.export(&p, "c1", 7, Some("R_1")).unwrap(), "/x")
        .unwrap_err();
    assert!(
        matches!(
            err,
            ManifestError::Store(StoreError::LedgerRootChanged { .. })
        ),
        "{err}"
    );
    assert!(!b.holds(p.iri()).unwrap(), "nothing was written");
    assert_eq!(b.ledger_root("R_1").unwrap().as_deref(), Some("other"));
}

// The map is authored where the project is (routing spec §1.2): an importing
// store's copy comes from the manifest only.
fn an_imported_projects_routing_map_cannot_be_changed_here<S: ManifestSubject, G>(
    fresh: &impl Fn() -> (S, G),
) {
    let (a, _ga, p, _, _) = authoring(fresh);
    let (b, _gb) = fresh();
    b.import(&a.export(&p, "c1", 7, None).unwrap(), "/x")
        .unwrap();
    let err = b.route(&p, &RoutingMap::starting()).unwrap_err();
    assert!(matches!(err, StoreError::Imported { .. }), "{err:?}");
    assert!(
        err.to_string().contains("change the routing map of"),
        "{err}"
    );
}

fn exporting_or_importing_against_the_wrong_kind_of_id_is_refused<S: ManifestSubject, G>(
    fresh: &impl Fn() -> (S, G),
) {
    let (a, _ga, p, g1, _) = authoring(fresh);
    let err = a
        .export(&ProjectId(g1.iri().clone()), "c1", 7, None)
        .unwrap_err();
    assert!(
        matches!(err, ManifestError::Store(StoreError::WrongKind { .. })),
        "{err}"
    );
    let err = a
        .export(&ProjectId(seq_iri(77)), "c1", 7, None)
        .unwrap_err();
    assert!(
        matches!(err, ManifestError::Store(StoreError::NotOwned { .. })),
        "{err}"
    );
    // `b` holds a gate whose IRI the manifest names as its project.
    let m = a.export(&p, "c1", 7, None).unwrap();
    let (b, _gb) = fresh();
    let q = b.add_project("/q").unwrap();
    let held = b
        .add_gate(&q, "held", kind(), selector(), 1, "c0", "o")
        .unwrap();
    let mut body = m.body.clone();
    body.project = ProjectId(held.iri().clone());
    for g in &mut body.gates {
        g.project = body.project.clone();
    }
    for t in &mut body.transitions {
        t.project = body.project.clone();
    }
    let renamed = Manifest {
        content_sha256: content_sha256(&body).unwrap(),
        body,
    };
    let err = b.import(&renamed, "/x").unwrap_err();
    assert!(
        matches!(err, ManifestError::Store(StoreError::WrongKind { .. })),
        "{err}"
    );
    assert_eq!(b.list_projects().unwrap().len(), 1, "nothing was written");
}

#[test]
fn redb_store_meets_the_manifest_contract() {
    manifest_contract(|| {
        let dir = tempfile::tempdir().unwrap();
        let s = RedbStore::open(&dir.path().join("t.redb")).unwrap();
        (s, dir)
    });
}

#[test]
fn ferrostore_in_memory_meets_the_manifest_contract() {
    manifest_contract(|| (FerroStore::in_memory(), ()));
}

#[test]
fn ferrostore_on_disk_meets_the_manifest_contract() {
    manifest_contract(|| {
        let dir = tempfile::tempdir().unwrap();
        let s = FerroStore::open(dir.path().join("t.fls")).unwrap();
        (s, dir)
    });
}

/// A FerroStore migrated from a new redb store: it starts at redb's
/// `FORMAT_VERSION`, not the newest constant, so every format-raise case
/// raises it for real.
#[test]
fn ferrostore_migrated_from_a_new_redb_meets_the_manifest_contract() {
    manifest_contract(|| {
        let dir = tempfile::tempdir().unwrap();
        let redb = RedbStore::open(&dir.path().join("src.redb")).unwrap();
        let s = FerroStore::open(dir.path().join("t.fls")).unwrap();
        crate::migrate_redb_to_ferrostore(&redb, &s).unwrap();
        assert_eq!(s.format_version().unwrap(), Some(crate::FORMAT_VERSION));
        const { assert!(crate::FORMAT_VERSION < FORMAT_WITH_IMPORTS) };
        (s, dir)
    });
}
