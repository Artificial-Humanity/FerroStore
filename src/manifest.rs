//! The committed manifest (GitHub tracker spec §4): a project's gates and
//! transitions, exported from the store that authors them, so another
//! machine can import them and resolve the gate an issue names.
//!
//! ⚠ The hash covers everything but itself. A manifest whose content does
//! not hash to the value it records was edited by hand, and is refused:
//! the store that authors the project is the only place a gate is authored.

use fl_core::ids::{GateId, ProjectId};
use fl_core::model::{GateDef, Transition};
use fl_core::routing::RoutingMap;
use fl_core::store::{Catalog, StoreError};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

/// Format 1: gates and transitions. Format 2 adds `ledger_root` (GitHub
/// ledger spec §6.1 step 4). Format 3 adds `routing` (routing spec §1.2),
/// with a ledger root or without. An export writes the oldest format that
/// holds what it carries ([`format_for`]), so a project with neither
/// still exports a manifest every older fl reads.
pub const MANIFEST_FORMAT_WITHOUT_LEDGER: u64 = 1;
pub const MANIFEST_FORMAT_WITH_LEDGER: u64 = 2;
pub const MANIFEST_FORMAT_WITH_ROUTING: u64 = 3;
/// The newest format this fl reads.
pub const MANIFEST_FORMAT: u64 = MANIFEST_FORMAT_WITH_ROUTING;
pub use fl_core::MANIFEST_PATH;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub commit: String,
    pub exported_at_unix: u64,
}

/// The first commit of the `fl/ledger` branch in the repository whose
/// `node_id` this names: every machine's anchor for the ledger's tamper
/// checks (GitHub ledger spec §3.5). The `node_id` lets an importing machine
/// key it without knowing the exporter's config.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LedgerRoot {
    pub repository_node_id: String,
    pub commit: String,
}

/// Everything the hash covers. ⚠ No store path and no project root: both
/// are specific to a machine, and the repository can be public (spec §4.1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Body {
    pub format_version: u64,
    pub provenance: Provenance,
    pub project: ProjectId,
    /// Sorted by id. Never carries a `last_pass_commit`: a pass mark is
    /// earned on one machine and means nothing on another.
    pub gates: Vec<GateDef>,
    /// Sorted by name.
    pub transitions: Vec<Transition>,
    /// Format 2 only. ⚠ Skipped when absent, so a format-1 body serializes —
    /// and hashes — byte for byte as it did before format 2 existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ledger_root: Option<LedgerRoot>,
    /// Format 3 only (routing spec §1.2). ⚠ Skipped when absent, so a body
    /// of format 1 or 2 serializes — and hashes — byte for byte as before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing: Option<RoutingMap>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub body: Body,
    pub content_sha256: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error(
        "the manifest is not valid: {0}. Export it again from the store that authors the project"
    )]
    Parse(String),
    #[error(
        "the manifest is format {found}, and this version of fl reads formats \
         {MANIFEST_FORMAT_WITHOUT_LEDGER} to {MANIFEST_FORMAT}. A newer fl wrote it: upgrade fl \
         to read it"
    )]
    Format { found: u64 },
    #[error(
        "the manifest was edited by hand: its content hashes to {found}, but it records \
         {recorded}. Gates are authored in the store that owns the project; export again \
         from there"
    )]
    HandEdited { recorded: String, found: String },
    #[error(
        "the manifest is inconsistent: {0}. Export it again from the store that authors the \
         project"
    )]
    Inconsistent(String),
    #[error("this store authors project {0}, so it cannot import it: it already holds the source")]
    AuthoringStore(ProjectId),
    #[error(
        "this store imported project {0} from a manifest, so it cannot export it. Export \
         from the store that authors it"
    )]
    NotAuthoring(ProjectId),
    #[error(
        "the manifest no longer lists gate `{name}` ({id}), which this store holds. A \
         re-import would remove a neighbour from every later verify, so it is refused. The \
         checked-out manifest may be older than the one this store imported — check out a \
         commit whose manifest lists the gate. If the gate really was removed where the \
         project is authored, restore it there and export again"
    )]
    WouldRemoveGate { id: GateId, name: String },
    #[error(
        "the manifest has no routing map, and this store imported one for project {0}. \
         Importing it would un-route the project on this machine alone, so it is refused. The \
         checked-out manifest may be older than the one this store imported: check out a \
         commit whose manifest has the routing map"
    )]
    WouldDropRouting(ProjectId),
    #[error(
        "project {other} in this store already uses the root {root}. Import into a separate \
         store with `fl --db <path> manifest import` — but doing that on the machine that \
         authors this project makes its IRIs ambiguous between the two stores — or use the \
         existing project instead of importing"
    )]
    RootTaken { root: String, other: ProjectId },
    #[error(transparent)]
    Store(#[from] StoreError),
}

/// How a store's gate compares with the manifest's copy of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Currency {
    Current,
    Differs,
    Absent,
}

/// SHA-256 of the body's compact JSON, as lowercase hex. serde_json writes
/// struct fields in declaration order and the body holds no maps, so equal
/// bodies always produce equal bytes.
pub fn content_sha256(body: &Body) -> Result<String, ManifestError> {
    let bytes = serde_json::to_vec(body).map_err(|e| ManifestError::Parse(e.to_string()))?;
    Ok(Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// The format of a body carrying a ledger root and a routing map, or not:
/// the oldest that holds both (routing spec §1.2).
pub fn format_for(ledger_root: bool, routing: bool) -> u64 {
    match (ledger_root, routing) {
        (_, true) => MANIFEST_FORMAT_WITH_ROUTING,
        (true, false) => MANIFEST_FORMAT_WITH_LEDGER,
        (false, false) => MANIFEST_FORMAT_WITHOUT_LEDGER,
    }
}

/// Every gate and transition of `project`, with pass marks cleared, the
/// ledger root when the project's repository has one, and the routing map
/// when the project has one.
pub fn export(
    catalog: &dyn Catalog,
    project: &ProjectId,
    commit: &str,
    exported_at_unix: u64,
    ledger_root: Option<LedgerRoot>,
    routing: Option<RoutingMap>,
) -> Result<Manifest, ManifestError> {
    if catalog.get_project(project)?.is_none() {
        return Err(ManifestError::Inconsistent(format!(
            "{project} is held by this store, but it is not a project"
        )));
    }
    let mut gates = catalog.list_gates(project)?;
    for g in &mut gates {
        g.last_pass_commit = None;
    }
    gates.sort_by(|a, b| a.id.cmp(&b.id));
    let mut transitions = catalog.list_transitions(project)?;
    transitions.sort_by(|a, b| a.name.cmp(&b.name));
    let body = Body {
        format_version: format_for(ledger_root.is_some(), routing.is_some()),
        provenance: Provenance {
            commit: commit.to_string(),
            exported_at_unix,
        },
        project: project.clone(),
        gates,
        transitions,
        ledger_root,
        routing,
    };
    let content_sha256 = content_sha256(&body)?;
    let m = Manifest {
        body,
        content_sha256,
    };
    // An export never writes a file its own parse would refuse — for
    // example a transition naming a gate of another project.
    m.check_consistent()?;
    Ok(m)
}

impl Manifest {
    /// Pretty JSON with a trailing newline, for a file a person reviews in a
    /// diff. The hash is over the compact form of the body, so layout is free.
    pub fn to_json(&self) -> String {
        let mut s = serde_json::to_string_pretty(self).expect("a manifest always serializes");
        s.push('\n');
        s
    }

    /// Parse, then refuse anything that is not exactly what an export wrote.
    pub fn parse(text: &str) -> Result<Manifest, ManifestError> {
        // The format is read first, loosely, so a future format is named as
        // one rather than reported as whatever field it renamed.
        let loose: serde_json::Value =
            serde_json::from_str(text).map_err(|e| ManifestError::Parse(e.to_string()))?;
        match loose
            .pointer("/body/format_version")
            .and_then(|v| v.as_u64())
        {
            Some(v) if (MANIFEST_FORMAT_WITHOUT_LEDGER..=MANIFEST_FORMAT).contains(&v) => {}
            Some(found) => return Err(ManifestError::Format { found }),
            None => {
                return Err(ManifestError::Parse(
                    "it has no `body.format_version`".into(),
                ));
            }
        }
        let m: Manifest =
            serde_json::from_value(loose).map_err(|e| ManifestError::Parse(e.to_string()))?;
        m.verify()?;
        Ok(m)
    }

    /// The hash and the internal consistency. `parse` calls it, and so does
    /// the store's import: a `Manifest` can be built by hand (its fields are
    /// public), and the store must not trust one it did not check.
    pub fn verify(&self) -> Result<(), ManifestError> {
        let found = content_sha256(&self.body)?;
        if found != self.content_sha256 {
            return Err(ManifestError::HandEdited {
                recorded: self.content_sha256.clone(),
                found,
            });
        }
        self.check_consistent()
    }

    fn check_consistent(&self) -> Result<(), ManifestError> {
        let f = self.body.format_version;
        let want = format_for(self.body.ledger_root.is_some(), self.body.routing.is_some());
        if f != want {
            let carried = match (&self.body.ledger_root, &self.body.routing) {
                (Some(_), Some(_)) => "a ledger root and a routing map",
                (Some(_), None) => "a ledger root",
                (None, Some(_)) => "a routing map",
                (None, None) => "neither a ledger root nor a routing map",
            };
            return Err(ManifestError::Inconsistent(format!(
                "it is format {f}, but what it carries — {carried} — is format {want}"
            )));
        }
        if let Some(root) = &self.body.ledger_root {
            fl_core::ledger_root_shape(&root.repository_node_id, &root.commit).map_err(|why| {
                ManifestError::Inconsistent(format!("its ledger root cannot be one: {why}"))
            })?;
        }
        if let Some(map) = &self.body.routing {
            map.check().map_err(|why| {
                ManifestError::Inconsistent(format!("its routing map is not valid: {why}"))
            })?;
        }
        let p = &self.body.project;
        let mut ids = BTreeSet::new();
        for g in &self.body.gates {
            if g.project != *p {
                return Err(ManifestError::Inconsistent(format!(
                    "gate {} belongs to project {}, not {p}",
                    g.id, g.project
                )));
            }
            if g.last_pass_commit.is_some() {
                return Err(ManifestError::Inconsistent(format!(
                    "gate {} carries a pass mark; pass marks are local to each machine and \
                     are never exported",
                    g.id
                )));
            }
            if !ids.insert(g.id.clone()) {
                return Err(ManifestError::Inconsistent(format!(
                    "gate {} is listed twice",
                    g.id
                )));
            }
        }
        let mut names = BTreeSet::new();
        for t in &self.body.transitions {
            if t.project != *p {
                return Err(ManifestError::Inconsistent(format!(
                    "transition `{}` belongs to project {}, not {p}",
                    t.name, t.project
                )));
            }
            if !names.insert(t.name.clone()) {
                return Err(ManifestError::Inconsistent(format!(
                    "transition `{}` is listed twice",
                    t.name
                )));
            }
            for g in &t.gates {
                if !ids.contains(g) {
                    return Err(ManifestError::Inconsistent(format!(
                        "transition `{}` names gate {g}, which the manifest does not list",
                        t.name
                    )));
                }
            }
        }
        Ok(())
    }

    /// Compare a store's gate with the manifest's copy, ignoring the pass
    /// mark (spec §4.3).
    pub fn currency_of(&self, def: &GateDef) -> Currency {
        match self.body.gates.iter().find(|g| g.id == def.id) {
            None => Currency::Absent,
            Some(g) => {
                let mut bare = def.clone();
                bare.last_pass_commit = None;
                if *g == bare {
                    Currency::Current
                } else {
                    Currency::Differs
                }
            }
        }
    }
}

/// Why [`ensure_import_current`] refused. Each message names what to run.
#[derive(Debug, thiserror::Error)]
pub enum ImportCheckError {
    #[error("{0} is held by this store, but it is not a project")]
    NotAProject(ProjectId),
    #[error("could not look for {path}: {cause}")]
    Look { path: String, cause: String },
    #[error(
        "there is no manifest at {path}. It comes from the machine that authors the project: \
         restore it from git (pull, or check it out), then run `fl manifest import`"
    )]
    Missing { path: String },
    #[error("could not read the manifest at {path}: {cause}")]
    Unreadable { path: String, cause: String },
    #[error("{path} was refused: {cause}")]
    Refused {
        path: String,
        cause: Box<ManifestError>,
    },
    #[error(
        "the manifest at {path} changed since this store imported it (imported {recorded}, \
         now {now}). Run `fl manifest import` before running its gates."
    )]
    Stale {
        path: String,
        recorded: String,
        now: String,
    },
    #[error(transparent)]
    Store(#[from] StoreError),
}

/// A store that records which manifest, if any, each project was imported
/// from. Implemented by every engine, so [`ensure_import_current`] serves the
/// CLI's [`crate::AnyStore`] and the hub's [`crate::RedbStore`] alike.
pub trait ImportRecord: Catalog {
    /// The hash of the manifest `project` was imported from, or `None` if this
    /// store authors it.
    fn imported_hash(&self, project: &ProjectId) -> Result<Option<String>, StoreError>;
}

impl ImportRecord for crate::RedbStore {
    fn imported_hash(&self, project: &ProjectId) -> Result<Option<String>, StoreError> {
        crate::RedbStore::imported_hash(self, project)
    }
}

impl ImportRecord for crate::AnyStore {
    fn imported_hash(&self, project: &ProjectId) -> Result<Option<String>, StoreError> {
        crate::AnyStore::imported_hash(self, project)
    }
}

/// Spec §4.5: on an importing machine, the working tree's manifest must be the one this store
/// imported. A project this store authors passes.
pub fn ensure_import_current<S: ImportRecord + ?Sized>(
    store: &S,
    project: &ProjectId,
) -> Result<(), ImportCheckError> {
    let Some(recorded) = store.imported_hash(project)? else {
        return Ok(());
    };
    let Some(p) = store.get_project(project)? else {
        return Err(ImportCheckError::NotAProject(project.clone()));
    };
    let path = std::path::Path::new(&p.root).join(MANIFEST_PATH);
    let shown = path.display().to_string();
    match path.try_exists() {
        Ok(true) => {}
        Ok(false) => return Err(ImportCheckError::Missing { path: shown }),
        Err(e) => {
            return Err(ImportCheckError::Look {
                path: shown,
                cause: e.to_string(),
            });
        }
    }
    let text = std::fs::read_to_string(&path).map_err(|e| ImportCheckError::Unreadable {
        path: shown.clone(),
        cause: e.to_string(),
    })?;
    let m = Manifest::parse(&text).map_err(|cause| ImportCheckError::Refused {
        path: shown.clone(),
        cause: Box::new(cause),
    })?;
    if m.content_sha256 != recorded {
        return Err(ImportCheckError::Stale {
            path: shown,
            recorded,
            now: m.content_sha256,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fl_core::MemStore;
    use fl_core::model::{CommandSpec, GateKind, PopulationDelivery, Regret, Selector, State};

    fn kind(program: &str) -> GateKind {
        GateKind::Command(CommandSpec {
            program: program.into(),
            args: vec![],
            delivery: PopulationDelivery::Args,
            timeout_secs: 5,
            pass_codes: vec![0],
        })
    }

    fn glob() -> Selector {
        Selector::Glob {
            pattern: "**/*.rs".into(),
        }
    }

    /// Two projects in one store; `p` has two gates and a transition.
    fn store() -> (MemStore, ProjectId, GateId, GateId) {
        let s = MemStore::default();
        let p = s.add_project("/p").unwrap();
        let other = s.add_project("/q").unwrap();
        s.add_gate(&other, "other", kind("true"), glob(), 1, "c0", "o")
            .unwrap();
        let g1 = s
            .add_gate(&p, "fmt", kind("true"), glob(), 1, "c1", "o")
            .unwrap();
        let g2 = s
            .add_gate(&p, "lint", kind("/opt/tools/lint"), glob(), 1, "c1", "o")
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
        (s, p, g1, g2)
    }

    #[test]
    fn an_export_round_trips_through_its_file_form() {
        let (s, p, _, _) = store();
        let m = export(&s, &p, "abc", 7, None, None).unwrap();
        assert_eq!(Manifest::parse(&m.to_json()).unwrap(), m);
    }

    #[test]
    fn an_export_carries_only_the_named_project() {
        let (s, p, g1, g2) = store();
        let m = export(&s, &p, "abc", 7, None, None).unwrap();
        let ids: Vec<_> = m.body.gates.iter().map(|g| g.id.clone()).collect();
        assert_eq!(ids, vec![g1, g2]);
        assert_eq!(m.body.transitions.len(), 1);
    }

    #[test]
    fn a_pass_mark_is_never_exported() {
        let (s, p, g1, _) = store();
        let mut def = s.get_gate(&g1).unwrap().unwrap();
        def.last_pass_commit = Some("abc".into());
        s.update_gate(&def).unwrap();
        let m = export(&s, &p, "abc", 7, None, None).unwrap();
        assert!(m.body.gates.iter().all(|g| g.last_pass_commit.is_none()));
    }

    #[test]
    fn a_hand_edit_is_refused() {
        let (s, p, _, _) = store();
        let text = export(&s, &p, "abc", 7, None, None)
            .unwrap()
            .to_json()
            .replace("\"name\": \"fmt\"", "\"name\": \"fmt2\"");
        assert!(text.contains("fmt2"), "the edit must have landed");
        let err = Manifest::parse(&text).unwrap_err();
        assert!(matches!(err, ManifestError::HandEdited { .. }), "{err}");
    }

    #[test]
    fn a_future_format_is_named_as_one() {
        let (s, p, _, _) = store();
        let text = export(&s, &p, "abc", 7, None, None)
            .unwrap()
            .to_json()
            .replace("\"format_version\": 1", "\"format_version\": 9");
        let err = Manifest::parse(&text).unwrap_err();
        assert!(matches!(err, ManifestError::Format { found: 9 }), "{err}");
        assert!(err.to_string().contains("upgrade fl"), "{err}");
    }

    /// `Body` and `Manifest` as format 1 was declared before format 2
    /// existed, frozen here: an older fl parses and hashes with exactly this.
    #[derive(Serialize, Deserialize)]
    #[serde(deny_unknown_fields)]
    #[allow(dead_code)]
    struct OlderBody {
        format_version: u64,
        provenance: Provenance,
        project: ProjectId,
        gates: Vec<GateDef>,
        transitions: Vec<Transition>,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    #[allow(dead_code)]
    struct OlderManifest {
        body: OlderBody,
        content_sha256: String,
    }

    // ⚠ A project with no GitHub ledger must keep exporting exactly what an
    // older fl reads: same fields, same bytes, same hash.
    #[test]
    fn a_manifest_with_no_ledger_root_is_format_1_and_an_older_fl_reads_it() {
        let (s, p, _, _) = store();
        let m = export(&s, &p, "abc", 7, None, None).unwrap();
        assert_eq!(m.body.format_version, 1);
        let text = m.to_json();
        assert!(!text.contains("ledger_root"), "{text}");
        let old: OlderManifest =
            serde_json::from_str(&text).expect("an older fl parses a format-1 export");
        let bytes = serde_json::to_vec(&old.body).unwrap();
        let hash: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(hash, old.content_sha256, "an older fl's hash check passes");
    }

    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

    fn root() -> LedgerRoot {
        LedgerRoot {
            repository_node_id: "R_1".into(),
            commit: COMMIT.into(),
        }
    }

    // Ruling 20: a root that cannot be one is refused by the consistency
    // check, so by parse, by import, and by export alike.
    #[test]
    fn a_ledger_root_that_cannot_be_one_is_refused_even_with_a_correct_hash() {
        let (s, p, _, _) = store();
        for (node, commit) in [("R_1", "abc123"), ("", COMMIT), ("1R", COMMIT)] {
            let mut m = export(&s, &p, "abc", 7, Some(root()), None).unwrap();
            m.body.ledger_root = Some(LedgerRoot {
                repository_node_id: node.into(),
                commit: commit.into(),
            });
            let err = Manifest::parse(&rehashed(m).to_json()).unwrap_err();
            assert!(
                matches!(err, ManifestError::Inconsistent(ref m) if m.contains("ledger root")),
                "{node} {commit}: {err}"
            );
            let err = export(
                &s,
                &p,
                "abc",
                7,
                Some(LedgerRoot {
                    repository_node_id: node.into(),
                    commit: commit.into(),
                }),
                None,
            )
            .unwrap_err();
            assert!(matches!(err, ManifestError::Inconsistent(_)), "{err}");
        }
    }

    #[test]
    fn a_manifest_with_a_ledger_root_is_format_2_and_round_trips() {
        let (s, p, _, _) = store();
        let m = export(&s, &p, "abc", 7, Some(root()), None).unwrap();
        assert_eq!(m.body.format_version, 2);
        let back = Manifest::parse(&m.to_json()).unwrap();
        assert_eq!(back.body.ledger_root, Some(root()));
        assert_eq!(back, m);
    }

    #[test]
    fn a_root_on_format_1_or_no_root_on_format_2_is_refused() {
        let (s, p, _, _) = store();
        let mut m = export(&s, &p, "abc", 7, None, None).unwrap();
        m.body.ledger_root = Some(root());
        let err = rehashed(m).verify().unwrap_err();
        assert!(matches!(err, ManifestError::Inconsistent(_)), "{err}");

        let mut m = export(&s, &p, "abc", 7, Some(root()), None).unwrap();
        m.body.ledger_root = None;
        let err = rehashed(m).verify().unwrap_err();
        assert!(matches!(err, ManifestError::Inconsistent(_)), "{err}");
    }

    fn routes() -> fl_core::RoutingMap {
        fl_core::RoutingMap::starting()
    }

    // Routing spec §1.2: the oldest format that holds what the manifest
    // carries, so a project without routing exports what every older fl
    // reads.
    #[test]
    fn the_format_is_the_oldest_that_holds_what_the_manifest_carries() {
        let (s, p, _, _) = store();
        for (root, routing, want) in [
            (None, None, 1),
            (Some(root()), None, 2),
            (None, Some(routes()), 3),
            (Some(root()), Some(routes()), 3),
        ] {
            let m = export(&s, &p, "abc", 7, root, routing.clone()).unwrap();
            assert_eq!(m.body.format_version, want);
            let back = Manifest::parse(&m.to_json()).unwrap();
            assert_eq!(back.body.routing, routing);
            assert_eq!(back, m);
        }
        let text = export(&s, &p, "abc", 7, None, None).unwrap().to_json();
        assert!(!text.contains("routing"), "skipped when absent: {text}");
    }

    #[test]
    fn a_map_on_a_format_it_does_not_belong_to_or_not_valid_is_refused() {
        let (s, p, _, _) = store();
        let mut m = export(&s, &p, "abc", 7, None, Some(routes())).unwrap();
        m.body.format_version = 2;
        let err = rehashed(m).verify().unwrap_err();
        assert!(
            matches!(err, ManifestError::Inconsistent(ref w) if w.contains("but what it carries")),
            "{err}"
        );
        let mut m = export(&s, &p, "abc", 7, None, Some(routes())).unwrap();
        m.body.routing = None;
        let err = rehashed(m).verify().unwrap_err();
        assert!(matches!(err, ManifestError::Inconsistent(_)), "{err}");
        let mut m = export(&s, &p, "abc", 7, None, Some(routes())).unwrap();
        m.body.routing.as_mut().unwrap().areas[0].area = "Code".into();
        let err = rehashed(m).verify().unwrap_err();
        assert!(
            matches!(err, ManifestError::Inconsistent(ref w) if w.contains("its routing map is not valid")),
            "{err}"
        );
        let mut bad = routes();
        bad.areas.swap(0, 1);
        assert!(
            export(&s, &p, "abc", 7, None, Some(bad)).is_err(),
            "never exported"
        );
    }

    #[test]
    fn a_hand_edited_routing_map_is_refused() {
        let (s, p, _, _) = store();
        let text = export(&s, &p, "abc", 7, None, Some(routes()))
            .unwrap()
            .to_json();
        let edited = text.replacen("\"tier\": \"local\"", "\"tier\": \"github\"", 1);
        assert_ne!(edited, text, "the edit must have landed");
        let err = Manifest::parse(&edited).unwrap_err();
        assert!(matches!(err, ManifestError::HandEdited { .. }), "{err}");
    }

    #[test]
    fn a_transition_naming_an_unlisted_gate_is_refused_even_with_a_correct_hash() {
        let (s, p, _, g2) = store();
        let mut m = export(&s, &p, "abc", 7, None, None).unwrap();
        m.body.gates.retain(|g| g.id != g2);
        m.content_sha256 = content_sha256(&m.body).unwrap();
        let err = Manifest::parse(&m.to_json()).unwrap_err();
        assert!(matches!(err, ManifestError::Inconsistent(_)), "{err}");
    }

    /// Re-hash after a hand change, so only `check_consistent` can refuse.
    fn rehashed(mut m: Manifest) -> Manifest {
        m.content_sha256 = content_sha256(&m.body).unwrap();
        m
    }

    #[test]
    fn a_gate_of_another_project_is_refused() {
        let (s, p, g1, _) = store();
        let mut m = export(&s, &p, "abc", 7, None, None).unwrap();
        let g = m.body.gates.iter_mut().find(|g| g.id == g1).unwrap();
        g.project = ProjectId(fl_core::ids::seq_iri(999));
        let err = rehashed(m).verify().unwrap_err();
        assert!(matches!(err, ManifestError::Inconsistent(_)), "{err}");
    }

    #[test]
    fn a_pass_mark_in_the_file_is_refused() {
        let (s, p, _, _) = store();
        let mut m = export(&s, &p, "abc", 7, None, None).unwrap();
        m.body.gates[0].last_pass_commit = Some("abc".into());
        let err = rehashed(m).verify().unwrap_err();
        assert!(matches!(err, ManifestError::Inconsistent(_)), "{err}");
    }

    #[test]
    fn a_gate_or_transition_listed_twice_is_refused() {
        let (s, p, _, _) = store();
        let mut m = export(&s, &p, "abc", 7, None, None).unwrap();
        let dup = m.body.gates[0].clone();
        m.body.gates.push(dup);
        let err = rehashed(m).verify().unwrap_err();
        assert!(matches!(err, ManifestError::Inconsistent(_)), "{err}");

        let mut m = export(&s, &p, "abc", 7, None, None).unwrap();
        let dup = m.body.transitions[0].clone();
        m.body.transitions.push(dup);
        let err = rehashed(m).verify().unwrap_err();
        assert!(matches!(err, ManifestError::Inconsistent(_)), "{err}");
    }

    #[test]
    fn currency_ignores_the_pass_mark_and_sees_every_other_change() {
        let (s, p, g1, _) = store();
        let m = export(&s, &p, "abc", 7, None, None).unwrap();
        let mut def = s.get_gate(&g1).unwrap().unwrap();
        def.last_pass_commit = Some("zzz".into());
        assert_eq!(m.currency_of(&def), Currency::Current);
        def.authored_at_commit = "c2".into();
        assert_eq!(m.currency_of(&def), Currency::Differs);
        def.id = GateId(fl_core::ids::seq_iri(999));
        assert_eq!(m.currency_of(&def), Currency::Absent);
    }
}
