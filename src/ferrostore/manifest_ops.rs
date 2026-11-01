//! Manifest export and import, with `RedbStore`'s signatures, errors and
//! [`ImportReport`]. An import is one write transaction: every refusal is
//! decided inside it, before anything is staged, and a refusal or a failed
//! write commits nothing.

use super::doc::{self, GATES, IMPORTS, PROJECTS, ROUTING, TRANSITIONS};
use super::{FerroStore, Txn};
use crate::manifest::{self, LedgerRoot, Manifest, ManifestError};
use crate::{FORMAT_WITH_IMPORTS, FORMAT_WITH_ROUTING, ImportReport, backend};
use fl_core::ids::{Kind, ProjectId};
use fl_core::model::{GateDef, Project, Transition};
use fl_core::routing::{Routes, RoutingMap};
use fl_core::store::{Bindings, StoreError};

impl FerroStore {
    /// Export `project`, which this store must author. `repository_node_id`
    /// is the repository the project's ledger is in, when its binding names
    /// one: the export carries that repository's ledger root, if this store
    /// records one. The same contract as `RedbStore::export_manifest`.
    pub fn export_manifest(
        &self,
        project: &ProjectId,
        commit: &str,
        exported_at_unix: u64,
        repository_node_id: Option<&str>,
    ) -> Result<Manifest, ManifestError> {
        let imported = self.with_read(|txn| {
            doc::check_kind(txn, &self.label, project.iri(), Kind::Project)?;
            doc::get_json::<String>(txn, IMPORTS, project.iri().as_str())
        })?;
        if imported.is_some() {
            return Err(ManifestError::NotAuthoring(project.clone()));
        }
        let ledger_root = match repository_node_id {
            None => None,
            Some(node) => self.ledger_root(node)?.map(|commit| LedgerRoot {
                repository_node_id: node.to_string(),
                commit,
            }),
        };
        manifest::export(
            self,
            project,
            commit,
            exported_at_unix,
            ledger_root,
            self.routes(project)?,
        )
    }

    /// Write the manifest's project, gates and transitions under their own
    /// IRIs, and mark the project imported, in ONE write transaction. The
    /// same contract as `RedbStore::import_manifest`.
    pub fn import_manifest(&self, m: &Manifest, root: &str) -> Result<ImportReport, ManifestError> {
        // A `Manifest` can be built by hand; the store checks it itself.
        m.verify()?;
        // A refusal is carried out of the write body here: the body returns
        // `Err`, so the transaction stages nothing and commits nothing.
        let mut refused: Option<ManifestError> = None;
        let written = self.with_write(|txn| {
            refused = None;
            let plan = match plan_import(txn, &self.label, m, root) {
                Ok(plan) => plan,
                Err(err) => {
                    refused = Some(err);
                    return Err(backend("the manifest import was refused"));
                }
            };
            apply_import(txn, m, root, &plan)?;
            Ok(plan.report)
        });
        match (written, refused) {
            (_, Some(err)) => Err(err),
            (Ok(report), None) => Ok(report),
            (Err(err), None) => Err(err.into()),
        }
    }
}

/// What an import will write, decided before any of it is staged.
struct Plan {
    held_project: bool,
    /// Gates to write, each with whether it is new to this store.
    writes: Vec<(GateDef, bool)>,
    stale_transitions: Vec<String>,
    report: ImportReport,
}

/// Every check `RedbStore::import_manifest` makes, in the same order, read
/// from `txn`. Stages nothing.
fn plan_import(txn: &Txn, label: &str, m: &Manifest, root: &str) -> Result<Plan, ManifestError> {
    let body = &m.body;
    let project = &body.project;

    if let Some(root) = &body.ledger_root {
        let held = doc::get_json::<String>(txn, doc::LEDGER_ROOTS, &root.repository_node_id)?;
        doc::refuse_a_changed_root(&root.repository_node_id, held, &root.commit)?;
    }

    let held_project = match doc::locate(txn, label, project.iri()) {
        Ok((_, Kind::Project)) => true,
        Ok((_, found)) => {
            return Err(StoreError::WrongKind {
                id: project.iri().clone(),
                expected: Kind::Project,
                found,
            }
            .into());
        }
        Err(StoreError::NotOwned { .. }) => false,
        Err(err) => return Err(err.into()),
    };
    if held_project && doc::get_json::<String>(txn, IMPORTS, project.iri().as_str())?.is_none() {
        return Err(ManifestError::AuthoringStore(project.clone()));
    }
    // ⚠ An older checked-out manifest would un-route this machine alone.
    if held_project
        && body.routing.is_none()
        && doc::get_json::<RoutingMap>(txn, ROUTING, project.iri().as_str())?.is_some()
    {
        return Err(ManifestError::WouldDropRouting(project.clone()));
    }
    let projects: Vec<Project> = doc::scan_json(txn, PROJECTS)?;
    // Routing spec decision 20: a routed store holds one project.
    if body.routing.is_some() {
        if let Some(other) = projects.iter().find(|p| p.id != *project) {
            return Err(StoreError::Backend(format!(
                "this store also holds project {} (root {}), and a routed project needs a store \
                 of its own: handles are numbered per store, so one number would name items of \
                 both. Give the project its own store — a config entry with its own `store` — \
                 and route it there",
                other.id, other.root
            ))
            .into());
        }
    } else if !held_project && doc::holds_routing(txn) {
        return Err(doc::routed_store_is_taken().into());
    }
    for other in &projects {
        if other.id != *project && other.root == root {
            return Err(ManifestError::RootTaken {
                root: root.to_string(),
                other: other.id.clone(),
            });
        }
    }
    let root_moved = if held_project {
        let held = projects
            .iter()
            .find(|p| p.id == *project)
            .ok_or_else(|| ManifestError::Inconsistent(format!("{project} vanished")))?;
        (held.root != root).then(|| (held.root.clone(), root.to_string()))
    } else {
        None
    };

    let held_gates: Vec<GateDef> = if held_project {
        doc::scan_json::<GateDef>(txn, GATES)?
            .into_iter()
            .filter(|g| g.project == *project)
            .collect()
    } else {
        vec![]
    };
    for g in &held_gates {
        if !body.gates.iter().any(|m| m.id == g.id) {
            return Err(ManifestError::WouldRemoveGate {
                id: g.id.clone(),
                name: g.name.clone(),
            });
        }
    }
    // Transitions mirror the manifest: one it no longer lists is removed.
    let stale_transitions: Vec<String> = if held_project {
        doc::scan_json::<Transition>(txn, TRANSITIONS)?
            .into_iter()
            .filter(|t| t.project == *project)
            .filter(|t| !body.transitions.iter().any(|m| m.name == t.name))
            .map(|t| t.name)
            .collect()
    } else {
        vec![]
    };

    let mut report = ImportReport {
        project: project.clone(),
        gates_added: 0,
        gates_changed: 0,
        gates_unchanged: 0,
        transitions: body.transitions.len(),
        transitions_removed: stale_transitions.clone(),
        root_moved,
        areas: body.routing.as_ref().map(|m| m.areas.len()),
    };
    let mut writes = Vec::new();
    for g in &body.gates {
        match doc::locate(txn, label, g.id.iri()) {
            Err(StoreError::NotOwned { .. }) => {
                writes.push((g.clone(), true));
                report.gates_added += 1;
            }
            Ok((_, Kind::Gate)) => {
                let Some(held) = held_gates.iter().find(|h| h.id == g.id) else {
                    return Err(ManifestError::Inconsistent(format!(
                        "gate {} is held by this store under another project",
                        g.id
                    )));
                };
                let mut bare = held.clone();
                bare.last_pass_commit = None;
                if bare == *g {
                    report.gates_unchanged += 1;
                } else {
                    writes.push((g.clone(), false));
                    report.gates_changed += 1;
                }
            }
            Ok((_, found)) => {
                return Err(StoreError::WrongKind {
                    id: g.id.iri().clone(),
                    expected: Kind::Gate,
                    found,
                }
                .into());
            }
            Err(err) => return Err(err.into()),
        }
    }
    Ok(Plan {
        held_project,
        writes,
        stale_transitions,
        report,
    })
}

/// Stage the writes `plan` decided. A failure here is a `StoreError`, which
/// aborts the transaction.
fn apply_import(txn: &mut Txn, m: &Manifest, root: &str, plan: &Plan) -> Result<(), StoreError> {
    let body = &m.body;
    let project = &body.project;
    if !plan.held_project || plan.report.root_moved.is_some() {
        if !plan.held_project {
            doc::index_new(txn, project.iri(), Kind::Project)?;
        }
        doc::put_json(
            txn,
            PROJECTS,
            project.iri().as_str(),
            &Project {
                id: project.clone(),
                root: root.to_string(),
            },
        )?;
    }
    for (g, is_new) in &plan.writes {
        if *is_new {
            doc::index_new(txn, g.id.iri(), Kind::Gate)?;
        }
        doc::put_json(txn, GATES, g.id.iri().as_str(), g)?;
    }
    for name in &plan.stale_transitions {
        txn.delete(TRANSITIONS, doc::pair(project.iri().as_str(), name));
    }
    for t in &body.transitions {
        doc::put_json(
            txn,
            TRANSITIONS,
            &doc::pair(t.project.iri().as_str(), &t.name),
            t,
        )?;
    }
    doc::put_json(txn, IMPORTS, project.iri().as_str(), &m.content_sha256)?;
    if let Some(map) = &body.routing {
        doc::put_json(txn, ROUTING, project.iri().as_str(), map)?;
        doc::raise_format(txn, FORMAT_WITH_ROUTING)?;
    }
    if let Some(root) = &body.ledger_root {
        doc::record_ledger_root(txn, &root.repository_node_id, &root.commit)?;
    }
    doc::raise_format(txn, FORMAT_WITH_IMPORTS)
}
