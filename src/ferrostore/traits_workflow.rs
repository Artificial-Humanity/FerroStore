//! Workflow definitions and runs.

use super::FerroStore;
use super::doc::{self, WORKFLOW_DEFS, WORKFLOW_RUNS, WORKFLOW_SLUGS};
use crate::{FORMAT_WITH_WORKFLOW, decode};
use fl_core::ids::{Kind, ProjectId, WorkflowId, WorkflowRunId};
use fl_core::iri::Iri;
use fl_core::store::{StoreError, WorkflowStore};
use fl_core::workflow::{WorkflowDef, WorkflowRun, validate_workflow_def};

fn check_project(
    txn: &super::Txn,
    label: &str,
    project: &ProjectId,
) -> Result<ProjectId, StoreError> {
    let (primary, kind) = doc::locate(txn, label, project.iri())?;
    if kind != Kind::Project {
        return Err(StoreError::WrongKind {
            id: project.0.clone(),
            expected: Kind::Project,
            found: kind,
        });
    }
    Ok(ProjectId(primary))
}

fn slug_id(
    txn: &super::Txn,
    project: &ProjectId,
    slug: &str,
) -> Result<Option<String>, StoreError> {
    doc::get_json(txn, WORKFLOW_SLUGS, &doc::pair(project.as_str(), slug))
}

impl WorkflowStore for FerroStore {
    fn upsert_workflow_def(&self, def: WorkflowDef) -> Result<WorkflowDef, StoreError> {
        validate_workflow_def(&def).map_err(|e| StoreError::InvalidWorkflow(e.to_string()))?;
        let label = self.label.clone();
        self.with_write(|txn| {
            let mut def = def.clone();
            let project = check_project(txn, &label, &def.project)?;
            if let Some(existing_id_str) = slug_id(txn, &project, &def.slug)? {
                let existing_iri = Iri::parse(&existing_id_str).map_err(decode)?;
                let canonical_id = WorkflowId(doc::resolve_iri(txn, &label, &existing_iri)?);
                def.id = canonical_id.clone();
                def.project = project;
                doc::put_json(txn, WORKFLOW_DEFS, canonical_id.as_str(), &def)?;
                doc::raise_format(txn, FORMAT_WITH_WORKFLOW)?;
                return Ok(def);
            }

            let canonical_id = WorkflowId(doc::resolve_if_kind(
                txn,
                &label,
                def.id.iri(),
                Kind::Workflow,
            )?);
            doc::index_imported(txn, Kind::Workflow, canonical_id.iri())?;
            let prev_def: Option<WorkflowDef> =
                doc::get_json(txn, WORKFLOW_DEFS, canonical_id.as_str())?;
            if let Some(prev) = prev_def
                && (prev.project != project || prev.slug != def.slug)
            {
                txn.delete(WORKFLOW_SLUGS, doc::pair(prev.project.as_str(), &prev.slug));
            }
            def.id = canonical_id.clone();
            def.project = project.clone();
            doc::put_json(
                txn,
                WORKFLOW_SLUGS,
                &doc::pair(project.as_str(), &def.slug),
                &canonical_id.as_str(),
            )?;
            doc::put_json(txn, WORKFLOW_DEFS, canonical_id.as_str(), &def)?;
            doc::raise_format(txn, FORMAT_WITH_WORKFLOW)?;
            Ok(def)
        })
    }

    fn get_workflow_def(&self, id: &WorkflowId) -> Result<Option<WorkflowDef>, StoreError> {
        let id = id.clone();
        let label = self.label.clone();
        self.with_read(|txn| {
            let canonical = doc::resolve_if_kind(txn, &label, id.iri(), Kind::Workflow)?;
            doc::get_json(txn, WORKFLOW_DEFS, canonical.as_str())
        })
    }

    fn get_workflow_def_by_slug(
        &self,
        project: &ProjectId,
        slug: &str,
    ) -> Result<Option<WorkflowDef>, StoreError> {
        let project = project.clone();
        let slug = slug.to_string();
        let label = self.label.clone();
        self.with_read(|txn| {
            let project = check_project(txn, &label, &project)?;
            let Some(id_str) = slug_id(txn, &project, &slug)? else {
                return Ok(None);
            };
            let wf_iri = Iri::parse(&id_str).map_err(decode)?;
            let canonical = doc::resolve_if_kind(txn, &label, &wf_iri, Kind::Workflow)?;
            doc::get_json(txn, WORKFLOW_DEFS, canonical.as_str())
        })
    }

    fn list_workflow_defs(
        &self,
        project: Option<&ProjectId>,
    ) -> Result<Vec<WorkflowDef>, StoreError> {
        let project = project.cloned();
        let label = self.label.clone();
        self.with_read(|txn| {
            let project_filter = match &project {
                Some(p) => Some(check_project(txn, &label, p)?),
                None => None,
            };
            let mut out = doc::scan_json::<WorkflowDef>(txn, WORKFLOW_DEFS)?;
            out.retain(|def| project_filter.as_ref().is_none_or(|p| &def.project == p));
            out.sort_by(|a, b| (&a.project, &a.slug).cmp(&(&b.project, &b.slug)));
            Ok(out)
        })
    }

    fn create_workflow_run(&self, run: WorkflowRun) -> Result<WorkflowRun, StoreError> {
        let label = self.label.clone();
        self.with_write(|txn| {
            let mut run = run.clone();
            let project = check_project(txn, &label, &run.project)?;
            let workflow_id = WorkflowId(doc::resolve_if_kind(
                txn,
                &label,
                run.workflow_id.iri(),
                Kind::Workflow,
            )?);
            if doc::get_json::<WorkflowDef>(txn, WORKFLOW_DEFS, workflow_id.as_str())?.is_none() {
                return Err(StoreError::NoSuchWorkflow(run.workflow_id.to_string()));
            }
            let run_id = WorkflowRunId(doc::resolve_if_kind(
                txn,
                &label,
                run.id.iri(),
                Kind::WorkflowRun,
            )?);
            doc::index_new(txn, run_id.iri(), Kind::WorkflowRun)?;
            run.id = run_id.clone();
            run.project = project;
            run.workflow_id = workflow_id;
            doc::put_json(txn, WORKFLOW_RUNS, run_id.as_str(), &run)?;
            doc::raise_format(txn, FORMAT_WITH_WORKFLOW)?;
            Ok(run)
        })
    }

    fn update_workflow_run(&self, run: WorkflowRun) -> Result<WorkflowRun, StoreError> {
        let label = self.label.clone();
        self.with_write(|txn| {
            let mut run = run.clone();
            let run_id = WorkflowRunId(doc::resolve_if_kind(
                txn,
                &label,
                run.id.iri(),
                Kind::WorkflowRun,
            )?);
            if doc::get_json::<WorkflowRun>(txn, WORKFLOW_RUNS, run_id.as_str())?.is_none() {
                return Err(StoreError::NoSuchWorkflowRun(run.id.to_string()));
            }
            run.id = run_id.clone();
            doc::put_json(txn, WORKFLOW_RUNS, run_id.as_str(), &run)?;
            doc::raise_format(txn, FORMAT_WITH_WORKFLOW)?;
            Ok(run)
        })
    }

    fn get_workflow_run(&self, id: &WorkflowRunId) -> Result<Option<WorkflowRun>, StoreError> {
        let id = id.clone();
        let label = self.label.clone();
        self.with_read(|txn| {
            let canonical = doc::resolve_if_kind(txn, &label, id.iri(), Kind::WorkflowRun)?;
            doc::get_json(txn, WORKFLOW_RUNS, canonical.as_str())
        })
    }

    fn list_workflow_runs(
        &self,
        project: Option<&ProjectId>,
        workflow_id: Option<&WorkflowId>,
    ) -> Result<Vec<WorkflowRun>, StoreError> {
        let project = project.cloned();
        let workflow_id = workflow_id.cloned();
        let label = self.label.clone();
        self.with_read(|txn| {
            let project_filter = match &project {
                Some(p) => Some(check_project(txn, &label, p)?),
                None => None,
            };
            let wf_filter = match &workflow_id {
                Some(w) => Some(WorkflowId(doc::resolve_if_kind(
                    txn,
                    &label,
                    w.iri(),
                    Kind::Workflow,
                )?)),
                None => None,
            };
            let mut out = doc::scan_json::<WorkflowRun>(txn, WORKFLOW_RUNS)?;
            out.retain(|run| {
                project_filter.as_ref().is_none_or(|p| &run.project == p)
                    && wf_filter.as_ref().is_none_or(|w| &run.workflow_id == w)
            });
            out.sort_by(|a, b| a.id.cmp(&b.id));
            Ok(out)
        })
    }

    fn import_workflow_def(&self, def: WorkflowDef) -> Result<(), StoreError> {
        validate_workflow_def(&def).map_err(|e| StoreError::InvalidWorkflow(e.to_string()))?;
        let label = self.label.clone();
        self.with_write(|txn| {
            let mut def = def.clone();
            let project = ProjectId(doc::resolve_if_kind(
                txn,
                &label,
                def.project.iri(),
                Kind::Project,
            )?);
            let canonical_id = WorkflowId(doc::resolve_if_kind(
                txn,
                &label,
                def.id.iri(),
                Kind::Workflow,
            )?);
            if let Some(existing_id_str) = slug_id(txn, &project, &def.slug)? {
                let existing_id = WorkflowId(Iri::parse(&existing_id_str).map_err(decode)?);
                if existing_id != canonical_id {
                    doc::alias_unpublished(
                        txn,
                        Kind::Workflow,
                        existing_id.iri(),
                        canonical_id.iri(),
                    )?;
                    txn.delete(WORKFLOW_DEFS, existing_id.as_str());
                    let rewritten = workflow_runs_retargeted(txn, &existing_id, &canonical_id)?;
                    for (key, run) in rewritten {
                        doc::put_json(txn, WORKFLOW_RUNS, &key, &run)?;
                    }
                }
            }
            let prev_def: Option<WorkflowDef> =
                doc::get_json(txn, WORKFLOW_DEFS, canonical_id.as_str())?;
            if let Some(prev) = prev_def
                && (prev.project != project || prev.slug != def.slug)
            {
                txn.delete(WORKFLOW_SLUGS, doc::pair(prev.project.as_str(), &prev.slug));
            }
            doc::index_imported(txn, Kind::Workflow, canonical_id.iri())?;
            def.id = canonical_id.clone();
            def.project = project.clone();
            doc::put_json(
                txn,
                WORKFLOW_SLUGS,
                &doc::pair(project.as_str(), &def.slug),
                &canonical_id.as_str(),
            )?;
            doc::put_json(txn, WORKFLOW_DEFS, canonical_id.as_str(), &def)?;
            doc::raise_format(txn, FORMAT_WITH_WORKFLOW)?;
            Ok(())
        })
    }

    fn import_workflow_run(&self, run: WorkflowRun) -> Result<(), StoreError> {
        let label = self.label.clone();
        self.with_write(|txn| {
            let mut run = run.clone();
            let project = ProjectId(doc::resolve_if_kind(
                txn,
                &label,
                run.project.iri(),
                Kind::Project,
            )?);
            let workflow_id = WorkflowId(doc::resolve_if_kind(
                txn,
                &label,
                run.workflow_id.iri(),
                Kind::Workflow,
            )?);
            if doc::get_json::<WorkflowDef>(txn, WORKFLOW_DEFS, workflow_id.as_str())?.is_none() {
                return Err(StoreError::NoSuchWorkflow(run.workflow_id.to_string()));
            }
            let run_id = WorkflowRunId(doc::resolve_if_kind(
                txn,
                &label,
                run.id.iri(),
                Kind::WorkflowRun,
            )?);
            doc::index_imported(txn, Kind::WorkflowRun, run_id.iri())?;
            run.id = run_id.clone();
            run.project = project;
            run.workflow_id = workflow_id;
            doc::put_json(txn, WORKFLOW_RUNS, run_id.as_str(), &run)?;
            doc::raise_format(txn, FORMAT_WITH_WORKFLOW)?;
            Ok(())
        })
    }
}

fn workflow_runs_retargeted(
    txn: &super::Txn,
    from: &WorkflowId,
    to: &WorkflowId,
) -> Result<Vec<(String, WorkflowRun)>, StoreError> {
    let mut out = Vec::new();
    for (key, doc) in txn.scan(WORKFLOW_RUNS) {
        let mut run: WorkflowRun = serde_json::from_value(doc).map_err(decode)?;
        if run.workflow_id == *from {
            run.workflow_id = to.clone();
            out.push((key, run));
        }
    }
    Ok(out)
}
