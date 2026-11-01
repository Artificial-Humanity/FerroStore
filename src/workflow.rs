//! `RedbStore` persistence for [`WorkflowStore`] (`FORMAT_WITH_WORKFLOW = 8`, workflow spec §2).

use fl_core::ids::{Kind, ProjectId, WorkflowId, WorkflowRunId};
use fl_core::iri::Iri;
use fl_core::store::{StoreError, WorkflowStore};
use fl_core::workflow::{WorkflowDef, WorkflowRun, validate_workflow_def};
use redb::{ReadableDatabase, ReadableTable, TableDefinition};

use crate::{FORMAT_WITH_WORKFLOW, RedbStore, additive, backend, decode, index_new, raise_format};

pub(crate) const WORKFLOW_DEFS: TableDefinition<&str, &str> = TableDefinition::new("workflow_defs");
pub(crate) const WORKFLOW_SLUGS: TableDefinition<(&str, &str), &str> =
    TableDefinition::new("workflow_slugs");
pub(crate) const WORKFLOW_RUNS: TableDefinition<&str, &str> = TableDefinition::new("workflow_runs");

impl RedbStore {
    fn check_project(&self, project: &ProjectId) -> Result<ProjectId, StoreError> {
        let (primary, kind) = self.locate(project.iri())?;
        if kind != Kind::Project {
            return Err(StoreError::WrongKind {
                id: project.0.clone(),
                expected: Kind::Project,
                found: kind,
            });
        }
        Ok(ProjectId(primary))
    }

    fn check_project_in_wtx(
        &self,
        tx: &redb::WriteTransaction,
        project: &ProjectId,
    ) -> Result<ProjectId, StoreError> {
        let (primary, kind) = self.locate_for_write(tx, project.iri())?;
        if kind != Kind::Project {
            return Err(StoreError::WrongKind {
                id: project.0.clone(),
                expected: Kind::Project,
                found: kind,
            });
        }
        Ok(ProjectId(primary))
    }

    fn resolve_if_kind(&self, id: &Iri, expected: Kind) -> Result<Iri, StoreError> {
        match self.locate(id) {
            Ok((_, found)) if found != expected => Err(StoreError::WrongKind {
                id: id.clone(),
                expected,
                found,
            }),
            Ok((primary, _)) => Ok(primary),
            Err(StoreError::NotOwned { .. }) => Ok(id.clone()),
            Err(e) => Err(e),
        }
    }

    fn resolve_if_kind_in_wtx(
        &self,
        tx: &redb::WriteTransaction,
        id: &Iri,
        expected: Kind,
    ) -> Result<Iri, StoreError> {
        match self.locate_for_write(tx, id) {
            Ok((_, found)) if found != expected => Err(StoreError::WrongKind {
                id: id.clone(),
                expected,
                found,
            }),
            Ok((primary, _)) => Ok(primary),
            Err(StoreError::NotOwned { .. }) => Ok(id.clone()),
            Err(e) => Err(e),
        }
    }
}

impl WorkflowStore for RedbStore {
    fn upsert_workflow_def(&self, mut def: WorkflowDef) -> Result<WorkflowDef, StoreError> {
        validate_workflow_def(&def).map_err(|e| StoreError::InvalidWorkflow(e.to_string()))?;
        let tx = self.db.begin_write().map_err(backend)?;
        let project = self.check_project_in_wtx(&tx, &def.project)?;

        let existing_slug_id = {
            let slugs = tx.open_table(WORKFLOW_SLUGS).map_err(backend)?;
            slugs
                .get((project.as_str(), def.slug.as_str()))
                .map_err(backend)?
                .map(|g| g.value().to_owned())
        };

        if let Some(existing_id_str) = existing_slug_id {
            let existing_iri = Iri::parse(&existing_id_str).map_err(decode)?;
            let canonical_id = WorkflowId(self.resolve_in_wtx(&tx, &existing_iri)?);
            def.id = canonical_id.clone();
            def.project = project;
            let json = serde_json::to_string(&def).map_err(backend)?;
            tx.open_table(WORKFLOW_DEFS)
                .map_err(backend)?
                .insert(canonical_id.as_str(), json.as_str())
                .map_err(backend)?;
            raise_format(&tx, FORMAT_WITH_WORKFLOW)?;
            tx.commit().map_err(backend)?;
            return Ok(def);
        }

        let canonical_id =
            WorkflowId(self.resolve_if_kind_in_wtx(&tx, def.id.iri(), Kind::Workflow)?);
        self.index_imported_in(&tx, Kind::Workflow, canonical_id.iri())?;

        let prev_def: Option<WorkflowDef> = {
            let defs = tx.open_table(WORKFLOW_DEFS).map_err(backend)?;
            match defs.get(canonical_id.as_str()).map_err(backend)? {
                Some(g) => Some(serde_json::from_str(g.value()).map_err(decode)?),
                None => None,
            }
        };
        if let Some(prev) = prev_def
            && (prev.project != project || prev.slug != def.slug)
        {
            tx.open_table(WORKFLOW_SLUGS)
                .map_err(backend)?
                .remove((prev.project.as_str(), prev.slug.as_str()))
                .map_err(backend)?;
        }

        def.id = canonical_id.clone();
        def.project = project.clone();
        let json = serde_json::to_string(&def).map_err(backend)?;
        tx.open_table(WORKFLOW_SLUGS)
            .map_err(backend)?
            .insert((project.as_str(), def.slug.as_str()), canonical_id.as_str())
            .map_err(backend)?;
        tx.open_table(WORKFLOW_DEFS)
            .map_err(backend)?
            .insert(canonical_id.as_str(), json.as_str())
            .map_err(backend)?;
        raise_format(&tx, FORMAT_WITH_WORKFLOW)?;
        tx.commit().map_err(backend)?;
        Ok(def)
    }

    fn get_workflow_def(&self, id: &WorkflowId) -> Result<Option<WorkflowDef>, StoreError> {
        let canonical = self.resolve_if_kind(id.iri(), Kind::Workflow)?;
        let tx = self.db.begin_read().map_err(backend)?;
        additive(&tx, WORKFLOW_DEFS, &canonical)
    }

    fn get_workflow_def_by_slug(
        &self,
        project: &ProjectId,
        slug: &str,
    ) -> Result<Option<WorkflowDef>, StoreError> {
        let canonical_project = self.check_project(project)?;
        let tx = self.db.begin_read().map_err(backend)?;
        let slugs = match tx.open_table(WORKFLOW_SLUGS) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(e) => return Err(backend(e)),
        };
        let Some(id_guard) = slugs
            .get((canonical_project.as_str(), slug))
            .map_err(backend)?
        else {
            return Ok(None);
        };
        let wf_iri = Iri::parse(id_guard.value()).map_err(decode)?;
        drop(id_guard);
        drop(slugs);
        drop(tx);
        let canonical_wf = self.resolve_if_kind(&wf_iri, Kind::Workflow)?;
        let tx = self.db.begin_read().map_err(backend)?;
        additive(&tx, WORKFLOW_DEFS, &canonical_wf)
    }

    fn list_workflow_defs(
        &self,
        project: Option<&ProjectId>,
    ) -> Result<Vec<WorkflowDef>, StoreError> {
        let project_filter = match project {
            Some(p) => Some(self.check_project(p)?),
            None => None,
        };
        let tx = self.db.begin_read().map_err(backend)?;
        let defs = match tx.open_table(WORKFLOW_DEFS) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(backend(e)),
        };
        let mut out = Vec::new();
        for row in defs.iter().map_err(backend)? {
            let (_k, v) = row.map_err(backend)?;
            let def: WorkflowDef = serde_json::from_str(v.value()).map_err(decode)?;
            if project_filter.as_ref().is_none_or(|p| &def.project == p) {
                out.push(def);
            }
        }
        out.sort_by(|a, b| (&a.project, &a.slug).cmp(&(&b.project, &b.slug)));
        Ok(out)
    }

    fn create_workflow_run(&self, mut run: WorkflowRun) -> Result<WorkflowRun, StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        let canonical_project = self.check_project_in_wtx(&tx, &run.project)?;
        let canonical_wf_id =
            WorkflowId(self.resolve_if_kind_in_wtx(&tx, run.workflow_id.iri(), Kind::Workflow)?);
        let wf_exists = tx
            .open_table(WORKFLOW_DEFS)
            .map_err(backend)?
            .get(canonical_wf_id.as_str())
            .map_err(backend)?
            .is_some();
        if !wf_exists {
            return Err(StoreError::NoSuchWorkflow(run.workflow_id.to_string()));
        }

        let canonical_run_id =
            WorkflowRunId(self.resolve_if_kind_in_wtx(&tx, run.id.iri(), Kind::WorkflowRun)?);
        index_new(&tx, canonical_run_id.iri(), Kind::WorkflowRun)?;

        run.id = canonical_run_id.clone();
        run.project = canonical_project;
        run.workflow_id = canonical_wf_id;
        let json = serde_json::to_string(&run).map_err(backend)?;
        tx.open_table(WORKFLOW_RUNS)
            .map_err(backend)?
            .insert(canonical_run_id.as_str(), json.as_str())
            .map_err(backend)?;
        raise_format(&tx, FORMAT_WITH_WORKFLOW)?;
        tx.commit().map_err(backend)?;
        Ok(run)
    }

    fn update_workflow_run(&self, mut run: WorkflowRun) -> Result<WorkflowRun, StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        let canonical_id =
            WorkflowRunId(self.resolve_if_kind_in_wtx(&tx, run.id.iri(), Kind::WorkflowRun)?);
        {
            let mut runs = tx.open_table(WORKFLOW_RUNS).map_err(backend)?;
            if runs.get(canonical_id.as_str()).map_err(backend)?.is_none() {
                return Err(StoreError::NoSuchWorkflowRun(run.id.to_string()));
            }
            run.id = canonical_id.clone();
            let json = serde_json::to_string(&run).map_err(backend)?;
            runs.insert(canonical_id.as_str(), json.as_str())
                .map_err(backend)?;
        }
        raise_format(&tx, FORMAT_WITH_WORKFLOW)?;
        tx.commit().map_err(backend)?;
        Ok(run)
    }

    fn get_workflow_run(&self, id: &WorkflowRunId) -> Result<Option<WorkflowRun>, StoreError> {
        let canonical = self.resolve_if_kind(id.iri(), Kind::WorkflowRun)?;
        let tx = self.db.begin_read().map_err(backend)?;
        additive(&tx, WORKFLOW_RUNS, &canonical)
    }

    fn list_workflow_runs(
        &self,
        project: Option<&ProjectId>,
        workflow_id: Option<&WorkflowId>,
    ) -> Result<Vec<WorkflowRun>, StoreError> {
        let project_filter = match project {
            Some(p) => Some(self.check_project(p)?),
            None => None,
        };
        let wf_filter = match workflow_id {
            Some(w) => Some(WorkflowId(self.resolve_if_kind(w.iri(), Kind::Workflow)?)),
            None => None,
        };
        let tx = self.db.begin_read().map_err(backend)?;
        let runs = match tx.open_table(WORKFLOW_RUNS) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(backend(e)),
        };
        let mut out = Vec::new();
        for row in runs.iter().map_err(backend)? {
            let (_k, v) = row.map_err(backend)?;
            let run: WorkflowRun = serde_json::from_str(v.value()).map_err(decode)?;
            if project_filter.as_ref().is_none_or(|p| &run.project == p)
                && wf_filter.as_ref().is_none_or(|w| &run.workflow_id == w)
            {
                out.push(run);
            }
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(out)
    }

    fn import_workflow_def(&self, mut def: WorkflowDef) -> Result<(), StoreError> {
        validate_workflow_def(&def).map_err(|e| StoreError::InvalidWorkflow(e.to_string()))?;
        let tx = self.db.begin_write().map_err(backend)?;
        let canonical_project =
            ProjectId(self.resolve_if_kind_in_wtx(&tx, def.project.iri(), Kind::Project)?);
        let canonical_id =
            WorkflowId(self.resolve_if_kind_in_wtx(&tx, def.id.iri(), Kind::Workflow)?);

        let existing_slug_id = {
            let slugs = tx.open_table(WORKFLOW_SLUGS).map_err(backend)?;
            slugs
                .get((canonical_project.as_str(), def.slug.as_str()))
                .map_err(backend)?
                .map(|g| g.value().to_owned())
        };
        if let Some(existing_id_str) = existing_slug_id {
            let existing_id = WorkflowId(Iri::parse(&existing_id_str).map_err(decode)?);
            if existing_id != canonical_id {
                self.alias_unpublished_id_in(
                    &tx,
                    Kind::Workflow,
                    existing_id.iri(),
                    canonical_id.iri(),
                )?;
                tx.open_table(WORKFLOW_DEFS)
                    .map_err(backend)?
                    .remove(existing_id.as_str())
                    .map_err(backend)?;
                let mut runs = tx.open_table(WORKFLOW_RUNS).map_err(backend)?;
                let mut updated_runs = Vec::new();
                for row in runs.iter().map_err(backend)? {
                    let (k, v) = row.map_err(backend)?;
                    let mut r: WorkflowRun = serde_json::from_str(v.value()).map_err(decode)?;
                    if r.workflow_id == existing_id {
                        r.workflow_id = canonical_id.clone();
                        updated_runs.push((
                            k.value().to_owned(),
                            serde_json::to_string(&r).map_err(backend)?,
                        ));
                    }
                }
                for (k, json) in updated_runs {
                    runs.insert(k.as_str(), json.as_str()).map_err(backend)?;
                }
            }
        }

        let prev_def: Option<WorkflowDef> = {
            let defs = tx.open_table(WORKFLOW_DEFS).map_err(backend)?;
            match defs.get(canonical_id.as_str()).map_err(backend)? {
                Some(g) => Some(serde_json::from_str(g.value()).map_err(decode)?),
                None => None,
            }
        };
        if let Some(prev) = prev_def
            && (prev.project != canonical_project || prev.slug != def.slug)
        {
            tx.open_table(WORKFLOW_SLUGS)
                .map_err(backend)?
                .remove((prev.project.as_str(), prev.slug.as_str()))
                .map_err(backend)?;
        }

        self.index_imported_in(&tx, Kind::Workflow, canonical_id.iri())?;
        def.id = canonical_id.clone();
        def.project = canonical_project.clone();
        let json = serde_json::to_string(&def).map_err(backend)?;
        tx.open_table(WORKFLOW_SLUGS)
            .map_err(backend)?
            .insert(
                (canonical_project.as_str(), def.slug.as_str()),
                canonical_id.as_str(),
            )
            .map_err(backend)?;
        tx.open_table(WORKFLOW_DEFS)
            .map_err(backend)?
            .insert(canonical_id.as_str(), json.as_str())
            .map_err(backend)?;
        raise_format(&tx, FORMAT_WITH_WORKFLOW)?;
        tx.commit().map_err(backend)?;
        Ok(())
    }

    fn import_workflow_run(&self, mut run: WorkflowRun) -> Result<(), StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        let canonical_project =
            ProjectId(self.resolve_if_kind_in_wtx(&tx, run.project.iri(), Kind::Project)?);
        let canonical_wf_id =
            WorkflowId(self.resolve_if_kind_in_wtx(&tx, run.workflow_id.iri(), Kind::Workflow)?);
        let wf_exists = tx
            .open_table(WORKFLOW_DEFS)
            .map_err(backend)?
            .get(canonical_wf_id.as_str())
            .map_err(backend)?
            .is_some();
        if !wf_exists {
            return Err(StoreError::NoSuchWorkflow(run.workflow_id.to_string()));
        }

        let canonical_id =
            WorkflowRunId(self.resolve_if_kind_in_wtx(&tx, run.id.iri(), Kind::WorkflowRun)?);
        self.index_imported_in(&tx, Kind::WorkflowRun, canonical_id.iri())?;
        run.id = canonical_id.clone();
        run.project = canonical_project;
        run.workflow_id = canonical_wf_id;
        let json = serde_json::to_string(&run).map_err(backend)?;
        tx.open_table(WORKFLOW_RUNS)
            .map_err(backend)?
            .insert(canonical_id.as_str(), json.as_str())
            .map_err(backend)?;
        raise_format(&tx, FORMAT_WITH_WORKFLOW)?;
        tx.commit().map_err(backend)?;
        Ok(())
    }
}
