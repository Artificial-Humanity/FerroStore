//! Catalog, tracker, ledger, handles, bindings, escalations, and the ledger cache.

use super::FerroStore;
use super::doc::{
    self, ATTEMPTS, CANDIDATE_ATTEMPTS, CANDIDATE_RUNS, ESCALATING, FINDINGS, GATE_RUNS, GATES,
    GITHUB_BINDINGS, HANDLE_OF, HANDLES, IDS, IMPORTS, LEDGER_CUTOVERS, LEDGER_HEADS,
    LEDGER_PUBLISHED, LEDGER_ROOTS, LEDGER_SEGMENTS, NEXT_ATTEMPT, NEXT_RUN, PROJECTS, RECORDS,
    ROUTING, TOMBSTONES, TRANSITIONS,
};
use crate::{CHANGE_ROUTING, FORMAT_WITH_ROUTING, backend, decode, mint};
use fl_core::escalation::{EscalationFault, Escalations, Mark, Tombstone};
use fl_core::finding::{Finding, FindingState};
use fl_core::ids::{FindingId, GateId, Kind, ProjectId, RecordId};
use fl_core::iri::Iri;
use fl_core::log::{Attempt, GateRun};
use fl_core::model::{GateDef, GateKind, Project, Record, Selector, State, Transition};
use fl_core::routing::{ForeignRecord, Routes, RoutingMap, check_foreign_for_local};
use fl_core::split::{CachedSegment, LedgerCache, Outbox, Pending};
use fl_core::store::{Bindings, Catalog, Handles, Ledger, StoreError, Tracker};

fn imported_hash(
    txn: &super::Txn,
    label: &str,
    project: &ProjectId,
) -> Result<Option<String>, StoreError> {
    doc::check_kind(txn, label, project.iri(), Kind::Project)?;
    doc::get_json(txn, IMPORTS, project.iri().as_str())
}

fn refuse_if_imported(
    txn: &super::Txn,
    label: &str,
    project: &ProjectId,
    action: &'static str,
) -> Result<(), StoreError> {
    if imported_hash(txn, label, project)?.is_some() {
        return Err(StoreError::Imported {
            id: project.iri().clone(),
            action,
        });
    }
    Ok(())
}

fn append_log<T: serde::Serialize>(
    txn: &mut super::Txn,
    counter: &str,
    table: &str,
    value: &T,
    candidate: Option<(&str, &Iri)>,
) -> Result<(), StoreError> {
    let seq = doc::bump_meta(txn, counter)?;
    doc::put_json(txn, table, &doc::key_u64(seq), value)?;
    if let Some((index, id)) = candidate {
        doc::put_json(txn, index, id.as_str(), &seq)?;
    }
    Ok(())
}

impl FerroStore {
    /// Write `project`'s routing map. The store must author it, and it must be
    /// the only project. Raises the store to format 5.
    pub fn set_routes(&self, project: &ProjectId, map: &RoutingMap) -> Result<(), StoreError> {
        let project = project.clone();
        let map = map.clone();
        self.with_write(|txn| {
            doc::check_kind(txn, &self.label, project.iri(), Kind::Project)?;
            refuse_if_imported(txn, &self.label, &project, CHANGE_ROUTING)?;
            if let Some(other) = doc::scan_json::<Project>(txn, PROJECTS)?
                .into_iter()
                .find(|held| held.id != project)
            {
                return Err(backend(format!(
                    "this store also holds project {} (root {}), and a routed project needs a store \
                     of its own: handles are numbered per store, so one number would name items of \
                     both. Give the project its own store — a config entry with its own `store` — \
                     and route it there",
                    other.id, other.root
                )));
            }
            map.check()
                .map_err(|why| backend(format!("the routing map is not valid: {why}")))?;
            doc::put_json(txn, ROUTING, project.iri().as_str(), &map)?;
            doc::raise_format(txn, FORMAT_WITH_ROUTING)?;
            Ok(())
        })
    }
}

impl Catalog for FerroStore {
    fn add_project(&self, root: &str) -> Result<ProjectId, StoreError> {
        let root = root.to_string();
        self.with_write(|txn| {
            if doc::holds_routing(txn) {
                return Err(doc::routed_store_is_taken());
            }
            let id = doc::insert_new(txn, Kind::Project, PROJECTS, None, |id| Project {
                id: ProjectId(id),
                root: root.clone(),
            })?;
            Ok(ProjectId(id))
        })
    }

    fn get_project(&self, id: &ProjectId) -> Result<Option<Project>, StoreError> {
        self.with_read(|txn| doc::get_owned(txn, &self.label, PROJECTS, id.iri()))
    }

    fn list_projects(&self) -> Result<Vec<Project>, StoreError> {
        self.with_read(|txn| doc::scan_json(txn, PROJECTS))
    }

    fn add_gate(
        &self,
        project: &ProjectId,
        name: &str,
        kind: GateKind,
        selector: Selector,
        min_population: u64,
        authored_at_commit: &str,
        authored_by: &str,
    ) -> Result<GateId, StoreError> {
        let project = project.clone();
        let name = name.to_string();
        let authored_at_commit = authored_at_commit.to_string();
        let authored_by = authored_by.to_string();
        self.with_write(|txn| {
            doc::check_kind(txn, &self.label, project.iri(), Kind::Project)?;
            refuse_if_imported(txn, &self.label, &project, "add a gate to")?;
            let id = doc::insert_new(txn, Kind::Gate, GATES, None, |id| GateDef {
                id: GateId(id),
                project: project.clone(),
                name: name.clone(),
                kind: kind.clone(),
                selector: selector.clone(),
                min_population,
                authored_at_commit: authored_at_commit.clone(),
                authored_by: authored_by.clone(),
                last_pass_commit: None,
            })?;
            Ok(GateId(id))
        })
    }

    fn get_gate(&self, id: &GateId) -> Result<Option<GateDef>, StoreError> {
        self.with_read(|txn| doc::get_owned(txn, &self.label, GATES, id.iri()))
    }

    fn list_gates(&self, project: &ProjectId) -> Result<Vec<GateDef>, StoreError> {
        let project = project.clone();
        self.with_read(|txn| {
            doc::check_kind(txn, &self.label, project.iri(), Kind::Project)?;
            let all: Vec<GateDef> = doc::scan_json(txn, GATES)?;
            Ok(all
                .into_iter()
                .filter(|gate| gate.project == project)
                .collect())
        })
    }

    fn update_gate(&self, def: &GateDef) -> Result<(), StoreError> {
        let def = def.clone();
        self.with_write(|txn| {
            let Some(held) = doc::get_owned::<GateDef>(txn, &self.label, GATES, def.id.iri())?
            else {
                return Err(StoreError::NoSuchGate(def.id.clone()));
            };
            // An imported gate may earn a local pass mark and nothing else.
            if imported_hash(txn, &self.label, &held.project)?.is_some() {
                let (mut before, mut after) = (held, def.clone());
                before.last_pass_commit = None;
                after.last_pass_commit = None;
                if before != after {
                    return Err(StoreError::Imported {
                        id: def.id.iri().clone(),
                        action: "change the definition of",
                    });
                }
            }
            doc::put_json(txn, GATES, def.id.iri().as_str(), &def)?;
            Ok(())
        })
    }

    fn add_transition(&self, transition: Transition) -> Result<(), StoreError> {
        self.with_write(|txn| {
            doc::check_kind(txn, &self.label, transition.project.iri(), Kind::Project)?;
            refuse_if_imported(txn, &self.label, &transition.project, "add a transition to")?;
            let key = doc::pair(transition.project.iri().as_str(), transition.name.as_str());
            doc::put_json(txn, TRANSITIONS, &key, &transition)?;
            Ok(())
        })
    }

    fn get_transition(
        &self,
        project: &ProjectId,
        name: &str,
    ) -> Result<Option<Transition>, StoreError> {
        let project = project.clone();
        let name = name.to_string();
        self.with_read(|txn| {
            doc::check_kind(txn, &self.label, project.iri(), Kind::Project)?;
            doc::get_json(txn, TRANSITIONS, &doc::pair(project.iri().as_str(), &name))
        })
    }

    fn list_transitions(&self, project: &ProjectId) -> Result<Vec<Transition>, StoreError> {
        let project = project.clone();
        self.with_read(|txn| {
            doc::check_kind(txn, &self.label, project.iri(), Kind::Project)?;
            let all: Vec<Transition> = doc::scan_json(txn, TRANSITIONS)?;
            Ok(all
                .into_iter()
                .filter(|transition| transition.project == project)
                .collect())
        })
    }

    fn kind_of(&self, id: &Iri) -> Result<Kind, StoreError> {
        self.with_read(|txn| locate_kind(txn, &self.label, id))
    }
}

fn locate_kind(txn: &super::Txn, label: &str, id: &Iri) -> Result<Kind, StoreError> {
    doc::locate(txn, label, id).map(|(_, kind)| kind)
}

impl Routes for FerroStore {
    fn routes(&self, project: &ProjectId) -> Result<Option<RoutingMap>, StoreError> {
        let project = project.clone();
        self.with_read(|txn| {
            doc::check_kind(txn, &self.label, project.iri(), Kind::Project)?;
            let Some(map) = doc::get_json::<RoutingMap>(txn, ROUTING, project.iri().as_str())?
            else {
                return Ok(None);
            };
            map.check().map_err(decode)?;
            Ok(Some(map))
        })
    }
}

impl Tracker for FerroStore {
    fn add_record_with_area(
        &self,
        project: &ProjectId,
        title: &str,
        area: Option<&str>,
    ) -> Result<RecordId, StoreError> {
        let project = project.clone();
        let title = title.to_string();
        let area = area.map(str::to_string);
        self.with_write(|txn| {
            doc::check_kind(txn, &self.label, project.iri(), Kind::Project)?;
            let raise = area.as_ref().map(|_| FORMAT_WITH_ROUTING);
            let id = doc::insert_new(txn, Kind::Record, RECORDS, raise, |id| Record {
                id: RecordId(id),
                project: project.clone(),
                title: title.clone(),
                state: State::Todo,
                also_known_as: vec![],
                area: area.clone(),
            })?;
            Ok(RecordId(id))
        })
    }

    fn get_record(&self, id: &RecordId) -> Result<Option<Record>, StoreError> {
        self.with_read(|txn| doc::get_owned(txn, &self.label, RECORDS, id.iri()))
    }

    fn list_records(&self, project: &ProjectId) -> Result<Vec<Record>, StoreError> {
        let project = project.clone();
        self.with_read(|txn| {
            doc::check_kind(txn, &self.label, project.iri(), Kind::Project)?;
            let gone = doc::tombstoned(txn)?;
            let all: Vec<Record> = doc::scan_json(txn, RECORDS)?;
            Ok(all
                .into_iter()
                .filter(|record| {
                    record.project == project && !gone.contains(record.id.iri().as_str())
                })
                .collect())
        })
    }

    fn set_record_state(&self, id: &RecordId, state: State) -> Result<(), StoreError> {
        let id = id.clone();
        self.with_write(|txn| {
            let changed =
                doc::rewrite(txn, &self.label, RECORDS, id.iri(), |mut record: Record| {
                    record.state = state;
                    record
                })?;
            if !changed {
                return Err(StoreError::NoSuchRecord(id.clone()));
            }
            Ok(())
        })
    }

    fn add_finding(&self, finding: Finding) -> Result<FindingId, StoreError> {
        self.with_write(|txn| {
            let finding = finding.clone();
            doc::check_kind(txn, &self.label, finding.project.iri(), Kind::Project)?;
            let raise = finding.area.as_ref().map(|_| FORMAT_WITH_ROUTING);
            let id = doc::insert_in(txn, mint()?, Kind::Finding, FINDINGS, raise, |txn, id| {
                let (record_primary, record_kind) =
                    doc::locate(txn, &self.label, finding.record.iri())?;
                if record_kind != Kind::Record {
                    return Err(StoreError::WrongKind {
                        id: finding.record.iri().clone(),
                        expected: Kind::Record,
                        found: record_kind,
                    });
                }
                // A marked record still takes findings. An escalated one does not.
                doc::refuse_escalated(txn, &record_primary)?;
                let mut finding = finding;
                finding.id = FindingId(id);
                finding.record = RecordId(record_primary);
                Ok(finding)
            })?;
            Ok(FindingId(id))
        })
    }

    fn add_finding_checked(
        &self,
        finding: Finding,
        record: ForeignRecord,
    ) -> Result<FindingId, StoreError> {
        self.with_write(|txn| {
            let finding = finding.clone();
            doc::check_kind(txn, &self.label, finding.project.iri(), Kind::Project)?;
            let held = doc::owns(txn, &self.label, record.id().iri())?;
            check_foreign_for_local(&record, held)?;
            let raise = finding.area.as_ref().map(|_| FORMAT_WITH_ROUTING);
            let id = doc::insert_new(txn, Kind::Finding, FINDINGS, raise, |id| {
                let mut finding = finding;
                finding.id = FindingId(id);
                finding.record = record.id().clone();
                finding
            })?;
            Ok(FindingId(id))
        })
    }

    fn get_finding(&self, id: &FindingId) -> Result<Option<Finding>, StoreError> {
        self.with_read(|txn| doc::get_owned(txn, &self.label, FINDINGS, id.iri()))
    }

    fn update_finding(&self, finding: &Finding) -> Result<(), StoreError> {
        let finding = finding.clone();
        self.with_write(|txn| {
            let changed = doc::rewrite(
                txn,
                &self.label,
                FINDINGS,
                finding.id.iri(),
                |kept: Finding| {
                    let mut stored = finding.clone();
                    stored.id = kept.id;
                    stored.also_known_as = kept.also_known_as;
                    stored.record = kept.record;
                    stored.raised_by = kept.raised_by;
                    stored.security = kept.security;
                    stored.area = kept.area;
                    stored
                },
            )?;
            if !changed {
                return Err(StoreError::NoSuchFinding(finding.id.clone()));
            }
            Ok(())
        })
    }

    fn list_findings(&self, project: &ProjectId) -> Result<Vec<Finding>, StoreError> {
        let project = project.clone();
        self.with_read(|txn| {
            doc::check_kind(txn, &self.label, project.iri(), Kind::Project)?;
            let gone = doc::tombstoned(txn)?;
            let all: Vec<Finding> = doc::scan_json(txn, FINDINGS)?;
            Ok(all
                .into_iter()
                .filter(|finding| {
                    finding.project == project && !gone.contains(finding.id.iri().as_str())
                })
                .collect())
        })
    }

    fn withdrawals_by(&self, actor: &str) -> Result<u64, StoreError> {
        let actor = actor.to_string();
        self.with_read(|txn| {
            let gone = doc::tombstoned(txn)?;
            let all: Vec<Finding> = doc::scan_json(txn, FINDINGS)?;
            Ok(all
                .into_iter()
                .filter(|finding| {
                    finding.raised_by == actor && finding.state == FindingState::Withdrawn
                })
                .filter(|finding| !gone.contains(finding.id.iri().as_str()))
                .count() as u64)
        })
    }

    fn add_alias(&self, primary: &Iri, alias: Iri) -> Result<(), StoreError> {
        let primary = primary.clone();
        self.with_write(|txn| {
            if doc::get_json::<String>(txn, IDS, alias.as_str())?.is_some() {
                return Err(StoreError::AlreadyExists(alias.clone()));
            }
            let Some(wire) = doc::get_json::<String>(txn, IDS, primary.as_str())? else {
                return Err(doc::not_owned(&self.label, &primary));
            };
            let (resolved, kind) = if wire == "alias" {
                let (resolved, wire) = {
                    let Some(named) =
                        doc::get_json::<String>(txn, super::doc::ALIASES, primary.as_str())?
                    else {
                        return Err(decode(format!("alias `{primary}` has no primary recorded")));
                    };
                    let resolved = Iri::parse(&named).map_err(decode)?;
                    let Some(wire) = doc::get_json::<String>(txn, IDS, resolved.as_str())? else {
                        return Err(decode(format!(
                            "alias `{primary}` names {resolved}, which this store does not hold"
                        )));
                    };
                    (resolved, wire)
                };
                let kind = Kind::from_wire(&wire)
                    .ok_or_else(|| decode(format!("unknown kind `{wire}` for {resolved}")))?;
                (resolved, kind)
            } else {
                let kind = Kind::from_wire(&wire)
                    .ok_or_else(|| decode(format!("unknown kind `{wire}` for {primary}")))?;
                (primary.clone(), kind)
            };
            doc::refuse_unwritable(txn, &resolved, kind)?;
            let table = match kind {
                Kind::Record => RECORDS,
                Kind::Finding => FINDINGS,
                other => {
                    return Err(StoreError::Backend(format!(
                        "{resolved} is a {}, and only a record or finding can carry an alias",
                        other.as_wire()
                    )));
                }
            };
            match kind {
                Kind::Record => {
                    let Some(mut record) = doc::get_json::<Record>(txn, table, resolved.as_str())?
                    else {
                        return Err(decode(format!(
                            "{resolved} is indexed as a {}, but this store holds no row for it",
                            kind.as_wire()
                        )));
                    };
                    record.also_known_as.push(alias.clone());
                    doc::put_json(txn, table, resolved.as_str(), &record)?;
                }
                Kind::Finding => {
                    let Some(mut finding) =
                        doc::get_json::<Finding>(txn, table, resolved.as_str())?
                    else {
                        return Err(decode(format!(
                            "{resolved} is indexed as a {}, but this store holds no row for it",
                            kind.as_wire()
                        )));
                    };
                    finding.also_known_as.push(alias.clone());
                    doc::put_json(txn, table, resolved.as_str(), &finding)?;
                }
                _ => unreachable!("checked above"),
            }
            doc::put_json(txn, IDS, alias.as_str(), &"alias")?;
            doc::put_json(txn, super::doc::ALIASES, alias.as_str(), &resolved.as_str())?;
            Ok(())
        })
    }
}

impl Escalations for FerroStore {
    fn mark(&self, id: &Iri, mark: &Mark) -> Result<(), StoreError> {
        let id = id.clone();
        let mark = mark.clone();
        self.with_write(|txn| {
            let (primary, kind) = doc::locate(txn, &self.label, &id)?;
            doc::refuse_escalated(txn, &primary)?;
            if !matches!(kind, Kind::Record | Kind::Finding) {
                return Err(StoreError::WrongKind {
                    id: id.clone(),
                    expected: Kind::Record,
                    found: kind,
                });
            }
            if doc::get_json::<Mark>(txn, ESCALATING, primary.as_str())?.is_some() {
                return Err(EscalationFault::AlreadyMarked { id: primary, kind }.into());
            }
            doc::put_json(txn, ESCALATING, primary.as_str(), &mark)?;
            doc::raise_format(txn, FORMAT_WITH_ROUTING)?;
            Ok(())
        })
    }

    fn mark_of(&self, id: &Iri) -> Result<Option<Mark>, StoreError> {
        let id = id.clone();
        self.with_read(|txn| {
            let Ok((primary, _)) = doc::locate(txn, &self.label, &id) else {
                return match doc::locate(txn, &self.label, &id) {
                    Err(StoreError::NotOwned { .. }) => Ok(None),
                    Err(err) => Err(err),
                    Ok(_) => unreachable!("locate failed and then succeeded"),
                };
            };
            doc::get_json(txn, ESCALATING, primary.as_str())
        })
    }

    fn unmark(&self, id: &Iri) -> Result<(), StoreError> {
        let id = id.clone();
        self.with_write(|txn| {
            let (primary, _) = doc::locate(txn, &self.label, &id)?;
            if doc::get_json::<Mark>(txn, ESCALATING, primary.as_str())?.is_none() {
                return Err(EscalationFault::NotMarked { id: primary }.into());
            }
            txn.delete(ESCALATING, primary.as_str());
            Ok(())
        })
    }

    fn tombstone(&self, id: &Iri, to: &Iri) -> Result<Tombstone, StoreError> {
        let id = id.clone();
        let to = to.clone();
        self.with_write(|txn| {
            let (primary, _) = doc::locate(txn, &self.label, &id)?;
            let Some(mark) = doc::take_json::<Mark>(txn, ESCALATING, primary.as_str())? else {
                return Err(EscalationFault::NotMarked { id: primary }.into());
            };
            let tomb = Tombstone {
                from: primary.clone(),
                to: to.clone(),
                by: mark.by,
                reason: mark.reason,
                at_ms: mark.at_ms,
            };
            doc::put_json(txn, TOMBSTONES, primary.as_str(), &tomb)?;
            Ok(tomb)
        })
    }

    fn tombstone_of(&self, id: &Iri) -> Result<Option<Tombstone>, StoreError> {
        let id = id.clone();
        self.with_read(|txn| {
            let primary = match doc::locate(txn, &self.label, &id) {
                Ok((primary, _)) => primary,
                Err(StoreError::NotOwned { .. }) => return Ok(None),
                Err(err) => return Err(err),
            };
            doc::get_json(txn, TOMBSTONES, primary.as_str())
        })
    }
}

impl Ledger for FerroStore {
    fn append_gate_run(&self, run: GateRun) -> Result<(), StoreError> {
        self.with_write(|txn| {
            let candidate = match (&run.id, &run.record) {
                (Some(id), Some(_)) => Some((CANDIDATE_RUNS, id)),
                _ => None,
            };
            append_log(txn, NEXT_RUN, GATE_RUNS, &run, candidate)
        })
    }

    fn append_attempt(&self, attempt: Attempt) -> Result<(), StoreError> {
        self.with_write(|txn| {
            let candidate = attempt.id.as_ref().map(|id| (CANDIDATE_ATTEMPTS, id));
            append_log(txn, NEXT_ATTEMPT, ATTEMPTS, &attempt, candidate)
        })
    }

    fn gate_runs(&self, gate: &GateId) -> Result<Vec<GateRun>, StoreError> {
        let gate = gate.clone();
        self.with_read(|txn| {
            doc::locate(txn, &self.label, gate.iri())?;
            let all = doc::scan_u64::<GateRun>(txn, GATE_RUNS)?;
            Ok(all
                .into_iter()
                .map(|(_, run)| run)
                .filter(|run| run.gate == gate)
                .collect())
        })
    }

    fn attempts(&self, project: &ProjectId) -> Result<Vec<Attempt>, StoreError> {
        let project = project.clone();
        self.with_read(|txn| {
            doc::check_kind(txn, &self.label, project.iri(), Kind::Project)?;
            let all = doc::scan_u64::<Attempt>(txn, ATTEMPTS)?;
            Ok(all
                .into_iter()
                .map(|(_, attempt)| attempt)
                .filter(|attempt| attempt.project == project)
                .collect())
        })
    }
}

impl Outbox for FerroStore {
    fn unpublished(&self, repo: &str, after: &Iri) -> Result<Pending, StoreError> {
        let repo = repo.to_string();
        let after = after.clone();
        self.with_read(|txn| {
            Ok(Pending {
                runs: doc::waiting(txn, CANDIDATE_RUNS, GATE_RUNS, &repo, &after)?,
                attempts: doc::waiting(txn, CANDIDATE_ATTEMPTS, ATTEMPTS, &repo, &after)?,
            })
        })
    }

    fn is_published(&self, repo: &str, id: &Iri) -> Result<bool, StoreError> {
        let repo = repo.to_string();
        let id = id.clone();
        self.with_read(|txn| {
            Ok(txn
                .get(LEDGER_PUBLISHED, &doc::pair(&repo, id.as_str()))
                .is_some())
        })
    }

    fn mark_published(&self, repo: &str, ids: &[Iri]) -> Result<(), StoreError> {
        self.settle(repo, ids, &[])
    }

    fn settle(&self, repo: &str, published: &[Iri], set_aside: &[Iri]) -> Result<(), StoreError> {
        let repo = repo.to_string();
        let published = published.to_vec();
        let set_aside = set_aside.to_vec();
        self.with_write(|txn| {
            for id in &published {
                doc::put_json(txn, LEDGER_PUBLISHED, &doc::pair(&repo, id.as_str()), &true)?;
            }
            doc::drop_candidates(txn, &published);
            doc::drop_candidates(txn, &set_aside);
            Ok(())
        })
    }

    fn cutover(&self, repo: &str) -> Result<Option<Iri>, StoreError> {
        let repo = repo.to_string();
        self.with_read(|txn| {
            let Some(raw) = doc::get_json::<String>(txn, LEDGER_CUTOVERS, &repo)? else {
                return Ok(None);
            };
            Iri::parse(&raw).map(Some).map_err(decode)
        })
    }

    fn set_cutover(&self, repo: &str, id: &Iri) -> Result<(), StoreError> {
        let repo = repo.to_string();
        let id = id.clone();
        self.with_write(|txn| {
            let held = doc::get_json::<String>(txn, LEDGER_CUTOVERS, &repo)?;
            match held {
                Some(held) if held == id.as_str() => Ok(()),
                Some(held) => Err(StoreError::CutoverChanged {
                    node_id: repo.clone(),
                    held: Iri::parse(&held).map_err(decode)?,
                    found: id.clone(),
                }),
                None => {
                    doc::put_json(txn, LEDGER_CUTOVERS, &repo, &id.as_str())?;
                    Ok(())
                }
            }
        })
    }
}

impl Handles for FerroStore {
    fn handle_of(&self, kind: Kind, id: &Iri) -> Result<Option<u64>, StoreError> {
        let id = id.clone();
        self.with_read(|txn| {
            match doc::get_json::<String>(txn, IDS, id.as_str())? {
                Some(wire) if wire == kind.as_wire() => {}
                _ => return Ok(None),
            }
            doc::get_json(txn, HANDLE_OF, id.as_str())
        })
    }

    fn resolve_handle(&self, kind: Kind, handle: u64) -> Result<Option<Iri>, StoreError> {
        self.with_read(|txn| {
            let Some(raw) =
                doc::get_json::<String>(txn, HANDLES, &doc::pair_u64(kind.as_wire(), handle))?
            else {
                return Ok(None);
            };
            Iri::parse(&raw).map(Some).map_err(decode)
        })
    }
}

impl Bindings for FerroStore {
    fn bound_node_id(&self, repo: &str) -> Result<Option<String>, StoreError> {
        let repo = repo.to_ascii_lowercase();
        self.with_read(|txn| doc::get_json(txn, GITHUB_BINDINGS, &repo))
    }

    fn bind_node_id(&self, repo: &str, node_id: &str) -> Result<(), StoreError> {
        let repo = repo.to_ascii_lowercase();
        let node_id = node_id.to_string();
        self.with_write(|txn| {
            doc::put_json(txn, GITHUB_BINDINGS, &repo, &node_id)?;
            Ok(())
        })
    }

    fn ledger_root(&self, node_id: &str) -> Result<Option<String>, StoreError> {
        let node_id = node_id.to_string();
        self.with_read(|txn| doc::get_json(txn, LEDGER_ROOTS, &node_id))
    }

    fn set_ledger_root(&self, node_id: &str, commit: &str) -> Result<(), StoreError> {
        let node_id = node_id.to_string();
        let commit = commit.to_string();
        self.with_write(|txn| {
            let held = doc::get_json::<String>(txn, LEDGER_ROOTS, &node_id)?;
            doc::refuse_a_changed_root(&node_id, held, &commit)?;
            doc::record_ledger_root(txn, &node_id, &commit)
        })
    }
}

impl LedgerCache for FerroStore {
    fn last_head(&self, repo: &str) -> Result<Option<String>, StoreError> {
        let repo = repo.to_string();
        self.with_read(|txn| doc::get_json(txn, LEDGER_HEADS, &repo))
    }

    fn cached(&self, repo: &str, path: &str) -> Result<Option<CachedSegment>, StoreError> {
        let key = doc::pair(repo, path);
        self.with_read(|txn| doc::get_json(txn, LEDGER_SEGMENTS, &key))
    }

    fn cached_under(
        &self,
        repo: &str,
        dir: &str,
    ) -> Result<Vec<(String, CachedSegment)>, StoreError> {
        let repo = repo.to_string();
        let prefix = format!("{dir}/");
        self.with_read(|txn| {
            let mut out = Vec::new();
            for (id, doc) in txn.scan(LEDGER_SEGMENTS) {
                let Some((held_repo, path)) = doc::split_pair(&id) else {
                    continue;
                };
                if held_repo != repo || !path.starts_with(&prefix) {
                    continue;
                }
                out.push((
                    path.to_string(),
                    serde_json::from_value(doc).map_err(decode)?,
                ));
            }
            out.sort_by(|a, b| a.0.cmp(&b.0));
            Ok(out)
        })
    }

    fn remember(
        &self,
        repo: &str,
        head: &str,
        segments: &[(String, CachedSegment)],
    ) -> Result<(), StoreError> {
        let repo = repo.to_string();
        let head = head.to_string();
        let segments = segments.to_vec();
        self.with_write(|txn| {
            doc::put_json(txn, LEDGER_HEADS, &repo, &head)?;
            for (path, segment) in &segments {
                doc::put_json(txn, LEDGER_SEGMENTS, &doc::pair(&repo, path), segment)?;
            }
            Ok(())
        })
    }
}
