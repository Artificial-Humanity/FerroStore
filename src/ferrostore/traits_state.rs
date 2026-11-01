//! Agent registry and state store.

use std::collections::BTreeMap;

use super::FerroStore;
use super::doc::{
    self, AGENT_BY_NAME, AGENT_EVENT_IDS, AGENT_EVENTS, AGENTS, AREAS, NEXT_AGENT_EVENT,
    NEXT_STATE_REV, STATE_CONFLICTS, STATE_ENTRIES, STATE_KEYS, STATE_REVISION_IDS,
    STATE_REVISIONS, STATE_SYNC_CURSORS,
};
use crate::{FORMAT_WITH_STATE, decode, mint};
use fl_core::agent::{
    AgentRecord, AgentRole, AgentStatus, AreaRecord, RegisterAgentArgs, RegistryEvent,
    RegistryEventKind, validate_agent_identity,
};
use fl_core::at::At;
use fl_core::ids::{AgentId, Kind, RevisionId, StateId};
use fl_core::iri::Iri;
use fl_core::state::{
    MemoryLifecycle, MergeMode, RuntimeEntry, SensitivityPatch, StateConcern, StateConflict,
    StateEntry, StatePayload, StateRevision, StateScope, StateSyncCursor, UpsertStateArgs,
    compute_append_merge, compute_last_writer_merge, is_ancestor_in_dag,
    lowest_common_ancestor_in_dag, sorted_parents, validate_slug,
};
use fl_core::store::{AgentRegistry, StateStore, StoreError};

fn append_registry_event(txn: &mut super::Txn, ev: &RegistryEvent) -> Result<u64, StoreError> {
    let seq = doc::bump_meta(txn, NEXT_AGENT_EVENT)?;
    doc::put_json(txn, AGENT_EVENTS, &doc::key_u64(seq), ev)?;
    doc::put_json(txn, AGENT_EVENT_IDS, ev.event_id.as_str(), &seq)?;
    Ok(seq)
}

fn append_state_revision(txn: &mut super::Txn, rev: &StateRevision) -> Result<u64, StoreError> {
    let seq = doc::bump_meta(txn, NEXT_STATE_REV)?;
    doc::put_json(txn, STATE_REVISIONS, &doc::key_u64(seq), rev)?;
    doc::put_json(txn, STATE_REVISION_IDS, rev.revision_id.as_str(), &seq)?;
    Ok(seq)
}

fn load_revisions(
    txn: &super::Txn,
    entry_id: &StateId,
) -> Result<BTreeMap<RevisionId, StateRevision>, StoreError> {
    let mut dag = BTreeMap::new();
    for (_seq, rev) in doc::scan_u64::<StateRevision>(txn, STATE_REVISIONS)? {
        if &rev.entry_id == entry_id {
            dag.insert(rev.revision_id.clone(), rev);
        }
    }
    Ok(dag)
}

fn revision_at(txn: &super::Txn, rev_id: &RevisionId) -> Result<StateRevision, StoreError> {
    let Some(seq) = doc::get_json::<u64>(txn, STATE_REVISION_IDS, rev_id.as_str())? else {
        return Err(StoreError::InvalidState(format!(
            "unknown revision `{rev_id}`"
        )));
    };
    doc::get_json(txn, STATE_REVISIONS, &doc::key_u64(seq))?
        .ok_or_else(|| decode("state_revision_ids points to missing sequence"))
}

fn apply_imported_state_revision(
    txn: &mut super::Txn,
    label: &str,
    rev: StateRevision,
    merge_at: At,
) -> Result<(), StoreError> {
    rev.payload
        .validate(&rev.entry_id)
        .map_err(|err| StoreError::InvalidState(err.to_string()))?;

    for parent in &rev.parent_revisions {
        if doc::get_json::<u64>(txn, STATE_REVISION_IDS, parent.as_str())?.is_none() {
            return Err(StoreError::InvalidState(format!(
                "revision `{}` references unknown parent `{}`",
                rev.revision_id, parent
            )));
        }
    }

    if let StatePayload::Memory(ref memory) = rev.payload
        && let Some(ref target_id) = memory.superseded_by
    {
        let resolved_target = doc::resolve_iri(txn, label, target_id.iri())?;
        match doc::get_json::<StateEntry>(txn, STATE_ENTRIES, resolved_target.as_str())? {
            Some(target) if target.concern == StateConcern::Memory => {}
            _ => {
                return Err(StoreError::InvalidState(format!(
                    "superseded_by target `{target_id}` does not exist as a Memory entry"
                )));
            }
        }
    }

    let scope_wire = rev.scope.as_wire();
    let concern_wire = rev.concern.as_wire();
    let slug = rev.slug.clone();
    let key = doc::triple(&scope_wire, concern_wire, &slug);

    if let Some(existing_id_str) = doc::get_json::<String>(txn, STATE_KEYS, &key)? {
        let existing_id = StateId(Iri::parse(&existing_id_str).map_err(decode)?);
        if existing_id != rev.entry_id {
            doc::alias_unpublished(txn, Kind::State, existing_id.iri(), rev.entry_id.iri())?;
            if let Some(mut local_entry) =
                doc::take_json::<StateEntry>(txn, STATE_ENTRIES, existing_id.as_str())?
            {
                local_entry.id = rev.entry_id.clone();
                doc::put_json(txn, STATE_ENTRIES, rev.entry_id.as_str(), &local_entry)?;
            }
            if let Some(mut conflict) =
                doc::take_json::<StateConflict>(txn, STATE_CONFLICTS, existing_id.as_str())?
            {
                conflict.entry_id = rev.entry_id.clone();
                doc::put_json(txn, STATE_CONFLICTS, rev.entry_id.as_str(), &conflict)?;
            }
            let mut updates = Vec::new();
            for (seq, mut stored) in doc::scan_u64::<StateRevision>(txn, STATE_REVISIONS)? {
                if stored.entry_id == existing_id {
                    stored.entry_id = rev.entry_id.clone();
                    updates.push((seq, stored));
                }
            }
            for (seq, stored) in updates {
                doc::put_json(txn, STATE_REVISIONS, &doc::key_u64(seq), &stored)?;
            }
            doc::put_json(txn, STATE_KEYS, &key, &rev.entry_id.as_str())?;
        }
    }

    let resolved_entry_id = StateId(doc::resolve_iri(txn, label, rev.entry_id.iri())?);
    let mut rev = rev;
    rev.entry_id = resolved_entry_id.clone();
    doc::index_imported(txn, Kind::State, rev.entry_id.iri())?;

    let existing_entry =
        doc::get_json::<StateEntry>(txn, STATE_ENTRIES, resolved_entry_id.as_str())?;
    if let Some(ref existing) = existing_entry
        && (existing.scope != rev.scope
            || existing.concern != rev.concern
            || existing.slug != rev.slug)
    {
        return Err(StoreError::Tampered {
            id: resolved_entry_id.0,
            detail: "revision changed immutable (scope, concern, slug) of existing state entry"
                .into(),
        });
    }

    append_state_revision(txn, &rev)?;

    let Some(existing) = existing_entry else {
        let entry = StateEntry {
            id: resolved_entry_id.clone(),
            scope: rev.scope,
            concern: rev.concern,
            slug: rev.slug,
            sensitive: rev.sensitive,
            ever_sensitive: rev.ever_sensitive || rev.sensitive,
            payload: rev.payload,
            head_revision: rev.revision_id,
            updated_at: rev.at,
            by: rev.by,
        };
        doc::put_json(txn, STATE_ENTRIES, resolved_entry_id.as_str(), &entry)?;
        doc::put_json(txn, STATE_KEYS, &key, &resolved_entry_id.as_str())?;
        return Ok(());
    };

    let dag = load_revisions(txn, &resolved_entry_id)?;
    let local_head = existing.head_revision.clone();
    let remote_head = rev.revision_id.clone();

    if is_ancestor_in_dag(&local_head, &remote_head, &dag) {
        let updated = StateEntry {
            id: resolved_entry_id.clone(),
            scope: existing.scope,
            concern: existing.concern,
            slug: existing.slug,
            sensitive: rev.sensitive,
            ever_sensitive: existing.ever_sensitive || rev.ever_sensitive || rev.sensitive,
            payload: rev.payload,
            head_revision: remote_head.clone(),
            updated_at: rev.at,
            by: rev.by,
        };
        doc::put_json(txn, STATE_ENTRIES, resolved_entry_id.as_str(), &updated)?;
        if let Some(conflict) =
            doc::get_json::<StateConflict>(txn, STATE_CONFLICTS, resolved_entry_id.as_str())?
            && is_ancestor_in_dag(&conflict.local_head, &remote_head, &dag)
            && is_ancestor_in_dag(&conflict.remote_head, &remote_head, &dag)
        {
            txn.delete(STATE_CONFLICTS, resolved_entry_id.as_str());
        }
        return Ok(());
    }

    if is_ancestor_in_dag(&remote_head, &local_head, &dag) {
        return Ok(());
    }

    let local_rev = dag.get(&local_head).cloned().ok_or_else(|| {
        StoreError::InvalidState(format!("missing local head revision `{local_head}`"))
    })?;
    let base_rev = lowest_common_ancestor_in_dag(&local_head, &remote_head, &dag)
        .and_then(|id| dag.get(&id).cloned());

    let auto_merged = match (&local_rev.payload, &rev.payload) {
        (StatePayload::Runtime(local_rt), StatePayload::Runtime(remote_rt))
            if local_rt.merge_mode == remote_rt.merge_mode =>
        {
            match local_rt.merge_mode {
                MergeMode::Conflict => None,
                MergeMode::Append => {
                    let base_arr = match base_rev.as_ref().map(|base| &base.payload) {
                        Some(StatePayload::Runtime(base_rt)) => base_rt.value.as_array().cloned(),
                        _ => Some(Vec::new()),
                    };
                    match (
                        base_arr,
                        local_rt.value.as_array(),
                        remote_rt.value.as_array(),
                    ) {
                        (Some(base), Some(local), Some(remote)) => {
                            Some(StatePayload::Runtime(RuntimeEntry {
                                value: serde_json::Value::Array(compute_append_merge(
                                    base, local, remote,
                                )),
                                merge_mode: MergeMode::Append,
                            }))
                        }
                        _ => None,
                    }
                }
                MergeMode::LastWriter => {
                    Some(compute_last_writer_merge(&local_rev, &rev).payload.clone())
                }
            }
        }
        _ => None,
    };

    if let Some(merged_payload) = auto_merged {
        let merge_rev_id = RevisionId(mint()?);
        let sensitive = local_rev.sensitive || rev.sensitive;
        let ever_sensitive =
            existing.ever_sensitive || local_rev.ever_sensitive || rev.ever_sensitive || sensitive;
        let merge_rev = StateRevision {
            revision_id: merge_rev_id.clone(),
            entry_id: resolved_entry_id.clone(),
            parent_revisions: sorted_parents(local_head, remote_head),
            scope: existing.scope.clone(),
            concern: existing.concern,
            slug: existing.slug.clone(),
            sensitive,
            ever_sensitive,
            payload: merged_payload.clone(),
            at: merge_at.clone(),
            by: "fl".to_string(),
        };
        append_state_revision(txn, &merge_rev)?;
        let updated = StateEntry {
            id: resolved_entry_id.clone(),
            scope: existing.scope,
            concern: existing.concern,
            slug: existing.slug,
            sensitive,
            ever_sensitive,
            payload: merged_payload,
            head_revision: merge_rev_id,
            updated_at: merge_at,
            by: "fl".to_string(),
        };
        doc::put_json(txn, STATE_ENTRIES, resolved_entry_id.as_str(), &updated)?;
        return Ok(());
    }

    let conflict = StateConflict {
        entry_id: resolved_entry_id.clone(),
        local_head,
        remote_head,
        detected_at: merge_at,
    };
    doc::put_json(txn, STATE_CONFLICTS, resolved_entry_id.as_str(), &conflict)?;
    Ok(())
}

impl AgentRegistry for FerroStore {
    fn upsert_area(&self, area: AreaRecord) -> Result<AreaRecord, StoreError> {
        let normalized = area
            .normalized()
            .map_err(|err| StoreError::InvalidState(err.to_string()))?;
        self.with_write(|txn| {
            doc::raise_format(txn, FORMAT_WITH_STATE)?;
            let ev = RegistryEvent {
                event_id: mint()?,
                at: normalized.updated_at.clone(),
                by: normalized.by.clone(),
                kind: RegistryEventKind::AreaUpsert(normalized.clone()),
            };
            doc::put_json(txn, AREAS, &normalized.area_key, &normalized)?;
            append_registry_event(txn, &ev)?;
            Ok(normalized.clone())
        })
    }

    fn get_area(&self, area_key: &str) -> Result<Option<AreaRecord>, StoreError> {
        let key =
            validate_slug(area_key).map_err(|err| StoreError::InvalidState(err.to_string()))?;
        self.with_read(|txn| doc::get_json(txn, AREAS, &key))
    }

    fn list_areas(&self) -> Result<Vec<AreaRecord>, StoreError> {
        self.with_read(|txn| {
            let mut out: Vec<AreaRecord> = doc::scan_json(txn, AREAS)?;
            out.sort_by(|a, b| a.area_key.cmp(&b.area_key));
            Ok(out)
        })
    }

    fn register_agent(&self, args: RegisterAgentArgs) -> Result<AgentRecord, StoreError> {
        let (name_key, display_name, email) =
            validate_agent_identity(&args.display_name, args.email.as_deref())
                .map_err(StoreError::InvalidState)?;
        let project = validate_slug(&args.project)
            .map_err(|err| StoreError::InvalidState(err.to_string()))?;
        let mut extra_areas = Vec::with_capacity(args.extra_areas.len());
        for area in &args.extra_areas {
            extra_areas.push(
                validate_slug(area).map_err(|err| StoreError::InvalidState(err.to_string()))?,
            );
        }
        extra_areas.sort();
        extra_areas.dedup();

        self.with_write(|txn| {
            doc::raise_format(txn, FORMAT_WITH_STATE)?;
            let mut pending_area_events = Vec::new();
            for area_key in std::iter::once(&project).chain(extra_areas.iter()) {
                if doc::get_json::<AreaRecord>(txn, AREAS, area_key)?.is_some() {
                    continue;
                }
                if !args.create_area {
                    return Err(StoreError::NoSuchArea(area_key.clone()));
                }
                let area = AreaRecord {
                    area_key: area_key.clone(),
                    name: area_key.clone(),
                    path: None,
                    purpose: format!("Registered area {area_key}"),
                    updated_at: args.at.clone(),
                    by: args.by.clone(),
                };
                doc::put_json(txn, AREAS, area_key, &area)?;
                pending_area_events.push(RegistryEvent {
                    event_id: mint()?,
                    at: args.at.clone(),
                    by: args.by.clone(),
                    kind: RegistryEventKind::AreaUpsert(area),
                });
            }
            for ev in &pending_area_events {
                append_registry_event(txn, ev)?;
            }

            let existing_by_name = match doc::get_json::<String>(txn, AGENT_BY_NAME, &name_key)? {
                Some(id_str) => doc::get_json::<AgentRecord>(txn, AGENTS, &id_str)?,
                None => None,
            };
            if let Some(ref existing) = existing_by_name
                && !args.replace
            {
                return Err(StoreError::AgentCollision {
                    name_key: name_key.clone(),
                    existing_id: existing.id.clone(),
                    status: existing.status,
                });
            }

            if args.role == AgentRole::Lead {
                let conflicting = doc::scan_json::<AgentRecord>(txn, AGENTS)?
                    .into_iter()
                    .find(|agent| {
                        agent.status == AgentStatus::Active
                            && agent.project == project
                            && agent.role == AgentRole::Lead
                            && agent.name_key != name_key
                    });
                if let Some(mut lead) = conflicting {
                    if !args.replace {
                        return Err(StoreError::LeadTaken {
                            project: project.clone(),
                            lead_name: lead.display_name,
                            lead_id: lead.id,
                        });
                    }
                    lead.status = AgentStatus::Retired;
                    lead.updated_at = args.at.clone();
                    lead.by = args.by.clone();
                    doc::put_json(txn, AGENTS, lead.id.as_str(), &lead)?;
                    let retire_ev = RegistryEvent {
                        event_id: mint()?,
                        at: args.at.clone(),
                        by: args.by.clone(),
                        kind: RegistryEventKind::AgentRetire { id: lead.id },
                    };
                    append_registry_event(txn, &retire_ev)?;
                }
            }

            let agent_id = match existing_by_name {
                Some(existing) => existing.id,
                None => {
                    let iri = mint()?;
                    doc::index_new(txn, &iri, Kind::Agent)?;
                    AgentId(iri)
                }
            };
            let record = AgentRecord {
                id: agent_id.clone(),
                name_key: name_key.clone(),
                display_name: display_name.clone(),
                email: email.clone(),
                project: project.clone(),
                role: args.role.clone(),
                extra_areas: extra_areas.clone(),
                status: AgentStatus::Active,
                updated_at: args.at.clone(),
                by: args.by.clone(),
            };
            doc::put_json(txn, AGENT_BY_NAME, &name_key, &agent_id.as_str())?;
            doc::put_json(txn, AGENTS, agent_id.as_str(), &record)?;
            let ev = RegistryEvent {
                event_id: mint()?,
                at: args.at.clone(),
                by: args.by.clone(),
                kind: RegistryEventKind::AgentUpsert(record.clone()),
            };
            append_registry_event(txn, &ev)?;
            Ok(record)
        })
    }

    fn retire_agent(&self, id: &AgentId, at: At, by: &str) -> Result<AgentRecord, StoreError> {
        let id = id.clone();
        let at = at.clone();
        let by = by.to_string();
        self.with_write(|txn| {
            doc::raise_format(txn, FORMAT_WITH_STATE)?;
            let (resolved_iri, kind) = doc::locate(txn, &self.label, id.iri())?;
            if kind != Kind::Agent {
                return Err(StoreError::WrongKind {
                    id: id.iri().clone(),
                    expected: Kind::Agent,
                    found: kind,
                });
            }
            let resolved = AgentId(resolved_iri);
            let Some(mut rec) = doc::get_json::<AgentRecord>(txn, AGENTS, resolved.as_str())?
            else {
                return Err(StoreError::NoSuchAgent(resolved.to_string()));
            };
            rec.status = AgentStatus::Retired;
            rec.updated_at = at.clone();
            rec.by = by.trim().to_string();
            doc::put_json(txn, AGENTS, resolved.as_str(), &rec)?;
            let ev = RegistryEvent {
                event_id: mint()?,
                at: at.clone(),
                by: rec.by.clone(),
                kind: RegistryEventKind::AgentRetire { id: resolved },
            };
            append_registry_event(txn, &ev)?;
            Ok(rec)
        })
    }

    fn get_agent(&self, id: &AgentId) -> Result<Option<AgentRecord>, StoreError> {
        let id = id.clone();
        self.with_read(|txn| {
            let (target, kind) = doc::locate(txn, &self.label, id.iri())?;
            if kind != Kind::Agent {
                return Ok(None);
            }
            doc::get_json(txn, AGENTS, target.as_str())
        })
    }

    fn agent_by_name(&self, name: &str) -> Result<Option<AgentRecord>, StoreError> {
        let key = validate_slug(name).map_err(|err| StoreError::InvalidState(err.to_string()))?;
        self.with_read(|txn| {
            let Some(id_str) = doc::get_json::<String>(txn, AGENT_BY_NAME, &key)? else {
                return Ok(None);
            };
            let id_iri = Iri::parse(&id_str).map_err(decode)?;
            let (target, kind) = doc::locate(txn, &self.label, &id_iri)?;
            if kind != Kind::Agent {
                return Ok(None);
            }
            doc::get_json(txn, AGENTS, target.as_str())
        })
    }

    fn list_agents(
        &self,
        project: Option<&str>,
        include_retired: bool,
    ) -> Result<Vec<AgentRecord>, StoreError> {
        let project_filter = match project {
            Some(project) => Some(
                validate_slug(project).map_err(|err| StoreError::InvalidState(err.to_string()))?,
            ),
            None => None,
        };
        self.with_read(|txn| {
            let mut out = Vec::new();
            for agent in doc::scan_json::<AgentRecord>(txn, AGENTS)? {
                if !include_retired && agent.status != AgentStatus::Active {
                    continue;
                }
                if let Some(ref project) = project_filter
                    && &agent.project != project
                    && !agent.extra_areas.contains(project)
                {
                    continue;
                }
                out.push(agent);
            }
            out.sort_by(|a, b| a.name_key.cmp(&b.name_key));
            Ok(out)
        })
    }

    fn registry_events_after(&self, seq: u64) -> Result<Vec<(u64, RegistryEvent)>, StoreError> {
        self.with_read(|txn| {
            Ok(doc::scan_u64::<RegistryEvent>(txn, AGENT_EVENTS)?
                .into_iter()
                .filter(|(row, _)| *row > seq)
                .collect())
        })
    }

    fn import_registry_events(&self, events: &[RegistryEvent]) -> Result<(), StoreError> {
        let events = events.to_vec();
        self.with_write(|txn| {
            doc::raise_format(txn, FORMAT_WITH_STATE)?;
            for ev in &events {
                if let Some(seq) = doc::get_json::<u64>(txn, AGENT_EVENT_IDS, ev.event_id.as_str())?
                {
                    let existing =
                        doc::get_json::<RegistryEvent>(txn, AGENT_EVENTS, &doc::key_u64(seq))?
                            .ok_or_else(|| decode("agent_event_ids points to missing sequence"))?;
                    if &existing != ev {
                        return Err(StoreError::Tampered {
                            id: ev.event_id.clone(),
                            detail: "registry event payload differs from stored copy".into(),
                        });
                    }
                    continue;
                }
                match &ev.kind {
                    RegistryEventKind::AreaUpsert(area) => {
                        let norm = area
                            .normalized()
                            .map_err(|err| StoreError::InvalidState(err.to_string()))?;
                        doc::put_json(txn, AREAS, &norm.area_key, &norm)?;
                    }
                    RegistryEventKind::AgentUpsert(rec) => {
                        if let Some(existing_id_str) =
                            doc::get_json::<String>(txn, AGENT_BY_NAME, &rec.name_key)?
                        {
                            let existing_id =
                                AgentId(Iri::parse(&existing_id_str).map_err(decode)?);
                            if existing_id != rec.id {
                                doc::alias_unpublished(
                                    txn,
                                    Kind::Agent,
                                    existing_id.iri(),
                                    rec.id.iri(),
                                )?;
                                txn.delete(AGENTS, existing_id.as_str());
                            }
                        }
                        if rec.role == AgentRole::Lead && rec.status == AgentStatus::Active {
                            let mut to_retire = Vec::new();
                            for agent in doc::scan_json::<AgentRecord>(txn, AGENTS)? {
                                if agent.project == rec.project
                                    && agent.role == AgentRole::Lead
                                    && agent.status == AgentStatus::Active
                                    && agent.id != rec.id
                                {
                                    let mut other = agent;
                                    other.status = AgentStatus::Retired;
                                    to_retire.push(other);
                                }
                            }
                            for other in to_retire {
                                doc::put_json(txn, AGENTS, other.id.as_str(), &other)?;
                            }
                        }
                        doc::index_imported(txn, Kind::Agent, rec.id.iri())?;
                        doc::put_json(txn, AGENT_BY_NAME, &rec.name_key, &rec.id.as_str())?;
                        doc::put_json(txn, AGENTS, rec.id.as_str(), rec)?;
                    }
                    RegistryEventKind::AgentRetire { id } => {
                        let resolved = AgentId(doc::resolve_iri(txn, &self.label, id.iri())?);
                        if let Some(mut rec) =
                            doc::get_json::<AgentRecord>(txn, AGENTS, resolved.as_str())?
                        {
                            rec.status = AgentStatus::Retired;
                            rec.updated_at = ev.at.clone();
                            rec.by = ev.by.clone();
                            doc::put_json(txn, AGENTS, resolved.as_str(), &rec)?;
                        }
                    }
                }
                append_registry_event(txn, ev)?;
            }
            Ok(())
        })
    }
}

impl StateStore for FerroStore {
    fn upsert_state(&self, args: UpsertStateArgs) -> Result<StateEntry, StoreError> {
        let slug =
            validate_slug(&args.slug).map_err(|err| StoreError::InvalidState(err.to_string()))?;
        let concern = args.payload.concern();
        let scope_wire = args.scope.as_wire();
        let concern_wire = concern.as_wire();
        self.with_write(|txn| {
            doc::raise_format(txn, FORMAT_WITH_STATE)?;
            let key = doc::triple(&scope_wire, concern_wire, &slug);
            let existing = match doc::get_json::<String>(txn, STATE_KEYS, &key)? {
                Some(id_str) => doc::get_json::<StateEntry>(txn, STATE_ENTRIES, &id_str)?,
                None => None,
            };
            if let Some(ref existing) = existing
                && let Some(conflict) =
                    doc::get_json::<StateConflict>(txn, STATE_CONFLICTS, existing.id.as_str())?
            {
                return Err(StoreError::StateConflict {
                    entry_id: existing.id.clone(),
                    local_head: conflict.local_head,
                    remote_head: conflict.remote_head,
                });
            }
            let entry_id = match existing {
                Some(ref existing) => existing.id.clone(),
                None => {
                    let iri = mint()?;
                    doc::index_new(txn, &iri, Kind::State)?;
                    StateId(iri)
                }
            };
            args.payload
                .validate(&entry_id)
                .map_err(|err| StoreError::InvalidState(err.to_string()))?;
            if let StatePayload::Memory(ref memory) = args.payload
                && let Some(ref target_id) = memory.superseded_by
            {
                let resolved_target = doc::resolve_iri(txn, &self.label, target_id.iri())?;
                match doc::get_json::<StateEntry>(txn, STATE_ENTRIES, resolved_target.as_str())? {
                    Some(target) if target.concern == StateConcern::Memory => {}
                    _ => {
                        return Err(StoreError::InvalidState(format!(
                            "superseded_by target `{target_id}` does not exist as a Memory entry"
                        )));
                    }
                }
            }
            let (sensitive, ever_sensitive, parent_revisions) = match existing {
                Some(ref existing) => {
                    let sensitive = match args.sensitivity {
                        SensitivityPatch::Inherit => existing.sensitive,
                        SensitivityPatch::SetSensitive => true,
                        SensitivityPatch::ClearSensitive => false,
                    };
                    (
                        sensitive,
                        existing.ever_sensitive || sensitive,
                        vec![existing.head_revision.clone()],
                    )
                }
                None => {
                    let sensitive = matches!(args.sensitivity, SensitivityPatch::SetSensitive);
                    (sensitive, sensitive, Vec::new())
                }
            };
            let rev_id = RevisionId(mint()?);
            let rev = StateRevision {
                revision_id: rev_id.clone(),
                entry_id: entry_id.clone(),
                parent_revisions,
                scope: args.scope.clone(),
                concern,
                slug: slug.clone(),
                sensitive,
                ever_sensitive,
                payload: args.payload.clone(),
                at: args.at.clone(),
                by: args.by.trim().to_string(),
            };
            let entry = StateEntry {
                id: entry_id.clone(),
                scope: args.scope.clone(),
                concern,
                slug: slug.clone(),
                sensitive,
                ever_sensitive,
                payload: args.payload.clone(),
                head_revision: rev_id,
                updated_at: args.at.clone(),
                by: args.by.trim().to_string(),
            };
            append_state_revision(txn, &rev)?;
            doc::put_json(txn, STATE_ENTRIES, entry_id.as_str(), &entry)?;
            doc::put_json(txn, STATE_KEYS, &key, &entry_id.as_str())?;
            Ok(entry)
        })
    }

    fn retire_memory(
        &self,
        scope: &StateScope,
        slug: &str,
        lifecycle: MemoryLifecycle,
        superseded_by: Option<StateId>,
        at: At,
        by: &str,
    ) -> Result<StateEntry, StoreError> {
        let slug = validate_slug(slug).map_err(|err| StoreError::InvalidState(err.to_string()))?;
        if matches!(lifecycle, MemoryLifecycle::Active) {
            return Err(StoreError::InvalidState(
                "retire_memory requires `retired` or `superseded` lifecycle".into(),
            ));
        }
        let current = self
            .get_state_by_key(scope, StateConcern::Memory, &slug)?
            .ok_or_else(|| StoreError::NoSuchState(format!("{}:memory:{slug}", scope.as_wire())))?;
        let StatePayload::Memory(mut memory) = current.payload else {
            return Err(StoreError::InvalidState(format!(
                "state entry `{}` is not a Memory entry",
                current.id
            )));
        };
        memory.lifecycle = lifecycle;
        memory.superseded_by = superseded_by;
        self.upsert_state(UpsertStateArgs {
            scope: scope.clone(),
            slug,
            sensitivity: SensitivityPatch::Inherit,
            payload: StatePayload::Memory(memory),
            at,
            by: by.to_string(),
        })
    }

    fn get_state(&self, id: &StateId) -> Result<Option<StateEntry>, StoreError> {
        let id = id.clone();
        self.with_read(|txn| {
            let (target, kind) = doc::locate(txn, &self.label, id.iri())?;
            if kind != Kind::State {
                return Ok(None);
            }
            doc::get_json(txn, STATE_ENTRIES, target.as_str())
        })
    }

    fn get_state_by_key(
        &self,
        scope: &StateScope,
        concern: StateConcern,
        slug: &str,
    ) -> Result<Option<StateEntry>, StoreError> {
        let slug = validate_slug(slug).map_err(|err| StoreError::InvalidState(err.to_string()))?;
        let key = doc::triple(&scope.as_wire(), concern.as_wire(), &slug);
        self.with_read(|txn| {
            let Some(id_str) = doc::get_json::<String>(txn, STATE_KEYS, &key)? else {
                return Ok(None);
            };
            let id_iri = Iri::parse(&id_str).map_err(decode)?;
            let (target, kind) = doc::locate(txn, &self.label, &id_iri)?;
            if kind != Kind::State {
                return Ok(None);
            }
            doc::get_json(txn, STATE_ENTRIES, target.as_str())
        })
    }

    fn list_state(
        &self,
        scope: Option<&StateScope>,
        concern: Option<StateConcern>,
    ) -> Result<Vec<StateEntry>, StoreError> {
        let scope = scope.cloned();
        self.with_read(|txn| {
            let mut out = Vec::new();
            for entry in doc::scan_json::<StateEntry>(txn, STATE_ENTRIES)? {
                if scope.as_ref().is_some_and(|want| &entry.scope != want) {
                    continue;
                }
                if concern.is_some_and(|want| entry.concern != want) {
                    continue;
                }
                out.push(entry);
            }
            out.sort_by(|a, b| (&a.scope, a.concern, &a.slug).cmp(&(&b.scope, b.concern, &b.slug)));
            Ok(out)
        })
    }

    fn state_revisions(&self, id: &StateId) -> Result<Vec<StateRevision>, StoreError> {
        let id = id.clone();
        self.with_read(|txn| {
            let (primary, kind) = doc::locate(txn, &self.label, id.iri())?;
            if kind != Kind::State {
                return Err(StoreError::WrongKind {
                    id: id.iri().clone(),
                    expected: Kind::State,
                    found: kind,
                });
            }
            let resolved = StateId(primary);
            let mut out = Vec::new();
            for (_seq, rev) in doc::scan_u64::<StateRevision>(txn, STATE_REVISIONS)? {
                if rev.entry_id == resolved {
                    out.push(rev);
                }
            }
            Ok(out)
        })
    }

    fn list_conflicts(&self) -> Result<Vec<StateConflict>, StoreError> {
        self.with_read(|txn| doc::scan_json(txn, STATE_CONFLICTS))
    }

    fn resolve_conflict(
        &self,
        id: &StateId,
        keep: &RevisionId,
        at: At,
        by: &str,
    ) -> Result<StateEntry, StoreError> {
        let id = id.clone();
        let keep = keep.clone();
        let at = at.clone();
        let by = by.to_string();
        self.with_write(|txn| {
            doc::raise_format(txn, FORMAT_WITH_STATE)?;
            let (resolved_iri, kind) = doc::locate(txn, &self.label, id.iri())?;
            if kind != Kind::State {
                return Err(StoreError::WrongKind {
                    id: id.iri().clone(),
                    expected: Kind::State,
                    found: kind,
                });
            }
            let resolved = StateId(resolved_iri);
            let Some(conflict) =
                doc::get_json::<StateConflict>(txn, STATE_CONFLICTS, resolved.as_str())?
            else {
                return Err(StoreError::InvalidState(format!(
                    "state entry `{resolved}` has no active conflict"
                )));
            };
            if keep != conflict.local_head && keep != conflict.remote_head {
                return Err(StoreError::InvalidState(format!(
                    "`--keep {keep}` must match either local head `{}` or remote head `{}`",
                    conflict.local_head, conflict.remote_head
                )));
            }
            let chosen = revision_at(txn, &keep)?;
            let local_rev = revision_at(txn, &conflict.local_head)?;
            let remote_rev = revision_at(txn, &conflict.remote_head)?;
            let existing = doc::get_json::<StateEntry>(txn, STATE_ENTRIES, resolved.as_str())?
                .ok_or_else(|| StoreError::NoSuchState(resolved.to_string()))?;
            let merge_rev_id = RevisionId(mint()?);
            let sensitive = local_rev.sensitive || remote_rev.sensitive;
            let ever_sensitive = existing.ever_sensitive
                || local_rev.ever_sensitive
                || remote_rev.ever_sensitive
                || sensitive;
            let merge_rev = StateRevision {
                revision_id: merge_rev_id.clone(),
                entry_id: resolved.clone(),
                parent_revisions: sorted_parents(conflict.local_head, conflict.remote_head),
                scope: existing.scope.clone(),
                concern: existing.concern,
                slug: existing.slug.clone(),
                sensitive,
                ever_sensitive,
                payload: chosen.payload.clone(),
                at: at.clone(),
                by: by.trim().to_string(),
            };
            append_state_revision(txn, &merge_rev)?;
            let updated = StateEntry {
                id: resolved.clone(),
                scope: existing.scope,
                concern: existing.concern,
                slug: existing.slug,
                sensitive,
                ever_sensitive,
                payload: chosen.payload,
                head_revision: merge_rev_id,
                updated_at: at.clone(),
                by: by.trim().to_string(),
            };
            txn.delete(STATE_CONFLICTS, resolved.as_str());
            doc::put_json(txn, STATE_ENTRIES, resolved.as_str(), &updated)?;
            Ok(updated)
        })
    }

    fn state_revisions_after(&self, seq: u64) -> Result<Vec<(u64, StateRevision)>, StoreError> {
        self.with_read(|txn| {
            Ok(doc::scan_u64::<StateRevision>(txn, STATE_REVISIONS)?
                .into_iter()
                .filter(|(row, _)| *row > seq)
                .collect())
        })
    }

    fn import_state_revisions(
        &self,
        revs: &[StateRevision],
        merge_at: At,
    ) -> Result<(), StoreError> {
        let revs = revs.to_vec();
        self.with_write(|txn| {
            doc::raise_format(txn, FORMAT_WITH_STATE)?;
            for rev in &revs {
                if let Some(seq) =
                    doc::get_json::<u64>(txn, STATE_REVISION_IDS, rev.revision_id.as_str())?
                {
                    let existing = doc::get_json::<StateRevision>(
                        txn,
                        STATE_REVISIONS,
                        &doc::key_u64(seq),
                    )?
                    .ok_or_else(|| decode("state_revision_ids points to missing sequence"))?;
                    let mut normalized = rev.clone();
                    normalized.entry_id =
                        StateId(doc::resolve_iri(txn, &self.label, rev.entry_id.iri())?);
                    if existing != normalized {
                        return Err(StoreError::Tampered {
                            id: rev.revision_id.0.clone(),
                            detail: "state revision payload differs from stored copy".into(),
                        });
                    }
                    continue;
                }
                apply_imported_state_revision(txn, &self.label, rev.clone(), merge_at.clone())?;
            }
            Ok(())
        })
    }

    fn sync_cursor(&self, remote: &str) -> Result<Option<StateSyncCursor>, StoreError> {
        let remote = remote.to_string();
        self.with_read(|txn| doc::get_json(txn, STATE_SYNC_CURSORS, &remote))
    }

    fn set_sync_cursor(&self, cursor: StateSyncCursor) -> Result<(), StoreError> {
        self.with_write(|txn| {
            doc::raise_format(txn, FORMAT_WITH_STATE)?;
            doc::put_json(txn, STATE_SYNC_CURSORS, &cursor.remote, &cursor)?;
            Ok(())
        })
    }
}
