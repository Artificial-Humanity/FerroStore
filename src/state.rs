//! `RedbStore` persistence for [`StateStore`] and [`AgentRegistry`] (Format 6).

use std::collections::BTreeMap;

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
use redb::{ReadableTable, TableDefinition};

use crate::{
    ALIASES, FORMAT_WITH_STATE, HANDLE_OF, HANDLES, IDS, META, RedbStore, additive, backend,
    decode, index_new, locate_in, mint, raise_format,
};

pub(crate) const STATE_ENTRIES: TableDefinition<&str, &str> = TableDefinition::new("state_entries");
pub(crate) const STATE_KEYS: TableDefinition<(&str, &str, &str), &str> =
    TableDefinition::new("state_keys");
pub(crate) const STATE_REVISIONS: TableDefinition<u64, &str> =
    TableDefinition::new("state_revisions");
pub(crate) const STATE_REVISION_IDS: TableDefinition<&str, u64> =
    TableDefinition::new("state_revision_ids");
pub(crate) const STATE_CONFLICTS: TableDefinition<&str, &str> =
    TableDefinition::new("state_conflicts");
pub(crate) const STATE_SYNC_CURSORS: TableDefinition<&str, &str> =
    TableDefinition::new("state_sync_cursors");
pub(crate) const AGENTS: TableDefinition<&str, &str> = TableDefinition::new("agents");
pub(crate) const AGENT_BY_NAME: TableDefinition<&str, &str> = TableDefinition::new("agent_by_name");
pub(crate) const AREAS: TableDefinition<&str, &str> = TableDefinition::new("areas");
pub(crate) const AGENT_EVENTS: TableDefinition<u64, &str> = TableDefinition::new("agent_events");
pub(crate) const AGENT_EVENT_IDS: TableDefinition<&str, u64> =
    TableDefinition::new("agent_event_ids");

const NEXT_STATE_REV: &str = "next_state_rev";
const NEXT_AGENT_EVENT: &str = "next_agent_event";

impl RedbStore {
    pub(crate) fn resolve_in_wtx(
        &self,
        tx: &redb::WriteTransaction,
        id: &Iri,
    ) -> Result<Iri, StoreError> {
        let ids = tx.open_table(IDS).map_err(backend)?;
        match locate_in(&self.label, id, &ids, || {
            tx.open_table(ALIASES).map_err(backend)
        }) {
            Ok((primary, _)) => Ok(primary),
            Err(StoreError::NotOwned { .. }) => Ok(id.clone()),
            Err(e) => Err(e),
        }
    }

    pub(crate) fn index_imported_in(
        &self,
        tx: &redb::WriteTransaction,
        kind: Kind,
        id: &Iri,
    ) -> Result<(), StoreError> {
        let already = {
            let ids = tx.open_table(IDS).map_err(backend)?;
            ids.get(id.as_str())
                .map_err(backend)?
                .map(|g| g.value().to_owned())
        };
        if let Some(wire) = already {
            if wire == "alias" {
                return Ok(());
            }
            let found = Kind::from_wire(&wire)
                .ok_or_else(|| decode(format!("unknown kind `{wire}` for {id}")))?;
            if found != kind {
                return Err(StoreError::WrongKind {
                    id: id.clone(),
                    expected: kind,
                    found,
                });
            }
            return Ok(());
        }
        index_new(tx, id, kind)
    }

    pub(crate) fn alias_unpublished_id_in(
        &self,
        tx: &redb::WriteTransaction,
        kind: Kind,
        old_id: &Iri,
        new_id: &Iri,
    ) -> Result<(), StoreError> {
        if old_id == new_id {
            return Ok(());
        }
        {
            let mut ids = tx.open_table(IDS).map_err(backend)?;
            let mut aliases = tx.open_table(ALIASES).map_err(backend)?;
            ids.insert(old_id.as_str(), "alias").map_err(backend)?;
            aliases
                .insert(old_id.as_str(), new_id.as_str())
                .map_err(backend)?;

            let mut existing_aliases = Vec::new();
            for row in aliases.iter().map_err(backend)? {
                let (k, v) = row.map_err(backend)?;
                if v.value() == old_id.as_str() && k.value() != old_id.as_str() {
                    existing_aliases.push(k.value().to_owned());
                }
            }
            for alias_key in existing_aliases {
                aliases
                    .insert(alias_key.as_str(), new_id.as_str())
                    .map_err(backend)?;
            }
            ids.insert(new_id.as_str(), kind.as_wire())
                .map_err(backend)?;
        }

        let old_handle = {
            let mut handle_of = tx.open_table(HANDLE_OF).map_err(backend)?;
            handle_of
                .remove(old_id.as_str())
                .map_err(backend)?
                .map(|g| g.value())
        };
        if let Some(h) = old_handle {
            tx.open_table(HANDLES)
                .map_err(backend)?
                .insert((kind.as_wire(), h), new_id.as_str())
                .map_err(backend)?;
            tx.open_table(HANDLE_OF)
                .map_err(backend)?
                .insert(new_id.as_str(), h)
                .map_err(backend)?;
        } else {
            let has_handle = tx
                .open_table(HANDLE_OF)
                .map_err(backend)?
                .get(new_id.as_str())
                .map_err(backend)?
                .is_some();
            if !has_handle {
                let key = format!("next_handle:{}", kind.as_wire());
                let mut meta = tx.open_table(META).map_err(backend)?;
                let n = meta
                    .get(key.as_str())
                    .map_err(backend)?
                    .map(|v| v.value())
                    .unwrap_or(0)
                    + 1;
                meta.insert(key.as_str(), n).map_err(backend)?;
                tx.open_table(HANDLES)
                    .map_err(backend)?
                    .insert((kind.as_wire(), n), new_id.as_str())
                    .map_err(backend)?;
                tx.open_table(HANDLE_OF)
                    .map_err(backend)?
                    .insert(new_id.as_str(), n)
                    .map_err(backend)?;
            }
        }
        Ok(())
    }

    fn append_registry_event_in(
        &self,
        tx: &redb::WriteTransaction,
        ev: &RegistryEvent,
    ) -> Result<u64, StoreError> {
        let seq = {
            let mut meta = tx.open_table(META).map_err(backend)?;
            let next = meta
                .get(NEXT_AGENT_EVENT)
                .map_err(backend)?
                .map(|g| g.value())
                .unwrap_or(0)
                + 1;
            meta.insert(NEXT_AGENT_EVENT, next).map_err(backend)?;
            next
        };
        let json = serde_json::to_string(ev).map_err(backend)?;
        tx.open_table(AGENT_EVENTS)
            .map_err(backend)?
            .insert(seq, json.as_str())
            .map_err(backend)?;
        tx.open_table(AGENT_EVENT_IDS)
            .map_err(backend)?
            .insert(ev.event_id.as_str(), seq)
            .map_err(backend)?;
        Ok(seq)
    }

    fn append_state_revision_in(
        &self,
        tx: &redb::WriteTransaction,
        rev: &StateRevision,
    ) -> Result<u64, StoreError> {
        let seq = {
            let mut meta = tx.open_table(META).map_err(backend)?;
            let next = meta
                .get(NEXT_STATE_REV)
                .map_err(backend)?
                .map(|g| g.value())
                .unwrap_or(0)
                + 1;
            meta.insert(NEXT_STATE_REV, next).map_err(backend)?;
            next
        };
        let json = serde_json::to_string(rev).map_err(backend)?;
        tx.open_table(STATE_REVISIONS)
            .map_err(backend)?
            .insert(seq, json.as_str())
            .map_err(backend)?;
        tx.open_table(STATE_REVISION_IDS)
            .map_err(backend)?
            .insert(rev.revision_id.as_str(), seq)
            .map_err(backend)?;
        Ok(seq)
    }

    fn load_revisions_for_entry_in(
        &self,
        tx: &redb::WriteTransaction,
        entry_id: &StateId,
    ) -> Result<BTreeMap<RevisionId, StateRevision>, StoreError> {
        let revs = tx.open_table(STATE_REVISIONS).map_err(backend)?;
        let mut dag = BTreeMap::new();
        for row in revs.iter().map_err(backend)? {
            let (_seq, raw) = row.map_err(backend)?;
            let r: StateRevision = serde_json::from_str(raw.value()).map_err(decode)?;
            if &r.entry_id == entry_id {
                dag.insert(r.revision_id.clone(), r);
            }
        }
        Ok(dag)
    }

    fn apply_imported_state_revision_in(
        &self,
        tx: &redb::WriteTransaction,
        rev: StateRevision,
        merge_at: At,
    ) -> Result<(), StoreError> {
        rev.payload
            .validate(&rev.entry_id)
            .map_err(|e| StoreError::InvalidState(e.to_string()))?;

        {
            let by_id = tx.open_table(STATE_REVISION_IDS).map_err(backend)?;
            for parent in &rev.parent_revisions {
                if by_id.get(parent.as_str()).map_err(backend)?.is_none() {
                    return Err(StoreError::InvalidState(format!(
                        "revision `{}` references unknown parent `{}`",
                        rev.revision_id, parent
                    )));
                }
            }
        }

        if let StatePayload::Memory(ref m) = rev.payload
            && let Some(ref target_id) = m.superseded_by
        {
            let resolved_target = self.resolve_in_wtx(tx, target_id.iri())?;
            let entries = tx.open_table(STATE_ENTRIES).map_err(backend)?;
            let raw = entries
                .get(resolved_target.as_str())
                .map_err(backend)?
                .map(|g| g.value().to_owned());
            match raw {
                Some(json) => {
                    let target: StateEntry = serde_json::from_str(&json).map_err(decode)?;
                    if target.concern != StateConcern::Memory {
                        return Err(StoreError::InvalidState(format!(
                            "superseded_by target `{target_id}` does not exist as a Memory entry"
                        )));
                    }
                }
                None => {
                    return Err(StoreError::InvalidState(format!(
                        "superseded_by target `{target_id}` does not exist as a Memory entry"
                    )));
                }
            }
        }

        let scope_wire = rev.scope.as_wire();
        let concern_wire = rev.concern.as_wire();
        let slug = rev.slug.clone();

        let existing_key_id = {
            let by_key = tx.open_table(STATE_KEYS).map_err(backend)?;
            by_key
                .get((scope_wire.as_str(), concern_wire, slug.as_str()))
                .map_err(backend)?
                .map(|g| g.value().to_owned())
        };
        if let Some(existing_id_str) = existing_key_id {
            let existing_id = StateId(Iri::parse(&existing_id_str).map_err(decode)?);
            if existing_id != rev.entry_id {
                self.alias_unpublished_id_in(
                    tx,
                    Kind::State,
                    existing_id.iri(),
                    rev.entry_id.iri(),
                )?;
                {
                    let mut entries = tx.open_table(STATE_ENTRIES).map_err(backend)?;
                    let removed = entries
                        .remove(existing_id.as_str())
                        .map_err(backend)?
                        .map(|g| g.value().to_owned());
                    if let Some(json) = removed {
                        let mut local_entry: StateEntry =
                            serde_json::from_str(&json).map_err(decode)?;
                        local_entry.id = rev.entry_id.clone();
                        let updated_json = serde_json::to_string(&local_entry).map_err(backend)?;
                        entries
                            .insert(rev.entry_id.as_str(), updated_json.as_str())
                            .map_err(backend)?;
                    }
                }
                {
                    let mut conflicts = tx.open_table(STATE_CONFLICTS).map_err(backend)?;
                    let removed = conflicts
                        .remove(existing_id.as_str())
                        .map_err(backend)?
                        .map(|g| g.value().to_owned());
                    if let Some(json) = removed {
                        let mut c: StateConflict = serde_json::from_str(&json).map_err(decode)?;
                        c.entry_id = rev.entry_id.clone();
                        let updated_json = serde_json::to_string(&c).map_err(backend)?;
                        conflicts
                            .insert(rev.entry_id.as_str(), updated_json.as_str())
                            .map_err(backend)?;
                    }
                }
                {
                    let mut revs = tx.open_table(STATE_REVISIONS).map_err(backend)?;
                    let mut updates = Vec::new();
                    for row in revs.iter().map_err(backend)? {
                        let (seq, raw) = row.map_err(backend)?;
                        let mut stored: StateRevision =
                            serde_json::from_str(raw.value()).map_err(decode)?;
                        if stored.entry_id == existing_id {
                            stored.entry_id = rev.entry_id.clone();
                            updates.push((
                                seq.value(),
                                serde_json::to_string(&stored).map_err(backend)?,
                            ));
                        }
                    }
                    for (seq, json) in updates {
                        revs.insert(seq, json.as_str()).map_err(backend)?;
                    }
                }
                {
                    let mut by_key = tx.open_table(STATE_KEYS).map_err(backend)?;
                    by_key
                        .insert(
                            (scope_wire.as_str(), concern_wire, slug.as_str()),
                            rev.entry_id.as_str(),
                        )
                        .map_err(backend)?;
                }
            }
        }

        let resolved_entry_id = StateId(self.resolve_in_wtx(tx, rev.entry_id.iri())?);
        let mut rev = rev;
        rev.entry_id = resolved_entry_id.clone();
        self.index_imported_in(tx, Kind::State, rev.entry_id.iri())?;

        let existing_entry: Option<StateEntry> = {
            let entries = tx.open_table(STATE_ENTRIES).map_err(backend)?;
            let raw = entries
                .get(resolved_entry_id.as_str())
                .map_err(backend)?
                .map(|g| g.value().to_owned());
            match raw {
                Some(json) => Some(serde_json::from_str(&json).map_err(decode)?),
                None => None,
            }
        };

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

        self.append_state_revision_in(tx, &rev)?;

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
            let entry_json = serde_json::to_string(&entry).map_err(backend)?;
            tx.open_table(STATE_ENTRIES)
                .map_err(backend)?
                .insert(resolved_entry_id.as_str(), entry_json.as_str())
                .map_err(backend)?;
            tx.open_table(STATE_KEYS)
                .map_err(backend)?
                .insert(
                    (scope_wire.as_str(), concern_wire, slug.as_str()),
                    resolved_entry_id.as_str(),
                )
                .map_err(backend)?;
            return Ok(());
        };

        let dag = self.load_revisions_for_entry_in(tx, &resolved_entry_id)?;
        let r_local = existing.head_revision.clone();
        let r_remote = rev.revision_id.clone();

        if is_ancestor_in_dag(&r_local, &r_remote, &dag) {
            let updated = StateEntry {
                id: resolved_entry_id.clone(),
                scope: existing.scope,
                concern: existing.concern,
                slug: existing.slug,
                sensitive: rev.sensitive,
                ever_sensitive: existing.ever_sensitive || rev.ever_sensitive || rev.sensitive,
                payload: rev.payload,
                head_revision: r_remote.clone(),
                updated_at: rev.at,
                by: rev.by,
            };
            let updated_json = serde_json::to_string(&updated).map_err(backend)?;
            tx.open_table(STATE_ENTRIES)
                .map_err(backend)?
                .insert(resolved_entry_id.as_str(), updated_json.as_str())
                .map_err(backend)?;

            let mut conflicts = tx.open_table(STATE_CONFLICTS).map_err(backend)?;
            let raw_conflict = conflicts
                .get(resolved_entry_id.as_str())
                .map_err(backend)?
                .map(|g| g.value().to_owned());
            if let Some(c_json) = raw_conflict {
                let c: StateConflict = serde_json::from_str(&c_json).map_err(decode)?;
                if is_ancestor_in_dag(&c.local_head, &r_remote, &dag)
                    && is_ancestor_in_dag(&c.remote_head, &r_remote, &dag)
                {
                    conflicts
                        .remove(resolved_entry_id.as_str())
                        .map_err(backend)?;
                }
            }
            return Ok(());
        }

        if is_ancestor_in_dag(&r_remote, &r_local, &dag) {
            return Ok(());
        }

        let local_rev = dag.get(&r_local).cloned().ok_or_else(|| {
            StoreError::InvalidState(format!("missing local head revision `{r_local}`"))
        })?;
        let base_rev = lowest_common_ancestor_in_dag(&r_local, &r_remote, &dag)
            .and_then(|id| dag.get(&id).cloned());

        let auto_merged: Option<StatePayload> = match (&local_rev.payload, &rev.payload) {
            (StatePayload::Runtime(l_rt), StatePayload::Runtime(r_rt))
                if l_rt.merge_mode == r_rt.merge_mode =>
            {
                match l_rt.merge_mode {
                    MergeMode::Conflict => None,
                    MergeMode::Append => {
                        let base_arr = match base_rev.as_ref().map(|b| &b.payload) {
                            Some(StatePayload::Runtime(b_rt)) => b_rt.value.as_array().cloned(),
                            _ => Some(Vec::new()),
                        };
                        match (base_arr, l_rt.value.as_array(), r_rt.value.as_array()) {
                            (Some(b), Some(l), Some(r)) => {
                                Some(StatePayload::Runtime(RuntimeEntry {
                                    value: serde_json::Value::Array(compute_append_merge(b, l, r)),
                                    merge_mode: MergeMode::Append,
                                }))
                            }
                            _ => None,
                        }
                    }
                    MergeMode::LastWriter => {
                        let win_payload =
                            compute_last_writer_merge(&local_rev, &rev).payload.clone();
                        Some(win_payload)
                    }
                }
            }
            _ => None,
        };

        if let Some(merged_payload) = auto_merged {
            let merge_rev_id = RevisionId(mint()?);
            let sensitive = local_rev.sensitive || rev.sensitive;
            let ever_sensitive = existing.ever_sensitive
                || local_rev.ever_sensitive
                || rev.ever_sensitive
                || sensitive;
            let merge_rev = StateRevision {
                revision_id: merge_rev_id.clone(),
                entry_id: resolved_entry_id.clone(),
                parent_revisions: sorted_parents(r_local, r_remote),
                scope: existing.scope.clone(),
                concern: existing.concern,
                slug: existing.slug.clone(),
                sensitive,
                ever_sensitive,
                payload: merged_payload.clone(),
                at: merge_at.clone(),
                by: "fl".to_string(),
            };
            self.append_state_revision_in(tx, &merge_rev)?;
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
            let updated_json = serde_json::to_string(&updated).map_err(backend)?;
            tx.open_table(STATE_ENTRIES)
                .map_err(backend)?
                .insert(resolved_entry_id.as_str(), updated_json.as_str())
                .map_err(backend)?;
            return Ok(());
        }

        let conflict = StateConflict {
            entry_id: resolved_entry_id.clone(),
            local_head: r_local,
            remote_head: r_remote,
            detected_at: merge_at,
        };
        let conflict_json = serde_json::to_string(&conflict).map_err(backend)?;
        tx.open_table(STATE_CONFLICTS)
            .map_err(backend)?
            .insert(resolved_entry_id.as_str(), conflict_json.as_str())
            .map_err(backend)?;
        Ok(())
    }
}

impl AgentRegistry for RedbStore {
    fn upsert_area(&self, area: AreaRecord) -> Result<AreaRecord, StoreError> {
        let normalized = area
            .normalized()
            .map_err(|e| StoreError::InvalidState(e.to_string()))?;
        let tx = self.db.begin_write().map_err(backend)?;
        raise_format(&tx, FORMAT_WITH_STATE)?;
        let ev = RegistryEvent {
            event_id: mint()?,
            at: normalized.updated_at.clone(),
            by: normalized.by.clone(),
            kind: RegistryEventKind::AreaUpsert(normalized.clone()),
        };
        let json = serde_json::to_string(&normalized).map_err(backend)?;
        tx.open_table(AREAS)
            .map_err(backend)?
            .insert(normalized.area_key.as_str(), json.as_str())
            .map_err(backend)?;
        self.append_registry_event_in(&tx, &ev)?;
        tx.commit().map_err(backend)?;
        Ok(normalized)
    }

    fn get_area(&self, area_key: &str) -> Result<Option<AreaRecord>, StoreError> {
        let key = validate_slug(area_key).map_err(|e| StoreError::InvalidState(e.to_string()))?;
        let tx = self.db.begin_read().map_err(backend)?;
        let areas = match tx.open_table(AREAS) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(e) => return Err(backend(e)),
        };
        let Some(raw) = areas.get(key.as_str()).map_err(backend)? else {
            return Ok(None);
        };
        Ok(Some(serde_json::from_str(raw.value()).map_err(decode)?))
    }

    fn list_areas(&self) -> Result<Vec<AreaRecord>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let areas = match tx.open_table(AREAS) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(backend(e)),
        };
        let mut out: Vec<AreaRecord> = Vec::new();
        for row in areas.iter().map_err(backend)? {
            let (_k, v) = row.map_err(backend)?;
            out.push(serde_json::from_str(v.value()).map_err(decode)?);
        }
        out.sort_by(|a, b| a.area_key.cmp(&b.area_key));
        Ok(out)
    }

    fn register_agent(&self, args: RegisterAgentArgs) -> Result<AgentRecord, StoreError> {
        let (name_key, display_name, email) =
            validate_agent_identity(&args.display_name, args.email.as_deref())
                .map_err(StoreError::InvalidState)?;
        let project =
            validate_slug(&args.project).map_err(|e| StoreError::InvalidState(e.to_string()))?;
        let mut extra_areas = Vec::with_capacity(args.extra_areas.len());
        for a in &args.extra_areas {
            extra_areas
                .push(validate_slug(a).map_err(|e| StoreError::InvalidState(e.to_string()))?);
        }
        extra_areas.sort();
        extra_areas.dedup();

        let tx = self.db.begin_write().map_err(backend)?;
        raise_format(&tx, FORMAT_WITH_STATE)?;

        let mut pending_area_events = Vec::new();
        {
            let mut areas = tx.open_table(AREAS).map_err(backend)?;
            for area_key in std::iter::once(&project).chain(extra_areas.iter()) {
                let exists = areas.get(area_key.as_str()).map_err(backend)?.is_some();
                if !exists {
                    if args.create_area {
                        let area = AreaRecord {
                            area_key: area_key.clone(),
                            name: area_key.clone(),
                            path: None,
                            purpose: format!("Registered area {area_key}"),
                            updated_at: args.at.clone(),
                            by: args.by.clone(),
                        };
                        let json = serde_json::to_string(&area).map_err(backend)?;
                        areas
                            .insert(area_key.as_str(), json.as_str())
                            .map_err(backend)?;
                        pending_area_events.push(RegistryEvent {
                            event_id: mint()?,
                            at: args.at.clone(),
                            by: args.by.clone(),
                            kind: RegistryEventKind::AreaUpsert(area),
                        });
                    } else {
                        return Err(StoreError::NoSuchArea(area_key.clone()));
                    }
                }
            }
        }
        for ev in pending_area_events {
            self.append_registry_event_in(&tx, &ev)?;
        }

        let existing_by_name: Option<AgentRecord> = {
            let by_name = tx.open_table(AGENT_BY_NAME).map_err(backend)?;
            let agents = tx.open_table(AGENTS).map_err(backend)?;
            let id_opt = by_name
                .get(name_key.as_str())
                .map_err(backend)?
                .map(|g| g.value().to_owned());
            match id_opt {
                Some(id_str) => {
                    let raw = agents
                        .get(id_str.as_str())
                        .map_err(backend)?
                        .map(|g| g.value().to_owned());
                    match raw {
                        Some(json) => Some(serde_json::from_str(&json).map_err(decode)?),
                        None => None,
                    }
                }
                None => None,
            }
        };
        if let Some(ref existing) = existing_by_name
            && !args.replace
        {
            return Err(StoreError::AgentCollision {
                name_key,
                existing_id: existing.id.clone(),
                status: existing.status,
            });
        }

        if args.role == AgentRole::Lead {
            let conflicting_lead: Option<AgentRecord> = {
                let agents = tx.open_table(AGENTS).map_err(backend)?;
                let mut found = None;
                for row in agents.iter().map_err(backend)? {
                    let (_k, v) = row.map_err(backend)?;
                    let a: AgentRecord = serde_json::from_str(v.value()).map_err(decode)?;
                    if a.status == AgentStatus::Active
                        && a.project == project
                        && a.role == AgentRole::Lead
                        && a.name_key != name_key
                    {
                        found = Some(a);
                        break;
                    }
                }
                found
            };
            if let Some(lead) = conflicting_lead {
                if !args.replace {
                    return Err(StoreError::LeadTaken {
                        project,
                        lead_name: lead.display_name,
                        lead_id: lead.id,
                    });
                }
                let mut retired = lead.clone();
                retired.status = AgentStatus::Retired;
                retired.updated_at = args.at.clone();
                retired.by = args.by.clone();
                let json = serde_json::to_string(&retired).map_err(backend)?;
                tx.open_table(AGENTS)
                    .map_err(backend)?
                    .insert(retired.id.as_str(), json.as_str())
                    .map_err(backend)?;
                let retire_ev = RegistryEvent {
                    event_id: mint()?,
                    at: args.at.clone(),
                    by: args.by.clone(),
                    kind: RegistryEventKind::AgentRetire { id: retired.id },
                };
                self.append_registry_event_in(&tx, &retire_ev)?;
            }
        }

        let agent_id = match existing_by_name {
            Some(existing) => existing.id,
            None => {
                let iri = mint()?;
                index_new(&tx, &iri, Kind::Agent)?;
                AgentId(iri)
            }
        };

        let record = AgentRecord {
            id: agent_id.clone(),
            name_key: name_key.clone(),
            display_name,
            email,
            project,
            role: args.role,
            extra_areas,
            status: AgentStatus::Active,
            updated_at: args.at.clone(),
            by: args.by.clone(),
        };
        let json = serde_json::to_string(&record).map_err(backend)?;
        tx.open_table(AGENT_BY_NAME)
            .map_err(backend)?
            .insert(name_key.as_str(), agent_id.as_str())
            .map_err(backend)?;
        tx.open_table(AGENTS)
            .map_err(backend)?
            .insert(agent_id.as_str(), json.as_str())
            .map_err(backend)?;
        let ev = RegistryEvent {
            event_id: mint()?,
            at: args.at,
            by: args.by,
            kind: RegistryEventKind::AgentUpsert(record.clone()),
        };
        self.append_registry_event_in(&tx, &ev)?;
        tx.commit().map_err(backend)?;
        Ok(record)
    }

    fn retire_agent(&self, id: &AgentId, at: At, by: &str) -> Result<AgentRecord, StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        raise_format(&tx, FORMAT_WITH_STATE)?;
        let (resolved_iri, kind) = self.locate_for_write(&tx, id.iri())?;
        if kind != Kind::Agent {
            return Err(StoreError::WrongKind {
                id: id.iri().clone(),
                expected: Kind::Agent,
                found: kind,
            });
        }
        let resolved = AgentId(resolved_iri);
        let mut rec: AgentRecord = {
            let agents = tx.open_table(AGENTS).map_err(backend)?;
            let Some(raw) = agents.get(resolved.as_str()).map_err(backend)? else {
                return Err(StoreError::NoSuchAgent(resolved.to_string()));
            };
            serde_json::from_str(raw.value()).map_err(decode)?
        };
        rec.status = AgentStatus::Retired;
        rec.updated_at = at.clone();
        rec.by = by.trim().to_string();
        let json = serde_json::to_string(&rec).map_err(backend)?;
        tx.open_table(AGENTS)
            .map_err(backend)?
            .insert(resolved.as_str(), json.as_str())
            .map_err(backend)?;
        let ev = RegistryEvent {
            event_id: mint()?,
            at,
            by: rec.by.clone(),
            kind: RegistryEventKind::AgentRetire { id: resolved },
        };
        self.append_registry_event_in(&tx, &ev)?;
        tx.commit().map_err(backend)?;
        Ok(rec)
    }

    fn get_agent(&self, id: &AgentId) -> Result<Option<AgentRecord>, StoreError> {
        let (target, kind) = self.locate(id.iri())?;
        if kind != Kind::Agent {
            return Ok(None);
        }
        let tx = self.db.begin_read().map_err(backend)?;
        additive(&tx, AGENTS, &target)
    }

    fn agent_by_name(&self, name: &str) -> Result<Option<AgentRecord>, StoreError> {
        let key = validate_slug(name).map_err(|e| StoreError::InvalidState(e.to_string()))?;
        let tx = self.db.begin_read().map_err(backend)?;
        let by_name = match tx.open_table(AGENT_BY_NAME) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(e) => return Err(backend(e)),
        };
        let Some(id_guard) = by_name.get(key.as_str()).map_err(backend)? else {
            return Ok(None);
        };
        let id_iri = Iri::parse(id_guard.value()).map_err(decode)?;
        drop(id_guard);
        drop(by_name);
        drop(tx);
        let (target, kind) = self.locate(&id_iri)?;
        if kind != Kind::Agent {
            return Ok(None);
        }
        let tx = self.db.begin_read().map_err(backend)?;
        additive(&tx, AGENTS, &target)
    }

    fn list_agents(
        &self,
        project: Option<&str>,
        include_retired: bool,
    ) -> Result<Vec<AgentRecord>, StoreError> {
        let project_filter = match project {
            Some(p) => Some(validate_slug(p).map_err(|e| StoreError::InvalidState(e.to_string()))?),
            None => None,
        };
        let tx = self.db.begin_read().map_err(backend)?;
        let agents = match tx.open_table(AGENTS) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(backend(e)),
        };
        let mut out = Vec::new();
        for row in agents.iter().map_err(backend)? {
            let (_k, v) = row.map_err(backend)?;
            let a: AgentRecord = serde_json::from_str(v.value()).map_err(decode)?;
            if !include_retired && a.status != AgentStatus::Active {
                continue;
            }
            if let Some(ref p) = project_filter
                && &a.project != p
                && !a.extra_areas.contains(p)
            {
                continue;
            }
            out.push(a);
        }
        out.sort_by(|a, b| a.name_key.cmp(&b.name_key));
        Ok(out)
    }

    fn registry_events_after(&self, seq: u64) -> Result<Vec<(u64, RegistryEvent)>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let events = match tx.open_table(AGENT_EVENTS) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(backend(e)),
        };
        let mut out = Vec::new();
        for row in events
            .range((std::ops::Bound::Excluded(seq), std::ops::Bound::Unbounded))
            .map_err(backend)?
        {
            let (k, v) = row.map_err(backend)?;
            let ev: RegistryEvent = serde_json::from_str(v.value()).map_err(decode)?;
            out.push((k.value(), ev));
        }
        Ok(out)
    }

    fn import_registry_events(&self, events: &[RegistryEvent]) -> Result<(), StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        raise_format(&tx, FORMAT_WITH_STATE)?;
        for ev in events {
            let existing_seq = {
                let by_id = tx.open_table(AGENT_EVENT_IDS).map_err(backend)?;
                by_id
                    .get(ev.event_id.as_str())
                    .map_err(backend)?
                    .map(|g| g.value())
            };
            if let Some(seq) = existing_seq {
                let tbl = tx.open_table(AGENT_EVENTS).map_err(backend)?;
                let raw = tbl
                    .get(seq)
                    .map_err(backend)?
                    .ok_or_else(|| decode("agent_event_ids points to missing sequence"))?;
                let existing: RegistryEvent = serde_json::from_str(raw.value()).map_err(decode)?;
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
                        .map_err(|e| StoreError::InvalidState(e.to_string()))?;
                    let json = serde_json::to_string(&norm).map_err(backend)?;
                    tx.open_table(AREAS)
                        .map_err(backend)?
                        .insert(norm.area_key.as_str(), json.as_str())
                        .map_err(backend)?;
                }
                RegistryEventKind::AgentUpsert(rec) => {
                    let existing_id_opt = {
                        let by_name = tx.open_table(AGENT_BY_NAME).map_err(backend)?;
                        by_name
                            .get(rec.name_key.as_str())
                            .map_err(backend)?
                            .map(|g| g.value().to_owned())
                    };
                    if let Some(existing_id_str) = existing_id_opt {
                        let existing_id = AgentId(Iri::parse(&existing_id_str).map_err(decode)?);
                        if existing_id != rec.id {
                            self.alias_unpublished_id_in(
                                &tx,
                                Kind::Agent,
                                existing_id.iri(),
                                rec.id.iri(),
                            )?;
                            tx.open_table(AGENTS)
                                .map_err(backend)?
                                .remove(existing_id.as_str())
                                .map_err(backend)?;
                        }
                    }
                    if rec.role == AgentRole::Lead && rec.status == AgentStatus::Active {
                        let mut agents = tx.open_table(AGENTS).map_err(backend)?;
                        let mut to_retire = Vec::new();
                        for row in agents.iter().map_err(backend)? {
                            let (_k, v) = row.map_err(backend)?;
                            let mut other: AgentRecord =
                                serde_json::from_str(v.value()).map_err(decode)?;
                            if other.project == rec.project
                                && other.role == AgentRole::Lead
                                && other.status == AgentStatus::Active
                                && other.id != rec.id
                            {
                                other.status = AgentStatus::Retired;
                                to_retire.push((
                                    other.id.as_str().to_owned(),
                                    serde_json::to_string(&other).map_err(backend)?,
                                ));
                            }
                        }
                        for (id_str, json) in to_retire {
                            agents
                                .insert(id_str.as_str(), json.as_str())
                                .map_err(backend)?;
                        }
                    }
                    self.index_imported_in(&tx, Kind::Agent, rec.id.iri())?;
                    let json = serde_json::to_string(rec).map_err(backend)?;
                    tx.open_table(AGENT_BY_NAME)
                        .map_err(backend)?
                        .insert(rec.name_key.as_str(), rec.id.as_str())
                        .map_err(backend)?;
                    tx.open_table(AGENTS)
                        .map_err(backend)?
                        .insert(rec.id.as_str(), json.as_str())
                        .map_err(backend)?;
                }
                RegistryEventKind::AgentRetire { id } => {
                    let resolved = AgentId(self.resolve_in_wtx(&tx, id.iri())?);
                    let mut agents = tx.open_table(AGENTS).map_err(backend)?;
                    let raw = agents
                        .get(resolved.as_str())
                        .map_err(backend)?
                        .map(|g| g.value().to_owned());
                    if let Some(json) = raw {
                        let mut rec: AgentRecord = serde_json::from_str(&json).map_err(decode)?;
                        rec.status = AgentStatus::Retired;
                        rec.updated_at = ev.at.clone();
                        rec.by = ev.by.clone();
                        let updated_json = serde_json::to_string(&rec).map_err(backend)?;
                        agents
                            .insert(resolved.as_str(), updated_json.as_str())
                            .map_err(backend)?;
                    }
                }
            }
            self.append_registry_event_in(&tx, ev)?;
        }
        tx.commit().map_err(backend)?;
        Ok(())
    }
}

impl StateStore for RedbStore {
    fn upsert_state(&self, args: UpsertStateArgs) -> Result<StateEntry, StoreError> {
        let slug =
            validate_slug(&args.slug).map_err(|e| StoreError::InvalidState(e.to_string()))?;
        let concern = args.payload.concern();
        let scope_wire = args.scope.as_wire();
        let concern_wire = concern.as_wire();

        let tx = self.db.begin_write().map_err(backend)?;
        raise_format(&tx, FORMAT_WITH_STATE)?;

        let existing: Option<StateEntry> = {
            let by_key = tx.open_table(STATE_KEYS).map_err(backend)?;
            let entries = tx.open_table(STATE_ENTRIES).map_err(backend)?;
            let id_opt = by_key
                .get((scope_wire.as_str(), concern_wire, slug.as_str()))
                .map_err(backend)?
                .map(|g| g.value().to_owned());
            match id_opt {
                Some(id_str) => {
                    let raw = entries
                        .get(id_str.as_str())
                        .map_err(backend)?
                        .map(|g| g.value().to_owned());
                    match raw {
                        Some(json) => Some(serde_json::from_str(&json).map_err(decode)?),
                        None => None,
                    }
                }
                None => None,
            }
        };

        if let Some(ref ex) = existing {
            let conflicts = tx.open_table(STATE_CONFLICTS).map_err(backend)?;
            if let Some(raw_c) = conflicts.get(ex.id.as_str()).map_err(backend)? {
                let c: StateConflict = serde_json::from_str(raw_c.value()).map_err(decode)?;
                return Err(StoreError::StateConflict {
                    entry_id: ex.id.clone(),
                    local_head: c.local_head,
                    remote_head: c.remote_head,
                });
            }
        }

        let entry_id = match existing {
            Some(ref ex) => ex.id.clone(),
            None => {
                let iri = mint()?;
                index_new(&tx, &iri, Kind::State)?;
                StateId(iri)
            }
        };

        args.payload
            .validate(&entry_id)
            .map_err(|e| StoreError::InvalidState(e.to_string()))?;

        if let StatePayload::Memory(ref m) = args.payload
            && let Some(ref target_id) = m.superseded_by
        {
            let resolved_target = self.resolve_in_wtx(&tx, target_id.iri())?;
            let entries = tx.open_table(STATE_ENTRIES).map_err(backend)?;
            let raw = entries
                .get(resolved_target.as_str())
                .map_err(backend)?
                .map(|g| g.value().to_owned());
            match raw {
                Some(json) => {
                    let target: StateEntry = serde_json::from_str(&json).map_err(decode)?;
                    if target.concern != StateConcern::Memory {
                        return Err(StoreError::InvalidState(format!(
                            "superseded_by target `{target_id}` does not exist as a Memory entry"
                        )));
                    }
                }
                None => {
                    return Err(StoreError::InvalidState(format!(
                        "superseded_by target `{target_id}` does not exist as a Memory entry"
                    )));
                }
            }
        }

        let (sensitive, ever_sensitive, parent_revisions) = match existing {
            Some(ref ex) => {
                let s = match args.sensitivity {
                    SensitivityPatch::Inherit => ex.sensitive,
                    SensitivityPatch::SetSensitive => true,
                    SensitivityPatch::ClearSensitive => false,
                };
                let ever = ex.ever_sensitive || s;
                (s, ever, vec![ex.head_revision.clone()])
            }
            None => {
                let s = matches!(args.sensitivity, SensitivityPatch::SetSensitive);
                (s, s, Vec::new())
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
            scope: args.scope,
            concern,
            slug: slug.clone(),
            sensitive,
            ever_sensitive,
            payload: args.payload,
            head_revision: rev_id,
            updated_at: args.at,
            by: args.by.trim().to_string(),
        };

        self.append_state_revision_in(&tx, &rev)?;
        let entry_json = serde_json::to_string(&entry).map_err(backend)?;
        tx.open_table(STATE_ENTRIES)
            .map_err(backend)?
            .insert(entry_id.as_str(), entry_json.as_str())
            .map_err(backend)?;
        tx.open_table(STATE_KEYS)
            .map_err(backend)?
            .insert(
                (scope_wire.as_str(), concern_wire, slug.as_str()),
                entry_id.as_str(),
            )
            .map_err(backend)?;
        tx.commit().map_err(backend)?;
        Ok(entry)
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
        let slug = validate_slug(slug).map_err(|e| StoreError::InvalidState(e.to_string()))?;
        if matches!(lifecycle, MemoryLifecycle::Active) {
            return Err(StoreError::InvalidState(
                "retire_memory requires `retired` or `superseded` lifecycle".into(),
            ));
        }
        let current = self
            .get_state_by_key(scope, StateConcern::Memory, &slug)?
            .ok_or_else(|| StoreError::NoSuchState(format!("{}:memory:{slug}", scope.as_wire())))?;
        let StatePayload::Memory(mut mem) = current.payload else {
            return Err(StoreError::InvalidState(format!(
                "state entry `{}` is not a Memory entry",
                current.id
            )));
        };
        mem.lifecycle = lifecycle;
        mem.superseded_by = superseded_by;
        self.upsert_state(UpsertStateArgs {
            scope: scope.clone(),
            slug,
            sensitivity: SensitivityPatch::Inherit,
            payload: StatePayload::Memory(mem),
            at,
            by: by.to_string(),
        })
    }

    fn get_state(&self, id: &StateId) -> Result<Option<StateEntry>, StoreError> {
        let (target, kind) = self.locate(id.iri())?;
        if kind != Kind::State {
            return Ok(None);
        }
        let tx = self.db.begin_read().map_err(backend)?;
        additive(&tx, STATE_ENTRIES, &target)
    }

    fn get_state_by_key(
        &self,
        scope: &StateScope,
        concern: StateConcern,
        slug: &str,
    ) -> Result<Option<StateEntry>, StoreError> {
        let slug = validate_slug(slug).map_err(|e| StoreError::InvalidState(e.to_string()))?;
        let scope_wire = scope.as_wire();
        let concern_wire = concern.as_wire();
        let tx = self.db.begin_read().map_err(backend)?;
        let by_key = match tx.open_table(STATE_KEYS) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(e) => return Err(backend(e)),
        };
        let Some(id_guard) = by_key
            .get((scope_wire.as_str(), concern_wire, slug.as_str()))
            .map_err(backend)?
        else {
            return Ok(None);
        };
        let id_iri = Iri::parse(id_guard.value()).map_err(decode)?;
        drop(id_guard);
        drop(by_key);
        drop(tx);
        let (target, kind) = self.locate(&id_iri)?;
        if kind != Kind::State {
            return Ok(None);
        }
        let tx = self.db.begin_read().map_err(backend)?;
        additive(&tx, STATE_ENTRIES, &target)
    }

    fn list_state(
        &self,
        scope: Option<&StateScope>,
        concern: Option<StateConcern>,
    ) -> Result<Vec<StateEntry>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let entries = match tx.open_table(STATE_ENTRIES) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(backend(e)),
        };
        let mut out = Vec::new();
        for row in entries.iter().map_err(backend)? {
            let (_k, v) = row.map_err(backend)?;
            let entry: StateEntry = serde_json::from_str(v.value()).map_err(decode)?;
            if scope.is_some_and(|s| &entry.scope != s) {
                continue;
            }
            if concern.is_some_and(|c| entry.concern != c) {
                continue;
            }
            out.push(entry);
        }
        out.sort_by(|a, b| (&a.scope, a.concern, &a.slug).cmp(&(&b.scope, b.concern, &b.slug)));
        Ok(out)
    }

    fn state_revisions(&self, id: &StateId) -> Result<Vec<StateRevision>, StoreError> {
        let (primary, kind) = self.locate(id.iri())?;
        if kind != Kind::State {
            return Err(StoreError::WrongKind {
                id: id.iri().clone(),
                expected: Kind::State,
                found: kind,
            });
        }
        let resolved = StateId(primary);
        let tx = self.db.begin_read().map_err(backend)?;
        let revs = match tx.open_table(STATE_REVISIONS) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(backend(e)),
        };
        let mut out = Vec::new();
        for row in revs.iter().map_err(backend)? {
            let (_seq, raw) = row.map_err(backend)?;
            let rev: StateRevision = serde_json::from_str(raw.value()).map_err(decode)?;
            if rev.entry_id == resolved {
                out.push(rev);
            }
        }
        Ok(out)
    }

    fn list_conflicts(&self) -> Result<Vec<StateConflict>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let conflicts = match tx.open_table(STATE_CONFLICTS) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(backend(e)),
        };
        let mut out = Vec::new();
        for row in conflicts.iter().map_err(backend)? {
            let (_k, v) = row.map_err(backend)?;
            out.push(serde_json::from_str(v.value()).map_err(decode)?);
        }
        Ok(out)
    }

    fn resolve_conflict(
        &self,
        id: &StateId,
        keep: &RevisionId,
        at: At,
        by: &str,
    ) -> Result<StateEntry, StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        raise_format(&tx, FORMAT_WITH_STATE)?;
        let (resolved_iri, kind) = self.locate_for_write(&tx, id.iri())?;
        if kind != Kind::State {
            return Err(StoreError::WrongKind {
                id: id.iri().clone(),
                expected: Kind::State,
                found: kind,
            });
        }
        let resolved = StateId(resolved_iri);

        let conflict: StateConflict = {
            let conflicts = tx.open_table(STATE_CONFLICTS).map_err(backend)?;
            let Some(raw) = conflicts.get(resolved.as_str()).map_err(backend)? else {
                return Err(StoreError::InvalidState(format!(
                    "state entry `{resolved}` has no active conflict"
                )));
            };
            serde_json::from_str(raw.value()).map_err(decode)?
        };

        if keep != &conflict.local_head && keep != &conflict.remote_head {
            return Err(StoreError::InvalidState(format!(
                "`--keep {keep}` must match either local head `{}` or remote head `{}`",
                conflict.local_head, conflict.remote_head
            )));
        }

        let (chosen_rev, local_rev, remote_rev): (StateRevision, StateRevision, StateRevision) = {
            let by_id = tx.open_table(STATE_REVISION_IDS).map_err(backend)?;
            let revs = tx.open_table(STATE_REVISIONS).map_err(backend)?;
            let load_rev = |rev_id: &RevisionId| -> Result<StateRevision, StoreError> {
                let Some(seq_guard) = by_id.get(rev_id.as_str()).map_err(backend)? else {
                    return Err(StoreError::InvalidState(format!(
                        "unknown revision `{rev_id}`"
                    )));
                };
                let seq = seq_guard.value();
                let raw = revs
                    .get(seq)
                    .map_err(backend)?
                    .ok_or_else(|| decode("state_revision_ids points to missing sequence"))?;
                serde_json::from_str(raw.value()).map_err(decode)
            };
            (
                load_rev(keep)?,
                load_rev(&conflict.local_head)?,
                load_rev(&conflict.remote_head)?,
            )
        };

        let existing: StateEntry = {
            let entries = tx.open_table(STATE_ENTRIES).map_err(backend)?;
            let raw = entries
                .get(resolved.as_str())
                .map_err(backend)?
                .ok_or_else(|| StoreError::NoSuchState(resolved.to_string()))?;
            serde_json::from_str(raw.value()).map_err(decode)?
        };

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
            payload: chosen_rev.payload.clone(),
            at: at.clone(),
            by: by.trim().to_string(),
        };
        self.append_state_revision_in(&tx, &merge_rev)?;

        let updated = StateEntry {
            id: resolved.clone(),
            scope: existing.scope,
            concern: existing.concern,
            slug: existing.slug,
            sensitive,
            ever_sensitive,
            payload: chosen_rev.payload,
            head_revision: merge_rev_id,
            updated_at: at,
            by: by.trim().to_string(),
        };
        let updated_json = serde_json::to_string(&updated).map_err(backend)?;
        tx.open_table(STATE_CONFLICTS)
            .map_err(backend)?
            .remove(resolved.as_str())
            .map_err(backend)?;
        tx.open_table(STATE_ENTRIES)
            .map_err(backend)?
            .insert(resolved.as_str(), updated_json.as_str())
            .map_err(backend)?;
        tx.commit().map_err(backend)?;
        Ok(updated)
    }

    fn state_revisions_after(&self, seq: u64) -> Result<Vec<(u64, StateRevision)>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let revs = match tx.open_table(STATE_REVISIONS) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(backend(e)),
        };
        let mut out = Vec::new();
        for row in revs
            .range((std::ops::Bound::Excluded(seq), std::ops::Bound::Unbounded))
            .map_err(backend)?
        {
            let (k, v) = row.map_err(backend)?;
            let rev: StateRevision = serde_json::from_str(v.value()).map_err(decode)?;
            out.push((k.value(), rev));
        }
        Ok(out)
    }

    fn import_state_revisions(
        &self,
        revs: &[StateRevision],
        merge_at: At,
    ) -> Result<(), StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        raise_format(&tx, FORMAT_WITH_STATE)?;
        for rev in revs {
            let existing_seq = {
                let by_id = tx.open_table(STATE_REVISION_IDS).map_err(backend)?;
                by_id
                    .get(rev.revision_id.as_str())
                    .map_err(backend)?
                    .map(|g| g.value())
            };
            if let Some(seq) = existing_seq {
                let tbl = tx.open_table(STATE_REVISIONS).map_err(backend)?;
                let raw = tbl
                    .get(seq)
                    .map_err(backend)?
                    .ok_or_else(|| decode("state_revision_ids points to missing sequence"))?;
                let existing: StateRevision = serde_json::from_str(raw.value()).map_err(decode)?;
                let mut normalized_rev = rev.clone();
                normalized_rev.entry_id = StateId(self.resolve_in_wtx(&tx, rev.entry_id.iri())?);
                if existing != normalized_rev {
                    return Err(StoreError::Tampered {
                        id: rev.revision_id.0.clone(),
                        detail: "state revision payload differs from stored copy".into(),
                    });
                }
                continue;
            }
            self.apply_imported_state_revision_in(&tx, rev.clone(), merge_at.clone())?;
        }
        tx.commit().map_err(backend)?;
        Ok(())
    }

    fn sync_cursor(&self, remote: &str) -> Result<Option<StateSyncCursor>, StoreError> {
        let tx = self.db.begin_read().map_err(backend)?;
        let sync = match tx.open_table(STATE_SYNC_CURSORS) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(e) => return Err(backend(e)),
        };
        let Some(raw) = sync.get(remote).map_err(backend)? else {
            return Ok(None);
        };
        Ok(Some(serde_json::from_str(raw.value()).map_err(decode)?))
    }

    fn set_sync_cursor(&self, cursor: StateSyncCursor) -> Result<(), StoreError> {
        let tx = self.db.begin_write().map_err(backend)?;
        raise_format(&tx, FORMAT_WITH_STATE)?;
        let json = serde_json::to_string(&cursor).map_err(backend)?;
        tx.open_table(STATE_SYNC_CURSORS)
            .map_err(backend)?
            .insert(cursor.remote.as_str(), json.as_str())
            .map_err(backend)?;
        tx.commit().map_err(backend)?;
        Ok(())
    }
}
