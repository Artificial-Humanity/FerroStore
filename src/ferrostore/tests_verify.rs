//! The migration verifier and the verified cutover (cutover spec §3.4).

use super::verify::{self, ParityReport};
use super::{FerroStore, migrate_redb_to_ferrostore};
use crate::RedbStore;
use crate::migrate::{migrate_verified, verify_redb_vs_ferrostore};
use fl_core::agent::{AgentRole, AreaRecord, RegisterAgentArgs};
use fl_core::at::At;
use fl_core::conformance::{entry_iri, sample_attempt, sample_record_run};
use fl_core::escalation::{Escalations, Mark};
use fl_core::ferrowire::{MessageKind, MessageUrgency, SendWireArgs, WireAddress};
use fl_core::finding::Finding;
use fl_core::ids::{GateId, ProjectId, RecordId, RevisionId, WorkflowId, WorkflowRunId, seq_iri};
use fl_core::model::{
    CommandSpec, GateKind, PopulationDelivery, Regret, Selector, State, Transition,
};
use fl_core::routing::RoutingMap;
use fl_core::split::{CachedSegment, LedgerCache, Outbox};

use fl_core::state::{
    HandoffEntry, SensitivityPatch, StatePayload, StateRevision, StateScope, StateSyncCursor,
    UpsertStateArgs,
};
use fl_core::store::{
    AgentRegistry, Bindings, Catalog, Ledger, MessageStore, StateStore, Tracker, WorkflowStore,
};
use fl_core::workflow::{
    AssigneeRule, DEFAULT_MAX_TOTAL_STEPS, DEFAULT_STEP_MAX_ATTEMPTS, DEFAULT_STEP_MAX_VISITS,
    END_STEP, EdgeCondition, EdgeDef, JoinPolicy, StepCompletion, StepDef, WorkflowDef,
    WorkflowRun,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

fn mode(path: &Path) -> u32 {
    std::fs::metadata(path).unwrap().permissions().mode() & 0o7777
}

struct Populated {
    project: ProjectId,
    gate: GateId,
    record: RecordId,
    /// Marked for escalation.
    marked: RecordId,
    /// Marked, then tombstoned.
    gone: RecordId,
}

fn handoff(summary: &str) -> StatePayload {
    StatePayload::Handoff(HandoffEntry {
        summary: summary.into(),
        shipped: vec![],
        in_flight: vec![],
        blockers: vec![],
        next_steps: vec![],
        notes: None,
    })
}

/// A redb store holding a row in every table but `imports`: catalog, tracker,
/// ledger with ids, cut-over and published marks, ledger cache, routing,
/// bindings, agents, state with a conflict and a sync cursor, escalation marks
/// and tombstones, FerroWire and workflows. `imports` needs a store of its own
/// ([`imported`]): a routed project is the only project in its store.
fn populate(redb: &RedbStore) -> Populated {
    let project = redb.add_project("/work/app").unwrap();
    let kind = GateKind::Command(CommandSpec {
        program: "true".into(),
        args: vec![],
        delivery: PopulationDelivery::Args,
        timeout_secs: 5,
        pass_codes: vec![0],
    });
    let selector = Selector::Glob {
        pattern: "**/*.rs".into(),
    };
    let gate = redb
        .add_gate(
            &project,
            "fmt",
            kind.clone(),
            selector.clone(),
            1,
            "c1",
            "o",
        )
        .unwrap();
    let lint = redb
        .add_gate(&project, "lint", kind, selector, 1, "c1", "o")
        .unwrap();
    redb.add_transition(Transition {
        project: project.clone(),
        name: "ship".into(),
        from: State::Review,
        to: State::Done,
        regret: Regret::High,
        gates: vec![gate.clone(), lint],
    })
    .unwrap();
    let record = redb.add_record(&project, "keep this").unwrap();
    redb.set_record_state(&record, State::Doing).unwrap();
    redb.add_finding(Finding::raise(
        project.clone(),
        record.clone(),
        "reviewer",
        "wrong on empty",
    ))
    .unwrap();
    redb.add_alias(record.iri(), entry_iri(90)).unwrap();
    redb.set_cutover("R_1", &entry_iri(0)).unwrap();
    for n in 1..=3 {
        redb.append_gate_run(sample_record_run(n, &gate, Some(&record)))
            .unwrap();
    }
    redb.append_attempt(sample_attempt(4, &project, &record))
        .unwrap();
    redb.append_attempt(sample_attempt(5, &project, &record))
        .unwrap();
    redb.mark_published("R_1", &[entry_iri(1)]).unwrap();
    redb.bind_node_id("Acme/Widgets", "R_1").unwrap();
    redb.set_ledger_root("R_1", "0123456789abcdef0123456789abcdef01234567")
        .unwrap();
    redb.remember(
        "R_1",
        "c1",
        &[(
            "runs/aa/1.jsonl".to_string(),
            CachedSegment {
                oid: "o1".into(),
                bytes: vec![0xff, 0xfe, 0xfd],
                closed: false,
            },
        )],
    )
    .unwrap();
    redb.set_routes(&project, &RoutingMap::starting()).unwrap();

    redb.upsert_area(
        AreaRecord::new(
            "ferroloop",
            Some("FerroLoop/".into()),
            "Agent orchestration",
            At::from_unix_millis(1),
            "owner",
        )
        .unwrap(),
    )
    .unwrap();
    redb.register_agent(RegisterAgentArgs {
        display_name: "bob".into(),
        email: None,
        project: "ferroloop".into(),
        role: AgentRole::Lead,
        extra_areas: vec![],
        create_area: false,
        replace: false,
        at: At::from_unix_millis(2),
        by: "owner".into(),
    })
    .unwrap();
    for slug in ["rule", "other"] {
        redb.upsert_state(UpsertStateArgs {
            scope: StateScope::Workspace,
            slug: slug.into(),
            sensitivity: SensitivityPatch::Inherit,
            payload: StatePayload::Handoff(HandoffEntry {
                summary: format!("{slug} v6"),
                shipped: vec![],
                in_flight: vec![],
                blockers: vec![],
                next_steps: vec![],
                notes: None,
            }),
            at: At::from_unix_millis(10),
            by: "carol".into(),
        })
        .unwrap();
    }

    let workflow = redb
        .upsert_workflow_def(WorkflowDef {
            id: WorkflowId(seq_iri(801)),
            project: project.clone(),
            slug: "ci-flow".into(),
            title: "CI Flow".into(),
            start_step: "build".into(),
            steps: vec![StepDef {
                step_id: "build".into(),
                title: "Build".into(),
                prompt: "Run build.".into(),
                assignee: AssigneeRule::Agent {
                    name: "alice".into(),
                },
                reviews_step: None,
                require_different_vendor: false,
                completion: StepCompletion::Ungated,
                join_policy: JoinPolicy::All,
                max_attempts: DEFAULT_STEP_MAX_ATTEMPTS,
                max_visits: DEFAULT_STEP_MAX_VISITS,
                budget_usd_micros: None,
                max_tokens: None,
                isolate_worktree: false,
            }],
            edges: vec![EdgeDef {
                from_step: "build".into(),
                to_step: END_STEP.into(),
                on: EdgeCondition::Pass,
            }],
            max_total_steps: DEFAULT_MAX_TOTAL_STEPS,
            budget_usd_micros: None,
            max_tokens: None,
            declared_keys: BTreeMap::new(),
            authored_at_commit: "0123456789abcdef0123456789abcdef01234567".into(),
            authored_by: "bob".into(),
        })
        .unwrap();
    redb.create_workflow_run(
        WorkflowRun::new(
            WorkflowRunId(seq_iri(802)),
            &workflow,
            "CI Run 1",
            At::from_unix_millis(100),
            "bob",
        )
        .unwrap(),
    )
    .unwrap();
    let (first, _) = redb
        .send_wire(SendWireArgs {
            id: None,
            sender_id: None,
            sender_name: "carol".into(),
            sender_host: Some("build-host".into()),
            to: WireAddress::AgentName("bob".into()),
            kind: MessageKind::Mail,
            urgency: MessageUrgency::Normal,
            durable: false,
            subject: "first".into(),
            body: "hello".into(),
            reply_to: None,
            thread_id: None,
            deadline_at: None,
            at: At::from_unix_millis(10),
        })
        .unwrap();
    redb.send_wire(SendWireArgs {
        id: None,
        sender_id: None,
        sender_name: "bob".into(),
        sender_host: None,
        to: WireAddress::AgentName("bob".into()),
        kind: MessageKind::Reply,
        urgency: MessageUrgency::Normal,
        durable: true,
        subject: "re: first".into(),
        body: "reply".into(),
        reply_to: Some(first.id.clone()),
        thread_id: Some(first.thread_id.clone()),
        deadline_at: None,
        at: At::from_unix_millis(11),
    })
    .unwrap();

    // A state conflict: a local edit and a remote revision off one base.
    let base = redb
        .upsert_state(UpsertStateArgs {
            scope: StateScope::Workspace,
            slug: "diverged".into(),
            sensitivity: SensitivityPatch::Inherit,
            payload: handoff("base"),
            at: At::from_unix_millis(20),
            by: "carol".into(),
        })
        .unwrap();
    redb.upsert_state(UpsertStateArgs {
        scope: StateScope::Workspace,
        slug: "diverged".into(),
        sensitivity: SensitivityPatch::Inherit,
        payload: handoff("local"),
        at: At::from_unix_millis(21),
        by: "carol".into(),
    })
    .unwrap();
    redb.import_state_revisions(
        &[StateRevision {
            revision_id: RevisionId(seq_iri(903)),
            entry_id: base.id.clone(),
            parent_revisions: vec![base.head_revision.clone()],
            scope: base.scope.clone(),
            concern: base.concern,
            slug: base.slug.clone(),
            sensitive: false,
            ever_sensitive: false,
            payload: handoff("remote"),
            at: At::from_unix_millis(22),
            by: "bob".into(),
        }],
        At::from_unix_millis(23),
    )
    .unwrap();
    assert_eq!(redb.list_conflicts().unwrap().len(), 1);
    redb.set_sync_cursor(StateSyncCursor {
        remote: "origin".into(),
        branch: "fl/state".into(),
        last_pushed_commit: Some("0123456789abcdef0123456789abcdef01234567".into()),
        last_revision_seq: 3,
        last_registry_seq: 1,
        last_wire_msg_seq: 2,
        last_wire_rcpt_seq: 2,
    })
    .unwrap();

    // Escalation, last: a mark blocks writes to its item. One record stays
    // marked; another is marked, then tombstoned.
    let mark = Mark {
        by: "bob".into(),
        reason: "to GitHub".into(),
        at_ms: 30,
    };
    let marked = redb.add_record(&project, "escalating").unwrap();
    redb.mark(marked.iri(), &mark).unwrap();
    let gone = redb.add_record(&project, "escalated").unwrap();
    redb.mark(gone.iri(), &mark).unwrap();
    redb.tombstone(gone.iri(), &entry_iri(91)).unwrap();

    Populated {
        project,
        gate,
        record,
        marked,
        gone,
    }
}

/// A redb store holding an imported project, and so an `imports` row,
/// migrated into a new on-disk FerroStore.
fn imported() -> (tempfile::TempDir, RedbStore, FerroStore, ProjectId) {
    let dir = tempfile::tempdir().unwrap();
    let author = RedbStore::open(&dir.path().join("author.redb")).unwrap();
    let project = author.add_project("/author").unwrap();
    author
        .add_gate(
            &project,
            "fmt",
            GateKind::Command(CommandSpec {
                program: "true".into(),
                args: vec![],
                delivery: PopulationDelivery::Args,
                timeout_secs: 5,
                pass_codes: vec![0],
            }),
            Selector::Glob {
                pattern: "**/*.rs".into(),
            },
            1,
            "c1",
            "o",
        )
        .unwrap();
    let manifest = author.export_manifest(&project, "c1", 1, None).unwrap();
    let redb = RedbStore::open(&dir.path().join("src.redb")).unwrap();
    redb.import_manifest(&manifest, "/imported").unwrap();
    let ferro = FerroStore::open(dir.path().join("dst.fls")).unwrap();
    migrate_redb_to_ferrostore(&redb, &ferro).unwrap();
    (dir, redb, ferro, project)
}

fn view<'a>(report: &'a ParityReport, prefix: &str) -> &'a verify::TableParity {
    report
        .trait_views
        .iter()
        .find(|v| v.table.starts_with(prefix))
        .unwrap_or_else(|| panic!("no view {prefix}"))
}

fn table<'a>(report: &'a ParityReport, name: &str) -> &'a verify::TableParity {
    report
        .tables
        .iter()
        .find(|t| t.table == name)
        .unwrap_or_else(|| panic!("no table {name}"))
}

fn mismatch_names(report: &ParityReport) -> Vec<String> {
    report
        .mismatches()
        .into_iter()
        .map(|m| m.table.clone())
        .collect()
}

/// A populated redb, migrated into a new on-disk FerroStore.
fn migrated() -> (tempfile::TempDir, RedbStore, FerroStore, Populated) {
    let dir = tempfile::tempdir().unwrap();
    let redb = RedbStore::open(&dir.path().join("src.redb")).unwrap();
    let populated = populate(&redb);
    let ferro = FerroStore::open(dir.path().join("dst.fls")).unwrap();
    migrate_redb_to_ferrostore(&redb, &ferro).unwrap();
    (dir, redb, ferro, populated)
}

#[test]
fn parity_is_clean_on_a_migrated_populated_store() {
    let (_dir, redb, ferro, populated) = migrated();
    let report = verify_redb_vs_ferrostore(&redb, &ferro).unwrap();
    assert!(report.is_clean(), "{:#?}", report.mismatches());

    // Not a vacuous pass: the entries that matter hold rows on both sides.
    for name in [
        "meta",
        "ids",
        "aliases",
        "handles",
        "handle_of",
        "projects",
        "gates",
        "transitions",
        "records",
        "findings",
        "gate_runs",
        "attempts",
        "github_bindings",
        "ledger_published",
        "ledger_cutovers",
        "ledger_candidate_runs",
        "ledger_candidate_attempts",
        "ledger_roots",
        "ledger_heads",
        "ledger_segments",
        "routing",
        "escalating",
        "tombstones",
        "state_entries",
        "state_keys",
        "state_revisions",
        "state_revision_ids",
        "state_conflicts",
        "state_sync_cursors",
        "agents",
        "agent_by_name",
        "areas",
        "agent_events",
        "agent_event_ids",
        "wire_messages",
        "wire_msg_log",
        "wire_msg_seq",
        "wire_inbox",
        "wire_rcpt_log",
        "wire_threads",
        "wire_replies",
        "workflow_defs",
        "workflow_slugs",
        "workflow_runs",
        "__meta",
    ] {
        let t = table(&report, name);
        assert!(t.redb_rows > 0, "{name} is empty: the fixture misses it");
        assert_eq!(t.redb_rows, t.ferro_rows, "{name}");
    }
    assert_eq!(table(&report, "ferrostore:unlisted").ferro_rows, 0);
    assert!(
        !report.tables.iter().any(|t| t.table.starts_with("redb:")),
        "every redb table is on the copy list"
    );
    for (prefix, at_least) in [
        ("Catalog::list_projects", 1),
        ("Catalog::list_gates(", 2),
        ("Catalog::list_transitions(", 1),
        ("Tracker::list_records(", 1),
        ("Tracker::list_findings(", 1),
        ("Ledger::attempts(", 2),
        ("Ledger::gate_runs(", 3),
        ("Outbox::unpublished(R_1", 2),
        ("LedgerCache::cached_under(R_1, runs)", 1),
        ("AgentRegistry::list_areas", 1),
        ("AgentRegistry::list_agents", 1),
        ("AgentRegistry::registry_events_after", 1),
        ("StateStore::list_state", 3),
        ("StateStore::list_conflicts", 1),
        ("StateStore::state_revisions_after", 2),
        ("StateStore::state_revisions(", 1),
        ("MessageStore::list_wire_inbox", 2),
        ("MessageStore::wire_messages_after", 2),
        ("MessageStore::wire_receipts_after", 2),
        ("MessageStore::list_wire_thread(", 2),
        ("MessageStore::list_wire_receipts(", 1),
        ("WorkflowStore::list_workflow_defs(all)", 1),
        ("WorkflowStore::list_workflow_runs(all)", 1),
    ] {
        let v = view(&report, prefix);
        assert!(v.redb_rows >= at_least, "{prefix}: {v:?}");
    }
    assert!(
        report
            .trait_views
            .iter()
            .any(|v| v.table == format!("Ledger::gate_runs({})", populated.gate))
    );
    assert!(
        report
            .trait_views
            .iter()
            .any(|v| v.table == format!("Tracker::list_records({})", populated.project))
    );

    // `imports` holds a row only in a store of its own.
    let (_dir, redb, ferro, project) = imported();
    let report = verify_redb_vs_ferrostore(&redb, &ferro).unwrap();
    assert!(report.is_clean(), "{:#?}", report.mismatches());
    let imports = table(&report, "imports");
    assert_eq!((imports.redb_rows, imports.ferro_rows), (1, 1));
    assert!(ferro.imported_hash(&project).unwrap().is_some());
}

#[test]
fn an_escalation_row_changed_after_migration_is_caught_by_its_table_hash() {
    // `escalating` and `tombstones` have no list-style trait view: only the
    // table hash sees them.
    let (_dir, redb, ferro, populated) = migrated();
    ferro
        .delete_doc("escalating", populated.marked.iri().as_str())
        .unwrap();
    let report = verify_redb_vs_ferrostore(&redb, &ferro).unwrap();
    let escalating = table(&report, "escalating");
    assert_eq!(escalating.ferro_rows + 1, escalating.redb_rows);
    assert_eq!(mismatch_names(&report), vec!["escalating".to_string()]);

    let (_dir, redb, ferro, populated) = migrated();
    let mut tomb = ferro
        .get_doc("tombstones", populated.gone.iri().as_str())
        .unwrap()
        .unwrap();
    tomb["reason"] = json!("changed");
    ferro
        .put_doc("tombstones", populated.gone.iri().as_str(), &tomb)
        .unwrap();
    let report = verify_redb_vs_ferrostore(&redb, &ferro).unwrap();
    let tombstones = table(&report, "tombstones");
    assert_eq!(tombstones.redb_rows, tombstones.ferro_rows);
    assert_ne!(tombstones.redb_hash, tombstones.ferro_hash);
    assert_eq!(mismatch_names(&report), vec!["tombstones".to_string()]);
}

#[test]
fn a_row_changed_after_migration_is_a_mismatch_in_its_table_and_its_view() {
    let (_dir, redb, ferro, populated) = migrated();
    let mut gate = ferro
        .get_doc("gates", populated.gate.iri().as_str())
        .unwrap()
        .unwrap();
    gate["name"] = json!("renamed");
    ferro
        .put_doc("gates", populated.gate.iri().as_str(), &gate)
        .unwrap();
    let report = verify_redb_vs_ferrostore(&redb, &ferro).unwrap();
    assert!(!report.is_clean());
    let gates = table(&report, "gates");
    assert_eq!(
        gates.redb_rows, gates.ferro_rows,
        "same count, other content"
    );
    assert_ne!(gates.redb_hash, gates.ferro_hash);
    let names = mismatch_names(&report);
    assert!(names.contains(&"gates".to_string()), "{names:?}");
    assert!(
        names.contains(&format!("Catalog::list_gates({})", populated.project)),
        "{names:?}"
    );
    assert!(!names.contains(&"records".to_string()), "{names:?}");
}

#[test]
fn a_missing_row_is_a_mismatch() {
    let (_dir, redb, ferro, populated) = migrated();
    ferro
        .delete_doc("records", populated.record.iri().as_str())
        .unwrap();
    let report = verify_redb_vs_ferrostore(&redb, &ferro).unwrap();
    let records = table(&report, "records");
    assert_eq!(records.ferro_rows + 1, records.redb_rows);
    let names = mismatch_names(&report);
    assert!(names.contains(&"records".to_string()), "{names:?}");
    assert!(
        names.contains(&format!("Tracker::list_records({})", populated.project)),
        "{names:?}"
    );
}

#[test]
fn a_raw_string_stored_as_parsed_json_is_a_mismatch() {
    // `put_raw` keeps a raw `&str` a JSON string; a value parsed from it is not.
    let (_dir, redb, ferro, populated) = migrated();
    let kind = ferro
        .get_doc("ids", populated.record.iri().as_str())
        .unwrap()
        .unwrap();
    assert!(kind.is_string(), "{kind}");
    ferro
        .put_doc(
            "ids",
            populated.record.iri().as_str(),
            &json!({"kind": kind}),
        )
        .unwrap();
    let report = verify_redb_vs_ferrostore(&redb, &ferro).unwrap();
    assert!(mismatch_names(&report).contains(&"ids".to_string()));
}

#[test]
fn data_outside_the_copy_list_is_a_mismatch_on_either_side() {
    let (_dir, redb, ferro, _) = migrated();
    ferro.put_doc("stray", "x", &json!(1)).unwrap();
    let report = verify_redb_vs_ferrostore(&redb, &ferro).unwrap();
    assert_eq!(
        mismatch_names(&report),
        vec!["ferrostore:unlisted".to_string()]
    );

    let (_dir, redb, ferro, _) = migrated();
    {
        let tx = redb.db.begin_write().unwrap();
        tx.open_table(redb::TableDefinition::<&str, &str>::new("mystery"))
            .unwrap()
            .insert("k", "v")
            .unwrap();
        tx.commit().unwrap();
    }
    let report = verify_redb_vs_ferrostore(&redb, &ferro).unwrap();
    assert_eq!(mismatch_names(&report), vec!["redb:mystery".to_string()]);
    assert_eq!(table(&report, "redb:mystery").redb_rows, 1);

    let (_dir, redb, ferro, _) = migrated();
    ferro
        .put_doc_unchecked(
            "__meta",
            "schema_version",
            &json!(crate::FORMAT_VERSION),
            false,
        )
        .unwrap();
    let report = verify_redb_vs_ferrostore(&redb, &ferro).unwrap();
    assert_eq!(mismatch_names(&report), vec!["__meta".to_string()]);
}

#[test]
fn the_hash_is_sha256_over_sorted_length_prefixed_canonical_rows() {
    let tmp = tempfile::tempdir().unwrap();
    let redb = RedbStore::open(&tmp.path().join("s.redb")).unwrap();
    let ferro = FerroStore::in_memory();
    migrate_redb_to_ferrostore(&redb, &ferro).unwrap();
    ferro
        .put_doc("projects", "b", &json!({"z": 1, "a": [{"d": 1, "c": 2}]}))
        .unwrap();
    ferro.put_doc("projects", "a", &json!("x")).unwrap();
    let report = verify_redb_vs_ferrostore(&redb, &ferro).unwrap();
    let projects = table(&report, "projects");
    let mut want = Sha256::new();
    for (key, value) in [("a", r#""x""#), ("b", r#"{"a":[{"c":2,"d":1}],"z":1}"#)] {
        want.update((key.len() as u64).to_le_bytes());
        want.update(key.as_bytes());
        want.update((value.len() as u64).to_le_bytes());
        want.update(value.as_bytes());
    }
    let want: [u8; 32] = want.finalize().into();
    assert_eq!(projects.ferro_rows, 2);
    assert_eq!(projects.ferro_hash, want);
    assert_eq!(projects.redb_rows, 0);
    assert_eq!(projects.redb_hash, <[u8; 32]>::from(Sha256::digest(b"")));
}

/// A populated redb store file at `dir/fl.redb`, closed.
fn redb_file(dir: &Path) -> std::path::PathBuf {
    let path = dir.join("fl.redb");
    let redb = RedbStore::open(&path).unwrap();
    populate(&redb);
    path
}

fn entries(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

#[test]
fn a_verified_migration_publishes_the_store_and_freezes_the_redb() {
    let tmp = tempfile::tempdir().unwrap();
    let redb_path = redb_file(tmp.path());
    let fls = tmp.path().join("fl.fls");
    let report = migrate_verified(&redb_path, &fls).unwrap();
    assert!(report.is_clean(), "{:#?}", report.mismatches());
    assert_eq!(
        entries(tmp.path()),
        vec!["fl.fls".to_string(), "fl.redb".to_string()],
        "no temp file is left"
    );
    assert_eq!(mode(&fls), 0o600);
    assert_eq!(mode(&redb_path), 0o400, "the redb is kept, read-only");
    let ferro = FerroStore::open(&fls).unwrap();
    assert_eq!(ferro.list_projects().unwrap().len(), 1);
    drop(ferro);

    // An existing store is never replaced.
    let held = std::fs::read(&fls).unwrap();
    std::fs::set_permissions(&redb_path, std::fs::Permissions::from_mode(0o600)).unwrap();
    let err = migrate_verified(&redb_path, &fls).unwrap_err();
    assert!(err.to_string().contains("already exists"), "{err}");
    assert_eq!(std::fs::read(&fls).unwrap(), held);
    assert_eq!(mode(&redb_path), 0o600, "a refusal changes nothing");

    // A missing redb is not created by the attempt.
    let missing = tmp.path().join("none.redb");
    assert!(migrate_verified(&missing, &tmp.path().join("none.fls")).is_err());
    assert!(!missing.exists());
    assert!(!tmp.path().join("none.fls").exists());
}

#[test]
fn an_unclean_verify_leaves_no_store_and_no_temp_file() {
    let tmp = tempfile::tempdir().unwrap();
    let redb_path = redb_file(tmp.path());
    let fls = tmp.path().join("fl.fls");
    let report = verify::migrate_verified_with(&redb_path, &fls, |dst| {
        let ids = dst.query_by_field("records", "title", "keep this")?;
        assert_eq!(ids.len(), 1);
        dst.delete_doc("records", &ids[0])
    })
    .unwrap();
    assert!(!report.is_clean());
    assert!(mismatch_names(&report).contains(&"records".to_string()));
    assert_eq!(
        entries(tmp.path()),
        vec!["fl.redb".to_string()],
        "neither the store nor the temp file is left"
    );
    assert_eq!(mode(&redb_path), 0o600, "the redb stays writable");
    // The caller keeps running on redb.
    let redb = RedbStore::open(&redb_path).unwrap();
    assert_eq!(redb.list_projects().unwrap().len(), 1);
}

#[test]
fn a_failed_step_after_the_copy_leaves_no_temp_file() {
    let tmp = tempfile::tempdir().unwrap();
    let redb_path = redb_file(tmp.path());
    let fls = tmp.path().join("fl.fls");
    let err = verify::migrate_verified_with(&redb_path, &fls, |_| {
        Err(crate::backend("injected failure"))
    })
    .unwrap_err();
    assert!(err.to_string().contains("injected failure"), "{err}");
    assert_eq!(entries(tmp.path()), vec!["fl.redb".to_string()]);
}

/// Run two verified migrations into `fls` at once, from `sources[0]` and
/// `sources[1]`; each meets the other at a barrier between its copy and its
/// verify when `meet` is set (two sources, so neither waits on the other's
/// redb lock).
fn race(
    sources: [&Path; 2],
    fls: &Path,
    meet: bool,
) -> Vec<Result<ParityReport, fl_core::StoreError>> {
    use std::sync::{Arc, Barrier};
    let start = Arc::new(Barrier::new(2));
    let copied = Arc::new(Barrier::new(2));
    std::thread::scope(|scope| {
        let handles: Vec<_> = sources
            .into_iter()
            .map(|source| {
                let (start, copied) = (Arc::clone(&start), Arc::clone(&copied));
                scope.spawn(move || {
                    // Both pass the "target exists" check before either publishes.
                    start.wait();
                    verify::migrate_verified_with(source, fls, |_| {
                        if meet {
                            copied.wait();
                        }
                        Ok(())
                    })
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// Exactly one of `results` published `fls`, the other is an error, no temp
/// file is left, and the published store verifies clean against `redb_path`.
fn published_once(
    results: &[Result<ParityReport, fl_core::StoreError>],
    dir: &Path,
    fls: &Path,
    redb_path: &Path,
    want: &[&str],
) {
    let published: Vec<&ParityReport> = results.iter().filter_map(|r| r.as_ref().ok()).collect();
    assert_eq!(published.len(), 1, "{results:?}");
    assert!(published[0].is_clean(), "{:#?}", published[0].mismatches());
    assert!(results.iter().any(Result::is_err), "{results:?}");
    let want: Vec<String> = want.iter().map(|name| name.to_string()).collect();
    assert_eq!(entries(dir), want, "no temp file is left");
    assert_eq!(mode(fls), 0o600);

    // What was published is a whole store that verifies clean.
    std::fs::set_permissions(redb_path, std::fs::Permissions::from_mode(0o600)).unwrap();
    let redb = RedbStore::open(redb_path).unwrap();
    let ferro = FerroStore::open(fls).unwrap();
    let report = verify_redb_vs_ferrostore(&redb, &ferro).unwrap();
    assert!(report.is_clean(), "{:#?}", report.mismatches());
}

#[test]
fn two_verified_migrations_of_one_target_publish_once_and_leave_no_temp_file() {
    // One source: the redb lock orders the two, and the second finds the
    // first's work done.
    let tmp = tempfile::tempdir().unwrap();
    let redb_path = redb_file(tmp.path());
    let fls = tmp.path().join("fl.fls");
    let results = race([&redb_path, &redb_path], &fls, false);
    published_once(
        &results,
        tmp.path(),
        &fls,
        &redb_path,
        &["fl.fls", "fl.redb"],
    );

    // Two copies of one source, both copied before either verifies: both
    // temp files exist at once, both verify clean, and only one links.
    let tmp = tempfile::tempdir().unwrap();
    let redb_path = redb_file(tmp.path());
    let copy = tmp.path().join("copy.redb");
    std::fs::copy(&redb_path, &copy).unwrap();
    let fls = tmp.path().join("fl.fls");
    let results = race([&redb_path, &copy], &fls, true);
    let lost = results
        .iter()
        .find_map(|r| r.as_ref().err())
        .expect("one migration loses");
    assert!(lost.to_string().contains("move"), "{lost}");
    published_once(
        &results,
        tmp.path(),
        &fls,
        &redb_path,
        &["copy.redb", "fl.fls", "fl.redb"],
    );
}

#[test]
fn a_taken_temp_name_is_an_error_and_is_left_alone() {
    let tmp = tempfile::tempdir().unwrap();
    let redb_path = redb_file(tmp.path());
    let fls = tmp.path().join("fl.fls");
    // A temp name already taken by someone else.
    let taken = tmp.path().join("fl.fls.migrating-taken");
    std::fs::write(&taken, b"not ours").unwrap();
    let err = verify::migrate_verified_with_temp(&redb_path, &fls, &taken).unwrap_err();
    assert!(err.to_string().contains("create"), "{err}");
    assert_eq!(std::fs::read(&taken).unwrap(), b"not ours");
    assert!(!fls.exists());
    assert_eq!(mode(&redb_path), 0o600);
}

#[test]
fn a_temp_file_replaced_before_the_link_is_not_published_or_removed() {
    let tmp = tempfile::tempdir().unwrap();
    let redb_path = redb_file(tmp.path());
    let fls = tmp.path().join("fl.fls");
    let dir = tmp.path().to_path_buf();
    let err = verify::migrate_verified_with(&redb_path, &fls, |_| {
        let temp = entries(&dir)
            .into_iter()
            .find(|name| name.contains(".migrating-"))
            .expect("the temp file exists during the copy");
        let impostor = dir.join("impostor");
        std::fs::write(&impostor, b"unverified").unwrap();
        std::fs::rename(&impostor, dir.join(temp)).unwrap();
        Ok(())
    })
    .unwrap_err();
    // The verify's own reads refuse the swap first: they check the path
    // (spec §5) and find a file that is no store. The identity check before
    // the link has its own test, with an impostor the verify accepts
    // (`a_temp_file_replaced_by_a_valid_copy_before_the_link_...`).
    assert!(
        err.to_string().contains("no valid FerroStore superblock"),
        "{err}"
    );
    assert!(!fls.exists(), "nothing is published");
    let left: Vec<String> = entries(tmp.path())
        .into_iter()
        .filter(|name| name.contains(".migrating-"))
        .collect();
    assert_eq!(left.len(), 1, "the file this call did not create is left");
    assert_eq!(
        std::fs::read(tmp.path().join(&left[0])).unwrap(),
        b"unverified"
    );
    assert_eq!(mode(&redb_path), 0o600, "the redb stays writable");
}

#[test]
fn a_temp_file_replaced_by_a_valid_copy_before_the_link_is_refused_by_the_identity_check() {
    let tmp = tempfile::tempdir().unwrap();
    let redb_path = redb_file(tmp.path());
    let fls = tmp.path().join("fl.fls");
    let dir = tmp.path().to_path_buf();
    let err = verify::migrate_verified_with(&redb_path, &fls, |_| {
        let temp = entries(&dir)
            .into_iter()
            .find(|name| name.contains(".migrating-"))
            .expect("the temp file exists during the copy");
        // A byte copy of the migrated temp: a valid store holding exactly
        // what was migrated, so the verify passes, and only the identity
        // check can tell that it is not the file this call created.
        let impostor = dir.join("impostor");
        std::fs::copy(dir.join(&temp), &impostor).unwrap();
        std::fs::rename(&impostor, dir.join(temp)).unwrap();
        Ok(())
    })
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("is no longer the file this migration verified (before the link)"),
        "{err}"
    );
    assert!(!fls.exists(), "nothing is published");
    let left: Vec<String> = entries(tmp.path())
        .into_iter()
        .filter(|name| name.contains(".migrating-"))
        .collect();
    assert_eq!(left.len(), 1, "the file this call did not create is left");
    assert!(
        FerroStore::open(tmp.path().join(&left[0])).is_ok(),
        "the copy is a valid store"
    );
    assert_eq!(mode(&redb_path), 0o600, "the redb stays writable");
}

/// The cause a no-wait open of `redb_path` fails with, or `None` when it opens.
fn open_failure(redb_path: &Path) -> Option<String> {
    match RedbStore::open_with_retries(redb_path, 0) {
        Ok(_) => None,
        Err(fl_core::StoreError::Unreachable { cause, .. }) => Some(cause),
        Err(other) => panic!("not an unreachable store: {other}"),
    }
}

/// What redb says when another open file description holds the lock.
const LOCKED: &str = "Database already open. Cannot acquire lock.";

/// A permission check is meaningless for root, who writes a mode-0400 file.
fn is_root() -> bool {
    // SAFETY: `geteuid` takes no arguments, cannot fail and touches no memory.
    unsafe { libc::geteuid() == 0 }
}

#[test]
fn the_redb_stays_locked_from_the_copy_until_after_the_chmod() {
    let tmp = tempfile::tempdir().unwrap();
    let redb_path = redb_file(tmp.path());
    let fls = tmp.path().join("fl.fls");
    let mut after_copy = None;
    let mut before_chmod = None;
    let report = verify::migrate_verified_with_before_chmod(
        &redb_path,
        &fls,
        |_| {
            after_copy = Some(open_failure(&redb_path));
            Ok(())
        },
        || {
            // Linked and synced, not yet frozen: only the lock can refuse this.
            assert!(fls.is_file(), "the store is published before the chmod");
            assert_eq!(mode(&redb_path), 0o600, "the chmod has not run yet");
            before_chmod = Some(open_failure(&redb_path));
        },
    )
    .unwrap();
    assert!(report.is_clean(), "{:#?}", report.mismatches());
    assert_eq!(after_copy, Some(Some(LOCKED.to_string())));
    assert_eq!(before_chmod, Some(Some(LOCKED.to_string())));
    assert_eq!(mode(&redb_path), 0o400);

    // Once the call returns the lock is gone: what refuses an open now is the
    // mode, and with the mode put back the file opens at once.
    if !is_root() {
        let cause = open_failure(&redb_path).expect("a 0400 redb does not open for writing");
        assert!(!cause.contains(LOCKED), "{cause}");
        assert!(cause.contains("Permission denied"), "{cause}");
    }
    std::fs::set_permissions(&redb_path, std::fs::Permissions::from_mode(0o600)).unwrap();
    assert_eq!(open_failure(&redb_path), None, "the lock was released");
}

/// Set `path`'s modification time `age` in the past.
fn age_by(path: &Path, age: std::time::Duration) {
    let when = std::time::SystemTime::now() - age;
    let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
    file.set_times(std::fs::FileTimes::new().set_modified(when))
        .unwrap();
    assert!(
        std::fs::metadata(path).unwrap().modified().unwrap() <= when,
        "the mtime was set"
    );
}

#[test]
fn a_migration_sweeps_only_this_targets_stale_temp_files() {
    let day = std::time::Duration::from_secs(24 * 60 * 60);
    let old = day + std::time::Duration::from_secs(60 * 60);
    let tmp = tempfile::tempdir().unwrap();
    let redb_path = redb_file(tmp.path());
    let fls = tmp.path().join("fl.fls");
    let dir = tmp.path();
    let write_aged = |name: &str, age: Option<std::time::Duration>| {
        let path = dir.join(name);
        std::fs::write(&path, b"left by a crash").unwrap();
        if let Some(age) = age {
            age_by(&path, age);
        }
        path
    };
    let stale = write_aged("fl.fls.migrating-4242-0-0badf00d", Some(old));
    let fresh = write_aged("fl.fls.migrating-4242-1-0badf00d", None);
    let unlike = [
        // Not the pattern: a suffix, a word, a short or upper-case nonce.
        "fl.fls.migrating-4242-0-0badf00d.bak",
        "fl.fls.migrating-taken",
        "fl.fls.migrating-4242-0-badf00d",
        "fl.fls.migrating-4242-0-0BADF00D",
        // Another target's temp files.
        "other.fls.migrating-4242-0-0badf00d",
        "xfl.fls.migrating-4242-0-0badf00d",
    ]
    .map(|name| write_aged(name, Some(old)));
    // A symlink with a matching name, to an old file of the right form: the
    // sweep looks at the link itself, which is not a regular file.
    let elsewhere = dir.join("elsewhere");
    std::fs::create_dir(&elsewhere).unwrap();
    let target = elsewhere.join("fl.fls.migrating-4242-2-0badf00d");
    std::fs::write(&target, b"not the sweep's").unwrap();
    age_by(&target, old);
    let link = dir.join("fl.fls.migrating-4242-2-0badf00d");
    std::os::unix::fs::symlink(&target, &link).unwrap();
    // A directory with a matching name is not a temp file either.
    let subdir = dir.join("fl.fls.migrating-4242-3-0badf00d");
    std::fs::create_dir(&subdir).unwrap();

    let report = migrate_verified(&redb_path, &fls).unwrap();
    assert!(report.is_clean(), "{:#?}", report.mismatches());
    assert!(!stale.exists(), "the stale temp file is removed");
    assert!(fresh.exists(), "a temp file written within 24 h is kept");
    for path in &unlike {
        assert!(path.exists(), "{} is kept", path.display());
    }
    assert!(
        std::fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink(),
        "the symlink is kept"
    );
    assert_eq!(std::fs::read(&target).unwrap(), b"not the sweep's");
    assert!(subdir.is_dir(), "the directory is kept");
    assert!(fls.is_file());
}

#[test]
fn a_frozen_redb_says_how_to_reach_the_store() {
    if is_root() {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let redb_path = redb_file(tmp.path());
    let fls = tmp.path().join("fl.fls");
    assert!(migrate_verified(&redb_path, &fls).unwrap().is_clean());

    let err = RedbStore::open_with_retries(&redb_path, 0).err().unwrap();
    let text = err.to_string();
    assert!(
        matches!(err, fl_core::StoreError::Unreachable { .. }),
        "{text}"
    );
    for want in [
        "Permission denied",
        "read-only (mode 400)",
        "FerroStore migration",
        &format!("{} (it exists)", fls.display()),
        "Use the FerroStore engine",
        "FL_STORE_BACKEND=ferro",
        &format!("chmod 600 '{}'", redb_path.display()),
    ] {
        assert!(text.contains(want), "missing {want:?} in: {text}");
    }

    // The .fls gone (or never made): the text says so.
    std::fs::remove_file(&fls).unwrap();
    let text = RedbStore::open_with_retries(&redb_path, 0)
        .err()
        .unwrap()
        .to_string();
    assert!(
        text.contains(&format!("{} (it does not exist)", fls.display())),
        "{text}"
    );

    // A refusal for any other reason keeps redb's own text.
    std::fs::set_permissions(&redb_path, std::fs::Permissions::from_mode(0o600)).unwrap();
    let held = RedbStore::open(&redb_path).unwrap();
    assert_eq!(open_failure(&redb_path), Some(LOCKED.to_string()));
    drop(held);
}
