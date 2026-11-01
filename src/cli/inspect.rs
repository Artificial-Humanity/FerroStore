//! `inspect`: what a `.fls` file holds, read through a read-only descriptor
//! ([`crate::ferrostore::inspect_at`]): both superblocks, the schema version,
//! the live and dead bytes, and the counts of each collection, column table
//! and vector collection. Nothing is opened that can write, migrate or
//! repair: no lock, no `.fls` made beside a `.redb`, no checkpoint.

use super::{Output, Result, Wording, line, present};
use crate::ferrostore::{Inspection, SlotReport, inspect_at};
use crate::{BackendKind, physical_path};
use std::path::Path;

/// Inspect the `.fls` that `path` names (`x.redb` names `x.fls`). A missing
/// `.fls` is refused, a lone `.redb` beside it named.
pub fn inspect(path: &Path, w: &Wording) -> Result<Inspection> {
    let physical = physical_path(path, BackendKind::Ferro);
    if !present(&physical) {
        if path != physical && path.exists() {
            refuse!(
                "{} holds a redb store and {} does not exist: `{} inspect` reads a FerroStore \
                 (.fls) file",
                path.display(),
                physical.display(),
                w.command
            );
        }
        refuse!(
            "no store at {}; `{} inspect` does not create one",
            physical.display(),
            w.command
        );
    }
    Ok(inspect_at(&physical)?)
}

/// 0 when the live state was read, 1 when it was not (`error` is set).
pub fn exit_code(report: &Inspection) -> i32 {
    i32::from(report.contents.is_none())
}

fn slot_text(out: &mut String, slot: &SlotReport) {
    let mut head = format!(
        "slot {} (offset {}): {}",
        slot.index, slot.offset, slot.state
    );
    if slot.live {
        head.push_str(", live");
    }
    if let Some(version) = slot.container_version {
        head.push_str(&format!(", container version {version}"));
    }
    if let (Some(engine), Some(incompat)) = (slot.engine, slot.incompat) {
        head.push_str(&format!(", engine {engine}, incompat {incompat:#x}"));
    }
    line(out, head);
    if let Some(sb) = &slot.superblock {
        line(
            out,
            format!(
                "  epoch {}, snapshot seq {}, snapshot {}+{}, wal {}+{}",
                sb.epoch,
                sb.snapshot_seq,
                sb.snapshot_offset,
                sb.snapshot_length,
                sb.wal_offset,
                sb.wal_length
            ),
        );
    }
}

/// One `label: value` line, the values in one column.
fn field(out: &mut String, label: &str, value: impl std::fmt::Display) {
    line(out, format!("{:<20}{value}", format!("{label}:")));
}

/// The text report: stdout, and the reason on stderr when the live state
/// could not be read.
pub fn text(report: &Inspection) -> Output {
    let mut out = Output::default();
    let o = &mut out.stdout;
    field(o, "file", report.path.display());
    field(o, "file bytes", report.file_bytes);
    for slot in &report.slots {
        slot_text(o, slot);
    }
    if let Some(c) = &report.contents {
        let schema = match (&c.schema_version, &c.schema_error) {
            (Some(v), _) => v.to_string(),
            (None, Some(e)) => format!("- ({e})"),
            (None, None) => "- (none recorded)".to_string(),
        };
        field(o, "schema version", schema);
        field(o, "epoch", c.epoch);
        field(o, "last seq", c.last_seq);
        field(o, "live bytes", c.live_bytes);
        field(o, "dead bytes", c.dead_bytes);
        field(o, "documents", c.documents);
        field(o, "collections", c.collections.len());
        for col in &c.collections {
            line(o, format!("  {}: {} docs", col.name, col.docs));
        }
        field(o, "column tables", c.column_tables.len());
        for t in &c.column_tables {
            line(
                o,
                format!(
                    "  {}: {} rows in {} row groups ({})",
                    t.name,
                    t.rows,
                    t.row_groups,
                    t.columns.join(", ")
                ),
            );
        }
        field(o, "vector collections", c.vector_collections.len());
        for v in &c.vector_collections {
            line(
                o,
                format!(
                    "  {}: {} vectors, dim {}, {}, {}, m {}, ef_construction {}, ef_search {}",
                    v.name,
                    v.count,
                    v.dim,
                    v.storage,
                    v.metric,
                    v.m,
                    v.ef_construction,
                    v.ef_search
                ),
            );
        }
    }
    if let Some(e) = &report.error {
        line(
            &mut out.stderr,
            format!("could not read the live state: {e}"),
        );
    }
    out
}
