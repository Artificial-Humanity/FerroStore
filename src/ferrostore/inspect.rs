//! What a `.fls` file holds, read without writing it: both superblock slots,
//! and the state the live one names (`ferrostore inspect`).
//!
//! The file is opened `O_RDONLY` and only read: no lock is taken, nothing is
//! created, initialised, repaired, migrated or checkpointed, so an inspection
//! is safe beside a live store and works on a 0400 file. It reads what
//! [`super::FerroStore::stats_at`] reads, and no more is written than there.

use super::compact::{StoreStats, store_stats};
use super::container_io::{ContainerFs, ContainerIo, RealFs};
use super::layout::{SLOT_LEN, SlotRead, parse_superblock, slot_header};
use super::{DocState, vector};
use super::{
    backend, columnar, doc, io_err, load_full, name_store, parsed_superblock, slot_offset,
};
use fl_core::StoreError;
use serde::Serialize;
use std::path::{Path, PathBuf};

/// What [`inspect_at`] read.
#[derive(Debug, Clone, Serialize)]
pub struct Inspection {
    /// The file read, as given.
    #[serde(serialize_with = "shown")]
    pub path: PathBuf,
    /// The file's length.
    pub file_bytes: u64,
    /// Slot 0 and slot 1, in that order.
    pub slots: Vec<SlotReport>,
    /// The state the live slot names; `None` when it could not be read
    /// (`error` says why).
    pub contents: Option<Contents>,
    /// Why `contents` is `None`.
    pub error: Option<String>,
}

/// One 4 KiB superblock slot.
#[derive(Debug, Clone, Serialize)]
pub struct SlotReport {
    pub index: usize,
    /// Where the slot starts in the file.
    pub offset: u64,
    /// `valid`, `invalid` (no magic, a torn or stale CRC, or bad fields),
    /// `newer` (a CRC-valid slot a newer build wrote) or `absent` (the file
    /// ends before the slot).
    pub state: &'static str,
    /// The slot the store reads: the valid one with the higher epoch.
    pub live: bool,
    /// The container version the slot records (read when it has the magic).
    pub container_version: Option<u32>,
    pub engine: Option<u32>,
    pub incompat: Option<u64>,
    /// The epoch and extents of a valid slot.
    pub superblock: Option<SuperblockReport>,
}

/// The fields of a valid superblock.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct SuperblockReport {
    pub epoch: u64,
    pub snapshot_seq: u64,
    pub snapshot_offset: u64,
    pub snapshot_length: u64,
    pub wal_offset: u64,
    pub wal_length: u64,
}

/// The state the live slot names: its snapshot with its WAL replayed.
#[derive(Debug, Clone, Serialize)]
pub struct Contents {
    /// The schema version the store records (`None`: none recorded).
    pub schema_version: Option<u64>,
    /// Why the schema version could not be read, when it could not.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema_error: Option<String>,
    pub epoch: u64,
    pub last_seq: u64,
    pub live_bytes: u64,
    pub dead_bytes: u64,
    /// Documents in every collection.
    pub documents: u64,
    /// Every document collection that holds a document, the store's own
    /// (`__`) included, in name order.
    pub collections: Vec<CollectionReport>,
    pub column_tables: Vec<ColumnTableReport>,
    pub vector_collections: Vec<VectorCollectionReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CollectionReport {
    pub name: String,
    pub docs: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ColumnTableReport {
    pub name: String,
    /// `name:type` for each column, in order.
    pub columns: Vec<String>,
    pub row_groups: u64,
    pub rows: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct VectorCollectionReport {
    pub name: String,
    pub count: u64,
    pub dim: u32,
    /// `f32`, `f16` or `i8`.
    pub storage: &'static str,
    /// `cosine`, `l2` or `dot`.
    pub metric: &'static str,
    pub m: u32,
    pub ef_construction: u32,
    pub ef_search: u32,
}

fn shown<S: serde::Serializer>(path: &Path, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&path.display().to_string())
}

/// Read the `.fls` file at `path` without writing it (see the module
/// documentation). A file that cannot be opened or read is an error; a file
/// whose state cannot be decoded (no valid slot, a newer build's slot, a
/// corrupt snapshot) is reported, slots and all, with `error` set.
pub fn inspect_at(path: impl AsRef<Path>) -> Result<Inspection, StoreError> {
    let path = path.as_ref();
    let file = RealFs
        .open_ro(path)
        .map_err(|err| backend(format!("open {}: {err}", path.display())))?;
    let file_bytes = file.len().map_err(io_err)?;
    let mut slots = Vec::with_capacity(2);
    for index in 0..2 {
        slots.push(read_slot(&*file, file_bytes, index)?);
    }
    let (contents, error) = match parsed_superblock(&*file) {
        Ok(Some((index, sb))) => {
            if let Some(slot) = slots.get_mut(index) {
                slot.live = true;
            }
            let id = file.file_id().map_err(io_err)?;
            match load_full(&*file, id, sb) {
                Ok(cache) => {
                    let stats = store_stats(file_bytes, &cache);
                    (Some(contents_of(&cache.state, &stats)), None)
                }
                Err(err) => (None, Some(name_store(err, path).to_string())),
            }
        }
        Ok(None) => (None, Some("no valid FerroStore superblock".to_string())),
        Err(err) => (None, Some(name_store(err, path).to_string())),
    };
    Ok(Inspection {
        path: path.to_path_buf(),
        file_bytes,
        slots,
        contents,
        error,
    })
}

fn read_slot(
    file: &dyn ContainerIo,
    file_bytes: u64,
    index: usize,
) -> Result<SlotReport, StoreError> {
    let offset = slot_offset(index);
    let mut report = SlotReport {
        index,
        offset,
        state: "absent",
        live: false,
        container_version: None,
        engine: None,
        incompat: None,
        superblock: None,
    };
    if file_bytes < offset + SLOT_LEN as u64 {
        return Ok(report);
    }
    let mut buf = [0u8; SLOT_LEN];
    file.read_at(&mut buf, offset).map_err(io_err)?;
    if let Some((version, fields)) = slot_header(&buf) {
        report.container_version = Some(version);
        if let Some((engine, incompat)) = fields {
            report.engine = Some(engine);
            report.incompat = Some(incompat);
        }
    }
    report.state = match parse_superblock(&buf) {
        SlotRead::Valid(sb) => {
            report.superblock = Some(SuperblockReport {
                epoch: sb.epoch,
                snapshot_seq: sb.snapshot_seq,
                snapshot_offset: sb.snapshot_offset,
                snapshot_length: sb.snapshot_length,
                wal_offset: sb.wal_offset,
                wal_length: sb.wal_length,
            });
            "valid"
        }
        SlotRead::Invalid => "invalid",
        SlotRead::Newer { .. } => "newer",
    };
    Ok(report)
}

fn contents_of(state: &DocState, stats: &StoreStats) -> Contents {
    let (schema_version, schema_error) =
        match doc::schema_version_in(|collection, id| state.docs.get(collection, id).cloned()) {
            Ok(version) => (version, None),
            Err(err) => (None, Some(err.to_string())),
        };
    let collections: Vec<CollectionReport> = state
        .docs
        .collections()
        .map(|name| CollectionReport {
            name: name.clone(),
            docs: state.docs.in_collection(name).count() as u64,
        })
        .collect();
    let column_tables = state
        .tables
        .iter()
        .map(|(name, table)| ColumnTableReport {
            name: name.clone(),
            columns: table
                .columns
                .iter()
                .map(|c| format!("{}:{}", c.name, columnar::type_name(c.ty)))
                .collect(),
            row_groups: table.groups.len() as u64,
            // A group was checked when it was decoded; a count that does not
            // read is counted as none rather than failing the inspection.
            rows: table
                .groups
                .iter()
                .map(|g| columnar::group_row_count(g).unwrap_or(0))
                .sum(),
        })
        .collect();
    let vector_collections = state
        .vectors
        .iter()
        .map(|(name, col)| VectorCollectionReport {
            name: name.clone(),
            count: col.vectors.len() as u64,
            dim: col.dim,
            storage: vector::storage_name(col.storage),
            metric: vector::metric_name(col.metric),
            m: col.m,
            ef_construction: col.ef_construction,
            ef_search: col.ef_search,
        })
        .collect();
    Contents {
        schema_version,
        schema_error,
        epoch: stats.epoch,
        last_seq: stats.last_seq,
        live_bytes: stats.live_bytes,
        dead_bytes: stats.dead_bytes,
        documents: state.docs.len() as u64,
        collections,
        column_tables,
        vector_collections,
    }
}
