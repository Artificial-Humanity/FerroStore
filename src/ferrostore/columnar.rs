//! Columnar row groups. One non-empty append is one group; the codec is stored in the blob.

use super::Txn;
use super::doc::{self, ATTEMPTS, GATE_RUNS};
use fl_core::StoreError;
use fl_core::ids::{GateId, ProjectId};
use fl_core::log::{Attempt, GateRun};
use fl_core::verdict::Verdict;
use serde_json::{Number, Value, json};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

const CODEC_DELTA: u8 = 1;
const CODEC_DELTA_RLE: u8 = 2;
const CODEC_PLAIN: u8 = 3;
const CODEC_DICT: u8 = 4;
const CODEC_RLE: u8 = 5;
const NAME_MAX: usize = 128;

/// Most rows one row group may hold. A group's `row_count` is a `u32` in the blob and is
/// otherwise bounded only by the null bitset (`row_count / 8` bytes per column), so without this
/// a few kilobytes of blob claim billions of rows. The limit is checked before any allocation,
/// by the writer (`encode_row_group`) and by the reader (`index_group`). 4 Mi rows is more than
/// any table with columns can receive in one `append_column_rows` call over the 16 MiB remote
/// frame: a cell is at least `{"t":"null"}` (12 bytes), so a one-column row is 15 bytes with its
/// comma and a frame carries about 1.1 M of them. Only a table with no columns could exceed it
/// (a row is `[]`, three bytes with its comma, about 5.6 M per frame); such a remote append of
/// more than 4 Mi rows is refused like any other over-limit group, and the limit is kept for
/// them because the reader's per-row charge is the same. Callers with more rows append several
/// groups. In-memory scans of the ledger (`scan_projected`) are chunked and not bound by it.
pub(super) const MAX_ROWS_PER_GROUP: usize = 1 << 22;

/// Upper bound, shared by all columns of one row group, on the bytes decoding may allocate
/// (1 GiB). Every decoder charges the size it is about to allocate against one [`Budget`] before
/// it allocates, so a hostile blob is refused with an error instead of an out-of-memory abort:
/// a 10-column group of 1 M rows whose columns each repeat a 255-byte text needs about 1.3 MiB of
/// blob and would otherwise allocate about 5 GB. The charge counts `size_of::<ColumnValue>()`
/// per value (not just the payload bytes), the per-row `Vec`, and the heap bytes of text and
/// byte values, and is an upper bound of what is allocated, not of what stays alive.
///
/// The writer applies the same estimate (`estimated_decode_bytes`) and refuses a group over the
/// budget, so nothing FerroStore writes is undecodable. **Behaviour change:** a store that
/// already holds a run-length or dictionary column (or a group of very many rows) that decodes to
/// more than this now fails to open with a "corrupt ferrostore data" error where it used to
/// allocate until the process was killed or, on a big machine, succeeded. Such a store needs
/// groups far larger than any this code base appends (millions of rows in one call).
///
/// There is no store-wide budget: this bounds one group, not their sum. Opening a store and
/// `validate_row_group` decode groups one at a time (peak about this much), but `scan_groups`
/// keeps every matching row of every group it scans, so a scan over N groups that each decode to
/// this size can hold N times it. Groups that large arise only from a corrupt file or snapshot
/// (the remote API's amplification is bounded by the 16 MiB JSON frame), not from this code base.
pub(super) const MAX_DECODE_BYTES: usize = 1024 * 1024 * 1024;

/// Bytes charged for one decoded cell: its slot in the final row vector, its slot in the
/// per-column vector, and its slot in the vector the codec decodes before nulls are interleaved.
const CELL_BYTES: usize = 3 * std::mem::size_of::<ColumnValue>();

/// Bytes charged per row for the row's own `Vec` header.
const ROW_BYTES: usize = std::mem::size_of::<Vec<ColumnValue>>();

/// Row and byte limits for one group. Tests lower them so a hostile blob is refused without the
/// test itself allocating hundreds of MiB.
#[derive(Debug, Clone, Copy)]
pub(super) struct Limits {
    pub(super) max_rows: usize,
    pub(super) max_bytes: usize,
}

impl Limits {
    pub(super) const DEFAULT: Limits = Limits {
        max_rows: MAX_ROWS_PER_GROUP,
        max_bytes: MAX_DECODE_BYTES,
    };
}

/// Bytes still available to one group's decoder; shared by all of its columns.
pub(super) struct Budget {
    left: usize,
    limit: usize,
}

impl Budget {
    pub(super) fn new(limit: usize) -> Self {
        Self { left: limit, limit }
    }

    /// Charge `bytes` before allocating them.
    fn charge(&mut self, bytes: usize, what: &str) -> Result<(), StoreError> {
        if bytes > self.left {
            return Err(corrupt(format!(
                "row group would decode to more than {} bytes ({what})",
                self.limit
            )));
        }
        self.left -= bytes;
        Ok(())
    }
}

fn mul(a: usize, b: usize, what: &str) -> Result<usize, StoreError> {
    a.checked_mul(b)
        .ok_or_else(|| corrupt(format!("{what} overflows")))
}

/// Upper bound of what `decode_indexed` charges for these rows, so the writer refuses what the
/// reader would refuse: `ROW_BYTES` per row, `CELL_BYTES` per cell, and every text or byte
/// value's length plus its four-byte prefix (at least what a plain, run-length or dictionary
/// decoder charges for it).
fn estimated_decode_bytes(columns: &[ColumnDef], rows: &[Vec<ColumnValue>]) -> usize {
    let per_row = ROW_BYTES.saturating_add(columns.len().saturating_mul(CELL_BYTES));
    let heap = rows
        .iter()
        .flatten()
        .map(|cell| match cell {
            ColumnValue::Utf8(text) => text.len().saturating_add(4),
            ColumnValue::Bytes(bytes) => bytes.len().saturating_add(4),
            _ => 0,
        })
        .fold(0usize, usize::saturating_add);
    rows.len().saturating_mul(per_row).saturating_add(heap)
}

fn backend(err: impl std::fmt::Display) -> StoreError {
    StoreError::Backend(err.to_string())
}

fn corrupt(detail: impl std::fmt::Display) -> StoreError {
    StoreError::Backend(format!("corrupt ferrostore data: {detail}"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    Int64,
    Float64,
    Bool,
    Utf8,
    Bytes,
    TimestampMs,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ColumnValue {
    Null,
    Int64(i64),
    Float64(f64),
    Bool(bool),
    Utf8(String),
    Bytes(Vec<u8>),
    TimestampMs(i64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnDef {
    pub name: String,
    pub ty: ColumnType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ColumnPredicate {
    Eq(ColumnValue),
    Lt(ColumnValue),
    Le(ColumnValue),
    Gt(ColumnValue),
    Ge(ColumnValue),
    Between { low: ColumnValue, high: ColumnValue },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColumnFilter {
    pub column: String,
    pub predicate: ColumnPredicate,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ZoneMap {
    pub null_count: u64,
    pub min: Option<ColumnValue>,
    pub max: Option<ColumnValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColumnScan {
    pub rows: Vec<Vec<ColumnValue>>,
    pub row_groups_total: u64,
    pub row_groups_skipped: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AttemptRollup {
    pub count: u64,
    pub cost_usd_micros: u64,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GateRunRollup {
    pub pass: u64,
    pub fail: u64,
    pub error: u64,
    pub duration_ms: u64,
    pub cost_usd_micros: u64,
}

#[derive(Clone, Debug)]
pub(super) struct ColumnTable {
    pub(super) columns: Vec<ColumnDef>,
    pub(super) groups: Vec<Vec<u8>>,
}

impl ZoneMap {
    /// Whether every value in this chunk fails `predicate`. `min`/`max` of `None`
    /// (all null, or only NaN) excludes every predicate.
    pub fn excludes(&self, predicate: &ColumnPredicate) -> bool {
        let (Some(min), Some(max)) = (&self.min, &self.max) else {
            return true;
        };
        match predicate {
            ColumnPredicate::Eq(value) => !(is_ge(value, min) && is_le(value, max)),
            ColumnPredicate::Lt(value) => !is_lt(min, value),
            ColumnPredicate::Le(value) => !is_le(min, value),
            ColumnPredicate::Gt(value) => !is_gt(max, value),
            ColumnPredicate::Ge(value) => !is_ge(max, value),
            ColumnPredicate::Between { low, high } => {
                !(is_le(low, high) && is_ge(max, low) && is_le(min, high))
            }
        }
    }
}

pub(super) fn type_tag(ty: ColumnType) -> u8 {
    match ty {
        ColumnType::Int64 => 1,
        ColumnType::Float64 => 2,
        ColumnType::Bool => 3,
        ColumnType::Utf8 => 4,
        ColumnType::Bytes => 5,
        ColumnType::TimestampMs => 6,
    }
}

pub(super) fn type_from_tag(tag: u8) -> Result<ColumnType, StoreError> {
    match tag {
        1 => Ok(ColumnType::Int64),
        2 => Ok(ColumnType::Float64),
        3 => Ok(ColumnType::Bool),
        4 => Ok(ColumnType::Utf8),
        5 => Ok(ColumnType::Bytes),
        6 => Ok(ColumnType::TimestampMs),
        _ => Err(corrupt(format!("unknown column type tag {tag}"))),
    }
}

/// `taken`: a table named `name` already exists.
pub(super) fn check_new_table(
    taken: bool,
    name: &str,
    columns: &[ColumnDef],
) -> Result<(), StoreError> {
    check_name("table", name)?;
    if taken {
        return Err(backend(format!("column table `{name}` already exists")));
    }
    let mut seen = HashSet::new();
    for column in columns {
        check_name("column", &column.name)?;
        if !seen.insert(column.name.as_str()) {
            return Err(backend(format!("duplicate column `{}`", column.name)));
        }
    }
    Ok(())
}

pub(super) fn check_name(kind: &str, name: &str) -> Result<(), StoreError> {
    if name.is_empty() {
        return Err(backend(format!("{kind} name is empty")));
    }
    if name.len() > NAME_MAX || name.as_bytes().contains(&b'/') || name.as_bytes().contains(&0) {
        return Err(backend(format!("invalid {kind} name {name:?}")));
    }
    Ok(())
}

/// Decode `bytes` against `columns`. A bad header, a bad codec, or a row
/// count of zero is rejected here, before a frame is published.
pub(super) fn validate_row_group(columns: &[ColumnDef], bytes: &[u8]) -> Result<(), StoreError> {
    validate_row_group_with(columns, bytes, &Limits::DEFAULT)
}

fn validate_row_group_with(
    columns: &[ColumnDef],
    bytes: &[u8],
    limits: &Limits,
) -> Result<(), StoreError> {
    let indexed = index_group_with(bytes, columns, limits)?;
    decode_indexed_with(columns, &indexed, limits)?;
    Ok(())
}

pub(super) fn encode_row_group(
    columns: &[ColumnDef],
    rows: &[Vec<ColumnValue>],
) -> Result<Vec<u8>, StoreError> {
    encode_row_group_with(columns, rows, &Limits::DEFAULT)
}

fn encode_row_group_with(
    columns: &[ColumnDef],
    rows: &[Vec<ColumnValue>],
    limits: &Limits,
) -> Result<Vec<u8>, StoreError> {
    if rows.len() > limits.max_rows {
        return Err(backend(format!(
            "{} rows in one group; the limit is {}, append in several groups",
            rows.len(),
            limits.max_rows
        )));
    }
    validate_rows(columns, rows)?;
    if estimated_decode_bytes(columns, rows) > limits.max_bytes {
        return Err(backend(format!(
            "row group would decode to more than {} bytes; append in several groups",
            limits.max_bytes
        )));
    }
    let mut buf = Vec::new();
    let row_count = u32::try_from(rows.len()).map_err(|_| backend("too many rows in one group"))?;
    write_u32(&mut buf, row_count);
    for (index, column) in columns.iter().enumerate() {
        let values = rows
            .iter()
            .map(|row| row[index].clone())
            .collect::<Vec<_>>();
        write_column(&mut buf, column.ty, &values)?;
    }
    Ok(buf)
}

fn validate_rows(columns: &[ColumnDef], rows: &[Vec<ColumnValue>]) -> Result<(), StoreError> {
    for (row_index, row) in rows.iter().enumerate() {
        if row.len() != columns.len() {
            return Err(backend(format!(
                "row {row_index} width {} does not match table width {}",
                row.len(),
                columns.len()
            )));
        }
        for (column, cell) in columns.iter().zip(row.iter()) {
            if !cell_matches(column.ty, cell) {
                return Err(backend(format!(
                    "row {row_index} column `{}` has the wrong type",
                    column.name
                )));
            }
        }
    }
    Ok(())
}

fn cell_matches(ty: ColumnType, value: &ColumnValue) -> bool {
    matches!(
        (ty, value),
        (_, ColumnValue::Null)
            | (ColumnType::Int64, ColumnValue::Int64(_))
            | (ColumnType::Float64, ColumnValue::Float64(_))
            | (ColumnType::Bool, ColumnValue::Bool(_))
            | (ColumnType::Utf8, ColumnValue::Utf8(_))
            | (ColumnType::Bytes, ColumnValue::Bytes(_))
            | (ColumnType::TimestampMs, ColumnValue::TimestampMs(_))
    )
}

fn write_column(
    buf: &mut Vec<u8>,
    ty: ColumnType,
    values: &[ColumnValue],
) -> Result<(), StoreError> {
    let zone = zone_map(values);
    write_zone(buf, ty, &zone)?;
    let non_null = values
        .iter()
        .filter(|value| !matches!(value, ColumnValue::Null))
        .cloned()
        .collect::<Vec<_>>();
    let codec = choose_codec(ty, &non_null)?;
    buf.push(codec);
    let mut body = Vec::new();
    write_bitset(&mut body, values);
    write_payload(&mut body, ty, codec, &non_null)?;
    let len = u32::try_from(body.len()).map_err(|_| backend("column chunk exceeds u32"))?;
    write_u32(buf, len);
    buf.extend_from_slice(&body);
    Ok(())
}

fn zone_map(values: &[ColumnValue]) -> ZoneMap {
    let mut null_count = 0u64;
    let mut bounds: Option<(ColumnValue, ColumnValue)> = None;
    for value in values {
        if matches!(value, ColumnValue::Null) {
            null_count = null_count.saturating_add(1);
            continue;
        }
        if let ColumnValue::Float64(number) = value
            && number.is_nan()
        {
            continue;
        }
        bounds = Some(match bounds {
            None => (value.clone(), value.clone()),
            Some((min, max)) => {
                let min = if is_lt(value, &min) {
                    value.clone()
                } else {
                    min
                };
                let max = if is_gt(value, &max) {
                    value.clone()
                } else {
                    max
                };
                (min, max)
            }
        });
    }
    match bounds {
        Some((min, max)) => ZoneMap {
            null_count,
            min: Some(min),
            max: Some(max),
        },
        None => ZoneMap {
            null_count,
            min: None,
            max: None,
        },
    }
}

fn write_zone(buf: &mut Vec<u8>, ty: ColumnType, zone: &ZoneMap) -> Result<(), StoreError> {
    write_u64(buf, zone.null_count);
    match (&zone.min, &zone.max) {
        (None, None) => buf.push(0),
        (Some(min), Some(max)) => {
            buf.push(1);
            write_typed(buf, ty, min)?;
            write_typed(buf, ty, max)?;
        }
        _ => return Err(backend("zone map has only one bound")),
    }
    Ok(())
}

fn choose_codec(ty: ColumnType, values: &[ColumnValue]) -> Result<u8, StoreError> {
    match ty {
        ColumnType::Int64 | ColumnType::TimestampMs => {
            let ints = collect_i64(ty, values)?;
            let deltas = int_deltas(&ints);
            if has_run_at_least(&deltas, 3) {
                Ok(CODEC_DELTA_RLE)
            } else {
                Ok(CODEC_DELTA)
            }
        }
        ColumnType::Bool => Ok(CODEC_RLE),
        ColumnType::Float64 => {
            let nums = collect_f64(values)?;
            if nums.len() >= 3 && all_same_bits(&nums) {
                Ok(CODEC_RLE)
            } else {
                Ok(CODEC_PLAIN)
            }
        }
        ColumnType::Bytes => {
            let blobs = collect_bytes(values)?;
            if blobs.len() >= 3 && blobs.iter().all(|blob| blob == &blobs[0]) {
                Ok(CODEC_RLE)
            } else {
                Ok(CODEC_PLAIN)
            }
        }
        ColumnType::Utf8 => Ok(utf8_codec(&collect_utf8(values)?)),
    }
}

fn utf8_codec(texts: &[String]) -> u8 {
    if texts.len() >= 3 && texts.iter().all(|text| text == &texts[0]) {
        return CODEC_RLE;
    }
    if texts.len() >= 2 {
        let mut seen = HashSet::new();
        for text in texts {
            seen.insert(text.as_str());
        }
        if seen.len() <= texts.len() / 2 {
            return CODEC_DICT;
        }
    }
    CODEC_PLAIN
}

fn write_payload(
    buf: &mut Vec<u8>,
    ty: ColumnType,
    codec: u8,
    values: &[ColumnValue],
) -> Result<(), StoreError> {
    match (ty, codec) {
        (ColumnType::Int64 | ColumnType::TimestampMs, CODEC_DELTA) => {
            write_delta(buf, &collect_i64(ty, values)?);
        }
        (ColumnType::Int64 | ColumnType::TimestampMs, CODEC_DELTA_RLE) => {
            write_delta_rle(buf, &collect_i64(ty, values)?);
        }
        (ColumnType::Utf8, CODEC_PLAIN) => {
            for text in collect_utf8(values)? {
                write_lp(buf, text.as_bytes())?;
            }
        }
        (ColumnType::Utf8, CODEC_DICT) => write_dict(buf, &collect_utf8(values)?)?,
        (ColumnType::Utf8, CODEC_RLE) => {
            let texts = collect_utf8(values)?;
            if let Some(text) = texts.first() {
                write_lp(buf, text.as_bytes())?;
            }
        }
        (ColumnType::Bytes, CODEC_PLAIN) => {
            for blob in collect_bytes(values)? {
                write_lp(buf, &blob)?;
            }
        }
        (ColumnType::Bytes, CODEC_RLE) => {
            let blobs = collect_bytes(values)?;
            if let Some(blob) = blobs.first() {
                write_lp(buf, blob)?;
            }
        }
        (ColumnType::Float64, CODEC_PLAIN) => {
            for number in collect_f64(values)? {
                buf.extend_from_slice(&number.to_le_bytes());
            }
        }
        (ColumnType::Float64, CODEC_RLE) => {
            let nums = collect_f64(values)?;
            if let Some(number) = nums.first() {
                buf.extend_from_slice(&number.to_le_bytes());
            }
        }
        (ColumnType::Bool, CODEC_RLE) => write_bool_rle(buf, &collect_bool(values)?),
        _ => {
            return Err(backend(format!(
                "codec {codec} does not match the column type"
            )));
        }
    }
    Ok(())
}

fn write_delta(buf: &mut Vec<u8>, values: &[i64]) {
    let Some(&first) = values.first() else {
        return;
    };
    write_i64(buf, first);
    let mut prev = first;
    for &next in &values[1..] {
        write_u64(buf, zigzag_encode(next.wrapping_sub(prev)));
        prev = next;
    }
}

fn write_delta_rle(buf: &mut Vec<u8>, values: &[i64]) {
    let Some(&first) = values.first() else {
        return;
    };
    write_i64(buf, first);
    let deltas = int_deltas(values);
    let Some(&mut_first) = deltas.first() else {
        return;
    };
    let mut current = mut_first;
    let mut len = 1u32;
    for &delta in &deltas[1..] {
        if delta == current {
            len += 1;
            continue;
        }
        write_u64(buf, zigzag_encode(current));
        write_u32(buf, len);
        current = delta;
        len = 1;
    }
    write_u64(buf, zigzag_encode(current));
    write_u32(buf, len);
}

fn write_dict(buf: &mut Vec<u8>, values: &[String]) -> Result<(), StoreError> {
    let mut dict: Vec<&str> = Vec::new();
    let mut index_of: HashMap<&str, u32> = HashMap::new();
    let mut indexes = Vec::with_capacity(values.len());
    for value in values {
        if let Some(&index) = index_of.get(value.as_str()) {
            indexes.push(index);
            continue;
        }
        let index = u32::try_from(dict.len()).map_err(|_| backend("dictionary exceeds u32"))?;
        index_of.insert(value.as_str(), index);
        dict.push(value.as_str());
        indexes.push(index);
    }
    write_u32(
        buf,
        u32::try_from(dict.len()).map_err(|_| backend("dictionary exceeds u32"))?,
    );
    for entry in dict {
        write_lp(buf, entry.as_bytes())?;
    }
    for index in indexes {
        write_u32(buf, index);
    }
    Ok(())
}

fn write_bool_rle(buf: &mut Vec<u8>, values: &[bool]) {
    let Some(&first) = values.first() else {
        return;
    };
    let mut current = first;
    let mut len = 1u32;
    for &value in &values[1..] {
        if value == current {
            len += 1;
            continue;
        }
        buf.push(u8::from(current));
        write_u32(buf, len);
        current = value;
        len = 1;
    }
    buf.push(u8::from(current));
    write_u32(buf, len);
}

fn write_bitset(buf: &mut Vec<u8>, values: &[ColumnValue]) {
    let mut bits = vec![0u8; values.len().div_ceil(8)];
    for (index, value) in values.iter().enumerate() {
        if matches!(value, ColumnValue::Null) {
            bits[index / 8] |= 1 << (index % 8);
        }
    }
    buf.extend_from_slice(&bits);
}

struct IndexedColumn<'a> {
    zone: ZoneMap,
    codec: u8,
    body: &'a [u8],
}

struct IndexedGroup<'a> {
    row_count: usize,
    columns: Vec<IndexedColumn<'a>>,
}

pub(super) fn scan_groups(
    columns: &[ColumnDef],
    groups: &[Vec<u8>],
    filters: &[ColumnFilter],
) -> Result<ColumnScan, StoreError> {
    let bound = bind_filters(columns, filters)?;
    let mut rows = Vec::new();
    let mut skipped = 0u64;
    for group in groups {
        let indexed = index_group(group, columns)?;
        let excluded = bound.iter().any(|filter| {
            indexed.columns[filter.index]
                .zone
                .excludes(filter.predicate)
        });
        if excluded {
            skipped = skipped.saturating_add(1);
            continue;
        }
        for row in decode_indexed(columns, &indexed)? {
            if bound
                .iter()
                .all(|filter| matches_predicate(&row[filter.index], filter.predicate))
            {
                rows.push(row);
            }
        }
    }
    Ok(ColumnScan {
        rows,
        row_groups_total: u64::try_from(groups.len())
            .map_err(|_| backend("too many row groups"))?,
        row_groups_skipped: skipped,
    })
}

struct BoundFilter<'a> {
    index: usize,
    predicate: &'a ColumnPredicate,
}

fn bind_filters<'a>(
    columns: &[ColumnDef],
    filters: &'a [ColumnFilter],
) -> Result<Vec<BoundFilter<'a>>, StoreError> {
    let mut bound = Vec::with_capacity(filters.len());
    for filter in filters {
        let Some(index) = columns
            .iter()
            .position(|column| column.name == filter.column)
        else {
            return Err(backend(format!("no column `{}`", filter.column)));
        };
        if !predicate_matches_type(columns[index].ty, &filter.predicate) {
            return Err(backend(format!(
                "predicate type does not match column `{}`",
                filter.column
            )));
        }
        bound.push(BoundFilter {
            index,
            predicate: &filter.predicate,
        });
    }
    Ok(bound)
}

fn predicate_matches_type(ty: ColumnType, predicate: &ColumnPredicate) -> bool {
    match predicate {
        ColumnPredicate::Eq(value)
        | ColumnPredicate::Lt(value)
        | ColumnPredicate::Le(value)
        | ColumnPredicate::Gt(value)
        | ColumnPredicate::Ge(value) => value_type(value) == Some(ty),
        ColumnPredicate::Between { low, high } => {
            value_type(low) == Some(ty) && value_type(high) == Some(ty)
        }
    }
}

fn value_type(value: &ColumnValue) -> Option<ColumnType> {
    match value {
        ColumnValue::Null => None,
        ColumnValue::Int64(_) => Some(ColumnType::Int64),
        ColumnValue::Float64(_) => Some(ColumnType::Float64),
        ColumnValue::Bool(_) => Some(ColumnType::Bool),
        ColumnValue::Utf8(_) => Some(ColumnType::Utf8),
        ColumnValue::Bytes(_) => Some(ColumnType::Bytes),
        ColumnValue::TimestampMs(_) => Some(ColumnType::TimestampMs),
    }
}

/// The rows one encoded row group holds: its leading count, read without
/// decoding the group (for `ferrostore inspect`).
pub(super) fn group_row_count(blob: &[u8]) -> Result<u64, StoreError> {
    Ok(u64::from(Cur::new(blob).u32()?))
}

fn index_group<'a>(blob: &'a [u8], columns: &[ColumnDef]) -> Result<IndexedGroup<'a>, StoreError> {
    index_group_with(blob, columns, &Limits::DEFAULT)
}

fn index_group_with<'a>(
    blob: &'a [u8],
    columns: &[ColumnDef],
    limits: &Limits,
) -> Result<IndexedGroup<'a>, StoreError> {
    let mut cur = Cur::new(blob);
    let row_count_u32 = cur.u32()?;
    if row_count_u32 == 0 {
        return Err(corrupt("row group has no rows"));
    }
    let row_count =
        usize::try_from(row_count_u32).map_err(|_| corrupt("row count does not fit"))?;
    // Before any allocation sized by the claim.
    if row_count > limits.max_rows {
        return Err(corrupt(format!(
            "row group claims {row_count} rows; the limit is {}",
            limits.max_rows
        )));
    }
    let mut indexed = Vec::with_capacity(columns.len());
    for column in columns {
        let zone = read_zone(&mut cur, column.ty)?;
        if zone.null_count > u64::from(row_count_u32) {
            return Err(corrupt("null count exceeds the row count"));
        }
        let codec = cur.u8()?;
        let body_len =
            usize::try_from(cur.u32()?).map_err(|_| corrupt("chunk length does not fit"))?;
        let body = cur.take(body_len)?;
        indexed.push(IndexedColumn { zone, codec, body });
    }
    cur.finish()?;
    Ok(IndexedGroup {
        row_count,
        columns: indexed,
    })
}

fn decode_indexed(
    columns: &[ColumnDef],
    indexed: &IndexedGroup<'_>,
) -> Result<Vec<Vec<ColumnValue>>, StoreError> {
    decode_indexed_with(columns, indexed, &Limits::DEFAULT)
}

fn decode_indexed_with(
    columns: &[ColumnDef],
    indexed: &IndexedGroup<'_>,
    limits: &Limits,
) -> Result<Vec<Vec<ColumnValue>>, StoreError> {
    // One budget for the whole group: the columns are decoded one after another and every one
    // of them is charged against what the earlier ones left.
    let mut budget = Budget::new(limits.max_bytes);
    let rows_count = indexed.row_count;
    budget.charge(
        mul(
            rows_count,
            ROW_BYTES + columns.len() * std::mem::size_of::<ColumnValue>(),
            "row count",
        )?,
        "rows",
    )?;
    let mut decoded = Vec::with_capacity(columns.len());
    for (column, indexed_column) in columns.iter().zip(&indexed.columns) {
        let values = decode_column(
            column.ty,
            rows_count,
            &indexed_column.zone,
            indexed_column.codec,
            indexed_column.body,
            &mut budget,
        )?;
        let computed = zone_map(&values);
        if !same_zone(&indexed_column.zone, &computed) {
            return Err(corrupt("zone map does not match the column values"));
        }
        decoded.push(values);
    }
    let mut rows: Vec<Vec<ColumnValue>> = (0..rows_count)
        .map(|_| Vec::with_capacity(columns.len()))
        .collect();
    for column in decoded {
        for (row, value) in rows.iter_mut().zip(column) {
            row.push(value);
        }
    }
    Ok(rows)
}

fn decode_column(
    ty: ColumnType,
    row_count: usize,
    zone: &ZoneMap,
    codec: u8,
    body: &[u8],
    budget: &mut Budget,
) -> Result<Vec<ColumnValue>, StoreError> {
    let bit_len = row_count.div_ceil(8);
    if body.len() < bit_len {
        return Err(corrupt("null bitset is truncated"));
    }
    let (bits, payload) = body.split_at(bit_len);
    if !row_count.is_multiple_of(8) {
        let used = row_count % 8;
        let last = bits[bits.len() - 1];
        if last >> used != 0 {
            return Err(corrupt("null bitset has unused bits set"));
        }
    }
    let ones = bits
        .iter()
        .map(|byte| u64::from(byte.count_ones()))
        .sum::<u64>();
    if ones != zone.null_count {
        return Err(corrupt("null bitset does not match null_count"));
    }
    let nulls = usize::try_from(zone.null_count).map_err(|_| corrupt("null count does not fit"))?;
    if nulls > row_count {
        return Err(corrupt("null count exceeds the row count"));
    }
    let non_null_count = row_count - nulls;
    let non_null = decode_non_null(ty, codec, payload, non_null_count, budget)?;
    budget.charge(
        mul(row_count, std::mem::size_of::<ColumnValue>(), "row count")?,
        "column",
    )?;
    let mut out = Vec::with_capacity(row_count);
    // Values move out of `non_null`; a text or byte value is not copied a second time.
    let mut values = non_null.into_iter();
    for index in 0..row_count {
        let is_null = bits[index / 8] & (1 << (index % 8)) != 0;
        if is_null {
            out.push(ColumnValue::Null);
            continue;
        }
        let Some(value) = values.next() else {
            return Err(corrupt("column ran out of non-null values"));
        };
        out.push(value);
    }
    if values.next().is_some() {
        return Err(corrupt("column has unused non-null values"));
    }
    Ok(out)
}

/// Decode `n` non-null values. The decoders trust nothing the blob claims: `n` is charged against
/// the group's budget before the first allocation, and plain encodings, whose values each take a
/// known minimum of bytes, are also checked against the payload that carries them.
fn decode_non_null(
    ty: ColumnType,
    codec: u8,
    payload: &[u8],
    n: usize,
    budget: &mut Budget,
) -> Result<Vec<ColumnValue>, StoreError> {
    budget.charge(
        mul(n, std::mem::size_of::<ColumnValue>(), "value count")?,
        "values",
    )?;
    match (ty, codec) {
        (ColumnType::Int64 | ColumnType::TimestampMs, CODEC_DELTA) => decode_delta(ty, payload, n),
        (ColumnType::Int64 | ColumnType::TimestampMs, CODEC_DELTA_RLE) => {
            decode_delta_rle(ty, payload, n)
        }
        (ColumnType::Utf8, CODEC_PLAIN) => decode_plain_text(payload, n),
        (ColumnType::Utf8, CODEC_DICT) => decode_dict_text(payload, n, budget),
        (ColumnType::Utf8, CODEC_RLE) => decode_rle_text(payload, n, budget),
        (ColumnType::Bytes, CODEC_PLAIN) => decode_plain_bytes(payload, n),
        (ColumnType::Bytes, CODEC_RLE) => decode_rle_bytes(payload, n, budget),
        (ColumnType::Float64, CODEC_PLAIN) => decode_plain_f64(payload, n),
        (ColumnType::Float64, CODEC_RLE) => decode_rle_f64(payload, n),
        (ColumnType::Bool, CODEC_RLE) => decode_rle_bool(payload, n),
        _ => Err(corrupt(format!(
            "codec {codec} does not match the column type"
        ))),
    }
}

/// A plain encoding of `n` values takes at least `min_each` bytes per value, so `n` cannot be
/// larger than the payload allows; refuse before `Vec::with_capacity(n)`.
fn check_plain_count(payload: &[u8], n: usize, min_each: usize) -> Result<(), StoreError> {
    if mul(n, min_each, "value count")? > payload.len() {
        return Err(corrupt(format!(
            "{n} values need at least {min_each} bytes each but the payload has {}",
            payload.len()
        )));
    }
    Ok(())
}

fn decode_delta(ty: ColumnType, payload: &[u8], n: usize) -> Result<Vec<ColumnValue>, StoreError> {
    if n == 0 {
        return empty_payload(payload, "delta");
    }
    check_plain_count(payload, n, 8)?;
    let mut cur = Cur::new(payload);
    let mut prev = cur.i64()?;
    let mut out = Vec::with_capacity(n);
    out.push(int_value(ty, prev));
    for _ in 1..n {
        prev = prev.wrapping_add(zigzag_decode(cur.u64()?));
        out.push(int_value(ty, prev));
    }
    cur.finish()?;
    Ok(out)
}

fn decode_delta_rle(
    ty: ColumnType,
    payload: &[u8],
    n: usize,
) -> Result<Vec<ColumnValue>, StoreError> {
    if n == 0 {
        return empty_payload(payload, "delta_rle");
    }
    let mut cur = Cur::new(payload);
    let mut prev = cur.i64()?;
    // `n` is already charged to the budget; the vector holds the values directly, with no
    // intermediate `Vec<i64>`.
    let mut out = Vec::with_capacity(n);
    out.push(int_value(ty, prev));
    while out.len() < n {
        let delta = zigzag_decode(cur.u64()?);
        let len = usize::try_from(cur.u32()?).map_err(|_| corrupt("delta run does not fit"))?;
        if len == 0 || len > n - out.len() {
            return Err(corrupt("delta run length is outside the column"));
        }
        for _ in 0..len {
            prev = prev.wrapping_add(delta);
            out.push(int_value(ty, prev));
        }
    }
    cur.finish()?;
    Ok(out)
}

fn empty_payload(payload: &[u8], codec: &str) -> Result<Vec<ColumnValue>, StoreError> {
    if payload.is_empty() {
        Ok(Vec::new())
    } else {
        Err(corrupt(format!("{codec} payload is not empty")))
    }
}

fn decode_plain_text(payload: &[u8], n: usize) -> Result<Vec<ColumnValue>, StoreError> {
    check_plain_count(payload, n, 4)?;
    let mut cur = Cur::new(payload);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(ColumnValue::Utf8(cur.lp_str()?));
    }
    cur.finish()?;
    Ok(out)
}

fn decode_dict_text(
    payload: &[u8],
    n: usize,
    budget: &mut Budget,
) -> Result<Vec<ColumnValue>, StoreError> {
    let mut cur = Cur::new(payload);
    let dict_len =
        usize::try_from(cur.u32()?).map_err(|_| corrupt("dictionary count does not fit"))?;
    // Every entry is at least its four-byte length prefix.
    if mul(dict_len, 4, "dictionary count")? > cur.remaining() {
        return Err(corrupt("dictionary count exceeds the payload"));
    }
    let mut dict = Vec::with_capacity(dict_len);
    for _ in 0..dict_len {
        dict.push(cur.lp_str()?);
    }
    // The indexes are the rest of the payload, four bytes each; sum what they expand to before
    // allocating any of it.
    let index_bytes = cur.take(mul(n, 4, "value count")?)?;
    cur.finish()?;
    let mut expanded = 0usize;
    let mut indexes = Cur::new(index_bytes);
    for _ in 0..n {
        let index = usize::try_from(indexes.u32()?)
            .map_err(|_| corrupt("dictionary index does not fit"))?;
        let Some(text) = dict.get(index) else {
            return Err(corrupt("dictionary index is out of range"));
        };
        expanded = expanded.saturating_add(text.len());
    }
    budget.charge(expanded, "dictionary text")?;
    let mut out = Vec::with_capacity(n);
    let mut indexes = Cur::new(index_bytes);
    for _ in 0..n {
        // Every index was range-checked above.
        let index = usize::try_from(indexes.u32()?)
            .map_err(|_| corrupt("dictionary index does not fit"))?;
        out.push(ColumnValue::Utf8(dict[index].clone()));
    }
    Ok(out)
}

fn decode_rle_text(
    payload: &[u8],
    n: usize,
    budget: &mut Budget,
) -> Result<Vec<ColumnValue>, StoreError> {
    if n == 0 {
        return empty_payload(payload, "rle");
    }
    let mut cur = Cur::new(payload);
    let text = cur.lp_str()?;
    cur.finish()?;
    budget.charge(mul(text.len(), n, "run length")?, "repeated text")?;
    Ok(vec![ColumnValue::Utf8(text); n])
}

fn decode_plain_bytes(payload: &[u8], n: usize) -> Result<Vec<ColumnValue>, StoreError> {
    check_plain_count(payload, n, 4)?;
    let mut cur = Cur::new(payload);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(ColumnValue::Bytes(cur.lp()?.to_vec()));
    }
    cur.finish()?;
    Ok(out)
}

fn decode_rle_bytes(
    payload: &[u8],
    n: usize,
    budget: &mut Budget,
) -> Result<Vec<ColumnValue>, StoreError> {
    if n == 0 {
        return empty_payload(payload, "rle");
    }
    let mut cur = Cur::new(payload);
    let blob = cur.lp()?.to_vec();
    cur.finish()?;
    budget.charge(mul(blob.len(), n, "run length")?, "repeated bytes")?;
    Ok(vec![ColumnValue::Bytes(blob); n])
}

fn decode_plain_f64(payload: &[u8], n: usize) -> Result<Vec<ColumnValue>, StoreError> {
    check_plain_count(payload, n, 8)?;
    let mut cur = Cur::new(payload);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(ColumnValue::Float64(cur.f64()?));
    }
    cur.finish()?;
    Ok(out)
}

fn decode_rle_f64(payload: &[u8], n: usize) -> Result<Vec<ColumnValue>, StoreError> {
    if n == 0 {
        return empty_payload(payload, "rle");
    }
    let mut cur = Cur::new(payload);
    let number = cur.f64()?;
    cur.finish()?;
    Ok(vec![ColumnValue::Float64(number); n])
}

fn decode_rle_bool(payload: &[u8], n: usize) -> Result<Vec<ColumnValue>, StoreError> {
    let mut cur = Cur::new(payload);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let bit = cur.u8()?;
        if bit > 1 {
            return Err(corrupt("bool run is not 0 or 1"));
        }
        let len = usize::try_from(cur.u32()?).map_err(|_| corrupt("bool run does not fit"))?;
        if len == 0 || len > n - out.len() {
            return Err(corrupt("bool run length is outside the column"));
        }
        out.extend(std::iter::repeat_n(ColumnValue::Bool(bit == 1), len));
    }
    cur.finish()?;
    Ok(out)
}

fn read_zone(cur: &mut Cur<'_>, ty: ColumnType) -> Result<ZoneMap, StoreError> {
    let null_count = cur.u64()?;
    match cur.u8()? {
        0 => Ok(ZoneMap {
            null_count,
            min: None,
            max: None,
        }),
        1 => {
            let min = read_typed(cur, ty)?;
            let max = read_typed(cur, ty)?;
            if !is_le(&min, &max) {
                return Err(corrupt("zone map bounds are not ordered"));
            }
            Ok(ZoneMap {
                null_count,
                min: Some(min),
                max: Some(max),
            })
        }
        _ => Err(corrupt("zone map flag is not 0 or 1")),
    }
}

fn same_zone(stored: &ZoneMap, computed: &ZoneMap) -> bool {
    stored.null_count == computed.null_count
        && same_bound(&stored.min, &computed.min)
        && same_bound(&stored.max, &computed.max)
}

fn same_bound(left: &Option<ColumnValue>, right: &Option<ColumnValue>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) => same_value(left, right),
        _ => false,
    }
}

fn same_value(left: &ColumnValue, right: &ColumnValue) -> bool {
    match (left, right) {
        (ColumnValue::Float64(a), ColumnValue::Float64(b)) => a.to_bits() == b.to_bits(),
        _ => left == right,
    }
}

fn matches_predicate(value: &ColumnValue, predicate: &ColumnPredicate) -> bool {
    if matches!(value, ColumnValue::Null) {
        return false;
    }
    match predicate {
        ColumnPredicate::Eq(bound) => is_eq(value, bound),
        ColumnPredicate::Lt(bound) => is_lt(value, bound),
        ColumnPredicate::Le(bound) => is_le(value, bound),
        ColumnPredicate::Gt(bound) => is_gt(value, bound),
        ColumnPredicate::Ge(bound) => is_ge(value, bound),
        ColumnPredicate::Between { low, high } => is_ge(value, low) && is_le(value, high),
    }
}

fn rel(left: &ColumnValue, right: &ColumnValue) -> Option<Ordering> {
    match (left, right) {
        (ColumnValue::Int64(a), ColumnValue::Int64(b)) => Some(a.cmp(b)),
        (ColumnValue::TimestampMs(a), ColumnValue::TimestampMs(b)) => Some(a.cmp(b)),
        (ColumnValue::Float64(a), ColumnValue::Float64(b)) => a.partial_cmp(b),
        (ColumnValue::Bool(a), ColumnValue::Bool(b)) => Some(a.cmp(b)),
        (ColumnValue::Utf8(a), ColumnValue::Utf8(b)) => Some(a.as_bytes().cmp(b.as_bytes())),
        (ColumnValue::Bytes(a), ColumnValue::Bytes(b)) => Some(a.cmp(b)),
        _ => None,
    }
}

fn is_lt(left: &ColumnValue, right: &ColumnValue) -> bool {
    matches!(rel(left, right), Some(Ordering::Less))
}

fn is_gt(left: &ColumnValue, right: &ColumnValue) -> bool {
    matches!(rel(left, right), Some(Ordering::Greater))
}

fn is_le(left: &ColumnValue, right: &ColumnValue) -> bool {
    matches!(rel(left, right), Some(Ordering::Less | Ordering::Equal))
}

fn is_ge(left: &ColumnValue, right: &ColumnValue) -> bool {
    matches!(rel(left, right), Some(Ordering::Greater | Ordering::Equal))
}

fn is_eq(left: &ColumnValue, right: &ColumnValue) -> bool {
    matches!(rel(left, right), Some(Ordering::Equal))
}

fn int_deltas(values: &[i64]) -> Vec<i64> {
    let mut out = Vec::new();
    let Some(&first) = values.first() else {
        return out;
    };
    let mut prev = first;
    for &next in &values[1..] {
        out.push(next.wrapping_sub(prev));
        prev = next;
    }
    out
}

fn has_run_at_least(deltas: &[i64], min: usize) -> bool {
    let mut run = 0usize;
    let mut prev: Option<i64> = None;
    for delta in deltas {
        if Some(*delta) == prev {
            run += 1;
        } else {
            prev = Some(*delta);
            run = 1;
        }
        if run >= min {
            return true;
        }
    }
    false
}

fn zigzag_encode(value: i64) -> u64 {
    ((value as u64) << 1) ^ ((value >> 63) as u64)
}

fn zigzag_decode(value: u64) -> i64 {
    ((value >> 1) as i64) ^ -((value & 1) as i64)
}

fn int_value(ty: ColumnType, value: i64) -> ColumnValue {
    match ty {
        ColumnType::TimestampMs => ColumnValue::TimestampMs(value),
        _ => ColumnValue::Int64(value),
    }
}

fn collect_i64(ty: ColumnType, values: &[ColumnValue]) -> Result<Vec<i64>, StoreError> {
    values
        .iter()
        .map(|value| match (ty, value) {
            (ColumnType::Int64, ColumnValue::Int64(number)) => Ok(*number),
            (ColumnType::TimestampMs, ColumnValue::TimestampMs(number)) => Ok(*number),
            _ => Err(backend("integer column has a non-integer value")),
        })
        .collect()
}

fn collect_f64(values: &[ColumnValue]) -> Result<Vec<f64>, StoreError> {
    values
        .iter()
        .map(|value| match value {
            ColumnValue::Float64(number) => Ok(*number),
            _ => Err(backend("float column has a non-float value")),
        })
        .collect()
}

fn collect_bool(values: &[ColumnValue]) -> Result<Vec<bool>, StoreError> {
    values
        .iter()
        .map(|value| match value {
            ColumnValue::Bool(bit) => Ok(*bit),
            _ => Err(backend("bool column has a non-bool value")),
        })
        .collect()
}

fn collect_utf8(values: &[ColumnValue]) -> Result<Vec<String>, StoreError> {
    values
        .iter()
        .map(|value| match value {
            ColumnValue::Utf8(text) => Ok(text.clone()),
            _ => Err(backend("utf8 column has a non-string value")),
        })
        .collect()
}

fn collect_bytes(values: &[ColumnValue]) -> Result<Vec<Vec<u8>>, StoreError> {
    values
        .iter()
        .map(|value| match value {
            ColumnValue::Bytes(blob) => Ok(blob.clone()),
            _ => Err(backend("bytes column has a non-bytes value")),
        })
        .collect()
}

fn all_same_bits(values: &[f64]) -> bool {
    let Some(first) = values.first() else {
        return true;
    };
    let bits = first.to_bits();
    values.iter().all(|value| value.to_bits() == bits)
}

fn write_typed(buf: &mut Vec<u8>, ty: ColumnType, value: &ColumnValue) -> Result<(), StoreError> {
    match (ty, value) {
        (ColumnType::Int64, ColumnValue::Int64(number))
        | (ColumnType::TimestampMs, ColumnValue::TimestampMs(number)) => write_i64(buf, *number),
        (ColumnType::Float64, ColumnValue::Float64(number)) => {
            buf.extend_from_slice(&number.to_le_bytes());
        }
        (ColumnType::Bool, ColumnValue::Bool(bit)) => buf.push(u8::from(*bit)),
        (ColumnType::Utf8, ColumnValue::Utf8(text)) => write_lp(buf, text.as_bytes())?,
        (ColumnType::Bytes, ColumnValue::Bytes(blob)) => write_lp(buf, blob)?,
        _ => return Err(backend("zone map bound does not match the column type")),
    }
    Ok(())
}

fn read_typed(cur: &mut Cur<'_>, ty: ColumnType) -> Result<ColumnValue, StoreError> {
    match ty {
        ColumnType::Int64 => Ok(ColumnValue::Int64(cur.i64()?)),
        ColumnType::TimestampMs => Ok(ColumnValue::TimestampMs(cur.i64()?)),
        ColumnType::Float64 => Ok(ColumnValue::Float64(cur.f64()?)),
        ColumnType::Bool => match cur.u8()? {
            0 => Ok(ColumnValue::Bool(false)),
            1 => Ok(ColumnValue::Bool(true)),
            _ => Err(corrupt("bool bound is not 0 or 1")),
        },
        ColumnType::Utf8 => Ok(ColumnValue::Utf8(cur.lp_str()?)),
        ColumnType::Bytes => Ok(ColumnValue::Bytes(cur.lp()?.to_vec())),
    }
}

fn write_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn write_u64(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn write_i64(buf: &mut Vec<u8>, value: i64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn write_lp(buf: &mut Vec<u8>, bytes: &[u8]) -> Result<(), StoreError> {
    let len = u32::try_from(bytes.len()).map_err(|_| backend("length does not fit in u32"))?;
    write_u32(buf, len);
    buf.extend_from_slice(bytes);
    Ok(())
}

struct Cur<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Cur<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, at: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.at)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], StoreError> {
        let end = self
            .at
            .checked_add(n)
            .ok_or_else(|| corrupt("length overflow"))?;
        if end > self.buf.len() {
            return Err(corrupt("truncated record"));
        }
        let bytes = &self.buf[self.at..end];
        self.at = end;
        Ok(bytes)
    }

    fn u8(&mut self) -> Result<u8, StoreError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, StoreError> {
        let bytes = self.take(4)?;
        let mut arr = [0u8; 4];
        arr.copy_from_slice(bytes);
        Ok(u32::from_le_bytes(arr))
    }

    fn u64(&mut self) -> Result<u64, StoreError> {
        let bytes = self.take(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(arr))
    }

    fn i64(&mut self) -> Result<i64, StoreError> {
        Ok(i64::from_le_bytes(self.u64()?.to_le_bytes()))
    }

    fn f64(&mut self) -> Result<f64, StoreError> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn lp(&mut self) -> Result<&'a [u8], StoreError> {
        let n = usize::try_from(self.u32()?).map_err(|_| corrupt("length does not fit"))?;
        if n > self.remaining() {
            return Err(corrupt("length prefix exceeds the record"));
        }
        self.take(n)
    }

    fn lp_str(&mut self) -> Result<String, StoreError> {
        let bytes = self.lp()?;
        String::from_utf8(bytes.to_vec()).map_err(|_| corrupt("text is not utf-8"))
    }

    fn finish(self) -> Result<(), StoreError> {
        if self.at != self.buf.len() {
            return Err(corrupt("trailing bytes inside a committed record"));
        }
        Ok(())
    }
}

pub(super) fn attempt_rollup(
    txn: &Txn,
    project: Option<&ProjectId>,
) -> Result<AttemptRollup, StoreError> {
    let attempts = doc::scan_u64::<Attempt>(txn, ATTEMPTS)?;
    let columns = attempt_columns();
    let mut rows = Vec::with_capacity(attempts.len());
    for (_, attempt) in attempts {
        rows.push(vec![
            ColumnValue::Utf8(attempt.project.as_str().to_string()),
            ColumnValue::Int64(u_to_i(attempt.cost_usd_micros)?),
            ColumnValue::Int64(u_to_i(attempt.tokens_in)?),
            ColumnValue::Int64(u_to_i(attempt.tokens_out)?),
            ColumnValue::Int64(u_to_i(attempt.duration_ms)?),
        ]);
    }
    let mut filters = Vec::new();
    if let Some(project) = project {
        filters.push(ColumnFilter {
            column: "project".to_string(),
            predicate: ColumnPredicate::Eq(ColumnValue::Utf8(project.as_str().to_string())),
        });
    }
    let scan = scan_projected(&columns, &rows, &filters)?;
    let mut rollup = AttemptRollup::default();
    for row in scan.rows {
        rollup.count = rollup
            .count
            .checked_add(1)
            .ok_or_else(|| backend("rollup overflow"))?;
        add_cell(&mut rollup.cost_usd_micros, &row[1])?;
        add_cell(&mut rollup.tokens_in, &row[2])?;
        add_cell(&mut rollup.tokens_out, &row[3])?;
        add_cell(&mut rollup.duration_ms, &row[4])?;
    }
    Ok(rollup)
}

pub(super) fn gate_run_rollup(
    txn: &Txn,
    gate: Option<&GateId>,
) -> Result<GateRunRollup, StoreError> {
    let runs = doc::scan_u64::<GateRun>(txn, GATE_RUNS)?;
    let columns = gate_columns();
    let mut rows = Vec::with_capacity(runs.len());
    for (_, run) in runs {
        rows.push(vec![
            ColumnValue::Utf8(run.gate.as_str().to_string()),
            ColumnValue::Utf8(verdict_name(&run.verdict).to_string()),
            ColumnValue::Int64(u_to_i(run.duration_ms)?),
            ColumnValue::Int64(u_to_i(run.cost_usd_micros)?),
        ]);
    }
    let mut filters = Vec::new();
    if let Some(gate) = gate {
        filters.push(ColumnFilter {
            column: "gate".to_string(),
            predicate: ColumnPredicate::Eq(ColumnValue::Utf8(gate.as_str().to_string())),
        });
    }
    let scan = scan_projected(&columns, &rows, &filters)?;
    let mut rollup = GateRunRollup::default();
    for row in scan.rows {
        match &row[1] {
            ColumnValue::Utf8(name) if name == "pass" => {
                rollup.pass = rollup
                    .pass
                    .checked_add(1)
                    .ok_or_else(|| backend("rollup overflow"))?;
            }
            ColumnValue::Utf8(name) if name == "fail" => {
                rollup.fail = rollup
                    .fail
                    .checked_add(1)
                    .ok_or_else(|| backend("rollup overflow"))?;
            }
            ColumnValue::Utf8(name) if name == "error" => {
                rollup.error = rollup
                    .error
                    .checked_add(1)
                    .ok_or_else(|| backend("rollup overflow"))?;
            }
            _ => return Err(backend("gate verdict column is not pass, fail, or error")),
        }
        add_cell(&mut rollup.duration_ms, &row[2])?;
        add_cell(&mut rollup.cost_usd_micros, &row[3])?;
    }
    Ok(rollup)
}

/// Rows per internal chunk of an in-memory scan (see [`scan_projected`]). Bounds the memory one
/// chunk's encoded blob and decoded rows take at once; it is not a limit on the ledger.
const SCAN_CHUNK_ROWS: usize = 1 << 18;

/// Scan rows the caller already holds in memory (the ledger rollups). The rows are encoded and
/// scanned in chunks, so the writer limits (`MAX_ROWS_PER_GROUP`, `MAX_DECODE_BYTES`) do not
/// bound how many rows a rollup can cover: only a stored table's group is bound by them. Chunks
/// run in row order and their matches are appended in that order, so the rows returned are
/// exactly what one big group would return; only the group counters count chunks.
fn scan_projected(
    columns: &[ColumnDef],
    rows: &[Vec<ColumnValue>],
    filters: &[ColumnFilter],
) -> Result<ColumnScan, StoreError> {
    scan_projected_with(
        columns,
        rows,
        filters,
        &Limits {
            max_rows: SCAN_CHUNK_ROWS.min(MAX_ROWS_PER_GROUP),
            max_bytes: MAX_DECODE_BYTES,
        },
    )
}

/// `scan_projected` with the chunk bounds a test can lower: a chunk ends at `max_rows` rows or
/// when the next row would take its estimated decode size over `max_bytes`.
fn scan_projected_with(
    columns: &[ColumnDef],
    rows: &[Vec<ColumnValue>],
    filters: &[ColumnFilter],
    chunk: &Limits,
) -> Result<ColumnScan, StoreError> {
    let mut merged = ColumnScan {
        rows: Vec::new(),
        row_groups_total: 0,
        row_groups_skipped: 0,
    };
    let mut start = 0usize;
    while start < rows.len() {
        let mut end = start;
        let mut bytes = 0usize;
        while end < rows.len() && end - start < chunk.max_rows {
            let cost = estimated_decode_bytes(columns, &rows[end..=end]);
            // A chunk always takes its first row; a single row over the budget is refused by the
            // encoder below, as it always was.
            if end > start && bytes.saturating_add(cost) > chunk.max_bytes {
                break;
            }
            bytes = bytes.saturating_add(cost);
            end += 1;
        }
        let blob = encode_row_group_with(columns, &rows[start..end], chunk)?;
        let scan = scan_groups(columns, &[blob], filters)?;
        merged.rows.extend(scan.rows);
        merged.row_groups_total = merged
            .row_groups_total
            .saturating_add(scan.row_groups_total);
        merged.row_groups_skipped = merged
            .row_groups_skipped
            .saturating_add(scan.row_groups_skipped);
        start = end;
    }
    Ok(merged)
}

fn attempt_columns() -> Vec<ColumnDef> {
    vec![
        ColumnDef {
            name: "project".to_string(),
            ty: ColumnType::Utf8,
        },
        ColumnDef {
            name: "cost_usd_micros".to_string(),
            ty: ColumnType::Int64,
        },
        ColumnDef {
            name: "tokens_in".to_string(),
            ty: ColumnType::Int64,
        },
        ColumnDef {
            name: "tokens_out".to_string(),
            ty: ColumnType::Int64,
        },
        ColumnDef {
            name: "duration_ms".to_string(),
            ty: ColumnType::Int64,
        },
    ]
}

fn gate_columns() -> Vec<ColumnDef> {
    vec![
        ColumnDef {
            name: "gate".to_string(),
            ty: ColumnType::Utf8,
        },
        ColumnDef {
            name: "verdict".to_string(),
            ty: ColumnType::Utf8,
        },
        ColumnDef {
            name: "duration_ms".to_string(),
            ty: ColumnType::Int64,
        },
        ColumnDef {
            name: "cost_usd_micros".to_string(),
            ty: ColumnType::Int64,
        },
    ]
}

fn verdict_name(verdict: &Verdict) -> &'static str {
    match verdict {
        Verdict::Pass { .. } => "pass",
        Verdict::Fail { .. } => "fail",
        Verdict::Error { .. } => "error",
    }
}

fn u_to_i(value: u64) -> Result<i64, StoreError> {
    i64::try_from(value).map_err(|_| backend("ledger value does not fit in i64"))
}

fn add_cell(slot: &mut u64, value: &ColumnValue) -> Result<(), StoreError> {
    let ColumnValue::Int64(number) = value else {
        return Err(backend("rollup column is not an int64"));
    };
    let number = u64::try_from(*number).map_err(|_| backend("rollup value is negative"))?;
    *slot = slot
        .checked_add(number)
        .ok_or_else(|| backend("rollup overflow"))?;
    Ok(())
}

pub(super) fn type_name(ty: ColumnType) -> &'static str {
    match ty {
        ColumnType::Int64 => "int64",
        ColumnType::Float64 => "float64",
        ColumnType::Bool => "bool",
        ColumnType::Utf8 => "utf8",
        ColumnType::Bytes => "bytes",
        ColumnType::TimestampMs => "timestamp_ms",
    }
}

pub(super) fn parse_type_name(name: &str) -> Result<ColumnType, StoreError> {
    match name {
        "int64" => Ok(ColumnType::Int64),
        "float64" => Ok(ColumnType::Float64),
        "bool" => Ok(ColumnType::Bool),
        "utf8" => Ok(ColumnType::Utf8),
        "bytes" => Ok(ColumnType::Bytes),
        "timestamp_ms" => Ok(ColumnType::TimestampMs),
        other => Err(backend(format!("unknown column type `{other}`"))),
    }
}

pub(super) fn column_def_json(column: &ColumnDef) -> Value {
    json!({
        "name": column.name,
        "ty": type_name(column.ty),
    })
}

pub(super) fn parse_columns(value: &Value) -> Result<Vec<ColumnDef>, StoreError> {
    let items = value
        .as_array()
        .ok_or_else(|| backend("columns is not an array"))?;
    items.iter().map(parse_column_def).collect()
}

fn parse_column_def(value: &Value) -> Result<ColumnDef, StoreError> {
    let name = value
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| backend("column is missing name"))?
        .to_string();
    let ty = value
        .get("ty")
        .and_then(Value::as_str)
        .ok_or_else(|| backend("column is missing ty"))?;
    Ok(ColumnDef {
        name,
        ty: parse_type_name(ty)?,
    })
}

pub(super) fn cell_json(cell: &ColumnValue) -> Result<Value, StoreError> {
    Ok(match cell {
        ColumnValue::Null => json!({"t": "null"}),
        ColumnValue::Int64(value) => json!({"t": "i64", "v": value}),
        ColumnValue::Float64(value) => {
            if value.is_finite() {
                let number = Number::from_f64(*value)
                    .ok_or_else(|| backend("f64 cell is not a finite json number"))?;
                json!({"t": "f64", "v": number})
            } else {
                json!({
                    "t": "f64",
                    "v": non_finite_f64_label(*value),
                    "bits": value.to_bits(),
                })
            }
        }
        ColumnValue::Bool(value) => json!({"t": "bool", "v": value}),
        ColumnValue::Utf8(value) => json!({"t": "utf8", "v": value}),
        ColumnValue::Bytes(value) => json!({"t": "bytes", "v": value}),
        ColumnValue::TimestampMs(value) => json!({"t": "ts", "v": value}),
    })
}

pub(super) fn parse_cell(value: &Value) -> Result<ColumnValue, StoreError> {
    let kind = value
        .get("t")
        .and_then(Value::as_str)
        .ok_or_else(|| backend("cell is missing t"))?;
    if kind == "null" {
        return Ok(ColumnValue::Null);
    }
    let raw = value.get("v").ok_or_else(|| backend("cell is missing v"))?;
    match kind {
        "i64" => Ok(ColumnValue::Int64(json_i64(raw)?)),
        "f64" => Ok(ColumnValue::Float64(parse_f64_cell(value)?)),
        "bool" => Ok(ColumnValue::Bool(
            raw.as_bool()
                .ok_or_else(|| backend("bool cell is not a bool"))?,
        )),
        "utf8" => Ok(ColumnValue::Utf8(
            raw.as_str()
                .ok_or_else(|| backend("utf8 cell is not a string"))?
                .to_string(),
        )),
        "bytes" => Ok(ColumnValue::Bytes(parse_bytes(raw)?)),
        "ts" => Ok(ColumnValue::TimestampMs(json_i64(raw)?)),
        other => Err(backend(format!("unknown cell type `{other}`"))),
    }
}

fn non_finite_f64_label(value: f64) -> &'static str {
    if value.is_nan() {
        "NaN"
    } else if value.is_sign_negative() {
        "-Infinity"
    } else {
        "Infinity"
    }
}

fn f64_label_matches(label: &str, value: f64) -> bool {
    match label {
        "NaN" => value.is_nan(),
        "Infinity" => value.is_infinite() && value.is_sign_positive(),
        "-Infinity" => value.is_infinite() && value.is_sign_negative(),
        _ => false,
    }
}

fn parse_f64_cell(cell: &Value) -> Result<f64, StoreError> {
    let raw = cell.get("v").ok_or_else(|| backend("cell is missing v"))?;
    if let Some(label) = raw.as_str() {
        if !matches!(label, "NaN" | "Infinity" | "-Infinity") {
            return Err(backend("finite f64 cell must not use a string"));
        }
        let Some(bits) = cell.get("bits") else {
            return Ok(match label {
                "NaN" => f64::NAN,
                "Infinity" => f64::INFINITY,
                _ => f64::NEG_INFINITY,
            });
        };
        let bits = bits
            .as_u64()
            .ok_or_else(|| backend("f64 bits are not an integer in 0..=u64::MAX"))?;
        let value = f64::from_bits(bits);
        if !f64_label_matches(label, value) {
            return Err(backend("f64 bits do not match v"));
        }
        return Ok(value);
    }
    if cell.get("bits").is_some() {
        return Err(backend("finite f64 cell must not include bits"));
    }
    raw.as_f64()
        .ok_or_else(|| backend("f64 cell is not a number"))
}

fn json_i64(value: &Value) -> Result<i64, StoreError> {
    value.as_i64().ok_or_else(|| backend("cell is not an i64"))
}

pub(super) fn parse_rows(value: &Value) -> Result<Vec<Vec<ColumnValue>>, StoreError> {
    let items = value
        .as_array()
        .ok_or_else(|| backend("rows is not an array"))?;
    let mut rows = Vec::with_capacity(items.len());
    for item in items {
        let cells = item
            .as_array()
            .ok_or_else(|| backend("row is not an array"))?;
        let mut row = Vec::with_capacity(cells.len());
        for cell in cells {
            row.push(parse_cell(cell)?);
        }
        rows.push(row);
    }
    Ok(rows)
}

pub(super) fn filter_json(filter: &ColumnFilter) -> Result<Value, StoreError> {
    let column = &filter.column;
    Ok(match &filter.predicate {
        ColumnPredicate::Eq(value) => {
            json!({"column": column, "op": "eq", "value": cell_json(value)?})
        }
        ColumnPredicate::Lt(value) => {
            json!({"column": column, "op": "lt", "value": cell_json(value)?})
        }
        ColumnPredicate::Le(value) => {
            json!({"column": column, "op": "le", "value": cell_json(value)?})
        }
        ColumnPredicate::Gt(value) => {
            json!({"column": column, "op": "gt", "value": cell_json(value)?})
        }
        ColumnPredicate::Ge(value) => {
            json!({"column": column, "op": "ge", "value": cell_json(value)?})
        }
        ColumnPredicate::Between { low, high } => json!({
            "column": column,
            "op": "between",
            "low": cell_json(low)?,
            "high": cell_json(high)?,
        }),
    })
}

pub(super) fn parse_filters(value: &Value) -> Result<Vec<ColumnFilter>, StoreError> {
    let items = value
        .as_array()
        .ok_or_else(|| backend("filters is not an array"))?;
    items.iter().map(parse_filter).collect()
}

fn parse_filter(value: &Value) -> Result<ColumnFilter, StoreError> {
    let column = value
        .get("column")
        .and_then(Value::as_str)
        .ok_or_else(|| backend("filter is missing column"))?
        .to_string();
    let op = value
        .get("op")
        .and_then(Value::as_str)
        .ok_or_else(|| backend("filter is missing op"))?;
    let predicate = match op {
        "eq" => ColumnPredicate::Eq(parse_cell(required(value, "value")?)?),
        "lt" => ColumnPredicate::Lt(parse_cell(required(value, "value")?)?),
        "le" => ColumnPredicate::Le(parse_cell(required(value, "value")?)?),
        "gt" => ColumnPredicate::Gt(parse_cell(required(value, "value")?)?),
        "ge" => ColumnPredicate::Ge(parse_cell(required(value, "value")?)?),
        "between" => ColumnPredicate::Between {
            low: parse_cell(required(value, "low")?)?,
            high: parse_cell(required(value, "high")?)?,
        },
        other => return Err(backend(format!("unknown filter op `{other}`"))),
    };
    Ok(ColumnFilter { column, predicate })
}

fn required<'a>(value: &'a Value, key: &str) -> Result<&'a Value, StoreError> {
    value
        .get(key)
        .ok_or_else(|| backend(format!("filter is missing {key}")))
}

pub(super) fn scan_json(scan: &ColumnScan) -> Result<Value, StoreError> {
    let mut rows = Vec::with_capacity(scan.rows.len());
    for row in &scan.rows {
        let mut cells = Vec::with_capacity(row.len());
        for cell in row {
            cells.push(cell_json(cell)?);
        }
        rows.push(Value::Array(cells));
    }
    Ok(json!({
        "rows": rows,
        "row_groups_total": scan.row_groups_total,
        "row_groups_skipped": scan.row_groups_skipped,
    }))
}

pub(super) fn parse_scan(value: &Value) -> Result<ColumnScan, StoreError> {
    let rows = parse_rows(
        value
            .get("rows")
            .ok_or_else(|| backend("scan is missing rows"))?,
    )?;
    Ok(ColumnScan {
        rows,
        row_groups_total: json_u64(value, "row_groups_total")?,
        row_groups_skipped: json_u64(value, "row_groups_skipped")?,
    })
}

pub(super) fn attempt_rollup_json(rollup: &AttemptRollup) -> Value {
    json!({
        "count": rollup.count,
        "cost_usd_micros": rollup.cost_usd_micros,
        "tokens_in": rollup.tokens_in,
        "tokens_out": rollup.tokens_out,
        "duration_ms": rollup.duration_ms,
    })
}

pub(super) fn parse_attempt_rollup(value: &Value) -> Result<AttemptRollup, StoreError> {
    Ok(AttemptRollup {
        count: json_u64(value, "count")?,
        cost_usd_micros: json_u64(value, "cost_usd_micros")?,
        tokens_in: json_u64(value, "tokens_in")?,
        tokens_out: json_u64(value, "tokens_out")?,
        duration_ms: json_u64(value, "duration_ms")?,
    })
}

pub(super) fn gate_rollup_json(rollup: &GateRunRollup) -> Value {
    json!({
        "pass": rollup.pass,
        "fail": rollup.fail,
        "error": rollup.error,
        "duration_ms": rollup.duration_ms,
        "cost_usd_micros": rollup.cost_usd_micros,
    })
}

pub(super) fn parse_gate_rollup(value: &Value) -> Result<GateRunRollup, StoreError> {
    Ok(GateRunRollup {
        pass: json_u64(value, "pass")?,
        fail: json_u64(value, "fail")?,
        error: json_u64(value, "error")?,
        duration_ms: json_u64(value, "duration_ms")?,
        cost_usd_micros: json_u64(value, "cost_usd_micros")?,
    })
}

fn json_u64(value: &Value, key: &str) -> Result<u64, StoreError> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| backend(format!("missing integer field `{key}`")))
}

pub(super) fn parse_bytes(value: &Value) -> Result<Vec<u8>, StoreError> {
    let items = value
        .as_array()
        .ok_or_else(|| backend("bytes are not an array"))?;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let number = item
            .as_u64()
            .ok_or_else(|| backend("byte is not an integer"))?;
        let byte = u8::try_from(number).map_err(|_| backend("byte is outside 0..=255"))?;
        out.push(byte);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column_codecs(blob: &[u8], columns: &[ColumnDef]) -> Result<Vec<u8>, StoreError> {
        Ok(index_group(blob, columns)?
            .columns
            .into_iter()
            .map(|column| column.codec)
            .collect())
    }

    fn int_col() -> Vec<ColumnDef> {
        vec![ColumnDef {
            name: "n".to_string(),
            ty: ColumnType::Int64,
        }]
    }

    fn round_trip(columns: &[ColumnDef], rows: &[Vec<ColumnValue>]) -> Vec<u8> {
        let blob = encode_row_group(columns, rows).unwrap();
        let scan = scan_groups(columns, std::slice::from_ref(&blob), &[]).unwrap();
        assert_eq!(scan.rows, rows);
        assert_eq!(scan.row_groups_total, 1);
        assert_eq!(scan.row_groups_skipped, 0);
        blob
    }

    #[test]
    fn int_delta_and_delta_rle_round_trip() {
        let columns = int_col();
        let varied = vec![
            vec![ColumnValue::Int64(1)],
            vec![ColumnValue::Int64(3)],
            vec![ColumnValue::Int64(7)],
        ];
        let blob = round_trip(&columns, &varied);
        assert_eq!(column_codecs(&blob, &columns).unwrap(), vec![CODEC_DELTA]);

        let identical = vec![
            vec![ColumnValue::Int64(5)],
            vec![ColumnValue::Int64(5)],
            vec![ColumnValue::Int64(5)],
            vec![ColumnValue::Int64(5)],
        ];
        let blob = round_trip(&columns, &identical);
        assert_eq!(
            column_codecs(&blob, &columns).unwrap(),
            vec![CODEC_DELTA_RLE]
        );

        let stepped = vec![
            vec![ColumnValue::Int64(0)],
            vec![ColumnValue::Int64(10)],
            vec![ColumnValue::Int64(20)],
            vec![ColumnValue::Int64(30)],
        ];
        let blob = round_trip(&columns, &stepped);
        assert_eq!(
            column_codecs(&blob, &columns).unwrap(),
            vec![CODEC_DELTA_RLE]
        );

        let with_nulls = vec![
            vec![ColumnValue::Int64(i64::MIN)],
            vec![ColumnValue::Null],
            vec![ColumnValue::Int64(i64::MAX)],
            vec![ColumnValue::Int64(i64::MAX)],
        ];
        round_trip(&columns, &with_nulls);

        let ts = vec![ColumnDef {
            name: "t".to_string(),
            ty: ColumnType::TimestampMs,
        }];
        round_trip(
            &ts,
            &[
                vec![ColumnValue::TimestampMs(10)],
                vec![ColumnValue::TimestampMs(11)],
                vec![ColumnValue::TimestampMs(12)],
                vec![ColumnValue::TimestampMs(13)],
            ],
        );
    }

    #[test]
    fn utf8_dictionary_and_plain_round_trip() {
        let columns = vec![ColumnDef {
            name: "s".to_string(),
            ty: ColumnType::Utf8,
        }];
        let repeated = ["a", "b", "a", "b"]
            .into_iter()
            .map(|text| vec![ColumnValue::Utf8(text.to_string())])
            .collect::<Vec<_>>();
        let blob = round_trip(&columns, &repeated);
        assert_eq!(column_codecs(&blob, &columns).unwrap(), vec![CODEC_DICT]);

        let unique = ["a", "b", "c", "d"]
            .into_iter()
            .map(|text| vec![ColumnValue::Utf8(text.to_string())])
            .collect::<Vec<_>>();
        let blob = round_trip(&columns, &unique);
        assert_eq!(column_codecs(&blob, &columns).unwrap(), vec![CODEC_PLAIN]);

        let same = vec![
            vec![ColumnValue::Utf8("z".to_string())],
            vec![ColumnValue::Utf8("z".to_string())],
            vec![ColumnValue::Utf8("z".to_string())],
        ];
        let blob = round_trip(&columns, &same);
        assert_eq!(column_codecs(&blob, &columns).unwrap(), vec![CODEC_RLE]);
    }

    #[test]
    fn other_codecs_round_trip_and_nan_is_unordered() {
        let blobs = vec![ColumnDef {
            name: "b".to_string(),
            ty: ColumnType::Bytes,
        }];
        let same = vec![
            vec![ColumnValue::Bytes(b"xy".to_vec())],
            vec![ColumnValue::Bytes(b"xy".to_vec())],
            vec![ColumnValue::Bytes(b"xy".to_vec())],
        ];
        let blob = round_trip(&blobs, &same);
        assert_eq!(column_codecs(&blob, &blobs).unwrap(), vec![CODEC_RLE]);
        round_trip(
            &blobs,
            &[
                vec![ColumnValue::Bytes(b"a".to_vec())],
                vec![ColumnValue::Bytes(b"b".to_vec())],
            ],
        );

        let bits = vec![ColumnDef {
            name: "f".to_string(),
            ty: ColumnType::Float64,
        }];
        let blob = round_trip(
            &bits,
            &[
                vec![ColumnValue::Float64(1.5)],
                vec![ColumnValue::Float64(1.5)],
                vec![ColumnValue::Float64(1.5)],
            ],
        );
        assert_eq!(column_codecs(&blob, &bits).unwrap(), vec![CODEC_RLE]);

        let flags = vec![ColumnDef {
            name: "p".to_string(),
            ty: ColumnType::Bool,
        }];
        let blob = round_trip(
            &flags,
            &[
                vec![ColumnValue::Bool(true)],
                vec![ColumnValue::Bool(true)],
                vec![ColumnValue::Bool(false)],
            ],
        );
        assert_eq!(column_codecs(&blob, &flags).unwrap(), vec![CODEC_RLE]);

        let nan_rows = vec![
            vec![ColumnValue::Float64(f64::NAN)],
            vec![ColumnValue::Float64(f64::NAN)],
        ];
        let blob = encode_row_group(&bits, &nan_rows).unwrap();
        let scan = scan_groups(
            &bits,
            &[blob],
            &[ColumnFilter {
                column: "f".to_string(),
                predicate: ColumnPredicate::Eq(ColumnValue::Float64(1.0)),
            }],
        )
        .unwrap();
        assert_eq!(scan.row_groups_skipped, 1);
        assert!(scan.rows.is_empty());
    }

    #[test]
    fn zone_map_excludes_eq_lt_gt_between_and_all_null() {
        let zone = ZoneMap {
            null_count: 0,
            min: Some(ColumnValue::Int64(1)),
            max: Some(ColumnValue::Int64(10)),
        };
        assert!(!zone.excludes(&ColumnPredicate::Eq(ColumnValue::Int64(1))));
        assert!(!zone.excludes(&ColumnPredicate::Eq(ColumnValue::Int64(10))));
        assert!(zone.excludes(&ColumnPredicate::Eq(ColumnValue::Int64(0))));
        assert!(zone.excludes(&ColumnPredicate::Eq(ColumnValue::Int64(11))));
        assert!(zone.excludes(&ColumnPredicate::Lt(ColumnValue::Int64(1))));
        assert!(!zone.excludes(&ColumnPredicate::Lt(ColumnValue::Int64(2))));
        assert!(zone.excludes(&ColumnPredicate::Gt(ColumnValue::Int64(10))));
        assert!(!zone.excludes(&ColumnPredicate::Gt(ColumnValue::Int64(9))));
        assert!(zone.excludes(&ColumnPredicate::Between {
            low: ColumnValue::Int64(11),
            high: ColumnValue::Int64(12),
        }));
        assert!(zone.excludes(&ColumnPredicate::Between {
            low: ColumnValue::Int64(-5),
            high: ColumnValue::Int64(0),
        }));
        assert!(!zone.excludes(&ColumnPredicate::Between {
            low: ColumnValue::Int64(10),
            high: ColumnValue::Int64(10),
        }));

        let empty = ZoneMap {
            null_count: 4,
            min: None,
            max: None,
        };
        assert!(empty.excludes(&ColumnPredicate::Eq(ColumnValue::Int64(1))));
        assert!(empty.excludes(&ColumnPredicate::Lt(ColumnValue::Int64(1))));
        assert!(empty.excludes(&ColumnPredicate::Gt(ColumnValue::Int64(1))));
        assert!(empty.excludes(&ColumnPredicate::Between {
            low: ColumnValue::Int64(0),
            high: ColumnValue::Int64(5),
        }));

        let flags = ZoneMap {
            null_count: 0,
            min: Some(ColumnValue::Bool(false)),
            max: Some(ColumnValue::Bool(true)),
        };
        assert!(flags.excludes(&ColumnPredicate::Lt(ColumnValue::Bool(false))));
        assert!(!flags.excludes(&ColumnPredicate::Eq(ColumnValue::Bool(false))));

        let floats = ZoneMap {
            null_count: 0,
            min: Some(ColumnValue::Float64(1.0)),
            max: Some(ColumnValue::Float64(2.0)),
        };
        assert!(!floats.excludes(&ColumnPredicate::Le(ColumnValue::Float64(1.0))));
        assert!(floats.excludes(&ColumnPredicate::Le(ColumnValue::Float64(0.5))));
        assert!(!floats.excludes(&ColumnPredicate::Ge(ColumnValue::Float64(2.0))));
        assert!(floats.excludes(&ColumnPredicate::Ge(ColumnValue::Float64(3.0))));
        assert!(floats.excludes(&ColumnPredicate::Eq(ColumnValue::Float64(f64::NAN))));
        assert!(floats.excludes(&ColumnPredicate::Lt(ColumnValue::Float64(f64::NAN))));
        assert!(floats.excludes(&ColumnPredicate::Le(ColumnValue::Float64(f64::NAN))));
        assert!(floats.excludes(&ColumnPredicate::Gt(ColumnValue::Float64(f64::NAN))));
        assert!(floats.excludes(&ColumnPredicate::Ge(ColumnValue::Float64(f64::NAN))));
        assert!(floats.excludes(&ColumnPredicate::Between {
            low: ColumnValue::Float64(2.0),
            high: ColumnValue::Float64(1.0),
        }));
    }

    #[test]
    fn non_finite_f64_json_round_trips_bits_and_rejects_a_mismatch() {
        let payload = f64::from_bits(0x7ff8_0000_0000_0001);
        assert_ne!(payload.to_bits(), f64::NAN.to_bits());
        for value in [payload, f64::INFINITY, f64::NEG_INFINITY, -0.0] {
            let encoded = cell_json(&ColumnValue::Float64(value)).unwrap();
            let text = serde_json::to_string(&encoded).unwrap();
            let parsed: Value = serde_json::from_str(&text).unwrap();
            let ColumnValue::Float64(number) = parse_cell(&parsed).unwrap() else {
                panic!("expected f64");
            };
            assert_eq!(number.to_bits(), value.to_bits(), "{text}");
            if value.is_finite() {
                assert!(parsed.get("bits").is_none(), "{parsed}");
            }
        }
        let ColumnValue::Float64(canonical) = parse_cell(&json!({"t": "f64", "v": "NaN"})).unwrap()
        else {
            panic!("expected f64");
        };
        assert_eq!(canonical.to_bits(), f64::NAN.to_bits());
        let ColumnValue::Float64(inf) = parse_cell(&json!({"t": "f64", "v": "Infinity"})).unwrap()
        else {
            panic!("expected f64");
        };
        assert_eq!(inf.to_bits(), f64::INFINITY.to_bits());
        let err = parse_cell(&json!({
            "t": "f64",
            "v": "Infinity",
            "bits": payload.to_bits(),
        }))
        .unwrap_err();
        assert!(err.to_string().contains("do not match"), "{err}");
        let err = parse_cell(&json!({"t": "f64", "v": "1.5"})).unwrap_err();
        assert!(err.to_string().contains("string"), "{err}");
        let err = parse_cell(&json!({"t": "f64", "v": "NaN", "bits": -1})).unwrap_err();
        assert!(err.to_string().contains("0..=u64::MAX"), "{err}");
        let err = parse_cell(&json!({"t": "f64", "v": "NaN", "bits": 1.5})).unwrap_err();
        assert!(err.to_string().contains("0..=u64::MAX"), "{err}");
        let err = parse_cell(&json!({"t": "f64", "v": 1.5, "bits": 0})).unwrap_err();
        assert!(err.to_string().contains("bits"), "{err}");
    }

    #[test]
    fn a_zero_row_group_and_a_nan_zone_bound_are_corrupt() {
        let columns = int_col();
        let mut empty = Vec::new();
        write_u32(&mut empty, 0);
        write_u64(&mut empty, 0);
        empty.push(0);
        empty.push(CODEC_DELTA);
        write_u32(&mut empty, 0);
        let err = validate_row_group(&columns, &empty).unwrap_err();
        assert!(err.to_string().contains("no rows"), "{err}");

        let good = encode_row_group(&columns, &[vec![ColumnValue::Int64(4)]]).unwrap();
        validate_row_group(&columns, &good).unwrap();
        let mut bad = good;
        assert_eq!(bad[29], CODEC_DELTA);
        bad[29] = 0xff;
        let err = validate_row_group(&columns, &bad).unwrap_err();
        assert!(err.to_string().contains("codec"), "{err}");

        let mut blob = Vec::new();
        write_u32(&mut blob, 1);
        write_u64(&mut blob, 0);
        blob.push(1);
        write_i64(&mut blob, 1);
        write_i64(&mut blob, 1);
        blob.push(CODEC_DELTA);
        let mut body = vec![0u8];
        write_i64(&mut body, 1);
        write_u32(&mut blob, u32::try_from(body.len()).unwrap());
        blob.extend_from_slice(&body);
        write_u64(&mut blob, 0);
        blob.push(1);
        blob.extend_from_slice(&f64::NAN.to_le_bytes());
        blob.extend_from_slice(&f64::NAN.to_le_bytes());
        blob.push(CODEC_PLAIN);
        let mut body = vec![0u8];
        body.extend_from_slice(&1.0f64.to_le_bytes());
        write_u32(&mut blob, u32::try_from(body.len()).unwrap());
        blob.extend_from_slice(&body);
        let columns = vec![
            ColumnDef {
                name: "n".to_string(),
                ty: ColumnType::Int64,
            },
            ColumnDef {
                name: "f".to_string(),
                ty: ColumnType::Float64,
            },
        ];
        let err = scan_groups(
            &columns,
            &[blob],
            &[ColumnFilter {
                column: "n".to_string(),
                predicate: ColumnPredicate::Eq(ColumnValue::Int64(99)),
            }],
        )
        .unwrap_err();
        assert!(err.to_string().contains("not ordered"), "{err}");
    }

    fn lp(bytes: &[u8]) -> Vec<u8> {
        let mut out = (bytes.len() as u32).to_le_bytes().to_vec();
        out.extend_from_slice(bytes);
        out
    }

    fn kib(n: usize) -> usize {
        n * 1024
    }

    fn rle_text_group(columns: usize, rows: usize, text_len: usize) -> (Vec<ColumnDef>, Vec<u8>) {
        let cols: Vec<ColumnDef> = (0..columns)
            .map(|i| ColumnDef {
                name: format!("c{i}"),
                ty: ColumnType::Utf8,
            })
            .collect();
        let row = vec![ColumnValue::Utf8("x".repeat(text_len)); columns];
        let blob = encode_row_group(&cols, &vec![row; rows]).unwrap();
        (cols, blob)
    }

    /// Audit S-3 (review follow-up): `row_count` is bounded only by the null bitset, so the
    /// number of rows a blob can claim is checked against a fixed limit before anything is
    /// allocated for them.
    #[test]
    fn a_row_count_over_the_limit_is_refused_before_allocating_for_it() {
        let (cols, blob) = rle_text_group(1, 20, 4);
        let limits = Limits {
            max_rows: 20,
            max_bytes: kib(64),
        };
        validate_row_group_with(&cols, &blob, &limits).unwrap();
        let mut hostile = blob;
        hostile[..4].copy_from_slice(&21u32.to_le_bytes());
        let err = index_group_with(&hostile, &cols, &limits)
            .err()
            .expect("21 rows is over a limit of 20")
            .to_string();
        assert!(
            err.contains("claims 21 rows") && err.contains("limit is 20"),
            "{err}"
        );
        // The real limit refuses a claim of 4 billion rows from a few bytes, and its error
        // names the limit rather than an allocation failure.
        let mut huge = hostile;
        huge[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        let err = index_group(&huge, &cols).err().unwrap().to_string();
        assert!(err.contains(&MAX_ROWS_PER_GROUP.to_string()), "{err}");
    }

    /// Audit S-3 (review follow-up): the row-count check uses the real limit. A claim one over
    /// it is refused from a four-byte blob, before any allocation sized by the claim.
    #[test]
    fn a_claim_of_one_row_over_the_real_row_limit_is_refused_from_four_bytes() {
        let cols = int_col();
        let claim = |rows: usize| u32::try_from(rows).unwrap().to_le_bytes().to_vec();
        let err = index_group(&claim(MAX_ROWS_PER_GROUP + 1), &cols)
            .err()
            .expect("one row over the limit")
            .to_string();
        assert!(
            err.contains(&(MAX_ROWS_PER_GROUP + 1).to_string())
                && err.contains(&format!("limit is {MAX_ROWS_PER_GROUP}")),
            "{err}"
        );
        // At the limit the row check passes; the blob is then refused for being short, not for
        // its row count.
        let err = index_group(&claim(MAX_ROWS_PER_GROUP), &cols)
            .err()
            .expect("the blob has no column data")
            .to_string();
        assert!(!err.contains("claims"), "{err}");
    }

    /// Audit S-3 (review follow-up): the row limit applies to a table with no columns too; such a
    /// group holds only a row count, but the reader's per-row charge is the same, so it is
    /// bounded the same way.
    #[test]
    fn a_table_without_columns_is_bound_by_the_row_limit_too() {
        let limits = Limits {
            max_rows: 5,
            max_bytes: MAX_DECODE_BYTES,
        };
        let five = vec![Vec::new(); 5];
        let blob = encode_row_group_with(&[], &five, &limits).unwrap();
        validate_row_group_with(&[], &blob, &limits).unwrap();
        let err = encode_row_group_with(&[], &vec![Vec::new(); 6], &limits)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("6 rows") && err.contains("limit is 5"),
            "{err}"
        );
    }

    fn rollup_columns_and_rows(n: usize) -> (Vec<ColumnDef>, Vec<Vec<ColumnValue>>) {
        let rows = (0..n)
            .map(|i| {
                vec![
                    ColumnValue::Utf8(format!("p{}", i % 3)),
                    ColumnValue::Int64(i64::try_from(i).unwrap() * 7 % 101),
                    ColumnValue::Int64(1),
                    ColumnValue::Int64(2),
                    ColumnValue::Int64(3),
                ]
            })
            .collect();
        (attempt_columns(), rows)
    }

    /// Audit S-3 (review follow-up): an in-memory rollup is not bound by the writer's group
    /// limits. Here a "group" holds at most 4 rows, 25 rows are scanned, and the result is the
    /// one a single group (under the real limits) gives: same rows, same order, filtered or not.
    #[test]
    fn an_in_memory_scan_over_more_rows_than_one_group_may_hold_equals_the_single_group_scan() {
        let (cols, rows) = rollup_columns_and_rows(25);
        let lowered = Limits {
            max_rows: 4,
            max_bytes: MAX_DECODE_BYTES,
        };
        // The premise: these rows do not fit one group under the lowered limits.
        assert!(encode_row_group_with(&cols, &rows, &lowered).is_err());
        let one_group = vec![encode_row_group(&cols, &rows).unwrap()];
        let eq = |name: &str| {
            vec![ColumnFilter {
                column: "project".to_string(),
                predicate: ColumnPredicate::Eq(ColumnValue::Utf8(name.to_string())),
            }]
        };
        for filters in [Vec::new(), eq("p0"), eq("p2"), eq("absent")] {
            let want = scan_groups(&cols, &one_group, &filters).unwrap();
            let got = scan_projected_with(&cols, &rows, &filters, &lowered).unwrap();
            assert_eq!(got.rows, want.rows, "{filters:?}");
            // Seven chunks of at most four rows each; none is skipped unless its zone excludes.
            assert_eq!(got.row_groups_total, 7);
        }
        // A byte bound ends a chunk too: every chunk of this budget holds fewer rows than 25.
        let tight = Limits {
            max_rows: 1000,
            max_bytes: estimated_decode_bytes(&cols, &rows[..6]),
        };
        assert!(encode_row_group_with(&cols, &rows, &tight).is_err());
        let got = scan_projected_with(&cols, &rows, &[], &tight).unwrap();
        assert_eq!(got.rows, rows);
        assert!(got.row_groups_total > 1);
        // The real function agrees with the single group on every ledger that fits one chunk,
        // counters included.
        for n in [0, 1, 2, 30] {
            let (cols, rows) = rollup_columns_and_rows(n);
            let groups: Vec<Vec<u8>> = if rows.is_empty() {
                Vec::new()
            } else {
                vec![encode_row_group(&cols, &rows).unwrap()]
            };
            let want = scan_groups(&cols, &groups, &eq("p1")).unwrap();
            let got = scan_projected(&cols, &rows, &eq("p1")).unwrap();
            assert_eq!(got.rows, want.rows, "{n}");
            assert_eq!(got.row_groups_total, want.row_groups_total, "{n}");
            assert_eq!(got.row_groups_skipped, want.row_groups_skipped, "{n}");
        }
    }

    /// Audit S-3 (review follow-up): the budget is one for the whole group. Each of the ten
    /// columns below fits the budget alone; together they do not.
    #[test]
    fn the_decode_budget_is_shared_across_the_columns_of_a_group() {
        let limits = Limits {
            max_rows: 1000,
            max_bytes: kib(48),
        };
        let (one_col, one) = rle_text_group(1, 100, 200);
        validate_row_group_with(&one_col, &one, &limits).unwrap();
        let (ten_cols, ten) = rle_text_group(10, 100, 200);
        // Ten columns of 200 repeated bytes are a few hundred bytes of blob per column.
        assert!(ten.len() < kib(8), "{}", ten.len());
        let err = validate_row_group_with(&ten_cols, &ten, &limits)
            .unwrap_err()
            .to_string();
        assert!(err.contains("would decode to more than"), "{err}");
        // An all-null column is not free either: it still costs a value per row.
        let cols = int_col();
        let blob = encode_row_group(&cols, &vec![vec![ColumnValue::Null]; 400]).unwrap();
        let err = validate_row_group_with(
            &cols,
            &blob,
            &Limits {
                max_rows: 1000,
                max_bytes: kib(8),
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("would decode to more than"), "{err}");
    }

    /// Audit S-3 (review follow-up): every codec that expands a few bytes into many values
    /// charges the budget before it allocates. Each payload below is a few bytes and claims
    /// two million values; with a 1 MiB budget each must be refused, and none of them gets to
    /// allocate (an unguarded decoder would build a 64 MB vector, which is the point of the
    /// check).
    #[test]
    fn every_expanding_codec_refuses_a_small_hostile_payload_before_allocating() {
        const N: usize = 2_000_000;
        let refused = |ty: ColumnType, codec: u8, payload: &[u8], n: usize| {
            let mut budget = Budget::new(kib(1024));
            let err = decode_non_null(ty, codec, payload, n, &mut budget)
                .err()
                .unwrap_or_else(|| panic!("codec {codec} decoded {n} values from a small blob"))
                .to_string();
            assert!(
                err.contains("would decode to more than"),
                "codec {codec}: {err}"
            );
        };
        let mut delta_rle = 5i64.to_le_bytes().to_vec();
        delta_rle.extend_from_slice(&2u64.to_le_bytes());
        delta_rle.extend_from_slice(&u32::try_from(N - 1).unwrap().to_le_bytes());
        refused(ColumnType::Int64, CODEC_DELTA_RLE, &delta_rle, N);
        refused(ColumnType::TimestampMs, CODEC_DELTA_RLE, &delta_rle, N);
        refused(ColumnType::Float64, CODEC_RLE, &1.5f64.to_le_bytes(), N);
        let mut bools = vec![1u8];
        bools.extend_from_slice(&u32::try_from(N).unwrap().to_le_bytes());
        refused(ColumnType::Bool, CODEC_RLE, &bools, N);
        refused(ColumnType::Utf8, CODEC_RLE, &lp(b"x"), N);
        refused(ColumnType::Bytes, CODEC_RLE, &lp(b"x"), N);
        // A dictionary of one 64 KiB entry referenced 300 times: 1.2 KB of indexes, 19 MB of
        // text, and the sum is taken before any of it is cloned.
        let big = vec![b'x'; kib(64)];
        let mut dict = 1u32.to_le_bytes().to_vec();
        dict.extend(lp(&big));
        for _ in 0..300 {
            dict.extend_from_slice(&0u32.to_le_bytes());
        }
        refused(ColumnType::Utf8, CODEC_DICT, &dict, 300);
        // Just under the limit still decodes, and a hostile run length past n is still corrupt.
        let mut budget = Budget::new(kib(1024));
        assert_eq!(
            decode_non_null(ColumnType::Bytes, CODEC_RLE, &lp(&big), 4, &mut budget)
                .unwrap()
                .len(),
            4
        );
        let mut budget = Budget::new(kib(1024));
        assert!(decode_non_null(ColumnType::Utf8, CODEC_DICT, &dict, 4, &mut budget).is_err());
    }

    /// Audit S-3 (review follow-up): a plain encoding cannot hold more values than its payload
    /// has bytes for, so `n` is checked against the payload before `Vec::with_capacity(n)`.
    #[test]
    fn plain_decoders_bound_the_value_count_by_the_payload() {
        let mut budget = Budget::new(MAX_DECODE_BYTES);
        for (ty, codec, min_each) in [
            (ColumnType::Utf8, CODEC_PLAIN, 4),
            (ColumnType::Bytes, CODEC_PLAIN, 4),
            (ColumnType::Float64, CODEC_PLAIN, 8),
            (ColumnType::Int64, CODEC_DELTA, 8),
        ] {
            // Four bytes of payload cannot hold a million values of any plain encoding.
            let err = decode_non_null(ty, codec, &[0; 4], 1_000_000, &mut budget)
                .err()
                .unwrap_or_else(|| panic!("codec {codec} accepted a million values"))
                .to_string();
            assert!(
                err.contains(&format!("at least {min_each} bytes")),
                "codec {codec}: {err}"
            );
        }
        assert!(
            decode_non_null(ColumnType::Utf8, CODEC_PLAIN, &[], 0, &mut budget)
                .unwrap()
                .is_empty()
        );
    }

    /// The writer applies the reader's estimate and limits, so what it accepts always decodes:
    /// grow a mixed-codec group (nulls, run-length, dictionary and plain columns) one row at a
    /// time under lowered limits; every size up to the largest the writer accepts must validate
    /// and decode back to the same rows, and the next size must be refused by the writer.
    #[test]
    fn the_largest_group_the_writer_accepts_decodes_under_the_same_limits() {
        let cols: Vec<ColumnDef> = [
            ColumnType::Int64,
            ColumnType::TimestampMs,
            ColumnType::Float64,
            ColumnType::Bool,
            ColumnType::Utf8,
            ColumnType::Utf8,
            ColumnType::Bytes,
        ]
        .into_iter()
        .enumerate()
        .map(|(i, ty)| ColumnDef {
            name: format!("c{i}"),
            ty,
        })
        .collect();
        let row = |i: usize| {
            vec![
                ColumnValue::Int64(i64::try_from(i).unwrap() * 3),
                ColumnValue::TimestampMs(1_000),
                if i.is_multiple_of(5) {
                    ColumnValue::Null
                } else {
                    ColumnValue::Float64(0.5)
                },
                ColumnValue::Bool(i % 7 < 3),
                ColumnValue::Utf8(format!("name-{}", i % 3)),
                ColumnValue::Utf8(format!("distinct-{i}")),
                ColumnValue::Bytes(vec![7; 40]),
            ]
        };
        let limits = Limits {
            max_rows: 400,
            max_bytes: kib(200),
        };
        let mut accepted = 0usize;
        let mut refused_at = None;
        for n in 1..=limits.max_rows + 1 {
            let rows: Vec<Vec<ColumnValue>> = (0..n).map(row).collect();
            match encode_row_group_with(&cols, &rows, &limits) {
                Ok(blob) => {
                    assert!(refused_at.is_none(), "{n} accepted after a smaller refusal");
                    validate_row_group_with(&cols, &blob, &limits)
                        .unwrap_or_else(|err| panic!("{n} rows encoded but do not decode: {err}"));
                    let indexed = index_group_with(&blob, &cols, &limits).unwrap();
                    assert_eq!(decode_indexed_with(&cols, &indexed, &limits).unwrap(), rows);
                    accepted = n;
                }
                Err(err) => {
                    refused_at.get_or_insert((n, err.to_string()));
                }
            }
        }
        let (refused, why) = refused_at.expect("the byte budget must bind before 401 rows");
        assert_eq!(refused, accepted + 1);
        assert!(why.contains("would decode to more than"), "{why}");
        assert!(
            accepted > 100,
            "the budget is not absurdly tight: {accepted}"
        );
        // The row limit binds on its own too.
        let few = Limits {
            max_rows: 5,
            max_bytes: MAX_DECODE_BYTES,
        };
        let six: Vec<Vec<ColumnValue>> = (0..6).map(row).collect();
        let err = encode_row_group_with(&cols, &six, &few)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("6 rows") && err.contains("limit is 5"),
            "{err}"
        );
        // And a group of ordinary size round-trips through the real limits.
        let rows: Vec<Vec<ColumnValue>> = (0..20_000).map(row).collect();
        let blob = encode_row_group(&cols, &rows).unwrap();
        let indexed = index_group(&blob, &cols).unwrap();
        assert_eq!(decode_indexed(&cols, &indexed).unwrap(), rows);
    }

    #[test]
    fn delta_and_run_codecs_reject_truncated_oversized_and_zero_length_runs() {
        // delta_rle: a run of zero length, a run past n, and a truncated payload.
        let mut zero_run = 5i64.to_le_bytes().to_vec();
        zero_run.extend_from_slice(&2u64.to_le_bytes());
        zero_run.extend_from_slice(&0u32.to_le_bytes());
        assert!(decode_delta_rle(ColumnType::Int64, &zero_run, 3).is_err());
        let mut past = 5i64.to_le_bytes().to_vec();
        past.extend_from_slice(&2u64.to_le_bytes());
        past.extend_from_slice(&9u32.to_le_bytes());
        assert!(decode_delta_rle(ColumnType::Int64, &past, 3).is_err());
        assert!(decode_delta_rle(ColumnType::Int64, &past[..10], 3).is_err());
        // A run of exactly n-1 repeats is the boundary that must pass.
        let mut exact = 5i64.to_le_bytes().to_vec();
        exact.extend_from_slice(&2u64.to_le_bytes()); // zigzag(1)
        exact.extend_from_slice(&2u32.to_le_bytes());
        let ok = decode_delta_rle(ColumnType::Int64, &exact, 3).unwrap();
        assert_eq!(
            ok,
            vec![
                ColumnValue::Int64(5),
                ColumnValue::Int64(6),
                ColumnValue::Int64(7)
            ]
        );
        // n == 0 requires an empty payload for every codec.
        assert!(decode_delta(ColumnType::Int64, &[0], 0).is_err());
        assert!(decode_rle_text(&[0], 0, &mut Budget::new(1024)).is_err());
        assert!(decode_delta(ColumnType::Int64, &[], 0).unwrap().is_empty());
        // Wrapping deltas at the i64 boundary round-trip.
        let rows: Vec<Vec<ColumnValue>> = [i64::MIN, i64::MAX, 0, i64::MIN, i64::MAX]
            .iter()
            .map(|v| vec![ColumnValue::Int64(*v)])
            .collect();
        let cols = vec![ColumnDef {
            name: "n".into(),
            ty: ColumnType::Int64,
        }];
        let blob = encode_row_group(&cols, &rows).unwrap();
        validate_row_group(&cols, &blob).unwrap();
        let idx = index_group(&blob, &cols).unwrap();
        assert_eq!(decode_indexed(&cols, &idx).unwrap(), rows);
    }

    #[test]
    fn a_row_count_the_bitset_cannot_cover_is_refused_without_allocating_for_it() {
        let cols = vec![ColumnDef {
            name: "s".into(),
            ty: ColumnType::Utf8,
        }];
        let good = encode_row_group(
            &cols,
            &[
                vec![ColumnValue::Utf8("a".into())],
                vec![ColumnValue::Utf8("a".into())],
                vec![ColumnValue::Utf8("a".into())],
            ],
        )
        .unwrap();
        // Claim 4 billion rows in a blob that carries three.
        let mut hostile = good.clone();
        hostile[..4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(validate_row_group(&cols, &hostile).is_err());
        let mut zero = good;
        zero[..4].copy_from_slice(&0u32.to_le_bytes());
        assert!(validate_row_group(&cols, &zero).is_err());
    }
}
