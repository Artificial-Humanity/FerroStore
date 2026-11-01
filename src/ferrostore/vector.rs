//! In-file vectors: f32 / f16 / i8 storage, HNSW, and hybrid fusion.
//!
//! Quantized bytes are what the WAL and the snapshot store. Distances run on
//! the dequantized `f32` values, accumulated in `f64`.

use super::columnar::check_name;
use fl_core::StoreError;
use serde_json::{Map, Number, Value, json};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};

const MAX_VECTOR_BYTES: u32 = 8 * 1024 * 1024;
const MAX_LEVEL: u8 = 16;
const ID_MAX: usize = 1024;

fn backend(err: impl std::fmt::Display) -> StoreError {
    StoreError::Backend(err.to_string())
}

fn corrupt(detail: impl std::fmt::Display) -> StoreError {
    StoreError::Backend(format!("corrupt ferrostore data: {detail}"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorStorage {
    F32,
    F16,
    I8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorMetric {
    Cosine,
    L2,
    DotProduct,
}

/// One vector search hit. [`crate::FerroStore::search_vectors`] returns the
/// same id and score as a pair.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorHit {
    pub id: String,
    pub score: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fusion {
    Rrf { k: u32 },
    Weighted { bm25: f64, vector: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct KnowledgeHit {
    pub id: String,
    pub score: f64,
    pub metadata: Option<Value>,
    pub doc: Option<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct VectorRecord {
    pub(super) bytes: Vec<u8>,
    pub(super) metadata: Option<Value>,
    pub(super) level: u8,
    /// `neighbors[layer]` for `layer` in `0..=level`.
    pub(super) neighbors: Vec<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct VectorCollection {
    pub(super) dim: u32,
    pub(super) storage: VectorStorage,
    pub(super) metric: VectorMetric,
    pub(super) m: u32,
    pub(super) ef_construction: u32,
    pub(super) ef_search: u32,
    pub(super) vectors: BTreeMap<String, VectorRecord>,
    pub(super) entry: Option<String>,
}

impl VectorCollection {
    pub(super) fn empty(
        dim: u32,
        storage: VectorStorage,
        metric: VectorMetric,
        m: u32,
        ef_construction: u32,
        ef_search: u32,
    ) -> Self {
        Self {
            dim,
            storage,
            metric,
            m,
            ef_construction,
            ef_search,
            vectors: BTreeMap::new(),
            entry: None,
        }
    }
}

/// A vector collection's parameters, without its vectors: what a write body
/// needs to check a vector op, and what a remote transaction reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct VectorParams {
    pub(super) dim: u32,
    pub(super) storage: VectorStorage,
    pub(super) metric: VectorMetric,
    pub(super) m: u32,
    pub(super) ef_construction: u32,
    pub(super) ef_search: u32,
}

impl VectorParams {
    pub(super) fn of(col: &VectorCollection) -> Self {
        Self {
            dim: col.dim,
            storage: col.storage,
            metric: col.metric,
            m: col.m,
            ef_construction: col.ef_construction,
            ef_search: col.ef_search,
        }
    }

    pub(super) fn to_json(self) -> Value {
        json!({
            "dim": self.dim,
            "storage": storage_name(self.storage),
            "metric": metric_name(self.metric),
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
        })
    }

    pub(super) fn parse(value: &Value) -> Result<Self, StoreError> {
        let number = |key: &str| -> Result<u32, StoreError> {
            value
                .get(key)
                .and_then(Value::as_u64)
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| backend(format!("vector parameters are missing `{key}`")))
        };
        let name = |key: &str| -> Result<&str, StoreError> {
            value
                .get(key)
                .and_then(Value::as_str)
                .ok_or_else(|| backend(format!("vector parameters are missing `{key}`")))
        };
        Ok(Self {
            dim: number("dim")?,
            storage: storage_from_name(name("storage")?)?,
            metric: metric_from_name(name("metric")?)?,
            m: number("m")?,
            ef_construction: number("ef_construction")?,
            ef_search: number("ef_search")?,
        })
    }
}

pub(super) fn storage_tag(storage: VectorStorage) -> u8 {
    match storage {
        VectorStorage::F32 => 1,
        VectorStorage::F16 => 2,
        VectorStorage::I8 => 3,
    }
}

pub(super) fn storage_from_tag(tag: u8) -> Result<VectorStorage, StoreError> {
    match tag {
        1 => Ok(VectorStorage::F32),
        2 => Ok(VectorStorage::F16),
        3 => Ok(VectorStorage::I8),
        _ => Err(corrupt(format!("unknown vector storage tag {tag}"))),
    }
}

pub(super) fn metric_tag(metric: VectorMetric) -> u8 {
    match metric {
        VectorMetric::Cosine => 1,
        VectorMetric::L2 => 2,
        VectorMetric::DotProduct => 3,
    }
}

pub(super) fn metric_from_tag(tag: u8) -> Result<VectorMetric, StoreError> {
    match tag {
        1 => Ok(VectorMetric::Cosine),
        2 => Ok(VectorMetric::L2),
        3 => Ok(VectorMetric::DotProduct),
        _ => Err(corrupt(format!("unknown vector metric tag {tag}"))),
    }
}

pub(super) fn storage_name(storage: VectorStorage) -> &'static str {
    match storage {
        VectorStorage::F32 => "f32",
        VectorStorage::F16 => "f16",
        VectorStorage::I8 => "i8",
    }
}

pub(super) fn storage_from_name(name: &str) -> Result<VectorStorage, StoreError> {
    match name {
        "f32" => Ok(VectorStorage::F32),
        "f16" => Ok(VectorStorage::F16),
        "i8" => Ok(VectorStorage::I8),
        _ => Err(backend(format!("unknown vector storage `{name}`"))),
    }
}

pub(super) fn metric_name(metric: VectorMetric) -> &'static str {
    match metric {
        VectorMetric::Cosine => "cosine",
        VectorMetric::L2 => "l2",
        VectorMetric::DotProduct => "dot",
    }
}

pub(super) fn metric_from_name(name: &str) -> Result<VectorMetric, StoreError> {
    match name {
        "cosine" => Ok(VectorMetric::Cosine),
        "l2" => Ok(VectorMetric::L2),
        "dot" => Ok(VectorMetric::DotProduct),
        _ => Err(backend(format!("unknown vector metric `{name}`"))),
    }
}

pub(super) fn check_vector_id(id: &str) -> Result<(), StoreError> {
    if id.is_empty() {
        return Err(backend("vector id is empty"));
    }
    if id.len() > ID_MAX || id.as_bytes().contains(&0) {
        return Err(backend(format!("invalid vector id {id:?}")));
    }
    Ok(())
}

fn bytes_per_value(storage: VectorStorage) -> u32 {
    match storage {
        VectorStorage::F32 => 4,
        VectorStorage::F16 => 2,
        VectorStorage::I8 => 1,
    }
}

pub(super) fn stored_len(dim: u32, storage: VectorStorage) -> Result<usize, StoreError> {
    if dim == 0 {
        return Err(backend("vector dimension must be greater than zero"));
    }
    let bytes = dim
        .checked_mul(bytes_per_value(storage))
        .ok_or_else(|| backend("vector dimension overflows"))?;
    if bytes > MAX_VECTOR_BYTES {
        return Err(backend("vector dimension exceeds 8MiB"));
    }
    usize::try_from(bytes).map_err(|_| backend("vector dimension exceeds 8MiB"))
}

pub(super) fn check_create(
    name: &str,
    dim: u32,
    storage: VectorStorage,
    m: u32,
    ef_construction: u32,
    ef_search: u32,
) -> Result<(), StoreError> {
    check_name("vector collection", name)?;
    stored_len(dim, storage)?;
    if !(2..=128).contains(&m) {
        return Err(backend("vector M is out of range"));
    }
    if ef_construction < m || ef_construction > 100_000 {
        return Err(backend("vector ef_construction is out of range"));
    }
    if !(1..=100_000).contains(&ef_search) {
        return Err(backend("vector ef_search is out of range"));
    }
    Ok(())
}

pub(super) fn vector_level(id: &str, m: u32) -> u8 {
    let hash = Sha256::digest(id.as_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&hash[..8]);
    let mut x = u64::from_le_bytes(bytes);
    if x == 0 {
        x = 1;
    }
    let u = (x as f64) / (u64::MAX as f64);
    let level = (-u.ln() * (1.0 / (m as f64).ln())).floor();
    if !level.is_finite() || level <= 0.0 {
        0
    } else if level >= f64::from(MAX_LEVEL) {
        MAX_LEVEL
    } else {
        level as u8
    }
}

pub(super) fn max_links(m: u32, layer: usize) -> usize {
    if layer == 0 {
        usize::try_from(m).unwrap_or(usize::MAX).saturating_mul(2)
    } else {
        usize::try_from(m).unwrap_or(usize::MAX)
    }
}

/// IEEE binary16, round ties to even. A finite value outside the finite f16
/// range, including one that rounds to infinity, is an error.
fn f32_to_f16(value: f32) -> Result<u16, StoreError> {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff || !value.is_finite() {
        return Err(backend("vector component is not finite"));
    }
    if exp == 0 {
        // Every f32 subnormal is smaller than half the tiniest f16 subnormal.
        return Ok(sign);
    }
    let unbiased = exp - 127;
    if unbiased > 15 {
        return Err(backend("f16 vector component is out of range"));
    }
    if unbiased >= -14 {
        let mut exp16 = (unbiased + 15) as u16;
        let round_bit = (mant >> 12) & 1;
        let sticky = mant & 0x0fff;
        let mut mant16 = (mant >> 13) as u16;
        let round_up = round_bit == 1 && (sticky != 0 || mant16 & 1 == 1);
        if round_up {
            mant16 += 1;
            if mant16 == 0x400 {
                mant16 = 0;
                exp16 += 1;
                if exp16 >= 31 {
                    return Err(backend("f16 vector component is out of range"));
                }
            }
        }
        return Ok(sign | (exp16 << 10) | mant16);
    }
    if unbiased < -25 {
        return Ok(sign);
    }
    // Subnormal f16. `shift` puts the f32 significand onto the 2^-24 grid.
    let sig = (1u32 << 23) | mant;
    let shift = (-unbiased - 1) as u32;
    let mut mant16 = (sig >> shift) as u16;
    let round_bit = if shift == 0 {
        0
    } else {
        (sig >> (shift - 1)) & 1
    };
    let sticky = if shift <= 1 {
        0
    } else {
        sig & ((1u32 << (shift - 1)) - 1)
    };
    let round_up = round_bit == 1 && (sticky != 0 || mant16 & 1 == 1);
    if round_up {
        mant16 = mant16.saturating_add(1);
    }
    if mant16 == 0x400 {
        return Ok(sign | (1 << 10));
    }
    Ok(sign | mant16)
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits) & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1f);
    let mant = u32::from(bits & 0x03ff);
    let decoded = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut m = mant;
            let mut exp32 = 127 - 15 + 1;
            while m & 0x400 == 0 {
                m <<= 1;
                exp32 -= 1;
            }
            m &= 0x3ff;
            sign | (exp32 << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        let exp32 = exp + (127 - 15);
        sign | (exp32 << 23) | (mant << 13)
    };
    f32::from_bits(decoded)
}

fn quantize_i8(value: f32) -> i8 {
    // `f32::round` is half away from zero. `quantize` rejects non-finite values first.
    value.round().clamp(-128.0, 127.0) as i8
}

pub(super) fn quantize(storage: VectorStorage, values: &[f32]) -> Result<Vec<u8>, StoreError> {
    let mut out = Vec::with_capacity(values.len() * bytes_per_value(storage) as usize);
    match storage {
        VectorStorage::F32 => {
            for value in values {
                if !value.is_finite() {
                    return Err(backend("vector component is not finite"));
                }
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        VectorStorage::F16 => {
            for value in values {
                if !value.is_finite() {
                    return Err(backend("vector component is not finite"));
                }
                out.extend_from_slice(&f32_to_f16(*value)?.to_le_bytes());
            }
        }
        VectorStorage::I8 => {
            for value in values {
                if !value.is_finite() {
                    return Err(backend("vector component is not finite"));
                }
                out.push(quantize_i8(*value) as u8);
            }
        }
    }
    Ok(out)
}

pub(super) fn dequantize(storage: VectorStorage, bytes: &[u8]) -> Result<Vec<f32>, StoreError> {
    match storage {
        VectorStorage::F32 => {
            if !bytes.len().is_multiple_of(4) {
                return Err(backend("stored f32 vector length is not a multiple of 4"));
            }
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.as_chunks::<4>().0 {
                let value = f32::from_le_bytes(*chunk);
                if !value.is_finite() {
                    return Err(backend("stored vector component is not finite"));
                }
                out.push(value);
            }
            Ok(out)
        }
        VectorStorage::F16 => {
            if !bytes.len().is_multiple_of(2) {
                return Err(backend("stored f16 vector length is not a multiple of 2"));
            }
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.as_chunks::<2>().0 {
                let value = f16_to_f32(u16::from_le_bytes(*chunk));
                if !value.is_finite() {
                    return Err(backend("stored vector component is not finite"));
                }
                out.push(value);
            }
            Ok(out)
        }
        VectorStorage::I8 => Ok(bytes.iter().map(|byte| *byte as i8 as f32).collect()),
    }
}

pub(super) fn check_stored(
    storage: VectorStorage,
    dim: u32,
    bytes: &[u8],
) -> Result<(), StoreError> {
    let expected = stored_len(dim, storage)?;
    if bytes.len() != expected {
        return Err(backend(format!(
            "stored vector has length {}, expected {expected}",
            bytes.len()
        )));
    }
    // The finiteness `dequantize` checks, without building the values.
    let finite = match storage {
        VectorStorage::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .all(|chunk| f32::from_le_bytes(*chunk).is_finite()),
        // Exponent 31 is infinity or NaN; every other f16 is finite.
        VectorStorage::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .all(|chunk| u16::from_le_bytes(*chunk) & 0x7c00 != 0x7c00),
        VectorStorage::I8 => true,
    };
    if !finite {
        return Err(backend("stored vector component is not finite"));
    }
    Ok(())
}

fn dot(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum()
}

fn euclidean(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| {
            let delta = f64::from(*a) - f64::from(*b);
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}

fn cosine(left: &[f32], right: &[f32]) -> f64 {
    let dot = dot(left, right);
    let left_norm = dot_self(left).sqrt();
    let right_norm = dot_self(right).sqrt();
    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot / (left_norm * right_norm)
    }
}

fn dot_self(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum()
}

fn similarity(metric: VectorMetric, left: &[f32], right: &[f32]) -> f64 {
    match metric {
        VectorMetric::L2 => 1.0 / (1.0 + euclidean(left, right)),
        VectorMetric::Cosine => cosine(left, right),
        VectorMetric::DotProduct => dot(left, right),
    }
}

fn graph_distance(metric: VectorMetric, left: &[f32], right: &[f32]) -> f64 {
    match metric {
        VectorMetric::L2 => euclidean(left, right),
        VectorMetric::Cosine => 1.0 - cosine(left, right),
        VectorMetric::DotProduct => -dot(left, right),
    }
}

/// Smaller graph distance, then lower id. The same order `sort_layer` writes.
fn cmp_distance_then_id(
    left_dist: f64,
    left_id: &str,
    right_dist: f64,
    right_id: &str,
) -> Ordering {
    left_dist
        .total_cmp(&right_dist)
        .then_with(|| left_id.cmp(right_id))
}

/// Highest level, then lowest id. `max_by` treats a lower id as greater.
/// The snapshot stores this entry and `validate_graph` requires it; an upsert
/// keeps it with `VectorCollection::admit_entry` instead of a rescan.
pub(super) fn canonical_entry(vectors: &BTreeMap<String, VectorRecord>) -> Option<String> {
    vectors
        .iter()
        .max_by(|(id_a, rec_a), (id_b, rec_b)| {
            rec_a.level.cmp(&rec_b.level).then_with(|| id_b.cmp(id_a))
        })
        .map(|(id, _)| id.clone())
}

#[derive(Clone)]
struct Rank {
    dist: f64,
    id: String,
}

impl Rank {
    fn new(dist: f64, id: String) -> Self {
        Self { dist, id }
    }
}

impl PartialEq for Rank {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Rank {}

impl PartialOrd for Rank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rank {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl VectorCollection {
    fn stored(&self, id: &str) -> Result<Vec<f32>, StoreError> {
        let Some(record) = self.vectors.get(id) else {
            return Err(backend(format!("vector `{id}` is missing")));
        };
        dequantize(self.storage, &record.bytes)
    }

    fn distance_to(&self, query: &[f32], id: &str) -> Result<f64, StoreError> {
        Ok(graph_distance(self.metric, query, &self.stored(id)?))
    }

    fn pair_distance(&self, origin: &[f32], id: &str) -> Result<f64, StoreError> {
        Ok(graph_distance(self.metric, origin, &self.stored(id)?))
    }

    pub(super) fn upsert(
        &mut self,
        id: String,
        bytes: Vec<u8>,
        metadata: Option<Value>,
    ) -> Result<(), StoreError> {
        check_vector_id(&id)?;
        check_stored(self.storage, self.dim, &bytes)?;
        if self.vectors.contains_key(&id) {
            self.remove_node(&id)?;
        }
        let level = vector_level(&id, self.m);
        let query = dequantize(self.storage, &bytes)?;
        if self.vectors.is_empty() {
            self.vectors.insert(
                id.clone(),
                VectorRecord {
                    bytes,
                    metadata,
                    level,
                    neighbors: vec![Vec::new(); usize::from(level) + 1],
                },
            );
            self.entry = Some(id);
            return Ok(());
        }
        let entry_id = self
            .entry
            .clone()
            .filter(|entry| self.vectors.contains_key(entry))
            .ok_or_else(|| backend("vector collection is missing an entry point"))?;
        let entry_level = self
            .vectors
            .get(&entry_id)
            .map(|rec| rec.level)
            .unwrap_or(0);
        let mut eps = vec![entry_id];
        if entry_level > level {
            for layer in (usize::from(level) + 1..=usize::from(entry_level)).rev() {
                let found = search_layer(self, &query, &eps, 1, layer)?;
                if let Some(rank) = found.first() {
                    eps = vec![rank.id.clone()];
                }
            }
        }
        self.vectors.insert(
            id.clone(),
            VectorRecord {
                bytes,
                metadata,
                level,
                neighbors: vec![Vec::new(); usize::from(level) + 1],
            },
        );
        let top = level.min(entry_level);
        for layer in (0..=usize::from(top)).rev() {
            self.link_layer(&id, &query, &mut eps, layer)?;
        }
        self.admit_entry(&id, level);
        Ok(())
    }

    pub(super) fn delete_id(&mut self, id: &str) -> Result<(), StoreError> {
        check_vector_id(id)?;
        if self.vectors.contains_key(id) {
            self.remove_node(id)?;
        }
        Ok(())
    }

    /// Remove `id` and re-link its former neighbours. A delete, and a replacing upsert (which
    /// removes the old node first), pay for the repair: it is bounded but not cheap. For each
    /// layer with `k` former neighbours it ranks up to `k - 1 + max_links` candidates per
    /// neighbour (`k` is at most `max_links`, i.e. `m`, or `2m` on layer 0), so about `k^2`
    /// dequantizing distance computations: roughly 5 thousand at `m = 16`, 500 thousand at
    /// `m = 128`, each over `dim` values. At `dim` near 1000 and `m = 128` that is several hundred
    /// milliseconds per delete, and a WAL replay pays it again for every delete it replays. The
    /// repair stays this simple on purpose: it must give the same graph on every replica.
    ///
    /// An error here (a stored vector that no longer decodes) aborts the operation. For a delete
    /// the caller applies it to a staged copy and commits nothing on error, so the graph is never
    /// left half repaired.
    fn remove_node(&mut self, id: &str) -> Result<(), StoreError> {
        let Some(record) = self.vectors.get(id) else {
            return Ok(());
        };
        let neighbors = record.neighbors.clone();
        for (layer, nids) in neighbors.iter().enumerate() {
            for nid in nids {
                if let Some(slot) = self
                    .vectors
                    .get_mut(nid)
                    .and_then(|rec| rec.neighbors.get_mut(layer))
                {
                    slot.retain(|other| other != id);
                }
            }
        }
        self.vectors.remove(id);
        // Dropping the node's links alone can leave a neighbour whose only way in was this node
        // unreachable from the entry point: still stored, never returned by a search.
        for (layer, nids) in neighbors.iter().enumerate() {
            self.repair_layer(nids, layer)?;
        }
        // Removing any node but the entry leaves the entry canonical: it still has the highest
        // level and, at that level, the lowest id. Removing the entry itself rescans, O(n).
        if self.entry.as_deref() == Some(id) {
            self.recompute_entry();
        }
        Ok(())
    }

    /// Reconnect `orphans`, the former neighbours of a deleted node on `layer`: each may link to
    /// the others (they were all one hop from the deleted node), keeping its `max_links` closest
    /// links, and every link stays reciprocal. An orphan never drops a link that would leave the
    /// node at the other end with fewer than two links (the same rule `make_room` applies), or the
    /// repair would itself strand a neighbour whose only way in was that orphan.
    fn repair_layer(&mut self, orphans: &[String], layer: usize) -> Result<(), StoreError> {
        let max = max_links(self.m, layer);
        let present: Vec<&String> = orphans
            .iter()
            .filter(|nid| {
                self.vectors
                    .get(*nid)
                    .is_some_and(|rec| rec.neighbors.len() > layer)
            })
            .collect();
        let mut touched: BTreeSet<String> = BTreeSet::new();
        for nid in &present {
            let current = self
                .vectors
                .get(*nid)
                .and_then(|rec| rec.neighbors.get(layer))
                .cloned()
                .unwrap_or_default();
            let mut pool: BTreeSet<String> = current.iter().cloned().collect();
            pool.extend(present.iter().filter(|o| *o != nid).map(|o| (*o).clone()));
            let origin = self.stored(nid)?;
            let mut ranked = Vec::with_capacity(pool.len());
            for candidate in pool {
                let distance = self.pair_distance(&origin, &candidate)?;
                ranked.push((distance, candidate));
            }
            ranked.sort_by(|left, right| cmp_distance_then_id(left.0, &left.1, right.0, &right.1));
            // A current link whose other end has no link to spare is kept first; the rest of the
            // room goes to the closest candidates.
            let protected = |col: &Self, candidate: &String| {
                current.contains(candidate)
                    && col
                        .vectors
                        .get(candidate)
                        .and_then(|rec| rec.neighbors.get(layer))
                        .is_some_and(|slot| slot.len() <= 2)
            };
            let mut keep: Vec<String> = ranked
                .iter()
                .filter(|(_, candidate)| protected(self, candidate))
                .map(|(_, candidate)| candidate.clone())
                .collect();
            for (_, candidate) in &ranked {
                if keep.len() >= max {
                    break;
                }
                if !keep.contains(candidate) {
                    keep.push(candidate.clone());
                }
            }
            // A link this node gives up must also leave the other side, or that side keeps a
            // one-way link that dangles once this node is deleted.
            for dropped in current.iter().filter(|other| !keep.contains(other)) {
                if let Some(slot) = self
                    .vectors
                    .get_mut(dropped)
                    .and_then(|rec| rec.neighbors.get_mut(layer))
                {
                    slot.retain(|other| other != *nid);
                }
            }
            for candidate in &keep {
                if let Some(slot) = self
                    .vectors
                    .get_mut(candidate)
                    .and_then(|rec| rec.neighbors.get_mut(layer))
                    && !slot.iter().any(|other| other == *nid)
                {
                    slot.push((*nid).clone());
                    touched.insert(candidate.clone());
                }
            }
            if let Some(slot) = self
                .vectors
                .get_mut(*nid)
                .and_then(|rec| rec.neighbors.get_mut(layer))
            {
                *slot = keep;
            }
            touched.insert((*nid).clone());
        }
        for nid in &touched {
            self.prune(nid, layer)?;
        }
        for nid in &touched {
            self.ensure_reciprocal(nid, layer);
        }
        // Closest-link selection can split the orphans into clusters that were joined only by
        // the deleted node. A chain through all of them (by id, so it is deterministic) keeps
        // them one component.
        let mut chain: Vec<&String> = present;
        chain.sort();
        for pair in chain.windows(2) {
            self.ensure_edge(pair[0], pair[1], layer, &mut touched)?;
        }
        for nid in &touched {
            self.sort_layer(nid, layer)?;
        }
        Ok(())
    }

    /// Link `a` and `b` on `layer` in both directions. A full node first gives up its farthest
    /// link, but never one that would leave that neighbour with fewer than two links. If either
    /// side cannot make room, nothing changes: a room already made for `a` is given back before
    /// returning, so a refused edge never leaves `a` one link short.
    fn ensure_edge(
        &mut self,
        a: &str,
        b: &str,
        layer: usize,
        touched: &mut BTreeSet<String>,
    ) -> Result<(), StoreError> {
        if a == b {
            return Ok(());
        }
        let linked = |col: &Self, x: &str, y: &str| {
            col.vectors
                .get(x)
                .and_then(|rec| rec.neighbors.get(layer))
                .is_some_and(|slot| slot.iter().any(|other| other == y))
        };
        if linked(self, a, b) {
            return Ok(());
        }
        let Some(evicted_for_a) = self.make_room(a, b, layer, touched)? else {
            return Ok(());
        };
        let Some(_) = self.make_room(b, a, layer, touched)? else {
            // `a` made room (evicting `victim`) but `b` cannot: put the evicted link back. The
            // nodes are in `touched`, whose lists are re-sorted at the end of the repair.
            if let Some(victim) = evicted_for_a {
                for (from, back) in [(a, victim.as_str()), (victim.as_str(), a)] {
                    if let Some(slot) = self
                        .vectors
                        .get_mut(from)
                        .and_then(|rec| rec.neighbors.get_mut(layer))
                    {
                        slot.push(back.to_string());
                    }
                }
            }
            return Ok(());
        };
        for (x, y) in [(a, b), (b, a)] {
            if let Some(slot) = self
                .vectors
                .get_mut(x)
                .and_then(|rec| rec.neighbors.get_mut(layer))
            {
                slot.push(y.to_string());
                touched.insert(x.to_string());
            }
        }
        Ok(())
    }

    /// Make room on `layer` for one more link on `x`. `Ok(None)`: it cannot (nothing changed);
    /// `Ok(Some(None))`: it already had room; `Ok(Some(Some(victim)))`: the link to `victim` was
    /// removed on both sides.
    fn make_room(
        &mut self,
        x: &str,
        keep: &str,
        layer: usize,
        touched: &mut BTreeSet<String>,
    ) -> Result<Option<Option<String>>, StoreError> {
        let max = max_links(self.m, layer);
        let Some(current) = self
            .vectors
            .get(x)
            .and_then(|rec| rec.neighbors.get(layer))
            .cloned()
        else {
            return Ok(None);
        };
        if current.len() < max {
            return Ok(Some(None));
        }
        let origin = self.stored(x)?;
        let mut ranked = Vec::with_capacity(current.len());
        for nid in current {
            let distance = self.pair_distance(&origin, &nid)?;
            ranked.push((distance, nid));
        }
        // Farthest first.
        ranked.sort_by(|left, right| cmp_distance_then_id(right.0, &right.1, left.0, &left.1));
        let victim = ranked.into_iter().map(|(_, nid)| nid).find(|nid| {
            nid != keep
                && self
                    .vectors
                    .get(nid)
                    .and_then(|rec| rec.neighbors.get(layer))
                    .is_some_and(|slot| slot.len() > 2)
        });
        let Some(victim) = victim else {
            return Ok(None);
        };
        for (from, gone) in [(x, victim.as_str()), (victim.as_str(), x)] {
            if let Some(slot) = self
                .vectors
                .get_mut(from)
                .and_then(|rec| rec.neighbors.get_mut(layer))
            {
                slot.retain(|other| other != gone);
                touched.insert(from.to_string());
            }
        }
        Ok(Some(Some(victim)))
    }

    fn recompute_entry(&mut self) {
        self.entry = canonical_entry(&self.vectors);
    }

    /// Keep the entry canonical after `id`, at `level`, was inserted, in O(1): `id` takes the
    /// entry when its level is higher, or equal with a lower id ([`canonical_entry`]'s order).
    /// The entry before the insert was canonical for the other nodes, so the result is the
    /// canonical entry of all of them. An entry that is not live (never so in a collection that
    /// passed `validate_graph` or was built by these ops) falls back to the rescan.
    fn admit_entry(&mut self, id: &str, level: u8) {
        let held = self
            .entry
            .as_deref()
            .and_then(|entry| self.vectors.get(entry).map(|rec| (entry, rec.level)));
        match held {
            Some((entry, entry_level)) => {
                if level > entry_level || (level == entry_level && id < entry) {
                    self.entry = Some(id.to_string());
                }
            }
            None => self.recompute_entry(),
        }
    }

    fn link_layer(
        &mut self,
        id: &str,
        query: &[f32],
        eps: &mut Vec<String>,
        layer: usize,
    ) -> Result<(), StoreError> {
        let ef = usize::try_from(self.ef_construction).unwrap_or(usize::MAX);
        let found = search_layer(self, query, eps, ef, layer)?;
        let selected: Vec<String> = found
            .iter()
            .map(|rank| rank.id.clone())
            .filter(|nid| nid != id)
            .take(max_links(self.m, layer))
            .collect();
        if let Some(slot) = self
            .vectors
            .get_mut(id)
            .and_then(|rec| rec.neighbors.get_mut(layer))
        {
            *slot = selected.clone();
        }
        for nid in &selected {
            if let Some(slot) = self
                .vectors
                .get_mut(nid)
                .and_then(|rec| rec.neighbors.get_mut(layer))
                && !slot.iter().any(|other| other == id)
            {
                slot.push(id.to_string());
            }
        }
        for nid in &selected {
            self.prune(nid, layer)?;
        }
        self.ensure_reciprocal(id, layer);
        self.sort_layer(id, layer)?;
        let next: Vec<String> = found
            .into_iter()
            .map(|rank| rank.id)
            .filter(|nid| nid != id)
            .collect();
        if !next.is_empty() {
            *eps = next;
        }
        Ok(())
    }

    fn prune(&mut self, id: &str, layer: usize) -> Result<(), StoreError> {
        let current = self
            .vectors
            .get(id)
            .and_then(|rec| rec.neighbors.get(layer))
            .cloned()
            .unwrap_or_default();
        let max = max_links(self.m, layer);
        if current.len() <= max {
            self.sort_layer(id, layer)?;
            return Ok(());
        }
        let origin = self.stored(id)?;
        let mut ranked = Vec::with_capacity(current.len());
        for nid in current {
            ranked.push((self.pair_distance(&origin, &nid)?, nid));
        }
        ranked.sort_by(|left, right| cmp_distance_then_id(left.0, &left.1, right.0, &right.1));
        ranked.dedup_by(|left, right| left.1 == right.1);
        let keep: Vec<String> = ranked
            .iter()
            .take(max)
            .map(|(_, nid)| nid.clone())
            .collect();
        let drop: Vec<String> = ranked
            .iter()
            .skip(max)
            .map(|(_, nid)| nid.clone())
            .collect();
        if let Some(slot) = self
            .vectors
            .get_mut(id)
            .and_then(|rec| rec.neighbors.get_mut(layer))
        {
            *slot = keep;
        }
        for nid in drop {
            if let Some(slot) = self
                .vectors
                .get_mut(&nid)
                .and_then(|rec| rec.neighbors.get_mut(layer))
            {
                slot.retain(|other| other != id);
            }
        }
        Ok(())
    }

    fn ensure_reciprocal(&mut self, id: &str, layer: usize) {
        let neighbors = self
            .vectors
            .get(id)
            .and_then(|rec| rec.neighbors.get(layer))
            .cloned()
            .unwrap_or_default();
        let keep: Vec<String> = neighbors
            .into_iter()
            .filter(|nid| {
                self.vectors
                    .get(nid)
                    .and_then(|rec| rec.neighbors.get(layer))
                    .is_some_and(|slot| slot.iter().any(|other| other == id))
            })
            .collect();
        if let Some(slot) = self
            .vectors
            .get_mut(id)
            .and_then(|rec| rec.neighbors.get_mut(layer))
        {
            *slot = keep;
        }
    }

    fn sort_layer(&mut self, id: &str, layer: usize) -> Result<(), StoreError> {
        let current = self
            .vectors
            .get(id)
            .and_then(|rec| rec.neighbors.get(layer))
            .cloned()
            .unwrap_or_default();
        if current.len() < 2 {
            return Ok(());
        }
        let origin = self.stored(id)?;
        let mut ranked = Vec::with_capacity(current.len());
        for nid in current {
            ranked.push((self.pair_distance(&origin, &nid)?, nid));
        }
        ranked.sort_by(|left, right| cmp_distance_then_id(left.0, &left.1, right.0, &right.1));
        let keep: Vec<String> = ranked.into_iter().map(|(_, nid)| nid).collect();
        if let Some(slot) = self
            .vectors
            .get_mut(id)
            .and_then(|rec| rec.neighbors.get_mut(layer))
        {
            *slot = keep;
        }
        Ok(())
    }
}

fn search_layer(
    col: &VectorCollection,
    query: &[f32],
    entry_ids: &[String],
    ef: usize,
    layer: usize,
) -> Result<Vec<Rank>, StoreError> {
    if ef == 0 || entry_ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut visited = HashSet::new();
    let mut candidates: BinaryHeap<std::cmp::Reverse<Rank>> = BinaryHeap::new();
    let mut best: Vec<Rank> = Vec::new();
    for id in entry_ids {
        if !visited.insert(id.clone()) || !col.vectors.contains_key(id) {
            continue;
        }
        let rank = Rank::new(col.distance_to(query, id)?, id.clone());
        candidates.push(std::cmp::Reverse(rank.clone()));
        best.push(rank);
    }
    if best.is_empty() {
        return Ok(Vec::new());
    }
    best.sort();
    if best.len() > ef {
        best.truncate(ef);
    }
    while let Some(std::cmp::Reverse(current)) = candidates.pop() {
        if best.len() >= ef
            && let Some(worst) = best.last()
            && current.dist.total_cmp(&worst.dist) == Ordering::Greater
        {
            break;
        }
        let neighbors = col
            .vectors
            .get(&current.id)
            .and_then(|rec| rec.neighbors.get(layer))
            .cloned()
            .unwrap_or_default();
        for nid in neighbors {
            if !visited.insert(nid.clone()) {
                continue;
            }
            let rank = Rank::new(col.distance_to(query, &nid)?, nid);
            let fits = match best.last() {
                Some(worst) if best.len() >= ef => rank < *worst,
                _ => true,
            };
            if fits {
                candidates.push(std::cmp::Reverse(rank.clone()));
                // `best` is kept sorted, so insert in place: re-sorting it for every visited
                // node made a wide search quadratic in `ef`.
                let at = best.partition_point(|held| *held < rank);
                best.insert(at, rank);
                if best.len() > ef {
                    best.truncate(ef);
                }
            }
        }
    }
    Ok(best)
}

pub(super) fn prepare_query(
    col: &VectorCollection,
    values: &[f32],
) -> Result<Vec<f32>, StoreError> {
    if u32::try_from(values.len()).ok() != Some(col.dim) {
        return Err(backend(format!(
            "vector values have length {}, expected {}",
            values.len(),
            col.dim
        )));
    }
    let bytes = quantize(col.storage, values)?;
    dequantize(col.storage, &bytes)
}

pub(super) fn search(
    col: &VectorCollection,
    values: &[f32],
    limit: usize,
    ef: Option<u32>,
) -> Result<Vec<(String, f64)>, StoreError> {
    let query = prepare_query(col, values)?;
    if limit == 0 || ef == Some(0) {
        return Ok(Vec::new());
    }
    if col.vectors.is_empty() {
        return Ok(Vec::new());
    }
    let entry = col
        .entry
        .clone()
        .filter(|id| col.vectors.contains_key(id))
        .ok_or_else(|| backend("vector collection is missing an entry point"))?;
    let entry_level = col.vectors.get(&entry).map(|rec| rec.level).unwrap_or(0);
    let mut eps = vec![entry];
    if entry_level >= 1 {
        for layer in (1..=usize::from(entry_level)).rev() {
            let found = search_layer(col, &query, &eps, 1, layer)?;
            if let Some(rank) = found.first() {
                eps = vec![rank.id.clone()];
            }
        }
    }
    let width = usize::try_from(ef.unwrap_or(col.ef_search)).unwrap_or(usize::MAX);
    let width = width.max(limit);
    let found = search_layer(col, &query, &eps, width, 0)?;
    let mut scored = Vec::with_capacity(found.len().min(limit));
    for rank in found.into_iter().take(limit) {
        let stored = col.stored(&rank.id)?;
        let score = similarity(col.metric, &query, &stored);
        if !score.is_finite() {
            return Err(backend("vector score is not finite"));
        }
        scored.push((rank.id, score));
    }
    scored.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    Ok(scored)
}

pub(super) fn validate_fusion(fusion: Fusion) -> Result<(), StoreError> {
    match fusion {
        Fusion::Rrf { k } => {
            if k < 1 {
                return Err(backend("rrf k must be >= 1"));
            }
        }
        Fusion::Weighted { bm25, vector } => {
            if !bm25.is_finite()
                || !vector.is_finite()
                || bm25 < 0.0
                || vector < 0.0
                || (bm25 == 0.0 && vector == 0.0)
                || !(bm25 + vector).is_finite()
            {
                return Err(backend(
                    "weighted fusion weights must be finite, >= 0, and not both zero",
                ));
            }
        }
    }
    Ok(())
}

pub(super) fn fuse(
    bm25: &[(String, f64)],
    vector: &[(String, f64)],
    fusion: Fusion,
    limit: usize,
) -> Result<Vec<(String, f64)>, StoreError> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    validate_fusion(fusion)?;
    let mut fused = match fusion {
        Fusion::Rrf { k } => {
            let mut scores: BTreeMap<String, f64> = BTreeMap::new();
            for (rank, (id, _)) in bm25.iter().enumerate() {
                let score = 1.0 / (f64::from(k) + rank as f64 + 1.0);
                *scores.entry(id.clone()).or_insert(0.0) += score;
            }
            for (rank, (id, _)) in vector.iter().enumerate() {
                let score = 1.0 / (f64::from(k) + rank as f64 + 1.0);
                *scores.entry(id.clone()).or_insert(0.0) += score;
            }
            scores.into_iter().collect::<Vec<_>>()
        }
        Fusion::Weighted {
            bm25: bm25_weight,
            vector: vector_weight,
        } => {
            let bm25_norm = normalize(bm25);
            let vector_norm = normalize(vector);
            let mut ids = BTreeSet::new();
            ids.extend(bm25_norm.keys().cloned());
            ids.extend(vector_norm.keys().cloned());
            ids.into_iter()
                .map(|id| {
                    let score = bm25_weight * bm25_norm.get(&id).copied().unwrap_or(0.0)
                        + vector_weight * vector_norm.get(&id).copied().unwrap_or(0.0);
                    (id, score)
                })
                .collect::<Vec<_>>()
        }
    };
    fused.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    fused.truncate(limit);
    Ok(fused)
}

fn normalize(hits: &[(String, f64)]) -> BTreeMap<String, f64> {
    if hits.is_empty() {
        return BTreeMap::new();
    }
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for (_, score) in hits {
        min = min.min(*score);
        max = max.max(*score);
    }
    let mut out = BTreeMap::new();
    for (id, score) in hits {
        let norm = if max == min {
            1.0
        } else {
            (score - min) / (max - min)
        };
        out.insert(id.clone(), norm);
    }
    out
}

/// The graph checks `FerroStore::open` runs on every vector collection a snapshot decodes:
/// the collection parameters; a canonical, live entry point; and for every vector its id, its
/// stored bytes, its level, one neighbor list per layer, and on every layer at most
/// `max_links` neighbors with no self link, no repeat, each neighbor live, present on that
/// layer, and linking back. These keep a corrupt but CRC-valid snapshot (a writer bug) from
/// loading a graph that a search or a later delete repair would walk off.
///
/// It does not check that each neighbor list is ordered by (distance, id): that is
/// [`validate_graph_full`]. Open skips it because a search's results do not depend on it:
/// every hit is a live vector with an exactly computed score, and the final result set of
/// `search_layer` is the same whatever order the lists are in (the order only steers which
/// near nodes an approximate search visits first). Delete-repair determinism does depend on
/// it: `remove_node` hands the deleted node's list, in stored order, to `repair_layer`, which
/// visits it in that order and changes the graph at each step. So the order is enforced at
/// write time (`link_layer` and `sort_layer` write it), and the snapshot encoder checks it
/// in full before every snapshot write (`encode_snapshot_v3` calls [`validate_graph_full`]).
/// A misordered but CRC-valid snapshot (only a writer bug can make one) is therefore
/// accepted on open, and the next checkpoint fails with the full check's error. The CRC
/// protects the bytes after the write. Checking the order on open cost one dequantize and
/// one distance per link, most of the open time of a large collection.
///
/// The checks run on integer indices into the collection: one id lookup per link, and a
/// reciprocity check that scans the neighbor's list of indices, with no allocation per link.
pub(super) fn validate_graph(name: &str, col: &VectorCollection) -> Result<(), StoreError> {
    checked_graph(name, col).map(|_| ())
}

/// [`validate_graph`], then the canonical order of every neighbor list: by graph distance
/// from the vector, then by id ([`cmp_distance_then_id`], the order `sort_layer` writes). The
/// snapshot encoder runs this before every snapshot it writes, and tests run it; it costs one
/// distance per link, so open runs [`validate_graph`] only.
///
/// Cost: the check dequantizes the whole collection into one `Vec<f32>` of `n * dim * 4`
/// bytes, so a checkpoint's peak memory is that much above the stored state: 1x the stored
/// bytes for f32, 2x for f16, 4x for i8.
pub(super) fn validate_graph_full(name: &str, col: &VectorCollection) -> Result<(), StoreError> {
    let Some(graph) = checked_graph(name, col)? else {
        return Ok(());
    };
    graph.check_order(col)
}

/// A collection's links as indices into its vectors, in id order.
struct GraphIndex<'a> {
    ids: Vec<&'a str>,
    records: Vec<&'a VectorRecord>,
    /// Every neighbor list, concatenated: vector `i`'s list on `layer` is
    /// `links[spans[first[i] + layer]]`.
    links: Vec<u32>,
    spans: Vec<std::ops::Range<usize>>,
    first: Vec<usize>,
}

impl GraphIndex<'_> {
    fn list(&self, node: usize, layer: usize) -> &[u32] {
        &self.links[self.spans[self.first[node] + layer].clone()]
    }

    /// The order check of [`validate_graph_full`]. It dequantizes every vector into one
    /// `Vec<f32>` (`n * dim * 4` bytes) that lives for the whole check: on every checkpoint
    /// that is extra peak memory of 1x the stored vector bytes for f32, 2x for f16, 4x for i8.
    fn check_order(&self, col: &VectorCollection) -> Result<(), StoreError> {
        let dim = usize::try_from(col.dim).map_err(|_| corrupt("vector dim does not fit"))?;
        let mut values = Vec::with_capacity(self.records.len().saturating_mul(dim));
        for record in &self.records {
            values.extend(dequantize(col.storage, &record.bytes).map_err(corrupt)?);
        }
        let point = |node: u32| {
            let at = node as usize * dim;
            &values[at..at + dim]
        };
        for (node, record) in self.records.iter().enumerate() {
            let origin = point(node as u32);
            for layer in 0..record.neighbors.len() {
                let mut previous: Option<(f64, &str)> = None;
                for &other in self.list(node, layer) {
                    let dist = graph_distance(col.metric, origin, point(other));
                    let other_id = self.ids[other as usize];
                    if let Some((prev_dist, prev_id)) = previous
                        && cmp_distance_then_id(prev_dist, prev_id, dist, other_id)
                            != Ordering::Less
                    {
                        return Err(corrupt(format!(
                            "vector `{}` neighbors on layer {layer} are not ordered by distance: \
                             the collection's neighbor order is not canonical and must be \
                             rebuilt (re-sort needed)",
                            self.ids[node]
                        )));
                    }
                    previous = Some((dist, other_id));
                }
            }
        }
        Ok(())
    }
}

/// The checks of [`validate_graph`]; the index they built, `None` for an empty collection.
fn checked_graph<'a>(
    name: &str,
    col: &'a VectorCollection,
) -> Result<Option<GraphIndex<'a>>, StoreError> {
    check_create(
        name,
        col.dim,
        col.storage,
        col.m,
        col.ef_construction,
        col.ef_search,
    )
    .map_err(corrupt)?;
    if col.vectors.is_empty() {
        if col.entry.is_some() {
            return Err(corrupt("empty vector collection has an entry point"));
        }
        return Ok(None);
    }
    let Some(entry) = &col.entry else {
        return Err(corrupt("vector collection is missing an entry point"));
    };
    if !col.vectors.contains_key(entry) {
        return Err(corrupt("vector entry point is not live"));
    }
    if col.entry.as_deref() != canonical_entry(&col.vectors).as_deref() {
        return Err(corrupt("vector entry point is not canonical"));
    }
    let count = col.vectors.len();
    if u32::try_from(count).is_err() {
        return Err(corrupt("too many vectors in one collection"));
    }
    let index: HashMap<&str, u32> = col
        .vectors
        .keys()
        .enumerate()
        .map(|(at, id)| (id.as_str(), at as u32))
        .collect();
    let levels: Vec<u8> = col.vectors.values().map(|rec| rec.level).collect();
    let mut graph = GraphIndex {
        ids: Vec::with_capacity(count),
        records: Vec::with_capacity(count),
        links: Vec::new(),
        spans: Vec::with_capacity(count),
        first: Vec::with_capacity(count),
    };
    // `seen[j] == list` when vector `j` is already in the list being checked: a repeat check
    // without a set per list.
    let mut seen = vec![usize::MAX; count];
    for (node, (id, record)) in col.vectors.iter().enumerate() {
        check_vector_id(id).map_err(corrupt)?;
        check_stored(col.storage, col.dim, &record.bytes).map_err(corrupt)?;
        if record.level != vector_level(id, col.m) || record.level > MAX_LEVEL {
            return Err(corrupt(format!("vector `{id}` level does not match")));
        }
        if record.neighbors.len() != usize::from(record.level) + 1 {
            return Err(corrupt(format!(
                "vector `{id}` layer count does not match its level"
            )));
        }
        graph.ids.push(id);
        graph.records.push(record);
        graph.first.push(graph.spans.len());
        for (layer, nids) in record.neighbors.iter().enumerate() {
            if nids.len() > max_links(col.m, layer) {
                return Err(corrupt(format!(
                    "vector `{id}` has too many neighbors on layer {layer}"
                )));
            }
            let list = graph.spans.len();
            let start = graph.links.len();
            for nid in nids {
                // A self link is to a live id, so the lookup finds it first.
                let Some(&other) = index.get(nid.as_str()) else {
                    return Err(corrupt(format!("vector neighbor `{nid}` is not live")));
                };
                let at = other as usize;
                if at == node {
                    return Err(corrupt(format!("vector `{id}` links to itself")));
                }
                if seen[at] == list {
                    return Err(corrupt(format!("vector `{id}` repeats neighbor `{nid}`")));
                }
                seen[at] = list;
                if usize::from(levels[at]) < layer {
                    return Err(corrupt(format!(
                        "vector neighbor `{nid}` does not exist on layer {layer}"
                    )));
                }
                graph.links.push(other);
            }
            graph.spans.push(start..graph.links.len());
        }
    }
    for node in 0..count {
        for layer in 0..graph.records[node].neighbors.len() {
            for &other in graph.list(node, layer) {
                // `other` exists on `layer`: checked above.
                if !graph.list(other as usize, layer).contains(&(node as u32)) {
                    return Err(corrupt(format!(
                        "vector link `{}` -> `{}` is not bidirectional",
                        graph.ids[node], graph.ids[other as usize]
                    )));
                }
            }
        }
    }
    Ok(Some(graph))
}

fn finite_number(value: f64, what: &str) -> Result<Value, StoreError> {
    let number = Number::from_f64(value).ok_or_else(|| backend(format!("{what} is not finite")))?;
    Ok(Value::Number(number))
}

pub(super) fn values_json(values: &[f32]) -> Result<Value, StoreError> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        if !value.is_finite() {
            return Err(backend("vector component is not finite"));
        }
        out.push(finite_number(f64::from(*value), "vector component")?);
    }
    Ok(Value::Array(out))
}

pub(super) fn parse_values(value: &Value) -> Result<Vec<f32>, StoreError> {
    let items = value
        .as_array()
        .ok_or_else(|| backend("vector values are not an array"))?;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let number = item
            .as_f64()
            .ok_or_else(|| backend("vector component is not a number"))?;
        if !number.is_finite() {
            return Err(backend("vector component is not finite"));
        }
        let value = number as f32;
        if !value.is_finite() {
            return Err(backend("vector component is not finite"));
        }
        out.push(value);
    }
    Ok(out)
}

pub(super) fn hits_json(hits: &[(String, f64)]) -> Result<Value, StoreError> {
    let mut out = Vec::with_capacity(hits.len());
    for (id, score) in hits {
        out.push(json!({
            "id": id,
            "score": finite_number(*score, "score")?,
        }));
    }
    Ok(Value::Array(out))
}

pub(super) fn parse_hits(value: &Value) -> Result<Vec<(String, f64)>, StoreError> {
    let items = value
        .as_array()
        .ok_or_else(|| backend("result is not an array"))?;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let id = item
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| backend("search hit id is not a string"))?
            .to_string();
        let score = item
            .get("score")
            .and_then(Value::as_f64)
            .ok_or_else(|| backend("search hit score is not a number"))?;
        if !score.is_finite() {
            return Err(backend("search hit score is not finite"));
        }
        out.push((id, score));
    }
    Ok(out)
}

pub(super) fn knowledge_json(hits: &[KnowledgeHit]) -> Result<Value, StoreError> {
    let mut out = Vec::with_capacity(hits.len());
    for hit in hits {
        out.push(json!({
            "id": hit.id,
            "score": finite_number(hit.score, "score")?,
            "metadata": hit.metadata.clone().unwrap_or(Value::Null),
            "doc": hit.doc.clone().unwrap_or(Value::Null),
        }));
    }
    Ok(Value::Array(out))
}

pub(super) fn parse_knowledge(value: &Value) -> Result<Vec<KnowledgeHit>, StoreError> {
    let items = value
        .as_array()
        .ok_or_else(|| backend("result is not an array"))?;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let id = item
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| backend("knowledge hit id is not a string"))?
            .to_string();
        let score = item
            .get("score")
            .and_then(Value::as_f64)
            .ok_or_else(|| backend("knowledge hit score is not a number"))?;
        if !score.is_finite() {
            return Err(backend("knowledge hit score is not finite"));
        }
        let metadata = match item.get("metadata") {
            None | Some(Value::Null) => None,
            Some(metadata) => Some(metadata.clone()),
        };
        let doc = match item.get("doc") {
            None | Some(Value::Null) => None,
            Some(doc) => Some(doc.clone()),
        };
        out.push(KnowledgeHit {
            id,
            score,
            metadata,
            doc,
        });
    }
    Ok(out)
}

pub(super) fn fusion_from_json(value: &Value) -> Result<Fusion, StoreError> {
    let method = value
        .get("method")
        .and_then(Value::as_str)
        .ok_or_else(|| backend("fusion is missing method"))?;
    let fusion = match method {
        "rrf" => {
            let k = match value.get("k") {
                None => 60,
                Some(k) => {
                    let k = k
                        .as_u64()
                        .ok_or_else(|| backend("rrf k is not an integer"))?;
                    u32::try_from(k).map_err(|_| backend("rrf k does not fit"))?
                }
            };
            Fusion::Rrf { k }
        }
        "weighted" => {
            let bm25 = value
                .get("bm25")
                .and_then(Value::as_f64)
                .ok_or_else(|| backend("weighted fusion is missing bm25"))?;
            let vector = value
                .get("vector")
                .and_then(Value::as_f64)
                .ok_or_else(|| backend("weighted fusion is missing vector"))?;
            Fusion::Weighted { bm25, vector }
        }
        other => return Err(backend(format!("unknown fusion method `{other}`"))),
    };
    validate_fusion(fusion)?;
    Ok(fusion)
}

pub(super) fn vector_get_json(
    values: &[f32],
    metadata: &Option<Value>,
) -> Result<Value, StoreError> {
    let mut item = Map::new();
    item.insert("values".into(), values_json(values)?);
    if let Some(metadata) = metadata {
        item.insert("metadata".into(), metadata.clone());
    }
    Ok(Value::Object(item))
}

#[allow(clippy::type_complexity)]
pub(super) fn parse_get(value: &Value) -> Result<Option<(Vec<f32>, Option<Value>)>, StoreError> {
    if value.is_null() {
        return Ok(None);
    }
    let values = parse_values(
        value
            .get("values")
            .ok_or_else(|| backend("vector is missing values"))?,
    )?;
    let metadata = value.get("metadata").cloned();
    Ok(Some((values, metadata)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-dimensional collection (`m = 1`, so a layer-0 node holds at most 2 links) with the
    /// given `(id, value, links)`.
    fn graph(nodes: &[(&str, f32, &[&str])]) -> VectorCollection {
        let mut col = VectorCollection::empty(1, VectorStorage::F32, VectorMetric::L2, 1, 8, 8);
        for (id, value, links) in nodes {
            col.vectors.insert(
                (*id).to_string(),
                VectorRecord {
                    bytes: quantize(VectorStorage::F32, &[*value]).unwrap(),
                    metadata: None,
                    level: 0,
                    neighbors: vec![links.iter().map(|l| (*l).to_string()).collect()],
                },
            );
        }
        col
    }

    /// A graph the ops built: 60 vectors at `m = 2`, so nodes reach layer 1 and above.
    fn built() -> VectorCollection {
        let mut col = VectorCollection::empty(3, VectorStorage::F32, VectorMetric::L2, 2, 8, 8);
        let mut x = 0x9e37_79b9_7f4a_7c15_u64;
        for n in 0..60 {
            let values: Vec<f32> = (0..3)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    (x % 1000) as f32 / 100.0
                })
                .collect();
            let bytes = quantize(VectorStorage::F32, &values).unwrap();
            col.upsert(format!("n{n:02}"), bytes, None).unwrap();
        }
        validate_graph_full("v", &col).unwrap();
        col
    }

    /// Both checks refuse `col` as corrupt, naming `what`.
    fn refused(col: &VectorCollection, what: &str) {
        for check in [validate_graph, validate_graph_full] {
            let err = check("v", col).unwrap_err();
            assert!(err.to_string().contains("corrupt"), "{what}: {err}");
            assert!(err.to_string().contains(what), "{what}: {err}");
        }
    }

    /// A node of at least `level` with room for one more link on `layer`, and a node on
    /// `layer` that it does not link to.
    fn with_room(col: &VectorCollection, level: u8, layer: usize) -> (String, String) {
        for (id, rec) in &col.vectors {
            if rec.level < level || rec.neighbors[layer].len() >= max_links(col.m, layer) {
                continue;
            }
            let stranger = col.vectors.iter().find(|(other, other_rec)| {
                *other != id
                    && usize::from(other_rec.level) >= layer
                    && !rec.neighbors[layer].contains(other)
            });
            if let Some((other, _)) = stranger {
                return (id.clone(), other.clone());
            }
        }
        panic!("no node with room on layer {layer}");
    }

    fn list<'a>(col: &'a mut VectorCollection, id: &str, layer: usize) -> &'a mut Vec<String> {
        &mut col.vectors.get_mut(id).unwrap().neighbors[layer]
    }

    /// Each structural fault on its own, in a graph that is otherwise sound: open's check
    /// (`validate_graph`) refuses every one, and so does the full check.
    #[test]
    fn the_open_check_refuses_every_structural_fault() {
        let good = built();
        validate_graph("v", &good).unwrap();
        let (id, stranger) = with_room(&good, 0, 0);

        let mut col = good.clone();
        list(&mut col, &id, 0).push(id.clone());
        refused(&col, "links to itself");

        let mut col = good.clone();
        let first = col.vectors[&id].neighbors[0][0].clone();
        list(&mut col, &id, 0).push(first);
        refused(&col, "repeats neighbor");

        let mut col = good.clone();
        list(&mut col, &id, 0).push("ghost".into());
        refused(&col, "is not live");

        let mut col = good.clone();
        list(&mut col, &id, 0).push(stranger.clone());
        refused(&col, "is not bidirectional");

        let mut col = good.clone();
        *list(&mut col, &id, 0) = (0..=max_links(2, 0)).map(|n| format!("g{n}")).collect();
        refused(&col, "too many neighbors");

        // The same on an upper layer, where the limit is `m`, not `2 * m`: a list one over it
        // is still under layer 0's limit, so a check that used that limit would accept it.
        assert!(max_links(2, 1) < max_links(2, 0));
        let (upper, _) = with_room(&good, 1, 1);
        assert!(good.vectors[&upper].level >= 1);
        let mut col = good.clone();
        *list(&mut col, &upper, 1) = (0..=max_links(2, 1)).map(|n| format!("g{n}")).collect();
        assert!(col.vectors[&upper].neighbors[1].len() <= max_links(2, 0));
        refused(&col, "too many neighbors on layer 1");

        // A layer-1 node linking to a node that stops at layer 0.
        let (upper, _) = with_room(&good, 1, 1);
        let ground = good
            .vectors
            .iter()
            .find(|(_, rec)| rec.level == 0)
            .map(|(other, _)| other.clone())
            .unwrap();
        let mut col = good.clone();
        list(&mut col, &upper, 1).push(ground.clone());
        refused(&col, "does not exist on layer 1");

        // `ground` one level up stays below the entry, so the entry is still canonical.
        let entry_level = good.vectors[good.entry.as_ref().unwrap()].level;
        assert!(entry_level >= 2, "the entry is at level {entry_level}");
        let mut col = good.clone();
        let rec = col.vectors.get_mut(&ground).unwrap();
        rec.level += 1;
        rec.neighbors.push(Vec::new());
        refused(&col, "level does not match");

        let mut col = good.clone();
        col.vectors
            .get_mut(&ground)
            .unwrap()
            .neighbors
            .push(Vec::new());
        refused(&col, "layer count does not match");

        let mut col = good.clone();
        col.vectors.get_mut(&ground).unwrap().bytes = [f32::NAN; 3]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        refused(&col, "not finite");

        let mut col = good.clone();
        col.entry = None;
        refused(&col, "missing an entry point");

        let mut col = good.clone();
        col.entry = Some("ghost".into());
        refused(&col, "entry point is not live");
    }

    /// A list out of order passes open's check, and the full check and the snapshot encoder
    /// refuse it with a message that says the order is not canonical and a re-sort is needed.
    #[test]
    fn a_misordered_list_fails_the_full_check_and_the_encoder_with_a_re_sort_message() {
        let good = built();
        let id = good
            .vectors
            .iter()
            .find(|(_, rec)| rec.neighbors[0].len() >= 2)
            .map(|(id, _)| id.clone())
            .unwrap();
        let mut col = good.clone();
        list(&mut col, &id, 0).reverse();
        validate_graph("v", &col).unwrap();
        let vectors: super::super::state::Vectors =
            BTreeMap::from([("v".to_string(), std::sync::Arc::new(col.clone()))]);
        let encoded = super::super::layout::encode_snapshot_v3(
            1,
            &Default::default(),
            &Default::default(),
            &vectors,
        );
        for err in [
            validate_graph_full("v", &col).unwrap_err(),
            encoded.unwrap_err(),
        ] {
            let text = err.to_string();
            assert!(text.contains("corrupt"), "{text}");
            assert!(text.contains(&format!("vector `{id}`")), "{text}");
            assert!(text.contains("not ordered by distance"), "{text}");
            assert!(text.contains("not canonical"), "{text}");
            assert!(text.contains("re-sort needed"), "{text}");
        }
    }

    /// The f16 finiteness check reads the bits: exponent 31 (infinity, NaN) is refused, the
    /// largest finite value and a subnormal are not.
    #[test]
    fn the_stored_f16_check_refuses_exactly_the_non_finite_bit_patterns() {
        for (bits, finite) in [
            (0x7bff_u16, true),
            (0x0001, true),
            (0x8000, true),
            (0x7c00, false),
            (0xfc00, false),
            (0x7e00, false),
        ] {
            let result = check_stored(VectorStorage::F16, 1, &bits.to_le_bytes());
            assert_eq!(result.is_ok(), finite, "{bits:#06x}");
            assert_eq!(f16_to_f32(bits).is_finite(), finite, "{bits:#06x}");
        }
    }

    fn links(col: &VectorCollection, id: &str) -> Vec<String> {
        let mut out = col.vectors[id].neighbors[0].clone();
        out.sort();
        out
    }

    /// The entry an upsert keeps incrementally is the one a rescan finds, after every step of
    /// random upserts (new ids and replacements) and deletes (the entry's included). `m = 2`
    /// spreads the levels, so the entry moves often; a small id pool makes replacements and
    /// deletes of the entry common.
    #[test]
    fn the_incremental_entry_is_always_the_canonical_entry() {
        let mut entry_deletes = 0;
        let mut moved_by_upsert = 0;
        for seed in 1..=40u64 {
            let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
            let mut next = move || {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            };
            let mut col = VectorCollection::empty(3, VectorStorage::F32, VectorMetric::L2, 2, 8, 8);
            for step in 0..150 {
                let id = format!("n{}", next() % 48);
                if next() % 4 == 0 {
                    if col.entry.as_deref() == Some(id.as_str()) {
                        entry_deletes += 1;
                    }
                    col.delete_id(&id).unwrap();
                } else {
                    let values: Vec<f32> = (0..3).map(|_| (next() % 1000) as f32 / 100.0).collect();
                    let bytes = quantize(VectorStorage::F32, &values).unwrap();
                    let before = col.entry.clone();
                    col.upsert(id, bytes, None).unwrap();
                    if before.is_some() && col.entry != before {
                        moved_by_upsert += 1;
                    }
                }
                assert_eq!(
                    col.entry,
                    canonical_entry(&col.vectors),
                    "seed {seed}, step {step}"
                );
            }
            validate_graph_full("v", &col).unwrap();
        }
        // The two paths the incremental rule must get right were both taken.
        assert!(entry_deletes > 0, "no delete of the entry");
        assert!(moved_by_upsert > 0, "no upsert moved the entry");
    }

    /// Review follow-up to audit S-2: `ensure_edge` used to make room on `a` (evicting a link on
    /// both sides) and only then find that `b` could not, leaving `a` one link short and no new
    /// edge. A refused edge must change nothing.
    #[test]
    fn a_refused_edge_leaves_both_nodes_as_they_were() {
        // `a` is full (2 links) and both its neighbours have a spare; `b` is full and both of
        // its neighbours are down to two links, so `b` cannot give one up.
        let col = graph(&[
            ("a", 0.0, &["p", "q"]),
            ("p", 1.0, &["a", "f1", "f2"]),
            ("q", 2.0, &["a", "f1", "f2"]),
            ("f1", 3.0, &["p", "q"]),
            ("f2", 4.0, &["p", "q"]),
            ("b", 10.0, &["r", "s"]),
            ("r", 9.0, &["b", "s"]),
            ("s", 11.0, &["b", "r"]),
        ]);
        let before = col.clone();
        let mut after = col;
        let mut touched = BTreeSet::new();
        after.ensure_edge("a", "b", 0, &mut touched).unwrap();
        for id in ["a", "p", "q", "f1", "f2", "b", "r", "s"] {
            assert_eq!(links(&after, id), links(&before, id), "links of {id}");
        }
        assert!(!links(&after, "a").iter().any(|l| l == "b"));
    }

    /// The edge is made when both sides can make room, and the evicted links go on both sides.
    #[test]
    fn an_edge_both_sides_can_make_room_for_is_made() {
        let col = graph(&[
            ("a", 0.0, &["p", "q"]),
            ("p", 1.0, &["a", "f1", "f2"]),
            ("q", 2.0, &["a", "f1", "f2"]),
            ("f1", 3.0, &["p", "q", "r", "s"]),
            ("f2", 4.0, &["p", "q"]),
            ("b", 10.0, &["r", "s"]),
            ("r", 9.0, &["b", "s", "f1"]),
            ("s", 11.0, &["b", "r", "f1"]),
        ]);
        let mut col = col;
        let mut touched = BTreeSet::new();
        col.ensure_edge("a", "b", 0, &mut touched).unwrap();
        assert!(links(&col, "a").iter().any(|l| l == "b"));
        assert!(links(&col, "b").iter().any(|l| l == "a"));
        assert_eq!(links(&col, "a").len(), 2);
        assert_eq!(links(&col, "b").len(), 2);
        // Whatever each side evicted is gone from the other end too.
        for id in ["a", "b", "p", "q", "r", "s"] {
            for other in &col.vectors[id].neighbors[0] {
                assert!(
                    col.vectors[other.as_str()].neighbors[0]
                        .iter()
                        .any(|l| l == id),
                    "{id} -> {other} is one-way"
                );
            }
        }
    }
}
