//! In-file vectors: f32 / f16 / i8 storage, HNSW, and hybrid fusion.
//!
//! Quantized bytes are what the WAL and the snapshot store. Distances run on
//! the dequantized `f32` values, accumulated in `f64`.

use super::columnar::check_name;
use fl_core::StoreError;
use serde_json::{Map, Number, Value, json};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashSet};

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
    dequantize(storage, bytes)?;
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
fn canonical_entry(vectors: &BTreeMap<String, VectorRecord>) -> Option<String> {
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
        self.recompute_entry();
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
        self.recompute_entry();
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

pub(super) fn validate_graph(name: &str, col: &VectorCollection) -> Result<(), StoreError> {
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
        return Ok(());
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
    for (id, record) in &col.vectors {
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
        let origin = dequantize(col.storage, &record.bytes).map_err(corrupt)?;
        for (layer, nids) in record.neighbors.iter().enumerate() {
            if nids.len() > max_links(col.m, layer) {
                return Err(corrupt(format!(
                    "vector `{id}` has too many neighbors on layer {layer}"
                )));
            }
            let mut seen = BTreeSet::new();
            for nid in nids {
                if nid == id {
                    return Err(corrupt(format!("vector `{id}` links to itself")));
                }
                if !seen.insert(nid.as_str()) {
                    return Err(corrupt(format!("vector `{id}` repeats neighbor `{nid}`")));
                }
                let Some(other) = col.vectors.get(nid) else {
                    return Err(corrupt(format!("vector neighbor `{nid}` is not live")));
                };
                if usize::from(other.level) < layer {
                    return Err(corrupt(format!(
                        "vector neighbor `{nid}` does not exist on layer {layer}"
                    )));
                }
                let reciprocal = other
                    .neighbors
                    .get(layer)
                    .is_some_and(|slot| slot.iter().any(|other_id| other_id == id));
                if !reciprocal {
                    return Err(corrupt(format!(
                        "vector link `{id}` -> `{nid}` is not bidirectional"
                    )));
                }
            }
            if nids.len() < 2 {
                continue;
            }
            let mut previous: Option<(f64, &str)> = None;
            for nid in nids {
                let Some(other) = col.vectors.get(nid) else {
                    return Err(corrupt(format!("vector neighbor `{nid}` is not live")));
                };
                let neighbor = dequantize(col.storage, &other.bytes).map_err(corrupt)?;
                let dist = graph_distance(col.metric, &origin, &neighbor);
                if let Some((prev_dist, prev_id)) = previous
                    && cmp_distance_then_id(prev_dist, prev_id, dist, nid) != Ordering::Less
                {
                    return Err(corrupt(format!(
                        "vector `{id}` neighbors on layer {layer} are not ordered by distance"
                    )));
                }
                previous = Some((dist, nid.as_str()));
            }
        }
    }
    Ok(())
}

fn bytes_json(bytes: &[u8]) -> Value {
    Value::Array(bytes.iter().copied().map(Value::from).collect())
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

pub(super) fn vectors_json(collections: &BTreeMap<String, VectorCollection>) -> Value {
    Value::Array(
        collections
            .iter()
            .map(|(name, col)| collection_json(name, col))
            .collect(),
    )
}

fn collection_json(name: &str, col: &VectorCollection) -> Value {
    let vectors = col
        .vectors
        .iter()
        .map(|(id, record)| {
            let mut item = Map::new();
            item.insert("id".into(), Value::String(id.clone()));
            item.insert("bytes".into(), bytes_json(&record.bytes));
            if let Some(metadata) = &record.metadata {
                item.insert("metadata".into(), metadata.clone());
            }
            Value::Object(item)
        })
        .collect::<Vec<_>>();
    let links = col
        .vectors
        .iter()
        .map(|(id, record)| {
            json!({
                "id": id,
                "level": record.level,
                "layers": record.neighbors.iter().map(|layer| {
                    Value::Array(layer.iter().cloned().map(Value::String).collect())
                }).collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();
    json!({
        "name": name,
        "dim": col.dim,
        "storage": storage_name(col.storage),
        "metric": metric_name(col.metric),
        "m": col.m,
        "ef_construction": col.ef_construction,
        "ef_search": col.ef_search,
        "vectors": vectors,
        "entry": col.entry,
        "links": links,
    })
}

pub(super) fn parse_vectors(
    value: &Value,
) -> Result<BTreeMap<String, VectorCollection>, StoreError> {
    let items = value
        .as_array()
        .ok_or_else(|| backend("vectors is not an array"))?;
    let mut collections = BTreeMap::new();
    for item in items {
        let name = item
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| backend("vector collection is missing name"))?
            .to_string();
        let col = parse_collection(item)?;
        validate_graph(&name, &col)?;
        if collections.insert(name.clone(), col).is_some() {
            return Err(backend(format!("duplicate vector collection `{name}`")));
        }
    }
    Ok(collections)
}

fn json_u32(value: &Value, key: &str) -> Result<u32, StoreError> {
    let number = value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| backend(format!("missing integer field `{key}`")))?;
    u32::try_from(number).map_err(|_| backend(format!("field `{key}` does not fit")))
}

fn parse_collection(value: &Value) -> Result<VectorCollection, StoreError> {
    let dim = json_u32(value, "dim")?;
    let storage = storage_from_name(
        value
            .get("storage")
            .and_then(Value::as_str)
            .ok_or_else(|| backend("vector collection is missing storage"))?,
    )?;
    let metric = metric_from_name(
        value
            .get("metric")
            .and_then(Value::as_str)
            .ok_or_else(|| backend("vector collection is missing metric"))?,
    )?;
    let m = json_u32(value, "m")?;
    let ef_construction = json_u32(value, "ef_construction")?;
    let ef_search = json_u32(value, "ef_search")?;
    let mut col = VectorCollection::empty(dim, storage, metric, m, ef_construction, ef_search);
    let vectors = value
        .get("vectors")
        .and_then(Value::as_array)
        .ok_or_else(|| backend("vector collection is missing vectors"))?;
    for item in vectors {
        let id = item
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| backend("vector is missing id"))?
            .to_string();
        let bytes = super::columnar::parse_bytes(
            item.get("bytes")
                .ok_or_else(|| backend("vector is missing bytes"))?,
        )?;
        let metadata = item.get("metadata").cloned();
        if col
            .vectors
            .insert(
                id.clone(),
                VectorRecord {
                    bytes,
                    metadata,
                    level: 0,
                    neighbors: Vec::new(),
                },
            )
            .is_some()
        {
            return Err(backend(format!("duplicate vector id `{id}`")));
        }
    }
    col.entry = match value.get("entry") {
        None | Some(Value::Null) => None,
        Some(Value::String(id)) => Some(id.clone()),
        _ => return Err(backend("vector entry is not a string")),
    };
    let links = value
        .get("links")
        .and_then(Value::as_array)
        .ok_or_else(|| backend("vector collection is missing links"))?;
    let mut seen = BTreeSet::new();
    for item in links {
        let id = item
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| backend("vector link is missing id"))?
            .to_string();
        if !seen.insert(id.clone()) {
            return Err(backend(format!("duplicate vector links for `{id}`")));
        }
        let level = item
            .get("level")
            .and_then(Value::as_u64)
            .ok_or_else(|| backend("vector link is missing level"))?;
        let level = u8::try_from(level).map_err(|_| backend("vector level does not fit"))?;
        let layers = item
            .get("layers")
            .and_then(Value::as_array)
            .ok_or_else(|| backend("vector link is missing layers"))?;
        let mut neighbors = Vec::with_capacity(layers.len());
        for layer in layers {
            let ids = layer
                .as_array()
                .ok_or_else(|| backend("vector layer is not an array"))?;
            let mut nids = Vec::with_capacity(ids.len());
            for nid in ids {
                nids.push(
                    nid.as_str()
                        .ok_or_else(|| backend("vector neighbor is not a string"))?
                        .to_string(),
                );
            }
            neighbors.push(nids);
        }
        let Some(record) = col.vectors.get_mut(&id) else {
            return Err(backend(format!("vector links name missing id `{id}`")));
        };
        record.level = level;
        record.neighbors = neighbors;
    }
    if seen.len() != col.vectors.len() {
        return Err(backend("vector links do not cover the collection"));
    }
    Ok(col)
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

    fn links(col: &VectorCollection, id: &str) -> Vec<String> {
        let mut out = col.vectors[id].neighbors[0].clone();
        out.sort();
        out
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
