//! FLEN v1 `RecordEnvelope` fuzz and boundary decoding: truncation, bit flips, oversized
//! records, invalid UTF-8, and unknown versions, kinds and flags.

use fl_store::{
    ColumnDef, ColumnType, EnvelopeKind, RecordEnvelope, SyncBatch, SyncOp, VectorMetric,
    VectorStorage,
};
use serde_json::{Value, json};

/// Deterministic xorshift so a failure reproduces from the seed alone.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn envelope(kind: EnvelopeKind) -> RecordEnvelope {
    RecordEnvelope {
        seq: 7,
        op_index: 2,
        kind,
        namespace: "ns".into(),
        record_id: "rid".into(),
        schema_version: 1,
        payload_bytes: br#"{"a":1}"#.to_vec(),
        metadata: Some(json!({"k": [1, 2, {"z": null}]})),
        vector_f32: Some(vec![0.5, -1.25, f32::MAX]),
    }
}

const KINDS: [EnvelopeKind; 7] = [
    EnvelopeKind::Document,
    EnvelopeKind::DeleteDocument,
    EnvelopeKind::ColumnTableSchema,
    EnvelopeKind::ColumnRowGroup,
    EnvelopeKind::VectorCollectionSchema,
    EnvelopeKind::VectorUpsert,
    EnvelopeKind::VectorDelete,
];

/// Hand-built frame so a test can place any byte anywhere.
struct FrameBuilder(Vec<u8>);

impl FrameBuilder {
    fn header(version: u16, kind: u8, schema: u16) -> Self {
        let mut out = b"FLEN".to_vec();
        out.extend_from_slice(&version.to_le_bytes());
        out.extend_from_slice(&1u64.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.push(kind);
        out.extend_from_slice(&schema.to_le_bytes());
        Self(out)
    }

    fn counted(mut self, bytes: &[u8]) -> Self {
        self.0
            .extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        self.0.extend_from_slice(bytes);
        self
    }

    fn raw_len(mut self, len: u32) -> Self {
        self.0.extend_from_slice(&len.to_le_bytes());
        self
    }

    fn byte(mut self, b: u8) -> Self {
        self.0.push(b);
        self
    }

    /// A valid frame body: ns "n", id "i", empty payload, no metadata, no vector.
    fn plain(version: u16, kind: u8, schema: u16) -> Self {
        Self::header(version, kind, schema)
            .counted(b"n")
            .counted(b"i")
            .counted(b"")
            .byte(0)
            .byte(0)
    }
}

#[test]
fn every_kind_round_trips_byte_for_byte() {
    for kind in KINDS {
        let original = envelope(kind);
        let bytes = original.encode().unwrap();
        let decoded = RecordEnvelope::decode(&bytes).unwrap();
        assert_eq!(decoded, original, "{kind:?}");
        assert_eq!(
            decoded.encode().unwrap(),
            bytes,
            "{kind:?} re-encode drifted"
        );
    }
}

#[test]
fn kind_tags_are_stable_on_the_wire() {
    for (tag, kind) in KINDS.iter().enumerate() {
        let bytes = envelope(*kind).encode().unwrap();
        // magic(4) + version(2) + seq(8) + op_index(4) = byte 18 is the kind tag.
        assert_eq!(bytes[18] as usize, tag, "{kind:?}");
    }
}

#[test]
fn the_frame_header_is_little_endian_and_versioned() {
    let bytes = envelope(EnvelopeKind::Document).encode().unwrap();
    assert_eq!(&bytes[..4], b"FLEN");
    assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), 1);
    assert_eq!(u64::from_le_bytes(bytes[6..14].try_into().unwrap()), 7);
    assert_eq!(u32::from_le_bytes(bytes[14..18].try_into().unwrap()), 2);
}

#[test]
fn every_truncation_of_every_kind_is_an_error_never_a_panic() {
    for kind in KINDS {
        let bytes = envelope(kind).encode().unwrap();
        for cut in 0..bytes.len() {
            assert!(
                RecordEnvelope::decode(&bytes[..cut]).is_err(),
                "{kind:?} cut at {cut}/{} decoded",
                bytes.len()
            );
        }
    }
}

#[test]
fn truncated_headers_name_a_truncation() {
    let bytes = envelope(EnvelopeKind::Document).encode().unwrap();
    for cut in [0, 1, 3, 4, 5, 6, 13, 17, 18, 19, 20] {
        let err = RecordEnvelope::decode(&bytes[..cut])
            .unwrap_err()
            .to_string();
        assert!(err.contains("truncated"), "cut {cut}: {err}");
    }
}

#[test]
fn trailing_bytes_are_refused() {
    let mut bytes = envelope(EnvelopeKind::Document).encode().unwrap();
    for extra in [1usize, 2, 4, 1024] {
        bytes.resize(bytes.len() + extra, 0);
        let err = RecordEnvelope::decode(&bytes).unwrap_err().to_string();
        assert!(err.contains("trailing"), "{err}");
        bytes.truncate(bytes.len() - extra);
    }
}

#[test]
fn bad_magic_is_refused_for_every_single_byte_change() {
    let good = envelope(EnvelopeKind::Document).encode().unwrap();
    for pos in 0..4 {
        for flip in [0x01u8, 0x20, 0x80, 0xFF] {
            let mut bytes = good.clone();
            bytes[pos] ^= flip;
            let err = RecordEnvelope::decode(&bytes).unwrap_err().to_string();
            assert!(err.contains("magic"), "pos {pos}: {err}");
        }
    }
}

#[test]
fn unknown_frame_versions_are_refused() {
    for version in [0u16, 2, 3, 255, 256, 0x7FFF, u16::MAX] {
        let bytes = FrameBuilder::plain(version, 0, 1).0;
        let err = RecordEnvelope::decode(&bytes).unwrap_err().to_string();
        assert!(err.contains("version"), "v{version}: {err}");
    }
    assert!(RecordEnvelope::decode(&FrameBuilder::plain(1, 0, 1).0).is_ok());
}

#[test]
fn unknown_kinds_are_refused_for_every_unassigned_tag() {
    for tag in 7u8..=255 {
        let bytes = FrameBuilder::plain(1, tag, 1).0;
        let err = RecordEnvelope::decode(&bytes).unwrap_err().to_string();
        assert!(err.contains("kind"), "tag {tag}: {err}");
    }
    for tag in 0u8..=6 {
        assert!(RecordEnvelope::decode(&FrameBuilder::plain(1, tag, 1).0).is_ok());
    }
}

#[test]
fn unknown_metadata_and_vector_flags_are_refused() {
    for flag in 2u8..=255 {
        let meta = FrameBuilder::header(1, 0, 1)
            .counted(b"n")
            .counted(b"i")
            .counted(b"")
            .byte(flag)
            .byte(0)
            .0;
        let err = RecordEnvelope::decode(&meta).unwrap_err().to_string();
        assert!(err.contains("metadata flag"), "{flag}: {err}");
        let vec = FrameBuilder::header(1, 0, 1)
            .counted(b"n")
            .counted(b"i")
            .counted(b"")
            .byte(0)
            .byte(flag)
            .0;
        let err = RecordEnvelope::decode(&vec).unwrap_err().to_string();
        assert!(err.contains("vector flag"), "{flag}: {err}");
    }
}

#[test]
fn invalid_utf8_in_namespace_or_record_id_is_refused() {
    let bad: [&[u8]; 6] = [
        &[0xFF],
        &[0xC0, 0x80],             // overlong NUL
        &[0xED, 0xA0, 0x80],       // lone surrogate
        &[0xF4, 0x90, 0x80, 0x80], // beyond U+10FFFF
        &[0xE2, 0x82],             // truncated sequence
        b"ok\xFFok",
    ];
    for field in &bad {
        let ns = FrameBuilder::header(1, 0, 1)
            .counted(field)
            .counted(b"i")
            .counted(b"")
            .byte(0)
            .byte(0)
            .0;
        let err = RecordEnvelope::decode(&ns).unwrap_err().to_string();
        assert!(err.contains("utf-8"), "ns {field:?}: {err}");
        let id = FrameBuilder::header(1, 0, 1)
            .counted(b"n")
            .counted(field)
            .counted(b"")
            .byte(0)
            .byte(0)
            .0;
        let err = RecordEnvelope::decode(&id).unwrap_err().to_string();
        assert!(err.contains("utf-8"), "id {field:?}: {err}");
    }
}

#[test]
fn valid_multibyte_and_control_text_round_trips() {
    for text in [
        "héllo",
        "日本語",
        "🦀🦀🦀",
        "a\u{0}b",
        "\u{10FFFF}",
        "tab\tnl\n",
        "",
    ] {
        let mut envelope = envelope(EnvelopeKind::Document);
        envelope.namespace = text.to_string();
        envelope.record_id = text.chars().rev().collect();
        let decoded = RecordEnvelope::decode(&envelope.encode().unwrap()).unwrap();
        assert_eq!(decoded, envelope, "{text:?}");
    }
}

#[test]
fn metadata_that_is_not_json_is_refused() {
    for raw in [&b"{"[..], b"", b"\xFF\xFE", b"{\"a\":}", b"nul", b"1 2"] {
        let bytes = FrameBuilder::header(1, 5, 1)
            .counted(b"n")
            .counted(b"i")
            .counted(b"")
            .byte(1)
            .counted(raw)
            .byte(0)
            .0;
        let err = RecordEnvelope::decode(&bytes).unwrap_err().to_string();
        assert!(err.contains("metadata is not json"), "{raw:?}: {err}");
    }
}

#[test]
fn metadata_null_is_distinct_from_absent_metadata() {
    let mut with_null = envelope(EnvelopeKind::VectorUpsert);
    with_null.metadata = Some(Value::Null);
    let mut absent = with_null.clone();
    absent.metadata = None;
    let a = RecordEnvelope::decode(&with_null.encode().unwrap()).unwrap();
    let b = RecordEnvelope::decode(&absent.encode().unwrap()).unwrap();
    assert_eq!(a.metadata, Some(Value::Null));
    assert_eq!(b.metadata, None);
}

#[test]
fn hostile_length_prefixes_never_allocate_or_panic() {
    // namespace length claims 4 GiB - 1 with nothing behind it.
    for len in [1u32, 100, 1 << 20, 1 << 31, u32::MAX - 1, u32::MAX] {
        let bytes = FrameBuilder::header(1, 0, 1).raw_len(len).0;
        let err = RecordEnvelope::decode(&bytes).unwrap_err().to_string();
        assert!(err.contains("truncated"), "len {len}: {err}");
    }
    // payload and metadata length lies.
    for len in [u32::MAX, 1 << 30] {
        let payload = FrameBuilder::header(1, 3, 1)
            .counted(b"n")
            .counted(b"i")
            .raw_len(len)
            .0;
        assert!(RecordEnvelope::decode(&payload).is_err());
        let meta = FrameBuilder::header(1, 5, 1)
            .counted(b"n")
            .counted(b"i")
            .counted(b"")
            .byte(1)
            .raw_len(len)
            .0;
        assert!(RecordEnvelope::decode(&meta).is_err());
    }
    // vector float count lies: count * 4 must not overflow or allocate.
    for count in [
        1u32,
        1 << 20,
        1 << 30,
        u32::MAX / 4,
        u32::MAX / 4 + 1,
        u32::MAX,
    ] {
        let bytes = FrameBuilder::header(1, 5, 1)
            .counted(b"n")
            .counted(b"i")
            .counted(b"")
            .byte(0)
            .byte(1)
            .raw_len(count)
            .0;
        let err = RecordEnvelope::decode(&bytes).unwrap_err().to_string();
        assert!(err.contains("truncated"), "count {count}: {err}");
    }
}

#[test]
fn a_vector_count_one_short_of_the_data_is_refused_and_exact_is_accepted() {
    let build = |count: u32, floats: usize| {
        let mut b = FrameBuilder::header(1, 5, 1)
            .counted(b"n")
            .counted(b"i")
            .counted(b"")
            .byte(0)
            .byte(1)
            .raw_len(count);
        for _ in 0..floats {
            b.0.extend_from_slice(&1.0f32.to_le_bytes());
        }
        b.0
    };
    assert_eq!(
        RecordEnvelope::decode(&build(5, 5)).unwrap().vector_f32,
        Some(vec![1.0; 5])
    );
    assert!(RecordEnvelope::decode(&build(5, 4)).is_err(), "short data");
    assert!(
        RecordEnvelope::decode(&build(4, 5)).is_err(),
        "trailing float"
    );
    assert_eq!(
        RecordEnvelope::decode(&build(0, 0)).unwrap().vector_f32,
        Some(Vec::new())
    );
}

#[test]
fn special_floats_survive_the_frame_bit_for_bit() {
    let values = [
        0.0f32,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE,
        f32::from_bits(1), // smallest subnormal
        f32::MAX,
        f32::MIN,
    ];
    let mut envelope = envelope(EnvelopeKind::VectorUpsert);
    envelope.vector_f32 = Some(values.to_vec());
    let decoded = RecordEnvelope::decode(&envelope.encode().unwrap()).unwrap();
    let got = decoded.vector_f32.unwrap();
    assert_eq!(got.len(), values.len());
    for (a, b) in got.iter().zip(values) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn large_records_round_trip() {
    for size in [0usize, 1, 255, 256, 65_535, 65_536, 1 << 20, 16 << 20] {
        let mut envelope = envelope(EnvelopeKind::ColumnRowGroup);
        envelope.payload_bytes = (0..size).map(|i| (i % 251) as u8).collect();
        envelope.metadata = None;
        envelope.vector_f32 = None;
        let bytes = envelope.encode().unwrap();
        let decoded = RecordEnvelope::decode(&bytes).unwrap();
        assert_eq!(decoded.payload_bytes.len(), size);
        assert_eq!(decoded, envelope);
    }
}

#[test]
fn a_large_vector_round_trips() {
    let mut envelope = envelope(EnvelopeKind::VectorUpsert);
    envelope.vector_f32 = Some((0..200_000).map(|i| i as f32 * 0.25).collect());
    let decoded = RecordEnvelope::decode(&envelope.encode().unwrap()).unwrap();
    assert_eq!(decoded, envelope);
}

#[test]
fn extreme_scalar_header_values_round_trip() {
    let mut envelope = envelope(EnvelopeKind::Document);
    envelope.seq = u64::MAX;
    envelope.op_index = u32::MAX;
    envelope.schema_version = u16::MAX;
    let decoded = RecordEnvelope::decode(&envelope.encode().unwrap()).unwrap();
    assert_eq!(decoded, envelope);
}

#[test]
fn every_single_bit_flip_either_errors_or_decodes_without_panicking() {
    for kind in KINDS {
        let good = envelope(kind).encode().unwrap();
        for byte in 0..good.len() {
            for bit in 0..8 {
                let mut bytes = good.clone();
                bytes[byte] ^= 1 << bit;
                // Any outcome except a panic or an abort is acceptable. A frame that
                // still decodes must re-encode to exactly the bytes it came from,
                // unless the flip landed inside a float's NaN payload (also exact).
                if let Ok(decoded) = RecordEnvelope::decode(&bytes) {
                    assert_eq!(
                        decoded.encode().unwrap(),
                        bytes,
                        "{kind:?} byte {byte} bit {bit}: decode is not canonical"
                    );
                }
            }
        }
    }
}

#[test]
fn flips_in_the_structural_prefix_are_always_detected() {
    let good = envelope(EnvelopeKind::Document).encode().unwrap();
    // magic (0..4) and version (4..6) have no valid neighbours.
    for byte in 0..6 {
        for bit in 0..8 {
            let mut bytes = good.clone();
            bytes[byte] ^= 1 << bit;
            assert!(
                RecordEnvelope::decode(&bytes).is_err(),
                "byte {byte} bit {bit}"
            );
        }
    }
}

#[test]
fn random_garbage_never_panics() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for round in 0..20_000 {
        let len = (rng.next() % 200) as usize;
        let mut bytes: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
        // Half the time give it a valid prefix so the fuzz reaches the field decoders.
        if round % 2 == 0 && len >= 6 {
            bytes[..4].copy_from_slice(b"FLEN");
            bytes[4..6].copy_from_slice(&1u16.to_le_bytes());
        }
        let _ = RecordEnvelope::decode(&bytes);
    }
}

#[test]
fn mutated_valid_frames_never_panic() {
    let mut rng = Rng(0xDEAD_BEEF_CAFE_F00D);
    for kind in KINDS {
        let good = envelope(kind).encode().unwrap();
        for _ in 0..5_000 {
            let mut bytes = good.clone();
            for _ in 0..(1 + rng.next() % 4) {
                let at = (rng.next() as usize) % bytes.len();
                bytes[at] = rng.next() as u8;
            }
            if rng.next().is_multiple_of(4) {
                bytes.truncate((rng.next() as usize) % (bytes.len() + 1));
            }
            let _ = RecordEnvelope::decode(&bytes);
        }
    }
}

#[test]
fn decode_is_safe_across_threads() {
    let good = envelope(EnvelopeKind::VectorUpsert).encode().unwrap();
    std::thread::scope(|scope| {
        for t in 0..8u64 {
            let good = &good;
            scope.spawn(move || {
                let mut rng = Rng(0x1234_5678 + t);
                for _ in 0..2_000 {
                    let mut bytes = good.clone();
                    let at = (rng.next() as usize) % bytes.len();
                    bytes[at] ^= 1 << (rng.next() % 8);
                    let _ = RecordEnvelope::decode(&bytes);
                    assert!(RecordEnvelope::decode(good).is_ok());
                }
            });
        }
    });
}

// ---- batch <-> envelope conversion ---------------------------------------------------

fn every_op() -> Vec<SyncOp> {
    vec![
        SyncOp::PutDoc {
            collection: "docs".into(),
            id: "a".into(),
            doc: json!({"x": [1, 2, 3], "s": "ü"}),
        },
        SyncOp::DeleteDoc {
            collection: "docs".into(),
            id: "a".into(),
        },
        SyncOp::CreateColumnTable {
            name: "t".into(),
            columns: vec![
                ColumnDef {
                    name: "n".into(),
                    ty: ColumnType::Int64,
                },
                ColumnDef {
                    name: "s".into(),
                    ty: ColumnType::Utf8,
                },
            ],
        },
        SyncOp::AppendColumnGroup {
            name: "t".into(),
            bytes: vec![1, 2, 3, 4],
        },
        SyncOp::CreateVectorCollection {
            name: "v".into(),
            dim: 3,
            storage: VectorStorage::F16,
            metric: VectorMetric::DotProduct,
            m: 8,
            ef_construction: 32,
            ef_search: 16,
        },
        SyncOp::UpsertVector {
            collection: "v".into(),
            id: "x".into(),
            bytes: vec![0, 1, 2, 3, 4, 5],
            metadata: Some(Value::Null),
        },
        SyncOp::DeleteVector {
            collection: "v".into(),
            id: "x".into(),
        },
    ]
}

#[test]
fn a_batch_of_every_op_round_trips_through_encoded_frames() {
    let batch = SyncBatch {
        seq: 42,
        ops: every_op(),
    };
    let frames: Vec<Vec<u8>> = batch
        .to_envelopes()
        .unwrap()
        .iter()
        .map(|e| e.encode().unwrap())
        .collect();
    let decoded: Vec<RecordEnvelope> = frames
        .iter()
        .map(|f| RecordEnvelope::decode(f).unwrap())
        .collect();
    for (index, envelope) in decoded.iter().enumerate() {
        assert_eq!(envelope.op_index as usize, index);
        assert_eq!(envelope.seq, 42);
        assert_eq!(envelope.schema_version, 1);
    }
    assert_eq!(SyncBatch::from_envelopes(42, &decoded).unwrap(), batch);
}

#[test]
fn envelope_order_does_not_matter_but_density_does() {
    let batch = SyncBatch {
        seq: 3,
        ops: every_op(),
    };
    let mut envelopes = batch.to_envelopes().unwrap();
    envelopes.reverse();
    assert_eq!(SyncBatch::from_envelopes(3, &envelopes).unwrap(), batch);

    let mut dup = batch.to_envelopes().unwrap();
    dup[1].op_index = 0;
    assert!(
        SyncBatch::from_envelopes(3, &dup).is_err(),
        "duplicate index"
    );

    let mut gap = batch.to_envelopes().unwrap();
    gap[2].op_index = 99;
    assert!(SyncBatch::from_envelopes(3, &gap).is_err(), "gap");

    let mut wrong_seq = batch.to_envelopes().unwrap();
    wrong_seq[4].seq = 4;
    let err = SyncBatch::from_envelopes(3, &wrong_seq)
        .unwrap_err()
        .to_string();
    assert!(err.contains("does not match"), "{err}");

    assert_eq!(
        SyncBatch::from_envelopes(9, &[]).unwrap(),
        SyncBatch {
            seq: 9,
            ops: Vec::new()
        }
    );
}

#[test]
fn an_unsupported_schema_version_is_refused_when_rebuilding_ops() {
    for schema in [0u16, 2, 99, u16::MAX] {
        let mut envelopes = SyncBatch {
            seq: 1,
            ops: every_op(),
        }
        .to_envelopes()
        .unwrap();
        envelopes[0].schema_version = schema;
        let err = SyncBatch::from_envelopes(1, &envelopes)
            .unwrap_err()
            .to_string();
        assert!(err.contains("schema version"), "{schema}: {err}");
    }
}

#[test]
fn corrupt_payloads_inside_a_valid_frame_are_refused_per_kind() {
    let rebuild = |kind: EnvelopeKind, payload: &[u8]| {
        let envelope = RecordEnvelope {
            seq: 1,
            op_index: 0,
            kind,
            namespace: "n".into(),
            record_id: "i".into(),
            schema_version: 1,
            payload_bytes: payload.to_vec(),
            metadata: None,
            vector_f32: None,
        };
        // The frame itself must be well formed; only the payload is wrong.
        let decoded = RecordEnvelope::decode(&envelope.encode().unwrap()).unwrap();
        SyncBatch::from_envelopes(1, &[decoded])
    };
    // Documents and column schemas must be JSON.
    for kind in [
        EnvelopeKind::Document,
        EnvelopeKind::ColumnTableSchema,
        EnvelopeKind::VectorCollectionSchema,
    ] {
        for bad in [&b""[..], b"{", b"\xFF", b"]["] {
            assert!(rebuild(kind, bad).is_err(), "{kind:?} {bad:?}");
        }
    }
    // Column schema: JSON, but not a column list.
    for bad in [
        &br#"{"a":1}"#[..],
        br#"[1]"#,
        br#"[{"name":"x"}]"#,
        br#"[{"name":"x","type":"nope"}]"#,
        br#"null"#,
    ] {
        assert!(
            rebuild(EnvelopeKind::ColumnTableSchema, bad).is_err(),
            "{bad:?}"
        );
    }
    // Vector schema: missing, mistyped, out-of-range, unknown enum names.
    for bad in [
        &br#"[]"#[..],
        br#"{}"#,
        br#"{"dim":3}"#,
        br#"{"dim":"3","storage":"f32","metric":"cosine","m":8,"ef_construction":32,"ef_search":16}"#,
        br#"{"dim":4294967296,"storage":"f32","metric":"cosine","m":8,"ef_construction":32,"ef_search":16}"#,
        br#"{"dim":-1,"storage":"f32","metric":"cosine","m":8,"ef_construction":32,"ef_search":16}"#,
        br#"{"dim":3,"storage":"f64","metric":"cosine","m":8,"ef_construction":32,"ef_search":16}"#,
        br#"{"dim":3,"storage":"f32","metric":"manhattan","m":8,"ef_construction":32,"ef_search":16}"#,
        br#"{"dim":3,"storage":"f32","metric":"cosine","m":8,"ef_construction":32}"#,
    ] {
        assert!(
            rebuild(EnvelopeKind::VectorCollectionSchema, bad).is_err(),
            "{}",
            String::from_utf8_lossy(bad)
        );
    }
    // The well-formed counterpart is accepted, so the refusals above are about the payload.
    assert!(
        rebuild(
            EnvelopeKind::VectorCollectionSchema,
            br#"{"dim":3,"storage":"f32","metric":"cosine","m":8,"ef_construction":32,"ef_search":16}"#
        )
        .is_ok()
    );
}

#[test]
fn a_vector_upsert_envelope_falls_back_to_floats_only_when_the_payload_is_empty() {
    let base = RecordEnvelope {
        seq: 1,
        op_index: 0,
        kind: EnvelopeKind::VectorUpsert,
        namespace: "v".into(),
        record_id: "x".into(),
        schema_version: 1,
        payload_bytes: Vec::new(),
        metadata: None,
        vector_f32: None,
    };
    let err = SyncBatch::from_envelopes(1, std::slice::from_ref(&base))
        .unwrap_err()
        .to_string();
    assert!(err.contains("no payload"), "{err}");

    let mut floats = base.clone();
    floats.vector_f32 = Some(vec![1.0, 2.0]);
    let batch = SyncBatch::from_envelopes(1, &[floats]).unwrap();
    let mut expected = 1.0f32.to_le_bytes().to_vec();
    expected.extend_from_slice(&2.0f32.to_le_bytes());
    assert_eq!(
        batch.ops,
        vec![SyncOp::UpsertVector {
            collection: "v".into(),
            id: "x".into(),
            bytes: expected,
            metadata: None,
        }]
    );

    // Bytes win over floats when both are present.
    let mut both = base;
    both.payload_bytes = vec![9, 9];
    both.vector_f32 = Some(vec![1.0]);
    let batch = SyncBatch::from_envelopes(1, &[both]).unwrap();
    match &batch.ops[0] {
        SyncOp::UpsertVector { bytes, .. } => assert_eq!(bytes, &vec![9, 9]),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_batch_with_thousands_of_ops_densifies_in_order() {
    let ops: Vec<SyncOp> = (0..5_000)
        .map(|i| SyncOp::PutDoc {
            collection: "c".into(),
            id: format!("id-{i}"),
            doc: json!({"i": i}),
        })
        .collect();
    let batch = SyncBatch { seq: 1, ops };
    let mut envelopes = batch.to_envelopes().unwrap();
    // Shuffle deterministically; from_envelopes must still restore op order.
    let mut rng = Rng(77);
    for i in (1..envelopes.len()).rev() {
        envelopes.swap(i, (rng.next() as usize) % (i + 1));
    }
    assert_eq!(SyncBatch::from_envelopes(1, &envelopes).unwrap(), batch);
}
