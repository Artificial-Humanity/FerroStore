//! Columnar codec / zone-map boundaries and HNSW recall, degenerate graphs and contention.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use fl_store::{
    ColumnDef, ColumnFilter, ColumnPredicate, ColumnType, ColumnValue, FerroStore, VectorMetric,
    VectorStorage,
};
use serde_json::json;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 23) as f32) - 1.0
    }
}

// ============================ columnar =================================================

fn col(name: &str, ty: ColumnType) -> ColumnDef {
    ColumnDef {
        name: name.into(),
        ty,
    }
}

fn filter(column: &str, predicate: ColumnPredicate) -> ColumnFilter {
    ColumnFilter {
        column: column.into(),
        predicate,
    }
}

fn ints(values: &[i64]) -> Vec<Vec<ColumnValue>> {
    values
        .iter()
        .map(|v| vec![ColumnValue::Int64(*v)])
        .collect()
}

fn int_table(groups: &[&[i64]]) -> FerroStore {
    let store = FerroStore::in_memory();
    store
        .create_column_table("t", &[col("n", ColumnType::Int64)])
        .unwrap();
    for group in groups {
        store.append_column_rows("t", &ints(group)).unwrap();
    }
    store
}

fn scan_ints(store: &FerroStore, predicate: ColumnPredicate) -> (Vec<i64>, u64, u64) {
    let scan = store
        .scan_column_table("t", &[filter("n", predicate)])
        .unwrap();
    let rows = scan
        .rows
        .iter()
        .map(|row| match row[0] {
            ColumnValue::Int64(v) => v,
            ref other => panic!("{other:?}"),
        })
        .collect();
    (rows, scan.row_groups_total, scan.row_groups_skipped)
}

fn rows_eq(left: &[Vec<ColumnValue>], right: &[Vec<ColumnValue>]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(a, b)| {
            a.len() == b.len()
                && a.iter().zip(b).all(|(x, y)| match (x, y) {
                    (ColumnValue::Float64(p), ColumnValue::Float64(q)) => {
                        p.to_bits() == q.to_bits()
                    }
                    _ => x == y,
                })
        })
}

#[test]
fn delta_codec_survives_i64_extremes_and_wrapping_deltas() {
    let cases: &[&[i64]] = &[
        &[i64::MIN, i64::MAX],
        &[i64::MAX, i64::MIN],
        &[i64::MIN, i64::MAX, i64::MIN, i64::MAX],
        &[0, i64::MAX, i64::MAX, i64::MAX, i64::MAX],
        &[i64::MIN, i64::MIN, i64::MIN, i64::MIN],
        &[i64::MIN, -1, 0, 1, i64::MAX],
        &[i64::MAX - 2, i64::MAX - 1, i64::MAX],
        &[i64::MIN, i64::MIN + 1, i64::MIN + 2, i64::MIN + 3],
        &[-1, 0, -1, 0, -1, 0],
    ];
    for case in cases {
        let store = int_table(&[case]);
        let scan = store.scan_column_table("t", &[]).unwrap();
        assert!(
            rows_eq(&scan.rows, &ints(case)),
            "{case:?} -> {:?}",
            scan.rows
        );
    }
}

#[test]
fn delta_rle_picks_up_arithmetic_runs_and_keeps_every_value() {
    // Constant stride triggers delta+RLE; the stride flips sign and size mid-stream.
    let mut values = Vec::new();
    values.extend((0..100).map(|i| i * 3));
    values.extend((0..100).map(|i| 297 - i * 7));
    values.extend(std::iter::repeat_n(-400, 50));
    values.extend((0..10).map(|i| i64::MAX - 100 + i));
    let store = int_table(&[&values]);
    let scan = store.scan_column_table("t", &[]).unwrap();
    assert!(rows_eq(&scan.rows, &ints(&values)));
}

#[test]
fn single_row_and_single_value_groups_round_trip_for_every_type() {
    let store = FerroStore::in_memory();
    let columns = [
        col("i", ColumnType::Int64),
        col("f", ColumnType::Float64),
        col("b", ColumnType::Bool),
        col("s", ColumnType::Utf8),
        col("y", ColumnType::Bytes),
        col("t", ColumnType::TimestampMs),
    ];
    store.create_column_table("all", &columns).unwrap();
    let row = vec![
        ColumnValue::Int64(-7),
        ColumnValue::Float64(1.5),
        ColumnValue::Bool(true),
        ColumnValue::Utf8("é".into()),
        ColumnValue::Bytes(vec![0, 255]),
        ColumnValue::TimestampMs(i64::MIN),
    ];
    store
        .append_column_rows("all", std::slice::from_ref(&row))
        .unwrap();
    let all_null = vec![ColumnValue::Null; columns.len()];
    store
        .append_column_rows("all", std::slice::from_ref(&all_null))
        .unwrap();
    let scan = store.scan_column_table("all", &[]).unwrap();
    assert!(rows_eq(&scan.rows, &[row, all_null]));
    assert_eq!(scan.row_groups_total, 2);
}

#[test]
fn rle_handles_runs_of_every_length_around_the_threshold() {
    for run in [1usize, 2, 3, 4, 5, 255, 256, 257, 65_535, 65_536, 65_537] {
        let store = FerroStore::in_memory();
        store
            .create_column_table(
                "t",
                &[
                    col("b", ColumnType::Bool),
                    col("s", ColumnType::Utf8),
                    col("y", ColumnType::Bytes),
                    col("f", ColumnType::Float64),
                ],
            )
            .unwrap();
        let mut rows = Vec::new();
        for i in 0..run {
            rows.push(vec![
                ColumnValue::Bool(true),
                ColumnValue::Utf8("same".into()),
                ColumnValue::Bytes(vec![7; 4]),
                ColumnValue::Float64(2.5),
            ]);
            if i % 5 == 4 {
                // Break runs now and then so both run and literal paths are exercised.
                rows.push(vec![
                    ColumnValue::Bool(false),
                    ColumnValue::Utf8("other".into()),
                    ColumnValue::Bytes(vec![]),
                    ColumnValue::Float64(-0.0),
                ]);
            }
        }
        store.append_column_rows("t", &rows).unwrap();
        let scan = store.scan_column_table("t", &[]).unwrap();
        assert!(rows_eq(&scan.rows, &rows), "run {run}");
    }
}

#[test]
fn dictionary_encoding_round_trips_low_and_high_cardinality_text() {
    let store = FerroStore::in_memory();
    store
        .create_column_table("t", &[col("s", ColumnType::Utf8)])
        .unwrap();
    let low: Vec<Vec<ColumnValue>> = (0..2_000)
        .map(|i| {
            vec![ColumnValue::Utf8(
                ["alpha", "beta", "", "γάμμα"][i % 4].into(),
            )]
        })
        .collect();
    let high: Vec<Vec<ColumnValue>> = (0..2_000)
        .map(|i| vec![ColumnValue::Utf8(format!("unique-{i}"))])
        .collect();
    let nulls: Vec<Vec<ColumnValue>> = (0..100)
        .map(|i| {
            vec![if i % 3 == 0 {
                ColumnValue::Null
            } else {
                ColumnValue::Utf8("x".into())
            }]
        })
        .collect();
    for group in [&low, &high, &nulls] {
        store.append_column_rows("t", group).unwrap();
    }
    let scan = store.scan_column_table("t", &[]).unwrap();
    let expected: Vec<_> = low.iter().chain(&high).chain(&nulls).cloned().collect();
    assert!(rows_eq(&scan.rows, &expected));
    let eq = store
        .scan_column_table(
            "t",
            &[filter(
                "s",
                ColumnPredicate::Eq(ColumnValue::Utf8("beta".into())),
            )],
        )
        .unwrap();
    assert_eq!(eq.rows.len(), 500);
}

#[test]
fn zone_map_boundaries_are_inclusive_where_they_should_be() {
    // Two groups: [10, 20] and [30, 40].
    let store = int_table(&[&[10, 15, 20], &[30, 35, 40]]);
    let cases: Vec<(ColumnPredicate, Vec<i64>, u64)> = vec![
        (ColumnPredicate::Eq(ColumnValue::Int64(20)), vec![20], 1),
        (ColumnPredicate::Eq(ColumnValue::Int64(21)), vec![], 2),
        (ColumnPredicate::Eq(ColumnValue::Int64(9)), vec![], 2),
        (ColumnPredicate::Eq(ColumnValue::Int64(41)), vec![], 2),
        (ColumnPredicate::Lt(ColumnValue::Int64(10)), vec![], 2),
        (ColumnPredicate::Lt(ColumnValue::Int64(11)), vec![10], 1),
        (ColumnPredicate::Le(ColumnValue::Int64(10)), vec![10], 1),
        (ColumnPredicate::Le(ColumnValue::Int64(9)), vec![], 2),
        (ColumnPredicate::Gt(ColumnValue::Int64(40)), vec![], 2),
        (ColumnPredicate::Gt(ColumnValue::Int64(39)), vec![40], 1),
        (ColumnPredicate::Ge(ColumnValue::Int64(40)), vec![40], 1),
        (ColumnPredicate::Ge(ColumnValue::Int64(41)), vec![], 2),
        (
            ColumnPredicate::Between {
                low: ColumnValue::Int64(20),
                high: ColumnValue::Int64(30),
            },
            vec![20, 30],
            0,
        ),
        (
            ColumnPredicate::Between {
                low: ColumnValue::Int64(21),
                high: ColumnValue::Int64(29),
            },
            vec![],
            2,
        ),
        (
            ColumnPredicate::Between {
                low: ColumnValue::Int64(i64::MIN),
                high: ColumnValue::Int64(i64::MAX),
            },
            vec![10, 15, 20, 30, 35, 40],
            0,
        ),
        // Inverted range matches nothing; it must not panic or match everything.
        (
            ColumnPredicate::Between {
                low: ColumnValue::Int64(40),
                high: ColumnValue::Int64(10),
            },
            vec![],
            2,
        ),
    ];
    for (predicate, expected, skipped) in cases {
        let (rows, total, got_skipped) = scan_ints(&store, predicate.clone());
        assert_eq!(rows, expected, "{predicate:?}");
        assert_eq!(total, 2, "{predicate:?}");
        assert_eq!(got_skipped, skipped, "{predicate:?} skipped");
    }
}

#[test]
fn predicates_at_the_i64_edges_do_not_overflow() {
    let store = int_table(&[&[i64::MIN, i64::MIN + 1], &[i64::MAX - 1, i64::MAX]]);
    assert_eq!(
        scan_ints(&store, ColumnPredicate::Lt(ColumnValue::Int64(i64::MIN))).0,
        Vec::<i64>::new()
    );
    assert_eq!(
        scan_ints(&store, ColumnPredicate::Le(ColumnValue::Int64(i64::MIN))).0,
        vec![i64::MIN]
    );
    assert_eq!(
        scan_ints(&store, ColumnPredicate::Gt(ColumnValue::Int64(i64::MAX))).0,
        Vec::<i64>::new()
    );
    assert_eq!(
        scan_ints(&store, ColumnPredicate::Ge(ColumnValue::Int64(i64::MAX))).0,
        vec![i64::MAX]
    );
    assert_eq!(
        scan_ints(
            &store,
            ColumnPredicate::Eq(ColumnValue::Int64(i64::MAX - 1))
        )
        .0,
        vec![i64::MAX - 1]
    );
}

#[test]
fn nulls_never_match_a_predicate_and_do_not_poison_the_zone_map() {
    let store = FerroStore::in_memory();
    store
        .create_column_table("t", &[col("n", ColumnType::Int64)])
        .unwrap();
    let with_nulls = vec![
        vec![ColumnValue::Null],
        vec![ColumnValue::Int64(5)],
        vec![ColumnValue::Null],
        vec![ColumnValue::Int64(9)],
    ];
    let all_null = vec![vec![ColumnValue::Null]; 4];
    store.append_column_rows("t", &with_nulls).unwrap();
    store.append_column_rows("t", &all_null).unwrap();
    let (rows, total, skipped) = scan_ints(
        &store,
        ColumnPredicate::Between {
            low: ColumnValue::Int64(5),
            high: ColumnValue::Int64(9),
        },
    );
    assert_eq!(rows, vec![5, 9]);
    assert_eq!(
        (total, skipped),
        (2, 1),
        "an all-null group has no bounds and must be skipped"
    );
    for predicate in [
        ColumnPredicate::Lt(ColumnValue::Int64(i64::MAX)),
        ColumnPredicate::Gt(ColumnValue::Int64(i64::MIN)),
        ColumnPredicate::Le(ColumnValue::Int64(i64::MAX)),
        ColumnPredicate::Ge(ColumnValue::Int64(i64::MIN)),
    ] {
        assert_eq!(
            scan_ints(&store, predicate.clone()).0,
            vec![5, 9],
            "{predicate:?}"
        );
    }
    // Null itself is not an equality target.
    let scan = store
        .scan_column_table("t", &[filter("n", ColumnPredicate::Eq(ColumnValue::Null))])
        .map(|s| s.rows.len());
    assert!(matches!(scan, Ok(0) | Err(_)), "{scan:?}");
}

#[test]
fn an_empty_table_and_an_empty_append_scan_cleanly() {
    let store = int_table(&[]);
    store.append_column_rows("t", &[]).unwrap();
    let scan = store.scan_column_table("t", &[]).unwrap();
    assert!(scan.rows.is_empty());
    assert_eq!((scan.row_groups_total, scan.row_groups_skipped), (0, 0));
    let scan = store
        .scan_column_table(
            "t",
            &[filter("n", ColumnPredicate::Eq(ColumnValue::Int64(1)))],
        )
        .unwrap();
    assert!(scan.rows.is_empty());
}

#[test]
fn float_zone_maps_handle_infinities_negative_zero_and_nan() {
    let store = FerroStore::in_memory();
    store
        .create_column_table("t", &[col("f", ColumnType::Float64)])
        .unwrap();
    let group = |values: &[f64]| -> Vec<Vec<ColumnValue>> {
        values
            .iter()
            .map(|v| vec![ColumnValue::Float64(*v)])
            .collect()
    };
    store
        .append_column_rows(
            "t",
            &group(&[f64::NEG_INFINITY, -1.5, -0.0, 0.0, f64::INFINITY]),
        )
        .unwrap();
    store
        .append_column_rows("t", &group(&[f64::MAX, f64::MIN_POSITIVE, 5e-324]))
        .unwrap();
    let scan = store.scan_column_table("t", &[]).unwrap();
    assert_eq!(scan.rows.len(), 8);
    assert!(rows_eq(
        &scan.rows[..5],
        &group(&[f64::NEG_INFINITY, -1.5, -0.0, 0.0, f64::INFINITY])
    ));
    let count = |predicate| {
        store
            .scan_column_table("t", &[filter("f", predicate)])
            .unwrap()
            .rows
            .len()
    };
    assert_eq!(
        count(ColumnPredicate::Gt(ColumnValue::Float64(f64::MAX))),
        1,
        "only +inf"
    );
    assert_eq!(
        count(ColumnPredicate::Lt(ColumnValue::Float64(f64::MIN))),
        1,
        "only -inf"
    );
    assert_eq!(
        count(ColumnPredicate::Eq(ColumnValue::Float64(0.0))),
        2,
        "-0.0 == 0.0"
    );
    assert_eq!(count(ColumnPredicate::Le(ColumnValue::Float64(5e-324))), 5);
}

/// No-panic smoke test: appending NaN may be accepted or refused (both are allowed and
/// neither is pinned). It only pins that nothing panics and that, in whichever branch is
/// taken, a scan stays consistent (NaN never satisfies a range predicate; a refusal
/// leaves the table empty).
#[test]
fn nan_append_and_scan_does_not_panic_and_stays_consistent() {
    let store = FerroStore::in_memory();
    store
        .create_column_table("t", &[col("f", ColumnType::Float64)])
        .unwrap();
    let rows = vec![
        vec![ColumnValue::Float64(1.0)],
        vec![ColumnValue::Float64(f64::NAN)],
        vec![ColumnValue::Float64(3.0)],
    ];
    match store.append_column_rows("t", &rows) {
        Ok(()) => {
            let scan = store.scan_column_table("t", &[]).unwrap();
            assert_eq!(scan.rows.len(), 3);
            let hits = store
                .scan_column_table(
                    "t",
                    &[filter("f", ColumnPredicate::Ge(ColumnValue::Float64(0.0)))],
                )
                .unwrap();
            // NaN is unordered: it may never satisfy a range predicate.
            assert!(
                hits.rows
                    .iter()
                    .all(|r| !matches!(r[0], ColumnValue::Float64(v) if v.is_nan())),
                "{:?}",
                hits.rows
            );
        }
        Err(_) => {
            assert!(store.scan_column_table("t", &[]).unwrap().rows.is_empty());
        }
    }
}

#[test]
fn text_zone_maps_order_by_bytes_including_multibyte_boundaries() {
    let store = FerroStore::in_memory();
    store
        .create_column_table("t", &[col("s", ColumnType::Utf8)])
        .unwrap();
    let group = |values: &[&str]| -> Vec<Vec<ColumnValue>> {
        values
            .iter()
            .map(|v| vec![ColumnValue::Utf8((*v).into())])
            .collect()
    };
    store
        .append_column_rows("t", &group(&["", "a", "b"]))
        .unwrap();
    store
        .append_column_rows("t", &group(&["é", "z", "日本"]))
        .unwrap();
    store.append_column_rows("t", &group(&["🦀"])).unwrap();
    let count = |predicate| {
        store
            .scan_column_table("t", &[filter("s", predicate)])
            .unwrap()
    };
    let ge_z = count(ColumnPredicate::Ge(ColumnValue::Utf8("z".into())));
    assert_eq!(
        ge_z.rows.len(),
        4,
        "é, z, 日本, 🦀: é (0xC3..) sorts after z (0x7A) by byte"
    );
    assert_eq!(ge_z.row_groups_skipped, 1);
    let empty = count(ColumnPredicate::Eq(ColumnValue::Utf8(String::new())));
    assert_eq!(empty.rows.len(), 1);
    assert_eq!(empty.row_groups_skipped, 2);
    let between = count(ColumnPredicate::Between {
        low: ColumnValue::Utf8("é".into()),
        high: ColumnValue::Utf8("é".into()),
    });
    assert_eq!(between.rows.len(), 1);
}

#[test]
fn timestamp_and_bytes_columns_prune_like_integers() {
    let store = FerroStore::in_memory();
    store
        .create_column_table(
            "t",
            &[
                col("ts", ColumnType::TimestampMs),
                col("y", ColumnType::Bytes),
            ],
        )
        .unwrap();
    for base in [0i64, 1_000, 2_000] {
        let rows: Vec<_> = (0..10)
            .map(|i| {
                vec![
                    ColumnValue::TimestampMs(base + i),
                    ColumnValue::Bytes(vec![(base / 1000) as u8; 3]),
                ]
            })
            .collect();
        store.append_column_rows("t", &rows).unwrap();
    }
    let scan = store
        .scan_column_table(
            "t",
            &[filter(
                "ts",
                ColumnPredicate::Between {
                    low: ColumnValue::TimestampMs(1_003),
                    high: ColumnValue::TimestampMs(1_005),
                },
            )],
        )
        .unwrap();
    assert_eq!(scan.rows.len(), 3);
    assert_eq!((scan.row_groups_total, scan.row_groups_skipped), (3, 2));
    let by_bytes = store
        .scan_column_table(
            "t",
            &[filter(
                "y",
                ColumnPredicate::Eq(ColumnValue::Bytes(vec![2, 2, 2])),
            )],
        )
        .unwrap();
    assert_eq!(by_bytes.rows.len(), 10);
    assert_eq!(by_bytes.row_groups_skipped, 2);
}

#[test]
fn multiple_filters_are_anded_and_either_can_prune() {
    let store = FerroStore::in_memory();
    store
        .create_column_table(
            "t",
            &[col("a", ColumnType::Int64), col("b", ColumnType::Int64)],
        )
        .unwrap();
    for g in 0..5i64 {
        let rows: Vec<_> = (0..4)
            .map(|i| {
                vec![
                    ColumnValue::Int64(g * 10 + i),
                    ColumnValue::Int64(100 - g * 10 - i),
                ]
            })
            .collect();
        store.append_column_rows("t", &rows).unwrap();
    }
    let scan = store
        .scan_column_table(
            "t",
            &[
                filter("a", ColumnPredicate::Ge(ColumnValue::Int64(10))),
                filter("b", ColumnPredicate::Ge(ColumnValue::Int64(70))),
            ],
        )
        .unwrap();
    // a >= 10 prunes group 0; b >= 70 prunes group 4 (b <= 60). Group 3 keeps only b = 70.
    assert_eq!(scan.row_groups_total, 5);
    assert!(scan.row_groups_skipped >= 2, "{}", scan.row_groups_skipped);
    for row in &scan.rows {
        let (ColumnValue::Int64(a), ColumnValue::Int64(b)) = (&row[0], &row[1]) else {
            panic!("{row:?}")
        };
        assert!(*a >= 10 && *b >= 70);
    }
    assert_eq!(
        scan.rows.len(),
        4 + 4 + 1,
        "groups 1 and 2 fully, group 3 only at b = 70"
    );
}

#[test]
fn scan_agrees_with_a_brute_force_oracle_over_random_data_and_filters() {
    let mut rng = Rng(0xC0FFEE);
    let store = FerroStore::in_memory();
    store
        .create_column_table(
            "t",
            &[col("n", ColumnType::Int64), col("s", ColumnType::Utf8)],
        )
        .unwrap();
    let mut all: Vec<(i64, String)> = Vec::new();
    for _ in 0..40 {
        let len = 1 + (rng.next() % 60) as usize;
        let base = (rng.next() % 2_000) as i64 - 1_000;
        let stride = (rng.next() % 5) as i64;
        let mut rows = Vec::new();
        for i in 0..len as i64 {
            let n = if rng.next().is_multiple_of(6) {
                (rng.next() % 2_000) as i64 - 1_000
            } else {
                base + stride * i
            };
            let s = format!("v{}", rng.next() % 12);
            rows.push(vec![ColumnValue::Int64(n), ColumnValue::Utf8(s.clone())]);
            all.push((n, s));
        }
        store.append_column_rows("t", &rows).unwrap();
    }
    for _ in 0..300 {
        let low = (rng.next() % 2_400) as i64 - 1_200;
        let high = low + (rng.next() % 600) as i64;
        let tag = format!("v{}", rng.next() % 12);
        let scan = store
            .scan_column_table(
                "t",
                &[
                    filter(
                        "n",
                        ColumnPredicate::Between {
                            low: ColumnValue::Int64(low),
                            high: ColumnValue::Int64(high),
                        },
                    ),
                    filter("s", ColumnPredicate::Eq(ColumnValue::Utf8(tag.clone()))),
                ],
            )
            .unwrap();
        let expected: Vec<_> = all
            .iter()
            .filter(|(n, s)| (low..=high).contains(n) && *s == tag)
            .map(|(n, s)| vec![ColumnValue::Int64(*n), ColumnValue::Utf8(s.clone())])
            .collect();
        assert!(rows_eq(&scan.rows, &expected), "[{low},{high}] {tag}");
        assert!(scan.row_groups_skipped <= scan.row_groups_total);
    }
}

#[test]
fn bad_schemas_rows_and_filters_are_refused_without_writing() {
    let store = FerroStore::in_memory();
    assert!(
        store
            .create_column_table("", &[col("n", ColumnType::Int64)])
            .is_err()
    );
    assert!(
        store
            .create_column_table("a/b", &[col("n", ColumnType::Int64)])
            .is_err()
    );
    assert!(
        store
            .create_column_table(&"x".repeat(129), &[col("n", ColumnType::Int64)])
            .is_err()
    );
    assert!(
        store
            .create_column_table("nul\0", &[col("n", ColumnType::Int64)])
            .is_err()
    );
    assert!(
        store
            .create_column_table(
                "dup",
                &[col("n", ColumnType::Int64), col("n", ColumnType::Bool)]
            )
            .is_err()
    );
    store
        .create_column_table("t", &[col("n", ColumnType::Int64)])
        .unwrap();
    assert!(
        store
            .create_column_table("t", &[col("n", ColumnType::Int64)])
            .is_err()
    );
    assert!(
        store
            .append_column_rows("t", &[vec![ColumnValue::Utf8("x".into())]])
            .is_err()
    );
    assert!(store.append_column_rows("t", &[vec![]]).is_err());
    assert!(
        store
            .append_column_rows("t", &[vec![ColumnValue::Int64(1), ColumnValue::Int64(2)]])
            .is_err()
    );
    // One bad row spoils the whole group, so none of its rows may land.
    assert!(
        store
            .append_column_rows(
                "t",
                &[vec![ColumnValue::Int64(1)], vec![ColumnValue::Bool(true)]]
            )
            .is_err()
    );
    assert!(store.scan_column_table("t", &[]).unwrap().rows.is_empty());
    assert!(store.scan_column_table("ghost", &[]).is_err());
    assert!(store.append_column_rows("ghost", &ints(&[1])).is_err());
    assert!(
        store
            .scan_column_table(
                "t",
                &[filter("nope", ColumnPredicate::Eq(ColumnValue::Int64(1)))]
            )
            .is_err()
    );
    assert!(
        store
            .scan_column_table(
                "t",
                &[filter(
                    "n",
                    ColumnPredicate::Eq(ColumnValue::Utf8("x".into()))
                )]
            )
            .is_err(),
        "a predicate of the wrong type"
    );
}

/// No-panic smoke test: creating, appending to and scanning a zero-column table may
/// succeed or fail (both are allowed and the results are deliberately not inspected). It
/// only pins that none of those calls panics.
#[test]
fn a_zero_column_table_does_not_panic() {
    let store = FerroStore::in_memory();
    match store.create_column_table("empty", &[]) {
        Ok(()) => {
            // Rows have no cells; appending them must not panic.
            let _ = store.append_column_rows("empty", &[vec![]]);
            let _ = store.scan_column_table("empty", &[]);
        }
        Err(_) => assert!(store.scan_column_table("empty", &[]).is_err()),
    }
}

#[test]
fn column_tables_survive_reopen_and_checkpoint_on_disk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cols.fls");
    let values: Vec<i64> = (0..500)
        .map(|i| i * 3 - 100)
        .chain([i64::MIN, i64::MAX])
        .collect();
    {
        let store = FerroStore::open(&path).unwrap();
        store
            .create_column_table("t", &[col("n", ColumnType::Int64)])
            .unwrap();
        store.append_column_rows("t", &ints(&values)).unwrap();
        store.append_column_rows("t", &ints(&[7, 7, 7, 7])).unwrap();
        store.checkpoint().unwrap();
        store.append_column_rows("t", &ints(&[123_457])).unwrap();
    }
    let store = FerroStore::open(&path).unwrap();
    let scan = store.scan_column_table("t", &[]).unwrap();
    let mut expected = values.clone();
    expected.extend([7, 7, 7, 7, 123_457]);
    assert!(rows_eq(&scan.rows, &ints(&expected)));
    assert_eq!(scan.row_groups_total, 3);
    let (rows, _, skipped) = scan_ints(&store, ColumnPredicate::Eq(ColumnValue::Int64(123_457)));
    assert_eq!(rows, vec![123_457]);
    // The first group spans i64::MIN..=i64::MAX, so only the all-7s group is pruned.
    assert_eq!(skipped, 1);
}

#[test]
fn concurrent_appenders_and_scanners_see_whole_row_groups_only() {
    let store = Arc::new(FerroStore::in_memory());
    store
        .create_column_table("t", &[col("n", ColumnType::Int64)])
        .unwrap();
    thread::scope(|scope| {
        for t in 0..6i64 {
            let store = &store;
            scope.spawn(move || {
                for g in 0..25i64 {
                    // Every group carries 10 rows tagged with a unique base.
                    let base = (t * 100 + g) * 10;
                    store
                        .append_column_rows("t", &ints(&(base..base + 10).collect::<Vec<_>>()))
                        .unwrap();
                }
            });
        }
        for _ in 0..3 {
            let store = &store;
            scope.spawn(move || {
                for _ in 0..40 {
                    let scan = store.scan_column_table("t", &[]).unwrap();
                    assert_eq!(scan.rows.len() % 10, 0, "a torn row group was visible");
                    assert_eq!(scan.rows.len() as u64, scan.row_groups_total * 10);
                }
            });
        }
    });
    let scan = store.scan_column_table("t", &[]).unwrap();
    assert_eq!(scan.row_groups_total, 150);
    let seen: BTreeSet<i64> = scan
        .rows
        .iter()
        .map(|r| match r[0] {
            ColumnValue::Int64(v) => v,
            _ => unreachable!(),
        })
        .collect();
    assert_eq!(seen.len(), 1_500, "a row was lost or duplicated");
}

// ============================ vectors ==================================================

fn make_collection(store: &FerroStore, dim: u32, storage: VectorStorage, metric: VectorMetric) {
    store
        .create_vector_collection("v", dim, storage, metric, 16, 100, 64)
        .unwrap();
}

fn exact_top(
    data: &[(String, Vec<f32>)],
    query: &[f32],
    metric: VectorMetric,
    k: usize,
) -> Vec<String> {
    let score = |v: &[f32]| -> f64 {
        match metric {
            VectorMetric::L2 => -v
                .iter()
                .zip(query)
                .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
                .sum::<f64>(),
            VectorMetric::DotProduct => v
                .iter()
                .zip(query)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum(),
            VectorMetric::Cosine => {
                let dot: f64 = v
                    .iter()
                    .zip(query)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum();
                let na: f64 = v.iter().map(|a| f64::from(*a).powi(2)).sum::<f64>().sqrt();
                let nb: f64 = query
                    .iter()
                    .map(|a| f64::from(*a).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if na == 0.0 || nb == 0.0 {
                    0.0
                } else {
                    dot / (na * nb)
                }
            }
        }
    };
    let mut scored: Vec<_> = data.iter().map(|(id, v)| (score(v), id.clone())).collect();
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    scored.into_iter().take(k).map(|(_, id)| id).collect()
}

/// Mean recall@k of the index against an exact scan over the *stored* (quantized) values.
fn mean_recall(storage: VectorStorage, metric: VectorMetric, scale: f32) -> f64 {
    const N: usize = 300;
    const DIM: u32 = 16;
    const K: usize = 10;
    let mut rng = Rng(0xFEED_FACE ^ (storage as u64) ^ ((metric as u64) << 8));
    let store = FerroStore::in_memory();
    make_collection(&store, DIM, storage, metric);
    let mut data = Vec::new();
    for i in 0..N {
        let raw: Vec<f32> = (0..DIM).map(|_| rng.unit() * scale).collect();
        let id = format!("v{i:04}");
        store.upsert_vector("v", &id, &raw, None).unwrap();
        // Compare against what the store actually holds after quantization.
        let (stored, _) = store.get_vector("v", &id).unwrap().unwrap();
        data.push((id, stored));
    }
    let mut total = 0.0;
    let queries = 25;
    for _ in 0..queries {
        let q: Vec<f32> = (0..DIM).map(|_| rng.unit() * scale).collect();
        let truth: BTreeSet<_> = exact_top(&data, &q, metric, K).into_iter().collect();
        let got = store.search_vectors("v", &q, K, Some(200)).unwrap();
        assert_eq!(got.len(), K);
        let hit = got.iter().filter(|(id, _)| truth.contains(id)).count();
        total += hit as f64 / K as f64;
    }
    total / f64::from(queries)
}

#[test]
fn recall_holds_for_every_storage_and_metric() {
    // I8 stores rounded integers, so its data spans a wide integer range.
    let cases = [
        (VectorStorage::F32, 1.0f32, 0.95),
        (VectorStorage::F16, 1.0, 0.93),
        (VectorStorage::I8, 100.0, 0.90),
    ];
    for (storage, scale, floor) in cases {
        for metric in [
            VectorMetric::L2,
            VectorMetric::Cosine,
            VectorMetric::DotProduct,
        ] {
            let recall = mean_recall(storage, metric, scale);
            assert!(
                recall >= floor,
                "{storage:?}/{metric:?} recall {recall:.3} < {floor}"
            );
        }
    }
}

#[test]
fn quantization_round_trips_within_each_storage_tolerance() {
    let store = FerroStore::in_memory();
    for (name, storage, tol) in [
        ("f32", VectorStorage::F32, 0.0f32),
        ("f16", VectorStorage::F16, 1e-3),
        ("i8", VectorStorage::I8, 0.5),
    ] {
        store
            .create_vector_collection(name, 4, storage, VectorMetric::L2, 8, 32, 16)
            .unwrap();
        let values = [0.25f32, -1.5, 100.0, 0.0];
        store
            .upsert_vector(name, "a", &values, Some(&json!({"k": "v"})))
            .unwrap();
        let (got, meta) = store.get_vector(name, "a").unwrap().unwrap();
        assert_eq!(meta, Some(json!({"k": "v"})));
        for (a, b) in got.iter().zip(values) {
            let allowed = tol.max(b.abs() * tol);
            assert!(
                (a - b).abs() <= allowed.max(if tol == 0.0 { 0.0 } else { tol }),
                "{name}: {a} vs {b}"
            );
        }
    }
    // i8 clamps out-of-range components; f16 refuses values it cannot represent finitely.
    let store = FerroStore::in_memory();
    store
        .create_vector_collection("i", 2, VectorStorage::I8, VectorMetric::L2, 8, 32, 16)
        .unwrap();
    store
        .upsert_vector("i", "a", &[1_000.0, -1_000.0], None)
        .unwrap();
    assert_eq!(
        store.get_vector("i", "a").unwrap().unwrap().0,
        vec![127.0, -128.0]
    );
    store
        .create_vector_collection("h", 1, VectorStorage::F16, VectorMetric::L2, 8, 32, 16)
        .unwrap();
    assert!(store.upsert_vector("h", "a", &[1e30], None).is_err());
}

#[test]
fn non_finite_and_wrong_width_vectors_are_refused() {
    let store = FerroStore::in_memory();
    make_collection(&store, 3, VectorStorage::F32, VectorMetric::Cosine);
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            store
                .upsert_vector("v", "a", &[1.0, bad, 0.0], None)
                .is_err()
        );
        assert!(
            store
                .search_vectors("v", &[1.0, bad, 0.0], 1, None)
                .is_err()
        );
    }
    for width in [0usize, 1, 2, 4, 100] {
        assert!(
            store
                .upsert_vector("v", "a", &vec![1.0; width], None)
                .is_err(),
            "{width}"
        );
        assert!(
            store
                .search_vectors("v", &vec![1.0; width], 1, None)
                .is_err(),
            "{width}"
        );
    }
    assert_eq!(
        store.get_vector("v", "a").unwrap(),
        None,
        "a refused upsert left a record"
    );
}

#[test]
fn collection_parameter_boundaries() {
    let store = FerroStore::in_memory();
    let make = |name: &str, dim, m, efc, efs| {
        store.create_vector_collection(name, dim, VectorStorage::F32, VectorMetric::L2, m, efc, efs)
    };
    assert!(make("zero-dim", 0, 8, 32, 16).is_err());
    assert!(make("m1", 3, 1, 32, 16).is_err());
    assert!(make("m2", 3, 2, 32, 16).is_ok());
    assert!(make("m128", 3, 128, 128, 16).is_ok());
    assert!(make("m129", 3, 129, 200, 16).is_err());
    assert!(make("efc-lt-m", 3, 8, 7, 16).is_err());
    assert!(make("efc-eq-m", 3, 8, 8, 16).is_ok());
    assert!(make("efc-max", 3, 8, 100_000, 16).is_ok());
    assert!(make("efc-over", 3, 8, 100_001, 16).is_err());
    assert!(make("efs-zero", 3, 8, 32, 0).is_err());
    assert!(make("efs-max", 3, 8, 32, 100_000).is_ok());
    assert!(make("efs-over", 3, 8, 32, 100_001).is_err());
    assert!(
        make("dim-max-f32", 2 * 1024 * 1024, 8, 32, 16).is_ok(),
        "8 MiB of f32"
    );
    assert!(make("dim-over-f32", 2 * 1024 * 1024 + 1, 8, 32, 16).is_err());
    assert!(make("dim-overflow", u32::MAX, 8, 32, 16).is_err());
    assert!(make("", 3, 8, 32, 16).is_err());
    assert!(make("dup-name", 3, 8, 32, 16).is_ok());
    assert!(make("dup-name", 3, 8, 32, 16).is_err());
}

#[test]
fn vector_ids_have_boundaries_too() {
    let store = FerroStore::in_memory();
    make_collection(&store, 2, VectorStorage::F32, VectorMetric::L2);
    assert!(store.upsert_vector("v", "", &[0.0, 0.0], None).is_err());
    assert!(
        store
            .upsert_vector("v", "nul\0id", &[0.0, 0.0], None)
            .is_err()
    );
    assert!(
        store
            .upsert_vector("v", &"x".repeat(1024), &[0.0, 0.0], None)
            .is_ok()
    );
    assert!(
        store
            .upsert_vector("v", &"x".repeat(1025), &[0.0, 0.0], None)
            .is_err()
    );
    assert!(
        store
            .upsert_vector("v", "日本語🦀", &[1.0, 1.0], None)
            .is_ok()
    );
    assert!(store.get_vector("v", "").is_err());
    assert!(store.delete_vector("v", "").is_err());
    assert_eq!(
        store.search_vectors("v", &[1.0, 1.0], 1, None).unwrap()[0].0,
        "日本語🦀"
    );
}

#[test]
fn an_empty_graph_answers_every_query_shape_with_nothing() {
    for storage in [VectorStorage::F32, VectorStorage::F16, VectorStorage::I8] {
        let store = FerroStore::in_memory();
        make_collection(&store, 4, storage, VectorMetric::Cosine);
        for (limit, ef) in [
            (0, None),
            (1, None),
            (10, Some(1)),
            (10, Some(0)),
            (usize::MAX, None),
        ] {
            assert!(
                store
                    .search_vectors("v", &[1.0; 4], limit, ef)
                    .unwrap()
                    .is_empty(),
                "{storage:?} limit {limit} ef {ef:?}"
            );
        }
        assert_eq!(store.get_vector("v", "ghost").unwrap(), None);
        store.delete_vector("v", "ghost").unwrap();
    }
    let store = FerroStore::in_memory();
    assert!(store.search_vectors("none", &[1.0], 1, None).is_err());
}

#[test]
fn a_single_element_graph_returns_that_element_for_any_query() {
    for metric in [
        VectorMetric::L2,
        VectorMetric::Cosine,
        VectorMetric::DotProduct,
    ] {
        let store = FerroStore::in_memory();
        make_collection(&store, 3, VectorStorage::F32, metric);
        store
            .upsert_vector("v", "only", &[1.0, 2.0, 3.0], None)
            .unwrap();
        for q in [[1.0, 2.0, 3.0], [-9.0, 0.0, 4.0], [0.0, 0.0, 0.0]] {
            let hits = store.search_vectors("v", &q, 5, None).unwrap();
            assert_eq!(hits.len(), 1, "{metric:?} {q:?}");
            assert_eq!(hits[0].0, "only");
            assert!(hits[0].1.is_finite());
        }
        assert!(
            store
                .search_vectors("v", &[1.0, 2.0, 3.0], 0, None)
                .unwrap()
                .is_empty()
        );
        store.delete_vector("v", "only").unwrap();
        assert!(
            store
                .search_vectors("v", &[1.0, 2.0, 3.0], 5, None)
                .unwrap()
                .is_empty()
        );
        store
            .upsert_vector("v", "again", &[0.0, 0.0, 1.0], None)
            .unwrap();
        assert_eq!(
            store
                .search_vectors("v", &[0.0, 0.0, 1.0], 5, None)
                .unwrap()[0]
                .0,
            "again"
        );
    }
}

#[test]
fn zero_vectors_and_duplicates_do_not_break_scoring() {
    for metric in [
        VectorMetric::L2,
        VectorMetric::Cosine,
        VectorMetric::DotProduct,
    ] {
        let store = FerroStore::in_memory();
        make_collection(&store, 3, VectorStorage::F32, metric);
        for i in 0..30 {
            store
                .upsert_vector("v", &format!("zero{i}"), &[0.0; 3], None)
                .unwrap();
            store
                .upsert_vector("v", &format!("same{i}"), &[1.0, 1.0, 1.0], None)
                .unwrap();
        }
        for q in [[0.0f32; 3], [1.0; 3], [-1.0; 3]] {
            let hits = store.search_vectors("v", &q, 20, None).unwrap();
            assert_eq!(hits.len(), 20, "{metric:?} {q:?}");
            assert!(hits.iter().all(|(_, s)| s.is_finite()));
            // Scores come back best-first.
            assert!(hits.windows(2).all(|w| w[0].1 >= w[1].1), "{hits:?}");
        }
    }
}

#[test]
fn extreme_magnitudes_score_finitely() {
    let store = FerroStore::in_memory();
    make_collection(&store, 2, VectorStorage::F32, VectorMetric::DotProduct);
    store
        .upsert_vector("v", "big", &[f32::MAX, f32::MAX], None)
        .unwrap();
    store
        .upsert_vector("v", "tiny", &[f32::MIN_POSITIVE, 0.0], None)
        .unwrap();
    store
        .upsert_vector("v", "neg", &[f32::MIN, f32::MIN], None)
        .unwrap();
    let hits = store
        .search_vectors("v", &[f32::MAX, f32::MAX], 3, None)
        .unwrap();
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].0, "big");
    assert!(hits.iter().all(|(_, s)| s.is_finite()));
}

#[test]
fn k_larger_than_the_collection_returns_everything_best_first() {
    let store = FerroStore::in_memory();
    make_collection(&store, 2, VectorStorage::F32, VectorMetric::L2);
    for i in 0..25 {
        store
            .upsert_vector("v", &format!("p{i:02}"), &[i as f32, 0.0], None)
            .unwrap();
    }
    let hits = store.search_vectors("v", &[0.0, 0.0], 1_000, None).unwrap();
    assert_eq!(hits.len(), 25);
    assert_eq!(hits[0].0, "p00");
    assert_eq!(hits[24].0, "p24");
    assert!(hits.windows(2).all(|w| w[0].1 >= w[1].1));
    // A tiny ef still yields `limit` results (the search widens to the limit).
    assert_eq!(
        store
            .search_vectors("v", &[0.0, 0.0], 10, Some(1))
            .unwrap()
            .len(),
        10
    );
}

#[test]
fn replacing_and_deleting_vectors_keeps_the_graph_searchable() {
    let store = FerroStore::in_memory();
    make_collection(&store, 2, VectorStorage::F32, VectorMetric::L2);
    for i in 0..60 {
        store
            .upsert_vector("v", &format!("p{i:02}"), &[i as f32, (i % 7) as f32], None)
            .unwrap();
    }
    // Replace an id with a far-away point: it must stop matching its old location.
    store
        .upsert_vector("v", "p00", &[1_000.0, 1_000.0], Some(&json!(1)))
        .unwrap();
    let hits = store.search_vectors("v", &[0.0, 0.0], 3, None).unwrap();
    assert_ne!(hits[0].0, "p00");
    assert_eq!(
        store
            .search_vectors("v", &[1_000.0, 1_000.0], 1, None)
            .unwrap()[0]
            .0,
        "p00"
    );
    // Delete everything one by one, including whichever node is the entry point.
    // Delete only unlinks (docs/ferrostore-spec.md), so a search may return fewer than
    // `limit` hits. What must always hold: something comes back while anything is
    // left, nothing deleted comes back, and no id repeats.
    let mut remaining: BTreeSet<String> = (0..60).map(|i| format!("p{i:02}")).collect();
    let mut rng = Rng(5);
    while !remaining.is_empty() {
        let hits = store
            .search_vectors("v", &[10.0, 3.0], 5, Some(100))
            .unwrap();
        assert!(
            !hits.is_empty() && hits.len() <= remaining.len().min(5),
            "{hits:?}"
        );
        assert!(
            hits.iter().all(|(id, _)| remaining.contains(id)),
            "deleted id returned"
        );
        let unique: BTreeSet<_> = hits.iter().map(|(id, _)| id).collect();
        assert_eq!(unique.len(), hits.len());
        let pick = remaining
            .iter()
            .nth((rng.next() as usize) % remaining.len())
            .cloned()
            .unwrap();
        store.delete_vector("v", &pick).unwrap();
        remaining.remove(&pick);
    }
    assert!(
        store
            .search_vectors("v", &[0.0, 0.0], 5, None)
            .unwrap()
            .is_empty()
    );
}

/// Delete once unlinked a node without reconnecting its neighbours, so the graph could
/// fragment and a search then returned fewer than `limit` hits although enough vectors
/// remained. `delete_vector` now repairs connectivity (reconnects the deleted node's neighbours to each
/// other), so a search returns `min(limit, remaining)` hits after any sequence of deletes.
#[test]
fn search_still_fills_the_limit_after_random_deletes() {
    let store = FerroStore::in_memory();
    store
        .create_vector_collection("v", 2, VectorStorage::F32, VectorMetric::L2, 16, 100, 64)
        .unwrap();
    for i in 0..60 {
        store
            .upsert_vector("v", &format!("p{i:02}"), &[i as f32, (i % 7) as f32], None)
            .unwrap();
    }
    let mut remaining: Vec<String> = (0..60).map(|i| format!("p{i:02}")).collect();
    let mut rng = Rng(7919);
    while !remaining.is_empty() {
        let hits = store
            .search_vectors("v", &[10.0, 3.0], 5, Some(100))
            .unwrap();
        assert_eq!(
            hits.len(),
            remaining.len().min(5),
            "{} left",
            remaining.len()
        );
        let at = (rng.next() as usize) % remaining.len();
        store.delete_vector("v", &remaining.remove(at)).unwrap();
    }
}

#[test]
fn searches_are_deterministic_for_identical_inserts() {
    let build = || {
        let store = FerroStore::in_memory();
        make_collection(&store, 8, VectorStorage::F16, VectorMetric::Cosine);
        let mut rng = Rng(99);
        for i in 0..300 {
            let v: Vec<f32> = (0..8).map(|_| rng.unit()).collect();
            store
                .upsert_vector("v", &format!("v{i:03}"), &v, None)
                .unwrap();
        }
        store
    };
    let (a, b) = (build(), build());
    let mut rng = Rng(1234);
    for _ in 0..20 {
        let q: Vec<f32> = (0..8).map(|_| rng.unit()).collect();
        assert_eq!(
            a.search_vectors("v", &q, 10, None).unwrap(),
            b.search_vectors("v", &q, 10, None).unwrap()
        );
    }
}

#[test]
fn vectors_survive_reopen_and_search_identically() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vec.fls");
    let mut rng = Rng(42);
    let queries: Vec<Vec<f32>> = (0..10)
        .map(|_| (0..6).map(|_| rng.unit()).collect())
        .collect();
    let before = {
        let store = FerroStore::open(&path).unwrap();
        make_collection(&store, 6, VectorStorage::I8, VectorMetric::L2);
        for i in 0..60 {
            let v: Vec<f32> = (0..6).map(|_| rng.unit() * 50.0).collect();
            store
                .upsert_vector("v", &format!("v{i:03}"), &v, Some(&json!({"i": i})))
                .unwrap();
            if i == 30 {
                store.checkpoint().unwrap();
            }
        }
        queries
            .iter()
            .map(|q| store.search_vectors("v", q, 10, Some(100)).unwrap())
            .collect::<Vec<_>>()
    };
    let store = FerroStore::open(&path).unwrap();
    for (q, expected) in queries.iter().zip(before) {
        assert_eq!(
            store.search_vectors("v", q, 10, Some(100)).unwrap(),
            expected
        );
    }
    assert_eq!(
        store.get_vector("v", "v050").unwrap().unwrap().1,
        Some(json!({"i": 50}))
    );
}

#[test]
fn concurrent_inserts_lose_nothing_and_every_vector_finds_itself() {
    const THREADS: usize = 8;
    const PER: usize = 60;
    let store = Arc::new(FerroStore::in_memory());
    make_collection(&store, 12, VectorStorage::F32, VectorMetric::L2);
    thread::scope(|scope| {
        for t in 0..THREADS {
            let store = &store;
            scope.spawn(move || {
                let mut rng = Rng(1000 + t as u64);
                for i in 0..PER {
                    let v: Vec<f32> = (0..12).map(|_| rng.unit()).collect();
                    store
                        .upsert_vector("v", &format!("t{t}-{i}"), &v, None)
                        .unwrap();
                }
            });
        }
    });
    let mut self_hits = 0;
    for t in 0..THREADS {
        for i in 0..PER {
            let id = format!("t{t}-{i}");
            let (v, _) = store.get_vector("v", &id).unwrap().expect("lost an insert");
            let hits = store.search_vectors("v", &v, 1, Some(200)).unwrap();
            if hits[0].0 == id {
                self_hits += 1;
            }
        }
    }
    let total = THREADS * PER;
    assert!(
        self_hits * 100 >= total * 98,
        "{self_hits}/{total} vectors found themselves"
    );
    let all = store
        .search_vectors("v", &[0.0; 12], total + 10, Some(2_000))
        .unwrap();
    assert!(
        all.len() * 100 >= total * 98,
        "graph reaches only {}/{total}",
        all.len()
    );
}

#[test]
fn high_contention_on_one_id_ends_in_exactly_one_consistent_record() {
    let store = Arc::new(FerroStore::in_memory());
    make_collection(&store, 2, VectorStorage::F32, VectorMetric::L2);
    thread::scope(|scope| {
        for t in 0..8u32 {
            let store = &store;
            scope.spawn(move || {
                for i in 0..80u32 {
                    let x = (t * 1000 + i) as f32;
                    store
                        .upsert_vector("v", "hot", &[x, x], Some(&json!(t)))
                        .unwrap();
                    if i % 9 == 0 {
                        store.delete_vector("v", "hot").unwrap();
                    }
                }
            });
        }
    });
    store
        .upsert_vector("v", "hot", &[1.0, 1.0], Some(&json!("final")))
        .unwrap();
    let (v, meta) = store.get_vector("v", "hot").unwrap().unwrap();
    assert_eq!((v, meta), (vec![1.0, 1.0], Some(json!("final"))));
    let hits = store.search_vectors("v", &[1.0, 1.0], 10, None).unwrap();
    assert_eq!(hits.len(), 1, "duplicate graph nodes for one id: {hits:?}");
}

#[test]
fn searches_during_inserts_and_deletes_never_error_or_return_ghosts() {
    let store = Arc::new(FerroStore::in_memory());
    make_collection(&store, 4, VectorStorage::F16, VectorMetric::Cosine);
    for i in 0..50 {
        store
            .upsert_vector("v", &format!("seed{i}"), &[1.0, i as f32, 0.5, 0.25], None)
            .unwrap();
    }
    let searches = AtomicUsize::new(0);
    thread::scope(|scope| {
        for t in 0..3 {
            let store = &store;
            scope.spawn(move || {
                let mut rng = Rng(t + 7);
                for i in 0..120 {
                    let v: Vec<f32> = (0..4).map(|_| rng.unit()).collect();
                    let id = format!("w{t}-{}", i % 40);
                    store.upsert_vector("v", &id, &v, None).unwrap();
                    if i % 3 == 0 {
                        store
                            .delete_vector("v", &format!("w{t}-{}", (i + 13) % 40))
                            .unwrap();
                    }
                }
            });
        }
        for t in 0..4 {
            let (store, searches) = (&store, &searches);
            scope.spawn(move || {
                let mut rng = Rng(100 + t);
                for _ in 0..150 {
                    let q: Vec<f32> = (0..4).map(|_| rng.unit()).collect();
                    let hits = store.search_vectors("v", &q, 8, Some(64)).unwrap();
                    assert!(!hits.is_empty() && hits.len() <= 8);
                    let ids: BTreeSet<_> = hits.iter().map(|(id, _)| id).collect();
                    assert_eq!(ids.len(), hits.len(), "duplicate ids in one result");
                    assert!(hits.iter().all(|(_, s)| s.is_finite()));
                    searches.fetch_add(1, Ordering::SeqCst);
                }
            });
        }
    });
    assert_eq!(searches.load(Ordering::SeqCst), 600);
}
