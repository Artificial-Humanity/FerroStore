# FerroStore (`fl-store`)

**A single-file embedded store: JSON documents with BM25 search, a durable message queue (FerroMQ), columnar tables, and HNSW vector and hybrid search, all in one `.fls` file or entirely in memory, optionally served over a Unix socket or `ws://`.**

FerroStore is an independent sub-product of the [FerroLoop](https://github.com/Artificial-Humanity/FerroLoop) suite. It can be used standalone as a Rust library, the `fl-store` crate. `FerroStore` also implements FerroLoop's `fl-core` store traits, so FerroLoop can use it as a storage backend. The file format and the wire protocol are specified in [docs/ferrostore-spec.md](docs/ferrostore-spec.md); this README is the overview, and the spec is the authority.

FerroStore is Unix-only: it takes an advisory `flock` on the file and uses Unix sockets, through `std::os::unix`.

## Storage models

One `.fls` file holds four native models (spec, "Storage models"):

| Model | What it is |
| --- | --- |
| Object / document | JSON documents addressed by collection and id, with a top-level field index and Okapi BM25 full-text search. |
| FerroMQ | A durable per-topic log with consumer-group acks and in-process subscribers. |
| Columnar | Typed tables stored as row groups with delta, run-length and dictionary encodings, and zone-map skipping on scans. |
| Vector | `f32`, `f16` and `i8` embeddings in HNSW collections, plus hybrid BM25 and vector search. |

## The `.fls` file layout

All integers are little-endian. Checksums are CRC-32 (ISO-HDLC, the zlib polynomial). The spec gives every field and offset; in outline:

| Range | Role |
| --- | --- |
| `0..4096` | Superblock slot 0 |
| `4096..8192` | Superblock slot 1 |
| `8192..` | Snapshot and WAL bytes |

* **Superblocks.** Each slot is 4096 bytes, begins with the magic `FERRSTR1` and carries a format version, a monotonic `epoch`, a CRC-32, and the offsets and lengths of the current snapshot and WAL. The current superblock is the valid slot with the highest `epoch`. Every commit writes the inactive slot with `epoch + 1` and `fsync`s, so a torn write fails its CRC and the other slot stays current.
* **Snapshot.** A copy-on-write blob with the magic `FERRSNAP`, written by `checkpoint`. Version 1 holds documents only, version 2 adds column tables, and version 3 is written when any vector collection exists. A version 1 snapshot still opens.
* **WAL.** Committed transactions are CRC-32 framed (payload length, payload CRC, payload). Each transaction has a file-wide, 1-based `seq`. Replay stops at a torn tail, and treats a bad frame that is followed by another complete frame, or a gap in `seq`, as corruption.
* **Derived state.** Secondary indexes and BM25 statistics are rebuilt from the documents after replay. They are not written to the file. Neither are the subscriber list or any sync adapter.
* **Concurrency.** Writers take `flock` (exclusive) only across the commit critical section, re-read the latest state under the lock, then append and publish the new superblock. Readers use a separate read-only descriptor and take no lock. Two `FerroStore::open` handles on one path both succeed.

## Quick start

Add the crate as a git dependency:

```toml
[dependencies]
fl-store = { git = "https://github.com/Artificial-Humanity/FerroStore.git" }
serde_json = "1"
```

```rust
use fl_store::{FerroStore, FerroStoreServer, Fusion, VectorMetric, VectorStorage};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Embedded: this process opens the file. `FerroStore::in_memory()` uses no file.
    let store = FerroStore::open("demo.fls")?;

    // Documents, the top-level field index, and BM25.
    store.put_doc("notes", "n1", &json!({"title": "durable queues", "tag": "mq"}))?;
    store.put_doc("notes", "n2", &json!({"title": "vector search", "tag": "ann"}))?;
    let by_tag = store.query_by_field("notes", "tag", "mq")?;        // ["n1"]
    let hits = store.search_bm25("notes", "queues", 10)?;            // [("n1", score)]

    // FerroMQ.
    let seq = store.publish_mq("orders", "k1", &json!({"n": 1}))?;
    let batch = store.poll_mq_group("orders", "workers", 10)?;
    store.ack_mq("orders", "workers", batch[0].seq)?;

    // Vectors and hybrid search.
    store.create_vector_collection("emb", 3, VectorStorage::F32, VectorMetric::Cosine, 16, 64, 32)?;
    store.upsert_vector("emb", "n1", &[1.0, 0.0, 0.0], None)?;
    store.upsert_vector("emb", "n2", &[0.0, 1.0, 0.0], Some(&json!({"src": "demo"})))?;
    let near = store.search_vectors("emb", &[0.9, 0.1, 0.0], 5, None)?;
    let both = store.search_hybrid("notes", "emb", "queues", &[0.9, 0.1, 0.0], 5, Fusion::Rrf { k: 60 })?;

    store.checkpoint()?;

    // Served: this process becomes the only opener of the file; clients use unix://.
    let _server = FerroStoreServer::bind_unix("demo.sock", store)?;
    let client = FerroStore::connect_uri("unix://demo.sock")?;
    let _doc = client.get_doc("notes", "n1")?;
    Ok(())
}
```

`FerroStore::connect_uri` accepts `mem://`, `file://`, `unix://` and `ws://` URIs (see [Served mode](#served-mode)). `wss://` is recognized and refused with an error, and any other scheme is an error.

## Document / BM25 engine

`put_doc`, `get_doc`, `delete_doc`, `query_by_field` and `search_bm25` (spec, "Documents"):

* A document is any JSON value stored under `(collection, id)`. `put_doc` replaces; `delete_doc` of a missing id succeeds; `get_doc` of a missing id is `Ok(None)`.
* The field index covers top-level object fields only: strings by value, numbers by their JSON text, bools as `true` and `false`. Null, arrays and objects are not indexed. `query_by_field` returns ids in ascending order.
* BM25 is Okapi with `k1 = 1.2` and `b = 0.75`. Tokens are lowercased Unicode alphanumeric runs. Every string in a document is indexed, recursively through object values and array elements, and object keys are not. Results are ordered by score descending, then id ascending, and truncated to `limit`.

## FerroMQ: durable queue and event log

`publish_mq`, `poll_mq_group`, `ack_mq` and `subscribe_mq` (spec, "FerroMQ"):

* Messages are stored as documents, so a publish is one ordinary durable transaction. Each topic has its own sequence, starting at 1 and increasing by 1 per publish. A sequence is not reused after its message document is deleted.
* `poll_mq_group` returns the oldest messages that a consumer group has not acked, up to `limit`, without acking them. A second poll without an ack returns the same messages. Each group has its own ack set. `ack_mq` is durable and idempotent.
* `subscribe_mq` returns an in-process `Receiver` of later publishes on that handle. It does not replay history, and it is not durable.

## Served mode

A `FerroStoreServer` owns the store and is the only opener of the `.fls` file; clients never open the file or take a lock (spec, "Served mode").

* `FerroStoreServer::bind_unix(path, store)` serves a Unix socket. A frame is a 4-byte big-endian length followed by that many bytes of UTF-8 JSON, between 1 byte and 16 MiB.
* `FerroStoreServer::bind_tcp(addr, store)` serves RFC 6455 WebSocket. One JSON object is one message, client frames are masked, and the payload is limited to 16 MiB. A client connects with a `ws://host:port` URI.
* Each call is `{"id": 1, "method": "put_doc", "params": {...}}`, answered by `{"id": 1, "ok": true, "result": ...}` or `{"id": 1, "ok": false, "error": "..."}`. The spec lists every method and its parameters: documents, FerroMQ, columnar, vectors, hybrid search, `read_state` and the compare-and-swap `commit_ops`.
* A client built by `connect_uri("unix://...")` or `connect_uri("ws://...")` exposes the same `FerroStore` methods. A remote write retries the compare-and-swap until it succeeds, so concurrent remote writers do not lose updates.
* `FerroStore::auto(path)` connects to the default socket for a store file (`default_socket_path_for_store` appends `.sock` to the file name), and opens the file in embedded mode when nothing is listening.

## Columnar analytics

`create_column_table`, `append_column_rows` and `scan_column_table` (spec, "Phases", Phase 3):

* Column types are `Int64`, `Float64`, `Bool`, `Utf8`, `Bytes` and `TimestampMs`. Null is allowed in every column.
* One non-empty append is one row group. The codec is chosen per column chunk when the group is written, from the column type and the data (delta, delta with run-length, dictionary, run-length or plain), and is stored in the row-group blob. Nulls are a bitset, not encoded as values.
* Each row-group column chunk keeps a zone map (min, max and null count). A scan with `ColumnFilter`s (each a column name plus a `ColumnPredicate`: `Eq`, `Lt`, `Le`, `Gt`, `Ge` or `Between`; the filters are combined with AND) skips a row group before decoding when a zone map excludes the predicate. `ColumnScan` reports `row_groups_total` and `row_groups_skipped`.
* `attempt_rollup` and `gate_run_rollup` aggregate FerroLoop's stored attempt and gate-run documents through an in-memory columnar scan.

## HNSW vector and hybrid search

`create_vector_collection`, `upsert_vector`, `get_vector`, `delete_vector`, `search_vectors`, `search_hybrid` and `search_knowledge` (spec, "Phases", Phase 4):

* A vector collection is separate from document collections. It fixes `dim`, a storage type (`f32`, `f16` or `i8`), a metric (cosine, L2 or dot product), and the HNSW parameters `M`, `ef_construction` and `ef_search`. There is no hidden default `M`.
* The HNSW level of an id is deterministic (derived from a SHA-256 hash of the id). Distances are computed on dequantized `f32` values and accumulated in `f64`. Scores are higher-is-better.
* `search_vectors` takes an optional `ef` and uses the collection's `ef_search` otherwise.
* `search_hybrid` fuses at most `limit` BM25 hits with at most `limit` vector hits, with `Fusion::Rrf { k }` (reciprocal rank fusion) or `Fusion::Weighted { bm25, vector }` (min-max normalized, weighted sum). `search_knowledge` uses the same fusion and returns the vector metadata and the document with each hit.
* FerroStore does not call an embedding model. The caller stores a document with `put_doc` and its embedding with `upsert_vector` under the same id.

## Sync adapters

A process may attach one `SyncAdapter` to replicate commits to a cloud service or a peer (spec, "Sync adapter"). The store links no cloud SDK; you implement the trait.

* `SyncMode` is `WriteThrough` (call `on_commit` after each commit), `WriteBehind` (queue until `sync_flush`), `LocalCache` (queue, and fill a document miss from `fetch_document`) or `CloudPreCache` (queue and fill misses, as `LocalCache` does).
* A commit becomes a `SyncBatch` of its sequence and ops. A failed `on_commit` leaves the commit applied and the batch pending, in order. `sync_flush` sends pending batches oldest first. `sync_pull` and `sync_prewarm` need an attached adapter and an empty pending buffer, and apply the adapter's batches only if every batch is valid and in sequence.
* The adapter, the mode, the pending batches and `synced_seq` are process memory. They are not written to the `.fls` file.
* `RecordEnvelope` and `EnvelopeKind` give a lossless per-op record form of a batch for replication.

## Relation to FerroLoop and the redb store

The crate also contains the redb-backed `RedbStore` and FerroLoop's store roles over it (catalog, routing, tracker, ledger, state, messaging and workflow roles, among others), and `FerroStore` implements those same `fl-core` store traits. `migrate_redb_to_ferrostore` copies a `RedbStore` into an empty `FerroStore`. These parts exist for FerroLoop and are not needed to use FerroStore as a document, queue, columnar or vector store.

### Why `fl-core` is pinned

`fl-store` depends on `fl-core`, FerroLoop's core crate, for its error and store trait types. `fl-core` is a crate in the FerroLoop workspace, so `Cargo.toml` takes it from the FerroLoop repository, pinned to a commit with `rev = "af85c42feb097fa2e1e483391c774a2044dfce0c"`, and `Cargo.lock` is committed. The pin keeps this repository's build reproducible and tied to the exact core API the extracted code was tested against; move it deliberately, and run the verification commands below when you do. The dev-dependency on `fl-core` enables its `conformance` feature for the store test suite.

## Installation

FerroStore is a library crate and needs Rust 1.98 or newer (edition 2024):

```bash
git clone https://github.com/Artificial-Humanity/FerroStore.git
cd FerroStore
cargo build
```

To use it from another project, add the git dependency shown in [Quick start](#quick-start). Cargo fetches `fl-core` from the FerroLoop repository, so a build needs network access to GitHub the first time.

## Verification

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --workspace -- -D warnings
cargo test --workspace
```

The tests use temporary directories and in-process servers on Unix sockets and loopback TCP. They need no external service.

## License

Apache-2.0. See [LICENSE](LICENSE).
