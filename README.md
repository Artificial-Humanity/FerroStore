# FerroStore (`fl-store`)

**A single-file embedded store: JSON documents with BM25 search, a durable message queue (FerroMQ), columnar tables, and HNSW vector and hybrid search, all in one `.fls` file or entirely in memory, optionally served over a Unix socket, `ws://` or `wss://`, with a `ferrostore` command-line tool.**

FerroStore is an independent sub-product of the [FerroLoop](https://github.com/Artificial-Humanity/FerroLoop) suite. It can be used standalone as a Rust library, the `fl-store` crate. `FerroStore` also implements FerroLoop's `fl-core` store traits, so FerroLoop can use it as a storage backend. The file format and the wire protocol are specified in [docs/ferrostore-spec.md](docs/ferrostore-spec.md); this README is the overview, and the spec is the authority.

FerroStore is Unix-only: it takes an advisory `flock` on the file and uses Unix sockets, through `std::os::unix`.

The crate builds a library and the `ferrostore` binary. The binary and the shared argument types in `fl_store::cli::args` are behind the `cli` feature, which is on by default.

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

* **Superblocks.** Each slot is 4096 bytes, begins with the magic `FERRSTR1` and carries a container format version (this build writes `2` and still reads `1`), a monotonic `epoch`, a CRC-32, the offsets and lengths of the current snapshot and WAL, an `engine` field and `incompat` feature bits. The current superblock is the valid slot with the highest `epoch`. Every publish writes the inactive slot with `epoch + 1` and `fsync`s, so a torn write fails its CRC and the other slot stays current. A slot written by a newer build is refused with `StoreError::ContainerFormat`, not read as torn.
* **Snapshot.** A copy-on-write blob with the magic `FERRSNAP`. A live snapshot's bytes are never overwritten. Version 1 holds documents only, version 2 adds column tables, and version 3 is written when any vector collection exists. A version 1 snapshot still opens.
* **WAL.** Committed transactions are CRC-32 framed (payload length, payload CRC, payload). Each transaction has a file-wide, 1-based `seq`. Replay stops at a torn tail, and treats a bad frame that is followed by another complete frame, or a gap in `seq`, as corruption.
* **Derived state.** Secondary indexes and BM25 statistics are rebuilt from the documents after replay. They are not written to the file. Neither are the subscriber list or any sync adapter.
* **Permissions.** A new store file is mode `0600` from its first byte, and a missing parent directory is made mode `0700`. An existing file keeps its mode; one that other users can read gets a warning.

## Engine

The spec sections "File layout", "Compaction" and "Concurrency" are the authority; in outline:

* **Copy-on-write.** No extent of a file is rewritten in place. A commit appends to the WAL and then publishes a new superblock; a `checkpoint` writes a new snapshot past both the live snapshot and the live WAL, `fsync`s it, and then publishes. A crash at any point leaves readers on the previous superblock.
* **Epoch cache.** Each handle keeps the decoded state of the file and the epoch it was read at. A read compares the superblocks with that epoch and, when a writer has published since, decodes and applies only the new WAL frames, copy-on-write, to the cached state. A full load runs only when the new frames do not follow on from the cached ones (after a checkpoint or a compaction, for example). Readers use a separate read-only descriptor and take no lock.
* **Group commit.** Concurrent writers to one store queue up. The first writer leads: it takes the write lock and the exclusive `flock`, refreshes, and takes what is queued (at most 64 members) as its group. Each member's body runs on its own thread against the group's running state, and the leader publishes all the frames with one data `fsync` and one superblock write. A member's failure drops only its own writes. Nobody waits to batch.
* **Transactions.** `FerroStore::transaction` runs a body over a `Transaction` and commits all its writes as one sequence and one WAL frame, or none of them. On disk the body runs under the write lock, inside a group commit.
* **Auto-checkpoint.** A commit whose WAL has grown past `max(snapshot_length, 4 MiB)` checkpoints after the commit's lock window, so no other writer's result waits for it. A failed auto-checkpoint does not fail the commit; it is tried again once the WAL has grown by another threshold.
* **Compaction.** Checkpoints leave the replaced snapshots and WALs in the file as dead bytes. `FerroStore::compact` writes the current state into a new sibling file and renames it over the store, keeping the epoch rising and `last_seq` unchanged; other handles detect the swap by `(dev, ino)` and move to the new file. A checkpoint that leaves more dead bytes than `max(live bytes, 64 MiB)` is followed by a compaction automatically. `FerroStore::stats` reports the file, live and dead bytes, the epoch and `last_seq`.
* **Durability.** A failed `fsync` poisons the handle: later writes through it are refused rather than trusted. A handle also makes the directory entry that names its file durable before its first commit into it.

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

`FerroStore::connect_uri` accepts `mem://`, `file://`, `unix://`, `ws://` and `wss://` URIs (see [Served mode](#served-mode)); any other scheme is an error. `FerroStore::connect_with(uri, &ConnectOptions)` adds a credential and TLS options.

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

* **Transports.** `FerroStoreServer::bind_unix(path, store)` serves a Unix socket (mode `0600`). A frame is a 4-byte big-endian length followed by that many bytes of UTF-8 JSON, between 1 byte and 16 MiB. `FerroStoreServer::bind_tcp(addr, store, auth)` serves RFC 6455 WebSocket, one JSON object per message, at `ws://host:port`, or at `wss://host:port` when the `ServerAuth` carries a TLS certificate and key.
* **Calls.** Each call is `{"id": 1, "method": "put_doc", "params": {...}}`, answered by `{"id": 1, "ok": true, "result": ...}` or `{"id": 1, "ok": false, "error": "..."}`. The spec lists every method and its parameters. Every write method takes an optional request id, so a resend after a lost reply never applies a write twice.
* **Authentication.** Every connection authenticates before its first request. A Unix connection is checked by peer uid (`SO_PEERCRED`), and optionally by a pre-shared key. A TCP bind needs a PSK, a bearer-token file, or both: tokens are stored as SHA-256 hashes with a `read`, `write` or `admin` scope, and the PSK runs a mutual HMAC-SHA256 challenge-response that, over `wss://`, is bound to the TLS session's exporter. Secret files must be mode `0600` or stricter. Each method needs a scope; unauthenticated connections are bounded in time, size and number.
* **TLS.** The server speaks TLS 1.3 and 1.2 through `rustls` with the `ring` provider. A `wss://` client verifies the server against the bundled Mozilla roots (`webpki-roots`), a CA file, or a pinned SubjectPublicKeyInfo SHA-256. Without TLS, `bind_tcp` binds only loopback or an address of an interface named with `ServerAuth::allow_plaintext_on`, and never `0.0.0.0` or `::`.
* **Remote transactions and read sessions.** A remote client never receives the whole state. A remote transaction runs its body on the client over a read session that pins one server snapshot; each read is answered with a hash, and `commit_txn` sends the read set with the ops. The server re-checks every read under the write lock and commits only when all still match, so remote transactions are serializable. A lost commit runs the body again over a new session, at most 8 attempts, and then returns `StoreError::Conflict`. Sessions expire after 30 s idle or 5 minutes, and are capped per connection, per credential and per server.
* **Auto.** `FerroStore::auto(path)` connects to the default socket for a store file (`default_socket_path_for_store` appends `.sock` to the file name), and opens the file in embedded mode when nothing is listening.

A served store with a token over TLS:

```rust
use fl_store::{ConnectOptions, FerroStore, FerroStoreServer, ServerAuth};

fn serve() -> Result<(), Box<dyn std::error::Error>> {
    let auth = ServerAuth::new()
        .with_token_file("tokens")?            // "<sha256 hex> <scope> <label>" per line
        .with_tls_files("cert.pem", "key.pem")?;
    let _server = FerroStoreServer::bind_tcp("0.0.0.0:7400", FerroStore::open("demo.fls")?, auth)?;
    let client = FerroStore::connect_with(
        "wss://store.example:7400",
        &ConnectOptions::token_file("client-token").ca_file("ca.pem"),
    )?;
    let _doc = client.get_doc("notes", "n1")?;
    Ok(())
}
```

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

## The `ferrostore` command

`cargo install --git https://github.com/Artificial-Humanity/FerroStore.git fl-store` installs the `ferrostore` binary (spec, "Command line"). A store is named with `--store`; a path `x.redb` or `x` names `x.fls`. A failure prints `error: ...` on stderr and exits 2.

| Command | What it does |
| --- | --- |
| `status --store PATH` | The engine, files, size, live and dead bytes and format version. Read-only. `--json` |
| `checkpoint --store PATH` | Folds the WAL into a new snapshot |
| `compact --store PATH` | Gives back dead bytes (see [Engine](#engine)). `--json` |
| `migrate --store PATH.redb` | A verified copy of a `.redb` store into the sibling `.fls`; the `.redb` is kept read-only. `--dry-run`, `--json` |
| `verify --store PATH.redb` | Compares a `.redb` and its `.fls` table by table. Exit 1 when they differ |
| `serve --store PATH` | Serves the store on a Unix socket (`--socket`, default `<file>.fls.sock`) or on TCP (`--tcp ADDR` with `--psk-file` and `--server-id`, `--token-file`, `--tls-cert` and `--tls-key`, `--allow-plaintext-on IFACE`) until SIGINT or SIGTERM. `--create` makes a missing store |
| `get --store PATH\|URI COLLECTION ID` | Prints a document as JSON. Exit 1 when it is missing |
| `put --store PATH\|URI COLLECTION ID JSON\|-` | Puts a document (`-` reads stdin). `--create` makes a missing store |
| `query --store PATH\|URI COLLECTION FIELD VALUE` | Field query; `--bm25` for BM25 text search, `--vector --values JSON` for vector search, `-k N` for the number of hits |
| `inspect PATH` | Both superblock slots, the schema version, counts and live and dead bytes, read without a lock. `--json` |
| `bench` | Times put, get, scan and vector workloads on fresh stores. `--json` |

`get`, `put` and `query` take a path (opened embedded) or a `unix://`, `ws://` or `wss://` URI, with the client flags `--psk-file` and `--server-id`, `--token-file`, `--ca-file` or `--pin-spki`, and `--allow-plaintext`. The `fl store` command of FerroLoop calls the same functions and prints the same output.

## Relation to FerroLoop and the redb store

The crate also contains the redb-backed `RedbStore` and FerroLoop's store roles over it (catalog, routing, tracker, ledger, state, messaging and workflow roles, among others), and `FerroStore` implements those same `fl-core` store traits. `migrate_redb_to_ferrostore` copies a `RedbStore` into an empty `FerroStore`. These parts exist for FerroLoop and are not needed to use FerroStore as a document, queue, columnar or vector store.

### Why `fl-core` is pinned

`fl-store` depends on `fl-core`, FerroLoop's core crate, for its error and store trait types. `fl-core` is a crate in the FerroLoop workspace, so `Cargo.toml` takes it from the FerroLoop repository, pinned to a commit with `rev = "2573b2ad01ed8a8ed45fe37f07d4aa5403ca439c"`, and `Cargo.lock` is committed. The pin keeps this repository's build reproducible and tied to the exact core API the extracted code was tested against; move it deliberately, and run the verification commands below when you do. The dev-dependency on `fl-core` enables its `conformance` feature for the store test suite.

## Installation

FerroStore needs Rust 1.98 or newer (edition 2024):

```bash
git clone https://github.com/Artificial-Humanity/FerroStore.git
cd FerroStore
cargo build
```

To use it from another project, add the git dependency shown in [Quick start](#quick-start); add `default-features = false` to leave out the `cli` feature and `clap`. Cargo fetches `fl-core` from the FerroLoop repository, so a build needs network access to GitHub the first time.

## Verification

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --workspace -- -D warnings
cargo test --workspace
```

The tests use temporary directories and in-process servers on Unix sockets and loopback TCP, and the test TLS certificates in `src/ferrostore/testdata/tls` (test-only keys, see the README there). They need no external service.

## License

Apache-2.0. See [LICENSE](LICENSE).
