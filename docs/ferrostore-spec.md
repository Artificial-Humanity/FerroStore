# FerroStore

FerroStore keeps an object/document collection, a FerroMQ log, columnar tables, and vector collections in one `.fls` file, or entirely in memory. Phase 2 serves that store over a Unix socket or `ws://`. Phase 4 stores `f32`, `f16`, and `i8` embeddings in the same file and searches them with HNSW.

Integers on disk are little-endian. The CRC-32 is ISO-HDLC: polynomial `0xEDB88320`, init `0xFFFFFFFF`, xorout `0xFFFFFFFF` (the reflected zlib polynomial). A checksum covers the bytes named below and nothing else.

## File layout

A `.fls` file begins with two 4096-byte superblocks.

| Range | Role |
| --- | --- |
| `0..4096` | Superblock slot 0 |
| `4096..8192` | Superblock slot 1 |
| `8192..` | Snapshot and WAL bytes |

Creating a missing file writes slot 0 and leaves slot 1 as zeros, then `fsync`s:

| Field | Initial value |
| --- | --- |
| `epoch` | `1` |
| `snapshot_offset` | `8192` |
| `snapshot_length` | `0` |
| `wal_offset` | `8192` |
| `wal_length` | `0` |
| `snapshot_seq` | `0` |

The create path holds the same short exclusive `flock` as a commit, only while it publishes that first superblock.

### Superblock

Each slot is 4096 bytes.

| Offset | Type | Contents |
| --- | --- | --- |
| `0..8` | bytes | Magic `FERRSTR1` |
| `8..12` | `u32` | Format version `1` |
| `12..20` | `u64` | `epoch`, monotonic, starting at `1` |
| `20..24` | `u32` | CRC-32 of this 4096-byte block with bytes `20..24` set to zero |
| `24..32` | `u64` | `snapshot_offset` (absolute) |
| `32..40` | `u64` | `snapshot_length` |
| `40..48` | `u64` | `wal_offset` (absolute) |
| `48..56` | `u64` | `wal_length` |
| `56..64` | `u64` | `snapshot_seq` |
| `64..4096` | bytes | Zeros |

`snapshot_seq` is the highest committed transaction sequence included in the snapshot, or `0` when no snapshot holds a commit.

A slot is valid only when magic, version, padding, and CRC all match. Any other bytes are ignored. The current superblock is the valid slot with the highest `epoch`. A torn write fails the CRC, so the other slot stays current. Equal epochs on two valid slots are corruption.

Every committed WAL append and every `checkpoint` publishes a new generation: write the inactive slot (the slot that does not hold the highest valid epoch) with `epoch + 1`, then `fsync`. The live slot is never rewritten in place.

### Snapshot

A snapshot is a copy-on-write blob. A live snapshot's bytes are never overwritten.

| Offset | Type | Contents |
| --- | --- | --- |
| `0..8` | bytes | Magic `FERRSNAP` |
| `8..12` | `u32` | Version `1` with no column tables and no vector collections, version `2` with column tables and no vector collections, or version `3` when any vector collection exists |
| `12..20` | `u64` | `seq`, the last committed sequence in the blob |
| `20..24` | `u32` | `count` of documents |
| `24..` | records | `count` documents, then column tables when the version is `2` or `3`, then vector collections when the version is `3` |
| tail | `u32` | CRC-32 of every preceding snapshot byte |

Each document record is `u32` length + bytes for the collection, then the same for the id, then the same for the raw JSON document. Version `1` ends there: `decode` calls `finish()` with no trailer, and a document-only `encode_snapshot` is byte-identical to that layout. A version `1` snapshot still opens.

Version `2` is written only when at least one column table exists and no vector collection exists. After the documents it stores `u32` table count, then tables in name order. Each table is a length-prefixed name, `u32` column count, each column as a length-prefixed name plus a `u8` type tag (`1` int64, `2` float64, `3` bool, `4` utf8, `5` bytes, `6` timestamp_ms), `u32` row-group count, and each row group as a length-prefixed blob (the same bytes the WAL stored). A version `2` snapshot with no tables is corrupt.

Version `3` is written when any vector collection exists, including an empty one. After the documents it stores the version `2` table section. `table_count` may be `0`. It then stores `u32` collection count, which must be at least `1` (0 is corrupt), and collections in name order. Each collection is a length-prefixed name, `u32` dim, `u8` storage tag (`1` f32, `2` f16, `3` i8), `u8` metric tag (`1` cosine, `2` l2, `3` dot), `u32` M, `u32` ef_construction, `u32` ef_search, `u32` vector count, then vectors in id order. Each vector is a length-prefixed id, a length-prefixed byte string, a `u8` metadata flag (`0` absent, `1` present) and, when the flag is `1`, a length-prefixed UTF-8 JSON value. Then an entry flag (`0` or `1`) and, when it is `1`, a length-prefixed entry id. Then, for each vector in id order, a `u8` level and, for each layer `0..=level`, a `u32` neighbor count and that many length-prefixed neighbor ids. Decode checks names, ranges, byte length, known tags, that the level matches the deterministic HNSW formula, and that neighbors exist, are unique, are not self, respect the layer's max links, and are bidirectional. Each layer's neighbor list must already be ordered by graph distance, then id. The entry id must be the canonical live node (highest level, then lowest id) when the collection is non-empty. The entry flag is `0` when it is empty. A bad snapshot is corrupt.

`checkpoint` writes a new blob at an offset at or past the end of the file, and past both the live snapshot and the live WAL, so it does not overlap either. It `fsync`s the blob, then publishes a superblock that points `snapshot_offset` / `snapshot_length` at the new blob, sets `snapshot_seq` to the last committed sequence, sets `wal_offset` to the end of the new blob, and sets `wal_length` to `0`. A crash after the blob is written and before that publish leaves readers on the previous superblock.

An empty snapshot (`snapshot_length == 0`) contributes no documents. Its sequence is the superblock's `snapshot_seq`. On `mem://`, `checkpoint` is a successful no-op.

### WAL frame

The committed WAL is the byte range `wal_offset .. wal_offset+wal_length`. Frames are packed with no padding.

| Offset | Type | Contents |
| --- | --- | --- |
| `0..4` | `u32` | Payload length |
| `4..8` | `u32` | CRC-32 of the payload bytes only |
| `8..` | bytes | Payload |

One payload is one transaction:

| Field | Type | Contents |
| --- | --- | --- |
| `seq` | `u64` | File sequence, `1`-based |
| `op_count` | `u32` | Number of operations |
| ops | | Repeated operations |

Each operation starts with a `u8` tag. The tag is read first. Tag `1` put and tag `2` delete then store `u32` collection length, collection UTF-8, `u32` id length, id UTF-8. A put also stores `u32` document length and the raw JSON bytes. A delete stops after the id. Tag `3` creates a column table: length-prefixed name, `u32` column count, then each column as a length-prefixed name and a `u8` type tag (the same tags as a snapshot). Tag `4` appends one row group: length-prefixed table name, then a length-prefixed blob. Tag `5` creates a vector collection: length-prefixed name, `u32` dim, `u8` storage, `u8` metric, `u32` M, `u32` ef_construction, `u32` ef_search. Tag `6` upserts a vector: length-prefixed collection, length-prefixed id, length-prefixed quantized bytes, `u8` metadata flag (`0` absent, `1` present) and, when the flag is `1`, a length-prefixed UTF-8 JSON value. Tag `7` deletes a vector: length-prefixed collection, length-prefixed id. Any other tag is corruption (`unknown wal op tag`). Tags `1` through `4` stay byte-identical to the earlier phases.

`seq` increases by exactly 1 for each committed transaction in the file. Puts, deletes, FerroMQ publishes, and FerroMQ acks are each one transaction. The sequence is a property of the file, not of a topic.

Replay walks the committed range:

- Fewer than 8 header bytes at the tail, or a declared payload that is not fully inside the range, is a torn tail. Stop. Ignore that partial frame. Earlier frames stay committed.
- A CRC mismatch on the last frame (its declared end is the end of the WAL range, or the next byte would fall past the range) is a torn tail. Ignore it.
- A CRC mismatch with another complete frame after it — a following header and a payload that fits in the range — is `StoreError::Backend` and the message contains `corrupt`.
- Apply frames with `seq > snapshot_seq` in order onto the snapshot. The first applied sequence must be `snapshot_seq + 1`, and each next frame must be the next integer. A gap or a duplicate is corruption (`StoreError::Backend`, message contains `corrupt`).
- A CRC-valid frame whose payload is not a well-formed transaction is the same corruption error.

Secondary indexes and BM25 statistics are rebuilt from the documents after replay. They are not written to the WAL. The in-process subscriber list is not written either.

## Concurrency

The write descriptor is opened read/write. `flock` is the libc call (`LOCK_EX = 2`, `LOCK_UN = 8`) on that descriptor. It is held only across the critical section, not for the life of the handle.

Commit:

1. Build the operation in memory.
2. Take `LOCK_EX`.
3. Re-read the latest superblock, snapshot, and WAL. Another process may have committed.
4. Apply the operation to that state. Assign `seq = last_seq + 1`.
5. Append one CRC32-framed transaction at `wal_offset + wal_length`.
6. `fsync` the data.
7. Publish the inactive superblock with the new `wal_length` and `epoch + 1`. Snapshot fields stay as they were. `fsync`.
8. Unlock.
9. Then notify in-process FerroMQ subscribers. Notification is outside the lock and is not durable.

`checkpoint` holds the same exclusive lock while it reads, writes the new snapshot, and publishes.

Readers open a separate `O_RDONLY` description and do not `flock`. A read loads both superblocks, selects the highest valid epoch, loads that snapshot, and applies WAL frames with `seq > snapshot_seq`. A torn superblock fails its CRC and is ignored. Two `FerroStore::open` handles on one path both succeed; the second open is not `StoreError::Unreachable`. A reader observes a writer's committed document without taking a file lock.

`mem://` uses the same document map and the same APIs, with no file and no `flock`.

## Storage models

Phase 1 implements two of the four native models.

| Model | Phase | Behavior in this version |
| --- | --- | --- |
| Object / document | 1 | JSON documents addressed by collection and id. Top-level field index and Okapi BM25. |
| FerroMQ | 1 | Durable per-topic log, consumer-group acks, in-process subscribers. |
| Columnar | 3 | Row groups, delta / RLE / dictionary encodings, and zone-map skips. See Phases. |
| Vector | 4 | `f32` / `f16` / `i8` embeddings, HNSW, and hybrid BM25+vector search. See Phases. |

## Connection URI

`FerroStore::connect_uri` accepts:

| URI | Result |
| --- | --- |
| `mem://`, `mem:///` | `in_memory()` |
| `file:///tmp/x.fls` | `open` of absolute `/tmp/x.fls` |
| `file://localhost/tmp/x.fls` | `open` of absolute `/tmp/x.fls` |
| `file://rel/x.fls` | `open` of relative `rel/x.fls` |
| `unix:///tmp/x.sock`, `unix://localhost/tmp/x.sock` | Client of absolute `/tmp/x.sock`. Served |
| `unix://rel/x.sock` | Client of relative `rel/x.sock`. Served |
| `ws://127.0.0.1:9`, `ws://127.0.0.1:9/store` | Client of that WebSocket endpoint. Served |
| `wss://...` | `StoreError::Backend`. The message names `wss` and does not say the scheme is unknown |
| Anything else, including `""` and `http://` | `StoreError::Backend` whose message contains `unknown` and does not describe the scheme as deferred |

`file://` and `unix://` paths are not percent-decoded. Only the host `localhost` is stripped; any other non-empty host is part of a relative path. A failed `unix://` or `ws://` connect is `StoreError::Backend`. It is not an unknown scheme.

## Served mode

Phase 2 is this served mode. The process that binds a `FerroStoreServer` is the only opener of the `.fls` file. Clients do not take `flock` and do not open the file.

A Unix frame is a 4-byte big-endian length followed by that many bytes of UTF-8 JSON. The length is `1..=16MiB` (`16 * 1024 * 1024`). A length of zero or a longer payload is an error.

A WebSocket message is one JSON object in one RFC 6455 frame. Client frames are masked. Server frames are not. The payload has no length prefix. The JSON payload is `1..=16MiB`. `wss://` is recognized and refused with a message that does not say the scheme is unknown.

Each call is one JSON object:

```json
{"id": 1, "method": "put_doc", "params": {}}
```

`id` is an integer chosen by the client. Success is `{"id": 1, "ok": true, "result": ...}`. Failure is `{"id": 1, "ok": false, "error": "..."}`.

| Method | Params | Result |
| --- | --- | --- |
| `put_doc` | `collection`, `id`, `doc` | `null` |
| `get_doc` | `collection`, `id` | `null` when the id is missing. A present value, including JSON `null`, is `{"doc": <value>}` |
| `delete_doc` | `collection`, `id` | `null` |
| `query_by_field` | `collection`, `field`, `value` | array of id strings |
| `search_bm25` | `collection`, `query`, `limit` | array of `[id, score]` |
| `publish_mq` | `topic`, `key`, `payload` | the sequence number |
| `poll_mq_group` | `topic`, `group`, `limit` | array of `{"seq", "key", "payload"}` |
| `ack_mq` | `topic`, `group`, `seq` | `null` |
| `checkpoint` | none | `null` |
| `subscribe_mq` | `topic` | `null`, then push frames |
| `read_state` | none | `{"last_seq": <u64>, "docs": [{"collection", "id", "doc"}, ...], "tables": [...], "vectors": [...]}`. Documents are in `(collection, id)` order. A stored JSON `null` is `"doc": null`. Missing documents are omitted. `tables` is name order: `{name, columns: [{name, ty}], groups}`, and each group is an array of bytes `0..=255`. `vectors` is name order. Each collection is `{name, dim, storage, metric, m, ef_construction, ef_search, vectors, entry, links}`. `storage` is `f32`, `f16`, or `i8`. `metric` is `cosine`, `l2`, or `dot`. Each stored vector is `{id, bytes, metadata?}` with `bytes` an array of integers `0..=255`. `metadata` is omitted when absent and present, including JSON `null`, when stored. `entry` is the entry id or `null`. `links` is `{id, level, layers}` and each layer is an array of neighbor ids |
| `create_vector_collection` | `name`, `dim`, `storage`, `metric`, `m`, `ef_construction`, `ef_search` | `null` |
| `upsert_vector` | `collection`, `id`, `values` (finite JSON numbers), `metadata` optional | `null`. Absent `metadata` stores none. Present `metadata`, including `null`, is stored |
| `delete_vector` | `collection`, `id` | `null` |
| `get_vector` | `collection`, `id` | `null` when the id is missing. A present value is `{"values": [<numbers>]}` plus `metadata` only when it was stored |
| `search_vectors` | `collection`, `values`, `limit`, optional `ef` | `[{"id", "score"}]`. `score` is a finite JSON number |
| `search_hybrid` | `doc_collection`, `vector_collection`, `query`, `values`, `limit`, `fusion` | `[{"id", "score"}]` |
| `search_knowledge` | same as `search_hybrid` | `[{"id", "score", "metadata", "doc"}]`. `metadata` is `null` when absent. `doc` is `null` when the document is missing |
| `create_column_table` | `name`, `columns` (`[{name, ty}]`; `ty` is `int64`, `float64`, `bool`, `utf8`, `bytes`, or `timestamp_ms`) | `null` |
| `append_column_rows` | `name`, `rows` (array of arrays of cells) | `null` |
| `scan_column_table` | `name`, `filters` (`[{column, op, value}]`; `op` is `eq`, `lt`, `le`, `gt`, `ge`, or `between`. `between` uses `low` and `high` instead of `value`) | `{"rows", "row_groups_total", "row_groups_skipped"}` |
| `attempt_rollup` | `project` (IRI string, or JSON `null` for every attempt) | `count`, `cost_usd_micros`, `tokens_in`, `tokens_out`, `duration_ms` |
| `gate_run_rollup` | `gate` (IRI string, or JSON `null` for every run) | `pass`, `fail`, `error`, `duration_ms`, `cost_usd_micros` |
| `commit_ops` | `expected_seq`, `ops` | `true` when the compare-and-swap commits. `false` when `last_seq` is not `expected_seq`, and nothing is written. A bad op is an error, not `false` |

`subscribe_mq` does not replay history. A later publish on that topic is pushed on the same connection as a frame with no `id`:

```json
{"method": "mq", "params": {"topic": "orders", "record": {"seq": 1, "key": "k", "payload": {}}}}
```

`read_state` is an atomic snapshot of the document map, the columnar tables, and the vector collections. `commit_ops` is a compare-and-swap of `ops` against `expected_seq`. Each op is `{"op":"put","collection","id","doc"}`, `{"op":"delete","collection","id"}`, `{"op":"create_column_table","name","columns"}`, `{"op":"append_column_group","name","bytes"}`, `{"op":"create_vector_collection","name","dim","storage","metric","m","ef_construction","ef_search"}`, `{"op":"upsert_vector","collection","id","bytes","metadata"?}`, or `{"op":"delete_vector","collection","id"}`. `bytes` is an array of integers `0..=255`. An upsert stores quantized bytes, not logical `f32` values. Absent `metadata` stores none. `false` means `last_seq` did not match, and the call writes nothing. A bad op is an error, not `false`. A non-empty success is one WAL frame. A remote `with_write` retries until the compare-and-swap succeeds, so concurrent remote writers do not lose updates. The body may run more than once on a remote handle. A remote `with_write` that only puts documents still works, and a stored JSON `null` stays distinct from a missing document. `fl_core::store` traits on `unix://`, `ws://`, and a connected `FerroStore::auto` use this path.

`fusion` is `{"method":"rrf","k":60}` (`k` optional, default 60, and `k >= 1`) or `{"method":"weighted","bm25":<finite >= 0>,"vector":<finite >= 0>}`. At least one weighted weight is greater than 0. `unix://` and `ws://` use the same JSON. On a remote handle the vector methods call these RPCs and return before a local transaction.

A columnar cell is `{"t":"null"}`, `{"t":"i64","v":1}`, `{"t":"f64","v":1.5}`, `{"t":"bool","v":true}`, `{"t":"utf8","v":"a"}`, `{"t":"bytes","v":[1,2]}` (numbers `0..=255`, not base64), or `{"t":"ts","v":1}`. A finite `f64`, including `-0.0`, is a JSON number and has no `bits` field. A non-finite `f64` is `{"t":"f64","v":"NaN","bits":<u64>}`, `{"t":"f64","v":"Infinity","bits":<u64>}`, or `{"t":"f64","v":"-Infinity","bits":<u64>}`, and `bits` is that value's `to_bits()`. A missing `bits` means canonical `NaN`, `Infinity`, or `-Infinity`. When `bits` is present it is an integer in `0..=u64::MAX` and `f64::from_bits` must be the same class as `v` (a `NaN` pattern with `v` of `Infinity` is an error). A finite number must not use a string `v`. `unix://` and `ws://` use the same JSON. On a remote handle the five columnar methods call these RPCs and do not also run a local transaction.

## Auto

`default_socket_path_for_store(path)` appends `.sock` to the file name. `dir/store.fls` is `dir/store.fls.sock`. A path with no file name (`/` or `..`) is that path joined with `ferrostore.sock`.

`FerroStore::auto(path)` connects to that socket. When the connect is accepted, the handle is a client of that server, and that connection is the client: it does not connect again. A refused connect — nothing listening, or the socket file missing — opens `path` in embedded mode and is not an error. An error from `open` is still returned.

## Documents

```text
put_doc(collection, id, doc) -> Result<(), StoreError>
get_doc(collection, id) -> Result<Option<Value>, StoreError>
delete_doc(collection, id) -> Result<(), StoreError>
query_by_field(collection, field, value) -> Result<Vec<String>, StoreError>
search_bm25(collection, query, limit) -> Result<Vec<(String, f64)>, StoreError>
```

`put_doc` replaces the document at `(collection, id)`. `delete_doc` of a missing id is success and still commits a delete. `get_doc` of a missing id is `Ok(None)`.

The field index covers top-level object fields only. Strings use their value. Numbers use their JSON text (`1`, not `1.0`, for integers; other numbers use `serde_json::Number`'s JSON rendering). Bools use `true` and `false`. Null, arrays, and objects are not indexed. `query_by_field` returns matching ids in ascending order. A delete removes that document's index hits. The index is rebuilt from documents after replay.

BM25 is Okapi with `k1 = 1.2` and `b = 0.75`. Tokens are Unicode alphanumeric runs, lowercased. Every string in the document is indexed, recursively through object values and array elements. Object keys are not indexed. For `N` documents in the collection, `df` documents containing the term, term frequency `tf`, document length `dl`, and average document length `avgdl`:

```text
IDF = ln(1 + (N - df + 0.5) / (df + 0.5))
score = sum over query tokens of
    IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
```

`ln` is the natural logarithm. A document scores only when at least one query token occurs in it. Results are ordered by score descending, then id ascending, and truncated to `limit`. An empty query, a query with no tokens, or `limit == 0` returns an empty vector. Statistics are rebuilt from documents after replay.

## FerroMQ

```text
publish_mq(topic, key, payload) -> Result<u64, StoreError>
poll_mq_group(topic, group, limit) -> Result<Vec<MqRecord>, StoreError>
ack_mq(topic, group, seq) -> Result<(), StoreError>
subscribe_mq(topic) -> Receiver<MqRecord>
```

`MqRecord` is `{ seq, key, payload }`.

Durable messages are documents in collection `mq:{topic}`. The id is the decimal sequence with no padding. The JSON object is `{"seq": N, "key": "...", "payload": ...}`. The sequence starts at 1 for each topic and increases by 1 for each publish on that topic. It is not reused after the message document is deleted. Each topic has a high-water document in collection `mq_hw:{topic}`, id `high_water`, JSON `{"seq": N}`. A publish is one transaction that puts the message and sets that high-water to the new sequence. The assigned sequence is one past the high-water. If the high-water document is absent, the next sequence is one past the highest message `seq` still stored for the topic, and the high-water is set to that new sequence. Deleting `mq:{topic}/{id}` does not delete or lower the high-water. Acks do not advance the topic sequence.

Acks are documents in collection `mq_ack:{topic}:{group}`, id the decimal sequence, JSON `{"seq": N}`. `ack_mq` is durable. Acking an unknown sequence still records the ack and is idempotent.

`poll_mq_group` returns the oldest unacked messages for that topic and group, up to `limit`, and does not ack them. A second poll without an ack returns the same messages. `limit == 0` returns an empty vector. Another group has its own ack set.

Subscribers are in-process only: a `Mutex<Vec<Sender<MqRecord>>>` map. `publish_mq` sends a clone after the durable commit and after the file lock is released. A disconnected subscriber is dropped. `subscribe_mq` does not replay history; it receives only later publishes on this handle.

## Phases

Phase 1 (Record 14) is this engine: the `.fls` container, `mem://`, `file://`, the document store, and FerroMQ.

Phase 2 (Record 13) is served mode: Unix-socket and WebSocket RPC, and `FerroStore::auto`. `unix://` and `ws://` are served. `wss://` is recognized and refused.

Phase 3 is the columnar model. One non-empty `append_column_rows` is one row group. An empty append writes nothing and is success. The codec is chosen when the group is written and stored in the row-group blob; a read does not pick a different one.

- `Int64` and `TimestampMs` store the first non-null value absolutely and each later non-null value as a zigzag delta of the previous non-null. If any run of equal deltas has length at least 3, the chunk uses `delta_rle`. Otherwise it uses `delta`. Identical values are a run of zero deltas, so a long enough run uses `delta_rle`.
- `Utf8` uses a dictionary when there are at least two non-null values and the distinct count is at most half the non-null count. If every non-null value is the same and there are at least three, it uses RLE instead. Otherwise it stores plain length-prefixed strings.
- `Bytes` uses RLE when every non-null blob is equal and there are at least three. Otherwise plain.
- `Float64` uses RLE when every non-null value is bit-identical and there are at least three. Otherwise plain little-endian `f64`.
- `Bool` uses RLE of runs.
- Nulls are a bitset (1 = null), not encoded as values. Zone-map min and max ignore nulls. `null_count` is the number of nulls in that column chunk. An all-null chunk has `min` and `max` of `None`. `NaN` is not ordered: float min and max ignore `NaN`, a comparison never matches `NaN`, and a group whose only non-null values are `NaN` is excluded by every comparison.

A scan skips a row group, before decoding its column values, when any filter's zone map excludes the predicate. `row_groups_skipped` counts only that exclusion, not rows dropped after a decode. Filters are AND. `ZoneMap::excludes` is true when `min` and `max` are `None`. `Eq(v)` excludes when `v < min` or `v > max`. `Lt(v)` excludes when `min >= v`. `Le(v)` excludes when `min > v`. `Gt(v)` excludes when `max <= v`. `Ge(v)` excludes when `max < v`. `Between { low, high }` is inclusive and excludes when `low > high`, or when `max < low` or `min > high`. A predicate that does not order against the bounds, including `NaN`, excludes the group. A stored bound that is `NaN`, or a min that is not `<=` max, is corrupt. `Int64` and `TimestampMs` compare as integers and do not compare with each other. `Float64` compares as ordered `f64` (no `NaN` in min or max). `Bool` compares as false < true. `Utf8` and `Bytes` compare bytewise. A group that is not skipped keeps a row only when every filter matches. Null fails every comparison.

Table and column names are non-empty, at most 128 bytes, and contain no `/` or NUL. Duplicate column names, a second `create_column_table` of the same name, an append to a missing table, a bad row width, and a cell whose type does not match the column are `StoreError::Backend` and write nothing. Null is allowed in every column. A predicate whose value type does not match the column, or a filter that names a missing column, is `StoreError::Backend`.

Version 1 is only a store with no column tables and no vector collections. Version 2 is column tables and no vector collections. Version 3 is any vector collection. A version 1 snapshot still opens. A version 2 snapshot with no tables is corrupt. Each version-2 table and row group is checked on open, and a row group whose row count is zero is corrupt. A create or append in a WAL frame is checked the same way before the frame is published. `attempt_rollup` and `gate_run_rollup` read the existing `attempts` and `gate_runs` documents, project them into an in-memory columnar table (they do not write a second durable copy, and they do not change `append_attempt` or `append_gate_run`), and aggregate with a columnar scan. `Pass` counts as pass, `Fail` as fail, and `Error` as error. `project` or `gate` of `None` sums every row; `Some` keeps that id only.

Phase 4 is the vector model. A vector collection is a separate map from documents: the same name may be a document collection and a vector collection. Names use the column-table rules (`vector collection` is the kind in the error). `dim` is a non-zero `u32`, and `dim * bytes_per_value` must be at most 8MiB and must not overflow. Storage is `f32` (4 little-endian bytes), `f16` (2 little-endian IEEE binary16 bytes, round ties to even; a finite value that overflows the finite f16 range is rejected; subnormals and round-to-zero are kept), or `i8` (one two's-complement byte; round half away from zero, then saturate to `-128..=127`). The metric is cosine, L2, or dot product. `M` is `2..=128`. `ef_construction` is `>= M` and `<= 100000`. `ef_search` is `1..=100000`. A duplicate create is `StoreError::Backend` and writes nothing.

Distances are computed on dequantized `f32` values and accumulated in `f64`. The query is quantized and dequantized with the collection storage before the distance, so search runs in storage space. A non-finite component, or a length other than `dim`, is `StoreError::Backend` and writes nothing. L2 similarity is `1 / (1 + euclidean)` and the graph distance is the euclidean length. Cosine similarity is `dot / (|a| |b|)`, or `0` when either norm is `0`, and the graph distance is `1 - similarity`. Dot-product similarity is the dot product and the graph distance is `-dot`. Scores are higher-better. Results are ordered by score descending, then id ascending.

An id is non-empty, has no NUL, and is at most 1024 bytes. `/` is allowed. `upsert_vector` replaces the bytes and the metadata. `None` metadata is absent. `Some(null)` stores JSON null. `get_vector` of a missing id is `Ok(None)`. `delete_vector` of a missing id succeeds and still commits a delete. A later upsert of that id works, and search does not return a deleted id. A missing collection is `StoreError::Backend` and writes nothing. `apply_op` checks the collection, the byte length, and that every stored `f32` or `f16` decodes finite. A bad check is an error, and `stage_ops` publishes nothing.

HNSW parameters live on the collection. There is no hidden default `M`. The level of an id is deterministic: hash the id with SHA-256, take the first 8 bytes as a little-endian `u64`, replace `0` with `1`, set `u = x / u64::MAX` in `(0, 1]`, and set `level = floor(-ln(u) / ln(M))`, clamped to `0..=16`. Layer 0 allows `2 * M` neighbors. Every upper layer allows `M`. Neighbors are ordered by smaller graph distance, then lower id. The first vector is the entry point and has no links. Later inserts search from the entry point: greedy with ef 1 above the new level, then ef `ef_construction` from `min(new level, entry level)` down to layer 0. Each layer keeps the closest `max_links` neighbors, links both ways, and prunes the other node back to its max links. The entry point is the live node with the highest level, then the lowest id. Upsert unlinks the id and inserts it again. Delete unlinks it and removes it. A neighbor id that is not live is not left behind. Links are bidirectional.

`search_vectors` is greedy from the entry point down to layer 1 with ef 1, then layer 0 with width `max(ef, limit)`. It returns at most `limit` live nodes. `limit == 0` or `ef == Some(0)` returns an empty vector and is not an error. `ef == None` uses the collection `ef_search`.

`search_hybrid` takes at most `limit` hits from `search_bm25` and at most `limit` hits from `search_vectors`, fuses that pool, and truncates to `limit`. An id on only one side still counts. `limit == 0` returns empty. An empty or token-free query makes the BM25 side empty and does not error; the vector side still uses the query vector. Reciprocal rank fusion (`k >= 1`) sums `1 / (k + rank)` with ranks starting at 1 in each side's existing order. Weighted fusion requires finite weights `>= 0` with at least one `> 0`. Each side is min-max normalized. An empty side adds nothing. When `max == min`, every hit on that side has normalized score 1. A missing side contributes 0. The fused score is `bm25_weight * norm_bm25 + vector_weight * norm_vector`. Order is fused score descending, then id ascending.

`search_knowledge` uses that fusion and those two collections. Each hit is the id, the fused score, the vector metadata when the id is in the vector collection (`None` when it is absent or when the id is BM25-only), and the document when the id is in the document collection (`None` otherwise). The caller stores a knowledge entry with `put_doc` and its embedding with `upsert_vector` under the same id. FerroStore does not call an embedding model, and this phase does not change the redb state format.

A checkpoint with any vector collection writes snapshot version 3. A document-only store stays version 1 and byte-identical. Columnar tables with no vector collections stay version 2. `migrate_redb_to_ferrostore` refuses a non-empty document map, column-table map, or vector map.

## Sync adapter

A process may attach one `SyncAdapter` in a `SyncMode`: `WriteThrough`, `WriteBehind`, `LocalCache`, or `CloudPreCache`. The mode, the adapter, the pending batches, and `synced_seq` are process memory. They are not written to the `.fls` file. A reopened store has no adapter and `synced_seq` of 0.

After a commit that advances the store sequence, the store builds a `SyncBatch` of that sequence and its ops. An empty op list does not. The adapter is not called while the memory document mutex or the disk `flock` is held. `WriteThrough` calls `on_commit` after the commit is published, and only one of those calls is in flight. A newer commit cannot call `on_commit` while an older notify is in flight or still queued. If the pending buffer is non-empty, or a `WriteThrough` `on_commit` is already in flight, the batch is appended and `on_commit` is not called. Success sets `synced_seq` to that sequence when it is greater. Failure leaves the commit applied, puts that batch at the front of the pending buffer, ahead of any batch appended during the call, does not advance `synced_seq`, and returns the error. `WriteBehind`, `LocalCache`, and `CloudPreCache` append the batch and do not call `on_commit`. A silent commit does not call `on_commit` and is not queued. It may set `synced_seq` to that sequence only when the pending buffer is empty and the sequence is greater. With no adapter, commits are unchanged.

`sync_flush` waits while a notify is in flight, then sends only the oldest pending batch. It repeats in order until the buffer is empty, in every mode. It does not send a batch that is not the front, and it does not send while another `on_commit` is in flight. The first error leaves that batch and the later ones pending. Success of a batch removes it when it is still the front and sets `synced_seq` to that sequence when it is greater. A failed `WriteThrough` batch stays ahead of later commits until that flush. A loud multi-batch apply calls `sync_flush` only in `WriteThrough`; the other modes keep the batches pending, and `sync_flush` from inside the in-flight `on_commit` returns without sending.

`get_doc` in `LocalCache` or `CloudPreCache`, and only when the pending buffer is empty, treats a missing document as a cache miss and calls `fetch_document`. `Ok(None)` stays a miss. `Ok(Some(doc))` is stored only when that id is still absent and the pending buffer is still empty. The check and the write hold the document lock and then the sync mutex, and no other commit registers until that check finishes, so a document inserted after the miss is not overwritten. If the id is already present, that stored document is returned and nothing is written. If the pending buffer filled during the fetch, the fetched value is dropped: an id that is now present is returned, and an id that is still absent stays missing. A silent put does not call `on_commit` and is not queued. When the pending buffer is empty, it sets `synced_seq` to the new sequence. A stored JSON null is a hit, not a miss. A missing remote document is a JSON null result. Other modes do not fetch. A fetch error is returned. A remote fill rechecks the client pending buffer, then sends `put_doc` with `silent` and `if_absent`. The server writes only when the id is absent. Silent means the server adapter is not notified. The client sets `synced_seq` from that store sequence only when its own pending buffer is still empty. If the server wrote nothing, the client returns the server's current document and does not advance `synced_seq`.

`sync_pull` calls `pull_since` with the store sequence. `sync_prewarm` calls `prewarm`. Both require an adapter and an empty pending buffer. Every batch is dry-run first. The store publishes them only when every batch applies: the first sequence is the current sequence plus one, each later sequence is strictly one greater, and an empty op list is not a commit. On disk, a multi-batch apply publishes one superblock whose `wal_length` covers every frame. Anything that fails before that publish leaves the previous superblock current. A bad op or a bad sequence writes nothing and leaves `synced_seq` unchanged. An empty result writes nothing and returns the current `synced_seq`. After a successful apply, `synced_seq` is the last applied sequence, and `on_commit` is not called. A remote client sends the batches in one `apply_sync_batches` with `silent` set, so the adapter on the store that owns the file is not notified. The mode, the adapter, the pending batches, and `synced_seq` stay process memory. They are not written to the `.fls` file.
