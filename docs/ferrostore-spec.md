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

Creating a missing file makes it mode `0600` from its first byte, and makes its parent directory mode `0700` when that is missing too. An existing file or directory keeps its mode; a store file that other users can read gets a warning (`private::warn_if_loose`), never a `chmod`. The create writes a snapshot at `8192` that holds one document, the schema version (below), at sequence `0`, `fsync`s, writes slot 0 and leaves slot 1 as zeros, `fsync`s, then `fsync`s the parent directory:

| Field | Initial value |
| --- | --- |
| `epoch` | `1` |
| `snapshot_offset` | `8192` |
| `snapshot_length` | the length of that snapshot |
| `wal_offset` | `8192 + snapshot_length` |
| `wal_length` | `0` |
| `snapshot_seq` | `0` |

The create path holds the same short exclusive `flock` as a commit, only while it publishes that first superblock. A file with no valid slot is created again only when it can hold nothing but a create cut short: every header byte is zero, and the bytes after the header are none or a prefix (up to all) of that first snapshot. Any other such file is never rewritten, and `open` fails. An open that creates the first superblock this way also `fsync`s the parent directory.

### Superblock

Each slot is 4096 bytes. Every slot this build writes is container format `2`.

| Offset | Type | Contents |
| --- | --- | --- |
| `0..8` | bytes | Magic `FERRSTR1` |
| `8..12` | `u32` | Container format version: `2` (format `1` is still read) |
| `12..20` | `u64` | `epoch`, monotonic, starting at `1`, and monotonic across compactions (see Compaction) |
| `20..24` | `u32` | CRC-32 of this 4096-byte block with bytes `20..24` set to zero |
| `24..32` | `u64` | `snapshot_offset` (absolute) |
| `32..40` | `u64` | `snapshot_length` |
| `40..48` | `u64` | `wal_offset` (absolute) |
| `48..56` | `u64` | `wal_length` |
| `56..64` | `u64` | `snapshot_seq` |
| `64..68` | `u32` | `engine`: `0` is snapshot plus WAL. Format `2` only |
| `68..76` | `u64` | `incompat` feature bits. None are defined. Format `2` only |
| `76..4096` | bytes | Zeros (`64..4096` in format `1`) |

`snapshot_seq` is the highest committed transaction sequence included in the snapshot, or `0` when no snapshot holds a commit.

A slot is valid only when magic, a known version, padding, and CRC all match, and its engine is `0` with no unknown `incompat` bit. A CRC-valid slot with a newer version, another engine, or an unknown `incompat` bit was written by a newer build: `open` fails with `StoreError::ContainerFormat`, which names the version found, and reads neither slot. Every later format keeps the magic at `0..8`, the version at `8..12` and the CRC at `20..24`, so a newer slot is told from a torn one. Any other bytes are ignored. The current superblock is the valid slot with the highest `epoch`. A torn write fails the CRC, so the other slot stays current. Equal epochs on two valid slots are corruption. A format-1 file is upgraded at its next slot write; both slots stay CRC-valid throughout.

### Schema version

The document `("__meta", "schema_version")` holds the schema version, on the same ladder as `RedbStore` (`FORMAT_VERSION` to `FORMAT_WITH_WORKFLOW` in `src/lib.rs`), so `format_version()` means the same on both. A new store starts at the newest constant. A write that needs a newer schema raises it in the same transaction, and a raise never lowers it. `open` refuses a version off the ladder with `StoreError::FormatVersion`. A store from before `__meta` reads its version from `("meta", "format_version")`, or has none: it opens, and `format_version()` is `None`. No trait method lists `__meta`. A collection whose name starts with `__` is reserved: only the store's own transactions write it. `put_doc` and `delete_doc`, local or over the socket, refuse it, and a cache miss on it is never fetched. Sync carries its ops like any other: an outbound batch holds every op of its commit, so sync sequences have no gaps. An inbound batch never applies a reserved op as it is. A put of `("__meta", "schema_version")` to a version on the ladder above the store's own raises it. A put of a version newer than this build refuses the whole batch with `StoreError::FormatVersion`, as `open` does. Every other reserved op (an older, equal or off-ladder version, another `__meta` id, a delete, another `__` collection) is dropped. The batch's sequence advances even when every op in it is dropped.

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

Version `3` is written when any vector collection exists, including an empty one. After the documents it stores the version `2` table section. `table_count` may be `0`. It then stores `u32` collection count, which must be at least `1` (0 is corrupt), and collections in name order. Each collection is a length-prefixed name, `u32` dim, `u8` storage tag (`1` f32, `2` f16, `3` i8), `u8` metric tag (`1` cosine, `2` l2, `3` dot), `u32` M, `u32` ef_construction, `u32` ef_search, `u32` vector count, then vectors in id order. Each vector is a length-prefixed id, a length-prefixed byte string, a `u8` metadata flag (`0` absent, `1` present) and, when the flag is `1`, a length-prefixed UTF-8 JSON value. Then an entry flag (`0` or `1`) and, when it is `1`, a length-prefixed entry id. Then, for each vector in id order, a `u8` level and, for each layer `0..=level`, a `u32` neighbor count and that many length-prefixed neighbor ids. Each layer's neighbor list is written ordered by graph distance, then id. The entry id must be the canonical live node (highest level, then lowest id) when the collection is non-empty. The entry flag is `0` when it is empty. Decode, and so `open`, checks the snapshot CRC, then the structure of each collection (`validate_graph`): names, ranges, known tags, each vector's id, its byte length and finite components, a level that matches the deterministic HNSW formula, one neighbor list per layer `0..=level`, and on each layer at most the layer's max links, no self link, no repeated neighbor, every neighbor live and present on that layer, and every link bidirectional; and the canonical live entry. Decode does not check the neighbor order. The full check (`validate_graph_full`) adds it: every list in (graph distance, id) order, one distance per link. The encoder runs the full check before it writes any version-`3` snapshot, and tests use it. Open skips the order check because search results do not depend on it: a search returns live ids with exact scores, and the final result set is the same in any list order. Delete-repair determinism does depend on it: a delete repairs the graph from the deleted node's list in stored order, and each step changes the graph. So the order is enforced at write time (every insert and every list a change touches writes it in canonical order) and checked in full by the encoder before every snapshot write. A misordered snapshot that passes the CRC (only a writer bug can make one) is accepted on open, and the next checkpoint fails with the full check's error, which says the collection's neighbor order is not canonical and a re-sort is needed. The CRC protects the bytes after the write. A bad snapshot is corrupt.

`checkpoint` writes a new blob at an offset at or past the end of the file, and past both the live snapshot and the live WAL, so it does not overlap either. No extent of a file is ever reused: the snapshot and WAL a checkpoint replaces stay in the file as dead bytes until a compaction (see Compaction). It `fsync`s the blob, then publishes a superblock that points `snapshot_offset` / `snapshot_length` at the new blob, sets `snapshot_seq` to the last committed sequence, sets `wal_offset` to the end of the new blob, and sets `wal_length` to `0`. A crash after the blob is written and before that publish leaves readers on the previous superblock. For a store with vector collections, the checkpoint's encoder first runs the full graph check, which dequantizes every vector of a collection into one `f32` buffer of `n * dim * 4` bytes: the peak memory of a checkpoint is that much above the stored state (1x the stored vector bytes for f32, 2x for f16, 4x for i8).

A commit also checkpoints on its own (auto-checkpoint) when the state it published has a WAL longer than `max(snapshot_length, 4 MiB)` (`AUTO_CHECKPOINT_BYTES`). A reopen then replays at most that much WAL. The checkpoint runs outside the commit's lock window: the writer first releases the commit gate, the write lock and `LOCK_EX`, wakes every group member with its own result, and delivers its own `WriteThrough` batch. Only then does it take the write lock and `LOCK_EX` again, refresh, and checkpoint if the refreshed state is still past the threshold (another handle may have checkpointed meanwhile). So no other commit's result waits for the checkpoint; the writer's own call returns after it. This applies after a group commit (the group's leader checkpoints), a sync-batch apply (after its `WriteThrough` flush) and a cache fill; a write that published nothing does not checkpoint, and neither does a state whose WAL ends torn. A failed auto-checkpoint does not fail the commit: the commit returns its own result. A failed `fsync` in the checkpoint poisons the handle, as any failed `fsync` does. Any other failure (a full disk, an encode error) leaves the file and the cache as they were and records the WAL length at the failure: the handle tries again only once the WAL has grown past that length by another `max(snapshot_length, 4 MiB)`, so a failure that persists costs one attempt per threshold of WAL, not one per commit. The first failure of a run is a warning on stderr; later ones are not, until a checkpoint (auto or manual) succeeds. `checkpoint` stays public.

A checkpoint, manual or automatic, that leaves the file due a compaction is followed by one, in a lock window of its own (see Compaction).

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

## Compaction

`FerroStore::compact` writes the current state into a new sibling file and renames it over the store, so the dead bytes earlier checkpoints left are given back. A file's *live bytes* are the 8192-byte header plus the live `snapshot_length` and `wal_length`; its *dead bytes* are its length less its live bytes. `FerroStore::stats` returns both, the file length, the live epoch and `last_seq`, read without a lock as a read is (`mem://`: every byte count and the epoch are `0`; a remote handle does not serve it).

First, before it takes any lock, the compactor removes this store's stale compaction temps: regular files beside it, not symlinks, named `<name>.compact-<pid>-<n>-<nonce>` (decimal pid and counter, eight lowercase hex digits), owned by this user and not modified for 24 hours. The rule and the code are the migration's (`<name>.migrating-…`), with another mark. A sweep that fails is a warning on stderr.

Then it holds the write mutex and the exclusive `flock` of the file the path names (taken as every writer takes it, see Concurrency) for the whole of:

1. Refresh, as a commit does.
2. Create `<name>.compact-<pid>-<n>-<nonce>` with `O_CREAT | O_EXCL`, mode `0600`, and take its own `flock`. `<name>` is the store's real path: `open` resolves the path it is given once (`realpath`), and every operation on a name (the swap check, the sibling, the rename, the directory sync, the sweep) uses the resolved path, so a store opened through a symlink compacts beside its file and the link goes on naming it. The handle's label stays the path as given.
3. Write slot 0 and one snapshot at `8192`, leaving slot 1 zeros and the WAL empty. The slot's `epoch` is the old file's current epoch `+ 1`; `snapshot_seq` is the state's `last_seq`, so `last_seq` is kept. The snapshot is the one `checkpoint` would write.
4. `fsync` the new file (`sync_all`: it was just created).
5. `rename` it over `<name>`.
6. `fsync` the directory.
7. Install the state in this handle's cache and move this handle's descriptions to the new file.
8. Release the new file's lock, then the old file's lock.

A failure before the rename removes the new file and leaves the old one as it was. A failed `fsync` (of the new file or of the directory) poisons the handle, as any failed `fsync` does (see Concurrency). A failed directory sync comes after the rename: the path already names the new file, and its name may not be durable. The new file's lock is held until the directory sync is done, so no writer commits into it before the compactor has tried to make its name durable.

**A handle never commits into a file whose name it has not itself made durable.** Each write description records whether this handle made the directory entry that names its file durable: true when this handle created the file at `open` (and synced the directory) or compacted into it (and its directory sync returned); false when `open` found the file already there, and after a swap check reopened the path. Before the first commit, checkpoint or compaction through a description that is false, once it holds the `flock` and has checked that the path names its file, the writer `fsync`s the directory and sets it true. A failure refuses the write before any byte of it is written, and poisons the handle like any failed `fsync`: after a failed `fsync` a later one can report success without the entry on disk, so no retry is trusted. So after a compactor's failed directory sync and a crash, the path can name the old file again only if no handle acknowledged a commit into the new one. Reads never sync the directory. `mem://` compaction is a successful no-op. A remote handle calls the `compact` RPC.

A leftover temp is never opened as the store: `open` reads only the store's own name. A cut at any step leaves exactly one file under that name, the old one or the new one, each a whole store with every acknowledged commit; the temp, when it survives, is swept by a later compaction once it is 24 hours old.

**Auto-compaction.** A checkpoint (manual or automatic) whose published file has more dead bytes than `max(live bytes, AUTO_COMPACT_BYTES)` (more than half the file, and more than 64 MiB) is followed by a compaction. It runs in a lock window of its own, once the checkpoint has released its locks (and, after a commit, once every member of the group has its result), refreshes, and compacts only when the refreshed file is still due. A failed auto-compaction does not fail the checkpoint or the commit before it. A failed `fsync` poisons the handle; any other failure leaves the old file as the store and records the dead bytes at the failure: the handle tries again only once the dead bytes have grown past that by another `max(live bytes, AUTO_COMPACT_BYTES)`. The first failure of a run is a warning on stderr; later ones are not, until a compaction succeeds.

**Swap detection.** Every path that takes the `flock` compares, once it holds the lock, `stat(path)` `(dev, ino)` with `fstat` of its description. When they differ, a compaction has renamed a new file over the path: the writer releases the lock, opens the path again for both its write and its read description, and tries again. A read refresh compares the same two before it reads the slots, and reopens its read description on a difference. A path that names nothing is no swap (a compaction's rename replaces the name in one step): a read goes on through its description, and a writer refuses, since a commit into an unlinked file would be lost. A description that is swapped out is closed once its last user drops it, so a read already reading the old file finishes on it; the old inode lives until its last descriptor closes.

**Epochs across files.** The old file is never published after the rename: every writer checks the path under the old file's lock, which the compactor holds through the rename. So the epoch orders every published state along the path's history, across compactions. A handle's cache orders two states of one file by epoch. Between two files, the file the path names (as the handle's last swap check found it) wins whatever the epochs: a file put in place at a lower epoch by other means (a restore by rename, a file replaced by hand) is read at the next refresh, and a slow read of a file the path no longer names never replaces a state of the current one. Two files neither of which the path names are ordered by epoch. A handle's reads never go back along the path's history, across a compaction too.

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

`checkpoint` holds the same exclusive lock while it reads, writes the new snapshot, and publishes. An auto-checkpoint takes that lock again once the commit that crossed the threshold has released it (see Snapshot).

Step 2 takes `LOCK_EX` on the file the path names: once the lock is held, `stat(path)` is compared with the description's `(dev, ino)`, and a difference (a compaction renamed a new file over the path) releases the lock, reopens the path for both descriptions and starts again (see Compaction). A compaction holds the lock of the file it replaces until the new file is renamed over it and the directory is synced.

Readers open a separate `O_RDONLY` description and do not `flock`. A read first compares `stat(path)` with that description and reopens it when a compaction swapped the file. It then loads both superblocks, selects the highest valid epoch, loads that snapshot, and applies WAL frames with `seq > snapshot_seq`. A torn superblock fails its CRC and is ignored. Two `FerroStore::open` handles on one path both succeed; the second open is not `StoreError::Unreachable`. A reader observes a writer's committed document without taking a file lock.

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
| `ws://127.0.0.1:9`, `ws://127.0.0.1:9/store` | Client of that WebSocket endpoint. Served. The query may name the credential (see Authentication) |
| `wss://host:9`, `wss://host:9/store` | Client of that WebSocket endpoint over TLS (see TLS). Served. The query takes the same parameters as `ws://`. An error about the URI itself names `wss` |
| Anything else, including `""` and `http://` | `StoreError::Backend` whose message contains `unknown` and does not describe the scheme as deferred |

`file://` and `unix://` paths are not percent-decoded. Only the host `localhost` is stripped; any other non-empty host is part of a relative path. A failed `unix://`, `ws://` or `wss://` connect is `StoreError::Backend`. It is not an unknown scheme.

`FerroStore::connect_with(uri, &ConnectOptions)` is `connect_uri` with a credential (see Authentication). `connect_uri(uri)` is `connect_with(uri, &ConnectOptions::new())`: no credential, which a `unix://` server that checks the peer uid alone accepts.

## Served mode

Phase 2 is this served mode. The process that binds a `FerroStoreServer` is the only opener of the `.fls` file. Clients do not take `flock` and do not open the file.

A Unix frame is a 4-byte big-endian length followed by that many bytes of UTF-8 JSON. The length is `1..=16MiB` (`16 * 1024 * 1024`). A length of zero or a longer payload is an error.

A WebSocket message is one JSON object in one RFC 6455 frame. Client frames are masked. Server frames are not. The payload has no length prefix. The JSON payload is `1..=16MiB`. `wss://` is the same WebSocket protocol inside TLS (see TLS).

Every connection authenticates before the server reads its first request (see Authentication). After that, each call is one JSON object:

```json
{"id": 1, "method": "put_doc", "params": {}}
```

`id` is an integer chosen by the client. Success is `{"id": 1, "ok": true, "result": ...}`. Failure is `{"id": 1, "ok": false, "error": "..."}`, and the client returns `StoreError::Backend` of that message. The one exception is a write that committed on the server but whose server-side `WriteThrough` delivery failed (`StoreError::Undelivered`). Its failure also carries `"kind": "undelivered"`, `"seq": <u64>` (the commit's sequence) and `"cause": "..."` (the cause's full message, as its `Display` reads). A `Backend` cause also carries `"cause_backend": "..."`, its detail alone. The client returns `StoreError::Undelivered { seq, cause }`: `cause` is `Backend(cause_backend)` when `cause_backend` is present, and otherwise `StoreError::Relayed(cause)`, which displays the server's words unchanged. So the client's message is the server's, word for word, whatever kind the cause was. The write is durable on the server, so the client does not retry it.

| Method | Params | Result |
| --- | --- | --- |
| `open_session` | none | `{"session": <u64>, "base_seq": <u64>}`: a read session pinning the server's current state (see Remote transactions) |
| `close_session` | `session` | `null`. An unknown session is no error |
| `txn_read` | `read`, optional `session` | `{"result", "hash"}`: the answer to one read primitive at the session's snapshot and the SHA-256 of its canonical JSON, as 64 hex digits. Without `session` the read opens one, and the reply also carries `session` and `base_seq` |
| `commit_txn` | `reads` (`[{"read", "hash"}]`), `ops`, optional `session`, `base_seq`, `silent`, `req` | `{"committed": true, "seq": <u64>}`, or `{"committed": false, "seq": null}` when a read lost: nothing was written. A bad op is an error |
| `put_doc` | `collection`, `id`, `doc` | `null` |
| `get_doc` | `collection`, `id` | `null` when the id is missing. A present value, including JSON `null`, is `{"doc": <value>}` |
| `delete_doc` | `collection`, `id` | `null` |
| `query_by_field` | `collection`, `field`, `value` | array of id strings |
| `search_bm25` | `collection`, `query`, `limit` | array of `[id, score]` |
| `publish_mq` | `topic`, `key`, `payload` | the sequence number |
| `poll_mq_group` | `topic`, `group`, `limit` | array of `{"seq", "key", "payload"}` |
| `ack_mq` | `topic`, `group`, `seq` | `null` |
| `checkpoint` | none | `null` |
| `compact` | none | `null`. The server compacts its store (see Compaction) |
| `subscribe_mq` | `topic` | `null`, then push frames |
| `create_vector_collection` | `name`, `dim`, `storage`, `metric`, `m`, `ef_construction`, `ef_search` | `null` |
| `upsert_vector` | `collection`, `id`, `values` (finite JSON numbers), `metadata` optional | `null`. Absent `metadata` stores none. Present `metadata`, including `null`, is stored |
| `delete_vector` | `collection`, `id` | `null` |
| `has_vector_collection` | `name` | `true` or `false`. Creates nothing |
| `get_vector` | `collection`, `id` | `null` when the id is missing. A present value is `{"values": [<numbers>]}` plus `metadata` only when it was stored |
| `search_vectors` | `collection`, `values`, `limit`, optional `ef` | `[{"id", "score"}]`. `score` is a finite JSON number |
| `search_hybrid` | `doc_collection`, `vector_collection`, `query`, `values`, `limit`, `fusion` | `[{"id", "score"}]` |
| `search_knowledge` | same as `search_hybrid` | `[{"id", "score", "metadata", "doc"}]`. `metadata` is `null` when absent. `doc` is `null` when the document is missing |
| `create_column_table` | `name`, `columns` (`[{name, ty}]`; `ty` is `int64`, `float64`, `bool`, `utf8`, `bytes`, or `timestamp_ms`) | `null` |
| `append_column_rows` | `name`, `rows` (array of arrays of cells) | `null` |
| `scan_column_table` | `name`, `filters` (`[{column, op, value}]`; `op` is `eq`, `lt`, `le`, `gt`, `ge`, or `between`. `between` uses `low` and `high` instead of `value`) | `{"rows", "row_groups_total", "row_groups_skipped"}` |
| `attempt_rollup` | `project` (IRI string, or JSON `null` for every attempt) | `count`, `cost_usd_micros`, `tokens_in`, `tokens_out`, `duration_ms` |
| `gate_run_rollup` | `gate` (IRI string, or JSON `null` for every run) | `pass`, `fail`, `error`, `duration_ms`, `cost_usd_micros` |
| `apply_sync_batches` | `batches`, optional `silent` | the sequence of the last batch applied |

`subscribe_mq` does not replay history. A later publish on that topic is pushed on the same connection as a frame with no `id`:

```json
{"method": "mq", "params": {"topic": "orders", "record": {"seq": 1, "key": "k", "payload": {}}}}
```

Every write method (`commit_txn`, `put_doc`, `delete_doc`, `publish_mq`, `ack_mq`, `create_column_table`, `append_column_rows`, `create_vector_collection`, `upsert_vector`, `delete_vector`, `apply_sync_batches`) takes an optional `req`: 32 hex digits, a client-chosen id for one logical attempt of the write. The server keeps the reply to each `req` it ran, at most 10 000 of them for at most 10 minutes. A `req` belongs to the principal that sent it (the credential a connection authenticated with, see Authentication): another principal's equal `req` is a different id, and runs as a fresh request. A write whose `req` its principal already sent is answered with that first reply, error or not, from any connection of that principal, and runs nothing; one whose `req` is still running waits for it. The same `req` for another method is refused with `request id already used for another method` and runs nothing. A running `req` is never forgotten, nor one answered less than 20 s (two call timeouts) after it arrived; when the table holds 10 000 of those, a write with a new `req` is refused with `dedup table full, retry` and runs nothing. One TCP principal (`token:*`, `psk`) holds at most 2 500 of them (a quarter; a Unix principal has no such quota, only the table's 10 000): at that quota its own oldest `req` answered more than 20 s ago is forgotten, and when none is, a write with a new `req` is refused with `dedup quota for this credential reached, retry` and runs nothing. So a resend after a lost reply cannot apply a write twice. A write without `req` is not remembered, and every other method ignores `req`. The client sends `req` with every write it makes (`commit_txn`, `publish_mq`, `apply_sync_batches`, and a cache fill's `put_doc`), a new one for every attempt (a lost `commit_txn` retries under a new one), and the same one again only when it resends an attempt whose reply did not come: it does so once, after the 10 s call timeout. A `dedup table full, retry` or `dedup quota for this credential reached, retry` refusal ran nothing, so the client takes it as a lost attempt: it sends the same `req` again after the wait of a lost commit (see Remote transactions), within 8 sends in all, and then returns the refusal.

### Authentication

A connection authenticates before the server reads anything from it as a request. Until then the server answers no method, `open_session` and `txn_read` included. The first frames of a connection are the handshake below, in the transport's framing (a length-prefixed JSON frame on a Unix socket, one WebSocket message on `ws://` and `wss://`).

**Unix socket.** The server reads the peer's uid with `SO_PEERCRED` (`getpeereid` on the BSDs and macOS) when it accepts. A uid other than the server's effective uid, or one the kernel does not report, is refused before anything is read, and the server prints `warning: FerroStore refused a Unix connection from uid <uid>: the server runs as uid <own>` to stderr. The socket is mode `0600` as well (see Auto). `bind_unix(path, store)` checks the uid alone. `bind_unix_with(path, store, auth)` given a `ServerAuth` with a PSK also runs the PSK challenge-response after the uid check; a configured PSK is always required, never optional. A token file is refused on a Unix bind. A Unix connection that passes holds the `admin` scope.

**TCP (`ws://`, `wss://`).** `bind_tcp(addr, store, auth)` needs a `ServerAuth` with a PSK, a token file, or both; without one it fails before it binds. Which addresses it may bind, and when it serves `wss://`, is in TLS. A connection is decided by its upgrade request:

* With one `Authorization: Bearer <token>` header (scheme case-insensitive; the token is visible ASCII), the server hashes the token with SHA-256 and compares the hash with every token-file entry in constant time. A match holds the entry's scope. Anything else in `Authorization`, two such headers, or no match is refused.
* Without `Authorization`, the server runs the PSK challenge-response. A connection that passes holds the `admin` scope. A server without a PSK still sends the challenge and refuses every answer.

**PSK challenge-response.** Each side proves the PSK to the other. On a Unix socket and over plain `ws://` the handshake is version 1:

1. Server: `{"auth": "challenge", "v": 1, "ns": <hex>}`. `ns` is 32 bytes from the system random source, fresh for every connection.
2. Client: `{"auth": "psk", "nc": <hex>, "proof": <hex>}`, `nc` 32 fresh random bytes, `proof = HMAC-SHA256(psk, "fl-store/1/client" ‖ L ‖ server_id ‖ ns ‖ nc)`.
3. The server checks the proof with `ring::hmac::verify` (constant time). It refuses a hello with any other key, a value that is not exactly 64 lowercase hex digits, or `nc` equal to `ns`.
4. Server: `{"auth": "ok", "scope": "admin", "proof": <hex>}`, `proof = HMAC-SHA256(psk, "fl-store/1/server" ‖ L ‖ server_id ‖ nc ‖ ns)`.
5. The client checks that proof the same way. A missing or wrong proof, or an `ok` without a challenge, is refused by the client with an error that says the server did not prove the PSK for that `server_id`, and no request is sent.

Over `wss://` the handshake is version 2, bound to the TLS session. The frames are the same, the challenge says `"v": 2`, and each proof covers the connection's TLS exporter `E`: 32 bytes of `export_keying_material(label = "EXPORTER-fl-store-psk", context = none)` (RFC 8446 §7.5; RFC 5705 for TLS 1.2), which both ends of one TLS session compute alike and no other session shares:

* client `proof = HMAC-SHA256(psk, "fl-store/2/client" ‖ L ‖ server_id ‖ ns ‖ nc ‖ E)`;
* server `proof = HMAC-SHA256(psk, "fl-store/2/server" ‖ L ‖ server_id ‖ nc ‖ ns ‖ E)`.

Each side takes the version from its own transport, never from the peer: a server over TLS sends version 2 and checks only version-2 proofs, and a client over TLS refuses a version-1 challenge (`the FerroStore server speaks another authentication version`) without answering it. A TLS connection whose exporter cannot be read is refused, never served unbound. So a handshake relayed into another TLS session (an attacker that terminates the client's TLS and opens its own to the server) does not verify: the two sessions have different exporters. For the same reason a PSK does not pass through a proxy that terminates TLS and speaks plain `ws://` to the server; a bearer token does.

Every `<hex>` is 64 lowercase hex digits (32 bytes). All four labels are 17 ASCII bytes and differ, so a proof made in one direction, or in one version, is never valid in the other. `L` is one byte, the length of `server_id`; the nonces are 32 bytes each, so the input has one reading. `server_id` is the name the server proves itself under: an operator-chosen name, 1 to 255 bytes of UTF-8 without control characters, configured with the PSK on the server (`ServerAuth::with_psk_file(server_id, path)`) and on every client as the server it expects (`ConnectOptions::psk_file(server_id, path)`). It is not an address. A client whose `server_id` differs from the server's fails the server's check, so a proof one server sees cannot be used at another that shares the PSK. A PSK is at least 32 bytes (see Secret files for its encoding in a file).

**Bearer and Unix accept.** With a valid token, or a Unix peer whose uid passed and no PSK, the server's first frame is `{"auth": "ok", "scope": <scope>}`. A client configured with a PSK refuses that frame, since it proves nothing.

**Principal.** A connection that passes is also given a principal, the key of its quotas (sessions, exactly-once ids) and of its request ids: `token:<sha256 hex>` for a bearer token, `psk` for the PSK over TCP, and `unix:<uid>` for a Unix peer (with or without a PSK). Every Unix connection of a server is therefore one principal, and it is the owner's own uid (`SO_PEERCRED`), so the per-principal quotas (sessions, exactly-once ids) bind only the TCP principals (`token:*`, `psk`); a `unix:<uid>` principal is bounded by the per-connection and server-wide limits alone. The key space of request ids is per principal for every kind.

**Unauthenticated connections are bounded.** A connection has 10 s from its accept to its `ok`, in total, however its bytes arrive: the accept loop, which wakes at least every 20 ms, shuts down one still authenticating past that (each read also has its own 10 s timeout). At most 128 TCP connections authenticate at once; a TCP connection accepted past that is closed at once, before anything is read from it or a thread is started for it. Connections that have authenticated are not counted. The cap does not apply to Unix connections (the socket is mode `0600` in a `0700` directory and checked with `SO_PEERCRED`, so only its owner can connect, and a burst of local connects is never refused): a Unix connection is not counted against it, and has the 10 s total deadline like any other. Before authentication a frame (the hello, and on the client the server's handshake frames) is at most 4 KiB: a larger declared length is refused from the frame header, before anything is allocated; after it, the 16 MiB limit applies. The upgrade's header block is at most 16 KiB and is covered by the 10 s.

**Failure.** Whatever check failed (a wrong PSK or `server_id`, an unknown token, a malformed upgrade or hello, a request sent instead of a hello, another uid, a frame over 4 KiB, a 10 s silence or the 10 s deadline), the server waits 200 ms, sends `{"auth": "failed", "error": "authentication failed"}` (not after a malformed upgrade, which has no frames yet) and closes. The client's error is `FerroStore authentication failed` in every case. Beyond the 128 concurrent TCP handshakes, the server does not limit the rate of attempts; a rate limit is a known gap.

**Secret files.** A PSK file (or a PSK environment variable) holds the key as 64 or more hex digits (32 or more random bytes; either case) and nothing else; one trailing `\n` or `\r\n` is dropped, on the server and the client alike, and the digits are decoded to the key's bytes for the HMAC. Anything else (raw bytes, fewer digits, an odd count, any other character) is refused with the command that makes one: `head -c 32 /dev/urandom | xxd -p -c 64 > psk && chmod 600 psk`. `ServerAuth::with_psk(server_id, bytes)` and `ConnectOptions::psk(server_id, bytes)` take the key's own bytes (32 or more), not its hex. A client token file holds the token on its first line. The server's token file has one token per line:

```text
# sha256-of-token scope label...
3f0a…(64 lowercase hex digits) read ci-dashboard
9b1c…(64 lowercase hex digits) write ingest bot
```

The first field is the SHA-256 of the token's bytes in lowercase hex, the second is `read`, `write` or `admin`, and the rest of the line is a label for people. Blank lines and lines that start with `#` are skipped. A bad field, a repeated hash or a file with no token is refused at load with the line number, and the error repeats nothing of the line. The server holds the hashes, never a token. Every secret file (a PSK file on either side, a token file on either side) must be a regular file without group or other permission bits (`0600` or stricter), owned by the process's effective uid or by root (a root-owned secret mount works); one with those bits is refused with the `chmod 600` that fixes it, and one owned by any other uid is refused with an error that names the path and the owner uid (never the contents). A secret file is opened without blocking (`O_NONBLOCK`), and a symlink is followed: the checks (regular file, no group or other bits, owner) run on the opened descriptor, so they apply to the symlink's target, and a symlinked secret (a Kubernetes secret volume) works. A FIFO or a device, also one a symlink points to, is refused without waiting for a writer. The token file is read once, at `ServerAuth::with_token_file`. To make an entry: `token=$(head -c 32 /dev/urandom | xxd -p -c 64)`, then `printf %s "$token" | sha256sum`.

**Client credentials.** `ConnectOptions::psk_file(server_id, path)`, `psk_env(server_id, var)` or `psk(server_id, bytes)`; `ConnectOptions::token_file(path)`, `token_env(var)` or `token(text)`. A bearer token is for `ws://` and `wss://` alone. A `ws://` or `wss://` URI given to `connect_uri` may name the credential in its query instead, but only by an environment variable whose name is `FERROSTORE_` and one or more letters, digits or `_`: `psk_env=<var>` with `server_id=<name>`, or `token_env=<var>` (`token_env=FERROSTORE_TOKEN`). A URI may come from a less trusted place than the code that connects, so it names no file (`psk_file=` and `token_file=` are refused; a file is named in `ConnectOptions`) and no other variable: it cannot make the client send an arbitrary secret to the URI's host. Values are not percent-decoded. A query that names a secret itself (`psk=`, `token=`, `secret=`, `password=`, `key=`) is refused, as is any other parameter, an empty or repeated one, two credentials, or a URI credential passed to `connect_with` with options that name one too.

**Plaintext.** Over plain `ws://`, a client sends no bearer token and runs no PSK handshake to an address that is not loopback (`127.0.0.0/8`, `::1`, or an IPv4-mapped loopback; the address checked is the one the TCP connect reached, before the upgrade is written), unless its `ConnectOptions` set `allow_plaintext(true)`; a URI cannot set it. The error says to use `wss://` or that option. A connection without a credential is not refused by the client. Over `wss://` a credential goes to any address, inside TLS.

**Bearer tokens and server identity.** A bearer token proves the client and nothing about the server: a client knows its server only through TLS. Use a token over `wss://`; over plain `ws://` only to loopback or on an allowed plaintext interface of a network you trust, where anyone on the path can read and replay it.

### TLS

`ServerAuth::with_tls_files(cert_chain, key)` makes `bind_tcp` serve `wss://`. `cert_chain` is a PEM file of one or more `CERTIFICATE` blocks, leaf first; `key` is a PEM private key (PKCS#8, PKCS#1 or SEC1) and is a secret file under the rules of Secret files (`0600` or stricter, owned by this uid or root, no FIFO); a key file others can read is refused with the `chmod 600` that fixes it, and no error repeats its bytes. A key that is not the certificate's is refused at load. The server speaks TLS 1.3 and TLS 1.2, `ring` provider, no client certificates; TLS 1.2 requires the extended master secret (RFC 7627), on the client too, so its exporter is the connection's own. The TLS handshake runs on the connection's own thread, within the 10 s handshake deadline of Unauthenticated connections (a peer that never sends its ClientHello, trickles it, or stops halfway is shut down at the deadline), and the 4 KiB pre-authentication frame limit applies inside TLS. TLS and `allow_plaintext_on` are refused on a Unix bind.

The key file's PEM text is zeroed as soon as it is decoded, and `rustls` (`ring` provider) zeroes the decoded DER once it loads the key. Not every copy is zeroed: the PEM decoder frees its own base64 buffer without zeroing it, and the loaded signing key stays in memory for the server's lifetime, so anyone who can read the process's memory can read the key. The key is never logged and is in no error or `Debug` output.

A connection that ends sends `close_notify` before its socket is shut down, and a fatal TLS error found while reading (a record that does not decrypt) sends its alert first. Both are best-effort: they are skipped when a writer is blocked on the socket or the handshake holds the connection, and they wait at most 100 ms for the socket, so a shutdown never blocks on them.

A `wss://` client checks the server's certificate in one of three ways, named in `ConnectOptions` and never in a URI:

| Option | The client accepts |
| --- | --- |
| none (the default) | A chain to one of the bundled Mozilla root certificates (`webpki-roots`), compiled into the binary (not the OS trust store), valid now, for the URI's host. A server whose certificate comes from an enterprise or private CA needs `ca_file` |
| `ca_file(path)` | A chain to a CA in that PEM file alone (not the bundled roots), valid now, for the URI's host: an enterprise or private CA |
| `pin_spki_sha256(hex)` | A leaf whose SubjectPublicKeyInfo DER has that SHA-256 (64 hex digits, either case, `:` between bytes allowed; `openssl x509 -in cert.pem -pubkey -noout \| openssl pkey -pubin -outform DER \| sha256sum`). No CA, name or date is checked: the pinned key is the server's identity, so a self-signed certificate, or one renewed over the same key, works. The TLS handshake must still be signed by that key: the certificate the signature is checked against is matched to the pin again first |

The host of the URI is the name verified and the SNI sent (an IP address is matched against the certificate's IP addresses). Naming both `ca_file` and `pin_spki_sha256`, a pin that is not 64 hex digits, or either option with a URI that is not `wss://`, is refused.

**Bind policy.** With TLS, `bind_tcp` may bind any address. Without it (plain `ws://`):

* a loopback address is always allowed;
* `0.0.0.0` and `::` (and `::ffff:0.0.0.0`) are always refused: they listen on every interface, the ones that are not loopback included;
* any other address is allowed only when `ServerAuth::allow_plaintext_on(["tailscale0"])` names an interface and the address is one of that interface's own addresses (`getifaddrs`; an IPv4-mapped address is compared as IPv4, and an IPv6 link-local address must also have that interface's scope id). Naming `lo` covers nothing beyond loopback.

A plaintext bind on an address of a named interface (`lo` named and `127.0.0.1` included) also pins the listening socket to that interface with `SO_BINDTODEVICE`, set before the bind. Linux delivers a packet to a local address whatever interface it arrived on (the weak host model), so without it a listener on a `tailscale0` address also answers a peer on `eth0` that routes that address to the host. The option needs no privilege since Linux 5.7; when setting it fails, the bind is refused with an error that names `SO_BINDTODEVICE` and the interface, and is never made without it. Off Linux there is no `SO_BINDTODEVICE`: the address rule applies but the listener is not pinned, so use TLS there.

Every address the bind's argument resolves to is checked before anything is bound, and the bound address once more. The refusal names the address and says to configure TLS or `allow_plaintext_on`. Authentication is required on every TCP bind, an allowed plaintext interface included.

**Scopes.** `admin` covers `write`, which covers `read`. A method the connection's scope does not cover is an error reply that names the method, the scope it needs and the scope the connection has. It runs nothing (it takes no exactly-once slot); the connection stays open. A method the table does not name is `no such method`.

| Scope | Methods |
| --- | --- |
| `read` | `open_session`, `close_session`, `txn_read`, `get_doc`, `query_by_field`, `search_bm25`, `poll_mq_group`, `subscribe_mq`, `scan_column_table`, `attempt_rollup`, `gate_run_rollup`, `has_vector_collection`, `get_vector`, `search_vectors`, `search_hybrid`, `search_knowledge` |
| `write` | `commit_txn` (even with empty `ops`: it takes the write lock, and its answer says whether the reads still match), `put_doc`, `delete_doc`, `publish_mq`, `ack_mq`, `create_column_table`, `append_column_rows`, `create_vector_collection`, `upsert_vector`, `delete_vector` |
| `admin` | `checkpoint`, `compact`, `apply_sync_batches` |

Creating a column table or a vector collection is `write`: the same op staged in a `commit_txn` is. A `commit_txn` may raise the schema version, as an ordinary write body does on first use of a feature; it lowers or deletes it only in a store that holds nothing outside `__meta` (see Remote transactions), so `write` cannot move the version of a store with data. `apply_sync_batches` is `admin`: it applies another replica's batches at their own sequence numbers, which is replication, not a client write. The one table is `METHODS` in `src/ferrostore/auth.rs`; a test checks it names every method the server dispatches, once.

### Remote transactions

A remote client never receives the whole state. A remote `transaction`, `with_write` or read runs its body on the client over a *read session*: the server pins the `Arc` of its current state under a session id (`open_session`, or the first `txn_read`), with that state's `last_seq` as `base_seq`. Every read of the body is one `txn_read` against that one snapshot, so the body sees one consistent state. A session ends when its commit is sent (lost or not) or the client closes it, and otherwise after 30 s with no read or commit, or 5 minutes after it opened, however often it is read. A connection holds at most 64 sessions, a TCP principal (all the connections of one credential, see Authentication; not a Unix one) at most 256, and a server at most 1024; an `open_session` (or a first `txn_read`) first drops the expired sessions, and then a connection, principal or server still at its limit refuses the new session with `too many open read sessions: this connection holds 64`, `session quota for this credential reached` or `too many open read sessions: this server holds 1024`, keeping the open ones. A `close_session`, a commit that ends a session and every 64th `txn_read` also drop the expired sessions. A connection's sessions end when it closes. A first `txn_read` whose answer fails opens no session. A read in a session that expired is an error that starts `read session expired, retry`; in a write the client takes it as a lost attempt and runs the body again over a new session, within the 8 attempts.

A read primitive is `{"p": <name>, ...}`:

| `p` | Fields | Answer |
| --- | --- | --- |
| `get` | `collection`, `id` | `null`, or `{"doc": <value>}` |
| `scan` | `collection` | `[[id, doc], ...]` in id order |
| `query` | `collection`, `field`, `value` | the matching ids, ascending (the `query_by_field` match) |
| `columns` | `name` | `null`, or the column table's `[{name, ty}]` |
| `vector_params` | `name` | `null`, or `{dim, storage, metric, m, ef_construction, ef_search}` |
| `get_vector` | `collection`, `id` | `{"err": <message>}` for a missing collection or a bad id, `null` for a missing id, or `{"values", "metadata"?}` |
| `inventory` | none | `{"collections", "tables", "vectors"}`: the names that hold anything |

The hash is computed on the server, over the answer's JSON bytes, both when it answers and when it validates. An answer longer than a frame holds (16 MiB less 4 KiB) is an error that names the primitive and the collection (`scan of` the collection, `: the result is <n> bytes, more than one 16 MiB frame holds`). A paginated read is a known limit.

The client caches each answer for the rest of the body, so a repeated read asks once, and records each distinct read with its hash: the read set. A read the body's own writes answer alone is not asked for and not recorded: a `get` of a document the body put or deleted, the columns of a table it created, the parameters of a vector collection it created. A `scan` or a `query` merges the body's writes over the base, so the base's answer is always asked for and recorded whole. A failed `txn_read` fails the whole transaction with its error, whatever the body returns.

Writes are staged in the client's overlay and checked there as a local write checks them, against the session's reads (a table's columns, a vector collection's parameters). A vector write (`create_vector_collection`, `upsert_vector`, `delete_vector`) is staged as an op only: the server applies it, HNSW and all, and the client holds no collection. So `get_vector` of a vector collection the same remote transaction created or changed is refused with `remote transaction cannot read back its own vector writes`.

A body that wrote nothing closes its session and sends no commit. One that wrote sends `commit_txn` with its read set and its ops. The server runs it as one ordinary write body through group commit: inside the write lock it recomputes every read against the committed state and compares the hashes. When all are equal it stages every op exactly as a local write stages it (the same column-table and vector checks; HNSW applied on the server) and commits them as one sequence. When any differs it writes nothing and answers `committed: false`. Every read is validated by value at the commit point, under the lock that commits, and a scan or a query by its whole result, so a remote transaction is serializable and a phantom insert is caught. A reserved (`__`) op is refused, except a put of `("__meta", "schema_version")` to a version on this build's ladder or its delete, which only the store's own bodies make. Checked against the committed state before the commit's ops, a put that raises the version is applied; a put that does not raise it, or a delete, is applied only when that state holds nothing outside `__meta` (no other document collection, no column table, no vector collection: the empty target a migration copies into), and otherwise the commit is refused with `commit_txn refused` and writes nothing. `commit_txn` needs the `write` scope even with empty `ops`: it takes the write lock and its answer says whether the read hashes still match. Each op is `{"op":"put","collection","id","doc"}`, `{"op":"delete","collection","id"}`, `{"op":"create_column_table","name","columns"}`, `{"op":"append_column_group","name","bytes"}`, `{"op":"create_vector_collection","name","dim","storage","metric","m","ef_construction","ef_search"}`, `{"op":"upsert_vector","collection","id","bytes","metadata"?}`, or `{"op":"delete_vector","collection","id"}`. `bytes` is an array of integers `0..=255`. An upsert stores quantized bytes, not logical `f32` values. Absent `metadata` stores none.

A lost commit runs the body again over a new session, at most 8 attempts in all. After lost attempt `n` (from 1) the client waits a time drawn uniformly from `[b, min(2b, 500 ms)]`, `b = min(5 ms · 2^(n-1), 500 ms)`; the jitter is seeded from the thread id and a per-process count. When all 8 attempts lose, the write returns `StoreError::Conflict` and has committed nothing. A single-op write (`put_doc`, `delete_doc`, `ack_mq`) is a `commit_txn` with an empty read set, which no other writer can make lose. The body may run more than once on a remote handle, so it must be free of side effects. A test runs every write body twice against the same state and checks that the second run stages the same ops (fresh minted ids aside). `fl_core::store` traits on `unix://`, `ws://`, and a connected `FerroStore::auto` use this path. The other methods in the table (`get_doc`, `query_by_field`, the searches, the columnar scans and rollups, `get_vector`, `publish_mq`, `poll_mq_group`) are one RPC each and need no session.

`fusion` is `{"method":"rrf","k":60}` (`k` optional, default 60, and `k >= 1`) or `{"method":"weighted","bm25":<finite >= 0>,"vector":<finite >= 0>}`. At least one weighted weight is greater than 0. `unix://` and `ws://` use the same JSON. On a remote handle the vector methods call these RPCs and return before a local transaction.

A columnar cell is `{"t":"null"}`, `{"t":"i64","v":1}`, `{"t":"f64","v":1.5}`, `{"t":"bool","v":true}`, `{"t":"utf8","v":"a"}`, `{"t":"bytes","v":[1,2]}` (numbers `0..=255`, not base64), or `{"t":"ts","v":1}`. A finite `f64`, including `-0.0`, is a JSON number and has no `bits` field. A non-finite `f64` is `{"t":"f64","v":"NaN","bits":<u64>}`, `{"t":"f64","v":"Infinity","bits":<u64>}`, or `{"t":"f64","v":"-Infinity","bits":<u64>}`, and `bits` is that value's `to_bits()`. A missing `bits` means canonical `NaN`, `Infinity`, or `-Infinity`. When `bits` is present it is an integer in `0..=u64::MAX` and `f64::from_bits` must be the same class as `v` (a `NaN` pattern with `v` of `Infinity` is an error). A finite number must not use a string `v`. `unix://` and `ws://` use the same JSON. On a remote handle the five columnar methods call these RPCs and do not also run a local transaction.

## Command line

The `ferrostore` binary (`src/bin/ferrostore.rs`, feature `cli`, on by default) is a thin argument parser over `fl_store::cli`. `fl store <command>` calls the same functions on the bound store, with the same flags, so the two print the same output except for the program's name. `ferrostore` never reads `FL_STORE_BACKEND`. A failure prints `error: <message>: <cause>…` on stderr and exits 2.

| Command | Flags | What it does |
| --- | --- | --- |
| `status --store PATH` | `--engine ferro\|redb`, `--json` | The engine, files, size, live and dead bytes and format version. Read-only |
| `checkpoint --store PATH` | `--engine` | Folds the WAL into the file. Nothing to do on redb |
| `compact --store PATH` | `--engine`, `--json` | See Compaction. Nothing to do on redb |
| `migrate --store PATH.redb` | `--dry-run`, `--json` | A verified migration to the sibling `.fls`. The `.redb` is kept, mode `0400`. Exit 1 when the copy does not verify |
| `verify --store PATH.redb` | `--json` | Compares the pair table by table and view by view. Exit 1 when they differ |
| `serve --store PATH` | `--create`; `--socket PATH` or `--tcp ADDR`; `--psk-file FILE` with `--server-id NAME`; `--token-file FILE`, `--tls-cert FILE` with `--tls-key FILE`, `--allow-plaintext-on IFACE` (repeatable), all TCP only; `--json` | Serves the store until SIGINT or SIGTERM (see Served mode) |
| `get --store PATH\|URI COLLECTION ID` | client flags | The document as JSON. Exit 1 when it is missing |
| `put --store PATH\|URI COLLECTION ID JSON\|-` | `--create`, client flags | Puts the document (`-` reads it from stdin, 64 MiB at most: a larger document is refused). Prints `{"collection", "id", "put": true}` |
| `query --store PATH\|URI COLLECTION FIELD VALUE` | client flags | `query_by_field`: `{"collection", "mode": "field", "field", "value", "ids"}` |
| `query --bm25 --store PATH\|URI COLLECTION TEXT` | `-k N` (default 10), client flags | `search_bm25`: `{"collection", "mode": "bm25", "query", "k", "hits": [{"id", "score"}]}` |
| `query --vector --store PATH\|URI COLLECTION --values JSON` | `-k N` (default 10), client flags | `search_vectors`: `{"collection", "mode": "vector", "k", "hits"}` |
| `inspect PATH` | `--json` | Both superblock slots, the schema version, the counts and the live and dead bytes. Read-only. Exit 1 when the live state cannot be read |
| `bench` | `--workload put,get,scan,vector\|all`, `--n N` (default 1000), `--engines ferro,redb`, `--dir DIR`, `--path STORE` with `--i-know-it-writes`, `--json` | Times each workload on each engine |

The client flags are `--psk-file FILE` with `--server-id NAME`, or `--token-file FILE`; `--ca-file FILE` or `--pin-spki HEX` for `wss://`; and `--allow-plaintext`. They map to `ConnectOptions` (see Authentication and TLS), and they apply to a `unix://`, `ws://` or `wss://` URI only. A path with any of them is refused. Any other `scheme://` is refused. `fl store get`, `put` and `query` take the URI with `--connect URI`, and the bound store otherwise. `fl store inspect` and `fl store serve` work on the bound store.

`--create` is a flag of `ferrostore put` and `ferrostore serve` only (see Which store). `fl store` has no `--create`: it creates the bound store itself.

**Which store.** A path names the `.fls` its engine uses: `x.redb` names `x.fls`. `x` names `x.fls` too. `get`, `put` and `query` open it embedded on FerroStore. `ferrostore put` and `ferrostore serve` create a missing `.fls` (mode `0600`) only with `--create`, and only in a directory that exists. Without `--create` they refuse a missing store. `fl store put` and `fl store serve` create the bound store's missing `.fls` without a flag, and make its missing parent directories (mode `0700`). They refuse when `FL_STORE_BACKEND=redb` puts the bound store on redb. Every other command refuses a missing store and creates nothing. A lone `.redb` with no `.fls` is refused, with `--create` too, and is not migrated: run `migrate`. This applies whether the store is named as `x.redb`, `x.fls`, `x`, or a symlink to the `.redb`. No store is created while a migration of it holds its migration lock. A reserved (`__`) collection is refused by `get`, `put` and `query`.

**Private copies.** `migrate --dry-run` and `verify` work on a copy of the `.redb`. `ferrostore` makes the copy in a new `0700` directory beside the store, `.<file>.scratch-<pid>-<n>`, on the store's own filesystem. `fl store` makes it in the system temp directory. Each command checks the free space first (each file's size plus 10%), and removes the directory before it returns, or on any failure.

**`serve`.** Without `--tcp`, the store is served on a Unix socket, `--socket` or `<file>.fls.sock`, checked by peer uid, and by the PSK when `--psk-file` is given. `--tcp` needs `--psk-file` with `--server-id`, `--token-file`, or both. With `--tls-cert` and `--tls-key` it serves `wss://` on any address. Without TLS it binds only a loopback address or an address of an interface that `--allow-plaintext-on` names. `serve` reads and checks the PSK, token and TLS files, and resolves the `--tcp` address and checks it against this bind policy, before it opens the store. So a `serve --create` refused for its flags, its auth files or its bind policy creates no `.fls` and no lock file. When the server listens, one line goes to stdout: `serving <file> on <uri> (auth: <checks>)`, or, with `--json`, one JSON object with `store`, `uri`, `auth` (`peer-uid`, `psk`, `token`), `tls` and `allow_plaintext_on`. No key, token or hash is printed. Before it opens anything, `serve` blocks SIGINT and SIGTERM in its thread (`pthread_sigmask`), so every thread it starts inherits the mask. When the server listens, it waits for one of the two signals (`sigwait`), drops the server, which stops accepting, closes its connections and removes its socket, prints `stopped on SIGINT` or `stopped on SIGTERM` on stderr, and exits 0.

**`inspect`.** The file is opened `O_RDONLY` and only read. No lock is taken. Nothing is created, initialised, repaired, migrated or checkpointed, so `inspect` is safe beside a live store and works on a `0400` file. For each slot it reports `state` (`valid`, `invalid`, `newer` or `absent`), `live`, `container_version`, `engine` and `incompat` (read when the slot has the magic), and for a valid slot its `epoch`, `snapshot_seq` and both extents. From the live slot's snapshot and WAL it reports `schema_version`, `epoch`, `last_seq`, `live_bytes`, `dead_bytes`, `documents`, each collection's document count (the store's own `__` collections included), each column table's columns, row groups and rows, and each vector collection's `count`, `dim`, `storage`, `metric`, `m`, `ef_construction` and `ef_search`. A file whose state cannot be read is still reported, slot by slot, with `error` set.

**`bench`.** By default each engine gets a fresh store in a new private directory, inside `--dir` or the system temp directory, which is removed afterwards. No existing store is opened. The inputs are fixed: document `i` is `doc-<i>`, and the vectors (dim 16, `f32`, cosine, `m` 16, `ef_construction` 64, `ef_search` 64) come from a fixed-seed generator. `put` times `n` single-document commits. `get` times `n` reads. `scan` times up to 20 reads of the whole collection. `vector` times `n` searches with k 10 over `n` vectors. `get` and `scan` load the documents first, untimed, when `put` did not run. redb has no vector search, so its `vector` workload is listed under `skipped`. Each result has `engine`, `workload`, `ops`, `items_per_op`, `p50_us`, `p99_us` (nearest rank), `mean_us`, `total_ms` and `ops_per_sec`. `--path STORE` runs on an existing FerroStore only with `--i-know-it-writes`, on `--engines ferro` and the document workloads. It writes to a new collection, `ferrostore-bench-<pid>-<n>`, and deletes its documents afterwards.

## Auto

`default_socket_path_for_store(path)` appends `.sock` to the file name. `dir/store.fls` is `dir/store.fls.sock`. A path with no file name (`/` or `..`) is that path joined with `ferrostore.sock`.

`FerroStoreServer::bind_unix(path, store)` makes the socket mode `0600` before it listens: `socket`, `bind`, `chmod 0600`, then `listen`, so no peer can connect while the mode is wider. A missing parent directory is created mode `0700`; an existing one keeps its mode. A socket already at `path` is removed only after a `connect` to it is refused. One that accepts belongs to a live server and the bind fails. A `path` that is not a socket is never removed.

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

## Transactions

```text
transaction(body: FnMut(&mut Transaction) -> Result<T, StoreError>) -> Result<T, StoreError>
```

`transaction` runs `body` over one `Transaction` and commits every write it makes as one sequence: one WAL frame on disk, one `SyncBatch`. Either all of them commit or none does. `Err` from `body` is returned and commits nothing. A `Transaction` write method that returns `Err` wrote nothing, and the transaction then cannot commit, even if `body` goes on and returns `Ok`. A body that writes nothing commits nothing.

| `Transaction` method | Same rules as |
| --- | --- |
| `get_doc(collection, id) -> Option<Value>` | `get_doc`, without a sync-adapter cache fill |
| `put_doc(collection, id, doc)`, `delete_doc(collection, id)` | `put_doc`, `delete_doc`. A reserved (`__`) collection is refused |
| `scan(collection) -> Vec<(String, Value)>` | every document, in id order |
| `query_by_field(collection, field, value) -> Vec<String>` | `query_by_field` |
| `create_column_table`, `append_column_rows` | the store methods |
| `has_vector_collection(name) -> bool` | `has_vector_collection` |
| `create_vector_collection`, `upsert_vector`, `delete_vector`, `get_vector` | the store methods. On a remote handle `get_vector` of a collection the transaction wrote is refused |

Every read sees the transaction's own earlier writes. Nothing is visible to another reader until the commit.

`transaction` closes read-modify-write races. On `mem://` and `file://` the body runs under the write lock: on disk it joins a group commit under `LOCK_EX`, against the state every other handle committed, so no commit lands between a read in the body and the write that depends on it. On a remote handle the body reads one snapshot through a read session, and the commit is validated by value on the server: it commits only when every read of the body still has the value the body saw, and otherwise the body runs again over a new session, at most 8 times in all (see Remote transactions). So the body must be free of side effects. On disk the body runs exactly once.

The body reads and writes through its `Transaction` only. While it runs, every call on the same store handle from the thread that runs it (a read, a write, `checkpoint`, a trait method) returns `StoreError::Backend("FerroStore called from inside its own transaction body")` at once and does nothing. Without that check, such a call would wait for the lock the body holds (`mem://`: any call; `file://`: a write), read the committed state without the body's own writes (a `file://` or remote read), or commit apart from the body (a remote write). The refusal goes back to the body. Returned with `?`, the transaction commits nothing and returns it. Ignored, the transaction commits what the body staged. In a group commit each member's body runs on the member's own thread, and that thread is refused only while its body runs. A call from another thread is never refused, so a body that waits on another thread that calls the same store deadlocks: that call waits for the lock the body holds (`mem://`: any call; `file://`: a write).

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

Subscribers are in-process only: a `Mutex<Vec<Sender<MqRecord>>>` map. `publish_mq` sends a clone after the durable commit and after the file lock is released. A publish that returns `StoreError::Undelivered` (durable, its batch not sent to the `WriteThrough` adapter) still sends it, so live push goes on through an adapter outage, server-side subscribers included. A publish that returns any other error committed nothing and sends nothing. A disconnected subscriber is dropped. `subscribe_mq` does not replay history; it receives only later publishes on this handle.

On a remote handle, a `publish_mq` that the server answers with `Undelivered` is durable on the server, and the server's subscribers get the message. The client registers no batch with its own adapter for it, because the error carries the store sequence and not the message sequence the batch's ops need, and the caller gets no message sequence either.

## Phases

Phase 1 (Record 14) is this engine: the `.fls` container, `mem://`, `file://`, the document store, and FerroMQ.

Phase 2 (Record 13) is served mode: Unix-socket and WebSocket RPC, and `FerroStore::auto`. `unix://`, `ws://` and `wss://` are served.

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

An id is non-empty, has no NUL, and is at most 1024 bytes. `/` is allowed. `upsert_vector` replaces the bytes and the metadata. `None` metadata is absent. `Some(null)` stores JSON null. `get_vector` of a missing id is `Ok(None)`. `delete_vector` of a missing id succeeds and still commits a delete. A later upsert of that id works, and search does not return a deleted id. A missing collection is `StoreError::Backend` and writes nothing.

```text
upsert_vectors(collection, items: &[(&str, &[f32], Option<&Value>)]) -> Result<Option<u64>, StoreError>
has_vector_collection(name) -> Result<bool, StoreError>
```

`upsert_vectors` is one commit for every item: one sequence, one WAL frame, and one copy of the collection. The result is the same as `upsert_vector` for each item in order, graph and entry point included, so an id given twice ends as its last item. Every item is checked (id, length, finite components) before any is applied: one bad item, or a missing collection, writes nothing. `Ok(Some(seq))` is the sequence the batch committed at. An empty `items` commits nothing and is `Ok(None)`; a missing collection is still an error. `has_vector_collection` is a read: it never creates the collection and commits nothing. A document collection of that name does not count.

`apply_op` checks the collection, the byte length, and that every stored `f32` or `f16` decodes finite. A bad check is an error, and `stage_ops` publishes nothing.

HNSW parameters live on the collection. There is no hidden default `M`. The level of an id is deterministic: hash the id with SHA-256, take the first 8 bytes as a little-endian `u64`, replace `0` with `1`, set `u = x / u64::MAX` in `(0, 1]`, and set `level = floor(-ln(u) / ln(M))`, clamped to `0..=16`. Layer 0 allows `2 * M` neighbors. Every upper layer allows `M`. Neighbors are ordered by smaller graph distance, then lower id. The first vector is the entry point and has no links. Later inserts search from the entry point: greedy with ef 1 above the new level, then ef `ef_construction` from `min(new level, entry level)` down to layer 0. Each layer keeps the closest `max_links` neighbors, links both ways, and prunes the other node back to its max links. The entry point is the live node with the highest level, then the lowest id. An insert keeps it in O(1): the new id becomes the entry when its level is higher, or equal with a lower id. Deleting the entry (also when an upsert replaces it) finds the next one by a scan of the collection. Upsert unlinks the id and inserts it again. Delete unlinks it and removes it. A neighbor id that is not live is not left behind. Links are bidirectional.

`search_vectors` is greedy from the entry point down to layer 1 with ef 1, then layer 0 with width `max(ef, limit)`. It returns at most `limit` live nodes. `limit == 0` or `ef == Some(0)` returns an empty vector and is not an error. `ef == None` uses the collection `ef_search`.

`search_hybrid` takes at most `limit` hits from `search_bm25` and at most `limit` hits from `search_vectors`, fuses that pool, and truncates to `limit`. An id on only one side still counts. `limit == 0` returns empty. An empty or token-free query makes the BM25 side empty and does not error; the vector side still uses the query vector. Reciprocal rank fusion (`k >= 1`) sums `1 / (k + rank)` with ranks starting at 1 in each side's existing order. Weighted fusion requires finite weights `>= 0` with at least one `> 0`. Each side is min-max normalized. An empty side adds nothing. When `max == min`, every hit on that side has normalized score 1. A missing side contributes 0. The fused score is `bm25_weight * norm_bm25 + vector_weight * norm_vector`. Order is fused score descending, then id ascending.

`search_knowledge` uses that fusion and those two collections. Each hit is the id, the fused score, the vector metadata when the id is in the vector collection (`None` when it is absent or when the id is BM25-only), and the document when the id is in the document collection (`None` otherwise). The caller stores a knowledge entry with `put_doc` and its embedding with `upsert_vector` under the same id. FerroStore does not call an embedding model, and this phase does not change the redb state format.

A checkpoint with any vector collection writes snapshot version 3. A document-only store stays version 1 and byte-identical. Columnar tables with no vector collections stay version 2. `migrate_redb_to_ferrostore` refuses a non-empty document map (the new store's own `__meta` document aside), column-table map, or vector map. It writes redb's `meta.format_version` to `("__meta", "schema_version")`.

`fl_store::migrate::verify_redb_vs_ferrostore(src, dst)` returns a `ParityReport`: one `TableParity` per redb table the migration copies, plus `__meta`, `redb:<name>` for a redb table outside the copy list, and `ferrostore:unlisted` for FerroStore data outside it; and one per list-style trait view, run on both stores. Each side is a row count and a SHA-256 over its rows sorted by key, each row `len(key) ‖ key ‖ len(value) ‖ canonical-JSON(value)` (lengths `u64` little-endian, canonical JSON with sorted object keys and no whitespace). A raw `&str` is hashed as the JSON string the migration stores. A view's items are keyed by their own canonical JSON, so a view compares content, not order. It is `Ok` when the stores differ and `Err` only when one cannot be read. `fl_store::migrate::migrate_verified(redb_path, fls_path)` first removes this target's stale temp files: regular files (not symlinks) beside `fls_path` named `<name>.migrating-<pid>-<n>-<nonce>` for this `<name>`, owned by the current user and not modified for 24 hours; a sweep that fails is a warning on stderr. It then migrates into a new `<name>.migrating-<pid>-<n>-<nonce>` (mode `0600`, `O_EXCL`, so a taken name is an error and never reused), verifies, and on a report that is not clean deletes that file and returns the report with no `fls_path` made. Otherwise it `fsync`s the file, moves it to `fls_path` (`link` then `unlink`, so an `fls_path` that appeared meanwhile is never replaced), `fsync`s the directory, and makes the `.redb` mode `0400`. It holds the `.redb` open, and so redb's exclusive lock, from before the copy until after the `chmod`, so no other process can write the `.redb` once it is copied. A redb open refused for permission on a file with no owner write bit fails with an error that names the sibling `.fls`, says whether it exists, and gives the two ways out: the FerroStore engine, or `chmod 600` on the `.redb` to roll back. It links only the file it created and verified (device and inode checked before and after the link) and removes only a temp file it created. Once the link is made the result is `Ok`: a failure to remove the temp file, sync the directory or `chmod` the `.redb` is a warning on stderr. It refuses an existing `fls_path`.

`fl_store::open_store` uses FerroStore by default, and runs `migrate_verified` on the first open of a store that has only a `.redb`. `FL_STORE_BACKEND=redb` selects redb instead.

## Sync adapter

A process may attach one `SyncAdapter` in a `SyncMode`: `WriteThrough`, `WriteBehind`, `LocalCache`, or `CloudPreCache`. The mode, the adapter, the pending batches, and `synced_seq` are process memory. They are not written to the `.fls` file. A reopened store has no adapter and `synced_seq` of 0.

After a commit that advances the store sequence, the store builds a `SyncBatch` of that sequence and all its ops. An empty op list does not. The adapter is never called by a thread that holds the memory document mutex or the disk `flock`. (A group member's `on_commit` can run while the group's leader holds the `flock` again for an auto-checkpoint; a write it makes waits for that checkpoint, as it would for any other writer.) `WriteThrough` calls `on_commit` after the commit is published, and only one of those calls is in flight. A newer commit cannot call `on_commit` while an older notify is in flight or still queued. If the pending buffer is non-empty, or a `WriteThrough` `on_commit` is already in flight, the batch is appended behind them, and the commit then drains with `sync_flush` (it waits out the notify in flight and sends the pending batches oldest first) until its own batch is sent. A `WriteThrough` commit returns `Ok` only once its own batch is sent. The one exception is a commit made inside the in-flight `on_commit`, on that call's own thread: it stays queued and returns, as `sync_flush` does there. Success sets `synced_seq` to that sequence when it is greater. Failure leaves the commit applied, puts that batch at the front of the pending buffer, ahead of any batch appended during the call, does not advance `synced_seq`, and returns `StoreError::Undelivered { seq, cause }`: `seq` is the commit's own sequence, `cause` is the adapter's error, and the message reads `committed at seq N (durable), not yet delivered to the sync adapter: <cause>`. A failed drain (an older queued batch failed) returns `Undelivered` the same way: the commit is durable and its batch stays pending. A write that did not commit never returns `Undelivered`, so a caller tells the two apart; retrying an `Undelivered` write applies it a second time. A loud multi-batch apply whose `WriteThrough` flush fails returns `Undelivered` at the last applied sequence.

An `on_commit` that blocks waiting for a commit made on another thread through the same handle deadlocks under `WriteThrough`: that commit waits for the in-flight `on_commit` to return before its own batch can be sent. An HTTP adapter pointed back at a server that writes to the same handle is one example. `WriteBehind`, `LocalCache`, and `CloudPreCache` append the batch and do not call `on_commit`. A silent commit does not call `on_commit` and is not queued. It may set `synced_seq` to that sequence only when the pending buffer is empty and the sequence is greater. With no adapter, commits are unchanged.

`sync_flush` waits while a notify is in flight, then sends only the oldest pending batch. It repeats in order until the buffer is empty, in every mode. It does not send a batch that is not the front, and it does not send while another `on_commit` is in flight. The first error leaves that batch and the later ones pending. Success of a batch removes it when it is still the front and sets `synced_seq` to that sequence when it is greater. A failed `WriteThrough` batch stays ahead of later commits until a flush sends it: an explicit `sync_flush` or the drain of the next `WriteThrough` commit. A loud multi-batch apply calls `sync_flush` only in `WriteThrough`; the other modes keep the batches pending, and `sync_flush` from inside the in-flight `on_commit` returns without sending.

`get_doc` in `LocalCache` or `CloudPreCache`, and only when the pending buffer is empty, treats a missing document as a cache miss and calls `fetch_document`. `Ok(None)` stays a miss. `Ok(Some(doc))` is stored only when that id is still absent and the pending buffer is still empty. The check and the write hold the document lock and then the sync mutex, and no other commit registers until that check finishes, so a document inserted after the miss is not overwritten. If the id is already present, that stored document is returned and nothing is written. If the pending buffer filled during the fetch, the fetched value is dropped: an id that is now present is returned, and an id that is still absent stays missing. A silent put does not call `on_commit` and is not queued. When the pending buffer is empty, it sets `synced_seq` to the new sequence. A stored JSON null is a hit, not a miss. A missing remote document is a JSON null result. Other modes do not fetch. A fetch error is returned. A remote fill rechecks the client pending buffer, then sends `put_doc` with `silent` and `if_absent`. The server writes only when the id is absent. Silent means the server adapter is not notified. The client sets `synced_seq` from that store sequence only when its own pending buffer is still empty. If the server wrote nothing, the client returns the server's current document and does not advance `synced_seq`.

`sync_pull` calls `pull_since` with the store sequence. `sync_prewarm` calls `prewarm`. Both require an adapter and an empty pending buffer. Every batch is dry-run first. The store publishes them only when every batch applies: the first sequence is the current sequence plus one, each later sequence is strictly one greater, and an empty op list is not a commit. Reserved ops are applied or dropped as `__meta` describes above; a batch whose ops were all dropped still advances the sequence, as an empty WAL frame. A loud apply queues the batches as they came in, not as applied. On disk, a multi-batch apply publishes one superblock whose `wal_length` covers every frame. Anything that fails before that publish leaves the previous superblock current. A bad op or a bad sequence writes nothing and leaves `synced_seq` unchanged. An empty result writes nothing and returns the current `synced_seq`. After a successful apply, `synced_seq` is the last applied sequence, and `on_commit` is not called. A remote client sends the batches in one `apply_sync_batches` with `silent` set, so the adapter on the store that owns the file is not notified. The mode, the adapter, the pending batches, and `synced_seq` stay process memory. They are not written to the `.fls` file.
