//! Track 0 of the v2 cutover spec: O-13 permissions (§3.1) and the container
//! and schema versions (§3.2).

use super::layout::{self, FORMAT_V1, FORMAT_V2, SLOT_LEN, SlotRead, Superblock};
use super::{FerroStore, FerroStoreServer, latest_superblock};
use crate::{FORMAT_VERSION, FORMAT_WITH_ROUTING, FORMAT_WITH_WORKFLOW};
use fl_core::StoreError;
use serde_json::json;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::{FileExt, FileTypeExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;

fn mode(path: &Path) -> u32 {
    std::fs::symlink_metadata(path)
        .unwrap()
        .permissions()
        .mode()
        & 0o7777
}

fn set_mode(path: &Path, mode: u32) {
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
}

fn read_slot(path: &Path, index: usize) -> [u8; SLOT_LEN] {
    let file = File::open(path).unwrap();
    let mut buf = [0u8; SLOT_LEN];
    file.read_exact_at(&mut buf, (index * SLOT_LEN) as u64)
        .unwrap();
    buf
}

fn write_raw_slot(path: &Path, index: usize, bytes: &[u8; SLOT_LEN]) {
    let file = OpenOptions::new().write(true).open(path).unwrap();
    file.write_all_at(bytes, (index * SLOT_LEN) as u64).unwrap();
    file.sync_all().unwrap();
}

/// The live slot's index and superblock.
fn live(path: &Path) -> (usize, Superblock) {
    latest_superblock(&File::open(path).unwrap()).unwrap()
}

// ---- §3.1 permissions -------------------------------------------------------

/// The work of the umask test: a new store, its new parent directory and a
/// server socket in another new directory are private. It runs in a child
/// process `sh` started under `umask 022`, because the umask is per process and
/// a test thread must not change it for the other tests in this binary.
fn new_store_dir_and_socket_are_private() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("fresh");
    let store = dir.join("store.fls");
    drop(FerroStore::open(&store).unwrap());
    assert_eq!(mode(&dir), 0o700, "the directory FerroStore created");
    assert_eq!(mode(&store), 0o600, "the store file FerroStore created");

    let sock_dir = tmp.path().join("run");
    let sock = sock_dir.join("store.fls.sock");
    let server = FerroStoreServer::bind_unix(&sock, FerroStore::in_memory()).unwrap();
    assert_eq!(mode(&sock_dir), 0o700, "the socket directory");
    assert!(
        std::fs::symlink_metadata(&sock)
            .unwrap()
            .file_type()
            .is_socket()
    );
    assert_eq!(mode(&sock), 0o600, "the socket");
    let client = FerroStore::connect_uri(&format!("unix://{}", sock.display())).unwrap();
    client.put_doc("c", "k", &json!(1)).unwrap();
    assert_eq!(server.store().get_doc("c", "k").unwrap(), Some(json!(1)));
}

#[test]
fn a_new_store_its_directory_and_the_socket_are_private_under_umask_022() {
    const CHILD: &str = "FL_FERROSTORE_UMASK_CHILD";
    if std::env::var_os(CHILD).is_some() {
        // Proof the child really runs under 022: a plain create is 0644 here.
        let probe_dir = tempfile::tempdir().unwrap();
        let probe = probe_dir.path().join("probe");
        drop(File::create(&probe).unwrap());
        assert_eq!(mode(&probe), 0o644, "the child must run under umask 022");
        new_store_dir_and_socket_are_private();
        return;
    }
    let exe = std::env::current_exe().unwrap();
    let out = std::process::Command::new("sh")
        .arg("-c")
        .arg(
            "umask 022; exec \"$0\" --exact \
             ferrostore::tests_track0::a_new_store_its_directory_and_the_socket_are_private_under_umask_022 \
             --test-threads=1",
        )
        .arg(exe)
        .env(CHILD, "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "child failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // A filter that matched no test also exits 0: the child must have run this one.
    assert!(
        stdout.contains("1 passed"),
        "the child did not run the test:\n{stdout}"
    );
}

#[test]
fn an_existing_loose_store_is_warned_about_and_never_chmodded() {
    crate::private::TEST_TTY.with(|t| t.set(true));
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("old");
    std::fs::create_dir(&dir).unwrap();
    set_mode(&dir, 0o755);
    let path = dir.join("store.fls");
    drop(FerroStore::open(&path).unwrap());
    // Loosened the way an older fl's umask would have left it.
    set_mode(&path, 0o644);

    let store = FerroStore::open(&path).unwrap();
    store.put_doc("c", "k", &json!(1)).unwrap();
    drop(store);
    crate::private::TEST_TTY.with(|t| t.set(false));
    assert_eq!(mode(&path), 0o644, "the file is not chmodded");
    assert_eq!(mode(&dir), 0o755, "the directory is not chmodded");
    let needle = dir.display().to_string();
    let warned: Vec<String> = crate::private::EMITTED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .filter(|w| w.contains(&needle))
        .cloned()
        .collect();
    let file_warning = warned
        .iter()
        .find(|w| w.contains("chmod 600"))
        .unwrap_or_else(|| panic!("no file warning: {warned:?}"));
    assert!(file_warning.contains(&path.display().to_string()));
    assert!(file_warning.contains("644"));
}

#[test]
fn an_existing_parent_directory_keeps_its_mode() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("shared");
    std::fs::create_dir(&dir).unwrap();
    set_mode(&dir, 0o755);
    let path = dir.join("store.fls");
    drop(FerroStore::open(&path).unwrap());
    assert_eq!(mode(&dir), 0o755);
    assert_eq!(mode(&path), 0o600);
}

#[test]
fn the_socket_is_0600_and_only_a_refusing_socket_is_replaced() {
    let tmp = tempfile::tempdir().unwrap();
    let sock = tmp.path().join("s.sock");
    let live_server = FerroStoreServer::bind_unix(&sock, FerroStore::in_memory()).unwrap();
    assert_eq!(mode(&sock), 0o600);

    // A live socket accepts a connection, so a second bind leaves it alone.
    let err = match FerroStoreServer::bind_unix(&sock, FerroStore::in_memory()) {
        Ok(_) => panic!("a second server replaced a live socket"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("already listening"), "{err}");
    let client = FerroStore::connect_uri(&format!("unix://{}", sock.display())).unwrap();
    client.put_doc("c", "k", &json!("still served")).unwrap();
    assert_eq!(
        live_server.store().get_doc("c", "k").unwrap(),
        Some(json!("still served"))
    );
    drop(client);
    drop(live_server);

    // A socket nobody listens on refuses the connect: it is stale and replaced.
    let stale = tmp.path().join("stale.sock");
    drop(UnixListener::bind(&stale).unwrap());
    assert!(std::fs::symlink_metadata(&stale).is_ok());
    assert!(UnixStream::connect(&stale).is_err());
    let server = FerroStoreServer::bind_unix(&stale, FerroStore::in_memory()).unwrap();
    assert_eq!(mode(&stale), 0o600);
    assert!(UnixStream::connect(&stale).is_ok());
    drop(server);

    // A path that is not a socket is never removed.
    let plain = tmp.path().join("plain");
    std::fs::write(&plain, b"keep").unwrap();
    assert!(FerroStoreServer::bind_unix(&plain, FerroStore::in_memory()).is_err());
    assert_eq!(std::fs::read(&plain).unwrap(), b"keep");
}

// ---- §3.2 container format --------------------------------------------------

#[test]
fn every_slot_written_is_format_2_and_round_trips() {
    let sb = Superblock {
        epoch: 7,
        snapshot_offset: 8192,
        snapshot_length: 3,
        wal_offset: 9000,
        wal_length: 4,
        snapshot_seq: 5,
    };
    let bytes = layout::encode_superblock(&sb);
    assert_eq!(layout::slot_version(&bytes), Some(FORMAT_V2));
    assert_eq!(&bytes[64..76], &[0u8; 12], "engine 0, no feature bits");
    assert_eq!(layout::parse_superblock(&bytes), SlotRead::Valid(sb));
    let v1 = layout::encode_superblock_as(&sb, FORMAT_V1, 0, 0);
    assert_eq!(layout::slot_version(&v1), Some(FORMAT_V1));
    assert_eq!(layout::parse_superblock(&v1), SlotRead::Valid(sb));
    // A torn slot is invalid, never "newer".
    let mut torn = layout::encode_superblock_as(&sb, 3, 0, 0);
    torn[30] ^= 0xff;
    assert_eq!(layout::parse_superblock(&torn), SlotRead::Invalid);
}

#[test]
fn a_v1_file_opens_and_its_next_slot_write_is_v2() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("old.fls");
    {
        let store = FerroStore::open(&path).unwrap();
        store.put_doc("c", "a", &json!(1)).unwrap();
    }
    // Rewrite both slots as format 1, the layout every older build wrote.
    for index in 0..2 {
        if let SlotRead::Valid(sb) = layout::parse_superblock(&read_slot(&path, index)) {
            write_raw_slot(
                &path,
                index,
                &layout::encode_superblock_as(&sb, FORMAT_V1, 0, 0),
            );
        }
    }
    let (live_index, before) = live(&path);
    assert_eq!(
        layout::slot_version(&read_slot(&path, live_index)),
        Some(FORMAT_V1)
    );

    let store = FerroStore::open(&path).unwrap();
    assert_eq!(store.get_doc("c", "a").unwrap(), Some(json!(1)));
    store.put_doc("c", "b", &json!(2)).unwrap();
    drop(store);

    let (next_index, after) = live(&path);
    assert_eq!(next_index, live_index ^ 1);
    assert_eq!(after.epoch, before.epoch + 1);
    assert_eq!(
        layout::slot_version(&read_slot(&path, next_index)),
        Some(FORMAT_V2),
        "the slot write upgraded the file"
    );
    // The other slot is still a valid format-1 slot: both stay CRC-valid.
    let other = read_slot(&path, live_index);
    assert_eq!(layout::slot_version(&other), Some(FORMAT_V1));
    assert_eq!(layout::parse_superblock(&other), SlotRead::Valid(before));
    let store = FerroStore::open(&path).unwrap();
    assert_eq!(store.get_doc("c", "a").unwrap(), Some(json!(1)));
    assert_eq!(store.get_doc("c", "b").unwrap(), Some(json!(2)));
}

/// A store whose next slot is a CRC-valid slot that `version`, `engine` and
/// `incompat` describe, at a higher epoch than the live one.
fn store_with_a_newer_slot(version: u32, engine: u32, incompat: u64) -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("store.fls");
    {
        let store = FerroStore::open(&path).unwrap();
        store.put_doc("c", "a", &json!(1)).unwrap();
    }
    let (index, mut sb) = live(&path);
    sb.epoch += 1;
    let slot = layout::encode_superblock_as(&sb, version, engine, incompat);
    write_raw_slot(&path, index ^ 1, &slot);
    tmp
}

#[test]
fn a_newer_container_is_refused_with_the_version_it_found() {
    for (version, engine, incompat, want_bits) in [
        (3, 0, 0, 0),
        (FORMAT_V2, 0, 1 << 7, 1 << 7),
        (FORMAT_V2, 1, 0, 0),
    ] {
        let tmp = store_with_a_newer_slot(version, engine, incompat);
        let path = tmp.path().join("store.fls");
        let before = std::fs::read(&path).unwrap();
        let err = match FerroStore::open(&path) {
            Ok(_) => panic!("opened a container from a newer build: {version} {engine} {incompat}"),
            Err(err) => err,
        };
        match &err {
            StoreError::ContainerFormat {
                store,
                found,
                engine: found_engine,
                unknown_incompat,
                newest,
            } => {
                assert_eq!(*found, version);
                assert_eq!(*found_engine, engine);
                assert_eq!(*unknown_incompat, want_bits);
                assert_eq!(*newest, FORMAT_V2);
                assert_eq!(store, &path.display().to_string());
            }
            other => panic!("not ContainerFormat: {other:?}"),
        }
        let msg = err.to_string();
        assert!(msg.contains("Upgrade fl"), "{msg}");
        assert!(
            msg.contains(&format!("container format {version}")),
            "{msg}"
        );
        assert_eq!(std::fs::read(&path).unwrap(), before, "nothing was written");
    }
}

#[test]
fn two_valid_slots_with_one_epoch_are_corrupt_across_formats() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("store.fls");
    {
        let store = FerroStore::open(&path).unwrap();
        store.put_doc("c", "a", &json!(1)).unwrap();
    }
    let (index, sb) = live(&path);
    write_raw_slot(
        &path,
        index ^ 1,
        &layout::encode_superblock_as(&sb, FORMAT_V1, 0, 0),
    );
    let err = match FerroStore::open(&path) {
        Ok(_) => panic!("opened a file whose two slots share an epoch"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("share an epoch"), "{err}");
}

// ---- §3.2 schema version ----------------------------------------------------

#[test]
fn a_new_store_starts_at_the_newest_schema_and_keeps_it() {
    assert_eq!(
        FerroStore::in_memory().format_version().unwrap(),
        Some(FORMAT_WITH_WORKFLOW)
    );
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("store.fls");
    {
        let store = FerroStore::open(&path).unwrap();
        assert_eq!(store.format_version().unwrap(), Some(FORMAT_WITH_WORKFLOW));
        store.put_doc("c", "a", &json!(1)).unwrap();
        store.checkpoint().unwrap();
    }
    let store = FerroStore::open(&path).unwrap();
    assert_eq!(store.format_version().unwrap(), Some(FORMAT_WITH_WORKFLOW));
    // The version document is not a WAL frame: a new store is at sequence 0.
    assert_eq!(
        super::read_consistent(&File::open(&path).unwrap())
            .unwrap()
            .2
            .last_seq,
        1,
        "one put and nothing else"
    );
}

#[test]
fn a_schema_newer_than_this_build_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("store.fls");
    {
        let store = FerroStore::open(&path).unwrap();
        store
            .put_doc_unchecked(
                "__meta",
                "schema_version",
                &json!(FORMAT_WITH_WORKFLOW + 1),
                false,
            )
            .unwrap();
    }
    let err = match FerroStore::open(&path) {
        Ok(_) => panic!("opened a store from a newer schema"),
        Err(err) => err,
    };
    assert!(
        matches!(
            err,
            StoreError::FormatVersion { found: Some(v), oldest: FORMAT_VERSION, newest: FORMAT_WITH_WORKFLOW }
                if v == FORMAT_WITH_WORKFLOW + 1
        ),
        "{err:?}"
    );
    let msg = err.to_string();
    assert!(msg.contains("Upgrade fl"), "{msg}");

    // One below the ladder is refused too, as redb refuses it.
    let path = tmp.path().join("old.fls");
    {
        let store = FerroStore::open(&path).unwrap();
        store
            .put_doc_unchecked("__meta", "schema_version", &json!(1), false)
            .unwrap();
    }
    assert!(matches!(
        FerroStore::open(&path),
        Err(StoreError::FormatVersion { found: Some(1), .. })
    ));
}

#[test]
fn a_store_from_before_schema_versioning_opens_and_raises_into_meta() {
    use fl_core::routing::RoutingMap;
    use fl_core::store::Catalog;
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("store.fls");
    {
        let store = FerroStore::open(&path).unwrap();
        store
            .delete_doc_unchecked("__meta", "schema_version")
            .unwrap();
    }
    let store = FerroStore::open(&path).unwrap();
    assert_eq!(store.format_version().unwrap(), None, "no version document");
    let p = store.add_project("/p").unwrap();
    store.set_routes(&p, &RoutingMap::starting()).unwrap();
    assert_eq!(store.format_version().unwrap(), Some(FORMAT_WITH_ROUTING));
    assert_eq!(
        store.get_doc("__meta", "schema_version").unwrap(),
        Some(json!(FORMAT_WITH_ROUTING))
    );
    drop(store);

    // A store that raised before `__meta` existed kept the version in `meta`.
    let path = tmp.path().join("legacy.fls");
    {
        let store = FerroStore::open(&path).unwrap();
        store
            .delete_doc_unchecked("__meta", "schema_version")
            .unwrap();
        store
            .put_doc("meta", "format_version", &json!(FORMAT_WITH_ROUTING))
            .unwrap();
    }
    let store = FerroStore::open(&path).unwrap();
    assert_eq!(store.format_version().unwrap(), Some(FORMAT_WITH_ROUTING));
}

#[test]
fn a_migration_copies_the_sources_schema_version() {
    let tmp = tempfile::tempdir().unwrap();
    let redb = crate::RedbStore::open(&tmp.path().join("src.redb")).unwrap();
    assert_eq!(redb.format_version().unwrap(), Some(FORMAT_VERSION));
    let ferro = FerroStore::open(tmp.path().join("dst.fls")).unwrap();
    assert_eq!(ferro.format_version().unwrap(), Some(FORMAT_WITH_WORKFLOW));
    crate::migrate_redb_to_ferrostore(&redb, &ferro).unwrap();
    assert_eq!(ferro.format_version().unwrap(), Some(FORMAT_VERSION));
    assert_eq!(ferro.get_doc("meta", "format_version").unwrap(), None);
}

#[test]
fn the_schema_document_is_in_no_trait_listing() {
    use fl_core::store::{Catalog, StateStore, WorkflowStore};
    let store = FerroStore::in_memory();
    // Not vacuous: the document is there to leak.
    assert_eq!(
        store.get_doc("__meta", "schema_version").unwrap(),
        Some(json!(FORMAT_WITH_WORKFLOW))
    );
    assert!(store.list_projects().unwrap().is_empty());
    assert!(store.list_state(None, None).unwrap().is_empty());
    assert!(store.list_workflow_defs(None).unwrap().is_empty());
    assert!(
        store
            .query_by_field(
                "__meta",
                "schema_version",
                &FORMAT_WITH_WORKFLOW.to_string()
            )
            .unwrap()
            .is_empty()
    );
}

// ---- a first open cut short -------------------------------------------------

/// The snapshot a new store's `init_empty` writes after the header.
fn new_store_blob() -> Vec<u8> {
    layout::encode_snapshot(0, &super::new_store_docs()).unwrap()
}

/// A file as `init_empty` leaves it when power is cut after the snapshot write
/// and before slot 0: a zero header, then `tail`.
fn cut_after_snapshot(path: &Path, tail: &[u8]) {
    let mut bytes = vec![0u8; layout::HEADER_BYTES as usize];
    bytes.extend_from_slice(tail);
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn a_first_open_cut_short_after_its_snapshot_reopens_as_a_new_store() {
    let blob = new_store_blob();
    assert!(blob.len() > 1);
    let tmp = tempfile::tempdir().unwrap();
    for (name, tail) in [
        ("whole.fls", &blob[..]),
        ("half.fls", &blob[..blob.len() / 2]),
        ("one.fls", &blob[..1]),
        ("none.fls", &blob[..0]),
    ] {
        let path = tmp.path().join(name);
        cut_after_snapshot(&path, tail);
        let store = FerroStore::open(&path).unwrap_or_else(|err| panic!("{name}: {err}"));
        assert_eq!(
            store.format_version().unwrap(),
            Some(FORMAT_WITH_WORKFLOW),
            "{name}"
        );
        store.put_doc("c", "a", &json!(1)).unwrap();
        drop(store);
        let store = FerroStore::open(&path).unwrap();
        assert_eq!(store.get_doc("c", "a").unwrap(), Some(json!(1)), "{name}");
    }
}

#[test]
fn a_zero_header_with_other_trailing_bytes_is_never_rewritten() {
    let blob = new_store_blob();
    let mut other = blob.clone();
    let last = other.len() - 1;
    other[last] ^= 0xff;
    let mut longer = blob.clone();
    longer.push(0);
    let tmp = tempfile::tempdir().unwrap();
    for (name, tail) in [
        ("differs.fls", other),
        ("longer.fls", longer),
        ("data.fls", b"real data".to_vec()),
    ] {
        let path = tmp.path().join(name);
        cut_after_snapshot(&path, &tail);
        let before = std::fs::read(&path).unwrap();
        assert!(FerroStore::open(&path).is_err(), "{name} opened");
        assert_eq!(
            std::fs::read(&path).unwrap(),
            before,
            "{name} was rewritten"
        );
    }
}

#[test]
fn ensure_superblock_says_when_it_initialized_the_file() {
    // `open` syncs the directory on `true`, so a file an earlier open created
    // and left without a superblock is made durable by the open that fills it.
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("store.fls");
    std::fs::write(&path, b"").unwrap();
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    assert!(super::ensure_superblock(&file).unwrap(), "empty file");
    assert!(
        !super::ensure_superblock(&file).unwrap(),
        "already has slot 0"
    );
    drop(file);
    let path = tmp.path().join("cut.fls");
    cut_after_snapshot(&path, &new_store_blob());
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    assert!(super::ensure_superblock(&file).unwrap(), "cut-short file");
}

// ---- reserved collections ---------------------------------------------------

mod reserved {
    use super::super::{DocState, FerroStore, Inner, Op, SyncAdapter, SyncBatch, SyncMode, SyncOp};
    use crate::{FORMAT_VERSION, FORMAT_WITH_ROUTING, FORMAT_WITH_STATE, FORMAT_WITH_WORKFLOW};
    use fl_core::StoreError;
    use fl_core::routing::RoutingMap;
    use fl_core::store::Catalog;
    use serde_json::{Value, json};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct Recorder {
        commits: Mutex<Vec<SyncBatch>>,
        pull: Mutex<Vec<SyncBatch>>,
        fetches: AtomicUsize,
    }

    impl SyncAdapter for Recorder {
        fn name(&self) -> &str {
            "recorder"
        }
        fn on_commit(&self, batch: &SyncBatch) -> Result<(), StoreError> {
            self.commits.lock().unwrap().push(batch.clone());
            Ok(())
        }
        fn pull_since(&self, _: u64) -> Result<Vec<SyncBatch>, StoreError> {
            Ok(self.pull.lock().unwrap().clone())
        }
        fn fetch_document(&self, _: &str, _: &str) -> Result<Option<Value>, StoreError> {
            self.fetches.fetch_add(1, Ordering::SeqCst);
            Ok(Some(json!(1)))
        }
        fn prewarm(&self, _: &[&str]) -> Result<Vec<SyncBatch>, StoreError> {
            Ok(self.pull.lock().unwrap().clone())
        }
    }

    fn attached(store: FerroStore, mode: SyncMode) -> (FerroStore, Arc<Recorder>) {
        let recorder = Arc::new(Recorder::default());
        let store = store
            .with_sync_adapter(mode, Arc::clone(&recorder) as Arc<dyn SyncAdapter>)
            .unwrap();
        (store, recorder)
    }

    fn assert_reserved(err: StoreError) {
        assert!(err.to_string().contains("reserved"), "{err}");
    }

    /// A store migrated from a new redb: schema `FORMAT_VERSION`, so a trait
    /// write raises it.
    fn migrated(dir: &std::path::Path) -> FerroStore {
        let redb = crate::RedbStore::open(&dir.join("src.redb")).unwrap();
        let store = FerroStore::in_memory();
        crate::migrate_redb_to_ferrostore(&redb, &store).unwrap();
        assert_eq!(store.format_version().unwrap(), Some(FORMAT_VERSION));
        store
    }

    #[test]
    fn public_writes_to_a_reserved_collection_are_refused() {
        let store = FerroStore::in_memory();
        assert_reserved(
            store
                .put_doc("__meta", "schema_version", &json!(1))
                .unwrap_err(),
        );
        assert_reserved(store.delete_doc("__meta", "schema_version").unwrap_err());
        assert_reserved(store.put_doc("__other", "x", &json!(1)).unwrap_err());
        assert_eq!(store.format_version().unwrap(), Some(FORMAT_WITH_WORKFLOW));
        assert_eq!(store.read_state().unwrap().last_seq, 0, "nothing committed");
        // One underscore is an ordinary collection.
        store.put_doc("_meta", "x", &json!(1)).unwrap();
    }

    #[test]
    fn writes_to_a_reserved_collection_over_the_socket_are_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let sock = tmp.path().join("store.sock");
        let server =
            super::super::FerroStoreServer::bind_unix(&sock, FerroStore::in_memory()).unwrap();
        let client = super::super::server::connect_unix(&sock.display().to_string()).unwrap();
        for (method, params) in [
            (
                "put_doc",
                json!({"collection": "__meta", "id": "schema_version", "doc": 1}),
            ),
            (
                "put_doc",
                json!({"collection": "__meta", "id": "x", "doc": 1, "if_absent": true}),
            ),
            (
                "delete_doc",
                json!({"collection": "__meta", "id": "schema_version"}),
            ),
        ] {
            assert_reserved(client.call(method, params).unwrap_err());
        }
        let remote = FerroStore::connect_uri(&format!("unix://{}", sock.display())).unwrap();
        assert_reserved(remote.put_doc("__meta", "x", &json!(1)).unwrap_err());
        assert_eq!(remote.format_version().unwrap(), Some(FORMAT_WITH_WORKFLOW));
        drop(remote);
        drop(client);
        drop(server);
    }

    /// An in-memory store at seq 0 whose schema version is `version` (`None`:
    /// no version document, a store from before schema versioning).
    fn at_schema(version: Option<u64>) -> FerroStore {
        let mut state = DocState::default();
        if let Some(version) = version {
            state.docs.insert(
                "__meta".to_string(),
                "schema_version".to_string(),
                json!(version).into(),
            );
        }
        FerroStore::assemble(
            Inner::Memory(Mutex::new(state.into())),
            "mem://".to_string(),
        )
    }

    fn put(collection: &str, id: &str, doc: Value) -> SyncOp {
        SyncOp::PutDoc {
            collection: collection.into(),
            id: id.into(),
            doc,
        }
    }

    fn schema_put(doc: Value) -> SyncOp {
        put("__meta", "schema_version", doc)
    }

    /// `ops` as batches 1, 2, … for `store` to pull.
    fn pulled(recorder: &Recorder, ops: Vec<Vec<SyncOp>>) {
        *recorder.pull.lock().unwrap() = ops
            .into_iter()
            .zip(1..)
            .map(|(ops, seq)| SyncBatch { seq, ops })
            .collect();
    }

    fn has(store: &FerroStore, collection: &str, id: &str) -> bool {
        store.read_state().unwrap().docs.contains(collection, id)
    }

    #[test]
    fn a_raise_is_sent_with_the_commit_that_made_it() {
        let tmp = tempfile::tempdir().unwrap();
        let (store, recorder) = attached(migrated(tmp.path()), SyncMode::WriteThrough);
        let project = store.add_project("/p").unwrap();
        store.set_routes(&project, &RoutingMap::starting()).unwrap();
        assert_eq!(store.format_version().unwrap(), Some(FORMAT_WITH_ROUTING));
        let commits = recorder.commits.lock().unwrap().clone();
        assert_eq!(commits.len(), 2, "{commits:?}");
        let last = commits.last().unwrap();
        assert!(
            last.ops.contains(&schema_put(json!(FORMAT_WITH_ROUTING))),
            "{last:?}"
        );
        assert!(!store.has_pending_sync());
        assert_eq!(store.synced_seq(), store.read_state().unwrap().last_seq);
    }

    #[test]
    fn a_reserved_only_commit_is_sent_and_a_replica_pulls_past_it() {
        let (source, recorder) = attached(at_schema(Some(FORMAT_VERSION)), SyncMode::WriteThrough);
        source.put_doc("docs", "a", &json!(1)).unwrap();
        // The socket's `commit_ops` path: a commit of a reserved op only.
        let raise = Op::Put {
            collection: "__meta".into(),
            id: "schema_version".into(),
            doc: json!(FORMAT_WITH_ROUTING),
        };
        assert!(
            source
                .commit_if_seq(1, std::slice::from_ref(&raise))
                .unwrap()
        );
        source.put_doc("docs", "b", &json!(2)).unwrap();
        let sent = recorder.commits.lock().unwrap().clone();
        assert_eq!(
            sent.iter().map(|b| b.seq).collect::<Vec<_>>(),
            vec![1, 2, 3],
            "every commit is sent, so the sequence has no gap"
        );
        assert_eq!(sent[1].ops, vec![SyncOp::from_op(&raise)]);
        assert_eq!(source.synced_seq(), 3);

        let (replica, inbound) = attached(at_schema(Some(FORMAT_VERSION)), SyncMode::WriteBehind);
        *inbound.pull.lock().unwrap() = sent;
        assert_eq!(replica.sync_pull().unwrap(), 3);
        assert_eq!(replica.read_state().unwrap().last_seq, 3);
        assert_eq!(
            replica.format_version().unwrap(),
            source.format_version().unwrap()
        );
        assert!(has(&replica, "docs", "a") && has(&replica, "docs", "b"));
    }

    #[test]
    fn a_migration_into_an_attached_store_does_not_wedge_its_replicas() {
        let tmp = tempfile::tempdir().unwrap();
        let (source, recorder) = attached(FerroStore::in_memory(), SyncMode::WriteThrough);
        let redb = crate::RedbStore::open(&tmp.path().join("src.redb")).unwrap();
        crate::migrate_redb_to_ferrostore(&redb, &source).unwrap();
        source.put_doc("docs", "a", &json!(1)).unwrap();
        let sent = recorder.commits.lock().unwrap().clone();
        assert_eq!(sent.iter().map(|b| b.seq).collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(sent[0].ops, vec![schema_put(json!(FORMAT_VERSION))]);

        let (replica, inbound) = attached(FerroStore::in_memory(), SyncMode::WriteBehind);
        *inbound.pull.lock().unwrap() = sent;
        assert_eq!(replica.sync_pull().unwrap(), 2);
        // An older version never lowers the replica's.
        assert_eq!(
            replica.format_version().unwrap(),
            Some(FORMAT_WITH_WORKFLOW)
        );
        assert!(has(&replica, "docs", "a"));
    }

    #[test]
    fn an_inbound_older_schema_version_is_dropped() {
        let (store, recorder) = attached(at_schema(Some(FORMAT_WITH_STATE)), SyncMode::WriteBehind);
        pulled(
            &recorder,
            vec![
                vec![schema_put(json!(FORMAT_VERSION))],
                vec![schema_put(json!(FORMAT_WITH_STATE))],
                vec![
                    put("docs", "a", json!(1)),
                    schema_put(json!(FORMAT_WITH_ROUTING)),
                ],
            ],
        );
        assert_eq!(store.sync_pull().unwrap(), 3);
        assert_eq!(store.read_state().unwrap().last_seq, 3);
        assert_eq!(store.format_version().unwrap(), Some(FORMAT_WITH_STATE));
        assert!(has(&store, "docs", "a"), "the batch's other op applies");
    }

    #[test]
    fn an_inbound_schema_version_off_the_ladder_is_dropped() {
        // No version document, so any version on the ladder would raise it.
        let (store, recorder) = attached(at_schema(None), SyncMode::WriteBehind);
        pulled(
            &recorder,
            vec![
                vec![schema_put(json!(FORMAT_VERSION - 1))],
                vec![schema_put(json!(0))],
                vec![schema_put(json!(FORMAT_WITH_ROUTING.to_string()))],
                vec![schema_put(json!(4.5))],
                vec![schema_put(json!(-1))],
                vec![schema_put(Value::Null)],
            ],
        );
        assert_eq!(store.sync_pull().unwrap(), 6);
        assert_eq!(store.read_state().unwrap().last_seq, 6);
        assert_eq!(store.format_version().unwrap(), None);
        assert!(!has(&store, "__meta", "schema_version"));
    }

    #[test]
    fn an_inbound_schema_newer_than_this_build_is_refused() {
        let (store, recorder) = attached(FerroStore::in_memory(), SyncMode::WriteBehind);
        let newer = FORMAT_WITH_WORKFLOW + 1;
        pulled(
            &recorder,
            vec![
                vec![put("docs", "a", json!(1))],
                vec![put("docs", "b", json!(2)), schema_put(json!(newer))],
            ],
        );
        for err in [
            store.sync_pull().unwrap_err(),
            store.sync_prewarm(&["docs"]).unwrap_err(),
        ] {
            assert!(
                matches!(
                    err,
                    StoreError::FormatVersion { found: Some(v), oldest: FORMAT_VERSION, newest: FORMAT_WITH_WORKFLOW }
                        if v == newer
                ),
                "{err:?}"
            );
        }
        assert_eq!(store.read_state().unwrap().last_seq, 0, "nothing applied");
        assert_eq!(store.format_version().unwrap(), Some(FORMAT_WITH_WORKFLOW));
        assert!(!has(&store, "docs", "a") && !has(&store, "docs", "b"));
    }

    #[test]
    fn other_inbound_reserved_ops_are_dropped_and_the_sequence_advances() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("replica.fls");
        let ops = vec![
            vec![put("__meta", "other", json!(1))],
            vec![SyncOp::DeleteDoc {
                collection: "__meta".into(),
                id: "schema_version".into(),
            }],
            vec![put("__other", "x", json!(1))],
            vec![
                SyncOp::DeleteDoc {
                    collection: "__other".into(),
                    id: "x".into(),
                },
                put("docs", "a", json!(1)),
            ],
        ];
        {
            let (store, recorder) =
                attached(FerroStore::open(&path).unwrap(), SyncMode::WriteBehind);
            pulled(&recorder, ops.clone());
            assert_eq!(store.sync_pull().unwrap(), 4);
        }
        // Reopened: the dropped batches are empty frames, so replay agrees.
        for store in [FerroStore::open(&path).unwrap(), {
            let (store, recorder) = attached(FerroStore::in_memory(), SyncMode::WriteBehind);
            pulled(&recorder, ops);
            store.sync_pull().unwrap();
            store
        }] {
            let state = store.read_state().unwrap();
            assert_eq!(state.last_seq, 4);
            assert_eq!(store.format_version().unwrap(), Some(FORMAT_WITH_WORKFLOW));
            let reserved: Vec<_> = state
                .docs
                .iter()
                .filter(|(collection, _, _)| collection.starts_with("__"))
                .map(|(collection, id, _)| (collection.as_str(), id.as_str()))
                .collect();
            assert_eq!(reserved, vec![("__meta", "schema_version")]);
            assert!(has(&store, "docs", "a"));
        }
    }

    #[test]
    fn a_loud_apply_relays_the_batches_as_they_came_in() {
        let (store, recorder) = attached(FerroStore::in_memory(), SyncMode::WriteThrough);
        let batches = vec![
            SyncBatch {
                seq: 1,
                ops: vec![schema_put(json!(FORMAT_VERSION))],
            },
            SyncBatch {
                seq: 2,
                ops: vec![put("docs", "a", json!(1))],
            },
        ];
        assert_eq!(store.apply_sync_batches(batches.clone(), false).unwrap(), 2);
        assert_eq!(*recorder.commits.lock().unwrap(), batches);
        assert_eq!(store.format_version().unwrap(), Some(FORMAT_WITH_WORKFLOW));
    }

    #[test]
    fn a_reserved_document_is_never_filled_from_the_adapter() {
        let (store, recorder) = attached(FerroStore::in_memory(), SyncMode::LocalCache);
        assert_eq!(store.get_doc("__meta", "missing").unwrap(), None);
        assert_eq!(recorder.fetches.load(Ordering::SeqCst), 0);
        assert_eq!(store.get_doc("docs", "missing").unwrap(), Some(json!(1)));
        assert_eq!(recorder.fetches.load(Ordering::SeqCst), 1);
    }
}
