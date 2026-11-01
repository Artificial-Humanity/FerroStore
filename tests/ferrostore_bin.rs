//! The `ferrostore` binary, run as a person runs it: in a sandbox with its own
//! HOME, XDG directories and TMPDIR, with `FL_DB` and `FL_STORE_BACKEND`
//! removed from the environment it inherits.
//!
//! The commands it shares with `fl store` (status, checkpoint, compact,
//! migrate, verify) are compared byte for byte with the `fl store` goldens
//! (`tests/golden/store/`, vendored from FerroLoop), after the program's name
//! and the place its `.redb` path comes from are substituted: the two programs
//! print the same thing.

use fl_core::Catalog;
use fl_store::{
    BackendKind, ColumnDef, ColumnType, ColumnValue, FerroStore, StoreOptions, VectorMetric,
    VectorStorage,
};
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};

struct Sandbox {
    root: tempfile::TempDir,
}

impl Sandbox {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        for sub in ["home", "config", "data", "run", "tmp", "stores"] {
            std::fs::create_dir(root.path().join(sub)).unwrap();
        }
        Self { root }
    }

    fn dir(&self, sub: &str) -> PathBuf {
        self.root.path().join(sub)
    }

    fn redb(&self) -> PathBuf {
        self.dir("stores").join("fl.redb")
    }

    fn fls(&self) -> PathBuf {
        self.dir("stores").join("fl.fls")
    }

    /// `ferrostore <args>` in the sandbox, no engine in the environment.
    fn cmd(&self, args: &[&str]) -> Command {
        let mut c = Command::new(env!("CARGO_BIN_EXE_ferrostore"));
        c.env("HOME", self.dir("home"))
            .env("XDG_CONFIG_HOME", self.dir("config"))
            .env("XDG_DATA_HOME", self.dir("data"))
            .env("XDG_RUNTIME_DIR", self.dir("run"))
            .env("TMPDIR", self.dir("tmp"))
            .env_remove("FL_DB")
            .env_remove("FL_STORE_BACKEND")
            .current_dir(self.dir("home"))
            .args(args);
        c
    }

    fn run(&self, args: &[&str]) -> Output {
        self.cmd(args).output().unwrap()
    }

    /// Run and require exit 0; stdout.
    fn ok(&self, args: &[&str]) -> String {
        let out = self.run(args);
        assert!(
            out.status.success(),
            "ferrostore {args:?}: {}\n{}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap()
    }

    fn json(&self, args: &[&str]) -> Value {
        serde_json::from_str(&self.ok(args)).unwrap()
    }

    /// The files of `stores/`, sorted.
    fn stores(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(self.dir("stores"))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

// ---- fixtures: the same as the `fl store` goldens' ----

fn redb_store(path: &Path) {
    let store = fl_store::RedbStore::open(path).unwrap();
    store.add_project("/golden/project-a").unwrap();
    store.add_project("/golden/project-b").unwrap();
}

fn ferro_store(path: &Path) {
    let store = FerroStore::open(path).unwrap();
    for round in 0..3 {
        for i in 0..10 {
            store
                .put_doc(
                    "docs",
                    &format!("r{round}-{i:02}"),
                    &serde_json::json!({"round": round, "i": i}),
                )
                .unwrap();
        }
        store.checkpoint().unwrap();
    }
}

fn migrated_pair(sb: &Sandbox) {
    redb_store(&sb.redb());
    assert!(
        fl_store::migrate_locked(&sb.redb(), &sb.fls())
            .unwrap()
            .is_clean()
    );
}

fn mismatched_pair(sb: &Sandbox) {
    migrated_pair(sb);
    let ferro = fl_store::open_store_with(
        &sb.redb(),
        &StoreOptions {
            backend: Some(BackendKind::Ferro),
            ..StoreOptions::default()
        },
    )
    .unwrap();
    ferro.add_project("/golden/only-in-ferro").unwrap();
}

// ---- the normalisation of the `fl store` goldens ----

fn time_after(line: &str, prefix: &str, until: &str) -> Option<String> {
    let at = line.find(prefix)? + prefix.len();
    let rest = &line[at..];
    let end = rest.find(until)?;
    let value = &rest[..end];
    if value.is_empty() || !value.chars().all(|c| c.is_ascii_digit() || c == '.') {
        return None;
    }
    Some(format!("{}<T>{}", &line[..at], &rest[end..]))
}

fn is_hash(s: &str, len: usize) -> bool {
    s.len() == len
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn uuids(line: &str) -> String {
    const URN: &str = "urn:uuid:";
    let mut out = String::new();
    let mut rest = line;
    while let Some(at) = rest.find(URN) {
        let start = at + URN.len();
        out.push_str(&rest[..start]);
        rest = &rest[start..];
        let candidate = rest.get(..36).unwrap_or("");
        let well_formed = candidate.len() == 36
            && candidate.bytes().enumerate().all(|(i, b)| {
                if [8, 13, 18, 23].contains(&i) {
                    b == b'-'
                } else {
                    b.is_ascii_digit() || (b'a'..=b'f').contains(&b)
                }
            });
        if well_formed {
            out.push_str("<UUID>");
            rest = &rest[36..];
        }
    }
    out.push_str(rest);
    out
}

#[derive(Default)]
struct Hashes {
    full: Vec<String>,
    short: Vec<String>,
}

impl Hashes {
    fn placeholder(&mut self, hash: &str) -> String {
        let (seen, label) = if hash.len() == 64 {
            (&mut self.full, "HASH")
        } else {
            (&mut self.short, "HASH12")
        };
        let n = seen.iter().position(|h| h == hash).unwrap_or_else(|| {
            seen.push(hash.to_string());
            seen.len() - 1
        });
        format!("<{label}#{}>", n + 1)
    }
}

fn normalise_line(line: &str, hashes: &mut Hashes) -> String {
    let line = &uuids(line);
    if let Some(l) = time_after(line, "time: ", "s\n") {
        return l;
    }
    if line.trim_start().starts_with("\"elapsed_ms\": ") {
        let end = if line.ends_with(",\n") { ",\n" } else { "\n" };
        if let Some(l) = time_after(line, "\"elapsed_ms\": ", end) {
            return l;
        }
    }
    for key in ["\"redb_hash\": \"", "\"ferro_hash\": \""] {
        if let Some(at) = line.find(key) {
            let start = at + key.len();
            if let Some(len) = line[start..].find('"')
                && is_hash(&line[start..start + len], 64)
            {
                return format!(
                    "{}{}{}",
                    &line[..start],
                    hashes.placeholder(&line[start..start + len]),
                    &line[start + len..]
                );
            }
        }
    }
    if line.starts_with("  ") && line.contains(": redb ") && line.contains(", ferro ") {
        let words: Vec<String> = line
            .trim_end_matches('\n')
            .split(' ')
            .map(|w| {
                let bare = w.trim_end_matches(',');
                if is_hash(bare, 12) {
                    w.replacen(bare, &hashes.placeholder(bare), 1)
                } else {
                    w.to_string()
                }
            })
            .collect();
        return format!("{}\n", words.join(" "));
    }
    line.to_string()
}

impl Sandbox {
    fn normalise(&self, text: &str) -> String {
        let text = text.replace(&self.root.path().display().to_string(), "<DIR>");
        let mut hashes = Hashes::default();
        text.split_inclusive('\n')
            .map(|l| normalise_line(l, &mut hashes))
            .collect()
    }

    /// `ferrostore <args>` printed what `fl store` printed in the golden
    /// `name`, the program's name and its `.redb` hint substituted. The
    /// golden's first line (the command shown) is not compared.
    fn same_as_fl(&self, name: &str, mut cmd: Command) {
        let out = cmd.output().unwrap();
        let got = format!(
            "exit: {}\n--- stdout\n{}--- stderr\n{}",
            out.status
                .code()
                .map_or_else(|| "signal".to_string(), |c| c.to_string()),
            self.normalise(&String::from_utf8_lossy(&out.stdout)),
            self.normalise(&String::from_utf8_lossy(&out.stderr)),
        );
        // The goldens are a snapshot of `fl store`'s output from FerroLoop,
        // vendored so `ferrostore` keeps printing exactly what `fl store` prints.
        let file = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/golden/store")
            .join(format!("{name}.txt"));
        let golden =
            std::fs::read_to_string(&file).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
        let (_, body) = golden.split_once('\n').unwrap();
        let want = body.replace("fl store", "ferrostore").replace(
            "the bound store, or --db <file>.redb",
            "--store <file>.redb",
        );
        assert_eq!(got, want, "ferrostore differs from fl store in {name}");
    }

    /// `ferrostore <cmd> --store <redb> <rest>`.
    fn on_store(&self, cmd: &str, rest: &[&str]) -> Command {
        let redb = self.redb();
        let mut args = vec![cmd, "--store", s(&redb)];
        args.extend_from_slice(rest);
        self.cmd(&args)
    }

    fn on_redb(&self, cmd: &str, rest: &[&str]) -> Command {
        let mut rest = rest.to_vec();
        rest.extend(["--engine", "redb"]);
        self.on_store(cmd, &rest)
    }
}

// ---- the commands shared with `fl store`: the same output ----

#[test]
fn status_prints_what_fl_store_prints() {
    let sb = Sandbox::new();
    sb.same_as_fl("status-redb-missing", sb.on_redb("status", &[]));
    sb.same_as_fl(
        "status-redb-missing-json",
        sb.on_redb("status", &["--json"]),
    );
    sb.same_as_fl("status-ferro-missing", sb.on_store("status", &[]));
    redb_store(&sb.redb());
    sb.same_as_fl("status-redb", sb.on_redb("status", &[]));
    sb.same_as_fl("status-redb-json", sb.on_redb("status", &["--json"]));
    sb.same_as_fl("status-redb-beside-ferro", sb.on_store("status", &[]));
    sb.same_as_fl(
        "status-redb-beside-ferro-json",
        sb.on_store("status", &["--json"]),
    );

    let sb = Sandbox::new();
    ferro_store(&sb.fls());
    sb.same_as_fl("status-ferro", sb.on_store("status", &[]));
    sb.same_as_fl("status-ferro-json", sb.on_store("status", &["--json"]));
    // The environment is never read: an engine named there changes nothing.
    let mut cmd = sb.on_store("status", &[]);
    cmd.env("FL_STORE_BACKEND", "redb");
    sb.same_as_fl("status-ferro", cmd);
}

#[test]
fn checkpoint_and_compact_print_what_fl_store_prints() {
    let sb = Sandbox::new();
    sb.same_as_fl("checkpoint-ferro-missing", sb.on_store("checkpoint", &[]));
    sb.same_as_fl("compact-ferro-missing", sb.on_store("compact", &[]));
    assert!(sb.stores().is_empty(), "{:?}", sb.stores());
    redb_store(&sb.redb());
    sb.same_as_fl("checkpoint-redb", sb.on_redb("checkpoint", &[]));
    sb.same_as_fl("compact-redb", sb.on_redb("compact", &[]));
    sb.same_as_fl("compact-redb-json", sb.on_redb("compact", &["--json"]));

    let sb = Sandbox::new();
    ferro_store(&sb.fls());
    sb.same_as_fl("checkpoint-ferro", sb.on_store("checkpoint", &[]));
    let sb = Sandbox::new();
    ferro_store(&sb.fls());
    sb.same_as_fl("compact-ferro", sb.on_store("compact", &[]));
    let sb = Sandbox::new();
    ferro_store(&sb.fls());
    sb.same_as_fl("compact-ferro-json", sb.on_store("compact", &["--json"]));
}

#[test]
fn migrate_prints_what_fl_store_prints() {
    let sb = Sandbox::new();
    sb.same_as_fl("migrate-no-redb", sb.on_store("migrate", &[]));
    let fls = sb.fls();
    sb.same_as_fl("migrate-fls-path", sb.cmd(&["migrate", "--store", s(&fls)]));
    redb_store(&sb.redb());
    sb.same_as_fl("migrate-dry-run", sb.on_store("migrate", &["--dry-run"]));
    sb.same_as_fl(
        "migrate-dry-run-json",
        sb.on_store("migrate", &["--dry-run", "--json"]),
    );
    // The dry run's private copy was made beside the store, not in $TMPDIR,
    // and is gone.
    assert_eq!(sb.stores(), ["fl.redb"]);
    assert_eq!(std::fs::read_dir(sb.dir("tmp")).unwrap().count(), 0);
    sb.same_as_fl("migrate", sb.on_store("migrate", &[]));
    sb.same_as_fl("migrate-already-migrated", sb.on_store("migrate", &[]));

    let sb = Sandbox::new();
    redb_store(&sb.redb());
    sb.same_as_fl("migrate-json", sb.on_store("migrate", &["--json"]));
}

#[test]
fn verify_prints_what_fl_store_prints() {
    let sb = Sandbox::new();
    redb_store(&sb.redb());
    sb.same_as_fl("verify-no-fls", sb.on_store("verify", &[]));
    let fls = sb.fls();
    sb.same_as_fl("verify-fls-path", sb.cmd(&["verify", "--store", s(&fls)]));

    let sb = Sandbox::new();
    migrated_pair(&sb);
    sb.same_as_fl("verify", sb.on_store("verify", &[]));
    sb.same_as_fl("verify-json", sb.on_store("verify", &["--json"]));
    let files = sb.stores();
    assert!(
        files.contains(&"fl.fls".to_string()) && !files.iter().any(|f| f.contains("scratch")),
        "the private copy was left behind: {files:?}"
    );
    assert_eq!(std::fs::read_dir(sb.dir("tmp")).unwrap().count(), 0);

    let sb = Sandbox::new();
    mismatched_pair(&sb);
    sb.same_as_fl("verify-mismatch", sb.on_store("verify", &[]));
    sb.same_as_fl("verify-mismatch-json", sb.on_store("verify", &["--json"]));
}

// ---- get, put and query ----

#[test]
fn get_put_and_query_round_trip_on_an_embedded_store() {
    let sb = Sandbox::new();
    let fls = sb.fls();
    let store = s(&fls);
    // Nothing is there: get refuses and creates nothing, and so does put
    // without --create; put --create creates it.
    let out = sb.run(&["get", "--store", store, "docs", "a"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("does not create one"));
    assert!(!fls.exists());
    let out = sb.run(&["put", "--store", store, "docs", "a", "1"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("only with --create"));
    assert!(!fls.exists());

    let put = sb.json(&[
        "put",
        "--create",
        "--store",
        store,
        "docs",
        "a",
        r#"{"kind": "fox", "text": "the quick red fox"}"#,
    ]);
    assert_eq!(
        put,
        serde_json::json!({"collection": "docs", "id": "a", "put": true})
    );
    // From stdin.
    let mut child = sb
        .cmd(&["put", "--store", store, "docs", "b", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(br#"{"kind": "dog", "text": "a lazy brown dog"}"#)
        .unwrap();
    assert!(child.wait_with_output().unwrap().status.success());
    assert_eq!(
        std::fs::metadata(&fls).unwrap().permissions().mode() & 0o777,
        0o600
    );

    assert_eq!(
        sb.json(&["get", "--store", store, "docs", "b"])["text"],
        "a lazy brown dog"
    );
    let missing = sb.run(&["get", "--store", store, "docs", "zzz"]);
    assert_eq!(missing.status.code(), Some(1));
    assert!(missing.stdout.is_empty());

    let field = sb.json(&["query", "--store", store, "docs", "kind", "fox"]);
    assert_eq!(field["mode"], "field");
    assert_eq!(field["ids"], serde_json::json!(["a"]));
    let bm25 = sb.json(&[
        "query",
        "--bm25",
        "--store",
        store,
        "docs",
        "brown dog",
        "-k",
        "1",
    ]);
    assert_eq!(bm25["mode"], "bm25");
    assert_eq!(bm25["hits"][0]["id"], "b");
    assert_eq!(bm25["hits"].as_array().unwrap().len(), 1);

    {
        let handle = FerroStore::open(&fls).unwrap();
        handle
            .create_vector_collection("vecs", 2, VectorStorage::F32, VectorMetric::L2, 4, 8, 8)
            .unwrap();
        handle
            .upsert_vector("vecs", "near", &[1.0, 1.0], None)
            .unwrap();
        handle
            .upsert_vector("vecs", "far", &[9.0, -9.0], None)
            .unwrap();
    }
    let vector = sb.json(&[
        "query", "--vector", "--store", store, "vecs", "--values", "[1, 0.9]", "-k", "1",
    ]);
    assert_eq!(vector["mode"], "vector");
    assert_eq!(vector["hits"][0]["id"], "near");

    // A reserved collection is refused, and a lone `.redb` is not migrated.
    for args in [
        vec!["get", "--store", store, "__meta", "schema_version"],
        vec!["put", "--store", store, "__meta", "x", "1"],
        vec!["query", "--store", store, "__meta", "a", "b"],
    ] {
        let out = sb.run(&args);
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        assert!(String::from_utf8_lossy(&out.stderr).contains("reserved"));
    }
    let lone = sb.dir("stores").join("lone.redb");
    redb_store(&lone);
    let out = sb.run(&["put", "--store", s(&lone), "docs", "a", "1"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("ferrostore migrate"));
    assert!(!sb.dir("stores").join("lone.fls").exists());
}

/// A `ferrostore serve` child, killed by PID when the test ends early.
struct Server {
    child: Child,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Server {
    fn start(sb: &Sandbox, args: &[&str]) -> (Self, String) {
        let mut child = sb
            .cmd(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut ready = String::new();
        BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut ready)
            .unwrap();
        (Self { child }, ready)
    }

    /// SIGTERM, then the exit status and stderr.
    fn stop(self) -> (std::process::ExitStatus, String) {
        self.stop_with(&[libc::SIGTERM])
    }

    /// `signals`, one after the other at once, then the exit status and
    /// stderr.
    fn stop_with(mut self, signals: &[libc::c_int]) -> (std::process::ExitStatus, String) {
        let pid = i32::try_from(self.child.id()).unwrap();
        for &signal in signals {
            // SAFETY: `pid` is this test's own child, not yet waited for.
            assert_eq!(unsafe { libc::kill(pid, signal) }, 0);
        }
        let status = self.child.wait().unwrap();
        let mut err = String::new();
        std::io::Read::read_to_string(&mut self.child.stderr.take().unwrap(), &mut err).unwrap();
        (status, err)
    }
}

#[test]
fn get_and_put_over_a_unix_socket_served_by_ferrostore_serve() {
    let sb = Sandbox::new();
    let fls = sb.fls();
    let (server, ready) = Server::start(&sb, &["serve", "--create", "--store", s(&fls)]);
    let socket = sb.dir("stores").join("fl.fls.sock");
    let uri = format!("unix://{}", socket.display());
    assert_eq!(
        ready,
        format!("serving {} on {uri} (auth: peer-uid)\n", fls.display())
    );
    assert_eq!(
        std::fs::metadata(&socket).unwrap().permissions().mode() & 0o777,
        0o600
    );
    sb.ok(&["put", "--store", &uri, "docs", "a", r#"{"over": "unix"}"#]);
    assert_eq!(
        sb.json(&["get", "--store", &uri, "docs", "a"])["over"],
        "unix"
    );
    assert_eq!(
        sb.json(&["query", "--store", &uri, "docs", "over", "unix"])["ids"],
        serde_json::json!(["a"])
    );
    let (status, err) = server.stop();
    assert_eq!(status.code(), Some(0), "{err}");
    assert_eq!(err, "stopped on SIGTERM\n");
    assert!(!socket.exists(), "the socket outlived the server");
    // The write is in the file.
    assert_eq!(
        FerroStore::open(&fls)
            .unwrap()
            .get_doc("docs", "a")
            .unwrap(),
        Some(serde_json::json!({"over": "unix"}))
    );
}

fn psk_file(sb: &Sandbox) -> PathBuf {
    let path = sb.dir("home").join("psk");
    std::fs::write(&path, format!("{}\n", "5a".repeat(32))).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    path
}

#[test]
fn get_and_put_over_ws_on_loopback_with_a_psk() {
    let sb = Sandbox::new();
    let fls = sb.fls();
    let psk = psk_file(&sb);
    // TCP without authentication is refused before anything binds.
    let out = sb.run(&["serve", "--store", s(&fls), "--tcp", "127.0.0.1:0"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("needs authentication"));

    let (server, ready) = Server::start(
        &sb,
        &[
            "serve",
            "--create",
            "--store",
            s(&fls),
            "--tcp",
            "127.0.0.1:0",
            "--psk-file",
            s(&psk),
            "--server-id",
            "lab",
            "--json",
        ],
    );
    let ready: Value = serde_json::from_str(&ready).unwrap();
    assert_eq!(ready["auth"], serde_json::json!(["psk"]));
    assert_eq!(ready["tls"], false);
    let uri = ready["uri"].as_str().unwrap().to_string();
    assert!(uri.starts_with("ws://127.0.0.1:"), "{uri}");
    assert!(
        !ready.to_string().contains(&"5a".repeat(32)),
        "the PSK was printed"
    );

    let creds = ["--psk-file", s(&psk), "--server-id", "lab"];
    let mut put = vec!["put", "--store", &uri, "docs", "a", r#"{"over": "ws"}"#];
    put.extend(creds);
    sb.ok(&put);
    let mut get = vec!["get", "--store", &uri, "docs", "a"];
    get.extend(creds);
    assert_eq!(sb.json(&get)["over"], "ws");
    // No credential, or the wrong server id: refused.
    let out = sb.run(&["get", "--store", &uri, "docs", "a"]);
    assert_eq!(out.status.code(), Some(2));
    let out = sb.run(&[
        "get",
        "--store",
        &uri,
        "--psk-file",
        s(&psk),
        "--server-id",
        "other",
        "docs",
        "a",
    ]);
    assert_eq!(out.status.code(), Some(2));
    let (status, err) = server.stop();
    assert_eq!(status.code(), Some(0), "{err}");
}

// ---- inspect ----

/// A store with documents, a column table and a vector collection.
fn mixed_store(path: &Path) {
    let store = FerroStore::open(path).unwrap();
    for i in 0..5 {
        store
            .put_doc("docs", &format!("d{i}"), &serde_json::json!({"i": i}))
            .unwrap();
    }
    store
        .put_doc("notes", "n", &serde_json::json!("one"))
        .unwrap();
    store
        .create_column_table(
            "events",
            &[
                ColumnDef {
                    name: "at".into(),
                    ty: ColumnType::TimestampMs,
                },
                ColumnDef {
                    name: "n".into(),
                    ty: ColumnType::Int64,
                },
            ],
        )
        .unwrap();
    for group in 0..2 {
        let rows: Vec<Vec<ColumnValue>> = (0..3)
            .map(|i| {
                vec![
                    ColumnValue::TimestampMs(1000 + i),
                    ColumnValue::Int64(group * 10 + i),
                ]
            })
            .collect();
        store.append_column_rows("events", &rows).unwrap();
    }
    store
        .create_vector_collection(
            "emb",
            3,
            VectorStorage::F16,
            VectorMetric::Cosine,
            8,
            32,
            16,
        )
        .unwrap();
    for i in 0..4 {
        store
            .upsert_vector("emb", &format!("v{i}"), &[1.0, i as f32, 0.5], None)
            .unwrap();
    }
    store.checkpoint().unwrap();
    store
        .put_doc("docs", "after", &serde_json::json!(1))
        .unwrap();
}

#[test]
fn inspect_reports_the_superblocks_and_every_kind_of_collection() {
    let sb = Sandbox::new();
    mixed_store(&sb.fls());
    let v = sb.json(&["inspect", s(&sb.redb()), "--json"]);
    assert_eq!(v["path"], s(&sb.fls()));
    assert_eq!(v["file_bytes"], std::fs::metadata(sb.fls()).unwrap().len());
    let slots = v["slots"].as_array().unwrap();
    assert_eq!(slots.len(), 2);
    let live: Vec<&Value> = slots.iter().filter(|s| s["live"] == true).collect();
    assert_eq!(live.len(), 1);
    assert_eq!(live[0]["state"], "valid");
    assert_eq!(live[0]["container_version"], 2);
    assert_eq!(live[0]["engine"], 0);
    assert_eq!(live[0]["incompat"], 0);
    let sbk = &live[0]["superblock"];
    assert!(sbk["wal_length"].as_u64().unwrap() > 0, "{sbk}");
    for key in [
        "epoch",
        "snapshot_seq",
        "snapshot_offset",
        "snapshot_length",
        "wal_offset",
    ] {
        assert!(sbk[key].is_u64(), "{key}: {sbk}");
    }
    assert_eq!(v["error"], Value::Null);
    let c = &v["contents"];
    assert_eq!(c["schema_version"], fl_store::FORMAT_WITH_WORKFLOW);
    assert_eq!(c["epoch"], sbk["epoch"]);
    let live_bytes = c["live_bytes"].as_u64().unwrap();
    let dead_bytes = c["dead_bytes"].as_u64().unwrap();
    assert_eq!(live_bytes + dead_bytes, v["file_bytes"].as_u64().unwrap());
    let counts: Vec<(String, u64)> = c["collections"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| {
            (
                c["name"].as_str().unwrap().to_string(),
                c["docs"].as_u64().unwrap(),
            )
        })
        .collect();
    assert!(counts.contains(&("docs".into(), 6)), "{counts:?}");
    assert!(counts.contains(&("notes".into(), 1)), "{counts:?}");
    assert!(counts.contains(&("__meta".into(), 1)), "{counts:?}");
    assert_eq!(
        c["column_tables"],
        serde_json::json!([{
            "name": "events",
            "columns": ["at:timestamp_ms", "n:int64"],
            "row_groups": 2,
            "rows": 6,
        }])
    );
    assert_eq!(
        c["vector_collections"],
        serde_json::json!([{
            "name": "emb",
            "count": 4,
            "dim": 3,
            "storage": "f16",
            "metric": "cosine",
            "m": 8,
            "ef_construction": 32,
            "ef_search": 16,
        }])
    );

    let text = sb.ok(&["inspect", s(&sb.fls())]);
    for want in [
        "schema version:",
        "  docs: 6 docs",
        "  events: 6 rows in 2 row groups (at:timestamp_ms, n:int64)",
        "  emb: 4 vectors, dim 3, f16, cosine, m 8, ef_construction 32, ef_search 16",
        "valid, live, container version 2, engine 0, incompat 0x0",
    ] {
        assert!(text.contains(want), "missing {want:?} in\n{text}");
    }
}

/// Size, mode, mtime and bytes of a file.
fn fingerprint(path: &Path) -> (u64, u32, i64, i64, Vec<u8>) {
    let m = std::fs::metadata(path).unwrap();
    (
        m.len(),
        m.mode(),
        m.mtime(),
        m.mtime_nsec(),
        std::fs::read(path).unwrap(),
    )
}

#[test]
fn inspect_writes_nothing() {
    let sb = Sandbox::new();
    mixed_store(&sb.fls());
    std::fs::set_permissions(sb.fls(), std::fs::Permissions::from_mode(0o400)).unwrap();
    let before = fingerprint(&sb.fls());
    let files = sb.stores();
    sb.ok(&["inspect", s(&sb.fls())]);
    sb.ok(&["inspect", s(&sb.fls()), "--json"]);
    assert_eq!(fingerprint(&sb.fls()), before);
    assert_eq!(sb.stores(), files, "inspect made a file");

    // A lone `.redb`: refused, never migrated, its file untouched.
    let sb = Sandbox::new();
    redb_store(&sb.redb());
    let before = fingerprint(&sb.redb());
    let out = sb.run(&["inspect", s(&sb.redb())]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("holds a redb store"));
    assert_eq!(fingerprint(&sb.redb()), before);
    assert_eq!(sb.stores(), ["fl.redb"]);

    // Nothing at all: refused, nothing created.
    let sb = Sandbox::new();
    let out = sb.run(&["inspect", s(&sb.fls())]);
    assert_eq!(out.status.code(), Some(2));
    assert!(sb.stores().is_empty());

    // A file that is not a store: its slots are reported, exit 1, untouched.
    let sb = Sandbox::new();
    std::fs::write(sb.fls(), vec![7u8; 9000]).unwrap();
    let before = fingerprint(&sb.fls());
    let out = sb.run(&["inspect", s(&sb.fls()), "--json"]);
    assert_eq!(out.status.code(), Some(1));
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["slots"][0]["state"], "invalid");
    assert_eq!(v["contents"], Value::Null);
    assert!(v["error"].as_str().unwrap().contains("superblock"));
    assert_eq!(fingerprint(&sb.fls()), before);
}

// ---- bench ----

#[test]
fn bench_runs_on_fresh_stores_and_reports_each_engine_and_workload() {
    let sb = Sandbox::new();
    let dir = sb.dir("tmp");
    let v = sb.json(&["bench", "--n", "3", "--json", "--dir", s(&dir)]);
    assert_eq!(v["command"], "bench");
    assert_eq!(v["n"], 3);
    assert_eq!(v["target"], "fresh");
    let results = v["results"].as_array().unwrap();
    let ran: Vec<(String, String)> = results
        .iter()
        .map(|r| {
            (
                r["engine"].as_str().unwrap().to_string(),
                r["workload"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    let want: Vec<(String, String)> = [
        ("ferro", "put"),
        ("ferro", "get"),
        ("ferro", "scan"),
        ("ferro", "vector"),
        ("redb", "put"),
        ("redb", "get"),
        ("redb", "scan"),
    ]
    .iter()
    .map(|(e, w)| (e.to_string(), w.to_string()))
    .collect();
    assert_eq!(ran, want);
    for r in results {
        assert_eq!(r["ops"], 3, "{r}");
        for key in ["p50_us", "p99_us", "mean_us", "total_ms", "ops_per_sec"] {
            assert!(r[key].is_number(), "{key}: {r}");
        }
        assert!(r["p50_us"].as_f64().unwrap() <= r["p99_us"].as_f64().unwrap());
    }
    assert_eq!(
        v["skipped"],
        serde_json::json!([{
            "engine": "redb",
            "workload": "vector",
            "reason": "the engine has no vector search",
        }])
    );
    // The fresh stores are gone.
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);

    let text = sb.ok(&[
        "bench",
        "--n",
        "2",
        "--workload",
        "get",
        "--engines",
        "redb",
        "--dir",
        s(&dir),
    ]);
    assert!(text.starts_with("bench: n = 2, on fresh\n"), "{text}");
    assert!(text.contains("redb    get"), "{text}");
}

#[test]
fn bench_touches_an_existing_store_only_when_told_twice() {
    let sb = Sandbox::new();
    ferro_store(&sb.fls());
    let before = fingerprint(&sb.fls());
    let out = sb.run(&["bench", "--n", "2", "--path", s(&sb.fls())]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--i-know-it-writes"));
    assert_eq!(fingerprint(&sb.fls()), before);

    let v = sb.json(&[
        "bench",
        "--n",
        "2",
        "--path",
        s(&sb.fls()),
        "--i-know-it-writes",
        "--engines",
        "ferro",
        "--workload",
        "put,get,scan",
        "--json",
    ]);
    assert_eq!(v["target"], s(&sb.fls()));
    assert_eq!(v["results"].as_array().unwrap().len(), 3);
    // What it wrote is gone again; the store's own documents are not.
    let inspected = sb.json(&["inspect", s(&sb.fls()), "--json"]);
    let names: Vec<&str> = inspected["contents"]["collections"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["__meta", "docs"]);
}

// ---- what put and serve create ----

/// The pid of a child that has exited and been reaped: no live process.
fn dead_pid() -> u32 {
    let mut child = Command::new("true").spawn().unwrap();
    let pid = child.id();
    child.wait().unwrap();
    // SAFETY: signal 0 sends nothing.
    let rc = unsafe { libc::kill(i32::try_from(pid).unwrap(), 0) };
    assert_eq!(rc, -1, "pid {pid} was reused");
    pid
}

#[test]
fn put_and_serve_never_create_a_fls_beside_a_lone_redb() {
    let sb = Sandbox::new();
    redb_store(&sb.redb());
    let fls = sb.fls();
    let socket = sb.dir("home").join("s.sock");
    // Symlinks to the `.redb`, outside `stores/`: one with the extension and
    // one without. Each names a `.fls` beside itself, which must not appear.
    let link = sb.dir("home").join("link.redb");
    let bare_link = sb.dir("home").join("alias");
    std::os::unix::fs::symlink(sb.redb(), &link).unwrap();
    std::os::unix::fs::symlink(sb.redb(), &bare_link).unwrap();
    let no_ext = sb.dir("stores").join("fl");
    // However the store is named: `fl.fls`, `fl.redb` and `fl` are the same
    // pair, and a symlink to the `.redb` is that redb store.
    for store in [s(&fls), s(&sb.redb()), s(&no_ext), s(&link), s(&bare_link)] {
        for args in [
            vec!["put", "--create", "--store", store, "docs", "a", "1"],
            vec!["put", "--store", store, "docs", "a", "1"],
            vec!["serve", "--create", "--store", store],
            vec![
                "serve",
                "--create",
                "--store",
                store,
                "--socket",
                s(&socket),
            ],
        ] {
            let out = sb.run(&args);
            assert_eq!(out.status.code(), Some(2), "{args:?}");
            let err = String::from_utf8_lossy(&out.stderr);
            assert!(
                err.contains("migrate it first") && err.contains("`ferrostore migrate`"),
                "{args:?}: {err}"
            );
            assert_eq!(sb.stores(), ["fl.redb"], "{args:?}");
            assert!(!socket.exists());
            assert!(!sb.dir("home").join("link.fls").exists(), "{args:?}");
            assert!(!sb.dir("home").join("alias.fls").exists(), "{args:?}");
        }
    }
}

#[test]
fn serve_refuses_plaintext_on_a_non_loopback_address_and_creates_nothing() {
    let sb = Sandbox::new();
    let fls = sb.fls();
    let psk = psk_file(&sb);
    let serve = |create: bool| {
        let mut args = vec!["serve"];
        if create {
            args.push("--create");
        }
        args.extend([
            "--store",
            s(&fls),
            "--tcp",
            "0.0.0.0:0",
            "--psk-file",
            s(&psk),
            "--server-id",
            "lab",
        ]);
        sb.run(&args)
    };
    // --create on a missing store: refused before the store is created, so
    // nothing at all is made in its directory (no `.fls`, no lock file).
    let out = serve(true);
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("refused 0.0.0.0:0 without TLS"), "{err}");
    assert!(sb.stores().is_empty(), "{:?}", sb.stores());
    // An existing store: refused the same, and left as it was.
    ferro_store(&fls);
    let before = sb.stores();
    let out = serve(false);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("without TLS"));
    assert_eq!(sb.stores(), before);
}

#[test]
fn serve_refuses_a_psk_file_others_can_read() {
    let sb = Sandbox::new();
    let fls = sb.fls();
    let psk = psk_file(&sb);
    std::fs::set_permissions(&psk, std::fs::Permissions::from_mode(0o644)).unwrap();
    for tcp in [None, Some("127.0.0.1:0")] {
        let mut args = vec![
            "serve",
            "--create",
            "--store",
            s(&fls),
            "--psk-file",
            s(&psk),
            "--server-id",
            "lab",
        ];
        if let Some(addr) = tcp {
            args.extend(["--tcp", addr]);
        }
        let out = sb.run(&args);
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(
            err.contains("mode 644") && err.contains("chmod 600"),
            "{args:?}: {err}"
        );
        assert!(!err.contains(&"5a".repeat(32)), "the PSK was printed");
        // Refused before the store is created.
        assert!(sb.stores().is_empty(), "{args:?}: {:?}", sb.stores());
    }
}

#[test]
fn put_from_stdin_refuses_a_document_over_64_mib() {
    let sb = Sandbox::new();
    let fls = sb.fls();
    let cap = 64 * 1024 * 1024;
    // A JSON string one byte over the cap, generated here.
    let mut doc = Vec::with_capacity(cap + 1);
    doc.push(b'"');
    doc.resize(cap, b'a');
    doc.push(b'"');
    assert_eq!(doc.len(), cap + 1);
    let mut child = sb
        .cmd(&["put", "--create", "--store", s(&fls), "docs", "big", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    // The reader stops at the cap and exits; a broken pipe is expected.
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&doc);
    });
    let out = child.wait_with_output().unwrap();
    writer.join().unwrap();
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("over the 64 MiB limit of `put -`"), "{err}");
    assert!(!fls.exists(), "the refused put created the store");
}

#[test]
fn put_and_serve_create_no_store_while_it_is_being_migrated() {
    let sb = Sandbox::new();
    let fls = sb.fls();
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(fl_store::MigrationLock::path_for(&fls))
        .unwrap();
    lock.try_lock().unwrap();
    for args in [
        vec!["put", "--create", "--store", s(&fls), "docs", "a", "1"],
        vec!["serve", "--create", "--store", s(&fls)],
    ] {
        let out = sb.run(&args);
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains("is being migrated"), "{args:?}: {err}");
        assert!(!fls.exists(), "{args:?}");
    }
    drop(lock);
    // Another test thread in this binary may be between fork and exec of its
    // own child: that child holds a copy of every open fd (O_CLOEXEC closes
    // them only at exec), so it shares this open file description and keeps
    // the flock alive for a moment after `drop(lock)`. Retry the put for a
    // bounded time; each failed attempt must be the exact refusal.
    let put = ["put", "--create", "--store", s(&fls), "docs", "a", "1"];
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let out = sb.run(&put);
        if out.status.success() {
            break;
        }
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.code() == Some(2) && err.contains("is being migrated"),
            "{put:?}: {}\n{err}",
            out.status
        );
        assert!(
            std::time::Instant::now() < deadline,
            "the lock was still held 5 s after it was dropped: {err}"
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert!(fls.exists());
}

#[test]
fn create_makes_a_store_only_in_a_directory_that_exists() {
    let sb = Sandbox::new();
    let typo = sb.dir("stores").join("typo").join("fl.fls");
    for args in [
        vec!["put", "--create", "--store", s(&typo), "docs", "a", "1"],
        vec!["serve", "--create", "--store", s(&typo)],
    ] {
        let out = sb.run(&args);
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains("only in a directory that exists"), "{err}");
        assert!(!sb.dir("stores").join("typo").exists(), "{args:?}");
    }
    // No --create: a missing store is refused, even in a directory that is.
    let out = sb.run(&["serve", "--store", s(&sb.fls())]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("only with --create"));
    assert!(sb.stores().is_empty());
}

#[test]
fn migrate_and_verify_sweep_the_copies_a_killed_run_left() {
    let sb = Sandbox::new();
    redb_store(&sb.redb());
    let dead = dead_pid();
    let me = std::process::id();
    let stores = sb.dir("stores");
    let stale = stores.join(format!(".fl.redb.scratch-{dead}-1"));
    let live = stores.join(format!(".fl.redb.scratch-{me}-2"));
    let plant = || {
        std::fs::create_dir(&stale).unwrap();
        std::fs::write(stale.join("fl.redb"), b"a copy").unwrap();
    };
    plant();
    std::fs::create_dir(&live).unwrap();
    sb.ok(&["migrate", "--dry-run", "--store", s(&sb.redb())]);
    assert!(
        !stale.exists(),
        "a dead run's copy outlived migrate --dry-run"
    );
    assert!(live.is_dir(), "a live run's copy was removed");
    plant();
    sb.ok(&["migrate", "--store", s(&sb.redb())]);
    assert!(!stale.exists(), "a dead run's copy outlived migrate");
    plant();
    sb.ok(&["verify", "--store", s(&sb.redb())]);
    assert!(!stale.exists(), "a dead run's copy outlived verify");
    assert!(live.is_dir());
}

#[test]
fn a_second_signal_during_shutdown_does_not_kill_serve() {
    let sb = Sandbox::new();
    let fls = sb.fls();
    for signals in [
        [libc::SIGINT, libc::SIGINT, libc::SIGINT],
        [libc::SIGINT, libc::SIGINT, libc::SIGTERM],
    ] {
        let (server, ready) = Server::start(&sb, &["serve", "--create", "--store", s(&fls)]);
        assert!(ready.starts_with("serving "), "{ready}");
        let (status, err) = server.stop_with(&signals);
        assert_eq!(status.code(), Some(0), "{status:?}: {err}");
        assert!(
            err == "stopped on SIGINT\n" || err == "stopped on SIGTERM\n",
            "{err:?}"
        );
    }
}

#[test]
fn bench_path_empties_the_collections_a_killed_run_left() {
    let sb = Sandbox::new();
    ferro_store(&sb.fls());
    let dead = format!("ferrostore-bench-{}-1", dead_pid());
    let live = format!("ferrostore-bench-{}-2", std::process::id());
    {
        let store = FerroStore::open(sb.fls()).unwrap();
        for i in 0..3 {
            store
                .put_doc(&dead, &format!("doc-{i}"), &serde_json::json!({"i": i}))
                .unwrap();
        }
        store
            .put_doc(&live, "doc-0", &serde_json::json!({"i": 0}))
            .unwrap();
    }
    sb.json(&[
        "bench",
        "--n",
        "2",
        "--path",
        s(&sb.fls()),
        "--i-know-it-writes",
        "--engines",
        "ferro",
        "--workload",
        "put",
        "--json",
    ]);
    let inspected = sb.json(&["inspect", s(&sb.fls()), "--json"]);
    let names: Vec<&str> = inspected["contents"]["collections"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["__meta", "docs", live.as_str()]);
}

#[test]
fn get_and_put_over_wss_with_a_psk_and_a_ca_file() {
    let sb = Sandbox::new();
    let fls = sb.fls();
    let tls = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/ferrostore/testdata/tls");
    let key = sb.dir("home").join("server.key");
    std::fs::copy(tls.join("server.key"), &key).unwrap();
    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600)).unwrap();
    let cert = tls.join("server.pem");
    let ca = tls.join("ca.pem");
    let psk = psk_file(&sb);
    let (server, ready) = Server::start(
        &sb,
        &[
            "serve",
            "--create",
            "--store",
            s(&fls),
            "--tcp",
            "127.0.0.1:0",
            "--tls-cert",
            s(&cert),
            "--tls-key",
            s(&key),
            "--psk-file",
            s(&psk),
            "--server-id",
            "lab",
        ],
    );
    // `serving <fls> on wss://127.0.0.1:<port> (auth: psk)`
    let prefix = format!("serving {} on wss://127.0.0.1:", fls.display());
    let port: u16 = ready
        .strip_prefix(&prefix)
        .and_then(|rest| rest.strip_suffix(" (auth: psk)\n"))
        .unwrap_or_else(|| panic!("ready line: {ready:?}"))
        .parse()
        .unwrap();
    let uri = format!("wss://127.0.0.1:{port}");

    let creds = [
        "--ca-file",
        s(&ca),
        "--psk-file",
        s(&psk),
        "--server-id",
        "lab",
    ];
    let mut put = vec!["put", "--store", &uri, "docs", "a", r#"{"over": "wss"}"#];
    put.extend(creds);
    sb.ok(&put);
    let mut get = vec!["get", "--store", &uri, "docs", "a"];
    get.extend(creds);
    assert_eq!(sb.json(&get)["over"], "wss");
    // Without the CA the server's certificate is not trusted.
    let out = sb.run(&[
        "get",
        "--store",
        &uri,
        "--psk-file",
        s(&psk),
        "--server-id",
        "lab",
        "docs",
        "a",
    ]);
    assert_eq!(out.status.code(), Some(2));
    assert!(out.stdout.is_empty());
    let (status, err) = server.stop();
    assert_eq!(status.code(), Some(0), "{err}");
    assert_eq!(
        FerroStore::open(&fls)
            .unwrap()
            .get_doc("docs", "a")
            .unwrap(),
        Some(serde_json::json!({"over": "wss"}))
    );
}
