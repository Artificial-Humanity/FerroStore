//! `get`, `put` and `query`: one document read or written, or one query run,
//! as JSON on stdout, on a store named by a path (embedded) or a URI (a
//! client of a `serve`d store).

use super::{Error, Output, Result, Wording, line, present};
use crate::{BackendKind, ConnectOptions, FerroStore, physical_path};
use serde::Serialize;
use serde::ser::SerializeMap;
use serde_json::Value;
use std::io::Read;
use std::path::{Path, PathBuf};

/// The default number of hits a BM25 or vector query returns.
pub const DEFAULT_K: usize = 10;

/// The store a command works on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    /// A store path: the `.fls` it names (`x.redb` names `x.fls`), opened
    /// embedded.
    Path(PathBuf),
    /// A `unix://`, `ws://` or `wss://` URI of a served store.
    Uri(String),
}

impl Target {
    /// A URI when `raw` starts `unix://`, `ws://` or `wss://`; any other
    /// `scheme://` is refused; anything else is a path.
    pub fn parse(raw: &str) -> Result<Self> {
        if ["unix://", "ws://", "wss://"]
            .iter()
            .any(|scheme| raw.starts_with(scheme))
        {
            return Ok(Self::Uri(raw.to_string()));
        }
        if let Some((scheme, _)) = raw.split_once("://")
            && !scheme.is_empty()
            && scheme
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "+-.".contains(c))
        {
            refuse!(
                "`{raw}` is not a store this command opens: give a path, or a unix://, ws:// \
                 or wss:// URI"
            );
        }
        Ok(Self::Path(PathBuf::from(raw)))
    }
}

/// How a client authenticates to, and checks, a served store: the files and
/// names of [`ConnectOptions`]. Every field is for a URI; a path takes none.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Credentials {
    /// The PSK file (64+ hex digits, mode 0600), with `server_id`.
    pub psk_file: Option<PathBuf>,
    /// The server's name, as it was set with its PSK.
    pub server_id: Option<String>,
    /// A bearer token file (the token on its first line, mode 0600).
    pub token_file: Option<PathBuf>,
    /// `wss://`: trust only the CAs in this PEM file.
    pub ca_file: Option<PathBuf>,
    /// `wss://`: trust the one server key with this SPKI SHA-256.
    pub pin_spki: Option<String>,
    /// Send a credential over plain `ws://` to an address that is not
    /// loopback.
    pub allow_plaintext: bool,
}

impl Credentials {
    fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// The [`ConnectOptions`] these name. A PSK needs its server id, and one
    /// method at most is named.
    pub fn connect_options(&self) -> Result<ConnectOptions> {
        let base = match (&self.psk_file, &self.server_id, &self.token_file) {
            (Some(_), _, Some(_)) => {
                refuse!("--psk-file and --token-file name two credentials: give one")
            }
            (Some(psk), Some(id), None) => ConnectOptions::psk_file(id.clone(), psk.clone()),
            (Some(_), None, None) => {
                refuse!("--psk-file needs --server-id, the name the server was given with its PSK")
            }
            (None, Some(_), _) => refuse!("--server-id goes with --psk-file"),
            (None, None, Some(token)) => ConnectOptions::token_file(token.clone()),
            (None, None, None) => ConnectOptions::new(),
        };
        let mut options = base.allow_plaintext(self.allow_plaintext);
        if let Some(ca) = &self.ca_file {
            options = options.ca_file(ca.clone());
        }
        if let Some(pin) = &self.pin_spki {
            options = options.pin_spki_sha256(pin.clone());
        }
        Ok(options)
    }
}

/// Whether a command may create the `.fls` a path names when nothing is
/// there. Whatever it says, a `.fls` is never created beside a `.redb`
/// (however the path is spelled: `x.fls` and `x.redb` both name that pair),
/// and only while holding the store's migration lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Create {
    /// Never: a missing store is refused (`get`, `query`, `bench --path`).
    Never,
    /// Not without `--create`: a missing store is refused, naming the flag
    /// (`ferrostore put` and `serve` without it).
    NeedsFlag,
    /// A new `.fls` in a directory that exists; a missing parent directory is
    /// refused (`ferrostore put --create`, `serve --create`).
    InExistingDir,
    /// A new `.fls`, and its missing parent directories (mode 0700): `fl
    /// store put` and `serve` on the bound store, as every `fl` command
    /// creates it.
    WithParents,
}

/// Open the FerroStore `target` names. A path is opened embedded, on the
/// ferro engine whatever the environment says: a lone `.redb` is refused
/// (never migrated), and a missing `.fls` is created only as `create` says
/// (else refused, naming `what`, the command). A URI connects with
/// `credentials`, which a path refuses.
pub fn open_target(
    target: &Target,
    credentials: &Credentials,
    create: Create,
    what: &str,
    w: &Wording,
) -> Result<FerroStore> {
    match target {
        Target::Path(path) => {
            if !credentials.is_empty() {
                refuse!(
                    "--psk-file, --token-file, --server-id, --ca-file, --pin-spki and \
                     --allow-plaintext are for a unix://, ws:// or wss:// URI, not a path"
                );
            }
            open_local(path, create, what, w)
        }
        Target::Uri(uri) => {
            let options = credentials.connect_options()?;
            FerroStore::connect_with(uri, &options)
                .map_err(|e| Error::context(format!("could not connect to {uri}"), e))
        }
    }
}

/// The `.fls` that `path` names, opened embedded. See [`open_target`].
///
/// A `.fls` that exists is opened. A missing one is refused when a redb store
/// is beside it (`path` itself when it is not the `.fls`, or the sibling
/// `.redb` of the `.fls`, so `x.fls` and `x.redb` are refused alike): every
/// `fl` process bound to that store would otherwise open the new, empty
/// `.fls` instead of migrating the `.redb`. Otherwise it is created as
/// `create` allows, while holding the store's migration lock (see
/// [`creation_lock`]), and the checks are made again under the lock.
pub(crate) fn open_local(
    path: &Path,
    create: Create,
    what: &str,
    w: &Wording,
) -> Result<FerroStore> {
    let physical = physical_path(path, BackendKind::Ferro);
    if present(&physical) {
        return Ok(FerroStore::open(&physical)?);
    }
    refuse_beside_redb(path, &physical, w)?;
    let cmd = w.command;
    let parent = match physical.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    match create {
        Create::Never => refuse!(
            "no store at {}; `{cmd} {what}` does not create one",
            physical.display()
        ),
        Create::NeedsFlag => refuse!(
            "no store at {}; `{cmd} {what}` creates one only with --create",
            physical.display()
        ),
        Create::InExistingDir => {
            if !parent.is_dir() {
                refuse!(
                    "no directory {} for the store {}; `{cmd} {what} --create` makes a store \
                     only in a directory that exists",
                    parent.display(),
                    physical.display()
                );
            }
        }
        Create::WithParents => {
            use std::os::unix::fs::DirBuilderExt;
            std::fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(&parent)
                .map_err(|e| Error::context(format!("could not make {}", parent.display()), e))?;
        }
    }
    let _lock = creation_lock(&physical)?;
    // Under the lock: a migration that finished meanwhile left its `.fls`
    // (opened as it is); a `.redb` that appeared meanwhile is refused.
    if !present(&physical) {
        refuse_beside_redb(path, &physical, w)?;
    }
    Ok(FerroStore::open(&physical)?)
}

/// Refuse when the `.fls` `physical` is missing and a redb store is beside
/// it: `path`, when it is another file than the `.fls`, or `x.redb` beside
/// `x.fls`, however the store was named.
fn refuse_beside_redb(path: &Path, physical: &Path, w: &Wording) -> Result<()> {
    let sibling = physical.with_extension("redb");
    let redb = [path, sibling.as_path()]
        .into_iter()
        .find(|p| *p != physical && present(p));
    if let Some(redb) = redb {
        refuse!(
            "{} holds a redb store and {} does not exist yet: migrate it first (`{} migrate`), \
             and nothing was created",
            redb.display(),
            physical.display(),
            w.command
        );
    }
    Ok(())
}

/// The migration lock of the store whose `.fls` is `physical`, taken without
/// waiting, for as long as the returned guard lives: a held lock is refused
/// (a migration of this store is running, and will make the `.fls` itself).
///
/// It is [`crate::MigrationLock`] itself (`<fls>.migrate.lock`, left in place
/// as that lock leaves it), so a migration and a creation exclude each other.
fn creation_lock(physical: &Path) -> Result<crate::MigrationLock> {
    match crate::MigrationLock::acquire_within(physical, std::time::Duration::ZERO) {
        Ok(lock) => Ok(lock),
        Err(e) if e.kind() == std::io::ErrorKind::TimedOut => refuse!(
            "{} is being migrated (another process holds {}): no store was created; run \
             this again once the migration is done",
            physical.display(),
            crate::MigrationLock::path_for(physical).display()
        ),
        Err(e) => Err(Error::context(
            format!(
                "could not take the migration lock {}",
                crate::MigrationLock::path_for(physical).display()
            ),
            e,
        )),
    }
}

/// Refuse a reserved (`__`) collection, as the store refuses a write to one.
fn refuse_reserved(collection: &str, what: &str, w: &Wording) -> Result<()> {
    if collection.starts_with("__") {
        refuse!(
            "`{} {what}` refused: collection `{collection}` is reserved (a `__` collection \
             belongs to the store itself)",
            w.command
        );
    }
    Ok(())
}

// ---- get ----

/// What `get` found.
#[derive(Debug, Clone, PartialEq)]
pub struct GetReport {
    pub collection: String,
    pub id: String,
    pub doc: Option<Value>,
}

/// The document `(collection, id)`.
pub fn get(store: &FerroStore, collection: &str, id: &str, w: &Wording) -> Result<GetReport> {
    refuse_reserved(collection, "get", w)?;
    Ok(GetReport {
        collection: collection.to_string(),
        id: id.to_string(),
        doc: store.get_doc(collection, id)?,
    })
}

impl GetReport {
    /// The document as pretty JSON on stdout (exit 0), or a line on stderr
    /// when there is none (exit 1).
    pub fn output(&self) -> Result<(Output, i32)> {
        let mut out = Output::default();
        match &self.doc {
            Some(doc) => {
                line(&mut out.stdout, super::to_json(doc)?);
                Ok((out, 0))
            }
            None => {
                line(
                    &mut out.stderr,
                    format!(
                        "no document `{}` in collection `{}`",
                        self.id, self.collection
                    ),
                );
                Ok((out, 1))
            }
        }
    }
}

// ---- put ----

/// The most bytes `put -` reads from stdin: 64 MiB. More is refused.
pub const MAX_STDIN_DOC: u64 = 64 * 1024 * 1024;

/// The document a `put` argument gives: the JSON text itself, or `-` for
/// the JSON on `stdin` (at most [`MAX_STDIN_DOC`] bytes; more is refused).
pub fn read_doc(arg: &str, stdin: impl Read) -> Result<Value> {
    read_doc_capped(arg, stdin, MAX_STDIN_DOC)
}

fn read_doc_capped(arg: &str, stdin: impl Read, cap: u64) -> Result<Value> {
    if arg != "-" {
        return serde_json::from_str(arg)
            .map_err(|e| Error::context("the document is not JSON", e));
    }
    // Bytes, not a string: a cut in the middle of a character is still
    // refused for its size, not reported as bad UTF-8.
    let mut bytes = Vec::new();
    stdin
        .take(cap.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|e| Error::context("could not read the document from stdin", e))?;
    if bytes.len() as u64 > cap {
        refuse!(
            "the document on stdin is over the {} MiB limit of `put -`",
            cap / (1024 * 1024)
        );
    }
    serde_json::from_slice(&bytes).map_err(|e| Error::context("the document is not JSON", e))
}

/// What `put` wrote. JSON: `{"collection", "id", "put": true}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PutReport {
    pub collection: String,
    pub id: String,
    pub put: bool,
}

/// Put `doc` at `(collection, id)`.
pub fn put(
    store: &FerroStore,
    collection: &str,
    id: &str,
    doc: &Value,
    w: &Wording,
) -> Result<PutReport> {
    refuse_reserved(collection, "put", w)?;
    store.put_doc(collection, id, doc)?;
    Ok(PutReport {
        collection: collection.to_string(),
        id: id.to_string(),
        put: true,
    })
}

// ---- query ----

/// One query.
#[derive(Debug, Clone, PartialEq)]
pub enum Query {
    /// The ids whose top-level `field` matches `value` (`query_by_field`).
    Field { field: String, value: String },
    /// The `k` best BM25 hits for `text`.
    Bm25 { text: String, k: usize },
    /// The `k` nearest vectors to `values`.
    Vector { values: Vec<f32>, k: usize },
}

impl Query {
    /// The query a command line names: `FIELD VALUE`; `--bm25 TEXT`; or
    /// `--vector --values JSON`, `-k` for the last two.
    pub fn from_args(
        bm25: bool,
        vector: bool,
        terms: &[String],
        values: Option<&str>,
        k: Option<usize>,
        w: &Wording,
    ) -> Result<Self> {
        let cmd = w.command;
        match (bm25, vector) {
            (true, true) => refuse!("--bm25 and --vector are two queries: give one"),
            (true, false) => {
                if values.is_some() {
                    refuse!("--values is for --vector");
                }
                let [text] = terms else {
                    refuse!("`{cmd} query --bm25 COLLECTION TEXT` takes one text (quote it)");
                };
                Ok(Self::Bm25 {
                    text: text.clone(),
                    k: k.unwrap_or(DEFAULT_K),
                })
            }
            (false, true) => {
                if !terms.is_empty() {
                    refuse!(
                        "`{cmd} query --vector COLLECTION --values JSON` takes the vector in \
                         --values, nothing after the collection"
                    );
                }
                let Some(raw) = values else {
                    refuse!("--vector needs --values '[1.0, 0.5, …]'");
                };
                Ok(Self::Vector {
                    values: parse_values(raw)?,
                    k: k.unwrap_or(DEFAULT_K),
                })
            }
            (false, false) => {
                if values.is_some() || k.is_some() {
                    refuse!("--values and -k are for --vector and --bm25");
                }
                let [field, value] = terms else {
                    refuse!("`{cmd} query COLLECTION FIELD VALUE` takes a field and a value");
                };
                Ok(Self::Field {
                    field: field.clone(),
                    value: value.clone(),
                })
            }
        }
    }
}

/// A JSON array of numbers, as `f32`.
fn parse_values(raw: &str) -> Result<Vec<f32>> {
    let parsed: Vec<f64> = serde_json::from_str(raw)
        .map_err(|e| Error::context("--values is not a JSON array of numbers", e))?;
    Ok(parsed.into_iter().map(|v| v as f32).collect())
}

/// What `query` found.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryReport {
    pub collection: String,
    pub query: Query,
    /// The ids of a field query, in ascending order.
    pub ids: Vec<String>,
    /// The `(id, score)` hits of a BM25 or vector query, best first.
    pub hits: Vec<(String, f64)>,
}

/// Run `query` on `collection`.
pub fn query(
    store: &FerroStore,
    collection: &str,
    query: &Query,
    w: &Wording,
) -> Result<QueryReport> {
    refuse_reserved(collection, "query", w)?;
    let (ids, hits) = match query {
        Query::Field { field, value } => {
            (store.query_by_field(collection, field, value)?, Vec::new())
        }
        Query::Bm25 { text, k } => (Vec::new(), store.search_bm25(collection, text, *k)?),
        Query::Vector { values, k } => (
            Vec::new(),
            store.search_vectors(collection, values, *k, None)?,
        ),
    };
    Ok(QueryReport {
        collection: collection.to_string(),
        query: query.clone(),
        ids,
        hits,
    })
}

#[derive(Serialize)]
struct Hit<'a> {
    id: &'a str,
    score: f64,
}

/// JSON: `collection` and `mode` (`field`, `bm25` or `vector`); then
/// `field`, `value` and `ids`; or `query` (BM25), `k` and `hits`
/// (`[{"id", "score"}]`).
impl Serialize for QueryReport {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(None)?;
        m.serialize_entry("collection", &self.collection)?;
        let hits = || {
            self.hits
                .iter()
                .map(|(id, score)| Hit { id, score: *score })
                .collect::<Vec<_>>()
        };
        match &self.query {
            Query::Field { field, value } => {
                m.serialize_entry("mode", "field")?;
                m.serialize_entry("field", field)?;
                m.serialize_entry("value", value)?;
                m.serialize_entry("ids", &self.ids)?;
            }
            Query::Bm25 { text, k } => {
                m.serialize_entry("mode", "bm25")?;
                m.serialize_entry("query", text)?;
                m.serialize_entry("k", k)?;
                m.serialize_entry("hits", &hits())?;
            }
            Query::Vector { k, .. } => {
                m.serialize_entry("mode", "vector")?;
                m.serialize_entry("k", k)?;
                m.serialize_entry("hits", &hits())?;
            }
        }
        m.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: Wording<'static> = Wording {
        command: "t",
        redb_path_hint: "h",
    };

    #[test]
    fn a_target_is_a_path_or_one_of_three_uris() {
        assert_eq!(
            Target::parse("/d/s.fls").unwrap(),
            Target::Path("/d/s.fls".into())
        );
        assert_eq!(
            Target::parse("unix:///d/s.sock").unwrap(),
            Target::Uri("unix:///d/s.sock".into())
        );
        assert_eq!(
            Target::parse("wss://h:1").unwrap(),
            Target::Uri("wss://h:1".into())
        );
        assert!(Target::parse("file:///d/s.fls").is_err());
        assert!(Target::parse("mem://").is_err());
    }

    #[test]
    fn credentials_name_one_method_and_a_psk_names_its_server() {
        let psk = Credentials {
            psk_file: Some("/k".into()),
            ..Credentials::default()
        };
        assert!(psk.connect_options().is_err());
        let both = Credentials {
            server_id: Some("s".into()),
            token_file: Some("/t".into()),
            ..psk.clone()
        };
        assert!(both.connect_options().is_err());
        let ok = Credentials {
            server_id: Some("s".into()),
            ..psk
        };
        assert!(ok.connect_options().is_ok());
    }

    #[test]
    fn a_query_is_read_from_its_arguments() {
        let terms = |t: &[&str]| t.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            Query::from_args(false, false, &terms(&["kind", "a"]), None, None, &W).unwrap(),
            Query::Field {
                field: "kind".into(),
                value: "a".into()
            }
        );
        assert_eq!(
            Query::from_args(true, false, &terms(&["red fox"]), None, Some(3), &W).unwrap(),
            Query::Bm25 {
                text: "red fox".into(),
                k: 3
            }
        );
        assert_eq!(
            Query::from_args(false, true, &[], Some("[1, 0.5]"), None, &W).unwrap(),
            Query::Vector {
                values: vec![1.0, 0.5],
                k: DEFAULT_K
            }
        );
        assert!(Query::from_args(true, true, &[], None, None, &W).is_err());
        assert!(Query::from_args(false, false, &terms(&["only"]), None, None, &W).is_err());
        assert!(Query::from_args(false, true, &[], None, None, &W).is_err());
        assert!(Query::from_args(false, true, &[], Some("{}"), None, &W).is_err());
    }

    #[test]
    fn a_document_comes_from_the_argument_or_stdin() {
        assert_eq!(read_doc(r#"{"a":1}"#, std::io::empty()).unwrap()["a"], 1);
        assert_eq!(read_doc("-", &b"[2]"[..]).unwrap()[0], 2);
        assert!(read_doc("not json", std::io::empty()).is_err());
    }

    #[test]
    fn stdin_is_read_up_to_the_cap_and_refused_past_it() {
        let cap = 2 * 1024 * 1024;
        // Exactly the cap: read whole.
        let mut at_cap = vec![b' '; usize::try_from(cap).unwrap()];
        at_cap[0] = b'7';
        assert_eq!(read_doc_capped("-", &at_cap[..], cap).unwrap(), 7);
        // One byte more: refused for its size, the limit named.
        let over = vec![b' '; usize::try_from(cap + 1).unwrap()];
        let err = read_doc_capped("-", &over[..], cap)
            .unwrap_err()
            .to_string();
        assert!(err.contains("over the 2 MiB limit"), "{err}");
        // A cut in the middle of a character is still a size refusal.
        let mut split = vec![b' '; usize::try_from(cap).unwrap()];
        split.extend_from_slice("é".as_bytes());
        let err = read_doc_capped("-", &split[..], cap)
            .unwrap_err()
            .to_string();
        assert!(err.contains("MiB limit"), "{err}");
        assert_eq!(MAX_STDIN_DOC, 64 * 1024 * 1024);
    }

    #[test]
    fn reserved_collections_are_refused_and_nothing_is_created() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.fls");
        let Err(err) = open_local(&path, Create::Never, "get", &W) else {
            panic!("get created a store");
        };
        let err = err.to_string();
        assert!(err.contains("does not create one"), "{err}");
        assert!(!path.exists());
        let store = open_local(&path, Create::InExistingDir, "put", &W).unwrap();
        for what in ["get", "put", "query"] {
            let err = refuse_reserved("__meta", what, &W).unwrap_err().to_string();
            assert!(err.contains("reserved"), "{err}");
        }
        assert!(get(&store, "__meta", "schema_version", &W).is_err());
        assert!(put(&store, "__meta", "x", &Value::Null, &W).is_err());
    }

    const EVERY_CREATE: [Create; 4] = [
        Create::Never,
        Create::NeedsFlag,
        Create::InExistingDir,
        Create::WithParents,
    ];

    #[test]
    fn a_lone_redb_is_refused_however_the_store_is_named() {
        let dir = tempfile::tempdir().unwrap();
        let redb = dir.path().join("s.redb");
        let fls = dir.path().join("s.fls");
        crate::RedbStore::open(&redb).unwrap();
        for named in [&fls, &redb] {
            for create in EVERY_CREATE {
                let Err(err) = open_local(named, create, "put", &W) else {
                    panic!("{} {create:?} created a store", named.display());
                };
                let err = err.to_string();
                assert!(err.contains("migrate it first"), "{err}");
                assert!(!fls.exists(), "{} {create:?}", named.display());
            }
        }
        // Not even the migration lock was made.
        assert!(!crate::MigrationLock::path_for(&fls).exists());
    }

    #[test]
    fn a_store_is_not_created_while_its_migration_lock_is_held() {
        use std::os::unix::fs::OpenOptionsExt;
        let dir = tempfile::tempdir().unwrap();
        let fls = dir.path().join("s.fls");
        let held = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(crate::MigrationLock::path_for(&fls))
            .unwrap();
        held.try_lock().unwrap();
        for create in [Create::InExistingDir, Create::WithParents] {
            let err = open_local(&fls, create, "serve", &W)
                .err()
                .unwrap()
                .to_string();
            assert!(err.contains("is being migrated"), "{err}");
            assert!(!fls.exists());
        }
        held.unlock().unwrap();
        // Released: created, and the lock file is left in place.
        open_local(&fls, Create::InExistingDir, "serve", &W).unwrap();
        assert!(fls.exists());
        assert!(crate::MigrationLock::path_for(&fls).exists());
    }

    #[test]
    fn creation_follows_the_create_policy() {
        let dir = tempfile::tempdir().unwrap();
        let fls = dir.path().join("s.fls");
        let err = open_local(&fls, Create::NeedsFlag, "put", &W)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("only with --create"), "{err}");
        assert!(!fls.exists());
        let deep = dir.path().join("typo").join("s.fls");
        let err = open_local(&deep, Create::InExistingDir, "put", &W)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("only in a directory that exists"), "{err}");
        assert!(!dir.path().join("typo").exists());
        open_local(&deep, Create::WithParents, "put", &W).unwrap();
        assert!(deep.exists());
        // An existing store opens under every policy.
        for create in EVERY_CREATE {
            open_local(&fls, Create::InExistingDir, "put", &W).unwrap();
            open_local(&fls, create, "get", &W).unwrap();
        }
    }
}
