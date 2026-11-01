//! The arguments of the commands `fl store` and `ferrostore` share, declared
//! once so the two programs take the same flags with the same help, and the
//! code that runs each from its arguments. A program flattens these into its
//! own subcommands and says where the store comes from (`fl store`: the bound
//! store; `ferrostore`: `--store`).

use super::bench::{self, BenchOptions, BenchTarget, Workload};
use super::data::{self, Create, Credentials, Query, Target};
use super::serve::{self, ServeOptions};
use super::{Output, Result, Scratch, SiblingScratch, Wording, inspect, to_json};
use clap::Args;
use std::io::Write;
use std::path::{Path, PathBuf};

/// A command's text: its stdout, then its stderr.
pub fn emit(out: &Output) {
    print!("{}", out.stdout);
    eprint!("{}", out.stderr);
}

/// How a client authenticates to a served store, and checks it.
#[derive(Args, Debug, Clone, Default)]
pub struct ClientArgs {
    /// For a unix://, ws:// or wss:// store: the PSK file (64+ hex digits,
    /// mode 0600). Needs --server-id.
    #[arg(long, value_name = "FILE")]
    pub psk_file: Option<PathBuf>,
    /// The name the server was given with its PSK.
    #[arg(long, value_name = "NAME")]
    pub server_id: Option<String>,
    /// For a ws:// or wss:// store: a bearer token file (the token on its
    /// first line, mode 0600).
    #[arg(long, value_name = "FILE")]
    pub token_file: Option<PathBuf>,
    /// For wss://: trust only the CA certificates in this PEM file.
    #[arg(long, value_name = "FILE")]
    pub ca_file: Option<PathBuf>,
    /// For wss://: trust the one server key whose SubjectPublicKeyInfo has
    /// this SHA-256 (64 hex digits), and no CA.
    #[arg(long, value_name = "HEX")]
    pub pin_spki: Option<String>,
    /// Send the credential over plain ws:// to an address that is not
    /// loopback (refused by default: use wss://).
    #[arg(long)]
    pub allow_plaintext: bool,
}

impl ClientArgs {
    pub fn credentials(&self) -> Credentials {
        Credentials {
            psk_file: self.psk_file.clone(),
            server_id: self.server_id.clone(),
            token_file: self.token_file.clone(),
            ca_file: self.ca_file.clone(),
            pin_spki: self.pin_spki.clone(),
            allow_plaintext: self.allow_plaintext,
        }
    }
}

/// `get COLLECTION ID`.
#[derive(Args, Debug, Clone)]
pub struct GetArgs {
    /// The document collection.
    pub collection: String,
    /// The document id.
    pub id: String,
    #[command(flatten)]
    pub client: ClientArgs,
}

impl GetArgs {
    /// The document as JSON on stdout; exit 1 when there is none. Creates no
    /// store.
    pub fn run(&self, target: &Target, w: &Wording) -> Result<(Output, i32)> {
        let store = data::open_target(target, &self.client.credentials(), Create::Never, "get", w)?;
        data::get(&store, &self.collection, &self.id, w)?.output()
    }
}

/// `put COLLECTION ID JSON|-`.
#[derive(Args, Debug, Clone)]
pub struct PutArgs {
    /// The document collection (not a reserved `__` one).
    pub collection: String,
    /// The document id.
    pub id: String,
    /// The document as JSON, or `-` to read it from stdin (64 MiB at most;
    /// more is refused).
    #[arg(value_name = "JSON|-")]
    pub json: String,
    #[command(flatten)]
    pub client: ClientArgs,
}

impl PutArgs {
    /// Put the document; `{"collection", "id", "put": true}` on stdout. A
    /// store path with nothing at it gets a new `.fls` as `create` allows.
    pub fn run(&self, target: &Target, create: Create, w: &Wording) -> Result<(Output, i32)> {
        let doc = data::read_doc(&self.json, std::io::stdin().lock())?;
        let store = data::open_target(target, &self.client.credentials(), create, "put", w)?;
        let report = data::put(&store, &self.collection, &self.id, &doc, w)?;
        json_output(&report)
    }
}

/// `query COLLECTION FIELD VALUE`, `query --bm25 COLLECTION TEXT`, or
/// `query --vector COLLECTION --values JSON`.
#[derive(Args, Debug, Clone)]
pub struct QueryArgs {
    /// Rank the documents of COLLECTION by BM25 against TEXT.
    #[arg(long)]
    pub bm25: bool,
    /// Find the vectors of the vector collection COLLECTION nearest to
    /// --values.
    #[arg(long)]
    pub vector: bool,
    /// The collection.
    pub collection: String,
    /// FIELD VALUE: the ids whose top-level FIELD matches VALUE. With
    /// --bm25: TEXT (quoted).
    #[arg(value_name = "TERMS")]
    pub terms: Vec<String>,
    /// With --vector: the query vector, a JSON array of numbers.
    #[arg(long, value_name = "JSON")]
    pub values: Option<String>,
    /// With --bm25 or --vector: how many hits (default 10).
    #[arg(short = 'k', long = "k", value_name = "N")]
    pub k: Option<usize>,
    #[command(flatten)]
    pub client: ClientArgs,
}

impl QueryArgs {
    /// The query's answer as JSON on stdout. Creates no store.
    pub fn run(&self, target: &Target, w: &Wording) -> Result<(Output, i32)> {
        let query = Query::from_args(
            self.bm25,
            self.vector,
            &self.terms,
            self.values.as_deref(),
            self.k,
            w,
        )?;
        let store = data::open_target(
            target,
            &self.client.credentials(),
            Create::Never,
            "query",
            w,
        )?;
        json_output(&data::query(&store, &self.collection, &query, w)?)
    }
}

fn json_output<T: serde::Serialize>(report: &T) -> Result<(Output, i32)> {
    let mut out = Output::default();
    super::line(&mut out.stdout, to_json(report)?);
    Ok((out, 0))
}

/// `serve`: where to listen and how connections authenticate.
#[derive(Args, Debug, Clone, Default)]
pub struct ServeArgs {
    /// The Unix socket to serve on (mode 0600; default: the store's `.fls`
    /// path plus `.sock`). Every peer must have this process's uid.
    #[arg(long, value_name = "PATH", conflicts_with = "tcp")]
    pub socket: Option<PathBuf>,
    /// Serve ws:// (or wss:// with --tls-cert) on this TCP address instead,
    /// such as 127.0.0.1:7878. Needs --psk-file, --token-file, or both.
    /// Without TLS only a loopback address, or one of an interface named by
    /// --allow-plaintext-on, is bound.
    #[arg(long, value_name = "ADDR")]
    pub tcp: Option<String>,
    /// The PSK file (64+ hex digits, mode 0600; make one with `head -c 32
    /// /dev/urandom | xxd -p -c 64`). Needs --server-id. On a Unix socket it
    /// is checked after the peer uid.
    #[arg(long, value_name = "FILE")]
    pub psk_file: Option<PathBuf>,
    /// The name this server proves itself under with its PSK (1 to 255
    /// bytes); clients give the same name.
    #[arg(long, value_name = "NAME")]
    pub server_id: Option<String>,
    /// With --tcp: the bearer token file (sha256, scope and label per line,
    /// mode 0600).
    #[arg(long, value_name = "FILE")]
    pub token_file: Option<PathBuf>,
    /// With --tcp: serve wss:// with this PEM certificate chain (leaf first).
    /// Needs --tls-key.
    #[arg(long, value_name = "FILE", requires = "tls_key")]
    pub tls_cert: Option<PathBuf>,
    /// With --tcp: the PEM private key of --tls-cert (mode 0600).
    #[arg(long, value_name = "FILE", requires = "tls_cert")]
    pub tls_key: Option<PathBuf>,
    /// With --tcp and no TLS: let plain ws:// bind an address of this
    /// interface (such as tailscale0). Repeat for more than one.
    #[arg(long, value_name = "IFACE")]
    pub allow_plaintext_on: Vec<String>,
    /// Print the ready line as one JSON object instead of text.
    #[arg(long)]
    pub json: bool,
}

impl ServeArgs {
    pub fn options(&self) -> ServeOptions {
        ServeOptions {
            socket: self.socket.clone(),
            tcp: self.tcp.clone(),
            psk_file: self.psk_file.clone(),
            server_id: self.server_id.clone(),
            token_file: self.token_file.clone(),
            tls_cert: self.tls_cert.clone(),
            tls_key: self.tls_key.clone(),
            allow_plaintext_on: self.allow_plaintext_on.clone(),
        }
    }

    /// Serve the store `path` names (created, when nothing is there, as
    /// `create` allows) until SIGINT or SIGTERM: one ready line on stdout
    /// once it listens (the URI to connect to), one line on stderr when it
    /// stops.
    pub fn run(&self, path: &Path, create: Create, w: &Wording) -> Result<i32> {
        let json = self.json;
        let (_, signal) = serve::serve(path, &self.options(), create, w, |serving| {
            let text = if json {
                serving.json_line()?
            } else {
                serving.text()
            };
            let mut stdout = std::io::stdout().lock();
            stdout.write_all(text.as_bytes())?;
            stdout.flush()?;
            Ok(())
        })?;
        eprintln!("stopped on {signal}");
        Ok(0)
    }
}

/// `inspect`.
#[derive(Args, Debug, Clone, Default)]
pub struct InspectArgs {
    /// Print one JSON object instead of text.
    #[arg(long)]
    pub json: bool,
}

impl InspectArgs {
    /// The superblocks and contents of the `.fls` `path` names, read only.
    /// Exit 1 when the live state could not be read.
    pub fn run(&self, path: &Path, w: &Wording) -> Result<(Output, i32)> {
        let report = inspect::inspect(path, w)?;
        let code = inspect::exit_code(&report);
        if self.json {
            let (out, _) = json_output(&report)?;
            return Ok((out, code));
        }
        Ok((inspect::text(&report), code))
    }
}

/// `bench`.
#[derive(Args, Debug, Clone)]
pub struct BenchArgs {
    /// Make the fresh stores in a private directory inside DIR (removed
    /// afterwards). Default: the system temp directory, often RAM-backed, so
    /// pass a directory on the disk the stores will live on.
    #[arg(long, value_name = "DIR", conflicts_with = "path")]
    pub dir: Option<PathBuf>,
    /// The workloads: put, get, scan, vector, a comma-separated list, or all.
    #[arg(long, value_name = "LIST", default_value = "all")]
    pub workload: String,
    /// Operations per workload.
    #[arg(long, value_name = "N", default_value_t = bench::DEFAULT_N)]
    pub n: usize,
    /// The engines: ferro, redb, or both comma-separated.
    #[arg(long, value_name = "LIST", default_value = "ferro,redb")]
    pub engines: String,
    /// Run on this existing FerroStore instead of fresh stores (put, get and
    /// scan only, in a collection of its own, `ferrostore-bench-<pid>-<n>`,
    /// that is emptied afterwards). Needs --i-know-it-writes. A run that is
    /// killed part way leaves up to N documents in that collection (the next
    /// `bench --path` on the store removes them, once that process is gone),
    /// and the writes can start the store's own automatic checkpoint or
    /// compaction.
    #[arg(long, value_name = "STORE")]
    pub path: Option<PathBuf>,
    /// Confirm that --path writes to that store.
    #[arg(long)]
    pub i_know_it_writes: bool,
    /// Print one JSON object instead of text.
    #[arg(long)]
    pub json: bool,
}

impl BenchArgs {
    pub fn options(&self) -> Result<BenchOptions> {
        Ok(BenchOptions {
            workloads: Workload::parse_list(&self.workload)?,
            n: self.n,
            engines: bench::parse_engines(&self.engines)?,
            target: match &self.path {
                Some(path) => BenchTarget::Existing {
                    path: path.clone(),
                    confirmed: self.i_know_it_writes,
                },
                None => BenchTarget::Fresh,
            },
        })
    }

    /// Run the bench. Fresh stores go in `--dir`, else in the directory
    /// `default_scratch` makes.
    pub fn run<S: Scratch>(
        &self,
        default_scratch: impl FnOnce() -> Result<S>,
        w: &Wording,
    ) -> Result<(Output, i32)> {
        let options = self.options()?;
        let report = match &self.dir {
            Some(dir) => bench::bench(&options, || SiblingScratch::within(dir), w)?,
            None => bench::bench(&options, default_scratch, w)?,
        };
        if self.json {
            return json_output(&report);
        }
        Ok((report.text(), 0))
    }
}
