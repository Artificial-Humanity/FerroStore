//! `ferrostore`: the FerroStore tool. A thin argument parser over
//! [`fl_store::cli`], which `fl store` calls too, so the two print the same.
//!
//! What is the binary's own: the store comes from `--store` (a path, or a
//! URI for `get`, `put` and `query`) or the `inspect` argument; the engine of
//! `status`, `checkpoint` and `compact` from `--engine` (the environment is
//! never read: `FL_STORE_BACKEND` means nothing here); and the private copies
//! of `migrate --dry-run` and `verify` are made beside the store, on its own
//! filesystem ([`SiblingScratch`]), not in the RAM-backed `/tmp`.

use clap::{Args, Parser, Subcommand};
use fl_store::cli::args::{BenchArgs, GetArgs, InspectArgs, PutArgs, QueryArgs, ServeArgs, emit};
use fl_store::cli::{self as store_cli, Create, Output, SiblingScratch, Target, Wording};
use fl_store::{BackendKind, choose_backend};
use std::path::PathBuf;

/// How `ferrostore` names itself in what the commands print.
const WORDING: Wording<'static> = Wording {
    command: "ferrostore",
    redb_path_hint: "--store <file>.redb",
};

#[derive(Parser)]
#[command(
    name = "ferrostore",
    version,
    about = "Look at, tend, serve and query a FerroStore (.fls) file."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args)]
struct StorePath {
    /// The store: its path (`x.redb` names the FerroStore `x.fls`).
    #[arg(long, short = 's', value_name = "PATH")]
    store: PathBuf,
}

#[derive(Args)]
struct StoreTarget {
    /// The store: a path (opened embedded; `x.redb` names `x.fls`), or the
    /// unix://, ws:// or wss:// URI of a served one.
    #[arg(long, short = 's', value_name = "PATH|URI")]
    store: String,
}

#[derive(Args)]
struct CreateFlag {
    /// Create the store (a new `.fls`, mode 0600) when nothing is at the path.
    /// Its directory must exist; a `.fls` is never made beside a `.redb`.
    #[arg(long)]
    create: bool,
}

impl CreateFlag {
    fn policy(&self) -> Create {
        if self.create {
            Create::InExistingDir
        } else {
            Create::NeedsFlag
        }
    }
}

#[derive(Args)]
struct Engine {
    /// The engine: ferro (the default) or redb. A .fls path is always ferro.
    /// Never read from the environment.
    #[arg(long, value_name = "ENGINE", value_parser = ["ferro", "redb"])]
    engine: Option<String>,
}

impl Engine {
    fn kind(&self, store: &std::path::Path) -> store_cli::Result<BackendKind> {
        Ok(choose_backend(store, self.engine.as_deref())?)
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// The engine, files and format version of a store. Read-only.
    Status {
        #[command(flatten)]
        store: StorePath,
        #[command(flatten)]
        engine: Engine,
        /// Print one JSON object instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Fold a FerroStore write-ahead log into the store file. redb has none.
    Checkpoint {
        #[command(flatten)]
        store: StorePath,
        #[command(flatten)]
        engine: Engine,
    },
    /// Rewrite a FerroStore as one snapshot and an empty log, giving back the
    /// space earlier checkpoints left dead. On redb it says so and does
    /// nothing. Never creates or migrates a store.
    Compact {
        #[command(flatten)]
        store: StorePath,
        #[command(flatten)]
        engine: Engine,
        /// Print one JSON object instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Copy a `.redb` store into a new `.fls`, verified; the `.redb` is kept,
    /// read-only (0400). Exit 1 when the copy does not verify.
    Migrate {
        #[command(flatten)]
        store: StorePath,
        /// Migrate a private copy, report, and delete it; the real files are
        /// not written. The copy is made in a private directory beside the
        /// store and needs room for two files the size of the .redb (checked
        /// first); the .redb is locked against writers while it is copied.
        #[arg(long)]
        dry_run: bool,
        /// Print one JSON object instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Compare an existing `.redb` and `.fls` pair, table by table and view by
    /// view. Exit 1 when they differ. Works on a private copy of the .redb
    /// made beside the store (room is checked first); the .redb is locked
    /// against writers while it is copied.
    Verify {
        #[command(flatten)]
        store: StorePath,
        /// Print one JSON object instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Serve a store on a Unix socket, or on TCP as ws:// or wss://, until
    /// SIGINT or SIGTERM. A missing store is refused unless --create is given.
    Serve {
        #[command(flatten)]
        store: StorePath,
        #[command(flatten)]
        create: CreateFlag,
        #[command(flatten)]
        args: ServeArgs,
    },
    /// Print one document as JSON. Exit 1 when there is none.
    Get {
        #[command(flatten)]
        store: StoreTarget,
        #[command(flatten)]
        args: GetArgs,
    },
    /// Put one document, given as JSON or on stdin (`-`). A missing store is
    /// refused unless --create is given.
    Put {
        #[command(flatten)]
        store: StoreTarget,
        #[command(flatten)]
        create: CreateFlag,
        #[command(flatten)]
        args: PutArgs,
    },
    /// Query a collection by field, by BM25, or by vector; JSON on stdout.
    Query {
        #[command(flatten)]
        store: StoreTarget,
        #[command(flatten)]
        args: QueryArgs,
    },
    /// The superblocks, versions, extents, live and dead bytes, and the counts
    /// of every collection, column table and vector collection of a `.fls`.
    /// Strictly read-only. Exit 1 when the live state cannot be read.
    Inspect {
        /// The store (`x.redb` names `x.fls`).
        path: PathBuf,
        #[command(flatten)]
        args: InspectArgs,
    },
    /// Time put, get, scan and vector-search workloads on fresh FerroStore
    /// and redb stores (an existing store only with --path and
    /// --i-know-it-writes): p50, p99 and throughput.
    Bench {
        #[command(flatten)]
        args: BenchArgs,
    },
}

fn run(cmd: Cmd) -> store_cli::Result<i32> {
    let w = &WORDING;
    let printed = |out: Output, code: i32| {
        emit(&out);
        Ok(code)
    };
    match cmd {
        Cmd::Status {
            store,
            engine,
            json,
        } => {
            let report = store_cli::status(&store.store, engine.kind(&store.store)?);
            if json {
                println!("{}", store_cli::to_json(&report)?);
            } else {
                print!("{}", report.text(w));
            }
            Ok(0)
        }
        Cmd::Checkpoint { store, engine } => {
            let report = store_cli::checkpoint(&store.store, engine.kind(&store.store)?, w)?;
            print!("{}", report.text());
            Ok(0)
        }
        Cmd::Compact {
            store,
            engine,
            json,
        } => {
            let report = store_cli::compact(&store.store, engine.kind(&store.store)?, w)?;
            if json {
                println!("{}", store_cli::to_json(&report)?);
            } else {
                print!("{}", report.text());
            }
            Ok(0)
        }
        Cmd::Migrate {
            store,
            dry_run,
            json,
        } => {
            let pair = store_cli::migrate_pair(&store.store, w)?;
            sweep(pair.redb());
            let report =
                store_cli::migrate(&pair, dry_run, || SiblingScratch::beside(pair.redb()))?;
            if json {
                println!("{}", store_cli::to_json(&report)?);
            } else {
                emit(&report.text(w));
            }
            Ok(report.exit_code())
        }
        Cmd::Verify { store, json } => {
            let pair = store_cli::verify_pair(&store.store, w)?;
            sweep(pair.redb());
            let report = store_cli::verify(&pair, || SiblingScratch::beside(pair.redb()))?;
            if json {
                println!("{}", store_cli::to_json(&report)?);
            } else {
                emit(&report.text());
            }
            Ok(report.exit_code())
        }
        Cmd::Serve {
            store,
            create,
            args,
        } => args.run(&store.store, create.policy(), w),
        Cmd::Get { store, args } => {
            let (out, code) = args.run(&Target::parse(&store.store)?, w)?;
            printed(out, code)
        }
        Cmd::Put {
            store,
            create,
            args,
        } => {
            let (out, code) = args.run(&Target::parse(&store.store)?, create.policy(), w)?;
            printed(out, code)
        }
        Cmd::Query { store, args } => {
            let (out, code) = args.run(&Target::parse(&store.store)?, w)?;
            printed(out, code)
        }
        Cmd::Inspect { path, args } => {
            let (out, code) = args.run(&path, w)?;
            printed(out, code)
        }
        Cmd::Bench { args } => {
            let (out, code) = args.run(|| SiblingScratch::within(&std::env::temp_dir()), w)?;
            printed(out, code)
        }
    }
}

/// Remove the private copies an earlier `migrate --dry-run` or `verify` on
/// this store left beside it when it was killed; what could not be removed
/// is a warning on stderr, and the command goes on.
fn sweep(redb: &std::path::Path) {
    for warning in SiblingScratch::sweep_beside(redb) {
        eprintln!("{warning}");
    }
}

/// `error: <message>: <cause>: …`, the whole chain, as `fl` prints it.
fn chain(e: &dyn std::error::Error) -> String {
    let mut text = e.to_string();
    let mut next = e.source();
    while let Some(cause) = next {
        text.push_str(": ");
        text.push_str(&cause.to_string());
        next = cause.source();
    }
    text
}

fn main() {
    let cli = Cli::parse();
    let code = match run(cli.cmd) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {}", chain(&e));
            2
        }
    };
    std::process::exit(code);
}
