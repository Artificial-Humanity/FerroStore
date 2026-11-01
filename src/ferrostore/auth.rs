//! Authentication and authorization of a served FerroStore (spec §6.2).
//!
//! Every connection authenticates before the server reads a request:
//!
//! * **Unix socket.** The peer's uid (`SO_PEERCRED`) must be the server's
//!   effective uid. A bind that configures a PSK ([`FerroStoreServer::bind_unix_with`])
//!   also runs the PSK challenge-response. Either way the connection holds
//!   [`Scope::Admin`].
//! * **TCP (`ws://`).** A bearer token in the upgrade's `Authorization`
//!   header gets the scope its token-file line names. Without that header
//!   the server runs the mutual PSK challenge-response, and a connection that
//!   passes holds [`Scope::Admin`].
//!
//! A failed authentication waits [`AUTH_FAIL_DELAY`], sends one frame that
//! names no check, and closes. Every method needs a scope, from the one table
//! [`METHODS`]; a method the connection's scope does not cover is an error
//! reply, and the connection stays open.
//!
//! Neither a PSK nor a token is ever printed: the types that hold them have
//! a redacted `Debug`, and no error repeats one.
//!
//! [`FerroStoreServer::bind_unix_with`]: super::FerroStoreServer::bind_unix_with

use super::bind;
use super::layout::backend;
use super::server::{Incoming, JsonIo};
use super::tls::{self, Binding, ServerTls, Verify};
use fl_core::StoreError;
use ring::hmac;
use ring::rand::{SecureRandom, SystemRandom};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// How long a refused connection waits before its failure frame and close,
/// whichever check failed.
pub(super) const AUTH_FAIL_DELAY: Duration = Duration::from_millis(200);
/// The longest either side waits for one frame of the handshake.
pub(super) const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
/// The longest a connection may take to authenticate, from its accept to
/// its `ok`, however its bytes trickle in: the server shuts down one still
/// authenticating past it (spec §6.2).
pub(super) const HANDSHAKE_DEADLINE: Duration = Duration::from_secs(10);
/// The most TCP connections a server lets authenticate at once. A TCP one
/// accepted past it is closed at once, before anything is read from it;
/// connections that have authenticated, and Unix connections, are not
/// counted.
pub(super) const MAX_HANDSHAKES: usize = 128;
/// The largest frame either side reads before the connection has
/// authenticated: every handshake frame is far smaller. The length a frame
/// declares is checked against it before anything is allocated.
pub(super) const PRE_AUTH_FRAME_BYTES: usize = 4096;
/// The message of every refused authentication, on the client.
pub(super) const AUTH_FAILED: &str = "FerroStore authentication failed";
/// The `error` of the failure frame: it names no check.
const FAILED_FRAME_ERROR: &str = "authentication failed";

/// The labels of the two proofs over a Unix socket or plain `ws://`
/// (version 1): the same length, and never equal, so a proof made in one
/// direction is never valid in the other.
const CLIENT_LABEL: &[u8; 17] = b"fl-store/1/client";
const SERVER_LABEL: &[u8; 17] = b"fl-store/1/server";
/// The labels over `wss://` (version 2), whose input also ends with the
/// TLS channel binding (spec §6.3). They differ from the version-1 labels,
/// so a proof of one version is never valid in the other.
const CLIENT_LABEL_TLS: &[u8; 17] = b"fl-store/2/client";
const SERVER_LABEL_TLS: &[u8; 17] = b"fl-store/2/server";
/// The handshake version the challenge names: 1 without TLS.
const AUTH_VERSION: u64 = 1;
/// The handshake version over TLS: the proofs are bound to the TLS session.
const AUTH_VERSION_TLS: u64 = 2;
const NONCE_LEN: usize = 32;
/// The shortest PSK accepted, in bytes.
pub(super) const MIN_PSK_LEN: usize = 32;
/// The fewest hex digits a PSK file or variable holds: [`MIN_PSK_LEN`] bytes.
const MIN_PSK_HEX: usize = 2 * MIN_PSK_LEN;
/// How an operator makes a PSK file.
const MAKE_PSK: &str = "head -c 32 /dev/urandom | xxd -p -c 64 > psk && chmod 600 psk";
/// The prefix of every environment variable a `ws://` URI may name.
pub(super) const URI_ENV_PREFIX: &str = "FERROSTORE_";
/// The longest `server_id`, in bytes: its length is one byte of the MAC input.
const MAX_SERVER_ID: usize = 255;
/// The largest secret file read (PSK or token file).
const MAX_SECRET_FILE: u64 = 1024 * 1024;

type Nonce = [u8; NONCE_LEN];

// ---- scopes ---------------------------------------------------------------------

/// Who a connection authenticated as: `token:<sha256 hex>` for a bearer
/// token, `psk` for the PSK over TCP, `unix:<uid>` for a Unix peer. The key
/// space of request ids is per principal; the quotas (sessions, exactly-once
/// ids) bind TCP principals only, since every Unix peer is the owner's own uid
/// (spec §6.1).
pub(super) type Principal = Arc<str>;

/// What a connection may do. `Admin` covers `Write`, which covers `Read`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Scope {
    Read,
    Write,
    Admin,
}

impl Scope {
    pub(super) fn name(self) -> &'static str {
        match self {
            Scope::Read => "read",
            Scope::Write => "write",
            Scope::Admin => "admin",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        match text {
            "read" => Some(Scope::Read),
            "write" => Some(Scope::Write),
            "admin" => Some(Scope::Admin),
            _ => None,
        }
    }

    /// `true` when this scope may run a method that needs `needed`.
    pub(super) fn covers(self, needed: Scope) -> bool {
        self >= needed
    }
}

/// One method the server answers: the scope it needs, and whether it writes
/// (a `req` on it makes it exactly-once, spec §6.1).
pub(super) struct MethodRule {
    pub(super) name: &'static str,
    pub(super) scope: Scope,
    pub(super) writes: bool,
}

const fn method(name: &'static str, scope: Scope, writes: bool) -> MethodRule {
    MethodRule {
        name,
        scope,
        writes,
    }
}

/// Every method the server answers, and the only place a method's scope is
/// decided. A method not here is `no such method`, whatever the scope.
///
/// * `read`: sessions and every read primitive or read RPC.
/// * `write`: every data write. `commit_txn` is `write` even with empty
///   `ops`: it takes the write lock, and its answer says whether the reads
///   still match. Creating a column table or a vector collection is a data
///   write, as the same op staged in a `commit_txn` is.
/// * `admin`: `checkpoint` and `compact` (they rewrite the file), and
///   `apply_sync_batches`, which applies another replica's batches at their
///   own sequence numbers: replication, not a client write.
pub(super) const METHODS: &[MethodRule] = &[
    method("open_session", Scope::Read, false),
    method("close_session", Scope::Read, false),
    method("txn_read", Scope::Read, false),
    method("get_doc", Scope::Read, false),
    method("query_by_field", Scope::Read, false),
    method("search_bm25", Scope::Read, false),
    method("poll_mq_group", Scope::Read, false),
    method("subscribe_mq", Scope::Read, false),
    method("scan_column_table", Scope::Read, false),
    method("attempt_rollup", Scope::Read, false),
    method("gate_run_rollup", Scope::Read, false),
    method("has_vector_collection", Scope::Read, false),
    method("get_vector", Scope::Read, false),
    method("search_vectors", Scope::Read, false),
    method("search_hybrid", Scope::Read, false),
    method("search_knowledge", Scope::Read, false),
    method("commit_txn", Scope::Write, true),
    method("put_doc", Scope::Write, true),
    method("delete_doc", Scope::Write, true),
    method("publish_mq", Scope::Write, true),
    method("ack_mq", Scope::Write, true),
    method("create_column_table", Scope::Write, true),
    method("append_column_rows", Scope::Write, true),
    method("create_vector_collection", Scope::Write, true),
    method("upsert_vector", Scope::Write, true),
    method("delete_vector", Scope::Write, true),
    method("checkpoint", Scope::Admin, false),
    method("compact", Scope::Admin, false),
    method("apply_sync_batches", Scope::Admin, true),
];

/// `method`'s row of [`METHODS`].
pub(super) fn method_rule(method: &str) -> Option<&'static MethodRule> {
    METHODS.iter().find(|rule| rule.name == method)
}

/// `method`'s rule when `scope` covers it. An unknown method is `no such
/// method`; one the scope does not cover names the scope it needs.
pub(super) fn authorize(scope: Scope, method: &str) -> Result<&'static MethodRule, StoreError> {
    let Some(rule) = method_rule(method) else {
        return Err(backend(format!("no such method `{method}`")));
    };
    if !scope.covers(rule.scope) {
        return Err(backend(format!(
            "method `{method}` needs the `{}` scope; this connection has `{}`",
            rule.scope.name(),
            scope.name()
        )));
    }
    Ok(rule)
}

// ---- the PSK and its proofs -----------------------------------------------------

/// A pre-shared key and the `server_id` its proofs are bound to.
///
/// `server_id` is the name the server proves itself under: an operator-chosen
/// name (not an address, which NAT or a tailnet may change), 1 to 255 bytes
/// of UTF-8 without control characters, configured on the server with its
/// PSK and on each client as the server it expects. A proof for one
/// `server_id` is never valid for another that shares the PSK.
#[derive(Clone)]
pub(super) struct Psk {
    server_id: String,
    key: hmac::Key,
}

impl std::fmt::Debug for Psk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Psk")
            .field("server_id", &self.server_id)
            .field("key", &"<redacted>")
            .finish()
    }
}

impl Psk {
    pub(super) fn new(server_id: &str, secret: &[u8]) -> Result<Self, StoreError> {
        check_server_id(server_id)?;
        if secret.len() < MIN_PSK_LEN {
            return Err(backend(format!(
                "a FerroStore PSK is at least {MIN_PSK_LEN} bytes; this one is {}",
                secret.len()
            )));
        }
        Ok(Self {
            server_id: server_id.to_string(),
            key: hmac::Key::new(hmac::HMAC_SHA256, secret),
        })
    }

    /// Without TLS (`binding` is `None`):
    /// `label ‖ len(server_id) as one byte ‖ server_id ‖ first ‖ second`.
    /// Over TLS: the same with the version-2 label, then the 32-byte TLS
    /// exporter `binding` at the end. Every part but `server_id` has a fixed
    /// length, its length is written first, and the label names whether a
    /// binding follows, so no two inputs read the same.
    fn input(
        &self,
        labels: (&[u8; 17], &[u8; 17]),
        first: &Nonce,
        second: &Nonce,
        binding: Option<&Binding>,
    ) -> Vec<u8> {
        let label = if binding.is_some() {
            labels.1
        } else {
            labels.0
        };
        let id = self.server_id.as_bytes();
        let mut input =
            Vec::with_capacity(label.len() + 1 + id.len() + 2 * NONCE_LEN + tls::BINDING_LEN);
        input.extend_from_slice(label);
        // `check_server_id` bounds it to one byte.
        input.push(u8::try_from(id.len()).unwrap_or(u8::MAX));
        input.extend_from_slice(id);
        input.extend_from_slice(first);
        input.extend_from_slice(second);
        if let Some(binding) = binding {
            input.extend_from_slice(binding);
        }
        input
    }

    fn sign(&self, input: &[u8]) -> [u8; 32] {
        let tag = hmac::sign(&self.key, input);
        let mut out = [0u8; 32];
        out.copy_from_slice(tag.as_ref());
        out
    }

    /// `HMAC-SHA256(psk, "fl-store/1/client" ‖ L ‖ server_id ‖ ns ‖ nc)`,
    /// or over TLS
    /// `HMAC-SHA256(psk, "fl-store/2/client" ‖ L ‖ server_id ‖ ns ‖ nc ‖ binding)`.
    pub(super) fn client_proof(
        &self,
        ns: &Nonce,
        nc: &Nonce,
        binding: Option<&Binding>,
    ) -> [u8; 32] {
        self.sign(&self.input((CLIENT_LABEL, CLIENT_LABEL_TLS), ns, nc, binding))
    }

    /// `HMAC-SHA256(psk, "fl-store/1/server" ‖ L ‖ server_id ‖ nc ‖ ns)`,
    /// or over TLS
    /// `HMAC-SHA256(psk, "fl-store/2/server" ‖ L ‖ server_id ‖ nc ‖ ns ‖ binding)`.
    pub(super) fn server_proof(
        &self,
        nc: &Nonce,
        ns: &Nonce,
        binding: Option<&Binding>,
    ) -> [u8; 32] {
        self.sign(&self.input((SERVER_LABEL, SERVER_LABEL_TLS), nc, ns, binding))
    }

    /// [`Self::client_proof`] checked in constant time.
    pub(super) fn verify_client(
        &self,
        ns: &Nonce,
        nc: &Nonce,
        binding: Option<&Binding>,
        proof: &[u8],
    ) -> bool {
        let input = self.input((CLIENT_LABEL, CLIENT_LABEL_TLS), ns, nc, binding);
        hmac::verify(&self.key, &input, proof).is_ok()
    }

    /// [`Self::server_proof`] checked in constant time.
    pub(super) fn verify_server(
        &self,
        nc: &Nonce,
        ns: &Nonce,
        binding: Option<&Binding>,
        proof: &[u8],
    ) -> bool {
        let input = self.input((SERVER_LABEL, SERVER_LABEL_TLS), nc, ns, binding);
        hmac::verify(&self.key, &input, proof).is_ok()
    }
}

fn check_server_id(server_id: &str) -> Result<(), StoreError> {
    if server_id.is_empty()
        || server_id.len() > MAX_SERVER_ID
        || server_id.chars().any(char::is_control)
    {
        return Err(backend(format!(
            "a FerroStore server_id is 1 to {MAX_SERVER_ID} bytes without control characters"
        )));
    }
    Ok(())
}

fn random_nonce() -> Result<Nonce, StoreError> {
    let mut nonce = [0u8; NONCE_LEN];
    SystemRandom::new()
        .fill(&mut nonce)
        .map_err(|_| backend("the system random source failed"))?;
    Ok(nonce)
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from(DIGITS[usize::from(byte >> 4)]));
        out.push(char::from(DIGITS[usize::from(byte & 0xf)]));
    }
    out
}

/// 32 bytes from exactly 64 lowercase hex digits.
fn unhex32(text: &str) -> Option<[u8; 32]> {
    fn digit(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            _ => None,
        }
    }
    let bytes = text.as_bytes();
    if bytes.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (slot, pair) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
        *slot = (digit(pair[0])? << 4) | digit(pair[1])?;
    }
    Some(out)
}

// ---- bearer tokens --------------------------------------------------------------

/// The token file: the SHA-256 of each token and its scope. No token is held.
struct TokenTable {
    entries: Vec<([u8; 32], Scope)>,
}

impl TokenTable {
    /// One token per line: `<sha256 hex> <scope> [label...]`, the hash in 64
    /// lowercase hex digits and the scope `read`, `write` or `admin`. Blank
    /// lines and lines that start with `#` are skipped. A bad or repeated
    /// line is an error that names its line number and nothing of its text.
    fn parse(text: &str) -> Result<Self, StoreError> {
        let mut entries: Vec<([u8; 32], Scope)> = Vec::new();
        for (index, line) in text.lines().enumerate() {
            let number = index + 1;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut fields = line.split_whitespace();
            let hash = fields.next().and_then(unhex32).ok_or_else(|| {
                backend(format!(
                    "token file line {number}: the first field is not 64 lowercase hex digits \
                     (the SHA-256 of a token)"
                ))
            })?;
            let scope = fields.next().and_then(Scope::parse).ok_or_else(|| {
                backend(format!(
                    "token file line {number}: the scope is not `read`, `write` or `admin`"
                ))
            })?;
            if entries.iter().any(|(seen, _)| *seen == hash) {
                return Err(backend(format!(
                    "token file line {number}: this token hash is already listed"
                )));
            }
            entries.push((hash, scope));
        }
        if entries.is_empty() {
            return Err(backend("the token file lists no token"));
        }
        Ok(Self { entries })
    }

    /// The scope of `token`, and its principal. Its SHA-256 is compared
    /// with every entry in constant time, and every entry is compared.
    fn lookup(&self, token: &str) -> Option<(Scope, Principal)> {
        let digest: [u8; 32] = Sha256::digest(token.as_bytes()).into();
        let mut found = None;
        for (hash, scope) in &self.entries {
            if equal_32(hash, &digest) {
                found = Some(*scope);
            }
        }
        found.map(|scope| (scope, Principal::from(format!("token:{}", hex(&digest)))))
    }
}

/// `a == b`, without an early exit on the first difference.
fn equal_32(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let diff = a
        .iter()
        .zip(b.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y));
    std::hint::black_box(diff) == 0
}

/// A token as a client may send it in a header: 1 or more visible ASCII
/// characters, so no header can be injected.
fn check_token(token: &str) -> Result<(), StoreError> {
    if token.is_empty() || !token.bytes().all(|byte| byte.is_ascii_graphic()) {
        return Err(backend(
            "a FerroStore bearer token is one or more visible ASCII characters, without spaces",
        ));
    }
    Ok(())
}

// ---- the server's configuration ---------------------------------------------------

/// How a [`FerroStoreServer`](super::FerroStoreServer) authenticates its
/// connections (spec §6.2), and how its TCP bind is secured (spec §6.3).
///
/// `bind_tcp` needs a PSK, a token file, or both. On a Unix socket the
/// peer's uid is always checked; `bind_unix_with` given a PSK requires the
/// PSK as well. A token file, TLS and `allow_plaintext_on` apply to TCP
/// alone.
///
/// With [`Self::with_tls_files`], `bind_tcp` serves `wss://` and may bind any
/// address. Without TLS it serves plain `ws://` and binds a loopback address,
/// or an address of an interface named in [`Self::allow_plaintext_on`];
/// anything else, the unspecified `0.0.0.0` and `::` included, is refused.
///
/// ```no_run
/// use fl_store::ferrostore::{FerroStore, FerroStoreServer, ServerAuth};
/// let auth = ServerAuth::new()
///     .with_psk_file("lab-store", "/etc/ferrostore/psk")?
///     .with_token_file("/etc/ferrostore/tokens")?
///     .with_tls_files("/etc/ferrostore/cert.pem", "/etc/ferrostore/key.pem")?;
/// let server = FerroStoreServer::bind_tcp("0.0.0.0:7878", FerroStore::in_memory(), auth)?;
/// # Ok::<(), fl_core::StoreError>(())
/// ```
#[derive(Default)]
pub struct ServerAuth {
    psk: Option<Psk>,
    tokens: Option<TokenTable>,
    tls: Option<ServerTls>,
    allow_plaintext_on: Vec<String>,
}

impl std::fmt::Debug for ServerAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerAuth")
            .field("psk", &self.psk)
            .field(
                "tokens",
                &self.tokens.as_ref().map(|table| table.entries.len()),
            )
            .field("tls", &self.tls.is_some())
            .field("allow_plaintext_on", &self.allow_plaintext_on)
            .finish()
    }
}

impl ServerAuth {
    /// No method yet: add a PSK or a token file.
    pub fn new() -> Self {
        Self::default()
    }

    /// Prove the server as `server_id` with the PSK `psk`: the key's own
    /// bytes, at least 32 of them (not its hex; a file holds hex, see
    /// [`Self::with_psk_file`]).
    pub fn with_psk(mut self, server_id: &str, psk: &[u8]) -> Result<Self, StoreError> {
        self.psk = Some(Psk::new(server_id, psk)?);
        Ok(self)
    }

    /// [`Self::with_psk`] with the PSK the file at `path` holds: 64 or more
    /// hex digits (32 or more random bytes), one trailing newline dropped,
    /// decoded to the key's bytes. Anything else is refused with the command
    /// that makes one (`head -c 32 /dev/urandom | xxd -p -c 64`). The file
    /// must be a regular file of mode 0600 or stricter: one that group or
    /// others can read or write is refused.
    pub fn with_psk_file(
        self,
        server_id: &str,
        path: impl AsRef<Path>,
    ) -> Result<Self, StoreError> {
        let text = read_secret(path.as_ref(), "PSK")?;
        let key = decode_psk(&text, &format!("the PSK file {}", shown(path.as_ref())))?;
        self.with_psk(server_id, &key)
    }

    /// Accept the bearer tokens of the token file at `path` (see the served
    /// mode in `docs/ferrostore-spec.md`). The same mode rule as a PSK file.
    pub fn with_token_file(mut self, path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let bytes = read_secret(path.as_ref(), "token")?;
        let text = std::str::from_utf8(&bytes).map_err(|_| {
            backend(format!(
                "the token file {} is not UTF-8",
                crate::private::shown_path(path.as_ref())
            ))
        })?;
        self.tokens = Some(TokenTable::parse(text)?);
        Ok(self)
    }

    /// Serve `wss://`: TLS with the certificate chain in the PEM file
    /// `cert_chain` (leaf first) and the private key in the PEM file `key`
    /// (PKCS#8, PKCS#1 or SEC1). The key file is a secret file, under the
    /// same rules as a PSK file (mode 0600 or stricter, owned by this uid or
    /// root, no FIFO). TLS 1.3 is preferred and TLS 1.2 allowed.
    pub fn with_tls_files(
        mut self,
        cert_chain: impl AsRef<Path>,
        key: impl AsRef<Path>,
    ) -> Result<Self, StoreError> {
        self.tls = Some(ServerTls::from_pem_files(
            cert_chain.as_ref(),
            key.as_ref(),
        )?);
        Ok(self)
    }

    /// Let plain `ws://` bind an address of one of these interfaces (by
    /// name: `["tailscale0"]`) as well as loopback. The bound address must
    /// be one of the named interface's own; `0.0.0.0` and `::` are refused
    /// whatever is named. Authentication is still required there.
    #[must_use]
    pub fn allow_plaintext_on<I, S>(mut self, interfaces: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allow_plaintext_on = interfaces.into_iter().map(Into::into).collect();
        self
    }

    /// `true` when a TCP client has a way in.
    pub(super) fn any_method(&self) -> bool {
        self.psk.is_some() || self.tokens.is_some()
    }

    pub(super) fn has_tokens(&self) -> bool {
        self.tokens.is_some()
    }

    /// The TLS configuration, when the server serves `wss://`.
    pub(super) fn tls(&self) -> Option<&ServerTls> {
        self.tls.as_ref()
    }

    /// `true` when this names anything only a TCP bind uses beyond the
    /// token file: TLS or `allow_plaintext_on`.
    pub(super) fn has_tcp_transport(&self) -> bool {
        self.tls.is_some() || !self.allow_plaintext_on.is_empty()
    }

    /// `Ok` when `addr` may be bound: anything with TLS; without it, see
    /// [`bind::check_plaintext_bind`]. `Some(name)`: a plaintext bind on an
    /// address of the named interface `name`, which the listener must be
    /// pinned to (`SO_BINDTODEVICE`, [`bind::listen`]).
    pub(super) fn check_bind(
        &self,
        addr: std::net::SocketAddr,
    ) -> Result<Option<String>, StoreError> {
        if self.tls.is_some() {
            return Ok(None);
        }
        // The unspecified address, or nothing named, needs no interface
        // list. A loopback address does when an interface is named: `lo`
        // named pins the listener to `lo`.
        if addr.ip().is_unspecified() || self.allow_plaintext_on.is_empty() {
            return bind::check_plaintext_bind(addr, &self.allow_plaintext_on, &[]);
        }
        let interfaces = bind::interfaces()
            .map_err(|err| backend(format!("list network interfaces (getifaddrs): {err}")))?;
        bind::check_plaintext_bind(addr, &self.allow_plaintext_on, &interfaces)
    }
}

/// A secret file's bytes, refused when others can reach it. One trailing
/// `\n` (or `\r\n`) is dropped.
fn read_secret(path: &Path, what: &str) -> Result<Vec<u8>, StoreError> {
    let mut bytes = crate::private::read_secret_file(path, MAX_SECRET_FILE)
        .map_err(|err| backend(format!("read the {what} file {}: {err}", shown(path))))?;
    strip_newline(&mut bytes);
    Ok(bytes)
}

fn shown(path: &Path) -> String {
    crate::private::shown_path(path)
}

/// Drop one trailing `\n` or `\r\n`.
fn strip_newline(bytes: &mut Vec<u8>) {
    if bytes.ends_with(b"\n") {
        bytes.pop();
        if bytes.ends_with(b"\r") {
            bytes.pop();
        }
    }
}

/// The key a PSK file or variable holds: [`MIN_PSK_HEX`] or more hex digits
/// (either case), an even number, nothing else (its one trailing newline is
/// already dropped). `source` names where it came from; the error repeats
/// nothing of the text.
pub(super) fn decode_psk(text: &[u8], source: &str) -> Result<Vec<u8>, StoreError> {
    fn digit(byte: u8) -> Option<u8> {
        match byte {
            b'0'..=b'9' => Some(byte - b'0'),
            b'a'..=b'f' => Some(byte - b'a' + 10),
            b'A'..=b'F' => Some(byte - b'A' + 10),
            _ => None,
        }
    }
    let refused = || {
        backend(format!(
            "{source} is not a FerroStore PSK: it holds {MIN_PSK_HEX} or more hex digits \
             (32 or more random bytes) and nothing else; make one with `{MAKE_PSK}`"
        ))
    };
    if text.len() < MIN_PSK_HEX || !text.len().is_multiple_of(2) {
        return Err(refused());
    }
    let mut key = Vec::with_capacity(text.len() / 2);
    for pair in text.as_chunks::<2>().0 {
        let (Some(high), Some(low)) = (digit(pair[0]), digit(pair[1])) else {
            return Err(refused());
        };
        key.push((high << 4) | low);
    }
    Ok(key)
}

// ---- the server's side of the handshake ----------------------------------------------

/// What the server knows of a peer before it reads from it.
pub(super) enum Peer {
    /// A Unix peer, and its uid when the kernel told it.
    Unix(Option<u32>),
    Tcp,
}

/// The credential a `ws://` upgrade carried.
pub(super) enum Credential {
    None,
    Bearer(String),
    /// An `Authorization` header that is not one bearer token.
    Invalid,
}

/// The credential of an upgrade whose `Authorization` header values are
/// `values`: none, one `Bearer <token>`, or anything else, which is invalid.
pub(super) fn credential(values: &[String]) -> Credential {
    match values {
        [] => Credential::None,
        [value] => {
            let value = value.trim();
            match value.split_once(' ') {
                Some((scheme, token))
                    if scheme.eq_ignore_ascii_case("bearer")
                        && check_token(token.trim()).is_ok() =>
                {
                    Credential::Bearer(token.trim().to_string())
                }
                _ => Credential::Invalid,
            }
        }
        _ => Credential::Invalid,
    }
}

/// `true` when a Unix peer of uid `peer` may connect to a server running
/// as `own`: only the same uid, and never an unknown one.
pub(super) fn peer_allowed(peer: Option<u32>, own: u32) -> bool {
    peer == Some(own)
}

/// This process's effective uid: the uid a Unix peer must have.
pub(super) fn own_uid() -> u32 {
    // SAFETY: plain syscall without arguments; it cannot fail.
    unsafe { libc::geteuid() }
}

/// The uid of the process at the other end of `stream`.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub(super) fn peer_uid(stream: &std::os::unix::net::UnixStream) -> std::io::Result<u32> {
    use std::os::fd::AsRawFd;
    // SAFETY: `ucred` is plain old data; all-zero is a valid value.
    let mut cred: libc::ucred = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::ucred>();
    let mut len = libc::socklen_t::try_from(size)
        .map_err(|_| std::io::Error::other("ucred does not fit a socklen_t"))?;
    // SAFETY: `cred` is a writable `ucred` and `len` is its size.
    let rc = unsafe {
        libc::getsockopt(
            stream.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            (&raw mut cred).cast::<libc::c_void>(),
            &raw mut len,
        )
    };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    if usize::try_from(len).ok() != Some(size) {
        return Err(std::io::Error::other("SO_PEERCRED answered a short ucred"));
    }
    Ok(cred.uid)
}

/// The uid of the process at the other end of `stream`.
#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "dragonfly"
))]
pub(super) fn peer_uid(stream: &std::os::unix::net::UnixStream) -> std::io::Result<u32> {
    use std::os::fd::AsRawFd;
    let mut uid: libc::uid_t = 0;
    let mut gid: libc::gid_t = 0;
    // SAFETY: `uid` and `gid` are writable.
    if unsafe { libc::getpeereid(stream.as_raw_fd(), &raw mut uid, &raw mut gid) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(uid)
}

/// No way to learn the peer here: every Unix peer is refused.
#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "dragonfly"
)))]
pub(super) fn peer_uid(_: &std::os::unix::net::UnixStream) -> std::io::Result<u32> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "no peer credentials on this platform",
    ))
}

/// Authenticate one connection before any request is read: `Some` with
/// its scope and principal once it passed (the server has sent `ok`),
/// `None` once it was refused (the server has waited [`AUTH_FAIL_DELAY`],
/// sent the failure frame and shut the connection down).
pub(super) fn serve_handshake(
    read: &mut dyn JsonIo,
    write: &mut dyn JsonIo,
    peer: &Peer,
    credential: Credential,
    auth: &ServerAuth,
    own: u32,
) -> Option<(Scope, Principal)> {
    let outcome = match peer {
        Peer::Unix(uid) => {
            if peer_allowed(*uid, own) {
                let principal = Principal::from(format!("unix:{own}"));
                match &auth.psk {
                    None => send_ok(write, Scope::Admin, None),
                    Some(psk) => psk_exchange(read, write, Some(psk)),
                }
                .map(|scope| (scope, principal))
            } else {
                warn_peer(*uid, own);
                Err(())
            }
        }
        Peer::Tcp => match credential {
            Credential::Bearer(token) => {
                match auth.tokens.as_ref().and_then(|table| table.lookup(&token)) {
                    Some((scope, principal)) => {
                        send_ok(write, scope, None).map(|scope| (scope, principal))
                    }
                    None => Err(()),
                }
            }
            Credential::Invalid => Err(()),
            Credential::None => psk_exchange(read, write, auth.psk.as_ref())
                .map(|scope| (scope, Principal::from("psk"))),
        },
    };
    match outcome {
        Ok(authed) => Some(authed),
        Err(()) => {
            refuse(write, true);
            None
        }
    }
}

/// Wait [`AUTH_FAIL_DELAY`], send the failure frame when `frame` (a
/// transport that has not finished its own handshake has no frames), and
/// shut the connection down.
pub(super) fn refuse(write: &mut dyn JsonIo, frame: bool) {
    std::thread::sleep(AUTH_FAIL_DELAY);
    if frame {
        let _ = write.write_value(&json!({"auth": "failed", "error": FAILED_FRAME_ERROR}));
    }
    write.shutdown_both();
}

fn send_ok(write: &mut dyn JsonIo, scope: Scope, proof: Option<&[u8; 32]>) -> Result<Scope, ()> {
    let mut frame = json!({"auth": "ok", "scope": scope.name()});
    if let Some(proof) = proof {
        frame["proof"] = Value::String(hex(proof));
    }
    write.write_value(&frame).map_err(|_| ())?;
    Ok(scope)
}

/// The server's half of the mutual PSK challenge-response. A server with no
/// PSK still sends its challenge, and refuses every answer.
///
/// Over TLS the challenge names version 2 and both proofs cover the
/// connection's channel binding (spec §6.3); a TLS connection whose binding
/// cannot be read is refused, never served unbound.
fn psk_exchange(
    read: &mut dyn JsonIo,
    write: &mut dyn JsonIo,
    psk: Option<&Psk>,
) -> Result<Scope, ()> {
    let binding = read.channel_binding().map_err(|_| ())?;
    let version = if binding.is_some() {
        AUTH_VERSION_TLS
    } else {
        AUTH_VERSION
    };
    let ns = random_nonce().map_err(|_| ())?;
    write
        .write_value(&json!({"auth": "challenge", "v": version, "ns": hex(&ns)}))
        .map_err(|_| ())?;
    let hello = match read.read_value(PRE_AUTH_FRAME_BYTES) {
        Ok(Incoming::Value(hello)) => hello,
        _ => return Err(()),
    };
    let (nc, proof) = parse_hello(&hello).ok_or(())?;
    let psk = psk.ok_or(())?;
    // An echo of the server's own nonce is never a client's.
    if nc == ns {
        return Err(());
    }
    if !psk.verify_client(&ns, &nc, binding.as_ref(), &proof) {
        return Err(());
    }
    send_ok(
        write,
        Scope::Admin,
        Some(&psk.server_proof(&nc, &ns, binding.as_ref())),
    )
}

/// `{"auth": "psk", "nc": <64 hex>, "proof": <64 hex>}`, and nothing else.
fn parse_hello(hello: &Value) -> Option<(Nonce, [u8; 32])> {
    let object = hello.as_object()?;
    if object.len() != 3 || object.get("auth")?.as_str()? != "psk" {
        return None;
    }
    let nc = unhex32(object.get("nc")?.as_str()?)?;
    let proof = unhex32(object.get("proof")?.as_str()?)?;
    Some((nc, proof))
}

/// Warnings this process recorded instead of printing (`cfg(test)` only).
#[cfg(test)]
static WARNINGS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

#[cfg(test)]
pub(super) fn warnings() -> Vec<String> {
    WARNINGS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

fn warn_peer(uid: Option<u32>, own: u32) {
    let peer = uid.map_or_else(|| "an unknown uid".to_string(), |uid| format!("uid {uid}"));
    let text = format!(
        "warning: FerroStore refused a Unix connection from {peer}: the server runs as uid {own}"
    );
    #[cfg(test)]
    WARNINGS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(text);
    #[cfg(not(test))]
    eprintln!("{text}");
}

// ---- the client's side ----------------------------------------------------------------

/// Where a client secret comes from. Never the URI.
#[derive(Clone)]
enum Secret {
    File(PathBuf),
    Env(String),
    Value(Vec<u8>),
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Secret::File(path) => f.debug_tuple("File").field(path).finish(),
            Secret::Env(name) => f.debug_tuple("Env").field(name).finish(),
            Secret::Value(_) => f.write_str("Value(<redacted>)"),
        }
    }
}

impl Secret {
    fn read(&self, what: &str) -> Result<Vec<u8>, StoreError> {
        match self {
            Secret::File(path) => read_secret(path, what),
            Secret::Env(name) => {
                use std::os::unix::ffi::OsStringExt;
                let mut bytes = std::env::var_os(name)
                    .map(OsStringExt::into_vec)
                    .ok_or_else(|| backend(format!("the {what} variable `{name}` is not set")))?;
                strip_newline(&mut bytes);
                Ok(bytes)
            }
            Secret::Value(bytes) => Ok(bytes.clone()),
        }
    }
}

#[derive(Clone, Debug, Default)]
enum ClientAuth {
    #[default]
    None,
    Psk {
        server_id: String,
        secret: Secret,
    },
    Bearer(Secret),
}

/// How [`FerroStore::connect_with`](super::FerroStore::connect_with)
/// authenticates (spec §6.2).
///
/// The default is no credential: a Unix socket whose server checks the
/// peer's uid alone. A PSK proves both sides to each other, over a Unix
/// socket or `ws://`; the client refuses a server that cannot prove the same
/// PSK under the `server_id` it expects. A bearer token is for `ws://` and `wss://`, sent
/// in the upgrade's `Authorization` header. A secret is read from a file
/// (mode 0600 or stricter) or an environment variable at connect time.
///
/// A PSK file or variable holds 64 or more hex digits, decoded to the key;
/// [`Self::psk`] takes the key's own bytes (32 or more).
///
/// Over plain `ws://` to an address that is not loopback, the client sends
/// no bearer token and runs no PSK handshake unless
/// [`Self::allow_plaintext`] is set: a token travels in clear there, and a
/// PSK handshake can be relayed. `wss://` is the remote path (spec §6.3):
/// there a credential goes to any host, inside TLS, and the PSK proofs are
/// bound to the TLS session. A bearer token proves the client alone; the
/// client knows the server only by TLS, so use a token over `wss://` (or
/// to loopback, or an allowed plaintext interface on a network you trust).
///
/// A `wss://` client checks the server's certificate against the bundled
/// Mozilla root certificates (`webpki-roots`) by default: they are compiled
/// into the binary, and the OS trust store is not read. [`Self::ca_file`]
/// trusts the CA in one PEM file instead (an enterprise or private CA), and [`Self::pin_spki_sha256`] trusts one server key. A URI never
/// sets these.
#[derive(Clone, Debug, Default)]
pub struct ConnectOptions {
    auth: ClientAuth,
    allow_plaintext: bool,
    ca_file: Option<PathBuf>,
    pin: Option<String>,
}

impl ConnectOptions {
    /// No credential.
    pub fn new() -> Self {
        Self::default()
    }

    /// The PSK in the file at `path`, for the server `server_id`.
    pub fn psk_file(server_id: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        Self::with(ClientAuth::Psk {
            server_id: server_id.into(),
            secret: Secret::File(path.into()),
        })
    }

    /// The PSK in the environment variable `var`, for the server `server_id`.
    pub fn psk_env(server_id: impl Into<String>, var: impl Into<String>) -> Self {
        Self::with(ClientAuth::Psk {
            server_id: server_id.into(),
            secret: Secret::Env(var.into()),
        })
    }

    /// The PSK `psk` itself, for the server `server_id`.
    pub fn psk(server_id: impl Into<String>, psk: impl Into<Vec<u8>>) -> Self {
        Self::with(ClientAuth::Psk {
            server_id: server_id.into(),
            secret: Secret::Value(psk.into()),
        })
    }

    /// The bearer token on the first line of the file at `path`.
    pub fn token_file(path: impl Into<PathBuf>) -> Self {
        Self::with(ClientAuth::Bearer(Secret::File(path.into())))
    }

    /// The bearer token in the environment variable `var`.
    pub fn token_env(var: impl Into<String>) -> Self {
        Self::with(ClientAuth::Bearer(Secret::Env(var.into())))
    }

    /// The bearer token `token` itself.
    pub fn token(token: impl Into<String>) -> Self {
        Self::with(ClientAuth::Bearer(Secret::Value(token.into().into_bytes())))
    }

    fn with(auth: ClientAuth) -> Self {
        Self {
            auth,
            ..Self::default()
        }
    }

    /// Over `wss://`, trust only the CA certificates in the PEM file at
    /// `path` (not the bundled Mozilla roots). Use it for a server whose
    /// certificate comes from an enterprise or private CA: the OS trust
    /// store is never read. The server's chain must reach one of
    /// them, be valid now, and name the URI's host.
    #[must_use]
    pub fn ca_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.ca_file = Some(path.into());
        self
    }

    /// Over `wss://`, trust the server whose leaf certificate's
    /// SubjectPublicKeyInfo has this SHA-256 (64 hex digits, `:` between
    /// bytes allowed), and no CA. The handshake must still be signed by that
    /// key; the certificate's chain, name and dates are not checked (the
    /// pinned key is the server's identity). Make a pin with
    /// `openssl x509 -in cert.pem -pubkey -noout | openssl pkey -pubin -outform DER | sha256sum`.
    #[must_use]
    pub fn pin_spki_sha256(mut self, hex: impl Into<String>) -> Self {
        self.pin = Some(hex.into());
        self
    }

    /// `true` when these options name a TLS verification.
    pub(super) fn has_tls_options(&self) -> bool {
        self.ca_file.is_some() || self.pin.is_some()
    }

    /// How a `wss://` connection checks the server: the bundled Mozilla roots,
    /// the CA file, or the pin; both named is an error.
    pub(super) fn verify(&self) -> Result<Verify, StoreError> {
        match (&self.ca_file, &self.pin) {
            (None, None) => Ok(Verify::BundledRoots),
            (Some(path), None) => Ok(Verify::CaFile(path.clone())),
            (None, Some(pin)) => Ok(Verify::Pin(tls::parse_pin(pin)?)),
            (Some(_), Some(_)) => Err(backend(
                "ConnectOptions names both ca_file and pin_spki_sha256: choose one way to \
                 check the server",
            )),
        }
    }

    /// Let a credential go over plain `ws://` to an address that is not
    /// loopback (default: refused). Only these options can allow it, never
    /// a URI.
    #[must_use]
    pub fn allow_plaintext(mut self, allow: bool) -> Self {
        self.allow_plaintext = allow;
        self
    }

    /// `true` when [`Self::allow_plaintext`] is set.
    pub(super) fn plaintext_allowed(&self) -> bool {
        self.allow_plaintext
    }

    /// `true` when these options name a credential.
    pub(super) fn has_auth(&self) -> bool {
        !matches!(self.auth, ClientAuth::None)
    }

    /// The credential, read now from its file or variable.
    pub(super) fn resolve(&self) -> Result<ClientCred, StoreError> {
        match &self.auth {
            ClientAuth::None => Ok(ClientCred::None),
            ClientAuth::Psk { server_id, secret } => {
                let bytes = secret.read("PSK")?;
                let key = match secret {
                    Secret::Value(_) => bytes,
                    Secret::File(path) => {
                        decode_psk(&bytes, &format!("the PSK file {}", shown(path)))?
                    }
                    Secret::Env(name) => decode_psk(&bytes, &format!("the PSK variable `{name}`"))?,
                };
                Ok(ClientCred::Psk(Psk::new(server_id, &key)?))
            }
            ClientAuth::Bearer(secret) => {
                let bytes = secret.read("token")?;
                let text = String::from_utf8(bytes)
                    .map_err(|_| backend("a FerroStore bearer token is not UTF-8"))?;
                // A file's first line; a variable's or a value's whole text.
                // Either trimmed, then checked: a line break left inside is
                // refused, never sent.
                let token = match secret {
                    Secret::File(_) => text.lines().next().unwrap_or(""),
                    Secret::Env(_) | Secret::Value(_) => text.as_str(),
                }
                .trim()
                .to_string();
                check_token(&token)?;
                Ok(ClientCred::Bearer(token))
            }
        }
    }
}

/// A client credential, read and checked.
pub(super) enum ClientCred {
    None,
    Psk(Psk),
    Bearer(String),
}

impl ClientCred {
    /// The token to send in the `ws://` upgrade.
    pub(super) fn bearer(&self) -> Option<&str> {
        match self {
            ClientCred::Bearer(token) => Some(token),
            _ => None,
        }
    }

    /// `Err` when this credential would go over plain `ws://` to `peer`, an
    /// address that is not loopback, and `allowed` (the options'
    /// [`ConnectOptions::allow_plaintext`]) is not set. Checked after the
    /// TCP connect and before the upgrade is written.
    pub(super) fn check_plaintext(
        &self,
        peer: SocketAddr,
        allowed: bool,
    ) -> Result<(), StoreError> {
        let what = match self {
            ClientCred::None => return Ok(()),
            ClientCred::Bearer(_) => "a bearer token",
            ClientCred::Psk(_) => "a PSK handshake",
        };
        if allowed || is_loopback(peer.ip()) {
            return Ok(());
        }
        Err(backend(format!(
            "refused to send {what} over plain ws:// to {peer}, which is not a loopback \
             address: use wss:// (spec §6.3), or set ConnectOptions::allow_plaintext(true)"
        )))
    }
}

/// `true` for a loopback address, an IPv4-mapped one included.
pub(super) fn is_loopback(ip: IpAddr) -> bool {
    #[cfg(test)]
    if LOOPBACK_IS_REMOTE.with(std::cell::Cell::get) {
        return false;
    }
    bind::ip_is_loopback(ip)
}

#[cfg(test)]
thread_local! {
    /// Set by a test on its thread: every address is taken to be remote, so
    /// the plaintext rule can be tested against a loopback server.
    pub(super) static LOOPBACK_IS_REMOTE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The auth parameters a `ws://` URI's query may name. No other parameter
/// is accepted, and no secret: a parameter that would carry one is refused.
///
/// A URI may come from a less trusted place than the code that connects, so
/// it names a credential only by an environment variable whose name starts
/// with [`URI_ENV_PREFIX`] (`token_env=FERROSTORE_TOKEN`), never a file and
/// never another variable: a URI cannot make the client send an arbitrary
/// secret to the URI's host. A file is named in [`ConnectOptions`].
///
/// `rest` is the URI after `ws://`. Returns it without the query, and the
/// options the query named (none when it had no query). Values are not
/// percent-decoded.
pub(super) fn split_uri_query(rest: &str) -> Result<(String, ConnectOptions), StoreError> {
    let Some((base, query)) = rest.split_once('?') else {
        return Ok((rest.to_string(), ConnectOptions::new()));
    };
    let mut psk_env = None;
    let mut server_id = None;
    let mut token_env = None;
    for pair in query.split('&').filter(|pair| !pair.is_empty()) {
        let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
        let slot = match key {
            "psk_env" => &mut psk_env,
            "server_id" => &mut server_id,
            "token_env" => &mut token_env,
            "psk_file" | "token_file" => {
                return Err(backend(format!(
                    "a FerroStore URI names no file: `{key}` is refused; name the file in \
                     ConnectOptions, or a `{URI_ENV_PREFIX}*` variable with `psk_env` or \
                     `token_env`"
                )));
            }
            "psk" | "token" | "secret" | "password" | "key" => {
                return Err(backend(format!(
                    "a secret never goes in a FerroStore URI: name its `{URI_ENV_PREFIX}*` \
                     variable with `psk_env` or `token_env`, or pass it in ConnectOptions, \
                     instead of `{key}`"
                )));
            }
            other => {
                return Err(backend(format!(
                    "unknown FerroStore URI parameter `{other}`: the query takes `psk_env`, \
                     `server_id` and `token_env`"
                )));
            }
        };
        if value.is_empty() || slot.is_some() {
            return Err(backend(format!(
                "the FerroStore URI parameter `{key}` is empty or given twice"
            )));
        }
        if key != "server_id" && !uri_env_name(value) {
            return Err(backend(format!(
                "the FerroStore URI parameter `{key}` names a variable that does not start \
                 with `{URI_ENV_PREFIX}`: a URI may name only those (letters, digits, `_`)"
            )));
        }
        *slot = Some(value.to_string());
    }
    if psk_env.is_some() && token_env.is_some() {
        return Err(backend(
            "a FerroStore URI names one credential: `psk_env` or `token_env`",
        ));
    }
    let options = match (psk_env, token_env) {
        (Some(var), None) => ConnectOptions::psk_env(server_id.ok_or_else(missing_server_id)?, var),
        (None, Some(var)) if server_id.is_none() => ConnectOptions::token_env(var),
        _ if server_id.is_some() => {
            return Err(backend(
                "the FerroStore URI parameter `server_id` goes with `psk_env`",
            ));
        }
        _ => ConnectOptions::new(),
    };
    Ok((base.to_string(), options))
}

/// `FERROSTORE_` and one or more ASCII letters, digits or `_`.
fn uri_env_name(name: &str) -> bool {
    name.strip_prefix(URI_ENV_PREFIX).is_some_and(|rest| {
        !rest.is_empty()
            && rest
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    })
}

fn missing_server_id() -> StoreError {
    backend("a FerroStore PSK needs `server_id`: the name of the server it expects")
}

/// The client's half of the handshake, before any request: read the
/// server's first frame and answer it as `cred` allows. `Ok` once the
/// server accepted the client and, for a PSK, proved itself.
pub(super) fn client_handshake(
    read: &mut dyn JsonIo,
    write: &mut dyn JsonIo,
    cred: &ClientCred,
) -> Result<Scope, StoreError> {
    let first = recv_auth(read)?;
    match first.get("auth").and_then(Value::as_str) {
        Some("ok") => match cred {
            ClientCred::Psk(psk) => Err(unproven(psk)),
            ClientCred::None | ClientCred::Bearer(_) => ok_scope(&first),
        },
        Some("challenge") => {
            let ClientCred::Psk(psk) = cred else {
                return Err(backend(
                    "the FerroStore server requires authentication: connect with a PSK or, \
                     over ws://, a bearer token (ConnectOptions)",
                ));
            };
            // Each side takes the version from its own transport: a client
            // over TLS never answers an unbound (version 1) challenge.
            let binding = read.channel_binding()?;
            let version = if binding.is_some() {
                AUTH_VERSION_TLS
            } else {
                AUTH_VERSION
            };
            if first.get("v").and_then(Value::as_u64) != Some(version) {
                return Err(backend(
                    "the FerroStore server speaks another authentication version",
                ));
            }
            let ns = first
                .get("ns")
                .and_then(Value::as_str)
                .and_then(unhex32)
                .ok_or_else(|| backend("the FerroStore server's challenge is malformed"))?;
            let nc = random_nonce()?;
            write.write_value(&json!({
                "auth": "psk",
                "nc": hex(&nc),
                "proof": hex(&psk.client_proof(&ns, &nc, binding.as_ref())),
            }))?;
            let answer = recv_auth(read)?;
            match answer.get("auth").and_then(Value::as_str) {
                Some("ok") => {
                    let proof = answer
                        .get("proof")
                        .and_then(Value::as_str)
                        .and_then(unhex32)
                        .ok_or_else(|| unproven(psk))?;
                    if !psk.verify_server(&nc, &ns, binding.as_ref(), &proof) {
                        return Err(unproven(psk));
                    }
                    ok_scope(&answer)
                }
                Some("failed") => Err(backend(AUTH_FAILED)),
                _ => Err(backend(
                    "the FerroStore server answered the PSK hello with no outcome",
                )),
            }
        }
        Some("failed") => Err(backend(AUTH_FAILED)),
        _ => Err(backend(
            "the FerroStore server sent no authentication frame first",
        )),
    }
}

fn unproven(psk: &Psk) -> StoreError {
    backend(format!(
        "the FerroStore server did not prove the PSK for `{}`: it is another server, or its \
         PSK differs; refused",
        psk.server_id
    ))
}

fn ok_scope(frame: &Value) -> Result<Scope, StoreError> {
    frame
        .get("scope")
        .and_then(Value::as_str)
        .and_then(Scope::parse)
        .ok_or_else(|| backend("the FerroStore server accepted the connection with no scope"))
}

fn recv_auth(read: &mut dyn JsonIo) -> Result<Value, StoreError> {
    match read.read_value(PRE_AUTH_FRAME_BYTES) {
        Ok(Incoming::Value(frame)) => Ok(frame),
        Ok(Incoming::Ping(_)) | Ok(Incoming::Close) | Err(_) => Err(backend(
            "the FerroStore connection closed during authentication",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trips_and_refuses_anything_but_64_lowercase_digits() {
        let bytes: [u8; 32] = std::array::from_fn(|i| u8::try_from(i * 8).unwrap());
        assert_eq!(unhex32(&hex(&bytes)), Some(bytes));
        assert_eq!(unhex32(&hex(&bytes).to_uppercase()), None);
        assert_eq!(unhex32(&hex(&bytes[..31])), None);
        assert_eq!(unhex32(&format!("{}0", hex(&bytes))), None);
        assert_eq!(unhex32(&"g".repeat(64)), None);
    }

    #[test]
    fn the_mac_input_is_fixed_length_but_for_the_length_prefixed_server_id() {
        let psk = Psk::new("ab", &[1u8; 32]).unwrap();
        let labels = (CLIENT_LABEL, CLIENT_LABEL_TLS);
        let input = psk.input(labels, &[2u8; 32], &[3u8; 32], None);
        assert_eq!(input.len(), 17 + 1 + 2 + 64);
        assert_eq!(&input[..17], b"fl-store/1/client");
        assert_eq!(input[17], 2);
        assert_eq!(&input[18..20], b"ab");
        assert_ne!(CLIENT_LABEL, SERVER_LABEL);
        // A different split of the same bytes between server_id and nonce
        // is a different input.
        let other = Psk::new("a", &[1u8; 32]).unwrap();
        assert_ne!(
            other.client_proof(&[2u8; 32], &[3u8; 32], None),
            psk.client_proof(&[2u8; 32], &[3u8; 32], None)
        );
        // Over TLS: the version-2 label, and the 32-byte binding at the end.
        let bound = psk.input(labels, &[2u8; 32], &[3u8; 32], Some(&[9u8; 32]));
        assert_eq!(bound.len(), 17 + 1 + 2 + 64 + 32);
        assert_eq!(&bound[..17], b"fl-store/2/client");
        assert_eq!(&bound[17..17 + 1 + 2 + 64], &input[17..]);
        assert_eq!(&bound[bound.len() - 32..], &[9u8; 32]);
        let server = psk.input(
            (SERVER_LABEL, SERVER_LABEL_TLS),
            &[2u8; 32],
            &[3u8; 32],
            Some(&[9u8; 32]),
        );
        assert_eq!(&server[..17], b"fl-store/2/server");
        for label in [CLIENT_LABEL, SERVER_LABEL] {
            assert_ne!(label, CLIENT_LABEL_TLS);
            assert_ne!(label, SERVER_LABEL_TLS);
        }
        // Another binding, or none, is another proof.
        let proof = psk.client_proof(&[2u8; 32], &[3u8; 32], Some(&[9u8; 32]));
        assert!(psk.verify_client(&[2u8; 32], &[3u8; 32], Some(&[9u8; 32]), &proof));
        assert!(!psk.verify_client(&[2u8; 32], &[3u8; 32], Some(&[8u8; 32]), &proof));
        assert!(!psk.verify_client(&[2u8; 32], &[3u8; 32], None, &proof));
    }

    #[test]
    fn a_bearer_header_is_one_token_of_visible_ascii() {
        assert!(matches!(credential(&[]), Credential::None));
        assert!(matches!(
            credential(&["Bearer abc".to_string()]),
            Credential::Bearer(ref t) if t == "abc"
        ));
        assert!(matches!(
            credential(&["bearer  abc ".to_string()]),
            Credential::Bearer(ref t) if t == "abc"
        ));
        for bad in ["Basic abc", "Bearer", "Bearer a b", "abc", "Bearer "] {
            assert!(
                matches!(credential(&[bad.to_string()]), Credential::Invalid),
                "{bad}"
            );
        }
        assert!(matches!(
            credential(&["Bearer a".to_string(), "Bearer b".to_string()]),
            Credential::Invalid
        ));
    }

    #[test]
    fn a_token_lookup_compares_hashes_and_holds_no_token() {
        let line = |token: &str, scope: &str| {
            format!("{} {scope}\n", hex(&Sha256::digest(token.as_bytes())))
        };
        let table =
            TokenTable::parse(&format!("{}{}", line("r", "read"), line("w", "write"))).unwrap();
        let scope = |token: &str| table.lookup(token).map(|(scope, _)| scope);
        assert_eq!(scope("r"), Some(Scope::Read));
        assert_eq!(scope("w"), Some(Scope::Write));
        assert_eq!(scope("x"), None);
        assert_eq!(scope(""), None);
        // The principal is the token's hash, never the token.
        let (_, principal) = table.lookup("r").unwrap();
        assert_eq!(&*principal, format!("token:{}", hex(&Sha256::digest(b"r"))));
        assert_ne!(principal, table.lookup("w").unwrap().1);
        assert!(equal_32(&[5u8; 32], &[5u8; 32]));
        let mut other = [5u8; 32];
        other[31] = 6;
        assert!(!equal_32(&[5u8; 32], &other));
    }
}
