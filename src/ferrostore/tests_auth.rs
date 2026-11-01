//! Spec §6.2: every served connection authenticates before any operation,
//! and every method needs a scope.
//!
//! Unix: `SO_PEERCRED` (the peer's uid is the server's), and a PSK when the
//! bind configures one. TCP (`ws://`, and `wss://` where a test runs over
//! both [`NETS`]): the mutual PSK challenge-response, or a bearer token in
//! the upgrade's `Authorization` header.

use super::auth::{self, ClientCred, Psk, Scope};
use super::server::{self, read_frame, write_frame};
use super::tests_tls::{test_ca, with_test_tls};
use super::tls::{Binding, Verify, Wire};
use super::ws;
use super::{ConnectOptions, FerroStore, FerroStoreServer, Inner, ServerAuth};
use fl_core::StoreError;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

/// The PSK every test server and client shares: 32 bytes.
pub(super) const TEST_PSK: &[u8] = b"0123456789abcdef0123456789abcdef";
/// The name test servers prove themselves under.
pub(super) const TEST_SERVER_ID: &str = "test-store";

/// The server auth of a test TCP bind: the test PSK.
pub(super) fn psk_auth() -> ServerAuth {
    ServerAuth::new()
        .with_psk(TEST_SERVER_ID, TEST_PSK)
        .unwrap()
}

/// The client options that match [`psk_auth`].
pub(super) fn psk_options() -> ConnectOptions {
    ConnectOptions::psk(TEST_SERVER_ID, TEST_PSK)
}

/// A client of the `ws://` server at `addr` under the test PSK.
pub(super) fn ws_client(addr: SocketAddr) -> FerroStore {
    FerroStore::connect_with(&format!("ws://{addr}"), &psk_options()).unwrap()
}

/// The TCP transports a test runs over: plain `ws://`, and `wss://` with
/// the test certificate, whose clients trust the test CA alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Net {
    Ws,
    Wss,
}

pub(super) const NETS: [Net; 2] = [Net::Ws, Net::Wss];

impl Net {
    /// The URI of the server at `addr`.
    pub(super) fn url(self, addr: SocketAddr) -> String {
        match self {
            Net::Ws => format!("ws://{addr}"),
            Net::Wss => format!("wss://{addr}"),
        }
    }

    /// `auth`, with the test certificate over `wss://`.
    pub(super) fn auth(self, auth: ServerAuth) -> ServerAuth {
        match self {
            Net::Ws => auth,
            Net::Wss => with_test_tls(auth),
        }
    }

    /// `options`, trusting the test CA over `wss://`.
    pub(super) fn options(self, options: ConnectOptions) -> ConnectOptions {
        match self {
            Net::Ws => options,
            Net::Wss => options.ca_file(test_ca()),
        }
    }

    /// How a raw client checks the server.
    pub(super) fn verify(self) -> Option<Verify> {
        match self {
            Net::Ws => None,
            Net::Wss => Some(Verify::CaFile(test_ca())),
        }
    }

    /// A served test store over this transport under `auth`.
    pub(super) fn serve(self, store: FerroStore, auth: ServerAuth) -> FerroStoreServer {
        FerroStoreServer::bind_tcp("127.0.0.1:0", store, self.auth(auth)).unwrap()
    }

    /// A client of the server at `addr` under the test PSK.
    pub(super) fn psk_client(self, addr: SocketAddr) -> FerroStore {
        FerroStore::connect_with(&self.url(addr), &self.options(psk_options())).unwrap()
    }

    fn token_client(self, addr: SocketAddr, token: &str) -> Result<FerroStore, StoreError> {
        FerroStore::connect_with(&self.url(addr), &self.options(ConnectOptions::token(token)))
    }

    fn token_raw(self, addr: SocketAddr, token: &str) -> server::RemoteClient {
        server::connect_tcp(
            &addr.to_string(),
            &ClientCred::Bearer(token.to_string()),
            false,
            self.verify().as_ref(),
        )
        .unwrap()
    }
}

fn remote(store: &FerroStore) -> &server::RemoteClient {
    match &store.inner {
        Inner::Remote(remote) => remote,
        _ => panic!("not a remote handle"),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn unhex32(text: &str) -> [u8; 32] {
    assert_eq!(text.len(), 64, "{text}");
    let mut out = [0u8; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&text[2 * i..2 * i + 2], 16).unwrap();
    }
    out
}

/// A raw `ws://` or `wss://` connection the test drives frame by frame.
pub(super) struct RawWs {
    pub(super) stream: Wire,
    /// The writing half, kept open with the reading one.
    pub(super) write_half: Wire,
}

impl RawWs {
    pub(super) fn over(net: Net, addr: SocketAddr, bearer: Option<&str>) -> Self {
        let (stream, write) =
            server::dial_wire(&addr.to_string(), bearer, net.verify().as_ref(), |_| Ok(()))
                .unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        Self {
            stream,
            write_half: write,
        }
    }

    /// This connection's TLS channel binding (`None` over `ws://`).
    pub(super) fn binding(&self) -> Option<Binding> {
        self.stream.channel_binding().unwrap()
    }

    pub(super) fn send(&mut self, value: &Value) {
        let bytes = serde_json::to_vec(value).unwrap();
        ws::write_frame(&mut self.write_half, true, 0x1, &bytes).unwrap();
    }

    /// The next JSON frame; `None` once the server closed.
    pub(super) fn recv(&mut self) -> Option<Value> {
        match ws::read_event(&mut self.stream, true, ws::MAX_PAYLOAD) {
            Ok(ws::WsEvent::Data(payload)) => Some(serde_json::from_slice(&payload).unwrap()),
            Ok(ws::WsEvent::Ping(_)) => self.recv(),
            Ok(ws::WsEvent::Close) | Err(_) => None,
        }
    }

    /// The server's challenge nonce. Its version is 1 over `ws://` and 2,
    /// bound to the TLS session, over `wss://`.
    pub(super) fn challenge(&mut self) -> [u8; 32] {
        let frame = self.recv().expect("a challenge");
        assert_eq!(frame["auth"], "challenge", "{frame}");
        let version = if self.binding().is_some() { 2 } else { 1 };
        assert_eq!(frame["v"], version, "{frame}");
        unhex32(frame["ns"].as_str().unwrap())
    }

    /// Send a PSK hello; the server's answer, `None` when it closed.
    pub(super) fn hello(&mut self, nc: &[u8; 32], proof: &[u8]) -> Option<Value> {
        self.hello_value(&json!({"auth": "psk", "nc": hex(nc), "proof": hex(proof)}))
    }

    /// Send `hello` as the answer to the challenge; the server's answer.
    fn hello_value(&mut self, hello: &Value) -> Option<Value> {
        self.send(hello);
        self.recv()
    }

    /// After a refusal: the failure frame, then the close.
    pub(super) fn refused(&mut self, first: Option<Value>) {
        assert_eq!(
            first,
            Some(json!({"auth": "failed", "error": "authentication failed"}))
        );
        assert_eq!(self.recv(), None, "the server closes after refusing");
    }
}

/// A served test store over TCP under `auth`.
fn serve_tcp(store: FerroStore, auth: ServerAuth) -> FerroStoreServer {
    FerroStoreServer::bind_tcp("127.0.0.1:0", store, auth).unwrap()
}

/// A 0600 token file in `dir` holding `lines`.
fn token_file(dir: &Path, lines: &str) -> PathBuf {
    let path = dir.join("tokens");
    std::fs::write(&path, lines).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    path
}

fn sha256_hex(token: &str) -> String {
    hex(&Sha256::digest(token.as_bytes()))
}

const READ_TOKEN: &str = "read-token-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
const WRITE_TOKEN: &str = "write-token-BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB";
const ADMIN_TOKEN: &str = "admin-token-CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC";

fn three_tokens(dir: &Path) -> PathBuf {
    token_file(
        dir,
        &format!(
            "# sha256 scope label\n{} read reader\n\n{} write writer bot\n{} admin ops\n",
            sha256_hex(READ_TOKEN),
            sha256_hex(WRITE_TOKEN),
            sha256_hex(ADMIN_TOKEN)
        ),
    )
}

fn token_client(addr: SocketAddr, token: &str) -> Result<FerroStore, StoreError> {
    FerroStore::connect_with(&format!("ws://{addr}"), &ConnectOptions::token(token))
}

/// A served in-memory store over TCP that accepts `n` write-scope tokens,
/// each its own principal; the tokens.
pub(super) fn token_principals(n: usize) -> (FerroStoreServer, Vec<String>, tempfile::TempDir) {
    token_principals_over(Net::Ws, n)
}

/// [`token_principals`] over `net`.
pub(super) fn token_principals_over(
    net: Net,
    n: usize,
) -> (FerroStoreServer, Vec<String>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let tokens: Vec<String> = (0..n).map(|i| format!("principal-token-{i:04}")).collect();
    let lines: String = tokens
        .iter()
        .map(|token| format!("{} write\n", sha256_hex(token)))
        .collect();
    let path = token_file(dir.path(), &lines);
    let server = net.serve(
        FerroStore::in_memory(),
        ServerAuth::new().with_token_file(&path).unwrap(),
    );
    (server, tokens, dir)
}

/// A raw client of `addr` under `token`, for calls the trait methods do not
/// make.
pub(super) fn token_raw(addr: SocketAddr, token: &str) -> server::RemoteClient {
    server::connect_ws(
        &addr.to_string(),
        &ClientCred::Bearer(token.to_string()),
        false,
    )
    .unwrap()
}

/// A 0600 PSK file in `dir` holding `text`.
fn psk_file(dir: &Path, text: &str) -> PathBuf {
    let path = dir.join("psk");
    std::fs::write(&path, text).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    path
}

fn needs(err: &StoreError, method: &str, scope: &str) {
    let message = err.to_string();
    assert!(
        message.contains(&format!("`{method}` needs the `{scope}` scope")),
        "{message}"
    );
}

// ---- Unix: SO_PEERCRED ---------------------------------------------------------

#[test]
fn the_peer_uid_decision_accepts_the_server_uid_alone() {
    assert!(auth::peer_allowed(Some(1000), 1000));
    assert!(!auth::peer_allowed(Some(1001), 1000));
    assert!(!auth::peer_allowed(Some(0), 1000), "root is another uid");
    assert!(
        !auth::peer_allowed(None, 1000),
        "unknown credentials refuse"
    );

    // A real connection reports this process's uid.
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("p.sock");
    let listener = UnixListener::bind(&sock).unwrap();
    let client = UnixStream::connect(&sock).unwrap();
    let (accepted, _) = listener.accept().unwrap();
    // SAFETY: plain syscall without arguments.
    let own = unsafe { libc::geteuid() };
    assert_eq!(auth::peer_uid(&accepted).unwrap(), own);
    drop(client);
}

#[test]
fn a_unix_peer_of_the_same_uid_is_served_and_another_uid_is_refused_unread() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("s.sock");
    let server = FerroStoreServer::bind_unix(&sock, FerroStore::in_memory()).unwrap();
    let uri = format!("unix://{}", sock.display());

    let same = FerroStore::connect_uri(&uri).unwrap();
    same.put_doc("docs", "a", &json!(1)).unwrap();
    assert_eq!(same.get_doc("docs", "a").unwrap(), Some(json!(1)));
    drop(same);

    // The server now sees every peer as another uid.
    // SAFETY: plain syscall without arguments.
    let other = unsafe { libc::geteuid() }.wrapping_add(1);
    server.set_peer_uid(Some(other));
    let before = server.dispatched();
    let mut raw = UnixStream::connect(&sock).unwrap();
    raw.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    // A request written at once is never read.
    write_frame(
        &mut raw,
        &json!({"id": 1, "method": "put_doc", "params": {"collection": "docs", "id": "b", "doc": 2}}),
    )
    .unwrap();
    let (frame, _) = read_frame(&mut raw, ws::MAX_PAYLOAD).unwrap();
    assert_eq!(
        frame,
        json!({"auth": "failed", "error": "authentication failed"})
    );
    assert!(
        read_frame(&mut raw, ws::MAX_PAYLOAD).is_err(),
        "closed after the refusal"
    );
    assert_eq!(server.dispatched(), before, "nothing ran");
    assert_eq!(server.store().get_doc("docs", "b").unwrap(), None);
    assert!(
        auth::warnings()
            .iter()
            .any(|warning| warning.contains(&format!("uid {other}"))),
        "{:?}",
        auth::warnings()
    );

    let err = FerroStore::connect_uri(&uri).err().unwrap();
    assert!(err.to_string().contains(auth::AUTH_FAILED), "{err}");
    server.set_peer_uid(None);
    FerroStore::connect_uri(&uri).unwrap();
}

#[test]
fn a_unix_bind_with_a_psk_requires_it_as_well_as_the_uid() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("s.sock");
    let _server =
        FerroStoreServer::bind_unix_with(&sock, FerroStore::in_memory(), psk_auth()).unwrap();
    let uri = format!("unix://{}", sock.display());

    let err = FerroStore::connect_uri(&uri).err().unwrap();
    assert!(err.to_string().contains("requires authentication"), "{err}");
    let wrong = ConnectOptions::psk(
        TEST_SERVER_ID,
        b"another-32-byte-pre-shared-key!!".as_slice(),
    );
    let err = FerroStore::connect_with(&uri, &wrong).err().unwrap();
    assert!(err.to_string().contains(auth::AUTH_FAILED), "{err}");

    let client = FerroStore::connect_with(&uri, &psk_options()).unwrap();
    client.put_doc("docs", "a", &json!(1)).unwrap();
    assert_eq!(client.get_doc("docs", "a").unwrap(), Some(json!(1)));

    // A token file has no meaning on a Unix socket.
    let tokens = three_tokens(dir.path());
    let err = FerroStoreServer::bind_unix_with(
        dir.path().join("t.sock"),
        FerroStore::in_memory(),
        ServerAuth::new().with_token_file(&tokens).unwrap(),
    )
    .err()
    .unwrap();
    assert!(err.to_string().contains("ws://"), "{err}");
}

#[test]
fn a_psk_client_refuses_a_unix_server_that_proves_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("s.sock");
    let _server = FerroStoreServer::bind_unix(&sock, FerroStore::in_memory()).unwrap();
    let err = FerroStore::connect_with(&format!("unix://{}", sock.display()), &psk_options())
        .err()
        .unwrap();
    assert!(err.to_string().contains("did not prove"), "{err}");
}

// ---- PSK challenge-response ------------------------------------------------------

#[test]
fn the_psk_handshake_serves_a_client_with_the_psk() {
    for net in NETS {
        let server = net.serve(FerroStore::in_memory(), psk_auth());
        let client = net.psk_client(server.local_addr());
        client.put_doc("docs", "a", &json!({"n": 1})).unwrap();
        assert_eq!(client.get_doc("docs", "a").unwrap(), Some(json!({"n": 1})));
        // PSK connections hold the admin scope.
        remote(&client).call("checkpoint", json!({})).unwrap();
        remote(&client).call("compact", json!({})).unwrap();
    }
}

#[test]
fn the_server_refuses_a_wrong_psk_and_a_wrong_server_id() {
    for net in NETS {
        refuses_a_wrong_psk_and_a_wrong_server_id(net);
    }
}

fn refuses_a_wrong_psk_and_a_wrong_server_id(net: Net) {
    let server = net.serve(FerroStore::in_memory(), psk_auth());
    let url = net.url(server.local_addr());
    let before = server.dispatched();
    for options in [
        ConnectOptions::psk(
            TEST_SERVER_ID,
            b"another-32-byte-pre-shared-key!!".as_slice(),
        ),
        ConnectOptions::psk("some-other-store", TEST_PSK),
        ConnectOptions::new(),
    ] {
        let err = FerroStore::connect_with(&url, &net.options(options))
            .err()
            .unwrap();
        let message = err.to_string();
        assert!(
            message.contains(auth::AUTH_FAILED) || message.contains("requires authentication"),
            "{net:?}: {message}"
        );
    }
    assert_eq!(server.dispatched(), before);
    // A PSK shorter than 32 bytes is refused before any connection.
    assert!(
        ServerAuth::new()
            .with_psk(TEST_SERVER_ID, b"short")
            .is_err()
    );
    assert!(ServerAuth::new().with_psk("", TEST_PSK).is_err());
}

/// How a fake server answers a client's hello, given the nonce it issued.
type Answer = dyn Fn(&Value, &[u8; 32]) -> Value + Send;

/// A fake Unix server that answers the challenge-response with `answer`,
/// given the client's hello and the nonce it issued.
fn fake_server(
    answer: impl Fn(&Value, &[u8; 32]) -> Value + Send + 'static,
) -> (String, thread::JoinHandle<()>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("fake.sock");
    let listener = UnixListener::bind(&sock).unwrap();
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let ns = [7u8; 32];
        write_frame(
            &mut stream,
            &json!({"auth": "challenge", "v": 1, "ns": hex(&ns)}),
        )
        .unwrap();
        let (hello, _) = read_frame(&mut stream, ws::MAX_PAYLOAD).unwrap();
        write_frame(&mut stream, &answer(&hello, &ns)).unwrap();
        // Anything the client sends after a refusal would be read here.
        let _ = read_frame(&mut stream, ws::MAX_PAYLOAD);
    });
    (format!("unix://{}", sock.display()), handle, dir)
}

#[test]
fn the_client_refuses_a_server_that_does_not_hold_the_psk() {
    let wrong = Psk::new(TEST_SERVER_ID, b"another-32-byte-pre-shared-key!!").unwrap();
    let answers: Vec<Box<Answer>> = vec![
        // Proof under another PSK.
        Box::new(move |hello, ns| {
            let nc = unhex32(hello["nc"].as_str().unwrap());
            json!({"auth": "ok", "scope": "admin", "proof": hex(&wrong.server_proof(&nc, ns, None))})
        }),
        // No proof at all.
        Box::new(|_, _| json!({"auth": "ok", "scope": "admin"})),
        // The client's own proof, reflected.
        Box::new(|hello, _| json!({"auth": "ok", "scope": "admin", "proof": hello["proof"]})),
    ];
    for answer in answers {
        let (uri, handle, _dir) = fake_server(answer);
        let err = FerroStore::connect_with(&uri, &psk_options())
            .err()
            .unwrap();
        assert!(
            err.to_string().contains("did not prove"),
            "the client accepted a server without the PSK: {err}"
        );
        handle.join().unwrap();
    }
    // The control: the right proof is accepted by the same fake server.
    let right = Psk::new(TEST_SERVER_ID, TEST_PSK).unwrap();
    let (uri, _handle, _dir) = fake_server(move |hello, ns| {
        let nc = unhex32(hello["nc"].as_str().unwrap());
        json!({"auth": "ok", "scope": "admin", "proof": hex(&right.server_proof(&nc, ns, None))})
    });
    FerroStore::connect_with(&uri, &psk_options()).unwrap();
}

/// The FerroWire W-1 lesson: a proof made in one direction never passes in
/// the other, and no proof outlives the connection whose nonce it covers.
#[test]
fn psk_handshake_is_not_open_to_a_reflection_or_a_replay() {
    for net in NETS {
        not_open_to_a_reflection_or_a_replay(net);
    }
}

fn not_open_to_a_reflection_or_a_replay(net: Net) {
    let server = net.serve(FerroStore::in_memory(), psk_auth());
    let addr = server.local_addr();
    let psk = Psk::new(TEST_SERVER_ID, TEST_PSK).unwrap();
    let before = server.dispatched();

    // An honest connection, recorded.
    let mut honest = RawWs::over(net, addr, None);
    let bound = honest.binding();
    assert_eq!(bound.is_some(), net == Net::Wss);
    let ns_a = honest.challenge();
    let nc_a = [0x11u8; 32];
    let client_a = psk.client_proof(&ns_a, &nc_a, bound.as_ref());
    let ok = honest.hello(&nc_a, &client_a).expect("an answer");
    assert_eq!(ok["auth"], "ok", "{ok}");
    assert_eq!(ok["scope"], "admin", "{ok}");
    let server_a = unhex32(ok["proof"].as_str().unwrap());
    assert!(psk.verify_server(&nc_a, &ns_a, bound.as_ref(), &server_a));

    // Direction separation: the server's proof is no client proof for the
    // transposed transcript (a server that had issued `nc_a`), and the
    // client's proof is no server proof for it.
    assert!(!psk.verify_client(&nc_a, &ns_a, bound.as_ref(), &server_a));
    assert!(!psk.verify_server(&ns_a, &nc_a, bound.as_ref(), &client_a));

    // The server's proof, presented as a client proof over its own nonce.
    let mut reflect = RawWs::over(net, addr, None);
    let ns_b = reflect.challenge();
    let answer = reflect.hello(&ns_a, &server_a);
    reflect.refused(answer);

    // The server's challenge echoed back as the client nonce.
    let mut echo = RawWs::over(net, addr, None);
    let ns_c = echo.challenge();
    let answer = echo.hello(
        &ns_c,
        &psk.client_proof(&ns_c, &ns_c, echo.binding().as_ref()),
    );
    echo.refused(answer);

    // A captured client hello replayed on a new connection: a fresh `ns`.
    let mut replay = RawWs::over(net, addr, None);
    let ns_d = replay.challenge();
    assert_ne!(ns_d, ns_a, "every connection gets a fresh nonce");
    assert_ne!(ns_d, ns_b);
    let answer = replay.hello(&nc_a, &client_a);
    replay.refused(answer);

    assert_eq!(server.dispatched(), before, "nothing ran");
}

// ---- bearer tokens and scopes --------------------------------------------------

#[test]
fn a_bearer_token_gets_its_scope_and_no_more() {
    for net in NETS {
        a_bearer_token_gets_its_scope(net);
    }
}

fn a_bearer_token_gets_its_scope(net: Net) {
    let dir = tempfile::tempdir().unwrap();
    let tokens = three_tokens(dir.path());
    let file = dir.path().join("s.fls");
    let server = net.serve(
        FerroStore::open(&file).unwrap(),
        ServerAuth::new().with_token_file(&tokens).unwrap(),
    );
    let addr = server.local_addr();
    let token_client = |addr: SocketAddr, token: &str| net.token_client(addr, token);

    let admin = token_client(addr, ADMIN_TOKEN).unwrap();
    admin.put_doc("docs", "a", &json!(1)).unwrap();

    // read: reads, but no commit (not even an empty one), put or checkpoint.
    let reader = token_client(addr, READ_TOKEN).unwrap();
    assert_eq!(reader.get_doc("docs", "a").unwrap(), Some(json!(1)));
    let empty_commit = json!({"reads": [], "ops": []});
    let err = remote(&reader)
        .call("commit_txn", empty_commit.clone())
        .unwrap_err();
    needs(&err, "commit_txn", "write");
    let err = reader.put_doc("docs", "b", &json!(2)).unwrap_err();
    needs(&err, "commit_txn", "write");
    let err = remote(&reader)
        .call(
            "put_doc",
            json!({"collection": "docs", "id": "b", "doc": 2}),
        )
        .unwrap_err();
    needs(&err, "put_doc", "write");
    let err = remote(&reader).call("checkpoint", json!({})).unwrap_err();
    needs(&err, "checkpoint", "admin");
    // The connection stays open after a refusal.
    assert_eq!(reader.get_doc("docs", "b").unwrap(), None);
    let reads_mq = reader.subscribe_mq("t");
    drop(reads_mq);

    // write: writes, but no compact, checkpoint or sync apply.
    let writer = token_client(addr, WRITE_TOKEN).unwrap();
    let committed = remote(&writer).call("commit_txn", empty_commit).unwrap();
    assert_eq!(committed["committed"], true, "{committed}");
    writer.put_doc("docs", "b", &json!(2)).unwrap();
    writer.publish_mq("t", "k", &json!({})).unwrap();
    let err = remote(&writer).call("compact", json!({})).unwrap_err();
    needs(&err, "compact", "admin");
    let err = remote(&writer).call("checkpoint", json!({})).unwrap_err();
    needs(&err, "checkpoint", "admin");
    let err = remote(&writer)
        .call("apply_sync_batches", json!({"batches": []}))
        .unwrap_err();
    needs(&err, "apply_sync_batches", "admin");
    assert_eq!(writer.get_doc("docs", "b").unwrap(), Some(json!(2)));

    // admin: all of it.
    remote(&admin).call("checkpoint", json!({})).unwrap();
    remote(&admin).call("compact", json!({})).unwrap();
    assert_eq!(admin.get_doc("docs", "b").unwrap(), Some(json!(2)));

    // An unknown token is refused at connect.
    let err = token_client(addr, "no-such-token").err().unwrap();
    assert!(err.to_string().contains(auth::AUTH_FAILED), "{err}");
    // A PSK client of a token-only server is refused too.
    let err = FerroStore::connect_with(&net.url(addr), &net.options(psk_options()))
        .err()
        .unwrap();
    assert!(err.to_string().contains(auth::AUTH_FAILED), "{err}");
}

#[test]
fn a_secret_file_others_can_read_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let tokens = three_tokens(dir.path());
    for mode in [0o644, 0o640, 0o604, 0o660] {
        std::fs::set_permissions(&tokens, std::fs::Permissions::from_mode(mode)).unwrap();
        let err = ServerAuth::new().with_token_file(&tokens).unwrap_err();
        assert!(err.to_string().contains("chmod 600"), "{mode:o}: {err}");
    }
    std::fs::set_permissions(&tokens, std::fs::Permissions::from_mode(0o600)).unwrap();
    ServerAuth::new().with_token_file(&tokens).unwrap();

    let psk = dir.path().join("psk");
    std::fs::write(&psk, format!("{}\n", hex(TEST_PSK))).unwrap();
    std::fs::set_permissions(&psk, std::fs::Permissions::from_mode(0o644)).unwrap();
    let err = ServerAuth::new()
        .with_psk_file(TEST_SERVER_ID, &psk)
        .unwrap_err();
    assert!(err.to_string().contains("chmod 600"), "{err}");
    std::fs::set_permissions(&psk, std::fs::Permissions::from_mode(0o600)).unwrap();
    let server = serve_tcp(
        FerroStore::in_memory(),
        ServerAuth::new()
            .with_psk_file(TEST_SERVER_ID, &psk)
            .unwrap(),
    );
    // The client reads the same file, under the same rule.
    let url = format!("ws://{}", server.local_addr());
    FerroStore::connect_with(&url, &ConnectOptions::psk_file(TEST_SERVER_ID, &psk)).unwrap();
    std::fs::set_permissions(&psk, std::fs::Permissions::from_mode(0o640)).unwrap();
    let err = FerroStore::connect_with(&url, &ConnectOptions::psk_file(TEST_SERVER_ID, &psk))
        .err()
        .unwrap();
    assert!(err.to_string().contains("chmod 600"), "{err}");
}

#[test]
fn a_token_file_with_a_bad_line_is_refused_by_line_number() {
    let dir = tempfile::tempdir().unwrap();
    for (text, line) in [
        (format!("{} superuser x\n", sha256_hex("a")), 1),
        (
            format!("# c\n{} read\n{} read\n", sha256_hex("a"), sha256_hex("a")),
            3,
        ),
        ("abc read\n".to_string(), 1),
        (format!("\n{}\n", sha256_hex("a")), 2),
        (format!("{} read\n", sha256_hex("a").to_uppercase()), 1),
    ] {
        let path = token_file(dir.path(), &text);
        let err = ServerAuth::new().with_token_file(&path).unwrap_err();
        assert!(
            err.to_string().contains(&format!("line {line}")),
            "{text:?}: {err}"
        );
    }
    let empty = token_file(dir.path(), "# nothing\n");
    assert!(
        ServerAuth::new().with_token_file(&empty).is_err(),
        "no tokens"
    );
}

#[test]
fn the_uri_names_only_a_ferrostore_variable_never_a_file_or_a_secret() {
    let dir = tempfile::tempdir().unwrap();
    let tokens = three_tokens(dir.path());
    let token = dir.path().join("my-token");
    std::fs::write(&token, format!("{WRITE_TOKEN}\n")).unwrap();
    std::fs::set_permissions(&token, std::fs::Permissions::from_mode(0o600)).unwrap();
    let psk = psk_file(dir.path(), &format!("{}\n", hex(TEST_PSK)));

    let by_token = serve_tcp(
        FerroStore::in_memory(),
        ServerAuth::new().with_token_file(&tokens).unwrap(),
    );
    let by_psk = serve_tcp(FerroStore::in_memory(), psk_auth());

    // Files are named in the options.
    let client = FerroStore::connect_with(
        &format!("ws://{}/store", by_token.local_addr()),
        &ConnectOptions::token_file(&token),
    )
    .unwrap();
    client.put_doc("docs", "a", &json!(1)).unwrap();
    let client = FerroStore::connect_with(
        &format!("ws://{}", by_psk.local_addr()),
        &ConnectOptions::psk_file(TEST_SERVER_ID, &psk),
    )
    .unwrap();
    client.put_doc("docs", "a", &json!(1)).unwrap();

    // A `FERROSTORE_` variable is accepted, and read at connect.
    let unset = "FERROSTORE_TEST_UNSET_6B_FIX";
    for query in [
        format!("token_env={unset}"),
        format!("psk_env={unset}&server_id={TEST_SERVER_ID}"),
    ] {
        let err = FerroStore::connect_uri(&format!("ws://{}?{query}", by_token.local_addr()))
            .err()
            .unwrap();
        assert!(
            err.to_string()
                .contains(&format!("variable `{unset}` is not set")),
            "{query}: {err}"
        );
    }
    let (_, options) = auth::split_uri_query(&format!("h:1?token_env={unset}")).unwrap();
    assert!(format!("{options:?}").contains(unset), "{options:?}");

    // No file, no other variable, no secret.
    for (query, needle) in [
        (format!("token_file={}", token.display()), "names no file"),
        (
            format!("psk_file={}&server_id={TEST_SERVER_ID}", psk.display()),
            "names no file",
        ),
        ("token_env=HOME".to_string(), "FERROSTORE_"),
        (
            "psk_env=AWS_SECRET_ACCESS_KEY&server_id=x".to_string(),
            "FERROSTORE_",
        ),
        ("token_env=FERROSTORE_".to_string(), "FERROSTORE_"),
        ("token_env=FERROSTORE_A-B".to_string(), "FERROSTORE_"),
        (format!("token={WRITE_TOKEN}"), "never"),
        (
            "psk=0123456789abcdef0123456789abcdef&server_id=x".to_string(),
            "never",
        ),
        (format!("psk_env={unset}"), "server_id"),
        (format!("token_env={unset}&server_id=x"), "server_id"),
        ("colour=blue".to_string(), "colour"),
        (
            format!("token_env={unset}&psk_env={unset}&server_id=x"),
            "one",
        ),
    ] {
        let err = FerroStore::connect_uri(&format!("ws://{}?{query}", by_psk.local_addr()))
            .err()
            .unwrap();
        let message = err.to_string();
        assert!(message.contains(needle), "{query}: {message}");
        assert!(
            !message.contains(WRITE_TOKEN),
            "the error repeats the secret"
        );
    }
    // Options and a URI that both name a credential are refused.
    let err = FerroStore::connect_with(
        &format!("ws://{}?token_env={unset}", by_token.local_addr()),
        &psk_options(),
    )
    .err()
    .unwrap();
    assert!(err.to_string().contains("twice"), "{err}");
    // An in-process store takes no credential.
    let err = FerroStore::connect_with("mem://", &psk_options())
        .err()
        .unwrap();
    assert!(err.to_string().contains("unix:// and ws://"), "{err}");
    // A TLS option outside wss:// is refused: it would check nothing.
    let err = FerroStore::connect_with(
        &format!("ws://{}", by_psk.local_addr()),
        &psk_options().ca_file(test_ca()),
    )
    .err()
    .unwrap();
    assert!(err.to_string().contains("wss:// only"), "{err}");
    // A bearer token is for ws:// and wss:// alone.
    let err = FerroStore::connect_with(
        &format!("unix://{}", dir.path().join("none.sock").display()),
        &ConnectOptions::token(WRITE_TOKEN),
    )
    .err()
    .unwrap();
    assert!(err.to_string().contains("ws://"), "{err}");
    // A token that would break the header is refused before it is sent.
    let err = token_client(by_token.local_addr(), "bad\r\nX-Injected: 1")
        .err()
        .unwrap();
    assert!(err.to_string().contains("token"), "{err}");
    assert!(!format!("{:?}", ConnectOptions::token(WRITE_TOKEN)).contains(WRITE_TOKEN));
    assert!(!format!("{:?}", psk_options()).contains("0123456789abcdef"));
}

// ---- plaintext ------------------------------------------------------------------

#[test]
fn a_credential_goes_over_plain_ws_only_to_loopback_unless_allowed() {
    let remote_v4: SocketAddr = "10.1.2.3:7000".parse().unwrap();
    let remote_v6: SocketAddr = "[2001:db8::1]:7000".parse().unwrap();
    let bearer = ClientCred::Bearer("t".into());
    let psk = ClientCred::Psk(Psk::new(TEST_SERVER_ID, TEST_PSK).unwrap());
    for cred in [&bearer, &psk] {
        for peer in [remote_v4, remote_v6] {
            let err = cred.check_plaintext(peer, false).unwrap_err();
            assert!(err.to_string().contains("plain ws://"), "{err}");
            assert!(err.to_string().contains("allow_plaintext"), "{err}");
            cred.check_plaintext(peer, true).unwrap();
        }
        for peer in [
            "127.0.0.1:1",
            "127.9.9.9:1",
            "[::1]:1",
            "[::ffff:127.0.0.1]:1",
        ] {
            cred.check_plaintext(peer.parse().unwrap(), false).unwrap();
        }
    }
    ClientCred::None.check_plaintext(remote_v4, false).unwrap();

    // Wired in: with every address taken as remote, a token is never
    // written to the socket.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let sink = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut got = Vec::new();
        let _ = stream.read_to_end(&mut got);
        got
    });
    auth::LOOPBACK_IS_REMOTE.with(|flag| flag.set(true));
    let err = token_client(addr, WRITE_TOKEN).err().unwrap();
    assert!(
        err.to_string().contains("refused to send a bearer token"),
        "{err}"
    );
    assert!(sink.join().unwrap().is_empty(), "bytes went to the peer");

    // A PSK handshake is refused the same way, and allowed by the options
    // (never by a URI).
    let server = serve_tcp(FerroStore::in_memory(), psk_auth());
    let url = format!("ws://{}", server.local_addr());
    let err = FerroStore::connect_with(&url, &psk_options())
        .err()
        .unwrap();
    assert!(
        err.to_string().contains("refused to send a PSK handshake"),
        "{err}"
    );
    let client = FerroStore::connect_with(&url, &psk_options().allow_plaintext(true)).unwrap();
    client.put_doc("docs", "a", &json!(1)).unwrap();
    // No credential: nothing to protect, the server decides.
    let err = FerroStore::connect_uri(&url).err().unwrap();
    assert!(err.to_string().contains("requires authentication"), "{err}");
    auth::LOOPBACK_IS_REMOTE.with(|flag| flag.set(false));
    FerroStore::connect_with(&url, &psk_options()).unwrap();
}

// ---- the PSK file -----------------------------------------------------------------

#[test]
fn a_psk_file_holds_64_or_more_hex_digits_decoded_to_the_key() {
    let dir = tempfile::tempdir().unwrap();
    let good = hex(TEST_PSK);
    for bad in [
        // The 32 bytes themselves: hex digits, but 32 of them, not 64.
        String::from_utf8(TEST_PSK.to_vec()).unwrap(),
        good[..62].to_string(),
        good[..63].to_string(),
        format!("{}g", &good[..63]),
        format!("{} {}", &good[..32], &good[32..]),
        format!("{good}\n\n"),
        String::new(),
    ] {
        let path = psk_file(dir.path(), &bad);
        for err in [
            ServerAuth::new()
                .with_psk_file(TEST_SERVER_ID, &path)
                .unwrap_err(),
            ConnectOptions::psk_file(TEST_SERVER_ID, &path)
                .resolve()
                .err()
                .unwrap(),
        ] {
            let message = err.to_string();
            assert!(
                message.contains("64 or more hex digits"),
                "{bad:?}: {message}"
            );
            assert!(
                message.contains("head -c 32 /dev/urandom | xxd -p -c 64"),
                "{message}"
            );
            assert!(bad.len() < 8 || !message.contains(&bad[..8]), "{message}");
        }
    }
    // Hex of either case, one trailing newline, 32 bytes or more: the
    // decoded key interoperates with the raw key in the options.
    for text in [
        format!("{good}\n"),
        format!("{}\r\n", good.to_uppercase()),
        good.clone(),
    ] {
        let path = psk_file(dir.path(), &text);
        let server = serve_tcp(
            FerroStore::in_memory(),
            ServerAuth::new()
                .with_psk_file(TEST_SERVER_ID, &path)
                .unwrap(),
        );
        let url = format!("ws://{}", server.local_addr());
        FerroStore::connect_with(&url, &psk_options()).unwrap();
        FerroStore::connect_with(&url, &ConnectOptions::psk_file(TEST_SERVER_ID, &path)).unwrap();
    }
    let long = psk_file(dir.path(), &hex(&[7u8; 64]));
    ServerAuth::new()
        .with_psk_file(TEST_SERVER_ID, &long)
        .unwrap();
    // Raw bytes in the options and in `with_psk` stay the key itself.
    assert!(
        ServerAuth::new()
            .with_psk(TEST_SERVER_ID, &[1u8; 31])
            .is_err()
    );
    ServerAuth::new()
        .with_psk(TEST_SERVER_ID, &[1u8; 32])
        .unwrap();
}

// ---- the single table of scopes -----------------------------------------------

#[test]
fn every_method_the_server_dispatches_is_classified_once() {
    let source = include_str!("server.rs");
    let start = source.find("\nfn dispatch(").expect("fn dispatch");
    let body = &source[start..];
    let end = body.find("\n        _ => ").expect("the dispatch fallback");
    let mut dispatched = BTreeSet::new();
    for line in body[..end].lines() {
        if let Some(rest) = line.strip_prefix("        \"")
            && let Some((name, tail)) = rest.split_once('"')
            && tail.starts_with(" =>")
        {
            dispatched.insert(name);
        }
    }
    // `subscribe_mq` is answered by the connection loop, not by `dispatch`.
    assert!(source.contains("rule.name == \"subscribe_mq\""));
    dispatched.insert("subscribe_mq");
    assert!(
        dispatched.len() >= 29,
        "only {} methods found",
        dispatched.len()
    );

    let classified: BTreeSet<&str> = auth::METHODS.iter().map(|rule| rule.name).collect();
    assert_eq!(
        classified.len(),
        auth::METHODS.len(),
        "a method is classified twice"
    );
    assert_eq!(dispatched, classified);

    // The spec's fixed points.
    let scope = |name: &str| auth::method_rule(name).unwrap().scope;
    assert_eq!(scope("commit_txn"), Scope::Write);
    assert_eq!(scope("open_session"), Scope::Read);
    assert_eq!(scope("txn_read"), Scope::Read);
    assert_eq!(scope("checkpoint"), Scope::Admin);
    assert_eq!(scope("compact"), Scope::Admin);
    assert_eq!(scope("apply_sync_batches"), Scope::Admin);
    // Every method that keeps an exactly-once reply writes.
    for rule in auth::METHODS {
        assert!(!rule.writes || rule.scope >= Scope::Write, "{}", rule.name);
    }
    assert!(Scope::Admin.covers(Scope::Write) && Scope::Write.covers(Scope::Read));
    assert!(!Scope::Read.covers(Scope::Write) && !Scope::Write.covers(Scope::Admin));
}

// ---- nothing before authentication ---------------------------------------------

#[test]
fn no_request_runs_before_the_connection_authenticates() {
    for net in NETS {
        no_request_runs_before_authentication(net);
    }
}

fn no_request_runs_before_authentication(net: Net) {
    let server = net.serve(FerroStore::in_memory(), psk_auth());
    let addr = server.local_addr();
    let before = server.dispatched();
    for request in [
        json!({"id": 1, "method": "open_session", "params": {}}),
        json!({"id": 1, "method": "get_doc", "params": {"collection": "docs", "id": "a"}}),
        json!({"id": 1, "method": "put_doc", "params": {"collection": "docs", "id": "a", "doc": 1}}),
    ] {
        // The request as the very first frame, then as the answer to the
        // challenge. The counter is checked before the frames: a server
        // that ran the request has already counted it when it replies.
        for read_challenge_first in [false, true] {
            let mut raw = RawWs::over(net, addr, None);
            if read_challenge_first {
                raw.challenge();
            }
            raw.send(&request);
            let mut frames = Vec::new();
            while let Some(frame) = raw.recv() {
                frames.push(frame);
            }
            assert_eq!(
                server.dispatched(),
                before,
                "{request} ran before authentication: {frames:?}"
            );
            let refusal = json!({"auth": "failed", "error": "authentication failed"});
            if read_challenge_first {
                assert_eq!(frames, vec![refusal], "{request}");
            } else {
                assert_eq!(frames.len(), 2, "{frames:?}");
                assert_eq!(frames[0]["auth"], "challenge", "{frames:?}");
                assert_eq!(frames[1], refusal, "{request}");
            }
        }
    }
    assert_eq!(server.open_sessions(), 0);
    assert_eq!(server.store().get_doc("docs", "a").unwrap(), None);
}

#[test]
fn a_failed_authentication_closes_after_the_fixed_delay_and_names_no_check() {
    for net in NETS {
        a_failed_authentication_names_no_check(net);
    }
}

fn a_failed_authentication_names_no_check(net: Net) {
    let dir = tempfile::tempdir().unwrap();
    let tokens = three_tokens(dir.path());
    let server = net.serve(
        FerroStore::in_memory(),
        psk_auth().with_token_file(&tokens).unwrap(),
    );
    let addr = server.local_addr();
    let url = net.url(addr);
    let mut messages = BTreeSet::new();

    let started = Instant::now();
    let wrong = ConnectOptions::psk(
        TEST_SERVER_ID,
        b"another-32-byte-pre-shared-key!!".as_slice(),
    );
    messages.insert(
        FerroStore::connect_with(&url, &net.options(wrong))
            .err()
            .unwrap()
            .to_string(),
    );
    assert!(started.elapsed() >= auth::AUTH_FAIL_DELAY, "wrong PSK");

    let started = Instant::now();
    messages.insert(net.token_client(addr, "unknown").err().unwrap().to_string());
    assert!(started.elapsed() >= auth::AUTH_FAIL_DELAY, "unknown token");

    let mut raw = RawWs::over(net, addr, None);
    raw.challenge();
    let started = Instant::now();
    raw.send(&json!({"auth": "psk", "nc": "zz"}));
    let answer = raw.recv();
    assert!(
        started.elapsed() >= auth::AUTH_FAIL_DELAY,
        "malformed hello"
    );
    raw.refused(answer);

    assert_eq!(messages.len(), 1, "the errors differ: {messages:?}");
    assert!(auth::AUTH_FAIL_DELAY >= Duration::from_millis(200));
}

#[test]
fn bind_tcp_without_authentication_is_an_error() {
    let err = FerroStoreServer::bind_tcp("127.0.0.1:0", FerroStore::in_memory(), ServerAuth::new())
        .err()
        .unwrap();
    assert!(err.to_string().contains("authentication"), "{err}");
}

// ---- unauthenticated peers are bounded ---------------------------------------------

/// Wait until `server` has `n` connections authenticating.
pub(super) fn wait_handshakes(server: &FerroStoreServer, n: usize) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while server.handshakes_in_flight() != n {
        assert!(
            Instant::now() < deadline,
            "{} connections authenticating, not {n}",
            server.handshakes_in_flight()
        );
        thread::sleep(Duration::from_millis(5));
    }
}

/// `true` once `stream` reads end-of-stream (or a reset) within `within`.
pub(super) fn closed_within(stream: &mut TcpStream, within: Duration) -> bool {
    stream
        .set_read_timeout(Some(Duration::from_millis(20)))
        .unwrap();
    let started = Instant::now();
    let mut buf = [0u8; 256];
    while started.elapsed() < within {
        match stream.read(&mut buf) {
            Ok(0) => return true,
            Ok(_) => {}
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(err) if err.kind() == std::io::ErrorKind::TimedOut => {}
            Err(_) => return true,
        }
    }
    false
}

/// [`closed_within`] for a Unix stream.
fn closed_within_unix(stream: &mut UnixStream, within: Duration) -> bool {
    stream
        .set_read_timeout(Some(Duration::from_millis(20)))
        .unwrap();
    let started = Instant::now();
    let mut buf = [0u8; 256];
    while started.elapsed() < within {
        match stream.read(&mut buf) {
            Ok(0) => return true,
            Ok(_) => {}
            Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(err) if err.kind() == std::io::ErrorKind::TimedOut => {}
            Err(_) => return true,
        }
    }
    false
}

/// A peer that sends one byte at a time, each well inside the per-read
/// timeout, is cut off at the total handshake deadline.
#[test]
fn a_trickling_handshake_is_cut_off_at_the_total_deadline() {
    let server = serve_tcp(FerroStore::in_memory(), psk_auth());
    let deadline = Duration::from_millis(600);
    server.set_handshake_deadline(deadline);
    let addr = server.local_addr();

    let mut trickle = TcpStream::connect(addr).unwrap();
    trickle
        .set_read_timeout(Some(Duration::from_millis(20)))
        .unwrap();
    let started = Instant::now();
    trickle.write_all(b"GET / HTTP/1.1\r\nX-Slow: ").unwrap();
    let mut closed = None;
    while started.elapsed() < Duration::from_secs(5) {
        // One byte per 50 ms: 16 KiB of headers would take 13 minutes.
        if trickle.write_all(b"a").is_err() {
            closed = Some(started.elapsed());
            break;
        }
        let mut buf = [0u8; 64];
        match trickle.read(&mut buf) {
            Ok(0) => {
                closed = Some(started.elapsed());
                break;
            }
            Ok(_) => {}
            Err(err)
                if matches!(
                    err.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) =>
            {
                thread::sleep(Duration::from_millis(30));
            }
            Err(_) => {
                closed = Some(started.elapsed());
                break;
            }
        }
    }
    let closed = closed.expect("the trickling peer was never cut off");
    assert!(
        closed >= deadline,
        "cut off after {closed:?}, before the deadline"
    );
    assert!(
        closed < deadline + Duration::from_secs(2),
        "cut off after {closed:?}"
    );
    wait_handshakes(&server, 0);

    // A client that authenticates at once is unaffected, and is not cut off
    // once served.
    let client = ws_client(addr);
    thread::sleep(deadline + Duration::from_millis(200));
    client.put_doc("docs", "a", &json!(1)).unwrap();
    assert_eq!(client.get_doc("docs", "a").unwrap(), Some(json!(1)));
    assert_eq!(auth::HANDSHAKE_DEADLINE, Duration::from_secs(10));
}

/// A pre-auth hello that declares 1 MiB is refused from its length: the
/// server answers at once, without waiting for (or allocating) the payload.
#[test]
fn an_oversized_hello_is_refused_from_its_declared_length() {
    for net in NETS {
        an_oversized_hello_is_refused(net);
    }
}

fn an_oversized_hello_is_refused(net: Net) {
    let server = net.serve(FerroStore::in_memory(), psk_auth());
    let mut raw = RawWs::over(net, server.local_addr(), None);
    raw.challenge();
    let mut header = vec![0x81, 0x80 | 127];
    header.extend_from_slice(&(1u64 << 20).to_be_bytes());
    header.extend_from_slice(&[9, 9, 9, 9]);
    let started = Instant::now();
    raw.stream.write_all(&header).unwrap();
    let answer = raw.recv();
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "the server waited {:?} for the payload",
        started.elapsed()
    );
    raw.refused(answer);

    // The same over a Unix socket with a PSK: the length prefix is checked.
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("s.sock");
    let _unix =
        FerroStoreServer::bind_unix_with(&sock, FerroStore::in_memory(), psk_auth()).unwrap();
    let mut stream = UnixStream::connect(&sock).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let (challenge, _) = read_frame(&mut stream, ws::MAX_PAYLOAD).unwrap();
    assert_eq!(challenge["auth"], "challenge");
    let started = Instant::now();
    stream.write_all(&(1u32 << 20).to_be_bytes()).unwrap();
    let (answer, _) = read_frame(&mut stream, ws::MAX_PAYLOAD).unwrap();
    assert!(started.elapsed() < Duration::from_secs(2));
    assert_eq!(
        answer,
        json!({"auth": "failed", "error": "authentication failed"})
    );
    // A hello at the limit is read whole (and refused as malformed).
    let mut raw = RawWs::over(net, server.local_addr(), None);
    raw.challenge();
    let padding = "x".repeat(auth::PRE_AUTH_FRAME_BYTES - 40);
    let answer = raw.hello_value(&json!({"auth": "psk", "pad": padding}));
    raw.refused(answer);
}

/// At most [`auth::MAX_HANDSHAKES`] connections authenticate at once; one
/// more is closed at accept, and served connections are not counted.
#[test]
fn the_129th_concurrent_unauthenticated_connection_is_refused() {
    assert_eq!(auth::MAX_HANDSHAKES, 128);
    let server = serve_tcp(FerroStore::in_memory(), psk_auth());
    let addr = server.local_addr();
    let served = ws_client(addr);
    served.put_doc("docs", "a", &json!(1)).unwrap();

    let silent: Vec<TcpStream> = (0..auth::MAX_HANDSHAKES)
        .map(|_| TcpStream::connect(addr).unwrap())
        .collect();
    wait_handshakes(&server, auth::MAX_HANDSHAKES);
    let refused = server.handshakes_refused();
    let started = Instant::now();
    let mut extra = TcpStream::connect(addr).unwrap();
    assert!(
        closed_within(&mut extra, Duration::from_secs(2)),
        "the extra connection was held open"
    );
    assert!(started.elapsed() < Duration::from_secs(2));
    assert_eq!(server.handshakes_refused(), refused + 1);
    assert_eq!(server.handshakes_in_flight(), auth::MAX_HANDSHAKES);

    // The served client keeps working while the slots are full.
    served.put_doc("docs", "b", &json!(2)).unwrap();
    assert_eq!(served.get_doc("docs", "a").unwrap(), Some(json!(1)));

    // Freed, the slots take new clients again.
    drop(silent);
    wait_handshakes(&server, 0);
    let late = ws_client(addr);
    assert_eq!(late.get_doc("docs", "b").unwrap(), Some(json!(2)));

    // The cap does not apply to a Unix socket (only its owner can connect):
    // with it lowered, more than the cap of silent peers are all served (each
    // gets its challenge), none is refused, and each still has the deadline.
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("s.sock");
    let unix =
        FerroStoreServer::bind_unix_with(&sock, FerroStore::in_memory(), psk_auth()).unwrap();
    unix.set_max_handshakes(2);
    let mut held: Vec<UnixStream> = (0..2 + 5)
        .map(|_| UnixStream::connect(&sock).unwrap())
        .collect();
    wait_handshakes(&unix, held.len());
    for stream in held.iter_mut() {
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let (challenge, _) = read_frame(stream, ws::MAX_PAYLOAD).unwrap();
        assert_eq!(challenge["auth"], "challenge", "a Unix peer was refused");
    }
    assert_eq!(unix.handshakes_refused(), 0);
    drop(held);
    wait_handshakes(&unix, 0);
    FerroStore::connect_with(&format!("unix://{}", sock.display()), &psk_options()).unwrap();

    // The Unix deadline holds: a silent peer is shut down past it.
    unix.set_handshake_deadline(Duration::from_millis(100));
    let mut slow = UnixStream::connect(&sock).unwrap();
    assert!(
        closed_within_unix(&mut slow, Duration::from_secs(3)),
        "a silent Unix peer was held past the deadline"
    );
}

// ---- per-principal quotas and request ids ----------------------------------------

#[test]
fn one_credential_holds_at_most_a_quarter_of_the_sessions() {
    for net in NETS {
        one_credential_holds_a_quarter_of_the_sessions(net);
    }
}

fn one_credential_holds_a_quarter_of_the_sessions(net: Net) {
    use super::session::{SESSION_QUOTA, SESSIONS_PER_CONN, SESSIONS_PER_PRINCIPAL};
    assert_eq!(SESSIONS_PER_PRINCIPAL, 256);
    let (server, tokens, _dir) = token_principals_over(net, 2);
    let addr = server.local_addr();
    let hog: Vec<server::RemoteClient> = (0..SESSIONS_PER_PRINCIPAL / SESSIONS_PER_CONN + 1)
        .map(|_| net.token_raw(addr, &tokens[0]))
        .collect();
    let open = |client: &server::RemoteClient| client.call("open_session", json!({}));
    for client in &hog[..hog.len() - 1] {
        for _ in 0..SESSIONS_PER_CONN {
            open(client).unwrap();
        }
    }
    assert_eq!(server.open_sessions(), SESSIONS_PER_PRINCIPAL);
    // A fresh connection of the same credential is at the quota.
    let err = open(hog.last().unwrap()).unwrap_err();
    assert!(err.to_string().contains(SESSION_QUOTA), "{err}");
    // So is a read that would open one.
    let err = hog
        .last()
        .unwrap()
        .call("txn_read", json!({"read": {"p": "inventory"}}))
        .unwrap_err();
    assert!(err.to_string().contains(SESSION_QUOTA), "{err}");
    // Another credential is not.
    let other = net.token_raw(addr, &tokens[1]);
    open(&other).unwrap();
    assert_eq!(server.open_sessions(), SESSIONS_PER_PRINCIPAL + 1);
}

/// A request id is the principal's own: another principal's equal id runs
/// as a fresh request and never gets the first principal's reply; the same
/// principal's id for another method is refused and runs nothing.
#[test]
fn a_request_id_is_per_credential_and_names_one_method() {
    for net in NETS {
        a_request_id_is_per_credential(net);
    }
}

fn a_request_id_is_per_credential(net: Net) {
    use super::session::DEDUP_OTHER_METHOD;
    let (server, tokens, _dir) = token_principals_over(net, 2);
    let addr = server.local_addr();
    let a = net.token_raw(addr, &tokens[0]);
    let b = net.token_raw(addr, &tokens[1]);
    let store = server.store();
    let req = format!("{:032x}", 0xfeed_u128);

    let first = a
        .call(
            "publish_mq",
            json!({"topic": "t", "key": "a", "payload": "from a", "req": req}),
        )
        .unwrap();
    let seq = store.last_seq().unwrap();
    // B's colliding id is B's own key space: it runs, and gets its reply.
    let theirs = b
        .call(
            "publish_mq",
            json!({"topic": "t", "key": "b", "payload": "from b", "req": req}),
        )
        .unwrap();
    assert_ne!(theirs, first, "B was answered with A's reply");
    assert_eq!(store.last_seq().unwrap(), seq + 1, "B's write did not run");
    let records = store.poll_mq_group("t", "g", 10).unwrap();
    assert_eq!(records.len(), 2, "{records:?}");
    // A's own replay still answers A's first reply and runs nothing.
    let again = a
        .call(
            "publish_mq",
            json!({"topic": "t", "key": "a", "payload": "from a", "req": req}),
        )
        .unwrap();
    assert_eq!(again, first);
    assert_eq!(store.last_seq().unwrap(), seq + 1);
    // The same id for another method is refused, and runs nothing.
    let before = server.dispatched();
    let err = a
        .call(
            "put_doc",
            json!({"collection": "d", "id": "x", "doc": 1, "req": req}),
        )
        .unwrap_err();
    assert!(err.to_string().contains(DEDUP_OTHER_METHOD), "{err}");
    assert_eq!(server.dispatched(), before, "the other method ran");
    assert_eq!(store.get_doc("d", "x").unwrap(), None);
    assert_eq!(store.last_seq().unwrap(), seq + 1);
}
