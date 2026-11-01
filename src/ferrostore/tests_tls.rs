//! Spec §6.3: `wss://`, the verification modes of its client, the PSK
//! channel binding, and the bind policy of plain `ws://`.
//!
//! The certificates in `testdata/tls/` are static fixtures, made once with
//! OpenSSL 3.5 (CA keys discarded; `server.key` is the one leaf key):
//!
//! ```text
//! leaf_ext() { printf 'subjectAltName=%s\nbasicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature\nextendedKeyUsage=serverAuth\n' "$1"; }
//! ca() { openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -keyout "$1.key" \
//!          -out "$1.pem" -days 36500 -subj "/CN=$2" \
//!          -addext basicConstraints=critical,CA:TRUE -addext keyUsage=critical,keyCertSign,cRLSign; }
//! ca ca "FerroStore test CA"
//! ca other-ca "FerroStore unrelated test CA"
//! openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 -out server.key
//! openssl req -new -key server.key -subj /CN=localhost -out server.csr
//! leaf_ext DNS:localhost,IP:127.0.0.1 > server.ext
//! leaf_ext DNS:wrong.example > wrong.ext
//! openssl x509 -req -in server.csr -CA ca.pem -CAkey ca.key -set_serial 2 -days 36500 \
//!     -extfile server.ext -out server.pem
//! openssl x509 -req -in server.csr -CA ca.pem -CAkey ca.key -set_serial 3 -days 36500 \
//!     -extfile wrong.ext -out wrong-host.pem
//! openssl x509 -req -in server.csr -CA ca.pem -CAkey ca.key -set_serial 4 \
//!     -not_before 20000101000000Z -not_after 20010101000000Z -extfile server.ext -out expired.pem
//! # the pin of server.key:
//! openssl x509 -in server.pem -noout -pubkey | openssl pkey -pubin -outform DER | sha256sum
//! ```
//!
//! Git keeps no file mode, so a test copies `server.key` into a temporary
//! directory and makes it 0600 before the server reads it.

use super::auth::{self, Psk};
use super::bind;
use super::server;
use super::tests_auth::{
    Net, RawWs, TEST_PSK, TEST_SERVER_ID, closed_within, psk_auth, psk_options, wait_handshakes,
};
use super::tests_retry::{run_contracts_over, run_public_writes_over};
use super::tls::{self, ServerTls, TlsStream, Verify, Wire};
use super::ws;
use super::{ConnectOptions, FerroStore, FerroStoreServer, ServerAuth};
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};

/// The SHA-256 of `server.key`'s SubjectPublicKeyInfo, as `openssl` prints it.
const SERVER_SPKI_SHA256: &str = "c456894463c2916d6eda9f1d3ffcbe4bea0d69a9b7c0ac084520eb98dfa8f91a";

pub(super) fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/ferrostore/testdata/tls")
        .join(name)
}

/// The test CA, which signed `server.pem`.
pub(super) fn test_ca() -> PathBuf {
    fixture("ca.pem")
}

/// `server.key` copied into `dir` as mode 0600.
fn private_key(dir: &Path) -> PathBuf {
    let path = dir.join("server.key");
    std::fs::copy(fixture("server.key"), &path).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    path
}

/// `auth` serving `wss://` with `server.pem` (`localhost`, `127.0.0.1`).
pub(super) fn with_test_tls(auth: ServerAuth) -> ServerAuth {
    with_cert(auth, "server.pem")
}

/// `auth` serving `wss://` with the fixture certificate `cert` over
/// `server.key`. The key is read here, so its copy may go.
fn with_cert(auth: ServerAuth, cert: &str) -> ServerAuth {
    let dir = tempfile::tempdir().unwrap();
    let key = private_key(dir.path());
    auth.with_tls_files(fixture(cert), &key).unwrap()
}

fn server_tls() -> ServerTls {
    let dir = tempfile::tempdir().unwrap();
    ServerTls::from_pem_files(&fixture("server.pem"), &private_key(dir.path())).unwrap()
}

/// Run `f` on its own thread; panic if it gives no answer in `secs`.
fn within<T: Send + 'static>(secs: u64, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(Duration::from_secs(secs))
        .unwrap_or_else(|_| panic!("no answer after {secs} s"))
}

fn error_of<T>(result: Result<T, fl_core::StoreError>) -> String {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(err) => err.to_string(),
    }
}

// ---- the contracts over TCP ------------------------------------------------------------

fn fresh_over(net: Net) -> (FerroStore, FerroStoreServer) {
    let server = net.serve(FerroStore::in_memory(), psk_auth());
    let client = net.psk_client(server.local_addr());
    (client, server)
}

#[test]
fn the_store_contracts_hold_over_wss() {
    let fresh = || fresh_over(Net::Wss);
    run_contracts_over(fresh);
    run_public_writes_over(fresh);
}

#[test]
fn the_store_contracts_hold_over_ws() {
    let fresh = || fresh_over(Net::Ws);
    run_contracts_over(fresh);
    run_public_writes_over(fresh);
}

/// Frames far larger than one TLS record, and pushes that arrive while the
/// client writes, cross a `wss://` connection whole.
#[test]
fn wss_moves_large_frames_and_pushes() {
    let (client, server) = fresh_over(Net::Wss);
    let big = "x".repeat(4 * 1024 * 1024);
    client
        .put_doc("docs", "big", &json!({ "pad": big }))
        .unwrap();
    let got = client.get_doc("docs", "big").unwrap().unwrap();
    assert_eq!(got["pad"].as_str().unwrap().len(), big.len());
    let rx = client.subscribe_mq("t");
    let publisher = Net::Wss.psk_client(server.local_addr());
    for i in 0..50 {
        publisher.publish_mq("t", "k", &json!(i)).unwrap();
        client
            .put_doc(
                "docs",
                &format!("d{i}"),
                &json!({ "pad": "y".repeat(100_000) }),
            )
            .unwrap();
    }
    for i in 0..50 {
        let record = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(record.payload, json!(i));
    }
}

/// Both ends write 8 MiB at once while their readers drain: a reader that
/// held the TLS state across its socket read, or a writer that held it
/// across its socket write, would stop the other half and, with the peer
/// doing the same, both sides would wait forever.
#[test]
fn a_split_tls_stream_reads_and_writes_at_once_both_ways() {
    const LEN: usize = 8 * 1024 * 1024;
    fn exchange(mut read: TlsStream, mut write: TlsStream, byte: u8, peer: u8) {
        let writer = thread::spawn(move || {
            let chunk = vec![byte; 100_000];
            let mut left = LEN;
            while left > 0 {
                let n = left.min(chunk.len());
                write.write_all(&chunk[..n]).unwrap();
                left -= n;
            }
            write
        });
        let mut got = 0;
        let mut buf = vec![0u8; 64 * 1024];
        while got < LEN {
            let n = read.read(&mut buf).unwrap();
            assert!(n > 0, "end of stream after {got} bytes");
            assert!(buf[..n].iter().all(|b| *b == peer));
            got += n;
        }
        let _write = writer.join().unwrap();
    }
    within(120, || {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let tls = server_tls();
        let server = thread::spawn(move || {
            let (tcp, _) = listener.accept().unwrap();
            let (mut read, write) = TlsStream::server(tcp, &tls).unwrap();
            read.handshake().unwrap();
            exchange(read, write, b's', b'c');
        });
        let tcp = TcpStream::connect(addr).unwrap();
        let config = tls::client_config(&Verify::CaFile(test_ca())).unwrap();
        let name = tls::server_name(&addr.to_string()).unwrap();
        let (read, write) = TlsStream::client(tcp, config, name).unwrap();
        exchange(read, write, b'c', b's');
        server.join().unwrap();
    });
}

/// A connected TLS pair over loopback: the server's halves, the client's
/// halves, and a raw clone of the client's socket.
fn tls_pair() -> ((TlsStream, TlsStream), (TlsStream, TlsStream), TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let tls = server_tls();
    let server = thread::spawn(move || {
        let (tcp, _) = listener.accept().unwrap();
        let (mut read, write) = TlsStream::server(tcp, &tls).unwrap();
        read.handshake().unwrap();
        (read, write)
    });
    let tcp = TcpStream::connect(addr).unwrap();
    let raw = tcp.try_clone().unwrap();
    let config = tls::client_config(&Verify::CaFile(test_ca())).unwrap();
    let name = tls::server_name(&addr.to_string()).unwrap();
    let client = TlsStream::client(tcp, config, name).unwrap();
    (server.join().unwrap(), client, raw)
}

/// A shutdown sends close_notify first: the peer reads a clean end of the
/// stream (`Ok(0)`), not a truncation (`UnexpectedEof`).
#[test]
fn a_tls_shutdown_sends_close_notify_and_the_peer_reads_a_clean_end() {
    within(30, || {
        let ((mut server_read, _server_write), (_client_read, client_write), _raw) = tls_pair();
        let mut client_write = Wire::Tls(client_write);
        client_write.write_all(b"last").unwrap();
        client_write.shutdown_both();
        let mut got = [0u8; 4];
        server_read.read_exact(&mut got).unwrap();
        assert_eq!(&got, b"last");
        let mut more = [0u8; 16];
        match server_read.read(&mut more) {
            Ok(0) => {}
            other => panic!("expected a clean close (Ok(0)), got {other:?}"),
        }
    });
}

/// A record the reader cannot open is an error, and the fatal alert that
/// names it reaches the peer before the connection goes (best-effort, from
/// the reading half), so the peer reads the reason, not a bare end.
#[test]
fn a_fatal_tls_read_error_sends_its_alert_to_the_peer() {
    within(30, || {
        let ((mut server_read, server_write), (mut client_read, _client_write), mut raw) =
            tls_pair();
        // An application-data record of 32 bytes that no key sealed.
        let mut forged = vec![0x17, 0x03, 0x03, 0x00, 0x20];
        forged.extend_from_slice(&[0x41; 32]);
        raw.write_all(&forged).unwrap();
        let err = server_read.read(&mut [0u8; 16]).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "{err}");
        // The server's halves stay open: what the client reads is the alert.
        // (`SO_RCVTIMEO` is the socket's: the raw clone sets it.)
        raw.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let err = client_read.read(&mut [0u8; 16]).unwrap_err();
        assert!(
            err.to_string().contains("received fatal alert"),
            "expected the server's alert, got {err:?}"
        );
        drop((server_read, server_write));
    });
}

// ---- verification -----------------------------------------------------------------------

#[test]
fn a_wss_client_trusts_its_ca_file_alone() {
    let server = Net::Wss.serve(FerroStore::in_memory(), psk_auth());
    let url = Net::Wss.url(server.local_addr());
    let client = FerroStore::connect_with(&url, &psk_options().ca_file(test_ca())).unwrap();
    client.put_doc("docs", "a", &json!(1)).unwrap();
    // `localhost` is a name of the certificate too.
    let by_name = format!("wss://localhost:{}", server.local_addr().port());
    FerroStore::connect_with(&by_name, &psk_options().ca_file(test_ca())).unwrap();
    let before = server.dispatched();

    // An unrelated CA, and the bundled Mozilla roots (the default), do not
    // know the test CA: refused in the TLS handshake, before any credential.
    for options in [
        psk_options().ca_file(fixture("other-ca.pem")),
        psk_options(),
        ConnectOptions::token("tok").ca_file(fixture("other-ca.pem")),
    ] {
        let err = error_of(FerroStore::connect_with(&url, &options));
        assert!(err.contains("TLS handshake"), "{err}");
        assert!(err.contains("invalid peer certificate"), "{err}");
        assert!(err.contains("UnknownIssuer"), "{err}");
    }
    // A plain client of a TLS server gets no answer it can read.
    let err = error_of(FerroStore::connect_with(
        &format!("ws://{}", server.local_addr()),
        &psk_options(),
    ));
    assert!(!err.contains("authenticated"), "{err}");
    // A CA file that is missing, or holds no certificate, is named.
    let dir = tempfile::tempdir().unwrap();
    let empty = dir.path().join("empty.pem");
    std::fs::write(&empty, "nothing here\n").unwrap();
    for (path, needle) in [
        (dir.path().join("missing.pem"), "missing.pem"),
        (empty, "holds no CERTIFICATE"),
    ] {
        let err = error_of(FerroStore::connect_with(
            &url,
            &psk_options().ca_file(&path),
        ));
        assert!(err.contains(needle), "{err}");
    }
    assert_eq!(server.dispatched(), before, "a refused client ran nothing");
}

#[test]
fn a_certificate_for_another_name_or_out_of_date_is_refused() {
    for (cert, needle) in [
        ("wrong-host.pem", "not valid for name"),
        ("expired.pem", "expired"),
    ] {
        let server = FerroStoreServer::bind_tcp(
            "127.0.0.1:0",
            FerroStore::in_memory(),
            with_cert(psk_auth(), cert),
        )
        .unwrap();
        let err = error_of(FerroStore::connect_with(
            &Net::Wss.url(server.local_addr()),
            &psk_options().ca_file(test_ca()),
        ));
        assert!(err.contains("invalid peer certificate"), "{cert}: {err}");
        assert!(err.contains(needle), "{cert}: {err}");
    }
}

#[test]
fn a_pinned_spki_replaces_the_ca_but_not_the_handshake_signature() {
    // The pin is the SPKI's SHA-256 as openssl prints it.
    let pem = std::fs::read(fixture("server.pem")).unwrap();
    let cert = CertificateDer::from_pem_slice(&pem).unwrap();
    let parsed = rustls::server::ParsedCertificate::try_from(&cert).unwrap();
    let digest = Sha256::digest(parsed.subject_public_key_info().as_ref());
    let ours: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(ours, SERVER_SPKI_SHA256);

    let server = Net::Wss.serve(FerroStore::in_memory(), psk_auth());
    let url = Net::Wss.url(server.local_addr());
    let colons: String = SERVER_SPKI_SHA256
        .to_uppercase()
        .as_bytes()
        .chunks(2)
        .map(|pair| std::str::from_utf8(pair).unwrap())
        .collect::<Vec<_>>()
        .join(":");
    for pin in [SERVER_SPKI_SHA256.to_string(), colons] {
        let client = FerroStore::connect_with(&url, &psk_options().pin_spki_sha256(pin)).unwrap();
        client.put_doc("docs", "a", &json!(1)).unwrap();
    }
    // Another key's pin is refused.
    let mut wrong = SERVER_SPKI_SHA256.to_string();
    wrong.replace_range(0..1, "d");
    let err = error_of(FerroStore::connect_with(
        &url,
        &psk_options().pin_spki_sha256(wrong),
    ));
    assert!(err.contains("invalid peer certificate"), "{err}");
    // The name and dates are not the pin's: the same key under another
    // name, or out of date, is the pinned server.
    for cert in ["wrong-host.pem", "expired.pem"] {
        let other = FerroStoreServer::bind_tcp(
            "127.0.0.1:0",
            FerroStore::in_memory(),
            with_cert(psk_auth(), cert),
        )
        .unwrap();
        FerroStore::connect_with(
            &Net::Wss.url(other.local_addr()),
            &psk_options().pin_spki_sha256(SERVER_SPKI_SHA256),
        )
        .unwrap();
    }
    // A pin that is not 64 hex digits, or a pin with a CA file, is refused.
    for bad in ["abc", &SERVER_SPKI_SHA256[1..], &"g".repeat(64)] {
        let err = error_of(FerroStore::connect_with(
            &url,
            &psk_options().pin_spki_sha256(bad),
        ));
        assert!(err.contains("64 hex digits"), "{bad}: {err}");
    }
    let err = error_of(FerroStore::connect_with(
        &url,
        &psk_options()
            .pin_spki_sha256(SERVER_SPKI_SHA256)
            .ca_file(test_ca()),
    ));
    assert!(err.contains("both"), "{err}");

    // A server that presents the pinned certificate but signs with another
    // key (it has the certificate, not its private key) is refused; so is
    // it under the CA.
    let pin = tls::parse_pin(SERVER_SPKI_SHA256).unwrap();
    for verify in [Verify::Pin(pin), Verify::CaFile(test_ca())] {
        let (addr, impostor) = impostor_server();
        let err = error_of(server::dial_wire(
            &addr.to_string(),
            None,
            Some(&verify),
            |_| Ok(()),
        ));
        assert!(err.contains("TLS handshake"), "{verify:?}: {err}");
        assert!(err.contains("BadSignature"), "{verify:?}: {err}");
        let _ = impostor.join();
    }
}

/// A TLS server that presents `server.pem` and signs its handshake with a
/// fresh key of its own. It serves one connection.
fn impostor_server() -> (SocketAddr, thread::JoinHandle<()>) {
    use ring::signature::{ECDSA_P256_SHA256_ASN1_SIGNING, EcdsaKeyPair};
    use rustls::server::{ClientHello, ResolvesServerCert};
    use rustls::sign::CertifiedKey;

    #[derive(Debug)]
    struct Fixed(Arc<CertifiedKey>);
    impl ResolvesServerCert for Fixed {
        fn resolve(&self, _: ClientHello<'_>) -> Option<Arc<CertifiedKey>> {
            Some(Arc::clone(&self.0))
        }
    }

    let pkcs8 = EcdsaKeyPair::generate_pkcs8(
        &ECDSA_P256_SHA256_ASN1_SIGNING,
        &ring::rand::SystemRandom::new(),
    )
    .unwrap();
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(pkcs8.as_ref().to_vec()));
    let provider = rustls::crypto::ring::default_provider();
    let signer = provider.key_provider.load_private_key(key).unwrap();
    let chain = vec![CertificateDer::from_pem_file(fixture("server.pem")).unwrap()];
    let certified = Arc::new(CertifiedKey::new(chain, signer));
    let config = rustls::ServerConfig::builder_with_provider(Arc::new(provider))
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_cert_resolver(Arc::new(Fixed(certified)));
    let config = Arc::new(config);
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        let (mut tcp, _) = listener.accept().unwrap();
        tcp.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut conn = rustls::ServerConnection::new(config).unwrap();
        while conn.is_handshaking() {
            if conn.complete_io(&mut tcp).is_err() {
                break;
            }
        }
    });
    (addr, handle)
}

// ---- channel binding ------------------------------------------------------------------

/// A relay that terminates each client's TLS with a certificate the client
/// trusts (the attacker holds one) and opens its own TLS session to
/// `upstream`, then copies the plaintext both ways. It reports each
/// connection's two channel bindings.
fn relay(upstream: SocketAddr) -> (SocketAddr, mpsc::Receiver<(tls::Binding, tls::Binding)>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let tls_config = server_tls();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for tcp in listener.incoming() {
            let Ok(tcp) = tcp else { break };
            let tls_config = tls_config.clone();
            let tx = tx.clone();
            thread::spawn(move || {
                let (mut down_read, down_write) = TlsStream::server(tcp, &tls_config).unwrap();
                if down_read.handshake().is_err() {
                    return;
                }
                let up_tcp = TcpStream::connect(upstream).unwrap();
                let config = tls::client_config(&Verify::CaFile(test_ca())).unwrap();
                let name = tls::server_name(&upstream.to_string()).unwrap();
                let (up_read, up_write) = TlsStream::client(up_tcp, config, name).unwrap();
                let _ = tx.send((down_read.exporter().unwrap(), up_read.exporter().unwrap()));
                pump_both(
                    Wire::Tls(down_read),
                    Wire::Tls(down_write),
                    Wire::Tls(up_read),
                    Wire::Tls(up_write),
                );
            });
        }
    });
    (addr, rx)
}

/// Copy `a_read` to `b_write` and `b_read` to `a_write` until either ends,
/// then shut both down.
fn pump_both(a_read: Wire, a_write: Wire, b_read: Wire, b_write: Wire) {
    fn pump(mut from: Wire, mut to: Wire) -> (Wire, Wire) {
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            match from.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    if to.write_all(&buf[..n]).is_err() {
                        break;
                    }
                }
            }
        }
        from.shutdown_both();
        to.shutdown_both();
        (from, to)
    }
    let forward = thread::spawn(move || pump(a_read, b_write));
    let _ = pump(b_read, a_write);
    let _ = forward.join();
}

/// The §6.2 relay gap, closed: a PSK handshake relayed between two TLS
/// sessions does not verify, since each session has its own exporter. The
/// relay itself works (a bearer token, which is not bound, goes through).
#[test]
fn a_psk_handshake_relayed_into_another_tls_session_fails() {
    let dir = tempfile::tempdir().unwrap();
    let token = "relay-token-0123456789";
    let tokens = dir.path().join("tokens");
    let hash: String = Sha256::digest(token.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    std::fs::write(&tokens, format!("{hash} write\n")).unwrap();
    std::fs::set_permissions(&tokens, std::fs::Permissions::from_mode(0o600)).unwrap();
    let server = Net::Wss.serve(
        FerroStore::in_memory(),
        psk_auth().with_token_file(&tokens).unwrap(),
    );
    let (relay_addr, bindings) = relay(server.local_addr());
    let via_relay = Net::Wss.url(relay_addr);

    // The relay pipes a session through.
    let client =
        FerroStore::connect_with(&via_relay, &ConnectOptions::token(token).ca_file(test_ca()))
            .unwrap();
    client.put_doc("docs", "through", &json!(1)).unwrap();
    let (down, up) = bindings.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_ne!(down, up, "two TLS sessions, two exporters");
    let before = server.dispatched();

    // A PSK client through it is refused by the server: its proof covers
    // the client's session with the relay, not the relay's with the server.
    let err = error_of(FerroStore::connect_with(
        &via_relay,
        &psk_options().ca_file(test_ca()),
    ));
    assert!(err.contains(auth::AUTH_FAILED), "{err}");
    let (down, up) = bindings.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_ne!(down, up);
    assert_eq!(server.dispatched(), before, "nothing ran");
    // The same client straight to the server is served.
    Net::Wss.psk_client(server.local_addr());
}

/// Each side takes the proof version from its own transport, so neither a
/// TLS client nor a TLS server takes part in an unbound handshake.
#[test]
fn over_tls_both_sides_bind_and_neither_accepts_an_unbound_proof() {
    let server = Net::Wss.serve(FerroStore::in_memory(), psk_auth());
    let psk = Psk::new(TEST_SERVER_ID, TEST_PSK).unwrap();
    // The server's challenge over TLS is version 2, and an unbound (version 1)
    // proof is refused.
    let mut raw = RawWs::over(Net::Wss, server.local_addr(), None);
    let binding = raw.binding().expect("a TLS binding");
    let ns = raw.challenge();
    let nc = [0x42u8; 32];
    let answer = raw.hello(&nc, &psk.client_proof(&ns, &nc, None));
    raw.refused(answer);
    // The bound proof passes, and the server's answer is bound too.
    let mut raw = RawWs::over(Net::Wss, server.local_addr(), None);
    let binding_b = raw.binding().unwrap();
    assert_ne!(binding, binding_b, "every connection has its own exporter");
    let ns = raw.challenge();
    let ok = raw
        .hello(&nc, &psk.client_proof(&ns, &nc, Some(&binding_b)))
        .unwrap();
    assert_eq!(ok["auth"], "ok", "{ok}");
    let proof: Vec<u8> = (0..32)
        .map(|i| u8::from_str_radix(&ok["proof"].as_str().unwrap()[2 * i..2 * i + 2], 16).unwrap())
        .collect();
    assert!(psk.verify_server(&nc, &ns, Some(&binding_b), &proof));
    assert!(!psk.verify_server(&nc, &ns, None, &proof));

    // A TLS client refuses a version-1 (unbound) challenge, and sends no
    // proof: a fake TLS server that offers one.
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let tls_config = server_tls();
    let fake = thread::spawn(move || {
        let (tcp, _) = listener.accept().unwrap();
        let (read, write) = TlsStream::server(tcp, &tls_config).unwrap();
        let mut read = Wire::Tls(read);
        read.start().unwrap();
        ws::accept(&mut read).unwrap();
        let ns = [7u8; 32];
        let challenge =
            serde_json::to_vec(&json!({"auth": "challenge", "v": 1, "ns": hex(&ns)})).unwrap();
        let mut write = Wire::Tls(write);
        ws::write_frame(&mut write, false, 0x1, &challenge).unwrap();
        read.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        // Whatever the client sends next: it must be nothing.
        ws::read_event(&mut read, false, ws::MAX_PAYLOAD).err()
    });
    let err = error_of(FerroStore::connect_with(
        &Net::Wss.url(addr),
        &psk_options().ca_file(test_ca()),
    ));
    assert!(err.contains("another authentication version"), "{err}");
    assert!(
        fake.join().unwrap().is_some(),
        "the client answered an unbound challenge"
    );
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// TLS 1.2 is allowed (with the extended master secret, so its exporter is
/// the connection's own), and binds the PSK the same way.
#[test]
fn a_tls_1_2_client_binds_the_psk_as_well() {
    let server = Net::Wss.serve(FerroStore::in_memory(), psk_auth());
    let addr = server.local_addr();
    let mut roots = rustls::RootCertStore::empty();
    roots
        .add(CertificateDer::from_pem_file(test_ca()).unwrap())
        .unwrap();
    let config = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_protocol_versions(&[&rustls::version::TLS12])
    .unwrap()
    .with_root_certificates(roots)
    .with_no_client_auth();
    let tcp = TcpStream::connect(addr).unwrap();
    let name = tls::server_name(&addr.to_string()).unwrap();
    let (read, write) = TlsStream::client(tcp, Arc::new(config), name).unwrap();
    let mut read = Wire::Tls(read);
    ws::upgrade(&mut read, &addr.to_string(), "/", None).unwrap();
    read.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    let mut raw = RawWs {
        stream: read,
        write_half: Wire::Tls(write),
    };
    let psk = Psk::new(TEST_SERVER_ID, TEST_PSK).unwrap();
    let binding = raw.binding().unwrap();
    let ns = raw.challenge();
    let nc = [0x24u8; 32];
    let ok = raw
        .hello(&nc, &psk.client_proof(&ns, &nc, Some(&binding)))
        .unwrap();
    assert_eq!(ok["auth"], "ok", "{ok}");
}

// ---- the server's key -------------------------------------------------------------------

#[test]
fn a_server_key_file_others_can_read_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let key = private_key(dir.path());
    let pem = std::fs::read_to_string(&key).unwrap();
    let body = pem.lines().nth(1).unwrap().to_string();
    for mode in [0o644, 0o640, 0o604] {
        std::fs::set_permissions(&key, std::fs::Permissions::from_mode(mode)).unwrap();
        let err = ServerAuth::new()
            .with_tls_files(fixture("server.pem"), &key)
            .unwrap_err()
            .to_string();
        assert!(err.contains("chmod 600"), "{mode:o}: {err}");
        assert!(!err.contains(&body), "the error repeats the key");
    }
    std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600)).unwrap();
    ServerAuth::new()
        .with_tls_files(fixture("server.pem"), &key)
        .unwrap();
    // A key file that holds no key, and a key that is not the certificate's,
    // are refused without their bytes.
    let not_pem = dir.path().join("not.key");
    std::fs::write(&not_pem, "secret-looking-bytes-0123456789\n").unwrap();
    std::fs::set_permissions(&not_pem, std::fs::Permissions::from_mode(0o600)).unwrap();
    let err = ServerAuth::new()
        .with_tls_files(fixture("server.pem"), &not_pem)
        .unwrap_err()
        .to_string();
    assert!(err.contains("holds no PEM private key"), "{err}");
    assert!(!err.contains("secret-looking"), "{err}");
    let other = dir.path().join("other.key");
    let pkcs8 = ring::signature::EcdsaKeyPair::generate_pkcs8(
        &ring::signature::ECDSA_P256_SHA256_ASN1_SIGNING,
        &ring::rand::SystemRandom::new(),
    )
    .unwrap();
    let encoded = base64_lines(pkcs8.as_ref());
    std::fs::write(
        &other,
        format!("-----BEGIN PRIVATE KEY-----\n{encoded}-----END PRIVATE KEY-----\n"),
    )
    .unwrap();
    std::fs::set_permissions(&other, std::fs::Permissions::from_mode(0o600)).unwrap();
    let err = ServerAuth::new()
        .with_tls_files(fixture("server.pem"), &other)
        .unwrap_err()
        .to_string();
    assert!(err.contains("do not make a server identity"), "{err}");
    // A certificate file with no certificate.
    let err = ServerAuth::new()
        .with_tls_files(&not_pem, &key)
        .unwrap_err()
        .to_string();
    assert!(err.contains("holds no CERTIFICATE"), "{err}");
    assert!(!format!("{:?}", with_test_tls(ServerAuth::new())).contains(&body));
}

fn base64_lines(data: &[u8]) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(TABLE[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    let mut lines = String::new();
    for line in out.as_bytes().chunks(64) {
        lines.push_str(std::str::from_utf8(line).unwrap());
        lines.push('\n');
    }
    lines
}

// ---- unauthenticated TLS peers are bounded ---------------------------------------------

/// The handshake deadline covers the TLS handshake: a peer that never sends
/// its ClientHello, one that trickles it, and one that stops after the
/// server's flight are all shut down at the deadline.
#[test]
fn a_stalled_tls_handshake_is_cut_off_at_the_total_deadline() {
    let server = Net::Wss.serve(FerroStore::in_memory(), psk_auth());
    let deadline = Duration::from_millis(600);
    server.set_handshake_deadline(deadline);
    let addr = server.local_addr();
    let in_time = |closed: Duration, what: &str| {
        assert!(
            closed >= deadline,
            "{what}: cut off after {closed:?}, early"
        );
        assert!(
            closed < deadline + Duration::from_secs(2),
            "{what}: cut off after {closed:?}"
        );
    };

    // Connects, sends nothing.
    let started = Instant::now();
    let mut silent = TcpStream::connect(addr).unwrap();
    assert!(closed_within(&mut silent, Duration::from_secs(5)), "silent");
    in_time(started.elapsed(), "silent");

    // A real ClientHello, one byte every 50 ms.
    let config = tls::client_config(&Verify::CaFile(test_ca())).unwrap();
    let name = tls::server_name(&addr.to_string()).unwrap();
    let mut conn = rustls::ClientConnection::new(Arc::clone(&config), name.clone()).unwrap();
    let mut hello = Vec::new();
    conn.write_tls(&mut hello).unwrap();
    assert!(hello.len() > 100, "a ClientHello of {} bytes", hello.len());
    let started = Instant::now();
    let mut trickle = TcpStream::connect(addr).unwrap();
    trickle
        .set_read_timeout(Some(Duration::from_millis(20)))
        .unwrap();
    let mut closed = None;
    for byte in &hello {
        if trickle.write_all(std::slice::from_ref(byte)).is_err() {
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
    in_time(
        closed.expect("the trickling ClientHello was never cut off"),
        "trickle",
    );

    // The whole ClientHello, then silence after the server's answer: the
    // server waits mid-handshake for the client's Finished.
    let mut conn = rustls::ClientConnection::new(config, name).unwrap();
    let mut hello = Vec::new();
    conn.write_tls(&mut hello).unwrap();
    let started = Instant::now();
    let mut stalled = TcpStream::connect(addr).unwrap();
    stalled.write_all(&hello).unwrap();
    stalled
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut flight = [0u8; 4096];
    let n = stalled.read(&mut flight).unwrap();
    assert!(
        n > 0 && flight[0] == 0x16,
        "the server answered with a handshake record"
    );
    assert!(
        closed_within(&mut stalled, Duration::from_secs(5)),
        "stalled"
    );
    in_time(started.elapsed(), "mid-handshake");

    wait_handshakes(&server, 0);
    // A client that authenticates at once is served, and stays served past
    // the deadline.
    let client = Net::Wss.psk_client(addr);
    thread::sleep(deadline + Duration::from_millis(200));
    client.put_doc("docs", "a", &json!(1)).unwrap();
}

// ---- the client's plaintext rule ---------------------------------------------------------

/// Over `wss://` a credential goes to any address; over `ws://` only to
/// loopback unless allowed (spec §6.2, kept).
#[test]
fn over_wss_a_credential_goes_to_any_host() {
    let server = Net::Wss.serve(FerroStore::in_memory(), psk_auth());
    let plain = Net::Ws.serve(FerroStore::in_memory(), psk_auth());
    auth::LOOPBACK_IS_REMOTE.with(|flag| flag.set(true));
    let tls_client = FerroStore::connect_with(
        &Net::Wss.url(server.local_addr()),
        &psk_options().ca_file(test_ca()),
    );
    let plain_err = FerroStore::connect_with(&Net::Ws.url(plain.local_addr()), &psk_options())
        .err()
        .map(|err| err.to_string());
    auth::LOOPBACK_IS_REMOTE.with(|flag| flag.set(false));
    tls_client.unwrap().put_doc("docs", "a", &json!(1)).unwrap();
    let plain_err = plain_err.expect("plain ws:// sent a PSK to a remote address");
    assert!(
        plain_err.contains("refused to send a PSK handshake"),
        "{plain_err}"
    );
}

#[test]
fn a_wss_uri_names_its_scheme_in_an_error_and_takes_the_ws_query() {
    let err = error_of(FerroStore::connect_uri("wss://example.test/store"));
    assert!(err.contains("wss"), "{err}");
    assert!(!err.contains("unknown"), "{err}");
    let err = error_of(FerroStore::connect_uri("wss://"));
    assert!(err.contains("wss uri is missing a host"), "{err}");
    // The query rules of ws:// hold: no secret, no file.
    let server = Net::Wss.serve(FerroStore::in_memory(), psk_auth());
    let err = error_of(FerroStore::connect_with(
        &format!("{}?token=abc", Net::Wss.url(server.local_addr())),
        &ConnectOptions::new().ca_file(test_ca()),
    ));
    assert!(err.contains("never"), "{err}");
    let unset = "FERROSTORE_TEST_UNSET_6C";
    let err = error_of(FerroStore::connect_with(
        &format!("{}?token_env={unset}", Net::Wss.url(server.local_addr())),
        &ConnectOptions::new().ca_file(test_ca()),
    ));
    assert!(err.contains("is not set"), "{err}");
}

// ---- the bind policy ----------------------------------------------------------------------

fn bind(addr: &str, auth: ServerAuth) -> Result<FerroStoreServer, fl_core::StoreError> {
    FerroStoreServer::bind_tcp(addr, FerroStore::in_memory(), auth)
}

/// A non-loopback address of this host, and its interface, if it has one
/// that can be bound (not link-local).
fn a_routable_address() -> Option<(String, IpAddr)> {
    bind::interfaces().unwrap().into_iter().find_map(|iface| {
        let usable = match iface.ip {
            IpAddr::V4(v4) => !v4.is_loopback() && !v4.is_unspecified() && !v4.is_link_local(),
            IpAddr::V6(v6) => {
                !v6.is_loopback() && !v6.is_unspecified() && (v6.segments()[0] & 0xffc0) != 0xfe80
            }
        };
        usable.then_some((iface.name, iface.ip))
    })
}

/// The interface `SO_BINDTODEVICE` names on this process's listening socket
/// bound to `addr` (found in `/proc/self/fd`), `None` when it names none.
#[cfg(target_os = "linux")]
fn listener_device(addr: SocketAddr) -> Option<String> {
    use std::os::fd::{AsFd, BorrowedFd};
    for entry in std::fs::read_dir("/proc/self/fd").unwrap() {
        let Ok(raw) = entry.unwrap().file_name().to_string_lossy().parse::<i32>() else {
            continue;
        };
        // SAFETY: the descriptor is only duplicated here; a descriptor
        // another thread closes meanwhile makes the dup fail, nothing more.
        let Ok(owned) = unsafe { BorrowedFd::borrow_raw(raw) }.try_clone_to_owned() else {
            continue;
        };
        let stream = TcpStream::from(owned);
        // A listener has this local address and no peer (an accepted
        // connection has the same local address and a peer).
        if stream.local_addr().ok() == Some(addr) && stream.peer_addr().is_err() {
            return bind::bound_device(stream.as_fd()).unwrap();
        }
    }
    panic!("no listening socket of this process is bound to {addr}")
}

#[test]
fn plain_ws_binds_loopback_or_a_named_interface_and_never_every_interface() {
    // Every interface: refused without TLS, whatever is named.
    for addr in ["0.0.0.0:0", "[::]:0"] {
        for auth in [psk_auth(), psk_auth().allow_plaintext_on(["lo", "eth0"])] {
            let err = error_of(bind(addr, auth));
            assert!(err.contains("every interface"), "{addr}: {err}");
            assert!(err.contains("with_tls_files"), "{addr}: {err}");
        }
    }
    // Loopback: allowed, by address or by a name that resolves to it.
    bind("127.0.0.1:0", psk_auth()).unwrap();
    bind("localhost:0", psk_auth()).unwrap();
    // `lo` named: allowed, and the listener is pinned to `lo`
    // (SO_BINDTODEVICE); it still serves.
    let pinned = bind("127.0.0.1:0", psk_auth().allow_plaintext_on(["lo"])).unwrap();
    #[cfg(target_os = "linux")]
    assert_eq!(listener_device(pinned.local_addr()).as_deref(), Some("lo"));
    let client =
        FerroStore::connect_with(&format!("ws://{}", pinned.local_addr()), &psk_options()).unwrap();
    client.put_doc("docs", "a", &json!(1)).unwrap();
    // Nothing named, or TLS: not pinned.
    #[cfg(target_os = "linux")]
    for auth in [
        psk_auth(),
        with_test_tls(psk_auth()).allow_plaintext_on(["lo"]),
    ] {
        let served = bind("127.0.0.1:0", auth).unwrap();
        assert_eq!(listener_device(served.local_addr()), None);
    }
    // Authentication is still required.
    let err = error_of(bind(
        "127.0.0.1:0",
        ServerAuth::new().allow_plaintext_on(["lo"]),
    ));
    assert!(err.contains("needs authentication"), "{err}");
    // TLS on a Unix socket is refused, as is allow_plaintext_on.
    let dir = tempfile::tempdir().unwrap();
    for auth in [
        with_test_tls(psk_auth()),
        psk_auth().allow_plaintext_on(["lo"]),
    ] {
        let err = error_of(FerroStoreServer::bind_unix_with(
            dir.path().join("s.sock"),
            FerroStore::in_memory(),
            auth,
        ));
        assert!(err.contains("bind_tcp"), "{err}");
    }
    // With TLS every address is allowed, the unspecified one included: the
    // rule, checked without listening on every interface (the matcher test
    // in bind.rs covers the plaintext rule for other interfaces).
    let tls = with_test_tls(psk_auth());
    for addr in ["0.0.0.0:0", "[::]:0", "192.0.2.1:0"] {
        assert_eq!(
            tls.check_bind(addr.parse().unwrap()).unwrap(),
            None,
            "{addr}"
        );
    }
}

/// Live binds on a non-loopback address of this host and on `0.0.0.0`:
/// they listen where other hosts can reach them, so they run only with
/// `FERROSTORE_TEST_LAN_BIND=1`.
#[test]
fn plain_ws_binds_a_named_lan_interface_when_asked() {
    if std::env::var("FERROSTORE_TEST_LAN_BIND").as_deref() != Ok("1") {
        eprintln!("skipped: set FERROSTORE_TEST_LAN_BIND=1 to bind a LAN address");
        return;
    }
    // An address of another interface: refused, refused when only `lo` is
    // named, and allowed (pinned to it) when its own interface is; with
    // TLS, allowed.
    match a_routable_address() {
        Some((name, ip)) => {
            let addr = SocketAddr::new(ip, 0).to_string();
            let err = error_of(bind(&addr, psk_auth()));
            assert!(err.contains("not a loopback address"), "{addr}: {err}");
            let err = error_of(bind(&addr, psk_auth().allow_plaintext_on(["lo"])));
            assert!(err.contains("allow_plaintext_on"), "{addr}: {err}");
            let served = bind(&addr, psk_auth().allow_plaintext_on([name.clone()])).unwrap();
            assert_eq!(served.local_addr().ip(), ip);
            #[cfg(target_os = "linux")]
            assert_eq!(listener_device(served.local_addr()), Some(name.clone()));
            // Its client is served (an allowed plaintext interface still
            // authenticates; the client side needs allow_plaintext too).
            let client = FerroStore::connect_with(
                &format!("ws://{}", served.local_addr()),
                &psk_options().allow_plaintext(true),
            )
            .unwrap();
            client.put_doc("docs", "a", &json!(1)).unwrap();
            let err = error_of(FerroStore::connect_uri(&format!(
                "ws://{}",
                served.local_addr()
            )));
            assert!(err.contains("requires authentication"), "{err}");
            bind(&addr, with_test_tls(psk_auth())).unwrap();
        }
        None => eprintln!("this host has no routable address"),
    }
    let any = bind("0.0.0.0:0", with_test_tls(psk_auth())).unwrap();
    let port = any.local_addr().port();
    let client = FerroStore::connect_with(
        &format!("wss://127.0.0.1:{port}"),
        &psk_options().ca_file(test_ca()),
    )
    .unwrap();
    client.put_doc("docs", "a", &json!(1)).unwrap();
}
