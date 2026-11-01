//! `wss://`: TLS for a served FerroStore (spec §6.3).
//!
//! * **Server.** [`ServerTls`] is a certificate chain and its private key,
//!   from PEM files; the key file obeys the secret-file rules of a PSK file.
//!   TLS 1.3 is preferred and TLS 1.2 allowed, both with the `ring`
//!   provider; TLS 1.2 requires the extended master secret (RFC 7627), so
//!   its exporter is unique to the connection.
//! * **Client.** [`Verify`] picks how the server's certificate is checked:
//!   the bundled Mozilla root certificates (`webpki-roots`, as FerroWire),
//!   compiled into the binary and not the OS trust store; only the CA in
//!   one file (an enterprise or private CA); or a pinned SHA-256 of the
//!   server's SubjectPublicKeyInfo ([`PinnedSpki`]).
//! * **Transport.** [`Wire`] is a plain `TcpStream` or one half of a TLS
//!   stream ([`TlsStream`]). A `rustls` stream cannot be cloned, so the
//!   halves share one connection state behind a mutex, and each half keeps
//!   its own clone of the socket. See [`TlsStream`] for why no lock is held
//!   across a blocking socket call.
//! * **Channel binding.** [`TlsStream::exporter`] is the RFC 8446 / RFC 5705
//!   exporter of the connection under [`EXPORTER_LABEL`], 32 bytes: the PSK
//!   proofs over TLS cover it (see `auth.rs`).

use super::layout::backend;
use fl_core::StoreError;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{CryptoProvider, WebPkiSupportedAlgorithms};
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName, UnixTime};
use rustls::server::ParsedCertificate;
use rustls::{
    ClientConfig, ClientConnection, Connection, DigitallySignedStruct, RootCertStore, ServerConfig,
    ServerConnection, SignatureScheme, SupportedProtocolVersion,
};
use sha2::{Digest, Sha256};
use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, TryLockError};
use std::time::Duration;
use zeroize::Zeroize;

/// The exporter label of the PSK channel binding (RFC 5705 / RFC 8446
/// §7.5), with no context: the binding is 32 bytes of
/// `export_keying_material(label = "EXPORTER-fl-store-psk", context = None)`.
pub(super) const EXPORTER_LABEL: &[u8] = b"EXPORTER-fl-store-psk";
/// The length of the channel binding, in bytes.
pub(super) const BINDING_LEN: usize = 32;
/// The channel binding of one TLS connection.
pub(super) type Binding = [u8; BINDING_LEN];

/// TLS 1.3 first, then TLS 1.2.
static VERSIONS: &[&SupportedProtocolVersion] = &[&rustls::version::TLS13, &rustls::version::TLS12];
/// The largest certificate or CA file read.
const MAX_CERT_FILE: u64 = 1024 * 1024;
/// The most plaintext one TLS write takes: one TLS record's worth.
const WRITE_CHUNK: usize = 16 * 1024;
/// The most ciphertext one socket read takes.
const READ_CHUNK: usize = 16 * 1024;
/// The longest a best-effort send (a fatal alert the reader queued, or a
/// `close_notify` at shutdown) may wait for the socket.
const BEST_EFFORT_SEND: Duration = Duration::from_millis(100);

fn provider() -> Arc<CryptoProvider> {
    Arc::new(rustls::crypto::ring::default_provider())
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The guard of `mutex` when no one holds it; `None` when someone does.
fn try_lock<T>(mutex: &Mutex<T>) -> Option<MutexGuard<'_, T>> {
    match mutex.try_lock() {
        Ok(guard) => Some(guard),
        Err(TryLockError::Poisoned(poisoned)) => Some(poisoned.into_inner()),
        Err(TryLockError::WouldBlock) => None,
    }
}

fn shown(path: &Path) -> String {
    crate::private::shown_path(path)
}

// ---- the server's certificate ---------------------------------------------------------

/// The server's TLS configuration: a certificate chain and its key.
#[derive(Clone)]
pub(super) struct ServerTls {
    config: Arc<ServerConfig>,
}

impl std::fmt::Debug for ServerTls {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ServerTls(<certificate and key>)")
    }
}

impl ServerTls {
    /// The chain in the PEM file `cert` (leaf first) and the private key in
    /// the PEM file `key` (PKCS#8, PKCS#1 or SEC1). The key file is a secret
    /// file: a regular file of mode 0600 or stricter, owned by this uid or
    /// root (`crate::private::read_secret_file`). No error repeats any of
    /// its bytes.
    ///
    /// **Key bytes in memory.** The PEM text read from the file is zeroed
    /// (`zeroize`) as soon as it is decoded. The decoded DER goes to
    /// `rustls`, whose `ring` key provider zeroes it after it loads the key.
    /// Not every copy is zeroed: the PEM decoder of `rustls-pki-types`
    /// frees its own base64 buffer without zeroing it, and the loaded
    /// signing key (`ring`'s key pair) holds the key in memory for the
    /// server's lifetime. Anyone who can read this process's memory can read
    /// the key. It is never logged, and no error or `Debug` output holds it.
    pub(super) fn from_pem_files(cert: &Path, key: &Path) -> Result<Self, StoreError> {
        let chain = read_certs(cert, "certificate")?;
        // Built before the key is read, so no error after it drops the key
        // without zeroing it.
        let builder = ServerConfig::builder_with_provider(provider())
            .with_protocol_versions(VERSIONS)
            .map_err(|err| backend(format!("TLS versions: {err}")))?
            .with_no_client_auth();
        let mut key_bytes = crate::private::read_secret_file(key, MAX_CERT_FILE)
            .map_err(|err| backend(format!("read the TLS key file {}: {err}", shown(key))))?;
        let parsed = PrivateKeyDer::from_pem_slice(&key_bytes);
        key_bytes.zeroize();
        let key_der = parsed.map_err(|_| {
            backend(format!(
                "the TLS key file {} holds no PEM private key (PKCS#8, PKCS#1 or SEC1)",
                shown(key)
            ))
        })?;
        // `with_single_cert` takes the DER by value and the `ring` provider
        // zeroes it, on success and on failure.
        let mut config = builder.with_single_cert(chain, key_der).map_err(|err| {
            backend(format!(
                "the TLS certificate {} and key {} do not make a server identity: {err}",
                shown(cert),
                shown(key)
            ))
        })?;
        config.require_ems = true;
        Ok(Self {
            config: Arc::new(config),
        })
    }
}

/// Every certificate in the PEM file at `path`; at least one.
fn read_certs(path: &Path, what: &str) -> Result<Vec<CertificateDer<'static>>, StoreError> {
    let bytes = read_public(path, what)?;
    let certs = CertificateDer::pem_slice_iter(&bytes)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| backend(format!("the {what} file {} is not PEM", shown(path))))?;
    if certs.is_empty() {
        return Err(backend(format!(
            "the {what} file {} holds no CERTIFICATE",
            shown(path)
        )));
    }
    Ok(certs)
}

/// A public file (a certificate or a CA), at most [`MAX_CERT_FILE`] bytes.
fn read_public(path: &Path, what: &str) -> Result<Vec<u8>, StoreError> {
    let file = std::fs::File::open(path)
        .map_err(|err| backend(format!("read the {what} file {}: {err}", shown(path))))?;
    let mut bytes = Vec::new();
    file.take(MAX_CERT_FILE + 1)
        .read_to_end(&mut bytes)
        .map_err(|err| backend(format!("read the {what} file {}: {err}", shown(path))))?;
    if bytes.len() as u64 > MAX_CERT_FILE {
        return Err(backend(format!(
            "the {what} file {} is over {MAX_CERT_FILE} bytes",
            shown(path)
        )));
    }
    Ok(bytes)
}

// ---- the client's verification ----------------------------------------------------------

/// How a `wss://` client checks the server's certificate.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) enum Verify {
    /// A chain to one of the bundled Mozilla root certificates
    /// (`webpki-roots`), valid now, for the URI's host. The roots are
    /// compiled into the binary; the OS trust store is not read, so a
    /// server whose certificate comes from an enterprise or private CA needs
    /// [`Verify::CaFile`] (`ConnectOptions::ca_file`).
    #[default]
    BundledRoots,
    /// A chain to a CA in this PEM file alone, valid now, for the URI's host.
    CaFile(PathBuf),
    /// A leaf whose SubjectPublicKeyInfo has this SHA-256, and a handshake
    /// signed by that key ([`PinnedSpki`]).
    Pin([u8; 32]),
}

/// The client configuration for `verify`.
pub(super) fn client_config(verify: &Verify) -> Result<Arc<ClientConfig>, StoreError> {
    let provider = provider();
    let builder = ClientConfig::builder_with_provider(Arc::clone(&provider))
        .with_protocol_versions(VERSIONS)
        .map_err(|err| backend(format!("TLS versions: {err}")))?;
    let mut config = match verify {
        Verify::BundledRoots => {
            // The Mozilla roots compiled in by `webpki-roots`, not the OS
            // trust store.
            let mut roots = RootCertStore::empty();
            roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
            builder.with_root_certificates(roots).with_no_client_auth()
        }
        Verify::CaFile(path) => {
            let mut roots = RootCertStore::empty();
            for cert in read_certs(path, "CA")? {
                roots.add(cert).map_err(|err| {
                    backend(format!(
                        "the CA file {} holds a certificate that is not a CA: {err}",
                        shown(path)
                    ))
                })?;
            }
            builder.with_root_certificates(roots).with_no_client_auth()
        }
        Verify::Pin(pin) => builder
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(PinnedSpki {
                pin: *pin,
                algorithms: provider.signature_verification_algorithms,
            }))
            .with_no_client_auth(),
    };
    config.require_ems = true;
    Ok(Arc::new(config))
}

/// A server certificate verifier that trusts one key, not a CA.
///
/// It accepts the server when the SHA-256 of its leaf certificate's
/// SubjectPublicKeyInfo (the DER of the whole SPKI structure, as
/// `openssl x509 -pubkey | openssl pkey -pubin -outform DER | sha256sum`
/// prints) equals the pin. The pin replaces the chain, the name check and the
/// validity dates, as an SSH host key does: the pinned key is the server's
/// identity, so a self-signed certificate, or one renewed over the same
/// key, works. It still requires the TLS handshake to be signed by that key:
/// both signature checks match the certificate against the pin again, then
/// verify the signature with the certificate's (the pinned) public key under
/// the provider's algorithms. So a peer that has the certificate but not
/// its private key cannot pass.
#[derive(Debug)]
pub(super) struct PinnedSpki {
    pin: [u8; 32],
    algorithms: WebPkiSupportedAlgorithms,
}

impl PinnedSpki {
    fn check(&self, cert: &CertificateDer<'_>) -> Result<(), rustls::Error> {
        let parsed = ParsedCertificate::try_from(cert)?;
        let digest: [u8; 32] = Sha256::digest(parsed.subject_public_key_info().as_ref()).into();
        if digest == self.pin {
            Ok(())
        } else {
            Err(rustls::Error::InvalidCertificate(
                rustls::CertificateError::ApplicationVerificationFailure,
            ))
        }
    }
}

impl ServerCertVerifier for PinnedSpki {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        self.check(end_entity)?;
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        self.check(cert)?;
        rustls::crypto::verify_tls12_signature(message, cert, dss, &self.algorithms)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        self.check(cert)?;
        rustls::crypto::verify_tls13_signature(message, cert, dss, &self.algorithms)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.algorithms.supported_schemes()
    }
}

/// 32 bytes from 64 hex digits of either case, `:` between bytes allowed
/// (`openssl` prints a fingerprint that way).
pub(super) fn parse_pin(text: &str) -> Result<[u8; 32], StoreError> {
    let digits: String = text.chars().filter(|c| *c != ':').collect();
    let refused = || {
        backend(
            "a SPKI pin is the SHA-256 of the server's SubjectPublicKeyInfo: 64 hex digits \
             (`openssl x509 -in cert.pem -pubkey -noout | openssl pkey -pubin -outform DER \
             | sha256sum`)",
        )
    };
    if digits.len() != 64 {
        return Err(refused());
    }
    let mut out = [0u8; 32];
    for (slot, pair) in out.iter_mut().zip(digits.as_bytes().as_chunks::<2>().0) {
        let pair = std::str::from_utf8(pair).map_err(|_| refused())?;
        *slot = u8::from_str_radix(pair, 16).map_err(|_| refused())?;
    }
    Ok(out)
}

/// The name the client verifies and sends as SNI: the URI's host, without
/// brackets for an IPv6 address.
pub(super) fn server_name(host_port: &str) -> Result<ServerName<'static>, StoreError> {
    let host = match host_port.strip_prefix('[') {
        Some(rest) => rest.split_once(']').map_or(rest, |(host, _)| host),
        None => host_port
            .rsplit_once(':')
            .map_or(host_port, |(host, _)| host),
    };
    ServerName::try_from(host.to_string()).map_err(|_| {
        backend(format!(
            "wss host `{host}` is not a DNS name or an IP address"
        ))
    })
}

// ---- the transport ------------------------------------------------------------------------

/// One connection's TLS state, shared by its halves.
///
/// Lock order: `send`, then `conn`. A writer takes both; a reader takes
/// `conn` alone; neither lock is held across a blocking socket call but
/// `send` across the socket write of the bytes it just encrypted (so
/// records reach the socket in the order they were sealed). See
/// [`TlsStream`].
struct TlsState {
    conn: Mutex<Connection>,
    send: Mutex<()>,
}

/// One half of a TLS connection: the reading half (which also writes) or
/// a writing half. Each holds its own `try_clone` of the socket.
///
/// **Why this cannot deadlock.** A `rustls` connection is one state machine
/// for both directions, and the server and client read on one thread while
/// others write. A reader that blocked in the socket read while holding the
/// state would stop every writer; a writer blocked on a full socket buffer
/// while holding the state would stop the reader from draining the socket,
/// and with the peer doing the same both sides would wait forever. So:
///
/// * the reader waits for socket bytes holding no lock, then takes `conn`
///   only to feed those bytes in, process them and take plaintext out;
/// * a writer takes `send`, then `conn` only to seal at most
///   [`WRITE_CHUNK`] bytes and drain every queued record into a buffer,
///   releases `conn`, and writes the buffer to the socket holding `send`
///   alone (`send` keeps two writers' records in sealing order);
/// * the reader does not write to the socket after the handshake: a record
///   that reading queues (a TLS 1.3 key update reply) stays in the
///   connection and goes out, in order, with the next write. The one
///   exception is a fatal error: the alert that names it is sent once,
///   best-effort ([`TlsStream::send_best_effort`]);
/// * a shutdown sends `close_notify` the same best-effort way, then shuts
///   the socket down.
///
/// A best-effort send never waits for a lock: it takes `send`, then
/// `conn`, with `try_lock`, and sends nothing when either is held (a
/// writer is blocked on the socket, or the handshake runs). It writes with
/// a [`BEST_EFFORT_SEND`] socket write timeout, so it never blocks for
/// longer. Every other lock is held for in-memory work only, except `send`,
/// which only writers take; the reader never waits for it. So a blocked
/// socket write never stops the reader, and a blocked socket read never
/// stops a writer.
///
/// The handshake runs on the reading half before the other halves are used
/// (the server's connection thread in `prepare`, the client before its
/// upgrade), so it may hold `conn` across its socket calls; a shutdown of
/// the socket (the server's handshake deadline) ends it like any other read.
pub(super) struct TlsStream {
    state: Arc<TlsState>,
    sock: TcpStream,
    /// `true` for the one half that reads.
    reads: bool,
    /// Ciphertext read from the socket and not yet fed to the connection.
    inbuf: Vec<u8>,
    /// The socket read end-of-stream.
    eof: bool,
}

impl TlsStream {
    fn pair(conn: Connection, sock: TcpStream) -> io::Result<(Self, Self)> {
        let writer_sock = sock.try_clone()?;
        let state = Arc::new(TlsState {
            conn: Mutex::new(conn),
            send: Mutex::new(()),
        });
        let reader = Self {
            state: Arc::clone(&state),
            sock,
            reads: true,
            inbuf: Vec::new(),
            eof: false,
        };
        let writer = Self {
            state,
            sock: writer_sock,
            reads: false,
            inbuf: Vec::new(),
            eof: false,
        };
        Ok((reader, writer))
    }

    /// The reading and the writing half of a server connection on `sock`.
    /// Nothing is read or written: the handshake runs in [`Self::handshake`]
    /// on the connection's own thread.
    pub(super) fn server(sock: TcpStream, tls: &ServerTls) -> Result<(Self, Self), StoreError> {
        let conn = ServerConnection::new(Arc::clone(&tls.config))
            .map_err(|err| backend(format!("TLS server connection: {err}")))?;
        Self::pair(Connection::Server(conn), sock)
            .map_err(|err| backend(format!("clone tcp stream: {err}")))
    }

    /// The reading and the writing half of a client connection on `sock`
    /// to `name`, its handshake done.
    pub(super) fn client(
        sock: TcpStream,
        config: Arc<ClientConfig>,
        name: ServerName<'static>,
    ) -> Result<(Self, Self), StoreError> {
        let conn = ClientConnection::new(config, name)
            .map_err(|err| backend(format!("TLS client connection: {err}")))?;
        let (mut reader, writer) = Self::pair(Connection::Client(conn), sock)
            .map_err(|err| backend(format!("clone tcp stream: {err}")))?;
        reader.handshake()?;
        Ok((reader, writer))
    }

    /// Run the TLS handshake to its end (the reading half only). On a
    /// failure the alert that names it is sent, as far as the socket takes
    /// it, and the error names the TLS reason.
    pub(super) fn handshake(&mut self) -> Result<(), StoreError> {
        if !self.reads {
            return Err(backend(
                "the writing half of a TLS stream runs no handshake",
            ));
        }
        let mut conn = lock(&self.state.conn);
        let failed = |err: &dyn std::fmt::Display| backend(format!("TLS handshake: {err}"));
        while conn.is_handshaking() {
            flush_records(&mut conn, &mut self.sock).map_err(|err| failed(&err))?;
            if !conn.is_handshaking() {
                break;
            }
            let n = conn.read_tls(&mut self.sock).map_err(|err| failed(&err))?;
            if n == 0 {
                return Err(failed(&"the peer closed the connection"));
            }
            if let Err(err) = conn.process_new_packets() {
                let _ = flush_records(&mut conn, &mut self.sock);
                return Err(failed(&err));
            }
        }
        flush_records(&mut conn, &mut self.sock).map_err(|err| failed(&err))
    }

    /// The channel binding: 32 bytes of the connection's exporter under
    /// [`EXPORTER_LABEL`], no context. An error before the handshake ended.
    pub(super) fn exporter(&self) -> Result<Binding, StoreError> {
        lock(&self.state.conn)
            .export_keying_material([0u8; BINDING_LEN], EXPORTER_LABEL, None)
            .map_err(|err| backend(format!("TLS exporter: {err}")))
    }

    fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        self.sock.set_read_timeout(timeout)
    }

    /// Send `close_notify` (best-effort, see [`Self::send_best_effort`]),
    /// then shut the socket down both ways. A peer that reads it sees the
    /// end of the stream as a clean close, not a truncation.
    fn shutdown_both(&self) {
        self.send_best_effort(true);
        let _ = self.sock.shutdown(Shutdown::Both);
    }

    /// Send the records the connection holds queued (a fatal alert), after
    /// queuing `close_notify` when `close` is set (`rustls` sends nothing
    /// after a fatal alert, and one `close_notify` at most). Best-effort and
    /// bounded: nothing is sent when a writer holds `send` or the handshake
    /// holds `conn` (`try_lock`; their own sends go first), or during the
    /// handshake, and the socket write waits at most [`BEST_EFFORT_SEND`].
    /// Whatever fails is dropped: the connection is ending.
    fn send_best_effort(&self, close: bool) {
        let Some(_send) = try_lock(&self.state.send) else {
            return;
        };
        let mut sealed = Vec::new();
        {
            let Some(mut conn) = try_lock(&self.state.conn) else {
                return;
            };
            if conn.is_handshaking() {
                return;
            }
            if close {
                conn.send_close_notify();
            }
            while conn.wants_write() {
                if conn.write_tls(&mut sealed).is_err() {
                    break;
                }
            }
        }
        if sealed.is_empty() {
            return;
        }
        // `SO_SNDTIMEO` is the socket's, shared by every clone; no writer
        // is in a socket write while `send` is held here, and the previous
        // value is put back.
        let previous = self.sock.write_timeout().ok().flatten();
        if self.sock.set_write_timeout(Some(BEST_EFFORT_SEND)).is_ok() {
            let _ = (&self.sock).write_all(&sealed);
            let _ = self.sock.set_write_timeout(previous);
        }
    }
}

/// Write every queued record of `conn` to `sock`.
fn flush_records(conn: &mut Connection, sock: &mut TcpStream) -> io::Result<()> {
    while conn.wants_write() {
        if conn.write_tls(sock)? == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "the socket took no TLS bytes",
            ));
        }
    }
    Ok(())
}

impl Read for TlsStream {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if !self.reads {
            return Err(io::Error::other(
                "the writing half of a TLS stream does not read",
            ));
        }
        if out.is_empty() {
            return Ok(0);
        }
        loop {
            {
                let mut conn = lock(&self.state.conn);
                loop {
                    match conn.reader().read(out) {
                        Ok(n) => return Ok(n),
                        Err(err) if err.kind() == io::ErrorKind::WouldBlock => {}
                        Err(err) => return Err(err),
                    }
                    if self.eof {
                        // The connection saw the socket's end and still
                        // asks for bytes: there are none.
                        return Err(io::ErrorKind::UnexpectedEof.into());
                    }
                    if self.inbuf.is_empty() {
                        break;
                    }
                    let mut pending = self.inbuf.as_slice();
                    let fed = conn.read_tls(&mut pending)?;
                    self.inbuf.drain(..fed);
                    if fed == 0 {
                        // A close_notify came: no more is read.
                        self.inbuf.clear();
                        self.eof = true;
                        continue;
                    }
                    if let Err(err) = conn.process_new_packets() {
                        // The alert that names the error goes to the peer,
                        // best-effort, before the error is returned.
                        drop(conn);
                        self.send_best_effort(false);
                        return Err(io::Error::new(io::ErrorKind::InvalidData, err));
                    }
                }
            }
            // No plaintext and no ciphertext to feed: wait for the socket,
            // holding no lock, so writers go on.
            let mut chunk = [0u8; READ_CHUNK];
            let n = self.sock.read(&mut chunk)?;
            if n == 0 {
                let mut conn = lock(&self.state.conn);
                // Tell the connection: its reader then answers `Ok(0)` after a
                // close_notify and `UnexpectedEof` without one.
                let _ = conn.read_tls(&mut io::empty());
                self.eof = true;
                continue;
            }
            self.inbuf.extend_from_slice(&chunk[..n]);
        }
    }
}

impl Write for TlsStream {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        if data.is_empty() {
            return Ok(0);
        }
        let _send = lock(&self.state.send);
        let take = data.len().min(WRITE_CHUNK);
        let mut sealed = Vec::with_capacity(take + 256);
        let taken = {
            let mut conn = lock(&self.state.conn);
            // Records the reader queued go first, in their order.
            while conn.wants_write() {
                conn.write_tls(&mut sealed)?;
            }
            let taken = conn.writer().write(&data[..take])?;
            while conn.wants_write() {
                conn.write_tls(&mut sealed)?;
            }
            taken
        };
        self.sock.write_all(&sealed)?;
        Ok(taken)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A connection's byte stream: plain TCP (`ws://`) or one half of TLS
/// (`wss://`).
pub(super) enum Wire {
    Plain(TcpStream),
    Tls(TlsStream),
}

impl Wire {
    pub(super) fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        match self {
            Wire::Plain(stream) => stream.set_read_timeout(timeout),
            Wire::Tls(stream) => stream.set_read_timeout(timeout),
        }
    }

    /// Shut the socket down both ways: a read blocked on it, the TLS
    /// handshake's included, returns.
    pub(super) fn shutdown_both(&self) {
        match self {
            Wire::Plain(stream) => {
                let _ = stream.shutdown(Shutdown::Both);
            }
            Wire::Tls(stream) => stream.shutdown_both(),
        }
    }

    /// Run the TLS handshake of a server's reading half (nothing for plain
    /// TCP, or a half whose handshake ran).
    pub(super) fn start(&mut self) -> Result<(), StoreError> {
        match self {
            Wire::Plain(_) => Ok(()),
            Wire::Tls(stream) => {
                if lock(&stream.state.conn).is_handshaking() {
                    stream.handshake()
                } else {
                    Ok(())
                }
            }
        }
    }

    /// `None` for plain TCP; the TLS channel binding otherwise.
    pub(super) fn channel_binding(&self) -> Result<Option<Binding>, StoreError> {
        match self {
            Wire::Plain(_) => Ok(None),
            Wire::Tls(stream) => stream.exporter().map(Some),
        }
    }
}

impl Read for Wire {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        match self {
            Wire::Plain(stream) => stream.read(out),
            Wire::Tls(stream) => stream.read(out),
        }
    }
}

impl Write for Wire {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        match self {
            Wire::Plain(stream) => stream.write(data),
            Wire::Tls(stream) => stream.write(data),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Wire::Plain(stream) => stream.flush(),
            Wire::Tls(stream) => stream.flush(),
        }
    }
}
