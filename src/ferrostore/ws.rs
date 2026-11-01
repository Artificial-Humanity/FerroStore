//! RFC 6455 handshake and frames for `ws://`. SHA-1 and base64 live here so
//! `fl-store` does not take a hub or codec dependency.
//!
//! One JSON payload is one WebSocket message. Client frames are masked.
//! Server frames are not. There is no 4-byte length prefix inside the payload.

use super::layout::backend;
use fl_core::StoreError;
use std::fs::File;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};

const WS_GUID: &str = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
pub(super) const MAX_PAYLOAD: usize = 16 * 1024 * 1024;
/// The largest upgrade request or response, headers included.
const MAX_HEADER_BYTES: usize = 16 * 1024;
/// The end of an HTTP header block.
const HEADER_END: &[u8; 4] = b"\r\n\r\n";

const OP_TEXT: u8 = 0x1;
const OP_BINARY: u8 = 0x2;
const OP_CLOSE: u8 = 0x8;
const OP_PING: u8 = 0x9;
const OP_PONG: u8 = 0xA;

#[derive(Debug)]
pub(super) enum WsEvent {
    Data(Vec<u8>),
    Ping(Vec<u8>),
    Close,
}

struct RawFrame {
    opcode: u8,
    payload: Vec<u8>,
}

/// Connect to `rest` (`host:port[/path]`) and upgrade. `bearer` is sent as
/// `Authorization: Bearer <token>`; the caller has checked that it is
/// visible ASCII, so it cannot end the header. `admit` is given the address
/// actually connected to before anything is written, and may refuse it.
#[cfg(test)]
pub(super) fn connect(
    rest: &str,
    bearer: Option<&str>,
    admit: impl FnOnce(SocketAddr) -> Result<(), StoreError>,
) -> Result<TcpStream, StoreError> {
    let (host_port, path) = split_target("ws", rest)?;
    check_bearer(bearer)?;
    let (mut stream, peer) = dial(host_port)?;
    admit(peer)?;
    upgrade(&mut stream, host_port, &path, bearer)?;
    Ok(stream)
}

/// `Err` unless `bearer` is absent or visible ASCII without spaces.
pub(super) fn check_bearer(bearer: Option<&str>) -> Result<(), StoreError> {
    if bearer.is_some_and(|token| token.is_empty() || !token.bytes().all(|b| b.is_ascii_graphic()))
    {
        return Err(backend("a bearer token is visible ASCII without spaces"));
    }
    Ok(())
}

/// The TCP connection to `host_port`, with no delay, and the address it
/// reached.
pub(super) fn dial(host_port: &str) -> Result<(TcpStream, SocketAddr), StoreError> {
    let stream = TcpStream::connect(host_port)
        .map_err(|err| backend(format!("connect {host_port}: {err}")))?;
    let peer = stream
        .peer_addr()
        .map_err(|err| backend(format!("peer address of {host_port}: {err}")))?;
    stream
        .set_nodelay(true)
        .map_err(|err| backend(format!("set nodelay: {err}")))?;
    Ok((stream, peer))
}

/// Send the upgrade request for `path` on `stream` and check the answer.
pub(super) fn upgrade<S: Read + Write>(
    stream: &mut S,
    host_port: &str,
    path: &str,
    bearer: Option<&str>,
) -> Result<(), StoreError> {
    check_bearer(bearer)?;
    let mut key_bytes = [0u8; 16];
    random_bytes(&mut key_bytes)?;
    let sec_key = base64_encode(&key_bytes);
    let authorization = bearer.map_or_else(String::new, |token| {
        format!("Authorization: Bearer {token}\r\n")
    });
    let req = format!(
        "GET {path} HTTP/1.1\r\n\
         Host: {host_port}\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Key: {sec_key}\r\n\
         Sec-WebSocket-Version: 13\r\n\
         {authorization}\r\n"
    );
    stream
        .write_all(req.as_bytes())
        .map_err(|err| backend(format!("write handshake: {err}")))?;
    let headers = read_headers(stream)?;
    check_accept(&headers, &sec_key)
}

/// Read and answer the client's upgrade. Returns the values of its
/// `Authorization` headers, for the caller to judge (spec §6.2).
pub(super) fn accept<S: Read + Write>(stream: &mut S) -> Result<Vec<String>, StoreError> {
    let text = read_headers(stream)?;
    let mut lines = text.split("\r\n");
    let start = lines.next().unwrap_or("");
    if !start.starts_with("GET ") || !start.ends_with(" HTTP/1.1") {
        return Err(backend("websocket handshake request is not HTTP/1.1 GET"));
    }
    let mut host = false;
    let mut upgrade = false;
    let mut connection = false;
    let mut version = false;
    let mut key = None;
    let mut authorization = Vec::new();
    for line in lines {
        if line.is_empty() {
            break;
        }
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        if name.trim().eq_ignore_ascii_case("host") {
            host = !value.is_empty();
        } else if name.trim().eq_ignore_ascii_case("upgrade") {
            upgrade = header_has_token(value, "websocket");
        } else if name.trim().eq_ignore_ascii_case("connection") {
            connection = header_has_token(value, "upgrade");
        } else if name.trim().eq_ignore_ascii_case("sec-websocket-version") {
            version = value == "13";
        } else if name.trim().eq_ignore_ascii_case("sec-websocket-key") && !value.is_empty() {
            key = Some(value.to_string());
        } else if name.trim().eq_ignore_ascii_case("authorization") {
            authorization.push(value.to_string());
        }
    }
    let Some(key) = key else {
        return Err(backend("websocket handshake is missing Sec-WebSocket-Key"));
    };
    if !host || !upgrade || !connection || !version {
        return Err(backend("websocket handshake is missing a required header"));
    }
    let accept_key = compute_accept_key(&key);
    let resp = format!(
        "HTTP/1.1 101 Switching Protocols\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Accept: {accept_key}\r\n\r\n"
    );
    stream
        .write_all(resp.as_bytes())
        .map_err(|err| backend(format!("write handshake: {err}")))?;
    Ok(authorization)
}

/// The next data frame, ping or close. A frame that declares a payload
/// longer than `max_len` is refused before anything is allocated for it.
pub(super) fn read_event<S: Read>(
    stream: &mut S,
    client: bool,
    max_len: usize,
) -> Result<WsEvent, StoreError> {
    loop {
        let frame = read_raw(stream, client, max_len)?;
        match frame.opcode {
            OP_TEXT | OP_BINARY => return Ok(WsEvent::Data(frame.payload)),
            OP_PING => return Ok(WsEvent::Ping(frame.payload)),
            OP_PONG => {}
            OP_CLOSE => return Ok(WsEvent::Close),
            other => {
                return Err(backend(format!("unsupported websocket opcode {other:#x}")));
            }
        }
    }
}

pub(super) fn write_frame<S: Write>(
    stream: &mut S,
    client: bool,
    opcode: u8,
    payload: &[u8],
) -> Result<(), StoreError> {
    if payload.len() > MAX_PAYLOAD {
        return Err(backend("frame length is outside 1..=16MiB"));
    }
    let mut buf = Vec::with_capacity(14 + payload.len());
    buf.push(0x80 | opcode);
    let mask_bit: u8 = if client { 0x80 } else { 0 };
    let len = payload.len();
    if len < 126 {
        let small = u8::try_from(len).map_err(|_| backend("frame length exceeds u8"))?;
        buf.push(mask_bit | small);
    } else if len <= usize::from(u16::MAX) {
        let medium = u16::try_from(len).map_err(|_| backend("frame length exceeds u16"))?;
        buf.push(mask_bit | 126);
        buf.extend_from_slice(&medium.to_be_bytes());
    } else {
        buf.push(mask_bit | 127);
        let wide = u64::try_from(len).map_err(|_| backend("frame length exceeds u64"))?;
        buf.extend_from_slice(&wide.to_be_bytes());
    }
    if client {
        let mut mask = [0u8; 4];
        random_bytes(&mut mask)?;
        buf.extend_from_slice(&mask);
        for (i, byte) in payload.iter().enumerate() {
            buf.push(byte ^ mask[i & 3]);
        }
    } else {
        buf.extend_from_slice(payload);
    }
    stream.write_all(&buf).map_err(backend)?;
    Ok(())
}

/// The error of a frame longer than `max_len` (at most [`MAX_PAYLOAD`]).
pub(super) fn too_long(max_len: usize) -> StoreError {
    if max_len >= MAX_PAYLOAD {
        backend("frame length is outside 1..=16MiB")
    } else {
        backend(format!(
            "frame length is over {max_len} bytes, the limit before authentication"
        ))
    }
}

fn read_raw<S: Read>(stream: &mut S, client: bool, max_len: usize) -> Result<RawFrame, StoreError> {
    let max_len = max_len.min(MAX_PAYLOAD);
    let mut hdr = [0u8; 2];
    stream.read_exact(&mut hdr).map_err(backend)?;
    if hdr[0] & 0x70 != 0 {
        return Err(backend("websocket rsv bits are set"));
    }
    let fin = hdr[0] & 0x80 != 0;
    let opcode = hdr[0] & 0x0F;
    let masked = hdr[1] & 0x80 != 0;
    if client {
        if masked {
            return Err(backend("server websocket frame is masked"));
        }
    } else if !masked {
        return Err(backend("client websocket frame is not masked"));
    }
    let mut len = u64::from(hdr[1] & 0x7F);
    if len == 126 {
        let mut ext = [0u8; 2];
        stream.read_exact(&mut ext).map_err(backend)?;
        len = u64::from(u16::from_be_bytes(ext));
    } else if len == 127 {
        let mut ext = [0u8; 8];
        stream.read_exact(&mut ext).map_err(backend)?;
        len = u64::from_be_bytes(ext);
    }
    if len > max_len as u64 {
        return Err(too_long(max_len));
    }
    let control = opcode >= OP_CLOSE;
    if control && (!fin || len > 125) {
        return Err(backend("websocket control frame is fragmented or too long"));
    }
    if !control && !fin {
        return Err(backend("fragmented websocket frames are not supported"));
    }
    let mut mask = [0u8; 4];
    if masked {
        stream.read_exact(&mut mask).map_err(backend)?;
    }
    let len_usize = usize::try_from(len).map_err(|_| backend("frame length exceeds usize"))?;
    let mut payload = vec![0u8; len_usize];
    if len_usize > 0 {
        stream.read_exact(&mut payload).map_err(backend)?;
    }
    if masked {
        for (i, byte) in payload.iter_mut().enumerate() {
            *byte ^= mask[i & 3];
        }
    }
    Ok(RawFrame { opcode, payload })
}

/// `rest` (`host:port[/path]`) as its `host:port` and its path (`/` when
/// none). `scheme` (`ws` or `wss`) names the URI in an error.
pub(super) fn split_target<'a>(
    scheme: &str,
    rest: &'a str,
) -> Result<(&'a str, String), StoreError> {
    if rest.is_empty() {
        return Err(backend(format!("{scheme} uri is missing a host")));
    }
    let (host_port, path) = match rest.find('/') {
        Some(0) => return Err(backend(format!("{scheme} uri is missing a host"))),
        Some(index) => (&rest[..index], rest[index..].to_string()),
        None => (rest, "/".to_string()),
    };
    if !host_has_port(host_port) {
        return Err(backend(format!(
            "{scheme} uri `{host_port}` is missing a port"
        )));
    }
    Ok((host_port, path))
}

fn host_has_port(host_port: &str) -> bool {
    let port = if let Some(rest) = host_port.strip_prefix('[') {
        let Some((host, after)) = rest.split_once(']') else {
            return false;
        };
        if host.is_empty() {
            return false;
        }
        let Some(port) = after.strip_prefix(':') else {
            return false;
        };
        port
    } else {
        let Some((host, port)) = host_port.rsplit_once(':') else {
            return false;
        };
        if host.is_empty() {
            return false;
        }
        port
    };
    port.parse::<u16>().is_ok()
}

fn header_has_token(value: &str, token: &str) -> bool {
    value
        .split(',')
        .any(|part| part.trim().eq_ignore_ascii_case(token))
}

/// The upgrade's header block, up to [`MAX_HEADER_BYTES`], and not one
/// byte past its end: the frames that follow stay in the stream. Each read
/// asks for as many bytes as cannot reach past the end (up to 4), not one
/// at a time. The server bounds the whole handshake in time
/// ([`super::auth::HANDSHAKE_DEADLINE`]), so a peer that trickles is cut off.
fn read_headers<S: Read>(stream: &mut S) -> Result<String, StoreError> {
    let mut buf = Vec::with_capacity(512);
    let mut chunk = [0u8; HEADER_END.len()];
    while buf.len() < MAX_HEADER_BYTES {
        let want = safe_read_len(&buf).min(MAX_HEADER_BYTES - buf.len());
        let n = stream
            .read(&mut chunk[..want])
            .map_err(|err| backend(format!("read handshake: {err}")))?;
        if n == 0 {
            return Err(backend("websocket handshake ended before the headers"));
        }
        buf.extend_from_slice(&chunk[..n]);
        if buf.ends_with(HEADER_END) {
            return String::from_utf8(buf).map_err(|_| backend("websocket handshake is not utf-8"));
        }
    }
    Err(backend("websocket handshake headers are too large"))
}

/// The most bytes that can follow `buf` without reaching past the first
/// [`HEADER_END`]: 4 less the longest end of `buf` that begins it.
fn safe_read_len(buf: &[u8]) -> usize {
    (1..HEADER_END.len())
        .rev()
        .find(|&k| buf.ends_with(&HEADER_END[..k]))
        .map_or(HEADER_END.len(), |k| HEADER_END.len() - k)
}

fn check_accept(text: &str, sec_key: &str) -> Result<(), StoreError> {
    let mut lines = text.split("\r\n");
    let status = lines.next().unwrap_or("");
    if !status.starts_with("HTTP/1.1 101") {
        return Err(backend(format!(
            "websocket handshake expected 101, got `{status}`"
        )));
    }
    let expected = compute_accept_key(sec_key);
    let mut ok = false;
    for line in lines {
        if let Some((name, value)) = line.split_once(':')
            && name.trim().eq_ignore_ascii_case("sec-websocket-accept")
            && value.trim() == expected
        {
            ok = true;
        }
    }
    if !ok {
        return Err(backend(
            "websocket handshake Sec-WebSocket-Accept does not match",
        ));
    }
    Ok(())
}

fn compute_accept_key(sec_key: &str) -> String {
    let mut input = Vec::with_capacity(sec_key.len() + WS_GUID.len());
    input.extend_from_slice(sec_key.trim().as_bytes());
    input.extend_from_slice(WS_GUID.as_bytes());
    base64_encode(&sha1_digest(&input))
}

fn random_bytes(buf: &mut [u8]) -> Result<(), StoreError> {
    let mut file =
        File::open("/dev/urandom").map_err(|err| backend(format!("open /dev/urandom: {err}")))?;
    file.read_exact(buf)
        .map_err(|err| backend(format!("read /dev/urandom: {err}")))
}

fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    let mut index = 0;
    while index + 3 <= data.len() {
        let n = (u32::from(data[index]) << 16)
            | (u32::from(data[index + 1]) << 8)
            | u32::from(data[index + 2]);
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        out.push(TABLE[((n >> 6) & 63) as usize] as char);
        out.push(TABLE[(n & 63) as usize] as char);
        index += 3;
    }
    let rest = data.len() - index;
    if rest == 1 {
        let n = u32::from(data[index]) << 16;
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rest == 2 {
        let n = (u32::from(data[index]) << 16) | (u32::from(data[index + 1]) << 8);
        out.push(TABLE[((n >> 18) & 63) as usize] as char);
        out.push(TABLE[((n >> 12) & 63) as usize] as char);
        out.push(TABLE[((n >> 6) & 63) as usize] as char);
        out.push('=');
    }
    out
}

fn sha1_digest(data: &[u8]) -> [u8; 20] {
    let mut h0: u32 = 0x6745_2301;
    let mut h1: u32 = 0xEFCD_AB89;
    let mut h2: u32 = 0x98BA_DCFE;
    let mut h3: u32 = 0x1032_5476;
    let mut h4: u32 = 0xC3D2_E1F0;

    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 80];
        for (i, word) in chunk.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*word);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }

        let mut a = h0;
        let mut b = h1;
        let mut c = h2;
        let mut d = h3;
        let mut e = h4;

        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A82_7999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1u32),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDCu32),
                _ => (b ^ c ^ d, 0xCA62_C1D6u32),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }

        h0 = h0.wrapping_add(a);
        h1 = h1.wrapping_add(b);
        h2 = h2.wrapping_add(c);
        h3 = h3.wrapping_add(d);
        h4 = h4.wrapping_add(e);
    }

    let mut out = [0u8; 20];
    out[0..4].copy_from_slice(&h0.to_be_bytes());
    out[4..8].copy_from_slice(&h1.to_be_bytes());
    out[8..12].copy_from_slice(&h2.to_be_bytes());
    out[12..16].copy_from_slice(&h3.to_be_bytes());
    out[16..20].copy_from_slice(&h4.to_be_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    fn hex(bytes: &[u8]) -> String {
        const HEX: &[u8] = b"0123456789abcdef";
        let mut out = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            out.push(HEX[usize::from(*byte >> 4)] as char);
            out.push(HEX[usize::from(*byte & 0xf)] as char);
        }
        out
    }

    #[test]
    fn sha1_and_accept_key_match_the_rfc_vectors() {
        assert_eq!(
            hex(&sha1_digest(b"")),
            "da39a3ee5e6b4b0d3255bfef95601890afd80709"
        );
        assert_eq!(
            hex(&sha1_digest(b"abc")),
            "a9993e364706816aba3e25717850c26c9cd0d89d"
        );
        assert_eq!(base64_encode(b"hello"), "aGVsbG8=");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(
            compute_accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
    }

    #[test]
    fn handshake_round_trips_masked_frames_of_each_length_form() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream.set_nodelay(true).unwrap();
            accept(&mut stream).unwrap();
            for expected in [4usize, 200, 70_000] {
                let WsEvent::Data(payload) = read_event(&mut stream, false, MAX_PAYLOAD).unwrap()
                else {
                    panic!("expected a data frame");
                };
                assert_eq!(payload.len(), expected);
                assert!(payload.iter().all(|byte| *byte == b'z'));
                write_frame(&mut stream, false, OP_TEXT, &payload).unwrap();
            }
            let WsEvent::Ping(payload) = read_event(&mut stream, false, MAX_PAYLOAD).unwrap()
            else {
                panic!("expected a ping");
            };
            assert_eq!(payload, b"xyz");
            write_frame(&mut stream, false, OP_PONG, &payload).unwrap();
        });
        let mut client = connect(&format!("{addr}/store"), None, |_| Ok(())).unwrap();
        for size in [4usize, 200, 70_000] {
            let payload = vec![b'z'; size];
            write_frame(&mut client, true, OP_BINARY, &payload).unwrap();
            let WsEvent::Data(got) = read_event(&mut client, true, MAX_PAYLOAD).unwrap() else {
                panic!("expected echoed data");
            };
            assert_eq!(got, payload);
        }
        write_frame(&mut client, true, OP_PING, b"xyz").unwrap();
        let pong = read_raw(&mut client, true, MAX_PAYLOAD).unwrap();
        assert_eq!(pong.opcode, OP_PONG);
        assert_eq!(pong.payload, b"xyz");
        drop(client);
        server.join().unwrap();
    }

    #[test]
    fn payload_over_16_mib_is_rejected() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            accept(&mut stream).unwrap();
            let err = read_event(&mut stream, false, MAX_PAYLOAD).unwrap_err();
            assert!(err.to_string().contains("16MiB"), "{err}");
        });
        let mut client = connect(&format!("{addr}"), None, |_| Ok(())).unwrap();
        let mut hdr = vec![0x82, 0x80 | 127];
        hdr.extend_from_slice(&u64::try_from(MAX_PAYLOAD + 1).unwrap().to_be_bytes());
        hdr.extend_from_slice(&[1, 2, 3, 4]);
        client.write_all(&hdr).unwrap();
        drop(client);
        server.join().unwrap();
    }

    /// A frame that declares more than the reader's limit is refused from
    /// its header: the reader does not wait for, or allocate, the payload.
    #[test]
    fn a_frame_over_the_callers_limit_is_refused_from_its_header() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        // 1 MiB declared, mask sent, no payload: a reader that allocated and
        // waited would block here until its timeout.
        server
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let mut hdr = vec![0x82, 0x80 | 127];
        hdr.extend_from_slice(&(1u64 << 20).to_be_bytes());
        hdr.extend_from_slice(&[1, 2, 3, 4]);
        client.write_all(&hdr).unwrap();
        let started = std::time::Instant::now();
        let err = read_event(&mut server, false, 4096).unwrap_err();
        assert!(
            err.to_string()
                .contains("over 4096 bytes, the limit before authentication"),
            "{err}"
        );
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
        // At the limit is fine (on a new connection: the refused frame's
        // mask is still in this one).
        let mut client = TcpStream::connect(addr).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let payload = vec![b'x'; 4096];
        write_frame(&mut client, true, OP_BINARY, &payload).unwrap();
        let WsEvent::Data(got) = read_event(&mut server, false, 4096).unwrap() else {
            panic!("expected data");
        };
        assert_eq!(got.len(), 4096);
    }

    #[test]
    fn a_header_read_never_reaches_past_the_end_of_the_headers() {
        assert_eq!(safe_read_len(b""), 4);
        assert_eq!(safe_read_len(b"GET /"), 4);
        assert_eq!(safe_read_len(b"x\r"), 3);
        assert_eq!(safe_read_len(b"x\r\n"), 2);
        assert_eq!(safe_read_len(b"x\r\n\r"), 1);
        assert_eq!(safe_read_len(b"x\r\r"), 3);
        // The bytes after the headers are left for the frame reader.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let mut client = TcpStream::connect(addr).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        client
            .write_all(b"GET / HTTP/1.1\r\nHost: x\r\n\r\nAFTER")
            .unwrap();
        let headers = read_headers(&mut server).unwrap();
        assert!(headers.ends_with("Host: x\r\n\r\n"), "{headers:?}");
        let mut rest = [0u8; 5];
        server.read_exact(&mut rest).unwrap();
        assert_eq!(&rest, b"AFTER");
    }
}
