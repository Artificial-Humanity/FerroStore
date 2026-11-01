//! Where `bind_tcp` may listen without TLS (spec §6.3).
//!
//! * A loopback address: always.
//! * The unspecified address (`0.0.0.0`, `::`): never. It listens on every
//!   interface, the ones that are not loopback included.
//! * Any other address: only when the server's `allow_plaintext_on` names
//!   an interface and the address is one of that interface's own addresses
//!   (`getifaddrs`). An IPv6 link-local address also needs the interface's
//!   scope id.
//!
//! With TLS every address is allowed. Authentication is required on every
//! TCP bind, an allowed plaintext interface included.
//!
//! **Pinned to the interface.** A plaintext bind on an address of a named
//! interface (`lo` named and `127.0.0.1` included) also sets
//! `SO_BINDTODEVICE` to that interface on the listening socket, before it
//! binds ([`listen`]). Linux delivers a packet to a local address whatever
//! interface it arrived on (the weak host model): without the option, a
//! listener on `100.64.0.7` (`tailscale0`) also answers a peer on `eth0`
//! that routes `100.64.0.7` to this host. With it, only connections that
//! arrive on `tailscale0` reach the listener. Setting the option needs no
//! privilege since Linux 5.7; when it fails the bind is refused, never
//! made without it.
//!
//! **Off Linux** there is no `SO_BINDTODEVICE`: the address rule above still
//! applies, but the listener is not pinned to the interface, so a peer that
//! routes the address to this host through another interface reaches it.
//! Use TLS there.

use super::layout::backend;
use fl_core::StoreError;
use std::ffi::CStr;
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener};

/// One address of one network interface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Interface {
    pub(super) name: String,
    pub(super) ip: IpAddr,
    /// The IPv6 scope id (the interface index of a link-local address);
    /// 0 for IPv4.
    pub(super) scope_id: u32,
}

/// `ip` as IPv4 when it is an IPv4-mapped IPv6 address.
fn canonical(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V6(v6) => v6.to_ipv4_mapped().map_or(ip, IpAddr::V4),
        IpAddr::V4(_) => ip,
    }
}

/// `true` for a loopback address, an IPv4-mapped one included. This is the
/// address itself; no test switch changes it.
pub(super) fn ip_is_loopback(ip: IpAddr) -> bool {
    match canonical(ip) {
        IpAddr::V4(v4) => v4.is_loopback(),
        IpAddr::V6(v6) => v6.is_loopback(),
    }
}

fn is_link_local_v6(ip: &Ipv6Addr) -> bool {
    (ip.segments()[0] & 0xffc0) == 0xfe80
}

/// `Ok` when `addr` may be bound without TLS, given the interface names in
/// `allow_on` and the host's `interfaces`: `Some(name)` when `addr` is an
/// address of the named interface `name`, which the listener is then pinned
/// to ([`listen`]); `None` for a loopback address no named interface holds.
pub(super) fn check_plaintext_bind(
    addr: SocketAddr,
    allow_on: &[String],
    interfaces: &[Interface],
) -> Result<Option<String>, StoreError> {
    let ip = canonical(addr.ip());
    if ip.is_unspecified() {
        return Err(backend(format!(
            "bind_tcp refused {addr} without TLS: an unspecified address listens on every \
             interface, the ones that are not loopback included; configure TLS \
             (ServerAuth::with_tls_files), or bind a loopback address or one address of an \
             interface named in allow_plaintext_on"
        )));
    }
    let scope = match addr {
        SocketAddr::V6(v6) if is_link_local_v6(v6.ip()) => Some(v6.scope_id()),
        _ => None,
    };
    let on_allowed = interfaces.iter().find(|iface| {
        allow_on.contains(&iface.name)
            && canonical(iface.ip) == ip
            && scope.is_none_or(|scope| scope == iface.scope_id)
    });
    if let Some(iface) = on_allowed {
        return Ok(Some(iface.name.clone()));
    }
    if ip_is_loopback(ip) {
        return Ok(None);
    }
    let named = if allow_on.is_empty() {
        "no interface is named in allow_plaintext_on".to_string()
    } else {
        format!(
            "it is not an address of {} (allow_plaintext_on)",
            allow_on.join(", ")
        )
    };
    Err(backend(format!(
        "bind_tcp refused {addr} without TLS: it is not a loopback address and {named}; \
         configure TLS (ServerAuth::with_tls_files), or name its interface in \
         ServerAuth::allow_plaintext_on"
    )))
}

/// A listening TCP socket on `addr`. With `device`, the socket is created
/// and `SO_BINDTODEVICE` set to that interface before it binds, so it
/// accepts only connections that arrive on it (see the module docs); a
/// failure to set it is an error, and nothing is bound. Without `device`,
/// `TcpListener::bind`.
///
/// Off Linux `device` is not applied (the module docs say what that leaves
/// open).
pub(super) fn listen(addr: SocketAddr, device: Option<&str>) -> io::Result<TcpListener> {
    match device {
        None => TcpListener::bind(addr),
        #[cfg(target_os = "linux")]
        Some(device) => listen_on_device(addr, device),
        #[cfg(not(target_os = "linux"))]
        Some(_) => TcpListener::bind(addr),
    }
}

/// The backlog of a pinned listener: the one `std` gives `TcpListener`.
#[cfg(target_os = "linux")]
const BACKLOG: libc::c_int = 128;

/// `socket`, `SO_REUSEADDR` (as `std` sets it), `SO_BINDTODEVICE`, `bind`,
/// `listen`: `TcpListener::bind` with the device set before the bind.
#[cfg(target_os = "linux")]
fn listen_on_device(addr: SocketAddr, device: &str) -> io::Result<TcpListener> {
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    let name = std::ffi::CString::new(device).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "an interface name holds no NUL byte",
        )
    })?;
    if name.as_bytes().len() >= libc::IFNAMSIZ {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "interface name `{device}` is over {} bytes",
                libc::IFNAMSIZ - 1
            ),
        ));
    }
    let family = match addr {
        SocketAddr::V4(_) => libc::AF_INET,
        SocketAddr::V6(_) => libc::AF_INET6,
    };
    // SAFETY: `socket` takes no pointer; a non-negative return is a new
    // descriptor that `OwnedFd` then owns (and closes on every error path).
    let fd = unsafe { libc::socket(family, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `fd` is a valid descriptor that nothing else owns.
    let socket = unsafe { OwnedFd::from_raw_fd(fd) };
    let one: libc::c_int = 1;
    set_option(
        &socket,
        libc::SO_REUSEADDR,
        (&raw const one).cast(),
        std::mem::size_of::<libc::c_int>(),
    )?;
    set_option(
        &socket,
        libc::SO_BINDTODEVICE,
        name.as_ptr().cast(),
        name.as_bytes_with_nul().len(),
    )
    .map_err(|err| {
        io::Error::new(
            err.kind(),
            format!(
                "set SO_BINDTODEVICE to `{device}` (pin the plaintext listener to its \
                 interface; Linux 5.7 or later): {err}"
            ),
        )
    })?;
    let (storage, len) = sockaddr_of(addr);
    // SAFETY: `storage` holds a `sockaddr_in` or `sockaddr_in6` of `len`
    // bytes, for the socket's own family.
    if unsafe { libc::bind(socket.as_raw_fd(), (&raw const storage).cast(), len) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `listen` takes no pointer.
    if unsafe { libc::listen(socket.as_raw_fd(), BACKLOG) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(TcpListener::from(socket))
}

/// `setsockopt(SOL_SOCKET, option)` with `len` bytes at `value`.
#[cfg(target_os = "linux")]
fn set_option(
    socket: &std::os::fd::OwnedFd,
    option: libc::c_int,
    value: *const libc::c_void,
    len: usize,
) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    let len = libc::socklen_t::try_from(len)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "socket option too long"))?;
    // SAFETY: `value` points to `len` readable bytes (the callers pass a
    // local `c_int` or a live `CString`).
    if unsafe { libc::setsockopt(socket.as_raw_fd(), libc::SOL_SOCKET, option, value, len) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// `addr` as a `sockaddr_storage` and its length.
#[cfg(target_os = "linux")]
fn sockaddr_of(addr: SocketAddr) -> (libc::sockaddr_storage, libc::socklen_t) {
    // SAFETY: an all-zero `sockaddr_storage` is a valid value.
    let mut storage: libc::sockaddr_storage = unsafe { std::mem::zeroed() };
    let len = match addr {
        SocketAddr::V4(v4) => {
            // SAFETY: `sockaddr_storage` is large and aligned enough for any
            // socket address, a `sockaddr_in` included.
            let sin = unsafe { &mut *(&raw mut storage).cast::<libc::sockaddr_in>() };
            sin.sin_family = libc::AF_INET as libc::sa_family_t;
            sin.sin_port = v4.port().to_be();
            sin.sin_addr = libc::in_addr {
                s_addr: u32::from_ne_bytes(v4.ip().octets()),
            };
            std::mem::size_of::<libc::sockaddr_in>()
        }
        SocketAddr::V6(v6) => {
            // SAFETY: as above, for a `sockaddr_in6`.
            let sin6 = unsafe { &mut *(&raw mut storage).cast::<libc::sockaddr_in6>() };
            sin6.sin6_family = libc::AF_INET6 as libc::sa_family_t;
            sin6.sin6_port = v6.port().to_be();
            sin6.sin6_flowinfo = v6.flowinfo();
            sin6.sin6_addr = libc::in6_addr {
                s6_addr: v6.ip().octets(),
            };
            sin6.sin6_scope_id = v6.scope_id();
            std::mem::size_of::<libc::sockaddr_in6>()
        }
    };
    // Both sizes are a few dozen bytes.
    (storage, len as libc::socklen_t)
}

/// The interface `SO_BINDTODEVICE` names on the socket `fd`, `None` when it
/// names none.
#[cfg(all(test, target_os = "linux"))]
pub(super) fn bound_device(fd: std::os::fd::BorrowedFd<'_>) -> io::Result<Option<String>> {
    use std::os::fd::AsRawFd;
    let mut buf = [0u8; libc::IFNAMSIZ];
    let mut len = libc::socklen_t::try_from(buf.len()).unwrap_or(0);
    // SAFETY: `buf` is writable for `len` bytes and `len` is a valid
    // in-out length.
    let rc = unsafe {
        libc::getsockopt(
            fd.as_raw_fd(),
            libc::SOL_SOCKET,
            libc::SO_BINDTODEVICE,
            buf.as_mut_ptr().cast(),
            &raw mut len,
        )
    };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    let name = &buf[..usize::try_from(len).unwrap_or(0).min(buf.len())];
    let name = name.split(|b| *b == 0).next().unwrap_or(&[]);
    Ok((!name.is_empty()).then(|| String::from_utf8_lossy(name).into_owned()))
}

/// Every IPv4 and IPv6 address of every interface of this host.
pub(super) fn interfaces() -> io::Result<Vec<Interface>> {
    let mut head: *mut libc::ifaddrs = std::ptr::null_mut();
    // SAFETY: `head` is a valid out-pointer; on success the list it points
    // to is freed below with `freeifaddrs`, once.
    if unsafe { libc::getifaddrs(&raw mut head) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let mut out = Vec::new();
    let mut cursor = head;
    while !cursor.is_null() {
        // SAFETY: `cursor` is a node of the list `getifaddrs` returned, which
        // stays valid until `freeifaddrs`.
        let entry = unsafe { &*cursor };
        cursor = entry.ifa_next;
        if entry.ifa_addr.is_null() || entry.ifa_name.is_null() {
            continue;
        }
        // SAFETY: `ifa_name` is a NUL-terminated string of the list.
        let name = unsafe { CStr::from_ptr(entry.ifa_name) }
            .to_string_lossy()
            .into_owned();
        // SAFETY: `ifa_addr` is non-null and points to a `sockaddr` whose
        // family says which larger struct it is.
        let family = i32::from(unsafe { (*entry.ifa_addr).sa_family });
        if family == libc::AF_INET {
            // SAFETY: an `AF_INET` address is a `sockaddr_in`.
            let sin = unsafe { &*entry.ifa_addr.cast::<libc::sockaddr_in>() };
            out.push(Interface {
                name,
                ip: IpAddr::V4(Ipv4Addr::from(sin.sin_addr.s_addr.to_ne_bytes())),
                scope_id: 0,
            });
        } else if family == libc::AF_INET6 {
            // SAFETY: an `AF_INET6` address is a `sockaddr_in6`.
            let sin6 = unsafe { &*entry.ifa_addr.cast::<libc::sockaddr_in6>() };
            out.push(Interface {
                name,
                ip: IpAddr::V6(Ipv6Addr::from(sin6.sin6_addr.s6_addr)),
                scope_id: sin6.sin6_scope_id,
            });
        }
    }
    // SAFETY: `head` came from a successful `getifaddrs` and is freed once.
    unsafe { libc::freeifaddrs(head) };
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iface(name: &str, ip: &str, scope_id: u32) -> Interface {
        Interface {
            name: name.to_string(),
            ip: ip.parse().unwrap(),
            scope_id,
        }
    }

    fn allowed(names: &[&str]) -> Vec<String> {
        names.iter().map(ToString::to_string).collect()
    }

    #[test]
    fn the_matcher_needs_a_named_interface_that_holds_the_address() {
        let host = vec![
            iface("lo", "127.0.0.1", 0),
            iface("lo", "::1", 0),
            iface("eth0", "192.168.1.5", 0),
            iface("tailscale0", "100.64.0.7", 0),
            iface("tailscale0", "fd7a:115c:a1e0::7", 0),
            iface("tailscale0", "fe80::1", 4),
            iface("eth0", "fe80::1", 2),
        ];
        let at = |text: &str| -> SocketAddr { text.parse().unwrap() };
        let ok = |addr: &str, names: &[&str]| {
            check_plaintext_bind(at(addr), &allowed(names), &host).is_ok()
        };
        // Loopback: always, named or not.
        for addr in [
            "127.0.0.1:1",
            "127.8.8.8:1",
            "[::1]:1",
            "[::ffff:127.0.0.1]:1",
        ] {
            assert!(ok(addr, &[]), "{addr}");
        }
        // Unspecified: never, whatever is named.
        for addr in ["0.0.0.0:1", "[::]:1", "[::ffff:0.0.0.0]:1"] {
            let err = check_plaintext_bind(at(addr), &allowed(&["lo", "eth0"]), &host)
                .unwrap_err()
                .to_string();
            assert!(err.contains("every interface"), "{addr}: {err}");
        }
        // Not named: refused.
        assert!(!ok("100.64.0.7:1", &[]));
        assert!(!ok("192.168.1.5:1", &["tailscale0"]));
        // Named and on that interface.
        assert!(ok("100.64.0.7:1", &["tailscale0"]));
        assert!(ok("[fd7a:115c:a1e0::7]:1", &["tailscale0"]));
        assert!(ok("[::ffff:100.64.0.7]:1", &["tailscale0"]));
        assert!(ok("192.168.1.5:1", &["eth0", "tailscale0"]));
        // Named, but the address is another interface's, or nobody's.
        assert!(!ok("100.64.0.7:1", &["eth0"]));
        assert!(!ok("10.9.9.9:1", &["eth0", "tailscale0"]));
        // `lo` named does not cover a non-loopback address.
        assert!(!ok("192.168.1.5:1", &["lo"]));
        // A link-local address also needs that interface's scope.
        assert!(ok("[fe80::1%4]:1", &["tailscale0"]));
        assert!(!ok("[fe80::1%2]:1", &["tailscale0"]));
        assert!(ok("[fe80::1%2]:1", &["eth0"]));
        let err = check_plaintext_bind(at("100.64.0.7:1"), &allowed(&["eth0"]), &host)
            .unwrap_err()
            .to_string();
        assert!(err.contains("allow_plaintext_on"), "{err}");
        assert!(err.contains("eth0"), "{err}");
        // The interface the listener is pinned to: the named one that holds
        // the address; none for a loopback address no named interface holds.
        let device = |addr: &str, names: &[&str]| {
            check_plaintext_bind(at(addr), &allowed(names), &host).unwrap()
        };
        assert_eq!(
            device("100.64.0.7:1", &["eth0", "tailscale0"]).as_deref(),
            Some("tailscale0")
        );
        assert_eq!(
            device("[fe80::1%2]:1", &["eth0", "tailscale0"]).as_deref(),
            Some("eth0")
        );
        assert_eq!(device("127.0.0.1:1", &["lo"]).as_deref(), Some("lo"));
        assert_eq!(
            device("[::ffff:127.0.0.1]:1", &["lo"]).as_deref(),
            Some("lo")
        );
        assert_eq!(device("127.0.0.1:1", &[]), None);
        assert_eq!(device("127.0.0.1:1", &["tailscale0"]), None);
        assert_eq!(device("127.8.8.8:1", &["lo"]), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_pinned_listener_carries_so_bindtodevice_and_accepts() {
        use std::io::{Read, Write};
        use std::os::fd::AsFd;
        let listener = listen("127.0.0.1:0".parse().unwrap(), Some("lo")).unwrap();
        assert_eq!(
            bound_device(listener.as_fd()).unwrap().as_deref(),
            Some("lo")
        );
        let addr = listener.local_addr().unwrap();
        assert!(addr.ip().is_loopback() && addr.port() != 0, "{addr}");
        let mut client = std::net::TcpStream::connect(addr).unwrap();
        let (mut served, _) = listener.accept().unwrap();
        client.write_all(b"ping").unwrap();
        let mut got = [0u8; 4];
        served.read_exact(&mut got).unwrap();
        assert_eq!(&got, b"ping");
        // Unpinned: no device.
        let plain = listen("127.0.0.1:0".parse().unwrap(), None).unwrap();
        assert_eq!(bound_device(plain.as_fd()).unwrap(), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_device_that_cannot_be_set_refuses_the_bind() {
        let err = listen("127.0.0.1:0".parse().unwrap(), Some("nosuchif0")).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("SO_BINDTODEVICE"), "{text}");
        assert!(text.contains("nosuchif0"), "{text}");
        let err = listen(
            "127.0.0.1:0".parse().unwrap(),
            Some("an-interface-name-too-long"),
        )
        .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput, "{err}");
    }

    #[test]
    fn the_host_lists_its_loopback_interface() {
        let host = interfaces().unwrap();
        assert!(
            host.iter()
                .any(|iface| iface.ip == IpAddr::V4(Ipv4Addr::LOCALHOST)),
            "{host:?}"
        );
    }
}
