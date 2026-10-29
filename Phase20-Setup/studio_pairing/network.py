"""LAN IP validation and local broadcast discovery. HTTP remains LAN-only."""
import ipaddress
import json
import socket
import threading

DISCOVERY_PORT = 45431
DISCOVERY_REQUEST = b'DGPU_DISCOVER_V1'


def private_ipv4(value: str) -> str:
    try:
        addr = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ValueError('Enter an IPv4 address, not a hostname or URL') from exc
    if addr.version != 4 or not (addr.is_private or addr.is_loopback) or addr.is_unspecified or addr.is_reserved or addr.is_link_local or addr.is_multicast:
        raise ValueError('Dedicated private LAN IPv4 required')
    return str(addr)


def controller_ip_for(peer: str) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((peer, 45400))
        return private_ipv4(s.getsockname()[0])


def discover(timeout=1.5):
    results = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.bind(('0.0.0.0', 0))
        s.settimeout(timeout)
        import time
        end = time.monotonic() + timeout
        for ip in ['255.255.255.255', '192.168.50.255']:
            try: s.sendto(DISCOVERY_REQUEST, (ip, DISCOVERY_PORT))
            except OSError: pass
        while time.monotonic() < end:
            try:
                data, source = s.recvfrom(512)
                info = json.loads(data.decode('utf-8'))
                ip = private_ipv4(source[0])
                if info.get('service') == 'dgpu-phase20-host':
                    results[ip] = {'ip': ip, 'name': str(info.get('name','Host'))[:64]}
            except socket.timeout: break
            except (ValueError, OSError, UnicodeError): continue
    return list(results.values())[:10]


def run_discovery(stop_event: threading.Event):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', DISCOVERY_PORT))
            s.settimeout(2)
            while not stop_event.is_set():
                try: data, peer = s.recvfrom(128)
                except socket.timeout: continue
                if data != DISCOVERY_REQUEST: continue
                try:
                    private_ipv4(peer[0]); local = controller_ip_for(peer[0]);
                    info = json.dumps({'service':'dgpu-phase20-host','name': socket.gethostname(), 'ip':local}).encode('utf-8')
                    s.sendto(info, peer)
                except (OSError, ValueError): pass
    except OSError:
        # UDP discovery is optional; manual host IP still works.
        pass
