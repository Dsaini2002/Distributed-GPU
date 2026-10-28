
import json
import socket
import subprocess
import time
import uuid
import ipaddress

from gpu_detector import detect_gpus

PORT = 45321
INTERVAL = 3
DEVICE_ID = uuid.uuid4().hex
HOSTNAME = socket.gethostname()

MAGIC = "DISTRIBUTED_GPU_V1"


def get_network_interfaces():
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "Get-NetIPAddress -AddressFamily IPv4 | "
            "Where-Object { $_.IPAddress -notlike '127.*' "
            "-and $_.PrefixLength -ge 1 "
            "-and $_.PrefixLength -le 30 } | "
            "Select-Object IPAddress,PrefixLength | "
            "ConvertTo-Json -Compress"
        )
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=15,
        check=True
    )

    if not result.stdout.strip():
        return []

    records = json.loads(result.stdout)

    if isinstance(records, dict):
        records = [records]

    interfaces = []

    for item in records:
        address = item["IPAddress"]
        prefix = int(item["PrefixLength"])

        network = ipaddress.IPv4Network(
            f"{address}/{prefix}",
            strict=False
        )

        interfaces.append((
            address,
            str(network.broadcast_address)
        ))

    return list(set(interfaces))


def make_packet():
    return json.dumps({
        "magic": MAGIC,
        "id": DEVICE_ID,
        "computer": HOSTNAME,
        "gpu": detect_gpus()
    }).encode("utf-8")


def send_discovery(interfaces):
    packet = make_packet()

    for local_ip, broadcast_ip in interfaces:
        try:
            with socket.socket(
                socket.AF_INET,
                socket.SOCK_DGRAM
            ) as sender:

                sender.setsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_BROADCAST,
                    1
                )

                sender.bind((local_ip, 0))
                sender.sendto(
                    packet,
                    (broadcast_ip, PORT)
                )

        except OSError as error:
            print(f"Network {local_ip}: {error}")


def show_remote_gpu(data, ip):
    print("\n" + "=" * 45)
    print("REMOTE COMPUTER DISCOVERED")
    print("=" * 45)

    print("Computer:", data["computer"])
    print("LAN IP:", ip)

    gpu_info = data.get("gpu", {})

    for gpu in gpu_info.get("gpus", []):
        print("\nGPU:", gpu["name"])
        print("Total VRAM:", gpu["total_vram_mb"], "MB")
        print("Free VRAM:", gpu["free_vram_mb"], "MB")
        print("GPU Usage:", gpu["utilization_percent"], "%")

    print("=" * 45)


def start_discovery():
    print("\n========== DISTRIBUTED GPU ==========")
    print("Local computer:", HOSTNAME)

    interfaces = get_network_interfaces()

    if not interfaces:
        print("No suitable IPv4 network adapter found.")
        return

    for ip, broadcast in interfaces:
        print("Network:", ip, "Broadcast:", broadcast)

    print("\nSearching for GPU computers...")
    print("Press Ctrl+C to stop.\n")

    local_ips = {ip for ip, _ in interfaces}
    peers = {}
    last_broadcast = 0

    with socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    ) as receiver:

        receiver.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1
        )

        receiver.bind(("0.0.0.0", PORT))
        receiver.settimeout(1)

        while True:
            now = time.monotonic()

            if now - last_broadcast >= INTERVAL:
                send_discovery(interfaces)
                last_broadcast = now

            try:
                raw, address = receiver.recvfrom(8192)
                data = json.loads(raw.decode("utf-8"))

                if not isinstance(data, dict):
                    continue

                if data.get("magic") != MAGIC:
                    continue

                if data.get("id") == DEVICE_ID:
                    continue

                if address[0] in local_ips:
                    continue

                if not isinstance(data.get("gpu"), dict):
                    continue

                peer_id = data.get("id")

                if not isinstance(peer_id, str):
                    continue

                if peer_id not in peers:
                    peers[peer_id] = address[0]
                    show_remote_gpu(data, address[0])

            except socket.timeout:
                pass
            except (UnicodeError, ValueError, KeyError, TypeError):
                pass


if __name__ == "__main__":
    try:
        start_discovery()
    except KeyboardInterrupt:
        print("\nGPU discovery stopped.")
    except (OSError, subprocess.SubprocessError) as error:
        print("ERROR:", error)
