import os
import json
import time
import uuid
import socket
import urllib.request

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / ".dgpu"
STATE_DIR.mkdir(exist_ok=True)

ID_FILE = STATE_DIR / "node_id.txt"

if ID_FILE.exists():
    NODE_ID = ID_FILE.read_text().strip()
else:
    NODE_ID = str(uuid.uuid4())
    ID_FILE.write_text(NODE_ID)

TOKEN = os.environ.get("DGPU_TOKEN")
CONTROLLER = os.environ.get(
    "DGPU_CONTROLLER",
    "http://192.168.50.1:45400"
)

if not TOKEN:
    raise RuntimeError("DGPU_TOKEN required")


def detect_gpus():
    result = []

    if not torch.cuda.is_available():
        return result

    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)

        result.append({
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "free_mib": free // (1024 * 1024),
            "total_mib": total // (1024 * 1024)
        })

    return result


def send_heartbeat():
    payload = {
        "node_id": NODE_ID,
        "hostname": socket.gethostname(),
        "gpus": detect_gpus()
    }

    request = urllib.request.Request(
        CONTROLLER + "/api/heartbeat",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + TOKEN
        },
        method="POST"
    )

    with urllib.request.urlopen(
        request, timeout=5
    ) as response:
        response.read()

    return payload


if __name__ == "__main__":
    print("Distributed-GPU Node Agent")
    print("Computer:", socket.gethostname())
    print("Controller:", CONTROLLER)

    while True:
        try:
            node = send_heartbeat()
            print(
                "Heartbeat OK | GPUs:",
                len(node["gpus"])
            )
        except Exception as error:
            print("Connection error:", error)

        time.sleep(5)
