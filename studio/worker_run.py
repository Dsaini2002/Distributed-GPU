import os
import time
import threading

import uvicorn
from .agent import send_heartbeat


def heartbeat_loop():
    while True:
        try:
            node = send_heartbeat()
            print(
                "Connected:",
                node["hostname"],
                "| GPUs:", len(node["gpus"])
            )
        except Exception as error:
            print("Heartbeat error:", error)

        time.sleep(5)


def main():
    thread = threading.Thread(
        target=heartbeat_loop,
        daemon=True
    )
    thread.start()

    bind_ip = os.environ.get(
        "DGPU_WORKER_BIND_IP", "127.0.0.1"
    )

    print("GPU training service:", bind_ip, 45410)

    uvicorn.run(
        "studio.worker_api:app",
        host=bind_ip,
        port=45410,
        log_level="warning"
    )


if __name__ == "__main__":
    main()
