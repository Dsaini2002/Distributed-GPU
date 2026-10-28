
import json
import socket
import struct
import sys

import torch

CONTROLLER_IP = "192.168.50.1"
WORKER_IP = "192.168.50.2"
PORT = 45322
MAX_MESSAGE = 1_000_000


def check_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")

    return torch.cuda.get_device_name(0)


def receive_exact(sock, count):
    data = bytearray()

    while len(data) < count:
        chunk = sock.recv(count - len(data))
        if not chunk:
            raise ConnectionError("Connection closed")
        data.extend(chunk)

    return bytes(data)


def send_data(sock, data):
    raw = json.dumps(data).encode("utf-8")
    if len(raw) > MAX_MESSAGE:
        raise ValueError("Message too large")

    sock.sendall(struct.pack("!I", len(raw)) + raw)


def receive_data(sock):
    header = receive_exact(sock, 4)
    size = struct.unpack("!I", header)[0]

    if size > MAX_MESSAGE:
        raise ValueError("Message too large")

    return json.loads(receive_exact(sock, size))


def run_worker():
    gpu_name = check_gpu()
    print("Worker GPU:", gpu_name)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((WORKER_IP, PORT))
        server.listen(1)

        print("Waiting for Laptop A...")

        conn, address = server.accept()

        with conn:
            if address[0] != CONTROLLER_IP:
                print("Rejected unknown computer:", address[0])
                return

            request = receive_data(conn)
            if request.get("operation") != "matmul":
                raise ValueError("Unsupported operation")

            matrix = request["matrix"]
            if (
                not isinstance(matrix, list)
                or len(matrix) != 64
                or any(
                    not isinstance(row, list) or len(row) != 64
                    for row in matrix
                )
            ):
                raise ValueError("Invalid matrix dimensions")

            tensor = torch.tensor(
                matrix,
                dtype=torch.float32,
                device="cuda"
            )

            result = torch.matmul(tensor, tensor)
            torch.cuda.synchronize()

            print("Computed matrix on:", gpu_name)

            send_data(conn, {
                "gpu": gpu_name,
                "matrix": result.cpu().tolist()
            })

    print("Worker computation completed")


def run_client():
    gpu_name = check_gpu()
    print("Controller GPU:", gpu_name)

    matrix = torch.rand(
        64, 64,
        dtype=torch.float32,
        device="cuda"
    )

    local_result = torch.matmul(matrix, matrix)
    torch.cuda.synchronize()

    print("Stage 1 completed on:", gpu_name)

    with socket.create_connection((WORKER_IP, PORT), timeout=15) as conn:
        conn.settimeout(30)

        send_data(conn, {
            "operation": "matmul",
            "matrix": local_result.cpu().tolist()
        })

        response = receive_data(conn)

    print("Stage 2 completed on:", response["gpu"])

    remote_result = torch.tensor(
        response["matrix"],
        dtype=torch.float32,
        device="cuda"
    )

    final_result = torch.relu(remote_result)
    torch.cuda.synchronize()

    print("Stage 3 completed on:", gpu_name)
    print("Final tensor shape:", tuple(final_result.shape))
    print("Final tensor mean:", final_result.mean().item())
    print("TWO-GPU COMPUTATION SUCCESSFUL")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("worker", "client"):
        print("Usage: python gpu_communication.py worker|client")
        sys.exit(1)

    try:
        if sys.argv[1] == "worker":
            run_worker()
        else:
            run_client()
    except (OSError, RuntimeError, ValueError, ConnectionError) as error:
        print("ERROR:", error)
        sys.exit(1)
