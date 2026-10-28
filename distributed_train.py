
import json
import socket
import struct
import sys

import torch
import torch.nn as nn

A_IP = "192.168.50.1"
B_IP = "192.168.50.2"
PORT = 45323
STEPS = 60
MAX_MESSAGE = 200_000


def read_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Connection closed")
        data.extend(chunk)
    return bytes(data)


def send(sock, obj):
    raw = json.dumps(obj).encode("utf-8")
    if len(raw) > MAX_MESSAGE:
        raise ValueError("Message too large")
    sock.sendall(struct.pack("!I", len(raw)) + raw)


def receive(sock):
    length = struct.unpack("!I", read_exact(sock, 4))[0]
    if not 0 < length <= MAX_MESSAGE:
        raise ValueError("Invalid message size")
    obj = json.loads(read_exact(sock, length))
    if not isinstance(obj, dict):
        raise ValueError("Invalid message")
    return obj


def run_worker():
    print("Laptop B GPU:", torch.cuda.get_device_name(0))

    model = nn.Linear(16, 1).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((B_IP, PORT))
        server.listen(1)

        print("Waiting for Laptop A...")
        conn, address = server.accept()

        with conn:
            if address[0] != A_IP:
                raise ConnectionError("Unknown controller")

            conn.settimeout(120)
            print("Connected to Laptop A!")

            for step in range(1, STEPS + 1):
                request = receive(conn)

                if request.get("op") != "forward":
                    raise ValueError("Expected forward")

                features = torch.tensor(
                    request["features"],
                    dtype=torch.float32,
                    device="cuda",
                    requires_grad=True
                )

                if features.shape != (32, 16):
                    raise ValueError("Invalid feature shape")

                optimizer.zero_grad(set_to_none=True)
                output = model(features)

                send(conn, {
                    "output": output.detach().cpu().tolist()
                })

                request = receive(conn)

                if request.get("op") != "backward":
                    raise ValueError("Expected backward")

                grad_output = torch.tensor(
                    request["gradient"],
                    dtype=torch.float32,
                    device="cuda"
                )

                if grad_output.shape != output.shape:
                    raise ValueError("Invalid gradient shape")

                output.backward(grad_output)

                grad_input = features.grad.detach().cpu().tolist()
                optimizer.step()

                send(conn, {"gradient": grad_input})

                if step == 1 or step % 10 == 0:
                    print(f"Worker training step {step}/{STEPS}")

    print("Laptop B training complete!")


def run_controller():
    print("Laptop A GPU:", torch.cuda.get_device_name(0))

    torch.manual_seed(7)

    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU()
    ).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    x = torch.randn(32, 8, device="cuda")
    target = torch.relu(
        0.75 * x[:, :1] + 0.25 * x[:, 1:2]
    )

    with socket.create_connection((B_IP, PORT), timeout=15) as conn:
        conn.settimeout(120)
        print("Connected to Laptop B!")
        print("Starting distributed ML training...\n")

        for step in range(1, STEPS + 1):
            optimizer.zero_grad(set_to_none=True)

            features = model(x)

            send(conn, {
                "op": "forward",
                "features": features.detach().cpu().tolist()
            })

            response = receive(conn)

            remote_output = torch.tensor(
                response["output"],
                dtype=torch.float32,
                device="cuda",
                requires_grad=True
            )

            if remote_output.shape != target.shape:
                raise ValueError("Invalid worker output")

            loss = nn.functional.mse_loss(remote_output, target)
            loss.backward()

            send(conn, {
                "op": "backward",
                "gradient": remote_output.grad.detach().cpu().tolist()
            })

            response = receive(conn)

            grad_features = torch.tensor(
                response["gradient"],
                dtype=torch.float32,
                device="cuda"
            )

            if grad_features.shape != features.shape:
                raise ValueError("Invalid worker gradient")

            features.backward(grad_features)
            optimizer.step()

            if step == 1 or step % 10 == 0:
                print(
                    f"Step {step:02d}/{STEPS} "
                    f"| Loss: {loss.item():.6f}"
                )

    print("\nDISTRIBUTED ML TRAINING COMPLETED")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("CUDA GPU unavailable")

    if len(sys.argv) != 2 or sys.argv[1] not in ("worker", "train"):
        sys.exit("Usage: python distributed_train.py worker|train")

    try:
        if sys.argv[1] == "worker":
            run_worker()
        else:
            run_controller()
    except (OSError, RuntimeError, ValueError, ConnectionError) as exc:
        sys.exit(f"ERROR: {exc}")
