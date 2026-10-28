
import json
import socket
import struct
import sys

import torch
from torch import nn

A_IP = "192.168.50.1"
B_IP = "192.168.50.2"
PORT = 45324

WIDTH = 4096
STEPS = 3
LIMIT = 300_000
MIB = 1024 ** 2


def receive_exact(conn, size):
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Connection closed")
        data.extend(chunk)
    return bytes(data)


def send(conn, obj):
    raw = json.dumps(obj).encode()
    if len(raw) > LIMIT:
        raise ValueError("Message too large")
    conn.sendall(struct.pack("!I", len(raw)) + raw)


def receive(conn):
    size = struct.unpack("!I", receive_exact(conn, 4))[0]
    if not 0 < size <= LIMIT:
        raise ValueError("Invalid message size")
    return json.loads(receive_exact(conn, size))


def check_memory(blocks):
    torch.cuda.init()
    free, total = torch.cuda.mem_get_info()

    # Float32: weights + gradients + 2 Adam states.
    estimate = blocks * WIDTH * WIDTH * 16 / MIB
    reserve = 512

    print("GPU:", torch.cuda.get_device_name(0))
    print(f"Free VRAM: {free / MIB:.0f} MiB")
    print(f"Estimated training state: {estimate:.0f} MiB")

    if free / MIB < estimate + reserve:
        raise RuntimeError(
            "Not enough free VRAM for this test. "
            "Try fewer blocks."
        )


def make_model(blocks, controller):
    layers = []

    if controller:
        layers += [nn.Linear(8, WIDTH), nn.ReLU()]

    for _ in range(blocks):
        layers += [nn.Linear(WIDTH, WIDTH), nn.ReLU()]

    if not controller:
        layers.append(nn.Linear(WIDTH, 1))

    return nn.Sequential(*layers).cuda()


def print_memory():
    torch.cuda.synchronize()
    print(
        "Peak allocated:",
        round(torch.cuda.max_memory_allocated() / MIB, 2),
        "MiB"
    )
    print(
        "Peak reserved:",
        round(torch.cuda.max_memory_reserved() / MIB, 2),
        "MiB"
    )


def worker(blocks):
    check_memory(blocks)
    model = make_model(blocks, False)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001, foreach=False
    )

    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((B_IP, PORT))
        server.listen(1)

        print("Waiting for Laptop A...")
        conn, address = server.accept()

        with conn:
            if address[0] != A_IP:
                raise ConnectionError("Unknown controller")

            conn.settimeout(180)

            for step in range(STEPS):
                msg = receive(conn)
                if msg.get("op") != "forward":
                    raise ValueError("Unexpected operation")

                x = torch.tensor(
                    msg["features"],
                    dtype=torch.float32,
                    device="cuda",
                    requires_grad=True
                )
                if x.shape != (1, WIDTH):
                    raise ValueError("Invalid input shape")

                optimizer.zero_grad(set_to_none=True)
                y = model(x)

                send(conn, {"output": y.detach().cpu().tolist()})

                msg = receive(conn)
                if msg.get("op") != "backward":
                    raise ValueError("Unexpected operation")

                grad = torch.tensor(
                    msg["gradient"],
                    dtype=torch.float32,
                    device="cuda"
                )
                if grad.shape != y.shape:
                    raise ValueError("Invalid gradient shape")

                y.backward(grad)
                input_grad = x.grad.detach().cpu().tolist()
                optimizer.step()

                send(conn, {"gradient": input_grad})
                print(f"Worker step {step + 1}/{STEPS}")

    print_memory()
    print("WORKER COMPLETED")


def controller(blocks):
    check_memory(blocks)
    model = make_model(blocks, True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001, foreach=False
    )

    x = torch.randn(1, 8, device="cuda")
    target = torch.tensor([[0.4]], device="cuda")

    with socket.create_connection((B_IP, PORT), timeout=15) as conn:
        conn.settimeout(180)
        print("Starting large-model training")

        for step in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            features = model(x)

            send(conn, {
                "op": "forward",
                "features": features.detach().cpu().tolist()
            })

            msg = receive(conn)
            y = torch.tensor(
                msg["output"],
                dtype=torch.float32,
                device="cuda",
                requires_grad=True
            )
            if y.shape != target.shape:
                raise ValueError("Invalid output shape")

            loss = nn.functional.mse_loss(y, target)
            loss.backward()

            send(conn, {
                "op": "backward",
                "gradient": y.grad.detach().cpu().tolist()
            })

            msg = receive(conn)
            grad = torch.tensor(
                msg["gradient"],
                dtype=torch.float32,
                device="cuda"
            )
            if grad.shape != features.shape:
                raise ValueError("Invalid returned gradient")

            features.backward(grad)
            optimizer.step()

            print(
                f"Step {step + 1}/{STEPS} "
                f"Loss: {loss.item():.6f}"
            )

    print_memory()
    print("DISTRIBUTED LARGE-MODEL TRAINING COMPLETE")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(
            "Usage: python large_model_train.py "
            "worker|train BLOCKS"
        )

    role = sys.argv[1]
    blocks = int(sys.argv[2])

    if role not in ("worker", "train") or not 1 <= blocks <= 8:
        sys.exit("Role must be worker/train; blocks must be 1-8")

    if not torch.cuda.is_available():
        sys.exit("CUDA unavailable")

    try:
        if role == "worker":
            worker(blocks)
        else:
            controller(blocks)
    except (OSError, RuntimeError, ValueError, ConnectionError) as e:
        sys.exit(f"ERROR: {e}")
