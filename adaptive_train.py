
import argparse
import json
import math
import socket
import struct
import sys

import torch
import torch.nn as nn

A_IP = "192.168.50.1"
B_IP = "192.168.50.2"
PORT = 45325

WIDTH = 4096
STEPS = 3
MIB = 1024 ** 2
MAX_MESSAGE = 400_000

# FP32: weights + gradients + 2 Adam states
BLOCK_MIB = WIDTH * WIDTH * 16 / MIB

# Conservative space for CUDA, activations, and overhead
HEADROOM_MIB = 768


def gpu_info():
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA GPU unavailable")

    torch.cuda.set_device(0)
    torch.cuda.init()

    free, total = torch.cuda.mem_get_info()
    free_mib = free / MIB

    capacity = max(
        0,
        math.floor(
            (free_mib - HEADROOM_MIB) / BLOCK_MIB
        )
    )

    return {
        "name": torch.cuda.get_device_name(0),
        "free_mib": round(free_mib, 2),
        "total_mib": round(total / MIB, 2),
        "capacity": capacity
    }


def choose_partition(total_blocks, capacity_a, capacity_b):
    if total_blocks < 2:
        raise ValueError("At least 2 total blocks required")

    if capacity_a < 1 or capacity_b < 1:
        raise RuntimeError("Both GPUs need free memory")

    if total_blocks > capacity_a + capacity_b:
        raise RuntimeError(
            f"Requested {total_blocks} blocks but estimated "
            f"safe capacity is only {capacity_a + capacity_b}"
        )

    minimum_a = max(1, total_blocks - capacity_b)
    maximum_a = min(capacity_a, total_blocks - 1)

    if minimum_a > maximum_a:
        raise RuntimeError("No valid two-GPU split available")

    suggested_a = round(
        total_blocks * capacity_a /
        (capacity_a + capacity_b)
    )

    blocks_a = max(
        minimum_a,
        min(suggested_a, maximum_a)
    )

    blocks_b = total_blocks - blocks_a

    return blocks_a, blocks_b


def read_exact(conn, count):
    data = bytearray()

    while len(data) < count:
        chunk = conn.recv(count - len(data))
        if not chunk:
            raise ConnectionError("Peer disconnected")
        data.extend(chunk)

    return bytes(data)


def send(conn, obj):
    raw = json.dumps(obj).encode("utf-8")

    if len(raw) > MAX_MESSAGE:
        raise ValueError("Message too large")

    conn.sendall(
        struct.pack("!I", len(raw)) + raw
    )


def receive(conn):
    header = read_exact(conn, 4)
    size = struct.unpack("!I", header)[0]

    if not 0 < size <= MAX_MESSAGE:
        raise ValueError("Invalid message size")

    message = json.loads(read_exact(conn, size))

    if not isinstance(message, dict):
        raise ValueError("Invalid message")

    return message


def build_model(blocks, controller):
    layers = []

    if controller:
        layers.extend([
            nn.Linear(8, WIDTH),
            nn.ReLU()
        ])

    for _ in range(blocks):
        layers.extend([
            nn.Linear(WIDTH, WIDTH),
            nn.ReLU()
        ])

    if not controller:
        layers.append(nn.Linear(WIDTH, 1))

    return nn.Sequential(*layers).cuda()


def show_memory(label):
    torch.cuda.synchronize()

    print(f"\n--- {label} ---")
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


def run_worker():
    info = gpu_info()

    print("\n===== DISTRIBUTED GPU WORKER =====")
    print("GPU:", info["name"])
    print("Free VRAM:", info["free_mib"], "MiB")
    print("Estimated capacity:", info["capacity"], "blocks")

    with socket.socket() as server:
        server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1
        )

        server.bind((B_IP, PORT))
        server.listen(1)

        print("\nWaiting for Laptop A...")

        conn, address = server.accept()

        with conn:
            if address[0] != A_IP:
                raise ConnectionError("Unknown controller")

            conn.settimeout(180)

            send(conn, {
                "op": "hello",
                **info
            })

            config = receive(conn)

            if config.get("op") != "configure":
                raise ValueError("Expected configuration")

            blocks = int(config["blocks"])

            # Check available memory again before allocation
            current_capacity = gpu_info()["capacity"]

            if not 1 <= blocks <= current_capacity:
                send(conn, {
                    "op": "error",
                    "message": "Insufficient worker GPU memory"
                })
                return

            print("\nAssigned blocks:", blocks)

            torch.manual_seed(7)
            torch.cuda.reset_peak_memory_stats()

            model = build_model(blocks, False)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=0.0001,
                foreach=False
            )

            send(conn, {
                "op": "ready",
                "blocks": blocks
            })

            for step in range(1, STEPS + 1):
                msg = receive(conn)

                if msg.get("op") != "forward":
                    raise ValueError("Expected forward request")

                x = torch.tensor(
                    msg["features"],
                    dtype=torch.float32,
                    device="cuda",
                    requires_grad=True
                )

                if x.shape != (1, WIDTH):
                    raise ValueError("Invalid feature shape")

                optimizer.zero_grad(set_to_none=True)

                output = model(x)

                send(conn, {
                    "op": "output",
                    "values": output.detach().cpu().tolist()
                })

                msg = receive(conn)

                if msg.get("op") != "backward":
                    raise ValueError("Expected backward request")

                grad = torch.tensor(
                    msg["gradient"],
                    dtype=torch.float32,
                    device="cuda"
                )

                if grad.shape != output.shape:
                    raise ValueError("Invalid output gradient")

                output.backward(grad)

                input_grad = x.grad.detach().cpu().tolist()

                optimizer.step()

                send(conn, {
                    "op": "input_gradient",
                    "values": input_grad
                })

                print(f"Worker step {step}/{STEPS}")

    show_memory("Laptop B GPU Memory")
    print("WORKER TRAINING COMPLETE")


def run_controller(total_blocks):
    info_a = gpu_info()

    print("\n===== ADAPTIVE DISTRIBUTED GPU =====")
    print("Laptop A:", info_a["name"])
    print("Free VRAM:", info_a["free_mib"], "MiB")

    with socket.create_connection(
        (B_IP, PORT), timeout=15
    ) as conn:

        conn.settimeout(180)

        info_b = receive(conn)

        if info_b.get("op") != "hello":
            raise ValueError("Invalid worker handshake")

        print("\nLaptop B:", info_b["name"])
        print("Free VRAM:", info_b["free_mib"], "MiB")

        blocks_a, blocks_b = choose_partition(
            total_blocks,
            info_a["capacity"],
            info_b["capacity"]
        )

        print("\n===== AUTOMATIC GPU PARTITION =====")
        print("Total model blocks:", total_blocks)
        print("Laptop A blocks:", blocks_a)
        print("Laptop B blocks:", blocks_b)
        print("Estimated capacity A:", info_a["capacity"])
        print("Estimated capacity B:", info_b["capacity"])

        send(conn, {
            "op": "configure",
            "blocks": blocks_b
        })

        response = receive(conn)

        if response.get("op") != "ready":
            raise RuntimeError(
                response.get("message", "Worker not ready")
            )

        torch.manual_seed(7)
        torch.cuda.reset_peak_memory_stats()

        model = build_model(blocks_a, True)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.0001,
            foreach=False
        )

        x = torch.randn(1, 8, device="cuda")
        target = torch.tensor(
            [[0.4]],
            device="cuda"
        )

        print("\nStarting distributed training...")

        for step in range(1, STEPS + 1):
            optimizer.zero_grad(set_to_none=True)

            features = model(x)

            send(conn, {
                "op": "forward",
                "features": features.detach().cpu().tolist()
            })

            response = receive(conn)

            if response.get("op") != "output":
                raise ValueError("Expected worker output")

            y = torch.tensor(
                response["values"],
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

            response = receive(conn)

            if response.get("op") != "input_gradient":
                raise ValueError("Expected input gradient")

            grad = torch.tensor(
                response["values"],
                dtype=torch.float32,
                device="cuda"
            )

            if grad.shape != features.shape:
                raise ValueError("Invalid feature gradient")

            features.backward(grad)

            optimizer.step()

            print(
                f"Step {step}/{STEPS} "
                f"| Loss: {loss.item():.6f}"
            )

    show_memory("Laptop A GPU Memory")

    print("\nADAPTIVE DISTRIBUTED TRAINING COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "role",
        choices=["worker", "train"]
    )

    parser.add_argument(
        "--blocks",
        type=int,
        default=16
    )

    args = parser.parse_args()

    if not 2 <= args.blocks <= 64:
        sys.exit("Total blocks must be between 2 and 64")

    try:
        if args.role == "worker":
            run_worker()
        else:
            run_controller(args.blocks)

    except (OSError, RuntimeError, ValueError,
            ConnectionError) as error:
        sys.exit(f"ERROR: {error}")
