import hashlib
import os
import argparse
import csv
import importlib.util
import json
import socket
import struct
from pathlib import Path

import torch
import torch.nn as nn

A_IP = "192.168.50.1"
B_IP = "192.168.50.2"
PORT = 45326
MAX_BYTES = 16_000_000
MIB = 1024 ** 2
HEADROOM = 512

ROOT = Path(__file__).resolve().parent
SAVE_DIR = ROOT / "checkpoints"


def send(sock, message):
    raw = json.dumps(message).encode("utf-8")

    if len(raw) > MAX_BYTES:
        raise ValueError("Network message too large")

    sock.sendall(struct.pack("!I", len(raw)) + raw)


def read_exact(sock, count):
    data = bytearray()

    while len(data) < count:
        chunk = sock.recv(count - len(data))
        if not chunk:
            raise ConnectionError("Connection closed")
        data.extend(chunk)

    return bytes(data)


def receive(sock):
    size = struct.unpack("!I", read_exact(sock, 4))[0]

    if not 0 < size <= MAX_BYTES:
        raise ValueError("Invalid message length")

    return json.loads(read_exact(sock, size))


CHUNK_SIZE = 1024 * 1024
MAX_CHECKPOINT_SIZE = 8 * 1024 ** 3


def file_hash(path):
    sha = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(CHUNK_SIZE)
            if not chunk:
                break
            sha.update(chunk)

    return sha.hexdigest()


def send_checkpoint(conn, path):
    path = Path(path)
    size = path.stat().st_size

    if not 0 < size <= MAX_CHECKPOINT_SIZE:
        raise ValueError("Invalid checkpoint file size")

    digest = file_hash(path)

    send(conn, {
        "op": "checkpoint",
        "size": size,
        "sha256": digest
    })

    print(f"Sending checkpoint: {size} bytes")

    with path.open("rb") as file:
        while True:
            chunk = file.read(CHUNK_SIZE)

            if not chunk:
                break

            conn.sendall(chunk)

    print("Checkpoint transmitted")


def receive_checkpoint(conn, destination):
    info = receive(conn)

    if info.get("op") != "checkpoint":
        raise ValueError("Expected checkpoint")

    size = info.get("size")
    expected_hash = info.get("sha256")

    if type(size) is not int or not 0 < size <= MAX_CHECKPOINT_SIZE:
        raise ValueError("Invalid checkpoint size")

    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(c not in "0123456789abcdef" for c in expected_hash)
    ):
        raise ValueError("Invalid checkpoint hash")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp_file = destination.with_name(destination.name + ".part")
    remaining = size
    sha = hashlib.sha256()

    try:
        with temp_file.open("wb") as file:
            while remaining > 0:
                chunk = conn.recv(min(CHUNK_SIZE, remaining))

                if not chunk:
                    raise ConnectionError("Checkpoint transfer interrupted")

                file.write(chunk)
                sha.update(chunk)
                remaining -= len(chunk)

        if sha.hexdigest() != expected_hash:
            raise ValueError("Checkpoint SHA-256 mismatch")

        os.replace(temp_file, destination)

    except Exception:
        temp_file.unlink(missing_ok=True)
        raise

    print(f"Checkpoint received: {destination}")
    print(f"Verified bytes: {size}")

    return size

def gpu_info():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")

    torch.cuda.init()
    free, total = torch.cuda.mem_get_info()

    return {
        "name": torch.cuda.get_device_name(0),
        "free_mib": free / MIB,
        "total_mib": total / MIB
    }


def load_model_blocks(model_path):
    path = Path(model_path).resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        "user_model", str(path)
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.get_model()

    if not isinstance(model, nn.Sequential):
        raise ValueError("Model must be nn.Sequential")

    blocks = []

    for layer in model.children():
        if isinstance(layer, nn.Linear):
            if blocks and blocks[-1]["out"] != layer.in_features:
                raise ValueError("Layer dimensions do not match")

            blocks.append({
                "in": layer.in_features,
                "out": layer.out_features,
                "relu": False
            })

        elif isinstance(layer, nn.ReLU):
            if not blocks or blocks[-1]["relu"]:
                raise ValueError("Invalid ReLU placement")

            blocks[-1]["relu"] = True

        else:
            raise ValueError(
                f"Unsupported layer: {type(layer).__name__}"
            )

    if len(blocks) < 2:
        raise ValueError("At least two Linear layers required")

    if blocks[0]["in"] < 2 or blocks[-1]["out"] != 1:
        raise ValueError(
            "This version requires at least 2 inputs "
            "and exactly 1 output"
        )

    return blocks


def build_stage(blocks):
    layers = []

    for block in blocks:
        layers.append(nn.Linear(block["in"], block["out"]))

        if block["relu"]:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers).cuda()


def stage_mib(blocks):
    parameters = sum(
        b["in"] * b["out"] + b["out"]
        for b in blocks
    )

    return parameters * 16 / MIB


def choose_split(blocks, free_a, free_b):
    budget_a = max(0, free_a - HEADROOM)
    budget_b = max(0, free_b - HEADROOM)

    total = stage_mib(blocks)
    desired_ratio = free_a / (free_a + free_b)

    candidates = []

    for cut in range(1, len(blocks)):
        cost_a = stage_mib(blocks[:cut])
        cost_b = stage_mib(blocks[cut:])

        if cost_a <= budget_a and cost_b <= budget_b:
            score = abs(cost_a / total - desired_ratio)
            candidates.append((score, cut))

    if not candidates:
        raise RuntimeError(
            "Model does not fit estimated GPU budgets"
        )

    return min(candidates)[1]


def load_dataset(path, input_size):
    if path is None:
        torch.manual_seed(42)
        x = torch.randn(256, input_size)
        y = 0.7 * x[:, :1] - 0.3 * x[:, 1:2]

        print("Using demonstration dataset")
        return x, y

    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or "target" not in reader.fieldnames:
            raise ValueError("CSV must contain target column")

        features = [
            col for col in reader.fieldnames
            if col != "target"
        ]

        if len(features) != input_size:
            raise ValueError(
                f"Expected {input_size} input columns; "
                f"found {len(features)}"
            )

        inputs, targets = [], []

        for row in reader:
            inputs.append([float(row[c]) for c in features])
            targets.append([float(row["target"])])

    if not inputs:
        raise ValueError("Dataset is empty")

    return (
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32)
    )


def run_worker():
    info = gpu_info()

    print("Laptop B GPU:", info["name"])

    with socket.socket() as server:
        server.setsockopt(
            socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
        )
        server.bind((B_IP, PORT))
        server.listen(1)

        print("Waiting for Laptop A...")

        conn, address = server.accept()

        with conn:
            if address[0] != A_IP:
                raise ConnectionError("Unknown controller")

            conn.settimeout(180)

            send(conn, {"op": "hello", **info})
            config = receive(conn)

            if config["op"] != "configure":
                raise ValueError("Invalid configuration")

            blocks = config["blocks"]
            lr = float(config["lr"])

            if (
                not isinstance(blocks, list)
                or not 1 <= len(blocks) <= 64
                or any(
                    not isinstance(b, dict)
                    or not isinstance(b.get("in"), int)
                    or not isinstance(b.get("out"), int)
                    or not 1 <= b["in"] <= 8192
                    or not 1 <= b["out"] <= 8192
                    or not isinstance(b.get("relu"), bool)
                    for b in blocks
                )
                or not 0 < lr <= 1
            ):
                raise ValueError("Invalid model configuration")

            free_now = gpu_info()["free_mib"]

            if stage_mib(blocks) > free_now - HEADROOM:
                raise RuntimeError("Worker GPU memory insufficient")

            model = build_stage(blocks)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                foreach=False
            )

            print("Assigned layers:", len(blocks))
            send(conn, {"op": "ready"})

            steps = 0

            while True:
                msg = receive(conn)

                if msg["op"] == "finish":
                    break

                if msg["op"] != "forward":
                    raise ValueError("Expected forward")

                x = torch.tensor(
                    msg["values"],
                    dtype=torch.float32,
                    device="cuda",
                    requires_grad=True
                )

                optimizer.zero_grad(set_to_none=True)
                output = model(x)

                send(conn, {
                    "op": "prediction",
                    "values": output.detach().cpu().tolist()
                })

                msg = receive(conn)

                if msg["op"] != "backward":
                    raise ValueError("Expected backward")

                gradient = torch.tensor(
                    msg["values"],
                    dtype=torch.float32,
                    device="cuda"
                )

                output.backward(gradient)

                input_gradient = x.grad.detach().cpu().tolist()
                optimizer.step()

                send(conn, {
                    "op": "gradient",
                    "values": input_gradient
                })

                steps += 1

            SAVE_DIR.mkdir(exist_ok=True)
            torch.save(
                model.state_dict(),
                SAVE_DIR / "worker_stage.pt"
            )

            send(conn, {"op": "saved"})

    print("Worker training steps:", steps)
    print("Worker weights saved")
    print("Peak VRAM:", round(
        torch.cuda.max_memory_allocated() / MIB, 2
    ), "MiB")


def run_train(args):
    blocks = load_model_blocks(args.model)

    x_data, y_data = load_dataset(
        args.data, blocks[0]["in"]
    )

    info_a = gpu_info()

    with socket.create_connection(
        (B_IP, PORT), timeout=15
    ) as conn:
        conn.settimeout(180)

        info_b = receive(conn)

        if info_b["op"] != "hello":
            raise RuntimeError("Invalid worker handshake")

        cut = choose_split(
            blocks,
            info_a["free_mib"],
            info_b["free_mib"]
        )

        print("\n===== CUSTOM DISTRIBUTED TRAINING =====")
        print("Laptop A:", info_a["name"])
        print("Laptop B:", info_b["name"])
        print("Total Linear layers:", len(blocks))
        print("Laptop A layers:", cut)
        print("Laptop B layers:", len(blocks) - cut)

        send(conn, {
            "op": "configure",
            "blocks": blocks[cut:],
            "lr": args.lr
        })

        if receive(conn)["op"] != "ready":
            raise RuntimeError("Worker not ready")

        model = build_stage(blocks[:cut])
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            foreach=False
        )

        for epoch in range(1, args.epochs + 1):
            order = torch.randperm(len(x_data))
            epoch_loss = 0.0

            for start in range(
                0, len(x_data), args.batch_size
            ):
                ids = order[start:start + args.batch_size]

                x = x_data[ids].cuda()
                target = y_data[ids].cuda()

                optimizer.zero_grad(set_to_none=True)
                features = model(x)

                send(conn, {
                    "op": "forward",
                    "values": features.detach().cpu().tolist()
                })

                response = receive(conn)

                if response["op"] != "prediction":
                    raise RuntimeError("Invalid worker response")

                output = torch.tensor(
                    response["values"],
                    dtype=torch.float32,
                    device="cuda",
                    requires_grad=True
                )

                loss = nn.functional.mse_loss(output, target)
                loss.backward()

                send(conn, {
                    "op": "backward",
                    "values": output.grad.detach().cpu().tolist()
                })

                response = receive(conn)

                if response["op"] != "gradient":
                    raise RuntimeError("Invalid gradient response")

                grad = torch.tensor(
                    response["values"],
                    dtype=torch.float32,
                    device="cuda"
                )

                features.backward(grad)
                optimizer.step()

                epoch_loss += loss.item() * len(ids)

            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"Loss: {epoch_loss / len(x_data):.6f}"
            )

        send(conn, {"op": "finish"})

        if receive(conn)["op"] != "saved":
            raise RuntimeError("Worker checkpoint failed")

    SAVE_DIR.mkdir(exist_ok=True)

    torch.save(
        model.state_dict(),
        SAVE_DIR / "controller_stage.pt"
    )

    (SAVE_DIR / "metadata.json").write_text(
        json.dumps({
            "blocks": blocks,
            "split": cut,
            "epochs": args.epochs,
            "task": "regression"
        }, indent=2),
        encoding="utf-8"
    )

    print("\nDISTRIBUTED TRAINING COMPLETE")
    print("Controller checkpoint saved")
    print("Worker checkpoint saved on Laptop B")
    print("Peak VRAM:", round(
        torch.cuda.max_memory_allocated() / MIB, 2
    ), "MiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("role", choices=["worker", "train"])
    parser.add_argument("--model", default="models/my_model.py")
    parser.add_argument("--data", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.003)

    args = parser.parse_args()

    if not 1 <= args.epochs <= 10000:
        parser.error("epochs must be 1-10000")
    if not 1 <= args.batch_size <= 64:
        parser.error("batch size must be 1-64")
    if not 0 < args.lr <= 1:
        parser.error("learning rate must be between 0 and 1")

    try:
        if args.role == "worker":
            run_worker()
        else:
            run_train(args)
    except (OSError, RuntimeError, ValueError,
            KeyError, ConnectionError) as exc:
        raise SystemExit(f"ERROR: {exc}")
