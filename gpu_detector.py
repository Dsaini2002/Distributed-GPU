
import csv
import io
import json
import socket
import subprocess
import sys


def to_number(value):
    value = value.strip()
    if value.isdigit():
        return int(value)
    return None


def detect_gpus():
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits"
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=15
        )
    except FileNotFoundError:
        raise RuntimeError(
            "nvidia-smi not found. Check NVIDIA driver installation."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("GPU detection timed out.")
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            error.stderr.strip() or "NVIDIA GPU detection failed."
        )

    gpus = []

    for row in csv.reader(io.StringIO(result.stdout)):
        if len(row) != 7:
            continue

        gpu = {
            "index": to_number(row[0]),
            "name": row[1].strip(),
            "total_vram_mb": to_number(row[2]),
            "used_vram_mb": to_number(row[3]),
            "free_vram_mb": to_number(row[4]),
            "utilization_percent": to_number(row[5]),
            "temperature_c": to_number(row[6])
        }

        gpus.append(gpu)

    return {
        "computer": socket.gethostname(),
        "gpu_count": len(gpus),
        "gpus": gpus
    }


def main():
    try:
        system = detect_gpus()
    except RuntimeError as error:
        print("ERROR:", error)
        sys.exit(1)

    if "--json" in sys.argv:
        print(json.dumps(system, indent=2))
        return

    print("\n========== DISTRIBUTED GPU ==========")
    print("Computer:", system["computer"])
    print("GPU Count:", system["gpu_count"])

    for gpu in system["gpus"]:
        print("\nGPU:", gpu["name"])
        print("Total VRAM:", gpu["total_vram_mb"], "MB")
        print("Used VRAM:", gpu["used_vram_mb"], "MB")
        print("Free VRAM:", gpu["free_vram_mb"], "MB")
        print("GPU Usage:", gpu["utilization_percent"], "%")
        print("Temperature:", gpu["temperature_c"], "C")

    if not system["gpus"]:
        print("No NVIDIA GPUs detected.")


if __name__ == "__main__":
    main()
