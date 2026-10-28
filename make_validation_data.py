
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

TRAIN_FILE = DATA_DIR / "demo_train.csv"
VAL_FILE = DATA_DIR / "demo_val.csv"

rng = random.Random(42)


def generate_rows(count):
    rows = []

    for _ in range(count):
        x = [rng.uniform(-1, 1) for _ in range(8)]

        target = (
            0.7 * x[0]
            - 0.3 * x[1]
            + 0.1 * x[2]
        )

        rows.append(x + [target])

    return rows


def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "f1", "f2", "f3", "f4",
            "f5", "f6", "f7", "f8",
            "target"
        ])

        writer.writerows(rows)


def main():
    DATA_DIR.mkdir(exist_ok=True)

    if TRAIN_FILE.exists() or VAL_FILE.exists():
        raise SystemExit(
            "Demo files already exist. "
            "Move them before regenerating."
        )

    train_rows = generate_rows(300)
    val_rows = generate_rows(100)

    save_csv(TRAIN_FILE, train_rows)
    save_csv(VAL_FILE, val_rows)

    print("TRAINING DATASET:", len(train_rows), "samples")
    print("VALIDATION DATASET:", len(val_rows), "samples")
    print("Dataset generation successful!")


if __name__ == "__main__":
    main()
