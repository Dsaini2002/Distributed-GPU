
import argparse
import csv
import json
import math

from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent


def load_model(path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    blocks = checkpoint["blocks"]
    layers = []

    for block in blocks:
        layers.append(
            nn.Linear(block["in"], block["out"])
        )

        if block["relu"]:
            layers.append(nn.ReLU())

    model = nn.Sequential(*layers)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()
    return model, blocks


def load_csv(path, expected_features):
    features = []
    targets = []

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("CSV header missing")

        if reader.fieldnames.count("target") != 1:
            raise ValueError("CSV needs one target column")

        feature_names = [
            name for name in reader.fieldnames
            if name != "target"
        ]

        if len(feature_names) != expected_features:
            raise ValueError(
                f"Expected {expected_features} features, "
                f"found {len(feature_names)}"
            )

        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Duplicate CSV columns")

        for row in reader:
            x = [float(row[name]) for name in feature_names]
            y = float(row["target"])

            if not all(math.isfinite(v) for v in x + [y]):
                raise ValueError("Non-finite CSV value")

            features.append(x)
            targets.append([y])

    if not features:
        raise ValueError("Empty dataset")

    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        feature_names
    )


def evaluate(model, x, y):
    with torch.no_grad():
        predictions = model(x)

        if predictions.shape != y.shape:
            raise ValueError("Prediction shape mismatch")

        errors = predictions - y

        mse = errors.square().mean().item()
        mae = errors.abs().mean().item()
        rmse = math.sqrt(mse)

    return {
        "samples": len(x),
        "mse": round(mse, 6),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6)
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="checkpoints/merged_model.pt"
    )
    parser.add_argument(
        "--train",
        default="data/demo_train.csv"
    )
    parser.add_argument(
        "--val",
        default="data/demo_val.csv"
    )

    args = parser.parse_args()

    model, blocks = load_model(ROOT / args.model)

    input_size = blocks[0]["in"]

    train_x, train_y, train_cols = load_csv(
        ROOT / args.train, input_size
    )

    val_x, val_y, val_cols = load_csv(
        ROOT / args.val, input_size
    )

    if train_cols != val_cols:
        raise ValueError(
            "Training and validation feature order differs"
        )

    train_metrics = evaluate(
        model, train_x, train_y
    )

    val_metrics = evaluate(
        model, val_x, val_y
    )

    print("\n===== DISTRIBUTED GPU EVALUATION =====")
    print("Model:", args.model)

    print("\nTRAINING DATA")
    for key, value in train_metrics.items():
        print(f"{key.upper()}: {value}")

    print("\nVALIDATION DATA")
    for key, value in val_metrics.items():
        print(f"{key.upper()}: {value}")

    report = {
        "model": args.model,
        "training": train_metrics,
        "validation": val_metrics,
        "feature_columns": train_cols
    }

    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "evaluation_report.json"

    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\nEvaluation report saved:", output_path)


if __name__ == "__main__":
    main()
