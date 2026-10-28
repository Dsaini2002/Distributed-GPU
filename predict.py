
import argparse
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent


def load_trained_model(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=str(ROOT / "checkpoints" / "merged_model.pt")
    )

    parser.add_argument(
        "--features",
        nargs="+",
        type=float,
        required=True
    )

    args = parser.parse_args()

    model, blocks = load_trained_model(args.model)

    expected = blocks[0]["in"]

    if len(args.features) != expected:
        parser.error(
            f"Expected {expected} features, "
            f"received {len(args.features)}"
        )

    x = torch.tensor(
        [args.features],
        dtype=torch.float32
    )

    with torch.no_grad():
        prediction = model(x)

    print("\n===== DISTRIBUTED GPU PREDICTION =====")
    print("Input features:", args.features)
    print("Prediction:", prediction.item())


if __name__ == "__main__":
    main()
