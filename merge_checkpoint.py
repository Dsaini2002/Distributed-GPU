
import json
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "checkpoints"


def build_layers(blocks):
    layers = []

    for block in blocks:
        layers.append(
            nn.Linear(block["in"], block["out"])
        )

        if block["relu"]:
            layers.append(nn.ReLU())

    return layers


def main():
    metadata = json.loads(
        (CHECKPOINTS / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    blocks = metadata["blocks"]
    split = metadata["split"]

    if not 0 < split < len(blocks):
        raise ValueError("Invalid checkpoint split")

    stage_a = nn.Sequential(
        *build_layers(blocks[:split])
    )

    stage_b = nn.Sequential(
        *build_layers(blocks[split:])
    )

    weights_a = torch.load(
        CHECKPOINTS / "controller_stage.pt",
        map_location="cpu",
        weights_only=True
    )

    weights_b = torch.load(
        CHECKPOINTS / "worker_stage.pt",
        map_location="cpu",
        weights_only=True
    )

    stage_a.load_state_dict(weights_a, strict=True)
    stage_b.load_state_dict(weights_b, strict=True)

    full_model = nn.Sequential(
        *list(stage_a.children()),
        *list(stage_b.children())
    )

    full_model.eval()

    output_path = CHECKPOINTS / "merged_model.pt"

    torch.save({
        "blocks": blocks,
        "model_state_dict": full_model.state_dict(),
        "task": metadata["task"]
    }, output_path)

    print("MODEL MERGING SUCCESSFUL!")
    print("Total Linear layers:", len(blocks))
    print("Saved:", output_path)

    example = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8]],
        dtype=torch.float32
    )

    if example.shape[1] == blocks[0]["in"]:
        with torch.no_grad():
            prediction = full_model(example)

        print("Example prediction:", prediction.item())


if __name__ == "__main__":
    main()
