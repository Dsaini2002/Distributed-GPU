def create_plan(nodes, blocks=16, block_mib=256, headroom=768):
    """Memory-based preview, not an actual execution plan."""
    devices = []

    for node in nodes:
        for gpu in node["gpus"]:
            usable = max(0, gpu["free_mib"] - headroom)
            capacity = usable // block_mib

            if capacity:
                devices.append({
                    "node": node["hostname"],
                    "gpu": gpu["name"],
                    "index": gpu["index"],
                    "capacity": capacity,
                    "assigned": 0
                })

    if sum(d["capacity"] for d in devices) < blocks:
        raise ValueError(
            "Not enough estimated GPU memory for these blocks"
        )

    for _ in range(blocks):
        available = [
            d for d in devices
            if d["assigned"] < d["capacity"]
        ]

        selected = min(
            available,
            key=lambda d: d["assigned"] / d["capacity"]
        )
        selected["assigned"] += 1

    start = 0
    stages = []

    for device in devices:
        count = device["assigned"]
        if not count:
            continue

        stages.append({
            "computer": device["node"],
            "gpu": device["gpu"],
            "gpu_index": device["index"],
            "start_block": start,
            "end_block": start + count - 1,
            "blocks": count,
            "estimated_mib": count * block_mib
        })

        start += count

    return {
        "total_blocks": blocks,
        "stage_count": len(stages),
        "stages": stages,
        "note": "Memory-only demonstration; no training started"
    }
