from functools import lru_cache

MIB = 1024 * 1024


def plan_model(blocks, nodes, batch_size=32):
    devices = []

    for node in nodes:
        if not node.get("online") or not node.get("address"):
            continue

        for gpu in node["gpus"]:
            usable = gpu["free_mib"] - 512

            if usable > 96:
                devices.append({
                    "hostname": node["hostname"],
                    "address": node["address"],
                    "gpu_index": gpu["index"],
                    "gpu": gpu["name"],
                    "usable_mib": usable
                })

    devices.sort(
        key=lambda x: x["usable_mib"],
        reverse=True
    )

    if not devices:
        raise ValueError("No available CUDA GPU workers")

    def segment_cost(start, end):
        selected = blocks[start:end]

        params = sum(
            b["in"] * b["out"] + b["out"]
            for b in selected
        )

        width = max(
            max(b["in"], b["out"])
            for b in selected
        )

        # Estimated FP32 parameters, gradients,
        # Adam optimizer state + activation allowance.
        return (
            64
            + params * 16 / MIB
            + batch_size * width * 32 / MIB
        )

    for count in range(
        min(len(devices), len(blocks)), 0, -1
    ):
        selected = devices[:count]
        length = len(blocks)

        @lru_cache(None)
        def solve(i, position):
            if i == count:
                if position == length:
                    return (0.0, ())
                return None

            best = None
            remaining = count - i - 1

            for end in range(
                position + 1,
                length - remaining + 1
            ):
                cost = segment_cost(position, end)
                capacity = selected[i]["usable_mib"]

                if cost > capacity:
                    continue

                tail = solve(i + 1, end)
                if tail is None:
                    continue

                score = max(cost / capacity, tail[0])
                candidate = (score, (end,) + tail[1])

                if best is None or candidate[0] < best[0]:
                    best = candidate

            return best

        result = solve(0, 0)

        if result is None:
            continue

        stages = []
        start = 0

        for i, end in enumerate(result[1]):
            stages.append({
                **selected[i],
                "stage_id": f"stage_{i}",
                "start": start,
                "end": end,
                "blocks": blocks[start:end],
                "estimated_mib": round(
                    segment_cost(start, end), 2
                )
            })
            start = end

        return stages

    raise ValueError(
        "Model cannot fit estimated GPU memory. "
        "Reduce batch size or use more suitable GPUs."
    )
