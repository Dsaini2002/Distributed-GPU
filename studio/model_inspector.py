import ast
import csv


def inspect_model(path):
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))

    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_model"
    ]

    calls = []

    for function in functions:
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue

            target = node.func

            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "nn"
            ):
                calls.append(target.attr)

    allowed = {"Sequential", "Linear", "ReLU"}
    unsupported = sorted(set(calls) - allowed)

    candidate = (
        bool(functions)
        and "Sequential" in calls
        and calls.count("Linear") >= 2
        and not unsupported
    )

    return {
        "type": "PyTorch architecture candidate",
        "linear_layers_detected": calls.count("Linear"),
        "relu_layers_detected": calls.count("ReLU"),
        "other_nn_calls": unsupported,
        "current_engine_candidate": candidate,
        "message": (
            "Candidate for current Linear/ReLU trainer"
            if candidate else
            "Needs architecture review or another adapter"
        ),
        "note": (
            "Static analysis only. Final compatibility "
            "is checked when training starts."
        )
    }


def inspect_dataset(path):
    with path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)
        columns = reader.fieldnames or []

        if len(set(columns)) != len(columns):
            raise ValueError("Duplicate CSV column names")

        features = [c for c in columns if c != "target"]
        has_target = columns.count("target") == 1

        checked = 0
        numeric_sample = True

        for row in reader:
            if checked >= 10:
                break

            try:
                import math
                values = [
                    float(row[c]) for c in columns
                ]
                if not all(math.isfinite(x) for x in values):
                    numeric_sample = False
            except (TypeError, ValueError, KeyError):
                numeric_sample = False

            checked += 1

    return {
        "columns": columns,
        "feature_count": len(features),
        "has_target": has_target,
        "sample_rows_checked": checked,
        "sample_is_numeric": numeric_sample,
        "current_engine_candidate": (
            has_target
            and bool(features)
            and checked > 0
            and numeric_sample
        ),
        "note": (
            "Only first 10 rows inspected. "
            "Full validation occurs before training."
        )
    }
