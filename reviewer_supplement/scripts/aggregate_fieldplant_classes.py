from __future__ import annotations

import json
import math
import os
import statistics
from pathlib import Path


ROOT = Path(os.environ.get(
    "EXTERNAL_OUT",
    r"C:\Users\SYK\Documents\Codex\2026-10-30\bi-a\work"
    r"\fieldplant-external-v2\external-runs",
))
SEEDS = [404, 101, 303, 2026, 3407, 606]
MODELS = ["yolov8n", "sfcc"]


def summary(values: list[float]) -> dict:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    se = sd / math.sqrt(len(values))
    return {
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - 2.5705818366147395 * se,
        "ci95_high": mean + 2.5705818366147395 * se,
        "wins": sum(value > 0 for value in values),
        "n": len(values),
    }


def main() -> None:
    rows = {}
    for model in MODELS:
        for seed in SEEDS:
            path = ROOT / f"{model}_seed{seed}.json"
            rows[(model, seed)] = json.loads(path.read_text(encoding="utf-8"))

    class_ids = sorted({
        int(class_id)
        for row in rows.values()
        for class_id in row["present_classes"]
    })
    output = {"classes": {}}
    for class_id in class_ids:
        class_name = next(
            row["present_classes"][str(class_id)]["class_name"]
            for row in rows.values()
            if str(class_id) in row["present_classes"]
        )
        output["classes"][str(class_id)] = {
            "class_name": class_name,
            "models": {
                model: {
                    metric: {
                        "mean": statistics.mean(
                            rows[(model, seed)]["present_classes"][
                                str(class_id)
                            ][metric]
                            for seed in SEEDS
                        ),
                        "sd": statistics.stdev(
                            rows[(model, seed)]["present_classes"][
                                str(class_id)
                            ][metric]
                            for seed in SEEDS
                        ),
                    }
                    for metric in ("ap50", "ap50_95")
                }
                for model in MODELS
            },
            "paired_sfcc_minus_yolov8n": {
                metric: summary([
                    rows[("sfcc", seed)]["present_classes"][str(class_id)][
                        metric
                    ]
                    - rows[("yolov8n", seed)]["present_classes"][
                        str(class_id)
                    ][metric]
                    for seed in SEEDS
                ])
                for metric in ("ap50", "ap50_95")
            },
        }
    (ROOT / "per_class_aggregate.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    print(ROOT / "per_class_aggregate.json")


if __name__ == "__main__":
    main()
