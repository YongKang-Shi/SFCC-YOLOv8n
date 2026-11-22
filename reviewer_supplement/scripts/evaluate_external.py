from __future__ import annotations

import gc
import json
import math
import os
import statistics
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(r"C:\Users\SYK\plantdoc-project")
SEED_EXP = ROOT / "sfca-seed-experiments"
SOURCE = SEED_EXP / "yolov8-source-sfca"
sys.path.insert(0, str(SOURCE))

from ultralytics import YOLO  # noqa: E402


DATA = Path(
    os.environ.get(
        "EXTERNAL_DATA",
        r"C:\Users\SYK\Documents\Codex\2026-10-30\bi-a\work"
        r"\fieldplant-external-v2\data.yaml",
    )
)
OUT = Path(
    os.environ.get(
        "EXTERNAL_OUT",
        str(DATA.parent / "external-runs"),
    )
)
SEEDS = [404, 101, 303, 2026, 3407, 606]
MODELS = ["yolov8n", "sfcc"]
METRICS = ["precision", "recall", "map50", "map50_95"]

P1_SFCC = SEED_EXP / "revision-20261002-sfcc-six-formal"
P1_YOLO = SEED_EXP / "revision-20261002-yolov8n-six-formal"
P2_SFCC = SEED_EXP / "revision-20261003-replication-sfcc-six-formal"
P2_YOLO = SEED_EXP / "revision-20261003-replication-yolov8n-six-formal"

CLASS_NAMES = [
    "Apple Scab Leaf",
    "Apple leaf",
    "Apple rust leaf",
    "Bell_pepper leaf spot",
    "Bell_pepper leaf",
    "Blueberry leaf",
    "Cherry leaf",
    "Corn Gray leaf spot",
    "Corn leaf blight",
    "Corn rust leaf",
    "Peach leaf",
    "Potato leaf early blight",
    "Potato leaf late blight",
    "Potato leaf",
    "Raspberry leaf",
    "Soyabean leaf",
    "Squash Powdery mildew leaf",
    "Strawberry leaf",
    "Tomato two spotted spider mites leaf",
    "Tomato Early blight leaf",
    "Tomato Septoria leaf spot",
    "Tomato leaf bacterial spot",
    "Tomato leaf late blight",
    "Tomato leaf mosaic virus",
    "Tomato leaf yellow virus",
    "Tomato leaf",
    "Tomato mold leaf",
    "grape leaf black rot",
    "grape leaf",
]


def checkpoint_for(seed: int, model_name: str) -> Path:
    if seed in (2026, 3407):
        if model_name == "yolov8n":
            return P1_YOLO / "runs" / f"seed{seed}_yolov8n" / "weights" / "best.pt"
        return P1_SFCC / "runs" / f"seed{seed}_sfca_stage2" / "weights" / "best.pt"
    if model_name == "yolov8n":
        return P2_YOLO / "runs" / f"seed{seed}_yolov8n" / "weights" / "best.pt"
    return P2_SFCC / "runs" / f"seed{seed}_sfca_stage2" / "weights" / "best.pt"


def paired_summary(rows: dict[tuple[str, int], dict]) -> dict:
    summary = {}
    for metric in METRICS:
        differences = [
            rows[("sfcc", seed)][metric] - rows[("yolov8n", seed)][metric]
            for seed in SEEDS
        ]
        mean_difference = statistics.mean(differences)
        sd_difference = statistics.stdev(differences)
        standard_error = sd_difference / math.sqrt(len(differences))
        summary[metric] = {
            "mean_difference": mean_difference,
            "sd_difference": sd_difference,
            "ci95_low": mean_difference - 2.5705818366147395 * standard_error,
            "ci95_high": mean_difference + 2.5705818366147395 * standard_error,
            "wins": sum(value > 0 for value in differences),
            "n": len(differences),
        }
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for model_name in MODELS:
        for seed in SEEDS:
            case = f"{model_name}_seed{seed}"
            checkpoint = checkpoint_for(seed, model_name)
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            print(f"START {case}", flush=True)
            model = YOLO(str(checkpoint), task="detect")
            try:
                metrics = model.val(
                    data=str(DATA),
                    split="test",
                    imgsz=640,
                    batch=32,
                    device=0,
                    workers=2,
                    conf=0.001,
                    iou=0.7,
                    max_det=300,
                    plots=False,
                    project=str(OUT),
                    name=case,
                    exist_ok=True,
                    verbose=False,
                )
                present_indices = [int(index) for index in metrics.box.ap_class_index]
                row = {
                    "model": model_name,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "precision": float(metrics.box.mp),
                    "recall": float(metrics.box.mr),
                    "map50": float(metrics.box.map50),
                    "map50_95": float(metrics.box.map),
                    "present_classes": {
                        str(class_index): {
                            "class_name": CLASS_NAMES[class_index],
                            "precision": float(metrics.box.p[position]),
                            "recall": float(metrics.box.r[position]),
                            "ap50": float(metrics.box.ap50[position]),
                            "ap50_95": float(metrics.box.ap[position]),
                        }
                        for position, class_index in enumerate(present_indices)
                    },
                }
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()
            (OUT / f"{case}.json").write_text(
                json.dumps(row, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            results[(model_name, seed)] = row
            print(
                json.dumps(
                    {
                        "case": case,
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "map50": row["map50"],
                        "map50_95": row["map50_95"],
                    }
                ),
                flush=True,
            )

    aggregate = {"models": {}, "paired_sfcc_minus_yolov8n": {}}
    for model_name in MODELS:
        aggregate["models"][model_name] = {
            metric: {
                "mean": statistics.mean(
                    results[(model_name, seed)][metric] for seed in SEEDS
                ),
                "sd": statistics.stdev(
                    results[(model_name, seed)][metric] for seed in SEEDS
                ),
            }
            for metric in METRICS
        }
    aggregate["paired_sfcc_minus_yolov8n"] = paired_summary(results)
    (OUT / "external_aggregate.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
