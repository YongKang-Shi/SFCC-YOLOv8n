"""Six-seed YOLOv8n baseline under the completed C2fGhost formal protocol."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT = Path(r"C:\Users\SYK\plantdoc-project")
ROOT = PROJECT / "sfca-seed-experiments" / "revision-20261002-yolov8n-six-formal"
SOURCE = PROJECT / "sfca-seed-experiments" / "yolov8-source-sfca"
DATA_ROOT = PROJECT / "plantdoc-github-official-yolo"
DATA = DATA_ROOT / "plantdoc_val.yaml"
DATA_MANIFEST = DATA_ROOT / "manifest.json"
PRETRAINED = PROJECT / "official-weights" / "yolov8n-v8.1.0-official.pt"
C2FGHOST_ROOT = PROJECT / "sfca-seed-experiments" / "full-github-six-formal"
SEEDS = [404, 101, 303, 2026, 3407, 606]
EPOCHS = 300
PATIENCE = 50
TAIL_CLASS_IDS = [18, 13, 23, 6, 4, 27, 15, 28, 3, 7]
RUNS = ROOT / "runs"
EVALS = ROOT / "evaluations"
RESULTS = ROOT / "results"
STATE = ROOT / "pipeline_state.json"
LOG = ROOT / "pipeline_events.log"

sys.path.insert(0, str(SOURCE))
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import init_seeds  # noqa: E402
import torch  # noqa: E402


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def event(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    init_seeds(seed, deterministic=True)


def release_model(model: object | None) -> None:
    if model is not None:
        trainer = getattr(model, "trainer", None)
        if trainer is not None:
            for attribute in ("train_loader", "test_loader"):
                loader = getattr(trainer, attribute, None)
                close = getattr(loader, "close", None)
                if close is not None:
                    close()
                setattr(trainer, attribute, None)
            model.trainer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def result_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def best_training_row(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    epoch_key = next(key for key in rows[0] if key.strip() == "epoch")
    map_key = next(key for key in rows[0] if "mAP50-95" in key)
    best = max(rows, key=lambda row: float(row[map_key]))
    return {"epoch": int(float(best[epoch_key])), "map50_95": float(best[map_key])}


def checkpoint_is_finalized(path: Path) -> bool:
    if not path.exists():
        return False
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint.get("epoch") == -1 and checkpoint.get("optimizer") is None


def common_args(seed: int, name: str) -> dict:
    return {
        "data": str(DATA),
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "batch": 32,
        "imgsz": 640,
        "device": 0,
        "workers": 2,
        "optimizer": "auto",
        "verbose": False,
        "seed": seed,
        "deterministic": True,
        "single_cls": False,
        "rect": False,
        "cos_lr": False,
        "close_mosaic": 10,
        "amp": True,
        "cache": "disk",
        "val": True,
        "split": "val",
        "save": True,
        "save_period": -1,
        "plots": False,
        "iou": 0.7,
        "max_det": 300,
        "project": str(RUNS),
        "name": name,
        "exist_ok": True,
    }


def train(seed: int) -> Path:
    run_name = f"seed{seed}_yolov8n"
    run_dir = RUNS / run_name
    marker = run_dir / "training_complete.json"
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    if marker.exists() and best.exists():
        event(f"SKIP completed train seed={seed}")
        return best

    model = None
    try:
        rows = result_rows(results_csv)
        if best.exists() and last.exists() and rows > 0 and checkpoint_is_finalized(last):
            event(f"RECOVER finalized train seed={seed} rows={rows}")
        elif last.exists() and rows > 0:
            event(f"RESUME train seed={seed} rows={rows}")
            model = YOLO(str(last), task="detect")
            model.train(resume=True, workers=2, cache="disk", plots=False)
        else:
            event(f"START train seed={seed} epochs={EPOCHS} patience={PATIENCE}")
            set_seed(seed)
            model = YOLO(str(PRETRAINED), task="detect")
            model.train(**common_args(seed, run_name))
    finally:
        release_model(model)

    if not best.exists():
        if last.exists():
            best = last
        else:
            raise RuntimeError(f"No checkpoint produced: {run_dir}")
    payload = {
        "completed_at": now(),
        "seed": seed,
        "model": "yolov8n",
        "max_epochs": EPOCHS,
        "patience": PATIENCE,
        "actual_epochs": result_rows(results_csv),
        "best_training_row": best_training_row(results_csv),
        "checkpoint": str(best),
        "checkpoint_sha256": sha256(best),
    }
    write_json(marker, payload)
    event(
        f"COMPLETE train seed={seed} actual_epochs={payload['actual_epochs']} "
        f"best={payload['best_training_row']}"
    )
    return best


def evaluate(seed: int, checkpoint: Path, test_box_counts: list[int]) -> dict:
    eval_name = f"seed{seed}_yolov8n_final"
    eval_dir = EVALS / eval_name
    marker = eval_dir / "metrics.json"
    if marker.exists():
        event(f"SKIP completed eval seed={seed}")
        return json.loads(marker.read_text(encoding="utf-8"))

    event(f"START final TEST evaluation seed={seed}")
    model = None
    try:
        model = YOLO(str(checkpoint), task="detect")
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
            save_json=False,
            project=str(EVALS),
            name=eval_name,
            exist_ok=True,
            verbose=False,
        )
        precision = float(metrics.box.mp)
        recall = float(metrics.box.mr)
        class_maps = [float(value) for value in metrics.box.maps]
        evaluable_tail = [class_id for class_id in TAIL_CLASS_IDS if test_box_counts[class_id] > 0]
        result = {
            "evaluated_at": now(),
            "model": "yolov8n",
            "seed": seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "tail_ap_evaluable": sum(class_maps[index] for index in evaluable_tail) / len(evaluable_tail),
            "tail_evaluable_class_ids": evaluable_tail,
            "tail_absent_class_ids": [class_id for class_id in TAIL_CLASS_IDS if test_box_counts[class_id] == 0],
            "class_ap50_95": {str(index): value for index, value in enumerate(class_maps)},
        }
    finally:
        release_model(model)
    write_json(marker, result)
    event(
        f"COMPLETE TEST seed={seed} P={result['precision']:.4f} R={result['recall']:.4f} "
        f"F1={result['f1']:.4f} mAP50={result['map50']:.4f} "
        f"mAP50-95={result['map50_95']:.4f} tail8={result['tail_ap_evaluable']:.4f}"
    )
    return result


def preflight() -> dict:
    required = [DATA, DATA_MANIFEST, PRETRAINED]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    yaml_text = DATA.read_text(encoding="utf-8")
    if "val_list.txt" not in yaml_text or "test_list.txt" not in yaml_text:
        raise RuntimeError("DATA yaml does not use the training-only val split")
    manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    train_split = manifest["splits"]["train"]
    test_split = manifest["splits"]["test"]
    if (train_split["converted_images"], test_split["converted_images"]) != (2355, 237):
        raise RuntimeError("Dataset is not the audited 2355/237 GitHub conversion")
    if (train_split["boxes"], test_split["boxes"]) != (8467, 452):
        raise RuntimeError("Dataset box counts changed")
    return {
        "checked_at": now(),
        "source_commit": manifest["source_commit"],
        "train_images": train_split["converted_images"],
        "train_boxes": train_split["boxes"],
        "test_images": test_split["converted_images"],
        "test_boxes": test_split["boxes"],
        "test_box_counts": [int(test_split["class_counts"][str(index)]) for index in range(29)],
        "data_manifest_sha256": sha256(DATA_MANIFEST),
        "data_yaml_sha256": sha256(DATA),
        "pretrained_sha256": sha256(PRETRAINED),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def write_summary(results: list[dict]) -> dict:
    metric_names = ["precision", "recall", "f1", "map50", "map50_95", "tail_ap_evaluable"]
    summary = {
        "completed_at": now(),
        "seeds": [item["seed"] for item in results],
        "count": len(results),
        "mean": {
            metric: float(np.mean([item[metric] for item in results])) for metric in metric_names
        },
        "sample_sd": {
            metric: float(np.std([item[metric] for item in results], ddof=1))
            if len(results) > 1
            else 0.0
            for metric in metric_names
        },
        "per_seed": results,
    }
    write_json(RESULTS / "six_seed_summary.json", summary)
    return summary


def main() -> None:
    for directory in (ROOT, RUNS, EVALS, RESULTS):
        directory.mkdir(parents=True, exist_ok=True)
    (ROOT / "pipeline.pid").write_text(str(os.getpid()), encoding="ascii")
    audit = preflight()
    state = {
        "started_at": now(),
        "status": "running",
        "seeds": SEEDS,
        "model": "YOLOv8n",
        "model_protocol": {"epochs": EPOCHS, "patience": PATIENCE},
        "execution": "one seed at a time; same formal C2fGhost environment except model architecture",
        "dataset_protocol": "audited GitHub conversion 2355/237 with a training-only validation split (2119/236/237, seed 20261002); TEST evaluated once",
        "preflight": audit,
        "current": None,
        "completed_seeds": [],
    }
    write_json(STATE, state)
    event(f"PREFLIGHT OK {json.dumps(audit, ensure_ascii=False)}")
    completed: list[dict] = []
    try:
        for seed in SEEDS:
            result_file = RESULTS / f"seed{seed}_complete.json"
            if result_file.exists():
                result = json.loads(result_file.read_text(encoding="utf-8"))
                completed.append(result)
                state["completed_seeds"].append(seed)
                write_json(STATE, state)
                event(f"SKIP completed seed={seed}")
                continue
            state["current"] = {"seed": seed, "phase": "train", "started_at": now()}
            write_json(STATE, state)
            checkpoint = train(seed)
            state["current"] = {"seed": seed, "phase": "final_evaluation", "started_at": now()}
            write_json(STATE, state)
            result = evaluate(seed, checkpoint, audit["test_box_counts"])
            write_json(result_file, result)
            completed.append(result)
            state["completed_seeds"].append(seed)
            state["last_completed"] = result
            state["current"] = None
            write_json(STATE, state)
            event(f"SEED COMPLETE seed={seed}")

        summary = write_summary(completed)
        state["status"] = "completed"
        state["completed_at"] = now()
        state["current"] = None
        state["summary"] = summary["mean"]
        write_json(STATE, state)
        event("ALL SIX YOLOv8n SEEDS COMPLETE")
    except BaseException as error:
        state["status"] = "failed"
        state["failed_at"] = now()
        state["error"] = repr(error)
        state["traceback"] = traceback.format_exc()
        write_json(STATE, state)
        event(f"FAILED {error!r}")
        raise


if __name__ == "__main__":
    main()
