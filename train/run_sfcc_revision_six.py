"""Sequential six-seed formal C2fGhost/SFCA experiment on the audited PlantDoc GitHub split."""

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
ROOT = PROJECT / "sfca-seed-experiments" / "revision-20261002-sfcc-six-formal"
SOURCE = PROJECT / "sfca-seed-experiments" / "yolov8-source-sfca"
DATA_ROOT = PROJECT / "plantdoc-github-official-yolo"
DATA = DATA_ROOT / "plantdoc_val.yaml"
DATA_MANIFEST = DATA_ROOT / "manifest.json"
PRETRAINED = PROJECT / "official-weights" / "yolov8n-v8.1.0-official.pt"
CONFIGS = {
    "c2fghost": PROJECT
    / "sfca-seed-experiments"
    / "github-official-paired-e20"
    / "configs"
    / "yolov8n-c2fghost-plantdoc-github.yaml",
    "sfca": PROJECT
    / "sfca-seed-experiments"
    / "github-official-paired-e20"
    / "configs"
    / "yolov8n-sfca-c2fghost-plantdoc-github.yaml",
}
SEEDS = [404, 101, 303, 2026, 3407, 606]
MODEL_PROTOCOL = {
    "c2fghost": {"epochs": 300, "patience": 50},
    "sfca": {"epochs": 300, "patience": 50},
}
STAGE2_EPOCHS = 2
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
    """Return True for an Ultralytics checkpoint stripped after a completed train call."""
    if not path.exists():
        return False
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint.get("epoch") == -1 and checkpoint.get("optimizer") is None


def common_args(seed: int, epochs: int, patience: int, project: Path, name: str) -> dict:
    return {
        "data": str(DATA),
        "epochs": epochs,
        "patience": patience,
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
        # Early stopping and checkpoint selection use the training-only
        # validation split defined in plantdoc_val.yaml; TEST is held out.
        "val": True,
        "split": "val",
        "save": True,
        "save_period": -1,
        "plots": False,
        "iou": 0.7,
        "max_det": 300,
        "project": str(project),
        "name": name,
        "exist_ok": True,
    }


def train_stage1(model_name: str, seed: int) -> Path:
    protocol = MODEL_PROTOCOL[model_name]
    run_name = f"seed{seed}_{model_name}_stage1"
    run_dir = RUNS / run_name
    marker = run_dir / "training_complete.json"
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    if marker.exists() and best.exists():
        event(f"SKIP completed stage1 seed={seed} model={model_name}")
        return best

    model = None
    os.environ.pop("ULTRALYTICS_SFCA_ONLY", None)
    try:
        rows = result_rows(results_csv)
        if best.exists() and last.exists() and rows > 0 and checkpoint_is_finalized(last):
            event(
                f"RECOVER finalized stage1 seed={seed} model={model_name} rows={rows}; "
                "creating missing completion marker without retraining"
            )
        elif last.exists() and rows > 0:
            event(f"RESUME stage1 seed={seed} model={model_name} rows={rows}")
            model = YOLO(str(last), task="detect")
            model.train(resume=True, workers=2, cache="disk", plots=False)
        else:
            event(
                f"START stage1 seed={seed} model={model_name} "
                f"epochs={protocol['epochs']} patience={protocol['patience']}"
            )
            set_seed(seed)
            model = YOLO(str(CONFIGS[model_name]), task="detect")
            model.load(str(PRETRAINED))
            model.train(
                **common_args(
                    seed,
                    protocol["epochs"],
                    protocol["patience"],
                    RUNS,
                    run_name,
                )
            )
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
        "model": model_name,
        "max_epochs": protocol["epochs"],
        "patience": protocol["patience"],
        "actual_epochs": result_rows(results_csv),
        "best_training_row": best_training_row(results_csv),
        "checkpoint": str(best),
        "checkpoint_sha256": sha256(best),
    }
    write_json(marker, payload)
    event(
        f"COMPLETE stage1 seed={seed} model={model_name} "
        f"actual_epochs={payload['actual_epochs']} best={payload['best_training_row']}"
    )
    return best


def freeze_non_sfca_batchnorm(trainer) -> None:
    for name, module in trainer.model.named_modules():
        if ".sfca" not in name and isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def train_sfca_stage2(seed: int, stage1: Path) -> Path:
    run_name = f"seed{seed}_sfca_stage2"
    run_dir = RUNS / run_name
    marker = run_dir / "training_complete.json"
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    if marker.exists() and best.exists():
        event(f"SKIP completed stage2 seed={seed}")
        return best

    model = None
    previous = os.environ.get("ULTRALYTICS_SFCA_ONLY")
    os.environ["ULTRALYTICS_SFCA_ONLY"] = "1"
    try:
        rows = result_rows(results_csv)
        if last.exists() and rows > 0:
            event(f"RESUME SFCA stage2 seed={seed} rows={rows}")
            model = YOLO(str(last), task="detect")
            model.add_callback("on_train_epoch_start", freeze_non_sfca_batchnorm)
            model.train(resume=True, workers=2, cache="disk", plots=False)
        else:
            event(f"START SFCA stage2 seed={seed} epochs={STAGE2_EPOCHS}")
            set_seed(seed)
            model = YOLO(str(stage1), task="detect")
            model.add_callback("on_train_epoch_start", freeze_non_sfca_batchnorm)
            args = common_args(seed, STAGE2_EPOCHS, 0, RUNS, run_name)
            args["warmup_epochs"] = 0.0
            model.train(**args)
    finally:
        if previous is None:
            os.environ.pop("ULTRALYTICS_SFCA_ONLY", None)
        else:
            os.environ["ULTRALYTICS_SFCA_ONLY"] = previous
        release_model(model)

    if not best.exists():
        if last.exists():
            best = last
        else:
            raise RuntimeError(f"No stage2 checkpoint produced: {run_dir}")
    payload = {
        "completed_at": now(),
        "seed": seed,
        "model": "sfca",
        "stage": 2,
        "epochs": STAGE2_EPOCHS,
        "warmup_epochs": 0.0,
        "non_sfca_batchnorm_frozen": True,
        "source_stage1": str(stage1),
        "actual_epochs": result_rows(results_csv),
        "best_training_row": best_training_row(results_csv),
        "checkpoint": str(best),
        "checkpoint_sha256": sha256(best),
    }
    write_json(marker, payload)
    event(f"COMPLETE SFCA stage2 seed={seed} best={payload['best_training_row']}")
    return best


def evaluate(model_name: str, seed: int, checkpoint: Path, test_box_counts: list[int]) -> dict:
    eval_name = f"seed{seed}_{model_name}_final"
    eval_dir = EVALS / eval_name
    marker = eval_dir / "metrics.json"
    if marker.exists():
        event(f"SKIP completed eval seed={seed} model={model_name}")
        return json.loads(marker.read_text(encoding="utf-8"))

    event(f"START final TEST evaluation seed={seed} model={model_name}")
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
            "model": model_name,
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
        f"COMPLETE TEST seed={seed} model={model_name} P={result['precision']:.4f} "
        f"R={result['recall']:.4f} F1={result['f1']:.4f} mAP50={result['map50']:.4f} "
        f"mAP50-95={result['map50_95']:.4f} tail8={result['tail_ap_evaluable']:.4f}"
    )
    return result


def preflight() -> dict:
    required = [DATA, DATA_MANIFEST, PRETRAINED, *CONFIGS.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    yaml_text = DATA.read_text(encoding="utf-8")
    if "val_list.txt" not in yaml_text or "test_list.txt" not in yaml_text:
        raise RuntimeError("DATA yaml does not use the training-only val split")
    manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))
    train = manifest["splits"]["train"]
    test = manifest["splits"]["test"]
    if (train["converted_images"], test["converted_images"]) != (2355, 237):
        raise RuntimeError("Dataset is not the audited 2355/237 GitHub conversion")
    if (train["boxes"], test["boxes"]) != (8467, 452):
        raise RuntimeError("Dataset box counts changed")
    # Verify the files the trainer can actually discover, not just manifest totals.
    image_suffixes = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"}
    for split_name, split in (("train", train), ("test", test)):
        expected_images = {record["image_output"] for record in split["records"]}
        actual_images = {
            path.name
            for path in (DATA_ROOT / "images" / split_name).iterdir()
            if path.is_file() and path.suffix.lower() in image_suffixes
        }
        if actual_images != expected_images:
            raise RuntimeError(
                f"{split_name} image directory differs from manifest: "
                f"expected={len(expected_images)} actual={len(actual_images)} "
                f"extra={sorted(actual_images - expected_images)} "
                f"missing={sorted(expected_images - actual_images)}"
            )
        expected_labels = {f"{record['safe_stem']}.txt" for record in split["records"]}
        actual_labels = {path.name for path in (DATA_ROOT / "labels" / split_name).glob("*.txt")}
        if actual_labels != expected_labels:
            raise RuntimeError(
                f"{split_name} label directory differs from manifest: "
                f"expected={len(expected_labels)} actual={len(actual_labels)}"
            )
    support = [int(train["image_support_counts"][str(index)]) for index in range(29)]
    if support != [83, 82, 79, 62, 53, 106, 47, 64, 180, 106, 103, 112, 98, 3, 112, 57, 125, 88, 2, 81, 140, 104, 102, 45, 70, 64, 85, 56, 57]:
        raise RuntimeError("TRAIN support counts changed")
    return {
        "checked_at": now(),
        "source_commit": manifest["source_commit"],
        "train_images": train["converted_images"],
        "train_boxes": train["boxes"],
        "test_images": test["converted_images"],
        "test_boxes": test["boxes"],
        "test_box_counts": [int(test["class_counts"][str(index)]) for index in range(29)],
        "data_manifest_sha256": sha256(DATA_MANIFEST),
        "data_yaml_sha256": sha256(DATA),
        "pretrained_sha256": sha256(PRETRAINED),
        "config_sha256": {name: sha256(path) for name, path in CONFIGS.items()},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def write_seed_result(seed: int, baseline: dict, sfca: dict) -> dict:
    metrics = ["precision", "recall", "f1", "map50", "map50_95", "tail_ap_evaluable"]
    payload = {
        "completed_at": now(),
        "seed": seed,
        "c2fghost": baseline,
        "sfca": sfca,
        "sfca_minus_c2fghost": {metric: sfca[metric] - baseline[metric] for metric in metrics},
    }
    write_json(RESULTS / f"seed{seed}_complete.json", payload)
    return payload


def main() -> None:
    for directory in (ROOT, RUNS, EVALS, RESULTS):
        directory.mkdir(parents=True, exist_ok=True)
    (ROOT / "pipeline.pid").write_text(str(os.getpid()), encoding="ascii")
    audit = preflight()
    state = {
        "started_at": now(),
        "status": "running",
        "seeds": SEEDS,
        "model_protocol": MODEL_PROTOCOL,
        "stage2_epochs": STAGE2_EPOCHS,
        "execution": "revision: one seed at a time; C2fGhost and SFCA stage1+stage2, both 300 epochs / patience 50; TEST evaluated once after training",
        "dataset_protocol": "audited GitHub conversion 2355/237 with a training-only validation split (2119/236/237, seed 20261002); TEST evaluated once",
        "preflight": audit,
        "current": None,
        "completed_seeds": [],
    }
    write_json(STATE, state)
    event(f"PREFLIGHT OK {json.dumps(audit, ensure_ascii=False)}")
    try:
        for seed in SEEDS:
            existing = RESULTS / f"seed{seed}_complete.json"
            if existing.exists():
                state["completed_seeds"].append(seed)
                write_json(STATE, state)
                event(f"SKIP completed pair seed={seed}")
                continue

            state["current"] = {"seed": seed, "model": "c2fghost", "phase": "stage1", "started_at": now()}
            write_json(STATE, state)
            baseline_checkpoint = train_stage1("c2fghost", seed)
            baseline = evaluate("c2fghost", seed, baseline_checkpoint, audit["test_box_counts"])

            state["current"] = {"seed": seed, "model": "sfca", "phase": "stage1", "started_at": now()}
            write_json(STATE, state)
            sfca_stage1 = train_stage1("sfca", seed)

            state["current"] = {"seed": seed, "model": "sfca", "phase": "stage2", "started_at": now()}
            write_json(STATE, state)
            sfca_checkpoint = train_sfca_stage2(seed, sfca_stage1)
            sfca = evaluate("sfca", seed, sfca_checkpoint, audit["test_box_counts"])
            paired = write_seed_result(seed, baseline, sfca)
            state["completed_seeds"].append(seed)
            state["last_completed"] = paired
            state["current"] = None
            write_json(STATE, state)
            event(
                f"PAIR COMPLETE seed={seed} delta_mAP50-95="
                f"{paired['sfca_minus_c2fghost']['map50_95']:+.6f}"
            )

        state["status"] = "completed"
        state["completed_at"] = now()
        state["current"] = None
        write_json(STATE, state)
        event("ALL SIX SEEDS COMPLETE")
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
