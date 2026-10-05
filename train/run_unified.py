"""Unified same-pipeline comparison under Ultralytics 8.3.253 with the ported SFCA tree.

Every model in this run uses:
  * the same ported source tree   -> sfca-8.3.253-src (ultralytics 8.3.253)
  * the same audited data split   -> plantdoc_val.yaml (2119 / 236 / 237, TEST scored once)
  * the same common_args           -> seed 404, 300 epochs / patience 50, batch 32, imgsz 640,
                                      optimizer "auto", deterministic, AMP, workers 2, cache disk,
                                      close_mosaic 10, conf 0.001 / IoU 0.7 / max_det 300

Models: SFCC (SFCA stage1 + SFCA-only stage2), YOLOv8n, and the five external detectors
YOLOv5nu / YOLOv9t / YOLOv10n / YOLO11n / YOLO12n, each starting from its official checkpoint.

The run is resumable: finished models are skipped via per-model completion JSON markers.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT = Path(r"C:\Users\SYK\plantdoc-project")
SOURCE = PROJECT / "sfca-8.3.253-src"
ROOT = PROJECT / "sfca-seed-experiments" / "final-20261004-unified-8.3.253"
DATA_ROOT = PROJECT / "plantdoc-github-official-yolo"
DATA = DATA_ROOT / "plantdoc_val.yaml"
MANIFEST = DATA_ROOT / "manifest.json"
PRETRAINED = PROJECT / "official-weights" / "yolov8n-v8.1.0-official.pt"
CONFIGS = {
    "c2fghost": PROJECT / "sfca-seed-experiments" / "github-official-paired-e20" / "configs" / "yolov8n-c2fghost-plantdoc-github.yaml",
    "sfcc": PROJECT / "sfca-seed-experiments" / "github-official-paired-e20" / "configs" / "yolov8n-sfca-c2fghost-plantdoc-github.yaml",
}

SEED = 404
EPOCHS = 300
PATIENCE = 50
STAGE2_EPOCHS = 2
TAIL_CLASS_IDS = [18, 13, 23, 6, 4, 27, 15, 28, 3, 7]

RUNS = ROOT / "runs"
EVALS = ROOT / "evaluations"
RESULTS = ROOT / "results"
STATE = ROOT / "pipeline_state.json"
LOG = ROOT / "pipeline_events.log"

EXTERNAL_WEIGHTS = {
    "yolov5nu": PROJECT / "yolov5nu.pt",
    "yolov9t": PROJECT / "yolov9t.pt",
    "yolov10n": PROJECT / "yolov10n.pt",
    "yolo11n": PROJECT / "yolo11n.pt",
    "yolo12n": PROJECT / "yolo12n.pt",
}
DEFAULT_MODELS = ["sfcc", "yolov8n", "yolov5nu", "yolov9t", "yolov10n", "yolo11n", "yolo12n"]

sys.path.insert(0, str(SOURCE))

import torch  # noqa: E402

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import init_seeds  # noqa: E402


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def event(message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


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
    epoch_key = next((key for key in rows[0] if key.strip() == "epoch"), None)
    map_key = next((key for key in rows[0] if "mAP50-95" in key), None)
    if epoch_key is None or map_key is None:
        return {}
    best = max(rows, key=lambda row: float(row[map_key]))
    return {"epoch": int(float(best[epoch_key])), "map50_95": float(best[map_key])}


def checkpoint_is_finalized(path: Path) -> bool:
    """True for an Ultralytics checkpoint stripped after a completed train() call."""
    if not path.exists():
        return False
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("epoch") == -1 and checkpoint.get("optimizer") is None


def common_args(name: str, epochs: int = EPOCHS, patience: int = PATIENCE) -> dict:
    """The exact common_args of the SFCC revision pipeline (identical for every model)."""
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
        "seed": SEED,
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


def freeze_non_sfca_batchnorm(trainer) -> None:
    """Stage-2 keeps every non-SFCA BatchNorm in eval mode (identical to the SFCC pipeline)."""
    for name, module in trainer.model.named_modules():
        if ".sfca" not in name and isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def train_single(model_name: str, weights: str | Path) -> Path:
    """Train one plain model (no stage 2) and return its best checkpoint."""
    run_name = f"{model_name}_s{SEED}"
    run_dir = RUNS / run_name
    marker = run_dir / "training_complete.json"
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    if marker.exists() and best.exists():
        event(f"SKIP completed {model_name}")
        return best

    model = None
    os.environ.pop("ULTRALYTICS_SFCA_ONLY", None)
    try:
        rows = result_rows(results_csv)
        if best.exists() and last.exists() and rows > 0 and checkpoint_is_finalized(last):
            event(f"RECOVER finalized {model_name} rows={rows}; creating missing completion marker")
        elif last.exists() and rows > 0:
            event(f"RESUME {model_name} rows={rows}")
            model = YOLO(str(last), task="detect")
            model.train(resume=True, workers=2, cache="disk", plots=False)
        else:
            event(f"START train {model_name} weights={weights}")
            set_seed(SEED)
            model = YOLO(str(weights), task="detect")
            model.train(**common_args(run_name))
    finally:
        release_model(model)

    if not best.exists():
        if last.exists():
            best = last
        else:
            raise RuntimeError(f"No checkpoint produced: {run_dir}")
    write_json(
        marker,
        {
            "completed_at": now(),
            "model": model_name,
            "seed": SEED,
            "started_from": str(weights),
            "actual_epochs": result_rows(results_csv),
            "best_training_row": best_training_row(results_csv),
            "checkpoint": str(best),
            "checkpoint_sha256": sha256(best),
        },
    )
    event(f"COMPLETE train {model_name} best={best_training_row(results_csv)}")
    return best


def train_sfcc_stage1() -> Path:
    run_name = f"sfcc_s{SEED}_stage1"
    run_dir = RUNS / run_name
    marker = run_dir / "training_complete.json"
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    if marker.exists() and best.exists():
        event("SKIP completed sfcc stage1")
        return best

    model = None
    os.environ.pop("ULTRALYTICS_SFCA_ONLY", None)
    try:
        rows = result_rows(results_csv)
        if best.exists() and last.exists() and rows > 0 and checkpoint_is_finalized(last):
            event(f"RECOVER finalized sfcc stage1 rows={rows}; creating missing completion marker")
        elif last.exists() and rows > 0:
            event(f"RESUME sfcc stage1 rows={rows}")
            model = YOLO(str(last), task="detect")
            model.train(resume=True, workers=2, cache="disk", plots=False)
        else:
            event(f"START sfcc stage1 from {PRETRAINED.name}")
            set_seed(SEED)
            model = YOLO(str(CONFIGS["sfcc"]), task="detect")
            model.load(str(PRETRAINED))
            model.train(**common_args(run_name))
    finally:
        release_model(model)

    if not best.exists():
        if last.exists():
            best = last
        else:
            raise RuntimeError(f"No stage1 checkpoint: {run_dir}")
    write_json(
        marker,
        {
            "completed_at": now(),
            "model": "sfcc",
            "stage": 1,
            "seed": SEED,
            "pretrained": str(PRETRAINED),
            "pretrained_sha256": sha256(PRETRAINED),
            "actual_epochs": result_rows(results_csv),
            "best_training_row": best_training_row(results_csv),
            "checkpoint": str(best),
            "checkpoint_sha256": sha256(best),
        },
    )
    event(f"COMPLETE sfcc stage1 best={best_training_row(results_csv)}")
    return best


def train_sfcc_stage2(stage1: Path) -> Path:
    run_name = f"sfcc_s{SEED}_stage2"
    run_dir = RUNS / run_name
    marker = run_dir / "training_complete.json"
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    if marker.exists() and best.exists():
        event("SKIP completed sfcc stage2")
        return best

    model = None
    previous = os.environ.get("ULTRALYTICS_SFCA_ONLY")
    os.environ["ULTRALYTICS_SFCA_ONLY"] = "1"
    try:
        rows = result_rows(results_csv)
        if last.exists() and rows > 0:
            event(f"RESUME sfcc stage2 rows={rows}")
            model = YOLO(str(last), task="detect")
            model.add_callback("on_train_epoch_start", freeze_non_sfca_batchnorm)
            model.train(resume=True, workers=2, cache="disk", plots=False)
        else:
            event(f"START sfcc stage2 epochs={STAGE2_EPOCHS} (SFCA-only, non-SFCA BN frozen)")
            set_seed(SEED)
            model = YOLO(str(stage1), task="detect")
            model.add_callback("on_train_epoch_start", freeze_non_sfca_batchnorm)
            args = common_args(run_name, epochs=STAGE2_EPOCHS, patience=0)
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
            raise RuntimeError(f"No stage2 checkpoint: {run_dir}")
    write_json(
        marker,
        {
            "completed_at": now(),
            "model": "sfcc",
            "stage": 2,
            "seed": SEED,
            "epochs": STAGE2_EPOCHS,
            "warmup_epochs": 0.0,
            "sfca_only": True,
            "non_sfca_batchnorm_frozen": True,
            "source_stage1": str(stage1),
            "actual_epochs": result_rows(results_csv),
            "best_training_row": best_training_row(results_csv),
            "checkpoint": str(best),
            "checkpoint_sha256": sha256(best),
        },
    )
    event(f"COMPLETE sfcc stage2 best={best_training_row(results_csv)}")
    return best


def evaluate(model_name: str, checkpoint: Path, test_box_counts: list[int]) -> dict:
    eval_name = f"{model_name}_s{SEED}_test"
    marker = EVALS / eval_name / "metrics.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
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
        evaluable = [class_id for class_id in TAIL_CLASS_IDS if test_box_counts[class_id] > 0]
        payload = {
            "evaluated_at": now(),
            "model": model_name,
            "seed": SEED,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "tail_ap_evaluable": sum(class_maps[i] for i in evaluable) / len(evaluable),
            "tail_evaluable_class_ids": evaluable,
            "class_ap50_95": {str(i): class_maps[i] for i in range(len(class_maps))},
        }
    finally:
        release_model(model)
    write_json(marker, payload)
    return payload


def complexity(checkpoint: Path) -> dict:
    model = None
    try:
        model = YOLO(str(checkpoint), task="detect")
        params = sum(parameter.numel() for parameter in model.model.parameters())
        try:
            from ultralytics.utils.torch_utils import get_flops

            flops = float(get_flops(model.model, imgsz=640))
        except Exception:
            flops = float("nan")
        return {"params_m": params / 1e6, "gflops_640": flops, "checkpoint_size_mb": os.path.getsize(checkpoint) / 1e6}
    finally:
        release_model(model)


def benchmark(checkpoint: Path) -> dict:
    model = None
    try:
        model = YOLO(str(checkpoint), task="detect")
        image = np.random.default_rng(SEED).integers(0, 255, (640, 640, 3), dtype=np.uint8)
        for _ in range(20):
            model.predict(image, imgsz=640, device=0, half=True, verbose=False)
        torch.cuda.synchronize()
        runs = 100
        start = time.perf_counter()
        for _ in range(runs):
            model.predict(image, imgsz=640, device=0, half=True, verbose=False)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) / runs * 1000.0
        return {
            "latency_ms_batch1_fp16": latency_ms,
            "fps_batch1_fp16": 1000.0 / latency_ms if latency_ms else float("nan"),
            "speed_protocol": "single 640x640 image, batch=1, FP16, warmup 20, 100 timed runs, includes pre/post-processing",
        }
    except Exception as error:  # a benchmark failure must not kill the run
        return {"benchmark_error": repr(error)}
    finally:
        release_model(model)


def preflight() -> dict:
    import ultralytics

    required = [DATA, MANIFEST, PRETRAINED, *CONFIGS.values(), *EXTERNAL_WEIGHTS.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    if not str(ultralytics.__file__).lower().startswith(str(SOURCE).lower()):
        raise RuntimeError(f"ultralytics resolved to {ultralytics.__file__}, expected the ported tree {SOURCE}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    train = manifest["splits"]["train"]
    test = manifest["splits"]["test"]
    if (train["converted_images"], test["converted_images"]) != (2355, 237):
        raise RuntimeError("Dataset is not the audited 2355/237 conversion")
    if (train["boxes"], test["boxes"]) != (8467, 452):
        raise RuntimeError("Dataset box counts changed")
    support = [int(train["image_support_counts"][str(i)]) for i in range(29)]
    if support != [83, 82, 79, 62, 53, 106, 47, 64, 180, 106, 103, 112, 98, 3, 112, 57, 125, 88, 2, 81, 140, 104, 102, 45, 70, 64, 85, 56, 57]:
        raise RuntimeError("TRAIN support counts changed")
    return {
        "checked_at": now(),
        "ultralytics_version": ultralytics.__version__,
        "ultralytics_source": str(ultralytics.__file__),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_images": train["converted_images"],
        "train_boxes": train["boxes"],
        "test_images": test["converted_images"],
        "test_boxes": test["boxes"],
        "test_box_counts": [int(test["class_counts"][str(i)]) for i in range(29)],
        "data_manifest_sha256": sha256(MANIFEST),
        "data_yaml_sha256": sha256(DATA),
        "pretrained_sha256": sha256(PRETRAINED),
        "stage2_epochs": STAGE2_EPOCHS,
        "config_sha256": {name: sha256(path) for name, path in CONFIGS.items()},
        "external_weight_sha256": {name: sha256(path) for name, path in EXTERNAL_WEIGHTS.items()},
    }


def main() -> None:
    import ultralytics

    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="preflight only, no training")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    args_cli = parser.parse_args()

    for directory in (ROOT, RUNS, EVALS, RESULTS):
        directory.mkdir(parents=True, exist_ok=True)
    audit = preflight()
    if args_cli.check:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return

    (ROOT / "pipeline.pid").write_text(str(os.getpid()), encoding="ascii")
    models = [name.strip() for name in args_cli.models.split(",") if name.strip()]
    state = {
        "started_at": now(),
        "status": "running",
        "seed": SEED,
        "models": models,
        "ultralytics_version": ultralytics.__version__,
        "ultralytics_source": str(ultralytics.__file__),
        "execution": (
            "single unified environment (ported ultralytics 8.3.253): one run per model at seed 404, "
            f"{EPOCHS} epochs / patience {PATIENCE}; SFCC adds a {STAGE2_EPOCHS}-epoch SFCA-only stage 2; "
            "TEST evaluated once per model"
        ),
        "preflight": audit,
        "current": None,
        "completed_models": [],
    }
    write_json(STATE, state)
    event(f"PREFLIGHT OK ultralytics={ultralytics.__version__} from {ultralytics.__file__}")

    try:
        for model_name in models:
            result_path = RESULTS / f"{model_name}_s{SEED}_complete.json"
            if result_path.exists():
                state["completed_models"].append(model_name)
                write_json(STATE, state)
                event(f"SKIP completed {model_name}")
                continue

            state["current"] = {"model": model_name, "phase": "train", "started_at": now()}
            write_json(STATE, state)
            if model_name == "sfcc":
                checkpoint = train_sfcc_stage2(train_sfcc_stage1())
            elif model_name == "c2fghost":
                checkpoint = train_single(model_name, CONFIGS["c2fghost"])
            elif model_name == "yolov8n":
                checkpoint = train_single(model_name, PRETRAINED)
            else:
                checkpoint = train_single(model_name, EXTERNAL_WEIGHTS[model_name])

            state["current"] = {"model": model_name, "phase": "test-eval", "started_at": now()}
            write_json(STATE, state)
            metrics = evaluate(model_name, checkpoint, audit["test_box_counts"])
            perf = complexity(checkpoint)
            speed = benchmark(checkpoint)

            payload = {
                "completed_at": now(),
                "model": model_name,
                "seed": SEED,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": metrics["checkpoint_sha256"],
                "epochs": STAGE2_EPOCHS if model_name == "sfcc" else EPOCHS,
                "patience": PATIENCE,
                **perf,
                **metrics,
                **speed,
            }
            write_json(result_path, payload)
            state["completed_models"].append(model_name)
            state["current"] = None
            state["last_completed"] = {
                "model": model_name,
                "map50_95": metrics["map50_95"],
                "tail_ap_evaluable": metrics["tail_ap_evaluable"],
            }
            write_json(STATE, state)
            event(
                f"COMPLETE {model_name} mAP50-95={metrics['map50_95']:.4f} "
                f"mAP50={metrics['map50']:.4f} tail={metrics['tail_ap_evaluable']:.4f} "
                f"params={perf['params_m']:.4f}M gflops={perf['gflops_640']:.2f}"
            )

        state["status"] = "completed"
        state["completed_at"] = now()
        state["current"] = None
        write_json(STATE, state)
        event("ALL UNIFIED RUNS COMPLETE")
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
