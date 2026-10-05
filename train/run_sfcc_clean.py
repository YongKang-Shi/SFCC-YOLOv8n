"""Uninterrupted (clean) SFCC seed-404 run under the ported Ultralytics 8.3.253 tree.

Identical protocol to final-20261004-unified-8.3.253 (same data split, same common_args,
same stage-1 300 epochs / patience 50 and 2-epoch SFCA-only stage 2, TEST scored once).
The only difference to the earlier SFCC entry is that this run is never interrupted, so it
tests whether the earlier kill+resume at epoch 45 depressed the result.
"""

from __future__ import annotations

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
ROOT = PROJECT / "sfca-seed-experiments" / "final-20261004-clean-sfcc-8.3.253"
DATA_ROOT = PROJECT / "plantdoc-github-official-yolo"
DATA = DATA_ROOT / "plantdoc_val.yaml"
MANIFEST = DATA_ROOT / "manifest.json"
PRETRAINED = PROJECT / "official-weights" / "yolov8n-v8.1.0-official.pt"
CONFIG = PROJECT / "sfca-seed-experiments" / "github-official-paired-e20" / "configs" / "yolov8n-sfca-c2fghost-plantdoc-github.yaml"

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


def common_args(name: str, epochs: int = EPOCHS, patience: int = PATIENCE) -> dict:
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
    for name, module in trainer.model.named_modules():
        if ".sfca" not in name and isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def train_stage1() -> Path:
    run_name = "sfcc_clean_s404_stage1"
    run_dir = RUNS / run_name
    marker = run_dir / "training_complete.json"
    best, last = run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    if marker.exists() and best.exists():
        event("SKIP completed clean stage1")
        return best

    model = None
    os.environ.pop("ULTRALYTICS_SFCA_ONLY", None)
    try:
        if not results_csv.exists() or result_rows(results_csv) == 0:
            event(f"START clean stage1 from {PRETRAINED.name}")
            set_seed(SEED)
            model = YOLO(str(CONFIG), task="detect")
            model.load(str(PRETRAINED))
            model.train(**common_args(run_name))
        else:  # only reachable if this clean run itself crashed
            event(f"RESUME clean stage1 rows={result_rows(results_csv)}")
            model = YOLO(str(last), task="detect")
            model.train(resume=True, workers=2, cache="disk", plots=False)
    finally:
        release_model(model)

    if not best.exists():
        best = last
    write_json(
        marker,
        {
            "completed_at": now(),
            "stage": 1,
            "seed": SEED,
            "pretrained_sha256": sha256(PRETRAINED),
            "actual_epochs": result_rows(results_csv),
            "best_training_row": best_training_row(results_csv),
            "checkpoint": str(best),
            "checkpoint_sha256": sha256(best),
        },
    )
    event(f"COMPLETE clean stage1 best={best_training_row(results_csv)}")
    return best


def train_stage2(stage1: Path) -> Path:
    run_name = "sfcc_clean_s404_stage2"
    run_dir = RUNS / run_name
    marker = run_dir / "training_complete.json"
    best, last = run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    if marker.exists() and best.exists():
        event("SKIP completed clean stage2")
        return best

    model = None
    previous = os.environ.get("ULTRALYTICS_SFCA_ONLY")
    os.environ["ULTRALYTICS_SFCA_ONLY"] = "1"
    try:
        if not results_csv.exists() or result_rows(results_csv) == 0:
            event(f"START clean stage2 ({STAGE2_EPOCHS} epochs, SFCA-only, non-SFCA BN frozen)")
            set_seed(SEED)
            model = YOLO(str(stage1), task="detect")
            model.add_callback("on_train_epoch_start", freeze_non_sfca_batchnorm)
            args = common_args(run_name, epochs=STAGE2_EPOCHS, patience=0)
            args["warmup_epochs"] = 0.0
            model.train(**args)
        else:
            event(f"RESUME clean stage2 rows={result_rows(results_csv)}")
            model = YOLO(str(last), task="detect")
            model.add_callback("on_train_epoch_start", freeze_non_sfca_batchnorm)
            model.train(resume=True, workers=2, cache="disk", plots=False)
    finally:
        if previous is None:
            os.environ.pop("ULTRALYTICS_SFCA_ONLY", None)
        else:
            os.environ["ULTRALYTICS_SFCA_ONLY"] = previous
        release_model(model)

    if not best.exists():
        best = last
    write_json(
        marker,
        {
            "completed_at": now(),
            "stage": 2,
            "seed": SEED,
            "epochs": STAGE2_EPOCHS,
            "warmup_epochs": 0.0,
            "sfca_only": True,
            "source_stage1": str(stage1),
            "actual_epochs": result_rows(results_csv),
            "best_training_row": best_training_row(results_csv),
            "checkpoint": str(best),
            "checkpoint_sha256": sha256(best),
        },
    )
    event(f"COMPLETE clean stage2 best={best_training_row(results_csv)}")
    return best


def evaluate(checkpoint: Path, test_box_counts: list[int]) -> dict:
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
            name="sfcc_clean_s404_test",
            exist_ok=True,
            verbose=False,
        )
        precision, recall = float(metrics.box.mp), float(metrics.box.mr)
        class_maps = [float(value) for value in metrics.box.maps]
        evaluable = [class_id for class_id in TAIL_CLASS_IDS if test_box_counts[class_id] > 0]
        return {
            "evaluated_at": now(),
            "model": "sfcc_clean",
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
        }
    except Exception as error:
        return {"benchmark_error": repr(error)}
    finally:
        release_model(model)


def preflight() -> dict:
    import ultralytics

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    test = manifest["splits"]["test"]
    return {
        "checked_at": now(),
        "ultralytics_version": ultralytics.__version__,
        "ultralytics_source": str(ultralytics.__file__),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "data_yaml_sha256": sha256(DATA),
        "test_box_counts": [int(test["class_counts"][str(i)]) for i in range(29)],
        "pretrained_sha256": sha256(PRETRAINED),
        "config_sha256": sha256(CONFIG),
    }


def main() -> None:
    import ultralytics

    for directory in (ROOT, RUNS, EVALS, RESULTS):
        directory.mkdir(parents=True, exist_ok=True)
    audit = preflight()
    result_path = RESULTS / "sfcc_clean_s404_complete.json"
    if result_path.exists():
        event("clean SFCC already complete; nothing to do")
        return

    (ROOT / "pipeline.pid").write_text(str(os.getpid()), encoding="ascii")
    state = {"started_at": now(), "status": "running", "audit": audit, "current": None}
    write_json(STATE, state)
    event(f"PREFLIGHT OK ultralytics={ultralytics.__version__} from {ultralytics.__file__}")
    try:
        stage1 = train_stage1()
        state["current"] = "stage2"
        write_json(STATE, state)
        stage2 = train_stage2(stage1)
        state["current"] = "test-eval"
        write_json(STATE, state)
        metrics = evaluate(stage2, audit["test_box_counts"])
        payload = {
            "completed_at": now(),
            "model": "sfcc_clean",
            "seed": SEED,
            "note": "uninterrupted run, identical protocol to the unified pipeline",
            **complexity(stage2),
            **metrics,
            **benchmark(stage2),
        }
        write_json(result_path, payload)
        state["status"] = "completed"
        state["completed_at"] = now()
        state["result"] = {k: payload[k] for k in ("map50_95", "map50", "tail_ap_evaluable", "params_m", "gflops_640")}
        write_json(STATE, state)
        event(
            f"COMPLETE clean SFCC mAP50-95={payload['map50_95']:.4f} mAP50={payload['map50']:.4f} "
            f"tail={payload['tail_ap_evaluable']:.4f} params={payload['params_m']:.4f}M"
        )
    except BaseException as error:
        state["status"] = "failed"
        state["error"] = repr(error)
        state["traceback"] = traceback.format_exc()
        write_json(STATE, state)
        event(f"FAILED {error!r}")
        raise


if __name__ == "__main__":
    main()
