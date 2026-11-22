from __future__ import annotations

import csv
import gc
import json
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


DATA_ROOT = ROOT / "plantdoc-github-official-yolo"
OUT = Path(r"C:\Users\SYK\Documents\Codex\2026-10-30\bi-a\work\confusion-six-seed")

SEEDS = [404, 101, 303, 2026, 3407, 606]
MODELS = ["yolov8n", "sfcc"]
TARGET_CLASSES = [21, 22, 26, 17, 25]

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


def load_test_set() -> tuple[list[Path], list[np.ndarray]]:
    image_paths = [
        Path(line.strip())
        for line in (DATA_ROOT / "test_list.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels: list[np.ndarray] = []
    for image_path in image_paths:
        label_path = DATA_ROOT / "labels" / "test" / f"{image_path.stem}.txt"
        rows = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                values = [float(value) for value in line.split()]
                if len(values) != 5:
                    continue
                class_id, xc, yc, width, height = values
                rows.append(
                    [
                        int(class_id),
                        xc - width / 2,
                        yc - height / 2,
                        xc + width / 2,
                        yc + height / 2,
                    ]
                )
        labels.append(np.asarray(rows, dtype=np.float32).reshape(-1, 5))
    return image_paths, labels


def iou_matrix(gt_boxes: np.ndarray, predicted_boxes: np.ndarray) -> np.ndarray:
    if len(gt_boxes) == 0 or len(predicted_boxes) == 0:
        return np.zeros((len(gt_boxes), len(predicted_boxes)), dtype=np.float32)
    top_left = np.maximum(gt_boxes[:, None, :2], predicted_boxes[None, :, :2])
    bottom_right = np.minimum(gt_boxes[:, None, 2:], predicted_boxes[None, :, 2:])
    intersection = np.prod(np.clip(bottom_right - top_left, 0, None), axis=2)
    gt_area = np.prod(np.clip(gt_boxes[:, 2:] - gt_boxes[:, :2], 0, None), axis=1)
    predicted_area = np.prod(
        np.clip(predicted_boxes[:, 2:] - predicted_boxes[:, :2], 0, None), axis=1
    )
    union = gt_area[:, None] + predicted_area[None, :] - intersection
    return intersection / np.clip(union, 1e-9, None)


def update_matrix(
    matrix: np.ndarray,
    ground_truth: np.ndarray,
    result,
) -> None:
    height, width = result.orig_shape
    if result.boxes is None or len(result.boxes) == 0:
        predicted_classes = np.empty(0, dtype=np.int64)
        predicted_boxes = np.empty((0, 4), dtype=np.float32)
    else:
        predicted_classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
        xyxy = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        predicted_boxes = xyxy / np.asarray(
            [width, height, width, height], dtype=np.float32
        )

    gt_classes = ground_truth[:, 0].astype(np.int64)
    gt_boxes = ground_truth[:, 1:5]
    if len(gt_boxes) == 0:
        for predicted_class in predicted_classes:
            matrix[predicted_class, len(CLASS_NAMES)] += 1
        return

    ious = iou_matrix(gt_boxes, predicted_boxes)
    rows, columns = np.where(ious > 0.7)
    matches = np.column_stack((rows, columns, ious[rows, columns]))
    if len(matches):
        matches = matches[np.argsort(matches[:, 2])[::-1]]
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
        matches = matches[np.argsort(matches[:, 2])[::-1]]
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

    matched_detection_ids: set[int] = set()
    for gt_index, ground_truth_class in enumerate(gt_classes):
        pair = matches[matches[:, 0] == gt_index]
        if len(pair) == 1:
            detection_index = int(pair[0, 1])
            matrix[predicted_classes[detection_index], ground_truth_class] += 1
            matched_detection_ids.add(detection_index)
        else:
            matrix[len(CLASS_NAMES), ground_truth_class] += 1

    for detection_index, predicted_class in enumerate(predicted_classes):
        if detection_index not in matched_detection_ids:
            matrix[predicted_class, len(CLASS_NAMES)] += 1


def summarize_matrix(matrix: np.ndarray, test_counts: list[int], case_name: str) -> dict:
    result = {
        "case": case_name,
        "test_ground_truth_count": int(matrix[:, : len(CLASS_NAMES)].sum()),
        "target_classes": {},
    }
    for class_id in TARGET_CLASSES:
        total = int(test_counts[class_id])
        correct = float(matrix[class_id, class_id])
        background = float(matrix[len(CLASS_NAMES), class_id])
        confusion_counts = {
            CLASS_NAMES[predicted_class]: float(matrix[predicted_class, class_id])
            for predicted_class in range(len(CLASS_NAMES))
            if predicted_class != class_id
            and matrix[predicted_class, class_id] > 0
        }
        top_class, top_count = max(
            confusion_counts.items(),
            key=lambda item: item[1],
            default=("None", 0.0),
        )
        result["target_classes"][str(class_id)] = {
            "class_name": CLASS_NAMES[class_id],
            "test_boxes": total,
            "correct_count": correct,
            "correct_rate": correct / total,
            "background_count": background,
            "background_rate": background / total,
            "top_confusion_class": top_class,
            "top_confusion_count": top_count,
            "top_confusion_rate": top_count / total,
            "confusion_counts": confusion_counts,
        }
    return result


def write_matrix(matrix: np.ndarray, output_path: Path) -> None:
    names = [*CLASS_NAMES, "background"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prediction\\ground_truth", *names])
        for predicted_class, class_name in enumerate(names):
            writer.writerow([class_name, *matrix[predicted_class].tolist()])


def aggregate(case_results: dict[str, dict]) -> dict:
    aggregate_result = {}
    for model_name in MODELS:
        model_cases = [
            case_results[f"{model_name}_seed{seed}"] for seed in SEEDS
        ]
        model_result = {}
        for class_id in TARGET_CLASSES:
            class_key = str(class_id)
            entries = [case["target_classes"][class_key] for case in model_cases]
            correct_rates = [entry["correct_rate"] for entry in entries]
            background_rates = [entry["background_rate"] for entry in entries]
            confusion_totals: dict[str, float] = {}
            for entry in entries:
                for class_name, count in entry["confusion_counts"].items():
                    confusion_totals[class_name] = (
                        confusion_totals.get(class_name, 0.0) + count
                    )
            top_class, top_total = max(
                confusion_totals.items(),
                key=lambda item: item[1],
                default=("None", 0.0),
            )
            test_boxes = entries[0]["test_boxes"]
            model_result[class_key] = {
                "class_name": entries[0]["class_name"],
                "test_boxes": test_boxes,
                "correct_count_mean": statistics.mean(
                    entry["correct_count"] for entry in entries
                ),
                "correct_count_sd": statistics.stdev(
                    entry["correct_count"] for entry in entries
                ),
                "correct_rate_mean": statistics.mean(correct_rates),
                "correct_rate_sd": statistics.stdev(correct_rates),
                "background_count_mean": statistics.mean(
                    entry["background_count"] for entry in entries
                ),
                "background_rate_mean": statistics.mean(background_rates),
                "background_rate_sd": statistics.stdev(background_rates),
                "top_confusion_class": top_class,
                "top_confusion_count_total": top_total,
                "top_confusion_count_mean": top_total / len(SEEDS),
                "top_confusion_rate_mean": top_total / (len(SEEDS) * test_boxes),
            }
        aggregate_result[model_name] = model_result
    return aggregate_result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    image_paths, labels = load_test_set()
    test_counts = [0] * len(CLASS_NAMES)
    for ground_truth in labels:
        for class_id in ground_truth[:, 0].astype(int):
            test_counts[class_id] += 1
    if len(image_paths) != 237 or sum(test_counts) != 452:
        raise RuntimeError(
            f"Unexpected test set: images={len(image_paths)}, boxes={sum(test_counts)}"
        )

    case_results: dict[str, dict] = {}
    for model_name in MODELS:
        for seed in SEEDS:
            case_name = f"{model_name}_seed{seed}"
            checkpoint = checkpoint_for(seed, model_name)
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            print(f"START {case_name}", flush=True)
            matrix = np.zeros(
                (len(CLASS_NAMES) + 1, len(CLASS_NAMES) + 1), dtype=np.float64
            )
            model = YOLO(str(checkpoint), task="detect")
            try:
                predictions = model.predict(
                    source=[str(path) for path in image_paths],
                    imgsz=640,
                    batch=32,
                    device=0,
                    conf=0.001,
                    iou=0.7,
                    max_det=300,
                    stream=True,
                    save=False,
                    verbose=False,
                )
                for image_index, result in enumerate(predictions):
                    update_matrix(matrix, labels[image_index], result)
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()

            if int(matrix[:, : len(CLASS_NAMES)].sum()) != 452:
                raise RuntimeError(
                    f"{case_name} assigned "
                    f"{int(matrix[:, : len(CLASS_NAMES)].sum())} "
                    "ground-truth boxes; expected 452"
                )
            case_dir = OUT / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            write_matrix(matrix, case_dir / "confusion_matrix_exact.csv")
            summary = summarize_matrix(matrix, test_counts, case_name)
            (case_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            case_results[case_name] = summary
            print(
                json.dumps(
                    {
                        "case": case_name,
                        "ground_truth": summary["test_ground_truth_count"],
                        "bacterial_spot_correct": summary["target_classes"]["21"][
                            "correct_count"
                        ],
                        "late_blight_correct": summary["target_classes"]["22"][
                            "correct_count"
                        ],
                    }
                ),
                flush=True,
            )

    aggregate_result = aggregate(case_results)
    (OUT / "six_seed_aggregate.json").write_text(
        json.dumps(aggregate_result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(aggregate_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
