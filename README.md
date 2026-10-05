# SFCC-YOLOv8n

Code, model configuration and dataset splits for **SFCC-YOLOv8n**, a support-frequency-coupled
lightweight detector for crop disease detection on low-power field devices. All results reported
in the manuscript were obtained on the 29-class PlantDoc benchmark.

The trained model weights are not stored in this repository; they are available from the
corresponding author on reasonable request.

## Contents

| Path | What it is |
|---|---|
| `configs/` | Model definitions: `yolov8n-sfca-c2fghost-plantdoc-github.yaml` (SFCC-YOLOv8n: C2fGhost neck + SGCRA + DetectSFC) and `yolov8n-c2fghost-plantdoc-github.yaml` (C2fGhost variant used in the ablation) |
| `src/ultralytics_patch/` | The seven Ultralytics 8.3.253 files that we modified: C2fGhost and SGCRA (SFCA) in `nn/modules/block.py`, the SFC detection head in `nn/modules/head.py`, module registration in `nn/modules/__init__.py` and `nn/tasks.py`, the auxiliary response loss (weight 0.05) in `utils/loss.py`, the stage-2 SFCA-only freezing in `engine/trainer.py`, and deterministic shuffling in `data/build.py` |
| `splits/` | `plantdoc_val.yaml` (29 classes, train/val/test paths), the exact `train_list.txt` / `val_list.txt` / `test_list.txt` used in the paper (2119 / 236 / 237 images), and `split_verification.txt` (class-wise counts, zero overlap between the three splits, random seed 20261002) |
| `results/` | Result files behind the comparison and deployment tables (`same_setting_comparison.csv`, `deployment_table.md`) |
| `train/` | Entry points: `run_sfcc_revision_six.py` and `run_yolov8n_revision_six.py` (six-seed study), `run_unified.py` (same-setting comparison of five detectors, seed 404), `run_sfcc_clean.py` (two-stage SFCC training) |

## Paper-to-code naming

The module names in the released code are the ones used during development; they differ
from the names used in the manuscript. The mapping is:

| Manuscript (Applied Sciences submission) | Code |
|---|---|
| SGCRA (support-guided class response attention) | `SFCA` class in `src/ultralytics_patch/nn/modules/block.py` |
| the neck block "Concat - SGCRA - C2fGhost" | `SFCAC2fGhost` in the same file, used by `configs/yolov8n-sfca-c2fghost-plantdoc-github.yaml` |
| SFC (support frequency calibration) applied after the detection head | `DetectSFC` in `src/ultralytics_patch/nn/modules/head.py` |
| C2fGhost neck block | `C2fGhost` in `src/ultralytics_patch/nn/modules/block.py` |

Only the names differ: `SFCA`, `SFCAC2fGhost` and `DetectSFC` are exactly the modules
described as SGCRA, the SGCRA-C2fGhost neck block and the SFC detection head in the paper.

## Environment

- Python 3.10.20, PyTorch 2.7.1+cu128, Ultralytics 8.3.253
- Single NVIDIA GeForce RTX 5060 Laptop GPU (8 GB), Windows
- Evaluation protocol used throughout: `conf = 0.001`, NMS `IoU = 0.7`, `max_det = 300`

## Reproducing the results in three steps

**1. Install the exact Ultralytics version**

```bash
pip install ultralytics==8.3.253 torch==2.7.1 torchvision
```

**2. Apply our modules on top of the installed Ultralytics tree**

```bash
python - <<'PY'
import pathlib, shutil, ultralytics

root = pathlib.Path(ultralytics.__file__).parent
here = pathlib.Path("src/ultralytics_patch")
for rel in (
    "nn/tasks.py",
    "nn/modules/__init__.py",
    "nn/modules/block.py",
    "nn/modules/head.py",
    "utils/loss.py",
    "engine/trainer.py",
    "data/build.py",
):
    shutil.copyfile(here / rel, root / rel)
    print("patched", root / rel)
PY
```

**3. Train / evaluate**

```bash
python train/run_sfcc_revision_six.py    # six-seed study (main result)
python train/run_unified.py              # same-setting comparison, seed 404
python train/run_sfcc_clean.py           # two-stage SFCC training
```

The scripts contain absolute paths (`PROJECT`, `ROOT`, `DATA`) pointing to the authors' machine.
Adjust those constants to your own layout before running; everything else (hyper-parameters,
seeds, augmentation, evaluation settings) is already fixed in the files.

## Key results

Six fixed random seeds (**404, 101, 303, 2026, 3407 and 606**, as reported in the paper),
PlantDoc test set scored once:

| Model | Params (M) | GFLOPs | mAP50-95 (mean ± SD) |
|---|---|---|---|
| YOLOv8n | 3.017 | 8.22 | 0.4663 ± 0.0150 |
| **SFCC-YOLOv8n** | **2.594** | **7.40** | **0.4892 ± 0.0074** |

Same-setting comparison (identical environment, training and evaluation settings, seed 404,
mAP50-95): SFCC-YOLOv8n 0.4914, YOLOv8n 0.4914, YOLOv10n 0.4772, YOLO11n 0.4492, YOLOv5nu 0.4428.
Deployment: 18.7 ms (53.5 FPS) per image in ONNX Runtime on CPU, and 10.6 ms (94.4 FPS) in FP16
on a laptop GPU.

## Dataset

PlantDoc is publicly available and is **not** redistributed here:
<https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset>. Only our own split lists are
provided, so that the exact train/validation/test partition of the paper can be rebuilt.

## License

This work modifies Ultralytics YOLO, which is licensed under **AGPL-3.0**; this repository is
therefore released under the same license (see `LICENSE`). Original Ultralytics copyright notices
are retained inside the patched files.

## Citation

If you use this code, please cite the accompanying paper (see `CITATION.cff`).

## Contact

Yongkang Shi, Wenchao Gao (corresponding author, gaowc@cumtb.edu.cn), Sen Song —
School of Artificial Intelligence, China University of Mining and Technology (Beijing),
Beijing 100083, China.

## Dataset paths

The dataset itself (PlantDoc converted to YOLO format) is **not** distributed here.
`splits/*_list.txt` therefore contain image paths **relative to the dataset root**
(for example `images/train/train_00001.jpg`). Set the `path:` entry of
`splits/plantdoc_val.yaml` to your local dataset root before training.
The training scripts additionally contain a `PROJECT` constant that points to the authors'
machine; adjust it in the same way.
