# Reviewer Revision Data Supplement

This archive supports the six-seed class-confusion analysis and the FieldPlant
field evaluation reported in the revised manuscript.

The companion `../reproducibility/` directory contains the exact Ultralytics
8.1.0 patch, the per-seed main results, the twelve corresponding best
checkpoints and their SHA-256 manifest. The public repository also provides
the separately reproduced Ultralytics 8.3.253 code and comparison tables. The
two software environments are not mixed.

## External dataset

- Dataset: FieldPlant, public Roboflow Universe version 11.
- Paper DOI: https://doi.org/10.1109/ACCESS.2023.3263042
- Dataset page:
  https://universe.roboflow.com/plant-disease-detection/fieldplant/dataset/11
- Evaluation mirror used:
  https://www.kaggle.com/datasets/manhhoangvan/fieldplant
- Licence: CC BY 4.0.
- Public export: 5,156 field images and 8,580 manually annotated leaf boxes
  across 27 classes.
- Collection: field-plant images with individual leaf annotations supervised by
  plant pathologists.

Six FieldPlant classes map unambiguously to PlantDoc:

| FieldPlant class | PlantDoc target |
| --- | --- |
| Corn Gray leaf spot | Corn Gray leaf spot (class 7) |
| Corn leaf blight | Corn leaf blight (class 8) |
| Corn rust leaf | Corn rust leaf (class 9) |
| Tomato healthy | Tomato leaf (class 25) |
| Tomato leaf mosaic virus | Tomato leaf mosaic virus (class 23) |
| Tomato leaf yellow virus | Tomato leaf yellow virus (class 24) |

Images containing any unmapped class were excluded. To keep the external test
reproducible and computationally bounded, up to 200 images per source class
were retained in ascending SHA-256 order of the source filename. This produced
367 images and 553 boxes. Exact SHA-256 and perceptual-hash screening removed
one PlantDoc duplicate and five internal near duplicates, leaving 361 images
and 546 boxes.

Raw images are not redistributed. Download them from the official source or
mirror above and place the selected images under
`external_validation/images/test`. Remapped labels are included under
`external_validation/labels/test`.

## External evaluation protocol

The six YOLOv8n checkpoints and the six corresponding SFCC-YOLOv8n checkpoints
were evaluated without retraining or tuning. The protocol used 640 x 640 input,
batch size 32, confidence threshold 0.001, NMS IoU threshold 0.7 and at most
300 detections per image.

| Model | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| YOLOv8n | 22.74 +/- 3.23% | 27.62 +/- 3.97% | 21.67 +/- 2.90% | 15.29 +/- 2.05% |
| SFCC-YOLOv8n | 26.59 +/- 1.99% | 31.33 +/- 5.12% | 24.36 +/- 1.41% | 16.88 +/- 1.19% |
| Paired difference, SFCC minus YOLOv8n | +3.85 pp | +3.70 pp | +2.70 pp | +1.59 pp |

The 95% confidence intervals for the paired differences are:

- Precision: [+0.07, +7.63] percentage points.
- Recall: [-3.36, +10.77] percentage points.
- mAP50: [+0.42, +4.98] percentage points, with six of six seed wins.
- mAP50-95: [-0.04, +3.22] percentage points, with five of six seed wins.

The mAP50 interval excludes zero, whereas the mAP50-95 interval remains
marginally compatible with no difference. The result therefore provides partial
external support at IoU 0.5, but not a conclusive strict-localization gain.

At class level, mean mAP50-95 increased by 4.56 percentage points for Corn rust
leaf, 2.05 for Corn leaf blight and 2.24 for Tomato leaf, with changes of
`+0.64`, `+0.49` and `-0.42` percentage points for Corn Gray leaf spot, Tomato
leaf mosaic virus and Tomato leaf yellow virus. None of the per-class confidence
intervals excluded zero.

## Six-seed class confusion

Predictions were matched one-to-one to all 452 PlantDoc test boxes at IoU > 0.7.
The table below reports mean correct assignments and the most frequent wrong
class across the six seeds.

| Class | Test boxes | YOLOv8n correct | SFCC correct | Principal SFCC confusion |
| --- | ---: | ---: | ---: | --- |
| Tomato leaf bacterial spot | 14 | 5.0 +/- 1.7 | 4.0 +/- 1.7 | Septoria: 37 assignments over 6 runs |
| Tomato leaf late blight | 14 | 10.3 +/- 0.8 | 8.7 +/- 2.0 | Mold: 8 assignments over 6 runs |
| Tomato mold leaf | 16 | 8.3 +/- 3.2 | 6.5 +/- 1.2 | Mosaic virus: 16 assignments over 6 runs |
| Strawberry leaf | 30 | 30.0 +/- 0.0 | 29.7 +/- 0.8 | Raspberry leaf: 1 assignment over 6 runs |
| Tomato leaf | 27 | 13.8 +/- 2.2 | 13.0 +/- 3.2 | Mosaic virus: 18 assignments over 6 runs |

## Contents

- `external_validation/`: class mapping, deterministic selection manifest,
  duplicate audit, portable remapped labels, data configuration, per-seed box
  metrics and aggregate results.
- `six_seed_confusion/`: aggregate and per-seed PlantDoc confusion summaries
  and full 30 x 30 count matrices.
- `scripts/`: deterministic FieldPlant selection, range extraction, duplicate
  filtering, external evaluation and PlantDoc confusion analysis.

The companion `../reproducibility/` directory contains the materials for the
main six-seed PlantDoc tables. The public images are not redistributed in
either directory.

## Reproduction notes

The scripts preserve the exact processing logic used for this revision.
Workspace-specific paths may need to be changed for another environment.
Raw PlantDoc and FieldPlant images are not redistributed here. Obtain them from
their original repositories and cite the source publications.
