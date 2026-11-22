# SFCC-YOLOv8n reproducibility artifacts

This archive contains the materials underlying the six-seed PlantDoc results in
Tables 2, 3, 5 and 6 of the resubmitted manuscript.

## Scope

- Exact Ultralytics 8.1.0 source commit and the seven-file working-tree patch used
  for the six-seed experiments.
- Full versions of the seven patched source files.
- The two model configuration files used by the main experiments.
- The exact train, validation and test lists for the 2,119 / 236 / 237 split.
- Twelve evaluation result files: one YOLOv8n and one SFCC-YOLOv8n result for
  each of seeds 404, 101, 303, 2026, 3407 and 606.
- All 36 per-seed ablation result files, six configurations across six seeds,
  together with the per-seed and aggregate ablation tables.
- The twelve corresponding `best.pt` checkpoints.
- SHA-256 hashes for all checkpoints, metrics, split lists and source files.

## Environment

The main six-seed experiments used Python 3.10.20, PyTorch 2.7.1+cu128 and a
locally patched Ultralytics 8.1.0 source tree on Windows with an NVIDIA GeForce
RTX 5060 Laptop GPU (8 GB). The patched Ultralytics source commit is
`808984c6cf32f4ac9cb28f52fd74d13b9d6ad6a0`.

The public repository at
https://github.com/YongKang-Shi/SFCC-YOLOv8n/tree/66e02665fc48f779bae839facf0db8c765a46289
contains the separately reproduced comparison and deployment setup based on
Ultralytics 8.3.253. The two environments are reported separately in the
manuscript.

## Source application

Apply `source_patch/ultralytics_8.1.0.patch` to the source commit listed above,
or copy the seven files in `source_patch/modified_files/` to the same relative
locations. The full files are provided as a safeguard against line-ending or
context differences.

## Checkpoints and metrics

Each `main_results/<model>_seed<seed>.json` file records the metric values,
checkpoint path used in the local experiment, and checkpoint SHA-256. Each
`checkpoints/<model>_seed<seed>.best.pt` is the corresponding best checkpoint.
The manifest records hashes for the archived copies.

The PlantDoc images themselves are public and are not redistributed. Obtain the
official dataset and reconstruct the split using the included list files. The
FieldPlant external-validation labels, duplicate audit and summaries are
provided separately in the reviewer-revision supplement.

## Licences

The modified source remains subject to AGPL-3.0 and retains the original
Ultralytics notices. PlantDoc and FieldPlant retain their original dataset
licences as stated in the manuscript.
