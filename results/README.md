# Results

This folder contains the result files that support the manuscript tables. The six-seed
primary comparison and ablation were evaluated under Ultralytics 8.1.0. The same-setting
detector comparison and deployment measurements were produced under Ultralytics 8.3.253
with random seed 404 and identical training and evaluation settings.

| File | Content |
|---|---|
| `same_setting_comparison.csv` | Same-setting comparison of the five detectors (manuscript Table 4 / Table 7 input): parameters, GFLOPs, mAP50, mAP50-95 and long-tailed class AP |
| `deployment_table.md` | Deployment and efficiency measurements on the same hardware and protocol (manuscript Table 7) |
| `six_seed/` | Raw per-seed evaluation JSON files, per-seed and summary CSV files, primary-run training logs and independent statistical verification for manuscript Tables 2, 3, 5 and 6 |

The per-run result files of the six-seed study are available in `six_seed/`. The twelve
corresponding best checkpoints are stored in `../reproducibility/checkpoints/`, with their
SHA-256 hashes and source-environment records in the same package.
