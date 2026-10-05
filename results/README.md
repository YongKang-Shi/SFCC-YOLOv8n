# Results

This folder contains the result files that support the comparison and deployment tables
of the manuscript. All values were produced by the scripts in `train/` under the protocol
described in the paper (Ultralytics 8.3.253, identical training and evaluation settings,
random seed 404, official PlantDoc test split scored once).

| File | Content |
|---|---|
| `same_setting_comparison.csv` | Same-setting comparison of the five detectors (manuscript Table 4 / Table 7 input): parameters, GFLOPs, mAP50, mAP50-95 and long-tailed class AP |
| `deployment_table.md` | Deployment and efficiency measurements on the same hardware and protocol (manuscript Table 7) |

The per-run result files of the six-seed study and the trained model weights are not stored
in this repository; they are available from the corresponding author on reasonable request.
