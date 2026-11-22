# Six-seed raw results

This directory contains the raw evaluation records and training logs for the six fixed
seeds used in the manuscript: 404, 101, 303, 2026, 3407 and 606. These experiments use
Ultralytics 8.1.0 and the fixed PlantDoc train/validation/test split.

| Path | Content |
|---|---|
| `raw_evaluation_json/` | All 36 evaluation records: three models, SFC on/off and six seeds. Each record includes overall metrics and per-class AP50-95. |
| `table_a_per_seed.csv` | Per-seed metrics for the six ablation configurations. |
| `table_a_summary.csv` and `table_a_summary.json` | Original mean and sample-standard-deviation summaries. |
| `tail_macro/` | Long-tailed macro-evaluation records. |
| `training_logs/` | `results.csv`, `args.yaml` and `training_complete.json` for the 12 primary YOLOv8n and SFCC runs, including both SFCC stages. |
| `SFCC_six_seed_verification.json` | Independent paired-statistics results and an inventory of the 12 primary checkpoints. |
| `SFCC_six_seed_paired_verification.csv` | Compact verification table with mean differences, confidence intervals, t statistics, raw and Holm-adjusted p values and exact Wilcoxon results. |
| `SHA256SUMS.txt` | SHA-256 hashes for the files in this directory. |

For manuscript Table 2, the primary comparison is unmodified YOLOv8n with SFC off versus
SFCC with SFC on, aligned by seed. Independent recomputation from the raw JSON records
reproduces the reported six metric means, sample standard deviations, 95% confidence
intervals and raw and Holm-adjusted p values. The JSON-to-CSV transcription check has zero
numeric discrepancy.

Checkpoint weights are stored separately in `../../reproducibility/checkpoints/`. The
verification inventory records their original locations and confirms that all 12 primary
checkpoints and all expected training logs existed when the archive was assembled on
20 November 2026.
