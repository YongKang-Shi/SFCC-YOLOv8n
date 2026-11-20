SFCC six-seed raw results verified on 20 November 2026

Source: C:\Users\SYK\plantdoc-project\sfca-seed-experiments\final-20261004-ablation
Seeds: [404, 101, 303, 2026, 3407, 606]

Contents
- raw_evaluation_json/: all 36 evaluation JSON files (3 models x SFC on/off x 6 seeds).
- table_a_per_seed.csv: 36 per-seed rows transcribed from those JSON files.
- table_a_summary.csv and table_a_summary.json: original six-seed summaries.
- tail_macro/: original long-tail macro-evaluation files.
- training_logs/: results.csv, args.yaml and training_complete.json for the 12 primary checkpoints.
- SFCC_six_seed_verification.json: independent paired-statistics and checkpoint inventory.
- SFCC_six_seed_paired_verification.csv: compact paired-statistics table.

Verification
- All 36 raw JSON files are present and use exactly the manuscript seeds.
- Maximum JSON-to-CSV transcription error: 0.0.
- All 12 primary checkpoints exist: True.
- All expected primary training logs exist: True.
- Checkpoint weights are not included in this ZIP; their original paths are listed in the verification JSON.
- Source files were not modified.
