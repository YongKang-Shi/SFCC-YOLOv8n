# Ultralytics patch files

These seven files come from Ultralytics **8.3.253** and are the files we modified. Together
they implement everything that is new in SFCC-YOLOv8n:

| File | Change |
|---|---|
| `nn/modules/block.py` | `C2fGhost`, `GhostBottleneck` and the SGCRA module (support-guided class response attention) |
| `nn/modules/head.py` | `DetectSFC` — the detection head with parameter-free support frequency calibration (SFC) applied to the class logits |
| `nn/modules/__init__.py` | exports the new modules |
| `nn/tasks.py` | registers the new module names so they can be parsed from the YAML configuration |
| `utils/loss.py` | the auxiliary class-response supervision term of Equation (5d), added to the detection loss with a weight of 0.05 |
| `engine/trainer.py` | the stage-2 `SFCA_ONLY` mode: every parameter whose name does not contain `.sfca.` is frozen, so only the four SGCRA/SFCA nodes are fine-tuned |
| `data/build.py` | deterministic shuffling controlled by the epoch (`set_epoch()` / `seed_worker()`), which makes repeated runs reproducible |

## How to apply

Copy each file over the same relative path inside your installed `ultralytics` package
(see step 2 of the top-level `README.md`). After patching, `configs/` can be used directly with
Ultralytics training commands.

## License

Ultralytics is licensed under AGPL-3.0. Because these files are modified copies of its source,
they remain under AGPL-3.0 (see `LICENSE` in the repository root). The original copyright and
license notices inside the files are kept unchanged.
