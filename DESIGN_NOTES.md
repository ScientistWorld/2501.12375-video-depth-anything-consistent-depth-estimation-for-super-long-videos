# Train/Test Evaluation Design Notes (iteration 11)

- **Scenes.** Procedural static corridor videos (floor, ceiling, walls, partition walls with doorways, furniture boxes) rendered by exact ray casting at 144x192, 12 fps. A moving, turning, tilting camera keeps changing the visible depth range. Z-depth, intrinsics and camera-to-world poses are exact.
- **Train (visible).** `eval/train/evaluate.py` renders seeds 101–104 (500 frames) and 201–202 (1500 frames) with `data/synthetic_rgbd.py`.
- **Test (hidden).** `eval/test/evaluate.py` renders 6×500 + 2×1500 frames from 63-bit seeds with its own renderer copy (`eval/test/scenes.py`). Labels stay in memory; only RGB frames go to a scratch directory, which is deleted afterwards.
- **Predictor contract.** Manifests in `method/predictors/` and `baseline/predictors/` hold a `command`. Each evaluator runs `command --jobs jobs.json` once per predictor and reads `prediction` `[T,H,W]` arrays. The same contract applies on train and test, and no method name is hardcoded.
- **Metrics.** Per video, one least-squares scale/shift aligns predicted inverse depth to ground-truth inverse depth (depth in [0.2, 20]). The metrics are absrel, delta1, the paper's TAE (×100, true poses, nearest-pixel reprojection with a z-buffer), and scale_drift (std of per-frame median log ratio, ×100). Each is averaged over the videos of an experiment.
- **Gate.** Missing, misshaped or non-finite output, or a predictor error, fails the run. So does failing to beat the static per-pixel template on absrel and delta1, or reaching TAE 30 or drift 50. A failed run gets the gate values on every metric, and the reason goes under `_status`.
- **Outputs.** Score files are always rewritten from scratch. Neither evaluator reads the other's score files, sidecars, or anything in `data/` apart from the train renderer.

Reasons for each choice and the audit numbers are in `HARDENING.md`.
