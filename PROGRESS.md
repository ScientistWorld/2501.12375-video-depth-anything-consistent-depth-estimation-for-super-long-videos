# Progress

## What Works

- Read the paper and identified the core reproducible claim: long-video inference improves temporal consistency over independent short-window processing while keeping the same model scale.
- Vendored the official Video Depth Anything implementation in `method/official`.
- Downloaded the released VDA-S checkpoint to `data/downloads/video_depth_anything_vits.pth`.
- Added a reproducible synthetic RGB-D long-video generator in `data/synthetic_rgbd.py`.
- Added method and baseline runners that write prediction artifacts for the evaluator.
- Added an artifact-only evaluator in `eval/metrics.py`.

## Results

No compute-node results yet. The first submitted job will run:

- `vda_independent_windows`
- `vda_oi_kr`

on the fixed synthetic long-video benchmark and write `scoring/scores.json`.

## Remaining

- Run the submitted GPU job and inspect `scoring/scores.json`.
- If the long-video strategy does not improve temporal consistency, diagnose whether the synthetic sequence is too easy, too short, or mismatched to the checkpoint and adjust the data scale without changing the method.
- After a successful run, update `MILESTONES.md` and compare against `scoring/reference.json`.

## Issues

- Local login-node Python did not include `imageio`; the synthetic data generator now treats MP4 writing as optional. The container installs `imageio` for normal runs.

## Deviations from Paper

- The paper evaluates on KITTI, ScanNet, Bonn, NYUv2, and Sintel. This first reproduction uses a small synthetic RGB-D video because the full benchmark suite is large and the minimum core claim can be tested on a fixed long sequence with known ground truth.
- The paper trains VDA on 0.55M video frames plus image data. This workspace uses the released VDA-S checkpoint for the first reproduction rather than retraining, preserving the paper's algorithm and model family while fitting the available compute budget.
- The evaluator uses `temporal_gradient_error` on ground-truth stable regions as the local temporal consistency metric. `scoring/reference.json` records the paper's KITTI temporal alignment error values as the external reference for the same claimed effect.
