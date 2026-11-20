# Evaluation

## Splits

Both splits are clips of the same two public benchmarks with the same lengths, metrics, alignment, gate and predictor contract. Sequences and drives do not overlap between them.

- **Train (visible).** `data/train_clips.json`: 6 Bonn RGB-D Dynamic sequences (first 500 frames), 4 KITTI drives (500 frames) and 2 KITTI drives (1500 frames). `scripts/download.sh` fetches their frames, depth maps, camera poses / GPS-IMU records and calibration into `data/downloads/`. `eval/train/evaluate.py` scores every predictor on them.
- **Test (hidden).** 12 other Bonn sequences (first 500 frames), 10 other KITTI drives (500 frames) and 5 clips of 3 other KITTI drives (1500 frames). They live only in `eval/test/`. Your predictor receives only their RGB frames, under opaque ids.

## Predictor contract

A method is scored by running it, so register it as a predictor: a JSON manifest at `method/predictors/<name>.json` (or `baseline/predictors/<name>.json`):

```json
{"command": ["python", "method/my_predictor.py", "--some-flag"]}
```

The evaluator runs `command + ["--jobs", "<path>/jobs.json"]` once per evaluation (all videos in one call, so a model is loaded once), with the workspace root as working directory. `jobs.json` is:

```json
{"videos": [{"id": "video_000", "frames": ["/abs/.../000000.png", "..."], "num_frames": 500, "fps": 30,
             "height": 480, "width": 640, "stride": 2, "out_height": 240, "out_width": 320,
             "output": "/abs/path/video_000.npz"}, ...]}
```

- `frames` is the ordered list of the video's RGB frames: 8-bit PNG files at native resolution, `height` x `width`. Frame sizes differ between videos.
- For every video, write `output` as an `.npz` holding `prediction`: a real-valued array `[num_frames, out_height, out_width]` of relative inverse depth. It can have any scale and shift; larger means closer.
- Predictions are evaluated on every `stride`-th row and column of the native frame. `prediction[t, i, j]` is the value for native pixel `(stride*i, stride*j)`, so a full-resolution map `D` is submitted as `D[:, ::stride, ::stride]`.
- Every predictor on disk is scored, each independently. The manifest name is the method name in the score files.

Rules:

- The predictor must be inference only: train beforehand and load your weights.
- It must run offline.
- It may read only the frames it is given and your own files. Reading labels, poses, evaluator files or anything under `eval/` is not allowed, and neither is looking the frames up in a copy of the benchmarks.
- On the hidden test, where the platform allows it, the predictor runs in a private mount/PID namespace in which `eval/test/` is empty.
- No wall-clock limit is imposed. The reference takes a few minutes per split on one A100, and the test node has a GPU.

## Metrics

Per video, the official benchmark protocol is used:

1. The prediction, clipped at 1e-3, is aligned to ground-truth inverse depth with one least-squares scale and shift shared by all frames of the video.
2. The fit uses pixels whose true depth is in (1e-3, max_depth). max_depth is 10 m for Bonn and 80 m for KITTI.
3. Aligned inverse depth is clipped at 1e-3, and depth at max_depth.

Each metric is averaged over the videos of an experiment.

- `absrel` (lower is better): mean |d - d*| / d* over the frame's evaluated pixels, then averaged over frames.
- `delta1` (higher is better): fraction of the frame's evaluated pixels with max(d/d*, d*/d) < 1.25, then averaged over frames.
- `tae` (lower is better): Temporal Alignment Error, x100. The aligned depth of every pixel of a frame is back-projected with the true intrinsics, moved with the true relative camera motion into the next frame and projected (nearest pixel; the closest surface wins). It is then compared with the next frame's aligned depth by AbsRel. The result is averaged over all consecutive pairs and both directions. Moving people and cars move independently of the camera, so even perfect depth does not reach 0.
- `scale_drift` (lower is better): standard deviation over frames (x100) of the per-frame median of log(aligned depth / true depth). Both scale jumps between parts of the video and slow drift raise it.

## Validity gate

A run fails in either of these cases:

- Any video's output is missing, has the wrong shape or contains NaN/inf, or the predictor exits non-zero.
- In any experiment:
  - its mean `absrel` is not below the level of the best image-agnostic answer, or its mean `delta1` is not above it. That answer is a static per-pixel depth template fitted to the test labels. Its levels are listed in `scoring/metrics_detail_test.json`.
  - its mean `tae` reaches 30, or its `scale_drift` reaches 50.

A failed run receives those level values as its metrics in every experiment, which is worse than any passing run on every metric. The reason is recorded under `_status` in the score file.

## Ranking

`scoring/reference.json` gives each experiment's weight:

| experiment | weight |
|---|---:|
| `bonn_500` | 0.35 |
| `kitti_500` | 0.35 |
| `kitti_1500` | 0.30 |

It also gives each metric's coefficient and direction: absrel 1, delta1 1, tae 1, scale_drift 0.5. Improvement is measured against the frozen reference scores in `scoring/scores_test.json`.

Current hidden-test scores are in `scoring/scores_test.json`; a summary is in `scoring/TARGETS.md`. The baseline (`vda_independent_windows`) uses the same released network but processes each 32-frame window independently and concatenates the results.

## Commands

```bash
bash scripts/evaluate_train.sh                 # all predictors on the train split -> scoring/scores_train.json
bash scripts/evaluate_train.sh --only NAME     # one predictor
bash scripts/evaluate.sh                       # same as train, written to scoring/scores.json
bash scripts/evaluate_test.sh                  # hidden test (no arguments) -> scoring/scores_test.json
```

Score files are always rewritten from scratch. Per-video numbers are in `scoring/metrics_detail_*.json`.
