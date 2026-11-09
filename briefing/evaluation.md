# Evaluation

## Splits

- **Train (visible).** `eval/train/` renders 4 videos of 500 frames (seeds 101-104) and 2 videos of 1500 frames (seeds 201-202) with `data/synthetic_rgbd.py`. You can render as many further videos as you like with that module (any other seed); each comes with RGB frames, z-depth, intrinsics and camera poses (`python data/synthetic_rgbd.py --seed S --frames N --output DIR`).
- **Test (hidden).** Same scene family, different scenes: 6 videos of 500 frames and 2 videos of 1500 frames. They are rendered inside the test evaluator at test time; only their RGB frames are handed to your predictor. Labels never touch the disk.

Both splits use the same experiments, metrics, alignment, gate and predictor contract.

## Predictor contract

A method is scored by running it, so register it as a predictor: a JSON manifest at `method/predictors/<name>.json` (or `baseline/predictors/<name>.json`):

```json
{"command": ["python", "method/my_predictor.py", "--some-flag"]}
```

The evaluator runs `command + ["--jobs", "<path>/jobs.json"]` with the workspace root as working directory, once per evaluation (all videos in one call, so a model is loaded once). `jobs.json` is

```json
{"videos": [{"id": "video_000", "frames": "/abs/path/video_000.npy", "fps": 12,
             "num_frames": 500, "height": 144, "width": 192,
             "output": "/abs/path/video_000.npz"}, ...]}
```

`frames` is a uint8 RGB array `[T, H, W, 3]`. For every video write `output` as an `.npz` holding `prediction`: a real-valued array of shape `[T, H, W]` with relative inverse depth (any scale and shift; larger = closer). Every predictor on disk is scored, each independently; the name of the manifest is the method name in the score files.

Rules: the predictor must be inference only (train beforehand and load your weights), must run offline, must finish within 4 hours for the whole test set (the reference takes about 6 minutes on one A100; the test node has a GPU), and may read only the frames it is given and your own files. Reading labels, evaluator files or anything under `eval/` from a predictor is not allowed.

## Metrics

Per video, the prediction is aligned to ground truth with one least-squares scale and shift in inverse depth shared by all frames of the video, using pixels whose true depth is in [0.2, 20]; aligned inverse depth is clipped at 1e-3 and depth at 20. Metrics are averaged over the videos of an experiment.

- `absrel` (lower is better): mean |d - d*| / d* over evaluated pixels.
- `delta1` (higher is better): fraction of evaluated pixels with max(d/d*, d*/d) < 1.25.
- `tae` (lower is better): Temporal Alignment Error, x100. The aligned depth of each frame is back-projected with the true intrinsics, moved with the true relative camera motion into the next frame, projected (nearest pixel, closest surface wins), and compared with the next frame's aligned depth by AbsRel; averaged over all consecutive pairs and both directions. Exact ground truth scores about 0.4 here (pixel rounding at depth edges).
- `scale_drift` (lower is better): standard deviation over frames (x100) of the per-frame median of log(aligned depth / true depth). Scale jumps between parts of the video and slow drift both raise it.

## Validity gate

A run is failed if any video's output is missing, has the wrong shape, contains NaN/inf, or if the predictor exits non-zero; or if, in any experiment, its mean `absrel` is not below and its mean `delta1` not above the level of the best image-agnostic answer (a static per-pixel depth template fitted to the test labels: about 0.39 / 0.35), or its mean `tae` reaches 30 or `scale_drift` reaches 50. A failed run receives, in every experiment, those level values as its metrics (worse than any passing run on every metric), and the reason is recorded under `_status` in the score file.

## Ranking

`scoring/reference.json` gives each experiment's weight (0.6 for `long_video_500`, 0.4 for `super_long_video_1500`) and each metric's coefficient (absrel 1, delta1 1, tae 1, scale_drift 0.5) and direction. Improvement is measured against the frozen reference scores in `scoring/scores_test.json`.

Current hidden-test scores:

| experiment | method | absrel | delta1 | tae | scale_drift |
|---|---|---:|---:|---:|---:|
| long_video_500 | reference (`vda_oi_kr`) | 0.1504 | 0.8146 | 1.173 | 12.96 |
| long_video_500 | independent-window baseline | 0.1734 | 0.7511 | 1.580 | 15.86 |
| super_long_video_1500 | reference (`vda_oi_kr`) | 0.1274 | 0.8616 | 1.377 | 10.10 |
| super_long_video_1500 | independent-window baseline | 0.1585 | 0.8024 | 1.782 | 13.56 |

The baseline uses the same released network but processes each 32-frame window independently and concatenates the results. For orientation, the paper reports AbsRel / delta1 / TAE of 0.157 / 0.826 / 0.874 for independent windows and 0.144 / 0.851 / 0.718 for its method on real benchmarks (<= 500 frames); those numbers are on different data and are not targets.

## Commands

```bash
bash scripts/evaluate_train.sh                 # all predictors on the train split -> scoring/scores_train.json
bash scripts/evaluate_train.sh --only NAME     # one predictor
bash scripts/evaluate.sh                       # same as train, written to scoring/scores.json
bash scripts/evaluate_test.sh                  # hidden test (no arguments) -> scoring/scores_test.json
```

Score files are always rewritten from scratch; per-video numbers are in `scoring/metrics_detail_*.json`.
