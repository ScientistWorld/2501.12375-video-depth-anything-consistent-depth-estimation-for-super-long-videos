# Overview

This workspace reproduces the long-video inference claim of Video Depth Anything, on real videos from two of the paper's benchmarks (Bonn RGB-D Dynamic and KITTI). The claim: processing a long video with overlap interpolation and key-frame referencing (OI+KR) gives more accurate, more temporally consistent and more scale-stable depth than processing fixed windows independently, with the same small network.

It packages the official implementation with the released VDA-S checkpoint and two evaluators. Each runs every registered predictor on real video clips:
- visible train clips in `data/downloads/`, scored by `eval/train/`;
- hidden test clips held only in `eval/test/`, scored by `eval/test/`.

The metrics are the paper's AbsRel, delta1 and Temporal Alignment Error under one scale/shift alignment per video, plus a scale-drift measure. They are computed on 500-frame indoor and outdoor clips and on 1500-frame driving clips.

To regenerate the reference scores, run `bash scripts/download.sh` (about 22 GB, needs internet), then `bash scripts/reproduce.sh` (one GPU).

---

# Problem Definition

## Research Question

Estimate dense depth for long monocular videos so that the predictions are both spatially accurate and temporally consistent over hundreds to thousands of frames. The input is an RGB video only (no camera poses, no intrinsics, no depth). The output is one relative depth map per frame. Relative means affine-invariant in inverse depth, but one scale and shift must hold for the whole video: if the depth scale of frame 1400 does not match frame 10, the prediction is wrong.

## Why It Matters

Video depth feeds robotics, mapping, 3D video conversion and video editing. Single-image depth estimators give plausible per-frame maps, but on video they flicker. Any model that sees only a bounded number of frames at a time tends to produce scale jumps between the pieces it processed, or a scale that drifts slowly over a long video. Downstream uses such as fusing frames into a point cloud need one consistent geometry for the whole clip.

## Setting in this environment

The videos are real recordings from two public RGB-D / LiDAR video benchmarks that the paper evaluates on, used zero-shot:

- **Indoor, dynamic.** Bonn RGB-D Dynamic: a handheld Kinect-style camera moves through a lab while people walk, carry boxes or play with balloons. 640x480 at 30 fps. Ground truth is the sensor's registered depth (dense, with holes), and the camera motion comes from a motion-capture system.
- **Outdoor, driving.** KITTI: the left colour camera of a car driving through city, residential and road scenes. About 1242x375 at 10 fps. Ground truth is the official annotated LiDAR depth (sparse, mostly the lower part of the image), and the camera motion comes from the car's GPS/IMU.

Three settings are evaluated: 500-frame clips of both benchmarks (the paper's long-video protocol), and 1500-frame KITTI clips (2.5 minutes of driving, roughly 70 times longer than a 32-frame processing window, and the scene at the end shares nothing with the start).

## Success Criteria

A successful system:

- has low depth error after one global scale/shift alignment per video (AbsRel, delta1);
- is temporally consistent: depth reprojected from one frame to the next with the true camera motion agrees with the next frame's depth (Temporal Alignment Error);
- keeps one depth scale over the whole video, without jumps or slow drift;
- does this zero-shot, at the small-model scale and training-data budget given in `scoring/CONSTRAINTS.md`, so gains come from better video-depth methods rather than a much larger model, extra supervision or fitting the benchmarks themselves.

---

# Method

The paper, "Video Depth Anything: Consistent Depth Estimation for Super-Long Videos" (arXiv 2501.12375), turns the Depth Anything V2 image model into a video depth model:

- **Architecture.** The frozen Depth Anything V2 ViT encoder processes every frame; the DPT head becomes a spatio-temporal head by inserting four temporal self-attention layers (attention along the time axis at each spatial position, absolute temporal position embeddings). The model consumes 32-frame clips.
- **Training.** Scale-shift-invariant spatial loss plus a temporal gradient matching loss (the change of depth between consecutive frames should match the ground truth change, on stable regions), on 0.55M synthetic video frames, jointly with distillation on unlabeled images. Only the head is trained.
- **Long-video inference (OI+KR).** The video is processed as 32-frame clips that advance by 22 frames. Each new clip contains 22 new frames, 8 overlap frames from the previous clip and 2 key frames (the first frame of the video and one earlier frame), so the network sees context it has already predicted. The new clip's predictions are scale/shift-aligned to the stored predictions of the key frames, and the 8 overlap frames are linearly cross-faded between the two clips. The paper's ablation (Table 5) shows this beats independent windows, overlap alignment and overlap interpolation alone.

## Implementation in this workspace

- `method/official/`: the released Video Depth Anything code (`video_depth_anything/video_depth.py::infer_video_depth` implements OI+KR with `INFER_LEN=32`, `OVERLAP=10`, `KEYFRAMES=[0,12,...]`, `INTERP_LEN=8`).
- `method/predict_vda.py`: predictor entry point (contract in `briefing/evaluation.md`); `--mode oi_kr` is the paper's inference, `--mode independent_windows` the no-stabilisation baseline. Both use the released VDA-S checkpoint (`data/downloads/video_depth_anything_vits.pth`, 28.4M parameters), the paper's 518-pixel input (the official code lowers it for frames wider than 16:9, e.g. to 280 for KITTI) and fp32, run on the native-resolution frames, and submit the full-resolution output sampled on the evaluation grid.
- `method/predictors/vda_oi_kr.json`, `baseline/predictors/vda_independent_windows.json`: the registered reference and baseline.
- `method/run_vda.py`: model loading and the independent-window loop.

The checkpoint is used as released; nothing is retrained here.

---

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

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 14m | **GPU**: 0.1h
- **Jobs**: 3 total (2 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-13 04:30] - none
- Read the paper and identified the core long-video inference claim.
- Vendored the official Video Depth Anything implementation.
- Added synthetic benchmark generation, artifact-based evaluation, scripts, and packaging for the first compute-node reproduction run.

### [2026-06-13 04:46] - method_runs
- GPU smoke job `57ef9201-abf` completed successfully.
- Ran the actual VDA-S model and official long-video inference path end to end on the synthetic benchmark.
- Scored both `vda_independent_windows` and `vda_oi_kr`; OI+KR improved temporal gradient error but not spatial metrics on the initial 72-frame sequence.

### [2026-06-13 04:49] - method_runs
- Job `4c759670-8bf` failed during evaluation because stale 72-frame output was mixed with fresh 128-frame ground truth.
- Patched scripts to clear full-run outputs and evaluate explicit method artifacts.

### [2026-06-13 04:52] - core_claim
- Job `679781e3-847` completed successfully on the 128-frame benchmark.
- OI+KR improved temporal gradient error from `0.003628` to `0.003295` with the same VDA-S parameter count and video training-data scale.
- Spatial accuracy stayed close but did not improve (`absrel` 0.1245 to 0.1247, `delta1` 0.8389 to 0.8328), so the reproduced claim is limited to long-video temporal consistency rather than the full paper benchmark profile.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 4 total (3 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-13 04:50] - core_claim
- Ran the released VDA-S model through the official OI+KR long-video inference path on a 128-frame RGB-D benchmark.
- Compared against independent 32-frame window inference with the same checkpoint and model scale.
- OI+KR improved the local temporal-gradient consistency metric while preserving close spatial depth quality.

### [2026-06-13 04:56] - core_claim_plus
- Re-ran the full reproduction after fixing temporal metric scaling.
- Added and completed a 256-frame setting to test the long-video strategy over more inference windows.
- OI+KR improved temporal-gradient consistency on both 128-frame and 256-frame settings at the same model size and training-data scale.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h
- **Jobs**: 5 total (3 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-13 05:02] - core_claim_plus
- Ran the official Video Depth Anything small model checkpoint through the reproduced OI+KR long-video inference path and the independent-window baseline.
- Scored both methods on fixed 128-frame and 256-frame synthetic RGB-D videos with artifact-only evaluation.
- The OI+KR method improved temporal-gradient error on both settings while keeping model size and training-data scale fixed.
- `python validate.py --compare` passes, including method/eval/data separation checks.

### [2026-06-13 05:06] - core_claim_plus
- Audited the continuing workspace from a reset `core_claim` milestone.
- Confirmed the implementation uses the official Video Depth Anything model and OI+KR long-video inference rather than a surrogate.
- Fixed the scoring target wording so the primary local metric is temporal consistency, with spatial accuracy retained as a constraint.
- Prepared a 518-input-size follow-up job to test whether the paper-resolution inference better preserves spatial accuracy.

### [2026-06-13 05:27] - core_claim_plus
- Reviewed failed job `6c4a8f31-83f`; inference and evaluation completed for both 518-input-size experiments, but validation failed because the new experiment names were missing from `reference.json`.
- Added reference and documentation entries for all four scored experiments.
- The 518-input-size runs again show OI+KR reducing temporal-gradient error at fixed model scale and training-data scale, with spatial metrics close to the independent-window baseline.

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 5 total (3 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-13 05:02] - core_claim_plus
- Ran the official Video Depth Anything small model checkpoint through the reproduced OI+KR long-video inference path and the independent-window baseline.
- Scored both methods on fixed 128-frame and 256-frame synthetic RGB-D videos with artifact-only evaluation.
- The OI+KR method improved temporal-gradient error on both settings while keeping model size and training-data scale fixed.
- `python validate.py --compare` passes, including method/eval/data separation checks.

### [2026-06-13 05:06] - core_claim_plus
- Audited the continuing workspace from a reset `core_claim` milestone.
- Confirmed the implementation uses the official Video Depth Anything model and OI+KR long-video inference rather than a surrogate.
- Fixed the scoring target wording so `absrel` is the primary paper-comparable metric, with temporal consistency retained as the key local benefit metric and spatial accuracy as a guardrail.
- Prepared a 518-input-size follow-up job to test whether the paper-resolution inference better preserves spatial accuracy.

### [2026-06-13 05:27] - core_claim_plus
- Reviewed failed job `6c4a8f31-83f`; inference and evaluation completed for both 518-input-size experiments, but validation failed because the new experiment names were missing from `reference.json`.
- Added reference and documentation entries for all four scored experiments.
- The 518-input-size runs again show OI+KR reducing temporal-gradient error at fixed model scale and training-data scale, with spatial metrics close to the independent-window baseline.

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 5 total (3 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-13 05:02] - core_claim_plus
- Ran the official Video Depth Anything small model checkpoint through the reproduced OI+KR long-video inference path and the independent-window baseline.
- Scored both methods on fixed 128-frame and 256-frame synthetic RGB-D videos with artifact-only evaluation.
- The OI+KR method improved temporal-gradient error on both settings while keeping model size and training-data scale fixed.
- `python validate.py --compare` passes, including method/eval/data separation checks.

### [2026-06-13 05:06] - core_claim_plus
- Audited the continuing workspace from a reset `core_claim` milestone.
- Confirmed the implementation uses the official Video Depth Anything model and OI+KR long-video inference rather than a surrogate.
- Fixed the scoring target wording so `absrel` is the primary paper-comparable metric, with temporal consistency retained as the key local benefit metric and spatial accuracy as a guardrail.
- Prepared a 518-input-size follow-up job to test whether the paper-resolution inference better preserves spatial accuracy.

### [2026-06-13 05:27] - core_claim_plus
- Reviewed failed job `6c4a8f31-83f`; inference and evaluation completed for both 518-input-size experiments, but validation failed because the new experiment names were missing from `reference.json`.
- Added reference and documentation entries for all four scored experiments.
- The 518-input-size runs again show OI+KR reducing temporal-gradient error at fixed model scale and training-data scale, with spatial metrics close to the independent-window baseline.

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 6 total (4 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-13 05:02] - core_claim_plus
- Ran the official Video Depth Anything small model checkpoint through the reproduced OI+KR long-video inference path and the independent-window baseline.
- Scored both methods on fixed 128-frame and 256-frame synthetic RGB-D videos with artifact-only evaluation.
- The OI+KR method improved temporal-gradient error on both settings while keeping model size and training-data scale fixed.
- `python validate.py --compare` passes, including method/eval/data separation checks.

### [2026-06-13 05:06] - core_claim_plus
- Audited the continuing workspace from a reset `core_claim` milestone.
- Confirmed the implementation uses the official Video Depth Anything model and OI+KR long-video inference rather than a surrogate.
- Fixed the scoring target wording so `absrel` is the primary paper-comparable metric, with temporal consistency retained as the key local benefit metric and spatial accuracy as a guardrail.
- Prepared a 518-input-size follow-up job to test whether the paper-resolution inference better preserves spatial accuracy.

### [2026-06-13 05:27] - core_claim_plus
- Reviewed failed job `6c4a8f31-83f`; inference and evaluation completed for both 518-input-size experiments, but validation failed because the new experiment names were missing from `reference.json`.
- Added reference and documentation entries for all four scored experiments.
- The 518-input-size runs again show OI+KR reducing temporal-gradient error at fixed model scale and training-data scale, with spatial metrics close to the independent-window baseline.

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 6 total (4 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-13 05:02] - core_claim_plus
- Ran the official Video Depth Anything small model checkpoint through the reproduced OI+KR long-video inference path and the independent-window baseline.
- Scored both methods on fixed 128-frame and 256-frame synthetic RGB-D videos with artifact-only evaluation.
- The OI+KR method improved temporal-gradient error on both settings while keeping model size and training-data scale fixed.
- `python validate.py --compare` passes, including method/eval/data separation checks.

### [2026-06-13 05:06] - core_claim_plus
- Audited the continuing workspace from a reset `core_claim` milestone.
- Confirmed the implementation uses the official Video Depth Anything model and OI+KR long-video inference rather than a surrogate.
- Fixed the scoring target wording so `absrel` is the primary paper-comparable metric, with temporal consistency retained as the key local benefit metric and spatial accuracy as a guardrail.
- Prepared a 518-input-size follow-up job to test whether the paper-resolution inference better preserves spatial accuracy.

### [2026-06-13 05:27] - core_claim_plus
- Reviewed failed job `6c4a8f31-83f`; inference and evaluation completed for both 518-input-size experiments, but validation failed because the new experiment names were missing from `reference.json`.
- Added reference and documentation entries for all four scored experiments.
- The 518-input-size runs again show OI+KR reducing temporal-gradient error at fixed model scale and training-data scale, with spatial metrics close to the independent-window baseline.

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h
- **Jobs**: 6 total (4 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-13 05:02] - core_claim_plus
- Ran the official Video Depth Anything small model checkpoint through the reproduced OI+KR long-video inference path and the independent-window baseline.
- Scored both methods on fixed 128-frame and 256-frame synthetic RGB-D videos with artifact-only evaluation.
- The OI+KR method improved temporal-gradient error on both settings while keeping model size and training-data scale fixed.
- `python validate.py --compare` passes, including method/eval/data separation checks.

### [2026-06-13 05:06] - core_claim_plus
- Audited the continuing workspace from a reset `core_claim` milestone.
- Confirmed the implementation uses the official Video Depth Anything model and OI+KR long-video inference rather than a surrogate.
- Fixed the scoring target wording so `absrel` is the primary paper-comparable metric, with temporal consistency retained as the key local benefit metric and spatial accuracy as a guardrail.
- Prepared a 518-input-size follow-up job to test whether the paper-resolution inference better preserves spatial accuracy.

### [2026-06-13 05:27] - core_claim_plus
- Reviewed failed job `6c4a8f31-83f`; inference and evaluation completed for both 518-input-size experiments, but validation failed because the new experiment names were missing from `reference.json`.
- Added reference and documentation entries for all four scored experiments.
- The 518-input-size runs again show OI+KR reducing temporal-gradient error at fixed model scale and training-data scale, with spatial metrics close to the independent-window baseline.

### [2026-06-13 05:40] - core_claim_plus
- Completed final verification of structure, scoring, documentation separation, and download URL reachability.
- Restored canonical score artifacts from completed full-evaluator container output and confirmed `python validate.py --compare` passes.
- Removed generated CUDA cache files from version control and documented the final gym-readiness judgment in `VERIFICATION.md`.

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h
- **Jobs**: 6 total (4 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-13 05:02] - core_claim_plus
- Ran the official Video Depth Anything small model checkpoint through the reproduced OI+KR long-video inference path and the independent-window baseline.
- Scored both methods on fixed 128-frame and 256-frame synthetic RGB-D videos with artifact-only evaluation.
- The OI+KR method improved temporal-gradient error on both settings while keeping model size and training-data scale fixed.
- `python validate.py --compare` passes, including method/eval/data separation checks.

### [2026-06-13 05:06] - core_claim_plus
- Audited the continuing workspace from a reset `core_claim` milestone.
- Confirmed the implementation uses the official Video Depth Anything model and OI+KR long-video inference rather than a surrogate.
- Fixed the scoring target wording so `absrel` is the primary paper-comparable metric, with temporal consistency retained as the key local benefit metric and spatial accuracy as a guardrail.
- Prepared a 518-input-size follow-up job to test whether the paper-resolution inference better preserves spatial accuracy.

### [2026-06-13 05:27] - core_claim_plus
- Reviewed failed job `6c4a8f31-83f`; inference and evaluation completed for both 518-input-size experiments, but validation failed because the new experiment names were missing from `reference.json`.
- Added reference and documentation entries for all four scored experiments.
- The 518-input-size runs again show OI+KR reducing temporal-gradient error at fixed model scale and training-data scale, with spatial metrics close to the independent-window baseline.

### [2026-06-13 05:40] - core_claim_plus
- Completed final verification of structure, scoring, documentation separation, and download URL reachability.
- Restored canonical score artifacts from completed full-evaluator container output and confirmed `python validate.py --compare` passes.
- Removed generated CUDA cache files from version control and documented the final gym-readiness judgment in `VERIFICATION.md`.

### [2026-06-13 05:47] - core_claim_plus
- Fixed the artifact evaluator so standalone runs replace `scores.json` by default instead of retaining stale experiments.
- Updated full-run scripts to opt into append mode only after clearing previous score artifacts.
- Submitted a CPU-only container test to verify score regeneration with the runtime dependencies available.

### [2026-06-13 05:51] - core_claim_plus
- Confirmed CPU test job `af51b2f2-70e` completed successfully in the compute container.
- Re-ran `python validate.py --compare` locally after the job; scoring, train/test, and separation checks pass.
- Final verification found no critical issues and preserved the `core_claim_plus` milestone.

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 5m | **GPU**: 0.1h
- **Jobs**: 6 total (4 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-13 05:02] - core_claim_plus
- Ran the official Video Depth Anything small model checkpoint through the reproduced OI+KR long-video inference path and the independent-window baseline.
- Scored both methods on fixed 128-frame and 256-frame synthetic RGB-D videos with artifact-only evaluation.
- The OI+KR method improved temporal-gradient error on both settings while keeping model size and training-data scale fixed.
- `python validate.py --compare` passes, including method/eval/data separation checks.

### [2026-06-13 05:06] - core_claim_plus
- Audited the continuing workspace from a reset `core_claim` milestone.
- Confirmed the implementation uses the official Video Depth Anything model and OI+KR long-video inference rather than a surrogate.
- Fixed the scoring target wording so `absrel` is the primary paper-comparable metric, with temporal consistency retained as the key local benefit metric and spatial accuracy as a guardrail.
- Prepared a 518-input-size follow-up job to test whether the paper-resolution inference better preserves spatial accuracy.

### [2026-06-13 05:27] - core_claim_plus
- Reviewed failed job `6c4a8f31-83f`; inference and evaluation completed for both 518-input-size experiments, but validation failed because the new experiment names were missing from `reference.json`.
- Added reference and documentation entries for all four scored experiments.
- The 518-input-size runs again show OI+KR reducing temporal-gradient error at fixed model scale and training-data scale, with spatial metrics close to the independent-window baseline.

### [2026-06-13 05:40] - core_claim_plus
- Completed final verification of structure, scoring, documentation separation, and download URL reachability.
- Restored canonical score artifacts from completed full-evaluator container output and confirmed `python validate.py --compare` passes.
- Removed generated CUDA cache files from version control and documented the final gym-readiness judgment in `VERIFICATION.md`.

### [2026-06-13 05:47] - core_claim_plus
- Fixed the artifact evaluator so standalone runs replace `scores.json` by default instead of retaining stale experiments.
- Updated full-run scripts to opt into append mode only after clearing previous score artifacts.
- Submitted a CPU-only container test to verify score regeneration with the runtime dependencies available.

### [2026-06-13 05:51] - core_claim_plus
- Confirmed CPU test job `af51b2f2-70e` completed successfully in the compute container.
- Re-ran `python validate.py --compare` locally after the job; scoring, train/test, and separation checks pass.
- Final verification found no critical issues and preserved the `core_claim_plus` milestone.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 23m | **GPU**: 1.7h
- **Jobs**: 7 total (5 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-13 05:02] - core_claim_plus
- Ran the official Video Depth Anything small model checkpoint through the reproduced OI+KR long-video inference path and the independent-window baseline.
- Scored both methods on fixed 128-frame and 256-frame synthetic RGB-D videos with artifact-only evaluation.
- The OI+KR method improved temporal-gradient error on both settings while keeping model size and training-data scale fixed.
- `python validate.py --compare` passes, including method/eval/data separation checks.

### [2026-06-13 05:06] - core_claim_plus
- Audited the continuing workspace from a reset `core_claim` milestone.
- Confirmed the implementation uses the official Video Depth Anything model and OI+KR long-video inference rather than a surrogate.
- Fixed the scoring target wording so `absrel` is the primary paper-comparable metric, with temporal consistency retained as the key local benefit metric and spatial accuracy as a guardrail.
- Prepared a 518-input-size follow-up job to test whether the paper-resolution inference better preserves spatial accuracy.

### [2026-06-13 05:27] - core_claim_plus
- Reviewed failed job `6c4a8f31-83f`; inference and evaluation completed for both 518-input-size experiments, but validation failed because the new experiment names were missing from `reference.json`.
- Added reference and documentation entries for all four scored experiments.
- The 518-input-size runs again show OI+KR reducing temporal-gradient error at fixed model scale and training-data scale, with spatial metrics close to the independent-window baseline.

### [2026-06-13 05:40] - core_claim_plus
- Completed final verification of structure, scoring, documentation separation, and download URL reachability.
- Restored canonical score artifacts from completed full-evaluator container output and confirmed `python validate.py --compare` passes.
- Removed generated CUDA cache files from version control and documented the final gym-readiness judgment in `VERIFICATION.md`.

### [2026-06-13 05:47] - core_claim_plus
- Fixed the artifact evaluator so standalone runs replace `scores.json` by default instead of retaining stale experiments.
- Updated full-run scripts to opt into append mode only after clearing previous score artifacts.
- Submitted a CPU-only container test to verify score regeneration with the runtime dependencies available.

### [2026-06-13 05:51] - core_claim_plus
- Confirmed CPU test job `af51b2f2-70e` completed successfully in the compute container.
- Re-ran `python validate.py --compare` locally after the job; scoring, train/test, and separation checks pass.
- Final verification found no critical issues and preserved the `core_claim_plus` milestone.

### [2026-11-09 17:00] - core_claim_plus
- Hardened the gym: hidden test videos are rendered in `eval/test/` and every registered predictor runs on them. Added the paper's TAE metric and a validity gate, and removed the unverifiable resource metrics.
- The reproduced OI+KR inference (released VDA-S checkpoint) now beats independent windows on AbsRel, delta1, TAE and scale drift on 500- and 1500-frame videos, matching the paper's Table 5 direction.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 49m | **GPU**: 3.4h
- **Jobs**: 9 total (7 completed, 2 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-13 05:02] - core_claim_plus
- Ran the official Video Depth Anything small model checkpoint through the reproduced OI+KR long-video inference path and the independent-window baseline.
- Scored both methods on fixed 128-frame and 256-frame synthetic RGB-D videos with artifact-only evaluation.
- The OI+KR method improved temporal-gradient error on both settings while keeping model size and training-data scale fixed.
- `python validate.py --compare` passes, including method/eval/data separation checks.

### [2026-06-13 05:06] - core_claim_plus
- Audited the continuing workspace from a reset `core_claim` milestone.
- Confirmed the implementation uses the official Video Depth Anything model and OI+KR long-video inference rather than a surrogate.
- Fixed the scoring target wording so `absrel` is the primary paper-comparable metric, with temporal consistency retained as the key local benefit metric and spatial accuracy as a guardrail.
- Prepared a 518-input-size follow-up job to test whether the paper-resolution inference better preserves spatial accuracy.

### [2026-06-13 05:27] - core_claim_plus
- Reviewed failed job `6c4a8f31-83f`; inference and evaluation completed for both 518-input-size experiments, but validation failed because the new experiment names were missing from `reference.json`.
- Added reference and documentation entries for all four scored experiments.
- The 518-input-size runs again show OI+KR reducing temporal-gradient error at fixed model scale and training-data scale, with spatial metrics close to the independent-window baseline.

### [2026-06-13 05:40] - core_claim_plus
- Completed final verification of structure, scoring, documentation separation, and download URL reachability.
- Restored canonical score artifacts from completed full-evaluator container output and confirmed `python validate.py --compare` passes.
- Removed generated CUDA cache files from version control and documented the final gym-readiness judgment in `VERIFICATION.md`.

### [2026-06-13 05:47] - core_claim_plus
- Fixed the artifact evaluator so standalone runs replace `scores.json` by default instead of retaining stale experiments.
- Updated full-run scripts to opt into append mode only after clearing previous score artifacts.
- Submitted a CPU-only container test to verify score regeneration with the runtime dependencies available.

### [2026-06-13 05:51] - core_claim_plus
- Confirmed CPU test job `af51b2f2-70e` completed successfully in the compute container.
- Re-ran `python validate.py --compare` locally after the job; scoring, train/test, and separation checks pass.
- Final verification found no critical issues and preserved the `core_claim_plus` milestone.

### [2026-11-09 17:00] - core_claim_plus
- Hardened the gym: hidden test videos are rendered in `eval/test/` and every registered predictor runs on them. Added the paper's TAE metric and a validity gate, and removed the unverifiable resource metrics.
- The reproduced OI+KR inference (released VDA-S checkpoint) now beats independent windows on AbsRel, delta1, TAE and scale drift on 500- and 1500-frame videos, matching the paper's Table 5 direction.

### [2026-11-21 09:30] - core_claim_plus
- Moved the gym from synthetic corridors to real long videos from two of the paper's benchmarks: Bonn RGB-D Dynamic (12 hidden clips of 500 frames) and KITTI (10 hidden clips of 500 frames, 5 hidden clips of 1500 frames), with disjoint visible train clips.
- The paper's protocol is used: per-video scale/shift, AbsRel, delta1, TAE with the true motion-capture or GPS/IMU poses.
- The released VDA-S with OI+KR gives 0.080 / 0.964 on Bonn, against the paper's 0.083 / 0.950, and 0.112 / 0.886 on KITTI. It beats independent windows on all four metrics in all three experiments, on test and on train.
- Hidden-test predictors run in an isolated namespace that hides `eval/test/`. Label-free test-time ensembles beat the reference by 3 to 6 paired standard errors. See `HARDENING.md`.

</details>


---

# Reproduction Milestones

**Current: core_claim_plus**

## Stop Justification
- Completed at milestone `core_claim_plus`.
