# Overview

This workspace reproduces the long-video inference claim of Video Depth Anything: processing a long video with overlap interpolation and key-frame referencing (OI+KR) gives more accurate, more temporally consistent and more scale-stable depth than processing fixed windows independently, with the same small network.

It packages the official implementation with the released VDA-S checkpoint, a procedural RGB-D corridor-video generator with exact depth and camera poses, and evaluators that run every registered predictor on rendered videos: visible train scenes in `eval/train/`, hidden test scenes rendered at test time in `eval/test/`. Metrics are AbsRel, delta1 and the paper's Temporal Alignment Error under one per-video scale/shift alignment, plus a scale-drift measure, on 500-frame and 1500-frame videos. Run `bash scripts/reproduce.sh` (one GPU, about 50 minutes) to regenerate the reference scores.

---

# Problem Definition

## Research Question

Estimate dense depth for long monocular videos so that the predictions are both spatially accurate and temporally consistent over hundreds to thousands of frames. The input is an RGB video only (no camera poses, no intrinsics, no depth); the output is one relative depth map per frame. Relative means affine-invariant in inverse depth, but a single scale and shift must hold for the whole video: if the depth scale of frame 1400 does not match frame 10, the prediction is wrong.

## Why It Matters

Video depth feeds robotics, mapping, 3D video conversion and video editing. Single-image depth estimators give plausible per-frame maps, but on video they flicker, and any model that sees only a bounded number of frames at a time tends to produce scale jumps between the pieces it processed, or a scale that drifts slowly over a long video. Downstream uses such as fusing frames into a point cloud need one consistent geometry for the whole clip.

## Setting in this environment

Videos are static indoor corridors (floor, ceiling, walls, partition walls with doorways, furniture) rendered with exact ray casting while a camera walks forward, sways, turns towards the walls and tilts. The visible depth range therefore changes a lot over time: some stretches look far down the corridor, others face a nearby wall. Lighting has a slow exposure drift, frame-to-frame gain flicker, fog and sensor noise. Frames are 144x192 at 12 fps. Two lengths are evaluated: 500 frames and 1500 frames (about two minutes, where the end of the video shares no content with the start).

## Success Criteria

A successful system:

- has low depth error after one global scale/shift alignment per video (AbsRel, delta1);
- is temporally consistent: depth reprojected from one frame to the next with the true camera motion agrees with the next frame's depth (Temporal Alignment Error);
- keeps one depth scale over the whole video, without jumps or slow drift;
- does this at the small-model scale and training-data budget given in `scoring/CONSTRAINTS.md`, so gains come from better video-depth methods rather than a much larger model or extra supervision.

---

# Method

The paper, "Video Depth Anything: Consistent Depth Estimation for Super-Long Videos" (arXiv 2501.12375), turns the Depth Anything V2 image model into a video depth model:

- **Architecture.** The frozen Depth Anything V2 ViT encoder processes every frame; the DPT head becomes a spatio-temporal head by inserting four temporal self-attention layers (attention along the time axis at each spatial position, absolute temporal position embeddings). The model consumes 32-frame clips.
- **Training.** Scale-shift-invariant spatial loss plus a temporal gradient matching loss (the change of depth between consecutive frames should match the ground truth change, on stable regions), on 0.55M synthetic video frames, jointly with distillation on unlabeled images. Only the head is trained.
- **Long-video inference (OI+KR).** The video is processed as 32-frame clips that advance by 22 frames. Each new clip contains 22 new frames, 8 overlap frames from the previous clip and 2 key frames (the first frame of the video and one earlier frame), so the network sees context it has already predicted. The new clip's predictions are scale/shift-aligned to the stored predictions of the key frames, and the 8 overlap frames are linearly cross-faded between the two clips. The paper's ablation (Table 5) shows this beats independent windows, overlap alignment and overlap interpolation alone.

## Implementation in this workspace

- `method/official/`: the released Video Depth Anything code (`video_depth_anything/video_depth.py::infer_video_depth` implements OI+KR with `INFER_LEN=32`, `OVERLAP=10`, `KEYFRAMES=[0,12,...]`, `INTERP_LEN=8`).
- `method/predict_vda.py`: predictor entry point (contract in `briefing/evaluation.md`); `--mode oi_kr` is the paper's inference, `--mode independent_windows` the no-stabilisation baseline. Both use the released VDA-S checkpoint (`data/downloads/video_depth_anything_vits.pth`, 28.4M parameters), 518-pixel input and fp32.
- `method/predictors/vda_oi_kr.json`, `baseline/predictors/vda_independent_windows.json`: the registered reference and baseline.
- `method/run_vda.py`: model loading and the independent-window loop (also a small CLI for PNG frame folders).

The checkpoint is used as released; nothing is retrained here.

---

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

---

# Hardening (iteration 11)

The benchmark, evaluators and scoring were rebuilt in iteration 11 after the red-team audit; see `HARDENING.md` for what changed, why, and the before/after scores of the reference, variants and shortcuts. Earlier iterations below describe the previous single-video benchmark (128/256-frame synthetic sequence split by frame blocks), which no longer exists.

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


### Iteration 11: claude-opus-5.5 (hardener)
- **Milestone**: `core_claim_plus`
- Rebuilt the benchmark as hidden rendered long corridor videos (500 and 1500 frames) with exact depth and poses; test videos are rendered inside `eval/test/` and only their frames reach predictors.
- Added the paper's Temporal Alignment Error, a validity gate, and a predictor contract that both evaluators share; removed self-reported resource metrics.
- The reproduced OI+KR inference now beats independent windows on AbsRel, delta1, TAE and scale drift in both experiments (train and test).

---

# Reproduction Milestones

**Current: core_claim_plus**

## Stop Justification
- Completed at milestone `core_claim_plus`.
