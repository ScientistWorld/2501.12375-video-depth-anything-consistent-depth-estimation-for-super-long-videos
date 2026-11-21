# Hardening report (iteration 12)

`REVIEW.md` found two remaining weaknesses:
- synthetic corridor videos stood in for the paper's real benchmarks;
- the best gains were test-time ensembles at only 1.7–2.9 SE.

This iteration moves the whole gym, train and hidden test, onto real long videos from two of the paper's own benchmarks. It also closes the main shortcut the previous hardening left open: a predictor reading the hidden test data while it runs. It then re-audits everything.

All "after" numbers come from job `7d38514d-b5f` (exit 0, one A100), which ran `redteam/hardening2/run_audit.sh`. The steps were:
1. A pre-flight check (`smoke.py`):
   - every test file is on disk;
   - with the true camera motion, ground-truth depth reprojects better than with no motion on every clip;
   - exact labels score perfectly;
   - GPS/IMU steps are consistent with the measured speed.
2. Forged score files were planted.
3. The hidden test ran with no arguments (`bash scripts/evaluate_test.sh` → official `scoring/scores_test.json`), and the plant was checked.
4. The train split ran (official `scoring/scores_train.json` / `scores.json`).
5. 16 audit predictors ran through the same hidden-test evaluator, so every method saw identical inputs.

Raw results:
- `redteam/hardening2/results.md` and `results.json`;
- `audit_scores_test.json` and `audit_detail_test.json` (per video);
- `smoke.json` and `stale_check.json`;
- the shortcut reports.

An earlier full run (job `499d2ba8-684`) is kept in `redteam/hardening2/run1/`. It found one test clip with a GPS glitch (see "Data fix during this iteration" below); the clip was replaced and everything was rerun.

**Rank used here (dR).** For each experiment, each metric's improvement over the reference is taken in the metric's better direction. It is divided by that metric's std across the 8 honest attempts in this audit and multiplied by |coefficient|. dR is the weighted sum of these over experiments. The paired figure "mean ± SE" is the same rank computed per video (videos of an experiment weighted equally), with its standard error over videos. "Before" numbers are copied from the iteration-11 `HARDENING.md` (same rank, old synthetic benchmark). Compare orderings and signs across the two columns, not magnitudes.

## What changed

1. **Real videos from the paper's benchmarks, hidden test only in `eval/test/`.**
   - `bonn_500`: the first 500 frames of 12 Bonn RGB-D Dynamic sequences (indoor, handheld camera, people moving). 640x480 at 30 fps, registered Kinect depth, motion-capture poses.
   - `kitti_500`: 500-frame clips of 10 KITTI drives, including the paper's two long KITTI validation drives. About 1242x375 at 10 fps, official annotated LiDAR depth, GPS/IMU poses.
   - `kitti_1500`: five 1500-frame clips (2.5 minutes) of 3 other KITTI drives. This is the paper's super-long setting.

   The visible train split (`data/train_clips.json`) has 6 / 4 / 2 further clips from disjoint sequences and drives. `scripts/download.sh` fetches the test clips into `eval/test/downloads/` (15 GB) and the train clips into `data/downloads/` (6.3 GB). It uses `data/fetch_clips.py`, a standard-library fetcher that range-reads only the needed members of the public archives and CRC-checks each one.

   The paper's protocol is followed:
   - one least-squares scale/shift per video in inverse depth;
   - AbsRel and delta1 per frame, then averaged;
   - max depth 10 m (Bonn) or 80 m (KITTI);
   - TAE as in the paper's Eq. 5 (x100) with the true relative poses.

   Predictions are scored on every 2nd row and column of the native frame. This keeps a predictor's test outputs at about 7 GB instead of about 28 GB. Uniform pixel subsampling leaves the metrics' means unchanged.
2. **Poses verified, not assumed.**
   - Bonn uses the dataset page's formula T_ROS^-1 T_k T_ROS T_m: T_m's rotation is orthonormalised and T_k is interpolated to the RGB timestamp. Other pose conventions were tested and rejected; this one gives the lowest ground-truth reprojection error (median 1.5% vs 6.7% for no motion).
   - KITTI uses the OXTS Mercator pose and the IMU→camera-2 calibration: 0.3% vs 1.7%.
   - Each Bonn RGB frame takes the depth map closest in time (within 20 ms). Pairing by list position, as the official extraction script does, drifts by up to 0.2 s on `crowd3` because depth frames were dropped.
3. **The predictor cannot read the test answers while it runs.** On the hidden test, each predictor runs under `unshare` in new user, mount and PID namespaces:
   - `eval/test/` is replaced by an empty tmpfs;
   - the input-frame directory is bind-mounted read-only;
   - both mounts are locked by a nested user namespace, so it can neither unmount nor remount them;
   - `/proc` is private, so it cannot reach another process's view of the filesystem.

   Ground truth is loaded only after the predictor returns. The evaluator self-tests the isolation at start-up and records the result in `metrics_detail_test.json` (`"sandbox": "active: ..."` in both jobs, GPU usable inside). Where a platform does not allow it, the predictor runs unisolated on copied frames, and the reason is recorded. Hard links are used only when the read-only mount protects them.
4. **Opaque inputs.** Predictors get native-resolution RGB PNG files under ids `video_000`…, with no sequence names, poses or intrinsics.
5. **No wall-clock limit** in the method's path (the old 4-hour timeout is now an opt-in flag).
6. **Kept from iteration 11:**
   - fixed experiment roster;
   - score files written from scratch, recomputed from raw `prediction` arrays;
   - no sidecars read;
   - the validity gate against the best image-agnostic static template, with caps TAE 30 and drift 50; a failed run gets those values on every metric.

**Weights.** These were 0.6 / 0.4 on two synthetic experiments. They are now:
- `bonn_500` 0.35 and `kitti_500` 0.35: the paper's quantitative Table 1 protocol on an indoor-dynamic and an outdoor benchmark.
- `kitti_1500` 0.30: the paper's headline super-long claim, shown only qualitatively in the paper.

Coefficients are unchanged: absrel 1, delta1 1, tae 1, scale_drift 0.5. The primary metric stays delta1.

## Does a better result now score higher?

Yes. The reference is deterministic (a fresh rerun is bit-identical: `ref_repeat` dR 0.000). Paired differences over the 27 hidden videos separate the variants cleanly.

| case | what it is | before (synthetic) | after: dR (paired mean ± SE) | videos better (bonn / kitti500 / kitti1500) |
|---|---|---:|---:|---|
| `better_flip_bidir` | average of 4 OI+KR passes (as is, mirrored, time-reversed, both), affine-matched to each other, no labels | — | **+2.703 ± 0.459** (5.9 SE) | 11/12, 9/10, 5/5 |
| `better_flip_ensemble` | average with a mirrored OI+KR pass | +0.613 ± 0.208 | **+1.381 ± 0.219** (6.3 SE) | 11/12, 9/10, 5/5 |
| `better_bidirectional` | average with a time-reversed OI+KR pass | +0.932 ± 0.559 | **+1.353 ± 0.406** (3.3 SE) | 10/12, 7/10, 4/5 |
| `vda_oi_kr` / `ref_repeat` | reference; fresh rerun | 0 / 0 | **0 / 0.000** | — |
| `better_temporal3` | [1,2,1]/4 temporal filter of the reference (trade-off: TAE −8 to −14%, loses absrel/delta1) | −0.058 ± 0.099 | −0.611 ± 0.120 | 9/12, 3/10, 0/5 |
| `weaker_scale_jitter` | reference × 4% per-frame scale noise | −3.561 ± 0.147 | **−4.958 ± 0.150** | 0/12, 0/10, 0/5 |
| `vda_independent_windows` | same network, no cross-window stabilisation (removes the paper's idea) | −7.259 ± 1.210 | **−6.065 ± 2.040** | 2/12, 1/10, 0/5 |

The flip ensemble now separates by 6.3 paired SE, against 2.9 before. It wins on 25 of 27 videos. The 4-pass ensemble separates by 5.9 SE. Every genuine improvement beats the reference in every experiment except one: the bidirectional pass on `kitti_1500` (+0.72 ± 0.51).

Raw hidden-test metrics (absrel / delta1 / tae / scale_drift):

| case | bonn_500 | kitti_500 | kitti_1500 |
|---|---|---|---|
| better_flip_bidir | 0.0744 / 0.9733 / 1.535 / 5.54 | 0.1066 / 0.8961 / 3.755 / 6.68 | 0.1040 / 0.9033 / 3.892 / 6.62 |
| better_flip_ensemble | 0.0792 / 0.9656 / 1.537 / 5.80 | 0.1091 / 0.8919 / 3.781 / 7.07 | 0.1054 / 0.9007 / 3.907 / 6.79 |
| better_bidirectional | 0.0751 / 0.9718 / 1.569 / 5.62 | 0.1100 / 0.8899 / 3.885 / 6.79 | 0.1090 / 0.8934 / 4.024 / 6.81 |
| **vda_oi_kr (reference)** | **0.0803 / 0.9636 / 1.572 / 5.96** | **0.1123 / 0.8859 / 3.916 / 7.14** | **0.1104 / 0.8908 / 4.040 / 6.96** |
| better_temporal3 | 0.0798 / 0.9622 / 1.439 / 5.98 | 0.1150 / 0.8776 / 3.380 / 7.13 | 0.1143 / 0.8796 / 3.646 / 6.95 |
| weaker_scale_jitter | 0.0823 / 0.9606 / 3.353 / 6.37 | 0.1174 / 0.8755 / 6.365 / 8.19 | 0.1152 / 0.8813 / 6.426 / 7.99 |
| vda_independent_windows | 0.0918 / 0.9460 / 1.841 / 8.83 | 0.1265 / 0.8564 / 4.167 / 9.26 | 0.1182 / 0.8745 / 4.343 / 8.60 |
| static template (gate level) | 0.2374 / 0.5823 / – / – | 0.1904 / 0.6936 / – / – | 0.2082 / 0.6629 / – / – |

## The reproduction shows the paper's claim

OI+KR beats independent windows on all four metrics in all three experiments, on both test and train (`results.json` → `paper_direction`: 12/12 on test). This matches the paper's Table 5 direction.

| split / experiment | independent windows | OI+KR (reference) |
|---|---|---|
| test bonn_500 | 0.0918 / 0.9460 / 1.841 / 8.83 | 0.0803 / 0.9636 / 1.572 / 5.96 |
| test kitti_500 | 0.1265 / 0.8564 / 4.167 / 9.26 | 0.1123 / 0.8859 / 3.916 / 7.14 |
| test kitti_1500 | 0.1182 / 0.8745 / 4.343 / 8.60 | 0.1104 / 0.8908 / 4.040 / 6.96 |
| train bonn_500 | 0.0861 / 0.9487 / 1.402 / 8.45 | 0.0701 / 0.9814 / 1.093 / 6.07 |
| train kitti_500 | 0.1552 / 0.7936 / 5.510 / 11.90 | 0.1240 / 0.8639 / 5.497 / 7.93 |
| train kitti_1500 | 0.1272 / 0.8567 / 4.367 / 9.73 | 0.1149 / 0.8824 / 4.092 / 7.94 |

On Bonn, the reproduced VDA-S reaches 0.080 / 0.964 against the paper's Table 1 VDA-S Bonn entry of 0.083 / 0.950. On KITTI it reaches 0.112 / 0.886 against the paper's 0.086 / 0.942. The paper's KITTI set is the 13 validation drives cut to at most 500 frames, most of them much shorter. Ours are 500-frame clips of 10 mostly longer, busier drives. The method is the released code and checkpoint with the official benchmark settings (518 input, fp32), unchanged.

An honest method scores alike on train and test (reference absrel 0.070 vs 0.080, 0.124 vs 0.112 and 0.115 vs 0.110).

## Shortcuts: fix and score before → after

| shortcut | before | fix | after |
|---|---|---|---|
| Predictor reads the hidden test data while it runs (iteration 11 left this open: import the test renderer and seeds from `eval/test/`) | open (exact labels) | Namespace isolation: `eval/test/` is an empty locked tmpfs, `/proc` is private, ground truth is loaded after the predictor returns | `shortcut_read_hidden_answers` searches the hidden data for its frames and reads their depth. It found nothing on all 27 videos (`root_visible: false`), so it fell back to the reference: **dR 0.000**. With access, the same lookup gives +24.9 (next row). |
| Predictor rewrites the shared test frames (hard links) | — (new) | Inputs are mounted read-only and locked in the sandbox; frames are copied when there is no sandbox | Probe: write, remount and unmount all fail (`inputs_read_only: true`). |
| Generator / analysis-by-synthesis of the synthetic scenes (iteration 11 left this open) | open | No generator exists any more: the data is real | Not applicable. |
| Image-agnostic static template (mean train-label inverse depth per pixel) | failed | Gate | `shortcut_static_template`: **failed**, dR −115.7 |
| Constant / zero output | failed (−91.8) | Gate | `shortcut_constant`, `shortcut_zero`: **failed**, −115.7 |
| NaN / empty output | failed (−91.8) | Shape/finiteness checks → failure values | `shortcut_nan`, `shortcut_empty`: **failed**, −115.7; other methods still scored |
| Forged / stale score files | passed | Output always rewritten from the fixed roster | Forged `scores_test.json`/`scores_train.json` planted before the official run: **no forged entry survived** (`stale_check.json`) |
| Short hand-written rules | — | (checked) | 9-frame temporal box filter of the reference: TAE −16 to −24%, but **−6.348 ± 0.882** (accuracy loss dominates). Per-frame median normalisation: **failed** the gate on KITTI. Neither outranks the reference. |
| Self-reported resources (iteration 10) | +5.79 | Not scored (iteration 11) | Still not read. |

## Shortcuts left open (for the separate grader)

- **Looking frames up in a private copy of the public benchmarks: +24.939 ± 1.624.** The test clips are public recordings, so an agent can download Bonn/KITTI themselves. It can match a frame by its bytes and read the depth. `shortcut_public_copy_lookup` does this from a copy outside `eval/test/` and gets AbsRel 0 / delta1 1 everywhere.

  No evaluator-side measure can close this while the test uses the paper's public benchmarks. Re-encoding or perturbing frames would only push it to near-duplicate matching. It is forbidden in `scoring/CONSTRAINTS.md`, `briefing/evaluation.md` and `scoring/DIRECTION.md`, and it is unmistakable in code: the method would contain or fetch Bonn/KITTI depth.
- **Training on Bonn/KITTI, including the visible train clips.** Forbidden as a zero-shot rule (the paper excludes all evaluation videos from training). The evaluator cannot see it.
- **Larger models or more video supervision.** Model size and training data cannot be measured from predictions; they are rules (≤30M parameters, ≤0.55M video frames). A released VDA-B/VDA-L would score higher.

## Reference score before and after

- **Before** (synthetic corridors):
  - `long_video_500`: 0.1504 / 0.8146 / 1.173 / 12.96;
  - `super_long_video_1500`: 0.1274 / 0.8616 / 1.377 / 10.10.
- **After** (real hidden videos):
  - `bonn_500`: 0.0803 / 0.9636 / 1.572 / 5.96;
  - `kitti_500`: 0.1123 / 0.8859 / 3.916 / 7.14;
  - `kitti_1500`: 0.1104 / 0.8908 / 4.040 / 6.96.

The reference is rank 0 by definition in both. `scoring/scores_test.json` and `scores_train.json` (and `scores.json`) were regenerated from the reference and baseline runs alone, from the final code and data.

## Headroom check

The reference is at the bound of none of the 12 scored entries:
- absrel is 0.080 / 0.112 / 0.110 against 0;
- delta1 is 0.964 / 0.886 / 0.891 against 1;
- scale_drift is 6.0 / 7.1 / 7.0 against 0;
- TAE is 1.6 / 3.9 / 4.0. It cannot reach 0 for any method because of independently moving people and cars, but label-free ensembles already lower it.

Three label-free variants beat the reference by 3.3–6.3 paired SE, and none comes near a bound. The independent-window baseline sits clearly below (−6.07) on a scale with room above.

The experiment where the paper's method falls short is `kitti_1500` (weight 0.30). Its 1500-frame clips are about 70 windows long, with drift 7.0 and absrel 0.110 against 0 for exact labels; the reference and baseline were run on it. The per-video detail also shows two Bonn sequences where OI+KR is clearly worse than independent windows:

| sequence | OI+KR absrel | independent windows absrel |
|---|---:|---:|
| `removing_obstructing_box` | 0.120 | 0.054 |
| `placing_obstructing_box` | 0.072 | 0.051 |

A likely cause is key-frame referencing carrying a poor early scale through the clip. Either way it is a concrete failure a better method can fix.

There is no candidate enumeration that the paper's method avoids. The nearest brute force is looking up the labels, covered above.

## Data fix during this iteration

The first full run flagged `kitti_20110930_0028_0005` in the pre-flight. Around raw frame 1005, its GPS altitude jumps by 9.9 m within 0.3 s, which gives 42 wrong relative camera poses. TAE on that clip was 7.0, against about 4 on clean clips.

`redteam/hardening2/oxts_scan.py` checks every consecutive GPS/IMU step against the measured speed. The fix:
- That clip was replaced by the same drive from frame 1505. All 21 KITTI clips (15 test, 6 train) now have 0 inconsistent steps.
- The scan is part of the pre-flight.
- Everything was rerun.

Run-1 numbers (glitched clip) are in `redteam/hardening2/run1/`. The ordering was identical: flip +1.363 ± 0.207, 4-pass +2.405 ± 0.470, bidirectional +1.121 ± 0.391, baseline −6.368 ± 2.090.

## Not fixed, stated plainly

- **Two of the paper's five video benchmarks.**
  - ScanNet is under its own terms of use and is not redistributable by a script.
  - NYUv2's raw videos are about 430 GB.
  - Sintel's clips are about 50 frames, not long videos.

  The paper reports TAE only on ScanNet, so there is no paper TAE for Bonn/KITTI to compare with.
- **KITTI accuracy below the paper's number** (0.112 / 0.886 vs 0.086 / 0.942): different and longer clips, explained above. The direction of every claim is reproduced, and Bonn matches.
- **TAE details.**
  - TAE uses a z-buffer (the closest surface wins); the official script lets the last write win.
  - It is computed on all pixels, including moving people and cars and the KITTI sky (clipped at 80 m), as in the official script.
  - In two Bonn sequences (`kidnapping_box2`, `person_tracking2`) the first RGB frame precedes the first motion-capture sample by more than 50 ms. The first pose is held for at most 1–2 frame pairs.
- **Test size.** 12 / 10 / 5 videos. `kitti_1500` has 5 clips from 3 drives; the bidirectional variant's gain there (+0.72 ± 0.51) is not significant on its own. The KITTI drives with ≥1500 annotated frames are few, and two of them are used for train.
- **Platform dependence of the isolation.** It needs unprivileged user namespaces (`unshare`), which the job container allows. Elsewhere the evaluator falls back to an unisolated run on copied frames and says so in the detail file.
- **Cost.**
  - Download: 22 GB.
  - The hidden test needs a GPU: about 24 minutes for the reference and baseline on one A100, including scoring.
  - Data licences: Bonn and KITTI are public; KITTI is CC BY-NC-SA 3.0.
- **Rank normalisation with failed runs.** Failure values sit far from honest values on all four metrics. If the external ranker includes failed attempts when it measures spreads, every metric's spread inflates by a similar factor, so their relative weights are roughly preserved.
- **Old red-team cases.** `redteam/` directories from iterations 10 and 11 (including `redteam/hardening/`) target the removed synthetic benchmarks and no longer run. They are kept as the record behind the "before" column.
