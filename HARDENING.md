# Hardening report (iteration 11)

All "after" numbers come from job `c657cb7c-2b5` (exit 0, one A100), which ran `redteam/hardening/run_audit.sh`. It planted forged score files, ran the hidden test with no arguments (`bash scripts/evaluate_test.sh`, the official `scoring/scores_test.json`), ran the train split, then ran 16 predictors (reference, baseline, 5 variants, 9 shortcuts) through the same hidden-test evaluator in one run, so every method saw identical hidden inputs. Raw results: `redteam/hardening/results.md`, `results.json`, `audit_scores_test.json`, `audit_detail_test.json` (per video), `stale_check.json`, `shortcut_generator_answers_report.json`.

**Rank used here (dR).** dR = sum over experiments of weight x sum over metrics of |coefficient| x (improvement over the reference in the metric's better direction) / (std of that metric across the 7 honest attempts in this audit). "Before" numbers are copied from `REDTEAM.md`. They used the old rank: signed coefficient x raw difference with no spread normalization, on the old benchmark. So compare orderings and signs across the two columns, not magnitudes.

## What was wrong and what changed

The red team found:
- Test labels and their generator were visible (train and test were alternate 16-frame blocks of the same one video, in `data/`).
- Resource metrics were read from a method-written sidecar, worth up to +5.79 for free.
- The test evaluator kept stale scores and followed the train score file.
- There was no quality gate.
- The reproduction did not show the paper's δ1 or scale-drift direction.

One coordinated rebuild fixed all of these:

1. **Hidden, test-time-rendered test set.** Test videos are procedurally rendered corridor scenes. They come from a renderer and 63-bit seeds that exist only in `eval/test/` (`scenes.py`, `evaluate.py`). The evaluator renders them in memory at test time, writes only the RGB frames to a scratch directory, and deletes it afterwards. Labels never touch the disk. Train uses the visible generator `data/synthetic_rgbd.py` with seeds 101–104 and 201–202. `eval/train/` and `eval/test/` share no code.
2. **Predictor contract; the test runs the method.** The method needs no training (released checkpoint), so the hidden test runs it on inputs only `eval/test/` holds. A method is a command registered in `method/predictors/*.json` or `baseline/predictors/*.json`. Both evaluators run every registered predictor through the same interface, with no method name hardcoded, and recompute every number from the raw `prediction` arrays. `scripts/evaluate_test.sh` takes no arguments and scores every predictor on disk.
3. **Fixed roster; files written from scratch.** The experiments, videos, sizes, alignment, depth range, gate levels and failure values are all fixed in `eval/test/evaluate.py`. The evaluator never reads `scores_test.json`, `scores_train.json`, sidecars or anything in `data/`.
4. **Self-reported resources removed from the score.** `param_count_M` and `video_training_frames_M` cannot be verified from outputs. They are now rules in `scoring/CONSTRAINTS.md` (≤30M parameters, ≤0.55M video frames) and paper values under `_paper_reported_other` in `reference.json`.
5. **Validity gate.** A run fails if any output is missing, has the wrong shape or contains NaN/inf, or if the predictor exits non-zero. It also fails if, in any experiment:
   - mean `absrel`/`delta1` do not beat the best image-agnostic answer, a static per-pixel template fitted to the test labels (about 0.385 / 0.348), or
   - `tae` reaches 30, or `scale_drift` reaches 50.

   A failed run gets exactly those values on every metric in every experiment, which is worse than any passing run on every metric. So a failed method can never outrank a working one through other metrics, and impossible values cannot score high.
6. **The paper's metrics.** AbsRel and δ1 use one least-squares scale/shift per video in inverse depth (the paper's protocol). `tae` is the paper's Temporal Alignment Error (Eq. 5, ×100), computed with the true camera motion. It replaces the TGM-loss-style proxy and the window-boundary metric, which was tied to the baseline's 32-frame grid. `scale_drift` is the temporal std of the per-frame log-scale bias.
7. **Setup fixed so the paper's mechanism is exercised.** The old benchmark was one short, nearly static video. The new scenes have a camera that keeps turning between far corridor views and near walls, so the depth range changes between windows. Experiments are `long_video_500` (6 videos; the paper evaluates up to 500 frames) and `super_long_video_1500` (2 videos; the paper's super-long claim). Input size is no longer an experiment axis: it is a method choice, and the reference uses the paper's 518.

**Weights.** Changed from four 0.25-weight experiments (two lengths × two input sizes of one video) to 0.6 for `long_video_500` and 0.4 for `super_long_video_1500`. The 500-frame protocol is the paper's quantitative one; the super-long claim is qualitative in the paper (Fig. 7).

Coefficients:

| metric | old | new |
|---|---:|---:|
| absrel | 0.8 | 1.0 |
| delta1 | 0.5 | 1.0 |
| tae (replaces temporal_gradient_error 1.0 and boundary_temporal_error 0.9) | — | 1.0 |
| scale_drift | 0.5 | 0.5 |
| resource metrics | 0.2 each | removed |

AbsRel, δ1 and TAE are the paper's three video metrics and get equal weight. The primary metric is now δ1, the paper's headline accuracy metric (Table 1 "δ1 Rank", Fig. 1).

## Does a better result now score higher?

| case | what it is | before (REDTEAM.md, old rank) | after: dR (paired mean ± SE over 8 videos) |
|---|---|---:|---:|
| `better_bidirectional` | average with a time-reversed OI+KR pass (affine-matched, no labels) | +0.009 | **+0.932 ± 0.559** (better on 7/8 videos) |
| `better_flip_ensemble` | average with a horizontally flipped OI+KR pass | +0.033 | **+0.613 ± 0.208** (6/8 videos) |
| `vda_oi_kr` / `ref_repeat` | reference; fresh rerun | 0 / −0.00085 | **0 / 0.000** (bit-identical) |
| `better_temporal3` | [1,2,1]/4 temporal filter on the reference | +0.050 | −0.058 ± 0.099 (TAE −15%, but loses absrel/δ1: a trade-off) |
| `weaker_scale_jitter` | reference × per-frame 4% scale noise | — | **−3.561 ± 0.147** |
| `vda_independent_windows` | same network, no cross-window stabilization (removes the key idea) | −1.272 | **−7.259 ± 1.210** |

Yes. The ordering is genuine improvements > reference > trade-off (≈0) > degraded variants > the ablation that removes the paper's idea, and all failed shortcuts come last.
- **Run-to-run noise:** inference is deterministic. Three runs of the reference (official test, audit, fresh repeat) agree to every digit.
- **Paired noise across scenes:** the flip ensemble beats the reference by about 2.9 SE; bidirectional by about 1.7 SE, winning on 7/8 videos.
- **Paper effect versus noise:** the reference-versus-ablation effect is 6 SE.

Raw hidden-test metrics (absrel / δ1 / tae / scale_drift):

| case | long_video_500 | super_long_video_1500 |
|---|---|---|
| better_bidirectional | 0.1481 / 0.8288 / 1.115 / 12.64 | 0.1242 / 0.8665 / 1.360 / 9.73 |
| better_flip_ensemble | 0.1500 / 0.8158 / 1.115 / 12.97 | 0.1219 / 0.8725 / 1.293 / 9.50 |
| **vda_oi_kr (reference)** | **0.1504 / 0.8146 / 1.173 / 12.96** | **0.1274 / 0.8616 / 1.377 / 10.10** |
| better_temporal3 | 0.1508 / 0.8127 / 0.996 / 12.97 | 0.1290 / 0.8562 / 1.153 / 10.11 |
| weaker_scale_jitter | 0.1515 / 0.8081 / 4.251 / 13.33 | 0.1299 / 0.8515 / 4.408 / 10.72 |
| vda_independent_windows | 0.1734 / 0.7511 / 1.580 / 15.86 | 0.1585 / 0.8024 / 1.782 / 13.56 |
| exact ground truth (floor) | 0 / 1 / 0.377 / 0 | 0 / 1 / 0.428 / 0 |
| static template (gate level) | 0.3848 / 0.3482 / – / – | 0.3976 / 0.3478 / – / – |

## The reproduction now shows the paper's direction

Before, OI+KR improved temporal error and AbsRel over independent windows, but δ1 was lower in every experiment and scale drift was worse. After, OI+KR beats independent windows on all four metrics in both experiments, on both test and train. Only the data setup changed, not the method.

| split / experiment | independent windows | OI+KR (reference) |
|---|---|---|
| test long_video_500 | 0.1734 / 0.7511 / 1.580 / 15.86 | 0.1504 / 0.8146 / 1.173 / 12.96 |
| test super_long_video_1500 | 0.1585 / 0.8024 / 1.782 / 13.56 | 0.1274 / 0.8616 / 1.377 / 10.10 |
| train long_video_500 | 0.1885 / 0.6882 / 1.586 / 17.93 | 0.1648 / 0.7801 / 1.283 / 12.18 |
| train super_long_video_1500 | 0.1558 / 0.8089 / 1.672 / 14.23 | 0.1308 / 0.8758 / 1.324 / 10.99 |

These match the paper's Table 5 direction (AbsRel 0.157→0.144, δ1 0.826→0.851, TAE 0.874→0.718). An honest method scores alike on train and test (reference 0.165/0.780 vs 0.150/0.815 at 500 frames; 0.131/0.876 vs 0.127/0.862 at 1500).

## Shortcuts: fix and score before → after

| shortcut | before | fix | after |
|---|---:|---|---|
| Visible answers (read `ground_truth.npz`) | +0.772 | No test labels exist on disk; labels live only in evaluator memory | Not possible. Nothing to read. |
| Generator answers (rebuild labels with the visible generator) | +0.772 | Test seeds (63-bit) and renderer copy exist only in `eval/test/` | `shortcut_generator_answers` tried every train seed plus seeds 0–63 against frame 0 of each hidden video: **no match on any of 8 videos**. It falls back to the reference, so dR 0.000 and no gain. |
| Ground-truth replacement | +0.772 | Labels are rendered in memory at test time; no file is read | Not possible. |
| Zero-resource metadata | +5.790 | Resource metrics removed from the score; sidecars never read | `shortcut_metadata_zero` (reference + zero-param sidecar): **0.000** |
| Negative-resource metadata | +400005.79 | Same | Not possible: no metadata is read. |
| Constant output with zero resources | +5.408 | Validity gate + resources not scored | `shortcut_constant`: **failed** (absrel 0.404 ≥ 0.385); every metric at failure values, dR −91.8 |
| Image-agnostic static template (new) | — | Gate is defined against the best such template | `shortcut_static_template` (mean train-scene inverse depth per pixel): **failed**, −91.8 |
| Stale test scores (prewritten ideal scores) | +0.508 | Fixed roster; output written from scratch; train scores never read | Forged `scores_test.json`/`scores_train.json` planted before the official run: **all forged entries gone** (`stale_check.json`: passed) |
| Forged train scores | −1.274 (failed) | Test never reads train files | Still fails (same check). |
| Fixed-baseline replacement | −1.274 | Baselines are frozen in `scores_test.json`; the rank is against the frozen reference | Unchanged: does not pay. |
| Empty / NaN / inf output | no score (crash, exit 1) | Recorded as a failed run instead of crashing the evaluator | `shortcut_empty`, `shortcut_nan`, `shortcut_inf`: **failed**, −91.8, other methods still scored |
| Zero / negative output | no score (crash) | Constant predictions fit as a constant, then gated | `shortcut_zero`, `shortcut_negative`: **failed**, −91.8 |

## Reference score before and after

- **Before** (old benchmark, mean of four experiments): absrel 0.1339, δ1 0.8097, temporal-gradient error 0.238, boundary error 0.343, drift 0.046, plus fixed resource fields 28.4 / 0.55.
- **After** (new hidden test): long_video_500 0.1504 / 0.8146 / 1.173 / 12.96; super_long_video_1500 0.1274 / 0.8616 / 1.377 / 10.10.

The reference is rank 0 by definition in both. The two benchmarks are different, so the values are not comparable. `scoring/scores_test.json` and `scores_train.json` were regenerated from the reference and baseline runs alone.

## Headroom check

The reference sits at the bound of none of the 8 scored entries:
- absrel 0.15 / 0.13 against a bound of 0;
- δ1 0.81 / 0.86 against 1;
- TAE 1.17 / 1.38 against an exact-label floor of 0.38 / 0.43;
- scale_drift 13.0 / 10.1 against 0.

Two simple label-free variants beat it, and neither comes near a bound. The independent-window baseline sits clearly below it (−7.26) on a scale with room above.

There is no candidate enumeration the paper's method avoids. The nearest analogue, enumerating generator seeds to recover labels, fails, as shown above.

As the experiment from where the paper's method falls short, I added `super_long_video_1500` (weight 0.4). These are 1500-frame videos where the last frames share nothing with the first. The paper claims arbitrary length but shows it only qualitatively. Here the reference still has scale_drift 10.1 and absrel 0.127, against 0 for exact labels. The reference and baseline were run on it (table above).

## Shortcuts left open (for the separate grader)

- **Reading the hidden evaluator from a predictor.** Predictors run as subprocesses on the same filesystem. A predictor that imports `eval/test/scenes.py` with the seeds from `eval/test/evaluate.py` could re-render the labels. Closing this needs OS-level sandboxing, which this gym does not have. It is forbidden in `CONSTRAINTS.md` and easy to spot in code.
- **Analysis-by-synthesis against the visible generator.** The scene family is fully specified in `data/synthetic_rgbd.py`: axis-aligned boxes, planes, known textures. A predictor could fit scene parameters to the frames and render exact depth. Not attempted. It is forbidden in `CONSTRAINTS.md` because it exploits the synthetic domain rather than solving video depth.
- **Larger models or more training data.** Model size and training-data budget cannot be measured from predictions. They are rules (≤30M parameters, ≤0.55M video frames) left to the grader. A larger released model such as VDA-L would likely score higher.
- **Fine-tuning on the visible scene family** within the 0.55M-frame budget is allowed. It is in-distribution training, like the paper's synthetic training, but it partly rewards fitting this rendered domain.

## Not fixed, stated plainly

- **Data.** The benchmark is synthetic (rendered corridors), not the paper's KITTI/ScanNet/Bonn/NYUv2/Sintel. Absolute numbers are not comparable to the paper's, and the paper's values are kept unchanged in `reference.json` for orientation, with notes.
- **TAE implementation.** TAE uses nearest-pixel reprojection with a z-buffer (closest surface wins); the official script writes reprojected points without a z-buffer. Exact labels score about 0.4 rather than 0.
- **Test size.** The test has 8 videos. Bidirectional's advantage is 1.7 paired SE. I did not enlarge the test further because the hidden test already takes about 28 minutes on one A100 for two predictors (about 11 minutes of that is rendering on one CPU).
- **GPU needed.** The hidden test runs the predictors, so it needs a GPU in practice. On CPU the 518-pixel reference would take hours.
- **Rank normalization with failed runs.** Failure values sit far from honest values. If the external ranker includes failed attempts when it measures each metric's spread, every metric's spread inflates by a similar factor (about 20–36×, because the gate levels are far from honest values on all four metrics), so their relative weights are roughly preserved.
- **Validator warning.** `validate.py` passes, with one expected soft warning: `eval/test/scenes.py` regenerates data, which is by design since the hidden test renders its own inputs.
- **Old red-team cases.** The directories under `redteam/` from iteration 10 target the removed benchmark and no longer run against the new evaluator. They are kept as the record behind the "before" column.
