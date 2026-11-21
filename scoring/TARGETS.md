# Scoring Targets

Three experiments score the same four metrics on hidden real videos:

| experiment | benchmark | weight |
|---|---|---:|
| `bonn_500` | Bonn RGB-D Dynamic, 12 × 500 frames | 0.35 |
| `kitti_500` | KITTI, 10 × 500 frames | 0.35 |
| `kitti_1500` | KITTI, 5 × 1500 frames | 0.30 |

The reference is `vda_oi_kr` in `scoring/scores_test.json`; improvements are measured against it.

## Benefit targets

- **Spatial accuracy under one scale/shift per video:** lower `absrel` and higher `delta1`. delta1 is the primary metric and the paper's headline accuracy metric. Coefficient 1.0 each.
- **Temporal consistency:** lower `tae`, the paper's Temporal Alignment Error, computed with the true camera motion. Coefficient 1.0.
- **One depth scale for the whole video:** lower `scale_drift`. Coefficient 0.5.

Reference on the hidden test (absrel / delta1 / tae / scale_drift):

| experiment | reference `vda_oi_kr` | paper, VDA-S (Table 1, ≤500 frames) |
|---|---|---|
| `bonn_500` | 0.0803 / 0.9636 / 1.572 / 5.96 | 0.083 / 0.950 / – / – |
| `kitti_500` | 0.1123 / 0.8859 / 3.916 / 7.14 | 0.086 / 0.942 / – / – |
| `kitti_1500` | 0.1104 / 0.8908 / 4.040 / 6.96 | no number at this length |

None of these is at a bound: absrel and scale_drift have a bound of 0 and delta1 a bound of 1. TAE does not reach 0 even for perfect depth, because people and cars move independently of the camera. Simple label-free test-time ensembles already beat the reference by several standard errors (see `HARDENING.md`), so there is measurable room above it.

## Constraint targets

- **Validity gate (scored).** Outputs must be finite and correctly shaped for every video. In every experiment:
  - mean `absrel` and `delta1` must beat the best image-agnostic static depth template fitted to the test labels: 0.237 / 0.582 (Bonn), 0.190 / 0.694 (KITTI 500), 0.208 / 0.663 (KITTI 1500);
  - `tae` must stay below 30 and `scale_drift` below 50.

  A failed run scores worse than any passing run on every metric.
- **No trading accuracy for smoothness.** Temporal gains that cost more spatial accuracy than they earn are not improvements, and the rank sums all four metrics. A 9-frame temporal box filter of the reference lowers TAE but ranks far below it.
- **Rules checked on the submitted code (`scoring/CONSTRAINTS.md`).** These cannot be measured from predictions:
  - zero-shot: no training on, or lookup of, the Bonn/KITTI recordings;
  - small-model scale;
  - the 0.55M video-frame budget.

## Ablation / baseline target

The independent-window baseline (`vda_independent_windows`) uses the same network without cross-window stabilisation. It scores below the reference on all four metrics in every experiment:

| experiment | baseline (absrel / delta1 / tae / scale_drift) |
|---|---|
| `bonn_500` | 0.0918 / 0.9460 / 1.841 / 8.83 |
| `kitti_500` | 0.1265 / 0.8564 / 4.167 / 9.26 |
| `kitti_1500` | 0.1182 / 0.8745 / 4.343 / 8.60 |
