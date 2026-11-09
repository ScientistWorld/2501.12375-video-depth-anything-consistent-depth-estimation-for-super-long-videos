# Scoring Targets

Both experiments (`long_video_500`, weight 0.6; `super_long_video_1500`, weight 0.4) score the same four metrics on hidden test videos. The reference is `vda_oi_kr` in `scoring/scores_test.json`; improvements are measured against it.

## Benefit targets

- Lower `absrel` and higher `delta1` (primary metric; the paper's headline accuracy metric): spatial accuracy under one scale/shift per video. Coefficients 1.0 each.
- Lower `tae`: the paper's Temporal Alignment Error, using the true camera motion. Coefficient 1.0.
- Lower `scale_drift`: one depth scale for the whole video. Coefficient 0.5.

Reference (hidden test): `long_video_500` 0.1504 / 0.8146 / 1.173 / 12.96 and `super_long_video_1500` 0.1274 / 0.8616 / 1.377 / 10.10 (absrel / delta1 / tae / scale_drift). Exact ground truth reaches 0 / 1 / about 0.4 / 0, so every metric has room to improve.

## Constraint targets

- Validity gate (scored): outputs must be finite and correctly shaped for every video, and in every experiment the mean `absrel`/`delta1` must beat the best image-agnostic static depth template (about 0.39 / 0.35) with `tae` < 30 and `scale_drift` < 50. A failed run scores worse than any passing run on every metric.
- Accuracy is not traded for smoothness: temporal gains that cost more spatial accuracy than they earn are not improvements, and the rank sums all four metrics.
- Model scale and training data stay within `scoring/CONSTRAINTS.md` (small-model scale, 0.55M video training frames). These cannot be measured from predictions, so they are rules checked on the submitted code rather than scored metrics.

## Ablation / baseline target

- The independent-window baseline (`vda_independent_windows`, same network, no cross-window stabilisation) scores clearly below the reference on all four metrics in both experiments: 0.1734 / 0.7511 / 1.580 / 15.86 and 0.1585 / 0.8024 / 1.782 / 13.56.
