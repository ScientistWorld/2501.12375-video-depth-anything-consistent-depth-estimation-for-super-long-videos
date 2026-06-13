# Scoring Targets

## synthetic_long_video_128 and synthetic_long_video_256

Primary paper-scale target:

- Lower `temporal_gradient_error` on the fixed RGB-D videos while keeping spatial accuracy in the same range.

Benefit targets:

- Lower `temporal_gradient_error`. This is a percent-scaled local temporal-gradient proxy for the paper's temporal alignment error, so it should be used for method comparison within this benchmark rather than exact paper-number matching.
- Lower `absrel`.
- Higher `delta1`.

Spatial-quality guardrails:

- `absrel` and `delta1` should remain close to the independent-window baseline unless a run is explicitly reported as a temporal-only ablation.
- Improvements obtained by smoothing depth so aggressively that spatial quality collapses do not satisfy the paper's claim.

Constraint targets:

- Keep `param_count_M` at or below the VDA-S scale unless explicitly reporting a separate larger-model result.
- Keep `video_training_frames_M` fixed when comparing against the provided VDA-S reproduction.

Ablation target:

- Compare against `vda_independent_windows`, which processes the same sequence without the paper's long-video consistency strategy.
