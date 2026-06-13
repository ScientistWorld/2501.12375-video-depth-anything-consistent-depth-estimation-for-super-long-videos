# Scoring Targets

## synthetic_long_video_128 and synthetic_long_video_256

Primary paper-scale target:

- Maintain or improve `delta1` on the fixed RGB-D videos while improving temporal consistency.

Benefit targets:

- Lower `absrel`.
- Higher `delta1`.
- Lower `temporal_gradient_error`. This is a percent-scaled local temporal-gradient proxy for the paper's temporal alignment error, so it should be used for method comparison within this benchmark rather than exact paper-number matching.

Constraint targets:

- Keep `param_count_M` at or below the VDA-S scale unless explicitly reporting a separate larger-model result.
- Keep `video_training_frames_M` fixed when comparing against the provided VDA-S reproduction.

Ablation target:

- Compare against `vda_independent_windows`, which processes the same sequence without the paper's long-video consistency strategy.
