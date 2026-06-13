# Scoring Targets

## synthetic_long_video

Primary target:

- Improve `delta1` on the fixed 72-frame RGB-D video.

Benefit targets:

- Lower `absrel`.
- Higher `delta1`.
- Lower `temporal_gradient_error`.

Constraint targets:

- Keep `param_count_M` at or below the VDA-S scale unless explicitly reporting a separate larger-model result.
- Keep `video_training_frames_M` fixed when comparing against the provided VDA-S reproduction.

Ablation target:

- Compare against `vda_independent_windows`, which processes the same sequence without the paper's long-video consistency strategy.
