# Scoring Targets

## synthetic_long_video_128, synthetic_long_video_256, synthetic_long_video_128_input518, and synthetic_long_video_256_input518

Primary paper-comparable target:

- Lower `absrel` on the fixed RGB-D videos. This is the primary metric because it has the same definition and scale as the paper's reported KITTI ablation values.

Benefit targets:

- Lower `temporal_gradient_error`. This is a percent-scaled local temporal-gradient proxy for the paper's temporal alignment error, so it should be used for method comparison within this benchmark rather than exact paper-number matching.
- Lower `absrel`.
- Higher `delta1`.

Spatial-quality guardrails:

- `absrel` and `delta1` should remain close to the independent-window baseline unless a run is explicitly reported as a temporal-only ablation.
- Improvements obtained by smoothing depth so aggressively that spatial quality collapses do not satisfy the paper's claim.

Constraint targets:

- Keep `param_count_M` at or below the provided small-model scale unless explicitly reporting a separate larger-model result.
- Keep `video_training_frames_M` fixed when comparing against the provided reproduction baseline.

Ablation target:

- Compare against `vda_independent_windows`, which processes the same sequence without the paper's long-video consistency strategy.

Coverage target:

- The `input518` variants use the higher reference inference size and should remain in the score set when testing future improvements. The 252-size variants are faster smoke tests that exercise the same benchmark at lower compute cost.
