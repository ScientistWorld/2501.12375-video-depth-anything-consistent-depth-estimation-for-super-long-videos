# Experiments

## synthetic_long_video_128

This experiment checks whether a long-video depth estimator can keep depth predictions accurate and temporally stable on a fixed 128-frame RGB-D sequence. It compares a naive windowed baseline against the reproduced paper method under the same model-size and training-data constraints.

The key question is whether the proposed long-video processing improves temporal behavior without sacrificing single-frame depth quality.

## synthetic_long_video_256

This experiment repeats the same comparison on a longer 256-frame RGB-D sequence. It stresses transitions across more windows and checks whether the same temporal-consistency benefit persists as video length increases.
