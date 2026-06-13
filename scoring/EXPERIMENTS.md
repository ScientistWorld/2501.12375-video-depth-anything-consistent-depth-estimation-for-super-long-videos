# Experiments

## synthetic_long_video

This experiment checks whether a long-video depth estimator can keep depth predictions accurate and temporally stable on a fixed 72-frame RGB-D sequence. It compares a naive windowed baseline against the reproduced paper method under the same model-size and training-data constraints.

The key question is whether the proposed long-video processing improves temporal behavior without sacrificing single-frame depth quality.
