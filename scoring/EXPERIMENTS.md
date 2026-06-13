# Experiments

## synthetic_long_video_128

This experiment checks whether a long-video depth estimator can keep depth predictions temporally stable on a fixed 128-frame RGB-D sequence while preserving comparable single-frame depth quality. It compares a naive windowed baseline against the reproduced paper method under the same model-size and training-data constraints.

The key question is whether the proposed long-video processing improves temporal behavior without changing the resource budget or collapsing spatial depth metrics.

## synthetic_long_video_256

This experiment repeats the same comparison on a longer 256-frame RGB-D sequence. It stresses transitions across more windows and checks whether the same temporal-consistency benefit persists as video length increases.

## synthetic_long_video_128_input518

This experiment repeats the 128-frame comparison at the paper's default 518-pixel inference input size. It checks whether the temporal-consistency effect remains visible under the same preprocessing scale used by the released model.

## synthetic_long_video_256_input518

This experiment combines the longer 256-frame sequence with the paper-size 518-pixel inference setting. It is the strongest local stress test in this environment because it includes more window transitions and the default model input scale.
