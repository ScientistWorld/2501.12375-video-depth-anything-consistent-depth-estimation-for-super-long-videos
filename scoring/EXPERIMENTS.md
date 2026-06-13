# Experiments

## synthetic_long_video_128

This experiment checks whether a long-video depth estimator can keep predictions temporally stable on a fixed 128-frame RGB-D sequence while preserving comparable single-frame depth quality. It compares submitted prediction artifacts against an independent-window reference under the same model-size and training-data constraints.

The key question is whether long-video processing improves temporal behavior without changing the resource budget or collapsing spatial depth metrics.

## synthetic_long_video_256

This experiment repeats the same comparison on a longer 256-frame RGB-D sequence. It stresses transitions across more windows and checks whether the same temporal-consistency benefit persists as video length increases.

## synthetic_long_video_128_input518

This experiment repeats the 128-frame comparison at a 518-pixel inference input size. It checks whether the temporal-consistency effect remains visible at the higher preprocessing scale used for the reference artifacts.

## synthetic_long_video_256_input518

This experiment combines the longer 256-frame sequence with the 518-pixel inference setting. It is the strongest local stress test in this environment because it includes more window transitions and the higher reference input scale.
