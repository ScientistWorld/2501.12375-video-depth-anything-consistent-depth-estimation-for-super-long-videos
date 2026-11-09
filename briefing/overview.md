# Overview

This workspace reproduces the long-video inference claim of Video Depth Anything: processing a long video with overlap interpolation and key-frame referencing (OI+KR) gives more accurate, more temporally consistent and more scale-stable depth than processing fixed windows independently, with the same small network.

It packages the official implementation with the released VDA-S checkpoint, a procedural RGB-D corridor-video generator with exact depth and camera poses, and evaluators that run every registered predictor on rendered videos: visible train scenes in `eval/train/`, hidden test scenes rendered at test time in `eval/test/`. Metrics are AbsRel, delta1 and the paper's Temporal Alignment Error under one per-video scale/shift alignment, plus a scale-drift measure, on 500-frame and 1500-frame videos. Run `bash scripts/reproduce.sh` (one GPU, about 50 minutes) to regenerate the reference scores.
