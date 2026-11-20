# Overview

This workspace reproduces the long-video inference claim of Video Depth Anything, on real videos from two of the paper's benchmarks (Bonn RGB-D Dynamic and KITTI). The claim: processing a long video with overlap interpolation and key-frame referencing (OI+KR) gives more accurate, more temporally consistent and more scale-stable depth than processing fixed windows independently, with the same small network.

It packages the official implementation with the released VDA-S checkpoint and two evaluators. Each runs every registered predictor on real video clips:
- visible train clips in `data/downloads/`, scored by `eval/train/`;
- hidden test clips held only in `eval/test/`, scored by `eval/test/`.

The metrics are the paper's AbsRel, delta1 and Temporal Alignment Error under one scale/shift alignment per video, plus a scale-drift measure. They are computed on 500-frame indoor and outdoor clips and on 1500-frame driving clips.

To regenerate the reference scores, run `bash scripts/download.sh` (about 22 GB, needs internet), then `bash scripts/reproduce.sh` (one GPU).
