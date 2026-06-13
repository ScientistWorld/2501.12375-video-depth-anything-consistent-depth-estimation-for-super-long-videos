# Progress

## What Works

- Read the paper and identified the core reproducible claim: long-video inference improves temporal consistency over independent short-window processing while keeping the same model scale.
- Vendored the official Video Depth Anything implementation in `method/official`.
- Downloaded the released VDA-S checkpoint to `data/downloads/video_depth_anything_vits.pth`.
- Added a reproducible synthetic RGB-D long-video generator in `data/synthetic_rgbd.py`.
- Added method and baseline runners that write prediction artifacts for the evaluator.
- Added an artifact-only evaluator in `eval/metrics.py`.

## Results

Job `57ef9201-abf` completed successfully on one GPU.

Initial 72-frame synthetic benchmark:

- `vda_independent_windows`: `absrel=0.1236`, `delta1=0.8355`, `temporal_gradient_error=0.004764`
- `vda_oi_kr`: `absrel=0.1247`, `delta1=0.8349`, `temporal_gradient_error=0.004494`

This demonstrates that the actual method runs and that the long-video path improves temporal consistency on the local benchmark. The spatial metrics are essentially tied but slightly worse for OI+KR, so I am not claiming `core_claim` yet.

Final 128-frame synthetic benchmark from job `679781e3-847`:

- `vda_independent_windows`: `absrel=0.1245`, `delta1=0.8389`, `temporal_gradient_error=0.003628`
- `vda_oi_kr`: `absrel=0.1247`, `delta1=0.8328`, `temporal_gradient_error=0.003295`

This supports the core long-video consistency claim: the actual OI+KR inference path improves temporal consistency by about 9.2% at the same model size and video training-data scale. Spatial depth quality is close but does not reproduce the paper's reported accuracy improvement on this synthetic benchmark. `temporal_gradient_error` is the primary local metric for the long-video consistency claim; `absrel` and `delta1` remain required spatial quality metrics.

## Remaining

- Higher milestones would require broader evaluation on the paper's real datasets or additional synthetic settings.
- A new 256-frame synthetic setting has been added and is ready for the next submitted job.
- Full training reproduction remains open; this environment uses the released VDA-S checkpoint.

## Issues

- Local login-node Python did not include `imageio` or NumPy, so metric/data smoke tests are run in the compute container.
- The initial synthetic sequence was short enough that OI+KR mostly affected temporal consistency, not spatial accuracy.
- Job `4c759670-8bf` failed because `evaluate.sh` mixed a fresh 128-frame baseline artifact with a stale 72-frame method artifact. Scripts now clear full-run outputs and allow explicit evaluation of requested methods.
- On the 128-frame benchmark, OI+KR improves temporal consistency but has a small `delta1` drop relative to independent windows.
- `temporal_gradient_error` is not numerically comparable to the paper's temporal alignment error, so it is not used as the primary paper-consistency metric.

## Deviations from Paper

- The paper evaluates on KITTI, ScanNet, Bonn, NYUv2, and Sintel. This first reproduction uses a small synthetic RGB-D video because the full benchmark suite is large and the minimum core claim can be tested on a fixed long sequence with known ground truth.
- The paper trains VDA on 0.55M video frames plus image data. This workspace uses the released VDA-S checkpoint for the first reproduction rather than retraining, preserving the paper's algorithm and model family while fitting the available compute budget.
- The evaluator uses `temporal_gradient_error` on ground-truth stable regions as the local temporal consistency metric. `scoring/reference.json` records the paper's KITTI temporal alignment error values as the external reference for the same claimed effect.
