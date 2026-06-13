# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-13 04:30] - none
- Read the paper and identified the core long-video inference claim.
- Vendored the official Video Depth Anything implementation.
- Added synthetic benchmark generation, artifact-based evaluation, scripts, and packaging for the first compute-node reproduction run.

### [2026-06-13 04:46] - method_runs
- GPU smoke job `57ef9201-abf` completed successfully.
- Ran the actual VDA-S model and official long-video inference path end to end on the synthetic benchmark.
- Scored both `vda_independent_windows` and `vda_oi_kr`; OI+KR improved temporal gradient error but not spatial metrics on the initial 72-frame sequence.

### [2026-06-13 04:49] - method_runs
- Job `4c759670-8bf` failed during evaluation because stale 72-frame output was mixed with fresh 128-frame ground truth.
- Patched scripts to clear full-run outputs and evaluate explicit method artifacts.

### [2026-06-13 04:52] - core_claim
- Job `679781e3-847` completed successfully on the 128-frame benchmark.
- OI+KR improved the local temporal gradient error from `0.003628` to `0.003295` with the same VDA-S parameter count and video training-data scale.
- Spatial accuracy stayed close but did not improve (`absrel` 0.1245 to 0.1247, `delta1` 0.8389 to 0.8328), so the reproduced claim is limited to long-video temporal consistency rather than the full paper benchmark profile.

## Stop Justification
- Completed at milestone `core_claim`.
