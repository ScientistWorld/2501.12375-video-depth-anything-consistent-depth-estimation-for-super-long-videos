# Reproduction Milestones

**Current: method_runs**

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

## Stop Justification

- Not stopped. The next step is a longer benchmark job to test the core temporal-consistency claim more clearly.
