# Verification Report

## Initial Assessment

The workspace claimed `core_claim_plus`, and the claim is justified. The method implementation uses the paper's released Video Depth Anything small checkpoint and its long-video inference path, with a matching independent-window baseline and artifact-only evaluation. Completed container runs produced scores for four synthetic long-video settings: 128 and 256 frames, each at fast and 518-pixel input sizes.

The workspace was mostly ready when verification started. The main gaps were housekeeping and documentation: the canonical score files had been disturbed by a local failed rebuild attempt, some from-scratch-facing docs used mild method-specific phrasing, and a generated CUDA cache was tracked.

## Issues Found

- Minor: `scoring/scores.json` and `scoring/metrics_detail.json` needed restoration from completed container-run outputs after a failed local rebuild attempt.
- Minor: `briefing/evaluation.md` did not list the newer `boundary_temporal_error` and `scale_drift_error` metrics.
- Minor: `scoring/EXPERIMENTS.md`, `scoring/TARGETS.md`, and `scoring/DIRECTION.md` included avoidable method-leaning wording for files that should be method-agnostic or method-safe.
- Minor: Generated CUDA cache files under `.nv/ComputeCache/` were tracked in git.
- Residual limitation: local login-node Python lacks the runtime packages needed to execute `scripts/evaluate.sh` directly, but the compute-container validation completed and `python validate.py --compare` passes on the finalized artifacts.

## Fixes Applied

- Restored canonical `scoring/scores.json` from the completed full-evaluator container output and rebuilt `scoring/metrics_detail.json` with matching method artifact paths.
- Updated `briefing/evaluation.md` to include all scored metrics and describe the benchmark target without exposing implementation details.
- Reworded `scoring/EXPERIMENTS.md`, `scoring/TARGETS.md`, and `scoring/DIRECTION.md` to keep the problem-facing documentation method-agnostic.
- Removed tracked `.nv/ComputeCache/` files and added `.nv/` to `.gitignore`.
- Verified the Hugging Face checkpoint URL in `scripts/download.sh` resolves to HTTP 200 after redirect.
- Ran `python validate.py --compare`; scoring and structure checks passed, including eval/method separation, no committed `shared/` references, train/test independence, method-key parity, and metric differentiation.

## Final Judgment

The workspace is ready to serve as a research gym. Improve-mode agents can run the reproduced method and compare against the canonical scores. From-scratch agents can use the method-agnostic problem, data, and evaluation interface without importing from `method/`. The scoring captures the central benefit, temporal consistency, while also tracking spatial accuracy and resource constraints.

- Milestone: core_claim_plus
- Ready for gym use: yes
- Confidence: high
- Key limitation: synthetic RGB-D benchmark and released-checkpoint inference only; full real-dataset training reproduction remains out of scope.
