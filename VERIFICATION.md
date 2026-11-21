# Verification Report

## Initial Assessment

The workspace entered final verification at `core_claim_plus`, and the claim remains justified. The reproduction uses the paper's released small checkpoint and long-video inference path, compares it against an independent-window reference, and evaluates prediction artifacts only. Completed runs cover four fixed synthetic long-video settings: 128 and 256 frames at both fast and 518-pixel input sizes.

The workspace was already close to ready. A follow-up CPU container job had been submitted to verify the score-regeneration fix; that job has now completed successfully.

## Issues Found

- Minor: `briefing/evaluation.md` described `scale_drift_error` using stale wording that did not exactly match the evaluator.
- Minor: `scoring/TARGETS.md` and the train/test evaluator docstrings contained avoidable method-specific phrasing.
- Minor: `MILESTONES.md`, `PROGRESS.md`, `VERIFICATION.md`, and `action.yaml` still reflected the pre-completion state of CPU test job `af51b2f2-70e`.
- No critical scoring, separation, download, or train/test issues were found.

## Fixes Applied

- Clarified the `scale_drift_error` definition to match the implemented metric.
- Reworded target/evaluator text to keep problem-facing documentation method-safe.
- Updated progress, milestone, and verification records to reflect the completed CPU container test.
- Verified the checkpoint URL in `scripts/download.sh` resolves to HTTP 200 after redirect.
- Ran `python validate.py --compare`; scoring and structure checks passed, including eval/method separation, no committed `shared/` references, train/test independence, method-key parity, and metric differentiation.

## Final Judgment

The workspace is ready to serve as a research gym. Improve-mode agents can run the reproduced method and compare against the canonical scores. From-scratch agents can use the method-agnostic problem, data, and evaluation interface without importing from `method/`. The scoring captures temporal consistency, spatial depth quality, and fixed resource constraints.

- Milestone: core_claim_plus
- Ready for gym use: yes
- Confidence: high
- Key limitation: synthetic RGB-D benchmark and released-checkpoint inference only; full real-dataset training reproduction remains out of scope.

## Iteration 11 note

The benchmark described above was replaced in iteration 11 (hidden rendered long videos, predictor contract, TAE, validity gate). See `HARDENING.md` for the current design and its verification.

## Iteration 12 note

The hidden test and the train split now use real Bonn RGB-D Dynamic and KITTI long videos instead of rendered corridors. Other changes:
- Hidden-test predictors run in a namespace sandbox.
- One test clip with a GPS glitch was replaced.

`python validate.py` passes on the final score files from job `7d38514d-b5f`. See `HARDENING.md` for the design, the audit and the remaining limitations.
