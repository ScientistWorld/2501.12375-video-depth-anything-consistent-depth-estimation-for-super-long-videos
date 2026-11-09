# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-13 05:02] - core_claim_plus
- Ran the official Video Depth Anything small model checkpoint through the reproduced OI+KR long-video inference path and the independent-window baseline.
- Scored both methods on fixed 128-frame and 256-frame synthetic RGB-D videos with artifact-only evaluation.
- The OI+KR method improved temporal-gradient error on both settings while keeping model size and training-data scale fixed.
- `python validate.py --compare` passes, including method/eval/data separation checks.

### [2026-06-13 05:06] - core_claim_plus
- Audited the continuing workspace from a reset `core_claim` milestone.
- Confirmed the implementation uses the official Video Depth Anything model and OI+KR long-video inference rather than a surrogate.
- Fixed the scoring target wording so `absrel` is the primary paper-comparable metric, with temporal consistency retained as the key local benefit metric and spatial accuracy as a guardrail.
- Prepared a 518-input-size follow-up job to test whether the paper-resolution inference better preserves spatial accuracy.

### [2026-06-13 05:27] - core_claim_plus
- Reviewed failed job `6c4a8f31-83f`; inference and evaluation completed for both 518-input-size experiments, but validation failed because the new experiment names were missing from `reference.json`.
- Added reference and documentation entries for all four scored experiments.
- The 518-input-size runs again show OI+KR reducing temporal-gradient error at fixed model scale and training-data scale, with spatial metrics close to the independent-window baseline.

### [2026-06-13 05:40] - core_claim_plus
- Completed final verification of structure, scoring, documentation separation, and download URL reachability.
- Restored canonical score artifacts from completed full-evaluator container output and confirmed `python validate.py --compare` passes.
- Removed generated CUDA cache files from version control and documented the final gym-readiness judgment in `VERIFICATION.md`.

### [2026-06-13 05:47] - core_claim_plus
- Fixed the artifact evaluator so standalone runs replace `scores.json` by default instead of retaining stale experiments.
- Updated full-run scripts to opt into append mode only after clearing previous score artifacts.
- Submitted a CPU-only container test to verify score regeneration with the runtime dependencies available.

### [2026-06-13 05:51] - core_claim_plus
- Confirmed CPU test job `af51b2f2-70e` completed successfully in the compute container.
- Re-ran `python validate.py --compare` locally after the job; scoring, train/test, and separation checks pass.
- Final verification found no critical issues and preserved the `core_claim_plus` milestone.

### [2026-11-09 17:00] - core_claim_plus
- Hardened the gym: hidden test videos are rendered in `eval/test/` and every registered predictor runs on them. Added the paper's TAE metric and a validity gate, and removed the unverifiable resource metrics.
- The reproduced OI+KR inference (released VDA-S checkpoint) now beats independent windows on AbsRel, delta1, TAE and scale drift on 500- and 1500-frame videos, matching the paper's Table 5 direction.

## Stop Justification
- Completed at milestone `core_claim_plus`.
