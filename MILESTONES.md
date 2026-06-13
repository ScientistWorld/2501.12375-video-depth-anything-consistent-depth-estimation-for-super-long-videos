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
- Fixed the scoring target wording so the primary local metric is temporal consistency, with spatial accuracy retained as a constraint.
- Prepared a 518-input-size follow-up job to test whether the paper-resolution inference better preserves spatial accuracy.

### [2026-06-13 05:27] - core_claim_plus
- Reviewed failed job `6c4a8f31-83f`; inference and evaluation completed for both 518-input-size experiments, but validation failed because the new experiment names were missing from `reference.json`.
- Added reference and documentation entries for all four scored experiments.
- The 518-input-size runs again show OI+KR reducing temporal-gradient error at fixed model scale and training-data scale, with spatial metrics close to the independent-window baseline.

## Stop Justification
- Completed at milestone `core_claim_plus`.
