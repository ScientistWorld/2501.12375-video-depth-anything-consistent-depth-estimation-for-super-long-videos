# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-13 04:50] - core_claim
- Ran the released VDA-S model through the official OI+KR long-video inference path on a 128-frame RGB-D benchmark.
- Compared against independent 32-frame window inference with the same checkpoint and model scale.
- OI+KR improved the local temporal-gradient consistency metric while preserving close spatial depth quality.

### [2026-06-13 04:56] - core_claim_plus
- Re-ran the full reproduction after fixing temporal metric scaling.
- Added and completed a 256-frame setting to test the long-video strategy over more inference windows.
- OI+KR improved temporal-gradient consistency on both 128-frame and 256-frame settings at the same model size and training-data scale.

## Stop Justification
- Stopping at `core_claim_plus`: the released VDA-S method and independent-window baseline run end-to-end, the core claim is supported on two synthetic long-video settings, and the reusable evaluation/package deliverables validate. Higher milestones would require downloading and preparing the paper's full real benchmark suite or retraining.
