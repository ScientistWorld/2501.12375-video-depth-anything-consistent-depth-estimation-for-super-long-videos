# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-13 04:50] - core_claim
- Ran the released VDA-S model through the official OI+KR long-video inference path on a 128-frame RGB-D benchmark.
- Compared against independent 32-frame window inference with the same checkpoint and model scale.
- OI+KR improved the local temporal-gradient consistency metric while preserving close spatial depth quality.

## Stop Justification
- Not stopped. A 256-frame follow-up job is being submitted to test the claim over additional windows.
