# Train/Test Evaluation Design Notes

## What I Checked

- Read `eval/train/evaluate.py`, `eval/test/evaluate.py`, and `scripts/evaluate_{train,test}.sh` side by side.
- Confirmed the split evaluators are artifact-only: they read prediction `.npz` files and do not import or run `method/`.
- Confirmed `scripts/evaluate_test.sh` invokes `eval/test/` and writes `scoring/scores_test.json`.
- Confirmed `scripts/evaluate_train.sh` invokes `eval/train/` and writes `scoring/scores_train.json`.
- Checked for train/test cross-imports and obvious paper-agreement metric names.
- Ran `bash -n` on both split scripts.
- Confirmed `scoring/scores_train.json` and `scoring/scores_test.json` are absent.

## What Changed

- Added `boundary_temporal_error` to both split evaluators.
  - The previous block split measured temporal pairs only inside selected 16-frame blocks, which skipped pairs crossing block boundaries.
  - That was a weak spot for this paper because long-video stitching and key-frame referencing are specifically meant to avoid boundary flicker and scale discontinuities.
  - Train and test now score disjoint 32-frame boundary sets via `BOUNDARY_PARITY`.
- Added `scale_drift_error` to both split evaluators.
  - This catches long-video scale drift by comparing early-vs-late median log depth bias after global alignment.
  - It complements `absrel` and `temporal_gradient_error`, making it harder to optimize only local smoothness or only spatial accuracy.
- Changed split evaluators to read `param_count_M` and `video_training_frames_M` from a sidecar JSON when present, with the existing VDA-S values as defaults.
  - This makes resource constraint metrics responsive to future methods that provide metadata instead of being permanently fixed constants.
- Updated `scoring/reference.json` so every scored metric has `higher_is_better`, `coefficient`, and `description`.
- Added `boundary_temporal_error` and `scale_drift_error` metric entries to all existing experiments.

## Runtime Concerns For The Runner Iteration

- This was a source-only pass. The new metrics were not numerically executed here.
- `boundary_temporal_error` should be checked on all four experiment artifacts. It expects at least one valid 32-frame boundary pair per split; the existing 128- and 256-frame videos should satisfy this.
- `scale_drift_error` should be inspected for saturation. If it is near zero for both baseline and method, keep it as a shortcut guardrail but do not overweight it further.
- The sidecar metadata path is `prediction_path.with_suffix(".json")`. Existing artifacts already have sidecars, but future methods without sidecars will receive the VDA-S defaults and should be documented as self-reporting constraints.
- `scripts/evaluate.sh` remains the original full-score evaluator and was intentionally not edited in this design iteration. The new train/test metrics are implemented in `eval/train/` and `eval/test/`.
