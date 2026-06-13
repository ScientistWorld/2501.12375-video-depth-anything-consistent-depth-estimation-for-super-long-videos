#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -euo pipefail

cd /home/user

rm -f \
  scoring/scores.json \
  scoring/metrics_detail.json \
  scoring/scores_train.json \
  scoring/scores_test.json \
  scoring/metrics_detail_train.json \
  scoring/metrics_detail_test.json

for exp in \
  synthetic_long_video_128 \
  synthetic_long_video_256 \
  synthetic_long_video_128_input518 \
  synthetic_long_video_256_input518
do
  gt_exp="${exp%_input518}"
  VDA_EXPERIMENT="$exp" \
  VDA_GT="data/${gt_exp}/ground_truth.npz" \
  bash scripts/evaluate.sh vda_independent_windows vda_oi_kr
done

bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh

python - <<'PY'
import json
from pathlib import Path

train = json.loads(Path("scoring/scores_train.json").read_text())
test = json.loads(Path("scoring/scores_test.json").read_text())

train_exps = set(train.get("experiments", {}))
test_exps = set(test.get("experiments", {}))
if train_exps != test_exps:
    raise SystemExit(f"experiment mismatch: train={sorted(train_exps)} test={sorted(test_exps)}")

metric_count = 0
for exp in sorted(train_exps):
    train_methods = set(train["experiments"][exp].get("results", {}))
    test_methods = set(test["experiments"][exp].get("results", {}))
    if train_methods != test_methods:
        raise SystemExit(f"method mismatch for {exp}: train={sorted(train_methods)} test={sorted(test_methods)}")
    for scores in train["experiments"][exp]["results"].values():
        metric_count += len(scores)
    for scores in test["experiments"][exp]["results"].values():
        metric_count += len(scores)

print(f"Split scoring complete: {len(train_exps)} experiments, {metric_count} train+test metric values.")
PY

python validate.py --compare
