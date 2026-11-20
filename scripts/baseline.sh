#!/bin/bash
# Run and score the independent-window baseline on the visible train clips.
#
#   bash scripts/baseline.sh [vda_independent_windows]
#
# The baseline is registered in baseline/predictors/vda_independent_windows.json:
# the same VDA-S checkpoint processes every 32-frame window on its own.

set -euo pipefail

cd /home/user

name="${1:-vda_independent_windows}"
if [ ! -f "baseline/predictors/${name}.json" ]; then
  echo "Unknown baseline: $name" >&2
  exit 2
fi

bash scripts/evaluate_train.sh --only "$name" --output "scoring/scores_${name}_train.json" \
  --detail-output "scoring/metrics_detail_${name}_train.json"
