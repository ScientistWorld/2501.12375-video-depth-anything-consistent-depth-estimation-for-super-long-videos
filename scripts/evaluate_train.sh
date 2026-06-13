#!/bin/bash
# Visible train-slice artifact evaluator.
#
# Mirrors scripts/evaluate.sh's method-facing contract:
#   bash scripts/evaluate_train.sh
#   bash scripts/evaluate_train.sh method_name
#   bash scripts/evaluate_train.sh custom_method=path/to/predictions.npz
#
# The script reads existing prediction artifacts and writes
# scoring/scores_train.json. It does not run or import method code.

set -euo pipefail

cd /home/user

mkdir -p scoring

python eval/train/evaluate.py \
  --output scoring/scores_train.json \
  --detail-output scoring/metrics_detail_train.json \
  "$@"
