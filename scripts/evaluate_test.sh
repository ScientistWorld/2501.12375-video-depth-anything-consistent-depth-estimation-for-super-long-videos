#!/bin/bash
# Hidden test evaluator. Run with no arguments:
#
#   bash scripts/evaluate_test.sh
#
# Renders the hidden test videos inside eval/test/, runs every predictor
# registered under method/predictors/*.json and baseline/predictors/*.json on
# their RGB frames, recomputes every metric from the raw predictions, and writes
# scoring/scores_test.json (always from scratch) plus
# scoring/metrics_detail_test.json. Needs no training; inference only.

set -euo pipefail

cd /home/user

mkdir -p scoring

python eval/test/evaluate.py \
  --output scoring/scores_test.json \
  --detail-output scoring/metrics_detail_test.json \
  "$@"
