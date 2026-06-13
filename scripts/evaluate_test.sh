#!/bin/bash
# Held-out test-slice artifact evaluator.
#
# Mirrors scripts/evaluate.sh's method-facing contract while defaulting to
# every method entry present in scoring/scores_train.json. The script reads
# existing prediction artifacts and writes scoring/scores_test.json.

set -euo pipefail

cd /home/user

mkdir -p scoring

python eval/test/evaluate.py \
  --output scoring/scores_test.json \
  --detail-output scoring/metrics_detail_test.json \
  --train-scores scoring/scores_train.json \
  --train-detail scoring/metrics_detail_train.json \
  "$@"
