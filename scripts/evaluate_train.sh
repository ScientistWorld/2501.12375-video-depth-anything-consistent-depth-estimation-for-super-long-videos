#!/bin/bash
# Visible train evaluator, same predictor contract and metrics as the test.
#
#   bash scripts/evaluate_train.sh                 # every registered predictor
#   bash scripts/evaluate_train.sh --only NAME     # one predictor
#   bash scripts/evaluate_train.sh --predictor-root my/extra/manifests
#
# Renders the visible train videos with data/synthetic_rgbd.py, runs every
# predictor registered under method/predictors/*.json and
# baseline/predictors/*.json, and writes scoring/scores_train.json plus
# scoring/metrics_detail_train.json (always from scratch).

set -euo pipefail

cd /home/user

mkdir -p scoring

python eval/train/evaluate.py \
  --output scoring/scores_train.json \
  --detail-output scoring/metrics_detail_train.json \
  "$@"
