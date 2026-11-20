#!/bin/bash
# Visible train evaluator, same predictor contract and metrics as the test.
#
#   bash scripts/evaluate_train.sh                 # every registered predictor
#   bash scripts/evaluate_train.sh --only NAME     # one predictor
#   bash scripts/evaluate_train.sh --predictor-root my/extra/manifests
#
# Runs every predictor registered under method/predictors/*.json and
# baseline/predictors/*.json on the visible train clips (data/train_clips.json,
# real Bonn RGB-D Dynamic and KITTI videos in data/downloads/) and writes
# scoring/scores_train.json plus scoring/metrics_detail_train.json (always from
# scratch).

set -euo pipefail

cd /home/user

mkdir -p scoring

python eval/train/evaluate.py \
  --output scoring/scores_train.json \
  --detail-output scoring/metrics_detail_train.json \
  "$@"
