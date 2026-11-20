#!/bin/bash
# Standard evaluation entry point: scores every registered predictor on the
# visible train clips and writes scoring/scores.json.
#
#   bash scripts/evaluate.sh                 # every predictor on disk
#   bash scripts/evaluate.sh --only NAME     # a single predictor
#
# A predictor is any command registered with a JSON manifest in
# method/predictors/ or baseline/predictors/ (contract: briefing/evaluation.md).
# Evaluation code lives in eval/ and never imports method/.

set -euo pipefail

cd /home/user

mkdir -p scoring

python eval/train/evaluate.py \
  --output scoring/scores.json \
  --detail-output scoring/metrics_detail.json \
  "$@"
