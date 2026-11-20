#!/bin/bash
# Hidden test evaluator. Run with no arguments:
#
#   bash scripts/evaluate_test.sh
#
# Runs every predictor registered under method/predictors/*.json and
# baseline/predictors/*.json on the RGB frames of the hidden test clips (real
# Bonn RGB-D Dynamic and KITTI videos held only in eval/test/), recomputes every
# metric from the raw predictions, and writes scoring/scores_test.json (always
# from scratch) plus scoring/metrics_detail_test.json. Inference only; needs a
# GPU in practice and the data from scripts/download.sh.

set -euo pipefail

cd /home/user

mkdir -p scoring

python eval/test/evaluate.py \
  --output scoring/scores_test.json \
  --detail-output scoring/metrics_detail_test.json \
  "$@"
