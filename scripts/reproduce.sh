#!/bin/bash
# Reproduce the reference end to end (inference only; the released VDA-S
# checkpoint is used, nothing is retrained).
#
#   1. visible train clips  -> scoring/scores_train.json (+ scoring/scores.json)
#   2. hidden test clips    -> scoring/scores_test.json
#
# Both evaluators run every registered predictor
# (method/predictors/vda_oi_kr.json and
# baseline/predictors/vda_independent_windows.json) on real Bonn RGB-D Dynamic
# and KITTI clips and score the outputs. Requires scripts/download.sh first.

set -euo pipefail

cd /home/user

test -f data/downloads/video_depth_anything_vits.pth || {
  echo "Missing checkpoint; run scripts/download.sh on a node with internet." >&2
  exit 1
}

bash scripts/evaluate_train.sh
cp scoring/scores_train.json scoring/scores.json
cp scoring/metrics_detail_train.json scoring/metrics_detail.json
bash scripts/evaluate_test.sh
python validate.py --compare
