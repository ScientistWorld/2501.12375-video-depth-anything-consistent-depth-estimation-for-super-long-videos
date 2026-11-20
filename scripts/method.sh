#!/bin/bash
# Run and score the paper's method (Video Depth Anything, OI+KR long-video
# inference with the released VDA-S checkpoint) on the visible train clips.
#
# The method is registered as a predictor in method/predictors/vda_oi_kr.json;
# the evaluator hands it the RGB frames, calls method/predict_vda.py and scores it.

set -euo pipefail

cd /home/user

bash scripts/evaluate_train.sh --only vda_oi_kr --output scoring/scores_method_train.json \
  --detail-output scoring/metrics_detail_method_train.json
