#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -euo pipefail

cd /home/user

rm -rf baseline/outputs method/outputs
rm -f scoring/scores.json scoring/metrics_detail.json

for frames in 128 256; do
  export VDA_SYNTH_FRAMES="$frames"
  export VDA_EXPERIMENT="synthetic_long_video_${frames}"
  export VDA_VIDEO_DIR="data/synthetic_video_${frames}"
  export VDA_BASELINE_ARTIFACT="baseline/outputs/${VDA_EXPERIMENT}/vda_independent_windows.npz"
  export VDA_METHOD_ARTIFACT="method/outputs/${VDA_EXPERIMENT}/vda_oi_kr.npz"
  export VDA_GT="${VDA_VIDEO_DIR}/ground_truth.npz"

  bash scripts/baseline.sh vda_independent_windows
  bash scripts/method.sh
  bash scripts/evaluate.sh vda_independent_windows vda_oi_kr
done
