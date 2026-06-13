#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -euo pipefail

cd /home/user

# Rebuild the full local score suite from scratch. `reproduce.sh` runs the
# faster 252-input-size benchmarks and clears stale scores; the loop below adds
# the paper-default 518-input-size variants before validation.
bash scripts/reproduce.sh

for frames in 128 256; do
  export VDA_INPUT_SIZE=518
  export VDA_SYNTH_FRAMES="$frames"
  export VDA_EXPERIMENT="synthetic_long_video_${frames}_input518"
  export VDA_VIDEO_DIR="data/synthetic_video_${frames}"
  export VDA_BASELINE_ARTIFACT="baseline/outputs/${VDA_EXPERIMENT}/vda_independent_windows.npz"
  export VDA_METHOD_ARTIFACT="method/outputs/${VDA_EXPERIMENT}/vda_oi_kr.npz"
  export VDA_GT="${VDA_VIDEO_DIR}/ground_truth.npz"

  bash scripts/baseline.sh vda_independent_windows
  bash scripts/method.sh
  bash scripts/evaluate.sh vda_independent_windows vda_oi_kr
done

python validate.py --compare
