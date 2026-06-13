#!/bin/bash
# Run a baseline method from the paper.
#
# Baselines are simple reference methods the paper compares against. 
# Implement this only if applicable, i.e. the paper has baselines.
# You do not need to implement all baselines to begin — 
# just enough to validate that the evaluation pipeline works 
# and provide a reference point. You can decide to expand more baselines later.
# Each baseline must be actually run and scored through evaluate.sh
# so the numbers are verified, not self-reported.
# You can decide what this baseline script should do yourself.
#
# Usage:
#   bash scripts/baseline.sh <name>

set -euo pipefail

cd /home/user

name="${1:-vda_independent_windows}"
if [ "$name" != "vda_independent_windows" ]; then
  echo "Unknown baseline: $name" >&2
  exit 2
fi

python data/synthetic_rgbd.py --output data/synthetic_video --frames "${VDA_SYNTH_FRAMES:-128}" --height 112 --width 160 --seed 7
python method/run_vda.py \
  --video-dir data/synthetic_video \
  --output baseline/outputs/vda_independent_windows.npz \
  --checkpoint data/downloads/video_depth_anything_vits.pth \
  --mode independent_windows \
  --input-size "${VDA_INPUT_SIZE:-252}"

bash scripts/evaluate.sh vda_independent_windows
