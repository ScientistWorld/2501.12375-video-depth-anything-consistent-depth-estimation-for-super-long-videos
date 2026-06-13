#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -euo pipefail

cd /home/user

python data/synthetic_rgbd.py --output data/synthetic_video --frames 72 --height 112 --width 160 --seed 7
python method/run_vda.py \
  --video-dir data/synthetic_video \
  --output method/outputs/vda_oi_kr.npz \
  --checkpoint data/downloads/video_depth_anything_vits.pth \
  --mode oi_kr \
  --input-size "${VDA_INPUT_SIZE:-252}"

bash scripts/evaluate.sh
