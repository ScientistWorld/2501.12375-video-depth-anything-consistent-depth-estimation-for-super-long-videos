#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -euo pipefail

cd /home/user

experiment="${VDA_EXPERIMENT:-synthetic_long_video_128}"
video_dir="${VDA_VIDEO_DIR:-data/synthetic_video_128}"
artifact="${VDA_METHOD_ARTIFACT:-method/outputs/${experiment}/vda_oi_kr.npz}"

python data/synthetic_rgbd.py --output "$video_dir" --frames "${VDA_SYNTH_FRAMES:-128}" --height 112 --width 160 --seed 7
python method/run_vda.py \
  --video-dir "$video_dir" \
  --output "$artifact" \
  --checkpoint data/downloads/video_depth_anything_vits.pth \
  --mode oi_kr \
  --input-size "${VDA_INPUT_SIZE:-252}"

VDA_EXPERIMENT="$experiment" VDA_GT="$video_dir/ground_truth.npz" VDA_METHOD_ARTIFACT="$artifact" bash scripts/evaluate.sh vda_oi_kr
