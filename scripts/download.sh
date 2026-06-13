#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0.12 GB
# Estimated disk usage after extraction: 0.12 GB
#
# This script writes only to workspace-local paths so a fresh clone can
# reproduce the same inputs without relying on machine-specific caches.

set -euo pipefail

cd /home/user

mkdir -p data/downloads

if [ ! -f data/downloads/video_depth_anything_vits.pth ]; then
  wget -nc https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth \
    -P data/downloads/
fi

echo "Checkpoint ready. Synthetic RGB-D videos are generated inside the container by scripts/baseline.sh and scripts/method.sh."
