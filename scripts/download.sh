#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container or
# submitting compute jobs. Idempotent: files already present are skipped.
# Needs only bash, wget and the Python 3 standard library.
#
#   1. Released Video Depth Anything Small checkpoint (Hugging Face, Apache-2.0)
#      -> data/downloads/video_depth_anything_vits.pth
#   2. Visible train clips (data/train_clips.json) -> data/downloads/{bonn,kitti}/
#   3. Hidden test clips (eval/test/clips.json)     -> eval/test/downloads/{bonn,kitti}/
#
# Clips are cut from the public Bonn RGB-D Dynamic dataset
# (https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/) and the KITTI raw +
# annotated depth benchmark (https://www.cvlibs.net/datasets/kitti/,
# CC BY-NC-SA 3.0). data/fetch_clips.py reads only the members each clip needs
# (RGB frames, depth maps, poses/OXTS, calibration) from the original zip
# archives with HTTP range requests and CRC-checks them.
#
# Estimated download size: ~0.1 GB checkpoint + ~6.3 GB train + ~15 GB test
# Estimated disk usage after download: ~22 GB (no archives are kept)

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

mkdir -p data/downloads eval/test/downloads

if [ ! -f data/downloads/video_depth_anything_vits.pth ]; then
  wget -nc https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth \
    -P data/downloads/
fi

python3 data/fetch_clips.py --manifest data/train_clips.json --dest data/downloads
python3 data/fetch_clips.py --manifest eval/test/clips.json --dest eval/test/downloads

echo "Checkpoint, train clips (data/downloads) and hidden test clips (eval/test/downloads) are ready."
