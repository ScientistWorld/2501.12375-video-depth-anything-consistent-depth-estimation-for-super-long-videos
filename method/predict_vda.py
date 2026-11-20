"""Predictor entry point for Video Depth Anything (VDA-S).

Implements the benchmark's predictor contract (see briefing/evaluation.md):

    python method/predict_vda.py --mode oi_kr --input-size 518 --jobs <jobs.json>

``jobs.json`` lists videos as ``{"videos": [{"id", "frames", "fps", "stride",
"out_height", "out_width", "output", ...}]}`` where ``frames`` is the ordered
list of the video's RGB frame files (PNG). For each video this writes
``output`` (``.npz``) with a float32 ``prediction`` array
``[T, out_height, out_width]`` of relative inverse depth: the full-resolution
prediction sampled at every ``stride``-th row and column.

Modes:
  oi_kr                the paper's long-video inference (overlap interpolation +
                       key-frame referencing), via the released implementation
  independent_windows  each 32-frame window is processed on its own and the
                       window outputs are concatenated (no cross-window
                       stabilisation)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from method.run_vda import _infer_independent_windows, _make_model  # noqa: E402


def read_frames(paths: list[str]) -> np.ndarray:
    def one(p):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(p)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with ThreadPoolExecutor(max_workers=8) as pool:
        return np.stack(list(pool.map(one, paths)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--mode", choices=["oi_kr", "independent_windows"], default="oi_kr")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--checkpoint", default="data/downloads/video_depth_anything_vits.pth")
    parser.add_argument("--fp16", action="store_true", help="use autocast (default is fp32 for reproducibility)")
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp32 = not args.fp16
    model = _make_model(ROOT / args.checkpoint, device)

    jobs = json.loads(Path(args.jobs).read_text())["videos"]
    for job in jobs:
        frames = read_frames(job["frames"])
        if args.mode == "oi_kr":
            pred, _ = model.infer_video_depth(
                frames, target_fps=job.get("fps", 30), input_size=args.input_size, device=device, fp32=fp32
            )
        else:
            pred = _infer_independent_windows(model, frames, args.input_size, device, fp32)
        s = int(job.get("stride", 1))
        pred = np.ascontiguousarray(np.asarray(pred, dtype=np.float32)[:, ::s, ::s])
        del frames
        out = Path(job["output"])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, prediction=pred)
        print(f"[predict_vda:{args.mode}] {job.get('id', out.stem)}: {pred.shape}", flush=True)


if __name__ == "__main__":
    main()
