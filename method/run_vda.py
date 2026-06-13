"""Run Video Depth Anything and write portable prediction artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "method" / "official"
sys.path.insert(0, str(OFFICIAL))

from video_depth_anything.video_depth import INFER_LEN, VideoDepthAnything  # noqa: E402


def _load_frames(video_dir: Path) -> np.ndarray:
    frames = []
    for path in sorted((video_dir / "frames").glob("*.png")):
        frames.append(imageio.imread(path)[..., :3])
    if not frames:
        raise FileNotFoundError(f"No PNG frames found under {video_dir / 'frames'}")
    return np.stack(frames, axis=0)


def _make_model(checkpoint: Path, device: str) -> VideoDepthAnything:
    model = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def _infer_independent_windows(model: VideoDepthAnything, frames: np.ndarray, input_size: int, device: str, fp32: bool) -> np.ndarray:
    chunks = []
    for start in range(0, len(frames), INFER_LEN):
        chunk = frames[start : start + INFER_LEN]
        if len(chunk) < INFER_LEN:
            pad = np.repeat(chunk[-1:,...], INFER_LEN - len(chunk), axis=0)
            chunk_for_model = np.concatenate([chunk, pad], axis=0)
        else:
            chunk_for_model = chunk
        pred, _ = model.infer_video_depth(chunk_for_model, target_fps=12, input_size=input_size, device=device, fp32=fp32)
        chunks.append(pred[: len(chunk)])
    return np.concatenate(chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="data/synthetic_video")
    parser.add_argument("--output", default="method/outputs/vda_oi_kr.npz")
    parser.add_argument("--checkpoint", default="data/downloads/video_depth_anything_vits.pth")
    parser.add_argument("--mode", choices=["oi_kr", "independent_windows"], default="oi_kr")
    parser.add_argument("--input-size", type=int, default=252)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = _load_frames(ROOT / args.video_dir)
    model = _make_model(ROOT / args.checkpoint, device)
    if args.mode == "oi_kr":
        pred, fps = model.infer_video_depth(frames, target_fps=12, input_size=args.input_size, device=device, fp32=args.fp32)
    else:
        pred = _infer_independent_windows(model, frames, args.input_size, device, args.fp32)
        fps = 12

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, prediction=pred.astype(np.float32), fps=fps, frames=len(frames))
    with out.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "Video Depth Anything",
                "encoder": "vits",
                "mode": args.mode,
                "checkpoint": args.checkpoint,
                "input_size": args.input_size,
                "artifact": str(out),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
