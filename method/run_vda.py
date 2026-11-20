"""Video Depth Anything helpers shared by the predictors in method/.

``_make_model`` loads the released VDA-S checkpoint into the official
implementation (method/official). ``_infer_independent_windows`` runs the same
network on consecutive 32-frame windows with no overlap, key frames or
alignment between them (the paper's "Baseline" inference strategy in Table 5).
The paper's own long-video strategy (overlap interpolation + key-frame
referencing) is ``VideoDepthAnything.infer_video_depth`` in the official code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "method" / "official"
sys.path.insert(0, str(OFFICIAL))

from video_depth_anything.video_depth import INFER_LEN, VideoDepthAnything  # noqa: E402


def _make_model(checkpoint: Path, device: str) -> VideoDepthAnything:
    model = VideoDepthAnything(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def _infer_independent_windows(model: VideoDepthAnything, frames: np.ndarray, input_size: int, device: str,
                               fp32: bool) -> np.ndarray:
    chunks = []
    for start in range(0, len(frames), INFER_LEN):
        chunk = frames[start: start + INFER_LEN]
        if len(chunk) < INFER_LEN:
            pad = np.repeat(chunk[-1:, ...], INFER_LEN - len(chunk), axis=0)
            chunk_for_model = np.concatenate([chunk, pad], axis=0)
        else:
            chunk_for_model = chunk
        pred, _ = model.infer_video_depth(chunk_for_model, target_fps=30, input_size=input_size, device=device,
                                          fp32=fp32)
        chunks.append(pred[: len(chunk)])
    return np.concatenate(chunks, axis=0)
