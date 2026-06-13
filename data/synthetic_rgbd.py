"""Generate a small RGB-D video benchmark for video depth evaluation.

The benchmark is synthetic because the paper's full video-depth benchmark
depends on large external datasets. It preserves the relevant evaluation
phenomenon: a monocular video has coherent camera motion, non-uniform depth,
occlusions, and lighting/texture changes, so per-window scale drift is
measurable while global affine alignment remains meaningful.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _intersect_box(origin: np.ndarray, direction: np.ndarray, center: np.ndarray, half_size: np.ndarray):
    inv = 1.0 / np.where(np.abs(direction) < 1e-6, np.sign(direction) * 1e-6 + 1e-6, direction)
    t0 = (center - half_size - origin) * inv
    t1 = (center + half_size - origin) * inv
    tmin = np.maximum.reduce(np.minimum(t0, t1), axis=-1)
    tmax = np.minimum.reduce(np.maximum(t0, t1), axis=-1)
    hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin > 0.0)
    return np.where(hit, tmin, np.inf)


def generate_sequence(frames: int = 72, height: int = 112, width: int = 160, seed: int = 7):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    x = (xx - width / 2.0) / (0.85 * width)
    y = -(yy - height / 2.0) / (0.85 * width)
    rays = np.stack([x, y, np.ones_like(x)], axis=-1)
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)

    boxes = [
        (np.array([-0.62, -0.18, 3.10]), np.array([0.32, 0.35, 0.22]), np.array([205, 82, 54])),
        (np.array([0.15, -0.10, 4.05]), np.array([0.48, 0.48, 0.30]), np.array([70, 145, 210])),
        (np.array([0.78, 0.08, 5.10]), np.array([0.35, 0.55, 0.32]), np.array([230, 195, 80])),
    ]

    rgb = np.empty((frames, height, width, 3), dtype=np.uint8)
    depth = np.empty((frames, height, width), dtype=np.float32)

    for t in range(frames):
        phase = t / max(frames - 1, 1)
        cam = np.array([
            0.28 * np.sin(2 * np.pi * phase),
            0.05 * np.sin(4 * np.pi * phase),
            -0.42 + 0.82 * phase,
        ])
        wall_z = 6.1 + 0.15 * np.sin(2 * np.pi * phase)
        floor_y = -0.82
        ceiling_y = 0.92

        wall_t = (wall_z - cam[2]) / rays[..., 2]
        floor_t = (floor_y - cam[1]) / np.where(np.abs(rays[..., 1]) < 1e-6, -1e-6, rays[..., 1])
        ceil_t = (ceiling_y - cam[1]) / np.where(np.abs(rays[..., 1]) < 1e-6, 1e-6, rays[..., 1])
        floor_t = np.where(floor_t > 0, floor_t, np.inf)
        ceil_t = np.where(ceil_t > 0, ceil_t, np.inf)

        d = np.minimum.reduce([wall_t, floor_t, ceil_t])
        color = np.zeros((height, width, 3), dtype=np.float32)
        color[:] = np.array([96, 102, 112], dtype=np.float32)

        floor_mask = floor_t <= d
        ceil_mask = ceil_t <= d
        wall_mask = wall_t <= d
        color[floor_mask] = np.array([92, 120, 98])
        color[ceil_mask] = np.array([122, 126, 138])
        color[wall_mask] = np.array([116, 106, 128])

        world = cam + rays * d[..., None]
        checker = (((np.floor(world[..., 0] * 2.2) + np.floor(world[..., 2] * 2.2)) % 2) * 28) - 14
        color += checker[..., None]

        for base_center, half_size, box_color in boxes:
            center = base_center.copy()
            center[0] += 0.13 * np.sin(2 * np.pi * phase + base_center[2])
            t_box = _intersect_box(cam, rays, center, half_size)
            hit = t_box < d
            if np.any(hit):
                d[hit] = t_box[hit]
                p = cam + rays * t_box[..., None]
                stripes = ((np.floor((p[..., 0] + p[..., 1]) * 8.0) % 2) * 34) - 17
                color[hit] = box_color + stripes[hit, None]

        vignette = 1.0 - 0.45 * (x**2 + y**2)
        color = color * vignette[..., None]
        color += rng.normal(0.0, 2.0, size=color.shape)
        rgb[t] = np.clip(color, 0, 255).astype(np.uint8)
        depth[t] = d.astype(np.float32)

    return rgb, depth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic_video")
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--height", type=int, default=112)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out = Path(args.output)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rgb, depth = generate_sequence(args.frames, args.height, args.width, args.seed)

    for i, frame in enumerate(rgb):
        Image.fromarray(frame).save(frames_dir / f"{i:04d}.png")
    np.savez_compressed(out / "ground_truth.npz", depth=depth)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(out / "input.mp4", list(rgb), fps=12, macro_block_size=1)
    except ImportError:
        # The PNG frames and ground-truth archive are the benchmark contract.
        # MP4 writing is a convenience for visual inspection inside the container.
        pass
    with (out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"frames": args.frames, "height": args.height, "width": args.width, "fps": 12, "seed": args.seed},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
