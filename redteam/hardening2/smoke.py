"""Fast pre-flight for the iteration-12 job (no predictors): every test clip is on
disk, camera poses are consistent with the ground-truth depth (a ground-truth
depth map reprojected with the true relative pose matches the next frame far
better than with no motion), and the metrics give the exact-label values.
Aborts the job early on any error."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gt_reproj(ev, gt: np.ndarray, K: np.ndarray, rel: np.ndarray, max_depth: float) -> float:
    """Median AbsRel of ground-truth depth of frame a reprojected into frame b."""
    a, b = gt
    v, u = np.nonzero((a > 1e-3) & (a < max_depth))
    z = a[v, u].astype(np.float64)
    x, y = (u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z
    p = rel[:3, :3] @ np.stack([x, y, z]) + rel[:3, 3:4]
    ok = p[2] > 0.1
    u2 = np.round(K[0, 0] * p[0][ok] / p[2][ok] + K[0, 2]).astype(int)
    v2 = np.round(K[1, 1] * p[1][ok] / p[2][ok] + K[1, 2]).astype(int)
    zp = p[2][ok]
    h, w = b.shape
    inb = (u2 >= 0) & (u2 < w) & (v2 >= 0) & (v2 < h)
    t = b[v2[inb], u2[inb]]
    m = (t > 1e-3) & (t < max_depth)
    if not m.any():
        return float("nan")  # a frame without depth (a few Bonn frames have none)
    return float(np.median(np.abs(t[m] - zp[inb][m]) / t[m]))


def main() -> None:
    report = {}
    for split, path in (("test", ROOT / "eval/test/evaluate.py"), ("train", ROOT / "eval/train/evaluate.py")):
        ev = load(path, f"ev_{split}")
        videos = ev.load_videos()
        print(f"[smoke] {split}: {len(videos)} videos, all files present", flush=True)
        for v in videos if split == "test" else []:
            gt = ev.load_depth(v)
            T = gt.shape[0]
            res = []
            for t in range(0, T - 3, max(1, T // 6)):
                rel = np.linalg.inv(v["poses"][t + 3]) @ v["poses"][t]
                res.append((gt_reproj(ev, gt[[t, t + 3]], v["K_eval"], rel, v["max_depth"]),
                            gt_reproj(ev, gt[[t, t + 3]], v["K_eval"], np.eye(4), v["max_depth"])))
            res = [r for r in res if np.isfinite(r[0]) and np.isfinite(r[1])]
            pose_err, still_err = float(np.mean([r[0] for r in res])), float(np.mean([r[1] for r in res]))
            exact = np.where(gt > 1e-3, 1.0 / np.maximum(gt, 1e-3), 0.0).astype(np.float32)
            sc = ev.score_video(exact, gt, v, temporal=False)
            report[v["clip"]] = {"gt_reproj_true_pose": pose_err, "gt_reproj_no_motion": still_err,
                                 "exact_absrel": sc["absrel"], "exact_delta1": sc["delta1"],
                                 "valid_fraction": float(((gt > 1e-3) & (gt < v["max_depth"])).mean()),
                                 "shape": list(gt.shape)}
            print(f"[smoke] {v['clip']}: {json.dumps(report[v['clip']])}", flush=True)
            assert sc["absrel"] < 1e-3 and sc["delta1"] > 0.999, "exact labels must score perfectly"
    sys.path.insert(0, str(ROOT / "redteam/hardening2"))
    from oxts_scan import scan
    report["_oxts_inconsistent_steps"] = {k: len(v) for k, v in
                                          scan(str(ROOT / "eval/test/downloads"), str(ROOT / "eval/test/clips.json")).items()}
    (ROOT / "redteam/hardening2/smoke.json").write_text(json.dumps(report, indent=1))
    worse = [k for k, r in report.items() if not k.startswith("_") and not r["gt_reproj_true_pose"] < r["gt_reproj_no_motion"]]
    print(f"[smoke] clips where the true pose does not beat no-motion: {worse}", flush=True)
    print(f"[smoke] cuda={torch.cuda.is_available()}", flush=True)


if __name__ == "__main__":
    main()
