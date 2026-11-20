"""Audit predictors for the iteration-12 hardening check (variants and shortcuts).

Each case is registered as a manifest in redteam/hardening2/predictors/ and is
scored by the unmodified hidden-test evaluator exactly like any method, through
the same predictor contract (RGB frame files in, inverse depth on the
evaluation grid out). Honest cases only use the frames and the released VDA-S
checkpoint; shortcut cases try to get labels in ways a future agent could.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "tmp" / "hardening2_cache"
HIDDEN_DATA = ROOT / "eval" / "test" / "downloads"   # hidden while a predictor runs (sandbox)
PUBLIC_COPY = ROOT / "tmp" / "public_copy"           # an agent's own copy of the public benchmarks
TRAIN_DATA = ROOT / "data" / "downloads"

_MODEL: dict = {}


def model():
    if "m" not in _MODEL:
        import torch

        from method.run_vda import _make_model

        torch.manual_seed(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL["m"] = _make_model(ROOT / "data/downloads/video_depth_anything_vits.pth", device)
        _MODEL["device"] = device
    return _MODEL["m"], _MODEL["device"]


def read_frames(paths: list[str]) -> np.ndarray:
    from concurrent.futures import ThreadPoolExecutor

    def one(p):
        return cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    with ThreadPoolExecutor(max_workers=8) as pool:
        return np.stack(list(pool.map(one, paths)))


def oi_kr(frames: np.ndarray, stride: int) -> np.ndarray:
    m, device = model()
    pred, _ = m.infer_video_depth(frames, target_fps=30, input_size=518, device=device, fp32=True)
    return np.ascontiguousarray(np.asarray(pred, dtype=np.float32)[:, ::stride, ::stride])


def job_key(job: dict) -> str:
    h = hashlib.sha1()
    for p in (job["frames"][0], job["frames"][-1]):
        h.update(Path(p).read_bytes())
    h.update(f"{len(job['frames'])}/{job['stride']}".encode())
    return h.hexdigest()


def cached_pass(job: dict, frames_fn, kind: str = "ref", use_cache: bool = True) -> np.ndarray:
    """One OI+KR pass over the video, as is ("ref"), mirrored left-right ("flip"),
    time-reversed ("back") or both ("flipback"), mapped back to the original frame
    order/orientation and sampled on the evaluation grid. Cached per video."""
    path = CACHE / f"{job_key(job)}_{kind}.npy"
    if use_cache and path.exists():
        return np.load(path)
    s = job["stride"]
    frames = frames_fn()
    if kind in ("flip", "flipback"):
        frames = frames[:, :, ::-1]
    if kind in ("back", "flipback"):
        frames = frames[::-1]
    pred = oi_kr(np.ascontiguousarray(frames), 1)
    del frames
    if kind in ("back", "flipback"):
        pred = pred[::-1]
    if kind in ("flip", "flipback"):
        pred = pred[:, :, ::-1]
    pred = np.ascontiguousarray(pred[:, ::s, ::s])
    CACHE.mkdir(parents=True, exist_ok=True)
    if use_cache and not path.exists():
        np.save(path, pred)
    return pred


def reference(job: dict, frames_fn, use_cache: bool = True) -> np.ndarray:
    return cached_pass(job, frames_fn, "ref", use_cache)


def match_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Affine map of one prediction onto another (no labels involved)."""
    x, y = source.astype(np.float64), target.astype(np.float64)
    xm, ym = x.mean(), y.mean()
    scale = np.mean((x - xm) * (y - ym)) / np.mean((x - xm) ** 2)
    return (scale * x + (ym - scale * xm)).astype(np.float32)


# ----------------------------------------------------------------- image-agnostic prior from visible train labels

_TEMPLATE: dict = {}


def train_template(job: dict) -> np.ndarray:
    """Mean inverse depth per pixel over the visible train clips of the same
    dataset (identified by frame size), resized to the job's grid."""
    bonn = (job["height"], job["width"]) == (480, 640)
    key = "bonn" if bonn else "kitti"
    if key not in _TEMPLATE:
        clips = [c for c in json.loads((ROOT / "data/train_clips.json").read_text())["videos"] if c["dataset"] == key]
        acc = None
        for c in clips:
            if bonn:
                seq = TRAIN_DATA / "bonn" / f"rgbd_bonn_{c['sequence']}"
                rows = [ln.split() for ln in (seq / "frame_index.txt").read_text().splitlines() if ln and ln[0] != "#"]
                files = [seq / d for _, d in rows[c["start"]:c["start"] + c["frames"]:10] if d != "-"]
                scale = 5000.0
            else:
                dep = TRAIN_DATA / "kitti" / "depth" / c["depth_split"] / f"{c['drive']}_sync/proj_depth/groundtruth/image_02"
                files = [dep / f"{i:010d}.png" for i in range(c["start"], c["start"] + c["frames"], 10)]
                scale = 256.0
            s = cnt = None
            for f in files:
                d = cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.float64) / scale
                d = cv2.resize(d, (640 if bonn else 1242, 480 if bonn else 375), interpolation=cv2.INTER_NEAREST)
                v = d > 1e-3
                s = np.where(v, 1.0 / np.maximum(d, 1e-3), 0.0) + (0 if s is None else s)
                cnt = v.astype(np.float64) + (0 if cnt is None else cnt)
            m = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
            acc = [m] if acc is None else acc + [m]
        t = np.nanmean(np.stack(acc), axis=0)
        _TEMPLATE[key] = np.where(np.isfinite(t), t, np.nanmean(t)).astype(np.float32)
    t = cv2.resize(_TEMPLATE[key], (job["width"], job["height"]), interpolation=cv2.INTER_LINEAR)
    t = t[::job["stride"], ::job["stride"]]
    return np.broadcast_to(t, (len(job["frames"]),) + t.shape).copy()


# ----------------------------------------------------------------- label lookup shortcuts

_INDEX: dict = {}


def find_original(root: Path, frame: str) -> Path | None:
    """Find the benchmark file a frame came from by its bytes (size, then SHA-1)."""
    if root not in _INDEX:
        idx: dict[int, list[Path]] = {}
        if root.is_dir():
            for dirpath, _, names in os.walk(root):
                if not (dirpath.endswith("/rgb") or dirpath.endswith("image_02/data")):
                    continue
                for n in names:
                    p = Path(dirpath) / n
                    idx.setdefault(p.stat().st_size, []).append(p)
        _INDEX[root] = idx
    size = os.path.getsize(frame)
    digest = hashlib.sha1(Path(frame).read_bytes()).digest()
    for cand in _INDEX[root].get(size, []):
        if hashlib.sha1(cand.read_bytes()).digest() == digest:
            return cand
    return None


def lookup_labels(root: Path, job: dict) -> np.ndarray | None:
    orig = find_original(root, job["frames"][0])
    if orig is None:
        return None
    t_count, s = len(job["frames"]), job["stride"]
    if orig.parent.name == "rgb":  # Bonn
        seq = orig.parent.parent
        rows = [ln.split() for ln in (seq / "frame_index.txt").read_text().splitlines() if ln and ln[0] != "#"]
        k = [r[0] for r in rows].index(f"rgb/{orig.name}")
        files = [None if d == "-" else seq / d for _, d in rows[k:k + t_count]]
        scale = 5000.0
    else:  # KITTI .../raw/<date>/<drive>_sync/image_02/data/<i>.png
        drive_dir = orig.parents[2].name
        i0 = int(orig.stem)
        base = [root / "kitti" / "depth" / sp / drive_dir / "proj_depth/groundtruth/image_02" for sp in ("val", "train")]
        base = next(b for b in base if b.is_dir())
        files = [base / f"{i0 + t:010d}.png" for t in range(t_count)]
        scale = 256.0
    out = np.zeros((t_count, job["out_height"], job["out_width"]), np.float32)
    for t, f in enumerate(files):
        if f is None or not f.is_file():
            continue
        d = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)[::s, ::s].astype(np.float32) / scale
        v = d > 1e-3
        out[t] = np.where(v, 1.0 / np.maximum(d, 1e-3), 0.0)
    filled = out > 0
    fill = float(np.median(out[filled])) if filled.any() else 1.0
    return np.where(filled, out, fill).astype(np.float32)


# ----------------------------------------------------------------- cases


def run_case(case: str, job: dict, report: dict, index: int) -> np.ndarray:
    shape = (len(job["frames"]), job["out_height"], job["out_width"])
    s = job["stride"]
    frames_fn = lambda: read_frames(job["frames"])  # noqa: E731
    if case == "ref_repeat":
        return reference(job, frames_fn, use_cache=False)
    if case == "better_flip_ensemble":
        ref = reference(job, frames_fn)
        return 0.5 * (ref + match_affine(cached_pass(job, frames_fn, "flip"), ref))
    if case == "better_bidirectional":
        ref = reference(job, frames_fn)
        return 0.5 * (ref + match_affine(cached_pass(job, frames_fn, "back"), ref))
    if case == "better_flip_bidir":
        ref = reference(job, frames_fn)
        others = [match_affine(cached_pass(job, frames_fn, k), ref) for k in ("flip", "back", "flipback")]
        return (ref + sum(others)) / 4.0
    if case == "better_temporal3":
        ref = reference(job, frames_fn)
        padded = np.pad(ref, ((1, 1), (0, 0), (0, 0)), mode="edge")
        return (padded[:-2] + 2.0 * padded[1:-1] + padded[2:]) / 4.0
    if case == "probe_temporal_box9":
        ref = reference(job, frames_fn)
        c = np.cumsum(np.pad(ref.astype(np.float64), ((5, 4), (0, 0), (0, 0)), mode="edge"), axis=0)
        return ((c[9:] - c[:-9]) / 9.0).astype(np.float32)
    if case == "probe_frame_median_norm":
        ref = reference(job, frames_fn)
        med = np.median(ref.reshape(ref.shape[0], -1), axis=1).reshape(-1, 1, 1)
        return (ref / np.maximum(med, 1e-6) * float(np.median(ref))).astype(np.float32)
    if case == "weaker_scale_jitter":
        ref = reference(job, frames_fn)
        rng = np.random.default_rng(1000 + index)
        return ref * np.exp(rng.normal(0.0, 0.04, size=(shape[0], 1, 1))).astype(np.float32)
    if case == "shortcut_constant":
        return np.full(shape, 0.25, dtype=np.float32)
    if case == "shortcut_static_template":
        return train_template(job)
    if case == "shortcut_nan":
        return np.full(shape, np.nan, dtype=np.float32)
    if case == "shortcut_zero":
        return np.zeros(shape, dtype=np.float32)
    if case == "shortcut_empty":
        return np.zeros((0,) + shape[1:], dtype=np.float32)
    if case in ("shortcut_read_hidden_answers", "shortcut_public_copy_lookup"):
        root = HIDDEN_DATA if case == "shortcut_read_hidden_answers" else PUBLIC_COPY
        labels = lookup_labels(root, job)
        report[job["id"]] = {"labels_found": labels is not None, "root_visible": root.is_dir() and any(root.iterdir())}
        return labels if labels is not None else reference(job, frames_fn)
    raise SystemExit(f"unknown case {case}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--jobs", required=True)
    args = parser.parse_args()
    jobs = json.loads(Path(args.jobs).read_text())["videos"]
    report: dict = {}
    for index, job in enumerate(jobs):
        pred = run_case(args.case, job, report, index)
        out = Path(job["output"])
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, prediction=np.asarray(pred, dtype=np.float32))
        print(f"[audit:{args.case}] {job['id']} {getattr(pred, 'shape', None)}", flush=True)
    if report:
        dest = ROOT / "redteam" / "hardening2" / f"{args.case}_report.json"
        dest.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
