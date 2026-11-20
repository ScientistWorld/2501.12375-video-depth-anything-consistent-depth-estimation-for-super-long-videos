"""Visible train evaluator for long-video depth estimation on real videos.

Same predictor contract, metrics, alignment, validity gate and output format as
the hidden test, on different clips of the same two public benchmarks (Bonn
RGB-D Dynamic, indoor with moving people; KITTI, outdoor driving). The clip list
is ``data/train_clips.json``; ``scripts/download.sh`` fetches the frames, depth
maps, poses and calibration into ``data/downloads/``.

Every predictor registered by a JSON manifest ``<root>/<name>.json`` (default
roots ``method/predictors`` and ``baseline/predictors``) is run as
``command + ["--jobs", jobs.json]`` from the workspace root. It receives only the
RGB frames (hard links in a scratch directory) and must write one ``.npz`` per
video with a ``prediction`` array (contract in briefing/evaluation.md).

Every number in the output is recomputed here from the raw predictions; the
output file is always written from scratch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data" / "downloads"
CLIPS = ROOT / "data" / "train_clips.json"
DEFAULT_ROOTS = ("method/predictors", "baseline/predictors")

# Fixed roster: experiment -> (dataset, number of videos, frames per video).
EXPERIMENTS = {
    "bonn_500": ("bonn", 6, 500),
    "kitti_500": ("kitti", 4, 500),
    "kitti_1500": ("kitti", 2, 1500),
}
METRICS = ("absrel", "delta1", "tae", "scale_drift")
DATASETS = {
    # Official benchmark depth ranges (Video Depth Anything benchmark/eval/eval.py).
    "bonn": {"max_depth": 10.0, "fps": 30},
    "kitti": {"max_depth": 80.0, "fps": 10},
}
MIN_DEPTH = 1e-3
STRIDE = 2  # evaluation grid: every STRIDE-th row and column of the native frame
# Validity caps: a run whose mean TAE or scale drift reaches these does not
# produce usable video depth and is scored as failed.
TAE_CAP = 30.0
DRIFT_CAP = 50.0

# Bonn RGB-D Dynamic calibration (dataset web page): RGB intrinsics, the
# motion-capture-marker-to-sensor transform T_m and the ROS axis fix T_ROS;
# camera-to-world pose = T_ROS^-1 * T_k * T_ROS * T_m with T_k from groundtruth.txt.
BONN_K = np.array([[542.822841, 0.0, 315.593520], [0.0, 542.576870, 237.756098], [0.0, 0.0, 1.0]])
BONN_TM = np.array([[1.0157, 0.1828, -0.2389, 0.0113], [0.0009, -0.8431, -0.6413, -0.0098],
                    [-0.3009, 0.6147, -0.8085, 0.0111], [0.0, 0.0, 0.0, 1.0]])
BONN_TROS = np.array([[-1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])


# --------------------------------------------------------------------------- data


def _orthonormal(m: np.ndarray) -> np.ndarray:
    """Gram-Schmidt on the columns of the (slightly non-rigid) calibrated rotation."""
    c0 = m[:, 0] / np.linalg.norm(m[:, 0])
    c1 = m[:, 1] - (m[:, 1] @ c0) * c0
    c1 /= np.linalg.norm(c1)
    return np.stack([c0, c1, np.cross(c0, c1)], axis=1)


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - a) * th) * q0 + math.sin(a * th) * q1) / math.sin(th)


def _bonn_clip(clip: dict) -> dict:
    seq = DATA / "bonn" / f"rgbd_bonn_{clip['sequence']}"
    rows = [ln.split() for ln in (seq / "frame_index.txt").read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    rows = rows[clip["start"]:clip["start"] + clip["frames"]]
    if len(rows) != clip["frames"]:
        raise RuntimeError(f"{seq}: expected {clip['frames']} frames, found {len(rows)}")
    gt = np.array([[float(x) for x in ln.split()] for ln in (seq / "groundtruth.txt").read_text().splitlines()
                   if ln.strip() and not ln.startswith("#")])
    tm = BONN_TM.copy()
    tm[:3, :3] = _orthonormal(BONN_TM[:3, :3])
    tros_inv = np.linalg.inv(BONN_TROS)
    poses = []
    for rgb_name, _ in rows:
        t = float(Path(rgb_name).stem)
        j = int(np.clip(np.searchsorted(gt[:, 0], t), 1, len(gt) - 1))
        t0, t1 = gt[j - 1, 0], gt[j, 0]
        a = float(np.clip((t - t0) / max(t1 - t0, 1e-9), 0.0, 1.0))
        tk = np.eye(4)
        tk[:3, :3] = _quat_to_rot(_slerp(gt[j - 1, 4:8], gt[j, 4:8], a))
        tk[:3, 3] = (1 - a) * gt[j - 1, 1:4] + a * gt[j, 1:4]
        poses.append(tros_inv @ tk @ BONN_TROS @ tm)
    return {"rgb": [seq / r for r, _ in rows], "depth": [None if d == "-" else seq / d for _, d in rows],
            "depth_scale": 5000.0, "K": BONN_K.copy(), "poses": np.stack(poses)}


def _kitti_calib(date: str) -> tuple[np.ndarray, np.ndarray]:
    def read(name):
        out = {}
        for ln in (DATA / "kitti" / "raw" / date / name).read_text().splitlines():
            if ":" in ln:
                k, v = ln.split(":", 1)
                try:
                    out[k] = np.array([float(x) for x in v.split()])
                except ValueError:
                    pass
        return out

    def rt(r, t):
        m = np.eye(4)
        m[:3, :3] = r.reshape(3, 3)
        m[:3, 3] = t
        return m

    c2c, i2v, v2c = read("calib_cam_to_cam.txt"), read("calib_imu_to_velo.txt"), read("calib_velo_to_cam.txt")
    p2 = c2c["P_rect_02"].reshape(3, 4)
    rect = np.eye(4)
    rect[:3, :3] = c2c["R_rect_00"].reshape(3, 3)
    t2 = np.eye(4)
    t2[0, 3] = p2[0, 3] / p2[0, 0]
    t_cam2_imu = t2 @ rect @ rt(v2c["R"], v2c["T"]) @ rt(i2v["R"], i2v["T"])
    return p2[:3, :3].copy(), t_cam2_imu


def _kitti_clip(clip: dict) -> dict:
    drive, date = clip["drive"], clip["drive"][:10]
    raw = DATA / "kitti" / "raw" / date / f"{drive}_sync"
    dep = DATA / "kitti" / "depth" / clip["depth_split"] / f"{drive}_sync" / "proj_depth" / "groundtruth" / "image_02"
    idx = range(clip["start"], clip["start"] + clip["frames"])
    K, t_cam2_imu = _kitti_calib(date)
    oxts = [np.array([float(x) for x in (raw / "oxts" / "data" / f"{i:010d}.txt").read_text().split()]) for i in idx]
    scale = math.cos(oxts[0][0] * math.pi / 180.0)
    er = 6378137.0
    inv_c = np.linalg.inv(t_cam2_imu)
    poses = []
    for o in oxts:
        lat, lon, alt, roll, pitch, yaw = o[:6]
        rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
        ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
        rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        t_w_imu = np.eye(4)
        t_w_imu[:3, :3] = rz @ ry @ rx
        t_w_imu[:3, 3] = [scale * lon * math.pi * er / 180.0,
                          scale * er * math.log(math.tan((90.0 + lat) * math.pi / 360.0)), alt]
        poses.append(t_w_imu @ inv_c)
    return {"rgb": [raw / "image_02" / "data" / f"{i:010d}.png" for i in idx],
            "depth": [dep / f"{i:010d}.png" for i in idx], "depth_scale": 256.0, "K": K, "poses": np.stack(poses)}


def load_videos() -> list[dict]:
    clips = json.loads(CLIPS.read_text())["videos"]
    videos = []
    for exp, (dataset, count, frames) in EXPERIMENTS.items():
        cs = [c for c in clips if c["experiment"] == exp]
        if len(cs) != count or any(c["dataset"] != dataset or c["frames"] != frames for c in cs):
            raise SystemExit(f"clips.json does not match the fixed roster for {exp}")
        for c in cs:
            v = _bonn_clip(c) if dataset == "bonn" else _kitti_clip(c)
            missing = [p for p in v["rgb"] + [d for d in v["depth"] if d is not None] if not p.is_file()]
            if missing:
                raise SystemExit(f"missing train data ({len(missing)} files, e.g. {missing[0]}); run scripts/download.sh")
            h, w = cv2.imread(str(v["rgb"][0]), cv2.IMREAD_UNCHANGED).shape[:2]
            ks = v["K"].copy()
            ks[:2] /= STRIDE
            v.update({"experiment": exp, "dataset": dataset, "clip": c["id"], "id": f"video_{len(videos):03d}",
                      "height": h, "width": w, "out_height": -(-h // STRIDE), "out_width": -(-w // STRIDE),
                      "K_eval": ks, "max_depth": DATASETS[dataset]["max_depth"], "fps": DATASETS[dataset]["fps"]})
            videos.append(v)
    return videos


def load_depth(video: dict) -> np.ndarray:
    """Ground-truth depth on the evaluation grid, metres, 0 where unknown."""
    shape = (video["out_height"], video["out_width"])

    def one(path):
        if path is None:
            return np.zeros(shape, np.float32)
        d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if d is None or d.ndim != 2:
            raise RuntimeError(f"unreadable depth map {path}")
        d = d[::STRIDE, ::STRIDE].astype(np.float32) / video["depth_scale"]
        if d.shape != shape:
            raise RuntimeError(f"depth map {path} has shape {d.shape}, expected {shape}")
        return d

    with ThreadPoolExecutor(max_workers=8) as pool:
        return np.stack(list(pool.map(one, video["depth"])))


# --------------------------------------------------------------------------- metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _align(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> tuple[float, float]:
    """Least-squares scale/shift of the predicted inverse depth (clipped at 1e-3) to
    ground-truth inverse depth, one pair for the whole video (paper protocol)."""
    x = pred[valid].double().clamp_min(MIN_DEPTH)
    y = 1.0 / (gt[valid].double() + 1e-8)
    n = float(x.numel())
    if n == 0:
        raise ValueError("no valid ground-truth pixels")
    sx, sy, sxx, sxy = float(x.sum()), float(y.sum()), float((x * x).sum()), float((x * y).sum())
    det = sxx * n - sx * sx
    if det <= 1e-12 * max(sxx * n, 1e-300):
        return 0.0, sy / n  # constant prediction: best constant fit
    return (n * sxy - sx * sy) / det, (sxx * sy - sx * sxy) / det


_GRID: dict = {}


def _reproj_absrel(src: torch.Tensor, dst: torch.Tensor, rel: torch.Tensor, K: np.ndarray) -> float:
    """One TAE term: AbsRel between dst depth and src depth reprojected into dst
    (nearest pixel, closest surface wins)."""
    h, w = src.shape
    if (h, w) not in _GRID:
        vs, us = torch.meshgrid(torch.arange(h, device=src.device, dtype=torch.float64),
                                torch.arange(w, device=src.device, dtype=torch.float64), indexing="ij")
        _GRID[(h, w)] = (us.reshape(-1), vs.reshape(-1))
    us, vs = _GRID[(h, w)]
    z = src.double().reshape(-1)
    x = (us - K[0, 2]) / K[0, 0] * z
    y = (vs - K[1, 2]) / K[1, 1] * z
    p = rel[:3, :3] @ torch.stack([x, y, z]) + rel[:3, 3:4]
    keep = p[2] > 1e-6
    zp = p[2][keep]
    u2 = torch.round(K[0, 0] * p[0][keep] / zp + K[0, 2]).long()
    v2 = torch.round(K[1, 1] * p[1][keep] / zp + K[1, 2]).long()
    inb = (u2 >= 0) & (u2 < w) & (v2 >= 0) & (v2 < h)
    if not bool(inb.any()):
        return 0.0
    proj = torch.full((h * w,), float("inf"), dtype=torch.float64, device=src.device)
    proj.scatter_reduce_(0, v2[inb] * w + u2[inb], zp[inb], reduce="amin")
    tgt = dst.double().reshape(-1)
    m = torch.isfinite(proj) & (tgt > 0)
    if not bool(m.any()):
        return 0.0
    return float((torch.abs(tgt[m] - proj[m]) / tgt[m]).mean())


def score_video(pred_np: np.ndarray, gt_np: np.ndarray, video: dict, temporal: bool = True) -> dict:
    max_depth = video["max_depth"]
    pred = torch.from_numpy(np.ascontiguousarray(pred_np, dtype=np.float32)).to(DEVICE)
    gt = torch.from_numpy(gt_np).to(DEVICE)
    valid = (gt > MIN_DEPTH) & (gt < max_depth)
    scale, shift = _align(pred, gt, valid)
    absrel, delta1, bias = [], [], []
    tae_sum = 0.0
    P = video["poses"]
    rel = [None] + [torch.from_numpy(np.linalg.inv(P[t]) @ P[t - 1]).to(DEVICE) for t in range(1, len(P))]
    rel_back = [None] + [torch.from_numpy(np.linalg.inv(P[t - 1]) @ P[t]).to(DEVICE) for t in range(1, len(P))]
    prev = None
    for t in range(pred.shape[0]):
        inv = (scale * pred[t].double().clamp_min(MIN_DEPTH) + shift).clamp_min(MIN_DEPTH)
        dep = (1.0 / inv).clamp(MIN_DEPTH, max_depth)
        m = valid[t]
        if bool(m.any()):
            d, g = dep[m], gt[t][m].double()
            absrel.append(float((torch.abs(d - g) / g).mean()))
            delta1.append(float((torch.maximum(d / g, g / d) < 1.25).double().mean()))
            bias.append(float(torch.median(torch.log(d) - torch.log(g))))
        if temporal and prev is not None:
            tae_sum += _reproj_absrel(prev, dep, rel[t], video["K_eval"])
            tae_sum += _reproj_absrel(dep, prev, rel_back[t], video["K_eval"])
        prev = dep
    if not absrel:
        raise ValueError("no frame with valid ground truth")
    out = {"absrel": float(np.mean(absrel)), "delta1": float(np.mean(delta1)),
           "affine_scale": scale, "affine_shift": shift}
    if temporal:
        out["tae"] = 100.0 * tae_sum / (2 * (pred.shape[0] - 1))
        out["scale_drift"] = 100.0 * float(np.std(bias))
    return out


def static_template(gt: np.ndarray, max_depth: float) -> np.ndarray:
    """Best image-agnostic answer: per-pixel mean ground-truth inverse depth of the
    video, the same for every frame (pixels never observed get the global mean)."""
    valid = (gt > MIN_DEPTH) & (gt < max_depth)
    inv = np.where(valid, 1.0 / np.maximum(gt, MIN_DEPTH), 0.0).sum(axis=0, dtype=np.float64)
    cnt = valid.sum(axis=0)
    glob = inv.sum() / max(cnt.sum(), 1)
    tmpl = np.where(cnt > 0, inv / np.maximum(cnt, 1), glob).astype(np.float32)
    return np.broadcast_to(tmpl, gt.shape)


# --------------------------------------------------------------------------- predictors


def discover(roots: list[str]) -> dict:
    found: dict[str, dict] = {}
    for root in roots:
        base = (ROOT / root) if not Path(root).is_absolute() else Path(root)
        for path in sorted(base.glob("*.json")):
            name = path.stem
            if name.startswith("_") or name in found:
                continue
            entry = {"manifest": str(path)}
            try:
                cmd = json.loads(path.read_text()).get("command")
                if not (isinstance(cmd, list) and cmd and all(isinstance(c, str) for c in cmd)):
                    raise ValueError("'command' must be a non-empty list of strings")
                entry["command"] = cmd
            except Exception as exc:  # malformed manifest -> failed run
                entry["error"] = f"invalid manifest: {exc}"
            found[name] = entry
    return found


def prepare_inputs(videos: list[dict], inputs: Path, link: bool) -> None:
    """Put every RGB frame into the scratch input directory once (hard links where
    possible, otherwise copies)."""
    def put(pair):
        src, dst = pair
        if link:
            try:
                os.link(src, dst)
                return
            except OSError:
                pass
        shutil.copyfile(src, dst)

    pairs = []
    for v in videos:
        d = inputs / v["id"]
        d.mkdir(parents=True)
        v["frame_files"] = [str(d / f"{t:06d}.png") for t in range(len(v["rgb"]))]
        pairs += list(zip(v["rgb"], v["frame_files"]))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(put, pairs))


def run_predictor(name: str, entry: dict, videos: list[dict], scratch: Path, prefix: list[str] | None,
                  timeout: float | None) -> dict:
    if "error" in entry:
        return {"error": entry["error"]}
    outputs = scratch / "outputs" / name
    shutil.rmtree(outputs, ignore_errors=True)
    outputs.mkdir(parents=True)
    jobs = [{"id": v["id"], "frames": v["frame_files"], "num_frames": len(v["frame_files"]), "fps": v["fps"],
             "height": v["height"], "width": v["width"], "stride": STRIDE, "out_height": v["out_height"],
             "out_width": v["out_width"], "output": str(outputs / f"{v['id']}.npz")} for v in videos]
    jobs_path = scratch / f"jobs_{name}.json"
    jobs_path.write_text(json.dumps({"videos": jobs}, indent=1))
    log_path = scratch / f"{name}.log"
    cmd = (prefix or []) + entry["command"] + ["--jobs", str(jobs_path)]
    t0 = time.time()
    try:
        with log_path.open("w") as log:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    except Exception as exc:
        rc = f"could not start: {exc}"
    tail = log_path.read_text(errors="replace").splitlines()[-15:] if log_path.exists() else []
    print(f"[train] predictor {name}: exit={rc} after {time.time() - t0:.0f}s", flush=True)
    for line in tail:
        print(f"    | {line[-200:]}", flush=True)
    if rc != 0:
        return {"error": f"predictor exited with {rc}"}
    return {"outputs": outputs}


def load_prediction(path: Path, shape: tuple) -> np.ndarray:
    if not path.is_file():
        raise ValueError("prediction file missing")
    with np.load(path, allow_pickle=False) as data:
        if "prediction" not in data.files:
            raise ValueError("expected an .npz with a 'prediction' array")
        arr = np.asarray(data["prediction"])
    if arr.shape != tuple(shape):
        raise ValueError(f"prediction shape {arr.shape} != expected {tuple(shape)}")
    if not (np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)):
        raise ValueError(f"prediction dtype {arr.dtype} is not real-valued")
    arr = arr.astype(np.float32)
    if not np.all(np.isfinite(arr)):
        raise ValueError("prediction contains NaN or infinite values")
    return arr


# --------------------------------------------------------------------------- aggregation


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in rows]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="scoring/scores_train.json")
    parser.add_argument("--detail-output", default="scoring/metrics_detail_train.json")
    parser.add_argument("--predictor-root", action="append", default=[], help="extra manifest directory")
    parser.add_argument("--no-default-roots", action="store_true")
    parser.add_argument("--only", action="append", default=[], help="score only these predictor names")
    parser.add_argument("--timeout", type=float, default=None, help="optional per-predictor limit (s); none by default")
    args = parser.parse_args()

    roots = ([] if args.no_default_roots else list(DEFAULT_ROOTS)) + args.predictor_root
    predictors = discover(roots)
    if args.only:
        predictors = {k: v for k, v in predictors.items() if k in set(args.only)}
    if not predictors:
        raise SystemExit("No predictor manifests found under " + ", ".join(roots))
    print(f"[train] predictors: {', '.join(predictors)}; metrics on {DEVICE}", flush=True)

    videos = load_videos()
    prefix = None  # train labels are visible anyway; no namespace isolation
    scratch = ROOT / "tmp" / f"eval_train_{uuid.uuid4().hex[:12]}"
    inputs = scratch / "inputs"
    inputs.mkdir(parents=True)
    levels: dict[str, dict] = {}
    runs: dict[str, dict] = {}
    try:
        prepare_inputs(videos, inputs, link=True)  # train data is visible; links save copying
        for name, entry in predictors.items():
            run = run_predictor(name, entry, videos, scratch, prefix, args.timeout)
            per_video = {}
            for v in videos:
                if "error" in run and v["id"] in levels:
                    continue
                gt = load_depth(v)
                if v["id"] not in levels:
                    levels[v["id"]] = score_video(static_template(gt, v["max_depth"]), gt, v, temporal=False)
                if "error" not in run:
                    try:
                        pred = load_prediction(run["outputs"] / f"{v['id']}.npz", gt.shape)
                        per_video[v["id"]] = score_video(pred, gt, v)
                        del pred
                    except Exception as exc:
                        run = {"error": f"{v['id']}: {exc}"}
                del gt
            if "outputs" in run:
                shutil.rmtree(run["outputs"], ignore_errors=True)
            shutil.rmtree(scratch / "outputs" / name, ignore_errors=True)
            runs[name] = run if "error" in run else {"per_video": per_video}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[train] scored {name}: {'failed: ' + run['error'] if 'error' in run else 'ok'}", flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    gates, level_summary = {}, {}
    for exp in EXPERIMENTS:
        st = [levels[v["id"]] for v in videos if v["experiment"] == exp]
        gates[exp] = {"absrel": _mean(st, "absrel"), "delta1": _mean(st, "delta1"), "tae": TAE_CAP,
                      "scale_drift": DRIFT_CAP}
        level_summary[exp] = {"static_template": {m: _mean(st, m) for m in ("absrel", "delta1")}}

    scores = {"experiments": {exp: {"results": {}} for exp in EXPERIMENTS}, "_status": {}}
    detail = {"stride": STRIDE,
              "experiments": {exp: {"dataset": EXPERIMENTS[exp][0], "videos": EXPERIMENTS[exp][1],
                                    "frames_per_video": EXPERIMENTS[exp][2], "failure_values": gates[exp],
                                    "levels": level_summary[exp], "results": {}} for exp in EXPERIMENTS},
              "per_video_levels": {v["clip"]: levels[v["id"]] for v in videos}}
    for name, run in runs.items():
        reasons, means = [], {}
        if "error" in run:
            reasons.append(run["error"])
        else:
            for exp in EXPERIMENTS:
                rows = [run["per_video"][v["id"]] for v in videos if v["experiment"] == exp]
                means[exp] = {m: _mean(rows, m) for m in METRICS}
                g = gates[exp]
                if not all(np.isfinite(list(means[exp].values()))):
                    reasons.append(f"{exp}: non-finite metric")
                    continue
                if not means[exp]["absrel"] < g["absrel"]:
                    reasons.append(f"{exp}: absrel {means[exp]['absrel']:.4f} is not below the static-template level {g['absrel']:.4f}")
                if not means[exp]["delta1"] > g["delta1"]:
                    reasons.append(f"{exp}: delta1 {means[exp]['delta1']:.4f} is not above the static-template level {g['delta1']:.4f}")
                if not means[exp]["tae"] < g["tae"]:
                    reasons.append(f"{exp}: tae {means[exp]['tae']:.3f} reaches the cap {g['tae']}")
                if not means[exp]["scale_drift"] < g["scale_drift"]:
                    reasons.append(f"{exp}: scale_drift {means[exp]['scale_drift']:.3f} reaches the cap {g['scale_drift']}")
        failed = bool(reasons)
        status = {"status": "failed" if failed else "ok"}
        if failed:
            status["reasons"] = reasons
        scores["_status"][name] = status
        for exp in EXPERIMENTS:
            scores["experiments"][exp]["results"][name] = dict(gates[exp]) if failed else means[exp]
            detail["experiments"][exp]["results"][name] = {
                "status": status["status"], "unfailed_mean": means.get(exp),
                "per_video": ({v["clip"]: run["per_video"][v["id"]] for v in videos if v["experiment"] == exp}
                              if "per_video" in run else None)}
        print(f"[train] {name}: {json.dumps(status)} "
              f"{json.dumps({e: scores['experiments'][e]['results'][name] for e in EXPERIMENTS})}", flush=True)

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2))
    det = ROOT / args.detail_output
    det.parent.mkdir(parents=True, exist_ok=True)
    det.write_text(json.dumps(detail, indent=1))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
