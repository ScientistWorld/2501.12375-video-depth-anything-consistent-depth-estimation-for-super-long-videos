"""Visible train evaluator for long-video depth estimation.

Renders the visible train videos with ``data/synthetic_rgbd.py`` (seeds below),
hands their RGB frames to every registered predictor, and scores the returned
predictions against the rendered ground truth. It uses the same predictor
contract and the same metric definitions as the held-out test, on different
scenes of the same family.

A predictor is registered by a JSON manifest ``<root>/<name>.json`` with a
``command`` list; the default roots are ``method/predictors`` and
``baseline/predictors``. The evaluator runs ``command + ["--jobs", jobs.json]``
from the workspace root (contract in briefing/evaluation.md).

Every number in the output is recomputed here from the raw predictions; the
output file is always written from scratch.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.synthetic_rgbd import render_video  # noqa: E402
DEFAULT_ROOTS = ("method/predictors", "baseline/predictors")

# Visible train set: experiment -> frames per video and scene seeds.
EXPERIMENTS = {
    "long_video_500": {"frames": 500, "seeds": [101, 102, 103, 104]},
    "super_long_video_1500": {"frames": 1500, "seeds": [201, 202]},
}
METRICS = ("absrel", "delta1", "tae", "scale_drift")

# Evaluation depth range (pixels outside are ignored, as with max_depth_eval).
D_MIN = 0.2
D_MAX = 20.0
# Validity caps: a run whose mean temporal error or scale drift reaches these
# values does not produce usable video depth and is scored as failed.
TAE_CAP = 30.0
DRIFT_CAP = 50.0
DEFAULT_TIMEOUT = 4 * 3600


# --------------------------------------------------------------------------- metrics


def _align(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> tuple[float, float, int]:
    """Least-squares scale/shift of the predicted inverse depth to GT inverse depth,
    shared by every frame of the video (paper protocol)."""
    sxx = sx = sxy = sy = 0.0
    n = 0
    for t in range(gt.shape[0]):
        m = valid[t]
        if not m.any():
            continue
        x = pred[t][m].astype(np.float64)
        y = 1.0 / gt[t][m].astype(np.float64)
        sxx += float(x @ x)
        sx += float(x.sum())
        sxy += float(x @ y)
        sy += float(y.sum())
        n += x.size
    if n == 0:
        raise ValueError("no valid ground-truth pixels")
    det = sxx * n - sx * sx
    if det <= 1e-9 * max(sxx * n, 1e-300):
        return 0.0, sy / n, n  # constant prediction: best constant fit
    return (n * sxy - sx * sy) / det, (sxx * sy - sx * sxy) / det, n


def _aligned_depth(pred_t: np.ndarray, scale: float, shift: float) -> np.ndarray:
    inv = np.maximum(scale * pred_t.astype(np.float64) + shift, 1e-3)
    return np.clip(1.0 / inv, 1e-3, D_MAX)


def _reproj_absrel(src, dst, rel, K, valid_src, valid_dst):
    """AbsRel between dst depth and src depth reprojected into dst (one TAE term)."""
    h, w = src.shape
    vs, us = np.nonzero(valid_src)
    if vs.size == 0:
        return None
    z = src[vs, us]
    x = (us - K[0, 2]) / K[0, 0] * z
    y = (vs - K[1, 2]) / K[1, 1] * z
    p = rel[:3, :3] @ np.stack([x, y, z]) + rel[:3, 3:4]
    keep = p[2] > 1e-6
    if not keep.any():
        return None
    zp = p[2][keep]
    u2 = np.round(K[0, 0] * p[0][keep] / zp + K[0, 2]).astype(np.int64)
    v2 = np.round(K[1, 1] * p[1][keep] / zp + K[1, 2]).astype(np.int64)
    inb = (u2 >= 0) & (u2 < w) & (v2 >= 0) & (v2 < h)
    if not inb.any():
        return None
    proj = np.full(h * w, np.inf)
    np.minimum.at(proj, v2[inb] * w + u2[inb], zp[inb])
    tgt = dst.reshape(-1)
    m = np.isfinite(proj) & valid_dst.reshape(-1)
    if not m.any():
        return None
    return float(np.mean(np.abs(tgt[m] - proj[m]) / tgt[m]))


def score_video(pred: np.ndarray, video: dict, temporal: bool = True) -> dict:
    gt = video["depth"]
    valid = video["valid"]
    K = video["K"]
    poses = video["poses"]
    scale, shift, n = _align(pred, gt, valid)
    abs_sum = 0.0
    d1_count = 0
    bias = []
    tae_sum = 0.0
    tae_n = 0
    prev = None
    for t in range(gt.shape[0]):
        dep = _aligned_depth(pred[t], scale, shift)
        m = valid[t]
        if m.any():
            g = gt[t][m].astype(np.float64)
            d = dep[m]
            abs_sum += float(np.sum(np.abs(d - g) / g))
            d1_count += int(np.count_nonzero(np.maximum(d / g, g / d) < 1.25))
            bias.append(float(np.median(np.log(d) - np.log(g))))
        if temporal and prev is not None:
            fwd = np.linalg.inv(poses[t]) @ poses[t - 1]
            for err in (
                _reproj_absrel(prev, dep, fwd, K, valid[t - 1], valid[t]),
                _reproj_absrel(dep, prev, np.linalg.inv(fwd), K, valid[t], valid[t - 1]),
            ):
                if err is not None:
                    tae_sum += err
                    tae_n += 1
        prev = dep
    out = {"absrel": abs_sum / n, "delta1": d1_count / n, "affine_scale": scale, "affine_shift": shift}
    if temporal:
        out["tae"] = 100.0 * tae_sum / max(tae_n, 1)
        out["scale_drift"] = 100.0 * float(np.std(bias)) if bias else float("nan")
    return out


# --------------------------------------------------------------------------- data


def render_train_set() -> list[dict]:
    videos = []
    for exp, cfg in EXPERIMENTS.items():
        for seed in cfg["seeds"]:
            t0 = time.time()
            v = render_video(seed, cfg["frames"])
            v["valid"] = (v["depth"] >= D_MIN) & (v["depth"] <= D_MAX)
            v["experiment"] = exp
            v["id"] = f"video_{len(videos):03d}"
            videos.append(v)
            print(f"[train] rendered {v['id']} ({cfg['frames']} frames) in {time.time() - t0:.1f}s", flush=True)
    return videos


def reference_levels(videos: list[dict]) -> dict:
    """Image-agnostic static template (per-pixel mean GT inverse depth, the best
    prediction that ignores frame content) and the exact-GT floor."""
    levels = {}
    for v in videos:
        inv = np.where(v["valid"], 1.0 / np.maximum(v["depth"], 1e-6), 0.0).astype(np.float64)
        cnt = v["valid"].sum(axis=0)
        tmpl = np.where(cnt > 0, inv.sum(axis=0) / np.maximum(cnt, 1), inv.sum() / max(cnt.sum(), 1))
        static = np.broadcast_to(tmpl.astype(np.float32), v["depth"].shape)
        exact = (1.0 / np.maximum(v["depth"], 1e-6)).astype(np.float32)
        levels[v["id"]] = {
            "static_template": score_video(static, v, temporal=False),
            "exact_ground_truth": score_video(exact, v, temporal=True),
        }
    return levels


# --------------------------------------------------------------------------- predictors


def discover(roots: list[str]) -> dict:
    found: dict[str, dict] = {}
    for root in roots:
        base = (ROOT / root) if not Path(root).is_absolute() else Path(root)
        for path in sorted(base.glob("*.json")):
            name = path.stem
            if name.startswith("_") or name in found:
                continue
            entry = {"manifest": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)}
            try:
                cmd = json.loads(path.read_text()).get("command")
                if not (isinstance(cmd, list) and cmd and all(isinstance(c, str) for c in cmd)):
                    raise ValueError("'command' must be a non-empty list of strings")
                entry["command"] = cmd
            except Exception as exc:  # malformed manifest -> failed run
                entry["error"] = f"invalid manifest: {exc}"
            found[name] = entry
    return found


def load_prediction(path: Path, shape: tuple) -> np.ndarray:
    if not path.is_file():
        raise ValueError("prediction file missing")
    data = np.load(path, allow_pickle=False)
    if not hasattr(data, "files") or "prediction" not in data.files:
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


def run_predictor(name: str, entry: dict, videos: list[dict], scratch: Path, timeout: int) -> dict:
    """Run one predictor on all train videos and return per-video metrics or an error."""
    if "error" in entry:
        return {"error": entry["error"]}
    inputs = scratch / "inputs"
    shutil.rmtree(inputs, ignore_errors=True)
    inputs.mkdir(parents=True)
    outputs = scratch / "outputs" / name
    shutil.rmtree(outputs, ignore_errors=True)
    outputs.mkdir(parents=True)
    jobs = []
    for v in videos:
        frames_path = inputs / f"{v['id']}.npy"
        np.save(frames_path, v["rgb"])
        t, h, w = v["depth"].shape
        jobs.append({"id": v["id"], "frames": str(frames_path), "fps": int(v["fps"]), "num_frames": t,
                     "height": h, "width": w, "output": str(outputs / f"{v['id']}.npz")})
    jobs_path = scratch / f"jobs_{name}.json"
    jobs_path.write_text(json.dumps({"videos": jobs}, indent=1))
    log_path = scratch / f"{name}.log"
    t0 = time.time()
    try:
        with log_path.open("w") as log:
            rc = subprocess.run(entry["command"] + ["--jobs", str(jobs_path)], cwd=ROOT, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    except Exception as exc:
        rc = f"could not start: {exc}"
    elapsed = time.time() - t0
    tail = log_path.read_text(errors="replace").splitlines()[-15:] if log_path.exists() else []
    print(f"[train] predictor {name}: exit={rc} after {elapsed:.0f}s", flush=True)
    for line in tail:
        print(f"    | {line[-200:]}", flush=True)
    if rc != 0:
        shutil.rmtree(outputs, ignore_errors=True)
        return {"error": f"predictor exited with {rc}"}
    per_video = {}
    try:
        for v in videos:
            pred = load_prediction(outputs / f"{v['id']}.npz", v["depth"].shape)
            per_video[v["id"]] = score_video(pred, v)
            del pred
    except Exception as exc:
        return {"error": f"{v['id']}: {exc}"}
    finally:
        shutil.rmtree(outputs, ignore_errors=True)
    return {"per_video": per_video}


# --------------------------------------------------------------------------- aggregation


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([r[key] for r in rows]))


def aggregate(name: str, run: dict, videos: list[dict], levels: dict, gates: dict):
    by_exp = {exp: [v["id"] for v in videos if v["experiment"] == exp] for exp in EXPERIMENTS}
    reasons = []
    means = {}
    if "error" in run:
        reasons.append(run["error"])
    else:
        for exp, ids in by_exp.items():
            rows = [run["per_video"][i] for i in ids]
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
    scores = {exp: (dict(gates[exp]) if failed else means[exp]) for exp in EXPERIMENTS}
    status = {"status": "failed" if failed else "ok"}
    if failed:
        status["reasons"] = reasons
    return scores, status, means


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="scoring/scores_train.json")
    parser.add_argument("--detail-output", default="scoring/metrics_detail_train.json")
    parser.add_argument("--predictor-root", action="append", default=[], help="extra manifest directory")
    parser.add_argument("--no-default-roots", action="store_true")
    parser.add_argument("--only", action="append", default=[], help="score only these predictor names")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    roots = ([] if args.no_default_roots else list(DEFAULT_ROOTS)) + args.predictor_root
    predictors = discover(roots)
    if args.only:
        predictors = {k: v for k, v in predictors.items() if k in set(args.only)}
    if not predictors:
        raise SystemExit("No predictor manifests found under " + ", ".join(roots))
    print(f"[train] predictors: {', '.join(predictors)}", flush=True)

    scratch = ROOT / "tmp" / f"eval_train_{uuid.uuid4().hex[:12]}"
    scratch.mkdir(parents=True)
    try:
        videos = render_train_set()
        levels = reference_levels(videos)
        gates, level_summary = {}, {}
        for exp in EXPERIMENTS:
            ids = [v["id"] for v in videos if v["experiment"] == exp]
            st = [levels[i]["static_template"] for i in ids]
            ex = [levels[i]["exact_ground_truth"] for i in ids]
            gates[exp] = {"absrel": _mean(st, "absrel"), "delta1": _mean(st, "delta1"),
                          "tae": TAE_CAP, "scale_drift": DRIFT_CAP}
            level_summary[exp] = {"static_template": {m: _mean(st, m) for m in ("absrel", "delta1")},
                                  "exact_ground_truth": {m: _mean(ex, m) for m in METRICS}}
        scores = {"experiments": {exp: {"results": {}} for exp in EXPERIMENTS}, "_status": {}}
        detail = {"experiments": {exp: {"frames_per_video": EXPERIMENTS[exp]["frames"],
                                        "videos": len(EXPERIMENTS[exp]["seeds"]),
                                        "failure_values": gates[exp], "levels": level_summary[exp],
                                        "results": {}} for exp in EXPERIMENTS},
                  "per_video_levels": levels}
        for name, entry in predictors.items():
            run = run_predictor(name, entry, videos, scratch, args.timeout)
            exp_scores, status, means = aggregate(name, run, videos, levels, gates)
            scores["_status"][name] = status
            for exp in EXPERIMENTS:
                scores["experiments"][exp]["results"][name] = exp_scores[exp]
                ids = [v["id"] for v in videos if v["experiment"] == exp]
                detail["experiments"][exp]["results"][name] = {
                    "status": status["status"],
                    "unfailed_mean": means.get(exp),
                    "per_video": {i: run["per_video"][i] for i in ids} if "per_video" in run else None,
                }
            print(f"[train] {name}: {json.dumps(status)} {json.dumps({e: exp_scores[e] for e in EXPERIMENTS})}",
                  flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2))
    det = ROOT / args.detail_output
    det.parent.mkdir(parents=True, exist_ok=True)
    det.write_text(json.dumps(detail, indent=1))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
