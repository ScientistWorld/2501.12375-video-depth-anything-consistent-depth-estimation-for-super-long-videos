"""Artifact-only evaluation for video depth predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRIC_KEYS = (
    "absrel",
    "delta1",
    "temporal_gradient_error",
    "boundary_temporal_error",
    "scale_drift_error",
    "param_count_M",
    "video_training_frames_M",
)
BOUNDARY_STRIDE = 32
DEFAULT_PARAM_COUNT_M = 28.4
DEFAULT_VIDEO_TRAINING_FRAMES_M = 0.55


def _align_inverse_prediction(pred: np.ndarray, gt_depth: np.ndarray):
    mask = np.isfinite(pred) & np.isfinite(gt_depth) & (pred > 0) & (gt_depth > 0)
    target_inv = 1.0 / np.clip(gt_depth, 1e-6, None)
    x = pred[mask].reshape(-1)
    y = target_inv[mask].reshape(-1)
    a00 = float(np.dot(x, x))
    a01 = float(np.sum(x))
    a11 = float(x.size)
    b0 = float(np.dot(x, y))
    b1 = float(np.sum(y))
    det = a00 * a11 - a01 * a01
    if abs(det) < 1e-12:
        scale, shift = 1.0, 0.0
    else:
        scale = (a11 * b0 - a01 * b1) / det
        shift = (-a01 * b0 + a00 * b1) / det
    aligned_inv = np.maximum(scale * pred + shift, 1e-6)
    return 1.0 / aligned_inv, mask, scale, shift


def _depth_metrics(pred_depth: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray):
    pred = pred_depth[mask]
    gt = gt_depth[mask]
    absrel = float(np.mean(np.abs(pred - gt) / np.maximum(gt, 1e-6)))
    ratio = np.maximum(pred / np.maximum(gt, 1e-6), gt / np.maximum(pred, 1e-6))
    delta1 = float(np.mean(ratio < 1.25))
    return absrel, delta1


def _temporal_gradient_error(pred_depth: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray):
    valid = mask[1:] & mask[:-1]
    pred_grad = np.abs(np.diff(pred_depth, axis=0))
    gt_grad = np.abs(np.diff(gt_depth, axis=0))
    stable = gt_grad < (0.05 * (np.max(gt_depth) - np.min(gt_depth)))
    valid = valid & stable
    # Report as a percentage, matching the scale convention used by the paper's TAE evaluator.
    return float(np.mean(np.abs(pred_grad[valid] - gt_grad[valid]) / np.maximum(gt_depth[:-1][valid], 1e-6)) * 100.0)


def _boundary_temporal_error(pred_depth: np.ndarray, gt_depth: np.ndarray):
    starts = np.arange(BOUNDARY_STRIDE, pred_depth.shape[0], BOUNDARY_STRIDE)
    if starts.size == 0:
        raise ValueError("No boundary pairs available for boundary temporal error")
    prev = starts - 1
    finite = (
        np.isfinite(pred_depth[starts])
        & np.isfinite(pred_depth[prev])
        & np.isfinite(gt_depth[starts])
        & np.isfinite(gt_depth[prev])
    )
    pred_jump = np.abs(pred_depth[starts] - pred_depth[prev])
    gt_jump = np.abs(gt_depth[starts] - gt_depth[prev])
    stable = gt_jump < (0.05 * (np.max(gt_depth) - np.min(gt_depth)))
    valid = finite & stable
    if not np.any(valid):
        raise ValueError("No valid boundary pixels for boundary temporal error")
    return float(np.mean(np.abs(pred_jump[valid] - gt_jump[valid]) / np.maximum(gt_depth[prev][valid], 1e-6)) * 100.0)


def _scale_drift_error(pred_depth: np.ndarray, gt_depth: np.ndarray):
    if pred_depth.shape[0] < 4:
        raise ValueError("Too few frames for scale drift error")
    chunk = max(1, pred_depth.shape[0] // 4)
    valid = np.isfinite(pred_depth) & np.isfinite(gt_depth) & (pred_depth > 0) & (gt_depth > 0)
    log_ratio = np.full(pred_depth.shape, np.nan, dtype=np.float64)
    log_ratio[valid] = np.log(pred_depth[valid] / gt_depth[valid])
    early_bias = np.nanmedian(log_ratio[:chunk], axis=(1, 2))
    late_bias = np.nanmedian(log_ratio[-chunk:], axis=(1, 2))
    return float(abs(np.nanmean(late_bias) - np.nanmean(early_bias)))


def _metadata_metrics(prediction_path: Path):
    meta_path = prediction_path.with_suffix(".json")
    param_count = DEFAULT_PARAM_COUNT_M
    training_frames = DEFAULT_VIDEO_TRAINING_FRAMES_M
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        param_count = float(metadata.get("param_count_M", param_count))
        training_frames = float(metadata.get("video_training_frames_M", training_frames))
    return param_count, training_frames


def evaluate_artifact(prediction_path: Path, gt_path: Path):
    pred = np.load(prediction_path)["prediction"].astype(np.float64)
    gt_depth = np.load(gt_path)["depth"].astype(np.float64)
    if pred.shape != gt_depth.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match GT shape {gt_depth.shape}")
    aligned, mask, scale, shift = _align_inverse_prediction(pred, gt_depth)
    absrel, delta1 = _depth_metrics(aligned, gt_depth, mask)
    tge = _temporal_gradient_error(aligned, gt_depth, mask)
    boundary_tge = _boundary_temporal_error(aligned, gt_depth)
    drift = _scale_drift_error(aligned, gt_depth)
    param_count, training_frames = _metadata_metrics(prediction_path)
    return {
        "absrel": absrel,
        "delta1": delta1,
        "temporal_gradient_error": tge,
        "boundary_temporal_error": boundary_tge,
        "scale_drift_error": drift,
        "param_count_M": param_count,
        "video_training_frames_M": training_frames,
        "affine_scale": float(scale),
        "affine_shift": float(shift),
        "artifact_path": str(prediction_path),
        "gt_path": str(gt_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--gt", default="data/synthetic_video/ground_truth.npz")
    parser.add_argument("--output", default="scoring/scores.json")
    parser.add_argument("--experiment", default="synthetic_long_video_128")
    args = parser.parse_args()

    results = {}
    for name, path in args.prediction:
        metrics = evaluate_artifact(Path(path), Path(args.gt))
        results[name] = metrics

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        with out.open("r", encoding="utf-8") as f:
            scores = json.load(f)
    else:
        scores = {"experiments": {}}
    scores.setdefault("experiments", {})[args.experiment] = {
        "results": {
            name: {
                key: vals[key]
                for key in METRIC_KEYS
            }
            for name, vals in results.items()
        }
    }
    with out.open("w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    detail_path = out.with_name("metrics_detail.json")
    if detail_path.exists():
        with detail_path.open("r", encoding="utf-8") as f:
            detail = json.load(f)
    else:
        detail = {}
    detail[args.experiment] = results
    with detail_path.open("w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
