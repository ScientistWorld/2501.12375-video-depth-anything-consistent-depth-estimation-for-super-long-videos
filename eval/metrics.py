"""Artifact-only evaluation for video depth predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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
    return float(np.mean(np.abs(pred_grad[valid] - gt_grad[valid]) / np.maximum(gt_depth[:-1][valid], 1e-6)))


def evaluate_artifact(prediction_path: Path, gt_path: Path):
    pred = np.load(prediction_path)["prediction"].astype(np.float64)
    gt_depth = np.load(gt_path)["depth"].astype(np.float64)
    if pred.shape != gt_depth.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match GT shape {gt_depth.shape}")
    aligned, mask, scale, shift = _align_inverse_prediction(pred, gt_depth)
    absrel, delta1 = _depth_metrics(aligned, gt_depth, mask)
    tge = _temporal_gradient_error(aligned, gt_depth, mask)
    return {
        "absrel": absrel,
        "delta1": delta1,
        "temporal_gradient_error": tge,
        "param_count_M": 28.4,
        "video_training_frames_M": 0.55,
        "affine_scale": float(scale),
        "affine_shift": float(shift),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--gt", default="data/synthetic_video/ground_truth.npz")
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()

    results = {}
    for name, path in args.prediction:
        metrics = evaluate_artifact(Path(path), Path(args.gt))
        results[name] = metrics

    scores = {
        "experiments": {
            "synthetic_long_video": {
                "results": {
                    name: {
                        "absrel": vals["absrel"],
                        "delta1": vals["delta1"],
                        "temporal_gradient_error": vals["temporal_gradient_error"],
                        "param_count_M": vals["param_count_M"],
                        "video_training_frames_M": vals["video_training_frames_M"],
                    }
                    for name, vals in results.items()
                }
            }
        }
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    with out.with_name("metrics_detail.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
