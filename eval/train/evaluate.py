"""Visible train-slice evaluator for artifact-only video-depth benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np


SLICE = "train"
BLOCK_SIZE = 16
BLOCK_PARITY = 0
SEARCH_ROOTS = (
    Path("baseline/outputs"),
    Path("method/outputs"),
    Path("checkpoints/outputs"),
    Path("outputs"),
)
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
BOUNDARY_PARITY = 0
DEFAULT_PARAM_COUNT_M = 28.4
DEFAULT_VIDEO_TRAINING_FRAMES_M = 0.55


def _selected_frames(frame_count: int) -> np.ndarray:
    frames = np.arange(frame_count)
    return ((frames // BLOCK_SIZE) % 2) == BLOCK_PARITY


def _align_inverse_prediction(pred: np.ndarray, gt_depth: np.ndarray, frame_mask: np.ndarray):
    slice_mask = frame_mask[:, None, None]
    mask = np.isfinite(pred) & np.isfinite(gt_depth) & (pred > 0) & (gt_depth > 0) & slice_mask
    if not np.any(mask):
        raise ValueError("No valid pixels in selected train slice")
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


def _temporal_gradient_error(pred_depth: np.ndarray, gt_depth: np.ndarray, frame_mask: np.ndarray):
    pair_mask = frame_mask[1:] & frame_mask[:-1]
    finite = np.isfinite(pred_depth[1:]) & np.isfinite(pred_depth[:-1]) & np.isfinite(gt_depth[1:]) & np.isfinite(gt_depth[:-1])
    pred_grad = np.abs(np.diff(pred_depth, axis=0))
    gt_grad = np.abs(np.diff(gt_depth, axis=0))
    stable = gt_grad < (0.05 * (np.max(gt_depth) - np.min(gt_depth)))
    valid = finite & stable & pair_mask[:, None, None]
    if not np.any(valid):
        raise ValueError("No valid temporal pairs in selected train slice")
    return float(np.mean(np.abs(pred_grad[valid] - gt_grad[valid]) / np.maximum(gt_depth[:-1][valid], 1e-6)) * 100.0)


def _boundary_temporal_error(pred_depth: np.ndarray, gt_depth: np.ndarray):
    starts = np.arange(BOUNDARY_STRIDE, pred_depth.shape[0], BOUNDARY_STRIDE)
    starts = starts[(np.arange(starts.size) % 2) == BOUNDARY_PARITY]
    if starts.size == 0:
        raise ValueError("No train boundary pairs available for boundary temporal error")
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
        raise ValueError("No valid train boundary pixels for boundary temporal error")
    return float(np.mean(np.abs(pred_jump[valid] - gt_jump[valid]) / np.maximum(gt_depth[prev][valid], 1e-6)) * 100.0)


def _scale_drift_error(pred_depth: np.ndarray, gt_depth: np.ndarray, frame_mask: np.ndarray):
    selected = np.flatnonzero(frame_mask)
    if selected.size < 4:
        raise ValueError("Too few selected train frames for scale drift error")
    chunk = max(1, selected.size // 4)
    early = selected[:chunk]
    late = selected[-chunk:]
    valid = np.isfinite(pred_depth) & np.isfinite(gt_depth) & (pred_depth > 0) & (gt_depth > 0)
    log_ratio = np.full(pred_depth.shape, np.nan, dtype=np.float64)
    log_ratio[valid] = np.log(pred_depth[valid] / gt_depth[valid])
    early_bias = np.nanmedian(log_ratio[early], axis=(1, 2))
    late_bias = np.nanmedian(log_ratio[late], axis=(1, 2))
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
    frame_mask = _selected_frames(pred.shape[0])
    aligned, spatial_mask, scale, shift = _align_inverse_prediction(pred, gt_depth, frame_mask)
    absrel, delta1 = _depth_metrics(aligned, gt_depth, spatial_mask)
    tge = _temporal_gradient_error(aligned, gt_depth, frame_mask)
    boundary_tge = _boundary_temporal_error(aligned, gt_depth)
    drift = _scale_drift_error(aligned, gt_depth, frame_mask)
    param_count, training_frames = _metadata_metrics(prediction_path)
    return {
        "slice": SLICE,
        "split": {
            "block_size": BLOCK_SIZE,
            "block_parity": BLOCK_PARITY,
            "boundary_stride": BOUNDARY_STRIDE,
            "boundary_parity": BOUNDARY_PARITY,
            "frame_count": int(frame_mask.sum()),
            "temporal_pair_count": int(np.sum(frame_mask[1:] & frame_mask[:-1])),
        },
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


def _reference_experiments() -> list[str]:
    ref_path = Path("scoring/reference.json")
    if ref_path.exists():
        with ref_path.open("r", encoding="utf-8") as f:
            ref = json.load(f)
        return list(ref.get("experiments", {}).keys())
    experiments = set()
    for root in SEARCH_ROOTS:
        if root.exists():
            experiments.update(path.parent.name for path in root.glob("*/*.npz"))
    return sorted(experiments)


def _infer_gt_path(experiment: str, explicit_gt: str | None, experiment_count: int) -> Path:
    if explicit_gt and experiment_count == 1:
        return Path(explicit_gt)
    match = re.search(r"synthetic_long_video_(\d+)", experiment)
    if match:
        return Path(f"data/synthetic_video_{match.group(1)}/ground_truth.npz")
    return Path("data") / experiment / "ground_truth.npz"


def _artifact_candidates(experiment: str, name: str) -> list[Path]:
    return [root / experiment / f"{name}.npz" for root in SEARCH_ROOTS]


def _discover_artifacts(experiment: str) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for root in SEARCH_ROOTS:
        for path in sorted((root / experiment).glob("*.npz")):
            found.setdefault(path.stem, path)
    return found


def _resolve_requested_artifacts(experiment: str, requested: list[str]) -> dict[str, Path]:
    if not requested:
        return _discover_artifacts(experiment)
    resolved: dict[str, Path] = {}
    for item in requested:
        if "=" in item:
            name, raw_path = item.split("=", 1)
            path = Path(raw_path.format(experiment=experiment, exp=experiment))
            if path.exists():
                resolved[name] = path
            continue
        for candidate in _artifact_candidates(experiment, item):
            if candidate.exists():
                resolved[item] = candidate
                break
        else:
            raise FileNotFoundError(f"No artifact found for method '{item}' in experiment '{experiment}'")
    return resolved


def _metric_subset(values: dict[str, object]) -> dict[str, float]:
    return {key: float(values[key]) for key in METRIC_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("methods", nargs="*", help="Method names or name=/path/to/predictions.npz entries.")
    parser.add_argument("--experiment", action="append", default=[])
    parser.add_argument("--gt", default=os.environ.get("VDA_GT"))
    parser.add_argument("--output", default="scoring/scores_train.json")
    parser.add_argument("--detail-output", default="scoring/metrics_detail_train.json")
    args = parser.parse_args()

    experiments = args.experiment or ([os.environ["VDA_EXPERIMENT"]] if os.environ.get("VDA_EXPERIMENT") else _reference_experiments())
    output_path = Path(args.output)
    detail_path = Path(args.detail_output)
    scores = {"experiments": {}}
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            scores = json.load(f)
    details = {}
    if detail_path.exists():
        with detail_path.open("r", encoding="utf-8") as f:
            details = json.load(f)

    scored_any = False
    for experiment in experiments:
        artifacts = _resolve_requested_artifacts(experiment, args.methods)
        if not artifacts:
            continue
        gt_path = _infer_gt_path(experiment, args.gt, len(experiments))
        experiment_details = {
            "slice": SLICE,
            "split": {"block_size": BLOCK_SIZE, "block_parity": BLOCK_PARITY},
            "results": {},
        }
        experiment_scores = {}
        for name, artifact in artifacts.items():
            metrics = evaluate_artifact(artifact, gt_path)
            experiment_details["results"][name] = metrics
            experiment_scores[name] = _metric_subset(metrics)
        scores.setdefault("experiments", {})[experiment] = {"results": experiment_scores}
        details[experiment] = experiment_details
        scored_any = True

    if not scored_any:
        raise SystemExit("No prediction artifacts found for train-slice evaluation.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    with detail_path.open("w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
