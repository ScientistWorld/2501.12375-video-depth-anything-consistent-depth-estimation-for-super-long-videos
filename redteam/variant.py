"""Candidate implementations. This module uses only future-agent-visible inputs.

No hidden evaluator, score reference, split indices, or paper is imported here.
All deliberately invalid candidates have shortcut in their case name.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/user')
sys.path.insert(0, str(ROOT))
EXPERIMENTS = (
    'synthetic_long_video_128',
    'synthetic_long_video_256',
    'synthetic_long_video_128_input518',
    'synthetic_long_video_256_input518',
)


def save(path, prediction, **metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, prediction=prediction)
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2))


def match_affine(source, target):
    # Alignment between two model predictions; never against labels.
    x, y = source.astype(np.float64), target.astype(np.float64)
    xm, ym = x.mean(), y.mean()
    scale = np.mean((x - xm) * (y - ym)) / np.mean((x - xm) ** 2)
    return scale * source + (ym - scale * xm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    case = args.case
    infer = case.startswith('reference_seed_') or case in (
        'stronger_bidirectional', 'stronger_flip_ensemble', 'weaker_no_stabilization')
    if infer:
        import torch
        from method.run_vda import _make_model, _load_frames, _infer_independent_windows
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = _make_model(ROOT / 'data/downloads/video_depth_anything_vits.pth', device)

    generator_report = {}
    for experiment in EXPERIMENTS:
        frames_n = int(experiment.split('_')[3])
        size = 518 if experiment.endswith('input518') else 252
        video_dir = ROOT / f'data/synthetic_video_{frames_n}'
        ref_path = ROOT / 'method/outputs' / experiment / 'vda_oi_kr.npz'
        baseline_path = ROOT / 'baseline/outputs' / experiment / 'vda_independent_windows.npz'
        out = args.output_root / experiment / 'vda_oi_kr.npz'
        meta = {'param_count_M': 28.4, 'video_training_frames_M': 0.55,
                'audit_case': case, 'seed': args.seed}
        if infer:
            frames = _load_frames(video_dir)
            if case == 'weaker_no_stabilization':
                pred = _infer_independent_windows(model, frames, size, device, False)
            elif case.startswith('reference_seed_'):
                pred, _ = model.infer_video_depth(frames, target_fps=12, input_size=size, device=device)
            else:
                original = np.load(ref_path)['prediction']
                if case == 'stronger_bidirectional':
                    augmented, _ = model.infer_video_depth(frames[::-1].copy(), target_fps=12, input_size=size, device=device)
                    augmented = augmented[::-1].copy()
                else:
                    augmented, _ = model.infer_video_depth(frames[:, :, ::-1].copy(), target_fps=12, input_size=size, device=device)
                    augmented = augmented[:, :, ::-1].copy()
                pred = 0.5 * (original + match_affine(augmented, original))
        elif case == 'real_fix_temporal3':
            original = np.load(ref_path)['prediction']
            padded = np.pad(original, ((1, 1), (0, 0), (0, 0)), mode='edge')
            pred = (padded[:-2] + 2 * padded[1:-1] + padded[2:]) / 4
        elif case == 'shortcut_visible_answers':
            pred = 1.0 / np.load(video_dir / 'ground_truth.npz')['depth'].astype(np.float64)
        elif case == 'shortcut_generator_answers':
            from data.synthetic_rgbd import generate_sequence
            config = json.loads((video_dir / 'metadata.json').read_text())
            _, depth = generate_sequence(config['frames'], config['height'], config['width'], config['seed'])
            pred = 1.0 / depth.astype(np.float64)
            visible = np.load(video_dir / 'ground_truth.npz')['depth']
            generator_report[experiment] = {'max_depth_difference': float(np.max(np.abs(visible - depth)))}
        elif case in ('constant_fixed_budget', 'shortcut_constant_zero_resources', 'shortcut_empty',
                      'shortcut_nan', 'shortcut_inf', 'shortcut_zero', 'shortcut_negative'):
            config = json.loads((video_dir / 'metadata.json').read_text())
            shape = (config['frames'], config['height'], config['width'])
            value = {'shortcut_nan': np.nan, 'shortcut_inf': np.inf,
                     'shortcut_zero': 0.0, 'shortcut_negative': -1.0}.get(case, 0.25)
            pred = np.full(shape, value, dtype=np.float32)
            if case == 'shortcut_empty':
                pred = pred[:0]
            if case == 'shortcut_constant_zero_resources':
                meta.update(param_count_M=0, video_training_frames_M=0)
        elif case in ('shortcut_baseline_replacement', 'shortcut_forged_train_scores'):
            pred = np.load(baseline_path)['prediction']
        elif case == 'shortcut_ground_truth_replacement':
            # Both inference-size experiments share the same GT file, so use
            # the same 252-pixel artifact for each length in this shortcut.
            base_exp = f'synthetic_long_video_{frames_n}'
            pred = np.load(ROOT / 'method/outputs' / base_exp / 'vda_oi_kr.npz')['prediction'].astype(np.float64)
            np.savez_compressed(video_dir / 'ground_truth.npz', depth=1.0 / np.maximum(pred, 1e-6))
        else:
            pred = np.load(ref_path)['prediction']
            if case == 'shortcut_metadata_zero':
                meta.update(param_count_M=0, video_training_frames_M=0)
            elif case == 'shortcut_metadata_negative':
                meta.update(param_count_M=-1000000, video_training_frames_M=-1000000)

        save(out, pred, **meta)
        if case == 'shortcut_baseline_replacement':
            # Replace a supposedly fixed comparison artifact, while the
            # candidate remains the unmodified independent-window output.
            save(baseline_path, np.full_like(pred, 0.25), param_count_M=28.4, video_training_frames_M=0.55)
    if generator_report:
        (args.output_root / 'generator_report.json').write_text(json.dumps(generator_report, indent=2))


if __name__ == '__main__':
    main()
