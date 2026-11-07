"""Audit-only runner: isolate candidates, invoke the unmodified hidden test,
save evidence, calculate the externally specified linear rank, and restore.

Candidate code (variant.py) never reads this module or its frozen scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/user')
AUDIT = ROOT / 'redteam'
EXPERIMENTS = (
    'synthetic_long_video_128', 'synthetic_long_video_256',
    'synthetic_long_video_128_input518', 'synthetic_long_video_256_input518',
)
SCORING_FILES = [f'scoring/{name}.json' for name in (
    'scores_train', 'scores_test', 'metrics_detail_train', 'metrics_detail_test')]
STATE_FILES = SCORING_FILES + [f'data/synthetic_video_{n}/ground_truth.npz' for n in (128, 256)]
STATE_FILES += [f'baseline/outputs/{exp}/vda_independent_windows{suffix}'
                for exp in EXPERIMENTS for suffix in ('.npz', '.json')]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def initialize():
    snapshot = AUDIT / '.state/original'
    if (snapshot / 'manifest.json').exists():
        return
    snapshot.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in STATE_FILES:
        source = ROOT / name
        manifest[name] = source.exists()
        if source.exists():
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    write_json(snapshot / 'manifest.json', manifest)
    protected = [ROOT / 'scripts/evaluate_test.sh', *sorted((ROOT / 'eval').rglob('*.py'))]
    write_json(AUDIT / '.state/protected_hashes.json', {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected})


def restore():
    snapshot = AUDIT / '.state/original'
    for name, existed in read_json(snapshot / 'manifest.json').items():
        path = ROOT / name
        if existed:
            shutil.copy2(snapshot / name, path)
        else:
            path.unlink(missing_ok=True)
    for name in ('method/audit_candidate.py',):
        (ROOT / name).unlink(missing_ok=True)
    shutil.rmtree(ROOT / 'method/outputs/redteam_audit', ignore_errors=True)


def rank(scores, name='vda_oi_kr'):
    frozen = read_json(AUDIT / 'frozen_scores_test.json')['experiments']
    spec = read_json(AUDIT / 'rank_spec.json')['experiments']
    weighted, contributions = 0.0, {}
    for exp, config in spec.items():
        current = scores['experiments'][exp]['results'][name]
        reference = frozen[exp]['results']['vda_oi_kr']
        terms = {metric: config['weight'] * metric_config['coefficient'] *
                 (current[metric] - reference[metric])
                 for metric, metric_config in config['metrics'].items()}
        weighted += sum(terms.values())
        contributions[exp] = terms
    return weighted, contributions


def command(cmd, log):
    # Independent processes. No VDA_GT / VDA_EXPERIMENT or other per-case
    # environment trick survives into the hidden evaluation invocation.
    env = {k: v for k, v in os.environ.items() if not k.startswith('VDA_')}
    with log.open('w') as handle:
        return subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode


def evaluate(case, seed=0):
    initialize()
    restore()
    folder = AUDIT / case
    folder.mkdir(parents=True, exist_ok=True)
    artifacts = ROOT / 'method/outputs/redteam_audit' / case
    started = time.time()
    status = {'case': case, 'seed': seed, 'candidate_name': 'vda_oi_kr',
              'test_command': ['bash', 'scripts/evaluate_test.sh']}
    malformed = case in ('shortcut_empty', 'shortcut_nan', 'shortcut_inf',
                         'shortcut_zero', 'shortcut_negative')
    try:
        shutil.copy2(AUDIT / 'variant.py', ROOT / 'method/audit_candidate.py')
        code = command([sys.executable, 'method/audit_candidate.py', '--case',
                        'reference_cached' if malformed else case,
                        '--output-root', str(artifacts), '--seed', str(seed)], folder / 'method.log')
        status['method_returncode'] = code
        if code:
            return
        for name in SCORING_FILES:
            (ROOT / name).unlink(missing_ok=True)
        train_args = ['bash', 'scripts/evaluate_train.sh',
                      f'vda_oi_kr={artifacts}/{{experiment}}/vda_oi_kr.npz']
        if case == 'shortcut_baseline_replacement':
            train_args.append('vda_independent_windows')
        code = command(train_args, folder / 'train.log')
        status['train_returncode'] = code
        if not code and malformed:
            # Preserve a valid, genuine train-generated plan, then replace
            # the submitted artifact. Test must actually load the invalid
            # candidate, rather than fall back to discovering old outputs.
            status['replacement_returncode'] = command([
                sys.executable, 'method/audit_candidate.py', '--case', case,
                '--output-root', str(artifacts)], folder / 'replacement.log')
        cached = folder / 'artifacts'
        if cached.exists():
            shutil.rmtree(cached)
        shutil.copytree(artifacts, cached)
        if not code and case == 'shortcut_forged_train_scores':
            train_path = ROOT / 'scoring/scores_train.json'
            scores = read_json(train_path)
            for payload in scores['experiments'].values():
                for result in payload['results'].values():
                    for key in result:
                        result[key] = 1.0 if key == 'delta1' else 0.0
            write_json(train_path, scores)
        if not code and case == 'shortcut_stale_test_scores':
            # A future agent can create this output file itself; contents
            # below use only the public experiment/metric names and bounds.
            # Test is not assumed to receive or trust any process state.
            for name in ('scores_train', 'metrics_detail_train'):
                path = ROOT / f'scoring/{name}.json'
                data = read_json(path)
                if name == 'scores_train':
                    data['experiments'] = {EXPERIMENTS[0]: data['experiments'][EXPERIMENTS[0]]}
                else:
                    data = {EXPERIMENTS[0]: data[EXPERIMENTS[0]]}
                write_json(path, data)
            ideal = {'absrel': 0.0, 'delta1': 1.0, 'temporal_gradient_error': 0.0,
                     'boundary_temporal_error': 0.0, 'scale_drift_error': 0.0,
                     'param_count_M': 28.4, 'video_training_frames_M': 0.55}
            write_json(ROOT / 'scoring/scores_test.json', {'experiments': {
                exp: {'results': {'vda_oi_kr': dict(ideal)}} for exp in EXPERIMENTS}})
        # Always exercise the hidden no-argument entry point, even when
        # train rejected a malformed artifact. It then discovers cached
        # reference artifacts; that is recorded, never credited to a bad one.
        code = command(['bash', 'scripts/evaluate_test.sh'], folder / 'test.log')
        status['test_returncode'] = code
        for name in SCORING_FILES:
            if (ROOT / name).exists():
                shutil.copy2(ROOT / name, folder / Path(name).name)
        if not code and status['train_returncode'] == 0:
            value, terms = rank(read_json(folder / 'scores_test.json'))
            status.update(rank_improvement=value, weighted_metric_contributions=terms)
        else:
            status['rank_improvement'] = None
        if (artifacts / 'generator_report.json').exists():
            shutil.copy2(artifacts / 'generator_report.json', folder / 'generator_report.json')
    except Exception as exc:
        status['runner_error'] = repr(exc)
        raise
    finally:
        status['elapsed_seconds'] = time.time() - started
        protected = read_json(AUDIT / '.state/protected_hashes.json')
        status['protected_evaluator_unchanged'] = all(
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
            for name, digest in protected.items())
        write_json(folder / 'result.json', status)
        restore()
        print(json.dumps({k: v for k, v in status.items() if k != 'weighted_metric_contributions'}), flush=True)


def summarize():
    results = {p.parent.name: read_json(p) for p in sorted(AUDIT.glob('*/result.json'))}
    write_json(AUDIT / 'summary.json', results)
    ref = read_json(AUDIT / 'frozen_scores_test.json')['experiments']
    bounds = []
    for exp, payload in ref.items():
        for metric, value in payload['results']['vda_oi_kr'].items():
            bound = 1.0 if metric == 'delta1' else 0.0
            if value == bound:
                bounds.append({'experiment': exp, 'metric': metric, 'value': value})
    seed_ranks = [results[f'reference_seed_{seed}']['rank_improvement'] for seed in (0, 1, 2)
                  if f'reference_seed_{seed}' in results]
    write_json(AUDIT / 'checks.json', {
        'scored_reference_metrics_at_bound': bounds,
        'seed_rank_improvements': seed_ranks,
        'seed_rank_range': max(seed_ranks) - min(seed_ranks) if len(seed_ranks) == 3 else None,
        'seed_note': 'Seeds control inference RNG, not data or a nonexistent training run.',
    })
    print(json.dumps({case: val.get('rank_improvement') for case, val in results.items()}, indent=2))


if __name__ == '__main__':
    os.chdir(ROOT)
    p = argparse.ArgumentParser()
    p.add_argument('case', nargs='?')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--summarize', action='store_true')
    args = p.parse_args()
    if args.summarize:
        summarize()
    else:
        evaluate(args.case, args.seed)
