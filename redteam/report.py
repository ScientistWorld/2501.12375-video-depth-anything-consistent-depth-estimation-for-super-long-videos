"""Summarize recorded JSON scores; no inference or evaluator changes."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def mean_metrics(scores, name='vda_oi_kr'):
    experiments = scores['experiments']
    keys = next(iter(experiments.values()))['results'][name]
    return {metric: sum(exp['results'][name][metric] for exp in experiments.values()) / len(experiments)
            for metric in keys}


def main():
    frozen = read(ROOT / 'frozen_scores_test.json')
    spec = read(ROOT / 'rank_spec.json')['experiments']
    seed_scores = [read(ROOT / f'reference_seed_{seed}/scores_test.json') for seed in range(3)]
    seed_spreads = {exp: {metric: max(s['experiments'][exp]['results']['vda_oi_kr'][metric] for s in seed_scores)
                        - min(s['experiments'][exp]['results']['vda_oi_kr'][metric] for s in seed_scores)
                        for metric in frozen['experiments'][exp]['results']['vda_oi_kr']}
                    for exp in frozen['experiments']}
    summary = read(ROOT / 'summary.json')
    means = {case: mean_metrics(read(ROOT / case / 'scores_test.json'))
             for case, result in summary.items() if result['test_returncode'] == 0}
    baseline_rank = sum(config['weight'] * sum(
        mc['coefficient'] * (frozen['experiments'][exp]['results']['vda_independent_windows'][metric]
                             - frozen['experiments'][exp]['results']['vda_oi_kr'][metric])
        for metric, mc in config['metrics'].items()) for exp, config in spec.items())
    headroom = sum(config['weight'] * sum(
        mc['coefficient'] * ((1.0 if metric == 'delta1' else 0.0)
                             - frozen['experiments'][exp]['results']['vda_oi_kr'][metric])
        for metric, mc in config['metrics'].items()
        if metric not in ('param_count_M', 'video_training_frames_M'))
        for exp, config in spec.items())
    info = {
        'job_id': '2eb74ca0-118', 'completed_runs': len(summary),
        'mean_metrics_by_case': means,
        'frozen_baseline_mean_metrics': mean_metrics(frozen, 'vda_independent_windows'),
        'frozen_baseline_rank_improvement': baseline_rank,
        'seed_metric_ranges': seed_spreads,
        'all_seed_scores_identical': seed_scores[0] == seed_scores[1] == seed_scores[2],
        'fresh_reference_rank_offset': summary['reference_seed_0']['rank_improvement'],
        'perfect_depth_gain_at_reference_resources': headroom,
        'constant_quality_penalty_at_reference_resources': summary['constant_fixed_budget']['rank_improvement'],
        'zero_resource_gain': 0.2 * 28.4 + 0.2 * 0.55,
        'all_protected_evaluator_hashes_unchanged': all(
            result['protected_evaluator_unchanged'] for result in summary.values()),
        'stale_score_shortcut_recomputed_experiments': list(read(
            ROOT / 'shortcut_stale_test_scores/metrics_detail_test.json')),
    }
    (ROOT / 'analysis.json').write_text(json.dumps(info, indent=2) + '\n')
    for case, result in summary.items():
        if result['test_returncode']:
            log = (ROOT / case / 'test.log').read_text().strip().splitlines()
            rejection = {'test_returncode': result['test_returncode'],
                         'rank_improvement': None, 'test_error': log[-1],
                         'note': 'The hidden test loaded the planned candidate and rejected it; no test scores were emitted.'}
            (ROOT / case / 'test_rejection.json').write_text(json.dumps(rejection, indent=2) + '\n')
    print(json.dumps({key: info[key] for key in (
        'completed_runs', 'all_seed_scores_identical', 'perfect_depth_gain_at_reference_resources',
        'all_protected_evaluator_hashes_unchanged')}, indent=2))


if __name__ == '__main__':
    main()
