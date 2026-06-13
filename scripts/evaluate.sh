#!/bin/bash
# Evaluation script — the standard way to evaluate work in this environment.
#
# This script must be SELF-CONTAINED using only code in eval/.
# Do NOT import from method/ — evaluate.sh must work even when the
# paper's method code is hidden from a scientist agent.
#
# Design it so that anyone following the same output format can run
# this script to evaluate their results.
#
# OUTPUT CONTRACT: This script MUST write scoring/scores.json containing
# a JSON dict nested by experiment, e.g.:
#   {"imagenet_classification": {"top1_accuracy": 82.1, "top5_accuracy": 94.8}}
# Experiment and metric names must match the keys in scoring/reference.json.
# This file is read by the system for automated comparison.
#
# Example usage:
#   ./evaluate.sh <method_and_run_name>                             # evaluate with defaults
#   ./evaluate.sh <method_and_run_name> checkpoints/epoch_10.pt     # evaluate a specific checkpoint
#
# Called from run.sh after training, or independently.

set -euo pipefail

cd /home/user

mkdir -p scoring

args=()
if [ -f baseline/outputs/vda_independent_windows.npz ]; then
  args+=(--prediction vda_independent_windows baseline/outputs/vda_independent_windows.npz)
fi
if [ -f method/outputs/vda_oi_kr.npz ]; then
  args+=(--prediction vda_oi_kr method/outputs/vda_oi_kr.npz)
fi

if [ "${#args[@]}" -eq 0 ]; then
  echo "No prediction artifacts found. Run scripts/baseline.sh or scripts/method.sh first." >&2
  exit 1
fi

python eval/metrics.py "${args[@]}" --gt data/synthetic_video/ground_truth.npz --output scoring/scores.json
