#!/bin/bash
# Hardening job: reproduce the reference on the fixed gym, then audit it.
#   1. stale/forged score files are planted, then the hidden test runs with no
#      arguments (official scoring/scores_test.json) and the plant is checked
#   2. visible train split (official scoring/scores_train.json / scores.json)
#   3. audit: variants and shortcuts through the same hidden-test evaluator
#   4. rank analysis + validate.py

set -uo pipefail

cd /home/user

LOG=tmp/hardening_logs
mkdir -p "$LOG"
nvidia-smi -L 2>/dev/null || echo "no GPU visible"
test -f data/downloads/video_depth_anything_vits.pth || { echo "missing checkpoint" >&2; exit 1; }

start=$(date +%s)
python redteam/hardening/forge_scores.py write
bash scripts/evaluate_test.sh > "$LOG/official_test.log" 2>&1
test_rc=$?
echo "official test rc=$test_rc after $(( $(date +%s) - start ))s"
grep -E "^\[test\]" "$LOG/official_test.log" | tail -n 40
python redteam/hardening/forge_scores.py check

bash scripts/evaluate_train.sh > "$LOG/official_train.log" 2>&1
train_rc=$?
echo "official train rc=$train_rc after $(( $(date +%s) - start ))s"
grep -E "^\[train\]" "$LOG/official_train.log" | tail -n 20
cp scoring/scores_train.json scoring/scores.json
cp scoring/metrics_detail_train.json scoring/metrics_detail.json

python eval/test/evaluate.py \
  --predictor-root redteam/hardening/predictors \
  --output redteam/hardening/audit_scores_test.json \
  --detail-output redteam/hardening/audit_detail_test.json > "$LOG/audit_test.log" 2>&1
audit_rc=$?
echo "audit rc=$audit_rc after $(( $(date +%s) - start ))s"
grep -E "^\[test\]" "$LOG/audit_test.log" | tail -n 80

python redteam/hardening/analyze.py
python validate.py --compare
echo "done after $(( $(date +%s) - start ))s (test=$test_rc train=$train_rc audit=$audit_rc)"
[ "$test_rc" -eq 0 ] && [ "$train_rc" -eq 0 ]
