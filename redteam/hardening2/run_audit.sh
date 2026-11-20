#!/bin/bash
# Hardening audit (iteration 12): real-video hidden test.
#   0. pre-flight (data on disk, poses vs ground-truth depth, exact-label metrics)
#   1. forged score files are planted, then the hidden test runs with no
#      arguments (official scoring/scores_test.json) and the plant is checked
#   2. visible train split (official scoring/scores_train.json / scores.json)
#   3. audit: variants and shortcuts through the same hidden-test evaluator
#      (an agent's own copy of the public benchmarks is simulated in tmp/public_copy)
#   4. rank analysis + validate.py
# Run as a GPU job: bash redteam/hardening2/run_audit.sh

set -uo pipefail

cd /home/user

LOG=redteam/hardening2/logs
mkdir -p "$LOG"
nvidia-smi -L 2>/dev/null || echo "no GPU visible"
test -f data/downloads/video_depth_anything_vits.pth || { echo "missing checkpoint" >&2; exit 1; }
start=$(date +%s)

python redteam/hardening2/smoke.py 2>&1 | tee "$LOG/smoke.log"
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "pre-flight failed"; exit 1; }
echo "smoke done after $(( $(date +%s) - start ))s"

python redteam/hardening2/forge_scores.py write
bash scripts/evaluate_test.sh > "$LOG/official_test.log" 2>&1
test_rc=$?
echo "official test rc=$test_rc after $(( $(date +%s) - start ))s"
grep -E "^\[test\]" "$LOG/official_test.log" | tail -n 30
python redteam/hardening2/forge_scores.py check

bash scripts/evaluate_train.sh > "$LOG/official_train.log" 2>&1
train_rc=$?
echo "official train rc=$train_rc after $(( $(date +%s) - start ))s"
grep -E "^\[train\]" "$LOG/official_train.log" | tail -n 20
cp scoring/scores_train.json scoring/scores.json
cp scoring/metrics_detail_train.json scoring/metrics_detail.json

rm -rf tmp/public_copy tmp/hardening2_cache
cp -al eval/test/downloads tmp/public_copy 2>/dev/null || cp -a eval/test/downloads tmp/public_copy
python eval/test/evaluate.py --no-default-roots --predictor-root redteam/hardening2/predictors \
  --output redteam/hardening2/audit_scores_test.json \
  --detail-output redteam/hardening2/audit_detail_test.json > "$LOG/audit_test.log" 2>&1
audit_rc=$?
rm -rf tmp/public_copy tmp/hardening2_cache
echo "audit rc=$audit_rc after $(( $(date +%s) - start ))s"
grep -E "^\[test\]" "$LOG/audit_test.log" | tail -n 60

python redteam/hardening2/analyze.py
python validate.py --compare
echo "done after $(( $(date +%s) - start ))s (test=$test_rc train=$train_rc audit=$audit_rc)"
[ "$test_rc" -eq 0 ] && [ "$train_rc" -eq 0 ]
