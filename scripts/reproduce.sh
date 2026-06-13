#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -euo pipefail

cd /home/user

rm -rf baseline/outputs method/outputs
rm -f scoring/scores.json scoring/metrics_detail.json

bash scripts/baseline.sh vda_independent_windows
bash scripts/method.sh
bash scripts/evaluate.sh vda_independent_windows vda_oi_kr
