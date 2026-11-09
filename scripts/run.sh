#!/bin/bash
# Job script: reproduce the reference end to end on the compute node.
#
#   visible train split -> scoring/scores_train.json (+ scoring/scores.json)
#   hidden test split   -> scoring/scores_test.json
#
# Inference only (released VDA-S checkpoint); runs every registered predictor
# (method/predictors/*.json, baseline/predictors/*.json). About 50 minutes on
# one A100. The hardening audit is redteam/hardening/run_audit.sh.

set -euo pipefail

cd /home/user

bash scripts/reproduce.sh
