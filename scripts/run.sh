#!/bin/bash
# Job script: reproduce the reference end to end on the compute node.
#
#   visible train clips -> scoring/scores_train.json (+ scoring/scores.json)
#   hidden test clips   -> scoring/scores_test.json
#
# Inference only (released VDA-S checkpoint); runs every registered predictor
# (method/predictors/*.json, baseline/predictors/*.json) on real Bonn RGB-D
# Dynamic and KITTI clips. Needs scripts/download.sh first and one GPU.
# The iteration-12 hardening audit is redteam/hardening2/run_audit.sh.

set -euo pipefail

cd /home/user

bash redteam/hardening2/run_audit.sh
