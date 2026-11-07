#!/bin/bash
set -euo pipefail
cd /home/user
python redteam/audit.py shortcut_stale_test_scores --seed 0
