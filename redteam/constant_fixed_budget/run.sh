#!/bin/bash
set -euo pipefail
cd /home/user
python redteam/audit.py constant_fixed_budget --seed 0
