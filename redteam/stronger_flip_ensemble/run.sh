#!/bin/bash
set -euo pipefail
cd /home/user
python redteam/audit.py stronger_flip_ensemble --seed 0
