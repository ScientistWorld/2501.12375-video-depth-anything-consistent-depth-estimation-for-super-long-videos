#!/bin/bash
set -euo pipefail
cd /home/user
python redteam/audit.py shortcut_metadata_negative --seed 0
