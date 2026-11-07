#!/bin/bash
set -euo pipefail
cd /home/user
bash redteam/reference_cached/run.sh
bash redteam/reference_seed_0/run.sh
bash redteam/reference_seed_1/run.sh
bash redteam/reference_seed_2/run.sh
bash redteam/weaker_no_stabilization/run.sh
bash redteam/stronger_bidirectional/run.sh
bash redteam/stronger_flip_ensemble/run.sh
bash redteam/real_fix_temporal3/run.sh
bash redteam/constant_fixed_budget/run.sh
bash redteam/shortcut_visible_answers/run.sh
bash redteam/shortcut_generator_answers/run.sh
bash redteam/shortcut_metadata_zero/run.sh
bash redteam/shortcut_metadata_negative/run.sh
bash redteam/shortcut_constant_zero_resources/run.sh
bash redteam/shortcut_forged_train_scores/run.sh
bash redteam/shortcut_stale_test_scores/run.sh
bash redteam/shortcut_ground_truth_replacement/run.sh
bash redteam/shortcut_baseline_replacement/run.sh
bash redteam/shortcut_empty/run.sh
bash redteam/shortcut_nan/run.sh
bash redteam/shortcut_inf/run.sh
bash redteam/shortcut_zero/run.sh
bash redteam/shortcut_negative/run.sh
python redteam/audit.py --summarize
