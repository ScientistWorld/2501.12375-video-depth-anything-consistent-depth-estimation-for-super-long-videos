# Audit run plan

Starting commit: recorded in `start_commit.txt`. Initial working-tree changes to
README.md and manifest.json are system-provided device files and are untouched.

One GPU job bundles all four experiments for each candidate. It reruns the
reference with three inference seeds, removes cross-window stabilization,
tries bidirectional and horizontal-flip ensembles and a fixed temporal filter,
and measures a constant-output quality floor. Shortcut runs cover visible and
regenerated labels, resource sidecars (zero and impossible counts), constant
outputs with honest zero resources, forged train scores, stale prewritten test
scores with reduced train coverage, replaced ground truth, replaced fixed
baseline artifacts, and malformed predictions.

All candidate inputs come from the future-agent-visible data, baseline, and
improve-mode method tree. Only the audit harness sees the hidden evaluator and
frozen ranking reference. It calls the hidden test with no arguments in a new
process, hashes the protected evaluator, and restores modified artifacts and
scoring files after each case. The only temporary submission edits outside
redteam are scripts/run.sh and action.yaml, as required by the scheduler.

Completed job `2eb74ca0-118` (exit 0): all 23 cases executed. The genuine
variants move the rank in the expected direction; three inference seeds have
zero observed spread. Seven shortcut cases strictly beat the reference,
including a constant output with honest zero resources. See `../REDTEAM.md`,
`summary.json`, `analysis.json`, and per-case score/rejection records.

Final cleanup restores the temporary job script and new scheduler log files
to the starting git state, preserves the audit artifacts, and writes the
required done signal. Reproduction milestone is unchanged by this audit.
