# Benchmark audit harness

Run `bash redteam/batch.sh` in a scheduled GPU job. Each case has its own
`run.sh`, installs its candidate as `method/audit_candidate.py`, writes real
prediction artifacts, runs the visible train evaluator, and invokes exactly
`bash scripts/evaluate_test.sh` with no arguments in an independent process.
The unmodified hidden evaluator and entry point are hashed for every run.
No evaluator process, environment, or Python installation is patched.

`variant.py` is candidate code and uses only future-agent-visible files.
`audit.py` is the auditor's isolation and reporting harness; its frozen test
scores and rank specification are used only after evaluation to calculate rank.
Each rank is a weighted sum of the seven metric improvements relative to the
frozen reproduced `vda_oi_kr` test scores; positive beats the reference, zero ties.
Paper values in `rank_spec.json` are not used as the reproduced reference.

The original files are snapshotted in ignored `.state/` and restored after each
run, including on failure. Large artifacts and logs are ignored; per-run scores,
detailed metrics, results, and scripts are retained. Re-running a case regenerates
its artifacts. This initial audit does not tune candidates on hidden feedback.

Malformed-output shortcuts first establish a valid train plan, then overwrite
the planned prediction artifacts. Thus the hidden test actually attempts to
load those outputs rather than silently discovering the cached reference.

`reference_seed_*` perform fresh inference with seeds 0, 1, and 2; this inference
benchmark has no training step. `weaker_no_stabilization` runs the independent
window path. Bidirectional and flip ensembles run extra real VDA inference and
align its predictions to the original predictions without labels. The three
frame filter is a fixed local temporal denoiser, also without labels.
