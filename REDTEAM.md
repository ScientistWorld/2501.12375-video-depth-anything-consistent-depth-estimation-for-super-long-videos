# Does this gym reward better research?

**The tested genuine improvements score higher and the key-idea ablation scores lower, but the gym fails overall:** constant depth wins by **+5.408170** through resource savings despite failing spatial depth estimation. The reproduction supports the reported temporal improvement and improves test AbsRel; it does **not** reproduce the reported delta1 improvement, and its scale drift is worse than the baseline.

All numbers below come from completed job `2eb74ca0-118` (exit 0), covering 23 runs and four experiments per full run. Each case invokes exactly `bash scripts/evaluate_test.sh`, with no arguments, in a separate process. No evaluator, Python installation, or evaluator environment was patched. Candidate code uses future-agent-visible inputs; the auditor uses the frozen hidden scores only to calculate rank afterwards. Protected evaluator hashes matched in all 23 runs.

## Do genuine improvements score higher?

Define **ΔR** as the specified weighted sum of coefficient times metric difference from the reproduced `vda_oi_kr` in the original `scoring/scores_test.json`. Larger is better; the reference has ΔR = 0. The four experiment weights are 0.25. The paper numbers embedded in `reference.json` are not substituted for the reproduced reference. Full scores and contributions are retained in each linked run directory; [analysis.json](redteam/analysis.json) and [summary.json](redteam/summary.json) collect the results.

The following table averages each metric over the four experiments. All rows use 28.4M parameters and 0.55M training frames for this controlled quality comparison. The constant row holds those fields fixed solely to isolate its quality penalty.

| Real variant / control | AbsRel ↓ | delta1 ↑ | Temporal ↓ | Boundary ↓ | Drift ↓ | ΔR |
|---|---:|---:|---:|---:|---:|---:|
| [Original reference artifact replay](redteam/reference_cached/run.sh), honest control | 0.133893 | 0.809712 | 0.238197 | 0.342576 | 0.046229 | **0.000000** |
| [Fresh reference inference](redteam/reference_seed_0/run.sh), seed 0 | 0.133898 | 0.809668 | 0.238004 | 0.343726 | 0.046186 | -0.000847 |
| [Remove cross-window stabilization](redteam/weaker_no_stabilization/run.sh) | 0.134805 | 0.813989 | 0.252081 | 1.747143 | 0.036427 | **-1.271684** |
| [Bidirectional ensemble](redteam/stronger_bidirectional/run.sh) | 0.133685 | 0.812396 | 0.231630 | 0.346413 | 0.036866 | **+0.009303** |
| [Horizontal-flip ensemble](redteam/stronger_flip_ensemble/run.sh) | 0.132563 | 0.809359 | 0.221379 | 0.325127 | 0.046577 | **+0.033236** |
| [Three-frame temporal denoiser](redteam/real_fix_temporal3/run.sh) | 0.134024 | 0.809411 | 0.208416 | 0.319747 | 0.046161 | **+0.050107** |
| [Constant inverse depth 0.25, fixed resource fields](redteam/constant_fixed_budget/run.sh) | 0.307942 | 0.359457 | 0.220578 | 0.337308 | 0.125875 | -0.381830 |

The ensembles run additional actual VDA inference and combine predictions without labels. They provide genuine accuracy/consistency improvements with small tradeoffs in other metrics; no strict improvement in every metric is claimed. The fixed denoiser improves temporal error while slightly worsening spatial metrics. The weaker run uses the original checkpoint and independent-window procedure, removing the central long-video stabilization idea. Every genuine variant's rank change exceeds both the observed seed spread and the fresh-versus-cached reference offset.

**Room above the reference:** none of its 28 scored metric entries is at its favorable mathematical bound. AbsRel, temporal error, boundary error, and drift are above zero; delta1 is below one; both resource counts are above zero. Resource counts are at the documented budget ceilings, not their favorable lower bounds. Perfect depth at the reference resource budget would gain **+0.771889**, so accuracy has measurable headroom. The cached independent-window baseline scores **-1.273870**, clearly below the reference. Its delta1 is about 0.81 rather than saturated near one.

**The rank fails the task-quality check.** Removing the resource charges gives any method `0.2 × 28.4 + 0.2 × 0.55 = 5.79` points. A constant program truthfully has zero learned parameters and zero video training frames. Its spatial failure costs only **0.381830** points, so it wins by **5.408170**, even beating perfect depth at the reference resource budget. This is an objective-design failure that needs no false resource declaration or label access. Even with resources held fixed, constant depth ranks above the useful independent-window baseline because window-boundary error dominates the score. Spatial quality is described as a guardrail but is not enforced as one.

## Seed spread and reproduction check

Fresh full inference runs with [seed 0](redteam/reference_seed_0/result.json), [seed 1](redteam/reference_seed_1/result.json), and [seed 2](redteam/reference_seed_2/result.json) each score **-0.0008467990154872364** relative to the frozen reference. Their maximum-minus-minimum rank spread is **0.0**. Every individual scored metric is also identical across those runs. These are genuine RNG-seeded inference reruns; this checkpoint-only gym has no training procedure to reseed.

There is a **0.000846799** offset between the cached reproduction and the fresh reruns, consistent with numerical execution differences; its cause was not isolated. Zero observed seed spread does not establish a zero noise floor across hardware or an uncertainty estimate over new videos. Changes below that observed execution offset should not be treated as demonstrated gains. None of the reported genuine improvements depends on such a small change.

For the original reproduced method and cached independent-window baseline, four-experiment means are:

| Metric | Independent windows | Reproduced method | Reported direction reproduced? |
|---|---:|---:|---|
| AbsRel ↓ | 0.134796 | 0.133893 | Yes, a small improvement |
| delta1 ↑ | 0.814022 | 0.809712 | **No; lower in every experiment** |
| Temporal gradient error ↓ | 0.252132 | 0.238197 | Yes; about 5.53% lower on average |
| Boundary temporal error ↓ | 1.749535 | 0.342576 | Yes for the local consistency target; about 80.42% lower |
| Scale drift error ↓ | 0.036440 | 0.046229 | **Worse in every experiment** |
| Parameters / training frames | 28.4M / 0.55M | 28.4M / 0.55M | Same reported resources |

The visible briefing reports an accuracy and temporal improvement in the paper. This gym supports the core temporal comparison at similar spatial accuracy, not the complete reported spatial improvement. Its synthetic temporal proxy also differs from the paper's reported temporal alignment metric. The fresh weaker run confirms the boundary advantage rather than merely reusing a favorable score record.

## Executed shortcuts

Every row is a runnable shortcut under the linked directory. **All ranks are relative to reference ΔR = 0.** A rejected test emits no candidate test score; its recorded rejection is linked. Visible-answer and generator shortcuts also work without the improve-mode method. Shortcuts that reuse reference predictions use improve-mode-visible artifacts. Large artifacts/logs remain in ignored case subdirectories; scripts, scores, metric details, results, and rejection records are committed.

| Shortcut | What it does | Hidden-test ΔR vs reference 0 | Works? |
|---|---|---:|---|
| [Visible-answer shortcut](redteam/shortcut_visible_answers/run.sh) | Reads visible `ground_truth.npz`; writes its inverse as predictions. | **+0.771889** | **Works**, essentially perfect depth metrics |
| [Generator-answer shortcut](redteam/shortcut_generator_answers/run.sh) | Rebuilds all depths using the visible generator and sequence metadata, then inverts them. Regenerated depths match the visible labels exactly. | **+0.771889** | **Works**; hiding the label file alone would not fix leakage |
| [Zero-resource-metadata shortcut](redteam/shortcut_metadata_zero/run.sh) | Copies unchanged reference predictions and claims zero parameters/training frames in sidecars. | **+5.790000** | **Works**, no depth improvement |
| [Negative-resource-metadata shortcut](redteam/shortcut_metadata_negative/run.sh) | Copies unchanged reference predictions; sets both resource fields to -1,000,000. | **+400005.790000** | **Works**; accepted impossible values make rank unbounded |
| [Constant-output shortcut](redteam/shortcut_constant_zero_resources/run.sh) | Returns inverse depth 0.25 for every pixel/frame and truthfully reports zero learned resources. | **+5.408170** | **Works**, despite AbsRel 0.307942 and delta1 0.359457 |
| [Forged-train-score shortcut](redteam/shortcut_forged_train_scores/run.sh) | Uses weaker cached independent-window predictions, then overwrites train metrics with ideal values. | -1.273870 | **Fails**; hidden test recomputes the metrics |
| [Stale-test-score shortcut](redteam/shortcut_stale_test_scores/run.sh) | Prewrites ideal test scores for all four public experiment names and retains only the first experiment in the train plan. Keeps actual reference predictions. | **+0.507776** | **Works**; only the first experiment is recomputed, three fabricated scores survive |
| [Ground-truth-replacement shortcut](redteam/shortcut_ground_truth_replacement/run.sh) | Reuses 252-size reference predictions at both sizes and replaces visible ground truth with their reciprocal. | **+0.771889** | **Works**; mutable labels move the target to the method |
| [Fixed-baseline-replacement shortcut](redteam/shortcut_baseline_replacement/run.sh) | Replaces the fixed baseline artifacts with constant predictions; the candidate stays the original weaker independent-window output. | -1.273870 | **Fails to beat reference**; comparison artifacts change, but the absolute reference-based rank does not |
| [Empty-output shortcut](redteam/shortcut_empty/run.sh) | Establishes a valid train plan, then replaces its prediction with shape `[0,112,160]`. | [No score; test exit 1](redteam/shortcut_empty/test_rejection.json) | **Fails**, shape mismatch |
| [NaN-output shortcut](redteam/shortcut_nan/run.sh) | Replaces the planned predictions with all NaNs. | [No score; test exit 1](redteam/shortcut_nan/test_rejection.json) | **Fails**, no valid test pixels |
| [Infinite-output shortcut](redteam/shortcut_inf/run.sh) | Replaces the planned predictions with positive infinity. | [No score; test exit 1](redteam/shortcut_inf/test_rejection.json) | **Fails**, no valid test pixels |
| [Zero-output shortcut](redteam/shortcut_zero/run.sh) | Replaces the planned predictions with zeros. | [No score; test exit 1](redteam/shortcut_zero/test_rejection.json) | **Fails**, no positive test pixels |
| [Negative-output shortcut](redteam/shortcut_negative/run.sh) | Replaces the planned predictions with -1 everywhere. | [No score; test exit 1](redteam/shortcut_negative/test_rejection.json) | **Fails**, no positive test pixels |

The original-reference replay's exact tie is an **honest replay control**, not evidence of a new contribution. No shortcut tie is hidden among the successful rows: all seven paying shortcut cases strictly beat the reference.

## What needs to change next

The major fixes are to enforce task-quality/resource constraints before ranking and rescale the objectives; independently validate resource declarations; evaluate against immutable held-out labels with a genuinely unseen generator/scene configuration; and rebuild scores from an empty output using a fixed complete experiment roster. Restoring the evaluator alone leaves its writable data, metadata, train plan, and output cache inputs exposed. Fixed-baseline replacement did not pay under this absolute rank, so it is a lower priority than the demonstrated failures.

All candidate side effects were restored after each run. Final cleanup restores the temporary job script and new scheduler log files to the starting git state; the audit additions remain under `redteam/` and this report, with the required final `action.yaml` signal.
