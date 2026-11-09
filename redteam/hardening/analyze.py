"""Rank every audited case against the reference and check paired noise.

Rank (as specified for this gym): for each experiment e (weight w_e) and scored
metric m (coefficient c_m), improvement over the reference is taken in the
metric's better direction and divided by the metric's spread (std) across the
honest attempts; dR = sum_e w_e sum_m |c_m| * improvement / spread.

Paired check: the same rank is computed per video (every case saw the same
hidden inputs), giving a mean and standard error over videos.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "redteam" / "hardening"
REF = "vda_oi_kr"
HONEST = ("vda_oi_kr", "vda_independent_windows", "ref_repeat", "better_flip_ensemble", "better_bidirectional",
          "better_temporal3", "weaker_scale_jitter")


def main() -> None:
    reference = json.loads((ROOT / "scoring/reference.json").read_text())["experiments"]
    scores = json.loads((HERE / "audit_scores_test.json").read_text())
    detail = json.loads((HERE / "audit_detail_test.json").read_text())
    official = json.loads((ROOT / "scoring/scores_test.json").read_text())
    exps = scores["experiments"]
    names = list(next(iter(exps.values()))["results"])

    spread = {}
    for e, payload in exps.items():
        spread[e] = {}
        for m in reference[e]["metrics"]:
            vals = [payload["results"][n][m] for n in HONEST if n in payload["results"]]
            spread[e][m] = float(np.std(vals)) or 1.0

    def contrib(e: str, row: dict, ref_row: dict) -> dict:
        out = {}
        for m, spec in reference[e]["metrics"].items():
            sign = 1.0 if spec["higher_is_better"] else -1.0
            out[m] = abs(spec["coefficient"]) * sign * (row[m] - ref_row[m]) / spread[e][m]
        return out

    results = {}
    for n in names:
        total = 0.0
        per_exp = {}
        paired_mean = 0.0
        paired_var = 0.0
        for e, payload in exps.items():
            w = reference[e]["weight"]
            c = contrib(e, payload["results"][n], payload["results"][REF])
            per_exp[e] = {"metrics": payload["results"][n], "contributions": c, "weighted_sum": w * sum(c.values())}
            total += w * sum(c.values())
            pv = detail["experiments"][e]["results"][n]["per_video"]
            pr = detail["experiments"][e]["results"][REF]["per_video"]
            status = scores["_status"][n]["status"]
            if pv is not None and status == "ok":
                per_video = np.array([sum(contrib(e, pv[i], pr[i]).values()) for i in pr])
                paired_mean += w * per_video.mean()
                paired_var += w**2 * (per_video.var(ddof=1) / len(per_video) if len(per_video) > 1 else 0.0)
            else:
                paired_mean = float("nan")
        results[n] = {
            "status": scores["_status"][n],
            "delta_rank": total,
            "paired_mean": paired_mean,
            "paired_se": float(np.sqrt(paired_var)),
            "per_experiment": per_exp,
        }

    # Reproducibility of the frozen reference: official run vs audit run vs fresh repeat.
    repro = {}
    for e in exps:
        frozen = official["experiments"][e]["results"][REF]
        repro[e] = {m: {"frozen": frozen[m], "audit": exps[e]["results"][REF][m],
                        "repeat": exps[e]["results"].get("ref_repeat", {}).get(m)} for m in frozen}

    direction = {}
    for e in exps:
        base = exps[e]["results"]["vda_independent_windows"]
        ref = exps[e]["results"][REF]
        direction[e] = {m: {"independent_windows": base[m], "oi_kr": ref[m],
                            "oi_kr_better": (ref[m] > base[m]) if reference[e]["metrics"][m]["higher_is_better"]
                            else (ref[m] < base[m])} for m in reference[e]["metrics"]}

    out = {"spread": spread, "cases": results, "reference_reproducibility": repro, "paper_direction": direction,
           "levels": {e: detail["experiments"][e]["levels"] for e in exps}}
    (HERE / "results.json").write_text(json.dumps(out, indent=1))

    lines = ["| case | status | dR | paired mean ± SE | " + " | ".join(f"{e}: absrel / delta1 / tae / drift" for e in exps) + " |",
             "|---|---|---:|---:|" + "---|" * len(exps)]
    for n, r in sorted(results.items(), key=lambda kv: -kv[1]["delta_rank"]):
        cells = []
        for e in exps:
            mm = r["per_experiment"][e]["metrics"]
            cells.append(f"{mm['absrel']:.4f} / {mm['delta1']:.4f} / {mm['tae']:.3f} / {mm['scale_drift']:.3f}")
        pm = "n/a" if not np.isfinite(r["paired_mean"]) else f"{r['paired_mean']:+.3f} ± {r['paired_se']:.3f}"
        lines.append(f"| {n} | {r['status']['status']} | {r['delta_rank']:+.3f} | {pm} | " + " | ".join(cells) + " |")
    (HERE / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({"paper_direction": direction, "reference_reproducibility": repro}, indent=1))


if __name__ == "__main__":
    main()
