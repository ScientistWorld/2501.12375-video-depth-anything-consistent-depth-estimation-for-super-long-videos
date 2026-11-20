"""Rank every audited case against the reference and check paired noise.

Rank (as specified for this gym): for each experiment e (weight w_e) and scored
metric m (coefficient c_m), the improvement over the reference is taken in the
metric's better direction and divided by the metric's spread (std) across the
honest attempts; dR = sum_e w_e sum_m |c_m| * improvement / spread.

Paired check: every case saw the same hidden inputs, so the same rank is also
computed per video (videos of an experiment weighted equally), giving a mean
and a standard error over videos.

Inputs: scoring/scores_test.json + scoring/metrics_detail_test.json (official
run: reference and baseline) and redteam/hardening2/audit_*.json (audit run).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "redteam" / "hardening2"
REF = "vda_oi_kr"
HONEST = ("vda_oi_kr", "vda_independent_windows", "ref_repeat", "better_flip_ensemble", "better_bidirectional", "better_flip_bidir",
          "better_temporal3", "weaker_scale_jitter")


def merged() -> tuple[dict, dict, dict]:
    off_s = json.loads((ROOT / "scoring/scores_test.json").read_text())
    off_d = json.loads((ROOT / "scoring/metrics_detail_test.json").read_text())
    scores = {"experiments": {e: {"results": dict(p["results"])} for e, p in off_s["experiments"].items()},
              "_status": dict(off_s["_status"])}
    detail = {e: dict(p["results"]) for e, p in off_d["experiments"].items()}
    audit = HERE / "audit_scores_test.json"
    if audit.exists():
        a_s = json.loads(audit.read_text())
        a_d = json.loads((HERE / "audit_detail_test.json").read_text())
        for e, p in a_s["experiments"].items():
            scores["experiments"][e]["results"].update(p["results"])
            detail[e].update(a_d["experiments"][e]["results"])
        scores["_status"].update(a_s["_status"])
    return scores, detail, off_d


def main() -> None:
    reference = json.loads((ROOT / "scoring/reference.json").read_text())["experiments"]
    scores, detail, off_d = merged()
    exps = scores["experiments"]
    names = list(next(iter(exps.values()))["results"])

    spread = {e: {m: float(np.std([exps[e]["results"][n][m] for n in HONEST if n in exps[e]["results"]])) or 1.0
                  for m in reference[e]["metrics"]} for e in exps}

    def contrib(e: str, row: dict, ref_row: dict) -> dict:
        out = {}
        for m, spec in reference[e]["metrics"].items():
            sign = 1.0 if spec["higher_is_better"] else -1.0
            out[m] = abs(spec["coefficient"]) * sign * (row[m] - ref_row[m]) / spread[e][m]
        return out

    results = {}
    for n in names:
        total, per_exp, pm, pv = 0.0, {}, 0.0, 0.0
        ok = scores["_status"][n]["status"] == "ok"
        for e, payload in exps.items():
            w = reference[e]["weight"]
            c = contrib(e, payload["results"][n], payload["results"][REF])
            per_exp[e] = {"metrics": payload["results"][n], "contributions": c, "weighted_sum": w * sum(c.values())}
            total += w * sum(c.values())
            rows, ref_rows = detail[e][n]["per_video"], detail[e][REF]["per_video"]
            if ok and rows is not None:
                per_video = np.array([sum(contrib(e, rows[i], ref_rows[i]).values()) for i in ref_rows])
                pm += w * per_video.mean()
                pv += w ** 2 * (per_video.var(ddof=1) / len(per_video))
                per_exp[e]["paired_mean"] = float(per_video.mean())
                per_exp[e]["paired_se"] = float(np.sqrt(per_video.var(ddof=1) / len(per_video)))
                per_exp[e]["videos_better"] = int((per_video > 0).sum())
                per_exp[e]["videos"] = int(len(per_video))
            else:
                pm = float("nan")
        results[n] = {"status": scores["_status"][n], "delta_rank": total, "paired_mean": pm,
                      "paired_se": float(np.sqrt(pv)), "per_experiment": per_exp}

    direction = {}
    for e in exps:
        base, ref = exps[e]["results"]["vda_independent_windows"], exps[e]["results"][REF]
        direction[e] = {m: {"independent_windows": base[m], "oi_kr": ref[m],
                            "oi_kr_better": (ref[m] > base[m]) if reference[e]["metrics"][m]["higher_is_better"]
                            else (ref[m] < base[m])} for m in reference[e]["metrics"]}
    repro = {e: {m: {"official": exps[e]["results"][REF][m], "repeat": exps[e]["results"].get("ref_repeat", {}).get(m)}
                 for m in reference[e]["metrics"]} for e in exps}

    out = {"spread": spread, "cases": results, "paper_direction": direction, "reference_reproducibility": repro,
           "levels": {e: off_d["experiments"][e]["levels"] for e in exps}, "sandbox": off_d.get("sandbox")}
    (HERE / "results.json").write_text(json.dumps(out, indent=1))

    hdr = " | ".join(f"{e}: absrel / delta1 / tae / drift" for e in exps)
    lines = [f"| case | status | dR | paired mean ± SE | {hdr} |", "|---|---|---:|---:|" + "---|" * len(exps)]
    for n, r in sorted(results.items(), key=lambda kv: -kv[1]["delta_rank"]):
        cells = []
        for e in exps:
            mm = r["per_experiment"][e]["metrics"]
            cells.append(f"{mm['absrel']:.4f} / {mm['delta1']:.4f} / {mm['tae']:.3f} / {mm['scale_drift']:.2f}")
        pm = "n/a" if not np.isfinite(r["paired_mean"]) else f"{r['paired_mean']:+.3f} ± {r['paired_se']:.3f}"
        lines.append(f"| {n} | {r['status']['status']} | {r['delta_rank']:+.3f} | {pm} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Per-experiment paired mean ± SE (videos better / videos):")
    for n, r in sorted(results.items(), key=lambda kv: -kv[1]["delta_rank"]):
        parts = [f"{e} {r['per_experiment'][e]['paired_mean']:+.3f} ± {r['per_experiment'][e]['paired_se']:.3f} "
                 f"({r['per_experiment'][e]['videos_better']}/{r['per_experiment'][e]['videos']})"
                 for e in exps if "paired_mean" in r["per_experiment"][e]]
        if parts:
            lines.append(f"- {n}: " + "; ".join(parts))
    (HERE / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({"paper_direction": direction, "reference_reproducibility": repro, "sandbox": out["sandbox"]},
                     indent=1))


if __name__ == "__main__":
    main()
