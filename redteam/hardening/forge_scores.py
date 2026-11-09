"""Stale/forged score check: pre-write ideal score files, then confirm the
hidden-test run replaces them with recomputed numbers.

    python redteam/hardening/forge_scores.py write   # before bash scripts/evaluate_test.sh
    python redteam/hardening/forge_scores.py check   # after it
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IDEAL = {"absrel": 0.0, "delta1": 1.0, "tae": 0.0, "scale_drift": 0.0}
EXPS = ("long_video_500", "super_long_video_1500", "synthetic_long_video_128", "synthetic_long_video_256")


def forged() -> dict:
    return {"experiments": {e: {"results": {"vda_oi_kr": dict(IDEAL), "forged_method": dict(IDEAL)}} for e in EXPS}}


def main() -> None:
    mode = sys.argv[1]
    test_path = ROOT / "scoring/scores_test.json"
    train_path = ROOT / "scoring/scores_train.json"
    if mode == "write":
        test_path.write_text(json.dumps(forged(), indent=2))
        train_path.write_text(json.dumps(forged(), indent=2))
        print("forged scores written")
        return
    scores = json.loads(test_path.read_text())
    exps = scores.get("experiments", {})
    survivors = []
    for exp, payload in exps.items():
        for name, metrics in payload.get("results", {}).items():
            if name == "forged_method" or metrics == IDEAL:
                survivors.append(f"{exp}/{name}")
    result = {
        "experiments_after_run": sorted(exps),
        "forged_entries_surviving": survivors,
        "stale_experiments_surviving": sorted(set(exps) - {"long_video_500", "super_long_video_1500"}),
        "passed": not survivors and set(exps) == {"long_video_500", "super_long_video_1500"},
    }
    print(json.dumps(result, indent=2))
    (ROOT / "redteam/hardening/stale_check.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
