"""Stale/forged score check: pre-write ideal score files, then confirm the
hidden-test run replaces them with recomputed numbers.

    python redteam/hardening2/forge_scores.py write   # before bash scripts/evaluate_test.sh
    python redteam/hardening2/forge_scores.py check   # after it
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IDEAL = {"absrel": 0.0, "delta1": 1.0, "tae": 0.0, "scale_drift": 0.0}
EXPS = ("bonn_500", "kitti_500", "kitti_1500", "long_video_500", "super_long_video_1500")


def main() -> None:
    mode = sys.argv[1]
    test_path = ROOT / "scoring/scores_test.json"
    train_path = ROOT / "scoring/scores_train.json"
    forged = {"experiments": {e: {"results": {"vda_oi_kr": dict(IDEAL), "forged_method": dict(IDEAL)}} for e in EXPS}}
    if mode == "write":
        test_path.write_text(json.dumps(forged, indent=2))
        train_path.write_text(json.dumps(forged, indent=2))
        print("forged scores written")
        return
    scores = json.loads(test_path.read_text())
    exps = scores.get("experiments", {})
    survivors = [f"{e}/{n}" for e, p in exps.items() for n, m in p.get("results", {}).items()
                 if n == "forged_method" or m == IDEAL]
    result = {"experiments_after_run": sorted(exps), "forged_entries_surviving": survivors, "passed": not survivors}
    (ROOT / "redteam/hardening2/stale_check.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
