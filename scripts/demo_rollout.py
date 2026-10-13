"""Send fresh, labeled simulated traffic through a real retrained canary.

Run bootstrap.py and demo_lifecycle.py first. This is an explicitly approved
local demonstration. Quality checks may either promote or reject the challenger.
"""

import sys, json, time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from mlops_platform.config import FEATURES
from mlops_platform.storage import state, model, models
from mlops_platform.registry.promote import assess
from mlops_platform.registry.canary import (
    start_canary,
    rollout,
    check_canary,
    complete_canary,
)
from mlops_platform.serving.main import predict, label
from mlops_platform.serving.schemas import Reading, Label
from mlops_platform.data.ingest import simulate

if __name__ == "__main__":
    champion = model(state("champion"))
    candidates = [
        m
        for m in models()
        if m["dataset"] != champion["dataset"] and assess(m["id"])["passed"]
    ]
    if not candidates:
        raise SystemExit(
            "No retrained candidate passed static gates; rejection is the correct result."
        )
    candidate = max(candidates, key=lambda m: m["validation"]["pr_auc"])
    rid = start_canary(candidate["id"], approved=True)["rollout"]["id"]
    sent = 0
    for i, row in simulate(500, 311, True).iterrows():
        if not state("active_rollout"):
            break
        result = predict(
            Reading(**row[FEATURES].to_dict()), request_id=f"labeled-rollout-{i}"
        )
        label(Label(prediction_id=result["prediction_id"], failure=int(row.failure)))
        sent += 1
    checks = check_canary(rid)
    if rollout(rid)["status"] == "active" and checks["passed"]:
        complete_canary(rid, approved=True)
    report = {
        "synthetic_traffic": True,
        "inprocess_calls_not_http_benchmark": True,
        "candidate": candidate,
        "checks": checks,
        "rollout": rollout(rid),
        "requests": sent,
        "champion_before": champion["id"],
        "champion_after": state("champion"),
        "note": "Same live safety thresholds as the API; no thresholds relaxed for the demo.",
    }
    (root / "reports/rollout-demo.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {"requests": sent, "status": report["rollout"]["status"], "checks": checks},
            indent=2,
        )
    )
