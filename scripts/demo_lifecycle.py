"""Measured drift -> approved retraining -> rejection -> fault rollback demonstration.

Run bootstrap.py first. The fault test uses a new registry version of the SAME
evaluated champion artifact as a control, explicitly not a retrained challenger.
"""

import sys, os, json, uuid, time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
os.environ["MLOPS_ENABLE_FAULT_INJECTION"] = "1"
os.environ.setdefault("PREFECT_HOME", str(root / "runtime/prefect"))
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
from mlops_platform.config import MODEL_NAME, FEATURES
from mlops_platform.storage import state, model, db, sql, audit
from mlops_platform.pipelines.flows import lifecycle_flow, approved_retraining
from mlops_platform.registry.promote import assess
from mlops_platform.registry.canary import start_canary, rollout
from mlops_platform.serving.main import predict
from mlops_platform.serving.schemas import Reading
from version_dataset import version
import mlflow

if __name__ == "__main__":
    champion = model(state("champion"))
    version(champion["dataset"])
    review = lifecycle_flow(champion["dataset"])
    current = review["review"]["report"]["current_dataset"]
    version(current)
    with db() as conn:
        sql(conn, "UPDATE state SET value=? WHERE key=?", (current, "approved_dataset"))
        audit(
            "dataset_approved",
            {"dataset": current, "actor": "explicit-demo-script"},
            conn,
        )
    # Use a real Prefect flow for retraining, with tracked task runs.
    from prefect import flow

    @flow(name="explicitly-approved-demo-retraining")
    def retrain():
        return approved_retraining()

    trained = retrain()
    assessments = [
        {**m, "assessment": assess(m["id"])}
        for batch in trained
        for m in batch["models"]
    ]
    client = mlflow.MlflowClient()
    source = client.get_model_version(MODEL_NAME, champion["registry_version"])
    release = client.create_model_version(
        MODEL_NAME, source.source, run_id=champion["run_id"]
    )
    control = str(uuid.uuid4())
    with db() as conn:
        sql(
            conn,
            "INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                control,
                champion["kind"],
                champion["dataset"],
                champion["run_id"],
                str(release.version),
                champion["artifact"],
                json.dumps(champion["metrics"]),
                json.dumps(champion["validation"]),
                champion["threshold"],
                champion["schema_version"],
                time.time(),
            ),
        )
        sql(
            conn,
            "INSERT INTO state VALUES ('demo_fault_model',?) ON CONFLICT (key) DO UPDATE SET value=excluded.value",
            (control,),
        )
    rid = start_canary(control, approved=True)["rollout"]["id"]
    failed = 0
    for i in range(1000):
        if not state("active_rollout"):
            break
        try:
            predict(
                Reading(
                    temperature=82,
                    vibration=4.2,
                    pressure=5.3,
                    rpm=1850,
                    operating_hours=8200,
                ),
                request_id=f"fault-demo-{i}",
            )
        except __import__("fastapi").HTTPException:
            failed += 1
    with db() as conn:
        sql(conn, "UPDATE state SET value='' WHERE key='demo_fault_model'")
    assert rollout(rid)["status"] == "aborted"
    assert state("champion") == champion["id"]
    report = {
        "data": "Self-authored simulated equipment readings; no real equipment forecast",
        "prefect_flow": review,
        "retrained_challengers": assessments,
        "fault_control": {
            "same_artifact_as_champion": True,
            "registry_version": str(release.version),
            "injected_candidate_errors": failed,
        },
        "rollout": rollout(rid),
        "champion_preserved": state("champion") == champion["id"],
        "automatic_promotion": False,
    }
    (root / "reports/lifecycle-demo.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "challengers": len(assessments),
                "rejections": sum(not m["assessment"]["passed"] for m in assessments),
                "injected_candidate_errors": failed,
                "rollout_status": rollout(rid)["status"],
                "champion_preserved": report["champion_preserved"],
            },
            indent=2,
        )
    )
