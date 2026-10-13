import json, time, uuid
import mlflow
import pytest
from mlops_platform.config import MODEL_NAME
from mlops_platform.storage import db, sql, model, state
from mlops_platform.registry.canary import (
    start_canary,
    select_model,
    observe,
    rollout,
    complete_canary,
    check_canary,
)
from mlops_platform.pipelines.retraining import inspect_batch, train_proposal
from mlops_platform.data.ingest import simulate, save_dataset


def release_copy(champion):
    client = mlflow.MlflowClient()
    source = client.get_model_version(MODEL_NAME, champion["registry_version"])
    version = client.create_model_version(
        MODEL_NAME, source.source, run_id=champion["run_id"]
    )
    mid = str(uuid.uuid4())
    with db() as conn:
        sql(
            conn,
            "INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                mid,
                champion["kind"],
                champion["dataset"],
                champion["run_id"],
                str(version.version),
                champion["artifact"],
                json.dumps(champion["metrics"]),
                json.dumps(champion["validation"]),
                champion["threshold"],
                champion["schema_version"],
                time.time(),
            ),
        )
    return mid


def test_canary_approval_errors_rollback_and_label_gate():
    champion = model(state("champion"))
    candidate = release_copy(champion)
    with pytest.raises(ValueError, match="approval"):
        start_canary(candidate)
    rid = start_canary(candidate, approved=True)["rollout"]["id"]
    routes = [select_model(str(i))[0] for i in range(1000)]
    assert 150 < routes.count(candidate) < 250
    assert select_model("stable") == select_model("stable")
    assert not complete_canary(rid, approved=True)["passed"]
    # Real observed candidate errors cross the minimum request threshold.
    for i in range(30):
        observe(rid, candidate, None, i != 0, 10)
    assert rollout(rid)["status"] == "aborted"
    assert state("champion") == champion["id"] and not state("active_rollout")
    assert select_model("stable") == (champion["id"], None)
    # Success needs labels for both concurrent cohorts and explicit full-rollout approval.
    rid = start_canary(candidate, approved=True)["rollout"]["id"]
    for mid in (candidate, champion["id"]):
        for i in range(40):
            pid = str(uuid.uuid4())
            label = i % 2
            risk = 0.9 if label else 0.01
            with db() as conn:
                sql(
                    conn,
                    "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?)",
                    (pid, mid, "{}", risk, label, label, 10, time.time()),
                )
            observe(rid, mid, pid, True, 10)
    assert check_canary(rid)["passed"]
    with pytest.raises(ValueError, match="approval"):
        complete_canary(rid)
    assert complete_canary(rid, approved=True)["passed"]
    assert state("champion") == candidate


def test_drift_proposal_cooldown_and_unapproved_data():
    champion = model(state("champion"))
    shifted = save_dataset(simulate(300, 213, True), {"seed": 213})
    review = inspect_batch(champion["dataset"], shifted["id"])
    assert review["triggered"]
    assert inspect_batch(champion["dataset"], shifted["id"])["cooldown"]
    with pytest.raises(ValueError, match="approved"):
        train_proposal(review["proposal"])
    assert state("champion") == champion["id"]


def test_recent_labeled_degradation_triggers_without_feature_drift():
    champion = model(state("champion"))
    # Latest 500 wrong, labeled predictions replace older good observations.
    with db() as conn:
        for i in range(500):
            actual = i % 2
            risk = 0.01 if actual else 0.99
            sql(
                conn,
                "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?)",
                (
                    str(uuid.uuid4()),
                    champion["id"],
                    "{}",
                    risk,
                    1 - actual,
                    actual,
                    1,
                    time.time() + i,
                ),
            )
    review = inspect_batch(champion["dataset"], champion["dataset"])
    assert not review["report"]["drift_detected"]
    assert review["triggered"]
    from mlops_platform.pipelines.retraining import proposals

    proposal = next(p for p in proposals() if p["id"] == review["proposal"])
    assert "Delayed-label recall degraded" in proposal["reason"]
    assert "Delayed-label calibration degraded" in proposal["reason"]
