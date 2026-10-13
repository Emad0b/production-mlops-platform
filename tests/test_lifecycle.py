from fastapi.testclient import TestClient
from mlops_platform.serving.main import app
from mlops_platform.data.ingest import simulate, save_dataset
from mlops_platform.training.train import train
from mlops_platform.registry.promote import promote, rollback, gate
from mlops_platform.storage import model, state
from mlops_platform.monitoring.drift import monitor


def test_train_register_gate_prediction_labels_drift_and_rollback():
    data = save_dataset(simulate(1200, 42), {"seed": 42, "shifted": False})
    outputs = train(data["id"])
    assert len(outputs) == 3 and all(o["registry_version"] for o in outputs)
    baseline = next(o for o in outputs if o["kind"] == "fixed-baseline")
    rejected = promote(baseline["id"], approved=True)
    assert not rejected["passed"] and not state("champion")
    approved = [model(o["id"]) for o in outputs if gate(model(o["id"]))["passed"]]
    assert approved, "No candidate passed fixed gates on the deterministic test fixture"
    selected = max(approved, key=lambda m: m["validation"]["pr_auc"])
    assert promote(selected["id"], approved=True)["passed"]
    with TestClient(app) as client:
        assert (
            client.post(
                "/api/models/promote", json={"model_id": selected["id"]}
            ).status_code
            == 403
        )
        valid = {
            "temperature": 85,
            "vibration": 4,
            "pressure": 5.3,
            "rpm": 1800,
            "operating_hours": 9000,
        }
        assert (
            client.post("/api/predict", json={**valid, "temperature": 999}).status_code
            == 422
        )
        assert (
            client.post("/api/predict", json={**valid, "failure": 1}).status_code == 422
        )
        result = client.post("/api/predict", json=valid).json()
        assert 0 <= result["risk"] <= 1 and result["model_id"] == selected["id"]
        assert client.get("/metrics").status_code == 200
        headers = {"X-Admin-Token": "integration-test-token"}
        body = {"prediction_id": result["prediction_id"], "failure": 1}
        assert client.post("/api/labels", json=body, headers=headers).status_code == 200
        assert client.post("/api/labels", json=body, headers=headers).status_code == 409
    shifted = save_dataset(simulate(300, 99, True), {"seed": 99, "shifted": True})
    before = state("champion")
    report = monitor(data["id"], shifted["id"])
    assert report["drift_detected"] and report["evidently_html"]
    assert state("champion") == before
    # Register the same evaluated artifact as a distinct release to exercise real version rollback.
    import uuid, json, time, mlflow
    from mlops_platform.config import MODEL_NAME
    from mlops_platform.storage import db, sql

    client = mlflow.MlflowClient()
    source = client.get_model_version(MODEL_NAME, selected["registry_version"])
    release = client.create_model_version(
        MODEL_NAME, source.source, run_id=selected["run_id"]
    )
    new_id = str(uuid.uuid4())
    with db() as conn:
        sql(
            conn,
            "INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                new_id,
                selected["kind"],
                selected["dataset"],
                selected["run_id"],
                str(release.version),
                selected["artifact"],
                json.dumps(selected["metrics"]),
                json.dumps(selected["validation"]),
                selected["threshold"],
                selected["schema_version"],
                time.time(),
            ),
        )
    assert promote(new_id, approved=True)["passed"]
    assert state("champion") == new_id
    assert rollback(approved=True)["passed"]
    assert state("champion") == selected["id"]
    assert (
        str(client.get_model_version_by_alias(MODEL_NAME, "champion").version)
        == selected["registry_version"]
    )
