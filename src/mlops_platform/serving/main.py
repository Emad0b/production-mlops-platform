import hmac, json, time, uuid, threading, joblib
from functools import lru_cache
import pandas as pd
from fastapi import FastAPI, Depends, Header, HTTPException, BackgroundTasks, Response
from fastapi.responses import FileResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from ..config import FEATURES, DATA, RUNTIME, ADMIN_TOKEN, SCHEMA_VERSION
from ..storage import db, sql, models, model, state, audit
from ..data.ingest import simulate, save_dataset, load_dataset
from ..training.train import train
from ..registry.promote import gate, promote, rollback
from ..monitoring.drift import monitor
from ..monitoring.performance import performance
from .schemas import Reading, Prepare, DatasetChoice, ModelChoice, MonitorRequest, Label
from typing import Annotated
import math, os, logging
from prometheus_client import Gauge
from ..registry.canary import (
    select_model,
    observe,
    start_canary,
    check_canary,
    abort_canary,
    complete_canary,
    all_rollouts,
)
from ..pipelines.retraining import inspect_batch, proposals, train_proposal

ERRORS = Counter(
    "mlops_platform_prediction_errors_total", "Failed predictions", ["model_id"]
)
DRIFT = Gauge(
    "mlops_platform_drifted_features",
    "Features that exceeded the most recent drift threshold",
)
CANARY = Gauge("mlops_platform_canary_active", "A canary deployment is active")
DEGRADED = Gauge(
    "mlops_platform_performance_degraded",
    "Recent labeled champion quality exceeded retraining thresholds",
)
logger = logging.getLogger("mlops_platform.requests")

app = FastAPI(title="ForgeOps / Predictive Maintenance", version="0.1.0")
COUNTER = Counter(
    "mlops_platform_predictions_total", "Scored sensor readings", ["model_id"]
)
LATENCY = Histogram(
    "mlops_platform_prediction_seconds",
    "Prediction request computation duration",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
)
_pipeline_lock = threading.Lock()
_job = {"status": "idle", "error": None}


def admin(x_admin_token: str | None = Header(default=None)):
    if (
        not ADMIN_TOKEN
        or not x_admin_token
        or not hmac.compare_digest(ADMIN_TOKEN, x_admin_token)
    ):
        raise HTTPException(403, "An administrator token is required")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "champion": state("champion") or None,
        "schema": SCHEMA_VERSION,
        "pipeline": _job,
        "storage": "postgresql"
        if __import__("mlops_platform.storage", fromlist=["PG"]).PG
        else "sqlite",
    }


@app.get("/metrics")
def metrics():
    CANARY.set(bool(state("active_rollout")))
    with db() as conn:
        latest = conn.execute(
            "SELECT payload FROM reports WHERE kind='drift' ORDER BY created DESC LIMIT 1"
        ).fetchone()
    if latest:
        DRIFT.set(sum(f["drifted"] for f in json.loads(latest[0])["features"]))
    champion = state("champion")
    measured = performance(champion) if champion else None
    from ..pipelines.retraining import POLICY

    degraded = False
    if (
        measured
        and measured["status"] == "measured"
        and measured["metrics"]["samples"] >= POLICY["min_labeled_samples"]
    ):
        current = measured["metrics"]
        baseline = model(champion)["metrics"]
        degraded = (
            current["recall"] < baseline["recall"] - POLICY["max_recall_drop"]
            or current["brier"] > baseline["brier"] + POLICY["max_brier_increase"]
        )
    DEGRADED.set(degraded)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/overview")
def overview():
    allmodels = models()
    champion = state("champion")
    with db() as conn:
        readings = conn.execute(
            "SELECT model_id,risk,predicted,label,latency,created FROM predictions ORDER BY created DESC LIMIT 500"
        ).fetchall()
        count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        reports = conn.execute(
            "SELECT kind,payload FROM reports ORDER BY created DESC LIMIT 20"
        ).fetchall()
        events = conn.execute(
            "SELECT action,payload,created FROM audit ORDER BY created DESC LIMIT 20"
        ).fetchall()
    times = sorted(r[4] for r in readings)
    return {
        "champion": champion or None,
        "previous": state("previous") or None,
        "models": [{**m, "gate": gate(m)} for m in allmodels],
        "predictions": count,
        "recent_predictions": [
            dict(
                model_id=r[0],
                risk=r[1],
                predicted=r[2],
                label=r[3],
                latency_ms=r[4],
                created=r[5],
            )
            for r in readings[:80]
        ],
        "p95_ms": times[int((len(times) - 1) * 0.95)] if times else None,
        "datasets": [json.loads(p.read_text()) for p in DATA.glob("*.json")],
        "reports": [{"kind": kind, **json.loads(payload)} for kind, payload in reports],
        "audit": [
            {"action": a, "payload": json.loads(p), "created": t} for a, p, t in events
        ],
        "pipeline": _job,
        "label_performance": performance(champion) if champion else None,
        "rollouts": [{**r, "checks": check_canary(r["id"])} for r in all_rollouts()],
        "proposals": proposals(),
    }


@app.post("/api/datasets/prepare", dependencies=[Depends(admin)])
def prepare(req: Prepare):
    return save_dataset(
        simulate(req.rows, req.seed, req.shifted),
        {
            "generator": "simulator-v1",
            "seed": req.seed,
            "shifted": req.shifted,
            "rows": req.rows,
        },
    )


@app.post("/api/datasets/approve", dependencies=[Depends(admin)])
def approve_dataset(req: DatasetChoice):
    try:
        load_dataset(req.dataset_id)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))
    with db() as conn:
        sql(
            conn,
            "UPDATE state SET value=? WHERE key=?",
            (req.dataset_id, "approved_dataset"),
        )
        audit("dataset_approved", {"dataset": req.dataset_id}, conn)
    return {"approved": req.dataset_id}


def execute_training(dataset_id):
    try:
        _job.update(status="running", error=None)
        train(dataset_id)
        _job.update(status="completed", error=None)
    except Exception as error:
        import logging

        logging.exception("Training failed")
        _job.update(status="failed", error=str(error))
    finally:
        _pipeline_lock.release()


@app.post("/api/train", dependencies=[Depends(admin)], status_code=202)
def training(background: BackgroundTasks):
    dataset = state("approved_dataset")
    if not dataset:
        raise HTTPException(409, "Approve a dataset before training")
    if not _pipeline_lock.acquire(blocking=False):
        raise HTTPException(409, "A training pipeline is already running")
    _job.update(status="queued", error=None)
    background.add_task(execute_training, dataset)
    return {"status": "queued", "dataset": dataset, "auto_promote": False}


@app.post("/api/models/promote", dependencies=[Depends(admin)])
def promotion(req: ModelChoice):
    if state("champion"):
        raise HTTPException(
            409, "Use an approved canary rollout for subsequent releases"
        )
    try:
        result = promote(req.model_id, approved=True)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not result["passed"]:
        raise HTTPException(409, result)
    return result


@app.post("/api/models/rollback", dependencies=[Depends(admin)])
def model_rollback():
    try:
        if state("active_rollout"):
            return abort_canary(
                state("active_rollout"), ["Operator requested rollback"]
            )
        return rollback(approved=True)
    except ValueError as e:
        raise HTTPException(409, str(e))


@lru_cache(maxsize=8)
def load_model(mid):
    return joblib.load(model(mid)["artifact"])


@app.post("/api/predict")
def predict(
    req: Reading, request_id: Annotated[str | None, Header(alias="X-Request-Id")] = None
):
    start = time.perf_counter()
    request_id = (request_id or str(uuid.uuid4()))[:128]
    mid, rid = select_model(request_id)
    if not mid:
        raise HTTPException(503, "No approved champion model is deployed")
    try:
        metadata = model(mid)
        if (
            os.getenv("MLOPS_ENABLE_FAULT_INJECTION") == "1"
            and state("demo_fault_model") == mid
        ):
            raise RuntimeError("Deliberately injected candidate failure")
        estimator = load_model(mid)
        reading = req.model_dump()
        risk = float(
            estimator.predict_proba(pd.DataFrame([reading], columns=FEATURES))[0, 1]
        )
        if not math.isfinite(risk) or not 0 <= risk <= 1:
            raise ValueError("Invalid model probability")
    except Exception:
        latency = (time.perf_counter() - start) * 1000
        ERRORS.labels(model_id=mid).inc()
        observe(rid, mid, None, False, latency)
        logger.exception(
            "Prediction failed request_id=%s model_id=%s rollout=%s",
            request_id,
            mid,
            rid,
        )
        raise HTTPException(
            503, "Model inference failed; rollout checks recorded the failure"
        )
    predicted = int(risk >= metadata["threshold"])
    pid = str(uuid.uuid4())
    latency = (time.perf_counter() - start) * 1000
    with db() as conn:
        sql(
            conn,
            "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?)",
            (
                pid,
                mid,
                json.dumps(reading),
                risk,
                predicted,
                None,
                latency,
                time.time(),
            ),
        )
    COUNTER.labels(model_id=mid).inc()
    LATENCY.observe((time.perf_counter() - start))
    observe(rid, mid, pid, True, (time.perf_counter() - start) * 1000)
    logger.info(
        "Prediction request_id=%s prediction_id=%s model_id=%s rollout=%s latency_ms=%.3f",
        request_id,
        pid,
        mid,
        rid,
        latency,
    )
    return {
        "request_id": request_id,
        "rollout_id": rid,
        "prediction_id": pid,
        "risk": risk,
        "failure_predicted": bool(predicted),
        "threshold": metadata["threshold"],
        "model_id": mid,
        "registry_version": metadata["registry_version"],
        "schema": SCHEMA_VERSION,
        "latency_ms": round(latency, 3),
    }


@app.post("/api/labels", dependencies=[Depends(admin)])
def label(req: Label):
    with db() as conn:
        row = sql(
            conn, "SELECT label FROM predictions WHERE id=?", (req.prediction_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "Prediction not found")
        if row[0] is not None:
            raise HTTPException(409, "This prediction already has a ground-truth label")
        sql(
            conn,
            "UPDATE predictions SET label=? WHERE id=?",
            (req.failure, req.prediction_id),
        )
        audit(
            "label_received",
            {"prediction": req.prediction_id, "failure": req.failure},
            conn,
        )
    rid = state("active_rollout")
    if rid:
        checks = check_canary(rid)
        if checks["reasons"]:
            abort_canary(rid, checks["reasons"])
    return {"recorded": True}


@app.post("/api/monitor", dependencies=[Depends(admin)])
def monitoring(req: MonitorRequest):
    try:
        return inspect_batch(req.reference_id, req.current_id)["report"]
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/reports/{rid}/html")
def report_html(rid: str):
    try:
        uuid.UUID(rid)
    except ValueError:
        raise HTTPException(404, "Report not found")
    path = RUNTIME / "reports" / (rid + ".html")
    if not path.exists():
        raise HTTPException(404, "Report not found")
    return FileResponse(path, media_type="text/html")


@app.post("/api/rollouts/start", dependencies=[Depends(admin)])
def rollout_start(req: ModelChoice):
    try:
        result = start_canary(req.model_id, approved=True)
        if not result["passed"]:
            raise HTTPException(409, result)
        return result
    except ValueError as error:
        raise HTTPException(409, str(error))


@app.post("/api/rollouts/{rid}/complete", dependencies=[Depends(admin)])
def rollout_complete(rid: str):
    try:
        result = complete_canary(rid, approved=True)
        if not result["passed"]:
            raise HTTPException(409, result)
        return result
    except ValueError as error:
        raise HTTPException(409, str(error))


@app.post("/api/rollouts/{rid}/abort", dependencies=[Depends(admin)])
def rollout_abort(rid: str):
    try:
        return abort_canary(rid, ["Operator requested rollback"])
    except ValueError as error:
        raise HTTPException(404, str(error))


def execute_proposal(pid):
    try:
        _job.update(status="running", error=None)
        train_proposal(pid)
        _job.update(status="completed", error=None)
    except Exception as error:
        logger.exception("Retraining failed")
        _job.update(status="failed", error=str(error))
    finally:
        _pipeline_lock.release()


@app.post("/api/retraining/{pid}/train", dependencies=[Depends(admin)], status_code=202)
def proposal_training(pid: str, background: BackgroundTasks):
    proposal = next((p for p in proposals() if p["id"] == pid), None)
    if not proposal or proposal["status"] != "pending":
        raise HTTPException(409, "An eligible pending proposal is required")
    if state("approved_dataset") != proposal["dataset"]:
        raise HTTPException(409, "Approve this proposal dataset first")
    if not _pipeline_lock.acquire(blocking=False):
        raise HTTPException(409, "A training pipeline is already running")
    _job.update(status="queued", error=None)
    background.add_task(execute_proposal, pid)
    return {"status": "queued", "proposal": pid, "auto_promote": False}


@app.post("/api/datasets/ingest", dependencies=[Depends(admin)])
async def ingest_csv(request: __import__("fastapi").Request):
    from io import BytesIO

    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > 5_000_000:
            raise HTTPException(413, "CSV exceeds the 5 MB limit")
    try:
        manifest = save_dataset(
            pd.read_csv(BytesIO(data)), {"source": "authenticated-csv-ingestion"}
        )
        from ..data.versioning import version

        version(manifest["id"])
        return manifest
    except Exception as error:
        raise HTTPException(422, f"CSV validation failed: {type(error).__name__}")
