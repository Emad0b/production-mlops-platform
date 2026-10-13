import hashlib, json, time, uuid
import mlflow
from ..config import ROOT, MODEL_NAME, TRACKING_URI
from ..storage import db, sql, state, model, audit
from .promote import assess, promote, synchronized
from ..training.evaluate import evaluate

POLICY = json.loads((ROOT / "configs/rollout.json").read_text())


def rollout(rid):
    with db() as conn:
        row = sql(conn, "SELECT * FROM rollouts WHERE id=?", (rid,)).fetchone()
    if not row:
        raise ValueError("Rollout not found")
    result = dict(
        zip(
            [
                "id",
                "candidate",
                "champion",
                "status",
                "fraction",
                "policy",
                "decision",
                "created",
            ],
            row,
        )
    )
    result["policy"] = json.loads(result["policy"])
    result["decision"] = json.loads(result["decision"]) if result["decision"] else None
    return result


def all_rollouts():
    with db() as conn:
        ids = conn.execute(
            "SELECT id FROM rollouts ORDER BY created DESC LIMIT 30"
        ).fetchall()
    return [rollout(r[0]) for r in ids]


@synchronized
def start_canary(mid, approved=False):
    if not approved:
        raise ValueError("Explicit canary deployment approval is required")
    champion = state("champion")
    if not champion:
        raise ValueError("Bootstrap an approved champion first")
    if mid == champion:
        raise ValueError("Candidate must be a distinct model version")
    if state("active_rollout"):
        raise ValueError("A rollout is already active")
    result = assess(mid)
    if not result["passed"]:
        audit("canary_rejected", {"candidate": mid, **result})
        return result
    rid = str(uuid.uuid4())
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.MlflowClient().set_registered_model_alias(
        MODEL_NAME, "candidate", model(mid)["registry_version"]
    )
    with db() as conn:
        sql(
            conn,
            "INSERT INTO rollouts VALUES (?,?,?,?,?,?,?,?)",
            (
                rid,
                mid,
                champion,
                "active",
                POLICY["canary_fraction"],
                json.dumps(POLICY),
                None,
                time.time(),
            ),
        )
        sql(conn, "UPDATE state SET value=? WHERE key=?", (rid, "active_rollout"))
        audit(
            "canary_started",
            {
                "rollout": rid,
                "candidate": mid,
                "champion": champion,
                "fraction": POLICY["canary_fraction"],
            },
            conn,
        )
    return {"passed": True, "rollout": rollout(rid)}


def select_model(request_id):
    rid = state("active_rollout")
    champion = state("champion")
    if not rid:
        return champion, None
    current = rollout(rid)
    bucket = int(hashlib.sha256(request_id.encode()).hexdigest()[:16], 16) / 2**64
    return (
        current["candidate"] if bucket < current["fraction"] else current["champion"]
    ), rid


@synchronized
def abort_canary(rid, reasons):
    current = rollout(rid)
    if current["status"] != "active":
        return current
    decision = {
        "passed": False,
        "reasons": reasons,
        "action": "candidate traffic disabled; champion retained",
    }
    with db() as conn:
        sql(
            conn,
            "UPDATE rollouts SET status=?,decision=? WHERE id=?",
            ("aborted", json.dumps(decision), rid),
        )
        sql(conn, "UPDATE state SET value='' WHERE key='active_rollout'")
        audit("canary_automatically_rolled_back", {"rollout": rid, **decision}, conn)
    return rollout(rid)


def check_canary(rid):
    current = rollout(rid)
    policy = current["policy"]
    with db() as conn:
        observations = sql(
            conn,
            "SELECT ok,latency FROM rollout_observations WHERE rollout=? AND model_id=?",
            (rid, current["candidate"]),
        ).fetchall()
        labeled = sql(
            conn,
            "SELECT p.model_id,p.risk,p.label FROM predictions p JOIN rollout_observations o ON o.prediction_id=p.id WHERE o.rollout=? AND p.label IS NOT NULL",
            (rid,),
        ).fetchall()
    reasons = []
    n = len(observations)
    errors = sum(not r[0] for r in observations) / n if n else 0
    latency = sorted(r[1] for r in observations)
    p95 = latency[int((n - 1) * 0.95)] if n else None
    if n >= policy["min_requests"]:
        if errors > policy["max_error_rate"]:
            reasons.append("Candidate error rate exceeds policy")
        if p95 > policy["max_p95_ms"]:
            reasons.append("Candidate p95 latency exceeds policy")
    metrics = {}
    for mid in [current["candidate"], current["champion"]]:
        rows = [r for r in labeled if r[0] == mid]
        if len(rows) >= policy["min_labels"] and len({r[2] for r in rows}) == 2:
            metrics[mid] = evaluate(
                [r[2] for r in rows], [r[1] for r in rows], model(mid)["threshold"]
            )
    cm = metrics.get(current["candidate"])
    bm = metrics.get(current["champion"])
    if cm:
        if cm["recall"] < policy["min_recall"]:
            reasons.append("Canary recall is below policy")
        if cm["precision"] < policy["min_precision"]:
            reasons.append("Canary precision is below policy")
        if cm["brier"] > policy["max_brier"]:
            reasons.append("Canary calibration exceeds policy")
        if bm and cm["pr_auc"] < bm["pr_auc"] - policy["max_pr_auc_regression"]:
            reasons.append(
                "Canary PR-AUC regressed against the concurrent champion cohort"
            )
    ready = n >= policy["min_requests"] and (
        not policy["require_label_performance"] or (cm is not None and bm is not None)
    )
    return {
        "passed": ready and not reasons,
        "ready": ready,
        "reasons": reasons,
        "candidate_requests": n,
        "error_rate": errors,
        "p95_ms": p95,
        "cohort_metrics": metrics,
        "waiting_for": "More candidate requests and labels for both cohorts"
        if not ready
        else None,
    }


def observe(rid, mid, prediction_id, ok, latency):
    if not rid:
        return
    with db() as conn:
        sql(
            conn,
            "INSERT INTO rollout_observations VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), rid, prediction_id, mid, int(ok), latency, time.time()),
        )
    if rollout(rid)["status"] == "active":
        result = check_canary(rid)
        if result["reasons"]:
            abort_canary(rid, result["reasons"])


@synchronized
def complete_canary(rid, approved=False):
    if not approved:
        raise ValueError("Explicit full-rollout approval is required")
    current = rollout(rid)
    if current["status"] != "active":
        raise ValueError("Rollout is not active")
    result = check_canary(rid)
    if not result["passed"]:
        if result["reasons"]:
            abort_canary(rid, result["reasons"])
        return result
    promotion = promote(current["candidate"], approved=True)
    if not promotion["passed"]:
        return promotion
    with db() as conn:
        sql(
            conn,
            "UPDATE rollouts SET status=?,decision=? WHERE id=?",
            ("completed", json.dumps(result), rid),
        )
        sql(conn, "UPDATE state SET value='' WHERE key='active_rollout'")
        audit(
            "canary_completed", {"rollout": rid, "champion": current["candidate"]}, conn
        )
    return {"passed": True, "rollout": rollout(rid)}
