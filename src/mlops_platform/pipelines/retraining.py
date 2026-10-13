import json, time, uuid
from ..config import ROOT
from ..storage import db, sql, state, model, audit
from ..monitoring.performance import performance
from ..monitoring.drift import monitor
from ..data.ingest import load_dataset
from ..training.train import train

POLICY = json.loads((ROOT / "configs/retraining.json").read_text())


def proposals():
    with db() as conn:
        rows = conn.execute("SELECT * FROM proposals ORDER BY created DESC").fetchall()
    return [
        dict(
            id=r[0],
            dataset=r[1],
            reason=json.loads(r[2]),
            status=r[3],
            models=json.loads(r[4]) if r[4] else [],
            created=r[5],
        )
        for r in rows
    ]


def inspect_batch(reference, current):
    report = monitor(reference, current)
    champion = state("champion")
    performance_report = performance(champion) if champion else None
    reasons = []
    if (
        sum(f["drifted"] for f in report["features"])
        >= POLICY["minimum_drifted_features"]
    ):
        reasons.append("Feature drift exceeded the configured trigger")
    if (
        performance_report
        and performance_report["status"] == "measured"
        and performance_report["metrics"]["samples"] >= POLICY["min_labeled_samples"]
    ):
        observed = performance_report["metrics"]
        baseline = model(champion)["metrics"]
        if observed["recall"] < baseline["recall"] - POLICY["max_recall_drop"]:
            reasons.append("Delayed-label recall degraded")
        if observed["brier"] > baseline["brier"] + POLICY["max_brier_increase"]:
            reasons.append("Delayed-label calibration degraded")
    if not reasons:
        return {"report": report, "proposal": None, "triggered": False}
    with db() as conn:
        duplicate = sql(
            conn,
            "SELECT id FROM proposals WHERE dataset=? AND created>?",
            (current, time.time() - POLICY["cooldown_seconds"]),
        ).fetchone()
        if duplicate:
            return {
                "report": report,
                "proposal": duplicate[0],
                "triggered": False,
                "cooldown": True,
            }
        pid = str(uuid.uuid4())
        sql(
            conn,
            "INSERT INTO proposals VALUES (?,?,?,?,?,?)",
            (pid, current, json.dumps(reasons), "pending", None, time.time()),
        )
        audit(
            "retraining_proposed",
            {"proposal": pid, "dataset": current, "reasons": reasons},
            conn,
        )
    return {"report": report, "proposal": pid, "triggered": True}


def train_proposal(pid):
    proposal = next((p for p in proposals() if p["id"] == pid), None)
    if not proposal:
        raise ValueError("Proposal not found")
    if proposal["status"] not in ("pending", "approved"):
        raise ValueError("Proposal is not eligible for training")
    if state("approved_dataset") != proposal["dataset"]:
        raise ValueError(
            "The dataset must be explicitly approved before automatic training"
        )
    # Claim once, preventing two orchestrators from executing the same proposal.
    with db() as conn:
        result = sql(
            conn,
            "UPDATE proposals SET status='training' WHERE id=? AND status IN ('pending','approved')",
            (pid,),
        )
        if result.rowcount != 1:
            raise ValueError("Proposal was already claimed")
    try:
        outputs = train(proposal["dataset"])
        with db() as conn:
            sql(
                conn,
                "UPDATE proposals SET status='trained',models=? WHERE id=?",
                (json.dumps([m["id"] for m in outputs]), pid),
            )
        audit(
            "challengers_registered",
            {
                "proposal": pid,
                "models": [m["id"] for m in outputs],
                "auto_promoted": False,
            },
        )
        return outputs
    except Exception:
        with db() as conn:
            sql(conn, "UPDATE proposals SET status='failed' WHERE id=?", (pid,))
        raise


def run_approved_proposals():
    results = []
    if POLICY["auto_train_approved_data"]:
        for proposal in proposals():
            if (
                proposal["status"] == "pending"
                and state("approved_dataset") == proposal["dataset"]
            ):
                results.append(
                    {
                        "proposal": proposal["id"],
                        "models": train_proposal(proposal["id"]),
                    }
                )
    return results
