import mlflow, math, threading
from functools import wraps
from ..config import GATES, SCHEMA_VERSION, TRACKING_URI, MODEL_NAME
from ..storage import model, state, db, sql, audit

_promotion_lock = threading.RLock()


def synchronized(function):
    @wraps(function)
    def call(*args, **kwargs):
        with _promotion_lock:
            return function(*args, **kwargs)

    return call


def gate(candidate, champion=None):
    m = candidate["metrics"]
    failures = []
    if any(
        not isinstance(m.get(k), (int, float))
        or not math.isfinite(m[k])
        or not 0 <= m[k] <= 1
        for k in ("pr_auc", "recall", "precision", "brier")
    ):
        return {
            "passed": False,
            "reasons": ["Evaluation metrics must be finite probabilities"],
            "gates": GATES,
        }
    if candidate["schema_version"] != SCHEMA_VERSION:
        failures.append("Feature schema is incompatible")
    for metric in ["pr_auc", "recall", "precision"]:
        if m[metric] < GATES["min_" + metric]:
            failures.append(
                f"{metric} {m[metric]:.3f} < required {GATES['min_' + metric]:.3f}"
            )
    if m["brier"] > GATES["max_brier"]:
        failures.append("Brier score exceeds the maximum")
    if champion:
        c = champion["metrics"]
        # Stored metrics cannot fairly compare different test populations.
        if champion["dataset"] != candidate["dataset"]:
            failures.append(
                "Different dataset: a common held-out comparison is required"
            )
        for metric in ["pr_auc", "recall"]:
            if m[metric] < c[metric] - GATES["max_" + metric + "_regression"]:
                failures.append(metric + " regressed beyond the allowed limit")
        if m["brier"] > c["brier"] + GATES["max_brier_regression"]:
            failures.append("Calibration regressed beyond the allowed limit")
    return {"passed": not failures, "reasons": failures, "gates": GATES}


def assess(mid, ignore_champion=False):
    candidate = model(mid)
    current = state("champion")
    champion = model(current) if current else None
    comparison = None
    if champion and not ignore_champion and candidate["dataset"] != champion["dataset"]:
        import json, joblib
        from pathlib import Path
        from ..data.ingest import load_dataset
        from ..data.preprocess import temporal_split
        from ..training.evaluate import evaluate
        from ..config import FEATURES

        _, _, test = temporal_split(load_dataset(candidate["dataset"]))
        lineage = json.loads(
            Path(champion["artifact"]).with_suffix(".lineage.json").read_text()
        )
        if set(lineage["train_ids"]) & set(test.record_id):
            result = {
                "passed": False,
                "reasons": [
                    "Common evaluation data overlaps champion training records"
                ],
            }
            audit("promotion_rejected", {"candidate": mid, **result})
            return result
        estimator = joblib.load(champion["artifact"])
        comparison = evaluate(
            test.failure,
            estimator.predict_proba(test[FEATURES])[:, 1],
            champion["threshold"],
        )
        champion = {**champion, "dataset": candidate["dataset"], "metrics": comparison}
    result = gate(candidate, None if ignore_champion else champion)
    if comparison:
        result["champion_on_candidate_holdout"] = comparison
    return result


@synchronized
def promote(mid, approved=False, rollback=False):
    if not approved:
        raise ValueError("Explicit approval is required")
    candidate = model(mid)
    current = state("champion")
    result = assess(mid, ignore_champion=rollback)
    if not result["passed"]:
        audit("promotion_rejected", {"candidate": mid, **result})
        return result
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.MlflowClient().set_registered_model_alias(
        MODEL_NAME, "champion", candidate["registry_version"]
    )
    with db() as conn:
        sql(conn, "UPDATE state SET value=? WHERE key=?", (current, "previous"))
        sql(conn, "UPDATE state SET value=? WHERE key=?", (mid, "champion"))
        audit(
            "rollback" if rollback else "promotion",
            {
                "from": current,
                "to": mid,
                "registry_version": candidate["registry_version"],
            },
            conn,
        )
    return {**result, "champion": mid}


def rollback(approved=False):
    previous = state("previous")
    if not previous:
        raise ValueError("No previous approved model is available")
    return promote(previous, approved=approved, rollback=True)
