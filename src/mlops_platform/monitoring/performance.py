import json
from ..storage import db, sql, model
from ..training.evaluate import evaluate


def performance(mid):
    with db() as conn:
        rows = sql(
            conn,
            "SELECT risk,label FROM predictions WHERE model_id=? AND label IS NOT NULL ORDER BY created DESC LIMIT 500",
            (mid,),
        ).fetchall()
    if len(rows) < 30 or len({r[1] for r in rows}) < 2:
        return {
            "status": "insufficient_labels",
            "labeled_samples": len(rows),
            "model_id": mid,
        }
    metrics = evaluate(
        [r[1] for r in rows], [r[0] for r in rows], model(mid)["threshold"]
    )
    return {
        "status": "measured",
        "model_id": mid,
        "metrics": metrics,
        "note": "Latest 500 labeled predictions; label selection bias can affect these estimates.",
    }
