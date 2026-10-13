import json, time, uuid
import numpy as np
from scipy.stats import ks_2samp
from ..config import FEATURES, RUNTIME
from ..data.ingest import load_dataset
from ..storage import db, sql, state, model, audit


def psi(reference, current):
    cuts = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    cuts[0] = -np.inf
    cuts[-1] = np.inf
    r = np.histogram(reference, bins=cuts)[0] / len(reference)
    c = np.histogram(current, bins=cuts)[0] / len(current)
    r = np.clip(r, 1e-6, None)
    c = np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def monitor(reference_id, current_id, include_evidently=True):
    reference = load_dataset(reference_id)
    current = load_dataset(current_id)
    features = []
    for feature in FEATURES:
        statistic, p = ks_2samp(reference[feature], current[feature])
        value = psi(reference[feature], current[feature])
        edges = np.linspace(
            min(reference[feature].min(), current[feature].min()),
            max(reference[feature].max(), current[feature].max()),
            17,
        )
        counts_r, _ = np.histogram(reference[feature], bins=edges)
        counts_c, _ = np.histogram(current[feature], bins=edges)
        features.append(
            dict(
                name=feature,
                psi=value,
                ks_statistic=float(statistic),
                p_value=float(p),
                drifted=bool(value > 0.2 and p < 0.01 / len(FEATURES)),
                reference_mean=float(reference[feature].mean()),
                current_mean=float(current[feature].mean()),
                distribution={
                    "reference": (100 * counts_r / len(reference)).tolist(),
                    "current": (100 * counts_c / len(current)).tolist(),
                    "edges": edges.tolist(),
                },
            )
        )
    rid = str(uuid.uuid4())
    report = dict(
        id=rid,
        reference_dataset=reference_id,
        current_dataset=current_id,
        features=features,
        drift_detected=any(f["drifted"] for f in features),
        created=time.time(),
        action="Review data quality and delayed-label performance before approving retraining",
        evidently_html=None,
    )
    champion = state("champion")
    if champion:
        import joblib

        estimator = joblib.load(model(champion)["artifact"])
        report["prediction_drift"] = {
            "psi": psi(
                estimator.predict_proba(reference[FEATURES])[:, 1],
                estimator.predict_proba(current[FEATURES])[:, 1],
            ),
            "model_id": champion,
        }
    if include_evidently:
        from evidently import Report, Dataset, DataDefinition
        from evidently.presets import DataDriftPreset

        definition = DataDefinition(numerical_columns=FEATURES)
        result = Report([DataDriftPreset()]).run(
            current_data=Dataset.from_pandas(
                current[FEATURES], data_definition=definition
            ),
            reference_data=Dataset.from_pandas(
                reference[FEATURES], data_definition=definition
            ),
        )
        directory = RUNTIME / "reports"
        directory.mkdir(exist_ok=True)
        result.save_html(str(directory / (rid + ".html")))
        report["evidently_html"] = f"/api/reports/{rid}/html"
    with db() as conn:
        sql(
            conn,
            "INSERT INTO reports VALUES (?,?,?,?)",
            (rid, "drift", json.dumps(report), time.time()),
        )
    audit(
        "drift_review_proposed",
        {
            "report": rid,
            "drift": report["drift_detected"],
            "retraining_auto_approved": False,
        },
    )
    return report
