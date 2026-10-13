import json, time, uuid, joblib
import mlflow, mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ..config import TRACKING_URI, MODEL_NAME, ARTIFACTS, FEATURES, SCHEMA_VERSION
from ..storage import db, sql, audit
from ..data.ingest import load_dataset
from ..data.preprocess import temporal_split
from .evaluate import evaluate, choose_threshold


def tracking():
    mlflow.set_tracking_uri(TRACKING_URI)
    if not mlflow.get_experiment_by_name("ForgeOps / Predictive maintenance"):
        mlflow.create_experiment(
            "ForgeOps / Predictive maintenance", artifact_location=ARTIFACTS.as_uri()
        )
    mlflow.set_experiment("ForgeOps / Predictive maintenance")


def train(dataset_id):
    tracking()
    frame = load_dataset(dataset_id)
    train_set, val, test = temporal_split(frame)
    definitions = [
        ("fixed-baseline", DummyClassifier(strategy="prior")),
        (
            "logistic",
            make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
            ),
        ),
        ("random-forest", None),
    ]
    outputs = []
    for kind, estimator in definitions:
        params = {}
        if kind == "random-forest":
            options = []
            for depth in [5, 9]:
                candidate = RandomForestClassifier(
                    n_estimators=180,
                    max_depth=depth,
                    min_samples_leaf=8,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                )
                candidate.fit(train_set[FEATURES], train_set.failure)
                score = evaluate(
                    val.failure, candidate.predict_proba(val[FEATURES])[:, 1]
                )["pr_auc"]
                options.append((score, depth, candidate))
            _, depth, estimator = max(options, key=lambda x: x[0])
            params = {
                "max_depth": depth,
                "n_estimators": 180,
                "selection": "validation PR-AUC",
            }
        else:
            estimator.fit(train_set[FEATURES], train_set.failure)
        probability = estimator.predict_proba(val[FEATURES])[:, 1]
        threshold = (
            0.5
            if kind == "fixed-baseline"
            else choose_threshold(val.failure, probability)
        )
        validation = evaluate(val.failure, probability, threshold)
        metrics = evaluate(
            test.failure, estimator.predict_proba(test[FEATURES])[:, 1], threshold
        )
        mid = str(uuid.uuid4())
        path = ARTIFACTS / (mid + ".joblib")
        joblib.dump(estimator, path)
        path.with_suffix(".lineage.json").write_text(
            json.dumps(
                {
                    "dataset": dataset_id,
                    "train_ids": train_set.record_id.tolist(),
                    "validation_ids": val.record_id.tolist(),
                    "test_ids": test.record_id.tolist(),
                }
            )
        )
        with mlflow.start_run(run_name=kind) as run:
            mlflow.log_params(
                {
                    "dataset_sha256": dataset_id,
                    "seed": 42,
                    "schema_version": SCHEMA_VERSION,
                    "threshold": threshold,
                    "split": "chronological 60/20/20",
                    **params,
                }
            )
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            )
            mlflow.log_dict(
                {
                    "held_out_test": metrics,
                    "validation": validation,
                    "train_rows": len(train_set),
                    "validation_rows": len(val),
                    "test_rows": len(test),
                    "features": FEATURES,
                },
                "evaluation.json",
            )
            info = mlflow.sklearn.log_model(
                estimator,
                name="maintenance-model",
                signature=infer_signature(
                    train_set[FEATURES], estimator.predict(train_set[FEATURES])
                ),
                input_example=train_set[FEATURES].head(2),
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
            version = mlflow.register_model(info.model_uri, MODEL_NAME)
            client = mlflow.MlflowClient()
            client.set_model_version_tag(
                MODEL_NAME, version.version, "schema_version", SCHEMA_VERSION
            )
            client.set_model_version_tag(
                MODEL_NAME, version.version, "dataset_sha256", dataset_id
            )
            with db() as conn:
                sql(
                    conn,
                    "INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        mid,
                        kind,
                        dataset_id,
                        run.info.run_id,
                        str(version.version),
                        str(path),
                        json.dumps(metrics),
                        json.dumps(validation),
                        threshold,
                        SCHEMA_VERSION,
                        time.time(),
                    ),
                )
            outputs.append(
                {
                    "id": mid,
                    "kind": kind,
                    "metrics": metrics,
                    "validation": validation,
                    "threshold": threshold,
                    "registry_version": str(version.version),
                    "run_id": run.info.run_id,
                }
            )
    audit(
        "training_completed",
        {"dataset": dataset_id, "models": [m["id"] for m in outputs]},
    )
    return outputs
