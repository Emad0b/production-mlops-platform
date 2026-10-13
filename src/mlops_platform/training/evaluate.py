import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)


def evaluate(y, probability, threshold=0.5):
    predicted = (np.asarray(probability) >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y, probability)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "brier": float(brier_score_loss(y, probability)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "samples": len(y),
        "failure_rate": float(np.mean(y)),
    }


def choose_threshold(y, probability):
    candidates = np.linspace(0.05, 0.75, 71)
    results = [
        (evaluate(y, probability, t)["f1"], float(t))
        for t in candidates
        if evaluate(y, probability, t)["recall"] >= 0.65
    ]
    if not results:
        return 0.05
    return max(results)[1]
