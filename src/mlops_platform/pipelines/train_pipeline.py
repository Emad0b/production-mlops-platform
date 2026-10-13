from ..storage import state, model, audit
from ..training.train import train


def approved_training():
    dataset = state("approved_dataset")
    if not dataset:
        raise ValueError("No dataset has been explicitly approved")
    outputs = train(dataset)
    audit(
        "retraining_proposed",
        {
            "dataset": dataset,
            "candidates": [m["id"] for m in outputs],
            "auto_promoted": False,
        },
    )
    return outputs
