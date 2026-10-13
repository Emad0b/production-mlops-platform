from prefect import flow, task
from ..data.ingest import simulate, save_dataset, load_dataset
from .retraining import inspect_batch, run_approved_proposals
from ..data.versioning import version


@task(retries=2, retry_delay_seconds=2)
def ingest(rows, seed, shifted):
    return save_dataset(
        simulate(rows, seed, shifted),
        {"generator": "simulator-v1", "seed": seed, "shifted": shifted},
    )


@task
def validate_manifest(dataset):
    load_dataset(dataset["id"])
    version(dataset["id"])
    return dataset["id"]


@task
def drift_and_proposal(reference, current):
    return inspect_batch(reference, current)


@task
def approved_retraining():
    return run_approved_proposals()


@flow(name="sensor-ingestion-and-lifecycle-review", log_prints=True)
def lifecycle_flow(reference_id, rows=1200, seed=77, shifted=True):
    current = validate_manifest(ingest(rows, seed, shifted))
    review = drift_and_proposal(reference_id, current)
    trained = approved_retraining()
    return {"review": review, "trained": trained, "automatic_promotion": False}
