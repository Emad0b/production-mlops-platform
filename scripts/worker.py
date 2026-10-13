import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prefect import flow
from mlops_platform.pipelines.retraining import run_approved_proposals


@flow(name="approved-retraining-scheduler")
def retraining_review():
    return run_approved_proposals()


if __name__ == "__main__":
    retraining_review.serve(name="approved-proposal-worker", interval=60)
