"""Send a reproducible shifted sensor batch to the actual HTTP API."""

import argparse, os, sys, time, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import httpx
from mlops_platform.data.ingest import simulate
from mlops_platform.config import FEATURES

parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, default=100)
parser.add_argument("--seed", type=int, default=299)
parser.add_argument("--shifted", action="store_true")
parser.add_argument("--label", action="store_true")
args = parser.parse_args()
token = os.getenv("MLOPS_ADMIN_TOKEN", "")
if args.label and not token:
    parser.error("MLOPS_ADMIN_TOKEN is required for labels")
with httpx.Client(base_url="http://127.0.0.1:8200", timeout=30) as client:
    for _, row in simulate(args.rows, args.seed, args.shifted).iterrows():
        response = client.post("/api/predict", json=row[FEATURES].to_dict())
        response.raise_for_status()
        if args.label:
            result = client.post(
                "/api/labels",
                headers={"X-Admin-Token": token},
                json={
                    "prediction_id": response.json()["prediction_id"],
                    "failure": int(row.failure),
                },
            )
            result.raise_for_status()
print(
    json.dumps(
        {
            "readings_scored": args.rows,
            "labels_recorded": args.rows if args.label else 0,
            "seed": args.seed,
            "shifted": args.shifted,
        }
    )
)
