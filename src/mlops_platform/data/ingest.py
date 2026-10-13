import hashlib, json, time
import numpy as np
import pandas as pd
from ..config import DATA
from .validate import validate


def simulate(rows=4000, seed=42, shift=False):
    if not 100 <= rows <= 100000:
        raise ValueError("Rows must be between 100 and 100,000")
    rng = np.random.default_rng(seed)
    temperature = np.clip(rng.normal(68 + (15 if shift else 0), 9, rows), 0, 160)
    vibration = np.clip(rng.gamma(2, 1.1, rows) + (2 if shift else 0), 0, 30)
    pressure = np.clip(rng.normal(6, 0.6, rows) - (1 if shift else 0), 0, 20)
    rpm = np.clip(rng.normal(1800, 140, rows), 100, 10000)
    hours = rng.uniform(0, 12000, rows)
    logit = (
        -4.6
        + 0.055 * (temperature - 65)
        + 0.72 * vibration
        + 0.00012 * hours
        + 0.7 * (pressure < 5.2)
        + 0.65 * ((temperature > 78) & (vibration > 3.5))
    )
    risk = 1 / (1 + np.exp(-logit))
    failure = (rng.uniform(0, 1, rows) < risk).astype(int)
    frame = pd.DataFrame(
        {
            "record_id": [f"{seed}-{i}" for i in range(rows)],
            "timestamp": 1704067200 + np.arange(rows) * 60.0,
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "rpm": rpm,
            "operating_hours": hours,
            "failure": failure,
        }
    )
    return validate(frame)


def save_dataset(frame, provenance):
    frame = validate(frame).sort_values("timestamp").reset_index(drop=True)
    raw = frame.to_csv(index=False, float_format="%.9f").encode()
    digest = hashlib.sha256(raw).hexdigest()
    path = DATA / (digest + ".csv")
    manifest = {
        "id": digest,
        "sha256": digest,
        "rows": len(frame),
        "created": time.time(),
        "schema": "sensor-v1",
        "provenance": provenance,
        "license": "Self-authored simulated data; no real equipment records",
        "failure_rate": float(frame.failure.mean()),
    }
    if not path.exists():
        path.write_bytes(raw)
        (DATA / (digest + ".json")).write_text(json.dumps(manifest, indent=2))
    return json.loads((DATA / (digest + ".json")).read_text())


def load_dataset(digest):
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid dataset identifier")
    raw = (DATA / (digest + ".csv")).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Dataset checksum mismatch")
    return validate(pd.read_csv(DATA / (digest + ".csv")))
