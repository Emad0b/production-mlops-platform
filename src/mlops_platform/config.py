import os, json
from pathlib import Path

ROOT = Path(
    os.getenv(
        "MLOPS_PROJECT_ROOT",
        str(
            Path(__file__).resolve().parents[2]
            if (Path(__file__).resolve().parents[2] / "configs").exists()
            else Path.cwd()
        ),
    )
).resolve()
RUNTIME = Path(os.getenv("MLOPS_HOME", str(ROOT / "runtime"))).resolve()
RUNTIME.mkdir(parents=True, exist_ok=True)
ARTIFACTS = RUNTIME / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
DATA = RUNTIME / "datasets"
DATA.mkdir(exist_ok=True)
DATABASE_URL = os.getenv("DATABASE_URL")
TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "sqlite:///" + str(RUNTIME / "mlflow.db").replace("\\", "/")
)
MODEL_NAME = "mlops_platform-maintenance"
FEATURES = ["temperature", "vibration", "pressure", "rpm", "operating_hours"]
SCHEMA_VERSION = "sensor-v1"
ADMIN_TOKEN = os.getenv("MLOPS_ADMIN_TOKEN", "")
GATES = json.loads((ROOT / "configs/gates.json").read_text())
