import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlops_platform.pipelines.flows import lifecycle_flow

if __name__ == "__main__":
    print(json.dumps(lifecycle_flow(sys.argv[1]), indent=2))
