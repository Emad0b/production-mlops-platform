"""Version a checksum-validated dataset using DVC."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlops_platform.data.versioning import version

if __name__ == "__main__":
    print(version(sys.argv[1]))
