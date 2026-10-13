"""Create a DVC pointer for an immutable, checksum-validated dataset."""

import sys, subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlops_platform.config import ROOT, DATA
from mlops_platform.data.ingest import load_dataset


def version(digest):
    load_dataset(digest)
    if not (ROOT / ".dvc").exists():
        subprocess.run(
            [sys.executable, "-m", "dvc", "init", "--no-scm"], cwd=ROOT, check=True
        )
    pointers = ROOT / "versioned"
    pointers.mkdir(exist_ok=True)
    target = pointers / (digest + '.csv')
    command = [sys.executable, '-m', 'dvc', 'add']
    if target.exists():
        command.append(str(target))
    else:
        command += [str(DATA / (digest + '.csv')), '--out', str(target)]
    subprocess.run(command, cwd=ROOT, check=True)
    return pointers / (digest + ".csv.dvc")


if __name__ == "__main__":
    print(version(sys.argv[1]))
