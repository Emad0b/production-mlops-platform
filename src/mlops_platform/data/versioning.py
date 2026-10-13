"""DVC snapshots of validated datasets; data stays outside Git."""

import sys, subprocess
from .ingest import load_dataset
from ..config import ROOT, DATA


def version(digest):
    load_dataset(digest)
    if not (ROOT / ".dvc").exists():
        subprocess.run(
            [sys.executable, "-m", "dvc", "init", "--no-scm"], cwd=ROOT, check=True
        )
    pointers = ROOT / "versioned"
    pointers.mkdir(exist_ok=True)
    target = pointers / (digest + ".csv")
    command = [sys.executable, "-m", "dvc", "add"]
    if target.exists():
        command.append(str(target))
    else:
        command += [str(DATA / (digest + ".csv")), "--out", str(target)]
    subprocess.run(command, cwd=ROOT, check=True)
    return pointers / (digest + ".csv.dvc")
