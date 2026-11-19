#!/usr/bin/env python3
"""Download Common Voice Scripted Speech 27.0 - Dutch from Mozilla Data Collective.

Each team member runs this in THEIR OWN account/session (the license forbids re-sharing).
Before running: create an account, accept the dataset terms on
  https://mozilladatacollective.com/datasets/cmu62gjbl00nso107n7ct8dyl
then create an API key in the dashboard and set it as an environment variable:
  export MDC_API_KEY=your-key     (Colab: os.environ["MDC_API_KEY"] = "your-key")

Usage:  python download_data.py --out /content/cv
Then:   python make_split.py --cv_dir /content/cv ...
"""
import argparse
import tarfile
from pathlib import Path

DATASET_ID = "cmu62gjbl00nso107n7ct8dyl"


def extract_if_needed(p: Path, out: Path):
    archives = [p] if p.is_file() else sorted(p.rglob("*.tar.gz"))
    for a in archives:
        if a.name.endswith(".tar.gz"):
            print("Extracting", a)
            with tarfile.open(a, "r:gz") as t:
                try:
                    t.extractall(out, filter="data")  # safe extraction (Python 3.12+, patched 3.10/3.11)
                except TypeError:
                    t.extractall(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cv")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from datacollective import download_dataset  # pip install datacollective
    p = Path(str(download_dataset(DATASET_ID)))
    print("Downloaded to:", p)
    extract_if_needed(p, out)
    hits = sorted(out.rglob("validated.tsv"))
    if not hits and p.is_dir():
        hits = sorted(p.rglob("validated.tsv"))
    if hits:
        print("Found validated.tsv in:", hits[0].parent)
        print("Use:  --cv_dir", hits[0].parent)
    else:
        print("validated.tsv not found yet. Look inside", out, "and", p)
    print("Tip: delete the .tar.gz after extracting to free about 3 GB.")


if __name__ == "__main__":
    main()
