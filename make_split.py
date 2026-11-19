#!/usr/bin/env python3
"""Build a small, speaker-disjoint train/dev/test split from Common Voice nl (v27).

Rule (from the professor): split by SPEAKER, not by clip. All clips of one speaker
go to the same split. Common Voice has no sessions, so client_id = speaker.

Two modes:
  (default)         our own split from validated.tsv: whole speakers -> train/dev/test, fixed seed
  --official_split  use the documented Common Voice train.tsv/dev.tsv/test.tsv, then take a
                    small subset of whole speakers from each (script checks speaker overlap)

Usage:
  python make_split.py --cv_dir /content/cv --train_hours 15 --dev_hours 1.5 --test_hours 1.5 --seed 42 --out_dir data
  python make_split.py --cv_dir /content/cv --official_split --out_dir data_official

--cv_dir can be the extraction folder itself; the script searches it for validated.tsv.
Outputs in out_dir: train.tsv, dev.tsv, test.tsv, split_stats.json, split_stats.md
Each TSV has: client_id, path, audio, sentence, duration_s
"""
import argparse
import csv
import json
import random
import shutil
from itertools import combinations
from pathlib import Path

import pandas as pd

from asr_eval import normalize

EST_SEC = 4.4  # average clip length in v27 (4.374 s); only used to pick speakers


def locate_cv_dir(path: Path) -> Path:
    """Find the folder that holds validated.tsv and clips/ (archives often add a subfolder)."""
    if (path / "validated.tsv").exists():
        return path
    hits = sorted(path.rglob("validated.tsv"))
    if not hits:
        raise SystemExit(f"No validated.tsv found under {path}. Did you extract the .tar.gz?")
    return hits[0].parent


def load_cv(cv_dir: Path, tsv_name: str) -> pd.DataFrame:
    p = cv_dir / tsv_name
    if not p.exists():
        raise SystemExit(f"{p} not found. Files here: {sorted(x.name for x in cv_dir.glob('*.tsv'))}")
    df = pd.read_csv(
        p, sep="\t", quoting=csv.QUOTE_NONE, dtype=str,
        usecols=["client_id", "path", "sentence"], on_bad_lines="skip",
    )
    df = df.dropna(subset=["client_id", "path", "sentence"])
    df["sentence"] = df["sentence"].str.strip()
    return df[df["sentence"] != ""].reset_index(drop=True)


def load_durations(cv_dir: Path) -> dict:
    """clip_durations.tsv exists in some Common Voice releases (milliseconds)."""
    p = cv_dir / "clip_durations.tsv"
    if not p.exists():
        return {}
    d = pd.read_csv(p, sep="\t", quoting=csv.QUOTE_NONE)
    dur_col = [c for c in d.columns if "dur" in c.lower()][0]
    vals = d[dur_col].astype(float)
    if vals.median() > 60:  # milliseconds
        vals = vals / 1000.0
    return dict(zip(d[d.columns[0]], vals))


def real_duration(path: Path):
    try:
        import soundfile as sf
        return float(sf.info(str(path)).duration)
    except Exception:
        pass
    try:
        import librosa
        return float(librosa.get_duration(path=str(path)))
    except Exception:
        return None


def cap_groups(df: pd.DataFrame, seed: int, cap: int) -> dict:
    """speaker -> up to `cap` clips (deterministic shuffle per speaker)."""
    groups = {}
    for spk, g in df.groupby("client_id"):
        rows = g.sort_values("path").to_dict("records")
        random.Random(f"{seed}-{spk}").shuffle(rows)
        groups[spk] = rows[:cap]
    return groups


def stats_for(df: pd.DataFrame) -> dict:
    return {
        "speakers": int(df["client_id"].nunique()),
        "utterances": int(len(df)),
        "hours": round(float(df["duration_s"].sum()) / 3600, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv_dir", required=True, help="extracted Common Voice nl folder")
    ap.add_argument("--tsv", default="validated.tsv")
    ap.add_argument("--official_split", action="store_true",
                    help="use Common Voice's own train/dev/test TSVs instead of our own split")
    ap.add_argument("--train_hours", type=float, default=15)
    ap.add_argument("--dev_hours", type=float, default=1.5)
    ap.add_argument("--test_hours", type=float, default=1.5)
    ap.add_argument("--max_clips_per_speaker", type=int, default=40,
                    help="cap per speaker so a few heavy speakers do not dominate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--copy_audio_to", default=None,
                    help="copy ONLY the selected mp3s here (for your own disk; do not share them)")
    ap.add_argument("--drop_test_overlap", action="store_true",
                    help="remove test sentences that also occur word-for-word in train")
    args = ap.parse_args()

    cv = locate_cv_dir(Path(args.cv_dir))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    known = load_durations(cv)
    targets = {"train": args.train_hours * 3600, "dev": args.dev_hours * 3600,
               "test": args.test_hours * 3600}
    print(f"Using {cv}  ({len(known)} durations known from clip_durations.tsv)")

    def est(rows):
        return sum(known.get(r["path"], EST_SEC) for r in rows)

    frames = {}
    if args.official_split:
        for s in ["train", "dev", "test"]:
            groups = cap_groups(load_cv(cv, f"{s}.tsv"), args.seed, args.max_clips_per_speaker)
            spks = sorted(groups)
            random.Random(f"{args.seed}-{s}").shuffle(spks)
            rows, acc = [], 0.0
            for spk in spks:
                if acc >= targets[s]:
                    break
                rows += [dict(r, split=s) for r in groups[spk]]
                acc += est(groups[spk])
            frames[s] = pd.DataFrame(rows)
    else:
        df = load_cv(cv, args.tsv)
        print(f"{len(df)} validated clips, {df['client_id'].nunique()} speakers")
        groups = cap_groups(df, args.seed, args.max_clips_per_speaker)
        speakers = sorted(groups)
        random.Random(args.seed).shuffle(speakers)
        acc = {k: 0.0 for k in targets}
        split_of = {}
        for spk in speakers:
            for s in ["test", "dev", "train"]:
                if acc[s] < targets[s]:
                    split_of[spk] = s
                    acc[s] += est(groups[spk])
                    break
            else:
                break  # all splits are full
        for s in targets:
            rows = [dict(r, split=s) for spk, sp in split_of.items() if sp == s for r in groups[spk]]
            frames[s] = pd.DataFrame(rows)

    # real durations for the selected clips
    failed = 0
    for s, f in frames.items():
        durs = []
        for p in f["path"]:
            d = known.get(p)
            if d is None:
                d = real_duration(cv / "clips" / p)
            if d is None:
                d, failed = EST_SEC, failed + 1
            durs.append(d)
        f["duration_s"] = durs
    if failed:
        print(f"WARNING: could not read {failed} clips; used {EST_SEC}s estimate. Check that clips/ exists.")

    # optional: remove test sentences seen in train
    train_sents = {normalize(x) for x in frames["train"]["sentence"]}
    n_test_before = len(frames["test"])
    overlap_mask = frames["test"]["sentence"].map(lambda x: normalize(x) in train_sents)
    overlap_count = int(overlap_mask.sum())
    if args.drop_test_overlap:
        frames["test"] = frames["test"][~overlap_mask].reset_index(drop=True)

    # safety check: NO speaker in more than one split
    sets = {s: set(f["client_id"]) for s, f in frames.items()}
    for a, b in combinations(sets, 2):
        shared = sets[a] & sets[b]
        assert not shared, (f"DATA LEAK: {len(shared)} speakers in both {a} and {b}. "
                            f"If you used --official_split, those splits are not speaker-disjoint: "
                            f"use our own split instead.")
    print("OK: no speaker appears in more than one split")

    # optional copy of only the selected audio
    for s, f in frames.items():
        if args.copy_audio_to:
            dst = Path(args.copy_audio_to)
            dst.mkdir(parents=True, exist_ok=True)
            for p in f["path"]:
                if not (dst / p).exists():
                    shutil.copy(cv / "clips" / p, dst / p)
            f["audio"] = [str(dst / p) for p in f["path"]]
        else:
            f["audio"] = [str(cv / "clips" / p) for p in f["path"]]
        f[["client_id", "path", "audio", "sentence", "duration_s"]].to_csv(
            out / f"{s}.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\"
        )

    # statistics the professor asked for
    stats = {s: stats_for(f) for s, f in frames.items()}
    stats["test_sentences_in_train_word_for_word"] = {
        "count": overlap_count,
        "of_test_utterances": n_test_before,
        "percent": round(100 * overlap_count / max(n_test_before, 1), 2),
        "dropped_from_test": bool(args.drop_test_overlap),
    }
    stats["settings"] = {k: v for k, v in vars(args).items()}
    stats["split_source"] = "Common Voice official split (subset)" if args.official_split else "own speaker-grouped split of validated.tsv"
    (out / "split_stats.json").write_text(json.dumps(stats, indent=2))

    md = ["| Split | Speakers | Utterances | Hours |", "|---|---|---|---|"]
    for s in ["train", "dev", "test"]:
        md.append(f"| {s} | {stats[s]['speakers']} | {stats[s]['utterances']} | {stats[s]['hours']} |")
    o = stats["test_sentences_in_train_word_for_word"]
    md.append("")
    md.append(f"Split source: {stats['split_source']}.")
    md.append(f"Test sentences that appear word for word in train: {o['count']} of {o['of_test_utterances']} "
              f"({o['percent']}%)" + (" - removed from test." if o["dropped_from_test"] else "."))
    md.append(f"Seed: {args.seed}. Speaker overlap between splits: none (checked by script).")
    (out / "split_stats.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
