#!/usr/bin/env python3
"""Shared evaluation for ALL systems (baselines and our own models).

One normalization policy + one scorer, so every system is compared fairly.

Predictions file (TSV): columns  path <TAB> hypothesis
Reference file   (TSV): the split file made by make_split.py (dev.tsv / test.tsv)

Example (dev, while tuning):
  python asr_eval.py --ref data/dev.tsv --pred preds/whisper-small_dev.tsv \
      --system whisper-small --group baseline

Example (test, ONLY once at the end, after everything is frozen):
  python asr_eval.py --ref data/test.tsv --pred preds/whisper-small_test.tsv \
      --system whisper-small --group baseline --final
"""
import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

import jiwer
import pandas as pd

NORMALIZATION_POLICY = (
    "NFKC, lowercase, curly->straight apostrophes, punctuation removed "
    "(apostrophes inside words kept), whitespace collapsed, numbers left as-is "
    "(use --spell_numbers to convert digits to Dutch words)"
)


def normalize(text: str, spell_numbers: bool = False) -> str:
    """The ONE normalization policy used for references and hypotheses."""
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = t.replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    t = re.sub(r"[^\w\s']", " ", t)            # drop punctuation, keep apostrophes
    t = re.sub(r"(?<!\w)'|'(?!\w)", " ", t)    # drop apostrophes not inside a word
    t = t.replace("_", " ")
    if spell_numbers:
        try:
            from num2words import num2words
            t = re.sub(r"\d+", lambda m: " " + num2words(int(m.group()), lang="nl") + " ", t)
        except ImportError:
            sys.exit("pip install num2words to use --spell_numbers")
    return re.sub(r"\s+", " ", t).strip()


def guard_test(path: str, final: bool):
    """Stop people from touching the test set before the end."""
    if "test" in Path(path).name.lower() and not final:
        sys.exit(
            "Refusing to use the TEST split without --final.\n"
            "Use dev.tsv while tuning. Test is used once, at the end, after everything is frozen."
        )


def score(refs, hyps):
    """Corpus-level WER and CER (micro-averaged)."""
    return {"wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="split TSV from make_split.py")
    ap.add_argument("--pred", required=True, help="predictions TSV: path, hypothesis")
    ap.add_argument("--system", required=True, help="short name, e.g. whisper-small")
    ap.add_argument("--group", required=True, choices=["baseline", "contribution"])
    ap.add_argument("--checkpoint", default=None, help="exact checkpoint name (overrides sidecar)")
    ap.add_argument("--revision", default=None, help="exact revision/commit (overrides sidecar)")
    ap.add_argument("--spell_numbers", action="store_true")
    ap.add_argument("--final", action="store_true", help="required to score the test split")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    guard_test(args.ref, args.final)

    ref = pd.read_csv(args.ref, sep="\t", quoting=csv.QUOTE_NONE, dtype=str)
    pred = pd.read_csv(args.pred, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    df = ref.merge(pred[["path", "hypothesis"]], on="path", how="left")
    missing = df["hypothesis"].isna().sum()
    if missing:
        sys.exit(f"{missing} reference clips have no prediction. Fix before scoring.")

    df["ref_n"] = df["sentence"].map(lambda s: normalize(s, args.spell_numbers))
    df["hyp_n"] = df["hypothesis"].map(lambda s: normalize(s, args.spell_numbers))
    df = df[df["ref_n"] != ""].reset_index(drop=True)

    res = score(df["ref_n"].tolist(), df["hyp_n"].tolist())
    df["utt_wer"] = [jiwer.wer(r, h) for r, h in zip(df["ref_n"], df["hyp_n"])]

    # sidecar written by run_baselines.py (exact checkpoint, revision, RTF)
    side_path = Path(args.pred).with_suffix(".json")
    side = json.loads(side_path.read_text()) if side_path.exists() else {}
    split = Path(args.ref).stem

    out = Path(args.out_dir)
    (out / "utt").mkdir(parents=True, exist_ok=True)
    df[["path", "client_id", "sentence", "hypothesis", "utt_wer"]].to_csv(
        out / "utt" / f"{args.system}_{split}.tsv", sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\"
    )

    row = {
        "system": args.system,
        "checkpoint": args.checkpoint or side.get("checkpoint", ""),
        "revision": args.revision or side.get("revision", ""),
        "split": split,
        "n_utts": len(df),
        "WER_%": round(100 * res["wer"], 2),
        "CER_%": round(100 * res["cer"], 2),
        "RTF": side.get("rtf", ""),
    }
    table = out / ("baselines.csv" if args.group == "baseline" else "contribution.csv")
    old = pd.read_csv(table, dtype=str) if table.exists() else pd.DataFrame()
    new = pd.concat([old, pd.DataFrame([row]).astype(str)], ignore_index=True)
    new = new.drop_duplicates(subset=["system", "split"], keep="last")
    new.to_csv(table, index=False)
    (out / "normalization_policy.txt").write_text(NORMALIZATION_POLICY + "\n")

    print(json.dumps(row, indent=2))
    print(f"-> appended to {table}")


if __name__ == "__main__":
    main()
