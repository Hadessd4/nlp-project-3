#!/usr/bin/env python3
"""Run one pretrained (or our own fine-tuned) model on a split and save predictions.

Writes  <out>.tsv  (path, hypothesis)  and  <out>.json  (exact checkpoint, revision, RTF).
Then score with asr_eval.py.

Examples (use dev.tsv while testing; test.tsv needs --final and is for the very end):
  python run_baselines.py --model whisper  --checkpoint openai/whisper-small --split data/dev.tsv --out preds/whisper-small_dev.tsv
  python run_baselines.py --model wav2vec2 --checkpoint jonatasgrosman/wav2vec2-large-xlsr-53-dutch --split data/dev.tsv --out preds/w2v2-nl_dev.tsv
  python run_baselines.py --model parakeet --checkpoint nvidia/parakeet-tdt-0.6b-v3 --split data/dev.tsv --out preds/parakeet_dev.tsv
  python run_baselines.py --model canary   --checkpoint nvidia/canary-1b-v2 --split data/dev.tsv --out preds/canary_dev.tsv
  python run_baselines.py --model qwen     --checkpoint Qwen/Qwen3-ASR-0.6B --split data/dev.tsv --out preds/qwen06_dev.tsv

Quick smoke test on 10 clips: add --limit 10
Our fine-tuned model: --model whisper --checkpoint outputs/whisper-small-nl/final
"""
import argparse
import csv
import json
import tempfile
import time
from pathlib import Path

import pandas as pd

from asr_eval import guard_test

SR = 16000


def load_audio(path):
    import librosa
    audio, _ = librosa.load(path, sr=SR, mono=True)
    return audio


def to_wavs(paths, tmp):
    """NeMo and Qwen are happiest with 16 kHz mono wav files."""
    import soundfile as sf
    out = []
    for i, p in enumerate(paths):
        w = Path(tmp) / f"{i}.wav"
        sf.write(str(w), load_audio(p), SR)
        out.append(str(w))
    return out


def text_of(x):
    return x if isinstance(x, str) else getattr(x, "text", str(x))


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def run_hf_pipeline(args, audio_paths):
    import torch
    from transformers import pipeline
    dev = 0 if torch.cuda.is_available() else -1
    kw = dict(model=args.checkpoint, device=dev)
    if args.revision:
        kw["revision"] = args.revision
    if dev == 0:
        kw["torch_dtype"] = torch.float16
    asr = pipeline("automatic-speech-recognition", **kw)
    gen = {"language": "dutch", "task": "transcribe"} if args.model == "whisper" else {}
    audios = [{"raw": load_audio(p), "sampling_rate": SR} for p in audio_paths]
    t0 = time.time()
    outs = asr(audios, batch_size=args.batch_size, generate_kwargs=gen) if gen else asr(audios, batch_size=args.batch_size)
    return [o["text"] for o in outs], time.time() - t0


def run_nemo(args, audio_paths):
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(args.checkpoint)
    model.eval()
    hyps = []
    t_total = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        wavs = to_wavs(audio_paths, tmp)
        for batch in chunks(wavs, args.batch_size):
            t0 = time.time()
            if args.model == "canary":
                # same source and target language = transcription (not translation)
                out = model.transcribe(batch, batch_size=args.batch_size, source_lang="nl", target_lang="nl")
            else:  # parakeet v3 detects the language automatically
                out = model.transcribe(batch, batch_size=args.batch_size)
            t_total += time.time() - t0
            hyps += [text_of(o) for o in out]
    return hyps, t_total


def run_qwen(args, audio_paths):
    import torch
    from qwen_asr import Qwen3ASRModel
    model = Qwen3ASRModel.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, device_map="cuda:0", max_new_tokens=256
    )
    hyps = []
    t_total = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        wavs = to_wavs(audio_paths, tmp)
        for batch in chunks(wavs, args.batch_size):
            t0 = time.time()
            res = model.transcribe(audio=batch, language=["Dutch"] * len(batch))
            t_total += time.time() - t0
            hyps += [r.text for r in res]
    return hyps, t_total


def resolve_revision(args):
    """Record the exact commit of the checkpoint (required by the professor)."""
    if Path(args.checkpoint).exists():
        return "local checkpoint (our own fine-tuning)"
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(args.checkpoint, revision=args.revision).sha
    except Exception as e:
        return f"unknown ({e.__class__.__name__}); write it from the model card by hand"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["whisper", "wav2vec2", "parakeet", "canary", "qwen"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--split", required=True, help="dev.tsv or test.tsv from make_split.py")
    ap.add_argument("--out", required=True, help="predictions TSV to write")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="only first N clips (smoke test)")
    ap.add_argument("--final", action="store_true", help="required for test.tsv")
    args = ap.parse_args()

    guard_test(args.split, args.final)

    df = pd.read_csv(args.split, sep="\t", quoting=csv.QUOTE_NONE, dtype={"duration_s": float}, keep_default_na=False)
    if args.limit:
        df = df.head(args.limit)
    paths = df["audio"].tolist()
    audio_seconds = float(df["duration_s"].sum())

    if args.model in ("whisper", "wav2vec2"):
        hyps, secs = run_hf_pipeline(args, paths)
    elif args.model in ("parakeet", "canary"):
        hyps, secs = run_nemo(args, paths)
    else:
        hyps, secs = run_qwen(args, paths)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": df["path"], "hypothesis": [h.replace("\t", " ").replace("\n", " ") for h in hyps]}).to_csv(
        out, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\"
    )
    side = {
        "checkpoint": args.checkpoint,
        "revision": resolve_revision(args),
        "rtf": round(secs / max(audio_seconds, 1e-9), 4),
        "n_clips": len(df),
        "split_file": args.split,
    }
    out.with_suffix(".json").write_text(json.dumps(side, indent=2))
    print(json.dumps(side, indent=2))


if __name__ == "__main__":
    main()
