#!/usr/bin/env python3
"""Fine-tune whisper-small on OUR train split (this is our contribution).

- Trains on data/train.tsv, picks the best checkpoint on data/dev.tsv. Never touches test.
- Audio is loaded on the fly (no huge feature cache on disk).
- Only the best model is kept (save_only_model) to save disk space.

Usage:
  python finetune_whisper.py --train data/train.tsv --dev data/dev.tsv \
      --out_dir outputs/whisper-small-nl --max_steps 1000

Low VRAM (8 GB or less): add --batch_size 4 --grad_accum 4
"""
import argparse
import csv
import os
from dataclasses import dataclass

import jiwer
import librosa
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from asr_eval import normalize

SR = 16000


class TsvDataset(Dataset):
    def __init__(self, tsv, limit=None):
        df = pd.read_csv(tsv, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
        if limit:
            df = df.sample(n=min(limit, len(df)), random_state=0)
        self.rows = df[["audio", "sentence"]].to_dict("records")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


@dataclass
class Collator:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, feats):
        audios = [librosa.load(f["audio"], sr=SR, mono=True)[0] for f in feats]
        batch = self.processor.feature_extractor(audios, sampling_rate=SR, return_tensors="pt")
        tok = self.processor.tokenizer([f["sentence"] for f in feats], padding=True, return_tensors="pt")
        labels = tok["input_ids"].masked_fill(tok["attention_mask"].ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--base", default="openai/whisper-small")
    ap.add_argument("--out_dir", default="outputs/whisper-small-nl")
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--eval_steps", type=int, default=250)
    ap.add_argument("--dev_limit", type=int, default=300, help="use a subset of dev for fast evaluation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    processor = WhisperProcessor.from_pretrained(args.base, language="dutch", task="transcribe")
    processor.tokenizer.set_prefix_tokens(language="dutch", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base)
    model.generation_config.language = "dutch"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False

    def compute_metrics(pred):
        ids = pred.predictions
        lab = pred.label_ids
        lab[lab == -100] = processor.tokenizer.pad_token_id
        hyp = [normalize(x) for x in processor.batch_decode(ids, skip_special_tokens=True)]
        ref = [normalize(x) for x in processor.batch_decode(lab, skip_special_tokens=True)]
        pairs = [(r, h) for r, h in zip(ref, hyp) if r]
        return {"wer": jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])}

    train_ds = TsvDataset(args.train)
    dev_ds = TsvDataset(args.dev, limit=args.dev_limit)

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=100,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=1,
        save_only_model=True,          # no optimizer state on disk
        load_best_model_at_end=True,
        metric_for_best_model="wer",   # chosen on DEV, never on test
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0 if os.name == "nt" else 2,
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=Collator(processor, model.config.decoder_start_token_id),
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )
    trainer.train()
    trainer.save_model(f"{args.out_dir}/final")
    processor.save_pretrained(f"{args.out_dir}/final")
    print("Best dev checkpoint saved to", f"{args.out_dir}/final")


if __name__ == "__main__":
    main()