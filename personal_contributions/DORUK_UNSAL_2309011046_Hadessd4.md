## Personal Contributions

**Name:** Doruk Ünsal  
**GitHub:** [`Hadessd4`](https://github.com/Hadessd4)  
**Student ID:** [2309011046]

### Summary of Work
* **Data Preparation & Split Verification:** Generated and verified the deterministic, speaker-disjoint dataset splits (`train.tsv`, `dev.tsv`, `test.tsv` with `--train_hours 8 --dev_hours 1 --test_hours 1 --seed 42`) using `make_split.py`, ensuring zero speaker overlap across splits.
* **Whisper Baseline Evaluation:** Ran and scored the pretrained `openai/whisper-small` baseline on both the `dev` set (833 clips: **12.82% WER**, **4.09% CER**) and the final `test` set (791 clips: **15.71% WER**, **5.60% CER**).
* **Model Fine-Tuning (Main Contribution):** Configured GPU-accelerated training (CUDA/PyTorch) and fine-tuned `openai/whisper-small` on the 8-hour Dutch training split (`finetune_whisper.py`, 1,000 steps, effective batch size 16, best checkpoint selected on `dev` WER).
* **Fine-Tuned Model Evaluation:** Evaluated `whisper-small-finetuned` using the shared normalization and scoring pipeline (`asr_eval.py`) and logged results to `results/contribution.csv` and `preds/*.json`:
  * **Dev set:** Reduced WER from **12.82% → 10.59%** (-2.23 pp / 17.4% relative reduction) and CER from **4.09% → 3.46%** (RTF: 0.0125).
  * **Test set (`--final`):** Reduced WER from **15.71% → 12.07%** (-3.64 pp / 23.2% relative reduction) and CER from **5.60% → 5.05%** (RTF: 0.0143).
* **Repository Management & Compliance:** Maintained GitHub repository hygiene in accordance with the Mozilla Data Collective license—excluded raw audio, `.tsv` splits, and model checkpoints (`outputs/`, `__pycache__/`), consolidated evaluation metrics, and finalized `README.md` and `LICENSE`.