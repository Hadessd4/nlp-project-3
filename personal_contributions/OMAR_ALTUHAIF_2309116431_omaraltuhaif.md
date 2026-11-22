## Personal Contributions

**Name:** Omar Altuhaif\
**GitHub:** [`omaraltuhaif`](https://github.com/omaraltuhaif)\
**Student ID:** [2309116431]

### Summary of Work

- **Environment & Data Setup (Google Colab, T4 GPU):** Set up a separate Colab environment for NVIDIA NeMo (`nemo_toolkit[asr]` plus `sox`, `libsndfile1` and `ffmpeg`). Downloaded Common Voice Scripted Speech 27.0 - Dutch (`cv-corpus-27.0`) in my own Mozilla Data Collective account and ran `make_split.py` with `--train_hours 8 --dev_hours 1 --test_hours 1 --seed 42`.
  - Checked that my split matched the team's: `data/split_stats.md` was identical to the committed version, and the `dev` and `test` clip lists were **IDENTICAL** to the shared split (same speakers and same clips).
- **Smoke Testing:** Ran 10-clip smoke tests (`--limit 10`) for both NeMo models on `dev`. I checked that the transcriptions were real Dutch text before starting the full runs.
- **NeMo Baseline Evaluation:** Ran `run_baselines.py` and scored the output with the shared `asr_eval.py` pipeline. The same normalization policy and corpus-level WER/CER were used as for every other system, and the batch size was the default 8 so RTF can be compared with the other baselines.
  - **`nvidia/canary-1b-v2`** (revision `d455706339a6b32e1aa40f82c713a482a0c938e2`), with `source_lang="nl"` and `target_lang="nl"` so it transcribes instead of translating:
    - **Dev (833 clips):** 3.80% WER, 1.30% CER, RTF 0.0248.
    - **Final test (791 clips, `--final`):** 4.74% WER, 1.74% CER, RTF 0.0256.
  - **`nvidia/parakeet-tdt-0.6b-v3`** (revision `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`), which detects the language automatically:
    - **Dev (833 clips):** 4.23% WER, 1.19% CER, RTF 0.0101.
    - **Final test (791 clips, `--final`):** 5.71% WER, 1.82% CER, RTF 0.0106.
- **Repository Integration:** Packaged only the outputs allowed under the dataset license (`nemo_results.zip`: the prediction JSON files and the new CSV rows). No audio, prediction `.tsv` files, per-utterance results or split `.tsv` files were included. Added the four prediction JSON files (`preds/canary-1b-v2_{dev,test}.json` and `preds/parakeet-tdt-0.6b-v3_{dev,test}.json`, which record the checkpoint, exact revision, RTF and clip count). Added the first `test` split metrics to `results/baselines.csv` together with the `dev` metrics, and submitted everything through Pull Request #2.
