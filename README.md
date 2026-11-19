# Project 3: Dutch ASR (Common Voice Scripted Speech 27.0 - Dutch)

Dataset: https://mozilladatacollective.com/datasets/cmu62gjbl00nso107n7ct8dyl (CC0, 3.17 GB)
The license forbids re-hosting/re-sharing the dataset and identifying speakers.
EVERY team member downloads it in their own account. Never send audio to each other.

## Run order (Colab / Kaggle recommended, so nothing big lands on your PC)
0. `pip install -r requirements.txt`
1. Accept the dataset terms on the page, create an API key, then:
   `export MDC_API_KEY=your-key` (Colab: `os.environ["MDC_API_KEY"] = "your-key"`)
   `python download_data.py --out /content/cv`
2. Build the split (same seed = same split for everyone):
   `python make_split.py --cv_dir /content/cv --train_hours 15 --dev_hours 1.5 --test_hours 1.5 --seed 42 --out_dir data`
   - Person A compares `data/split_stats.md` with the others to confirm the splits match.
   - Add `--drop_test_overlap` if many test sentences also appear in train.
   - Alternative: `--official_split` uses Common Voice's documented splits (script still checks speaker overlap; say in the README which you used).
3. Test each system on dev first (add `--limit 10` for a smoke test):
   `python run_baselines.py --model whisper --checkpoint openai/whisper-small --split data/dev.tsv --out preds/whisper-small_dev.tsv`
   `python asr_eval.py --ref data/dev.tsv --pred preds/whisper-small_dev.tsv --system whisper-small --group baseline`
4. Person D fine-tunes: `python finetune_whisper.py --train data/train.tsv --dev data/dev.tsv`
   then runs it through run_baselines.py and scores it with `--group contribution`.
5. Final (Saturday, after freezing everything): rerun each system on `data/test.tsv`
   with `--final` in BOTH run_baselines.py and asr_eval.py.

## Rules built into the code
- Split is by speaker (client_id); the script asserts no speaker is in two splits.
- Test split is blocked unless `--final` is passed.
- One normalization policy (`asr_eval.normalize`) and one scorer for every system.
- Baselines go to `results/baselines.csv`, our trained models to `results/contribution.csv`.
- Exact checkpoint + commit hash + RTF are saved next to each prediction file.

## Notes for the report
- About 73.5% of the dataset's sentences come from Europarl (parliamentary text).
- Report speakers, utterances, hours per split and the test/train sentence overlap (in split_stats.md).
- Link every model card, write the exact revision, and state when a model does not support Dutch.

## What to commit to GitHub
- Commit: the .py files, README.md, requirements.txt, LICENSE, `results/baselines.csv`, `results/contribution.csv`,
  `preds/*.json` (exact checkpoint, revision, RTF) and the split statistics (`split_stats.md` / `.json`, copied to `docs/`).
- Do NOT commit: the dataset, mp3/wav files, `data/*.tsv`, model checkpoints, API keys (the dataset license forbids re-sharing).
