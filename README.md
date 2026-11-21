# Project 3: Dutch ASR (Common Voice Scripted Speech 27.0 - Dutch)

Dataset: https://mozilladatacollective.com/datasets/cmu62gjbl00nso107n7ct8dyl (CC0, 3.17 GB)
The license forbids re-hosting/re-sharing the dataset and identifying speakers.
EVERY team member downloads it in their own account. Never send audio to each other.

# Project 3: Dutch ASR (Common Voice Scripted Speech 27.0 – Dutch)

Benchmarking six pretrained ASR models and fine-tuning `openai/whisper-small` on **Common Voice Scripted Speech 27.0 – Dutch**.

---

## 1. Dataset & Licensing

* **Dataset:** [Common Voice Scripted Speech 27.0 – Dutch](https://mozilladatacollective.com/datasets/cmu62gjbl00nso107n7ct8dyl) (CC0, ~3.17 GB).
* **License & Privacy:** The Mozilla Data Collective license forbids re-hosting/re-sharing the dataset and identifying speakers. **Every team member downloads it independently in their own account. Never send audio or raw transcripts to each other.**
* **Domain Note:** Approximately **73.5%** of the dataset's sentences originate from Europarl (parliamentary text).
* **Split Strategy:** We used our deterministic, speaker-disjoint split (`make_split.py` with `--train_hours 8 --dev_hours 1 --test_hours 1 --seed 42`, rather than `--official_split`). Full speaker counts, utterance counts, hours per split, and train/test sentence overlap are documented in [`data/split_stats.md`](data/split_stats.md) and [`data/split_stats.json`](data/split_stats.json).

---

## 2. Rules Built Into the Code

* **Speaker-Disjoint Splits:** Splits are constructed by speaker (`client_id`); `make_split.py` asserts that no speaker appears in more than one split.
* **Test-Set Guard:** `data/test.tsv` is blocked by default in both `run_baselines.py` and `asr_eval.py` unless `--final` is explicitly passed.
* **Unified Evaluation:** One normalization policy (`asr_eval.normalize`, documented in `results/normalization_policy.txt`) and one scorer (`asr_eval.py`) are used for every system.
* **Separated Results:** Pretrained baselines are logged to `results/baselines.csv`; our fine-tuned model is logged to `results/contribution.csv`.
* **Provenance Tracking:** Exact checkpoint name, commit hash (`revision`), and Real-Time Factor (`RTF`) are saved in a `.json` sidecar next to each prediction file.

---

## 3. Models & Results

| System | Group | Model Card & Checkpoint | Exact Revision (Commit) | Dev WER (%) | Dev CER (%) | Dev RTF | Test WER (%) | Test CER (%) | Test RTF |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`canary-1b-v2`** | Baseline | [`nvidia/canary-1b-v2`](https://huggingface.co/nvidia/canary-1b-v2) | `d455706339a6b32e1aa40f82c713a482a0c938e2` | 3.80 | 1.30 | 0.0248 | 4.74 | 1.74 | 0.0256 |
| **`parakeet-tdt-0.6b-v3`** | Baseline | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | `541d1f99c6b0c3cd0b11a95167540bb8edefd82b` | 4.23 | 1.19 | 0.0101 | 5.71 | 1.82 | 0.0106 |
| **`hubert-nl`** | Baseline | [`coen22/Speech-Recognition-AWO-L`](https://huggingface.co/coen22/Speech-Recognition-AWO-L) | `358f57fa790aae826229d0ac396fb54492eb315e` | 7.54 | 2.06 | 0.0154 | — | — | — |
| **`w2v2-nl`** | Baseline | [`jonatasgrosman/wav2vec2-large-xlsr-53-dutch`](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-dutch) | `46f221381d200f7bef268309b3f02023ccf11fcc` | 8.48 | 2.45 | 0.0042 | 9.49 | 2.98 | 0.0046 |
| **`qwen06`** | Baseline | [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` | 10.48 | 3.75 | 0.1290 | 13.63 | 5.38 | 0.1342 |
| **`whisper-small`** | Baseline | [`openai/whisper-small`](https://huggingface.co/openai/whisper-small) | `973afd24965f72e36ca33b3055d56a652f456b4d` | 12.82 | 4.09 | 0.7593* | 15.71 | 5.60 | 0.0480 |
| **`whisper-small-finetuned`** | **Contribution** | Local fine-tune of `openai/whisper-small` | `local checkpoint (our own fine-tuning)` | **10.59** | **3.46** | **0.0125** | **12.07** | **5.05** | **0.0143** |

*\*`whisper-small` dev baseline ran on CPU before CUDA setup; all other runs executed on GPU.*

### Model & Report Notes
* **Fine-Tuning Contribution:** `openai/whisper-small` was fine-tuned on our 8-hour Dutch `train.tsv` split (`max_steps=1000`, `batch_size=4`, `grad_accum=4`, `lr=1e-5`, best checkpoint selected on `dev` WER).
  * **Dev set (833 clips):** WER improved from **12.82% to 10.59%** (-2.23 pp / 17.4% relative reduction); CER improved from **4.09% to 3.46%**.
  * **Test set (791 clips):** WER improved from **15.71% to 12.07%** (-3.64 pp / 23.2% relative reduction); CER improved from **5.60% to 5.05%**.
* **Dutch SSL + CTC Baselines:** Both `jonatasgrosman/wav2vec2-large-xlsr-53-dutch` (`w2v2-nl`) and `coen22/Speech-Recognition-AWO-L` (`hubert-nl`) were evaluated as Dutch SSL + CTC baselines.

---

## 4. Run Order (Colab / Kaggle or Local GPU)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(For NeMo models `parakeet` / `canary`, install `pip install "nemo_toolkit[asr]"`. For `Qwen3-ASR`, install `pip install -U qwen-asr`.)*

2. **Download Common Voice 27.0 – Dutch:**
   Accept the dataset terms on Mozilla Data Collective, create an API key, and run:
   ```bash
   # Linux / Mac / Colab (or os.environ["MDC_API_KEY"] = "your-key")
   export MDC_API_KEY="your-key"
   # PowerShell
   $env:MDC_API_KEY="your-key"

   python download_data.py --out /content/cv
   ```

3. **Build the split (same seed = identical split for everyone):**
   ```bash
   python make_split.py --cv_dir /content/cv --train_hours 8 --dev_hours 1 --test_hours 1 --seed 42 --out_dir data
   ```
   * Compare `data/split_stats.md` across team members to confirm splits match.
   * Optional flags: `--drop_test_overlap` (removes test sentences appearing in train) or `--official_split` (uses Common Voice's official splits while still asserting zero speaker overlap).

4. **Evaluate baselines on `dev` first** *(add `--limit 10` for a quick smoke test)*:
   ```bash
   python run_baselines.py --model whisper --checkpoint openai/whisper-small --split data/dev.tsv --out preds/whisper-small_dev.tsv
   python asr_eval.py --ref data/dev.tsv --pred preds/whisper-small_dev.tsv --system whisper-small --group baseline
   ```

5. **Fine-tune `whisper-small` (`contribution`):**
   ```bash
   python finetune_whisper.py --train data/train.tsv --dev data/dev.tsv --out_dir outputs/whisper-small-nl --max_steps 1000 --batch_size 4 --grad_accum 4
   python run_baselines.py --model whisper --checkpoint outputs/whisper-small-nl/final --split data/dev.tsv --out preds/whisper-small-ft_dev.tsv
   python asr_eval.py --ref data/dev.tsv --pred preds/whisper-small-ft_dev.tsv --system whisper-small-finetuned --group contribution
   ```

6. **Final test evaluation (single pass, after freezing everything):**
   Rerun each system on `data/test.tsv` with `--final` passed to **both** `run_baselines.py` and `asr_eval.py`.

---

## 5. What to Commit to GitHub

* **Commit:**
  * All `.py` scripts (`download_data.py`, `make_split.py`, `run_baselines.py`, `finetune_whisper.py`, `asr_eval.py`)
  * `README.md`, `requirements.txt`, `LICENSE`, `.gitignore`
  * `results/baselines.csv`, `results/contribution.csv`, `results/normalization_policy.txt`
  * `preds/*.json` (exact checkpoint, revision, and RTF sidecars)
  * `data/split_stats.md` and `data/split_stats.json`
* **Do NOT commit:**
  * The dataset (`cv/`), `.mp3`/`.wav` audio files, or `.tar.gz` archives
  * `data/*.tsv`, `preds/*.tsv`, or `results/utt/`
  * Model checkpoints (`outputs/`), `__pycache__/`, or API keys

---

## License

Code in this repository is licensed under the MIT License (see [`LICENSE`](LICENSE)). Common Voice 27.0 and all pretrained models remain subject to their respective licenses.
````[cite: 4, 5]
