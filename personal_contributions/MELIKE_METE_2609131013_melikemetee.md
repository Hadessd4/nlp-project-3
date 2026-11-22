## Personal Contributions

**Name:** Melike Mete\
**GitHub:** [`melikemetee`](https://github.com/melikemetee)\
**Student ID:** [2609131013]

### Summary of Work

- **Evaluation Setup:** Set up Google Colab with a T4 GPU, downloaded the Dutch Common Voice dataset through Mozilla Data Collective, and generated the shared speaker-disjoint splits using `make_split.py` with seed 42.
- **wav2vec2 Baseline Evaluation:** Ran the pretrained `jonatasgrosman/wav2vec2-large-xlsr-53-dutch` model with CTC decoding and scored the predictions using `asr_eval.py`.
  - **Dev (833 clips):** 8.48% WER, 2.45% CER, RTF 0.0042.
  - **Final test (791 clips):** 9.49% WER, 2.98% CER, RTF 0.0046.
- **Qwen3-ASR Baseline Evaluation:** Set up a separate Python environment for `Qwen/Qwen3-ASR-0.6B` and evaluated it on the same dev and final test splits.
  - **Dev (833 clips):** 10.48% WER, 3.75% CER, RTF 0.129.
  - **Final test (791 clips):** 13.63% WER, 5.38% CER, RTF 0.1342.
- **Dutch HuBERT + CTC Evaluation:** Identified the `coen22/Speech-Recognition-AWO-L` checkpoint and ran it with CTC decoding on all 833 dev clips. It scored 7.54% WER and 2.06% CER (RTF 0.0154). I verified the scores by recalculating them from the saved predictions and recorded the model revision.
- **Results and Pull Request Review:** Added the wav2vec2 and Qwen dev results to `results/baselines.csv` through PR #1. I also reviewed and approved PR #2 after checking that the Canary and Parakeet JSON metadata matched the corresponding CSV rows.
