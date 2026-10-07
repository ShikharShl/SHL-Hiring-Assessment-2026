# SHL Hiring Assessment 2026 -- Grammar Scoring Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle Public Score](https://img.shields.io/badge/Kaggle%20Public%20Score-0.6166-brightgreen.svg)](#kaggle-leaderboard-result)

An explainable multimodal speech-to-score pipeline developed for the **SHL Hiring Assessment 2026** Kaggle challenge. The engine predicts continuous spoken grammar proficiency scores (`0.0` to `5.0`) by fusing physical acoustic/prosodic properties with automatic speech-to-text (ASR) linguistic features.

---

## Performance Summary

| Metric | Score | Validation Context |
|:---|:---:|:---|
| **Kaggle Public Score** | **`0.6166`** | Official Kaggle public leaderboard evaluation |
| **5-Fold CV Pearson Correlation** | **`0.7944 +/- 0.0405`** | Out-of-sample cross-validation generalization |
| **5-Fold CV RMSE** | **`0.7454 +/- 0.0345`** | Out-of-sample cross-validation error |
| **In-Sample Training RMSE** | **`0.2765`** | In-sample training fit on all 769 instances |
| **Total Multimodal Features** | **`63`** | 45 physical acoustic/prosodic + 18 linguistic/ASR |
| **Acoustic Baseline CV RMSE** | `0.7765 +/- 0.0380` | Reference Phase 1 baseline |
| **Acoustic Baseline CV Pearson** | `0.7727 +/- 0.0467` | Reference Phase 1 baseline |

---

## Dataset Overview

The challenge dataset consists of real-world candidate monologue recordings:
- **Audio Format:** 16,000 Hz (16 kHz), 1-Channel (Mono), 16-bit PCM WAV.
- **Training Set:** **769 samples** with continuous grammar scores from `0.0` to `5.0` (`train.csv`).
- **Test Set:** **216 samples** evaluated on Kaggle (`test.csv`).
- **Target Distribution:**
  - Range: `0.0` to `5.0` (Mean: `3.3134`, Median: `3.0000`, Std: `1.2390`).
  - Score counts: `0.0` (37), `1.0` (1), `1.5` (3), `2.0` (102), `2.5` (79), `3.0` (174), `3.5` (64), `4.0` (124), `4.5` (52), `5.0` (133).
  - *Quality Finding:* Audio files with score `0.0` contain continuous static noise/corruption with zero silence and extreme zero-crossing rates (~0.46), which our acoustic feature pipeline identifies reliably.

---

## Architecture & Approach

```
                      Raw Spoken Audio (.wav @ 16 kHz Mono)
                                      |
             +------------------------+------------------------+
             |                                                 |
             v                                                 v
  [Acoustic Processing]                             [Automatic Speech Recog]
  - 25ms Window, 10ms Hop                           - Whisper Tiny (CPU 4-core)
  - RMS Energy & Dynamics                           - English-constrained decoding
  - Zero-Crossing Rate (ZCR)                        - Caching to JSON transcripts
  - Pause Frequency & Silence %                                |
  - Spectral Centroid, Rolloff, Contrast                       v
  - 13 MFCC Coefficients (Mean & Std)               [Linguistic Feature Eng]
             |                                      - Vocabulary Diversity (Root TTR)
             | (45 Features)                        - Unique Word & Token Counts
             |                                      - Sentence Complexity & Length
             |                                      - Filler Word Rates & Repetition
             |                                      - Lexical Density & Speech Rate
             |                                                 | (18 Features)
             +------------------------+------------------------+
                                      |
                                      v
                        [Early Feature Fusion Layer]
                        - 63 Normalized Features
                        - Zero Missing Values
                                      |
                                      v
                     [RandomForest Regressor Ensemble]
                     - 100 Estimators, random_state=42
                     - 5-Fold Cross-Validation (KFold, seed=42)
                                      |
                                      v
                     [Continuous Grammar Score (0.0 - 5.0)]
```

### 1. Acoustic & Prosodic Features (45 Features)
- **Temporal & Prosody (11):** `duration`, `rms_mean`, `rms_std`, `energy_variability`, `silence_ratio`, `active_speech_ratio`, `speech_pause_ratio`, `pause_freq_per_min`, `mean_pause_duration`.
- **Spectral Descriptors (8):** `zcr_mean`, `zcr_std`, `spec_cent_mean`, `spec_cent_std`, `spec_bw_mean`, `spec_bw_std`, `spec_rf_mean`, `spec_rf_std`, `spec_ct_mean`, `spec_ct_std`.
- **MFCC Statistics (26):** Mean and Standard Deviation of the first 13 Mel-Frequency Cepstral Coefficients (`mfcc_1_mean` to `mfcc_13_std`).

### 2. Whisper ASR & Linguistic Features (18 Features)
- **ASR Engine:** OpenAI Whisper `tiny` executed with 4 CPU threads and incremental caching.
- **Lexical Diversity & Richness (5):** `vocabulary_diversity` (Root TTR = $V / \sqrt{N}$), `type_token_ratio`, `unique_word_count`, `word_count`, `character_count`.
- **Syntactic Complexity (5):** `sentence_count`, `avg_words_per_sentence`, `avg_word_length`, `long_word_ratio`, `short_word_ratio`.
- **Fluency & Delivery Rates (6):** `words_per_second`, `sentence_rate`, `filler_word_count`, `filler_word_rate`, `repeated_word_ratio`.
- **Content & Discourse (2):** `lexical_density`, `question_count`, `exclamation_count`.

---

## 5-Fold Cross-Validation Benchmark

All models were evaluated strictly on the **769 training samples** using identical 5-fold cross-validation splits (`random_state=42`):

| Model Architecture | Input Features | CV RMSE (Mean +/- Std) | CV Pearson R (Mean +/- Std) | vs Baseline |
|:---|:---|:---:|:---:|:---:|
| **Acoustic HistGBDT** *(Phase 1 Baseline)* | Acoustic Only (45) | `0.7765 +/- 0.0380` | `0.7727 +/- 0.0467` | *Baseline* |
| **Combined Ridge** | Acoustic + Linguistic (63) | `0.7552 +/- 0.0449` | `0.7875 +/- 0.0492` | +0.0148 Pearson |
| **Combined HistGradientBoosting** | Acoustic + Linguistic (63) | `0.7462 +/- 0.0309` | `0.7935 +/- 0.0383` | +0.0208 Pearson |
| **Combined RandomForest** *(Selected Final)* | Acoustic + Linguistic (63) | **`0.7454 +/- 0.0345`** | **`0.7944 +/- 0.0405`** | **+0.0217 Pearson** |

### Top Predictive Features (Combined RandomForest):
1. **`vocabulary_diversity`** (12.36% importance) -- Root TTR; measures lexical richness.
2. **`mfcc_2_std`** (7.55% importance) -- Vocal tract timbre dynamics.
3. **`unique_word_count`** (6.06% importance) -- Total distinct vocabulary size.
4. **`mfcc_5_std`** (4.97% importance) -- Higher-order resonance stability.
5. **`energy_variability`** (4.46% importance) -- Dynamic volume modulation and stress cadence.

---

## Repository Structure

```
SHL-Hiring-Assessment/
├── .gitignore                              # Excludes raw audio, zips, models, caches
├── README.md                               # Project documentation & benchmark overview
├── requirements.txt                        # Minimal pinned dependencies
├── SHL_Hiring_Assessment_2026_Final.ipynb  # Primary deliverable notebook
├── final_submission.csv                    # Official Kaggle submission (Public Score: 0.6166)
├── training_metrics.json                   # In-sample training metrics (Training RMSE: 0.2765)
├── src/                                    # Modular source code
│   ├── __init__.py
│   ├── extract_acoustic_features.py       # 45 acoustic/prosodic feature extractor
│   ├── transcribe_whisper.py              # Whisper Tiny ASR transcription
│   ├── extract_linguistic_features.py     # 18 linguistic metrics computer
│   ├── fuse_features.py                   # Feature merging & validation
│   ├── train_and_evaluate.py              # 5-fold CV evaluation & training logic
│   └── run_pipeline.py                    # End-to-end execution script
└── figures/                                # Benchmark plots
    ├── phase1_acoustic_baseline_plots.png
    └── phase2_combined_evaluation_plots.png
```

---

## How to Reproduce the Pipeline

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/<your-username>/SHL-Hiring-Assessment.git
cd SHL-Hiring-Assessment
pip install -r requirements.txt
```

### 2. Dataset Setup
Place the competition dataset inside the project folder:
```
Dataset_Final/
├── train/          # 769 audio files
├── test/           # 216 audio files
├── train.csv
└── test.csv
```

### 3. Run the End-to-End Pipeline
Execute the complete pipeline to extract features, evaluate cross-validation, train the final model, and generate `final_submission.csv`:
```bash
python -m src.run_pipeline --data_dir Dataset_Final --output_sub final_submission.csv
```

---

## Final Deliverable Notebook

The primary deliverable notebook with interactive charts and walkthroughs is available at:
[`SHL_Hiring_Assessment_2026_Final.ipynb`](SHL_Hiring_Assessment_2026_Final.ipynb)
