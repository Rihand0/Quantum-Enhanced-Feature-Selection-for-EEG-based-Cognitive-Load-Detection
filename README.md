# Quantum-Enhanced-Feature-Selection-for-EEG-based-Cognitive-Load-Detection

# EEG Mental Arithmetic Classification — Quantum-Inspired Feature Selection

Subject-independent classification of mental arithmetic vs. resting-state EEG, using a classical feature-engineering + mRMR pipeline combined with a Grover Adaptive Search (GAS) quantum-inspired feature selector. Built as part of a B.Tech research project on quantum-enhanced EEG feature selection, supervised by Dr. Supreeti Kamilya.

## Dataset

[EEG During Mental Arithmetic Tasks](https://physionet.org/content/eegmat/1.0.0/) (PhysioNet) — 36 subjects, EDF format, 500 Hz sampling rate. Each subject has two recordings: a resting/relaxed baseline and a mental-arithmetic cognitive-load condition.

- **Classes:** baseline/resting (label 0) vs. mental arithmetic (label 1)
- **Not included in this repo** — download separately and set `DATA_PATH` in the notebook to point at it.

## Pipeline Overview

| Stage | What it does |
|---|---|
| **1. Preprocessing** | Per-recording: 1–40 Hz bandpass filter → 50 Hz notch filter → ICA ocular-artifact removal (frontal channel used as an EOG proxy, since no dedicated EOG channel exists) |
| **2. Windowing + Feature Extraction** | 4 s windows, 50% overlap. 16 features per channel across 5 categories: time-domain (RMS, std, skewness, kurtosis), Hjorth parameters (activity, mobility, complexity), frequency-domain (relative theta/alpha/beta power via Welch PSD), frequency ratios (theta/alpha, beta/alpha, beta/(alpha+theta)), and entropy-based (spectral, differential, sample entropy) |
| **2.5 Baseline Normalization** | Per-subject z-scoring against each subject's own resting-state segments, to reduce inter-subject amplitude variance |
| **3. mRMR** | Minimum Redundancy Maximum Relevance selects the top 16 hand-crafted features |
| **4. Classical ML Comparison** | Multiple classifiers including a GridSearchCV-tuned SVM (RBF), evaluated on a subject-wise `GroupKFold` split |
| **5. Grover Adaptive Search (GAS)** | Quantum-inspired combinatorial feature-subset search, using a QUBO/Ridge-regression surrogate fitness function in place of exhaustive classical SVM evaluations |
| **Ablation Studies 1–4** | Feature-subset size sweep, per-classifier optimal subset, feature-category contribution analysis, resource-usage comparison |
| **6. LOSO Cross-Validation** | Leave-One-Subject-Out across all 36 subjects — the methodologically honest subject-independent accuracy estimate, since Stage 4's single fixed split only has a handful of held-out test subjects |

### A note on GAS as a search engine, not a post-filter

An earlier version of this pipeline used Grover's algorithm only to re-rank an already classically-selected feature set — amplifying a result rather than actually searching with it. The current version uses GAS as the real combinatorial search engine over the feature space, with a QUBO/Ridge-regression surrogate standing in for a very large number of classical SVM evaluations. This surrogate simplification should be disclosed as such wherever this work is written up — it is not a literal implementation of Gilliam et al. (2021).

## Setup

Designed to run in **Google Colab** with the dataset in Google Drive.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!pip install mne pyedflib antropy mrmr-selection scikit-learn xgboost lightgbm catboost qiskit qiskit-aer qiskit-ibm-runtime
```

Set `DATA_PATH` near the top of the notebook to wherever the PhysioNet EEGMAT dataset lives in your Drive.

## Project Structure

```
.
├── PhysioNet_Dataset_v2_GAS_BASELINE_LOSO_T1.ipynb   # main pipeline notebook
└── README.md
```

## Acknowledgments

Research supervised by **Dr. Supreeti Kamilya**. Based on the PhysioNet EEG During Mental Arithmetic Tasks dataset.
Research supervised by **Dr. Supreeti Kamilya**. Based on the Shin et al. (2018) open EEG+NIRS dataset.
