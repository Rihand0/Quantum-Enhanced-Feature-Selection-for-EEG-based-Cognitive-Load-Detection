# Quantum-Enhanced-Feature-Selection-for-EEG-based-Cognitive-Load-Detection

# EEG Mental Arithmetic Classification — Quantum-Inspired Feature Selection

Subject-independent classification of mental arithmetic vs. resting-state EEG, using a classical feature-engineering + mRMR pipeline combined with a Grover Adaptive Search (GAS) quantum-inspired feature selector. Built as part of a B.Tech research project on quantum-enhanced EEG feature selection, supervised by Dr. Supreeti Kamilya.

The primary results for this work were obtained on the **PhysioNet EEG mental arithmetic dataset** (36 subjects). This notebook applies the same pipeline to the **Shin et al. EEG+NIRS dataset** (29 subjects) as a secondary validation; it did not reach the same level of performance as the PhysioNet pipeline, which is expected given this dataset's harder, better-controlled rest-vs-task paradigm (see [Dataset](#dataset) below).

## Dataset

[Shin et al. (2018) EEG+NIRS dataset](https://doi.org/10.1038/sdata.2018.3) — 29 subjects, simultaneous EEG/NIRS recordings across three cognitive tasks. This pipeline uses only the **mental arithmetic sessions** (sessions 1, 3, 5 per subject) and only the EEG channels (30 channels, VEOG/HEOG dropped).

- **Classes:** resting baseline (label 0) vs. mental arithmetic (label 1), alternating blocks within each session
- **Not included in this repo** — download separately and set `DATASET_PATH` in the notebook to point at it. The loader auto-detects whether the dataset's pre-cleaned `without occular artifact` folder is present and uses it when available (skips redundant in-pipeline ICA).

## Pipeline Overview

| Stage | What it does |
|---|---|
| **1. Preprocessing** | Per-session: linear detrend → bad-channel detection/interpolation → 1–40 Hz bandpass → 50 Hz notch → ICA ocular-artifact removal (skipped automatically if pre-cleaned data is used) → Common Average Reference → per-block baseline (DC-offset) correction |
| **2. Windowing + Feature Extraction** | 12 s windows, 50% overlap, wavelet-denoised. 16 features/channel (480 total across 30 channels): relative theta/alpha power, std, differential entropy, RMS, theta/alpha ratio, Hjorth activity/mobility/complexity, energy, spectral entropy, wavelet energy/entropy, absolute theta/alpha/beta power |
| **2.1 Baseline Normalization** | Per-subject z-scoring against each subject's own resting-state segments, to reduce inter-subject amplitude variance |
| **3. mRMR** | Minimum Redundancy Maximum Relevance selects the top 16 hand-crafted features from the 480 |
| **CSP** | Common Spatial Patterns spatial filters (8 components), fit only on the training fold's raw windows, concatenated onto the mRMR-selected features |
| **4. Classical ML Comparison** | 9 classifiers (Logistic Regression, SVM (RBF, GridSearchCV-tuned), Random Forest, Decision Tree, KNN, MLP, XGBoost, LightGBM, CatBoost), subject-wise `GroupKFold` split. Block-level majority voting reported alongside raw window-level accuracy |
| **5. Grover Adaptive Search (GAS)** | Quantum-inspired combinatorial feature-subset search over the mRMR+CSP feature set, using a QUBO/Ridge-regression surrogate fitness function in place of ~2000 classical SVM evaluations |
| **6. LOSO Cross-Validation** | Leave-One-Subject-Out across all 29 subjects — the methodologically honest subject-independent accuracy estimate, since Stage 4's single fixed split only has a handful of held-out test subjects |
| **Ablation Studies 1–4** | Feature-subset size sweep, per-classifier optimal subset, feature-category contribution analysis, resource-usage comparison |

### A note on GAS as a search engine, not a post-filter

An earlier version of this pipeline used Grover's algorithm only to re-rank an already classically-selected feature set — amplifying a result rather than actually searching with it. The current version uses GAS as the real combinatorial search engine over the feature space, with a QUBO/Ridge-regression surrogate standing in for ~2000 classical SVM evaluations (replaced by ~80–150 total per run). This surrogate simplification should be disclosed as such wherever this work is written up — it is not a literal implementation of Gilliam et al. (2021).

## Setup

Designed to run in **Google Colab** with the dataset in Google Drive.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!pip install mne pyedflib antropy PyWavelets mrmr-selection scikit-learn xgboost lightgbm catboost qiskit qiskit-aer qiskit-ibm-runtime
```

Set `DATASET_PATH` near the top of the notebook to wherever the Shin dataset lives in your Drive.

## Running the Notebook

The notebook includes a **checkpoint** partway through (after Stage 4, before GAS). Stage 1's EEG loading/ICA + Stage 2–4 are the heaviest, longest-running part of the pipeline — on Colab's standard RAM tier, running the entire notebook start to finish in one process can exhaust memory.

1. Run top to bottom through Stage 4. The checkpoint cell saves everything Stage 5 onward needs to Drive as a single `.pkl` file.
2. **Runtime → Restart session** (not "Restart and run all").
3. Re-run only: the Drive-mount cell, the imports cell (no need to re-run `!pip install` — packages persist across a session restart), and the "Reload Checkpoint" cell.
4. Continue from Stage 5 onward in a fresh process.

If Stage 1 is still unstable on your Colab tier, two toggles near the top of that cell let you shed the heaviest optional steps without touching any other code:

```python
ENABLE_ICA = ...                   # auto-disabled if pre-cleaned EEG data is detected
ENABLE_BAD_CHANNEL_INTERP = True   # bad-channel detection + interpolation
```

## Project Structure

```
.
├── Shin_Dataset_Pipeline_v9_baseline_GAS_T2_Final.ipynb   # main pipeline notebook
└── README.md
```

## Acknowledgments

Research supervised by **Dr. Supreeti Kamilya**. Based on the Shin et al. (2018) open EEG+NIRS dataset.
