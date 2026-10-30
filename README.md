# Human Activity Recognition with Hidden Markov Models

Individual Formative 2 project using smartphone accelerometer and gyroscope recordings to classify **standing, walking, jumping and still/no movement**.

## Dataset overview

- 52 labelled recordings: 13 per activity
- 18,594 synchronized sensor rows
- Each activity exceeds the 90-second duration requirement
- Nominal sampling rate: 50 Hz; measured median is approximately 49.87 Hz
- 2-second windows with a 1-second hop
- 42 extracted numeric features per window; 12 selected for the Gaussian HMM
- Recording-level split: 44 training recordings and 8 held-out recordings (two per activity)

The current held-out evaluation reports 45/45 correctly decoded windows (100% accuracy). This is a small test set from one participant and one phone, so it should not be interpreted as proof of perfect real-world generalization.

## Repository structure

```text
HMM_Human_Activity_Recognition/
├── data/
│   ├── raw/<activity>/<recording_id>/
│   ├── processed/        # Combined data, manifest, window features and splits
│   └── features/         # Named raw and normalized feature exports
├── notebooks/
│   ├── 01_data_preprocessing_and_eda.ipynb
│   ├── 02_feature_extraction.ipynb
│   └── 03_hmm_implementation.ipynb
├── scripts/              # Reusable preprocessing, extraction and HMM scripts
├── src/                  # HMM algorithm functions
├── results/              # Metrics, fitted parameters and decoded sequences
├── figures/              # EDA and HMM visualizations
├── report/               # Project report
├── evidence/video_links.md
└── requirements.txt
```

## Run the notebooks in order

Open the notebooks from the project folder and run them in this order:

1. `notebooks/01_data_preprocessing_and_eda.ipynb` — reads the raw recording folders, synchronizes accelerometer and gyroscope samples, checks quality and sampling rate, and writes `data/processed/combined_sensor_data.csv` plus the recording manifest and EDA plots.
2. `notebooks/02_feature_extraction.ipynb` — reads `data/processed/combined_sensor_data.csv`, creates 2-second windows with a 1-second hop, extracts time- and frequency-domain features, fits Z-score values on training windows only, and writes `data/processed/window_features.csv` and related tables. It also writes `data/features/HMM_Window_Features.csv` and the normalized export for standalone use.
3. `notebooks/03_hmm_implementation.ipynb` — reads the newly generated `data/processed/window_features.csv`, trains the Gaussian HMM with Baum-Welch, decodes held-out recordings with Viterbi, and writes metrics, matrices and figures into `results/` and `figures/`.

## Run from the terminal

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/preprocess.py
python scripts/extract_features.py
python scripts/run_hmm.py
```

## Methods

- Calibrated accelerometer and gyroscope streams are joined using matching timestamps. Uncalibrated exports are retained as raw provenance.
- Feature windows are two seconds long, with a one-second hop. Scaling parameters are fitted on training windows only.
- A diagonal-covariance Gaussian HMM uses label-informed initialization followed by Baum-Welch refinement. Training checks the absolute change in training log-likelihood against `1e-5`.
- Viterbi finds the most likely hidden-state sequence for each held-out recording.
- Evaluation reports class sensitivity, specificity and accuracy, plus a confusion matrix.

## Interpretation and limitations

Each recording contains one activity class. As a result, the learned transition matrix is dominated by self-transitions and does not fully model natural transitions between activities. The held-out test recordings come from the same participant, phone and collection period. More participants, devices and continuous mixed-activity sessions would be needed to test broader generalization.

## Video evidence

The four collection-video URLs are listed in `evidence/video_links.md`. Check that all links open for the instructor and that each video meets the assignment's requirements for duration, full-body visibility, face visibility, lighting and proof of recording the activity.

## Before submission

- Insert the student's name and confirm report length against the 4–5-page requirement.
- Confirm the four video links and video contents.
- Upload this folder to GitHub without changing the relative paths.
