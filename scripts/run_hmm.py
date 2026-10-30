#!/usr/bin/env python3
"""Train the Gaussian HMM and regenerate evaluation outputs from window features.
Run from the project root: python scripts/run_hmm.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hmm_from_scratch import logsumexp, log_emission, forward_backward, baum_welch, viterbi
FEATURE_FILE = ROOT / "data" / "processed" / "window_features.csv"
if not FEATURE_FILE.exists():
    FEATURE_FILE = ROOT / "data" / "features" / "HMM_Window_Features.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True, parents=True)
FIGURES.mkdir(exist_ok=True, parents=True)

states = ["standing", "walking", "jumping", "still"]
state_idx = {s: i for i, s in enumerate(states)}
features = [
    "acc_mag_mean", "acc_mag_std", "acc_mag_rms",
    "acc_mag_dominant_frequency_hz", "acc_mag_spectral_energy",
    "gyro_mag_mean", "gyro_mag_std", "gyro_mag_rms",
    "gyro_mag_dominant_frequency_hz", "gyro_mag_spectral_energy",
    "acc_signal_magnitude_area", "gyro_signal_magnitude_area",
]

df = pd.read_csv(FEATURE_FILE)
train, test = df[df["split"] == "train"].copy(), df[df["split"] == "test"].copy()

# Fit standardization on training windows only to avoid test-data leakage.
mu = train[features].mean().to_numpy()
sd = train[features].std(ddof=0).to_numpy()
sd = np.where(sd < 1e-8, 1.0, sd)

def make_sequences(frame):
    out = []
    for rid, g in frame.groupby("recording_id", sort=False):
        g = g.sort_values("window_index")
        X = (g[features].to_numpy() - mu) / sd
        out.append((rid, X, g["activity"].to_numpy()))
    return out

train_seqs, test_seqs = make_sequences(train), make_sequences(test)
K, D = len(states), len(features)

# Label-informed initialization from the training set, then Baum-Welch EM refinement.
means, variances = np.zeros((K, D)), np.zeros((K, D))
for state, i in state_idx.items():
    Xs = (train.loc[train.activity == state, features].to_numpy() - mu) / sd
    means[i] = Xs.mean(axis=0)
    variances[i] = np.maximum(Xs.var(axis=0), 0.05)
A = np.ones((K, K)) * 0.05
pi = np.ones(K) * 0.05
for _, X, y in train_seqs:
    yi = np.array([state_idx[v] for v in y])
    pi[yi[0]] += 1
    for a, b in zip(yi[:-1], yi[1:]):
        A[a, b] += 1
pi /= pi.sum()
A /= A.sum(axis=1, keepdims=True)

means, variances, A, pi, ll_history = baum_welch(
    train_seqs, means, variances, A, pi, max_iter=100, tol=1e-5
)

# Decode each held-out recording independently.
decoded_rows, y_true, y_pred = [], [], []
for rid, X, y in test_seqs:
    path, _ = viterbi(X, pi, A, means, variances)
    preds = [states[i] for i in path]
    for w, (yt, yp) in enumerate(zip(y, preds)):
        decoded_rows.append([rid, w, yt, yp, yt == yp])
    y_true.extend(y)
    y_pred.extend(preds)
decoded = pd.DataFrame(decoded_rows, columns=["recording_id", "window_index", "true_activity", "predicted_activity", "correct"])
decoded.to_csv(RESULTS / "decoded_test_sequences.csv", index=False)

cm = np.zeros((K, K), dtype=int)
for yt, yp in zip(y_true, y_pred):
    cm[state_idx[yt], state_idx[yp]] += 1
N = cm.sum()
rows = []
for state, i in state_idx.items():
    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp
    tn = N - tp - fn - fp
    rows.append([state, int(cm[i, :].sum()), tp / (tp + fn) if tp + fn else np.nan,
                 tn / (tn + fp) if tn + fp else np.nan, (tp + tn) / N if N else np.nan])
metrics = pd.DataFrame(rows, columns=["state", "number_of_test_windows", "sensitivity", "specificity", "overall_accuracy"])
metrics.to_csv(RESULTS / "evaluation_metrics.csv", index=False)
pd.DataFrame(A, index=states, columns=states).to_csv(RESULTS / "transition_matrix.csv")
pd.DataFrame(means, index=states, columns=features).to_csv(RESULTS / "emission_means_zscore.csv")
pd.DataFrame(variances, index=states, columns=features).to_csv(RESULTS / "emission_variances_zscore.csv")
pd.DataFrame({"state": states, "initial_probability": pi}).to_csv(RESULTS / "initial_state_probabilities.csv", index=False)
pd.DataFrame({"iteration": np.arange(1, len(ll_history) + 1), "log_likelihood": ll_history}).to_csv(RESULTS / "baum_welch_log_likelihood.csv", index=False)

# Confusion matrix.
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, aspect="auto")
ax.set_xticks(range(K), states, rotation=25); ax.set_yticks(range(K), states)
ax.set_xlabel("Predicted activity"); ax.set_ylabel("True activity"); ax.set_title("Confusion Matrix - Unseen Test Data")
fig.colorbar(im, ax=ax, label="Windows")
for i in range(K):
    for j in range(K): ax.text(j, i, str(cm[i, j]), ha="center", va="center")
fig.tight_layout(); fig.savefig(FIGURES / "hmm_confusion_matrix.png", dpi=180); plt.close(fig)

# Transition probability heatmap.
fig, ax = plt.subplots(figsize=(7, 5)); im = ax.imshow(A, aspect="auto")
ax.set_xticks(range(K), states, rotation=25); ax.set_yticks(range(K), states)
ax.set_xlabel("Next state"); ax.set_ylabel("Current state"); ax.set_title("HMM Transition Probabilities")
fig.colorbar(im, ax=ax, label="Probability")
for i in range(K):
    for j in range(K): ax.text(j, i, f"{A[i,j]:.3f}", ha="center", va="center")
fig.tight_layout(); fig.savefig(FIGURES / "hmm_transition_probability_heatmap.png", dpi=180); plt.close(fig)

# Emission means heatmap (selected features).
selected = ["acc_mag_mean", "acc_mag_std", "acc_mag_dominant_frequency_hz", "acc_mag_spectral_energy", "gyro_mag_mean", "gyro_mag_dominant_frequency_hz", "gyro_mag_spectral_energy", "acc_signal_magnitude_area"]
idx = [features.index(v) for v in selected]
fig, ax = plt.subplots(figsize=(10, 5)); im = ax.imshow(means[:, idx], aspect="auto")
ax.set_xticks(range(len(selected)), [s.replace("_", " ") for s in selected], rotation=45, ha="right")
ax.set_yticks(range(K), states); ax.set_xlabel("Emission feature"); ax.set_ylabel("State")
ax.set_title("Gaussian Emission Parameter Means (z-score)"); fig.colorbar(im, ax=ax, label="Mean")
fig.tight_layout(); fig.savefig(FIGURES / "hmm_emission_parameter_heatmap.png", dpi=180); plt.close(fig)

# Normalized emission likelihood by true test class, as a visualization only (not a generative probability calibration report).
like_rows = []
for _, X, y in test_seqs:
    le = log_emission(X, means, variances)
    from src.hmm_from_scratch import logsumexp
    p = np.exp(le - logsumexp(le, axis=1, keepdims=True))
    for label, probs in zip(y, p): like_rows.append([label] + probs.tolist())
like_df = pd.DataFrame(like_rows, columns=["true_activity"] + states)
avg = like_df.groupby("true_activity")[states].mean().reindex(states)
fig, ax = plt.subplots(figsize=(7, 5)); im = ax.imshow(avg.values, aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(K), states, rotation=25); ax.set_yticks(range(K), states)
ax.set_xlabel("Candidate hidden state"); ax.set_ylabel("True activity"); ax.set_title("Normalized Emission Likelihood - Test Windows")
fig.colorbar(im, ax=ax, label="Normalized likelihood")
for i in range(K):
    for j in range(K): ax.text(j, i, f"{avg.iloc[i,j]:.2f}", ha="center", va="center")
fig.tight_layout(); fig.savefig(FIGURES / "hmm_emission_probability_heatmap.png", dpi=180); plt.close(fig)

# Log-likelihood convergence.
fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(range(1, len(ll_history)+1), ll_history, marker="o")
ax.set_xlabel("Baum-Welch iteration"); ax.set_ylabel("Training log-likelihood"); ax.set_title("Baum-Welch Convergence")
fig.tight_layout(); fig.savefig(FIGURES / "hmm_baum_welch_convergence.png", dpi=180); plt.close(fig)

print(f"Train recordings: {len(train_seqs)} | Test recordings: {len(test_seqs)}")
print(f"Test windows: {N} | Accuracy: {np.mean(np.array(y_true) == np.array(y_pred)):.3%}")
print(f"Baum-Welch iterations: {len(ll_history)} | Last training log-likelihood: {ll_history[-1]:.6f}")
print(metrics.to_string(index=False))

# Decoded-state sequence plot. Since recordings are single-activity segments, lines are expected to remain in one state per recording.
fig, axes = plt.subplots(4, 2, figsize=(12, 9), sharey=True)
axes = np.ravel(axes)
for ax, (recording_id, group) in zip(axes, decoded.groupby("recording_id", sort=False)):
    true_codes = [state_idx[v] for v in group["true_activity"]]
    pred_codes = [state_idx[v] for v in group["predicted_activity"]]
    ax.step(range(len(true_codes)), true_codes, where="mid", label="True")
    ax.step(range(len(pred_codes)), pred_codes, where="mid", linestyle="--", label="Viterbi")
    ax.set_title(recording_id)
    ax.set_yticks(range(K), states)
    ax.set_xlabel("Window index")
    ax.grid(alpha=0.2)
axes[0].legend()
fig.suptitle("Viterbi-Decoded Activity Sequences - Unseen Test Recordings")
fig.tight_layout()
fig.savefig(FIGURES / "hmm_viterbi_decoded_sequences.png", dpi=180)
plt.close(fig)
