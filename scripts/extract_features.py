"""Window-level feature extraction for the HMM human activity recognition project.

Uses 2-second windows with 1-second hop (50% overlap). The train/test split is
inherited from the recording-level manifest contained in the merged sensor CSV.
Z-score parameters are fitted only on training windows to prevent leakage.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ACTIVITIES = ["standing", "walking", "jumping", "still"]
SENSOR_AXES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
WINDOW_SECONDS = 2.0
HOP_SECONDS = 1.0


def _safe_corr(a, b):
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else 0.0


def _dominant_frequency_and_energy(x, fs):
    """Return dominant non-DC frequency and normalized FFT spectral energy."""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < 4 or fs <= 0:
        return 0.0, 0.0
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    amplitude = np.abs(spectrum)
    eligible = freqs >= 0.25  # ignore DC and very slow drift
    if not eligible.any():
        dom = 0.0
    else:
        idxs = np.flatnonzero(eligible)
        dom = float(freqs[idxs[np.argmax(amplitude[idxs])]])
    # Normalized sum of squared FFT magnitudes, excluding the DC component.
    energy = float(np.sum(np.abs(spectrum[1:]) ** 2) / (n ** 2))
    return dom, energy


def _features_for_window(g, fs):
    f = {}
    signal_arrays = {c: g[c].to_numpy(dtype=float) for c in SENSOR_AXES}
    for axis, x in signal_arrays.items():
        f[f"{axis}_mean"] = float(np.mean(x))
        f[f"{axis}_std"] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        f[f"{axis}_variance"] = float(np.var(x, ddof=1)) if len(x) > 1 else 0.0
        f[f"{axis}_rms"] = float(np.sqrt(np.mean(x ** 2)))

    acc = np.column_stack([signal_arrays[f"acc_{a}"] for a in "xyz"])
    gyro = np.column_stack([signal_arrays[f"gyro_{a}"] for a in "xyz"])
    acc_mag = np.sqrt(np.sum(acc ** 2, axis=1))
    gyro_mag = np.sqrt(np.sum(gyro ** 2, axis=1))
    for name, mag in [("acc_mag", acc_mag), ("gyro_mag", gyro_mag)]:
        f[f"{name}_mean"] = float(np.mean(mag))
        f[f"{name}_std"] = float(np.std(mag, ddof=1)) if len(mag) > 1 else 0.0
        f[f"{name}_rms"] = float(np.sqrt(np.mean(mag ** 2)))
        dom, energy = _dominant_frequency_and_energy(mag, fs)
        f[f"{name}_dominant_frequency_hz"] = dom
        f[f"{name}_spectral_energy"] = energy

    # Signal magnitude area per sample: mean of |x| + |y| + |z|.
    f["acc_signal_magnitude_area"] = float(np.mean(np.sum(np.abs(acc), axis=1)))
    f["gyro_signal_magnitude_area"] = float(np.mean(np.sum(np.abs(gyro), axis=1)))

    for prefix, arrays in [("acc", acc), ("gyro", gyro)]:
        for (i, j, pair) in [(0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")]:
            f[f"{prefix}_corr_{pair}"] = _safe_corr(arrays[:, i], arrays[:, j])
    return f


def extract_window_features(combined, window_seconds=WINDOW_SECONDS, hop_seconds=HOP_SECONDS):
    required = {"recording_id", "activity", "split", "time_from_start_s", *SENSOR_AXES}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"combined_sensor_data.csv is missing columns: {sorted(missing)}")

    rows = []
    for recording_id, g0 in combined.groupby("recording_id", sort=True):
        g = g0.sort_values("time_from_start_s").reset_index(drop=True)
        activity = str(g["activity"].iloc[0])
        split = str(g["split"].iloc[0])
        t = pd.to_numeric(g["time_from_start_s"], errors="coerce").to_numpy(dtype=float)
        dt = np.diff(t)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt) == 0:
            continue
        fs = float(1.0 / np.median(dt))
        last_t = float(np.max(t))
        start = 0.0
        win_index = 0
        min_samples = max(4, int(fs * window_seconds * 0.80))
        while start + window_seconds <= last_t + (1.0 / fs):
            mask = (t >= start) & (t < start + window_seconds)
            gw = g.loc[mask]
            tw = t[mask]
            if len(gw) >= min_samples and len(tw) > 1 and (tw.max() - tw.min()) >= window_seconds * 0.75:
                feats = _features_for_window(gw, fs)
                row = {
                    "recording_id": recording_id,
                    "activity": activity,
                    "split": split,
                    "window_index": win_index,
                    "window_start_s": float(start),
                    "window_end_s": float(start + window_seconds),
                    "window_samples": int(len(gw)),
                    "estimated_sampling_rate_hz": fs,
                }
                row.update(feats)
                rows.append(row)
                win_index += 1
            start += hop_seconds

    features = pd.DataFrame(rows)
    if features.empty:
        raise ValueError("No windows were extracted. Check the input data and window duration.")
    numeric = features.select_dtypes(include=[np.number]).columns
    if not np.isfinite(features[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Feature extraction generated non-finite values.")
    return features


def zscore_using_training_only(features, feature_columns):
    train_mask = features["split"].eq("train")
    if not train_mask.any():
        raise ValueError("No training windows found in the split column.")
    means = features.loc[train_mask, feature_columns].mean()
    scales = features.loc[train_mask, feature_columns].std(ddof=0).replace(0, 1.0)
    scaled_values = (features[feature_columns] - means) / scales
    scaled = features[[c for c in features.columns if c not in feature_columns]].copy()
    for c in feature_columns:
        scaled[c] = scaled_values[c]
    scaler = pd.DataFrame({"feature": feature_columns, "training_mean": means.values, "training_std_used": scales.values})
    return scaled, scaler


def build_features(project_root):
    project_root = Path(project_root)
    processed = project_root / "data" / "processed"
    figures = project_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    combined_path = processed / "combined_sensor_data.csv"
    if not combined_path.exists():
        raise FileNotFoundError(f"Could not find {combined_path}. Run from the project folder or adjust project_root.")
    combined = pd.read_csv(combined_path)
    features = extract_window_features(combined)
    metadata_cols = ["recording_id", "activity", "split", "window_index", "window_start_s", "window_end_s", "window_samples", "estimated_sampling_rate_hz"]
    feature_cols = [c for c in features.columns if c not in metadata_cols]
    scaled, scaler = zscore_using_training_only(features, feature_cols)

    processed.mkdir(parents=True, exist_ok=True)
    features.to_csv(processed / "window_features.csv", index=False)
    features.loc[features["split"].eq("train")].to_csv(processed / "train_window_features.csv", index=False)
    features.loc[features["split"].eq("test")].to_csv(processed / "test_window_features.csv", index=False)
    scaled.to_csv(processed / "window_features_zscore.csv", index=False)
    scaled.loc[scaled["split"].eq("train")].to_csv(processed / "train_window_features_zscore.csv", index=False)
    scaled.loc[scaled["split"].eq("test")].to_csv(processed / "test_window_features_zscore.csv", index=False)
    scaler.to_csv(processed / "feature_scaler_parameters.csv", index=False)

    # Summary for notebook and report.
    summary = features.groupby(["activity", "split"], as_index=False).agg(
        windows=("window_index", "size"),
        recordings=("recording_id", "nunique"),
        median_window_samples=("window_samples", "median"),
        median_sampling_rate_hz=("estimated_sampling_rate_hz", "median"),
    )
    summary.to_csv(processed / "window_feature_summary.csv", index=False)

    plot_cols = ["acc_mag_std", "gyro_mag_std", "acc_mag_dominant_frequency_hz", "gyro_mag_spectral_energy"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flat, plot_cols):
        series, labels = [], []
        for activity in ACTIVITIES:
            vals = features.loc[features["activity"].eq(activity), col].dropna().to_numpy()
            if len(vals):
                series.append(vals)
                labels.append(activity.capitalize())
        ax.boxplot(series, tick_labels=labels, showfliers=False)
        ax.set_title(col.replace("_", " "))
        ax.set_ylabel("Feature value")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Selected window features by activity (all recordings; exploratory only)")
    fig.tight_layout()
    fig.savefig(figures / "window_feature_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Feature-count summary plot
    counts = features.groupby("activity")["window_index"].count().reindex(ACTIVITIES)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Number of extracted windows by activity")
    ax.set_xlabel("Activity")
    ax.set_ylabel("Windows")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "window_counts_by_activity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {"features": features, "scaled": scaled, "scaler": scaler, "summary": summary, "feature_columns": feature_cols}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = build_features(root)
    print("Feature extraction complete")
    print(f"Windows: {len(result['features'])}; numeric features: {len(result['feature_columns'])}")
    print(result["summary"].to_string(index=False))
