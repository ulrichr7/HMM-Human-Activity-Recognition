from pathlib import Path
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ACTIVITIES = ["standing", "walking", "jumping", "still"]
TEST_RECORDINGS_PER_ACTIVITY = 2
RANDOM_SEED = 42


def _read_sensor(path):
    df = pd.read_csv(path)
    required = {"time", "seconds_elapsed", "x", "y", "z"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} missing required columns: {required - set(df.columns)}")
    df = df[["time", "seconds_elapsed", "x", "y", "z"]].copy()
    df["time"] = pd.to_numeric(df["time"], errors="raise").astype("int64")
    for c in ["seconds_elapsed", "x", "y", "z"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    if df.isna().any().any() or not np.isfinite(df[["seconds_elapsed", "x", "y", "z"]].to_numpy()).all():
        raise ValueError(f"Missing or non-finite values in {path}")
    if df["time"].duplicated().any():
        raise ValueError(f"Duplicate timestamps in {path}")
    return df.sort_values("time").reset_index(drop=True)


def build_dataset(project_root=None):
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    raw_root = root / "data" / "raw"
    processed = root / "data" / "processed"
    figures = root / "figures"
    processed.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    combined_parts = []
    examples = {}
    for activity in ACTIVITIES:
        activity_dir = raw_root / activity
        if not activity_dir.exists():
            continue
        rec_dirs = sorted([p for p in activity_dir.iterdir() if p.is_dir()])
        for rec_dir in rec_dirs:
            acc_path = rec_dir / "Accelerometer.csv"
            gyro_path = rec_dir / "Gyroscope.csv"
            if not acc_path.exists() or not gyro_path.exists():
                raise FileNotFoundError(f"Expected calibrated sensor CSVs in {rec_dir}")
            acc = _read_sensor(acc_path)
            gyro = _read_sensor(gyro_path)
            # All included recordings were verified to have exact accelerometer/gyroscope timestamp alignment.
            if len(acc) != len(gyro) or not np.array_equal(acc["time"].to_numpy(), gyro["time"].to_numpy()):
                raise ValueError(f"Accelerometer and gyroscope timestamps do not match exactly in {rec_dir}")
            if not np.allclose(acc["seconds_elapsed"].to_numpy(), gyro["seconds_elapsed"].to_numpy(), atol=1e-8):
                raise ValueError(f"Elapsed timestamps differ in {rec_dir}")

            t = acc["seconds_elapsed"].to_numpy(dtype=float)
            t_rel = t - t[0]
            dt = np.diff(t)
            duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
            median_dt = float(np.median(dt)) if len(dt) else float("nan")
            rate = 1.0 / median_dt if median_dt > 0 else float("nan")
            max_gap_ms = float(np.max(dt) * 1000) if len(dt) else float("nan")

            # Read only non-identifying fields from sanitized metadata.
            metadata_path = rec_dir / "Metadata.csv"
            meta = pd.read_csv(metadata_path, dtype=str).iloc[0].to_dict() if metadata_path.exists() and metadata_path.stat().st_size else {}
            capture_datetime = meta.get("recording time", "")
            device_name = meta.get("device name", "")
            nominal_ms = meta.get("sampleRateMs", "")
            rows = pd.DataFrame({
                "recording_id": rec_dir.name,
                "activity": activity,
                "sample_index": np.arange(len(acc), dtype=int),
                "timestamp_ns": acc["time"].to_numpy(dtype=np.int64),
                "seconds_elapsed": t,
                "time_from_start_s": t_rel,
                "acc_x": acc["x"].to_numpy(dtype=float),
                "acc_y": acc["y"].to_numpy(dtype=float),
                "acc_z": acc["z"].to_numpy(dtype=float),
                "gyro_x": gyro["x"].to_numpy(dtype=float),
                "gyro_y": gyro["y"].to_numpy(dtype=float),
                "gyro_z": gyro["z"].to_numpy(dtype=float),
            })
            combined_parts.append(rows)
            manifest_rows.append({
                "recording_id": rec_dir.name,
                "activity": activity,
                "capture_datetime": capture_datetime,
                "device_name": device_name,
                "nominal_sample_interval_ms": nominal_ms,
                "samples_accelerometer": len(acc),
                "samples_gyroscope": len(gyro),
                "duration_s": duration,
                "median_sampling_rate_hz": rate,
                "median_interval_ms": median_dt * 1000 if np.isfinite(median_dt) else np.nan,
                "max_interval_ms": max_gap_ms,
                "missing_values_in_selected_signals": int(rows[["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]].isna().sum().sum()),
                "timestamp_alignment": "exact",
                "source_csv_directory": str(rec_dir.relative_to(root)),
            })
            if activity not in examples or rec_dir.name < examples[activity][0]:
                examples[activity] = (rec_dir.name, rows.copy())

    if not combined_parts:
        raise RuntimeError(f"No recordings found under {raw_root}")
    manifest = pd.DataFrame(manifest_rows).sort_values(["activity", "recording_id"]).reset_index(drop=True)

    # Hold out complete recordings, not individual rows, to avoid train/test leakage from the same recording.
    rng = np.random.default_rng(RANDOM_SEED)
    test_ids = set()
    for activity in ACTIVITIES:
        ids = sorted(manifest.loc[manifest.activity == activity, "recording_id"].tolist())
        if len(ids) < TEST_RECORDINGS_PER_ACTIVITY + 1:
            raise ValueError(f"Not enough recordings for {activity} to create a train/test split")
        chosen = rng.choice(ids, size=TEST_RECORDINGS_PER_ACTIVITY, replace=False)
        test_ids.update(chosen.tolist())
    manifest["split"] = np.where(manifest["recording_id"].isin(test_ids), "test", "train")

    combined = pd.concat(combined_parts, ignore_index=True)
    combined = combined.merge(manifest[["recording_id", "split"]], on="recording_id", how="left", validate="many_to_one")
    combined = combined[["recording_id", "activity", "split", "sample_index", "timestamp_ns", "seconds_elapsed", "time_from_start_s", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]]
    combined = combined.sort_values(["activity", "recording_id", "sample_index"]).reset_index(drop=True)

    # Per-activity summary from measured first-to-last timestamps in each recording.
    activity_summary = manifest.groupby("activity", sort=False).agg(
        recordings=("recording_id", "count"),
        total_duration_s=("duration_s", "sum"),
        average_duration_s=("duration_s", "mean"),
        median_sampling_rate_hz=("median_sampling_rate_hz", "median"),
        min_sampling_rate_hz=("median_sampling_rate_hz", "min"),
        max_sampling_rate_hz=("median_sampling_rate_hz", "max"),
        total_samples=("samples_accelerometer", "sum"),
        max_inter_sample_gap_ms=("max_interval_ms", "max"),
    ).reset_index()
    activity_summary["target_duration_s"] = 90.0
    activity_summary["meets_90s_target_in_current_bundle"] = activity_summary["total_duration_s"] >= 90.0
    activity_summary["seconds_short_of_target"] = np.maximum(0, 90.0 - activity_summary["total_duration_s"])

    # Write tables.
    combined.to_csv(processed / "combined_sensor_data.csv", index=False, float_format="%.9f")
    combined[combined.split == "train"].to_csv(processed / "train_sensor_data.csv", index=False, float_format="%.9f")
    combined[combined.split == "test"].to_csv(processed / "test_sensor_data.csv", index=False, float_format="%.9f")
    manifest.to_csv(processed / "recording_manifest.csv", index=False, float_format="%.6f")
    activity_summary.to_csv(processed / "activity_summary.csv", index=False, float_format="%.6f")

    # Quality report.
    quality_rows = []
    for activity in ACTIVITIES:
        m = manifest[manifest.activity == activity]
        ids_activity = set(m.recording_id)
        c = combined[combined.activity == activity]
        quality_rows.append({
            "activity": activity,
            "recording_count": len(m),
            "sample_count": len(c),
            "total_duration_s": float(m.duration_s.sum()),
            "median_sampling_rate_hz": float(m.median_sampling_rate_hz.median()),
            "sampling_rate_min_hz": float(m.median_sampling_rate_hz.min()),
            "sampling_rate_max_hz": float(m.median_sampling_rate_hz.max()),
            "selected_signal_missing_values": int(c[["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]].isna().sum().sum()),
            "recordings_in_train": int((m.split == "train").sum()),
            "recordings_in_test": int((m.split == "test").sum()),
            "timestamp_alignment_exact_for_all_recordings": bool((m.timestamp_alignment == "exact").all()),
        })
    pd.DataFrame(quality_rows).to_csv(processed / "data_quality_summary.csv", index=False, float_format="%.6f")

    # Plots: one representative recording per class, preserving native signal values and units.
    for prefix, cols, title, filename in [
        ("acc", ["acc_x","acc_y","acc_z"], "Accelerometer signals from one representative recording per activity", "sample_accelerometer_signals.png"),
        ("gyro", ["gyro_x","gyro_y","gyro_z"], "Gyroscope signals from one representative recording per activity", "sample_gyroscope_signals.png"),
    ]:
        fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=False)
        for ax, activity in zip(axes, ACTIVITIES):
            rec_id = examples[activity][0]
            ex = examples[activity][1]
            for col in cols:
                ax.plot(ex["time_from_start_s"], ex[col], label=col.replace(prefix+"_", ""))
            ax.set_title(f"{activity.title()} — {rec_id}")
            ax.set_ylabel("Sensor value")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", ncol=3, fontsize=8)
        axes[-1].set_xlabel("Time from start (s)")
        fig.suptitle(title, y=1.005, fontsize=13)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = activity_summary.set_index("activity").reindex(ACTIVITIES)
    bars = ax.bar(plot_df.index, plot_df["total_duration_s"])
    ax.axhline(90, linestyle="--", linewidth=1.5, label="90-second target")
    for bar, value in zip(bars, plot_df["total_duration_s"]):
        ax.text(bar.get_x()+bar.get_width()/2, value+0.8, f"{value:.1f}s", ha="center", va="bottom", fontsize=9)
    ax.set_title("Recorded sensor duration by activity (timestamp span)")
    ax.set_ylabel("Total duration (seconds)")
    ax.set_ylim(0, max(100, float(plot_df["total_duration_s"].max()) + 10))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "activity_duration_summary.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Rate distribution by recording.
    fig, ax = plt.subplots(figsize=(11, 5))
    for activity in ACTIVITIES:
        grp = manifest[manifest.activity == activity]
        ax.scatter(grp.recording_id, grp.median_sampling_rate_hz, label=activity, s=28)
    ax.axhline(50, linestyle="--", linewidth=1.2, label="Nominal 50 Hz")
    ax.set_title("Measured median sampling rate by recording")
    ax.set_ylabel("Sampling rate (Hz)")
    ax.set_xlabel("Recording ID")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "sampling_rate_by_recording.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {"manifest": manifest, "combined": combined, "activity_summary": activity_summary, "quality_rows": pd.DataFrame(quality_rows)}


if __name__ == "__main__":
    result = build_dataset()
    print("Prepared row-level dataset:", len(result["combined"]), "samples")
    print(result["activity_summary"].to_string(index=False))
