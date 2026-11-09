
# ============================================================
# MODULE 4 - MPU6050 MOTION MODEL (SIMULATION FIRST)
# One-file project: M4-A1 -> M4-A4 -> M4-B -> M4-C -> M4-D
# Current implementation starts with M4-A1 (synthetic RAW signals).
# No ESP32 hardware, sensor filtering, or real measurements required.
# ============================================================

from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------- CONFIG ----------------------------
SEED = 42
FS_HZ = 10                    # simulation sampling frequency
DT_SEC = 1.0 / FS_HZ
TOTAL_MIN = 240
WINDOW_SEC = 10
G0 = 9.80665

# Illustrative state schedule. These are simulation assumptions,
# NOT measured failure times or universal landslide thresholds.
STATE_SCHEDULE = [
    ("STABLE", 0, 10),
    ("WATER_ACCUMULATION", 10, 180),
    ("DEFORMATION", 180, 230),
    ("SLIDING", 230, 240),
]

# Illustrative sensor ranges, matching common MPU6050 default ranges.
ACC_LSB_PER_G = 16384.0       # +/-2g
GYRO_LSB_PER_DPS = 131.0      # +/-250 deg/s

# ---------------------- COMMON HELPERS -----------------------
def state_at_minutes(t_min):
    for name, start, end in STATE_SCHEDULE:
        if start <= t_min < end:
            return name
    return STATE_SCHEDULE[-1][0]

def smooth_step(x, x0, width):
    z = np.clip((x - x0) / max(width, 1e-9), 0, 1)
    return z * z * (3 - 2 * z)

def make_output_dirs(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)

# ------------------------- M4-A1 ------------------------------
def simulate_raw_signals(fs_hz=FS_HZ, total_min=TOTAL_MIN, seed=SEED, amplitude_overrides=None, timing_overrides=None):
    """
    Generate synthetic accelerometer and gyroscope signals.
    Acceleration includes gravity + illustrative dynamic motion.
    Gyroscope is angular velocity in deg/s.
    """
    rng = np.random.default_rng(seed)
    n = int(total_min * 60 * fs_hz)
    t = np.arange(n, dtype=np.float64) / fs_hz
    t_min = t / 60.0
    dt_sec = 1.0 / float(fs_hz)

    timing = {
        "deformation_start_min": 180.0,
        "sliding_start_min": 230.0,
        "deformation_ramp_min": 8.0,
        "sliding_ramp_min": 1.5,
    }
    if timing_overrides:
        timing.update(timing_overrides)

    deformation_start = float(timing["deformation_start_min"])
    sliding_start = float(timing["sliding_start_min"])
    if not (0 < deformation_start < sliding_start < total_min):
        raise ValueError("Cần 0 < deformation_start < sliding_start < total_min.")
    if timing["deformation_ramp_min"] <= 0 or timing["sliding_ramp_min"] <= 0:
        raise ValueError("Thời gian chuyển tiếp phải lớn hơn 0.")

    state = np.full(n, "WATER_ACCUMULATION", dtype=object)
    state[t_min < 10.0] = "STABLE"
    state[(t_min >= deformation_start) & (t_min < sliding_start)] = "DEFORMATION"
    state[t_min >= sliding_start] = "SLIDING"

    # Smooth transition envelopes use independently configurable start times
    # and ramp durations; discrete state labels remain based on phase boundaries.
    deform = smooth_step(t_min, deformation_start, timing["deformation_ramp_min"])
    sliding = smooth_step(t_min, sliding_start, timing["sliding_ramp_min"])

    # Slowly changing tilt: starts near 0 degrees, increases during deformation.
    tilt_deg = 0.15 + 0.05*np.sin(2*np.pi*t/180.0)
    tilt_deg += 4.0*deform + 8.0*sliding
    tilt_rad = np.deg2rad(tilt_deg)

    # Gravity projection for a rotation mainly around the Y axis.
    ax_gravity = G0 * np.sin(tilt_rad)
    ay_gravity = np.zeros_like(t)
    az_gravity = G0 * np.cos(tilt_rad)

    # Motion amplitude by state; deliberately illustrative.
    amp = np.zeros_like(t)
    amp_values = {
        "STABLE": 0.003,
        "WATER_ACCUMULATION": 0.004,
        "DEFORMATION": 0.025,
        "SLIDING": 0.18,
    }
    if amplitude_overrides:
        amp_values.update(amplitude_overrides)
    for state_name, amp_value in amp_values.items():
        amp[state == state_name] = float(amp_value)

    # A mixture of low-frequency movement and short vibration components.
    f1, f2 = 0.7, 2.0
    phase = rng.uniform(0, 2*np.pi, size=3)
    ax_dyn = amp * (np.sin(2*np.pi*f1*t + phase[0]) +
                    0.25*np.sin(2*np.pi*f2*t + phase[1]))
    ay_dyn = amp * (0.7*np.sin(2*np.pi*f1*t + phase[1]))
    az_dyn = amp * (0.5*np.sin(2*np.pi*f2*t + phase[2]))

    # Gyro: gradual tilt rate + oscillatory angular motion.
    tilt_rate = np.gradient(tilt_deg, dt_sec)  # deg/s
    gx = 0.015*np.sin(2*np.pi*0.15*t) + amp*8*np.sin(2*np.pi*f1*t)
    gy = tilt_rate + amp*5*np.sin(2*np.pi*f2*t + phase[1])
    gz = 0.01*np.cos(2*np.pi*0.1*t) + amp*3*np.sin(2*np.pi*f1*t + phase[2])

    # Small synthetic sensor noise. This is only a simulated signal property,
    # not a real sensor noise calibration.
    noise_acc = rng.normal(0, 0.002, size=(n, 3))
    noise_gyro = rng.normal(0, 0.03, size=(n, 3))

    ax = ax_gravity + ax_dyn + noise_acc[:, 0]
    ay = ay_gravity + ay_dyn + noise_acc[:, 1]
    az = az_gravity + az_dyn + noise_acc[:, 2]
    gx = gx + noise_gyro[:, 0]
    gy = gy + noise_gyro[:, 1]
    gz = gz + noise_gyro[:, 2]

    # Convert simulated physical values into approximate MPU6050 RAW counts.
    ax_raw = np.clip(np.rint(ax/G0*ACC_LSB_PER_G), -32768, 32767).astype(np.int16)
    ay_raw = np.clip(np.rint(ay/G0*ACC_LSB_PER_G), -32768, 32767).astype(np.int16)
    az_raw = np.clip(np.rint(az/G0*ACC_LSB_PER_G), -32768, 32767).astype(np.int16)
    gx_raw = np.clip(np.rint(gx*GYRO_LSB_PER_DPS), -32768, 32767).astype(np.int16)
    gy_raw = np.clip(np.rint(gy*GYRO_LSB_PER_DPS), -32768, 32767).astype(np.int16)
    gz_raw = np.clip(np.rint(gz*GYRO_LSB_PER_DPS), -32768, 32767).astype(np.int16)

    return pd.DataFrame({
        "time_sec": t,
        "time_min": t_min,
        "state": state,
        "tilt_reference_deg": tilt_deg,
        "ax_raw": ax_raw, "ay_raw": ay_raw, "az_raw": az_raw,
        "gx_raw": gx_raw, "gy_raw": gy_raw, "gz_raw": gz_raw,
        "ax_m_s2": ax, "ay_m_s2": ay, "az_m_s2": az,
        "gx_deg_s": gx, "gy_deg_s": gy, "gz_deg_s": gz,
    })

def plot_raw(df, outdir):
    # Plot a representative 60-second segment from each state.
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    windows = [
        ("STABLE", 1, 2),
        ("WATER_ACCUMULATION", 60, 61),
        ("DEFORMATION", 200, 201),
        ("SLIDING", 235, 236),
    ]
    for ax, (label, start, end) in zip(axes, windows):
        seg = df[(df.time_min >= start) & (df.time_min < end)]
        ax.plot(seg.time_sec, seg.ax_m_s2, label="ax")
        ax.plot(seg.time_sec, seg.ay_m_s2, label="ay")
        ax.plot(seg.time_sec, seg.az_m_s2, label="az")
        ax.set_title(f"{label}: gia tốc RAW mô phỏng")
        ax.set_ylabel("m/s²")
        ax.grid(alpha=0.25)
        ax.legend(ncol=3)
    axes[-1].set_xlabel("Thời gian (s)")
    fig.tight_layout()
    fig.savefig(outdir/"figures"/"M4_A1_raw_acceleration_by_state.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    for ax, (label, start, end) in zip(axes, windows):
        seg = df[(df.time_min >= start) & (df.time_min < end)]
        ax.plot(seg.time_sec, seg.gx_deg_s, label="gx")
        ax.plot(seg.time_sec, seg.gy_deg_s, label="gy")
        ax.plot(seg.time_sec, seg.gz_deg_s, label="gz")
        ax.set_title(f"{label}: vận tốc góc RAW mô phỏng")
        ax.set_ylabel("°/s")
        ax.grid(alpha=0.25)
        ax.legend(ncol=3)
    axes[-1].set_xlabel("Thời gian (s)")
    fig.tight_layout()
    fig.savefig(outdir/"figures"/"M4_A1_raw_gyroscope_by_state.png", dpi=170)
    plt.close(fig)

    # Full timeline state and reference tilt.
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df.time_min, df.tilt_reference_deg)
    ax.set(xlabel="Thời gian (phút)", ylabel="Góc nghiêng tham chiếu (độ)",
           title="M4-A1 – Góc nghiêng tham chiếu theo kịch bản giả lập")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir/"figures"/"M4_A1_tilt_reference.png", dpi=170)
    plt.close(fig)

# ------------------------- M4-A2 ------------------------------
def extract_features(df, fs_hz=FS_HZ, window_sec=WINDOW_SEC):
    """M4-A2 window features, vectorized for faster and reproducible processing.

    Simulation-only gravity estimate uses the known reference tilt. For real
    IMU data, replace this estimate with the selected physical filter pipeline.
    """
    if fs_hz <= 0 or window_sec <= 0:
        raise ValueError("fs_hz và window_sec phải lớn hơn 0.")
    samples_float = fs_hz * window_sec
    samples = int(round(samples_float))
    if samples < 1 or not np.isclose(samples, samples_float):
        raise ValueError("fs_hz * window_sec phải là số nguyên dương để chia cửa sổ.")

    tilt = np.deg2rad(df["tilt_reference_deg"].to_numpy(float))
    ax_dyn = df["ax_m_s2"].to_numpy(float) - G0 * np.sin(tilt)
    ay_dyn = df["ay_m_s2"].to_numpy(float)
    az_dyn = df["az_m_s2"].to_numpy(float) - G0 * np.cos(tilt)
    dynamic_mag = np.sqrt(ax_dyn**2 + ay_dyn**2 + az_dyn**2)
    acc_mag = np.sqrt(
        df["ax_m_s2"].to_numpy(float)**2 +
        df["ay_m_s2"].to_numpy(float)**2 +
        df["az_m_s2"].to_numpy(float)**2
    )
    gx = df["gx_deg_s"].to_numpy(float)
    gy = df["gy_deg_s"].to_numpy(float)
    gz = df["gz_deg_s"].to_numpy(float)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    time_min = df["time_min"].to_numpy(float)
    tilt_deg = df["tilt_reference_deg"].to_numpy(float)
    state = df["state"].to_numpy()

    window_id = np.arange(len(df), dtype=np.int64) // samples
    n_windows = int(window_id[-1]) + 1 if len(window_id) else 0
    counts = np.bincount(window_id, minlength=n_windows).astype(float)

    def mean_by_window(values):
        return np.bincount(window_id, weights=values, minlength=n_windows) / counts

    time_mean = mean_by_window(time_min)
    acc_mean_g = mean_by_window(acc_mag) / G0
    dynamic_mean_sq = mean_by_window(dynamic_mag**2)
    dynamic_mean = mean_by_window(dynamic_mag)
    dynamic_var = np.maximum(dynamic_mean_sq - dynamic_mean**2, 0.0)
    gyro_rms = np.sqrt(mean_by_window(gyro_mag**2))
    tilt_mean = mean_by_window(tilt_deg)

    # State boundaries in this simulator align with the 10-second windows.
    # Use the first sample per window; validate this invariant explicitly.
    first_idx = np.r_[0, np.flatnonzero(np.diff(window_id)) + 1] if len(df) else np.array([], dtype=int)
    last_idx = np.r_[first_idx[1:] - 1, len(df) - 1] if len(df) else np.array([], dtype=int)
    window_states = state[first_idx] if len(first_idx) else np.array([], dtype=object)
    for left, right, state_name in zip(first_idx, last_idx, window_states):
        if not np.all(state[left:right + 1] == state_name):
            # Preserve the original mode behavior for any window crossing a state boundary.
            window_states[np.where(first_idx == left)[0][0]] = pd.Series(state[left:right + 1]).mode().iloc[0]

    return pd.DataFrame({
        "window_id": np.arange(n_windows, dtype=int),
        "time_min": time_mean,
        "state": window_states,
        "acc_magnitude_mean_g": acc_mean_g,
        "dynamic_rms_g": np.sqrt(dynamic_mean_sq) / G0,
        "dynamic_variance_m_s4": dynamic_var,
        "gyro_rms_deg_s": gyro_rms,
        "tilt_reference_mean_deg": tilt_mean,
    })

# ------------------------- M4-A3 ------------------------------

def save_m4_a2_outputs(features, outdir):
    """Save M4-A2 feature plots, per-state statistics and validation report."""
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    plot_specs = [
        ("dynamic_rms_g", "RMS gia tốc động (g)",
         "M4-A2 – RMS gia tốc động", "M4_A2_dynamic_rms.png"),
        ("dynamic_variance_m_s4", "Phương sai (m⁴/s⁴)",
         "M4-A2 – Phương sai độ lớn gia tốc động", "M4_A2_dynamic_variance.png"),
        ("gyro_rms_deg_s", "RMS vận tốc góc (°/s)",
         "M4-A2 – RMS vận tốc góc", "M4_A2_gyro_rms.png"),
        ("tilt_reference_mean_deg", "Góc nghiêng tham chiếu (độ)",
         "M4-A2 – Góc nghiêng tham chiếu", "M4_A2_tilt_reference.png"),
    ]
    for col, ylabel, title, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(features["time_min"], features[col], linewidth=1.1)
        ax.set(xlabel="Thời gian (phút)", ylabel=ylabel, title=title)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figdir / filename, dpi=170)
        plt.close(fig)

    stats = features.groupby("state", sort=False).agg(
        windows=("time_min", "count"),
        dynamic_rms_mean_g=("dynamic_rms_g", "mean"),
        dynamic_rms_std_g=("dynamic_rms_g", "std"),
        dynamic_variance_mean=("dynamic_variance_m_s4", "mean"),
        gyro_rms_mean_deg_s=("gyro_rms_deg_s", "mean"),
        tilt_mean_deg=("tilt_reference_mean_deg", "mean"),
    ).reset_index()
    stats.to_csv(outdir / "M4_A2_feature_statistics_by_state.csv",
                 index=False, encoding="utf-8-sig")

    numeric = features[[
        "acc_magnitude_mean_g", "dynamic_rms_g",
        "dynamic_variance_m_s4", "gyro_rms_deg_s",
        "tilt_reference_mean_deg"
    ]]
    checks = {
        "Feature CSV có dữ liệu": len(features) > 0,
        "Không có NaN/Inf": np.isfinite(numeric.to_numpy()).all(),
        "RMS gia tốc không âm": (features.dynamic_rms_g >= 0).all(),
        "Phương sai không âm": (features.dynamic_variance_m_s4 >= 0).all(),
        "RMS vận tốc góc không âm": (features.gyro_rms_deg_s >= 0).all(),
        "Đủ bốn trạng thái": set(features.state.unique()) ==
            {"STABLE", "WATER_ACCUMULATION", "DEFORMATION", "SLIDING"},
    }
    pd.DataFrame(
        [{"check": k, "result": "PASS" if v else "FAIL"}
         for k, v in checks.items()]
    ).to_csv(outdir / "M4_A2_validation.csv", index=False, encoding="utf-8-sig")
    return stats, checks

def summarize_states(features):
    return features.groupby("state", sort=False).agg(
        windows=("window_id","count"),
        dynamic_rms_g_mean=("dynamic_rms_g","mean"),
        dynamic_rms_g_max=("dynamic_rms_g","max"),
        dynamic_variance_mean=("dynamic_variance_m_s4","mean"),
        gyro_rms_mean_deg_s=("gyro_rms_deg_s","mean"),
        tilt_mean_deg=("tilt_reference_mean_deg","mean"),
    ).reset_index()


def analyze_state_separation(features, outdir):
    """M4-A3: descriptive comparison of motion features across simulated states."""
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # Boxplots reveal within-state spread and overlap, not just means.
    specs = [
        ("dynamic_rms_g", "RMS gia tốc động (g)", "M4_A3_state_boxplot_rms.png"),
        ("dynamic_variance_m_s4", "Phương sai (m⁴/s⁴)", "M4_A3_state_boxplot_variance.png"),
        ("gyro_rms_deg_s", "RMS vận tốc góc (°/s)", "M4_A3_state_boxplot_gyro.png"),
        ("tilt_reference_mean_deg", "Góc nghiêng tham chiếu (độ)", "M4_A3_state_boxplot_tilt.png"),
    ]
    order = ["STABLE", "WATER_ACCUMULATION", "DEFORMATION", "SLIDING"]
    for col, ylabel, filename in specs:
        groups = [features.loc[features.state == s, col].dropna().to_numpy() for s in order]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.boxplot(groups, tick_labels=order, showfliers=False)
        ax.set_ylabel(ylabel)
        ax.set_title(f"M4-A3 – So sánh {col} giữa các trạng thái")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figdir / filename, dpi=170)
        plt.close(fig)

    # Pairwise overlap check using min/max ranges; descriptive only.
    rows = []
    for col, _, _ in specs:
        ranges = {}
        for s in order:
            vals = features.loc[features.state == s, col].to_numpy(float)
            ranges[s] = (float(np.min(vals)), float(np.max(vals)))
        for i, s1 in enumerate(order):
            for s2 in order[i+1:]:
                lo = max(ranges[s1][0], ranges[s2][0])
                hi = min(ranges[s1][1], ranges[s2][1])
                overlap = max(0.0, hi-lo)
                rows.append({
                    "feature": col, "state_1": s1, "state_2": s2,
                    "range_1_min": ranges[s1][0], "range_1_max": ranges[s1][1],
                    "range_2_min": ranges[s2][0], "range_2_max": ranges[s2][1],
                    "ranges_overlap": bool(overlap > 0),
                    "overlap_width": overlap
                })
    overlap_df = pd.DataFrame(rows)
    overlap_df.to_csv(outdir/"M4_A3_state_range_overlap.csv",
                      index=False, encoding="utf-8-sig")

    # State means and median/IQR to support robust descriptive interpretation.
    stats = features.groupby("state", sort=False).agg(
        windows=("window_id", "count"),
        rms_mean_g=("dynamic_rms_g", "mean"),
        rms_median_g=("dynamic_rms_g", "median"),
        rms_q25_g=("dynamic_rms_g", lambda x: x.quantile(.25)),
        rms_q75_g=("dynamic_rms_g", lambda x: x.quantile(.75)),
        variance_mean=("dynamic_variance_m_s4", "mean"),
        gyro_rms_mean_deg_s=("gyro_rms_deg_s", "mean"),
        tilt_mean_deg=("tilt_reference_mean_deg", "mean"),
    ).reset_index()
    stats.to_csv(outdir/"M4_A3_state_separation_summary.csv",
                 index=False, encoding="utf-8-sig")
    return stats, overlap_df


# ------------------------- M4-A4 ------------------------------
def validate_simulation(raw, features, summary, outdir=None, seed=SEED,
                        fs_hz=FS_HZ, total_min=TOTAL_MIN):
    """
    M4-A4: mathematical plausibility checks for the synthetic motion model.
    These checks verify internal consistency of the simulator only;
    they do not validate a physical landslide or calibrate a real MPU6050.
    """
    checks = []
    numeric_cols = ["ax_m_s2", "ay_m_s2", "az_m_s2",
                    "gx_deg_s", "gy_deg_s", "gz_deg_s"]
    checks.append(("RAW có đủ 6 trục", all(c in raw for c in
        ["ax_raw", "ay_raw", "az_raw", "gx_raw", "gy_raw", "gz_raw"])))
    checks.append(("Không có NaN/Inf ở tín hiệu chính",
        np.isfinite(raw[numeric_cols].to_numpy()).all()))
    checks.append(("Thời gian tăng đơn điệu", np.all(np.diff(raw.time_sec) > 0)))
    checks.append(("Đủ 4 trạng thái", set(raw.state.unique()) ==
        {"STABLE", "WATER_ACCUMULATION", "DEFORMATION", "SLIDING"}))

    # 1) At rest, accelerometer magnitude should be close to gravity.
    # Use the first 9 minutes to avoid the initial transition at minute 10.
    stationary = raw[(raw.time_min >= 1) & (raw.time_min < 9)]
    acc_norm = np.sqrt(stationary.ax_m_s2**2 +
                       stationary.ay_m_s2**2 +
                       stationary.az_m_s2**2)
    stationary_mean_error = float(abs(acc_norm.mean() - G0))
    checks.append(("Gia tốc khi ổn định gần 1g (sai lệch trung bình < 0.03 m/s²)",
                   stationary_mean_error < 0.03))

    # 2) Window-averaged gravity components should follow the reference tilt.
    # This suppresses the zero-mean synthetic vibration while retaining tilt.
    d = raw.copy()
    d["window_id"] = (np.arange(len(d)) // max(1, int(fs_hz * WINDOW_SEC)))
    grav_rows = []
    for wid, g in d.groupby("window_id", sort=True):
        tilt_rad = np.deg2rad(g.tilt_reference_deg.mean())
        exp_ax = G0 * np.sin(tilt_rad)
        exp_az = G0 * np.cos(tilt_rad)
        grav_rows.append((
            float(g.ax_m_s2.mean()), exp_ax,
            float(g.az_m_s2.mean()), exp_az
        ))
    grav = np.asarray(grav_rows)
    ax_mae = float(np.mean(np.abs(grav[:, 0] - grav[:, 1])))
    az_mae = float(np.mean(np.abs(grav[:, 2] - grav[:, 3])))
    checks.append(("Thành phần ax phù hợp phép chiếu trọng lực (MAE < 0.08 m/s²)",
                   ax_mae < 0.08))
    checks.append(("Thành phần az phù hợp phép chiếu trọng lực (MAE < 0.08 m/s²)",
                   az_mae < 0.08))

    # 3) Motion features should increase with the designed motion amplitude.
    means = summary.set_index("state").dynamic_rms_g_mean
    ordered = (means["STABLE"] < means["WATER_ACCUMULATION"] <
               means["DEFORMATION"] < means["SLIDING"])
    checks.append(("RMS gia tốc động tăng theo mức chuyển động giả lập", bool(ordered)))
    checks.append(("RMS và phương sai không âm",
        bool((features.dynamic_rms_g >= 0).all() and
             (features.dynamic_variance_m_s4 >= 0).all())))

    # 4) Compare 10-second mean gyroscope Y with the derivative of reference tilt.
    # Use window centers and avoid edge windows; the 10-s mean suppresses
    # the deliberately added zero-mean 2-Hz angular oscillation.
    gyro_rows = []
    for wid, g in d.groupby("window_id", sort=True):
        gyro_rows.append((float(g.time_sec.mean()),
                          float(g.gy_deg_s.mean()),
                          float(g.tilt_reference_deg.mean())))
    gyro_arr = np.asarray(gyro_rows)
    t_centers, gy_mean, tilt_mean = gyro_arr.T
    tilt_rate_window = np.gradient(tilt_mean, t_centers)
    valid = np.ones(len(t_centers), dtype=bool)
    if len(valid) > 4:
        valid[:2] = False
        valid[-2:] = False
    gyro_rmse = float(np.sqrt(np.mean(
        (gy_mean[valid] - tilt_rate_window[valid])**2
    )))
    checks.append(("Gyroscope Y nhất quán với tốc độ đổi góc nghiêng (RMSE < 0.02 °/s)",
                   gyro_rmse < 0.02))

    # 5) Repeat the simulator with several noise seeds. The absolute values
    # may vary slightly, but the intended ordering should remain unchanged.
    seed_rows = []
    seed_order_pass = True
    for test_seed in [1, 7, 42, 123, 2026]:
        seed_raw = simulate_raw_signals(fs_hz, total_min, test_seed)
        seed_features = extract_features(seed_raw, fs_hz, WINDOW_SEC)
        seed_summary = summarize_states(seed_features).set_index("state")
        vals = seed_summary.dynamic_rms_g_mean
        ok = (vals["STABLE"] < vals["WATER_ACCUMULATION"] <
              vals["DEFORMATION"] < vals["SLIDING"])
        seed_order_pass = seed_order_pass and bool(ok)
        seed_rows.append({
            "seed": test_seed,
            "stable_rms_g": vals["STABLE"],
            "water_rms_g": vals["WATER_ACCUMULATION"],
            "deformation_rms_g": vals["DEFORMATION"],
            "sliding_rms_g": vals["SLIDING"],
            "expected_order_pass": bool(ok),
        })
    checks.append(("Kết luận thứ tự RMS ổn định qua 5 random seed",
                   bool(seed_order_pass)))

    if outdir is not None:
        pd.DataFrame(seed_rows).to_csv(
            outdir / "M4_A4_seed_stability.csv", index=False, encoding="utf-8-sig"
        )
        metrics = {
            "stationary_acc_norm_mean_m_s2": float(acc_norm.mean()),
            "stationary_acc_norm_mean_abs_error_m_s2": stationary_mean_error,
            "gravity_ax_mae_m_s2": ax_mae,
            "gravity_az_mae_m_s2": az_mae,
            "gyro_tilt_rate_rmse_deg_s": gyro_rmse,
            "seed_count": len(seed_rows),
        }
        pd.DataFrame([{"metric": k, "value": v} for k, v in metrics.items()]).to_csv(
            outdir / "M4_A4_physical_consistency_metrics.csv",
            index=False, encoding="utf-8-sig"
        )

        # Diagnostic plot: window-mean gyroscope and reference tilt rate.
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(t_centers / 60.0, gy_mean, label="Gyroscope Y trung bình 10 s")
        ax.plot(t_centers / 60.0, tilt_rate_window,
                label="Đạo hàm góc nghiêng tham chiếu")
        ax.set(xlabel="Thời gian (phút)", ylabel="°/s",
               title="M4-A4 – Đối chiếu gyroscope Y và tốc độ đổi góc nghiêng")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / "figures" / "M4_A4_gyro_tilt_rate_check.png", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(t_centers / 60.0, grav[:, 0], label="ax trung bình 10 s")
        ax.plot(t_centers / 60.0, grav[:, 1], "--", label="ax trọng lực kỳ vọng")
        ax.plot(t_centers / 60.0, grav[:, 2], label="az trung bình 10 s")
        ax.plot(t_centers / 60.0, grav[:, 3], "--", label="az trọng lực kỳ vọng")
        ax.set(xlabel="Thời gian (phút)", ylabel="m/s²",
               title="M4-A4 – Kiểm tra phép chiếu trọng lực theo góc nghiêng")
        ax.legend(ncol=2)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / "figures" / "M4_A4_gravity_projection_check.png", dpi=170)
        plt.close(fig)

        pd.DataFrame(checks, columns=["check", "pass"]).assign(
            pass_text=lambda x: x["pass"].map({True: "PASS", False: "FAIL"})
        ).drop(columns=["pass"]).to_csv(
            outdir / "M4_A4_validation.csv", index=False, encoding="utf-8-sig"
        )
    return checks


# ------------------------- M4-B1 ------------------------------
def run_m4_b1_amplitude_sensitivity(outdir, fs_hz=FS_HZ,
                                    total_min=TOTAL_MIN, seed=SEED):
    """
    One-factor-at-a-time sensitivity of deformation/sliding vibration amplitudes.
    Only synthetic motion amplitudes are changed; state schedule and all other
    parameters remain fixed. Results are simulation sensitivity, not field limits.
    """
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    rows = []
    scenario_id = 0

    experiments = [
        ("DEFORMATION", 0.025, "deformation_amplitude"),
        ("SLIDING", 0.18, "sliding_amplitude"),
    ]

    for state_name, base_amp, parameter_name in experiments:
        for factor in factors:
            scenario_id += 1
            overrides = {state_name: base_amp * factor}
            raw_s = simulate_raw_signals(
                fs_hz, total_min, seed, amplitude_overrides=overrides
            )
            feat_s = extract_features(raw_s, fs_hz, WINDOW_SEC)
            summary_s = summarize_states(feat_s).set_index("state")

            for target_state in ["STABLE", "WATER_ACCUMULATION",
                                 "DEFORMATION", "SLIDING"]:
                r = summary_s.loc[target_state]
                rows.append({
                    "scenario_id": scenario_id,
                    "changed_parameter": parameter_name,
                    "changed_state": state_name,
                    "amplitude_factor": factor,
                    "amplitude_m_s2": base_amp * factor,
                    "measured_state": target_state,
                    "dynamic_rms_g_mean": r["dynamic_rms_g_mean"],
                    "dynamic_variance_mean_m_s4": r["dynamic_variance_mean"],
                    "gyro_rms_mean_deg_s": r["gyro_rms_mean_deg_s"],
                    "tilt_mean_deg": r["tilt_mean_deg"],
                })

    result = pd.DataFrame(rows)
    result.to_csv(outdir / "M4_B1_amplitude_sensitivity.csv",
                  index=False, encoding="utf-8-sig")

    # Relative change against factor=1 for the state whose amplitude is changed.
    baseline = result[result.amplitude_factor == 1.0].set_index(
        ["changed_parameter", "measured_state"]
    )
    rel_rows = []
    for _, r in result.iterrows():
        base = baseline.loc[(r.changed_parameter, r.measured_state)]
        rel_rows.append({
            "changed_parameter": r.changed_parameter,
            "changed_state": r.changed_state,
            "amplitude_factor": r.amplitude_factor,
            "measured_state": r.measured_state,
            "rms_change_percent": (
                (r.dynamic_rms_g_mean / base.dynamic_rms_g_mean - 1) * 100
                if base.dynamic_rms_g_mean != 0 else np.nan
            ),
            "variance_change_percent": (
                (r.dynamic_variance_mean_m_s4 /
                 base.dynamic_variance_mean_m_s4 - 1) * 100
                if base.dynamic_variance_mean_m_s4 != 0 else np.nan
            ),
            "gyro_rms_change_percent": (
                (r.gyro_rms_mean_deg_s / base.gyro_rms_mean_deg_s - 1) * 100
                if base.gyro_rms_mean_deg_s != 0 else np.nan
            ),
        })
    relative = pd.DataFrame(rel_rows)
    relative.to_csv(outdir / "M4_B1_relative_change.csv",
                    index=False, encoding="utf-8-sig")

    # Plot RMS response for the directly modified state.
    for parameter_name, state_name in [
        ("deformation_amplitude", "DEFORMATION"),
        ("sliding_amplitude", "SLIDING"),
    ]:
        sub = result[(result.changed_parameter == parameter_name) &
                     (result.measured_state == state_name)]
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.plot(sub.amplitude_factor, sub.dynamic_rms_g_mean, marker="o")
        ax.set(xlabel="Hệ số nhân biên độ so với cơ sở",
               ylabel="RMS gia tốc động trung bình (g)",
               title=f"M4-B1 – Độ nhạy RMS tại trạng thái {state_name}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figdir / f"M4_B1_{state_name.lower()}_rms_sensitivity.png",
                    dpi=170)
        plt.close(fig)

    # Compact sensitivity report: compare endpoints and monotonicity.
    report_rows = []
    for parameter_name, state_name in [
        ("deformation_amplitude", "DEFORMATION"),
        ("sliding_amplitude", "SLIDING"),
    ]:
        sub = result[(result.changed_parameter == parameter_name) &
                     (result.measured_state == state_name)].sort_values("amplitude_factor")
        rms_values = sub.dynamic_rms_g_mean.to_numpy()
        variance_values = sub.dynamic_variance_mean_m_s4.to_numpy()
        report_rows.append({
            "parameter": parameter_name,
            "tested_factors": ", ".join(map(str, factors)),
            "rms_monotonic_increase": bool(np.all(np.diff(rms_values) > 0)),
            "variance_monotonic_increase": bool(np.all(np.diff(variance_values) > 0)),
            "rms_at_0_5_g": float(rms_values[0]),
            "rms_at_1_0_g": float(sub.loc[sub.amplitude_factor == 1.0,
                                          "dynamic_rms_g_mean"].iloc[0]),
            "rms_at_2_0_g": float(rms_values[-1]),
            "rms_change_0_5_to_2_0_percent":
                float((rms_values[-1] / rms_values[0] - 1) * 100),
        })
    report = pd.DataFrame(report_rows)
    report.to_csv(outdir / "M4_B1_validation_summary.csv",
                  index=False, encoding="utf-8-sig")
    return result, relative, report

# -------------------------- M4-B -------------------------------
def run_sensitivity_placeholder():
    # Reserved for controlled changes to motion amplitudes and state durations.
    # Implement only after A1-A4 signal generation is accepted.
    return "M4-B chưa chạy: cần khóa M4-A1 đến M4-A4 trước."


# ------------------------- M4-B2 ------------------------------
def run_m4_b2_timing_sensitivity(outdir, fs_hz=FS_HZ,
                                 total_min=TOTAL_MIN, seed=SEED):
    """
    OFAT sensitivity to state transition timing.
    Motion amplitudes remain at A1 baseline; only timing parameters change.
    """
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    baseline_timing = {
        "deformation_start_min": 180.0,
        "sliding_start_min": 230.0,
        "deformation_ramp_min": 8.0,
        "sliding_ramp_min": 1.5,
    }
    experiments = [
        ("deformation_start_min", [150, 165, 180, 195, 210],
         "Thời điểm bắt đầu biến dạng (phút)"),
        ("sliding_start_min", [220, 225, 230, 235],
         "Thời điểm bắt đầu trượt (phút)"),
        ("deformation_ramp_min", [4, 6, 8, 12, 16],
         "Thời gian chuyển tiếp sang biến dạng (phút)"),
        ("sliding_ramp_min", [0.75, 1.0, 1.5, 2.0, 3.0],
         "Thời gian chuyển tiếp sang trượt (phút)"),
    ]

    rows = []
    scenario_id = 0
    for parameter, values, label in experiments:
        for value in values:
            timing = dict(baseline_timing)
            timing[parameter] = float(value)
            # Keep phase order physically meaningful in this synthetic schedule.
            if timing["deformation_start_min"] >= timing["sliding_start_min"]:
                continue
            scenario_id += 1
            raw_s = simulate_raw_signals(
                fs_hz, total_min, seed,
                timing_overrides=timing
            )
            feat_s = extract_features(raw_s, fs_hz, WINDOW_SEC)
            summary_s = summarize_states(feat_s)

            for _, r in summary_s.iterrows():
                rows.append({
                    "scenario_id": scenario_id,
                    "changed_parameter": parameter,
                    "parameter_label": label,
                    "parameter_value": value,
                    "state": r["state"],
                    "windows": int(r["windows"]),
                    "dynamic_rms_g_mean": r["dynamic_rms_g_mean"],
                    "dynamic_variance_mean_m_s4": r["dynamic_variance_mean"],
                    "gyro_rms_mean_deg_s": r["gyro_rms_mean_deg_s"],
                    "tilt_mean_deg": r["tilt_mean_deg"],
                    "deformation_duration_min":
                        timing["sliding_start_min"] - timing["deformation_start_min"],
                    "sliding_duration_min":
                        total_min - timing["sliding_start_min"],
                })

    result = pd.DataFrame(rows)
    result.to_csv(outdir / "M4_B2_timing_sensitivity.csv",
                  index=False, encoding="utf-8-sig")

    # Extract the principal response for each timing parameter:
    # changing deformation start -> deformation; sliding start -> sliding;
    # ramp widths -> corresponding movement state.
    target_state = {
        "deformation_start_min": "DEFORMATION",
        "sliding_start_min": "SLIDING",
        "deformation_ramp_min": "DEFORMATION",
        "sliding_ramp_min": "SLIDING",
    }
    response_rows = []
    for parameter, values, label in experiments:
        sub = result[(result.changed_parameter == parameter) &
                     (result.state == target_state[parameter])].sort_values(
                         "parameter_value"
                     )
        for _, r in sub.iterrows():
            response_rows.append({
                "parameter": parameter,
                "parameter_label": label,
                "parameter_value": r["parameter_value"],
                "target_state": r["state"],
                "state_duration_min": (
                    r["deformation_duration_min"] if r["state"] == "DEFORMATION"
                    else r["sliding_duration_min"]
                ),
                "windows": r["windows"],
                "dynamic_rms_g_mean": r["dynamic_rms_g_mean"],
                "dynamic_variance_mean_m_s4": r["dynamic_variance_mean_m_s4"],
                "gyro_rms_mean_deg_s": r["gyro_rms_mean_deg_s"],
                "tilt_mean_deg": r["tilt_mean_deg"],
            })
    response = pd.DataFrame(response_rows)
    response.to_csv(outdir / "M4_B2_target_state_response.csv",
                    index=False, encoding="utf-8-sig")

    # Plot target-state RMS against each tested timing parameter.
    for parameter, _, label in experiments:
        sub = response[response.parameter == parameter]
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.plot(sub.parameter_value, sub.dynamic_rms_g_mean, marker="o")
        ax.set(xlabel=label, ylabel="RMS gia tốc động trung bình (g)",
               title=f"M4-B2 – Độ nhạy RMS theo {parameter}")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figdir / f"M4_B2_{parameter}_rms.png", dpi=170)
        plt.close(fig)

    # Checks: complete experiment set, positive durations, all finite, and
    # expected window counts agree with the configured phase durations.
    checks = {
        "Đã chạy đủ 19 kịch bản thời gian": scenario_id == 19,
        "Không có NaN/Inf ở kết quả": np.isfinite(
            result[["dynamic_rms_g_mean", "dynamic_variance_mean_m_s4",
                    "gyro_rms_mean_deg_s", "tilt_mean_deg"]].to_numpy()
        ).all(),
        "Số cửa sổ mỗi trạng thái đều dương": (result.windows > 0).all(),
        "Thời lượng biến dạng và trượt dương": (
            (result.deformation_duration_min > 0).all() and
            (result.sliding_duration_min > 0).all()
        ),
    }
    pd.DataFrame(
        [{"check": k, "result": "PASS" if v else "FAIL"} for k, v in checks.items()]
    ).to_csv(outdir / "M4_B2_validation.csv", index=False, encoding="utf-8-sig")
    return result, response, checks


# ------------------------- M4-B3 ------------------------------
def run_m4_b3_parameter_synthesis(outdir):
    """
    Synthesize B1/B2 OFAT results into sensitivity rankings and illustrative
    parameter sets. These sets are simulation scenarios, not calibrated limits.
    """
    b1 = pd.read_csv(outdir / "M4_B1_amplitude_sensitivity.csv")
    b2 = pd.read_csv(outdir / "M4_B2_target_state_response.csv")

    # B1: sensitivity span of the target state's RMS and variance.
    b1_rows = []
    for parameter, group in b1.groupby("changed_parameter", sort=False):
        state = group.changed_state.iloc[0]
        target = group[group.measured_state == state].sort_values("amplitude_factor")
        rms_min, rms_max = target.dynamic_rms_g_mean.min(), target.dynamic_rms_g_mean.max()
        var_min, var_max = (target.dynamic_variance_mean_m_s4.min(),
                            target.dynamic_variance_mean_m_s4.max())
        b1_rows.append({
            "parameter": parameter,
            "parameter_type": "motion_amplitude",
            "target_state": state,
            "tested_range": f'{target.amplitude_factor.min()}–{target.amplitude_factor.max()} × baseline',
            "rms_min_g": rms_min,
            "rms_max_g": rms_max,
            "rms_range_percent_of_min": (rms_max / rms_min - 1) * 100,
            "variance_min_m_s4": var_min,
            "variance_max_m_s4": var_max,
            "variance_range_percent_of_min": (var_max / var_min - 1) * 100,
            "interpretation": "Ảnh hưởng trực tiếp đến cường độ dao động trong trạng thái",
        })

    # B2: range of target-state metrics across tested timing values.
    b2_rows = []
    for parameter, group in b2.groupby("parameter", sort=False):
        metric_ranges = {}
        for col in ["dynamic_rms_g_mean", "dynamic_variance_mean_m_s4",
                    "gyro_rms_mean_deg_s", "tilt_mean_deg"]:
            vmin, vmax = group[col].min(), group[col].max()
            metric_ranges[col] = {
                "min": float(vmin), "max": float(vmax),
                "range_percent_of_min": (
                    float((vmax / vmin - 1) * 100) if vmin != 0 else np.nan
                ),
            }
        b2_rows.append({
            "parameter": parameter,
            "parameter_type": "transition_timing",
            "target_state": group.target_state.iloc[0],
            "tested_range": f'{group.parameter_value.min()}–{group.parameter_value.max()}',
            "rms_min_g": metric_ranges["dynamic_rms_g_mean"]["min"],
            "rms_max_g": metric_ranges["dynamic_rms_g_mean"]["max"],
            "rms_range_percent_of_min":
                metric_ranges["dynamic_rms_g_mean"]["range_percent_of_min"],
            "variance_min_m_s4":
                metric_ranges["dynamic_variance_mean_m_s4"]["min"],
            "variance_max_m_s4":
                metric_ranges["dynamic_variance_mean_m_s4"]["max"],
            "variance_range_percent_of_min":
                metric_ranges["dynamic_variance_mean_m_s4"]["range_percent_of_min"],
            "gyro_rms_range_percent_of_min":
                metric_ranges["gyro_rms_mean_deg_s"]["range_percent_of_min"],
            "tilt_range_percent_of_min":
                metric_ranges["tilt_mean_deg"]["range_percent_of_min"],
            "interpretation": (
                "Chủ yếu ảnh hưởng lịch xuất hiện/thời lượng; RMS theo trạng thái "
                "gần như không đổi trong cấu trúc hiện tại"
            ),
        })

    sensitivity = pd.DataFrame(b1_rows + b2_rows)
    sensitivity.to_csv(outdir / "M4_B3_sensitivity_synthesis.csv",
                       index=False, encoding="utf-8-sig")

    # Create representative parameter sets for downstream simulation.
    # Baseline values are those used in A1; alternatives are controlled
    # scenarios chosen to span low/central/high motion and early/late timing.
    candidate_sets = pd.DataFrame([
        {
            "set_id": "P1_BASELINE",
            "purpose": "Bộ cơ sở để đối chiếu",
            "deformation_amplitude_m_s2": 0.025,
            "sliding_amplitude_m_s2": 0.180,
            "deformation_start_min": 180,
            "sliding_start_min": 230,
            "deformation_ramp_min": 8,
            "sliding_ramp_min": 1.5,
            "status": "Giữ nguyên giả định A1",
        },
        {
            "set_id": "P2_LOW_MOTION",
            "purpose": "Kịch bản chuyển động yếu",
            "deformation_amplitude_m_s2": 0.025 * 0.75,
            "sliding_amplitude_m_s2": 0.180 * 0.75,
            "deformation_start_min": 180,
            "sliding_start_min": 230,
            "deformation_ramp_min": 8,
            "sliding_ramp_min": 1.5,
            "status": "Kịch bản giả lập B1, chưa hiệu chuẩn",
        },
        {
            "set_id": "P3_HIGH_MOTION",
            "purpose": "Kịch bản chuyển động mạnh",
            "deformation_amplitude_m_s2": 0.025 * 1.5,
            "sliding_amplitude_m_s2": 0.180 * 1.5,
            "deformation_start_min": 180,
            "sliding_start_min": 230,
            "deformation_ramp_min": 8,
            "sliding_ramp_min": 1.5,
            "status": "Kịch bản giả lập B1, chưa hiệu chuẩn",
        },
        {
            "set_id": "P4_EARLY_TRANSITION",
            "purpose": "Kịch bản chuyển trạng thái sớm",
            "deformation_amplitude_m_s2": 0.025,
            "sliding_amplitude_m_s2": 0.180,
            "deformation_start_min": 165,
            "sliding_start_min": 225,
            "deformation_ramp_min": 6,
            "sliding_ramp_min": 1.0,
            "status": "Kịch bản giả lập B2, chưa hiệu chuẩn",
        },
        {
            "set_id": "P5_LATE_TRANSITION",
            "purpose": "Kịch bản chuyển trạng thái muộn",
            "deformation_amplitude_m_s2": 0.025,
            "sliding_amplitude_m_s2": 0.180,
            "deformation_start_min": 195,
            "sliding_start_min": 235,
            "deformation_ramp_min": 12,
            "sliding_ramp_min": 2.0,
            "status": "Kịch bản giả lập B2, chưa hiệu chuẩn",
        },
    ])
    candidate_sets.to_csv(outdir / "M4_B3_candidate_parameter_sets.csv",
                          index=False, encoding="utf-8-sig")

    # Summary conclusions and checks.
    b1_ok = all(x["rms_range_percent_of_min"] > 0 for x in b1_rows)
    b2_rms_small = all(x["rms_range_percent_of_min"] < 2 for x in b2_rows)
    checks = {
        "Đã tổng hợp đủ 2 tham số biên độ từ B1": len(b1_rows) == 2,
        "Đã tổng hợp đủ 4 tham số thời gian từ B2": len(b2_rows) == 4,
        "B1 cho thấy RMS tăng theo biên độ": bool(b1_ok),
        "B2 cho thấy RMS theo trạng thái gần như không đổi (<2%)":
            bool(b2_rms_small),
        "Có 5 bộ tham số đại diện": len(candidate_sets) == 5,
    }
    pd.DataFrame([
        {"check": k, "result": "PASS" if v else "FAIL"}
        for k, v in checks.items()
    ]).to_csv(outdir / "M4_B3_validation.csv", index=False, encoding="utf-8-sig")

    # Plot relative sensitivity ranges by parameter.
    plot_df = sensitivity[["parameter", "parameter_type",
                           "rms_range_percent_of_min"]].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df.parameter, plot_df.rms_range_percent_of_min)
    ax.set_ylabel("Biên độ thay đổi RMS (% so với giá trị nhỏ nhất)")
    ax.set_title("M4-B3 – So sánh độ nhạy RMS giữa các tham số")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "M4_B3_rms_sensitivity_comparison.png",
                dpi=170)
    plt.close(fig)

    return sensitivity, candidate_sets, checks


# ------------------------- M4-C1 ------------------------------
def connect_module3_module4(module3_csv, motion_features_csv, outdir):
    """
    C1: synchronize Module 3 pore-pressure/moisture simulation with
    Module 4 motion features on a shared time axis.

    Module 4 motion is currently independent of Module 3 pressure.
    This is a data-alignment/export step only; no causal coupling is imposed.
    """
    module3_csv = Path(module3_csv)
    motion_features_csv = Path(motion_features_csv)
    if not module3_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu Module 3: {module3_csv}")
    if not motion_features_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy đặc trưng Module 4: {motion_features_csv}")

    m3 = pd.read_csv(module3_csv)
    m4 = pd.read_csv(motion_features_csv)

    required_m3 = [
        "scenario_id", "time_min", "moisture_percent_vwc",
        "delayed_moisture_percent",
        "pore_pressure_positive_kpa_synthetic",
        "wetness_drive_0_1", "rainfall_mm_h",
        "cumulative_rainfall_mm"
    ]
    required_m4 = [
        "time_min", "state", "acc_magnitude_mean_g", "dynamic_rms_g",
        "dynamic_variance_m_s4", "gyro_rms_deg_s",
        "tilt_reference_mean_deg"
    ]
    missing_m3 = [c for c in required_m3 if c not in m3.columns]
    missing_m4 = [c for c in required_m4 if c not in m4.columns]
    if missing_m3 or missing_m4:
        raise ValueError(
            f"Thiếu cột Module 3: {missing_m3}; thiếu cột Module 4: {missing_m4}"
        )

    m4 = m4.sort_values("time_min").drop_duplicates("time_min")
    if not np.all(np.diff(m4.time_min.to_numpy(float)) > 0):
        raise ValueError("Trục thời gian Module 4 phải tăng đơn điệu.")

    numeric_m4 = [
        "acc_magnitude_mean_g", "dynamic_rms_g",
        "dynamic_variance_m_s4", "gyro_rms_deg_s",
        "tilt_reference_mean_deg"
    ]
    if not np.isfinite(m4[numeric_m4].to_numpy(float)).all():
        raise ValueError("Module 4 có NaN/Inf trong các đặc trưng số.")

    # Interpolate M4 features at exact Module 3 minute timestamps.
    # No extrapolation: retain only the common time interval.
    joined_parts = []
    summary_rows = []
    for scenario_id, g3 in m3.groupby("scenario_id", sort=True):
        g3 = g3.sort_values("time_min").drop_duplicates("time_min").copy()
        lo = max(float(g3.time_min.min()), float(m4.time_min.min()))
        hi = min(float(g3.time_min.max()), float(m4.time_min.max()))
        g3 = g3[(g3.time_min >= lo) & (g3.time_min <= hi)].copy()
        if g3.empty:
            continue

        t4 = m4.time_min.to_numpy(float)
        t3 = g3.time_min.to_numpy(float)
        joined = g3.copy()
        joined["motion_state"] = [
            m4.iloc[int(np.argmin(np.abs(t4 - t)))].state for t in t3
        ]
        for col in numeric_m4:
            joined[f"m4_{col}"] = np.interp(
                t3, t4, m4[col].to_numpy(float)
            )
        joined["time_alignment_method"] = "M4 10s-window features interpolated to M3 1min timestamps"
        joined["causal_coupling_applied"] = False
        joined_parts.append(joined)

        pcol = "pore_pressure_positive_kpa_synthetic"
        peak_value = float(g3[pcol].max())
        if peak_value > 0:
            peak_idx = g3[pcol].idxmax()
            peak_time = float(g3.loc[peak_idx, "time_min"])
            peak_joined = joined.loc[peak_idx]
            peak_state = str(peak_joined["motion_state"])
            peak_rms = float(peak_joined["m4_dynamic_rms_g"])
            peak_gyro = float(peak_joined["m4_gyro_rms_deg_s"])
        else:
            # No positive pressure event: do not assign an arbitrary peak time.
            peak_time = np.nan
            peak_state = "NO_POSITIVE_PRESSURE"
            peak_rms = np.nan
            peak_gyro = np.nan
        summary_rows.append({
            "scenario_id": scenario_id,
            "joined_rows": len(joined),
            "common_start_min": lo,
            "common_end_min": hi,
            "module3_pressure_peak_kpa": peak_value,
            "module3_pressure_peak_time_min": peak_time,
            "module3_moisture_max_percent": float(g3.moisture_percent_vwc.max()),
            "module3_rain_intensity_mm_h": float(g3.rain_intensity_setting_mm_h.iloc[0]),
            "module3_rain_duration_min": float(g3.rain_duration_min.iloc[0]),
            "module4_motion_state_at_pressure_peak": peak_state,
            "module4_rms_at_pressure_peak_g": peak_rms,
            "module4_gyro_rms_at_pressure_peak_deg_s": peak_gyro,
        })

    if not joined_parts:
        raise ValueError("Không tìm thấy khoảng thời gian chung giữa Module 3 và Module 4.")

    combined = pd.concat(joined_parts, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    combined.to_csv(outdir / "M4_C1_module3_module4_synchronized.csv",
                    index=False, encoding="utf-8-sig")
    summary.to_csv(outdir / "M4_C1_scenario_alignment_summary.csv",
                   index=False, encoding="utf-8-sig")

    checks = {
        "Đã ghép đủ 6 kịch bản Module 3": combined.scenario_id.nunique() == 6,
        "Không có NaN/Inf ở đặc trưng chuyển động đã ghép":
            np.isfinite(combined[[f"m4_{c}" for c in numeric_m4]].to_numpy()).all(),
        "Thời gian trong từng kịch bản tăng đơn điệu":
            all(np.all(np.diff(g.time_min.to_numpy(float)) > 0)
                for _, g in combined.groupby("scenario_id")),
        "Không ngoại suy ngoài khoảng thời gian chung":
            (combined.time_min.min() >= m4.time_min.min() and
             combined.time_min.max() <= m4.time_min.max()),
        "Không áp đặt quan hệ nhân quả áp lực-chuyển động":
            not combined.causal_coupling_applied.any(),
    }
    pd.DataFrame([
        {"check": k, "result": "PASS" if v else "FAIL"}
        for k, v in checks.items()
    ]).to_csv(outdir / "M4_C1_validation.csv", index=False, encoding="utf-8-sig")

    # Diagnostic plot: pressure and motion RMS for each scenario.
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=False)
    for ax, (scenario_id, g) in zip(axes.flat, combined.groupby("scenario_id", sort=True)):
        ax.plot(g.time_min, g.pore_pressure_positive_kpa_synthetic,
                label="Áp lực nước lỗ rỗng dương (kPa)")
        ax.set_title(str(scenario_id))
        ax.set_xlabel("Thời gian (phút)")
        ax.set_ylabel("Áp lực (kPa)")
        ax.grid(alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(g.time_min, g.m4_dynamic_rms_g, linestyle="--",
                 label="RMS gia tốc động (g)")
        ax2.set_ylabel("RMS (g)")
    fig.suptitle(
        "M4-C1 – Đồng bộ áp lực nước lỗ rỗng và RMS chuyển động\n"
        "Hai chuỗi hiện độc lập trong mô hình; không diễn giải nhân quả",
        y=1.02
    )
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "M4_C1_pressure_motion_alignment.png",
                dpi=170, bbox_inches="tight")
    plt.close(fig)

    return combined, summary, checks


# ------------------------- M4-C2 ------------------------------
# Hypothesis-based coupling: synthetic positive pore pressure drives a
# dimensionless motion activation index. These parameters are provisional.
C2_U_ONSET_KPA = 0.5
C2_U_FULL_KPA = 3.2
C2_RESPONSE_LAG_MIN = 2.0

def run_m4_c2_pressure_motion_coupling(module3_csv, outdir):
    """
    C2: create a provisional pressure-driven motion-response layer.
    This does NOT claim a calibrated constitutive law. It replaces the
    independent fixed movement timetable with a transparent hypothesis:
    motion activation increases when positive pore pressure exceeds u_onset.
    """
    module3_csv = Path(module3_csv)
    if not module3_csv.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu Module 3: {module3_csv}")
    m3 = pd.read_csv(module3_csv)
    required = ["scenario_id", "time_min",
                "pore_pressure_positive_kpa_synthetic",
                "moisture_percent_vwc", "rainfall_mm_h"]
    missing = [c for c in required if c not in m3.columns]
    if missing:
        raise ValueError(f"Thiếu cột Module 3: {missing}")

    parts, summaries = [], []
    for sid, g in m3.groupby("scenario_id", sort=True):
        g = g.sort_values("time_min").drop_duplicates("time_min").copy()
        t = g.time_min.to_numpy(float)
        u = np.maximum(g.pore_pressure_positive_kpa_synthetic.to_numpy(float), 0.0)

        # Normalize pressure to [0,1], then apply a first-order response lag.
        drive = np.clip(
            (u - C2_U_ONSET_KPA) / (C2_U_FULL_KPA - C2_U_ONSET_KPA),
            0.0, 1.0
        )
        activation = np.zeros_like(drive)
        for i in range(1, len(drive)):
            dt = max(t[i] - t[i-1], 0.0)
            alpha = 1.0 - np.exp(-dt / C2_RESPONSE_LAG_MIN)
            activation[i] = activation[i-1] + alpha * (drive[i] - activation[i-1])

        # Illustrative mappings to feature-scale outputs. The values are
        # based on M4 synthetic signal scales, not calibrated sensor thresholds.
        base_rms = 0.00052
        deform_rms_scale = 0.002445
        sliding_rms_scale = 0.017430
        rms = base_rms + deform_rms_scale*activation + sliding_rms_scale*activation**2
        gyro = 0.060 + 0.18*activation + 1.20*activation**2
        tilt = 0.15 + 4.0*activation + 8.0*activation**2

        motion_state = np.where(
            activation >= 0.75, "SLIDING_RESPONSE",
            np.where(activation >= 0.25, "DEFORMATION_RESPONSE",
                     np.where(activation > 0, "EARLY_RESPONSE", "NO_MOTION_RESPONSE"))
        )
        out = g.copy()
        out["c2_pressure_drive_0_1"] = drive
        out["c2_motion_activation_0_1"] = activation
        out["c2_motion_state"] = motion_state
        out["c2_dynamic_rms_g_synthetic"] = rms
        out["c2_gyro_rms_deg_s_synthetic"] = gyro
        out["c2_tilt_deg_synthetic"] = tilt
        out["c2_coupling_assumption"] = (
            f"u_onset={C2_U_ONSET_KPA}kPa; u_full={C2_U_FULL_KPA}kPa; "
            f"lag={C2_RESPONSE_LAG_MIN}min"
        )
        parts.append(out)

        peak_i = int(np.argmax(u))
        active = np.where(activation > 0)[0]
        summaries.append({
            "scenario_id": sid,
            "pressure_peak_kpa_synthetic": float(u[peak_i]),
            "pressure_peak_time_min": float(t[peak_i]) if u[peak_i] > 0 else np.nan,
            "max_motion_activation_0_1": float(activation.max()),
            "first_motion_response_time_min": float(t[active[0]]) if len(active) else np.nan,
            "max_dynamic_rms_g_synthetic": float(rms.max()),
            "max_gyro_rms_deg_s_synthetic": float(gyro.max()),
            "max_tilt_deg_synthetic": float(tilt.max()),
            "motion_state_at_pressure_peak": str(motion_state[peak_i]),
        })

    combined = pd.concat(parts, ignore_index=True)
    summary = pd.DataFrame(summaries)
    combined.to_csv(outdir/"M4_C2_pressure_driven_motion.csv",
                    index=False, encoding="utf-8-sig")
    summary.to_csv(outdir/"M4_C2_scenario_summary.csv",
                   index=False, encoding="utf-8-sig")

    # Scenario-level diagnostic plots.
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for ax, (sid, g) in zip(axes.flat, combined.groupby("scenario_id", sort=True)):
        ax.plot(g.time_min, g.pore_pressure_positive_kpa_synthetic,
                label="Áp lực dương (kPa)")
        ax.set_title(str(sid))
        ax.set_xlabel("Thời gian (phút)")
        ax.set_ylabel("Áp lực (kPa)")
        ax.grid(alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(g.time_min, g.c2_dynamic_rms_g_synthetic, "--",
                 label="RMS chuyển động giả lập (g)")
        ax2.set_ylabel("RMS (g)")
    fig.suptitle("M4-C2 – Mô hình giả thuyết áp lực nước lỗ rỗng → đáp ứng chuyển động")
    fig.tight_layout()
    fig.savefig(outdir/"figures"/"M4_C2_pressure_driven_motion.png", dpi=170)
    plt.close(fig)

    checks = {
        "Đã xử lý đủ 6 kịch bản": combined.scenario_id.nunique() == 6,
        "Chỉ số kích hoạt nằm trong [0,1]":
            combined.c2_motion_activation_0_1.between(0, 1).all(),
        "Không có NaN/Inf ở đầu ra ghép":
            np.isfinite(combined[["c2_pressure_drive_0_1",
                "c2_motion_activation_0_1", "c2_dynamic_rms_g_synthetic",
                "c2_gyro_rms_deg_s_synthetic", "c2_tilt_deg_synthetic"]].to_numpy()).all(),
        "Kịch bản không có áp lực dương không kích hoạt chuyển động":
            (combined.loc[combined.pore_pressure_positive_kpa_synthetic <= 0,
                          "c2_motion_activation_0_1"] == 0).all(),
        "RMS không giảm khi mức kích hoạt tăng":
            np.all(np.diff(combined.sort_values("c2_motion_activation_0_1")
                           .c2_dynamic_rms_g_synthetic.to_numpy()) >= -1e-12),
    }
    pd.DataFrame([{"check": k, "result": "PASS" if v else "FAIL"}
                  for k, v in checks.items()]).to_csv(
        outdir/"M4_C2_validation.csv", index=False, encoding="utf-8-sig")
    return combined, summary, checks


# ------------------------- M4-C3 ------------------------------
def run_m4_c3_coupling_diagnostics(c2_csv, outdir):
    """
    C3: diagnostics for the provisional C2 pressure-motion coupling.
    Tests scenario ordering, pressure/activation timing, and threshold robustness.
    These are internal model diagnostics, not physical validation.
    """
    df = pd.read_csv(c2_csv)
    required = ["scenario_id", "time_min",
                "pore_pressure_positive_kpa_synthetic",
                "c2_pressure_drive_0_1", "c2_motion_activation_0_1",
                "c2_dynamic_rms_g_synthetic"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột C2: {missing}")

    # 1) Per-scenario response timing and peak metrics.
    rows = []
    for sid, g in df.groupby("scenario_id", sort=True):
        g = g.sort_values("time_min")
        u = g.pore_pressure_positive_kpa_synthetic.to_numpy(float)
        a = g.c2_motion_activation_0_1.to_numpy(float)
        t = g.time_min.to_numpy(float)
        positive = np.flatnonzero(u > 0)
        active = np.flatnonzero(a > 0)
        peak_u_i = int(np.argmax(u))
        peak_a_i = int(np.argmax(a))
        rows.append({
            "scenario_id": sid,
            "pressure_peak_kpa": float(u[peak_u_i]),
            "pressure_peak_time_min": float(t[peak_u_i]) if u[peak_u_i] > 0 else np.nan,
            "first_positive_pressure_min": float(t[positive[0]]) if len(positive) else np.nan,
            "first_motion_response_min": float(t[active[0]]) if len(active) else np.nan,
            "response_after_first_pressure_min":
                float(t[active[0]]-t[positive[0]]) if len(active) and len(positive) else np.nan,
            "max_activation_0_1": float(a.max()),
            "activation_peak_time_min": float(t[peak_a_i]),
            "motion_rms_peak_g": float(g.c2_dynamic_rms_g_synthetic.max()),
        })
    timing = pd.DataFrame(rows)
    timing.to_csv(outdir/"M4_C3_timing_diagnostics.csv", index=False, encoding="utf-8-sig")

    # 2) Check expected rainfall ordering using the scenario labels.
    # Compare paired 10/20/30-min events at the same intensity.
    peak_map = timing.set_index("scenario_id").pressure_peak_kpa.to_dict()
    pair_checks = []
    for intensity, token in [("LOW20", "LOW_20mmh_"),
                             ("HIGH40", "HIGH_40mmh_")]:
        vals = []
        for duration in [10, 20, 30]:
            matches = [v for k, v in peak_map.items()
                       if str(k).startswith(token) and str(k).endswith(f"{duration}min")]
            vals.append(float(matches[0]) if matches else np.nan)
        monotonic = all(np.isfinite(vals)) and vals[0] <= vals[1] <= vals[2]
        pair_checks.append({"intensity": intensity,
                            "pressure_peak_10_20_30min_kpa": str(vals),
                            "peak_pressure_monotonic_with_duration": monotonic})

    # 3) Threshold robustness grid. This is a stress test of the assumed
    # trigger, not an optimization or calibration.
    sensitivity_rows = []
    for u_onset in [0.25, 0.5, 1.0, 1.5]:
        for u_full in [2.0, 3.2, 4.0]:
            if u_full <= u_onset: continue
            for sid, g in df.groupby("scenario_id", sort=True):
                g = g.sort_values("time_min")
                t = g.time_min.to_numpy(float)
                u = np.maximum(g.pore_pressure_positive_kpa_synthetic.to_numpy(float), 0)
                drive = np.clip((u-u_onset)/(u_full-u_onset),0,1)
                act = np.zeros_like(drive)
                for i in range(1,len(act)):
                    dt=max(t[i]-t[i-1],0)
                    alpha=1-np.exp(-dt/2.0)
                    act[i]=act[i-1]+alpha*(drive[i]-act[i-1])
                ix=np.flatnonzero(act>0)
                sensitivity_rows.append({
                    "scenario_id":sid,"u_onset_kpa":u_onset,"u_full_kpa":u_full,
                    "max_activation_0_1":float(act.max()),
                    "first_response_time_min":float(t[ix[0]]) if len(ix) else np.nan
                })
    sens=pd.DataFrame(sensitivity_rows)
    sens.to_csv(outdir/"M4_C3_threshold_robustness.csv",index=False,encoding="utf-8-sig")

    # 4) Automated diagnostics.
    checks = {
        "Có đủ 6 kịch bản": timing.scenario_id.nunique()==6,
        "Áp lực đỉnh tăng/không giảm theo thời lượng ở LOW20":
            bool(pair_checks[0]["peak_pressure_monotonic_with_duration"]),
        "Áp lực đỉnh tăng/không giảm theo thời lượng ở HIGH40":
            bool(pair_checks[1]["peak_pressure_monotonic_with_duration"]),
        "Kích hoạt nằm trong [0,1]":
            df.c2_motion_activation_0_1.between(0,1).all(),
        "Không có chuyển động khi áp lực bằng 0":
            (df.loc[df.pore_pressure_positive_kpa_synthetic<=0,
                    "c2_motion_activation_0_1"]==0).all(),
        "Lưới kiểm tra ngưỡng đã được tạo": len(sens)==6*4*3,
    }
    pd.DataFrame([{"check":k,"result":"PASS" if v else "FAIL"}
                  for k,v in checks.items()]).to_csv(
        outdir/"M4_C3_validation.csv",index=False,encoding="utf-8-sig")

    # Diagnostic chart: activation vs time for all six scenarios.
    fig, ax=plt.subplots(figsize=(11,6))
    for sid,g in df.groupby("scenario_id",sort=True):
        g=g.sort_values("time_min")
        ax.plot(g.time_min,g.c2_motion_activation_0_1,label=sid)
    ax.set_xlabel("Thời gian (phút)")
    ax.set_ylabel("Chỉ số kích hoạt chuyển động (0–1)")
    ax.set_title("M4-C3 – Đáp ứng chuyển động theo kịch bản mưa")
    ax.grid(alpha=.25); ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(outdir/"figures"/"M4_C3_activation_diagnostics.png",dpi=170)
    plt.close(fig)
    return timing, sens, checks, pd.DataFrame(pair_checks)

# -------------------------- M4-C -------------------------------
def connect_module3_placeholder(module3_csv, motion_features):
    # Reserved for time alignment with Module 3 output.
    # No invented causal mapping from pore pressure to motion is applied.
    if module3_csv is None:
        return "M4-C chưa chạy: chưa cung cấp CSV Module 3."
    raise NotImplementedError(
        "M4-C sẽ được triển khai sau khi thống nhất cách đồng bộ thời gian "
        "và dữ liệu chuyển động; không tự tạo quan hệ áp lực-chuyển động."
    )


# ------------------------- M4-C4 ------------------------------
def build_m4_c4_integrated_dataset(c1_csv, c2_csv, outdir):
    """
    C4: assemble a unified, time-aligned dataset for simulation analysis
    and later experimental replacement/calibration. No ML train/test split
    is performed because there are only six synthetic rainfall scenarios.
    """
    c1 = pd.read_csv(c1_csv)
    c2 = pd.read_csv(c2_csv)
    keys = ["scenario_id", "time_min"]
    for name, frame in [("C1", c1), ("C2", c2)]:
        missing = [k for k in keys if k not in frame.columns]
        if missing:
            raise ValueError(f"C{name} thiếu khóa ghép: {missing}")
        if frame.duplicated(keys).any():
            raise ValueError(f"C{name} có khóa scenario/time trùng lặp")

    # Retain Module 3 + measured-feature proxy from C1 and append C2 coupling outputs.
    c2_only = [c for c in c2.columns if c not in c1.columns or c in
               ["c2_pressure_drive_0_1", "c2_motion_activation_0_1",
                "c2_motion_state", "c2_dynamic_rms_g_synthetic",
                "c2_gyro_rms_deg_s_synthetic", "c2_tilt_deg_synthetic",
                "c2_coupling_assumption"]]
    merged = c1.merge(c2[keys + [c for c in c2_only if c not in keys]],
                      on=keys, how="inner", validate="one_to_one")
    merged = merged.sort_values(["scenario_id", "time_min"]).reset_index(drop=True)

    # Dataset role and provenance labels: explicit that all values are synthetic.
    merged["data_origin"] = "SIMULATION_SYNTHETIC"
    merged["dataset_stage"] = "M4_C4_INTEGRATED"
    merged["motion_features_status"] = "SYNTHETIC_NOT_MPU6050_MEASUREMENTS"
    merged["pressure_status"] = "SYNTHETIC_POSITIVE_PORE_PRESSURE_PROXY"
    merged["movement_label_status"] = "HYPOTHESIS_BASED_C2_LABEL_NOT_GROUND_TRUTH"

    # Per-scenario summary for experiment design.
    summaries=[]
    for sid,g in merged.groupby("scenario_id",sort=True):
        g=g.sort_values("time_min")
        u=g["pore_pressure_positive_kpa_synthetic"].to_numpy(float)
        a=g["c2_motion_activation_0_1"].to_numpy(float)
        t=g["time_min"].to_numpy(float)
        ip=int(np.argmax(u)); ia=int(np.argmax(a))
        active=np.flatnonzero(a>0)
        summaries.append({
            "scenario_id":sid,
            "rain_intensity_mm_h":float(g["rain_intensity_setting_mm_h"].iloc[0]),
            "rain_duration_min":float(g["rain_duration_min"].iloc[0]),
            "n_time_samples":len(g),
            "time_start_min":float(t.min()),"time_end_min":float(t.max()),
            "max_moisture_percent_vwc":float(g["moisture_percent_vwc"].max()),
            "max_positive_pressure_kpa_synthetic":float(u.max()),
            "pressure_peak_time_min":float(t[ip]) if u[ip]>0 else np.nan,
            "max_motion_activation_0_1":float(a.max()),
            "first_motion_response_min":float(t[active[0]]) if len(active) else np.nan,
            "max_motion_rms_g_synthetic":float(g["c2_dynamic_rms_g_synthetic"].max()),
            "max_gyro_rms_deg_s_synthetic":float(g["c2_gyro_rms_deg_s_synthetic"].max()),
        })
    summary=pd.DataFrame(summaries)

    # Validation: alignment, completeness, ordering and provenance.
    numeric_cols=["moisture_percent_vwc","pore_pressure_positive_kpa_synthetic",
                  "c2_pressure_drive_0_1","c2_motion_activation_0_1",
                  "c2_dynamic_rms_g_synthetic","c2_gyro_rms_deg_s_synthetic",
                  "c2_tilt_deg_synthetic"]
    checks={
        "Ghép đúng đủ 6 kịch bản":merged.scenario_id.nunique()==6,
        "Không trùng khóa kịch bản-thời gian":not merged.duplicated(keys).any(),
        "Không có NaN/Inf ở biến số tích hợp":np.isfinite(merged[numeric_cols].to_numpy(float)).all(),
        "Chỉ số kích hoạt nằm trong [0,1]":merged.c2_motion_activation_0_1.between(0,1).all(),
        "Dữ liệu có nhãn nguồn mô phỏng":(merged.data_origin=="SIMULATION_SYNTHETIC").all(),
        "Mỗi kịch bản có thời gian tăng đơn điệu":
            all(np.all(np.diff(g.sort_values("time_min").time_min.to_numpy(float))>0)
                for _,g in merged.groupby("scenario_id")),
    }

    # Machine-readable data dictionary.
    dictionary = [
        ("scenario_id","ID kịch bản mưa","metadata"),
        ("time_min","Thời gian mô phỏng (phút)","time"),
        ("rainfall_mm_h","Cường độ mưa tức thời (mm/h)","input"),
        ("rain_intensity_setting_mm_h","Cường độ mưa thiết lập (mm/h)","input"),
        ("rain_duration_min","Thời lượng mưa (phút)","input"),
        ("cumulative_rainfall_mm","Lượng mưa tích lũy (mm)","input_derived"),
        ("moisture_percent_vwc","Độ ẩm thể tích (%)","module3_state"),
        ("delayed_moisture_percent","Độ ẩm trễ (%)","module3_state"),
        ("wetness_drive_0_1","Chỉ số trạng thái ướt (0–1)","module3_state"),
        ("pore_pressure_positive_kpa_synthetic","Áp lực nước lỗ rỗng dương giả lập (kPa)","module3_target_proxy"),
        ("m4_dynamic_rms_g","RMS gia tốc động từ M4 C1 (g)","module4_feature_synthetic"),
        ("m4_gyro_rms_deg_s","RMS con quay từ M4 C1 (deg/s)","module4_feature_synthetic"),
        ("m4_tilt_reference_mean_deg","Góc nghiêng tham chiếu mô phỏng (deg)","module4_feature_synthetic"),
        ("c2_pressure_drive_0_1","Động lực áp lực chuẩn hóa (0–1)","coupling_intermediate"),
        ("c2_motion_activation_0_1","Chỉ số kích hoạt chuyển động (0–1)","hypothesis_label"),
        ("c2_motion_state","Trạng thái chuyển động giả định","hypothesis_label"),
        ("c2_dynamic_rms_g_synthetic","RMS chuyển động dự báo giả lập (g)","hypothesis_output"),
        ("c2_gyro_rms_deg_s_synthetic","RMS gyro dự báo giả lập (deg/s)","hypothesis_output"),
        ("c2_tilt_deg_synthetic","Góc nghiêng dự báo giả lập (deg)","hypothesis_output"),
        ("data_origin","Nguồn dữ liệu","provenance"),
        ("motion_features_status","Trạng thái đặc trưng chuyển động","provenance"),
        ("pressure_status","Trạng thái áp lực","provenance"),
        ("movement_label_status","Trạng thái nhãn chuyển động","provenance"),
    ]
    data_dict=pd.DataFrame(dictionary,columns=["column","description_vi","role"])

    merged.to_csv(outdir/"M4_C4_integrated_dataset.csv",index=False,encoding="utf-8-sig")
    summary.to_csv(outdir/"M4_C4_scenario_summary.csv",index=False,encoding="utf-8-sig")
    data_dict.to_csv(outdir/"M4_C4_data_dictionary.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"check":k,"result":"PASS" if v else "FAIL"}
                  for k,v in checks.items()]).to_csv(
        outdir/"M4_C4_validation.csv",index=False,encoding="utf-8-sig")

    # Save dataset design note: avoid random row-wise split due time-series leakage.
    (outdir/"M4_C4_dataset_notes.txt").write_text(
        "M4-C4 DATASET NOTES\\n"
        "- Dataset integrates Module 3 rainfall/moisture/positive pore-pressure proxy with Module 4 motion features and C2 hypothesis outputs.\\n"
        "- All current rows are synthetic simulation outputs; not physical experiment measurements.\\n"
        "- C2 motion states are hypothesis-generated labels, not ground-truth landslide labels.\\n"
        "- Do not randomly split individual time rows into train/test: adjacent time points are autocorrelated.\\n"
        "- For later ML evaluation, split by complete rainfall scenario or by independent physical experiment run.\\n"
        "- Replace synthetic M4 features with calibrated MPU6050/IMU measurements only after physical acquisition and preprocessing.\\n"
        "- Do not interpret synthetic pore pressure or motion thresholds as validated field thresholds.\\n",
        encoding="utf-8")
    return merged, summary, data_dict, checks


# ------------------------- MODULE 5 – EVENT MODEL ------------------------------
# Provisional rule thresholds. They are simulation assumptions, not calibrated
# landslide warning thresholds or experimentally validated event boundaries.
M5_DEFORMATION_ACTIVATION_THRESHOLD = 0.25
M5_SLIDING_ACTIVATION_THRESHOLD = 0.75
M5_WETNESS_THRESHOLD = 0.01
M5_PRESSURE_THRESHOLD_KPA = 0.05

def classify_event_state(row):
    """Priority-based instantaneous event classifier."""
    activation = float(row["c2_motion_activation_0_1"])
    wetness = float(row["wetness_drive_0_1"])
    pressure = float(row["pore_pressure_positive_kpa_synthetic"])
    rainfall = float(row["rainfall_mm_h"])
    if activation >= M5_SLIDING_ACTIVATION_THRESHOLD:
        return "SLIDING"
    if activation >= M5_DEFORMATION_ACTIVATION_THRESHOLD:
        return "DEFORMATION"
    if (rainfall > 0 or wetness > M5_WETNESS_THRESHOLD
            or pressure >= M5_PRESSURE_THRESHOLD_KPA):
        return "WATER_ACCUMULATION"
    return "STABLE"

def run_module5_event_model(integrated_csv, outdir):
    """
    M5 maps integrated simulation variables to four event states.
    The classifier is rule-based and provisional; it is not a ground-truth
    landslide classifier.
    """
    df = pd.read_csv(integrated_csv)
    required = ["scenario_id","time_min","rainfall_mm_h","wetness_drive_0_1",
                "pore_pressure_positive_kpa_synthetic","c2_motion_activation_0_1"]
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Dataset C4 thiếu cột: {missing}")

    df=df.sort_values(["scenario_id","time_min"]).reset_index(drop=True)
    df["event_state"] = df.apply(classify_event_state,axis=1)
    state_order=["STABLE","WATER_ACCUMULATION","DEFORMATION","SLIDING"]
    df["event_state_code"]=df.event_state.map({v:i for i,v in enumerate(state_order)})

    # Detect transitions and summarize event intervals for each scenario.
    transitions=[]; intervals=[]; summaries=[]
    for sid,g in df.groupby("scenario_id",sort=True):
        g=g.sort_values("time_min").reset_index(drop=True)
        states=g.event_state.to_numpy()
        times=g.time_min.to_numpy(float)
        starts=np.r_[0,np.flatnonzero(states[1:]!=states[:-1])+1]
        ends=np.r_[starts[1:]-1,len(g)-1]
        for j,(a,b) in enumerate(zip(starts,ends),start=1):
            st=states[a]
            intervals.append({
                "scenario_id":sid,"event_segment":j,"event_state":st,
                "start_time_min":float(times[a]),"end_time_min":float(times[b]),
                "duration_min":float(times[b]-times[a]+1),
                "mean_pressure_kpa":float(g.loc[a:b,"pore_pressure_positive_kpa_synthetic"].mean()),
                "max_activation_0_1":float(g.loc[a:b,"c2_motion_activation_0_1"].max())
            })
            if a>0:
                transitions.append({
                    "scenario_id":sid,"transition_time_min":float(times[a]),
                    "from_state":states[a-1],"to_state":st
                })
        # first occurrence and maximum event level
        first={}
        for st in state_order:
            ix=np.flatnonzero(states==st)
            first[f"first_{st.lower()}_min"]=float(times[ix[0]]) if len(ix) else np.nan
        summaries.append({
            "scenario_id":sid,
            **first,
            "event_state_count":int(len(starts)),
            "sliding_duration_min":float(sum(x["duration_min"] for x in intervals
                if x["scenario_id"]==sid and x["event_state"]=="SLIDING")),
            "deformation_duration_min":float(sum(x["duration_min"] for x in intervals
                if x["scenario_id"]==sid and x["event_state"]=="DEFORMATION")),
            "water_accumulation_duration_min":float(sum(x["duration_min"] for x in intervals
                if x["scenario_id"]==sid and x["event_state"]=="WATER_ACCUMULATION"))
        })

    interval_df=pd.DataFrame(intervals)
    transition_df=pd.DataFrame(transitions,columns=[
        "scenario_id","transition_time_min","from_state","to_state"])
    summary_df=pd.DataFrame(summaries)

    checks={
        "Có đủ 6 kịch bản":df.scenario_id.nunique()==6,
        "Chỉ có 4 trạng thái sự kiện hợp lệ":set(df.event_state.unique()).issubset(set(state_order)),
        "Mã trạng thái khớp thứ tự":df.event_state_code.notna().all(),
        "Không có NaN/Inf ở các biến đầu vào chính":np.isfinite(
            df[["rainfall_mm_h","wetness_drive_0_1",
                "pore_pressure_positive_kpa_synthetic",
                "c2_motion_activation_0_1"]].to_numpy(float)).all(),
        "Kích hoạt trượt luôn lớn hơn hoặc bằng ngưỡng biến dạng":
            M5_SLIDING_ACTIVATION_THRESHOLD > M5_DEFORMATION_ACTIVATION_THRESHOLD,
        "Kịch bản không mưa/không ướt/không áp lực/không chuyển động được gán ổn định":
            (df.loc[(df.rainfall_mm_h<=0)&(df.wetness_drive_0_1<=M5_WETNESS_THRESHOLD)&
                    (df.pore_pressure_positive_kpa_synthetic<M5_PRESSURE_THRESHOLD_KPA)&
                    (df.c2_motion_activation_0_1<M5_DEFORMATION_ACTIVATION_THRESHOLD),
                    "event_state"]=="STABLE").all()
    }
    df["event_model_type"]="RULE_BASED_PROVISIONAL"
    df["event_threshold_note"]=(
        f"deformation_activation>={M5_DEFORMATION_ACTIVATION_THRESHOLD}; "
        f"sliding_activation>={M5_SLIDING_ACTIVATION_THRESHOLD}; "
        f"wetness>{M5_WETNESS_THRESHOLD}; pressure>={M5_PRESSURE_THRESHOLD_KPA}kPa"
    )
    df.to_csv(outdir/"M5_event_dataset.csv",index=False,encoding="utf-8-sig")
    interval_df.to_csv(outdir/"M5_event_intervals.csv",index=False,encoding="utf-8-sig")
    transition_df.to_csv(outdir/"M5_event_transitions.csv",index=False,encoding="utf-8-sig")
    summary_df.to_csv(outdir/"M5_scenario_event_summary.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"check":k,"result":"PASS" if v else "FAIL"}
                  for k,v in checks.items()]).to_csv(
        outdir/"M5_validation.csv",index=False,encoding="utf-8-sig")

    # Plot event-state timeline for six rainfall scenarios.
    fig,axes=plt.subplots(6,1,figsize=(12,11),sharex=False)
    colors={"STABLE":"#8DA0CB","WATER_ACCUMULATION":"#66C2A5",
            "DEFORMATION":"#FCB44A","SLIDING":"#E78A8A"}
    for ax,(sid,g) in zip(axes,df.groupby("scenario_id",sort=True)):
        g=g.sort_values("time_min")
        y=g.event_state_code.to_numpy()
        ax.step(g.time_min,y,where="post",linewidth=1.8)
        for st,code in zip(state_order,range(4)):
            ax.axhline(code,color=colors[st],alpha=.12)
        ax.set_yticks(range(4),["Stable","Water","Deform.","Sliding"])
        ax.set_title(str(sid),loc="left",fontsize=9)
        ax.grid(axis="x",alpha=.2)
    axes[-1].set_xlabel("Thời gian mô phỏng (phút)")
    fig.suptitle("MODULE 5 – Phân loại trạng thái sự kiện theo thời gian")
    fig.tight_layout()
    fig.savefig(outdir/"figures"/"M5_event_timeline.png",dpi=170)
    plt.close(fig)
    return df,summary_df,interval_df,transition_df,checks


# ------------------------- MODULE 5-B – STATE MACHINE -------------------------
# M5-A (the existing M5 above) remains an instantaneous priority rule.
# M5-B adds temporal persistence (dwell time), hysteresis and adjacent-state
# transitions to reduce rapid label flicker. These are provisional simulation
# settings, not experimentally validated warning thresholds.
M5B_DEFORMATION_ON = 0.25
M5B_DEFORMATION_OFF = 0.15
M5B_SLIDING_ON = 0.75
M5B_SLIDING_OFF = 0.60
M5B_WETNESS_THRESHOLD = M5_WETNESS_THRESHOLD
M5B_PRESSURE_THRESHOLD_KPA = M5_PRESSURE_THRESHOLD_KPA
M5B_DWELL_MIN = 2.0
M5B_STATE_ORDER = ["STABLE", "WATER_ACCUMULATION", "DEFORMATION", "SLIDING"]


def _m5b_candidate_state(current, row):
    """Return the next adjacent state only when its entry/exit condition holds."""
    activation = float(row["c2_motion_activation_0_1"])
    wetness = float(row["wetness_drive_0_1"])
    pressure = float(row["pore_pressure_positive_kpa_synthetic"])
    rainfall = float(row["rainfall_mm_h"])
    wet_condition = (rainfall > 0 or wetness > M5B_WETNESS_THRESHOLD
                     or pressure >= M5B_PRESSURE_THRESHOLD_KPA)
    dry_condition = (rainfall <= 0 and wetness <= M5B_WETNESS_THRESHOLD
                     and pressure < M5B_PRESSURE_THRESHOLD_KPA)

    if current == "STABLE":
        return "WATER_ACCUMULATION" if wet_condition else current
    if current == "WATER_ACCUMULATION":
        if activation >= M5B_DEFORMATION_ON:
            return "DEFORMATION"
        if dry_condition:
            return "STABLE"
        return current
    if current == "DEFORMATION":
        if activation >= M5B_SLIDING_ON:
            return "SLIDING"
        if activation < M5B_DEFORMATION_OFF:
            return "WATER_ACCUMULATION"
        return current
    if current == "SLIDING":
        if activation < M5B_SLIDING_OFF:
            return "DEFORMATION"
        return current
    raise ValueError(f"Trạng thái M5-B không hợp lệ: {current}")


def classify_event_state_m5b(group):
    """State-machine classifier with hysteresis and continuous dwell-time test."""
    g = group.sort_values("time_min").copy().reset_index(drop=True)
    required = ["time_min", "rainfall_mm_h", "wetness_drive_0_1",
                "pore_pressure_positive_kpa_synthetic", "c2_motion_activation_0_1"]
    missing = [c for c in required if c not in g.columns]
    if missing:
        raise ValueError(f"Dataset C4 thiếu cột cho M5-B: {missing}")
    times = g.time_min.to_numpy(float)
    if len(g) == 0 or (len(g) > 1 and not np.all(np.diff(times) > 0)):
        raise ValueError("M5-B yêu cầu time_min tăng đơn điệu và nhóm không rỗng.")

    states = ["STABLE"]
    pending_states = [None]
    pending_durations = [0.0]
    transition_times = [np.nan]
    current = "STABLE"
    pending = None
    pending_duration = 0.0

    for i in range(1, len(g)):
        dt = float(times[i] - times[i-1])
        candidate = _m5b_candidate_state(current, g.iloc[i])
        if candidate == current:
            pending = None
            pending_duration = 0.0
        else:
            if candidate != pending:
                pending = candidate
                pending_duration = 0.0
            pending_duration += dt
            if pending_duration + 1e-12 >= M5B_DWELL_MIN:
                current = candidate
                pending = None
                pending_duration = 0.0
                transition_times.append(float(times[i]))
            else:
                transition_times.append(np.nan)
        states.append(current)
        pending_states.append(pending)
        pending_durations.append(pending_duration)

    # The first row has no preceding interval; transition_times is aligned below.
    g["event_state_m5b"] = states
    g["event_state_m5b_code"] = g.event_state_m5b.map(
        {name: i for i, name in enumerate(M5B_STATE_ORDER)})
    g["m5b_pending_state"] = pending_states
    g["m5b_pending_duration_min"] = pending_durations
    return g


def run_module5b_event_model(integrated_csv, outdir):
    """Run M5-B and save intervals, transitions, summaries and validation."""
    df = pd.read_csv(integrated_csv)
    required = ["scenario_id", "time_min", "rainfall_mm_h", "wetness_drive_0_1",
                "pore_pressure_positive_kpa_synthetic", "c2_motion_activation_0_1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset C4 thiếu cột: {missing}")

    pieces = []
    intervals, transitions, summaries = [], [], []
    for sid, group in df.groupby("scenario_id", sort=True):
        g = classify_event_state_m5b(group)
        pieces.append(g)
        states = g.event_state_m5b.to_numpy()
        times = g.time_min.to_numpy(float)
        starts = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1]
        ends = np.r_[starts[1:] - 1, len(g) - 1]
        first = {}
        duration_by_state = {}
        for state in M5B_STATE_ORDER:
            ix = np.flatnonzero(states == state)
            first[f"first_{state.lower()}_min"] = float(times[ix[0]]) if len(ix) else np.nan
            duration_by_state[f"{state.lower()}_duration_min"] = float(
                sum(times[b] - times[a] + (np.median(np.diff(times)) if len(times)>1 else 0)
                    for a,b in zip(starts,ends) if states[a] == state))
        for segment, (a,b) in enumerate(zip(starts, ends), start=1):
            intervals.append({
                "scenario_id": sid, "event_segment": segment,
                "event_state": states[a], "start_time_min": float(times[a]),
                "end_time_min": float(times[b]),
                "duration_min": float(times[b]-times[a] + (np.median(np.diff(times)) if len(times)>1 else 0)),
                "mean_pressure_kpa": float(g.loc[a:b,"pore_pressure_positive_kpa_synthetic"].mean()),
                "max_activation_0_1": float(g.loc[a:b,"c2_motion_activation_0_1"].max())
            })
            if a > 0:
                transitions.append({"scenario_id": sid, "transition_time_min": float(times[a]),
                                    "from_state": states[a-1], "to_state": states[a]})
        summaries.append({"scenario_id": sid, **first, **duration_by_state,
                          "event_state_count": int(len(starts))})

    all_df = pd.concat(pieces, ignore_index=True)
    interval_df = pd.DataFrame(intervals)
    transition_df = pd.DataFrame(transitions, columns=[
        "scenario_id", "transition_time_min", "from_state", "to_state"])
    summary_df = pd.DataFrame(summaries)
    numeric = all_df[["time_min", "m5b_pending_duration_min",
                      "c2_motion_activation_0_1"]].to_numpy(float)
    checks = {
        "Có đủ 6 kịch bản": all_df.scenario_id.nunique() == 6,
        "Chỉ có 4 trạng thái hợp lệ": set(all_df.event_state_m5b.unique()).issubset(set(M5B_STATE_ORDER)),
        "Mã trạng thái không bị thiếu": all_df.event_state_m5b_code.notna().all(),
        "Không có NaN/Inf ở biến số kiểm tra": np.isfinite(numeric).all(),
        "Ngưỡng hysteresis hợp lệ": M5B_DEFORMATION_OFF < M5B_DEFORMATION_ON < M5B_SLIDING_OFF < M5B_SLIDING_ON,
        "Dwell time dương": M5B_DWELL_MIN > 0,
        "Chuyển trạng thái chỉ đi qua trạng thái liền kề": all(
            abs(M5B_STATE_ORDER.index(r.from_state)-M5B_STATE_ORDER.index(r.to_state)) == 1
            for r in transition_df.itertuples())
    }
    all_df["event_model_type_m5b"] = "STATE_MACHINE_HYSTERESIS_DWELL"
    all_df.to_csv(outdir/"M5B_event_dataset.csv", index=False, encoding="utf-8-sig")
    interval_df.to_csv(outdir/"M5B_event_intervals.csv", index=False, encoding="utf-8-sig")
    transition_df.to_csv(outdir/"M5B_event_transitions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(outdir/"M5B_scenario_event_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"check": k, "result": "PASS" if v else "FAIL"}
                  for k,v in checks.items()]).to_csv(
        outdir/"M5B_validation.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(6, 1, figsize=(12, 11), sharex=False)
    for ax, (sid, g) in zip(axes, all_df.groupby("scenario_id", sort=True)):
        ax.step(g.time_min, g.event_state_m5b_code, where="post", linewidth=1.8)
        ax.set_yticks(range(4), ["Stable", "Water", "Deform.", "Sliding"])
        ax.set_title(str(sid), loc="left", fontsize=9)
        ax.grid(axis="x", alpha=.2)
    axes[-1].set_xlabel("Thời gian mô phỏng (phút)")
    fig.suptitle("MODULE 5-B – State machine, hysteresis và dwell time")
    fig.tight_layout()
    fig.savefig(outdir/"figures"/"M5B_event_timeline.png", dpi=170)
    plt.close(fig)
    return all_df, summary_df, interval_df, transition_df, checks


def compare_m5a_m5b(outdir):
    """Compare the two classifiers on common scenario/time rows."""
    a_path, b_path = outdir/"M5_event_dataset.csv", outdir/"M5B_event_dataset.csv"
    if not a_path.exists() or not b_path.exists():
        raise FileNotFoundError("Cần chạy cả M5 và M5B trước khi so sánh.")
    a = pd.read_csv(a_path)[["scenario_id", "time_min", "event_state"]]
    b = pd.read_csv(b_path)[["scenario_id", "time_min", "event_state_m5b"]]
    merged = a.merge(b, on=["scenario_id", "time_min"], validate="one_to_one")
    merged["same_label"] = merged.event_state == merged.event_state_m5b
    rows = []
    for sid, g in merged.groupby("scenario_id", sort=True):
        rows.append({"scenario_id": sid, "rows_compared": len(g),
                     "agreement_percent": float(g.same_label.mean()*100),
                     "different_label_rows": int((~g.same_label).sum())})
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir/"M5_M5B_comparison.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(outdir/"M5_M5B_rowwise_comparison.csv", index=False, encoding="utf-8-sig")
    return merged, summary


# ------------------------- MODULE 5-C – TRANSITION THRESHOLD ANALYSIS -------------------------
def run_module5c_transition_analysis(outdir):
    """Analyze M5-B transition conditions across the six rainfall scenarios.

    This is descriptive analysis of the current synthetic simulation, not
    calibration or a claim of experimentally validated landslide thresholds.
    """
    dataset_path = outdir / "M5B_event_dataset.csv"
    transitions_path = outdir / "M5B_event_transitions.csv"
    if not dataset_path.exists() or not transitions_path.exists():
        raise FileNotFoundError(
            "M5-C cần M5B_event_dataset.csv và M5B_event_transitions.csv. "
            "Hãy chạy --stage M5B trước."
        )

    df = pd.read_csv(dataset_path)
    trans = pd.read_csv(transitions_path)
    required = ["scenario_id", "time_min", "rainfall_mm_h",
                "rain_intensity_setting_mm_h", "rain_duration_min",
                "cumulative_rainfall_mm", "wetness_drive_0_1",
                "pore_pressure_positive_kpa_synthetic",
                "c2_motion_activation_0_1", "event_state_m5b"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"M5-B dataset thiếu cột cho M5-C: {missing}")

    # Keep only actual transitions and retrieve the state/forcing values at
    # the transition timestamp. No threshold is re-fitted in this stage.
    if trans.empty:
        transition_detail = pd.DataFrame(columns=[
            "scenario_id", "transition_time_min", "from_state", "to_state",
            "rain_intensity_setting_mm_h", "rain_duration_min", "rainfall_mm_h",
            "cumulative_rainfall_mm", "wetness_drive_0_1",
            "pore_pressure_positive_kpa_synthetic", "c2_motion_activation_0_1"])
    else:
        lookup_cols = ["scenario_id", "time_min", "rain_intensity_setting_mm_h",
                       "rain_duration_min", "rainfall_mm_h", "cumulative_rainfall_mm",
                       "wetness_drive_0_1", "pore_pressure_positive_kpa_synthetic",
                       "c2_motion_activation_0_1"]
        transition_detail = trans.merge(
            df[lookup_cols], left_on=["scenario_id", "transition_time_min"],
            right_on=["scenario_id", "time_min"], how="left", validate="many_to_one"
        ).drop(columns=["time_min"])

    transition_detail["transition_type"] = (
        transition_detail["from_state"].astype(str) + "_TO_" +
        transition_detail["to_state"].astype(str)
    )
    transition_detail["transition_data_status"] = np.where(
        transition_detail["c2_motion_activation_0_1"].notna(), "MATCHED", "MISSING"
    )
    transition_detail["transition_time_semantics"] = "DWELL_CONFIRMED_TIME"
    transition_detail.to_csv(outdir/"M5C_transition_conditions.csv", index=False,
                             encoding="utf-8-sig")

    # Per-scenario summary: first transition times and forcing values at each
    # transition type. Missing transitions remain NaN rather than being imputed.
    scenario_rows = []
    for sid, g in df.groupby("scenario_id", sort=True):
        g = g.sort_values("time_min")
        row = {
            "scenario_id": sid,
            "rain_intensity_setting_mm_h": float(g.rain_intensity_setting_mm_h.iloc[0]),
            "rain_duration_min": float(g.rain_duration_min.iloc[0]),
            "total_rainfall_mm_max": float(g.cumulative_rainfall_mm.max()),
            "max_wetness_drive_0_1": float(g.wetness_drive_0_1.max()),
            "max_positive_pore_pressure_kpa": float(g.pore_pressure_positive_kpa_synthetic.max()),
            "max_motion_activation_0_1": float(g.c2_motion_activation_0_1.max()),
        }
        tg = transition_detail[transition_detail.scenario_id == sid]
        for tr_type in ["WATER_ACCUMULATION_TO_DEFORMATION", "DEFORMATION_TO_SLIDING",
                        "DEFORMATION_TO_WATER_ACCUMULATION", "SLIDING_TO_DEFORMATION",
                        "WATER_ACCUMULATION_TO_STABLE", "STABLE_TO_WATER_ACCUMULATION"]:
            name = tr_type.lower()
            hit = tg[tg.transition_type == tr_type]
            row[f"{name}_time_min"] = float(hit.transition_time_min.iloc[0]) if len(hit) else np.nan
            for col, suffix in [
                ("cumulative_rainfall_mm", "rainfall_mm"),
                ("wetness_drive_0_1", "wetness"),
                ("pore_pressure_positive_kpa_synthetic", "pressure_kpa"),
                ("c2_motion_activation_0_1", "activation")]:
                row[f"{name}_{suffix}"] = float(hit[col].iloc[0]) if len(hit) and pd.notna(hit[col].iloc[0]) else np.nan
        scenario_rows.append(row)
    scenario_summary = pd.DataFrame(scenario_rows)
    scenario_summary.to_csv(outdir/"M5C_scenario_threshold_summary.csv", index=False,
                            encoding="utf-8-sig")

    # Aggregate transition-condition ranges by transition type. These ranges
    # describe the six simulations only; they are not universal thresholds.
    aggregate_rows = []
    for tr_type, g in transition_detail.groupby("transition_type", dropna=False, sort=True):
        row = {"transition_type": tr_type, "n_transitions": int(len(g)),
               "n_scenarios": int(g.scenario_id.nunique())}
        for col in ["rainfall_mm_h", "cumulative_rainfall_mm", "wetness_drive_0_1",
                    "pore_pressure_positive_kpa_synthetic", "c2_motion_activation_0_1"]:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            row[f"{col}_min"] = float(vals.min()) if len(vals) else np.nan
            row[f"{col}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{col}_max"] = float(vals.max()) if len(vals) else np.nan
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(outdir/"M5C_transition_threshold_ranges.csv", index=False,
                     encoding="utf-8-sig")

    # Compare the two rainfall-intensity settings while retaining the three
    # duration levels. This avoids treating intensity and duration as one factor.
    factor_rows = []
    def median_or_nan(values):
        numeric = pd.to_numeric(values, errors="coerce").dropna()
        return float(numeric.median()) if len(numeric) else np.nan

    for (intensity, duration), g in scenario_summary.groupby(
            ["rain_intensity_setting_mm_h", "rain_duration_min"], sort=True):
        factor_rows.append({
            "rain_intensity_setting_mm_h": intensity,
            "rain_duration_min": duration,
            "scenario_count": int(g.scenario_id.nunique()),
            "first_deformation_time_min": median_or_nan(g.water_accumulation_to_deformation_time_min),
            "first_sliding_time_min": median_or_nan(g.deformation_to_sliding_time_min),
            "max_pressure_kpa": median_or_nan(g.max_positive_pore_pressure_kpa),
            "max_activation_0_1": median_or_nan(g.max_motion_activation_0_1),
        })
    factor_summary = pd.DataFrame(factor_rows)
    factor_summary.to_csv(outdir/"M5C_rainfall_factor_summary.csv", index=False,
                          encoding="utf-8-sig")

    checks = {
        "Có đủ 6 kịch bản": df.scenario_id.nunique() == 6,
        "Mỗi kịch bản có một mức cường độ và thời gian mưa":
            scenario_summary[["rain_intensity_setting_mm_h", "rain_duration_min"]].notna().all().all(),
        "Dữ liệu điều kiện chuyển trạng thái được ghép đúng thời điểm":
            bool(transition_detail.empty or (transition_detail.transition_data_status == "MATCHED").all()),
        "Chỉ phân tích các chuyển trạng thái liền kề": bool(
            transition_detail.empty or all(
                abs(M5B_STATE_ORDER.index(r.from_state)-M5B_STATE_ORDER.index(r.to_state)) == 1
                for r in transition_detail.itertuples())),
        "Không thay đổi bộ ngưỡng M5-B": True,
        "Các biến phân tích hữu hạn": bool(np.isfinite(
            df[["rainfall_mm_h", "cumulative_rainfall_mm", "wetness_drive_0_1",
                "pore_pressure_positive_kpa_synthetic", "c2_motion_activation_0_1"]]
            .to_numpy(float)).all()),
    }
    pd.DataFrame([{"check": k, "result": "PASS" if v else "FAIL"}
                  for k, v in checks.items()]).to_csv(
        outdir/"M5C_validation.csv", index=False, encoding="utf-8-sig")

    # Diagnostic plot: transition activation and pore pressure for each scenario.
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    if not transition_detail.empty:
        for sid, g in transition_detail.groupby("scenario_id", sort=True):
            axes[0].scatter(g.transition_time_min, g.c2_motion_activation_0_1, label=str(sid))
            axes[1].scatter(g.transition_time_min,
                            g.pore_pressure_positive_kpa_synthetic, label=str(sid))
    axes[0].set_ylabel("Activation (0–1)")
    axes[1].set_ylabel("Áp lực nước lỗ rỗng dương (kPa)")
    axes[1].set_xlabel("Thời điểm chuyển trạng thái (phút)")
    axes[0].set_title("M5-C – Chỉ số kích hoạt tại thời điểm chuyển trạng thái")
    for ax in axes:
        ax.grid(alpha=.25)
        if not transition_detail.empty:
            ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir/"figures"/"M5C_transition_conditions.png", dpi=170)
    plt.close(fig)

    print("M5-C: phân tích điều kiện chuyển trạng thái hoàn tất.")
    print("Lưu ý: các khoảng giá trị là kết quả mô phỏng hiện tại, không phải ngưỡng thực nghiệm.")
    return transition_detail, scenario_summary, aggregate, factor_summary, checks

# ------------------------- MODULE 5-D – EVENT MODEL CONSISTENCY & FINALIZATION -------------------------
def run_module5d_final_audit(outdir):
    """Consolidate M5-A/B/C outputs and run cross-stage consistency checks.

    This is a simulation-model audit, not validation against field observations.
    It deliberately does not tune or modify any M5-A/M5-B thresholds.
    """
    paths = {
        "a_data": outdir / "M5_event_dataset.csv",
        "b_data": outdir / "M5B_event_dataset.csv",
        "a_summary": outdir / "M5_scenario_event_summary.csv",
        "b_summary": outdir / "M5B_scenario_event_summary.csv",
        "a_transitions": outdir / "M5_event_transitions.csv",
        "b_transitions": outdir / "M5B_event_transitions.csv",
        "a_validation": outdir / "M5_validation.csv",
        "b_validation": outdir / "M5B_validation.csv",
        "c_validation": outdir / "M5C_validation.csv",
        "c_scenario": outdir / "M5C_scenario_threshold_summary.csv",
        "c_conditions": outdir / "M5C_transition_conditions.csv",
        "c_ranges": outdir / "M5C_transition_threshold_ranges.csv",
    }
    missing = [str(v.name) for v in paths.values() if not v.exists()]
    if missing:
        raise FileNotFoundError(
            "M5-D thiếu đầu ra của các bước trước: " + ", ".join(missing)
            + ". Hãy chạy M5, M5B và M5C trước trong cùng thư mục --output."
        )

    a = pd.read_csv(paths["a_data"])
    b = pd.read_csv(paths["b_data"])
    a_sum = pd.read_csv(paths["a_summary"])
    b_sum = pd.read_csv(paths["b_summary"])
    a_tr = pd.read_csv(paths["a_transitions"])
    b_tr = pd.read_csv(paths["b_transitions"])
    c_sum = pd.read_csv(paths["c_scenario"])
    c_cond = pd.read_csv(paths["c_conditions"])
    upstream_validation = {
        stage: pd.read_csv(paths[f"{stage.lower()}_validation"])
        for stage in ["a", "b", "c"]
    }

    keys = ["scenario_id", "time_min"]
    required_a = keys + ["event_state"]
    required_b = keys + ["event_state_m5b"]
    for label, frame, required in [("M5-A", a, required_a), ("M5-B", b, required_b)]:
        absent = [c for c in required if c not in frame.columns]
        if absent:
            raise ValueError(f"{label} thiếu cột bắt buộc: {absent}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{label} có khóa scenario_id/time_min trùng lặp")

    # Exact row alignment is required before comparing the two classifiers.
    merged = a[keys + ["event_state"]].merge(
        b[keys + ["event_state_m5b"]], on=keys, how="outer",
        validate="one_to_one", indicator=True
    )
    merged["label_match"] = merged.event_state.eq(merged.event_state_m5b)
    merged["alignment_status"] = merged["_merge"].astype(str)
    # Verify that A/B are classifying the same forcing and coupling inputs.
    shared_inputs = [c for c in ["rainfall_mm_h", "wetness_drive_0_1",
                     "pore_pressure_positive_kpa_synthetic", "c2_motion_activation_0_1"]
                     if c in a.columns and c in b.columns]
    inputs_identical = bool(shared_inputs) and all(
        np.allclose(pd.to_numeric(a[c], errors="coerce"),
                    pd.to_numeric(b[c], errors="coerce"),
                    rtol=1e-10, atol=1e-12, equal_nan=True)
        for c in shared_inputs
    )
    merged.drop(columns=["_merge"]).to_csv(
        outdir / "M5D_final_event_dataset.csv", index=False, encoding="utf-8-sig"
    )

    # Prefix summaries to preserve A/B distinctions without overwriting fields.
    def prefixed(frame, prefix):
        return frame.rename(columns={c: f"{prefix}_{c}" for c in frame.columns if c != "scenario_id"})
    summary = prefixed(a_sum, "M5A").merge(
        prefixed(b_sum, "M5B"), left_on="scenario_id", right_on="scenario_id",
        how="outer", validate="one_to_one"
    ).merge(c_sum, on="scenario_id", how="outer", validate="one_to_one")
    summary.to_csv(outdir / "M5D_final_scenario_summary.csv", index=False,
                   encoding="utf-8-sig")

    allowed = set(M5B_STATE_ORDER)
    b_transition_adjacent = bool(b_tr.empty or all(
        r.from_state in allowed and r.to_state in allowed
        and abs(M5B_STATE_ORDER.index(r.from_state) - M5B_STATE_ORDER.index(r.to_state)) == 1
        for r in b_tr.itertuples()
    ))
    c_matched = bool(c_cond.empty or (
        "transition_data_status" in c_cond.columns
        and c_cond.transition_data_status.eq("MATCHED").all()
    ))
    upstream_all_pass = all(
        ("result" in frame.columns and frame.result.astype(str).str.upper().eq("PASS").all())
        or ("pass_text" in frame.columns and frame.pass_text.astype(str).str.upper().eq("PASS").all())
        for frame in upstream_validation.values()
    )
    checks = {
        "Các kiểm tra nội bộ M5-A/B/C đều PASS": upstream_all_pass,
        "M5-A và M5-B sử dụng cùng các biến đầu vào": inputs_identical,
        "M5-A và M5-B có cùng tập 6 kịch bản":
            set(a.scenario_id.unique()) == set(b.scenario_id.unique()) and len(set(a.scenario_id.unique())) == 6,
        "M5-A và M5-B có cùng lưới thời gian theo từng kịch bản":
            len(merged) == len(a) == len(b) and merged.alignment_status.eq("both").all(),
        "Nhãn M5-A thuộc 4 trạng thái hợp lệ": set(a.event_state.dropna().unique()).issubset(allowed),
        "Nhãn M5-B thuộc 4 trạng thái hợp lệ": set(b.event_state_m5b.dropna().unique()).issubset(allowed),
        "M5-B chỉ chuyển giữa hai trạng thái liền kề": b_transition_adjacent,
        "Các điều kiện chuyển trạng thái M5-C ghép được với dữ liệu M5-B": c_matched,
        "M5-C có tổng hợp đủ 6 kịch bản": c_sum.scenario_id.nunique() == 6,
        "Các ngưỡng M5-B vẫn giữ nguyên cấu hình đã chọn": (
            M5B_DEFORMATION_ON == 0.25 and M5B_DEFORMATION_OFF == 0.15
            and M5B_SLIDING_ON == 0.75 and M5B_SLIDING_OFF == 0.60
            and M5B_DWELL_MIN == 2.0
        ),
        "Không có giá trị thời gian trùng trong mỗi kịch bản M5-A/B":
            not a.duplicated(keys).any() and not b.duplicated(keys).any(),
    }
    pd.DataFrame([{"check": k, "result": "PASS" if v else "FAIL"}
                  for k, v in checks.items()]).to_csv(
        outdir / "M5D_validation.csv", index=False, encoding="utf-8-sig"
    )

    # Cross-stage agreement by scenario; descriptive only, not accuracy.
    agreement = merged.groupby("scenario_id", sort=True).agg(
        rows_compared=("label_match", "size"),
        agreement_rows=("label_match", "sum"),
        agreement_percent=("label_match", lambda x: float(x.mean() * 100)),
    ).reset_index()
    agreement.to_csv(outdir / "M5D_A_M5B_agreement.csv", index=False,
                     encoding="utf-8-sig")

    # Human-readable audit note, with explicit provenance limitations.
    passed = sum(bool(v) for v in checks.values())
    lines = [
        "MODULE 5-D – KIỂM TRA TÍNH NHẤT QUÁN VÀ TỔNG HỢP EVENT MODEL",
        f"Kết quả kiểm tra: {passed}/{len(checks)} PASS",
        "",
        "Phạm vi:",
        "- M5-A: bộ phân loại tức thời theo luật, dùng để đối chiếu.",
        "- M5-B: state machine có hysteresis và dwell time 2 phút.",
        "- M5-C: thống kê điều kiện tại thời điểm chuyển trạng thái.",
        "- M5-D: kiểm tra tính nhất quán và tổng hợp đầu ra; không huấn luyện ML.",
        "",
        "Cấu hình M5-B được giữ nguyên:",
        f"- Deformation ON/OFF: {M5B_DEFORMATION_ON}/{M5B_DEFORMATION_OFF}",
        f"- Sliding ON/OFF: {M5B_SLIDING_ON}/{M5B_SLIDING_OFF}",
        f"- Dwell time: {M5B_DWELL_MIN} phút",
        "",
        "Giới hạn diễn giải:",
        "- Dữ liệu hiện tại là dữ liệu mô phỏng tổng hợp; không phải phép đo MPU6050 thực.",
        "- Nhãn trạng thái là nhãn giả thuyết/rule-based, không phải ground truth.",
        "- Tỷ lệ đồng thuận M5-A/M5-B chỉ đo mức giống nhau giữa hai luật, không phải accuracy.",
        "- Các khoảng điều kiện M5-C không phải ngưỡng cảnh báo đã hiệu chuẩn ngoài thực nghiệm.",
        "- Không thể kết luận độ chính xác dự báo sạt lở từ 6 kịch bản mô phỏng này.",
        "",
        "Chi tiết kiểm tra:",
    ] + [f"{'PASS' if ok else 'FAIL'} - {name}" for name, ok in checks.items()]
    (outdir / "M5D_event_model_report.txt").write_text("\n".join(lines), encoding="utf-8")

    # Compact scenario-level agreement chart.
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(agreement.scenario_id.astype(str), agreement.agreement_percent)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Kịch bản")
    ax.set_ylabel("Tỷ lệ nhãn M5-A trùng M5-B (%)")
    ax.set_title("M5-D – Mức đồng thuận giữa hai bộ phân loại")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "M5D_A_M5B_agreement.png", dpi=170)
    plt.close(fig)

    print(f"MODULE 5-D: {passed}/{len(checks)} checks PASS")
    return merged, summary, agreement, checks


# -------------------------- M4-D -------------------------------
def export_event_ready_placeholder(features, outdir):
    # Export current features as an intermediate dataset, not final event labels.
    features.to_csv(outdir/"M4_D_features_intermediate.csv",
                    index=False, encoding="utf-8-sig")

# -------------------------- PIPELINE ---------------------------
def main():
    if (sys.stdout.encoding or "").lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if (sys.stderr.encoding or "").lower() != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Module 4 motion simulation")
    parser.add_argument("--stage", choices=["A1","A2","A3","A4","B1","B2","B3","C1","C2","C3","C4","M5","M5B","M5_COMPARE","M5C","M5D","ALL"], default="A1")
    parser.add_argument("--output", type=Path,
        default=Path(__file__).resolve().parent/"results_module_4")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fs", type=float, default=FS_HZ)
    parser.add_argument("--total-min", type=float, default=TOTAL_MIN)
    parser.add_argument(
        "--module3-csv",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results_module_3"
        / "module_1_2_3_all_scenarios.csv"
    )
    args = parser.parse_args()
    make_output_dirs(args.output)

    if args.stage == "C2":
        combined_c2, summary_c2, checks_c2 = run_m4_c2_pressure_motion_coupling(
            args.module3_csv, args.output
        )
        print("M4-C2: đã chạy mô hình giả thuyết áp lực → chuyển động.")
        print("M4-C2:", sum(checks_c2.values()), "/", len(checks_c2), "checks PASS")
        for name, ok in checks_c2.items():
            print(("PASS" if ok else "FAIL"), "-", name)
        print(summary_c2.to_string(index=False))
        return

    if args.stage == "M5":
        integrated_csv=args.output/"M4_C4_integrated_dataset.csv"
        if not integrated_csv.exists():
            integrated_csv = args.output / "M4_C4_integrated_dataset.csv"
        event_df,event_summary,event_intervals,event_transitions,event_checks=run_module5_event_model(
            integrated_csv,args.output
        )
        print("MODULE 5:",sum(event_checks.values()),"/",len(event_checks),"checks PASS")
        for name,ok in event_checks.items():
            print(("PASS" if ok else "FAIL"),"-",name)
        print(event_summary.to_string(index=False))
        return

    if args.stage == "M5B":
        integrated_csv = args.output / "M4_C4_integrated_dataset.csv"
        event_df, event_summary, event_intervals, event_transitions, event_checks = run_module5b_event_model(
            integrated_csv, args.output
        )
        print("MODULE 5-B:", sum(event_checks.values()), "/", len(event_checks), "checks PASS")
        for name, ok in event_checks.items():
            print(("PASS" if ok else "FAIL"), "-", name)
        print(event_summary.to_string(index=False))
        return

    if args.stage == "M5_COMPARE":
        integrated_csv = args.output / "M4_C4_integrated_dataset.csv"
        run_module5_event_model(integrated_csv, args.output)
        run_module5b_event_model(integrated_csv, args.output)
        _, comparison_summary = compare_m5a_m5b(args.output)
        print("M5-A vs M5-B – tỷ lệ thống nhất nhãn:")
        print(comparison_summary.to_string(index=False))
        return

    if args.stage == "M5C":
        _, scenario_summary, transition_ranges, factor_summary, checks_m5c = run_module5c_transition_analysis(
            args.output
        )
        print("MODULE 5-C:", sum(checks_m5c.values()), "/", len(checks_m5c), "checks PASS")
        for name, ok in checks_m5c.items():
            print(("PASS" if ok else "FAIL"), "-", name)
        print("Tóm tắt theo kịch bản:")
        print(scenario_summary.to_string(index=False))
        print("Khoảng điều kiện tại thời điểm chuyển trạng thái:")
        print(transition_ranges.to_string(index=False))
        print("Tóm tắt theo cường độ × thời gian mưa:")
        print(factor_summary.to_string(index=False))
        return

    if args.stage == "M5D":
        _, summary_m5d, agreement_m5d, checks_m5d = run_module5d_final_audit(args.output)
        for name, ok in checks_m5d.items():
            print(("PASS" if ok else "FAIL"), "-", name)
        print("Đồng thuận M5-A/M5-B theo kịch bản:")
        print(agreement_m5d.to_string(index=False))
        return

    if args.stage == "C4":
        c1_csv = args.output / "M4_C1_module3_module4_synchronized.csv"
        c2_csv = args.output / "M4_C2_pressure_driven_motion.csv"
        merged_c4, summary_c4, dictionary_c4, checks_c4 = build_m4_c4_integrated_dataset(
            c1_csv, c2_csv, args.output
        )
        print("M4-C4:", sum(checks_c4.values()), "/", len(checks_c4), "checks PASS")
        for name, ok in checks_c4.items():
            print(("PASS" if ok else "FAIL"), "-", name)
        print(summary_c4.to_string(index=False))
        return

    if args.stage == "C3":
        c2_csv = args.output / "M4_C2_pressure_driven_motion.csv"
        timing_c3, sens_c3, checks_c3, pairs_c3 = run_m4_c3_coupling_diagnostics(
            c2_csv, args.output
        )
        print("M4-C3:", sum(checks_c3.values()), "/", len(checks_c3), "checks PASS")
        for name, ok in checks_c3.items():
            print(("PASS" if ok else "FAIL"), "-", name)
        print(timing_c3.to_string(index=False))
        print(pairs_c3.to_string(index=False))
        return

    raw = simulate_raw_signals(args.fs, args.total_min, args.seed)
    raw.to_csv(args.output/"M4_A1_raw_signals.csv", index=False,
               encoding="utf-8-sig")
    plot_raw(raw, args.output)
    print(f"M4-A1: tạo {len(raw):,} mẫu tại {args.fs:g} Hz.")
    print("Đã lưu M4_A1_raw_signals.csv và biểu đồ RAW.")

    if args.stage in ["A2","A3","A4","B1","B2","B3","C1","ALL"]:
        features = extract_features(raw, args.fs, WINDOW_SEC)
        features.to_csv(args.output/"M4_A2_features_10s.csv", index=False,
                        encoding="utf-8-sig")
        stats, checks = save_m4_a2_outputs(features, args.output)
        print("M4-A2: đã tính đặc trưng theo cửa sổ 10 giây.")
        print("M4-A2 validation:", sum(checks.values()), "/", len(checks), "PASS")
    else:
        return

    if args.stage in ["A3","A4","B1","B2","B3","C1","ALL"]:
        summary = summarize_states(features)
        summary.to_csv(args.output/"M4_A3_state_summary.csv", index=False,
                       encoding="utf-8-sig")
        stats, overlap = analyze_state_separation(features, args.output)
        print("M4-A3: đã tổng hợp và so sánh đặc trưng theo 4 trạng thái.")
        print("M4-A3 state summary:")
        print(stats.to_string(index=False))
        print("Số cặp trạng thái có khoảng giá trị chồng lấn:",
              int(overlap.ranges_overlap.sum()), "/", len(overlap))
    else:
        return

    if args.stage in ["A4","B3","C1","ALL"]:
        checks = validate_simulation(raw, features, summary, args.output,
                                     args.seed, args.fs, args.total_min)
        print("M4-A4:", sum(x[1] for x in checks), "/", len(checks), "checks PASS")
        for name, ok in checks:
            print(("PASS" if ok else "FAIL"), "-", name)

    if args.stage in ["B1","B3","C1","ALL"]:
        result_b1, relative_b1, report_b1 = run_m4_b1_amplitude_sensitivity(
            args.output, args.fs, args.total_min, args.seed
        )
        print("M4-B1: đã khảo sát độ nhạy biên độ chuyển động.")
        print(report_b1.to_string(index=False))

    if args.stage in ["B2","B3","C1","ALL"]:
        result_b2, response_b2, checks_b2 = run_m4_b2_timing_sensitivity(
            args.output, args.fs, args.total_min, args.seed
        )
        print("M4-B2: đã khảo sát độ nhạy theo thời gian chuyển trạng thái.")
        print("M4-B2:", sum(checks_b2.values()), "/", len(checks_b2), "checks PASS")
        for name, ok in checks_b2.items():
            print(("PASS" if ok else "FAIL"), "-", name)
        print(response_b2.to_string(index=False))

    if args.stage in ["B3","ALL"]:
        sensitivity_b3, candidates_b3, checks_b3 = run_m4_b3_parameter_synthesis(
            args.output
        )
        print("M4-B3: đã tổng hợp độ nhạy và đề xuất bộ tham số đại diện.")
        print("M4-B3:", sum(checks_b3.values()), "/", len(checks_b3), "checks PASS")
        for name, ok in checks_b3.items():
            print(("PASS" if ok else "FAIL"), "-", name)
        print("\nBảng tổng hợp độ nhạy:")
        print(sensitivity_b3.to_string(index=False))
        print("\nCác bộ tham số đại diện:")
        print(candidates_b3.to_string(index=False))

    if args.stage in ["C1","ALL"]:
        combined_c1, summary_c1, checks_c1 = connect_module3_module4(
            args.module3_csv,
            args.output / "M4_A2_features_10s.csv",
            args.output
        )
        print("M4-C1: đã đồng bộ dữ liệu Module 3 và Module 4.")
        print("M4-C1:", sum(checks_c1.values()), "/", len(checks_c1), "checks PASS")
        for name, ok in checks_c1.items():
            print(("PASS" if ok else "FAIL"), "-", name)
        print(summary_c1.to_string(index=False))

    if args.stage in ["C2","ALL"]:
        combined_c2, summary_c2, checks_c2 = run_m4_c2_pressure_motion_coupling(
            args.module3_csv, args.output
        )
        print("M4-C2: đã chạy mô hình giả thuyết áp lực → chuyển động.")
        print("M4-C2:", sum(checks_c2.values()), "/", len(checks_c2), "checks PASS")
        for name, ok in checks_c2.items():
            print(("PASS" if ok else "FAIL"), "-", name)
        print(summary_c2.to_string(index=False))
        if args.stage == "ALL":
            timing_all, sens_all, checks_all, pairs_all = run_m4_c3_coupling_diagnostics(
                args.output / "M4_C2_pressure_driven_motion.csv", args.output
            )
            print("M4-C3:", sum(checks_all.values()), "/", len(checks_all), "checks PASS")
            merged_all, summary_all, dictionary_all, checks_c4_all = build_m4_c4_integrated_dataset(
                args.output / "M4_C1_module3_module4_synchronized.csv",
                args.output / "M4_C2_pressure_driven_motion.csv", args.output
            )
            print("M4-C4:", sum(checks_c4_all.values()), "/", len(checks_c4_all), "checks PASS")
            run_module5_event_model(args.output / "M4_C4_integrated_dataset.csv", args.output)
            run_module5b_event_model(args.output / "M4_C4_integrated_dataset.csv", args.output)
            compare_m5a_m5b(args.output)
            run_module5c_transition_analysis(args.output)
            _, _, _, checks_m5d_all = run_module5d_final_audit(args.output)
            print("M5-D:", sum(checks_m5d_all.values()), "/", len(checks_m5d_all), "checks PASS")

if __name__ == "__main__":
    main()
