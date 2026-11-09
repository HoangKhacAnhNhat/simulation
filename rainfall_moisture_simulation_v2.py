"""
MODULE 1 + MODULE 2 (Version 2)
Two rainfall intensities x three manually controlled rainfall durations.

Model scope:
- One constant rainfall intensity during each rainfall event.
- Rain starts at RAIN_START_MIN and stops after the selected duration.
- The rainfall duration is manually controlled in the physical experiment
  using a stopwatch; the simulation represents the planned elapsed time.
- After rainfall stops, the model continues through a post-rain observation period.

Units:
time: min
rainfall intensity: mm/h
cumulative rainfall: mm
theta: m3/m3 (volumetric water content)

This is a reduced-order phenomenological model, not a Richards-equation
or spatial infiltration model. kin and kd are effective coefficients,
not independently measured soil hydraulic properties.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# A. EXPERIMENTAL DESIGN - EDIT THESE VALUES AFTER CALIBRATION
# ============================================================

# Replace these illustrative values with measured rainfall intensities.
RAIN_LOW_MM_H = 20.0
RAIN_HIGH_MM_H = 40.0

# Provisional durations. Adjust after a pilot test with the small soil box.
DURATIONS_MIN = [10.0, 20.0, 30.0]

# Time setup
DT_MIN = 1.0
RAIN_START_MIN = 10.0          # baseline observation before rainfall
POST_RAIN_OBSERVATION_MIN = 120.0

# Reference intensity for normalized rainfall
R_REF_MM_H = 40.0

# Initial moisture model parameters (illustrative, not calibrated)
THETA_MIN = 0.10
THETA_MAX = 0.45
THETA_INITIAL = 0.15
K_IN_PER_MIN = 0.025
K_D_PER_MIN = 0.003

OUTPUT_DIR = Path(__file__).resolve().parent / "results_module_1_2_v2"


# ============================================================
# B. AUTOMATICALLY BUILD THE 2 x 3 EXPERIMENT MATRIX
# ============================================================

def build_scenarios():
    """
    Six scenarios:
      low intensity x 10, 20, 30 min
      high intensity x 10, 20, 30 min
    """
    scenarios = {}
    intensity_levels = {
        "low": RAIN_LOW_MM_H,
        "high": RAIN_HIGH_MM_H,
    }

    for level_name, intensity in intensity_levels.items():
        for duration in DURATIONS_MIN:
            duration_label = int(duration) if float(duration).is_integer() else duration
            scenario_id = f"{level_name.upper()}_{intensity:g}mmh_{duration_label}min"
            scenarios[scenario_id] = {
                "intensity_mm_h": float(intensity),
                "duration_min": float(duration),
                "theta_initial": THETA_INITIAL,
                "rain_start_min": RAIN_START_MIN,
            }
    return scenarios


# ============================================================
# C. MODULE 1 - CONSTANT-INTENSITY RAINFALL EVENT
# ============================================================

def generate_rainfall(time_min, intensity_mm_h, start_min, duration_min):
    """
    Constant rainfall intensity during [start_min, start_min + duration_min).
    Zero before and after the rainfall event.
    """
    end_min = start_min + duration_min
    rainfall = np.where(
        (time_min >= start_min) & (time_min < end_min),
        intensity_mm_h,
        0.0
    )
    return rainfall


def calculate_cumulative_rainfall(rainfall_mm_h, dt_min):
    """
    P[t] = sum(R[t] * dt / 60)
    Rainfall rate at sample t is interpreted over the following time interval.
    """
    return np.cumsum(rainfall_mm_h * dt_min / 60.0)


def rainfall_intensity_from_collected_water(volume_litre, collector_area_m2,
                                             collection_time_min):
    """
    Estimate rainfall intensity from collected water.

    Since 1 litre per square metre = 1 mm:
    R(mm/h) = 60 * V(L) / (A(m2) * T(min))
    """
    if collector_area_m2 <= 0 or collection_time_min <= 0:
        raise ValueError("Collector area and collection time must be positive.")
    if volume_litre < 0:
        raise ValueError("Collected volume cannot be negative.")
    return 60.0 * volume_litre / (collector_area_m2 * collection_time_min)


# ============================================================
# D. MODULE 2 - REDUCED-ORDER SOIL MOISTURE RESPONSE
# ============================================================

def simulate_soil_moisture(
    rainfall_mm_h,
    dt_min,
    theta_initial,
    theta_min,
    theta_max,
    rainfall_reference_mm_h,
    k_in_per_min,
    k_d_per_min,
):
    """
    d(theta)/dt =
       kin * (R/Rref) * (theta_max - theta)
       - kd * (theta - theta_min)

    Explicit Euler integration.
    """
    if not (0 <= theta_min < theta_max <= 1):
        raise ValueError("Require 0 <= theta_min < theta_max <= 1.")
    if not (theta_min <= theta_initial <= theta_max):
        raise ValueError("theta_initial must be between theta_min and theta_max.")
    if rainfall_reference_mm_h <= 0 or dt_min <= 0:
        raise ValueError("Reference rainfall and time step must be positive.")
    if k_in_per_min < 0 or k_d_per_min < 0:
        raise ValueError("kin and kd must be non-negative.")

    theta = np.zeros(len(rainfall_mm_h), dtype=float)
    theta[0] = theta_initial

    normalized_rain = rainfall_mm_h / rainfall_reference_mm_h

    for i in range(1, len(rainfall_mm_h)):
        theta_prev = theta[i - 1]
        r_now = normalized_rain[i - 1]

        dtheta_dt = (
            k_in_per_min * r_now * (theta_max - theta_prev)
            - k_d_per_min * (theta_prev - theta_min)
        )

        theta[i] = np.clip(
            theta_prev + dt_min * dtheta_dt,
            theta_min,
            theta_max
        )

    return theta


# ============================================================
# E. RUN ONE SCENARIO
# ============================================================

def run_scenario(scenario_id, scenario):
    rain_start = scenario["rain_start_min"]
    rain_duration = scenario["duration_min"]
    total_time = rain_start + rain_duration + POST_RAIN_OBSERVATION_MIN

    time_min = np.arange(0.0, total_time + DT_MIN, DT_MIN)

    rainfall = generate_rainfall(
        time_min=time_min,
        intensity_mm_h=scenario["intensity_mm_h"],
        start_min=rain_start,
        duration_min=rain_duration,
    )

    cumulative_rain = calculate_cumulative_rainfall(rainfall, DT_MIN)

    theta = simulate_soil_moisture(
        rainfall_mm_h=rainfall,
        dt_min=DT_MIN,
        theta_initial=scenario["theta_initial"],
        theta_min=THETA_MIN,
        theta_max=THETA_MAX,
        rainfall_reference_mm_h=R_REF_MM_H,
        k_in_per_min=K_IN_PER_MIN,
        k_d_per_min=K_D_PER_MIN,
    )

    end_rain_min = rain_start + rain_duration
    moisture_rate = np.gradient(theta, DT_MIN)

    df = pd.DataFrame({
        "scenario_id": scenario_id,
        "time_min": time_min,
        "rainfall_mm_h": rainfall,
        "cumulative_rainfall_mm": cumulative_rain,
        "moisture_vwc_m3_m3": theta,
        "moisture_percent_vwc": theta * 100.0,
        "moisture_rate_per_min": moisture_rate,
        "rain_start_min": rain_start,
        "rain_end_min": end_rain_min,
        "rain_duration_min": rain_duration,
        "rain_intensity_setting_mm_h": scenario["intensity_mm_h"],
    })

    return df


# ============================================================
# F. VALIDATION AND SUMMARY
# ============================================================

def validate_scenario(df):
    assert df["rainfall_mm_h"].ge(0).all()
    assert df["cumulative_rainfall_mm"].is_monotonic_increasing
    assert df["moisture_vwc_m3_m3"].between(THETA_MIN, THETA_MAX).all()
    assert np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all()

    # Check integrated rainfall against R * duration / 60
    expected = (
        df["rain_intensity_setting_mm_h"].iloc[0]
        * df["rain_duration_min"].iloc[0] / 60.0
    )
    actual = df["cumulative_rainfall_mm"].iloc[-1]
    if not np.isclose(expected, actual, atol=1e-8):
        raise AssertionError(
            f"Rainfall total mismatch: expected {expected}, got {actual}"
        )


def create_summary(all_data):
    rows = []
    for scenario_id, df in all_data.items():
        rain_total = float(df["cumulative_rainfall_mm"].iloc[-1])
        idx_peak = int(df["moisture_vwc_m3_m3"].idxmax())
        rows.append({
            "scenario_id": scenario_id,
            "rain_intensity_mm_h": float(df["rain_intensity_setting_mm_h"].iloc[0]),
            "rain_duration_min": float(df["rain_duration_min"].iloc[0]),
            "rain_total_mm": rain_total,
            "moisture_initial_percent": float(df["moisture_percent_vwc"].iloc[0]),
            "moisture_peak_percent": float(df.loc[idx_peak, "moisture_percent_vwc"]),
            "time_of_moisture_peak_min": float(df.loc[idx_peak, "time_min"]),
            "moisture_final_percent": float(df["moisture_percent_vwc"].iloc[-1]),
        })
    return pd.DataFrame(rows)


# ============================================================
# G. PLOTS
# ============================================================

def plot_results(all_data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: rainfall input, six experiments
    fig, ax = plt.subplots(figsize=(11, 5))
    for scenario_id, df in all_data.items():
        ax.step(df["time_min"], df["rainfall_mm_h"],
                where="post", label=scenario_id)
    ax.set_xlabel("Time (minute)")
    ax.set_ylabel("Rainfall intensity (mm/hour)")
    ax.set_title("Module 1 - Six constant-intensity rainfall scenarios")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "01_rainfall_6_scenarios.png", dpi=180)
    plt.close(fig)

    # Plot 2: moisture response, six experiments
    fig, ax = plt.subplots(figsize=(11, 5))
    for scenario_id, df in all_data.items():
        ax.plot(df["time_min"], df["moisture_percent_vwc"],
                label=scenario_id)
    ax.set_xlabel("Time (minute)")
    ax.set_ylabel("Volumetric water content (%)")
    ax.set_title("Module 2 - Soil moisture response")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "02_moisture_6_scenarios.png", dpi=180)
    plt.close(fig)

    # One dual-axis chart per experiment, use different line styles
    for scenario_id, df in all_data.items():
        fig, ax1 = plt.subplots(figsize=(10, 4.8))
        line1 = ax1.step(
            df["time_min"], df["rainfall_mm_h"],
            where="post", linestyle="--", label="Rainfall (mm/h)"
        )
        ax1.set_xlabel("Time (minute)")
        ax1.set_ylabel("Rainfall intensity (mm/hour)")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        line2 = ax2.plot(
            df["time_min"], df["moisture_percent_vwc"],
            linewidth=2, label="Moisture (% VWC)"
        )
        ax2.set_ylabel("Volumetric water content (%)")

        lines = line1 + line2
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="best")
        ax1.axvline(df["rain_start_min"].iloc[0], linestyle=":", alpha=0.7)
        ax1.axvline(df["rain_end_min"].iloc[0], linestyle=":", alpha=0.7)
        ax1.set_title(f"{scenario_id} - Rainfall and moisture")
        fig.tight_layout()
        fig.savefig(output_dir / f"{scenario_id}_response.png", dpi=180)
        plt.close(fig)


# ============================================================
# H. MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = build_scenarios()

    all_data = {}
    frames = []

    print("SIX-SCENARIO EXPERIMENT MATRIX")
    print("-" * 78)

    for scenario_id, scenario in scenarios.items():
        df = run_scenario(scenario_id, scenario)
        validate_scenario(df)
        all_data[scenario_id] = df
        frames.append(df)

        print(
            f"{scenario_id:28s} | "
            f"R={scenario['intensity_mm_h']:5.1f} mm/h | "
            f"T={scenario['duration_min']:4.0f} min | "
            f"P={df['cumulative_rainfall_mm'].iloc[-1]:6.2f} mm | "
            f"theta_peak={df['moisture_percent_vwc'].max():5.2f}%"
        )

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(
        OUTPUT_DIR / "module_1_2_all_scenarios.csv",
        index=False,
        encoding="utf-8-sig"
    )

    for scenario_id, df in all_data.items():
        df.to_csv(
            OUTPUT_DIR / f"{scenario_id}.csv",
            index=False,
            encoding="utf-8-sig"
        )

    summary = create_summary(all_data)
    summary.to_csv(
        OUTPUT_DIR / "experiment_matrix_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    plot_results(all_data, OUTPUT_DIR)

    print("\nSimulation completed successfully.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print("Main files:")
    print("- module_1_2_all_scenarios.csv")
    print("- experiment_matrix_summary.csv")
    print("- six individual scenario CSV files")
    print("- rainfall and moisture plots")


if __name__ == "__main__":
    main()
