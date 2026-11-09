"""
MODULE 3 - Pore Water Pressure Simulation
Reads the combined CSV produced by Module 1 + Module 2 (version 2).

Important scientific scope:
- This is a phenomenological / reduced-order synthetic model.
- It does NOT solve Richards' equation.
- It does NOT calculate pore pressure from soil strength or hydraulic
  properties.
- The kPa scale and all response coefficients are illustrative assumptions.
  They must be calibrated against a real pore-pressure sensor before
  interpreting the output as physical measurements.

Input:
  module_1_2_all_scenarios.csv

Output:
  results_module_3/
    module_1_2_3_all_scenarios.csv
    one CSV per scenario
    module_3_summary.csv
    plots
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# A. CONFIGURATION
# ============================================================

# Input CSV from Module 1 + Module 2.
# Put this script in the same folder as the input CSV, or change this path.
INPUT_CSV = (
    Path(__file__).resolve().parent
    / "results_module_1_2_v2"
    / "module_1_2_all_scenarios.csv"
)

OUTPUT_DIR = Path(__file__).resolve().parent / "results_module_3"

# Numerical time step (minutes). The current Module 1 + 2 output uses 1 min.
DT_MIN = 1.0

# Initial conditions
INITIAL_DELAYED_MOISTURE = None  # None -> use first moisture value in each scenario
INITIAL_POSITIVE_PRESSURE_KPA = 0.0

# Illustrative phenomenological parameters (NOT measured soil properties)
THETA_CRITICAL = 0.18       # m3/m3; moisture activation point (illustrative)
THETA_MAX_MODEL = 0.45      # m3/m3; consistent with Module 2 theta upper bound
DELAY_TIME_MIN = 8.0        # minutes; response lag
PRESSURE_GENERATION_RATE = 0.18  # kPa/min at normalized wetness = 1
PRESSURE_RELAXATION_RATE = 0.008 # 1/min
PRESSURE_CAP_KPA = 8.0      # illustrative output scale only


# ============================================================
# B. INPUT VALIDATION
# ============================================================

REQUIRED_COLUMNS = [
    "scenario_id",
    "time_min",
    "rainfall_mm_h",
    "cumulative_rainfall_mm",
    "moisture_vwc_m3_m3",
    "moisture_percent_vwc",
    "rain_start_min",
    "rain_end_min",
    "rain_duration_min",
    "rain_intensity_setting_mm_h",
]


def load_input(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Place the Module 1+2 combined CSV "
            "in the same folder as this script, or edit INPUT_CSV."
        )

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    if df.empty:
        raise ValueError("Input CSV is empty.")

    for col in [
        "time_min", "rainfall_mm_h", "cumulative_rainfall_mm",
        "moisture_vwc_m3_m3", "moisture_percent_vwc",
    ]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    return df


# ============================================================
# C. MODULE 3 - DELAYED MOISTURE AND POSITIVE PORE PRESSURE
# ============================================================

def simulate_pressure_for_scenario(group):
    """
    Two coupled first-order states:

    1) Delayed moisture:
       d(theta_delay)/dt = (theta - theta_delay) / tau

    2) Positive pore-pressure proxy:
       du/dt = kg * wetness_drive * (1 - u/u_cap) - kr * u

       wetness_drive = clip(
           (theta_delay - theta_critical) /
           (theta_max_model - theta_critical), 0, 1
       )

    Forward Euler integration, with physical bounds imposed on the
    synthetic state variables.
    """
    g = group.sort_values("time_min").copy().reset_index(drop=True)

    time = g["time_min"].to_numpy(dtype=float)
    theta = g["moisture_vwc_m3_m3"].to_numpy(dtype=float)

    if len(g) < 2:
        raise ValueError("Each scenario must contain at least two time points.")
    if not np.all(np.diff(time) > 0):
        raise ValueError("time_min must be strictly increasing per scenario.")
    if not np.isfinite(theta).all():
        raise ValueError("Moisture data contains NaN or infinite values.")
    if DELAY_TIME_MIN <= 0:
        raise ValueError("DELAY_TIME_MIN must be positive.")
    if THETA_MAX_MODEL <= THETA_CRITICAL:
        raise ValueError("THETA_MAX_MODEL must exceed THETA_CRITICAL.")
    if PRESSURE_CAP_KPA <= 0:
        raise ValueError("PRESSURE_CAP_KPA must be positive.")

    theta_delay = np.zeros(len(g), dtype=float)
    pressure = np.zeros(len(g), dtype=float)
    wetness_drive = np.zeros(len(g), dtype=float)
    pressure_generation = np.zeros(len(g), dtype=float)
    pressure_relaxation = np.zeros(len(g), dtype=float)

    theta_delay[0] = (
        theta[0] if INITIAL_DELAYED_MOISTURE is None
        else float(INITIAL_DELAYED_MOISTURE)
    )
    pressure[0] = INITIAL_POSITIVE_PRESSURE_KPA

    for i in range(1, len(g)):
        dt = time[i] - time[i - 1]

        # Lagged moisture state
        dtheta_delay = (
            (theta[i - 1] - theta_delay[i - 1]) / DELAY_TIME_MIN
        )
        theta_delay[i] = np.clip(
            theta_delay[i - 1] + dt * dtheta_delay,
            0.0,
            THETA_MAX_MODEL
        )

        # Normalized moisture excess, dimensionless
        wetness_drive[i - 1] = np.clip(
            (theta_delay[i - 1] - THETA_CRITICAL)
            / (THETA_MAX_MODEL - THETA_CRITICAL),
            0.0,
            1.0
        )

        # Pressure generation and relaxation
        generation = (
            PRESSURE_GENERATION_RATE
            * wetness_drive[i - 1]
            * (1.0 - pressure[i - 1] / PRESSURE_CAP_KPA)
        )
        relaxation = PRESSURE_RELAXATION_RATE * pressure[i - 1]

        dpressure = generation - relaxation
        pressure[i] = np.clip(
            pressure[i - 1] + dt * dpressure,
            0.0,
            PRESSURE_CAP_KPA
        )

        pressure_generation[i - 1] = generation
        pressure_relaxation[i - 1] = relaxation

    # Calculate final-row diagnostics
    wetness_drive[-1] = np.clip(
        (theta_delay[-1] - THETA_CRITICAL)
        / (THETA_MAX_MODEL - THETA_CRITICAL),
        0.0,
        1.0
    )
    pressure_generation[-1] = (
        PRESSURE_GENERATION_RATE
        * wetness_drive[-1]
        * (1.0 - pressure[-1] / PRESSURE_CAP_KPA)
    )
    pressure_relaxation[-1] = PRESSURE_RELAXATION_RATE * pressure[-1]

    g["delayed_moisture_vwc_m3_m3"] = theta_delay
    g["delayed_moisture_percent"] = theta_delay * 100.0
    g["wetness_drive_0_1"] = wetness_drive
    g["pore_pressure_positive_kpa_synthetic"] = pressure
    g["pressure_generation_rate_kpa_min"] = pressure_generation
    g["pressure_relaxation_rate_kpa_min"] = pressure_relaxation

    # This flag only indicates that the model's moisture activation
    # condition is met. It is NOT a landslide event label.
    g["pressure_generation_active"] = wetness_drive > 0.0

    return g


# ============================================================
# D. SCENARIO SUMMARY
# ============================================================

def make_summary(all_df):
    rows = []

    for scenario_id, g in all_df.groupby("scenario_id", sort=False):
        g = g.sort_values("time_min").reset_index(drop=True)

        i_peak = int(g["pore_pressure_positive_kpa_synthetic"].idxmax())
        i_moisture_peak = int(g["moisture_vwc_m3_m3"].idxmax())

        rain_end = float(g["rain_end_min"].iloc[0])
        post_rain = g[g["time_min"] >= rain_end]

        if len(post_rain):
            post_peak_idx = int(
                post_rain["pore_pressure_positive_kpa_synthetic"].idxmax()
            )
            post_peak_time = float(g.loc[post_peak_idx, "time_min"])
            post_peak_pressure = float(
                g.loc[post_peak_idx, "pore_pressure_positive_kpa_synthetic"]
            )
        else:
            post_peak_time = np.nan
            post_peak_pressure = np.nan

        rows.append({
            "scenario_id": scenario_id,
            "rain_intensity_mm_h": float(g["rain_intensity_setting_mm_h"].iloc[0]),
            "rain_duration_min": float(g["rain_duration_min"].iloc[0]),
            "rain_total_mm": float(g["cumulative_rainfall_mm"].iloc[-1]),
            "moisture_initial_percent": float(g["moisture_percent_vwc"].iloc[0]),
            "moisture_peak_percent": float(g.loc[i_moisture_peak, "moisture_percent_vwc"]),
            "moisture_peak_time_min": float(g.loc[i_moisture_peak, "time_min"]),
            "pressure_peak_synthetic_kpa": float(
                g.loc[i_peak, "pore_pressure_positive_kpa_synthetic"]
            ),
            "pressure_peak_time_min": float(g.loc[i_peak, "time_min"]),
            "pressure_at_rain_end_synthetic_kpa": float(
                g.loc[(g["time_min"] - rain_end).abs().idxmin(),
                      "pore_pressure_positive_kpa_synthetic"]
            ),
            "post_rain_pressure_peak_synthetic_kpa": post_peak_pressure,
            "post_rain_pressure_peak_time_min": post_peak_time,
            "pressure_final_synthetic_kpa": float(
                g["pore_pressure_positive_kpa_synthetic"].iloc[-1]
            ),
        })

    return pd.DataFrame(rows)


# ============================================================
# E. PLOTS
# ============================================================

def plot_all_scenarios(all_df, output_dir):
    # Pressure response for all scenarios
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for scenario_id, g in all_df.groupby("scenario_id", sort=False):
        ax.plot(
            g["time_min"],
            g["pore_pressure_positive_kpa_synthetic"],
            label=scenario_id
        )
    ax.set_xlabel("Time (minute)")
    ax.set_ylabel("Synthetic positive pore pressure (kPa)")
    ax.set_title("Module 3 - Positive pore pressure response (synthetic)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "03_pressure_all_scenarios.png", dpi=180)
    plt.close(fig)

    # Moisture, delayed moisture and pressure per scenario
    for scenario_id, g in all_df.groupby("scenario_id", sort=False):
        fig, ax1 = plt.subplots(figsize=(10, 5))
        l1 = ax1.plot(
            g["time_min"], g["moisture_percent_vwc"],
            label="Soil moisture (% VWC)"
        )
        l2 = ax1.plot(
            g["time_min"], g["delayed_moisture_percent"],
            linestyle="--", label="Delayed moisture (%)"
        )
        ax1.set_xlabel("Time (minute)")
        ax1.set_ylabel("Moisture (%)")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        l3 = ax2.plot(
            g["time_min"],
            g["pore_pressure_positive_kpa_synthetic"],
            linewidth=2,
            label="Synthetic positive pore pressure (kPa)"
        )
        ax2.set_ylabel("Synthetic pore pressure (kPa)")

        rain_start = float(g["rain_start_min"].iloc[0])
        rain_end = float(g["rain_end_min"].iloc[0])
        ax1.axvline(rain_start, linestyle=":", alpha=0.7)
        ax1.axvline(rain_end, linestyle=":", alpha=0.7)

        lines = l1 + l2 + l3
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="best", fontsize=8)
        ax1.set_title(f"{scenario_id} - Moisture and pore-pressure response")
        fig.tight_layout()
        fig.savefig(output_dir / f"{scenario_id}_pressure_response.png", dpi=180)
        plt.close(fig)

    # Pressure vs moisture scatter: shows the simulated relationship/hysteresis
    fig, ax = plt.subplots(figsize=(8, 6))
    for scenario_id, g in all_df.groupby("scenario_id", sort=False):
        ax.plot(
            g["moisture_percent_vwc"],
            g["pore_pressure_positive_kpa_synthetic"],
            marker=".",
            markersize=2,
            linewidth=1,
            label=scenario_id
        )
    ax.set_xlabel("Soil moisture (% VWC)")
    ax.set_ylabel("Synthetic positive pore pressure (kPa)")
    ax.set_title("Module 3 - Moisture / pore-pressure relationship")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "04_moisture_pressure_relationship.png", dpi=180)
    plt.close(fig)


# ============================================================
# F. MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source = load_input(INPUT_CSV)
    scenario_frames = []

    for scenario_id, group in source.groupby("scenario_id", sort=False):
        result = simulate_pressure_for_scenario(group)
        scenario_frames.append(result)

    all_df = pd.concat(scenario_frames, ignore_index=True)

    numeric = all_df.select_dtypes(include=[np.number]).to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError("Output contains NaN or infinite numeric values.")

    if not all_df["pore_pressure_positive_kpa_synthetic"].between(
        0.0, PRESSURE_CAP_KPA
    ).all():
        raise AssertionError("Synthetic pressure exceeded configured bounds.")

    all_df.to_csv(
        OUTPUT_DIR / "module_1_2_3_all_scenarios.csv",
        index=False,
        encoding="utf-8-sig"
    )

    for scenario_id, group in all_df.groupby("scenario_id", sort=False):
        safe_name = str(scenario_id).replace("/", "_")
        group.to_csv(
            OUTPUT_DIR / f"{safe_name}_module3.csv",
            index=False,
            encoding="utf-8-sig"
        )

    summary = make_summary(all_df)
    summary.to_csv(
        OUTPUT_DIR / "module_3_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    plot_all_scenarios(all_df, OUTPUT_DIR)

    print("MODULE 3 - PORE PRESSURE SIMULATION")
    print("=" * 100)
    for _, row in summary.iterrows():
        print(
            f"{row['scenario_id']:24s} | "
            f"R={row['rain_intensity_mm_h']:4.0f} mm/h | "
            f"T={row['rain_duration_min']:2.0f} min | "
            f"theta_peak={row['moisture_peak_percent']:5.2f}% | "
            f"u_peak={row['pressure_peak_synthetic_kpa']:6.3f} kPa "
            f"at t={row['pressure_peak_time_min']:5.1f} min"
        )

    print("\nCompleted.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print("Main files:")
    print("- module_1_2_3_all_scenarios.csv")
    print("- module_3_summary.csv")
    print("- one Module 3 CSV per scenario")
    print("- pressure and moisture-pressure plots")
    print("\nWARNING: pressure values are synthetic and not calibrated physical kPa.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 3 pore-pressure simulation")
    parser.add_argument("--input", type=Path, default=INPUT_CSV,
                        help="Module 1+2 combined CSV")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR,
                        help="Module 3 output directory")
    args = parser.parse_args()
    INPUT_CSV = args.input
    OUTPUT_DIR = args.output
    main()
