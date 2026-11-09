M3-1 – PEARSON AND SPEARMAN CORRELATION
Source: Giai_doan_1_Synthetic_Dataset_6_kich_ban.csv
Data: 900 rows, 6 rainfall scenarios, 21 selected numeric variables.

Method:
- Pearson r: linear association.
- Spearman rho: rank/monotonic association.
- Correlations are calculated for all 900 rows and separately within each scenario.
- P-values are descriptive only. The time-series rows are autocorrelated and are not independent observations; therefore p-values should not be interpreted as conventional independent-sample evidence.
- Constant variables or pairs with a constant variable yield undefined correlation (NaN).

Important interpretation:
- The C2 features are synthetic features generated from the pressure-driven coupling assumption. Their high correlation with pressure is structurally expected and is not independent experimental evidence.
- M4-A1/A2 reference IMU features are a separate branch without imposed causal coupling to pore pressure. They should not be expected to correlate strongly with pressure.
- Pooled correlations can be affected by scenario differences and common time trends. Per-scenario results must be considered.
- Correlation does not establish causation.
- These are synthetic-model findings, not experimentally validated sensor relationships or alarm thresholds.

Selected pooled results:
- cumulative_rainfall_mm vs moisture_percent_vwc: Pearson=0.940583; Spearman=0.941952; n=900
- moisture_percent_vwc vs delayed_moisture_percent: Pearson=0.961554; Spearman=0.957238; n=900
- delayed_moisture_percent vs pore_pressure_positive_kpa_synthetic: Pearson=0.748084; Spearman=0.823981; n=900
- wetness_drive_0_1 vs pore_pressure_positive_kpa_synthetic: Pearson=0.738961; Spearman=0.831587; n=900
- pore_pressure_positive_kpa_synthetic vs c2_motion_activation_0_1: Pearson=0.977652; Spearman=0.965993; n=900
- c2_motion_activation_0_1 vs c2_dynamic_rms_g_synthetic: Pearson=0.964955; Spearman=1.000000; n=900
- c2_motion_activation_0_1 vs c2_gyro_rms_deg_s_synthetic: Pearson=0.965620; Spearman=1.000000; n=900
- c2_motion_activation_0_1 vs c2_tilt_deg_synthetic: Pearson=0.981015; Spearman=1.000000; n=900
- pore_pressure_positive_kpa_synthetic vs m4_dynamic_rms_g: Pearson=0.186363; Spearman=0.125479; n=900
- pore_pressure_positive_kpa_synthetic vs m4_gyro_rms_deg_s: Pearson=0.077094; Spearman=0.086328; n=900
- pore_pressure_positive_kpa_synthetic vs m4_tilt_reference_mean_deg: Pearson=0.000374; Spearman=0.031832; n=900
- rainfall_mm_h vs moisture_percent_vwc: Pearson=-0.021552; Spearman=-0.076390; n=900
