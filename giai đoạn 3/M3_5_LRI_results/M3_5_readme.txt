M3-5 – LANDSLIDE RISK INDEX (LRI), SYNTHETIC EXPLORATORY VERSION
Input: Giai_doan_1_Synthetic_Dataset_6_kich_ban.csv (900 rows, 6 scenarios).
Components:
1) moisture_percent_vwc
2) pore_pressure_positive_kpa_synthetic
3) c2_dynamic_rms_g_synthetic (one representative C2 motion feature; gyro and tilt excluded to reduce redundant counting)
Thresholds: midpoint between M5-B state medians for adjacent pairs STABLE/WATER_ACCUMULATION, WATER_ACCUMULATION/DEFORMATION, DEFORMATION/SLIDING.
Normalization: piecewise linear interpolation using [observed minimum, T1, T2, T3] -> [0, 1/3, 2/3, 1], clamped outside range.
Weights: equal 1/3 each, selected as a transparent provisional baseline, not statistically optimized or physically calibrated.
LRI = 100 * mean(component scores).
Exploratory levels: LOW <25; MODERATE 25–<50; HIGH 50–<75; VERY_HIGH >=75.
Critical limitations:
- M5-B labels are synthetic rule-generated labels. Thresholds derived from these labels are not independent ground truth; using them in LRI creates circular dependence when comparing LRI to M5-B labels.
- C2 features are generated from pressure-coupled hypothesis branch and are not independent physical sensor evidence.
- Only one scenario contains SLIDING, so high-risk/sliding calibration is weak.
- Equal weights and LRI category bands are provisional design choices, not validated.
- This LRI is for synthetic-model analysis and method development only, not field warning/deployment.
- For experimental validation, calculate LRI with fixed formula/thresholds only after separately defining calibration from physical measurements; do not claim validation from this synthetic dataset.
