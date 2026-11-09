M3-4 – SENSOR-SPECIFIC CANDIDATE THRESHOLDS
Source: Giai_doan_1_Synthetic_Dataset_6_kich_ban.csv (900 rows).
Primary labels: event_label / M5-B, rule-generated.
State counts are imbalanced and SLIDING occurs in only one scenario.
For each sensor and adjacent ordinal state pair:
- candidate threshold = midpoint of the two state medians;
- trend direction determined by the ordering of medians;
- ordered AUC summarizes how well the variable ranks the higher state above lower state (direction-adjusted);
- IQR overlap ratio is a descriptive overlap indicator.
These are exploratory candidate thresholds, not validated alarm thresholds.
The labels are produced by M5-B rules, so thresholds estimated against them partly recover the rules used to create the labels (circularity).
C2 features are pressure-coupled synthetic outputs, not independent sensor evidence.
M4-A1/A2 are reference features and are not pressure-coupled.
Only one scenario contains SLIDING; the DEFORMATION-to-SLIDING threshold is especially weak and should not be treated as generalizable.
No threshold is recommended for operational deployment from this dataset alone.
