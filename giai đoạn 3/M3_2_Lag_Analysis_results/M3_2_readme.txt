M3-2 – LAG ANALYSIS
Source: Giai_doan_1_Synthetic_Dataset_6_kich_ban.csv (900 rows, 6 scenarios).
Convention: lag > 0 means X leads Y by lag minutes, using corr(X[t],Y[t+lag]).
Both raw levels and first differences are analyzed separately, per scenario.
The selected best lag maximizes absolute Pearson correlation within the stated search window.
This is exploratory: maximum cross-correlation is not automatically a physical response time.
Level-series correlations may be inflated by shared trends and autocorrelation. First-difference analysis helps assess co-movement but can be noisy.
Operational onset times use explicit computational thresholds:
- moisture onset: moisture exceeds its first-row value by 0.02 percentage points;
- positive pressure onset: pore pressure > 0.01 kPa;
- C2 activation onset: activation > 0.01.
These are analysis definitions, not experimentally validated thresholds.
C2 is synthetically generated from pressure with an imposed 2-minute lag; results are structurally dependent and not independent physical validation.
M4-A1/A2 reference features are a separate branch and are not pressure-coupled.
Minute-level time series are autocorrelated; p-values are not interpreted as independent-sample evidence.
