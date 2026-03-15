# UACR Variation Analysis (from provided sheet values)

Formula used by app (`app/utils.py`):

`UACR (mg/g) = 100 * Albumin (mg/L) / Creatinine (mg/dL)`

## Summary

- Rows analyzed: **22**
- MAE between `UACR Value` column and formula output: **320.67 mg/g**
- MAE between `Correct Value` and formula output: **0.0038 mg/g**
- Exact matches (`Correct Value` vs formula, rounded 2dp): **20/22**
- Max absolute error in `UACR Value` vs formula: **1336.39 mg/g**

Observation: `Correct Value` is effectively the direct formula output (minor rounding differences in two rows), while `UACR Value` appears to be produced from different inputs than the displayed Albumin/Creatinine pair.

## Rows with largest variation (`UACR Value` - formula)

| Row | Albumin | Creatinine | UACR Value | Formula | Delta | Correct Value |
|---:|---:|---:|---:|---:|---:|---:|
| 13 | 10 | 50 | 1356.39 | 20.00 | +1336.39 | 20.00 |
| 7 | 80 | 50 | 1418.34 | 160.00 | +1258.34 | 160.00 |
| 9 | 3 | 150 | 683.53 | 2.00 | +681.53 | 2.00 |
| 18 | 10 | 100 | 631.10 | 10.00 | +621.10 | 10.00 |
| 21 | 30 | 100 | 609.41 | 30.00 | +579.41 | 30.00 |
| 17 | 3 | 100 | 578.00 | 3.00 | +575.00 | 3.00 |
| 16 | 30 | 100 | 564.76 | 30.00 | +534.76 | 30.00 |
| 22 | 150 | 100 | 602.59 | 150.00 | +452.59 | 150.00 |

## Likely cause of variation

- In the app pipeline, albumin and creatinine are color-inferred and then snapped/overridden by chart bins for display.
- If UACR is computed from **pre-snap continuous model outputs** but the sheet compares against UACR derived from **snapped/displayed values**, mismatches like these occur.
- Some rows show extreme drift (e.g., Albumin=3 and Creatinine=150 should yield ~2, but sheet UACR is 683.53), which indicates stale/wrong variables may be feeding UACR calculation in that workflow.

## Recommended fix

1. Compute UACR from the exact values shown to user (post-snap values), not from hidden pre-snap values.
2. Add a consistency check in code and logs: `abs(uacr - 100*A/C) < tolerance` with the same A/C values rendered in UI/report.
3. In exported sheets, include explicit columns: `Albumin_used_for_uacr`, `Creatinine_used_for_uacr`, `UACR_formula`, `UACR_display` for traceability.
4. Standardize rounding (e.g., compute with full precision and round only at final display to 2dp).