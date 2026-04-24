# Albumin extraction issue: analysis and recommended handling

## What the existing analysis shows

From `analysis/uacr_variation_analysis.md`:
- `Correct Value` almost perfectly matches formula output from displayed Albumin/Creatinine values.
- `UACR Value` in the sheet has very large deviations (MAE ~320.67 mg/g; max error 1336.39 mg/g).
- This pattern strongly suggests a value lineage mismatch (UACR being computed from variables different from displayed analyte values).

## Current pipeline behavior (important for debugging)

In `app/utils.py`, the current flow for albumin (microalbumin/pod2) is:
1. Segment pod mask from model output.
2. White-balance image.
3. Extract robust pod color using eroded-mask median RGB.
4. Predict continuous value via regression and chart-based estimate.
5. Apply quality corrections and low-end snapping.
6. Compute corrected UACR from snapped/displayed values.

The code already computes both traces:
- legacy continuous trace (`uacr_legacy_trace`)
- corrected snapped/displayed trace (`uacr_corrected_trace`)

This is good because it provides direct observability for drift introduced by extraction or post-processing.

## Root cause pattern for albumin instability

Albumin extraction drift usually comes from one or more of these:
1. **Mask quality error**: boundary leakage, tiny mask, border-touching mask, or elongated mask.
2. **Photometric instability**: glare/specular spots and non-uniform illumination causing chroma instability.
3. **Label-space mismatch**: ground truth label tied to chart-snapped class while model is trained/validated on continuous or vice versa.
4. **Pipeline inconsistency**: using raw/continuous value for one stage and snapped/display value for another (especially for UACR formula).

## Recommended handling for ground-truth and color-extraction changes

### 1) Freeze value lineage contract
Always store and compare the following per sample:
- `raw_regression_value`
- `raw_color_chart_value`
- `final_display_value` (snapped)
- `uacr_formula_from_display_values`

Treat `final_display_value` as the only source for displayed UACR.

### 2) Version your extraction protocol
When changing extraction (e.g., new erosion kernel, white-balance, glare filtering), increment a protocol version field, e.g.:
- `extraction_protocol_version = v2.1`
- `segmentation_model_version`
- `color_centroid_version`

Never mix old and new protocol outputs in one evaluation split without stratifying by version.

### 3) Maintain two explicit targets for albumin
- **Continuous target** for regression calibration.
- **Discrete chart target** for displayed bins.

Evaluate both:
- continuous MAE/RMSE
- bin accuracy + adjacent-bin tolerance accuracy

### 4) Rebuild centroids only from QC-passed masks
For centroid updates, include only samples where:
- mask quality flags are clean,
- glare flags are below threshold,
- confidence is moderate/high.

This prevents contaminated centroid drift.

### 5) Add automatic drift gates in CI/validation
Fail validation if any of these regress:
- UACR consistency delta median or p95 exceeds threshold
- albumin bin confusion increases above tolerance
- fraction of low-confidence albumin predictions rises unexpectedly

### 6) Calibration strategy when segmentation changes
After segmentation model update:
1. Re-run extraction on calibration dataset.
2. Refit microalbumin regressor (or at minimum re-calibrate with isotonic/Platt-like mapping).
3. Re-estimate centroid references from refreshed features.
4. Recompute low-end snapping thresholds from new feature distributions.

## Practical recommendation for your issue now

Given the analysis results, prioritize in this order:
1. Confirm every reported UACR in outputs/export sheets is derived from snapped/displayed analyte values only.
2. Audit albumin samples with largest UACR deltas for mask/glare/edge-touch flags.
3. Recompute albumin centroid table from QC-filtered set after segmentation changes.
4. Re-validate against both continuous and discrete targets before release.

This sequence addresses both immediate correctness (UACR mismatch) and medium-term robustness (albumin color extraction drift after segmentation updates).
