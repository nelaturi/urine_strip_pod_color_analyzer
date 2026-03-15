import csv
from pathlib import Path

INPUT_CSV = Path('analysis/uacr_input_sample.csv')
OUTPUT_MD = Path('analysis/uacr_variation_analysis.md')


def main() -> None:
    rows = []
    with INPUT_CSV.open() as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})

    for r in rows:
        r['uacr_formula'] = 100.0 * r['albumin_mg_l'] / r['creatinine_mg_dl']
        r['sheet_minus_formula'] = r['uacr_value_in_sheet_mg_g'] - r['uacr_formula']
        r['correct_minus_formula'] = r['correct_value'] - r['uacr_formula']

    n = len(rows)
    mae_sheet = sum(abs(r['sheet_minus_formula']) for r in rows) / n
    mae_correct = sum(abs(r['correct_minus_formula']) for r in rows) / n
    max_abs = max(abs(r['sheet_minus_formula']) for r in rows)
    formula_matches_correct = sum(round(r['uacr_formula'], 2) == round(r['correct_value'], 2) for r in rows)

    largest = sorted(
        enumerate(rows, start=1),
        key=lambda t: abs(t[1]['sheet_minus_formula']),
        reverse=True,
    )[:8]

    lines = []
    lines.append('# UACR Variation Analysis (from provided sheet values)\n')
    lines.append('Formula used by app (`app/utils.py`):\n')
    lines.append('`UACR (mg/g) = 100 * Albumin (mg/L) / Creatinine (mg/dL)`\n')
    lines.append('## Summary\n')
    lines.append(f'- Rows analyzed: **{n}**')
    lines.append(f'- MAE between `UACR Value` column and formula output: **{mae_sheet:.2f} mg/g**')
    lines.append(f'- MAE between `Correct Value` and formula output: **{mae_correct:.4f} mg/g**')
    lines.append(f'- Exact matches (`Correct Value` vs formula, rounded 2dp): **{formula_matches_correct}/{n}**')
    lines.append(f'- Max absolute error in `UACR Value` vs formula: **{max_abs:.2f} mg/g**\n')
    lines.append('Observation: `Correct Value` is effectively the direct formula output (minor rounding differences in two rows), while `UACR Value` appears to be produced from different inputs than the displayed Albumin/Creatinine pair.\n')
    lines.append('## Rows with largest variation (`UACR Value` - formula)\n')
    lines.append('| Row | Albumin | Creatinine | UACR Value | Formula | Delta | Correct Value |')
    lines.append('|---:|---:|---:|---:|---:|---:|---:|')
    for i, r in largest:
        lines.append(
            f"| {i} | {r['albumin_mg_l']:.0f} | {r['creatinine_mg_dl']:.0f} | "
            f"{r['uacr_value_in_sheet_mg_g']:.2f} | {r['uacr_formula']:.2f} | "
            f"{r['sheet_minus_formula']:+.2f} | {r['correct_value']:.2f} |"
        )

    lines.append('\n## Likely cause of variation\n')
    lines.append('- In the app pipeline, albumin and creatinine are color-inferred and then snapped/overridden by chart bins for display.')
    lines.append('- If UACR is computed from **pre-snap continuous model outputs** but the sheet compares against UACR derived from **snapped/displayed values**, mismatches like these occur.')
    lines.append('- Some rows show extreme drift (e.g., Albumin=3 and Creatinine=150 should yield ~2, but sheet UACR is 683.53), which indicates stale/wrong variables may be feeding UACR calculation in that workflow.\n')

    lines.append('## Recommended fix\n')
    lines.append('1. Compute UACR from the exact values shown to user (post-snap values), not from hidden pre-snap values.')
    lines.append('2. Add a consistency check in code and logs: `abs(uacr - 100*A/C) < tolerance` with the same A/C values rendered in UI/report.')
    lines.append('3. In exported sheets, include explicit columns: `Albumin_used_for_uacr`, `Creatinine_used_for_uacr`, `UACR_formula`, `UACR_display` for traceability.')
    lines.append('4. Standardize rounding (e.g., compute with full precision and round only at final display to 2dp).')

    OUTPUT_MD.write_text('\n'.join(lines))
    print(f'Wrote {OUTPUT_MD}')


if __name__ == '__main__':
    main()
