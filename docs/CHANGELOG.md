# Change Log — Microalbumin / UACR Decision Logic

A running, plain-English record of changes to the albumin-bin and A1/A2 decision
logic. Newest entry on top. Each entry explains **what** changed, **why**, **how**
it works, **what stayed the same**, and **how it was verified** so any reader —
clinical, QA, or engineering — can understand the change without reading code.

Background shared by all entries:

- The strip color is mapped to an albumin reading in one of three low bins:
  **3, 10, or 30 mg/L**.
- The final category comes from `UACR = 100 × albumin_mg_l / creatinine_mg_dl`:
  - `UACR < 30 mg/g` → **A1** (healthy)
  - `UACR 30–300 mg/g` → **A2** (concern)
- So an over-eager **30 mg/L** pick can wrongly push a true **A1** case into **A2**.

### Glossary (terms used throughout)

- **ΔE (delta-E) — color distance.** How far the strip color is from a bin's
  reference color. **Smaller = better match.** We compute one per bin:
  - `de_3`, `de_10`, `de_30` = distance from the strip to the 3 / 10 / 30 mg/L
    reference colors. `de_30 = 4` means 30 fits closely; `de_30 = 12` means poor.
- **frac — pixel fraction (pixel support).** Of all pod pixels, the share that
  fell in each bin's color band. **Bigger = more pixels vote for that bin.**
  - `frac_3`, `frac_10`, `frac_30`. e.g. `frac_30 = 0.20` → 20% of pixels look
    like 30.
  - ΔE and frac are two **independent** signals: ΔE = how close the color is;
    frac = how many pixels agree.
- **Adv30 — color advantage of 30.** `Adv30 = min(de_3, de_10) − de_30`.
  How much **better** 30 fits than the next-best small bin. Big positive → 30
  clearly wins on color; near zero → ambiguous.
  - Example: de {3:8, 10:7, 30:4} → `Adv30 = 7 − 4 = 3.0` (30 wins big).
- **Dom30 — pixel dominance of 30.** `Dom30 = frac_30 − max(frac_3, frac_10)`.
  How much **more** pixel support 30 has than 3 or 10. Big positive → most pixels
  really vote 30; near zero/negative → no real lead.
  - Example: frac {3:0.09, 10:0.10, 30:0.11} → `Dom30 = 0.11 − 0.10 = 0.01` (weak).
- **Why use both:** a safe 30 should win on **both** color and pixels, **or** win
  **hugely** on color alone — otherwise an ambiguous 30 can slip through near the
  A1/A2 line.

---

## 2026-06-01 — A1/A2 boundary made inclusive at the A1 side (UACR == 30 -> A1)

**Files changed**
- `app/utils.py` — 5 staging functions (exact + provisional-range, both unit systems).
- `tests/test_unit_conversions.py` — added boundary tests.
- `tests/test_microalbumin_legacy_recovery.py` — updated `test_uacr_staging`.
- `requirements.txt` — added missing `scikit-image` dependency (see note below).

**Clinical authority:** the inclusive `<= 30` boundary was specified by the doctor.

### What changed
The A1/A2 cutoff is now **inclusive at the A1 side**. The exact boundary value is
classified as **A1** instead of A2:

| UACR (mg/g)      | Stage |
|------------------|-------|
| `<= 30`          | A1    |
| `> 30` to `<= 300` | A2  |
| `> 300`          | A3    |

SI ACR (mg/mmol) mirrors this: `<= 3` -> A1, `> 3` to `<= 30` -> A2, `> 30` -> A3.

Concretely, the only value whose stage moves is the **exact boundary**:
`UACR == 30.0` (and `ACR == 3.0 mg/mmol`) now reads **A1** (was A2). Everything
strictly below or above is unchanged. The A2/A3 line at 300 mg/g (SI 30 mg/mmol)
is untouched and was already inclusive to A2.

### How (minimal, operator-only edits)
`<` -> `<=` at the A1 cutoff, with the matching range-arm flips. No new constants,
no restructuring. Spots:
- Exact: `stage_uacr_value`, `calculate_uacr_and_category`, `stage_acr_si_mg_mmol`.
- Provisional range: `calculate_uacr_range_and_stage`, `calculate_acr_si_range`
  (`*_high < X` -> `<= X`; `*_low >= X` -> `> X`; `*_low < X` -> `<= X`).
- Reference-range display strings updated to `<= 30 mg/G` and `<= 3 mg/mmol`.

### What did NOT change
- The A2/A3 boundary (300 mg/g; SI 30 mg/mmol).
- Bin selection (3/10/30), aqua tiers, unconfirmed, A3, legacy recovery.
- All staging behavior for values not exactly on the A1/A2 boundary.

### Dependency fix (incidental, required to run the full suite)
`app/utils.py` hard-imports `skimage.color` (`rgb2lab`, `deltaE_ciede2000`) but
`scikit-image` was missing from `requirements.txt`. Added it. This was a
pre-existing gap, unrelated to the boundary change.

### Verification
```
python -m py_compile app/utils.py
pytest tests/test_unit_conversions.py tests/test_microalbumin_low_bin_selector.py \
       tests/test_microalbumin_legacy_recovery.py::test_uacr_staging -q   # 27 passed
```
New tests assert: `UACR == 30 -> A1`, `30.01 -> A2`, `300 -> A2`, `300.01 -> A3`,
`ACR == 3 -> A1`, and provisional range high `== 30`/`== 3` -> provisional A1.

**Known unrelated failures:** 28 tests in `test_microalbumin_shade_guard.py`,
`test_microalbumin_balanced_guard.py`, and `test_microalbumin_provisional_scenario.py`
fail in the guard/mapping layer (e.g. `provisional_available` False, albumin map
`400 != 150`). Verified identical before and after this change — pre-existing, not
caused here. Out of scope; flagged for separate investigation.

---

## 2026-06-01 — Boundary-sensitive 30 mg/L confirmation for low-bin selection

**Files changed**
- `app/utils.py` — function `select_low_albumin_bin_3_10_30()` + 3 new constants.
- `tests/test_microalbumin_low_bin_selector.py` — updated/added focused tests.

### Problem
The selector accepted **30 mg/L** too easily. The old "clear 30 evidence" rule
passed if **either**:
1. 30 had a clear color (ΔE) advantage over 3 and 10, **or**
2. 30's pixel fraction crossed `MICRO_LOW_30_PIXEL_FRACTION_MIN` (0.18).

Rule #2 alone is too permissive **near the A1/A2 boundary**: a little pixel support
could keep an ambiguous 30, flipping a true A1 case into A2.

### What we changed
We added a stricter check that applies **only** when **both** are true:
- the case is on the A1/A2 boundary (`low_30_uacr_boundary_risk == True`), **and**
- the selector already picked **30** (`selected_low_bin == 30`).

In that narrow case we now ignore the permissive base rule and require
**boundary-sensitive confirmation** before keeping 30.

Two evidence meters:
- `Adv30 = min(median_de_3, median_de_10) − median_de_30` — how much **better** 30
  matches by color (higher is stronger).
- `Dom30 = pixel_fraction_30 − max(pixel_fraction_3, pixel_fraction_10)` — how much
  **more** pixel support 30 has (higher is stronger).

30 is kept on the boundary **only if**:

```
(Adv30 >= 2.0  AND  Dom30 >= 0.10)
   OR
 Adv30 >= 3.0
```

If confirmation **fails**, we downgrade **within the low family only** (never out of
it), in this order:
1. If `UACR at 10 mg/L` would **still be ≥ 30** (still A2) → pick **3**.
   (Downgrading to 10 would not help the patient.)
2. Else if 3 clearly beats 10 by color
   (`de_3 + MICRO_LOW_3_CLEAR_ADVANTAGE_OVER_10 < de_10`) → pick **3**.
3. Else → pick **10**.

The existing kill-switch `MICRO_LOW_30_REQUIRE_CLEAR_EVIDENCE_WHEN_UACR_BOUNDARY`
is **kept** and still gates the boundary path — set it to `False` to disable the
new strict behavior instantly.

### New tuning constants (`app/utils.py`)
```
MICRO_LOW_30_BOUNDARY_DE_ADVANTAGE_MIN    = 2.0   # color win needed (with dominance)
MICRO_LOW_30_BOUNDARY_DOMINANCE_MIN       = 0.10  # pixel win needed (with color win)
MICRO_LOW_30_BOUNDARY_DE_ADVANTAGE_STRONG = 3.0   # color win alone is enough
```

### New diagnostic fields in the selector return payload
Visibility only — these do **not** change any decision:
- `low_30_de_advantage` — the `Adv30` value.
- `low_30_support_dominance` — the `Dom30` value.
- `base_clear_30_evidence` — result of the old (legacy) 30 rule.
- `boundary_sensitive_30_evidence` — result of the new strict rule.

Note: the existing `clear_30_evidence` field keeps its legacy meaning off the
boundary; on the boundary-risk path it now reflects the stricter rule actually used.

### What did NOT change
- The global A1/A2 threshold (still `UACR = 30 mg/g`).
- Strong aqua, moderate aqua, high-watch, unconfirmed, A3, legacy recovery, and
  non-low-family logic.
- Cases where the selector picked 3 or 10 (not 30).
- Non-boundary 30 cases — the legacy rule still applies there unchanged.
- UI wording.

**Net behavior change is one narrow case:** on the A1/A2 boundary + picked 30 +
weak evidence → now downgrades to 3 or 10 instead of keeping 30.

### Worked examples
- creat=80, ΔE {3:8, 10:7, 30:4}, frac_30=0.20 → `Adv30 = 7−4 = 3.0` ≥ 3.0 →
  **keep 30**.
- creat=80, ΔE {3:5.4, 10:5.2, 30:5.0}, frac {0.09,0.10,0.11} →
  `Adv30 = 0.2`, `Dom30 = 0.01` → fail. `UACR@10 = 12.5 (<30)`, 3 not clearly
  better → **pick 10**.
- Same ΔE/frac but creat=20 → `UACR@10 = 50 (≥30, still A2)` → **pick 3**.

### Verification
```
python -m py_compile app/utils.py
pytest tests/test_microalbumin_low_bin_selector.py -q   # 15 passed
```
Tests cover: weak-30 boundary → downgrade to 10; weak-30 boundary where 10 is
still A2 → downgrade to 3; strong-30 boundary → keep 30; non-boundary 30 support
→ keep 30; and unchanged non-30, moderate-aqua, and strong-aqua branches.

---

<!--
Template for future entries — copy above this line:

## YYYY-MM-DD — <short title>

**Files changed**
- ...

### Problem
...

### What we changed
...

### What did NOT change
...

### Verification
...
-->
