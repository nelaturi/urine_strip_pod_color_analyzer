"""
Distributional analysis over a batch's per-image JSON outputs.

Usage:
    python scripts/analyze_batch_jsons.py <path/to/batch_runs/<run_id>/data>

Reads every *__result.json, computes:
  - guard_action frequency table
  - aqua/low pixel fractions and DEs split by actual_class
  - albumin/uacr confidence distributions and ceiling
  - predicted x actual class confusion matrix
  - rows where rare guard branches fired
  - top wins / top losses by bin error

All output is written to <data_dir>/analysis_summary.md alongside the JSONs.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_rows(data_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(data_dir.glob("*__result.json")):
        try:
            with p.open(encoding="utf-8") as f:
                rows.append(json.load(f))
        except Exception as e:
            print(f"  [WARN] {p.name}: {e}")
    return rows


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    vs = sorted(values)
    n = len(vs)
    def q(p):
        if n == 1:
            return vs[0]
        idx = p * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return vs[lo] + (vs[hi] - vs[lo]) * (idx - lo)
    return {
        "n": n,
        "min": vs[0],
        "p10": q(0.10),
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": vs[-1],
        "mean": sum(vs) / n,
    }


def fmt_q(q: dict) -> str:
    if q["n"] == 0:
        return "n=0"
    return (f"n={q['n']:>4d}  min={q['min']:.3f}  p10={q['p10']:.3f}  "
            f"p25={q['p25']:.3f}  med={q['median']:.3f}  p75={q['p75']:.3f}  "
            f"p90={q['p90']:.3f}  max={q['max']:.3f}  mean={q['mean']:.3f}")


def main(data_dir: Path) -> None:
    rows = load_rows(data_dir)
    out_lines: list[str] = []
    p = out_lines.append

    p(f"# Batch JSON analysis — `{data_dir.parent.name}`\n")
    p(f"**Records loaded:** {len(rows)}  ")
    statuses = Counter(r.get("status") for r in rows)
    p(f"**Status:** {dict(statuses)}\n")

    rows = [r for r in rows if r.get("status") == "success"]
    p(f"_Filtering to status==success leaves {len(rows)} rows for the rest of this report._\n")

    # ---------------- 1. guard_action frequency ----------------
    p("## 1. `guard_action` frequency\n")
    actions = Counter(r.get("guard_action") for r in rows)
    p("| Guard action | Count | % |")
    p("|---|---:|---:|")
    for act, n in actions.most_common():
        p(f"| `{act}` | {n} | {n / len(rows) * 100:.1f}% |")
    p("")

    # ---------------- 2. report_mode frequency ----------------
    p("## 2. `guard_report_mode` frequency\n")
    modes = Counter(r.get("guard_report_mode") for r in rows)
    p("| Report mode | Count | % |")
    p("|---|---:|---:|")
    for m, n in modes.most_common():
        p(f"| `{m}` | {n} | {n / len(rows) * 100:.1f}% |")
    p("")

    # ---------------- 3. distributions split by actual_class ----------------
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        ac = r.get("actual_class") or "unknown"
        by_class[ac].append(r)

    fields = [
        ("guard_aqua_pixel_fraction", "aqua_pixel_fraction"),
        ("guard_median_aqua_de", "median_aqua_DE"),
        ("guard_low_pixel_fraction", "low_pixel_fraction"),
        ("guard_median_low_de", "median_low_DE"),
        ("albumin_confidence", "albumin_confidence"),
        ("creatinine_confidence", "creatinine_confidence"),
    ]
    for fkey, label in fields:
        p(f"## 3.{fields.index((fkey, label)) + 1}. `{label}` by `actual_class`\n")
        p("| class | " + " | ".join(["n", "min", "p10", "p25", "med", "p75", "p90", "max", "mean"]) + " |")
        p("|---" + "|---:" * 9 + "|")
        for cls in sorted(by_class.keys()):
            vals = [r.get(fkey) for r in by_class[cls] if isinstance(r.get(fkey), (int, float))]
            q = quantiles(vals)
            if q["n"] == 0:
                p(f"| {cls} | 0 |  |  |  |  |  |  |  |  |")
            else:
                p(f"| {cls} | {q['n']} | {q['min']:.3f} | {q['p10']:.3f} | {q['p25']:.3f} | {q['median']:.3f} | {q['p75']:.3f} | {q['p90']:.3f} | {q['max']:.3f} | {q['mean']:.3f} |")
        p("")

    # ---------------- 4. confidence ceiling ----------------
    p("## 4. Confidence ceiling check\n")
    for fkey, label in [("albumin_confidence", "albumin"), ("creatinine_confidence", "creatinine"), ("uacr_confidence", "uacr")]:
        vals = [r.get(fkey) for r in rows if isinstance(r.get(fkey), (int, float))]
        if not vals:
            p(f"- {label}: no data")
            continue
        gates = [(0.45, ">=0.45"), (0.50, ">=0.50"), (0.55, ">=0.55"), (0.60, ">=0.60"), (0.65, ">=0.65"), (0.70, ">=0.70"), (0.75, ">=0.75")]
        cnts = ", ".join(f"{label}{g[1]}: {sum(1 for v in vals if v >= g[0])}" for g in gates)
        p(f"- **{label}**: max={max(vals):.3f}  mean={statistics.mean(vals):.3f}  ceiling check — {cnts}")
    p("")

    # ---------------- 5. confusion matrix ----------------
    p("## 5. Predicted × actual class confusion matrix\n")
    classes = sorted({r.get("actual_class") for r in rows} | {r.get("predicted_class") for r in rows})
    classes = [c for c in classes if c is not None]
    cm = defaultdict(lambda: defaultdict(int))
    for r in rows:
        cm[r.get("actual_class")][r.get("predicted_class")] += 1
    p("| actual ↓ \\ predicted → | " + " | ".join(classes) + " | total |")
    p("|---" + "|---:" * (len(classes) + 1) + "|")
    for ac in classes:
        row_total = sum(cm[ac].values())
        p(f"| **{ac}** | " + " | ".join(str(cm[ac].get(pc, 0)) for pc in classes) + f" | {row_total} |")
    correct = sum(cm[c].get(c, 0) for c in classes)
    p(f"\n**Class accuracy (predicted == actual): {correct}/{len(rows)} = {correct / len(rows) * 100:.1f}%**\n")

    # ---------------- 6. rare branches ----------------
    p("## 6. Rare guard branches that did fire\n")
    rare_actions = ["confirmed_or_override_low_exact_3_10_30",
                    "strong_aqua_matched_150", "strong_aqua_matched_250", "strong_aqua_matched_400",
                    "strong_aqua_high_watch_600", "strong_aqua_default_to_150",
                    "strong_aqua_candidate_250_insufficient_confidence",
                    "strong_aqua_candidate_400_insufficient_confidence",
                    "strong_aqua_600_not_finalized_provisional_150_400",
                    "guarded_overbright_no_aqua_to_10",
                    "guarded_overbright_weak_aqua_to_80",
                    "unconfirmed_very_high_without_validated_support",
                    "unchanged_high_watch_600_799_confidence_ok",
                    "high_watch_600_799_low_confidence_provisional_150_400"]
    for act in rare_actions:
        n = actions.get(act, 0)
        if n > 0:
            p(f"- `{act}` — {n} rows")
    if not any(actions.get(a, 0) > 0 for a in rare_actions):
        p("- _none of the rare branches fired in this batch_")
    p("")

    # ---------------- 7. exact correct predictions ----------------
    p("## 7. Correct exact predictions (predicted == actual class AND not provisional/unconfirmed)\n")
    wins = [r for r in rows
            if r.get("predicted_class") == r.get("actual_class")
            and r.get("predicted_class") is not None
            and not r.get("is_provisional")
            and not r.get("is_unconfirmed")]
    p(f"- **{len(wins)} of {len(rows)} ({len(wins) / len(rows) * 100:.1f}%) correct exact**")
    win_classes = Counter(r.get("actual_class") for r in wins)
    if win_classes:
        p("- by class: " + ", ".join(f"{c}={n}" for c, n in win_classes.most_common()))
    p("")

    # ---------------- 8. cross-tab is_provisional vs accuracy ----------------
    p("## 8. Provisional / unconfirmed flag breakdown\n")
    flags = [
        ("exact",        lambda r: not r.get("is_provisional") and not r.get("is_unconfirmed") and not r.get("is_high_watch")),
        ("provisional",  lambda r: r.get("is_provisional")),
        ("unconfirmed",  lambda r: r.get("is_unconfirmed")),
        ("high_watch",   lambda r: r.get("is_high_watch")),
    ]
    p("| flag | count | % |")
    p("|---|---:|---:|")
    for label, fn in flags:
        n = sum(1 for r in rows if fn(r))
        p(f"| {label} | {n} | {n / len(rows) * 100:.1f}% |")
    p("")

    # ---------------- 9. write report ----------------
    out_path = data_dir / "analysis_summary.md"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/analyze_batch_jsons.py <data_dir>")
        sys.exit(1)
    main(Path(sys.argv[1]))
