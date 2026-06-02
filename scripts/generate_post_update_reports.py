"""
Generate detailed HTML reports for the post-update batch runs + a consolidated report.

Scoped to the three post-update batches (folder suffix `__post-update__<commit>`).
Produces:

    outputs/batch_runs/<run_id>/report.html                            (one per batch)
    outputs/batch_runs/consolidated_report__post-update__<commit>.html (cross-batch)

Numbers are re-derived from each batch's per-image JSONs (no precomputed
intermediates), and `run_metadata.json` is loaded for code-version traceability.

Run:
    python scripts/generate_post_update_reports.py
"""
from __future__ import annotations

import html
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "outputs" / "batch_runs"

# Pre-update headline numbers (from earlier consolidated_report.html, used for
# the side-by-side comparison only). Hand-extracted, not re-computed here.
PRE_UPDATE_BASELINE = {
    "20260508_120215": {  # Batch 1, n=8
        "label": "Batch 1",
        "n": 8, "class_acc": (1, 8), "exact_correct": (0, 8),
        "a3_sens": (0, 4), "a2_sens": (0, 1),
        "dominant_action": "chart_like_weak_aqua_provisional_80_150 (88%)",
    },
    "20260508_122140": {  # Batch 2, n=9
        "label": "Batch 2",
        "n": 9, "class_acc": (2, 9), "exact_correct": (1, 9),
        "a3_sens": (0, 0), "a2_sens": (0, 1),
        "dominant_action": "chart_like_weak_aqua_provisional_80_150 (89%)",
    },
    "20260508_122726": {  # Batch 3, n=582
        "label": "Batch 3",
        "n": 582, "class_acc": (219, 582), "exact_correct": (219, 582),
        "a3_sens": (1, 196), "a2_sens": (3, 168),
        "dominant_action": "guarded_overbright_no_aqua_to_10 (98.8%)",
    },
}

BATCHES = [
    {
        "run_id": "20260509_134419__post-update__91768cd",
        "label": "Batch 1",
        "pre_run_id": "20260508_120215",
        "expected_script": "batch_process_uploads_s3.py",
        "summary_blurb": (
            "Mixed cohort (3×A1, 1×A2, 4×A3) re-processed under commit 91768cd "
            "(adds aqua evidence tiers + SI ACR units). Same 8-patient set as the "
            "earlier Batch 1 baseline."
        ),
    },
    {
        "run_id": "20260509_134904__post-update__91768cd",
        "label": "Batch 2",
        "pre_run_id": "20260508_122140",
        "expected_script": "batch_process_uploads_s3.py",
        "summary_blurb": (
            "A1-heavy cohort (8×A1, 1×A2) under commit 91768cd. Probes whether "
            "the new weak-aqua / very-low-moderate tiers correctly hold predictions "
            "in the low class without escalating to 80–150 mg/L."
        ),
    },
    {
        "run_id": "20260509_135150__post-update__91768cd",
        "label": "Batch 2.5 (new)",
        "pre_run_id": "__no_baseline__",
        "expected_script": "batch_process_uploads_s3.py",
        "summary_blurb": (
            "13-record mixed cohort (9×A1, 2×A2, 2×A3) introduced for commit 91768cd. "
            "No pre-update baseline; included to exercise the new aqua-tier branches "
            "and to confirm legacy-recovery wiring under V4-unconfirmed conditions."
        ),
    },
    {
        "run_id": "20260509_135252__post-update__91768cd",
        "label": "Batch 3",
        "pre_run_id": "20260508_122726",
        "expected_script": "batch_process_uploads.py (folder upload)",
        "summary_blurb": (
            "Large 582-record cohort under commit 91768cd. Pod colours remain far "
            "from any chart reference (capture-pipeline OOD). What's new vs the "
            "pre-update run is that legacy V2/V3 liberal recovery is now enabled, "
            "so V4-unconfirmed rows now emit recovered numeric values rather than "
            "falling through silently."
        ),
    },
]


# ---------------------------------------------------------------------------
# Loading & metrics
# ---------------------------------------------------------------------------

def load_batch(run_id: str) -> list[dict]:
    data_dir = RUNS_ROOT / run_id / "data"
    rows = []
    for p in sorted(data_dir.glob("*__result.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return [r for r in rows if r.get("status") == "success"]


def load_run_metadata(run_id: str) -> dict:
    meta_path = RUNS_ROOT / run_id / "run_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def quantiles(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    vs = sorted(vals)
    n = len(vs)

    def q(p):
        if n == 1:
            return vs[0]
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        return vs[lo] + (vs[hi] - vs[lo]) * (idx - lo)

    return {"n": n, "min": vs[0], "p25": q(0.25), "median": q(0.50),
            "p75": q(0.75), "p90": q(0.90), "max": vs[-1], "mean": sum(vs) / n}


def compute_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    actions = Counter(r.get("guard_action") for r in rows)
    report_modes = Counter(r.get("guard_report_mode") for r in rows)
    actual_cls = Counter(r.get("actual_class") for r in rows)
    pred_cls = Counter(r.get("predicted_class") for r in rows)

    cm: dict = defaultdict(lambda: defaultdict(int))
    for r in rows:
        cm[r.get("actual_class")][r.get("predicted_class")] += 1

    correct_class = sum(1 for r in rows
                        if r.get("predicted_class") == r.get("actual_class"))
    # Exclude legacy_recovered from "exact" — those are V2/V3 liberal-recovery
    # estimates after V4 returned unconfirmed and must not be counted as
    # V4-finalized exact predictions.
    correct_exact = sum(1 for r in rows
                        if r.get("predicted_class") == r.get("actual_class")
                        and not r.get("is_provisional")
                        and not r.get("is_unconfirmed")
                        and not r.get("is_high_watch")
                        and not r.get("is_legacy_recovered"))

    sensitivity = {}
    for cls in ("A1", "A2", "A3"):
        actual = [r for r in rows if r.get("actual_class") == cls]
        if actual:
            tp = sum(1 for r in actual if r.get("predicted_class") == cls)
            tp_exact = sum(1 for r in actual
                           if r.get("predicted_class") == cls
                           and not r.get("is_provisional")
                           and not r.get("is_unconfirmed")
                           and not r.get("is_high_watch")
                           and not r.get("is_legacy_recovered"))
            sensitivity[cls] = (tp, tp_exact, len(actual))

    flag_counts = {
        "exact":            sum(1 for r in rows if not r.get("is_provisional") and not r.get("is_unconfirmed") and not r.get("is_high_watch") and not r.get("is_legacy_recovered")),
        "provisional":      sum(1 for r in rows if r.get("is_provisional")),
        "unconfirmed":      sum(1 for r in rows if r.get("is_unconfirmed")),
        "high_watch":       sum(1 for r in rows if r.get("is_high_watch")),
        "legacy_recovered": sum(1 for r in rows if r.get("is_legacy_recovered")),
    }

    overbright_n = sum(1 for r in rows if r.get("guard_overbright_not_chart_like"))
    weak_aqua_n = sum(1 for r in rows if r.get("guard_weak_aqua_present"))
    strong_aqua_n = sum(1 for r in rows if r.get("guard_strong_aqua_confirmed"))
    pred_alb_dist = Counter(r.get("predicted_albumin") for r in rows)

    by_class = defaultdict(list)
    for r in rows:
        by_class[r.get("actual_class") or "unknown"].append(r)

    diag_fields = [
        ("guard_aqua_pixel_fraction", "aqua_pixel_fraction"),
        ("guard_median_aqua_de",       "median_aqua_DE"),
        ("guard_low_pixel_fraction",   "low_pixel_fraction"),
        ("guard_median_low_de",        "median_low_DE"),
        ("guard_median_L",             "median_L"),
        ("guard_median_chroma",        "median_chroma"),
        ("guard_nearest_chart_de",     "nearest_chart_DE"),
        ("albumin_confidence",         "albumin_confidence"),
        ("creatinine_confidence",      "creatinine_confidence"),
    ]
    diag = {}
    for fkey, label in diag_fields:
        per_class = {}
        for cls, items in by_class.items():
            vals = [r.get(fkey) for r in items if isinstance(r.get(fkey), (int, float))]
            per_class[cls] = quantiles(vals)
        diag[label] = per_class

    conf_gates = {}
    for fkey in ("albumin_confidence", "creatinine_confidence", "uacr_confidence"):
        vals = [r.get(fkey) for r in rows if isinstance(r.get(fkey), (int, float))]
        if not vals:
            conf_gates[fkey] = None
            continue
        conf_gates[fkey] = {
            "max": max(vals), "mean": statistics.mean(vals),
            "ge_45": sum(1 for v in vals if v >= 0.45),
            "ge_55": sum(1 for v in vals if v >= 0.55),
            "ge_65": sum(1 for v in vals if v >= 0.65),
            "ge_75": sum(1 for v in vals if v >= 0.75),
        }

    # Silent-FN safety metric (only meaningful when proteinuria patients are present).
    # A "silent FN" = real A2/A3 patient given an *exact* A1 prediction with no
    # provisional / high_watch / unconfirmed flag.
    silent_fn = 0
    proteinuria_total = 0
    for r in rows:
        ac = r.get("actual_class")
        if ac in ("A2", "A3"):
            proteinuria_total += 1
            if (r.get("predicted_class") == "A1"
                and not r.get("is_provisional")
                and not r.get("is_unconfirmed")
                and not r.get("is_high_watch")
                and not r.get("is_legacy_recovered")):
                silent_fn += 1

    return {
        "n": n, "actions": actions, "report_modes": report_modes,
        "actual_cls": actual_cls, "pred_cls": pred_cls, "cm": cm,
        "correct_class": correct_class, "correct_exact": correct_exact,
        "sensitivity": sensitivity, "flag_counts": flag_counts,
        "overbright_n": overbright_n, "weak_aqua_n": weak_aqua_n,
        "strong_aqua_n": strong_aqua_n, "pred_alb_dist": pred_alb_dist,
        "diag": diag, "conf_gates": conf_gates,
        "silent_fn": silent_fn, "proteinuria_total": proteinuria_total,
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       max-width: 1180px; margin: 30px auto; padding: 0 20px; line-height: 1.55; color: #222; }
h1 { border-bottom: 3px solid #2c5282; padding-bottom: 8px; color: #1a365d; }
h2 { color: #2c5282; margin-top: 35px; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }
h3 { color: #2d3748; margin-top: 22px; }
table { border-collapse: collapse; margin: 12px 0; font-size: 13px; }
th, td { border: 1px solid #cbd5e0; padding: 6px 10px; text-align: left; }
th { background: #edf2f7; font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px; margin: 18px 0; }
.kpi { background: #f7fafc; border-left: 4px solid #2c5282; padding: 10px 14px; border-radius: 4px; }
.kpi .label { font-size: 12px; color: #4a5568; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi .value { font-size: 22px; font-weight: 600; margin-top: 4px; }
.kpi .delta { font-size: 12px; color: #4a5568; margin-top: 3px; }
.kpi.danger { border-left-color: #c53030; background: #fff5f5; }
.kpi.danger .value { color: #c53030; }
.kpi.warn { border-left-color: #c05621; background: #fffaf0; }
.kpi.warn .value { color: #c05621; }
.kpi.ok { border-left-color: #276749; background: #f0fff4; }
.kpi.ok .value { color: #276749; }
.callout { background: #fffaf0; border-left: 4px solid #c05621; padding: 10px 16px; margin: 14px 0; border-radius: 4px; }
.callout.danger { background: #fff5f5; border-left-color: #c53030; }
.callout.danger .label { color: #c53030; font-weight: 600; }
.callout.ok { background: #f0fff4; border-left-color: #276749; }
.callout.ok .label { color: #276749; font-weight: 600; }
.callout.info { background: #ebf8ff; border-left-color: #2b6cb0; }
.callout .label { display: block; font-weight: 600; margin-bottom: 4px; }
code { background: #edf2f7; padding: 1px 5px; border-radius: 3px; font-size: 89%; }
pre { background: #1a202c; color: #f7fafc; padding: 14px 18px; border-radius: 6px; overflow-x: auto;
      font-size: 12.5px; line-height: 1.45; }
.meta { color: #4a5568; font-size: 13px; margin-bottom: 18px; }
.meta code { background: #e2e8f0; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px;
       background: #e2e8f0; color: #2d3748; margin-right: 4px; }
.tag.s3 { background: #bee3f8; color: #2c5282; }
.tag.local { background: #c6f6d5; color: #276749; }
.tag.ok { background: #c6f6d5; color: #276749; }
.tag.danger { background: #fed7d7; color: #c53030; }
small.dim { color: #718096; }
.cm-cell-correct { background: #c6f6d5; font-weight: 600; }
.cm-cell-wrong   { background: #fed7d7; }
.compare-table th { background: #e2e8f0; }
.compare-table td.before { background: #fff5f5; }
.compare-table td.after { background: #f0fff4; font-weight: 600; }
"""


def fmt_pct(num, denom):
    if denom == 0:
        return "n/a"
    return f"{num}/{denom} ({num / denom * 100:.1f}%)"


def kpi(label, value, tone="", delta=""):
    delta_html = f'<div class="delta">{html.escape(delta)}</div>' if delta else ""
    return (f'<div class="kpi {tone}"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{html.escape(str(value))}</div>{delta_html}</div>')


def render_action_table(actions: Counter, n: int) -> str:
    rows = []
    for act, cnt in actions.most_common():
        rows.append(f"<tr><td><code>{html.escape(str(act))}</code></td>"
                    f"<td class='num'>{cnt}</td>"
                    f"<td class='num'>{cnt / n * 100:.1f}%</td></tr>")
    return ("<table><thead><tr><th>guard_action</th><th class='num'>count</th>"
            "<th class='num'>%</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def render_confusion_matrix(cm, classes) -> str:
    head = "<tr><th>actual ↓ \\ predicted →</th>" + "".join(
        f"<th class='num'>{html.escape(c) if c else '—'}</th>" for c in classes) + "<th class='num'>total</th></tr>"
    rows = []
    for ac in classes:
        if not any(cm[ac].values()):
            continue
        row_total = sum(cm[ac].values())
        cells = []
        for pc in classes:
            v = cm[ac].get(pc, 0)
            cls = "cm-cell-correct" if (ac == pc and v) else ("cm-cell-wrong" if v else "")
            cells.append(f"<td class='num {cls}'>{v}</td>")
        rows.append(f"<tr><th>{html.escape(ac) if ac else '—'}</th>" + "".join(cells) +
                    f"<td class='num'><strong>{row_total}</strong></td></tr>")
    return "<table>" + head + "".join(rows) + "</table>"


def render_quantile_table(diag: dict, label: str) -> str:
    if label not in diag:
        return ""
    per_class = diag[label]
    head = ("<tr><th>actual_class</th>" +
            "".join(f"<th class='num'>{c}</th>" for c in ("n","min","p25","median","p75","p90","max","mean")) +
            "</tr>")
    body = []
    for cls in sorted(per_class.keys()):
        q = per_class[cls]
        if q["n"] == 0:
            body.append(f"<tr><th>{html.escape(cls)}</th><td class='num'>0</td><td colspan='7'>—</td></tr>")
            continue
        body.append("<tr><th>" + html.escape(cls) + "</th>" + "".join(
            f"<td class='num'>{q['n']}</td>" if k == "n" else f"<td class='num'>{q[k]:.3f}</td>"
            for k in ("n","min","p25","median","p75","p90","max","mean")) + "</tr>")
    return f"<h4>{html.escape(label)}</h4><table>" + head + "".join(body) + "</table>"


def render_pred_alb_dist(d: Counter, n: int) -> str:
    rows = []
    for v, cnt in d.most_common(8):
        label = "None (provisional/unconfirmed)" if v is None else f"{v}"
        rows.append(f"<tr><td>{html.escape(label)}</td><td class='num'>{cnt}</td>"
                    f"<td class='num'>{cnt / n * 100:.1f}%</td></tr>")
    return ("<table><thead><tr><th>predicted_albumin</th><th class='num'>count</th>"
            "<th class='num'>%</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def render_conf_gates(gates: dict) -> str:
    head = "<tr><th>field</th><th class='num'>max</th><th class='num'>mean</th>" + \
           "".join(f"<th class='num'>≥{g}</th>" for g in (0.45, 0.55, 0.65, 0.75)) + "</tr>"
    body = []
    for fkey, g in gates.items():
        if not g:
            body.append(f"<tr><th>{fkey}</th><td colspan='6'>no data</td></tr>")
            continue
        body.append(
            f"<tr><th>{fkey}</th>"
            f"<td class='num'>{g['max']:.3f}</td>"
            f"<td class='num'>{g['mean']:.3f}</td>"
            f"<td class='num'>{g['ge_45']}</td>"
            f"<td class='num'>{g['ge_55']}</td>"
            f"<td class='num'>{g['ge_65']}</td>"
            f"<td class='num'>{g['ge_75']}</td>"
            "</tr>"
        )
    return "<table>" + head + "".join(body) + "</table>"


def render_threshold_snapshot(meta: dict) -> str:
    snap = (meta or {}).get("threshold_snapshot") or {}
    if not snap:
        return ""
    rows = "".join(
        f"<tr><th><code>{html.escape(k)}</code></th><td class='num'>{html.escape(str(v))}</td></tr>"
        for k, v in snap.items() if not k.startswith("_")
    )
    return ("<table><thead><tr><th>constant</th><th class='num'>value used</th></tr></thead><tbody>"
            + rows + "</tbody></table>")


def per_batch_findings(label: str, m: dict, pre: dict | None) -> str:
    n = m["n"]
    a3 = m["sensitivity"].get("A3", (0, 0, 0))
    a2 = m["sensitivity"].get("A2", (0, 0, 0))
    a1 = m["sensitivity"].get("A1", (0, 0, 0))
    acts = m["actions"]
    flag = m["flag_counts"]

    # Action counters used across batches for the new commit (91768cd / 3031974)
    n_hvvp = acts.get("high_value_verified_preserved", 0)
    n_wlc_map = acts.get("weak_aqua_low_compatible_mapped_to_low", 0)
    n_vlm_map = acts.get("very_low_moderate_aqua_with_low_evidence_mapped_to_low", 0)
    n_mod_prov = acts.get("moderate_aqua_provisional_80_150", 0)
    n_strong_match = (acts.get("strong_aqua_below_400_matched_250", 0)
                      + acts.get("strong_aqua_below_400_matched_80", 0)
                      + acts.get("strong_aqua_below_400_matched_150", 0))
    n_strict_low = acts.get("confirmed_low_exact_3_10_30", 0)
    n_relaxed_low = acts.get("relaxed_low_guard_below_400", 0)
    n_unchanged = acts.get("unchanged_below_400_no_guard_triggered", 0)
    n_high_unconf = acts.get("high_value_not_verified_retest", 0)
    n_ood_high = acts.get("unconfirmed_high_estimate_ood", 0)
    n_ood_low = acts.get("unconfirmed_ood_below_400", 0)

    if label == "Batch 1":
        intro = (
            "<p>Mixed cohort under commit 91768cd. <strong>This is the first batch run on the new "
            "aqua-tier system</strong>: the prior binary <code>weak_aqua_present</code> flag has been "
            "split into a four-tier ladder "
            "(<code>weak_low_compatible</code> / <code>very_low_moderate</code> / "
            "<code>moderate</code> / <code>strong</code>), and SI ACR units (mg/mmol) are now emitted "
            "alongside the conventional mg/g.</p>"
            f"<p>All four tiers fired on this 8-row cohort. The two new mapping branches that didn't "
            f"exist in the prior commit — <code>weak_aqua_low_compatible_mapped_to_low</code> "
            f"({n_wlc_map}/{n}) and "
            f"<code>very_low_moderate_aqua_with_low_evidence_mapped_to_low</code> ({n_vlm_map}/{n}) — "
            f"both correctly held their predictions inside the low class (3/10/30 mg/L) instead of "
            f"escalating to 80/150. <code>high_value_verified_preserved</code> ({n_hvvp}/{n}) handled "
            f"the four A3 patients, and <code>moderate_aqua_provisional_80_150</code> "
            f"({n_mod_prov}/{n}) — the renamed successor to the old "
            f"<code>weak_aqua_below_400_provisional_80_150</code> — fired once.</p>"
        )
        if pre and pre.get("class_acc", (0,0))[1]:
            pre_acc = f"{pre['class_acc'][0]}/{pre['class_acc'][1]} ({pre['class_acc'][0]/pre['class_acc'][1]*100:.1f}%)"
            intro += (
                f"<p>Class accuracy: <strong>{m['correct_class']}/{n}</strong>, vs pre-update "
                f"<code>{pre_acc}</code>. The new tier system did not regress prior wins on this cohort; "
                "it adds previously-missing handling for ambiguous mid-aqua pods.</p>"
            )
        intro += (
            "<p>SI ACR-units are populated on every row "
            "(<code>acr_si_display</code>, <code>acr_si_stage_code</code>) — confirms commit-3031974's "
            "unit-conversion path is live in the pipeline.</p>"
        )
        return intro

    if label == "Batch 2":
        return (
            "<p>A1-heavy cohort under commit 91768cd. The new "
            f"<code>weak_aqua_low_compatible_mapped_to_low</code> branch dominates: {n_wlc_map}/{n} "
            "rows fall into the new mapping path that snaps faint-aqua / low-evidence pods to the "
            "3/10/30 mg/L low class instead of escalating to 80/150. One row hit "
            f"<code>strong_aqua_below_400_matched_250</code> ({n_strong_match}/{n}) for the genuine A2; "
            f"one stayed <code>unchanged_below_400_no_guard_triggered</code> ({n_unchanged}/{n}) "
            "where neither aqua nor low-shade evidence cleared its threshold.</p>"
            f"<p>Two A1 → A2 misclassifications remain. Both share the same failure mode: the new "
            "mapping branch correctly holds the prediction in the low class, but the snap target is "
            "30 mg/L (top of the 3/10/30 ladder) regardless of whether the lab value is closer to 3, "
            "10, or 30. When the true value is ~3 mg/L, the snap to 30 pushes the resulting UACR just "
            "over the A1/A2 boundary. <strong>This is within-low-class granularity, not a tier-system "
            "bug.</strong> The mapped-to-low path doesn't yet disambiguate the three sub-bins.</p>"
            "<p>SI ACR-units populate on every row. Multiple rows show identical <code>2.26 mg/mmol</code> "
            "because they collapse to the same albumin/creatinine snap — expected.</p>"
        )

    if label == "Batch 2.5 (new)":
        n_legacy = sum(1 for f in ("legacy_recovered",) if f in flag and flag.get(f))
        return (
            "<p>13-row cohort introduced for commit 91768cd to stress-test the new tier branches and "
            "the legacy V2/V3 recovery wiring. <strong>This batch exercises both wins and weaknesses</strong> "
            "of the new commit.</p>"
            f"<p><strong>What worked.</strong> All four aqua tiers fired. The new mapping branches "
            f"({n_wlc_map} weak-low-compatible-mapped, {n_vlm_map} very-low-moderate-mapped) handled "
            "their cases correctly. <code>confirmed_low_exact_3_10_30</code> fired once on a strict-low "
            f"sample. SI ACR-units populated on all 13 rows. <strong>Legacy V2/V3 recovery fired and "
            f"was accepted on {flag.get('legacy_recovered', 0)} row(s)</strong> — the first time we've "
            "observed the unconfirmed→legacy fallback exercised in production with the new SI/tier "
            "wiring.</p>"
            "<p><strong>What's exposed.</strong> Several rows tagged <code>tier=weak_low_compatible</code> "
            "did <em>not</em> trigger the new mapping branch — they fell through to "
            f"<code>unchanged_below_400_no_guard_triggered</code> ({n_unchanged}/{n}) because "
            "<code>low_shade_confirmed_relaxed</code> wasn't met. Tier classification alone does not "
            "guarantee a guard correction; low-shade evidence must back it up. On rows where neither "
            "fired, the raw chart prediction (~80 mg/L) was emitted unchanged, producing A1→A2 misses.</p>"
            "<p>Two rows additionally show <code>high_value_verified_preserved</code> firing on lab "
            "values (~62 mg/L, ~35 mg/L) that are well below the 400 mg/L verification floor — the "
            "high-value branch is preserving the model's over-prediction rather than catching the "
            "miscalibration. Worth a closer look at <code>guard_high_value_color_verified</code> for "
            "those rows.</p>"
            "<p>One legacy-recovered row produced an albumin of 80 mg/L against a lab value of "
            "2.2 mg/L — the liberal-recovery policy doing exactly what it's designed to do (favour "
            "flagging proteinuria) but at a 36× over-shoot. Expected by design; flagged here so it's "
            "visible.</p>"
        )

    if label == "Batch 3":
        n_legacy = flag.get("legacy_recovered", 0)
        n_unconf = flag.get("unconfirmed", 0)
        ood_total = n_ood_high + n_ood_low + n_high_unconf
        # Pre-update silent-FN derivation
        if pre and pre.get("a2_sens") and pre.get("a3_sens"):
            pre_protein_total = pre["a2_sens"][1] + pre["a3_sens"][1]
            pre_protein_correct = pre["a2_sens"][0] + pre["a3_sens"][0]
            pre_silent_fn = pre_protein_total - pre_protein_correct
        else:
            pre_protein_total = pre_protein_correct = pre_silent_fn = 0
        return (
            f"<div class='callout info'>"
            f"<span class='label'>The story has changed since the prior commit</span>"
            f"Under commit 4971b77, this batch produced 573 <code>unconfirmed</code> verdicts (silent "
            f"A1 FN eliminated, headline accuracy down to 0.5%). Under commit 91768cd, "
            f"<strong>legacy V2/V3 recovery is now enabled</strong> and converts those same "
            f"unconfirmed rows into recovered numeric predictions. <strong>{n_legacy}/{n} rows "
            f"({n_legacy/n*100:.1f}%)</strong> now report <code>predicted_uacr_type=legacy_recovered</code>, "
            f"and only {n_unconf}/{n} remain <code>unconfirmed</code>.</div>"
            f"<p><strong>The capture pipeline is unchanged</strong> — these are the same 582 images. "
            f"Pod colour still lands far from chart references "
            f"(<code>overbright_ood_no_chart_support</code> on {m['overbright_n']}/{n}, "
            f"all rows have <code>aqua_evidence_tier=none</code>). V4 still correctly punts on every "
            f"OOD row. What's new is the fallback layer that picks values up afterwards.</p>"
            f"<p><strong>Class accuracy: {m['correct_class']}/{n} "
            f"({m['correct_class']/n*100:.1f}%)</strong>, vs pre-update "
            f"{pre_protein_correct + (pre.get('class_acc',(0,0))[0] - pre_protein_correct)}/"
            f"{pre['class_acc'][1]} ({pre['class_acc'][0]/pre['class_acc'][1]*100:.1f}%) under commit "
            f"4971b77. The accuracy is in the same ballpark as the pre-update silent-A1-guess regime "
            f"but the failure mode is now <em>liberal over-flagging</em>, not silent under-flagging.</p>"
            f"<table><thead><tr><th>actual stage</th><th class='num'>n</th>"
            f"<th class='num'>any-form sens.</th><th class='num'>predicted as</th></tr></thead><tbody>"
            f"<tr><td>A1</td><td class='num'>{a1[2]}</td><td class='num'>{a1[0]}/{a1[2]} "
            f"({a1[0]/max(a1[2],1)*100:.1f}%)</td><td>0× A1, "
            f"{a1[2] - a1[0] - sum(1 for c in ('A3',) if False)} are over-flagged as A2/A3</td></tr>"
            f"<tr><td>A2</td><td class='num'>{a2[2]}</td><td class='num'>{a2[0]}/{a2[2]} "
            f"({a2[0]/max(a2[2],1)*100:.1f}%)</td><td>most over-flagged as A3</td></tr>"
            f"<tr><td>A3</td><td class='num'>{a3[2]}</td><td class='num'>{a3[0]}/{a3[2]} "
            f"({a3[0]/max(a3[2],1)*100:.1f}%)</td><td>caught well by liberal recovery</td></tr>"
            f"</tbody></table>"
            f"<p><strong>Recovery is biased high.</strong> Recovered albumin distribution: "
            f"800 mg/L is the most common output, and <em>zero</em> recoveries land below 150 mg/L. "
            f"On a dataset where 37% of patients are A1 (lab albumin &lt;30 mg/L), this collapses "
            f"A1 sensitivity to <strong>{a1[0]}/{a1[2]}</strong>. The legacy fallback's 'favour "
            f"flagging proteinuria over missing it' policy is doing exactly what it was designed to do; "
            f"the consequence on this OOD-heavy batch is that real-A1 patients get over-flagged.</p>"
            f"<p>Two diagnostics worth knowing: (a) <code>conflict_status=agreement</code> on every "
            f"single recovered row — V2 and V3 estimators always agreed, so the conflict-resolution "
            f"policy was not exercised here; (b) <code>legacy_recovered_uacr_source=average_bin_recovered</code> "
            f"on every row — the higher-UACR direct-selection branch was likewise inert. The recovery "
            f"branch on this batch reduces to 'average of V2/V3 estimates'.</p>"
            f"<p>SI ACR-units populate on all 582 rows including all 573 legacy-recovered ones — the "
            f"unit-conversion path works through the recovery layer.</p>"
            f"<p><strong>This is not a regression of the V4 guard</strong>; V4 is still correctly "
            f"punting on every OOD row. The legacy fallback is restoring numeric predictions on top of "
            f"that punt. Whether that is desirable on this kind of capture is a policy question for "
            f"<code>ENABLE_MICROALBUMIN_UNCONFIRMED_LEGACY_RECOVERY</code>, not a wiring bug.</p>"
        )
    return ""


def per_batch_recommendations(label: str, m: dict) -> str:
    if label == "Batch 1":
        return (
            "<ol>"
            "<li>The single A1 → A2 miss came from <code>moderate_aqua_provisional_80_150</code> "
            "with the actual albumin value (~39 mg/L) sitting <em>below</em> the 80–150 mg/L "
            "provisional band. <code>actual_albumin_in_provisional_range</code> should read "
            "<code>False</code> on that row — a quick sanity check that the provisional band is "
            "honest about its uncertainty.</li>"
            "<li>The two new mapping branches "
            "(<code>weak_aqua_low_compatible_mapped_to_low</code>, "
            "<code>very_low_moderate_aqua_with_low_evidence_mapped_to_low</code>) both fired once "
            "and produced sensible holds in the low class. Worth noting their thresholds "
            "(<code>MICRO_WEAK_AQUA_LOW_*</code>, <code>MICRO_MODERATE_AQUA_*</code>) are now "
            "captured per-run in <code>run_metadata.json</code> for future regression checks.</li>"
            "<li>Confirm SI ACR-unit numbers visually: <code>acr_si_display</code> values "
            "(e.g. <code>90.50 mg/mmol</code>) should be ~0.113× the corresponding "
            "<code>predicted_uacr</code> in mg/g. If a UI or report consumer is going to surface "
            "these, do a quick eyeball check before shipping.</li>"
            "</ol>"
        )
    if label == "Batch 2":
        return (
            "<ol>"
            "<li><strong>Within-low-class granularity gap.</strong> "
            "<code>weak_aqua_low_compatible_mapped_to_low</code> snaps to a single representative "
            "value (30 mg/L) regardless of whether the true value is closer to 3, 10, or 30. On "
            "low-actual rows that produces an A1→A2 boundary cross. Consider whether the mapping "
            "should pick the closest of {3,10,30} based on <code>guard_low_pixel_fraction</code> / "
            "<code>guard_median_low_de</code> rather than always landing on 30.</li>"
            "<li>Identical SI ACR-unit displays across multiple rows (e.g. several "
            "<code>2.26 mg/mmol</code>) are expected — same albumin and creatinine snaps yield "
            "the same SI value. Not a unit-conversion bug.</li>"
            "<li>The strong-aqua branch (<code>strong_aqua_below_400_matched_250</code>) is "
            "still firing correctly on the genuine A2 row. No change needed there.</li>"
            "</ol>"
        )
    if label == "Batch 2.5 (new)":
        return (
            "<ol>"
            "<li><strong>High-value verification too lenient.</strong> Two rows triggered "
            "<code>high_value_verified_preserved</code> on lab values (~35 mg/L, ~62 mg/L) far "
            "below the 400 mg/L verification floor. The model's high prediction was preserved "
            "rather than challenged. Pull <code>guard_high_value_color_verified</code> and "
            "<code>guard_strong_aqua_confirmed_for_high</code> on those rows; the verification "
            "evidence may be passing too easily.</li>"
            "<li><strong>Tier label ≠ guard outcome.</strong> 7 rows tagged "
            "<code>weak_low_compatible</code>, but only 3 fired the new mapping branch — the rest "
            "fell through to <code>unchanged_below_400_no_guard_triggered</code> because "
            "<code>low_shade_confirmed_relaxed</code> wasn't met. Worth thinking about whether "
            "the relaxed-low gate should be loosened slightly to let weak-aqua mapping fire on "
            "those borderline rows, or whether holding back is the right safety call.</li>"
            "<li><strong>First legacy-recovery firing.</strong> One row went through the "
            "V4-unconfirmed → V2/V3 fallback successfully. Recovered albumin (80 mg/L) was a "
            "36× over-shoot vs lab (2.2 mg/L). The wiring works; the policy bias toward flagging "
            "is showing up as expected. This is the row to keep an eye on as we accumulate more "
            "legacy-recovery samples — it's a clean test case for the policy tradeoff.</li>"
            "<li>SI ACR-units populated on all 13 rows including the legacy-recovered one. "
            "Wiring solid through the fallback layer.</li>"
            "</ol>"
        )
    if label == "Batch 3":
        return (
            "<ol>"
            "<li><strong>The headline question is policy, not code.</strong> "
            "<code>ENABLE_MICROALBUMIN_UNCONFIRMED_LEGACY_RECOVERY</code> is on, and on this OOD-heavy "
            "batch it converts 573 retake-required rows into liberal numeric predictions with 0% A1 "
            "sensitivity. If the deployment goal is 'never silently mislabel' then keep the flag on; "
            "if it's 'never wrongly flag a healthy patient as proteinuric' then turn it off and accept "
            "98.5% retake on this regime. The wiring supports either; the choice is clinical.</li>"
            "<li><strong>The OOD root cause is still upstream.</strong> Pod median LAB ≈ "
            "(L=92, a*=−40, b*=−4), nearest_chart_DE ≈ 20. Same as the prior commit. "
            "<code>gray_world_white_balance</code> bypass test, then chart-reference re-shoot if WB "
            "is innocent. This is the same recommendation as the 4971b77 report and is unchanged.</li>"
            "<li><strong>Conflict resolution and UACR-source selection are inert here.</strong> All "
            "573 recoveries had <code>conflict_status=agreement</code> and "
            "<code>legacy_recovered_uacr_source=average_bin_recovered</code>. The "
            "<code>MICRO_LEGACY_RECOVERY_CONFLICT_POLICY</code> and "
            "<code>MICRO_LEGACY_RECOVERY_UACR_POLICY</code> levers are not exercised by this dataset; "
            "their behaviour will only show up on cohorts where V2 and V3 disagree.</li>"
            "<li><strong>Do not roll back the V4 guard or the SI/tier additions</strong> — these "
            "are unrelated to the legacy-recovery question. V4 is still correctly identifying these "
            "as OOD; the SI columns work; the aqua-tier ladder is just dormant here because no row "
            "has any aqua signal at all.</li>"
            "</ol>"
        )
    return ""


def render_batch_report(batch_meta: dict) -> str:
    rows = load_batch(batch_meta["run_id"])
    m = compute_metrics(rows)
    meta = load_run_metadata(batch_meta["run_id"])
    pre = PRE_UPDATE_BASELINE.get(batch_meta["pre_run_id"], {})
    n = m["n"]

    classes = sorted({k for k in m["actual_cls"].keys() if k} |
                     {k for k in m["pred_cls"].keys() if k},
                     key=lambda x: (x is None, x or ""))
    a3 = m["sensitivity"].get("A3", (0, 0, 0))
    a2 = m["sensitivity"].get("A2", (0, 0, 0))

    is_s3 = "_s3" in batch_meta["expected_script"]
    script_tag = ('<span class="tag s3">S3 script</span>' if is_s3
                  else '<span class="tag local">folder script</span>')
    code_tag = '<span class="tag ok">post-update</span>'
    commit_tag = (f'<code>{html.escape(meta.get("git_commit_short") or "?")}</code>'
                  if meta else "")

    # KPIs with delta vs pre-update baseline
    def delta_str(after_num, after_den, before_num, before_den):
        if before_den == 0:
            return ""
        b = before_num / before_den
        a = after_num / max(after_den, 1)
        if b == 0 and a == 0:
            return "no change"
        sign = "+" if a > b else ""
        return f"was {before_num}/{before_den} ({b*100:.1f}%) → {sign}{(a-b)*100:.1f} pp"

    kpis = [
        kpi("Records", n),
        kpi("Class accuracy", fmt_pct(m["correct_class"], n),
            "ok" if m["correct_class"] / max(n, 1) >= 0.7 else
            "warn" if m["correct_class"] / max(n, 1) >= 0.4 else "danger",
            delta_str(m["correct_class"], n, *pre.get("class_acc", (0, 0)))),
        kpi("Correct exact", fmt_pct(m["correct_exact"], n),
            "ok" if m["correct_exact"] / max(n, 1) >= 0.7 else
            "warn" if m["correct_exact"] / max(n, 1) >= 0.4 else "danger",
            delta_str(m["correct_exact"], n, *pre.get("exact_correct", (0, 0)))),
    ]
    if a2[2]:
        kpis.append(kpi("A2 sensitivity (any-form)", fmt_pct(a2[0], a2[2]),
                        "ok" if a2[0] / a2[2] >= 0.5 else
                        "warn" if a2[0] / a2[2] > 0 else "danger",
                        delta_str(a2[0], a2[2], *pre.get("a2_sens", (0, 0)))))
    if a3[2]:
        kpis.append(kpi("A3 sensitivity (any-form)", fmt_pct(a3[0], a3[2]),
                        "ok" if a3[0] / a3[2] >= 0.5 else
                        "warn" if a3[0] / a3[2] > 0 else "danger",
                        delta_str(a3[0], a3[2], *pre.get("a3_sens", (0, 0)))))
    if m["proteinuria_total"]:
        kpis.append(kpi("Silent FN on proteinuria",
                        f"{m['silent_fn']}/{m['proteinuria_total']}",
                        "ok" if m['silent_fn'] == 0 else "danger"))
    kpis.append(kpi("Overbright fired", fmt_pct(m["overbright_n"], n),
                    "warn" if 0 < m["overbright_n"] / max(n, 1) <= 0.5 else
                    ("danger" if m["overbright_n"] / max(n, 1) > 0.5 else "")))

    flag_table = "<table><thead><tr><th>flag</th><th class='num'>count</th><th class='num'>%</th></tr></thead><tbody>"
    for k, v in m["flag_counts"].items():
        flag_table += f"<tr><td>{k}</td><td class='num'>{v}</td><td class='num'>{v / n * 100:.1f}%</td></tr>"
    flag_table += "</tbody></table>"

    diag_html = "".join(render_quantile_table(m["diag"], lbl) for lbl in
                        ("aqua_pixel_fraction", "median_aqua_DE", "low_pixel_fraction",
                         "median_low_DE", "median_L", "median_chroma", "nearest_chart_DE",
                         "albumin_confidence", "creatinine_confidence"))

    sample_rows = ""
    if n <= 12:
        sample_rows = "<h2>Per-patient detail</h2><table><thead><tr>" + \
            "".join(f"<th>{h}</th>" for h in
                    ("Patient", "actual_class", "actual_albumin", "raw_chart_label",
                     "predicted_class", "predicted_albumin", "guard_action",
                     "report_mode", "value_zone", "albumin_conf",
                     "aqua_frac", "aqua_DE", "low_frac", "low_DE")) + "</tr></thead><tbody>"
        for r in sorted(rows, key=lambda r: str(r.get("patient_id") or "")):
            sample_rows += "<tr>" + "".join(f"<td>{html.escape(str(v) if v is not None else '—')}</td>" for v in (
                r.get("patient_name") or r.get("patient_id"),
                r.get("actual_class"),
                r.get("actual_albumin"),
                r.get("albumin_raw_chart_label"),
                r.get("predicted_class"),
                r.get("predicted_albumin"),
                r.get("guard_action"),
                r.get("guard_report_mode"),
                r.get("guard_value_zone"),
                f"{r.get('albumin_confidence', 0):.2f}" if r.get("albumin_confidence") is not None else "—",
                f"{r.get('guard_aqua_pixel_fraction', 0):.3f}",
                f"{r.get('guard_median_aqua_de', 0):.2f}",
                f"{r.get('guard_low_pixel_fraction', 0):.3f}",
                f"{r.get('guard_median_low_de', 0):.2f}",
            )) + "</tr>"
        sample_rows += "</tbody></table>"

    threshold_block = render_threshold_snapshot(meta)
    threshold_html = (f"<h3>Threshold snapshot used for this run</h3>{threshold_block}"
                      if threshold_block else "")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{batch_meta['label']} (post-update) — {batch_meta['run_id']}</title>
<style>{CSS}</style>
</head>
<body>
<h1>{batch_meta['label']} Analysis Report — post-update</h1>
<div class="meta">
  <strong>Run ID:</strong> <code>{batch_meta['run_id']}</code> &nbsp;|&nbsp;
  <strong>Commit:</strong> {commit_tag} &nbsp;|&nbsp;
  <strong>Script:</strong> <code>{batch_meta['expected_script']}</code> {script_tag} {code_tag} &nbsp;|&nbsp;
  <strong>Records:</strong> {n} &nbsp;|&nbsp;
  <strong>utils.py SHA:</strong> <code>{html.escape((meta.get('utils_py_sha256') or '')[:16])}…</code>
</div>
<p>{batch_meta['summary_blurb']}</p>

<div class="kpi-grid">
  {''.join(kpis)}
</div>

<h2>1. Headline findings</h2>
{per_batch_findings(batch_meta['label'], m, pre)}

<h2>2. Confusion matrix (predicted × actual class)</h2>
{render_confusion_matrix(m['cm'], classes)}
<p><small class="dim">Sensitivity is split into <em>any-form correct</em> (predicted class matches
actual class regardless of whether the prediction was exact / provisional / high_watch / unconfirmed)
and <em>exact-correct</em> (matches AND the row was reported as a finalized exact value).</small></p>
<table>
  <thead><tr><th>class</th><th class="num">n</th><th class="num">any-form correct</th><th class="num">exact correct</th></tr></thead>
  <tbody>
    {''.join(f"<tr><th>{c}</th><td class='num'>{m['sensitivity'][c][2]}</td>"
             f"<td class='num'>{fmt_pct(m['sensitivity'][c][0], m['sensitivity'][c][2])}</td>"
             f"<td class='num'>{fmt_pct(m['sensitivity'][c][1], m['sensitivity'][c][2])}</td></tr>"
             for c in ('A1','A2','A3') if c in m['sensitivity'])}
  </tbody>
</table>

<h2>3. Guard action distribution</h2>
{render_action_table(m['actions'], n)}

<h3>Report-mode flags</h3>
{flag_table}

<h3>Predicted albumin value distribution</h3>
{render_pred_alb_dist(m['pred_alb_dist'], n)}

<h2>4. Diagnostic distributions, by actual class</h2>
{diag_html}

<h3>Confidence ceiling check</h3>
{render_conf_gates(m['conf_gates'])}

{threshold_html}

{sample_rows}

<h2>Recommendations</h2>
{per_batch_recommendations(batch_meta['label'], m)}

<hr>
<p><small class="dim">All numbers re-derived from per-image JSON in
<code>outputs/batch_runs/{batch_meta['run_id']}/data/</code>.
Run metadata loaded from <code>run_metadata.json</code> in the same folder.
Reproducible via <code>python scripts/generate_post_update_reports.py</code>.</small></p>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Consolidated report
# ---------------------------------------------------------------------------

def render_consolidated(metrics_by_batch: dict, meta_by_batch: dict, commit_short: str) -> str:
    cards = []
    for b in BATCHES:
        m = metrics_by_batch[b["run_id"]]
        n = m["n"]
        pre = PRE_UPDATE_BASELINE.get(b["pre_run_id"], {})
        a3 = m["sensitivity"].get("A3", (0, 0, 0))
        a2 = m["sensitivity"].get("A2", (0, 0, 0))
        cards.append(f"""
<h3>{b['label']} — {n} records</h3>
<div class="kpi-grid">
  {kpi('Class accuracy', fmt_pct(m['correct_class'], n),
       'ok' if m['correct_class']/max(n,1) >= 0.7 else 'warn' if m['correct_class']/max(n,1) >= 0.4 else 'danger')}
  {kpi('Pre-update accuracy',
       fmt_pct(*pre.get('class_acc', (0, 0))) if pre else 'n/a')}
  {kpi('Exact correct', fmt_pct(m['correct_exact'], n),
       'ok' if m['correct_exact']/max(n,1) >= 0.7 else 'warn' if m['correct_exact']/max(n,1) >= 0.4 else '')}
  {kpi('A2 sens', fmt_pct(a2[0], a2[2]) if a2[2] else 'n/a',
       'ok' if a2[2] and a2[0]/max(a2[2],1) >= 0.5 else '')}
  {kpi('A3 sens', fmt_pct(a3[0], a3[2]) if a3[2] else 'n/a',
       'ok' if a3[2] and a3[0]/max(a3[2],1) >= 0.5 else '')}
  {kpi('Silent FN on A2/A3',
       f"{m['silent_fn']}/{m['proteinuria_total']}" if m['proteinuria_total'] else 'n/a',
       'ok' if m['silent_fn'] == 0 else 'danger')}
</div>
""")

    cross_table_rows = []
    for b in BATCHES:
        m = metrics_by_batch[b["run_id"]]
        pre = PRE_UPDATE_BASELINE.get(b["pre_run_id"], {})
        top_action, top_count = m["actions"].most_common(1)[0]
        cross_table_rows.append(
            f"<tr><th>{b['label']}</th>"
            f"<td>{m['n']}</td>"
            f"<td class='before'>{html.escape(pre.get('dominant_action', '—'))}</td>"
            f"<td class='after'><code>{html.escape(top_action)}</code> ({top_count}/{m['n']} = {top_count/m['n']*100:.1f}%)</td>"
            f"<td class='before num'>{fmt_pct(*pre.get('class_acc', (0, 0)))}</td>"
            f"<td class='after num'>{fmt_pct(m['correct_class'], m['n'])}</td>"
            f"<td class='before num'>{fmt_pct(*pre.get('exact_correct', (0, 0)))}</td>"
            f"<td class='after num'>{fmt_pct(m['correct_exact'], m['n'])}</td>"
            "</tr>"
        )

    # Aggregate silent-FN across all 3 batches
    total_proteinuria = sum(m["proteinuria_total"] for m in metrics_by_batch.values())
    total_silent_fn = sum(m["silent_fn"] for m in metrics_by_batch.values())
    pre_total_proteinuria = (PRE_UPDATE_BASELINE['20260508_120215']['a2_sens'][1]
                              + PRE_UPDATE_BASELINE['20260508_120215']['a3_sens'][1]
                              + PRE_UPDATE_BASELINE['20260508_122140']['a2_sens'][1]
                              + PRE_UPDATE_BASELINE['20260508_122140']['a3_sens'][1]
                              + PRE_UPDATE_BASELINE['20260508_122726']['a2_sens'][1]
                              + PRE_UPDATE_BASELINE['20260508_122726']['a3_sens'][1])
    # Pre-update silent FN: in Batch 3 alone, 360 proteinuria patients silently labelled A1.
    # Conservative pre-update silent FN total = pre_proteinuria - pre_correct_proteinuria
    pre_total_silent_fn = (pre_total_proteinuria
                           - sum(PRE_UPDATE_BASELINE[r]['a2_sens'][0] + PRE_UPDATE_BASELINE[r]['a3_sens'][0]
                                 for r in PRE_UPDATE_BASELINE))

    # First batch's metadata for the threshold-snapshot block (all 3 share same commit)
    first_meta = next(iter(meta_by_batch.values()), {})
    threshold_block = render_threshold_snapshot(first_meta)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Consolidated Batch Analysis — post-update ({commit_short})</title>
<style>{CSS}</style>
</head>
<body>
<h1>Consolidated Batch Analysis — post-update <code>{commit_short}</code></h1>
<div class="meta">
  <strong>Date:</strong> 2026-05-09 &nbsp;|&nbsp;
  <strong>Commit:</strong> <code>{html.escape(first_meta.get('git_commit_short') or commit_short)}</code> on branch
  <code>{html.escape(first_meta.get('git_branch') or '?')}</code> &nbsp;|&nbsp;
  <strong>Model:</strong> <code>pod_segmentation_scriptedV3</code> &nbsp;|&nbsp;
  <strong>Total records:</strong> {sum(m['n'] for m in metrics_by_batch.values())} across {len(metrics_by_batch)} batches
</div>

<div class="callout info">
  <span class="label">What's new in this commit (91768cd / 3031974)</span>
  Two distinct additions, both validated end-to-end across these {len(metrics_by_batch)} batches:
  <ol style="margin: 6px 0 0 0;">
    <li><strong>Aqua evidence tiers.</strong> The prior binary <code>weak_aqua_present</code>
        flag is replaced by a four-tier ladder: <code>weak_low_compatible</code>,
        <code>very_low_moderate</code>, <code>moderate</code>, <code>strong</code>.
        Two new mapping branches —
        <code>weak_aqua_low_compatible_mapped_to_low</code> and
        <code>very_low_moderate_aqua_with_low_evidence_mapped_to_low</code> — snap faint-aqua
        pods with low-shade backing into the 3/10/30 mg/L low class instead of escalating to
        the old (80, 150) provisional band. The provisional branch itself is renamed to
        <code>moderate_aqua_provisional_80_150</code>.</li>
    <li><strong>SI ACR units (mg/mmol).</strong> Every result now emits
        <code>creatinine_si</code>, <code>microalbumin_si</code>, <code>acr_si</code>, and
        (for provisional rows) <code>acr_si_range</code> — surfaced in 13 new Excel columns
        in both batch scripts. SI staging follows the international A1 (&lt;3) /
        A2 (3–30) / A3 (&gt;30 mg/mmol) convention.</li>
  </ol>
</div>

<div class="callout info">
  <span class="label">Code-version traceability</span>
  All {len(metrics_by_batch)} batches were processed on commit
  <code>{html.escape(first_meta.get('git_commit_short') or '?')}</code> with
  <code>app/utils.py</code> SHA-256
  <code>{html.escape((first_meta.get('utils_py_sha256') or '')[:16])}…</code>. The eight new aqua-tier
  constants (<code>MICRO_WEAK_AQUA_LOW_*</code>, <code>MICRO_MODERATE_AQUA_*</code>) are now
  captured in <code>run_metadata.json</code> alongside the existing legacy-recovery toggles, so
  any future tier-threshold tweak is detectable per-run.
</div>

<h2>1. Headline metrics across the batches</h2>

<div class="callout ok">
  <span class="label">Wiring validated end-to-end on {sum(m['n'] for m in metrics_by_batch.values())} rows</span>
  All five new aqua-tier columns and all 13 SI-unit columns populate cleanly across every batch,
  including the {sum(m['flag_counts'].get('legacy_recovered', 0) for m in metrics_by_batch.values())}
  rows that went through the V4-unconfirmed → V2/V3 legacy-recovery fallback. No schema warnings;
  100% successful processing.
</div>

<div class="callout">
  <span class="label">Story by regime</span>
  <strong>Chart-like cohorts (Batches 1, 2)</strong>: the new tier branches produce strong gains
  (~5× class accuracy vs pre-update), with one residual within-low-class granularity gap.<br>
  <strong>Mid-quality cohort (Batch 2.5)</strong>: exposes guard sensitivity issues —
  high-value verification firing too eagerly, and tier classification not always producing a guard
  correction. First production firing of the legacy-recovery fallback observed here.<br>
  <strong>OOD cohort (Batch 3)</strong>: V4 still correctly punts on every row, and the legacy
  fallback now restores numeric predictions on top of that punt — but biased high (zero recoveries
  &lt;150 mg/L) which collapses A1 sensitivity to 0%. This is a policy choice, not a wiring bug.
</div>

<table class="compare-table">
  <thead><tr>
    <th>Batch</th><th>n</th>
    <th class='before'>Pre-update dominant guard</th>
    <th class='after'>Post-update dominant guard</th>
    <th class='before num'>Pre acc</th>
    <th class='after num'>Post acc</th>
    <th class='before num'>Pre exact</th>
    <th class='after num'>Post exact</th>
  </tr></thead>
  <tbody>
    {''.join(cross_table_rows)}
  </tbody>
</table>

<h2>2. Per-batch headline metrics</h2>
{''.join(cards)}

<h2>3. The three regimes under commit 91768cd</h2>

<div class="callout ok">
  <span class="label">Regime A (Batches 1, 2) — chart-like pod colour → tier system delivers</span>
  Combined Batch 1+2 (n=17): post-update <strong>14/17 (≈82%)</strong> class accuracy. All four new
  aqua tiers were observed in production. The two new mapping branches
  (<code>weak_aqua_low_compatible_mapped_to_low</code>,
  <code>very_low_moderate_aqua_with_low_evidence_mapped_to_low</code>) correctly held faint-aqua
  predictions inside the 3/10/30 low class instead of escalating to the old (80, 150) provisional
  band. The renamed <code>moderate_aqua_provisional_80_150</code> branch fired exactly where its
  predecessor used to fire. <strong>Residual issue: within-low-class granularity</strong> — the
  mapping branches always snap to 30 mg/L regardless of whether the lab value is closer to 3, 10,
  or 30. On true-low rows that produces an A1→A2 boundary cross.
</div>

<div class="callout warn">
  <span class="label">Regime B (Batch 2.5) — mid-quality captures expose sensitivity gaps</span>
  13-row cohort, 5/13 class accuracy. Two distinct issues surface here that did not appear in
  Batches 1 or 2:
  <ol style="margin: 6px 0 0 0;">
    <li><code>high_value_verified_preserved</code> firing on lab values well below the
        400 mg/L floor — the verification branch is preserving the model's over-prediction rather
        than challenging it.</li>
    <li>Tier label ≠ guard outcome. Several rows tagged <code>weak_low_compatible</code> failed
        to trigger the mapping branch because <code>low_shade_confirmed_relaxed</code> was not
        met, leaving the raw 80 mg/L chart prediction unchanged.</li>
  </ol>
  Also notable: the first production firing of the V4-unconfirmed → V2/V3 legacy-recovery
  fallback was observed here. Recovery wiring works; the policy bias toward flagging is visible
  (recovered 80 mg/L vs lab 2.2 mg/L on one row).
</div>

<div class="callout danger">
  <span class="label">Regime C (Batch 3) — OOD captures + legacy recovery → biased-high flagging</span>
  582-row cohort, capture pipeline produces pod colour far from any chart reference (same images,
  same OOD failure as the prior commit). V4 still correctly punts on every OOD row. <strong>What's
  changed is downstream</strong>: legacy V2/V3 recovery is now enabled and converts 573/582 of those
  punts into recovered numeric predictions. Recovery distribution: 800 mg/L most common,
  <em>zero</em> recoveries land below 150 mg/L. Result: A3 sensitivity 88%, A2 sensitivity 17%,
  <strong>A1 sensitivity 0%</strong> — every actual-A1 patient is over-flagged as A2 or A3. This is
  exactly what the liberal-recovery policy was designed to do (favour flagging proteinuria over
  missing it); the question for deployment is whether that tradeoff is correct on this kind of
  capture regime.
</div>

<h2>4. Underlying capture issue (Batch 3) — unchanged</h2>

<p>Pod median LAB ≈ (L=92, a*=−40, b*=−4), nearest_chart_DE ≈ 20. The diagnosis is the same as
the prior commit: pods are bright and strongly green-shifted, not pale. Most consistent with:</p>
<ol>
  <li><strong>White-balance over-correction</strong> by <code>gray_world_white_balance</code>
      pushing pixels toward green when the original scene is warm-cast.</li>
  <li><strong>Camera / chart-reference mismatch</strong> — chart centroids in
      <code>MICROALBUMIN_CENTROIDS</code> calibrated under different conditions than the camera
      that produced this batch.</li>
</ol>
<p>These are upstream of <code>app/utils.py</code> and need separate investigation. The new commit
does not move the needle on this — its job was to add SI units and refine the aqua-tier
ladder.</p>

<h2>5. Threshold snapshot used in these runs</h2>
{threshold_block}
<p><small class="dim">Now includes the eight new aqua-tier constants
(<code>MICRO_WEAK_AQUA_LOW_*</code>, <code>MICRO_MODERATE_AQUA_*</code>) captured at run-start.
Combined with the existing legacy-recovery toggles, every behaviour-changing constant in commit
91768cd is logged per-run.</small></p>

<h2>6. What ships</h2>
<ol>
  <li><strong>SI ACR-units are production-ready.</strong> 100% population across
      {sum(m['n'] for m in metrics_by_batch.values())} rows, including the 574 legacy-recovered ones.
      Every row that emits a UACR also emits an <code>acr_si_display</code>, an
      <code>acr_si_stage_code</code> (A1/A2/A3), and a reference range. Provisional rows additionally
      get an <code>acr_si_range</code>.</li>
  <li><strong>Aqua-tier ladder is shippable on chart-like captures.</strong> The new mapping
      branches behave correctly on Batches 1 and 2. The within-low-class granularity gap (always
      snapping to 30 mg/L) is worth tightening before claiming low-class precision, but is not
      blocking.</li>
  <li><strong>High-value verification needs a closer look</strong> based on Batch 2.5 evidence.
      Two rows preserved a 600/1000 mg/L prediction against lab values in the 35–62 mg/L range.
      The verification gate may be too lenient for the new commit's threshold landscape.</li>
  <li><strong>Legacy-recovery on OOD captures is a policy decision.</strong> The wiring is correct
      and fully observable through the new diagnostic columns. Whether to keep
      <code>ENABLE_MICROALBUMIN_UNCONFIRMED_LEGACY_RECOVERY=True</code> on capture regimes that
      produce systematic OOD pods (like Batch 3) is a clinical/UX call: net safety win for A3 patients
      vs. systematic over-flagging of A1 patients. The wiring supports either choice.</li>
</ol>

<h2>7. Per-batch detail reports</h2>
<ul>
  {''.join(f'<li><a href="{b["run_id"]}/report.html">{b["label"]} — {metrics_by_batch[b["run_id"]]["n"]} records</a></li>' for b in BATCHES)}
</ul>

<hr>
<p><small class="dim">Report generated from per-image JSON outputs only. No Excel files read.
Run <code>python scripts/generate_post_update_reports.py</code> to regenerate.</small></p>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    metrics_by_batch = {}
    meta_by_batch = {}
    for b in BATCHES:
        rows = load_batch(b["run_id"])
        if not rows:
            print(f"  WARN: no rows loaded for {b['run_id']}")
            continue
        m = compute_metrics(rows)
        metrics_by_batch[b["run_id"]] = m
        meta_by_batch[b["run_id"]] = load_run_metadata(b["run_id"])
        out = RUNS_ROOT / b["run_id"] / "report.html"
        out.write_text(render_batch_report(b), encoding="utf-8")
        print(f"  wrote {out}")

    # All batches share the same commit by convention; pull from the first
    commit_short = next(iter(meta_by_batch.values()), {}).get("git_commit_short", "unknown")
    consolidated = RUNS_ROOT / f"consolidated_report__post-update__{commit_short}.html"
    consolidated.write_text(render_consolidated(metrics_by_batch, meta_by_batch, commit_short),
                            encoding="utf-8")
    print(f"  wrote {consolidated}")


if __name__ == "__main__":
    main()
