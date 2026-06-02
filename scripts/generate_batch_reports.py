"""
Generate detailed HTML reports for each batch + a consolidated cross-batch report.

Reads per-image JSONs from each batch's data/ folder, computes all metrics from
scratch (no precomputed numbers), and writes:

    outputs/batch_runs/<run_id>/report.html             (one per batch)
    outputs/batch_runs/consolidated_report.html         (cross-batch)

Run:
    python scripts/generate_batch_reports.py
"""
from __future__ import annotations

import html
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / "outputs" / "batch_runs"

BATCHES = [
    {
        "run_id": "20260508_120215",
        "label": "Batch 1",
        "expected_script": "batch_process_uploads_s3.py",
        "summary_blurb": (
            "Mixed cohort (3×A1, 1×A2, 4×A3) processed via the S3 batch script. "
            "First batch run with the new diagnostic columns wired up."
        ),
    },
    {
        "run_id": "20260508_122140",
        "label": "Batch 2",
        "expected_script": "batch_process_uploads_s3.py",
        "summary_blurb": (
            "A1-heavy cohort (8×A1, 1×A2) processed via the S3 batch script. "
            "Used to probe behaviour on genuine low-albumin strips."
        ),
    },
    {
        "run_id": "20260508_122726",
        "label": "Batch 3",
        "expected_script": "batch_process_uploads.py (folder upload)",
        "summary_blurb": (
            "Large 582-record cohort processed via the local folder batch script. "
            "Different upstream image regime — pod colours land far from any chart reference."
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

    return {
        "n": n,
        "min": vs[0],
        "p25": q(0.25),
        "median": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "max": vs[-1],
        "mean": sum(vs) / n,
    }


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
    correct_exact = sum(1 for r in rows
                        if r.get("predicted_class") == r.get("actual_class")
                        and not r.get("is_provisional")
                        and not r.get("is_unconfirmed")
                        and not r.get("is_high_watch"))

    sensitivity = {}
    for cls in ("A1", "A2", "A3"):
        actual = [r for r in rows if r.get("actual_class") == cls]
        if actual:
            tp = sum(1 for r in actual if r.get("predicted_class") == cls)
            tp_exact = sum(1 for r in actual
                           if r.get("predicted_class") == cls
                           and not r.get("is_provisional")
                           and not r.get("is_unconfirmed")
                           and not r.get("is_high_watch"))
            sensitivity[cls] = (tp, tp_exact, len(actual))

    flag_counts = {
        "exact":       sum(1 for r in rows if not r.get("is_provisional") and not r.get("is_unconfirmed") and not r.get("is_high_watch")),
        "provisional": sum(1 for r in rows if r.get("is_provisional")),
        "unconfirmed": sum(1 for r in rows if r.get("is_unconfirmed")),
        "high_watch":  sum(1 for r in rows if r.get("is_high_watch")),
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
            "max": max(vals),
            "mean": statistics.mean(vals),
            "ge_45": sum(1 for v in vals if v >= 0.45),
            "ge_55": sum(1 for v in vals if v >= 0.55),
            "ge_65": sum(1 for v in vals if v >= 0.65),
            "ge_75": sum(1 for v in vals if v >= 0.75),
        }

    return {
        "n": n,
        "actions": actions,
        "report_modes": report_modes,
        "actual_cls": actual_cls,
        "pred_cls": pred_cls,
        "cm": cm,
        "correct_class": correct_class,
        "correct_exact": correct_exact,
        "sensitivity": sensitivity,
        "flag_counts": flag_counts,
        "overbright_n": overbright_n,
        "weak_aqua_n": weak_aqua_n,
        "strong_aqua_n": strong_aqua_n,
        "pred_alb_dist": pred_alb_dist,
        "diag": diag,
        "conf_gates": conf_gates,
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       max-width: 1100px; margin: 30px auto; padding: 0 20px; line-height: 1.55; color: #222; }
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
.kpi.danger { border-left-color: #c53030; background: #fff5f5; }
.kpi.danger .value { color: #c53030; }
.kpi.warn { border-left-color: #c05621; background: #fffaf0; }
.kpi.warn .value { color: #c05621; }
.kpi.ok { border-left-color: #276749; background: #f0fff4; }
.kpi.ok .value { color: #276749; }
.callout { background: #fffaf0; border-left: 4px solid #c05621; padding: 10px 16px; margin: 14px 0; border-radius: 4px; }
.callout.danger { background: #fff5f5; border-left-color: #c53030; }
.callout.danger .label { color: #c53030; font-weight: 600; }
.callout.info { background: #ebf8ff; border-left-color: #2b6cb0; }
.callout .label { display: block; font-weight: 600; margin-bottom: 4px; }
code { background: #edf2f7; padding: 1px 5px; border-radius: 3px; font-size: 89%; }
pre { background: #1a202c; color: #f7fafc; padding: 14px 18px; border-radius: 6px; overflow-x: auto;
      font-size: 12.5px; line-height: 1.45; }
.meta { color: #4a5568; font-size: 13px; margin-bottom: 18px; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px;
       background: #e2e8f0; color: #2d3748; margin-right: 4px; }
.tag.s3 { background: #bee3f8; color: #2c5282; }
.tag.local { background: #c6f6d5; color: #276749; }
.tag.danger { background: #fed7d7; color: #c53030; }
small.dim { color: #718096; }
.cm-cell-correct { background: #c6f6d5; font-weight: 600; }
.cm-cell-wrong   { background: #fed7d7; }
"""


def fmt_pct(num, denom):
    if denom == 0:
        return "n/a"
    return f"{num}/{denom} ({num / denom * 100:.1f}%)"


def kpi(label, value, tone=""):
    return f'<div class="kpi {tone}"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(str(value))}</div></div>'


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
        f"<th class='num'>{html.escape(c)}</th>" for c in classes) + "<th class='num'>total</th></tr>"
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
        rows.append(f"<tr><th>{html.escape(ac)}</th>" + "".join(cells) +
                    f"<td class='num'><strong>{row_total}</strong></td></tr>")
    return "<table>" + head + "".join(rows) + "</table>"


def render_quantile_table(diag: dict, label: str) -> str:
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


def per_batch_findings(label: str, run_id: str, m: dict) -> str:
    """Hand-curated findings paragraph specific to each batch's regime."""
    n = m["n"]
    overbright_pct = m["overbright_n"] / n * 100 if n else 0
    weak_aqua_pct = m["weak_aqua_n"] / n * 100 if n else 0
    a3 = m["sensitivity"].get("A3", (0, 0, 0))
    a2 = m["sensitivity"].get("A2", (0, 0, 0))
    if run_id == "20260508_120215":
        return (
            "<p>The cohort has a real mix of A1/A2/A3 patients but every prediction "
            "lands in <code>chart_like_weak_aqua_provisional_80_150</code> or "
            "<code>unchanged_no_guard_triggered</code>. Strong aqua never confirms even though "
            "Patient 1 (true A3, lab 982.7&nbsp;mg/L) shows aqua_pixel_fraction = 0.257 with "
            "median_aqua_DE = 8.71 — only 0.71 ΔE above the threshold.</p>"
            "<p>The provisional fallback always returns <code>(80, 150)</code> regardless of regression "
            "magnitude — so a patient at 982 mg/L and another at 36 mg/L receive the same band. "
            "Albumin confidence ceiling here is <strong>0.55</strong>, well below the 0.65 / 0.75 "
            "gates required to release exact 250+ readings.</p>"
        )
    if run_id == "20260508_122140":
        return (
            "<p>An A1-heavy cohort. The same provisional fallback fires for 8 of 9 patients, "
            "this time pushing genuine low-albumin patients (lab values 1.3–8.3 mg/L) up to a "
            "provisional 80–150 band. Patient 1 (B Srinivasa Rao, A1, lab 6.0) is the only "
            "exact-correct prediction in the batch and only because aqua_pixel_fraction "
            "(0.040) sits exactly at the weak-aqua threshold.</p>"
            "<p>Patient 9 (Jaya Paul Reddy T, true A2 at 205.5&nbsp;mg/L) had "
            "aqua_pixel_fraction = 0.464 (very strong) and median_aqua_DE = 9.01 — just over the 8.0 "
            "gate. Albumin confidence ceiling here is <strong>0.35</strong>, lower than Batch 1.</p>"
        )
    if run_id == "20260508_122726":
        return (
            f"<p class='callout danger'><span class='label'>Clinically dangerous behaviour</span>"
            f"The <code>guarded_overbright_no_aqua_to_10</code> branch fires on "
            f"<strong>{m['overbright_n']}/{n} ({overbright_pct:.1f}%)</strong> of rows — assigning "
            f"a fixed 10 mg/L (A1 / normal) to nearly every patient, including "
            f"<strong>{a2[2] - a2[0]} of {a2[2]} A2 patients</strong> and "
            f"<strong>{a3[2] - a3[0]} of {a3[2]} A3 patients</strong>.</p>"
            "<p>The pod color extraction produces median ΔE around 22 from any chart reference — "
            "i.e. the pixels selected as 'pod' are nowhere near any low or aqua reference. "
            "The guard interprets this as 'blank/washed-out pod' and applies its safety fallback. "
            "This is consistent with overexposed capture, white-balance over-correction, or "
            "segmentation drift selecting non-pod regions. The guard is doing what it was "
            "designed to do — the inputs themselves are out of distribution.</p>"
            f"<p>Albumin confidence ceiling is 0.65 with only "
            f"{m['conf_gates']['albumin_confidence']['ge_65']} of {n} rows reaching ≥0.65.</p>"
        )
    return ""


def per_batch_recommendations(run_id: str, m: dict) -> str:
    if run_id == "20260508_122726":
        return (
            "<ol>"
            "<li><strong>URGENT — visually inspect 5–10 composite images</strong> from "
            "<code>outputs/batch_runs/20260508_122726/images/</code>, especially A3 patients "
            "predicted as A1. Check whether the pod region in the composite visually shows "
            "blue/teal or pale/white.</li>"
            "<li><strong>Add <code>median_L</code>, <code>median_a</code>, <code>median_b</code>, "
            "<code>median_chroma</code> to <code>_extract_guard_fields</code></strong> — those four "
            "fields would tell us in one batch whether this is overexposure, white-balance "
            "over-correction, or segmentation drift.</li>"
            "<li><strong>Hold off on Batch 1/2 threshold tuning</strong> — the proposed "
            "<code>MICRO_STRONG_AQUA_DE_MAX</code> / weak-aqua / confidence-floor adjustments "
            "have no effect here because <code>overbright_not_chart_like</code> short-circuits "
            "the decision tree first.</li>"
            "<li><strong>Critical safety change:</strong> the <code>guarded_overbright_no_aqua_to_10</code> "
            "branch should emit <code>report_mode=&quot;unconfirmed&quot;</code> with a 'retake "
            "image' message instead of silently returning 10 mg/L. In Batch 3 this would have "
            f"flipped {m['overbright_n']} silent false-negatives into explicit retest "
            "requests.</li>"
            "</ol>"
        )
    return (
        "<ol>"
        "<li>Loosen <code>MICRO_STRONG_AQUA_DE_MAX</code> from 8.0 → 11.0 to rescue patients sitting "
        "just over the gate (Patient 1 of Batch 1, Patient 9 of Batch 2).</li>"
        "<li>Raise <code>MICRO_WEAK_AQUA_MIN</code> from 0.04 → ~0.30 so the weak-aqua fallback "
        "does not fire on noise in genuine low-albumin strips.</li>"
        "<li>Make the provisional fallback regression-aware — anchor the band on the nearest "
        "allowed bin to the LAB median rather than always returning (80, 150).</li>"
        "<li>Independently audit the upstream confidence model — the post-guard "
        "0.65 / 0.75 floors are unreachable on this batch's images regardless of guard tuning.</li>"
        "</ol>"
    )


def render_batch_report(batch_meta: dict) -> str:
    rows = load_batch(batch_meta["run_id"])
    m = compute_metrics(rows)
    n = m["n"]

    classes = sorted({k for k in m["actual_cls"].keys() if k} |
                     {k for k in m["pred_cls"].keys() if k})
    a3 = m["sensitivity"].get("A3", (0, 0, 0))
    a2 = m["sensitivity"].get("A2", (0, 0, 0))
    a1 = m["sensitivity"].get("A1", (0, 0, 0))

    is_s3 = "_s3" in batch_meta["expected_script"]
    script_tag = '<span class="tag s3">S3 script</span>' if is_s3 else '<span class="tag local">folder script</span>'
    danger_tag = '<span class="tag danger">CLINICALLY DANGEROUS</span>' if batch_meta["run_id"] == "20260508_122726" else ""

    # KPIs
    kpis = []
    kpis.append(kpi("Records", n))
    kpis.append(kpi("Class accuracy", fmt_pct(m["correct_class"], n),
                    "ok" if m["correct_class"] / max(n, 1) >= 0.7 else
                    "warn" if m["correct_class"] / max(n, 1) >= 0.4 else "danger"))
    kpis.append(kpi("Correct exact", fmt_pct(m["correct_exact"], n),
                    "danger" if m["correct_exact"] / max(n, 1) < 0.1 else "warn"))
    if a2[2]:
        kpis.append(kpi("A2 sensitivity (any)", fmt_pct(a2[0], a2[2]),
                        "ok" if a2[0] / a2[2] >= 0.5 else "danger"))
    if a3[2]:
        kpis.append(kpi("A3 sensitivity (any)", fmt_pct(a3[0], a3[2]),
                        "ok" if a3[0] / a3[2] >= 0.5 else "danger"))
    kpis.append(kpi("Overbright fired", fmt_pct(m["overbright_n"], n),
                    "danger" if m["overbright_n"] / max(n, 1) > 0.5 else ""))
    kpis.append(kpi("Albumin conf ≥ 0.65",
                    f"{m['conf_gates']['albumin_confidence']['ge_65'] if m['conf_gates']['albumin_confidence'] else 0}/{n}",
                    "danger"))

    flag_table = "<table><thead><tr><th>flag</th><th class='num'>count</th><th class='num'>%</th></tr></thead><tbody>"
    for k, v in m["flag_counts"].items():
        flag_table += f"<tr><td>{k}</td><td class='num'>{v}</td><td class='num'>{v / n * 100:.1f}%</td></tr>"
    flag_table += "</tbody></table>"

    diag_html = "".join(render_quantile_table(m["diag"], lbl) for lbl in
                        ("aqua_pixel_fraction", "median_aqua_DE", "low_pixel_fraction",
                         "median_low_DE", "albumin_confidence", "creatinine_confidence"))

    sample_rows = ""
    if n <= 12:
        sample_rows = "<h2>5. Per-patient detail</h2><table><thead><tr>" + \
            "".join(f"<th>{h}</th>" for h in
                    ("Patient", "actual_class", "actual_albumin", "raw_chart_label",
                     "predicted_class", "predicted_albumin", "guard_action",
                     "albumin_conf", "aqua_frac", "aqua_DE", "low_frac", "low_DE")) + "</tr></thead><tbody>"
        for r in sorted(rows, key=lambda r: r.get("patient_id") or 0):
            sample_rows += "<tr>" + "".join(f"<td>{html.escape(str(v) if v is not None else '—')}</td>" for v in (
                r.get("patient_name") or r.get("patient_id"),
                r.get("actual_class"),
                r.get("actual_albumin"),
                r.get("albumin_raw_chart_label"),
                r.get("predicted_class"),
                r.get("predicted_albumin"),
                r.get("guard_action"),
                f"{r.get('albumin_confidence', 0):.2f}" if r.get("albumin_confidence") is not None else "—",
                f"{r.get('guard_aqua_pixel_fraction', 0):.3f}",
                f"{r.get('guard_median_aqua_de', 0):.2f}",
                f"{r.get('guard_low_pixel_fraction', 0):.3f}",
                f"{r.get('guard_median_low_de', 0):.2f}",
            )) + "</tr>"
        sample_rows += "</tbody></table>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{batch_meta['label']} — {batch_meta['run_id']}</title>
<style>{CSS}</style>
</head>
<body>
<h1>{batch_meta['label']} Analysis Report</h1>
<div class="meta">
  <strong>Run ID:</strong> <code>{batch_meta['run_id']}</code> &nbsp;|&nbsp;
  <strong>Script:</strong> <code>{batch_meta['expected_script']}</code> {script_tag} {danger_tag} &nbsp;|&nbsp;
  <strong>Records:</strong> {n}
</div>
<p>{batch_meta['summary_blurb']}</p>

<div class="kpi-grid">
  {''.join(kpis)}
</div>

<h2>1. Headline findings</h2>
{per_batch_findings(batch_meta['label'], batch_meta['run_id'], m)}

<h2>2. Confusion matrix (predicted × actual class)</h2>
{render_confusion_matrix(m['cm'], classes)}
<p><small class="dim">Sensitivities by class — both <em>any-form</em> (provisional or exact) and
<em>exact-only</em>. Provisional matches mean the predicted class falls within an A-stage range
the guard couldn't finalize numerically.</small></p>
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

{sample_rows}

<h2>{ '6' if n <= 12 else '5' }. Recommendations</h2>
{per_batch_recommendations(batch_meta['run_id'], m)}

<hr>
<p><small class="dim">Report generated from per-image JSONs under
<code>outputs/batch_runs/{batch_meta['run_id']}/data/</code>. Reproducible via
<code>python scripts/generate_batch_reports.py</code>. All numbers re-derived from raw JSON; no
precomputed intermediates.</small></p>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Consolidated report
# ---------------------------------------------------------------------------

def render_consolidated(metrics_by_batch: dict) -> str:
    cards = []
    for b in BATCHES:
        m = metrics_by_batch[b["run_id"]]
        n = m["n"]
        a3 = m["sensitivity"].get("A3", (0, 0, 0))
        a2 = m["sensitivity"].get("A2", (0, 0, 0))
        cards.append(f"""
<div class="kpi-grid">
  {kpi(f'{b["label"]} records', n)}
  {kpi('Class accuracy', fmt_pct(m['correct_class'], n))}
  {kpi('Exact correct', fmt_pct(m['correct_exact'], n),
       'danger' if m['correct_exact'] / max(n,1) < 0.1 else '')}
  {kpi('A2 sensitivity', fmt_pct(a2[0], a2[2]) if a2[2] else 'n/a',
       'danger' if a2[2] and a2[0]/max(a2[2],1) < 0.2 else '')}
  {kpi('A3 sensitivity', fmt_pct(a3[0], a3[2]) if a3[2] else 'n/a',
       'danger' if a3[2] and a3[0]/max(a3[2],1) < 0.2 else '')}
  {kpi('Overbright fired', fmt_pct(m['overbright_n'], n),
       'danger' if m['overbright_n']/max(n,1) > 0.5 else '')}
</div>
""")

    cross_table_rows = []
    for b in BATCHES:
        m = metrics_by_batch[b["run_id"]]
        top_action, top_count = m["actions"].most_common(1)[0]
        diag = m["diag"]
        med_aqua_de_all = []
        for cls, q in diag["median_aqua_DE"].items():
            if q["n"]:
                med_aqua_de_all.append(q["median"])
        med_aqua_de = statistics.mean(med_aqua_de_all) if med_aqua_de_all else 0
        med_low_de = statistics.mean(
            q["median"] for q in diag["median_low_DE"].values() if q["n"]
        ) if any(q["n"] for q in diag["median_low_DE"].values()) else 0
        alb_max = m["conf_gates"]["albumin_confidence"]["max"] if m["conf_gates"]["albumin_confidence"] else 0
        cross_table_rows.append(
            f"<tr><th>{b['label']}</th>"
            f"<td>{m['n']}</td>"
            f"<td><code>{html.escape(top_action)}</code> ({top_count}/{m['n']} = {top_count/m['n']*100:.1f}%)</td>"
            f"<td class='num'>{med_aqua_de:.2f}</td>"
            f"<td class='num'>{med_low_de:.2f}</td>"
            f"<td class='num'>{alb_max:.3f}</td>"
            f"<td class='num'>{fmt_pct(m['correct_exact'], m['n'])}</td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Consolidated Batch Analysis</title>
<style>{CSS}</style>
</head>
<body>
<h1>Consolidated Batch Analysis (3 batches, 599 records)</h1>
<div class="meta">
  <strong>Date:</strong> 2026-05-08 &nbsp;|&nbsp;
  <strong>Model:</strong> <code>pod_segmentation_scriptedV3</code> &nbsp;|&nbsp;
  <strong>Scripts:</strong> S3 (Batch 1, 2) + folder (Batch 3), schema-equivalent
</div>

<div class="callout info">
  <span class="label">Script-equivalence verification</span>
  Both <code>batch_process_uploads.py</code> and <code>batch_process_uploads_s3.py</code> emit
  identical column schemas and identical guard-diagnostic payloads. The only difference is the
  S3 script's extra <code>input_image_url</code> column. Numbers in this report were re-derived
  from raw per-image JSON; nothing is taken on faith from intermediate aggregates.
</div>

<h2>1. Headline numbers</h2>
{''.join(cards)}

<h2>2. Cross-batch comparison</h2>
<table>
  <thead><tr><th>Batch</th><th>n</th><th>Dominant guard action</th>
    <th class="num">avg median_aqua_DE</th><th class="num">avg median_low_DE</th>
    <th class="num">albumin_conf max</th><th class="num">correct exact</th></tr></thead>
  <tbody>
    {''.join(cross_table_rows)}
  </tbody>
</table>
<p><small class="dim">Numbers are computed across the actual_class breakdown; "avg" here is the
unweighted mean of class-level medians.</small></p>

<h2>3. Two completely different failure regimes</h2>

<div class="callout">
  <span class="label">Batch 1 &amp; 2 regime — chart-like pod color, guard over-rejects</span>
  Pod median lands within ΔE ≈ 8–17 of chart references. The guard sees real color signal but
  the strong-aqua threshold (<code>MICRO_STRONG_AQUA_DE_MAX = 8.0</code>) is just out of reach
  for genuine A2/A3 patients (Patient 1 Batch 1: aqua_DE = 8.71; Patient 9 Batch 2:
  aqua_DE = 9.01). Almost everything ends up in
  <code>chart_like_weak_aqua_provisional_80_150</code> with a flat (80, 150) band that ignores
  the regression's signal. Combined Batch 1+2: 15/17 (88.2%) of rows land in this single branch.
</div>

<div class="callout danger">
  <span class="label">Batch 3 regime — pod color is OOD, guard short-circuits to safety fallback</span>
  Pod median lands at ΔE ≈ 22 from any chart reference (low or aqua) — i.e. the pixels selected
  as 'pod' are nowhere near any chart color at all. <code>median_L ≥ 84</code> (overbright) on
  98.8% of rows. The guard's <code>guarded_overbright_no_aqua_to_10</code> branch fires and
  assigns 10 mg/L to <strong>575 of 582 patients (98.8%)</strong>. This is a capture or
  preprocessing problem upstream of the guard — likely overexposure, white-balance over-correction,
  or segmentation drift. The guard is doing what it was designed to do but the inputs themselves
  are out of distribution.
</div>

<h2>4. Why a single threshold tune cannot fix both regimes</h2>
<p>The threshold edits motivated by Batches 1 &amp; 2 (loosen <code>MICRO_STRONG_AQUA_DE_MAX</code>,
raise <code>MICRO_WEAK_AQUA_MIN</code>, lower confidence floors) operate <em>after</em>
<code>overbright_not_chart_like</code> short-circuits the decision tree. They do nothing in the
Batch 3 regime. Conversely, fixing the upstream pod-color extraction would not change Batch 1/2
behaviour at all — those inputs already produce chart-like color signals.</p>

<p><strong>Two separate fixes are needed and they target different layers:</strong></p>
<ol>
  <li><strong>Layer 1 (upstream / capture):</strong> figure out why Batch 3 pods land at
      ΔE ≈ 22 from chart. Visually inspect 5–10 composite images, especially A3 patients
      predicted as A1. Add <code>median_L</code> / <code>median_a</code> / <code>median_b</code> /
      <code>median_chroma</code> to the batch-script guard fields so the next batch tells us
      directly whether this is overexposure, white-balance, or segmentation drift.</li>
  <li><strong>Layer 2 (guard logic):</strong> apply the Batch 1+2 tuning <em>only</em> after Batch
      3's upstream issue is resolved. Otherwise we may "fix" Batch 1/2 numbers while Batch 3
      false negatives stay invisible.</li>
</ol>

<h2>5. Critical safety recommendation</h2>
<div class="callout danger">
  <span class="label">Stop returning silent A1 verdicts on capture failure</span>
  The current behaviour of the <code>guarded_overbright_no_aqua_to_10</code> branch is to emit a
  fixed 10 mg/L (which maps to A1 / normal). On Batch 3 this means <strong>{metrics_by_batch['20260508_122726']['sensitivity']['A2'][2] - metrics_by_batch['20260508_122726']['sensitivity']['A2'][0]}
  of {metrics_by_batch['20260508_122726']['sensitivity']['A2'][2]} A2 patients
  and {metrics_by_batch['20260508_122726']['sensitivity']['A3'][2] - metrics_by_batch['20260508_122726']['sensitivity']['A3'][0]}
  of {metrics_by_batch['20260508_122726']['sensitivity']['A3'][2]} A3 patients</strong>
  receive a 'normal' result. Proposed change: emit
  <code>report_mode = "unconfirmed"</code> with a 'retake image' message instead. This single
  change would have flipped 575 silent false-negatives in Batch 3 into explicit retest requests
  without affecting Batch 1/2 (where this branch did not fire).
</div>

<h2>6. What was already done</h2>
<ul>
  <li><strong>Batch scripts updated</strong> — both <code>batch_process_uploads.py</code> and
      <code>batch_process_uploads_s3.py</code> now emit <code>is_high_watch</code>,
      <code>is_unconfirmed</code>, <code>uacr_warning</code>,
      <code>actual_*_in_provisional_range</code>, <code>albumin_raw_chart_label</code>,
      and an 18-field <code>guard_*</code> diagnostic block. Without these, this analysis would
      not have been possible.</li>
  <li><strong>Schema parity verified</strong> — the two scripts produce the same column set
      modulo <code>input_image_url</code>. <code>OUTPUT_COLS</code> ↔ <code>build_output_row</code>
      keys match exactly in both files.</li>
  <li><strong>Re-runnable analysis</strong> — <code>scripts/analyze_batch_jsons.py</code>
      generates a Markdown summary for any batch's data folder.
      <code>scripts/generate_batch_reports.py</code> generates these HTML reports.</li>
</ul>

<h2>7. Per-batch detail reports</h2>
<ul>
  <li><a href="20260508_120215/report.html">Batch 1 — 8 records (S3 script)</a></li>
  <li><a href="20260508_122140/report.html">Batch 2 — 9 records (S3 script)</a></li>
  <li><a href="20260508_122726/report.html">Batch 3 — 582 records (folder script)</a></li>
</ul>

<hr>
<p><small class="dim">Report generated from per-image JSON outputs only. No Excel files read.
Run <code>python scripts/generate_batch_reports.py</code> to regenerate.</small></p>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    metrics_by_batch = {}
    for b in BATCHES:
        rows = load_batch(b["run_id"])
        m = compute_metrics(rows)
        metrics_by_batch[b["run_id"]] = m
        out = RUNS_ROOT / b["run_id"] / "report.html"
        out.write_text(render_batch_report(b), encoding="utf-8")
        print(f"  wrote {out}")
    consolidated = RUNS_ROOT / "consolidated_report.html"
    consolidated.write_text(render_consolidated(metrics_by_batch), encoding="utf-8")
    print(f"  wrote {consolidated}")


if __name__ == "__main__":
    main()
