"""
Streamlit dashboard for batch run results.

Run:
    streamlit run scripts/dashboard.py
"""

import io
import warnings

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Strip Analyzer Results", layout="wide", initial_sidebar_state="expanded")

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "displayModeBar": False,
}

BIAS_Y_COL = "Mean error (actual - predicted)"
ERROR_LABEL = "actual - predicted"
BIN_ERROR_LABEL = "Bin error (predicted - expected)"


def _plotly(fig, container=st):
    """Render Plotly charts without hijacking page scrolling or drag gestures."""
    fig.update_layout(dragmode=False)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    container.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def _add_log(logs, message):
    if message:
        logs.append(str(message))


def _fmt_pct(value, decimals=0, na="--"):
    if value is None or pd.isna(value):
        return na
    return f"{value:.{decimals}%}"


def _fmt_num(value, decimals=2, na="--"):
    if value is None or pd.isna(value):
        return na
    return f"{value:.{decimals}f}"


def _bin_sort_key(value):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not pd.isna(numeric):
        return (0, float(numeric))
    return (1, str(value))


def _per_bin_accuracy_table(df, expected_col, predicted_col):
    if df.empty:
        return pd.DataFrame()
    rows = []
    ordered_bins = sorted(df[expected_col].dropna().unique(), key=_bin_sort_key)
    for expected_bin in ordered_bins:
        grp = df[df[expected_col] == expected_bin].copy()
        grp = grp.copy()
        exact_rate = (grp[predicted_col] == grp[expected_col]).mean()
        rows.append(
            {
                "Expected bin": str(expected_bin),
                "Samples": len(grp),
                "Exact match": _fmt_pct(exact_rate, 0),
                "Top predicted bin": str(grp[predicted_col].mode().iloc[0]) if not grp[predicted_col].mode().empty else "--",
            }
        )
    return pd.DataFrame(rows)


def _bin_crosstab_table(df, expected_col, predicted_col):
    if df.empty:
        return pd.DataFrame()
    expected_bins = sorted(df[expected_col].dropna().unique(), key=_bin_sort_key)
    predicted_bins = sorted(df[predicted_col].dropna().unique(), key=_bin_sort_key)
    table = pd.crosstab(df[expected_col], df[predicted_col])
    table = table.reindex(index=expected_bins, columns=predicted_bins, fill_value=0)
    table.index = table.index.astype(str)
    table.columns = table.columns.astype(str)
    table.index.name = "Expected bin"
    return table.reset_index()


def _report_rows_from_df(df, max_rows=30):
    if df is None or df.empty:
        return []
    clean = df.head(max_rows).fillna("").astype(str)
    return [clean.columns.tolist()] + clean.values.tolist()


def _display_df(df):
    """Keep Streamlit display tables Arrow-safe when columns mix counts and labels."""
    if df is None or df.empty:
        return pd.DataFrame()
    return df.fillna("").astype(str)


def _bin_error_interpretation(df, error_col):
    if df.empty or error_col not in df.columns:
        return "No bin-error rows were available."

    errs = pd.to_numeric(df[error_col], errors="coerce").dropna()
    if errs.empty:
        return "No bin-error rows were available."

    exact = float((errs == 0).mean())
    near = float((errs.abs() <= 1).mean())
    mean_err = float(errs.mean())

    if mean_err > 0.2:
        direction = "slight tendency to read high"
    elif mean_err < -0.2:
        direction = "slight tendency to read low"
    else:
        direction = "no clear high/low tendency"

    return (
        f"Exact matches: {_fmt_pct(exact, 0)}. "
        f"Within +/-1 bin: {_fmt_pct(near, 0)}. "
        f"Mean signed bin error: {_fmt_num(mean_err, 2)}. "
        f"Observed pattern: {direction}."
    )


def _table(data, col_widths=None, header_bg=colors.HexColor("#E8EEF8")):
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), header_bg),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7E2")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def _build_pdf_report(audience, run_metadata, summary_metrics, findings, tables):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.55 * inch,
        leftMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )
    styles = getSampleStyleSheet()
    story = []

    title = "Clinical Findings Report" if audience == "clinician" else "Technical Findings Report"
    subtitle = "Urine Strip Analyzer Batch Summary"
    story.append(Paragraph(title, styles["Title"]))
    story.append(Paragraph(subtitle, styles["Heading3"]))
    story.append(Spacer(1, 12))

    metadata_rows = [["Field", "Value"]]
    for label, value in run_metadata:
        metadata_rows.append([label, value or "--"])
    story.append(Paragraph("Batch Details", styles["Heading2"]))
    story.append(_table(metadata_rows, col_widths=[2.0 * inch, 4.7 * inch]))
    story.append(Spacer(1, 12))

    metric_rows = [["Metric", "Value"]]
    for label, value in summary_metrics:
        metric_rows.append([label, value])
    story.append(Paragraph("Summary Metrics", styles["Heading2"]))
    story.append(_table(metric_rows, col_widths=[3.2 * inch, 3.5 * inch]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Key Findings", styles["Heading2"]))
    for item in findings:
        story.append(Paragraph(f"- {item}", styles["BodyText"]))
        story.append(Spacer(1, 4))
    story.append(Spacer(1, 8))

    for section_title, rows in tables:
        if len(rows) <= 1:
            continue
        story.append(Paragraph(section_title, styles["Heading2"]))
        story.append(_table(rows))
        story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ---------------------------------------------------------------------------
# Sidebar - upload only
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Strip Analyzer")
    uploaded = st.file_uploader("Upload results.xlsx", type=["xlsx"], label_visibility="collapsed")
    st.caption("Upload the `results.xlsx` from a batch run.")

    st.markdown("---")
    st.markdown(
        "**UACR stages**\n"
        "| Stage | Value | Risk |\n"
        "|---|---|---|\n"
        "| A1 | < 30 mg/G | Normal |\n"
        "| A2 | 30-300 mg/G | Elevated |\n"
        "| A3 | > 300 mg/G | High risk |"
    )

if uploaded is None:
    st.title("Urine Strip Analyzer - Results Dashboard")
    st.info("Upload a `results.xlsx` file using the sidebar to begin.")
    st.stop()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading results...")
def _load_results(file_bytes: bytes) -> pd.DataFrame:
    """Load and clean the results Excel. Cached per uploaded file so re-runs
    caused by widget interactions don't re-read or re-clean the file."""

    def _safe_to_float(x):
        if pd.isna(x):
            return float("nan")
        cleaned = "".join(c for c in str(x) if c.isdigit() or c in ".-")
        try:
            return float(cleaned) if cleaned else float("nan")
        except ValueError:
            return float("nan")

    _NUMERIC_COLS = [
        "actual_uacr", "actual_albumin", "actual_creatinine",
        "predicted_uacr", "predicted_albumin", "predicted_creatinine",
        "actual_vs_pred_uacr_delta", "actual_vs_pred_albumin_delta", "actual_vs_pred_creatinine_delta",
        "uacr_delta_legacy_vs_corrected",
        "predicted_uacr_provisional_range_low", "predicted_uacr_provisional_range_high",
        "predicted_albumin_provisional_range_low", "predicted_albumin_provisional_range_high",
        "uacr_confidence", "albumin_confidence", "creatinine_confidence",
        "albumin_bin_error", "creatinine_bin_error",
        "inference_time_sec",
    ]
    df = pd.read_excel(io.BytesIO(file_bytes))
    for _col in _NUMERIC_COLS:
        if _col in df.columns:
            df[_col] = df[_col].apply(_safe_to_float)
    return df


df_raw = _load_results(uploaded.getvalue())
df = df_raw.copy()
dashboard_logs = []

if "status" in df.columns:
    df_ok = df[df["status"] == "success"].copy()
    failed_rows = int((df["status"] != "success").sum())
    _add_log(dashboard_logs, f"Loaded {len(df)} rows from upload.")
    _add_log(dashboard_logs, f"Successful rows: {len(df_ok)}.")
    if failed_rows:
        _add_log(dashboard_logs, f"Failed rows present: {failed_rows}.")
else:
    df_ok = df.copy()
    _add_log(dashboard_logs, "No 'status' column found; using all rows in the dashboard.")

CLASSES = ["A1", "A2", "A3"]

for required_col in ["actual_class", "predicted_class"]:
    if required_col not in df_ok.columns:
        df_ok[required_col] = ""
        _add_log(dashboard_logs, f"Missing column '{required_col}' in upload; classification views may be incomplete.")

df_ok["actual_class"] = df_ok["actual_class"].fillna("").astype(str).str.upper().str.strip()
df_ok["predicted_class"] = df_ok["predicted_class"].fillna("").astype(str).str.upper().str.strip()

df_cls = df_ok[
    df_ok["actual_class"].isin(CLASSES) & df_ok["predicted_class"].isin(CLASSES)
].copy()

has_cls = not df_cls.empty
y_true = df_cls["actual_class"] if has_cls else pd.Series(dtype=str)
y_pred = df_cls["predicted_class"] if has_cls else pd.Series(dtype=str)
labels_present = sorted(set(y_true) | set(y_pred)) if has_cls else []

if has_cls:
    _add_log(dashboard_logs, f"Rows with valid actual/predicted stage pairs: {len(df_cls)}.")
else:
    _add_log(dashboard_logs, "No rows with valid actual/predicted stage pairs were found.")

acc = accuracy_score(y_true, y_pred) if has_cls else None
f1 = f1_score(y_true, y_pred, average="macro", labels=labels_present, zero_division=0) if has_cls else None

total = len(df)
success = len(df_ok)
failed = total - success
prov_rate = df_ok["is_provisional"].mean() if "is_provisional" in df_ok.columns else None
retest_rate = df_ok["retest_required"].mean() if "retest_required" in df_ok.columns else None

run_id_value = str(df["run_id"].dropna().iloc[0]) if "run_id" in df.columns and not df["run_id"].dropna().empty else "--"
model_value = str(df["model_version"].dropna().iloc[0]) if "model_version" in df.columns and not df["model_version"].dropna().empty else "--"
timestamp_series = df["timestamp"].dropna() if "timestamp" in df.columns else pd.Series(dtype=str)
started_value = str(timestamp_series.min()) if not timestamp_series.empty else "--"
ended_value = str(timestamp_series.max()) if not timestamp_series.empty else "--"
workers_value = str(int(df["workers_used"].dropna().iloc[0])) if "workers_used" in df.columns and not df["workers_used"].dropna().empty else "--"

stage_report_rows = [["Stage", "Precision", "Recall", "F1", "Samples"]]
if has_cls:
    stage_report = classification_report(
        y_true, y_pred, labels=labels_present, output_dict=True, zero_division=0
    )
    class_counts = y_true.value_counts()
    for cls in labels_present:
        if cls in stage_report:
            stage_report_rows.append(
                [
                    cls,
                    _fmt_num(stage_report[cls]["precision"], 3),
                    _fmt_num(stage_report[cls]["recall"], 3),
                    _fmt_num(stage_report[cls]["f1-score"], 3),
                    str(int(class_counts.get(cls, 0))),
                ]
            )

mae_rows = [["Analyte", "MAE", "Samples"]]
for name, ac, pc in [
    ("UACR (mg/G)", "actual_uacr", "predicted_uacr"),
    ("Albumin (mg/L)", "actual_albumin", "predicted_albumin"),
    ("Creatinine (mg/dl)", "actual_creatinine", "predicted_creatinine"),
]:
    if ac in df_ok.columns and pc in df_ok.columns:
        sub = df_ok[df_ok[[ac, pc]].notna().all(axis=1)]
        if not sub.empty:
            mae_rows.append([name, _fmt_num(mean_absolute_error(sub[ac], sub[pc]), 2), str(len(sub))])

flag_impact_rows = [["Issue", "Accuracy Drop", "Flagged Samples"]]
for flag_col, flag_label in [
    ("albumin_flag_glare", "Albumin glare"),
    ("albumin_flag_non_uniform", "Albumin non-uniform lighting"),
    ("albumin_flag_mask_quality", "Albumin strip detection quality"),
    ("creatinine_flag_glare", "Creatinine glare"),
    ("creatinine_flag_non_uniform", "Creatinine non-uniform lighting"),
    ("creatinine_flag_mask_quality", "Creatinine strip detection quality"),
]:
    if has_cls and flag_col in df_cls.columns:
        clean = df_cls[df_cls[flag_col] == 0]
        flagged = df_cls[df_cls[flag_col] == 1]
        if len(clean) > 0 and len(flagged) > 0:
            drop = accuracy_score(clean["actual_class"], clean["predicted_class"]) - accuracy_score(
                flagged["actual_class"], flagged["predicted_class"]
            )
            flag_impact_rows.append([flag_label, _fmt_pct(drop, 1), str(len(flagged))])

provisional_rows = [["Measure", "Value"]]
provisional_rows.append(["Provisional result rate", _fmt_pct(prov_rate, 1)])
if "is_provisional" in df_ok.columns:
    provisional_rows.append(["Provisional result count", str(int((df_ok["is_provisional"] == True).sum()))])
if "retest_required" in df_ok.columns:
    provisional_rows.append(["Retest recommended rate", _fmt_pct(retest_rate, 1)])

mae_export_rows = [["Analyte", "MAE", "Samples", "Evaluation basis"]]
for row in mae_rows[1:]:
    analyte = row[0]
    basis = "All successful rows"
    if analyte in {"UACR (mg/G)", "Albumin (mg/L)"}:
        basis = "Exact-mode rows only"
    mae_export_rows.append([analyte, row[1], row[2], basis])

bias_cols = {
    "UACR (mg/G)": ("actual_vs_pred_uacr_delta", "actual_class"),
    "Albumin (mg/L)": ("actual_vs_pred_albumin_delta", "actual_class"),
    "Creatinine (mg/dl)": ("actual_vs_pred_creatinine_delta", "actual_class"),
}
bias_rows = []
for label, (delta_col, cls_col) in bias_cols.items():
    if delta_col in df_ok.columns and cls_col in df_ok.columns:
        sub = df_ok[df_ok[[delta_col, cls_col]].notna().all(axis=1)]
        sub = sub[sub[cls_col].isin(CLASSES)]
        for stage, grp in sub.groupby(cls_col):
            bias_rows.append({
                "Measurement": label,
                "Stage": stage,
                BIAS_Y_COL: round(grp[delta_col].mean(), 2),
            })

bias_export_rows = [["Measurement", "Stage", BIAS_Y_COL]]
for row in bias_rows:
    bias_export_rows.append([row["Measurement"], row["Stage"], _fmt_num(row[BIAS_Y_COL], 2)])

stage_distribution_rows = [["Stage", "Successful rows"]]
if "actual_class" in df_ok.columns:
    stage_counts = df_ok[df_ok["actual_class"].isin(CLASSES)]["actual_class"].value_counts()
    for stage in CLASSES:
        if stage in stage_counts:
            stage_distribution_rows.append([stage, str(int(stage_counts[stage]))])

albumin_bin_cols = [
    "predicted_albumin_bin",
    "expected_albumin_bin",
    "albumin_bin_error",
    "albumin_within_1_bin",
]
creatinine_bin_cols = [
    "predicted_creatinine_bin",
    "expected_creatinine_bin",
    "creatinine_bin_error",
    "creatinine_within_1_bin",
]
missing_albumin_bin_cols = [col for col in albumin_bin_cols if col not in df_ok.columns]
missing_creatinine_bin_cols = [col for col in creatinine_bin_cols if col not in df_ok.columns]

if missing_albumin_bin_cols:
    _add_log(dashboard_logs, "Missing albumin bin columns: " + ", ".join(missing_albumin_bin_cols))
else:
    _add_log(dashboard_logs, "Albumin bin columns detected in upload.")

if missing_creatinine_bin_cols:
    _add_log(dashboard_logs, "Missing creatinine bin columns: " + ", ".join(missing_creatinine_bin_cols))
else:
    _add_log(dashboard_logs, "Creatinine bin columns detected in upload.")

has_albumin_bin_cols = not missing_albumin_bin_cols
has_creatinine_bin_cols = not missing_creatinine_bin_cols

df_bin_alb_report = pd.DataFrame()
df_bin_cre_report = pd.DataFrame()
if has_albumin_bin_cols:
    df_bin_alb_report = df_ok[df_ok["albumin_bin_error"].notna()].copy()
    if not df_bin_alb_report.empty:
        df_bin_alb_report["albumin_bin_error"] = df_bin_alb_report["albumin_bin_error"].astype(int)
if has_creatinine_bin_cols:
    df_bin_cre_report = df_ok[df_ok["creatinine_bin_error"].notna()].copy()
    if not df_bin_cre_report.empty:
        df_bin_cre_report["creatinine_bin_error"] = df_bin_cre_report["creatinine_bin_error"].astype(int)

albumin_exact_mode_rows = 0
if "albumin_bin_error" in df_ok.columns:
    albumin_exact_mode_rows = int(df_ok["albumin_bin_error"].notna().sum())
    if albumin_exact_mode_rows == 0:
        if "is_provisional" in df_ok.columns and not df_ok.empty and bool((df_ok["is_provisional"] == True).all()):
            _add_log(dashboard_logs, "Albumin bin metrics are empty because all successful rows are provisional.")
        else:
            _add_log(dashboard_logs, "Albumin bin column exists but no exact-mode rows were available.")

pod_summary_rows = [["Pod metric", "Albumin", "Creatinine"]]
albumin_bin_scored = int(df_ok["albumin_bin_error"].notna().sum()) if "albumin_bin_error" in df_ok.columns else 0
creatinine_bin_scored = int(df_ok["creatinine_bin_error"].notna().sum()) if "creatinine_bin_error" in df_ok.columns else 0
albumin_exact_match = (
    _fmt_pct((df_ok.loc[df_ok["albumin_bin_error"].notna(), "albumin_bin_error"] == 0).mean(), 0)
    if "albumin_bin_error" in df_ok.columns and albumin_bin_scored > 0 else "n/a"
)
creatinine_exact_match = (
    _fmt_pct((df_ok.loc[df_ok["creatinine_bin_error"].notna(), "creatinine_bin_error"] == 0).mean(), 0)
    if "creatinine_bin_error" in df_ok.columns and creatinine_bin_scored > 0 else "n/a"
)
albumin_pm1 = (
    _fmt_pct(df_ok.loc[df_ok["albumin_within_1_bin"].notna(), "albumin_within_1_bin"].mean(), 0)
    if "albumin_within_1_bin" in df_ok.columns and df_ok["albumin_within_1_bin"].notna().any() else "n/a"
)
creatinine_pm1 = (
    _fmt_pct(df_ok.loc[df_ok["creatinine_within_1_bin"].notna(), "creatinine_within_1_bin"].mean(), 0)
    if "creatinine_within_1_bin" in df_ok.columns and df_ok["creatinine_within_1_bin"].notna().any() else "n/a"
)
pod_summary_rows.extend(
    [
        ["Rows scored", str(albumin_bin_scored) if "albumin_bin_error" in df_ok.columns else "missing", str(creatinine_bin_scored) if "creatinine_bin_error" in df_ok.columns else "missing"],
        ["Exact bin match", albumin_exact_match, creatinine_exact_match],
        ["Within +/-1 bin", albumin_pm1, creatinine_pm1],
    ]
)

albumin_per_bin_rows = _report_rows_from_df(
    _per_bin_accuracy_table(df_bin_alb_report, "expected_albumin_bin", "predicted_albumin_bin")
)
creatinine_per_bin_rows = _report_rows_from_df(
    _per_bin_accuracy_table(df_bin_cre_report, "expected_creatinine_bin", "predicted_creatinine_bin")
)
albumin_bin_confusion_rows = _report_rows_from_df(
    _bin_crosstab_table(df_bin_alb_report, "expected_albumin_bin", "predicted_albumin_bin")
)
creatinine_bin_confusion_rows = _report_rows_from_df(
    _bin_crosstab_table(df_bin_cre_report, "expected_creatinine_bin", "predicted_creatinine_bin")
)

provisional_coverage_rows = [["Measure", "Value"]]
df_prov_report = (
    df_ok[df_ok["is_provisional"] == True].copy()
    if "is_provisional" in df_ok.columns else pd.DataFrame()
)
provisional_coverage_rows.append(["Provisional rows", str(len(df_prov_report))])
provisional_coverage_rows.append(["Exact rows", str(len(df_ok) - len(df_prov_report))])
range_cols = ["actual_uacr", "predicted_uacr_provisional_range_low", "predicted_uacr_provisional_range_high"]
if not df_prov_report.empty and all(c in df_prov_report.columns for c in range_cols):
    dr = df_prov_report[df_prov_report[range_cols].notna().all(axis=1)]
    if not dr.empty:
        in_range = (
            (dr["actual_uacr"] >= dr["predicted_uacr_provisional_range_low"]) &
            (dr["actual_uacr"] <= dr["predicted_uacr_provisional_range_high"])
        )
        range_width = dr["predicted_uacr_provisional_range_high"] - dr["predicted_uacr_provisional_range_low"]
        provisional_coverage_rows.extend(
            [
                ["Rows with lab value and provisional range", str(len(dr))],
                ["True UACR inside provisional range", _fmt_pct(in_range.mean(), 1)],
                ["Average provisional range width (mg/G)", _fmt_num(range_width.mean(), 1)],
            ]
        )

provisional_boundary_rows = [["Predicted boundary", "Count", "Clinical meaning"]]
if not df_prov_report.empty and "provisional_class" in df_prov_report.columns:
    boundary_counts = df_prov_report["provisional_class"].fillna("Unknown").value_counts()
    boundary_meaning = {
        "A1_A2_boundary_provisional": "Normal / Elevated boundary",
        "A2_A3_boundary_provisional": "Elevated / High-risk boundary",
    }
    for boundary, count in boundary_counts.items():
        provisional_boundary_rows.append([boundary, str(int(count)), boundary_meaning.get(boundary, boundary)])

clinician_findings = []
if has_cls:
    clinician_findings.append(f"Correct stage identification rate was {_fmt_pct(acc, 0)} across {len(df_cls)} classifiable samples.")
    clinician_findings.append(f"{int(round((1 - acc) * len(df_cls)))} patients were assigned to the wrong stage.")
else:
    clinician_findings.append("No rows contained both confirmed and predicted stage labels, so stage accuracy could not be summarized.")
if prov_rate is not None:
    clinician_findings.append(f"Provisional results occurred in {_fmt_pct(prov_rate, 0)} of successful samples.")
if retest_rate is not None:
    clinician_findings.append(f"Retest was recommended for {_fmt_pct(retest_rate, 0)} of successful samples.")
if len(stage_report_rows) > 1:
    lowest_recall_row = min(stage_report_rows[1:], key=lambda row: float(row[2]))
    clinician_findings.append(
        f"The hardest group to detect was {lowest_recall_row[0]}, with recall {lowest_recall_row[2]}."
    )

technical_findings = []
if has_cls:
    technical_findings.append(
        f"Classification accuracy was {_fmt_num(acc, 3)} and macro F1 was {_fmt_num(f1, 3)} on {len(df_cls)} labeled rows."
    )
else:
    technical_findings.append("No labeled class pairs were available, so classification metrics could not be computed.")
if len(mae_rows) > 1:
    worst_mae = max(mae_rows[1:], key=lambda row: float(row[1]))
    technical_findings.append(f"The largest numeric error was for {worst_mae[0]} with MAE {worst_mae[1]}.")
technical_findings.append(
    f"UACR and albumin numeric MAE are computed on exact-mode rows only, while creatinine MAE uses all rows with numeric predictions."
)
if albumin_bin_scored or creatinine_bin_scored:
    technical_findings.append(
        f"Pod-level scoring rows: albumin {albumin_bin_scored}, creatinine {creatinine_bin_scored}."
    )
if len(flag_impact_rows) > 1:
    worst_flag = max(flag_impact_rows[1:], key=lambda row: float(row[1].rstrip('%')))
    technical_findings.append(
        f"The biggest observed quality-related accuracy drop was {worst_flag[1]} for {worst_flag[0]}."
    )
if prov_rate is not None:
    technical_findings.append(f"Provisional mode was triggered on {_fmt_pct(prov_rate, 1)} of successful rows.")

with st.sidebar:
    st.markdown("---")
    st.markdown("### Experiment Run")

    run_ids = df["run_id"].dropna().unique() if "run_id" in df.columns else []
    if len(run_ids) == 1:
        st.markdown(f"**Run ID:** `{run_ids[0]}`")

    model_versions = df["model_version"].dropna().unique() if "model_version" in df.columns else []
    if len(model_versions) == 1:
        st.markdown(f"**Model:** `{model_versions[0]}`")

    timestamps = df["timestamp"].dropna() if "timestamp" in df.columns else pd.Series(dtype=str)
    if not timestamps.empty:
        st.markdown(f"**Started:** {str(timestamps.min())}")
        st.markdown(f"**Ended:** {str(timestamps.max())}")

    workers_col = df["workers_used"].dropna().unique() if "workers_used" in df.columns else []
    if len(workers_col) >= 1:
        st.markdown(f"**Workers:** {int(workers_col[0])}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("Total samples", total)
    c2.metric("Successful", success)
    if failed:
        st.warning(f"{failed} failed - see raw data tab")

    if "inference_time_sec" in df_ok.columns and not timestamps.empty:
        try:
            t_start = pd.to_datetime(timestamps.min())
            t_end = pd.to_datetime(timestamps.max())
            wall_sec = (t_end - t_start).total_seconds()
        except Exception:
            wall_sec = None
        cpu_total = df_ok["inference_time_sec"].sum()
        avg_time = df_ok["inference_time_sec"].mean()
        st.markdown("---")
        st.markdown("### Inference Time")
        t1, t2, t3 = st.columns(3)
        t1.metric("Wall time", f"{wall_sec:.0f}s" if wall_sec is not None else "--")
        t2.metric("Total CPU time", f"{cpu_total:.1f}s")
        t3.metric("Avg / image", f"{avg_time:.2f}s")

# ---------------------------------------------------------------------------
# View switcher / actions
# ---------------------------------------------------------------------------

toolbar_left, toolbar_right = st.columns([3, 1])
with toolbar_left:
    selected_view = st.radio(
        "Dashboard view",
        ["For Clinicians", "For Data Teams"],
        horizontal=True,
        label_visibility="collapsed",
    )

report_audience = "clinician" if selected_view == "For Clinicians" else "technical"
report_run_metadata = [
    ("Run ID", run_id_value),
    ("Model", model_value),
    ("Started", started_value),
    ("Ended", ended_value),
    ("Workers", workers_value),
]
report_summary_metrics = [
    ("Total uploaded rows", str(total)),
    ("Successful rows", str(success)),
    ("Failed rows", str(failed)),
]
if has_cls:
    report_summary_metrics.extend(
        [
            ("Stage accuracy", _fmt_pct(acc, 1)),
            ("Macro F1", _fmt_num(f1, 3)),
        ]
    )
if prov_rate is not None:
    report_summary_metrics.append(("Provisional rate", _fmt_pct(prov_rate, 1)))
if retest_rate is not None:
    report_summary_metrics.append(("Retest rate", _fmt_pct(retest_rate, 1)))

report_findings = clinician_findings if report_audience == "clinician" else technical_findings
report_tables = [
    ("Stage Distribution", stage_distribution_rows),
    ("Stage Performance", stage_report_rows),
    ("Measurement Error", mae_export_rows),
    ("Directional Bias", bias_export_rows),
    ("Provisional and Retest Summary", provisional_rows),
    ("Provisional Coverage", provisional_coverage_rows),
    ("Provisional Boundaries", provisional_boundary_rows),
]
report_tables.append(("Pod Summary", pod_summary_rows))
report_tables.append(("Albumin Per-Bin Summary", albumin_per_bin_rows))
report_tables.append(("Creatinine Per-Bin Summary", creatinine_per_bin_rows))
if report_audience == "technical":
    report_tables.append(("Albumin Raw Bin Confusion", albumin_bin_confusion_rows))
    report_tables.append(("Creatinine Raw Bin Confusion", creatinine_bin_confusion_rows))
    report_tables.append(("Image Quality Impact", flag_impact_rows))

pdf_bytes = _build_pdf_report(
    report_audience,
    report_run_metadata,
    report_summary_metrics,
    report_findings,
    report_tables,
)

pdf_filename = f"{run_id_value if run_id_value != '--' else 'batch'}_{report_audience}_findings.pdf"

with toolbar_right:
    st.download_button(
        label=f"Download {selected_view} PDF",
        data=pdf_bytes,
        file_name=pdf_filename,
        mime="application/pdf",
        use_container_width=True,
    )


def render_data_team_model_performance():
    st.markdown("## Model Performance")

    if has_cls:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc:.3f}")
        c2.metric("F1 (macro)", f"{f1:.3f}")
        c3.metric(
            "Precision (macro)",
            f"{precision_score(y_true, y_pred, average='macro', labels=labels_present, zero_division=0):.3f}",
        )
        c4.metric(
            "Recall (macro)",
            f"{recall_score(y_true, y_pred, average='macro', labels=labels_present, zero_division=0):.3f}",
        )

        col_cm2, col_cls = st.columns(2)
        with col_cm2:
            cm = confusion_matrix(y_true, y_pred, labels=labels_present)
            fig = ff.create_annotated_heatmap(
                z=cm, x=labels_present, y=labels_present, colorscale="Blues", showscale=True,
            )
            fig.update_layout(
                xaxis_title="Predicted ->", yaxis_title="<- True",
                height=340, margin=dict(t=10, b=40), title="Confusion matrix",
            )
            _plotly(fig)

        with col_cls:
            report = classification_report(
                y_true, y_pred, labels=labels_present, output_dict=True, zero_division=0
            )
            rows = [
                {"Stage": cls, "Precision": round(report[cls]["precision"], 3),
                 "Recall": round(report[cls]["recall"], 3), "F1": round(report[cls]["f1-score"], 3)}
                for cls in labels_present if cls in report
            ]
            fig = px.bar(
                pd.DataFrame(rows).melt(id_vars="Stage"),
                x="Stage", y="value", color="variable", barmode="group",
                labels={"value": "Score (0-1)", "variable": "Metric"},
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Precision / Recall / F1 per stage",
            )
            fig.update_layout(height=340, yaxis_range=[0, 1.1], margin=dict(t=40))
            _plotly(fig)
    else:
        st.warning("No rows with both actual and predicted stage labels.")

    st.markdown("---")
    st.markdown("## Directional Bias")
    st.caption("Signed error grouped by stage and analyte.")

    if bias_rows:
        col_bias, col_dist = st.columns(2)
        with col_bias:
            fig = px.bar(
                pd.DataFrame(bias_rows),
                x="Stage", y=BIAS_Y_COL,
                color="Measurement", barmode="group",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Mean signed error by stage",
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=340, margin=dict(t=40))
            _plotly(fig)

        with col_dist:
            delta_data = []
            for label, (delta_col, _) in bias_cols.items():
                if delta_col in df_ok.columns:
                    for v in df_ok[delta_col].dropna():
                        delta_data.append({"Analyte": label, "Error": v})
            if delta_data:
                fig = px.box(
                    pd.DataFrame(delta_data), x="Analyte", y="Error",
                    color="Analyte", points="all",
                    color_discrete_sequence=px.colors.qualitative.Set1,
                    labels={"Error": ERROR_LABEL},
                    title="Error spread per analyte",
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.update_layout(showlegend=False, height=340, margin=dict(t=40))
                _plotly(fig)
    else:
        st.info("No delta columns available to compute bias.")

    st.markdown("---")
    st.markdown("## Measurement Accuracy (MAE)")
    if mae_rows:
        summary_df = pd.DataFrame(mae_rows)
        st.dataframe(_display_df(summary_df), use_container_width=True, hide_index=True)
    else:
        st.info("No actual/predicted value pairs available for MAE.")


def render_data_team_pod_bin():
    st.markdown("## Pod & Bin Level Analysis")
    st.caption(
        "This section is focused on pod-level color-chart performance. "
        "Albumin metrics use exact-mode rows only. Creatinine metrics use all rows with bin outputs."
    )

    has_albumin_bin_cols = not missing_albumin_bin_cols
    has_creatinine_bin_cols = not missing_creatinine_bin_cols

    if not has_albumin_bin_cols and not has_creatinine_bin_cols:
        st.info("No pod/bin columns were found in this upload. Re-run the batch script and upload the newest results file.")
        return

    df_bin_alb = pd.DataFrame()
    df_bin_cre = pd.DataFrame()
    if has_albumin_bin_cols:
        df_bin_alb = df_ok[df_ok["albumin_bin_error"].notna()].copy()
        if not df_bin_alb.empty:
            df_bin_alb["albumin_bin_error"] = df_bin_alb["albumin_bin_error"].astype(int)
    if has_creatinine_bin_cols:
        df_bin_cre = df_ok[df_ok["creatinine_bin_error"].notna()].copy()
        if not df_bin_cre.empty:
            df_bin_cre["creatinine_bin_error"] = df_bin_cre["creatinine_bin_error"].astype(int)

    summary_rows = []
    summary_rows.append({
        "Metric": "Rows scored",
        "Albumin": len(df_bin_alb) if has_albumin_bin_cols else "Missing columns",
        "Creatinine": len(df_bin_cre) if has_creatinine_bin_cols else "Missing columns",
    })
    summary_rows.append({
        "Metric": "Exact bin match",
        "Albumin": f"{(df_bin_alb['albumin_bin_error'] == 0).mean():.0%}" if has_albumin_bin_cols and not df_bin_alb.empty else "n/a",
        "Creatinine": f"{(df_bin_cre['creatinine_bin_error'] == 0).mean():.0%}" if has_creatinine_bin_cols and not df_bin_cre.empty else "n/a",
    })
    summary_rows.append({
        "Metric": "Within +/-1 bin",
        "Albumin": f"{df_bin_alb['albumin_within_1_bin'].mean():.0%}" if has_albumin_bin_cols and not df_bin_alb.empty else "n/a",
        "Creatinine": f"{df_bin_cre['creatinine_within_1_bin'].mean():.0%}" if has_creatinine_bin_cols and not df_bin_cre.empty else "n/a",
    })
    summary_rows.append({
        "Metric": "Mean signed bin error",
        "Albumin": f"{df_bin_alb['albumin_bin_error'].mean():.2f}" if has_albumin_bin_cols and not df_bin_alb.empty else "n/a",
        "Creatinine": f"{df_bin_cre['creatinine_bin_error'].mean():.2f}" if has_creatinine_bin_cols and not df_bin_cre.empty else "n/a",
    })
    summary_rows.append({
        "Metric": "Provisional rows excluded",
        "Albumin": str(int((df_ok["is_provisional"] == True).sum())) if "is_provisional" in df_ok.columns else "n/a",
        "Creatinine": "0",
    })
    st.dataframe(_display_df(pd.DataFrame(summary_rows)), use_container_width=True, hide_index=True)

    note_parts = []
    if missing_albumin_bin_cols:
        note_parts.append("Missing albumin columns: " + ", ".join(missing_albumin_bin_cols))
    if missing_creatinine_bin_cols:
        note_parts.append("Missing creatinine columns: " + ", ".join(missing_creatinine_bin_cols))
    if note_parts:
        st.warning(" | ".join(note_parts))

    if has_albumin_bin_cols and df_bin_alb.empty:
        if "is_provisional" in df_ok.columns and not df_ok.empty and bool((df_ok["is_provisional"] == True).all()):
            st.info("Albumin columns are present, but all successful rows are provisional so there are no exact-mode albumin rows to score.")
        else:
            st.info("Albumin columns are present, but no exact-mode albumin rows were available in this upload.")

    col_alb_bin, col_cre_bin = st.columns(2)
    with col_alb_bin:
        st.markdown("### Albumin Pod")
        if not has_albumin_bin_cols:
            st.info("Albumin pod/bin metrics unavailable for this upload.")
        elif not df_bin_alb.empty:
            fig = px.histogram(
                df_bin_alb, x="albumin_bin_error",
                color_discrete_sequence=["#3498db"],
                labels={"albumin_bin_error": BIN_ERROR_LABEL},
                title="Albumin bin error distribution",
            )
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=280, margin=dict(t=40))
            _plotly(fig)
            st.caption(_bin_error_interpretation(df_bin_alb, "albumin_bin_error"))
        else:
            st.info("No exact-mode albumin rows available.")

    with col_cre_bin:
        st.markdown("### Creatinine Pod")
        if not has_creatinine_bin_cols:
            st.info("Creatinine pod/bin metrics unavailable for this upload.")
        elif not df_bin_cre.empty:
            fig = px.histogram(
                df_bin_cre, x="creatinine_bin_error",
                color_discrete_sequence=["#e67e22"],
                labels={"creatinine_bin_error": BIN_ERROR_LABEL},
                title="Creatinine bin error distribution",
            )
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=280, margin=dict(t=40))
            _plotly(fig)
            st.caption(_bin_error_interpretation(df_bin_cre, "creatinine_bin_error"))
        else:
            st.info("No creatinine bin rows available.")

    with st.expander("Pod bias by stage"):
        bias_bin_rows = []
        if has_albumin_bin_cols and not df_bin_alb.empty:
            for stage, grp in df_bin_alb[df_bin_alb["actual_class"].isin(CLASSES)].groupby("actual_class"):
                bias_bin_rows.append({"Analyte": "Albumin", "Stage": stage, "Mean bin error": round(grp["albumin_bin_error"].mean(), 2)})
        if has_creatinine_bin_cols and not df_bin_cre.empty:
            for stage, grp in df_bin_cre[df_bin_cre["actual_class"].isin(CLASSES)].groupby("actual_class"):
                bias_bin_rows.append({"Analyte": "Creatinine", "Stage": stage, "Mean bin error": round(grp["creatinine_bin_error"].mean(), 2)})
        if bias_bin_rows:
            fig = px.bar(
                pd.DataFrame(bias_bin_rows),
                x="Stage", y="Mean bin error", color="Analyte", barmode="group",
                color_discrete_map={"Albumin": "#3498db", "Creatinine": "#e67e22"},
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=300, margin=dict(t=10))
            _plotly(fig)
        else:
            st.info("Not enough data to compute bin-level bias by stage.")

    with st.expander("Bin-wise performance tables"):
        alb_table, cre_table = st.columns(2)
        with alb_table:
            st.markdown("### Albumin per-bin summary")
            if has_albumin_bin_cols and not df_bin_alb.empty:
                st.caption("For each expected albumin bin: how many samples were there, how often was the exact bin correct, and what wrong answer happened most often.")
                st.dataframe(
                    _display_df(_per_bin_accuracy_table(df_bin_alb, "expected_albumin_bin", "predicted_albumin_bin")),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("Interpret bins with low sample counts cautiously.")
                st.caption("Albumin bin tables use exact-mode rows only. Provisional rows are excluded from exact bin scoring.")
            else:
                st.info("Albumin bin-wise table unavailable.")

        with cre_table:
            st.markdown("### Creatinine per-bin summary")
            if has_creatinine_bin_cols and not df_bin_cre.empty:
                st.caption("For each expected creatinine bin: how many samples were there, how often was the exact bin correct, and what wrong answer happened most often.")
                st.dataframe(
                    _display_df(_per_bin_accuracy_table(df_bin_cre, "expected_creatinine_bin", "predicted_creatinine_bin")),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption("Interpret bins with low sample counts cautiously.")
            else:
                st.info("Creatinine bin-wise table unavailable.")

    with st.expander("Raw bin confusion counts"):
        alb_cross, cre_cross = st.columns(2)
        with alb_cross:
            st.markdown("### Albumin raw confusion table")
            if has_albumin_bin_cols and not df_bin_alb.empty:
                st.caption("Rows are the true/expected bins. Columns are the model's predicted bins. Diagonal counts are correct; off-diagonal counts show the exact confusion pattern.")
                st.dataframe(
                    _display_df(_bin_crosstab_table(df_bin_alb, "expected_albumin_bin", "predicted_albumin_bin")),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Albumin crosstab unavailable.")
        with cre_cross:
            st.markdown("### Creatinine raw confusion table")
            if has_creatinine_bin_cols and not df_bin_cre.empty:
                st.caption("Rows are the true/expected bins. Columns are the model's predicted bins. Diagonal counts are correct; off-diagonal counts show the exact confusion pattern.")
                st.dataframe(
                    _display_df(_bin_crosstab_table(df_bin_cre, "expected_creatinine_bin", "predicted_creatinine_bin")),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Creatinine crosstab unavailable.")


def render_data_team_quality_reliability():
    st.markdown("## Quality & Reliability")
    st.caption("Image quality, repeat-failure patterns, and provisional-mode behavior.")

    flags = [
        ("albumin_flag_glare", "Albumin - Glare"),
        ("albumin_flag_non_uniform", "Albumin - Non-Uniform Lighting"),
        ("albumin_flag_mask_quality", "Albumin - Strip Not Detected Well"),
        ("creatinine_flag_glare", "Creatinine - Glare"),
        ("creatinine_flag_non_uniform", "Creatinine - Non-Uniform Lighting"),
        ("creatinine_flag_mask_quality", "Creatinine - Strip Not Detected Well"),
    ]

    if has_cls:
        flag_impact = []
        for flag_col, flag_label in flags:
            if flag_col in df_cls.columns:
                clean = df_cls[df_cls[flag_col] == 0]
                flagged = df_cls[df_cls[flag_col] == 1]
                if len(clean) > 0 and len(flagged) > 0:
                    drop = accuracy_score(clean["actual_class"], clean["predicted_class"]) - accuracy_score(flagged["actual_class"], flagged["predicted_class"])
                    flag_impact.append({
                        "Issue": flag_label,
                        "Accuracy drop when present": round(drop, 3),
                        "n_flagged": len(flagged),
                    })
        if flag_impact:
            df_impact = pd.DataFrame(flag_impact).sort_values("Accuracy drop when present", ascending=True)
            fig = px.bar(
                df_impact, x="Accuracy drop when present", y="Issue",
                orientation="h", text="n_flagged",
                color="Accuracy drop when present",
                color_continuous_scale=["#f9f9f9", "#e74c3c"],
            )
            fig.update_traces(texttemplate="n=%{text}", textposition="outside")
            fig.update_layout(height=360, showlegend=False, coloraxis_showscale=False, margin=dict(t=10, r=80))
            _plotly(fig)

    st.markdown("---")
    st.markdown("## Provisional Results")
    provisional_summary = pd.DataFrame(
        [
            {"Metric": "Provisional rate", "Value": _fmt_pct(prov_rate, 1)},
            {"Metric": "Retest rate", "Value": _fmt_pct(retest_rate, 1)},
            {"Metric": "Successful rows", "Value": success},
        ]
    )
    st.dataframe(_display_df(provisional_summary), use_container_width=True, hide_index=True)

    if "patient_id" in df_ok.columns and "retest_required" in df_ok.columns:
        st.markdown("---")
        st.markdown("## Repeat Failure Patterns")
        pt_stats = (
            df_ok.groupby("patient_id")
            .agg(
                total=("patient_id", "count"),
                retests=("retest_required", "sum"),
                provisional=("is_provisional", "sum") if "is_provisional" in df_ok.columns else ("patient_id", "count"),
            )
            .reset_index()
        )
        repeat = pt_stats[pt_stats["total"] > 1].sort_values("retests", ascending=False)
        if repeat.empty:
            st.info("Each patient appears only once in this batch.")
        else:
            st.dataframe(_display_df(repeat.head(25)), use_container_width=True, hide_index=True)


# ===========================================================================
# CLINICAL TAB  - plain language, 4 clear sections
# ===========================================================================

if selected_view == "For Clinicians":

    # -- 1. The bottom line --------------------------------------------------
    st.markdown("## How well did the ML model perform?")

    if has_cls:
        wrong = int(round((1 - acc) * len(df_cls)))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Correct stage identified",
            f"{acc:.0%}",
            help="Out of every 100 patients, this many were placed in the right risk category (A1 / A2 / A3).",
        )
        c2.metric(
            "Patients misclassified",
            f"{wrong} of {len(df_cls)}",
            help="Number of patients assigned to the wrong category.",
        )
        c3.metric(
            "Provisional results",
            f"{prov_rate:.0%}" if prov_rate is not None else "-",
            help="Results where the model was uncertain and gave a range instead of a single value. These need a retest.",
        )
        c4.metric(
            "Retest recommended",
            f"{retest_rate:.0%}" if retest_rate is not None else "-",
            help="How often the model flagged that the photo was unclear and the test should be repeated.",
        )
    else:
        st.warning("No rows with both a lab-confirmed and a predicted stage found.")

    st.markdown("---")

    # -- 2. Which stage caused the most errors? ------------------------------
    st.markdown("## Which patients were hardest to classify correctly?")
    st.caption(
        "A1 = normal, A2 = elevated, A3 = high risk. "
        "The bar shows what percentage of patients in each group the model got right. "
        "A low bar on A3 is the most critical - it means high-risk patients are being missed."
    )

    if has_cls:
        col_bar, col_cm = st.columns([1, 1])

        with col_bar:
            report = classification_report(
                y_true, y_pred, labels=labels_present, output_dict=True, zero_division=0
            )
            stage_rows = [
                {"Stage": cls, "Detection rate": report[cls]["recall"], "Label": f"{report[cls]['recall']:.0%}"}
                for cls in labels_present if cls in report
            ]
            fig = px.bar(
                pd.DataFrame(stage_rows),
                x="Stage", y="Detection rate", text="Label",
                color="Stage",
                color_discrete_map={"A1": "#2ecc71", "A2": "#f39c12", "A3": "#e74c3c"},
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                showlegend=False, height=320, yaxis_range=[0, 1.25],
                yaxis_title="% of patients correctly identified",
                margin=dict(t=10, b=10),
            )
            _plotly(fig)
            st.caption("Of all true A3 patients, what fraction did the model correctly flag as high-risk?")

        with col_cm:
            cm = confusion_matrix(y_true, y_pred, labels=labels_present)
            fig = ff.create_annotated_heatmap(
                z=cm, x=labels_present, y=labels_present,
                colorscale="Blues", showscale=False,
            )
            fig.update_layout(
                xaxis_title="Model predicted ->",
                yaxis_title="<- Lab confirmed",
                height=320, margin=dict(t=10, b=40),
            )
            _plotly(fig)
            st.caption(
                "Diagonal = correct. Off-diagonal = wrong. "
                "E.g. a number in the A2 row, A1 column means an A2 patient was reported as A1."
            )

    st.markdown("---")

    # -- 3. Does the model read high or low? --------------------------------
    st.markdown("## Does the ML model consistently read too high or too low?")
    st.caption(
        "Positive = model gave a lower reading than the lab. "
        "Negative = model gave a higher reading than the lab. "
        "Ideally each bar should be close to zero. "
        "A consistent direction suggests a systematic bias for that patient group."
    )

    if bias_rows:
        fig = px.bar(
            pd.DataFrame(bias_rows),
            x="Stage", y=BIAS_Y_COL,
            color="Measurement", barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=320, margin=dict(t=10))
        _plotly(fig)
    else:
        st.info("Not enough data to compute directional bias.")

    st.markdown("---")

    # -- 4. When should you trust the result? -------------------------------
    st.markdown("## When can you trust the ML model's result?")
    st.caption(
        "The model rates its own confidence after analysing each photo. "
        "If confidence is working correctly, High confidence should mean more accurate results - "
        "and retest-flagged results should have lower accuracy."
    )

    col_conf, col_rt = st.columns(2)

    with col_conf:
        if "uacr_confidence_bucket" in df_cls.columns and has_cls:
            rows = []
            for bucket in ["Low", "Moderate", "High"]:
                sub = df_cls[df_cls["uacr_confidence_bucket"] == bucket]
                if len(sub):
                    a = accuracy_score(sub["actual_class"], sub["predicted_class"])
                    rows.append({"Confidence": bucket, "Accuracy": a, "Label": f"{a:.0%} ({len(sub)} samples)"})
            if rows:
                fig = px.bar(
                    pd.DataFrame(rows), x="Confidence", y="Accuracy", text="Label",
                    color="Confidence",
                    color_discrete_map={"Low": "#e74c3c", "Moderate": "#f39c12", "High": "#2ecc71"},
                    category_orders={"Confidence": ["Low", "Moderate", "High"]},
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    showlegend=False, height=300, yaxis_range=[0, 1.25],
                    yaxis_title="Correct stage rate", margin=dict(t=10),
                )
                _plotly(fig)
                st.caption("Accuracy by model confidence level - High should be clearly better than Low.")

    with col_rt:
        if "retest_required" in df_cls.columns and has_cls:
            rows = []
            for val, label in [(False, "No retest needed"), (True, "Retest flagged")]:
                sub = df_cls[df_cls["retest_required"] == val]
                if len(sub):
                    a = accuracy_score(sub["actual_class"], sub["predicted_class"])
                    rows.append({"Group": label, "Accuracy": a, "Label": f"{a:.0%} ({len(sub)} samples)"})
            if rows:
                fig = px.bar(
                    pd.DataFrame(rows), x="Group", y="Accuracy", text="Label",
                    color="Group",
                    color_discrete_map={"No retest needed": "#2ecc71", "Retest flagged": "#e74c3c"},
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    showlegend=False, height=300, yaxis_range=[0, 1.25],
                    yaxis_title="Correct stage rate", margin=dict(t=10),
                )
                _plotly(fig)
                st.caption("If retest-flagged samples are less accurate, the flag is meaningful.")

    st.markdown("---")

    # -- 5. Provisional results -----------------------------------------------
    st.markdown("## When the ML model was uncertain - Provisional Results")
    st.markdown(
        "Sometimes the model cannot confidently assign a single value. "
        "Instead it reports a **range** (e.g. 'UACR is between 20-60 mg/G') and marks the result as **Provisional**. "
        "This is not a failure - it is the model being honest about uncertainty. "
        "**Provisional results should not be used for clinical decisions without a clearer retest image.**"
    )

    df_prov_clin = (
        df_ok[df_ok["is_provisional"] == True].copy()
        if "is_provisional" in df_ok.columns else pd.DataFrame()
    )
    n_prov = len(df_prov_clin)
    n_exact = len(df_ok) - n_prov

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Provisional results",
        f"{n_prov} of {len(df_ok)}",
        help="Number of results where the model gave a range instead of a single value.",
    )
    c2.metric(
        "Definitive (exact) results",
        str(n_exact),
        help="Results where the model was confident enough to give a single number.",
    )

    range_cols = ["actual_uacr", "predicted_uacr_provisional_range_low", "predicted_uacr_provisional_range_high"]
    if not df_prov_clin.empty and all(c in df_prov_clin.columns for c in range_cols):
        dr = df_prov_clin[df_prov_clin[range_cols].notna().all(axis=1)]
        if not dr.empty:
            in_range = (
                (dr["actual_uacr"] >= dr["predicted_uacr_provisional_range_low"]) &
                (dr["actual_uacr"] <= dr["predicted_uacr_provisional_range_high"])
            )
            c3.metric(
                "True value inside predicted range",
                f"{in_range.mean():.0%}",
                help="Of provisional results with a known lab value, how often did the true UACR fall inside the model's reported range. High = the range is still clinically useful even when uncertain.",
            )

    if not df_prov_clin.empty:
        col_bound, col_action = st.columns(2)

        with col_bound:
            if "provisional_class" in df_prov_clin.columns:
                bound_counts = (
                    df_prov_clin["provisional_class"]
                    .fillna("Unknown")
                    .value_counts()
                    .reset_index()
                )
                bound_counts.columns = ["Predicted boundary", "Count"]
                bound_counts["Clinical meaning"] = bound_counts["Predicted boundary"].map({
                    "A1_A2_boundary_provisional": "Normal / Elevated boundary",
                    "A2_A3_boundary_provisional": "Elevated / High-risk boundary",
                }).fillna(bound_counts["Predicted boundary"])
                fig = px.bar(
                    bound_counts,
                    x="Clinical meaning", y="Count", text="Count",
                    color="Clinical meaning",
                    color_discrete_sequence=["#f39c12", "#e74c3c"],
                    title="Which boundary do provisional results span?",
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False, height=300, margin=dict(t=40))
                _plotly(fig)
                st.caption(
                    "A result spanning the Normal/Elevated boundary means the patient may or may not need treatment. "
                    "A result spanning Elevated/High-risk is more urgent - a retest is critical."
                )

        with col_action:
            if "actual_class" in df_prov_clin.columns:
                stage_prov = (
                    df_prov_clin[df_prov_clin["actual_class"].isin(CLASSES)]["actual_class"]
                    .value_counts()
                    .reset_index()
                )
                stage_prov.columns = ["True stage", "Provisional count"]
                if not stage_prov.empty:
                    fig = px.bar(
                        stage_prov,
                        x="True stage", y="Provisional count", text="Provisional count",
                        color="True stage",
                        color_discrete_map={"A1": "#2ecc71", "A2": "#f39c12", "A3": "#e74c3c"},
                        title="Which patient groups most often got a provisional result?",
                    )
                    fig.update_traces(textposition="outside")
                    fig.update_layout(showlegend=False, height=300, margin=dict(t=40))
                    _plotly(fig)
                    st.caption(
                        "If A3 (high-risk) patients frequently get provisional results, "
                        "it means the model struggles most where accuracy matters most - "
                        "a strong signal to improve image capture for these patients."
                    )

    st.markdown("---")

    # -- 6. Batch overview ---------------------------------------------------
    st.markdown("## About this batch")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if "actual_class" in df_ok.columns:
            counts = df_ok[df_ok["actual_class"].isin(CLASSES)]["actual_class"].value_counts().reset_index()
            counts.columns = ["Stage", "Count"]
            fig = px.pie(
                counts, names="Stage", values="Count",
                color="Stage",
                color_discrete_map={"A1": "#2ecc71", "A2": "#f39c12", "A3": "#e74c3c"},
                title="Patient stages (lab confirmed)",
            )
            fig.update_layout(height=260, margin=dict(t=40, b=0))
            _plotly(fig)

    with col_b:
        if "predicted_uacr_type" in df_ok.columns:
            counts = df_ok["predicted_uacr_type"].value_counts().reset_index()
            counts.columns = ["Type", "Count"]
            fig = px.pie(
                counts, names="Type", values="Count",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title="Model result type",
            )
            fig.update_layout(height=260, margin=dict(t=40, b=0))
            _plotly(fig)
            st.caption("'range' = model was uncertain and reported a range instead of one number.")

    with col_c:
        if "uacr_confidence_bucket" in df_ok.columns:
            counts = df_ok["uacr_confidence_bucket"].value_counts().reset_index()
            counts.columns = ["Confidence", "Count"]
            fig = px.pie(
                counts, names="Confidence", values="Count",
                color="Confidence",
                color_discrete_map={"Low": "#e74c3c", "Moderate": "#f39c12", "High": "#2ecc71"},
                title="Model confidence",
            )
            fig.update_layout(height=260, margin=dict(t=40, b=0))
            _plotly(fig)
            st.caption("A large Low slice suggests image quality issues across the batch.")


# ===========================================================================
# TECHNICAL TAB
# ===========================================================================

if selected_view == "For Data Teams":
    data_team_section = st.radio(
        "Data team section",
        ["Model Performance", "Pod & Bin", "Quality & Reliability"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if data_team_section == "Model Performance":
        render_data_team_model_performance()
    elif data_team_section == "Pod & Bin":
        render_data_team_pod_bin()
    else:
        render_data_team_quality_reliability()
    st.stop()

