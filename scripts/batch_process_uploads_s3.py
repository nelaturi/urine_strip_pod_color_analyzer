"""
Batch processor for strip images sourced from S3 URLs.

Mirrors ``batch_process_uploads.py`` exactly, except the image for each row is
downloaded from an HTTPS S3 URL given in the Excel sheet instead of being
matched from a local ``images/`` subfolder.

Expected Excel layout (single header row):
    Row 1 headers: <id_col>, Patient Name, Input Image,
                   Urine Albumin(mg/L), Urine Creatinine (mg/dl),
                   UACR Value (mg/g), Stage
    Row 2+: data rows.  "Input Image" must contain an HTTPS S3 URL, e.g.
        https://hellokidneydata.s3.ap-south-1.amazonaws.com/uacr/input/<hash>.png

Rows are skipped (with a warning) when:
  - Lab values (albumin, creatinine, UACR) or stage are missing.
  - The Input Image cell is empty.
  - The S3 download fails (recorded as a failed row, not skipped).

Usage:
    python scripts/batch_process_uploads_s3.py <main_folder> [--output-root <path>] [--workers N] [--checkpoint N]

``main_folder`` must contain exactly one Excel file.  No ``images/`` subfolder
is required; downloaded images are cached under ``<run_dir>/_downloads/``.
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

import openpyxl
from openpyxl.styles import Font

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.routes import DEVICE, MODEL_PATH, model  # noqa: E402
from app.utils import (  # noqa: E402
    CREATININE_CENTROIDS,
    MICROALBUMIN_CENTROIDS,
    process_image_and_get_pods,
)
from scripts.run_metadata import build_run_id, sanitize_label, write_run_metadata  # noqa: E402


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
UPLOAD_DIR = REPO_ROOT / "app" / "static" / "uploads"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "batch_runs"
DOWNLOAD_TIMEOUT_SEC = 30


MODEL_VERSION = Path(MODEL_PATH).stem

# ---------------------------------------------------------------------------
# Bin structures
# ---------------------------------------------------------------------------

_ALBUMIN_BINS: list = sorted(MICROALBUMIN_CENTROIDS.keys())
_CREATININE_BIN_NUMERIC: dict = {float(k.split()[0]): k for k in CREATININE_CENTROIDS.keys()}
_CREATININE_BINS: list = [_CREATININE_BIN_NUMERIC[v] for v in sorted(_CREATININE_BIN_NUMERIC.keys())]

# ---------------------------------------------------------------------------
# Excel column name mapping  (header text → internal key)
# Column index 0 is the patient ID by position regardless of header name.
# ---------------------------------------------------------------------------
_EXCEL_HEADER_MAP = {
    "patient name":             "patient_name",
    "input image":              "s3_url",
    "urine albumin(mg/l)":      "actual_microalbumin",
    "urine creatinine (mg/dl)": "actual_creatinine",
    "uacr value (mg/g)":        "actual_uacr_value",
    "stage":                    "uacr_stage",
}

OUTPUT_COLS = [
    "patient_id",
    "patient_name",
    "run_id",
    "timestamp",
    "model_version",
    "input_image_path",
    "input_image_url",
    "output_image_path",
    "result_json_path",

    "actual_class",
    "predicted_class",
    "is_provisional",
    "is_high_watch",
    "is_unconfirmed",
    "is_legacy_recovered",
    "provisional_class",
    "provisional_guard_action",

    "actual_uacr",
    "predicted_uacr_type",
    "predicted_uacr",
    "predicted_uacr_provisional_range_low",
    "predicted_uacr_provisional_range_high",
    "actual_uacr_in_provisional_range",
    "actual_vs_pred_uacr_delta",
    "uacr_delta_legacy_vs_corrected",
    "uacr_warning",

    "actual_albumin",
    "predicted_albumin_type",
    "predicted_albumin",
    "predicted_albumin_provisional_range_low",
    "predicted_albumin_provisional_range_high",
    "actual_albumin_in_provisional_range",
    "actual_vs_pred_albumin_delta",

    "actual_creatinine",
    "predicted_creatinine_type",
    "predicted_creatinine",
    "actual_vs_pred_creatinine_delta",

    "uacr_confidence",
    "albumin_confidence",
    "creatinine_confidence",
    "uacr_confidence_bucket",
    "albumin_confidence_bucket",
    "creatinine_confidence_bucket",

    "predicted_albumin_bin",
    "albumin_raw_chart_label",
    "expected_albumin_bin",
    "albumin_bin_error",
    "albumin_within_1_bin",
    "predicted_creatinine_bin",
    "expected_creatinine_bin",
    "creatinine_bin_error",
    "creatinine_within_1_bin",

    # Microalbumin shade-guard diagnostics
    "guard_action",
    "guard_reason",
    "guard_report_mode",
    "guard_confidence_bucket",
    "guard_current_confidence_used",
    "guard_corrected_albumin_value",
    "guard_value_zone",
    # LAB pod color — diagnoses upstream capture failures (Batch-3-style)
    "guard_median_L",
    "guard_median_a",
    "guard_median_b",
    "guard_median_chroma",
    # Chart-distance evidence
    "guard_nearest_chart_bin",
    "guard_nearest_chart_de",
    "guard_second_nearest_chart_de",
    "guard_nearest_vs_second_margin",
    # Low-shade evidence
    "guard_low_class_mg_l",
    "guard_low_ambiguous",
    "guard_low_pixel_fraction",
    "guard_median_low_de",
    "guard_low_margin",
    "guard_low_shade_confirmed",
    "guard_low_shade_confirmed_relaxed",
    # Per-bin low-shade diagnostics (3/10/30 mg/L) — inputs to the low-bin selector
    "guard_median_de_3",
    "guard_median_de_10",
    "guard_median_de_30",
    "guard_low_pixel_fraction_3",
    "guard_low_pixel_fraction_10",
    "guard_low_pixel_fraction_30",
    # Low-bin selector outcome (commit 8955def): prevents low-compatible cases
    # from snapping to 30 mg/L when 3 or 10 mg/L is the better choice.
    "guard_selected_low_bin",
    "guard_low_bin_margin",
    "guard_low_bin_selection_reason",
    "guard_low_30_clear_evidence",
    "guard_low_30_uacr_boundary_risk",
    "guard_uacr_if_low_3",
    "guard_uacr_if_low_10",
    "guard_uacr_if_low_30",
    # Aqua / strong-aqua evidence
    "guard_aqua_candidate_class_mg_l",
    "guard_strong_aqua_bin",
    "guard_strong_aqua_de",
    "guard_strong_aqua_confirmed",
    "guard_weak_aqua_present",
    # Aqua evidence tiers (commit 3031974)
    "guard_weak_aqua_low_compatible",
    "guard_very_low_moderate_aqua",
    "guard_moderate_aqua_present",
    "guard_aqua_evidence_tier",
    "guard_moderate_aqua_threshold_used",
    "guard_aqua_pixel_fraction",
    "guard_median_aqua_de",
    "guard_aqua_margin",
    # High-value verification
    "guard_high_value_color_verified",
    "guard_strong_aqua_confirmed_for_high",
    # Pre-guard zone flags
    "guard_current_is_high_watch",
    "guard_current_is_very_high",
    # OOD / overbright signals
    "guard_overbright_not_chart_like",
    "guard_overbright_ood_no_chart_support",
    # Threshold snapshot per row
    "guard_weak_aqua_threshold_used",
    "guard_strong_aqua_threshold_used",
    "guard_strong_aqua_de_max_used",

    # Legacy V2/V3 recovery (post V4-unconfirmed liberal fallback)
    "legacy_recovery_attempted",
    "legacy_recovery_enabled",
    "legacy_recovery_accepted",
    "legacy_recovery_mode",
    "legacy_recovery_conflict_status",
    "legacy_recovery_v2_bin",
    "legacy_recovery_v3_bin",
    "legacy_recovery_recovered_albumin_bin",
    "legacy_recovered_uacr_value",
    "legacy_recovered_uacr_source",
    "albumin_before_legacy_recovery",
    "albumin_after_legacy_recovery",
    "albumin_report_mode_before_legacy_recovery",
    "albumin_report_mode_after_legacy_recovery",
    "uacr_report_mode_before_legacy_recovery",
    "uacr_report_mode_after_legacy_recovery",
    "legacy_recovery_warning",

    # SI units (commit 3031974)
    "creatinine_umol_l",
    "creatinine_mmol_l",
    "creatinine_si_display",
    "microalbumin_g_l",
    "microalbumin_si_display",
    "acr_mg_mmol",
    "acr_si_display",
    "acr_si_stage_code",
    "acr_si_reference_range",
    "acr_si_range_low_mg_mmol",
    "acr_si_range_high_mg_mmol",
    "acr_si_range_display",
    "acr_si_range_stage_code",

    "albumin_flag_glare",
    "albumin_flag_non_uniform",
    "albumin_flag_mask_quality",
    "creatinine_flag_glare",
    "creatinine_flag_non_uniform",
    "creatinine_flag_mask_quality",

    "retest_required",
    "retest_reason",

    "inference_time_sec",
    "workers_used",
    "status",
    "error",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch process strip images from S3 URLs listed in a master Excel file. "
            "Writes a combined results Excel and per-image JSON files under a "
            "timestamped experiment directory."
        )
    )
    parser.add_argument(
        "main_folder",
        help="Folder containing the master Excel file. The Excel must have an 'Input Image' column with HTTPS S3 URLs.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Base folder where timestamped experiment results will be written.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(max(1, (os.cpu_count() or 4) - 1), 4),
        help="Number of parallel worker processes (default: min(cpu_count-1, 4)).",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=50,
        help="Save an intermediate results.xlsx every N completed images (default: 50). 0 disables.",
    )
    parser.add_argument(
        "--code-label",
        default="post-update",
        help=(
            "Short tag for the code version used in this run (default: 'post-update'). "
            "Embedded in the run folder name and recorded in run_metadata.json."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Excel I/O
# ---------------------------------------------------------------------------

def find_master_excel(main_folder: Path) -> Path:
    candidates = (
        list(main_folder.glob("*.xlsx"))
        + list(main_folder.glob("*.xlsm"))
        + list(main_folder.glob("*.xls"))
    )
    if not candidates:
        raise FileNotFoundError(f"No Excel file found in {main_folder}")
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ValueError(
            f"Multiple Excel files found ({names}). Place exactly one master Excel in the folder."
        )
    return candidates[0]


def read_master_excel(excel_path: Path) -> list[dict]:
    """
    Auto-detect 1-row vs 2-row header layout:
      - If row 1 contains the recognized headers, treat it as a single header row.
      - Otherwise (e.g. row 1 is a title like "MindRay quantitative UACR"),
        use row 2 as the header and skip row 1.

    Column 0 is always the patient ID by position regardless of its header text.
    """
    wb = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
    ws = wb.active

    all_rows = list(ws.iter_rows(min_row=1, values_only=True))
    if len(all_rows) < 2:
        wb.close()
        raise ValueError("Excel must have at least 2 rows.")

    def _build_col_idx(header_row) -> dict[str, int]:
        idx_map: dict[str, int] = {"patient_id": 0}
        for i, h in enumerate(header_row):
            if h is None or i == 0:
                continue
            key = _EXCEL_HEADER_MAP.get(str(h).strip().lower())
            if key:
                idx_map[key] = i
        return idx_map

    col_idx = _build_col_idx(all_rows[0])
    data_start = 1
    if "s3_url" not in col_idx and len(all_rows) >= 3:
        # Probably a 2-row header (row 0 is a title). Try row 1.
        alt = _build_col_idx(all_rows[1])
        if "s3_url" in alt:
            col_idx = alt
            data_start = 2

    if "s3_url" not in col_idx:
        wb.close()
        raise ValueError("Excel header is missing required 'Input Image' column.")

    rows = []
    for raw_row in all_rows[data_start:]:
        if all(v is None for v in raw_row):
            continue
        record = {}
        for key, idx in col_idx.items():
            record[key] = raw_row[idx] if idx < len(raw_row) else None
        rows.append(record)

    wb.close()
    return rows


def write_output_excel(run_dir: Path, output_rows: list) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Results"

    for col_idx, header in enumerate(OUTPUT_COLS, start=1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        cell.font = Font(bold=True)

    for row_idx, row_data in enumerate(output_rows, start=2):
        for col_idx, col in enumerate(OUTPUT_COLS, start=1):
            ws.cell(row=row_idx, column=col_idx, value=row_data.get(col))

    excel_path = run_dir / "data" / "results.xlsx"
    tmp_path = excel_path.with_name("results_tmp.xlsx")
    wb.save(tmp_path)
    try:
        if excel_path.exists():
            excel_path.unlink()
        tmp_path.rename(excel_path)
    except PermissionError:
        print(f"  [WARN] Could not replace results.xlsx (file is open). Saved as: {tmp_path.name}")
        return tmp_path
    return excel_path


# ---------------------------------------------------------------------------
# S3 download
# ---------------------------------------------------------------------------

def _filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = unquote(Path(parsed.path).name) or "image"
    return name


def download_s3_image(url: str, dest_dir: Path) -> Path:
    """
    Download an HTTPS URL to dest_dir and return the local file path.
    Filename is derived from the URL; a uuid prefix is added to avoid collisions
    when the same key is reused across rows.
    """
    name = _filename_from_url(url)
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension in URL: {suffix or '(none)'}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    local_path = dest_dir / f"{uuid.uuid4().hex[:8]}_{name}"

    req = Request(url, headers={"User-Agent": "batch-uploads/1.0"})
    with urlopen(req, timeout=DOWNLOAD_TIMEOUT_SEC) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} when fetching {url}")
        with open(local_path, "wb") as f:
            shutil.copyfileobj(resp, f)
    return local_path


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

_EXPECTED_RESULT_KEYS = {
    "composite_img",
    "uacr_report_mode",
    "uacr_value",
    "uacr_stage",
    "uacr_confidence_pct",
    "uacr_confidence_bucket",
    "uacr_retest_recommended",
    "uacr_guarded_range_mg_g",
    "uacr_guarded_albumin_range_mg_l",
    "uacr_delta_legacy_minus_corrected",
    "pod_quality",
    # Legacy V2/V3 liberal recovery (post V4-unconfirmed)
    "uacr_report_mode_before_legacy_recovery",
    "uacr_report_mode_after_legacy_recovery",
    "legacy_recovered_uacr_value",
    "legacy_recovered_uacr_source",
    # SI-unit conversions (commit 3031974)
    "creatinine_si",
    "microalbumin_si",
    "acr_si",
    "acr_si_range",
}

_EXPECTED_POD_KEYS = {
    "final_display_value",
    "confidence",
    "confidence_bucket",
    "flag_glare",
    "flag_non_uniform",
    "flag_mask_quality",
    "mask_quality_reasons",
    "raw_color_chart_label",
}

_EXPECTED_MICROALBUMIN_KEYS = _EXPECTED_POD_KEYS | {
    "shade_sanity_check",
    "guarded_uacr_scenario",
    "legacy_recovery_attempted",
    "legacy_recovery_enabled",
    "legacy_recovery_mode",
    "legacy_recovery_result",
    "final_display_value_before_legacy_recovery",
    "final_display_value_after_legacy_recovery",
    "microalbumin_report_mode_before_legacy_recovery",
    "microalbumin_report_mode_after_legacy_recovery",
    # SI-unit conversion (commit 3031974)
    "microalbumin_si",
}


def _validate_result_schema(result: dict, patient_id: str) -> list[str]:
    warnings = []
    missing_top = _EXPECTED_RESULT_KEYS - set(result.keys())
    if missing_top:
        warnings.append(f"[SCHEMA] Patient {patient_id}: result missing top-level keys: {sorted(missing_top)}")

    pod_quality = result.get("pod_quality", {})
    if not isinstance(pod_quality, dict):
        warnings.append(f"[SCHEMA] Patient {patient_id}: 'pod_quality' is not a dict — got {type(pod_quality)}")
        return warnings

    for pod_name, expected_keys in [
        ("creatinine", _EXPECTED_POD_KEYS),
        ("microalbumin", _EXPECTED_MICROALBUMIN_KEYS),
    ]:
        pod = pod_quality.get(pod_name, {})
        if not isinstance(pod, dict):
            warnings.append(f"[SCHEMA] Patient {patient_id}: pod_quality['{pod_name}'] is not a dict")
            continue
        missing_pod = expected_keys - set(pod.keys())
        if missing_pod:
            warnings.append(f"[SCHEMA] Patient {patient_id}: pod_quality['{pod_name}'] missing keys: {sorted(missing_pod)}")

    return warnings


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

_STAGE_CODE_RE = re.compile(r"\b(A[0-9](?:_A[0-9])?(?:_boundary)?)\b", re.IGNORECASE)


def _extract_stage_code(raw: str) -> str | None:
    if not raw:
        return None
    raw = raw.strip()
    slash = re.match(r"(A\d)/(A\d)", raw, re.IGNORECASE)
    if slash:
        return f"{slash.group(1).upper()}_{slash.group(2).upper()}"
    m = _STAGE_CODE_RE.search(raw)
    if m:
        return m.group(1).upper()
    if "unconfirmed" in raw.lower():
        return "unconfirmed"
    return raw


def _safe_delta(actual, predicted) -> float | None:
    try:
        return round(float(actual) - float(predicted), 2)
    except (TypeError, ValueError):
        return None


def _snap_albumin_bin(lab_value):
    try:
        v = float(lab_value)
    except (TypeError, ValueError):
        return None
    return min(_ALBUMIN_BINS, key=lambda b: abs(b - v))


def _snap_creatinine_bin(lab_value):
    try:
        v = float(lab_value)
    except (TypeError, ValueError):
        return None
    nearest = min(_CREATININE_BIN_NUMERIC.keys(), key=lambda b: abs(b - v))
    return _CREATININE_BIN_NUMERIC[nearest]


def _bin_error(predicted, expected, ordered_bins) -> int | None:
    if predicted is None or expected is None:
        return None
    try:
        return ordered_bins.index(predicted) - ordered_bins.index(expected)
    except ValueError:
        return None


def _value_in_range(value, low_high) -> bool | None:
    """True if `value` lies within `low_high` (inclusive). None if either side missing."""
    if value is None or not low_high:
        return None
    try:
        v = float(value)
        lo = float(low_high[0])
        hi = float(low_high[1])
    except (TypeError, ValueError, IndexError):
        return None
    if lo > hi:
        lo, hi = hi, lo
    return lo <= v <= hi


def _extract_guard_fields(microalbumin_q: dict) -> dict:
    """Pull the shade-guard diagnostics into flat columns for Excel.

    Mirrors the field set in `microalbumin_shade_sanity_check._response`
    in app/utils.py. Adding/removing fields here must stay in sync with `OUTPUT_COLS`.
    """
    g = microalbumin_q.get("shade_sanity_check", {}) or {}
    return {
        "guard_action": g.get("action"),
        "guard_reason": g.get("guard_reason"),
        "guard_report_mode": g.get("report_mode"),
        "guard_confidence_bucket": g.get("confidence_bucket"),
        "guard_current_confidence_used": g.get("current_confidence_used"),
        "guard_corrected_albumin_value": g.get("corrected_albumin_value"),
        "guard_value_zone": g.get("value_zone"),
        # LAB pod color — diagnose upstream capture failures
        "guard_median_L": g.get("median_L"),
        "guard_median_a": g.get("median_a"),
        "guard_median_b": g.get("median_b"),
        "guard_median_chroma": g.get("median_chroma"),
        # Chart-distance evidence
        "guard_nearest_chart_bin": g.get("nearest_chart_bin"),
        "guard_nearest_chart_de": g.get("nearest_chart_de"),
        "guard_second_nearest_chart_de": g.get("second_nearest_chart_de"),
        "guard_nearest_vs_second_margin": g.get("nearest_vs_second_margin"),
        # Low-shade evidence
        "guard_low_class_mg_l": g.get("low_candidate_class_mg_l"),
        "guard_low_ambiguous": g.get("low_ambiguous"),
        "guard_low_pixel_fraction": g.get("low_pixel_fraction"),
        "guard_median_low_de": g.get("median_low_de"),
        "guard_low_margin": g.get("low_margin"),
        "guard_low_shade_confirmed": g.get("low_shade_confirmed"),
        "guard_low_shade_confirmed_relaxed": g.get("low_shade_confirmed_relaxed"),
        # Per-bin low-shade diagnostics (commit 8955def). Pixel-fraction dict
        # uses string keys ("3"/"10"/"30") per microalbumin_shade_sanity_check._response.
        "guard_median_de_3": g.get("median_de_3"),
        "guard_median_de_10": g.get("median_de_10"),
        "guard_median_de_30": g.get("median_de_30"),
        "guard_low_pixel_fraction_3": (g.get("low_pixel_fraction_by_class") or {}).get("3"),
        "guard_low_pixel_fraction_10": (g.get("low_pixel_fraction_by_class") or {}).get("10"),
        "guard_low_pixel_fraction_30": (g.get("low_pixel_fraction_by_class") or {}).get("30"),
        # Low-bin selector outcome (commit 8955def)
        "guard_selected_low_bin": g.get("selected_low_bin"),
        "guard_low_bin_margin": g.get("low_bin_margin"),
        "guard_low_bin_selection_reason": g.get("low_bin_selection_reason"),
        "guard_low_30_clear_evidence": g.get("low_30_clear_evidence"),
        "guard_low_30_uacr_boundary_risk": g.get("low_30_uacr_boundary_risk"),
        "guard_uacr_if_low_3": g.get("uacr_if_low_3"),
        "guard_uacr_if_low_10": g.get("uacr_if_low_10"),
        "guard_uacr_if_low_30": g.get("uacr_if_low_30"),
        # Aqua / strong-aqua evidence
        "guard_aqua_candidate_class_mg_l": g.get("aqua_candidate_class_mg_l"),
        "guard_strong_aqua_bin": g.get("strong_aqua_candidate_bin"),
        "guard_strong_aqua_de": g.get("strong_aqua_candidate_de"),
        "guard_strong_aqua_confirmed": g.get("strong_aqua_confirmed"),
        "guard_weak_aqua_present": g.get("weak_aqua_present"),
        # Aqua evidence tiers (commit 3031974)
        "guard_weak_aqua_low_compatible": g.get("weak_aqua_low_compatible"),
        "guard_very_low_moderate_aqua": g.get("very_low_moderate_aqua"),
        "guard_moderate_aqua_present": g.get("moderate_aqua_present"),
        "guard_aqua_evidence_tier": g.get("aqua_evidence_tier"),
        "guard_moderate_aqua_threshold_used": g.get("moderate_aqua_threshold_used"),
        "guard_aqua_pixel_fraction": g.get("aqua_pixel_fraction"),
        "guard_median_aqua_de": g.get("median_aqua_de"),
        "guard_aqua_margin": g.get("aqua_margin"),
        # High-value verification
        "guard_high_value_color_verified": g.get("high_value_color_verified"),
        "guard_strong_aqua_confirmed_for_high": g.get("strong_aqua_confirmed_for_high"),
        # Pre-guard zone flags
        "guard_current_is_high_watch": g.get("current_is_high_watch"),
        "guard_current_is_very_high": g.get("current_is_very_high"),
        # OOD / overbright signals
        "guard_overbright_not_chart_like": g.get("overbright_not_chart_like"),
        "guard_overbright_ood_no_chart_support": g.get("overbright_ood_no_chart_support"),
        # Threshold snapshot per row
        "guard_weak_aqua_threshold_used": g.get("weak_aqua_threshold_used"),
        "guard_strong_aqua_threshold_used": g.get("strong_aqua_threshold_used"),
        "guard_strong_aqua_de_max_used": g.get("strong_aqua_de_max_used"),
    }


def _extract_legacy_recovery_fields(microalbumin_q: dict, result: dict) -> dict:
    """Flatten V4-unconfirmed → V2/V3 liberal-recovery diagnostics into row columns."""
    lr = microalbumin_q.get("legacy_recovery_result") or {}
    return {
        "legacy_recovery_attempted": microalbumin_q.get("legacy_recovery_attempted"),
        "legacy_recovery_enabled": microalbumin_q.get("legacy_recovery_enabled"),
        "legacy_recovery_accepted": lr.get("accepted"),
        "legacy_recovery_mode": microalbumin_q.get("legacy_recovery_mode"),
        "legacy_recovery_conflict_status": lr.get("conflict_status"),
        "legacy_recovery_v2_bin": lr.get("v2_bin"),
        "legacy_recovery_v3_bin": lr.get("v3_bin"),
        "legacy_recovery_recovered_albumin_bin": lr.get("recovered_albumin_bin"),
        "legacy_recovered_uacr_value": result.get("legacy_recovered_uacr_value"),
        "legacy_recovered_uacr_source": result.get("legacy_recovered_uacr_source"),
        "albumin_before_legacy_recovery": microalbumin_q.get("final_display_value_before_legacy_recovery"),
        "albumin_after_legacy_recovery": microalbumin_q.get("final_display_value_after_legacy_recovery"),
        "albumin_report_mode_before_legacy_recovery": microalbumin_q.get("microalbumin_report_mode_before_legacy_recovery"),
        "albumin_report_mode_after_legacy_recovery": microalbumin_q.get("microalbumin_report_mode_after_legacy_recovery"),
        "uacr_report_mode_before_legacy_recovery": result.get("uacr_report_mode_before_legacy_recovery"),
        "uacr_report_mode_after_legacy_recovery": result.get("uacr_report_mode_after_legacy_recovery"),
        "legacy_recovery_warning": result.get("legacy_recovery_warning"),
    }


def _extract_si_fields(creatinine_q: dict, microalbumin_q: dict, result: dict) -> dict:
    """Flatten SI-unit conversions (commit 3031974) into row columns."""
    cr_si = creatinine_q.get("creatinine_si") or result.get("creatinine_si") or {}
    mi_si = microalbumin_q.get("microalbumin_si") or result.get("microalbumin_si") or {}
    acr_si = result.get("acr_si") or {}
    acr_si_range = result.get("acr_si_range") or {}
    acr_range_pair = acr_si_range.get("acr_si_range_mg_mmol")
    return {
        "creatinine_umol_l": cr_si.get("creatinine_umol_l"),
        "creatinine_mmol_l": cr_si.get("creatinine_mmol_l"),
        "creatinine_si_display": cr_si.get("creatinine_si_display"),
        "microalbumin_g_l": mi_si.get("microalbumin_g_l"),
        "microalbumin_si_display": mi_si.get("microalbumin_si_display"),
        "acr_mg_mmol": acr_si.get("acr_mg_mmol"),
        "acr_si_display": acr_si.get("acr_si_display"),
        "acr_si_stage_code": acr_si.get("acr_si_stage_code"),
        "acr_si_reference_range": acr_si.get("acr_si_reference_range"),
        "acr_si_range_low_mg_mmol": acr_range_pair[0] if acr_range_pair else None,
        "acr_si_range_high_mg_mmol": acr_range_pair[1] if acr_range_pair else None,
        "acr_si_range_display": acr_si_range.get("acr_si_range_display"),
        "acr_si_range_stage_code": acr_si_range.get("acr_si_stage_code"),
    }


def _build_retest_reason(creatinine_q: dict, microalbumin_q: dict, report_mode: str) -> str | None:
    reasons = []
    for mask_reason in creatinine_q.get("mask_quality_reasons", []):
        reasons.append(f"creatinine: {mask_reason}")
    shade_guard = microalbumin_q.get("shade_sanity_check", {})
    guard_reason = shade_guard.get("guard_reason")
    if guard_reason and report_mode in ("provisional_range", "unconfirmed", "high_watch"):
        reasons.append(f"albumin: {guard_reason}")
    if report_mode == "legacy_recovered":
        reasons.append(
            "albumin: V4 unconfirmed; legacy V2/V3 liberal recovery selected the higher UACR — interpret as legacy-recovered, not V4-confirmed."
        )
    return "; ".join(reasons) if reasons else None


def build_output_row(
    result: dict,
    excel_row: dict,
    run_id: str,
    output_image_path: str | None = None,
    result_json_path: str | None = None,
    inference_time_sec: float | None = None,
    workers_used: int | None = None,
    status: str = "success",
    error: str | None = None,
) -> dict:
    pod_quality = result.get("pod_quality", {})
    creatinine_q = pod_quality.get("creatinine", {})
    microalbumin_q = pod_quality.get("microalbumin", {})

    report_mode = result.get("uacr_report_mode", "exact")
    is_provisional = report_mode == "provisional_range"
    is_unconfirmed = report_mode == "unconfirmed"
    is_high_watch = report_mode == "high_watch"
    is_legacy_recovered = report_mode == "legacy_recovered"
    has_exact_value = report_mode in ("exact", "high_watch")
    has_numeric_value = has_exact_value or is_legacy_recovered

    uacr_value = result.get("uacr_value")
    uacr_range = result.get("uacr_guarded_range_mg_g")
    albumin_value = microalbumin_q.get("final_display_value")
    albumin_range = result.get("uacr_guarded_albumin_range_mg_l")
    creatinine_value = creatinine_q.get("final_display_value")

    if is_provisional:
        pred_uacr_type = "range"
        pred_albumin_type = "range"
    elif is_unconfirmed:
        pred_uacr_type = "unconfirmed"
        pred_albumin_type = "unconfirmed"
    elif is_high_watch:
        pred_uacr_type = "high_watch"
        pred_albumin_type = "high_watch"
    elif is_legacy_recovered:
        pred_uacr_type = "legacy_recovered"
        pred_albumin_type = "legacy_recovered"
    else:
        pred_uacr_type = "exact"
        pred_albumin_type = "exact"

    actual_uacr = excel_row.get("actual_uacr_value")
    actual_albumin = excel_row.get("actual_microalbumin")
    actual_creatinine = excel_row.get("actual_creatinine")
    actual_class = _extract_stage_code(str(excel_row.get("uacr_stage") or ""))

    predicted_class = _extract_stage_code(result.get("uacr_stage") or "")
    guarded_scenario = microalbumin_q.get("guarded_uacr_scenario", {})
    provisional_class = (
        guarded_scenario.get("provisional_uacr_stage_code") if is_provisional else None
    )

    actual_vs_pred_uacr = _safe_delta(actual_uacr, uacr_value) if has_numeric_value else None
    actual_vs_pred_albumin = _safe_delta(actual_albumin, albumin_value) if has_numeric_value else None
    actual_vs_pred_creatinine = _safe_delta(actual_creatinine, creatinine_value)

    actual_uacr_in_range = _value_in_range(actual_uacr, uacr_range) if is_provisional else None
    actual_albumin_in_range = _value_in_range(actual_albumin, albumin_range) if is_provisional else None

    uacr_delta_internal = result.get("uacr_delta_legacy_minus_corrected")

    uacr_conf_pct = result.get("uacr_confidence_pct")
    uacr_confidence = round(uacr_conf_pct / 100.0, 4) if uacr_conf_pct is not None else None

    raw_alb_chart_label = microalbumin_q.get("raw_color_chart_label")
    pred_alb_bin = _snap_albumin_bin(albumin_value) if has_numeric_value and albumin_value is not None else None
    exp_alb_bin = _snap_albumin_bin(actual_albumin) if (has_numeric_value and actual_albumin is not None) else None
    alb_bin_err = _bin_error(pred_alb_bin, exp_alb_bin, _ALBUMIN_BINS)

    pred_cre_bin = creatinine_q.get("raw_color_chart_label")
    exp_cre_bin = _snap_creatinine_bin(actual_creatinine) if actual_creatinine is not None else None
    cre_bin_err = _bin_error(pred_cre_bin, exp_cre_bin, _CREATININE_BINS)

    guard_fields = _extract_guard_fields(microalbumin_q)
    legacy_recovery_fields = _extract_legacy_recovery_fields(microalbumin_q, result)
    si_fields = _extract_si_fields(creatinine_q, microalbumin_q, result)

    return {
        "patient_id": excel_row.get("patient_id"),
        "patient_name": excel_row.get("patient_name"),
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_version": MODEL_VERSION,
        "input_image_path": excel_row.get("input_image_path"),
        "input_image_url": excel_row.get("s3_url"),
        "output_image_path": output_image_path,
        "result_json_path": result_json_path,

        "actual_class": actual_class,
        "predicted_class": predicted_class,
        "is_provisional": is_provisional,
        "is_high_watch": is_high_watch,
        "is_unconfirmed": is_unconfirmed,
        "is_legacy_recovered": is_legacy_recovered,
        "provisional_class": provisional_class,
        "provisional_guard_action": (
            microalbumin_q.get("shade_sanity_check", {}).get("action") if is_provisional else None
        ),

        "actual_uacr": actual_uacr,
        "predicted_uacr_type": pred_uacr_type,
        "predicted_uacr": uacr_value if has_numeric_value else None,
        "predicted_uacr_provisional_range_low": uacr_range[0] if uacr_range else None,
        "predicted_uacr_provisional_range_high": uacr_range[1] if uacr_range else None,
        "actual_uacr_in_provisional_range": actual_uacr_in_range,
        "actual_vs_pred_uacr_delta": actual_vs_pred_uacr,
        "uacr_delta_legacy_vs_corrected": uacr_delta_internal,
        "uacr_warning": result.get("uacr_warning"),

        "actual_albumin": actual_albumin,
        "predicted_albumin_type": pred_albumin_type,
        "predicted_albumin": albumin_value if has_numeric_value else None,
        "predicted_albumin_provisional_range_low": albumin_range[0] if albumin_range else None,
        "predicted_albumin_provisional_range_high": albumin_range[1] if albumin_range else None,
        "actual_albumin_in_provisional_range": actual_albumin_in_range,
        "actual_vs_pred_albumin_delta": actual_vs_pred_albumin,

        "actual_creatinine": actual_creatinine,
        "predicted_creatinine_type": "exact",
        "predicted_creatinine": creatinine_value,
        "actual_vs_pred_creatinine_delta": actual_vs_pred_creatinine,

        "uacr_confidence": uacr_confidence,
        "albumin_confidence": microalbumin_q.get("confidence"),
        "creatinine_confidence": creatinine_q.get("confidence"),
        "uacr_confidence_bucket": result.get("uacr_confidence_bucket"),
        "albumin_confidence_bucket": microalbumin_q.get("confidence_bucket"),
        "creatinine_confidence_bucket": creatinine_q.get("confidence_bucket"),

        "albumin_flag_glare": microalbumin_q.get("flag_glare"),
        "albumin_flag_non_uniform": microalbumin_q.get("flag_non_uniform"),
        "albumin_flag_mask_quality": microalbumin_q.get("flag_mask_quality"),
        "creatinine_flag_glare": creatinine_q.get("flag_glare"),
        "creatinine_flag_non_uniform": creatinine_q.get("flag_non_uniform"),
        "creatinine_flag_mask_quality": creatinine_q.get("flag_mask_quality"),

        "predicted_albumin_bin": pred_alb_bin,
        "albumin_raw_chart_label": raw_alb_chart_label,
        "expected_albumin_bin": exp_alb_bin,
        "albumin_bin_error": alb_bin_err,
        "albumin_within_1_bin": (abs(alb_bin_err) <= 1) if alb_bin_err is not None else None,
        "predicted_creatinine_bin": pred_cre_bin,
        "expected_creatinine_bin": exp_cre_bin,
        "creatinine_bin_error": cre_bin_err,
        "creatinine_within_1_bin": (abs(cre_bin_err) <= 1) if cre_bin_err is not None else None,

        **guard_fields,

        **legacy_recovery_fields,

        **si_fields,

        "retest_required": bool(result.get("uacr_retest_recommended", False)),
        "retest_reason": _build_retest_reason(creatinine_q, microalbumin_q, report_mode),

        "inference_time_sec": inference_time_sec,
        "workers_used": workers_used,
        "status": status,
        "error": error,
    }


def empty_output_row(excel_row: dict, run_id: str, error: str, workers_used: int | None = None) -> dict:
    row = {col: None for col in OUTPUT_COLS}
    row["patient_id"] = excel_row.get("patient_id")
    row["patient_name"] = excel_row.get("patient_name")
    row["run_id"] = run_id
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    row["model_version"] = MODEL_VERSION
    row["input_image_path"] = excel_row.get("input_image_path")
    row["input_image_url"] = excel_row.get("s3_url")
    row["actual_class"] = _extract_stage_code(str(excel_row.get("uacr_stage") or ""))
    row["actual_uacr"] = excel_row.get("actual_uacr_value")
    row["actual_albumin"] = excel_row.get("actual_microalbumin")
    row["actual_creatinine"] = excel_row.get("actual_creatinine")
    row["inference_time_sec"] = None
    row["workers_used"] = workers_used
    row["status"] = "failed"
    row["error"] = error
    return row


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def process_one_image(image_path: Path, run_dir: Path):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    temp_name = f"temp_{uuid.uuid4().hex}_{image_path.name}"
    temp_upload_path = UPLOAD_DIR / temp_name
    shutil.copy2(image_path, temp_upload_path)

    pipeline_output_path = None
    try:
        result = process_image_and_get_pods(str(temp_upload_path), model, DEVICE)
        if not result or not result.get("composite_img"):
            raise RuntimeError("Pipeline completed without returning a composite image.")

        pipeline_output_path = UPLOAD_DIR / result["composite_img"]
        if not pipeline_output_path.exists():
            raise FileNotFoundError(
                f"Pipeline reported '{result['composite_img']}' but the file was not found."
            )

        final_image_name = f"{image_path.stem}__{pipeline_output_path.name}"
        final_image_path = run_dir / "images" / final_image_name
        shutil.copy2(pipeline_output_path, final_image_path)
        return result, final_image_path
    finally:
        if temp_upload_path.exists():
            temp_upload_path.unlink()
        if pipeline_output_path is not None and pipeline_output_path.exists():
            pipeline_output_path.unlink()


# ---------------------------------------------------------------------------
# Per-row worker
# ---------------------------------------------------------------------------

def _process_row(
    index: int,
    excel_row: dict,
    run_id: str,
    run_dir: Path,
    data_dir: Path,
    download_dir: Path,
    workers_used: int = 1,
) -> tuple[dict, str, str]:
    url = excel_row.get("s3_url")
    downloaded_path: Path | None = None
    try:
        if not url or not str(url).strip():
            raise ValueError("Empty 'Input Image' URL.")
        url = str(url).strip()
        if not url.lower().startswith(("http://", "https://")):
            raise ValueError(f"Unsupported URL scheme (expected http/https): {url}")

        downloaded_path = download_s3_image(url, download_dir)
        excel_row["input_image_path"] = str(downloaded_path)

        if downloaded_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {downloaded_path.suffix}")

        t0 = time.perf_counter()
        result, final_image_path = process_one_image(downloaded_path, run_dir)
        inference_time = round(time.perf_counter() - t0, 3)

        for warn in _validate_result_schema(result, str(excel_row.get("patient_id", index))):
            print(warn)

        json_path = data_dir / f"{downloaded_path.stem}__result.json"
        row = build_output_row(
            result=result,
            excel_row=excel_row,
            run_id=run_id,
            output_image_path=str(final_image_path),
            result_json_path=str(json_path),
            inference_time_sec=inference_time,
            workers_used=workers_used,
            status="success",
        )
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)

        return row, "OK", final_image_path.name

    except Exception as exc:
        row = empty_output_row(excel_row, run_id, str(exc), workers_used=workers_used)
        return row, "FAIL", str(exc)
    finally:
        if downloaded_path is not None and downloaded_path.exists():
            try:
                downloaded_path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    input_path = Path(args.main_folder).resolve()
    output_root = Path(args.output_root).resolve()
    workers = max(1, args.workers)
    checkpoint_every = args.checkpoint
    code_label = sanitize_label(args.code_label)

    if model is None:
        raise RuntimeError("Segmentation model failed to load. Batch processing cannot continue.")
    if not input_path.exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in {".xlsx", ".xlsm", ".xls"}:
            raise ValueError(f"Expected an Excel file (.xlsx/.xlsm/.xls), got: {input_path.suffix}")
        excel_path = input_path
        main_folder = input_path.parent
    else:
        main_folder = input_path
        excel_path = find_master_excel(main_folder)
    print(f"Master Excel : {excel_path.name}")

    all_rows = read_master_excel(excel_path)
    if not all_rows:
        raise ValueError("Master Excel has no data rows.")
    print(f"Excel rows   : {len(all_rows)} total")

    REQUIRED_KEYS = ("actual_microalbumin", "actual_creatinine", "actual_uacr_value", "uacr_stage")
    skipped_no_data = 0
    skipped_no_url = 0
    runnable_rows: list[dict] = []

    for row in all_rows:
        pid = row.get("patient_id")
        if not pid:
            skipped_no_data += 1
            continue

        missing = [k for k in REQUIRED_KEYS if row.get(k) is None]
        if missing:
            print(f"  [SKIP] {pid}: missing {', '.join(missing)}")
            skipped_no_data += 1
            continue

        url = row.get("s3_url")
        if not url or not str(url).strip():
            print(f"  [SKIP] {pid}: empty Input Image URL")
            skipped_no_url += 1
            continue

        runnable_rows.append(row)

    total = len(runnable_rows)
    print(f"  Skipped (no lab data) : {skipped_no_data}")
    print(f"  Skipped (no URL)      : {skipped_no_url}")
    print(f"  Queued for processing : {total}")

    if total == 0:
        print("Nothing to process. Exiting.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = build_run_id(timestamp, code_label, REPO_ROOT)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "images").mkdir()
    (run_dir / "data").mkdir()
    (run_dir / "_downloads").mkdir()
    data_dir = run_dir / "data"
    download_dir = run_dir / "_downloads"

    meta = write_run_metadata(
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        code_label=code_label,
        script_used="batch_process_uploads_s3",
        run_id=run_id,
    )

    print(f"\nRun ID       : {run_id}")
    print(f"Code label   : {code_label}  (commit {meta.get('git_commit_short') or '-'}, "
          f"utils.py sha {(meta.get('utils_py_sha256') or '')[:10]})")
    print(f"Output folder: {run_dir}")
    print(f"Workers      : {workers}  (CPU cores detected: {os.cpu_count()})")
    print(f"Checkpoint   : every {checkpoint_every} images" if checkpoint_every else "Checkpoint   : disabled")
    print(f"Processing   : {total} rows\n")

    output_rows: list[dict | None] = [None] * total
    success_count = 0
    failure_count = 0
    done_count = 0

    def _checkpoint(done: int) -> None:
        completed = [r for r in output_rows if r is not None]
        write_output_excel(run_dir, completed)
        print(f"  [Checkpoint] {done}/{total} done — Excel saved ({len(completed)} rows)")

    futures = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, excel_row in enumerate(runnable_rows, start=1):
            future = executor.submit(
                _process_row, index, excel_row, run_id, run_dir, data_dir, download_dir, workers
            )
            futures[future] = (index, excel_row)

        for future in as_completed(futures):
            index, excel_row = futures[future]
            patient_id = excel_row.get("patient_id", f"row_{index}")
            row, status, msg = future.result()

            output_rows[index - 1] = row
            done_count += 1
            if status == "OK":
                success_count += 1
                print(f"  [{done_count}/{total}] OK   {patient_id} → {msg}")
            else:
                failure_count += 1
                print(f"  [{done_count}/{total}] FAIL {patient_id}: {msg}")

            if checkpoint_every and done_count % checkpoint_every == 0:
                _checkpoint(done_count)

    final_rows = [r for r in output_rows if r is not None]
    excel_out = write_output_excel(run_dir, final_rows)

    try:
        if download_dir.exists() and not any(download_dir.iterdir()):
            download_dir.rmdir()
    except OSError:
        pass

    print(f"\nCompleted. Success: {success_count}  Failed: {failure_count}")
    print(f"Results Excel: {excel_out}")
    print(f"Images       : {run_dir / 'images'}")
    print(f"Data         : {run_dir / 'data'}")


if __name__ == "__main__":
    main()
