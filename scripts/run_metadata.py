"""
Run-metadata helper for batch scripts.

Writes a `run_metadata.json` into each new batch run folder so we can later
tell which code version produced which results, and appends a one-line entry
to the top-level `outputs/batch_runs/INDEX.md`.

Captured fields:
  - git_commit, git_branch, git_dirty (any uncommitted changes?)
  - utils_py_sha256 (hash of app/utils.py — survives uncommitted edits)
  - code_label (human tag, e.g. "post-update")
  - script_used, started_at
  - threshold_snapshot — key MICRO_* constants read live from app.utils
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

# Constants we want a snapshot of for every run.  Read live from app.utils so
# any edit to those values is captured even if the file is not committed.
_THRESHOLD_NAMES = (
    "MICRO_STRONG_AQUA_DE_MAX",
    "MICRO_WEAK_AQUA_MIN",
    "MICRO_LOW_PIXEL_FRACTION_MIN",
    "MICRO_VERY_HIGH_GUARD_MIN",
    "MICRO_MIN_CONF_FOR_250_400",
    "MICRO_MIN_CONF_FOR_600",
    "MICRO_OVERBRIGHT_L_MAX",
    # Legacy V2/V3 liberal recovery — most behavior-changing toggle in this
    # commit; capturing it is the whole point of the run-metadata system.
    "ENABLE_MICROALBUMIN_UNCONFIRMED_LEGACY_RECOVERY",
    "MICRO_LEGACY_RECOVERY_MODE",
    "MICRO_LEGACY_RECOVERY_MIN_CHARTLIKE_DE",
    "MICRO_LEGACY_RECOVERY_MIN_MARGIN",
    "MICRO_LEGACY_RECOVERY_ACCEPT_OOD",
    "MICRO_LEGACY_RECOVERY_ACCEPT_LOW_CONFIDENCE",
    "MICRO_LEGACY_RECOVERY_CONFLICT_POLICY",
    "MICRO_LEGACY_RECOVERY_UACR_POLICY",
    "MICRO_LEGACY_RECOVERY_ALLOWED_BINS",
    # Aqua evidence tiers (commit 3031974). The weak/moderate split changes
    # which V4 branches fire for ambiguous-aqua pods.
    "MICRO_WEAK_AQUA_LOW_MIN",
    "MICRO_WEAK_AQUA_LOW_MAX",
    "MICRO_WEAK_AQUA_LOW_DE_MAX",
    "MICRO_WEAK_AQUA_LOW_LOW_SUPPORT_MIN",
    "MICRO_WEAK_AQUA_LOW_MARGIN_TOL",
    "MICRO_MODERATE_AQUA_MIN",
    "MICRO_MODERATE_AQUA_DE_MAX",
    "MICRO_MODERATE_AQUA_LOW_SUPPRESSION_MAX",
)

_LABEL_OK = re.compile(r"^[A-Za-z0-9._-]+$")


def sanitize_label(label: str) -> str:
    """Reject empty / whitespace / path-unsafe labels early."""
    label = (label or "").strip()
    if not label:
        raise ValueError("--code-label must not be empty")
    if not _LABEL_OK.match(label):
        raise ValueError(
            f"--code-label '{label}' contains unsafe characters; "
            "use letters/digits/._- only"
        )
    return label


def _git(args: list[str], repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _hash_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _threshold_snapshot() -> dict:
    snap: dict = {}
    try:
        from app import utils as _u  # noqa: WPS433  (deferred import on purpose)
    except Exception as exc:
        snap["_error"] = f"could not import app.utils: {exc}"
        return snap
    for name in _THRESHOLD_NAMES:
        snap[name] = getattr(_u, name, None)
    return snap


def commit_suffix(repo_root: Path) -> str:
    """Return short SHA of HEAD (e.g. '4971b77'), or 'nogit' if unavailable.

    We deliberately ignore the working-tree dirty flag here: dirty state is
    almost always driven by uncommitted batch-script tweaks, which are not
    what these runs are evaluating. The thing that matters — `app/utils.py` —
    is captured separately in `run_metadata.json` as `utils_py_sha256`, so
    edits to the core ML code are still detectable even on a 'dirty' tree.
    """
    commit = _git(["rev-parse", "HEAD"], repo_root)
    if not commit:
        return "nogit"
    return commit[:7]


def build_run_id(timestamp: str, code_label: str, repo_root: Path) -> str:
    """Compose: <timestamp>__<label>__<shortsha>[-dirty]."""
    return f"{timestamp}__{code_label}__{commit_suffix(repo_root)}"


def collect_metadata(
    repo_root: Path,
    code_label: str,
    script_used: str,
    run_id: str,
    extra: dict | None = None,
) -> dict:
    commit = _git(["rev-parse", "HEAD"], repo_root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    status = _git(["status", "--porcelain"], repo_root)
    dirty = bool(status) if status is not None else None

    meta = {
        "run_id": run_id,
        "code_label": code_label,
        "script_used": script_used,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_commit_short": commit[:7] if commit else None,
        "git_branch": branch,
        "git_dirty": dirty,
        "utils_py_sha256": _hash_file(repo_root / "app" / "utils.py"),
        "threshold_snapshot": _threshold_snapshot(),
    }
    if extra:
        meta.update(extra)
    return meta


def write_run_metadata(
    run_dir: Path,
    repo_root: Path,
    code_label: str,
    script_used: str,
    run_id: str,
    extra: dict | None = None,
) -> dict:
    """Write run_metadata.json into run_dir and append an INDEX.md entry."""
    meta = collect_metadata(repo_root, code_label, script_used, run_id, extra)

    meta_path = run_dir / "run_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    _append_index(run_dir.parent, meta, run_dir.name)
    return meta


def _append_index(output_root: Path, meta: dict, folder_name: str) -> None:
    index_path = output_root / "INDEX.md"
    is_new = not index_path.exists()
    line = (
        f"| `{folder_name}` "
        f"| {meta.get('code_label')} "
        f"| {meta.get('git_commit_short') or '-'} "
        f"| {'dirty' if meta.get('git_dirty') else 'clean' if meta.get('git_dirty') is False else '-'} "
        f"| {meta.get('script_used')} "
        f"| {meta.get('started_at')} |\n"
    )
    with index_path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write("# Batch run index\n\n")
            f.write(
                "Auto-appended by `scripts/run_metadata.py` on every batch run. "
                "Folder names embed the `code_label` so pre/post-update runs are "
                "distinguishable at a glance.\n\n"
            )
            f.write("| folder | code_label | commit | tree | script | started_at |\n")
            f.write("|---|---|---|---|---|---|\n")
        f.write(line)
