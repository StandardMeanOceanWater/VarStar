"""Photometry output-path builders and directory helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def format_session_date(session_date: str) -> str:
    return f"{session_date[:4]}-{session_date[4:6]}-{session_date[6:8]}"


def build_run_layout(
    *,
    project_root: Path,
    data_root: Path,
    session_date: str,
    group: str,
    target: str,
    channel: str,
    split_subdir: str = "splits",
    run_ts: str | None = None,
    create_dirs: bool = True,
) -> dict[str, Path | str]:
    channel_key = str(channel).upper()
    date_fmt = format_session_date(session_date)
    field_root = data_root / date_fmt / group
    wcs_dir = field_root / split_subdir / channel_key
    active_run_ts = run_ts or datetime.now().strftime("%Y%m%d_%H%M")
    run_root = project_root / "output" / date_fmt / group / target / active_run_ts
    out_dir = run_root / "1_photometry"
    diag_dir = run_root / "2_regression_diag"
    lc_dir = run_root / "3_light_curve"
    pa_dir = run_root / "4_period_analysis"

    if create_dirs:
        for path in (out_dir, diag_dir, lc_dir, pa_dir):
            path.mkdir(parents=True, exist_ok=True)

    return {
        "date_fmt": date_fmt,
        "field_root": field_root,
        "wcs_dir": wcs_dir,
        "run_ts": active_run_ts,
        "run_root": run_root,
        "out_dir": out_dir,
        "diag_dir": diag_dir,
        "lc_dir": lc_dir,
        "pa_dir": pa_dir,
        "phot_out_csv": out_dir / f"photometry_{channel_key}_{session_date}.csv",
        "phot_out_png": lc_dir / f"light_curve_{channel_key}_{session_date}.png",
        "stars_csv": out_dir / "stars_detected.csv",
    }


def build_target_log_path(run_root: Path, active_date: str, log_ts: str) -> Path:
    return run_root / "1_photometry" / f"photometry_{active_date}_{log_ts}.log"


def get_field_catalog_path(run_root: Path) -> Path:
    return run_root.parent.parent / "catalog.csv"


def build_vsx_run_layout(
    *,
    project_root: Path,
    date_fmt: str,
    group: str,
    target: str,
    run_ts: str,
    create_dirs: bool = True,
) -> dict[str, Path]:
    run_root = project_root / "output" / date_fmt / group / target / run_ts
    out_dir = run_root / "1_photometry"
    diag_dir = run_root / "2_regression_diag"
    lc_dir = run_root / "3_light_curve"
    pa_dir = run_root / "4_period_analysis"

    if create_dirs:
        for path in (out_dir, diag_dir, lc_dir, pa_dir):
            path.mkdir(parents=True, exist_ok=True)

    return {
        "run_root": run_root,
        "out_dir": out_dir,
        "diag_dir": diag_dir,
        "lc_dir": lc_dir,
        "pa_dir": pa_dir,
    }


__all__ = [
    "build_run_layout",
    "build_target_log_path",
    "build_vsx_run_layout",
    "format_session_date",
    "get_field_catalog_path",
]
