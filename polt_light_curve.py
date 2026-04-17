"""
Replot light-curve PNGs from existing photometry CSV files.

Default output: sibling run root's 3_light_curve directory, e.g.
E:\\VarStar\\output\\YYYY-MM-DD\\...\\<run_ts>\\3_light_curve\\

This script does not modify CSVs. It only reads them and writes PNGs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_pipeline_imports(script_path: Path) -> None:
    pipeline_dir = script_path.parent if script_path.parent.name != "tools" else script_path.parent.parent
    if str(pipeline_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_dir))


def _infer_channel_and_date(csv_path: Path) -> tuple[str | None, str | None]:
    # Expected: photometry_<CHANNEL>_<YYYYMMDD>.csv
    m = re.match(r"photometry_([A-Za-z0-9]+)_(\d{8})\.csv$", csv_path.name)
    if not m:
        return None, None
    return m.group(1).upper(), m.group(2)


def _infer_run_root(csv_path: Path) -> Path | None:
    # Typical: ...\<run_ts>\1_photometry\<csv>
    if csv_path.parent.name != "1_photometry":
        return None
    return csv_path.parent.parent


def _infer_target_name(csv_path: Path) -> str | None:
    # Typical output path:
    # E:\VarStar\output\YYYY-MM-DD\<group>\<target>\<run_ts>\1_photometry\...
    parts = list(csv_path.parts)
    try:
        output_idx = parts.index("output")
    except ValueError:
        return None
    # Ensure we have at least: output / date / group / target / run_ts / 1_photometry / file
    if len(parts) <= output_idx + 4:
        return None
    return parts[output_idx + 3]


def _build_cfg(
    target_name: str | None,
    channel: str | None,
    tz_offset_hours: float,
    obs_date: str | None,
) -> SimpleNamespace:
    # Minimal cfg object for plot_light_curve
    return SimpleNamespace(
        target_name=target_name or "UnknownTarget",
        phot_band=(channel or "V"),
        tz_offset_hours=tz_offset_hours,
    )


def plot_light_curve(
    df: pd.DataFrame,
    out_png: Path,
    channel: str,
    cfg,
    obs_date: "str | None" = None,
    ylim: "tuple[float, float] | None" = None,
):
    """
    Generate a light curve plot from a photometry DataFrame.

    ylim: (ymin, ymax) in mag; the caller may pass inverted bounds already.
    """
    from astropy.time import Time as ATime
    import datetime as _dt_local

    time_key = "bjd_tdb" if "bjd_tdb" in df.columns else "jd"
    if time_key not in df.columns:
        print(f"[PLOT] Error: time column '{time_key}' not found.")
        return

    mag_col = "m_var"
    d = df[(df["ok"] == 1) & np.isfinite(df[time_key]) & np.isfinite(df[mag_col])].copy()
    if len(d) == 0:
        print("[PLOT] No valid photometry points to plot.")
        return

    bjd_arr = d[time_key].values
    bjd_min = float(bjd_arr.min())
    bjd_max = float(bjd_arr.max())
    tz_offset_h = float(getattr(cfg, "tz_offset_hours", 8))

    def _bjd_to_local_hm(bjd_val):
        try:
            t_utc = ATime(bjd_val, format="jd", scale="tdb").to_datetime()
            return t_utc + _dt_local.timedelta(hours=tz_offset_h)
        except Exception:
            return None

    t_local_min = _bjd_to_local_hm(bjd_min)
    t_local_max = _bjd_to_local_hm(bjd_max)
    label30_bjd_ticks = []
    label30_labels = []
    minor_bjd_ticks = []
    if t_local_min is not None and t_local_max is not None:
        cur = t_local_min.replace(second=0, microsecond=0)
        cur = cur.replace(minute=(cur.minute // 10) * 10)
        while cur <= t_local_max + _dt_local.timedelta(minutes=1):
            utc = cur - _dt_local.timedelta(hours=tz_offset_h)
            b_tick = ATime(utc).jd
            minor_bjd_ticks.append(b_tick)
            if cur.minute in (0, 30):
                label30_bjd_ticks.append(b_tick)
                label30_labels.append(cur.strftime("%H:%M"))
            cur += _dt_local.timedelta(minutes=10)

    fig, ax = plt.subplots(figsize=(12, 4))
    y_vals = d[mag_col].values
    if "t_sigma_mag" in d.columns and np.isfinite(d["t_sigma_mag"]).any():
        ax.errorbar(
            bjd_arr,
            y_vals,
            yerr=d["t_sigma_mag"].values,
            fmt="o",
            ms=4,
            capsize=2,
            lw=0.8,
            label="± σ",
            zorder=3,
        )
    else:
        ax.plot(bjd_arr, y_vals, "o-", ms=4, lw=0.8, label="± σ", zorder=3)

    ax.invert_yaxis()
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
        y_lo = min(ylim)
        y_hi = max(ylim)
        yticks = np.arange(np.ceil(y_lo * 20) / 20, y_hi + 0.001, 0.05)
        ax.set_yticks(yticks)
    ax.set_xlim(bjd_min - 0.002, bjd_max + 0.002)

    if minor_bjd_ticks:
        ax.set_xticks(minor_bjd_ticks, minor=True)
    if label30_bjd_ticks:
        ax.set_xticks(label30_bjd_ticks)
        ax.set_xticklabels(label30_labels, color="navy", fontsize=9)
    ax.tick_params(axis="x", which="major", direction="out", colors="navy", length=6)
    ax.tick_params(axis="x", which="minor", direction="out", colors="navy", length=3)

    xlim = ax.get_xlim()
    xspan = xlim[1] - xlim[0]
    for idx, tick_v in enumerate(label30_bjd_ticks):
        if idx == 0:
            continue
        xf = (tick_v - xlim[0]) / xspan
        if 0.0 <= xf <= 1.0:
            ax.text(
                xf,
                0.01,
                f"{tick_v:.4f}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=7,
                color="navy",
                alpha=0.4,
                rotation=45,
                clip_on=True,
                zorder=5,
            )

    ax.text(0.0, 0.01, "BJD", transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color="navy", zorder=5)
    ax.set_xlabel("Local Time (HH:MM)", color="navy", loc="left", fontsize=9, labelpad=2)
    ax.set_ylabel("Calibrated Magnitude (mag)")

    title_star = getattr(cfg, "display_name", None) or getattr(cfg, "target_name", "Target")
    ax.set_title(f"{title_star} Light Curve [{channel}]", fontsize=22, fontweight="bold", pad=10)

    fs = 16
    obs_str = obs_date or ""
    ax.text(0.01, 1.02, obs_str, transform=ax.transAxes, ha="left", va="bottom", fontsize=fs, color="navy")
    ax.legend(
        fontsize=fs,
        loc="upper left",
        frameon=True,
        edgecolor="gray",
        borderaxespad=0.2,
        handletextpad=0.0,
        handlelength=1.0,
        borderpad=0.2,
        labelspacing=0.2,
    )

    lat = getattr(cfg, "obs_lat_deg", None)
    lon = getattr(cfg, "obs_lon_deg", None)
    if lat is not None and lon is not None:
        left = f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}"
        right = f"{abs(lon):.2f}°{'E' if lon >= 0 else 'W'}"
        ax.text(0.99, 1.02, f"{left} {right}", transform=ax.transAxes, ha="right", va="bottom", fontsize=fs, color="#2d6a4f")

    ax.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close("all")
    print(f"[PNG] saved → {out_png}")


def _resolve_output_dir(csv_path: Path, explicit_out_dir: Path | None) -> Path:
    if explicit_out_dir is not None:
        return explicit_out_dir
    run_root = _infer_run_root(csv_path)
    if run_root is not None:
        return run_root / "3_light_curve"
    return csv_path.parent


def _resolve_period_analysis_dir(csv_path: Path) -> Path | None:
    run_root = _infer_run_root(csv_path)
    if run_root is None:
        return None
    return run_root / "4_period_analysis"


def _pipeline_dir(script_path: Path) -> Path:
    return script_path.parent if script_path.parent.name != "tools" else script_path.parent.parent


def _project_root(script_path: Path) -> Path:
    return _pipeline_dir(script_path).parent


def _safe_target_stem(target_name: str | None) -> str:
    return (target_name or "UnknownTarget").replace(" ", "")


def _normalize_target_name(name: str | None) -> str:
    if not name:
        return ""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _extract_first_float(text: str) -> float | None:
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _load_known_period_days(project_root: Path, target_name: str | None) -> tuple[float | None, Path | None]:
    target_norm = _normalize_target_name(target_name)
    if not target_norm:
        return None, None

    table_path = project_root / ".codex" / "memories" / "project" / "project_target_params.md"
    if not table_path.exists():
        return None, None
    try:
        lines = table_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None, None

    for line in lines:
        s = line.strip()
        if not s.startswith("|"):
            continue
        if "---" in s:
            continue
        cols = [c.strip() for c in s.strip("|").split("|")]
        if len(cols) < 3:
            continue
        row_target_norm = _normalize_target_name(cols[0])
        if row_target_norm != target_norm:
            continue
        period_cell = cols[2].replace("*", "")
        period_d = _extract_first_float(period_cell)
        if period_d is not None:
            return period_d, table_path
    return None, table_path


def _load_fourier_fit(csv_path: Path, channel: str | None) -> dict | None:
    pa_dir = _resolve_period_analysis_dir(csv_path)
    if pa_dir is None or channel is None:
        return None
    target_name = _infer_target_name(csv_path)
    fit_path = pa_dir / f"period_fourier_fit_{_safe_target_stem(target_name)}_{channel}.json"
    if not fit_path.exists():
        return None
    try:
        return json.loads(fit_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fourier_series_from_coeffs(phase: np.ndarray, coeffs: list[float]) -> np.ndarray:
    a0 = coeffs[0]
    result = np.full_like(phase, a0, dtype=float)
    n_harmonics = (len(coeffs) - 1) // 2
    for i in range(n_harmonics):
        n = i + 1
        a_n = coeffs[1 + 2 * i]
        b_n = coeffs[2 + 2 * i]
        angle = 2.0 * np.pi * n * phase
        result += a_n * np.cos(angle) + b_n * np.sin(angle)
    return result


def _build_fourier_curve(
    fit_payload: dict,
    t_min: float,
    t_max: float,
    points: int = 2000,
    period_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    coeffs = [float(v) for v in fit_payload["coefficients"]]
    period = float(period_override if period_override is not None else fit_payload["best_period_d"])
    t0 = float(fit_payload["t0"])
    t_dense = np.linspace(t_min, t_max, points)
    phase = ((t_dense - t0) / period) % 1.0
    mag_dense = _fourier_series_from_coeffs(phase, coeffs)
    return t_dense, mag_dense


def _find_latest_run_root(output_root: Path) -> Path | None:
    if not output_root.exists():
        return None
    required_channels = {"R", "G1", "G2", "B"}
    run_candidates: dict[Path, list[Path]] = {}
    all_csvs: list[Path] = []
    for csv_path in output_root.rglob("1_photometry/photometry_*_*.csv"):
        all_csvs.append(csv_path)
        run_root = _infer_run_root(csv_path)
        if run_root is None:
            continue
        run_candidates.setdefault(run_root, []).append(csv_path)

    def _run_has_valid_channels(csv_paths: list[Path]) -> bool:
        seen = set()
        for p in csv_paths:
            ch, _ = _infer_channel_and_date(p)
            if ch:
                seen.add(ch)
        return required_channels.issubset(seen)

    def _run_has_valid_rows(csv_paths: list[Path]) -> bool:
        for p in csv_paths:
            try:
                df = pd.read_csv(p)
            except Exception:
                return False
            if "ok" in df.columns:
                df = df[df["ok"] == 1]
            if len(df) == 0:
                return False
        return True

    best_root = None
    best_mtime = None
    for run_root, csv_paths in run_candidates.items():
        if not _run_has_valid_channels(csv_paths):
            continue
        if not _run_has_valid_rows(csv_paths):
            continue
        try:
            mtime = max(p.stat().st_mtime for p in csv_paths)
        except Exception:
            continue
        if best_mtime is None or mtime > best_mtime:
            best_mtime = mtime
            best_root = run_root

    if best_root is not None:
        return best_root

    # Fallback: pick latest CSV mtime even if run root incomplete
    latest_csv = None
    latest_mtime = None
    for csv_path in all_csvs:
        try:
            mtime = csv_path.stat().st_mtime
        except Exception:
            continue
        if latest_mtime is None or mtime > latest_mtime:
            latest_mtime = mtime
            latest_csv = csv_path
    if latest_csv is None:
        return None
    return _infer_run_root(latest_csv)


def _infer_obs_date_from_csvs(csv_paths: list[Path]) -> str | None:
    dates: dict[str, int] = {}
    for p in csv_paths:
        _, dt = _infer_channel_and_date(p)
        if dt:
            dates[dt] = dates.get(dt, 0) + 1
    if not dates:
        return None
    return sorted(dates.items(), key=lambda x: (-x[1], x[0]))[0][0]


def _plot_one(
    csv_path: Path,
    out_dir: Path,
    channel: str | None,
    obs_date: str | None,
    tz_offset_hours: float,
    ylim: tuple[float, float] | None = None,
) -> Path:
    from pipeline_config import load_pipeline_config
    from photometry import cfg_from_yaml

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError(f"CSV has no rows: {csv_path}")

    # Ensure v_err exists for downstream compatibility (plot does not require it)
    if "v_err" not in df.columns and "t_sigma_mag" in df.columns:
        df["v_err"] = df["t_sigma_mag"]

    target_name = _infer_target_name(csv_path)
    cfg = None
    if target_name and channel and obs_date:
        try:
            yaml_dict = load_pipeline_config()
            cfg = cfg_from_yaml(yaml_dict, target_name, obs_date, channel=channel)
        except Exception:
            cfg = None

    if cfg is None:
        cfg = _build_cfg(target_name, channel, tz_offset_hours, obs_date)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep name consistent with main pipeline
    ch = channel or "UNK"
    date_str = obs_date or "unknown"
    out_png = out_dir / f"light_curve_{ch}_{date_str}.png"

    plot_light_curve(df, out_png, ch, cfg, obs_date=obs_date, ylim=ylim)
    return out_png


def _compute_shared_ylim(csv_paths: list[Path], mag_col: str = "m_var") -> tuple[float, float] | None:
    mags: list[float] = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if mag_col not in df.columns:
            mag_col = "m_var"
        if "ok" in df.columns:
            df = df[df["ok"] == 1]
        vals = df[mag_col].dropna().values
        if len(vals):
            mags.extend([float(v) for v in vals])
    if not mags:
        return None
    margin = 0.02
    ymin = min(mags) - margin
    ymax = max(mags) + margin
    # plot_light_curve expects (ymin, ymax) with inversion already considered
    return (ymax, ymin)


def _load_cfg_from_csv(csv_path: Path, channel: str | None, obs_date: str | None, tz_offset_hours: float) -> SimpleNamespace:
    from pipeline_config import load_pipeline_config
    from photometry import cfg_from_yaml

    target_name = _infer_target_name(csv_path)
    if target_name and channel and obs_date:
        try:
            yaml_dict = load_pipeline_config()
            return cfg_from_yaml(yaml_dict, target_name, obs_date, channel=channel)
        except Exception:
            pass
    return _build_cfg(target_name, channel, tz_offset_hours, obs_date)


def _plot_overlay(
    csv_by_channel: dict[str, Path],
    out_dir: Path,
    obs_date: str | None,
    tz_offset_hours: float,
    out_name: str | None = None,
    width_cm: float | None = None,
    width_px: int | None = None,
    height_px: int | None = None,
    dpi: int = 360,
    save_jpg: bool = False,
) -> Path:
    # Overlay style config (集中管理避免魔數)
    style = {
        "colors": {
            "data": {"B": "#9CDDEE", "G1": "#22B14C", "G2": "#9CC51A", "R": "#F6A8C2"},
            "err": {"B": "#76A8B5", "G1": "#1A8439", "G2": "#789813", "R": "#CE8CA2"},
            "bg": "#232323",
            "axes_bg": "#2b2b2b",
            "title": "#FFFFFF",
            "date": "#FFF200",
            "coord": "#FFA500",
            "bjd": "#AB82FF",
            "local_time": "#00BFFF",
            "text": "#FFFFFF",
            "grid": "#8a8a8a",
            "legend_bg": "#2a2a2a",
        },
        "sizes": {
            "title": 11,
            "date": 8,
            "coord": 8,
            "ylabel": 6,
            "ytick": 6,
            "xtick": 5,
            "local_label": 5,
            "bjd_label": 4,
            "legend": 7,
            "dot": 3.3,
            "tick_main": 3,
            "tick_bjd": 6,
            "spine_w": 2.0,
        },
        "layout": {
            "left": 0.07,
            "right": 0.985,
            "bottom": 0.06,
            "top": 0.915,
        },
        "bjd": {
            "y": 0.01125,
            "alpha": 0.6,
            "rotation": 45,
        },
        "local_time": {
            "y": 0.0,
        },
        "header": {
            "y": 1.0,
        },
        "ylim": {
            "margin": 0.02,
            "extra": 0.1,
        },
        "grid": {
            "alpha": 0.9,
            "lw": 0.9,
        },
    }
    data_colors = style["colors"]["data"]
    err_colors = style["colors"]["err"]
    bg_color = style["colors"]["bg"]
    axes_bg_color = style["colors"]["axes_bg"]
    title_color = style["colors"]["title"]
    date_color = style["colors"]["date"]
    coord_color = style["colors"]["coord"]
    bjd_color = style["colors"]["bjd"]
    local_time_color = style["colors"]["local_time"]
    text_color = style["colors"]["text"]
    grid_color = style["colors"]["grid"]

    # Use first available channel to build cfg for title/coords
    first_ch = next(iter(csv_by_channel.keys()))
    cfg = _load_cfg_from_csv(csv_by_channel[first_ch], first_ch, obs_date, tz_offset_hours)

    # Collect data and determine time key
    series = []
    time_key = None
    for ch, path in csv_by_channel.items():
        df = pd.read_csv(path)
        if "ok" in df.columns:
            df = df[df["ok"] == 1]
        # prefer bjd_tdb, then jd, then mjd
        if time_key is None:
            if "bjd_tdb" in df.columns:
                time_key = "bjd_tdb"
            elif "jd" in df.columns:
                time_key = "jd"
            else:
                time_key = "mjd"
        if time_key not in df.columns:
            continue
        mag_col = "m_var" if "m_var" in df.columns else "m_var_norm"
        df = df[pd.notna(df[time_key]) & pd.notna(df[mag_col])]
        err_col = None
        if "t_sigma_mag" in df.columns and pd.notna(df["t_sigma_mag"]).any():
            err_col = "t_sigma_mag"
        elif "v_err" in df.columns and pd.notna(df["v_err"]).any():
            err_col = "v_err"
        yerr = df[err_col].values if err_col else None
        series.append((ch, df[time_key].values, df[mag_col].values, yerr))

    if not series:
        raise ValueError("No valid data to plot in overlay.")

    # Compute shared y-limits across all channels
    all_mags = [m for _, _, mags, _ in series for m in mags]
    ymin = min(all_mags) - style["ylim"]["margin"] - style["ylim"]["extra"]
    ymax = max(all_mags) + style["ylim"]["margin"] + style["ylim"]["extra"]

    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = obs_date or "unknown"
    if out_name:
        out_png = out_dir / out_name
    else:
        out_png = out_dir / f"light_curve_overlay_{date_str}.png"

    # Plot
    # Match original plot_light_curve proportions (3:1)
    if width_cm and width_cm > 0:
        width_in = width_cm / 2.54
        height_in = width_in / 3.0
        fig_size = (width_in, height_in)
    elif width_px and height_px:
        fig_size = (width_px / dpi, height_px / dpi)
    else:
        fig_size = (2400 / dpi, 800 / dpi)
    plt.figure(figsize=fig_size, facecolor=bg_color)
    ax = plt.gca()
    ax.set_facecolor(axes_bg_color)
    ax.grid(True, color=grid_color, alpha=style["grid"]["alpha"], linewidth=style["grid"]["lw"])
    for spine in ax.spines.values():
        spine.set_color(text_color)
    ax.spines["left"].set_color("#FFFFFF")
    ax.spines["right"].set_color("#FFFFFF")
    ax.spines["left"].set_linewidth(style["sizes"]["spine_w"])
    ax.spines["right"].set_linewidth(style["sizes"]["spine_w"])

    for ch, tvals, mvals, yerr in series:
        ax.errorbar(
            tvals,
            mvals,
            yerr=yerr,
            fmt="o",
            ms=style["sizes"]["dot"],
            capsize=2,
            lw=0.8,
            elinewidth=0.8,
            color=data_colors.get(ch, "#FFFFFF"),
            ecolor=err_colors.get(ch, data_colors.get(ch, "#FFFFFF")),
            alpha=0.9,
            label=ch,
            zorder=3,
        )

    ax.set_xlabel("")
    ax.set_ylabel("Calibrated Magnitude (mag)", color=text_color, fontsize=style["sizes"]["ylabel"])
    ax.tick_params(
        axis="x",
        colors=local_time_color,
        labelsize=style["sizes"]["xtick"],
        direction="out",
        length=style["sizes"]["tick_main"],
        pad=0,
    )
    ax.tick_params(axis="y", colors=text_color, labelsize=style["sizes"]["ytick"])
    ax.xaxis.get_offset_text().set_color(bjd_color)
    ax.xaxis.get_offset_text().set_fontsize(6)

    # Invert y-axis for magnitudes
    ax.set_ylim(ymax, ymin)

    title = f"{getattr(cfg, 'target_name', 'Target')} Light Curve"
    ax.set_title(title, color=title_color, fontsize=style["sizes"]["title"], fontweight="bold", pad=1, y=1.0)
    ax.legend(
        fontsize=style["sizes"]["legend"],
        frameon=True,
        facecolor=style["colors"]["legend_bg"],
        edgecolor=text_color,
        labelcolor=text_color,
        loc="upper right",
    )

    # Header text aligned to top axis, inside left/right edges
    ax.text(
        0.01,
        style["header"]["y"],
        date_str,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=style["sizes"]["date"],
        color=date_color,
    )
    if getattr(cfg, "obs_lat_deg", None) is not None and getattr(cfg, "obs_lon_deg", None) is not None:
        ax.text(
            0.99,
            style["header"]["y"],
            f"{cfg.obs_lat_deg:.2f}N {cfg.obs_lon_deg:.2f}E",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=style["sizes"]["coord"],
            color=coord_color,
        )
    # Local Time labels on bottom; BJD labels on upper side of bottom axis
    try:
        from astropy.time import Time as ATime
        import datetime as _dt_local

        if time_key is None:
            time_key = "bjd_tdb"
        _time_format = "jd" if time_key in ("bjd_tdb", "jd") else "mjd"

        bjd_all = [t for _, tvals, _, _ in series for t in tvals]
        bjd_min = float(min(bjd_all))
        bjd_max = float(max(bjd_all))

        def _to_local(bjd_val):
            t_utc = ATime(bjd_val, format=_time_format, scale="tdb").to_datetime()
            return t_utc + _dt_local.timedelta(hours=float(tz_offset_hours))

        _t_local_min = _to_local(bjd_min)
        _t_local_max = _to_local(bjd_max)
        _label_bjd_ticks = []
        _label_local = []
        _cur = _t_local_min.replace(second=0, microsecond=0)
        _cur = _cur.replace(minute=(_cur.minute // 10) * 10)
        while _cur <= _t_local_max + _dt_local.timedelta(minutes=1):
            _utc = _cur - _dt_local.timedelta(hours=float(tz_offset_hours))
            _b_tick = ATime(_utc).jd if _time_format == "jd" else ATime(_utc).mjd
            if _cur.minute in (0, 30):
                _label_bjd_ticks.append(_b_tick)
                _label_local.append(_cur.strftime("%H:%M"))
            _cur += _dt_local.timedelta(minutes=10)

        ax.set_xticks(_label_bjd_ticks)
        if _label_local:
            _label_local[0] = "Local Time (HH:MM)"
        ax.set_xticklabels(_label_local, color=local_time_color, fontsize=style["sizes"]["xtick"])
        # Hide the first tick mark above Local Time label
        ticks = ax.xaxis.get_major_ticks()
        if ticks:
            ticks[0].tick1line.set_markersize(0)
            ticks[0].tick2line.set_markersize(0)

        bjd_ax = ax.twiny()
        bjd_ax.set_xlim(ax.get_xlim())
        bjd_ax.set_xticks(_label_bjd_ticks)
        bjd_ax.tick_params(
            axis="x",
            colors=bjd_color,
            direction="in",
            length=style["sizes"]["tick_bjd"],
            pad=-14,
            labelsize=style["sizes"]["bjd_label"],
        )
        bjd_ax.set_xticklabels([])
        bjd_ax.xaxis.set_ticks_position("bottom")
        bjd_ax.xaxis.set_label_position("bottom")
        bjd_ax.spines["top"].set_visible(False)
        bjd_ax.spines["bottom"].set_color(text_color)
        bjd_ax.spines["left"].set_visible(False)
        bjd_ax.spines["right"].set_visible(False)
        bjd_ax.yaxis.set_visible(False)
        bjd_ax.set_xlabel("")

        # Draw BJD numbers near the bottom-axis upper edge
        x0, x1 = ax.get_xlim()
        xspan = x1 - x0
        for v in _label_bjd_ticks:
            xf = (v - x0) / xspan
            if 0.0 <= xf <= 1.0:
                ax.text(
                    xf,
                    style["bjd"]["y"],
                    f"{v:.4f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=style["sizes"]["bjd_label"],
                    color=bjd_color,
                    alpha=style["bjd"]["alpha"],
                    rotation=style["bjd"]["rotation"],
                    clip_on=True,
                )
        ax.text(
            0.015,
            style["bjd"]["y"],
            "BJD",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=style["sizes"]["bjd_label"],
            color=bjd_color,
        )

    except Exception:
        pass
    fig = ax.get_figure()
    fig.subplots_adjust(
        left=style["layout"]["left"],
        right=style["layout"]["right"],
        bottom=style["layout"]["bottom"],
        top=style["layout"]["top"],
    )
    plt.savefig(out_png, dpi=dpi, facecolor=bg_color)
    if save_jpg:
        out_jpg = out_png.with_suffix(".jpg")
        plt.savefig(out_jpg, dpi=dpi, facecolor=bg_color)
    plt.close()

    return out_png


def _plot_overlay_with_fourier(
    csv_by_channel: dict[str, Path],
    out_dir: Path,
    obs_date: str | None,
    tz_offset_hours: float,
    out_name: str | None = None,
    width_px: int = 2400,
    height_px: int = 800,
    dpi: int = 360,
) -> Path:
    style = {
        "colors": {
            "data": {"B": "#9CDDEE", "G1": "#22B14C", "G2": "#9CC51A", "R": "#F6A8C2"},
            "err": {"B": "#76A8B5", "G1": "#1A8439", "G2": "#789813", "R": "#CE8CA2"},
            "fit": {"B": "#A6EDFF", "G1": "#30FA6B", "G2": "#C9FF20", "R": "#E11B23"},
            "bg": "#232323",
            "axes_bg": "#2b2b2b",
            "title": "#FFFFFF",
            "date": "#FFF200",
            "coord": "#FFA500",
            "bjd": "#AB82FF",
            "local_time": "#00BFFF",
            "text": "#FFFFFF",
            "grid": "#8a8a8a",
            "legend_bg": "#2a2a2a",
        },
        "sizes": {
            "title": 11,
            "date": 8,
            "coord": 8,
            "ylabel": 6,
            "ytick": 6,
            "xtick": 5,
            "bjd_label": 4,
            "legend": 7,
            "dot": 3.3,
            "tick_main": 3,
            "tick_bjd": 6,
            "spine_w": 2.0,
        },
        "layout": {"left": 0.07, "right": 0.985, "bottom": 0.06, "top": 0.915},
        "bjd": {"y": 0.01125, "alpha": 0.6, "rotation": 45},
        "header": {"y": 1.0},
        "ylim": {"margin": 0.02, "extra": 0.1},
        "grid": {"alpha": 0.9, "lw": 0.9},
    }
    first_ch = next(iter(csv_by_channel.keys()))
    cfg = _load_cfg_from_csv(csv_by_channel[first_ch], first_ch, obs_date, tz_offset_hours)

    series = []
    time_key = None
    for ch, path in csv_by_channel.items():
        df = pd.read_csv(path)
        if "ok" in df.columns:
            df = df[df["ok"] == 1]
        if time_key is None:
            if "bjd_tdb" in df.columns:
                time_key = "bjd_tdb"
            elif "jd" in df.columns:
                time_key = "jd"
            else:
                time_key = "mjd"
        if time_key not in df.columns:
            continue
        mag_col = "m_var" if "m_var" in df.columns else "m_var_norm"
        df = df[pd.notna(df[time_key]) & pd.notna(df[mag_col])]
        err_col = None
        if "t_sigma_mag" in df.columns and pd.notna(df["t_sigma_mag"]).any():
            err_col = "t_sigma_mag"
        elif "v_err" in df.columns and pd.notna(df["v_err"]).any():
            err_col = "v_err"
        fit_payload = _load_fourier_fit(path, ch)
        series.append(
            (
                ch,
                df[time_key].values,
                df[mag_col].values,
                df[err_col].values if err_col else None,
                fit_payload,
            )
        )

    if not series:
        raise ValueError("No valid data to plot in Fourier overlay.")

    all_mags = [m for _, _, mags, _, _ in series for m in mags]
    ymin = min(all_mags) - style["ylim"]["margin"] - style["ylim"]["extra"]
    ymax = max(all_mags) + style["ylim"]["margin"] + style["ylim"]["extra"]

    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = obs_date or "unknown"
    out_png = out_dir / (out_name or f"light_curve_overlay_{date_str}_fourier.png")

    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), facecolor=style["colors"]["bg"])
    ax = plt.gca()
    ax.set_facecolor(style["colors"]["axes_bg"])
    ax.grid(True, color=style["colors"]["grid"], alpha=style["grid"]["alpha"], linewidth=style["grid"]["lw"])
    for spine in ax.spines.values():
        spine.set_color(style["colors"]["text"])
    ax.spines["left"].set_linewidth(style["sizes"]["spine_w"])
    ax.spines["right"].set_linewidth(style["sizes"]["spine_w"])

    for ch, tvals, mvals, yerr, fit_payload in series:
        ax.errorbar(
            tvals,
            mvals,
            yerr=yerr,
            fmt="o",
            ms=style["sizes"]["dot"],
            capsize=2,
            lw=0.8,
            elinewidth=0.8,
            color=style["colors"]["data"].get(ch, "#FFFFFF"),
            ecolor=style["colors"]["err"].get(ch, "#FFFFFF"),
            alpha=0.9,
            label=ch,
            zorder=3,
        )
        if fit_payload:
            t_dense, mag_dense = _build_fourier_curve(fit_payload, float(np.min(tvals)), float(np.max(tvals)))
            ax.plot(
                t_dense,
                mag_dense,
                color=style["colors"]["fit"].get(ch, "#FFFFFF"),
                lw=1.6,
                alpha=0.95,
                zorder=4,
            )

    ax.set_xlabel("")
    ax.set_ylabel("Calibrated Magnitude (mag)", color=style["colors"]["text"], fontsize=style["sizes"]["ylabel"])
    ax.tick_params(axis="x", colors=style["colors"]["local_time"], labelsize=style["sizes"]["xtick"], direction="out", length=style["sizes"]["tick_main"], pad=0)
    ax.tick_params(axis="y", colors=style["colors"]["text"], labelsize=style["sizes"]["ytick"])
    ax.set_ylim(ymax, ymin)
    ax.set_title(f"{getattr(cfg, 'target_name', 'Target')} Light Curve", color=style["colors"]["title"], fontsize=style["sizes"]["title"], fontweight="bold", pad=1, y=1.0)
    ax.legend(fontsize=style["sizes"]["legend"], frameon=True, facecolor=style["colors"]["legend_bg"], edgecolor=style["colors"]["text"], labelcolor=style["colors"]["text"], loc="upper right")
    ax.text(0.01, style["header"]["y"], date_str, transform=ax.transAxes, ha="left", va="bottom", fontsize=style["sizes"]["date"], color=style["colors"]["date"])
    if getattr(cfg, "obs_lat_deg", None) is not None and getattr(cfg, "obs_lon_deg", None) is not None:
        ax.text(0.99, style["header"]["y"], f"{cfg.obs_lat_deg:.2f}N {cfg.obs_lon_deg:.2f}E", transform=ax.transAxes, ha="right", va="bottom", fontsize=style["sizes"]["coord"], color=style["colors"]["coord"])

    try:
        from astropy.time import Time as ATime
        import datetime as _dt_local

        _time_format = "jd" if time_key in ("bjd_tdb", "jd") else "mjd"
        bjd_all = [t for _, tvals, _, _, _ in series for t in tvals]
        bjd_min = float(min(bjd_all))
        bjd_max = float(max(bjd_all))

        def _to_local(bjd_val: float):
            t_utc = ATime(bjd_val, format=_time_format, scale="tdb").to_datetime()
            return t_utc + _dt_local.timedelta(hours=float(tz_offset_hours))

        _t_local_min = _to_local(bjd_min)
        _t_local_max = _to_local(bjd_max)
        _label_bjd_ticks = []
        _label_local = []
        _cur = _t_local_min.replace(second=0, microsecond=0)
        _cur = _cur.replace(minute=(_cur.minute // 10) * 10)
        while _cur <= _t_local_max + _dt_local.timedelta(minutes=1):
            _utc = _cur - _dt_local.timedelta(hours=float(tz_offset_hours))
            _b_tick = ATime(_utc).jd if _time_format == "jd" else ATime(_utc).mjd
            if _cur.minute in (0, 30):
                _label_bjd_ticks.append(_b_tick)
                _label_local.append(_cur.strftime("%H:%M"))
            _cur += _dt_local.timedelta(minutes=10)

        ax.set_xticks(_label_bjd_ticks)
        if _label_local:
            _label_local[0] = "Local Time (HH:MM)"
        ax.set_xticklabels(_label_local, color=style["colors"]["local_time"], fontsize=style["sizes"]["xtick"])
        ticks = ax.xaxis.get_major_ticks()
        if ticks:
            ticks[0].tick1line.set_markersize(0)
            ticks[0].tick2line.set_markersize(0)

        bjd_ax = ax.twiny()
        bjd_ax.set_xlim(ax.get_xlim())
        bjd_ax.set_xticks(_label_bjd_ticks)
        bjd_ax.tick_params(axis="x", colors=style["colors"]["bjd"], direction="in", length=style["sizes"]["tick_bjd"], pad=-14, labelsize=style["sizes"]["bjd_label"])
        bjd_ax.set_xticklabels([])
        bjd_ax.xaxis.set_ticks_position("bottom")
        bjd_ax.xaxis.set_label_position("bottom")
        bjd_ax.spines["top"].set_visible(False)
        bjd_ax.spines["bottom"].set_color(style["colors"]["text"])
        bjd_ax.spines["left"].set_visible(False)
        bjd_ax.spines["right"].set_visible(False)
        bjd_ax.yaxis.set_visible(False)

        x0, x1 = ax.get_xlim()
        xspan = x1 - x0
        for v in _label_bjd_ticks:
            xf = (v - x0) / xspan
            if 0.0 <= xf <= 1.0:
                ax.text(xf, style["bjd"]["y"], f"{v:.4f}", transform=ax.transAxes, ha="left", va="bottom", fontsize=style["sizes"]["bjd_label"], color=style["colors"]["bjd"], alpha=style["bjd"]["alpha"], rotation=style["bjd"]["rotation"], clip_on=True)
        ax.text(0.015, style["bjd"]["y"], "BJD", transform=ax.transAxes, ha="left", va="bottom", fontsize=style["sizes"]["bjd_label"], color=style["colors"]["bjd"])
    except Exception:
        pass

    fig.subplots_adjust(left=style["layout"]["left"], right=style["layout"]["right"], bottom=style["layout"]["bottom"], top=style["layout"]["top"])
    plt.savefig(out_png, dpi=dpi, facecolor=style["colors"]["bg"])
    plt.close()
    return out_png


def _plot_channel_with_fourier(
    csv_path: Path,
    out_dir: Path,
    channel: str,
    obs_date: str | None,
    tz_offset_hours: float,
    out_name: str | None = None,
    accepted_period_days: float | None = None,
    width_px: int = 2400,
    height_px: int = 800,
    dpi: int = 360,
) -> Path:
    fit_payload = _load_fourier_fit(csv_path, channel)
    if fit_payload is None:
        raise ValueError(f"Missing Fourier fit json for {channel}: {csv_path}")

    df = pd.read_csv(csv_path)
    if "ok" in df.columns:
        df = df[df["ok"] == 1]
    time_key = "bjd_tdb" if "bjd_tdb" in df.columns else "jd" if "jd" in df.columns else "mjd"
    mag_col = "m_var" if "m_var" in df.columns else "m_var_norm"
    df = df[pd.notna(df[time_key]) & pd.notna(df[mag_col])].copy()
    if len(df) == 0:
        raise ValueError(f"No valid rows for channel {channel}: {csv_path}")

    err_col = None
    if "t_sigma_mag" in df.columns and pd.notna(df["t_sigma_mag"]).any():
        err_col = "t_sigma_mag"
    elif "v_err" in df.columns and pd.notna(df["v_err"]).any():
        err_col = "v_err"

    cfg = _load_cfg_from_csv(csv_path, channel, obs_date, tz_offset_hours)
    colors = {
        "data": {"B": "#9CDDEE", "G1": "#22B14C", "G2": "#9CC51A", "R": "#F6A8C2"},
        "err": {"B": "#76A8B5", "G1": "#1A8439", "G2": "#789813", "R": "#CE8CA2"},
        "fit": {"B": "#A6EDFF", "G1": "#30FA6B", "G2": "#C9FF20", "R": "#E11B23"},
        "bg": "#232323",
        "axes_bg": "#2b2b2b",
        "title": "#FFFFFF",
        "date": "#FFF200",
        "coord": "#FFA500",
        "bjd": "#AB82FF",
        "local_time": "#00BFFF",
        "text": "#FFFFFF",
        "grid": "#8A8A8A",
        "legend_bg": "#2a2a2a",
    }
    sizes = {
        "title": 11,
        "date": 8,
        "coord": 8,
        "ylabel": 6,
        "ytick": 6,
        "xtick": 5,
        "bjd_label": 4,
        "legend": 7,
        "dot": 3.3,
        "tick_main": 3,
        "tick_bjd": 6,
        "spine_w": 2.0,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = obs_date or "unknown"
    out_png = out_dir / (out_name or f"light_curve_{channel}_{date_str}_fourier.png")

    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), facecolor=colors["bg"])
    ax.set_facecolor(colors["axes_bg"])
    ax.grid(True, color=colors["grid"], alpha=0.9, linewidth=0.9)
    for spine in ax.spines.values():
        spine.set_color(colors["text"])
        spine.set_linewidth(sizes["spine_w"])

    ax.errorbar(
        df[time_key].values,
        df[mag_col].values,
        yerr=df[err_col].values if err_col else None,
        fmt="o",
        ms=sizes["dot"],
        capsize=2,
        lw=0.8,
        elinewidth=0.9,
        color=colors["data"].get(channel, "#FFFFFF"),
        ecolor=colors["err"].get(channel, colors["data"].get(channel, "#FFFFFF")),
        alpha=0.95,
        label="±1σ",
        zorder=3,
    )

    t_min = float(df[time_key].min())
    t_max = float(df[time_key].max())
    t_dense, mag_dense = _build_fourier_curve(fit_payload, t_min, t_max)
    period = float(fit_payload["best_period_d"])
    n_harm = int(fit_payload["n_harmonics"])
    ax.plot(
        t_dense,
        mag_dense,
        color=colors["fit"].get(channel, "#FFFFFF"),
        lw=2.4,
        label=f"Fourier\nP={period * 24.0:.2f} h\nN={n_harm}",
        zorder=4,
    )

    if accepted_period_days is not None:
        t_ref, mag_ref = _build_fourier_curve(
            fit_payload,
            t_min,
            t_max,
            period_override=float(accepted_period_days),
        )
        ax.plot(
            t_ref,
            mag_ref,
            color=colors["coord"],
            lw=1.6,
            ls="--",
            alpha=0.9,
            label=f"Ref\nP={float(accepted_period_days) * 24.0:.2f} h\nN={n_harm}",
            zorder=4,
        )

    all_vals = list(df[mag_col].values) + list(mag_dense)
    if accepted_period_days is not None:
        all_vals.extend(list(mag_ref))
    ymin = min(all_vals) - 0.12
    ymax = max(all_vals) + 0.12
    ax.set_ylim(ymax, ymin)
    ax.set_xlabel("", color=colors["local_time"])
    ax.set_ylabel("Calibrated Magnitude (mag)", color=colors["text"], fontsize=sizes["ylabel"])
    ax.tick_params(axis="x", colors=colors["local_time"], labelsize=sizes["xtick"], direction="out", length=sizes["tick_main"], pad=0)
    ax.tick_params(axis="y", colors=colors["text"], labelsize=sizes["ytick"])
    ax.xaxis.get_offset_text().set_color(colors["bjd"])
    ax.xaxis.get_offset_text().set_fontsize(6)
    ax.text(0.01, 1.0, date_str, transform=ax.transAxes, ha="left", va="bottom", fontsize=sizes["date"], color=colors["date"])
    if getattr(cfg, "obs_lat_deg", None) is not None and getattr(cfg, "obs_lon_deg", None) is not None:
        ax.text(0.99, 1.0, f"{cfg.obs_lat_deg:.2f}N {cfg.obs_lon_deg:.2f}E", transform=ax.transAxes, ha="right", va="bottom", fontsize=sizes["coord"], color=colors["coord"])

    # Title above frame; keep small and two-line
    fig.text(0.5, 0.985, "Light Curve\nFourier", ha="center", va="top", fontsize=sizes["title"], color=colors["title"], fontweight="bold")

    try:
        from astropy.time import Time as ATime
        import datetime as _dt_local

        _time_format = "jd" if time_key in ("bjd_tdb", "jd") else "mjd"
        bjd_all = df[time_key].values
        bjd_min = float(np.min(bjd_all))
        bjd_max = float(np.max(bjd_all))

        def _to_local(bjd_val: float):
            t_utc = ATime(bjd_val, format=_time_format, scale="tdb").to_datetime()
            return t_utc + _dt_local.timedelta(hours=float(tz_offset_hours))

        _t_local_min = _to_local(bjd_min)
        _t_local_max = _to_local(bjd_max)
        _label_bjd_ticks = []
        _label_local = []
        _cur = _t_local_min.replace(second=0, microsecond=0)
        _cur = _cur.replace(minute=(_cur.minute // 10) * 10)
        while _cur <= _t_local_max + _dt_local.timedelta(minutes=1):
            _utc = _cur - _dt_local.timedelta(hours=float(tz_offset_hours))
            _b_tick = ATime(_utc).jd if _time_format == "jd" else ATime(_utc).mjd
            if _cur.minute in (0, 30):
                _label_bjd_ticks.append(_b_tick)
                _label_local.append(_cur.strftime("%H:%M"))
            _cur += _dt_local.timedelta(minutes=10)

        ax.set_xticks(_label_bjd_ticks)
        if _label_local:
            _label_local[0] = "Local Time (HH:MM)"
        ax.set_xticklabels(_label_local, color=colors["local_time"], fontsize=sizes["xtick"])
        ticks = ax.xaxis.get_major_ticks()
        if ticks:
            ticks[0].tick1line.set_markersize(0)
            ticks[0].tick2line.set_markersize(0)

        bjd_ax = ax.twiny()
        bjd_ax.set_xlim(ax.get_xlim())
        bjd_ax.set_xticks(_label_bjd_ticks)
        bjd_ax.tick_params(axis="x", colors=colors["bjd"], direction="in", length=sizes["tick_bjd"], pad=-14, labelsize=sizes["bjd_label"])
        bjd_ax.set_xticklabels([])
        bjd_ax.xaxis.set_ticks_position("bottom")
        bjd_ax.xaxis.set_label_position("bottom")
        bjd_ax.spines["top"].set_visible(False)
        bjd_ax.spines["bottom"].set_color(colors["text"])
        bjd_ax.spines["left"].set_visible(False)
        bjd_ax.spines["right"].set_visible(False)
        bjd_ax.yaxis.set_visible(False)

        x0, x1 = ax.get_xlim()
        xspan = x1 - x0
        for v in _label_bjd_ticks:
            xf = (v - x0) / xspan
            if 0.0 <= xf <= 1.0:
                ax.text(xf, 0.01125, f"{v:.4f}", transform=ax.transAxes, ha="left", va="bottom", fontsize=sizes["bjd_label"], color=colors["bjd"], alpha=0.6, rotation=45, clip_on=True)
        ax.text(0.015, 0.01125, "BJD", transform=ax.transAxes, ha="left", va="bottom", fontsize=sizes["bjd_label"], color=colors["bjd"])
    except Exception:
        pass

    ax.legend(loc="upper right", frameon=True, facecolor=colors["legend_bg"], edgecolor=colors["text"], fontsize=sizes["legend"], labelcolor=colors["text"])
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.06, top=0.915)
    plt.savefig(out_png, dpi=dpi, facecolor=colors["bg"])
    plt.close(fig)
    return out_png


def _plot_overlay_fourier_plotly(
    csv_by_channel: dict[str, Path],
    out_dir: Path,
    obs_date: str | None,
    out_name: str | None = None,
    accepted_period_days: float | None = None,
    width_px: int = 1800,
    height_px: int = 900,
    tz_offset_hours: float = 8.0,
) -> Path:
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        raise RuntimeError(f"plotly import failed: {exc}") from exc

    colors = {"B": "#9CDDEE", "G1": "#22B14C", "G2": "#9CC51A", "R": "#F6A8C2"}
    fit_colors = {"B": "#A6EDFF", "G1": "#30FA6B", "G2": "#C9FF20", "R": "#E11B23"}
    ref_color = "#FFA500"
    ref_shift_days = 12.0 / 1440.0
    ref_amp_semi = 0.27 / 2.0

    def _to_local_datetime(vals: np.ndarray, time_key: str) -> list:
        arr = np.asarray(vals, dtype=float)
        jd = arr + 2400000.5 if time_key == "mjd" else arr
        ts = pd.to_datetime(jd, unit="D", origin="julian", utc=True)
        return (ts + pd.to_timedelta(float(tz_offset_hours), unit="h")).to_pydatetime().tolist()

    date_str = obs_date or "unknown"
    target_name = _infer_target_name(next(iter(csv_by_channel.values()))) or "Target"
    fig = go.Figure()

    loaded: dict[str, dict] = {}
    all_t: list[float] = []
    r_model_t = None
    r_model_y = None
    r_time_key = None
    channel_order = ["R", "G1", "G2", "B"]

    for ch in channel_order:
        path = csv_by_channel.get(ch)
        if path is None:
            continue

        df = pd.read_csv(path)
        if "ok" in df.columns:
            df = df[df["ok"] == 1]
        time_key = "bjd_tdb" if "bjd_tdb" in df.columns else "jd" if "jd" in df.columns else "mjd"
        mag_col = "m_var" if "m_var" in df.columns else "m_var_norm"
        df = df[pd.notna(df[time_key]) & pd.notna(df[mag_col])].copy()
        if len(df) == 0:
            continue

        all_t.extend([float(v) for v in df[time_key].values])
        mean_mag = float(df[mag_col].mean())

        err_col = None
        if "t_sigma_mag" in df.columns and pd.notna(df["t_sigma_mag"]).any():
            err_col = "t_sigma_mag"
        elif "v_err" in df.columns and pd.notna(df["v_err"]).any():
            err_col = "v_err"

        fit_payload = _load_fourier_fit(path, ch)
        if fit_payload is None:
            continue

        t_dense, mag_dense = _build_fourier_curve(
            fit_payload,
            float(df[time_key].min()),
            float(df[time_key].max()),
        )
        best_p = float(fit_payload["best_period_d"])
        n_h = int(fit_payload["n_harmonics"])

        loaded[ch] = {
            "time_key": time_key,
            "x_data": _to_local_datetime(df[time_key].values, time_key),
            "y_data": (df[mag_col].values - mean_mag),
            "yerr": df[err_col].values if err_col else None,
            "x_fit": _to_local_datetime(t_dense, time_key),
            "y_fit": (mag_dense - mean_mag),
            "best_p": best_p,
            "n_h": n_h,
        }

        if ch == "R":
            r_model_t = np.asarray(t_dense, dtype=float)
            r_model_y = np.asarray(mag_dense - mean_mag, dtype=float)
            r_time_key = time_key

    # Legend row 1: data points
    for ch in channel_order:
        if ch not in loaded:
            continue
        d = loaded[ch]
        fig.add_trace(
            go.Scatter(
                x=d["x_data"],
                y=d["y_data"],
                mode="markers",
                name=f"{ch} 1-sigma",
                marker={"size": 11, "color": colors.get(ch, "#FFFFFF")},
                error_y=(
                    {
                        "type": "data",
                        "array": d["yerr"],
                        "visible": True,
                        "color": colors.get(ch, "#FFFFFF"),
                        "thickness": 1.4,
                        "width": 3,
                    }
                    if d["yerr"] is not None
                    else None
                ),
                legend="legend",
            )
        )

    # Legend row 2: Fourier lines
    for ch in channel_order:
        if ch not in loaded:
            continue
        d = loaded[ch]
        fig.add_trace(
            go.Scatter(
                x=d["x_fit"],
                y=d["y_fit"],
                mode="lines",
                name=f"{ch} Fourier P={d['best_p'] * 24.0:.2f} h n={d['n_h']}",
                line={"width": 2.4, "color": fit_colors.get(ch, "#FFFFFF")},
                legend="legend2",
            )
        )

    # Single global reference line on row 2 end
    if accepted_period_days is not None and all_t:
        t_min = float(min(all_t))
        t_max = float(max(all_t))
        t_ref = np.linspace(t_min, t_max, 3000)

        t_fit = r_model_t if r_model_t is not None and len(r_model_t) > 10 else t_ref
        y_fit = r_model_y if r_model_y is not None and len(r_model_y) > 10 else np.zeros_like(t_fit)
        phases = np.linspace(0.0, 1.0, 6001)
        base = 2.0 * np.pi * ((t_fit - t_fit.min()) / float(accepted_period_days))
        best_phi = 0.0
        best_sse = None
        for phi in phases:
            y_try = ref_amp_semi * np.sin(base + 2.0 * np.pi * phi)
            sse = float(np.sum((y_fit - y_try) ** 2))
            if best_sse is None or sse < best_sse:
                best_sse = sse
                best_phi = float(phi)

        y_ref = ref_amp_semi * np.sin(
            2.0 * np.pi * ((((t_ref - ref_shift_days) - t_ref.min()) / float(accepted_period_days)) + best_phi)
        )
        tk = r_time_key if r_time_key else "bjd_tdb"
        fig.add_trace(
            go.Scatter(
                x=_to_local_datetime(t_ref, tk),
                y=y_ref,
                mode="lines",
                name=f"Ref P={float(accepted_period_days) * 24.0:.2f} h n=1",
                line={"width": 2.6, "color": ref_color, "dash": "dash"},
                legend="legend2",
            )
        )

    fig.update_layout(
        title=f"{target_name} Demeaned Overlay + Fourier",
        template="plotly_dark",
        width=width_px,
        height=height_px,
        paper_bgcolor="#232323",
        plot_bgcolor="#2b2b2b",
        font={"size": 16},
        xaxis_title="Local Time (HH:MM)",
        yaxis_title="Delta Magnitude (mag)",
        margin={"l": 95, "r": 8, "t": 95, "b": 118},
        legend={
            "orientation": "h",
            "traceorder": "normal",
            "x": -0.012,
            "xanchor": "left",
            "y": -0.082,
            "yanchor": "top",
            "entrywidthmode": "pixels",
            "entrywidth": 285,
            "bgcolor": "rgba(35,35,35,0.0)",
        },
        legend2={
            "orientation": "h",
            "traceorder": "normal",
            "x": -0.012,
            "xanchor": "left",
            "y": -0.146,
            "yanchor": "top",
            "entrywidthmode": "pixels",
            "entrywidth": 285,
            "bgcolor": "rgba(35,35,35,0.0)",
        },
    )
    fig.update_yaxes(autorange="reversed", gridcolor="#8A8A8A")
    fig.update_xaxes(gridcolor="#8A8A8A", tickformat="%H:%M")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (out_name or f"light_curve_overlay_{date_str}_fourier_norm.html")
    fig.write_html(out_path, include_plotlyjs="cdn", config={"responsive": True})
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replot light-curve PNGs from photometry CSV files."
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        required=False,
        help="One or more photometry CSV files.",
    )
    parser.add_argument(
        "--csv-dir",
        default=None,
        help="Directory containing photometry CSVs (e.g. ...\\1_photometry).",
    )
    parser.add_argument(
        "--run-root",
        default=None,
        help="Run root directory containing 1_photometry and 3_light_curve.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Override output directory. Default: run_root/3_light_curve if detectable.",
    )
    parser.add_argument(
        "--channel",
        default=None,
        help="Override channel name (e.g. R, G1). If omitted, inferred from filename.",
    )
    parser.add_argument(
        "--obs-date",
        default=None,
        help="Override observation date (YYYYMMDD). If omitted, inferred from filename.",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["R", "G1", "G2", "B"],
        help="Channels to replot when using --csv-dir. Default: R G1 G2 B.",
    )
    parser.add_argument(
        "--sync-g1g2-ylim",
        action="store_true",
        help="Use shared Y limits for G1 and G2 plots.",
    )
    parser.add_argument(
        "--overlay-all",
        action="store_true",
        help="Produce a single overlay plot for all channels.",
    )
    parser.add_argument(
        "--overlay-out-name",
        default=None,
        help="Filename for overlay output PNG (in output dir).",
    )
    parser.add_argument(
        "--overlay-width-cm",
        type=float,
        default=None,
        help="Overlay output width in cm (keeps 3:1 aspect).",
    )
    parser.add_argument(
        "--overlay-jpg",
        action="store_true",
        help="Also save JPG for overlay output.",
    )
    parser.add_argument(
        "--overlay-only",
        action="store_true",
        help="Only output overlay plot; skip per-channel plots.",
    )
    parser.add_argument(
        "--overlay-fourier",
        action="store_true",
        help="Output overlay plot with per-channel Fourier model lines.",
    )
    parser.add_argument(
        "--channel-fourier",
        action="store_true",
        help="Output one white-background Fourier overlay PNG per channel.",
    )
    parser.add_argument(
        "--overlay-fourier-out-name",
        default=None,
        help="Filename for Fourier overlay PNG (in output dir).",
    )
    parser.add_argument(
        "--overlay-fourier-plotly",
        action="store_true",
        help="Output interactive Plotly HTML for demeaned overlay plus Fourier.",
    )
    parser.add_argument(
        "--overlay-fourier-plotly-out-name",
        default=None,
        help="Filename for Fourier Plotly HTML (in output dir).",
    )
    parser.add_argument(
        "--accepted-period-days",
        type=float,
        default=None,
        help="Optional reference period (days) to plot as an extra dashed Fourier line.",
    )
    parser.add_argument(
        "--plotly-width",
        type=int,
        default=1280,
        help="Plotly HTML width in pixels. Default: 1280.",
    )
    parser.add_argument(
        "--plotly-height",
        type=int,
        default=720,
        help="Plotly HTML height in pixels. Default: 720.",
    )
    parser.add_argument(
        "--tz-offset-hours",
        type=float,
        default=8.0,
        help="Local time offset hours for x-axis labeling. Default: 8 (UTC+8).",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    _ensure_pipeline_imports(script_path)

    auto_mode = False
    if not args.csv and not args.csv_dir and not args.run_root:
        auto_mode = True
        project_root = _project_root(script_path)
        output_root = project_root / "output"
        run_root = _find_latest_run_root(output_root)
        if run_root is None:
            print(f"[FAIL] No run root found under: {output_root}")
            return 1
        args.run_root = str(run_root)

    if args.run_root and not args.csv_dir:
        args.csv_dir = str(Path(args.run_root) / "1_photometry")

    out_dir_override = Path(args.out_dir) if args.out_dir else None

    failures = 0

    csv_list: list[Path] = []
    if args.csv:
        csv_list.extend([Path(p).resolve() for p in args.csv])
    if args.csv_dir:
        base = Path(args.csv_dir).resolve()
        if not base.exists():
            print(f"[SKIP] CSV dir not found: {base}")
            return 1
        for ch in args.channels:
            ch_up = str(ch).upper()
            if args.obs_date:
                csv_list.append(base / f"photometry_{ch_up}_{args.obs_date}.csv")
            else:
                # Fall back to any matching file if date not provided
                csv_list.extend(sorted(base.glob(f"photometry_{ch_up}_*.csv")))

    if not csv_list:
        print("[FAIL] No CSV files provided. Use --csv, --csv-dir, or --run-root.")
        return 1

    if auto_mode and not args.obs_date:
        inferred_date = _infer_obs_date_from_csvs(csv_list)
        if inferred_date:
            args.obs_date = inferred_date
    if args.overlay_all and not args.obs_date:
        inferred_date = _infer_obs_date_from_csvs(csv_list)
        if inferred_date:
            args.obs_date = inferred_date

    # Compute shared ylim for G1/G2 to align absolute Y positions
    shared_ylim_g1g2 = None
    g1 = [p for p in csv_list if p.name.upper().startswith("PHOTOMETRY_G1_")]
    g2 = [p for p in csv_list if p.name.upper().startswith("PHOTOMETRY_G2_")]
    if g1 and g2:
        shared_ylim_g1g2 = _compute_shared_ylim(g1 + g2)

    # Overlay-family plots if requested
    if (
        args.overlay_all
        or auto_mode
        or args.overlay_fourier
        or args.channel_fourier
        or args.overlay_fourier_plotly
    ):
        csv_by_channel: dict[str, Path] = {}
        for csv_path in csv_list:
            ch, dt = _infer_channel_and_date(csv_path)
            if not ch:
                continue
            if args.obs_date and dt and dt != args.obs_date:
                continue
            csv_by_channel[ch] = csv_path
        if not csv_by_channel:
            print("[FAIL] No channel CSVs found for overlay.")
            return 1

        accepted_period_days = args.accepted_period_days
        if accepted_period_days is None:
            first_csv = next(iter(csv_by_channel.values()))
            target_name = _infer_target_name(first_csv)
            project_root = _project_root(script_path)
            accepted_period_days, period_source = _load_known_period_days(project_root, target_name)
            if accepted_period_days is not None:
                print(f"[INFO] accepted period auto-loaded: {accepted_period_days:.7f} d ({target_name}) from {period_source}")
            else:
                print(f"[INFO] accepted period not found for target: {target_name}")

        # Use any CSV to resolve output dir
        out_dir = _resolve_output_dir(next(iter(csv_by_channel.values())), out_dir_override)
        if auto_mode:
            date_str = args.obs_date or "unknown"
            try:
                out_png = _plot_overlay(
                    csv_by_channel,
                    out_dir,
                    args.obs_date,
                    args.tz_offset_hours,
                    out_name=f"light_curve_overlay_{date_str}.png",
                    width_px=2400,
                    height_px=800,
                    dpi=360,
                    save_jpg=False,
                )
                print(f"[OK] overlay short -> {out_png}")
            except Exception as exc:
                print(f"[FAIL] overlay short: {exc}")
                failures += 1
            try:
                out_png = _plot_overlay(
                    csv_by_channel,
                    out_dir,
                    args.obs_date,
                    args.tz_offset_hours,
                    out_name=f"light_curve_overlay_{date_str}_h1483.png",
                    width_px=2400,
                    height_px=1483,
                    dpi=360,
                    save_jpg=False,
                )
                print(f"[OK] overlay tall -> {out_png}")
            except Exception as exc:
                print(f"[FAIL] overlay tall: {exc}")
                failures += 1
        else:
            try:
                out_png = _plot_overlay(
                    csv_by_channel,
                    out_dir,
                    args.obs_date,
                    args.tz_offset_hours,
                    out_name=args.overlay_out_name,
                    width_cm=args.overlay_width_cm,
                    save_jpg=args.overlay_jpg,
                )
                print(f"[OK] overlay -> {out_png}")
            except Exception as exc:
                print(f"[FAIL] overlay: {exc}")
                failures += 1

        if args.overlay_fourier:
            try:
                out_png = _plot_overlay_with_fourier(
                    csv_by_channel,
                    out_dir,
                    args.obs_date,
                    args.tz_offset_hours,
                    out_name=args.overlay_fourier_out_name,
                    width_px=2400,
                    height_px=800,
                    dpi=360,
                )
                print(f"[OK] overlay fourier -> {out_png}")
            except Exception as exc:
                print(f"[FAIL] overlay fourier: {exc}")
                failures += 1

        if args.channel_fourier:
            for ch, csv_path in csv_by_channel.items():
                ch_out_dir = _resolve_output_dir(csv_path, None)
                try:
                    out_png = _plot_channel_with_fourier(
                        csv_path,
                        ch_out_dir,
                        ch,
                        args.obs_date,
                        args.tz_offset_hours,
                        out_name=f"light_curve_{ch}_{args.obs_date or 'unknown'}_fourier.png",
                        accepted_period_days=accepted_period_days,
                        width_px=2400,
                        height_px=800,
                        dpi=360,
                    )
                    print(f"[OK] channel fourier {ch} -> {out_png}")
                except Exception as exc:
                    print(f"[FAIL] channel fourier {ch}: {exc}")
                    failures += 1

        if args.overlay_fourier_plotly:
            try:
                out_html = _plot_overlay_fourier_plotly(
                    csv_by_channel,
                    out_dir,
                    args.obs_date,
                    out_name=args.overlay_fourier_plotly_out_name,
                    accepted_period_days=accepted_period_days,
                    width_px=args.plotly_width,
                    height_px=args.plotly_height,
                    tz_offset_hours=args.tz_offset_hours,
                )
                print(f"[OK] overlay fourier plotly -> {out_html}")
            except Exception as exc:
                print(f"[FAIL] overlay fourier plotly: {exc}")
                failures += 1

    if not args.overlay_only or auto_mode:
        for csv_path in csv_list:
            csv_path = Path(csv_path).resolve()
            if not csv_path.exists():
                print(f"[SKIP] Missing CSV: {csv_path}")
                failures += 1
                continue

            inferred_ch, inferred_date = _infer_channel_and_date(csv_path)
            channel = (args.channel or inferred_ch)
            obs_date = (args.obs_date or inferred_date)

            if channel is None or obs_date is None:
                print(f"[WARN] Could not infer channel/date from {csv_path.name}. "
                      f"Use --channel and --obs-date to override.")

            out_dir = _resolve_output_dir(csv_path, out_dir_override)
            ylim = None
            if shared_ylim_g1g2 and channel in ("G1", "G2"):
                ylim = shared_ylim_g1g2
            try:
                out_png = _plot_one(csv_path, out_dir, channel, obs_date, args.tz_offset_hours, ylim=ylim)
                print(f"[OK] {csv_path.name} -> {out_png}")
            except Exception as exc:
                print(f"[FAIL] {csv_path.name}: {exc}")
                failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
