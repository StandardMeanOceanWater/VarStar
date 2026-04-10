"""
Replot light-curve PNGs from existing photometry CSV files.

Default output: sibling run root's 3_light_curve directory, e.g.
E:\\VarStar\\output\\YYYY-MM-DD\\...\\<run_ts>\\3_light_curve\\

This script does not modify CSVs. It only reads them and writes PNGs.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import matplotlib.pyplot as plt


def _ensure_pipeline_imports(script_path: Path) -> None:
    pipeline_dir = script_path.parent.parent
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


def _resolve_output_dir(csv_path: Path, explicit_out_dir: Path | None) -> Path:
    if explicit_out_dir is not None:
        return explicit_out_dir
    run_root = _infer_run_root(csv_path)
    if run_root is not None:
        return run_root / "3_light_curve"
    return csv_path.parent


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
    from photometry import plot_light_curve  # deferred import after sys.path setup
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
        project_root = script_path.parent.parent.parent
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

    # Overlay plot (single image) if requested
    if args.overlay_all or auto_mode:
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
