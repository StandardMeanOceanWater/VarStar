# -*- coding: utf-8 -*-
"""
phot_diagnostics.py
Diagnostics plots for photometry.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save_regression_diagnostic(
    frame_name: str,
    aavso_matched: "pd.DataFrame | None",
    apass_matched: "pd.DataFrame | None",
    fit_aavso: "dict | None",
    fit_apass: "dict | None",
    active_source: str,
    diag_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    x_all = []
    for label, df_m, fit, color, marker in [
        ("AAVSO", aavso_matched, fit_aavso, "red", "o"),
        ("APASS", apass_matched, fit_apass, "steelblue", "s"),
    ]:
        if df_m is None or len(df_m) == 0:
            continue
        m_cat = df_m["m_cat"].values
        m_inst = df_m["m_inst_matched"].values if "m_inst_matched" in df_m.columns else np.full(len(df_m), np.nan)
        ok = np.isfinite(m_cat) & np.isfinite(m_inst)
        if not ok.any():
            continue
        ax.scatter(m_cat[ok], m_inst[ok], s=18, alpha=0.7,
                   color=color, marker=marker,
                   label=f"{label} (n={ok.sum()})")
        x_all.extend(m_cat[ok].tolist())

        if fit is not None and np.isfinite(fit.get("a", np.nan)):
            x_line = np.linspace(m_cat[ok].min(), m_cat[ok].max(), 100)
            y_line = fit["a"] * x_line + fit["b"]
            r2_str = f"R簡={fit['r2']:.3f}" if np.isfinite(fit.get("r2", np.nan)) else ""
            ax.plot(x_line, y_line, color=color, lw=1.5,
                    label=f"{label} fit  a={fit['a']:.3f}  b={fit['b']:.3f}  {r2_str}")

    if x_all:
        x_range = np.array([min(x_all), max(x_all)])
        ax.plot(x_range, x_range, "k--", lw=0.8, alpha=0.4, label="ideal (slope=1)")

    ax.invert_yaxis()
    ax.set_xlabel("Catalogue magnitude  $m_{cat}$")
    ax.set_ylabel("Instrumental magnitude  $m_{inst}$")
    ax.set_title(
        f"Regression Fit Diagnostic  [{frame_name}]\n"
        f"Active source: {active_source}",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = diag_dir / f"reg_diag_{Path(frame_name).stem}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_light_curve(
    df: pd.DataFrame,
    out_png: Path,
    channel: str,
    cfg,
    obs_date: "str | None" = None,
    ylim: "tuple[float, float] | None" = None,
):
    from astropy.time import Time as ATime
    import matplotlib.ticker as mticker
    import datetime as _dt_local

    time_key = "bjd_tdb" if "bjd_tdb" in df.columns else "jd"
    if time_key not in df.columns:
        print(f"[PLOT] Error: time column '{time_key}' not found.")
        return

    _mag_col = "m_var"
    d = df[(df["ok"] == 1) & np.isfinite(df[time_key]) & np.isfinite(df[_mag_col])].copy()
    if len(d) == 0:
        print("[PLOT] No valid photometry points to plot.")
        return

    bjd_arr = d[time_key].values
    bjd_min = float(bjd_arr.min())
    bjd_max = float(bjd_arr.max())

    _tz_offset_h = float(getattr(cfg, "tz_offset_hours", 8))

    def _bjd_to_local_hm(bjd_val):
        try:
            t_utc = ATime(bjd_val, format="jd", scale="tdb").to_datetime()
            return t_utc + _dt_local.timedelta(hours=_tz_offset_h)
        except Exception:
            return None

    _t_local_min = _bjd_to_local_hm(bjd_min)
    _t_local_max = _bjd_to_local_hm(bjd_max)
    _label30_bjd_ticks = []
    _label30_labels = []
    _minor_bjd_ticks = []
    if _t_local_min is not None and _t_local_max is not None:
        _cur = _t_local_min.replace(second=0, microsecond=0)
        _cur = _cur.replace(minute=(_cur.minute // 10) * 10)
        while _cur <= _t_local_max + _dt_local.timedelta(minutes=1):
            _utc = _cur - _dt_local.timedelta(hours=_tz_offset_h)
            _b_tick = ATime(_utc).jd
            _minor_bjd_ticks.append(_b_tick)
            if _cur.minute in (0, 30):
                _label30_bjd_ticks.append(_b_tick)
                _label30_labels.append(_cur.strftime("%H:%M"))
            _cur += _dt_local.timedelta(minutes=10)

    fig, ax = plt.subplots(figsize=(12, 4))

    _y_vals = d[_mag_col].values
    if "t_sigma_mag" in d.columns and np.isfinite(d["t_sigma_mag"]).any():
        ax.errorbar(bjd_arr, _y_vals, yerr=d["t_sigma_mag"].values,
                    fmt="o", ms=4, capsize=2, lw=0.8, label="簣 ?", zorder=3)
    else:
        ax.plot(bjd_arr, _y_vals, "o-", ms=4, lw=0.8, label="簣 ?", zorder=3)

    ax.invert_yaxis()
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
        _y_lo = min(ylim)
        _y_hi = max(ylim)
        _yticks = np.arange(np.ceil(_y_lo * 20) / 20, _y_hi + 0.001, 0.05)
        ax.set_yticks(_yticks)
    ax.set_xlim(bjd_min - 0.002, bjd_max + 0.002)

    if _minor_bjd_ticks:
        ax.set_xticks(_minor_bjd_ticks, minor=True)
    if _label30_bjd_ticks:
        ax.set_xticks(_label30_bjd_ticks)
        ax.set_xticklabels(_label30_labels, color="navy", fontsize=9)
    ax.tick_params(axis="x", which="major", direction="out", colors="navy", length=6)
    ax.tick_params(axis="x", which="minor", direction="out", colors="navy", length=3)

    _xlim = ax.get_xlim()
    _xspan = _xlim[1] - _xlim[0]
    for _i, _tick_v in enumerate(_label30_bjd_ticks):
        if _i == 0:
            continue
        ax.text(
            _tick_v, ax.get_ylim()[0] + 0.02,
            f"{_tick_v:.3f}",
            fontsize=7, color="navy", ha="center", va="bottom",
        )

    _obs_str = f"{channel}  {obs_date}" if obs_date else f"{channel}"
    ax.set_title(_obs_str, fontsize=11, loc="left")

    ax.set_xlabel("Local Time (UTC+8)")
    ax.set_ylabel("Magnitude (V)")

    ax.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close("all")
    print(f"[PNG] saved ??{out_png}")
