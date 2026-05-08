# -*- coding: utf-8 -*-
"""Diagnostic plot writers for active photometry products."""
from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _channel_date_from_csv(out_csv: Path, channel: str) -> tuple[str, str]:
    parts = out_csv.stem.split("_")
    if len(parts) >= 3 and parts[0] == "photometry":
        return parts[1], parts[2]
    return channel, "unknown"


def _finite_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _flag_summary_text(df: pd.DataFrame) -> str:
    if len(df) == 0:
        return "No frame rows were produced."

    ok_count = int((df.get("ok", pd.Series(dtype=int)) == 1).sum()) if "ok" in df.columns else 0
    total = int(len(df))
    lines = [f"Frames ok: {ok_count}/{total}"]

    if "ok_flag" in df.columns:
        flags = df.loc[df.get("ok", pd.Series(0, index=df.index)) != 1, "ok_flag"]
        flags = flags.fillna("unclassified").replace("", "unclassified")
        for flag, count in flags.value_counts().sort_index().items():
            lines.append(f"{flag}: {int(count)}")

    if "comp_available" in df.columns and df["comp_available"].notna().any():
        comp_vals = _finite_array(df["comp_available"].values)
        if comp_vals.size:
            lines.append(
                "comparison stars per frame: "
                f"min={int(np.nanmin(comp_vals))}, "
                f"median={float(np.nanmedian(comp_vals)):.1f}, "
                f"max={int(np.nanmax(comp_vals))}"
            )

    return "\n".join(lines)


def save_regression_overview(
    *,
    df: pd.DataFrame,
    out_csv: Path,
    channel: str,
    cfg,
    time_key: str,
    first_frame_diag_data,
) -> Path:
    """Save the mandatory per-channel regression overview plot."""
    diag_dir = Path(cfg.regression_diag_dir)
    diag_dir.mkdir(parents=True, exist_ok=True)
    ch, date_str = _channel_date_from_csv(Path(out_csv), channel)
    out_path = diag_dir / f"reg_overview_{ch}_{date_str}.png"

    fig = plt.figure(figsize=(8, 11))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.30)
    ax_sc = fig.add_subplot(gs[0])
    ax_sc.set_box_aspect(1.0)
    ax_ts = fig.add_subplot(gs[1])

    _draw_first_frame_regression(ax_sc, first_frame_diag_data, cfg)
    _draw_residual_timeseries(ax_ts, df, time_key)

    fig.suptitle(
        f"Regression Diagnostic | {getattr(cfg, 'target_name', '')} "
        f"channel={channel} {Path(out_csv).stem}",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[reg_diag] saved -> {out_path}")
    return out_path


def _draw_first_frame_regression(ax, first_frame_diag_data, cfg) -> None:
    if first_frame_diag_data is None:
        ax.text(
            0.5,
            0.56,
            "No regression frame data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.text(
            0.5,
            0.46,
            "No frame reached the regression step.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )
        ax.set_axis_off()
        return

    comp_m_cat, comp_m_inst, fit = first_frame_diag_data[:3]
    target_catalog_mag = first_frame_diag_data[3] if len(first_frame_diag_data) > 3 else np.nan
    target_m_inst = first_frame_diag_data[4] if len(first_frame_diag_data) > 4 else np.nan

    comp_m_cat = np.asarray(comp_m_cat, dtype=float)
    comp_m_inst = np.asarray(comp_m_inst, dtype=float)
    ok_pts = np.isfinite(comp_m_cat) & np.isfinite(comp_m_inst)

    if ok_pts.any():
        ax.scatter(
            comp_m_cat[ok_pts],
            comp_m_inst[ok_pts],
            s=18,
            alpha=0.65,
            color="steelblue",
            label=f"comp (n={int(ok_pts.sum())})",
        )
    else:
        ax.text(
            0.5,
            0.55,
            "No finite comparison-star points",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )

    fit_ok = bool(
        fit
        and np.isfinite(fit.get("a", np.nan))
        and np.isfinite(fit.get("b", np.nan))
    )
    if fit_ok and ok_pts.any():
        a = float(fit["a"])
        b = float(fit["b"])
        r2 = float(fit.get("r2", np.nan))
        x_line = np.linspace(float(comp_m_cat[ok_pts].min()), float(comp_m_cat[ok_pts].max()), 100)
        ax.plot(
            x_line,
            a * x_line + b,
            color="red",
            lw=1.8,
            label=f"m_inst={a:.3f} m_cat+({b:.3f})  R2={r2:.3f}",
        )
        if np.isfinite(target_m_inst) and a != 0:
            target_m_var = (float(target_m_inst) - b) / a
            ax.scatter(
                [target_m_var],
                [target_m_inst],
                s=100,
                marker="o",
                facecolors="none",
                edgecolors="red",
                linewidths=2.0,
                zorder=5,
                label=f"target m_var={target_m_var:.2f}",
            )
    else:
        ax.text(
            0.5,
            0.08,
            "No valid regression fit for this channel.",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="darkred",
        )
        if ok_pts.any() and np.isfinite(target_catalog_mag) and np.isfinite(target_m_inst):
            ax.scatter(
                [float(target_catalog_mag)],
                [float(target_m_inst)],
                s=100,
                marker="o",
                facecolors="none",
                edgecolors="red",
                linewidths=2.0,
                zorder=5,
                label="target approx mag",
            )

    _apply_regression_axes(ax, comp_m_cat, comp_m_inst, target_catalog_mag, target_m_inst, fit, cfg)


def _apply_regression_axes(ax, comp_m_cat, comp_m_inst, target_catalog_mag, target_m_inst, fit, cfg) -> None:
    x_vals = _finite_array(comp_m_cat)
    y_vals = _finite_array(comp_m_inst)

    if np.isfinite(target_catalog_mag):
        x_vals = np.append(x_vals, float(target_catalog_mag))
    if np.isfinite(target_m_inst):
        y_vals = np.append(y_vals, float(target_m_inst))

    fit_ok = bool(
        fit
        and np.isfinite(fit.get("a", np.nan))
        and np.isfinite(fit.get("b", np.nan))
        and x_vals.size > 0
    )
    if fit_ok:
        a = float(fit["a"])
        b = float(fit["b"])
        y_vals = np.append(y_vals, a * np.array([float(np.nanmin(x_vals)), float(np.nanmax(x_vals))]) + b)

    if x_vals.size > 0:
        x_min = float(np.nanmin(x_vals))
        x_max = float(np.nanmax(x_vals))
        pad = max((x_max - x_min) * 0.08, 0.1)
        ax.set_xlim(x_min - pad, x_max + pad)
    if y_vals.size > 0:
        y_min = float(np.nanmin(y_vals))
        y_max = float(np.nanmax(y_vals))
        pad = max((y_max - y_min) * 0.12, 0.1)
        ax.set_ylim(y_max + pad, y_min - pad)
    else:
        ax.invert_yaxis()

    ax.set_xlabel(f"m_cat ({getattr(cfg, 'phot_band', '')})")
    ax.set_ylabel("m_inst")
    ax.set_title("Regression Fit - first usable frame", fontsize=12)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, frameon=False, loc="best")


def _draw_residual_timeseries(ax, df: pd.DataFrame, time_key: str) -> None:
    if (
        time_key in df.columns
        and "reg_residual_rms" in df.columns
        and np.isfinite(df["reg_residual_rms"]).any()
    ):
        data = df[
            np.isfinite(df[time_key])
            & np.isfinite(df["reg_residual_rms"])
            & (df.get("ok", pd.Series(1, index=df.index)) == 1)
        ].copy()
        if len(data) > 0:
            ax.plot(
                data[time_key],
                data["reg_residual_rms"],
                "o-",
                ms=2,
                lw=0.6,
                color="steelblue",
                alpha=0.75,
            )
            median = float(np.nanmedian(data["reg_residual_rms"]))
            ax.axhline(median, color="red", lw=1, ls="--", label=f"median={median:.4f}")
            ax.set_xlabel(time_key.upper(), fontsize=9)
            ax.set_ylabel("Residual RMS (mag)", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
            return

    ax.text(
        0.02,
        0.88,
        "No residual RMS time-series data",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    ax.text(
        0.02,
        0.70,
        _flag_summary_text(df),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
    )
    ax.set_axis_off()
