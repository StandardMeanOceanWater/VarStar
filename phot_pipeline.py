# -*- coding: utf-8 -*-
"""
phot_pipeline.py
Main photometry pipeline execution.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.time import Time as ATime
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
from photutils.detection import IRAFStarFinder

from phot_core import (
    m_inst_from_flux,
    max_pixel_in_box,
    compute_annulus_radii,
    aperture_photometry,
    is_saturated,
    radec_to_pixel,
    in_bounds,
)
from phot_regression import robust_linear_fit, mag_error_from_flux, differential_mag
from phot_timing import apply_gain_from_header, require_cfg_values, time_from_header
from phot_ensemble import ensemble_normalize
from phot_diagnostics import _save_regression_diagnostic, plot_light_curve

_phot_logger = logging.getLogger("photometry")

class _FrameCompCache:
    """同視野多目標共用的比較星測光快取。
    第一顆目標測完後存入，後續目標直接讀取，避免重複孔徑測光。
    Key: (frame_stem, ra_rounded_5dp, dec_rounded_5dp)
    """
    def __init__(self):
        self._data: dict = {}

    def _key(self, frame_stem: str, ra: float, dec: float) -> tuple:
        return (frame_stem, round(float(ra), 5), round(float(dec), 5))

    def get(self, frame_stem: str, ra: float, dec: float):
        return self._data.get(self._key(frame_stem, ra, dec))

    def set(self, frame_stem: str, ra: float, dec: float, result: dict):
        self._data[self._key(frame_stem, ra, dec)] = result

    def __len__(self):
        return len(self._data)



def plot_light_curve(
    df: pd.DataFrame,
    out_png: Path,
    channel: str,
    cfg,
    obs_date: "str | None" = None,
    ylim: "tuple[float, float] | None" = None,
):
    """
    Generate the light curve plot from a photometry result DataFrame.

    ylim : (ymin, ymax) in mag — 若提供則鎖定 Y 軸範圍（已含 invert）。
    """
    from astropy.time import Time as ATime
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import datetime as _dt_local

    time_key = "bjd_tdb" if "bjd_tdb" in df.columns else "jd"
    if time_key not in df.columns:
        print(f"[PLOT] Error: time column '{time_key}' not found.")
        return

    # Filter valid points
    # ensemble_normalize 啟用時優先用 m_var_norm；否則退回 m_var
    _mag_col = "m_var"  # v1.5 邏輯：直接用回歸校正星等；ensemble normalization 已知有問題
    d = df[(df["ok"] == 1) & np.isfinite(df[time_key]) & np.isfinite(df[_mag_col])].copy()
    if len(d) == 0:
        print("[PLOT] No valid photometry points to plot.")
        return

    bjd_arr = d[time_key].values
    bjd_min = float(bjd_arr.min())
    bjd_max = float(bjd_arr.max())

    # Time scaling for Local Time (UTC+8 default)
    _tz_offset_h = float(getattr(cfg, "tz_offset_hours", 8))

    def _bjd_to_local_hm(bjd_val):
        try:
            t_utc = ATime(bjd_val, format="jd", scale="tdb").to_datetime()
            return t_utc + _dt_local.timedelta(hours=_tz_offset_h)
        except Exception:
            return None

    # Compute Ticks
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

    # Error bars or dots
    _y_vals = d[_mag_col].values
    if "t_sigma_mag" in d.columns and np.isfinite(d["t_sigma_mag"]).any():
        ax.errorbar(bjd_arr, _y_vals, yerr=d["t_sigma_mag"].values,
                    fmt="o", ms=4, capsize=2, lw=0.8, label="± σ", zorder=3)
    else:
        ax.plot(bjd_arr, _y_vals, "o-", ms=4, lw=0.8, label="± σ", zorder=3)

    ax.invert_yaxis()
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])   # ylim 應已考慮 invert（大值在下）
        # 強制統一 Y 軸 tick 間距（0.05 mag），避免 matplotlib 自動選不同間距
        _y_lo = min(ylim)   # 亮端（小數值）
        _y_hi = max(ylim)   # 暗端（大數值）
        _yticks = np.arange(np.ceil(_y_lo * 20) / 20, _y_hi + 0.001, 0.05)
        ax.set_yticks(_yticks)
    ax.set_xlim(bjd_min - 0.002, bjd_max + 0.002)

    # Ticks & Label (Navy)
    if _minor_bjd_ticks:
        ax.set_xticks(_minor_bjd_ticks, minor=True)
    if _label30_bjd_ticks:
        ax.set_xticks(_label30_bjd_ticks)
        ax.set_xticklabels(_label30_labels, color="navy", fontsize=9)
    ax.tick_params(axis="x", which="major", direction="out", colors="navy", length=6)
    ax.tick_params(axis="x", which="minor", direction="out", colors="navy", length=3)

    # BJD Text (Navy Alpha)
    _xlim = ax.get_xlim()
    _xspan = _xlim[1] - _xlim[0]
    for _i, _tick_v in enumerate(_label30_bjd_ticks):
        if _i == 0: continue
        _xf = (_tick_v - _xlim[0]) / _xspan
        if 0.0 <= _xf <= 1.0:
            ax.text(_xf, 0.01, f"{_tick_v:.4f}", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=7, color="navy", alpha=0.4,
                    rotation=45, clip_on=True, zorder=5)

    ax.text(0.0, 0.01, "BJD", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, color="navy", zorder=5)

    # Labels
    ax.set_xlabel("Local Time (HH:MM)", color="navy", loc="left", fontsize=9, labelpad=2)
    ax.set_ylabel("Calibrated Magnitude (mag)")

    # Title: Bold, Large (22), pad (10)
    _title_star = getattr(cfg, "display_name", None) or getattr(cfg, "target_name", "Target")
    ax.set_title(f"{_title_star} Light Curve [{channel}]", fontsize=22, fontweight='bold', pad=10)

    # Legend & Date
    _fs = 16
    _obs_str = obs_date if obs_date else ""
    ax.text(0.01, 1.02, _obs_str, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=_fs, color="navy")

    # Legend using V6-style compact parameters
    ax.legend(fontsize=_fs, loc="upper left", frameon=True, edgecolor='gray',
              borderaxespad=0.2, handletextpad=0.0, handlelength=1.0, borderpad=0.2, labelspacing=0.2)

    # Coordinates
    _lat = getattr(cfg, "obs_lat_deg", None)
    _lon = getattr(cfg, "obs_lon_deg", None)
    if _lat is not None and _lon is not None:
        _ls = f"{abs(_lat):.2f}°{'N' if _lat >= 0 else 'S'}"
        _rs = f"{abs(_lon):.2f}°{'E' if _lon >= 0 else 'W'}"
        ax.text(0.99, 1.02, f"{_ls} {_rs}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=_fs, color="#2d6a4f")

    ax.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close("all")
    print(f"[PNG] saved → {out_png}")


def run_photometry_on_wcs_dir(cfg, 
    wcs_dir: Path,
    out_csv: Path,
    out_png: Path,
    comp_refs: list,
    check_star=None,
    ap_radius: "float | None" = None,
    channel: str = "B",
    shared_cache: "_FrameCompCache | None" = None,
) -> "tuple[pd.DataFrame, dict[str, pd.Series]]":
    """
    Per-frame aperture differential photometry.

    Time system : BJD_TDB (Eastman et al., 2010) at exposure midpoint.
    Zero point  : robust iterative linear regression (see robust_linear_fit).
    Airmass     : Young (1994); frames with X > 2.0 are flagged but kept.

    Returns
    -------
    df               : 每幀測光結果 DataFrame（含 m_var、m_var_norm 等欄位）。
    comp_lightcurves : {star_id: Series(bjd_tdb → m_inst)}，每顆比較星的
                       儀器星等時間序列。僅在 cfg.ensemble_normalize=True 且
                       比較星數 ≥ cfg.ensemble_min_comp 時填充；否則回傳空 dict。
                       star_id 格式："{ra_deg:.6f}_{dec_deg:.6f}"。
    """
    ra_t, dec_t = cfg.target_radec_deg
    if ap_radius is None:
        ap_radius = cfg.aperture_radius
    r_in, r_out = compute_annulus_radii(ap_radius, cfg.annulus_r_in, cfg.annulus_r_out)
    margin = int(np.ceil(r_out + 2))

    # ── ⏸ Ensemble 正規化設定（已停用，保留供未來重啟）────────────────────────────
    # 原本邏輯：_ensemble_on=True 時調用 ensemble_normalize() 迭代計算 Δ(t)。
    # 現況：簡化版本已跳過呼叫，直接使用回歸結果（見下文 3659-3700 之簡化版）。
    _ensemble_on = bool(getattr(cfg, "ensemble_normalize", False))
    _ensemble_min_comp = int(getattr(cfg, "ensemble_min_comp", 3))
    _ensemble_max_iter = int(getattr(cfg, "ensemble_max_iter", 10))
    _ensemble_tol = float(getattr(cfg, "ensemble_convergence_tol", 1e-4))

    # split/{channel}/ 的 FITS 命名規則：*_{channel}.fits
    wcs_files_sorted = sorted(wcs_dir.glob(f"*_{channel}.fits"))
    if not wcs_files_sorted:
        raise FileNotFoundError(
            f"No split/{channel} FITS found in: {wcs_dir}\n"
            "Check that debayer step completed successfully."
        )

    check_coord  = (SkyCoord(ra=check_star[0] * u.deg, dec=check_star[1] * u.deg)
                    if check_star is not None else None)
    target_coord = SkyCoord(ra=ra_t * u.deg, dec=dec_t * u.deg)
    cfg_checked      = False
    n_skipped        = 0
    n_high_fwhm      = 0
    n_low_sharpness  = 0
    n_low_peak_ratio = 0
    n_low_reg_r2      = 0
    rows = []
    _first_frame_diag_data = None   # (comp_m_cat, comp_m_inst, fit) for diag plot

    # ── comp_lightcurves 累積結構：{star_id: {t_val: m_inst}} ────────────────
    # star_id = "{ra_deg:.6f}_{dec_deg:.6f}"
    # 每幀測光後，把該幀所有有效比較星的 (t, m_inst) 追加進去。
    # 幀迴圈結束後轉為 pd.Series 供 ensemble_normalize 使用。
    _comp_lc_buf: "dict[str, dict]" = {}
    # 對應 star_id → 初始權重（來自幀迴圈中計算的距離＋誤差複合權重）
    # 只取第一次出現的值（各幀理論上相同，因為距離和 m_err 不隨時間變化）
    _comp_init_weights: "dict[str, float]" = {}

    for f in wcs_files_sorted:
        with fits.open(f) as hdul:
            img = hdul[0].data.astype(np.float32)
            hdr = hdul[0].header
            wcs_obj = WCS(hdr)

        apply_gain_from_header(hdr, cfg)
        if not cfg_checked:
            require_cfg_values(cfg)
            cfg_checked = True

        # ── Time, BJD_TDB, airmass ───────────────────────────────────────────
        mjd, bjd_tdb, airmass = time_from_header(hdr, ra_t, dec_t, cfg)

        xt, yt = radec_to_pixel(wcs_obj, ra_t, dec_t)
        rec = {
            "file": f.name,
            "mjd": mjd,
            "bjd_tdb": bjd_tdb,
            "airmass": airmass,
            "xt": xt, "yt": yt,
            "ok": 0,
            "m_var": np.nan,
            "frame_fwhm_median": np.nan,
            "sharpness_index": np.nan,
        }

        # ── 高度角截斷（altitude < ALT_MIN_DEG の幀は除外）────────────────────
        # 大氣消光在低高度角時急增，ALT_MIN_DEG=45° 對應 airmass≈1.41。
        # airmass 已知（非 NaN）時才做截斷；location 未設定時不截斷（airmass=NaN）。
        if np.isfinite(airmass) and airmass > cfg.alt_min_airmass:
            n_skipped += 1
            rec["ok_flag"] = "high_airmass"
            rows.append(rec)
            continue

        # ── [1] 幀層級 FWHM 篩選 ─────────────────────────────────────────────
        # 用 IRAFStarFinder 估算全幀中位數 FWHM，剔除模糊/拖影幀。
        # 注意：DAOStarFinder 不輸出 fwhm 欄，改用 IRAFStarFinder（有 fwhm 欄）。
        _frame_fwhm_median = np.nan
        try:
            _f_mean, _f_med, _f_std = sigma_clipped_stats(img, sigma=3.0, maxiters=3)
            if _f_std > 0:
                _iraf_fwhm = IRAFStarFinder(fwhm=4.0, threshold=5.0 * _f_std)
                _tbl_fwhm = _iraf_fwhm(img - _f_med)
                if _tbl_fwhm is not None and len(_tbl_fwhm) > 0:
                    _frame_fwhm_median = float(np.median(_tbl_fwhm["fwhm"]))
        except Exception:
            pass
        rec["frame_fwhm_median"] = _frame_fwhm_median
        if (np.isfinite(_frame_fwhm_median)
                and _frame_fwhm_median > cfg.comp_fwhm_max):
            rec["ok"] = 0
            rec["ok_flag"] = "high_fwhm"
            n_high_fwhm += 1
            rows.append(rec)
            continue

        if not in_bounds(img, xt, yt, margin=margin):
            rows.append(rec)
            continue

        phot_t = aperture_photometry(img, xt, yt, ap_radius, r_in, r_out)
        if phot_t.get("ok") != 1 or not np.isfinite(phot_t.get("flux_net")):
            rows.append(rec)
            continue

        sat_t = is_saturated(phot_t.get("max_pix", np.nan), cfg.saturation_threshold)
        if sat_t and not cfg.allow_saturated_target:
            rows.append(rec)
            continue

        m_inst_t = m_inst_from_flux(phot_t["flux_net"])
        if not np.isfinite(m_inst_t):
            rows.append(rec)
            continue

        # ── [2] Sharpness Index 篩選 ─────────────────────────────────────────
        # S = flux(r=3px) / flux(r=8px)（均扣背景，對象：最亮未飽和比較星）
        # S < sharpness_min 代表星點過度擴散或拖影，剔除該幀。
        # 注意：刻意用最亮比較星而非目標星 — 亮星 S/N 高，sharpness 判斷更穩定。
        _sharpness = np.nan
        _sharpness_min = float(getattr(cfg, "sharpness_min", 0.3))
        if _sharpness_min > 0 and comp_refs:
            # 找視場內最亮（m_cat 最小）且未飽和的比較星
            _s_bright_x, _s_bright_y = None, None
            _s_bright_mag = np.inf
            for _sref in comp_refs:
                _s_ra, _s_dec, _s_m = float(_sref[0]), float(_sref[1]), float(_sref[2])
                if not np.isfinite(_s_m) or _s_m >= _s_bright_mag:
                    continue
                _s_xc, _s_yc = radec_to_pixel(wcs_obj, _s_ra, _s_dec)
                if not in_bounds(img, _s_xc, _s_yc, margin=margin):
                    continue
                _s_phot8 = aperture_photometry(img, _s_xc, _s_yc, 8.0, r_in, r_out)
                if is_saturated(_s_phot8.get("max_pix", np.nan), cfg.saturation_threshold):
                    continue
                _s_bright_x, _s_bright_y = _s_xc, _s_yc
                _s_bright_mag = _s_m
            if _s_bright_x is not None:
                try:
                    _phot_s3 = aperture_photometry(img, _s_bright_x, _s_bright_y, 3.0, r_in, r_out)
                    _phot_s8 = aperture_photometry(img, _s_bright_x, _s_bright_y, 8.0, r_in, r_out)
                    if (_phot_s3.get("ok") == 1 and _phot_s8.get("ok") == 1
                            and np.isfinite(_phot_s3.get("flux_net", np.nan))
                            and np.isfinite(_phot_s8.get("flux_net", np.nan))
                            and _phot_s8["flux_net"] > 0):
                        _sharpness = float(_phot_s3["flux_net"] / _phot_s8["flux_net"])
                except Exception:
                    pass
        rec["sharpness_index"] = _sharpness
        if np.isfinite(_sharpness) and _sharpness < _sharpness_min:
            rec["ok"] = 0
            rec["ok_flag"] = "low_sharpness"
            n_low_sharpness += 1
            rows.append(rec)
            continue

        # ── [2b] Peak-ratio 篩選（次鏡起霧 / 甜甜圈 PSF 偵測）─────────────────
        # peak_ratio = t_max_pix / t_flux_net
        # 次鏡起霧時中心被掏空，峰值相對總通量驟降。
        # [DEPRECATED] peak_ratio_min: 固定門檻，已由自適應 peak_ratio_k 取代。
        # peak_ratio_min > 0 仍可運作但不建議使用。
        _peak_ratio_min = float(getattr(cfg, "peak_ratio_min", 0.0))
        # 計算 peak_ratio（重用已有的 phot_t，不重複做 aperture_photometry）
        _flux_pr = phot_t.get("flux_net", np.nan)
        _peak_pr = phot_t.get("max_pix",  np.nan)
        _peak_ratio = float(_peak_pr / _flux_pr) if (
            np.isfinite(_flux_pr) and np.isfinite(_peak_pr) and _flux_pr > 0
        ) else np.nan
        rec["peak_ratio"] = _peak_ratio
        if _peak_ratio_min > 0 and np.isfinite(_peak_ratio) and _peak_ratio < _peak_ratio_min:
            rec["ok"] = 0
            rec["ok_flag"] = "low_peak_ratio"
            n_low_peak_ratio += 1
            rows.append(rec)
            continue

        # ── Comparison ensemble ──────────────────────────────────────────────
        comp_m_inst, comp_m_cat, comp_weights = [], [], []
        for ref in comp_refs:
            ra_c, dec_c, m_cat = ref[0], ref[1], ref[2]
            m_err   = ref[3] if len(ref) > 3 else None
            # weight = 1 / (d + ε)²，由 auto_select_comps 計算並存入 ref[4]
            w_i     = float(ref[4]) if len(ref) > 4 else 1.0
            sc = SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg)
            if check_coord is not None and sc.separation(check_coord).arcsec <= cfg.apass_match_arcsec:
                continue
            xc, yc = radec_to_pixel(wcs_obj, ra_c, dec_c)
            if not in_bounds(img, xc, yc, margin=margin):
                continue
            # 同視野多目標共用快取：命中則跳過重測
            _cached_phot = shared_cache.get(f.stem, ra_c, dec_c) if shared_cache else None
            if _cached_phot is not None:
                phot_c = _cached_phot
            else:
                phot_c = aperture_photometry(img, xc, yc, ap_radius, r_in, r_out)
                if shared_cache is not None:
                    shared_cache.set(f.stem, ra_c, dec_c, phot_c)
            if phot_c.get("ok") != 1 or not np.isfinite(phot_c.get("flux_net")):
                continue
            if is_saturated(phot_c.get("max_pix", np.nan), cfg.saturation_threshold):
                continue
            m_inst_c = m_inst_from_flux(phot_c["flux_net"])
            if not np.isfinite(m_inst_c):
                continue
            dist_arcsec   = float(target_coord.separation(sc).arcsec)
            # ε = plate_scale / 2，防止距離趨近於零時權重爆炸
            epsilon       = cfg.plate_scale_arcsec / 2.0
            weight        = 1.0 / (dist_arcsec + epsilon) ** 2
            # 若星表誤差已知，納入誤差加權（誤差越大，權重越低）
            if m_err is not None and np.isfinite(m_err) and float(m_err) > 0:
                weight /= float(m_err) ** 2
            # 實測 S/N 加權（暫時停用，待驗證影響後再啟用）
            # _flux_c = phot_c.get("flux_net", 0.0)
            # if _flux_c > 0:
            #     weight *= np.sqrt(_flux_c)
            comp_m_inst.append(m_inst_c)
            comp_m_cat.append(float(m_cat))
            comp_weights.append(weight)

            # ── ⏸ 累積 comp_lightcurves（ensemble 正規化用，已停用）──────────────
            # 原本用途：ensemble_normalize() 需要所有比較星的 m_inst 時間序列。
            # 現況：_ensemble_on 永遠 False，此段無法執行，保留作參考。
            if _ensemble_on:
                _sid = f"{ra_c:.6f}_{dec_c:.6f}"
                if _sid not in _comp_lc_buf:
                    _comp_lc_buf[_sid] = {}
                    _comp_init_weights[_sid] = weight
                _comp_lc_buf[_sid][bjd_tdb if np.isfinite(bjd_tdb) else mjd] = m_inst_c

        comp_m_inst  = np.asarray(comp_m_inst, dtype=float)
        comp_m_cat   = np.asarray(comp_m_cat,  dtype=float)

        if len(comp_m_inst) < cfg.robust_regression_min_points:
            rows.append(rec)
            continue

        fit = robust_linear_fit(
            comp_m_inst, comp_m_cat,
            sigma=cfg.robust_regression_sigma,
            max_iter=cfg.robust_regression_max_iter,
            min_points=cfg.robust_regression_min_points,
            weights=np.asarray(comp_weights, dtype=float),
        )

        # 捕捉第一幀資料供診斷圖使用（全部比較星 + 實際 fit）
        if _first_frame_diag_data is None and len(comp_m_cat) >= 2:
            _first_frame_diag_data = (
                comp_m_cat.copy(), comp_m_inst.copy(), fit,
                float(getattr(cfg, "vmag_approx", np.nan)),
                float(m_inst_t),
            )

        # ── 校正擬合：自由斜率回歸（v1.5 邏輯，使用者要求）────────────────
        # 參考：Paxson 2010 JAAVSO；比較星色彩多樣時 slope=1 假設不成立。
        # 目標星在比較星星等範圍內 → 插值，不外插。
        if fit is not None:
            a      = fit["a"]
            b      = fit["b"]
            r2     = fit["r2"]
            mask   = fit["mask"]
            _a_fit = a
        else:
            a, b, r2, _a_fit = np.nan, np.nan, np.nan, np.nan
            mask = np.zeros(len(comp_m_inst), dtype=bool)

        # ── [DISABLED — 使用者要求停用] 純差分測光（slope=1.0）────────────
        # v1.6D 在 context 超載情況下誤加，與使用者方法不符，已停用。
        # 使用者方法：自由斜率回歸校正（非 slope=1 純差分）。
        # mask = np.isfinite(comp_m_inst) & np.isfinite(comp_m_cat)
        # zp   = float(np.nanmedian(comp_m_cat[mask] - comp_m_inst[mask])) if mask.any() else np.nan
        # a, b = 1.0, -zp
        # if fit is not None:
        #     _a_fit, _r2_fit = fit["a"], fit["r2"]
        #     mask = fit["mask"]
        #     zp = float(np.nanmedian(comp_m_cat[mask] - comp_m_inst[mask])) if mask.any() else zp
        #     a, b = 1.0, -zp
        #     r2 = _r2_fit
        # else:
        #     _a_fit, r2 = np.nan, np.nan

        if not (np.isfinite(a) and np.isfinite(b)):
            rows.append(rec)
            continue

        # ── [3] 回歸 R² 幀層級篩選 ────────────────────────────────────────────
        # reg_r2_min 預設 0.0（停用）；> 0 時才自動剔除，保留 WARN 輸出。
        _reg_r2_min = float(getattr(cfg, "reg_r2_min", 0.0))
        if np.isfinite(r2) and _reg_r2_min > 0 and r2 < _reg_r2_min:
            _phot_logger.warning(
                "[WARN] low reg R2=%.4f < %.4f in %s (channel=%s)",
                r2, _reg_r2_min, f.name, channel,
            )
            rec["reg_r2"] = r2
            rec["ok"] = 0
            rec["ok_flag"] = "low_reg_r2"
            n_low_reg_r2 += 1
            rows.append(rec)
            continue
        elif np.isfinite(r2) and r2 < max(_reg_r2_min, 0.5):
            _phot_logger.warning(
                "[WARN] low reg R2=%.4f in %s (channel=%s) — consider raising reg_r2_min",
                r2, f.name, channel,
            )

        m_var       = (m_inst_t - b) / a
        # ── 外插檢查：目標星儀器星等是否在比較星範圍內 ─────────────────────
        _comp_m_inst_used = comp_m_inst[mask]
        _flag_extrap = 0
        if len(_comp_m_inst_used) >= 2:
            _m_lo, _m_hi = float(np.nanmin(_comp_m_inst_used)), float(np.nanmax(_comp_m_inst_used))
            if np.isfinite(m_inst_t) and (m_inst_t < _m_lo or m_inst_t > _m_hi):
                _flag_extrap = 1
                _phot_logger.warning(
                    "[EXTRAP] target m_inst=%.3f outside comp range [%.3f, %.3f] in %s",
                    m_inst_t, _m_lo, _m_hi, f.name,
                )
        comp_used   = int(np.count_nonzero(mask))
        sigma_mag   = mag_error_from_flux(
            flux_net=phot_t["flux_net"],
            b_sky_std=phot_t.get("b_sky_std", phot_t["b_sky"]),  # b_sky_std 優先
            n_pix=phot_t["n_pix"],
            gain_e_per_adu=cfg.gain_e_per_adu,
            read_noise_e=cfg.read_noise_e,
            n_sky=phot_t.get("n_sky"),
        )
        snr = float(1.0857 / sigma_mag) if np.isfinite(sigma_mag) and sigma_mag > 0 else np.nan

        # ── FLAG_SLOPE_DEVIATION（§3.8）：regression slope 監控 ────────────
        # a 是自由斜率回歸的 slope；偏離 1.0 超過 0.05 時記錄警告。
        _slope_flag = int(np.isfinite(_a_fit) and abs(_a_fit - 1.0) > 0.05)
        if _slope_flag:
            _phot_logger.debug(
                "[FLAG] slope_deviation a_fit=%.4f (|a-1|=%.4f > 0.05) in %s",
                _a_fit, abs(_a_fit - 1.0), f.name,
            )

        rec.update({
            "ok": 1,
            "t_flux_net": phot_t["flux_net"],
            "t_b_sky": phot_t["b_sky"],
            "t_n_pix": phot_t["n_pix"],
            "t_max_pix": phot_t["max_pix"],
            "t_saturated": int(sat_t),
            "t_m_inst": m_inst_t,
            "t_sigma_mag": sigma_mag,
            "t_snr": snr,
            "ap_radius": ap_radius,
            "annulus_r_in": r_in,
            "annulus_r_out": r_out,
            "reg_slope": a,           # 自由斜率回歸 slope
            "reg_intercept": b,       # 自由斜率回歸 intercept
            "reg_r2": r2,             # regression R²（監控用）
            "comp_used": comp_used,
            "flag_slope_dev": _slope_flag,
            "reg_slope_fit": _a_fit,  # regression slope（監控用，取代 flag_extrapolated）
            "flag_extrapolated": _flag_extrap,
            "m_var": m_var,
        })

        # ── 零點殘差 RMS（用於診斷圖）────────────────────────────────────────
        _m_cat_fit  = comp_m_cat[mask]
        _m_inst_fit = comp_m_inst[mask]
        _m_inst_pred = a * _m_cat_fit + b   # 預測 m_inst（擬合方向：m_inst = a*m_cat + b）
        _residuals   = _m_inst_fit - _m_inst_pred
        reg_resid_rms = float(np.sqrt(np.mean(_residuals ** 2))) if len(_residuals) > 0 else np.nan
        rec["reg_residual_rms"] = reg_resid_rms

        # ── Check star ───────────────────────────────────────────────────────
        if check_star is not None:
            ra_k, dec_k, m_cat_k = check_star
            xk, yk = radec_to_pixel(wcs_obj, ra_k, dec_k)
            if in_bounds(img, xk, yk, margin=margin):
                phot_k = aperture_photometry(img, xk, yk, ap_radius, r_in, r_out)
                if phot_k.get("ok") == 1 and np.isfinite(phot_k.get("flux_net")):
                    sat_k = is_saturated(phot_k.get("max_pix", np.nan), cfg.saturation_threshold)
                    if not sat_k or cfg.allow_saturated_check:
                        m_inst_k = m_inst_from_flux(phot_k["flux_net"])
                        if np.isfinite(m_inst_k):
                            m_check = (m_inst_k - b) / a
                            rec["k_m_inst"] = m_inst_k
                            rec["k_m_var"]  = m_check
                            rec["k_m_cat"]  = m_cat_k
                            if m_cat_k is not None and np.isfinite(m_cat_k):
                                rec["k_minus_c"] = m_check - float(m_cat_k)
        rows.append(rec)

    df = pd.DataFrame(rows)

    if len(df) == 0:
        print("[WARN] 所有幀均被跳過（airmass），無任何資料列。")
        df = pd.DataFrame(columns=["file", "mjd", "bjd_tdb", "airmass", "ok", "m_var",
                                    "m_var_norm", "delta_ensemble"])
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        return df, {}

    # ── Choose time axis: prefer BJD_TDB ─────────────────────────────────────
    if "bjd_tdb" in df.columns and np.isfinite(df["bjd_tdb"]).any():
        time_key = "bjd_tdb"
    elif "mjd" in df.columns:
        time_key = "mjd"
        print("[WARN] BJD_TDB not available; falling back to MJD. "
              "Check obs_lat_deg / obs_lon_deg in Cfg.")
    else:
        time_key = "bjd_tdb"   # 保險 fallback，sort 會忽略 NaN

    df = df.sort_values(time_key, na_position="last")

    # ── Sigma-clip m_var：排除零點崩潰造成的極端離群值 ──────────────────────
    # 對 ok==1 且 m_var 有限的子集，計算 median 和 MAD（robust σ = 1.4826 × MAD）。
    # |m_var − median| > 3σ 的幀，ok 改為 0，記錄 ok_flag = "sigma_clip"。
    # CSV 保留所有列，畫圖/週期分析只用 ok==1。
    _ok_mask = (df["ok"] == 1) & np.isfinite(df["m_var"])
    _n_before_clip = int(_ok_mask.sum())
    if _n_before_clip >= 5:
        _m = df.loc[_ok_mask, "m_var"].values
        _med = float(np.median(_m))
        _mad = float(np.median(np.abs(_m - _med)))
        _robust_sigma = 1.4826 * _mad if _mad > 0 else 1e-9
        _clip_mask = np.abs(_m - _med) > 3.0 * _robust_sigma
        _clip_idx = df.index[_ok_mask][_clip_mask]
        if len(_clip_idx) > 0:
            df.loc[_clip_idx, "ok"] = 0
            if "ok_flag" not in df.columns:
                df["ok_flag"] = ""
            df.loc[_clip_idx, "ok_flag"] = "sigma_clip"
            print(f"[sigma-clip] median={_med:.4f}  robust_sigma={_robust_sigma:.4f}  "
                  f"clipped {len(_clip_idx)} frames "
                  f"(|m_var - median| > 3sigma = {3.0 * _robust_sigma:.4f})")
    _n_after_clip = int((df["ok"] == 1).sum())

    # ── 大氣消光一階改正（optional）──────────────────────────────────────────
    # m_var_ext = m_var - k × (X_frame - X_ref)
    # k = extinction_k (mag/airmass)，X_ref = 觀測中最小氣團
    # 差分測光中目標與比較星消光大致抵消，但長基線或大氣團差時殘差累積。
    _ext_k = float(getattr(cfg, "extinction_k", 0.0))
    if _ext_k > 0 and "airmass" in df.columns:
        _ok_am = (df["ok"] == 1) & np.isfinite(df["airmass"]) & np.isfinite(df["m_var"])
        if _ok_am.sum() >= 3:
            _X_ref = float(df.loc[_ok_am, "airmass"].min())
            _delta_ext = _ext_k * (df["airmass"] - _X_ref)
            df["m_var_raw"] = df["m_var"].copy()
            df.loc[_ok_am, "m_var"] = df.loc[_ok_am, "m_var"] - _delta_ext[_ok_am]
            _med_corr = float(_delta_ext[_ok_am].median())
            print(f"[extinction] k={_ext_k:.3f} mag/airmass  X_ref={_X_ref:.3f}  "
                  f"median correction={_med_corr:.4f} mag  ({_ok_am.sum()} frames)")

    # ── 回歸截距突變篩選 ───────────────────────────────────────────────────────
    # 薄雲或透明度驟變時所有比較星同步變暗，零點截距 b 會系統性漂移。
    # 滾動中位數（±5 幀）捕捉短時突變，偏離超過 reg_intercept_sigma × MAD 則剔除。
    n_reg_jump = 0
    _reg_intercept_sigma = float(getattr(cfg, "reg_intercept_sigma", 0.0))
    if _reg_intercept_sigma > 0 and "reg_intercept" in df.columns:
        _reg_col = df["reg_intercept"].copy()
        _reg_roll_med = _reg_col.rolling(window=11, center=True, min_periods=3).median()
        _reg_resid = (_reg_col - _reg_roll_med).abs()
        _reg_mad = float(np.nanmedian(_reg_resid[df["ok"] == 1]))
        _reg_thresh = _reg_intercept_sigma * 1.4826 * _reg_mad if _reg_mad > 0 else np.inf
        _reg_jump_mask = (df["ok"] == 1) & (_reg_resid > _reg_thresh)
        if _reg_jump_mask.any():
            df.loc[_reg_jump_mask, "ok"] = 0
            if "ok_flag" not in df.columns:
                df["ok_flag"] = ""
            df.loc[_reg_jump_mask, "ok_flag"] = "reg_jump"
            n_reg_jump = int(_reg_jump_mask.sum())
            print(f"[reg_jump] MAD={_reg_mad:.4f}  thresh={_reg_thresh:.4f}  "
                  f"clipped {n_reg_jump} frames")

    # ── 天空背景突升篩選 ──────────────────────────────────────────────────────
    # 起霧或散射光使目標孔徑背景環中位數升高，t_b_sky 突升可做為霧的早期指標。
    # 滾動中位數（±5 幀）捕捉短時突變，偏離超過 sky_sigma × MAD 則剔除。
    n_high_sky = 0
    _sky_sigma = float(getattr(cfg, "sky_sigma", 0.0))
    if _sky_sigma > 0 and "t_b_sky" in df.columns:
        _sky_col = df["t_b_sky"].copy()
        _sky_roll_med = _sky_col.rolling(window=11, center=True, min_periods=3).median()
        _sky_resid = _sky_col - _sky_roll_med  # 只看正向突升
        _sky_mad = float(np.nanmedian(_sky_resid[df["ok"] == 1].abs()))
        _sky_thresh = _sky_sigma * 1.4826 * _sky_mad if _sky_mad > 0 else np.inf
        _sky_jump_mask = (df["ok"] == 1) & (_sky_resid > _sky_thresh)
        if _sky_jump_mask.any():
            df.loc[_sky_jump_mask, "ok"] = 0
            if "ok_flag" not in df.columns:
                df["ok_flag"] = ""
            df.loc[_sky_jump_mask, "ok_flag"] = "high_sky"
            n_high_sky = int(_sky_jump_mask.sum())
            print(f"[high_sky] MAD={_sky_mad:.4f}  thresh={_sky_thresh:.4f}  "
                  f"clipped {n_high_sky} frames")

    # ── peak_ratio 自適應篩選 ─────────────────────────────────────────────────
    # 自適應門檻，不依賴絕對值，可移植至不同望遠鏡/相機。
    # peak_ratio 極低代表 PSF 中心被挖空（次鏡起霧甜甜圈）。
    # 門檻 = median(peak_ratio) - peak_ratio_k × MAD，整夜所有幀一起算。
    n_low_peak_ratio_adaptive = 0
    _peak_ratio_k = float(getattr(cfg, "peak_ratio_k", 0.0))
    if _peak_ratio_k > 0 and "peak_ratio" in df.columns:
        _pr_vals = df.loc[df["ok"] == 1, "peak_ratio"].dropna()
        if len(_pr_vals) >= 5:
            _pr_med = float(np.median(_pr_vals))
            _pr_mad = float(np.median(np.abs(_pr_vals - _pr_med)))
            _pr_thresh = _pr_med - _peak_ratio_k * 1.4826 * _pr_mad
            _pr_mask = (df["ok"] == 1) & (df["peak_ratio"].notna()) & (df["peak_ratio"] < _pr_thresh)
            if _pr_mask.any():
                df.loc[_pr_mask, "ok"] = 0
                if "ok_flag" not in df.columns:
                    df["ok_flag"] = ""
                df.loc[_pr_mask, "ok_flag"] = "low_peak_ratio"
                n_low_peak_ratio_adaptive = int(_pr_mask.sum())
                print(f"[peak_ratio adaptive] median={_pr_med:.4f}  MAD={_pr_mad:.4f}  "
                      f"thresh={_pr_thresh:.4f}  clipped {n_low_peak_ratio_adaptive} frames")

    # ── 剔除統計表 ────────────────────────────────────────────────────────────
    # 統計結果同步存檔（帶時間戳，不覆蓋），供調整篩選閾值時對照各版本剔除效果
    _n_total_fits  = len(wcs_files_sorted)
    _n_in_df       = len(df)
    _n_ok_final    = int((df["ok"] == 1).sum())
    _n_alt_skip    = n_skipped  # alt_too_low（未進 df）
    _n_sigma_clip  = int(((df["ok"] == 0) & (df.get("ok_flag", pd.Series("", index=df.index)) == "sigma_clip")).sum()) if "ok_flag" in df.columns else (_n_before_clip - _n_after_clip)

    _n_high_fwhm_val     = n_high_fwhm
    _n_low_sharpness_val = n_low_sharpness
    _n_low_peak_ratio_val = n_low_peak_ratio
    _n_low_reg_r2_val     = n_low_reg_r2
    # 重新計算 _n_phot_fail：排除所有已命名篩選計數
    _n_qual_filtered = (_n_high_fwhm_val + _n_low_sharpness_val
                        + _n_low_peak_ratio_val + _n_low_reg_r2_val
                        + n_reg_jump + n_high_sky + n_low_peak_ratio_adaptive)
    _n_phot_fail   = _n_in_df - _n_ok_final - _n_sigma_clip - _n_qual_filtered

    _sep = "-" * 68
    print(f"\n[剔除統計] 通道 {channel}  {getattr(cfg, 'target_name', '')}  {str(out_csv.stem).split('_')[-1]}")
    print(_sep)
    print(f"  {'原因':<24} {'幀數':>6}    公式")
    print(_sep)
    print(f"  {'高氣團跳過':<24} {_n_alt_skip:>6}    airmass > {cfg.alt_min_airmass:.2f}")
    print(f"  {'高 FWHM 幀剔除':<24} {_n_high_fwhm_val:>6}    frame_fwhm_median > {cfg.comp_fwhm_max:.1f} px")
    print(f"  {'低 Sharpness 剔除':<24} {_n_low_sharpness_val:>6}    S=flux(r=3)/flux(r=8) < {float(getattr(cfg,'sharpness_min',0.3)):.2f}")
    print(f"  {'低 Peak Ratio 剔除':<24} {_n_low_peak_ratio_val:>6}    peak/flux < {float(getattr(cfg,'peak_ratio_min',0.0)):.3f} (0=停用)")
    print(f"  {'低 reg R2 剔除':<24} {_n_low_reg_r2_val:>6}    reg_r2 < {float(getattr(cfg,'reg_r2_min',0.0)):.2f} (0=停用)")
    print(f"  {'孔徑/WCS/邊界失敗':<24} {_n_phot_fail:>6}    flux/位置無效")
    print(f"  {'sigma_clip':<24} {_n_sigma_clip:>6}    |m_var - median| > 3 * 1.4826 * MAD")
    print(f"  {'回歸截距突變':<24} {n_reg_jump:>6}    rolling median ± {float(getattr(cfg,'reg_intercept_sigma',0.0)):.1f} MAD (0=停用)")
    print(f"  {'天空背景突升':<24} {n_high_sky:>6}    rolling median + {float(getattr(cfg,'sky_sigma',0.0)):.1f} MAD (0=停用)")
    print(f"  {'Peak Ratio 自適應':<24} {n_low_peak_ratio_adaptive:>6}    median - {float(getattr(cfg,'peak_ratio_k',0.0)):.1f} MAD (0=停用)")
    print(_sep)
    print(f"  {'保留 (ok=1)':<24} {_n_ok_final:>6} / {_n_total_fits} 幀")
    print()

    # 存檔：帶時間戳，不覆蓋
    _rej_rows = [
        {"reason": "高氣團跳過",       "count": _n_alt_skip,            "threshold": f"airmass > {cfg.alt_min_airmass:.2f}",                              "config_key": "max_airmass",         "config_value": cfg.alt_min_airmass},
        {"reason": "高 FWHM 幀剔除",   "count": _n_high_fwhm_val,       "threshold": f"fwhm > {cfg.comp_fwhm_max:.1f} px",                                "config_key": "max_fwhm_px",         "config_value": cfg.comp_fwhm_max},
        {"reason": "低 Sharpness 剔除", "count": _n_low_sharpness_val,   "threshold": f"S < {float(getattr(cfg,'sharpness_min',0.3)):.2f}",                 "config_key": "sharpness_min",       "config_value": float(getattr(cfg, "sharpness_min", 0.3))},
        {"reason": "低 Peak Ratio 剔除","count": _n_low_peak_ratio_val,  "threshold": f"peak/flux < {float(getattr(cfg,'peak_ratio_min',0.0)):.3f}",        "config_key": "peak_ratio_min",      "config_value": float(getattr(cfg, "peak_ratio_min", 0.0))},
        {"reason": "低 reg R2 剔除",    "count": _n_low_reg_r2_val,       "threshold": f"reg_r2 < {float(getattr(cfg,'reg_r2_min',0.0)):.2f}",                 "config_key": "reg_r2_min",           "config_value": float(getattr(cfg, "reg_r2_min", 0.0))},
        {"reason": "孔徑/WCS/邊界失敗", "count": _n_phot_fail,           "threshold": "flux/位置無效",                                                     "config_key": "—",                   "config_value": "—"},
        {"reason": "sigma_clip",        "count": _n_sigma_clip,          "threshold": "|m_var - median| > 3 * 1.4826 * MAD",                               "config_key": "—",                   "config_value": "—"},
        {"reason": "回歸截距突變",       "count": n_reg_jump,              "threshold": f"rolling |reg_intercept - med| > {float(getattr(cfg,'reg_intercept_sigma',0.0)):.1f} MAD", "config_key": "reg_intercept_sigma", "config_value": float(getattr(cfg, "reg_intercept_sigma", 0.0))},
        {"reason": "天空背景突升",       "count": n_high_sky,                    "threshold": f"rolling (t_b_sky - med) > {float(getattr(cfg,'sky_sigma',0.0)):.1f} MAD",                     "config_key": "sky_sigma",           "config_value": float(getattr(cfg, "sky_sigma", 0.0))},
        {"reason": "Peak Ratio 自適應", "count": n_low_peak_ratio_adaptive,    "threshold": f"peak_ratio < median - {float(getattr(cfg,'peak_ratio_k',0.0)):.1f} * 1.4826 * MAD",           "config_key": "peak_ratio_k",        "config_value": float(getattr(cfg, "peak_ratio_k", 0.0))},
        {"reason": "保留 (ok=1)",       "count": _n_ok_final,                   "threshold": f"/ {_n_total_fits} 幀",                                                                         "config_key": "—",                   "config_value": "—"},
    ]
    _rej_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _rej_stem = out_csv.stem  # e.g. photometry_G1_20251220
    _rej_path = out_csv.parent / f"rejection_stats_{_rej_stem}_{_rej_ts}.csv"
    pd.DataFrame(_rej_rows).to_csv(_rej_path, index=False, encoding="utf-8-sig")
    print(f"[剔除統計] saved → {_rej_path}")

    # 第一次寫入（ensemble normalize 後會再寫一次，更新 m_var_norm/delta_ensemble）
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    n_written = len(df)
    print(f"[CSV] saved → {out_csv}  "
          f"({n_written} rows written, {_n_after_clip} successful "
          f"[{_n_before_clip - _n_after_clip} sigma-clipped], "
          f"{n_skipped} skipped [airmass > {cfg.alt_min_airmass:.3f}])")

    # ── 回歸診斷總覽圖：回歸散佈圖（主）+ RMS 時序（輔）──────────────────────
    if cfg.save_regression_diagnostic:
        try:
            _reg_diag_path = cfg.regression_diag_dir / (
                f"reg_overview_{out_csv.stem.split('_')[1]}_{out_csv.stem.split('_', 1)[-1]}.png"
            )
            _df_ok = df[df["ok"] == 1].copy()
            import datetime as _dt_reg
            import matplotlib.ticker as _mticker_reg
            import matplotlib.gridspec as _mgs

            fig_diag = plt.figure(figsize=(8, 11))
            gs = _mgs.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.30)
            ax_sc = fig_diag.add_subplot(gs[0])
            ax_sc.set_box_aspect(1.0)   # 正方形圖框
            ax_ts = fig_diag.add_subplot(gs[1])

            # ── 上：第一幀回歸散佈圖（全部比較星，正方形）────────────────────
            _ffd = _first_frame_diag_data
            if _ffd is not None:
                _mc, _mi, _fit = _ffd[:3]
                _tgt_vmag_diag = _ffd[3] if len(_ffd) > 3 else np.nan
                _tgt_minst     = _ffd[4] if len(_ffd) > 4 else np.nan
                _ok_pts = np.isfinite(_mc) & np.isfinite(_mi)
                ax_sc.scatter(_mc[_ok_pts], _mi[_ok_pts], s=18, alpha=0.6,
                              color="steelblue", label=f"comp (n={_ok_pts.sum()})")
                if _fit and np.isfinite(_fit.get("a", np.nan)):
                    _a, _b, _r2 = _fit["a"], _fit["b"], _fit.get("r2", np.nan)
                    _xl = np.linspace(_mc[_ok_pts].min(), _mc[_ok_pts].max(), 100)
                    ax_sc.plot(_xl, _a * _xl + _b, "r-", lw=1.8,
                               label=f"$m_{{inst}}={_a:.3f}\\,m_{{cat}}+({_b:.3f})$  $R^2={_r2:.3f}$")
                # 目標星紅圈
                if _fit and np.isfinite(_tgt_minst):
                    _a_d, _b_d = _fit["a"], _fit["b"]
                    if np.isfinite(_a_d) and _a_d != 0:
                        _tgt_mvar = (_tgt_minst - _b_d) / _a_d
                        ax_sc.scatter([_tgt_mvar], [_tgt_minst], s=100, marker="o",
                                      facecolors="none", edgecolors="red", linewidths=2.0,
                                      zorder=5, label=f"target  $m_{{var}}$={_tgt_mvar:.1f}")
                # x 軸：左界=比較星+目標星中最亮，右界=comp_mag_max
                _brightest_comp = float(_mc[_ok_pts].min()) if _ok_pts.any() else float(cfg.comp_mag_min)
                _x_lo = _brightest_comp
                if _fit and np.isfinite(_tgt_vmag_diag):
                    _x_lo = min(_x_lo, float(_tgt_vmag_diag))
                _x_hi = float(cfg.comp_mag_max)
                ax_sc.set_xlim(_x_lo - 0.2, _x_hi + 0.1)
                # 外插區域：回歸線用虛線
                if _fit and np.isfinite(_fit.get("a", np.nan)) and _ok_pts.any():
                    _comp_lo = float(_mc[_ok_pts].min())
                    _comp_hi = float(_mc[_ok_pts].max())
                    # 亮端外插（比較星最亮 → x 左界）
                    if _x_lo < _comp_lo - 0.01:
                        _xl_ext = np.linspace(_x_lo - 0.2, _comp_lo, 50)
                        ax_sc.plot(_xl_ext, _a * _xl_ext + _b, "r--", lw=1.2, alpha=0.5)
                    # 暗端外插（比較星最暗 → x 右界）
                    if _x_hi > _comp_hi + 0.01:
                        _xl_ext = np.linspace(_comp_hi, _x_hi + 0.1, 50)
                        ax_sc.plot(_xl_ext, _a * _xl_ext + _b, "r--", lw=1.2, alpha=0.5)
                # y 軸：緊貼資料 + 目標星 + 回歸線在 x 範圍內的值
                _all_mi = list(_mi[_ok_pts]) if _ok_pts.any() else []
                if _fit and np.isfinite(_tgt_minst):
                    _all_mi.append(float(_tgt_minst))
                if _fit and np.isfinite(_fit.get("a", np.nan)):
                    _all_mi.append(float(_fit["a"] * _x_lo + _fit["b"]))
                    _all_mi.append(float(_fit["a"] * _x_hi + _fit["b"]))
                if _all_mi:
                    _mi_lo, _mi_hi = min(_all_mi), max(_all_mi)
                    _mi_pad = (_mi_hi - _mi_lo) * 0.12
                    ax_sc.set_ylim(_mi_hi + _mi_pad, _mi_lo - _mi_pad)  # inverted
                else:
                    ax_sc.invert_yaxis()
                ax_sc.set_xlabel(f"$m_{{cat}}$ ({cfg.phot_band})", fontsize=11)
                ax_sc.set_ylabel("$m_{inst}$", fontsize=11)
                _extrap_note = ""
                if _ok_pts.any() and np.isfinite(_tgt_vmag_diag):
                    if _tgt_vmag_diag < float(_mc[_ok_pts].min()):
                        _extrap_note = "  [EXTRAPOLATION]"
                ax_sc.set_title(
                    f"Regression Fit — Frame 1  ({_x_lo:.1f}–{_x_hi:.1f} mag){_extrap_note}",
                    fontsize=12,
                )
                ax_sc.legend(fontsize=8, frameon=False)
                ax_sc.grid(True, alpha=0.3)
            else:
                ax_sc.text(0.5, 0.5, "No first-frame data",
                           transform=ax_sc.transAxes, ha="center", va="center")

            # ── 下：RMS 時序圖（參考用）───────────────────────────────────────
            if "reg_residual_rms" in _df_ok.columns and np.isfinite(_df_ok["reg_residual_rms"]).any():
                ax_ts.plot(
                    _df_ok[time_key], _df_ok["reg_residual_rms"],
                    "o-", ms=2, lw=0.6, color="steelblue", alpha=0.7
                )
                ax_ts.axhline(
                    _df_ok["reg_residual_rms"].median(), color="red",
                    lw=1, ls="--", label=f"median={_df_ok['reg_residual_rms'].median():.4f}"
                )
                ax_ts.set_xlabel(time_key.upper(), fontsize=9)
                ax_ts.set_ylabel("Residual RMS (mag)", fontsize=9)
                ax_ts.legend(fontsize=7)
                ax_ts.grid(True, alpha=0.3)
                ax_ts.tick_params(labelsize=8)
                ax_ts.xaxis.set_major_formatter(
                    _mticker_reg.FuncFormatter(lambda v, _: f"{v:.2f}")
                )
            else:
                ax_ts.text(0.5, 0.5, "No residual data", transform=ax_ts.transAxes,
                           ha="center", va="center")

            fig_diag.suptitle(
                f"Regression Diagnostic  |  {cfg.target_name}  "
                f"channel={channel}  {out_csv.stem}",
                fontsize=11
            )
            fig_diag.savefig(_reg_diag_path, dpi=150, bbox_inches="tight")
            plt.close(fig_diag)
            print(f"[reg_diag] saved → {_reg_diag_path}")
        except Exception as _e:
            import traceback; traceback.print_exc()
            print(f"[WARN] regression diagnostic failed: {_e}")

    # ── Rejection timeline 圖 ────────────────────────────────────────────────
    try:
        _rej_colors = {
            "high_airmass":  "#aaaaaa",
            "high_fwhm":     "#e67e22",
            "low_sharpness": "#9b59b6",
            "low_peak_ratio":"#1abc9c",
            "low_reg_r2":     "#e74c3c",
            "phot_fail":     "#c0392b",
            "sigma_clip":    "#2980b9",
            "reg_jump":       "#f39c12",
            "high_sky":      "#16a085",
            "ok":            "#cccccc",
        }
        _rej_fig, _rej_ax = plt.subplots(figsize=(12, 3))
        _df_all = df.copy()
        if "ok_flag" not in _df_all.columns:
            _df_all["ok_flag"] = None
        _df_all["_flag"] = _df_all["ok_flag"].fillna("ok").where(_df_all["ok"] == 0, "ok")
        # 用 bjd_tdb 轉本地時間（UTC+8）
        _tz_h = float(getattr(cfg, "tz_offset_hours", 8))
        if time_key in _df_all.columns and _df_all[time_key].notna().any():
            _t_local = (_df_all[time_key] - 2400000.5) * 86400  # MJD → 秒，只取相對值
            _t0 = _t_local.min()
            _t_rel = (_t_local - _t0) / 3600  # 相對小時
        else:
            _t_rel = pd.Series(range(len(_df_all)), dtype=float)
        for _flag, _grp in _df_all.groupby("_flag"):
            _col = _rej_colors.get(_flag, "#888888")
            _size = 15 if _flag == "ok" else 40
            _zord = 2 if _flag == "ok" else 3
            _idx = _df_all.index[_df_all["_flag"] == _flag]
            _rej_ax.scatter(_t_rel.iloc[_idx], _df_all.loc[_df_all["_flag"] == _flag, "airmass"],
                            c=_col, s=_size, label=_flag, zorder=_zord, alpha=0.85)
        _rej_ax.set_xlabel("Elapsed time (hours from first frame)")
        _rej_ax.set_ylabel("Airmass")
        _rej_ax.invert_yaxis()
        _rej_ax.set_title(
            f"Rejection timeline  |  {cfg.target_name}  ch={cfg.phot_band}  {out_csv.stem}",
            fontsize=9)
        _rej_ax.legend(fontsize=7, ncol=5, loc="upper right")
        _rej_ax.grid(True, alpha=0.25)
        _rej_fig.tight_layout()
        _rej_path = out_csv.parent / f"rejection_timeline_{out_csv.stem}.png"
        _rej_fig.savefig(_rej_path, dpi=150)
        plt.close(_rej_fig)
        print(f"[剔除時序圖] saved → {_rej_path}")
    except Exception as _e:
        print(f"[WARN] 剔除時序圖輸出失敗：{_e}")

    # ── Plot light curve ──────────────────────────────────────────────────────
    d = df[(df["ok"] == 1) & np.isfinite(df[time_key]) & np.isfinite(df["m_var"])].copy()
    if len(d) == 0:
        print("[PLOT] No valid photometry points to plot.")
        df["m_var_norm"] = df["m_var"]
        df["delta_ensemble"] = np.nan
        return df, {}

    import matplotlib.ticker as mticker
    # ATime 已在行 2950 附近作為 local 變數使用，不在此重新 import（會遮蔽）

    # 從 csv 檔名取觀測日期
    _lc_obs_date = str(out_csv.stem).split("_")[-1]

    bjd_arr = d[time_key].values
    bjd_min = float(bjd_arr.min())
    bjd_max = float(bjd_arr.max())

    # 本地時間（UTC+8）換算：BJD_TDB ≈ BJD_UTC + 67.2s（微小），近似用 UTC+8
    _tz_offset_h = float(getattr(cfg, "tz_offset_hours", 8))

    def _bjd_to_local_hm(bjd_val):
        """BJD → 本地時間 HH:MM（近似，UTC+8）"""
        try:
            t_utc = ATime(bjd_val, format="jd", scale="tdb").to_datetime()
            import datetime as _dt_local
            t_local = t_utc + _dt_local.timedelta(hours=_tz_offset_h)
            return t_local
        except Exception:
            return None

    # 計算 Local Time 刻度位置（BJD）
    _t_local_min = _bjd_to_local_hm(bjd_min)
    _t_local_max = _bjd_to_local_hm(bjd_max)

    _major_bjd_ticks = []   # 整點
    _label30_bjd_ticks = []  # 整點 + 半點（有數字標示）
    _label30_labels = []
    _minor_bjd_ticks = []   # 每 10 分鐘
    if _t_local_min is not None and _t_local_max is not None:
        import datetime as _dt_local
        # 每 10 分鐘刻度，收集整點 / 半點 / 其他
        _cur = _t_local_min.replace(second=0, microsecond=0)
        _min_round = (_cur.minute // 10) * 10
        _cur = _cur.replace(minute=_min_round)
        while _cur <= _t_local_max + _dt_local.timedelta(minutes=1):
            _utc = _cur - _dt_local.timedelta(hours=_tz_offset_h)
            _bjd_tick = ATime(_utc).jd
            _minor_bjd_ticks.append(_bjd_tick)
            if _cur.minute == 0:
                _major_bjd_ticks.append(_bjd_tick)
            if _cur.minute in (0, 30):
                _label30_bjd_ticks.append(_bjd_tick)
                _label30_labels.append(_cur.strftime("%H:%M"))
            _cur += _dt_local.timedelta(minutes=10)

    _lc_obs_date = str(out_csv.stem).split("_")[-1]
    # 光變曲線圖移至 ensemble normalize 之後繪製（使用 m_var_norm）

    # ── Check star residuals ──────────────────────────────────────────────────
    if check_star is not None and "k_minus_c" in df.columns:
        k = df[(df["ok"] == 1) & np.isfinite(df[time_key])
               & np.isfinite(df["k_minus_c"])].copy()
        if len(k) > 1:
            _t0_check = float(k[time_key].min())
            k["t_rel_d"] = k[time_key] - _t0_check
            sigma_k = float(np.nanstd(k["k_minus_c"]))
            flag    = " [!] EXCEEDS THRESHOLD" if sigma_k > cfg.check_star_max_sigma else ""
            print(f"[CHECK] K-C sigma = {sigma_k:.5f} mag "
                  f"(threshold = {cfg.check_star_max_sigma}){flag}")
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.plot(k["t_rel_d"], k["k_minus_c"], "o-", ms=4, lw=0.8)
            ax2.xaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, _: f"{int(x)}d {int((x % 1)*24)}h {int(((x % 1)*24 % 1)*60)}m"
            ))
            ax2.invert_yaxis()
            ax2.set_xlabel("t_rel (days)")
            ax2.set_ylabel("K - C  (mag)")
            ax2.set_title("Check Star Validation")
            ax2.grid(True, alpha=0.4)
            fig2.tight_layout()
            plt.close("all")

    # ── ⏸ 待決定：Broeg (2005) ensemble 正規化 ─────────────────────────────────
    # 理由：逐幀自由斜率回歸已消除逐幀大氣，ensemble 重複修正可能引入雜訊。
    # 待驗：移除後 R² 是否有變化（預期無）。
    # 原本邏輯：_ensemble_on=True 時迭代算 Δ(t)，然後 m_var_norm = m_var - Δ(t)。
    # 簡化版本：關閉 ensemble，m_var_norm 直接 = m_var（回歸後的校正星等）。
    """
    # ── 原始 ensemble 迴圈（已停用，整塊保留作參考）────────────────────────────────
    comp_lightcurves: "dict[str, pd.Series]" = {}
    if _ensemble_on:
        # _comp_lc_buf → pd.Series，index = time 值
        for _sid, _tdict in _comp_lc_buf.items():
            if len(_tdict) >= 2:   # 至少 2 幀才有意義
                comp_lightcurves[_sid] = pd.Series(_tdict, dtype=float)

        _n_comp_lc = len(comp_lightcurves)
        print(f"[ensemble] 比較星光變曲線：{_n_comp_lc} 顆")

        if _n_comp_lc >= _ensemble_min_comp:
            df, _delta_series = ensemble_normalize(
                df,
                comp_lightcurves=comp_lightcurves,
                initial_weights=_comp_init_weights,
                time_key=time_key,
                min_comp_stars=_ensemble_min_comp,
                max_iter=_ensemble_max_iter,
                convergence_tol=_ensemble_tol,
            )
            # ── CSV 更新（已含 m_var_norm、delta_ensemble）────────────────────
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"[CSV] ensemble 正規化後重新寫入 → {out_csv}")
        else:
            _phot_logger.warning(
                "[ensemble] 比較星光變曲線數量 %d < min_comp_stars %d，"
                "跳過 ensemble 正規化。",
                _n_comp_lc, _ensemble_min_comp,
            )
            df["m_var_norm"] = df["m_var"]
            df["delta_ensemble"] = np.nan
    else:
        # ensemble 停用：m_var_norm 直通 m_var
        df["m_var_norm"] = df["m_var"]
        df["delta_ensemble"] = np.nan
    """
    # ── 簡化版本：直接用回歸結果 ────────────────────────────────────────────────
    df["m_var_norm"] = df["m_var"]        # 直通回歸的 m_var，不加 ensemble 修正
    df["delta_ensemble"] = np.nan         # 預留欄位，暫不計算

    # ── 光變曲線圖（ensemble 之後重畫，使用 m_var_norm）──────────────────────
    plot_light_curve(df, out_png, channel, cfg, obs_date=_lc_obs_date)

    # ── 單通道摘要報告 ──────────────────────────────────────────────────────────
    try:
        _ok_df = df[df["ok"] == 1]
        _n_ok = len(_ok_df)
        _n_total = len(df)
        _summary_lines = [
            f"=== VarStar Photometry Summary ===",
            f"Target    : {cfg.target_name}",
            f"Channel   : {channel}",
            f"Date      : {out_csv.stem.split('_')[-1]}",
            f"Frames    : {_n_ok}/{_n_total} ok  "
            f"({n_skipped} airmass-cut, "
            f"{_n_before_clip - _n_after_clip} sigma-clipped)",
            f"Aperture  : {ap_radius:.1f} px  "
            f"(annulus {r_in:.1f}–{r_out:.1f})",
        ]
        if _n_ok > 0:
            _mvar = _ok_df["m_var_norm"] if "m_var_norm" in _ok_df.columns else _ok_df["m_var"]
            _mvar_ok = _mvar[np.isfinite(_mvar)]
            if len(_mvar_ok) > 0:
                _summary_lines.append(
                    f"Mag range : {_mvar_ok.min():.3f} – {_mvar_ok.max():.3f}  "
                    f"(median {_mvar_ok.median():.3f}, std {_mvar_ok.std():.4f})"
                )
            if "reg_r2" in _ok_df.columns:
                _r2_ok = _ok_df["reg_r2"][np.isfinite(_ok_df["reg_r2"])]
                if len(_r2_ok) > 0:
                    _summary_lines.append(
                        f"Reg R²    : median {_r2_ok.median():.4f}  "
                        f"min {_r2_ok.min():.4f}  max {_r2_ok.max():.4f}"
                    )
            if "airmass" in _ok_df.columns:
                _am_ok = _ok_df["airmass"][np.isfinite(_ok_df["airmass"])]
                if len(_am_ok) > 0:
                    _summary_lines.append(
                        f"Airmass   : {_am_ok.min():.3f} – {_am_ok.max():.3f}"
                    )
            if "t_snr" in _ok_df.columns:
                _snr_ok = _ok_df["t_snr"][np.isfinite(_ok_df["t_snr"])]
                if len(_snr_ok) > 0:
                    _summary_lines.append(
                        f"SNR       : median {_snr_ok.median():.1f}  "
                        f"min {_snr_ok.min():.1f}"
                    )
            if "comp_used" in _ok_df.columns:
                _comp_ok = _ok_df["comp_used"][np.isfinite(_ok_df["comp_used"])]
                if len(_comp_ok) > 0:
                    _summary_lines.append(
                        f"Comp stars: median {_comp_ok.median():.0f}  "
                        f"min {_comp_ok.min():.0f}"
                    )
            if _ext_k > 0:
                _summary_lines.append(
                    f"Extinction: k={_ext_k:.3f} mag/airmass applied"
                )
            _baseline_hr = 0.0
            if time_key in _ok_df.columns:
                _t_ok = _ok_df[time_key][np.isfinite(_ok_df[time_key])]
                if len(_t_ok) > 1:
                    _baseline_hr = (_t_ok.max() - _t_ok.min()) * 24
                    _summary_lines.append(
                        f"Baseline  : {_baseline_hr:.2f} hr"
                    )
        _summary_lines.append(f"Output    : {out_csv}")
        _summary_lines.append("")

        _summary_text = "\n".join(_summary_lines)
        _summary_path = out_csv.parent / f"summary_{out_csv.stem}.txt"
        _summary_path.write_text(_summary_text, encoding="utf-8")
        # Ensure console output does not depend on local ANSI codepage
        print(f"[摘要] saved → {_summary_path}")
    except Exception as _e_summary:
        print(f"[WARN] 摘要報告輸出失敗：{_e_summary}")

    return df, {}  # comp_lightcurves 已隨 ensemble 停用，回傳空 dict
