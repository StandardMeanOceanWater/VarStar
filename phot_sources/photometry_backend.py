"""Per-frame photometry backend and output artifact writer."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS

from phot_ensemble import ensemble_normalize
from phot_sources.core import (
    aperture_photometry,
    compute_annulus_radii,
    in_bounds,
    is_saturated,
    m_inst_from_flux,
    radec_to_pixel,
)
from phot_sources.logging_utils import emit_progress, emit_progress_done
from phot_sources.regression import mag_error_from_flux, robust_linear_fit
from phot_timing import apply_gain_from_header, require_cfg_values, time_from_header
from polt_light_curve import plot_light_curve

_phot_logger = logging.getLogger("photometry")


def _emit_progress(logger, message):
    emit_progress(logger, message)


def _emit_progress_done(logger, stage_label, started_at):
    emit_progress_done(logger, stage_label, started_at)


def _emit_photometry_products(
    df: pd.DataFrame,
    out_csv: Path,
    out_png: Path,
    channel: str,
    cfg,
    time_key: str,
    check_star,
    ap_radius: float,
    r_in: float,
    r_out: float,
    n_skipped: int,
    n_before_clip: int,
    n_after_clip: int,
    ext_k: float,
    first_frame_diag_data,
) -> bool:
    """Write post-run artifacts after the science dataframe is finalized."""
    # 第一次寫入（ensemble normalize 後會再寫一次，更新 m_var_norm/delta_ensemble）
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    n_written = len(df)
    print(f"[CSV] saved → {out_csv}  "
          f"({n_written} rows written, {n_after_clip} successful "
          f"[{n_before_clip - n_after_clip} sigma-clipped], "
          f"{n_skipped} skipped [airmass > {cfg.alt_min_airmass:.3f}])")

    d = df[(df["ok"] == 1) & np.isfinite(df[time_key]) & np.isfinite(df["m_var"])].copy()
    if len(d) == 0:
        print("[PLOT] No valid photometry points to plot.")
        df["m_var_norm"] = df["m_var"]
        df["delta_ensemble"] = np.nan
        return False

    _lc_obs_date = str(out_csv.stem).split("_")[-1]

    if check_star is not None and "k_minus_c" in df.columns:
        k = df[(df["ok"] == 1) & np.isfinite(df[time_key])
               & np.isfinite(df["k_minus_c"])].copy()
        if len(k) > 1:
            sigma_k = float(np.nanstd(k["k_minus_c"]))
            flag = " [!] EXCEEDS THRESHOLD" if sigma_k > cfg.check_star_max_sigma else ""
            print(f"[CHECK] K-C sigma = {sigma_k:.5f} mag "
                  f"(threshold = {cfg.check_star_max_sigma}){flag}")

    df["m_var_norm"] = df["m_var"]
    df["delta_ensemble"] = np.nan

    plot_light_curve(df, out_png, channel, cfg, obs_date=_lc_obs_date)

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
            f"{n_before_clip - n_after_clip} sigma-clipped)",
            f"Aperture  : {ap_radius:.1f} px  "
            f"(annulus {r_in:.1f}–{r_out:.1f})",
        ]
        _summary_lines.append(
            f"GrowthFr  : {getattr(cfg, 'aperture_growth_fraction', float('nan')):.3f}"
        )
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
            if ext_k > 0:
                _summary_lines.append(
                    f"Extinction: k={ext_k:.3f} mag/airmass applied"
                )
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
        print(f"[摘要] saved → {_summary_path}")
        for _sl in _summary_lines:
            if _sl:
                print(f"  {_sl}")
    except Exception as _e_summary:
        print(f"[WARN] 摘要報告輸出失敗：{_e_summary}")

    return True


def run_photometry_on_wcs_dir(
    wcs_dir: Path,
    out_csv: Path,
    out_png: Path,
    comp_refs: list,
    cfg_obj,
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
    ra_t, dec_t = cfg_obj.target_radec_deg
    if ap_radius is None:
        ap_radius = cfg_obj.aperture_radius
    r_in, r_out = compute_annulus_radii(ap_radius, cfg_obj.annulus_r_in, cfg_obj.annulus_r_out)
    margin = int(np.ceil(r_out + 2))

    # ── ⏸ Ensemble 正規化設定（已停用，保留供未來重啟）────────────────────────────
    # 原本邏輯：_ensemble_on=True 時調用 ensemble_normalize() 迭代計算 Δ(t)。
    # 現況：簡化版本已跳過呼叫，直接使用回歸結果（見下文 3659-3700 之簡化版）。
    _ensemble_on = bool(getattr(cfg_obj, "ensemble_normalize", False))
    _ensemble_min_comp = int(getattr(cfg_obj, "ensemble_min_comp", 3))
    _ensemble_max_iter = int(getattr(cfg_obj, "ensemble_max_iter", 10))
    _ensemble_tol = float(getattr(cfg_obj, "ensemble_convergence_tol", 1e-4))

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
    _frame_stage_started = time.perf_counter()
    _cache_label = f"{cfg_obj.target_name}:{channel}"
    _frame_total_times: "list[float]" = []
    _target_ap_times: "list[float]" = []
    _comp_ap_times: "list[float]" = []
    _check_ap_times: "list[float]" = []
    _io_load_times: "list[float]" = []
    _time_wcs_times: "list[float]" = []
    _sharpness_times: "list[float]" = []
    _comp_loop_times: "list[float]" = []
    _fit_times: "list[float]" = []
    _check_total_times: "list[float]" = []
    _comp_phot_calls = 0
    _cached_comp_hits = 0
    _emit_progress(
        _phot_logger,
        f"channel {channel} frame loop start total={len(wcs_files_sorted)}"
    )
    _last_heartbeat_at = _frame_stage_started

    for _frame_idx, f in enumerate(wcs_files_sorted, start=1):
        _frame_started = time.perf_counter()
        _now = time.perf_counter()
        if (_frame_idx % 25 == 0) or (_now - _last_heartbeat_at >= 30.0):
            _emit_progress(
                _phot_logger,
                f"channel {channel} heartbeat frames={_frame_idx}/{len(wcs_files_sorted)} "
                f"elapsed={_now - _frame_stage_started:.1f}s"
            )
            _last_heartbeat_at = _now
        _io_started = time.perf_counter()
        with fits.open(f) as hdul:
            img = hdul[0].data.astype(np.float32)
            hdr = hdul[0].header
            wcs_obj = WCS(hdr)
        _io_load_times.append(time.perf_counter() - _io_started)

        apply_gain_from_header(hdr, cfg_obj)
        if not cfg_checked:
            require_cfg_values(cfg_obj)
            cfg_checked = True

        # ── Time, BJD_TDB, airmass ───────────────────────────────────────────
        _time_wcs_started = time.perf_counter()
        mjd, bjd_tdb, airmass = time_from_header(hdr, ra_t, dec_t, cfg_obj)
        xt, yt = radec_to_pixel(wcs_obj, ra_t, dec_t)
        _time_wcs_times.append(time.perf_counter() - _time_wcs_started)
        rec = {
            "file": f.name,
            "mjd": mjd,
            "bjd_tdb": bjd_tdb,
            "airmass": airmass,
            "xt": xt, "yt": yt,
            "ok": 0,
            "m_var": np.nan,
            "sharpness_index": np.nan,
        }

        # ── 高度角截斷（altitude < ALT_MIN_DEG の幀は除外）────────────────────
        # 大氣消光在低高度角時急增，ALT_MIN_DEG=45° 對應 airmass≈1.41。
        # airmass 已知（非 NaN）時才做截斷；location 未設定時不截斷（airmass=NaN）。
        if np.isfinite(airmass) and airmass > cfg_obj.alt_min_airmass:
            n_skipped += 1
            rec["ok_flag"] = "high_airmass"
            _frame_total_times.append(time.perf_counter() - _frame_started)
            rows.append(rec)
            continue

        # ── [1] 幀層級 FWHM 篩選 ─────────────────────────────────────────────
        if not in_bounds(img, xt, yt, margin=margin):
            _frame_total_times.append(time.perf_counter() - _frame_started)
            rows.append(rec)
            continue

        _target_ap_started = time.perf_counter()
        phot_t = aperture_photometry(img, xt, yt, ap_radius, r_in, r_out)
        _target_ap_times.append(time.perf_counter() - _target_ap_started)
        if phot_t.get("ok") != 1 or not np.isfinite(phot_t.get("flux_net")):
            _frame_total_times.append(time.perf_counter() - _frame_started)
            rows.append(rec)
            continue

        sat_t = is_saturated(phot_t.get("max_pix", np.nan), cfg_obj.saturation_threshold)
        if sat_t and not cfg_obj.allow_saturated_target:
            _frame_total_times.append(time.perf_counter() - _frame_started)
            rows.append(rec)
            continue

        m_inst_t = m_inst_from_flux(phot_t["flux_net"])
        if not np.isfinite(m_inst_t):
            _frame_total_times.append(time.perf_counter() - _frame_started)
            rows.append(rec)
            continue

        # ── [2] Sharpness Index 篩選 ─────────────────────────────────────────
        # S = flux(r=3px) / flux(r=8px)（均扣背景，對象：最亮未飽和比較星）
        # S < sharpness_min 代表星點過度擴散或拖影，剔除該幀。
        # 注意：刻意用最亮比較星而非目標星 — 亮星 S/N 高，sharpness 判斷更穩定。
        _sharpness = np.nan
        _sharpness_started = time.perf_counter()
        _sharpness_min = float(getattr(cfg_obj, "sharpness_min", 0.3))
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
                _comp_ap_started = time.perf_counter()
                _s_phot8 = aperture_photometry(img, _s_xc, _s_yc, 8.0, r_in, r_out)
                _comp_ap_times.append(time.perf_counter() - _comp_ap_started)
                _comp_phot_calls += 1
                if is_saturated(_s_phot8.get("max_pix", np.nan), cfg_obj.saturation_threshold):
                    continue
                _s_bright_x, _s_bright_y = _s_xc, _s_yc
                _s_bright_mag = _s_m
            if _s_bright_x is not None:
                try:
                    _comp_ap_started = time.perf_counter()
                    _phot_s3 = aperture_photometry(img, _s_bright_x, _s_bright_y, 3.0, r_in, r_out)
                    _comp_ap_times.append(time.perf_counter() - _comp_ap_started)
                    _comp_phot_calls += 1
                    _comp_ap_started = time.perf_counter()
                    _phot_s8 = aperture_photometry(img, _s_bright_x, _s_bright_y, 8.0, r_in, r_out)
                    _comp_ap_times.append(time.perf_counter() - _comp_ap_started)
                    _comp_phot_calls += 1
                    if (_phot_s3.get("ok") == 1 and _phot_s8.get("ok") == 1
                            and np.isfinite(_phot_s3.get("flux_net", np.nan))
                            and np.isfinite(_phot_s8.get("flux_net", np.nan))
                            and _phot_s8["flux_net"] > 0):
                        _sharpness = float(_phot_s3["flux_net"] / _phot_s8["flux_net"])
                except Exception:
                    pass
        _sharpness_times.append(time.perf_counter() - _sharpness_started)
        rec["sharpness_index"] = _sharpness
        if np.isfinite(_sharpness) and _sharpness < _sharpness_min:
            rec["ok"] = 0
            rec["ok_flag"] = "low_sharpness"
            n_low_sharpness += 1
            _frame_total_times.append(time.perf_counter() - _frame_started)
            rows.append(rec)
            continue

        # ── [2b] Peak-ratio 篩選（次鏡起霧 / 甜甜圈 PSF 偵測）─────────────────
        # peak_ratio = t_max_pix / t_flux_net
        # 次鏡起霧時中心被掏空，峰值相對總通量驟降。
        # [DEPRECATED] peak_ratio_min: 固定門檻，已由自適應 peak_ratio_k 取代。
        # peak_ratio_min > 0 仍可運作但不建議使用。
        _peak_ratio_min = float(getattr(cfg_obj, "peak_ratio_min", 0.0))
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
            _frame_total_times.append(time.perf_counter() - _frame_started)
            rows.append(rec)
            continue

        # ── Comparison ensemble ──────────────────────────────────────────────
        comp_m_inst, comp_m_cat, comp_weights = [], [], []
        _comp_loop_started = time.perf_counter()
        for ref in comp_refs:
            ra_c, dec_c, m_cat = ref[0], ref[1], ref[2]
            m_err   = ref[3] if len(ref) > 3 else None
            # weight = 1 / (d + ε)²，由 auto_select_comps 計算並存入 ref[4]
            w_i     = float(ref[4]) if len(ref) > 4 else 1.0
            sc = SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg)
            if check_coord is not None and sc.separation(check_coord).arcsec <= cfg_obj.apass_match_arcsec:
                continue
            xc, yc = radec_to_pixel(wcs_obj, ra_c, dec_c)
            if not in_bounds(img, xc, yc, margin=margin):
                continue
            # 同視野多目標共用快取：命中則跳過重測
            _cached_phot = (
                shared_cache.get(f.stem, ra_c, dec_c, label=_cache_label)
                if shared_cache else None
            )
            if _cached_phot is not None:
                _cached_comp_hits += 1
                phot_c = _cached_phot
            else:
                _comp_ap_started = time.perf_counter()
                phot_c = aperture_photometry(img, xc, yc, ap_radius, r_in, r_out)
                _comp_ap_times.append(time.perf_counter() - _comp_ap_started)
                _comp_phot_calls += 1
                if shared_cache is not None:
                    shared_cache.set(f.stem, ra_c, dec_c, phot_c, label=_cache_label)
            if phot_c.get("ok") != 1 or not np.isfinite(phot_c.get("flux_net")):
                continue
            if is_saturated(phot_c.get("max_pix", np.nan), cfg_obj.saturation_threshold):
                continue
            m_inst_c = m_inst_from_flux(phot_c["flux_net"])
            if not np.isfinite(m_inst_c):
                continue
            dist_arcsec   = float(target_coord.separation(sc).arcsec)
            # ε = plate_scale / 2，防止距離趨近於零時權重爆炸
            epsilon       = cfg_obj.plate_scale_arcsec / 2.0
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
        _comp_loop_times.append(time.perf_counter() - _comp_loop_started)

        comp_m_inst  = np.asarray(comp_m_inst, dtype=float)
        comp_m_cat   = np.asarray(comp_m_cat,  dtype=float)

        if len(comp_m_inst) < cfg_obj.robust_regression_min_points:
            _frame_total_times.append(time.perf_counter() - _frame_started)
            rows.append(rec)
            continue

        _fit_started = time.perf_counter()
        fit = robust_linear_fit(
            comp_m_inst, comp_m_cat,
            sigma=cfg_obj.robust_regression_sigma,
            max_iter=cfg_obj.robust_regression_max_iter,
            min_points=cfg_obj.robust_regression_min_points,
            weights=np.asarray(comp_weights, dtype=float),
        )
        _fit_times.append(time.perf_counter() - _fit_started)

        # 捕捉第一幀資料供診斷圖使用（全部比較星 + 實際 fit）
        if _first_frame_diag_data is None and len(comp_m_cat) >= 2:
            _first_frame_diag_data = (
                comp_m_cat.copy(), comp_m_inst.copy(), fit,
                float(getattr(cfg_obj, "vmag_approx", np.nan)),
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
            _frame_total_times.append(time.perf_counter() - _frame_started)
            rows.append(rec)
            continue

        # ── [3] 回歸 R² 幀層級篩選 ────────────────────────────────────────────
        # reg_r2_min 預設 0.0（停用）；> 0 時才自動剔除，保留 WARN 輸出。
        _reg_r2_min = float(getattr(cfg_obj, "reg_r2_min", 0.0))
        if np.isfinite(r2) and _reg_r2_min > 0 and r2 < _reg_r2_min:
            _phot_logger.warning(
                "[WARN] low reg R2=%.4f < %.4f in %s (channel=%s)",
                r2, _reg_r2_min, f.name, channel,
            )
            rec["reg_r2"] = r2
            rec["ok"] = 0
            rec["ok_flag"] = "low_reg_r2"
            n_low_reg_r2 += 1
            _frame_total_times.append(time.perf_counter() - _frame_started)
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
            gain_e_per_adu=cfg_obj.gain_e_per_adu,
            read_noise_e=cfg_obj.read_noise_e,
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
        _abs_residuals = np.abs(_residuals)
        reg_resid_rms = float(np.sqrt(np.mean(_residuals ** 2))) if len(_residuals) > 0 else np.nan
        reg_abs_resid_med = float(np.median(_abs_residuals)) if len(_abs_residuals) > 0 else np.nan
        reg_abs_resid_p90 = float(np.percentile(_abs_residuals, 90)) if len(_abs_residuals) > 0 else np.nan
        reg_abs_resid_max = float(np.max(_abs_residuals)) if len(_abs_residuals) > 0 else np.nan
        rec["reg_residual_rms"] = reg_resid_rms
        rec["reg_abs_resid_med"] = reg_abs_resid_med
        rec["reg_abs_resid_p90"] = reg_abs_resid_p90
        rec["reg_abs_resid_max"] = reg_abs_resid_max

        # ── Check star ───────────────────────────────────────────────────────
        _check_total_started = time.perf_counter()
        if check_star is not None:
            ra_k, dec_k, m_cat_k = check_star
            xk, yk = radec_to_pixel(wcs_obj, ra_k, dec_k)
            if in_bounds(img, xk, yk, margin=margin):
                _check_ap_started = time.perf_counter()
                phot_k = aperture_photometry(img, xk, yk, ap_radius, r_in, r_out)
                _check_ap_times.append(time.perf_counter() - _check_ap_started)
                if phot_k.get("ok") == 1 and np.isfinite(phot_k.get("flux_net")):
                    sat_k = is_saturated(phot_k.get("max_pix", np.nan), cfg_obj.saturation_threshold)
                    if not sat_k or cfg_obj.allow_saturated_check:
                        m_inst_k = m_inst_from_flux(phot_k["flux_net"])
                        if np.isfinite(m_inst_k):
                            m_check = (m_inst_k - b) / a
                            rec["k_m_inst"] = m_inst_k
                            rec["k_m_var"]  = m_check
                            rec["k_m_cat"]  = m_cat_k
                            if m_cat_k is not None and np.isfinite(m_cat_k):
                                rec["k_minus_c"] = m_check - float(m_cat_k)
        _check_total_times.append(time.perf_counter() - _check_total_started)
        rows.append(rec)
        _frame_total_times.append(time.perf_counter() - _frame_started)

    _emit_progress_done(_phot_logger, f"channel {channel} frame loop", _frame_stage_started)
    _frame_total_arr = np.asarray(_frame_total_times, dtype=float)
    _target_ap_arr = np.asarray(_target_ap_times, dtype=float)
    _comp_ap_arr = np.asarray(_comp_ap_times, dtype=float)
    _check_ap_arr = np.asarray(_check_ap_times, dtype=float)
    _io_load_arr = np.asarray(_io_load_times, dtype=float)
    _time_wcs_arr = np.asarray(_time_wcs_times, dtype=float)
    _sharpness_arr = np.asarray(_sharpness_times, dtype=float)
    _comp_loop_arr = np.asarray(_comp_loop_times, dtype=float)
    _fit_arr = np.asarray(_fit_times, dtype=float)
    _check_total_arr = np.asarray(_check_total_times, dtype=float)
    _cache_stats = shared_cache.stats(_cache_label) if shared_cache is not None else {"hits": 0, "misses": 0, "sets": 0}
    _cache_lookups = int(_cache_stats["hits"]) + int(_cache_stats["misses"])
    _cache_hit_rate = (float(_cache_stats["hits"]) / _cache_lookups) if _cache_lookups > 0 else np.nan
    _phot_logger.info(
        "[timing] target=%s channel=%s total_frames=%d avg_sec_frame=%.3f median_sec_frame=%.3f "
        "target_ap_total=%.3f comp_ap_total=%.3f check_ap_total=%.3f comp_phot_calls=%d cached_comp_hits=%d",
        cfg_obj.target_name,
        channel,
        len(_frame_total_times),
        float(np.nanmean(_frame_total_arr)) if len(_frame_total_arr) > 0 else np.nan,
        float(np.nanmedian(_frame_total_arr)) if len(_frame_total_arr) > 0 else np.nan,
        float(np.nansum(_target_ap_arr)) if len(_target_ap_arr) > 0 else 0.0,
        float(np.nansum(_comp_ap_arr)) if len(_comp_ap_arr) > 0 else 0.0,
        float(np.nansum(_check_ap_arr)) if len(_check_ap_arr) > 0 else 0.0,
        int(_comp_phot_calls),
        int(_cached_comp_hits),
    )
    _phot_logger.info(
        "[timing_detail] target=%s channel=%s io_load_total=%.3f time_wcs_total=%.3f "
        "sharpness_total=%.3f comp_loop_total=%.3f fit_total=%.3f check_total=%.3f",
        cfg_obj.target_name,
        channel,
        float(np.nansum(_io_load_arr)) if len(_io_load_arr) > 0 else 0.0,
        float(np.nansum(_time_wcs_arr)) if len(_time_wcs_arr) > 0 else 0.0,
        float(np.nansum(_sharpness_arr)) if len(_sharpness_arr) > 0 else 0.0,
        float(np.nansum(_comp_loop_arr)) if len(_comp_loop_arr) > 0 else 0.0,
        float(np.nansum(_fit_arr)) if len(_fit_arr) > 0 else 0.0,
        float(np.nansum(_check_total_arr)) if len(_check_total_arr) > 0 else 0.0,
    )
    _phot_logger.info(
        "[shared_cache] target=%s channel=%s hits=%d misses=%d sets=%d hit_rate=%s",
        cfg_obj.target_name,
        channel,
        int(_cache_stats["hits"]),
        int(_cache_stats["misses"]),
        int(_cache_stats["sets"]),
        f"{_cache_hit_rate:.3f}" if np.isfinite(_cache_hit_rate) else "na",
    )
    _post_df_started = time.perf_counter()
    df = pd.DataFrame(rows)

    if len(df) == 0:
        print("[WARN] 所有幀均被跳過（airmass），無任何資料列。")
        df = pd.DataFrame(columns=["file", "mjd", "bjd_tdb", "airmass", "ok", "m_var",
                                    "m_var_norm", "delta_ensemble"])
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        return df, {}

    _dataframe_build_total = time.perf_counter() - _post_df_started
    _post_filter_started = time.perf_counter()
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
    # 2026-04-14: disabled by user decision.
    # Reason:
    # - current mainline already uses per-frame free-slope regression
    # - applying extinction here risks double-correcting atmospheric effects
    # Keep `_ext_k` for summary/report compatibility, but do not modify `m_var`.
    _ext_k = float(getattr(cfg_obj, "extinction_k", 0.0))
    # if _ext_k > 0 and "airmass" in df.columns:
    #     _ok_am = (df["ok"] == 1) & np.isfinite(df["airmass"]) & np.isfinite(df["m_var"])
    #     if _ok_am.sum() >= 3:
    #         _X_ref = float(df.loc[_ok_am, "airmass"].min())
    #         _delta_ext = _ext_k * (df["airmass"] - _X_ref)
    #         df["m_var_raw"] = df["m_var"].copy()
    #         df.loc[_ok_am, "m_var"] = df.loc[_ok_am, "m_var"] - _delta_ext[_ok_am]
    #         _med_corr = float(_delta_ext[_ok_am].median())
    #         print(f"[extinction] k={_ext_k:.3f} mag/airmass  X_ref={_X_ref:.3f}  "
    #               f"median correction={_med_corr:.4f} mag  ({_ok_am.sum()} frames)")

    # ── 回歸截距突變篩選 ───────────────────────────────────────────────────────
    # 薄雲或透明度驟變時所有比較星同步變暗，零點截距 b 會系統性漂移。
    # 滾動中位數（±5 幀）捕捉短時突變，偏離超過 reg_intercept_sigma × MAD 則剔除。
    n_reg_jump = 0
    _reg_intercept_sigma = float(getattr(cfg_obj, "reg_intercept_sigma", 0.0))
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

    # ── 回歸殘差散布篩選 ───────────────────────────────────────────────────────
    # 若只有少數比較星局部失真，截距未必大跳，但比較星殘差散布會變寬。
    # 這裡保留兩種 dispersion 指標：
    #   reg_residual_rms     : 全體 inlier residual 的 RMS
    #   reg_abs_resid_p90    : |residual| 的第 90 百分位，對少數壞比較星較敏感
    n_reg_resid_rms = 0
    _reg_residual_rms_sigma = float(getattr(cfg_obj, "reg_residual_rms_sigma", 0.0))
    if _reg_residual_rms_sigma > 0 and "reg_residual_rms" in df.columns:
        _rr_col = df["reg_residual_rms"].copy()
        _rr_roll_med = _rr_col.rolling(window=11, center=True, min_periods=3).median()
        _rr_resid = _rr_col - _rr_roll_med
        _rr_mad = float(np.nanmedian(_rr_resid[df["ok"] == 1].abs()))
        _rr_thresh = _reg_residual_rms_sigma * 1.4826 * _rr_mad if _rr_mad > 0 else np.inf
        _rr_mask = (df["ok"] == 1) & (_rr_resid > _rr_thresh)
        if _rr_mask.any():
            df.loc[_rr_mask, "ok"] = 0
            if "ok_flag" not in df.columns:
                df["ok_flag"] = ""
            df.loc[_rr_mask, "ok_flag"] = "reg_resid_rms"
            n_reg_resid_rms = int(_rr_mask.sum())
            print(f"[reg_resid_rms] MAD={_rr_mad:.4f}  thresh={_rr_thresh:.4f}  "
                  f"clipped {n_reg_resid_rms} frames")

    n_reg_resid_p90 = 0
    _reg_residual_p90_sigma = float(getattr(cfg_obj, "reg_residual_p90_sigma", 0.0))
    if _reg_residual_p90_sigma > 0 and "reg_abs_resid_p90" in df.columns:
        _rp_col = df["reg_abs_resid_p90"].copy()
        _rp_roll_med = _rp_col.rolling(window=11, center=True, min_periods=3).median()
        _rp_resid = _rp_col - _rp_roll_med
        _rp_mad = float(np.nanmedian(_rp_resid[df["ok"] == 1].abs()))
        _rp_thresh = _reg_residual_p90_sigma * 1.4826 * _rp_mad if _rp_mad > 0 else np.inf
        _rp_mask = (df["ok"] == 1) & (_rp_resid > _rp_thresh)
        if _rp_mask.any():
            df.loc[_rp_mask, "ok"] = 0
            if "ok_flag" not in df.columns:
                df["ok_flag"] = ""
            df.loc[_rp_mask, "ok_flag"] = "reg_resid_p90"
            n_reg_resid_p90 = int(_rp_mask.sum())
            print(f"[reg_resid_p90] MAD={_rp_mad:.4f}  thresh={_rp_thresh:.4f}  "
                  f"clipped {n_reg_resid_p90} frames")

    # ── 天空背景突升篩選 ──────────────────────────────────────────────────────
    # 起霧或散射光使目標孔徑背景環中位數升高，t_b_sky 突升可做為霧的早期指標。
    # 滾動中位數（±5 幀）捕捉短時突變，偏離超過 sky_sigma × MAD 則剔除。
    n_high_sky = 0
    _sky_sigma = float(getattr(cfg_obj, "sky_sigma", 0.0))
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
    _peak_ratio_k = float(getattr(cfg_obj, "peak_ratio_k", 0.0))
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

    _n_low_sharpness_val = n_low_sharpness
    _n_low_peak_ratio_val = n_low_peak_ratio
    _n_low_reg_r2_val     = n_low_reg_r2
    # 重新計算 _n_phot_fail：排除所有已命名篩選計數
    _n_qual_filtered = (_n_low_sharpness_val
                        + _n_low_peak_ratio_val + _n_low_reg_r2_val
                        + n_reg_jump + n_reg_resid_rms + n_reg_resid_p90
                        + n_high_sky + n_low_peak_ratio_adaptive)
    _n_phot_fail   = _n_in_df - _n_ok_final - _n_sigma_clip - _n_qual_filtered

    _sep = "-" * 68
    print(f"\n[剔除統計] 通道 {channel}  {getattr(cfg_obj, 'target_name', '')}  {str(out_csv.stem).split('_')[-1]}")
    print(_sep)
    print(f"  {'原因':<24} {'幀數':>6}    公式")
    print(_sep)
    print(f"  {'高氣團跳過':<24} {_n_alt_skip:>6}    airmass > {cfg_obj.alt_min_airmass:.2f}")
    print(f"  {'低 Sharpness 剔除':<24} {_n_low_sharpness_val:>6}    S=flux(r=3)/flux(r=8) < {float(getattr(cfg_obj,'sharpness_min',0.3)):.2f}")
    print(f"  {'低 Peak Ratio 剔除':<24} {_n_low_peak_ratio_val:>6}    peak/flux < {float(getattr(cfg_obj,'peak_ratio_min',0.0)):.3f} (0=停用)")
    print(f"  {'低 reg R2 剔除':<24} {_n_low_reg_r2_val:>6}    reg_r2 < {float(getattr(cfg_obj,'reg_r2_min',0.0)):.2f} (0=停用)")
    print(f"  {'孔徑/WCS/邊界失敗':<24} {_n_phot_fail:>6}    flux/位置無效")
    print(f"  {'sigma_clip':<24} {_n_sigma_clip:>6}    |m_var - median| > 3 * 1.4826 * MAD")
    print(f"  {'回歸截距突變':<24} {n_reg_jump:>6}    rolling median ± {float(getattr(cfg_obj,'reg_intercept_sigma',0.0)):.1f} MAD (0=停用)")
    print(f"  {'回歸殘差 RMS 突升':<24} {n_reg_resid_rms:>6}    rolling median + {float(getattr(cfg_obj,'reg_residual_rms_sigma',0.0)):.1f} MAD (0=停用)")
    print(f"  {'回歸殘差 P90 突升':<24} {n_reg_resid_p90:>6}    rolling median + {float(getattr(cfg_obj,'reg_residual_p90_sigma',0.0)):.1f} MAD (0=停用)")
    print(f"  {'天空背景突升':<24} {n_high_sky:>6}    rolling median + {float(getattr(cfg_obj,'sky_sigma',0.0)):.1f} MAD (0=停用)")
    print(f"  {'Peak Ratio 自適應':<24} {n_low_peak_ratio_adaptive:>6}    median - {float(getattr(cfg_obj,'peak_ratio_k',0.0)):.1f} MAD (0=停用)")
    print(_sep)
    print(f"  {'保留 (ok=1)':<24} {_n_ok_final:>6} / {_n_total_fits} 幀")
    print()

    # 存檔：帶時間戳，不覆蓋
    _phot_logger.info(
        "[post_timing] target=%s channel=%s dataframe_build_total=%.3f post_filter_total=%.3f",
        cfg_obj.target_name,
        channel,
        _dataframe_build_total,
        time.perf_counter() - _post_filter_started,
    )
    _rej_rows = [
        {"reason": "高氣團跳過",       "count": _n_alt_skip,            "threshold": f"airmass > {cfg_obj.alt_min_airmass:.2f}",                              "config_key": "max_airmass",         "config_value": cfg_obj.alt_min_airmass},
        {"reason": "低 Sharpness 剔除", "count": _n_low_sharpness_val,   "threshold": f"S < {float(getattr(cfg_obj,'sharpness_min',0.3)):.2f}",                 "config_key": "sharpness_min",       "config_value": float(getattr(cfg_obj, "sharpness_min", 0.3))},
        {"reason": "低 Peak Ratio 剔除","count": _n_low_peak_ratio_val,  "threshold": f"peak/flux < {float(getattr(cfg_obj,'peak_ratio_min',0.0)):.3f}",        "config_key": "peak_ratio_min",      "config_value": float(getattr(cfg_obj, "peak_ratio_min", 0.0))},
        {"reason": "低 reg R2 剔除",    "count": _n_low_reg_r2_val,       "threshold": f"reg_r2 < {float(getattr(cfg_obj,'reg_r2_min',0.0)):.2f}",                 "config_key": "reg_r2_min",           "config_value": float(getattr(cfg_obj, "reg_r2_min", 0.0))},
        {"reason": "孔徑/WCS/邊界失敗", "count": _n_phot_fail,           "threshold": "flux/位置無效",                                                     "config_key": "—",                   "config_value": "—"},
        {"reason": "sigma_clip",        "count": _n_sigma_clip,          "threshold": "|m_var - median| > 3 * 1.4826 * MAD",                               "config_key": "—",                   "config_value": "—"},
        {"reason": "回歸截距突變",       "count": n_reg_jump,              "threshold": f"rolling |reg_intercept - med| > {float(getattr(cfg_obj,'reg_intercept_sigma',0.0)):.1f} MAD", "config_key": "reg_intercept_sigma", "config_value": float(getattr(cfg_obj, "reg_intercept_sigma", 0.0))},
        {"reason": "回歸殘差 RMS 突升",  "count": n_reg_resid_rms,         "threshold": f"rolling (reg_residual_rms - med) > {float(getattr(cfg_obj,'reg_residual_rms_sigma',0.0)):.1f} MAD", "config_key": "reg_residual_rms_sigma", "config_value": float(getattr(cfg_obj, "reg_residual_rms_sigma", 0.0))},
        {"reason": "回歸殘差 P90 突升",  "count": n_reg_resid_p90,         "threshold": f"rolling (reg_abs_resid_p90 - med) > {float(getattr(cfg_obj,'reg_residual_p90_sigma',0.0)):.1f} MAD", "config_key": "reg_residual_p90_sigma", "config_value": float(getattr(cfg_obj, "reg_residual_p90_sigma", 0.0))},
        {"reason": "天空背景突升",       "count": n_high_sky,                    "threshold": f"rolling (t_b_sky - med) > {float(getattr(cfg_obj,'sky_sigma',0.0)):.1f} MAD",                     "config_key": "sky_sigma",           "config_value": float(getattr(cfg_obj, "sky_sigma", 0.0))},
        {"reason": "Peak Ratio 自適應", "count": n_low_peak_ratio_adaptive,    "threshold": f"peak_ratio < median - {float(getattr(cfg_obj,'peak_ratio_k',0.0)):.1f} * 1.4826 * MAD",           "config_key": "peak_ratio_k",        "config_value": float(getattr(cfg_obj, "peak_ratio_k", 0.0))},
        {"reason": "保留 (ok=1)",       "count": _n_ok_final,                   "threshold": f"/ {_n_total_fits} 幀",                                                                         "config_key": "—",                   "config_value": "—"},
    ]
    _rej_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _rej_stem = out_csv.stem  # e.g. photometry_G1_20251220
    _rej_path = out_csv.parent / f"rejection_stats_{_rej_stem}_{_rej_ts}.csv"
    pd.DataFrame(_rej_rows).to_csv(_rej_path, index=False, encoding="utf-8-sig")
    print(f"[剔除統計] saved → {_rej_path}")

    if not _emit_photometry_products(
        df=df,
        out_csv=out_csv,
        out_png=out_png,
        channel=channel,
        cfg=cfg_obj,
        time_key=time_key,
        check_star=check_star,
        ap_radius=ap_radius,
        r_in=r_in,
        r_out=r_out,
        n_skipped=n_skipped,
        n_before_clip=_n_before_clip,
        n_after_clip=_n_after_clip,
        ext_k=_ext_k,
        first_frame_diag_data=_first_frame_diag_data,
    ):
        return df, {}

    return df, {}  # comp_lightcurves 已隨 ensemble 停用，回傳空 dict


# ── 週期分析已移至 period_analysis.py（統一從 YAML 讀取參數）────────────────
# 原內建 LS + Fourier 程式碼已保留於 photometry_ls_legacy.py，供手動使用。
# 移出日期：2026-03-21


__all__ = [
    "_emit_photometry_products",
    "run_photometry_on_wcs_dir",
]
