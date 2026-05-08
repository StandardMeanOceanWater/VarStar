"""Stage4 light-curve and period-analysis helpers."""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from phot_sources.logging_utils import emit_progress, emit_progress_done
from polt_light_curve import plot_light_curve

_phot_logger = logging.getLogger("photometry")


def _stage4_replot_shared_g1g2_ylim(cfg, active_date, channel_results):
    # ── G1/G2 共享 Y 軸重繪 ───────────────────────────────────────────
    _shared_chs = [c for c in ("G1", "G2") if c in channel_results]
    if len(_shared_chs) == 2:
        _all_ok_mag = []
        for _sc in _shared_chs:
            _sdf = channel_results[_sc]
            _sdf_ok = _sdf[(_sdf["ok"] == 1) & np.isfinite(_sdf["m_var"])]
            _all_ok_mag.extend(_sdf_ok["m_var"].tolist())
        if _all_ok_mag:
            _margin = 0.02
            _ymin = min(_all_ok_mag) - _margin   # 亮端（小數值）
            _ymax = max(_all_ok_mag) + _margin   # 暗端（大數值）
            _shared_ylim = (_ymax, _ymin)         # invert: 大值在下(bottom)
            for _sc in _shared_chs:
                # 直接寫回原 run_root，不重建 cfg（避免產生新時間戳目錄）
                _sc_png = cfg.run_root / "3_light_curve" / f"light_curve_{_sc}_{active_date}.png"
                plot_light_curve(channel_results[_sc], _sc_png, _sc,
                                cfg, obs_date=active_date, ylim=_shared_ylim)
            print(f"[PLOT] G1/G2 Y 軸已鎖定：{_ymin:.3f} – {_ymax:.3f} mag")


def _stage4_run_period_analysis(active_target, channel_results, stage4_run_root):
    # ── 週期分析（統一使用 period_analysis.py）─────────────────────────
    from period_analysis import run_period_analysis

    _pa_any = False
    for _ls_ch, _ls_df in channel_results.items():
        if _ls_df is None:
            continue
        _n_ok = int(_ls_df["ok"].sum())
        if _n_ok < 10:
            print(f"[LS] {_ls_ch} 有效幀數 {_n_ok} < 10，跳過")
            continue
        print(f"\n[LS] 通道 {_ls_ch}（有效幀數：{_n_ok}）")
        try:
            _pa_dir = stage4_run_root / "4_period_analysis"
            # 準備 DataFrame 欄位名稱對應（period_analysis 期望 ok, bjd_tdb, m_var/m_var_norm, v_err）
            _pa_df = _ls_df.copy()
            if "v_err" not in _pa_df.columns and "t_sigma_mag" in _pa_df.columns:
                _pa_df["v_err"] = _pa_df["t_sigma_mag"]
            _pa_result = run_period_analysis(
                _pa_df,
                target_name=active_target,
                channel=_ls_ch,
                out_dir=_pa_dir,
            )
            if _pa_result:
                _pa_any = True
                _ls_r = _pa_result.get("ls_result", {})
                _bp = _ls_r.get("best_period", np.nan)
                _fap = _ls_r.get("fap", np.nan)
                if np.isfinite(_bp):
                    print(f"[LS] Best period = {_bp:.6f} d  ({_bp * 24:.4f} h)")
                    print(f"[LS] FAP         = {_fap:.2e}")
                else:
                    print(f"[LS] {_ls_ch} 週期分析無有效結果（keys: {list(_pa_result.keys())}）")
                _fit_r = _pa_result.get("fit_result", {})
                if _fit_r:
                    _amp = _fit_r.get("amplitude", np.nan)
                    print(f"[Fourier] Amplitude = {_amp:.4f} mag")
                elif _ls_r:
                    print(f"[Fourier] 擬合失敗或跳過")
            else:
                print(f"[LS] {_ls_ch} run_period_analysis 回傳空結果")
        except Exception as _e_ls:
            print(f"[LS] {_ls_ch} 失敗：{_e_ls}")

    if not _pa_any:
        print("[LS] 跳過（所有通道有效幀數 < 10 或分析失敗）")


def _stage4_run_g1g2_ratio_products(cfg, active_target, active_date, channel_results, stage4_run_root):
    from period_analysis import run_period_analysis

    _dg1 = channel_results.get("G1")
    _dg2 = channel_results.get("G2")
    if _dg1 is None or _dg2 is None:
        return

    _need_cols = ["bjd_tdb", "t_flux_net", "m_var", "ok"]
    if not all(_c in _dg1.columns for _c in _need_cols) or not all(
        _c in _dg2.columns for _c in _need_cols
    ):
        print("[G1/G2] skip ratio products: missing required columns")
        return

    _mg = pd.merge(
        _dg1[_need_cols].rename(
            columns={"t_flux_net": "flux_G1", "m_var": "m_G1", "ok": "ok_G1"}
        ),
        _dg2[_need_cols].rename(
            columns={"t_flux_net": "flux_G2", "m_var": "m_G2", "ok": "ok_G2"}
        ),
        on="bjd_tdb",
        how="inner",
    )
    _mg = _mg[(_mg["ok_G1"] == 1) & (_mg["ok_G2"] == 1)].copy()
    if len(_mg) < 3:
        print(f"[G1/G2] skip ratio products: matched_ok={len(_mg)} < 3")
        return

    _mg["flux_ratio_G1G2"] = _mg["flux_G1"] / _mg["flux_G2"].replace(0, np.nan)
    _mg["mag_diff_G1G2"] = _mg["m_G1"] - _mg["m_G2"]

    _ratio_csv = stage4_run_root / "1_photometry" / f"G1G2_ratio_{active_date}.csv"
    _mg[
        ["bjd_tdb", "flux_ratio_G1G2", "mag_diff_G1G2", "flux_G1", "flux_G2", "m_G1", "m_G2"]
    ].to_csv(_ratio_csv, index=False, float_format="%.8f")

    _med_ratio = float(np.nanmedian(_mg["flux_ratio_G1G2"]))
    _med_mag = float(np.nanmedian(_mg["mag_diff_G1G2"]))
    _rms_mag = float(np.sqrt(np.nanmean((_mg["mag_diff_G1G2"] - _med_mag) ** 2)))
    print(
        f"[G1/G2] ratio csv={_ratio_csv} rows={len(_mg)} "
        f"median_flux_ratio={_med_ratio:.4f} "
        f"median_mag_diff={_med_mag:.4f} rms_mag_diff={_rms_mag:.4f}"
    )

    if len(_mg) < 10:
        return

    try:
        _ratio_df = pd.DataFrame(
            {
                "bjd_tdb": _mg["bjd_tdb"].values,
                "m_var": _mg["mag_diff_G1G2"].values,
                "v_err": np.full(len(_mg), 0.001),
                "ok": 1,
            }
        )
        _pa_dir_r = stage4_run_root / "4_period_analysis"
        run_period_analysis(
            _ratio_df,
            target_name=active_target,
            channel="G1G2ratio",
            out_dir=_pa_dir_r,
        )
        print("[G1/G2] ratio period analysis done")
    except Exception as _e_ratio:
        print(f"[G1/G2] ratio period analysis failed: {_e_ratio}")


def _run_stage4_postprocess(cfg, active_target, active_date, channel_results, stage4_run_root):
    _stage4_started = time.perf_counter()
    emit_progress(_phot_logger, f"stage4 postprocess start target={active_target} date={active_date}")
    _stage4_replot_shared_g1g2_ylim(cfg, active_date, channel_results)
    _stage4_run_period_analysis(active_target, channel_results, stage4_run_root)
    _stage4_run_g1g2_ratio_products(
        cfg, active_target, active_date, channel_results, stage4_run_root
    )

    emit_progress_done(_phot_logger, f"stage4 postprocess target={active_target} date={active_date}", _stage4_started)
    print(f"\n[完成] {active_target} {active_date} 全部通道週期分析完成")


__all__ = [
    "_run_stage4_postprocess",
    "_stage4_replot_shared_g1g2_ylim",
    "_stage4_run_g1g2_ratio_products",
    "_stage4_run_period_analysis",
]
