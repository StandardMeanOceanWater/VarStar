# -*- coding: utf-8 -*-
"""
phot_ensemble.py
Ensemble normalization (Broeg 2005).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

_phot_logger = logging.getLogger("photometry")


def ensemble_normalize(
    df: pd.DataFrame,
    comp_lightcurves: "dict[str, pd.Series]",
    initial_weights: "dict[str, float]",
    time_key: str = "bjd_tdb",
    min_comp_stars: int = 3,
    max_iter: int = 10,
    convergence_tol: float = 1e-4,
) -> "tuple[pd.DataFrame, pd.Series]":
    """
    ⏸ 已停用：Broeg (2005) ensemble normalisation.

    理由：逐幀自由斜率回歸已完全消除逐幀大氣漂移，
         ensemble 正規化解決的是同一問題，二者不應疊加（可能引入雜訊）。
    待驗：確認移除後 R² 無退化。

    ─── 原始文件 ───

    每顆比較星 i 的儀器星等時間序列 m_i(t) 估計共同大氣漂移 Δ(t)，
    然後對目標星星等做修正：m_var_norm(t) = m_var(t) − Δ(t)。

    演算法
    ------
    1. 初始權重：以 initial_weights（距離 + 測光誤差複合）為種子。
    2. 迭代：
       a. 計算各比較星偏差 δ_i(t) = m_i(t) − median(m_i)
       b. 計算漂移 Δ(t) = Σ w_i δ_i(t) / Σ w_i（僅對各幀中有效的比較星加總）
       c. 扣除漂移後計算每顆比較星殘差 RMS
       d. 更新權重 w_i = 1 / RMS_i²（Broeg 2005 eq. 4）
       e. 收斂判斷：max |Δ_new(t) − Δ_old(t)| < convergence_tol
    3. 將最終 Δ(t) 寫回 df["delta_ensemble"]，修正值寫入 df["m_var_norm"]。

    Parameters
    ----------
    df               : run_photometry_on_wcs_dir 回傳的 DataFrame。
    comp_lightcurves : {star_id: Series(time_key → m_inst)}，每顆比較星的
                       儀器星等時間序列。NaN 表示該幀該星不可用。
    initial_weights  : {star_id: float}，初始權重種子。
    time_key         : 時間欄位名稱（"bjd_tdb" 或 "mjd"）。
    min_comp_stars   : 計算 Δ(t) 所需的最少可用比較星數；不足時 Δ(t) = NaN。
    max_iter         : Broeg 迭代上限。
    convergence_tol  : 連續兩次迭代 Δ(t) 最大差值的收斂閾值（mag）。

    Returns
    -------
    df_out  : 含 m_var_norm、delta_ensemble 欄位的 DataFrame（原始 df 的副本）。
    delta_t : Δ(t) 時間序列（pd.Series，index = time_key 的值）。

    References
    ----------
    Broeg, C., Fernandez, M., & Neuhauser, R. (2005). A new algorithm for
    differential photometry: Computing an optimum artificial comparison star.
    Astronomische Nachrichten, 326(2), 134-142.
    https://doi.org/10.1002/asna.200410350
    """
    if len(comp_lightcurves) < min_comp_stars:
        _phot_logger.warning(
            "[ensemble] 比較星數量 %d < min_comp_stars %d，跳過 ensemble 正規化。",
            len(comp_lightcurves), min_comp_stars,
        )
        df_out = df.copy()
        df_out["m_var_norm"] = df_out.get("m_var", np.nan)
        df_out["delta_ensemble"] = np.nan
        return df_out, pd.Series(dtype=float)

    star_ids = list(comp_lightcurves.keys())

    t_index = df[time_key].values
    m_matrix = pd.DataFrame(index=range(len(t_index)), columns=star_ids, dtype=float)
    for sid in star_ids:
        series = comp_lightcurves[sid]
        for row_i, t_val in enumerate(t_index):
            if np.isfinite(t_val) and t_val in series.index:
                m_matrix.loc[row_i, sid] = series[t_val]
            else:
                m_matrix.loc[row_i, sid] = np.nan
    m_arr = m_matrix.values.astype(float)   # shape: (n_frames, n_comp)

    med_i = np.nanmedian(m_arr, axis=0)     # shape: (n_comp,)

    w = np.array(
        [float(initial_weights.get(sid, 1.0)) for sid in star_ids],
        dtype=float,
    )
    _w_valid = w[np.isfinite(w) & (w > 0)]
    _w_fallback = float(np.median(_w_valid)) if len(_w_valid) > 0 else 1.0
    w = np.where(np.isfinite(w) & (w > 0), w, _w_fallback)

    delta_old = np.full(len(t_index), np.nan)

    for iteration in range(int(max_iter)):
        delta_new = np.full(len(t_index), np.nan)
        for row_i in range(len(t_index)):
            row = m_arr[row_i, :]
            valid = np.isfinite(row) & np.isfinite(med_i)
            if int(valid.sum()) < min_comp_stars:
                continue
            delta_i = row[valid] - med_i[valid]
            w_v = w[valid]
            delta_new[row_i] = float(np.sum(w_v * delta_i) / np.sum(w_v))

        both_finite = np.isfinite(delta_old) & np.isfinite(delta_new)
        if both_finite.any():
            max_diff = float(np.max(np.abs(delta_new[both_finite]
                                           - delta_old[both_finite])))
            if max_diff < convergence_tol:
                _phot_logger.debug(
                    "[ensemble] 第 %d 次迭代收斂（max|ΔΔ|=%.2e < tol=%.2e）。",
                    iteration + 1, max_diff, convergence_tol,
                )
                break

        delta_old = delta_new.copy()

        for ci, sid in enumerate(star_ids):
            col = m_arr[:, ci]
            finite_rows = np.isfinite(col) & np.isfinite(delta_new)
            if int(finite_rows.sum()) < 2:
                continue
            resid = col[finite_rows] - med_i[ci] - delta_new[finite_rows]
            rms_i = float(np.sqrt(np.mean(resid ** 2)))
            if rms_i > 0:
                w[ci] = 1.0 / rms_i ** 2

    df_out = df.copy()
    df_out["delta_ensemble"] = delta_new

    m_var_col = df_out.get("m_var") if "m_var" in df_out.columns else pd.Series(
        np.nan, index=df_out.index
    )
    _reg_slope = df_out["reg_slope"].values if "reg_slope" in df_out.columns else np.ones(len(df_out))
    _reg_slope_safe = np.where(np.isfinite(_reg_slope) & (_reg_slope != 0), _reg_slope, 1.0)
    delta_scaled = delta_new / _reg_slope_safe

    m_var_norm = np.where(
        np.isfinite(df_out["m_var"].values) & np.isfinite(delta_scaled),
        df_out["m_var"].values - delta_scaled,
        np.nan,
    )
    df_out["m_var_norm"] = m_var_norm

    finite_delta = delta_new[np.isfinite(delta_new)]
    if len(finite_delta) > 0:
        _phot_logger.info(
            "[ensemble] 完成。Δ(t) median=%.4f  rms=%.4f  有效幀數=%d/%d  比較星數=%d",
            float(np.median(finite_delta)),
            float(np.std(finite_delta)),
            int(np.isfinite(delta_new).sum()),
            len(delta_new),
            len(star_ids),
        )
        print(
            f"[ensemble] Δ(t) median={float(np.median(finite_delta)):.4f}  "
            f"rms={float(np.std(finite_delta)):.4f}  "
            f"有效幀={int(np.isfinite(delta_new).sum())}/{len(delta_new)}"
        )
    else:
        _phot_logger.warning("[ensemble] 無有效 Δ(t)，m_var_norm 全為 NaN。")

    delta_series = pd.Series(delta_new, index=t_index, name="delta_ensemble")
    return df_out, delta_series
