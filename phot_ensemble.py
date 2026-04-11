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
    Broeg (2005) ensemble normalisation.
    """
    if len(comp_lightcurves) < min_comp_stars:
        _phot_logger.warning(
            "[ensemble] 瘥????%d < min_comp_stars %d嚗歲??ensemble 甇????",
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
    m_arr = m_matrix.values.astype(float)

    med_i = np.nanmedian(m_arr, axis=0)

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
                    "[ensemble] 蝚?%d 甈∟翮隞???max|??|=%.2e < tol=%.2e嚗?",
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

    df_out["m_var_norm"] = m_var_col - delta_scaled
    return df_out, pd.Series(delta_new, index=df_out[time_key])
