# -*- coding: utf-8 -*-
"""
phot_regression.py — 回歸校正與誤差模型（純數學函式，無全域狀態依賴）

從 photometry.py 拆出。包含：
  - robust_linear_fit   (穩健線性回歸 m_inst = a·m_cat + b)
  - mag_error_from_flux  (CCD 噪聲方程式，Merline & Howell 1995)
  - differential_mag     (單比較星差分測光，已停用)
"""

import logging

import numpy as np

_phot_logger = logging.getLogger("photometry")


# ── 穩健線性回歸 ──────────────────────────────────────────────────────────────

def robust_linear_fit(
    m_inst: np.ndarray,
    m_cat: np.ndarray,
    sigma: float = 3.0,
    max_iter: int = 5,
    min_points: int = 3,
    weights: "np.ndarray | None" = None,
) -> "dict | None":
    """
    Iteratively re-weighted linear regression: m_inst = a * m_cat + b.

    Returns
    -------
    dict with keys:
        a        – slope
        b        – intercept
        r2       – coefficient of determination (R²) on the inlier subset
        mask     – boolean array marking inliers used in the final fit
    or None if fewer than min_points finite values exist.

    Physical note
    -------------
    In single-night differential photometry the slope 'a' should be close
    to 1.0.  Significant deviation indicates colour-dependent atmospheric
    extinction or a poorly-matched comparison ensemble; investigate before
    accepting the result.  R² is provided so the caller can flag low-quality
    regression solutions (R² < 0.95 warrants inspection).
    """
    m_inst = np.asarray(m_inst, dtype=float)
    m_cat  = np.asarray(m_cat,  dtype=float)

    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape != m_inst.shape:
            w = None

    mask = np.isfinite(m_inst) & np.isfinite(m_cat)
    if w is not None:
        mask = mask & np.isfinite(w) & (w > 0)
    if int(mask.sum()) < int(min_points):
        return None

    def _fit(msk):
        """Huber 強壯回歸（epsilon=1.35）；失敗時退回加權 OLS。"""
        x_fit = m_cat[msk]
        y_fit = m_inst[msk]
        sample_w = np.sqrt(w[msk]) if w is not None else None
        try:
            from sklearn.linear_model import HuberRegressor
            X = x_fit.reshape(-1, 1)
            hr = HuberRegressor(epsilon=1.35, max_iter=200)
            if sample_w is not None:
                hr.fit(X, y_fit, sample_weight=sample_w)
            else:
                hr.fit(X, y_fit)
            return float(hr.coef_[0]), float(hr.intercept_)
        except Exception:
            # sklearn 不可用時退回 polyfit（原始 OLS）
            if sample_w is not None:
                return np.polyfit(x_fit, y_fit, 1, w=sample_w)
            return np.polyfit(x_fit, y_fit, 1)

    a, b = _fit(mask)
    for _ in range(int(max_iter)):
        resid   = m_inst - (a * m_cat + b)
        std     = float(np.nanstd(resid[mask])) if np.any(mask) else np.nan
        if not np.isfinite(std) or std == 0:
            break
        new_mask = mask & (np.abs(resid) <= sigma * std)
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask
        if int(mask.sum()) < int(min_points):
            break
        a, b = _fit(mask)

    # ── R² on the inlier subset ───────────────────────────────────────────────
    y_true = m_inst[mask]
    y_pred = a * m_cat[mask] + b
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    if r2 < 0.95:
        _phot_logger.debug("[WARN] Zero-point fit R²=%.4f < 0.95. "
                           "Check comparison star quality or airmass spread.", r2)

    return {"a": float(a), "b": float(b), "r2": float(r2), "mask": mask}


# ── CCD 噪聲方程式 ───────────────────────────────────────────────────────────

def mag_error_from_flux(
    flux_net: float,
    b_sky_std: float,
    n_pix: int,
    gain_e_per_adu: "float | None",
    read_noise_e: "float | None",
    n_sky: "int | None" = None,
) -> float:
    """CCD noise equation (Merline & Howell, 1995).

    完整公式（DESIGN_DECISIONS_v6.md §3.6.5）：

        sigma_flux² = N_pix × [S/G + b_sky_std² × (1 + N_pix/N_sky)]
        sigma_mag   = (2.5/ln10) × (sigma_flux / flux_net)

    Parameters
    ----------
    flux_net      : 孔徑淨通量（DN）= 孔徑總和 - N_pix × b_sky_median
    b_sky_std     : 背景環像素標準差（DN/px）——必須用標準差，非中位數
    n_pix         : 孔徑像素數
    gain_e_per_adu: gain（e-/DN）
    read_noise_e  : 讀出噪聲（e-）
    n_sky         : 背景環像素數；None 時退化為 sky_correction=1（舊式下限）

    Notes
    -----
    v5 之前的 bug：誤用 b_sky（中位數）代入噪聲項，導致 sigma_mag 偏高 5-7 倍。
    此版本已修正為使用 b_sky_std。
    """
    if gain_e_per_adu is None:
        return np.nan
    if not (np.isfinite(flux_net) and flux_net > 0):
        return np.nan
    if n_pix is None or n_pix <= 0:
        return np.nan

    gain = float(gain_e_per_adu)
    rn = float(read_noise_e) if read_noise_e is not None else 0.0

    # 訊號項（e-）
    f_e = flux_net * gain
    if f_e <= 0:
        return np.nan

    # 背景噪聲修正因子 (1 + N_pix/N_sky)；n_sky=None 時因子退化為 1.0
    if n_sky is not None and n_sky > 0:
        sky_correction = 1.0 + n_pix / float(n_sky)
    else:
        sky_correction = 1.0

    # 背景標準差換算為 e-/px
    bstd_e = float(b_sky_std) * gain if np.isfinite(b_sky_std) else 0.0

    # 完整噪聲（e-）
    sigma_flux_e = np.sqrt(
        max(f_e, 0.0)
        + n_pix * sky_correction * (max(bstd_e ** 2, 0.0) + rn ** 2)
    )

    return float(1.0857 * (sigma_flux_e / f_e))


# ── 差分測光（已停用）────────────────────────────────────────────────────────

def differential_mag(m_var_inst: float, m_ref_inst: float, m_ref_cat: float) -> float:
    """
    ⏸ 已停用：differential_mag 為舊式單比較星差分測光。
    m_var = m_var_inst - (m_ref_inst - m_ref_cat)

    應用場景：無回歸時的手工差分校正。
    目前不用，因逐幀自由斜率回歸已足夠。
    """
    if not np.isfinite(m_var_inst) or not np.isfinite(m_ref_inst):
        return np.nan
    return float(m_var_inst - (m_ref_inst - m_ref_cat))
