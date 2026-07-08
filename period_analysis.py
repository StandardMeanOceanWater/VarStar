# -*- coding: utf-8 -*-
"""
period_analysis.py — 進階週期分析模組（使用者選用）

定位
----
本模組為 photometry.py 標準輸出之外的選用進階分析工具。
管線標準週期輸出（LS + 相位折疊圖）由 photometry.py 內建，不受本模組控制。

學理基礎
--------
1. Lomb-Scargle Periodogram（Lomb, 1976; Scargle, 1982; VanderPlas, 2018）
   處理非均勻時間採樣的最佳估計法。
2. Discrete Fourier Transform（Lenz & Breger, 2005 — Period04）
   作為 LS 的交叉驗證。
3. FAP Bootstrap 收斂迭代（本實作）
   打亂時間序列重算 LS，迭代至收斂（相鄰 100 次視窗相對變化 < 5%）
   或達硬性上限 1000 次，取先到者。
4. 傅立葉擬合——BIC 最小化選階（本實作）
   BIC(N) = k*ln(n) - 2*ln(L_hat)，k=2N+1，自動選取最佳諧波數。
5. 週期不確定度下限（Kovacs, 1981）
   sigma_f = sigma_res / (T * sqrt(N))，sigma_P = sigma_f / f^2。
   sigma_res 使用傅立葉擬合殘差（非 LS 殘差）。
6. 相位零點兩步迭代 + 驗證（Breger et al.）
   以亮度極大值（mag 最小值）為 phi=0；驗證失敗時 raise ValueError。
7. 預白化（Breger et al., 1993）
   殘差 S/N < 4.0 停止；輸出每次迭代診斷圖；CSV 預留但停用。

使用方式
--------
    from period_analysis import run_period_analysis
    results = run_period_analysis(df, target_name="AlVel", channel="R",
                                  out_dir=Path("output/period_analysis"),
                                  config_path=Path("observation_config.yaml"))
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

# =============================================================================
# 設定讀取
# =============================================================================

_DEFAULT_CONFIG: Dict = {
    "period_analysis": {
        "lomb_scargle": {
            "normalization": "standard",
            "period_min_hr": 0.5,
            "period_max_hours": 8.0,
            "detrend_order": 1,
            "oversampling": 10,
            "fap_method": "bootstrap",
            "fap_bootstrap_max_iter": 1000,
            "fap_bootstrap_converge_window": 100,
            "fap_bootstrap_converge_tol": 0.05,
            "fap_threshold": 0.001,
        },
        "fourier_fit": {
            "max_harmonics": 8,
            "model_selection": "BIC",
            "min_data_points": 50,
            "sn_threshold": 4.0,
        },
        "pre_whitening": {
            "enabled": True,
            "sn_threshold": 4.0,
            "max_frequencies": 10,
            "save_csv": True,
        },
    },
    "output": {
        "plots": {
            "dpi": 150,
            "figsize_periodogram": [10, 4],
            "figsize_phase_folded": [8, 5],
            "figsize_prewhitening": [15, 4],
        },
    },
}


def _load_config(config_path: Optional[Path]) -> Dict:
    """
    讀取 observation_config.yaml（統一使用 pipeline_config.py）。
    若路徑不存在或解析失敗，回退至預設值並記錄警告。
    """
    try:
        from pipeline_config import load_pipeline_config
        return load_pipeline_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("設定載入失敗：%s，使用預設參數。", exc)
        return _DEFAULT_CONFIG


def _get(cfg: Dict, *keys, default=None):
    """安全地從巢狀 dict 取值。"""
    node = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
        if node is default:
            return default
    return node


# =============================================================================
# 數學模型
# =============================================================================

def _fourier_series(phase: np.ndarray, *params) -> np.ndarray:
    """
    傅立葉級數模型：
        V(phi) = a0 + sum_n [ a_n cos(2*pi*n*phi) + b_n sin(2*pi*n*phi) ]

    Parameters
    ----------
    phase  : 相位陣列，值域 [0, 1)。
    params : (a0, a1, b1, a2, b2, …)，長度 = 2N+1。
    """
    a0 = params[0]
    result = np.full_like(phase, a0, dtype=float)
    n_harmonics = (len(params) - 1) // 2
    for i in range(n_harmonics):
        n = i + 1
        a_n = params[1 + 2 * i]
        b_n = params[2 + 2 * i]
        angle = 2.0 * np.pi * n * phase
        result += a_n * np.cos(angle) + b_n * np.sin(angle)
    return result


def _fit_fourier(
    phase: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    n_harmonics: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    對指定諧波數進行傅立葉擬合，回傳 (popt, pcov)。

    Raises
    ------
    RuntimeError
        curve_fit 未能收斂。
    """
    p0 = [float(np.mean(mag))] + [0.0] * (2 * n_harmonics)
    sigma = err if (np.isfinite(err).all() and np.any(err > 0)) else None
    popt, pcov = curve_fit(
        _fourier_series,
        phase,
        mag,
        p0=p0,
        sigma=sigma,
        absolute_sigma=True,
        maxfev=100_000,
    )
    return popt, pcov


def _compute_bic(
    mag: np.ndarray,
    mag_pred: np.ndarray,
    n_params: int,
    err: np.ndarray,
) -> float:
    """
    計算 BIC（貝葉斯資訊準則）。

    高斯誤差假設下：
        ln(L_hat) = -0.5 * sum [ ((m_i - m_pred_i) / sigma_i)^2 ]
        BIC = k * ln(n) - 2 * ln(L_hat)

    Parameters
    ----------
    mag      : 觀測星等陣列。
    mag_pred : 傅立葉模型預測值。
    n_params : 模型參數數量（= 2N+1）。
    err      : 測光誤差陣列。
    """
    n = len(mag)
    sigma = err if (np.isfinite(err).all() and np.any(err > 0)) else np.ones(n)
    weighted_ssr = float(np.sum(((mag - mag_pred) / sigma) ** 2))
    log_likelihood = -0.5 * weighted_ssr
    bic = n_params * np.log(n) - 2.0 * log_likelihood
    return bic


def _select_harmonics_breger(
    phase: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    max_harmonics: int,
    sn_threshold: float = 4.0,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    以 Breger et al. (1993) S/N 準則選取傅立葉諧波數。

    逐階遞增：若第 N 諧波振幅的 S/N >= sn_threshold 則接受，否則停止。
    S/N = A_N / (σ_res × sqrt(2/n))
        A_N      = sqrt(a_N² + b_N²)，第 N 諧波振幅
        σ_res    = 加入第 N 諧波後的殘差 RMS
        sqrt(2/n) = DFT 振幅雜訊等級的近似（n = 資料點數）

    BIC 在 σ 很小（高精度測光）時無法正確剔刀（penalty << chi²），
    故改用此天文學標準準則。max_harmonics 仍作為硬上限。

    Returns
    -------
    (best_n, best_popt, best_pcov)
    """
    n_data = len(mag)
    best_n = 1
    best_popt: np.ndarray = np.array([])
    best_pcov: np.ndarray = np.array([])

    for n in range(1, max_harmonics + 1):
        try:
            popt, pcov = _fit_fourier(phase, mag, err, n)
        except (RuntimeError, ValueError) as exc:
            logger.debug("  N=%d 擬合失敗：%s", n, exc)
            break

        mag_pred = _fourier_series(phase, *popt)
        residuals = mag - mag_pred
        sigma_res = float(np.std(residuals))

        # 第 N 諧波振幅
        a_n = popt[1 + 2 * (n - 1)]
        b_n = popt[2 + 2 * (n - 1)]
        amp_n = float(np.sqrt(a_n ** 2 + b_n ** 2))

        noise_level = sigma_res * np.sqrt(2.0 / n_data) if n_data > 0 else np.inf
        sn = amp_n / noise_level if noise_level > 0 else 0.0

        logger.debug("  N=%d  A=%.4f  σ_res=%.4f  S/N=%.2f", n, amp_n, sigma_res, sn)

        if sn >= sn_threshold or n == 1:   # N=1 永遠接受，不能有 0 諧波
            best_n = n
            best_popt = popt
            best_pcov = pcov
        else:
            logger.info("Breger 準則停止：N=%d S/N=%.2f < %.1f，採用 N=%d", n, sn, sn_threshold, best_n)
            break
    else:
        logger.info("Breger 準則：已達上限 N=%d，採用 N=%d", max_harmonics, best_n)

    if best_popt.size == 0:
        raise RuntimeError("所有諧波數均擬合失敗，無法選取最佳階數。")

    logger.info("Breger 選階完成：N=%d（上限 %d，S/N 閾值 %.1f）", best_n, max_harmonics, sn_threshold)
    return best_n, best_popt, best_pcov


def classical_dft_amplitude(
    t: np.ndarray,
    mag: np.ndarray,
    freqs: np.ndarray,
) -> np.ndarray:
    """
    計算經典 DFT 振幅譜（Lenz & Breger, 2005）。

        A(v) = (2/N) * | sum_i (m_i - <m>) * exp(-i 2pi v t_i) |
    """
    mag_norm = mag - np.mean(mag)
    n = len(t)
    phase_matrix = -2.0j * np.pi * t[:, None] * freqs[None, :]
    dft = np.sum(mag_norm[:, None] * np.exp(phase_matrix), axis=0)
    return (2.0 / n) * np.abs(dft)


# =============================================================================
# FAP Bootstrap 收斂迭代
# =============================================================================

def _bootstrap_fap(
    ls: LombScargle,
    t: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    peak_power: float,
    freqs: np.ndarray,
    max_iter: int = 1000,
    converge_window: int = 100,
    converge_tol: float = 0.05,
) -> Tuple[float, int, str]:
    """
    Bootstrap FAP 收斂迭代。

    演算法
    ------
    1. 打亂 mag（保留 t 不變），重新計算 LS 最大功率值。
    2. 統計隨機最大功率 ≥ peak_power 的比例 → 當前 FAP 估計。
    3. 每 converge_window 次計算一次當前估計，與上一視窗比較。
       若相對變化 < converge_tol（5%），視為收斂，提前終止。
    4. 達到 max_iter 無論如何停止。

    Parameters
    ----------
    ls           : 已初始化的 LombScargle 物件（僅用於頻率格網，不重用）。
    t            : 時間陣列。
    mag          : 星等陣列（未打亂）。
    err          : 誤差陣列。
    peak_power   : 原始 LS 最大功率值。
    freqs        : 與原始 LS 相同的頻率格網。
    max_iter     : 硬性迭代上限。
    converge_window : 每隔此次數檢查一次收斂。
    converge_tol : 相鄰視窗 FAP 相對變化閾值。

    Returns
    -------
    (fap, n_iter, converge_status)
        fap            : 最終 FAP 估計值。
        n_iter         : 實際迭代次數。
        converge_status: "converged" 或 "max_iter"。
    """
    rng = np.random.default_rng(seed=42)
    exceed_count = 0
    fap_prev_window: Optional[float] = None
    converge_status = "max_iter"

    logger.info(
        "Bootstrap FAP 迭代開始（上限 %d 次，收斂視窗 %d，容差 %.1f%%）…",
        max_iter, converge_window, converge_tol * 100,
    )

    for i in range(1, max_iter + 1):
        mag_shuffled = rng.permutation(mag)
        ls_boot = LombScargle(t, mag_shuffled, dy=err)
        _, power_boot = ls_boot.autopower(
            minimum_frequency=freqs[0],
            maximum_frequency=freqs[-1],
            samples_per_peak=1,   # 不需精細格網，只要找最大值
        )
        if float(np.max(power_boot)) >= peak_power:
            exceed_count += 1

        # 每 converge_window 次檢查一次收斂
        if i % converge_window == 0:
            fap_current = exceed_count / i
            logger.info(
                "  [Bootstrap] iter=%d  FAP=%.4e  超過次數=%d",
                i, fap_current, exceed_count,
            )
            if fap_prev_window is not None and fap_prev_window > 0:
                rel_change = abs(fap_current - fap_prev_window) / fap_prev_window
                if rel_change < converge_tol:
                    converge_status = "converged"
                    logger.info(
                        "  [Bootstrap] 收斂（相對變化 %.2f%% < %.1f%%），"
                        "提前停止於第 %d 次迭代。",
                        rel_change * 100, converge_tol * 100, i,
                    )
                    return float(fap_current), i, converge_status
            fap_prev_window = fap_current

    fap_final = exceed_count / max_iter
    logger.info(
        "[Bootstrap] 達到上限 %d 次，最終 FAP=%.4e（狀態：%s）",
        max_iter, fap_final, converge_status,
    )
    return float(fap_final), max_iter, converge_status


# =============================================================================
# Lomb-Scargle + DFT
# =============================================================================

def run_ls_and_dft(
    t: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    cfg: Dict,
) -> Dict:
    """
    執行 Lomb-Scargle 週期圖與 DFT 交叉驗證。

    Parameters
    ----------
    t   : BJD_TDB 時間陣列（days）。
    mag : 差分星等陣列。
    err : 測光誤差陣列。
    cfg : 從 yaml 讀取的完整設定 dict。

    Returns
    -------
    dict 包含：
        freqs, ls_power, dft_amp, best_freq, best_period,
        fap, fap_n_iter, fap_status, peak_power
    """
    ls_cfg = _get(cfg, "period_analysis", "lomb_scargle", default={})
    period_min_days = _get(ls_cfg, "period_min_hr", default=0.5) / 24.0
    period_max_days = float(_get(ls_cfg, "period_max_hours", default=8.0)) / 24.0
    oversampling = _get(ls_cfg, "oversampling", default=10)
    max_iter = _get(ls_cfg, "fap_bootstrap_max_iter", default=1000)
    converge_window = _get(ls_cfg, "fap_bootstrap_converge_window", default=100)
    converge_tol = _get(ls_cfg, "fap_bootstrap_converge_tol", default=0.05)
    fap_threshold = _get(ls_cfg, "fap_threshold", default=0.001)

    min_freq = 1.0 / period_max_days
    max_freq = 1.0 / period_min_days

    # ── 趨勢扣除（線性 detrend，去除大氣漂移） ────────────────────────────
    # 基線 < 2× 搜尋範圍內最長週期時自動關閉：此時線性趨勢與訊號本身
    # 簡併，detrend 會咬掉訊號並把峰值往短週期推
    detrend_order = int(_get(cfg, "period_analysis", "lomb_scargle", "detrend_order", default=0))
    _baseline_d = float(np.max(t) - np.min(t))
    if detrend_order > 0 and _baseline_d < 2.0 * period_max_days:
        logger.info(
            "[detrend] 基線 %.3f d < 2×period_max %.3f d，自動關閉趨勢扣除。",
            _baseline_d, period_max_days,
        )
        detrend_order = 0
    if detrend_order > 0:
        _t_c = t - np.mean(t)
        _poly = np.polyfit(_t_c, mag, detrend_order)
        mag = mag - np.polyval(_poly, _t_c)

    ls = LombScargle(t, mag, dy=err)
    freqs, ls_power = ls.autopower(
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
        samples_per_peak=oversampling,
    )

    best_idx = int(np.argmax(ls_power))
    best_freq = float(freqs[best_idx])
    best_period = 1.0 / best_freq
    peak_power = float(ls_power[best_idx])

    logger.info(
        "[LS] 最佳週期 = %.6f d  (%.4f hr)  峰值功率 = %.4f",
        best_period, best_period * 24.0, peak_power,
    )

    # Bootstrap FAP 收斂迭代
    fap, fap_n_iter, fap_status = _bootstrap_fap(
        ls=ls,
        t=t,
        mag=mag,
        err=err,
        peak_power=peak_power,
        freqs=freqs,
        max_iter=max_iter,
        converge_window=converge_window,
        converge_tol=converge_tol,
    )

    logger.info(
        "[LS] FAP = %.4e  迭代 %d 次  狀態：%s",
        fap, fap_n_iter, fap_status,
    )
    if fap > fap_threshold:
        logger.warning(
            "[LS] FAP (%.2e) 大於閾值 (%.3f)，週期信號可能不顯著。",
            fap, fap_threshold,
        )

    # DFT 交叉驗證
    dft_amp = classical_dft_amplitude(t, mag, freqs)

    return {
        "freqs": freqs,
        "ls_power": ls_power,
        "dft_amp": dft_amp,
        "best_freq": best_freq,
        "best_period": best_period,
        "peak_power": peak_power,
        "fap": fap,
        "fap_n_iter": fap_n_iter,
        "fap_status": fap_status,
        "period_max_days": period_max_days,
    }


def _phase_coverage(phase: np.ndarray, n_bins: int = 20) -> float:
    """折疊後相位覆蓋率：有資料點的相位 bin 比例。"""
    hist, _ = np.histogram(phase, bins=n_bins, range=(0.0, 1.0))
    return float(np.count_nonzero(hist)) / n_bins


def _bic_over_harmonics(
    phase: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    max_harmonics: int,
) -> Tuple[float, int]:
    """
    掃 N=1..max_harmonics 全部擬合，回傳 (最小 BIC, 對應 N)。

    候選週期比較必須用此法而非 Breger 貪婪停階：Breger 對不同候選
    停在不同 N（路徑相依），χ² 比較會失真；BIC 取全域最小則對每個
    候選一視同仁，參數量懲罰由 BIC 自身承擔。
    """
    best_bic = np.inf
    best_n = 0
    for n in range(1, max_harmonics + 1):
        try:
            popt, _ = _fit_fourier(phase, mag, err, n)
        except (RuntimeError, ValueError):
            break
        mag_pred = _fourier_series(phase, *popt)
        bic = _compute_bic(mag, mag_pred, 2 * n + 1, err)
        if bic < best_bic:
            best_bic = bic
            best_n = n
    if best_n == 0:
        raise RuntimeError("所有諧波數均擬合失敗")
    return best_bic, best_n


def _refine_period(
    t: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    period: float,
    max_harmonics: int,
) -> Optional[Tuple[float, float]]:
    """
    以週期為自由參數的傅立葉擬合精修。

    LS 候選週期為頻率格網值，與真值的微小偏差在折疊多個週期後累積
    相位漂移，會讓 2P 模型的半頻項假性吸收殘差（P/2P 判別失真）。
    精修範圍限制在頻率 ±1/(2T) 內（半個自然峰寬），不會跳至其他峰。

    Returns
    -------
    (refined_period, period_err) 或 None（擬合失敗/撞到邊界時）。
    """
    t0 = float(t[0])
    baseline = float(np.max(t) - np.min(t))
    if baseline <= 0:
        return None
    phase = ((t - t0) / period) % 1.0
    try:
        _, n_h = _bic_over_harmonics(phase, mag, err, max_harmonics)
        popt_fixed, _ = _fit_fourier(phase, mag, err, n_h)
    except (RuntimeError, ValueError):
        return None

    f0 = 1.0 / period
    df = 0.5 / baseline
    p_lo = 1.0 / (f0 + df)
    p_hi = 1.0 / (f0 - df) if f0 > df else period * 2.0
    # 上界鎖在基線內：P > 基線折不滿一週期，擬合退化為趨勢，
    # 未扣除的慢漂移會把自由週期往長端拉（2026-07-08 CCAnd 3.67h 事故）
    p_hi = min(p_hi, baseline)
    if p_lo >= p_hi:
        return None

    def _model(tt, period_, *coef):
        ph = ((tt - t0) / period_) % 1.0
        return _fourier_series(ph, *coef)

    n_coef = len(popt_fixed)
    p0 = [period] + [float(v) for v in popt_fixed]
    lb = [p_lo] + [-np.inf] * n_coef
    ub = [p_hi] + [np.inf] * n_coef
    sigma = err if (np.isfinite(err).all() and np.any(err > 0)) else None
    try:
        popt, pcov = curve_fit(
            _model, t, mag, p0=p0, sigma=sigma, absolute_sigma=True,
            bounds=(lb, ub), maxfev=100_000,
        )
    except (RuntimeError, ValueError):
        return None

    refined = float(popt[0])
    # 撞邊界視為失敗（峰形異常，不信任）
    _edge = 0.02 * (p_hi - p_lo)
    if refined <= p_lo + _edge or refined >= p_hi - _edge:
        return None
    period_err = float(np.sqrt(pcov[0, 0])) if np.isfinite(pcov[0, 0]) else float("nan")
    return refined, period_err


def scan_bic_interval(
    t: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    period: float,
    cfg: Dict,
    n_grid: int = 121,
) -> Optional[Dict]:
    """
    ΔBIC 掃描區間：以週期掃描 BIC 地形，回報 ΔBIC < delta 的週期區間。

    共變異矩陣誤差在 BIC 地形平坦（基線 ≈ 週期）時嚴重低估不確定度；
    此掃描把「碗有多平」誠實寫進結果。掃描範圍 = 頻率 ±1/(2T)（半個
    自然峰寬），與 _refine_period 一致。

    Returns
    -------
    dict: {period_lo_d, period_hi_d, err_half_width_d, delta, n_valid}
          或 None（掃描失敗）。
    """
    delta = float(_get(cfg, "period_analysis", "bic_interval", "delta", default=2.0))
    fit_cfg = _get(cfg, "period_analysis", "fourier_fit", default={})
    max_harmonics = int(_get(fit_cfg, "max_harmonics", default=8))

    baseline = float(np.max(t) - np.min(t))
    if baseline <= 0 or period <= 0:
        return None
    f0 = 1.0 / period
    df = 0.5 / baseline
    f_grid = np.linspace(max(f0 - df, 1e-9), f0 + df, n_grid)
    t0 = float(t[0])

    bics = np.full(n_grid, np.inf)
    for i, f in enumerate(f_grid):
        p = 1.0 / f
        if p > baseline:
            continue
        phase = ((t - t0) / p) % 1.0
        if _phase_coverage(phase) < 0.9:
            continue
        try:
            bics[i], _ = _bic_over_harmonics(phase, mag, err, max_harmonics)
        except RuntimeError:
            continue

    valid = np.isfinite(bics)
    if valid.sum() < 5:
        return None
    bic_min = float(np.min(bics[valid]))
    in_band = valid & (bics <= bic_min + delta)
    p_in = 1.0 / f_grid[in_band]
    period_lo, period_hi = float(np.min(p_in)), float(np.max(p_in))
    # 區間頂到掃描邊界 = 碗底延伸出視窗，區間為下限
    touches_edge = bool(in_band[0] or in_band[-1])
    return {
        "period_lo_d": period_lo,
        "period_hi_d": period_hi,
        "err_half_width_d": 0.5 * (period_hi - period_lo),
        "delta": delta,
        "touches_edge": touches_edge,
        "n_valid": int(valid.sum()),
    }


def joint_period_scan(
    channels: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    center_period: float,
    cfg: Dict,
    n_grid: int = 161,
) -> Optional[Dict]:
    """
    多通道聯合週期仲裁：各通道共享試驗週期、各自傅立葉係數，BIC 相加。

    單通道 BIC 地形在基線 ≈ 週期時平坦（ΔBIC < 5，無決定性）；四通道
    聯合讓碗深相加，是唯一真正增加資訊量的判別方案（2026-07-06 評測：
    合成單夜回收 RMSE 0.035h vs 單通道 0.238h；真實 CCAnd 聯合谷底
    3.00h vs VSX 3.008h，對比度 ΔBIC=22）。

    Parameters
    ----------
    channels : {channel: (t, mag, err)}，各通道已篩選的有效資料。
    center_period : 掃描中心週期（days），通常取各通道結果的中位數。

    Returns
    -------
    dict: {period_d, period_lo_d, period_hi_d, delta_bic_contrast,
           n_channels, channels, per_point_valid} 或 None。
    """
    jp_cfg = _get(cfg, "period_analysis", "joint_period", default={})
    if not bool(_get(jp_cfg, "enabled", default=True)):
        return None
    delta = float(_get(cfg, "period_analysis", "bic_interval", "delta", default=2.0))
    min_channels = int(_get(jp_cfg, "min_channels", default=2))
    fit_cfg = _get(cfg, "period_analysis", "fourier_fit", default={})
    max_harmonics = int(_get(fit_cfg, "max_harmonics", default=8))

    usable = {ch: arr for ch, arr in channels.items() if len(arr[0]) >= 10}
    if len(usable) < min_channels:
        return None

    baseline = max(float(np.max(t) - np.min(t)) for (t, _, _) in usable.values())
    if baseline <= 0 or center_period <= 0:
        return None
    f0 = 1.0 / center_period
    df = 0.5 / baseline
    f_grid = np.linspace(max(f0 - df, 1e-9), f0 + df, n_grid)

    joint = np.full(n_grid, np.inf)
    for i, f in enumerate(f_grid):
        p = 1.0 / f
        if p > baseline:
            continue
        total = 0.0
        ok = True
        for (t, mag, err) in usable.values():
            phase = ((t - float(t[0])) / p) % 1.0
            if _phase_coverage(phase) < 0.9:
                ok = False
                break
            try:
                bic, _ = _bic_over_harmonics(phase, mag, err, max_harmonics)
            except RuntimeError:
                ok = False
                break
            total += bic
        if ok:
            joint[i] = total

    valid = np.isfinite(joint)
    if valid.sum() < 5:
        return None
    i_min = int(np.argmin(np.where(valid, joint, np.inf)))
    bic_min = float(joint[i_min])
    in_band = valid & (joint <= bic_min + delta)
    p_in = 1.0 / f_grid[in_band]
    # 對比度：區間外最接近谷底的 BIC 差（衡量決定性）
    out_band = valid & ~in_band
    contrast = float(np.min(joint[out_band]) - bic_min) if out_band.any() else float("nan")
    return {
        "period_d": float(1.0 / f_grid[i_min]),
        "period_lo_d": float(np.min(p_in)),
        "period_hi_d": float(np.max(p_in)),
        "delta": delta,
        "delta_bic_contrast": contrast,
        "touches_edge": bool(in_band[0] or in_band[-1]),
        "n_channels": len(usable),
        "channels": sorted(usable.keys()),
    }


def select_best_period_candidate(
    t: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    freqs: np.ndarray,
    ls_power: np.ndarray,
    cfg: Dict,
) -> Dict:
    """
    LS 前 N 峰候選週期挑選（VanderPlas 2018：最高峰不保證為真週期）。

    方法：取 LS 週期圖前 n_candidates 個局部極大（頻率間隔 >= 1/T 視為
    獨立峰），每個候選各自折疊 + Breger 選階傅立葉擬合，比較 BIC——
    即 multi-harmonic AoV（Schwarzenberg-Czerny 1996）的 BIC 版本。

    誠實性約束：
    1. 折不滿一個完整週期的候選（P > 基線）直接排除，不參與比較。
    2. 候選需贏過 LS 主峰 ΔBIC > delta_bic_margin 才取代之；
       否則保留主峰並標記 decisive=False（資料不足以區分）。

    Returns
    -------
    dict:
        period     : 採用的週期（days）
        switched   : True 表示採用了非主峰候選
        decisive   : ΔBIC 是否具決定性
        candidates : 各候選明細 list（period_d/power/n_harmonics/bic/delta_bic）
        status     : "selected" | "kept" | "skipped_disabled" | "no_candidates"
    """
    cs_cfg = _get(cfg, "period_analysis", "candidate_selection", default={})
    enabled = bool(_get(cs_cfg, "enabled", default=True))
    n_candidates = int(_get(cs_cfg, "n_candidates", default=5))
    delta_bic_margin = float(_get(cs_cfg, "delta_bic_margin", default=10.0))
    min_phase_coverage = float(_get(cs_cfg, "min_phase_coverage", default=0.9))

    best_idx = int(np.argmax(ls_power))
    result = {
        "period": float(1.0 / freqs[best_idx]),
        "switched": False,
        "decisive": False,
        "candidates": [],
        "status": "kept",
    }
    if not enabled:
        result["status"] = "skipped_disabled"
        return result

    baseline = float(np.max(t) - np.min(t))
    min_freq_sep = 1.0 / baseline if baseline > 0 else 0.0

    # 候選只在可折疊區（P <= 基線）內尋找：P > 基線折不滿一週期無法評分，
    # 且若 argmax 落在該區被排除，同峰內可折疊的真週期區域會連帶消失，
    # fallback 會跳去遠處垃圾候選（2026-07-08 CCAnd 0.5h 事故）
    _p = np.asarray(ls_power, dtype=float).copy()
    _foldable = (1.0 / np.asarray(freqs, dtype=float)) <= baseline
    if not _foldable.any():
        result["status"] = "no_candidates"
        return result
    _p[~_foldable] = -np.inf
    _fold_best_idx = int(np.argmax(_p))
    if not _foldable[best_idx]:
        logger.info(
            "[候選] LS 全域主峰 P=%.4f h 超出基線 %.4f h，改以可折疊區主峰 "
            "P=%.4f h 為錨點。",
            (1.0 / freqs[best_idx]) * 24.0, baseline * 24.0,
            (1.0 / freqs[_fold_best_idx]) * 24.0,
        )
    best_idx = _fold_best_idx
    result["period"] = float(1.0 / freqs[best_idx])
    _local_max = np.flatnonzero(
        (_p[1:-1] > _p[:-2]) & (_p[1:-1] >= _p[2:])
    ) + 1
    # 端點也可能是峰（頻率窗邊界）
    if _p.size >= 2 and _p[0] > _p[1]:
        _local_max = np.concatenate(([0], _local_max))
    if _p.size >= 2 and _p[-1] > _p[-2]:
        _local_max = np.concatenate((_local_max, [_p.size - 1]))
    if _local_max.size == 0:
        result["status"] = "no_candidates"
        return result

    # 依功率排序，貪婪選取頻率間隔 >= 1/T 的獨立峰
    _order = _local_max[np.argsort(_p[_local_max])[::-1]]
    picked: list = []
    for _i in _order:
        if len(picked) >= n_candidates:
            break
        if all(abs(freqs[_i] - freqs[_j]) >= min_freq_sep for _j in picked):
            picked.append(int(_i))

    fit_cfg = _get(cfg, "period_analysis", "fourier_fit", default={})
    max_harmonics = int(_get(fit_cfg, "max_harmonics", default=8))
    t0 = float(t[0])

    candidates = []
    for _i in picked:
        period = float(1.0 / freqs[_i])
        entry = {
            "period_d": period,
            "period_hr": period * 24.0,
            "power": float(_p[_i]),
            "is_ls_peak": bool(_i == best_idx),
            "n_harmonics": None,
            "bic": None,
            "phase_coverage": None,
            "excluded": None,
        }
        if period > baseline:
            entry["excluded"] = "period_exceeds_baseline"
            candidates.append(entry)
            continue
        phase = ((t - t0) / period) % 1.0
        coverage = _phase_coverage(phase)
        entry["phase_coverage"] = coverage
        if coverage < min_phase_coverage:
            # 相位缺口讓傅立葉模型有免費自由度，BIC 比較會失真
            entry["excluded"] = f"phase_coverage {coverage:.2f} < {min_phase_coverage}"
            candidates.append(entry)
            continue
        try:
            bic, n_h = _bic_over_harmonics(phase, mag, err, max_harmonics)
            entry["n_harmonics"] = int(n_h)
            entry["bic"] = float(bic)
        except RuntimeError as exc:
            entry["excluded"] = f"fit_failed: {exc}"
        candidates.append(entry)

    scored = [c for c in candidates if c["bic"] is not None]
    if not scored:
        result["status"] = "no_candidates"
        result["candidates"] = candidates
        logger.warning("[候選] 無可評分候選（全部超出基線或擬合失敗），保留 LS 主峰。")
        return result

    bic_min = min(c["bic"] for c in scored)
    for c in candidates:
        c["delta_bic"] = (c["bic"] - bic_min) if c["bic"] is not None else None
    result["candidates"] = candidates

    for c in scored:
        logger.info(
            "[候選] P=%.4f h  power=%.3f  N=%d  BIC=%.1f  ΔBIC=%.1f%s",
            c["period_hr"], c["power"], c["n_harmonics"], c["bic"],
            c["delta_bic"], "  ←LS主峰" if c["is_ls_peak"] else "",
        )
    for c in candidates:
        if c["excluded"]:
            logger.info(
                "[候選] P=%.4f h  排除：%s", c["period_hr"], c["excluded"]
            )

    winner = min(scored, key=lambda c: c["bic"])
    ls_peak_entry = next((c for c in scored if c["is_ls_peak"]), None)

    if ls_peak_entry is None:
        # LS 主峰本身被排除（超出基線）：直接採用可評分候選中最佳者
        result["period"] = winner["period_d"]
        result["switched"] = True
        result["decisive"] = True
        result["status"] = "selected"
        logger.warning(
            "[候選] LS 主峰折不滿一個週期，改用候選 P=%.4f h。", winner["period_hr"]
        )
        return result

    if winner is not ls_peak_entry:
        gain = ls_peak_entry["bic"] - winner["bic"]
        if gain > delta_bic_margin:
            result["period"] = winner["period_d"]
            result["switched"] = True
            result["decisive"] = True
            result["status"] = "selected"
            logger.warning(
                "[候選] BIC 判別：P=%.4f h 顯著優於 LS 主峰 %.4f h"
                "（ΔBIC=%.1f > %.1f），採用之。",
                winner["period_hr"], ls_peak_entry["period_hr"],
                gain, delta_bic_margin,
            )
        else:
            result["decisive"] = False
            logger.info(
                "[候選] P=%.4f h BIC 略優但未達決定性（ΔBIC=%.1f ≤ %.1f），"
                "保留 LS 主峰。",
                winner["period_hr"], gain, delta_bic_margin,
            )
    else:
        result["decisive"] = all(
            (c is ls_peak_entry) or (c["delta_bic"] > delta_bic_margin)
            for c in scored
        )
        logger.info(
            "[候選] LS 主峰即最佳候選（decisive=%s）。", result["decisive"]
        )
    return result


def resolve_half_period_ambiguity(
    t: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    best_period: float,
    cfg: Dict,
) -> Dict:
    """
    P vs 2P 判別（W UMa 型食雙星防護）。

    W UMa / 橢球變星一個軌道週期有兩個近似等深的極小，LS（正弦基底）
    最高峰常落在真週期的一半。判別方法：分別以 P 與 2P 折疊做傅立葉擬合
    （Breger 選階），比較 BIC。2P 模型的偶數諧波已完全涵蓋 P 模型，
    唯有奇數諧波（兩極小深度差）帶顯著振幅時 BIC 才會勝出——此即食雙星
    特徵，對 delta Sct 等單極大脈動星不會誤觸發。

    Returns
    -------
    dict:
        period    : 採用的週期（days）
        doubled   : True 表示改用 2P
        delta_bic : BIC(P) - BIC(2P)；未執行比較時為 nan
        status    : "doubled" | "kept" | "skipped_disabled"
                    | "skipped_baseline" | "skipped_coverage" | "fit_failed"
    """
    dp_cfg = _get(cfg, "period_analysis", "double_period_check", default={})
    enabled = bool(_get(dp_cfg, "enabled", default=True))
    min_cycles = float(_get(dp_cfg, "min_cycles", default=1.0))
    delta_bic_threshold = float(_get(dp_cfg, "delta_bic_threshold", default=10.0))

    result = {
        "period": float(best_period),
        "doubled": False,
        "delta_bic": float("nan"),
        "status": "kept",
    }
    if not enabled:
        result["status"] = "skipped_disabled"
        return result

    double_period = 2.0 * best_period
    baseline = float(np.max(t) - np.min(t))
    if baseline < min_cycles * double_period:
        logger.info(
            "[P/2P] 基線 %.3f d 不足 2P 的 %.1f 個週期（2P=%.3f d），跳過判別。",
            baseline, min_cycles, double_period,
        )
        result["status"] = "skipped_baseline"
        return result

    fit_cfg = _get(cfg, "period_analysis", "fourier_fit", default={})
    max_harmonics = int(_get(fit_cfg, "max_harmonics", default=8))
    min_phase_coverage = float(
        _get(dp_cfg, "min_phase_coverage", default=0.9)
    )

    t0 = float(t[0])
    _phase_2p = ((t - t0) / double_period) % 1.0
    _cov_2p = _phase_coverage(_phase_2p)
    if _cov_2p < min_phase_coverage:
        logger.info(
            "[P/2P] 2P 折疊相位覆蓋率 %.2f < %.2f（缺口使 BIC 失真），跳過判別。",
            _cov_2p, min_phase_coverage,
        )
        result["status"] = "skipped_coverage"
        return result

    try:
        bics = {}
        for label, period in (("P", best_period), ("2P", double_period)):
            phase = ((t - t0) / period) % 1.0
            bic, n_h = _bic_over_harmonics(phase, mag, err, max_harmonics)
            bics[label] = bic
            logger.info(
                "[P/2P] %s=%.6f d  N=%d  BIC=%.1f", label, period, n_h, bics[label]
            )
    except RuntimeError as exc:
        logger.warning("[P/2P] 傅立葉擬合失敗，保留 LS 週期：%s", exc)
        result["status"] = "fit_failed"
        return result

    delta_bic = bics["P"] - bics["2P"]
    result["delta_bic"] = float(delta_bic)
    if delta_bic > delta_bic_threshold:
        logger.warning(
            "[P/2P] BIC 判別：2P 顯著較優（ΔBIC=%.1f > %.1f），"
            "週期改用 %.6f d——疑似食雙星（W UMa 型）半週期混淆。",
            delta_bic, delta_bic_threshold, double_period,
        )
        result["period"] = float(double_period)
        result["doubled"] = True
        result["status"] = "doubled"
    else:
        logger.info(
            "[P/2P] BIC 判別：保留 LS 週期（ΔBIC=%.1f ≤ %.1f）。",
            delta_bic, delta_bic_threshold,
        )
    return result


def _safe_stem(text: str) -> str:
    return str(text).replace(" ", "")


def _write_ls_spectrum_csv(
    out_dir: Path,
    target_name: str,
    channel: str,
    ls_result: Dict,
) -> Path:
    freqs = np.asarray(ls_result["freqs"], dtype=float)
    df = pd.DataFrame(
        {
            "channel": channel,
            "frequency_d_inv": freqs,
            "period_d": 1.0 / freqs,
            "period_hr": 24.0 / freqs,
            "ls_power": np.asarray(ls_result["ls_power"], dtype=float),
        }
    )
    out_path = out_dir / f"period_ls_spectrum_{_safe_stem(target_name)}_{channel}.csv"
    df.to_csv(out_path, index=False, float_format="%.10f")
    return out_path


def _write_dft_spectrum_csv(
    out_dir: Path,
    target_name: str,
    channel: str,
    ls_result: Dict,
) -> Path:
    freqs = np.asarray(ls_result["freqs"], dtype=float)
    df = pd.DataFrame(
        {
            "channel": channel,
            "frequency_d_inv": freqs,
            "period_d": 1.0 / freqs,
            "period_hr": 24.0 / freqs,
            "dft_amplitude": np.asarray(ls_result["dft_amp"], dtype=float),
        }
    )
    out_path = out_dir / f"period_dft_spectrum_{_safe_stem(target_name)}_{channel}.csv"
    df.to_csv(out_path, index=False, float_format="%.10f")
    return out_path


def _write_fourier_fit_json(
    out_dir: Path,
    target_name: str,
    channel: str,
    fit_result: Dict,
    ls_result: Dict,
) -> Path:
    payload = {
        "channel": channel,
        "best_period_d": float(fit_result["period"]),
        "best_frequency_d_inv": float(ls_result["best_freq"]),
        "peak_power": float(ls_result["peak_power"]),
        "fap": float(ls_result["fap"]),
        "fap_n_iter": int(ls_result["fap_n_iter"]),
        "fap_status": ls_result["fap_status"],
        "t0": float(fit_result["t0"]),
        "period_err": float(fit_result["period_err"]),
        "n_harmonics": int(fit_result["n_harmonics"]),
        "amplitude": float(fit_result["amplitude"]),
        "rms_residuals": float(fit_result["rms_residuals"]),
        "coefficients": [float(v) for v in np.asarray(fit_result["popt"], dtype=float)],
    }
    _interval = ls_result.get("bic_interval")
    if _interval:
        payload["period_interval_dbic"] = {
            "period_lo_d": _interval["period_lo_d"],
            "period_hi_d": _interval["period_hi_d"],
            "err_half_width_d": _interval["err_half_width_d"],
            "delta": _interval["delta"],
            "touches_edge": _interval["touches_edge"],
        }
    cand_sel = ls_result.get("candidate_selection")
    if cand_sel:
        payload["candidate_selection"] = {
            "status": cand_sel["status"],
            "switched": bool(cand_sel["switched"]),
            "decisive": bool(cand_sel["decisive"]),
            "candidates": [
                {
                    "period_hr": float(c["period_hr"]),
                    "power": float(c["power"]),
                    "n_harmonics": c["n_harmonics"],
                    "bic": c["bic"],
                    "delta_bic": c.get("delta_bic"),
                    "is_ls_peak": bool(c["is_ls_peak"]),
                    "excluded": c["excluded"],
                }
                for c in cand_sel["candidates"]
            ],
        }
    dp_check = ls_result.get("double_period_check")
    if dp_check:
        payload["double_period_check"] = {
            "status": dp_check["status"],
            "doubled": bool(dp_check["doubled"]),
            "delta_bic": (
                float(dp_check["delta_bic"])
                if np.isfinite(dp_check["delta_bic"]) else None
            ),
        }
    out_path = out_dir / f"period_fourier_fit_{_safe_stem(target_name)}_{channel}.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return out_path


def _write_fourier_model_csv(
    out_dir: Path,
    target_name: str,
    channel: str,
    fit_result: Dict,
) -> Path:
    df = pd.DataFrame(
        {
            "channel": channel,
            "phase": np.asarray(fit_result["phi_dense"], dtype=float),
            "model_mag": np.asarray(fit_result["mag_dense"], dtype=float),
        }
    )
    out_path = out_dir / f"period_fourier_model_{_safe_stem(target_name)}_{channel}.csv"
    df.to_csv(out_path, index=False, float_format="%.10f")
    return out_path


# =============================================================================
# 相位折疊與傅立葉擬合
# =============================================================================

def compute_period_uncertainty(
    t: np.ndarray,
    fourier_residuals: np.ndarray,
    freq: float,
) -> Tuple[float, float]:
    """
    依據 Kovacs (1981) 估計頻率與週期不確定度下限。

    公式：sigma_f = sigma_res / (T * sqrt(N))
          sigma_P = sigma_f / f^2

    Parameters
    ----------
    t                : 時間陣列（days）。
    fourier_residuals: 傅立葉擬合後的殘差陣列（必須使用擬合殘差，非 LS 殘差）。
    freq             : 傅立葉擬合使用的頻率（1/day）。
    """
    t_span = float(np.max(t) - np.min(t))
    n = len(t)
    sigma_res = float(np.std(fourier_residuals))
    if t_span <= 0 or n <= 1:
        return np.nan, np.nan
    sigma_f = sigma_res / (t_span * np.sqrt(n))
    sigma_p = sigma_f / (freq ** 2)
    return float(sigma_f), float(sigma_p)


def fit_phase_folded_model(
    t: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    period: float,
    cfg: Dict,
) -> Dict:
    """
    兩步迭代傅立葉擬合，BIC 自動選階，驗證相位零點。

    步驟
    ----
    Step 1  以 t[0] 為暫時零點，做 BIC 選階 + 傅立葉擬合。
            在稠密相位格網上找極大亮度相位 phi_max。
    Step 2  t0_final = t[0] + phi_max * period。
            以 t0_final 重新折疊，第二次傅立葉擬合（沿用 Step 1 階數）。
    驗證    在第二次擬合的稠密格網上確認極大值相位 phi_check < 0.05。
            若不符，logger.error 輸出 t0_final 並 raise ValueError。

    週期不確定度使用傅立葉擬合殘差（Step 2）。

    Parameters
    ----------
    t      : 時間陣列。
    mag    : 差分星等陣列。
    err    : 測光誤差陣列。
    period : 從 LS 取得的最佳週期（days）。
    cfg    : 完整設定 dict。

    Returns
    -------
    dict 包含：
        t0, period, period_err, n_harmonics, amplitude,
        rms_residuals, phase, residuals, popt, pcov,
        phi_dense, mag_dense
    """
    fit_cfg = _get(cfg, "period_analysis", "fourier_fit", default={})
    max_harmonics = _get(fit_cfg, "max_harmonics", default=8)
    sn_threshold = float(_get(fit_cfg, "sn_threshold", default=4.0))

    freq = 1.0 / period
    phi_dense = np.linspace(0.0, 1.0, 10_000)

    # ── Step 1：暫時零點擬合 ─────────────────────────────────────────────────
    t0_temp = float(t[0])
    phase_temp = ((t - t0_temp) / period) % 1.0

    logger.info("[Step 1] Breger S/N 選階（上限 N=%d，閾值 %.1f）…", max_harmonics, sn_threshold)
    try:
        best_n, popt_temp, _ = _select_harmonics_breger(
            phase_temp, mag, err, max_harmonics, sn_threshold
        )
    except RuntimeError as exc:
        logger.error("[Step 1] Breger 選階失敗：%s", exc)
        return {}

    # 在稠密格網找極大亮度相位（mag 最小值）
    mag_dense_temp = _fourier_series(phi_dense, *popt_temp)
    phi_max = float(phi_dense[np.argmin(mag_dense_temp)])
    logger.info("[Step 1] 極大亮度相位 phi_max = %.4f", phi_max)

    # ── Step 2：修正零點重新擬合 ──────────────────────────────────────────────
    t0_final = t0_temp + phi_max * period
    phase_final = ((t - t0_final) / period) % 1.0

    logger.info("[Step 2] t0_final = %.8f  重新擬合 N=%d …", t0_final, best_n)
    try:
        popt_final, pcov_final = _fit_fourier(phase_final, mag, err, best_n)
    except (RuntimeError, ValueError) as exc:
        logger.error("[Step 2] 傅立葉擬合失敗（t0_final=%.8f）：%s", t0_final, exc)
        return {}

    # ── 驗證相位零點 ──────────────────────────────────────────────────────────
    mag_dense_final = _fourier_series(phi_dense, *popt_final)
    phi_check = float(phi_dense[np.argmin(mag_dense_final)])
    logger.info("[驗證] Step 2 極大亮度相位 phi_check = %.4f", phi_check)

    # phi_check 應接近 0（或接近 1，相位捲繞的情況）
    phi_check_dist = min(phi_check, 1.0 - phi_check)
    if phi_check_dist > 0.05:
        logger.error(
            "[驗證失敗] 相位零點未對齊：phi_check = %.4f（距離 0 超過 0.05）。"
            " t0_final = %.8f d。"
            " 可能原因：LS 最佳週期不精確，或擬合收斂到局部解。"
            " 請手動確認 LS 週期並重試。",
            phi_check, t0_final,
        )
        raise ValueError(
            f"相位零點驗證失敗：phi_check={phi_check:.4f} > 0.05。"
            f" t0_final={t0_final:.8f} d。"
        )

    # ── 計算殘差與不確定度 ────────────────────────────────────────────────────
    mag_pred = _fourier_series(phase_final, *popt_final)
    residuals = mag - mag_pred
    rms_residuals = float(np.std(residuals))

    # 週期不確定度：使用傅立葉擬合殘差（Kovacs, 1981）
    sigma_f, sigma_p = compute_period_uncertainty(t, residuals, freq)

    # 振幅
    amplitude = float(np.max(mag_dense_final) - np.min(mag_dense_final))

    logger.info(
        "[擬合] 振幅=%.4f mag  RMS殘差=%.4f mag  sigma_P=%.2e d",
        amplitude, rms_residuals, sigma_p,
    )

    return {
        "t0": t0_final,
        "period": period,
        "period_err": sigma_p,
        "n_harmonics": best_n,
        "amplitude": amplitude,
        "rms_residuals": rms_residuals,
        "phase": phase_final,
        "residuals": residuals,
        "popt": popt_final,
        "pcov": pcov_final,
        "phi_dense": phi_dense,
        "mag_dense": mag_dense_final,
    }


# =============================================================================
# 預白化（Pre-whitening）
# =============================================================================

def _estimate_noise_level(
    freqs: np.ndarray,
    dft_amp: np.ndarray,
    signal_freq: float,
    exclude_halfwidth: float = 1.0,
) -> float:
    """
    估計噪聲基準：排除信號頻率附近 ±exclude_halfwidth d⁻¹ 的頻段，
    取剩餘 DFT 振幅的中位數作為噪聲水準。

    Parameters
    ----------
    freqs            : 頻率格網（1/day）。
    dft_amp          : DFT 振幅譜。
    signal_freq      : 當前信號頻率（1/day）。
    exclude_halfwidth: 排除頻段寬度（d⁻¹），預設 ±1。
    """
    mask = np.abs(freqs - signal_freq) > exclude_halfwidth
    if np.sum(mask) < 5:
        return float(np.median(dft_amp))
    return float(np.median(dft_amp[mask]))


def run_pre_whitening(
    t: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    cfg: Dict,
    target_name: str,
    channel: str,
    out_dir: Path,
) -> List[Dict]:
    """
    預白化迭代（Breger et al., 1993）。

    每次迭代：
        1. 對當前殘差執行 LS，找最高功率頻率。
        2. 用單頻傅立葉模型擬合，從殘差中減去。
        3. 計算 S/N：信號振幅 / 噪聲基準（DFT 中位數，排除信號附近頻段）。
        4. S/N < sn_threshold（預設 4.0）→ 停止。
        5. 達到 max_frequencies → 停止。

    輸出：所有迭代的診斷圖合併為一個 PNG。
    CSV 輸出預留但停用（yaml save_csv: false）。

    Parameters
    ----------
    t           : 時間陣列。
    mag         : 差分星等陣列（初始殘差 = 原始光度）。
    err         : 測光誤差陣列。
    cfg         : 完整設定 dict。
    target_name : 目標名稱（用於圖標題和檔名）。
    channel     : 波段（"R" / "G1" / "B"）。
    out_dir     : 輸出目錄。

    Returns
    -------
    List[Dict]，每個 dict 包含一次迭代的萃取結果：
        freq, period, amplitude, sn, residuals_after
    """
    pw_cfg = _get(cfg, "period_analysis", "pre_whitening", default={})
    sn_threshold = float(_get(pw_cfg, "sn_threshold", default=4.0))
    max_freq_count = int(_get(pw_cfg, "max_frequencies", default=10))
    save_csv = bool(_get(pw_cfg, "save_csv", default=False))

    ls_cfg = _get(cfg, "period_analysis", "lomb_scargle", default={})
    period_min_days = _get(ls_cfg, "period_min_hr", default=0.5) / 24.0
    period_max_days = float(_get(ls_cfg, "period_max_hours", default=8.0)) / 24.0
    oversampling = _get(ls_cfg, "oversampling", default=10)

    plot_cfg = _get(cfg, "output", "plots", default={})
    dpi = _get(plot_cfg, "dpi", default=150)
    figsize_pw = _get(plot_cfg, "figsize_prewhitening", default=[15, 4])

    out_dir.mkdir(parents=True, exist_ok=True)

    residuals = mag.copy()
    extracted: List[Dict] = []
    fig_rows: List[plt.Figure] = []

    logger.info(
        "[Pre-whitening] 開始  S/N 閾值=%.1f  最大頻率數=%d",
        sn_threshold, max_freq_count,
    )

    for iteration in range(1, max_freq_count + 1):
        # ── LS 在殘差上 ───────────────────────────────────────────────────────
        ls_iter = LombScargle(t, residuals, dy=err)
        freqs_iter, power_iter = ls_iter.autopower(
            minimum_frequency=1.0 / period_max_days,
            maximum_frequency=1.0 / period_min_days,
            samples_per_peak=oversampling,
        )

        best_idx = int(np.argmax(power_iter))
        signal_freq = float(freqs_iter[best_idx])
        signal_period = 1.0 / signal_freq

        # 單頻傅立葉擬合（N=1）
        phase_iter = ((t - t[0]) / signal_period) % 1.0
        try:
            popt_iter, _ = _fit_fourier(phase_iter, residuals, err, 1)
        except (RuntimeError, ValueError) as exc:
            logger.error("[Pre-whitening] 迭代 %d 擬合失敗：%s", iteration, exc)
            break

        amplitude_iter = float(
            np.sqrt(popt_iter[1] ** 2 + popt_iter[2] ** 2) * np.sqrt(2)
        )

        # DFT 振幅譜 → 噪聲估計 → S/N
        dft_iter = classical_dft_amplitude(t, residuals, freqs_iter)
        noise_level = _estimate_noise_level(freqs_iter, dft_iter, signal_freq)
        sn = amplitude_iter / noise_level if noise_level > 0 else np.inf

        logger.info(
            "[Pre-whitening] 迭代 %d：freq=%.4f d⁻¹  P=%.4f d  "
            "A=%.4f mag  S/N=%.1f",
            iteration, signal_freq, signal_period, amplitude_iter, sn,
        )

        # ── 診斷圖（此次迭代）─────────────────────────────────────────────────
        fig_iter, axes = plt.subplots(1, 3, figsize=figsize_pw, dpi=dpi)
        fig_iter.suptitle(
            f"{target_name} [{channel}]  Pre-whitening Iter {iteration}"
            f"  —  f={signal_freq:.4f} d⁻¹, P={signal_period:.4f} d",
            fontsize=10,
        )

        # 子圖 1：當前 LS 週期圖
        ax = axes[0]
        ax.plot(1.0 / freqs_iter, power_iter, "k-", lw=0.5)
        ax.axvline(signal_period, color="r", ls="--", lw=1, alpha=0.8)
        ax.set_xlabel("Period (days)")
        ax.set_ylabel("LS Power")
        ax.set_title("Lomb-Scargle (residuals)")

        # 子圖 2：殘差光變曲線（時域）
        ax = axes[1]
        ax.errorbar(
            t, residuals, yerr=err, fmt="ko", ms=2,
            elinewidth=0.6, capsize=0, alpha=0.5,
        )
        ax.invert_yaxis()
        ax.set_xlabel("BJD_TDB")
        ax.set_ylabel("Residual (mag)")
        ax.set_title("Residual Light Curve")

        # 子圖 3：S/N 標示
        ax = axes[2]
        ax.bar(["Amplitude", "Noise", "S/N threshold"],
               [amplitude_iter, noise_level, sn_threshold * noise_level],
               color=["steelblue", "gray", "tomato"], alpha=0.8)
        ax.set_ylabel("mag")
        ax.set_title(f"S/N = {sn:.2f}  ({'[OK] continue' if sn >= sn_threshold else '[STOP]'})")

        plt.tight_layout()
        fig_rows.append(fig_iter)

        # ── 從殘差中減去信號 ──────────────────────────────────────────────────
        model_iter = _fourier_series(phase_iter, *popt_iter)
        residuals = residuals - (model_iter - popt_iter[0])  # 減去 AC 部分

        extracted.append({
            "iteration": iteration,
            "freq": signal_freq,
            "period": signal_period,
            "amplitude": amplitude_iter,
            "sn": sn,
            "residuals_after": residuals.copy(),
        })

        # ── 停止判斷 ──────────────────────────────────────────────────────────
        if sn < sn_threshold:
            logger.info(
                "[Pre-whitening] S/N=%.2f < %.1f，停止迭代（共萃取 %d 個頻率）。",
                sn, sn_threshold, len(extracted),
            )
            break
    else:
        logger.warning(
            "[Pre-whitening] 達到最大頻率數上限 %d，強制停止。", max_freq_count
        )

    # ── 合併所有迭代圖 ────────────────────────────────────────────────────────
    if fig_rows:
        out_png = out_dir / f"prewhitening_{target_name.replace(' ', '')}_{channel}.png"
        n_rows = len(fig_rows)
        fig_combined, axes_combined = plt.subplots(
            n_rows, 3,
            figsize=(figsize_pw[0], figsize_pw[1] * n_rows),
            dpi=dpi,
        )
        if n_rows == 1:
            axes_combined = [axes_combined]

        for row_idx, fig_src in enumerate(fig_rows):
            for col_idx, ax_src in enumerate(fig_src.axes):
                ax_dst = axes_combined[row_idx][col_idx]
                # 重繪每個子圖的內容
                for line in ax_src.lines:
                    ax_dst.plot(
                        line.get_xdata(), line.get_ydata(),
                        color=line.get_color(), lw=line.get_linewidth(),
                        ls=line.get_linestyle(), alpha=line.get_alpha() or 1.0,
                    )
                for child in ax_src.get_children():
                    if hasattr(child, "get_vlines"):
                        pass  # axvline 已在 lines 中
                ax_dst.set_xlabel(ax_src.get_xlabel())
                ax_dst.set_ylabel(ax_src.get_ylabel())
                ax_dst.set_title(ax_src.get_title())
                if ax_src.yaxis_inverted():
                    ax_dst.invert_yaxis()
            axes_combined[row_idx][0].set_ylabel(
                f"Iter {row_idx + 1}\n" + axes_combined[row_idx][0].get_ylabel()
            )

        plt.suptitle(
            f"Pre-whitening Summary: {target_name} [{channel}]"
            f"  —  {len(extracted)} 頻率萃取",
            fontsize=12, y=1.01,
        )
        plt.tight_layout()
        plt.savefig(out_png, bbox_inches="tight")
        plt.close(fig_combined)
        logger.info("[Pre-whitening] 診斷圖已儲存：%s", out_png)

        for fig in fig_rows:
            plt.close(fig)

    # CSV 輸出
    if save_csv and extracted:
        csv_path = out_dir / f"prewhitening_{target_name}_{channel}.csv"
        rows = []
        for i, e in enumerate(extracted, 1):
            rows.append({
                "iter": i,
                "freq_d": e["freq"],
                "period_d": e["period"],
                "amplitude_mag": e["amplitude"],
                "sn": e["sn"],
            })
        pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.8f")
        logger.info("[Pre-whitening] CSV 已儲存：%s", csv_path)

    return extracted


# =============================================================================
# 繪圖
# =============================================================================

def plot_period_analysis(
    target_name: str,
    channel: str,
    ls_result: Dict,
    fit_result: Dict,
    t: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    out_png: Path,
    cfg: Dict,
    obs_date: str = "",
) -> None:
    """
    輸出 1×3 組合診斷圖：LS 週期圖、DFT 振幅譜、相位折疊圖（含誤差棒）。

    Parameters
    ----------
    target_name : 目標星名稱。
    channel     : 波段。
    ls_result   : run_ls_and_dft() 的回傳值。
    fit_result  : fit_phase_folded_model() 的回傳值。
    t, mag, err : 原始資料陣列。
    out_png     : 輸出 PNG 路徑。
    cfg         : 完整設定 dict。
    """
    if not fit_result:
        logger.error("[繪圖] 無有效擬合結果，跳過。")
        return

    plot_cfg = _get(cfg, "output", "plots", default={})
    dpi = _get(plot_cfg, "dpi", default=150)

    freqs = ls_result["freqs"]
    best_p = fit_result["period"]
    best_p_err = fit_result["period_err"]
    fap = ls_result["fap"]
    fap_n_iter = ls_result["fap_n_iter"]
    fap_status = ls_result["fap_status"]

    # ── 色彩定義（深→淺：標題 > 主線 > 外框 > 座標軸） ──────────────────────
    # LS 綠色系
    _LS_TITLE = "#081c15"
    _LS_LINE  = "#1b4332"
    _LS_SPINE = "#52b788"
    _LS_TICK  = "#95d5b2"
    # DFT 紫色系
    _DFT_TITLE = "#240046"
    _DFT_LINE  = "#5a0080"
    _DFT_SPINE = "#9d4edd"
    _DFT_TICK  = "#c77dff"

    # ── 標題組裝：display_name + 日期 + 座標（多色） ────────────────────────
    _display_name = _get(cfg, "targets", target_name, "display_name", default=target_name)
    _lat, _lon = None, None
    for _sess in _get(cfg, "obs_sessions", default=[]):
        _sess_targets = _get(_sess, "targets", default=[])
        if isinstance(_sess_targets, list) and target_name in _sess_targets:
            _lat = _get(_sess, "obs_lat_deg", default=None)
            _lon = _get(_sess, "obs_lon_deg", default=None)
            break

    fig, axes = plt.subplots(
        1, 3, figsize=(22, 5), dpi=dpi,
        gridspec_kw={"width_ratios": [2.5, 1.5, 3]},
    )

    # 標題：照 photometry.py 做法，單行 + 座標右上角
    _date_str = f"  {obs_date}" if obs_date else ""
    fig.text(
        0.02, 0.95,
        f"{_display_name}  [{channel}]{_date_str}",
        fontsize=32, fontweight="bold", ha="left",
    )
    if _lat is not None and _lon is not None:
        _coord_disp = (
            f"{abs(_lat):.2f}°{'N' if _lat >= 0 else 'S'}"
            f" {abs(_lon):.2f}°{'E' if _lon >= 0 else 'W'}"
        )
        fig.text(0.98, 0.97, _coord_disp, fontsize=14, color="#2d6a4f", ha="right", va="top")
    _sigma_med = float(np.nanmedian(err)) if len(err) > 0 else float("nan")
    fig.text(
        0.98, 0.92,
        f"Reliability: σ_med = {_sigma_med:.4f} mag",
        ha="right", va="top", fontsize=14, color="saddlebrown",
    )

    # ── 子圖 1：LS 週期圖 ─────────────────────────────────────────────────────
    ax = axes[0]
    periods_days = 1.0 / freqs
    periods_hr = periods_days * 24.0
    best_p_hr = best_p * 24.0
    _best_h = int(best_p_hr)
    _best_m = int((best_p_hr - _best_h) * 60)

    ax.plot(periods_hr, ls_result["ls_power"], color=_LS_LINE, lw=1.0)
    ax.axvline(best_p_hr, color="r", ls="--", lw=1, alpha=0.8,
               label=f"P = {_best_h}:{_best_m:02d}")

    # x 軸顯示上限
    _plot_xmax_days = ls_result.get("period_max_days", periods_days.max())
    _plot_xmax_hr = _plot_xmax_days * 24.0
    ax.set_xlim(0.0, _plot_xmax_hr)

    # P2：與最高峰距離 >= 20% * best_p，在顯示範圍內
    _min_sep_hr = 0.20 * best_p_hr
    _mask_p2 = (
        (np.abs(periods_hr - best_p_hr) >= _min_sep_hr)
        & (periods_hr <= _plot_xmax_hr)
    )
    if _mask_p2.any():
        _p2_idx = int(np.argmax(ls_result["ls_power"][_mask_p2]))
        _p2_hr = float(periods_hr[_mask_p2][_p2_idx])
        _p2_h = int(_p2_hr)
        _p2_m = int((_p2_hr - _p2_h) * 60)
        ax.axvline(_p2_hr, color="orange", ls="--", lw=1, alpha=0.8,
                   label=f"P2 = {_p2_h}:{_p2_m:02d}")

    # P3：短週期峰（< 4h），抓最高峰，排除已標記的 P1/P2 附近
    _short_limit_hr = min(4.0, _plot_xmax_hr * 0.4)
    _mask_short = (periods_hr > 0.3) & (periods_hr <= _short_limit_hr)
    if _mask_short.any():
        _p3_idx_local = int(np.argmax(ls_result["ls_power"][_mask_short]))
        _p3_hr = float(periods_hr[_mask_short][_p3_idx_local])
        _p3_power = float(ls_result["ls_power"][_mask_short][_p3_idx_local])
        # 僅在與 P1/P2 距離夠遠時標記
        _p3_far = abs(_p3_hr - best_p_hr) > 0.20 * best_p_hr
        if _mask_p2.any():
            _p3_far = _p3_far and abs(_p3_hr - _p2_hr) > 0.5
        if _p3_far:
            _p3_h = int(_p3_hr)
            _p3_m = int((_p3_hr - _p3_h) * 60)
            ax.axvline(_p3_hr, color="#2196f3", ls=":", lw=1.2, alpha=0.9,
                       label=f"P3 = {_p3_h}:{_p3_m:02d}")
            ax.annotate(
                f"{_p3_power:.2f}",
                xy=(_p3_hr, _p3_power),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, color="#2196f3",
            )

    ax.set_xlabel("Period (hours)", color=_LS_TICK)
    ax.set_ylabel("LS Power", color=_LS_TICK)
    ax.set_title(
        f"Lomb-Scargle\nFAP={fap:.2e}  ({fap_n_iter} iter, {fap_status})",
        color=_LS_TITLE,
    )
    for spine in ax.spines.values():
        spine.set_edgecolor(_LS_SPINE)
    ax.tick_params(colors=_LS_TICK)
    ax.legend(fontsize=8)

    # ── 子圖 2：DFT 振幅譜 ───────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(periods_hr, ls_result["dft_amp"], color=_DFT_LINE, lw=1.0)
    ax.axvline(best_p_hr, color="r", ls="--", lw=1, alpha=0.8)
    ax.set_xlim(0.0, _plot_xmax_hr)

    # DFT 峰值標注
    _dft_peak_idx = int(np.argmax(ls_result["dft_amp"]))
    _dft_peak_p_hr = float(periods_hr[_dft_peak_idx])
    _dft_peak_amp = float(ls_result["dft_amp"][_dft_peak_idx])
    _dft_ph = int(_dft_peak_p_hr)
    _dft_pm = int((_dft_peak_p_hr - _dft_ph) * 60)
    ax.annotate(
        f"P={_dft_ph}:{_dft_pm:02d}\nA={_dft_peak_amp:.3f}",
        xy=(_dft_peak_p_hr, _dft_peak_amp),
        xytext=(6, -14), textcoords="offset points",
        fontsize=8, color=_DFT_TITLE,
        arrowprops=dict(arrowstyle="->", color=_DFT_SPINE, lw=0.8),
    )

    ax.set_xlabel("Period (hours)", color=_DFT_TICK)
    ax.set_ylabel("Amplitude (mag)", color=_DFT_TICK)
    ax.set_title("DFT Amplitude (cross-check)", color=_DFT_TITLE)
    for spine in ax.spines.values():
        spine.set_edgecolor(_DFT_SPINE)
    ax.tick_params(colors=_DFT_TICK)

    # ── 子圖 3：相位折疊圖（展開 2 個週期） ──────────────────────────────────
    ax = axes[2]
    phase = fit_result["phase"]
    phase_ext = np.concatenate([phase, phase + 1.0])
    mag_ext = np.concatenate([mag, mag])
    err_ext = np.concatenate([err, err])
    phi_dense_ext = np.concatenate(
        [fit_result["phi_dense"], fit_result["phi_dense"] + 1.0]
    )
    mag_dense_ext = np.concatenate(
        [fit_result["mag_dense"], fit_result["mag_dense"]]
    )

    ax.errorbar(
        phase_ext, mag_ext, yerr=err_ext,
        fmt="ko", ms=2, elinewidth=0.6, capsize=2, alpha=0.5,
        zorder=1, label="obs ±1σ",
    )
    ax.plot(
        phi_dense_ext, mag_dense_ext,
        "r-", lw=2, zorder=2,
        label=f"Fit (N={fit_result['n_harmonics']})",
    )
    ax.invert_yaxis()
    ax.set_xlabel("Phase (phi=0: max brightness)")
    ax.set_ylabel("Calibrated magnitude")
    ax.set_title(
        f"P = {best_p:.6f} ± {best_p_err:.1e} d\n"
        f"Amp = {fit_result['amplitude']:.3f} mag"
        f"  RMS = {fit_result['rms_residuals']:.3f} mag"
    )

    # 圖例：相位折疊圖右下角
    ax.legend(loc="lower right", fontsize=8, frameon=True, borderpad=0.5, handlelength=2.0)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    logger.info("[繪圖] 週期分析圖已儲存：%s", out_png)


# =============================================================================
# 主入口
# =============================================================================

def run_period_analysis(
    df: pd.DataFrame,
    target_name: str,
    channel: str,
    out_dir: Path,
    config_path: Optional[Path] = None,
    run_prewhitening: bool = True,
    obs_date: str = "",
) -> Dict:
    """
    從測光 CSV DataFrame 執行完整進階週期分析。

    Parameters
    ----------
    df              : 測光 DataFrame，需含欄位 ok, bjd_tdb, m_var,
                      plus v_err or t_sigma_mag.
    target_name     : 目標星名稱（用於圖標題和檔名）。
    channel         : 波段（"R" / "G1" / "B"）。
    out_dir         : 輸出目錄（圖片存放位置）。
    config_path     : observation_config.yaml 路徑；None 時使用預設值。
    run_prewhitening: 是否執行預白化分析（可獨立關閉）。

    Returns
    -------
    dict 包含：
        ls_result, fit_result, prewhitening_results（若啟用）

    Raises
    ------
    ValueError
        有效資料點不足，或相位零點驗證失敗（後者含 t0 資訊）。
    """
    cfg = _load_config(config_path)
    fit_cfg = _get(cfg, "period_analysis", "fourier_fit", default={})
    min_pts = int(_get(fit_cfg, "min_data_points", default=50))
    pw_enabled = bool(
        _get(cfg, "period_analysis", "pre_whitening", "enabled", default=True)
    )

    # 資料篩選
    # ensemble_normalize 啟用時優先用 m_var_norm；否則退回 m_var
    mag_col = "m_var_norm" if "m_var_norm" in df.columns else "m_var"
    logger.info("週期分析使用欄位：%s", mag_col)

    valid_cols = {"ok", "bjd_tdb", mag_col}
    missing = valid_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame 缺少必要欄位：{missing}")

    base_valid = df[
        (df["ok"] == 1)
        & np.isfinite(df["bjd_tdb"])
        & np.isfinite(df[mag_col])
    ].copy()
    if base_valid.empty:
        raise ValueError("沒有有效測光資料（ok=1 且時間/星等有限），略過週期分析。")

    if "v_err" not in df.columns:
        if "t_sigma_mag" in df.columns:
            df = df.copy()
            df["v_err"] = df["t_sigma_mag"]
        else:
            raise ValueError("DataFrame 缺少測光誤差欄位：需要 v_err 或 t_sigma_mag")

    d = df[
        (df["ok"] == 1)
        & np.isfinite(df["bjd_tdb"])
        & np.isfinite(df[mag_col])
        & np.isfinite(df["v_err"])
    ].copy()

    n_valid = len(d)
    if n_valid < min_pts:
        raise ValueError(
            f"有效資料點不足（{n_valid} < {min_pts}），無法進行週期分析。"
            f" 請降低 min_data_points 或增加觀測幀數。"
        )

    t = d["bjd_tdb"].values
    mag = d[mag_col].values
    err = d["v_err"].values

    logger.info(
        "=== period_analysis.py 開始：%s [%s]  N=%d ===",
        target_name, channel, n_valid,
    )

    # LS + DFT
    ls_result = run_ls_and_dft(t, mag, err, cfg)

    # LS 前 N 峰候選挑選（最高峰不保證為真週期）
    cand_sel = select_best_period_candidate(
        t, mag, err, ls_result["freqs"], ls_result["ls_power"], cfg
    )
    ls_result["candidate_selection"] = cand_sel
    if cand_sel["switched"]:
        ls_result["ls_peak_period"] = ls_result["best_period"]
        ls_result["best_period"] = cand_sel["period"]
        ls_result["best_freq"] = 1.0 / cand_sel["period"]

    # 週期精修（週期為自由參數；消除 LS 格網偏差，避免 P/2P 判別失真）
    _mh = int(_get(fit_cfg, "max_harmonics", default=8))
    _ref = _refine_period(t, mag, err, ls_result["best_period"], _mh)
    if _ref is not None:
        logger.info(
            "[精修] 週期 %.6f d -> %.6f d（±%.2e d）",
            ls_result["best_period"], _ref[0], _ref[1],
        )
        ls_result["best_period"] = _ref[0]
        ls_result["best_freq"] = 1.0 / _ref[0]
        ls_result["refined_period_err"] = _ref[1]

    # P vs 2P 判別（W UMa 型食雙星的 LS 峰常落在半週期）
    dp_check = resolve_half_period_ambiguity(
        t, mag, err, ls_result["best_period"], cfg
    )
    ls_result["double_period_check"] = dp_check
    if dp_check["doubled"]:
        ls_result["ls_peak_period"] = ls_result["best_period"]
        ls_result["best_period"] = dp_check["period"]
        ls_result["best_freq"] = 1.0 / dp_check["period"]
        _ref2 = _refine_period(t, mag, err, dp_check["period"], _mh)
        if _ref2 is not None:
            ls_result["best_period"] = _ref2[0]
            ls_result["best_freq"] = 1.0 / _ref2[0]
            ls_result["refined_period_err"] = _ref2[1]

    # ΔBIC 掃描區間（誠實的週期不確定度；共變異誤差在平坦地形會低估）
    _interval = scan_bic_interval(t, mag, err, ls_result["best_period"], cfg)
    if _interval is not None:
        ls_result["bic_interval"] = _interval
        logger.info(
            "[區間] P ∈ [%.4f, %.4f] h（ΔBIC<%.1f）±%.4f h%s",
            _interval["period_lo_d"] * 24.0, _interval["period_hi_d"] * 24.0,
            _interval["delta"], _interval["err_half_width_d"] * 24.0,
            "；觸及掃描邊界，區間為下限" if _interval["touches_edge"] else "",
        )

    # 傅立葉擬合（BIC 選階 + 相位零點驗證）
    # ValueError 在此不捕捉，讓呼叫端看到完整錯誤訊息（含 t0_final）
    fit_result = fit_phase_folded_model(
        t, mag, err, ls_result["best_period"], cfg
    )

    if not fit_result:
        logger.error("[run_period_analysis] 傅立葉擬合失敗，終止。")
        return {"ls_result": ls_result, "fit_result": {}}

    # 輸出主圖
    out_dir.mkdir(parents=True, exist_ok=True)
    ls_csv_path = _write_ls_spectrum_csv(out_dir, target_name, channel, ls_result)
    dft_csv_path = _write_dft_spectrum_csv(out_dir, target_name, channel, ls_result)
    fit_json_path = _write_fourier_fit_json(
        out_dir, target_name, channel, fit_result, ls_result
    )
    model_csv_path = _write_fourier_model_csv(
        out_dir, target_name, channel, fit_result
    )
    out_png = out_dir / (
        f"period_analysis_{target_name.replace(' ', '')}_{channel}.png"
    )
    plot_period_analysis(
        target_name, channel, ls_result, fit_result, t, mag, err, out_png, cfg,
        obs_date=obs_date,
    )

    results: Dict = {"ls_result": ls_result, "fit_result": fit_result}
    results["output_files"] = {
        "ls_spectrum_csv": str(ls_csv_path),
        "dft_spectrum_csv": str(dft_csv_path),
        "fourier_fit_json": str(fit_json_path),
        "fourier_model_csv": str(model_csv_path),
        "period_analysis_png": str(out_png),
    }

    # 預白化
    if run_prewhitening and pw_enabled:
        pw_results = run_pre_whitening(
            t, mag, err, cfg, target_name, channel, out_dir
        )
        results["prewhitening_results"] = pw_results
    else:
        logger.info("[run_period_analysis] 預白化已停用，略過。")

    logger.info(
        "=== period_analysis.py 完成：%s [%s] ===", target_name, channel
    )
    return results
