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
            "period_max_hr": 24.0,
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
        },
        "pre_whitening": {
            "enabled": True,
            "sn_threshold": 4.0,
            "max_frequencies": 10,
            "save_csv": False,
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
    讀取 observation_config.yaml。
    若路徑不存在或解析失敗，回退至預設值並記錄警告。
    """
    if config_path is None or not config_path.exists():
        logger.warning(
            "找不到 observation_config.yaml，使用預設參數。"
            "（路徑：%s）", config_path
        )
        return _DEFAULT_CONFIG

    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        if not cfg:
            return _DEFAULT_CONFIG

        # ── 推算 project_root（支援相對路徑）────────────────────────────────
        try:
            import google.colab  # noqa: F401
            root = cfg["paths"]["colab"]["project_root"]
        except (ImportError, KeyError):
            root = cfg["paths"]["local"]["project_root"]
        p = Path(root)
        if not p.is_absolute():
            p = (Path(config_path).parent / p).resolve()
        cfg["_project_root"] = p
        cfg["_data_root"] = p / "data"

        return cfg
    except yaml.YAMLError as exc:
        logger.warning("observation_config.yaml 解析失敗：%s，使用預設參數。", exc)
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


def _select_harmonics_bic(
    phase: np.ndarray,
    mag: np.ndarray,
    err: np.ndarray,
    max_harmonics: int,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    以 BIC 最小化自動選取最佳傅立葉諧波數。

    搜尋範圍：N = 1 … max_harmonics。
    對每個 N 嘗試擬合，收集 BIC；選取 BIC 最小的 N。
    若所有 N 均擬合失敗，raise RuntimeError。

    Returns
    -------
    (best_n, best_popt, best_pcov)
    """
    bic_values: List[float] = []
    results: List[Tuple[np.ndarray, np.ndarray]] = []

    for n in range(1, max_harmonics + 1):
        try:
            popt, pcov = _fit_fourier(phase, mag, err, n)
            mag_pred = _fourier_series(phase, *popt)
            n_params = 2 * n + 1
            bic = _compute_bic(mag, mag_pred, n_params, err)
            bic_values.append(bic)
            results.append((popt, pcov))
            logger.debug("  N=%d  BIC=%.2f", n, bic)
        except (RuntimeError, ValueError) as exc:
            logger.debug("  N=%d 擬合失敗：%s", n, exc)
            bic_values.append(np.inf)
            results.append((np.array([]), np.array([])))

    if all(np.isinf(b) for b in bic_values):
        raise RuntimeError("所有諧波數均擬合失敗，無法選取最佳階數。")

    best_n = int(np.argmin(bic_values)) + 1
    best_popt, best_pcov = results[best_n - 1]
    logger.info(
        "BIC 選階完成：最佳 N=%d，BIC=%.2f（搜尋範圍 N=1–%d）",
        best_n, bic_values[best_n - 1], max_harmonics,
    )
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
    period_max_days = _get(ls_cfg, "period_max_hr", default=24.0) / 24.0
    oversampling = _get(ls_cfg, "oversampling", default=10)
    max_iter = _get(ls_cfg, "fap_bootstrap_max_iter", default=1000)
    converge_window = _get(ls_cfg, "fap_bootstrap_converge_window", default=100)
    converge_tol = _get(ls_cfg, "fap_bootstrap_converge_tol", default=0.05)
    fap_threshold = _get(ls_cfg, "fap_threshold", default=0.001)

    min_freq = 1.0 / period_max_days
    max_freq = 1.0 / period_min_days

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
    }


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

    freq = 1.0 / period
    phi_dense = np.linspace(0.0, 1.0, 10_000)

    # ── Step 1：暫時零點擬合 ─────────────────────────────────────────────────
    t0_temp = float(t[0])
    phase_temp = ((t - t0_temp) / period) % 1.0

    logger.info("[Step 1] BIC 選階（最大諧波數 N=%d）…", max_harmonics)
    try:
        best_n, popt_temp, _ = _select_harmonics_bic(
            phase_temp, mag, err, max_harmonics
        )
    except RuntimeError as exc:
        logger.error("[Step 1] BIC 選階失敗：%s", exc)
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
    period_max_days = _get(ls_cfg, "period_max_hr", default=24.0) / 24.0
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
        ax.set_title(f"S/N = {sn:.2f}  ({'✓ 繼續' if sn >= sn_threshold else '✗ 停止'})")

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

    # CSV 預留（目前停用）
    if save_csv:
        # TODO：啟用時在此實作 CSV 輸出
        # _save_prewhitening_csv(extracted, out_dir, target_name, channel)
        logger.info("[Pre-whitening] save_csv=True 但函式尚未實作，略過。")

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

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=dpi)
    fig.suptitle(
        f"Period Analysis: {target_name} [{channel}]",
        fontsize=12,
    )

    # 子圖 1：LS 週期圖
    ax = axes[0]
    ax.plot(1.0 / freqs, ls_result["ls_power"], "k-", lw=0.5)
    ax.axvline(best_p, color="r", ls="--", lw=1, alpha=0.8, label=f"P={best_p:.4f} d")
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("LS Power")
    ax.set_title(
        f"Lomb-Scargle\nFAP={fap:.2e}  ({fap_n_iter} iter, {fap_status})"
    )
    ax.legend(fontsize=8)

    # 子圖 2：DFT 振幅譜
    ax = axes[1]
    ax.plot(1.0 / freqs, ls_result["dft_amp"], "b-", lw=0.5)
    ax.axvline(best_p, color="r", ls="--", lw=1, alpha=0.8)
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("Amplitude (mag)")
    ax.set_title("DFT Amplitude (cross-check)")

    # 子圖 3：相位折疊圖（展開 2 個週期）
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
        fmt="ko", ms=2, elinewidth=0.6, capsize=0, alpha=0.5,
        zorder=1, label="Data",
    )
    ax.plot(
        phi_dense_ext, mag_dense_ext,
        "r-", lw=2, zorder=2,
        label=f"Fit (N={fit_result['n_harmonics']})",
    )
    ax.invert_yaxis()
    ax.set_xlabel("Phase (phi=0: max brightness)")
    ax.set_ylabel("Diff Magnitude")
    ax.set_title(
        f"P = {best_p:.6f} ± {best_p_err:.1e} d\n"
        f"Amp = {fit_result['amplitude']:.3f} mag"
        f"  RMS = {fit_result['rms_residuals']:.3f} mag"
    )
    ax.legend(fontsize=8)

    plt.tight_layout()
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
) -> Dict:
    """
    從測光 CSV DataFrame 執行完整進階週期分析。

    Parameters
    ----------
    df              : 測光 DataFrame，需含欄位 ok, bjd_tdb, m_var, v_err。
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
    valid_cols = {"ok", "bjd_tdb", "m_var", "v_err"}
    missing = valid_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame 缺少必要欄位：{missing}")

    d = df[
        (df["ok"] == 1)
        & np.isfinite(df["bjd_tdb"])
        & np.isfinite(df["m_var"])
        & np.isfinite(df["v_err"])
    ].copy()

    n_valid = len(d)
    if n_valid < min_pts:
        raise ValueError(
            f"有效資料點不足（{n_valid} < {min_pts}），無法進行週期分析。"
            f" 請降低 min_data_points 或增加觀測幀數。"
        )

    t = d["bjd_tdb"].values
    mag = d["m_var"].values
    err = d["v_err"].values

    logger.info(
        "=== period_analysis.py 開始：%s [%s]  N=%d ===",
        target_name, channel, n_valid,
    )

    # LS + DFT
    ls_result = run_ls_and_dft(t, mag, err, cfg)

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
    out_png = out_dir / (
        f"period_analysis_{target_name.replace(' ', '')}_{channel}.png"
    )
    plot_period_analysis(
        target_name, channel, ls_result, fit_result, t, mag, err, out_png, cfg
    )

    results: Dict = {"ls_result": ls_result, "fit_result": fit_result}

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
