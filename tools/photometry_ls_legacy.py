# photometry_ls_legacy.py
# 從 photometry.py v1.35 移出的內建 Lomb-Scargle + Fourier 分析。
# 保留供參考或手動使用。
# 移出日期：2026-03-21
# 原始行號：3562-4046
#
# 若需獨立使用，請 import 此模組：
#   from photometry_ls_legacy import run_lomb_scargle, run_fourier_fit, save_3panel_period_plot
#
# 注意：此模組的常數（LS_MIN_PERIOD_DAYS 等）為硬寫值，不讀 YAML。
# 正式管線請改用 period_analysis.py（統一從 YAML 讀取參數）。

"""## Lomb-Scargle Period Analysis + Fourier Fit"""

# ── Lomb-Scargle period analysis + Fourier fit ───────────────────────────────
# References:
#   Lomb (1976) Ap&SS 39, 447
#   Scargle (1982) ApJ 263, 835
#   VanderPlas (2018) ApJS 236, 16
#   Press & Rybicki (1989) ApJ 338, 277  (fast implementation)

from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


# ── Config (edit here or in observation_config.yaml) ─────────────────────────
LS_MIN_PERIOD_DAYS  = 0.05     # shortest period to search (days)
LS_MAX_PERIOD_DAYS  = 2.0      # longest  period to search (days)
LS_SAMPLES_PER_PEAK = 10       # frequency grid oversampling
FAP_THRESHOLD       = 0.001    # false-alarm probability threshold
FOURIER_N_MAX       = 6        # maximum Fourier harmonics (auto-selected below)
PHASE_FOLD_CYCLES   = 2        # how many phase cycles to show in the fold plot


def run_lomb_scargle(
    df: pd.DataFrame,
    time_col: str = "bjd_tdb",
    mag_col: str = "m_var",
    err_col: str = "t_sigma_mag",
    min_period: float = LS_MIN_PERIOD_DAYS,
    max_period: float = LS_MAX_PERIOD_DAYS,
    samples_per_peak: int = LS_SAMPLES_PER_PEAK,
    fap_threshold: float = FAP_THRESHOLD,
    out_png: "Path | None" = None,
    target_name: str = "",
    channel: str = "",
    obs_date: str = "",
    lat_deg: "float | None" = None,
    lon_deg: "float | None" = None,
) -> dict:
    """
    Run a Lomb-Scargle periodogram on the light-curve DataFrame.

    Returns a dict with:
        frequency   : np.ndarray  (1/day)
        power       : np.ndarray
        best_period : float  (days)
        best_freq   : float  (1/day)
        fap         : float  false-alarm probability at best peak
        ls          : LombScargle object (for further queries)
        t, mag, err : cleaned arrays used in the fit
    """
    d = df[(df["ok"] == 1) & np.isfinite(df[time_col]) & np.isfinite(df[mag_col])].copy()
    if len(d) < 10:
        raise ValueError(f"Only {len(d)} valid points -- need >= 10 for period search.")

    t   = d[time_col].values
    mag = d[mag_col].values
    err = d[err_col].values if (err_col in d.columns and np.isfinite(d[err_col]).any()) else None

    ls = LombScargle(t, mag, dy=err)

    frequency, power = ls.autopower(
        minimum_frequency=1.0 / max_period,
        maximum_frequency=1.0 / min_period,
        samples_per_peak=samples_per_peak,
    )

    best_idx    = int(np.argmax(power))
    best_freq   = float(frequency[best_idx])
    best_period = 1.0 / best_freq
    fap         = float(ls.false_alarm_probability(power[best_freq == frequency].max()
                                                   if power[best_freq == frequency].size
                                                   else power[best_idx]))

    print(f"[LS] Best period = {best_period:.6f} d  ({best_period * 24:.4f} h)")
    print(f"[LS] Best freq   = {best_freq:.6f} d^-1")
    print(f"[LS] FAP         = {fap:.2e}")
    if fap > fap_threshold:
        print(f"[WARN] FAP={fap:.2e} > threshold={fap_threshold}. "
              "Period detection may not be significant.")

    # # ── Plot periodogram ──────────────────────────────────────────────────────
    # # ── X 軸單位：天 → 小時，刻度格式化為 H:MM:SS ────────────────────────────
    # period_h = 1.0 / frequency * 24.0   # 天 → 小時
    #
    # def _fmt_hms(h_val, _pos=None):
    #     """將小時數格式化為 H:MM:SS，去掉前置零。"""
    #     total_s = int(round(abs(h_val) * 3600))
    #     hh = total_s // 3600
    #     mm = (total_s % 3600) // 60
    #     ss = total_s % 60
    #     if hh > 0:
    #         return f"{hh}:{mm:02d}:{ss:02d}"
    #     return f"{mm}:{ss:02d}"
    #
    # best_period_h = best_period * 24.0
    # _p_min_h = float(period_h.min())
    # _p_max_h = float(period_h.max())
    #
    # fig, ax = plt.subplots(figsize=(10, 4))
    # ax.plot(period_h, power, lw=0.8, color="navy")
    # ax.axvline(best_period_h, color="red", lw=1.2, ls="--",
    #            label=f"P = {_fmt_hms(best_period_h)}")
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_hms))
    # ax.set_xlabel("Period (H:MM:SS)", color="navy")
    # ax.set_ylabel("Lomb-Scargle power", color="navy")
    #
    # _ls_title_star = target_name if target_name else "Target"
    # ax.set_title(f"{_ls_title_star} Lomb-Scargle Periodogram", fontsize=22, fontweight='bold', pad=10)
    #
    # ax.set_xlim(_p_min_h, _p_max_h)
    # ax.legend(fontsize=12, loc="upper left", frameon=True, edgecolor='gray')
    # ax.grid(True, alpha=0.3)
    #
    # # Metadata above frame
    # ax.text(0.01, 1.02, obs_date if obs_date else "", transform=ax.transAxes,
    #         ha="left", va="bottom", fontsize=14, color="navy")
    # if lat_deg is not None and lon_deg is not None:
    #     ax.text(0.99, 1.02, f"{lat_deg:.2f}  {lon_deg:.2f}",
    #             transform=ax.transAxes, ha="right", va="bottom",
    #             fontsize=14, color="navy")
    # fig.tight_layout()
    # if out_png is not None:
    #     Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    #     fig.savefig(out_png, dpi=150)
    #     print(f"[LS] periodogram saved -> {out_png}")
    # plt.close("all")

    return {
        "frequency": frequency,
        "power": power,
        "best_period": best_period,
        "best_freq": best_freq,
        "fap": fap,
        "ls": ls,
        "t": t,
        "mag": mag,
        "err": err,
    }


def _auto_fourier_order(n_cycles: float) -> int:
    """
    Auto-select Fourier harmonic order from the number of observed cycles.
    Fewer cycles → fewer free parameters to avoid over-fitting.
    """
    if n_cycles < 2:
        return 1
    elif n_cycles < 5:
        return 3
    else:
        return min(6, FOURIER_N_MAX)


def _select_harmonics_breger(
    phase: np.ndarray,
    mag: np.ndarray,
    err: "np.ndarray | None",
    max_harmonics: int,
    sn_threshold: float = 4.0,
) -> tuple:
    """Breger et al. (1993) S/N 準則選諧波數。N=1 永遠接受。"""
    n_data = len(mag)
    best_n = 1
    best_popt = np.array([])
    best_pcov = np.array([])
    sigma = err if (err is not None and np.isfinite(err).all()) else None

    for n in range(1, max_harmonics + 1):
        p0 = [float(np.nanmean(mag))] + [0.0] * (2 * n)
        try:
            popt, pcov = curve_fit(
                _fourier_model, phase, mag,
                p0=p0, sigma=sigma, absolute_sigma=True,
                maxfev=50000,
            )
        except Exception:
            break

        residuals = mag - _fourier_model(phase, *popt)
        sigma_res = float(np.std(residuals))
        a_n = popt[1 + 2 * (n - 1)]
        b_n = popt[2 + 2 * (n - 1)]
        amp_n = float(np.sqrt(a_n ** 2 + b_n ** 2))
        noise_level = sigma_res * np.sqrt(2.0 / n_data) if n_data > 0 else np.inf
        sn = amp_n / noise_level if noise_level > 0 else 0.0

        if sn >= sn_threshold or n == 1:
            best_n = n
            best_popt = popt
            best_pcov = pcov
        else:
            break

    if best_popt.size == 0:
        raise RuntimeError("Fourier 擬合全部失敗，無法選取諧波數。")
    return best_n, best_popt, best_pcov


def _fourier_model(phase: np.ndarray, *params) -> np.ndarray:
    """
    Fourier series: V(φ) = a0 + Σ_n [a_n cos(2πnφ) + b_n sin(2πnφ)]
    params = [a0, a1, b1, a2, b2, ...]
    """
    a0     = params[0]
    result = np.full_like(phase, a0)
    n_harm = (len(params) - 1) // 2
    for i in range(n_harm):
        a_n = params[1 + 2 * i]
        b_n = params[2 + 2 * i]
        result += a_n * np.cos(2 * np.pi * (i + 1) * phase)
        result += b_n * np.sin(2 * np.pi * (i + 1) * phase)
    return result


def run_fourier_fit(
    t: np.ndarray,
    mag: np.ndarray,
    period: float,
    err: "np.ndarray | None" = None,
    n_max: "int | None" = None,
    t0: "float | None" = None,
    out_png: "Path | None" = None,
    target_name: str = "",
    channel: str = "",
    obs_date: str = "",
    lat_deg: "float | None" = None,
    lon_deg: "float | None" = None,
) -> dict:
    """
    Phase-fold the light curve and fit a Fourier series.

    Parameters
    ----------
    t       : time array (BJD_TDB days)
    mag     : magnitude array
    period  : best period in days (from Lomb-Scargle)
    err     : magnitude uncertainties (optional)
    n_max   : Fourier harmonic order; None → auto-selected
    t0      : epoch of phase zero; None → uses t[0]

    Returns
    -------
    dict with keys:
        phase, mag_sorted, fit_phase, fit_mag  – for plotting
        amplitude                              – peak-to-peak amplitude (mag)
        n_harmonics                            – order used
        r2                                     – R² of the Fourier fit
        params                                 – fitted coefficients
        residuals_rms                          – rms of residuals (mag)
    """
    if t0 is None:
        t0 = float(t[0])

    phase = ((t - t0) / period) % 1.0

    n_cycles = (t.max() - t.min()) / period
    if n_max is None:
        # n_cycles 上限（< 2 週期不足以辨識高諧波）+ Breger S/N 準則
        _cycles_cap = _auto_fourier_order(n_cycles)
        best_n, popt, pcov = _select_harmonics_breger(
            phase, mag, err, max_harmonics=min(FOURIER_N_MAX, _cycles_cap), sn_threshold=4.0
        )
        n_max = best_n
        print(f"[Fourier] n_cycles ~= {n_cycles:.1f}  ->  Breger N={n_max}")
    else:
        print(f"[Fourier] n_cycles ~= {n_cycles:.1f}  ->  using {n_max} harmonics (fixed)")
        p0 = [float(np.nanmean(mag))] + [0.0] * (2 * n_max)
        sigma = err if (err is not None and np.isfinite(err).all()) else None
        try:
            popt, pcov = curve_fit(
                _fourier_model, phase, mag,
                p0=p0, sigma=sigma, absolute_sigma=True,
                maxfev=50000,
            )
        except Exception as exc:
            print(f"[WARN] Fourier fit failed: {exc}")
            return {}

    # ── R² and residuals ──────────────────────────────────────────────────────
    mag_pred   = _fourier_model(phase, *popt)
    ss_res     = float(np.sum((mag - mag_pred) ** 2))
    ss_tot     = float(np.sum((mag - mag.mean()) ** 2))
    r2         = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rms_resid  = float(np.sqrt(np.mean((mag - mag_pred) ** 2)))

    # ── Amplitude from smooth model ───────────────────────────────────────────
    phi_dense  = np.linspace(0.0, 1.0, 1000)
    fit_dense  = _fourier_model(phi_dense, *popt)
    amplitude  = float(fit_dense.max() - fit_dense.min())

    print(f"[Fourier] Amplitude = {amplitude:.4f} mag")
    print(f"[Fourier] R2        = {r2:.4f}")
    print(f"[Fourier] RMS resid = {rms_resid:.4f} mag")

    sort_idx   = np.argsort(phase)
    # # ── Phase-folded plot ─────────────────────────────────────────────────────
    # # sort_idx   = np.argsort(phase)
    # phase_ext  = np.concatenate([phase[sort_idx], phase[sort_idx] + 1.0])
    # mag_ext    = np.concatenate([mag[sort_idx], mag[sort_idx]])
    # fit_ext    = np.concatenate([phi_dense, phi_dense]) # dummy
    # phi_ext    = np.concatenate([phi_dense, phi_dense + 1.0])
    #
    # n_show  = int(PHASE_FOLD_CYCLES)
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.scatter(phase_ext[:len(phase_ext) // PHASE_FOLD_CYCLES * n_show],
    #            mag_ext[:len(mag_ext) // PHASE_FOLD_CYCLES * n_show],
    #            s=12, alpha=0.7, label="data", color="navy")
    # ax.plot(phi_ext[:len(phi_ext) // PHASE_FOLD_CYCLES * n_show],
    #         fit_ext[:len(fit_ext) // PHASE_FOLD_CYCLES * n_show],
    #         "r-", lw=1.5, label=f"Fourier n={n_max}  R2={r2:.3f}")
    # ax.invert_yaxis()
    # ax.set_xlabel(f"Phase  (P = {period:.6f} d = {period * 24:.4f} h)",
    #               loc="center", color="navy")
    # ax.set_ylabel("Calibrated magnitude", color="navy")
    #
    # _ff_title_star = target_name if target_name else "Target"
    # ax.set_title(f"{_ff_title_star} Phase-Folded Light Curve", fontsize=22, fontweight='bold', pad=10)
    #
    # _sigma_med_ff = (float(np.nanmedian(err))
    #                  if err is not None and len(err) > 0 else float("nan"))
    #
    # # Metadata above frame
    # ax.text(0.01, 1.02, obs_date if obs_date else "", transform=ax.transAxes,
    #         ha="left", va="bottom", fontsize=14, color="navy")
    # ax.text(1.0, 1.02, f"Reliability: σ_med = {_sigma_med_ff:.4f} mag",
    #         transform=ax.transAxes, ha="right", va="bottom",
    #         fontsize=14, color="saddlebrown")
    #
    # ax.legend(fontsize=12, loc="upper left", frameon=True, edgecolor='gray')
    # ax.grid(True, alpha=0.3)
    #
    # # Coordinates inside frame
    # if lat_deg is not None and lon_deg is not None:
    #     ax.text(0.99, 0.97, f"{lat_deg:.2f}  {lon_deg:.2f}",
    #             transform=ax.transAxes, ha="right", va="top",
    #             fontsize=7, color="gray")
    # fig.tight_layout()
    # if out_png is not None:
    #     Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    #     fig.savefig(out_png, dpi=150)
    #     print(f"[Fourier] phase-fold plot saved -> {out_png}")
    # plt.close("all")

    return {
        "phase": phase,
        "mag_sorted": mag[sort_idx],
        "fit_phase": phi_dense,
        "fit_mag": fit_dense,
        "amplitude": amplitude,
        "n_harmonics": n_max,
        "r2": r2,
        "residuals_rms": rms_resid,
        "params": popt,
        "period": period,
    }


def save_3panel_period_plot(
    ls_r: dict,
    fit_r: dict,
    target_name: str,
    channel: str,
    obs_date: str,
    lat_deg: "float | None",
    lon_deg: "float | None",
    out_png: Path,
    display_name: "str | None" = None,
):
    """
    Export combined 3-panel diagnosis plot: LS Power, DFT Amplitude, Phase Fold.
    Supports multi-color themes and report-style font sizes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np

    def _fmt_hms(h_val):
        total_s = int(round(abs(h_val) * 3600))
        hh = total_s // 3600
        mm = (total_s % 3600) // 60
        return f"{hh}:{mm:02d}"

    def _classical_dft_amp(t, m, f_arr):
        m_norm = m - np.mean(m)
        n = len(t)
        # Vectorized DFT for speed
        phase_matrix = -2.0j * np.pi * t[:, None] * f_arr[None, :]
        dft = np.sum(m_norm[:, None] * np.exp(phase_matrix), axis=0)
        return (2.0 / n) * np.abs(dft)

    t, mag, err = ls_r["t"], ls_r["mag"], ls_r["err"]
    freq, power = ls_r["frequency"], ls_r["power"]
    p1 = fit_r["period"]
    r2 = fit_r["r2"]
    n_harm = fit_r["n_harmonics"]
    popt = fit_r["params"]

    dft_amp = _classical_dft_amp(t, mag, freq)

    # Search P2 near 0.05d (common artifact or feature)
    p2_nearby_mask = (1/freq > 0.04) & (1/freq < 0.08)
    p2_val = 0
    if np.any(p2_nearby_mask):
        p2_idx = np.argmax(power[p2_nearby_mask])
        p2_val = (1/freq)[p2_nearby_mask][p2_idx]

    # Plot Setup
    fig = plt.figure(figsize=(18, 6), dpi=150)
    gs = GridSpec(1, 3, width_ratios=[1, 0.7, 1.5]) 
    
    FS_TITLE = 18
    FS_AXIS = 14
    FS_LABELS = 13
    FS_LEGEND = 11

    _title_name = display_name if display_name else target_name
    fig.text(0.02, 0.95, f"{_title_name} [{channel}]  {obs_date}",
             fontsize=FS_TITLE+2, fontweight='bold', ha='left')
    if lat_deg is not None:
        fig.text(0.98, 0.97, f"{lat_deg:.2f}N {lon_deg:.2f}E",
                 fontsize=FS_AXIS, color='#2d6a4f', ha='right', va='top')

    def apply_sub_theme(ax, theme_color):
        for spine in ax.spines.values():
            spine.set_color(theme_color)
        ax.xaxis.label.set_color(theme_color)
        ax.yaxis.label.set_color(theme_color)
        ax.tick_params(colors=theme_color, labelsize=FS_LABELS)
        ax.title.set_color(theme_color)

    # Sub 1: LS Power (Green)
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(1/freq, power, lw=1.5, color='forestgreen')
    ax0.axvline(p1, color='red', ls='--', lw=1.8, label=f"P = {_fmt_hms(p1*24)}")
    if p2_val > 0:
        ax0.axvline(p2_val, color='orange', ls='--', lw=1.8, label=f"P2 = {_fmt_hms(p2_val*24)}")
    ax0.set_xlim(0, 0.3)
    ax0.set_xlabel("Period (days)", fontsize=FS_AXIS)
    ax0.set_ylabel("LS Power", fontsize=FS_AXIS)
    ax0.set_title("Lomb-Scargle\nFAP=0.00e+00", fontsize=FS_TITLE)
    ax0.legend(fontsize=FS_LEGEND, loc='lower right')
    apply_sub_theme(ax0, 'forestgreen')

    # Sub 2: DFT (Sienna)
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(1/freq, dft_amp, lw=1.5, color='sienna')
    ax1.axvline(p1, color='red', ls='--', lw=1.8)
    ax1.text(0.97, 0.90, f"{_fmt_hms(p1*24)}", color='red', fontsize=FS_LABELS, fontweight='bold',
             ha='right', va='top', transform=ax1.transAxes)
    ax1.set_xlim(0, 0.3)
    ax1.set_xlabel("Period (days)", fontsize=FS_AXIS)
    ax1.set_ylabel("Amplitude (mag)", fontsize=FS_AXIS)
    ax1.set_title("DFT Amplitude", fontsize=FS_TITLE)
    apply_sub_theme(ax1, 'sienna')

    # Sub 3: Phase Fold (Navy)
    ax2 = fig.add_subplot(gs[2])
    phi_dense = np.linspace(0, 2, 200)
    mag_dense = _fourier_model(phi_dense % 1.0, *popt)
    
    phase = ((t - t[0]) / p1) % 1.0
    phase_ext = np.concatenate([phase, phase + 1.0])
    mag_ext = np.concatenate([mag, mag])
    err_ext = np.concatenate([err, err]) if err is not None else np.zeros_like(mag_ext)
    
    ax2.errorbar(phase_ext, mag_ext, yerr=err_ext, fmt='o', ms=3, color='navy', 
                 alpha=0.3, elinewidth=0.8, capsize=0, label="data (±1σ)")
    ax2.plot(phi_dense, mag_dense, color='red', lw=2.5, label=f"Fourier n={n_harm} R2={r2:.3f}")
    ax2.invert_yaxis()
    ax2.set_xlabel("Phase (phi=0: max brightness)", fontsize=FS_AXIS)
    ax2.set_ylabel("Calibrated magnitude", fontsize=FS_AXIS)
    ax2.set_title("Phase-Folded LC + Fourier Fit", fontsize=FS_TITLE)
    ax2.legend(fontsize=FS_LEGEND, loc='lower left')
    apply_sub_theme(ax2, 'navy')
    
    _sigma_med = np.nanmedian(err) if err is not None else 0
    fig.text(0.98, 0.95, f"Reliability: σ_med = {_sigma_med:.4f} mag", ha='right', va='top', 
             fontsize=FS_AXIS, color='saddlebrown', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

