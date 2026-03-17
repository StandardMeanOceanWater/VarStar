import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
import sys
import os

# Ensure we can import from pipeline
script_dir = Path(__file__).parent.absolute()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

def _fmt_hms(h_val):
    total_s = int(round(abs(h_val) * 3600))
    hh = total_s // 3600
    mm = (total_s % 3600) // 60
    return f"{hh}:{mm:02d}"

def _fourier_model(phase, *params):
    a0 = params[0]
    result = np.full_like(phase, a0)
    n_harm = (len(params) - 1) // 2
    for i in range(n_harm):
        a_n = params[1 + 2 * i]
        b_n = params[2 + 2 * i]
        result += a_n * np.cos(2 * np.pi * (i + 1) * phase)
        result += b_n * np.sin(2 * np.pi * (i + 1) * phase)
    return result

def replot_period():
    target = "V1162Ori"
    display_name = "V1162 ORI"
    date_str = "20251220"
    channel = "G2"
    
    csv_path = Path(f"d:/VarStar/data/targets/{target}/output/photometry_{channel}_{date_str}.csv")
    out_png = Path(f"C:/Users/JIN/.gemini/antigravity/brain/2308730e-59e1-4a3f-acfd-9497074949bf/period_analysis_{channel}_{date_str}_3panel_v5.png")
    
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    d = df[(df["ok"] == 1) & np.isfinite(df["bjd_tdb"]) & np.isfinite(df["m_var"])].copy()
    
    t = d["bjd_tdb"].values
    mag = d["m_var"].values
    err = d["t_sigma_mag"].values if "t_sigma_mag" in d.columns else None
    
    # LS
    ls = LombScargle(t, mag, dy=err)
    freq, power = ls.autopower(minimum_frequency=1/0.3, maximum_frequency=1/0.05, samples_per_peak=10)
    
    best_idx = np.argmax(power)
    f1 = freq[best_idx]
    p1 = 1/f1
    
    # Search P2 towards zero from P1
    # P1 is at 1/f1. "Towards zero" in Period means "towards infinity" in Frequency, 
    # but the image says "P2往P1的前面(零點方向)找", which usually means shorter period (higher frequency).
    # Re-reading: "P2往P1的前面(零點方向)找" -> towards P=0 (Frequency inf).
    mask_p2 = freq > f1
    if any(mask_p2):
        p2_idx_in_mask = np.argmax(power[mask_p2])
        f2 = freq[mask_p2][p2_idx_in_mask]
        p2 = 1/f2
    else:
        p2 = 0
        f2 = 0

    # DFT (Simplified for plot)
    def classical_dft_amp(t, m, f_arr):
        m_norm = m - np.mean(m)
        n = len(t)
        res = []
        for f in f_arr:
            phase = -2.0j * np.pi * t * f
            dft = np.sum(m_norm * np.exp(phase))
            res.append((2.0/n) * np.abs(dft))
        return np.array(res)
    
    dft_amp = classical_dft_amp(t, mag, freq)
    
    # Phase Fold
    phase = ((t - t[0]) / p1) % 1.0
    n_fit = 3
    p0 = [np.mean(mag)] + [0.0] * (2 * n_fit)
    popt, _ = curve_fit(_fourier_model, phase, mag, p0=p0)
    
    mag_pred = _fourier_model(phase, *popt)
    ss_res = np.sum((mag - mag_pred)**2)
    ss_tot = np.sum((mag - np.mean(mag))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Plotting: Use gridspec to adjust widths
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(18, 6), dpi=150)
    gs = GridSpec(1, 3, width_ratios=[1, 0.7, 1.5]) 
    
    # Global Title info
    lat_val = 24.82 
    lon_val = 121.01
    coord_str = f"{lat_val:.2f}N {lon_val:.2f}E"
    
    # Fonts
    FS_TITLE = 18
    FS_AXIS = 14
    FS_LABELS = 13
    FS_LEGEND = 11
    
    # Global Title (V1162 ORI)
    fig.text(0.02, 0.95, f"{target.upper()} [{channel}]  {date_str}  {coord_str}", 
             fontsize=FS_TITLE+2, fontweight='bold', ha='left')
    
    def apply_sub_theme(ax, theme_color):
        ax.spines['bottom'].set_color(theme_color)
        ax.spines['top'].set_color(theme_color)
        ax.spines['right'].set_color(theme_color)
        ax.spines['left'].set_color(theme_color)
        ax.xaxis.label.set_color(theme_color)
        ax.yaxis.label.set_color(theme_color)
        ax.tick_params(colors=theme_color, labelsize=FS_LABELS)
        ax.title.set_color(theme_color)

    # Sub 1: LS Power (Green theme)
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(1/freq, power, lw=1.5, color='forestgreen')
    ax0.axvline(p1, color='red', ls='--', lw=1.8, label=f"P = {_fmt_hms(p1*24)}")
    
    p2_nearby_mask = (1/freq > 0.04) & (1/freq < 0.08)
    if any(p2_nearby_mask):
        p2_idx = np.argmax(power[p2_nearby_mask])
        p2_val = (1/freq)[p2_nearby_mask][p2_idx]
        ax0.axvline(p2_val, color='orange', ls='--', lw=1.8, label=f"P2 = {_fmt_hms(p2_val*24)}")
    
    ax0.set_xlim(0, 0.3)
    ax0.set_xlabel("Period (days)", fontsize=FS_AXIS)
    ax0.set_ylabel("LS Power", fontsize=FS_AXIS)
    ax0.set_title("Lomb-Scargle\nFAP=0.00e+00", fontsize=FS_TITLE)
    ax0.legend(fontsize=FS_LEGEND, loc='lower right')
    apply_sub_theme(ax0, 'forestgreen')
    
    # Sub 2: DFT (Sienna theme)
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(1/freq, dft_amp, lw=1.5, color='sienna')
    ax1.axvline(p1, color='red', ls='--', lw=1.8)
    ax1.text(p1, ax1.get_ylim()[1]*0.9, f" {_fmt_hms(p1*24)}", color='red', fontsize=FS_LABELS, fontweight='bold')
    ax1.set_xlim(0, 0.3)
    ax1.set_xlabel("Period (days)", fontsize=FS_AXIS)
    ax1.set_ylabel("Amplitude (mag)", fontsize=FS_AXIS)
    ax1.set_title("DFT Amplitude", fontsize=FS_TITLE)
    apply_sub_theme(ax1, 'sienna')
    
    # Sub 3: Phase Fold (Navy theme)
    ax2 = fig.add_subplot(gs[2])
    phi_dense = np.linspace(0, 2, 200)
    mag_dense = _fourier_model(phi_dense % 1.0, *popt)
    
    phase_ext = np.concatenate([phase, phase + 1.0])
    mag_ext = np.concatenate([mag, mag])
    err_ext = np.concatenate([err, err]) if err is not None else np.zeros_like(mag_ext)
    
    ax2.errorbar(phase_ext, mag_ext, yerr=err_ext, fmt='o', ms=3, color='navy', 
                 alpha=0.3, elinewidth=0.8, capsize=0, label="data (±1σ)")
    ax2.plot(phi_dense, mag_dense, color='red', lw=2.5, label=f"Fourier n={n_fit} R2={r2:.3f}")
    ax2.invert_yaxis()
    ax2.set_xlabel("Phase (phi=0: max brightness)", fontsize=FS_AXIS)
    ax2.set_ylabel("Calibrated magnitude", fontsize=FS_AXIS)
    ax2.set_title("Phase-Folded LC + Fourier Fit", fontsize=FS_TITLE)
    ax2.legend(fontsize=FS_LEGEND, loc='lower left')
    apply_sub_theme(ax2, 'navy')
    
    # Reliability top right
    _sigma_med = np.nanmedian(err) if err is not None else 0
    fig.text(0.98, 0.95, f"Reliability: σ_med = {_sigma_med:.4f} mag", ha='right', va='top', 
             fontsize=FS_AXIS, color='saddlebrown', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(out_png, bbox_inches='tight')
    print(f"Saved: {out_png}")
    
    # Trigger sound
    os.system('powershell -c "[console]::beep(440,300); [console]::beep(659,300)"')

if __name__ == "__main__":
    replot_period()
