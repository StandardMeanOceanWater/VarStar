# -*- coding: utf-8 -*-
"""
phot_core.py — 孔徑測光核心（純數學函式，無全域狀態依賴）

從 photometry.py 拆出。包含：
  - gaussian2d / fit_gaussian_psf  (PSF 擬合)
  - m_inst_from_flux               (儀器星等)
  - max_pixel_in_box               (最大像素值)
  - compute_annulus_radii           (背景環半徑)
  - aperture_photometry             (孔徑測光)
  - is_saturated                    (飽和判定)
"""

import numpy as np
from scipy.optimize import curve_fit
from astropy.wcs import WCS


# ── PSF 擬合 ──────────────────────────────────────────────────────────────────

def gaussian2d(coords, A, x0, y0, sx, sy, B):
    x, y = coords
    g = A * np.exp(-(((x - x0) ** 2) / (2 * sx ** 2)
                     + ((y - y0) ** 2) / (2 * sy ** 2))) + B
    return g.ravel()


def fit_gaussian_psf(image: np.ndarray, x: float, y: float, box: int = 25) -> dict:
    """
    Fit a 2-D elliptical Gaussian to a star cutout.

    Returns a dict with keys: ok, x0, y0, fwhm_x, fwhm_y, flux_net, b_sky.
    ok == 0 means the fit failed or the result is unphysical.
    """
    half = box // 2
    x0i, y0i = int(round(x)), int(round(y))
    y_min = max(y0i - half, 0)
    y_max = min(y0i + half + 1, image.shape[0])
    x_min = max(x0i - half, 0)
    x_max = min(x0i + half + 1, image.shape[1])
    cut   = image[y_min:y_max, x_min:x_max].astype(np.float64)
    yy, xx = np.mgrid[y_min:y_max, x_min:x_max]

    b_sky0 = float(np.median(cut))
    A0     = float(np.nanmax(cut) - b_sky0)
    if not np.isfinite(A0) or A0 <= 0:
        return {"ok": 0}

    iy, ix = np.unravel_index(np.argmax(cut), cut.shape)
    x0c, y0c = x_min + ix, y_min + iy

    p0    = (A0, float(x0c), float(y0c), 2.0, 2.0, b_sky0)
    lower = (0.0, x_min, y_min, 0.5, 0.5, float(np.min(cut)))
    upper = (A0 * 3.0, x_max - 1, y_max - 1, 15.0, 15.0, float(np.max(cut)))

    try:
        popt, _ = curve_fit(gaussian2d, (xx, yy), cut.ravel(),
                            p0=p0, bounds=(lower, upper), maxfev=12000)
    except Exception:
        return {"ok": 0}

    A, x0f, y0f, sx, sy, b_sky = map(float, popt)

    # Sanity checks before accepting the fit
    if abs(sx) < 0.3 or abs(sy) < 0.3:
        return {"ok": 0}
    flux_net = A * 2.0 * np.pi * abs(sx) * abs(sy)
    if not np.isfinite(flux_net) or flux_net <= 0:
        return {"ok": 0}

    fwhm_x = 2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(sx)
    fwhm_y = 2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(sy)

    return {
        "ok": 1,
        "x0": x0f, "y0": y0f,
        "fwhm_x": fwhm_x, "fwhm_y": fwhm_y,
        "flux_net": flux_net,
        "b_sky": b_sky,
    }


# ── 儀器星等 ──────────────────────────────────────────────────────────────────

def m_inst_from_flux(flux_net: float) -> float:
    if (flux_net is None) or (not np.isfinite(flux_net)) or flux_net <= 0:
        return np.nan
    return float(-2.5 * np.log10(flux_net))


# ── 像素工具 ──────────────────────────────────────────────────────────────────

def max_pixel_in_box(image: np.ndarray, x: float, y: float, box: int = 5) -> float:
    half = int(box) // 2
    x0, y0 = int(round(x)), int(round(y))
    y_min = max(y0 - half, 0)
    y_max = min(y0 + half + 1, image.shape[0])
    x_min = max(x0 - half, 0)
    x_max = min(x0 + half + 1, image.shape[1])
    cut = image[y_min:y_max, x_min:x_max]
    if cut.size == 0:
        return np.nan
    return float(np.nanmax(cut))


# ── 背景環半徑 ────────────────────────────────────────────────────────────────

def compute_annulus_radii(
    r: float,
    r_in: "float | None" = None,
    r_out: "float | None" = None,
) -> "tuple[float, float]":
    """背景環半徑（DESIGN_DECISIONS_v6.md §3.6.2）。

    自動公式（Howell, 2006）：
        r_in  = max(r * 1.5, r + 5.0)     # 至少隔離 0.5r，避開 PSF 翼部
        r_out = max(r * 2.5, r_in + 10.0) # 至少 10 px 寬，確保足夠背景像素

    若 yaml sky_annulus_inner_px / sky_annulus_outer_px 有填值，
    則以 yaml 值覆蓋自動公式。
    """
    if r_in is not None:
        r_in_eff = float(r_in)
    else:
        r_in_eff = max(r * 1.5, r + 5.0)

    if r_out is not None:
        r_out_eff = float(r_out)
    else:
        r_out_eff = max(r * 2.5, r_in_eff + 10.0)

    return r_in_eff, r_out_eff


# ── 孔徑測光 ──────────────────────────────────────────────────────────────────

def aperture_photometry(
    image: np.ndarray,
    x: float,
    y: float,
    r: float,
    r_in: float,
    r_out: float,
) -> dict:
    h, w = image.shape
    x0, y0 = float(x), float(y)
    x_min = max(int(np.floor(x0 - r_out - 1)), 0)
    x_max = min(int(np.ceil(x0 + r_out + 1)), w - 1)
    y_min = max(int(np.floor(y0 - r_out - 1)), 0)
    y_max = min(int(np.ceil(y0 + r_out + 1)), h - 1)
    if x_min >= x_max or y_min >= y_max:
        return {"ok": 0}
    cut = image[y_min:y_max + 1, x_min:x_max + 1].astype(np.float64)
    yy, xx = np.mgrid[y_min:y_max + 1, x_min:x_max + 1]
    r2 = (xx - x0) ** 2 + (yy - y0) ** 2
    ap_mask  = r2 <= r ** 2
    ann_mask = (r2 >= r_in ** 2) & (r2 <= r_out ** 2)
    n_pix = int(np.count_nonzero(ap_mask))
    if n_pix == 0 or np.count_nonzero(ann_mask) == 0:
        return {"ok": 0}
    ann_pixels = cut[ann_mask]
    n_sky      = int(np.count_nonzero(ann_mask))
    b_sky      = float(np.median(ann_pixels))                # 主背景估計量（中位數）
    b_sky_std  = float(np.std(ann_pixels, ddof=1)) if n_sky > 1 else 0.0  # 噪聲估計
    flux_net   = float(cut[ap_mask].sum() - b_sky * n_pix)
    max_pix    = float(np.max(cut[ap_mask])) if n_pix > 0 else np.nan
    return {
        "ok": 1,
        "flux_net": flux_net,
        "b_sky": b_sky,          # 中位數（用於 flux_net 計算）
        "b_sky_std": b_sky_std,  # 標準差（用於誤差方程式）
        "n_pix": n_pix,
        "n_sky": n_sky,
        "max_pix": max_pix,
    }


# ── 飽和判定 ──────────────────────────────────────────────────────────────────

def is_saturated(max_pix: float, threshold: "float | None") -> bool:
    return threshold is not None and np.isfinite(max_pix) and max_pix >= threshold


# ── 座標工具 ──────────────────────────────────────────────────────────────────

def radec_to_pixel(wcs: WCS, ra_deg: float, dec_deg: float) -> tuple[float, float]:
    # world_to_pixel_values 直接吃度數（RA/Dec）
    x, y = wcs.world_to_pixel_values(ra_deg, dec_deg)
    return float(x), float(y)

def in_bounds(image: np.ndarray, x: float, y: float, margin: int = 10) -> bool:
    h, w = image.shape
    return (margin <= x < (w - margin)) and (margin <= y < (h - margin))
