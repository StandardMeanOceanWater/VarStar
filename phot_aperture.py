# -*- coding: utf-8 -*-
"""
phot_aperture.py
Aperture growth-curve helpers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils.detection import IRAFStarFinder

from phot_core import (
    compute_annulus_radii,
    aperture_photometry,
    is_saturated,
    radec_to_pixel,
    in_bounds,
)

def _growth_radius_for_star(

    image: np.ndarray,

    x: float,

    y: float,

    r_min: int,

    r_max: int,

    growth_fraction: float,

    annulus_r_in=None,

    annulus_r_out=None,

) -> "float | None":

    radii  = np.arange(r_min, r_max + 0.5, 0.5)

    fluxes = []

    # 顯式傳入 annulus 參數，避免依賴全域 cfg

    _ann_in  = annulus_r_in if annulus_r_in is not None else getattr(cfg, 'annulus_r_in', None)

    _ann_out = annulus_r_out if annulus_r_out is not None else getattr(cfg, 'annulus_r_out', None)

    for r in radii:

        r_in, r_out = compute_annulus_radii(r, _ann_in, _ann_out)

        phot = aperture_photometry(image, x, y, r, r_in, r_out)

        fluxes.append(

            phot["flux_net"] if phot.get("ok") == 1 and np.isfinite(phot.get("flux_net"))

            else np.nan

        )

    fluxes   = np.array(fluxes, dtype=float)

    max_flux = np.nanmax(fluxes) if np.isfinite(fluxes).any() else 0.0

    if max_flux <= 0:

        return None

    idx = np.where(fluxes >= max_flux * growth_fraction)[0]

    return float(radii[idx[0]]) if len(idx) else None

def estimate_aperture_radius(

    wcs_fits_path: Path,

    comp_df: "pd.DataFrame | None",

    r_min: int,

    r_max: int,

    growth_fraction: float,

    max_stars: int = 20,

) -> "float | None":

    with fits.open(wcs_fits_path) as hdul:

        img = hdul[0].data.astype(np.float32)

        wcs_obj = WCS(hdul[0].header)



    stars_xy: list[tuple[float, float]] = []

    if comp_df is not None and len(comp_df) > 0:

        # 按星等接近目標星排序（而非取最亮），避免亮星 PSF 過集中導致孔徑偏小

        if "m_cat" in comp_df.columns:

            _vmag_t = float(getattr(cfg, "vmag_approx", np.nan)) if 'cfg' in globals() else np.nan

            if np.isfinite(_vmag_t):

                sub = comp_df.iloc[(comp_df["m_cat"] - _vmag_t).abs().argsort()].head(max_stars)

            else:

                sub = comp_df.sort_values("m_cat").head(max_stars)

        else:

            sub = comp_df.head(max_stars)

        for _, row in sub.iterrows():

            x, y = radec_to_pixel(wcs_obj, float(row["ra_deg"]), float(row["dec_deg"]))

            if in_bounds(img, x, y, margin=int(r_max + 6)):

                stars_xy.append((x, y))



    if not stars_xy:

        df = detect_stars_with_radec(wcs_fits_path, fwhm=3.0,

                                     threshold_sigma=5.0, max_stars=200)

        if len(df) == 0:

            return None

        for _, row in df.sort_values("flux", ascending=False).head(max_stars).iterrows():

            x, y = float(row["x"]), float(row["y"])

            if in_bounds(img, x, y, margin=int(r_max + 6)):

                stars_xy.append((x, y))



    # 從全域 cfg 取 annulus 和 saturation 參數（暫保 fallback，逐步消除全域依賴）

    _ann_in  = getattr(cfg, 'annulus_r_in', None) if 'cfg' in dir() or 'cfg' in globals() else None

    _ann_out = getattr(cfg, 'annulus_r_out', None) if 'cfg' in dir() or 'cfg' in globals() else None

    _sat_th  = getattr(cfg, 'saturation_threshold', None) if 'cfg' in dir() or 'cfg' in globals() else None



    radii = []

    for x, y in stars_xy:

        r_sel = _growth_radius_for_star(img, x, y, r_min, r_max, growth_fraction,

                                         annulus_r_in=_ann_in, annulus_r_out=_ann_out)

        if r_sel is None:

            continue

        if _sat_th is not None:

            r_in, r_out = compute_annulus_radii(r_sel, _ann_in, _ann_out)

            phot = aperture_photometry(img, x, y, r_sel, r_in, r_out)

            if is_saturated(phot.get("max_pix", np.nan), _sat_th):

                continue

        radii.append(r_sel)

    return float(np.nanmedian(radii)) if radii else None
