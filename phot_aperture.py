# -*- coding: utf-8 -*-
"""
phot_aperture.py
Aperture growth-curve estimation helpers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from phot_sources.core import (
    aperture_photometry,
    compute_annulus_radii,
    in_bounds,
    is_saturated,
    radec_to_pixel,
)


def _growth_radius_for_star(
    image: np.ndarray,
    x: float,
    y: float,
    r_min: int,
    r_max: int,
    growth_fraction: float,
    cfg_obj=None,
    annulus_r_in=None,
    annulus_r_out=None,
) -> "float | None":
    radii = np.arange(r_min, r_max + 0.5, 0.5)
    fluxes = []
    ann_in = annulus_r_in if annulus_r_in is not None else getattr(cfg_obj, "annulus_r_in", None)
    ann_out = annulus_r_out if annulus_r_out is not None else getattr(cfg_obj, "annulus_r_out", None)
    for r in radii:
        r_in, r_out = compute_annulus_radii(r, ann_in, ann_out)
        phot = aperture_photometry(image, x, y, r, r_in, r_out)
        fluxes.append(
            phot["flux_net"] if phot.get("ok") == 1 and np.isfinite(phot.get("flux_net"))
            else np.nan
        )

    fluxes = np.array(fluxes, dtype=float)
    max_flux = np.nanmax(fluxes) if np.isfinite(fluxes).any() else 0.0
    if max_flux <= 0:
        return None

    idx = np.where(fluxes >= max_flux * growth_fraction)[0]
    return float(radii[idx[0]]) if len(idx) else None


def estimate_aperture_radius(
    wcs_fits_path: Path,
    comp_df: "object | None",
    r_min: int,
    r_max: int,
    growth_fraction: float,
    cfg_obj=None,
    max_stars: int = 20,
    detect_stars_fn: "Callable | None" = None,
) -> "float | None":
    with fits.open(wcs_fits_path) as hdul:
        img = hdul[0].data.astype(np.float32)
        wcs_obj = WCS(hdul[0].header)

    stars_xy: list[tuple[float, float]] = []
    if comp_df is not None and len(comp_df) > 0:
        if "m_cat" in comp_df.columns:
            vmag_t = float(getattr(cfg_obj, "vmag_approx", np.nan))
            if np.isfinite(vmag_t):
                sub = comp_df.iloc[(comp_df["m_cat"] - vmag_t).abs().argsort()].head(max_stars)
            else:
                sub = comp_df.sort_values("m_cat").head(max_stars)
        else:
            sub = comp_df.head(max_stars)
        for _, row in sub.iterrows():
            x, y = radec_to_pixel(wcs_obj, float(row["ra_deg"]), float(row["dec_deg"]))
            if in_bounds(img, x, y, margin=int(r_max + 6)):
                stars_xy.append((x, y))

    if not stars_xy:
        if detect_stars_fn is None:
            return None
        df = detect_stars_fn(wcs_fits_path, fwhm=3.0, threshold_sigma=5.0, max_stars=200)
        if len(df) == 0:
            return None
        for _, row in df.sort_values("flux", ascending=False).head(max_stars).iterrows():
            x, y = float(row["x"]), float(row["y"])
            if in_bounds(img, x, y, margin=int(r_max + 6)):
                stars_xy.append((x, y))

    ann_in = getattr(cfg_obj, "annulus_r_in", None)
    ann_out = getattr(cfg_obj, "annulus_r_out", None)
    sat_th = getattr(cfg_obj, "saturation_threshold", None)

    radii = []
    for x, y in stars_xy:
        r_sel = _growth_radius_for_star(
            img,
            x,
            y,
            r_min,
            r_max,
            growth_fraction,
            cfg_obj=cfg_obj,
            annulus_r_in=ann_in,
            annulus_r_out=ann_out,
        )
        if r_sel is None:
            continue
        if sat_th is not None:
            r_in, r_out = compute_annulus_radii(r_sel, ann_in, ann_out)
            phot = aperture_photometry(img, x, y, r_sel, r_in, r_out)
            if is_saturated(phot.get("max_pix", np.nan), sat_th):
                continue
        radii.append(r_sel)
    return float(np.nanmedian(radii)) if radii else None


def estimate_aperture_radius_with_detection(
    wcs_fits_path: Path,
    comp_df: "object | None",
    r_min: int,
    r_max: int,
    growth_fraction: float,
    cfg_obj=None,
    max_stars: int = 20,
) -> "float | None":
    from phot_detect import detect_stars_with_radec

    return estimate_aperture_radius(
        wcs_fits_path,
        comp_df,
        r_min,
        r_max,
        growth_fraction,
        cfg_obj=cfg_obj,
        max_stars=max_stars,
        detect_stars_fn=detect_stars_with_radec,
    )


__all__ = [
    "_growth_radius_for_star",
    "estimate_aperture_radius",
    "estimate_aperture_radius_with_detection",
]
