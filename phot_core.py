# -*- coding: utf-8 -*-
"""
Compatibility wrapper for phot_sources.core.
"""
from phot_sources.core import (
    aperture_photometry,
    compute_annulus_radii,
    fit_gaussian_psf,
    gaussian2d,
    in_bounds,
    is_saturated,
    m_inst_from_flux,
    max_pixel_in_box,
    radec_to_pixel,
)

__all__ = [
    "aperture_photometry",
    "compute_annulus_radii",
    "fit_gaussian_psf",
    "gaussian2d",
    "in_bounds",
    "is_saturated",
    "m_inst_from_flux",
    "max_pixel_in_box",
    "radec_to_pixel",
]
