# -*- coding: utf-8 -*-
"""
Compatibility wrapper for phot_sources.regression.
"""
from phot_sources.regression import (
    differential_mag,
    mag_error_from_flux,
    robust_linear_fit,
)

__all__ = [
    "differential_mag",
    "mag_error_from_flux",
    "robust_linear_fit",
]
