# -*- coding: utf-8 -*-
"""
phot_detect.py
Star detection with WCS coordinates.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder

def detect_stars_with_radec(

    wcs_fits_path: Path,

    fwhm: float = 3.0,

    threshold_sigma: float = 5.0,

    max_stars: int = 200,

    border: int = 20,

):

    with fits.open(wcs_fits_path) as hdul:

        img = hdul[0].data.astype(np.float32)

        hdr = hdul[0].header



    wcs = WCS(hdr)



    mean, median, std = sigma_clipped_stats(img, sigma=3.0, maxiters=5)



    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)

    tbl = daofind(img - median)



    if tbl is None or len(tbl) == 0:

        return pd.DataFrame()



    df = tbl.to_pandas()



    h, w = img.shape

    df = df[(df["xcentroid"] > border) & (df["xcentroid"] < w - border) &

            (df["ycentroid"] > border) & (df["ycentroid"] < h - border)].copy()



    if len(df) == 0:

        return pd.DataFrame()



    if "flux" in df.columns:

        df = df.sort_values("flux", ascending=False).head(max_stars).copy()

    else:

        df = df.head(max_stars).copy()



    ra, dec = wcs.pixel_to_world_values(df["xcentroid"].to_numpy(), df["ycentroid"].to_numpy())

    df["ra_deg"] = ra.astype(float)

    df["dec_deg"] = dec.astype(float)



    df = df.rename(columns={"xcentroid": "x", "ycentroid": "y"})

    df.insert(0, "file", wcs_fits_path.name)

    if "id" not in df.columns:

        df.insert(1, "id", np.arange(1, len(df) + 1))





    return df

def batch_detect_stars(wcs_files, out_csv: Path, fwhm=3.0, threshold_sigma=5.0, max_stars=300):

    all_df = []

    for f in wcs_files:

        df = detect_stars_with_radec(

            f, fwhm=fwhm, threshold_sigma=threshold_sigma, max_stars=max_stars

        )

        if len(df):

            all_df.append(df)



    if not all_df:

        print("[detect] no stars detected in all frames.")

        return pd.DataFrame()



    out = pd.concat(all_df, ignore_index=True)

    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("[CSV saved]", out_csv, "rows=", len(out))

    return out
