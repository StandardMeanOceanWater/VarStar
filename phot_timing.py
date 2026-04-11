# -*- coding: utf-8 -*-
"""
phot_timing.py
Header/time utilities and sensor gain handling for photometry.
"""
from __future__ import annotations

import logging

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time

_phot_logger = logging.getLogger("photometry")

# Camera sensor database (PTC method; Canon 6D2 values from photonstophotos.net)
CAMERA_SENSOR_DB = {
    "Canon_6D2": {
        800:  {"gain": 3.78, "read_noise": 5.89},
        1600: {"gain": 1.89, "read_noise": 4.21},
        3200: {"gain": 0.94, "read_noise": 3.61},
    }
}

ISO_HEADER_KEYS = [
    "ISO", "ISOSPEED", "ISOSPEEDRATINGS", "ISOSPEEDRATING", "ISO_SPEED",
]

AIRMASS_WARN_THRESHOLD = 2.0


def _parse_iso(val) -> "int | None":
    try:
        iso = int(float(str(val).strip()))
        if iso > 0:
            return iso
    except Exception:
        return None
    return None


def _float_or_none(val) -> "float | None":
    try:
        return float(val)
    except Exception:
        return None


def get_iso_value(header, cfg) -> "int | None":
    """
    ISO priority: (1) FITS header keys, (2) cfg.iso_setting.
    """
    for key in ISO_HEADER_KEYS:
        if key in header:
            iso = _parse_iso(header.get(key))
            if iso is not None:
                return iso
    if cfg.iso_setting is not None:
        return int(cfg.iso_setting)
    return None


def apply_gain_from_header(header, cfg, force: bool = False) -> None:
    """
    Populate cfg.gain_e_per_adu and cfg.read_noise_e.
    Priority: (1) FITS GAIN / RDNOISE keywords, (2) sensor DB look-up by ISO.
    """
    if not force:
        fits_gain = header.get("GAIN")
        fits_rn = header.get("RDNOISE")
        if fits_gain is not None and cfg.gain_e_per_adu is None:
            try:
                cfg.gain_e_per_adu = float(fits_gain)
            except Exception:
                pass
        if fits_rn is not None and cfg.read_noise_e is None:
            try:
                cfg.read_noise_e = float(fits_rn)
            except Exception:
                pass
        if cfg.gain_e_per_adu is not None and cfg.read_noise_e is not None:
            return

    if cfg.camera_model is None or cfg.camera_sensor_db is None:
        return
    iso = get_iso_value(header, cfg)
    if iso is None:
        print("[WARN] ISO not found. Set cfg.iso_setting or add GAIN/RDNOISE to FITS header.")
        return
    model_db = cfg.camera_sensor_db.get(cfg.camera_model)
    if not model_db:
        print(f"[WARN] camera_model '{cfg.camera_model}' not in CAMERA_SENSOR_DB.")
        return
    entry = model_db.get(iso)
    if not entry:
        print(f"[WARN] ISO {iso} not in CAMERA_SENSOR_DB for {cfg.camera_model}.")
        return
    if cfg.gain_e_per_adu is None and entry.get("gain") is not None:
        cfg.gain_e_per_adu = float(entry["gain"])
    if cfg.read_noise_e is None and entry.get("read_noise") is not None:
        cfg.read_noise_e = float(entry["read_noise"])
    print(f"[GAIN] ISO {iso} ??gain={cfg.gain_e_per_adu:.4f} e-/DN, "
          f"read_noise={cfg.read_noise_e:.4f} e-")


def require_cfg_values(cfg) -> None:
    missing = []
    if cfg.gain_e_per_adu is None:
        missing.append("gain_e_per_adu")
    if cfg.read_noise_e is None:
        missing.append("read_noise_e")
    if missing:
        raise ValueError("Missing cfg values: " + ", ".join(missing))


def compute_airmass(
    t: Time,
    ra_deg: float,
    dec_deg: float,
    lat: float,
    lon: float,
    height: float,
) -> float:
    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height * u.m)
    sc = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    altaz = sc.transform_to(AltAz(obstime=t, location=location))
    alt_deg = float(altaz.alt.deg)
    if alt_deg <= 0.0:
        return np.inf
    z_deg = 90.0 - alt_deg
    cos_z = np.cos(np.radians(z_deg))
    airmass = 1.0 / (cos_z + 0.50572 * (96.07995 - z_deg) ** (-1.6364))
    return float(airmass)


def time_from_header(
    header,
    ra_deg: float,
    dec_deg: float,
    cfg,
) -> "tuple[float, float, float]":
    """
    Extract timing and airmass from a FITS header.
    Returns (mjd, bjd_tdb, airmass).
    """
    mid_obs = header.get("MID-OBS")
    date_obs = header.get("DATE-OBS") or header.get("DATEOBS")

    time_str = mid_obs if mid_obs else date_obs
    used_midobs = mid_obs is not None

    if not time_str:
        return np.nan, np.nan, np.nan

    try:
        t = Time(time_str, format="isot", scale="utc")
    except Exception:
        try:
            t = Time(time_str, format="iso", scale="utc")
        except Exception:
            return np.nan, np.nan, np.nan

    if not used_midobs:
        exptime = header.get("EXPTIME") or header.get("EXPOSURE")
        if exptime is not None:
            try:
                t = t + float(exptime) / 2.0 * u.second
            except Exception:
                pass
        else:
            print("[WARN] EXPTIME missing; DATE-OBS used as-is (not midpoint). "
                  "Add EXPTIME to FITS header for accurate phase computation.")

    mjd = float(t.mjd)

    lat = cfg.obs_lat_deg
    lon = cfg.obs_lon_deg
    height = cfg.obs_height_m

    if lat is None or lon is None:
        lat = _float_or_none(header.get("SITELAT") or header.get("OBS-LAT")
                             or header.get("OBSLAT"))
        lon = _float_or_none(header.get("SITELONG") or header.get("SITELON")
                             or header.get("OBS-LON") or header.get("OBSLON"))
        height = _float_or_none(
            header.get("SITEALT") or header.get("SITEELEV") or header.get("OBSALT")
            or header.get("OBSHGT") or header.get("OBSGEO-H")
        ) or cfg.obs_height_m

    bjd_tdb = np.nan
    airmass = np.nan

    if lat is None or lon is None:
        print("[WARN] Observatory location not set. "
              "Set cfg.obs_lat_deg / cfg.obs_lon_deg for BJD_TDB and airmass.")
        return mjd, bjd_tdb, airmass

    lat, lon, height = float(lat), float(lon), float(height) if height else 0.0

    try:
        location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height * u.m)
        sc = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
        ltt = t.light_travel_time(sc, "barycentric", location=location)
        bjd_tdb = float((t.tdb + ltt).jd)
    except Exception as exc:
        print(f"[WARN] BJD_TDB calculation failed: {exc}")

    try:
        airmass = compute_airmass(t, ra_deg, dec_deg, lat, lon, height)
        if np.isfinite(airmass) and airmass > AIRMASS_WARN_THRESHOLD:
            alt_deg = float(
                SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
                .transform_to(AltAz(
                    obstime=t,
                    location=EarthLocation(lat=lat * u.deg, lon=lon * u.deg,
                                           height=height * u.m),
                ))
                .alt.deg
            )
            _phot_logger.debug(
                "[WARN] High airmass X=%.2f (altitude=%.1f簞) in %s. "
                "Differential photometry accuracy may be reduced.",
                airmass, alt_deg, header.get("FILENAME", "unknown"),
            )
    except Exception as exc:
        print(f"[WARN] Airmass calculation failed: {exc}")

    return mjd, bjd_tdb, airmass
