# -*- coding: utf-8 -*-
"""
phot_wcs.py
Plate-solving utilities (ASTAP) and WCS helpers.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

from astropy.io import fits

# cfg is injected from photometry wrapper
cfg = None

def _has_wcs(header) -> bool:

    need = ["CTYPE1","CTYPE2","CRVAL1","CRVAL2","CRPIX1","CRPIX2"]

    return all(k in header for k in need)

def _estimate_fov_deg(path: Path) -> float | None:

    try:

        with fits.open(path) as hdul:

            h = hdul[0].header

        nx = h.get("NAXIS1")

        pix_um = h.get("XPIXSZ") or h.get("PIXSZ")

        focal_mm = h.get("FOCALLEN")

        if not (nx and pix_um and focal_mm):

            return None

        scale_arcsec = 206.265 * float(pix_um) / float(focal_mm)

        return scale_arcsec * float(nx) / 3600.0

    except Exception:

        return None

def run_astap_plate_solve(in_fits: Path, out_wcs_fits: Path, timeout_sec: int = 180):

    out_wcs_fits.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(in_fits, out_wcs_fits)



    spd = 90.0 + float(cfg.astap_dec_deg)

    fov_deg = cfg.astap_fov_override_deg if cfg.astap_fov_override_deg > 0 else _estimate_fov_deg(out_wcs_fits)



    cmd = [

        str(cfg.astap_exe),

        "-f", str(out_wcs_fits),

        "-r", str(cfg.astap_search_radius_deg),

        "-z", str(cfg.astap_downsample),

        "-d", str(cfg.astap_db_path),

        "-ra", f"{cfg.astap_ra_hours}",

        "-spd", f"{spd}",

    ]



    if fov_deg:

        cmd += ["-fov", f"{fov_deg:.3f}"]

    if cfg.astap_speed:

        cmd += ["-speed", cfg.astap_speed]



    cmd += ["-update", "-log"]



    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)

    if proc.returncode != 0:

        raise RuntimeError(

    f"[ASTAP FAIL] {in_fits.name}\n"

    f"CMD: {' '.join(cmd)}\n"

    f"STDOUT:\n{proc.stdout}\n"

    f"STDERR:\n{proc.stderr}"

)





    with fits.open(out_wcs_fits) as hdul:

        if not _has_wcs(hdul[0].header):

            raise RuntimeError(f"[ASTAP] no WCS keywords in {out_wcs_fits.name}")

def batch_plate_solve_all(timeout_sec: int = 180, force: bool = False):

    fits_files = sorted(list(cfg.fits_dir.glob("*.fits")) + list(cfg.fits_dir.glob("*.fit")))

    ok, fail = 0, 0

    out_paths = []



    for f in fits_files:

        out = cfg.wcs_out_dir / (f.stem + "_wcs.fits")

        if out.exists() and not force:

            out_paths.append(out)

            continue





        try:

            run_astap_plate_solve(f, out, timeout_sec=timeout_sec)

            ok += 1

            out_paths.append(out)

            print("[OK]", f.name, "->", out.name)

        except Exception as e:

            fail += 1

            print("[FAIL]", f.name, ":", e)



    print(f"ASTAP finished: ok={ok}, fail={fail}, total_wcs={len(out_paths)}")

    return out_paths
