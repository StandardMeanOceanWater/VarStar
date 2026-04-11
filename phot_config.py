# -*- coding: utf-8 -*-
"""
phot_config.py
Configuration dataclass and YAML -> Cfg mapping for photometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path


@dataclass
class Cfg:
    # Paths / output
    wcs_dir: Path = Path(".")
    out_dir: Path = Path(".")
    phot_out_csv: Path = Path("photometry.csv")
    phot_out_png: Path = Path("light_curve.png")
    regression_diag_dir: Path = Path("regression_diag")
    run_root: Path = Path(".")     # output/{date}/{group}/{target}/{timestamp}/

    # Target
    target_name: str = ""
    target_radec_deg: tuple = (0.0, 0.0)
    vmag_approx: float = 8.0

    # Comparison selection
    selection_radius_mode: str = "half_short_side"
    comp_mag_range: float = 4.0        # deprecated
    comp_mag_bright: float = 4.0
    comp_mag_faint: float = 2.0
    comp_mag_min: float = 4.0
    comp_mag_max: float = 12.0
    comp_max: int = 20
    comp_fwhm_min: float = 2.0
    comp_fwhm_max: float = 8.0
    comp_min_sep_arcsec: float = 30.0
    apass_radius_deg: float = 1.0
    apass_match_arcsec: float = 2.0
    apass_maxrec: int = 5000
    aavso_fov_arcmin: float = 100.0
    aavso_maglimit: float = 15.0
    aavso_min_stars: int = 5
    aavso_star_name: str | None = None
    aavso_seq_csv: Path | None = None
    aavso_use_api: bool = True
    catalog_priority: list = field(default_factory=lambda: ["AAVSO", "APASS"])
    save_regression_diagnostic: bool = True

    # Aperture photometry
    aperture_auto: bool = True
    aperture_radius: float = 8.0
    aperture_min_radius: int = 2
    aperture_max_radius: int = 12
    aperture_growth_fraction: float = 0.97
    annulus_r_in: float | None = None
    annulus_r_out: float | None = None
    saturation_threshold: float = 65536.0
    saturation_box: int = 5
    allow_saturated_target: bool = True
    allow_saturated_check: bool = False

    # Sensor / noise
    gain_e_per_adu: float | None = None
    read_noise_e: float | None = None
    camera_model: str | None = None
    camera_sensor_db: dict | None = None
    iso_setting: int | None = None

    # Plate scale
    plate_scale_arcsec: float = 1.485

    # Check star
    check_star_radec_deg: tuple | None = None
    check_star_max_sigma: float = 0.02

    # Photometry band mapping
    phot_band: str = "V"

    # Timing / observatory
    apply_bjd: bool = True
    obs_lat_deg: float | None = None
    obs_lon_deg: float | None = None
    obs_height_m: float = 0.0

    # Airmass
    alt_min_deg: float = 30.0
    alt_min_airmass: float = 2.366

    # Extinction
    extinction_k: float = 0.0

    # Robust regression
    robust_regression_sigma: float = 3.0
    robust_regression_max_iter: int = 5
    robust_regression_min_points: int = 3

    # Ensemble normalization
    ensemble_normalize: bool = False
    ensemble_min_comp: int = 3
    ensemble_max_iter: int = 10
    ensemble_convergence_tol: float = 1e-4

    # Quality filters
    sharpness_min: float = 0.3
    reg_r2_min: float = 0.0
    peak_ratio_min: float = 0.0
    peak_ratio_k: float = 0.0
    reg_intercept_sigma: float = 0.0
    sky_sigma: float = 0.0

    # Plate-solve / detect paths
    fits_dir: Path = Path(".")
    astap_exe: Path = Path("astap_cli")
    astap_ra_hours: float = 0.0
    astap_dec_deg: float = 0.0
    astap_search_radius_deg: float = 30.0
    astap_downsample: int = 2
    astap_db_path: Path = Path("C:/Program Files/astap/d80")
    astap_speed: str = "slow"
    astap_fov_override_deg: float = 1.61
    max_fwhm_px: float = 8.0
    wcs_out_dir: Path = Path(".")
    stars_csv: Path = Path("stars.csv")


def _resolve_extinction_k(yaml_dict: dict, channel: str, phot_cfg: dict) -> float:
    """Resolve per-channel extinction coefficient with photometry fallback."""
    ext_coeffs = yaml_dict.get("extinction", {}).get("coefficients", {})
    ch_key = channel[0] if channel in ("G1", "G2") else channel
    if ch_key in ext_coeffs:
        return float(ext_coeffs[ch_key])
    return float(phot_cfg.get("extinction_k", 0.0))


def cfg_from_yaml(
    yaml_dict: dict,
    target: str,
    session_date: str,
    channel: str = "B",
    split_subdir: str = "splits",
    out_tag: "str | None" = None,
    run_ts: "str | None" = None,
) -> Cfg:
    """
    Build a Cfg instance from observation_config.yaml structure.
    """
    data_root = yaml_dict["_data_root"]
    project_root = yaml_dict["_project_root"]

    tgt = yaml_dict.get("targets", {}).get(target, {})

    group = tgt.get("group", target)
    _date_fmt = f"{session_date[:4]}-{session_date[4:6]}-{session_date[6:8]}"
    field_root = data_root / _date_fmt / group

    if "ra_deg" in tgt:
        ra_deg = float(tgt["ra_deg"])
    elif "ra_hint_h" in tgt:
        ra_deg = float(tgt["ra_hint_h"]) * 15.0
    else:
        raise ValueError(
            f"{target}：yaml targets 缺少 ra_deg 或 ra_hint_h，無法建立座標\n"
            f"  請檢查 observation_config.yaml 的 targets.{target} 欄位"
        )

    if "dec_deg" in tgt:
        dec_deg = float(tgt["dec_deg"])
    elif "dec_hint_deg" in tgt:
        dec_deg = float(tgt["dec_hint_deg"])
    else:
        raise ValueError(
            f"{target}：yaml targets 缺少 dec_deg 或 dec_hint_deg，無法建立座標"
        )
    vmag = float(tgt.get("vmag_approx", 8.0))
    display = tgt.get("display_name", target)

    _cmp = yaml_dict.get("comparison_stars", {})
    if "comp_mag_bright" in _cmp or "comp_mag_faint" in _cmp:
        _bright = float(_cmp.get("comp_mag_bright", 4.0))
        _faint = float(_cmp.get("comp_mag_faint", 2.0))
        comp_mag_min = vmag - _bright
        comp_mag_max = vmag + _faint
    else:
        print(
            "[WARN] comparison_stars.comp_mag_bright / comp_mag_faint 未設定，"
            "改用 mag_range_delta（deprecated），建議回填 yaml。"
        )
        _range = float(_cmp.get("mag_range_delta", 4.0))
        comp_mag_min = vmag - _range
        comp_mag_max = vmag + _range
    if "comp_mag_min" in _cmp:
        comp_mag_min = max(comp_mag_min, float(_cmp["comp_mag_min"]))
    if "comp_mag_max" in _cmp:
        comp_mag_max = min(comp_mag_max, float(_cmp["comp_mag_max"]))

    def _session_has_target(s: dict, tgt_name: str) -> bool:
        raw = s.get("targets", s.get("target"))
        if raw is None:
            return False
        if isinstance(raw, list):
            return tgt_name in [str(x) for x in raw]
        return str(raw) == tgt_name

    session = next(
        (s for s in yaml_dict.get("obs_sessions", [])
         if _session_has_target(s, target)
         and str(s.get("date")) == str(session_date)),
        {}
    )
    telescope_id = session.get("telescope", "")
    camera_id = session.get("camera", "")
    iso = int(session.get("iso", 0))

    tel_cfg = yaml_dict.get("telescopes", {}).get(telescope_id, {})
    _cameras = yaml_dict.get("cameras", {})
    cam_cfg = _cameras.get(camera_id, {})
    if not cam_cfg:
        for _ckey, _cval in _cameras.items():
            if isinstance(_cval, dict) and (_cval.get("name", "") == camera_id
                                            or camera_id in _ckey or _ckey in camera_id):
                cam_cfg = _cval
                camera_id = _ckey
                break
    if not cam_cfg:
        print(f"  [WARN] camera '{camera_id}' not found in cameras config")
    obs_cfg = yaml_dict.get("observatory", {})
    ap_cfg = yaml_dict.get("photometry", {})
    cmp_cfg = yaml_dict.get("comparison_stars", {})
    phot_cfg = yaml_dict.get("photometry", {})

    focal_mm = float(tel_cfg.get("focal_length_mm", 800.0))
    pixel_um = float(cam_cfg.get("pixel_size_um", 5.76))
    eff_pixel = pixel_um * 2.0
    plate_scale = 206.265 * eff_pixel / focal_mm

    sensor_db = cam_cfg.get("sensor_db", {})
    iso_entry = sensor_db.get(iso, {})
    if isinstance(iso_entry, (list, tuple)):
        gain_e = float(iso_entry[0]) if len(iso_entry) > 0 else None
        rn_e = float(iso_entry[1]) if len(iso_entry) > 1 else None
    else:
        gain_e = iso_entry.get("gain")
        rn_e = iso_entry.get("read_noise")
    _sat_raw = cam_cfg.get("saturation_adu", 11469.0)
    sat_adu = None if _sat_raw is None else float(_sat_raw)

    channel = str(channel).upper()
    wcs_dir = field_root / split_subdir / channel

    _run_ts = run_ts or datetime.now().strftime("%Y%m%d_%H%M")
    run_root = project_root / "output" / _date_fmt / group / target / _run_ts
    out_dir = run_root / "1_photometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = run_root / "2_regression_diag"
    diag_dir.mkdir(parents=True, exist_ok=True)
    lc_dir = run_root / "3_light_curve"
    lc_dir.mkdir(parents=True, exist_ok=True)
    pa_dir = run_root / "4_period_analysis"
    pa_dir.mkdir(parents=True, exist_ok=True)

    _band_map = {"R": "r", "G1": "V", "G2": "V", "B": "B"}
    phot_band = _band_map.get(channel.upper(), "V")

    _auto = {}
    _cfg_fields = {f.name: f for f in fields(Cfg)}
    for _key, _val in phot_cfg.items():
        if _key in _cfg_fields and _val is not None:
            _ftype = _cfg_fields[_key].type
            try:
                if _ftype in (float, "float"):
                    _auto[_key] = float(_val)
                elif _ftype in (int, "int"):
                    _auto[_key] = int(_val)
                elif _ftype in (bool, "bool"):
                    _auto[_key] = bool(_val)
                elif _ftype in (str, "str"):
                    _auto[_key] = str(_val)
                else:
                    _auto[_key] = _val
            except (ValueError, TypeError):
                pass

    _manual = dict(
        run_root=run_root,
        wcs_dir=wcs_dir,
        out_dir=out_dir,
        phot_out_csv=out_dir / f"photometry_{channel}_{session_date}.csv",
        phot_out_png=lc_dir / f"light_curve_{channel}_{session_date}.png",
        regression_diag_dir=diag_dir,

        target_name=display,
        target_radec_deg=(ra_deg, dec_deg),
        vmag_approx=vmag,
        aavso_star_name=display,

        comp_mag_bright=float(_cmp.get("comp_mag_bright", 4.0)),
        comp_mag_faint=float(_cmp.get("comp_mag_faint", 2.0)),
        comp_mag_min=comp_mag_min,
        comp_mag_max=comp_mag_max,
        comp_max=int(cmp_cfg.get("max_stars", 15)),
        comp_fwhm_min=float(cmp_cfg.get("comp_fwhm_min", 2.0)),
        comp_fwhm_max=float(cmp_cfg.get("comp_fwhm_max", 8.0)),
        comp_min_sep_arcsec=float(cmp_cfg.get("min_separation_arcsec", 30.0)),
        apass_radius_deg=float(cmp_cfg.get("apass_radius_deg", 1.0)),
        apass_match_arcsec=float(cmp_cfg.get("apass_match_arcsec", 2.0)),
        aavso_fov_arcmin=float(cmp_cfg.get("aavso_fov_arcmin", 100.0)),
        aavso_maglimit=float(cmp_cfg.get("aavso_maglimit", 15.0)),
        aavso_min_stars=int(cmp_cfg.get("aavso_min_stars", 5)),
        catalog_priority=list(cmp_cfg.get("catalog_priority", ["AAVSO", "APASS"])),
        save_regression_diagnostic=bool(cmp_cfg.get("save_regression_diagnostic", True)),

        aperture_auto=True,
        aperture_radius=float(ap_cfg.get("aperture_radius", 8.0)),
        aperture_min_radius=int(min(ap_cfg.get("aperture_growth_radii", [2]), default=2)),
        aperture_max_radius=int(max(ap_cfg.get("aperture_growth_radii", [12]), default=12)),
        aperture_growth_fraction=float(ap_cfg.get("aperture_growth_fraction", 0.95)),
        annulus_r_in=ap_cfg.get("sky_annulus_inner_px"),
        annulus_r_out=ap_cfg.get("sky_annulus_outer_px"),
        saturation_threshold=sat_adu,

        gain_e_per_adu=float(gain_e) if gain_e is not None else None,
        read_noise_e=float(rn_e) if rn_e is not None else None,
        camera_model=cam_cfg.get("camera_model"),
        camera_sensor_db={cam_cfg.get("camera_model", ""): sensor_db},
        iso_setting=iso,
        phot_band=phot_band,

        plate_scale_arcsec=plate_scale,

        obs_lat_deg=(
            float(session["obs_lat_deg"]) if "obs_lat_deg" in session
            else float(obs_cfg["latitude_deg"]) if "latitude_deg" in obs_cfg
            else None
        ),
        obs_lon_deg=(
            float(session["obs_lon_deg"]) if "obs_lon_deg" in session
            else float(obs_cfg["longitude_deg"]) if "longitude_deg" in obs_cfg
            else None
        ),
        obs_height_m=(
            float(session["obs_height_m"]) if "obs_height_m" in session
            else float(obs_cfg.get("elevation_m", 0.0))
        ),

        extinction_k=_resolve_extinction_k(yaml_dict, channel, phot_cfg),

        fits_dir=wcs_dir,
        astap_exe=Path(yaml_dict.get("astrometry", {}).get("astap", {})
                       .get("executable", "astap_cli")),
        astap_ra_hours=ra_deg / 15.0,
        astap_dec_deg=dec_deg,
        astap_db_path=Path(yaml_dict.get("astrometry", {}).get("astap", {})
                           .get("db_path", "C:/Program Files/astap/d80")),
        wcs_out_dir=wcs_dir,
        stars_csv=out_dir / "stars_detected.csv",
    )
    _merged = {**_auto, **_manual}
    return Cfg(**_merged)
