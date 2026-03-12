# -*- coding: utf-8 -*-
"""
photometry.py — 變星差分測光管線 步驟 4
使用方式：
    python photometry.py
    python photometry.py --target V1162Ori --date 20251220 --channels R G1 B
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install -U numpy pandas matplotlib astropy scipy

# Commented out IPython magic to ensure Python compatibility.
# %pip install -U photutils

# Commented out IPython magic to ensure Python compatibility.
# %pip install -U pytest ruff mypy

"""## Plate solve each frame (ASTAP → WCS in FITS)"""

# -*- coding: utf-8 -*-
"""
Cell 4 — 設定載入
從 observation_config.yaml 讀取所有參數，填入 Cfg dataclass。
不再有硬編碼路徑或 Windows 路徑，Colab 和本地執行自動切換。
"""
import subprocess
import shutil
import sys
import os
import logging
import warnings
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
import matplotlib.pyplot as plt
from photutils.detection import DAOStarFinder

# ── Module-level logger（__main__ 會加 handler；函數直接用此 logger）──────────
_phot_logger = logging.getLogger("photometry")

# ── 自動偵測環境，找到 observation_config.yaml ────────────────────────────────
def _find_config() -> Path:
    """
    搜尋順序：
    1. 環境變數 VARSTAR_CONFIG
    2. 目前工作目錄
    3. 本 notebook 所在目錄的上層（pipeline/ 的上層 = project_root）
    4. Colab 的 Drive 預設路徑
    """
    env = os.environ.get("VARSTAR_CONFIG")
    if env:
        return Path(env)

    candidates = [
        Path.cwd() / "observation_config.yaml",
        Path(__file__).parent.parent / "observation_config.yaml"
        if "__file__" in dir() else Path("/nonexistent"),
        Path("/content/drive/Shareddrives/VarStar/pipeline/observation_config.yaml"),
        Path("D:/VarStar/pipeline/observation_config.yaml"),
    ]
    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            continue

    raise FileNotFoundError(
        "找不到 observation_config.yaml。\n"
        "請設定環境變數 VARSTAR_CONFIG 或把 yaml 放在工作目錄下。\n"
        "例如：os.environ['VARSTAR_CONFIG'] = 'D:/VarStar/pipeline/observation_config.yaml'"
    )


def _detect_project_root(cfg_dict: dict, config_path: Path) -> Path:
    try:
        import google.colab  # noqa: F401
        root = cfg_dict["paths"]["colab"]["project_root"]
    except (ImportError, KeyError):
        root = cfg_dict["paths"]["local"]["project_root"]
    p = Path(root)
    if not p.is_absolute():
        p = (config_path.parent / p).resolve()
    return p


def load_pipeline_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = _find_config()
    with config_path.open("r", encoding="utf-8") as fh:
        cfg_dict = yaml.safe_load(fh)
    cfg_dict["_project_root"] = _detect_project_root(cfg_dict, config_path)
    cfg_dict["_data_root"] = cfg_dict["_project_root"] / "data"
    cfg_dict["_config_path"] = config_path
    return cfg_dict


@dataclass
class Cfg:
    # ── 路徑 ──────────────────────────────────────────────────────────────────
    wcs_dir: Path = Path(".")
    out_dir: Path = Path(".")
    phot_out_csv: Path = Path("photometry.csv")
    phot_out_png: Path = Path("light_curve.png")
    zeropoint_diag_dir: Path = Path("zeropoint_diag")

    # ── 目標 ──────────────────────────────────────────────────────────────────
    target_name: str = ""
    target_radec_deg: tuple = (0.0, 0.0)
    vmag_approx: float = 8.0

    # ── 比較星選取 ─────────────────────────────────────────────────────────────
    # 選取範圍：以目標星為圓心，影像短邊像素數一半為半徑
    selection_radius_mode: str = "half_short_side"
    comp_mag_range: float = 4.0       # 目標星 ± comp_mag_range 等
    comp_mag_min: float = 4.0         # 動態計算後填入
    comp_mag_max: float = 12.0        # 動態計算後填入
    comp_max: int = 15
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
    save_zeropoint_diagnostic: bool = True

    # ── 孔徑測光 ──────────────────────────────────────────────────────────────
    aperture_auto: bool = True
    aperture_radius: float = 8.0
    aperture_min_radius: int = 2
    aperture_max_radius: int = 12
    aperture_growth_fraction: float = 0.95
    annulus_r_in: float | None = None
    annulus_r_out: float | None = None
    saturation_threshold: float = 11469.0
    saturation_box: int = 5
    allow_saturated_target: bool = True
    allow_saturated_check: bool = False

    # ── 誤差模型 ──────────────────────────────────────────────────────────────
    gain_e_per_adu: float | None = None
    read_noise_e: float | None = None
    camera_model: str | None = None
    camera_sensor_db: dict | None = None
    iso_setting: int | None = None

    # ── 權重（ε 由 plate_scale 自動計算，不手填）─────────────────────────────
    plate_scale_arcsec: float = 1.485   # Vixen R200SS + Canon 6D2  pixel pitch 5.76 μm
    # comp_weight_epsilon = plate_scale_arcsec / 2（程式自動計算）

    # ── 檢查星 ────────────────────────────────────────────────────────────────
    check_star_radec_deg: tuple | None = None
    check_star_max_sigma: float = 0.02

    # ── 測光波段（APASS 星色對應）────────────────────────────────────────────
    phot_band: str = "V"   # R→r, G1/G2→V, B→B；決定 APASS 哪個欄位做零點回歸

    # ── 時間系統 ──────────────────────────────────────────────────────────────
    apply_bjd: bool = True            # True = BJD_TDB（AAVSO 標準）
    obs_lat_deg: float | None = None
    obs_lon_deg: float | None = None
    obs_height_m: float = 0.0

    # ── 高度角截斷 ────────────────────────────────────────────────────────────
    # 低高度角時大氣消光急增，差分測光精度下降。
    # alt_min_deg=45° 對應 airmass ≈ 1.41（Young 1994）。
    # airmass=NaN（location 未設定）時不截斷。
    alt_min_deg: float = 30.0
    alt_min_airmass: float = 1.994   # Young (1994) at alt=30°

    # ── 穩健回歸 ──────────────────────────────────────────────────────────────
    robust_regression_sigma: float = 3.0
    robust_regression_max_iter: int = 5
    robust_regression_min_points: int = 3

    # ── Ensemble 正規化（Broeg 2005, §3.9）───────────────────────────────────
    ensemble_normalize: bool = False       # yaml: ensemble_normalize: true
    ensemble_min_comp: int = 3             # 啟用 ensemble 所需最少比較星數
    ensemble_max_iter: int = 10            # Broeg 迭代上限
    ensemble_convergence_tol: float = 1e-4  # 收斂閾值（mag）

    # ── 幀品質篩選 ────────────────────────────────────────────────────────────
    sharpness_min: float = 0.3        # Sharpness index 下限（0.0 = 停用）
    zp_r2_min: float = 0.0            # ZP 回歸 R² 下限（0.0 = 停用）

    # ── 舊版相容（ASTAP 板塊解算，保留供 Cell 5 使用）────────────────────────
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


def cfg_from_yaml(
    yaml_dict: dict,
    target: str,
    session_date: str,
    channel: str = "B",
    split_subdir: str = "split",
) -> Cfg:
    """
    從 yaml 字典建立 Cfg 物件。

    Parameters
    ----------
    yaml_dict    : load_pipeline_config() 回傳的字典。
    target       : 目標名稱，對應 yaml targets 區塊的鍵值。
    session_date : 觀測日期字串，例如 "20241220"。
    """
    data_root = yaml_dict["_data_root"]
    target_root = data_root / "targets" / target

    # ── 取得目標設定 ───────────────────────────────────────────────────────────
    tgt = yaml_dict.get("targets", {}).get(target, {})

    # ra_deg 優先；fallback 到 ra_hint_h（小時角 × 15 轉為度）
    if "ra_deg" in tgt:
        ra_deg = float(tgt["ra_deg"])
    elif "ra_hint_h" in tgt:
        ra_deg = float(tgt["ra_hint_h"]) * 15.0
    else:
        ra_deg = 0.0
        print(f"[WARN] {target}：yaml 缺少 ra_deg / ra_hint_h，座標設為 0.0")

    # dec_deg 優先；fallback 到 dec_hint_deg
    if "dec_deg" in tgt:
        dec_deg = float(tgt["dec_deg"])
    elif "dec_hint_deg" in tgt:
        dec_deg = float(tgt["dec_hint_deg"])
    else:
        dec_deg = 0.0
        print(f"[WARN] {target}：yaml 缺少 dec_deg / dec_hint_deg，座標設為 0.0")
    vmag    = float(tgt.get("vmag_approx", 8.0))
    display = tgt.get("display_name", target)

    # ── 比較星星等範圍（動態）──────────────────────────────────────────────────
    comp_range = float(
        yaml_dict.get("comparison_stars", {}).get("mag_range_delta", 4.0)
    )
    comp_mag_min = vmag - comp_range
    comp_mag_max = vmag + comp_range
    # yaml 可直接覆蓋 comp_mag_min / comp_mag_max
    _cmp = yaml_dict.get("comparison_stars", {})
    if "comp_mag_min" in _cmp:
        comp_mag_min = float(_cmp["comp_mag_min"])
    if "comp_mag_max" in _cmp:
        comp_mag_max = float(_cmp["comp_mag_max"])

    # ── 儀器參數 ───────────────────────────────────────────────────────────────
    # 找到此 target + date 對應的 session
    # 支援 targets 列表（複數）與舊格式 target 單數
    def _session_has_target(s: dict, tgt: str) -> bool:
        raw = s.get("targets", s.get("target"))
        if raw is None:
            return False
        if isinstance(raw, list):
            return tgt in [str(x) for x in raw]
        return str(raw) == tgt

    session = next(
        (s for s in yaml_dict.get("obs_sessions", [])
         if _session_has_target(s, target)
         and str(s.get("date")) == str(session_date)),
        {}
    )
    telescope_id = session.get("telescope", "")
    camera_id    = session.get("camera", "")
    iso          = int(session.get("iso", 0))

    tel_cfg = yaml_dict.get("telescopes", {}).get(telescope_id, {})
    cam_cfg = yaml_dict.get("cameras", {}).get(camera_id, {})
    obs_cfg = yaml_dict.get("observatory", {})
    ap_cfg  = yaml_dict.get("photometry", {})   # aperture 參數位於 photometry: 區塊
    cmp_cfg = yaml_dict.get("comparison_stars", {})
    phot_cfg = yaml_dict.get("photometry", {})

    focal_mm   = float(tel_cfg.get("focal_length_mm", 800.0))
    pixel_um   = float(cam_cfg.get("pixel_size_um", 5.76))  # Canon 6D2 pixel pitch
    # 拆色後像素大小 = 原始 × 2
    eff_pixel  = pixel_um * 2.0
    plate_scale = 206.265 * eff_pixel / focal_mm   # arcsec/px

    sensor_db  = cam_cfg.get("sensor_db", {})
    iso_entry  = sensor_db.get(iso, {})
    # sensor_db 支援兩種格式：
    #   dict 格式：{gain: 0.232, read_noise: 1.69}
    #   list 格式：[0.232, 1.69]  → index 0 = gain, index 1 = read_noise
    if isinstance(iso_entry, (list, tuple)):
        gain_e = float(iso_entry[0]) if len(iso_entry) > 0 else None
        rn_e   = float(iso_entry[1]) if len(iso_entry) > 1 else None
    else:
        gain_e = iso_entry.get("gain")
        rn_e   = iso_entry.get("read_noise")
    sat_adu    = float(cam_cfg.get("saturation_adu", 11469.0))

    # ── 路徑 ───────────────────────────────────────────────────────────────────
    # 測光通道：split/B/（拆色後 B 通道；FITS 內含 WCS，由 DeBayer_RGGB.py 傳遞）
    channel = str(channel).upper()   # 使用傳入參數，不從 yaml 讀
    wcs_dir  = target_root / split_subdir / channel
    out_dir  = target_root / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = out_dir / "zeropoint_diag"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # 通道對應 APASS 波段
    _band_map = {"R": "r", "G1": "V", "G2": "V", "B": "B"}
    phot_band = _band_map.get(channel.upper(), "V")

    return Cfg(
        # 路徑
        wcs_dir=wcs_dir,
        out_dir=out_dir,
        phot_out_csv=out_dir / f"photometry_{channel}_{session_date}.csv",
        phot_out_png=out_dir / f"light_curve_{channel}_{session_date}.png",
        zeropoint_diag_dir=diag_dir,

        # 目標
        target_name=display,
        target_radec_deg=(ra_deg, dec_deg),
        vmag_approx=vmag,
        aavso_star_name=display,

        # 比較星
        comp_mag_range=comp_range,
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
        save_zeropoint_diagnostic=bool(cmp_cfg.get("save_zeropoint_diagnostic", True)),

        # 孔徑測光
        # yaml photometry: 區塊用 aperture_growth_radii list；
        # aperture_auto / aperture_radius / aperture_growth_fraction 為程式內部預設。
        # sky_annulus_inner_px / sky_annulus_outer_px 對應 annulus_r_in / annulus_r_out。
        aperture_auto=True,                   # yaml 無此欄位，固定 True（生長曲線法）
        aperture_radius=float(ap_cfg.get("aperture_radius", 8.0)),
        aperture_min_radius=int(min(ap_cfg.get("aperture_growth_radii", [2]), default=2)),
        aperture_max_radius=int(max(ap_cfg.get("aperture_growth_radii", [12]), default=12)),
        aperture_growth_fraction=float(ap_cfg.get("aperture_growth_fraction", 0.95)),
        annulus_r_in=ap_cfg.get("sky_annulus_inner_px"),
        annulus_r_out=ap_cfg.get("sky_annulus_outer_px"),
        saturation_threshold=sat_adu,

        # 誤差模型
        gain_e_per_adu=float(gain_e) if gain_e is not None else None,
        read_noise_e=float(rn_e) if rn_e is not None else None,
        camera_model=cam_cfg.get("camera_model"),
        camera_sensor_db={cam_cfg.get("camera_model", ""): sensor_db},
        iso_setting=iso,
        phot_band=phot_band,

        # 板塊比例尺（用於權重 ε）
        plate_scale_arcsec=plate_scale,

        # 觀測站（BJD_TDB 計算用）
        # 優先從 obs_sessions[i] 讀取，fallback 到頂層 observatory 區塊
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

        # 穩健回歸
        robust_regression_sigma=float(phot_cfg.get("robust_regression_sigma", 3.0)),
        robust_regression_max_iter=int(phot_cfg.get("robust_regression_max_iter", 5)),
        robust_regression_min_points=int(phot_cfg.get("robust_regression_min_points", 3)),

        # Ensemble 正規化（Broeg 2005）
        ensemble_normalize=bool(phot_cfg.get("ensemble_normalize", False)),
        ensemble_min_comp=int(phot_cfg.get("ensemble_min_comp", 3)),
        ensemble_max_iter=int(phot_cfg.get("ensemble_max_iter", 10)),
        ensemble_convergence_tol=float(
            phot_cfg.get("ensemble_convergence_tol", 1e-4)
        ),

        # 幀品質篩選
        sharpness_min=float(phot_cfg.get("sharpness_min", 0.3)),
        zp_r2_min=float(phot_cfg.get("zp_r2_min", 0.0)),

        # 舊版相容
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


# ── 使用方法 ──────────────────────────────────────────────────────────────────
# python photometry.py
# python photometry.py --target V1162Ori --date 20251220 --channels R G1 B
# 第一次執行時設定：
#   os.environ['VARSTAR_CONFIG'] = 'D:/VarStar/pipeline/observation_config.yaml'


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


    print(f"ASTAP finished: ok={ok}, fail={fail}, total_wcs={len(out_paths)}")
    return out_paths

# batch_plate_solve_all 已停用（步驟 3 已完成 plate solve）

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

# batch_detect_stars 診斷碼已移除（僅 notebook 互動用）

"""## RA/Dec → Pixel (per frame)"""

from astropy.coordinates import SkyCoord
import astropy.units as u

def radec_to_pixel(wcs: WCS, ra_deg: float, dec_deg: float) -> tuple[float, float]:
    # world_to_pixel_values 直接吃度數（RA/Dec）
    x, y = wcs.world_to_pixel_values(ra_deg, dec_deg)
    return float(x), float(y)

def in_bounds(image: np.ndarray, x: float, y: float, margin: int = 10) -> bool:
    h, w = image.shape
    return (margin <= x < (w - margin)) and (margin <= y < (h - margin))

"""## Gaussian PSF fit (FWHM + Flux)"""

from scipy.optimize import curve_fit


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



"""## Instrumental magnitude"""

def m_inst_from_flux(flux_net: float) -> float:
    if (flux_net is None) or (not np.isfinite(flux_net)) or flux_net <= 0:
        return np.nan
    return float(-2.5 * np.log10(flux_net))

# -*- coding: utf-8 -*-
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import json
import urllib.parse
import urllib.request

# ── Camera sensor database ────────────────────────────────────────────────────
# Source: Janesick (2001) PTC method; Canon 6D2 values from photonstophotos.net
# ISO 800 : gain = 3.78 e-/DN, read_noise = 5.89 e-   (PTC-measured)
# ISO 1600: gain = 1.89 e-/DN, read_noise = 4.21 e-
# ISO 3200: gain = 0.94 e-/DN, read_noise = 3.61 e-
# NOTE: gain_e_per_adu is also written into FITS GAIN header by Calibration_v5.
#       apply_gain_from_header() will prefer the FITS header value over this DB.
CAMERA_SENSOR_DB = {
    "Canon_6D2": {
        800:  {"gain": 3.78, "read_noise": 5.89},
        1600: {"gain": 1.89, "read_noise": 4.21},
        3200: {"gain": 0.94, "read_noise": 3.61},
    }
}

# Airmass warning threshold  (X > 2.0 → altitude < ~30°)

ISO_HEADER_KEYS = [
    "ISO", "ISOSPEED", "ISOSPEEDRATINGS", "ISOSPEEDRATING", "ISO_SPEED",
]

AIRMASS_WARN_THRESHOLD = 2.0   # alt ≈ 30°；低於此仰角輸出 WARN，仍繼續測光
# camera_sensor_db 由 cfg_from_yaml() 從 yaml sensor_db 建立；
# CAMERA_SENSOR_DB 為備用參考值，不在此處注入（避免覆蓋 yaml 設定）


def _parse_iso(val) -> "int | None":
    try:
        iso = int(float(str(val).strip()))
        if iso > 0:
            return iso
    except Exception:
        return None
    return None


def get_iso_value(header) -> "int | None":
    """
    ISO priority: (1) FITS header keys, (2) cfg.iso_setting.
    The interactive input() fallback has been removed to allow
    unattended batch execution in both local and Colab environments.
    Set cfg.iso_setting in the Cfg block above if your FITS headers
    do not carry an ISO key.
    """
    for key in ISO_HEADER_KEYS:
        if key in header:
            iso = _parse_iso(header.get(key))
            if iso is not None:
                return iso
    if cfg.iso_setting is not None:
        return int(cfg.iso_setting)
    return None


def apply_gain_from_header(header, force: bool = False) -> None:
    """
    Populate cfg.gain_e_per_adu and cfg.read_noise_e.
    Priority: (1) FITS GAIN / RDNOISE keywords written by Calibration_v5,
              (2) CAMERA_SENSOR_DB look-up by ISO.
    """
    # Priority 1 – FITS header written by Calibration_v5
    if not force:
        fits_gain = header.get("GAIN")
        fits_rn   = header.get("RDNOISE")
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

    # Priority 2 – sensor DB look-up
    if cfg.camera_model is None or cfg.camera_sensor_db is None:
        return
    iso = get_iso_value(header)
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
    print(f"[GAIN] ISO {iso} → gain={cfg.gain_e_per_adu:.4f} e-/DN, "
          f"read_noise={cfg.read_noise_e:.4f} e-")


def require_cfg_values() -> None:
    missing = []
    if cfg.saturation_threshold is None:
        missing.append("saturation_threshold")
    if cfg.gain_e_per_adu is None:
        missing.append("gain_e_per_adu")
    if cfg.read_noise_e is None:
        missing.append("read_noise_e")
    if missing:
        raise ValueError("Missing cfg values: " + ", ".join(missing))


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


def is_saturated(max_pix: float, threshold: "float | None") -> bool:
    return threshold is not None and np.isfinite(max_pix) and max_pix >= threshold


def _growth_radius_for_star(
    image: np.ndarray,
    x: float,
    y: float,
    r_min: int,
    r_max: int,
    growth_fraction: float,
) -> "float | None":
    radii  = np.arange(r_min, r_max + 0.5, 0.5)
    fluxes = []
    for r in radii:
        r_in, r_out = compute_annulus_radii(r, cfg.annulus_r_in, cfg.annulus_r_out)
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
        sub = (comp_df.sort_values("m_cat") if "m_cat" in comp_df.columns
               else comp_df).head(max_stars)
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

    radii = []
    for x, y in stars_xy:
        r_sel = _growth_radius_for_star(img, x, y, r_min, r_max, growth_fraction)
        if r_sel is None:
            continue
        if cfg.saturation_threshold is not None:
            r_in, r_out = compute_annulus_radii(r_sel, cfg.annulus_r_in, cfg.annulus_r_out)
            phot = aperture_photometry(img, x, y, r_sel, r_in, r_out)
            if is_saturated(phot.get("max_pix", np.nan), cfg.saturation_threshold):
                continue
        radii.append(r_sel)
    return float(np.nanmedian(radii)) if radii else None


def robust_zero_point(
    m_inst: np.ndarray,
    m_cat: np.ndarray,
    sigma: float = 3.0,
    max_iter: int = 5,
    min_points: int = 3,
    weights: "np.ndarray | None" = None,
) -> "dict | None":
    """
    Iteratively re-weighted linear regression: m_inst = a * m_cat + b.

    Returns
    -------
    dict with keys:
        a        – slope
        b        – intercept
        r2       – coefficient of determination (R²) on the inlier subset
        mask     – boolean array marking inliers used in the final fit
    or None if fewer than min_points finite values exist.

    Physical note
    -------------
    In single-night differential photometry the slope 'a' should be close
    to 1.0.  Significant deviation indicates colour-dependent atmospheric
    extinction or a poorly-matched comparison ensemble; investigate before
    accepting the result.  R² is provided so the caller can flag low-quality
    zero-point solutions (R² < 0.95 warrants inspection).
    """
    m_inst = np.asarray(m_inst, dtype=float)
    m_cat  = np.asarray(m_cat,  dtype=float)

    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape != m_inst.shape:
            w = None

    mask = np.isfinite(m_inst) & np.isfinite(m_cat)
    if w is not None:
        mask = mask & np.isfinite(w) & (w > 0)
    if int(mask.sum()) < int(min_points):
        return None

    def _fit(msk):
        if w is None:
            return np.polyfit(m_cat[msk], m_inst[msk], 1)
        return np.polyfit(m_cat[msk], m_inst[msk], 1, w=np.sqrt(w[msk]))

    a, b = _fit(mask)
    for _ in range(int(max_iter)):
        resid   = m_inst - (a * m_cat + b)
        std     = float(np.nanstd(resid[mask])) if np.any(mask) else np.nan
        if not np.isfinite(std) or std == 0:
            break
        new_mask = mask & (np.abs(resid) <= sigma * std)
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask
        if int(mask.sum()) < int(min_points):
            break
        a, b = _fit(mask)

    # ── R² on the inlier subset ───────────────────────────────────────────────
    y_true = m_inst[mask]
    y_pred = a * m_cat[mask] + b
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    if r2 < 0.95:
        _phot_logger.debug("[WARN] Zero-point fit R²=%.4f < 0.95. "
                           "Check comparison star quality or airmass spread.", r2)

    return {"a": float(a), "b": float(b), "r2": float(r2), "mask": mask}


def mag_error_from_flux(
    flux_net: float,
    b_sky_std: float,
    n_pix: int,
    gain_e_per_adu: "float | None",
    read_noise_e: "float | None",
    n_sky: "int | None" = None,
) -> float:
    """CCD noise equation (Merline & Howell, 1995).

    完整公式（DESIGN_DECISIONS_v6.md §3.6.5）：

        sigma_flux² = N_pix × [S/G + b_sky_std² × (1 + N_pix/N_sky)]
        sigma_mag   = (2.5/ln10) × (sigma_flux / flux_net)

    Parameters
    ----------
    flux_net      : 孔徑淨通量（DN）= 孔徑總和 - N_pix × b_sky_median
    b_sky_std     : 背景環像素標準差（DN/px）——必須用標準差，非中位數
    n_pix         : 孔徑像素數
    gain_e_per_adu: gain（e-/DN）
    read_noise_e  : 讀出噪聲（e-）
    n_sky         : 背景環像素數；None 時退化為 sky_correction=1（舊式下限）

    Notes
    -----
    v5 之前的 bug：誤用 b_sky（中位數）代入噪聲項，導致 sigma_mag 偏高 5-7 倍。
    此版本已修正為使用 b_sky_std。
    """
    if gain_e_per_adu is None:
        return np.nan
    if not (np.isfinite(flux_net) and flux_net > 0):
        return np.nan
    if n_pix is None or n_pix <= 0:
        return np.nan

    gain = float(gain_e_per_adu)
    rn = float(read_noise_e) if read_noise_e is not None else 0.0

    # 訊號項（e-）
    f_e = flux_net * gain
    if f_e <= 0:
        return np.nan

    # 背景噪聲修正因子 (1 + N_pix/N_sky)；n_sky=None 時因子退化為 1.0
    if n_sky is not None and n_sky > 0:
        sky_correction = 1.0 + n_pix / float(n_sky)
    else:
        sky_correction = 1.0

    # 背景標準差換算為 e-/px
    bstd_e = float(b_sky_std) * gain if np.isfinite(b_sky_std) else 0.0

    # 完整噪聲（e-）
    sigma_flux_e = np.sqrt(
        max(f_e, 0.0)
        + n_pix * sky_correction * (max(bstd_e ** 2, 0.0) + rn ** 2)
    )

    return float(1.0857 * (sigma_flux_e / f_e))


def _pick_col(df: pd.DataFrame, names: list) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"Missing columns. Tried: {names}, available: {list(df.columns)}")


def _parse_ra_dec(ra_val, dec_val) -> "tuple[float, float]":
    if ra_val is None or dec_val is None:
        return np.nan, np.nan
    try:
        return float(str(ra_val).strip()), float(str(dec_val).strip())
    except Exception:
        pass
    ra_s, dec_s = str(ra_val).strip(), str(dec_val).strip()
    for unit_pair in [(u.hourangle, u.deg), (u.deg, u.deg)]:
        try:
            sc = SkyCoord(ra_s, dec_s, unit=unit_pair)
            return sc.ra.deg, sc.dec.deg
        except Exception:
            continue
    return np.nan, np.nan


def read_aavso_seq_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df) == 0:
        return df
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    ra_col  = _pick_col(df, ["ra", "ra_deg", "raj2000", "ra (deg)", "ra_hms"])
    dec_col = _pick_col(df, ["dec", "dec_deg", "dej2000", "dec (deg)", "dec_dms"])
    m_col   = _pick_col(df, ["v", "v_mag", "vmag", "mag_v"])
    err_col = next((c for c in ["v_err", "verr", "e_v", "v_error", "v_err_mag"]
                    if c in df.columns), None)
    rows = []
    for _, r in df.iterrows():
        ra_deg, dec_deg = _parse_ra_dec(r[ra_col], r[dec_col])
        if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
            continue
        m_cat = r[m_col]
        if not np.isfinite(m_cat):
            continue
        row = {"ra_deg": float(ra_deg), "dec_deg": float(dec_deg), "m_cat": float(m_cat)}
        if err_col is not None and np.isfinite(r[err_col]):
            row["m_err"] = float(r[err_col])
        rows.append(row)
    return pd.DataFrame(rows)


def fetch_aavso_vsp_api(
    star_name: str,
    fov_arcmin: float,
    maglimit: float,
    ra_deg: "float | None" = None,
    dec_deg: "float | None" = None,
) -> pd.DataFrame:
    # www.aavso.org/apps/vsp/api/chart/ 是正確的匿名可用 endpoint
    # 優先用星名查詢；若 400（星名不在 AAVSO）則 fallback 到 ra/dec 座標查詢
    params = {"star": star_name, "fov": fov_arcmin, "maglimit": maglimit, "format": "json"}
    url = "https://www.aavso.org/apps/vsp/api/chart/?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as _e:
        # 星名查詢失敗（例如 400 Bad Request）→ 改用 ra/dec 座標查詢
        if ra_deg is None or dec_deg is None:
            raise
        _ra_hms = f"{ra_deg / 15:.6f}"  # AAVSO 接受小時制或度制；用度直接傳
        params_coord = {
            "ra": f"{ra_deg:.6f}",
            "dec": f"{dec_deg:.6f}",
            "fov": fov_arcmin,
            "maglimit": maglimit,
            "format": "json",
        }
        url_coord = "https://www.aavso.org/apps/vsp/api/chart/?" + urllib.parse.urlencode(params_coord)
        print(f"  [AAVSO] 星名查詢失敗({_e})，改用座標查詢：RA={ra_deg:.4f} Dec={dec_deg:.4f}")
        with urllib.request.urlopen(url_coord, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    phot = data.get("photometry", [])
    rows = []
    for item in phot:
        ra_deg, dec_deg = _parse_ra_dec(item.get("ra"), item.get("dec"))
        if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
            continue
        m_cat, m_err = None, None
        bands  = item.get("bands", {})
        vinfo  = None
        if isinstance(bands, dict):
            vinfo = bands.get("V") or bands.get("v")
        elif isinstance(bands, list):
            vinfo = next((b for b in bands if str(b.get("band", "")).upper() == "V"), None)
        if isinstance(vinfo, dict):
            m_cat = vinfo.get("mag") or vinfo.get("magnitude") or vinfo.get("value")
            m_err = vinfo.get("error") or vinfo.get("err") or vinfo.get("sigma")
        if m_cat is None:
            m_cat = next((item.get(k) for k in ("v", "v_mag", "vmag") if k in item), None)
        if m_cat is None:
            continue
        row = {"ra_deg": float(ra_deg), "dec_deg": float(dec_deg), "m_cat": float(m_cat)}
        if m_err is not None and np.isfinite(m_err):
            row["m_err"] = float(m_err)
        rows.append(row)
    return pd.DataFrame(rows)


def filter_catalog_in_frame(
    df: pd.DataFrame,
    wcs_obj: WCS,
    shape: "tuple[int, int]",
    margin: int = 10,
) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    h, w = shape
    rows = []
    for _, r in df.iterrows():
        try:
            ra, dec = float(r["ra_deg"]), float(r["dec_deg"])
        except Exception:
            continue
        x, y = radec_to_pixel(wcs_obj, ra, dec)
        if margin <= x < w - margin and margin <= y < h - margin:
            row = r.to_dict()
            row["x"], row["y"] = float(x), float(y)
            rows.append(row)
    return pd.DataFrame(rows)


def select_comp_from_catalog(
    catalog_df: pd.DataFrame,
    mag_min: float,
    mag_max: float,
    max_refs: int = 15,
) -> "tuple[list, pd.DataFrame]":
    if catalog_df is None or len(catalog_df) == 0:
        return [], pd.DataFrame()
    df = catalog_df.copy()
    if "m_cat" in df.columns:
        df = df[np.isfinite(df["m_cat"])]
        df = df[(df["m_cat"] >= mag_min) & (df["m_cat"] <= mag_max)]
    if "m_err" in df.columns:
        df = df.sort_values(["m_err", "m_cat"])
    elif "m_cat" in df.columns:
        df = df.sort_values("m_cat")
    df = df.head(max_refs).copy()
    comp_refs = []
    for _, r in df.iterrows():
        m_err = None
        if "m_err" in r:
            try:
                m_err = float(r["m_err"]) if np.isfinite(r["m_err"]) else None
            except Exception:
                pass
        comp_refs.append((float(r["ra_deg"]), float(r["dec_deg"]), float(r["m_cat"]), m_err))
    return comp_refs, df


def select_check_star(target_radec_deg, comp_refs, catalog_df):
    if cfg.check_star_radec_deg is not None:
        ra_c, dec_c = cfg.check_star_radec_deg
        m_cat = None
        if catalog_df is not None and len(catalog_df) > 0 and "m_cat" in catalog_df.columns:
            sc_cat = SkyCoord(catalog_df["ra_deg"].values * u.deg,
                              catalog_df["dec_deg"].values * u.deg)
            sc_k   = SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg)
            sep    = sc_cat.separation(sc_k).arcsec
            idx    = int(np.argmin(sep))
            if np.isfinite(sep[idx]) and sep[idx] <= cfg.apass_match_arcsec:
                m_cat = float(catalog_df.iloc[idx]["m_cat"])
        return (float(ra_c), float(dec_c), m_cat)

    if catalog_df is None or len(catalog_df) == 0:
        print("[WARN] No catalog available for auto check-star selection.")
        return None

    cand = catalog_df.copy()
    if "max_pix" in cand.columns and cfg.saturation_threshold is not None:
        cand = cand[~cand["max_pix"].apply(lambda v: is_saturated(v, cfg.saturation_threshold))]
    if comp_refs:
        comp_coords = SkyCoord(
            [r[0] for r in comp_refs] * u.deg,
            [r[1] for r in comp_refs] * u.deg,
        )
        cand_coords = SkyCoord(cand["ra_deg"].values * u.deg,
                               cand["dec_deg"].values * u.deg)
        _, sep2d, _ = cand_coords.match_to_catalog_sky(comp_coords)
        cand = cand[sep2d.arcsec > cfg.apass_match_arcsec].copy()

    if len(cand) == 0:
        print("[WARN] Auto check-star selection failed: no safe candidates.")
        return None

    target      = SkyCoord(ra=target_radec_deg[0] * u.deg, dec=target_radec_deg[1] * u.deg)
    cand_coords = SkyCoord(cand["ra_deg"].values * u.deg, cand["dec_deg"].values * u.deg)
    idx = int(np.nanargmin(target.separation(cand_coords).arcsec))
    row = cand.iloc[idx]
    print("[INFO] check_star_radec_deg not set; auto-selected nearest non-comp candidate.")
    return (
        float(row["ra_deg"]),
        float(row["dec_deg"]),
        float(row["m_cat"]) if "m_cat" in row and np.isfinite(row["m_cat"]) else None,
    )


def _float_or_none(val) -> "float | None":
    try:
        return float(val)
    except Exception:
        return None


def compute_airmass(
    t: Time,
    ra_deg: float,
    dec_deg: float,
    lat: float,
    lon: float,
    height: float,
) -> float:
    """
    Compute airmass using Young (1994) formula, which is more accurate
    than the simple sec(z) approximation at high zenith angles.

        X = 1 / (cos(z) + 0.50572 * (96.07995 - z_deg)^-1.6364)

    Parameters
    ----------
    t       : astropy Time object (UTC)
    ra_deg  : target right ascension in degrees
    dec_deg : target declination in degrees
    lat     : observatory geodetic latitude in degrees
    lon     : observatory geodetic longitude in degrees
    height  : observatory elevation in metres
    """
    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height * u.m)
    sc       = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    altaz    = sc.transform_to(AltAz(obstime=t, location=location))
    alt_deg  = float(altaz.alt.deg)
    if alt_deg <= 0.0:
        return np.inf                          # target below horizon
    z_deg    = 90.0 - alt_deg
    cos_z    = np.cos(np.radians(z_deg))
    airmass  = 1.0 / (cos_z + 0.50572 * (96.07995 - z_deg) ** (-1.6364))
    return float(airmass)


def time_from_header(
    header,
    ra_deg: float,
    dec_deg: float,
) -> "tuple[float, float, float]":
    """
    Extract timing and airmass from a FITS header.

    Time system
    -----------
    Returns BJD_TDB (Barycentric Julian Date, TDB timescale), which is
    the current AAVSO standard for variable star photometry (Eastman et al.,
    2010).  HJD is no longer returned; BJD_TDB replaces it throughout.

    Exposure mid-point correction
    ------------------------------
    For periodic variables with periods < 1 h (e.g. delta Scuti), using the
    exposure start time (DATE-OBS) instead of the midpoint introduces a phase
    error of EXPTIME / (2 * P).  For EXPTIME = 60 s and P = 3800 s this is
    ~0.008 phase units — detectable on a well-sampled light curve.
    MID-OBS = DATE-OBS + EXPTIME/2 is therefore preferred when available.

    Returns
    -------
    (mjd, bjd_tdb, airmass)
        mjd      : Modified Julian Date (UTC) at exposure midpoint
        bjd_tdb  : Barycentric Julian Date (TDB) at exposure midpoint
        airmass  : Young (1994) airmass; np.nan if location unavailable
    """
    # ── 1. Resolve observation time (prefer MID-OBS, fall back to DATE-OBS) ──
    mid_obs  = header.get("MID-OBS")
    date_obs = header.get("DATE-OBS") or header.get("DATEOBS")

    time_str  = mid_obs if mid_obs else date_obs
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

    # ── 2. If only DATE-OBS available, add EXPTIME/2 to get midpoint ─────────
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

    # ── 3. BJD_TDB conversion ─────────────────────────────────────────────────
    lat    = cfg.obs_lat_deg
    lon    = cfg.obs_lon_deg
    height = cfg.obs_height_m

    if lat is None or lon is None:
        lat    = _float_or_none(header.get("SITELAT") or header.get("OBS-LAT")
                                or header.get("OBSLAT"))
        lon    = _float_or_none(header.get("SITELONG") or header.get("SITELON")
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
        sc       = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
        ltt      = t.light_travel_time(sc, "barycentric", location=location)
        bjd_tdb  = float((t.tdb + ltt).jd)
    except Exception as exc:
        print(f"[WARN] BJD_TDB calculation failed: {exc}")

    # ── 4. Airmass (Young 1994) + warning ─────────────────────────────────────
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
                "[WARN] High airmass X=%.2f (altitude=%.1f°) in %s. "
                "Differential photometry accuracy may be reduced.",
                airmass, alt_deg, header.get("FILENAME", "unknown"),
            )
    except Exception as exc:
        print(f"[WARN] Airmass calculation failed: {exc}")

    return mjd, bjd_tdb, airmass



"""## Auto-select comparison stars (APASS)

"""

# -*- coding: utf-8 -*-
"""
Cell 18 — 比較星自動選取

設計依據（本研究）：
    選取範圍：以目標星為圓心，影像短邊像素數一半為半徑的圓內
              所有偵測到的未飽和星。
    距離加權：w_i = 1 / (d_i + ε)²
              ε = plate_scale_arcsec / 2（防浮點數除以零）
    星表優先：AAVSO ≥ aavso_min_stars 顆 → 用 AAVSO 的 m_cat
              否則 → APASS
    診斷圖  ：每張影像輸出 AAVSO（紅）+ APASS（藍）雙色散佈圖

參考：Honeycutt (1992) PASP 104, 435.
"""
from astropy.coordinates import SkyCoord
import astropy.units as u
import io
import urllib.parse
import urllib.request
from pathlib import Path

APASS_SCS_URL = "https://dc.g-vo.org/apass/q/cone/scs.xml"


def _selection_radius_px(image: np.ndarray) -> float:
    """
    回傳選取圓的半徑（像素）。
    策略：短邊像素數的一半，保證不同幀之間使用相同標準。
    """
    return float(min(image.shape)) / 2.0


def _stars_in_circle(
    cand_df: pd.DataFrame,
    center_x: float,
    center_y: float,
    radius_px: float,
    image_shape: tuple,
) -> pd.DataFrame:
    """
    從候選星表中篩選落在選取圓內的星。

    選取圓以**影像中心**為圓心（非目標星），半徑為影像短邊的一半。
    以影像中心為圓心可使比較星在空間上對稱分布，大氣梯度的系統誤差
    互相抵消效果最佳。

    Parameters
    ----------
    cand_df     : 含 x, y 欄位的候選星 DataFrame（像素座標）。
    center_x    : 選取圓圓心 x（影像中心，= NAXIS1 / 2）。
    center_y    : 選取圓圓心 y（影像中心，= NAXIS2 / 2）。
    radius_px   : 選取圓半徑（像素，= 影像短邊 / 2）。
    image_shape : 影像 (height, width)，保留供未來使用。
    """
    rows = []
    for _, r in cand_df.iterrows():
        x, y = float(r.get("x", np.nan)), float(r.get("y", np.nan))
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        dist_px = float(np.hypot(x - center_x, y - center_y))
        if dist_px <= radius_px:
            r2 = r.to_dict()
            r2["dist_px"] = dist_px
            rows.append(r2)

    return pd.DataFrame(rows)


def fetch_apass_cone(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float = 1.0,
    maxrec: int = 5000,
) -> pd.DataFrame:
    """查詢 APASS DR9（astroquery.vizier），回傳視場內所有 APASS 星。

    改用 VizieR 因 dc.g-vo.org endpoint 已失效（404）。
    結果欄位統一為 ra_deg, dec_deg, m_cat, m_err。
    """
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord as _SkyCoord
        import astropy.units as _u

        # 查所有通道需要的波段：V（G1/G2）、B、Rc（R 通道）
        # 注意：改機 Canon 6D2 R 通道延伸至近紅外，Rc 僅為近似，
        # 嚴格色彩轉換係數需另行標定（見已知限制 #10）。
        # APASS DR9 VizieR 欄位：r 波段名稱含單引號 r'mag / e_r'mag
        v = Vizier(
            columns=["RAJ2000", "DEJ2000", "Vmag", "e_Vmag", "Bmag", "e_Bmag",
                     "r'mag", "e_r'mag"],
            row_limit=maxrec,
        )
        result = v.query_region(
            _SkyCoord(ra_deg, dec_deg, unit="deg"),
            radius=radius_deg * _u.deg,
            catalog="II/336/apass9",
        )
        if not result:
            print("  [WARN] APASS VizieR 查詢無結果。")
            return pd.DataFrame()
        tbl = result[0]
        tbl_df = tbl.to_pandas()
        col_map = {
            "RAJ2000": "ra_deg", "DEJ2000": "dec_deg",
            "Vmag":    "vmag",   "e_Vmag":  "e_vmag",
            "Bmag":    "bmag",   "e_Bmag":  "e_bmag",
            "r'mag":   "rmag",   "e_r'mag": "e_rmag",
        }
        df = tbl_df.rename(columns={k: v for k, v in col_map.items() if k in tbl_df.columns})
        df = df[np.isfinite(df["vmag"])].reset_index(drop=True)
        return df
    except Exception as exc:
        print(f"  [WARN] APASS 查詢失敗：{exc}")
        return pd.DataFrame()


def fetch_tycho2_cone(
    ra_deg: float,
    dec_deg: float,
    radius_arcmin: float = 40.0,
    mag_min: float = 6.0,
    mag_max: float = 13.0,
) -> pd.DataFrame:
    """
    查 Tycho-2（I/259/tyc2），回傳 ra_deg, dec_deg, vmag, e_vmag, source。
    色彩轉換：V = VT - 0.090 × (BT - VT)（ESA 1997 Vol.1 eq.1.3.20）
    誤差傳遞：e_V = sqrt(e_VT² + 0.090² × (e_BT² + e_VT²))
    BT 或 VT 缺失的星跳過。
    """
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord as _SkyCoord
        import astropy.units as _u

        v = Vizier(
            columns=["RAmdeg", "DEmdeg", "BTmag", "VTmag", "e_BTmag", "e_VTmag"],
            row_limit=2000,
        )
        result = v.query_region(
            _SkyCoord(ra_deg, dec_deg, unit="deg"),
            radius=radius_arcmin * _u.arcmin,
            catalog="I/259/tyc2",
        )
        if not result:
            print("  [WARN] Tycho-2 查詢無結果。")
            return pd.DataFrame()
        df = result[0].to_pandas()
        rows = []
        for _, r in df.iterrows():
            try:
                bt = float(r["BTmag"]); vt = float(r["VTmag"])
                e_bt = float(r["e_BTmag"]); e_vt = float(r["e_VTmag"])
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(bt) and np.isfinite(vt) and
                    np.isfinite(e_bt) and np.isfinite(e_vt)):
                continue
            v_mag = vt - 0.090 * (bt - vt)
            tycho_mag_min = max(mag_min, 7.0)
            if not (tycho_mag_min <= v_mag <= mag_max):
                continue
            e_v = float(np.sqrt(e_vt**2 + 0.090**2 * (e_bt**2 + e_vt**2)))
            rows.append({
                "ra_deg":  float(r["RAmdeg"]),
                "dec_deg": float(r["DEmdeg"]),
                "vmag":    v_mag,
                "e_vmag":  e_v,
                "BT":      bt,
                "VT":      vt,
                "source":  "Tycho2",
            })
        df_out = pd.DataFrame(rows)
        print(f"  [Tycho-2] 查詢={len(df)}  色彩轉換後通過 mag 篩選={len(df_out)}")
        return df_out
    except Exception as exc:
        print(f"  [WARN] Tycho-2 查詢失敗：{exc}")
        return pd.DataFrame()


def fetch_gaia_dr3_cone(
    ra_deg: float,
    dec_deg: float,
    radius_arcmin: float = 40.0,
    mag_min: float = 6.0,
    mag_max: float = 19.0,
    channel: str = "G1",
) -> pd.DataFrame:
    """
    查 Gaia DR3（I/355/gaiadr3），依通道回傳轉換後星等。
    G1/G2：G→V  (Riello et al. 2021)
      V = G + 0.02704 - 0.01424c + 0.2156c² - 0.01426c³  (c=BP-RP)
    R   ：G→Rc  (Riello et al. 2021)
      Rc = G - (0.02275 + 0.3961c - 0.1243c² - 0.01396c³ + 0.003775c⁴)
    B   ：G→B   (使用者指定公式)
      B = G + 0.0939 + 0.6758c + 0.0743c²
      ⚠️ 僅適用 BP-RP < 2，超出範圍仍計算但標記 warn_color
    Gaia 亮星容忍：mag_min 最小 6.0（G<3 才飽和）。
    """
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord as _SkyCoord
        import astropy.units as _u

        gaia_mag_min = max(mag_min, 6.0)
        ch = channel.upper()

        v = Vizier(
            columns=["RA_ICRS", "DE_ICRS", "Gmag", "BPmag", "RPmag", "e_Gmag"],
            row_limit=1800,
        )
        result = v.query_region(
            _SkyCoord(ra_deg, dec_deg, unit="deg"),
            radius=radius_arcmin * _u.arcmin,
            catalog="I/355/gaiadr3",
        )
        if not result:
            print("  [WARN] Gaia DR3 查詢無結果。")
            return pd.DataFrame()
        df = result[0].to_pandas()
        rows = []
        n_warn_color = 0
        for _, r in df.iterrows():
            try:
                g = float(r["Gmag"])
                bp = float(r["BPmag"])
                rp = float(r["RPmag"])
                e_g = float(r["e_Gmag"])
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(g) and np.isfinite(bp) and np.isfinite(rp)):
                continue
            c = bp - rp
            warn_color = False

            if ch in ("G1", "G2"):
                conv_mag = g + 0.02704 - 0.01424*c + 0.2156*c**2 - 0.01426*c**3
                df_dc = -0.01424 + 0.4312*c - 0.04278*c**2
                e_bp_safe = 0.02; e_rp_safe = 0.02
                e_conv = float(np.sqrt(e_g**2 + df_dc**2 * (e_bp_safe**2 + e_rp_safe**2)))
            elif ch == "R":
                conv_mag = g - (0.02275 + 0.3961*c - 0.1243*c**2 - 0.01396*c**3 + 0.003775*c**4)
                dg_dc = 0.3961 - 0.2486*c - 0.04188*c**2 + 0.01510*c**3
                e_bp_safe = 0.02; e_rp_safe = 0.02
                e_conv = float(np.sqrt(e_g**2 + dg_dc**2 * (e_bp_safe**2 + e_rp_safe**2)))
            elif ch == "B":
                if c >= 2.0:
                    warn_color = True
                    n_warn_color += 1
                conv_mag = g + 0.0939 + 0.6758*c + 0.0743*c**2
                db_dc = 0.6758 + 0.1486*c
                e_bp_safe = 0.02; e_rp_safe = 0.02
                e_conv = float(np.sqrt(e_g**2 + db_dc**2 * (e_bp_safe**2 + e_rp_safe**2)))
            else:
                continue

            if not (gaia_mag_min <= conv_mag <= mag_max):
                continue
            rows.append({
                "ra_deg":      float(r["RA_ICRS"]),
                "dec_deg":     float(r["DE_ICRS"]),
                "vmag":        conv_mag,
                "e_vmag":      e_conv,
                "Gmag":        g,
                "BPmag":       bp,
                "RPmag":       rp,
                "warn_color":  warn_color,
                "source":      "GaiaDR3",
            })
        if n_warn_color > 0:
            print(f"  [WARN] Gaia B 通道：{n_warn_color} 顆星 BP-RP ≥ 2，色彩轉換誤差偏大（> 0.1 等）")
        df_out = pd.DataFrame(rows)
        print(f"  [Gaia DR3] ch={ch}  查詢={len(df)}  轉換後通過 mag 篩選={len(df_out)}")
        return df_out
    except Exception as exc:
        print(f"  [WARN] Gaia DR3 查詢失敗：{exc}")
        return pd.DataFrame()


def _match_catalog_to_detected(
    detected_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    ra_col: str,
    dec_col: str,
    mag_col: str,
    err_col: "str | None",
    match_arcsec: float,
    mag_min: float,
    mag_max: float,
) -> pd.DataFrame:
    """
    把偵測到的星（pixel 座標）與星表（RA/Dec）做位置匹配，
    回傳含 m_cat 欄位的 DataFrame。
    """
    if len(detected_df) == 0 or len(catalog_df) == 0:
        return pd.DataFrame()

    det_coords = SkyCoord(
        detected_df["ra_deg"].values * u.deg,
        detected_df["dec_deg"].values * u.deg,
    )
    cat_coords = SkyCoord(
        catalog_df[ra_col].astype(float).values * u.deg,
        catalog_df[dec_col].astype(float).values * u.deg,
    )
    idx, sep2d, _ = det_coords.match_to_catalog_sky(cat_coords)

    matched = detected_df.copy().reset_index(drop=True)
    matched["m_cat"] = catalog_df.iloc[idx][mag_col].values
    if err_col and err_col in catalog_df.columns:
        matched["m_err"] = catalog_df.iloc[idx][err_col].values
    matched["cat_sep_arcsec"] = sep2d.arcsec

    matched = matched[
        (matched["cat_sep_arcsec"] <= match_arcsec)
        & np.isfinite(matched["m_cat"])
        & (matched["m_cat"] >= mag_min)
        & (matched["m_cat"] <= mag_max)
    ].copy()

    return matched


def _save_zeropoint_diagnostic(
    frame_name: str,
    aavso_matched: "pd.DataFrame | None",
    apass_matched: "pd.DataFrame | None",
    fit_aavso: "dict | None",
    fit_apass: "dict | None",
    active_source: str,
    diag_dir: Path,
) -> None:
    """
    輸出每張影像的零點診斷散佈圖。

    x 軸：星表星等 m_cat
    y 軸：儀器星等 m_inst
    紅色：AAVSO 比較星 + 回歸線（若有）
    藍色：APASS 比較星 + 回歸線（若有）
    灰色虛線：理想斜率 = 1 的參考線
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    x_all = []
    for label, df_m, fit, color, marker in [
        ("AAVSO", aavso_matched, fit_aavso, "red", "o"),
        ("APASS", apass_matched, fit_apass, "steelblue", "s"),
    ]:
        if df_m is None or len(df_m) == 0:
            continue
        m_cat  = df_m["m_cat"].values
        m_inst = df_m["m_inst_matched"].values if "m_inst_matched" in df_m.columns else np.full(len(df_m), np.nan)
        ok = np.isfinite(m_cat) & np.isfinite(m_inst)
        if not ok.any():
            continue
        ax.scatter(m_cat[ok], m_inst[ok], s=18, alpha=0.7,
                   color=color, marker=marker,
                   label=f"{label} (n={ok.sum()})")
        x_all.extend(m_cat[ok].tolist())

        if fit is not None and np.isfinite(fit.get("a", np.nan)):
            x_line = np.linspace(m_cat[ok].min(), m_cat[ok].max(), 100)
            y_line = fit["a"] * x_line + fit["b"]
            r2_str = f"R²={fit['r2']:.3f}" if np.isfinite(fit.get("r2", np.nan)) else ""
            ax.plot(x_line, y_line, color=color, lw=1.5,
                    label=f"{label} fit  a={fit['a']:.3f}  b={fit['b']:.3f}  {r2_str}")

    if x_all:
        x_range = np.array([min(x_all), max(x_all)])
        ax.plot(x_range, x_range, "k--", lw=0.8, alpha=0.4, label="ideal (slope=1)")

    ax.invert_yaxis()
    ax.set_xlabel("Catalogue magnitude  $m_{cat}$")
    ax.set_ylabel("Instrumental magnitude  $m_{inst}$")
    ax.set_title(
        f"Zero-point diagnostic  [{frame_name}]\n"
        f"Active source: {active_source}",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = diag_dir / f"zp_diag_{Path(frame_name).stem}.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def auto_select_comps(
    wcs_fits_path: Path,
    target_radec_deg: tuple,
    max_detect: int = 500,
    psf_box: int = 25,
    threshold_sigma: float = 5.0,
):
    """
    比較星自動選取主函式。

    選取邏輯
    --------
    1. 偵測視場內所有星（DAOStarFinder）
    2. 取短邊一半半徑圓內的未飽和星
    3. AAVSO ≥ aavso_min_stars 顆 → 用 AAVSO 做正式回歸
       否則 → 用 APASS
    4. 同時查詢 AAVSO 和 APASS 以輸出診斷散佈圖

    Returns
    -------
    comp_refs       : list of (ra, dec, m_cat, m_err | None)
    comp_df_matched : 選入回歸的比較星 DataFrame
    check_star      : (ra, dec, m_cat | None) 或 None
    aavso_matched   : AAVSO 匹配結果（診斷用）
    apass_matched   : APASS 匹配結果（診斷用）
    active_source   : "AAVSO" 或 "APASS"
    """
    with fits.open(wcs_fits_path) as hdul:
        img = hdul[0].data.astype(np.float32)
        hdr = hdul[0].header
        wcs_obj = WCS(hdr)

    ra_t, dec_t = float(target_radec_deg[0]), float(target_radec_deg[1])
    epsilon = cfg.plate_scale_arcsec / 2.0    # 防零 ε（arcsec）
    tgt_sc = SkyCoord(ra=ra_t * u.deg, dec=dec_t * u.deg)
    _h, _w = img.shape

    # 孔徑參數：使用 cfg 靜態值（孔徑估算在本函式之後才執行）
    _ap_r = float(cfg.aperture_radius)
    _r_in, _r_out = compute_annulus_radii(_ap_r, cfg.annulus_r_in, cfg.annulus_r_out)
    _margin = int(np.ceil(_r_out + 2))

    def _catalog_direct_phot(cat_df, ra_col, dec_col, mag_col, err_col, mag_min, mag_max):
        """從星表座標直接做孔徑測光，回傳通過篩選的 DataFrame。
        排除條件：(a) 超出影像邊界+孔徑；(b) aperture 內 max_pix >= 飽和閾值。
        """
        rows = []
        n_bounds, n_sat, n_phot = 0, 0, 0
        for _, row in cat_df.iterrows():
            try:
                ra_c  = float(row[ra_col])
                dec_c = float(row[dec_col])
                m_c   = float(row[mag_col])
                m_e   = float(row[err_col]) if (err_col and err_col in row.index
                                                and np.isfinite(row[err_col])) else np.nan
            except (KeyError, TypeError, ValueError):
                continue
            if not (mag_min <= m_c <= mag_max):
                continue
            xc, yc = radec_to_pixel(wcs_obj, ra_c, dec_c)
            if not in_bounds(img, xc, yc, margin=_margin):
                n_bounds += 1
                continue
            phot = aperture_photometry(img, xc, yc, _ap_r, _r_in, _r_out)
            if phot.get("ok") != 1 or not np.isfinite(phot.get("flux_net", np.nan)):
                n_phot += 1
                continue
            if is_saturated(phot.get("max_pix", np.nan), cfg.saturation_threshold):
                n_sat += 1
                continue
            m_inst = m_inst_from_flux(phot["flux_net"])
            if not np.isfinite(m_inst):
                continue
            rows.append({
                "ra_deg": ra_c, "dec_deg": dec_c,
                "m_cat": m_c, "m_err": m_e,
                "m_inst": m_inst, "m_inst_matched": m_inst,
            })
        print(f"  [直接測光] 星表={len(cat_df)}  mag範圍外跳過  "
              f"邊界排除={n_bounds}  飽和排除={n_sat}  測光失敗={n_phot}  通過={len(rows)}")
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ── 3–5. 依 catalog_priority 查詢並合併比較星 ─────────────────────────────
    # 查詢順序由 cfg.catalog_priority 決定（yaml 設定，預設 [AAVSO, APASS]）。
    # 各星表結果合併後去重（位置 < 3 arcsec 保留優先順序較前的來源）。
    # 合併後統一進強制孔徑測光流程（_catalog_direct_phot）。

    _catalog_raws: "list[pd.DataFrame]" = []   # 各星表查詢結果（標準化欄位）
    _sources_used: "list[str]" = []

    for _cat_name in cfg.catalog_priority:
        _cat_name_up = str(_cat_name).upper()

        if _cat_name_up == "AAVSO":
            seq_df = None
            if cfg.aavso_seq_csv is not None and Path(cfg.aavso_seq_csv).exists():
                seq_df = read_aavso_seq_csv(Path(cfg.aavso_seq_csv))
            elif cfg.aavso_star_name and cfg.aavso_use_api:
                try:
                    seq_df = fetch_aavso_vsp_api(
                        cfg.aavso_star_name, cfg.aavso_fov_arcmin, cfg.aavso_maglimit,
                        ra_deg=ra_t, dec_deg=dec_t,
                    )
                except Exception as exc:
                    print(f"  [WARN] AAVSO API 查詢失敗：{exc}")
            if seq_df is not None and len(seq_df) > 0:
                _std = seq_df.rename(columns={"m_cat": "vmag", "m_err": "e_vmag"})
                _std = _std[["ra_deg", "dec_deg", "vmag"] +
                             (["e_vmag"] if "e_vmag" in _std.columns else [])].copy()
                _std["source"] = "AAVSO"
                _catalog_raws.append(_std)
                _sources_used.append("AAVSO")

        elif _cat_name_up == "APASS":
            apass_raw = fetch_apass_cone(
                ra_t, dec_t, radius_deg=cfg.apass_radius_deg, maxrec=cfg.apass_maxrec,
            )
            if len(apass_raw) > 0:
                _ra_col  = _pick_col(apass_raw, ["ra", "raj2000", "ra_deg", "ra_icrs"])
                _dec_col = _pick_col(apass_raw, ["dec", "dej2000", "dec_deg", "dec_icrs"])
                _band = getattr(cfg, "phot_band", "V")
                _mag_cands = {
                    "B": ["mag_b", "bmag", "b", "b_mag"],
                    "V": ["mag_v", "vmag", "v", "v_mag"],
                    "r": ["mag_r", "rmag", "r", "r_mag", "r_sloan"],
                }
                _err_cands = {
                    "B": ["err_mag_b", "e_bmag", "e_b", "b_err", "bmag_err"],
                    "V": ["err_mag_v", "e_vmag", "e_v", "v_err", "vmag_err"],
                    "r": ["err_mag_r", "e_rmag", "e_r", "r_err", "rmag_err"],
                }
                _mag_col = _pick_col(apass_raw, _mag_cands.get(_band, _mag_cands["V"]))
                _err_col = next(
                    (c for c in _err_cands.get(_band, _err_cands["V"]) if c in apass_raw.columns),
                    None,
                )
                _cols_needed = [_ra_col, _dec_col, _mag_col] + ([_err_col] if _err_col else [])
                _std = apass_raw[_cols_needed].copy().rename(columns={
                    _ra_col: "ra_deg", _dec_col: "dec_deg",
                    _mag_col: "vmag",
                    **({_err_col: "e_vmag"} if _err_col else {}),
                })
                _std["source"] = "APASS"
                _catalog_raws.append(_std)
                _sources_used.append("APASS")

        elif _cat_name_up == "TYCHO2":
            tycho_raw = fetch_tycho2_cone(
                ra_t, dec_t,
                radius_arcmin=cfg.apass_radius_deg * 60.0,
                mag_min=cfg.comp_mag_min, mag_max=cfg.comp_mag_max,
            )
            if len(tycho_raw) > 0:
                _keep_tycho = ["ra_deg", "dec_deg", "vmag", "e_vmag", "source"]
                for _extra_tycho in ["BT", "VT"]:
                    if _extra_tycho in tycho_raw.columns:
                        _keep_tycho.append(_extra_tycho)
                _std = tycho_raw[_keep_tycho].copy()
                _catalog_raws.append(_std)
                _sources_used.append("Tycho2")

        elif _cat_name_up in ("GAIA", "GAIADR3", "GAIA_DR3"):
            _gaia_channel = getattr(cfg, "phot_band", "G1")
            # phot_band 使用 APASS 波段名（V/B/r），轉換回通道名稱
            _band_to_ch = {"V": "G1", "B": "B", "r": "R"}
            _gaia_channel = _band_to_ch.get(_gaia_channel, _gaia_channel)
            gaia_raw = fetch_gaia_dr3_cone(
                ra_t, dec_t,
                radius_arcmin=cfg.apass_radius_deg * 60.0,
                mag_min=cfg.comp_mag_min,
                mag_max=cfg.comp_mag_max,
                channel=_gaia_channel,
            )
            if len(gaia_raw) > 0:
                _keep_cols = ["ra_deg", "dec_deg", "vmag", "e_vmag", "source"]
                for _extra in ["Gmag", "BPmag", "RPmag", "warn_color"]:
                    if _extra in gaia_raw.columns:
                        _keep_cols.append(_extra)
                _std = gaia_raw[_keep_cols].copy()
                _catalog_raws.append(_std)
                _sources_used.append("GaiaDR3")

    # 合併並去重：位置 < 3 arcsec 保留優先順序較前的
    if not _catalog_raws:
        combined_raw = pd.DataFrame()
    else:
        combined_raw = pd.concat(_catalog_raws, ignore_index=True)
        if len(combined_raw) > 1:
            from astropy.coordinates import SkyCoord as _SC
            keep = np.ones(len(combined_raw), dtype=bool)
            coords = _SC(
                combined_raw["ra_deg"].values * u.deg,
                combined_raw["dec_deg"].values * u.deg,
            )
            for i in range(len(combined_raw)):
                if not keep[i]:
                    continue
                seps = coords[i].separation(coords).arcsec
                dups = np.where((seps < 3.0) & (np.arange(len(combined_raw)) > i))[0]
                keep[dups] = False
            combined_raw = combined_raw[keep].reset_index(drop=True)

    # 統一欄位名稱：vmag → m_cat，e_vmag → m_err
    if len(combined_raw) > 0:
        combined_raw = combined_raw.rename(columns={"vmag": "m_cat", "e_vmag": "m_err"})
        if "m_err" not in combined_raw.columns:
            combined_raw["m_err"] = np.nan
        print(f"  [星表合併] {'+'.join(_sources_used)}：合併後 {len(combined_raw)} 筆")

    # 強制孔徑測光篩選
    aavso_matched = None   # 保留給 check_star 選取用（診斷）
    apass_matched = None
    active_df     = None
    active_source = "NONE"

    if len(combined_raw) > 0:
        active_df = _catalog_direct_phot(
            combined_raw, "ra_deg", "dec_deg", "m_cat", "m_err",
            cfg.comp_mag_min, cfg.comp_mag_max,
        )
        active_source = "+".join(_sources_used) if _sources_used else "NONE"
        # 供 check_star 診斷用
        apass_matched = active_df
        aavso_matched = active_df

    if active_df is None or len(active_df) == 0:
        raise RuntimeError(
            f"所有星表（{cfg.catalog_priority}）都找不到足夠的比較星。\n"
            "建議：(1) 放寬 comp_mag_range；(2) 增大 apass_radius_deg；"
            "(3) 確認 catalog_priority 設定。"
        )

    # ── 6. 建立 comp_refs（含距離加權 ε）──────────────────────────────────────
    comp_refs = []
    for _, r in active_df.iterrows():
        ra_c  = float(r["ra_deg"])
        dec_c = float(r["dec_deg"])
        m_cat = float(r["m_cat"])
        m_err = float(r["m_err"]) if ("m_err" in r and np.isfinite(r["m_err"])) else None

        sc_c      = SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg)
        d_arcsec  = float(tgt_sc.separation(sc_c).arcsec)
        weight    = 1.0 / (d_arcsec + epsilon) ** 2    # w_i = 1 / (d_i + ε)²
        comp_refs.append((ra_c, dec_c, m_cat, m_err, weight))

    # ── 7. 檢查星 ──────────────────────────────────────────────────────────────
    catalog_for_check = aavso_matched if aavso_matched is not None and len(aavso_matched) > 0 else apass_matched
    check_star = select_check_star(
        cfg.target_radec_deg, comp_refs, catalog_for_check
    )

    print(f"[比較星] 使用 {active_source}：{len(comp_refs)} 顆")
    print(f"[比較星] epsilon = {epsilon:.4f} arcsec")

    # ── 儲存各星表 CSV ────────────────────────────────────────────────────────
    if hasattr(cfg, "out_dir") and cfg.out_dir is not None:
        _cat_dir = Path(cfg.out_dir) / "catalogs"
        _cat_dir.mkdir(parents=True, exist_ok=True)
        for _raw_df in _catalog_raws:
            if len(_raw_df) == 0:
                continue
            _src_name = str(_raw_df["source"].iloc[0]) if "source" in _raw_df.columns else "unknown"
            _cat_csv = _cat_dir / f"catalog_{_src_name}.csv"
            _raw_df.to_csv(_cat_csv, index=False)
            print(f"  [星表] 已儲存 → {_cat_csv}  ({len(_raw_df)} 筆)")

    return comp_refs, active_df, check_star, aavso_matched, apass_matched, active_source


def differential_mag(m_var_inst: float, m_ref_inst: float, m_ref_cat: float) -> float:
    """
    m_var = m_var_inst - (m_ref_inst - m_ref_cat)
    """
    if not np.isfinite(m_var_inst) or not np.isfinite(m_ref_inst):
        return np.nan
    return float(m_var_inst - (m_ref_inst - m_ref_cat))


def ensemble_normalize(
    df: pd.DataFrame,
    comp_lightcurves: "dict[str, pd.Series]",
    initial_weights: "dict[str, float]",
    time_key: str = "bjd_tdb",
    min_comp_stars: int = 3,
    max_iter: int = 10,
    convergence_tol: float = 1e-4,
) -> "tuple[pd.DataFrame, pd.Series]":
    """
    Broeg (2005) ensemble normalisation.

    每顆比較星 i 的儀器星等時間序列 m_i(t) 估計共同大氣漂移 Δ(t)，
    然後對目標星星等做修正：m_var_norm(t) = m_var(t) − Δ(t)。

    演算法
    ------
    1. 初始權重：以 initial_weights（距離 + 測光誤差複合）為種子。
    2. 迭代：
       a. 計算各比較星偏差 δ_i(t) = m_i(t) − median(m_i)
       b. 計算漂移 Δ(t) = Σ w_i δ_i(t) / Σ w_i（僅對各幀中有效的比較星加總）
       c. 扣除漂移後計算每顆比較星殘差 RMS
       d. 更新權重 w_i = 1 / RMS_i²（Broeg 2005 eq. 4）
       e. 收斂判斷：max |Δ_new(t) − Δ_old(t)| < convergence_tol
    3. 將最終 Δ(t) 寫回 df["delta_ensemble"]，修正值寫入 df["m_var_norm"]。

    Parameters
    ----------
    df               : run_photometry_on_wcs_dir 回傳的 DataFrame。
    comp_lightcurves : {star_id: Series(time_key → m_inst)}，每顆比較星的
                       儀器星等時間序列。NaN 表示該幀該星不可用。
    initial_weights  : {star_id: float}，初始權重種子。
    time_key         : 時間欄位名稱（"bjd_tdb" 或 "mjd"）。
    min_comp_stars   : 計算 Δ(t) 所需的最少可用比較星數；不足時 Δ(t) = NaN。
    max_iter         : Broeg 迭代上限。
    convergence_tol  : 連續兩次迭代 Δ(t) 最大差值的收斂閾值（mag）。

    Returns
    -------
    df_out  : 含 m_var_norm、delta_ensemble 欄位的 DataFrame（原始 df 的副本）。
    delta_t : Δ(t) 時間序列（pd.Series，index = time_key 的值）。

    References
    ----------
    Broeg, C., Fernández, M., & Neuhäuser, R. (2005). A new algorithm for
    differential photometry: Computing an optimum artificial comparison star.
    Astronomische Nachrichten, 326(2), 134–142.
    https://doi.org/10.1002/asna.200410350
    """
    if len(comp_lightcurves) < min_comp_stars:
        _phot_logger.warning(
            "[ensemble] 比較星數量 %d < min_comp_stars %d，跳過 ensemble 正規化。",
            len(comp_lightcurves), min_comp_stars,
        )
        df_out = df.copy()
        df_out["m_var_norm"] = df_out.get("m_var", np.nan)
        df_out["delta_ensemble"] = np.nan
        return df_out, pd.Series(dtype=float)

    star_ids = list(comp_lightcurves.keys())

    # ── 建立比較星矩陣 M：index = time_key 值，columns = star_id ─────────────
    # 對齊到 df 的時間軸；缺測幀填 NaN
    t_index = df[time_key].values
    m_matrix = pd.DataFrame(index=range(len(t_index)), columns=star_ids, dtype=float)
    for sid in star_ids:
        series = comp_lightcurves[sid]
        for row_i, t_val in enumerate(t_index):
            if np.isfinite(t_val) and t_val in series.index:
                m_matrix.loc[row_i, sid] = series[t_val]
            else:
                m_matrix.loc[row_i, sid] = np.nan
    m_arr = m_matrix.values.astype(float)   # shape: (n_frames, n_comp)

    # ── 各比較星時間序列中位數（作為基準，扣除後得到殘差）────────────────────
    med_i = np.nanmedian(m_arr, axis=0)     # shape: (n_comp,)

    # ── 初始權重向量 ──────────────────────────────────────────────────────────
    w = np.array(
        [float(initial_weights.get(sid, 1.0)) for sid in star_ids],
        dtype=float,
    )
    # 無效（非正、NaN）的初始權重設為中位數，避免被單顆壞星主導
    _w_valid = w[np.isfinite(w) & (w > 0)]
    _w_fallback = float(np.median(_w_valid)) if len(_w_valid) > 0 else 1.0
    w = np.where(np.isfinite(w) & (w > 0), w, _w_fallback)

    delta_old = np.full(len(t_index), np.nan)

    for iteration in range(int(max_iter)):
        # ── 計算各幀漂移 Δ(t) ────────────────────────────────────────────────
        delta_new = np.full(len(t_index), np.nan)
        for row_i in range(len(t_index)):
            row = m_arr[row_i, :]                  # 各比較星在此幀的 m_inst
            valid = np.isfinite(row) & np.isfinite(med_i)
            if int(valid.sum()) < min_comp_stars:
                continue
            delta_i = row[valid] - med_i[valid]    # δ_i(t) = m_i(t) − m̄_i
            w_v = w[valid]
            delta_new[row_i] = float(np.sum(w_v * delta_i) / np.sum(w_v))

        # ── 收斂判斷 ──────────────────────────────────────────────────────────
        both_finite = np.isfinite(delta_old) & np.isfinite(delta_new)
        if both_finite.any():
            max_diff = float(np.max(np.abs(delta_new[both_finite]
                                           - delta_old[both_finite])))
            if max_diff < convergence_tol:
                _phot_logger.debug(
                    "[ensemble] 第 %d 次迭代收斂（max|ΔΔ|=%.2e < tol=%.2e）。",
                    iteration + 1, max_diff, convergence_tol,
                )
                break

        delta_old = delta_new.copy()

        # ── 更新比較星權重：w_i = 1 / RMS_i²（Broeg 2005 eq. 4）───────────
        for ci, sid in enumerate(star_ids):
            col = m_arr[:, ci]
            finite_rows = np.isfinite(col) & np.isfinite(delta_new)
            if int(finite_rows.sum()) < 2:
                continue
            resid = col[finite_rows] - med_i[ci] - delta_new[finite_rows]
            rms_i = float(np.sqrt(np.mean(resid ** 2)))
            if rms_i > 0:
                w[ci] = 1.0 / rms_i ** 2

    # ── 修正目標星星等 ────────────────────────────────────────────────────────
    df_out = df.copy()
    df_out["delta_ensemble"] = delta_new

    m_var_col = df_out.get("m_var") if "m_var" in df_out.columns else pd.Series(
        np.nan, index=df_out.index
    )
    m_var_norm = np.where(
        np.isfinite(df_out["m_var"].values) & np.isfinite(delta_new),
        df_out["m_var"].values - delta_new,
        np.nan,
    )
    df_out["m_var_norm"] = m_var_norm

    # ── 最終 Δ(t) 統計摘要 ───────────────────────────────────────────────────
    finite_delta = delta_new[np.isfinite(delta_new)]
    if len(finite_delta) > 0:
        _phot_logger.info(
            "[ensemble] 完成。Δ(t) median=%.4f  rms=%.4f  "
            "有效幀數=%d/%d  比較星數=%d",
            float(np.median(finite_delta)),
            float(np.std(finite_delta)),
            int(np.isfinite(delta_new).sum()),
            len(delta_new),
            len(star_ids),
        )
        print(
            f"[ensemble] Δ(t) median={float(np.median(finite_delta)):.4f}  "
            f"rms={float(np.std(finite_delta)):.4f}  "
            f"有效幀={int(np.isfinite(delta_new).sum())}/{len(delta_new)}"
        )
    else:
        _phot_logger.warning("[ensemble] 無有效 Δ(t)，m_var_norm 全為 NaN。")

    delta_series = pd.Series(delta_new, index=t_index, name="delta_ensemble")
    return df_out, delta_series


def run_photometry_on_wcs_dir(
    wcs_dir: Path,
    out_csv: Path,
    out_png: Path,
    comp_refs: list,
    check_star=None,
    ap_radius: "float | None" = None,
    channel: str = "B",
) -> "tuple[pd.DataFrame, dict[str, pd.Series]]":
    """
    Per-frame aperture differential photometry.

    Time system : BJD_TDB (Eastman et al., 2010) at exposure midpoint.
    Zero point  : robust iterative linear regression (see robust_zero_point).
    Airmass     : Young (1994); frames with X > 2.0 are flagged but kept.

    Returns
    -------
    df               : 每幀測光結果 DataFrame（含 m_var、m_var_norm 等欄位）。
    comp_lightcurves : {star_id: Series(bjd_tdb → m_inst)}，每顆比較星的
                       儀器星等時間序列。僅在 cfg.ensemble_normalize=True 且
                       比較星數 ≥ cfg.ensemble_min_comp 時填充；否則回傳空 dict。
                       star_id 格式："{ra_deg:.6f}_{dec_deg:.6f}"。
    """
    ra_t, dec_t = cfg.target_radec_deg
    if ap_radius is None:
        ap_radius = cfg.aperture_radius
    r_in, r_out = compute_annulus_radii(ap_radius, cfg.annulus_r_in, cfg.annulus_r_out)
    margin = int(np.ceil(r_out + 2))

    # ── Ensemble 正規化設定（從 cfg 讀取，cfg_from_yaml 負責從 yaml 注入）────
    _ensemble_on = bool(getattr(cfg, "ensemble_normalize", False))
    _ensemble_min_comp = int(getattr(cfg, "ensemble_min_comp", 3))
    _ensemble_max_iter = int(getattr(cfg, "ensemble_max_iter", 10))
    _ensemble_tol = float(getattr(cfg, "ensemble_convergence_tol", 1e-4))

    # split/{channel}/ 的 FITS 命名規則：*_{channel}.fits
    wcs_files_sorted = sorted(wcs_dir.glob(f"*_{channel}.fits"))
    if not wcs_files_sorted:
        raise FileNotFoundError(
            f"No split/{channel} FITS found in: {wcs_dir}\n"
            "Check that debayer step completed successfully."
        )

    check_coord  = (SkyCoord(ra=check_star[0] * u.deg, dec=check_star[1] * u.deg)
                    if check_star is not None else None)
    target_coord = SkyCoord(ra=ra_t * u.deg, dec=dec_t * u.deg)
    cfg_checked      = False
    n_skipped        = 0
    n_high_fwhm      = 0
    n_low_sharpness  = 0
    n_low_zp_r2      = 0
    rows = []
    _first_frame_diag_data = None   # (comp_m_cat, comp_m_inst, fit) for diag plot

    # ── comp_lightcurves 累積結構：{star_id: {t_val: m_inst}} ────────────────
    # star_id = "{ra_deg:.6f}_{dec_deg:.6f}"
    # 每幀測光後，把該幀所有有效比較星的 (t, m_inst) 追加進去。
    # 幀迴圈結束後轉為 pd.Series 供 ensemble_normalize 使用。
    _comp_lc_buf: "dict[str, dict]" = {}
    # 對應 star_id → 初始權重（來自幀迴圈中計算的距離＋誤差複合權重）
    # 只取第一次出現的值（各幀理論上相同，因為距離和 m_err 不隨時間變化）
    _comp_init_weights: "dict[str, float]" = {}

    for f in wcs_files_sorted:
        with fits.open(f) as hdul:
            img = hdul[0].data.astype(np.float32)
            hdr = hdul[0].header
            wcs_obj = WCS(hdr)

        apply_gain_from_header(hdr)
        if not cfg_checked:
            require_cfg_values()
            cfg_checked = True

        # ── Time, BJD_TDB, airmass ───────────────────────────────────────────
        mjd, bjd_tdb, airmass = time_from_header(hdr, ra_t, dec_t)

        xt, yt = radec_to_pixel(wcs_obj, ra_t, dec_t)
        rec = {
            "file": f.name,
            "mjd": mjd,
            "bjd_tdb": bjd_tdb,
            "airmass": airmass,
            "xt": xt, "yt": yt,
            "ok": 0,
            "m_var": np.nan,
            "frame_fwhm_median": np.nan,
            "sharpness": np.nan,
        }

        # ── 高度角截斷（altitude < ALT_MIN_DEG の幀は除外）────────────────────
        # 大氣消光在低高度角時急增，ALT_MIN_DEG=45° 對應 airmass≈1.41。
        # airmass 已知（非 NaN）時才做截斷；location 未設定時不截斷（airmass=NaN）。
        if np.isfinite(airmass) and airmass > cfg.alt_min_airmass:
            n_skipped += 1
            continue

        # ── [1] 幀層級 FWHM 篩選 ─────────────────────────────────────────────
        # 用 DAOStarFinder 估算全幀中位數 FWHM，剔除模糊/拖影幀。
        _frame_fwhm_median = np.nan
        try:
            _f_mean, _f_med, _f_std = sigma_clipped_stats(img, sigma=3.0, maxiters=3)
            if _f_std > 0:
                _dao_fwhm = DAOStarFinder(fwhm=4.0, threshold=5.0 * _f_std)
                _tbl_fwhm = _dao_fwhm(img - _f_med)
                if _tbl_fwhm is not None and len(_tbl_fwhm) > 0:
                    _frame_fwhm_median = float(np.median(_tbl_fwhm["fwhm"]))
        except Exception:
            pass
        rec["frame_fwhm_median"] = _frame_fwhm_median
        if (np.isfinite(_frame_fwhm_median)
                and _frame_fwhm_median > cfg.comp_fwhm_max):
            rec["ok"] = 0
            rec["ok_flag"] = "high_fwhm"
            n_high_fwhm += 1
            rows.append(rec)
            continue

        if not in_bounds(img, xt, yt, margin=margin):
            rows.append(rec)
            continue

        phot_t = aperture_photometry(img, xt, yt, ap_radius, r_in, r_out)
        if phot_t.get("ok") != 1 or not np.isfinite(phot_t.get("flux_net")):
            rows.append(rec)
            continue

        sat_t = is_saturated(phot_t.get("max_pix", np.nan), cfg.saturation_threshold)
        if sat_t and not cfg.allow_saturated_target:
            rows.append(rec)
            continue

        m_inst_t = m_inst_from_flux(phot_t["flux_net"])
        if not np.isfinite(m_inst_t):
            rows.append(rec)
            continue

        # ── [2] Sharpness Index 篩選 ─────────────────────────────────────────
        # S = flux(r=3px) / flux(r=8px)（均扣背景）
        # S < sharpness_min 代表星點過度擴散或拖影，剔除該幀。
        _sharpness = np.nan
        _sharpness_min = float(getattr(cfg, "sharpness_min", 0.3))
        if _sharpness_min > 0:
            try:
                _phot_3 = aperture_photometry(img, xt, yt, 3.0, r_in, r_out)
                _phot_8 = aperture_photometry(img, xt, yt, 8.0, r_in, r_out)
                if (_phot_3.get("ok") == 1 and _phot_8.get("ok") == 1
                        and np.isfinite(_phot_3.get("flux_net", np.nan))
                        and np.isfinite(_phot_8.get("flux_net", np.nan))
                        and _phot_8["flux_net"] > 0):
                    _sharpness = float(_phot_3["flux_net"] / _phot_8["flux_net"])
            except Exception:
                pass
        rec["sharpness"] = _sharpness
        if np.isfinite(_sharpness) and _sharpness < _sharpness_min:
            rec["ok"] = 0
            rec["ok_flag"] = "low_sharpness"
            n_low_sharpness += 1
            rows.append(rec)
            continue

        # ── Comparison ensemble ──────────────────────────────────────────────
        comp_m_inst, comp_m_cat, comp_weights = [], [], []
        for ref in comp_refs:
            ra_c, dec_c, m_cat = ref[0], ref[1], ref[2]
            m_err   = ref[3] if len(ref) > 3 else None
            # weight = 1 / (d + ε)²，由 auto_select_comps 計算並存入 ref[4]
            w_i     = float(ref[4]) if len(ref) > 4 else 1.0
            sc = SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg)
            if check_coord is not None and sc.separation(check_coord).arcsec <= cfg.apass_match_arcsec:
                continue
            xc, yc = radec_to_pixel(wcs_obj, ra_c, dec_c)
            if not in_bounds(img, xc, yc, margin=margin):
                continue
            phot_c = aperture_photometry(img, xc, yc, ap_radius, r_in, r_out)
            if phot_c.get("ok") != 1 or not np.isfinite(phot_c.get("flux_net")):
                continue
            if is_saturated(phot_c.get("max_pix", np.nan), cfg.saturation_threshold):
                continue
            m_inst_c = m_inst_from_flux(phot_c["flux_net"])
            if not np.isfinite(m_inst_c):
                continue
            dist_arcsec   = float(target_coord.separation(sc).arcsec)
            # ε = plate_scale / 2，防止距離趨近於零時權重爆炸
            epsilon       = cfg.plate_scale_arcsec / 2.0
            weight        = 1.0 / (dist_arcsec + epsilon) ** 2
            # 若星表誤差已知，納入誤差加權（誤差越大，權重越低）
            if m_err is not None and np.isfinite(m_err) and float(m_err) > 0:
                weight /= float(m_err) ** 2
            comp_m_inst.append(m_inst_c)
            comp_m_cat.append(float(m_cat))
            comp_weights.append(weight)

            # ── 累積 comp_lightcurves（ensemble 正規化用）────────────────────
            if _ensemble_on:
                _sid = f"{ra_c:.6f}_{dec_c:.6f}"
                if _sid not in _comp_lc_buf:
                    _comp_lc_buf[_sid] = {}
                    _comp_init_weights[_sid] = weight
                _comp_lc_buf[_sid][bjd_tdb if np.isfinite(bjd_tdb) else mjd] = m_inst_c

        comp_m_inst  = np.asarray(comp_m_inst, dtype=float)
        comp_m_cat   = np.asarray(comp_m_cat,  dtype=float)

        if len(comp_m_inst) < cfg.robust_regression_min_points:
            rows.append(rec)
            continue

        fit = robust_zero_point(
            comp_m_inst, comp_m_cat,
            sigma=cfg.robust_regression_sigma,
            max_iter=cfg.robust_regression_max_iter,
            min_points=cfg.robust_regression_min_points,
            weights=np.asarray(comp_weights, dtype=float),
        )

        # 捕捉第一幀資料供診斷圖使用
        if _first_frame_diag_data is None and len(comp_m_cat) >= 2:
            _first_frame_diag_data = (
                comp_m_cat.copy(), comp_m_inst.copy(), fit
            )

        if fit is None:
            zp   = float(np.nanmedian(comp_m_cat - comp_m_inst))
            a, b, r2 = 1.0, -zp, np.nan
            mask = np.isfinite(comp_m_inst) & np.isfinite(comp_m_cat)
        else:
            a, b, r2, mask = fit["a"], fit["b"], fit["r2"], fit["mask"]

        if not np.isfinite(a) or a == 0:
            rows.append(rec)
            continue

        # ── [3] ZP R² 幀層級篩選 ─────────────────────────────────────────────
        # zp_r2_min 預設 0.0（停用）；> 0 時才自動剔除，保留 WARN 輸出。
        _zp_r2_min = float(getattr(cfg, "zp_r2_min", 0.0))
        if np.isfinite(r2) and _zp_r2_min > 0 and r2 < _zp_r2_min:
            _phot_logger.warning(
                "[WARN] low ZP R2=%.4f < %.4f in %s (channel=%s)",
                r2, _zp_r2_min, f.name, channel,
            )
            rec["zp_r2"] = r2
            rec["ok"] = 0
            rec["ok_flag"] = "low_zp_r2"
            n_low_zp_r2 += 1
            rows.append(rec)
            continue
        elif np.isfinite(r2) and r2 < max(_zp_r2_min, 0.5):
            _phot_logger.warning(
                "[WARN] low ZP R2=%.4f in %s (channel=%s) — consider raising zp_r2_min",
                r2, f.name, channel,
            )

        m_var       = (m_inst_t - b) / a
        comp_used   = int(np.count_nonzero(mask))
        sigma_mag   = mag_error_from_flux(
            flux_net=phot_t["flux_net"],
            b_sky_std=phot_t.get("b_sky_std", phot_t["b_sky"]),  # b_sky_std 優先
            n_pix=phot_t["n_pix"],
            gain_e_per_adu=cfg.gain_e_per_adu,
            read_noise_e=cfg.read_noise_e,
            n_sky=phot_t.get("n_sky"),
        )
        snr = float(1.0857 / sigma_mag) if np.isfinite(sigma_mag) and sigma_mag > 0 else np.nan

        # ── FLAG_SLOPE_DEVIATION（§3.8）：|a − 1.0| > 0.05 ──────────────────
        _slope_flag = int(abs(a - 1.0) > 0.05)
        if _slope_flag:
            _phot_logger.debug(
                "[FLAG] slope_deviation a=%.4f (|a-1|=%.4f > 0.05) in %s",
                a, abs(a - 1.0), f.name,
            )

        rec.update({
            "ok": 1,
            "t_flux_net": phot_t["flux_net"],
            "t_b_sky": phot_t["b_sky"],
            "t_n_pix": phot_t["n_pix"],
            "t_max_pix": phot_t["max_pix"],
            "t_saturated": int(sat_t),
            "t_m_inst": m_inst_t,
            "t_sigma_mag": sigma_mag,
            "t_snr": snr,
            "ap_radius": ap_radius,
            "annulus_r_in": r_in,
            "annulus_r_out": r_out,
            "zp_slope": a,
            "zp_intercept": b,
            "zp_r2": r2,
            "comp_used": comp_used,
            "flag_slope_dev": _slope_flag,
            "m_var": m_var,
        })

        # ── 零點殘差 RMS（用於診斷圖）────────────────────────────────────────
        _m_cat_fit  = comp_m_cat[mask]
        _m_inst_fit = comp_m_inst[mask]
        _m_cat_pred = a * _m_inst_fit + b   # 預測 m_cat
        _residuals  = _m_cat_fit - _m_cat_pred
        zp_resid_rms = float(np.sqrt(np.mean(_residuals ** 2))) if len(_residuals) > 0 else np.nan
        rec["zp_residual_rms"] = zp_resid_rms

        # ── Check star ───────────────────────────────────────────────────────
        if check_star is not None:
            ra_k, dec_k, m_cat_k = check_star
            xk, yk = radec_to_pixel(wcs_obj, ra_k, dec_k)
            if in_bounds(img, xk, yk, margin=margin):
                phot_k = aperture_photometry(img, xk, yk, ap_radius, r_in, r_out)
                if phot_k.get("ok") == 1 and np.isfinite(phot_k.get("flux_net")):
                    sat_k = is_saturated(phot_k.get("max_pix", np.nan), cfg.saturation_threshold)
                    if not sat_k or cfg.allow_saturated_check:
                        m_inst_k = m_inst_from_flux(phot_k["flux_net"])
                        if np.isfinite(m_inst_k):
                            m_check = (m_inst_k - b) / a
                            rec["k_m_inst"] = m_inst_k
                            rec["k_m_var"]  = m_check
                            rec["k_m_cat"]  = m_cat_k
                            if m_cat_k is not None and np.isfinite(m_cat_k):
                                rec["k_minus_c"] = m_check - float(m_cat_k)
        rows.append(rec)

    df = pd.DataFrame(rows)

    if len(df) == 0:
        print("[WARN] 所有幀均被跳過（airmass），無任何資料列。")
        df = pd.DataFrame(columns=["file", "mjd", "bjd_tdb", "airmass", "ok", "m_var",
                                    "m_var_norm", "delta_ensemble"])
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        return df, {}

    # ── Choose time axis: prefer BJD_TDB ─────────────────────────────────────
    if "bjd_tdb" in df.columns and np.isfinite(df["bjd_tdb"]).any():
        time_key = "bjd_tdb"
    elif "mjd" in df.columns:
        time_key = "mjd"
        print("[WARN] BJD_TDB not available; falling back to MJD. "
              "Check obs_lat_deg / obs_lon_deg in Cfg.")
    else:
        time_key = "bjd_tdb"   # 保險 fallback，sort 會忽略 NaN

    df = df.sort_values(time_key, na_position="last")

    # ── Sigma-clip m_var：排除零點崩潰造成的極端離群值 ──────────────────────
    # 對 ok==1 且 m_var 有限的子集，計算 median 和 MAD（robust σ = 1.4826 × MAD）。
    # |m_var − median| > 3σ 的幀，ok 改為 0，記錄 ok_flag = "sigma_clip"。
    # CSV 保留所有列，畫圖/週期分析只用 ok==1。
    _ok_mask = (df["ok"] == 1) & np.isfinite(df["m_var"])
    _n_before_clip = int(_ok_mask.sum())
    if _n_before_clip >= 5:
        _m = df.loc[_ok_mask, "m_var"].values
        _med = float(np.median(_m))
        _mad = float(np.median(np.abs(_m - _med)))
        _robust_sigma = 1.4826 * _mad if _mad > 0 else 1e-9
        _clip_mask = np.abs(_m - _med) > 3.0 * _robust_sigma
        _clip_idx = df.index[_ok_mask][_clip_mask]
        if len(_clip_idx) > 0:
            df.loc[_clip_idx, "ok"] = 0
            if "ok_flag" not in df.columns:
                df["ok_flag"] = ""
            df.loc[_clip_idx, "ok_flag"] = "sigma_clip"
            print(f"[sigma-clip] median={_med:.4f}  robust_sigma={_robust_sigma:.4f}  "
                  f"clipped {len(_clip_idx)} frames "
                  f"(|m_var - median| > 3sigma = {3.0 * _robust_sigma:.4f})")
    _n_after_clip = int((df["ok"] == 1).sum())

    # ── 剔除統計表 ────────────────────────────────────────────────────────────
    _n_total_fits  = len(wcs_files_sorted)
    _n_in_df       = len(df)
    _n_ok_final    = int((df["ok"] == 1).sum())
    _n_alt_skip    = n_skipped  # alt_too_low（未進 df）
    _n_sigma_clip  = int(((df["ok"] == 0) & (df.get("ok_flag", pd.Series("", index=df.index)) == "sigma_clip")).sum()) if "ok_flag" in df.columns else (_n_before_clip - _n_after_clip)

    _n_high_fwhm_val    = n_high_fwhm
    _n_low_sharpness_val = n_low_sharpness
    _n_low_zp_r2_val    = n_low_zp_r2
    # 重新計算 _n_phot_fail：排除三層篩選計數
    _n_qual_filtered = _n_high_fwhm_val + _n_low_sharpness_val + _n_low_zp_r2_val
    _n_phot_fail   = _n_in_df - _n_ok_final - _n_sigma_clip - _n_qual_filtered

    _sep = "-" * 68
    print(f"\n[剔除統計] 通道 {channel}  {getattr(cfg, 'target_name', '')}  {str(out_csv.stem).split('_')[-1]}")
    print(_sep)
    print(f"  {'原因':<24} {'幀數':>6}    公式")
    print(_sep)
    print(f"  {'高氣團跳過':<24} {_n_alt_skip:>6}    airmass > {cfg.alt_min_airmass:.2f}")
    print(f"  {'高 FWHM 幀剔除':<24} {_n_high_fwhm_val:>6}    frame_fwhm_median > {cfg.comp_fwhm_max:.1f} px")
    print(f"  {'低 Sharpness 剔除':<24} {_n_low_sharpness_val:>6}    S=flux(r=3)/flux(r=8) < {float(getattr(cfg,'sharpness_min',0.3)):.2f}")
    print(f"  {'低 ZP R2 剔除':<24} {_n_low_zp_r2_val:>6}    zp_r2 < {float(getattr(cfg,'zp_r2_min',0.0)):.2f} (0=停用)")
    print(f"  {'孔徑/WCS/邊界失敗':<24} {_n_phot_fail:>6}    flux/位置無效")
    print(f"  {'sigma_clip':<24} {_n_sigma_clip:>6}    |m_var - median| > 3 * 1.4826 * MAD")
    print(_sep)
    print(f"  {'保留 (ok=1)':<24} {_n_ok_final:>6} / {_n_total_fits} 幀")
    print()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    n_written = len(df)
    print(f"[CSV] saved → {out_csv}  "
          f"({n_written} rows written, {_n_after_clip} successful "
          f"[{_n_before_clip - _n_after_clip} sigma-clipped], "
          f"{n_skipped} skipped [alt < {cfg.alt_min_deg:.0f}°])")

    # ── 零點診斷總覽圖：殘差時序 + 第一幀散佈圖 ─────────────────────────────
    if cfg.save_zeropoint_diagnostic:
        try:
            _zp_diag_path = cfg.zeropoint_diag_dir / (
                f"zp_overview_{out_csv.stem.split('_')[1]}_{out_csv.stem.split('_', 1)[-1]}.png"
            )
            _df_ok = df[df["ok"] == 1].copy()
            fig_diag, axes_diag = plt.subplots(
                1, 2, figsize=(14, 5),
                gridspec_kw={"width_ratios": [2, 1]}
            )

            # 左：殘差時序圖
            ax_ts = axes_diag[0]
            if "zp_residual_rms" in _df_ok.columns and np.isfinite(_df_ok["zp_residual_rms"]).any():
                ax_ts.plot(
                    _df_ok[time_key], _df_ok["zp_residual_rms"],
                    "o-", ms=3, lw=0.8, color="steelblue", alpha=0.7
                )
                ax_ts.axhline(
                    _df_ok["zp_residual_rms"].median(), color="red",
                    lw=1, ls="--", label=f"median={_df_ok['zp_residual_rms'].median():.4f}"
                )
                ax_ts.set_xlabel(time_key.upper())
                ax_ts.set_ylabel("Zero-point residual RMS (mag)")
                ax_ts.set_title("Zero-point residual RMS vs. time")
                ax_ts.legend(fontsize=8)
                ax_ts.grid(True, alpha=0.3)
            else:
                ax_ts.text(0.5, 0.5, "No residual data", transform=ax_ts.transAxes,
                           ha="center", va="center")

            # 右：第一幀回歸散佈圖（從 _first_frame_diag_data 取得，若有）
            ax_sc = axes_diag[1]
            _ffd = _first_frame_diag_data
            if _ffd is not None:
                _mc, _mi, _fit = _ffd
                _ok_pts = np.isfinite(_mc) & np.isfinite(_mi)
                ax_sc.scatter(_mc[_ok_pts], _mi[_ok_pts], s=12, alpha=0.6,
                              color="steelblue", label=f"n={_ok_pts.sum()}")
                if _fit and np.isfinite(_fit.get("a", np.nan)):
                    _xl = np.linspace(_mc[_ok_pts].min(), _mc[_ok_pts].max(), 100)
                    ax_sc.plot(_xl, _fit["a"] * _xl + _fit["b"], "r-", lw=1.5,
                               label=f"a={_fit['a']:.3f} b={_fit['b']:.3f} R2={_fit['r2']:.3f}")
                _xr = np.array([_mc[_ok_pts].min(), _mc[_ok_pts].max()])
                ax_sc.plot(_xr, _xr, "k--", lw=0.8, alpha=0.4)
                ax_sc.invert_yaxis()
                ax_sc.set_xlabel(f"$m_{{cat}}$ ({cfg.phot_band})")
                ax_sc.set_ylabel("$m_{inst}$")
                ax_sc.set_title("Frame 1 zero-point scatter")
                ax_sc.legend(fontsize=7)
                ax_sc.grid(True, alpha=0.3)
            else:
                ax_sc.text(0.5, 0.5, "No first-frame data",
                           transform=ax_sc.transAxes, ha="center", va="center")

            fig_diag.suptitle(
                f"Zero-point diagnostics  |  {cfg.target_name}  "
                f"channel={cfg.phot_band}  {out_csv.stem}",
                fontsize=10
            )
            fig_diag.tight_layout()
            fig_diag.savefig(_zp_diag_path, dpi=120)
            plt.close(fig_diag)
            print(f"[診斷圖] saved → {_zp_diag_path}")
        except Exception as _e:
            print(f"[WARN] 零點診斷圖輸出失敗：{_e}")

    # ── Plot light curve ──────────────────────────────────────────────────────
    d = df[(df["ok"] == 1) & np.isfinite(df[time_key]) & np.isfinite(df["m_var"])].copy()
    if len(d) == 0:
        print("[PLOT] No valid photometry points to plot.")
        df["m_var_norm"] = df["m_var"]
        df["delta_ensemble"] = np.nan
        return df, {}

    import matplotlib.ticker as mticker
    from astropy.time import Time as ATime

    # 從 csv 檔名取觀測日期
    _lc_obs_date = str(out_csv.stem).split("_")[-1]

    bjd_arr = d[time_key].values
    bjd_min = float(bjd_arr.min())
    bjd_max = float(bjd_arr.max())

    # 本地時間（UTC+8）換算：BJD_TDB ≈ BJD_UTC + 67.2s（微小），近似用 UTC+8
    _tz_offset_h = float(getattr(cfg, "tz_offset_hours", 8))

    def _bjd_to_local_hm(bjd_val):
        """BJD → 本地時間 HH:MM（近似，UTC+8）"""
        try:
            t_utc = ATime(bjd_val, format="jd", scale="tdb").to_datetime()
            import datetime
            t_local = t_utc + datetime.timedelta(hours=_tz_offset_h)
            return t_local
        except Exception:
            return None

    # 計算 Local Time 整點刻度位置（BJD）
    _t_local_min = _bjd_to_local_hm(bjd_min)
    _t_local_max = _bjd_to_local_hm(bjd_max)

    _major_bjd_ticks = []
    _major_labels = []
    _minor_bjd_ticks = []
    if _t_local_min is not None and _t_local_max is not None:
        import datetime
        # 整點刻度
        _cur = _t_local_min.replace(minute=0, second=0, microsecond=0)
        if _cur < _t_local_min:
            _cur += datetime.timedelta(hours=1)
        while _cur <= _t_local_max + datetime.timedelta(minutes=1):
            # 反算回 BJD
            _utc = _cur - datetime.timedelta(hours=_tz_offset_h)
            _bjd_tick = ATime(_utc).jd
            _major_bjd_ticks.append(_bjd_tick)
            _major_labels.append(_cur.strftime("%H:%M"))
            _cur += datetime.timedelta(hours=1)
        # 每 10 分鐘小刻度
        _cur = _t_local_min.replace(second=0, microsecond=0)
        _min_round = (_cur.minute // 10) * 10
        _cur = _cur.replace(minute=_min_round)
        while _cur <= _t_local_max + datetime.timedelta(minutes=1):
            _utc = _cur - datetime.timedelta(hours=_tz_offset_h)
            _bjd_tick = ATime(_utc).jd
            _minor_bjd_ticks.append(_bjd_tick)
            _cur += datetime.timedelta(minutes=10)

    fig, ax = plt.subplots(figsize=(12, 4))

    if "t_sigma_mag" in d.columns and np.isfinite(d["t_sigma_mag"]).any():
        ax.errorbar(bjd_arr, d["m_var"].values, yerr=d["t_sigma_mag"].values,
                    fmt="o", ms=4, capsize=2, lw=0.8, label="target", zorder=3)
    else:
        ax.plot(bjd_arr, d["m_var"].values, "o-", ms=4, lw=0.8, label="target", zorder=3)

    ax.invert_yaxis()
    ax.set_xlim(bjd_min - 0.002, bjd_max + 0.002)

    # 主 x 軸（下方）：本地時間 HH:MM，黑色，刻度朝外（預設）
    if _major_bjd_ticks:
        ax.set_xticks(_major_bjd_ticks)
        ax.set_xticklabels(_major_labels, color="black", fontsize=9)
    if _minor_bjd_ticks:
        ax.set_xticks(_minor_bjd_ticks, minor=True)
    ax.tick_params(axis="x", which="major", direction="out", colors="black", length=6)
    ax.tick_params(axis="x", which="minor", direction="out", colors="black", length=3)

    # 副 x 軸（上方）：BJD，淺藍色，刻度朝內
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    # 使用與主軸相同的整點刻度位置，但顯示 BJD 值
    if _major_bjd_ticks:
        ax2.set_xticks(_major_bjd_ticks)
        _bjd_labels = [f"{v:.4f}" for v in _major_bjd_ticks]
        ax2.set_xticklabels(_bjd_labels, color="lightblue", fontsize=7, rotation=30, ha="left")
    ax2.tick_params(axis="x", which="major", direction="in",
                    colors="lightblue", length=8)

    # 軸標題：黃色，靠左
    ax.set_xlabel("Local Time (HH:MM)", color="yellow", loc="left", fontsize=9)
    ax2.set_xlabel("BJD", color="yellow", loc="left", fontsize=9)

    ax.set_ylabel("Differential magnitude (mag)")

    # 標題：星名 通道 日期
    _lc_tname = getattr(cfg, "target_name", "")
    _lc_title = f"{_lc_tname}  [{channel}]  {_lc_obs_date}"
    ax.set_title(_lc_title)

    # 右上角：觀測站座標
    _lat = getattr(cfg, "obs_lat_deg", None)
    _lon = getattr(cfg, "obs_lon_deg", None)
    if _lat is not None and _lon is not None:
        ax.text(0.99, 0.97, f"{_lat:.2f}  {_lon:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="gray")

    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close("all")
    print(f"[PNG] saved → {out_png}")

    # ── Check star residuals ──────────────────────────────────────────────────
    if check_star is not None and "k_minus_c" in df.columns:
        k = df[(df["ok"] == 1) & np.isfinite(df[time_key])
               & np.isfinite(df["k_minus_c"])].copy()
        if len(k) > 1:
            k["t_rel_d"] = k[time_key] - t0
            sigma_k = float(np.nanstd(k["k_minus_c"]))
            flag    = " [!] EXCEEDS THRESHOLD" if sigma_k > cfg.check_star_max_sigma else ""
            print(f"[CHECK] K-C sigma = {sigma_k:.5f} mag "
                  f"(threshold = {cfg.check_star_max_sigma}){flag}")
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.plot(k["t_rel_d"], k["k_minus_c"], "o-", ms=4, lw=0.8)
            ax2.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_dhm))
            ax2.invert_yaxis()
            ax2.set_xlabel(xlabel)
            ax2.set_ylabel("K - C  (mag)")
            ax2.set_title("Check Star Validation")
            ax2.grid(True, alpha=0.4)
            fig2.tight_layout()
            plt.close("all")

    # ── Broeg (2005) ensemble 正規化 ─────────────────────────────────────────
    # 條件：ensemble_normalize: true 且累積到的比較星數 ≥ ensemble_min_comp。
    # 結果欄位：m_var_norm（修正後）、delta_ensemble（Δ(t) 漂移量）。
    # ensemble 停用或條件不足時，m_var_norm = m_var（直通，不影響現有欄位）。
    comp_lightcurves: "dict[str, pd.Series]" = {}
    if _ensemble_on:
        # _comp_lc_buf → pd.Series，index = time 值
        for _sid, _tdict in _comp_lc_buf.items():
            if len(_tdict) >= 2:   # 至少 2 幀才有意義
                comp_lightcurves[_sid] = pd.Series(_tdict, dtype=float)

        _n_comp_lc = len(comp_lightcurves)
        print(f"[ensemble] 比較星光變曲線：{_n_comp_lc} 顆")

        if _n_comp_lc >= _ensemble_min_comp:
            df, _delta_series = ensemble_normalize(
                df,
                comp_lightcurves=comp_lightcurves,
                initial_weights=_comp_init_weights,
                time_key=time_key,
                min_comp_stars=_ensemble_min_comp,
                max_iter=_ensemble_max_iter,
                convergence_tol=_ensemble_tol,
            )
            # ── CSV 更新（已含 m_var_norm、delta_ensemble）────────────────────
            df.to_csv(out_csv, index=False, encoding="utf-8-sig")
            print(f"[CSV] ensemble 正規化後重新寫入 → {out_csv}")
        else:
            _phot_logger.warning(
                "[ensemble] 比較星光變曲線數量 %d < min_comp_stars %d，"
                "跳過 ensemble 正規化。",
                _n_comp_lc, _ensemble_min_comp,
            )
            df["m_var_norm"] = df["m_var"]
            df["delta_ensemble"] = np.nan
    else:
        # ensemble 停用：m_var_norm 直通 m_var
        df["m_var_norm"] = df["m_var"]
        df["delta_ensemble"] = np.nan

    return df, comp_lightcurves


"""## Lomb-Scargle Period Analysis + Fourier Fit"""

# ── Lomb-Scargle period analysis + Fourier fit ───────────────────────────────
# References:
#   Lomb (1976) Ap&SS 39, 447
#   Scargle (1982) ApJ 263, 835
#   VanderPlas (2018) ApJS 236, 16
#   Press & Rybicki (1989) ApJ 338, 277  (fast implementation)

from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


# ── Config (edit here or in observation_config.yaml) ─────────────────────────
LS_MIN_PERIOD_DAYS  = 0.05     # shortest period to search (days)
LS_MAX_PERIOD_DAYS  = 2.0      # longest  period to search (days)
LS_SAMPLES_PER_PEAK = 10       # frequency grid oversampling
FAP_THRESHOLD       = 0.001    # false-alarm probability threshold
FOURIER_N_MAX       = 6        # maximum Fourier harmonics (auto-selected below)
PHASE_FOLD_CYCLES   = 2        # how many phase cycles to show in the fold plot


def run_lomb_scargle(
    df: pd.DataFrame,
    time_col: str = "bjd_tdb",
    mag_col: str = "m_var",
    err_col: str = "t_sigma_mag",
    min_period: float = LS_MIN_PERIOD_DAYS,
    max_period: float = LS_MAX_PERIOD_DAYS,
    samples_per_peak: int = LS_SAMPLES_PER_PEAK,
    fap_threshold: float = FAP_THRESHOLD,
    out_png: "Path | None" = None,
    target_name: str = "",
    channel: str = "",
    obs_date: str = "",
    lat_deg: "float | None" = None,
    lon_deg: "float | None" = None,
) -> dict:
    """
    Run a Lomb-Scargle periodogram on the light-curve DataFrame.

    Returns a dict with:
        frequency   : np.ndarray  (1/day)
        power       : np.ndarray
        best_period : float  (days)
        best_freq   : float  (1/day)
        fap         : float  false-alarm probability at best peak
        ls          : LombScargle object (for further queries)
        t, mag, err : cleaned arrays used in the fit
    """
    d = df[(df["ok"] == 1) & np.isfinite(df[time_col]) & np.isfinite(df[mag_col])].copy()
    if len(d) < 10:
        raise ValueError(f"Only {len(d)} valid points -- need >= 10 for period search.")

    t   = d[time_col].values
    mag = d[mag_col].values
    err = d[err_col].values if (err_col in d.columns and np.isfinite(d[err_col]).any()) else None

    ls = LombScargle(t, mag, dy=err)

    frequency, power = ls.autopower(
        minimum_frequency=1.0 / max_period,
        maximum_frequency=1.0 / min_period,
        samples_per_peak=samples_per_peak,
    )

    best_idx    = int(np.argmax(power))
    best_freq   = float(frequency[best_idx])
    best_period = 1.0 / best_freq
    fap         = float(ls.false_alarm_probability(power[best_freq == frequency].max()
                                                   if power[best_freq == frequency].size
                                                   else power[best_idx]))

    print(f"[LS] Best period = {best_period:.6f} d  ({best_period * 24:.4f} h)")
    print(f"[LS] Best freq   = {best_freq:.6f} d^-1")
    print(f"[LS] FAP         = {fap:.2e}")
    if fap > fap_threshold:
        print(f"[WARN] FAP={fap:.2e} > threshold={fap_threshold}. "
              "Period detection may not be significant.")

    # ── Plot periodogram ──────────────────────────────────────────────────────
    # ── X 軸單位：天 → 小時，刻度格式化為 H:MM:SS ────────────────────────────
    period_h = 1.0 / frequency * 24.0   # 天 → 小時

    def _fmt_hms(h_val, _pos=None):
        """將小時數格式化為 H:MM:SS，去掉前置零。"""
        total_s = int(round(abs(h_val) * 3600))
        hh = total_s // 3600
        mm = (total_s % 3600) // 60
        ss = total_s % 60
        if hh > 0:
            return f"{hh}:{mm:02d}:{ss:02d}"
        return f"{mm}:{ss:02d}"

    best_period_h = best_period * 24.0
    _p_min_h = float(period_h.min())
    _p_max_h = float(period_h.max())

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(period_h, power, lw=0.8)
    ax.axvline(best_period_h, color="red", lw=1.2, ls="--",
               label=f"P = {_fmt_hms(best_period_h)}")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_hms))
    ax.set_xlabel("Period (H:MM:SS)")
    ax.set_ylabel("Lomb-Scargle power")
    _ls_title = f"{target_name}  [{channel}]  {obs_date}" if target_name else "Lomb-Scargle Periodogram"
    ax.set_title(_ls_title + "\nLomb-Scargle Periodogram")
    ax.set_xlim(_p_min_h, _p_max_h)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.4)
    # 右上角：觀測站座標
    if lat_deg is not None and lon_deg is not None:
        ax.text(0.99, 0.97, f"{lat_deg:.2f}  {lon_deg:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="gray")
    fig.tight_layout()
    if out_png is not None:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150)
        print(f"[LS] periodogram saved -> {out_png}")
    plt.close("all")

    return {
        "frequency": frequency,
        "power": power,
        "best_period": best_period,
        "best_freq": best_freq,
        "fap": fap,
        "ls": ls,
        "t": t,
        "mag": mag,
        "err": err,
    }


def _auto_fourier_order(n_cycles: float) -> int:
    """
    Auto-select Fourier harmonic order from the number of observed cycles.
    Fewer cycles → fewer free parameters to avoid over-fitting.
    """
    if n_cycles < 2:
        return 1
    elif n_cycles < 5:
        return 3
    else:
        return min(6, FOURIER_N_MAX)


def _fourier_model(phase: np.ndarray, *params) -> np.ndarray:
    """
    Fourier series: V(φ) = a0 + Σ_n [a_n cos(2πnφ) + b_n sin(2πnφ)]
    params = [a0, a1, b1, a2, b2, ...]
    """
    a0     = params[0]
    result = np.full_like(phase, a0)
    n_harm = (len(params) - 1) // 2
    for i in range(n_harm):
        a_n = params[1 + 2 * i]
        b_n = params[2 + 2 * i]
        result += a_n * np.cos(2 * np.pi * (i + 1) * phase)
        result += b_n * np.sin(2 * np.pi * (i + 1) * phase)
    return result


def run_fourier_fit(
    t: np.ndarray,
    mag: np.ndarray,
    period: float,
    err: "np.ndarray | None" = None,
    n_max: "int | None" = None,
    t0: "float | None" = None,
    out_png: "Path | None" = None,
    target_name: str = "",
    channel: str = "",
    obs_date: str = "",
    lat_deg: "float | None" = None,
    lon_deg: "float | None" = None,
) -> dict:
    """
    Phase-fold the light curve and fit a Fourier series.

    Parameters
    ----------
    t       : time array (BJD_TDB days)
    mag     : magnitude array
    period  : best period in days (from Lomb-Scargle)
    err     : magnitude uncertainties (optional)
    n_max   : Fourier harmonic order; None → auto-selected
    t0      : epoch of phase zero; None → uses t[0]

    Returns
    -------
    dict with keys:
        phase, mag_sorted, fit_phase, fit_mag  – for plotting
        amplitude                              – peak-to-peak amplitude (mag)
        n_harmonics                            – order used
        r2                                     – R² of the Fourier fit
        params                                 – fitted coefficients
        residuals_rms                          – rms of residuals (mag)
    """
    if t0 is None:
        t0 = float(t[0])

    phase = ((t - t0) / period) % 1.0

    n_cycles = (t.max() - t.min()) / period
    if n_max is None:
        n_max = _auto_fourier_order(n_cycles)
    print(f"[Fourier] n_cycles ~= {n_cycles:.1f}  ->  using {n_max} harmonics")

    # Initial guess: all coefficients = 0, constant = mean magnitude
    p0 = [float(np.nanmean(mag))] + [0.0] * (2 * n_max)
    sigma = err if (err is not None and np.isfinite(err).all()) else None

    try:
        popt, pcov = curve_fit(
            _fourier_model, phase, mag,
            p0=p0, sigma=sigma, absolute_sigma=True,
            maxfev=50000,
        )
    except Exception as exc:
        print(f"[WARN] Fourier fit failed: {exc}")
        return {}

    # ── R² and residuals ──────────────────────────────────────────────────────
    mag_pred   = _fourier_model(phase, *popt)
    ss_res     = float(np.sum((mag - mag_pred) ** 2))
    ss_tot     = float(np.sum((mag - mag.mean()) ** 2))
    r2         = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rms_resid  = float(np.sqrt(np.mean((mag - mag_pred) ** 2)))

    # ── Amplitude from smooth model ───────────────────────────────────────────
    phi_dense  = np.linspace(0.0, 1.0, 1000)
    fit_dense  = _fourier_model(phi_dense, *popt)
    amplitude  = float(fit_dense.max() - fit_dense.min())

    print(f"[Fourier] Amplitude = {amplitude:.4f} mag")
    print(f"[Fourier] R2        = {r2:.4f}")
    print(f"[Fourier] RMS resid = {rms_resid:.4f} mag")

    # ── Phase-folded plot ─────────────────────────────────────────────────────
    sort_idx   = np.argsort(phase)
    phase_ext  = np.concatenate([phase[sort_idx], phase[sort_idx] + 1.0])
    mag_ext    = np.concatenate([mag[sort_idx], mag[sort_idx]])
    fit_ext    = np.concatenate([fit_dense, fit_dense])
    phi_ext    = np.concatenate([phi_dense, phi_dense + 1.0])

    n_show  = int(PHASE_FOLD_CYCLES)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(phase_ext[:len(phase_ext) // PHASE_FOLD_CYCLES * n_show],
               mag_ext[:len(mag_ext) // PHASE_FOLD_CYCLES * n_show],
               s=12, alpha=0.7, label="data")
    ax.plot(phi_ext[:len(phi_ext) // PHASE_FOLD_CYCLES * n_show],
            fit_ext[:len(fit_ext) // PHASE_FOLD_CYCLES * n_show],
            "r-", lw=1.5, label=f"Fourier n={n_max}  R2={r2:.3f}")
    ax.invert_yaxis()
    ax.set_xlabel(f"Phase  (P = {period:.6f} d = {period * 24:.4f} h)",
                  loc="center")
    ax.set_ylabel("Differential magnitude")
    _ff_title = f"{target_name}  [{channel}]  {obs_date}" if target_name else "Phase-Folded Light Curve"
    ax.set_title(_ff_title + "\nPhase-Folded Light Curve + Fourier Fit")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.4)
    # 右上角：觀測站座標
    if lat_deg is not None and lon_deg is not None:
        ax.text(0.99, 0.97, f"{lat_deg:.2f}  {lon_deg:.2f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="gray")
    fig.tight_layout()
    if out_png is not None:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150)
        print(f"[Fourier] phase-fold plot saved -> {out_png}")
    plt.close("all")

    return {
        "phase": phase,
        "mag_sorted": mag[sort_idx],
        "fit_phase": phi_dense,
        "fit_mag": fit_dense,
        "amplitude": amplitude,
        "n_harmonics": n_max,
        "r2": r2,
        "residuals_rms": rms_resid,
        "params": popt,
        "period": period,
    }


# ── 執行 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(description="差分測光管線 — 步驟 4")
    _parser.add_argument("--target", default=None,
                         help="目標星（例如 V1162Ori）")
    _parser.add_argument("--date", default=None,
                         help="觀測日期（例如 20251220）")
    _parser.add_argument("--channels", default=None, nargs="+",
                         help="通道列表（例如 --channels R G1 B）")
    _parser.add_argument("--split_subdir", default="split",
                         help="split 子目錄名稱（例如 splitNINA）")
    _parser.add_argument("--all", action="store_true",
                         help="處理 yaml 中所有 obs_sessions 的全部目標")
    _args = _parser.parse_args()

    # ── Logger 初始化 ──────────────────────────────────────────────────────────
    _log_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    _logger   = logging.getLogger("photometry")
    _logger.setLevel(logging.DEBUG)
    # Console：只印重要摘要訊息
    _con_hdl  = logging.StreamHandler(sys.stdout)
    _con_hdl.setLevel(logging.INFO)
    _KEEP_PREFIXES = (
        "[sigma-clip]", "[CSV]", "[完成]", "[LS]", "[Fourier]",
        "[比較星]", "[生長曲線]", "[SKIP]", "孔徑",
        "======", "  通道", "  FITS", "  輸出", "所有通道",
        "[photometry]", "找到",
    )
    class _SummaryFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return (
                record.levelno >= logging.ERROR
                or any(msg.startswith(p) for p in _KEEP_PREFIXES)
            )
    _con_hdl.addFilter(_SummaryFilter())
    _con_hdl.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_con_hdl)
    # FITSFixedWarning → logger DEBUG（只進 log 檔）
    warnings.showwarning = lambda msg, cat, fn, ln, f=None, li=None: \
        _logger.debug("[astropy] %s", str(msg))
    # log 檔 handler 在取得 out_dir 後才加入（下方）

    _yaml = load_pipeline_config()

    # ── 通道清單（所有目標共用）──────────────────────────────────────────────
    if _args.channels:
        CHANNELS = [str(ch).upper() for ch in _args.channels]
        if "G1" in CHANNELS and "G2" not in CHANNELS:
            _g1_idx = CHANNELS.index("G1")
            CHANNELS.insert(_g1_idx + 1, "G2")
    else:
        _ch_raw = _yaml.get("photometry", {}).get(
            "channels", _yaml.get("photometry", {}).get("channel", ["B"])
        )
        if isinstance(_ch_raw, str):
            _ch_raw = [_ch_raw]
        CHANNELS = [str(ch).upper() for ch in _ch_raw]
        # G2 自動補入（G1 存在但 G2 未列時）
        if "G1" in CHANNELS and "G2" not in CHANNELS:
            _g1_idx = CHANNELS.index("G1")
            CHANNELS.insert(_g1_idx + 1, "G2")

    # ── 建立要處理的 (target, date) 列表 ─────────────────────────────────────
    if _args.all or (_args.target is None and _args.date is None):
        _targets_list = [
            (str(_t), str(_sess["date"]))
            for _sess in _yaml.get("obs_sessions", [])
            for _t in _sess.get("targets", [])
        ]
        if not _targets_list:
            _targets_list = [("V1162Ori", "20251220")]
    else:
        _targets_list = [(_args.target or "V1162Ori", _args.date or "20251220")]

    print(f"[photometry] 待處理目標：{_targets_list}  通道：{CHANNELS}")

    # ── 各目標迴圈 ────────────────────────────────────────────────────────────
    for (ACTIVE_TARGET, ACTIVE_DATE) in _targets_list:
        print(f"\n{'#'*60}")
        print(f"# 目標：{ACTIVE_TARGET}  日期：{ACTIVE_DATE}  通道：{CHANNELS}")
        print(f"{'#'*60}")

        # 關閉前一目標的 log 檔 handler
        for _h in list(_logger.handlers):
            if isinstance(_h, logging.FileHandler):
                _h.close()
                _logger.removeHandler(_h)

        # ── 第一通道 cfg（用於比較星選取和孔徑估計）──────────────────────────
        try:
            cfg = cfg_from_yaml(_yaml, ACTIVE_TARGET, ACTIVE_DATE, channel=CHANNELS[0],
                                split_subdir=_args.split_subdir)
        except Exception as _e_cfg:
            print(f"[SKIP] {ACTIVE_TARGET}/{ACTIVE_DATE} cfg 建立失敗：{_e_cfg}")
            continue

        # ── Log 檔 handler（每個目標獨立 log 檔）─────────────────────────────
        _log_path = cfg.out_dir / f"photometry_{ACTIVE_DATE}_{_log_ts}.log"
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        _file_hdl = logging.FileHandler(_log_path, encoding="utf-8")
        _file_hdl.setLevel(logging.DEBUG)
        _file_hdl.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        _logger.addHandler(_file_hdl)
        print(f"[LOG] {_log_path}")

        _ch0 = CHANNELS[0]
        wcs_files = sorted(cfg.wcs_dir.glob(f"*_{_ch0}.fits"))
        if not wcs_files:
            print(f"[SKIP] 找不到 split/{_ch0} FITS：{cfg.wcs_dir}，跳過此目標")
            continue
        print(f"找到 split/{_ch0} FITS：{len(wcs_files)} 張")

        # ── 比較星選取（天球座標，通道無關，只做一次）────────────────────────
        try:
            (comp_refs, comp_df_matched, check_star,
             aavso_matched, apass_matched, active_source) = auto_select_comps(
                wcs_files[0], cfg.target_radec_deg
            )
        except RuntimeError as _e:
            print(f"[SKIP] {ACTIVE_TARGET} 比較星選取失敗，跳過此目標：{_e}")
            continue
        print(f"check_star：{check_star}")

        # ── 孔徑估計（以第一通道為準，所有通道共用）──────────────────────────
        if cfg.aperture_auto:
            ap_r = estimate_aperture_radius(
                wcs_files[0], comp_df_matched,
                cfg.aperture_min_radius, cfg.aperture_max_radius,
                cfg.aperture_growth_fraction,
                max_stars=(
                    max(10, min(30, len(comp_df_matched)))
                    if comp_df_matched is not None else 20
                ),
            )
            if ap_r is not None:
                cfg.aperture_radius = ap_r
                print(
                    f"[生長曲線] 自動孔徑半徑 = {cfg.aperture_radius:.2f} px"
                    "（所有通道共用）"
                )

        ap_r_in, ap_r_out = compute_annulus_radii(
            cfg.aperture_radius, cfg.annulus_r_in, cfg.annulus_r_out
        )
        print(f"孔徑 r={cfg.aperture_radius:.2f}  r_in={ap_r_in:.2f}  r_out={ap_r_out:.2f}")
        _shared_aperture_radius = cfg.aperture_radius

        # ── 多通道測光迴圈 ────────────────────────────────────────────────────
        channel_results: dict = {}

        for _ch in CHANNELS:
            print(f"\n{'='*55}")
            print(f"  通道 {_ch}  ({CHANNELS.index(_ch) + 1}/{len(CHANNELS)})")
            print(f"{'='*55}")

            cfg_ch = cfg_from_yaml(_yaml, ACTIVE_TARGET, ACTIVE_DATE, channel=_ch,
                                   split_subdir=_args.split_subdir)
            cfg_ch.aperture_radius = _shared_aperture_radius
            cfg_ch.gain_e_per_adu  = cfg.gain_e_per_adu
            cfg_ch.read_noise_e    = cfg.read_noise_e

            _split_dir = cfg_ch.wcs_dir
            _fits_ch   = sorted(_split_dir.glob(f"*_{_ch}.fits"))
            if not _fits_ch:
                print(f"  [SKIP] split/{_ch}/ 找不到 FITS，跳過此通道")
                continue

            print(f"  FITS 張數：{len(_fits_ch)}")
            print(f"  輸出 CSV ：{cfg_ch.phot_out_csv}")

            df_ch, _comp_lc_ch = run_photometry_on_wcs_dir(
                _split_dir,
                cfg_ch.phot_out_csv,
                cfg_ch.phot_out_png,
                comp_refs=comp_refs,
                check_star=check_star,
                ap_radius=cfg_ch.aperture_radius,
                channel=_ch,
            )
            channel_results[_ch] = df_ch
            _ok_cnt = int(df_ch['ok'].sum()) if 'ok' in df_ch.columns else 0
            print(f"  [完成] {_ch}：ok={_ok_cnt} / {len(df_ch)} 幀")

            # R 通道：額外跑 Gaia RP→Rc 輸出獨立 R_GAIA 結果
            if str(_ch).upper() == "R":
                try:
                    print(f"\n[R_GAIA] 對 R 通道額外執行 Gaia RP→Rc 測光")
                    _gaia_r_raw = fetch_gaia_dr3_cone(
                        cfg_ch.target_radec_deg[0], cfg_ch.target_radec_deg[1],
                        radius_arcmin=cfg_ch.apass_radius_deg * 60.0,
                        mag_min=cfg_ch.comp_mag_min, mag_max=cfg_ch.comp_mag_max,
                        channel="R",
                    )
                    if len(_gaia_r_raw) >= cfg_ch.ensemble_min_comp:
                        if hasattr(cfg_ch, "out_dir"):
                            _cat_dir_r = Path(cfg_ch.out_dir) / "catalogs"
                            _cat_dir_r.mkdir(parents=True, exist_ok=True)
                            _gaia_r_raw.to_csv(_cat_dir_r / "catalog_GaiaDR3_Rc.csv", index=False)
                            print(f"  [R_GAIA] 星表已儲存 ({len(_gaia_r_raw)} 筆)")
                    else:
                        print(f"  [R_GAIA] Gaia RP→Rc 有效星不足，跳過")
                except Exception as _e_gaia_r:
                    print(f"  [WARN] R_GAIA 失敗：{_e_gaia_r}")

        print(f"\n所有通道完成：{list(channel_results.keys())}")

        # ── LS + Fourier 分析：全部通道（含 G2）──────────────────────────────
        _ls_results_all: dict = {}
        for _ls_ch, _ls_df in channel_results.items():
            if _ls_df is None:
                continue
            _n_ok = int(_ls_df["ok"].sum())
            if _n_ok < 10:
                print(f"[LS] {_ls_ch} 有效幀數 {_n_ok} < 10，跳過")
                continue
            print(f"\n[LS] 通道 {_ls_ch}（有效幀數：{_n_ok}）")
            try:
                _pa_dir  = cfg_ch.out_dir / "period_analysis"
                _pa_dir.mkdir(parents=True, exist_ok=True)
                _ls_png  = _pa_dir / f"periodogram_{_ls_ch}_{ACTIVE_DATE}.png"
                _fit_png = _pa_dir / f"phase_fold_{_ls_ch}_{ACTIVE_DATE}.png"
                _ls_r  = run_lomb_scargle(
                    _ls_df, out_png=_ls_png,
                    target_name=ACTIVE_TARGET, channel=_ls_ch, obs_date=ACTIVE_DATE,
                    lat_deg=cfg.obs_lat_deg, lon_deg=cfg.obs_lon_deg,
                )
                _fit_r = run_fourier_fit(
                    _ls_r["t"], _ls_r["mag"],
                    _ls_r["best_period"], _ls_r["err"],
                    out_png=_fit_png,
                    target_name=ACTIVE_TARGET, channel=_ls_ch, obs_date=ACTIVE_DATE,
                    lat_deg=cfg.obs_lat_deg, lon_deg=cfg.obs_lon_deg,
                )
                _ls_results_all[_ls_ch] = {"ls": _ls_r, "fit": _fit_r}
            except Exception as _e_ls:
                print(f"[LS] {_ls_ch} 失敗：{_e_ls}")

        if not _ls_results_all:
            print("[LS] 跳過（所有通道有效幀數 < 10）")

        # ── G1/G2 比值光變曲線 ────────────────────────────────────────────────
        _dg1 = channel_results.get("G1")
        _dg2 = channel_results.get("G2")
        if _dg1 is not None and _dg2 is not None:
            _need_cols = ["bjd_tdb", "t_flux_net", "m_var", "ok"]
            _g1_ok = all(c in _dg1.columns for c in _need_cols)
            _g2_ok = all(c in _dg2.columns for c in _need_cols)
            if _g1_ok and _g2_ok:
                _mg = pd.merge(
                    _dg1[_need_cols].rename(columns={
                        "t_flux_net": "flux_G1", "m_var": "m_G1", "ok": "ok_G1"
                    }),
                    _dg2[_need_cols].rename(columns={
                        "t_flux_net": "flux_G2", "m_var": "m_G2", "ok": "ok_G2"
                    }),
                    on="bjd_tdb", how="inner",
                )
                _mg = _mg[(_mg["ok_G1"] == 1) & (_mg["ok_G2"] == 1)].copy()
                if len(_mg) >= 3:
                    _mg["flux_ratio_G1G2"] = (
                        _mg["flux_G1"] / _mg["flux_G2"].replace(0, np.nan)
                    )
                    _mg["mag_diff_G1G2"] = _mg["m_G1"] - _mg["m_G2"]
                    # CSV 輸出
                    _ratio_csv = cfg_ch.out_dir / f"G1G2_ratio_{ACTIVE_DATE}.csv"
                    _mg[["bjd_tdb", "flux_ratio_G1G2", "mag_diff_G1G2",
                          "flux_G1", "flux_G2", "m_G1", "m_G2"]].to_csv(
                        _ratio_csv, index=False, float_format="%.8f"
                    )
                    print(f"[G1/G2] 比值 CSV：{_ratio_csv}（{len(_mg)} 幀）")
                    # 繪圖：flux ratio + mag diff
                    _ratio_png = cfg_ch.out_dir / f"G1G2_ratio_{ACTIVE_DATE}.png"
                    _fig_r, _ax_r = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
                    _med_ratio = float(np.nanmedian(_mg["flux_ratio_G1G2"]))
                    _med_mag   = float(np.nanmedian(_mg["mag_diff_G1G2"]))
                    _ax_r[0].plot(_mg["bjd_tdb"], _mg["flux_ratio_G1G2"], "g.", ms=4)
                    _ax_r[0].axhline(_med_ratio, color="k", ls="--", lw=0.8,
                                     label=f"median={_med_ratio:.4f}")
                    _ax_r[0].set_ylabel("G1 / G2 flux ratio")
                    _ax_r[0].set_title(f"{ACTIVE_TARGET} {ACTIVE_DATE} — G1/G2 flux ratio")
                    _ax_r[0].legend(fontsize=8)
                    _ax_r[1].plot(_mg["bjd_tdb"], _mg["mag_diff_G1G2"], "g.", ms=4)
                    _ax_r[1].axhline(_med_mag, color="k", ls="--", lw=0.8,
                                     label=f"median={_med_mag:.4f} mag")
                    _ax_r[1].set_ylabel("G1 − G2 (mag)")
                    _ax_r[1].set_xlabel("BJD_TDB")
                    _ax_r[1].legend(fontsize=8)
                    plt.tight_layout()
                    _fig_r.savefig(_ratio_png, dpi=150)
                    plt.close("all")
                    print(f"[G1/G2] 比值圖：{_ratio_png}")
                    # LS on G1/G2 ratio
                    if len(_mg) >= 10:
                        try:
                            _ratio_df = pd.DataFrame({
                                "bjd_tdb":     _mg["bjd_tdb"].values,
                                "m_var":       _mg["mag_diff_G1G2"].values,
                                "t_sigma_mag": np.full(len(_mg), 0.001),
                                "ok":          1,
                            })
                            _pa_dir_r = cfg_ch.out_dir / "period_analysis"
                            _pa_dir_r.mkdir(parents=True, exist_ok=True)
                            _ls_ratio = run_lomb_scargle(
                                _ratio_df,
                                out_png=_pa_dir_r / f"periodogram_G1G2ratio_{ACTIVE_DATE}.png",
                                target_name=ACTIVE_TARGET, channel="G1/G2 ratio",
                                obs_date=ACTIVE_DATE,
                                lat_deg=cfg.obs_lat_deg, lon_deg=cfg.obs_lon_deg,
                            )
                            run_fourier_fit(
                                _ls_ratio["t"], _ls_ratio["mag"],
                                _ls_ratio["best_period"], _ls_ratio["err"],
                                out_png=_pa_dir_r / f"phase_fold_G1G2ratio_{ACTIVE_DATE}.png",
                                target_name=ACTIVE_TARGET, channel="G1/G2 ratio",
                                obs_date=ACTIVE_DATE,
                                lat_deg=cfg.obs_lat_deg, lon_deg=cfg.obs_lon_deg,
                            )
                            print("[G1/G2] ratio LS + Fourier 完成")
                        except Exception as _e_ratio:
                            print(f"[G1/G2] LS 失敗：{_e_ratio}")
                else:
                    print(f"[G1/G2] 對齊幀數 {len(_mg)} < 3，跳過比值計算")
            else:
                print("[G1/G2] 欄位不完整（需要 t_flux_net、m_var、ok），跳過比值計算")
        else:
            print("[G1/G2] G1 或 G2 通道資料不存在，跳過比值計算")

        print(f"\n[完成] {ACTIVE_TARGET} {ACTIVE_DATE} 全部通道週期分析完成")
