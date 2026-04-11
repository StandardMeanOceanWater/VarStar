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

VERSION = "1.6"

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
warnings.filterwarnings("ignore", message=".*datfix.*")

# Windows cp950 終端機：強制 stdout/stderr 使用 UTF-8，防止 Unicode crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from datetime import datetime
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
ATime = Time   # 別名，供函式內部使用（避免 local import 遮蔽）
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from photutils.detection import DAOStarFinder, IRAFStarFinder

# ── Module-level logger（__main__ 會加 handler；函數直接用此 logger）──────────
_phot_logger = logging.getLogger("photometry")

# ── 共用設定讀取（統一由 pipeline_config.py 提供）─────────────────────────────
from pipeline_config import load_pipeline_config  # noqa: E402

# ── 拆分模組匯入 ─────────────────────────────────────────────────────────────
from phot_core import (                          # noqa: E402
    gaussian2d, fit_gaussian_psf,
    m_inst_from_flux, max_pixel_in_box,
    compute_annulus_radii, aperture_photometry,
    is_saturated, radec_to_pixel, in_bounds,
)
from phot_regression import (                    # noqa: E402
    robust_linear_fit, mag_error_from_flux,
    differential_mag,
)
from phot_catalog import (                       # noqa: E402
    _pick_col, _parse_ra_dec,
    read_aavso_seq_csv, fetch_aavso_vsp_api,
    filter_catalog_in_frame, select_comp_from_catalog,
    _selection_radius_px, _stars_in_circle,
    _fetch_apass_from_cache, fetch_apass_cone,
    fetch_tycho2_cone, fetch_gaia_dr3_cone,
    _match_catalog_to_detected,
    APASS_SCS_URL,
)


@dataclass
class Cfg:
    # ── 路徑 ──────────────────────────────────────────────────────────────────
    wcs_dir: Path = Path(".")
    out_dir: Path = Path(".")
    phot_out_csv: Path = Path("photometry.csv")
    phot_out_png: Path = Path("light_curve.png")
    regression_diag_dir: Path = Path("regression_diag")
    run_root: Path = Path(".")     # output/{date}/{group}/{target}/{timestamp}/

    # ── 目標 ──────────────────────────────────────────────────────────────────
    target_name: str = ""
    target_radec_deg: tuple = (0.0, 0.0)
    vmag_approx: float = 8.0

    # ── 比較星選取 ─────────────────────────────────────────────────────────────
    # 選取範圍：以目標星為圓心，影像短邊像素數一半為半徑
    selection_radius_mode: str = "half_short_side"
    comp_mag_range: float = 4.0        # deprecated：± 對稱範圍，優先用下方兩個
    comp_mag_bright: float = 4.0      # 比目標星亮最多 N 等
    comp_mag_faint: float = 2.0       # 比目標星暗最多 N 等
    comp_mag_min: float = 4.0         # 動態計算後填入
    comp_mag_max: float = 12.0        # 動態計算後填入
    comp_max: int = 20    # 100→64星拖垮ZP fit; 20顆近目標亮度即可
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

    # ── 孔徑測光 ──────────────────────────────────────────────────────────────
    aperture_auto: bool = True
    aperture_radius: float = 8.0
    aperture_min_radius: int = 2
    aperture_max_radius: int = 12
    aperture_growth_fraction: float = 0.97
    annulus_r_in: float | None = None
    annulus_r_out: float | None = None
    saturation_threshold: float = 65536.0  # yaml saturation_adu 覆蓋此預設值；65536=全開（14-bit 物理最大值，以 R² 監控非線性取代硬截斷）
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
    alt_min_airmass: float = 2.366   # Young (1994) at alt=25°（原 1.994=30°）

    # ── 大氣消光改正（optional）────────────────────────────────────────────────
    # 差分測光假設目標與比較星消光相同；長基線 (>2 hr) 或大氣團差 (ΔX>0.5) 時
    # 可啟用一階消光改正：m_corrected = m_var - extinction_k * (X_frame - X_ref)
    # extinction_k 為濾鏡相關消光係數 (mag/airmass)，典型值 V~0.15, R~0.10, B~0.25
    # X_ref 為參考氣團 (觀測中最小氣團)
    # 設 0.0 = 停用（預設）
    extinction_k: float = 0.0

    # ── 穩健回歸 ──────────────────────────────────────────────────────────────
    robust_regression_sigma: float = 3.0
    robust_regression_max_iter: int = 5
    robust_regression_min_points: int = 3

    # ── ⏸ Ensemble 正規化（Broeg 2005, §3.9）── 已停用，待決定 ──────────────────
    # 理由：v1.6+ 逐幀自由斜率回歸已消除逐幀大氣。
    #       ensemble 重複修正同一問題，可能引入額外雜訊。
    # 設定保留供未來重啟，但 run_photometry_on_wcs_dir() 簡化版本已跳過呼叫。
    ensemble_normalize: bool = False       # yaml: ensemble_normalize: true（已停用）
    ensemble_min_comp: int = 3             # 啟用 ensemble 所需最少比較星數（已停用）
    ensemble_max_iter: int = 10            # Broeg 迭代上限（已停用）
    ensemble_convergence_tol: float = 1e-4  # 收斂閾值（mag）（已停用）

    # ── 幀品質篩選 ────────────────────────────────────────────────────────────
    sharpness_min: float = 0.3        # Sharpness index 下限（0.0 = 停用）
    reg_r2_min: float = 0.0            # 回歸 R² 下限（0.0 = 停用）
    peak_ratio_min: float = 0.0       # [deprecated] 峰值比固定下限（0.0=停用）；建議改用 peak_ratio_k
    peak_ratio_k: float = 0.0         # 自適應門檻倍數（0.0=停用）；peak_ratio < median - k×MAD 時剔除
    reg_intercept_sigma: float = 0.0   # 回歸截距突變閾值（倍 MAD，0.0 = 停用）；薄雲/透明度驟變偵測
    sky_sigma: float = 0.0            # 背景突升閾值（倍 MAD，0.0 = 停用）；起霧/散射光偵測

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


def _resolve_extinction_k(yaml_dict: dict, channel: str, phot_cfg: dict) -> float:
    """從 extinction.coefficients 取 per-channel 值，fallback 到 photometry.extinction_k。"""
    ext_coeffs = yaml_dict.get("extinction", {}).get("coefficients", {})
    # channel 映射：G1/G2 → G
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
    從 yaml 字典建立 Cfg 物件。

    Parameters
    ----------
    yaml_dict    : load_pipeline_config() 回傳的字典。
    target       : 目標名稱，對應 yaml targets 區塊的鍵值。
    session_date : 觀測日期字串，例如 "20241220"。
    """
    data_root = yaml_dict["_data_root"]
    project_root = yaml_dict["_project_root"]

    # ── 取得目標設定 ───────────────────────────────────────────────────────────
    tgt = yaml_dict.get("targets", {}).get(target, {})

    # ── 照片組 (group) ─────────────────────────────────────────────────────
    group = tgt.get("group", target)
    _date_fmt = f"{session_date[:4]}-{session_date[4:6]}-{session_date[6:8]}"
    field_root = data_root / _date_fmt / group

    # ra_deg 優先；fallback 到 ra_hint_h（小時角 × 15 轉為度）
    if "ra_deg" in tgt:
        ra_deg = float(tgt["ra_deg"])
    elif "ra_hint_h" in tgt:
        ra_deg = float(tgt["ra_hint_h"]) * 15.0
    else:
        raise ValueError(
            f"{target}：yaml targets 缺少 ra_deg 或 ra_hint_h，無法繼續。"
            f"  請在 observation_config.yaml 的 targets.{target} 補上座標。"
        )

    # dec_deg 優先；fallback 到 dec_hint_deg
    if "dec_deg" in tgt:
        dec_deg = float(tgt["dec_deg"])
    elif "dec_hint_deg" in tgt:
        dec_deg = float(tgt["dec_hint_deg"])
    else:
        raise ValueError(
            f"{target}：yaml targets 缺少 dec_deg 或 dec_hint_deg，無法繼續。"
        )
    vmag    = float(tgt.get("vmag_approx", 8.0))
    display = tgt.get("display_name", target)

    # ── 比較星星等範圍（動態，非對稱）────────────────────────────────────────
    _cmp = yaml_dict.get("comparison_stars", {})
    if "comp_mag_bright" in _cmp or "comp_mag_faint" in _cmp:
        # 新參數：非對稱範圍
        _bright = float(_cmp.get("comp_mag_bright", 4.0))
        _faint  = float(_cmp.get("comp_mag_faint",  2.0))
        comp_mag_min = vmag - _bright
        comp_mag_max = vmag + _faint
    else:
        # fallback：舊的對稱參數
        print(
            "[WARN] comparison_stars.comp_mag_bright / comp_mag_faint 未設定，"
            "退回 mag_range_delta（deprecated）。請更新 yaml。"
        )
        _range = float(_cmp.get("mag_range_delta", 4.0))
        comp_mag_min = vmag - _range
        comp_mag_max = vmag + _range
    # yaml 固定上下限作為夾鉗（不飽和保護 / 信噪比下限），取更嚴格的那側
    if "comp_mag_min" in _cmp:
        comp_mag_min = max(comp_mag_min, float(_cmp["comp_mag_min"]))
    if "comp_mag_max" in _cmp:
        comp_mag_max = min(comp_mag_max, float(_cmp["comp_mag_max"]))

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
    _cameras = yaml_dict.get("cameras", {})
    cam_cfg = _cameras.get(camera_id, {})
    if not cam_cfg:
        # session camera 全名 vs cameras 區塊簡稱不匹配 → 模糊搜尋
        for _ckey, _cval in _cameras.items():
            if isinstance(_cval, dict) and (_cval.get("name", "") == camera_id or camera_id in _ckey or _ckey in camera_id):
                cam_cfg = _cval
                print(f"  [WARN] camera '{camera_id}' 不在 cameras 區塊，模糊匹配到 '{_ckey}'")
                break
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
    _sat_raw   = cam_cfg.get("saturation_adu", 11469.0)
    sat_adu    = None if _sat_raw is None else float(_sat_raw)

    # ── 路徑 ───────────────────────────────────────────────────────────────────
    # 測光通道：split/B/（拆色後 B 通道；FITS 內含 WCS，由 DeBayer_RGGB.py 傳遞）
    channel = str(channel).upper()   # 使用傳入參數，不從 yaml 讀
    wcs_dir  = field_root / split_subdir / channel
    # ── 輸出目錄（時間戳制，每次執行獨立）─────────────────────────────────
    _run_ts = run_ts or datetime.now().strftime("%Y%m%d_%H%M")
    run_root = project_root / "output" / _date_fmt / group / target / _run_ts
    out_dir  = run_root / "1_photometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = run_root / "2_regression_diag"
    diag_dir.mkdir(parents=True, exist_ok=True)
    lc_dir   = run_root / "3_light_curve"
    lc_dir.mkdir(parents=True, exist_ok=True)
    pa_dir   = run_root / "4_period_analysis"
    pa_dir.mkdir(parents=True, exist_ok=True)

    # 通道對應 APASS 波段
    _band_map = {"R": "r", "G1": "V", "G2": "V", "B": "B"}
    phot_band = _band_map.get(channel.upper(), "V")

    # ── Auto-fill：phot_cfg 中與 Cfg 欄位同名的 key 自動帶入 ──────────────
    # 手動指定的值（下方 Cfg(...)）優先；auto-fill 只補漏。
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
                pass  # 轉型失敗就跳過，讓手動值或預設值生效

    # ── 手動指定的值（優先於 auto-fill）────────────────────────────────────
    _manual = dict(
        # 路徑
        run_root=run_root,
        wcs_dir=wcs_dir,
        out_dir=out_dir,
        phot_out_csv=out_dir / f"photometry_{channel}_{session_date}.csv",
        phot_out_png=lc_dir / f"light_curve_{channel}_{session_date}.png",
        regression_diag_dir=diag_dir,

        # 目標
        target_name=display,
        target_radec_deg=(ra_deg, dec_deg),
        vmag_approx=vmag,
        aavso_star_name=display,

        # 比較星
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

        # 孔徑測光
        aperture_auto=True,
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

        # 板塊比例尺
        plate_scale_arcsec=plate_scale,

        # 觀測站
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

        # 消光
        extinction_k=_resolve_extinction_k(yaml_dict, channel, phot_cfg),

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
    # auto-fill 墊底，手動值覆蓋
    _merged = {**_auto, **_manual}
    return Cfg(**_merged
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

# [已拆至 phot_core.py] radec_to_pixel, in_bounds

"""## Gaussian PSF fit (FWHM + Flux)"""

from scipy.optimize import curve_fit

# [已拆至 phot_core.py] gaussian2d, fit_gaussian_psf



"""## Instrumental magnitude"""

# [已拆至 phot_core.py] m_inst_from_flux

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
    if False:  # saturation_threshold 允許 null（關閉飽和篩選）
        missing.append("saturation_threshold")
    if cfg.gain_e_per_adu is None:
        missing.append("gain_e_per_adu")
    if cfg.read_noise_e is None:
        missing.append("read_noise_e")
    if missing:
        raise ValueError("Missing cfg values: " + ", ".join(missing))


# [已拆至 phot_core.py] max_pixel_in_box


# [已拆至 phot_core.py] compute_annulus_radii, aperture_photometry, is_saturated


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


# [已拆至 phot_regression.py] robust_linear_fit, mag_error_from_flux


# [已拆至 phot_catalog.py] _pick_col, _parse_ra_dec


# [已拆至 phot_catalog.py] read_aavso_seq_csv


# [已拆至 phot_catalog.py] fetch_aavso_vsp_api


# [已拆至 phot_catalog.py] filter_catalog_in_frame, select_comp_from_catalog


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

# [已拆至 phot_catalog.py] _selection_radius_px, _stars_in_circle,
# _fetch_apass_from_cache, fetch_apass_cone, fetch_tycho2_cone,
# fetch_gaia_dr3_cone, _match_catalog_to_detected, APASS_SCS_URL

def _save_regression_diagnostic(
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
        f"Regression Fit Diagnostic  [{frame_name}]\n"
        f"Active source: {active_source}",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = diag_dir / f"reg_diag_{Path(frame_name).stem}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_or_build_unified_catalog(
    cfg,
    ra_t: float,
    dec_t: float,
    field_cat_dir: Path,
) -> "tuple[pd.DataFrame, list[str]]":
    """
    統一視場星表：查詢所有星表來源一次，合併多波段欄位。

    Returns (unified_df, sources_used).
    unified_df 欄位：
        ra_deg, dec_deg,
        V_mag, V_err, source_V,   — Johnson V: AAVSO > APASS > Tycho VT→V
        B_mag, B_err, source_B,   — Johnson B: APASS B > Tycho BT→B (B<13 only)
        R_mag, R_err, source_R,   — Cousins Rc: Gaia RP→Rc ONLY
        BT, VT, Gmag, BPmag, RPmag  — 原始診斷欄位
    """
    field_cat_path = field_cat_dir / "field_catalog_unified.csv"
    _sources_used: "list[str]" = []

    if field_cat_path.exists():
        print(f"  [unified catalog] read: {field_cat_path.name}")
        unified_df = pd.read_csv(field_cat_path)
        for _sc in ("source_V", "source_B", "source_R"):
            if _sc in unified_df.columns:
                _sources_used.extend(unified_df[_sc].dropna().unique().tolist())
        _sources_used = sorted(set(_sources_used))
        print(f"  [unified catalog] {len(unified_df)} stars, sources: {'+'.join(_sources_used)}")
        return unified_df, _sources_used

    # ── 查詢各星表（各查一次）─────────────────────────────────────────────────
    # 每個 source 以統一格式收集：ra_deg, dec_deg + 各波段欄位
    _source_rows: "list[dict]" = []

    _radius_arcmin = cfg.apass_radius_deg * 60.0

    # --- AAVSO: 只有 V ---
    _has_aavso = False
    for _cat_name in cfg.catalog_priority:
        if str(_cat_name).upper() == "AAVSO":
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
                    print(f"  [WARN] AAVSO API query failed: {exc}")
            if seq_df is not None and len(seq_df) > 0:
                for _, r in seq_df.iterrows():
                    _source_rows.append({
                        "ra_deg": float(r["ra_deg"]), "dec_deg": float(r["dec_deg"]),
                        "V_mag": float(r["m_cat"]),
                        "V_err": float(r.get("m_err", np.nan)) if "m_err" in r.index else np.nan,
                        "source_V": "AAVSO",
                    })
                _has_aavso = True
                _sources_used.append("AAVSO")
                print(f"  [unified] AAVSO: {len(seq_df)} stars (V only)")
            break

    # --- APASS: V + B ---
    _has_apass = False
    for _cat_name in cfg.catalog_priority:
        if str(_cat_name).upper() == "APASS":
            apass_raw = fetch_apass_cone(
                ra_t, dec_t, radius_deg=cfg.apass_radius_deg, maxrec=cfg.apass_maxrec,
            )
            if len(apass_raw) > 0:
                _ra_col = _pick_col(apass_raw, ["ra", "raj2000", "ra_deg", "ra_icrs"])
                _dec_col = _pick_col(apass_raw, ["dec", "dej2000", "dec_deg", "dec_icrs"])
                # V band columns
                _v_col = next((c for c in ["vmag", "mag_v", "v", "v_mag"] if c in apass_raw.columns), None)
                _ve_col = next((c for c in ["e_vmag", "err_mag_v", "v_err"] if c in apass_raw.columns), None)
                # B band columns
                _b_col = next((c for c in ["bmag", "mag_b", "b", "b_mag"] if c in apass_raw.columns), None)
                _be_col = next((c for c in ["e_bmag", "err_mag_b", "b_err"] if c in apass_raw.columns), None)
                n_v, n_b = 0, 0
                for _, r in apass_raw.iterrows():
                    row = {"ra_deg": float(r[_ra_col]), "dec_deg": float(r[_dec_col])}
                    has_any = False
                    if _v_col and np.isfinite(float(r.get(_v_col, np.nan))):
                        row["V_mag"] = float(r[_v_col])
                        row["V_err"] = float(r[_ve_col]) if _ve_col and np.isfinite(float(r.get(_ve_col, np.nan))) else np.nan
                        row["source_V"] = "APASS"
                        has_any = True
                        n_v += 1
                    if _b_col and np.isfinite(float(r.get(_b_col, np.nan))):
                        row["B_mag"] = float(r[_b_col])
                        row["B_err"] = float(r[_be_col]) if _be_col and np.isfinite(float(r.get(_be_col, np.nan))) else np.nan
                        row["source_B"] = "APASS"
                        has_any = True
                        n_b += 1
                    if has_any:
                        _source_rows.append(row)
                _has_apass = True
                _sources_used.append("APASS")
                print(f"  [unified] APASS: {len(apass_raw)} stars (V={n_v}, B={n_b})")
            break

    # --- Tycho-2: V (VT→V) + B (BT→B) ---
    _has_tycho = False
    for _cat_name in cfg.catalog_priority:
        if str(_cat_name).upper() == "TYCHO2":
            tycho_raw = fetch_tycho2_cone(
                ra_t, dec_t,
                radius_arcmin=_radius_arcmin,
                mag_min=3.0, mag_max=16.0,
            )
            if len(tycho_raw) > 0:
                n_v, n_b = 0, 0
                for _, r in tycho_raw.iterrows():
                    row = {"ra_deg": float(r["ra_deg"]), "dec_deg": float(r["dec_deg"])}
                    # V from Tycho (already converted VT→V in fetch_tycho2_cone)
                    _vm = float(r.get("vmag", np.nan))
                    if np.isfinite(_vm):
                        row["V_mag"] = _vm
                        row["V_err"] = float(r.get("e_vmag", np.nan))
                        row["source_V"] = "Tycho2"
                        n_v += 1
                    # B from BT→B conversion: B = BT - 0.240*(BT-VT)
                    # B < 13 mag 的亮星 APASS 已覆蓋且精度更好，
                    # Tycho-2 BT→B 只填 B >= 13 的暗星（APASS 沒有的）
                    _bt = float(r.get("BT", np.nan))
                    _vt = float(r.get("VT", np.nan))
                    if np.isfinite(_bt) and np.isfinite(_vt):
                        _b_mag = _bt - 0.240 * (_bt - _vt)
                        if _b_mag >= 13.0:
                            _e_bt = 0.05  # Tycho-2 typical BT error
                            _e_vt = 0.05
                            _e_b = float(np.sqrt(_e_bt**2 * (1 - 0.240)**2 + _e_vt**2 * 0.240**2))
                            row["B_mag"] = _b_mag
                            row["B_err"] = _e_b
                            row["source_B"] = "Tycho2"
                            n_b += 1
                    # 保留原始欄位
                    if np.isfinite(_bt):
                        row["BT"] = _bt
                    if np.isfinite(_vt):
                        row["VT"] = _vt
                    _source_rows.append(row)
                _has_tycho = True
                _sources_used.append("Tycho2")
                print(f"  [unified] Tycho-2: {len(tycho_raw)} stars (V={n_v}, B={n_b})")
            break

    # --- Gaia DR3: R (RP→Rc) ONLY ---
    _has_gaia = False
    for _cat_name in cfg.catalog_priority:
        if str(_cat_name).upper() in ("GAIA", "GAIADR3", "GAIA_DR3"):
            gaia_raw = fetch_gaia_dr3_cone(
                ra_t, dec_t,
                radius_arcmin=_radius_arcmin,
                mag_min=3.0, mag_max=16.0,
                channel="R",   # 只取 RP→Rc
            )
            if len(gaia_raw) > 0:
                for _, r in gaia_raw.iterrows():
                    _rc = float(r.get("vmag", np.nan))  # fetch_gaia_dr3_cone 回傳 vmag=Rc
                    if not np.isfinite(_rc):
                        continue
                    row = {
                        "ra_deg": float(r["ra_deg"]), "dec_deg": float(r["dec_deg"]),
                        "R_mag": _rc,
                        "R_err": float(r.get("e_vmag", np.nan)),
                        "source_R": "GaiaDR3",
                    }
                    for _extra in ("Gmag", "BPmag", "RPmag"):
                        if _extra in r.index and np.isfinite(float(r.get(_extra, np.nan))):
                            row[_extra] = float(r[_extra])
                    _source_rows.append(row)
                _has_gaia = True
                _sources_used.append("GaiaDR3")
                print(f"  [unified] Gaia DR3: {len(gaia_raw)} stars (R=Rc only)")
            break

    _sources_used = sorted(set(_sources_used))

    if not _source_rows:
        print("  [unified catalog] 所有星表查詢無結果")
        return pd.DataFrame(), _sources_used

    # ── 合併：同位置 (<3") 的不同源合併多波段欄位 ──────────────────────────────
    all_df = pd.DataFrame(_source_rows)
    # 確保所有欄位存在
    for _col in ("V_mag", "V_err", "source_V", "B_mag", "B_err", "source_B",
                 "R_mag", "R_err", "source_R", "BT", "VT", "Gmag", "BPmag", "RPmag"):
        if _col not in all_df.columns:
            all_df[_col] = np.nan if not _col.startswith("source") else None

    from astropy.coordinates import SkyCoord as _SC
    coords = _SC(all_df["ra_deg"].values * u.deg, all_df["dec_deg"].values * u.deg)

    # 用 union-find 聚類 <3" 的星
    n = len(all_df)
    parent = list(range(n))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    # 建立鄰近關係（O(n²) 但 n 通常 <2000）
    for i in range(n):
        seps = coords[i].separation(coords[i+1:]).arcsec
        for j_off, sep in enumerate(seps):
            if sep < 3.0:
                _union(i, i + 1 + j_off)

    # 按群組合併
    from collections import defaultdict as _defaultdict
    groups = _defaultdict(list)
    for i in range(n):
        groups[_find(i)].append(i)

    # V 優先級：AAVSO > APASS > Tycho2（照 catalog_priority 順序）
    _v_priority = {"AAVSO": 0, "APASS": 1, "Tycho2": 2}
    _b_priority = {"APASS": 0, "Tycho2": 1}

    unified_rows = []
    for _root, indices in groups.items():
        # 取群組的平均位置（以第一個為主）
        _ra = float(all_df.iloc[indices[0]]["ra_deg"])
        _dec = float(all_df.iloc[indices[0]]["dec_deg"])
        row = {"ra_deg": _ra, "dec_deg": _dec}

        # 從群組中挑最佳 V, B, R
        best_v, best_b, best_r = None, None, None
        best_v_pri, best_b_pri = 999, 999

        for idx in indices:
            r = all_df.iloc[idx]

            # V
            _sv = r.get("source_V")
            _vm = r.get("V_mag")
            if _sv and isinstance(_sv, str) and np.isfinite(float(_vm if _vm is not None else np.nan)):
                pri = _v_priority.get(_sv, 99)
                if pri < best_v_pri:
                    best_v_pri = pri
                    best_v = (float(_vm), float(r.get("V_err", np.nan)), _sv)

            # B
            _sb = r.get("source_B")
            _bm = r.get("B_mag")
            if _sb and isinstance(_sb, str) and np.isfinite(float(_bm if _bm is not None else np.nan)):
                pri = _b_priority.get(_sb, 99)
                if pri < best_b_pri:
                    best_b_pri = pri
                    best_b = (float(_bm), float(r.get("B_err", np.nan)), _sb)

            # R (only GaiaDR3)
            _sr = r.get("source_R")
            _rm = r.get("R_mag")
            if _sr and isinstance(_sr, str) and _sr == "GaiaDR3" and np.isfinite(float(_rm if _rm is not None else np.nan)):
                best_r = (float(_rm), float(r.get("R_err", np.nan)), "GaiaDR3")

            # 保留原始診斷欄位
            for _diag in ("BT", "VT", "Gmag", "BPmag", "RPmag"):
                _dv = r.get(_diag)
                if _dv is not None and np.isfinite(float(_dv)):
                    row.setdefault(_diag, float(_dv))

        if best_v:
            row["V_mag"], row["V_err"], row["source_V"] = best_v
        if best_b:
            row["B_mag"], row["B_err"], row["source_B"] = best_b
        if best_r:
            row["R_mag"], row["R_err"], row["source_R"] = best_r

        # 至少有一個波段才保留
        if best_v or best_b or best_r:
            unified_rows.append(row)

    unified_df = pd.DataFrame(unified_rows)

    # 確保欄位順序與完整性
    _all_cols = ["ra_deg", "dec_deg",
                 "V_mag", "V_err", "source_V",
                 "B_mag", "B_err", "source_B",
                 "R_mag", "R_err", "source_R",
                 "BT", "VT", "Gmag", "BPmag", "RPmag"]
    for _c in _all_cols:
        if _c not in unified_df.columns:
            unified_df[_c] = np.nan if not _c.startswith("source") else None
    unified_df = unified_df[_all_cols]

    # 儲存快取
    field_cat_dir.mkdir(parents=True, exist_ok=True)
    unified_df.to_csv(field_cat_path, index=False, encoding="utf-8-sig")
    print(f"  [unified catalog] saved: {field_cat_path.name} ({len(unified_df)} rows)")
    print(f"  [unified catalog] V={int(unified_df['V_mag'].notna().sum())}  "
          f"B={int(unified_df['B_mag'].notna().sum())}  "
          f"R={int(unified_df['R_mag'].notna().sum())}  "
          f"sources: {'+'.join(_sources_used)}")

    return unified_df, _sources_used


## _remap_comp_refs_to_band — REMOVED by unified catalog refactor (v1.7)
## 每通道現在直接從 unified catalog 獨立選取比較星，不再需要 remap。


def auto_select_comps(
    wcs_fits_path: Path,
    target_radec_deg: tuple,
    band: str = "V",
    max_detect: int = 500,
    psf_box: int = 25,
    threshold_sigma: float = 5.0,
):
    """
    比較星自動選取主函式（unified catalog 版）。

    每個通道獨立呼叫此函式，band 指定使用 unified catalog 的哪個波段欄位。

    Parameters
    ----------
    band : str
        "V" (G1/G2 通道), "B" (B 通道), "R" (R 通道)

    Returns
    -------
    comp_refs       : list of (ra, dec, m_cat, m_err, weight)
    comp_df_matched : 選入回歸的比較星 DataFrame
    check_star      : (ra, dec, m_cat | None) 或 None
    aavso_matched   : 匹配結果（診斷用）
    apass_matched   : 匹配結果（診斷用）
    active_source   : 使用的星表來源
    vsx_field       : VSX 查詢結果 DataFrame（供額外目標星測光用）
    """
    # Band → unified catalog column mapping
    _band_mag_col = {"V": "V_mag", "B": "B_mag", "R": "R_mag"}
    _band_err_col = {"V": "V_err", "B": "B_err", "R": "R_err"}
    _band_src_col = {"V": "source_V", "B": "source_B", "R": "source_R"}
    mag_col = _band_mag_col.get(band, "V_mag")
    err_col = _band_err_col.get(band, "V_err")
    src_col = _band_src_col.get(band, "source_V")

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

    def _catalog_direct_phot(cat_df, ra_col, dec_col, mag_col_, err_col_, mag_min, mag_max):
        """從星表座標直接做孔徑測光，回傳通過篩選的 DataFrame。
        排除條件：(a) mag 範圍外；(b) 選取圓外；(c) 影像邊界外；
                  (d) 飽和；(e) 測光失敗。
        """
        rows = []
        n_mag_out, n_circle, n_bounds, n_sat, n_phot = 0, 0, 0, 0, 0
        for _, row in cat_df.iterrows():
            try:
                ra_c  = float(row[ra_col])
                dec_c = float(row[dec_col])
                m_c   = float(row[mag_col_])
                m_e   = float(row[err_col_]) if (err_col_ and err_col_ in row.index
                                                and np.isfinite(row[err_col_])) else np.nan
            except (KeyError, TypeError, ValueError):
                continue
            if not (mag_min <= m_c <= mag_max):
                n_mag_out += 1
                continue
            xc, yc = radec_to_pixel(wcs_obj, ra_c, dec_c)
            # 選取圓：影像中心為圓心，半徑 = 短邊 / 2
            _cx, _cy = _w / 2.0, _h / 2.0
            _sel_r   = float(min(_h, _w)) / 2.0
            if np.hypot(xc - _cx, yc - _cy) > _sel_r:
                n_circle += 1
                continue
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
        print(f"  [直接測光] band={band}  星表={len(cat_df)}  "
              f"mag範圍外={n_mag_out}({mag_min:.1f}-{mag_max:.1f})  "
              f"選取圓外={n_circle}  邊界排除={n_bounds}  "
              f"飽和={n_sat}  測光失敗={n_phot}  通過={len(rows)}")
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ── 3–5. 統一視場星表（所有波段共用一份快取）────────────────────────────
    # catalogs 放在 group（星場）層級，同星場多目標共用
    _field_cat_dir  = cfg.out_dir.parent.parent.parent / "catalogs"

    unified_df, _sources_used = _load_or_build_unified_catalog(
        cfg, ra_t, dec_t, _field_cat_dir,
    )

    # 篩選此波段有值的星
    if len(unified_df) == 0 or mag_col not in unified_df.columns:
        raise RuntimeError(
            f"統一星表中無 {band} 波段資料。\n"
            "建議：(1) 刪除 field_catalog_unified.csv 重建；"
            "(2) 確認 catalog_priority 設定。"
        )

    band_df = unified_df[unified_df[mag_col].notna()].copy()
    # 建立 m_cat / m_err 欄位供 _catalog_direct_phot 使用
    band_df = band_df.rename(columns={mag_col: "m_cat", err_col: "m_err"})
    if "m_err" not in band_df.columns:
        band_df["m_err"] = np.nan

    # 來源統計
    _band_sources = []
    if src_col in unified_df.columns:
        _band_sources = sorted(unified_df.loc[unified_df[mag_col].notna(), src_col].dropna().unique().tolist())
    active_source = "+".join(_band_sources) if _band_sources else "+".join(_sources_used)

    print(f"  [unified] band={band}: {len(band_df)} stars with {mag_col} "
          f"(sources: {active_source})")

    # 強制孔徑測光篩選
    aavso_matched = None
    apass_matched = None
    active_df     = None

    # ── 以目標星在該 band 的實際星等重算 comp mag range ──────────────────
    #   cfg.comp_mag_min/max 是用 vmag_approx(V) 算的，B/R 波段需要修正。
    #   從 unified catalog 查目標星在此 band 的星等，用相同 bright/faint offset 重算。
    #   注意：必須在 VSX 排除之前查，因為目標星本身是變星會被 VSX 踢掉。
    _comp_mag_min_band = cfg.comp_mag_min
    _comp_mag_max_band = cfg.comp_mag_max
    if band != "V" and len(band_df) > 0:
        _tgt_cat = SkyCoord(band_df["ra_deg"].values * u.deg,
                            band_df["dec_deg"].values * u.deg)
        _sep = tgt_sc.separation(_tgt_cat).arcsec
        _nearest_idx = int(np.argmin(_sep))
        if _sep[_nearest_idx] < 10.0:  # 10" 以內才信
            _tgt_band_mag = float(band_df.iloc[_nearest_idx]["m_cat"])
            # 用 cfg 裡的 bright/faint offset（從 vmag 反推）
            _bright_offset = cfg.vmag_approx - cfg.comp_mag_min
            _faint_offset  = cfg.comp_mag_max - cfg.vmag_approx
            _comp_mag_min_band = _tgt_band_mag - _bright_offset
            _comp_mag_max_band = _tgt_band_mag + _faint_offset
            # 再套 yaml 夾鉗
            _yaml_floor = 6.0    # comp_mag_min 硬下限
            _yaml_ceil  = 13.0   # comp_mag_max 硬上限
            _comp_mag_min_band = max(_comp_mag_min_band, _yaml_floor)
            _comp_mag_max_band = min(_comp_mag_max_band, _yaml_ceil)
            print(f"  [mag range] band={band}: 目標星 {band}_mag={_tgt_band_mag:.2f} "
                  f"→ comp range [{_comp_mag_min_band:.1f}, {_comp_mag_max_band:.1f}] "
                  f"(V 基準 [{cfg.comp_mag_min:.1f}, {cfg.comp_mag_max:.1f}])")

    # ── VSX 已知變星排除（比較星不能是變星）──────────────────────────────
    _vsx = pd.DataFrame()
    if len(band_df) > 0:
        try:
            from tools.local_catalog import query_vsx_cone, filter_known_variables
            _vsx = query_vsx_cone(ra_t, dec_t, radius_deg=cfg.apass_radius_deg)
            if len(_vsx) > 0:
                _n_before_vsx = len(band_df)
                band_df = filter_known_variables(band_df, _vsx, match_arcsec=10.0)
                print(f"  [VSX] 比較星候選：{_n_before_vsx} → {len(band_df)}（排除已知變星）")
        except Exception as _e_vsx:
            print(f"  [VSX] 變星排除跳過：{_e_vsx}")

    if len(band_df) > 0:
        active_df = _catalog_direct_phot(
            band_df, "ra_deg", "dec_deg", "m_cat", "m_err",
            _comp_mag_min_band, _comp_mag_max_band,
        )
        apass_matched = active_df
        aavso_matched = active_df

    if active_df is None or len(active_df) == 0:
        raise RuntimeError(
            f"band={band} 在星表中找不到足夠的比較星。\n"
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

    # ── 6b. comp_max 截斷 ──────────────────────────────────────────────────────
    if cfg.comp_max > 0 and len(comp_refs) > cfg.comp_max:
        _vmag_t = float(getattr(cfg, "vmag_approx", np.nan))
        if np.isfinite(_vmag_t):
            comp_refs.sort(key=lambda x: abs(x[2] - _vmag_t))
        else:
            comp_refs.sort(key=lambda x: x[4], reverse=True)
        _n_before = len(comp_refs)
        comp_refs = comp_refs[:cfg.comp_max]
        print(f"[比較星] comp_max={cfg.comp_max}：{_n_before} → {len(comp_refs)} 顆"
              f"（按星等接近度排序）")
        _kept_coords = {(r[0], r[1]) for r in comp_refs}
        active_df = active_df[
            active_df.apply(lambda row: (row["ra_deg"], row["dec_deg"]) in _kept_coords, axis=1)
        ].reset_index(drop=True)

    # ── 7. 檢查星 ──────────────────────────────────────────────────────────────
    # band_df 含該波段所有候選星（含 mag range 外），比 active_df 寬得多，
    # 確保扣除 comp 後仍有 check star 候選。
    check_star = select_check_star(
        cfg.target_radec_deg, comp_refs, band_df
    )

    print(f"[比較星] band={band} 使用 {active_source}：{len(comp_refs)} 顆")
    print(f"[比較星] epsilon = {epsilon:.4f} arcsec")

    return comp_refs, active_df, check_star, aavso_matched, apass_matched, active_source, _vsx


# =============================================================================
# ⏸ 待決定：差分測光 (differential_mag) vs 自由斜率回歸 (ensemble normalization)
# =============================================================================
# 背景：v1.6+ 採用逐幀自由斜率回歸 fit(m_inst = a·m_cat + b)，已消除大氣漂移。
#      差分測光 & ensemble 正規化均解決同一問題（逐幀大氣修正），二選一即可。
#      目前回歸已完全滿足需求（R²≥0.9），故暫停差分測光&正規化。
# 決議：待驗證 ensemble normalization 的實際貢獻（可能引入雜訊）。
#      完全消除後需併同此檔案再行決定。

# [已拆至 phot_regression.py] differential_mag


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
    ⏸ 已停用：Broeg (2005) ensemble normalisation.

    理由：逐幀自由斜率回歸已完全消除逐幀大氣漂移，
         ensemble 正規化解決的是同一問題，二者不應疊加（可能引入雜訊）。
    待驗：確認移除後 R² 無退化。

    ─── 原始文件 ───

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
    # delta_new 是儀器星等空間的漂移量；m_var 是校正星等空間 (m_inst-b)/a。
    # 轉換：delta_catalog = delta_inst / a，才能從 m_var 中正確扣除。
    _reg_slope = df_out["reg_slope"].values if "reg_slope" in df_out.columns else np.ones(len(df_out))
    _reg_slope_safe = np.where(np.isfinite(_reg_slope) & (_reg_slope != 0), _reg_slope, 1.0)
    delta_scaled = delta_new / _reg_slope_safe

    m_var_norm = np.where(
        np.isfinite(df_out["m_var"].values) & np.isfinite(delta_scaled),
        df_out["m_var"].values - delta_scaled,
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


class _FrameCompCache:
    """同視野多目標共用的比較星測光快取。
    第一顆目標測完後存入，後續目標直接讀取，避免重複孔徑測光。
    Key: (frame_stem, ra_rounded_5dp, dec_rounded_5dp)
    """
    def __init__(self):
        self._data: dict = {}

    def _key(self, frame_stem: str, ra: float, dec: float) -> tuple:
        return (frame_stem, round(float(ra), 5), round(float(dec), 5))

    def get(self, frame_stem: str, ra: float, dec: float):
        return self._data.get(self._key(frame_stem, ra, dec))

    def set(self, frame_stem: str, ra: float, dec: float, result: dict):
        self._data[self._key(frame_stem, ra, dec)] = result

    def __len__(self):
        return len(self._data)



def plot_light_curve(
    df: pd.DataFrame,
    out_png: Path,
    channel: str,
    cfg,
    obs_date: "str | None" = None,
    ylim: "tuple[float, float] | None" = None,
):
    """
    Generate the light curve plot from a photometry result DataFrame.

    ylim : (ymin, ymax) in mag — 若提供則鎖定 Y 軸範圍（已含 invert）。
    """
    from astropy.time import Time as ATime
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import datetime as _dt_local

    time_key = "bjd_tdb" if "bjd_tdb" in df.columns else "jd"
    if time_key not in df.columns:
        print(f"[PLOT] Error: time column '{time_key}' not found.")
        return

    # Filter valid points
    # ensemble_normalize 啟用時優先用 m_var_norm；否則退回 m_var
    _mag_col = "m_var"  # v1.5 邏輯：直接用回歸校正星等；ensemble normalization 已知有問題
    d = df[(df["ok"] == 1) & np.isfinite(df[time_key]) & np.isfinite(df[_mag_col])].copy()
    if len(d) == 0:
        print("[PLOT] No valid photometry points to plot.")
        return

    bjd_arr = d[time_key].values
    bjd_min = float(bjd_arr.min())
    bjd_max = float(bjd_arr.max())

    # Time scaling for Local Time (UTC+8 default)
    _tz_offset_h = float(getattr(cfg, "tz_offset_hours", 8))

    def _bjd_to_local_hm(bjd_val):
        try:
            t_utc = ATime(bjd_val, format="jd", scale="tdb").to_datetime()
            return t_utc + _dt_local.timedelta(hours=_tz_offset_h)
        except Exception:
            return None

    # Compute Ticks
    _t_local_min = _bjd_to_local_hm(bjd_min)
    _t_local_max = _bjd_to_local_hm(bjd_max)
    _label30_bjd_ticks = []
    _label30_labels = []
    _minor_bjd_ticks = []
    if _t_local_min is not None and _t_local_max is not None:
        _cur = _t_local_min.replace(second=0, microsecond=0)
        _cur = _cur.replace(minute=(_cur.minute // 10) * 10)
        while _cur <= _t_local_max + _dt_local.timedelta(minutes=1):
            _utc = _cur - _dt_local.timedelta(hours=_tz_offset_h)
            _b_tick = ATime(_utc).jd
            _minor_bjd_ticks.append(_b_tick)
            if _cur.minute in (0, 30):
                _label30_bjd_ticks.append(_b_tick)
                _label30_labels.append(_cur.strftime("%H:%M"))
            _cur += _dt_local.timedelta(minutes=10)

    fig, ax = plt.subplots(figsize=(12, 4))

    # Error bars or dots
    _y_vals = d[_mag_col].values
    if "t_sigma_mag" in d.columns and np.isfinite(d["t_sigma_mag"]).any():
        ax.errorbar(bjd_arr, _y_vals, yerr=d["t_sigma_mag"].values,
                    fmt="o", ms=4, capsize=2, lw=0.8, label="± σ", zorder=3)
    else:
        ax.plot(bjd_arr, _y_vals, "o-", ms=4, lw=0.8, label="± σ", zorder=3)

    ax.invert_yaxis()
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])   # ylim 應已考慮 invert（大值在下）
        # 強制統一 Y 軸 tick 間距（0.05 mag），避免 matplotlib 自動選不同間距
        _y_lo = min(ylim)   # 亮端（小數值）
        _y_hi = max(ylim)   # 暗端（大數值）
        _yticks = np.arange(np.ceil(_y_lo * 20) / 20, _y_hi + 0.001, 0.05)
        ax.set_yticks(_yticks)
    ax.set_xlim(bjd_min - 0.002, bjd_max + 0.002)

    # Ticks & Label (Navy)
    if _minor_bjd_ticks:
        ax.set_xticks(_minor_bjd_ticks, minor=True)
    if _label30_bjd_ticks:
        ax.set_xticks(_label30_bjd_ticks)
        ax.set_xticklabels(_label30_labels, color="navy", fontsize=9)
    ax.tick_params(axis="x", which="major", direction="out", colors="navy", length=6)
    ax.tick_params(axis="x", which="minor", direction="out", colors="navy", length=3)

    # BJD Text (Navy Alpha)
    _xlim = ax.get_xlim()
    _xspan = _xlim[1] - _xlim[0]
    for _i, _tick_v in enumerate(_label30_bjd_ticks):
        if _i == 0: continue
        _xf = (_tick_v - _xlim[0]) / _xspan
        if 0.0 <= _xf <= 1.0:
            ax.text(_xf, 0.01, f"{_tick_v:.4f}", transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=7, color="navy", alpha=0.4,
                    rotation=45, clip_on=True, zorder=5)

    ax.text(0.0, 0.01, "BJD", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, color="navy", zorder=5)

    # Labels
    ax.set_xlabel("Local Time (HH:MM)", color="navy", loc="left", fontsize=9, labelpad=2)
    ax.set_ylabel("Calibrated Magnitude (mag)")

    # Title: Bold, Large (22), pad (10)
    _title_star = getattr(cfg, "display_name", None) or getattr(cfg, "target_name", "Target")
    ax.set_title(f"{_title_star} Light Curve [{channel}]", fontsize=22, fontweight='bold', pad=10)

    # Legend & Date
    _fs = 16
    _obs_str = obs_date if obs_date else ""
    ax.text(0.01, 1.02, _obs_str, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=_fs, color="navy")

    # Legend using V6-style compact parameters
    ax.legend(fontsize=_fs, loc="upper left", frameon=True, edgecolor='gray',
              borderaxespad=0.2, handletextpad=0.0, handlelength=1.0, borderpad=0.2, labelspacing=0.2)

    # Coordinates
    _lat = getattr(cfg, "obs_lat_deg", None)
    _lon = getattr(cfg, "obs_lon_deg", None)
    if _lat is not None and _lon is not None:
        _ls = f"{abs(_lat):.2f}°{'N' if _lat >= 0 else 'S'}"
        _rs = f"{abs(_lon):.2f}°{'E' if _lon >= 0 else 'W'}"
        ax.text(0.99, 1.02, f"{_ls} {_rs}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=_fs, color="#2d6a4f")

    ax.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close("all")
    print(f"[PNG] saved → {out_png}")


def run_photometry_on_wcs_dir(
    wcs_dir: Path,
    out_csv: Path,
    out_png: Path,
    comp_refs: list,
    check_star=None,
    ap_radius: "float | None" = None,
    channel: str = "B",
    shared_cache: "_FrameCompCache | None" = None,
) -> "tuple[pd.DataFrame, dict[str, pd.Series]]":
    """
    Per-frame aperture differential photometry.

    Time system : BJD_TDB (Eastman et al., 2010) at exposure midpoint.
    Zero point  : robust iterative linear regression (see robust_linear_fit).
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

    # ── ⏸ Ensemble 正規化設定（已停用，保留供未來重啟）────────────────────────────
    # 原本邏輯：_ensemble_on=True 時調用 ensemble_normalize() 迭代計算 Δ(t)。
    # 現況：簡化版本已跳過呼叫，直接使用回歸結果（見下文 3659-3700 之簡化版）。
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
    n_low_peak_ratio = 0
    n_low_reg_r2      = 0
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
            "sharpness_index": np.nan,
        }

        # ── 高度角截斷（altitude < ALT_MIN_DEG の幀は除外）────────────────────
        # 大氣消光在低高度角時急增，ALT_MIN_DEG=45° 對應 airmass≈1.41。
        # airmass 已知（非 NaN）時才做截斷；location 未設定時不截斷（airmass=NaN）。
        if np.isfinite(airmass) and airmass > cfg.alt_min_airmass:
            n_skipped += 1
            rec["ok_flag"] = "high_airmass"
            rows.append(rec)
            continue

        # ── [1] 幀層級 FWHM 篩選 ─────────────────────────────────────────────
        # 用 IRAFStarFinder 估算全幀中位數 FWHM，剔除模糊/拖影幀。
        # 注意：DAOStarFinder 不輸出 fwhm 欄，改用 IRAFStarFinder（有 fwhm 欄）。
        _frame_fwhm_median = np.nan
        try:
            _f_mean, _f_med, _f_std = sigma_clipped_stats(img, sigma=3.0, maxiters=3)
            if _f_std > 0:
                _iraf_fwhm = IRAFStarFinder(fwhm=4.0, threshold=5.0 * _f_std)
                _tbl_fwhm = _iraf_fwhm(img - _f_med)
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
        # S = flux(r=3px) / flux(r=8px)（均扣背景，對象：最亮未飽和比較星）
        # S < sharpness_min 代表星點過度擴散或拖影，剔除該幀。
        # 注意：刻意用最亮比較星而非目標星 — 亮星 S/N 高，sharpness 判斷更穩定。
        _sharpness = np.nan
        _sharpness_min = float(getattr(cfg, "sharpness_min", 0.3))
        if _sharpness_min > 0 and comp_refs:
            # 找視場內最亮（m_cat 最小）且未飽和的比較星
            _s_bright_x, _s_bright_y = None, None
            _s_bright_mag = np.inf
            for _sref in comp_refs:
                _s_ra, _s_dec, _s_m = float(_sref[0]), float(_sref[1]), float(_sref[2])
                if not np.isfinite(_s_m) or _s_m >= _s_bright_mag:
                    continue
                _s_xc, _s_yc = radec_to_pixel(wcs_obj, _s_ra, _s_dec)
                if not in_bounds(img, _s_xc, _s_yc, margin=margin):
                    continue
                _s_phot8 = aperture_photometry(img, _s_xc, _s_yc, 8.0, r_in, r_out)
                if is_saturated(_s_phot8.get("max_pix", np.nan), cfg.saturation_threshold):
                    continue
                _s_bright_x, _s_bright_y = _s_xc, _s_yc
                _s_bright_mag = _s_m
            if _s_bright_x is not None:
                try:
                    _phot_s3 = aperture_photometry(img, _s_bright_x, _s_bright_y, 3.0, r_in, r_out)
                    _phot_s8 = aperture_photometry(img, _s_bright_x, _s_bright_y, 8.0, r_in, r_out)
                    if (_phot_s3.get("ok") == 1 and _phot_s8.get("ok") == 1
                            and np.isfinite(_phot_s3.get("flux_net", np.nan))
                            and np.isfinite(_phot_s8.get("flux_net", np.nan))
                            and _phot_s8["flux_net"] > 0):
                        _sharpness = float(_phot_s3["flux_net"] / _phot_s8["flux_net"])
                except Exception:
                    pass
        rec["sharpness_index"] = _sharpness
        if np.isfinite(_sharpness) and _sharpness < _sharpness_min:
            rec["ok"] = 0
            rec["ok_flag"] = "low_sharpness"
            n_low_sharpness += 1
            rows.append(rec)
            continue

        # ── [2b] Peak-ratio 篩選（次鏡起霧 / 甜甜圈 PSF 偵測）─────────────────
        # peak_ratio = t_max_pix / t_flux_net
        # 次鏡起霧時中心被掏空，峰值相對總通量驟降。
        # [DEPRECATED] peak_ratio_min: 固定門檻，已由自適應 peak_ratio_k 取代。
        # peak_ratio_min > 0 仍可運作但不建議使用。
        _peak_ratio_min = float(getattr(cfg, "peak_ratio_min", 0.0))
        # 計算 peak_ratio（重用已有的 phot_t，不重複做 aperture_photometry）
        _flux_pr = phot_t.get("flux_net", np.nan)
        _peak_pr = phot_t.get("max_pix",  np.nan)
        _peak_ratio = float(_peak_pr / _flux_pr) if (
            np.isfinite(_flux_pr) and np.isfinite(_peak_pr) and _flux_pr > 0
        ) else np.nan
        rec["peak_ratio"] = _peak_ratio
        if _peak_ratio_min > 0 and np.isfinite(_peak_ratio) and _peak_ratio < _peak_ratio_min:
            rec["ok"] = 0
            rec["ok_flag"] = "low_peak_ratio"
            n_low_peak_ratio += 1
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
            # 同視野多目標共用快取：命中則跳過重測
            _cached_phot = shared_cache.get(f.stem, ra_c, dec_c) if shared_cache else None
            if _cached_phot is not None:
                phot_c = _cached_phot
            else:
                phot_c = aperture_photometry(img, xc, yc, ap_radius, r_in, r_out)
                if shared_cache is not None:
                    shared_cache.set(f.stem, ra_c, dec_c, phot_c)
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
            # 實測 S/N 加權（暫時停用，待驗證影響後再啟用）
            # _flux_c = phot_c.get("flux_net", 0.0)
            # if _flux_c > 0:
            #     weight *= np.sqrt(_flux_c)
            comp_m_inst.append(m_inst_c)
            comp_m_cat.append(float(m_cat))
            comp_weights.append(weight)

            # ── ⏸ 累積 comp_lightcurves（ensemble 正規化用，已停用）──────────────
            # 原本用途：ensemble_normalize() 需要所有比較星的 m_inst 時間序列。
            # 現況：_ensemble_on 永遠 False，此段無法執行，保留作參考。
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

        fit = robust_linear_fit(
            comp_m_inst, comp_m_cat,
            sigma=cfg.robust_regression_sigma,
            max_iter=cfg.robust_regression_max_iter,
            min_points=cfg.robust_regression_min_points,
            weights=np.asarray(comp_weights, dtype=float),
        )

        # 捕捉第一幀資料供診斷圖使用（全部比較星 + 實際 fit）
        if _first_frame_diag_data is None and len(comp_m_cat) >= 2:
            _first_frame_diag_data = (
                comp_m_cat.copy(), comp_m_inst.copy(), fit,
                float(getattr(cfg, "vmag_approx", np.nan)),
                float(m_inst_t),
            )

        # ── 校正擬合：自由斜率回歸（v1.5 邏輯，使用者要求）────────────────
        # 參考：Paxson 2010 JAAVSO；比較星色彩多樣時 slope=1 假設不成立。
        # 目標星在比較星星等範圍內 → 插值，不外插。
        if fit is not None:
            a      = fit["a"]
            b      = fit["b"]
            r2     = fit["r2"]
            mask   = fit["mask"]
            _a_fit = a
        else:
            a, b, r2, _a_fit = np.nan, np.nan, np.nan, np.nan
            mask = np.zeros(len(comp_m_inst), dtype=bool)

        # ── [DISABLED — 使用者要求停用] 純差分測光（slope=1.0）────────────
        # v1.6D 在 context 超載情況下誤加，與使用者方法不符，已停用。
        # 使用者方法：自由斜率回歸校正（非 slope=1 純差分）。
        # mask = np.isfinite(comp_m_inst) & np.isfinite(comp_m_cat)
        # zp   = float(np.nanmedian(comp_m_cat[mask] - comp_m_inst[mask])) if mask.any() else np.nan
        # a, b = 1.0, -zp
        # if fit is not None:
        #     _a_fit, _r2_fit = fit["a"], fit["r2"]
        #     mask = fit["mask"]
        #     zp = float(np.nanmedian(comp_m_cat[mask] - comp_m_inst[mask])) if mask.any() else zp
        #     a, b = 1.0, -zp
        #     r2 = _r2_fit
        # else:
        #     _a_fit, r2 = np.nan, np.nan

        if not (np.isfinite(a) and np.isfinite(b)):
            rows.append(rec)
            continue

        # ── [3] 回歸 R² 幀層級篩選 ────────────────────────────────────────────
        # reg_r2_min 預設 0.0（停用）；> 0 時才自動剔除，保留 WARN 輸出。
        _reg_r2_min = float(getattr(cfg, "reg_r2_min", 0.0))
        if np.isfinite(r2) and _reg_r2_min > 0 and r2 < _reg_r2_min:
            _phot_logger.warning(
                "[WARN] low reg R2=%.4f < %.4f in %s (channel=%s)",
                r2, _reg_r2_min, f.name, channel,
            )
            rec["reg_r2"] = r2
            rec["ok"] = 0
            rec["ok_flag"] = "low_reg_r2"
            n_low_reg_r2 += 1
            rows.append(rec)
            continue
        elif np.isfinite(r2) and r2 < max(_reg_r2_min, 0.5):
            _phot_logger.warning(
                "[WARN] low reg R2=%.4f in %s (channel=%s) — consider raising reg_r2_min",
                r2, f.name, channel,
            )

        m_var       = (m_inst_t - b) / a
        # ── 外插檢查：目標星儀器星等是否在比較星範圍內 ─────────────────────
        _comp_m_inst_used = comp_m_inst[mask]
        _flag_extrap = 0
        if len(_comp_m_inst_used) >= 2:
            _m_lo, _m_hi = float(np.nanmin(_comp_m_inst_used)), float(np.nanmax(_comp_m_inst_used))
            if np.isfinite(m_inst_t) and (m_inst_t < _m_lo or m_inst_t > _m_hi):
                _flag_extrap = 1
                _phot_logger.warning(
                    "[EXTRAP] target m_inst=%.3f outside comp range [%.3f, %.3f] in %s",
                    m_inst_t, _m_lo, _m_hi, f.name,
                )
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

        # ── FLAG_SLOPE_DEVIATION（§3.8）：regression slope 監控 ────────────
        # a 是自由斜率回歸的 slope；偏離 1.0 超過 0.05 時記錄警告。
        _slope_flag = int(np.isfinite(_a_fit) and abs(_a_fit - 1.0) > 0.05)
        if _slope_flag:
            _phot_logger.debug(
                "[FLAG] slope_deviation a_fit=%.4f (|a-1|=%.4f > 0.05) in %s",
                _a_fit, abs(_a_fit - 1.0), f.name,
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
            "reg_slope": a,           # 自由斜率回歸 slope
            "reg_intercept": b,       # 自由斜率回歸 intercept
            "reg_r2": r2,             # regression R²（監控用）
            "comp_used": comp_used,
            "flag_slope_dev": _slope_flag,
            "reg_slope_fit": _a_fit,  # regression slope（監控用，取代 flag_extrapolated）
            "flag_extrapolated": _flag_extrap,
            "m_var": m_var,
        })

        # ── 零點殘差 RMS（用於診斷圖）────────────────────────────────────────
        _m_cat_fit  = comp_m_cat[mask]
        _m_inst_fit = comp_m_inst[mask]
        _m_inst_pred = a * _m_cat_fit + b   # 預測 m_inst（擬合方向：m_inst = a*m_cat + b）
        _residuals   = _m_inst_fit - _m_inst_pred
        reg_resid_rms = float(np.sqrt(np.mean(_residuals ** 2))) if len(_residuals) > 0 else np.nan
        rec["reg_residual_rms"] = reg_resid_rms

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

    # ── 大氣消光一階改正（optional）──────────────────────────────────────────
    # m_var_ext = m_var - k × (X_frame - X_ref)
    # k = extinction_k (mag/airmass)，X_ref = 觀測中最小氣團
    # 差分測光中目標與比較星消光大致抵消，但長基線或大氣團差時殘差累積。
    _ext_k = float(getattr(cfg, "extinction_k", 0.0))
    if _ext_k > 0 and "airmass" in df.columns:
        _ok_am = (df["ok"] == 1) & np.isfinite(df["airmass"]) & np.isfinite(df["m_var"])
        if _ok_am.sum() >= 3:
            _X_ref = float(df.loc[_ok_am, "airmass"].min())
            _delta_ext = _ext_k * (df["airmass"] - _X_ref)
            df["m_var_raw"] = df["m_var"].copy()
            df.loc[_ok_am, "m_var"] = df.loc[_ok_am, "m_var"] - _delta_ext[_ok_am]
            _med_corr = float(_delta_ext[_ok_am].median())
            print(f"[extinction] k={_ext_k:.3f} mag/airmass  X_ref={_X_ref:.3f}  "
                  f"median correction={_med_corr:.4f} mag  ({_ok_am.sum()} frames)")

    # ── 回歸截距突變篩選 ───────────────────────────────────────────────────────
    # 薄雲或透明度驟變時所有比較星同步變暗，零點截距 b 會系統性漂移。
    # 滾動中位數（±5 幀）捕捉短時突變，偏離超過 reg_intercept_sigma × MAD 則剔除。
    n_reg_jump = 0
    _reg_intercept_sigma = float(getattr(cfg, "reg_intercept_sigma", 0.0))
    if _reg_intercept_sigma > 0 and "reg_intercept" in df.columns:
        _reg_col = df["reg_intercept"].copy()
        _reg_roll_med = _reg_col.rolling(window=11, center=True, min_periods=3).median()
        _reg_resid = (_reg_col - _reg_roll_med).abs()
        _reg_mad = float(np.nanmedian(_reg_resid[df["ok"] == 1]))
        _reg_thresh = _reg_intercept_sigma * 1.4826 * _reg_mad if _reg_mad > 0 else np.inf
        _reg_jump_mask = (df["ok"] == 1) & (_reg_resid > _reg_thresh)
        if _reg_jump_mask.any():
            df.loc[_reg_jump_mask, "ok"] = 0
            if "ok_flag" not in df.columns:
                df["ok_flag"] = ""
            df.loc[_reg_jump_mask, "ok_flag"] = "reg_jump"
            n_reg_jump = int(_reg_jump_mask.sum())
            print(f"[reg_jump] MAD={_reg_mad:.4f}  thresh={_reg_thresh:.4f}  "
                  f"clipped {n_reg_jump} frames")

    # ── 天空背景突升篩選 ──────────────────────────────────────────────────────
    # 起霧或散射光使目標孔徑背景環中位數升高，t_b_sky 突升可做為霧的早期指標。
    # 滾動中位數（±5 幀）捕捉短時突變，偏離超過 sky_sigma × MAD 則剔除。
    n_high_sky = 0
    _sky_sigma = float(getattr(cfg, "sky_sigma", 0.0))
    if _sky_sigma > 0 and "t_b_sky" in df.columns:
        _sky_col = df["t_b_sky"].copy()
        _sky_roll_med = _sky_col.rolling(window=11, center=True, min_periods=3).median()
        _sky_resid = _sky_col - _sky_roll_med  # 只看正向突升
        _sky_mad = float(np.nanmedian(_sky_resid[df["ok"] == 1].abs()))
        _sky_thresh = _sky_sigma * 1.4826 * _sky_mad if _sky_mad > 0 else np.inf
        _sky_jump_mask = (df["ok"] == 1) & (_sky_resid > _sky_thresh)
        if _sky_jump_mask.any():
            df.loc[_sky_jump_mask, "ok"] = 0
            if "ok_flag" not in df.columns:
                df["ok_flag"] = ""
            df.loc[_sky_jump_mask, "ok_flag"] = "high_sky"
            n_high_sky = int(_sky_jump_mask.sum())
            print(f"[high_sky] MAD={_sky_mad:.4f}  thresh={_sky_thresh:.4f}  "
                  f"clipped {n_high_sky} frames")

    # ── peak_ratio 自適應篩選 ─────────────────────────────────────────────────
    # 自適應門檻，不依賴絕對值，可移植至不同望遠鏡/相機。
    # peak_ratio 極低代表 PSF 中心被挖空（次鏡起霧甜甜圈）。
    # 門檻 = median(peak_ratio) - peak_ratio_k × MAD，整夜所有幀一起算。
    n_low_peak_ratio_adaptive = 0
    _peak_ratio_k = float(getattr(cfg, "peak_ratio_k", 0.0))
    if _peak_ratio_k > 0 and "peak_ratio" in df.columns:
        _pr_vals = df.loc[df["ok"] == 1, "peak_ratio"].dropna()
        if len(_pr_vals) >= 5:
            _pr_med = float(np.median(_pr_vals))
            _pr_mad = float(np.median(np.abs(_pr_vals - _pr_med)))
            _pr_thresh = _pr_med - _peak_ratio_k * 1.4826 * _pr_mad
            _pr_mask = (df["ok"] == 1) & (df["peak_ratio"].notna()) & (df["peak_ratio"] < _pr_thresh)
            if _pr_mask.any():
                df.loc[_pr_mask, "ok"] = 0
                if "ok_flag" not in df.columns:
                    df["ok_flag"] = ""
                df.loc[_pr_mask, "ok_flag"] = "low_peak_ratio"
                n_low_peak_ratio_adaptive = int(_pr_mask.sum())
                print(f"[peak_ratio adaptive] median={_pr_med:.4f}  MAD={_pr_mad:.4f}  "
                      f"thresh={_pr_thresh:.4f}  clipped {n_low_peak_ratio_adaptive} frames")

    # ── 剔除統計表 ────────────────────────────────────────────────────────────
    # 統計結果同步存檔（帶時間戳，不覆蓋），供調整篩選閾值時對照各版本剔除效果
    _n_total_fits  = len(wcs_files_sorted)
    _n_in_df       = len(df)
    _n_ok_final    = int((df["ok"] == 1).sum())
    _n_alt_skip    = n_skipped  # alt_too_low（未進 df）
    _n_sigma_clip  = int(((df["ok"] == 0) & (df.get("ok_flag", pd.Series("", index=df.index)) == "sigma_clip")).sum()) if "ok_flag" in df.columns else (_n_before_clip - _n_after_clip)

    _n_high_fwhm_val     = n_high_fwhm
    _n_low_sharpness_val = n_low_sharpness
    _n_low_peak_ratio_val = n_low_peak_ratio
    _n_low_reg_r2_val     = n_low_reg_r2
    # 重新計算 _n_phot_fail：排除所有已命名篩選計數
    _n_qual_filtered = (_n_high_fwhm_val + _n_low_sharpness_val
                        + _n_low_peak_ratio_val + _n_low_reg_r2_val
                        + n_reg_jump + n_high_sky + n_low_peak_ratio_adaptive)
    _n_phot_fail   = _n_in_df - _n_ok_final - _n_sigma_clip - _n_qual_filtered

    _sep = "-" * 68
    print(f"\n[剔除統計] 通道 {channel}  {getattr(cfg, 'target_name', '')}  {str(out_csv.stem).split('_')[-1]}")
    print(_sep)
    print(f"  {'原因':<24} {'幀數':>6}    公式")
    print(_sep)
    print(f"  {'高氣團跳過':<24} {_n_alt_skip:>6}    airmass > {cfg.alt_min_airmass:.2f}")
    print(f"  {'高 FWHM 幀剔除':<24} {_n_high_fwhm_val:>6}    frame_fwhm_median > {cfg.comp_fwhm_max:.1f} px")
    print(f"  {'低 Sharpness 剔除':<24} {_n_low_sharpness_val:>6}    S=flux(r=3)/flux(r=8) < {float(getattr(cfg,'sharpness_min',0.3)):.2f}")
    print(f"  {'低 Peak Ratio 剔除':<24} {_n_low_peak_ratio_val:>6}    peak/flux < {float(getattr(cfg,'peak_ratio_min',0.0)):.3f} (0=停用)")
    print(f"  {'低 reg R2 剔除':<24} {_n_low_reg_r2_val:>6}    reg_r2 < {float(getattr(cfg,'reg_r2_min',0.0)):.2f} (0=停用)")
    print(f"  {'孔徑/WCS/邊界失敗':<24} {_n_phot_fail:>6}    flux/位置無效")
    print(f"  {'sigma_clip':<24} {_n_sigma_clip:>6}    |m_var - median| > 3 * 1.4826 * MAD")
    print(f"  {'回歸截距突變':<24} {n_reg_jump:>6}    rolling median ± {float(getattr(cfg,'reg_intercept_sigma',0.0)):.1f} MAD (0=停用)")
    print(f"  {'天空背景突升':<24} {n_high_sky:>6}    rolling median + {float(getattr(cfg,'sky_sigma',0.0)):.1f} MAD (0=停用)")
    print(f"  {'Peak Ratio 自適應':<24} {n_low_peak_ratio_adaptive:>6}    median - {float(getattr(cfg,'peak_ratio_k',0.0)):.1f} MAD (0=停用)")
    print(_sep)
    print(f"  {'保留 (ok=1)':<24} {_n_ok_final:>6} / {_n_total_fits} 幀")
    print()

    # 存檔：帶時間戳，不覆蓋
    _rej_rows = [
        {"reason": "高氣團跳過",       "count": _n_alt_skip,            "threshold": f"airmass > {cfg.alt_min_airmass:.2f}",                              "config_key": "max_airmass",         "config_value": cfg.alt_min_airmass},
        {"reason": "高 FWHM 幀剔除",   "count": _n_high_fwhm_val,       "threshold": f"fwhm > {cfg.comp_fwhm_max:.1f} px",                                "config_key": "max_fwhm_px",         "config_value": cfg.comp_fwhm_max},
        {"reason": "低 Sharpness 剔除", "count": _n_low_sharpness_val,   "threshold": f"S < {float(getattr(cfg,'sharpness_min',0.3)):.2f}",                 "config_key": "sharpness_min",       "config_value": float(getattr(cfg, "sharpness_min", 0.3))},
        {"reason": "低 Peak Ratio 剔除","count": _n_low_peak_ratio_val,  "threshold": f"peak/flux < {float(getattr(cfg,'peak_ratio_min',0.0)):.3f}",        "config_key": "peak_ratio_min",      "config_value": float(getattr(cfg, "peak_ratio_min", 0.0))},
        {"reason": "低 reg R2 剔除",    "count": _n_low_reg_r2_val,       "threshold": f"reg_r2 < {float(getattr(cfg,'reg_r2_min',0.0)):.2f}",                 "config_key": "reg_r2_min",           "config_value": float(getattr(cfg, "reg_r2_min", 0.0))},
        {"reason": "孔徑/WCS/邊界失敗", "count": _n_phot_fail,           "threshold": "flux/位置無效",                                                     "config_key": "—",                   "config_value": "—"},
        {"reason": "sigma_clip",        "count": _n_sigma_clip,          "threshold": "|m_var - median| > 3 * 1.4826 * MAD",                               "config_key": "—",                   "config_value": "—"},
        {"reason": "回歸截距突變",       "count": n_reg_jump,              "threshold": f"rolling |reg_intercept - med| > {float(getattr(cfg,'reg_intercept_sigma',0.0)):.1f} MAD", "config_key": "reg_intercept_sigma", "config_value": float(getattr(cfg, "reg_intercept_sigma", 0.0))},
        {"reason": "天空背景突升",       "count": n_high_sky,                    "threshold": f"rolling (t_b_sky - med) > {float(getattr(cfg,'sky_sigma',0.0)):.1f} MAD",                     "config_key": "sky_sigma",           "config_value": float(getattr(cfg, "sky_sigma", 0.0))},
        {"reason": "Peak Ratio 自適應", "count": n_low_peak_ratio_adaptive,    "threshold": f"peak_ratio < median - {float(getattr(cfg,'peak_ratio_k',0.0)):.1f} * 1.4826 * MAD",           "config_key": "peak_ratio_k",        "config_value": float(getattr(cfg, "peak_ratio_k", 0.0))},
        {"reason": "保留 (ok=1)",       "count": _n_ok_final,                   "threshold": f"/ {_n_total_fits} 幀",                                                                         "config_key": "—",                   "config_value": "—"},
    ]
    _rej_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _rej_stem = out_csv.stem  # e.g. photometry_G1_20251220
    _rej_path = out_csv.parent / f"rejection_stats_{_rej_stem}_{_rej_ts}.csv"
    pd.DataFrame(_rej_rows).to_csv(_rej_path, index=False, encoding="utf-8-sig")
    print(f"[剔除統計] saved → {_rej_path}")

    # 第一次寫入（ensemble normalize 後會再寫一次，更新 m_var_norm/delta_ensemble）
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    n_written = len(df)
    print(f"[CSV] saved → {out_csv}  "
          f"({n_written} rows written, {_n_after_clip} successful "
          f"[{_n_before_clip - _n_after_clip} sigma-clipped], "
          f"{n_skipped} skipped [airmass > {cfg.alt_min_airmass:.3f}])")

    # ── 回歸診斷總覽圖：回歸散佈圖（主）+ RMS 時序（輔）──────────────────────
    if cfg.save_regression_diagnostic:
        try:
            _reg_diag_path = cfg.regression_diag_dir / (
                f"reg_overview_{out_csv.stem.split('_')[1]}_{out_csv.stem.split('_', 1)[-1]}.png"
            )
            _df_ok = df[df["ok"] == 1].copy()
            import datetime as _dt_reg
            import matplotlib.ticker as _mticker_reg
            import matplotlib.gridspec as _mgs

            fig_diag = plt.figure(figsize=(8, 11))
            gs = _mgs.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.30)
            ax_sc = fig_diag.add_subplot(gs[0])
            ax_sc.set_box_aspect(1.0)   # 正方形圖框
            ax_ts = fig_diag.add_subplot(gs[1])

            # ── 上：第一幀回歸散佈圖（全部比較星，正方形）────────────────────
            _ffd = _first_frame_diag_data
            if _ffd is not None:
                _mc, _mi, _fit = _ffd[:3]
                _tgt_vmag_diag = _ffd[3] if len(_ffd) > 3 else np.nan
                _tgt_minst     = _ffd[4] if len(_ffd) > 4 else np.nan
                _ok_pts = np.isfinite(_mc) & np.isfinite(_mi)
                ax_sc.scatter(_mc[_ok_pts], _mi[_ok_pts], s=18, alpha=0.6,
                              color="steelblue", label=f"comp (n={_ok_pts.sum()})")
                if _fit and np.isfinite(_fit.get("a", np.nan)):
                    _a, _b, _r2 = _fit["a"], _fit["b"], _fit.get("r2", np.nan)
                    _xl = np.linspace(_mc[_ok_pts].min(), _mc[_ok_pts].max(), 100)
                    ax_sc.plot(_xl, _a * _xl + _b, "r-", lw=1.8,
                               label=f"$m_{{inst}}={_a:.3f}\\,m_{{cat}}+({_b:.3f})$  $R^2={_r2:.3f}$")
                # 目標星紅圈
                if _fit and np.isfinite(_tgt_minst):
                    _a_d, _b_d = _fit["a"], _fit["b"]
                    if np.isfinite(_a_d) and _a_d != 0:
                        _tgt_mvar = (_tgt_minst - _b_d) / _a_d
                        ax_sc.scatter([_tgt_mvar], [_tgt_minst], s=100, marker="o",
                                      facecolors="none", edgecolors="red", linewidths=2.0,
                                      zorder=5, label=f"target  $m_{{var}}$={_tgt_mvar:.1f}")
                # x 軸：左界=比較星+目標星中最亮，右界=comp_mag_max
                _brightest_comp = float(_mc[_ok_pts].min()) if _ok_pts.any() else float(cfg.comp_mag_min)
                _x_lo = _brightest_comp
                if _fit and np.isfinite(_tgt_vmag_diag):
                    _x_lo = min(_x_lo, float(_tgt_vmag_diag))
                _x_hi = float(cfg.comp_mag_max)
                ax_sc.set_xlim(_x_lo - 0.2, _x_hi + 0.1)
                # 外插區域：回歸線用虛線
                if _fit and np.isfinite(_fit.get("a", np.nan)) and _ok_pts.any():
                    _comp_lo = float(_mc[_ok_pts].min())
                    _comp_hi = float(_mc[_ok_pts].max())
                    # 亮端外插（比較星最亮 → x 左界）
                    if _x_lo < _comp_lo - 0.01:
                        _xl_ext = np.linspace(_x_lo - 0.2, _comp_lo, 50)
                        ax_sc.plot(_xl_ext, _a * _xl_ext + _b, "r--", lw=1.2, alpha=0.5)
                    # 暗端外插（比較星最暗 → x 右界）
                    if _x_hi > _comp_hi + 0.01:
                        _xl_ext = np.linspace(_comp_hi, _x_hi + 0.1, 50)
                        ax_sc.plot(_xl_ext, _a * _xl_ext + _b, "r--", lw=1.2, alpha=0.5)
                # y 軸：緊貼資料 + 目標星 + 回歸線在 x 範圍內的值
                _all_mi = list(_mi[_ok_pts]) if _ok_pts.any() else []
                if _fit and np.isfinite(_tgt_minst):
                    _all_mi.append(float(_tgt_minst))
                if _fit and np.isfinite(_fit.get("a", np.nan)):
                    _all_mi.append(float(_fit["a"] * _x_lo + _fit["b"]))
                    _all_mi.append(float(_fit["a"] * _x_hi + _fit["b"]))
                if _all_mi:
                    _mi_lo, _mi_hi = min(_all_mi), max(_all_mi)
                    _mi_pad = (_mi_hi - _mi_lo) * 0.12
                    ax_sc.set_ylim(_mi_hi + _mi_pad, _mi_lo - _mi_pad)  # inverted
                else:
                    ax_sc.invert_yaxis()
                ax_sc.set_xlabel(f"$m_{{cat}}$ ({cfg.phot_band})", fontsize=11)
                ax_sc.set_ylabel("$m_{inst}$", fontsize=11)
                _extrap_note = ""
                if _ok_pts.any() and np.isfinite(_tgt_vmag_diag):
                    if _tgt_vmag_diag < float(_mc[_ok_pts].min()):
                        _extrap_note = "  [EXTRAPOLATION]"
                ax_sc.set_title(
                    f"Regression Fit — Frame 1  ({_x_lo:.1f}–{_x_hi:.1f} mag){_extrap_note}",
                    fontsize=12,
                )
                ax_sc.legend(fontsize=8, frameon=False)
                ax_sc.grid(True, alpha=0.3)
            else:
                ax_sc.text(0.5, 0.5, "No first-frame data",
                           transform=ax_sc.transAxes, ha="center", va="center")

            # ── 下：RMS 時序圖（參考用）───────────────────────────────────────
            if "reg_residual_rms" in _df_ok.columns and np.isfinite(_df_ok["reg_residual_rms"]).any():
                ax_ts.plot(
                    _df_ok[time_key], _df_ok["reg_residual_rms"],
                    "o-", ms=2, lw=0.6, color="steelblue", alpha=0.7
                )
                ax_ts.axhline(
                    _df_ok["reg_residual_rms"].median(), color="red",
                    lw=1, ls="--", label=f"median={_df_ok['reg_residual_rms'].median():.4f}"
                )
                ax_ts.set_xlabel(time_key.upper(), fontsize=9)
                ax_ts.set_ylabel("Residual RMS (mag)", fontsize=9)
                ax_ts.legend(fontsize=7)
                ax_ts.grid(True, alpha=0.3)
                ax_ts.tick_params(labelsize=8)
                ax_ts.xaxis.set_major_formatter(
                    _mticker_reg.FuncFormatter(lambda v, _: f"{v:.2f}")
                )
            else:
                ax_ts.text(0.5, 0.5, "No residual data", transform=ax_ts.transAxes,
                           ha="center", va="center")

            fig_diag.suptitle(
                f"Regression Diagnostic  |  {cfg.target_name}  "
                f"channel={channel}  {out_csv.stem}",
                fontsize=11
            )
            fig_diag.savefig(_reg_diag_path, dpi=150, bbox_inches="tight")
            plt.close(fig_diag)
            print(f"[reg_diag] saved → {_reg_diag_path}")
        except Exception as _e:
            import traceback; traceback.print_exc()
            print(f"[WARN] regression diagnostic failed: {_e}")

    # ── Rejection timeline 圖 ────────────────────────────────────────────────
    try:
        _rej_colors = {
            "high_airmass":  "#aaaaaa",
            "high_fwhm":     "#e67e22",
            "low_sharpness": "#9b59b6",
            "low_peak_ratio":"#1abc9c",
            "low_reg_r2":     "#e74c3c",
            "phot_fail":     "#c0392b",
            "sigma_clip":    "#2980b9",
            "reg_jump":       "#f39c12",
            "high_sky":      "#16a085",
            "ok":            "#cccccc",
        }
        _rej_fig, _rej_ax = plt.subplots(figsize=(12, 3))
        _df_all = df.copy()
        _df_all["_flag"] = _df_all["ok_flag"].fillna("ok").where(_df_all["ok"] == 0, "ok")
        # 用 bjd_tdb 轉本地時間（UTC+8）
        _tz_h = float(getattr(cfg, "tz_offset_hours", 8))
        if time_key in _df_all.columns and _df_all[time_key].notna().any():
            _t_local = (_df_all[time_key] - 2400000.5) * 86400  # MJD → 秒，只取相對值
            _t0 = _t_local.min()
            _t_rel = (_t_local - _t0) / 3600  # 相對小時
        else:
            _t_rel = pd.Series(range(len(_df_all)), dtype=float)
        for _flag, _grp in _df_all.groupby("_flag"):
            _col = _rej_colors.get(_flag, "#888888")
            _size = 15 if _flag == "ok" else 40
            _zord = 2 if _flag == "ok" else 3
            _idx = _df_all.index[_df_all["_flag"] == _flag]
            _rej_ax.scatter(_t_rel.iloc[_idx], _df_all.loc[_df_all["_flag"] == _flag, "airmass"],
                            c=_col, s=_size, label=_flag, zorder=_zord, alpha=0.85)
        _rej_ax.set_xlabel("Elapsed time (hours from first frame)")
        _rej_ax.set_ylabel("Airmass")
        _rej_ax.invert_yaxis()
        _rej_ax.set_title(
            f"Rejection timeline  |  {cfg.target_name}  ch={cfg.phot_band}  {out_csv.stem}",
            fontsize=9)
        _rej_ax.legend(fontsize=7, ncol=5, loc="upper right")
        _rej_ax.grid(True, alpha=0.25)
        _rej_fig.tight_layout()
        _rej_path = out_csv.parent / f"rejection_timeline_{out_csv.stem}.png"
        _rej_fig.savefig(_rej_path, dpi=150)
        plt.close(_rej_fig)
        print(f"[剔除時序圖] saved → {_rej_path}")
    except Exception as _e:
        print(f"[WARN] 剔除時序圖輸出失敗：{_e}")

    # ── Plot light curve ──────────────────────────────────────────────────────
    d = df[(df["ok"] == 1) & np.isfinite(df[time_key]) & np.isfinite(df["m_var"])].copy()
    if len(d) == 0:
        print("[PLOT] No valid photometry points to plot.")
        df["m_var_norm"] = df["m_var"]
        df["delta_ensemble"] = np.nan
        return df, {}

    import matplotlib.ticker as mticker
    # ATime 已在行 2950 附近作為 local 變數使用，不在此重新 import（會遮蔽）

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
            import datetime as _dt_local
            t_local = t_utc + _dt_local.timedelta(hours=_tz_offset_h)
            return t_local
        except Exception:
            return None

    # 計算 Local Time 刻度位置（BJD）
    _t_local_min = _bjd_to_local_hm(bjd_min)
    _t_local_max = _bjd_to_local_hm(bjd_max)

    _major_bjd_ticks = []   # 整點
    _label30_bjd_ticks = []  # 整點 + 半點（有數字標示）
    _label30_labels = []
    _minor_bjd_ticks = []   # 每 10 分鐘
    if _t_local_min is not None and _t_local_max is not None:
        import datetime as _dt_local
        # 每 10 分鐘刻度，收集整點 / 半點 / 其他
        _cur = _t_local_min.replace(second=0, microsecond=0)
        _min_round = (_cur.minute // 10) * 10
        _cur = _cur.replace(minute=_min_round)
        while _cur <= _t_local_max + _dt_local.timedelta(minutes=1):
            _utc = _cur - _dt_local.timedelta(hours=_tz_offset_h)
            _bjd_tick = ATime(_utc).jd
            _minor_bjd_ticks.append(_bjd_tick)
            if _cur.minute == 0:
                _major_bjd_ticks.append(_bjd_tick)
            if _cur.minute in (0, 30):
                _label30_bjd_ticks.append(_bjd_tick)
                _label30_labels.append(_cur.strftime("%H:%M"))
            _cur += _dt_local.timedelta(minutes=10)

    _lc_obs_date = str(out_csv.stem).split("_")[-1]
    # 光變曲線圖移至 ensemble normalize 之後繪製（使用 m_var_norm）

    # ── Check star residuals ──────────────────────────────────────────────────
    if check_star is not None and "k_minus_c" in df.columns:
        k = df[(df["ok"] == 1) & np.isfinite(df[time_key])
               & np.isfinite(df["k_minus_c"])].copy()
        if len(k) > 1:
            _t0_check = float(k[time_key].min())
            k["t_rel_d"] = k[time_key] - _t0_check
            sigma_k = float(np.nanstd(k["k_minus_c"]))
            flag    = " [!] EXCEEDS THRESHOLD" if sigma_k > cfg.check_star_max_sigma else ""
            print(f"[CHECK] K-C sigma = {sigma_k:.5f} mag "
                  f"(threshold = {cfg.check_star_max_sigma}){flag}")
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            ax2.plot(k["t_rel_d"], k["k_minus_c"], "o-", ms=4, lw=0.8)
            ax2.xaxis.set_major_formatter(plt.FuncFormatter(
                lambda x, _: f"{int(x)}d {int((x % 1)*24)}h {int(((x % 1)*24 % 1)*60)}m"
            ))
            ax2.invert_yaxis()
            ax2.set_xlabel("t_rel (days)")
            ax2.set_ylabel("K - C  (mag)")
            ax2.set_title("Check Star Validation")
            ax2.grid(True, alpha=0.4)
            fig2.tight_layout()
            plt.close("all")

    # ── ⏸ 待決定：Broeg (2005) ensemble 正規化 ─────────────────────────────────
    # 理由：逐幀自由斜率回歸已消除逐幀大氣，ensemble 重複修正可能引入雜訊。
    # 待驗：移除後 R² 是否有變化（預期無）。
    # 原本邏輯：_ensemble_on=True 時迭代算 Δ(t)，然後 m_var_norm = m_var - Δ(t)。
    # 簡化版本：關閉 ensemble，m_var_norm 直接 = m_var（回歸後的校正星等）。
    """
    # ── 原始 ensemble 迴圈（已停用，整塊保留作參考）────────────────────────────────
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
    """
    # ── 簡化版本：直接用回歸結果 ────────────────────────────────────────────────
    df["m_var_norm"] = df["m_var"]        # 直通回歸的 m_var，不加 ensemble 修正
    df["delta_ensemble"] = np.nan         # 預留欄位，暫不計算

    # ── 光變曲線圖（ensemble 之後重畫，使用 m_var_norm）──────────────────────
    plot_light_curve(df, out_png, channel, cfg, obs_date=_lc_obs_date)

    # ── 單通道摘要報告 ──────────────────────────────────────────────────────────
    try:
        _ok_df = df[df["ok"] == 1]
        _n_ok = len(_ok_df)
        _n_total = len(df)
        _summary_lines = [
            f"=== VarStar Photometry Summary ===",
            f"Target    : {cfg.target_name}",
            f"Channel   : {channel}",
            f"Date      : {out_csv.stem.split('_')[-1]}",
            f"Frames    : {_n_ok}/{_n_total} ok  "
            f"({n_skipped} airmass-cut, "
            f"{_n_before_clip - _n_after_clip} sigma-clipped)",
            f"Aperture  : {ap_radius:.1f} px  "
            f"(annulus {r_in:.1f}–{r_out:.1f})",
        ]
        _summary_lines.append(
            f"GrowthFr  : {getattr(cfg, 'aperture_growth_fraction', float('nan')):.3f}"
        )
        if _n_ok > 0:
            _mvar = _ok_df["m_var_norm"] if "m_var_norm" in _ok_df.columns else _ok_df["m_var"]
            _mvar_ok = _mvar[np.isfinite(_mvar)]
            if len(_mvar_ok) > 0:
                _summary_lines.append(
                    f"Mag range : {_mvar_ok.min():.3f} – {_mvar_ok.max():.3f}  "
                    f"(median {_mvar_ok.median():.3f}, std {_mvar_ok.std():.4f})"
                )
            if "reg_r2" in _ok_df.columns:
                _r2_ok = _ok_df["reg_r2"][np.isfinite(_ok_df["reg_r2"])]
                if len(_r2_ok) > 0:
                    _summary_lines.append(
                        f"Reg R²    : median {_r2_ok.median():.4f}  "
                        f"min {_r2_ok.min():.4f}  max {_r2_ok.max():.4f}"
                    )
            if "airmass" in _ok_df.columns:
                _am_ok = _ok_df["airmass"][np.isfinite(_ok_df["airmass"])]
                if len(_am_ok) > 0:
                    _summary_lines.append(
                        f"Airmass   : {_am_ok.min():.3f} – {_am_ok.max():.3f}"
                    )
            if "t_snr" in _ok_df.columns:
                _snr_ok = _ok_df["t_snr"][np.isfinite(_ok_df["t_snr"])]
                if len(_snr_ok) > 0:
                    _summary_lines.append(
                        f"SNR       : median {_snr_ok.median():.1f}  "
                        f"min {_snr_ok.min():.1f}"
                    )
            if "comp_used" in _ok_df.columns:
                _comp_ok = _ok_df["comp_used"][np.isfinite(_ok_df["comp_used"])]
                if len(_comp_ok) > 0:
                    _summary_lines.append(
                        f"Comp stars: median {_comp_ok.median():.0f}  "
                        f"min {_comp_ok.min():.0f}"
                    )
            if _ext_k > 0:
                _summary_lines.append(
                    f"Extinction: k={_ext_k:.3f} mag/airmass applied"
                )
            _baseline_hr = 0.0
            if time_key in _ok_df.columns:
                _t_ok = _ok_df[time_key][np.isfinite(_ok_df[time_key])]
                if len(_t_ok) > 1:
                    _baseline_hr = (_t_ok.max() - _t_ok.min()) * 24
                    _summary_lines.append(
                        f"Baseline  : {_baseline_hr:.2f} hr"
                    )
        _summary_lines.append(f"Output    : {out_csv}")
        _summary_lines.append("")

        _summary_text = "\n".join(_summary_lines)
        _summary_path = out_csv.parent / f"summary_{out_csv.stem}.txt"
        _summary_path.write_text(_summary_text, encoding="utf-8")
        print(f"[摘要] saved → {_summary_path}")
        for _sl in _summary_lines:
            if _sl:
                print(f"  {_sl}")
    except Exception as _e_summary:
        print(f"[WARN] 摘要報告輸出失敗：{_e_summary}")

    return df, {}  # comp_lightcurves 已隨 ensemble 停用，回傳空 dict


# ── 週期分析已移至 period_analysis.py（統一從 YAML 讀取參數）────────────────
# 原內建 LS + Fourier 程式碼已保留於 photometry_ls_legacy.py，供手動使用。
# 移出日期：2026-03-21

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
    _parser.add_argument("--split_subdir", default="splits",
                         help="split 子目錄名稱（例如 splits_raw）")
    _parser.add_argument("--all", action="store_true",
                         help="處理 yaml 中所有 obs_sessions 的全部目標")
    _parser.add_argument("--out-tag", default=None, dest="out_tag",
                         help="輸出子目錄後綴，例如 sat65536 → output_sat65536（不填則用 output）")
    _parser.add_argument("--no-vsx", action="store_true", dest="no_vsx",
                         help="跳過 VSX 額外目標星測光，只跑主目標")
    _parser.add_argument("--raw", action="store_true",
                         help="使用 raw 未校正影像（splits_raw/）")
    _args = _parser.parse_args()
    if _args.raw:
        _args.split_subdir = "splits_raw"
        if not _args.out_tag:
            _args.out_tag = "raw"

    # ── Logger 初始化 ──────────────────────────────────────────────────────────
    _log_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    if _args.out_tag:
        _log_ts = f"{_log_ts}_{_args.out_tag}"
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
    if _args.all:
        _targets_list = [
            (str(_t), str(_sess["date"]))
            for _sess in _yaml.get("obs_sessions", [])
            for _t in _sess.get("targets", [])
        ]
    elif _args.target is None and _args.date is not None:
        # 只指定日期：跑該場次所有目標
        _targets_list = []
        for _sess in _yaml.get("obs_sessions", []):
            if str(_sess["date"]) == str(_args.date):
                for _t in _sess.get("targets", []):
                    _targets_list.append((str(_t), str(_args.date)))
        if not _targets_list:
            print(f"[WARN] yaml 中找不到日期 {_args.date} 的場次，試圖直接推測單一目標")
            _targets_list = [("V1162Ori", _args.date)]
    elif _args.target is None and _args.date is None:
        # 全空：預設
        _targets_list = [("V1162Ori", "20251220")]
    else:
        # 指定目標（或目標+日期）
        _targets_list = [(_args.target or "V1162Ori", _args.date or "20251220")]

    print(f"[photometry] 待處理目標：{_targets_list}  通道：{CHANNELS}")

    # ── 同視野偵測：以 split/{ch}/ 前三個 FITS 檔名為 field_key ────────────────
    # 相同視野的目標（hardlink 或同一 session 同一視野）共用 _FrameCompCache，
    # 避免重複做比較星孔徑測光。
    _logger.info(f"[photometry] targets={_targets_list} channels={CHANNELS}")

    def _get_field_key(tgt, dt):
        try:
            _cfg_tmp = cfg_from_yaml(_yaml, tgt, dt, channel=CHANNELS[0],
                                     split_subdir=_args.split_subdir)
            _ff = sorted(_cfg_tmp.wcs_dir.glob(f"*_{CHANNELS[0]}.fits"))
            return tuple(f.name for f in _ff[:3]) if _ff else None
        except Exception:
            return None

    from collections import defaultdict as _defaultdict
    _field_groups: "dict[tuple, list]" = _defaultdict(list)
    for _tgt, _dt in _targets_list:
        _fk = _get_field_key(_tgt, _dt)
        if _fk:
            _field_groups[_fk].append((_tgt, _dt))

    # 為多目標視野建立共用快取（單目標視野不需要）
    _field_caches: "dict[tuple, _FrameCompCache]" = {
        _fk: _FrameCompCache()
        for _fk, _grp in _field_groups.items()
        if len(_grp) > 1
    }
    _multi_fields = sum(1 for v in _field_groups.values() if len(v) > 1)
    if _multi_fields:
        _logger.info(f"[photometry] shared_field_groups={_multi_fields}")
        print(f"[photometry] 偵測到 {_multi_fields} 個多目標視野，啟用共用比較星快取")

    # ── 各目標迴圈 ────────────────────────────────────────────────────────────
    for (ACTIVE_TARGET, ACTIVE_DATE) in _targets_list:
        _logger.info(f"[target] start target={ACTIVE_TARGET} date={ACTIVE_DATE} channels={CHANNELS}")
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
                                split_subdir=_args.split_subdir, out_tag=_args.out_tag,
                                run_ts=_log_ts)
        except Exception as _e_cfg:
            _logger.warning(f"[SKIP] {ACTIVE_TARGET}/{ACTIVE_DATE} cfg error: {_e_cfg}")
            print(f"[SKIP] {ACTIVE_TARGET}/{ACTIVE_DATE} cfg 建立失敗：{_e_cfg}")
            continue

        # ── Log 檔 handler（每個目標獨立 log 檔）─────────────────────────────
        _log_path = cfg.run_root / "1_photometry" / f"photometry_{ACTIVE_DATE}_{_log_ts}.log"
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        _file_hdl = logging.FileHandler(_log_path, encoding="utf-8")
        _file_hdl.setLevel(logging.DEBUG)
        _file_hdl.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        _logger.addHandler(_file_hdl)
        _logger.info(f"[LOG] {_log_path}")
        print(f"[LOG] {_log_path}")
        _logger.info(
            f"[runtime] target={ACTIVE_TARGET} date={ACTIVE_DATE} "
            f"aperture_growth_fraction={cfg.aperture_growth_fraction:.3f}"
        )

        _ch0 = CHANNELS[0]
        wcs_files = sorted(cfg.wcs_dir.glob(f"*_{_ch0}.fits"))
        if not wcs_files:
            _logger.warning(f"[SKIP] no split FITS for channel={_ch0} dir={cfg.wcs_dir}")
            print(f"[SKIP] 找不到 split/{_ch0} FITS：{cfg.wcs_dir}，跳過此目標")
            continue
        print(f"找到 split/{_ch0} FITS：{len(wcs_files)} 張")

        # ── 統一視場星表預建（所有通道共用一次 cone search）──────────────────
        _field_cat_dir_pre = cfg.out_dir.parent.parent.parent / "catalogs"
        _field_cat_path_pre = _field_cat_dir_pre / "field_catalog_unified.csv"
        if not _field_cat_path_pre.exists():
            print(f"[catalog] 預建統一視場星表...")
            _load_or_build_unified_catalog(
                cfg, cfg.target_radec_deg[0], cfg.target_radec_deg[1], _field_cat_dir_pre,
            )

        # ── 孔徑估計用比較星（首通道 V band，僅供 estimate_aperture_radius 使用）─
        # 注意：此時用 YAML 預設孔徑（8px）做初選測光。
        # 孔徑估計在之後用比較星做，正式測光用估計後的孔徑。
        # 初選與正式孔徑的微小差異（~1-2px）不影響比較星篩選結果。
        try:
            (_comp_ap, comp_df_matched, check_star,
             aavso_matched, apass_matched, active_source,
             _vsx_field) = auto_select_comps(
                wcs_files[0], cfg.target_radec_deg, band="V"
            )
        except RuntimeError as _e:
            _logger.warning(f"[SKIP] {ACTIVE_TARGET} aperture preselect failed: {_e}")
            print(f"[SKIP] {ACTIVE_TARGET} 孔徑估算比較星選取失敗，跳過此目標：{_e}")
            continue
        print(f"check_star：{check_star}")

        # ── 孔徑估計（用比較星，亮度接近目標星）──────────────────────────────
        _logger.info(f"[check_star] {check_star}")
        ap_r = None
        if cfg.aperture_auto:
            _logger.info(
                f"[aperture_auto] growth_fraction={cfg.aperture_growth_fraction:.3f} "
                f"r_min={cfg.aperture_min_radius} r_max={cfg.aperture_max_radius}"
            )
            ap_r = estimate_aperture_radius(
                wcs_files[0], comp_df_matched,   # comp_df_matched from V-band initial selection
                cfg.aperture_min_radius, cfg.aperture_max_radius,
                cfg.aperture_growth_fraction,
                max_stars=(
                    max(10, min(30, len(comp_df_matched)))
                    if comp_df_matched is not None else 20
                ),
            )
            if ap_r is not None:
                _logger.info(
                    f"[aperture_auto] selected_radius={ap_r:.2f} "
                    f"growth_fraction={cfg.aperture_growth_fraction:.3f}"
                )
                cfg.aperture_radius = ap_r
                print(
                    f"[生長曲線] 自動孔徑半徑 = {cfg.aperture_radius:.2f} px"
                    "（所有通道共用）"
                )

        if cfg.aperture_auto and ap_r is None:
            _logger.warning(
                f"[aperture_auto] estimate failed; keep fixed aperture={cfg.aperture_radius:.2f} "
                f"growth_fraction={cfg.aperture_growth_fraction:.3f}"
            )

        ap_r_in, ap_r_out = compute_annulus_radii(
            cfg.aperture_radius, cfg.annulus_r_in, cfg.annulus_r_out
        )
        print(f"孔徑 r={cfg.aperture_radius:.2f}  r_in={ap_r_in:.2f}  r_out={ap_r_out:.2f}")
        _logger.info(
            f"[aperture] radius={cfg.aperture_radius:.2f} r_in={ap_r_in:.2f} "
            f"r_out={ap_r_out:.2f} growth_fraction={cfg.aperture_growth_fraction:.3f}"
        )
        _shared_aperture_radius = cfg.aperture_radius

        # ── 多通道測光迴圈 ────────────────────────────────────────────────────
        channel_results: dict = {}
        _comp_refs_per_ch: dict = {}   # {channel: comp_refs} for VSX extra targets
        _check_star_per_ch: dict = {}  # {channel: check_star}
        _split_dir_per_ch: dict = {}   # {channel: split_dir}

        for _ch in CHANNELS:
            _logger.info(f"[channel] start channel={_ch} shared_aperture={_shared_aperture_radius:.2f}")
            print(f"\n{'='*55}")
            print(f"  通道 {_ch}  ({CHANNELS.index(_ch) + 1}/{len(CHANNELS)})")
            print(f"{'='*55}")

            cfg_ch = cfg_from_yaml(_yaml, ACTIVE_TARGET, ACTIVE_DATE, channel=_ch,
                                   split_subdir=_args.split_subdir, out_tag=_args.out_tag,
                                   run_ts=_log_ts)
            cfg_ch.aperture_radius = _shared_aperture_radius
            # gain/read_noise：cfg_ch 從 sensor_db 讀取；若為 None 才從首通道 fallback
            if cfg_ch.gain_e_per_adu is None and cfg.gain_e_per_adu is not None:
                cfg_ch.gain_e_per_adu = cfg.gain_e_per_adu
            if cfg_ch.read_noise_e is None and cfg.read_noise_e is not None:
                cfg_ch.read_noise_e = cfg.read_noise_e

            _split_dir = cfg_ch.wcs_dir
            _split_dir_per_ch[_ch] = _split_dir
            _fits_ch   = sorted(_split_dir.glob(f"*_{_ch}.fits"))
            if not _fits_ch:
                _logger.warning(f"[SKIP] channel={_ch} no split FITS in dir={_split_dir}")
                print(f"  [SKIP] split/{_ch}/ 找不到 FITS，跳過此通道")
                continue

            print(f"  FITS 張數：{len(_fits_ch)}")
            print(f"  輸出 CSV ：{cfg_ch.phot_out_csv}")

            # ── 比較星 m_cat 波段重映射（每通道用正確星表）──
            _band_map_ch = {"R": "R", "G1": "V", "G2": "V", "B": "B"}
            _band_ch = _band_map_ch.get(str(_ch).upper(), "V")
            try:
                (_comp_refs_ch, _comp_df_ch, _check_star_ch,
                 _aavso_ch, _apass_ch, _active_source_ch,
                 _vsx_ch) = auto_select_comps(
                    _fits_ch[0], cfg_ch.target_radec_deg, band=_band_ch
                )
                _logger.info(f"[comp] channel={_ch} source={_active_source_ch} n={len(_comp_refs_ch)}")
                print(f"  [comp] {_ch} source={_active_source_ch} n={len(_comp_refs_ch)}")
                _comp_refs_per_ch[_ch] = _comp_refs_ch
                _check_star_per_ch[_ch] = _check_star_ch
            except RuntimeError as _e_comp_ch:
                _logger.warning(f"[SKIP] channel={_ch} comp selection failed: {_e_comp_ch}")
                print(f"  [SKIP] {_ch} 比較星讀取失敗：{_e_comp_ch}")
                continue

            # 同視野多目標：傳入共用比較星快取
            _active_field_key = _get_field_key(ACTIVE_TARGET, ACTIVE_DATE)
            _active_cache = _field_caches.get(_active_field_key) if _active_field_key else None
            df_ch, _comp_lc_ch = run_photometry_on_wcs_dir(
                _split_dir,
                cfg_ch.phot_out_csv,
                cfg_ch.phot_out_png,
                comp_refs=_comp_refs_ch,
                check_star=_check_star_ch,
                ap_radius=cfg_ch.aperture_radius,
                channel=_ch,
                shared_cache=_active_cache,
            )
            if _active_cache:
                print(f"  [快取] 比較星快取 {len(_active_cache)} 筆")
            channel_results[_ch] = df_ch
            _ok_cnt = int(df_ch['ok'].sum()) if 'ok' in df_ch.columns else 0
            _logger.info(f"[channel] done channel={_ch} ok={_ok_cnt}/{len(df_ch)}")
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
                            _cat_dir_r = cfg_ch.run_root / "1_photometry" / "catalogs"
                            _cat_dir_r.mkdir(parents=True, exist_ok=True)
                            _gaia_r_raw.to_csv(_cat_dir_r / "catalog_GaiaDR3_Rc.csv", index=False)
                            print(f"  [R_GAIA] 星表已儲存 ({len(_gaia_r_raw)} 筆)")
                    else:
                        print(f"  [R_GAIA] Gaia RP→Rc 有效星不足，跳過")
                except Exception as _e_gaia_r:
                    print(f"  [WARN] R_GAIA 失敗：{_e_gaia_r}")

        print(f"\n所有通道完成：{list(channel_results.keys())}")

        # ── G1/G2 共享 Y 軸重繪 ───────────────────────────────────────────
        _shared_chs = [c for c in ("G1", "G2") if c in channel_results]
        if len(_shared_chs) == 2:
            _all_ok_mag = []
            for _sc in _shared_chs:
                _sdf = channel_results[_sc]
                _sdf_ok = _sdf[(_sdf["ok"] == 1) & np.isfinite(_sdf["m_var"])]
                _all_ok_mag.extend(_sdf_ok["m_var"].tolist())
            if _all_ok_mag:
                _margin = 0.02
                _ymin = min(_all_ok_mag) - _margin   # 亮端（小數值）
                _ymax = max(_all_ok_mag) + _margin   # 暗端（大數值）
                _shared_ylim = (_ymax, _ymin)         # invert: 大值在下(bottom)
                for _sc in _shared_chs:
                    # 直接寫回原 run_root，不重建 cfg（避免產生新時間戳目錄）
                    _sc_png = cfg.run_root / "3_light_curve" / f"light_curve_{_sc}_{ACTIVE_DATE}.png"
                    plot_light_curve(channel_results[_sc], _sc_png, _sc,
                                    cfg, obs_date=ACTIVE_DATE, ylim=_shared_ylim)
                print(f"[PLOT] G1/G2 Y 軸已鎖定：{_ymin:.3f} – {_ymax:.3f} mag")

        # ── 週期分析（統一使用 period_analysis.py）─────────────────────────
        from period_analysis import run_period_analysis
        _pa_any = False
        for _ls_ch, _ls_df in channel_results.items():
            if _ls_df is None:
                continue
            _n_ok = int(_ls_df["ok"].sum())
            if _n_ok < 10:
                print(f"[LS] {_ls_ch} 有效幀數 {_n_ok} < 10，跳過")
                continue
            print(f"\n[LS] 通道 {_ls_ch}（有效幀數：{_n_ok}）")
            try:
                _pa_dir = cfg_ch.run_root / "4_period_analysis"
                # 準備 DataFrame 欄位名稱對應（period_analysis 期望 ok, bjd_tdb, m_var/m_var_norm, v_err）
                _pa_df = _ls_df.copy()
                if "v_err" not in _pa_df.columns and "t_sigma_mag" in _pa_df.columns:
                    _pa_df["v_err"] = _pa_df["t_sigma_mag"]
                _pa_result = run_period_analysis(
                    _pa_df,
                    target_name=ACTIVE_TARGET,
                    channel=_ls_ch,
                    out_dir=_pa_dir,
                )
                if _pa_result:
                    _pa_any = True
                    _ls_r = _pa_result.get("ls_result", {})
                    _bp = _ls_r.get("best_period", np.nan)
                    _fap = _ls_r.get("fap", np.nan)
                    if np.isfinite(_bp):
                        print(f"[LS] Best period = {_bp:.6f} d  ({_bp * 24:.4f} h)")
                        print(f"[LS] FAP         = {_fap:.2e}")
                    else:
                        print(f"[LS] {_ls_ch} 週期分析無有效結果（keys: {list(_pa_result.keys())}）")
                    _fit_r = _pa_result.get("fit_result", {})
                    if _fit_r:
                        _amp = _fit_r.get("amplitude", np.nan)
                        print(f"[Fourier] Amplitude = {_amp:.4f} mag")
                    elif _ls_r:
                        print(f"[Fourier] 擬合失敗或跳過")
                else:
                    print(f"[LS] {_ls_ch} run_period_analysis 回傳空結果")
            except Exception as _e_ls:
                print(f"[LS] {_ls_ch} 失敗：{_e_ls}")

        if not _pa_any:
            print("[LS] 跳過（所有通道有效幀數 < 10 或分析失敗）")

        # ── G1/G2 比值光變曲線 ────────────────────────────────────────────────
        _dg1 = channel_results.get("G1")
        _dg2 = channel_results.get("G2")
        if _dg1 is not None and _dg2 is not None:
            _need_cols = ["bjd_tdb", "t_flux_net", "m_var", "ok"]
            _g1_ok = all(_c in _dg1.columns for _c in _need_cols)
            _g2_ok = all(_c in _dg2.columns for _c in _need_cols)
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
                    _ratio_csv = cfg_ch.run_root / "1_photometry" / f"G1G2_ratio_{ACTIVE_DATE}.csv"
                    _mg[["bjd_tdb", "flux_ratio_G1G2", "mag_diff_G1G2",
                          "flux_G1", "flux_G2", "m_G1", "m_G2"]].to_csv(
                        _ratio_csv, index=False, float_format="%.8f"
                    )
                    print(f"[G1/G2] 比值 CSV：{_ratio_csv}（{len(_mg)} 幀）")
                    # 繪圖：flux ratio + mag diff
                    _ratio_png = cfg_ch.run_root / "3_light_curve" / f"G1G2_ratio_{ACTIVE_DATE}.png"
                    # sharex=False：兩軸各自設 formatter，避免互相覆蓋
                    _fig_r, _ax_r = plt.subplots(
                        2, 1, figsize=(12, 6), sharex=False,
                        gridspec_kw={"hspace": 0.12},
                    )
                    _med_ratio = float(np.nanmedian(_mg["flux_ratio_G1G2"]))
                    _med_mag   = float(np.nanmedian(_mg["mag_diff_G1G2"]))
                    _rms_mag   = float(np.sqrt(np.nanmean(
                        (_mg["mag_diff_G1G2"] - _med_mag) ** 2
                    )))

                    # ── 資料繪製 ────────────────────────────────────────────
                    from matplotlib.transforms import blended_transform_factory
                    import matplotlib.ticker as _mticker

                    _bjd_arr_r = _mg["bjd_tdb"].values
                    _bjd_lo    = float(_bjd_arr_r.min())
                    _bjd_hi    = float(_bjd_arr_r.max())
                    _xlim_r    = (_bjd_lo - 0.002, _bjd_hi + 0.002)
                    for _a in _ax_r:
                        _a.set_xlim(_xlim_r)

                    _ax_r[0].plot(_mg["bjd_tdb"], _mg["flux_ratio_G1G2"], "g.", ms=4)
                    _ax_r[0].axhline(_med_ratio, color="k", ls="--", lw=0.8)
                    _ax_r[0].set_ylabel("G1 / G2 flux ratio")
                    # median 標註：繪圖區左端，緊貼虛線上方
                    _ax_r[0].text(
                        0.0, _med_ratio,
                        f"  median={_med_ratio:.4f}",
                        transform=blended_transform_factory(
                            _ax_r[0].transAxes, _ax_r[0].transData
                        ),
                        ha="left", va="bottom", fontsize=8,
                    )

                    _ax_r[1].plot(_mg["bjd_tdb"], _mg["mag_diff_G1G2"], "g.", ms=4)
                    _ax_r[1].axhline(_med_mag, color="k", ls="--", lw=0.8)
                    _ax_r[1].set_ylabel("G1 − G2 (mag)")
                    _ax_r[1].set_xlabel("BJD_TDB", color="steelblue", fontsize=12)
                    _ax_r[1].tick_params(axis='x', colors='steelblue')
                    
                    if cfg.obs_lat_deg is not None:
                        _ax_r[0].text(1.0, 1.12, f"{cfg.obs_lat_deg:.2f}N {cfg.obs_lon_deg:.2f}E", 
                                      transform=_ax_r[0].transAxes, ha='right', va='bottom', 
                                      fontsize=12, color='#2d6a4f')
                    # median 標註：繪圖區左端，緊貼虛線上方
                    _ax_r[1].text(
                        0.0, _med_mag,
                        f"  median={_med_mag:.4f} mag",
                        transform=blended_transform_factory(
                            _ax_r[1].transAxes, _ax_r[1].transData
                        ),
                        ha="left", va="bottom", fontsize=8,
                    )

                    # ── 標題（fontsize=14）+ 棕色可信度文字 ─────────────────
                    _ax_r[0].set_title(
                        f"{ACTIVE_TARGET} {ACTIVE_DATE} — G1/G2 flux ratio",
                        fontsize=14, loc="left",
                    )
                    _ax_r[0].text(
                        1.0, 1.01,
                        f"Reliability: G1-G2 RMS = {_rms_mag:.4f} mag",
                        transform=_ax_r[0].transAxes,
                        ha="right", va="bottom",
                        fontsize=14, color="saddlebrown",
                    )

                    # ── 兩圖之間 Local Time 刻度軸 ───────────────────────────
                    # 計算 HH:MM 刻度（UTC+8）
                    import datetime as _dt
                    _tz_r = float(getattr(cfg, "tz_offset_hours", 8))
                    _tmin_r    = (ATime(_bjd_lo, format="jd", scale="tdb").to_datetime()
                                  + _dt.timedelta(hours=_tz_r))
                    _tmax_r    = (ATime(_bjd_hi, format="jd", scale="tdb").to_datetime()
                                  + _dt.timedelta(hours=_tz_r))

                    _ticks_30  = []   # 每 30 分（含整點）→ 淡藍線
                    _ticks_60  = []   # 整點 → 稍深藍線
                    _tick_labs = []   # HH:MM 標籤（整點 + 半點）
                    _cur_r = _tmin_r.replace(second=0, microsecond=0)
                    _cur_r = _cur_r.replace(minute=(_cur_r.minute // 30) * 30)
                    while _cur_r <= _tmax_r + _dt.timedelta(minutes=1):
                        _utc_r  = _cur_r - _dt.timedelta(hours=_tz_r)
                        _bjd_r  = ATime(_utc_r).jd
                        _ticks_30.append(_bjd_r)
                        _tick_labs.append(_cur_r.strftime("%H:%M"))
                        if _cur_r.minute == 0:
                            _ticks_60.append(_bjd_r)
                        _cur_r += _dt.timedelta(minutes=30)

                    # 垂直線畫在兩個子圖上
                    for _bv in _ticks_30:
                        _ax_r[0].axvline(_bv, color="steelblue", lw=0.6, alpha=0.25, zorder=1)
                        _ax_r[1].axvline(_bv, color="steelblue", lw=0.6, alpha=0.25, zorder=1)
                    for _bv in _ticks_60:
                        _ax_r[0].axvline(_bv, color="steelblue", lw=0.8, alpha=0.5, zorder=1)
                        _ax_r[1].axvline(_bv, color="steelblue", lw=0.8, alpha=0.5, zorder=1)

                    # _ax_r[0] 底部（兩圖中間）：HH:MM，FuncFormatter 各自獨立
                    def _bjd_to_hhmm(bjd_val, pos):
                        try:
                            _t = (ATime(bjd_val, format="jd", scale="tdb")
                                  .to_datetime()
                                  + _dt.timedelta(hours=_tz_r))
                            return _t.strftime("%H:%M")
                        except Exception:
                            return ""

                    _ax_r[0].xaxis.set_major_locator(
                        _mticker.FixedLocator(_ticks_30)
                    )
                    _ax_r[0].xaxis.set_major_formatter(
                        _mticker.FuncFormatter(_bjd_to_hhmm)
                    )
                    _ax_r[0].tick_params(
                        axis="x", which="major",
                        bottom=True, labelbottom=True,
                        top=False, labeltop=False,
                        direction="in", length=5,
                        colors="steelblue", labelsize=8,
                        labelcolor="steelblue",
                    )
                    # "Local Time" 標籤：略偏出繪圖區左側，避免與第一個刻度重疊
                    _ax_r[0].text(
                        -0.002, -0.01, "Local Time",
                        transform=_ax_r[0].transAxes,
                        ha="right", va="top",
                        fontsize=8, color="steelblue",
                        clip_on=False,
                    )

                    # _ax_r[1] 頂部：刻度朝內；底部：完整 BJD 數值
                    _ax_r[1].xaxis.set_major_locator(
                        _mticker.FixedLocator(_ticks_30)
                    )
                    _ax_r[1].xaxis.set_major_formatter(
                        _mticker.FuncFormatter(lambda v, _: f"{v:.2f}")
                    )
                    _ax_r[1].tick_params(
                        axis="x", which="major",
                        top=True, labeltop=False,
                        bottom=True, labelbottom=True,
                        direction="in", length=5,
                        colors="steelblue", labelsize=8,
                        labelcolor="black",
                    )

                    _fig_r.savefig(_ratio_png, dpi=150, bbox_inches="tight")
                    plt.close("all")
                    print(f"[G1/G2] 比值圖：{_ratio_png}")
                    # LS on G1/G2 ratio（使用 period_analysis）
                    if len(_mg) >= 10:
                        try:
                            _ratio_df = pd.DataFrame({
                                "bjd_tdb": _mg["bjd_tdb"].values,
                                "m_var":   _mg["mag_diff_G1G2"].values,
                                "v_err":   np.full(len(_mg), 0.001),
                                "ok":      1,
                            })
                            _pa_dir_r = cfg_ch.run_root / "4_period_analysis"
                            run_period_analysis(
                                _ratio_df,
                                target_name=ACTIVE_TARGET,
                                channel="G1G2ratio",
                                out_dir=_pa_dir_r,
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

        # ── VSX 額外目標星測光 ────────────────────────────────────────────────
        # 從 VSX 查詢結果中篩選 9.8–11.2 等的已知變星，逐顆跑差分測光。
        # 使用與主目標相同的比較星、孔徑、FITS 檔案。
        # 輸出目錄與 YAML 目標相同層級：output/{date}/{group}/{VSXname}/{timestamp}/
        _logger.info(f"[target] base target complete target={ACTIVE_TARGET} date={ACTIVE_DATE}")
        _VSX_MAG_MIN, _VSX_MAG_MAX = 6.0, 12.0
        if _args.no_vsx:
            _emit("info", "[VSX] disabled by --no-vsx")
            print("[VSX 額外目標] --no-vsx 旗標啟用，跳過額外目標測光")
        elif _vsx_field is not None and len(_vsx_field) > 0:
            _vsx_mag_col = None
            for _vc in ("max", "min"):
                if _vc in _vsx_field.columns:
                    _vsx_mag_col = _vc
                    break
            if _vsx_mag_col is not None:
                _vsx_cand = _vsx_field.copy()
                _vsx_cand["_mag"] = pd.to_numeric(_vsx_cand[_vsx_mag_col], errors="coerce")
                _vsx_cand = _vsx_cand[
                    _vsx_cand["_mag"].between(_VSX_MAG_MIN, _VSX_MAG_MAX)
                ].reset_index(drop=True)

                # 排除主目標自身（10" 以內）
                if len(_vsx_cand) > 0:
                    _logger.info(f"[vsx] candidates={len(_vsx_cand)} mag_window={_VSX_MAG_MIN}-{_VSX_MAG_MAX}")
                    _tgt_sc = SkyCoord(ra=cfg.target_radec_deg[0] * u.deg,
                                       dec=cfg.target_radec_deg[1] * u.deg)
                    _vsx_sc = SkyCoord(ra=_vsx_cand["ra_deg"].values * u.deg,
                                       dec=_vsx_cand["dec_deg"].values * u.deg)
                    _sep = _tgt_sc.separation(_vsx_sc).arcsec
                    _vsx_cand = _vsx_cand[_sep > 10.0].reset_index(drop=True)

                if len(_vsx_cand) > 0:
                    print(f"\n{'='*55}")
                    print(f"  [VSX 額外目標] {len(_vsx_cand)} 顆 ({_VSX_MAG_MIN}-{_VSX_MAG_MAX} mag)")
                    print(f"{'='*55}")

                    # 取得主目標的 group 和 project_root（用於建構輸出路徑）
                    _main_tgt_yaml = _yaml.get("targets", {}).get(ACTIVE_TARGET, {})
                    _vsx_group = _main_tgt_yaml.get("group", ACTIVE_TARGET)
                    _vsx_project_root = Path(_yaml["_project_root"])
                    _vsx_date_fmt = f"{ACTIVE_DATE[:4]}-{ACTIVE_DATE[4:6]}-{ACTIVE_DATE[6:8]}"

                    def _vsx_short_name(raw: str) -> str:
                        """VSX 星名 → 短目錄名。
                        短名（ASAS/NSV/V*/...）直接去空格；
                        Gaia DR3 長 ID 截後 8 碼：GDR3_12345678。
                        ASASSN-V 去掉 J 後面的秒數小數。
                        """
                        s = raw.strip()
                        if s.startswith("Gaia DR3 "):
                            return "GDR3_" + s.split()[-1][-8:]
                        if s.startswith("ASASSN-V "):
                            # ASASSN-V J083045.90-480325.0 → ASASSN_J083045-4803
                            tail = s[len("ASASSN-V "):].strip()
                            tail = tail.replace("J", "J", 1)
                            # 簡化：去小數點後的秒數
                            import re as _re
                            m = _re.match(r"(J\d{6})\.\d+([+-]\d{4})", tail)
                            if m:
                                return "ASASSN_" + m.group(1) + m.group(2)
                        return s.replace(" ", "")

                    for _vi, _vr in _vsx_cand.iterrows():
                        _vsx_name_raw = str(_vr.get("Name", f"VSX_{_vi}")).strip()
                        _vsx_name = _vsx_short_name(_vsx_name_raw)
                        _vsx_ra   = float(_vr["ra_deg"])
                        _vsx_dec  = float(_vr["dec_deg"])
                        _vsx_type = str(_vr.get("Type", "?"))
                        _vsx_per  = _vr.get("Period", "?")
                        _vsx_mag  = float(_vr["_mag"])
                        _vsx_ready_channels = [
                            _vch for _vch in CHANNELS
                            if _vch in _comp_refs_per_ch
                            and _vch in _split_dir_per_ch
                            and any(_split_dir_per_ch[_vch].glob(f"*_{_vch}.fits"))
                        ]
                        if not _vsx_ready_channels:
                            _logger.warning(
                                f"[VSX SKIP] name={_vsx_name} no ready channel before target tree creation"
                            )
                            continue
                        _logger.info(
                            f"[vsx] target={_vsx_name} mag={_vsx_mag:.2f} ready_channels={_vsx_ready_channels}"
                        )
                        print(f"\n  ── {_vsx_name_raw}  ({_vsx_type})  "
                              f"mag={_vsx_mag:.2f}  P={_vsx_per}d  "
                              f"RA={_vsx_ra:.4f}  Dec={_vsx_dec:.4f}")

                        # 輸出路徑：與 YAML 目標相同層級
                        # output/{date}/{group}/{VSXname}/{timestamp}/
                        _vsx_run_root = (_vsx_project_root / "output" / _vsx_date_fmt
                                         / _vsx_group / _vsx_name / _log_ts)
                        _vsx_out_dir = _vsx_run_root / "1_photometry"
                        _vsx_out_dir.mkdir(parents=True, exist_ok=True)
                        _vsx_lc_dir = _vsx_run_root / "3_light_curve"
                        _vsx_lc_dir.mkdir(parents=True, exist_ok=True)

                        _vsx_ch_results = {}
                        for _vch in CHANNELS:
                            if _vch not in _comp_refs_per_ch:
                                _logger.warning(f"[VSX SKIP] target={_vsx_name} channel={_vch} missing comp refs")
                                print(f"    {_vch}: 跳過（主目標該通道無比較星）")
                                continue
                            _vsx_csv = _vsx_out_dir / f"photometry_{_vch}_{ACTIVE_DATE}.csv"
                            _vsx_png = _vsx_lc_dir / f"light_curve_{_vch}_{ACTIVE_DATE}.png"

                            try:
                                # 暫存並覆寫全域 cfg
                                _cfg_backup_radec = cfg.target_radec_deg
                                _cfg_backup_name  = cfg.target_name
                                _cfg_backup_band  = cfg.phot_band
                                _cfg_backup_regdir = cfg.regression_diag_dir
                                _vsx_band_map = {"R": "r", "G1": "V", "G2": "V", "B": "B"}
                                cfg.target_radec_deg = (_vsx_ra, _vsx_dec)
                                cfg.target_name = _vsx_name
                                cfg.phot_band = _vsx_band_map.get(_vch.upper(), "V")
                                # 回歸診斷圖存到 VSX 目標自己的目錄，避免覆蓋主目標
                                _vsx_reg_dir = _vsx_run_root / "2_regression_diag"
                                _vsx_reg_dir.mkdir(parents=True, exist_ok=True)
                                cfg.regression_diag_dir = _vsx_reg_dir

                                _vsx_df, _ = run_photometry_on_wcs_dir(
                                    _split_dir_per_ch[_vch],
                                    _vsx_csv,
                                    _vsx_png,
                                    comp_refs=_comp_refs_per_ch[_vch],
                                    check_star=_check_star_per_ch.get(_vch, check_star),
                                    ap_radius=_shared_aperture_radius,
                                    channel=_vch,
                                    shared_cache=_active_cache,
                                )
                                _vsx_ch_results[_vch] = _vsx_df

                                _n_ok = int(_vsx_df["ok"].sum()) if "ok" in _vsx_df.columns else 0
                                if _n_ok > 0 and "m_var" in _vsx_df.columns:
                                    _ok_rows = _vsx_df[_vsx_df["ok"] == 1]
                                    _med = float(_ok_rows["m_var"].median())
                                    _amp = float(_ok_rows["m_var"].max() - _ok_rows["m_var"].min())
                                    _logger.info(
                                        f"[vsx] target={_vsx_name} channel={_vch} ok={_n_ok}/{len(_vsx_df)} "
                                        f"median={_med:.3f} amp={_amp:.3f}"
                                    )
                                    print(f"    {_vch}: ok={_n_ok}/{len(_vsx_df)}  "
                                          f"median={_med:.3f}  amp={_amp:.3f}")
                                else:
                                    _logger.info(
                                        f"[vsx] target={_vsx_name} channel={_vch} ok={_n_ok}/{len(_vsx_df)}"
                                    )
                                    print(f"    {_vch}: ok={_n_ok}/{len(_vsx_df)}")
                            except Exception as _e_vsx_phot:
                                _logger.warning(
                                    f"[vsx] target={_vsx_name} channel={_vch} error: {_e_vsx_phot}"
                                )
                                print(f"    {_vch}: 測光失敗 — {_e_vsx_phot}")
                            finally:
                                cfg.target_radec_deg = _cfg_backup_radec
                                cfg.target_name = _cfg_backup_name
                                cfg.phot_band = _cfg_backup_band
                                cfg.regression_diag_dir = _cfg_backup_regdir

                    print(f"\n  [VSX 額外目標] 全部完成")
