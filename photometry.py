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
import time
import hashlib

import warnings
warnings.filterwarnings("ignore", message=".*datfix.*")

# Windows cp950 終端機：強制 stdout/stderr 使用 UTF-8，防止 Unicode crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
ATime = Time   # 別名，供函式內部使用（避免 local import 遮蔽）
import matplotlib
if not os.environ.get("MPLBACKEND"):
    # Keep CLI runs on a non-interactive backend so batch jobs return to shell cleanly.
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from photutils.detection import DAOStarFinder

# ── Module-level logger（__main__ 會加 handler；函數直接用此 logger）──────────
_phot_logger = logging.getLogger("photometry")


# ── 共用設定讀取（統一由 pipeline_config.py 提供）─────────────────────────────
from pipeline_config import load_pipeline_config  # noqa: E402

# ── 拆分模組匯入 ─────────────────────────────────────────────────────────────
from phot_sources.core import (                  # noqa: E402
    gaussian2d, fit_gaussian_psf,
    m_inst_from_flux, max_pixel_in_box,
    compute_annulus_radii, aperture_photometry,
    is_saturated, radec_to_pixel, in_bounds,
)
from phot_sources.regression import (            # noqa: E402
    robust_linear_fit, mag_error_from_flux,
    differential_mag,
)
from phot_config import Cfg, cfg_from_yaml  # noqa: E402
from phot_aperture import (  # noqa: E402
    _growth_radius_for_star,
    estimate_aperture_radius_with_detection as estimate_aperture_radius,
)
from phot_ensemble import ensemble_normalize  # noqa: E402
from phot_timing import (  # noqa: E402
    apply_gain_from_header,
    compute_airmass,
    get_iso_value,
    require_cfg_values,
    time_from_header,
)
from phot_detect import (  # noqa: E402
    batch_detect_stars,
    detect_stars_with_radec,
)
from phot_wcs import (  # noqa: E402
    _estimate_fov_deg,
    _has_wcs,
    make_cfg_bound_astap_functions,
)
from polt_light_curve import plot_light_curve  # noqa: E402
from phot_sources.io_paths import (  # noqa: E402
    build_target_log_path,
    build_vsx_run_layout,
    format_session_date,
    get_field_catalog_dir,
)
from phot_sources.logging_utils import (  # noqa: E402
    attach_file_handler,
    build_log_timestamp,
    detach_file_handlers,
    emit_progress,
    emit_progress_done,
    init_summary_logger,
    redirect_warnings_to_logger,
)
from phot_sources.runner import (  # noqa: E402
    run_main as _owner_run_main,
)
from phot_sources.stage4 import (  # noqa: E402
    _run_stage4_postprocess as _owner_run_stage4_postprocess,
)
from phot_sources.vsx import (  # noqa: E402
    _prepare_vsx_candidates as _owner_prepare_vsx_candidates,
    _run_vsx_targets_for_target as _owner_run_vsx_targets_for_target,
)
from phot_sources.photometry_backend import run_photometry_on_wcs_dir  # noqa: E402
from phot_sources.catalog import (  # noqa: E402
    _load_or_build_unified_catalog,
    auto_select_comps,
    select_check_star,
)
from phot_sources.field import (  # noqa: E402
    _FrameCompCache,
    _build_shared_field_caches,
    _compute_field_key,
    _field_center_from_wcs_fits,
    _resolve_active_field_cache,
)


# ── 使用方法 ──────────────────────────────────────────────────────────────────
# python photometry.py
# python photometry.py --target V1162Ori --date 20251220 --channels R G1 B
# 第一次執行時設定：
#   os.environ['VARSTAR_CONFIG'] = 'D:/VarStar/pipeline/observation_config.yaml'

run_astap_plate_solve, batch_plate_solve_all = make_cfg_bound_astap_functions(lambda: cfg)

# batch_plate_solve_all 已停用（步驟 3 已完成 plate solve）

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

# camera_sensor_db 由 cfg_from_yaml() 從 yaml sensor_db 建立；
# CAMERA_SENSOR_DB 為備用參考值，不在此處注入（避免覆蓋 yaml 設定）


# [已拆至 phot_core.py] max_pixel_in_box


# [已拆至 phot_core.py] compute_annulus_radii, aperture_photometry, is_saturated


# [已拆至 phot_regression.py] robust_linear_fit, mag_error_from_flux


# [已拆至 phot_catalog.py] _pick_col, _parse_ra_dec


# [已拆至 phot_catalog.py] read_aavso_seq_csv


# [已拆至 phot_catalog.py] fetch_aavso_vsp_api


# [已拆至 phot_catalog.py] filter_catalog_in_frame, select_comp_from_catalog


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

## _remap_comp_refs_to_band — REMOVED by unified catalog refactor (v1.7)
## 每通道現在直接從 unified catalog 獨立選取比較星，不再需要 remap。


# =============================================================================
# ⏸ 待決定：差分測光 (differential_mag) vs 自由斜率回歸 (ensemble normalization)
# =============================================================================
# 背景：v1.6+ 採用逐幀自由斜率回歸 fit(m_inst = a·m_cat + b)，已消除大氣漂移。
#      差分測光 & ensemble 正規化均解決同一問題（逐幀大氣修正），二選一即可。
#      目前回歸已完全滿足需求（R²≥0.9），故暫停差分測光&正規化。
# 決議：待驗證 ensemble normalization 的實際貢獻（可能引入雜訊）。
#      完全消除後需併同此檔案再行決定。

# [已拆至 phot_regression.py] differential_mag


# ── 執行 ──────────────────────────────────────────────────────────────────────
def main(argv=None):
    return _owner_run_main(
        argv,
        photometry_func=run_photometry_on_wcs_dir,
        stage4_postprocess_func=_owner_run_stage4_postprocess,
        prepare_vsx_candidates_func=_owner_prepare_vsx_candidates,
        run_vsx_targets_func=partial(
            _owner_run_vsx_targets_for_target,
            photometry_func=run_photometry_on_wcs_dir,
        ),
    )


if __name__ == "__main__":
    main()
