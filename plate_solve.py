# -*- coding: utf-8 -*-
"""
plate_solve.py  —  星圖解算模組
專案：變星測光管線 v0.99
描述：對校正後的 FITS 執行 WCS 星圖解算。
      本地環境使用 ASTAP CLI；Colab 環境使用 astrometry.net API。
      輸出 WCS FITS 到 data/{date}/{group}/wcs，不覆蓋原始校正幀。

後端選擇邏輯
------------
    偵測到 google.colab 可 import  → astrometry_net
    否則                           → astap
    observation_config.yaml backend: "astap" / "astrometry_net" 可強制指定

astrometry.net 大檔案問題的解法
--------------------------------
上傳前對影像做 4× 降採樣（data[::4, ::4]），
6720×4480 → 1680×1120，檔案從 ~90 MB 降至 ~7 MB。
解算成功後，CRPIX 和 CD matrix 換算回原始解析度再寫入完整 FITS。

ASTAP 星表
-----------
D80 星表（80 等深度），比 H18（18 等）深兩個星等，
適合銀緯較低的南天目標（Al Vel、SX Phe）。

ASTAP hint 機制
---------------
yaml 的 targets 區段若有 ra_hint_h / dec_hint_deg，
解算時傳 -ra（度）/ -spd（南極距，= 90 + dec）給 ASTAP，
將盲搜範圍壓縮到 search_radius_deg 半徑內，避免假陽性天區。
"""

from __future__ import annotations

import io
import json
import os
import shutil
import stat
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
from astropy.io import fits
from matplotlib.colors import LinearSegmentedColormap
from astropy.wcs import WCS  # noqa: F401  (保留供呼叫端 import)

from Calibration import load_config, read_raw_image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 環境偵測
# =============================================================================

def detect_backend(cfg: dict) -> str:
    """
    偵測執行環境，回傳要使用的後端名稱。

    Priority：
        1. yaml backend 欄位明確指定（非 "auto"）→ 直接使用
        2. 偵測到 google.colab               → "astrometry_net"
        3. 否則                               → "astap"
    """
    backend = cfg.get("astrometry", {}).get("backend", "auto")
    allowed = {"auto", "", "astap", "astrometry_net"}
    if backend not in allowed:
        raise ValueError(
            f"Unsupported astrometry backend: {backend!r}. "
            "Allowed values: auto, astap, astrometry_net."
        )
    if backend not in ("auto", ""):
        return backend

    try:
        import google.colab  # noqa: F401
        return "astrometry_net"
    except ImportError:
        return "astap"


# =============================================================================
# WCS 換算工具
# =============================================================================

def _scale_wcs_to_original(
    header_solved: fits.Header,
    downsample: int,
    row_offset: int = 0,
    col_offset: int = 0,
) -> fits.Header:
    """
    將降採樣影像解算出的 WCS 換算回原始解析度。

    換算規則（stride = downsample 的子採樣）：
        CRPIX_new = (CRPIX_solved - 0.5) * downsample + 0.5
        CD{i}_{j}_new = CD{i}_{j}_solved / downsample

    Parameters
    ----------
    header_solved : 降採樣影像解算出的 FITS 標頭（含 WCS）。
    downsample    : 降採樣倍率（整數）。
    row_offset    : 子陣列的列偏移（像素），一般為 0。
    col_offset    : 子陣列的行偏移（像素），一般為 0。

    Returns
    -------
    fits.Header
        已更新 WCS 的標頭（副本）。
    """
    h = header_solved.copy()
    factor = float(downsample)

    for axis in (1, 2):
        key = f"CRPIX{axis}"
        if key in h:
            offset = col_offset if axis == 1 else row_offset
            h[key] = (h[key] - 0.5) * factor + 0.5 + offset

    for i in (1, 2):
        for j in (1, 2):
            key = f"CD{i}_{j}"
            if key in h:
                h[key] = h[key] / factor

    # CDELT（若有）
    for axis in (1, 2):
        key = f"CDELT{axis}"
        if key in h:
            h[key] = h[key] / factor

    return h


_WCS_REQUIRED_KEYS = ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2")
_WCS_COPY_KEYS = (
    "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2",
    "CRPIX1", "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2",
    "CDELT1", "CDELT2", "CROTA2", "EQUINOX", "RADESYS",
)


def _angular_sep_deg(
    ra1_deg: float,
    dec1_deg: float,
    ra2_deg: float,
    dec2_deg: float,
) -> float:
    ra1 = np.deg2rad(ra1_deg)
    dec1 = np.deg2rad(dec1_deg)
    ra2 = np.deg2rad(ra2_deg)
    dec2 = np.deg2rad(dec2_deg)
    sin_ddec = np.sin((dec2 - dec1) / 2.0)
    sin_dra = np.sin((ra2 - ra1) / 2.0)
    a = sin_ddec * sin_ddec + np.cos(dec1) * np.cos(dec2) * sin_dra * sin_dra
    return float(np.rad2deg(2.0 * np.arcsin(min(1.0, np.sqrt(max(0.0, a))))))


def _hint_world_center(
    ra_hint_h: Optional[float],
    spd_hint_deg: Optional[float],
) -> Optional[tuple[float, float]]:
    if ra_hint_h is None or spd_hint_deg is None:
        return None
    return (float(ra_hint_h) * 15.0) % 360.0, float(spd_hint_deg) - 90.0


def _validation_max_offset_deg(
    astap_cfg: dict,
    ra_hint_h: Optional[float],
    spd_hint_deg: Optional[float],
) -> Optional[float]:
    if _hint_world_center(ra_hint_h, spd_hint_deg) is None:
        return None
    search_radius = float(astap_cfg.get("search_radius_deg", 5.0))
    default_limit = max(2.0, search_radius * 1.25)
    return float(astap_cfg.get("validation_max_center_offset_deg", default_limit))


def _validate_wcs_header(
    header: fits.Header,
    data_shape: tuple[int, ...],
    *,
    ra_hint_h: Optional[float] = None,
    spd_hint_deg: Optional[float] = None,
    max_offset_deg: Optional[float] = None,
) -> tuple[bool, str]:
    if len(data_shape) != 2:
        return False, f"data is not 2D: shape={data_shape}"

    missing = [key for key in _WCS_REQUIRED_KEYS if key not in header]
    if missing:
        return False, "missing WCS keys: " + ", ".join(missing)

    has_cd = "CD1_1" in header and "CD2_2" in header
    has_cdelt = "CDELT1" in header and "CDELT2" in header
    if not has_cd and not has_cdelt:
        return False, "missing WCS scale matrix"

    try:
        wcs = WCS(header)
    except Exception as exc:
        return False, f"WCS construction failed: {exc}"

    if not wcs.has_celestial:
        return False, "WCS has no celestial axes"

    try:
        matrix = np.asarray(wcs.pixel_scale_matrix, dtype=float)
        determinant = float(np.linalg.det(matrix))
    except Exception as exc:
        return False, f"WCS pixel scale check failed: {exc}"
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        return False, "WCS pixel scale matrix is not finite"
    if abs(determinant) < 1.0e-16:
        return False, "WCS pixel scale matrix is singular"

    height, width = data_shape
    center_x = (float(width) - 1.0) / 2.0
    center_y = (float(height) - 1.0) / 2.0
    try:
        ra_deg, dec_deg = wcs.wcs_pix2world([[center_x, center_y]], 0)[0]
        ra_deg = float(ra_deg)
        dec_deg = float(dec_deg)
    except Exception as exc:
        return False, f"WCS center projection failed: {exc}"

    if not np.isfinite(ra_deg) or not np.isfinite(dec_deg):
        return False, "WCS center is not finite"
    if dec_deg < -90.0 or dec_deg > 90.0:
        return False, f"WCS center Dec out of range: {dec_deg:.6f}"

    hint_center = _hint_world_center(ra_hint_h, spd_hint_deg)
    if hint_center is not None and max_offset_deg is not None:
        hint_ra_deg, hint_dec_deg = hint_center
        offset_deg = _angular_sep_deg(ra_deg, dec_deg, hint_ra_deg, hint_dec_deg)
        if offset_deg > max_offset_deg:
            return (
                False,
                f"WCS center offset {offset_deg:.3f} deg exceeds "
                f"limit {max_offset_deg:.3f} deg",
            )

    return True, "ok"


def _validate_wcs_file(
    fits_path: Path,
    *,
    ra_hint_h: Optional[float] = None,
    spd_hint_deg: Optional[float] = None,
    max_offset_deg: Optional[float] = None,
) -> tuple[bool, str]:
    try:
        with fits.open(fits_path, ignore_missing_end=True, memmap=False) as hdul:
            hdu = next((item for item in hdul if item.data is not None), None)
            if hdu is None:
                return False, "FITS contains no image data"
            data_shape = tuple(hdu.data.shape)
            header = hdu.header.copy()
    except Exception as exc:
        return False, f"cannot read FITS: {exc}"

    return _validate_wcs_header(
        header,
        data_shape,
        ra_hint_h=ra_hint_h,
        spd_hint_deg=spd_hint_deg,
        max_offset_deg=max_offset_deg,
    )


def _write_fits_with_wcs_header(
    src_fits: Path,
    wcs_path: Path,
    out_fits: Path,
    *,
    ra_hint_h: Optional[float] = None,
    spd_hint_deg: Optional[float] = None,
    max_offset_deg: Optional[float] = None,
) -> tuple[bool, str]:
    try:
        with fits.open(src_fits, ignore_missing_end=True, memmap=False) as hdul:
            hdu = next((item for item in hdul if item.data is not None), None)
            if hdu is None:
                return False, "source FITS contains no image data"
            data = np.asarray(hdu.data).copy()
            header = hdu.header.copy()
    except Exception as exc:
        return False, f"cannot read ASTAP WCS output: {exc}"
    try:
        wcs_header = fits.getheader(wcs_path)
    except Exception:
        try:
            wcs_text = wcs_path.read_text(encoding="ascii", errors="ignore")
            wcs_header = fits.Header.fromstring(wcs_text, sep="\n")
        except Exception as exc:
            return False, f"cannot parse ASTAP WCS header: {exc}"

    for key in _WCS_COPY_KEYS:
        if key in header:
            del header[key]
        if key in wcs_header:
            header[key] = wcs_header[key]
    header["HISTORY"] = f"WCS solved by ASTAP using {wcs_path.name}"

    valid, reason = _validate_wcs_header(
        header,
        data.shape,
        ra_hint_h=ra_hint_h,
        spd_hint_deg=spd_hint_deg,
        max_offset_deg=max_offset_deg,
    )
    if not valid:
        return False, reason

    try:
        out_fits.parent.mkdir(parents=True, exist_ok=True)
        fits.PrimaryHDU(data=data, header=header).writeto(out_fits, overwrite=True)
        _set_writable(out_fits)
    except OSError as exc:
        return False, f"cannot write merged WCS FITS: {exc}"
    return True, "ok"


def _subprocess_tail(text: str, limit: int = 500) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return "..." + text[-limit:]


def _resolve_astap_tmp_root(project_root: Path, astap_cfg: dict) -> Path:
    configured = astap_cfg.get("tmp_dir") or astap_cfg.get("work_dir") or ""
    if configured:
        tmp_root = Path(str(configured))
        if not tmp_root.is_absolute():
            tmp_root = project_root / tmp_root
        return tmp_root
    return project_root / "TEMP" / "plate_solve_tmp"


def _set_writable(path: Path) -> None:
    try:
        mode = stat.S_IREAD | stat.S_IWRITE
        if path.is_dir():
            mode |= stat.S_IEXEC
        os.chmod(path, mode)
    except OSError:
        pass


def _cleanup_astap_work_dir(work_dir: Path) -> None:
    if not work_dir.exists():
        return

    for root, dirs, files in os.walk(work_dir, topdown=False):
        for name in files:
            _set_writable(Path(root) / name)
        for name in dirs:
            _set_writable(Path(root) / name)
    _set_writable(work_dir)

    def _onerror(func, raw_path, _exc_info) -> None:
        path = Path(raw_path)
        _set_writable(path)
        func(raw_path)

    last_exc: Optional[Exception] = None
    for attempt in range(8):
        try:
            shutil.rmtree(work_dir, onerror=_onerror)
            return
        except Exception as exc:
            last_exc = exc
            time.sleep(0.5)
    print(f"  [WARN] ASTAP temp cleanup failed: {work_dir} ({last_exc})")


# =============================================================================
# ASTAP hint 解析工具
# =============================================================================

def _get_hint_for_target(
    cfg: dict,
    target_name: str,
) -> tuple[Optional[float], Optional[float]]:
    """
    從 yaml targets 區段取得目標星的 ASTAP hint 座標。

    ASTAP CLI 參數單位：
        -ra  : 小時（hours），範圍 0–24
        -spd : 南極距（degrees），SPD = 90 + Dec
               北天 Dec > 0 → SPD > 90；南天 Dec < 0 → SPD < 90

    Parameters
    ----------
    cfg         : 完整 yaml config 字典。
    target_name : 目標星名稱（須與 yaml targets 的 key 完全相符）。

    Returns
    -------
    tuple[float | None, float | None]
        (ra_hours, spd_deg)。任一欄位缺失時回傳 (None, None)。
    """
    targets_cfg = cfg.get("targets", {})
    if target_name not in targets_cfg:
        return None, None

    t = targets_cfg[target_name]
    ra_h = t.get("ra_hint_h")
    dec_deg = t.get("dec_hint_deg")

    if ra_h is None or dec_deg is None:
        return None, None

    try:
        ra_hours = float(ra_h)            # 單位已是小時，直接使用
        dec_value = float(dec_deg)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid astrometry hint for target {target_name}: "
            f"ra_hint_h={ra_h!r}, dec_hint_deg={dec_deg!r}"
        ) from exc
    if not 0.0 <= ra_hours < 24.0:
        raise ValueError(
            f"Invalid ra_hint_h for target {target_name}: {ra_hours!r}"
        )
    if not -90.0 <= dec_value <= 90.0:
        raise ValueError(
            f"Invalid dec_hint_deg for target {target_name}: {dec_value!r}"
        )
    spd_deg = 90.0 + dec_value            # 南極距：SPD = 90 + Dec
    return ra_hours, spd_deg


def _session_groups(cfg: dict, target_list: list[str]) -> list[dict[str, object]]:
    """Group session targets by product group while preserving session order."""
    grouped: list[dict[str, object]] = []
    by_group: dict[str, dict[str, object]] = {}
    for raw_target in target_list:
        target = str(raw_target)
        target_cfg = cfg.get("targets", {}).get(target, {})
        group = str(target_cfg.get("group", target))
        if group not in by_group:
            entry: dict[str, object] = {
                "group": group,
                "hint_target": target,
                "targets": [target],
            }
            by_group[group] = entry
            grouped.append(entry)
        else:
            targets = by_group[group]["targets"]
            assert isinstance(targets, list)
            targets.append(target)
    return grouped


def _relative_offset_arcmin(
    ra_deg: float,
    dec_deg: float,
    ref_ra_deg: float,
    ref_dec_deg: float,
) -> tuple[float, float]:
    dra_deg = (ra_deg - ref_ra_deg + 180.0) % 360.0 - 180.0
    x_arcmin = dra_deg * float(np.cos(np.deg2rad(ref_dec_deg))) * 60.0
    y_arcmin = (dec_deg - ref_dec_deg) * 60.0
    return float(x_arcmin), float(y_arcmin)


def _parse_local_hour(header: fits.Header) -> Optional[float]:
    for key in ("MID-OBS", "DATE-OBS"):
        value = header.get(key)
        if not value:
            continue
        text = str(value).strip().replace("Z", "")
        try:
            dt_utc = datetime.fromisoformat(text)
        except ValueError:
            continue
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        dt_local = dt_utc.astimezone(timezone(timedelta(hours=8)))
        return (
            float(dt_local.hour)
            + float(dt_local.minute) / 60.0
            + float(dt_local.second) / 3600.0
        )
    return None


def _local_hour_to_color_value(local_hour: Optional[float]) -> float:
    if local_hour is None:
        return 0.5
    hour = local_hour % 24.0
    if hour >= 18.0:
        return (hour - 18.0) / 12.0
    return (hour + 6.0) / 12.0


def _drift_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "night_cycle",
        [
            (0.0, "#00A2E8"),
            (4.0 / 12.0, "#1b3358"),
            (8.0 / 12.0, "#1b3358"),
            (1.0, "#00A2E8"),
        ],
    )


def _collect_drift_entries(wcs_files: list[Path]) -> list[dict]:
    entries: list[dict] = []
    for seq, wcs_path in enumerate(sorted(wcs_files), start=1):
        with fits.open(wcs_path, ignore_missing_end=True) as hdul:
            hdu = next((item for item in hdul if item.data is not None), None)
            if hdu is None:
                continue
            data = np.asarray(hdu.data)
            if data.ndim != 2:
                continue
            header = hdu.header.copy()
        valid, _reason = _validate_wcs_header(header, data.shape)
        if not valid:
            continue
        wcs = WCS(header)
        height, width = data.shape
        pixel_map = {
            "center": ((width - 1) / 2.0, (height - 1) / 2.0),
            "tl": (0.0, 0.0),
            "tr": (width - 1.0, 0.0),
            "bl": (0.0, height - 1.0),
            "br": (width - 1.0, height - 1.0),
        }
        entry = {
            "seq": seq,
            "file": wcs_path.name,
            "local_hour": _parse_local_hour(header),
            "points": {},
        }
        for label, (x_pix, y_pix) in pixel_map.items():
            ra_deg, dec_deg = wcs.wcs_pix2world([[x_pix, y_pix]], 0)[0]
            entry["points"][label] = (float(ra_deg), float(dec_deg))
        entries.append(entry)
    return entries


def _plot_drift_report(
    wcs_files: list[Path],
    out_png: Path,
    title: str,
) -> bool:
    entries = _collect_drift_entries(wcs_files)
    if not entries:
        return False

    fig, ax = plt.subplots(figsize=(10, 8), dpi=140)
    ref_ra_deg, ref_dec_deg = entries[0]["points"]["center"]
    center_ras: list[float] = []
    center_decs: list[float] = []
    center_colors: list[float] = []

    for entry in entries:
        tl = entry["points"]["tl"]
        tr = entry["points"]["tr"]
        br = entry["points"]["br"]
        bl = entry["points"]["bl"]
        outline_ra = [tl[0], tr[0], br[0], bl[0], tl[0]]
        outline_dec = [tl[1], tr[1], br[1], bl[1], tl[1]]
        ax.plot(outline_ra, outline_dec, color="#7f8c8d", linewidth=0.7, alpha=0.18)

        center_ra, center_dec = entry["points"]["center"]
        center_ras.append(center_ra)
        center_decs.append(center_dec)
        center_colors.append(_local_hour_to_color_value(entry.get("local_hour")))
        ax.text(
            center_ra,
            center_dec,
            str(entry["seq"]),
            color="#0b1220",
            fontsize=7,
            ha="left",
            va="bottom",
        )

    scatter = ax.scatter(
        center_ras,
        center_decs,
        s=42,
        c=center_colors,
        cmap=_drift_colormap(),
        marker="o",
        edgecolors="none",
        alpha=0.95,
        zorder=3,
    )

    ax.axhline(ref_dec_deg, color="0.7", linewidth=0.8)
    ax.axvline(ref_ra_deg, color="0.7", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Right Ascension (deg)")
    ax.set_ylabel("Declination (deg)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.invert_xaxis()
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0.0, 4.0 / 12.0, 8.0 / 12.0, 1.0])
    cbar.set_ticklabels(["18:00", "22:00", "02:00", "06:00"])
    cbar.set_label("Local time")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return True


# =============================================================================
# ASTAP 後端
# =============================================================================

def _run_astap(
    fits_path: Path,
    out_path: Path,
    astap_cfg: dict,
    ra_hint_h: Optional[float] = None,
    spd_hint_deg: Optional[float] = None,
    max_offset_deg: Optional[float] = None,
    tmp_root: Optional[Path] = None,
) -> bool:
    """
    呼叫 ASTAP CLI 對單張 FITS 執行星圖解算，輸出到 out_path。

    ASTAP 解算流程：
        1. 建立顯式 ASTAP 工作目錄，正式 FITS 只讀不複製
        2. 執行 ASTAP CLI，將 .wcs/.ini 輸出寫到工作目錄
           傳入 -ra（小時）/ -spd（南極距）hint，修正 NINA 標頭
           RA/DEC 不可靠（實測為北極點座標）導致 ASTAP 搜尋範圍
           偏離目標天區的問題。不縮減 search_radius，避免影響
           ASTAP 內部搜尋格網的步驟起點計算。
        3. 驗證 WCS 中心在合理天區內
        4. 將 .wcs 合併回 FITS data/header，寫到 out_path

    Parameters
    ----------
    fits_path    : 輸入的校正 FITS（不會被修改）。
    out_path     : 輸出的 WCS FITS 路徑。
    astap_cfg    : yaml 裡 astrometry.astap 的設定字典。
    ra_hint_h    : hint RA（小時，ASTAP -ra 單位，範圍 0–24），
                   None 表示不傳 hint（ASTAP 改用標頭座標，通常不可靠）。
    spd_hint_deg : hint 南極距（度，ASTAP -spd 單位，= 90 + Dec），
                   None 表示不傳 hint。

    Returns
    -------
    bool
        解算成功回傳 True，失敗回傳 False。
    """
    executable = astap_cfg.get("executable", "astap_cli")
    db_path = astap_cfg.get("db_path", "")
    search_radius = float(astap_cfg.get("search_radius_deg", 5.0))
    downsample = max(1, int(astap_cfg.get("downsample", 2)))
    fov = float(astap_cfg.get("fov_override_deg", 0.0))
    timeout = int(astap_cfg.get("timeout_sec", 180))
    max_retries = max(1, int(astap_cfg.get("max_retries", 2)))

    if tmp_root is None:
        tmp_root = Path.cwd() / "TEMP" / "plate_solve_tmp"
    try:
        tmp_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"  [ERROR] ASTAP temp root create failed: {tmp_root} ({exc})")
        return False

    work_dir = tmp_root / f"astap_{os.getpid()}_{time.time_ns()}"
    try:
        work_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print(f"  [ERROR] ASTAP temp workdir create failed: {work_dir} ({exc})")
        return False

    try:
        astap_out_base = work_dir / fits_path.name
        print(f"  [TMP] ASTAP workdir: {work_dir}")
        cmd = [
            executable,
            "-f", str(fits_path),
            "-r", str(search_radius),
            "-d", db_path,
            "-o", str(astap_out_base),
        ]
        if downsample > 1:
            cmd += ["-z", str(downsample)]
        if fov > 0:
            cmd += ["-fov", str(fov)]

        # hint：修正 NINA 標頭 RA/DEC 不可靠問題
        # -ra 單位：小時（0–24）；-spd 單位：度（= 90 + Dec）
        if ra_hint_h is not None and spd_hint_deg is not None:
            cmd += ["-ra", f"{ra_hint_h:.4f}"]
            cmd += ["-spd", f"{spd_hint_deg:.4f}"]

        for attempt in range(max_retries):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(work_dir),
                )
                if result.returncode == 0:
                    solved_fits = fits_path
                    valid, reason = _validate_wcs_file(
                        fits_path,
                        ra_hint_h=ra_hint_h,
                        spd_hint_deg=spd_hint_deg,
                        max_offset_deg=max_offset_deg,
                    )
                    if not valid:
                        wcs_path = astap_out_base.with_suffix(".wcs")
                        merged_fits = work_dir / f"{fits_path.stem}_merged.fits"
                        if not wcs_path.exists():
                            print(
                                f"  [WARN] ASTAP returned 0 but no usable WCS was found: "
                                f"{fits_path.name} ({reason})"
                            )
                            continue
                        valid, reason = _write_fits_with_wcs_header(
                            fits_path,
                            wcs_path,
                            merged_fits,
                            ra_hint_h=ra_hint_h,
                            spd_hint_deg=spd_hint_deg,
                            max_offset_deg=max_offset_deg,
                        )
                        if not valid:
                            print(
                                f"  [WARN] ASTAP .wcs merge failed: "
                                f"{fits_path.name} ({reason})"
                            )
                            continue
                        solved_fits = merged_fits
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copyfile(solved_fits, out_path)
                        _set_writable(out_path)
                    except OSError as exc:
                        print(
                            f"  [ERROR] ASTAP output copy failed: "
                            f"{solved_fits} -> {out_path} ({exc})"
                        )
                        return False
                    return True
                else:
                    if attempt == max_retries - 1:
                        stderr_tail = _subprocess_tail(result.stderr)
                        stdout_tail = _subprocess_tail(result.stdout)
                        if stderr_tail:
                            print(f"  [ASTAP stderr] {stderr_tail}")
                        if stdout_tail:
                            print(f"  [ASTAP stdout] {stdout_tail}")
                    if attempt < max_retries - 1:
                        print(f"  [WARN] ASTAP 第 {attempt + 1} 次解算失敗，重試…")
                        time.sleep(2)

            except subprocess.TimeoutExpired:
                print(f"  [WARN] ASTAP 超時（{timeout}s），第 {attempt + 1} 次。")
            except FileNotFoundError:
                print(f"  [錯誤] 找不到 ASTAP 執行檔：{executable}")
                return False

        print(f"  [失敗] ASTAP 解算失敗：{fits_path.name}")
        return False
    finally:
        _cleanup_astap_work_dir(work_dir)


# =============================================================================
# astrometry.net 後端
# =============================================================================

def _upload_to_astrometry_net(
    data_small: np.ndarray,
    api_key: str,
    timeout: int = 300,
) -> Optional[int]:
    """
    將降採樣影像上傳到 nova.astrometry.net，回傳 job_id。

    Parameters
    ----------
    data_small : 降採樣後的 2D numpy 陣列。
    api_key    : nova.astrometry.net 的 API key。
    timeout    : 等待解算完成的最長秒數。

    Returns
    -------
    int | None
        解算成功回傳 job_id，失敗回傳 None。
    """
    base_url = "http://nova.astrometry.net/api"

    login_data = json.dumps({"apikey": api_key}).encode()
    req = urllib.request.Request(
        f"{base_url}/login",
        data=urllib.parse.urlencode(
            {"request-json": login_data.decode()}
        ).encode(),
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        login_resp = json.loads(resp.read())
    if login_resp.get("status") != "success":
        print(f"  [錯誤] astrometry.net 登入失敗：{login_resp}")
        return None
    session = login_resp["session"]

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tf:
        tmp_name = tf.name
    try:
        fits.writeto(tmp_name, data_small.astype(np.float32), overwrite=True)
        with open(tmp_name, "rb") as fh:
            file_bytes = fh.read()
    finally:
        os.unlink(tmp_name)

    upload_params = {
        "session": session,
        "allow_commercial_use": "n",
        "allow_modifications": "n",
        "publicly_visible": "n",
    }
    boundary = "AstrometryNetBoundary"
    body = b""
    for key, val in upload_params.items():
        body += (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
            f"{val}\r\n"
        ).encode()
    body += (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="upload.fits"\r\n'
        f"Content-Type: application/fits\r\n\r\n"
    ).encode() + file_bytes + f"\r\n--{boundary}--\r\n".encode()

    upload_req = urllib.request.Request(
        f"{base_url}/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(upload_req, timeout=60) as resp:
        upload_resp = json.loads(resp.read())

    if upload_resp.get("status") != "success":
        print(f"  [錯誤] astrometry.net 上傳失敗：{upload_resp}")
        return None

    submission_id = upload_resp["subid"]

    deadline = time.time() + timeout
    job_id = None
    while time.time() < deadline:
        time.sleep(10)
        with urllib.request.urlopen(
            f"{base_url}/submissions/{submission_id}", timeout=30
        ) as resp:
            sub_status = json.loads(resp.read())

        jobs = sub_status.get("jobs", [])
        if jobs and jobs[0] is not None:
            job_id = jobs[0]
            break

    if job_id is None:
        print("  [錯誤] astrometry.net 解算超時，未產生 job_id。")
        return None

    deadline2 = time.time() + timeout
    while time.time() < deadline2:
        with urllib.request.urlopen(
            f"{base_url}/jobs/{job_id}", timeout=30
        ) as resp:
            job_status = json.loads(resp.read())
        status = job_status.get("status")
        if status == "success":
            return job_id
        elif status == "failure":
            print(f"  [錯誤] astrometry.net Job {job_id} 解算失敗。")
            return None
        time.sleep(10)

    print("  [錯誤] astrometry.net 等待 job 完成超時。")
    return None


def _run_astrometry_net(
    fits_path: Path,
    out_path: Path,
    anet_cfg: dict,
    ra_hint_h: Optional[float] = None,
    spd_hint_deg: Optional[float] = None,
    max_offset_deg: Optional[float] = None,
) -> bool:
    """
    使用 astrometry.net API 解算，結果寫入 out_path。

    解算流程：
        1. 降採樣 4× 減少上傳大小
        2. 上傳到 nova.astrometry.net
        3. 等待解算完成，下載 WCS 標頭
        4. 把 WCS 換算回原始解析度（_scale_wcs_to_original）
        5. 合併 WCS 標頭與原始 FITS 數據，寫出到 out_path

    Parameters
    ----------
    fits_path : 輸入的校正 FITS。
    out_path  : 輸出的 WCS FITS 路徑。
    anet_cfg  : yaml 裡 astrometry.astrometry_net 的設定字典。

    Returns
    -------
    bool
        解算成功回傳 True，失敗回傳 False。
    """
    api_key = anet_cfg.get("api_key", "")
    if not api_key:
        print("  [錯誤] astrometry_net.api_key 未設定。")
        return False

    upload_downsample = int(anet_cfg.get("upload_downsample", 4))
    timeout = int(anet_cfg.get("timeout_sec", 300))

    with fits.open(fits_path) as hdul:
        data_orig = hdul[0].data.astype(np.float32)
        header_orig = hdul[0].header.copy()

    data_small = data_orig[::upload_downsample, ::upload_downsample]

    job_id = _upload_to_astrometry_net(data_small, api_key, timeout)
    if job_id is None:
        return False

    wcs_url = f"http://nova.astrometry.net/wcs_file/{job_id}"
    try:
        with urllib.request.urlopen(wcs_url, timeout=30) as resp:
            wcs_bytes = resp.read()
    except Exception as exc:
        print(f"  [錯誤] 下載 WCS 失敗：{exc}")
        return False

    with fits.open(io.BytesIO(wcs_bytes)) as wcs_hdul:
        wcs_header_small = wcs_hdul[0].header.copy()

    wcs_header_full = _scale_wcs_to_original(wcs_header_small, upload_downsample)

    wcs_keys = [
        "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2",
        "CRPIX1", "CRPIX2", "CD1_1", "CD1_2", "CD2_1", "CD2_2",
        "CDELT1", "CDELT2", "CROTA2", "EQUINOX", "RADESYS",
    ]
    for key in wcs_keys:
        if key in wcs_header_full:
            header_orig[key] = wcs_header_full[key]

    header_orig["HISTORY"] = (
        f"WCS solved by astrometry.net job {job_id} "
        f"(upload downsample={upload_downsample}x)"
    )
    valid, reason = _validate_wcs_header(
        header_orig,
        data_orig.shape,
        ra_hint_h=ra_hint_h,
        spd_hint_deg=spd_hint_deg,
        max_offset_deg=max_offset_deg,
    )
    if not valid:
        print(f"  [ERROR] astrometry.net returned invalid WCS: {reason}")
        return False

    hdu = fits.PrimaryHDU(data=data_orig, header=header_orig)
    hdu.verify("silentfix")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(out_path, overwrite=True)
    return True


# =============================================================================
# 主解算管線
# =============================================================================

def run_plate_solve(config_path: str | Path, *, raw_mode: bool = False,
                    src_dir: str = "", wcs_out_dir: str = "") -> None:
    """
    主星圖解算管線入口。

    讀取 observation_config.yaml，對每個 obs_session 的每個 group，
    處理 data/{date}/{group}/wcs 內尚未解算的校正 FITS，
    輸出 *_wcs.fits 到同一目錄。

    ASTAP hint 機制：
        yaml targets 區段若有 ra_hint_h / dec_hint_deg，
        自動傳 -ra（度）/-spd（南極距）給 ASTAP，
        search_radius_deg 自動壓縮至 5°，防止假陽性。

    Parameters
    ----------
    config_path : observation_config.yaml 的路徑。
    """
    cfg = load_config(config_path)
    backend = detect_backend(cfg)
    astro_cfg = cfg.get("astrometry", {})
    astap_cfg = astro_cfg.get("astap", {})
    anet_cfg = astro_cfg.get("astrometry_net", {})

    _mode_tag = " [RAW MODE]" if raw_mode else ""
    print("\n" + "=" * 60)
    print(f"  變星測光管線 — 星圖解算模組  plate_solve.py  [{backend}]{_mode_tag}")
    print("=" * 60)

    sessions = cfg.get("obs_sessions", [])
    if not sessions:
        raise ValueError("observation_config.yaml 裡沒有 obs_sessions。")

    for session in sessions:
        # yaml targets 是列表；逐一處理每個目標
        target_list = session.get("targets", [])
        if not target_list:
            print(f"[SKIP] session {session.get('date', '?')}：targets 為空。")
            continue

        date = str(session["date"])
        data_root = cfg["_data_root"]
        project_root = cfg["_project_root"]
        astap_tmp_root = _resolve_astap_tmp_root(project_root, astap_cfg)

        for group_info in _session_groups(cfg, target_list):
            _group = str(group_info["group"])
            hint_target = str(group_info["hint_target"])
            group_targets = [str(t) for t in group_info["targets"]]
            _date_fmt = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
            field_root = data_root / _date_fmt / _group
            output_dir = project_root / "output" / _date_fmt / _group
            if src_dir:
                cal_dir = field_root / src_dir
                wcs_dir = field_root / (wcs_out_dir or src_dir)
            elif raw_mode:
                cal_dir = field_root / "raw"
                wcs_dir = field_root / "wcs_raw"
            else:
                cal_dir = field_root / "wcs"
                wcs_dir = cal_dir  # WCS output to same directory
            wcs_dir.mkdir(parents=True, exist_ok=True)
            if not cal_dir.exists():
                print(f"[SKIP] {_group}/{date}: source directory not found: {cal_dir}")
                continue

            # ── 收集 FITS + CR2 檔案清單 ─────────────────────────────
            fits_files = sorted(
                f for f in cal_dir.glob("*.fits")
                if f.is_file() and "wcs" not in f.stem.lower()
            )
            cr2_files = sorted(
                f for f in cal_dir.iterdir()
                if f.is_file() and f.suffix.lower() == ".cr2"
            )
            if cr2_files:
                print(f"  [CR2] 偵測到 {len(cr2_files)} 個 CR2 待解算")

            # 合併 FITS 和 CR2（CR2 需逐檔暫存轉換）
            all_sources = [(f, False) for f in fits_files] + \
                          [(f, True) for f in cr2_files]
            total_count = len(all_sources)

            ra_hint, spd_hint = _get_hint_for_target(cfg, hint_target)
            max_offset_deg = _validation_max_offset_deg(
                astap_cfg,
                ra_hint,
                spd_hint,
            )
            targets_text = ", ".join(group_targets)
            if ra_hint is not None:
                print(
                    f"\n[Session] {_group} / {date}  ({total_count} 幀)"
                    f"  hint target={hint_target} RA={ra_hint:.4f}h SPD={spd_hint:.4f}°"
                    f"  targets=[{targets_text}]"
                )
            else:
                print(
                    f"\n[Session] {_group} / {date}  ({total_count} 幀)"
                    f"  targets=[{targets_text}]"
                    f"  hint 未設定（yaml 無目標座標）"
                )

            if not all_sources:
                _src = "raw/" if raw_mode else "wcs/"
                print(f"[SKIP] {_group}/{date}：{_src} 目錄裡找不到 FITS/CR2。")
                continue

            success = failed = skipped = 0
            for src_path, is_cr2 in all_sources:
                out_path = wcs_dir / (src_path.stem + "_wcs.fits")

                if out_path.exists():
                    valid, reason = _validate_wcs_file(
                        out_path,
                        ra_hint_h=ra_hint,
                        spd_hint_deg=spd_hint,
                        max_offset_deg=max_offset_deg,
                    )
                    if valid:
                        skipped += 1
                        continue
                    print(
                        f"  [WARN] 既有 WCS 驗證失敗，將重解："
                        f"{out_path.name} ({reason})"
                    )

                # CR2：逐檔轉暫存 FITS，解算後刪除
                tmp_fits = None
                if is_cr2:
                    try:
                        data, hdr = read_raw_image(src_path)
                        tmp_fits = cal_dir / (src_path.stem + "_tmp.fits")
                        fits.writeto(tmp_fits, data, header=hdr,
                                     overwrite=True)
                        solve_path = tmp_fits
                    except Exception as exc:
                        print(f"  [WARN] CR2→FITS 失敗：{src_path.name} — {exc}")
                        failed += 1
                        continue
                else:
                    solve_path = src_path

                print(f"  解算：{src_path.name} … ", end="", flush=True)

                if backend == "astap":
                    ok = _run_astap(
                        solve_path,
                        out_path,
                        astap_cfg,
                        ra_hint_h=ra_hint,
                        spd_hint_deg=spd_hint,
                        max_offset_deg=max_offset_deg,
                        tmp_root=astap_tmp_root,
                    )
                elif backend == "astrometry_net":
                    ok = _run_astrometry_net(
                        solve_path,
                        out_path,
                        anet_cfg,
                        ra_hint_h=ra_hint,
                        spd_hint_deg=spd_hint,
                        max_offset_deg=max_offset_deg,
                    )
                else:
                    raise ValueError(f"Unsupported astrometry backend: {backend!r}")

                # 清理暫存
                if tmp_fits is not None and tmp_fits.exists():
                    tmp_fits.unlink()

                if ok:
                    print("OK")
                    success += 1
                else:
                    print("FAIL")
                    failed += 1

            print(
                f"\n[完成] {_group}/{date}：成功 {success}，失敗 {failed}，"
                f"已跳過（存在）{skipped}"
            )
            print(f"       WCS 輸出目錄：{wcs_dir}")

            drift_suffix = "raw" if raw_mode else "cal"
            drift_png = output_dir / f"plate_solve_drift_{drift_suffix}.png"
            drift_title = f"Drift {date} ({_group})"
            wcs_files = sorted(wcs_dir.glob("*_wcs.fits"))
            if _plot_drift_report(wcs_files, drift_png, drift_title):
                print(f"       Drift plot: {drift_png}")

    print("\n" + "=" * 20)
    print("所有 Session 星圖解算完成。")
    print("Next step: split.py -> Photometry.ipynb")
    print("=" * 20)


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        default_cfg = Path(__file__).parent / "observation_config.yaml"
        cfg_path = default_cfg
    else:
        cfg_path = Path(sys.argv[1])

    run_plate_solve(cfg_path)
