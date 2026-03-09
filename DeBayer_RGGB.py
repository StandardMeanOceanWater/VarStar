# -*- coding: utf-8 -*-
"""
DeBayer_RGGB.py  —  Bayer 拆色模組
專案：變星測光管線 v1.0
描述：讀取 plate solve 後的 WCS FITS，按 Bayer pattern 拆分為
      R / G1 / G2 / B 四個通道，各自輸出為獨立 FITS。

      完整繼承 FITS 標頭（含 DATE-OBS、MID-OBS、BJD_TDB 預留欄位），
      並依 stride=2 子採樣規則修正 WCS（CRPIX、CD matrix）。

      G2 與 G1 為同一波段，G2 僅作品質驗證用，不用於獨立科學輸出。

WCS 修正規則（stride=2 子採樣）
---------------------------------
    CRPIX_new = (CRPIX_orig - offset - 0.5) / 2.0 + 0.5
    CD{i}_{j}_new = CD{i}_{j}_orig * 2.0

    其中 offset 為各通道的 Bayer 子格偏移（列、行）：
        R  : row_offset=0, col_offset=0
        G1 : row_offset=0, col_offset=1
        G2 : row_offset=1, col_offset=0
        B  : row_offset=1, col_offset=1
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from astropy.io import fits
from tqdm import tqdm

from Calibration import load_config


# =============================================================================
# Bayer pattern 定義
# =============================================================================

# 每個 pattern 的定義格式：
# channel : (row_slice, col_slice, row_offset, col_offset)
_BAYER_PATTERNS: Dict[str, Dict[str, Tuple[slice, slice, int, int]]] = {
    "RGGB": {
        "R":  (slice(0, None, 2), slice(0, None, 2), 0, 0),
        "G1": (slice(0, None, 2), slice(1, None, 2), 0, 1),
        "G2": (slice(1, None, 2), slice(0, None, 2), 1, 0),
        "B":  (slice(1, None, 2), slice(1, None, 2), 1, 1),
    },
    "BGGR": {
        "B":  (slice(0, None, 2), slice(0, None, 2), 0, 0),
        "G1": (slice(0, None, 2), slice(1, None, 2), 0, 1),
        "G2": (slice(1, None, 2), slice(0, None, 2), 1, 0),
        "R":  (slice(1, None, 2), slice(1, None, 2), 1, 1),
    },
    "GRBG": {
        "G1": (slice(0, None, 2), slice(0, None, 2), 0, 0),
        "R":  (slice(0, None, 2), slice(1, None, 2), 0, 1),
        "B":  (slice(1, None, 2), slice(0, None, 2), 1, 0),
        "G2": (slice(1, None, 2), slice(1, None, 2), 1, 1),
    },
    "GBRG": {
        "G1": (slice(0, None, 2), slice(0, None, 2), 0, 0),
        "B":  (slice(0, None, 2), slice(1, None, 2), 0, 1),
        "R":  (slice(1, None, 2), slice(0, None, 2), 1, 0),
        "G2": (slice(1, None, 2), slice(1, None, 2), 1, 1),
    },
}


# =============================================================================
# Bayer pattern 自動偵測
# =============================================================================

def detect_bayer_pattern(
    header: fits.Header,
    yaml_default: str = "RGGB",
) -> str:
    """
    自動偵測 Bayer pattern，優先順序：

    1. FITS 標頭 BAYERPAT（ZWO 相機會寫入）
    2. FITS 標頭 COLORTYP（部分軟體使用）
    3. yaml 設定的預設值（Canon 6D2 = RGGB）

    Parameters
    ----------
    header      : FITS 標頭。
    yaml_default: yaml cameras.{model}.bayer_pattern 的值。

    Returns
    -------
    str
        四字元 Bayer pattern 字串，例如 "RGGB"。
    """
    for key in ("BAYERPAT", "COLORTYP"):
        val = header.get(key, "")
        if isinstance(val, str):
            val = val.strip().upper()
            if val in _BAYER_PATTERNS:
                return val

    default = str(yaml_default).strip().upper()
    if default in _BAYER_PATTERNS:
        return default

    print(
        f"  [WARN] 無法偵測 Bayer pattern（標頭無 BAYERPAT，"
        f"yaml 預設 '{yaml_default}' 不認識），退而使用 RGGB。"
    )
    return "RGGB"


# =============================================================================
# WCS 傳遞
# =============================================================================

def _propagate_wcs(
    header_orig: fits.Header,
    row_offset: int,
    col_offset: int,
) -> fits.Header:
    """
    將全幅影像的 WCS 換算到 stride=2 子採樣通道的 WCS。

    換算規則：
        CRPIX1_new = (CRPIX1_orig - col_offset - 0.5) / 2.0 + 0.5
        CRPIX2_new = (CRPIX2_orig - row_offset - 0.5) / 2.0 + 0.5
        CD{i}_{j}_new = CD{i}_{j}_orig * 2.0
    """
    h = header_orig.copy()

    if "CRPIX1" in h:
        h["CRPIX1"] = (h["CRPIX1"] - col_offset - 0.5) / 2.0 + 0.5
    if "CRPIX2" in h:
        h["CRPIX2"] = (h["CRPIX2"] - row_offset - 0.5) / 2.0 + 0.5

    for i in (1, 2):
        for j in (1, 2):
            key = f"CD{i}_{j}"
            if key in h:
                h[key] = h[key] * 2.0

    for axis in (1, 2):
        key = f"CDELT{axis}"
        if key in h:
            h[key] = h[key] * 2.0

    return h


# =============================================================================
# 標頭建構
# =============================================================================

def _build_channel_header(
    orig_header: fits.Header,
    channel_name: str,
    orig_shape: tuple,
    new_shape: tuple,
    bayer_pattern: str,
    row_offset: int,
    col_offset: int,
    telescope_name: str = "",
    camera_name: str = "",
    focal_length_mm: Optional[float] = None,
    pixel_size_um: Optional[float] = None,
    observer: str = "",
    object_name: str = "",
) -> fits.Header:
    """
    從原始 FITS 標頭繼承所有關鍵資訊，更新拆色後的通道標頭。
    特別保留 DATE-OBS 和 MID-OBS 供測光模組計算 BJD_TDB。
    """
    h = _propagate_wcs(orig_header, row_offset, col_offset)

    h["NAXIS1"] = new_shape[1]
    h["NAXIS2"] = new_shape[0]
    h["BITPIX"] = -32

    if pixel_size_um is not None:
        eff_pixel = pixel_size_um * 2.0
        h["XPIXSZ"] = (eff_pixel, "[um] Effective pixel size after Bayer split")
        h["YPIXSZ"] = (eff_pixel, "[um] Effective pixel size after Bayer split")
        if focal_length_mm is not None and focal_length_mm > 0:
            plate_scale = 206.265 * eff_pixel / focal_length_mm
            h["SCALE"] = (plate_scale, "[arcsec/px] Plate scale after Bayer split")

    if telescope_name:
        h["TELESCOP"] = (telescope_name, "Telescope")
    if camera_name:
        h["INSTRUME"] = (camera_name, "Camera")
    if focal_length_mm is not None:
        h["FOCALLEN"] = (focal_length_mm, "[mm] Focal length")
    if observer:
        h["OBSERVER"] = (observer, "Observer")
    if object_name:
        h["OBJECT"] = (object_name, "Target object")

    h["CHANNEL"] = (channel_name, "Bayer sub-channel extracted")
    h["BAYERPAT"] = (bayer_pattern, "Original Bayer pattern")
    h["DEBAYER"] = ("NO", "Split by slicing - no interpolation")

    if "DATE-OBS" not in h:
        print(f"  [WARN] {channel_name}：原始 FITS 缺少 DATE-OBS，BJD_TDB 將無法計算。")
    if "MID-OBS" not in h:
        print(f"  [WARN] {channel_name}：原始 FITS 缺少 MID-OBS。")

    proc_time = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    h["PROCDATE"] = (proc_time, "UTC time of Bayer split processing")
    h["HISTORY"] = (
        f"Bayer split: channel {channel_name}, stride=2, "
        f"row_offset={row_offset}, col_offset={col_offset}"
    )
    h["HISTORY"] = "Data unit: ADU (float32, linear, calibrated)"

    return h


# =============================================================================
# 主拆色管線
# =============================================================================

def run_debayer(config_path: "str | Path") -> None:
    """
    主 Bayer 拆色管線入口。

    讀取 observation_config.yaml，對每個 obs_session 的
    calibrated/wcs/ 目錄內所有 WCS FITS 執行拆色，
    輸出到 split/{R,G1,G2,B}/ 子目錄。

    Parameters
    ----------
    config_path : observation_config.yaml 的路徑。
    """
    cfg = load_config(config_path)
    sessions = cfg.get("obs_sessions", [])
    if not sessions:
        raise ValueError("observation_config.yaml 裡沒有 obs_sessions。")

    print("\n" + "=" * 60)
    print("  變星測光管線 — Bayer 拆色模組  DeBayer_RGGB.py")
    print("=" * 60)

    for session in sessions:
        # 支援 targets 列表（複數）與舊格式 target 單數
        targets_raw = session.get("targets", session.get("target"))
        if targets_raw is None:
            print("[WARN] session 缺少 targets 欄位，跳過。")
            continue
        targets_list = (
            targets_raw if isinstance(targets_raw, list) else [targets_raw]
        )

        date = str(session["date"])
        telescope_id = session.get("telescope", "")
        camera_id = session.get("camera", "")

        tel_cfg = cfg.get("telescopes", {}).get(telescope_id, {})
        cam_cfg = cfg.get("cameras", {}).get(camera_id, {})
        obs_cfg = cfg.get("observatory", {})

        telescope_name = tel_cfg.get("name", telescope_id)
        camera_name = cam_cfg.get("name", camera_id)
        focal_length_mm = tel_cfg.get("focal_length_mm")
        pixel_size_um = cam_cfg.get("pixel_size_um")
        bayer_default = cam_cfg.get("bayer_pattern", "RGGB")
        observer = obs_cfg.get("name", "")

        data_root = cfg["_data_root"]

        for target in targets_list:
            target_cfg = cfg.get("targets", {}).get(target, {})
            object_name = target_cfg.get("display_name", target)

            target_root = data_root / "targets" / target
            wcs_dir = target_root / "calibrated" / "wcs"
            split_base = target_root / "split"

            wcs_files = sorted(wcs_dir.glob("*_wcs.fits"))
            if not wcs_files:
                print(f"[SKIP] {target}/{date}：wcs/ 目錄裡找不到 *_wcs.fits。")
                continue

            print(f"\n[Session] {target} / {date}  ({len(wcs_files)} 幀)")

            for ch in ("R", "G1", "G2", "B"):
                (split_base / ch).mkdir(parents=True, exist_ok=True)

            success = failed = 0
            for fits_path in tqdm(wcs_files, desc=f"{target} 拆色進度"):
                try:
                    with fits.open(fits_path) as hdul:
                        hdu = next((h for h in hdul if h.data is not None), None)
                        if hdu is None:
                            print(f"\n  [跳過] {fits_path.name}：無數據層。")
                            continue
                        data = hdu.data.astype(np.float32)
                        header = hdu.header.copy()

                    if data.ndim != 2:
                        print(
                            f"\n  [跳過] {fits_path.name}：非 2D（ndim={data.ndim}），"
                            "疑似已 Debayer。"
                        )
                        continue

                    bayer_pattern = detect_bayer_pattern(header, bayer_default)
                    pattern_def = _BAYER_PATTERNS[bayer_pattern]
                    base_name = fits_path.stem

                    for ch_key, (row_sl, col_sl, row_off, col_off) in pattern_def.items():
                        ch_data = data[row_sl, col_sl].copy().astype(np.float32)
                        # ch_data = np.clip(ch_data, 0.0, None)  #GEMINI建議刪除

                        ch_header = _build_channel_header(
                            orig_header=header,
                            channel_name=ch_key,
                            orig_shape=data.shape,
                            new_shape=ch_data.shape,
                            bayer_pattern=bayer_pattern,
                            row_offset=row_off,
                            col_offset=col_off,
                            telescope_name=telescope_name,
                            camera_name=camera_name,
                            focal_length_mm=focal_length_mm,
                            pixel_size_um=pixel_size_um,
                            observer=observer,
                            object_name=object_name,
                        )

                        out_name = f"{base_name}_{ch_key}.fits"
                        out_path = split_base / ch_key / out_name
                        new_hdu = fits.PrimaryHDU(data=ch_data, header=ch_header)
                        new_hdu.verify("silentfix")
                        new_hdu.writeto(out_path, overwrite=True)

                    success += 1

                except Exception as exc:
                    print(f"\n  [失敗] {fits_path.name}：{exc}")
                    failed += 1

            print(f"\n[完成] {target}/{date}：成功 {success} 幀，失敗 {failed} 幀")
            print(f"       輸出目錄：{split_base}/{{R,G1,G2,B}}/")

    print("\n" + "🔭 " * 20)
    print("所有 Session 拆色完成。")
    print("G2 通道僅供品質驗證（G1 vs G2 residual rms = PRNU 估計值）。")
    print("下一步：Photometry.ipynb")
    print("🔭 " * 20)


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        default_cfg = Path(__file__).parent.parent / "observation_config.yaml"
        cfg_path = default_cfg
    else:
        cfg_path = Path(sys.argv[1])

    run_debayer(cfg_path)
