# -*- coding: utf-8 -*-
"""
Calibration.py  —  天文科學級單張校正管線
專案：變星測光管線 v0.99
描述：讀取 observation_config.yaml，對 CR2 / FITS 格式的觀測幀執行
      Bias / Dark / Flat 校正，輸出保留 Bayer 排列的 float32 線性 FITS。

學理基礎
--------
校正公式（Howell, 2006）：
    有 Dark:  Cal = (Light − Master_Dark) / Master_Flat_norm
    無 Dark:  Cal = (Light − Master_Bias) / Master_Flat_norm
    無 Flat:  Cal = Light − Master_Dark（或 Master_Bias）

Master 合成採用二階 Remedian 近似中位數（Rousseeuw & Bassett, 1990），
RAM 用量從全載的 O(N) 降至 O(chunk_size)。

時間系統（Eastman et al., 2010）：
    EXIF 台灣時間（UTC+8）→ 減 8 小時 → DATE-OBS（UTC）
    DATE-OBS + EXPTIME/2  → MID-OBS（曝光中點，UTC）

絕對禁止對輸出做 Bayer 插值（Debayer）。

限制
----
- 僅接受二維（單色）FITS 輸入，三維彩色 FITS 視為錯誤。
- 相機線性度校正（Linearity Correction）超出本模組範圍，
  輸出 FITS 的 COMMENT 欄位會提醒此限制。
"""

from __future__ import annotations

import gc
import os
import re
import shutil
import tempfile
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from tqdm import tqdm

try:
    import rawpy
    import exifread
except ImportError:
    print("[系統] 正在安裝必備套件 rawpy, exifread …")
    os.system("pip install rawpy exifread --quiet")
    import rawpy
    import exifread

warnings.simplefilter("ignore", category=AstropyWarning)

# ── 型別別名 ─────────────────────────────────────────────────────────────────
Header = fits.Header


# =============================================================================
# 設定載入
# =============================================================================

def _detect_project_root(cfg: dict, config_path: Path) -> Path:
    """
    自動偵測執行環境，回傳對應的 project_root。

    若 yaml 填相對路徑（例如 ".."），以 yaml 檔所在目錄為錨點解析，
    使整個專案資料夾可自由搬移而不需修改 yaml。
    """
    try:
        import google.colab  # noqa: F401
        root = cfg["paths"]["colab"]["project_root"]
    except (ImportError, KeyError):
        root = cfg["paths"]["local"]["project_root"]
    p = Path(root)
    if not p.is_absolute():
        p = (config_path.parent / p).resolve()
    return p


def load_config(config_path: str | Path) -> dict:
    """
    載入 observation_config.yaml。

    Parameters
    ----------
    config_path : str | Path
        yaml 檔案路徑。

    Returns
    -------
    dict
        展開後的設定字典，已加入 ``_project_root`` 和 ``_data_root``。

    Raises
    ------
    FileNotFoundError
        找不到 yaml 檔案。
    KeyError
        yaml 缺少必要欄位。
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"找不到設定檔：{config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    project_root = _detect_project_root(cfg, config_path)
    cfg["_project_root"] = project_root
    cfg["_data_root"] = project_root / "data"
    return cfg


# =============================================================================
# 路徑解析
# =============================================================================

def resolve_session_paths(
    cfg: dict,
    session: dict,
) -> Dict[str, Path]:
    """
    根據 obs_session 設定解析所有輸入輸出路徑。

    搜尋 calibration 幀的邏輯：
        calibration 幀統一存放在 data/share/calibration/{date}_{telescope}_{camera}/{type}/

    Dark 子目錄選取（_find_dark_dir）：
        1. dark/ 根層直接有影像 → 直接使用（向下相容）
        2. 有子目錄：
           a. 解析子目錄名稱，格式 ^[數字][.數字]C$（例如 3.7C、5C）
           b. 若 session 有 light_temp_c：選溫度最接近者
           c. 若無 light_temp_c：選溫度最低者（dark current 最小，殘差方向較安全）
           d. 子目錄無法解析溫度：[WARN] 後選第一個（按名稱排序）

    Parameters
    ----------
    cfg     : 完整設定字典。
    session : obs_sessions 裡的單一 session 項目。

    Returns
    -------
    Dict[str, Path]
        包含 light_dir, dark_dir, flat_dir, bias_dir,
        calibrated_dir, masters_dir, dark_temp_c 的字典。
        dark_temp_c 為 float 或 None，供 run_calibration() 寫入 FITS 標頭。
    """
    # ── 正則：子目錄溫度解析，例如 "3.7C"、"5C"、"10.0c" ──────────────────
    _TEMP_RE = re.compile(r'^(\d+\.?\d*)[Cc]$')

    data_root = cfg["_data_root"]
    date = str(session["date"])
    # 優先讀單數 "target"（由 run_calibration 迴圈注入）；
    # 若不存在才從 "targets" list 取第一個（直接呼叫時的 fallback）。
    if "target" in session and session["target"]:
        target = str(session["target"])
    else:
        raw_targets = session.get("targets", "")
        target = (
            raw_targets[0]
            if isinstance(raw_targets, list)
            else str(raw_targets)
        )

    _tgt_cfg = cfg.get("targets", {}).get(target, {})
    _group = _tgt_cfg.get("group", target)
    _date_fmt = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    field_root = data_root / _date_fmt / _group
    _cal_label = session.get("cal_label")
    if not _cal_label:
        _cal_label = f"{session.get('telescope', '')}_{session.get('camera', '')}"
    _cal_label = f"{date}_{_cal_label}"
    shared_cal = data_root / "share" / "calibration" / _cal_label

    light_temp_c: Optional[float] = session.get("light_temp_c")
    if light_temp_c is not None:
        light_temp_c = float(light_temp_c)

    def _find_cal_dir(frame_type: str) -> Optional[Path]:
        """搜尋 shared calibration 目錄。"""
        shared = shared_cal / frame_type
        if shared.exists() and any(shared.iterdir()):
            return shared
        return None

    def _find_flat_dirs_by_format() -> Dict[str, Path]:
        """
        掃描所有 flat* 目錄，依檔案格式分類回傳。

        Returns
        -------
        dict : {"cr2": Path, "fits": Path} 視實際存在而定。
        """
        result: Dict[str, Path] = {}
        if not shared_cal.exists():
            return result
        for d in sorted(shared_cal.iterdir()):
            if not d.is_dir() or not d.name.lower().startswith("flat"):
                continue
            files = [f for f in d.iterdir() if f.is_file()]
            if not files:
                continue
            has_cr2 = any(f.suffix.lower() == ".cr2" for f in files)
            has_fits = any(f.suffix.lower() in (".fits", ".fit") for f in files)
            # 每種格式只取第一個找到的目錄
            if has_cr2 and "cr2" not in result:
                result["cr2"] = d
            if has_fits and "fits" not in result:
                result["fits"] = d
        return result

    def _parse_temp(subdir: Path) -> Optional[float]:
        """嘗試從子目錄名稱解析溫度（°C）；失敗回傳 None。"""
        m = _TEMP_RE.match(subdir.name)
        return float(m.group(1)) if m else None

    def _has_images(directory: Path) -> bool:
        """判斷目錄內是否有支援格式的影像（不遞迴）。"""
        valid_exts = {".fit", ".fits", ".cr2"}
        return any(
            f.is_file() and f.suffix.lower() in valid_exts
            for f in directory.iterdir()
        )

    def _find_dark_dir() -> Tuple[Optional[Path], Optional[float]]:
        """
        找 dark 目錄，回傳 (path, resolved_dark_temp_c)。

        resolved_dark_temp_c 優先取 session['dark_temp_c']；
        若未填，則從子目錄名稱解析。
        """
        # 候選位置
        candidates = [
            shared_cal / "dark",
        ]
        explicit_temp: Optional[float] = session.get("dark_temp_c")
        if explicit_temp is not None:
            explicit_temp = float(explicit_temp)

        for dark_root in candidates:
            if not dark_root.exists():
                continue

            # ── 情況 1：根層直接有影像 ────────────────────────────────────
            if _has_images(dark_root):
                return dark_root, explicit_temp

            # ── 情況 2：有子目錄 ──────────────────────────────────────────
            subdirs = sorted(
                [d for d in dark_root.iterdir() if d.is_dir()],
                key=lambda d: d.name,
            )
            if not subdirs:
                continue

            # 解析各子目錄溫度
            parsed: List[Tuple[float, Path]] = []
            unparsed: List[Path] = []
            for sd in subdirs:
                if not _has_images(sd):
                    continue
                t = _parse_temp(sd)
                if t is not None:
                    parsed.append((t, sd))
                else:
                    unparsed.append(sd)

            if parsed:
                ref_temp = light_temp_c  # 以觀測溫度為基準
                if ref_temp is not None:
                    # 選最接近觀測溫度的 dark
                    chosen_temp, chosen_dir = min(
                        parsed, key=lambda x: abs(x[0] - ref_temp)
                    )
                else:
                    # 無觀測溫度：選最低溫（dark current 最小）
                    chosen_temp, chosen_dir = min(parsed, key=lambda x: x[0])
                    print(
                        f"  [WARN] session 未填 light_temp_c，"
                        f"dark 子目錄自動選最低溫：{chosen_dir.name}"
                    )
                resolved_temp = (
                    explicit_temp
                    if explicit_temp is not None
                    else chosen_temp
                )
                return chosen_dir, resolved_temp

            # 全部無法解析溫度
            if unparsed:
                print(
                    f"  [WARN] dark 子目錄名稱無法解析溫度，"
                    f"已選第一個：{unparsed[0].name}。"
                    f"可用子目錄：{[d.name for d in unparsed]}"
                )
                return unparsed[0], explicit_temp

        return None, None

    dark_dir, resolved_dark_temp = _find_dark_dir()

    return {
        "light_dir": field_root / "raw",
        "dark_dir": dark_dir,
        "flat_dir": _find_cal_dir("flat"),
        "flat_dirs_by_format": _find_flat_dirs_by_format(),
        "bias_dir": _find_cal_dir("bias"),
        "calibrated_dir": field_root / "wcs",
        "masters_dir": shared_cal / "masters",
        "dark_temp_c": resolved_dark_temp,
        "light_temp_c": light_temp_c,
    }


# =============================================================================
# 多格式影像讀取
# =============================================================================

def _exif_to_utc(date_str: str, tz_offset_hours: int) -> str:
    """
    將 EXIF 時間字串（相機本地時間）轉換為 ISO 8601 UTC 字串。

    Parameters
    ----------
    date_str        : EXIF DateTimeOriginal，格式 "%Y:%m:%d %H:%M:%S"。
    tz_offset_hours : 相機設定的時區偏移，台灣 = 8。

    Returns
    -------
    str
        ISO 8601 UTC 字串，例如 "2024-12-20T13:45:30"。
    """
    dt_local = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
    dt_utc = dt_local - timedelta(hours=tz_offset_hours)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%S")


def read_raw_image(
    filepath: str | Path,
    tz_offset_hours: int = 8,
    gain_e_per_adu: Optional[float] = None,
    read_noise_e: Optional[float] = None,
    saturation_adu: Optional[float] = None,
    iso: Optional[int] = None,
) -> Tuple[np.ndarray, Header]:
    """
    多格式混合讀取引擎。

    無論輸入為 CR2 或 FITS，統一回傳：
    - 保留 Bayer 排列的單色線性 float32 矩陣（已減去黑電平）
    - 含有時間戳記和儀器參數的 FITS 標頭

    Parameters
    ----------
    filepath        : 輸入檔案路徑（.cr2 / .fit / .fits）。
    tz_offset_hours : EXIF 時間的時區偏移（台灣 = 8）。
    gain_e_per_adu  : 相機增益（e⁻/DN），由 sensor_db 傳入。
    read_noise_e    : 讀出雜訊（e⁻），由 sensor_db 傳入。
    saturation_adu  : 飽和閾值（DN）。
    iso             : 相機 ISO 設定；None 表示未知。

    Returns
    -------
    (data, header) : float32 二維陣列、FITS 標頭。

    Raises
    ------
    ValueError
        FITS 無數據層，或數據為非二維（疑似已 Debayer）。
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    header = fits.Header()

    # ── 分支 A：DSLR CR2 ──────────────────────────────────────────────────────
    if suffix == ".cr2":
        fd, tmp = tempfile.mkstemp(suffix=".cr2")
        os.close(fd)
        try:
            shutil.copy2(filepath, tmp)

            # A-1：EXIF 時間 → DATE-OBS（UTC）
            date_obs = "UNKNOWN"
            exptime: Optional[float] = None
            with open(tmp, "rb") as fh:
                tags = exifread.process_file(
                    fh,
                    stop_tag="EXIF DateTimeOriginal",
                    details=False,
                )
                if "EXIF DateTimeOriginal" in tags:
                    raw_str = str(tags["EXIF DateTimeOriginal"])
                    try:
                        date_obs = _exif_to_utc(raw_str, tz_offset_hours)
                    except ValueError as exc:
                        print(f"  [WARN] EXIF 時間解析失敗（{exc}），DATE-OBS 設為 UNKNOWN。")

                for key in ("EXIF ExposureTime", "Image ExposureTime"):
                    if key in tags:
                        try:
                            exptime = float(tags[key].values[0].num
                                            / tags[key].values[0].den)
                        except Exception:
                            pass
                        break

            # A-2：rawpy 讀取，黑電平正確性要求：
            #       Canon 的 black_level_per_channel 是每個 Bayer 通道分別的值，
            #       取平均足以用於單色校正。
            with rawpy.imread(tmp) as raw:
                bl = float(np.mean(raw.black_level_per_channel))
                data = raw.raw_image_visible.astype(np.float32) - bl

            # MID-OBS：曝光中點（供測光模組計算 BJD_TDB）
            mid_obs = "UNKNOWN"
            if date_obs != "UNKNOWN" and exptime is not None:
                try:
                    dt_start = datetime.strptime(date_obs, "%Y-%m-%dT%H:%M:%S")
                    dt_mid = dt_start + timedelta(seconds=exptime / 2.0)
                    mid_obs = dt_mid.strftime("%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    pass

            header["INSTRUME"] = ("DSLR_CR2", "Camera")
            header["DATE-OBS"] = (date_obs, "UTC exposure start (EXIF - timezone offset)")
            header["MID-OBS"] = (mid_obs, "UTC exposure midpoint (DATE-OBS + EXPTIME/2)")
            header["BLACKLVL"] = (bl, "[DN] Mean black level subtracted")
            header["BAYERPAT"] = ("RGGB", "Bayer pattern")
            if exptime is not None:
                header["EXPTIME"] = (exptime, "[s] Exposure time")

        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # ── 分支 B：FITS ──────────────────────────────────────────────────────────
    elif suffix in (".fit", ".fits"):
        with fits.open(filepath, ignore_missing_end=True) as hdul:
            hdu = next((h for h in hdul if h.data is not None), None)
            if hdu is None:
                raise ValueError(f"此 FITS 沒有數據層：{filepath.name}")

            data = hdu.data.astype(np.float32)
            header = hdu.header.copy() if hdu.header else fits.Header()

            if data.ndim != 2:
                raise ValueError(
                    f"不合格：只允許單色（2D）數據。"
                    f"{filepath.name} 為 {data.ndim}D，疑似已做 Debayer。"
                )

            # 移除 BZERO / BSCALE 避免重複縮放
            for key in ("BZERO", "BSCALE"):
                header.remove(key, ignore_missing=True)
    else:
        raise ValueError(f"不支援的格式：{filepath.suffix}")

    # ── 共用：寫入儀器參數標頭（A-2）────────────────────────────────────────
    if gain_e_per_adu is not None:
        header["GAIN"] = (float(gain_e_per_adu), "[e-/DN] CCD gain from sensor_db")
    if read_noise_e is not None:
        header["RDNOISE"] = (float(read_noise_e), "[e-] Read noise from sensor_db")
    if saturation_adu is not None:
        header["SATURATE"] = (float(saturation_adu), "[DN] 70% full-well saturation threshold")
    if iso is not None and iso > 0:
        header["ISOSPEED"] = (int(iso), "[ISO] Camera ISO speed setting")

    header["DEBAYER"] = ("NO", "Preserved as RAW Bayer data - do NOT interpolate")
    header.add_comment(
        "Linearity correction not applied. "
        "Canon 6D2 linearity error < 0.5% in 30%-85% full-well range "
        "(< 0.005 mag). Acceptable for current photometric precision."
    )

    return data, header


# =============================================================================
# Master 幀合成
# =============================================================================

def _list_image_files(directory: Optional[Path]) -> List[Path]:
    """回傳目錄內所有支援格式的影像檔路徑清單。"""
    if directory is None or not directory.exists():
        return []
    valid_exts = {".fit", ".fits", ".cr2"}
    files = [
        f for f in directory.rglob("*")
        if f.is_file()
        and f.suffix.lower() in valid_exts
        and not f.name.startswith("._")
    ]
    return sorted(set(files))


def create_master(
    file_list: List[Path],
    desc: str,
    chunk_size: int = 10,
    tz_offset_hours: int = 8,
) -> np.ndarray:
    """
    採用二階 Remedian 近似中位數合成 Master 幀。

    Remedian 演算法（Rousseeuw & Bassett, 1990）：
        每 chunk_size 張取一次中位數 → 收集所有中位數再取一次中位數。
        記憶體用量：O(chunk_size)，偏差遠小於逐幀噪聲。

    Parameters
    ----------
    file_list       : 待合成的影像檔路徑清單。
    desc            : 進度條說明文字。
    chunk_size      : 每批載入的幀數（控制 RAM 用量）。
    tz_offset_hours : CR2 讀取時的時區偏移。

    Returns
    -------
    np.ndarray (float32)
        合成後的 Master 幀。

    Raises
    ------
    ValueError
        file_list 為空，或各幀尺寸不一致。
    """
    if not file_list:
        raise ValueError(f"沒有 {desc} 幀，無法合成 Master。")

    print(f"\n[Master] 合成 {desc}（共 {len(file_list)} 張）…")

    # 確認參考尺寸
    ref_data, _ = read_raw_image(file_list[0], tz_offset_hours=tz_offset_hours)
    ref_shape = ref_data.shape
    del ref_data
    gc.collect()

    chunk_medians: List[np.ndarray] = []

    for i in tqdm(range(0, len(file_list), chunk_size), desc=desc):
        batch = file_list[i: i + chunk_size]
        stack: List[np.ndarray] = []
        for fp in batch:
            try:
                d, _ = read_raw_image(fp, tz_offset_hours=tz_offset_hours)
                if d.shape != ref_shape:
                    print(f"  [WARN] 尺寸不符，跳過：{fp.name} "
                          f"({d.shape} ≠ {ref_shape})")
                    continue
                stack.append(d)
            except Exception as exc:
                print(f"  [WARN] 讀取失敗，跳過：{fp.name}  ({exc})")
        if stack:
            chunk_medians.append(
                np.nanmedian(np.array(stack, dtype=np.float32), axis=0).astype(np.float32)
            )
        del stack
        gc.collect()

    if not chunk_medians:
        raise ValueError(f"所有 {desc} 幀均讀取失敗。")

    master = np.nanmedian(
        np.array(chunk_medians, dtype=np.float32), axis=0
    ).astype(np.float32)

    del chunk_medians
    gc.collect()
    return master


# =============================================================================
# 平場後處理
# =============================================================================

def normalize_flat(
    flat_raw: np.ndarray,
    master_bias: Optional[np.ndarray],
    bad_pixel_threshold: float = 0.3,
    sigma_clip_iters: int = 3,
    sigma_clip_sigma: float = 3.0,
) -> np.ndarray:
    """
    平場歸一化，含 sigma clipping 和壞像素保護。

    步驟：
        1. 若有 Bias，先減去（消除 offset）
        2. Sigma clipping 剔除異常像素（宇宙射線、熱像素）
        3. 以 clipping 後的中位數歸一化
        4. 將歸一化值 clip 到 [bad_pixel_threshold, 5.0]

    Parameters
    ----------
    flat_raw             : 原始 Master Flat（未減 Bias）。
    master_bias          : Master Bias，None 表示跳過。
    bad_pixel_threshold  : 歸一化後低於此值視為壞像素（default 0.3）。
    sigma_clip_iters     : Sigma clipping 最大迭代次數。
    sigma_clip_sigma     : Sigma clipping 的 σ 倍數。

    Returns
    -------
    np.ndarray (float32)
        歸一化後的 Master Flat。

    Raises
    ------
    ValueError
        歸一化中位數 ≤ 0（平場全黑損毀）。
    """
    flat = flat_raw.copy().astype(np.float64)
    if master_bias is not None:
        flat -= master_bias.astype(np.float64)

    # Sigma clipping（Gemini G-2 建議）
    mask = np.ones(flat.shape, dtype=bool)
    for _ in range(sigma_clip_iters):
        median = np.nanmedian(flat[mask])
        std = np.nanstd(flat[mask])
        if std == 0:
            break
        new_mask = np.abs(flat - median) <= sigma_clip_sigma * std
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask

    norm_val = float(np.nanmedian(flat[mask]))
    if norm_val <= 0:
        raise ValueError("Master Flat 歸一化中位數 ≤ 0，平場疑似損毀。")

    flat_norm = (flat / norm_val).astype(np.float32)

    # 壞像素保護：防止除以接近 0 的邊角死點
    flat_norm = np.clip(flat_norm, bad_pixel_threshold, 5.0)
    return flat_norm


# =============================================================================
# 校正狀態標記
# =============================================================================

def determine_cal_mode(
    has_dark: bool,
    has_bias: bool,
    has_flat: bool,
) -> str:
    """
    依據可用的校正幀組合，回傳 CAL_MODE 字串。

    CAL_MODE 值定義：
        FULL       : Dark + Flat（標準完整校正）
        NO_DARK    : Bias + Flat（無暗場，用偏壓替代）
        FLAT_ONLY  : 只有 Flat（僅平場校正）
        NO_FLAT    : Dark 或 Bias，但無 Flat
        RAW        : 無任何校正幀（直接輸出）
    """
    if has_dark and has_flat:
        return "FULL"
    elif has_bias and has_flat and not has_dark:
        return "NO_DARK"
    elif has_flat and not has_dark and not has_bias:
        return "FLAT_ONLY"
    elif (has_dark or has_bias) and not has_flat:
        return "NO_FLAT"
    else:
        return "RAW"


# =============================================================================
# 主校正管線
# =============================================================================

def run_calibration(config_path: str | Path) -> None:
    """
    主校正管線入口。

    讀取 observation_config.yaml，對每個 obs_session 執行：
        1. 路徑解析
        2. Master Bias / Dark / Flat 合成
        3. 逐幀校正
        4. 輸出 float32 Bayer FITS

    Parameters
    ----------
    config_path : observation_config.yaml 的路徑。
    """
    cfg = load_config(config_path)

    cal_cfg = cfg.get("calibration", {})
    chunk_size = int(cal_cfg.get("median_chunk_size", 10))
    bad_pixel_thr = float(cal_cfg.get("flat_bad_pixel_threshold", 0.3))

    sessions = cfg.get("obs_sessions", [])
    if not sessions:
        raise ValueError("observation_config.yaml 裡沒有 obs_sessions。")

    print("\n" + "=" * 60)
    print("  變星測光管線 — 影像校正模組  Calibration.py")
    print("=" * 60)

    for session in sessions:
        # targets / target 雙格式相容
        raw_targets = session.get("targets", session.get("target", "UNKNOWN"))
        targets_list: List[str] = (
            [str(raw_targets)]
            if isinstance(raw_targets, str)
            else [str(t) for t in raw_targets]
        )
        date = str(session["date"])
        telescope_id = session.get("telescope", "")
        camera_id = session.get("camera", "")
        iso = int(session.get("iso", 0))

        # ── 取得儀器參數（session 共用，在 target 迴圈外取一次）────────────
        cam_cfg = cfg.get("cameras", {}).get(camera_id, {})
        tz_offset = int(
            cfg.get("calibration", {}).get("tz_offset_hours", 8)
        )

        # ── Master 幀：同一 session 所有 target 共用，只合成一次 ────────────
        # 路徑解析用 targets_list[0] 代表該 session（calibration 幀共用）
        _ref_session = dict(session)
        _ref_session["target"] = targets_list[0]
        paths_ref = resolve_session_paths(cfg, _ref_session)

        # ── ISO fallback：yaml → 第一張 light 檔 EXIF/標頭 → 0（[WARN]）──
        # yaml 填 iso > 0：直接使用，最優先
        # yaml 未填或填 0：從第一張 light 檔讀取
        #   CR2  → exifread EXIF ISOSpeedRatings
        #   FITS → astropy FITS 標頭 ISOSPEED
        # 兩者都失敗：iso = 0，發出 [WARN]
        if iso == 0:
            _light_dir = paths_ref["light_dir"]
            _first_light: Optional[Path] = None
            if _light_dir.exists():
                for _ext in ("*.CR2", "*.cr2", "*.fits", "*.FITS", "*.fit", "*.FIT"):
                    _found = next(_light_dir.glob(_ext), None)
                    if _found:
                        _first_light = _found
                        break

            if _first_light is None:
                print(
                    f"  [WARN] yaml 未填 iso，且找不到 light 幀"
                    f"（搜尋路徑：{_light_dir}），"
                    "GAIN/RDNOISE 標頭將空白。"
                )
            elif _first_light.suffix.lower() == ".cr2":
                try:
                    with open(_first_light, "rb") as _fh:
                        _tags = exifread.process_file(_fh, details=False)
                    _iso_tag = _tags.get("EXIF ISOSpeedRatings")
                    if _iso_tag:
                        iso = int(str(_iso_tag))
                        print(
                            f"  [INFO] yaml 未填 iso，"
                            f"從 CR2 EXIF 讀取：ISO {iso}"
                        )
                    else:
                        print(
                            "  [WARN] yaml 未填 iso，"
                            "CR2 EXIF 無 ISOSpeedRatings，"
                            "GAIN/RDNOISE 標頭將空白。"
                        )
                except Exception as _exc:
                    print(
                        f"  [WARN] yaml 未填 iso，"
                        f"CR2 EXIF 讀取失敗（{_exc}），"
                        "GAIN/RDNOISE 標頭將空白。"
                    )
            else:
                # FITS：讀 ISOSPEED 標頭
                try:
                    _hdr = fits.getheader(_first_light)
                    _iso_val = _hdr.get("ISOSPEED", 0)
                    if _iso_val and int(_iso_val) > 0:
                        iso = int(_iso_val)
                        print(
                            f"  [INFO] yaml 未填 iso，"
                            f"從 FITS 標頭 ISOSPEED 讀取：ISO {iso}"
                        )
                    else:
                        print(
                            "  [WARN] yaml 未填 iso，"
                            "FITS 標頭無 ISOSPEED，"
                            "GAIN/RDNOISE 標頭將空白。"
                        )
                except Exception as _exc:
                    print(
                        f"  [WARN] yaml 未填 iso，"
                        f"FITS 標頭讀取失敗（{_exc}），"
                        "GAIN/RDNOISE 標頭將空白。"
                    )

        # ── sensor_db 查詢（iso 確定後）──────────────────────────────────────
        sensor_db = cam_cfg.get("sensor_db", {})
        iso_entry = sensor_db.get(iso, {})
        # sensor_db 值為 [gain, read_noise] 列表或 dict 兩種格式均相容
        if isinstance(iso_entry, (list, tuple)):
            gain_e: Optional[float] = float(iso_entry[0]) if len(iso_entry) > 0 else None
            rn_e: Optional[float] = float(iso_entry[1]) if len(iso_entry) > 1 else None
        else:
            gain_e = iso_entry.get("gain")
            rn_e = iso_entry.get("read_noise")
        sat_adu = cam_cfg.get("saturation_adu")

        if gain_e is None:
            print(f"  [WARN] sensor_db 缺少 ISO {iso} 的 gain，FITS GAIN 標頭將空白。")
        if rn_e is None:
            print(f"  [WARN] sensor_db 缺少 ISO {iso} 的 read_noise，FITS RDNOISE 標頭將空白。")

        dark_temp_c: Optional[float] = paths_ref.get("dark_temp_c")
        light_temp_c_val: Optional[float] = paths_ref.get("light_temp_c")

        dark_tmp_str = (
            f"{dark_temp_c:.1f} °C" if dark_temp_c is not None else "未知"
        )
        light_tmp_str = (
            f"{light_temp_c_val:.1f} °C"
            if light_temp_c_val is not None
            else "未知"
        )
        print(
            f"\n[Session] 日期：{date}  "
            f"儀器：{telescope_id} + {camera_id}  ISO：{iso}"
        )
        print(f"  暗場溫度：{dark_tmp_str}  / 觀測溫度：{light_tmp_str}")

        if dark_temp_c is not None and light_temp_c_val is not None:
            temp_diff = abs(light_temp_c_val - dark_temp_c)
            if temp_diff > 10.0:
                print(
                    f"  [WARN] 暗場與觀測溫差 {temp_diff:.1f} °C > 10 °C，"
                    f"暗電流縮放殘差可能顯著。建議補拍接近觀測溫度的暗場。"
                )

        dark_files = _list_image_files(paths_ref["dark_dir"])
        bias_files = _list_image_files(paths_ref["bias_dir"])

        # ── Flat：依格式分別建 Master ──────────────────────────────────────
        flat_dirs_by_fmt = paths_ref.get("flat_dirs_by_format", {})
        # 舊的 flat_dir 做 fallback（向後相容）
        if not flat_dirs_by_fmt and paths_ref.get("flat_dir"):
            flat_files_old = _list_image_files(paths_ref["flat_dir"])
            if flat_files_old:
                # 偵測格式
                _ext0 = flat_files_old[0].suffix.lower()
                _key = "cr2" if _ext0 == ".cr2" else "fits"
                flat_dirs_by_fmt = {_key: paths_ref["flat_dir"]}

        # 為每種格式收集 flat 檔案
        flat_files_by_fmt: Dict[str, List[Path]] = {}
        for fmt, fdir in flat_dirs_by_fmt.items():
            flist = _list_image_files(fdir)
            if flist:
                flat_files_by_fmt[fmt] = flist
                print(f"  Flat ({fmt.upper()}) : {len(flist)} 幀  →  {fdir}")

        has_any_flat = len(flat_files_by_fmt) > 0

        cal_mode = determine_cal_mode(
            has_dark=len(dark_files) > 0,
            has_bias=len(bias_files) > 0,
            has_flat=has_any_flat,
        )
        print(f"  校正模式（CAL_MODE）：{cal_mode}")

        # 合成 Master 幀
        paths_ref["masters_dir"].mkdir(parents=True, exist_ok=True)
        save_masters = cfg.get("calibration", {}).get("save_masters", True)

        master_bias: Optional[np.ndarray] = None
        if bias_files:
            master_bias = create_master(
                bias_files, "Master Bias", chunk_size, tz_offset
            )
            if save_masters:
                bias_out = paths_ref["masters_dir"] / f"master_bias_{date}.fits"
                fits.writeto(bias_out, master_bias, overwrite=True)
                print(f"  [儲存] Master Bias → {bias_out.name}")

        master_dark: Optional[np.ndarray] = None
        if dark_files:
            dark_raw = create_master(
                dark_files, "Master Dark", chunk_size, tz_offset
            )
            master_dark = (
                dark_raw - master_bias
                if master_bias is not None
                else dark_raw
            )
            if save_masters:
                dark_out = paths_ref["masters_dir"] / f"master_dark_{date}.fits"
                fits.writeto(dark_out, master_dark, overwrite=True)
                print(f"  [儲存] Master Dark → {dark_out.name}")

        # 為每種格式建 Master Flat
        master_flat_by_fmt: Dict[str, np.ndarray] = {}
        for fmt, flist in flat_files_by_fmt.items():
            flat_raw = create_master(
                flist, f"Master Flat ({fmt.upper()})", chunk_size, tz_offset
            )
            mf = normalize_flat(flat_raw, master_bias, bad_pixel_thr)
            master_flat_by_fmt[fmt] = mf
            del flat_raw
            gc.collect()
            if save_masters:
                _suffix = "" if fmt == "cr2" else f"_{fmt}"
                flat_out = (
                    paths_ref["masters_dir"]
                    / f"master_flat_norm_{date}{_suffix}.fits"
                )
                fits.writeto(flat_out, mf, overwrite=True)
                print(f"  [儲存] Master Flat ({fmt.upper()}) → {flat_out.name}")

        # ── 逐 target 校正 ───────────────────────────────────────────────────
        for target in targets_list:
            print(f"\n[Target] {target}")
            _t_session = dict(session)
            _t_session["target"] = target
            paths = resolve_session_paths(cfg, _t_session)

            light_files = _list_image_files(paths["light_dir"])
            if not light_files:
                print(f"  [SKIP] 找不到 Light 幀：{paths['light_dir']}")
                continue

            # 統計 light 的格式分佈
            _light_cr2 = [f for f in light_files if f.suffix.lower() == ".cr2"]
            _light_fits = [f for f in light_files
                           if f.suffix.lower() in (".fits", ".fit")]
            print(f"  Light  : {len(light_files)} 幀  →  {paths['light_dir']}")
            if _light_cr2 and _light_fits:
                print(f"           [WARN] 混合格式："
                      f"{len(_light_fits)} FITS + {len(_light_cr2)} CR2")
            print(f"  Dark   : {len(dark_files)} 幀  →  {paths_ref['dark_dir']}")
            print(f"  Bias   : {len(bias_files)} 幀  →  {paths_ref['bias_dir']}")

            paths["calibrated_dir"].mkdir(parents=True, exist_ok=True)
            out_dir = paths["calibrated_dir"]

            print(f"\n[校正] 開始逐幀校正 {len(light_files)} 張 Light 幀…")
            success = 0
            failed = 0

            for idx, light_path in enumerate(
                tqdm(light_files, desc=f"{target} 校正進度")
            ):
                try:
                    light, header = read_raw_image(
                        light_path,
                        tz_offset_hours=tz_offset,
                        gain_e_per_adu=gain_e,
                        read_noise_e=rn_e,
                        saturation_adu=sat_adu,
                        iso=iso if iso > 0 else None,
                    )

                    # ── 選擇格式匹配的 Master Flat ─────────────────────────
                    _l_ext = light_path.suffix.lower()
                    _l_fmt = "cr2" if _l_ext == ".cr2" else "fits"
                    master_flat_norm: Optional[np.ndarray] = None
                    _flat_match = "none"

                    if _l_fmt in master_flat_by_fmt:
                        # 格式完全匹配
                        master_flat_norm = master_flat_by_fmt[_l_fmt]
                        _flat_match = "match"
                    elif master_flat_by_fmt:
                        # 格式不匹配，用可用的（並警告）
                        _alt_fmt = next(iter(master_flat_by_fmt))
                        master_flat_norm = master_flat_by_fmt[_alt_fmt]
                        _flat_match = "mismatch"
                        if idx == 0:
                            print(
                                f"\n  [WARN] [格式不匹配] Light 為 "
                                f"{_l_fmt.upper()}，"
                                f"但 Flat 為 {_alt_fmt.upper()}。\n"
                                f"     暗角校正可能不準確（不同讀出路徑的"
                                f"漸暈模式不同）。\n"
                                f"     建議：補拍與 Light 相同格式的 Flat。"
                            )

                    # ── 天文校正公式 ────────────────────────────────────────
                    cal = light.copy()

                    if master_dark is not None:
                        cal -= master_dark
                    elif master_bias is not None:
                        cal -= master_bias

                    if master_flat_norm is not None:
                        cal /= master_flat_norm

                    cal = cal.astype(np.float32)

                    # ── 更新標頭 ────────────────────────────────────────────
                    header["BITPIX"] = -32
                    header["CAL_MODE"] = (cal_mode, "Calibration frames used")

                    header["DARKTMP"] = (
                        float(dark_temp_c)
                        if dark_temp_c is not None
                        else "UNKNOWN",
                        "[degC] Dark frame sensor temperature",
                    )
                    header["LIGHTTMP"] = (
                        float(light_temp_c_val)
                        if light_temp_c_val is not None
                        else "UNKNOWN",
                        "[degC] Light frame sensor temperature",
                    )
                    if dark_temp_c is not None and light_temp_c_val is not None:
                        header["DTMPDIFF"] = (
                            round(light_temp_c_val - dark_temp_c, 2),
                            "[degC] Light minus Dark temperature",
                        )
                    else:
                        header["DTMPDIFF"] = (
                            "N/A", "[degC] Light minus Dark temperature"
                        )

                    header["TELESCOP"] = (
                        cfg.get("telescopes", {})
                        .get(telescope_id, {})
                        .get("name", telescope_id),
                        "Telescope",
                    )
                    header["FOCALLEN"] = (
                        cfg.get("telescopes", {})
                        .get(telescope_id, {})
                        .get("focal_length_mm", ""),
                        "[mm] Focal length",
                    )
                    header["FLATFMT"] = (
                        _flat_match,
                        "Flat format: match/mismatch/none",
                    )
                    header["HISTORY"] = (
                        "Calibrated by Calibration.py (variable star pipeline)"
                    )
                    header["HISTORY"] = f"CAL_MODE={cal_mode}  ISO={iso}"
                    if _flat_match == "mismatch":
                        header["HISTORY"] = (
                            f"WARNING: Flat format mismatch "
                            f"(Light={_l_fmt}, Flat={_alt_fmt})"
                        )

                    out_name = f"Cal_{light_path.stem}_{idx + 1:04d}.fits"
                    out_path = out_dir / out_name

                    hdu = fits.PrimaryHDU(data=cal, header=header)
                    hdu.verify("silentfix")
                    hdu.writeto(out_path, overwrite=True)
                    success += 1

                except Exception as exc:
                    print(f"\n  [失敗] {light_path.name}：{exc}")
                    failed += 1

            print(f"\n[完成] {target} / {date}：成功 {success} 幀，失敗 {failed} 幀")
            print(f"       輸出目錄：{out_dir}")

        # 釋放 Master 幀記憶體供下一個 session 使用
        del master_bias, master_dark, master_flat_by_fmt
        gc.collect()

    print("\n" + "=" * 60)
    print("所有 Session 校正完成。輸出為保留 Bayer 排列的 float32 線性 FITS。")
    print("下一步：plate_solve.py → DeBayer_RGGB.py → Photometry.ipynb")
    print("=" * 60)


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # 預設尋找同目錄的 observation_config.yaml
        default_cfg = Path(__file__).parent.parent / "observation_config.yaml"
        cfg_path = default_cfg
    else:
        cfg_path = Path(sys.argv[1])

    run_calibration(cfg_path)
