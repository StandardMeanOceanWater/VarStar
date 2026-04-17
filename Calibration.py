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
import math
import os
import re
import shutil
import tempfile
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
Array = np.ndarray


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
    if p.exists():
        return p

    fallback = config_path.parent.parent.resolve()
    if (fallback / "data").exists():
        return fallback
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
    # ── 三分立校正路徑 ────────────────────────────────────────────────────────
    # bias/dark 依相機+ISO；flat 依望遠鏡+相機+拍攝日期
    _camera_label = str(session.get("camera_label", "6D2_ISO3200"))
    _scope_label  = str(session.get("scope_label",  "R200SS_6D2"))
    _flat_date    = str(session.get("flat_date", date))

    cal_root  = data_root / "share" / "calibration"
    bias_root = cal_root / "bias" / _camera_label
    dark_root = cal_root / "dark" / _camera_label
    flat_root = cal_root / "flat" / _scope_label / _flat_date

    light_temp_c: Optional[float] = session.get("light_temp_c")
    if light_temp_c is not None:
        light_temp_c = float(light_temp_c)

    def _find_bias_dir() -> Optional[Path]:
        """bias 目錄：bias/{camera_label}/"""
        if bias_root.exists() and _has_images(bias_root):
            return bias_root
        return None

    def _find_flat_dirs_by_format() -> Dict[str, Path]:
        """
        flat raw 目錄（僅供 master 不存在時重建用）。
        flat/{scope_label}/{flat_date}/ → 偵測格式回傳。
        主要路徑已改為直接從 master 目錄選取，此函式為 fallback。
        """
        result: Dict[str, Path] = {}
        if not flat_root.exists():
            return result
        files = [f for f in flat_root.iterdir()
                 if f.is_file() and not f.name.lower().startswith("master_")]
        if any(f.suffix.lower() == ".cr2" for f in files):
            result["cr2"] = flat_root
        if any(f.suffix.lower() in (".fits", ".fit") for f in files):
            result["fits"] = flat_root
        return result

    def _parse_temp(subdir: Path) -> Optional[float]:
        """嘗試從子目錄名稱解析溫度（°C）；失敗回傳 None。"""
        m = _TEMP_RE.match(subdir.name)
        return float(m.group(1)) if m else None

    def _has_images(directory: Path) -> bool:
        """判斷目錄內是否有支援格式的影像（不遞迴）。"""
        valid_exts = {".fit", ".fits", ".cr2"}
        return any(
            f.is_file()
            and f.suffix.lower() in valid_exts
            and not f.name.lower().startswith("master_")
            for f in directory.iterdir()
        )

    def _find_dark_dir() -> Tuple[Optional[Path], Optional[float]]:
        """
        找 dark 目錄，回傳 (path, resolved_dark_temp_c)。

        resolved_dark_temp_c 優先取 session['dark_temp_c']；
        若未填，則從子目錄名稱解析。
        """
        # 候選位置：dark/{camera_label}/
        candidates = [
            dark_root,
        ]
        explicit_temp: Optional[float] = session.get("dark_temp_c")
        if explicit_temp is not None:
            explicit_temp = float(explicit_temp)

        for _dark_candidate in candidates:
            if not _dark_candidate.exists():
                continue

            # ── 情況 1：根層直接有影像 ────────────────────────────────────
            if _has_images(_dark_candidate):
                return _dark_candidate, explicit_temp

            # ── 情況 2：有子目錄 ──────────────────────────────────────────
            subdirs = sorted(
                [d for d in _dark_candidate.iterdir() if d.is_dir()],
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
                if explicit_temp is not None:
                    # dark_temp_c 明確指定 → 直接選最接近該值的目錄
                    chosen_temp, chosen_dir = min(
                        parsed, key=lambda x: abs(x[0] - explicit_temp)
                    )
                    if abs(chosen_temp - explicit_temp) > 0.5:
                        print(
                            f"  [WARN] dark_temp_c={explicit_temp}°C，"
                            f"最近子目錄為 {chosen_dir.name}（差 "
                            f"{abs(chosen_temp - explicit_temp):.1f}°C）"
                        )
                elif light_temp_c is not None:
                    # 自動模式：選最接近觀測氣溫的 dark
                    chosen_temp, chosen_dir = min(
                        parsed, key=lambda x: abs(x[0] - light_temp_c)
                    )
                    print(
                        f"  [INFO] dark 自動選溫：light_temp={light_temp_c}°C "
                        f"→ {chosen_dir.name}"
                    )
                else:
                    # 無任何溫度資訊：選最低溫（dark current 最小）
                    chosen_temp, chosen_dir = min(parsed, key=lambda x: x[0])
                    print(
                        f"  [WARN] session 未填 dark_temp_c / light_temp_c，"
                        f"dark 子目錄自動選最低溫：{chosen_dir.name}"
                    )
                resolved_temp = chosen_temp
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
        "light_dir":           field_root / "raw",
        "bias_dir":            _find_bias_dir(),
        "dark_dir":            dark_dir,
        "flat_dir":            flat_root if flat_root.exists() else None,
        "flat_dirs_by_format": _find_flat_dirs_by_format(),
        "calibrated_dir":      field_root / "wcs",
        # 集中 master 目錄（bias/dark master 日期由 run_calibration 從 EXIF 讀取）
        "masters_dir":  cal_root / "master",
        "flat_date":    _flat_date,
        "dark_temp_c":  resolved_dark_temp,
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

            # ── EXPTIME：先從標頭讀，失敗時從 NINA 檔名解析 ────────────────
            # NINA 已知 bug：某段時間 EXPTIME 寫錯或遺失（含錯誤座標問題）
            # NINA 檔名格式：{date}_{time}__{exptime}_{gain}s_{idx}.fits
            _exptime_val: Optional[float] = None
            for _ekey in ("EXPTIME", "EXPOSURE", "EXP_TIME"):
                try:
                    _v = float(header.get(_ekey, 0) or 0)
                    if _v > 0:
                        _exptime_val = _v
                        break
                except (TypeError, ValueError):
                    pass
            if _exptime_val is None or _exptime_val <= 0:
                import re as _re
                _m = _re.search(r'__(\d+\.?\d+)_', filepath.stem)
                if _m:
                    _exptime_val = float(_m.group(1))
                    print(f"  [WARN] {filepath.name}: EXPTIME 標頭無效，"
                          f"從檔名解析 = {_exptime_val}s")
            if _exptime_val is not None and _exptime_val > 0:
                header["EXPTIME"] = (_exptime_val, "[s] Exposure time")
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
        and not f.name.lower().startswith("master_")   # 排除已合成的 master 幀
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
    dark_rate: Optional[np.ndarray] = None,
    t_flat: Optional[float] = None,
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
    # 公式 4：Flat_pure = Flat − Bias − dark_rate × t_flat
    if dark_rate is not None and t_flat is not None and t_flat > 0:
        flat -= dark_rate.astype(np.float64) * t_flat

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

Array = np.ndarray


def _session_date_to_dir(date_text: str) -> str:
    return f"{date_text[:4]}-{date_text[4:6]}-{date_text[6:8]}"


def _first_target_name(session: dict) -> str:
    raw_targets = session.get("targets", session.get("target", "UNKNOWN"))
    if isinstance(raw_targets, list):
        return str(raw_targets[0])
    return str(raw_targets)


def _target_group(cfg: dict, target: str) -> str:
    return str(cfg.get("targets", {}).get(target, {}).get("group", target))


def _legacy_output_dir(cfg: dict, session: dict, target: str) -> Path:
    date_text = str(session["date"])
    group = _target_group(cfg, target)
    return cfg["_project_root"] / "output" / _session_date_to_dir(date_text) / group / "calibrate"


def _frame_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".cr2":
        return "cr2"
    if suffix in (".fit", ".fits"):
        return "fits"
    raise ValueError(f"unsupported frame type: {path}")


def _extract_fractional_exposure(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        pass
    text = str(value).strip()
    if "/" in text:
        left, right = text.split("/", 1)
        try:
            numerator = float(left)
            denominator = float(right)
        except ValueError:
            return None
        if denominator == 0:
            return None
        return numerator / denominator
    try:
        return float(text)
    except ValueError:
        return None


def _read_cr2_tags(path: Path) -> Dict[str, object]:
    with path.open("rb") as handle:
        return exifread.process_file(handle, details=False)


def _metadata_override_for_fits(session_date: str, path: Path) -> Dict[str, object]:
    if session_date == "20251122" and _frame_type(path) == "fits":
        return {
            "iso": 100,
            "iso_source": "override:nina_known_bad_header",
            "exposure_s": 30.0,
            "exposure_source": "override:nina_known_bad_header",
        }
    return {}


def read_frame_metadata(path: Path, session_date: str) -> Dict[str, object]:
    path = Path(path)
    frame_type = _frame_type(path)
    meta: Dict[str, object] = {
        "path": path,
        "frame_type": frame_type,
        "iso": None,
        "iso_source": None,
        "exposure_s": None,
        "exposure_source": None,
        "black_level": None,
        "black_level_source": None,
    }
    if frame_type == "cr2":
        tags = _read_cr2_tags(path)
        iso_tag = tags.get("EXIF ISOSpeedRatings") or tags.get("Image ISOSpeedRatings")
        exp_tag = tags.get("EXIF ExposureTime") or tags.get("Image ExposureTime")
        if iso_tag is not None:
            try:
                meta["iso"] = int(str(iso_tag).strip())
                meta["iso_source"] = "exif"
            except ValueError:
                pass
        exposure_s = _extract_fractional_exposure(exp_tag)
        if exposure_s is not None:
            meta["exposure_s"] = exposure_s
            meta["exposure_source"] = "exif"
    else:
        header = fits.getheader(path)
        for key in ("ISO", "ISOSPEED", "ISOSPEEDR"):
            value = header.get(key)
            if value not in (None, ""):
                try:
                    meta["iso"] = int(float(value))
                    meta["iso_source"] = f"header:{key}"
                    break
                except (TypeError, ValueError):
                    pass
        for key in ("EXPTIME", "EXPOSURE", "EXP_TIME"):
            exposure_s = _extract_fractional_exposure(header.get(key))
            if exposure_s is not None and exposure_s > 0:
                meta["exposure_s"] = exposure_s
                meta["exposure_source"] = f"header:{key}"
                break
        for key in ("BLACKLVL", "BLACKLEVEL"):
            value = header.get(key)
            if value not in (None, ""):
                try:
                    meta["black_level"] = float(value)
                    meta["black_level_source"] = f"header:{key}"
                    break
                except (TypeError, ValueError):
                    pass
    meta.update({k: v for k, v in _metadata_override_for_fits(session_date, path).items() if v is not None})
    return meta


def _infer_calibration_iso(paths: Sequence[Path], session_date: str) -> Optional[int]:
    values: List[int] = []
    for path in paths:
        meta = read_frame_metadata(path, session_date)
        if meta.get("iso") is not None:
            values.append(int(meta["iso"]))
    if not values:
        return None
    counts: Dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _infer_common_exposure(paths: Sequence[Path], session_date: str) -> Optional[float]:
    values: List[float] = []
    for path in paths:
        meta = read_frame_metadata(path, session_date)
        exposure_s = meta.get("exposure_s")
        if exposure_s is not None:
            values.append(float(exposure_s))
    if not values:
        return None
    return float(values[0])


def _build_master_header(kind: str, session_date: str, frame_type: str, iso_value: Optional[int], exposure_s: Optional[float], extra: Optional[Dict[str, object]] = None) -> fits.Header:
    header = fits.Header()
    header["MASTERK"] = (kind, "master kind")
    header["FRMTYPE"] = (frame_type, "source frame type")
    header["SESSDATE"] = (session_date, "session date")
    if iso_value is not None:
        header["ISOSPEED"] = (int(iso_value), "source iso")
    if exposure_s is not None:
        header["EXPTIME"] = (float(exposure_s), "[s] source exposure")
    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            header[key] = value
    return header


def _master_root(cfg: dict) -> Path:
    return cfg["_data_root"] / "share" / "master"


def _ensure_master_root(cfg: dict) -> Path:
    root = _master_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _format_temp_tag(temp_c: Optional[float]) -> str:
    if temp_c is None:
        return "unk"
    return f"{temp_c:.1f}c".replace("+", "")


def _master_bias_paths(master_root: Path, session_date: str, frame_type: str, iso_value: Optional[int]) -> List[Path]:
    iso_tag = f"_iso{iso_value}" if iso_value is not None else ""
    return [
        master_root / f"master_bias_raw_{session_date}_{frame_type}{iso_tag}.fits",
        master_root / f"master_bias_{session_date}_{frame_type}{iso_tag}.fits",
        master_root / f"master_bias_{session_date}_{frame_type}.fits",
        master_root / f"master_bias_{session_date}_cr2.fits",
    ]


def _master_dark_paths(master_root: Path, session_date: str, frame_type: str, iso_value: Optional[int], dark_temp_c: Optional[float]) -> List[Path]:
    iso_tag = f"_iso{iso_value}" if iso_value is not None else ""
    temp_tag = f"_{_format_temp_tag(dark_temp_c)}"
    return [
        master_root / f"master_dark_raw_{session_date}_{frame_type}{iso_tag}{temp_tag}.fits",
        master_root / f"master_dark_{session_date}_{frame_type}{iso_tag}{temp_tag}.fits",
        master_root / f"master_dark_{session_date}{temp_tag}_{frame_type}.fits",
        master_root / f"master_dark_{session_date}{temp_tag}_cr2.fits",
    ]


def _dark_rate_paths(master_root: Path, session_date: str, frame_type: str, iso_value: Optional[int], dark_temp_c: Optional[float]) -> List[Path]:
    iso_tag = f"_iso{iso_value}" if iso_value is not None else ""
    temp_tag = f"_{_format_temp_tag(dark_temp_c)}"
    return [
        master_root / f"dark_rate_{session_date}_{frame_type}{iso_tag}{temp_tag}.fits",
    ]


def _master_flat_product_paths(master_root: Path, flat_date: str, frame_type: str, use_median_normalization: bool) -> List[Path]:
    if use_median_normalization:
        return [
            master_root / f"master_flatnorm_{flat_date}_{frame_type}.fits",
            master_root / f"master_flat_{flat_date}_{frame_type}.fits",
        ]
    return [
        master_root / f"master_flatdirect_{flat_date}_{frame_type}.fits",
    ]


def _load_first_existing(paths: Sequence[Path]) -> Optional[Tuple[Array, fits.Header, Path]]:
    for path in paths:
        if path.exists():
            with fits.open(path, ignore_missing_end=True) as hdul:
                hdu = next((item for item in hdul if item.data is not None), None)
                if hdu is None:
                    continue
                return hdu.data.astype(np.float32), hdu.header.copy(), path
    return None


def _write_master(path: Path, data: Array, header: fits.Header) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, data.astype(np.float32), header=header, overwrite=True)


def _warn(message: str) -> None:
    print(f"  [WARN] {message}")


def _info(message: str) -> None:
    print(f"  [INFO] {message}")


def compute_dark_rate(master_dark_signal: Array, t_dark: Optional[float]) -> Optional[Array]:
    if t_dark is None or t_dark <= 0:
        return None
    return (master_dark_signal.astype(np.float32) / float(t_dark)).astype(np.float32)


def compute_flat_pure(master_flat_raw: Array, master_bias: Optional[Array], dark_rate: Optional[Array], t_flat: Optional[float]) -> Array:
    flat_pure = master_flat_raw.astype(np.float64).copy()
    if master_bias is not None:
        flat_pure -= master_bias.astype(np.float64)
    if dark_rate is not None and t_flat is not None and t_flat > 0:
        flat_pure -= dark_rate.astype(np.float64) * float(t_flat)
    return flat_pure.astype(np.float32)


def normalize_flat_pure(flat_pure: Array, bad_pixel_floor: float, use_median_normalization: bool = True) -> Array:
    if use_median_normalization:
        median_value = float(np.nanmedian(flat_pure))
        if not math.isfinite(median_value) or median_value == 0.0:
            raise ValueError("flat median is zero or non-finite")
        master_flatnorm = (flat_pure.astype(np.float64) / median_value).astype(np.float32)
        master_flatnorm = np.where(np.isfinite(master_flatnorm), master_flatnorm, bad_pixel_floor)
        master_flatnorm = np.clip(master_flatnorm, bad_pixel_floor, None)
        return master_flatnorm.astype(np.float32)

    fallback = float(np.nanmedian(flat_pure))
    if not math.isfinite(fallback) or fallback <= 0.0:
        fallback = 1.0
    flat_direct = np.where(np.isfinite(flat_pure), flat_pure, fallback).astype(np.float32)
    flat_direct = np.where(flat_direct > 0.0, flat_direct, fallback).astype(np.float32)
    return flat_direct


def compute_dark_scaled(dark_rate: Optional[Array], t_light: Optional[float]) -> Optional[Array]:
    if dark_rate is None or t_light is None or t_light <= 0:
        return None
    return (dark_rate.astype(np.float32) * float(t_light)).astype(np.float32)


def calibrate_image(light_linear: Array, master_bias: Optional[Array], dark_scaled: Optional[Array], master_flatnorm: Array) -> Array:
    calibrated = light_linear.astype(np.float64).copy()
    if master_bias is not None:
        calibrated -= master_bias.astype(np.float64)
    if dark_scaled is not None:
        calibrated -= dark_scaled.astype(np.float64)
    calibrated /= master_flatnorm.astype(np.float64)
    return calibrated.astype(np.float32)


def _select_matching_flat(light_type: str, flat_map: Dict[str, Array]) -> Array:
    if light_type not in flat_map:
        available = ", ".join(sorted(flat_map))
        raise ValueError(f"flat type mismatch: light={light_type}, available={available}")
    return flat_map[light_type]


def _sensor_db_for_session(cfg: dict, session: dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    camera_id = session.get("camera", "")
    camera_cfg = cfg.get("cameras", {}).get(camera_id, {})
    session_iso = session.get("iso")
    gain_e = None
    read_noise_e = None
    if session_iso is not None:
        sensor_entry = camera_cfg.get("sensor_db", {}).get(int(session_iso))
        if isinstance(sensor_entry, (list, tuple)) and len(sensor_entry) >= 2:
            gain_e = float(sensor_entry[0])
            read_noise_e = float(sensor_entry[1])
    saturation_adu = camera_cfg.get("saturation_adu")
    if saturation_adu is not None:
        saturation_adu = float(saturation_adu)
    return gain_e, read_noise_e, saturation_adu


def _light_iso_for_read(session: dict, light_path: Path) -> Optional[int]:
    date_text = str(session["date"])
    if date_text == "20251122" and _frame_type(light_path) == "fits":
        return 100
    session_iso = session.get("iso")
    if session_iso is None:
        return None
    return int(session_iso)


def _read_frame_data(path: Path, session: dict, tz_offset: int, gain_e: Optional[float], read_noise_e: Optional[float], saturation_adu: Optional[float]) -> Tuple[Array, fits.Header]:
    return read_raw_image(
        path,
        tz_offset_hours=tz_offset,
        gain_e_per_adu=gain_e,
        read_noise_e=read_noise_e,
        saturation_adu=saturation_adu,
        iso=_light_iso_for_read(session, path),
    )


def _build_master_bias(cfg: dict, session: dict, paths_ref: dict, tz_offset: int) -> Tuple[Optional[Array], Optional[fits.Header], Optional[int]]:
    bias_files = _list_image_files(paths_ref["bias_dir"])
    if not bias_files:
        return None, None, None
    master_root = _ensure_master_root(cfg)
    frame_type = _frame_type(bias_files[0])
    iso_value = _infer_calibration_iso(bias_files, str(session["date"]))
    existing = _load_first_existing(_master_bias_paths(master_root, str(session["date"]), frame_type, iso_value))
    if existing is not None:
        data, header, path = existing
        _info(f"loaded master bias: {path.name}")
        return data.astype(np.float32), header, int(header["ISOSPEED"]) if "ISOSPEED" in header else iso_value
    master_bias = create_master(bias_files, "Master Bias Raw", tz_offset_hours=tz_offset).astype(np.float32)
    header = _build_master_header("bias_raw", str(session["date"]), frame_type, iso_value, None)
    out_path = _master_bias_paths(master_root, str(session["date"]), frame_type, iso_value)[0]
    _write_master(out_path, master_bias, header)
    _info(f"saved master bias: {out_path.name}")
    return master_bias, header, iso_value


def _build_master_dark_signal(cfg: dict, session: dict, paths_ref: dict, tz_offset: int, master_bias: Optional[Array]) -> Tuple[Optional[Array], Optional[fits.Header], Optional[float], Optional[int]]:
    dark_files = _list_image_files(paths_ref["dark_dir"])
    if not dark_files:
        return None, None, None, None
    master_root = _ensure_master_root(cfg)
    frame_type = _frame_type(dark_files[0])
    iso_value = _infer_calibration_iso(dark_files, str(session["date"]))
    dark_temp_c = paths_ref.get("dark_temp_c")
    existing = _load_first_existing(_master_dark_paths(master_root, str(session["date"]), frame_type, iso_value, dark_temp_c))
    if existing is not None:
        data, header, path = existing
        _info(f"loaded master dark signal: {path.name}")
        exposure_s = _extract_fractional_exposure(header.get("EXPTIME"))
        return data.astype(np.float32), header, exposure_s, int(header["ISOSPEED"]) if "ISOSPEED" in header else iso_value
    dark_raw = create_master(dark_files, "Master Dark Raw", tz_offset_hours=tz_offset).astype(np.float32)
    t_dark = _infer_common_exposure(dark_files, str(session["date"]))
    master_dark_signal = dark_raw.copy()
    if master_bias is not None:
        master_dark_signal -= master_bias.astype(np.float32)
    header = _build_master_header(
        "dark_signal",
        str(session["date"]),
        frame_type,
        iso_value,
        t_dark,
        extra={"DARKTEMP": float(dark_temp_c) if dark_temp_c is not None else "UNKNOWN"},
    )
    out_path = _master_dark_paths(master_root, str(session["date"]), frame_type, iso_value, dark_temp_c)[0]
    _write_master(out_path, master_dark_signal, header)
    _info(f"saved master dark signal: {out_path.name}")
    return master_dark_signal, header, t_dark, iso_value


def _build_dark_rate(cfg: dict, session: dict, dark_signal: Optional[Array], dark_header: Optional[fits.Header], t_dark: Optional[float], dark_iso: Optional[int], dark_temp_c: Optional[float]) -> Optional[Array]:
    if dark_signal is None:
        return None
    master_root = _ensure_master_root(cfg)
    frame_type = "cr2"
    if dark_header is not None and "FRMTYPE" in dark_header:
        frame_type = str(dark_header["FRMTYPE"]).strip().lower()
    existing = _load_first_existing(_dark_rate_paths(master_root, str(session["date"]), frame_type, dark_iso, dark_temp_c))
    if existing is not None:
        data, _, path = existing
        _info(f"loaded dark rate: {path.name}")
        return data.astype(np.float32)
    dark_rate = compute_dark_rate(dark_signal, t_dark)
    if dark_rate is None:
        return None
    header = _build_master_header("dark_rate", str(session["date"]), frame_type, dark_iso, t_dark, extra={"DARKTEMP": float(dark_temp_c) if dark_temp_c is not None else "UNKNOWN"})
    out_path = _dark_rate_paths(master_root, str(session["date"]), frame_type, dark_iso, dark_temp_c)[0]
    _write_master(out_path, dark_rate, header)
    _info(f"saved dark rate: {out_path.name}")
    return dark_rate


def _build_flatnorm_map(cfg: dict, session: dict, paths_ref: dict, tz_offset: int, master_bias: Optional[Array], dark_rate: Optional[Array], bad_pixel_floor: float, use_median_normalization: bool) -> Dict[str, Array]:
    flat_map: Dict[str, Array] = {}
    flat_dirs_by_format = dict(paths_ref.get("flat_dirs_by_format", {}))
    master_root = _ensure_master_root(cfg)
    for frame_type, flat_dir in sorted(flat_dirs_by_format.items()):
        existing = _load_first_existing(_master_flat_product_paths(master_root, str(paths_ref["flat_date"]), frame_type, use_median_normalization))
        if existing is not None:
            data, _, path = existing
            _info(f"loaded master flatnorm: {path.name}")
            flat_map[frame_type] = data.astype(np.float32)
            continue
        flat_files = _list_image_files(flat_dir)
        if not flat_files:
            continue
        master_flat_raw = create_master(flat_files, f"Master Flat Raw ({frame_type.upper()})", tz_offset_hours=tz_offset).astype(np.float32)
        t_flat = _infer_common_exposure(flat_files, str(session["date"]))
        flat_pure = compute_flat_pure(master_flat_raw, master_bias, dark_rate, t_flat)
        master_flatnorm = normalize_flat_pure(flat_pure, bad_pixel_floor, use_median_normalization=use_median_normalization)
        header = _build_master_header(
            "flatnorm" if use_median_normalization else "flatdirect",
            str(paths_ref["flat_date"]),
            frame_type,
            None,
            t_flat,
        )
        out_path = _master_flat_product_paths(master_root, str(paths_ref["flat_date"]), frame_type, use_median_normalization)[0]
        _write_master(out_path, master_flatnorm, header)
        _info(f"saved master flatnorm: {out_path.name}")
        flat_map[frame_type] = master_flatnorm.astype(np.float32)
    return flat_map


def _check_iso_warning(light_meta: Dict[str, object], bias_iso: Optional[int], dark_iso: Optional[int]) -> List[str]:
    warnings: List[str] = []
    light_iso = light_meta.get("iso")
    if light_iso is None:
        warnings.append("light iso unknown")
        return warnings
    if bias_iso is not None and int(light_iso) != int(bias_iso):
        warnings.append(f"bias iso mismatch: light={light_iso}, bias={bias_iso}")
    if dark_iso is not None and int(light_iso) != int(dark_iso):
        warnings.append(f"dark iso mismatch: light={light_iso}, dark={dark_iso}")
    return warnings


def _apply_light_overrides(header: fits.Header, light_meta: Dict[str, object]) -> None:
    if light_meta.get("iso") is not None:
        header["ISOSPEED"] = (int(light_meta["iso"]), "light iso")
    exposure_s = light_meta.get("exposure_s")
    if exposure_s is not None:
        header["EXPTIME"] = (float(exposure_s), "[s] light exposure")


def _prepare_reference_paths(cfg: dict, session: dict) -> dict:
    ref_session = dict(session)
    ref_session["target"] = _first_target_name(session)
    return resolve_session_paths(cfg, ref_session)


def _validate_shapes(light_shape: Tuple[int, int], master_bias: Optional[Array], dark_rate: Optional[Array], master_flatnorm: Array) -> None:
    if master_bias is not None and master_bias.shape != light_shape:
        raise ValueError(f"bias shape mismatch: light={light_shape}, bias={master_bias.shape}")
    if dark_rate is not None and dark_rate.shape != light_shape:
        raise ValueError(f"dark shape mismatch: light={light_shape}, dark_rate={dark_rate.shape}")
    if master_flatnorm.shape != light_shape:
        raise ValueError(f"flat shape mismatch: light={light_shape}, flat={master_flatnorm.shape}")


def _calibrate_target(
    cfg: dict,
    session: dict,
    target: str,
    no_flat: bool,
    master_bias: Optional[Array],
    bias_iso: Optional[int],
    dark_rate: Optional[Array],
    dark_iso: Optional[int],
    flat_map: Dict[str, Array],
    output_subdir: str,
    use_bias: bool,
    use_dark: bool,
    use_median_normalization: bool,
) -> None:
    paths = resolve_session_paths(cfg, {**session, "target": target})
    output_dir = _legacy_output_dir(cfg, session, target).parent / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    light_files = _list_image_files(paths["light_dir"])
    if not light_files:
        _warn(f"no light frames: {paths['light_dir']}")
        return
    tz_offset = int(cfg.get("calibration", {}).get("tz_offset_hours", 8))
    gain_e, read_noise_e, saturation_adu = _sensor_db_for_session(cfg, session)
    success = 0
    failed = 0
    print(f"\n[Target] {target}")
    print(f"  [INFO] output dir: {output_dir}")
    for index, light_path in enumerate(tqdm(light_files, desc=f"{target} calibrate")):
        try:
            light_data, header = _read_frame_data(light_path, session, tz_offset, gain_e, read_noise_e, saturation_adu)
            light_meta = read_frame_metadata(light_path, str(session["date"]))
            _apply_light_overrides(header, light_meta)
            light_type = _frame_type(light_path)
            if no_flat:
                raise ValueError("no_flat mode is not supported by calibration_rewrite")
            master_flatnorm = _select_matching_flat(light_type, flat_map)
            _validate_shapes(light_data.shape, master_bias, dark_rate, master_flatnorm)
            effective_bias = master_bias if use_bias else None
            effective_dark_rate = dark_rate if use_dark else None
            dark_scaled = compute_dark_scaled(effective_dark_rate, light_meta.get("exposure_s"))
            calibrated = calibrate_image(light_data, effective_bias, dark_scaled, master_flatnorm)
            warnings = _check_iso_warning(light_meta, bias_iso if use_bias else None, dark_iso if use_dark else None)
            header["BITPIX"] = -32
            header["CALVER"] = ("rewrite_v1", "calibration rewrite version")
            header["CALFMT"] = (light_type, "light frame type")
            header["BLPOLICY"] = ("fits_zero" if light_type == "fits" else "cr2_reader", "black level policy")
            header["USEBIAS"] = (1 if use_bias else 0, "bias subtraction enabled")
            header["USEDARK"] = (1 if use_dark else 0, "dark subtraction enabled")
            header["FLATNORM"] = ("median" if use_median_normalization else "off", "flat median normalization mode")
            if light_type == "fits":
                header["BLASSUME"] = (0.0, "assumed fits black level")
            if dark_scaled is not None:
                header["DARKSCAL"] = (float(light_meta["exposure_s"]), "[s] scaled dark exposure")
            if warnings:
                for warning in warnings:
                    _warn(f"{light_path.name}: {warning}")
                    header["HISTORY"] = f"WARNING: {warning}"
            out_name = f"Cal_{light_path.stem}_{index + 1:04d}.fits"
            out_path = output_dir / out_name
            fits.writeto(out_path, calibrated.astype(np.float32), header=header, overwrite=True)
            success += 1
        except Exception as exc:
            failed += 1
            _warn(f"{light_path.name}: {exc}")
    print(f"  [INFO] calibrated: success={success}, failed={failed}")


def run_calibration(config_path: str | Path, *, no_flat: bool = False) -> None:
    cfg = load_config(config_path)
    sessions = cfg.get("obs_sessions", [])
    if not sessions:
        raise ValueError("no obs_sessions in config")
    cal_cfg = cfg.get("calibration", {})
    bad_pixel_floor = float(cal_cfg.get("flat_bad_pixel_threshold", 0.3))
    tz_offset = int(cal_cfg.get("tz_offset_hours", 8))
    use_bias = bool(cal_cfg.get("use_bias", True))
    use_dark = bool(cal_cfg.get("use_dark", True))
    use_median_normalization = bool(cal_cfg.get("flat_normalize_median", True))
    output_subdir = str(cal_cfg.get("output_subdir", "calibrate"))
    print("=" * 60)
    print("Calibration rewrite start")
    print("=" * 60)
    for session in sessions:
        session_date = str(session["date"])
        print(f"\n[Session] {session_date}")
        paths_ref = _prepare_reference_paths(cfg, session)
        master_bias = None
        bias_iso = None
        if use_bias:
            master_bias, _, bias_iso = _build_master_bias(cfg, session, paths_ref, tz_offset)
        dark_signal = None
        dark_header = None
        t_dark = None
        dark_iso = None
        if use_dark:
            dark_signal, dark_header, t_dark, dark_iso = _build_master_dark_signal(cfg, session, paths_ref, tz_offset, master_bias)
        dark_rate = _build_dark_rate(cfg, session, dark_signal, dark_header, t_dark, dark_iso, paths_ref.get("dark_temp_c")) if use_dark else None
        flat_map = _build_flatnorm_map(
            cfg,
            session,
            paths_ref,
            tz_offset,
            master_bias if use_bias else None,
            dark_rate if use_dark else None,
            bad_pixel_floor,
            use_median_normalization,
        )
        if not no_flat and not flat_map:
            raise ValueError(f"no matching flats available for session {session_date}")
        raw_targets = session.get("targets", session.get("target", "UNKNOWN"))
        targets = [str(raw_targets)] if isinstance(raw_targets, str) else [str(item) for item in raw_targets]
        for target in targets:
            _calibrate_target(
                cfg,
                session,
                target,
                no_flat,
                master_bias,
                bias_iso,
                dark_rate,
                dark_iso,
                flat_map,
                output_subdir,
                use_bias,
                use_dark,
                use_median_normalization,
            )
    print("=" * 60)
    print("Calibration rewrite finished")
    print("=" * 60)

# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # 預設尋找同目錄的 observation_config.yaml
        default_cfg = Path(__file__).parent / "observation_config.yaml"
        cfg_path = default_cfg
    else:
        cfg_path = Path(sys.argv[1])

    run_calibration(cfg_path)
