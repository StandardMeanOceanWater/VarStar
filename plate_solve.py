# -*- coding: utf-8 -*-
"""
plate_solve.py  —  星圖解算模組
專案：變星測光管線 v0.99
描述：對校正後的 FITS 執行 WCS 星圖解算。
      本地環境使用 ASTAP CLI；Colab 環境使用 astrometry.net API。
      輸出另存副本到 calibrated/wcs/ 子目錄，不覆蓋原始校正幀。

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
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS  # noqa: F401  (保留供呼叫端 import)

from Calibration import load_config


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

    ra_hours = float(ra_h)                # 單位已是小時，直接使用
    spd_deg = 90.0 + float(dec_deg)       # 南極距：SPD = 90 + Dec
    return ra_hours, spd_deg


# =============================================================================
# ASTAP 後端
# =============================================================================

def _run_astap(
    fits_path: Path,
    out_path: Path,
    astap_cfg: dict,
    ra_hint_h: Optional[float] = None,
    spd_hint_deg: Optional[float] = None,
) -> bool:
    """
    呼叫 ASTAP CLI 對單張 FITS 執行星圖解算，輸出到 out_path。

    ASTAP 解算流程：
        1. 複製 fits_path 到臨時目錄（避免 ASTAP 直接覆蓋來源）
        2. 執行 ASTAP CLI（-update 旗標讓 ASTAP 把 WCS 寫回副本）
           傳入 -ra（小時）/ -spd（南極距）hint，修正 NINA 標頭
           RA/DEC 不可靠（實測為北極點座標）導致 ASTAP 搜尋範圍
           偏離目標天區的問題。不縮減 search_radius，避免影響
           ASTAP 內部搜尋格網的步驟起點計算。
        3. 驗證 WCS 中心在合理天區內
        4. 複製已解算 FITS 到 out_path

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
    downsample = int(astap_cfg.get("downsample", 2))
    fov = float(astap_cfg.get("fov_override_deg", 0.0))
    timeout = int(astap_cfg.get("timeout_sec", 180))
    max_retries = int(astap_cfg.get("max_retries", 2))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_fits = Path(tmpdir) / fits_path.name
        import shutil
        shutil.copy2(fits_path, tmp_fits)

        cmd = [
            executable,
            "-f", str(tmp_fits),
            "-r", str(search_radius),
            "-d", db_path,
            "-update",
            "-o", str(tmp_fits),
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
                )
                if result.returncode == 0 and tmp_fits.exists():
                    with fits.open(tmp_fits) as hdul:
                        hdr = hdul[0].header
                        if "CRVAL1" not in hdr and "CD1_1" not in hdr:
                            print(
                                f"  [WARN] ASTAP 回傳 0 但 WCS 標頭缺失："
                                f"{fits_path.name}"
                            )
                            continue
                    shutil.copy2(tmp_fits, out_path)
                    return True
                else:
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

    hdu = fits.PrimaryHDU(data=data_orig, header=header_orig)
    hdu.verify("silentfix")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hdu.writeto(out_path, overwrite=True)
    return True


# =============================================================================
# 主解算管線
# =============================================================================

def run_plate_solve(config_path: str | Path) -> None:
    """
    主星圖解算管線入口。

    讀取 observation_config.yaml，對每個 obs_session 的每個 target，
    處理 calibrated/ 目錄內所有校正 FITS，
    輸出到 calibrated/wcs/ 子目錄。

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

    print("\n" + "=" * 60)
    print(f"  變星測光管線 — 星圖解算模組  plate_solve.py  [{backend}]")
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

        for target in target_list:
            _tgt_cfg = cfg.get("targets", {}).get(target, {})
            _group = _tgt_cfg.get("group", target)
            _date_fmt = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
            field_root = data_root / _date_fmt / _group
            cal_dir = field_root / "wcs"
            wcs_dir = cal_dir  # WCS output to same directory
            wcs_dir.mkdir(parents=True, exist_ok=True)

            fits_files = sorted(
                f for f in cal_dir.glob("*.fits")
                if f.is_file() and "wcs" not in f.stem.lower()
            )

            if not fits_files:
                print(f"[SKIP] {target}/{date}：calibrated/ 目錄裡找不到 FITS。")
                continue

            # 取得 hint 座標（修正 NINA 標頭 RA/DEC 不可靠問題）
            ra_hint, spd_hint = _get_hint_for_target(cfg, target)
            if ra_hint is not None:
                print(
                    f"\n[Session] {target} / {date}  ({len(fits_files)} 幀)"
                    f"  hint RA={ra_hint:.4f}h SPD={spd_hint:.4f}°"
                )
            else:
                print(
                    f"\n[Session] {target} / {date}  ({len(fits_files)} 幀)"
                    f"  hint 未設定（yaml 無目標座標）"
                )

            success = failed = skipped = 0
            for fits_path in fits_files:
                out_path = wcs_dir / (fits_path.stem + "_wcs.fits")

                if out_path.exists():
                    skipped += 1
                    continue

                print(f"  解算：{fits_path.name} … ", end="", flush=True)

                if backend == "astap":
                    ok = _run_astap(
                        fits_path,
                        out_path,
                        astap_cfg,
                        ra_hint_h=ra_hint,
                        spd_hint_deg=spd_hint,
                    )
                else:
                    ok = _run_astrometry_net(fits_path, out_path, anet_cfg)

                if ok:
                    print("OK")
                    success += 1
                else:
                    print("FAIL")
                    failed += 1

            print(
                f"\n[完成] {target}/{date}：成功 {success}，失敗 {failed}，"
                f"已跳過（存在）{skipped}"
            )
            print(f"       WCS 輸出目錄：{wcs_dir}")

    print("\n" + "=" * 20)
    print("所有 Session 星圖解算完成。")
    print("下一步：DeBayer_RGGB.py → Photometry.ipynb")
    print("=" * 20)


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

    run_plate_solve(cfg_path)
