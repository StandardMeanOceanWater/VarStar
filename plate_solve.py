# -*- coding: utf-8 -*-
"""
plate_solve.py  —  星圖解算模組
專案：變星測光管線 v1.0
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
from astropy.wcs import WCS

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
# ASTAP 後端
# =============================================================================

def _run_astap(
    fits_path: Path,
    out_path: Path,
    astap_cfg: dict,
    ra_hint_h: float | None = None,
    dec_hint_deg: float | None = None,
) -> bool:
    """
    呼叫 ASTAP CLI 對單張 FITS 執行星圖解算，輸出到 out_path。

    ASTAP 解算流程：
        1. 複製 fits_path 到臨時目錄（避免 ASTAP 直接覆蓋來源）
        2. 執行 ASTAP CLI（-update 旗標讓 ASTAP 把 WCS 寫回副本）
        3. 把 WCS 從 ASTAP 輸出的 .wcs 或更新後的 FITS 讀進來
        4. 將 WCS 標頭合併到原始 FITS 數據，寫出到 out_path

    Parameters
    ----------
    fits_path    : 輸入的校正 FITS（不會被修改）。
    out_path     : 輸出的 WCS FITS 路徑。
    astap_cfg    : yaml 裡 astrometry.astap 的設定字典。
    ra_hint_h    : 起始 RA（小時），覆蓋 FITS 標頭值；None 表示用標頭值。
    dec_hint_deg : 起始 DEC（度），覆蓋 FITS 標頭值；None 表示用標頭值。

    Returns
    -------
    bool
        解算成功回傳 True，失敗回傳 False。
    """
    executable = astap_cfg.get("executable", "astap_cli")
    db_path = astap_cfg.get("db_path", "")
    search_radius = float(astap_cfg.get("search_radius_deg", 30.0))
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
        # hint 座標：覆蓋 FITS 標頭的 RA/DEC，用於標頭座標不可靠的情況
        # ASTAP -spd 為南極距（South Polar Distance）= 90 - DEC
        if ra_hint_h is not None:
            cmd += ["-ra", str(ra_hint_h)]
        if dec_hint_deg is not None:
            spd = 90.0 - dec_hint_deg
            cmd += ["-spd", str(spd)]

        for attempt in range(max_retries):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                if result.returncode == 0 and tmp_fits.exists():
                    # 驗證 WCS 是否真的寫入
                    with fits.open(tmp_fits) as hdul:
                        hdr = hdul[0].header
                        if "CRVAL1" not in hdr and "CD1_1" not in hdr:
                            print(f"  [WARN] ASTAP 回傳 0 但 WCS 標頭缺失：{fits_path.name}")
                            continue
                    # 把解算後的整個 FITS 複製到 out_path
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

    # 登入取得 session key
    login_data = json.dumps({"apikey": api_key}).encode()
    req = urllib.request.Request(
        f"{base_url}/login",
        data=urllib.parse.urlencode({"request-json": login_data.decode()}).encode(),
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        login_resp = json.loads(resp.read())
    if login_resp.get("status") != "success":
        print(f"  [錯誤] astrometry.net 登入失敗：{login_resp}")
        return None
    session = login_resp["session"]

    # 把降採樣影像存成暫時 FITS
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tf:
        tmp_name = tf.name
    try:
        fits.writeto(tmp_name, data_small.astype(np.float32), overwrite=True)

        with open(tmp_name, "rb") as fh:
            file_bytes = fh.read()
    finally:
        os.unlink(tmp_name)

    # 上傳
    upload_params = {
        "session": session,
        "allow_commercial_use": "n",
        "allow_modifications": "n",
        "publicly_visible": "n",
    }
    # multipart/form-data 手工組裝（避免依賴 requests）
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

    # 等待解算完成
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

    # 確認 job 成功
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

    # 降採樣
    data_small = data_orig[::upload_downsample, ::upload_downsample]

    job_id = _upload_to_astrometry_net(data_small, api_key, timeout)
    if job_id is None:
        return False

    # 下載 WCS 標頭
    wcs_url = f"http://nova.astrometry.net/wcs_file/{job_id}"
    try:
        with urllib.request.urlopen(wcs_url, timeout=30) as resp:
            wcs_bytes = resp.read()
    except Exception as exc:
        print(f"  [錯誤] 下載 WCS 失敗：{exc}")
        return False

    with fits.open(io.BytesIO(wcs_bytes)) as wcs_hdul:
        wcs_header_small = wcs_hdul[0].header.copy()

    # 換算 WCS 回原始解析度
    wcs_header_full = _scale_wcs_to_original(wcs_header_small, upload_downsample)

    # 合併到原始標頭
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

    讀取 observation_config.yaml，對每個 obs_session 的
    calibrated/ 目錄內所有校正 FITS 執行星圖解算，
    輸出到 calibrated/wcs/ 子目錄。

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
        # targets / target 雙格式相容（與 Calibration.py 一致）
        raw_targets = session.get("targets", session.get("target", "UNKNOWN"))
        targets_list: list[str] = (
            [str(raw_targets)]
            if isinstance(raw_targets, str)
            else [str(t) for t in raw_targets]
        )
        date = str(session["date"])
        data_root = cfg["_data_root"]

        for target in targets_list:
            cal_dir = data_root / "targets" / target / "calibrated"
            wcs_dir = cal_dir / "wcs"
            wcs_dir.mkdir(parents=True, exist_ok=True)

            fits_files = sorted(
                f for f in cal_dir.glob("*.fits")
                if f.is_file() and "wcs" not in f.stem.lower()
            )

            if not fits_files:
                print(f"[SKIP] {target}/{date}：calibrated/ 目錄裡找不到 FITS。")
                continue

            print(f"\n[Session] {target} / {date}  ({len(fits_files)} 幀)")

            # target 層級的座標 hint（用於 FITS 標頭 RA/DEC 不可靠的情況）
            target_cfg = cfg.get("targets", {}).get(target, {})
            ra_hint_h: float | None = target_cfg.get("ra_hint_h")
            dec_hint_deg: float | None = target_cfg.get("dec_hint_deg")
            if ra_hint_h is not None or dec_hint_deg is not None:
                print(
                    f"  [INFO] 使用座標 hint："
                    f"RA={ra_hint_h}h  DEC={dec_hint_deg}°"
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
                        fits_path, out_path, astap_cfg,
                        ra_hint_h=ra_hint_h,
                        dec_hint_deg=dec_hint_deg,
                    )
                else:
                    ok = _run_astrometry_net(fits_path, out_path, anet_cfg)

                if ok:
                    print("✓")
                    success += 1
                else:
                    print("✗")
                    failed += 1

            print(f"\n[完成] {target}/{date}：成功 {success}，失敗 {failed}，"
                  f"已跳過（存在）{skipped}")
            print(f"       WCS 輸出目錄：{wcs_dir}")

    print("\n" + "🔭 " * 20)
    print("所有 Session 星圖解算完成。")
    print("下一步：DeBayer_RGGB.py → Photometry.ipynb")
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

    run_plate_solve(cfg_path)
