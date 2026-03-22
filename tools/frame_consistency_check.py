# -*- coding: utf-8 -*-
"""
frame_consistency_check.py — 照片一致性檢查工具
================================================
在解星完成後，巡檢同一 session 所有 _wcs.fits 的 FITS header，
檢查曝光時間、ISO、檔案類型（CR2 vs FITS 原始來源）、中心天球座標等是否一致。
不一致時輸出警告；一致時僅在 LOG 回報 OK。

用法：
  python tools/frame_consistency_check.py                          # 全部 session
  python tools/frame_consistency_check.py --date 20251122          # 單一日期
  python tools/frame_consistency_check.py --date 20251220 --group Ori  # 指定群組

輸出：
  - 終端摘要（一致 → OK，不一致 → WARNING + 詳細列表）
  - 可選 CSV 輸出到 output/_pipeline_log/frame_consistency_{date}_{group}.csv

2026-03-22 建立
"""

import sys
import argparse
from pathlib import Path
from collections import Counter

import numpy as np

# Windows cp950 安全輸出
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _safe_float(val, default=None):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _detect_source_format(header):
    """從 FITS header 推測原始來源格式"""
    # NINA FITS 通常有 INSTRUME=ASCOM.DSLR 或 SOFTWARE=N.I.N.A.
    sw = str(header.get("SOFTWARE", "")).upper()
    instr = str(header.get("INSTRUME", "")).upper()
    if "NINA" in sw or "N.I.N.A" in sw:
        return "FITS(NINA)"
    if "ASCOM" in instr:
        return "FITS(ASCOM)"
    # rawpy/dcraw 處理過的 CR2 通常沒有 SOFTWARE
    # 但 Calibration.py 會寫 HISTORY
    history = str(header.get("HISTORY", ""))
    if "CR2" in history.upper() or "rawpy" in history.lower():
        return "CR2"
    # 從檔名推測
    return "unknown"


def check_session(wcs_dir: Path, verbose: bool = True):
    """
    檢查一個 wcs/ 目錄下所有 _wcs.fits 的 header 一致性。
    回傳 (ok: bool, summary: dict, issues: list[str])
    """
    from astropy.io import fits

    fits_files = sorted(wcs_dir.glob("*_wcs.fits"))
    if not fits_files:
        return True, {"n_frames": 0}, ["無 _wcs.fits 檔案"]

    records = []
    for f in fits_files:
        try:
            hdr = fits.getheader(f)
        except Exception as exc:
            records.append({"file": f.name, "error": str(exc)})
            continue

        rec = {
            "file":     f.name,
            "exptime":  _safe_float(hdr.get("EXPTIME") or hdr.get("EXPOSURE")),
            "iso":      hdr.get("ISO") or hdr.get("ISOSPEED") or hdr.get("GAIN"),
            "xbinning": hdr.get("XBINNING"),
            "naxis1":   hdr.get("NAXIS1"),
            "naxis2":   hdr.get("NAXIS2"),
            "crval1":   _safe_float(hdr.get("CRVAL1")),  # RA center (deg)
            "crval2":   _safe_float(hdr.get("CRVAL2")),  # Dec center (deg)
            "date_obs": str(hdr.get("DATE-OBS", "")),
            "source_fmt": _detect_source_format(hdr),
            "bitpix":   hdr.get("BITPIX"),
        }
        records.append(rec)

    if not records:
        return False, {"n_frames": 0}, ["全部讀取失敗"]

    n = len(records)
    issues = []

    # --- 檢查各欄位一致性 ---

    # 曝光時間
    exptimes = [r["exptime"] for r in records if r.get("exptime") is not None]
    if exptimes:
        unique_exp = set(exptimes)
        if len(unique_exp) > 1:
            counts = Counter(exptimes)
            detail = ", ".join(f"{v}s x{c}" for v, c in counts.most_common())
            issues.append(f"[WARNING] 曝光時間不一致：{detail}")
    else:
        issues.append("[WARNING] 無法讀取任何幀的曝光時間")

    # ISO / GAIN
    isos = [r["iso"] for r in records if r.get("iso") is not None]
    if isos:
        unique_iso = set(str(i) for i in isos)
        if len(unique_iso) > 1:
            counts = Counter(str(i) for i in isos)
            detail = ", ".join(f"ISO{v} x{c}" for v, c in counts.most_common())
            issues.append(f"[WARNING] ISO/GAIN 不一致：{detail}")
    else:
        issues.append("[INFO] 無法讀取 ISO（Canon CR2 已知問題，非異常）")

    # 影像尺寸
    dims = [(r.get("naxis1"), r.get("naxis2")) for r in records
            if r.get("naxis1") is not None]
    if dims:
        unique_dims = set(dims)
        if len(unique_dims) > 1:
            counts = Counter(dims)
            detail = ", ".join(f"{w}x{h} x{c}" for (w, h), c in counts.most_common())
            issues.append(f"[WARNING] 影像尺寸不一致：{detail}")

    # 存檔類型
    fmts = [r.get("source_fmt", "unknown") for r in records]
    unique_fmts = set(fmts)
    if len(unique_fmts) > 1:
        counts = Counter(fmts)
        detail = ", ".join(f"{f} x{c}" for f, c in counts.most_common())
        issues.append(f"[WARNING] 存檔類型混合：{detail}")

    # 天球座標一致性（中心座標偏移 > 0.5 度視為異常）
    ras = [r["crval1"] for r in records if r.get("crval1") is not None]
    decs = [r["crval2"] for r in records if r.get("crval2") is not None]
    if ras and decs:
        ra_range = max(ras) - min(ras)
        dec_range = max(decs) - min(decs)
        if ra_range > 0.5 or dec_range > 0.5:
            issues.append(
                f"[WARNING] 天球座標偏移過大："
                f"RA 範圍 {ra_range:.3f} deg, Dec 範圍 {dec_range:.3f} deg"
            )
            # 找出最偏離的幀
            ra_med = np.median(ras)
            dec_med = np.median(decs)
            for r in records:
                if r.get("crval1") is not None:
                    dist = np.sqrt((r["crval1"] - ra_med)**2 +
                                   (r["crval2"] - dec_med)**2)
                    if dist > 0.3:
                        issues.append(
                            f"  -> {r['file']}: "
                            f"RA={r['crval1']:.4f}, Dec={r['crval2']:.4f} "
                            f"(偏離中位 {dist:.3f} deg)"
                        )
    else:
        issues.append("[WARNING] 無法讀取天球座標（WCS 可能不完整）")

    # BITPIX 一致性
    bitpixes = [r.get("bitpix") for r in records if r.get("bitpix") is not None]
    if bitpixes:
        unique_bp = set(bitpixes)
        if len(unique_bp) > 1:
            counts = Counter(bitpixes)
            detail = ", ".join(f"BITPIX={v} x{c}" for v, c in counts.most_common())
            issues.append(f"[WARNING] BITPIX 不一致：{detail}")

    # 組裝摘要
    ok = all(not i.startswith("[WARNING]") for i in issues)
    summary = {
        "n_frames":   n,
        "exptime":    exptimes[0] if exptimes and len(set(exptimes)) == 1 else "mixed",
        "iso":        isos[0] if isos and len(set(str(i) for i in isos)) == 1 else "mixed/unknown",
        "dimensions": f"{dims[0][0]}x{dims[0][1]}" if dims and len(set(dims)) == 1 else "mixed",
        "source_fmt": fmts[0] if len(unique_fmts) == 1 else "mixed",
        "ra_center":  f"{np.median(ras):.4f}" if ras else "N/A",
        "dec_center": f"{np.median(decs):.4f}" if decs else "N/A",
    }

    return ok, summary, issues, records


def main():
    import yaml

    parser = argparse.ArgumentParser(description="照片一致性檢查")
    parser.add_argument("--date", type=str, default=None, help="觀測日期 (YYYYMMDD)")
    parser.add_argument("--group", type=str, default=None, help="照片組名稱")
    parser.add_argument("--save-csv", action="store_true", help="輸出 CSV 到 pipeline log")
    args = parser.parse_args()

    cfg_path = Path(__file__).parent.parent / "observation_config.yaml"
    if not cfg_path.exists():
        print(f"找不到設定檔：{cfg_path}")
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg.get("data_root", r"D:\VarStar\data"))
    sessions = cfg.get("obs_sessions", [])

    print("=" * 60)
    print("  照片一致性檢查  frame_consistency_check.py")
    print("=" * 60)

    total_ok = 0
    total_warn = 0

    for session in sessions:
        date = str(session["date"])
        if args.date and date != args.date:
            continue

        targets = session.get("targets", [])
        targets_cfg = cfg.get("targets", {})

        for target_name in targets:
            tgt = targets_cfg.get(target_name, {})
            group = tgt.get("group", target_name)
            if args.group and group != args.group:
                continue

            date_fmt = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
            wcs_dir = data_root / date_fmt / group / "wcs"

            if not wcs_dir.exists():
                print(f"\n[SKIP] {target_name}/{date}: wcs/ 不存在")
                continue

            ok, summary, issues, records = check_session(wcs_dir)

            if ok:
                total_ok += 1
                print(
                    f"\n[OK] {target_name}/{date}: "
                    f"{summary['n_frames']} 幀, "
                    f"exp={summary['exptime']}s, "
                    f"fmt={summary['source_fmt']}, "
                    f"center=({summary['ra_center']}, {summary['dec_center']})"
                )
            else:
                total_warn += 1
                print(f"\n[!!!] {target_name}/{date}: {summary['n_frames']} 幀")
                for issue in issues:
                    print(f"  {issue}")

            # 可選 CSV 輸出
            if args.save_csv and records:
                import pandas as pd
                log_dir = Path(cfg.get("output_root", r"D:\VarStar\output")) / "_pipeline_log"
                log_dir.mkdir(parents=True, exist_ok=True)
                csv_path = log_dir / f"frame_consistency_{date}_{group}.csv"
                pd.DataFrame(records).to_csv(csv_path, index=False, encoding="utf-8-sig")
                print(f"  CSV: {csv_path}")

    print(f"\n{'='*60}")
    print(f"  結果：{total_ok} OK, {total_warn} WARNING")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
