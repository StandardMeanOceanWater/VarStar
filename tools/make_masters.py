# -*- coding: utf-8 -*-
"""
make_masters.py — 只合成 Master Bias / Dark / Flat，不校正科學幀
================================================================
用法：
  python tools/make_masters.py               # 全部 session
  python tools/make_masters.py --date 20251122
  python tools/make_masters.py --date 20251220
"""

import sys
import argparse
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Calibration import (
    load_config,
    resolve_session_paths,
    create_master,
    normalize_flat,
    _list_image_files,
)
from astropy.io import fits


def shooting_date(files: list) -> str:
    """從第一個 raw 幀讀拍攝日期，回傳 YYYYMMDD 字串。"""
    if not files:
        return "unknown"
    f = files[0]
    try:
        if f.suffix.lower() == ".cr2":
            import struct
            with open(f, 'rb') as fh:
                h = fh.read(8)
                endian = '<' if h[:2] == b'II' else '>'
                fh.seek(4)
                ifd0 = struct.unpack(endian+'I', fh.read(4))[0]
                fh.seek(ifd0)
                n = struct.unpack(endian+'H', fh.read(2))[0]
                exif_off = None
                for _ in range(n):
                    tag, typ, cnt, val = struct.unpack(endian+'HHII', fh.read(12))
                    if tag == 0x8769:
                        exif_off = val
                if exif_off:
                    fh.seek(exif_off)
                    n = struct.unpack(endian+'H', fh.read(2))[0]
                    for _ in range(n):
                        tag, typ, cnt, val = struct.unpack(endian+'HHII', fh.read(12))
                        if tag == 0x9003:  # DateTimeOriginal
                            fh.seek(val)
                            dt = fh.read(20).decode('ascii', errors='ignore').strip('\x00')
                            # 格式 "2025:12:20 03:12:00"
                            return dt[:10].replace(':', '')
        else:
            from astropy.io import fits as _fits
            with _fits.open(f) as hh:
                date_obs = hh[0].header.get("DATE-OBS", "")
                if date_obs:
                    return date_obs[:10].replace('-', '')
    except Exception:
        pass
    return "unknown"


def make_masters_for_session(cfg: dict, session: dict) -> None:
    date = str(session["date"])

    # 路徑解析（用第一個 target 代表該 session）
    raw_targets = session.get("targets", session.get("target", "UNKNOWN"))
    first_target = raw_targets[0] if isinstance(raw_targets, list) else str(raw_targets)
    s = dict(session)
    s["target"] = first_target
    paths = resolve_session_paths(cfg, s)

    cal_cfg   = cfg.get("calibration", {})
    chunk     = int(cal_cfg.get("median_chunk_size", 10))
    bad_thr   = float(cal_cfg.get("flat_bad_pixel_threshold", 0.3))
    tz_offset = int(cal_cfg.get("tz_offset_hours", 8))

    masters_dir = paths["masters_dir"]
    flat_date   = paths["flat_date"]
    # calib_date 固定從 EXIF 自動讀取，不由 YAML 手填，避免記錯日期
    calib_date  = None  # bias_files 讀到後再呼叫 shooting_date()

    print(f"\n{'='*60}")
    print(f"  Session {date}  (calib_date={calib_date}  flat_date={flat_date})")
    print(f"  bias     : {paths['bias_dir']}")
    print(f"  dark     : {paths['dark_dir']}  ({paths['dark_temp_c']} C)")
    print(f"  flat raw : {paths.get('flat_dir')}")
    print(f"  masters  : {masters_dir}")
    print(f"{'='*60}")

    # ── Master Bias ──────────────────────────────────────────────────────────
    bias_files = _list_image_files(paths["bias_dir"])
    master_bias = None
    if bias_files:
        if calib_date is None:
            calib_date = shooting_date(bias_files)   # YAML 未填，從 EXIF 讀
        bias_out = masters_dir / f"master_bias_{calib_date}_cr2.fits"
        if bias_out.exists():
            print(f"  [SKIP] Master Bias 已存在：{bias_out.name}")
            with fits.open(str(bias_out)) as hh:
                master_bias = hh[0].data.astype("float32")
        else:
            master_bias = create_master(bias_files, "Master Bias", chunk, tz_offset)
            masters_dir.mkdir(parents=True, exist_ok=True)
            fits.writeto(str(bias_out), master_bias, overwrite=True)
            print(f"  [OK] Master Bias  -> {bias_out}")
    else:
        print(f"  [SKIP] bias 目錄無影像：{paths['bias_dir']}")

    # ── Master Dark ──────────────────────────────────────────────────────────
    dark_files = _list_image_files(paths["dark_dir"])
    master_dark = None
    if dark_files:
        temp_tag = f"_{paths['dark_temp_c']}c" if paths["dark_temp_c"] is not None else ""
        dark_out = masters_dir / f"master_dark_{calib_date}{temp_tag}_cr2.fits"
        if dark_out.exists():
            print(f"  [SKIP] Master Dark 已存在：{dark_out.name}")
            with fits.open(str(dark_out)) as hh:
                master_dark = hh[0].data.astype("float32")
        else:
            dark_raw = create_master(dark_files, "Master Dark", chunk, tz_offset)
            master_dark = dark_raw - master_bias if master_bias is not None else dark_raw
            masters_dir.mkdir(parents=True, exist_ok=True)
            fits.writeto(str(dark_out), master_dark, overwrite=True)
            print(f"  [OK] Master Dark  -> {dark_out}")
    else:
        print(f"  [SKIP] dark 目錄無影像：{paths['dark_dir']}")

    # ── Master Flat ──────────────────────────────────────────────────────────
    flat_dirs_by_fmt = paths.get("flat_dirs_by_format", {})
    if flat_dirs_by_fmt:
        for fmt, fdir in flat_dirs_by_fmt.items():
            flat_files = _list_image_files(fdir)
            if not flat_files:
                continue
            flat_out = masters_dir / f"master_flat_{flat_date}_{fmt}.fits"
            if flat_out.exists():
                print(f"  [SKIP] Master Flat ({fmt.upper()}) 已存在：{flat_out.name}")
                continue
            flat_raw = create_master(flat_files, f"Master Flat ({fmt.upper()})", chunk, tz_offset)
            mf = normalize_flat(flat_raw, master_bias, bad_thr)
            masters_dir.mkdir(parents=True, exist_ok=True)
            fits.writeto(str(flat_out), mf, overwrite=True)
            print(f"  [OK] Master Flat ({fmt.upper()}) -> {flat_out}")
    else:
        print(f"  [SKIP] flat 目錄無影像：{paths.get('flat_dir')}")


def main():
    parser = argparse.ArgumentParser(description="合成 Master Bias/Dark/Flat")
    parser.add_argument("--date", help="只處理指定日期的 session，例如 20251122")
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parents[1] / "observation_config.yaml"
    cfg = load_config(config_path)

    sessions = cfg.get("obs_sessions", [])
    if args.date:
        sessions = [s for s in sessions if str(s["date"]) == args.date]
        if not sessions:
            print(f"[ERROR] 找不到 date={args.date} 的 session")
            sys.exit(1)

    for session in sessions:
        make_masters_for_session(cfg, session)

    print("\n完成。")


if __name__ == "__main__":
    main()
