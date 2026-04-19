# -*- coding: utf-8 -*-
"""
build_superflat.py — 從科學幀建立通用型 superflat
=================================================

核心流程：
  1. 從 raw 目錄自動推回 session date 與 group
  2. 依副檔名分流（FITS / CR2）
  3. 套用既有 bias + dark_rate 做基本校正
  4. 預掃描每張影像的 robust normalization 與飽和值比例，過濾異常幀
  5. 對保留幀做 chunked median-of-medians 疊成 superflat
  6. 輸出到 data/share/master，可作為 flat-like 候選幀供驗證用

預設輸出檔名：
  superflat_<YYYYMMDD>_<GROUP>_<fits|cr2>.fits
"""

from __future__ import annotations

import argparse
import gc
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits
from tqdm import tqdm

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Calibration import (  # noqa: E402
    _dark_rate_paths,
    _ensure_master_root,
    _list_image_files,
    _load_first_existing,
    _master_bias_paths,
    _master_dark_paths,
    _read_frame_data,
    _sensor_db_for_session,
    compute_dark_rate,
    compute_dark_scaled,
    load_config,
    read_frame_metadata,
)


@dataclass
class FrameScan:
    path: Path
    norm: float
    sat_frac: float
    exposure_s: Optional[float]


def _ascii_safe(text: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return cleaned or fallback


def _infer_from_raw_dir(raw_dir: Path) -> Tuple[str, str]:
    parts = raw_dir.resolve().parts
    try:
        idx = next(i for i, part in enumerate(parts) if part.lower() == "data")
    except StopIteration as exc:
        raise ValueError(f"raw 目錄不在 data/ 結構下：{raw_dir}") from exc

    if idx + 2 >= len(parts):
        raise ValueError(f"無法從 raw 目錄推回日期與 group：{raw_dir}")
    date_part = parts[idx + 1]
    group_part = parts[idx + 2]
    date_digits = date_part.replace("-", "")
    if not re.fullmatch(r"\d{8}", date_digits):
        raise ValueError(f"無法解析日期：{date_part}")
    return date_digits, group_part


def _select_session(cfg: dict, session_date: str, group_name: str) -> dict:
    target_cfg = cfg.get("targets", {})
    for session in cfg.get("obs_sessions", []):
        if str(session.get("date")) != str(session_date):
            continue
        targets = session.get("targets", [])
        if not isinstance(targets, list):
            targets = [session.get("target")] if session.get("target") else []
        for target_name in targets:
            if target_cfg.get(target_name, {}).get("group") == group_name:
                return dict(session)
    raise ValueError(f"找不到 date={session_date}, group={group_name} 對應的 session")


def _master_roots(cfg: dict) -> List[Path]:
    roots = [cfg["_data_root"] / "share" / "calibration" / "master", _ensure_master_root(cfg)]
    deduped: List[Path] = []
    for root in roots:
        if root not in deduped and root.exists():
            deduped.append(root)
    return deduped


def _load_first_existing_from_roots(roots: Sequence[Path], relpaths: Sequence[Path]) -> Optional[Tuple[np.ndarray, fits.Header, Path]]:
    for root in roots:
        existing = _load_first_existing([root / rel.name for rel in relpaths])
        if existing is not None:
            return existing
    return None


def _glob_first(root: Path, patterns: Sequence[str]) -> Optional[Tuple[np.ndarray, fits.Header, Path]]:
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(root.glob(pattern))
    candidates = sorted({path for path in candidates}, key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    return _load_first_existing([candidates[0]])


def _load_master_bias(cfg: dict, session: dict) -> Tuple[Optional[np.ndarray], Optional[Path]]:
    roots = _master_roots(cfg)
    iso_value = int(session["iso"]) if session.get("iso") is not None else None
    probe_root = _ensure_master_root(cfg)
    relpaths = [path.relative_to(probe_root) for path in _master_bias_paths(probe_root, str(session["date"]), "cr2", iso_value)]
    existing = _load_first_existing_from_roots(roots, relpaths)
    if existing is None:
        iso_tag = f"_iso{iso_value}" if iso_value is not None else ""
        patterns = [
            f"master_bias_raw_*_cr2{iso_tag}.fits",
            f"master_bias_*_cr2{iso_tag}.fits",
            "master_bias_*_cr2.fits",
        ]
        for root in roots:
            existing = _glob_first(root, patterns)
            if existing is not None:
                break
    if existing is None:
        return None, None
    data, _, path = existing
    return data.astype(np.float32), path


def _load_dark_rate(cfg: dict, session: dict) -> Tuple[Optional[np.ndarray], Optional[Path]]:
    roots = _master_roots(cfg)
    iso_value = int(session["iso"]) if session.get("iso") is not None else None
    dark_temp_c = session.get("dark_temp_c")

    probe_root = _ensure_master_root(cfg)
    relpaths = [path.relative_to(probe_root) for path in _dark_rate_paths(probe_root, str(session["date"]), "cr2", iso_value, dark_temp_c)]
    existing = _load_first_existing_from_roots(roots, relpaths)
    if existing is not None:
        data, _, path = existing
        return data.astype(np.float32), path

    dark_relpaths = [path.relative_to(probe_root) for path in _master_dark_paths(probe_root, str(session["date"]), "cr2", iso_value, dark_temp_c)]
    dark_existing = _load_first_existing_from_roots(roots, dark_relpaths)
    if dark_existing is None:
        iso_tag = f"_iso{iso_value}" if iso_value is not None else ""
        temp_token = f"{dark_temp_c:.1f}c" if dark_temp_c is not None else ""
        patterns = [
            f"master_dark_raw_*{temp_token}*cr2{iso_tag}*.fits",
            f"master_dark_*{temp_token}*cr2{iso_tag}*.fits",
            f"master_dark_*{temp_token}*cr2*.fits",
            f"master_dark_raw_*cr2{iso_tag}*.fits",
            f"master_dark_*cr2{iso_tag}*.fits",
            "master_dark_*cr2*.fits",
        ]
        for root in roots:
            dark_existing = _glob_first(root, patterns)
            if dark_existing is not None:
                break
    if dark_existing is None:
        return None, None

    data, header, path = dark_existing
    t_dark = header.get("EXPTIME")
    dark_rate = compute_dark_rate(data.astype(np.float32), float(t_dark) if t_dark is not None else None)
    if dark_rate is None:
        return None, None
    return dark_rate, path


def _frame_normalization_value(frame: np.ndarray, low_pct: float, high_pct: float) -> float:
    finite = frame[np.isfinite(frame)]
    if finite.size == 0:
        raise ValueError("frame has no finite pixels")
    lo = np.nanpercentile(finite, low_pct)
    hi = np.nanpercentile(finite, high_pct)
    window = finite[(finite >= lo) & (finite <= hi)]
    if window.size == 0:
        window = finite
    norm = float(np.nanmedian(window))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("frame normalization median is zero or non-finite")
    return norm


def _scan_frames(
    files: Sequence[Path],
    cfg: dict,
    session: dict,
    master_bias: Optional[np.ndarray],
    dark_rate: Optional[np.ndarray],
    low_pct: float,
    high_pct: float,
    sat_threshold: float,
) -> Tuple[List[FrameScan], Tuple[int, int]]:
    tz_offset = 8
    gain_e, read_noise_e, saturation_adu = _sensor_db_for_session(cfg, session)
    scans: List[FrameScan] = []
    ref_shape: Optional[Tuple[int, int]] = None

    for path in tqdm(files, desc="scan frames", unit="file"):
        try:
            light_data, _ = _read_frame_data(path, session, tz_offset, gain_e, read_noise_e, saturation_adu)
            if ref_shape is None:
                ref_shape = light_data.shape
            if light_data.shape != ref_shape:
                print(f"  [WARN] 尺寸不符，跳過：{path.name} ({light_data.shape} != {ref_shape})")
                continue
            meta = read_frame_metadata(path, str(session["date"]))
            calibrated = light_data.astype(np.float64)
            if master_bias is not None:
                calibrated -= master_bias.astype(np.float64)
            dark_scaled = compute_dark_scaled(dark_rate, meta.get("exposure_s"))
            if dark_scaled is not None:
                calibrated -= dark_scaled.astype(np.float64)
            norm = _frame_normalization_value(calibrated, low_pct=low_pct, high_pct=high_pct)
            finite = calibrated[np.isfinite(calibrated)]
            if finite.size == 0:
                continue
            sat_limit = np.nanpercentile(finite, 99.9)
            sat_frac = float(np.mean(finite >= sat_limit))
            if sat_frac > sat_threshold:
                print(f"  [WARN] 飽和值比例過高，跳過：{path.name} sat_frac={sat_frac:.4f}")
                continue
            scans.append(FrameScan(path=path, norm=norm, sat_frac=sat_frac, exposure_s=meta.get("exposure_s")))
        except Exception as exc:
            print(f"  [WARN] 預掃描失敗，跳過：{path.name} ({exc})")

    if ref_shape is None:
        raise ValueError("沒有可讀取的 light frame")
    return scans, ref_shape


def _filter_scans(scans: Sequence[FrameScan], sigma: float) -> List[FrameScan]:
    if not scans:
        return []
    norms = np.array([item.norm for item in scans], dtype=np.float64)
    median = float(np.nanmedian(norms))
    mad = float(np.nanmedian(np.abs(norms - median)))
    if not math.isfinite(mad) or mad == 0.0:
        return list(scans)
    robust_sigma = 1.4826 * mad
    lo = median - sigma * robust_sigma
    hi = median + sigma * robust_sigma
    kept = [item for item in scans if lo <= item.norm <= hi]
    print(
        f"[INFO] norm filter : median={median:.6f}, robust_sigma={robust_sigma:.6f}, "
        f"window=[{lo:.6f}, {hi:.6f}], kept={len(kept)}/{len(scans)}"
    )
    return kept


def _superflat_stack(
    scans: Sequence[FrameScan],
    ref_shape: Tuple[int, int],
    cfg: dict,
    session: dict,
    master_bias: Optional[np.ndarray],
    dark_rate: Optional[np.ndarray],
    chunk_size: int,
) -> Tuple[np.ndarray, int]:
    tz_offset = 8
    gain_e, read_noise_e, saturation_adu = _sensor_db_for_session(cfg, session)
    chunk_medians: List[np.ndarray] = []
    used_count = 0

    chunk_total = math.ceil(len(scans) / chunk_size)
    for chunk_index, start in enumerate(range(0, len(scans), chunk_size), start=1):
        batch = scans[start:start + chunk_size]
        print(f"[INFO] chunk {chunk_index}/{chunk_total} : loading {len(batch)} frames")
        stack: List[np.ndarray] = []
        for item in tqdm(batch, desc=f"stack chunk {chunk_index}", unit="file", leave=False):
            try:
                light_data, _ = _read_frame_data(item.path, session, tz_offset, gain_e, read_noise_e, saturation_adu)
                if light_data.shape != ref_shape:
                    print(f"  [WARN] 尺寸不符，跳過：{item.path.name}")
                    continue
                calibrated = light_data.astype(np.float64)
                if master_bias is not None:
                    calibrated -= master_bias.astype(np.float64)
                dark_scaled = compute_dark_scaled(dark_rate, item.exposure_s)
                if dark_scaled is not None:
                    calibrated -= dark_scaled.astype(np.float64)
                normalized = calibrated / float(item.norm)
                normalized = np.where(np.isfinite(normalized), normalized, np.nan).astype(np.float32)
                stack.append(normalized)
                used_count += 1
            except Exception as exc:
                print(f"  [WARN] 疊圖失敗，跳過：{item.path.name} ({exc})")
        if stack:
            chunk_median = np.nanmedian(np.array(stack, dtype=np.float32), axis=0).astype(np.float32)
            chunk_medians.append(chunk_median)
        del stack
        gc.collect()

    if not chunk_medians:
        raise ValueError("所有 light frame 都無法用於 superflat")

    superflat = np.nanmedian(np.array(chunk_medians, dtype=np.float32), axis=0).astype(np.float32)
    del chunk_medians
    gc.collect()
    return superflat, used_count


def _finalize_superflat(superflat_raw: np.ndarray, floor: float) -> np.ndarray:
    median_value = float(np.nanmedian(superflat_raw))
    if not math.isfinite(median_value) or median_value == 0.0:
        raise ValueError("superflat raw median is zero or non-finite")
    superflat = (superflat_raw.astype(np.float64) / median_value).astype(np.float32)
    superflat = np.where(np.isfinite(superflat), superflat, floor)
    superflat = np.clip(superflat, floor, None).astype(np.float32)
    return superflat


def _build_output_header(
    session: dict,
    group_name: str,
    frame_type: str,
    source_count: int,
    scanned_count: int,
    raw_dir: Path,
    master_bias_path: Optional[Path],
    dark_source_path: Optional[Path],
) -> fits.Header:
    header = fits.Header()
    header["IMAGETYP"] = ("SUPERFLAT", "derived from science frames")
    header["SESSDATE"] = (str(session["date"]), "session date")
    header["GROUP"] = (_ascii_safe(group_name, "group"), "source group")
    header["FRMTYPE"] = (frame_type, "source frame type")
    header["NCOMBINE"] = (int(source_count), "number of frames stacked")
    header["NSCANNED"] = (int(scanned_count), "number of frames scanned")
    header["SRCDIR"] = (_ascii_safe(raw_dir.name, "raw"), "source raw dir name")
    header["USEBIAS"] = (1 if master_bias_path is not None else 0, "bias subtraction enabled")
    header["USEDARK"] = (1 if dark_source_path is not None else 0, "dark subtraction enabled")
    if master_bias_path is not None:
        header["BIASREF"] = (master_bias_path.name, "bias master used")
    if dark_source_path is not None:
        header["DARKREF"] = (dark_source_path.name, "dark source used")
    header["NORMMODE"] = ("p20-80-median", "per-frame normalization mode")
    header["COMMENT"] = "science-frame derived candidate superflat"
    header["COMMENT"] = "inspect visually before any production calibration use"
    return header


def _build_output_path(
    cfg: dict,
    date_text: str,
    group_name: str,
    frame_type: str,
    output: Optional[Path],
    multiple: bool,
) -> Path:
    safe_group = _ascii_safe(group_name, "group")
    default_name = f"superflat_{date_text}_{safe_group}_{frame_type}.fits"
    if output is None:
        return _ensure_master_root(cfg) / default_name
    if not multiple:
        return output
    suffix = output.suffix or ".fits"
    stem = output.stem if output.suffix else output.name
    return output.with_name(f"{stem}_{frame_type}{suffix}")


def _group_files_by_format(files: Iterable[Path], mode: str) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for path in files:
        ext = path.suffix.lower()
        if ext in (".fits", ".fit"):
            frame_type = "fits"
        elif ext == ".cr2":
            frame_type = "cr2"
        else:
            continue
        if mode != "all" and frame_type != mode:
            continue
        groups.setdefault(frame_type, []).append(path)
    return {k: sorted(v) for k, v in groups.items()}


def _run_one_format(
    files: Sequence[Path],
    cfg: dict,
    session: dict,
    raw_dir: Path,
    group_name: str,
    frame_type: str,
    output_path: Path,
    chunk_size: int,
    floor: float,
    low_pct: float,
    high_pct: float,
    filter_sigma: float,
    sat_threshold: float,
) -> None:
    print(f"\n[RUN] {group_name} {session['date']} {frame_type.upper()} : {len(files)} frames")
    master_bias, master_bias_path = _load_master_bias(cfg, session)
    dark_rate, dark_source_path = _load_dark_rate(cfg, session)
    print(f"[INFO] raw dir      : {raw_dir}")
    print(f"[INFO] bias master  : {master_bias_path if master_bias_path is not None else 'None'}")
    print(f"[INFO] dark source  : {dark_source_path if dark_source_path is not None else 'None'}")

    scans, ref_shape = _scan_frames(
        files,
        cfg,
        session,
        master_bias=master_bias,
        dark_rate=dark_rate,
        low_pct=low_pct,
        high_pct=high_pct,
        sat_threshold=sat_threshold,
    )
    kept = _filter_scans(scans, sigma=filter_sigma)
    if len(kept) < 10:
        raise ValueError(f"可用幀太少：{len(kept)}")

    superflat_raw, used_count = _superflat_stack(
        kept,
        ref_shape,
        cfg,
        session,
        master_bias=master_bias,
        dark_rate=dark_rate,
        chunk_size=chunk_size,
    )
    superflat = _finalize_superflat(superflat_raw, floor=floor)

    header = _build_output_header(
        session=session,
        group_name=group_name,
        frame_type=frame_type,
        source_count=used_count,
        scanned_count=len(scans),
        raw_dir=raw_dir,
        master_bias_path=master_bias_path,
        dark_source_path=dark_source_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(output_path, superflat.astype(np.float32), header=header, overwrite=True)

    print(f"[OK] superflat -> {output_path}")
    print(
        f"[INFO] stats: min={np.nanmin(superflat):.6f} "
        f"median={np.nanmedian(superflat):.6f} max={np.nanmax(superflat):.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="從 raw 目錄建立通用型 superflat")
    parser.add_argument("--raw-dir", type=Path, required=True, help="例如 data/2025-12-20/Ori/raw")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "observation_config.yaml")
    parser.add_argument("--format", choices=["all", "fits", "cr2"], default="all")
    parser.add_argument("--output", type=Path, help="單一格式輸出路徑；多格式時會自動加上 _fits/_cr2")
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--limit", type=int, help="每種格式只用前 N 張，供快速測試")
    parser.add_argument("--floor", type=float, default=0.3)
    parser.add_argument("--low-pct", type=float, default=20.0)
    parser.add_argument("--high-pct", type=float, default=80.0)
    parser.add_argument("--filter-sigma", type=float, default=3.0, help="以 norm MAD 過濾異常幀")
    parser.add_argument("--sat-threshold", type=float, default=0.02, help="高亮尾端像素比例上限")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = args.raw_dir.resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"找不到 raw 目錄：{raw_dir}")

    session_date, group_name = _infer_from_raw_dir(raw_dir)
    session = _select_session(cfg, session_date, group_name)
    print(f"[INFO] inferred date  : {session_date}")
    print(f"[INFO] inferred group : {group_name}")

    all_files = _list_image_files(raw_dir)
    grouped = _group_files_by_format(all_files, mode=args.format)
    if not grouped:
        raise ValueError(f"找不到可處理的 {args.format} 檔案：{raw_dir}")

    for frame_type, files in grouped.items():
        if args.limit is not None:
            files = files[:args.limit]
        out_path = _build_output_path(
            cfg,
            date_text=session_date,
            group_name=group_name,
            frame_type=frame_type,
            output=args.output,
            multiple=len(grouped) > 1,
        )
        _run_one_format(
            files=files,
            cfg=cfg,
            session=session,
            raw_dir=raw_dir,
            group_name=group_name,
            frame_type=frame_type,
            output_path=out_path,
            chunk_size=args.chunk_size,
            floor=args.floor,
            low_pct=args.low_pct,
            high_pct=args.high_pct,
            filter_sigma=args.filter_sigma,
            sat_threshold=args.sat_threshold,
        )


if __name__ == "__main__":
    main()
