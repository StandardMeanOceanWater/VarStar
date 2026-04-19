# -*- coding: utf-8 -*-
"""
validate_superflat_r.py — 驗證 FITS superflat 對 R 通道的影響
=============================================================

輸出：
  - 每幀 raw / darkonly / superflat 的 R 通道背景統計
  - 一份 CSV 方便後續比較

用途：
  先做最小驗證，不碰主管線。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Calibration import (  # noqa: E402
    _read_frame_data,
    _sensor_db_for_session,
    compute_dark_scaled,
    load_config,
    read_frame_metadata,
)
from astropy.io import fits  # noqa: E402


def _extract_r_channel(data: np.ndarray) -> np.ndarray:
    return data[0::2, 0::2].astype(np.float32)


def _stats(arr: np.ndarray) -> dict[str, float]:
    finite = arr[np.isfinite(arr)]
    h, w = arr.shape
    cy, cx = h // 2, w // 2
    box = min(100, max(10, min(h, w) // 8))
    center = arr[cy - box:cy + box, cx - box:cx + box]
    bh = max(1, h // 10)
    bw = max(1, w // 10)
    corners = np.concatenate(
        [
            arr[:bh, :bw].ravel(),
            arr[:bh, -bw:].ravel(),
            arr[-bh:, :bw].ravel(),
            arr[-bh:, -bw:].ravel(),
        ]
    )
    return {
        "median": float(np.nanmedian(finite)),
        "mean": float(np.nanmean(finite)),
        "std": float(np.nanstd(finite)),
        "p05": float(np.nanpercentile(finite, 5)),
        "p95": float(np.nanpercentile(finite, 95)),
        "center_med": float(np.nanmedian(center)),
        "corner_med": float(np.nanmedian(corners)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="驗證 CCAnd FITS superflat 的 R 通道效果")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--superflat", type=Path, required=True)
    parser.add_argument("--bias", type=Path, required=True)
    parser.add_argument("--dark-rate", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "observation_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    session = next(s for s in cfg["obs_sessions"] if str(s["date"]) == "20251122")
    gain_e, read_noise_e, saturation_adu = _sensor_db_for_session(cfg, session)

    raw_files = sorted(args.raw_dir.glob("*.fits"))[: args.limit]
    if not raw_files:
        raise ValueError("找不到 FITS raw")

    with fits.open(args.bias) as hdul:
        bias = hdul[0].data.astype(np.float32)
    with fits.open(args.dark_rate) as hdul:
        dark_rate = hdul[0].data.astype(np.float32)
    with fits.open(args.superflat) as hdul:
        superflat = hdul[0].data.astype(np.float32)

    rows: list[dict[str, object]] = []
    for path in raw_files:
        light_data, _ = _read_frame_data(path, session, 8, gain_e, read_noise_e, saturation_adu)
        meta = read_frame_metadata(path, str(session["date"]))
        dark_scaled = compute_dark_scaled(dark_rate, meta.get("exposure_s"))

        raw_r = _extract_r_channel(light_data)
        darkonly_r = _extract_r_channel(light_data - bias - dark_scaled)
        super_r = _extract_r_channel((light_data - bias - dark_scaled) / superflat)

        for mode, arr in [("raw", raw_r), ("darkonly", darkonly_r), ("superflat", super_r)]:
            stat = _stats(arr)
            row = {"file": path.name, "mode": mode}
            row.update(stat)
            rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["file", "mode", "median", "mean", "std", "p05", "p95", "center_med", "corner_med"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] wrote {args.output_csv}")
    print(f"[INFO] frames={len(raw_files)}")


if __name__ == "__main__":
    main()
