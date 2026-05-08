from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Calibration import _list_image_files, read_raw_image


EPS = 1e-6


@dataclass
class ImageMetrics:
    path: Path
    bg_median: float
    bg_mad: float
    bg_mad_norm: float
    gradient_span: float
    gradient_norm: float
    center_corner_delta: float
    center_corner_norm: float
    finite_fraction: float


@dataclass
class PairMetrics:
    raw_path: Path
    cal_path: Path
    raw: ImageMetrics
    cal: ImageMetrics


def _load_raw_like(path: Path) -> np.ndarray:
    data, _ = read_raw_image(path, tz_offset_hours=8)
    return data.astype(np.float32)


def _load_fits_2d(path: Path) -> np.ndarray:
    with fits.open(path, ignore_missing_end=True) as hdul:
        hdu = next((item for item in hdul if item.data is not None), None)
        if hdu is None:
            raise ValueError(f"no data in FITS: {path.name}")
        data = np.asarray(hdu.data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"expected 2D FITS data: {path.name}")
    return data


def _load_image_for_metrics(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".cr2":
        return _load_raw_like(path)
    if suffix in (".fit", ".fits"):
        if path.name.startswith("Cal_"):
            return _load_fits_2d(path)
        return _load_raw_like(path)
    raise ValueError(f"unsupported file type: {path}")


def _background_window(data: np.ndarray) -> np.ndarray:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("image has no finite pixels")
    p05 = float(np.nanpercentile(finite, 5.0))
    p70 = float(np.nanpercentile(finite, 70.0))
    window = finite[(finite >= p05) & (finite <= p70)]
    if window.size < 128:
        window = finite
    return window


def _robust_median_and_mad(data: np.ndarray) -> Tuple[float, float]:
    window = _background_window(data)
    median = float(np.nanmedian(window))
    mad = float(np.nanmedian(np.abs(window - median)))
    return median, mad


def _tile_slices(length: int, parts: int) -> List[Tuple[int, int]]:
    edges = np.linspace(0, length, parts + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(parts)]


def _tile_background_medians(data: np.ndarray, rows: int = 4, cols: int = 4) -> np.ndarray:
    row_slices = _tile_slices(data.shape[0], rows)
    col_slices = _tile_slices(data.shape[1], cols)
    medians = np.zeros((rows, cols), dtype=np.float64)
    for r_index, (r0, r1) in enumerate(row_slices):
        for c_index, (c0, c1) in enumerate(col_slices):
            tile = data[r0:r1, c0:c1]
            medians[r_index, c_index] = _robust_median_and_mad(tile)[0]
    return medians


def compute_metrics(path: Path) -> ImageMetrics:
    data = _load_image_for_metrics(path)
    bg_median, bg_mad = _robust_median_and_mad(data)
    tile_medians = _tile_background_medians(data, rows=4, cols=4)
    gradient_span = float(np.nanmax(tile_medians) - np.nanmin(tile_medians))
    denom = max(abs(bg_median), EPS)
    gradient_norm = gradient_span / denom
    bg_mad_norm = bg_mad / denom

    center = float(np.nanmean(tile_medians[1:3, 1:3]))
    corners = float(
        np.nanmean(
            np.array(
                [
                    tile_medians[0, 0],
                    tile_medians[0, -1],
                    tile_medians[-1, 0],
                    tile_medians[-1, -1],
                ],
                dtype=np.float64,
            )
        )
    )
    center_corner_delta = center - corners
    center_corner_norm = abs(center_corner_delta) / denom
    finite_fraction = float(np.mean(np.isfinite(data)))
    return ImageMetrics(
        path=path,
        bg_median=bg_median,
        bg_mad=bg_mad,
        bg_mad_norm=bg_mad_norm,
        gradient_span=gradient_span,
        gradient_norm=gradient_norm,
        center_corner_delta=center_corner_delta,
        center_corner_norm=center_corner_norm,
        finite_fraction=finite_fraction,
    )


def _match_calibrated_file(raw_path: Path, cal_files: Sequence[Path]) -> Optional[Path]:
    stem = re.escape(raw_path.stem)
    pattern = re.compile(rf"^Cal_{stem}_\d{{4}}\.fits$", re.IGNORECASE)
    matches = [path for path in cal_files if pattern.match(path.name)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return sorted(matches)[0]
    return None


def pair_files(raw_dir: Path, cal_dir: Path, limit: Optional[int]) -> List[Tuple[Path, Path]]:
    raw_files = _list_image_files(raw_dir)
    cal_files = [path for path in _list_image_files(cal_dir) if path.suffix.lower() in (".fit", ".fits")]
    pairs: List[Tuple[Path, Path]] = []
    for raw_path in raw_files:
        cal_path = _match_calibrated_file(raw_path, cal_files)
        if cal_path is not None:
            pairs.append((raw_path, cal_path))
        if limit is not None and len(pairs) >= limit:
            break
    return pairs


def _mean_metric(items: Iterable[float]) -> float:
    values = [float(value) for value in items if math.isfinite(float(value))]
    if not values:
        return float("nan")
    return float(np.mean(values))


def _summary_table(pair_metrics: Sequence[PairMetrics]) -> List[str]:
    rows: List[Tuple[str, float, float, float, str]] = []

    def add_row(name: str, raw_values: Iterable[float], cal_values: Iterable[float], lower_is_better: bool) -> None:
        raw_mean = _mean_metric(raw_values)
        cal_mean = _mean_metric(cal_values)
        delta = cal_mean - raw_mean
        better = "yes" if ((cal_mean < raw_mean) if lower_is_better else (cal_mean > raw_mean)) else "no"
        rows.append((name, raw_mean, cal_mean, delta, better))

    add_row("gradient_norm", (item.raw.gradient_norm for item in pair_metrics), (item.cal.gradient_norm for item in pair_metrics), True)
    add_row("center_corner_norm", (item.raw.center_corner_norm for item in pair_metrics), (item.cal.center_corner_norm for item in pair_metrics), True)
    add_row("bg_mad_norm", (item.raw.bg_mad_norm for item in pair_metrics), (item.cal.bg_mad_norm for item in pair_metrics), True)
    add_row("finite_fraction", (item.raw.finite_fraction for item in pair_metrics), (item.cal.finite_fraction for item in pair_metrics), False)

    lines = [
        "metric               raw_mean       cal_mean       delta          better",
        "-----------------------------------------------------------------------",
    ]
    for name, raw_mean, cal_mean, delta, better in rows:
        lines.append(f"{name:<20} {raw_mean:>12.6f} {cal_mean:>12.6f} {delta:>12.6f} {better:>8}")
    return lines


def _pair_line(item: PairMetrics) -> str:
    return (
        f"{item.raw_path.name} -> {item.cal_path.name} | "
        f"grad {item.raw.gradient_norm:.6f} -> {item.cal.gradient_norm:.6f} | "
        f"corner {item.raw.center_corner_norm:.6f} -> {item.cal.center_corner_norm:.6f} | "
        f"mad_norm {item.raw.bg_mad_norm:.6f} -> {item.cal.bg_mad_norm:.6f}"
    )


def run(raw_dir: Path, cal_dir: Path, limit: Optional[int], show_pairs: bool) -> int:
    pairs = pair_files(raw_dir, cal_dir, limit=limit)
    if not pairs:
        print("[ERROR] no matched raw/calibrated pairs found")
        return 1

    pair_metrics: List[PairMetrics] = []
    for raw_path, cal_path in pairs:
        raw_metrics = compute_metrics(raw_path)
        cal_metrics = compute_metrics(cal_path)
        pair_metrics.append(
            PairMetrics(
                raw_path=raw_path,
                cal_path=cal_path,
                raw=raw_metrics,
                cal=cal_metrics,
            )
        )

    print(f"[INFO] raw dir : {raw_dir}")
    print(f"[INFO] cal dir : {cal_dir}")
    print(f"[INFO] matched pairs: {len(pair_metrics)}")
    print("")
    for line in _summary_table(pair_metrics):
        print(line)

    if show_pairs:
        print("")
        print("per_pair")
        print("--------")
        for item in pair_metrics:
            print(_pair_line(item))

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare raw vs calibrated image quality metrics")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory containing raw light frames")
    parser.add_argument("--cal-dir", type=Path, required=True, help="Directory containing calibrated FITS outputs")
    parser.add_argument("--limit", type=int, default=None, help="Maximum matched pairs to analyze")
    parser.add_argument("--show-pairs", action="store_true", help="Print per-pair metrics")
    args = parser.parse_args()

    return run(args.raw_dir.resolve(), args.cal_dir.resolve(), args.limit, args.show_pairs)


if __name__ == "__main__":
    raise SystemExit(main())
