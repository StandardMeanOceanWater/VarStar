from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple


Array = np.ndarray


def build_master_dark_signal_chain(
    *,
    cfg: dict,
    session: dict,
    paths_ref: dict,
    tz_offset: int,
    master_bias: Optional[Array],
    list_image_files: Callable[[Optional[Path]], List[Path]],
    ensure_master_root: Callable[[dict], Path],
    frame_type_of: Callable[[Path], str],
    infer_calibration_iso: Callable[[Sequence[Path], str], Optional[int]],
    load_first_existing: Callable[[Sequence[Path]], Optional[Tuple[Array, object, Path]]],
    master_dark_paths: Callable[[Path, str, str, Optional[int], Optional[float]], List[Path]],
    extract_fractional_exposure: Callable[[object], Optional[float]],
    create_master: Callable[..., Array],
    infer_common_exposure: Callable[[Sequence[Path], str], Optional[float]],
    build_master_header: Callable[[str, str, str, Optional[int], Optional[float], Optional[dict]], object],
    write_master: Callable[[Path, Array, object], None],
    info: Callable[[str], None],
) -> Tuple[Optional[Array], Optional[object], Optional[float], Optional[int]]:
    dark_files = list_image_files(paths_ref["dark_dir"])
    if not dark_files:
        return None, None, None, None

    master_root = ensure_master_root(cfg)
    frame_type = frame_type_of(dark_files[0])
    iso_value = infer_calibration_iso(dark_files, str(session["date"]))
    dark_temp_c = paths_ref.get("dark_temp_c")
    existing = load_first_existing(master_dark_paths(master_root, str(session["date"]), frame_type, iso_value, dark_temp_c))
    if existing is not None:
        data, header, path = existing
        info(f"loaded master dark signal: {path.name}")
        exposure_s = extract_fractional_exposure(header.get("EXPTIME"))
        header_iso = int(header["ISOSPEED"]) if "ISOSPEED" in header else iso_value
        return data.astype(np.float32), header, exposure_s, header_iso

    dark_raw = create_master(
        dark_files,
        "Master Dark Raw",
        tz_offset_hours=tz_offset,
    ).astype(np.float32)
    t_dark = infer_common_exposure(dark_files, str(session["date"]))
    master_dark_signal = dark_raw.copy()
    if master_bias is not None:
        master_dark_signal -= master_bias.astype(np.float32)
    header = build_master_header(
        "dark_signal",
        str(session["date"]),
        frame_type,
        iso_value,
        t_dark,
        extra={"DARKTEMP": float(dark_temp_c) if dark_temp_c is not None else "UNKNOWN"},
    )
    out_path = master_dark_paths(master_root, str(session["date"]), frame_type, iso_value, dark_temp_c)[0]
    write_master(out_path, master_dark_signal, header)
    info(f"saved master dark signal: {out_path.name}")
    return master_dark_signal, header, t_dark, iso_value


def build_dark_rate_chain(
    *,
    cfg: dict,
    session: dict,
    dark_signal: Optional[Array],
    dark_header: Optional[object],
    t_dark: Optional[float],
    dark_iso: Optional[int],
    dark_temp_c: Optional[float],
    ensure_master_root: Callable[[dict], Path],
    load_first_existing: Callable[[Sequence[Path]], Optional[Tuple[Array, object, Path]]],
    dark_rate_paths: Callable[[Path, str, str, Optional[int], Optional[float]], List[Path]],
    compute_dark_rate: Callable[[Array, Optional[float]], Optional[Array]],
    build_master_header: Callable[[str, str, str, Optional[int], Optional[float], Optional[dict]], object],
    write_master: Callable[[Path, Array, object], None],
    info: Callable[[str], None],
) -> Optional[Array]:
    if dark_signal is None:
        return None

    master_root = ensure_master_root(cfg)
    frame_type = "cr2"
    if dark_header is not None and "FRMTYPE" in dark_header:
        frame_type = str(dark_header["FRMTYPE"]).strip().lower()
    existing = load_first_existing(dark_rate_paths(master_root, str(session["date"]), frame_type, dark_iso, dark_temp_c))
    if existing is not None:
        data, _, path = existing
        info(f"loaded dark rate: {path.name}")
        return data.astype(np.float32)

    dark_rate = compute_dark_rate(dark_signal, t_dark)
    if dark_rate is None:
        return None
    header = build_master_header(
        "dark_rate",
        str(session["date"]),
        frame_type,
        dark_iso,
        t_dark,
        extra={"DARKTEMP": float(dark_temp_c) if dark_temp_c is not None else "UNKNOWN"},
    )
    out_path = dark_rate_paths(master_root, str(session["date"]), frame_type, dark_iso, dark_temp_c)[0]
    write_master(out_path, dark_rate, header)
    info(f"saved dark rate: {out_path.name}")
    return dark_rate
