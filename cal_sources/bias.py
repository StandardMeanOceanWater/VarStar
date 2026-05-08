from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple


Array = np.ndarray


def build_master_bias_chain(
    *,
    cfg: dict,
    session: dict,
    paths_ref: dict,
    tz_offset: int,
    list_image_files: Callable[[Optional[Path]], List[Path]],
    ensure_master_root: Callable[[dict], Path],
    frame_type_of: Callable[[Path], str],
    infer_calibration_iso: Callable[[Sequence[Path], str], Optional[int]],
    load_first_existing: Callable[[Sequence[Path]], Optional[Tuple[Array, object, Path]]],
    master_bias_paths: Callable[[Path, str, str, Optional[int]], List[Path]],
    create_master: Callable[..., Array],
    build_master_header: Callable[[str, str, str, Optional[int], Optional[float]], object],
    write_master: Callable[[Path, Array, object], None],
    info: Callable[[str], None],
) -> Tuple[Optional[Array], Optional[object], Optional[int]]:
    bias_files = list_image_files(paths_ref["bias_dir"])
    if not bias_files:
        return None, None, None

    master_root = ensure_master_root(cfg)
    frame_type = frame_type_of(bias_files[0])
    iso_value = infer_calibration_iso(bias_files, str(session["date"]))
    existing = load_first_existing(master_bias_paths(master_root, str(session["date"]), frame_type, iso_value))
    if existing is not None:
        data, header, path = existing
        info(f"loaded master bias: {path.name}")
        header_iso = int(header["ISOSPEED"]) if "ISOSPEED" in header else iso_value
        return data.astype(np.float32), header, header_iso

    master_bias = create_master(
        bias_files,
        "Master Bias Raw",
        tz_offset_hours=tz_offset,
    ).astype(np.float32)
    header = build_master_header("bias_raw", str(session["date"]), frame_type, iso_value, None)
    out_path = master_bias_paths(master_root, str(session["date"]), frame_type, iso_value)[0]
    write_master(out_path, master_bias, header)
    info(f"saved master bias: {out_path.name}")
    return master_bias, header, iso_value
