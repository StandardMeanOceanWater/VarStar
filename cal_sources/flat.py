from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .superflat import candidate_superflat_paths, validate_superflat_array


Array = np.ndarray


def _load_superflat_fallback(
    *,
    master_root: Path,
    session_date: str,
    frame_type: str,
    groups: Sequence[str],
    load_first_existing: Callable[[Sequence[Path]], Optional[tuple]],
    warn: Callable[[str], None],
    info: Callable[[str], None],
) -> Optional[Array]:
    for candidate in candidate_superflat_paths(master_root, session_date, frame_type, groups):
        loaded = load_first_existing([candidate])
        if loaded is None:
            continue
        data, header, path = loaded
        image_type = str(header.get("IMAGETYP", "")).strip().upper()
        if image_type != "SUPERFLAT" and not path.name.lower().startswith("superflat_"):
            warn(f"rejected superflat fallback (tag mismatch): {path.name}")
            continue
        ok, reason = validate_superflat_array(data, min_finite_ratio=0.80)
        if not ok:
            warn(f"rejected superflat fallback (quality gate): {path.name} {reason}")
            continue
        info(f"loaded superflat fallback: {path.name}")
        return data.astype(np.float32)
    return None


def build_flatnorm_map_chain(
    *,
    cfg: dict,
    session: dict,
    paths_ref: dict,
    tz_offset: int,
    master_bias: Optional[Array],
    dark_rate: Optional[Array],
    bad_pixel_floor: float,
    use_median_normalization: bool,
    session_groups: Callable[[dict, dict], List[str]],
    session_light_frame_types: Callable[[dict, dict], List[str]],
    list_image_files: Callable[[Optional[Path]], List[Path]],
    frame_type_of: Callable[[Path], str],
    ensure_master_root: Callable[[dict], Path],
    master_root: Callable[[dict], Path],
    master_flat_product_paths: Callable[[Path, str, str, bool], List[Path]],
    load_first_existing: Callable[[Sequence[Path]], Optional[tuple]],
    create_master: Callable[..., Array],
    infer_common_exposure: Callable[[Sequence[Path], str], Optional[float]],
    compute_flat_pure: Callable[[Array, Optional[Array], Optional[Array], Optional[float]], Array],
    normalize_flat_pure: Callable[[Array, float, bool], Array],
    build_master_header: Callable[[str, str, str, Optional[int], Optional[float]], object],
    write_master: Callable[[Path, Array, object], None],
    warn: Callable[[str], None],
    info: Callable[[str], None],
) -> Dict[str, Array]:
    flat_map: Dict[str, Array] = {}
    flat_dirs_by_format = dict(paths_ref.get("flat_dirs_by_format", {}))
    writable_master_root = ensure_master_root(cfg)
    superflat_master_root = master_root(cfg)
    groups = session_groups(cfg, session)
    required_frame_types = session_light_frame_types(cfg, session)
    if not required_frame_types:
        required_frame_types = sorted(flat_dirs_by_format)

    for frame_type in required_frame_types:
        flat_dir = flat_dirs_by_format.get(frame_type)
        if flat_dir is not None:
            flat_files = [path for path in list_image_files(flat_dir) if frame_type_of(path) == frame_type]
            if flat_files:
                try:
                    master_flat_raw = create_master(
                        flat_files,
                        f"Master Flat Raw ({frame_type.upper()})",
                        tz_offset_hours=tz_offset,
                    ).astype(np.float32)
                    t_flat = infer_common_exposure(flat_files, str(session["date"]))
                    flat_pure = compute_flat_pure(master_flat_raw, master_bias, dark_rate, t_flat)
                    master_flatnorm = normalize_flat_pure(
                        flat_pure,
                        bad_pixel_floor,
                        use_median_normalization=use_median_normalization,
                    )
                    header = build_master_header(
                        "flatnorm" if use_median_normalization else "flatdirect",
                        str(paths_ref["flat_date"]),
                        frame_type,
                        None,
                        t_flat,
                    )
                    out_path = master_flat_product_paths(
                        writable_master_root,
                        str(paths_ref["flat_date"]),
                        frame_type,
                        use_median_normalization,
                    )[0]
                    write_master(out_path, master_flatnorm, header)
                    info(f"saved master flatnorm from normal flats: {out_path.name}")
                    flat_map[frame_type] = master_flatnorm.astype(np.float32)
                    continue
                except Exception as exc:
                    warn(f"normal flat build failed for {frame_type}: {exc}")
            else:
                warn(f"normal flat frames missing for {frame_type}: {flat_dir}")

        superflat = _load_superflat_fallback(
            master_root=superflat_master_root,
            session_date=str(session["date"]),
            frame_type=frame_type,
            groups=groups,
            load_first_existing=load_first_existing,
            warn=warn,
            info=info,
        )
        if superflat is not None:
            flat_map[frame_type] = superflat.astype(np.float32)
            continue

        existing = load_first_existing(
            master_flat_product_paths(
                writable_master_root,
                str(paths_ref["flat_date"]),
                frame_type,
                use_median_normalization,
            )
        )
        if existing is not None:
            data, _, path = existing
            info(f"loaded existing master flat fallback: {path.name}")
            flat_map[frame_type] = data.astype(np.float32)
            continue

        warn(f"no flat source available for frame type {frame_type}")

    return flat_map
