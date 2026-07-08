"""Field-level helpers for center, metadata, and cache keys."""
from __future__ import annotations

from collections import defaultdict as _defaultdict
from pathlib import Path

from astropy.io import fits
from astropy.wcs import WCS


class _FrameCompCache:
    """同視野多目標共用的比較星測光快取。
    第一顆目標測完後存入，後續目標直接讀取，避免重複孔徑測光。
    Key: (frame_stem, ra_rounded_5dp, dec_rounded_5dp, radius_rounded_3dp)
    半徑入 key：不同目標的自動孔徑可能不同，跨孔徑共用測光值會偏置零點。
    """
    def __init__(self):
        self._data: dict = {}
        self._stats: dict = {}

    def _key(self, frame_stem: str, ra: float, dec: float, radius: float) -> tuple:
        return (
            frame_stem, round(float(ra), 5), round(float(dec), 5),
            round(float(radius), 3),
        )

    def _bucket(self, label: str | None) -> dict:
        _label = str(label or "global")
        if _label not in self._stats:
            self._stats[_label] = {"hits": 0, "misses": 0, "sets": 0}
        return self._stats[_label]

    def get(self, frame_stem: str, ra: float, dec: float, radius: float,
            label: str | None = None):
        _value = self._data.get(self._key(frame_stem, ra, dec, radius))
        _bucket = self._bucket(label)
        if _value is None:
            _bucket["misses"] += 1
        else:
            _bucket["hits"] += 1
        return _value

    def set(self, frame_stem: str, ra: float, dec: float, radius: float,
            result: dict, label: str | None = None):
        self._data[self._key(frame_stem, ra, dec, radius)] = result
        self._bucket(label)["sets"] += 1

    def stats(self, label: str | None = None) -> dict:
        if label is None:
            _total = {"hits": 0, "misses": 0, "sets": 0}
            for _bucket in self._stats.values():
                for _k in _total:
                    _total[_k] += int(_bucket.get(_k, 0))
            return _total
        return dict(self._bucket(label))

    def __len__(self):
        return len(self._data)




def _field_center_from_wcs_fits(wcs_fits_path: Path) -> "tuple[float, float]":
    with fits.open(wcs_fits_path) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data
        if data is None or len(data.shape) < 2:
            raise RuntimeError(f"Cannot infer image center from FITS: {wcs_fits_path}")
        h, w = data.shape[-2], data.shape[-1]
        wcs_obj = WCS(hdr)

    x_center = (float(w) - 1.0) / 2.0
    y_center = (float(h) - 1.0) / 2.0
    ra_deg, dec_deg = wcs_obj.all_pix2world([[x_center, y_center]], 0)[0]
    return float(ra_deg), float(dec_deg)


def _compute_field_key(yaml_cfg, target, date, ch0, split_subdir):
    try:
        from phot_config import cfg_from_yaml

        _targets_cfg = yaml_cfg.get("targets", {})
        _tgt_cfg = _targets_cfg.get(target, {})
        _group = str(_tgt_cfg.get("group", target))
        _cfg_tmp = cfg_from_yaml(
            yaml_cfg, target, date, channel=ch0, split_subdir=split_subdir
        )
        _ff = sorted(_cfg_tmp.wcs_dir.glob(f"*_{ch0}.fits"))
        if not _ff:
            return None
        _summary = [f.name for f in _ff[:2]]
        if len(_ff) > 2:
            _summary.append(_ff[-1].name)
        return (
            str(date),
            _group,
            str(split_subdir),
            str(ch0).upper(),
            len(_ff),
            *_summary,
        )
    except Exception:
        return None


def _build_shared_field_caches(yaml_cfg, targets_list, ch0, split_subdir):
    from collections import defaultdict as _defaultdict

    _field_groups: "dict[tuple, list]" = _defaultdict(list)
    for _tgt, _dt in targets_list:
        _fk = _compute_field_key(yaml_cfg, _tgt, _dt, ch0, split_subdir)
        if _fk:
            _field_groups[_fk].append((_tgt, _dt))
    _field_caches: "dict[tuple, _FrameCompCache]" = {
        _fk: _FrameCompCache()
        for _fk, _grp in _field_groups.items()
        if len(_grp) > 1
    }
    return _field_groups, _field_caches


def _resolve_active_field_cache(
    yaml_cfg, active_target, active_date, channels, split_subdir, field_caches,
):
    _active_field_key = _compute_field_key(
        yaml_cfg, active_target, active_date, channels[0], split_subdir
    )
    return field_caches.get(_active_field_key) if _active_field_key else None



__all__ = [
    "_FrameCompCache",
    "_build_shared_field_caches",
    "_compute_field_key",
    "_field_center_from_wcs_fits",
    "_resolve_active_field_cache",
]
