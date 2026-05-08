"""Catalog loaders and catalog query helpers."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS

from phot_sources.catalog_backend import (
    _pick_col,
    fetch_aavso_vsp_api,
    fetch_apass_cone,
    fetch_gaia_dr3_cone,
    fetch_tycho2_cone,
    read_aavso_seq_csv,
)
from phot_sources.core import (
    aperture_photometry,
    compute_annulus_radii,
    in_bounds,
    is_saturated,
    m_inst_from_flux,
    radec_to_pixel,
)
from phot_sources.field import _field_center_from_wcs_fits
from phot_sources.io_paths import get_field_catalog_path


_phot_logger = logging.getLogger("photometry")


def select_check_star(target_radec_deg, comp_refs, catalog_df, cfg_obj):
    if cfg_obj.check_star_radec_deg is not None:
        ra_c, dec_c = cfg_obj.check_star_radec_deg
        m_cat = None
        if catalog_df is not None and len(catalog_df) > 0 and "m_cat" in catalog_df.columns:
            sc_cat = SkyCoord(catalog_df["ra_deg"].values * u.deg,
                              catalog_df["dec_deg"].values * u.deg)
            sc_k   = SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg)
            sep    = sc_cat.separation(sc_k).arcsec
            idx    = int(np.argmin(sep))
            if np.isfinite(sep[idx]) and sep[idx] <= cfg_obj.apass_match_arcsec:
                m_cat = float(catalog_df.iloc[idx]["m_cat"])
        return (float(ra_c), float(dec_c), m_cat)

    if catalog_df is None or len(catalog_df) == 0:
        print("[WARN] No catalog available for auto check-star selection.")
        return None

    cand = catalog_df.copy()
    if "max_pix" in cand.columns and cfg_obj.saturation_threshold is not None:
        cand = cand[~cand["max_pix"].apply(lambda v: is_saturated(v, cfg_obj.saturation_threshold))]
    if comp_refs:
        comp_coords = SkyCoord(
            [r[0] for r in comp_refs] * u.deg,
            [r[1] for r in comp_refs] * u.deg,
        )
        cand_coords = SkyCoord(cand["ra_deg"].values * u.deg,
                               cand["dec_deg"].values * u.deg)
        _, sep2d, _ = cand_coords.match_to_catalog_sky(comp_coords)
        cand = cand[sep2d.arcsec > cfg_obj.apass_match_arcsec].copy()

    if len(cand) == 0:
        print("[WARN] Auto check-star selection failed: no safe candidates.")
        return None

    target      = SkyCoord(ra=target_radec_deg[0] * u.deg, dec=target_radec_deg[1] * u.deg)
    cand_coords = SkyCoord(cand["ra_deg"].values * u.deg, cand["dec_deg"].values * u.deg)
    idx = int(np.nanargmin(target.separation(cand_coords).arcsec))
    row = cand.iloc[idx]
    print("[INFO] check_star_radec_deg not set; auto-selected nearest non-comp candidate.")
    return (
        float(row["ra_deg"]),
        float(row["dec_deg"]),
        float(row["m_cat"]) if "m_cat" in row and np.isfinite(row["m_cat"]) else None,
    )


def _load_or_build_unified_catalog(
    cfg,
    ra_center_deg: float,
    dec_center_deg: float,
    field_cat_path: Path,
) -> "tuple[pd.DataFrame, list[str]]":
    """
    統一視場星表：查詢所有星表來源一次，合併多波段欄位。

    Returns (unified_df, sources_used).
    unified_df 欄位：
        ra_deg, dec_deg,
        V_mag, V_err, source_V,   — Johnson V: AAVSO > APASS > Tycho VT→V
        B_mag, B_err, source_B,   — Johnson B: APASS B > Tycho BT→B
        R_mag, R_err, source_R,   — Cousins Rc: Gaia RP→Rc ONLY
        BT, VT, Gmag, BPmag, RPmag  — 原始診斷欄位
    """
    _sources_used: "list[str]" = []
    _phot_logger.info(
        "[unified catalog] request field_center=(%.6f, %.6f) path=%s",
        float(ra_center_deg), float(dec_center_deg), field_cat_path,
    )

    if field_cat_path.exists():
        _candidate_df = pd.read_csv(field_cat_path)
        _stored_ra = _candidate_df.get("field_center_ra_deg")
        _stored_dec = _candidate_df.get("field_center_dec_deg")
        _reuse_existing = False
        if _stored_ra is not None and _stored_dec is not None and len(_candidate_df) > 0:
            _stored_center = SkyCoord(
                ra=float(_stored_ra.iloc[0]) * u.deg,
                dec=float(_stored_dec.iloc[0]) * u.deg,
            )
            _requested_center = SkyCoord(
                ra=float(ra_center_deg) * u.deg,
                dec=float(dec_center_deg) * u.deg,
            )
            _center_sep_arcsec = float(_stored_center.separation(_requested_center).arcsec)
            _reuse_existing = _center_sep_arcsec <= 10.0
            if not _reuse_existing:
                print(
                    "  [unified catalog] existing cache center mismatch; "
                    f"rebuild required ({_center_sep_arcsec:.2f} arcsec)"
                )
                _phot_logger.info(
                    "[unified catalog] rebuild reason=center_mismatch path=%s sep_arcsec=%.2f",
                    field_cat_path, _center_sep_arcsec,
                )
        if not _reuse_existing:
            print("  [unified catalog] existing cache missing valid field center; rebuilding")
            _phot_logger.info("[unified catalog] action=rebuild existing_cache_invalid=1 path=%s", field_cat_path)
        else:
            for _sc in ("source_V", "source_B", "source_R"):
                if _sc in _candidate_df.columns:
                    _sources_used.extend(_candidate_df[_sc].dropna().unique().tolist())
            _sources_used = sorted(set(_sources_used))
            print(f"  [unified catalog] read: {field_cat_path.name}")
            print(f"  [unified catalog] {len(_candidate_df)} stars, sources: {'+'.join(_sources_used)}")
            _phot_logger.info("[unified catalog] action=read rows=%d path=%s", len(_candidate_df), field_cat_path)
            return _candidate_df, _sources_used
    else:
        _phot_logger.info("[unified catalog] action=rebuild existing_cache_missing=1")

    if field_cat_path.exists():
        _phot_logger.info("[unified catalog] action=rebuild path=%s", field_cat_path)

    # ── 查詢各星表（各查一次）─────────────────────────────────────────────────
    # 每個 source 以統一格式收集：ra_deg, dec_deg + 各波段欄位
    _source_rows: "list[dict]" = []

    _radius_arcmin = cfg.apass_radius_deg * 60.0

    # --- AAVSO: 只有 V ---
    _has_aavso = False
    for _cat_name in cfg.catalog_priority:
        if str(_cat_name).upper() == "AAVSO":
            seq_df = None
            if cfg.aavso_seq_csv is not None and Path(cfg.aavso_seq_csv).exists():
                seq_df = read_aavso_seq_csv(Path(cfg.aavso_seq_csv))
            elif cfg.aavso_star_name and cfg.aavso_use_api:
                try:
                    seq_df = fetch_aavso_vsp_api(
                        cfg.aavso_star_name, cfg.aavso_fov_arcmin, cfg.aavso_maglimit,
                        ra_deg=ra_center_deg, dec_deg=dec_center_deg,
                    )
                except Exception as exc:
                    print(f"  [WARN] AAVSO API query failed: {exc}")
            if seq_df is not None and len(seq_df) > 0:
                for _, r in seq_df.iterrows():
                    _source_rows.append({
                        "ra_deg": float(r["ra_deg"]), "dec_deg": float(r["dec_deg"]),
                        "V_mag": float(r["m_cat"]),
                        "V_err": float(r.get("m_err", np.nan)) if "m_err" in r.index else np.nan,
                        "source_V": "AAVSO",
                    })
                _has_aavso = True
                _sources_used.append("AAVSO")
                print(f"  [unified] AAVSO: {len(seq_df)} stars (V only)")
            break

    # --- APASS: V + B ---
    _has_apass = False
    for _cat_name in cfg.catalog_priority:
        if str(_cat_name).upper() == "APASS":
            apass_raw = fetch_apass_cone(
                ra_center_deg, dec_center_deg,
                radius_deg=cfg.apass_radius_deg, maxrec=cfg.apass_maxrec,
            )
            if len(apass_raw) > 0:
                _ra_col = _pick_col(apass_raw, ["ra", "raj2000", "ra_deg", "ra_icrs"])
                _dec_col = _pick_col(apass_raw, ["dec", "dej2000", "dec_deg", "dec_icrs"])
                # V band columns
                _v_col = next((c for c in ["vmag", "mag_v", "v", "v_mag"] if c in apass_raw.columns), None)
                _ve_col = next((c for c in ["e_vmag", "err_mag_v", "v_err"] if c in apass_raw.columns), None)
                # B band columns
                _b_col = next((c for c in ["bmag", "mag_b", "b", "b_mag"] if c in apass_raw.columns), None)
                _be_col = next((c for c in ["e_bmag", "err_mag_b", "b_err"] if c in apass_raw.columns), None)
                n_v, n_b = 0, 0
                for _, r in apass_raw.iterrows():
                    row = {"ra_deg": float(r[_ra_col]), "dec_deg": float(r[_dec_col])}
                    has_any = False
                    if _v_col and np.isfinite(float(r.get(_v_col, np.nan))):
                        row["V_mag"] = float(r[_v_col])
                        row["V_err"] = float(r[_ve_col]) if _ve_col and np.isfinite(float(r.get(_ve_col, np.nan))) else np.nan
                        row["source_V"] = "APASS"
                        has_any = True
                        n_v += 1
                    if _b_col and np.isfinite(float(r.get(_b_col, np.nan))):
                        row["B_mag"] = float(r[_b_col])
                        row["B_err"] = float(r[_be_col]) if _be_col and np.isfinite(float(r.get(_be_col, np.nan))) else np.nan
                        row["source_B"] = "APASS"
                        has_any = True
                        n_b += 1
                    if has_any:
                        _source_rows.append(row)
                _has_apass = True
                _sources_used.append("APASS")
                print(f"  [unified] APASS: {len(apass_raw)} stars (V={n_v}, B={n_b})")
            break

    # --- Tycho-2: V (VT→V) + B (BT→B) ---
    _has_tycho = False
    for _cat_name in cfg.catalog_priority:
        if str(_cat_name).upper() == "TYCHO2":
            tycho_raw = fetch_tycho2_cone(
                ra_center_deg, dec_center_deg,
                radius_arcmin=_radius_arcmin,
                mag_min=3.0, mag_max=16.0,
            )
            if len(tycho_raw) > 0:
                n_v, n_b = 0, 0
                for _, r in tycho_raw.iterrows():
                    row = {"ra_deg": float(r["ra_deg"]), "dec_deg": float(r["dec_deg"])}
                    # V from Tycho (already converted VT→V in fetch_tycho2_cone)
                    _vm = float(r.get("vmag", np.nan))
                    if np.isfinite(_vm):
                        row["V_mag"] = _vm
                        row["V_err"] = float(r.get("e_vmag", np.nan))
                        row["source_V"] = "Tycho2"
                        n_v += 1
                    # B from BT to Johnson B conversion.
                    _bt = float(r.get("BT", np.nan))
                    _vt = float(r.get("VT", np.nan))
                    if np.isfinite(_bt) and np.isfinite(_vt):
                        _b_mag = _bt - 0.240 * (_bt - _vt)
                        _e_bt = float(r.get("e_BT", r.get("e_BTmag", np.nan)))
                        _e_vt = float(r.get("e_VT", r.get("e_VTmag", np.nan)))
                        if not np.isfinite(_e_bt):
                            _e_bt = 0.05
                        if not np.isfinite(_e_vt):
                            _e_vt = 0.05
                        _e_b = float(np.sqrt(_e_bt**2 * (1 - 0.240)**2 + _e_vt**2 * 0.240**2))
                        row["B_mag"] = _b_mag
                        row["B_err"] = _e_b
                        row["source_B"] = "Tycho2"
                        row["e_BT"] = _e_bt
                        row["e_VT"] = _e_vt
                        n_b += 1
                    # 保留原始欄位
                    if np.isfinite(_bt):
                        row["BT"] = _bt
                    if np.isfinite(_vt):
                        row["VT"] = _vt
                    _source_rows.append(row)
                _has_tycho = True
                _sources_used.append("Tycho2")
                print(f"  [unified] Tycho-2: {len(tycho_raw)} stars (V={n_v}, B={n_b})")
            break

    # --- Gaia DR3: R (RP→Rc) ONLY ---
    _has_gaia = False
    for _cat_name in cfg.catalog_priority:
        if str(_cat_name).upper() in ("GAIA", "GAIADR3", "GAIA_DR3"):
            gaia_raw = fetch_gaia_dr3_cone(
                ra_center_deg, dec_center_deg,
                radius_arcmin=_radius_arcmin,
                mag_min=3.0, mag_max=16.0,
                channel="R",   # 只取 RP→Rc
            )
            if len(gaia_raw) > 0:
                for _, r in gaia_raw.iterrows():
                    _rc = float(r.get("vmag", np.nan))  # fetch_gaia_dr3_cone 回傳 vmag=Rc
                    if not np.isfinite(_rc):
                        continue
                    row = {
                        "ra_deg": float(r["ra_deg"]), "dec_deg": float(r["dec_deg"]),
                        "R_mag": _rc,
                        "R_err": float(r.get("e_vmag", np.nan)),
                        "source_R": "GaiaDR3",
                    }
                    for _extra in ("Gmag", "BPmag", "RPmag"):
                        if _extra in r.index and np.isfinite(float(r.get(_extra, np.nan))):
                            row[_extra] = float(r[_extra])
                    _source_rows.append(row)
                _has_gaia = True
                _sources_used.append("GaiaDR3")
                print(f"  [unified] Gaia DR3: {len(gaia_raw)} stars (R=Rc only)")
            break

    _sources_used = sorted(set(_sources_used))

    if not _source_rows:
        print("  [unified catalog] 所有星表查詢無結果")
        return pd.DataFrame(), _sources_used

    # ── 合併：同位置 (<3") 的不同源合併多波段欄位 ──────────────────────────────
    all_df = pd.DataFrame(_source_rows)
    # 確保所有欄位存在
    for _col in ("V_mag", "V_err", "source_V", "B_mag", "B_err", "source_B",
                 "R_mag", "R_err", "source_R", "BT", "VT", "e_BT", "e_VT",
                 "Gmag", "BPmag", "RPmag"):
        if _col not in all_df.columns:
            all_df[_col] = np.nan if not _col.startswith("source") else None

    from astropy.coordinates import SkyCoord as _SC
    coords = _SC(all_df["ra_deg"].values * u.deg, all_df["dec_deg"].values * u.deg)

    # 用 union-find 聚類 <3" 的星
    n = len(all_df)
    parent = list(range(n))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    # 建立鄰近關係（O(n²) 但 n 通常 <2000）
    for i in range(n):
        seps = coords[i].separation(coords[i+1:]).arcsec
        for j_off, sep in enumerate(seps):
            if sep < 3.0:
                _union(i, i + 1 + j_off)

    # 按群組合併
    from collections import defaultdict as _defaultdict
    groups = _defaultdict(list)
    for i in range(n):
        groups[_find(i)].append(i)

    # V 優先級：AAVSO > APASS > Tycho2（照 catalog_priority 順序）
    _v_priority = {"AAVSO": 0, "APASS": 1, "Tycho2": 2}
    _b_priority = {"APASS": 0, "Tycho2": 1}

    unified_rows = []
    for _root, indices in groups.items():
        # 取群組的平均位置（以第一個為主）
        _ra = float(all_df.iloc[indices[0]]["ra_deg"])
        _dec = float(all_df.iloc[indices[0]]["dec_deg"])
        row = {"ra_deg": _ra, "dec_deg": _dec}

        # 從群組中挑最佳 V, B, R
        best_v, best_b, best_r = None, None, None
        best_v_pri, best_b_pri = 999, 999

        for idx in indices:
            r = all_df.iloc[idx]

            # V
            _sv = r.get("source_V")
            _vm = r.get("V_mag")
            if _sv and isinstance(_sv, str) and np.isfinite(float(_vm if _vm is not None else np.nan)):
                pri = _v_priority.get(_sv, 99)
                if pri < best_v_pri:
                    best_v_pri = pri
                    best_v = (float(_vm), float(r.get("V_err", np.nan)), _sv)

            # B
            _sb = r.get("source_B")
            _bm = r.get("B_mag")
            if _sb and isinstance(_sb, str) and np.isfinite(float(_bm if _bm is not None else np.nan)):
                pri = _b_priority.get(_sb, 99)
                if pri < best_b_pri:
                    best_b_pri = pri
                    best_b = (float(_bm), float(r.get("B_err", np.nan)), _sb)

            # R (only GaiaDR3)
            _sr = r.get("source_R")
            _rm = r.get("R_mag")
            if _sr and isinstance(_sr, str) and _sr == "GaiaDR3" and np.isfinite(float(_rm if _rm is not None else np.nan)):
                best_r = (float(_rm), float(r.get("R_err", np.nan)), "GaiaDR3")

            # 保留原始診斷欄位
            for _diag in ("BT", "VT", "e_BT", "e_VT", "Gmag", "BPmag", "RPmag"):
                _dv = r.get(_diag)
                if _dv is not None and np.isfinite(float(_dv)):
                    row.setdefault(_diag, float(_dv))

        if best_v:
            row["V_mag"], row["V_err"], row["source_V"] = best_v
        if best_b:
            row["B_mag"], row["B_err"], row["source_B"] = best_b
        if best_r:
            row["R_mag"], row["R_err"], row["source_R"] = best_r

        # 至少有一個波段才保留
        if best_v or best_b or best_r:
            unified_rows.append(row)

    unified_df = pd.DataFrame(unified_rows)

    # 確保欄位順序與完整性
    _all_cols = ["ra_deg", "dec_deg",
                 "V_mag", "V_err", "source_V",
                 "B_mag", "B_err", "source_B",
                 "R_mag", "R_err", "source_R",
                 "BT", "VT", "e_BT", "e_VT", "Gmag", "BPmag", "RPmag",
                 "field_center_ra_deg", "field_center_dec_deg"]
    for _c in _all_cols:
        if _c not in unified_df.columns:
            unified_df[_c] = np.nan if not _c.startswith("source") else None
    unified_df["field_center_ra_deg"] = float(ra_center_deg)
    unified_df["field_center_dec_deg"] = float(dec_center_deg)
    unified_df = unified_df[_all_cols]

    # 儲存快取
    field_cat_path.parent.mkdir(parents=True, exist_ok=True)
    unified_df.to_csv(field_cat_path, index=False, encoding="utf-8-sig")
    print(f"  [unified catalog] saved: {field_cat_path.name} ({len(unified_df)} rows)")
    _phot_logger.info(
        "[unified catalog] action=save rows=%d field_center=(%.6f, %.6f)",
        len(unified_df), float(ra_center_deg), float(dec_center_deg),
    )
    print(f"  [unified catalog] V={int(unified_df['V_mag'].notna().sum())}  "
          f"B={int(unified_df['B_mag'].notna().sum())}  "
          f"R={int(unified_df['R_mag'].notna().sum())}  "
          f"sources: {'+'.join(_sources_used)}")

    return unified_df, _sources_used

def auto_select_comps(
    wcs_fits_path: Path,
    target_radec_deg: tuple,
    cfg_obj=None,
    band: str = "V",
    max_detect: int = 500,
    psf_box: int = 25,
    threshold_sigma: float = 5.0,
):
    """
    比較星自動選取主函式（unified catalog 版）。

    每個通道獨立呼叫此函式，band 指定使用 unified catalog 的哪個波段欄位。

    Parameters
    ----------
    band : str
        "V" (G1/G2 通道), "B" (B 通道), "R" (R 通道)

    Returns
    -------
    comp_refs       : list of (ra, dec, m_cat, m_err, weight)
    comp_df_matched : 選入回歸的比較星 DataFrame
    check_star      : (ra, dec, m_cat | None) 或 None
    aavso_matched   : 匹配結果（診斷用）
    apass_matched   : 匹配結果（診斷用）
    active_source   : 使用的星表來源
    vsx_field       : VSX 查詢結果 DataFrame（供額外目標星測光用）
    """
    # Band → unified catalog column mapping
    if cfg_obj is None:
        raise RuntimeError("auto_select_comps requires cfg_obj")

    _band_mag_col = {"V": "V_mag", "B": "B_mag", "R": "R_mag"}
    _band_err_col = {"V": "V_err", "B": "B_err", "R": "R_err"}
    _band_src_col = {"V": "source_V", "B": "source_B", "R": "source_R"}
    mag_col = _band_mag_col.get(band, "V_mag")
    err_col = _band_err_col.get(band, "V_err")
    src_col = _band_src_col.get(band, "source_V")

    with fits.open(wcs_fits_path) as hdul:
        img = hdul[0].data.astype(np.float32)
        hdr = hdul[0].header
        wcs_obj = WCS(hdr)

    ra_t, dec_t = float(target_radec_deg[0]), float(target_radec_deg[1])
    field_ra_deg, field_dec_deg = _field_center_from_wcs_fits(wcs_fits_path)
    epsilon = cfg_obj.plate_scale_arcsec / 2.0    # 防零 ε（arcsec）
    tgt_sc = SkyCoord(ra=ra_t * u.deg, dec=dec_t * u.deg)
    _h, _w = img.shape

    # 孔徑參數：使用 cfg 靜態值（孔徑估算在本函式之後才執行）
    _ap_r = float(cfg_obj.aperture_radius)
    _r_in, _r_out = compute_annulus_radii(_ap_r, cfg_obj.annulus_r_in, cfg_obj.annulus_r_out)
    _margin = int(np.ceil(_r_out + 2))

    def _catalog_direct_phot(cat_df, ra_col, dec_col, mag_col_, err_col_, mag_min, mag_max):
        """從星表座標直接做孔徑測光，回傳通過篩選的 DataFrame。
        排除條件：(a) mag 範圍外；(b) 選取圓外；(c) 影像邊界外；
                  (d) 飽和；(e) 測光失敗。
        """
        rows = []
        n_mag_out, n_circle, n_bounds, n_sat, n_phot = 0, 0, 0, 0, 0
        for _, row in cat_df.iterrows():
            try:
                ra_c  = float(row[ra_col])
                dec_c = float(row[dec_col])
                m_c   = float(row[mag_col_])
                m_e   = float(row[err_col_]) if (err_col_ and err_col_ in row.index
                                                and np.isfinite(row[err_col_])) else np.nan
            except (KeyError, TypeError, ValueError):
                continue
            if not (mag_min <= m_c <= mag_max):
                n_mag_out += 1
                continue
            xc, yc = radec_to_pixel(wcs_obj, ra_c, dec_c)
            # 選取圓：影像中心為圓心，半徑 = 短邊 / 2
            _cx, _cy = _w / 2.0, _h / 2.0
            _sel_r   = float(min(_h, _w)) / 2.0
            if np.hypot(xc - _cx, yc - _cy) > _sel_r:
                n_circle += 1
                continue
            if not in_bounds(img, xc, yc, margin=_margin):
                n_bounds += 1
                continue
            phot = aperture_photometry(img, xc, yc, _ap_r, _r_in, _r_out)
            if phot.get("ok") != 1 or not np.isfinite(phot.get("flux_net", np.nan)):
                n_phot += 1
                continue
            if is_saturated(phot.get("max_pix", np.nan), cfg_obj.saturation_threshold):
                n_sat += 1
                continue
            m_inst = m_inst_from_flux(phot["flux_net"])
            if not np.isfinite(m_inst):
                continue
            rows.append({
                "ra_deg": ra_c, "dec_deg": dec_c,
                "m_cat": m_c, "m_err": m_e,
                "m_inst": m_inst, "m_inst_matched": m_inst,
            })
        print(f"  [直接測光] band={band}  星表={len(cat_df)}  "
              f"mag範圍外={n_mag_out}({mag_min:.1f}-{mag_max:.1f})  "
              f"選取圓外={n_circle}  邊界排除={n_bounds}  "
              f"飽和={n_sat}  測光失敗={n_phot}  通過={len(rows)}")
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # Shared field-level catalog.
    _field_cat_path = get_field_catalog_path(cfg_obj.run_root)

    unified_df, _sources_used = _load_or_build_unified_catalog(
        cfg_obj, field_ra_deg, field_dec_deg, _field_cat_path,
    )

    # 篩選此波段有值的星
    if len(unified_df) == 0 or mag_col not in unified_df.columns:
        raise RuntimeError(
            f"統一星表中無 {band} 波段資料。\n"
            "建議：(1) 刪除 catalog.csv 重建；"
            "(2) 確認 catalog_priority 設定。"
        )

    band_df = unified_df[unified_df[mag_col].notna()].copy()
    # 建立 m_cat / m_err 欄位供 _catalog_direct_phot 使用
    band_df = band_df.rename(columns={mag_col: "m_cat", err_col: "m_err"})
    if "m_err" not in band_df.columns:
        band_df["m_err"] = np.nan

    # 來源統計
    _band_sources = []
    if src_col in unified_df.columns:
        _band_sources = sorted(unified_df.loc[unified_df[mag_col].notna(), src_col].dropna().unique().tolist())
    active_source = "+".join(_band_sources) if _band_sources else "+".join(_sources_used)

    print(f"  [unified] band={band}: {len(band_df)} stars with {mag_col} "
          f"(sources: {active_source})")

    # 強制孔徑測光篩選
    aavso_matched = None
    apass_matched = None
    active_df     = None

    # ── 以目標星在該 band 的實際星等重算 comp mag range ──────────────────
    #   cfg.comp_mag_min/max 是用 vmag_approx(V) 算的，B/R 波段需要修正。
    #   從 unified catalog 查目標星在此 band 的星等，用相同 bright/faint offset 重算。
    #   注意：必須在 VSX 排除之前查，因為目標星本身是變星會被 VSX 踢掉。
    _comp_mag_min_band = cfg_obj.comp_mag_min
    _comp_mag_max_band = cfg_obj.comp_mag_max
    if band != "V" and len(band_df) > 0:
        _tgt_cat = SkyCoord(band_df["ra_deg"].values * u.deg,
                            band_df["dec_deg"].values * u.deg)
        _sep = tgt_sc.separation(_tgt_cat).arcsec
        _nearest_idx = int(np.argmin(_sep))
        if _sep[_nearest_idx] < 10.0:  # 10" 以內才信
            _tgt_band_mag = float(band_df.iloc[_nearest_idx]["m_cat"])
            # 用 cfg 裡的 bright/faint offset（從 vmag 反推）
            _bright_offset = cfg_obj.vmag_approx - cfg_obj.comp_mag_min
            _faint_offset  = cfg_obj.comp_mag_max - cfg_obj.vmag_approx
            _comp_mag_min_band = _tgt_band_mag - _bright_offset
            _comp_mag_max_band = _tgt_band_mag + _faint_offset
            # 再套 yaml 夾鉗
            _yaml_floor = 6.0    # comp_mag_min 硬下限
            _yaml_ceil  = 13.0   # comp_mag_max 硬上限
            _comp_mag_min_band = max(_comp_mag_min_band, _yaml_floor)
            _comp_mag_max_band = min(_comp_mag_max_band, _yaml_ceil)
            print(f"  [mag range] band={band}: 目標星 {band}_mag={_tgt_band_mag:.2f} "
                  f"→ comp range [{_comp_mag_min_band:.1f}, {_comp_mag_max_band:.1f}] "
                  f"(V 基準 [{cfg_obj.comp_mag_min:.1f}, {cfg_obj.comp_mag_max:.1f}])")

    # ── VSX 已知變星排除（比較星不能是變星）──────────────────────────────
    _vsx = pd.DataFrame()
    if len(band_df) > 0:
        try:
            from tools.local_catalog import query_vsx_cone, filter_known_variables
            _vsx = query_vsx_cone(
                field_ra_deg, field_dec_deg, radius_deg=cfg_obj.apass_radius_deg
            )
            if len(_vsx) > 0:
                _n_before_vsx = len(band_df)
                band_df = filter_known_variables(band_df, _vsx, match_arcsec=10.0)
                print(f"  [VSX] 比較星候選：{_n_before_vsx} → {len(band_df)}（排除已知變星）")
        except Exception as _e_vsx:
            print(f"  [VSX] 變星排除跳過：{_e_vsx}")

    if len(band_df) > 0:
        active_df = _catalog_direct_phot(
            band_df, "ra_deg", "dec_deg", "m_cat", "m_err",
            _comp_mag_min_band, _comp_mag_max_band,
        )
        apass_matched = active_df
        aavso_matched = active_df

    if active_df is None or len(active_df) == 0:
        raise RuntimeError(
            f"band={band} 在星表中找不到足夠的比較星。\n"
            "建議：(1) 放寬 comp_mag_range；(2) 增大 apass_radius_deg；"
            "(3) 確認 catalog_priority 設定。"
        )

    # ── 6. 建立 comp_refs（含距離加權 ε）──────────────────────────────────────
    comp_refs = []
    for _, r in active_df.iterrows():
        ra_c  = float(r["ra_deg"])
        dec_c = float(r["dec_deg"])
        m_cat = float(r["m_cat"])
        m_err = float(r["m_err"]) if ("m_err" in r and np.isfinite(r["m_err"])) else None

        sc_c      = SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg)
        d_arcsec  = float(tgt_sc.separation(sc_c).arcsec)
        weight    = 1.0 / (d_arcsec + epsilon) ** 2    # w_i = 1 / (d_i + ε)²
        comp_refs.append((ra_c, dec_c, m_cat, m_err, weight))

    # ── 6b. comp_max 截斷 ──────────────────────────────────────────────────────
    if cfg_obj.comp_max > 0 and len(comp_refs) > cfg_obj.comp_max:
        _vmag_t = float(getattr(cfg_obj, "vmag_approx", np.nan))
        if np.isfinite(_vmag_t):
            comp_refs.sort(key=lambda x: abs(x[2] - _vmag_t))
        else:
            comp_refs.sort(key=lambda x: x[4], reverse=True)
        _n_before = len(comp_refs)
        comp_refs = comp_refs[:cfg_obj.comp_max]
        print(f"[比較星] comp_max={cfg_obj.comp_max}：{_n_before} → {len(comp_refs)} 顆"
              f"（按星等接近度排序）")
        _kept_coords = {(r[0], r[1]) for r in comp_refs}
        active_df = active_df[
            active_df.apply(lambda row: (row["ra_deg"], row["dec_deg"]) in _kept_coords, axis=1)
        ].reset_index(drop=True)

    # ── 7. 檢查星 ──────────────────────────────────────────────────────────────
    # band_df 含該波段所有候選星（含 mag range 外），比 active_df 寬得多，
    # 確保扣除 comp 後仍有 check star 候選。
    check_star = select_check_star(
        cfg_obj.target_radec_deg, comp_refs, band_df, cfg_obj
    )

    print(f"[比較星] band={band} 使用 {active_source}：{len(comp_refs)} 顆")
    print(f"[比較星] epsilon = {epsilon:.4f} arcsec")

    return comp_refs, active_df, check_star, aavso_matched, apass_matched, active_source, _vsx



__all__ = [
    "_load_or_build_unified_catalog",
    "auto_select_comps",
    "select_check_star",
]
