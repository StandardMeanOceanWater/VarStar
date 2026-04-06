# -*- coding: utf-8 -*-
"""
phot_catalog.py — 星表查詢、匹配、比較星篩選（純函式，無全域狀態依賴）

從 photometry.py 拆出。包含：
  - _pick_col, _parse_ra_dec       (DataFrame / 座標工具)
  - read_aavso_seq_csv             (AAVSO 本機序列星)
  - fetch_aavso_vsp_api            (AAVSO VSP 線上查詢)
  - filter_catalog_in_frame        (視場內篩選)
  - select_comp_from_catalog       (星等篩選 + 排序)
  - _selection_radius_px           (選取圓半徑)
  - _stars_in_circle               (圓內篩選)
  - _fetch_apass_from_cache        (APASS 本機快取)
  - fetch_apass_cone               (APASS DR9 查詢)
  - fetch_tycho2_cone              (Tycho-2 查詢)
  - fetch_gaia_dr3_cone            (Gaia DR3 查詢)
  - _match_catalog_to_detected     (偵測星 ↔ 星表匹配)
"""

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

from phot_core import radec_to_pixel


# ── DataFrame / 座標工具 ─────────────────────────────────────────────────────

def _pick_col(df: pd.DataFrame, names: list) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"Missing columns. Tried: {names}, available: {list(df.columns)}")


def _parse_ra_dec(ra_val, dec_val) -> "tuple[float, float]":
    if ra_val is None or dec_val is None:
        return np.nan, np.nan
    try:
        return float(str(ra_val).strip()), float(str(dec_val).strip())
    except Exception:
        pass
    ra_s, dec_s = str(ra_val).strip(), str(dec_val).strip()
    for unit_pair in [(u.hourangle, u.deg), (u.deg, u.deg)]:
        try:
            sc = SkyCoord(ra_s, dec_s, unit=unit_pair)
            return sc.ra.deg, sc.dec.deg
        except Exception:
            continue
    return np.nan, np.nan


# ── AAVSO 本機序列星 ─────────────────────────────────────────────────────────

def read_aavso_seq_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if len(df) == 0:
        return df
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    ra_col  = _pick_col(df, ["ra", "ra_deg", "raj2000", "ra (deg)", "ra_hms"])
    dec_col = _pick_col(df, ["dec", "dec_deg", "dej2000", "dec (deg)", "dec_dms"])
    m_col   = _pick_col(df, ["v", "v_mag", "vmag", "mag_v"])
    err_col = next((c for c in ["v_err", "verr", "e_v", "v_error", "v_err_mag"]
                    if c in df.columns), None)
    rows = []
    for _, r in df.iterrows():
        ra_deg, dec_deg = _parse_ra_dec(r[ra_col], r[dec_col])
        if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
            continue
        m_cat = r[m_col]
        if not np.isfinite(m_cat):
            continue
        row = {"ra_deg": float(ra_deg), "dec_deg": float(dec_deg), "m_cat": float(m_cat)}
        if err_col is not None and np.isfinite(r[err_col]):
            row["m_err"] = float(r[err_col])
        rows.append(row)
    return pd.DataFrame(rows)


# ── AAVSO VSP 線上查詢 ───────────────────────────────────────────────────────

def fetch_aavso_vsp_api(
    star_name: str,
    fov_arcmin: float,
    maglimit: float,
    ra_deg: "float | None" = None,
    dec_deg: "float | None" = None,
) -> pd.DataFrame:
    # www.aavso.org/apps/vsp/api/chart/ 是正確的匿名可用 endpoint
    # 優先用星名查詢；若 400（星名不在 AAVSO）則 fallback 到 ra/dec 座標查詢
    params = {"star": star_name, "fov": fov_arcmin, "maglimit": maglimit, "format": "json"}
    url = "https://www.aavso.org/apps/vsp/api/chart/?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as _e:
        # 星名查詢失敗（例如 400 Bad Request）→ 改用 ra/dec 座標查詢
        if ra_deg is None or dec_deg is None:
            raise
        _ra_hms = f"{ra_deg / 15:.6f}"  # AAVSO 接受小時制或度制；用度直接傳
        params_coord = {
            "ra": f"{ra_deg:.6f}",
            "dec": f"{dec_deg:.6f}",
            "fov": fov_arcmin,
            "maglimit": maglimit,
            "format": "json",
        }
        url_coord = "https://www.aavso.org/apps/vsp/api/chart/?" + urllib.parse.urlencode(params_coord)
        print(f"  [AAVSO] 星名查詢失敗({_e})，改用座標查詢：RA={ra_deg:.4f} Dec={dec_deg:.4f}")
        with urllib.request.urlopen(url_coord, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    phot = data.get("photometry", [])
    rows = []
    for item in phot:
        ra_deg, dec_deg = _parse_ra_dec(item.get("ra"), item.get("dec"))
        if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
            continue
        m_cat, m_err = None, None
        bands  = item.get("bands", {})
        vinfo  = None
        if isinstance(bands, dict):
            vinfo = bands.get("V") or bands.get("v")
        elif isinstance(bands, list):
            vinfo = next((b for b in bands if str(b.get("band", "")).upper() == "V"), None)
        if isinstance(vinfo, dict):
            m_cat = vinfo.get("mag") or vinfo.get("magnitude") or vinfo.get("value")
            m_err = vinfo.get("error") or vinfo.get("err") or vinfo.get("sigma")
        if m_cat is None:
            m_cat = next((item.get(k) for k in ("v", "v_mag", "vmag") if k in item), None)
        if m_cat is None:
            continue
        row = {"ra_deg": float(ra_deg), "dec_deg": float(dec_deg), "m_cat": float(m_cat)}
        if m_err is not None and np.isfinite(m_err):
            row["m_err"] = float(m_err)
        rows.append(row)
    return pd.DataFrame(rows)


# ── 視場內篩選 ────────────────────────────────────────────────────────────────

def filter_catalog_in_frame(
    df: pd.DataFrame,
    wcs_obj: WCS,
    shape: "tuple[int, int]",
    margin: int = 10,
) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    h, w = shape
    rows = []
    for _, r in df.iterrows():
        try:
            ra, dec = float(r["ra_deg"]), float(r["dec_deg"])
        except Exception:
            continue
        x, y = radec_to_pixel(wcs_obj, ra, dec)
        if margin <= x < w - margin and margin <= y < h - margin:
            row = r.to_dict()
            row["x"], row["y"] = float(x), float(y)
            rows.append(row)
    return pd.DataFrame(rows)


# ── 比較星星等篩選 ────────────────────────────────────────────────────────────

def select_comp_from_catalog(
    catalog_df: pd.DataFrame,
    mag_min: float,
    mag_max: float,
    max_refs: int = 15,
) -> "tuple[list, pd.DataFrame]":
    if catalog_df is None or len(catalog_df) == 0:
        return [], pd.DataFrame()
    df = catalog_df.copy()
    if "m_cat" in df.columns:
        df = df[np.isfinite(df["m_cat"])]
        df = df[(df["m_cat"] >= mag_min) & (df["m_cat"] <= mag_max)]
    if "m_err" in df.columns:
        df = df.sort_values(["m_err", "m_cat"])
    elif "m_cat" in df.columns:
        df = df.sort_values("m_cat")
    df = df.head(max_refs).copy()
    comp_refs = []
    for _, r in df.iterrows():
        m_err = None
        if "m_err" in r:
            try:
                m_err = float(r["m_err"]) if np.isfinite(r["m_err"]) else None
            except Exception:
                pass
        comp_refs.append((float(r["ra_deg"]), float(r["dec_deg"]), float(r["m_cat"]), m_err))
    return comp_refs, df


# ── 選取圓 ────────────────────────────────────────────────────────────────────

def _selection_radius_px(image: np.ndarray) -> float:
    """
    回傳選取圓的半徑（像素）。
    策略：短邊像素數的一半，保證不同幀之間使用相同標準。
    """
    return float(min(image.shape)) / 2.0


def _stars_in_circle(
    cand_df: pd.DataFrame,
    center_x: float,
    center_y: float,
    radius_px: float,
    image_shape: tuple,
) -> pd.DataFrame:
    """
    從候選星表中篩選落在選取圓內的星。

    選取圓以**影像中心**為圓心（非目標星），半徑為影像短邊的一半。
    以影像中心為圓心可使比較星在空間上對稱分布，大氣梯度的系統誤差
    互相抵消效果最佳。

    Parameters
    ----------
    cand_df     : 含 x, y 欄位的候選星 DataFrame（像素座標）。
    center_x    : 選取圓圓心 x（影像中心，= NAXIS1 / 2）。
    center_y    : 選取圓圓心 y（影像中心，= NAXIS2 / 2）。
    radius_px   : 選取圓半徑（像素，= 影像短邊 / 2）。
    image_shape : 影像 (height, width)，保留供未來使用。
    """
    rows = []
    for _, r in cand_df.iterrows():
        x, y = float(r.get("x", np.nan)), float(r.get("y", np.nan))
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        dist_px = float(np.hypot(x - center_x, y - center_y))
        if dist_px <= radius_px:
            r2 = r.to_dict()
            r2["dist_px"] = dist_px
            rows.append(r2)

    return pd.DataFrame(rows)


# ── APASS 本機快取 ────────────────────────────────────────────────────────────

def _fetch_apass_from_cache(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    cache_dir: Path,
) -> pd.DataFrame:
    """從本機快取（download_apass_cache.py 產生的 CSV）做錐形查詢。"""
    import re as _re
    candidates = list(cache_dir.glob("*.csv"))
    if not candidates:
        return pd.DataFrame()
    best = None
    best_r = -1.0
    for p in candidates:
        try:
            m = _re.search(r"_r([\d.]+)deg", p.stem)
            r_cache = float(m.group(1)) if m else 99.0
            if r_cache >= radius_deg and r_cache > best_r:
                df_tmp = pd.read_csv(p, nrows=1000)
                if "ra_deg" not in df_tmp.columns:
                    continue
                cache_ra  = float(df_tmp["ra_deg"].median())
                cache_dec = float(df_tmp["dec_deg"].median())
                sep = np.sqrt(
                    ((ra_deg - cache_ra) * np.cos(np.radians(dec_deg))) ** 2
                    + (dec_deg - cache_dec) ** 2
                )
                if sep <= r_cache:
                    best = p
                    best_r = r_cache
        except Exception:
            continue
    if best is None:
        return pd.DataFrame()
    df_full = pd.read_csv(best)
    if "ra_deg" not in df_full.columns:
        return pd.DataFrame()
    sep = np.sqrt(
        ((df_full["ra_deg"] - ra_deg) * np.cos(np.radians(dec_deg))) ** 2
        + (df_full["dec_deg"] - dec_deg) ** 2
    )
    df_cone = df_full[sep <= radius_deg].reset_index(drop=True)
    print(f"  [APASS] 本機快取命中：{best.name}，{len(df_cone)} 筆")
    return df_cone


# ── APASS DR9 查詢 ────────────────────────────────────────────────────────────

APASS_SCS_URL = "https://dc.g-vo.org/apass/q/cone/scs.xml"

def fetch_apass_cone(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float = 1.0,
    maxrec: int = 5000,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """查詢 APASS DR9。優先使用本機快取，快取不存在時改用 VizieR。

    結果欄位統一為 ra_deg, dec_deg, vmag, e_vmag, bmag, e_bmag, rmag, e_rmag。
    """
    # 1a. 2026-03-22: 嘗試全天本機星表
    try:
        from tools.local_catalog import query_local_apass as _qla
        _local = _qla(ra_deg, dec_deg, radius_deg=radius_deg)
        if len(_local) > 0:
            return _local
    except Exception:
        pass

    # 1b. 嘗試 per-field 本機快取（舊版）
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "data" / "catalogs" / "apass"
    if cache_dir.exists():
        df_cache = _fetch_apass_from_cache(ra_deg, dec_deg, radius_deg, cache_dir)
        if not df_cache.empty and "vmag" in df_cache.columns:
            return df_cache[np.isfinite(df_cache["vmag"])].reset_index(drop=True)

    # 2. 回退到 VizieR 線上查詢
    """查詢 APASS DR9（astroquery.vizier），回傳視場內所有 APASS 星。

    改用 VizieR 因 dc.g-vo.org endpoint 已失效（404）。
    結果欄位統一為 ra_deg, dec_deg, m_cat, m_err。
    """
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord as _SkyCoord
        import astropy.units as _u

        # 查所有通道需要的波段：V（G1/G2）、B、Rc（R 通道）
        # 注意：改機 Canon 6D2 R 通道延伸至近紅外，Rc 僅為近似，
        # 嚴格色彩轉換係數需另行標定（見已知限制 #10）。
        # APASS DR9 VizieR 欄位：r 波段名稱含單引號 r'mag / e_r'mag
        v = Vizier(
            columns=["RAJ2000", "DEJ2000", "Vmag", "e_Vmag", "Bmag", "e_Bmag",
                     "r'mag", "e_r'mag"],
            row_limit=maxrec,
        )
        result = v.query_region(
            _SkyCoord(ra_deg, dec_deg, unit="deg"),
            radius=radius_deg * _u.deg,
            catalog="II/336/apass9",
        )
        if not result:
            print("  [WARN] APASS VizieR 查詢無結果。")
            return pd.DataFrame()
        tbl = result[0]
        tbl_df = tbl.to_pandas()
        col_map = {
            "RAJ2000": "ra_deg", "DEJ2000": "dec_deg",
            "Vmag":    "vmag",   "e_Vmag":  "e_vmag",
            "Bmag":    "bmag",   "e_Bmag":  "e_bmag",
            "r'mag":   "rmag",   "e_r'mag": "e_rmag",
        }
        df = tbl_df.rename(columns={k: v for k, v in col_map.items() if k in tbl_df.columns})
        df = df[np.isfinite(df["vmag"])].reset_index(drop=True)
        return df
    except Exception as exc:
        print(f"  [WARN] APASS 查詢失敗：{exc}")
        return pd.DataFrame()


# ── Tycho-2 查詢 ──────────────────────────────────────────────────────────────

def fetch_tycho2_cone(
    ra_deg: float,
    dec_deg: float,
    radius_arcmin: float = 40.0,
    mag_min: float = 6.0,
    mag_max: float = 13.0,
) -> pd.DataFrame:
    """
    查 Tycho-2（I/259/tyc2），回傳 ra_deg, dec_deg, vmag, e_vmag, source。
    色彩轉換：V = VT - 0.090 × (BT - VT)（ESA 1997 Vol.1 eq.1.3.20）
    誤差傳遞：e_V = sqrt(e_VT² + 0.090² × (e_BT² + e_VT²))
    BT 或 VT 缺失的星跳過。
    """
    # 2026-03-22: 優先查本機全天星表
    try:
        from tools.local_catalog import query_local_tycho2 as _qlt
        _local = _qlt(ra_deg, dec_deg, radius_deg=radius_arcmin / 60.0,
                       mag_min=mag_min, mag_max=mag_max)
        if len(_local) > 0:
            print(f"  [Tycho-2] 本機全天星表命中：{len(_local)} 筆")
            return _local
    except Exception:
        pass

    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord as _SkyCoord
        import astropy.units as _u

        v = Vizier(
            columns=["RAmdeg", "DEmdeg", "BTmag", "VTmag", "e_BTmag", "e_VTmag"],
            row_limit=2000,
        )
        result = v.query_region(
            _SkyCoord(ra_deg, dec_deg, unit="deg"),
            radius=radius_arcmin * _u.arcmin,
            catalog="I/259/tyc2",
        )
        if not result:
            print("  [WARN] Tycho-2 查詢無結果。")
            return pd.DataFrame()
        df = result[0].to_pandas()
        rows = []
        for _, r in df.iterrows():
            try:
                bt = float(r["BTmag"]); vt = float(r["VTmag"])
                e_bt = float(r["e_BTmag"]); e_vt = float(r["e_VTmag"])
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(bt) and np.isfinite(vt) and
                    np.isfinite(e_bt) and np.isfinite(e_vt)):
                continue
            v_mag = vt - 0.090 * (bt - vt)
            tycho_mag_min = max(mag_min, 7.0)
            if not (tycho_mag_min <= v_mag <= mag_max):
                continue
            e_v = float(np.sqrt(e_vt**2 + 0.090**2 * (e_bt**2 + e_vt**2)))
            rows.append({
                "ra_deg":  float(r["RAmdeg"]),
                "dec_deg": float(r["DEmdeg"]),
                "vmag":    v_mag,
                "e_vmag":  e_v,
                "BT":      bt,
                "VT":      vt,
                "source":  "Tycho2",
            })
        df_out = pd.DataFrame(rows)
        print(f"  [Tycho-2] 查詢={len(df)}  色彩轉換後通過 mag 篩選={len(df_out)}")
        return df_out
    except Exception as exc:
        print(f"  [WARN] Tycho-2 查詢失敗：{exc}")
        return pd.DataFrame()


# ── Gaia DR3 查詢 ─────────────────────────────────────────────────────────────

def fetch_gaia_dr3_cone(
    ra_deg: float,
    dec_deg: float,
    radius_arcmin: float = 40.0,
    mag_min: float = 6.0,
    mag_max: float = 19.0,
    channel: str = "G1",
) -> pd.DataFrame:
    """
    查 Gaia DR3（I/355/gaiadr3），依通道回傳轉換後星等。
    G1/G2：G→V  (Riello et al. 2021)
      V = G + 0.02704 - 0.01424c + 0.2156c² - 0.01426c³  (c=BP-RP)
    R   ：G→Rc  (Riello et al. 2021)
      Rc = G - (0.02275 + 0.3961c - 0.1243c² - 0.01396c³ + 0.003775c⁴)
    B   ：G→B   (使用者指定公式)
      B = G + 0.0939 + 0.6758c + 0.0743c²
      [WARN] 僅適用 BP-RP < 2，超出範圍仍計算但標記 warn_color
    Gaia 亮星容忍：mag_min 最小 6.0（G<3 才飽和）。
    """
    # 2026-03-22: 優先查本機全天星表
    try:
        from tools.local_catalog import query_local_gaia as _qlg
        _local = _qlg(ra_deg, dec_deg, radius_deg=radius_arcmin / 60.0,
                       mag_min=mag_min, mag_max=mag_max, channel=channel)
        if len(_local) > 0:
            return _local
    except Exception:
        pass  # 本機不可用時 fallback 到線上

    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord as _SkyCoord
        import astropy.units as _u

        gaia_mag_min = max(mag_min, 6.0)
        ch = channel.upper()

        v = Vizier(
            columns=["RA_ICRS", "DE_ICRS", "Gmag", "BPmag", "RPmag", "e_Gmag"],
            row_limit=1800,
        )
        result = v.query_region(
            _SkyCoord(ra_deg, dec_deg, unit="deg"),
            radius=radius_arcmin * _u.arcmin,
            catalog="I/355/gaiadr3",
        )
        if not result:
            print("  [WARN] Gaia DR3 查詢無結果。")
            return pd.DataFrame()
        df = result[0].to_pandas()
        rows = []
        n_warn_color = 0
        for _, r in df.iterrows():
            try:
                g = float(r["Gmag"])
                bp = float(r["BPmag"])
                rp = float(r["RPmag"])
                e_g = float(r["e_Gmag"])
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(g) and np.isfinite(bp) and np.isfinite(rp)):
                continue
            c = bp - rp
            warn_color = False

            if ch in ("G1", "G2"):
                conv_mag = g + 0.02704 - 0.01424*c + 0.2156*c**2 - 0.01426*c**3
                df_dc = -0.01424 + 0.4312*c - 0.04278*c**2
                e_bp_safe = 0.02; e_rp_safe = 0.02
                e_conv = float(np.sqrt(e_g**2 + df_dc**2 * (e_bp_safe**2 + e_rp_safe**2)))
            elif ch == "R":
                conv_mag = g - (0.02275 + 0.3961*c - 0.1243*c**2 - 0.01396*c**3 + 0.003775*c**4)
                dg_dc = 0.3961 - 0.2486*c - 0.04188*c**2 + 0.01510*c**3
                e_bp_safe = 0.02; e_rp_safe = 0.02
                e_conv = float(np.sqrt(e_g**2 + dg_dc**2 * (e_bp_safe**2 + e_rp_safe**2)))
            elif ch == "B":
                if c >= 2.0:
                    warn_color = True
                    n_warn_color += 1
                conv_mag = g + 0.0939 + 0.6758*c + 0.0743*c**2
                db_dc = 0.6758 + 0.1486*c
                e_bp_safe = 0.02; e_rp_safe = 0.02
                e_conv = float(np.sqrt(e_g**2 + db_dc**2 * (e_bp_safe**2 + e_rp_safe**2)))
            else:
                continue

            if not (gaia_mag_min <= conv_mag <= mag_max):
                continue
            rows.append({
                "ra_deg":      float(r["RA_ICRS"]),
                "dec_deg":     float(r["DE_ICRS"]),
                "vmag":        conv_mag,
                "e_vmag":      e_conv,
                "Gmag":        g,
                "BPmag":       bp,
                "RPmag":       rp,
                "warn_color":  warn_color,
                "source":      "GaiaDR3",
            })
        if n_warn_color > 0:
            print(f"  [WARN] Gaia B 通道：{n_warn_color} 顆星 BP-RP ≥ 2，色彩轉換誤差偏大（> 0.1 等）")
        df_out = pd.DataFrame(rows)
        print(f"  [Gaia DR3] ch={ch}  查詢={len(df)}  轉換後通過 mag 篩選={len(df_out)}")
        return df_out
    except Exception as exc:
        print(f"  [WARN] Gaia DR3 查詢失敗：{exc}")
        return pd.DataFrame()


# ── 偵測星 ↔ 星表匹配 ────────────────────────────────────────────────────────

def _match_catalog_to_detected(
    detected_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    ra_col: str,
    dec_col: str,
    mag_col: str,
    err_col: "str | None",
    match_arcsec: float,
    mag_min: float,
    mag_max: float,
) -> pd.DataFrame:
    """
    把偵測到的星（pixel 座標）與星表（RA/Dec）做位置匹配，
    回傳含 m_cat 欄位的 DataFrame。
    """
    if len(detected_df) == 0 or len(catalog_df) == 0:
        return pd.DataFrame()

    det_coords = SkyCoord(
        detected_df["ra_deg"].values * u.deg,
        detected_df["dec_deg"].values * u.deg,
    )
    cat_coords = SkyCoord(
        catalog_df[ra_col].astype(float).values * u.deg,
        catalog_df[dec_col].astype(float).values * u.deg,
    )
    idx, sep2d, _ = det_coords.match_to_catalog_sky(cat_coords)

    matched = detected_df.copy().reset_index(drop=True)
    matched["m_cat"] = catalog_df.iloc[idx][mag_col].values
    if err_col and err_col in catalog_df.columns:
        matched["m_err"] = catalog_df.iloc[idx][err_col].values
    matched["cat_sep_arcsec"] = sep2d.arcsec

    matched = matched[
        (matched["cat_sep_arcsec"] <= match_arcsec)
        & np.isfinite(matched["m_cat"])
        & (matched["m_cat"] >= mag_min)
        & (matched["m_cat"] <= mag_max)
    ].copy()

    return matched
