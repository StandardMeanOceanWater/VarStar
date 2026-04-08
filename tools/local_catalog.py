# -*- coding: utf-8 -*-
"""
local_catalog.py — 全天本機星表錐形查詢
========================================
為 download_catalogs.py 下載的全天星表提供快速錐形查詢。
使用 pandas 讀取 CSV + 球面距離篩選。

支援的星表：
  - Gaia DR3 (G<11): gaia_dr3_G11.csv
  - Tycho-2 (full):  tycho2_full.csv
  - APASS DR9 (V<11): apass_dr9_V11.csv

用法（作為模組）：
  from tools.local_catalog import query_local_gaia, query_local_tycho2, query_local_apass

  df = query_local_gaia(ra_deg=85.0, dec_deg=-1.0, radius_deg=1.0, channel="R")
  # 回傳 DataFrame: ra_deg, dec_deg, vmag, e_vmag, source, [Gmag, BPmag, RPmag]

2026-03-22 建立
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 全天星表預設目錄
ALLSKY_DIR = Path(r"D:\VarStar\data\share\catalogs\allsky")

# 快取：第一次讀取後保留在記憶體
_cache = {}


def _load_csv(name: str) -> pd.DataFrame:
    """懶載入 CSV，快取在記憶體中"""
    if name in _cache:
        return _cache[name]
    csv_path = ALLSKY_DIR / name
    if not csv_path.exists():
        print(f"  [local_catalog] 找不到 {csv_path}")
        return pd.DataFrame()
    print(f"  [local_catalog] 載入 {csv_path.name} ...", end="", flush=True)
    df = pd.read_csv(csv_path)
    # 去掉 RA/Dec 為 NaN 的列
    ra_col = [c for c in df.columns if c.lower() in ("ra_deg", "ramdeg", "ra_icrs")][0]
    dec_col = [c for c in df.columns if c.lower() in ("dec_deg", "demdeg", "de_icrs")][0]
    df = df.dropna(subset=[ra_col, dec_col]).reset_index(drop=True)
    _cache[name] = df
    print(f" {len(df)} 列")
    return df


def _cone_search(df: pd.DataFrame, ra_col: str, dec_col: str,
                 ra_deg: float, dec_deg: float, radius_deg: float) -> pd.DataFrame:
    """球面近似錐形查詢（cos(dec) 修正）"""
    cos_dec = np.cos(np.radians(dec_deg))
    dra = (df[ra_col].values - ra_deg) * cos_dec
    ddec = df[dec_col].values - dec_deg
    sep2 = dra**2 + ddec**2
    mask = sep2 <= radius_deg**2
    return df[mask].reset_index(drop=True)


# ── Gaia DR3 ──────────────────────────────────────────────────────

def query_local_gaia(ra_deg: float, dec_deg: float, radius_deg: float = 1.0,
                     mag_min: float = 6.0, mag_max: float = 13.0,
                     channel: str = "G1") -> pd.DataFrame:
    """
    從本機 Gaia DR3 全天星表做錐形查詢，並轉換為對應通道星等。

    轉換公式（Riello et al. 2021）：
      G1/G2 → V:  V  = G + 0.02704 - 0.01424c + 0.2156c^2 - 0.01426c^3
      R     → Rc: Rc = G - (0.02275 + 0.3961c - 0.1243c^2 - 0.01396c^3 + 0.003775c^4)
      B     → B:  B  = G + 0.0939 + 0.6758c + 0.0743c^2
    其中 c = BP - RP。
    """
    # 優先用 G<13 星表，fallback 到 G<14 → G<11
    df = _load_csv("gaia_dr3_G13.csv")
    if df.empty:
        df = _load_csv("gaia_dr3_G14.csv")
    if df.empty:
        df = _load_csv("gaia_dr3_G11.csv")
    if df.empty:
        return pd.DataFrame()

    cone = _cone_search(df, "ra_deg", "dec_deg", ra_deg, dec_deg, radius_deg)
    if cone.empty:
        return pd.DataFrame()

    # 轉換星等
    ch = channel.upper()
    g = cone["Gmag"].values
    bp = cone["BPmag"].values
    rp = cone["RPmag"].values
    e_g = cone["e_Gmag"].values
    c = bp - rp

    valid = np.isfinite(g) & np.isfinite(bp) & np.isfinite(rp)

    if ch in ("G1", "G2"):
        conv_mag = g + 0.02704 - 0.01424*c + 0.2156*c**2 - 0.01426*c**3
        df_dc = -0.01424 + 0.4312*c - 0.04278*c**2
    elif ch == "R":
        conv_mag = g - (0.02275 + 0.3961*c - 0.1243*c**2 - 0.01396*c**3 + 0.003775*c**4)
        df_dc = 0.3961 - 0.2486*c - 0.04188*c**2 + 0.01510*c**3
    elif ch == "B":
        conv_mag = g + 0.0939 + 0.6758*c + 0.0743*c**2
        df_dc = 0.6758 + 0.1486*c
    else:
        return pd.DataFrame()

    e_bp_safe = 0.02
    e_rp_safe = 0.02
    e_conv = np.sqrt(e_g**2 + df_dc**2 * (e_bp_safe**2 + e_rp_safe**2))

    # 星等範圍篩選
    gaia_mag_min = max(mag_min, 6.0)
    mag_ok = valid & (conv_mag >= gaia_mag_min) & (conv_mag <= mag_max)
    warn_color = (c >= 2.0) & valid

    result = pd.DataFrame({
        "ra_deg":     cone["ra_deg"].values[mag_ok],
        "dec_deg":    cone["dec_deg"].values[mag_ok],
        "vmag":       conv_mag[mag_ok],
        "e_vmag":     e_conv[mag_ok],
        "Gmag":       g[mag_ok],
        "BPmag":      bp[mag_ok],
        "RPmag":      rp[mag_ok],
        "warn_color": warn_color[mag_ok],
        "source":     "GaiaDR3",
    })

    n_warn = int(warn_color[mag_ok].sum())
    if n_warn > 0 and ch == "B":
        print(f"  [local Gaia] ch={ch}: {n_warn} 顆 BP-RP >= 2，色彩轉換誤差偏大")
    print(f"  [local Gaia] ch={ch}: 查詢={len(cone)}, 通過={len(result)}")
    return result


# ── Tycho-2 ────────────────────────────────────────────────────────

def query_local_tycho2(ra_deg: float, dec_deg: float, radius_deg: float = 1.0,
                       mag_min: float = 6.0, mag_max: float = 13.0) -> pd.DataFrame:
    """
    從本機 Tycho-2 全天星表做錐形查詢。
    BT/VT → V 轉換：V = VT - 0.090 * (BT - VT)（ESA 1997）
    """
    df = _load_csv("tycho2_full.csv")
    if df.empty:
        return pd.DataFrame()

    cone = _cone_search(df, "ra_deg", "dec_deg", ra_deg, dec_deg, radius_deg)
    if cone.empty:
        return pd.DataFrame()

    bt = cone["BTmag"].values
    vt = cone["VTmag"].values
    valid = np.isfinite(bt) & np.isfinite(vt)

    v_mag = vt - 0.090 * (bt - vt)
    e_bt = cone["e_BTmag"].values if "e_BTmag" in cone.columns else np.full(len(cone), 0.05)
    e_vt = cone["e_VTmag"].values if "e_VTmag" in cone.columns else np.full(len(cone), 0.05)
    e_v = np.sqrt(e_vt**2 + 0.090**2 * (e_bt**2 + e_vt**2))

    tycho_mag_min = max(mag_min, 7.0)
    mag_ok = valid & (v_mag >= tycho_mag_min) & (v_mag <= mag_max)

    result = pd.DataFrame({
        "ra_deg":  cone["ra_deg"].values[mag_ok],
        "dec_deg": cone["dec_deg"].values[mag_ok],
        "vmag":    v_mag[mag_ok],
        "e_vmag":  e_v[mag_ok],
        "BT":      bt[mag_ok],
        "VT":      vt[mag_ok],
        "source":  "Tycho2",
    })
    print(f"  [local Tycho-2] 查詢={len(cone)}, 通過={len(result)}")
    return result


# ── APASS DR9 ──────────────────────────────────────────────────────

def query_local_apass(ra_deg: float, dec_deg: float, radius_deg: float = 1.0,
                      mag_min: float = 4.0, mag_max: float = 13.0) -> pd.DataFrame:
    """
    從本機 APASS DR9 全天星表做錐形查詢。
    注意：本機版只有 Vmag, Bmag（無 rmag/imag）。
    """
    df = _load_csv("apass_dr9_V11.csv")
    if df.empty:
        return pd.DataFrame()

    cone = _cone_search(df, "ra_deg", "dec_deg", ra_deg, dec_deg, radius_deg)
    if cone.empty:
        return pd.DataFrame()

    vmag = cone["Vmag"].values
    valid = np.isfinite(vmag)
    mag_ok = valid & (vmag >= mag_min) & (vmag <= mag_max)

    keep_cols = ["ra_deg", "dec_deg", "Vmag", "e_Vmag"]
    if "Bmag" in cone.columns:
        keep_cols += ["Bmag", "e_Bmag"]

    result = cone.loc[mag_ok, keep_cols].copy().reset_index(drop=True)
    result = result.rename(columns={"Vmag": "vmag", "e_Vmag": "e_vmag",
                                     "Bmag": "bmag", "e_Bmag": "e_bmag"})
    result["source"] = "APASS"
    print(f"  [local APASS] 查詢={len(cone)}, 通過={len(result)}")
    return result


# ── VSX 變星排除 ──────────────────────────────────────────────────

def query_vsx_cone(ra_deg: float, dec_deg: float, radius_deg: float = 1.0
                   ) -> pd.DataFrame:
    """
    查詢 AAVSO VSX（Variable Star Index），回傳視場內已知變星的位置和類型。
    用於從比較星候選中排除已知變星。

    優先使用本機快取（allsky/vsx_variables.csv），
    找不到時 fallback 到 VizieR 線上查詢。
    """
    # 1. 嘗試本機快取
    local_csv = ALLSKY_DIR / "vsx_variables.csv"
    if local_csv.exists():
        df = _load_csv("vsx_variables.csv")
        if not df.empty:
            cone = _cone_search(df, "ra_deg", "dec_deg", ra_deg, dec_deg, radius_deg)
            if len(cone) > 0:
                print(f"  [local VSX] 查詢={len(cone)} 顆已知變星")
                return cone
            # 本機快取無此天區資料，fallback 到線上查詢

    # 2. 線上查詢 VizieR，查完後追加到本機快取
    try:
        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord as _SC
        import astropy.units as _u

        viz = Vizier(
            columns=["RAJ2000", "DEJ2000", "Name", "Type", "max", "min", "Period"],
            row_limit=5000,
        )
        result = viz.query_region(
            _SC(ra_deg, dec_deg, unit="deg"),
            radius=radius_deg * _u.deg,
            catalog="B/vsx/vsx",
        )
        if not result:
            print("  [VSX] 查詢無結果")
            return pd.DataFrame()
        df = result[0].to_pandas()
        df = df.rename(columns={"RAJ2000": "ra_deg", "DEJ2000": "dec_deg"})
        print(f"  [VSX online] {len(df)} 顆已知變星")

        # 追加到本機快取（累積式）
        try:
            ALLSKY_DIR.mkdir(parents=True, exist_ok=True)
            if local_csv.exists():
                existing = pd.read_csv(local_csv)
                combined = pd.concat([existing, df], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=["ra_deg", "dec_deg"], keep="first"
                ).reset_index(drop=True)
            else:
                combined = df
            combined.to_csv(local_csv, index=False, encoding="utf-8-sig")
            # 更新記憶體快取
            _cache.pop("vsx_variables.csv", None)
            print(f"  [VSX] 已追加到本機快取（累計 {len(combined)} 顆）")
        except Exception as exc_save:
            print(f"  [VSX] 快取寫入失敗：{exc_save}")

        return df
    except Exception as exc:
        print(f"  [WARN] VSX 查詢失敗：{exc}")
        return pd.DataFrame()


def filter_known_variables(comp_df: pd.DataFrame, vsx_df: pd.DataFrame,
                           match_arcsec: float = 10.0) -> pd.DataFrame:
    """
    從比較星候選 DataFrame 中移除與 VSX 已知變星位置匹配的星。

    Parameters:
        comp_df: 比較星候選（需有 ra_deg, dec_deg）
        vsx_df: VSX 查詢結果
        match_arcsec: 匹配半徑（角秒）

    Returns:
        過濾後的 comp_df（移除了已知變星）
    """
    if vsx_df.empty or comp_df.empty:
        return comp_df

    from astropy.coordinates import SkyCoord
    import astropy.units as u

    comp_coords = SkyCoord(comp_df["ra_deg"].values, comp_df["dec_deg"].values, unit="deg")
    vsx_coords = SkyCoord(vsx_df["ra_deg"].values, vsx_df["dec_deg"].values, unit="deg")

    idx, sep, _ = comp_coords.match_to_catalog_sky(vsx_coords)
    is_variable = sep.arcsec < match_arcsec

    n_removed = int(is_variable.sum())
    if n_removed > 0:
        # 印出被排除的變星名稱
        for i in np.where(is_variable)[0]:
            vsx_row = vsx_df.iloc[idx[i]]
            name = vsx_row.get("Name", "?")
            vtype = vsx_row.get("Type", "?")
            print(f"    排除已知變星：{name} (type={vtype}, sep={sep[i].arcsec:.1f}\")")
        print(f"  [VSX] 排除 {n_removed} 顆已知變星，保留 {len(comp_df)-n_removed} 顆")

    return comp_df[~is_variable].reset_index(drop=True)


# ── 測試 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # 測試：CC And 附近
    print("=== CC And (RA=10.95, Dec=+41.25) ===")
    for fn in [query_local_tycho2, query_local_apass]:
        df = fn(10.95, 41.25, radius_deg=1.0)
        if not df.empty:
            print(f"  {df['source'].iloc[0]}: {len(df)} stars, vmag {df['vmag'].min():.1f}-{df['vmag'].max():.1f}")

    # Gaia 測試（如果已下載）
    for ch in ["G1", "R", "B"]:
        df = query_local_gaia(10.95, 41.25, radius_deg=1.0, channel=ch)
        if not df.empty:
            print(f"  Gaia({ch}): {len(df)} stars, vmag {df['vmag'].min():.1f}-{df['vmag'].max():.1f}")
