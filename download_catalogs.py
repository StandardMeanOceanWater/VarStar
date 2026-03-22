"""
download_catalogs.py — 下載全天 V/G < 11 等星表到本機
=======================================================
目標目錄：D:/VarStar/data/share/catalogs/allsky/
星表：APASS DR9, Tycho-2, Gaia DR3（AAVSO VSP 無批量下載，不處理）

用法：
  python download_catalogs.py                # 下載全部
  python download_catalogs.py --catalog gaia  # 只下載 Gaia
  python download_catalogs.py --catalog tycho2
  python download_catalogs.py --catalog apass
  python download_catalogs.py --dry-run       # 只估算大小，不下載

每個星表拆成多次錐形查詢（VizieR 單次有 row_limit），
最後合併去重存為單一 CSV。

2026-03-22 建立
"""

import argparse
import sys
import time
from pathlib import Path

# Windows cp950 安全輸出
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

OUT_DIR = Path(r"D:\VarStar\data\share\catalogs\allsky")


def _ensure_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── HEALPix 格點產生查詢中心 ───────────────────────────────────────
def _generate_query_centers(radius_deg: float = 3.0):
    """
    在全天均勻分佈查詢中心點，確保覆蓋。
    用簡單的 Dec 帶 + RA 等間距。鄰域有適度重疊（合併時去重）。
    """
    centers = []
    step = radius_deg * 1.6  # 重疊 ~20%
    for dec in np.arange(-90 + radius_deg, 90 - radius_deg + 0.1, step):
        cos_dec = max(np.cos(np.radians(dec)), 0.05)
        ra_step = step / cos_dec
        for ra in np.arange(0, 360, ra_step):
            centers.append((ra, dec))
    # 極區補充
    centers.append((0.0, 89.0))
    centers.append((0.0, -89.0))
    print(f"  全天查詢格點：{len(centers)} 個（半徑 {radius_deg}°，步進 {step:.2f}°）")
    return centers


# ── Gaia DR3 ───────────────────────────────────────────────────────
def download_gaia(dry_run=False):
    """Gaia DR3, G < 11, 保留 RA, Dec, G, BP, RP, e_G"""
    out_csv = OUT_DIR / "gaia_dr3_G11.csv"
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        print(f"[Gaia DR3] 已存在 {out_csv}（{len(existing)} 列），跳過。刪除檔案可重新下載。")
        return

    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    radius_deg = 3.0
    centers = _generate_query_centers(radius_deg)
    if dry_run:
        print(f"[Gaia DR3 dry-run] {len(centers)} 次查詢 × ~500-2000 列/次 ~ 250k-350k 列")
        print(f"  預估 CSV：250-400 MB")
        return

    _ensure_dir()
    all_rows = []
    viz = Vizier(
        columns=["RA_ICRS", "DE_ICRS", "Gmag", "BPmag", "RPmag", "e_Gmag"],
        row_limit=50000,
        column_filters={"Gmag": "<11"},
    )

    t0 = time.time()
    for i, (ra, dec) in enumerate(centers):
        try:
            result = viz.query_region(
                SkyCoord(ra, dec, unit="deg"),
                radius=radius_deg * u.deg,
                catalog="I/355/gaiadr3",
            )
            if result:
                df = result[0].to_pandas()
                all_rows.append(df)
                if (i + 1) % 50 == 0:
                    n_total = sum(len(d) for d in all_rows)
                    elapsed = time.time() - t0
                    print(f"  [{i+1}/{len(centers)}] 累計 {n_total} 列  "
                          f"（{elapsed:.0f}s，{elapsed/(i+1):.1f}s/查詢）")
        except Exception as exc:
            print(f"  [WARN] 格點 ({ra:.1f}, {dec:.1f}) 失敗：{exc}")
            time.sleep(2)
        # VizieR 禮貌等待
        time.sleep(0.3)

    if not all_rows:
        print("[Gaia DR3] 全部失敗！")
        return

    merged = pd.concat(all_rows, ignore_index=True)
    # 去重（RA/Dec 相同到 0.001°）
    merged = merged.drop_duplicates(subset=["RA_ICRS", "DE_ICRS"], keep="first")
    merged = merged.rename(columns={
        "RA_ICRS": "ra_deg", "DE_ICRS": "dec_deg",
        "Gmag": "Gmag", "BPmag": "BPmag", "RPmag": "RPmag", "e_Gmag": "e_Gmag",
    })
    merged = merged.sort_values("Gmag").reset_index(drop=True)
    merged.to_csv(out_csv, index=False)
    elapsed = time.time() - t0
    print(f"[Gaia DR3] 完成：{len(merged)} 列，{out_csv.stat().st_size/1e6:.1f} MB，"
          f"耗時 {elapsed/60:.1f} 分鐘")


# ── Tycho-2 ────────────────────────────────────────────────────────
def download_tycho2(dry_run=False):
    """Tycho-2 全表（~250 萬星，本身就是 V<12）"""
    out_csv = OUT_DIR / "tycho2_full.csv"
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        print(f"[Tycho-2] 已存在 {out_csv}（{len(existing)} 列），跳過。")
        return

    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    radius_deg = 5.0  # Tycho-2 星數少，用大半徑減少查詢次數
    centers = _generate_query_centers(radius_deg)
    if dry_run:
        print(f"[Tycho-2 dry-run] {len(centers)} 次查詢，全表 ~250 萬列")
        print(f"  預估 CSV：250-350 MB")
        return

    _ensure_dir()
    all_rows = []
    viz = Vizier(
        columns=["RAmdeg", "DEmdeg", "BTmag", "VTmag", "e_BTmag", "e_VTmag",
                 "pmRA", "pmDE", "HIP"],
        row_limit=50000,
    )

    t0 = time.time()
    for i, (ra, dec) in enumerate(centers):
        try:
            result = viz.query_region(
                SkyCoord(ra, dec, unit="deg"),
                radius=radius_deg * u.deg,
                catalog="I/259/tyc2",
            )
            if result:
                df = result[0].to_pandas()
                all_rows.append(df)
                if (i + 1) % 20 == 0:
                    n_total = sum(len(d) for d in all_rows)
                    elapsed = time.time() - t0
                    print(f"  [{i+1}/{len(centers)}] 累計 {n_total} 列  "
                          f"（{elapsed:.0f}s，{elapsed/(i+1):.1f}s/查詢）")
        except Exception as exc:
            print(f"  [WARN] 格點 ({ra:.1f}, {dec:.1f}) 失敗：{exc}")
            time.sleep(2)
        time.sleep(0.3)

    if not all_rows:
        print("[Tycho-2] 全部失敗！")
        return

    merged = pd.concat(all_rows, ignore_index=True)
    merged = merged.drop_duplicates(subset=["RAmdeg", "DEmdeg"], keep="first")
    merged = merged.rename(columns={"RAmdeg": "ra_deg", "DEmdeg": "dec_deg"})
    merged = merged.sort_values("VTmag").reset_index(drop=True)
    merged.to_csv(out_csv, index=False)
    elapsed = time.time() - t0
    print(f"[Tycho-2] 完成：{len(merged)} 列，{out_csv.stat().st_size/1e6:.1f} MB，"
          f"耗時 {elapsed/60:.1f} 分鐘")


# ── APASS DR9 ──────────────────────────────────────────────────────
def download_apass(dry_run=False):
    """APASS DR9 (II/336), V < 11, 保留多色星等"""
    out_csv = OUT_DIR / "apass_dr9_V11.csv"
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        print(f"[APASS] 已存在 {out_csv}（{len(existing)} 列），跳過。")
        return

    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    radius_deg = 3.0
    centers = _generate_query_centers(radius_deg)
    if dry_run:
        print(f"[APASS dry-run] {len(centers)} 次查詢 × ~100-500 列/次 ~ 50k-80k 列")
        print(f"  預估 CSV：80-140 MB")
        return

    _ensure_dir()
    all_rows = []
    viz = Vizier(
        columns=["RAJ2000", "DEJ2000", "Vmag", "e_Vmag",
                 "Bmag", "e_Bmag", "g_mag", "r_mag", "i_mag",
                 "e_g_mag", "e_r_mag", "e_i_mag"],
        row_limit=50000,
        column_filters={"Vmag": "<11"},
    )

    t0 = time.time()
    for i, (ra, dec) in enumerate(centers):
        try:
            result = viz.query_region(
                SkyCoord(ra, dec, unit="deg"),
                radius=radius_deg * u.deg,
                catalog="II/336/apass9",
            )
            if result:
                df = result[0].to_pandas()
                all_rows.append(df)
                if (i + 1) % 50 == 0:
                    n_total = sum(len(d) for d in all_rows)
                    elapsed = time.time() - t0
                    print(f"  [{i+1}/{len(centers)}] 累計 {n_total} 列  "
                          f"（{elapsed:.0f}s，{elapsed/(i+1):.1f}s/查詢）")
        except Exception as exc:
            print(f"  [WARN] 格點 ({ra:.1f}, {dec:.1f}) 失敗：{exc}")
            time.sleep(2)
        time.sleep(0.3)

    if not all_rows:
        print("[APASS] 全部失敗！")
        return

    merged = pd.concat(all_rows, ignore_index=True)
    merged = merged.drop_duplicates(subset=["RAJ2000", "DEJ2000"], keep="first")
    merged = merged.rename(columns={"RAJ2000": "ra_deg", "DEJ2000": "dec_deg"})
    merged = merged.sort_values("Vmag").reset_index(drop=True)
    merged.to_csv(out_csv, index=False)
    elapsed = time.time() - t0
    print(f"[APASS] 完成：{len(merged)} 列，{out_csv.stat().st_size/1e6:.1f} MB，"
          f"耗時 {elapsed/60:.1f} 分鐘")


# ── 主程式 ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="下載全天星表（V/G < 11）到本機")
    parser.add_argument("--catalog", choices=["gaia", "tycho2", "apass", "all"],
                        default="all", help="要下載的星表（預設 all）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只估算大小，不實際下載")
    args = parser.parse_args()

    print(f"輸出目錄：{OUT_DIR}")
    print(f"模式：{'dry-run（僅估算）' if args.dry_run else '實際下載'}")
    print()

    targets = {
        "apass":  download_apass,
        "tycho2": download_tycho2,
        "gaia":   download_gaia,
    }

    if args.catalog == "all":
        for name, func in targets.items():
            print(f"{'='*60}")
            func(dry_run=args.dry_run)
            print()
    else:
        targets[args.catalog](dry_run=args.dry_run)

    print("完成。")


if __name__ == "__main__":
    main()
