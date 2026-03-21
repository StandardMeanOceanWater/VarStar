"""
download_apass_cache.py — APASS DR9 本機星表快取下載工具
=========================================================
用途：預先下載各目標視野的 APASS DR9 星表，存成本地 CSV，
      供 photometry.py 在無網路時使用。

存放位置：data/catalogs/apass/{TARGET}_r{radius_deg}deg.csv

使用方式
--------
  python download_apass_cache.py                    # 下載 yaml 所有 target
  python download_apass_cache.py --target V1162Ori  # 只下載指定 target
  python download_apass_cache.py --radius 2.0       # 自訂半徑（deg，預設 1.5）
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def download_apass_field(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float = 1.5,
    maxrec: int = 50000,
) -> pd.DataFrame:
    """查詢 APASS DR9（VizieR），回傳 DataFrame。"""
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    v = Vizier(
        columns=["RAJ2000", "DEJ2000", "Vmag", "e_Vmag", "Bmag", "e_Bmag",
                 "r'mag", "e_r'mag"],
        row_limit=maxrec,
    )
    result = v.query_region(
        SkyCoord(ra_deg, dec_deg, unit="deg"),
        radius=radius_deg * u.deg,
        catalog="II/336/apass9",
    )
    if not result:
        return pd.DataFrame()

    tbl_df = result[0].to_pandas()
    col_map = {
        "RAJ2000": "ra_deg",  "DEJ2000": "dec_deg",
        "Vmag":    "vmag",    "e_Vmag":  "e_vmag",
        "Bmag":    "bmag",    "e_Bmag":  "e_bmag",
        "r'mag":   "rmag",    "e_r'mag": "e_rmag",
    }
    df = tbl_df.rename(columns={k: v for k, v in col_map.items() if k in tbl_df.columns})
    return df.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description="APASS DR9 本機快取下載")
    p.add_argument("--config", type=Path,
                   default=Path(__file__).parent / "observation_config.yaml")
    p.add_argument("--target", default=None, help="只下載指定 target（預設全部）")
    p.add_argument("--radius", type=float, default=1.5,
                   help="下載半徑 deg（預設 1.5）")
    p.add_argument("--force", action="store_true",
                   help="強制重新下載（即使快取已存在）")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    project_root = Path(cfg["paths"]["local"]["project_root"])
    cache_dir = project_root / "data" / "catalogs" / "apass"
    cache_dir.mkdir(parents=True, exist_ok=True)

    targets_cfg = cfg.get("targets", {})
    if args.target:
        targets_cfg = {k: v for k, v in targets_cfg.items() if k == args.target}

    if not targets_cfg:
        print("[apass] 找不到目標設定，請確認 yaml targets 區塊。")
        return

    for name, tcfg in targets_cfg.items():
        ra_h  = tcfg.get("ra_hint_h")
        dec_d = tcfg.get("dec_hint_deg")
        if ra_h is None or dec_d is None:
            print(f"[apass] {name}: 缺少 ra_hint_h / dec_hint_deg，跳過。")
            continue

        ra_deg = float(ra_h) * 15.0
        dec_deg = float(dec_d)
        r = args.radius
        out_csv = cache_dir / f"{name}_r{r:.1f}deg.csv"

        if out_csv.exists() and not args.force:
            print(f"[apass] {name}: 快取已存在（{out_csv.name}），跳過。"
                  f"（--force 重新下載）")
            continue

        print(f"[apass] {name}: 下載 RA={ra_deg:.3f}° Dec={dec_deg:.3f}°"
              f" r={r}° ...", end="", flush=True)
        try:
            df = download_apass_field(ra_deg, dec_deg, radius_deg=r)
        except Exception as e:
            print(f" 失敗：{e}")
            continue

        if df.empty:
            print(" 無結果。")
            continue

        df.to_csv(out_csv, index=False)
        print(f" {len(df)} 筆 → {out_csv}")

    print("[apass] 完成。本機快取位置：", cache_dir)


if __name__ == "__main__":
    main()
