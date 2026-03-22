# -*- coding: utf-8 -*-
"""
download_gaia_tap.py — 用 Gaia Archive TAP 下載 G<11 全天星表
===============================================================
比 VizieR 快得多。Gaia Archive 支援 ADQL 直接篩選。

用法：
  python download_gaia_tap.py

輸出：D:/VarStar/data/share/catalogs/allsky/gaia_dr3_G11.csv

2026-03-22 建立
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path(r"D:\VarStar\data\share\catalogs\allsky")
OUT_CSV = OUT_DIR / "gaia_dr3_G11.csv"


def download_gaia_tap():
    if OUT_CSV.exists():
        df = pd.read_csv(OUT_CSV)
        print(f"已存在 {OUT_CSV}（{len(df)} 列），跳過。刪除檔案可重新下載。")
        return

    from astroquery.gaia import Gaia

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Gaia G<11 全天：預計 ~250-350 萬列
    # 用 async job 處理大查詢
    print("提交 Gaia TAP async 查詢（G < 11 全天）...")
    t0 = time.time()

    query = """
    SELECT ra, dec,
           phot_g_mean_mag AS Gmag,
           phot_bp_mean_mag AS BPmag,
           phot_rp_mean_mag AS RPmag,
           SQRT(1.0/phot_g_mean_flux_over_error) * 1.0857 AS e_Gmag
    FROM gaiadr3.gaia_source
    WHERE phot_g_mean_mag < 11
      AND phot_bp_mean_mag IS NOT NULL
      AND phot_rp_mean_mag IS NOT NULL
    """

    job = Gaia.launch_job_async(query)
    print(f"Job ID: {job.jobid}")
    print("等待查詢完成...")

    results = job.get_results()
    df = results.to_pandas()

    dt = time.time() - t0
    print(f"查詢完成：{len(df)} 列，{dt:.0f} 秒")

    # 重命名欄位
    df = df.rename(columns={"ra": "ra_deg", "dec": "dec_deg"})

    # 去掉 NaN
    df = df.dropna(subset=["ra_deg", "dec_deg", "Gmag"]).reset_index(drop=True)

    # 排序
    df = df.sort_values("Gmag").reset_index(drop=True)

    # 存檔
    df.to_csv(OUT_CSV, index=False)
    size_mb = OUT_CSV.stat().st_size / 1e6
    print(f"已存至 {OUT_CSV}（{size_mb:.1f} MB）")
    print(f"Gmag 範圍：{df['Gmag'].min():.2f} - {df['Gmag'].max():.2f}")
    print(f"總列數：{len(df)}")


if __name__ == "__main__":
    download_gaia_tap()
