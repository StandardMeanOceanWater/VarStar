# -*- coding: utf-8 -*-
"""
sync_ok_frames.py — 同步 G1/G2（及其他通道）的 ok 幀集合
=========================================================
讀取同一 run 下多通道的 photometry CSV，取 ok=1 幀的**交集**，
將不在交集中的幀設為 ok=0, ok_flag="cross_channel_sync"，
重新計算 ensemble（如啟用），輸出同步後的 CSV 和光變曲線。

用途：消除因各通道獨立篩選導致的時間採樣差異，
      這是 G1/G2 相關性降低的一個非物理原因。

用法：
  python tools/sync_ok_frames.py D:/VarStar/output/2025-11-22/And/CCAnd/20260321_232048
  python tools/sync_ok_frames.py <run_root> --channels G1 G2
  python tools/sync_ok_frames.py <run_root> --channels G1 G2 R B --dry-run

2026-03-22 建立
"""

import sys
import argparse
from pathlib import Path

import re

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _base_name(filename: str) -> str:
    """Cal_..._wcs_G1.fits -> Cal_..._wcs.fits（去掉通道後綴）"""
    return re.sub(r'_(?:G1|G2|R|B)\.fits$', '.fits', filename)


def find_photometry_csvs(run_root: Path, channels: list):
    """找到 run_root/1_photometry/ 下各通道的 CSV"""
    phot_dir = run_root / "1_photometry"
    found = {}
    for ch in channels:
        candidates = sorted(phot_dir.glob(f"photometry_{ch}_*.csv"))
        if candidates:
            found[ch] = candidates[-1]  # 取最新的
    return found


def sync_ok_frames(csv_dict: dict, dry_run: bool = False):
    """
    讀取多通道 CSV，取 ok=1 的 file 欄位交集，
    將不在交集中的幀標記為 ok=0。

    Returns: {channel: (df_synced, n_removed)}
    """
    # 讀取所有通道
    dfs = {}
    for ch, csv_path in csv_dict.items():
        df = pd.read_csv(csv_path)
        dfs[ch] = df
        print(f"  {ch}: {len(df)} 幀, ok={int(df['ok'].sum())}")

    # 取 ok=1 幀的基礎檔名交集
    ok_sets = {}
    for ch, df in dfs.items():
        ok_bases = set(df.loc[df["ok"] == 1, "file"].apply(_base_name).values)
        ok_sets[ch] = ok_bases

    common_ok = set.intersection(*ok_sets.values())
    print(f"\n  各通道 ok 幀數：{', '.join(f'{ch}={len(s)}' for ch, s in ok_sets.items())}")
    print(f"  交集 ok 幀數：{len(common_ok)}")

    # 計算每通道被移除的幀
    results = {}
    for ch, df in dfs.items():
        was_ok = (df["ok"] == 1)
        in_common = df["file"].apply(_base_name).isin(common_ok)
        to_remove = was_ok & ~in_common
        n_removed = int(to_remove.sum())

        if n_removed > 0:
            removed_files = df.loc[to_remove, "file"].tolist()
            print(f"\n  {ch}: 移除 {n_removed} 幀（不在交集中）：")
            for f in removed_files[:5]:
                print(f"    - {f}")
            if len(removed_files) > 5:
                print(f"    ... 還有 {len(removed_files)-5} 幀")

        if not dry_run:
            df_sync = df.copy()
            df_sync.loc[to_remove, "ok"] = 0
            if "ok_flag" in df_sync.columns:
                df_sync.loc[to_remove, "ok_flag"] = "cross_channel_sync"
            else:
                df_sync["ok_flag"] = ""
                df_sync.loc[to_remove, "ok_flag"] = "cross_channel_sync"
            results[ch] = (df_sync, n_removed)
        else:
            results[ch] = (df, n_removed)

    return results, common_ok


def recompute_stats(df, channel_name):
    """重新計算同步後的基本統計"""
    ok_df = df[df["ok"] == 1]
    if len(ok_df) == 0:
        return {"channel": channel_name, "n_ok": 0, "m_var_std": np.nan, "m_var_mean": np.nan}

    return {
        "channel": channel_name,
        "n_ok": len(ok_df),
        "m_var_mean": ok_df["m_var"].mean(),
        "m_var_std": ok_df["m_var"].std(),
    }


def main():
    parser = argparse.ArgumentParser(description="同步多通道 ok 幀集合")
    parser.add_argument("run_root", type=str, help="run 根目錄路徑")
    parser.add_argument("--channels", nargs="+", default=["G1", "G2"],
                        help="要同步的通道（預設 G1 G2）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只分析，不修改 CSV")
    parser.add_argument("--save-suffix", type=str, default="_synced",
                        help="同步後 CSV 的後綴（預設 _synced）")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    if not run_root.exists():
        print(f"路徑不存在：{run_root}")
        sys.exit(1)

    print("=" * 60)
    print("  多通道 ok 幀同步  sync_ok_frames.py")
    print("=" * 60)
    print(f"  Run: {run_root}")
    print(f"  通道：{args.channels}")
    print(f"  模式：{'dry-run' if args.dry_run else '寫入同步 CSV'}")
    print()

    # 找 CSV
    csv_dict = find_photometry_csvs(run_root, args.channels)
    if len(csv_dict) < 2:
        print(f"找到的通道不足 2 個：{list(csv_dict.keys())}")
        sys.exit(1)

    missing = set(args.channels) - set(csv_dict.keys())
    if missing:
        print(f"[WARN] 找不到以下通道的 CSV：{missing}")

    # 同步
    results, common_ok = sync_ok_frames(csv_dict, dry_run=args.dry_run)

    # 統計
    print(f"\n{'='*60}")
    print("  同步後統計：")
    stats = []
    for ch in args.channels:
        if ch in results:
            df_sync, n_removed = results[ch]
            s = recompute_stats(df_sync, ch)
            s["n_removed"] = n_removed
            stats.append(s)
            print(f"  {ch}: ok={s['n_ok']}, removed={n_removed}, "
                  f"m_var std={s['m_var_std']:.4f}")

    # G_avg 計算（如果有 G1 和 G2）
    if "G1" in results and "G2" in results:
        df_g1, _ = results["G1"]
        df_g2, _ = results["G2"]
        # 合併 ok=1 的幀
        ok_g1 = df_g1[df_g1["ok"] == 1][["file", "m_var"]].copy()
        ok_g1["base"] = ok_g1["file"].apply(_base_name)
        ok_g1 = ok_g1.rename(columns={"m_var": "m_var_G1"})
        ok_g2 = df_g2[df_g2["ok"] == 1][["file", "m_var"]].copy()
        ok_g2["base"] = ok_g2["file"].apply(_base_name)
        ok_g2 = ok_g2.rename(columns={"m_var": "m_var_G2"})
        merged = ok_g1.merge(ok_g2, on="base", how="inner")
        if len(merged) > 0:
            merged["m_var_Gavg"] = (merged["m_var_G1"] + merged["m_var_G2"]) / 2
            corr = merged["m_var_G1"].corr(merged["m_var_G2"])
            print(f"\n  G1-G2 corr (synced): {corr:.4f}")
            print(f"  G_avg std: {merged['m_var_Gavg'].std():.4f}")

    # 寫入
    if not args.dry_run:
        for ch, csv_path in csv_dict.items():
            if ch in results:
                df_sync, n_removed = results[ch]
                out_path = csv_path.parent / (csv_path.stem + args.save_suffix + ".csv")
                df_sync.to_csv(out_path, index=False, encoding="utf-8-sig")
                print(f"\n  [CSV] {out_path}")

    print(f"\n{'='*60}")
    print("  完成。")


if __name__ == "__main__":
    main()
