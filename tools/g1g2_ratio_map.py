# -*- coding: utf-8 -*-
"""
g1g2_ratio_map.py — G1/G2 空間比值圖（通用版）
================================================
從已拆色的 G1、G2 目錄讀取 FITS，計算區塊中值比值的空間分布圖。
自動取樣 N 幀（頭、中、尾），檢查 CRA 微透鏡效應的穩定性。

用法：
  python tools/g1g2_ratio_map.py --g1 <G1_dir> --g2 <G2_dir>
  python tools/g1g2_ratio_map.py --g1 <G1_dir> --g2 <G2_dir> --block 25 --frames 5
  python tools/g1g2_ratio_map.py --g1 <G1_dir> --g2 <G2_dir> --out ratio.png

block 建議：
  100  粗覽（快速，低噪聲）
   50  標準（預設，平衡解析度與噪聲）
   25  精細（看微透鏡結構，但噪聲較高）
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits

matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def load_fits(path: Path) -> np.ndarray:
    with fits.open(path) as h:
        return h[0].data.astype(np.float32)


def block_ratio(g1: np.ndarray, g2: np.ndarray, block: int) -> np.ndarray:
    """區塊中值比值矩陣"""
    ny, nx = g1.shape
    rows, cols = ny // block, nx // block
    ratio = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            g1b = g1[r * block:(r + 1) * block, c * block:(c + 1) * block]
            g2b = g2[r * block:(r + 1) * block, c * block:(c + 1) * block]
            med_g2 = np.median(g2b)
            ratio[r, c] = np.median(g1b) / med_g2 if med_g2 > 0 else np.nan
    return ratio


def match_g1g2_files(g1_dir: Path, g2_dir: Path):
    """配對 G1/G2 檔案（按共同前綴匹配）"""
    g1_files = sorted(g1_dir.glob("*.fits"))
    g2_files = sorted(g2_dir.glob("*.fits"))

    # 嘗試用檔名去掉 _G1/_G2 後綴來配對
    g2_map = {}
    for f in g2_files:
        key = f.stem.replace("_G2", "").replace("_g2", "")
        g2_map[key] = f

    pairs = []
    for f1 in g1_files:
        key = f1.stem.replace("_G1", "").replace("_g1", "")
        if key in g2_map:
            pairs.append((f1, g2_map[key]))

    if not pairs:
        # fallback：按排序順序配對
        n = min(len(g1_files), len(g2_files))
        pairs = list(zip(g1_files[:n], g2_files[:n]))

    return pairs


def sample_indices(total: int, n_frames: int):
    """均勻取樣 n_frames 個索引（含頭尾）"""
    if total <= n_frames:
        return list(range(total))
    if n_frames == 1:
        return [0]
    return [round(i * (total - 1) / (n_frames - 1)) for i in range(n_frames)]


def main():
    parser = argparse.ArgumentParser(
        description="G1/G2 空間比值圖 — 檢查 CRA 微透鏡效應"
    )
    parser.add_argument("--g1", required=True, help="G1 splits 目錄")
    parser.add_argument("--g2", required=True, help="G2 splits 目錄")
    parser.add_argument("--block", type=int, default=50,
                        help="區塊大小（像素），預設 50")
    parser.add_argument("--frames", type=int, default=3,
                        help="取樣幀數，預設 3（頭、中、尾）")
    parser.add_argument("--out", type=str, default=None,
                        help="輸出 PNG 路徑（預設存在 G1 目錄的上層）")
    args = parser.parse_args()

    g1_dir = Path(args.g1)
    g2_dir = Path(args.g2)

    if not g1_dir.is_dir():
        print(f"[ERROR] G1 目錄不存在：{g1_dir}")
        sys.exit(1)
    if not g2_dir.is_dir():
        print(f"[ERROR] G2 目錄不存在：{g2_dir}")
        sys.exit(1)

    pairs = match_g1g2_files(g1_dir, g2_dir)
    if not pairs:
        print("[ERROR] 找不到可配對的 G1/G2 FITS 檔案")
        sys.exit(1)

    print(f"配對成功：{len(pairs)} 幀，block={args.block}px")

    indices = sample_indices(len(pairs), args.frames)
    n = len(indices)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]

    # 先算全部 ratio 以統一色階
    ratios = []
    labels = []
    for idx in indices:
        f1, f2 = pairs[idx]
        g1 = load_fits(f1)
        g2 = load_fits(f2)
        ratio = block_ratio(g1, g2, args.block)
        ratios.append(ratio)
        labels.append(f"Frame {idx}/{len(pairs)-1}\n{f1.stem[:19]}")
        print(f"  [{idx:3d}] std={np.nanstd(ratio):.5f}  "
              f"center={np.nanmedian(ratio):.5f}")

    vmin = min(np.nanpercentile(r, 1) for r in ratios)
    vmax = max(np.nanpercentile(r, 99) for r in ratios)
    # 對稱色階
    dev = max(abs(1.0 - vmin), abs(vmax - 1.0))
    vmin, vmax = 1.0 - dev, 1.0 + dev

    for ax, ratio, label in zip(axes, ratios, labels):
        std = np.nanstd(ratio)
        im = ax.imshow(ratio, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                       interpolation="nearest", aspect="auto")
        ax.set_title(f"{label}\nstd={std:.5f}", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    parent_name = g1_dir.parent.name
    fig.suptitle(f"G1/G2 ratio map — {parent_name}  "
                 f"(block={args.block}px, {len(pairs)} frames total)",
                 fontsize=11)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = g1_dir.parent / f"g1g2_ratio_block{args.block}.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\n儲存：{out_path}")


if __name__ == "__main__":
    if sys.stdout and hasattr(sys.stdout, "encoding"):
        if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
