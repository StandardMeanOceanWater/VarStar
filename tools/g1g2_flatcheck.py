# -*- coding: utf-8 -*-
"""
g1g2_flatcheck.py — CC And G1/G2 比值三幀比對
=============================================
取第 1、中間、最後一幀的 G1/G2 splits，
比較 raw 和 FITS master flat 校正後的空間分布圖。

用法：
  python tools/g1g2_flatcheck.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# ── 路徑設定 ──────────────────────────────────────────────────────────────────
# 已校正全幀 FITS（由之前的 flat_check 腳本產生）
FRAMES = [
    (Path(r"D:\JUNK\CC AND\flat_check\frame_001_flatNEW.fits"), "Frame 1（最早）"),
    (Path(r"D:\JUNK\CC AND\flat_check\frame_041_flatNEW.fits"), "Frame 41（中間）"),
    (Path(r"D:\JUNK\CC AND\flat_check\frame_081_flatNEW.fits"), "Frame 81（最晚）"),
]
FLAT_FITS = Path(r"D:\VarStar\data\share\calibration\20251122_R200SS_6D2\masters\master_flat_norm_20251122_fits.fits")
OUT_PNG   = Path(r"D:\VarStar\output\g1g2_flatcheck_3frames.png")

BLOCK = 50   # 區塊大小（像素）

# ── 工具函式 ──────────────────────────────────────────────────────────────────

def load_frame(path: Path) -> np.ndarray:
    with fits.open(path) as h:
        return h[0].data.astype(np.float32)


def debayer_g1g2(frame: np.ndarray):
    """RGGB Bayer 拆 G1（偶列奇行）和 G2（奇列偶行）"""
    g1 = frame[0::2, 1::2]  # row=even, col=odd
    g2 = frame[1::2, 0::2]  # row=odd,  col=even
    return g1, g2


def block_ratio(g1: np.ndarray, g2: np.ndarray, block: int = 50) -> np.ndarray:
    """50×50 像素區塊的 G1/G2 中值比值矩陣"""
    ny, nx = g1.shape
    rows = ny // block
    cols = nx // block
    ratio = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            g1b = g1[r*block:(r+1)*block, c*block:(c+1)*block]
            g2b = g2[r*block:(r+1)*block, c*block:(c+1)*block]
            med_g2 = np.median(g2b)
            ratio[r, c] = np.median(g1b) / med_g2 if med_g2 > 0 else np.nan
    return ratio


def apply_flat(img: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """flat 歸一化後除法校正"""
    flat_norm = flat / np.median(flat)
    flat_norm[flat_norm <= 0] = np.nan
    return img / flat_norm


def plot_ratio(ax, ratio, title, vmin=0.990, vmax=1.010):
    std = np.nanstd(ratio)
    im = ax.imshow(ratio, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                   interpolation="nearest", aspect="auto")
    ax.set_title(f"{title}\nstd={std:.5f}", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    print(f"載入 master flat：{FLAT_FITS.name}")
    flat_raw = load_frame(FLAT_FITS)   # 全幅 Bayer or 灰階

    fig, axes = plt.subplots(len(FRAMES), 2,
                             figsize=(11, 4 * len(FRAMES)),
                             constrained_layout=True)
    fig.suptitle("CC And G1/G2 比值空間分布（已校正幀 DeBayer）\n"
                 f"{BLOCK}×{BLOCK} px 區塊中值比", fontsize=11)

    for row_i, (fpath, label) in enumerate(FRAMES):
        frame = load_frame(fpath)
        g1, g2 = debayer_g1g2(frame)

        # flat 尺寸判斷
        if flat_raw.shape == g1.shape:
            flat_g1, flat_g2 = flat_raw, flat_raw
        elif flat_raw.shape[0] == g1.shape[0] * 2:
            flat_g1 = flat_raw[0::2, 1::2]
            flat_g2 = flat_raw[1::2, 0::2]
        else:
            print(f"  [WARN] flat {flat_raw.shape} vs G1 {g1.shape}，跳過額外 flat 校正")
            flat_g1 = np.ones_like(g1)
            flat_g2 = np.ones_like(g2)

        r_raw  = block_ratio(g1, g2, BLOCK)
        g1c    = apply_flat(g1, flat_g1)
        g2c    = apply_flat(g2, flat_g2)
        r_corr = block_ratio(g1c, g2c, BLOCK)

        vmin = min(np.nanpercentile(r_raw, 1),  np.nanpercentile(r_corr, 1))
        vmax = max(np.nanpercentile(r_raw, 99), np.nanpercentile(r_corr, 99))

        ax_raw, ax_corr = axes[row_i]
        plot_ratio(ax_raw,  r_raw,  f"{label}  Raw G1/G2",  vmin, vmax)
        plot_ratio(ax_corr, r_corr, f"{label}  +FITS flat", vmin, vmax)

        print(f"  {label}: raw std={np.nanstd(r_raw):.5f}  corr std={np.nanstd(r_corr):.5f}")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"\n儲存：{OUT_PNG}")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
