"""
fog_detect.py — 起霧偵測門檻探索工具
=====================================
用途：讀取 photometry CSV，對每一幀逐一測試各起霧判準，
      輸出每幀狀態表格供人眼核對，協助找出最適合的分界門檻。

使用方式
--------
  python fog_detect.py --csv <photometry_csv> [--out <output_dir>]
  python fog_detect.py --target V1162Ori --date 20251220 --channel G1

輸出
----
  fog_frame_table_<ch>_<date>.csv   每幀 × 各判準 pass/fail 明細
  fog_threshold_sweep_<ch>_<date>.png  各指標分布圖（直方圖 + 門檻線）
  fog_summary_<ch>_<date>.csv       各門檻值下的通過/剔除幀數統計
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 各判準定義 ────────────────────────────────────────────────────────────────
# 每個判準：欄位名稱、方向（">" 代表「大於門檻則剔除」）、掃描範圍
CRITERIA = {
    "fwhm": {
        "col": "frame_fwhm_median",
        "direction": ">",          # fwhm 過大 → 起霧/模糊
        "sweep": np.arange(2.0, 10.0, 0.5),
        "unit": "px",
        "label": "Frame FWHM",
    },
    "sharpness": {
        "col": "sharpness_index",
        "direction": "<",          # sharpness 過低 → 起霧
        "sweep": np.arange(0.05, 0.60, 0.05),
        "unit": "",
        "label": "Sharpness Index",
    },
    "peak_ratio": {
        "col": "peak_ratio",
        "direction": "<",          # peak_ratio 過低 → 甜甜圈/次鏡起霧
        "sweep": np.arange(0.05, 0.80, 0.05),
        "unit": "",
        "label": "Peak Ratio (peak/flux)",
    },
    "sky": {
        "col": "t_b_sky",
        "direction": ">",          # 天空背景突升 → 起霧/散射光
        "sweep": None,             # 動態產生（基於資料分位數）
        "unit": "DN",
        "label": "Sky Background",
    },
    "ap_radius": {
        "col": "ap_radius",
        "direction": ">",          # 孔徑半徑膨脹 → PSF 擴散（起霧）
        "sweep": None,             # 動態產生
        "unit": "px",
        "label": "Aperture Radius",
    },
}


def _dynamic_sweep(series: pd.Series, n: int = 15) -> np.ndarray:
    """從資料分位數自動產生門檻掃描範圍。"""
    valid = series.dropna()
    if len(valid) < 3:
        return np.array([])
    lo, hi = np.percentile(valid, [5, 95])
    if lo >= hi:
        return np.array([lo])
    return np.linspace(lo, hi, n)


def build_frame_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    回傳每幀 × 各判準的詳細表格。
    各判準欄位格式：{criterion}_flag（1=被此判準剔除，0=通過）
    另附各判準的原始值欄位。
    """
    base_cols = ["file", "bjd_tdb", "airmass", "ok_flag"]
    out = df[[c for c in base_cols if c in df.columns]].copy()

    for name, info in CRITERIA.items():
        col = info["col"]
        if col not in df.columns:
            out[col] = np.nan
            out[f"{name}_flag"] = np.nan
            continue
        out[col] = df[col]

    return out


def threshold_sweep(df: pd.DataFrame) -> dict:
    """
    對每個判準，在一組門檻值下計算被剔除幀數。
    回傳 dict：{criterion: DataFrame(threshold, n_rejected, pct_rejected)}
    """
    results = {}
    n_total = len(df)

    for name, info in CRITERIA.items():
        col = info["col"]
        if col not in df.columns:
            continue
        series = df[col]

        sweep = info["sweep"]
        if sweep is None:
            sweep = _dynamic_sweep(series)
        if len(sweep) == 0:
            continue

        rows = []
        direction = info["direction"]
        for thr in sweep:
            if direction == ">":
                mask = series > thr
            else:
                mask = series < thr
            n_rej = int(mask.sum())
            rows.append({
                "threshold": round(float(thr), 4),
                "n_rejected": n_rej,
                "pct_rejected": round(100.0 * n_rej / n_total, 1) if n_total > 0 else 0.0,
            })
        results[name] = pd.DataFrame(rows)

    return results


def flag_frames(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    依指定門檻對每幀標記各判準 pass/fail。

    Parameters
    ----------
    df         : photometry CSV DataFrame
    thresholds : {criterion: threshold_value}，未指定的判準跳過

    Returns
    -------
    DataFrame，含原始指標值 + 各判準 flag + 綜合 fog_score
    """
    out = build_frame_table(df)

    fog_score = np.zeros(len(df), dtype=int)
    for name, thr in thresholds.items():
        info = CRITERIA.get(name)
        if info is None:
            continue
        col = info["col"]
        if col not in df.columns:
            continue
        series = df[col]
        direction = info["direction"]
        if direction == ">":
            flag = (series > thr).astype(int)
        else:
            flag = (series < thr).astype(int)
        out[f"{name}_flag"] = flag.values
        fog_score += flag.values

    out["fog_score"] = fog_score
    return out


def plot_distributions(df: pd.DataFrame, out_path: Path) -> None:
    """輸出各判準指標分布直方圖（含建議門檻線），供人眼判斷。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[fog_detect] matplotlib 未安裝，跳過繪圖。")
        return

    n_criteria = len(CRITERIA)
    fig, axes = plt.subplots(1, n_criteria, figsize=(4 * n_criteria, 4), dpi=120)
    if n_criteria == 1:
        axes = [axes]

    fig.suptitle("Fog Detection — Metric Distributions", fontsize=13,
                 fontweight="bold", fontfamily="Arial")

    for ax, (name, info) in zip(axes, CRITERIA.items()):
        col = info["col"]
        if col not in df.columns or df[col].dropna().empty:
            ax.set_title(f"{info['label']}\n(no data)")
            ax.axis("off")
            continue

        vals = df[col].dropna()
        ax.hist(vals, bins=30, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.set_title(f"{info['label']}", fontsize=9)
        ax.set_xlabel(f"{col} [{info['unit']}]" if info["unit"] else col, fontsize=8)
        ax.set_ylabel("Frames", fontsize=8)
        ax.tick_params(labelsize=7)

        # 標記 5th / 50th / 95th 分位數
        for q, color, ls in [(5, "orange", "--"), (50, "gray", ":"), (95, "red", "--")]:
            qv = np.percentile(vals, q)
            ax.axvline(qv, color=color, lw=1, ls=ls,
                       label=f"p{q}={qv:.2f}")
        ax.legend(fontsize=6)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[fog_detect] 分布圖已儲存：{out_path}")


def run(csv_path: Path, out_dir: Path, thresholds: dict | None = None) -> None:
    """
    主執行函式。

    輸出：
    1. fog_frame_table_*.csv     — 每幀 × 各判準明細（含 fog_score）
    2. fog_threshold_sweep_*.csv — 各判準門檻掃描統計
    3. fog_distributions_*.png   — 分布圖
    """
    if not csv_path.exists():
        print(f"[fog_detect] 找不到檔案：{csv_path}")
        return

    stem = csv_path.stem  # e.g. photometry_G1_20251220
    df = pd.read_csv(csv_path)
    print(f"[fog_detect] 載入 {csv_path.name}：{len(df)} 幀")

    # 1. 分布圖
    plot_distributions(df, out_dir / f"fog_distributions_{stem}.png")

    # 2. 門檻掃描
    sweep_results = threshold_sweep(df)
    for name, tbl in sweep_results.items():
        out_csv = out_dir / f"fog_sweep_{name}_{stem}.csv"
        tbl.to_csv(out_csv, index=False)
        print(f"[fog_detect] 門檻掃描 [{name}] → {out_csv.name}")

    # 3. 每幀明細表格
    if thresholds is None:
        # 預設：使用各指標 90th 分位數（方向：>）或 10th（方向：<）作為示範門檻
        thresholds = {}
        for name, info in CRITERIA.items():
            col = info["col"]
            if col not in df.columns or df[col].dropna().empty:
                continue
            vals = df[col].dropna()
            if info["direction"] == ">":
                thresholds[name] = float(np.percentile(vals, 90))
            else:
                thresholds[name] = float(np.percentile(vals, 10))

    frame_tbl = flag_frames(df, thresholds)
    out_csv = out_dir / f"fog_frame_table_{stem}.csv"
    frame_tbl.to_csv(out_csv, index=False)
    print(f"[fog_detect] 幀明細表格 → {out_csv}")

    # 摘要
    n_fog = int((frame_tbl["fog_score"] >= 2).sum())
    print(f"[fog_detect] 使用示範門檻：fog_score>=2 的幀數 = {n_fog}/{len(df)}")
    print(f"[fog_detect] 示範門檻：{thresholds}")
    print("[fog_detect] 請用人眼核對 fog_frame_table_*.csv，調整門檻後再重跑。")


# ── CLI 入口 ─────────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser(description="起霧偵測門檻探索工具")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", type=Path, help="直接指定 photometry CSV 路徑")
    g.add_argument("--target", help="目標星名稱（需搭配 --date --channel）")
    p.add_argument("--date", default=None)
    p.add_argument("--channel", default="G1")
    p.add_argument("--out", type=Path, default=None, help="輸出目錄（預設與 CSV 同目錄）")
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "observation_config.yaml")
    p.add_argument(
        "--thresholds", nargs="*", metavar="KEY=VALUE",
        help="手動指定門檻，例如：fwhm=5.0 sharpness=0.15",
    )
    return p.parse_args()


def _parse_thresholds(args_list: list[str] | None) -> dict | None:
    if not args_list:
        return None
    out = {}
    for item in args_list:
        k, _, v = item.partition("=")
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            print(f"[fog_detect] 警告：無法解析門檻 '{item}'，跳過。")
    return out or None


if __name__ == "__main__":
    args = _parse_args()

    if args.csv:
        csv_path = args.csv
    else:
        # 從 yaml 找路徑
        import yaml
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        root = Path(cfg["paths"]["local"]["project_root"])
        date = args.date or "unknown"
        csv_path = root / "data" / "targets" / args.target / "output" / \
                   f"photometry_{args.channel}_{date}.csv"

    out_dir = args.out or csv_path.parent
    thresholds = _parse_thresholds(args.thresholds)

    run(csv_path, out_dir, thresholds)
