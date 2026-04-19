"""
從現有 CSV 重新產生圖檔，不重跑測光。
產生：光變曲線 × 4 通道、3 欄式週期分析圖 × 4 通道
"""
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

script_dir = Path(__file__).parent.absolute()
pipeline_dir = script_dir.parent
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

try:
    from polt_light_curve import plot_light_curve
    from photometry import load_pipeline_config, cfg_from_yaml
    from period_analysis import run_period_analysis
    print("匯入模組成功。")
except ImportError:
    traceback.print_exc()
    sys.exit(1)
except Exception:
    traceback.print_exc()
    sys.exit(1)


TARGET    = "V1162Ori"
DATE_STR  = "20251220"
CHANNELS  = ["R", "G1", "G2", "B"]
BASE      = Path("d:/VarStar")
OUT_DIR   = BASE / "data/targets" / TARGET / "output"
CONFIG    = pipeline_dir / "observation_config.yaml"


def replot():
    yaml_dict = load_pipeline_config(CONFIG)

    for ch in CHANNELS:
        csv_path = OUT_DIR / f"photometry_{ch}_{DATE_STR}.csv"
        if not csv_path.exists():
            print(f"[SKIP] CSV 不存在：{csv_path}")
            continue

        df = pd.read_csv(csv_path)
        if "t_sigma_mag" in df.columns and "v_err" not in df.columns:
            df["v_err"] = df["t_sigma_mag"]
        cfg = cfg_from_yaml(yaml_dict, TARGET, DATE_STR, channel=ch)
        cfg.display_name = "V1162 ORI"

        # ── 光變曲線 ──────────────────────────────────────────────
        lc_png = OUT_DIR / f"light_curve_{ch}_{DATE_STR}.png"
        print(f"[光變曲線] {ch} → {lc_png.name}")
        try:
            plot_light_curve(df, lc_png, ch, cfg, obs_date=DATE_STR)
        except Exception:
            print(f"  光變曲線失敗：")
            traceback.print_exc()

        # ── 3 欄式週期分析圖（detrend + Breger S/N） ─────────────
        print(f"[週期分析] {ch} → period_analysis_V1162Ori_{ch}.png")
        try:
            run_period_analysis(
                df, TARGET, ch, OUT_DIR,
                config_path=CONFIG,
                run_prewhitening=True,
                obs_date=DATE_STR,
            )
        except Exception:
            print(f"  週期分析失敗：")
            traceback.print_exc()

    print("\n完成。")


if __name__ == "__main__":
    replot()
