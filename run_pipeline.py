# -*- coding: utf-8 -*-
"""
run_pipeline.py  —  變星測光管線整合入口
專案：變星測光管線 v0.99

描述
----
依序執行管線的步驟：
    Step 1  calibration      — Bias/Dark/Flat 校正（Calibration.py）
    Step 2  plate_solve      — 星圖解算，寫入 WCS（plate_solve.py）
    Step 3  debayer          — Bayer 拆色，傳遞 WCS（DeBayer_RGGB.py）
    Step 4  photometry       — 差分測光，輸出光變曲線（Photometry.ipynb）
    Step 5  period_analysis  — 進階週期分析（period_analysis.py）【選用】
    Step 6  report           — 品質報告（quality_report.py，預留介面）

使用方式
--------
    # 標準全流程（步驟 1–4 + 6，不含進階週期分析）
    python run_pipeline.py --config observation_config.yaml

    # 只執行指定步驟（可多選，空格分隔）
    python run_pipeline.py --config observation_config.yaml \\
        --steps calibration plate_solve

    # 從第 N 步開始執行到底
    python run_pipeline.py --config observation_config.yaml --from-step debayer

    # 執行進階週期分析（需先完成 photometry，output/photometry_*.csv 存在）
    python run_pipeline.py --config observation_config.yaml \\
        --steps period_analysis

    # 列出可用步驟
    python run_pipeline.py --list-steps

    # 任一步驟失敗時立即停止
    python run_pipeline.py --config observation_config.yaml --stop-on-error

限制
----
Step 4（photometry）必須在 Jupyter 環境中互動執行（生長曲線診斷、
零點散佈圖需要人工確認）。此腳本輸出提示後跳過，不強制執行。

Step 5（period_analysis）為選用進階模組，預設不在全流程中執行。
需透過 --steps period_analysis 明確指定。
前置條件：Step 4 已完成，output/photometry_*.csv 存在。
輸入：output/photometry_*.csv
輸出：output/period_analysis_*.{csv,png}

Step 6（report）為預留介面，目前輸出 [SKIP] 提示。

設計規格
--------
DESIGN_DECISIONS_v5.md，§4「各模組職責」。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path


# ── 步驟定義（名稱, 說明, 是否預設執行）─────────────────────────────────────
#
# period_analysis 標記 default=False：
#   執行「全部步驟」時自動排除，必須透過 --steps 或 --from-step 明確指定。
#   理由：此步驟依賴 Step 4 的 CSV 輸出，而全自動執行時 Step 4 被跳過（需
#   人工互動）。若 CSV 不存在，步驟會以前置條件不符的方式失敗並說明原因。
#
_STEPS: list[tuple[str, str, bool]] = [
    (
        "calibration",
        "Bias/Dark/Flat 校正  →  calibrated/*.fits",
        True,
    ),
    (
        "plate_solve",
        "星圖解算（ASTAP/astrometry.net）  →  calibrated/wcs/*_wcs.fits",
        True,
    ),
    (
        "debayer",
        "Bayer 拆色，傳遞 WCS  →  split/{R,G1,G2,B}/*.fits",
        True,
    ),
    (
        "photometry",
        "差分測光，光變曲線  →  output/photometry_*.csv",
        True,
    ),
    (
        "period_analysis",
        "【進階選用】週期分析（LS + DFT + 預白化）  →  output/period_analysis_*",
        False,   # 預設不執行，需明確指定
    ),
    (
        "report",
        "觀測品質報告（預留介面，待實作）",
        True,
    ),
]
_STEP_NAMES: list[str] = [s[0] for s in _STEPS]
_STEP_DEFAULTS: list[str] = [s[0] for s in _STEPS if s[2]]


# =============================================================================
# 步驟執行函式
# =============================================================================

def _run_calibration(config_path: Path) -> bool:
    """
    呼叫 Calibration.run_calibration()。

    Returns
    -------
    bool
        True = 成功，False = 例外。
    """
    try:
        from Calibration import run_calibration
        run_calibration(config_path)
        return True
    except ImportError as exc:
        print(
            f"[ERROR] 無法匯入 Calibration.py：{exc}\n"
            "        請確認 Calibration.py 與本腳本在同一目錄，\n"
            "        或已加入 PYTHONPATH。"
        )
        return False
    except Exception as exc:
        print(f"[ERROR] calibration 步驟失敗：{exc}")
        return False


def _run_plate_solve(config_path: Path, auto_yes: bool = False) -> bool:
    """呼叫 plate_solve.run_plate_solve()，完成後自動執行 VSX 查詢。"""
    try:
        from plate_solve import run_plate_solve
        run_plate_solve(config_path)
    except ImportError as exc:
        print(
            f"[ERROR] 無法匯入 plate_solve.py：{exc}\n"
            "        請確認 plate_solve.py 與本腳本在同一目錄。"
        )
        return False
    except Exception as exc:
        print(f"[ERROR] plate_solve 步驟失敗：{exc}")
        return False

    # plate_solve 成功後自動執行 VSX 查詢（失敗不中斷 pipeline）
    _run_vsx_query(config_path, auto_yes=auto_yes)
    return True


def _run_vsx_query(config_path: Path, auto_yes: bool = False) -> None:
    """
    plate_solve 完成後，從 WCS 結果取視場中心，查詢 AAVSO VSX。

    流程
    ----
    1. 讀 yaml，找出所有 session 的 WCS 目錄
    2. 讀 CRVAL1/CRVAL2，取中位數作為視場中心
    3. 呼叫 vsx_query.query_vsx()
    4. 符合條件的目標寫入 yaml（人工確認或 --yes 跳過）
    失敗只印 WARN，不中斷 pipeline。
    """
    import re

    import numpy as np
    import yaml
    from astropy.io import fits

    # ── 讀設定 ─────────────────────────────────────────────────────────────
    try:
        with config_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except Exception as exc:
        print(f"[WARN] VSX：無法讀取設定檔：{exc}")
        return

    vsx_cfg = cfg.get("vsx", {})
    if not vsx_cfg.get("auto_query", True):
        return

    radius = float(vsx_cfg.get("search_radius_deg", 1.55))
    mag_max = float(vsx_cfg.get("auto_add_mag_max", 13.0))
    add_types = [t.upper() for t in vsx_cfg.get("auto_add_types", [])]

    try:
        import google.colab  # noqa: F401
        project_root = Path(cfg["paths"]["colab"]["project_root"])
    except ImportError:
        project_root = Path(cfg["paths"]["local"]["project_root"])

    sessions = cfg.get("obs_sessions", [])
    if not sessions:
        print("[WARN] VSX：yaml 無 obs_sessions，跳過。")
        return

    existing_keys = set((cfg.get("targets") or {}).keys())

    for session in sessions:
        date = str(session.get("date", ""))
        targets = session.get("targets", [])

        for target in targets:
            wcs_dir = (
                project_root / "data" / "targets" / target
                / "calibrated" / "wcs"
            )
            wcs_files = sorted(wcs_dir.glob("*_wcs.fits")) if wcs_dir.exists() else []
            if not wcs_files:
                print(f"[WARN] VSX [{target}]：找不到 WCS 檔案，跳過。")
                continue

            # ── 取 CRVAL 中位數 ─────────────────────────────────────────
            ra_vals, dec_vals = [], []
            for wf in wcs_files:
                try:
                    hdr = fits.getheader(wf)
                    if "CRVAL1" in hdr and "CRVAL2" in hdr:
                        ra_vals.append(float(hdr["CRVAL1"]))
                        dec_vals.append(float(hdr["CRVAL2"]))
                except Exception:
                    continue

            if not ra_vals:
                print(f"[WARN] VSX [{target}]：所有 WCS 檔案均無 CRVAL，跳過。")
                continue

            center_ra  = float(np.median(ra_vals))
            center_dec = float(np.median(dec_vals))
            print(f"\n[VSX] {target} 視場中心：RA={center_ra:.4f}°  Dec={center_dec:+.4f}°  r={radius}°")

            # ── 呼叫 VSX ────────────────────────────────────────────────
            try:
                from vsx_query import query_vsx, print_table, save_csv
            except ImportError as exc:
                print(f"[WARN] VSX：無法匯入 vsx_query.py：{exc}")
                return

            out_dir = project_root / "data" / "targets" / target / "output"
            df = query_vsx(center_ra, center_dec, radius)
            if df is None or df.empty:
                print(f"[VSX] {target}：查詢結果為空。")
                continue

            print_table(df)

            # CSV 存成 vsx_candidates_{date}.csv
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / f"vsx_candidates_{date}.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"[VSX] saved → {csv_path}")

            # ── 篩選符合條件的候選 ──────────────────────────────────────
            candidates = df[df["max_mag_num"] <= mag_max].copy()
            if add_types:
                candidates = candidates[
                    candidates["var_type"].str.upper().isin(add_types)
                ]
            # 排除已在 yaml 的目標（key = name 去空白）
            def _to_key(name: str) -> str:
                return re.sub(r"\s+", "", str(name))

            new_candidates = candidates[
                ~candidates["name"].apply(_to_key).isin(existing_keys)
            ]

            if new_candidates.empty:
                print(f"[VSX] {target}：無符合條件的新目標需要加入。")
                continue

            # ── 確認提示 ────────────────────────────────────────────────
            print(f"\n[VSX] 以下 {len(new_candidates)} 筆符合條件，準備寫入 yaml：")
            for _, row in new_candidates.iterrows():
                print(f"  {row['name']:<20} {row['var_type']:<10} "
                      f"MaxMag={row['max_mag']} {row['max_band']}  "
                      f"Period={row['period']}d  AUID={row['auid']}")

            if not auto_yes:
                ans = input("\n確認寫入 observation_config.yaml？[y/N] ").strip().lower()
                if ans != "y":
                    print("[VSX] 跳過寫入。")
                    continue

            # ── 寫入 yaml（字串 append，保留現有註解）──────────────────
            _vsx_append_targets(
                config_path, new_candidates, date, existing_keys
            )
            # 更新 existing_keys 避免同一次執行重複加入
            for _, row in new_candidates.iterrows():
                existing_keys.add(_to_key(str(row["name"])))


def _vsx_append_targets(
    config_path: Path,
    candidates: "pd.DataFrame",
    obs_date: str,
    existing_keys: set,
) -> None:
    """
    將 VSX 候選目標以字串 append 方式寫入 yaml targets 區塊末尾。
    保留現有所有註解與格式，不使用 yaml.dump()。
    """
    import re

    text = config_path.read_text(encoding="utf-8")

    new_blocks = []
    for _, row in candidates.iterrows():
        raw_name = str(row["name"]).strip()
        key = re.sub(r"\s+", "", raw_name)
        if key in existing_keys:
            continue
        ra_h = row["ra_deg"] / 15.0 if row["ra_deg"] == row["ra_deg"] else 0.0
        dec  = row["dec_deg"] if row["dec_deg"] == row["dec_deg"] else 0.0
        period_str = str(row["period"]).strip()
        try:
            period_val = f"{float(period_str):.5f}"
        except (ValueError, TypeError):
            period_val = period_str or "null"
        mag_str = str(row["max_mag"]).strip()
        try:
            vmag_val = f"{float(mag_str):.2f}"
        except (ValueError, TypeError):
            vmag_val = mag_str
        block = (
            f"  {key}:  # auto_added: true — VSX 自動加入，需人工審核後正式納入\n"
            f"    ra_hint_h: {ra_h:.6f}    # {row['ra_deg']:.4f} deg\n"
            f"    dec_hint_deg: {dec:.4f}\n"
            f"    display_name: \"{raw_name}\"\n"
            f"    vmag_approx: {vmag_val}    # {row['max_band']} band\n"
            f"    var_type: \"{row['var_type']}\"\n"
            f"    period_d: {period_val}\n"
            f"    auid: \"{row['auid']}\"\n"
            f"    obs_date: \"{obs_date}\"\n"
        )
        new_blocks.append(block)

    if not new_blocks:
        return

    # 找到 targets: 區塊最後一個有效條目的結尾，插入新條目
    # 策略：在下一個頂層區塊（^[a-z]）之前插入
    insert_marker = re.compile(r"^(?=\S)", re.MULTILINE)
    # 找 targets: 之後第一個頂層 key（非 targets 本身）
    targets_pos = text.find("\ntargets:")
    if targets_pos == -1:
        targets_pos = text.find("targets:")
    after_targets = text[targets_pos:]
    next_section = re.search(r"\n(?=[a-z_]+:\s*\n)", after_targets[9:])

    if next_section:
        insert_at = targets_pos + 9 + next_section.start() + 1
        new_text = text[:insert_at] + "\n" + "".join(new_blocks) + text[insert_at:]
    else:
        new_text = text.rstrip() + "\n\n" + "".join(new_blocks) + "\n"

    config_path.write_text(new_text, encoding="utf-8")
    print(f"[VSX] {len(new_blocks)} 筆目標已寫入 {config_path.name}（標注 auto_added）")


def _run_debayer(config_path: Path) -> bool:
    """呼叫 DeBayer_RGGB.run_debayer()。"""
    try:
        from DeBayer_RGGB import run_debayer
        run_debayer(config_path)
        return True
    except ImportError as exc:
        print(
            f"[ERROR] 無法匯入 DeBayer_RGGB.py：{exc}\n"
            "        請確認 DeBayer_RGGB.py 與本腳本在同一目錄。"
        )
        return False
    except Exception as exc:
        print(f"[ERROR] debayer 步驟失敗：{exc}")
        return False


def _run_photometry(config_path: Path) -> bool:  # noqa: ARG001
    """
    Step 4 為 Jupyter notebook，需人工互動執行。

    原因：生長曲線診斷圖和零點散佈圖需要人工確認比較星選取
    結果是否合理，無法安全地全自動化（DESIGN_DECISIONS_v5.md §3.6）。

    若確認比較星已驗證，可改為批次執行：
        jupyter nbconvert --to notebook --execute \\
            photometry/Photometry.ipynb
    """
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║  Step 4 photometry 需在 Jupyter 中互動執行               ║")
    print("  ║                                                          ║")
    print("  ║  本機：                                                  ║")
    print("  ║    jupyter notebook photometry/Photometry.ipynb          ║")
    print("  ║                                                          ║")
    print("  ║  Colab：掛載 Drive 後開啟 Photometry.ipynb               ║")
    print("  ║                                                          ║")
    print("  ║  強制批次（比較星已事先驗證）：                          ║")
    print("  ║    jupyter nbconvert --to notebook --execute \\           ║")
    print("  ║      photometry/Photometry.ipynb                         ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()
    return True   # 跳過不算失敗


def _run_period_analysis(config_path: Path) -> bool:
    """
    Step 5【進階選用】呼叫 period_analysis.run_period_analysis()。

    前置條件
    --------
    Step 4（photometry）必須已完成，且 output/photometry_*.csv 存在。
    此步驟預設不在全流程中執行，需透過 --steps period_analysis 明確指定。

    學理說明
    --------
    本模組執行比 photometry.py 內建 LS 更完整的週期分析流程：
      - Lomb-Scargle 週期圖（astropy），bootstrap FAP 收斂迭代
      - DFT 交叉驗證
      - Fourier 擬合（BIC 自動選階，上限由 yaml 設定）
      - Pre-whitening 迭代（停止條件：S/N < 4，Breger et al., 1993）
      - 週期不確定度：傅立葉擬合殘差（非 LS 殘差）
    詳見 DESIGN_DECISIONS_v5.md §4.5。

    Returns
    -------
    bool
        True = 成功，False = 例外或前置條件不符。
    """
    # ── 前置條件：讀取設定檔，確認 photometry CSV 存在 ────────────────────
    try:
        import yaml
        with config_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
    except FileNotFoundError:
        print(f"[ERROR] 找不到設定檔：{config_path}")
        return False
    except yaml.YAMLError as exc:
        print(f"[ERROR] 設定檔 YAML 格式錯誤：{exc}")
        return False

    try:
        import google.colab  # noqa: F401
        project_root = Path(cfg["paths"]["colab"]["project_root"])
    except ImportError:
        project_root = Path(cfg["paths"]["local"]["project_root"])

    output_csvs = list(
        (project_root / "data" / "targets").glob(
            "*/output/photometry_*.csv"
        )
    )
    if not output_csvs:
        print()
        print("  ╔══════════════════════════════════════════════════════════╗")
        print("  ║  [SKIP] Step 5 period_analysis：前置條件不符             ║")
        print("  ║                                                          ║")
        print("  ║  找不到 output/photometry_*.csv。                        ║")
        print("  ║  請先完成 Step 4（photometry），再執行此步驟。           ║")
        print("  ║                                                          ║")
        print("  ║  確認測光完成後，執行：                                  ║")
        print("  ║    python run_pipeline.py --steps period_analysis        ║")
        print("  ╚══════════════════════════════════════════════════════════╝")
        print()
        return False

    print(f"  找到 {len(output_csvs)} 個測光 CSV，開始週期分析。")

    # ── 呼叫模組：逐 CSV 執行 run_period_analysis ────────────────────────────
    try:
        import yaml as _yaml
        import pandas as _pd
        from period_analysis import run_period_analysis

        any_ok = False
        for csv_path in sorted(output_csvs):
            # 從路徑解析 target / channel
            # 路徑格式：.../targets/{TARGET}/output/photometry_{CH}_{DATE}.csv
            parts = csv_path.parts
            try:
                tgt_idx = parts.index("targets") + 1
                target_name = parts[tgt_idx]
            except (ValueError, IndexError):
                target_name = csv_path.parent.parent.name
            stem = csv_path.stem  # photometry_G1_20251220
            channel = stem.split("_")[1] if "_" in stem else "G1"
            out_dir = csv_path.parent

            df = _pd.read_csv(csv_path)
            # 欄位對映：photometry CSV → period_analysis 所需欄位名稱
            # ok 欄已存在（1=通過，0=被篩除），直接使用；只需補 v_err
            if "t_sigma_mag" in df.columns and "v_err" not in df.columns:
                df = df.rename(columns={"t_sigma_mag": "v_err"})
            print(f"  週期分析：{target_name} [{channel}]  ({len(df)} 幀)")
            try:
                run_period_analysis(
                    df=df,
                    target_name=target_name,
                    channel=channel,
                    out_dir=out_dir,
                    config_path=config_path,
                )
                any_ok = True
            except Exception as exc_inner:
                print(f"  [WARN] {target_name}/{channel} 週期分析失敗：{exc_inner}")

        return any_ok

    except ImportError as exc:
        print(
            f"[ERROR] 無法匯入 period_analysis.py：{exc}\n"
            "        請確認 period_analysis.py 與本腳本在同一目錄。"
        )
        return False
    except Exception as exc:
        print(f"[ERROR] period_analysis 步驟失敗：{exc}")
        return False


def _run_report(config_path: Path) -> bool:  # noqa: ARG001
    """
    Step 6 report 預留介面。

    quality_report.py 屬第二批（D 組）待實作項目
    （DESIGN_DECISIONS_v5.md §6）。
    """
    print()
    print("  [SKIP] Step 6 report：quality_report.py 尚未實作（第二批 D 組）。")
    print("         待實作後此步驟將自動生效。")
    print()
    return True


_STEP_RUNNERS: dict[str, object] = {
    "calibration":     _run_calibration,
    "plate_solve":     _run_plate_solve,
    "debayer":         _run_debayer,
    "photometry":      _run_photometry,
    "period_analysis": _run_period_analysis,
    "report":          _run_report,
}


# =============================================================================
# 設定檔解析
# =============================================================================

def _resolve_config(config_arg: Path | None) -> Path:
    """
    依下列優先順序尋找 observation_config.yaml：

    1. --config 命令列參數
    2. 環境變數 VARSTAR_CONFIG
    3. 目前工作目錄
    4. 本腳本所在目錄
    5. 本腳本所在目錄的上層（pipeline/ 的上層 = project_root）
    """
    if config_arg is not None:
        if not config_arg.exists():
            raise FileNotFoundError(
                f"--config 指定的路徑不存在：{config_arg}"
            )
        return config_arg.resolve()

    env = os.environ.get("VARSTAR_CONFIG")
    if env:
        p = Path(env)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(
            f"環境變數 VARSTAR_CONFIG 指定的路徑不存在：{env}"
        )

    candidates = [
        Path.cwd() / "observation_config.yaml",
        Path(__file__).parent / "observation_config.yaml",
        Path(__file__).parent.parent / "observation_config.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()

    raise FileNotFoundError(
        "找不到 observation_config.yaml。\n"
        "請使用 --config 指定路徑，或設定環境變數 VARSTAR_CONFIG。\n"
        "搜尋過的路徑：\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


# =============================================================================
# CLI
# =============================================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description=(
            "變星測光管線整合入口。\n"
            "不指定 --steps 則執行標準步驟（calibration → plate_solve → "
            "debayer → photometry → report）。\n"
            "進階週期分析需明確指定：--steps period_analysis"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "範例：\n"
            "  # 標準全流程（不含進階週期分析）\n"
            "  python run_pipeline.py --config observation_config.yaml\n\n"
            "  # 只執行前三步\n"
            "  python run_pipeline.py --config observation_config.yaml "
            "--steps calibration plate_solve debayer\n\n"
            "  # 從拆色開始\n"
            "  python run_pipeline.py --config observation_config.yaml "
            "--from-step debayer\n\n"
            "  # 執行進階週期分析（需先完成 photometry）\n"
            "  python run_pipeline.py --config observation_config.yaml "
            "--steps period_analysis\n\n"
            "  # 列出所有步驟與說明\n"
            "  python run_pipeline.py --list-steps"
        ),
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help=(
            "observation_config.yaml 路徑。"
            "預設依序搜尋環境變數 VARSTAR_CONFIG → 目前目錄 → pipeline/ 上一層。"
        ),
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=_STEP_NAMES,
        metavar="STEP",
        default=None,
        help=(
            f"指定步驟（可多選）。可用值：{', '.join(_STEP_NAMES)}。"
            "不指定則執行標準步驟（period_analysis 除外）。"
        ),
    )
    parser.add_argument(
        "--from-step",
        choices=_STEP_NAMES,
        metavar="STEP",
        default=None,
        dest="from_step",
        help=(
            "從此步驟開始執行到最後一步。與 --steps 互斥。"
            "注意：period_analysis 仍不會自動加入，除非起始步驟就是它。"
        ),
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        dest="list_steps",
        help="列出所有可用步驟後退出。",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        dest="stop_on_error",
        help="任一步驟失敗時立即中止後續步驟（預設：繼續執行）。",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        dest="auto_yes",
        help="VSX 自動新增目標時跳過確認提示（預設：互動確認）。",
    )
    return parser.parse_args(argv)


def _select_steps(args: argparse.Namespace) -> list[str]:
    """
    根據 --steps 或 --from-step 決定步驟清單。

    規則
    ----
    - 未指定任何選項  → 執行 _STEP_DEFAULTS（period_analysis 不在其中）
    - --steps 明確指定 → 照指定執行，含 period_analysis 亦合法
    - --from-step 指定 → 從該步驟到最後，但仍跳過 period_analysis
                         （除非起點本身就是 period_analysis）
    - --steps 與 --from-step 互斥

    保持 _STEP_NAMES 的固定執行順序，使用者亂序指定時自動排序。
    """
    if args.steps and args.from_step:
        raise ValueError("--steps 與 --from-step 互斥，請擇一使用。")

    if args.steps:
        requested = set(args.steps)
        return [s for s in _STEP_NAMES if s in requested]

    if args.from_step:
        start = _STEP_NAMES.index(args.from_step)
        candidates = _STEP_NAMES[start:]
        # period_analysis 仍需明確指定，除非它就是起始步驟
        return [
            s for s in candidates
            if s != "period_analysis" or s == args.from_step
        ]

    return list(_STEP_DEFAULTS)


# =============================================================================
# 主函式
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    """
    管線主函式。

    Returns
    -------
    int
        0 = 全部成功，1 = 至少一步失敗或有步驟未執行。
    """
    args = _parse_args(argv)

    # ── --list-steps ─────────────────────────────────────────────────────────
    if args.list_steps:
        print("\n可用步驟（依執行順序）：\n")
        for name, desc, default in _STEPS:
            tag = "（預設執行）" if default else "【進階選用，需明確指定】"
            print(f"  {name:<18}  {tag}")
            print(f"  {'':<18}  {desc}")
            print()
        return 0

    # ── 找設定檔 ──────────────────────────────────────────────────────────────
    try:
        config_path = _resolve_config(args.config)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    # ── 決定步驟 ──────────────────────────────────────────────────────────────
    try:
        steps_to_run = _select_steps(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    # ── 執行 ──────────────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print("  變星測光管線  run_pipeline.py")
    print(f"  設定檔  ：{config_path}")
    print(f"  執行步驟：{', '.join(steps_to_run)}")
    if "period_analysis" in steps_to_run:
        print("  ※ period_analysis 為進階選用模組，需先完成 photometry。")
    print(f"  開始時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(sep)

    results: dict[str, bool] = {}
    pipeline_start = time.monotonic()
    aborted = False

    for step in steps_to_run:
        step_desc = next(desc for name, desc, _ in _STEPS if name == step)
        print(f"\n{'─' * 60}")
        print(f"  ▶  {step.upper():<18} {step_desc}")
        print("─" * 60)

        step_start = time.monotonic()
        runner = _STEP_RUNNERS[step]
        if step == "plate_solve":
            ok: bool = runner(config_path, auto_yes=args.auto_yes)
        else:
            ok: bool = runner(config_path)
        elapsed = time.monotonic() - step_start

        status = "✓ 完成" if ok else "✗ 失敗"
        print(f"\n  {status}  （耗時 {elapsed:.1f} 秒）")
        results[step] = ok

        if not ok and args.stop_on_error:
            print(f"\n[STOP] --stop-on-error：{step} 失敗，中止後續步驟。")
            aborted = True
            break

    # ── 彙總報告 ──────────────────────────────────────────────────────────────
    total_elapsed = time.monotonic() - pipeline_start
    print(f"\n{sep}")
    print("  管線執行彙總")
    print(sep)

    all_ok = True
    for step in steps_to_run:
        if step not in results:
            print(f"  {'─':>3}  {step:<18}  （未執行）")
            all_ok = False
        else:
            mark = "✓" if results[step] else "✗"
            print(f"  {mark:>3}  {step:<18}")
            if not results[step]:
                all_ok = False

    minutes, seconds = divmod(int(total_elapsed), 60)
    elapsed_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    print(f"\n  總耗時  ：{elapsed_str}")
    print(f"  結束時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(sep)

    if aborted or not all_ok:
        print("\n[WARNING] 有步驟失敗，請檢查上方錯誤訊息。")
        return 1

    print("\n✅ 所有步驟完成。")
    if "photometry" in steps_to_run and "period_analysis" not in steps_to_run:
        print("   下一步：")
        print("     1. 開啟 photometry/Photometry.ipynb 執行測光分析（Step 4）")
        print("     2. 測光完成後，執行進階週期分析（選用）：")
        print("        python run_pipeline.py --steps period_analysis")
    return 0


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    sys.exit(main())
