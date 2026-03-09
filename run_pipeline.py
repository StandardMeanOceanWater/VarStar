# -*- coding: utf-8 -*-
"""
run_pipeline.py  —  變星測光管線整合入口
專案：變星測光管線 v1.0

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


def _run_plate_solve(config_path: Path) -> bool:
    """呼叫 plate_solve.run_plate_solve()。"""
    try:
        from plate_solve import run_plate_solve
        run_plate_solve(config_path)
        return True
    except ImportError as exc:
        print(
            f"[ERROR] 無法匯入 plate_solve.py：{exc}\n"
            "        請確認 plate_solve.py 與本腳本在同一目錄。"
        )
        return False
    except Exception as exc:
        print(f"[ERROR] plate_solve 步驟失敗：{exc}")
        return False


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

    # ── 呼叫模組 ─────────────────────────────────────────────────────────────
    try:
        from period_analysis import run_period_analysis
        run_period_analysis(config_path)
        return True
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
