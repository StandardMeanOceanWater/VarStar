"""Regenerate the four CC And period-analysis outputs for the latest smoke run."""

from __future__ import annotations

import sys
import time
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PIPELINE_DIR.parent
RUN_DIR = (
    PROJECT_DIR
    / "output"
    / "2025-11-22"
    / "And"
    / "CCAnd"
    / "20260708_185042_smoke_a1_20260708"
)
CONFIG_PATH = PIPELINE_DIR / "observation_config.yaml"
CHANNELS = ("R", "G1", "G2", "B")
OBS_DATE = "2025-11-22"


def main() -> int:
    sys.path.insert(0, str(PIPELINE_DIR))

    import pandas as pd
    from period_analysis import run_period_analysis

    photometry_dir = RUN_DIR / "1_photometry"
    output_dir = RUN_DIR / "4_period_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, channel in enumerate(CHANNELS):
        csv_path = photometry_dir / f"photometry_{channel}_{OBS_DATE.replace('-', '')}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing input CSV: {csv_path}")

        started = time.perf_counter()
        result = run_period_analysis(
            df=pd.read_csv(csv_path),
            target_name="CCAnd",
            channel=channel,
            out_dir=output_dir,
            config_path=CONFIG_PATH,
            run_prewhitening=False,
            obs_date=OBS_DATE,
        )
        elapsed = time.perf_counter() - started
        ls_result = result.get("ls_result", {})
        fit_result = result.get("fit_result", {})
        if not fit_result:
            raise RuntimeError(f"No Fourier fit was produced for channel {channel}.")

        period_hr = float(fit_result["period"]) * 24.0
        candidate = ls_result.get("candidate_selection", {})
        png_path = output_dir / f"period_analysis_CCAnd_{channel}.png"
        json_path = output_dir / f"period_fourier_fit_CCAnd_{channel}.json"
        if not png_path.is_file() or not json_path.is_file():
            raise FileNotFoundError(
                f"Expected outputs are missing for {channel}: {png_path}, {json_path}"
            )

        print(
            f"{channel}: P={period_hr:.6f} h; "
            f"status={candidate.get('status')}; "
            f"fallback={candidate.get('fallback_to_foldable_peak')}; "
            f"elapsed={elapsed:.2f} s; PNG={png_path}"
        )
        if index == 0:
            print("R smoke run completed; continuing with G1, G2, and B.")

    print("All four CC And period-analysis PNG and JSON outputs were found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
