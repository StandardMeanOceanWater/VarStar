"""
tess_external_validation.py

讀取 TESS FITS 光變曲線資料夾，逐檔計算 Lomb–Scargle 週期圖，
由全域 LS 峰的 f_peak/n（n=1..8）建立基頻候選，以 Fourier BIC 選擇週期，
並與 period_analysis.py 的 JSON 結果比較。

輸出：
    tess_external_validation.md
    tess_external_validation.json

本模組是獨立的選用驗證步驟，不修改既有週期分析結果。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.timeseries import LombScargle


FLUX_COLUMNS = (
    "PDCSAP_FLUX", "KSPSAP_FLUX", "SAP_FLUX", "DET_FLUX", "FLUX"
)
ERROR_COLUMNS = {
    "PDCSAP_FLUX": "PDCSAP_FLUX_ERR",
    "KSPSAP_FLUX": "KSPSAP_FLUX_ERR",
    "SAP_FLUX": "SAP_FLUX_ERR",
    "DET_FLUX": "DET_FLUX_ERR",
    "FLUX": "FLUX_ERR",
}
QUALITY_COLUMN = "QUALITY"
STATUS_SUPPORT = "支持"
STATUS_CONFLICT = "衝突"
STATUS_INSUFFICIENT = "資料不足"


class ValidationError(ValueError):
    """FITS 欄位或時間資訊不足，無法安全分析。"""


def _column_map(names: list[str]) -> dict[str, str]:
    return {str(name).upper(): str(name) for name in names}


def _find_lightcurve_hdu(hdul: fits.HDUList) -> tuple[Any, dict[str, str]]:
    """找出同時有 TIME 與支援通量欄位的二進位表格。"""
    for hdu in hdul:
        data = getattr(hdu, "data", None)
        names = getattr(data, "names", None)
        if not names:
            continue

        columns = _column_map(list(names))
        if "TIME" in columns and any(name in columns for name in FLUX_COLUMNS):
            return hdu, columns

    raise ValidationError("找不到同時含 TIME 與支援通量欄位的 FITS 表格")


def _btjd_time(
    raw_time: np.ndarray,
    header: dict[str, Any],
) -> tuple[np.ndarray, bool, str]:
    """直接以 TIME 原值作為 BTJD 天，不加減 BJDREF。"""
    time = np.asarray(raw_time, dtype=float)
    if not np.any(np.isfinite(time)):
        raise ValidationError("TIME 欄沒有有限數值")

    timesys = str(header.get("TIMESYS", "")).strip().upper()
    if timesys and timesys != "TDB":
        raise ValidationError(f"TIME 系統不是 TDB：{timesys}")

    ref = None
    if "BJDREF" in header:
        ref = float(header["BJDREF"])
    elif "BJDREFI" in header or "BJDREFF" in header:
        ref = float(header.get("BJDREFI", 0.0)) + float(
            header.get("BJDREFF", 0.0)
        )

    ref_verified = ref is not None and math.isclose(
        ref, 2457000.0, rel_tol=0.0, abs_tol=1e-6
    )
    if ref is not None and not ref_verified:
        raise ValidationError(f"BJDREF 不是 2457000：{ref}")

    time_verified = timesys == "TDB" and ref_verified
    time_note = "TIME 原值直接作為 BTJD；未加減 BJDREF"
    if not time_verified:
        time_note += "；TIMESYS 或 BJDREF 未提供，時間基準未確認"
    return time, time_verified, time_note


def _infer_pipeline(filename: str, header: dict[str, Any]) -> str:
    """依檔名與 FITS header 判斷光變曲線 pipeline。"""
    text = " ".join(
        [filename]
        + [str(header.get(key, "")) for key in ("CREATOR", "ORIGIN", "PIPELINE")]
    ).upper()

    if "QLP" in text:
        return "QLP"
    if "TARS" in text:
        return "TARS"
    if "SPOC" in text:
        return "SPOC"

    # 任務級 SPOC 光變曲線檔名以 -S_LC.FITS 結尾。
    if re.fullmatch(r"TESS\d+-S\d{4}-.+-S_LC\.FITS", filename.upper()):
        return "SPOC"

    return "UNKNOWN"


def _infer_sector(filename: str, header: dict[str, Any]) -> int | None:
    if header.get("SECTOR") is not None:
        try:
            return int(header["SECTOR"])
        except (TypeError, ValueError):
            pass

    match = re.search(
        r"(?:^|[-_])S0*(\d{1,4})(?=[-_.]|$)",
        filename,
        flags=re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


def read_fits_curve(path: Path) -> dict[str, Any]:
    """讀取 FITS 光變曲線、欄位、時間和品質資訊。"""
    with fits.open(path, mode="readonly", memmap=False) as hdul:
        hdu, columns = _find_lightcurve_hdu(hdul)
        header = dict(hdul[0].header)
        header.update(dict(hdu.header))

        time_name = columns["TIME"]
        raw_time = np.asarray(hdu.data[time_name], dtype=float)
        time, timesys_verified, time_note = _btjd_time(raw_time, header)

        flux_name = next(
            name for name in FLUX_COLUMNS if name in columns
        )
        flux = np.asarray(hdu.data[columns[flux_name]], dtype=float)

        finite = np.isfinite(time) & np.isfinite(flux)
        n_finite = int(np.count_nonzero(finite))

        quality_present = QUALITY_COLUMN in columns
        quality_rejected = 0
        if quality_present:
            quality = np.asarray(
                hdu.data[columns[QUALITY_COLUMN]], dtype=float
            )
            quality_ok = np.isfinite(quality) & (quality == 0)
            quality_rejected = int(np.count_nonzero(finite & ~quality_ok))
            good = finite & quality_ok
        else:
            good = finite

        time = time[good]
        flux = flux[good]
        if time.size < 2:
            raise ValidationError("品質篩選後不足兩筆有效資料")

        flux_median = float(np.median(flux))
        if not math.isfinite(flux_median) or flux_median == 0:
            raise ValidationError("有效通量中位數為零或非有限值")
        relative_flux = flux / flux_median - 1.0

        error_name = ERROR_COLUMNS.get(flux_name)
        flux_error = None
        if error_name and error_name in columns:
            raw_error = np.asarray(hdu.data[columns[error_name]], dtype=float)[good]
            if np.all(np.isfinite(raw_error) & (raw_error > 0)):
                flux_error = raw_error / abs(flux_median)

        unique_time = np.unique(np.sort(time))
        spacing = np.diff(unique_time)
        spacing = spacing[spacing > 0]
        cadence_sec = (
            float(np.median(spacing) * 86400.0) if spacing.size else None
        )

        return {
            "time_btjd": time,
            "relative_flux": relative_flux,
            "flux_error_relative": flux_error,
            "header": header,
            "flux_column": flux_name,
            "n_total": int(len(finite)),
            "n_finite": n_finite,
            "n_valid": int(time.size),
            "quality_present": quality_present,
            "quality_status": "已提供" if quality_present else "未提供",
            "quality_rejected": quality_rejected,
            "cadence_sec": cadence_sec,
            "time_note": time_note,
            "timesys_verified": timesys_verified,
        }


def _phase_coverage(phase: np.ndarray, bins: int = 20) -> float:
    hist, _ = np.histogram(phase, bins=bins, range=(0.0, 1.0))
    return float(np.count_nonzero(hist)) / bins


def _fourier_bic(
    time: np.ndarray,
    values: np.ndarray,
    period_hr: float,
    errors: np.ndarray | None,
    max_harmonics: int,
    min_coverage: float = 0.90,
    nyquist_frequency: float | None = None,
) -> dict[str, Any] | None:
    """對一個候選週期掃描諧波階數，回傳最低 BIC 與相位覆蓋率。

    諧波階數同時受 Nyquist 限制：k×f 超過 Nyquist 的諧波只是在擬合超取樣假訊號，
    會讓短週期候選（例：1800 s 取樣下的 1.8 h）用高階項硬吃殘差、擊敗真正的 3.0 h 解。
    """
    period_day = period_hr / 24.0
    phase = ((time - time[0]) / period_day) % 1.0
    coverage = _phase_coverage(phase)
    if coverage < min_coverage:
        return None

    n_data = len(values)
    max_n = min(max_harmonics, (n_data - 2) // 2)
    if nyquist_frequency is not None and nyquist_frequency > 0:
        fundamental_frequency = 1.0 / period_day
        max_n = min(max_n, int(math.floor(nyquist_frequency / fundamental_frequency)))
    if max_n < 1:
        return None

    best: dict[str, Any] | None = None
    for n_harmonics in range(1, max_n + 1):
        columns = [np.ones_like(phase)]
        for n in range(1, n_harmonics + 1):
            angle = 2.0 * np.pi * n * phase
            columns.extend((np.cos(angle), np.sin(angle)))
        design = np.column_stack(columns)

        if errors is not None:
            weighted_design = design / errors[:, None]
            weighted_values = values / errors
        else:
            weighted_design = design
            weighted_values = values

        try:
            coefficients, *_ = np.linalg.lstsq(
                weighted_design, weighted_values, rcond=None
            )
        except np.linalg.LinAlgError:
            continue

        residual = weighted_values - weighted_design @ coefficients
        rss = float(np.dot(residual, residual))
        if not math.isfinite(rss):
            continue

        k = design.shape[1]
        bic = n_data * math.log(max(rss / n_data, 1e-300)) + k * math.log(n_data)
        if best is None or bic < best["bic"]:
            best = {
                "bic": float(bic),
                "n_harmonics": n_harmonics,
                "phase_coverage": coverage,
            }
    return best


def _make_candidates(
    frequency: np.ndarray,
    power: np.ndarray,
    raw_peak_frequency: float,
    baseline_day: float,
    min_frequency: float,
    max_frequency: float,
    min_period_hr: float,
    max_harmonics: int,
    nyquist_frequency: float | None = None,
) -> list[dict[str, Any]]:
    """建立 f_peak/n 候選；排除超 Nyquist 基頻及範圍外週期。"""
    candidates: list[dict[str, Any]] = []

    if (
        frequency.size == 0
        or power.size != frequency.size
        or not math.isfinite(raw_peak_frequency)
        or raw_peak_frequency <= 0
        or baseline_day <= 0
        or min_frequency <= 0
        or max_frequency <= 0
    ):
        return candidates

    candidate_min_period_hr = max(0.3, float(min_period_hr))
    candidate_max_period_hr = min(
        8.0,
        baseline_day * 24.0,
        24.0 / min_frequency,
    )
    if candidate_max_period_hr < candidate_min_period_hr:
        return candidates

    for divisor in range(1, max_harmonics + 1):
        f0 = raw_peak_frequency / divisor
        if not math.isfinite(f0) or f0 <= 0:
            continue

        # 候選基頻必須位於 LS 搜尋範圍，且不高於 Nyquist。
        if f0 < min_frequency or f0 > max_frequency:
            continue
        if f0 < frequency[0] or f0 > frequency[-1]:
            continue
        if nyquist_frequency is not None and f0 > nyquist_frequency:
            continue

        period_hr = 24.0 / f0
        if not (
            candidate_min_period_hr
            <= period_hr
            <= candidate_max_period_hr
        ):
            continue

        harmonic_powers = []
        score = 0.0
        for harmonic in range(1, max_harmonics + 1):
            fh = f0 * harmonic
            if fh > frequency[-1]:
                break
            p_h = float(np.interp(fh, frequency, power))
            harmonic_powers.append(p_h)
            score += p_h / harmonic

        candidates.append({
            "frequency": float(f0),
            "sources": [f"raw_LS_peak_divided_by_{divisor}"],
            "raw_peak_divisor": divisor,
            "period_hr": period_hr,
            "comb_score": float(score),
            "base_power": float(np.interp(f0, frequency, power)),
            "harmonic_powers": harmonic_powers,
        })

    # 取樣 alias 反射：峰落在 Nyquist 之上時，真實頻率是 reflections×f_s − f_peak，
    # 不是 f_peak 除以整數。例：1800 s 取樣 f_s = 48 c/d，真值 8.006 c/d 會在 39.994 c/d 出峰；
    # 兩者功率幾乎相同（LS 0.674 vs 0.672）正是 alias 的指紋。
    if nyquist_frequency is not None and nyquist_frequency > 0:
        sampling_frequency = 2.0 * nyquist_frequency
        for reflection in (1, 2):
            f_alias = abs(reflection * sampling_frequency - raw_peak_frequency)
            if not math.isfinite(f_alias) or f_alias <= 0:
                continue
            if f_alias > nyquist_frequency:
                continue
            if f_alias < min_frequency or not (
                frequency[0] <= f_alias <= frequency[-1]
            ):
                continue
            period_hr = 24.0 / f_alias
            if not (
                candidate_min_period_hr
                <= period_hr
                <= candidate_max_period_hr
            ):
                continue
            if any(abs(c["frequency"] - f_alias) < 1e-6 for c in candidates):
                for c in candidates:
                    if abs(c["frequency"] - f_alias) < 1e-6:
                        tag = f"sampling_alias_reflection_k{reflection}"
                        if tag not in c["sources"]:
                            c["sources"].append(tag)
                continue

            harmonic_powers = []
            score = 0.0
            for harmonic in range(1, max_harmonics + 1):
                fh = f_alias * harmonic
                if fh > frequency[-1]:
                    break
                p_h = float(np.interp(fh, frequency, power))
                harmonic_powers.append(p_h)
                score += p_h / harmonic

            candidates.append({
                "frequency": float(f_alias),
                "sources": [f"sampling_alias_reflection_k{reflection}"],
                "raw_peak_divisor": None,
                "period_hr": period_hr,
                "comb_score": float(score),
                "base_power": float(np.interp(f_alias, frequency, power)),
                "harmonic_powers": harmonic_powers,
            })

    return candidates


def _classify(
    candidates: list[dict[str, Any]],
    pipeline_period_hr: float,
    interval_hr: tuple[float, float] | None,
    baseline_day: float,
    relative_tolerance: float,
    delta_bic_margin: float,
) -> tuple[str, dict[str, Any] | None, float, float | None]:
    """依 pipeline 區間和 BIC 差判定支持、衝突或資料不足。"""
    fitted = [item for item in candidates if item.get("bic") is not None]
    if not fitted:
        return STATUS_INSUFFICIENT, None, 0.0, None

    best = min(fitted, key=lambda item: item["bic"])
    rayleigh_hr = pipeline_period_hr**2 / (24.0 * baseline_day)
    tolerance_hr = max(
        pipeline_period_hr * relative_tolerance,
        rayleigh_hr,
    )

    lo, hi = interval_hr or (pipeline_period_hr, pipeline_period_hr)
    lo -= tolerance_hr
    hi += tolerance_hr

    inside = [item for item in fitted if lo <= item["period_hr"] <= hi]
    outside = [item for item in fitted if not lo <= item["period_hr"] <= hi]
    if not inside:
        return STATUS_CONFLICT, best, tolerance_hr, None
    if not outside:
        return STATUS_SUPPORT, min(inside, key=lambda item: item["bic"]), tolerance_hr, None

    best_inside = min(inside, key=lambda item: item["bic"])
    best_outside = min(outside, key=lambda item: item["bic"])
    margin = best_outside["bic"] - best_inside["bic"]

    if margin >= delta_bic_margin:
        status = STATUS_SUPPORT
    elif margin <= -delta_bic_margin:
        status = STATUS_CONFLICT
    else:
        status = STATUS_INSUFFICIENT

    return status, best, tolerance_hr, float(margin)


def analyze_file(
    path: Path,
    pipeline_period_hr: float,
    interval_hr: tuple[float, float] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """單檔分析；單檔錯誤會回報為資料不足，不中止整批。"""
    row: dict[str, Any] = {
        "file": path.name,
        "pipeline": "UNKNOWN",
        "sector": None,
        "status": STATUS_INSUFFICIENT,
    }

    try:
        curve = read_fits_curve(path)
        header = curve["header"]
        pipeline = _infer_pipeline(path.name, header)
        sector = _infer_sector(path.name, header)
        time = curve["time_btjd"]
        values = curve["relative_flux"]
        errors = curve["flux_error_relative"]

        baseline_day = float(np.max(time) - np.min(time))
        if curve["n_valid"] < args.min_points:
            raise ValidationError(
                f"有效點數 {curve['n_valid']} 小於門檻 {args.min_points}"
            )
        if baseline_day <= 0:
            raise ValidationError("觀測時間基線為零")

        cadence_sec = curve["cadence_sec"]
        cadence_days = None
        nyquist = None
        if (
            cadence_sec is not None
            and math.isfinite(float(cadence_sec))
            and float(cadence_sec) > 0
        ):
            cadence_days = float(cadence_sec) / 86400.0
            nyquist = 1.0 / (2.0 * cadence_days)

        effective_min_period_hr = max(
            0.3, float(args.min_period_hours)
        )
        max_period_hr = min(
            8.0,
            float(args.max_period_hours),
            baseline_day * 24.0,
        )
        if max_period_hr < effective_min_period_hr:
            raise ValidationError("週期搜尋範圍無效")

        min_frequency = 24.0 / max_period_hr
        max_frequency = min(
            50.0,
            24.0 / effective_min_period_hr,
        )
        if max_frequency <= min_frequency:
            raise ValidationError("週期搜尋範圍無效")

        if errors is None:
            ls = LombScargle(
                time, values, fit_mean=True, center_data=True
            )
        else:
            ls = LombScargle(
                time, values, dy=errors,
                fit_mean=True, center_data=True,
            )

        frequency, power = ls.autopower(
            minimum_frequency=min_frequency,
            maximum_frequency=max_frequency,
            samples_per_peak=10,
        )
        finite_power = np.isfinite(power)
        frequency = frequency[finite_power]
        power = power[finite_power]
        if power.size < 3:
            raise ValidationError("LS 頻率格點不足")

        raw_peak_index = int(np.argmax(power))
        raw_peak_frequency = float(frequency[raw_peak_index])
        raw_peak_period_hr = 24.0 / raw_peak_frequency
        raw_peak_super_nyquist = (
            None if nyquist is None
            else bool(raw_peak_frequency > nyquist)
        )

        candidates = _make_candidates(
            frequency=frequency,
            power=power,
            raw_peak_frequency=raw_peak_frequency,
            baseline_day=baseline_day,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            min_period_hr=effective_min_period_hr,
            max_harmonics=args.max_harmonics,
            nyquist_frequency=nyquist,
        )

        for candidate in candidates:
            fit = _fourier_bic(
                time=time,
                values=values,
                period_hr=candidate["period_hr"],
                errors=errors,
                max_harmonics=args.max_harmonics,
                min_coverage=args.min_phase_coverage,
                nyquist_frequency=nyquist,
            )
            if fit is not None:
                candidate.update(fit)

        status, best, tolerance_hr, delta_bic = _classify(
            candidates=candidates,
            pipeline_period_hr=pipeline_period_hr,
            interval_hr=interval_hr,
            baseline_day=baseline_day,
            relative_tolerance=args.relative_tolerance,
            delta_bic_margin=args.delta_bic,
        )

        if not curve["timesys_verified"]:
            status = STATUS_INSUFFICIENT

        best_period_hr = best["period_hr"] if best else None
        harmonic_order = best["raw_peak_divisor"] if best else None
        best_via_alias = bool(
            best
            and any(
                str(s).startswith("sampling_alias_reflection")
                for s in best.get("sources", [])
            )
        )

        if best and raw_peak_super_nyquist and best_via_alias:
            raw_peak_interpretation = (
                "原始峰超 Nyquist；BIC 最佳候選為取樣 alias 反射"
                f"（{_fmt(raw_peak_period_hr, 4)} h ↔ {_fmt(best_period_hr, 4)} h）"
            )
        elif best and raw_peak_super_nyquist:
            raw_peak_interpretation = (
                "原始峰超 Nyquist；"
                f"BIC 最佳候選判為第{harmonic_order}諧波"
            )
        elif best and harmonic_order == 1:
            raw_peak_interpretation = "BIC 最佳候選：原始峰為基頻"
        elif best and harmonic_order is None:
            raw_peak_interpretation = "BIC 最佳候選：取樣 alias 反射"
        elif best:
            raw_peak_interpretation = (
                f"BIC 最佳候選：原始峰是第{harmonic_order}諧波"
            )
        elif raw_peak_super_nyquist:
            raw_peak_interpretation = (
                "原始峰超 Nyquist；未找到合格的基頻候選"
            )
        else:
            raw_peak_interpretation = "未判定"

        row.update({
            "pipeline": pipeline,
            "sector": sector,
            "n_total": curve["n_total"],
            "n_finite": curve["n_finite"],
            "n_valid": curve["n_valid"],
            "quality_present": curve["quality_present"],
            "quality_status": curve["quality_status"],
            "quality_rejected": curve["quality_rejected"],
            "flux_column": curve["flux_column"],
            "cadence_sec": cadence_sec,
            "nyquist_frequency_d_inv": nyquist,
            "baseline_day": baseline_day,
            "raw_peak_period_hr": raw_peak_period_hr,
            "raw_peak_frequency_d_inv": raw_peak_frequency,
            "raw_peak_super_nyquist": raw_peak_super_nyquist,
            "best_fundamental_candidate_hr": best_period_hr,
            "harmonic_order": harmonic_order,
            "raw_peak_interpretation": raw_peak_interpretation,
            "tolerance_hr": tolerance_hr,
            "delta_bic_match_vs_other": delta_bic,
            "status": status,
            "time_note": curve["time_note"],
            "timesys_verified": curve["timesys_verified"],
            "candidates": candidates,
        })
        return row

    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["pipeline"] = _infer_pipeline(path.name, {})
        row["sector"] = _infer_sector(path.name, {})
        return row


def load_pipeline_reference(path: Path) -> dict[str, Any]:
    """讀取 period_analysis.py 產生的週期 JSON。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    period_day = float(payload["period_d"])
    if not math.isfinite(period_day) or period_day <= 0:
        raise ValidationError("pipeline JSON 的 period_d 無效")

    lo = float(payload["period_lo_d"]) * 24.0
    hi = float(payload["period_hi_d"]) * 24.0
    if (
        not math.isfinite(lo)
        or not math.isfinite(hi)
        or lo <= 0
        or hi < lo
    ):
        raise ValidationError("pipeline JSON 的 period_lo_d/period_hi_d 無效")

    delta_bic_contrast = float(payload["delta_bic_contrast"])
    if not math.isfinite(delta_bic_contrast):
        raise ValidationError("pipeline JSON 的 delta_bic_contrast 無效")

    return {
        "period_hr": period_day * 24.0,
        "interval_hr": (lo, hi),
        "delta": payload.get("delta"),
        "delta_bic_contrast": delta_bic_contrast,
        "touches_edge": bool(payload["touches_edge"]),
        "channels": payload.get("channels"),
        "per_channel_period_d": payload.get("per_channel_period_d"),
        "source_json": path.name,
    }


def summarize_pipeline_consistency(
    rows: list[dict[str, Any]],
    relative_tolerance: float,
) -> dict[str, Any]:
    """同一 sector 內比較不同 pipeline 的已判定週期。"""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("sector") is not None:
            groups.setdefault(str(row["sector"]), []).append(row)

    sectors: list[dict[str, Any]] = []
    for sector, items in sorted(groups.items()):
        usable = [
            item for item in items
            if item.get("status") in (STATUS_SUPPORT, STATUS_CONFLICT)
            and item.get("best_fundamental_candidate_hr") is not None
            and item.get("pipeline") != "UNKNOWN"
        ]
        pipeline_count = len({item["pipeline"] for item in usable})

        if pipeline_count < 2:
            sector_status = STATUS_INSUFFICIENT
            median_period = None
            spread = None
        else:
            periods = [
                float(item["best_fundamental_candidate_hr"])
                for item in usable
            ]
            median_period = float(np.median(periods))
            spread = float(max(periods) - min(periods))
            tolerance = max(
                median_period * relative_tolerance,
                max(float(item.get("tolerance_hr", 0.0)) for item in usable),
            )

            if spread > 2.0 * tolerance:
                sector_status = "不一致"
            elif all(item["status"] == STATUS_SUPPORT for item in usable):
                sector_status = STATUS_SUPPORT
            elif all(item["status"] == STATUS_CONFLICT for item in usable):
                sector_status = STATUS_CONFLICT
            else:
                sector_status = STATUS_INSUFFICIENT

        sectors.append({
            "sector": sector,
            "pipeline_count": pipeline_count,
            "median_period_hr": median_period,
            "spread_hr": spread,
            "status": sector_status,
        })

    has_support = any(item["status"] == STATUS_SUPPORT for item in sectors)
    has_conflict = any(item["status"] == STATUS_CONFLICT for item in sectors)
    if has_support and not has_conflict:
        overall = STATUS_SUPPORT
    elif has_conflict and not has_support:
        overall = STATUS_CONFLICT
    else:
        overall = STATUS_INSUFFICIENT

    return {"overall_status": overall, "sectors": sectors}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(
    rows: list[dict[str, Any]],
    reference: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    lines = [
        "# TESS 外部週期驗證",
        "",
        f"- Pipeline 週期：{reference['period_hr']:.6f} 小時",
        f"- Pipeline JSON：`{reference['source_json']}`",
        f"- Pipeline ΔBIC contrast：{reference['delta_bic_contrast']:.3f}",
        f"- Pipeline touches edge：{reference['touches_edge']}",
        f"- 整體判定：**{summary['overall_status']}**",
        "",
        "## 逐檔結果",
        "",
        "| 檔案 | Pipeline | Sector | 有效點 | QUALITY | 取樣秒 | 原始峰(h) | 原始峰判讀 | 基頻候選(h) | 諧波階 | ΔBIC | 判定 |",
        "|---|---|---:|---:|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {file} | {pipeline} | {sector} | {n_valid} | {quality} | "
            "{cadence} | {raw} | {interpretation} | {fundamental} | "
            "{harmonic} | {delta} | {status} |".format(
                file=_fmt(row.get("file")),
                pipeline=_fmt(row.get("pipeline")),
                sector=_fmt(row.get("sector")),
                n_valid=_fmt(row.get("n_valid")),
                quality=_fmt(row.get("quality_status")),
                cadence=_fmt(row.get("cadence_sec"), 0),
                raw=_fmt(row.get("raw_peak_period_hr")),
                interpretation=_fmt(row.get("raw_peak_interpretation")),
                fundamental=_fmt(row.get("best_fundamental_candidate_hr")),
                harmonic=_fmt(row.get("harmonic_order"), 0),
                delta=_fmt(row.get("delta_bic_match_vs_other")),
                status=_fmt(row.get("status")),
            )
        )

    lines += [
        "",
        "## 同 sector 跨 pipeline 一致性",
        "",
        "| Sector | Pipeline 數 | 週期中位數(h) | 週期跨度(h) | 判定 |",
        "|---:|---:|---:|---:|---|",
    ]
    for item in summary["sectors"]:
        lines.append(
            "| {sector} | {count} | {median} | {spread} | {status} |".format(
                sector=_fmt(item["sector"]),
                count=_fmt(item["pipeline_count"], 0),
                median=_fmt(item["median_period_hr"]),
                spread=_fmt(item["spread_hr"]),
                status=_fmt(item["status"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def validate_folder(
    fits_dir: Path,
    pipeline_json: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    if not fits_dir.is_dir():
        raise ValidationError(f"FITS 資料夾不存在：{fits_dir}")
    if not pipeline_json.is_file():
        raise ValidationError(f"Pipeline JSON 不存在：{pipeline_json}")

    reference = load_pipeline_reference(pipeline_json)
    files = sorted(
        path for path in fits_dir.rglob("*")
        if path.is_file()
        and path.name.lower().endswith((".fits", ".fit", ".fts"))
    )
    if not files:
        raise ValidationError(f"資料夾內找不到 FITS 檔：{fits_dir}")

    rows = [
        analyze_file(
            path=path,
            pipeline_period_hr=reference["period_hr"],
            interval_hr=reference["interval_hr"],
            args=args,
        )
        for path in files
    ]
    summary = summarize_pipeline_consistency(
        rows, args.relative_tolerance
    )

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_reference": reference,
        "thresholds": {
            "relative_tolerance": args.relative_tolerance,
            "delta_bic": args.delta_bic,
            "min_phase_coverage": args.min_phase_coverage,
            "min_points": args.min_points,
            "max_harmonics": args.max_harmonics,
        },
        "summary": summary,
        "files": rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tess_external_validation.json"
    markdown_path = output_dir / "tess_external_validation.md"

    json_path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_markdown(rows, reference, summary),
        encoding="utf-8",
    )
    return markdown_path, json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="以 TESS FITS 光變曲線外部驗證 pipeline 週期"
    )
    parser.add_argument("--fits-dir", type=Path, required=True)
    parser.add_argument("--pipeline-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    # FITS TIME is consumed directly as BTJD days; no origin override is used.
    parser.add_argument("--min-period-hours", type=float, default=0.5)
    parser.add_argument("--max-period-hours", type=float, default=8.0)
    parser.add_argument("--max-harmonics", type=int, default=8)
    parser.add_argument("--min-points", type=int, default=30)
    parser.add_argument("--min-phase-coverage", type=float, default=0.90)
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    parser.add_argument("--delta-bic", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        markdown_path, json_path = validate_folder(
            fits_dir=args.fits_dir,
            pipeline_json=args.pipeline_json,
            output_dir=args.output_dir,
            args=args,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(f"[OK] Markdown: {markdown_path}")
    print(f"[OK] JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
