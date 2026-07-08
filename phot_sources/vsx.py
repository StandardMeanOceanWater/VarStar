"""VSX candidate discovery and per-channel subtarget spec builders.

v1.80：VSX 額外目標不再各自跑獨立幀迴圈，改為併入主目標的多子目標
單趟測光（run_photometry_multi_on_wcs_dir）。本模組負責：
  1. _prepare_vsx_candidates：從視野 VSX 表挑出額外目標候選
  2. build_vsx_channel_subtargets：為單一通道建構額外目標的 subtarget 規格
  3. log_vsx_channel_result：額外目標單通道結果摘要輸出
"""

from __future__ import annotations

import copy
import re as _re
from pathlib import Path

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord

from phot_sources.io_paths import build_vsx_run_layout, format_session_date
from phot_sources.logging_utils import emit_progress


def _prepare_vsx_candidates(logger, args, vsx_field, cfg):
    _VSX_MAG_MIN, _VSX_MAG_MAX = 6.0, 12.0
    if args.no_vsx:
        emit_progress(logger, "VSX disabled by --no-vsx")
        logger.info("[VSX] disabled by --no-vsx")
        print("[VSX 額外目標] --no-vsx 旗標啟用，跳過額外目標測光")
        return None
    if vsx_field is None or len(vsx_field) == 0:
        return None

    _vsx_mag_col = None
    for _vc in ("max", "min"):
        if _vc in vsx_field.columns:
            _vsx_mag_col = _vc
            break
    if _vsx_mag_col is None:
        return None

    _vsx_cand = vsx_field.copy()
    _vsx_cand["_mag"] = pd.to_numeric(_vsx_cand[_vsx_mag_col], errors="coerce")
    _vsx_cand = _vsx_cand[
        _vsx_cand["_mag"].between(_VSX_MAG_MIN, _VSX_MAG_MAX)
    ].reset_index(drop=True)

    # 排除主目標自身（10" 以內）
    if len(_vsx_cand) > 0:
        logger.info(f"[vsx] candidates={len(_vsx_cand)} mag_window={_VSX_MAG_MIN}-{_VSX_MAG_MAX}")
        _tgt_sc = SkyCoord(ra=cfg.target_radec_deg[0] * u.deg,
                           dec=cfg.target_radec_deg[1] * u.deg)
        _vsx_sc = SkyCoord(ra=_vsx_cand["ra_deg"].values * u.deg,
                           dec=_vsx_cand["dec_deg"].values * u.deg)
        _sep = _tgt_sc.separation(_vsx_sc).arcsec
        _vsx_cand = _vsx_cand[_sep > 10.0].reset_index(drop=True)

    return _vsx_cand, _VSX_MAG_MIN, _VSX_MAG_MAX


def _vsx_short_name(raw: str) -> str:
    """VSX 星名 → 短目錄名。
    短名（ASAS/NSV/V*/...）直接去空格；
    Gaia DR3 長 ID 截後 8 碼：GDR3_12345678。
    ASASSN-V 去掉 J 後面的秒數小數。
    """
    s = raw.strip()
    if s.startswith("Gaia DR3 "):
        return "GDR3_" + s.split()[-1][-8:]
    if s.startswith("ASASSN-V "):
        # ASASSN-V J083045.90-480325.0 → ASASSN_J083045-4803
        tail = s[len("ASASSN-V "):].strip()
        tail = tail.replace("J", "J", 1)
        # 簡化：去小數點後的秒數
        m = _re.match(r"(J\d{6})\.\d+([+-]\d{4})", tail)
        if m:
            return "ASASSN_" + m.group(1) + m.group(2)
    return s.replace(" ", "")


# 額外目標各通道使用的星表波段（沿用 v1.79 前 _run_single_vsx_channel 的映射）
_VSX_BAND_MAP = {"R": "r", "G1": "V", "G2": "V", "B": "B"}


def print_vsx_candidate_listing(vsx_cand) -> None:
    """主目標開跑前列出本視野額外目標（原獨立 VSX 階段的逐目標抬頭）。"""
    for _vi, _vr in vsx_cand.iterrows():
        _name_raw = str(_vr.get("Name", f"VSX_{_vi}")).strip()
        _type = str(_vr.get("Type", "?"))
        _per = _vr.get("Period", "?")
        _mag = float(_vr["_mag"])
        print(f"  ── {_name_raw}  ({_type})  mag={_mag:.2f}  P={_per}d  "
              f"RA={float(_vr['ra_deg']):.4f}  Dec={float(_vr['dec_deg']):.4f}")


def build_vsx_channel_subtargets(
    logger, yaml_cfg, active_target, active_date, log_ts, channel,
    cfg_ch, vsx_cand, comp_refs_ch, check_star_ch, shared_aperture_radius,
):
    """為單一通道建構 VSX 額外目標的 subtarget 規格列表。

    回傳 (specs, metas)：
      specs : run_photometry_multi_on_wcs_dir 的 subtarget dict 列表
      metas : 對應的 {name, name_raw, mag} 摘要資訊（供結果輸出）
    每個額外目標用 cfg_ch 的淺複本（不再覆寫共用 cfg），輸出路徑與主目標
    同層級：output/{date}/{group}/{VSXname}/{timestamp}/。
    """
    _main_tgt_yaml = yaml_cfg.get("targets", {}).get(active_target, {})
    _group = _main_tgt_yaml.get("group", active_target)
    _project_root = Path(yaml_cfg["_project_root"])
    _date_fmt = format_session_date(active_date)

    specs: list = []
    metas: list = []
    for _vi, _vr in vsx_cand.iterrows():
        _name_raw = str(_vr.get("Name", f"VSX_{_vi}")).strip()
        _name = _vsx_short_name(_name_raw)
        _ra = float(_vr["ra_deg"])
        _dec = float(_vr["dec_deg"])
        _layout = build_vsx_run_layout(
            project_root=_project_root,
            date_fmt=_date_fmt,
            group=_group,
            target=_name,
            run_ts=log_ts,
        )
        _run_root = _layout["run_root"]
        _reg_dir = _run_root / "2_regression_diag"
        _reg_dir.mkdir(parents=True, exist_ok=True)

        _cfg_v = copy.copy(cfg_ch)
        _cfg_v.target_radec_deg = (_ra, _dec)
        _cfg_v.target_name = _name
        _cfg_v.phot_band = _VSX_BAND_MAP.get(str(channel).upper(), "V")
        _cfg_v.regression_diag_dir = _reg_dir

        specs.append(dict(
            cfg_obj=_cfg_v,
            out_csv=_run_root / "1_photometry" / f"photometry_{channel}_{active_date}.csv",
            out_png=_run_root / "3_light_curve" / f"light_curve_{channel}_{active_date}.png",
            comp_refs=comp_refs_ch,
            check_star=check_star_ch,
            ap_radius=shared_aperture_radius,
        ))
        metas.append(dict(name=_name, name_raw=_name_raw, mag=float(_vr["_mag"])))
        logger.info(
            f"[vsx] subtarget queued target={_name} channel={channel} "
            f"mag={float(_vr['_mag']):.2f}"
        )
    return specs, metas


def log_vsx_channel_result(logger, meta, channel, vsx_df) -> None:
    """額外目標單通道結果摘要（原 _run_single_vsx_channel 的輸出部分）。"""
    _name = meta["name"]
    if vsx_df is None:
        logger.warning(f"[vsx] target={_name} channel={channel} failed; see log")
        print(f"    [VSX] {_name} {channel}: 測光失敗（見 log）")
        return
    _n_ok = int(vsx_df["ok"].sum()) if "ok" in vsx_df.columns else 0
    if _n_ok > 0 and "m_var" in vsx_df.columns:
        _ok_rows = vsx_df[vsx_df["ok"] == 1]
        _med = float(_ok_rows["m_var"].median())
        _amp = float(_ok_rows["m_var"].max() - _ok_rows["m_var"].min())
        logger.info(
            f"[vsx] target={_name} channel={channel} ok={_n_ok}/{len(vsx_df)} "
            f"median={_med:.3f} amp={_amp:.3f}"
        )
        print(f"    [VSX] {_name} {channel}: ok={_n_ok}/{len(vsx_df)}  "
              f"median={_med:.3f}  amp={_amp:.3f}")
    else:
        logger.info(
            f"[vsx] target={_name} channel={channel} ok={_n_ok}/{len(vsx_df)}"
        )
        print(f"    [VSX] {_name} {channel}: ok={_n_ok}/{len(vsx_df)}")


__all__ = [
    "_prepare_vsx_candidates",
    "_vsx_short_name",
    "build_vsx_channel_subtargets",
    "log_vsx_channel_result",
    "print_vsx_candidate_listing",
]
