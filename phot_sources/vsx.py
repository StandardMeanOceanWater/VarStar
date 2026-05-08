"""VSX candidate discovery and VSX target runners."""

from __future__ import annotations

import re as _re
import time
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


def _get_vsx_ready_channels(channels, comp_refs_per_ch, split_dir_per_ch):
    return [
        _vch for _vch in channels
        if _vch in comp_refs_per_ch
        and _vch in split_dir_per_ch
        and any(split_dir_per_ch[_vch].glob(f"*_{_vch}.fits"))
    ]


def _run_single_vsx_channel(
    logger, cfg, vsx_name, vsx_ra, vsx_dec, vsx_run_root, active_date, vch,
    comp_refs_per_ch, check_star_per_ch, check_star, split_dir_per_ch,
    shared_aperture_radius, active_cache,
    photometry_func,
):
    _vsx_csv = (vsx_run_root / "1_photometry" / f"photometry_{vch}_{active_date}.csv")
    _vsx_png = (vsx_run_root / "3_light_curve" / f"light_curve_{vch}_{active_date}.png")
    try:
        # 暫存並覆寫全域 cfg
        _cfg_backup_radec = cfg.target_radec_deg
        _cfg_backup_name = cfg.target_name
        _cfg_backup_band = cfg.phot_band
        _cfg_backup_regdir = cfg.regression_diag_dir
        _vsx_band_map = {"R": "r", "G1": "V", "G2": "V", "B": "B"}
        cfg.target_radec_deg = (vsx_ra, vsx_dec)
        cfg.target_name = vsx_name
        cfg.phot_band = _vsx_band_map.get(vch.upper(), "V")
        # 回歸診斷圖存到 VSX 目標自己的目錄，避免覆蓋主目標
        _vsx_reg_dir = vsx_run_root / "2_regression_diag"
        _vsx_reg_dir.mkdir(parents=True, exist_ok=True)
        cfg.regression_diag_dir = _vsx_reg_dir

        _vsx_df, _ = photometry_func(
            split_dir_per_ch[vch],
            _vsx_csv,
            _vsx_png,
            comp_refs=comp_refs_per_ch[vch],
            cfg_obj=cfg,
            check_star=check_star_per_ch.get(vch, check_star),
            ap_radius=shared_aperture_radius,
            channel=vch,
            shared_cache=active_cache,
        )
        _n_ok = int(_vsx_df["ok"].sum()) if "ok" in _vsx_df.columns else 0
        if _n_ok > 0 and "m_var" in _vsx_df.columns:
            _ok_rows = _vsx_df[_vsx_df["ok"] == 1]
            _med = float(_ok_rows["m_var"].median())
            _amp = float(_ok_rows["m_var"].max() - _ok_rows["m_var"].min())
            logger.info(
                f"[vsx] target={vsx_name} channel={vch} ok={_n_ok}/{len(_vsx_df)} "
                f"median={_med:.3f} amp={_amp:.3f}"
            )
            print(f"    {vch}: ok={_n_ok}/{len(_vsx_df)}  "
                  f"median={_med:.3f}  amp={_amp:.3f}")
        else:
            logger.info(
                f"[vsx] target={vsx_name} channel={vch} ok={_n_ok}/{len(_vsx_df)}"
            )
            print(f"    {vch}: ok={_n_ok}/{len(_vsx_df)}")
        return _vsx_df
    except Exception as _e_vsx_phot:
        logger.warning(
            f"[vsx] target={vsx_name} channel={vch} error: {_e_vsx_phot}"
        )
        print(f"    {vch}: 測光失敗 — {_e_vsx_phot}")
        return None
    finally:
        cfg.target_radec_deg = _cfg_backup_radec
        cfg.target_name = _cfg_backup_name
        cfg.phot_band = _cfg_backup_band
        cfg.regression_diag_dir = _cfg_backup_regdir


def _run_single_vsx_target(
    logger, cfg, channels, active_date, log_ts, vsx_group, vsx_project_root, vsx_date_fmt,
    vsx_row, vsx_idx, comp_refs_per_ch, check_star_per_ch, split_dir_per_ch,
    shared_aperture_radius, active_cache, check_star,
    photometry_func,
):
    _vsx_name_raw = str(vsx_row.get("Name", f"VSX_{vsx_idx}")).strip()
    _vsx_name = _vsx_short_name(_vsx_name_raw)
    _vsx_ra = float(vsx_row["ra_deg"])
    _vsx_dec = float(vsx_row["dec_deg"])
    _vsx_type = str(vsx_row.get("Type", "?"))
    _vsx_per = vsx_row.get("Period", "?")
    _vsx_mag = float(vsx_row["_mag"])
    _vsx_ready_channels = _get_vsx_ready_channels(
        channels, comp_refs_per_ch, split_dir_per_ch
    )
    if not _vsx_ready_channels:
        logger.warning(
            f"[VSX SKIP] name={_vsx_name} no ready channel before target tree creation"
        )
        return
    logger.info(
        f"[vsx] target={_vsx_name} mag={_vsx_mag:.2f} ready_channels={_vsx_ready_channels}"
    )
    print(f"\n  ── {_vsx_name_raw}  ({_vsx_type})  "
          f"mag={_vsx_mag:.2f}  P={_vsx_per}d  "
          f"RA={_vsx_ra:.4f}  Dec={_vsx_dec:.4f}")

    # 輸出路徑：與 YAML 目標相同層級
    # output/{date}/{group}/{VSXname}/{timestamp}/
    _vsx_layout = build_vsx_run_layout(
        project_root=vsx_project_root,
        date_fmt=vsx_date_fmt,
        group=vsx_group,
        target=_vsx_name,
        run_ts=log_ts,
    )
    _vsx_run_root = _vsx_layout["run_root"]

    _vsx_ch_results = {}
    for _vch in channels:
        if _vch not in comp_refs_per_ch:
            logger.warning(f"[VSX SKIP] target={_vsx_name} channel={_vch} missing comp refs")
            print(f"    {_vch}: 跳過（主目標該通道無比較星）")
            continue
        _vsx_df = _run_single_vsx_channel(
            logger, cfg, _vsx_name, _vsx_ra, _vsx_dec, _vsx_run_root, active_date, _vch,
            comp_refs_per_ch, check_star_per_ch, check_star, split_dir_per_ch,
            shared_aperture_radius, active_cache,
            photometry_func,
        )
        if _vsx_df is not None:
            _vsx_ch_results[_vch] = _vsx_df


def _run_vsx_targets_for_target(
    logger, yaml_cfg, active_target, active_date, log_ts, channels,
    cfg, vsx_cand, comp_refs_per_ch, check_star_per_ch, split_dir_per_ch,
    shared_aperture_radius, active_cache, check_star,
    photometry_func,
):
    _vsx_started = time.perf_counter()
    emit_progress(logger, f"VSX start target={active_target} count={len(vsx_cand)}")
    # 取得主目標的 group 和 project_root（用於建構輸出路徑）
    _main_tgt_yaml = yaml_cfg.get("targets", {}).get(active_target, {})
    _vsx_group = _main_tgt_yaml.get("group", active_target)
    _vsx_project_root = Path(yaml_cfg["_project_root"])
    _vsx_date_fmt = format_session_date(active_date)
    _last_vsx_heartbeat_at = _vsx_started

    for _loop_idx, (_vi, _vr) in enumerate(vsx_cand.iterrows(), start=1):
        _now = time.perf_counter()
        if (_loop_idx % 25 == 0) or (_now - _last_vsx_heartbeat_at >= 30.0):
            emit_progress(
                logger,
                f"VSX heartbeat target={active_target} processed={_loop_idx}/{len(vsx_cand)} "
                f"elapsed={_now - _vsx_started:.1f}s"
            )
            _last_vsx_heartbeat_at = _now
        _run_single_vsx_target(
            logger, cfg, channels, active_date, log_ts, _vsx_group, _vsx_project_root, _vsx_date_fmt,
            _vr, _vi, comp_refs_per_ch, check_star_per_ch, split_dir_per_ch,
            shared_aperture_radius, active_cache, check_star,
            photometry_func,
        )

    print(f"\n  [VSX 額外目標] 全部完成")


__all__ = [
    "_get_vsx_ready_channels",
    "_prepare_vsx_candidates",
    "_run_single_vsx_channel",
    "_run_single_vsx_target",
    "_run_vsx_targets_for_target",
    "_vsx_short_name",
]
