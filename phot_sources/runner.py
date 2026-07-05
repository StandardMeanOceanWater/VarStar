"""Photometry CLI and runner orchestration helpers."""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

from phot_aperture import estimate_aperture_radius
from phot_config import cfg_from_yaml
from phot_sources.catalog import (
    _load_or_build_unified_catalog,
    auto_select_comps,
)
from phot_sources.core import compute_annulus_radii
from phot_sources.field import (
    _build_shared_field_caches,
    _compute_field_key,
    _field_center_from_wcs_fits,
    _resolve_active_field_cache,
)
from phot_sources.io_paths import build_target_log_path, get_field_catalog_path
from phot_sources.logging_utils import (
    attach_file_handler,
    build_log_timestamp,
    detach_file_handlers,
    emit_progress,
    emit_progress_done,
    init_summary_logger,
    redirect_warnings_to_logger,
)
from pipeline_config import load_pipeline_config


def _normalize_channels(raw_channels) -> "list[str]":
    channels: list[str] = []
    for raw in raw_channels or []:
        ch = str(raw).strip().upper()
        if ch and ch not in channels:
            channels.append(ch)
    if "G1" in channels and "G2" not in channels:
        channels.insert(channels.index("G1") + 1, "G2")
    if not channels:
        raise ValueError("photometry channels are empty")
    return channels


def _resolve_channels(args, yaml_cfg) -> "list[str]":
    if args.channels:
        return _normalize_channels(args.channels)
    _ch_raw = yaml_cfg.get("photometry", {}).get(
        "channels", yaml_cfg.get("photometry", {}).get("channel", ["B"])
    )
    if isinstance(_ch_raw, str):
        _ch_raw = [_ch_raw]
    return _normalize_channels(_ch_raw)


def _parse_cli_args(argv=None):
    import argparse

    _parser = argparse.ArgumentParser(description="差分測光管線 — 步驟 4")
    _parser.add_argument("--target", default=None,
                         help="目標星（例如 V1162Ori）")
    _parser.add_argument("--date", default=None,
                         help="觀測日期（例如 20251220）")
    _parser.add_argument("--config", default=None, type=Path,
                         help="observation_config.yaml path")
    _parser.add_argument("--channels", default=None, nargs="+",
                         help="通道列表（例如 --channels R G1 B）")
    _parser.add_argument("--split_subdir", default="splits",
                         help="split 子目錄名稱（例如 splits_raw）")
    _parser.add_argument("--all", action="store_true",
                         help="處理 yaml 中所有 obs_sessions 的全部目標")
    _parser.add_argument("--out-tag", default=None, dest="out_tag",
                         help="輸出子目錄後綴，例如 sat65536 → output_sat65536（不填則用 output）")
    _parser.add_argument("--no-vsx", action="store_true", dest="no_vsx",
                         help="跳過 VSX 額外目標星測光，只跑主目標")
    _parser.add_argument("--raw", action="store_true",
                         help="使用 raw 未校正影像（splits_raw/）")
    _args = _parser.parse_args(argv)
    if _args.raw:
        _args.split_subdir = "splits_raw"
        if not _args.out_tag:
            _args.out_tag = "raw"
    return _args


def _init_summary_logger(args):
    _log_ts = build_log_timestamp(args.out_tag)
    _logger = init_summary_logger(logger_name="photometry", stream=sys.stdout)
    redirect_warnings_to_logger(_logger)
    return _logger, _log_ts


def _load_pipeline_yaml(config_path: Path | None = None):
    return load_pipeline_config(config_path)


def _session_targets(session: dict) -> "list[str]":
    raw = session.get("targets")
    if raw is None:
        raw = session.get("target")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    one = str(raw).strip()
    return [one] if one else []


def _build_targets_list(args, yaml_cfg) -> "list[tuple[str, str]]":
    if args.all:
        _all_targets = []
        for _sess in yaml_cfg.get("obs_sessions", []):
            _sess_targets = _session_targets(_sess)
            if not _sess_targets:
                print(f"[WARN] obs_session missing target(s): date={_sess.get('date')}")
                continue
            for _t in _sess_targets:
                _all_targets.append((str(_t), str(_sess["date"])))
        return _all_targets
    if args.target is None and args.date is not None:
        _targets_list = []
        for _sess in yaml_cfg.get("obs_sessions", []):
            if str(_sess["date"]) == str(args.date):
                _sess_targets = _session_targets(_sess)
                if not _sess_targets:
                    print(f"[WARN] obs_session missing target(s): date={_sess.get('date')}")
                    continue
                for _t in _sess_targets:
                    _targets_list.append((str(_t), str(args.date)))
        if not _targets_list:
            print(f"[WARN] yaml 中找不到日期 {args.date} 的場次")
        return _targets_list
    if args.target is None and args.date is None:
        for _sess in yaml_cfg.get("obs_sessions", []):
            _sess_targets = _session_targets(_sess)
            if _sess_targets:
                return [(str(_sess_targets[0]), str(_sess["date"]))]
        return []
    if args.target is not None and args.date is None:
        for _sess in yaml_cfg.get("obs_sessions", []):
            if str(args.target) in _session_targets(_sess):
                return [(str(args.target), str(_sess["date"]))]
        print(f"[WARN] yaml 中找不到目標 {args.target} 的場次")
        return []
    return [(str(args.target), str(args.date))]


def _summarize_field_key(field_key) -> str:
    if not field_key:
        return "none"
    _raw = "|".join(str(x) for x in field_key)
    _digest = hashlib.sha1(_raw.encode("utf-8")).hexdigest()[:12]
    _head = ",".join(str(x) for x in field_key[:4])
    return f"{_head}#{_digest}"


def _comp_signature(comp_refs: list) -> str:
    if not comp_refs:
        return "empty"
    _rows = sorted(
        (
            f"{float(ra):.5f},{float(dec):.5f},{float(m_cat):.3f}"
            for ra, dec, m_cat, _m_err, _weight in comp_refs
        )
    )
    _digest = hashlib.sha1("|".join(_rows).encode("utf-8")).hexdigest()[:12]
    _preview = ";".join(_rows[:3])
    return f"{_digest}|{_preview}"


def _prepare_target_aperture_state(logger, cfg, wcs_files, active_target):
    _catalog_started = time.perf_counter()
    emit_progress(logger, f"catalog preload start target={active_target}")
    _field_cat_path_pre = get_field_catalog_path(cfg.run_root)
    _field_ra_deg, _field_dec_deg = _field_center_from_wcs_fits(wcs_files[0])
    logger.info(
        "[field center] target=%s field_center=(%.6f, %.6f) target_center=(%.6f, %.6f) catalog=%s",
        active_target,
        float(_field_ra_deg),
        float(_field_dec_deg),
        float(cfg.target_radec_deg[0]),
        float(cfg.target_radec_deg[1]),
        _field_cat_path_pre,
    )
    if not _field_cat_path_pre.exists():
        print(f"[catalog] 預建統一視場星表...")
    _load_or_build_unified_catalog(
        cfg, _field_ra_deg, _field_dec_deg, _field_cat_path_pre,
    )

    emit_progress_done(logger, f"catalog preload target={active_target}", _catalog_started)

    try:
        (_comp_ap, comp_df_matched, check_star,
         aavso_matched, apass_matched, active_source,
         _vsx_field) = auto_select_comps(
            wcs_files[0], cfg.target_radec_deg, cfg_obj=cfg, band="V"
        )
    except RuntimeError as _e:
        logger.warning(f"[SKIP] {active_target} aperture preselect failed: {_e}")
        print(f"[SKIP] {active_target} 孔徑估算比較星選取失敗，跳過此目標：{_e}")
        return None
    print(f"check_star：{check_star}")

    logger.info(f"[check_star] {check_star}")
    ap_r = None
    if cfg.aperture_auto:
        _aperture_started = time.perf_counter()
        emit_progress(logger, f"aperture estimate start target={active_target}")
        logger.info(
            f"[aperture_auto] growth_fraction={cfg.aperture_growth_fraction:.3f} "
            f"r_min={cfg.aperture_min_radius} r_max={cfg.aperture_max_radius}"
        )
        ap_r = estimate_aperture_radius(
            wcs_files[0], comp_df_matched,   # comp_df_matched from V-band initial selection
            cfg.aperture_min_radius, cfg.aperture_max_radius,
            cfg.aperture_growth_fraction,
            cfg_obj=cfg,
            max_stars=(
                max(10, min(30, len(comp_df_matched)))
                if comp_df_matched is not None else 20
            ),
        )
        if ap_r is not None:
            logger.info(
                f"[aperture_auto] selected_radius={ap_r:.2f} "
                f"growth_fraction={cfg.aperture_growth_fraction:.3f}"
            )
            cfg.aperture_radius = ap_r
            print(
                f"[生長曲線] 自動孔徑半徑 = {cfg.aperture_radius:.2f} px"
                "（所有通道共用）"
            )

    if cfg.aperture_auto:
        emit_progress_done(logger, f"aperture estimate target={active_target}", _aperture_started)

    if cfg.aperture_auto and ap_r is None:
        logger.warning(
            f"[aperture_auto] estimate failed; keep fixed aperture={cfg.aperture_radius:.2f} "
            f"growth_fraction={cfg.aperture_growth_fraction:.3f}"
        )

    ap_r_in, ap_r_out = compute_annulus_radii(
        cfg.aperture_radius, cfg.annulus_r_in, cfg.annulus_r_out
    )
    print(f"孔徑 r={cfg.aperture_radius:.2f}  r_in={ap_r_in:.2f}  r_out={ap_r_out:.2f}")
    logger.info(
        f"[aperture] radius={cfg.aperture_radius:.2f} r_in={ap_r_in:.2f} "
        f"r_out={ap_r_out:.2f} growth_fraction={cfg.aperture_growth_fraction:.3f}"
    )
    _shared_aperture_radius = cfg.aperture_radius
    return _shared_aperture_radius, check_star, _vsx_field, ap_r_in, ap_r_out


def _open_target_run(logger, yaml_cfg, args, log_ts, active_target, active_date, channels):
    detach_file_handlers(logger)

    try:
        cfg = cfg_from_yaml(
            yaml_cfg, active_target, active_date, channel=channels[0],
            split_subdir=args.split_subdir, out_tag=args.out_tag, run_ts=log_ts
        )
    except Exception as _e_cfg:
        logger.warning(f"[SKIP] {active_target}/{active_date} cfg error: {_e_cfg}")
        print(f"[SKIP] {active_target}/{active_date} cfg 錯誤：{_e_cfg}")
        return None

    _log_path = build_target_log_path(cfg.run_root, active_date, log_ts)
    _file_hdl = attach_file_handler(logger, _log_path)
    logger.info(f"[LOG] {_log_path}")
    print(f"[LOG] {_log_path}")
    logger.info(
        f"[runtime] target={active_target} date={active_date} "
        f"aperture_growth_fraction={cfg.aperture_growth_fraction:.3f}"
    )

    _ch0 = channels[0]
    wcs_files = sorted(cfg.wcs_dir.glob(f"*_{_ch0}.fits"))
    if not wcs_files:
        logger.warning(f"[SKIP] no split FITS for channel={_ch0} dir={cfg.wcs_dir}")
        print(f"[SKIP] 找不到 split/{_ch0} FITS：{cfg.wcs_dir}，跳過此目標")
        detach_file_handlers(logger)
        return None
    print(f"找到 split/{_ch0} FITS：{len(wcs_files)} 張")

    return cfg, wcs_files, _file_hdl


def _prepare_target_run_state(logger, yaml_cfg, args, log_ts, active_target, active_date, channels):
    logger.info(f"[target] start target={active_target} date={active_date} channels={channels}")
    _target_started = time.perf_counter()
    emit_progress(logger, f"main target start target={active_target} date={active_date}")
    print(f"\n{'#'*60}")
    print(f"# 目標：{active_target}  日期：{active_date}  通道：{channels}")
    print(f"{'#'*60}")

    _target_boot = _open_target_run(
        logger, yaml_cfg, args, log_ts, active_target, active_date, channels
    )
    if _target_boot is None:
        return None
    cfg, wcs_files, _file_hdl = _target_boot

    _prep_state = _prepare_target_aperture_state(logger, cfg, wcs_files, active_target)
    if _prep_state is None:
        detach_file_handlers(logger)
        return None
    _shared_aperture_radius, check_star, _vsx_field, ap_r_in, ap_r_out = _prep_state
    emit_progress_done(logger, f"main target setup target={active_target}", _target_started)
    return cfg, _shared_aperture_radius, check_star, _vsx_field, ap_r_in, ap_r_out


def _build_channel_cfg(
    yaml_cfg, args, log_ts, active_target, active_date,
    channel, base_cfg, shared_aperture_radius,
):
    cfg_ch = cfg_from_yaml(
        yaml_cfg, active_target, active_date, channel=channel,
        split_subdir=args.split_subdir, out_tag=args.out_tag,
        run_ts=log_ts,
    )
    cfg_ch.aperture_radius = shared_aperture_radius
    # gain/read_noise：cfg_ch 從 sensor_db 讀取；若為 None 才從首通道 fallback
    if cfg_ch.gain_e_per_adu is None and base_cfg.gain_e_per_adu is not None:
        cfg_ch.gain_e_per_adu = base_cfg.gain_e_per_adu
    if cfg_ch.read_noise_e is None and base_cfg.read_noise_e is not None:
        cfg_ch.read_noise_e = base_cfg.read_noise_e
    return cfg_ch


def _select_channel_comps(logger, channel, fits_files, cfg_ch):
    # ── 比較星 m_cat 波段重映射（每通道用正確星表）──
    _band_map_ch = {"R": "R", "G1": "V", "G2": "V", "B": "B"}
    _band_ch = _band_map_ch.get(str(channel).upper(), "V")
    try:
        (_comp_refs_ch, _comp_df_ch, _check_star_ch,
         _aavso_ch, _apass_ch, _active_source_ch,
         _vsx_ch) = auto_select_comps(
            fits_files[0], cfg_ch.target_radec_deg, cfg_obj=cfg_ch, band=_band_ch
        )
        logger.info(f"[comp] channel={channel} source={_active_source_ch} n={len(_comp_refs_ch)}")
        logger.info(
            "[comp trace] target=%s channel=%s comp_count=%d check_star=%s comp_signature=%s",
            cfg_ch.target_name,
            channel,
            len(_comp_refs_ch),
            _check_star_ch,
            _comp_signature(_comp_refs_ch),
        )
        print(f"  [comp] {channel} source={_active_source_ch} n={len(_comp_refs_ch)}")
        return _comp_refs_ch, _check_star_ch
    except RuntimeError as _e_comp_ch:
        logger.warning(f"[SKIP] channel={channel} comp selection failed: {_e_comp_ch}")
        print(f"  [SKIP] {channel} 比較星讀取失敗：{_e_comp_ch}")
        return None


def _prepare_channel_fits_inputs(logger, channel, cfg_ch):
    _split_dir = cfg_ch.wcs_dir
    _fits_ch = sorted(_split_dir.glob(f"*_{channel}.fits"))
    if not _fits_ch:
        logger.warning(f"[SKIP] channel={channel} no split FITS in dir={_split_dir}")
        print(f"  [SKIP] split/{channel}/ 找不到 FITS，跳過此通道")
        return _split_dir, None
    print(f"  FITS 張數：{len(_fits_ch)}")
    print(f"  輸出 CSV ：{cfg_ch.phot_out_csv}")
    return _split_dir, _fits_ch


def _run_channel_photometry(
    logger, channel, split_dir, cfg_ch, comp_refs_ch, check_star_ch, active_cache,
    photometry_func,
):
    # 同視野多目標：傳入共用比較星快取
    df_ch, _comp_lc_ch = photometry_func(
        split_dir,
        cfg_ch.phot_out_csv,
        cfg_ch.phot_out_png,
        comp_refs=comp_refs_ch,
        cfg_obj=cfg_ch,
        check_star=check_star_ch,
        ap_radius=cfg_ch.aperture_radius,
        channel=channel,
        shared_cache=active_cache,
    )
    if active_cache:
        print(f"  [快取] 比較星快取 {len(active_cache)} 筆")
    _ok_cnt = int(df_ch['ok'].sum()) if 'ok' in df_ch.columns else 0
    logger.info(f"[channel] done channel={channel} ok={_ok_cnt}/{len(df_ch)}")
    print(f"  [完成] {channel}：ok={_ok_cnt} / {len(df_ch)} 幀")
    return df_ch


def _run_single_channel(
    logger, yaml_cfg, args, log_ts, active_target, active_date, channels,
    base_cfg, shared_aperture_radius, channel, active_cache,
    photometry_func,
):
    logger.info(f"[channel] start channel={channel} shared_aperture={shared_aperture_radius:.2f}")
    _channel_started = time.perf_counter()
    emit_progress(logger, f"channel start target={active_target} channel={channel}")
    print(f"\n{'='*55}")
    print(f"  通道 {channel}  ({channels.index(channel) + 1}/{len(channels)})")
    print(f"{'='*55}")

    cfg_ch = _build_channel_cfg(
        yaml_cfg, args, log_ts, active_target, active_date,
        channel, base_cfg, shared_aperture_radius,
    )
    _split_dir, _fits_ch = _prepare_channel_fits_inputs(logger, channel, cfg_ch)
    if _fits_ch is None:
        return cfg_ch.run_root, _split_dir, None

    _comp_pick = _select_channel_comps(logger, channel, _fits_ch, cfg_ch)
    if _comp_pick is None:
        return cfg_ch.run_root, _split_dir, None
    _comp_refs_ch, _check_star_ch = _comp_pick

    df_ch = _run_channel_photometry(
        logger, channel, _split_dir, cfg_ch,
        _comp_refs_ch, _check_star_ch, active_cache,
        photometry_func,
    )
    emit_progress_done(logger, f"channel target={active_target} channel={channel}", _channel_started)
    return cfg_ch.run_root, _split_dir, (df_ch, _comp_refs_ch, _check_star_ch)


def _run_channel_loop(
    logger, yaml_cfg, args, log_ts, active_target, active_date, channels,
    cfg, shared_aperture_radius, active_cache,
    photometry_func,
):
    channel_results: dict = {}
    _comp_refs_per_ch: dict = {}
    _check_star_per_ch: dict = {}
    _split_dir_per_ch: dict = {}
    _stage4_run_root = None

    for _ch in channels:
        _run_root_ch, _split_dir, _single_result = _run_single_channel(
            logger, yaml_cfg, args, log_ts, active_target, active_date, channels,
            cfg, shared_aperture_radius, _ch, active_cache,
            photometry_func,
        )
        _stage4_run_root = _run_root_ch
        _split_dir_per_ch[_ch] = _split_dir
        if _single_result is None:
            continue
        df_ch, _comp_refs_ch, _check_star_ch = _single_result
        _comp_refs_per_ch[_ch] = _comp_refs_ch
        _check_star_per_ch[_ch] = _check_star_ch
        channel_results[_ch] = df_ch

    return (
        channel_results, _comp_refs_per_ch, _check_star_per_ch, _split_dir_per_ch,
        _stage4_run_root,
    )


def _run_channels_for_target(
    logger, yaml_cfg, args, log_ts, active_target, active_date, channels,
    cfg, shared_aperture_radius, field_caches,
    photometry_func,
):
    _active_field_key = _compute_field_key(
        yaml_cfg, active_target, active_date, channels[0], args.split_subdir
    )
    _active_cache = _resolve_active_field_cache(
        yaml_cfg, active_target, active_date, channels, args.split_subdir, field_caches,
    )
    logger.info(
        "[field cache] target=%s key=%s shared_cache=%s cache_entries=%d",
        active_target,
        _summarize_field_key(_active_field_key),
        "enabled" if _active_cache is not None else "disabled",
        len(_active_cache) if _active_cache is not None else 0,
    )
    (
        channel_results, _comp_refs_per_ch, _check_star_per_ch, _split_dir_per_ch,
        _stage4_run_root,
    ) = _run_channel_loop(
        logger, yaml_cfg, args, log_ts, active_target, active_date, channels,
        cfg, shared_aperture_radius, _active_cache,
        photometry_func,
    )

    print(f"\n所有通道完成：{list(channel_results.keys())}")
    return (
        channel_results, _comp_refs_per_ch, _check_star_per_ch, _split_dir_per_ch,
        _active_cache, _stage4_run_root
    )


def run_main(
    argv=None,
    *,
    photometry_func,
    stage4_postprocess_func,
    prepare_vsx_candidates_func,
    run_vsx_targets_func,
):
    _args = _parse_cli_args(argv)
    _logger, _log_ts = _init_summary_logger(_args)

    _yaml = _load_pipeline_yaml(_args.config)

    CHANNELS = _resolve_channels(_args, _yaml)
    _targets_list = _build_targets_list(_args, _yaml)

    print(f"[photometry] 待處理目標：{_targets_list}  通道：{CHANNELS}")
    if not _targets_list:
        print("[ERROR] no photometry targets resolved from CLI/config")
        return 1

    _logger.info(f"[photometry] targets={_targets_list} channels={CHANNELS}")
    _field_groups, _field_caches = _build_shared_field_caches(
        _yaml, _targets_list, CHANNELS[0], _args.split_subdir
    )
    _multi_fields = sum(1 for v in _field_groups.values() if len(v) > 1)
    if _multi_fields:
        _logger.info(f"[photometry] shared_field_groups={_multi_fields}")
        print(f"[photometry] 偵測到 {_multi_fields} 個多目標視野，啟用共用比較星快取")

    _processed_targets = 0
    _processed_channels = 0

    for (ACTIVE_TARGET, ACTIVE_DATE) in _targets_list:
        _main_target_started = time.perf_counter()
        _target_state = _prepare_target_run_state(
            _logger, _yaml, _args, _log_ts, ACTIVE_TARGET, ACTIVE_DATE, CHANNELS
        )
        if _target_state is None:
            continue
        cfg, _shared_aperture_radius, check_star, _vsx_field, ap_r_in, ap_r_out = _target_state

        try:
            channel_results, _comp_refs_per_ch, _check_star_per_ch, _split_dir_per_ch, _active_cache, _stage4_run_root = (
                _run_channels_for_target(
                    _logger, _yaml, _args, _log_ts, ACTIVE_TARGET, ACTIVE_DATE, CHANNELS,
                    cfg, _shared_aperture_radius, _field_caches,
                    photometry_func,
                )
            )
            _processed_targets += 1
            _processed_channels += len(channel_results)
            if _stage4_run_root is None:
                _stage4_run_root = cfg.run_root

            stage4_postprocess_func(
                cfg, ACTIVE_TARGET, ACTIVE_DATE, channel_results, _stage4_run_root
            )

            _logger.info(f"[target] base target complete target={ACTIVE_TARGET} date={ACTIVE_DATE}")
            _vsx_shell = prepare_vsx_candidates_func(_logger, _args, _vsx_field, cfg)
            if _vsx_shell is not None:
                _vsx_cand, _VSX_MAG_MIN, _VSX_MAG_MAX = _vsx_shell
                if len(_vsx_cand) > 0:
                    print(f"\n{'='*55}")
                    print(f"  [VSX 額外目標] {len(_vsx_cand)} 顆 ({_VSX_MAG_MIN}-{_VSX_MAG_MAX} mag)")
                    print(f"{'='*55}")
                    run_vsx_targets_func(
                        _logger, _yaml, ACTIVE_TARGET, ACTIVE_DATE, _log_ts, CHANNELS,
                        cfg, _vsx_cand, _comp_refs_per_ch, _check_star_per_ch, _split_dir_per_ch,
                        _shared_aperture_radius, _active_cache, check_star,
                    )
                    emit_progress(_logger, f"VSX done target={ACTIVE_TARGET} count={len(_vsx_cand)}")
            emit_progress_done(
                _logger,
                f"main target target={ACTIVE_TARGET} date={ACTIVE_DATE}",
                _main_target_started,
            )
        finally:
            detach_file_handlers(_logger)

    if _processed_channels == 0:
        print(
            "[ERROR] photometry produced no channel outputs "
            f"(targets_processed={_processed_targets})"
        )
        return 1

    return 0


__all__ = [
    "_build_channel_cfg",
    "_build_targets_list",
    "_comp_signature",
    "_init_summary_logger",
    "_load_pipeline_yaml",
    "_normalize_channels",
    "_open_target_run",
    "_parse_cli_args",
    "_prepare_channel_fits_inputs",
    "_prepare_target_aperture_state",
    "_prepare_target_run_state",
    "_resolve_channels",
    "_run_channel_loop",
    "_run_channel_photometry",
    "_run_channels_for_target",
    "_run_single_channel",
    "_select_channel_comps",
    "_session_targets",
    "_summarize_field_key",
    "run_main",
]
