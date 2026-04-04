"""
regen_zp_diag.py — 從現有 CSV + catalog 重新生成 ZP 診斷圖
只需跑第一幀，不重跑全部測光。

用法：
  python regen_zp_diag.py --target CCAnd --date 20251122 --out-tag sat65536
  python regen_zp_diag.py --target CCAnd --date 20251122   # 預設 output 目錄
  python regen_zp_diag.py --all-targets --date 20251122 --out-tag sat65536
"""
import sys, argparse, glob, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as _mticker
import datetime as _dt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time as ATime

warnings.filterwarnings("ignore")

# ── photometry.py 裡的工具函式直接 import ──────────────────
sys.path.insert(0, str(Path(__file__).parent))
from photometry import aperture_photometry, robust_zero_point, is_saturated, load_pipeline_config, cfg_from_yaml

DATA_ROOT = Path(r"D:\VarStar\data\targets")

# ──────────────────────────────────────────────────────────────
def _get_wcs_and_data(fits_path):
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float32)
        hdr  = hdul[0].header
    wcs = WCS(hdr)
    return data, wcs, hdr

def _inst_mag(flux):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(flux > 0, -2.5 * np.log10(flux), np.nan)

def regen_one(target, date, out_tag, channels, cfg_yaml):
    """重新生成一個目標一個日期的 ZP 診斷圖（所有通道）"""
    out_dir_name = f"output_{out_tag}" if out_tag else "output"
    tgt_dir  = DATA_ROOT / target
    out_dir  = tgt_dir / out_dir_name
    cat_dir  = out_dir / "catalogs"
    diag_dir = out_dir / "calibration_diag"
    diag_dir.mkdir(parents=True, exist_ok=True)

    # 讀 catalog（合併 AAVSO + APASS + Tycho2）
    cat_files = list(cat_dir.glob("catalog_*.csv"))
    if not cat_files:
        print(f"[SKIP] {target}: 找不到 catalog CSV in {cat_dir}")
        return
    cats = []
    for cf in cat_files:
        try:
            cats.append(pd.read_csv(cf))
        except Exception:
            pass
    catalog = pd.concat(cats, ignore_index=True).drop_duplicates(
        subset=["ra_deg","dec_deg"]).reset_index(drop=True)
    print(f"[{target}] catalog: {len(catalog)} 顆比較星")

    for ch in channels:
        # ── 找 CSV ──────────────────────────────────────────
        csv_path = out_dir / f"photometry_{ch}_{date}.csv"
        if not csv_path.exists():
            print(f"  [{ch}] CSV 不存在，跳過")
            continue
        df = pd.read_csv(csv_path)
        df_ok = df[df["ok"] == 1].copy()

        # ── 找 split FITS（第一張）──────────────────────────
        split_dir = tgt_dir / "split" / ch
        fits_files = sorted(split_dir.glob("*.fits"))
        if not fits_files:
            print(f"  [{ch}] 找不到 split FITS，只畫左圖")
            first_frame_data = None
        else:
            first_fits = fits_files[0]
            print(f"  [{ch}] 使用第一幀: {first_fits.name}")
            try:
                img, wcs, hdr = _get_wcs_and_data(first_fits)
                # 讀 cfg 取孔徑設定
                cfg = cfg_from_yaml(cfg_yaml, target, date, channel=ch, out_tag=out_tag)
                r    = float(cfg.aperture_radius)
                r_in  = r * 2.5 if cfg.annulus_r_in  is None else cfg.annulus_r_in
                r_out = r * 4.0 if cfg.annulus_r_out is None else cfg.annulus_r_out

                # 對每顆比較星做孔徑測光
                h, w = img.shape
                m_cat_list, m_inst_list = [], []
                for _, row in catalog.iterrows():
                    px, py = wcs.all_world2pix(row["ra_deg"], row["dec_deg"], 0)
                    if not (np.isfinite(px) and np.isfinite(py)):
                        continue
                    if not (0 <= px < w and 0 <= py < h):
                        continue
                    ph = aperture_photometry(img, float(px), float(py), r, r_in, r_out)
                    # 飽和剔除已停用（cfg.saturation_threshold = 65536 = 全開）
                    # V1162Ori 有 4 等亮星；ZP scatter 保留亮端資料，依賴 R² 監控非線性
                    if ph.get("ok") == 1 and ph.get("flux_net", 0) > 0:
                        m_cat_list.append(float(row["vmag"]))
                        m_inst_list.append(_inst_mag(ph["flux_net"]))

                m_cat_arr  = np.array(m_cat_list)
                m_inst_arr = np.array(m_inst_list)

                # 目標星星等 → 動態篩選範圍
                tgt_vmag = float(getattr(cfg, "vmag_approx", np.nan))
                if np.isfinite(tgt_vmag):
                    MAG_BRIGHT = tgt_vmag - 1.0
                    MAG_FAINT  = tgt_vmag + 1.5
                else:
                    MAG_BRIGHT, MAG_FAINT = 8.0, 12.0
                mag_ok = (m_cat_arr >= MAG_BRIGHT) & (m_cat_arr <= MAG_FAINT) \
                         & np.isfinite(m_cat_arr) & np.isfinite(m_inst_arr)
                mc_filt = m_cat_arr[mag_ok]
                mi_filt = m_inst_arr[mag_ok]
                tgt_ra, tgt_dec = cfg.target_radec_deg
                tpx, tpy = wcs.all_world2pix(tgt_ra, tgt_dec, 0)
                if np.isfinite(tpx) and np.isfinite(tpy):
                    tph = aperture_photometry(img, float(tpx), float(tpy), r, r_in, r_out)
                    tgt_minst = _inst_mag(tph.get("flux_net", 0)) if tph.get("ok")==1 else np.nan
                else:
                    tgt_minst = np.nan

                # ZP 擬合（只用篩選後的星）
                fit = robust_zero_point(mi_filt, mc_filt) if len(mc_filt) >= 3 else None
                r2_str = f"{fit['r2']:.3f}" if fit else "N/A"
                first_frame_data = (mc_filt, mi_filt, fit, tgt_vmag, tgt_minst)
                print(f"  [{ch}] 比較星 n={len(mc_filt)} (篩{MAG_BRIGHT:.1f}–{MAG_FAINT:.1f}等), R²={r2_str}")
            except Exception as e:
                print(f"  [{ch}] 第一幀測光失敗: {e}")
                first_frame_data = None

        # ── 繪圖 ────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                                  gridspec_kw={"width_ratios": [2, 1]})
        time_key = "bjd_tdb" if "bjd_tdb" in df_ok.columns else "mjd"

        # 左圖：RMS vs time
        ax_ts = axes[0]
        if "zp_residual_rms" in df_ok.columns and np.isfinite(df_ok["zp_residual_rms"]).any():
            ax_ts.plot(df_ok[time_key], df_ok["zp_residual_rms"],
                       "o-", ms=3, lw=0.8, color="steelblue", alpha=0.7)
            med = df_ok["zp_residual_rms"].median()
            ax_ts.axhline(med, color="red", lw=1, ls="--",
                          label=f"median={med:.4f}")
            ax_ts.set_xlabel(time_key.upper())
            ax_ts.set_ylabel("Zero-point residual RMS (mag)")
            ax_ts.set_title("Zero-point residual RMS vs. time")
            ax_ts.legend(fontsize=8)
            ax_ts.grid(True, alpha=0.3)
            ax_ts.xaxis.set_major_formatter(
                _mticker.FuncFormatter(lambda v,_: f"{v:.2f}"))

            # Local Time 上軸
            tz = 8
            ts_arr = df_ok[time_key].dropna().values
            if len(ts_arr) > 0:
                bjd_lo, bjd_hi = float(ts_arr.min()), float(ts_arr.max())
                tmin = (ATime(bjd_lo, format="jd", scale="tdb").to_datetime()
                        + _dt.timedelta(hours=tz))
                tmax = (ATime(bjd_hi, format="jd", scale="tdb").to_datetime()
                        + _dt.timedelta(hours=tz))
                tks = []
                cur = tmin.replace(second=0, microsecond=0)
                cur = cur.replace(minute=(cur.minute//30)*30)
                while cur <= tmax + _dt.timedelta(minutes=1):
                    utc = cur - _dt.timedelta(hours=tz)
                    tks.append(ATime(utc).jd)
                    cur += _dt.timedelta(minutes=30)
                ax2 = ax_ts.twiny()
                ax2.set_xlim(ax_ts.get_xlim())
                ax2.xaxis.set_major_locator(_mticker.FixedLocator(tks))
                ax2.xaxis.set_major_formatter(_mticker.FuncFormatter(
                    lambda v,_: (ATime(v,format="jd",scale="tdb").to_datetime()
                                 +_dt.timedelta(hours=tz)).strftime("%H:%M")))
                ax2.tick_params(axis="x", colors="steelblue",
                                labelsize=7, labelcolor="steelblue")
                ax2.text(-0.002, 1.01, "Local Time",
                         transform=ax2.transAxes, ha="right", va="bottom",
                         fontsize=7, color="steelblue", clip_on=False)
        else:
            ax_ts.text(0.5, 0.5, "No residual data",
                       transform=ax_ts.transAxes, ha="center", va="center")

        # 右圖：Frame 1 scatter
        ax_sc = axes[1]
        if first_frame_data is not None:
            mc, mi, fit, tgt_mc, tgt_mi = first_frame_data
            ok = np.isfinite(mc) & np.isfinite(mi)
            ax_sc.scatter(mc[ok], mi[ok], s=18, alpha=0.6,
                          color="steelblue", label=f"comp (n={ok.sum()})")
            if fit and np.isfinite(fit.get("a", np.nan)):
                a, b, r2 = fit["a"], fit["b"], fit.get("r2", np.nan)
                if ok.any():
                    xl = np.linspace(mc[ok].min(), mc[ok].max(), 100)
                    ax_sc.plot(xl, a*xl+b, "r-", lw=1.5,
                               label=f"$m_{{inst}}={a:.3f}\\,m_{{cat}}+({b:.3f})$  $R^2={r2:.3f}$")
            if np.isfinite(tgt_mc) and np.isfinite(tgt_mi):
                ax_sc.scatter([tgt_mc], [tgt_mi], s=80, marker="o",
                              facecolors="none", edgecolors="red", linewidths=1.8,
                              zorder=5, label=f"target  $m_{{cat}}$={tgt_mc:.1f}")
            if ok.any():
                xr = np.array([mc[ok].min(), mc[ok].max()])
                ax_sc.plot(xr, xr, "k--", lw=0.8, alpha=0.4, label="ideal")
            ax_sc.invert_yaxis()
            ax_sc.set_xlabel("$m_{cat}$ (V)")
            ax_sc.set_ylabel("$m_{inst}$")
            ax_sc.set_title(f"Frame 1 ZP scatter ({MAG_BRIGHT:.1f}–{MAG_FAINT:.1f} mag)")
            ax_sc.legend(fontsize=7, frameon=False)
            ax_sc.grid(True, alpha=0.3)
        else:
            ax_sc.text(0.5, 0.5, "No first-frame data",
                       transform=ax_sc.transAxes, ha="center", va="center")

        fig.suptitle(
            f"Zero-point diagnostics  |  {target}  channel={ch}  "
            f"photometry_{ch}_{date}",
            fontsize=10)
        fig.tight_layout()

        out_png = diag_dir / f"zp_overview_{ch}_{ch}_{date}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  [{ch}] ZP 圖已輸出: {out_png}")


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target",      default=None)
    p.add_argument("--all-targets", action="store_true")
    p.add_argument("--date",        required=True)
    p.add_argument("--out-tag",     default=None)
    p.add_argument("--channels",    nargs="+", default=["G1","G2","R","B"])
    args = p.parse_args()

    _yaml = load_pipeline_config()

    if args.all_targets:
        targets = [d.name for d in DATA_ROOT.iterdir() if d.is_dir()]
    else:
        targets = [args.target]

    for tgt in targets:
        out_tag = args.out_tag or ""
        out_dir_name = f"output_{out_tag}" if out_tag else "output"
        csv_check = DATA_ROOT / tgt / out_dir_name / f"photometry_G1_{args.date}.csv"
        if not csv_check.exists():
            print(f"[SKIP] {tgt}: 無 {out_dir_name}/photometry_G1_{args.date}.csv")
            continue
        print(f"\n{'='*50}\n目標: {tgt}\n{'='*50}")
        regen_one(tgt, args.date, out_tag, args.channels, _yaml)
