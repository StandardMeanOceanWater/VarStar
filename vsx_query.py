"""
vsx_query.py — AAVSO VSX cone search 工具

用法：
    python vsx_query.py --ra 83.14 --dec -7.21 --radius 1.55
    python vsx_query.py --ra 83.14 --dec -7.21 --out D:/VarStar/data/targets/V1162Ori/output

輸出：
    - 終端機表格（依 V mag 排序）
    - CSV：{out_dir}/vsx_candidates_{ra:.3f}_{dec:+.3f}.csv
"""

import argparse
import sys
from pathlib import Path

import xml.etree.ElementTree as ET

import pandas as pd
import requests

# VSX VOTable API 端點（www.aavso.org 301 轉址至此）
VSX_URL = "https://vsx.aavso.org/index.php"
DEFAULT_RADIUS_DEG = 1.55   # Canon 6D2 + R200SS 對角半徑（來自 yaml vsx_search_radius_deg）
REQUEST_TIMEOUT_SEC = 30


def query_vsx(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float = DEFAULT_RADIUS_DEG,
) -> pd.DataFrame | None:
    """
    呼叫 VSX VOTable API，回傳 DataFrame；失敗時回傳 None 並印警告。

    Parameters
    ----------
    ra_deg     : 視場中心 RA（度，十進制）
    dec_deg    : 視場中心 Dec（度，十進制）
    radius_deg : 搜尋半徑（度）

    Returns
    -------
    DataFrame，欄位：
        auid, name, const, coords_j2000, var_type,
        max_mag, max_band, min_mag, min_band,
        epoch, period, rise_dur, spec_type, discoverer,
        ra_deg, dec_deg（從 coords_j2000 解析）
    或 None（網路 / 解析失敗）
    """
    # coords 格式：「RA Dec」，十進位度數，空白分隔
    coords_str = f"{ra_deg:.6f} {dec_deg:+.6f}"
    params = {
        "view": "query.votable",
        "coords": coords_str,
        "size": f"{radius_deg:.4f}",
    }

    try:
        resp = requests.get(VSX_URL, params=params, timeout=REQUEST_TIMEOUT_SEC)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[WARN] VSX 查詢失敗（網路）：{exc}", file=sys.stderr)
        return None

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as exc:
        print(f"[WARN] VSX VOTable XML 解析失敗：{exc}", file=sys.stderr)
        return None

    # 取出欄位順序（FIELD id 屬性）
    fields = [el.get("id") for el in root.iter("FIELD")]
    rows_xml = list(root.iter("TR"))

    if not rows_xml:
        print(f"[INFO] VSX 搜尋結果為空（r={radius_deg}°，共 0 筆）")
        return pd.DataFrame()

    col_map = {
        "auid":      "auid",
        "name":      "name",
        "const":     "const",
        "radec2000": "coords_j2000",
        "varType":   "var_type",
        "maxMag":    "max_mag",
        "maxPass":   "max_band",
        "minMag":    "min_mag",
        "minPass":   "min_band",
        "epoch":     "epoch",
        "period":    "period",
        "riseDur":   "rise_dur",
        "specType":  "spec_type",
        "disc":      "discoverer",
    }

    rows = []
    for tr in rows_xml:
        tds = [td.text or "" for td in tr.findall("TD")]
        raw = dict(zip(fields, tds))
        row = {df_key: raw.get(vot_key, "") for vot_key, df_key in col_map.items()}
        rows.append(row)

    df = pd.DataFrame(rows)

    # coords_j2000 格式：「83.14358000,-7.21114000」（逗號分隔十進位度數）
    ra_list, dec_list = [], []
    for coords in df["coords_j2000"]:
        ra_d, dec_d = _parse_coords(str(coords))
        ra_list.append(ra_d)
        dec_list.append(dec_d)
    df["ra_deg"]  = ra_list
    df["dec_deg"] = dec_list

    # max_mag 轉數值（供排序）
    df["max_mag_num"] = pd.to_numeric(df["max_mag"], errors="coerce")

    return df


def _parse_coords(coords_str: str) -> tuple[float, float]:
    """
    解析 VSX coords_j2000 字串，回傳 (ra_deg, dec_deg)。
    支援格式：
      「83.14358000,-7.21114000」（逗號分隔十進位，VSX 實際回傳格式）
      「HH MM SS.s ±DD MM SS」（空白分隔 HMS，舊格式備用）
    解析失敗回傳 (nan, nan)。
    """
    coords_str = coords_str.strip()
    # 逗號分隔十進位
    if "," in coords_str:
        parts = coords_str.split(",")
        try:
            return float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            pass
    # 空白分隔 HMS
    parts = coords_str.split()
    try:
        if len(parts) == 6:
            ra_h, ra_m, ra_s = float(parts[0]), float(parts[1]), float(parts[2])
            dec_sign = -1.0 if parts[3].startswith("-") else 1.0
            dec_d = float(parts[3].lstrip("+-"))
            dec_m, dec_s = float(parts[4]), float(parts[5])
            ra_deg  = (ra_h + ra_m / 60.0 + ra_s / 3600.0) * 15.0
            dec_deg = dec_sign * (dec_d + dec_m / 60.0 + dec_s / 3600.0)
            return ra_deg, dec_deg
        elif len(parts) == 2:
            return float(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        pass
    return float("nan"), float("nan")


def print_table(df: pd.DataFrame, max_rows: int = 50) -> None:
    """終端機表格輸出，依 max_mag_num 排序。"""
    if df.empty:
        print("（無資料）")
        return

    show = df.sort_values("max_mag_num", na_position="last").head(max_rows)
    header = f"{'#':<4} {'Name':<20} {'Type':<12} {'MaxMag':<8} {'Band':<5} {'MinMag':<8} {'Period(d)':<12} {'RA(deg)':<10} {'Dec(deg)':<10} {'AUID'}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for i, (_, row) in enumerate(show.iterrows(), 1):
        period_str = str(row["period"]) if str(row["period"]).strip() else "—"
        print(
            f"{i:<4} {str(row['name']):<20} {str(row['var_type']):<12} "
            f"{str(row['max_mag']):<8} {str(row['max_band']):<5} "
            f"{str(row['min_mag']):<8} {period_str:<12} "
            f"{row['ra_deg']:<10.4f} {row['dec_deg']:<10.4f} {row['auid']}"
        )
    print(sep)
    print(f"共 {len(df)} 筆（顯示前 {min(len(df), max_rows)} 筆，依 MaxMag 排序）")


def save_csv(df: pd.DataFrame, out_dir: Path, ra_deg: float, dec_deg: float) -> Path:
    """存成 CSV，檔名含座標。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"vsx_candidates_{ra_deg:.3f}_{dec_deg:+.3f}.csv"
    out_path = out_dir / fname
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AAVSO VSX cone search — 查詢視場內變星候選目標"
    )
    parser.add_argument("--ra",     type=float, required=True,  help="視場中心 RA（度，十進制）")
    parser.add_argument("--dec",    type=float, required=True,  help="視場中心 Dec（度，十進制）")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS_DEG,
                        help=f"搜尋半徑（度，預設 {DEFAULT_RADIUS_DEG}）")
    parser.add_argument("--out",    type=Path,  default=Path("."),
                        help="CSV 輸出目錄（預設：目前目錄）")
    parser.add_argument("--max-rows", type=int, default=50,
                        dest="max_rows", help="終端機顯示上限（預設 50）")
    args = parser.parse_args(argv)

    print(f"[VSX] 查詢中：RA={args.ra:.4f}°  Dec={args.dec:+.4f}°  r={args.radius}°")
    df = query_vsx(args.ra, args.dec, args.radius)

    if df is None:
        return 1
    if df.empty:
        return 0

    print_table(df, max_rows=args.max_rows)

    out_path = save_csv(df, args.out, args.ra, args.dec)
    print(f"[VSX] saved → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
