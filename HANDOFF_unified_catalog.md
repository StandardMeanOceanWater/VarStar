# 交接文件：Unified Field Catalog Refactor

## 狀態：實作完成，待首次驗證執行

完成日期：2026-04-08

---

## 已完成的修改（photometry.py）

### 新增：`_load_or_build_unified_catalog()`
- **取代**舊的 `_load_or_build_field_catalog(channel_name, ...)`（per-channel 版）
- 所有星表只查一次，輸出 `field_catalog_unified.csv`
- 輸出欄位：`ra_deg, dec_deg, V_mag, V_err, source_V, B_mag, B_err, source_B, R_mag, R_err, source_R, BT, VT, Gmag, BPmag, RPmag`
- **V** 優先序：AAVSO > APASS > Tycho2（VT→V 轉換已在 fetch_tycho2_cone 做）
- **B** 優先序：APASS B > Tycho2（BT→B = BT - 0.240*(BT-VT)，ESA 1997）
- **R**：Gaia DR3 RP→Rc ONLY（Riello 2021 公式）
- 同位置 <3" 的不同源：union-find 聚類，合併欄位，**不刪星**

### 修改：`auto_select_comps()`
- 加了 `band: str = "V"` 參數
- 從 unified catalog 取 `{band}_mag` 欄位做孔徑測光篩選
- 每通道獨立呼叫，不再共用一次選星結果

### 刪除：`_remap_comp_refs_to_band()`
- 完全移除（舊 patch 邏輯）
- 留有一行說明性注解

### main loop 更新
- 進入多通道迴圈前：預建 unified catalog 一次
- 首通道（band="V"）選星用於孔徑估算（`estimate_aperture_radius`）
- 每通道用 `_band_map_ch = {"R": "R", "G1": "V", "G2": "V", "B": "B"}` 對應 band 呼叫 `auto_select_comps(band=_band_ch)`

---

## 首次驗證步驟

### 1. 刪舊 per-channel 快取（必做，否則 unified 不會重建）
```
# Windows 路徑範例，依實際 output 目錄調整
del E:\VarStar\output\20251122\*\CCAnd\catalogs\field_catalog_*.csv
# 如有其他日期/目標也一併刪
```

### 2. 執行測光
```bash
cd E:/VarStar
python -u pipeline/photometry.py --target CCAnd --date 20251122 --raw
```

### 3. 驗證重點

| 項目 | 預期結果 | 舊版比較 |
|------|---------|---------|
| `field_catalog_unified.csv` 建立 | 輸出 catalogs/ 目錄下有此檔 | 以前是 4 個 field_catalog_{ch}.csv |
| R 通道比較星範圍 | 應可到 <9 等（無 dedup 截斷） | 舊版 ~8.3–10.2 等（可能更亮） |
| G1 通道比較星範圍 | 應可到 <9 等（APASS 亮星進來） | 舊版 ~9.0–10.7 等 |
| B 通道 source | source_B = APASS 或 Tycho2（無 Gaia、無 AAVSO） | 正確 |
| R 通道 source | source_R = GaiaDR3 only | 正確 |
| ok 幀數 | R 通道 ≥ 69/83（舊版 patch 水準） | 基準是 69/83 |

### 4. 看 log 的關鍵訊息
```
[unified catalog] saved: field_catalog_unified.csv (xxx rows)
[unified catalog] V=xxx  B=xxx  R=xxx  sources: AAVSO+APASS+GaiaDR3+Tycho2
[比較星] band=R 使用 GaiaDR3：N 顆
[比較星] band=V 使用 AAVSO+APASS+Tycho2：N 顆
```

---

## 已知風險 / 若出錯的排查方向

### `KeyError: 'V_mag'` 或類似欄位錯誤
- 原因：unified catalog 沒有該欄位（星表查詢全失敗）
- 排查：看 `[unified]` 那幾行的輸出，哪個星表回傳 0 星

### R 通道比較星 = 0
- 原因：Gaia local catalog 查詢失敗且 VizieR 也失敗
- 排查：`tools/local_catalog.py` 的 `query_local_gaia` 是否正常

### unified catalog 沒有重建（仍讀舊快取）
- 原因：`field_catalog_unified.csv` 已存在（可能殘留）
- 解法：手動刪除 `catalogs/field_catalog_unified.csv`

### 比較星選取失敗 RuntimeError
- 訊息：`band=R 在星表中找不到足夠的比較星`
- 解法：放寬 `comp_mag_range`，或確認 Gaia 查詢有結果

---

## 後續可選清理（本次未做）

- 刪除 `R_GAIA` 額外測光段（main loop line ~3319-3338），因為 R 通道現在直接用 unified catalog 的 Gaia 資料，R_GAIA 已多餘
- 刪除 `phot_catalog.py` 裡 `fetch_gaia_dr3_cone` 的 G1/G2/B channel 轉換分支（現在只需 R）
- `Cfg` dataclass 可移除 per-channel catalog 相關欄位（如有）

---

## 關鍵程式碼位置（修改後）

| 函式 | 位置 | 備註 |
|------|------|------|
| `_load_or_build_unified_catalog` | photometry.py ~line 1198 | 新函式 |
| `auto_select_comps` | photometry.py ~line 1508 | 有 band 參數 |
| main loop 預建 + 孔徑估算 | photometry.py ~line 3218 | 新邏輯 |
| main loop 通道迴圈 | photometry.py ~line 3262 | band_map_ch 對應 |
