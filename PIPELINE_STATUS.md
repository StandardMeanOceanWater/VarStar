# PIPELINE_STATUS.md
最後更新：2026-03-10（對話 v12）

---

## 模組狀態

| 模組 | 狀態 | 備註 |
|------|------|------|
| Calibration.py | ✅ 完成 | 297 幀全數通過 |
| plate_solve.py | ✅ 完成 | AlVel 17/17、CCAnd 81/86、V1162Ori 187/190 |
| DeBayer_RGGB.py | ✅ 完成 | AlVel 17/17、CCAnd 81/81、V1162Ori 187/187 |
| photometry.py | ✅ 可用，待實作項目見下方 | V1162Ori 四通道實測通過 |
| period_analysis.py | ⚠️ 存在但未完整驗證 | LS 週期偵測可用；Broeg (2005) 未實作 |
| run_pipeline.py | ✅ 完成 | |

---

## 最新實測結果（V1162Ori，20251220）

| 通道 | 有效幀 | sigma-clipped | 跳過(alt<20°) | robust_σ |
|------|--------|---------------|---------------|----------|
| R | 126/165 | 22 | 22 | 0.622 mag |
| G1 | 151/165 | 13 | 22 | 0.110 mag |
| G2 | 157/165 | 7 | 22 | 0.155 mag |
| B | 156/165 | 8 | 22 | 0.104 mag |

LS 選最多有效幀通道（B，156幀）：Best period = 0.069 d（1.66 h），FAP = 1.62e-18。
**注意：此週期不可信，為觀測時段間隙造成的假頻率（spectral window function 旁瓣）。根本原因是 Broeg (2005) ensemble 正規化尚未實作，段間基線偏移未校正。**

R 通道 robust_σ 異常大（0.622 mag）：改機後近紅外靈敏度高，視場內 114 顆比較星候選飽和，APASS 僅匹配到 5 顆有效比較星，零點回歸不穩定。

---

## 不可翻案事項

### 資料夾結構
- `targets/` 位於 `data/targets/`，與 `shared_calibration/` 同層
- Light 幀直接放在 `raw/{date}/`，無 `light/` 子層
- ASTAP 星表副檔名：`.1476`（D80 星表）
- Master 幀存在 `shared_calibration/{date}/masters/`

### 儀器參數
- 像素尺寸：5.76 μm（原廠）
- Plate scale（拆色後 stride=2）：3.384 arcsec/px；實測：2.9702 arcsec/px
- 相機：Canon EOS 6D Mark II，已改機（IR 截止濾鏡移除），RGGB
- 觀測 ISO：3200（gain=0.232 e-/DN，read_noise=1.69 e-）
- 飽和閾值：11469 DN

### 測光設計
- 通道：R / G1 / G2 / B 各輸出獨立 CSV/PNG；G2 僅交叉驗證，**不參與 LS 週期分析**
- APASS 波段對應：R→r'、G1/G2→V、B→B
- 比較星選取：AAVSO 和 APASS 都有效時取數量較多的；AAVSO 門檻 ≥ 5 顆
- 孔徑：生長曲線法，第一通道估計，所有通道共用
- 時間系統：BJD_TDB，曝光中點，de432s
- 測光誤差：Merline & Howell (1995)
- 零點：robust iterative linear regression（3σ clipping，max 5 iter）
- 高度角截斷：altitude < 20°（airmass > 2.903，Young 1994）的幀跳過，不寫入 CSV；airmass=NaN 時不截斷
- AIRMASS_WARN_THRESHOLD = 2.0（alt ≈ 30°）：印 WARN 但繼續測光寫入 CSV
- sigma-clip：|m_var − median| > 3 × 1.4826 × MAD 的幀 ok 設 0，ok_flag="sigma_clip"，CSV 保留
- FLAG_SLOPE_DEVIATION（§3.8）：|a − 1.0| > 0.05 時 flag_slope_dev=1，寫入 CSV，LOG 記錄，幀保留
- LS 通道選擇：排除 G2，取其餘通道中 ok 幀數最多的

### LOG 系統
- LOG 檔：`output/photometry_{date}_{timestamp}.log`，每次執行新建
- 終端機：進度摘要、sigma-clip、最終統計、LS 結果
- LOG 檔（不進終端機）：Zero-point fit R² WARN、High airmass WARN、FITSFixedWarning、FLAG_SLOPE_DEVIATION

### 已知限制

| # | 限制 | 處理方式 |
|---|------|----------|
| 1 | 週期不確定度為下限（Kovacs 公式假設噪聲主導） | 文件記錄 |
| 2 | BIC 選階假設高斯誤差 | 文件記錄 |
| 3 | 預白化假設頻率間無耦合 | 文件記錄 |
| 4 | Bootstrap FAP 極低案例需更多迭代 | fap_bootstrap_max_iter: 5000 |
| 5 | 孔徑由比較星生長曲線決定，目標星 PSF 更胖時孔徑可能偏小 | 已知限制 |
| 6 | 孔徑硬邊界，無次像素加權 | 已知限制 |
| 7 | 距離與星表誤差雙重加權未正規化 | 已知限制 |
| 8 | ensemble_normalize() 為簡化版，Broeg (2005) 待實作 | **核心待實作項目** |
| 9 | 大氣消光修正暫不實作 | 視野<2°、仰角>30° 時差分抵消 |
| 10 | 改機 Canon 6D2 R 通道帶通偏移至近紅外 | FITS 標頭 IR_CUT=REMOVED |
| 11 | 暗場無溫度追蹤 | FITS 標頭記錄溫度差 |
| 12 | SIP 畸變 WCS 傳遞 | 已知問題 |
| 13 | SXPhe 20251122 全 4 幀曝光 1s，ASTAP 無法偵測星點 | 此批不做 |
| 14 | V1162Ori FITS 標頭 RA/DEC 錯誤 | yaml 填 ra_hint_h/dec_hint_deg（已修正） |
| 15 | DeBayer 輸出無 MID-OBS 標頭 | 從 DATE-OBS + EXPTIME/2 補算 |
| 16 | AAVSO API HTTP 400 | Fallback 到 APASS |
| 17 | check_star 無候選 | 設計上可接受 |
| 18 | V1162Ori R 通道比較星飽和嚴重（114/210 顆飽和） | 待 Broeg (2005) 實作後評估 |
| 19 | 零點回歸函數形式（線性是否足夠）尚未評估 | 待診斷後決定是否加入二次多項式選項 |

---

## 待實作清單

### 高優先（下次對話必做）

| 項目 | 說明 | 相關章節 |
|------|------|----------|
| **Broeg (2005) ensemble 正規化** | run_photometry_on_wcs_dir() 輸出 comp_lightcurves；測光後整體估計 Δ(t) 並從 m_var 中減去 | §3.9 |

### 中優先

| 項目 | 說明 |
|------|------|
| 零點回歸函數形式評估 | 診斷線性 vs 二次多項式；需先有 comp_lightcurves |
| 大氣消光修正 | 視野/仰角條件達到時實作；Chromey & Hasselbacher (1996) |

### 低優先

| 項目 | 說明 |
|------|------|
| Gaia DR3 比較星介面 | R 通道飽和問題的根治方案；視場密度高 10 倍 |
| 預白化 CSV 輸出 | yaml save_csv: true 時啟用 |
| 目標孔徑獨立生長曲線 | 取比較星與目標星孔徑較大值 |
| FLAG_SLOPE_DEVIATION 統計輸出 | 目前只寫 CSV 和 LOG；可在結尾印出 flag 幀數統計 |

---

## 執行環境
- venv Python：`C:\Users\JIN\.venv\Scripts\python.exe`
- ASTAP CLI：`C:\Program Files\astap\astap_cli.exe`
- ASTAP 星表：`C:\Program Files\astap\d80\`

## 執行命令
```bash
& C:\Users\JIN\.venv\Scripts\python.exe d:/VarStar/pipeline/photometry.py --channels R G1 B
```
