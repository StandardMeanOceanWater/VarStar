# 變星測光管線 — 狀態快照
**最後更新：2026-03-09 17:30 UTC+8 | 本檔永遠只有一份，直接覆蓋更新**

---

## 1. 模組狀態

| 模組 | 檔名 | 狀態 | 備註 |
|------|------|------|------|
| 校正 | `Calibration.py` | ✅ 完成 | Dark 曝光縮放、逐通道黑電平、CAL_MODE、GAIN/RDNOISE 標頭 |
| 星圖解算 | `plate_solve.py` | ✅ 完成 | ASTAP 本機 + astrometry.net Colab，yaml 一次偵測 |
| Bayer 拆色 | `DeBayer_RGGB.py` | ✅ 完成 | Bayer pattern 自動偵測、WCS 傳遞修正（CRPIX、CD matrix、stride=2）|
| 測光（含標準 LS） | `photometry.py` + `Photometry.ipynb` | 🔧 主體完成，待修 | Cell 22 光變曲線圖缺誤差線；Cell 18 零點診斷圖缺誤差線 |
| 進階週期分析 | `period_analysis.py` | ✅ 完成 | 使用者選用進階模組 |
| 管線入口 | `run_pipeline.py` | ✅ 完成 | period_analysis 選用步驟（方案 A）已整合 |
| 環境設定 | `00_setup.ipynb` | ✅ 完成 | targets bug 修正、logs/ 目錄、選用模組測試 |
| 設定檔 | `observation_config.yaml` | ✅ 完成 | v2.1，period_analysis 區段全面更新 |
| 品質報告 | `quality_report.py` | ⏳ 待實作 | 第二批 D 組，預留介面 |

---

## 2. 不可翻案事項

以下決策已鎖定，不再討論替代方案。

### 儀器參數
- 像素尺寸：**5.76 μm**（各向同性近似，水平 5.753 / 垂直 5.769，差 0.28%；photonstophotos.net PTC 實測）
- Plate scale（拆色前）：**1.485 arcsec/px**（= 206.265 × 5.76 / 800）
- Plate scale（拆色後）：**2.970 arcsec/px**（× 2，stride=2 子採樣）
- 飽和閾值：11469 DN（14-bit 滿井 70%，線性響應安全上限）
- 相機：Canon EOS 6D Mark II，已改機移除 IR cut filter，Bayer pattern RGGB
- 望遠鏡：Vixen R200SS，焦距 800 mm，口徑 200 mm
- 觀測站：阿里山，23.481197°N，120.885415°E，海拔 2610 m
- Gain / Read noise（ISO 3200）：0.232 e⁻/DN / 1.69 e⁻
  - 來源：photonstophotos.net ISO 100 實測（K_ADC 7.428 e⁻/DN，σ_READ 7.305 DN）線性推算至 ISO 3200
  - **警告：高 ISO 類比/數位增益混用，read noise 為下限估計，待 PTC 實測取代**

### 校正流程
- Dark 縮放：`cal = light - bias - dark_rate × t_light`，`dark_rate = (master_dark - master_bias) / t_dark`
- 黑電平：逐通道扣除（R/G1/G2/B 各異），不取平均
- 負值保留：校正後負值不截斷（Poisson 噪聲統計正確性）
- Flat 公式：`flat_pure = master_flat - master_bias - dark_rate × t_flat`
- Master 合成：Remedian 二階近似中位數（Rousseeuw & Bassett, 1990）
- 時間系統：EXIF 時間為 UTC+8，DATE-OBS 寫入 UTC，BJD_TDB 以 astropy 計算

### 測光
- 時間系統：BJD_TDB（Eastman et al., 2010），曝光中點，ephemeris de432s，AAVSO 提交標準
- Airmass：Young (1994) 公式，X > 2.0 時警告並記錄仰角數值
- 孔徑：生長曲線自動決定（累積通量 95%），同視場所有星統一孔徑（Howell, 2006）
- 背景環：`r_in = r + 5 px`，`r_out = √(3r² + r_in²)`（確保背景面積 ≥ 孔徑面積 × 3）
- 飽和：閾值 11469 DN，目標星飽和標記 `flag=1` 後保留資料點
- 測光誤差：Merline & Howell (1995) CCD 噪聲方程式
- 誤差線原則：能加的全部加

### 比較星選取
- 星表優先順序：AAVSO VSP API（≥ 5 顆）→ APASS GAVO SCS → Gaia DR3（預留介面，未實作）
- 選取圓：以目標星為圓心，影像短邊像素數一半為半徑
- 距離加權：`w_i = 1 / (d_i + ε)²`，`ε = plate_scale / 2`（防除以零）
- AAVSO endpoint：`https://www.aavso.org/apps/vsp/api/chart/`（匿名可用，勿改）
- APASS endpoint：`https://dc.g-vo.org/apass/q/cone/scs.xml`，關鍵欄位 `ra`, `dec`, `mag_v`, `err_mag_v`
- APASS err_col 查詢順序：`["err_mag_v", "e_vmag", "e_v", "v_err", "vmag_err"]`
- 禁用：ASTAP 的 `variable_stars.csv`（已知變星清單，不是標準星）

### 週期分析
- 主方法：Lomb-Scargle（astropy），DFT 交叉驗證（Period04 方法，Lenz & Breger, 2005）
- 兩者結果並排輸出，結果一致才可信
- FAP：bootstrap 收斂迭代（max 1000 次，收斂閾值 5%），從 yaml 讀取
- 諧波數：BIC 最小化自動選階，上限從 yaml 讀取
- 相位零點：兩步迭代，亮度極大值（magnitude 最小值）= φ = 0（Breger et al., 1993，δ Sct 慣例）
- Pre-whitening 停止條件：殘差 S/N < 4（Breger et al., 1993）；多頻率聯立擬合（非逐步減去）
- 週期不確定度：傅立葉擬合殘差（非 LS 殘差）；`σ_f ≈ 1/(T × √N)`，`σ_P = σ_f / f²`
- 傅立葉階數自動選擇：覆蓋週期數 < 2 → n=1；2–5 → n=3；> 5 → n=min(6, fourier_n_max)
- **定位：使用者選用進階模組，不替換 `photometry.py` 標準輸出**

### 部署架構
- 步驟 1–3：本機執行（50 GB CR2 在本機，ASTAP 只能本機安裝）
- 步驟 4–5：本機或 Colab（輸入為已處理 FITS/CSV）
- 雙環境路徑分叉：只在 `observation_config.yaml` 做一次，各模組統一讀取
- 學生使用情境：步驟 1–3 由老師本機跑完，學生拿輸出 FITS 在 Colab 跑步驟 4–5

---

## 3. 待實作清單

### 進行中 / 近期
- [x] `period_analysis.py` 修訂版完成
- [x] `run_pipeline.py` 整合 `period_analysis` 為選用步驟（方案 A）
- [x] `00_setup.ipynb` Cell 4 `targets` 欄位讀取 bug 修正
- [x] `00_setup.ipynb` Cell 5 補建 `pipeline/logs/` 目錄
- [x] `00_setup.ipynb` Cell 8 加入 `period_analysis` 選用模組測試
- [ ] `Photometry.ipynb` Cell 22 補誤差線（光變曲線圖）
- [ ] `Photometry.ipynb` Cell 18 補誤差線（零點診斷散佈圖）
- [ ] `period_analysis.py` 整合至 `Photometry.ipynb`（選用 cell）
- [ ] 預白化 CSV 輸出（yaml `save_csv: true` 時啟用，code 已預留）
- [ ] Gaia DR3 比較星介面（目前輸出警告，未實作查詢）
- [ ] **ISO 3200 gain / read noise PTC 實測**
  - 方法：master flat pair 差影法（Janesick, 2001 §3）
  - `Calibration.py` 執行完成後輸出實測值至 `pipeline/ptc_result.json`
  - 欄位：`iso`、`gain_e_per_dn`、`read_noise_e`、`date`、`source_files`
  - 確認後手動填入 `observation_config.yaml`，取代現有推算值

### 第二批（報告書後）
- [ ] `quality_report.py`：觀測品質報告（G1 vs G2 residual rms = PRNU 估計）
- [ ] `aavso_catalog.py`：AAVSO 比較星快取，記錄 Chart ID 與查詢日期（確保可重現性）
- [ ] `color_transform.py`：儀器通道 → 標準 V/Rc/Bc（需 Landolt 標準星標定）
- [ ] 比較星跨夜一致性檢驗

---

## 4. 已知限制與 Bug

| 項目 | 狀態 | 說明 |
|------|------|------|
| Gain / Read noise | ⚠️ 推算值 | ISO 3200 數值為 ISO 100 線性外推，高 ISO 類比/數位增益混用，read noise 為下限估計 |
| SIP 畸變 WCS 傳遞 | ⚠️ 已知問題 | astrometry.net 有時產生 SIP 係數，DeBayer 拆色時需額外處理，尚未實作 |
| ASTAP 星表目錄名稱 | ⚠️ 待確認 | H18/D80 實際目錄名大小寫，等安裝完成後確認 |
| Gaia DR3 | ⚠️ 預留介面 | AAVSO 和 APASS 均不足時輸出警告，介面未實作 |
| Betelgeuse 飽和 | ℹ️ 已知限制 | V ≈ 0.5–1.5，必定飽和，需減光片或極短曝光 |
| 週期不確定度 | ℹ️ 下限 | Kovacs 公式假設噪聲主導 |
| BIC 選階 | ℹ️ 假設 | 假設高斯誤差，大氣閃爍主導時可能低估最佳階數 |
| 預白化 | ℹ️ 假設 | 假設頻率間無耦合，非線性脈動需謹慎解釋 |
| Bootstrap FAP 極低 | ℹ️ 建議 | FAP < 10⁻⁴ 案例建議手動提高 `max_iter` 至 5000 |
| `obs_sessions` 格式 | ℹ️ 相容性 | `targets` 欄位為列表格式；舊格式單一字串 `target` 仍相容，建議統一改為列表 |
| `JanesiCalibration.py` | ⚠️ 來源不明 | yaml 註解中出現此程式名稱，無任何實作或討論記錄，疑為某次對話提及但未實作，待查明 |

---

## 5. 參考文獻

Breger, M., Stich, J., Garrido, R., Martin, B., Jiang, S. Y., Li, Z. P., Hube, D. P., Ostermann, W., Paparo, M., & Scheck, M. (1993). Nonradial pulsation of the delta Scuti star BU Cancri in the Praesepe cluster. *Astronomy & Astrophysics, 271*, 482–486.

Eastman, J., Siverd, R., & Gaudi, B. S. (2010). Achieving better than 1 minute accuracy in the heliocentric and barycentric Julian dates. *Publications of the Astronomical Society of the Pacific, 122*(894), 935–946.

Honeycutt, R. K. (1992). CCD ensemble photometry on an inhomogeneous set of exposures. *Publications of the Astronomical Society of the Pacific, 104*(676), 435–440.

Howell, S. B. (2006). *Handbook of CCD astronomy* (2nd ed.). Cambridge University Press.

Janesick, J. R. (2001). *Scientific charge-coupled devices*. SPIE Press.

Lenz, P., & Breger, M. (2005). Period04 user guide. *Communications in Asteroseismology, 146*, 53–136.

Lomb, N. R. (1976). Least-squares frequency analysis of unequally spaced data. *Astrophysics and Space Science, 39*(2), 447–462.

Merline, W. J., & Howell, S. B. (1995). A realistic model for point-sources imaged on array detectors. *Experimental Astronomy, 6*(3), 163–210.

Rousseeuw, P. J., & Bassett, G. W. (1990). The remedian: A robust averaging method for large data sets. *Journal of the American Statistical Association, 85*(409), 97–104.

Scargle, J. D. (1982). Studies in astronomical time series analysis. II. Statistical aspects of spectral analysis of unevenly spaced data. *The Astrophysical Journal, 263*, 835–853.

VanderPlas, J. T. (2018). Understanding the Lomb-Scargle periodogram. *The Astrophysical Journal Supplement Series, 236*(1), 16.

Young, A. T. (1994). Air mass and refraction. *Applied Optics, 33*(6), 1108–1110.

---

## 6. 修訂歷程

歷程由此次對話起點開始累積，新的記錄在上方。

---

### 2026-03-09 17:30 UTC+8

**對話主題：`run_pipeline.py`、`00_setup.ipynb` 修訂；狀態檔重組；儀器參數修正**

**儀器參數修正（重要）**
- `pixel_size_um` 錯誤值 6.56 μm → 正確值 **5.76 μm**
- `plate_scale` 對應修正：1.692 → **1.485 arcsec/px**（拆色前）
- `sensor_db` 舊值來源不明，改為 photonstophotos.net ISO 100 線性推算至 ISO 3200：gain 0.232 e⁻/DN，read noise 1.69 e⁻
- `JanesiCalibration.py` 出處不明，記錄為已知問題待查

**`run_pipeline.py`**
- 加入 `period_analysis` 為第 5 步（選用，`default=False`，需明確指定）
- 新增前置條件檢查（搜尋 `output/photometry_*.csv`）
- 全部 `DESIGN_DECISIONS_v2.md` 引用修正為 `v5`

**`00_setup.ipynb`**
- Cell 4：`targets` 欄位讀取 bug 修正（`s.get('target')` → 讀取列表）
- Cell 5：新增 `pipeline/logs/` 及 `output/period_analysis/` 目錄建立
- Cell 8：加入 `period_analysis` 選用模組測試，失敗不影響標準管線

**`PIPELINE_STATUS.md`**
- 結構重組（模組狀態 / 不可翻案 / 待實作 / 已知限制 / 參考文獻 / 修訂歷程）
- 從三份舊狀態檔萃取並補入：背景環公式、API endpoint、距離加權公式、傅立葉選階規則、完整參考文獻
- 新增待實作項目：Calibration.py PTC 輸出 JSON

---

### （歷程起點，本次對話之前的記錄未完整保存）
