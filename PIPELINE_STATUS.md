# 變星測光管線 — 狀態快照
**最後更新：2026-03-28 UTC+8 | 版號 v1.61 | 本檔永遠只有一份，直接覆蓋更新**

---

## 📝 修訂規則（v1.01 起施行）

**目的**：防止前後修改互相覆蓋，確保全局一致性

**鎖定區段** (LOCKED)：修改前需通知使用者
| 檔案 | 區段 | 行號 | 內容 |
|------|------|------|------|
| `observation_config.yaml` | 2. 望遠鏡設定 | 18–26 | 儀器參數（像素、焦距、飽和等） |
| `observation_config.yaml` | 3. 相機設定 | 28–46 | 相機規格、Bayer 模式 |
| `observation_config.yaml` | 4. 觀測 Session | 50–73 | 觀測站座標、日期、目標列表 |
| `observation_config.yaml` | 6. 星圖解算 | 117–139 | ASTAP/astrometry.net 後端設定 |
| `PIPELINE_STATUS.md` | 1. 模組狀態 | 6–18 | 模組版號狀態表 |
| `CLAUDE.md` | 全文 | — | 專案規範書，勿修改（存檔專用） |

**開放區段** (OPEN)：可單項修改
| 檔案 | 區段 | 內容 |
|------|------|------|
| `observation_config.yaml` | 5. 目標星 hint | 目標座標、顯示名稱、近視星等 |
| `observation_config.yaml` | 8–14. 測光參數 | 孔徑、背景環、測光誤差、時間系統 |
| `PIPELINE_STATUS.md` | 修訂歷程 | 每次改動記錄 |

**版號同步規則**
- 主版本改動（如 v0.99 → v1.01）：
  1. `observation_config.yaml` 第 3 行版號更新 ✓
  2. `PIPELINE_STATUS.md` 第 2 行版號更新 ✓
  3. `CLAUDE.md` 第 3 行版號更新（見下方）
  4. 新增修訂歷程項目
  5. `git commit -m "vX.XX: <brief description>"`

---

## 0. 輸出項目一覽

每次執行 `photometry.py` 後，在 `data/targets/{TARGET}/output/` 下產生以下檔案。
`{ch}` = R / G1 / G2 / B，`{date}` = YYYYMMDD，`{ts}` = 執行時間戳。

| # | 類型 | 檔名格式 | 子目錄 | 說明 |
|---|------|----------|--------|------|
| 1 | CSV | `photometry_{ch}_{date}.csv` | `output/` | 主要測光結果；欄位含 BJD_TDB / mag / mag_err / flux / flag 等 |
| 2 | PNG | `light_curve_{ch}_{date}.png` | `output/` | 原始光變曲線圖（ensemble 正規化後） |
| 3 | PNG | `zp_overview_{ch}_{date}.png` | `output/zp_diag/` | 零點診斷總覽：殘差時序 + 第一幀散佈圖（yaml `save_zeropoint_diagnostic: true`） |
| 4 | PNG | `zp_diag_{frame_stem}.png` | `output/zp_diag/` | 每幀零點回歸圖（幀層級，數量 = 有效幀數） |
| 5 | CSV | `catalog_{source}.csv` | `output/catalogs/` | 比較星星表（source = AAVSO / APASS / Tycho2 / GaiaDR3） |
| 6 | CSV | `catalog_GaiaDR3_Rc.csv` | `output/catalogs/` | Gaia G→Rc 轉換星表（R 通道強制附加） |
| 7 | PNG | `periodogram_{ch}_{date}.png` | `output/period_analysis/` | Lomb-Scargle 週期圖 |
| 8 | PNG | `phase_fold_{ch}_{date}.png` | `output/period_analysis/` | 相位折疊曲線圖（Fourier 擬合疊加） |
| 9 | PNG | `G1G2_ratio_{date}.png` | `output/` | G1/G2 通量比值時序圖（同時跑 G1+G2 才產生） |
| 10 | PNG | `periodogram_G1G2ratio_{date}.png` | `output/period_analysis/` | G1/G2 比值 LS 週期圖 |
| 11 | PNG | `phase_fold_G1G2ratio_{date}.png` | `output/period_analysis/` | G1/G2 比值相位折疊圖 |
| 12 | LOG | `photometry_{date}_{ts}.log` | `output/` | 完整執行日誌（`FileHandler encoding=utf-8`） |

> **注意**：`pipeline/logs/pipeline_{date}.log`（yaml 中的 `log_file` 欄位）目前未被程式使用，實際 log 存於 `output/` 下。

---

## 1. 模組狀態

| 模組 | 檔名 | 狀態 | 備註 |
|------|------|------|------|
| 校正 | `Calibration.py` | ✅ 完成（已實測） | Bug 修正 v2；297 幀全數通過 |
| 星圖解算 | `plate_solve.py` | ✅ 完成（已實測） | hint 單位修正；V1162Ori 187/190 |
| Bayer 拆色 | `DeBayer_RGGB.py` | ✅ 完成 | WCS 子採樣修正已實作 |
| 測光（含標準 LS） | `photometry.py` + `Photometry.ipynb` | ✅ 完成實測（v1.35） | **20251122**：AlVel 0/81、CCAnd 0/81；**20251220**：Orion 多目標批次進行中。V1162Ori 已產出整合式 3 欄分析圖。 |
| 進階週期分析 | `period_analysis.py` | ✅ 完成 | 相對路徑支援已加入 |
| 管線入口 | `run_pipeline.py` | ✅ 完成 | period_analysis 選用步驟已整合 |
| 環境設定 | `00_setup.ipynb` | ✅ 完成 | Cell 5 無 `light/`、Cell 6 支援 `.1476` |
| 設定檔 | `observation_config.yaml` | ✅ 完成 | dark_temp_c / light_temp_c 已加入 |
| 品質報告 | `quality_report.py` | ⏳ 待實作 | 第二批 D 組，預留介面 |

---

## 2. 不可翻案事項

以下決策已鎖定，不再討論替代方案。

### 資料夾結構
- `targets/` 位於 `data/targets/`，與 `shared_calibration/` 同層，均在 `data/` 下
- Light 幀直接放在 `raw/{date}/`，無 `light/` 子層
- ASTAP 星表副檔名：`.1476`（D80 星表）

### 儀器參數
- 像素尺寸：5.76 μm（Canon 原廠規格；ASTAP 實測 plate scale 1.481 arcsec/px 吻合）
- Plate scale：1.485 arcsec/px（206.265 × 5.76 / 800）
- 飽和閾值：11469 DN
- 相機：Canon EOS 6D Mark II，已改機移除 IR cut filter，Bayer pattern RGGB

### 校正流程
- 黑電平：逐通道讀取，負值保留
- Master 合成：Remedian 二階近似中位數
- Dark 縮放：依曝光時間比例縮放
- 暗場溫度：`dark_temp_c` 選填；子目錄選取以 `light_temp_c` 為基準選最接近者；無 `light_temp_c` 時選最低溫

### 星圖解算
- ASTAP hint：`-ra` 單位為**小時**（0–24），`-spd` 單位為度（= 90 + Dec）
- 不縮減 `search_radius`（縮減會造成 ASTAP 搜尋格網步驟起點偏移）
- `fov_override_deg: 0`（讓 ASTAP 自動估算 FOV）
- NINA 標頭 RA/DEC 不可信（實測為北極點座標），必須傳 hint

### 測光
- 時間系統：BJD_TDB，曝光中點，de432s
- 比較星星表階層：AAVSO → Tycho-2（vmag≤11，comp_mag_min=7.0）→ APASS → Gaia DR3
- G1/G2：Gaia G→V（Riello 2021）；R：Gaia G→Rc（Riello 2021）+ 強制附加 R_GAIA；B：Gaia G→B（使用者公式，BP-RP<2）
- 孔徑：生長曲線法，固定孔徑
- 背景：背景環中位數
- 測光誤差：Merline & Howell (1995)
- Airmass：Young (1994)
- 飽和篩除：peak_flux > 11469 DN 的候選比較星強制排除
- 星表存檔：`output/catalogs/catalog_{source}.csv`（每個星表一份）

### 週期分析
- 主方法：Lomb-Scargle（astropy），DFT 交叉驗證
- FAP：bootstrap 收斂迭代（max 1000 次，收斂閾值 5%），從 yaml 讀取
- 諧波數：BIC 最小化自動選階，上限從 yaml 讀取
- 相位零點：亮度極大值（magnitude 最小值）= φ = 0，兩步迭代驗證
- Pre-whitening 停止條件：S/N < 4（Breger et al., 1993）
- 週期不確定度：傅立葉擬合殘差（非 LS 殘差）
- **定位：使用者選用進階模組，不替換 `photometry.py` 標準輸出**

### 部署架構
- 步驟 1–3：本機執行
- 步驟 4–5：本機或 Colab
- 雙環境路徑分叉：只在 `observation_config.yaml` 做一次，各模組統一讀取
- `project_root` 支援相對路徑（建議填 `".."`），以 yaml 位置為錨點解析

---

## 3. 待實作清單

- [x] `period_analysis.py` 修訂版完成
- [x] `run_pipeline.py` 整合 `period_analysis` 為選用步驟（方案 A）
- [x] `00_setup.ipynb` Cell 4 `targets` 欄位讀取 bug 修正
- [x] `00_setup.ipynb` Cell 5 補建 `pipeline/logs/` 目錄
- [x] `00_setup.ipynb` Cell 8 加入 `period_analysis` 選用模組測試
- [x] `Calibration.py` 暗場溫度子目錄支援（`_find_dark_dir()`）
- [x] `Calibration.py` `targets` 列表迴圈（Master 幀共用，逐 target 校正）
- [x] `Calibration.py` / `photometry.py` / `period_analysis.py` 相對路徑支援
- [x] `observation_config.yaml` 加入 `dark_temp_c` / `light_temp_c`
- [x] `Calibration.py` `light_dir` 移除 `/light` 層
- [x] `00_setup.ipynb` Cell 5 確認無 `light/` 目錄建立
- [x] `00_setup.ipynb` Cell 6 星表驗證加入 `.1476` 副檔名
- [x] `targets/` 移入 `data/` 下（使用者手動完成目錄搬移）
- [x] `Calibration.py` bug 修正 v2：target 路徑錯誤（複數/單數 key 衝突）、em dash FITS 標頭
- [x] **Calibration 實測通過**：AlVel 17、CCAnd 86、SXPhe 4、V1162Ori 190，共 297 幀全數成功
- [x] `plate_solve.py` hint 機制修正：`-ra` 單位改為小時、移除 `effective_radius` 縮減邏輯
- [x] **plate_solve 實測通過**：V1162Ori 187/190（98.4%）；AlVel 17/17、CCAnd 79/86（跳過）
- [x] `photometry.py` 實測通過：V1162Ori R=113/187、G1=110/187、G2=123/187、B=91/187
- [x] AAVSO API endpoint 確認為 `/chart/`；星名需有空格（`display_name: "V1162 Ori"`）
- [x] APASS 改用 astroquery VizieR（`dc.g-vo.org` 已 404）；同時查 V/B/Rc 三波段
- [x] `period_analysis.py` 自動選 `m_var_norm`（ensemble 啟用時）
- [x] 預白化 CSV 輸出實作（`save_csv: true` 時輸出 `prewhitening_{target}_{ch}.csv`）
- [x] `observation_config.yaml` 加入 `observatory:` 區塊（`latitude_deg/longitude_deg/elevation_m`）
- [x] `comp_mag_min/max` yaml 覆蓋機制（固定 6–13 mag，避免依賴 `vmag_approx` 估算）
- [x] 觀測站實際座標已填入：Tataka Shandonpu Parking / Cingjing Observatory
- [x] CCAnd yaml hint 座標修正：原值錯填 23.647767h/38.584°，正確為 0.730003h/42.282°（V* CC And, SIMBAD）
- [x] AlVel yaml hint 座標修正：原值錯填 9.458297h/-44.682°，正確為 8.519801h/-47.666°（Al Vel, SIMBAD）；WCS 像素驗證確認解星正確
- [x] **V1162Ori plate_solve + debayer + photometry + period_analysis 重跑**（完成，2026-03-14）：WCS hint ra=5.534h dec=−7.257°，187 WCS／187 split／R=130/G1=120/G2=121/B=128 有效幀；⚠️ m_var_norm 末段幀異常（frames 0142/0143 值 7.35 vs 正常 ~10），週期分析結果不可信，需先修 ensemble 正規化
- [ ] 診斷 EXIF ISO 讀出為 0（FITS GAIN/RDNOISE 空白），在 photometry 前處理
- [x] Gaia DR3 比較星介面（`fetch_gaia_dr3_cone`，G→V/Rc/B 三通道轉換，⚠️ 尚未實測）
- [ ] APASS 本機快取：`download_apass_cache.py` 已實作（2026-03-14），各目標視野 CSV 存 `data/catalogs/apass/`；下載中
- [ ] `quality_report.py` 實作（第二批 D 組）
- [ ] SXPhe 4 幀全部失敗（1s 曝光，星點太少）—— 不補觀測，標記為已知限制
- [ ] 起霧偵測模組：`fog_detect.py` 已實作（2026-03-14）；門檻探索工具完成；待辦：(a) 逐幀生長曲線孔徑，(b) FWHM 估計器修復（DAOStarFinder NaN），(c) 三指標整合加權評分；實測方案：逐幀測試各判準 → 人眼核對表格 → 找最適門檻
- [ ] ZP R² 門檻：由 V1162Ori 重跑結果與 AAVSO 週期比對後決定
- [ ] 初始幀篩選自動化：`select_comp_stars` 若第一幀品質不佳，自動往後試 N 幀（FWHM＋比較星數雙重條件）
- [ ] sigma_clip / ensemble 不穩診斷：frames 0142/0143 m_var_norm=7.35（正常~10），末段高氣團幀 ensemble 正規化崩潰；需分析原因並在 period_analysis 前加 sigma clip 保護
- [ ] VSX 查詢整合：視野中心查 6–9 等變星，長邊 1/2 圓內（`vsx_query.py` 已有草稿）
- [x] 多目標共用拆色檔架軌（session-centric）：已實作，一份 split FITS 跑多顆目標。
- [x] 整合式 3 欄式分析圖 (v1.35)：含 LS/DFT/Phase Fold，支援報告級字體與配色。
- [x] G1/G2 比值週期分析自動化：修正變數定義 Bug，整合入主流程。
- [/] 獵戶座 5 星批次執行：V1162Ori 完成，V1643Ori 進行中。

---

## 4. 已知限制

1. 週期不確定度為下限（Kovacs 公式假設噪聲主導）
2. BIC 選階假設高斯誤差，大氣閃爍主導時可能低估最佳階數
3. 預白化假設頻率間無耦合，非線性脈動需謹慎解釋
4. Bootstrap FAP 極低（< 10⁻⁴）案例建議手動提高 `max_iter` 至 5000
5. `astrometry.net` 後端依賴網路，離線環境只能用 ASTAP
6. `obs_sessions` 的 `targets` 欄位為列表格式；舊格式單一字串 `target` 仍相容，但建議統一改為列表
7. 暗場溫度子目錄命名規則：`{數字}C`（例如 `3.7C`），不支援負溫度格式
8. `photometry.py` 第 1 行為 Jupyter magic，`py_compile` 無法直接語法檢查
9. EXIF ISO 讀出為 0（rawpy 無法從 Canon 6D2 CR2 取得 ISO），FITS GAIN/RDNOISE 標頭空白；不影響校正，測光前需補值
10. NINA 望遠鏡指向座標未同步，FITS 標頭 `RA/DEC` 實測為北極點（RA≈5.39h, Dec≈+89.84°），`OBJCTRA/OBJCTDEC` 為零；plate_solve 依賴 yaml `targets` hint 修正
11. V1162Ori AAVSO 比較星僅 4 顆（aavso_min_stars 調降至 3），改用 APASS 43 顆
12. 改機 Canon 6D2 R 通道帶通延伸至近紅外，APASS Rc 僅為近似；嚴格轉換係數需 Landolt 標準場標定
13. Windows cp950 編碼：`photometry.py` 已在 import 區塊加入 `sys.stdout/stderr.reconfigure(encoding='utf-8')`，所有 Unicode 字符可正常輸出
14. `observatory:` 區塊座標：已更正為實際觀測站（塔塔加 / 清境觀星園）

---

## 5. 修訂歷程

---

### 2026-03-28 UTC+8 (v1.61)

**對話主題：校正幀基礎架構重建 — 三分立路徑 + 集中 Master 目錄**

**背景**
- 原架構：bias/dark/flat 全放同一個 `share/calibration/{date}_{scope}_{cam}/` 資料夾，flat 的實際拍攝日期與資料夾名稱脫鉤，pipeline 不知情。
- 本次重建的核心：Bias/Dark/Flat 依賴條件不同（bias=相機、dark=相機+溫度、flat=相機+望遠鏡），各自有獨立來源路徑。

**新資料夾結構**
```
share/calibration/
├── bias/6D2_ISO3200/           ← raw CR2 bias 幀
├── dark/6D2_ISO3200/3.7C/      ← raw CR2 dark 幀（按溫度分子目錄）
├── dark/6D2_ISO3200/5.3C/
├── flat/R200SS_6D2/20251220/   ← raw CR2 天光平場（12/20 凌晨）
├── flat/R200SS_6D2/20260313/   ← raw FITS 天光平場（3/13 學校補拍）
└── master/                     ← 所有合成 Master 集中存放
    ├── master_bias_20251221_cr2.fits
    ├── master_dark_20251221_3.7c_cr2.fits
    ├── master_dark_20251221_5.3c_cr2.fits
    ├── master_flat_20251221_cr2.fits
    └── master_flat_20260313_fits.fits
```

**Master 命名規則**
- 格式：`master_{type}_{拍攝日期}_{溫度（dark only）}_{原始格式}.fits`
- 全小寫；日期從 EXIF 自動讀取，不由 YAML 手填（避免記錯）
- flat 日期來自 YAML `flat_date`（選擇用哪批 flat 資料夾，必須手填）

**`Calibration.py` 修改**
1. `resolve_session_paths()`：
   - 移除 `shared_cal` 單一路徑，改為 `bias_root`/`dark_root`/`flat_root` 三條獨立路徑
   - YAML 新欄位：`camera_label`（bias/dark 子目錄）、`scope_label`（flat 子目錄）、`flat_date`
   - 回傳 dict 改為 `masters_dir`（統一指向 `share/calibration/master/`）
2. `_list_image_files()`：排除 `master_*` 前綴，防止已合成 master 被當 raw 幀
3. `_has_images()`：同上，排除 `master_*`，防止 dark root 被誤判
4. `_find_flat_dirs_by_format()`：排除 `master_*`，防止舊 master 觸發假 FITS 平場
5. `_find_dark_dir()`：修正閉包變數遮蔽 bug（`for dark_root` → `for _dark_candidate`）
6. `run_calibration()`：master 儲存路徑改用 `masters_dir`，命名對應新規則

**`observation_config.yaml` 修改**
1. 兩個 session 移除 `cal_label`/`calibration_date`，改為 `camera_label`/`scope_label`/`flat_date`
2. `calib_date` 欄位**不填**，改由 EXIF 自動讀取（見 make_masters.py）
3. `calibration` section：`chunk_size` 改正為 `median_chunk_size`（修正長期失效的 key）
4. 補上 `flat_bad_pixel_threshold: 0.3` 和 `save_masters: true`（原在注解區，現在生效）
5. `dark_temp_c`/`light_temp_c`/`flat_date` 加上用途說明注解
6. ⚠️ Section 編號 9 重複（Bayer 拆色 / 比較星選取），不影響程式，待下次整理

**新工具**
- `tools/make_masters.py`：獨立 master 合成工具，不跑科學幀校正
  - `calib_date`：YAML 有 `calib_date` 欄位用 YAML，否則從 EXIF 自動讀取
  - 已存在的 master 自動 SKIP，不重複合成
  - 支援 `--date YYYYMMDD` 只跑指定 session
- `tools/g1g2_flatcheck.py`：三幀 G1/G2 比值空間分布診斷工具

**本次確認的 ISO 情況**
- bias/dark/flat（CR2）全部 ISO 3200（EXIF 驗證）
- 11/22 科學幀（FITS/NINA）：GAIN=0（NINA DSLR driver 已知 bug，不記錄 ISO）
- 12/20 科學幀（CR2）：ISO 3200 ✓
- 3/13 FITS flat：GAIN=14（NINA 索引值），使用者確認實際為 ISO 3200

**待辦（本次未完成）**
- [ ] 科學幀 superflat 方案討論中（G1/G2 空間分布顯示 flat 校正後反而更糟）
- [ ] bias/dark G1/G2 三條橫線確認（可能提示需要拍攝抖動）
- [ ] 重新校正科學幀（使用新 master）
- [ ] YAML Section 編號 9 重複修正

---

### 2026-03-16 UTC+8 (v1.35)

**對話主題：3 欄式圖表整合、Orion 多目標批次處理、視覺細節微調**

**實作項目**
1. `photometry.py` 升級至 v1.35：
   - 整合 `save_3panel_period_plot`：單張圖包含 LS Power, DFT Amplitude, Phase Fold。
   - 視覺最佳化：標題 20pt、軸標 14pt、擬合線加粗 (LW=2.5)、±1σ 誤差棒標註。
   - 格式化：LS 週期標籤改為 HH:MM 格式。
2. 批次處理邏輯：
   - 支援 `--date YYYYMMDD` 自動處理 yaml 中該場次所有目標。
   - 失敗容錯：演算失敗（如 WCS）記錄日誌並跳過，不中斷批次執行。
3. 視覺微調 (User Feedback)：
   - 經緯度座標移至右上角，改為 `darkgreen` 墨綠色。
   - G1/G2 比值圖 BJD 軸標題與刻度統一為 `steelblue`。
4. Bug 修正：
   - 修正 G1/G2 比值計算中 `c` vs `_c` 的 NameError。
   - 修正 `run_fourier_fit` 中 `sort_idx` 未定義導致的 NameError。

**執行進度**
- 20251220 場次處理中。
- V1162Ori：已完成，產出 v1.35 規格圖表。
- V1643Ori / V1373Ori / HHOri / Gaia301：佇列執行中。

---

### 2026-03-14 凌晨 UTC+8

**對話主題：AlVel 盲解驗證、管線自動化處理**

**AlVel 盲解結果**
- 用 PowerShell 以 `-r 180`（全天）對 17 張 calibrated FITS 盲解
- 17/17 全部 PLTSOLVD=True，CRVAL≈(128.0°, −47.93°) ✅
- 對應 Al Vel（船帆座），RA=08h 31m，Dec=−47°39'，與 SIMBAD 一致
- hint 更新：`ra_hint_h: 8.519801`，`dec_hint_deg: -47.666`
- AlVel obs_session 已解除註解，debayer 完成（17/17）

**AlVel 測光結果**
- 全部 17 幀 airmass=4.4–5.1（仰角 11–13°）
- 門檻 2.366（25°）全部被篩除，0/17 有效
- **根本原因：AlVel Dec=−47.7°，從塔塔加（北緯 24°）觀測最高仰角僅 ~19°，幾何限制**
- 這批資料不適合精密測光，建議視為練習資料

**V1162Ori 狀態**
- WCS 已用 hint 重新解算（187/187），CRVAL=(83.557°, −6.985°）
- ASTAP GUI 盲解驗證：CRVAL=(83.784°, −7.482°)，offset=48'（正常）
- photometry G2：比較星 40 顆，108/187 幀保留，週期 2.99h，FAP=7.35e-18

**待辦**
- AlVel 測光：若要強跑，需把 airmass 門檻暫時調到 5.5
- V1162Ori 全通道測光（目前只有 G2）
- sigma_clip 28 幀問題（比較星 40 顆可能不夠 ensemble 穩定）

---

### 2026-03-14 UTC+8

**對話主題：測光驗證、選取圓修正、ASTAP 盲解校驗**

**已實作**
1. `_catalog_direct_phot`：在 pixel 座標計算後加入選取圓篩選（影像中心為圓心，短邊/2 為半徑），邊界排除從 55 → 6 筆
2. 三組診斷 print 移除（pyc 快取問題，刪除後生效）
3. airmass 門檻：dataclass 預設值從 1.994（30°）改為 2.366（25°），高氣團跳過從 51 → 34 幀
4. 全幀 CSV：airmass 砍掉的幀現在也記錄進 CSV（`ok_flag=high_airmass`），CSV 從 136 → 187 rows
5. Rejection timeline 圖：`output/rejection_timeline_*.png`，x=elapsed time，y=airmass，顏色=篩除原因
6. zp_overview 散佈圖：加入回歸公式文字標注 + 目標星空心紅圈，已驗證成功（R²=0.974，slope=1.027）
7. `ATime = Time` 頂層別名，修正 UnboundLocalError
8. `FITSFixedWarning` 抑制：`warnings.filterwarnings("ignore", message=".*datfix.*")`

**ASTAP 盲解校驗結果**
- 用一張 V1162Ori 校正後 FITS（全幅 6264×4180）盲解
- 結果：CRVAL=(83.784°, −7.482°)，Offset=48.0'，plate scale=1.485 arcsec/px ✅
- 管線現有 WCS：CRVAL=(83.557°, −6.985°)，差約 0.5°
- **結論：V1162Ori 的 WCS 需要重新 plate solve**（用正確 hint 重跑）

**待辦（下次繼續）**
- ~~多目標共用拆色檔架構（session-centric）~~ → **進行中**（本次實作）
- 初始幀篩選：`select_comp_stars` 若第一張幀品質不佳，自動往後找（前 N 張試驗，FWHM+比較星數雙重條件）
- sigma_clip 從 7 → 28 的問題（比較星從 68 → 40，ensemble 可能不穩）
- VSX 查詢整合（視野中心查 6–9 等變星，長邊 1/2 圓內）

---

### 2026-03-13 晚 UTC+8（第二段）

**對話主題：架構討論、Huber 回歸、診斷輸出改善**

**已實作**
1. `robust_zero_point`：`_fit` 改用 Huber 回歸（sklearn HuberRegressor，epsilon=1.35），失敗時退回 polyfit；舊版 OLS 保留為註解
2. `zp_overview` 散佈圖：加入回歸公式文字標注（左上角紅字）＋目標星空心紅圈
3. `ATime` UnboundLocalError 修正：頂層加 `ATime = Time` 別名，移除函式內重複 import
4. FITSFixedWarning 海嘯：頂層加 `warnings.filterwarnings("ignore", message=".*datfix.*")`
5. `datetime` 遮蔽修正（`import datetime as _dt_local`）已於第一段完成

**架構討論結論（待實作）**
- **Airmass 門檻**：改為 25°（airmass ≈ 2.37），原 30° 太嚴
- **全幀記錄 CSV**：所有 187 幀都進 CSV，airmass 砍掉的填 `ok_flag=high_airmass`
- **Rejection timeline 圖**：x=本地時間，每幀一點，顏色=篩除原因，存至 `output/`
- **多目標共用拆色檔**：session-centric 架構，一份 split FITS 跑多顆目標（待設計）
- **VSX 查詢整合**：用視野中心座標查 VSX，篩出 6–9 等、在長邊 1/2 半徑圓內的變星（待實作）
- **解星校驗**：用 ASTAP 盲解校正後 FITS，確認 CRVAL 是否正確

**術語確認**
- 本管線做的是**校正測光（calibrated photometry）**，非差分測光
- y 軸標籤已是 `Calibrated Magnitude (mag)`（前次已改）

---

### 2026-03-13 UTC+8

**對話主題：V1162Ori 20251220 比較星選取失敗診斷 + WCS CRPIX/CD 修正**

**診斷過程（只加 print，不改邏輯）**
1. `_catalog_direct_phot` 加入三組診斷 print：
   - mag 篩選範圍（mag_min/max）
   - 星表 m_cat 分布（min/max/median/NaN 數）
   - mag 範圍外跳過統計（NaN vs 真正超出範圍）
2. 加入影像尺寸與 margin 診斷
3. 加入邊界排除候選星的 pixel 座標診斷（前 20 筆）

**診斷結果（V1162Ori R 通道）**
- mag 篩選範圍：6.0 ～ 11.0
- 星表 3932 筆：min=4.62、max=17.69、**median=15.04**（絕大多數太暗）
- mag 範圍外跳過 3817 筆（NaN=268，超範圍=3549）
- 剩餘 115 筆進入測光：邊界排除 113、飽和排除 2、通過 0

**WCS 修正：`_fix_debayer_wcs()`**
- 問題：plate_solve 對全幅 Bayer 影像解算，CRPIX 是全幅像素，拆色後影像為一半
- 修正：新增 `_fix_debayer_wcs(wcs_obj)` helper（行 628 前），套用至三處 `WCS(hdr)` 呼叫（行 967、1928、2437）
- 效果：x 從 7654 縮至 3826（正確縮 ½），通過從 0 → 2

**根本原因（待處理）**
- 邊界排除的 113 筆座標仍完全不合理（x≈3500–4000，y≈−11000）
- 原因：**V1162Ori 所有 FITS 的 CRVAL=(83.56°, −6.98°)，這是 V1643Ori 的座標**
- plate_solve 解星時 hint 帶錯，V1162Ori 的 WCS 全部指向 V1643Ori 天區
- 星表查詢位置正確（ra=86°, dec=+12°），但影像 WCS 指向 19° 外
- **待辦：V1162Ori 全部 FITS 需以正確 hint（ra=5.7374h, dec=+12.763°）重跑 plate_solve**

---

### 2026-03-12（v1.1）UTC+8

**對話主題：輸出表格整理、格式問題修正**

**修正項目**
1. `photometry.py`：加入 `sys.stdout/stderr.reconfigure(encoding='utf-8', errors='replace')`，一次解決 Windows cp950 Unicode crash（38 行受影響，含 `²`、`≥`、`−`、`⚠️` 等）
2. `photometry.py`：統一所有診斷圖 DPI 為 150（原 `zp_diag` 逐幀圖為 100，`zp_overview` 為 120）
3. `photometry.py` Sharpness Index 篩選：改為掃描 `comp_refs` 找最亮未飽和比較星計算 S（原用目標星）
4. 輸出項目彙整（共 12 項）：CSV 3 種 × 通道、PNG 9 種、LOG 1 種

**確認無問題**
- 剔除統計表格（L2702–2714）已包含 sharpness 行，共六項
- Log 路徑使用 `photometry_{date}_{ts}.log` 存於 out_dir，yaml 的 `pipeline_{date}.log` 目前未使用（無衝突）
- `batch_detect_stars` 獨立函式，不在主測光流程中呼叫

---

### 2026-03-12 20:30 UTC+8

**對話主題：YAML 觀測站確認、區段號修正**

**修正項目**
1. `observation_config.yaml` 區段號編號：1–14 全部更正（前次修訂時區段號錯位）
2. 觀測站信息補入：
   - **Session 20251122**：Tataka Shandonpu Parking（塔塔加上東埔停車場）
     - Lat: 24.076856°N, Lon: 121.171665°E, Elev: 2106 m
     - 目標：AlVel、CCAnd、SXPhe
   - **Session 20251220**：Cingjing Observatory（清境觀星園）
     - Lat: 23.481197°N, Lon: 120.885415°E, Elev: 2610 m
     - 目標：V1162Ori
3. 加入 `site` 欄位至 `obs_sessions`，便於日誌識別

---

### 2026-03-12 20:00 UTC+8

**對話主題：photometry.py v0.99 實測、CCAnd 全通道測光**

**實測結果與觀察**
1. **AlVel / 20251122**：17 幀，0 個有效測光點；比較星 4 個
2. **CCAnd / 20251122**：81 幀，全通道 0 個有效點
   - 根本原因：33 幀被 airmass > 1.99 過濾（觀測高度太低，質量不佳）
   - 次要原因：48 幀因 flux/WCS/光度異常被過濾
   - 比較星：9 個（AAVSO 75 + APASS 3863 + Tycho-2 164）
3. **SXPhe / 20251122**：跳過（無 split/R FITS 檔案）
4. **V1162Ori / 20251220**：187 幀，全通道有效點佳
   - R：130/136（有效）；LS 週期 0.289741 d (6.95 h)，FAP=9.97e-08
   - G1：134/136（有效）；LS 週期 0.202962 d (4.87 h)，FAP=7.68e-14 ✓
   - G2：133/136（有效）；LS 週期 0.067728 d (1.63 h)，FAP=1.68e-06
   - B：132/136（有效）；LS 週期 0.067728 d (1.63 h)，FAP=0.523（不顯著）

**結論**
- v0.99 版本測光流程穩定；CCAnd/AlVel 觀測條件差（airmass 過高）導致無有效數據
- V1162Ori 週期檢測成功，多通道結果不一致（可能為光度/色彩變化或系統效應）
- 全通道測光（含 G2）已驗證，無額外銳利化處理

---

### 2026-03-12 UTC+8

**對話主題：Gaia DR3 整合、星表儲存、相位折疊圖存檔修正**

**修正問題清單**
1. `run_fourier_fit`：相位折疊圖有畫但未存檔（`plt.close()` 前缺少 `savefig`）→ 新增 `out_png` 參數，輸出 `phase_fold_{ch}_{date}.png`
2. 新增 `fetch_gaia_dr3_cone`：支援 G1/G2（G→V）、R（G→Rc）、B（G→B）三通道轉換（Riello et al. 2021）
3. B 通道：BP-RP ≥ 2 時標記 `warn_color` 並印出警告
4. `fetch_tycho2_cone`：保留 BT/VT 欄位；亮星下限收緊至 7.0 等（Tycho-2 V<7 不可靠）
5. `auto_select_comps`：加入 GAIA 分支（支援 yaml 寫 `"GAIA"` / `"GAIADR3"` / `"GAIA_DR3"`）
6. 星表儲存：`output/catalogs/catalog_{source}.csv`，一個星表一個 CSV
7. R 通道強制附加 Gaia RP→Rc，儲存 `catalog_GaiaDR3_Rc.csv`
8. 飽和篩除機制：peak_flux > 11469 DN 的候選比較星強制排除

**⚠️ 尚未實測，photometry.py 所有通道（AlVel / CCAnd / SXPhe / V1162Ori）需重新跑**

---

### 2026-03-11 17:00 UTC+8

**對話主題：比較星查詢修正（APASS VizieR）、版號定為 v0.99**

**修正問題清單**
1. `photometry_vizier.py` 建立：APASS 改用 `astroquery.vizier` 查詢（`dc.g-vo.org` endpoint 已 404）
2. 全部模組版號統一更新為 v0.99（原各模組版號 v1.0 / v1.1 不一致）
3. `DESIGN_DECISIONS_v6.md` 版本升為 v6.2

---

### 2026-03-11 UTC+8

**對話主題：photometry.py 首次實測通過**

**修正問題清單**
1. AAVSO API：endpoint 確認為 `/chart/`（非 `/photometry/`）；星名需空格，yaml 加 `display_name`
2. APASS：`dc.g-vo.org` 404，改用 `astroquery.vizier` 查 APASS DR9；同時取 V/B/Rc 欄位
3. APASS r 欄位名稱：VizieR 實際名稱為 `r'mag`（含單引號），非 `r_mag`
4. `comp_mag_min/max` yaml 覆蓋機制：避免 vmag_approx 估算錯誤，直接固定 6–13 mag
5. `NameError: fits_path`：迴圈變數應為 `f`，第 2270 行修正
6. Windows cp950 Unicode 錯誤：移除 print/raise 中的 σ、−、ε、⁻、≥、⚠️ 等符號
7. `observatory:` 區塊：yaml 正確位置確認，舊加的 `timing:` 下方座標無效已移除
8. `period_analysis.py`：自動選 `m_var_norm`（ensemble 啟用時）
9. 預白化 CSV 實作完成（`save_csv: true` 啟用）

**實測結果**
- V1162Ori / 20251220：R=113/187、G1=110/187、G2=123/187、B=91/187
- APASS 43 顆比較星；LS 初步週期 R 通道 **1.17 h**
- 觀測站座標待填入實際地點

---

### 2026-03-10 17:20 UTC+8

**對話主題：plate_solve.py hint 機制診斷與修正**

**問題診斷過程**
- 初始症狀：V1162Ori 190 幀全部失敗（0/190）
- 根本原因一：`observation_config.yaml` 的 `fov_override_deg: 3.16`（應為 `0`），ASTAP 強制錯誤 FOV 導致 quad 比對失敗
- 根本原因二：`plate_solve.py` 的 `-ra` hint 傳入**度**（`ra_hint_deg`），但 ASTAP `-ra` 參數單位為**小時**；導致搜尋中心偏移至錯誤天區
- 根本原因三：NINA 標頭 `RA/DEC` 為北極點座標（非目標），不傳 hint 時 ASTAP 從北極點出發，V1162 Ori（Dec≈−7°）距北極點 97°，遠超搜尋半徑 5°

**`observation_config.yaml` 修正**
- `fov_override_deg: 3.16` → `0`

**`plate_solve.py` 修正**
- `_run_astap()` 參數 `ra_hint_deg` 改名為 `ra_hint_h`（單位小時，直接對應 ASTAP `-ra`）
- 移除 `effective_radius = min(search_radius, 5.0)` 縮減邏輯（縮減不影響解算但增加混淆）
- `_get_hint_for_target()` 回傳 `(ra_hours, spd_deg)`，單位確認正確

**實測結果**
- V1162Ori / 20251220：187/190（98.4%）✅
- AlVel / 20251122：17/17（已跳過，先前已解算）✅
- CCAnd / 20251122：79/86（已跳過 79 幀）✅
- SXPhe / 20251122：0/4 ✗（1s 曝光，星點不足，已知限制）
- 總耗時：5m 48s

---

### 2026-03-10 00:35 UTC+8

**對話主題：Calibration.py bug 修正 v2、實測通過**

**`Calibration.py` 修正三項**
1. `resolve_session_paths()`：`session.get("targets", ...)` 優先於迴圈注入的 `session["target"]`，導致所有 target 路徑指向列表第一個（AlVel）。修正為優先讀單數 `"target"` key。
2. `resolve_session_paths()`：`light_dir` 誤加 `/ "light"` 子層。移除。
3. `build_header()`：FITS comment 含 em dash（U+2014）違反 FITS ASCII 限制。改為 ASCII hyphen。

**實測結果**
- AlVel / 20251122：17/17 ✅
- CCAnd / 20251122：86/86 ✅
- SXPhe / 20251122：4/4 ✅
- V1162Ori / 20251220：190/190 ✅（暗場自動選 3.7C，觀測溫度 4.0°C）
- 總耗時：4m 12s

---

### 2026-03-09 20:30 UTC+8

**對話主題：資料夾結構修正、ASTAP 星表副檔名修正**

- `targets/` 手動移入 `data/` 下
- `Calibration.py` `light_dir` 移除 `/light`
- `00_setup.ipynb` Cell 6 加入 `.1476` 星表副檔名

---

### 2026-03-09 19:30 UTC+8

**對話主題：暗場溫度子目錄、targets 迴圈修正、相對路徑支援**

- `Calibration.py`：`_find_dark_dir()`、targets 迴圈、相對路徑、FITS 溫度標頭
- `photometry.py` / `period_analysis.py`：相對路徑支援
- `observation_config.yaml`：`dark_temp_c` / `light_temp_c`

---

### 2026-03-09 16:45 UTC+8

**對話主題：`run_pipeline.py` 與 `00_setup.ipynb` 修訂、狀態檔整理**

- `run_pipeline.py`：加入 `period_analysis` 選用步驟
- `00_setup.ipynb`：Cell 4/5/8 修訂
- `PIPELINE_STATUS.md`：結構重組，修訂歷程建立

---

### （歷程起點，本次對話之前的記錄未保存）
