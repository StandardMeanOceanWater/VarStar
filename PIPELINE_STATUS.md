# 變星測光管線 — 狀態快照
**最後更新：2026-03-10 02:00 UTC+8 | 本檔永遠只有一份，直接覆蓋更新**

---

## 1. 模組狀態

| 模組 | 檔名 | 狀態 | 備註 |
|------|------|------|------|
| 校正 | `Calibration.py` | ✅ 完成（已實測） | Bug 修正 v2；297 幀全數通過 |
| 星圖解算 | `plate_solve.py` | ✅ 完成 | ASTAP / astrometry.net 雙後端 |
| Bayer 拆色 | `DeBayer_RGGB.py` | ✅ 完成 | WCS 子採樣修正已實作 |
| 測光（含標準 LS） | `photometry.py` + `Photometry.ipynb` | ✅ 完成 | 相對路徑支援已加入 |
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
- 像素尺寸：6.56 μm（PTC 實測）
- Plate scale：1.692 arcsec/px
- 飽和閾值：11469 DN
- 相機：Canon EOS 6D Mark II，已改機移除 IR cut filter，Bayer pattern RGGB

### 校正流程
- 黑電平：逐通道讀取，負值保留
- Master 合成：Remedian 二階近似中位數
- Dark 縮放：依曝光時間比例縮放
- 暗場溫度：`dark_temp_c` 選填；子目錄選取以 `light_temp_c` 為基準選最接近者；無 `light_temp_c` 時選最低溫

### 測光
- 時間系統：BJD_TDB，曝光中點，de432s
- 比較星星表順序：AAVSO → APASS
- 孔徑：生長曲線法，固定孔徑
- 背景：背景環中位數
- 測光誤差：Merline & Howell (1995)
- Airmass：Young (1994)
- 高度角截斷：altitude < 45°（airmass > 1.413）的幀跳過，不寫入 CSV；location 未設定（airmass=NaN）時不截斷
- APASS 波段對應：R→r'、G1/G2→V、B→B（決定零點回歸用哪個星等欄位）

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
- [ ] 診斷 EXIF ISO 讀出為 0（FITS GAIN/RDNOISE 空白），在 photometry 前處理
- [ ] 預白化 CSV 輸出（yaml `save_csv: true` 時啟用，code 已預留）
- [ ] Gaia DR3 比較星介面（目前輸出警告，未實作查詢）
- [ ] `quality_report.py` 實作（第二批 D 組）
- [x] DeBayer_RGGB.py targets 列表支援（已實測通過，2026-03-10）
- [x] DeBayer_RGGB.py em dash FITS header bug 修正（已實測通過，2026-03-10）

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

---

## 5. 修訂歷程

---

### 2026-03-10 02:00 UTC+8

**對話主題：photometry.py 高度角截斷、WARN 顯示修正**

**`photometry.py` 修正兩項**
1. 高度角截斷：altitude < 45°（airmass > 1.413, Young 1994）の幀輸出 `[SKIP]` 並跳過，不寫入 CSV。airmass=NaN（location 未設定）時不截斷。截斷閾值以 `Cfg.alt_min_deg` / `Cfg.alt_min_airmass` 儲存。
2. WARN 顯示 bug 修正：`time_from_header()` 高 airmass 警告中，`alt_deg` 誤套 `90 - .alt.deg`（把高度角算成天頂距），改為直接取 `.alt.deg`。airmass 數值本身未受影響，僅訊息標籤錯誤。

**不可翻案新增**
- 高度角截斷 alt < 45° 為不可翻案（B 通道消光每 airmass ≈ 0.2 mag）
- APASS 波段對應 R→r'、G1/G2→V、B→B 為不可翻案

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
