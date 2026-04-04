# Session Handoff — 2026-04-03

## 本次完成

### 1. v1.5 vs v1.64 raw 比對（CCAnd）
- G1/G2 median 在 VSX 範圍內（9.19–9.46），兩版本差 ~0.09 mag
- ZP R²：G1/G2 兩版本一致（~0.92），R/B 較差
- v1.5 保留幀較少（airmass 門檻有效），v1.64 raw 較多（airmass=99）
- **週期分析**：
  - v1.5 G1: P=0.1424d (R²=0.811)，v1.64 raw G1: P=0.1154d
  - VSX 公認 P=0.1249d 落在兩者之間
  - 兩版本都沒精確命中，原因是時間基線僅 ~3h（不到 1.2 個週期）
  - R/B 通道週期不可靠（v1.5 R: P=2d 完全錯誤）

### 2. CR2 混合格式修復（plate_solve.py）
- **根因**：`--raw` 模式跳過 Calibration，而 plate_solve 只 glob `*.fits`，143 張 CR2 被完全忽略
- **修復**：plate_solve.py 新增 CR2 逐檔暫存處理
  - 偵測 raw/ 裡的 .CR2 檔案
  - 逐檔：read_raw_image() → 暫存 FITS → ASTAP 解算 → 寫 WCS → 刪暫存
  - 磁碟僅需 ~105MB 暫存（不需 14GB 同時存所有轉換結果）
- **結果**：Ori 全部 5 個目標 × 188 幀 debayer 成功
- **尚未 commit**

### 3. V1162Ori 188 幀 photometry 結果
- 幀數從 46 → 188（CR2 修復生效）
- ok 幀：R=110, G1=122, G2=123, B=91
- **ZP R² 依然很差**：G1=-0.011, G2=-0.013, B=-0.122, R=0.242
  - 跟之前 46 幀版本一樣，不是幀數問題
  - 嫌疑：比較星飽和（t_max_pix=16383），field 本身問題
  - 但 30s 曝光不應該飽和 V~10 等星 → 需進一步調查
- LS 週期未命中 VSX (0.0787d)：R=0.333d, G1/G2=0.200d, B=0.048d
- 時間基線 ~6h 足夠覆蓋多個週期，問題不在基線

### 4. 磁碟空間管理
- And/Vel 的 wcs_raw 搬到 G: 共用雲端硬碟（VarStarData）
- 清理試跑資料夾（送回收筒，已清空）
- vsx_full 合併進 pure_diff（CCAnd）
- 刪除 feedback_pure_diff.md 記憶（已被 regression_vs_zp 覆蓋）

---

## 下次接續（優先順序）

### P0：硬碟搬家
1. **D:\VarStar\ 整個搬到外接硬碟**
   - 只需改 `observation_config.yaml` 的 `project_root`
   - `D:\JUNK\` 也一起搬（1.9GB）
   - pipeline 無 hardcoded D: 路徑

### P1：V1162Ori ZP R² 問題
2. **診斷 V1162Ori 比較星飽和問題**
   - t_max_pix=16383 出現在目標星（但 30s 曝光不應飽和 V~10 等星）
   - 需檢查：comp star 的 t_max_pix 分佈、是否有非飽和 comp 被排除
   - 可能原因：CR2 黑電平扣除問題、field catalog 匹配錯誤、改機 IR 通量偏高
3. **Commit plate_solve.py CR2 修復**

### P2：跑剩餘目標
4. **用新 code 跑其餘 Ori 目標的 photometry**
   - V1643Ori, V1373Ori, HHOri, Gaia301（debayer 已完成，可直接跑）
5. **Gaia375 用新 code 重跑**（11/22 session）

### P3：其他
6. VSX 額外目標 wcs_dir 路徑 bug（photometry.py ~4325 行）
7. photometry.py 拆分（phot_plots / phot_catalog / phot_ensemble）
8. CC And 週期確認（需多夜觀測或更長基線）
9. Ensemble 實測驗證

---

## 關鍵路徑索引
- Pipeline repo: `D:/VarStar/pipeline/`（唯一 git repo，main branch）
- Raw data: `D:/VarStar/data/{date}/{group}/raw/`（FITS + CR2 混合）
- Raw splits: `D:/VarStar/data/{date}/{group}/splits_raw/{R,G1,G2,B}/`
- WCS (Ori): `D:/VarStar/data/2025-12-20/Ori/wcs_raw/`（188 檔，17GB）
- WCS (And/Vel): **已搬到 G:\共用雲端硬碟\VarStarData\**
- Output: `D:/VarStar/output/{date}/{group}/{target}/`
- v1.5 比對基準: `D:/JUNK/{CC AND, 1162Ori, ALVAL}/`
- YAML config: `D:/VarStar/pipeline/observation_config.yaml`

## 磁碟空間狀態
- D: 157GB 總量，151GB 已用，**6.2GB 剩餘（97%）**
- data/ = 148GB（佔絕大部分）
- G: 共用雲端硬碟剩餘 ~89GB
- **外接硬碟已購買，尚未到手**

## 注意事項
- D:/JUNK/ 是使用者備份，**禁止刪除**
- 自由斜率回歸校正（非純差分），標籤叫「校正擬合線」
- 校正幀暫停使用（flat 不合格），用 `--raw` 模式
- `ensemble_normalize: false`，用 m_var
- `alt_min_airmass: 99`，airmass 截斷已關閉
- plate_solve.py 有未 commit 的 CR2 修復（重要！）
