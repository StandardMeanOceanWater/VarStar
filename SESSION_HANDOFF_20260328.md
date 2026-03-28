# Session 交接備忘錄 — 2026-03-28

## 本 session 完成的事（v1.62）

### 校正完整流程跑通（11/22 + 12/20 兩個 session）
- `make_masters.py`：Master Bias / Dark / Flat 全部生成
- `Calibration.py`：所有目標的 light frame 校正完成（bias + dark + flat）
- `plate_solve.py`：ASTAP 本地解星，WCS 寫入 `*_wcs.fits`
- `DeBayer_RGGB.py`：Bayer 拆色完成，輸出到 `splits/{R,G1,G2,B}/`

### 各目標幀數
| Session | 目標 | 幀數 |
|---------|------|------|
| 11/22 | AlVel | 17 |
| 11/22 | CCAnd | 169 |
| 11/22 | Gaia375 | 169 |
| 12/20 | V1162Ori | 187 |
| 12/20 | V1643Ori | 187 |
| 12/20 | V1373Ori | 187 |
| 12/20 | HHOri | 187 |
| 12/20 | Gaia301 | 187 |

### Bug 修正
- `Calibration.py` L1092：`parent.parent` → `parent`（config 路徑錯誤）
- `plate_solve.py` L597：同上
- `DeBayer_RGGB.py` L365：同上
- `observation_config.yaml`：`flat_date` 12/20 session 改為 `"20251221"`（flat 資料夾已更名）

---

## 下一步（最優先）

### 1. 跑測光 Photometry
```bash
cd D:/VarStar/pipeline
python photometry.py --target CCAnd --date 20251122 --channels G1 G2 --no-vsx
```
splits 都已備好，可以直接接測光。

### 2. 驗證光變曲線
跑完後把 `3_light_curve/` 的圖和 v1.5 市賽版比對（路徑見下方）。

---

## 繼續未完成的事

### 3. 星表載入慢
每次跑都重載全天星表（APASS 75萬列、Tycho-2 239萬列、Gaia 92萬列）。
改法：plate solve 完後先裁出視場小星表，存在 output 的 catalogs/ 裡。

### 4. G1/G2 殘差空間結構（microlens 角度響應）
重要：這不是 superflat 問題，是 **PDAF 微透鏡角度依賴色響應**。
- 正確做法：先觀察已校正科學幀的 G1/G2 殘差空間結構，再設計演算法
- 不能用 flat 的 G1/G2 比例圖去校正（flat 光場條件不同）
- 現有 superflat：`D:/VarStar/output/superflat_20251122.fits`（效果有限）

### 5. 消光改正學理問題
- 現在消光係數設為 0（保留在 YAML 註解中）
- 需驗證：有消光 vs 無消光的圖，觀察高 airmass 幀的系統性偏移

### 6. 12/20 混合格式問題
- 47 FITS + 143 CR2 per target，flat 是 CR2
- Pipeline 警告但繼續執行，功能上 OK
- 待後續評估是否需要統一格式

### 7. Calibration.py 重複 master 生成
- Calibration.py 跑時會再用 session 日期（20251122）建一組 master
- make_masters.py 已用 EXIF 日期（20251221）建了一組
- 兩組並存，不影響功能，但待清理

### 8. ZP 診斷圖標籤
- 圖名叫「ZP scatter」應改叫「校正擬合線」
- y 軸確認是否顯示「Calibrated Magnitude」

---

## 重要學理備忘

- 使用者方法是**自由斜率回歸校正**，不是純差分（slope=1）
  - 學理依據：Paxson 2010 JAAVSO
  - 目標星在比較星範圍內 → 插值，不是外插
- **Ensemble normalization 壞的，一律用 `m_var` 不用 `m_var_norm`**
- G1/G2 通道 → APASS V；R → Gaia Rc（非 APASS）；B → APASS B
- git repo 在 `D:/VarStar/pipeline/`（不是 `D:/VarStar/`）

---

## 路徑備忘
- 星表：`D:/VarStar/data/share/catalogs/allsky/`
- 校正幀輸出：`D:/VarStar/data/{date}/{group}/calibrated/`
- WCS 輸出：`D:/VarStar/data/{date}/{group}/wcs/`
- Splits 輸出：`D:/VarStar/data/{date}/{group}/splits/{R,G1,G2,B}/`
- Masters：`D:/VarStar/data/share/calibration/masters/`
- v1.5 市賽版輸出：`G:/共用雲端硬碟/VarStar/IMPORTANT/市賽版/CC AND/`
- v1.6 最新輸出：`D:/VarStar/output/2025-11-22/And/CCAnd/20260327_193610/`
- Superflat：`D:/VarStar/output/superflat_20251122.fits`
- FINDINGS 文件：`D:/VarStar/pipeline/FINDINGS_20260322.md`
