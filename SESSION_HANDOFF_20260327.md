# Session 交接備忘錄 — 2026-03-27

## 本 session 完成的事

- `photometry.py` 第 2538 行：畫圖改回用 `m_var`（v1.5 邏輯），不用 `m_var_norm`
- `observation_config.yaml`：消光係數暫停（設為 0，原值保留在註解）
- `photometry.py`：新增 `--no-vsx` 參數，跳過 VSX 額外目標，只跑主目標
- v1.6D commit 已 push 上 GitHub（StandardMeanOceanWater/VarStar）
- git repo 確認在 `D:/VarStar/pipeline/`，不是 `D:/VarStar/`

---

## 未完成的事（依優先順序）

### 1. 驗證光變曲線（最優先）
修完 m_var 和消光之後，還沒實際跑圖確認 v1.6 輸出和 v1.5 一致。
```bash
cd D:/VarStar/pipeline
python photometry.py --target CCAnd --date 20251122 --channels G1 G2 --no-vsx
```
跑完後把 `3_light_curve/` 的圖和 v1.5 市賽版比對。

### 2. 星表載入慢
每次跑都重載全天星表（APASS 75萬列、Tycho-2 239萬列、Gaia 92萬列）。
改法：plate solve 完後先裁出視場小星表，存在 output 的 catalogs/ 裡，
下次跑直接讀，不用重載全天。

### 3. 消光改正的學理問題
回歸線是否已吸收大部分消光？需要：
- 跑有消光（原值）vs 無消光（現在）的圖做比較
- 看晚觀測幀（高 airmass）有沒有系統性偏移

### 4. Superflat 未整合進 pipeline
FINDINGS_20260322.md 記錄 superflat 效果最好（G1/G2 std 改善 66%），
但 code 裡還沒有實作，`superflat_20251122.fits` 存在 output 根目錄。

### 5. ZP 診斷圖標籤
圖名叫「ZP scatter」但實際是自由斜率回歸，應改叫「校正擬合線」。
y 軸標籤「Ensemble-Normalized Magnitude」也需要確認是否已改回「Calibrated Magnitude」。

### 6. Output 資料夾清理
今天測試跑了多次，`D:/VarStar/output/2025-11-22/And/CCAnd/` 下有很多時間戳資料夾。
保留最後一次（20260327_193610），其餘可刪除。

---

## 重要學理備忘

- 使用者方法是**自由斜率回歸校正**，不是純差分（slope=1）
  - 學理依據：Paxson 2010 JAAVSO、比較星顏色多樣時 slope=1 假設不成立
  - 目標星在比較星範圍內 → 插值，不是外插
- 圖的名稱：「校正擬合線」而不是「ZP」
- Ensemble normalization 是壞的，一律用 m_var 不用 m_var_norm
- git repo 在 `D:/VarStar/pipeline/`

## 路徑備忘
- 星表：`D:/VarStar/data/share/catalogs/allsky/`
- v1.5 市賽版輸出：`G:/共用雲端硬碟/VarStar/IMPORTANT/市賽版/CC AND/`
- v1.6 最新輸出：`D:/VarStar/output/2025-11-22/And/CCAnd/20260327_193610/`
- Superflat：`D:/VarStar/output/superflat_20251122.fits`
- FINDINGS 文件：`D:/VarStar/pipeline/FINDINGS_20260322.md`
