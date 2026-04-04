# Session Handoff — 2026-03-31

## 本次完成

### 程式碼變更（已 commit）
1. **v1.64** (229901a) — Raw 模式支援 + 5 項 bug 修復
   - `run_pipeline.py`: `--raw`（跳過校正）/ `--no-flat`（僅 bias+dark）
   - `plate_solve.py`: `raw_mode` / `src_dir` / `wcs_out_dir` 參數
   - `DeBayer_RGGB.py`: `raw_mode` / `wcs_subdir` / `splits_subdir` 參數
   - `Calibration.py`: `no_flat` 參數
   - `photometry.py`: `--raw` flag → `splits_raw/` + `out_tag=raw`
   - Bug fixes:
     - ensemble_normalize 量綱不匹配：delta 除以 zp_slope 再扣除
     - 外插警告：新增 `flag_extrapolated` 欄位 + runtime 檢查
     - `_t0` 未定義風險：check star 區段改用獨立 `_t0_check`
     - VSX 額外目標 `phot_band` 未隨通道更新：同步 backup/restore
     - 死碼移除：raise 後 unreachable print

2. **v1.64b** (e6eda30) — YAML auto-fill 防坑 + airmass 修復
   - `cfg_from_yaml` 新增 auto-fill：`phot_cfg` 中與 `Cfg` dataclass 同名 key 自動帶入
   - 修復 `alt_min_airmass` 從未被 YAML 讀取的 bug
   - YAML: `alt_min_airmass` 移至 `photometry:` 區段，設為 99（關閉）

3. **父 repo `.git` 已移除** — pipeline/ 是唯一 repo，v1.35 tag 已打在 966eb16

### 分析報告（4 個 Agent）
1. **ZP 回歸學理審查** → 數學正確，唯一缺外插 guard（已修）
2. **Ensemble 根因** → 量綱不匹配：m_var 在校正空間，delta 在儀器空間（已修）
3. **Photometry.py bug hunt** → 3 中等 + 5 輕微（中等全修了）
4. **已知問題整理** → 19 已修 / 18 待修 / 6 過時

### 管線全架構地圖
完整審查已完成，涵蓋 5 階段資料流、篩選邏輯位置、全域狀態耦合。
Agent output 存於 task a7f5292cca2612752。

### Raw 測光（v1.64b code, airmass=99）
| 目標 | Session | 狀態 | 備註 |
|------|---------|------|------|
| CCAnd | 11/22 | ✅ 全通道+週期 | 20260330_181912_raw |
| AlVel | 11/22 | ✅ 主目標 ok | 20260330_181912_raw，VSX 有路徑 bug |
| Gaia375 | 11/22 | ✅ 全通道+週期 | 20260330_180824_raw（舊 code） |
| V1162Ori | 12/20 | ✅ 主目標 ok | 20260331_* _raw，VSX 有路徑 bug |
| V1643Ori | 12/20 | ⚠️ 跑過一次（舊code） | 需用新 code 重跑 |
| V1373Ori | 12/20 | ⬜ 未跑 | |
| HHOri | 12/20 | ⬜ 未跑 | |
| Gaia301 | 12/20 | ⬜ 未跑 | |

---

## 下次接續（優先順序）

### P0：必做
1. **v1.64 vs v1.5 (D:/JUNK/) 光變曲線比對**
   - v1.5 資料在 `D:/JUNK/{CC AND, 1162Ori, ALVAL}/`
   - 比對 G1 通道：m_var median/std、ok 幀數、LS period
   - 這是已知問題 #16（HIGH），市賽版驗證

2. **跑剩餘目標**（用 v1.64b code）
   - `python photometry.py --target V1643Ori --date 20251220 --raw`
   - `python photometry.py --target V1373Ori --date 20251220 --raw`
   - `python photometry.py --target HHOri --date 20251220 --raw`
   - `python photometry.py --target Gaia301 --date 20251220 --raw`

### P1：重要
3. **VSX 額外目標 wcs_dir 路徑 bug**
   - 症狀：B 通道在 splits_raw/R 找 FITS
   - 根因：VSX 迴圈中 `cfg.wcs_dir` 未隨 `_vch` 更新
   - 位置：photometry.py ~4325 行

4. **Gaia375 用新 code 重跑**（舊 code 跑的結果可能有差異）

### P2：拆分
5. **最小拆分方案**（使用者同意）
   - `phot_plots.py`：~600 行畫圖
   - `phot_catalog.py`：~500 行星表載入/cache
   - `phot_ensemble.py`：~170 行正規化
   - 主流程留在 photometry.py

### P3：其他
6. CC And 週期確認（G_avg=2.73h vs VSX 0.1249d=2.998h）
7. Ensemble 量綱修復的實測驗證
8. 星表載入效能（同一 field 重複讀百萬行）

---

## 關鍵路徑索引
- Pipeline repo: `D:/VarStar/pipeline/`（唯一 git repo，main branch）
- Raw splits: `D:/VarStar/data/{date}/{group}/splits_raw/{R,G1,G2,B}/`
- Output: `D:/VarStar/output/{date}/{group}/{target}/{timestamp}_raw/`
- v1.5 比對基準: `D:/JUNK/{CC AND, 1162Ori, ALVAL}/`
- YAML config: `D:/VarStar/pipeline/observation_config.yaml`

## 注意事項
- D:/JUNK/ 是使用者備份，**禁止刪除**
- 使用者用的是自由斜率回歸（非 slope=1 純差分），標籤叫「校正擬合線」
- 校正幀（bias/dark/flat）全部暫停使用，直接用 raw
- `ensemble_normalize: false`（YAML），目前用 m_var 不用 m_var_norm
- `alt_min_airmass: 99`（YAML），airmass 截斷已關閉
