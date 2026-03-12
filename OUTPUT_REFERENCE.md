# 輸出項目一覽

`photometry.py` 每次執行後，在 `data/targets/{TARGET}/output/` 下產生以下檔案。

- `{ch}` = R / G1 / G2 / B
- `{date}` = YYYYMMDD
- `{ts}` = 執行時間戳

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
| 12 | LOG | `photometry_{date}_{ts}.log` | `output/` | 完整執行日誌（FileHandler encoding=utf-8） |

## 備註

- `pipeline/logs/pipeline_{date}.log`（yaml `log_file` 欄位）目前未被程式使用，實際 log 存於 `output/`。
- 第 3–4 項（zp_diag）受 yaml `save_zeropoint_diagnostic: true` 控制。
- 第 9–11 項僅在同一次執行同時包含 G1 和 G2 通道時才產生。
- 所有 PNG DPI 統一為 150。
