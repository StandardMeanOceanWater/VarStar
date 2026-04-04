# CLAUDE.md — 變星測光管線專案指引
**版號：v1.1 | 最後更新：2026-03-14 | 修訂規則見 PIPELINE_STATUS.md 開頭**

## 專案目的
Canon EOS 6D Mark II 改機（移除 IR cut filter）+ 800mm 望遠鏡的變星差分測光自動管線。
處理流程：CR2 校正 → Plate Solve → Bayer 拆色 → 孔徑測光 → 週期分析。

---

## 語言規則
- 與使用者溝通：**繁體中文（zh-TW）**
- 程式碼、識別字、CLI 指令：保持英文原文
- 文件、註解（若需新增）：繁體中文

---

## 目錄結構

```
D:\VarStar\
├── pipeline\                        ← 工作目錄（本倉庫）
│   ├── CLAUDE.md
│   ├── observation_config.yaml
│   ├── pipeline_config.py
│   ├── photometry.py
│   ├── period_analysis.py
│   └── ...
├── data\
│   ├── share\
│   │   ├── calibration\{date}_{telescope}_{camera}\  {bias/, dark/, flat/, masters/}
│   │   └── catalogs\
│   └── {YYYY-MM-DD}\               ← 觀測日期
│       └── {GROUP}\                 ← 照片組（如 Ori, And, Vel）
│           ├── raw\
│           ├── wcs\
│           └── splits\  {R/, G1/, G2/, B/}
└── output\
    ├── _pipeline_log\
    └── {YYYY-MM-DD}\
        └── {GROUP}\
            └── {TARGET}\
                └── {YYYYMMDD_HHMM}\  ← 每次執行獨立
                    ├── 1_photometry\
                    ├── 2_calibration_diag\
                    ├── 3_light_curve\
                    └── 4_period_analysis\
```

---

## 儀器參數（不可更動）

| 參數 | 值 | 來源 |
|------|-----|------|
| 像素尺寸 | 5.76 μm | Canon 原廠規格 |
| Plate scale | 1.485 arcsec/px | 206.265 × 5.76 / 800 |
| 飽和閾值 | 65536 DN | 全開（R² 監控非線性） |
| Bayer pattern | RGGB | Canon 6D2 規格 |
| IR cut filter | 已移除 | R 通道帶通延伸至近紅外；APASS Rc 為近似，嚴格轉換需 Landolt 標定 |
| 時間系統 | BJD_TDB，曝光中點，de432s | FITS/AAVSO 標準 |

---

## 已鎖定設計決策（勿翻案）

詳見 `DESIGN_DECISIONS_v6.md` 第二部分。以下為最常需要查閱的項目：

- **ASTAP hint**：`-ra` 單位為**小時（0–24）**，`-spd` = 90.0 + Dec（南極距，度）
- **NINA RA/DEC 不可信**：實測為北極點座標，必須從 yaml `targets` 讀 `ra_hint_h` / `dec_hint_deg`
- **Masters 存放**：`shared_calibration/{date}/masters/`（非 target 目錄）
- **Light 幀路徑**：`raw/{date}/`，無 `light/` 子層
- **fov_override_deg**：yaml 必須填 `0`，讓 ASTAP 自動估 FOV
- **黑電平**：逐通道讀取，負值保留（不截斷）
- **Master 合成**：Remedian 二階近似中位數
- **測光誤差**：Merline & Howell (1995)，噪聲項用 `b_sky_std`（非 `b_sky`）
- **比較星選取圓**：以**影像中心**為圓心（非目標星）
- **Ensemble 正規化**：Broeg (2005) 迭代演算法，yaml `ensemble_normalize: true` 啟用

---

## 模組狀態（截至 2026-03-11）

| 模組 | 狀態 | 備註 |
|------|------|------|
| `Calibration.py` | ✅ 完成實測 | 297 幀全數通過 |
| `plate_solve.py` | ✅ 完成實測 | V1162Ori 187/190（98.4%） |
| `DeBayer_RGGB.py` | ✅ 完成 | WCS stride=2 子採樣修正 |
| `photometry.py` | ✅ 完成 | Broeg ensemble + 選取圓修正 |
| `period_analysis.py` | ✅ 完成 | BIC 諧波 + bootstrap FAP |
| `run_pipeline.py` | ✅ 完成 | period_analysis 為選用步驟 |
| `00_setup.ipynb` | ✅ 完成 | Cell 5/6/8 修正 |
| `observation_config.yaml` | ✅ 完成 | |
| `quality_report.py` | ⏳ 待實作 | 第二批 D 組 |

---

## 目前工作

**比較星查詢修正（2026-03-11）：**
- AAVSO：endpoint 已確認為 `/chart/`（非 `/photometry/`）；星名需有空格（`"V1162 Ori"`），yaml 加 `display_name`
- APASS：`dc.g-vo.org` endpoint 404，改用 `astroquery.vizier` 查詢（已驗證可取得 50+ 筆）
- Gaia DR3：列為第三層 fallback，待實作
- **APASS 本機星表**：約 1.5 GB，VizieR 可下載；離線環境備用，低優先列入考慮

---

## 已知限制（重要）

1. **EXIF ISO 讀取為 0**：rawpy 無法從 Canon 6D2 CR2 取得 ISO，FITS GAIN/RDNOISE 空白；不影響校正，測光前需補值（`Calibration.py` 已有三層 fallback）
2. **SXPhe 4 幀**：1s 曝光，ASTAP 星點不足，全部失敗，評估是否補觀測
3. **週期不確定度**：Kovacs 公式為下限
4. **Bootstrap FAP < 10⁻⁴**：需手動設 `fap_bootstrap_max_iter: 5000`
5. **暗場溫度目錄**：命名格式 `{數字}C`（如 `3.7C`），不支援負溫度

完整清單見 `DESIGN_DECISIONS_v6.md` 第五部分。

---

## 工作流程規則

1. **修改程式前先讀取**：用 `Read` 工具讀取目標檔案，理解現有邏輯再修改
2. **設計決策有疑問先查文件**：`DESIGN_DECISIONS_v6.md` 是唯一學理規格
3. **狀態更新**：實測通過後更新 `PIPELINE_STATUS.md`
4. **不過度工程**：只做被要求或明確必要的修改；不加多餘 docstring、類型標注、或重構
5. **安全邊際**：破壞性操作（刪除分支、force push）前先確認
6. **`photometry.py` 第 1 行為 Jupyter magic**：`py_compile` 無法直接語法檢查

---

## 常用指令

```bash
# 執行校正
python run_pipeline.py --step calibration

# 執行解星
python run_pipeline.py --step plate_solve

# 執行拆色
python run_pipeline.py --step debayer

# 執行完整管線（步驟 1–3）
python run_pipeline.py
```

---

## 參考文獻快查

- Merline & Howell (1995)：測光誤差公式
- Broeg et al. (2005)：Ensemble 正規化
- Breger et al. (1993)：Pre-whitening S/N < 4 停止條件
- Young (1994)：Airmass 公式
- Rousseeuw & Bassett (1990)：Remedian 中位數近似
