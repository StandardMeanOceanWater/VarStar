# 變星測光管線 — 設計決策文件

**版本：v6.0　|　日期：2026-03-10　|　狀態：學理完備**

本文件為整個管線的唯一學理規格。所有程式實作均以此為準。「決策理由」欄位記錄設計原因，防止日後誤改。

---

## 第一部分：資料夾結構

### 1.1 本地結構（D:\VarStar\）

```
D:\VarStar\
├── pipeline\
│   ├── observation_config.yaml     ← 唯一設定入口
│   ├── Calibration.py
│   ├── plate_solve.py
│   ├── DeBayer_RGGB.py
│   ├── photometry.py               ← 步驟 4-5（含標準 LS）
│   ├── period_analysis.py          ← 使用者選用進階分析模組
│   ├── run_pipeline.py             ← 步驟 1-3 整合入口
│   ├── Photometry.ipynb
│   └── 00_setup.ipynb
└── data\
    ├── shared_calibration\
    │   ├── 20251122\  {bias/, dark/, flat/, masters/}
    │   └── 20251220\  {bias/, dark/{3.7C/, 5.3C/}, flat/, masters/}
    └── targets\
        └── {TARGET}\
            ├── raw\{date}\         ← Light 幀直接放這裡，無 light/ 子層
            ├── calibrated\         {wcs/}
            ├── split\              {R/, G1/, G2/, B/}
            └── output\
                ├── photometry_{ch}.csv
                ├── light_curve.png
                ├── periodogram.png
                └── period_analysis\
```

### 1.2 Masters 存放位置（v6 修正）

Master 幀（Master_Bias、Master_Dark、Master_Flat_norm）統一存放於：

```
shared_calibration/{date}/masters/
```

v5 誤存在第一個 target 目錄下，導致多 target session 無法共用 masters。

### 1.3 雙環境路徑對應

步驟 1–3 只在本機執行（原始 CR2 資料量大、ASTAP 只能本機安裝）。步驟 4–5 本機或 Colab 皆可。

唯一分叉點：`load_config()` 偵測環境一次，決定 `project_root`，yaml 同時記錄兩個路徑。程式本體無環境分叉。

---

## 第二部分：不可翻案的核心決策

以下決策已鎖定，不再重複討論。若需變更，須另立正式決策審查。

| 項目 | 決策 | 依據 |
|------|------|------|
| 像素尺寸 | 6.56 μm（photonstophotos.net PTC 實測） | — |
| Plate scale | 1.692 arcsec/px（程式自動計算） | 206.265 × 6.56 / 800 |
| 飽和閾值 | 11469 DN（14-bit 滿井 70%） | 線性響應安全上限 |
| 時間系統 | BJD_TDB，曝光中點，de432s 星曆 | FITS/AAVSO 標準 |
| 黑電平 | 逐通道讀取，校正前減去，負值保留 | 噪聲對稱性 |
| Master 合成 | Remedian 二階近似中位數（Rousseeuw & Bassett, 1990） | RAM 效率 |
| Flat 歸一化 | sigma clipping 後取中位數，壞像素 clip [0.3, 5.0] | — |
| Masters 存放位置 | `shared_calibration/{date}/masters/`（非 target 目錄） | 多 target 共用同一組 masters |
| 比較星星表優先順序 | AAVSO（≥5顆）→ APASS → 警告 | — |
| 孔徑選擇 | 生長曲線法，固定孔徑貫穿整夜（Howell, 2006） | 比較星+目標共用同一孔徑 |
| 背景估計主要量 | 背景環中位數（魯棒性優於均值） | 抗離群值 |
| 測光誤差 | Merline & Howell (1995) 完整公式（v5 修正） | 見 §3.6 |
| Airmass | Young (1994) 低仰角修正公式 | — |
| 管線標準週期輸出 | photometry.py 內建 LS，不受 period_analysis.py 控制 | — |
| Pre-whitening 停止條件 | 殘差 S/N < 4.0（Breger et al., 1993） | 脈動變星社群標準 |
| 相位零點 | 亮度極大值 = φ = 0，兩步迭代確認 | δ Scuti 慣例 |

---

## 第三部分：模組設計規格

### 3.1 校正流程（Calibration.py）

#### 校正公式（CAL_MODE = FULL）

```
dark_rate = (Master_Dark_raw − Master_Bias) / t_dark
Light_signal = Light_raw − Master_Bias − (dark_rate × t_light)
Flat_signal = Master_Flat_raw − Master_Bias − (dark_rate × t_flat)
Master_Flat_norm = Flat_signal / median(Flat_signal)
Calibrated = Light_signal / Master_Flat_norm
```

負值不截斷：Bias 減法後背景像素分佈以 0 為中心，負值是物理上合理的噪聲採樣。截斷會造成截尾半高斯分佈，使低訊號恆星的測光不確定度被系統性低估。

Master 合成採用 Remedian 二階近似中位數（Rousseeuw & Bassett, 1990）。RAM 用量從 O(N×pixels) 降至 O(chunk_size×pixels)，偏差遠小於逐幀噪聲。

#### ISO Fallback 機制（v6 新增）

`sensor_db` 查詢所需的 ISO 值，優先順序：

1. yaml `iso` 欄位填寫且 > 0 → 直接使用
2. yaml 未填或填 0 → 從第一張 light 幀讀取：
   - `.cr2` → `exifread` 讀 `EXIF ISOSpeedRatings`
   - `.fits` → `astropy` 讀 `ISOSPEED` 標頭
3. 兩者都失敗 → `iso = 0`，`[WARN]` 顯示實際搜尋路徑

### 3.2 FITS 標頭規範

| 鍵值 | 內容 |
|------|------|
| DATE-OBS | UTC 曝光開始（EXIF − UTC+8） |
| MID-OBS | UTC 曝光中點（DATE-OBS + EXPTIME/2） |
| EXPTIME | 曝光秒數 |
| ISOSPEED | 相機 ISO 值（優先從 CR2 EXIF 讀取；無 EXIF 時為 0） |
| GAIN | e⁻/DN（ISO 查 sensor_db） |
| RDNOISE | e⁻ |
| SATURATE | 11469.0 DN |
| CAL_MODE | FULL / NO_DARK / NO_BIAS / FLAT_ONLY |
| BLACKLVL | Canon 黑電平 DN（逐通道均值） |
| DEBAYER | NO（保留 Bayer 排列，禁止插值） |
| BAYERPAT | RGGB |

### 3.3 Plate Solve（plate_solve.py）

後端自動選擇：Colab → astrometry.net；本機 → ASTAP。輸出另存副本至 `calibrated/wcs/{stem}_wcs.fits`，不覆蓋原始校正幀。astrometry.net 上傳前降採樣 4×；WCS 解算後換算回原始解析度。

#### ASTAP CLI 正確參數（v6 修正）

ASTAP CLI 的 `-s` 參數為取樣數（整數），**不是**星表類型。程式碼中移除 `-s db_type` 的錯誤用法。星表路徑由 `-d db_path` 指定。

```
astap_cli -f input.fits -r 30 -d /path/to/d80 -z 2 -update
```

#### 座標 Hint 機制（v6 新增）

當 FITS 標頭 RA/DEC 不可靠時（例如 NINA 赤道儀連線異常），可在 yaml `targets` 區段填入起始座標：

```yaml
targets:
  V1162Ori:
    ra_hint_h: 5.507      # 單位：小時
    dec_hint_deg: 5.11    # 單位：度
```

程式偵測到 hint 時，傳入 `-ra {ra_hint_h} -spd {spd}` 給 ASTAP，其中 `spd = 90.0 - dec_hint_deg`（南極距）。無 hint 則不傳，使用 FITS 標頭值。

#### WCS 傳遞至拆色通道（stride=2 子採樣）

```
CRPIX_new = (CRPIX_orig − offset − 0.5) / 2.0 + 0.5
CD{i}_{j}_new = CD{i}_{j}_orig × 2.0
```

學理依據：仿射變換縮放規則。子採樣放大座標系兩倍，參考像素需重新置中，CD matrix 各元素（每像素角距離變化率）等比例放大。

### 3.4 Bayer 拆色（DeBayer_RGGB.py）

RGGB 排列：R(0,0)、G1(0,1)、G2(1,0)、B(1,1)。G2 僅用於品質驗證，不輸出獨立光變曲線。

### 3.5 比較星選取

距離加權 ensemble photometry（Honeycutt, 1992 概念）。AAVSO ≥ 5 顆時採用；否則退至 APASS；兩者不足時輸出警告。

距離加權：w_i = 1 / (d_i + ε)²，ε = plate_scale / 2，防止距離趨近零時權重爆炸。

若星表誤差已知，加乘誤差加權：w_i /= m_err²。雙重加權未正規化為已知限制（見第五部分）。

### 3.6 孔徑測光（v5 大幅修訂）

#### 3.6.1 生長曲線選孔徑（Howell, 2006）

比較星（comp_df）生長曲線中位數決定孔徑半徑，整夜固定套用至所有恆星。孔徑為比較星與目標星共用的單一值。

**已知限制：** 若目標星 PSF 比比較星更胖（如視野邊角，像差較大），此孔徑可能低估目標通量。改進方向：對目標星另做生長曲線，取兩者較大值。目前記錄為已知限制，不強制修改。

#### 3.6.2 背景環半徑（v5 改為隨孔徑縮放）

v5 改為以孔徑半徑 r 為基準縮放（Howell, 2006）：

```
r_in  = max(r × 1.5, r + 5.0)    # 至少距孔徑邊緣 0.5r，隔離 PSF 翼部
r_out = max(r × 2.5, r_in + 10.0) # 至少 10 px 寬，確保足夠背景像素
```

#### 3.6.3 背景統計量（v5 新增）

`aperture_photometry()` 回傳完整背景統計：

| 欄位 | 內容 | 用途 |
|------|------|------|
| b_sky | 背景中位數（DN/px） | flux_net 計算的主要估計量 |
| b_sky_mean | 背景均值（DN/px） | 診斷：均值與中位數差距大 → 背景有污染源 |
| b_sky_std | 背景標準差（DN/px） | 誤差方程式（必須用此值，非均值） |
| b_sky_q25 / b_sky_q75 | 第 25 / 75 百分位數 | IQR 診斷 |
| n_sky | 背景環像素數 | 誤差方程式傳遞項 |

#### 3.6.4 邊界截切判斷（v5 改為精確檢查）

v5 改為 `in_bounds_precise()`：確認星像中心 + r_out 在四個方向均在影像內（含 1.0 px 安全邊距）。背景環被截切的幀直接捨棄，符合「可靠至上」原則。

#### 3.6.5 測光誤差方程式（v5 修正——Merline & Howell, 1995）

完整公式：

```
sigma_flux² = N_pix × [S/G + b_sky_std² × (1 + N_pix/N_sky)]
            + N_pix² × sigma_flat² / N_flat

sigma_mag = (2.5 / ln10) × (sigma_flux / flux_net)
```

其中：

- `S` = 孔徑內總 DN（未減背景）
- `G` = gain（e⁻/DN）
- `b_sky_std` = 背景環標準差（DN/px）——**必須用此值，非 b_sky**
- `N_pix` = 孔徑像素數
- `N_sky` = 背景環像素數
- `sigma_flat` = Flat 幀噪聲（DN）
- `N_flat` = 合成 Flat 的幀數

v5 之前的 bug：誤用 `b_sky`（中位數）代入噪聲項，導致 sigma_mag 偏高 5–7 倍。

### 3.7 差分測光

```
dm_target = m_inst_target − m_comp_weighted
m_comp_weighted = Σ(w_i × m_inst_i) / Σ(w_i)
```

### 3.8 零點斜率 Flag（v5 新增）

對每幀計算 m_inst vs m_cat 的線性回歸斜率 a。若 |a − 1| > 0.05，標記 `FLAG_SLOPE_DEVIATION = True`，表示顏色依賴性誤差超出可接受範圍。

### 3.9 Ensemble 正規化（v5 新增，預設停用）

`ensemble_normalize()`：以比較星的整夜中位數為基準，修正大氣漂移引入的系統趨勢。完整 Broeg (2005) 演算法待 `comp_lightcurves` 擴充後實作。

### 3.10 PSF 診斷圖（v5 新增）

每幀輸出 `m_inst_psf vs m_cat` 散佈圖，用於診斷線性響應範圍與孔徑一致性。輸出至 `output/diag_psf/`。

---

## 第四部分：週期分析規格

### 4.1 模組分工（不可翻案）

| 模組 | 定位 | 輸入 | 觸發方式 |
|------|------|------|----------|
| photometry.py 內建 LS | 管線標準輸出 | 測光過程中自動計算 | 自動執行 |
| period_analysis.py | 使用者選用進階分析 | 測光 CSV | 手動 import 或單獨執行 |

`period_analysis.py` 不替換 `photometry.py` 的 LS 輸出，兩者獨立。

### 4.2 主方法：Lomb-Scargle（Lomb, 1976；Scargle, 1982；VanderPlas, 2018）

使用 `astropy.timeseries.LombScargle`，`normalization='standard'`，頻率格網 `oversampling=10`。

#### FAP 估計——Bootstrap 收斂迭代

採用 bootstrap 方法（打亂時間序列重算 LS，統計假陽性）。

- 每 100 次迭代計算一次當前 FAP 估計值
- 相鄰兩個視窗的 FAP 相對變化 < 5%，視為收斂，提前終止
- 硬性上限 1000 次（FAP < 10⁻⁴ 的案例建議手動設 `fap_bootstrap_max_iter: 5000`）
- 輸出：最終 FAP 值、實際迭代次數、收斂狀態（`'converged'` / `'max_iter'`）

FAP 顯著閾值：0.001（0.1%）。

### 4.3 交叉驗證：DFT（Lenz & Breger, 2005 — Period04）

經典離散傅立葉轉換振幅譜，與 LS 週期圖並排顯示，目視確認主頻一致。

```
A(ν) = (2/N) × |Σᵢ (mᵢ − m̄) × exp(−i 2π ν tᵢ)|
```

### 4.4 傅立葉擬合與諧波數選擇——BIC 最小化

模型：V(φ) = a₀ + Σₙ [ aₙ cos(2πnφ) + bₙ sin(2πnφ) ]，n = 1 ... N

諧波數 N 由 BIC 最小化自動選取，上限為 yaml 設定的 `max_harmonics`（預設 8）：

```
BIC(N) = k·ln(n_data) − 2·ln(L̂)    k = 2N+1，高斯誤差假設
```

決策理由：n_cycles//2 沒有統計依據，在稀疏觀測時會過擬合。BIC 對模型複雜度施加懲罰，已知限制為高斯誤差假設在大氣閃爍主導時可能低估最佳階數。

### 4.5 週期不確定度（Kovacs, 1981）

```
σ_f ≈ σ_res / (T × √N)
σ_P = σ_f / f²
```

**重要：** σ_res 必須使用傅立葉擬合後的殘差標準差（非 LS 殘差）。f 使用傅立葉擬合收斂後的有效頻率（= 1 / period）。此為不確定度下限。

### 4.6 相位零點：兩步迭代（Breger et al.）

δ Scuti 型變星慣例：以亮度極大值（magnitude 最小值）為 φ = 0。

- 步驟 1：以 t[0] 為暫時零點做初次傅立葉擬合，在 φ ∈ [0,1] 稠密格網上找極大值相位 φ_max
- 步驟 2：計算 `t0_final = t[0] + φ_max × period`，以此重新折疊並做第二次擬合
- 驗證：第二次擬合後在稠密格網重新計算極大值相位 φ_check。若 φ_check > 0.05，`logger.error()` 輸出含 t0_final 數值的錯誤訊息，`raise ValueError` 阻止後續分析

### 4.7 預白化（Pre-whitening，Breger et al., 1993）

適用情境：Delta Scuti、RR Lyrae 等多頻率脈動變星。

**迭代流程：**

- 步驟 1：對當前殘差執行 LS 週期圖，找最高功率頻率 ν₁
- 步驟 2：傅立葉擬合（固定頻率 ν₁），從光度中減去擬合模型
- 步驟 3：計算殘差 S/N（在 ν₁ 附近 ±1 d⁻¹ 以外估計噪聲基準）
- 步驟 4：S/N ≥ 4.0 → 回到步驟 1；否則停止
- 步驟 5：已萃取頻率數 ≥ `max_frequencies`（yaml 預設 10）時強制停止

**輸出：**

- 圖（已實作）：每次迭代 1×3 診斷圖，所有迭代合併為單一 PNG：`prewhitening_{target}_{channel}.png`
- CSV（預留停用）：yaml `save_csv: false` 時不輸出，啟用時無需改 code

---

## 第五部分：已知限制

| # | 限制 | 處理方式 |
|---|------|----------|
| 1 | 週期不確定度為下限（Kovacs 公式假設噪聲主導） | 文件記錄，報告中說明 |
| 2 | BIC 選階假設高斯誤差，大氣閃爍主導時可能低估最佳階數 | 文件記錄 |
| 3 | 預白化假設頻率間無耦合，非線性脈動需謹慎解釋 | 文件記錄 |
| 4 | Bootstrap FAP 極低（< 10⁻⁴）案例收斂需更多迭代 | 手動設 `fap_bootstrap_max_iter: 5000` |
| 5 | 孔徑由比較星生長曲線決定，目標星 PSF 更胖時孔徑可能偏小 | 記錄為已知限制；差分測光部分抵消 |
| 6 | 孔徑硬邊界（hard-edge），無次像素加權 | 差分測光中部分抵消；PSF 漂移時殘差可能出現系統趨勢 |
| 7 | 距離與星表誤差雙重加權未正規化，某類權重可能主導 | 記錄為已知限制 |
| 8 | ensemble_normalize() 目前為滾動中位數簡化版 | 完整 Broeg (2005) 待 comp_lightcurves 擴充後實作 |
| 9 | 大氣消光修正（一階 + 色散）暫不實作 | 視野 < 2°、仰角 > 30° 時差分抵消；條件超出時須實作 |
| 10 | 改機 Canon 6D2 R 通道帶通偏移至近紅外 | FITS 標頭 IR_CUT=REMOVED，色彩轉換係數需自行標定 |
| 11 | 暗場無溫度追蹤 | FITS 標頭記錄溫差（DTMPDIFF） |
| 12 | SIP 畸變 WCS 傳遞（astrometry.net 有時產生 SIP 係數） | 拆色時另外處理，已知問題 |
| 13 | SXPhe 20251122 全 4 幀曝光 1s，ASTAP 無法偵測星點 | 此批資料不做 plate solve 與測光 |
| 14 | V1162Ori FITS 標頭 RA/DEC 錯誤（NINA 赤道儀連線問題） | yaml 填 `ra_hint_h` / `dec_hint_deg`；plate_solve.py 傳 `-ra` / `-spd` 給 ASTAP |
| 15 | DeBayer_RGGB.py 輸出的 split FITS 無 MID-OBS 標頭（ASTAP 不傳遞此欄位）|    photometry.py 從 DATE-OBS + EXPTIME/2 補算，不影響後續測光。

---

## 第六部分：待實作清單

| 優先級 | 項目 | 說明 |
|--------|------|------|
| 高 | 套用 photometry_patch.py | aperture_photometry、mag_error_from_flux、compute_annulus_radii、in_bounds_precise、FLAG_SLOPE_DEVIATION、ensemble_normalize、plot_psf_diagnostic |
| 高 | run_photometry_on_wcs_dir() 輸出 comp_lightcurves | 每顆比較星的逐幀儀器星等時間序列，啟用完整 Broeg (2005) ensemble 正規化 |
| 中 | 大氣消光修正 | 一階 + 色散消光。視野/仰角條件達到時實作。參考 Chromey & Hasselbacher (1996) |
| 中 | period_analysis.py 整合至 Photometry.ipynb | 選用 cell，不影響標準管線 |
| 低 | 預白化 CSV 輸出 | yaml `save_csv: true` 時啟用，code 已預留 |
| 低 | Gaia DR3 比較星介面 | 目前輸出警告，未實作查詢 |
| 低 | 目標孔徑獨立生長曲線驗證 | 對目標星也跑生長曲線，取較大值 |
| 低 | quality_report.py 實作 | 第二批 D 組 |

---

## 第七部分：v5 → v6 變更摘要

| 類別 | 變更 | 影響 |
|------|------|------|
| plate_solve.py 修正（§3.3） | 移除錯誤的 `-s db_type` 參數；ASTAP CLI 的 `-s` 為取樣數整數，非星表類型 | plate_solve 從全部失敗到正常運作 |
| targets 列表支援 | plate_solve.py / DeBayer_RGGB.py 支援 `obs_sessions.targets` 列表格式 | 多 target session 正確迭代 |
| 座標 hint 機制（§3.3） | yaml `targets` 區段加 `ra_hint_h` / `dec_hint_deg`；傳 `-ra` / `-spd` 給 ASTAP | V1162Ori 187/190 成功解算 |
| masters_dir 修正（§1.2） | Master 幀改存 `shared_calibration/{date}/masters/` | 多 target 共用同一組 masters |
| ISO fallback 機制（§3.1） | yaml → CR2 EXIF → FITS ISOSPEED 三層 fallback | ISO 3200 自動偵測，GAIN/RDNOISE 標頭正確填入 |
| 已知限制新增（§5） | #13 SXPhe 1s 曝光；#14 V1162Ori RA/DEC 標頭錯誤 | — |

---

## 第八部分：實測結果彙整

### Calibration（v5 基準）

| Target | Session | 幀數 | 結果 |
|--------|---------|------|------|
| AlVel | 20251122 | 17 | 17/17 ✅ |
| CCAnd | 20251122 | 86 | 86/86 ✅ |
| SXPhe | 20251122 | 4 | 4/4 ✅ |
| V1162Ori | 20251220 | 190 | 190/190 ✅（暗場自動選 3.7C） |

總耗時：4m 18s

### Plate Solve（v6 修正後）

| Target | Session | 幀數 | 成功 | 失敗 | 備註 |
|--------|---------|------|------|------|------|
| AlVel | 20251122 | 17 | 17 | 0 | — |
| CCAnd | 20251122 | 86 | 81 | 5 | 觀測品質（雲/導星） |
| SXPhe | 20251122 | 4 | 0 | 4 | 曝光 1s，無星可偵測 |
| V1162Ori | 20251220 | 190 | 187 | 3 | hint 機制正常運作；3 幀觀測品質 |

總耗時：9m 39s

---

## 第九部分：參考文獻

Berry, R., & Burnell, J. (2005). *The handbook of astronomical image processing* (2nd ed.). Willmann-Bell.

Breger, M., Stich, J., Garrido, R., Martin, B., Jiang, S. Y., Li, Z. P., Hube, D. P., Ostermann, W., Paparo, M., & Scheck, M. (1993). Nonradial pulsation of the delta Scuti star BU Cancri in the Praesepe cluster. *Astronomy & Astrophysics, 271*, 482–486.

Broeg, C., Fernández, M., & Neuhäuser, R. (2005). A new algorithm for differential photometry: Computing an optimum artificial comparison star. *Astronomische Nachrichten, 326*(2), 134–142. https://doi.org/10.1002/asna.200410350

Chromey, F. R., & Hasselbacher, D. A. (1996). The flat sky: Illumination corrections in CCD photometry. *Publications of the Astronomical Society of the Pacific, 108*(719), 944–949. https://doi.org/10.1086/133818

Eastman, J., Siverd, R., & Gaudi, B. S. (2010). Achieving better than 1 minute accuracy in the heliocentric and barycentric Julian dates. *Publications of the Astronomical Society of the Pacific, 122*(894), 935–946. https://doi.org/10.1086/655938

Henden, A. A., Templeton, M., Terrell, D., Smith, T. C., Levine, S., & Welch, D. (2012). APASS: The AAVSO photometric all-sky survey. *Journal of the American Association of Variable Star Observers, 40*(1), 430.

Honeycutt, R. K. (1992). CCD ensemble photometry on an inhomogeneous set of exposures. *Publications of the Astronomical Society of the Pacific, 104*(676), 435–440. https://doi.org/10.1086/133017

Howell, S. B. (2006). *Handbook of CCD astronomy* (2nd ed.). Cambridge University Press.

Janesick, J. R. (2001). *Scientific charge-coupled devices*. SPIE Press.

Kovacs, G. (1981). Frequency determination of double-mode cepheids from the analysis of their light curves. *Acta Astronomica, 31*, 75–93.

Lenz, P., & Breger, M. (2005). Period04 user guide. *Communications in Asteroseismology, 146*, 53–136.

Lomb, N. R. (1976). Least-squares frequency analysis of unequally spaced data. *Astrophysics and Space Science, 39*(2), 447–462. https://doi.org/10.1007/BF00648343

Merline, W. J., & Howell, S. B. (1995). A realistic model for point-sources imaged on array detectors: The model and initial results. *Experimental Astronomy, 6*(3), 163–210. https://doi.org/10.1007/BF00424040

Rousseeuw, P. J., & Bassett, G. W. (1990). The remedian: A robust averaging method for large data sets. *Journal of the American Statistical Association, 85*(409), 97–104. https://doi.org/10.2307/2289726

Scargle, J. D. (1982). Studies in astronomical time series analysis. II. Statistical aspects of spectral analysis of unevenly spaced data. *The Astrophysical Journal, 263*, 835–853. https://doi.org/10.1086/160554

Tamuz, O., Mazeh, T., & Zucker, S. (2005). Correcting systematic effects in a large set of photometric light curves. *Monthly Notices of the Royal Astronomical Society, 356*(4), 1466–1470. https://doi.org/10.1111/j.1365-2966.2004.08585.x

VanderPlas, J. T. (2018). Understanding the Lomb–Scargle periodogram. *The Astrophysical Journal Supplement Series, 236*(1), 16. https://doi.org/10.3847/1538-4365/aab766

Young, A. T. (1994). Air mass and refraction. *Applied Optics, 33*(6), 1108–1110. https://doi.org/10.1364/AO.33.001108
