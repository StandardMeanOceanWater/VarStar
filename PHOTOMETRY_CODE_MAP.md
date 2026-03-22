# photometry.py 程式碼地圖

**版本**：1.6 | **總行數**：4059 | **最後更新**：2026-03-22

---

## 1. 頂層結構

### 1.1 設定與資料類別

| 行號 | 識別字 | 說明 |
|------|--------|------|
| 18 | `VERSION` | 模組版本常數 `"1.6"` |
| 64–187 | `class Cfg` | 所有測光參數的 dataclass；路徑、目標、比較星、孔徑、誤差模型、時間系統、幀篩選等 |
| 189–196 | `_resolve_extinction_k()` | 從 YAML `extinction.coefficients` 取通道消光係數，fallback 到 `photometry.extinction_k` |
| 199–448 | `cfg_from_yaml()` | 從 YAML 字典建立 `Cfg` 實例；處理路徑拼接、座標解析、儀器參數、波段對應等 |
| 718–724 | `CAMERA_SENSOR_DB` | 硬編碼 Canon 6D2 感測器增益/讀出噪聲查找表（備用） |
| 728–730 | `ISO_HEADER_KEYS`, `AIRMASS_WARN_THRESHOLD` | FITS header ISO 關鍵字列表；氣團警告閾值 |

### 1.2 FITS / WCS 工具

| 行號 | 識別字 | 說明 |
|------|--------|------|
| 458–460 | `_has_wcs()` | 檢查 FITS header 是否含完整 WCS 關鍵字 |
| 463–475 | `_estimate_fov_deg()` | 從 FITS header 估算視場角（度） |
| 478–514 | `run_astap_plate_solve()` | 呼叫 ASTAP 做單張 plate solve（舊版遺留） |
| 517–543 | `batch_plate_solve_all()` | 批次 plate solve（已停用） |

### 1.3 星點偵測

| 行號 | 識別字 | 說明 |
|------|--------|------|
| 547–592 | `detect_stars_with_radec()` | DAOStarFinder 偵測星點，附加 RA/Dec 座標 |
| 594–610 | `batch_detect_stars()` | 批次偵測並輸出 CSV |

### 1.4 座標轉換

| 行號 | 識別字 | 說明 |
|------|--------|------|
| 619–622 | `radec_to_pixel()` | RA/Dec → 像素座標（透過 WCS） |
| 624–626 | `in_bounds()` | 檢查像素座標是否在影像邊界內（含 margin） |

### 1.5 PSF 擬合

| 行號 | 識別字 | 說明 |
|------|--------|------|
| 633–637 | `gaussian2d()` | 二維橢圓高斯函數（供 `curve_fit` 用） |
| 640–692 | `fit_gaussian_psf()` | 對星點切片做 2D 高斯 PSF 擬合，回傳 FWHM 和通量 |

### 1.6 測光核心

| 行號 | 識別字 | 說明 |
|------|--------|------|
| 698–701 | `m_inst_from_flux()` | 淨通量 → 儀器星等（-2.5 log10） |
| 737–762 | `get_iso_value()`, `_parse_iso()` | 從 FITS header 讀取 ISO |
| 765–808 | `apply_gain_from_header()` | 設定 `cfg.gain_e_per_adu` 和 `cfg.read_noise_e`；優先 FITS header，備用 sensor DB |
| 811–820 | `require_cfg_values()` | 檢查必要 cfg 值（gain, read_noise）是否已填入 |
| 823–833 | `max_pixel_in_box()` | 取星點附近方框內最大像素值 |
| 836–860 | `compute_annulus_radii()` | 計算背景環內外半徑（Howell 2006 公式或 YAML 覆蓋） |
| 863–901 | `aperture_photometry()` | **核心孔徑測光**：圓形孔徑 + 環形背景估計，回傳淨通量、背景、像素數等 |
| 904–905 | `is_saturated()` | 判斷最大像素是否超過飽和閾值 |
| 908–975 | `_growth_radius_for_star()`, `estimate_aperture_radius()` | **生長曲線法**自動估算最佳孔徑半徑 |

### 1.7 零點回歸

| 行號 | 識別字 | 說明 |
|------|--------|------|
| 978–1071 | `robust_zero_point()` | **穩健零點回歸**：Huber 回歸（fallback OLS）+ 迭代 sigma clipping；回傳 slope a、intercept b、R² |
| 1074–1133 | `mag_error_from_flux()` | CCD 噪聲方程式（Merline & Howell 1995）計算星等誤差 |

### 1.8 星表查詢與匹配

| 行號 | 識別字 | 說明 |
|------|--------|------|
| 1136–1157 | `_pick_col()`, `_parse_ra_dec()` | DataFrame 欄位名稱容錯選取；RA/Dec 字串解析 |
| 1160–1182 | `read_aavso_seq_csv()` | 讀取本地 AAVSO 序列星 CSV |
| 1185–1239 | `fetch_aavso_vsp_api()` | 查詢 AAVSO VSP API（星名優先，fallback 座標查詢） |
| 1242–1262 | `filter_catalog_in_frame()` | 篩選落在影像視場內的星表星 |
| 1265–1291 | `select_comp_from_catalog()` | 從星表中按星等範圍和誤差排序選取比較星 |
| 1294–1338 | `select_check_star()` | 選取檢查星（手動指定或自動選最近非比較星候選） |
| 1348–1380 | `compute_airmass()` | Young (1994) 氣團公式 |
| 1383–1498 | `time_from_header()` | 從 FITS header 提取時間 → MJD, BJD_TDB, airmass |
| 1531–1572 | `_selection_radius_px()`, `_stars_in_circle()` | 計算選取圓半徑；篩選圓內候選星 |
| 1575–1618 | `_fetch_apass_from_cache()` | 從本機 APASS CSV 快取做錐形查詢 |
| 1621–1681 | `fetch_apass_cone()` | APASS DR9 查詢（本機快取優先 → VizieR fallback） |
| 1684–1744 | `fetch_tycho2_cone()` | Tycho-2 查詢（VizieR）；含 BT/VT → V 色彩轉換 |
| 1747–1843 | `fetch_gaia_dr3_cone()` | Gaia DR3 查詢（VizieR）；G → V/Rc/B 色彩轉換（依通道） |
| 1846–1887 | `_match_catalog_to_detected()` | 偵測星與星表位置交叉匹配（SkyCoord.match_to_catalog_sky） |

### 1.9 零點診斷

| 行號 | 識別字 | 說明 |
|------|--------|------|
| 1890–1951 | `_save_zeropoint_diagnostic()` | 輸出單幀零點回歸散佈圖（AAVSO 紅 + APASS 藍） |

### 1.10 比較星自動選取

| 行號 | 識別字 | 說明 |
|------|--------|------|
| 1954–2227 | `auto_select_comps()` | **比較星自動選取主函式**：偵測 → 星表查詢（依 catalog_priority）→ 合併去重 → 直接孔徑測光篩選 → 距離加權 → 回傳 comp_refs |

### 1.11 差分測光與 Ensemble

| 行號 | 識別字 | 說明 |
|------|--------|------|
| 2230–2236 | `differential_mag()` | 單星差分星等公式 |
| 2239–2400 | `ensemble_normalize()` | **Broeg (2005) ensemble 正規化**：迭代加權漂移估計 Δ(t)，更新權重 w = 1/RMS² |
| 2403–2421 | `class _FrameCompCache` | 同視野多目標共用的比較星測光快取 |

### 1.12 光變曲線繪圖

| 行號 | 識別字 | 說明 |
|------|--------|------|
| 2425–2550 | `plot_light_curve()` | 繪製光變曲線圖（Local Time 雙軸、BJD 標註、誤差棒） |

### 1.13 主測光迴圈

| 行號 | 識別字 | 說明 |
|------|--------|------|
| 2553–3534 | `run_photometry_on_wcs_dir()` | **逐幀差分測光主函式**（~980 行）：遍歷所有 FITS → 幀篩選 → 孔徑測光 → 零點回歸 → sigma clip → 消光改正 → ensemble → 輸出 CSV/PNG/診斷圖 |

### 1.14 CLI 入口

| 行號 | 識別字 | 說明 |
|------|--------|------|
| 3542–4059 | `if __name__ == "__main__"` | CLI 入口：argparse → 建立 (target, date) 列表 → 比較星選取 → 孔徑估計 → 多通道測光迴圈 → 週期分析 → G1/G2 比值 |

---

## 2. 資料流圖

```
observation_config.yaml
        │
        ▼
 ┌──────────────┐
 │ cfg_from_yaml │  ← load_pipeline_config()
 └──────┬───────┘
        │  Cfg dataclass
        ▼
 ┌──────────────────────────┐
 │ __main__                  │
 │  (target, date) 列表      │
 └──────┬───────────────────┘
        │
        ▼  對每個 (target, date)：
 ┌──────────────────────────┐
 │ auto_select_comps()       │ ← 第一張 FITS
 │  ├─ fetch_aavso_vsp_api() │
 │  ├─ fetch_apass_cone()    │     ← VizieR / 本機快取
 │  ├─ fetch_tycho2_cone()   │
 │  ├─ fetch_gaia_dr3_cone() │
 │  ├─ 合併去重              │
 │  ├─ _catalog_direct_phot()│     ← aperture_photometry()
 │  └─ select_check_star()   │
 └──────┬───────────────────┘
        │  comp_refs, check_star
        ▼
 ┌──────────────────────────┐
 │ estimate_aperture_radius()│ ← 生長曲線法
 │  └─ _growth_radius_for_star()
 │     └─ aperture_photometry()
 └──────┬───────────────────┘
        │  ap_radius
        ▼  對每個 channel (R, G1, G2, B)：
 ┌──────────────────────────────────────────────────────────┐
 │ run_photometry_on_wcs_dir()                               │
 │  │                                                        │
 │  │  對每張 FITS：                                          │
 │  │  ├─ time_from_header() → mjd, bjd_tdb, airmass        │
 │  │  ├─ 高度角截斷（airmass > alt_min_airmass → skip）      │
 │  │  ├─ FWHM 篩選（IRAFStarFinder）                        │
 │  │  ├─ 目標孔徑測光 ← aperture_photometry()                │
 │  │  ├─ 飽和檢查 ← is_saturated()                          │
 │  │  ├─ Sharpness Index 篩選                                │
 │  │  ├─ Peak Ratio 篩選                                     │
 │  │  ├─ 比較星孔徑測光 ← aperture_photometry() (+ 快取)     │
 │  │  ├─ robust_zero_point() → a, b, R²                     │
 │  │  ├─ ZP R² 篩選                                         │
 │  │  ├─ m_var = (m_inst_t - b) / a                         │
 │  │  ├─ mag_error_from_flux() → sigma_mag                   │
 │  │  └─ 檢查星測光                                          │
 │  │                                                        │
 │  ├─ Sigma clip（3σ MAD）                                   │
 │  ├─ 大氣消光一階改正（optional）                             │
 │  ├─ ZP 截距突變篩選（rolling median MAD）                    │
 │  ├─ 天空背景突升篩選（rolling median MAD）                   │
 │  ├─ Peak Ratio 自適應篩選（全夜 median - k×MAD）            │
 │  ├─ ensemble_normalize()（Broeg 2005，optional）            │
 │  ├─ plot_light_curve() → PNG                               │
 │  ├─ 零點診斷總覽圖 → PNG                                    │
 │  ├─ 剔除時序圖 → PNG                                       │
 │  └─ CSV, rejection_stats CSV, summary TXT                  │
 └──────────────────────────────────────────────────────────┘
        │
        ▼
 ┌──────────────────────────┐
 │ period_analysis.py        │ ← 外部模組
 │  run_period_analysis()    │
 └──────────────────────────┘
        │
        ▼
 ┌──────────────────────────┐
 │ G1/G2 比值光變曲線        │ ← 選用（G1 與 G2 同時存在時）
 └──────────────────────────┘
```

### 呼叫鏈簡圖

```
__main__
  ├─ cfg_from_yaml()
  ├─ auto_select_comps()
  │   ├─ fetch_aavso_vsp_api()
  │   ├─ fetch_apass_cone() → _fetch_apass_from_cache() / VizieR
  │   ├─ fetch_tycho2_cone()
  │   ├─ fetch_gaia_dr3_cone()
  │   ├─ _catalog_direct_phot() → aperture_photometry()
  │   └─ select_check_star()
  ├─ estimate_aperture_radius()
  │   └─ _growth_radius_for_star() → aperture_photometry()
  └─ run_photometry_on_wcs_dir()   [per channel]
      ├─ apply_gain_from_header()
      ├─ time_from_header() → compute_airmass()
      ├─ aperture_photometry()     [target + each comp]
      ├─ robust_zero_point()       [per frame]
      ├─ mag_error_from_flux()
      ├─ ensemble_normalize()      [post-loop, optional]
      ├─ plot_light_curve()
      └─ run_period_analysis()     [external]
```

---

## 3. 設定參數（Cfg dataclass）

### 3.1 路徑

| 欄位 | 預設 | 說明 |
|------|------|------|
| `wcs_dir` | `.` | split/{channel}/ FITS 目錄 |
| `out_dir` | `.` | 1_photometry 輸出目錄 |
| `phot_out_csv` | `photometry.csv` | 測光結果 CSV 路徑 |
| `phot_out_png` | `light_curve.png` | 光變曲線 PNG 路徑 |
| `zeropoint_diag_dir` | `zeropoint_diag` | 零點診斷圖目錄 |
| `run_root` | `.` | 執行根目錄（含時間戳） |

### 3.2 目標

| 欄位 | 預設 | 說明 |
|------|------|------|
| `target_name` | `""` | 目標星顯示名稱 |
| `target_radec_deg` | `(0, 0)` | 目標 RA/Dec（度） |
| `vmag_approx` | `8.0` | 目標近似 V 星等（用於比較星星等範圍計算） |

### 3.3 比較星選取

| 欄位 | 預設 | 說明 |
|------|------|------|
| `comp_mag_bright` | `4.0` | 比目標亮最多 N 等 |
| `comp_mag_faint` | `2.0` | 比目標暗最多 N 等 |
| `comp_mag_min/max` | 動態計算 | 實際星等篩選範圍 = vmag ± bright/faint |
| `comp_max` | `15` | 最多選幾顆比較星 |
| `comp_fwhm_min/max` | `2.0 / 8.0` | FWHM 容許範圍（px） |
| `comp_min_sep_arcsec` | `30.0` | 比較星最小間距（arcsec） |
| `catalog_priority` | `["AAVSO", "APASS"]` | 星表查詢優先序 |
| `aavso_fov_arcmin` | `100.0` | AAVSO 查詢視場（arcmin） |
| `apass_radius_deg` | `1.0` | APASS 查詢半徑（度） |
| `apass_match_arcsec` | `2.0` | 星表匹配容差（arcsec） |

### 3.4 孔徑測光

| 欄位 | 預設 | 說明 |
|------|------|------|
| `aperture_auto` | `True` | 啟用生長曲線自動孔徑 |
| `aperture_radius` | `8.0` | 孔徑半徑（px） |
| `aperture_min/max_radius` | `2 / 12` | 生長曲線搜尋範圍 |
| `aperture_growth_fraction` | `0.95` | 生長曲線收斂比例 |
| `annulus_r_in/out` | `None` | 手動背景環半徑（None = 自動 Howell 2006） |
| `saturation_threshold` | `65536` | 飽和閾值（DN） |
| `allow_saturated_target` | `True` | 允許目標星飽和 |

### 3.5 誤差模型

| 欄位 | 預設 | 說明 |
|------|------|------|
| `gain_e_per_adu` | `None` | 增益（e-/DN） |
| `read_noise_e` | `None` | 讀出噪聲（e-） |
| `plate_scale_arcsec` | `1.485` | 板塊比例尺（用於權重 ε） |

### 3.6 時間與大氣

| 欄位 | 預設 | 說明 |
|------|------|------|
| `apply_bjd` | `True` | 啟用 BJD_TDB 轉換 |
| `obs_lat/lon_deg` | `None` | 觀測站座標 |
| `alt_min_deg` | `30.0` | 最低仰角 |
| `alt_min_airmass` | `2.366` | 氣團截斷閾值 |
| `extinction_k` | `0.0` | 消光係數（0=停用） |

### 3.7 穩健回歸

| 欄位 | 預設 | 說明 |
|------|------|------|
| `robust_regression_sigma` | `3.0` | 迭代 sigma clipping 閾值 |
| `robust_regression_max_iter` | `5` | 最大迭代次數 |
| `robust_regression_min_points` | `3` | 最少回歸點數 |

### 3.8 Ensemble 正規化

| 欄位 | 預設 | 說明 |
|------|------|------|
| `ensemble_normalize` | `False` | 啟用 Broeg (2005) |
| `ensemble_min_comp` | `3` | 最少比較星數 |
| `ensemble_max_iter` | `10` | 迭代上限 |
| `ensemble_convergence_tol` | `1e-4` | 收斂閾值（mag） |

### 3.9 幀品質篩選

| 欄位 | 預設 | 說明 |
|------|------|------|
| `sharpness_min` | `0.3` | Sharpness Index 下限（flux(r=3)/flux(r=8)） |
| `zp_r2_min` | `0.0` | 零點 R² 下限（0=停用） |
| `peak_ratio_min` | `0.0` | Peak Ratio 固定下限（0=停用，已棄用） |
| `peak_ratio_k` | `0.0` | Peak Ratio 自適應門檻（median - k×MAD） |
| `zp_intercept_sigma` | `0.0` | ZP 截距突變閾值（倍 MAD，0=停用） |
| `sky_sigma` | `0.0` | 背景突升閾值（倍 MAD，0=停用） |

---

## 4. 關鍵決策點

### 4.1 比較星選取標準

**位置**：`auto_select_comps()` (L1954–2227) 及 `_catalog_direct_phot()` (L1996–2041)

- **選取圓**：以**影像中心**（非目標星）為圓心，半徑 = 影像短邊 / 2（L2016–2018）
- **星表優先序**：`catalog_priority`（預設 AAVSO → APASS），各星表結果合併後去重（位置 < 3 arcsec 保留優先序較前者）(L2143–2161)
- **星等篩選**：`comp_mag_min` 至 `comp_mag_max`（由 vmag_approx ± bright/faint 動態計算）(L252–273)
- **飽和排除**：`is_saturated(max_pix, saturation_threshold)` (L2028)
- **距離加權**：`w = 1 / (d_arcsec + ε)²`，ε = plate_scale / 2 (L2203)
- **誤差加權**：若 m_err 已知，`w /= m_err²` (L2789–2790)

### 4.2 零點回歸方法

**位置**：`robust_zero_point()` (L978–1071)

- **主要方法**：HuberRegressor（sklearn，epsilon=1.35）；sklearn 不可用時 fallback 到 `np.polyfit` (L1022–1039)
- **迭代 sigma clipping**：殘差 > σ × std 的點標記為 outlier，重新擬合，直到收斂或達到 `max_iter` (L1047–1058)
- **外插降級**：若目標儀器星等落在比較星範圍外，降級為 slope=1 純偏移 (L2851–2859)
- **R² 品質指標**：R² < 0.95 時輸出 WARN；`zp_r2_min > 0` 時自動剔除 (L1067–1069, L2866–2883)

### 4.3 Sigma Clipping / 離群值剔除

**位置**：`run_photometry_on_wcs_dir()` 內 (L2975–2995)

- **方法**：m_var 的 median ± 3 × 1.4826 × MAD（穩健 σ 估計）
- **前提**：至少 5 個 ok=1 的點才執行
- **效果**：被 clip 的幀 ok=0，ok_flag="sigma_clip"，仍保留在 CSV

### 4.4 幀品質篩選

**位置**：`run_photometry_on_wcs_dir()` 幀迴圈內

| 篩選器 | 行號 | 決策邏輯 | 預設狀態 |
|--------|------|---------|---------|
| 高氣團 | L2649 | `airmass > alt_min_airmass` → skip | 啟用（2.366） |
| 高 FWHM | L2655–2675 | IRAFStarFinder 全幀中位 FWHM > `comp_fwhm_max` | 啟用（8.0 px） |
| 低 Sharpness | L2696–2734 | flux(r=3)/flux(r=8) < `sharpness_min` | 啟用（0.3） |
| 低 Peak Ratio | L2736–2754 | max_pix/flux < `peak_ratio_min`（固定門檻） | 停用（0.0） |
| 低 ZP R² | L2865–2883 | R² < `zp_r2_min` | 停用（0.0） |
| ZP 截距突變 | L3014–3033 | rolling median 殘差 > `zp_intercept_sigma` × MAD | 停用（0.0） |
| 背景突升 | L3035–3054 | t_b_sky rolling 殘差 > `sky_sigma` × MAD | 停用（0.0） |
| Peak Ratio 自適應 | L3056–3076 | peak_ratio < median - `peak_ratio_k` × MAD | 停用（0.0） |
| Sigma clip | L2975–2995 | \|m_var - median\| > 3 × robust σ | 永遠啟用 |

---

## 5. 已知陷阱與耦合

### 5.1 全域狀態

- **`cfg` 是模組級全域變數**：`cfg_from_yaml()` 回傳新的 `Cfg` 物件，但 `__main__` 中直接賦值給 `cfg` 變數（L3677），且許多函式（`apply_gain_from_header`, `get_iso_value`, `require_cfg_values`, `_growth_radius_for_star`, `auto_select_comps` 等）直接讀寫全域 `cfg`。多通道迴圈中 `cfg` 被覆蓋（L3746–3751），上一通道的 gain/read_noise 被手動複製回來（L3750–3751）。
- **`apply_gain_from_header()` 有副作用**：直接修改 `cfg.gain_e_per_adu` 和 `cfg.read_noise_e` (L776–806)。第一幀設定後，後續幀不再觸發（因為不為 None）。

### 5.2 緊密耦合

- **`auto_select_comps()` ↔ `aperture_photometry()`**：前者在選取比較星時已用靜態 `cfg.aperture_radius` 做測光（L2024），但自動孔徑估計（`estimate_aperture_radius`）在其之後執行（L3714–3730）。這表示比較星的初選測光與正式測光使用不同孔徑。
- **`run_photometry_on_wcs_dir()` ↔ `plot_light_curve()`**：主函式內部有大段重複的 Local Time 刻度計算邏輯（L3369–3392），與 `plot_light_curve()` 內的相同邏輯（L2458–2481）完全重複。
- **`robust_zero_point()` 的 `_fit()` 內部函式**：依賴 sklearn 的 HuberRegressor，import 在函式內部（L1027）。sklearn 不可用時靜默 fallback 到 OLS，行為差異較大但無顯式警告。

### 5.3 非直覺行為

1. **Sharpness Index 用最亮比較星而非目標星**（L2701–2727）：篩選標準基於最亮未飽和比較星的通量比，而非目標星本身。
2. **Peak Ratio 有兩套篩選**：固定門檻 `peak_ratio_min`（幀迴圈內，L2736）和自適應門檻 `peak_ratio_k`（幀迴圈後，L3056）。前者已棄用但仍保留。
3. **外插降級**（L2851–2859）：當目標星儀器星等超出比較星範圍時，零點回歸的 slope 強制為 1.0，完全忽略回歸斜率。這在目標星很亮或很暗時會靜默生效。
4. **重複的 dead code**（L542–543）：`batch_plate_solve_all()` 的 return 語句出現兩次，第二次永遠不會執行。
5. **多處重複 import**：`SkyCoord`, `astropy.units`, `urllib`, `datetime` 等在不同位置重複 import（L616, L704–709, L1521–1526）。這是 Jupyter notebook 轉 .py 的遺跡。
6. **Ensemble normalize 後重寫 CSV**（L3443）：整個 CSV 被寫入兩次（先在 L3136，ensemble 後再在 L3443）。
7. **`_first_frame_diag_data` 的星等窄化**（L2820–2842）：診斷散佈圖只顯示 vmag_approx ± 1.0/1.5 範圍內的比較星，與實際回歸使用的範圍（comp_mag_min ~ comp_mag_max）不同。

---

## 6. 建議模組切分

若將此檔案拆分為獨立模組，以下是自然邊界：

### 模組 1：`phot_config.py` — 設定（~450 行）
- `Cfg` dataclass (L64–187)
- `cfg_from_yaml()` (L199–448)
- `_resolve_extinction_k()` (L189–196)

### 模組 2：`phot_core.py` — 測光核心（~350 行）
- `aperture_photometry()` (L863–901)
- `compute_annulus_radii()` (L836–860)
- `m_inst_from_flux()` (L698–701)
- `is_saturated()` (L904–905)
- `max_pixel_in_box()` (L823–833)
- `gaussian2d()`, `fit_gaussian_psf()` (L633–692)
- `_growth_radius_for_star()`, `estimate_aperture_radius()` (L908–975)
- `mag_error_from_flux()` (L1074–1133)

### 模組 3：`phot_zeropoint.py` — 零點回歸（~100 行）
- `robust_zero_point()` (L978–1071)
- `differential_mag()` (L2230–2236)

### 模組 4：`phot_catalog.py` — 星表查詢與匹配（~750 行）
- `fetch_aavso_vsp_api()` (L1185–1239)
- `fetch_apass_cone()`, `_fetch_apass_from_cache()` (L1575–1681)
- `fetch_tycho2_cone()` (L1684–1744)
- `fetch_gaia_dr3_cone()` (L1747–1843)
- `_match_catalog_to_detected()` (L1846–1887)
- `filter_catalog_in_frame()` (L1242–1262)
- `select_comp_from_catalog()` (L1265–1291)
- `select_check_star()` (L1294–1338)
- `read_aavso_seq_csv()` (L1160–1182)
- `_pick_col()`, `_parse_ra_dec()` (L1136–1157)

### 模組 5：`phot_compsel.py` — 比較星選取（~300 行）
- `auto_select_comps()` (L1954–2227)
- `_selection_radius_px()`, `_stars_in_circle()` (L1531–1572)

### 模組 6：`phot_ensemble.py` — Ensemble 正規化（~170 行）
- `ensemble_normalize()` (L2239–2400)

### 模組 7：`phot_timing.py` — 時間與座標（~200 行）
- `time_from_header()` (L1383–1498)
- `compute_airmass()` (L1348–1380)
- `radec_to_pixel()`, `in_bounds()` (L619–626)
- `get_iso_value()`, `apply_gain_from_header()`, `require_cfg_values()` (L737–820)

### 模組 8：`phot_diagnostics.py` — 診斷圖（~200 行）
- `_save_zeropoint_diagnostic()` (L1890–1951)
- `plot_light_curve()` (L2425–2550)
- 零點診斷總覽圖邏輯（L3143–3288）
- 剔除時序圖邏輯（L3290–3336）

### 模組 9：`phot_pipeline.py` — 主測光迴圈與 CLI（~1000 行）
- `run_photometry_on_wcs_dir()` (L2553–3534)
- `_FrameCompCache` (L2403–2421)
- `__main__` (L3542–4059)

### 遺留 / 可刪除

- `run_astap_plate_solve()`, `batch_plate_solve_all()` (L478–543)：已停用，可移除
- `detect_stars_with_radec()`, `batch_detect_stars()` (L547–610)：僅 `auto_select_comps` fallback 路徑使用
- `CAMERA_SENSOR_DB` (L718–724)：備用查找表，已被 YAML sensor_db 取代
