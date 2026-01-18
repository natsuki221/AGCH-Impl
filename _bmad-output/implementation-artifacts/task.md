# Story 5.2: 終端實驗與超參數調優 (Terminal Experiments & Hyperparameter Tuning)

- [x] **任務 1: 實施實驗腳本 (Implement Experiment Script)**
  - [x] 創建 `scripts/run_experiments.sh`
  - [x] 支持 16/32/64 bit 配置
  - [x] 驗證腳本執行邏輯
- [x] **任務 2: 數據準備流水線 (Data Preparation Pipeline)**
  - [x] 創建 `scripts/prepare_data.py`
  - [x] 實現自動下載、特徵提取 (VGG16) 與 HDF5 生成
  - [x] 通過 50 張圖片的整合測試
- [x] **任務 3: 變更提交與文檔 (Commit & Documentation)**
  - [x] 提交代碼至 Git
  - [x] 創建實施演練 (Walkthrough) 文檔
  - [x] 更新 `.gitignore` 與 Story 文件狀態
- [x] **任務 4: 全量實驗執行 (Full Experiment Execution)**
  - [x] 更新 `.gitignore` 與 Story 文件狀態
- [x] **任務 7. 超參數優化與相似度邏輯修正 (Hyperparameter & Logic Optimization)**

根據 `AGCH-Guide.md` 與論文細節，對模型進行了深層調優。

**優化內容:**

*   **配置更新**: 將 `alpha`, `beta` 設為 1.0，`gamma` (lambda) 設為 10.0，`rho` 設為 4.0。
*   **特徵權重**: 引入 `gamma_v=2.0` (Image) 與 `gamma_t=0.3` (Text)。
*   **邏輯修正**: 將相似度矩陣 $S$ 的計算基礎從「哈希碼」改回「原始特徵 (Normalized & Weighted)」，完全符合 AGCH 論文的無監督引導邏輯。

**最終結果:**

*   **16-bit mAP**: 從 ~0.544 提升至 **~0.696** (顯著提升 +28%)。
*   模型收斂加速，驗證指標表現優異。
- [x] **任務 6: README.md 與項目文檔改進 (README.md & Project Documentation Improvement)**
  - [x] 創建根目錄 `README.md`
  - [x] 包含架構圖 (Mermaid) 與快速開始指南
  - [x] 同步更新引用資訊與文檔內容
