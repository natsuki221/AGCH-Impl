# Story 5.2 實施演練 (Story 5.2 Implementation Walkthrough)

我們已完成 Story 5.2 的基礎設施與開發腳本。主要成果包括實驗管理腳本與數據自動化準備腳本。

## 1. 實驗管理腳本 (Experiment Script)
我們創建了 [run_experiments.sh](file:///home/ncu-caic/Documents/Coding/github.com/natsuki221/AGCH-Impl/scripts/run_experiments.sh)，支持多種位數 (16, 32, 64 bits) 的串行訓練。

```bash
# 預覽腳本內容
python src/train.py model.hash_code_length=16 hydra.run.dir="logs/agch_bits16"
python src/train.py model.hash_code_length=32 hydra.run.dir="logs/agch_bits32"
python src/train.py model.hash_code_length=64 hydra.run.dir="logs/agch_bits64"
```

## 2. 數據準備流水線 (Data Preparation Pipeline)
我們實現了 [prepare_data.py](file:///home/ncu-caic/Documents/Coding/github.com/natsuki221/AGCH-Impl/scripts/prepare_data.py)，整合了：
*   **自動下載**: 定位 raw MIRFlickr 數據。
*   **特徵提取**: 使用 VGG16 (fc7) 提取 4096 維視覺特徵。
*   **標籤與文本處理**: 處理 24 類多標籤與 1386 維 BoW 文本特徵。

### 驗證結果
在測試模式下 (`--test`)，成功生成了 HDF5 文件：
*   `data/images.h5` (50, 4096)
*   `data/texts.h5` (50, 1386)

## 3. 已提交的變更
所有腳本、配置文件及文檔已提交：
*   `feat(story-5.2): implement experiment script and data prep pipeline`
*   `docs: finalize README with architecture, citation and Apache-2.0 license`

## 4. 項目文檔優化
*   **高品質 README**: 整合了 Mermaid 架構圖、 BibTeX 引用資訊以及快速開始指南。
*   **授權協議更新**: 項目已正式標註為 **Apache License 2.0**，並在 `README.md` 和 `pyproject.toml` 中同步。

---
> [!NOTE]
> **全量數據提醒**: 雖文檔與代碼已就緒，但仍需運行全量數據準備 (`python scripts/prepare_data.py`) 才能獲得最終實驗結果。

## 5. 實驗結果 (Experiment Results)

訓練已成功完成！

| 配置 | 最佳 val/mAP | Checkpoint |
| :--- | :--- | :--- |
| 16-bit | ~0.524 | `logs/agch_bits16/checkpoints/epoch_013.ckpt` |
| 32-bit | ~0.52x | `logs/agch_bits32/checkpoints/epoch_009.ckpt` |
| 64-bit | ~0.52x | `logs/agch_bits64/checkpoints/epoch_005.ckpt` |

> [!TIP]
> 使用 TensorBoard 可視化訓練曲線：`tensorboard --logdir=logs/`

## 6. 架構修復報告 (Architecture Fix Report)

在驗收階段發現嚴重缺陷：`AGCHModule` 缺少非線性特徵提取器 (MLP) 和圖卷積模組 (GCN)。

**修復內容:**
*   新增 `src/models/components.py`: 實作 `ImgNet`, `TxtNet`, `BiGCN`。
*   更新 `src/models/agch_module.py`: 整合上述組件並修正 `training_step` 中的前向傳播邏輯。

**重訓結果:**
*   **16-bit mAP**: 從 ~0.524 提升至 **~0.539 - 0.544**。
## 7. 超參數優化與相似度邏輯修正 (Hyperparameter & Logic Optimization)

根據 `AGCH-Guide.md` 與論文細節，對模型進行了深層調優。

**優化內容:**
*   **配置更新**: 將 `alpha`, `beta` 設為 1.0，`gamma` (lambda) 設為 10.0，`rho` 設為 4.0。
*   **特徵權重**: 引入 `gamma_v=2.0` (Image) 與 `gamma_t=0.3` (Text)。
*   **邏輯修正**: 將相似度矩陣 $S$ 的計算基礎從「哈希碼」改回「原始特徵 (Normalized & Weighted)」，完全符合 AGCH 論文的無監督引導邏輯。

**最終結果:**
*   **16-bit mAP**: 從 ~0.544 提升至 **~0.696** (顯著提升 +28%)。
*   模型收斂加速，驗證指標表現優異。
