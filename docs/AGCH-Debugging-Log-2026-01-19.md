# AGCH 模型偵錯與效能優化紀錄

**日期**: 2026-01-19
**作者**: Ncu-caic (協作 AI: Tech Writer)
**狀態**: 已修復 (mAP 0.525 → 0.648)

---

## 🚀 摘要

本文件記錄了針對 AGCH (Aggregation-Based Graph Convolutional Hashing) 模型在 MIRFlickr-25K 資料集上效能低落問題的完整偵錯過程。經過四個階段的關鍵修復，模型 mAP 從接近隨機猜測的 **0.525** 提升至 **0.648**，並成功解決了嚴重的模式崩塌 (Mode Collapse) 問題。

## 🔍 問題診斷與修復歷程

### 1. 相似度矩陣計算錯誤 (Similarity Matrix Miscalculation)

**問題描述**:
原始實作錯誤地使用了加權和公式，且相關參數 (Dist項) 計算不符合論文定義。這導致構建的鄰接矩陣無法正確反映樣本間的語義相似度。

**修正方案**:
- 將公式 `S = α*C + β*D` 修正為論文要求的 Hadamard 乘積 `S = C ⊙ D`。
- 修正 `D` 矩陣計算公式，使用歐式距離平方根 `exp(-√dist/ρ)`。
- 添加關鍵的量化步驟 `S = 2S - 1` 以將範圍映射至 [-1, 1]。

```python
# Before
S = alpha * C + beta * D

# After (Correct)
S = C * D
S = 2.0 * S - 1.0
```

### 2. 文本編碼器輸入稀疏性 (Sparse Input Issue)

**問題描述**:
Text Encoder (`TxtNet`) 的輸出均值接近零 (`0.013`)。深入分析發現輸入的 BoW (Bag-of-Words) 特徵極度稀疏 (非零比例僅 0.25%)，導致神經網絡在前向傳播中訊號衰減嚴重。

**修正方案**:
- 在 `TxtNet` 輸入層添加 `BatchNorm1d` 以標準化稀疏輸入。
- 使用 `LeakyReLU` 替代 `ReLU` 以避免死神經元問題。
- 調整初始化策略，增大第一層權重建 (Gain=2.0)。

**結果**:
修正後 Text Encoder 輸出均值提升至 `0.095` (約 7 倍)，訊號恢復正常。

### 3. 跨模態對齊失效 (Cross-modal Misalignment)

**問題描述**:
儘管分別訓練了 Image 和 Text 編碼器，但同一樣本的圖像和文本 Hash Code 之間的漢明距離 (Hamming Distance) 高達 32.89 (接近隨機的 32)，表明模型未能學習到同一對象的跨模態一致性。

**修正方案**:
在 `loss_cm` 中添加直接對齊項，強制同一樣本的 Image Code (`B_v`) 與 Text Code (`B_t`) 相互接近。

```python
# Before
loss_cm = 0.5 * (loss(B_v, B_h) + loss(B_t, B_h))

# After
loss_cm = 0.5 * (loss(B_v, B_h) + loss(B_t, B_h)) + mean((B_v - B_t)**2)
```

### 4. 模式崩塌 (Mode Collapse) - **關鍵問題**

**問題描述**:
在修復上述問題後，發現所有樣本生成的 Hash Code **完全相同** (Unique Codes = 1/500)。這導致所有樣本間的相似度均為 1.0，mAP 因此停滯在隨機基線。

**修正方案**:
引入兩個正則化損失項：
1.  **量化損失 (Quantization Loss)**: `(|B| - 1)²`，強制 Hash Code 接近二值化 (±1)。
2.  **平衡損失 (Balance Loss)**: `mean(B)²`，強制每個 Bit 的正負分佈均勻，防止全 1 或全 -1。

同時調整損失權重，降低 L3 (跨模態損失) 的權重，給予模型更多自由度保持樣本差異。

**結果**:
- Unique Codes 恢復正常 (500/500)。
- 正負樣本對的相似度 Gap 拉開至 0.1174。

---

## 📊 最終效能 (Validation Results)

| 指標 (Metric) | 初始狀態 | 修復後 (Epoch 50) | 修復後 (Epoch 100) |
| :--- | :--- | :--- | :--- |
| **mAP (Total)** | 0.525 | 0.641 | **0.648** |
| mAP (I → T) | 0.525 | 0.630 | **0.638** |
| mAP (T → I) | 0.525 | 0.651 | **0.657** |

---

## 📝 後續建議

雖然模型效能已大幅改善，但距離論文報告的 0.85+ 仍有差距。建議下一步：
1.  **特徵對齊**: 確認輸入特徵提取方式 (如 VGG vs AlexNet) 是否與論文完全一致。
2.  **GCN 架構微調**: 實驗增加 GCN 層數或調整鄰域聚合方式。
3.  **超參數優化**: 重新進行超參數搜索 (特別是 `delta` 和 `gamma`)。
