# AGCH-Impl 專案文件索引 (Updated)

> **專案狀態**: 實作與驗證階段 (Implementation & Verification)  
> **掃描日期**: 2026-01-19  
> **掃描模式**: 全量全面掃描 (Exhaustive Full Scan)

---

## 專案概覽

| 項目 | 說明 |
|------|------|
| **專案名稱** | AGCH-Impl |
| **當前版本** | v1.2.0 (基於全面掃描) |
| **主要性能** | 16-bit mAP ≈ **0.696** (MIRFlickr-25K) |
| **架構類型** | 異構圖卷積雜湊 (Asymmetric GCN Hashing) |

---

## 核心文檔目錄

### 1. 系統架構與設計
- [**架構分析文件 (Architecture Analysis)**](./architecture-analysis.md) - 詳細的模型組件、前向傳播與損失函數邏輯。
- [**實作導論 (AGCH-Guide)**](./AGCH-Guide.md) - 繁體中文實作參考手冊。

### 2. 開發手冊
- [**專案概覽 (Project Overview)**](./project-overview.md) - 快速了解專案背景與技術指標。
- [**數據準備指南**](./project-overview.md#相關數據) - 關於 MIRFlickr 數據集處理的說明。

### 3. 開發記錄與驗收
- [**實施演練日誌 (Walkthrough)**](../_bmad-output/implementation-artifacts/walkthrough.md) - 包含重大失誤修復與超參數優化記錄。
- [**任務追踪清單 (Task List)**](../_bmad-output/implementation-artifacts/task.md) - 歷史任務完成狀態。

---

## 目錄結構分析

```
AGCH-Impl/
├── configs/             # Hydra 配置系統 (Model, Data, Trainer)
├── src/                 # 核心源代碼
│   ├── models/          # AGCHModule, BiGCN, ImgNet
│   ├── data/            # AGCHDataModule, HDF5 Dataset
│   ├── utils/           # 評估指標與日誌工具
│   └── train.py         # 訓練入口
├── scripts/             # 數據準備與實驗自動化腳本
├── docs/                # 自動化生成的專案文檔 (本目錄)
└── _bmad-output/        # 流程產物與歷史記錄
```

---

## 下一步開發建議

1. **多位元實驗**: 驗證 32-bit 與 64-bit 下的擴展性。
2. **多資料集驗證**: 嘗試在 NUS-WIDE 或 Wiki 資料集上運行。
3. **推論優化**: 將模型導出為 TorchScript 或 ONNX。

---

*此文件由 Paige (Tech Writer) 通過 BMM 全面掃描流程自動生成*
