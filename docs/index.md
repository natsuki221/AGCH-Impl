# AGCH-Impl 專案文件索引

> **專案狀態**: 預實作階段（Pre-implementation）  
> **掃描日期**: 2026-01-18  
> **掃描模式**: 快速掃描

---

## 專案概覽

| 項目 | 說明 |
|------|------|
| **專案名稱** | AGCH-Impl |
| **專案類型** | 研究論文實作（Library/Data） |
| **目標技術** | Python + PyTorch（預計） |
| **主要領域** | 無監督跨模態檢索、圖神經網路、雜湊學習 |

---

## 快速參考

### 核心技術

- **AGCH**: Aggregation-based Graph Convolutional Hashing
- **關鍵創新**: 聚合相似度矩陣（Hadamard 乘積融合餘弦相似度與歐幾里得距離）
- **網路架構**: AlexNet（影像）+ MLP（文本）+ GCN + 融合模組

### 目標資料集

| 資料集 | 樣本數 | 類別 | 文本特徵維度 |
|--------|--------|------|--------------|
| Wiki | 2,866 | 10 | 10-d (LDA) |
| MIRFlickr-25K | 25,000 | 24 | 1386-d (PCA on BoW) |
| NUS-WIDE | 186,577 | 10 | 1000-d (Tag Index) |

---

## 參考文件

### 已存在文件

| 文件 | 說明 |
|------|------|
| [AGCH-Guide.md](./AGCH-Guide.md) | 實作參考指南（繁體中文） |
| [論文 PDF](./Aggregation-Based_Graph_Convolutional_Hashing_for_Unsupervised_Cross-Modal_Retrieval.pdf) | 原始論文全文 |

### 待生成文件

- [ ] `architecture.md` - 系統架構設計
- [ ] `development-guide.md` - 開發環境設置
- [ ] `api-reference.md` - API 參考文件

---

## 目錄結構

```
AGCH-Impl/
├── .agent/              # AI 助手配置
├── .github/             # GitHub 工作流程
├── _bmad/               # BMAD 工作流程系統
├── _bmad-output/        # BMAD 輸出產物
├── docs/                # 文件目錄
│   ├── index.md         # 本索引文件
│   ├── AGCH-Guide.md    # 實作指南
│   └── *.pdf            # 論文原文
└── AGENTS.md            # AI 技能配置
```

---

## 下一步建議

1. **執行 PRD 工作流程** - 定義產品需求規格
2. **執行架構設計工作流程** - 設計系統架構
3. **建立開發環境** - 設置 Python/PyTorch 環境

---

*此文件由 Document Project 工作流程自動生成*
