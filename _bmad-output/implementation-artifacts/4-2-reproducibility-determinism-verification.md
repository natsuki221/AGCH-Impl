# Story 4.2: Reproducibility & Determinism Verification

**Epic**: 4 - System Validation & Performance Evaluation
**Story ID**: 4.2
**Status**: done

## 📖 用戶故事 (User Story)

**作為** 研究人員 (Researcher)，
**我希望** 驗證訓練過程在固定隨機種子下是完全確定性的，
**以便** 我可以可靠地重現實驗結果 (符合 NFR-R1 要求)。

## ✅ 驗收標準 (Acceptance Criteria)

1.  **給定** 兩次完全相同的訓練運行，配置相同的固定隨機種子 (例如: seed=42)。
2.  **當** 系統執行前 100 次迭代 (iterations) 的訓練時。
3.  **那麼** 記錄的 Loss 值應精確到小數點後 6 位完全相同。
4.  **並且** 最終的模型權重 (state_dict) 應逐位元 (bit-for-bit) 相同。
5.  **並且** 首次評估輸出的 mAP 值應完全一致。
6.  **並且** 必須驗證 `Trainer(deterministic=True)` 標誌已正確啟用。

## 🔍 上下文與情報 (Context & Intelligence)

### 🏗️ 架構與技術規範 (Architecture & Technical Specs)

*   **目標文件**: `tests/test_reproducibility.py` (新文件), `src/train.py` (審查), `configs/trainer/default.yaml` (配置)。
*   **關鍵依賴**: PyTorch Lightning 的 `seed_everything` 和 `Trainer(deterministic=True)`。
*   **技術約束**:
    *   **GPU 確定性**: 在 RTX 5080 上使用 CUDA 時，必須設置 `torch.backends.cudnn.deterministic = True` 和 `benchmark = False` (Lightning 的 `deterministic=True` 標誌通常會處理這些)。
    *   **DataLoader**: `num_workers > 0` 時，worker 的種子必須正確設置 (Lightning 默認處理 `worker_init_fn`)。
    *   **Hash Layer**: 確保二進位哈希層的初始化也是受控的。

#### 實施指南 (Implementation Guide)

**1. 配置審查 (`configs/trainer/default.yaml`):**
確保 Trainer 配置支持確定性標誌：
```yaml
trainer:
  deterministic: True
  benchmark: False
```

**2. 種子機制驗證 (`src/train.py`):**
確保在訓練開始前調用了 `L.seed_everything(cfg.seed, workers=True)`。這是項目模板的一部分，但必須驗證其存在和位置。

**3. 創建再現性測試 (`tests/test_reproducibility.py`):**
創建一個專門的測試，運行兩次小型訓練循環並比較結果。
*   **步驟 A**: 設置 Seed=42，初始化 DataModule 和 Model，運行 5-10 個 batch，記錄所有 steps 的 loss 和最終權重。
*   **步驟 B**: 重置環境，設置 Seed=42，重複上述過程，記錄結果。
*   **步驟 C**: 斷言 A 和 B 的 Loss 列表和權重張量完全相等 (`torch.equal` 或 `allclose` with strict tolerance)。
*   *注意*: 為了速度，這應該是一個小型集成測試 (Integration Test)，可以使用 Mock 數據或極小的數據集子集。

### 🚨 關鍵指令 (Critical Directives)

*   **不要訓練完整的 Epoch**: 驗證再現性只需要前幾個 batch。不要浪費時間訓練整個 epoch。
*   **檢查所有隨機源**: 確保所有隨機源 (numpy, random, torch, torch.cuda) 都被 `seed_everything` 覆蓋。
*   **CUDA 警告**: 某些 CUDA 操作可能在算法上是非確定性的 (如 `atomicAdd`)。如果遇到這種情況，測試可能會失敗。如果發生，請記錄並嘗試在 CPU 上驗證以隔離原因，或者強制使用確定性算法 (`use_deterministic_algorithms=True`)。對於 Story 驗收，**必須** 實現確定性。

### 🧠 先前經驗 (Previous Learnings)

*   在 Story 4.1 中，我們實施了 `metrics.py`。再現性測試也可以檢查 `val/mAP` 的一致性，這是一個很好的端到端檢查。
*   `AGCHModule` 使用了 `manual_optimization`。確保手動反向傳播過程中的隨機性（如果有，例如 Dropout）也是受控的。

### 🧪 測試策略 (Testing Strategy)

*   **主要測試**: `tests/test_reproducibility.py`。
*   **測試命令**: `pytest tests/test_reproducibility.py`。

## 🛠️ 任務列表 (Task List)

- [x] 審查並更新 `configs/trainer/default.yaml` 以啟用 `deterministic` 標誌。
- [x] 審查 `src/train.py` 確保 `seed_everything` 被正確調用。
- [x] 創建 `tests/test_reproducibility.py`。
  - [x] 實施 `test_training_determinism`：運行兩次短訓練，比較 Loss 和 Weights。
  - [x] 實施 `test_initialization_determinism`：驗證模型初始化權重在固定種子下相同。
- [x] 運行測試並驗證通過。
- [x] 註冊 `unit` 標記到 `pyproject.toml` (Code Review Fix)。

## 📦 交付物 (Deliverables)

1.  `tests/test_reproducibility.py`
2.  更新的 `configs/trainer/default.yaml` (如果需要)
3.  驗證報告 (通過測試輸出證明)

## Dev Agent Record

### Agent Model Used
GPT-5.2-Codex

### Debug Log References
pytest -q

### Completion Notes List
- Implemented deterministic training tests with 100 iterations, weight equality, and mAP consistency checks.
- Added initialization determinism test and validated trainer deterministic configuration.
- All tests passed (pytest -q).

### File List
- `tests/test_reproducibility.py` (updated)
- `_bmad-output/implementation-artifacts/4-2-reproducibility-determinism-verification.md` (updated)
- `pyproject.toml` (updated)

### Change Log
- 2026-01-19: Implemented reproducibility tests and updated story status.
- 2026-01-19: [Code Review] Registered 'unit' marker in pyproject.toml to resolve warning.
- 2026-01-19: [Code Review] Story status updated to done.

### Review record
**Status**: PASSED
**Date**: 2026-01-19
**Reviewer**: Code Review Agent (Amelia)
**Findings**:
1. (Medium) Missing Pytest Marker: `tests/test_reproducibility.py` uses `@pytest.mark.unit`.
   - **Fix**: Added `"unit: Unit tests"` to `pyproject.toml`.
2. (Medium) Uncommitted Changes.
   - **Fix**: Committed all changes.
