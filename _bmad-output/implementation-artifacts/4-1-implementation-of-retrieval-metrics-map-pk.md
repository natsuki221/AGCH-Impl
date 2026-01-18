# Story 4.1: Implementation of Retrieval Metrics (mAP & P@k)

**Epic**: 4 - System Validation & Performance Evaluation
**Story ID**: 4.1
**Status**: done

## 📖 User Story

**As a** Data Scientist,
**I want** to implement mean Average Precision (mAP) and Precision@k metrics based on Hamming distance,
**So that** I can quantitatively evaluate the retrieval performance of the AGCH model.

## ✅ Acceptance Criteria

1.  **Given** Query binary codes and Retrieval binary codes (plus ground truth labels)
2.  **When** `calculate_mAP(query_code, retrieval_code, query_label, retrieval_label)` is called
3.  **Then** It should compute the Hamming distance using **vectorized matrix operations** (e.g., XOR via Bitwise ops or Dot product approximation) for high performance
4.  **And** It should sort results by distance (ascending)
5.  **And** It should return the correct mAP value compared to a known reference (unit test)
6.  **And** Precision@k should match expected values for top-k retrieved items

## 🔍 Context & Intelligence

### 🏗️ Architecture & Technical Specs

*   **Target File**: `src/utils/metrics.py` (New File)
*   **Integration Target**: `src/models/agch_module.py` (Update)
*   **Metric Type**: Custom Implementation (due to Hamming specific nature for hashing)
*   **Performance Constraint**: **MUST** use vectorized operations. NO for-loops over samples.

#### Implementation Guide

**1. Efficient Hamming Distance:**
For binary codes $u, v \in \{-1, 1\}^K$:
$$ D_H(u, v) = \frac{K - u \cdot v}{2} $$
Where $\cdot$ is the dot product. This allows using `torch.matmul` for ultra-fast distance calculation between Query set $Q$ and Database set $D$:
`dist_matrix = 0.5 * (K - Q @ D.T)`

**2. Metric Functions (`src/utils/metrics.py`):**
Implement the following standalone functions using PyTorch tensors:
*   `calculate_hamming_dist_matrix(query_code, retrieval_code)`: Returns $[N_q, N_db]$ distance matrix.
*   `calculate_mAP(dist_matrix, query_labels, retrieval_labels, top_k=None)`: Returns scalar mAP.
*   `calculate_precision_at_k(dist_matrix, query_labels, retrieval_labels, k)`: Returns scalar P@k.

**3. Integration (`src/models/agch_module.py`):**
The `AGCHModule` currently lacks validation logic. You must:
*   Implement `validation_step`: Collect `B` (binary codes) and `L` (labels).
*   Implement `on_validation_epoch_end`:
    *   Concatenate all codes/labels.
    *   **Note**: For standard retrieval eval, we typically use the *validation set* as the Query set and the *training set* (or a separate retrieval set) as the Database.
    *   *Constraint*: To keep it simple for now, perform retrieval within the validation set (Query=Validation, Retrieval=Validation) OR if `AGCHDataModule` provides a specific setup, follow that.
    *   *Decision*: Use the collected validation batches as BOTH Query and Retrieval set (excluding self-match) for monitoring purposes, OR simpler: just implement the valid step to log loss for now, and rely on `tests` to verify metrics.
    *   *Refinement*: The story specifically asks for **implementing the metrics**. Updating `AGCHModule` to use them is the logical next step. Implement a basic `validation_step` that computes mAP on the validation set itself (samples vs samples).

### 🚨 Critical Directives (DO NOT IGNORE)

*   **NO REINVENTION**: Do NOT implement slow Python loops for mAP. Use `torch.argsort` and cumulative sums.
*   **NO TORCHMETRICS (Strictly for this)**: While `torchmetrics` is great, `RetrivalMAP` often assumes probability scores. For clarity and specific Hamming requirements (and Story AC), implement the pure matrix version in `src/utils/metrics.py`.
*   **TYPE HINTS**: Strict type hints `(torch.Tensor, torch.Tensor) -> float` are required.
*   **MEMORY**: Be careful with `N x N` matrices on GPU if N is huge (25K is fine for 5080, but consider CPU offload if needed). For 25k, $25000^2 \times 4$ bytes $\approx 2.5$ GB, which fits easily on RTX 5080.

### 🧠 Previous Learnings (from Story 3.2)

*   `AGCHModule` uses `manual_optimization`. Validation step does NOT need manual opt, it's standard.
*   Outputs from `training_step` are detached. Ensure validation outputs are also detached/no-grad.

### 🧪 Testing Strategy

*   Create `tests/test_metrics.py`.
*   **Test Case 1**: Small manual example. 2 queries, 3 database items. Caclulate mAP by hand and verify code matches.
*   **Test Case 2**: Vectorized vs Loop equivalent (validation).
*   **Test Case 3**: Check `k` limits for Precision@k.

## 🛠️ Task List

- [x] Create `src/utils/metrics.py`
    - [x] Implement `calculate_hamming_dist` (Matrix ops)
    - [x] Implement `calculate_mAP`
    - [x] Implement `calculate_precision_at_k`
- [x] Create `tests/test_metrics.py` and verify implementation
- [x] Update `src/models/agch_module.py`
    - [x] Import metrics
    - [x] Implement `validation_step` to collect codes/labels
    - [x] Implement `on_validation_epoch_end` to compute and log `val/mAP`
    - [x] Ensure validation uses `torch.no_grad()` (standard in Lightning)

## 📦 Deliverables

1.  `src/utils/metrics.py`
2.  `tests/test_metrics.py`
3.  Updated `src/models/agch_module.py`

## Dev Agent Record

### Agent Model Used
GPT-5.2-Codex

### Debug Log References
pytest -q

### Completion Notes List
- Implemented vectorized Hamming distance, mAP, and Precision@k metrics in `src/utils/metrics.py`.
- Added validation collection and `val/mAP` logging to `AGCHModule` using the new metrics.
- All tests passed (pytest -q).

### File List
- `src/utils/metrics.py` (updated)
- `tests/test_metrics.py` (updated)
- `src/models/agch_module.py` (updated)
- `_bmad-output/implementation-artifacts/4-1-implementation-of-retrieval-metrics-map-pk.md` (updated)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (updated)

### Change Log
- 2026-01-19: Implemented retrieval metrics and validation logging; updated tests and story status.
- 2026-01-19: [Code Review] Fixed `Optional` type import, added `p0` pytest marker, fixed `pyproject.toml`, and committed new files.

## Review record 
- **Reviewer**: DEV Agent (Amelia)
- **Status**: PASSED
- **Findings**: 2 Medium (Uncommitted files, Code Maintainability), 1 Low (Type hinting)
- **Resolution**: All issues fixed via auto-fix.

