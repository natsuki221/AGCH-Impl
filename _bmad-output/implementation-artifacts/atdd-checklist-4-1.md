
# ATDD Checklist: Story 4.1 - Implementation of Retrieval Metrics (mAP & P@k)

**Story ID**: 4.1
**Date**: 2026-01-19
**Author**: TEA Agent (Murat)

## 🎯 Acceptance Criteria & Test Coverage

| Criteria ID | Requirement | Test File | Test Method | Status |
|-------------|-------------|-----------|-------------|--------|
| AC 1-3 | Compute Hamming distance using vectorized matrix operations | `tests/test_metrics.py` | `test_hamming_distance_matrix_correctness` | 🔴 RED |
| AC 3 (Perf)| Ensure solution supports batching (Vectorized) | `tests/test_metrics.py` | `test_vectorized_high_performance_requirement` | 🔴 RED |
| AC 4-5 | Return correct mAP value compared to manual calc | `tests/test_metrics.py` | `test_map_calculation_manual` | 🔴 RED |
| AC 6 | Precision@k matches expected values | `tests/test_metrics.py` | `test_precision_at_k_correctness` | 🔴 RED |

## 🏗️ Supporting Infrastructure

### Test Data Strategy
*   **Vectorized Tensors**: Tests use `torch.tensor` directly to verify mathematical correctness.
*   **Determinism**: Hardcoded binary vectors `[1, 1, -1]` used in tests to ensure consistent, debuggable failures.
*   **Fixture Needs**: None currently required (pure function logic).

### Mocking Requirements
*   None. This is a pure utility module implementation.

## 📝 Implementation Checklist (Red-Green-Refactor)

**Current State**: 🔴 RED (All 4 tests failing)

### Step-by-Step Implementation Guide

#### 1. Implement Hamming Distance
- [ ] Implement `calculate_hamming_dist_matrix` in `src/utils/metrics.py`.
- [ ] Use formula: $D_H(u, v) = \frac{K - u \cdot v}{2}$.
- [ ] Use `torch.matmul` for the dot product.
- [ ] Verify `tests/test_metrics.py::test_hamming_distance_matrix_correctness` passes.
- [ ] Verify `tests/test_metrics.py::test_vectorized_high_performance_requirement` passes.

#### 2. Implement mAP
- [ ] Implement `calculate_mAP` in `src/utils/metrics.py`.
- [ ] Use `torch.argsort(dist_matrix)` to get ranking.
- [ ] Compute cumulative matches.
- [ ] Verify `tests/test_metrics.py::test_map_calculation_manual` passes.

#### 3. Implement Precision@k
- [ ] Implement `calculate_precision_at_k` in `src/utils/metrics.py`.
- [ ] Slice the sorted results at `k`.
- [ ] Verify `tests/test_metrics.py::test_precision_at_k_correctness` passes.

## 🚀 Execution Instructions

```bash
# Run the specific test suite
pytest tests/test_metrics.py -v

# Run with loop failure detection (if implemented in future)
pytest tests/test_metrics.py --durations=0
```

## 🧠 Technical Context for Dev
*   **Vectorization is Key**: The test `test_vectorized_high_performance_requirement` checks correct shape handling for batches. Do not iterate.
*   **Type Hints**: Ensure inputs are `torch.Tensor` and outputs are standard floats for scalar metrics.
*   **Validation**: The `AGCHModule` integration is NOT covered by these unit tests (it will be covered in integration tests later), but these functions must work correctly first.
