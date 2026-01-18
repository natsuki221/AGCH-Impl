# ATDD Checklist - Story 5.2: Hyperparameter Tuning & Final Experiments

**Status**: RED (Tests Failing)
**Story ID**: 5.2
**Primary Test Level**: Integration / Script Verification

## Acceptance Criteria & Test Coverage

| ID | Criteria | Test Level | Test File | Status |
|----|----------|------------|-----------|--------|
| 1 | Code Length Support (16, 32, 64) | Integration | `tests/test_experiments_script.py` | 🔴 |
| 2 | Hyperparameter Config (CLI) | Integration | Verified in Story 5.1 (test_cli_overrides) | 🟢 |
| 3 | Experiment Script (Exists & Executable) | Integration | `tests/test_experiments_script.py` | 🔴 |
| 4 | Performance Goal | Manual/Analysis | N/A (Checked in Final Report) | ⚪ |
| 5 | Result Aggregation | Manual/Analysis | N/A (Checked in Final Report) | ⚪ |

## Test Infrastructure Created

*   [x] **Test File**: `tests/test_experiments_script.py`
    *   Verifies script existence.
    *   Verifies script permissions (+x).
    *   Verifies script content (calls 16/32/64 bits).

## Implementation Checklist

### 1. Experiment Script (`scripts/run_experiments.sh`)

- [ ] Create `scripts/` directory.
- [ ] Create `scripts/run_experiments.sh`.
- [ ] Add shebang `#!/bin/bash`.
- [ ] Add commands for 16-bit, 32-bit, 64-bit runs.
- [ ] Use `hydra.run.dir` to separate logs (e.g., `logs/experiments/16bit`).
- [ ] Make executable (`chmod +x`).

### 2. Execution & Analysis

- [ ] Run the script.
- [ ] Watch TensorBoard.
- [ ] Aggregate results into a table.

## Red-Green-Refactor Workflow

1.  **RED**: `pytest tests/test_experiments_script.py` fails (Script missing).
2.  **GREEN**: Create the script with required commands.
3.  **REFACTOR**: Add comments, robustness (set -e).

## Execution Commands

```bash
# Run script verification tests
pytest tests/test_experiments_script.py
```
