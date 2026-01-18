# ATDD Checklist - Story 3.2: Implement Alternating Optimization Logic

**Story**: 3-2-implement-alternating-optimization-logic
**Status**: Ready for Development (RED Phase)

## Acceptance Criteria Breakdown

1. **Manual Optimization Flow**
   - [x] Test Created: `test_manual_optimization_flow_ac1`
   - [ ] Implementation Status: Pending Refinement

2. **Alternating Logic**
   - [x] Test Created: `test_alternating_logic_execution_ac2_ac3_ac4`
   - [ ] Implementation Status: Skeleton exists, needs true alternating logic

3. **Phase 1 (Update F) & Phase 2 (Update B)**
   - [x] Test Created: `test_alternating_logic_execution_ac2_ac3_ac4`
   - [ ] Implementation Status: Pending

4. **Loss Functions**
   - [x] Test Created: `test_loss_functions_existence_ac5`
   - [ ] Implementation Status: **MISSING (RED)**

5. **Gradient Isolation**
   - [x] Test Created: `test_gradient_isolation_intent_ac6`
   - [ ] Implementation Status: Runtime check in place

6. **Logging**
   - [x] Test Created: `test_logging_keys_ac7`
   - [ ] Implementation Status: Basic logging exists, specific keys need verification

## Test Files

- `tests/test_story_3_2_atdd.py`: 5 Unit/Integration tests with Mocking

## Implementation Checklist

- [ ] Implement `compute_loss_rec`, `compute_loss_str`, `compute_loss_cm` methods in `AGCHModule`.
- [ ] Refactor `training_step` to implement true alternating phases (e.g. toggle by batch index or global step).
- [ ] Ensure correct optimizer selection using `self.optimizers()`.
- [ ] Implement proper gradient isolation (`.detach()`) for non-active phases.
- [ ] Update logging to include split losses (`train/loss_f`, `train/loss_b`).

## Execution Commands

```bash
# Run ATDD tests
pytest tests/test_story_3_2_atdd.py -v
```

## Red-Green-Refactor Cycle

- **RED**: `test_loss_functions_existence_ac5` is failing.
- **GREEN**: Implement loss methods and alternating logic.
- **REFACTOR**: Move complex loss logic to separate file if needed.
