# Story 3.2: Implement Alternating Optimization Logic

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a Algorithm Engineer,
I want to implement the custom `training_step` with alternating optimization phases (Update F then Update B),
so that the model properly optimizes both the feature extraction network and the discrete hash codes.

## Acceptance Criteria

1. **Manual Optimization Flow**: Must use `self.optimizers()` to retrieve `opt_f` and `opt_b`. [Source: architecture.md#Training Architecture]
2. **Alternating Logic**: Implement a "multi-phase" training step within a single `training_step` or by using `self.trainer.global_step` to toggle phases.
3. **Phase 1 (Update F)**:
    - Fixed $B$ (Binary Codes).
    - Update Feature Networks ($img\_enc$, $txt\_enc$, $proj$).
    - Calculate Loss $L = \alpha L_1 + \delta L_2 + L_3$.
    - Perform `opt_f.zero_grad()`, `self.manual_backward(loss)`, and `opt_f.step()`.
4. **Phase 2 (Update B)**:
    - Fixed Feature Networks.
    - Update Hash components ($gcn$, $hash\_layer$).
    - Perform `opt_b.zero_grad()`, `self.manual_backward(loss)`, and `opt_b.step()`.
5. **Loss Functions**: Implement placeholders or initial versions of:
    - $L_{rec}$ (Reconstruction): Measures Hamming distance approximation to $S$.
    - $L_{str}$ (Structure): GCN neighborhood consistency.
    - $L_{cm}$ (Cross-Modal): Alignment between $B_v$ and $B_t$.
6. **Gradient Isolation**: Verify use of `.detach()` to ensure gradients do not leak between phases where parameters should be fixed. [Source: epics.md#Story 3.2 AC]
7. **Logging**: Individually log `loss_f`, `loss_b`, and total `loss` using `self.log()`.

## Tasks / Subtasks

- [x] **Task 1: Implement Loss Functions (AC: 5)**
  - [x] Implement `compute_loss_rec` (L1).
  - [x] Implement `compute_loss_str` (L2).
  - [x] Implement `compute_loss_cm` (L3).
- [x] **Task 2: Design Alternating Execution (AC: 1, 2)**
  - [x] Modify `training_step` to alternate between Phase 1 and Phase 2.
  - [x] Use `self.optimizers()` to handle manual stepping.
- [x] **Task 3: Implement Phase 1: Update F (AC: 3, 6)**
  - [x] Core training logic for feature networks.
  - [x] Ensure B-related targets are detached.
- [x] **Task 4: Implement Phase 2: Update B (AC: 4, 6)**
  - [x] Core training logic for GCN and Hash layer.
- [x] **Task 5: Precision & Logging (AC: 7)**
  - [x] Add `self.log` calls for all components.
  - [x] Add `bfloat16` / `autocast` compatibility note for RTX 5080.

## Dev Notes

- **Hadamard Product**: $S = C \odot D$ calculation is critical.
- **Detaching**: Be careful! Torch tensors carry gradients by default. Use `.detach()` on any tensor coming from the "fixed" model during the other's update.
- **Paper Notation**: Keep using $B$, $F$, $S$ to maintain consistency with the research report.

### Project Structure Notes

- Logic stays within `src/models/agch_module.py`.
- Complex loss math can be moved to `src/models/losses.py` if it gets too large.

### References

- [Research: Loss Functions](file:///home/ncu-caic/Documents/Coding/github.com/natsuki221/AGCH-Impl/_bmad-output/planning-artifacts/research/technical-AGCH-research-2026-01-18.md#L156-L169)
- [Research: Alternating Update Strategy](file:///home/ncu-caic/Documents/Coding/github.com/natsuki221/AGCH-Impl/_bmad-output/planning-artifacts/research/technical-AGCH-research-2026-01-18.md#L213-L226)
- [Architecture: Manual Optimization](file:///home/ncu-caic/Documents/Coding/github.com/natsuki221/AGCH-Impl/_bmad-output/planning-artifacts/architecture.md#L139-L146)

## Dev Agent Record

### Agent Model Used
GPT-5.2-Codex

### Debug Log References
pytest -q

### Completion Notes List
- Implemented alternating optimization with phase toggling via `global_step` and manual optimizers.
- Added loss functions for reconstruction, structure, and cross-modal alignment with similarity matrix construction.
- Enforced gradient isolation with `.detach()` in both phases and added detailed logging metrics.
- Updated ATDD tests for explicit phase assertions and detach verification.
- All tests passed (pytest -q).

### File List
- `src/models/agch_module.py` (updated)
- `tests/test_story_3_2_atdd.py` (updated)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (updated)
- `_bmad-output/implementation-artifacts/3-2-implement-alternating-optimization-logic.md` (updated)

### Change Log
- 2026-01-18: Implemented alternating optimization, loss placeholders, logging, and ATDD updates.
