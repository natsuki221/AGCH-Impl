# Story 5.2: Hyperparameter Tuning & Final Experiments

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a **Researcher**,
I want to **run experiments with different hyperparameters (alpha, beta, gamma, hash code length)**,
so that **I can identify the optimal configuration and report the final performance metrics (mAP) comparable to state-of-the-art methods.**

## Acceptance Criteria

1.  **Code Length Support**: The system must support training and evaluation with hash code lengths of **16, 32, and 64 bits**.
2.  **Hyperparameter Config**: Key hyperparameters (`alpha`, `beta`, `gamma`) must be explicitly configurable via Hydra CLI (already verified in Story 5.1, but needs extensive usage here).
3.  **Experiment Script**: A shell script `scripts/run_experiments.sh` must be created to automatically run the grid search or set of required experiments (e.g., varying bits).
4.  **Performance Goal**: The final best model must achieve mAP scores on MIRFLICKR-25K that are **comparable** (within reasonable margin, e.g., +/- 1-2%) to the reported results in the AGCH paper (if known) or baseline expectations for deep hashing.
5.  **Result Aggregation**: A method (e.g., notebook or script) to aggregate results from multiple runs into a summary table/report.

## Tasks / Subtasks

- [x] **Task 1: Implement Experiment Script (AC: 1, 3)**
  - [x] Create `scripts/run_experiments.sh`.
  - [x] Include commands for 16, 32, 64 bits.
  - [x] Ensure unique `hydra.run.dir` or `experiment` names for each run.
- [ ] **Task 2: Run Experiments (AC: 1, 2)**
  - [ ] Execute `scripts/run_experiments.sh`.
  - [ ] Monitor training (TensorBoard/WandB).
  - [ ] *Self-Correction*: If loss diverges, adjust LR or weights.
- [ ] **Task 3: Analyze & Tune (AC: 4, 5)**
  - [ ] Aggregate mAP results.
  - [ ] Compare with baselines.
  - [ ] If underperforming, perform grid search on `alpha`, `beta`.
- [ ] **Task 4: Final Report**
  - [ ] Update `README.md` with "How to reproduce results".
  - [ ] Create a summary markdown/artifact showing the final mAP scores.

## Dev Notes

- **Parallel Execution**: Experiment script might run sequentially (slow) or parallel (needs GPU management). For now, sequential is safer unless we have multi-GPU.
- **Config Groups**: Ensure `configs/model/agch.yaml` defaults are sensible.
- **Resource Usage**: Full experiments can take time. Use `fast_dev_run` strictly for debugging the *script*, not the model.

### Project Structure Notes

- **Scripts**: Place in `scripts/`.
- **Logs**: Ensure `logs/` directory is structured by experiment name.

### References

- [Epics: Story 5.2](file:///home/ncu-caic/Documents/Coding/github.com/natsuki221/AGCH-Impl/_bmad-output/planning-artifacts/epics.md#Story-5.2)
- [Story 5.1: Integration](file:///home/ncu-caic/Documents/Coding/github.com/natsuki221/AGCH-Impl/_bmad-output/implementation-artifacts/5-1-end-to-end-system-integration.md)

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

- ✅ Processed `scripts/` folder logic (Task 1).
- ✅ Created `scripts/prepare_data.py` for standard MIRFlickr download and feature extraction.
- ⚠️ Data Preparation partially complete (Test mode: 50 images). User opted to commit code first. Full data prep required before experiments.
- ✅ Created `scripts/run_experiments.sh` for multi-bit experiments.


- ✅ Added `scripts/run_experiments.sh` with explicit 16/32/64-bit runs and unique `hydra.run.dir` per run; validated via existing tests.

### File List

- scripts/run_experiments.sh
