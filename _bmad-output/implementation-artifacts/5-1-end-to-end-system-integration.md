# Story 5.1: End-to-End System Integration

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a **Machine Learning Engineer**,
I want to **implement the full training logic in `src/train.py` by instantiating the DataModule, Model, and Trainer with correct callbacks and loggers**,
so that **I can execute the complete training pipeline from a single command and ensure all components work together correctly.**

## Acceptance Criteria

1.  **Configuration Loading**: The system must correctly load Hydra configuration (`configs/train.yaml` or similar) and instantiate classes based on it.
2.  **Pipeline Initialization**:
    - Instantiate `AGCHDataModule` (calling `prepare_data` / `setup` implies working HDF5 paths).
    - Instantiate `AGCHModule` (with correct `net` configuration).
    - Instantiate `Trainer` with:
        - `ModelCheckpoint` (monitoring `val/mAP`).
        - `RichProgressBar` or `TQDMProgressBar`.
        - `WandbLogger` or `TensorBoardLogger`.
3.  **Execution**: Calling `trainer.fit(model, datamodule=datamodule)` must run without errors for a full epoch.
4.  **Testing Integration**: `trainer.test()` must be called after fitting (if `test: True` in config) using the best checkpoint.
5.  **Smoke Test (Auto-Integration)**: Must pass `tests/test_integration_train.py` using `fast_dev_run=True` (or `experiment=debug`) to verify crash-free execution.
6.  **Artifact Verification**: The output directory (hydra-managed) must contain valid `.ckpt` files and `config.yaml`.
7.  **Override Testing**: CLI overrides (e.g., `python src/train.py model.alpha=0.5`) must be propagated to the saved configuration.

## Tasks / Subtasks

- [x] **Task 1: Implement Integration Test Skeleton (Test-First) (AC: 5, 6, 7)**
  - [x] Create `tests/test_integration_train.py`.
  - [x] Implement `test_train_fast_dev_run` (Smoke Test).
  - [x] Implement `test_hydra_artifacts` (Artifact Verification).
  - [x] Implement `test_cli_overrides` (Override Testing).
  - [x] *Note*: These tests will fail initially (RED).
- [x] **Task 2: Implement Main Training Script (AC: 1, 2, 3, 4)**
  - [x] Update `src/train.py` to replace placeholder logic.
  - [x] Use `hydra.utils.instantiate` for modular initialization.
  - [x] Setup `L.seed_everything` properly.
  - [x] Implement the `train(cfg)` function with `fit` and `test` calls.
  - [x] Ensure `utils.log_hyperparameters` is utilized.
- [x] **Task 3: Verify & Debug (AC: 3, 5)**
  - [x] Run `python src/train.py experiment=debug` (or fast_dev_run) manually.
  - [x] Run `pytest tests/test_integration_train.py` until GREEN.
- [x] **Task 4: Documentation & Cleanup**
  - [x] Ensure `src/train.py` has type hints.
  - [x] Update project README if run commands change.

## Dev Notes

- **Hydra Instantiation**: We rely heavily on Hydra. Ensure `configs/train.yaml` instantiates the *correct* classes (`src.models.agch_module.AGCHModule`, etc.).
- **Worker Seeds**: `seed_everything(workers=True)` is critical for Reproducibility (Story 4.2).
- **Metric Monitoring**: Checkpoint callback should monitor `val/mAP` (mode='max'). If `val/mAP` isn't available in the first epoch validation, the trainer might crash or not save. Ensure `AGCHModule` logs it properly.

### Project Structure Notes

- **Entry Point**: `src/train.py` is the singular entry point.
- **Tests**: Integration tests go to `tests/` folder.

### References

- [Architecture: Training Loop](file:///home/ncu-caic/Documents/Coding/github.com/natsuki221/AGCH-Impl/_bmad-output/planning-artifacts/architecture.md#L139)
- [Epics: Story 5.1](file:///home/ncu-caic/Documents/Coding/github.com/natsuki221/AGCH-Impl/_bmad-output/planning-artifacts/epics.md#Story-5.1)
- [Story 4.2: Reproducibility](file:///home/ncu-caic/Documents/Coding/github.com/natsuki221/AGCH-Impl/_bmad-output/implementation-artifacts/4-2-reproducibility-determinism-verification.md)

## Dev Agent Record

### Agent Model Used
GPT-5.2-Codex

Claude 3.5 Sonnet

### Debug Log References
python src/train.py trainer.fast_dev_run=True trainer.accelerator=cpu logger=csv extras.print_config=False data.data_dir=/tmp/agch_dummy
pytest tests/test_integration_train.py -q
pytest -q

### Completion Notes List
- Implemented Hydra-based training pipeline with instantiated datamodule, model, callbacks, and logger.
- Added CSV logger config and extras config for CLI overrides used in integration tests.
- Updated callbacks to monitor `val/mAP` and aligned Trainer/callback targets with `pytorch_lightning`.
- Verified fast-dev-run and integration tests; full test suite passes.

### File List
- `src/train.py` (updated)
- `src/data/agch_datamodule.py` (updated)
- `src/models/agch_module.py` (updated)
- `src/utils/__init__.py` (updated)
- `configs/train.yaml` (updated)
- `configs/model/agch.yaml` (updated)
- `configs/trainer/default.yaml` (updated)
- `configs/callbacks/default.yaml` (updated)
- `configs/logger/tensorboard.yaml` (updated)
- `configs/logger/csv.yaml` (created)
- `configs/extras/default.yaml` (created)
- `tests/test_integration_train.py` (updated)
- `_bmad-output/implementation-artifacts/5-1-end-to-end-system-integration.md` (updated)
- `_bmad-output/implementation-artifacts/sprint-status.yaml` (updated)

### Change Log
- 2026-01-19: Implemented end-to-end training pipeline and integration tests; updated configs and story status.
