# ATDD Checklist - Story 5.1: End-to-End System Integration

**Status**: RED (Tests Failing)
**Story ID**: 5.1
**Primary Test Level**: Integration / E2E (System Level)

## Acceptance Criteria & Test Coverage

| ID | Criteria | Test Level | Test File | Status |
|----|----------|------------|-----------|--------|
| 1 | Configuration Loading | Integration | `tests/test_integration_train.py` | 🔴 |
| 2 | Pipeline Initialization (Data, Model, Trainer) | Integration | `tests/test_integration_train.py` | 🔴 |
| 3 | Execution (trainer.fit error-free) | Integration | `tests/test_integration_train.py` | 🔴 |
| 5 | Smoke Test (fast_dev_run) | Integration | `test_smoke_fast_dev_run` | 🔴 |
| 6 | Artifact Verification (.ckpt, config) | Integration | `test_artifacts_generation` | 🔴 |
| 7 | Override Testing (CLI propagation) | Integration | `test_cli_overrides` | 🔴 |

## Test Infrastructure Created

*   [x] **Test File**: `tests/test_integration_train.py`
    *   Uses `subprocess` to verify CLI entry point.
    *   Uses `tmp_path` fixture for isolated outputs.
*   [ ] **Fixtures**: Relies on Hydra configuration fixtures (to be verified).

## Implementation Checklist

### 1. Main Training Script (`src/train.py`)

- [ ] Import `AGCHModule` and `AGCHDataModule`.
- [ ] Use `hydra.utils.instantiate` to create objects from `cfg`.
- [ ] Setup loggers (WandB/TensorBoard) based on `cfg.logger`.
- [ ] Setup callbacks (ModelCheckpoint, RichProgressBar).
- [ ] Call `trainer.fit(model=model, datamodule=datamodule)`.
- [ ] Call `trainer.test()` if configured.
- [ ] Ensure `L.seed_everything` is called with `workers=True`.

### 2. Configuration (`configs/`)

- [ ] Verify `configs/train.yaml` has correct defaults.
- [ ] Verify `configs/model/default.yaml` points to `src.models.agch_module.AGCHModule`.
- [ ] Verify `configs/data/default.yaml` points to `src.data.agch_datamodule.AGCHDataModule`.

### 3. Verification

- [ ] Run `pytest tests/test_integration_train.py`.
- [ ] Ensure `checkpoints/` folder is created.
- [ ] Ensure `config.yaml` reflects overrides.

## Red-Green-Refactor Workflow

1.  **RED**: `pytest tests/test_integration_train.py` fails (Checkpoints missing, config structure issues).
2.  **GREEN**: Implement `src/train.py` to substantiate the pipeline.
3.  **REFACTOR**: Clean up imports, ensure type safety.

## Execution Commands

```bash
# Run integration tests
pytest tests/test_integration_train.py

# Debug run
pytest tests/test_integration_train.py -s
```
