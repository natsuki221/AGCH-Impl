# Story 1.2: Configure Paths & Logging Infrastructure

**Status:** ready-for-dev

## Story

As a Developer,
I want to configure the project's path management and logging system via Hydra,
So that all experiments, checkpoints, and logs are automatically organized in the `logs/` directory without manual intervention.

## Acceptance Criteria

1. **Given** The initialized project structure / **When** I configure `configs/paths/default.yaml` and `configs/logger/tensorboard.yaml` / **Then** Running a dummy experiment should generate logs in `logs/train/runs/YYYY-MM-DD_HH-MM-SS`.
2. **And** The `data_dir` should point to the local `data/` directory.
3. **And** TensorBoard logging should be enabled by default.
4. **And** Hydra configuration files should include comments or schemas for Type Hints to prevent runtime configuration errors.

## Technical Requirements (Dev Agent Guardrails)

### File Structure Requirements

```bash
configs/
├── paths/
│   └── default.yaml     # Root paths: root_dir, data_dir, log_dir, output_dir, work_dir
├── logger/
│   └── tensorboard.yaml # TensorBoard logger configuration
├── callbacks/
│   └── default.yaml     # Default callbacks (ModelCheckpoint, EarlyStopping, etc.)
└── hydra/
    └── default.yaml     # Hydra runtime configuration (output directory pattern)
```

### Key Configuration Details

**paths/default.yaml**:
- `root_dir`: Project root (use rootutils)
- `data_dir`: `${paths.root_dir}/data`
- `log_dir`: `${paths.root_dir}/logs`
- `output_dir`: `${hydra:runtime.output_dir}`
- `work_dir`: `${hydra:runtime.cwd}`

**logger/tensorboard.yaml**:
- `_target_`: `lightning.pytorch.loggers.TensorBoardLogger`
- `save_dir`: `${paths.output_dir}`
- `name`: `tensorboard`
- `default_hp_metric`: false

**hydra/default.yaml**:
- Output directory pattern: `logs/train/runs/${now:%Y-%m-%d_%H-%M-%S}`
- `sweep.dir`: `logs/train/multiruns/${now:%Y-%m-%d_%H-%M-%S}`

### Implementation Steps

1. Update `configs/paths/default.yaml` with full path configuration
2. Create `configs/logger/` directory and `tensorboard.yaml`
3. Create `configs/callbacks/default.yaml` with basic callbacks
4. Create `configs/hydra/default.yaml` with output directory pattern
5. Update `configs/train.yaml` to include new defaults
6. Create a simple test script to verify log generation
7. Add tests to verify configuration

## Dev Notes

- **Architecture Compliance**: All paths should be relative to `${paths.root_dir}` for portability.
- **Hydra Integration**: Use Hydra's interpolation syntax `${...}` for dynamic paths.
- **NFR-R2**: This configuration enables one-click reproduction via config snapshots.

## Verification Plan

### Automated Tests
- Run `python src/train.py --help` to verify Hydra configuration loads correctly
- Check that `logs/` directory is created with correct structure

### Manual Verification
- Execute: `python src/train.py` and verify logs appear in `logs/train/runs/YYYY-MM-DD_HH-MM-SS`
- Launch TensorBoard: `tensorboard --logdir logs/` and verify it loads
