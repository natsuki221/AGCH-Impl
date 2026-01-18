import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml


def _write_dummy_hdf5(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)

    images_path = data_dir / "images.h5"
    texts_path = data_dir / "texts.h5"

    rng = np.random.default_rng(42)
    images = rng.standard_normal((32, 4096)).astype(np.float32)
    texts = rng.standard_normal((32, 1386)).astype(np.float32)
    labels = rng.integers(0, 2, size=(32, 24)).astype(np.float32)

    with h5py.File(images_path, "w") as f:
        f.create_dataset("features", data=images)
        f.create_dataset("labels", data=labels)

    with h5py.File(texts_path, "w") as f:
        f.create_dataset("features", data=texts)


@pytest.mark.integration
def test_smoke_fast_dev_run(tmp_path):
    """
    AC 5: Smoke Test (Auto-Integration)
    Run the training script with fast_dev_run=True to ensure it executes without errors (Crash-free).
    """
    _write_dummy_hdf5(tmp_path)

    cmd = [
        sys.executable,
        "src/train.py",
        "trainer.fast_dev_run=True",
        f"hydra.run.dir={tmp_path}",
        f"data.data_dir={tmp_path}",
        "data.batch_size=4",
        "data.num_workers=0",
        "logger=csv",  # Minimal logger
        "extras.print_config=False",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Training script failed with stderr:\n{result.stderr}"

    # Verify it actually initialized the modules (Current skeleton prints "Training logic will be implemented", so this assertion might fail until implemented)
    # Once implemented, we expect "Datamodule initialized" or similar implicitly via logs,
    # but for now, the return code 0 is the baseline.
    # To make it fail *before* implementation if it's just a skeleton:
    # assert "AGCHModule" in result.stderr or "AGCHModule" in result.stdout # Pytorch lightning prints model summary


@pytest.mark.integration
def test_artifacts_generation(tmp_path):
    """
    AC 6: Artifact Verification
    Run a short training loop (1 epoch, 1 batch) and verify artifacts (.ckpt, config.yaml) are created.
    """
    _write_dummy_hdf5(tmp_path)

    cmd = [
        sys.executable,
        "src/train.py",
        "trainer.max_epochs=1",
        "trainer.limit_train_batches=1",
        "trainer.limit_val_batches=1",
        "trainer.limit_test_batches=0",
        "trainer.accelerator=cpu",  # Force CPU for CI/test stability
        "logger=csv",
        f"hydra.run.dir={tmp_path}",
        f"data.data_dir={tmp_path}",
        "data.batch_size=4",
        "data.num_workers=0",
        "extras.print_config=False",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Training failed:\n{result.stderr}"

    # Verify Config
    assert (tmp_path / ".hydra" / "config.yaml").exists()
    assert (tmp_path / ".hydra" / "hydra.yaml").exists()

    # Verify Checkpoint (This requires ModelCheckpoint callback to be active and triggered)
    # Note: If no validation runs, and monitor is val/mAP, it might not save "best" k.
    # But usually last.ckpt or similar is saved if configured.
    # We will assert that *some* .ckpt exists in checkpoints/ folder.
    checkpoints_dir = tmp_path / "checkpoints"
    # Currently code skeleton won't create 'checkpoints', so this SHOULD FAIL (RED).
    assert checkpoints_dir.exists(), "Checkpoints directory not created"
    ckpts = list(checkpoints_dir.glob("*.ckpt"))
    assert len(ckpts) > 0, "No checkpoint files found (.ckpt)"


@pytest.mark.integration
def test_cli_overrides(tmp_path):
    """
    AC 7: Override Testing
    Verify CLI overrides are propagated to the saved configuration.
    Example: model.alpha=0.5
    """
    _write_dummy_hdf5(tmp_path)

    cmd = [
        sys.executable,
        "src/train.py",
        "trainer.fast_dev_run=True",
        "model.alpha=0.505",  # Unique value
        f"hydra.run.dir={tmp_path}",
        f"data.data_dir={tmp_path}",
        "data.batch_size=4",
        "data.num_workers=0",
        "logger=csv",
        "extras.print_config=False",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"run failed: {result.stderr}"

    # Read saved config
    config_path = tmp_path / ".hydra" / "config.yaml"
    assert config_path.exists()

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Validating the override
    # Note: skeleton might not even require model.alpha, but if we override it via hydra, it usually ends up in cfg object
    # If the skeleton uses @hydra.main, it saves the config.
    # The CURRENT skeleton DOES use @hydra.main, so this specific test MIGHT PASS even if logic is missing,
    # unless 'model' group doesn't exist yet in config.
    # If model is not in defaults, this arg will fail with "Could not override 'model.alpha'".
    # This proves the config structure exists.
    assert "model" in cfg, "Model config group not found"
    assert cfg["model"]["alpha"] == 0.505
