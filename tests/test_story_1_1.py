"""Test Story 1.1: Project Structure Sanity Check."""

import subprocess
import sys


def test_import_lightning():
    """AC#5: Verify lightning can be imported."""
    import lightning

    assert lightning.__version__ is not None


def test_import_hydra():
    """AC#5: Verify hydra can be imported."""
    import hydra

    assert hydra.__version__ is not None


def test_import_torch():
    """AC#5: Verify torch can be imported."""
    import torch

    assert torch.__version__ is not None


def test_src_modules_exist():
    """AC#2: Verify src subdirectories are importable."""
    import src
    import src.data
    import src.models
    import src.utils

    assert src is not None
    assert src.data is not None
    assert src.models is not None
    assert src.utils is not None


def test_configs_exist():
    """AC#1: Verify config directory structure exists."""
    from pathlib import Path

    # Get project root (assuming tests are run from project root)
    root = Path(__file__).parent.parent

    assert (root / "configs").is_dir()
    assert (root / "configs" / "train.yaml").is_file()
    assert (root / "configs" / "paths").is_dir()
    assert (root / "configs" / "trainer").is_dir()
    assert (root / "configs" / "model").is_dir()
    assert (root / "configs" / "data").is_dir()
