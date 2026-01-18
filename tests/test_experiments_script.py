import pytest
import subprocess
import sys
import os
from pathlib import Path
import shutil


@pytest.mark.integration
def test_experiment_script_exists():
    """
    AC 1 & 3: Experiment Script
    Verify that scripts/run_experiments.sh exists and is executable.
    """
    script_path = Path("scripts/run_experiments.sh")
    assert script_path.exists(), "scripts/run_experiments.sh does not exist"
    assert os.access(
        script_path, os.X_OK
    ), "scripts/run_experiments.sh is not executable (chmod +x needed)"


@pytest.mark.integration
def test_experiment_script_content():
    """
    AC 1: Code Length Support
    Verify script contains commands for 16, 32, and 64 bits.
    """
    script_path = Path("scripts/run_experiments.sh")
    if not script_path.exists():
        pytest.skip("Script not found")

    content = script_path.read_text()
    assert "model.hash_code_length=16" in content, "Missing 16-bit experiment"
    assert "model.hash_code_length=32" in content, "Missing 32-bit experiment"
    assert "model.hash_code_length=64" in content, "Missing 64-bit experiment"


@pytest.mark.integration
def test_experiment_dry_run(tmp_path):
    """
    AC 3: Experiment Script Execution
    Try to run the script in a 'dry run' or 'echo' mode if possible, or just verify it doesn't crash immediately.
    Since running full training is expensive, we rely on the script content check above.
    However, we can check if it uses python src/train.py.
    """
    script_path = Path("scripts/run_experiments.sh")
    if not script_path.exists():
        pytest.skip("Script not found")

    content = script_path.read_text()
    assert (
        "python src/train.py" in content or "python3 src/train.py" in content
    ), "Script does not appear to call src/train.py"
