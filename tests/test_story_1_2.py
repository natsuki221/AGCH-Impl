"""Test Story 1.2: Paths & Logging Infrastructure."""

from pathlib import Path


def test_logger_config_exists():
    """AC#3: Verify TensorBoard logger config exists."""
    root = Path(__file__).parent.parent
    assert (root / "configs" / "logger" / "tensorboard.yaml").is_file()


def test_hydra_config_exists():
    """AC#1: Verify Hydra output directory config exists."""
    root = Path(__file__).parent.parent
    assert (root / "configs" / "hydra" / "default.yaml").is_file()


def test_callbacks_config_exists():
    """Verify callbacks config exists."""
    root = Path(__file__).parent.parent
    assert (root / "configs" / "callbacks" / "default.yaml").is_file()


def test_paths_config_has_required_keys():
    """AC#2: Verify paths config has required keys."""
    import yaml

    root = Path(__file__).parent.parent
    with open(root / "configs" / "paths" / "default.yaml") as f:
        config = yaml.safe_load(f)

    # Check for required path keys
    assert "data_dir" in config
    assert "log_dir" in config
    assert "root_dir" in config


def test_train_yaml_includes_all_defaults():
    """AC#4: Verify train.yaml includes all config defaults."""
    root = Path(__file__).parent.parent
    train_yaml = (root / "configs" / "train.yaml").read_text()

    # Check for required defaults
    assert "paths: default" in train_yaml
    assert "logger: tensorboard" in train_yaml
    assert "callbacks: default" in train_yaml
    assert "hydra: default" in train_yaml
