import pytest
import torch
import pytorch_lightning as L
from typing import List, Tuple

# Try to import AGCHModule, if it doesn't exist, we skip some tests or fail
try:
    from src.models.agch_module import AGCHModule
except ImportError:
    AGCHModule = None


@pytest.fixture
def module_kwargs():
    """Default init arguments for AGCHModule."""
    return {
        "hash_code_len": 32,
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 1.0,
        "learning_rate": 1e-4,
    }


def test_agch_module_exists():
    """AC 1: AGCHModule class must exist and be importable."""
    assert AGCHModule is not None, "AGCHModule class not found check src/models/agch_module.py"


@pytest.mark.skipif(AGCHModule is None, reason="AGCHModule not implemented")
def test_lightning_inheritance(module_kwargs):
    """AC 1: Must inherit from L.LightningModule."""
    model = AGCHModule(**module_kwargs)
    assert isinstance(model, L.LightningModule)


@pytest.mark.skipif(AGCHModule is None, reason="AGCHModule not implemented")
def test_manual_optimization_flag(module_kwargs):
    """AC 1: Must set automatic_optimization = False."""
    model = AGCHModule(**module_kwargs)
    assert model.automatic_optimization is False, "Manual optimization must be enabled"


@pytest.mark.skipif(AGCHModule is None, reason="AGCHModule not implemented")
def test_submodules_structure(module_kwargs):
    """AC 2: Must define placeholders for encoders, gcn, and hash_layer."""
    model = AGCHModule(**module_kwargs)
    assert hasattr(model, "img_enc"), "Missing img_enc"
    assert hasattr(model, "txt_enc"), "Missing txt_enc"
    assert hasattr(model, "gcn"), "Missing gcn"
    assert hasattr(model, "hash_layer"), "Missing hash_layer"

    # Check if they are modules (even Identity placeholders)
    assert isinstance(model.img_enc, torch.nn.Module)
    assert isinstance(model.txt_enc, torch.nn.Module)


@pytest.mark.skipif(AGCHModule is None, reason="AGCHModule not implemented")
def test_configure_optimizers_structure(module_kwargs):
    """AC 4: configure_optimizers must return valid optimizer config."""
    model = AGCHModule(**module_kwargs)
    opt_config = model.configure_optimizers()

    # Expecting at least 2 optimizers for Alternating Optimization (or a list of them)
    # The AC says "returning a list of optimizers"
    if isinstance(opt_config, (list, tuple)):
        optimizers = opt_config
    elif isinstance(opt_config, dict):
        optimizers = opt_config.get("optimizer", [])
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
    else:
        optimizers = [opt_config]

    for opt in optimizers:
        assert isinstance(opt, torch.optim.Optimizer)


@pytest.mark.skipif(AGCHModule is None, reason="AGCHModule not implemented")
def test_forward_shape(module_kwargs):
    """AC 3, 5: forward must accept image/text inputs and return hash codes."""
    model = AGCHModule(**module_kwargs)
    batch_size = 8

    # Mock inputs matching AlexNet features (4096) and Text BoW (1386)
    img_input = torch.randn(batch_size, 4096)
    txt_input = torch.randn(batch_size, 1386)

    # If the model is skeletal, forward might raise NotImplementedError or return zeros
    # Ideally it should return [batch, hash_code_len]
    try:
        output = model(img_input=img_input, txt_input=txt_input)
        assert output.shape == (batch_size, module_kwargs["hash_code_len"])
    except NotImplementedError:
        pytest.fail("Forward method not implemented")
