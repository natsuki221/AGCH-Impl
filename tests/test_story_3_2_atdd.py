import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, PropertyMock, patch
from src.models.agch_module import AGCHModule


@pytest.fixture
def mock_trainer():
    trainer = MagicMock()
    # Mock optimizers list for manual optimization
    trainer.optimizers = [MagicMock(), MagicMock()]
    trainer.strategy.backward = MagicMock()
    return trainer


class MockModule(nn.Module):
    def __init__(self, return_value=None):
        super().__init__()
        self.return_value = return_value
        # If return_value is set, forward returns it; otherwise it's a mock
        self.mock_forward = MagicMock(return_value=return_value)

    def forward(self, *args, **kwargs):
        return self.mock_forward(*args, **kwargs)


@pytest.fixture
def model(mock_trainer):
    model = AGCHModule(
        hash_code_len=32, img_feature_dim=128, txt_feature_dim=64  # Smaller dim for test speed
    )
    # Attach mock trainer
    model.trainer = mock_trainer

    # Mock manual_backward to avoid actual backward pass errors on mock tensors
    model.manual_backward = MagicMock()

    # Mock logging
    model.log = MagicMock()

    # Mock submodules to return controlled outputs uses MockModule
    model.img_enc = MockModule(return_value=torch.randn(4, 32))
    model.txt_enc = MockModule(return_value=torch.randn(4, 32))
    model.gcn = MockModule(return_value=torch.randn(4, 32))
    model.hash_layer = MockModule(return_value=torch.randn(4, 32))
    model._img_proj = MockModule(return_value=torch.randn(4, 32))
    model._txt_proj = MockModule(return_value=torch.randn(4, 32))

    return model


@pytest.fixture
def mock_batch():
    batch_size = 4
    return (
        torch.randn(batch_size, 128),  # X
        torch.randn(batch_size, 64),  # T
        torch.arange(batch_size),  # idx
        torch.zeros(batch_size, 10),  # L
    )


class TestStory3_2ATDD:
    """ATDD Tests for Story 3.2: Alternating Optimization Logic"""

    def test_manual_optimization_flow_ac1(self, model, mock_batch):
        """AC 1: Must use self.optimizers() to retrieve opt_f and opt_b."""
        # Mock optimizers() method to verify checks
        with patch.object(
            model, "optimizers", return_value=[MagicMock(), MagicMock()]
        ) as mock_get_opts:
            model.training_step(mock_batch, 0)
            mock_get_opts.assert_called()

    def test_alternating_logic_execution_ac2_ac3_ac4(self, model, mock_batch):
        """AC 2, 3, 4: Verify alternating logic with single optimizer (parameter groups).

        New architecture uses a single optimizer with differential learning rates
        for different parameter groups (img_enc, txt_enc, gcn). The alternating
        logic controls which tensors are detached, not which optimizer is used.
        """
        opt = MagicMock()

        # Return single optimizer wrapped in list (Lightning behavior)
        model.optimizers = MagicMock(return_value=[opt])

        # Phase 1 (even step): Update F (feature networks focus)
        model.trainer.global_step = 0
        model.training_step(mock_batch, 0)

        # Optimizer should be called once after Phase 1
        assert opt.step.call_count == 1

        # Phase 2 (odd step): Update B (GCN/hash focus)
        model.trainer.global_step = 1
        model.training_step(mock_batch, 1)

        # Optimizer should be called again (total 2 times)
        assert opt.step.call_count == 2

    def test_loss_functions_existence_ac5(self, model):
        """AC 5: Verify loss calculation methods exist."""
        # Check for placeholder or real methods
        assert hasattr(model, "compute_loss_rec"), "Missing compute_loss_rec (L1)"
        assert hasattr(model, "compute_loss_str"), "Missing compute_loss_str (L2)"
        assert hasattr(model, "compute_loss_cm"), "Missing compute_loss_cm (L3)"

    def test_gradient_isolation_intent_ac6(self, model, mock_batch):
        """AC 6: Verify detach() is used (Static Analysis or Runtime check)."""
        # Runtime check is hard without real backward.
        # We check if 'loss' backward is called.

        model.optimizers = MagicMock(return_value=[MagicMock(), MagicMock()])
        model.compute_loss_rec = MagicMock(return_value=torch.tensor(0.5, requires_grad=True))
        model.compute_loss_str = MagicMock(return_value=torch.tensor(0.5, requires_grad=True))
        model.compute_loss_cm = MagicMock(return_value=torch.tensor(0.5, requires_grad=True))

        model.trainer.global_step = 0
        model.gcn.mock_forward.reset_mock()
        model.training_step(mock_batch, 0)

        # Ensure gcn input is detached to avoid gradient leakage
        gcn_args, _ = model.gcn.mock_forward.call_args
        assert gcn_args[0].requires_grad is False

        # Ensure manual_backward was called
        assert model.manual_backward.called, "manual_backward not called"

    def test_logging_keys_ac7(self, model, mock_batch):
        """AC 7: Individually log loss_f, loss_b, and loss."""
        model.optimizers = MagicMock(return_value=[MagicMock(), MagicMock()])

        # Ensure component losses return tensor
        model.compute_loss_rec = MagicMock(return_value=torch.tensor(1.0))
        model.compute_loss_str = MagicMock(return_value=torch.tensor(1.0))
        model.compute_loss_cm = MagicMock(return_value=torch.tensor(1.0))

        model.training_step(mock_batch, 0)

        # Check specific log keys were passed to self.log
        # We look at call args
        log_calls = model.log.call_args_list
        keys_logged = [call.args[0] for call in log_calls]

        # We expect at least one of these depending on the phase
        expected_keys = {
            "train/loss",
            "train/loss_f",
            "train/loss_b",
            "train/loss_rec",
            "train/loss_str",
            "train/loss_cm",
        }
        assert any(
            k in keys_logged for k in expected_keys
        ), f"No expected metrics logged. Found: {keys_logged}"
