import os

import pytest
import torch
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader, TensorDataset

from src.models.agch_module import AGCHModule


class TinyDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_batches: int,
        img_dim: int,
        txt_dim: int,
        label_dim: int,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.img_dim = img_dim
        self.txt_dim = txt_dim
        self.label_dim = label_dim

    def setup(self, stage=None):
        total = self.batch_size * self.num_batches
        self._train = TensorDataset(
            torch.randn(total, self.img_dim),
            torch.randn(total, self.txt_dim),
            torch.arange(total),
            torch.randint(0, 2, (total, self.label_dim)).float(),
        )
        self._val = self._train

    def train_dataloader(self):
        return DataLoader(self._train, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self._val, batch_size=self.batch_size, shuffle=False, num_workers=0)


class LossHistoryCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        elif torch.is_tensor(outputs):
            loss = outputs
        else:
            return

        self.losses.append(loss.detach().cpu())


@pytest.mark.integration
def test_training_determinism(tmp_path):
    """
    Verify that training is fully deterministic under fixed seed (Loss and Weights).
    Story 4.2 Acceptance Criteria 1, 2.
    """

    def run_short_training(seed: int):
        # Set seed
        seed_everything(seed, workers=True)

        # Create config
        cfg = OmegaConf.create(
            {
                "model": {
                    "alpha": 0.1,
                    "beta": 0.1,
                    "gamma": 0.1,
                    "hash_code_len": 12,
                    "img_feature_dim": 12,
                    "txt_feature_dim": 6,
                },
                "data": {"batch_size": 4, "num_batches": 100, "label_dim": 4},
                "trainer": {
                    "max_epochs": 1,
                    "accelerator": "cpu",
                    "devices": 1,
                    "deterministic": True,
                    "logger": False,
                    "enable_checkpointing": False,
                    "limit_train_batches": 100,
                    "limit_val_batches": 1,
                },
            }
        )

        datamodule = TinyDataModule(
            batch_size=cfg.data.batch_size,
            num_batches=cfg.data.num_batches,
            img_dim=cfg.model.img_feature_dim,
            txt_dim=cfg.model.txt_feature_dim,
            label_dim=cfg.data.label_dim,
        )
        model = AGCHModule(**cfg.model)

        loss_callback = LossHistoryCallback()
        trainer = Trainer(default_root_dir=str(tmp_path), callbacks=[loss_callback], **cfg.trainer)
        trainer.fit(model, datamodule=datamodule)

        loss_history = torch.stack(loss_callback.losses)
        state_dict = model.state_dict()
        map_value = trainer.callback_metrics.get("val/mAP")

        return loss_history, state_dict, map_value

    # Run twice with same seed
    loss_1, weights_1, map_1 = run_short_training(seed=42)
    loss_2, weights_2, map_2 = run_short_training(seed=42)

    # Assert Loss equality for first 100 iterations
    assert loss_1.numel() == 100
    assert torch.allclose(loss_1, loss_2, atol=1e-6), f"Loss mismatch: {loss_1} != {loss_2}"

    # Assert Weights equality
    for key in weights_1:
        assert torch.equal(weights_1[key], weights_2[key]), f"Weight mismatch in validation: {key}"

    # Assert mAP equality
    assert map_1 is not None and map_2 is not None
    assert torch.allclose(map_1, map_2, atol=1e-6)


@pytest.mark.unit
def test_initialization_determinism():
    seed_everything(42, workers=True)
    model_a = AGCHModule(hash_code_len=12, img_feature_dim=8, txt_feature_dim=6)

    seed_everything(42, workers=True)
    model_b = AGCHModule(hash_code_len=12, img_feature_dim=8, txt_feature_dim=6)

    for key in model_a.state_dict():
        assert torch.equal(model_a.state_dict()[key], model_b.state_dict()[key])


@pytest.mark.unit
def test_trainer_configuration_determinism():
    """
    Verify that default trainer config enables deterministic mode.
    Story 4.2 Acceptance Criteria 6.
    """
    config_path = "configs/trainer"
    if not os.path.exists(config_path):
        pytest.skip("Trainer config directory not found")

    config_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "configs", "trainer", "default.yaml")
    )
    if not os.path.exists(config_file):
        pytest.skip("Trainer config file not found")

    cfg = OmegaConf.load(config_file)

    assert cfg.get("deterministic") is True, "Trainer config must have 'deterministic: True'"
    assert cfg.get("benchmark") is False, "Trainer config must have 'benchmark: False' for reproducibility"
