#!/usr/bin/env python3
"""AGCH Training Entry Point.

This is the main entry point for training the AGCH model.
It uses Hydra for configuration management and PyTorch Lightning for training.
"""

import logging
from typing import List, Optional

import rootutils

# Setup root directory before importing project modules
root = rootutils.setup_root(__file__, indicator=".git", pythonpath=True)

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger

from src.utils import log_hyperparameters

log = logging.getLogger(__name__)


def _instantiate_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []
    if not cfg.get("callbacks"):
        return callbacks

    for _, cb_conf in cfg.callbacks.items():
        if isinstance(cb_conf, DictConfig) and cb_conf.get("_target_"):
            callbacks.append(instantiate(cb_conf))
    return callbacks


def _instantiate_logger(cfg: DictConfig) -> Optional[Logger]:
    if not cfg.get("logger"):
        return None

    if isinstance(cfg.logger, DictConfig) and cfg.logger.get("_target_"):
        return instantiate(cfg.logger)

    return None


def train(cfg: DictConfig) -> Optional[float]:
    """Train and optionally test the model based on configuration."""
    if cfg.get("seed") is not None:
        log.info(f"Setting seed: {cfg.seed}")
        seed_everything(cfg.seed, workers=True)

    if cfg.get("extras", {}).get("print_config", True):
        log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)

    callbacks = _instantiate_callbacks(cfg)
    logger = _instantiate_logger(cfg)

    trainer: Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    log_hyperparameters(cfg, trainer, model, datamodule)

    if cfg.get("train", True):
        trainer.fit(model, datamodule=datamodule)

    if cfg.get("test", False):
        trainer.test(model, datamodule=datamodule, ckpt_path="best")

    return None


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main training function.

    Args:
        cfg: Hydra configuration composed from configs/train.yaml

    Returns:
        Optional metric value for hyperparameter optimization
    """
    try:
        return train(cfg)
    except Exception as e:
        log.exception(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
