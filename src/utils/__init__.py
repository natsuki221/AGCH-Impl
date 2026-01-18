"""Utility helpers for training and logging."""

from typing import Any, Dict

from omegaconf import OmegaConf


def log_hyperparameters(cfg: Any, trainer: Any, model: Any, datamodule: Any) -> None:
	"""Log hyperparameters to the configured logger(s)."""
	if trainer is None or getattr(trainer, "logger", None) is None:
		return

	hparams: Dict[str, Any] = {
		"cfg": OmegaConf.to_container(cfg, resolve=True),
		"model_params": sum(p.numel() for p in model.parameters()),
	}

	logger = trainer.logger
	if isinstance(logger, list):
		for item in logger:
			item.log_hyperparams(hparams)
	else:
		logger.log_hyperparams(hparams)


__all__ = ["log_hyperparameters"]
