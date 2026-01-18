#!/usr/bin/env python3
"""AGCH Training Entry Point.

This is the main entry point for training the AGCH model.
It uses Hydra for configuration management and PyTorch Lightning for training.
"""

import rootutils

# Setup root directory before importing project modules
root = rootutils.setup_root(__file__, indicator=".git", pythonpath=True)

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration composed from configs/train.yaml
    """
    # Import here to ensure rootutils has set up the path
    from lightning import seed_everything

    # Set seed for reproducibility (NFR-R1)
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    # TODO: Implement training logic in Story 3.1
    print(f"AGCH Training initialized with seed: {cfg.get('seed', 'None')}")
    print(f"Task: {cfg.get('task_name', 'train')}")
    print("Training logic will be implemented in Epic 3.")


if __name__ == "__main__":
    main()
