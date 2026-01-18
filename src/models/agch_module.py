"""AGCH Lightning Module for Cross-Modal Hashing.

This module implements the AGCH (Asymmetric Guided Compatible Hashing) model
as a PyTorch Lightning module with manual optimization support for
alternating updates between Feature Network (F) and Binary Codes (B).

Reference: AGCH Paper (TMM 2022)
"""

from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as L
import torch
import torch.nn as nn
from torch.optim import Adam


class AGCHModule(L.LightningModule):
    """AGCH Model with Manual Optimization for Alternating Updates.

    This module serves as the core training wrapper for the AGCH model.
    It uses manual optimization (`automatic_optimization=False`) to handle
    the alternating optimization strategy required by AGCH:
    - Phase 1: Fix B (Binary Codes), Update F (Feature Networks)
    - Phase 2: Fix F, Update B (Discrete Optimization)

    Attributes:
        img_enc: Image encoder network (placeholder).
        txt_enc: Text encoder network (placeholder).
        gcn: Graph Convolutional Network for neighborhood aggregation (placeholder).
        hash_layer: Hashing layer to generate binary codes (placeholder).
        hash_code_len: Length of the binary hash codes.
    """

    def __init__(
        self,
        hash_code_len: int = 32,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        learning_rate: float = 1e-4,
        img_feature_dim: int = 4096,
        txt_feature_dim: int = 1386,
    ) -> None:
        """Initialize AGCH Module.

        Args:
            hash_code_len: Length of the output hash codes (default: 32).
            alpha: Weight for reconstruction loss L1.
            beta: Weight for structure loss L2.
            gamma: Weight for cross-modal loss L3.
            learning_rate: Learning rate for optimizers.
            img_feature_dim: Dimension of image features (AlexNet fc7 = 4096).
            txt_feature_dim: Dimension of text features (BoW/PCA = 1386).
        """
        super().__init__()

        # CRITICAL: Enable manual optimization for alternating updates
        self.automatic_optimization = False

        # Save hyperparameters for logging and checkpointing
        self.save_hyperparameters()

        # --- Sub-module Placeholders (AC 2) ---
        # These will be replaced with actual implementations in subsequent stories
        self.img_enc = nn.Identity()  # Placeholder for Image Encoder
        self.txt_enc = nn.Identity()  # Placeholder for Text Encoder
        self.gcn = nn.Identity()  # Placeholder for GCN module
        self.hash_layer = nn.Linear(img_feature_dim, hash_code_len)  # Simple linear for skeleton

        # Internal projection layers (to be replaced)
        self._img_proj = nn.Linear(img_feature_dim, hash_code_len)
        self._txt_proj = nn.Linear(txt_feature_dim, hash_code_len)

    def forward(
        self,
        img_input: Optional[torch.Tensor] = None,
        txt_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate hash codes from image and/or text inputs.

        Uses paper notation:
        - X: Image features [Batch, img_feature_dim]
        - T: Text features [Batch, txt_feature_dim]
        - B: Binary hash codes [Batch, hash_code_len]

        Args:
            img_input: Image features X [Batch, 4096].
            txt_input: Text features T [Batch, 1386].

        Returns:
            B: Hash codes [Batch, hash_code_len] in range (-1, 1) via tanh.
        """
        # Paper notation: X for image, T for text
        X = img_input
        T = txt_input

        if X is not None and T is not None:
            # Fusion mode: combine both modalities
            F_I = self._img_proj(X)  # [Batch, hash_code_len]
            F_T = self._txt_proj(T)  # [Batch, hash_code_len]
            O_H = F_I + F_T  # Simple fusion (placeholder)
        elif X is not None:
            O_H = self._img_proj(X)
        elif T is not None:
            O_H = self._txt_proj(T)
        else:
            raise ValueError("At least one of img_input or txt_input must be provided")

        # Generate continuous hash codes via tanh (discrete in inference)
        B = torch.tanh(O_H)  # [Batch, hash_code_len]

        return B

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Configure optimizers for alternating optimization.

        Returns two optimizers for the alternating update strategy:
        - opt_f: Optimizer for feature network parameters (img_enc, txt_enc)
        - opt_b: Optimizer for hash-related parameters (gcn, hash_layer)

        Returns:
            List of optimizers for manual optimization.
        """
        lr = self.hparams.learning_rate

        # Optimizer for Feature Networks (Phase 1: Update F, Fix B)
        opt_f = Adam(
            list(self.img_enc.parameters())
            + list(self.txt_enc.parameters())
            + list(self._img_proj.parameters())
            + list(self._txt_proj.parameters()),
            lr=lr,
        )

        # Optimizer for Hash-related components (Phase 2: Update B, Fix F)
        opt_b = Adam(
            list(self.gcn.parameters()) + list(self.hash_layer.parameters()),
            lr=lr,
        )

        return [opt_f, opt_b]

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> Dict[str, Any]:
        """Execute one training step with manual optimization.

        This is a placeholder implementation. The actual alternating
        optimization logic will be implemented in Story 3.2.

        Args:
            batch: Tuple of (image, text, index, label) tensors.
            batch_idx: Index of the current batch.

        Returns:
            Dictionary with loss values for logging.
        """
        # Unpack batch (expected from AGCHDataModule)
        X, T, idx, L = batch  # Paper notation: X=image, T=text, L=label

        # Get optimizers
        opt_f, opt_b = self.optimizers()

        # Placeholder: Simple forward pass and dummy loss
        B = self.forward(img_input=X, txt_input=T)

        # Dummy loss for skeleton (will be replaced with actual AGCH losses)
        loss = B.mean()

        # Manual backward (required for manual optimization)
        opt_f.zero_grad()
        self.manual_backward(loss)
        opt_f.step()

        # Log the loss
        self.log("train/loss", loss, prog_bar=True)

        return {"loss": loss}
