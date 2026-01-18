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
import torch.nn.functional as F
from torch.optim import Adam

__all__ = ["AGCHModule"]


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
        # Note: img_enc and txt_enc are currently Identity placeholders
        # with no parameters. They will be replaced in Story 3.2+.
        opt_f = Adam(
            list(self.img_enc.parameters())
            + list(self.txt_enc.parameters())
            + list(self._img_proj.parameters())
            + list(self._txt_proj.parameters()),
            lr=lr,
        )

        # Optimizer for Hash-related components (Phase 2: Update B, Fix F)
        # Note: gcn is currently Identity placeholder with no parameters.
        opt_b = Adam(
            list(self.gcn.parameters()) + list(self.hash_layer.parameters()),
            lr=lr,
        )

        return [opt_f, opt_b]

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> Dict[str, Any]:
        """Execute one training step with manual optimization.

        Alternating optimization:
        - Even step: Update F (feature networks), fix B
        - Odd step: Update B (hash/GCN), fix F

        Note: For RTX 5080, consider wrapping forward/loss in
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) when enabled.

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

        # Compute hash codes for each modality and fused representation
        B_v, B_t, B_h = self._compute_hash_codes(X, T)

        # Determine phase (prefer global_step if available)
        step_idx = getattr(self.trainer, "global_step", batch_idx)
        update_f_phase = (step_idx % 2) == 0

        if update_f_phase:
            # Phase 1: Update F, Fix B (detach B-related targets)
            B_h_fixed = B_h.detach()
            S = self._compute_similarity_matrix(B_h_fixed)
            B_g = self._compute_b_g(B_h_fixed)

            loss_rec = self.compute_loss_rec(B_h, S.detach())
            loss_str = self.compute_loss_str(B_g.detach(), B_h)
            loss_cm = self.compute_loss_cm(B_v, B_t, B_h)

            loss = (
                self.hparams.alpha * loss_rec
                + self.hparams.beta * loss_str
                + self.hparams.gamma * loss_cm
            )

            opt_f.zero_grad()
            self.manual_backward(loss)
            opt_f.step()

            self.log("train/loss_f", loss, prog_bar=True)
        else:
            # Phase 2: Update B, Fix F (detach feature networks)
            B_h_fixed = B_h.detach()
            B_v_fixed = B_v.detach()
            B_t_fixed = B_t.detach()

            S = self._compute_similarity_matrix(B_h_fixed)
            B_g = self._compute_b_g(B_h_fixed)

            loss_rec = self.compute_loss_rec(B_g, S)
            loss_str = self.compute_loss_str(B_g, B_h_fixed)
            loss_cm = self.compute_loss_cm(B_v_fixed, B_t_fixed, B_h_fixed)

            loss = (
                self.hparams.alpha * loss_rec
                + self.hparams.beta * loss_str
                + self.hparams.gamma * loss_cm
            )

            opt_b.zero_grad()
            self.manual_backward(loss)
            opt_b.step()

            self.log("train/loss_b", loss, prog_bar=True)

        # Log shared metrics
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/loss_rec", loss_rec)
        self.log("train/loss_str", loss_str)
        self.log("train/loss_cm", loss_cm)

        return {"loss": loss}

    def _compute_hash_codes(
        self, X: torch.Tensor, T: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute modality-specific and fused hash codes.

        Returns:
            B_v, B_t, B_h: Hash codes for image, text, and fused representations.
        """
        F_I = self.img_enc(X)
        F_T = self.txt_enc(T)

        O_I = self._img_proj(F_I)
        O_T = self._txt_proj(F_T)

        B_v = torch.tanh(O_I)
        B_t = torch.tanh(O_T)
        B_h = torch.tanh(O_I + O_T)

        return B_v, B_t, B_h

    def _compute_b_g(self, B_h: torch.Tensor) -> torch.Tensor:
        """Compute GCN-based hash codes from fused representation."""
        H_g = self.gcn(B_h)
        B_g = torch.tanh(self.hash_layer(H_g))
        return B_g

    def _compute_similarity_matrix(self, Z: torch.Tensor, rho: float = 1.0) -> torch.Tensor:
        """Compute aggregated similarity matrix S = C ⊙ D (Hadamard product)."""
        Z_norm = F.normalize(Z, dim=1)
        C = Z_norm @ Z_norm.T
        D = torch.exp(-torch.cdist(Z, Z, p=2) / rho)
        S = C * D
        S = 2.0 * S - 1.0
        return S

    def compute_loss_rec(self, B: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """L1: Reconstruction loss in Hamming space (placeholder)."""
        B_sim = (B @ B.T) / B.shape[1]
        return torch.mean((B_sim - S) ** 2)

    def compute_loss_str(self, B_g: torch.Tensor, B_h: torch.Tensor) -> torch.Tensor:
        """L2: Structure loss for GCN neighborhood consistency (placeholder)."""
        return torch.mean((B_g - B_h) ** 2)

    def compute_loss_cm(
        self, B_v: torch.Tensor, B_t: torch.Tensor, B_h: torch.Tensor
    ) -> torch.Tensor:
        """L3: Cross-modal alignment loss (placeholder)."""
        loss_v = torch.mean((B_v - B_h) ** 2)
        loss_t = torch.mean((B_t - B_h) ** 2)
        return 0.5 * (loss_v + loss_t)
