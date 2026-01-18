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

from src.utils.metrics import calculate_hamming_dist_matrix, calculate_mAP

__all__ = ["AGCHModule"]


from src.models.components import BiGCN, ImgNet, TxtNet


class AGCHModule(L.LightningModule):
    """AGCH Model with Manual Optimization for Alternating Updates.

    ... (docstring preserved)
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
        rho: float = 1.0,
        gamma_v: float = 1.0,
        gamma_t: float = 1.0,
        gcn_normalize: bool = True,
    ) -> None:
        """Initialize AGCH Module."""
        super().__init__()

        # CRITICAL: Enable manual optimization for alternating updates
        self.automatic_optimization = False

        # Save hyperparameters
        self.save_hyperparameters()

        # --- Actual Components ---
        # Image Encoder: MLP (FC -> Tanh implicit in logic, but here we use ImgNet)
        # Note: ImgNet in components.py is Linear only, activation handled in forward
        self.img_enc = ImgNet(input_dim=img_feature_dim, hash_code_len=hash_code_len)

        # Text Encoder: 3-Layer MLP
        self.txt_enc = TxtNet(input_dim=txt_feature_dim, hash_code_len=hash_code_len)

        # GCN Module: Bi-Layer GCN with configurable normalization
        self.gcn = BiGCN(hash_code_len=hash_code_len, hidden_dim=4096, normalize=gcn_normalize)

        # Hashing layer is part of GCN in paper (last layer), or separate?
        # In components.py BiGCN outputs hash_Code_len directly.
        # So independent 'hash_layer' might be redundant or used for specific projection.
        # Paper says: "GCN output -> Fully Connected -> Hash"
        # Our BiGCN includes that structure.
        # We will keep hash_layer as typically it's the final output layer if not in GCN.
        # But BiGCN returns sizing `hash_code_len`.
        # Let's keep `hash_layer` as Identity or remove it if BiGCN covers it.
        # Reference line 289 in original code: B_g = torch.tanh(self.hash_layer(H_g))
        # If BiGCN outputs [Batch, hash_code_len], then hash_layer should be Identity
        # OR BiGCN should output intermediate and this layer projects.
        # Let's assume BiGCN does the heavy lifting to 'c'.
        self.hash_layer = nn.Identity()

        # Internal projection layers (REMOVED: logic moved to Encoders)
        # self._img_proj = nn.Linear(img_feature_dim, hash_code_len)
        # self._txt_proj = nn.Linear(txt_feature_dim, hash_code_len)

    def forward(
        self,
        img_input: Optional[torch.Tensor] = None,
        txt_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate hash codes from image and/or text inputs."""
        # Paper notation: X for image, T for text
        X = img_input
        T = txt_input

        if X is not None and T is not None:
            # Fusion mode: combine both modalities
            # ImgNet/TxtNet outputs are already projected to hash_code_len
            F_I = self.img_enc(X)  # [Batch, hash_code_len]
            F_T = self.txt_enc(T)  # [Batch, hash_code_len]
            O_H = F_I + F_T  # Simple fusion
        elif X is not None:
            O_H = self.img_enc(X)
        elif T is not None:
            O_H = self.txt_enc(T)
        else:
            raise ValueError("At least one of img_input or txt_input must be provided")

        # Generate continuous hash codes via tanh (discrete in inference)
        B = torch.tanh(O_H)  # [Batch, hash_code_len]

        return B

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Configure optimizers for alternating optimization."""
        lr = self.hparams.learning_rate

        # Optimizer for Feature Networks (Phase 1: Update F, Fix B)
        # Components: ImgNet + TxtNet
        opt_f = Adam(
            list(self.img_enc.parameters()) + list(self.txt_enc.parameters()),
            lr=lr,
        )

        # Optimizer for Hash-related components (Phase 2: Update B, Fix F)
        # Components: GCN (contains hash layers internally now)
        # Note: self.hash_layer is effectively Identity/unused if BiGCN outputs final dim
        opt_b = Adam(
            list(self.gcn.parameters()),
            lr=lr,
        )

        return [opt_f, opt_b]

    # ... training_step ... (reuse existing logic, logic is generic enough)

    def _compute_hash_codes(
        self, X: torch.Tensor, T: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute modality-specific and fused hash codes."""
        # Updated to use new Encoders
        # They project directly to Hash Code Space (pre-activation)
        O_I = self.img_enc(X)
        O_T = self.txt_enc(T)

        B_v = torch.tanh(O_I)
        B_t = torch.tanh(O_T)
        B_h = torch.tanh(O_I + O_T)

        return B_v, B_t, B_h

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
            # AGCH Guide: Compute S from raw features X, T
            S = self._compute_similarity_matrix(X, T)
            B_g = self._compute_b_g(B_h_fixed, S.detach())

            loss_rec = self.compute_loss_rec(B_h, S.detach())
            loss_str = self.compute_loss_str(B_g.detach(), B_h)
            # Cross-modal loss: pred = fused code B_h, targets = B_v, B_t
            loss_cm = self.compute_loss_cm(B_h, B_v, B_t)

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

            # AGCH Guide: Compute S from raw features X, T
            S = self._compute_similarity_matrix(X, T)
            # Pass S as adj
            B_g = self._compute_b_g(B_h_fixed, S)

            loss_rec = self.compute_loss_rec(B_g, S)
            loss_str = self.compute_loss_str(B_g, B_h_fixed)
            # Cross-modal loss for Phase2: pred = B_g (depends on hash params), targets = B_v_fixed, B_t_fixed
            loss_cm = self.compute_loss_cm(B_g, B_v_fixed, B_t_fixed)

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

    def on_validation_epoch_start(self) -> None:
        self._val_codes: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        """Collect validation codes and labels for retrieval evaluation."""
        with torch.no_grad():
            X, T, idx, L = batch
            B = self.forward(img_input=X, txt_input=T)
            self._val_codes.append(B.detach().cpu())
            self._val_labels.append(L.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation mAP using collected codes/labels."""
        if not self._val_codes or not self._val_labels:
            return

        with torch.no_grad():
            codes = torch.cat(self._val_codes, dim=0)
            labels = torch.cat(self._val_labels, dim=0)

            dist_matrix = calculate_hamming_dist_matrix(codes, codes)
            dist_matrix.fill_diagonal_(float("inf"))

            map_value = calculate_mAP(dist_matrix, labels, labels)
            self.log("val/mAP", map_value, prog_bar=True)

    def _compute_hash_codes(
        self, X: torch.Tensor, T: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute modality-specific and fused hash codes.

        Returns:
            B_v, B_t, B_h: Hash codes for image, text, and fused representations.
        """
        F_I = self.img_enc(X)
        F_T = self.txt_enc(T)

        O_I = F_I  # Already projected
        O_T = F_T  # Already projected

        B_v = torch.tanh(O_I)
        B_t = torch.tanh(O_T)
        B_h = torch.tanh(O_I + O_T)

        return B_v, B_t, B_h

    def _compute_b_g(self, B_h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Compute GCN-based hash codes from fused representation.

        Args:
            B_h: Fused hash codes [Batch, hash_code_len]
            adj: Adjacency matrix (Similarity Matrix S) [Batch, Batch]
        """
        # BiGCN takes (x, adj)
        B_g = self.gcn(B_h, adj)
        return torch.tanh(B_g)

    def _compute_similarity_matrix(
        self, X: torch.Tensor, T: torch.Tensor, rho: Optional[float] = None
    ) -> torch.Tensor:
        """Compute aggregated similarity matrix S = C * D.

        Improved version per expert feedback:
        1. Normalize X and T (L2).
        2. Apply weights gamma_v and gamma_t.
        3. Concatenate: Z = [gamma_v * norm(X), gamma_t * norm(T)]
        4. Re-normalize Z (L2) to ensure Cosine C is in [-1, 1].
        5. Compute Cosine Similarity C and Gaussian-kernel Euclidean D.
        6. Fuse: S = alpha * C + beta * D (no 2*S-1 scaling).
        """
        if rho is None:
            rho = float(self.hparams.get("rho", 1.0))

        gv = float(self.hparams.get("gamma_v", 1.0))
        gt = float(self.hparams.get("gamma_t", 1.0))
        alpha = float(self.hparams.get("alpha", 1.0))
        beta = float(self.hparams.get("beta", 1.0))

        # 1. Normalize modality features (L2)
        X_norm = F.normalize(X, p=2, dim=1)
        T_norm = F.normalize(T, p=2, dim=1)

        # 2. Apply weights and concatenate
        Z = torch.cat([gv * X_norm, gt * T_norm], dim=1)

        # 3. Re-normalize Z to ensure Cosine Similarity C is strictly in [-1, 1]
        Z = F.normalize(Z, p=2, dim=1)

        # 4. Compute Cosine Similarity (C)
        # Since Z is now L2-normalized, mm(Z, Z^T) gives Cosine directly
        C = torch.mm(Z, Z.t())

        # 5. Compute Euclidean Distance based Similarity (D) using Gaussian Kernel
        # D_ij = exp(-||Z_i - Z_j||^2 / (2 * rho^2))
        dist_sq = torch.cdist(Z, Z, p=2).pow(2)
        D = torch.exp(-dist_sq / (2.0 * rho * rho))

        # 6. Fuse (weighted sum instead of Hadamard to stabilize gradients)
        # Removed: S = 2.0 * S - 1.0 (causes range issues)
        S = alpha * C + beta * D

        return S

    def compute_loss_rec(self, B: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """L1: Reconstruction loss in Hamming space (placeholder)."""
        B_sim = (B @ B.T) / B.shape[1]
        return torch.mean((B_sim - S) ** 2)

    def compute_loss_str(self, B_g: torch.Tensor, B_h: torch.Tensor) -> torch.Tensor:
        """L2: Structure loss for GCN neighborhood consistency (placeholder)."""
        return torch.mean((B_g - B_h) ** 2)

    def compute_loss_cm(
        self, B_pred: torch.Tensor, B_a: torch.Tensor, B_b: torch.Tensor
    ) -> torch.Tensor:
        """L3: Cross-modal alignment loss (placeholder).

        Args:
            B_pred: predicted hash codes (e.g., fused `B_h` or GCN-based `B_g`).
            B_a: target modality A (e.g., `B_v`).
            B_b: target modality B (e.g., `B_t`).
        """
        loss_a = torch.mean((B_a - B_pred) ** 2)
        loss_b = torch.mean((B_b - B_pred) ** 2)
        return 0.5 * (loss_a + loss_b)
