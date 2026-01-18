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
        gamma: float = 10.0,
        delta: float = 0.01,  # δ: L2 GCN 結構損失權重
        learning_rate: float = 1e-4,
        lr_img_encoder: float = 1e-4,  # Image Encoder 學習率
        lr_txt_encoder: float = 1e-2,  # Text Encoder 學習率
        lr_gcn: float = 1e-3,  # GCN 學習率
        lr_fusion: float = 1e-2,  # Fusion Module 學習率
        weight_decay: float = 5e-4,  # 權重衰減
        momentum: float = 0.9,  # 動量
        img_feature_dim: int = 4096,
        txt_feature_dim: int = 1386,
        rho: float = 4.0,
        gamma_v: float = 2.0,
        gamma_t: float = 0.3,
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
        """Configure optimizers - 暫時使用聯合優化策略進行測試。

        測試簡化方案：使用單一 Adam 優化器聯合訓練所有參數。
        """
        # 聯合優化：所有參數使用單一 Adam
        opt = torch.optim.Adam(
            self.parameters(),
            lr=1e-3,  # 使用較高的統一學習率
            weight_decay=self.hparams.weight_decay,
        )

        # 返回兩個相同的優化器以兼容現有的 training_step
        return [opt, opt]

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

            # 正則化損失 1：量化損失（強制 codes 接近 ±1）
            loss_quant = torch.mean((B_h.abs() - 1.0) ** 2)

            # 正則化損失 2：平衡損失（防止所有 bits 趨向相同值）
            loss_balance = torch.mean(B_h.mean(dim=0) ** 2)

            # 調整權重：降低 L3 權重，增加 L1 權重
            loss = (
                10.0 * loss_rec  # 增強 L1 以保留樣本間差異
                + self.hparams.delta * loss_str
                + 1.0 * loss_cm  # 降低 L3 權重防止模式崩塌
                + 0.1 * loss_quant
                + 1.0 * loss_balance
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

            # 正則化損失（與 Phase 1 相同）
            loss_quant = torch.mean((B_g.abs() - 1.0) ** 2)
            loss_balance = torch.mean(B_g.mean(dim=0) ** 2)

            # 增強 GCN 訓練權重（δ: 0.01 → 1.0）
            loss = (
                10.0 * loss_rec
                + 1.0 * loss_str  # 增強結構損失權重
                + 1.0 * loss_cm
                + 0.1 * loss_quant
                + 1.0 * loss_balance
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
        # 收集三種 hash codes：Image-only, Text-only, Fused
        self._val_img_codes: List[torch.Tensor] = []
        self._val_txt_codes: List[torch.Tensor] = []
        self._val_fused_codes: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> None:
        """Collect validation codes and labels for retrieval evaluation."""
        with torch.no_grad():
            X, T, idx, L = batch

            # 計算各模態的 hash codes
            B_img = torch.sign(torch.tanh(self.img_enc(X)))  # Image-only
            B_txt = torch.sign(torch.tanh(self.txt_enc(T)))  # Text-only
            B_fused = self.forward(img_input=X, txt_input=T)
            B_fused = torch.sign(B_fused)  # Fused (binarized)

            self._val_img_codes.append(B_img.detach().cpu())
            self._val_txt_codes.append(B_txt.detach().cpu())
            self._val_fused_codes.append(B_fused.detach().cpu())
            self._val_labels.append(L.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation mAP for I→T, T→I, and Fused retrieval."""
        if not self._val_labels:
            return

        with torch.no_grad():
            img_codes = torch.cat(self._val_img_codes, dim=0)
            txt_codes = torch.cat(self._val_txt_codes, dim=0)
            fused_codes = torch.cat(self._val_fused_codes, dim=0)
            labels = torch.cat(self._val_labels, dim=0)

            # I→T: Query=Image, Database=Text (論文主要指標)
            dist_i2t = calculate_hamming_dist_matrix(img_codes, txt_codes)
            dist_i2t.fill_diagonal_(float("inf"))
            map_i2t = calculate_mAP(dist_i2t, labels, labels)

            # T→I: Query=Text, Database=Image
            dist_t2i = calculate_hamming_dist_matrix(txt_codes, img_codes)
            dist_t2i.fill_diagonal_(float("inf"))
            map_t2i = calculate_mAP(dist_t2i, labels, labels)

            # Fused (原有指標，用於監控收斂)
            dist_fused = calculate_hamming_dist_matrix(fused_codes, fused_codes)
            dist_fused.fill_diagonal_(float("inf"))
            map_fused = calculate_mAP(dist_fused, labels, labels)

            # 平均 mAP（論文常用報告方式）
            map_avg = (map_i2t + map_t2i) / 2.0

            self.log("val/mAP_I2T", map_i2t, prog_bar=True)
            self.log("val/mAP_T2I", map_t2i, prog_bar=True)
            self.log("val/mAP_Fused", map_fused)
            self.log("val/mAP", map_avg, prog_bar=True)  # 主監控指標

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
        """Compute aggregated similarity matrix S = C ⊙ D (Hadamard product).

        論文公式 (AGCH-Guide.md)：
        1. 特徵預處理：Z = [γ_v * norm(X), γ_t * norm(T)]
        2. 方向相似度 C_ij = Z_i^T @ Z_j (Cosine Similarity)
        3. 差異相似度 D_ij = exp(-√||Z_i - Z_j||₂ / ρ)
        4. 聚合融合：S = C ⊙ D (Hadamard 乘積)
        5. 量化調節：S = 2S - 1
        """
        if rho is None:
            rho = float(self.hparams.get("rho", 4.0))

        gv = float(self.hparams.get("gamma_v", 2.0))
        gt = float(self.hparams.get("gamma_t", 0.3))

        # 1. L2 正規化模態特徵
        X_norm = F.normalize(X, p=2, dim=1)
        T_norm = F.normalize(T, p=2, dim=1)

        # 2. 加權拼接
        Z = torch.cat([gv * X_norm, gt * T_norm], dim=1)

        # 3. 再次正規化以確保 Cosine Similarity 在 [-1, 1]
        Z = F.normalize(Z, p=2, dim=1)

        # 4. 計算方向相似度 C（Cosine Similarity）
        C = torch.mm(Z, Z.t())

        # 5. 計算差異相似度 D（論文公式：exp(-√dist / ρ)）
        # 注意：論文使用歐式距離的平方根，非平方
        dist = torch.cdist(Z, Z, p=2)  # 歐式距離
        D = torch.exp(-dist / rho)  # 高斯核（使用距離，非距離平方）

        # 6. Hadamard 乘積融合（論文核心：元素級相乘）
        S = C * D

        # 7. 量化調節到 [-1, 1] 範圍（論文要求）
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
        self, B_pred: torch.Tensor, B_a: torch.Tensor, B_b: torch.Tensor
    ) -> torch.Tensor:
        """L3: Cross-modal alignment loss.

        強制跨模態對齊：
        1. B_pred (fused/GCN) 與 B_a (image) 對齊
        2. B_pred (fused/GCN) 與 B_b (text) 對齊
        3. **關鍵**：B_a 與 B_b 直接對齊（同一樣本的 I 和 T 應相似）

        Args:
            B_pred: predicted hash codes (e.g., fused `B_h` or GCN-based `B_g`).
            B_a: target modality A (e.g., `B_v`).
            B_b: target modality B (e.g., `B_t`).
        """
        # 原有：B_h 與 B_v、B_t 分別對齊
        loss_a = torch.mean((B_a - B_pred) ** 2)
        loss_b = torch.mean((B_b - B_pred) ** 2)

        # 關鍵添加：B_v 與 B_t 直接對齊（同一樣本的跨模態對齊）
        loss_cross = torch.mean((B_a - B_b) ** 2)

        # β=1.0 用於原有對齊，額外添加跨模態直接對齊
        return 0.5 * (loss_a + loss_b) + loss_cross
