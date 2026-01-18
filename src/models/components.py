import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImgNet(nn.Module):
    """Image Encoder Network (MLP).

    Architecture: Input(4096) -> FC(hash_code_len) -> Tanh
    Used to project VGG16 fc7 features to hash code space.
    """

    def __init__(self, input_dim: int = 4096, hash_code_len: int = 64):
        super().__init__()
        self.fc = nn.Linear(input_dim, hash_code_len)
        # Activation is handled by AGCHModule (tanh), but including it here for completeness
        # if used standalone. However, AGCHModule logic applies tanh manually.
        # We will follow standard PyTorch practice:
        # Linear + Normalization (if any)
        # Tanh is applied at the output of this module in AGCHModule's forward pass logic usually,
        # but let's encapsulate the projection here.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image features [Batch, 4096]
        Returns:
            Projected features [Batch, hash_code_len] (before activation if handled externally,
            but standard ImgNet usually outputs the features directly).
            In AGCH paper: f^v(x; \theta_v) -> we typically valid output range (-1, 1).
        """
        return self.fc(x)


class TxtNet(nn.Module):
    """Text Encoder Network (Enhanced for sparse BoW input).

    Architecture: Input(K) -> BatchNorm -> FC(4096) -> BatchNorm -> LeakyReLU -> FC(hash_code_len)

    針對稀疏 BoW 特徵的增強：
    1. 輸入 BatchNorm 穩定稀疏輸入分佈
    2. 使用 LeakyReLU 避免稀疏輸入導致的死神經元
    3. 中間層 BatchNorm 改善梯度流
    """

    def __init__(self, input_dim: int, hash_code_len: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # 輸入層 BatchNorm 處理稀疏輸入
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU 處理稀疏激活
            nn.Linear(4096, hash_code_len),
        )

        # 初始化：對第一個 Linear 層使用較大的初始值以對抗稀疏輸入
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用 Xavier 初始化並略微放大
                nn.init.xavier_uniform_(m.weight, gain=2.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Text features [Batch, Input_Dim] (sparse BoW/Tag)
        Returns:
            Projected features [Batch, hash_code_len]
        """
        return self.net(x)


class SpectralGraphConv(nn.Module):
    """Spectral Graph Convolution Layer.

    Formula (when normalized=True):
        H(l) = sigma( D^(-1/2) * A_hat * D^(-1/2) * H(l-1) * W(l) )
    where A_hat = A + I (self-loop), and D is computed using |A_hat| to handle negative similarities.

    Formula (when normalized=False):
        H(l) = sigma( A * H(l-1) * W(l) )  [Simplified propagation]
    """

    def __init__(self, in_features: int, out_features: int, normalize: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.normalize = normalize
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # Standard Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input_feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_feat: Node features [Batch, in_features]
            adj: Adjacency matrix (Affinity Matrix S) [Batch, Batch]
        """
        # 1. Linear Transformation: H * W
        support = torch.mm(input_feat, self.weight)

        if self.normalize:
            # 2. Laplacian Normalization (Paper Eq. 8 & 15)
            # Add self-loops for stability: A_hat = A + I
            A_hat = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)

            # Compute Degree Matrix using absolute values to handle negative similarities
            # D_ii = sum_j |A_ij|
            D_hat = torch.sum(torch.abs(A_hat), dim=1)

            # Inverse Square Root: D^(-1/2)
            D_inv_sqrt = torch.pow(D_hat, -0.5)
            D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
            D_mat = torch.diag(D_inv_sqrt)

            # Symmetric Normalization: D^(-1/2) * A_hat * D^(-1/2)
            norm_adj = torch.mm(torch.mm(D_mat, A_hat), D_mat)

            # 3. Propagate: norm_adj * support
            output = torch.mm(norm_adj, support)
        else:
            # Simplified Propagation (Original implementation): A * H * W
            output = torch.mm(adj, support)

        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + f", normalize={self.normalize})"
        )


class BiGCN(nn.Module):
    """Bi-Layer GCN Network.

    Architecture: Input(c) -> GCN(hidden_dim) -> ReLU -> GCN(c)

    Args:
        hash_code_len: Dimension of input/output hash codes (c).
        hidden_dim: Hidden layer dimension (default 4096).
        normalize: Whether to use Laplacian normalization in GCN layers.
    """

    def __init__(self, hash_code_len: int, hidden_dim: int = 4096, normalize: bool = True):
        super().__init__()
        self.gc1 = SpectralGraphConv(hash_code_len, hidden_dim, normalize=normalize)
        self.relu = nn.ReLU(inplace=True)
        self.gc2 = SpectralGraphConv(hidden_dim, hash_code_len, normalize=normalize)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input hash codes [Batch, hash_code_len]
            adj: Similarity matrix [Batch, Batch]
        """
        x = self.gc1(x, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        return x
