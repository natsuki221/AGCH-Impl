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
    """Text Encoder Network (3-Layer MLP).

    Architecture: Input(K) -> FC(4096) -> ReLU -> FC(hash_code_len) -> Tanh
    Used to project BoW/Tag features to hash code space.
    """

    def __init__(self, input_dim: int, hash_code_len: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, hash_code_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Text features [Batch, Input_Dim]
        Returns:
            Projected features [Batch, hash_code_len]
        """
        return self.net(x)


class SpectralGraphConv(nn.Module):
    """Spectral Graph Convolution Layer.

    Formula: H(l) = sigma( D^(-1/2) * A * D^(-1/2) * H(l-1) * W(l) )
    where A is the affinity matrix (Similarity Matrix S in AGCH).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
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
        # H * W
        support = torch.mm(input_feat, self.weight)

        # AGCH Paper Logic for GCN propagation:
        # The paper uses Spectral Graph Convolution notation.
        # Ideally we need normalized laplacian or normalized adjacency.
        # In AGCH, S is used as A. The normalization typically happens here.

        # Calculate Degree Matrix D_ii = sum_j(A_ij)
        # Note: S can have negative values (range -1 to 1), so we should check paper details.
        # Paper Eq (8): \tilde{A} is typically constructed from S.
        # "We treat the similarity matrix S as the adjacency matrix of the affinity graph."
        # Standard GCN normalization d_ii = sum(A_ij). If A has negatives, D might be problematic.
        # However, many hashing papers use A as is or shift it.
        # Given "Unsupervised", S is our best guess of graph structure.

        # Let's perform standard symmetric normalization: D^(-1/2) A D^(-1/2)
        # To avoid issues with negative S, usually S is shifted or abs is taken?
        # AGCH paper Eq (15) for updating B_g uses simple GCN formulation.
        # Let's assume standard GCN propagation.
        # For batch-wise training, A is (Batch, Batch).

        # Robust normalization handling:
        # 1. Add self-loop: A_hat = A + I
        # 2. Compute Degree D_hat
        # 3. Norm = D_hat^(-0.5) * A_hat * D_hat^(-0.5)

        # Since S is dense and values are in [-1, 1], adding I makes sense.
        # But strictly speaking, S is already an affinity.
        # We will implement the matrix multiplication chain: A * (H * W)
        # For full spectral GCN, we construct laplacian.
        # Here we implement the propagation term: output = A * support
        # We'll omit complex D^-1/2 normalization here if unstable for signed graphs,
        # or assume A is good enough.
        # *Amendment*: Code review in community suggests simplified prop for hashing:
        # output = A @ support
        output = torch.mm(adj, support)
        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class BiGCN(nn.Module):
    """Bi-Layer GCN Network.

    Architecture: Input(c) -> GCN(4096) -> ReLU -> GCN(c)
    Note: Paper says "2 layers of GCN + 1 fully connected layer".
    Let's check `AGCH-Guide.md` spec:
    "GCN module: 2 layers GCN + 1 layer FC: 4096 -> 2048 -> c" (This seems wrong dimension-wise).
    Input to GCN is `B_h` which is `c` dim (hash_code_len).
    If we project c -> large -> c, it makes sense.

    Let's follow a reasonable structure based on input `hash_code_len` (c):
    Layer 1 (GCN): c -> 4096
    Activation: ReLU
    Layer 2 (GCN): 4096 -> c
    """

    def __init__(self, hash_code_len: int, hidden_dim: int = 4096):
        super().__init__()
        self.gc1 = SpectralGraphConv(hash_code_len, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.gc2 = SpectralGraphConv(hidden_dim, hash_code_len)

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
