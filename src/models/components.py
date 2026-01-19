import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImgNet(nn.Module):
    """
    Image Encoder Network (Stabilized MLP).
    Architecture: Input(4096) -> FC(4096) -> BN -> ReLU -> Dropout -> FC(hash_code_len)
    Fix: Added BatchNorm1d to prevent gradient collapse in deep unsupervised hashing.
    """

    def __init__(
        self,
        input_dim: int = 4096,
        hash_code_len: int = 64,
        hidden_dim: int = 4096,
        dropout: float = 0.5,
    ):
        super(ImgNet, self).__init__()

        # 1. Hidden Layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # 2. Critical Fix: Batch Normalization
        # BN is essential before activation to keep feature distribution stable
        self.bn = nn.BatchNorm1d(hidden_dim)

        # 3. Activation & Regularization
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

        # 4. Hash Layer (Projection)
        self.fc2 = nn.Linear(hidden_dim, hash_code_len)

        self._init_weights()

    def _init_weights(self):
        # Use Kaiming Init for ReLU layers, Xavier for output
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn(x)  # <--- Stabilize distribution
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TxtNet(nn.Module):
    """
    Text Encoder Network (Stabilized MLP).
    Architecture: Input(BoW) -> FC(4096) -> BN -> ReLU -> Dropout -> FC(hash_code_len)
    """

    def __init__(
        self,
        input_dim: int,
        hash_code_len: int = 64,
        hidden_dim: int = 4096,
        dropout: float = 0.5,
    ):
        super(TxtNet, self).__init__()

        # 1. Hidden Layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # 2. Critical Fix: Batch Normalization
        self.bn = nn.BatchNorm1d(hidden_dim)

        # 3. Activation & Regularization
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

        # 4. Hash Layer
        self.fc2 = nn.Linear(hidden_dim, hash_code_len)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn(x)  # <--- Stabilize distribution
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SpectralGraphConv(nn.Module):
    """
    Spectral Graph Convolution Layer.
    Implementation of y = D^-1/2 * A * D^-1/2 * x * W
    """

    def __init__(self, in_features, out_features, normalize=True):
        super(SpectralGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.normalize = normalize

        nn.init.xavier_uniform_(self.weight)

    def forward(self, input_feat, adj):
        support = torch.mm(input_feat, self.weight)

        if self.normalize:
            # Robust Laplacian Normalization
            # Adding self-loop: A_hat = A + I
            A_hat = adj + torch.eye(adj.size(0), device=adj.device)

            # Absolute Degree Calculation to handle potential negative similarities safely
            D_hat = torch.sum(torch.abs(A_hat), dim=1)
            D_inv_sqrt = torch.pow(D_hat, -0.5)
            D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
            D_mat = torch.diag(D_inv_sqrt)

            # D^-1/2 * A * D^-1/2
            norm_adj = torch.mm(torch.mm(D_mat, A_hat), D_mat)
            output = torch.mm(norm_adj, support)
        else:
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
    """
    Bi-Layer GCN Network.
    Architecture: Input(hash_len) -> GCN(4096) -> ReLU -> GCN(hash_len)
    """

    def __init__(self, hash_code_len: int, hidden_dim: int = 4096, normalize: bool = True):
        super(BiGCN, self).__init__()
        self.gc1 = SpectralGraphConv(hash_code_len, hidden_dim, normalize=normalize)
        self.relu = nn.ReLU(inplace=True)
        self.gc2 = SpectralGraphConv(hidden_dim, hash_code_len, normalize=normalize)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.gc1(x, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        return x
