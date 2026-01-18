import torch
from typing import Optional


def calculate_hamming_dist_matrix(
    query_code: torch.Tensor, retrieval_code: torch.Tensor
) -> torch.Tensor:
    """
    Compute Hamming distance matrix between query and retrieval codes using vectorized operations.

    Args:
        query_code: [N_q, K] binary codes
        retrieval_code: [N_db, K] binary codes

    Returns:
        dist_matrix: [N_q, N_db] Hamming distances
    """
    if query_code.dim() != 2 or retrieval_code.dim() != 2:
        raise ValueError("query_code and retrieval_code must be 2D tensors")

    if query_code.size(1) != retrieval_code.size(1):
        raise ValueError("query_code and retrieval_code must share the same code length")

    code_len = query_code.size(1)
    dist_matrix = 0.5 * (code_len - query_code @ retrieval_code.T)
    return dist_matrix


def calculate_mAP(
    dist_matrix: torch.Tensor,
    query_labels: torch.Tensor,
    retrieval_labels: torch.Tensor,
    top_k: Optional[int] = None,
) -> float:
    """
    Calculate Mean Average Precision (mAP).

    Args:
        dist_matrix: [N_q, N_db] Hamming distances
        query_labels: [N_q, L] or [N_q] labels
        retrieval_labels: [N_db, L] or [N_db] labels
        top_k: Optional k limit for MAP@k

    Returns:
        mAP: scalar float
    """
    if dist_matrix.dim() != 2:
        raise ValueError("dist_matrix must be 2D")

    n_query, n_db = dist_matrix.shape
    if top_k is None or top_k > n_db:
        top_k = n_db

    sorted_idx = torch.argsort(dist_matrix, dim=1)
    sorted_idx = sorted_idx[:, :top_k]

    relevance = _compute_relevance(query_labels, retrieval_labels)
    relevance = relevance.gather(1, sorted_idx)

    cumsum_rel = torch.cumsum(relevance, dim=1)
    rank = torch.arange(1, top_k + 1, device=dist_matrix.device, dtype=dist_matrix.dtype)
    precision_at_k = cumsum_rel / rank

    rel_counts = relevance.sum(dim=1)
    ap = (precision_at_k * relevance).sum(dim=1) / torch.clamp(rel_counts, min=1.0)
    ap = torch.where(rel_counts > 0, ap, torch.zeros_like(ap))

    return ap.mean().item()


def calculate_precision_at_k(
    dist_matrix: torch.Tensor,
    query_labels: torch.Tensor,
    retrieval_labels: torch.Tensor,
    k: int,
) -> float:
    """
    Calculate Precision@k.

    Args:
        dist_matrix: [N_q, N_db] Hamming distances
        query_labels: [N_q, L] labels
        retrieval_labels: [N_db, L] labels
        k: rank position to evaluate

    Returns:
        P@k: scalar float
    """
    if dist_matrix.dim() != 2:
        raise ValueError("dist_matrix must be 2D")

    n_query, n_db = dist_matrix.shape
    k = min(k, n_db)

    sorted_idx = torch.argsort(dist_matrix, dim=1)[:, :k]
    relevance = _compute_relevance(query_labels, retrieval_labels)
    relevance = relevance.gather(1, sorted_idx)

    precision = relevance.sum(dim=1) / float(k)
    return precision.mean().item()


def _compute_relevance(query_labels: torch.Tensor, retrieval_labels: torch.Tensor) -> torch.Tensor:
    """
    Compute relevance matrix between queries and database items.

    Relevance is defined as having at least one shared label (dot product > 0).

    Args:
        query_labels: [N_q, L]
        retrieval_labels: [N_db, L]

    Returns:
        relevance: [N_q, N_db] boolean-like tensor (1.0 for relevant, 0.0 for irrelevant)
    """
    if query_labels.dim() == 1:
        query_labels = query_labels.view(-1, 1)

    if retrieval_labels.dim() == 1:
        retrieval_labels = retrieval_labels.view(-1, 1)

    if query_labels.size(1) != retrieval_labels.size(1):
        raise ValueError("query_labels and retrieval_labels must share the same label dimension")

    relevance = (query_labels.float() @ retrieval_labels.float().T) > 0
    return relevance.to(dtype=dist_matrix_dtype(query_labels, retrieval_labels))


def dist_matrix_dtype(query_labels: torch.Tensor, retrieval_labels: torch.Tensor) -> torch.dtype:
    """Helper to determine dtype for relevance matrix."""
    if query_labels.is_floating_point() or retrieval_labels.is_floating_point():
        return torch.float32
    return torch.float32
