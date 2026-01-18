"""Pytest configuration for AGCH-Impl test suite.

This conftest.py provides shared fixtures for all tests.
"""

import pytest
import torch
from pathlib import Path


# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def configs_dir(project_root: Path) -> Path:
    """Return the configs directory."""
    return project_root / "configs"


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    """Return the data directory (may not exist in CI)."""
    return project_root / "data"


# =============================================================================
# Device Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Return the best available device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def gpu_available() -> bool:
    """Check if GPU is available."""
    return torch.cuda.is_available()


# =============================================================================
# Seed Fixtures (Reproducibility - NFR-R1)
# =============================================================================


@pytest.fixture
def seed() -> int:
    """Return a fixed seed for reproducibility."""
    return 42


@pytest.fixture(autouse=False)
def set_seed(seed: int):
    """Set random seeds for reproducibility.

    Use this fixture explicitly when you need deterministic behavior:
        def test_something(set_seed):
            ...
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_image_features(device: torch.device) -> torch.Tensor:
    """Generate sample image features [batch, feature_dim]."""
    return torch.randn(32, 4096, device=device)


@pytest.fixture
def sample_text_features(device: torch.device) -> torch.Tensor:
    """Generate sample text features [batch, feature_dim]."""
    return torch.randn(32, 1386, device=device)


@pytest.fixture
def sample_labels(device: torch.device) -> torch.Tensor:
    """Generate sample multi-label ground truth [batch, num_classes]."""
    labels = torch.zeros(32, 24, device=device)
    # Each sample has 1-3 positive labels
    for i in range(32):
        num_positive = torch.randint(1, 4, (1,)).item()
        positive_indices = torch.randperm(24)[:num_positive]
        labels[i, positive_indices] = 1.0
    return labels


@pytest.fixture
def sample_binary_codes(device: torch.device) -> torch.Tensor:
    """Generate sample binary hash codes [batch, code_len]."""
    # Binary codes are in {-1, +1}
    return torch.sign(torch.randn(32, 32, device=device))
