# AGCH-Impl Test Suite

## Overview

This test suite uses **pytest** for testing the AGCH (Aggregation-based Graph Convolutional Hashing) implementation.

## Directory Structure

```
tests/
├── conftest.py          # Shared fixtures (device, seeds, sample data)
├── test_story_1_1.py    # Story 1.1: Project structure tests
├── test_story_1_2.py    # Story 1.2: Config & logging tests
└── README.md            # This file
```

## Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_story_1_1.py

# Run tests matching pattern
pytest -k "import"

# Run with coverage (requires pytest-cov)
pytest --cov=src --cov-report=html
```

## Available Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `project_root` | session | Path to project root |
| `configs_dir` | session | Path to configs directory |
| `device` | session | Best available device (cuda/mps/cpu) |
| `gpu_available` | session | Boolean: GPU available |
| `seed` | function | Fixed seed (42) for reproducibility |
| `set_seed` | function | Auto-sets all random seeds (NFR-R1) |
| `sample_image_features` | function | Tensor [32, 4096] |
| `sample_text_features` | function | Tensor [32, 1386] |
| `sample_labels` | function | Multi-label tensor [32, 24] |
| `sample_binary_codes` | function | Binary codes [32, 32] in {-1, +1} |

## Test Naming Convention

- `test_story_X_Y.py` - Tests for Story X.Y
- `test_<component>.py` - Unit tests for specific component
- `test_integration_<flow>.py` - Integration tests

## Reproducibility (NFR-R1)

Use the `set_seed` fixture for deterministic tests:

```python
def test_deterministic(set_seed, sample_image_features):
    # This test will produce identical results every run
    result = model(sample_image_features)
    assert result.sum() == expected_value
```
