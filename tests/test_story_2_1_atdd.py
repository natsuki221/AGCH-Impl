"""ATDD Tests for Story 2.1: HDF5 Data Module & Caching.

These tests are in RED phase - they define expected behavior BEFORE implementation.
Run these tests to verify implementation as you develop AGCHDataModule.

RED → GREEN → REFACTOR cycle:
1. Run tests (all should FAIL - RED phase)
2. Implement AGCHDataModule to make tests pass (GREEN phase)
3. Refactor with confidence (REFACTOR phase)
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch


# =============================================================================
# ATDD Test Suite for Story 2.1: HDF5 Data Module
# =============================================================================


class TestAGCHDataModuleSetup:
    """AC: Given HDF5 feature files, When setup() called, Then verify file existence."""

    def test_setup_verifies_hdf5_files_exist(self, tmp_path: Path):
        """AC#1: setup() should verify file existence."""
        # GIVEN: AGCHDataModule with path to non-existent files
        from src.data.agch_datamodule import AGCHDataModule

        datamodule = AGCHDataModule(
            data_dir=tmp_path / "nonexistent",
            batch_size=32,
        )

        # WHEN/THEN: setup() should raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="HDF5"):
            datamodule.setup(stage="fit")

    def test_setup_loads_data_into_memory_when_cache_enabled(self, tmp_path: Path, mock_hdf5_files):
        """AC#1: load data into memory if cache_in_memory=True (NFR-C2)."""
        # GIVEN: Valid HDF5 files and cache_in_memory=True
        from src.data.agch_datamodule import AGCHDataModule

        datamodule = AGCHDataModule(
            data_dir=tmp_path,
            batch_size=32,
            cache_in_memory=True,
        )

        # WHEN: setup() is called
        datamodule.setup(stage="fit")

        # THEN: Data should be loaded into memory (tensors, not file handles)
        assert datamodule._cached_images is not None
        assert isinstance(datamodule._cached_images, torch.Tensor)
        assert datamodule._cached_texts is not None
        assert isinstance(datamodule._cached_texts, torch.Tensor)


class TestAGCHDataModuleDataSplitting:
    """AC: Data should be split into train/retrieval/query sets."""

    def test_setup_creates_train_retrieval_query_splits(self, tmp_path: Path, mock_hdf5_files):
        """AC#2: split data into train/retrieval/query sets as per standard protocol."""
        # GIVEN: Valid HDF5 files
        from src.data.agch_datamodule import AGCHDataModule

        datamodule = AGCHDataModule(
            data_dir=tmp_path,
            batch_size=32,
        )

        # WHEN: setup() is called
        datamodule.setup(stage="fit")

        # THEN: Three splits should exist
        assert hasattr(datamodule, "train_dataset")
        assert hasattr(datamodule, "retrieval_dataset")
        assert hasattr(datamodule, "query_dataset")

        # AND: Splits should have correct sizes (standard protocol)
        # MIRFlickr-25K: 10K train, 10K retrieval, 2K query
        assert len(datamodule.train_dataset) > 0
        assert len(datamodule.retrieval_dataset) > 0
        assert len(datamodule.query_dataset) > 0


class TestAGCHDataLoaderOutput:
    """AC: train_dataloader() should return (image, text, index, label) tuples."""

    def test_train_dataloader_returns_correct_tuple_format(self, tmp_path: Path, mock_hdf5_files):
        """AC#3: train_dataloader() returns (image, text, index, label) tuples."""
        # GIVEN: Configured AGCHDataModule
        from src.data.agch_datamodule import AGCHDataModule

        datamodule = AGCHDataModule(
            data_dir=tmp_path,
            batch_size=32,
        )
        datamodule.setup(stage="fit")

        # WHEN: Getting a batch from train_dataloader
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))

        # THEN: Batch should be a tuple of 4 elements
        assert isinstance(batch, (tuple, list))
        assert len(batch) == 4, f"Expected 4 elements (image, text, index, label), got {len(batch)}"

        image, text, index, label = batch

        # AND: Each element should have correct type and shape
        assert isinstance(image, torch.Tensor), "image should be Tensor"
        assert isinstance(text, torch.Tensor), "text should be Tensor"
        assert isinstance(index, torch.Tensor), "index should be Tensor"
        assert isinstance(label, torch.Tensor), "label should be Tensor"

    def test_batch_size_is_respected(self, tmp_path: Path, mock_hdf5_files):
        """Verify batch_size parameter works correctly."""
        # GIVEN: AGCHDataModule with batch_size=16
        from src.data.agch_datamodule import AGCHDataModule

        batch_size = 16
        datamodule = AGCHDataModule(
            data_dir=tmp_path,
            batch_size=batch_size,
        )
        datamodule.setup(stage="fit")

        # WHEN: Getting a batch
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))

        # THEN: First tensor should have correct batch dimension
        assert batch[0].shape[0] == batch_size


# =============================================================================
# Fixtures for ATDD Tests (Mock HDF5 Data)
# =============================================================================


@pytest.fixture
def mock_hdf5_files(tmp_path: Path):
    """Create mock HDF5 files for testing.

    This fixture creates minimal HDF5 files that match expected format.
    Replace with real HDF5 creation when implementing AGCHDataModule.
    """
    import h5py
    import numpy as np

    # Create mock image features file
    image_file = tmp_path / "images.h5"
    with h5py.File(image_file, "w") as f:
        # 100 samples, 4096 feature dimensions (VGG-19 style)
        f.create_dataset("features", data=np.random.randn(100, 4096).astype(np.float32))
        f.create_dataset("labels", data=np.random.randint(0, 2, (100, 24)).astype(np.float32))

    # Create mock text features file
    text_file = tmp_path / "texts.h5"
    with h5py.File(text_file, "w") as f:
        # 100 samples, 1386 feature dimensions (BoW style)
        f.create_dataset("features", data=np.random.randn(100, 1386).astype(np.float32))

    return {"images": image_file, "texts": text_file}
