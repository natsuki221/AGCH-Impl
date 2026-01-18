"""AGCHDataModule for loading pre-extracted features from HDF5 files.

This module implements LightningDataModule for the AGCH cross-modal retrieval task.
It supports in-memory caching for maximum GPU utilization (NFR-C2).
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import h5py
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class AGCHDataset(Dataset):
    """Dataset for AGCH features.

    Holds image features, text features, indices, and labels.
    """

    def __init__(
        self,
        images: torch.Tensor,
        texts: torch.Tensor,
        labels: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
    ):
        """Initialize AGCHDataset.

        Args:
            images: Image feature tensor [N, image_dim]
            texts: Text feature tensor [N, text_dim]
            labels: Multi-label tensor [N, num_classes]
            indices: Optional index tensor [N]
        """
        self.images = images
        self.texts = texts
        self.labels = labels
        self.indices = indices if indices is not None else torch.arange(len(images))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (image, text, index, label) tuple."""
        return (
            self.images[idx],
            self.texts[idx],
            self.indices[idx],
            self.labels[idx],
        )


class AGCHDataModule(LightningDataModule):
    """LightningDataModule for AGCH cross-modal retrieval.

    Loads pre-extracted image and text features from HDF5 files.
    Supports in-memory caching for optimal GPU utilization (NFR-C2).

    Expected HDF5 file format:
        - images.h5: 'features' [N, 4096], 'labels' [N, 24]
        - texts.h5: 'features' [N, 1386]

    Data split protocol (MIRFlickr-25K):
        - Train: 10,000 samples
        - Retrieval: 10,000 samples
        - Query: 2,000 samples
    """

    # Standard MIRFlickr-25K split sizes
    TRAIN_SIZE = 10000
    RETRIEVAL_SIZE = 10000
    QUERY_SIZE = 2000

    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 128,
        num_workers: int = 8,
        pin_memory: bool = True,
        cache_in_memory: bool = True,
        image_file: str = "images.h5",
        text_file: str = "texts.h5",
    ):
        """Initialize AGCHDataModule.

        Args:
            data_dir: Directory containing HDF5 files
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for dataloaders
            pin_memory: Whether to pin memory for GPU transfer
            cache_in_memory: Whether to load all data into memory (NFR-C2)
            image_file: Name of HDF5 file containing image features
            text_file: Name of HDF5 file containing text features
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.cache_in_memory = cache_in_memory
        self.image_file = image_file
        self.text_file = text_file

        # Cached data (when cache_in_memory=True)
        self._cached_images: Optional[torch.Tensor] = None
        self._cached_texts: Optional[torch.Tensor] = None
        self._cached_labels: Optional[torch.Tensor] = None

        # Dataset splits
        self.train_dataset: Optional[AGCHDataset] = None
        self.retrieval_dataset: Optional[AGCHDataset] = None
        self.query_dataset: Optional[AGCHDataset] = None

    def _verify_files_exist(self) -> None:
        """Verify that required HDF5 files exist.

        Raises:
            FileNotFoundError: If any required HDF5 file is missing
        """
        image_path = self.data_dir / self.image_file
        text_path = self.data_dir / self.text_file

        if not self.data_dir.exists():
            raise FileNotFoundError(f"HDF5 data directory not found: {self.data_dir}")
        if not image_path.exists():
            raise FileNotFoundError(f"HDF5 image file not found: {image_path}")
        if not text_path.exists():
            raise FileNotFoundError(f"HDF5 text file not found: {text_path}")

    def _load_hdf5_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load data from HDF5 files.

        Returns:
            Tuple of (images, texts, labels) as numpy arrays
        """
        image_path = self.data_dir / self.image_file
        text_path = self.data_dir / self.text_file

        with h5py.File(image_path, "r") as f:
            images = f["features"][:]
            labels = f["labels"][:]

        with h5py.File(text_path, "r") as f:
            texts = f["features"][:]

        return images, texts, labels

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for each stage.

        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict')

        Raises:
            FileNotFoundError: If required HDF5 files are missing
        """
        # Verify files exist
        self._verify_files_exist()

        # Load data
        images, texts, labels = self._load_hdf5_data()

        # Convert to tensors
        images_t = torch.from_numpy(images).float()
        texts_t = torch.from_numpy(texts).float()
        labels_t = torch.from_numpy(labels).float()

        # Cache in memory if requested (NFR-C2)
        if self.cache_in_memory:
            self._cached_images = images_t
            self._cached_texts = texts_t
            self._cached_labels = labels_t

        # Total samples
        n_total = len(images_t)

        # Calculate split indices
        # Standard protocol: first TRAIN_SIZE for train, next RETRIEVAL_SIZE for retrieval, last QUERY_SIZE for query
        train_end = min(self.TRAIN_SIZE, n_total)
        retrieval_end = min(train_end + self.RETRIEVAL_SIZE, n_total)
        query_end = min(retrieval_end + self.QUERY_SIZE, n_total)

        # If dataset is smaller than expected, use proportional splits
        if n_total < self.TRAIN_SIZE + self.RETRIEVAL_SIZE + self.QUERY_SIZE:
            # Use 45% train, 45% retrieval, 10% query for small datasets
            train_end = int(n_total * 0.45)
            retrieval_end = int(n_total * 0.90)
            query_end = n_total

        # Create splits
        train_indices = torch.arange(0, train_end)
        retrieval_indices = torch.arange(train_end, retrieval_end)
        query_indices = torch.arange(retrieval_end, query_end)

        # Create datasets
        self.train_dataset = AGCHDataset(
            images=images_t[:train_end],
            texts=texts_t[:train_end],
            labels=labels_t[:train_end],
            indices=train_indices,
        )

        self.retrieval_dataset = AGCHDataset(
            images=images_t[train_end:retrieval_end],
            texts=texts_t[train_end:retrieval_end],
            labels=labels_t[train_end:retrieval_end],
            indices=retrieval_indices,
        )

        self.query_dataset = AGCHDataset(
            images=images_t[retrieval_end:query_end],
            texts=texts_t[retrieval_end:query_end],
            labels=labels_t[retrieval_end:query_end],
            indices=query_indices,
        )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation (query) dataloader."""
        return DataLoader(
            self.query_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test (retrieval) dataloader."""
        return DataLoader(
            self.retrieval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
