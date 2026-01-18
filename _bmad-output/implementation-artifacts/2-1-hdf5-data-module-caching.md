# Story 2.1: HDF5 Data Module & Caching

**Status:** ready-for-dev

## Story

As a Researcher,
I want a `LightningDataModule` that loads pre-extracted features from HDF5 files and supports in-memory caching,
So that I can maximize GPU utilization during training by eliminating disk I/O bottlenecks.

## Acceptance Criteria

1. **Given** Existing HDF5 feature files / **When** `AGCHDataModule.setup()` is called / **Then** It should verify file existence and load data into memory if `cache_in_memory=True` (NFR-C2).
2. **And** It should split data into train/retrieval/query sets as per standard protocol.
3. **And** The `train_dataloader()` should return batches of `(image, text, index, label)` tuples.

## Technical Requirements

### Class Structure
```python
# src/data/agch_datamodule.py
class AGCHDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path,
        batch_size: int = 128,
        num_workers: int = 8,
        pin_memory: bool = True,
        cache_in_memory: bool = True,  # NFR-C2
    ): ...
    
    def setup(self, stage: str) -> None:
        # Verify HDF5 files exist
        # Load data into memory if cache_in_memory
        # Create train/retrieval/query splits
    
    def train_dataloader(self) -> DataLoader:
        # Return DataLoader yielding (image, text, index, label)
```

### HDF5 File Format
- `images.h5`: `features` dataset [N, 4096], `labels` dataset [N, 24]
- `texts.h5`: `features` dataset [N, 1386]

### Data Split Protocol (MIRFlickr-25K)
- Train: 10,000 samples
- Retrieval: 10,000 samples  
- Query: 2,000 samples

## ATDD Status

**RED Phase**: Failing tests created at `tests/test_story_2_1_atdd.py`

Run tests to verify implementation:
```bash
pytest tests/test_story_2_1_atdd.py -v
```

## Implementation Checklist

- [ ] Create `src/data/agch_datamodule.py`
- [ ] Implement `__init__` with all parameters
- [ ] Implement `setup()` with file verification
- [ ] Implement `cache_in_memory` loading logic
- [ ] Implement train/retrieval/query splitting
- [ ] Implement `train_dataloader()` returning correct tuple format
- [ ] Implement `val_dataloader()` and `test_dataloader()`
- [ ] Run ATDD tests → All pass (GREEN phase)
