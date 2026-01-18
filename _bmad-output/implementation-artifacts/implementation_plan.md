# implementation_plan.md - MIRFlickr Data Preparation

## Goal Description
The user needs to run experiments for Story 5.2 but is missing the required dataset (`MIRFlickr-25K`). The project expects pre-extracted features (HDF5 format), but only raw data links (images and annotations) are available.
We will implement a data preparation script `scripts/prepare_mirflickr.py` that:
1.  Downloads the raw MIRFlickr-25K dataset and annotations.
2.  Extracts visual features (4096-dim) using a pre-trained VGG16 model.
3.  Process text tags to generate bag-of-words features (1386-dim, top freq tags).
4.  Process ground truth annotations into multi-hot labels (24 classes).
5.  Saves the result as `data/images.h5` and `data/texts.h5` compatible with `AGCHDataModule`.

## User Review Required
> [!IMPORTANT]
> **Data Processing Time**: Feature extraction for 25,000 images using VGG16 will take time (minutes to hours depending on GPU). The script should be run on a machine with a GPU if possible.
> **Disk Space**: The raw dataset is ~3GB. Extracted features will be another ~1GB.

## Proposed Changes

### Scripts
#### [NEW] [scripts/prepare_mirflickr.py](file:///home/ncu-caic/Documents/Coding/github.com/natsuki221/AGCH-Impl/scripts/prepare_mirflickr.py)
A standalone Python script to handle the full pipeline:
- `download_data()`: Fetches zips from valid mirrors (using user provided links).
- `extract_tags()`: Reads image metadata, computes top 1386 frequent tags, builds BoW vectors.
- `extract_labels()`: Reads annotation files, builds [N, 24] label matrix.
- `extract_visual_features()`: Uses `torchvision.models.vgg16`, removes classifier head, extracts 4096-dim vector for all images.
- `save_hdf5()`: Writes final files.

## Verification Plan

### Automated Tests
- We can add a test mode to the script (e.g., `--test`) that only processes the first 10 images to verify pipeline functionality without waiting for the full dataset.
- `pytest tests/test_data_prep.py` (New test file) to invoke this test mode and check output HDF5 shapes.

### Manual Verification
- Run `python scripts/prepare_mirflickr.py --test` and check if `data/images.h5` and `data/texts.h5` are created.
