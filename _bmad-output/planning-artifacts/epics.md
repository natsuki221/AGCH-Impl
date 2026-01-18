---
stepsCompleted:
  - step-01-validate-prerequisites
  - step-02-design-epics
  - step-03-create-stories
  - step-04-final-validation
inputDocuments:
  - _bmad-output/planning-artifacts/prd.md
  - _bmad-output/planning-artifacts/architecture.md
---

# AGCH-Impl - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for AGCH-Impl, decomposing the requirements from the PRD, UX Design if it exists, and Architecture requirements into implementable stories.

## Requirements Inventory

### Functional Requirements

FR1: Data Pipeline - Efficient data loading from MIRFlickr-25K dataset with support for pre-extracted features in HDF5 format.
FR2: Model Architecture - Implementation of the AGCH model, including AlexNet/MLP encoders, Graph Convolutional Networks (GCN), and a hashing layer for generating discrete binary codes.
FR3: Training Strategy - Support for alternating optimization with discrete constraints and professional GPU acceleration (RTX 5080).
FR4: Evaluation Metrics - Integration of standard image-text retrieval metrics, including mAP and Precision@k based on Hamming distance.

### NonFunctional Requirements

NFR-R1: Reproducibility - Fixed seeds and deterministic behavior across all training runs.
NFR-R2: Reproducibility - One-click reproduction of experiment results via configuration snapshot.
NFR-P1: Precision - Numerical stability during loss calculation and parameter updates (preventing NaN/Inf).
NFR-P2: Precision - Balanced loss weights (alpha, beta, gamma) for optimal convergence.
NFR-C1: Performance - Support for Mixed Precision training (FP16/BF16) on NVIDIA RTX 5080 (Blackwell).
NFR-C2: Performance - High-throughput data loading utilizing Intel Core i9-14900KF parallelism.

### Additional Requirements

- **Starter Template**: Use `ashleve/lightning-hydra-template` for project structure and configuration management.
- **Optimization Strategy**: Implement PyTorch Lightning **Manual Optimization** (`self.automatic_optimization = False`) to handle the alternating update logic (B vs F).
- **Data Storage**: Use **HDF5** for storing and loading pre-extracted features.
- **Formatting & Linting**: Enforce `black` and `flake8` for code quality.
- **Logging**: Use **TensorBoard** for experiment tracking.
- **Type Safety**: Require Python Type Hints and Tensor shape documentation in code.

### FR Coverage Map

FR1: Epic 2 - Data Pipeline & Feature Extraction
FR2: Epic 3 - AGCH Core Algorithm & Training Loop
FR3: Epic 3 - AGCH Core Algorithm & Training Loop
FR4: Epic 4 - System Validation & Performance Evaluation

## Epic List

### Epic 1: Project Foundation & Environment Setup
Establish the core project structure based on the PyTorch Lightning + Hydra template, ensuring all dependencies and configurations are ready for development.
**FRs covered:** N/A (Foundation for all FRs)

### Story 1.1: Initialize Project Structure & Dependencies

As a Developer,
I want to initialize the project repository using the `lightning-hydra-template` and configure the development environment,
So that I have a proven, reproducible foundation for deep learning research and development.

**Acceptance Criteria:**

**Given** A clean working directory and necessary tools (git, python, conda/pip)
**When** I initialize the project
**Then** A complete directory structure following `ashleve/lightning-hydra-template` standard should be created (configs/, src/, tests/)
**And** Explicit subdirectories in `src/` should be established: `src/data/`, `src/models/`, `src/utils/` (Architecture Consistency)
**And** All dependencies listed in `requirements.txt` should be installable without conflict including PyTorch 2.9+, Lightning 2.6+, Hydra 1.3+
**And** The `black` formatter and `flake8` linter should be configured in `pyproject.toml` or `.pre-commit-config.yaml`
**And** A basic sanity check (e.g., `python -c "import lightning; import hydra"`) should pass

### Story 1.2: Configure Paths & Logging Infrastructure

As a Developer,
I want to configure the project's path management and logging system via Hydra,
So that all experiments, checkpoints, and logs are automatically organized in the `logs/` directory without manual intervention.

**Acceptance Criteria:**

**Given** The initialized project structure
**When** I configure `configs/paths/default.yaml` and `configs/logger/tensorboard.yaml`
**Then** Running a dummy experiment should generate logs in `logs/train/runs/YYYY-MM-DD_HH-MM-SS`
**And** The `data_dir` should point to the local `data/` directory
**And** TensorBoard logging should be enabled by default
**And** Hydra configuration files should include comments or schemas for Type Hints to prevent runtime configuration errors



### Epic 2: Data Pipeline & Feature Extraction
Enable the system to process the MIRFlickr-25K dataset and store pre-extracted features in HDF5 format for efficient training.
**FRs covered:** FR1

### Story 2.1: HDF5 Data Module & Caching

As a Researcher,
I want a `LightningDataModule` that loads pre-extracted features from HDF5 files and supports in-memory caching,
So that I can maximize GPU utilization during training by eliminating disk I/O bottlenecks.

**Acceptance Criteria:**

**Given** Existing HDF5 feature files (images and tags)
**When** `AGCHDataModule.setup()` is called
**Then** It should verify file existence and load data into memory if `cache_in_memory=True` (NFR-C2)
**And** It should split data into train/retrieval/query sets as per standard protocol
**And** The `train_dataloader()` should return batches of `(image, text, index, label)` tuples


### Epic 3: AGCH Core Algorithm & Training Loop
Implement the end-to-end differentiable AGCH model, including encoders, GCN, hashing layer, AND the manual optimization training loop logic. This epic delivers a trainable model where loss converges.
**FRs covered:** FR2, FR3

### Story 3.1: AGCH Lightning Module Skeleton & Manual Optimization Setup

As a Algorithm Engineer,
I want to create the `AGCHModule` class inheriting from `LightningModule` with Manual Optimization enabled,
So that I can implement the alternating training logic for discrete hash codes.

**Acceptance Criteria:**

**Given** The initialized project structure
**When** I create `src/models/agch_module.py`
**Then** It should be a `LightningModule` with `self.automatic_optimization = False` set in `__init__`
**And** It should define empty placeholders for `self.img_enc`, `self.txt_enc`, `self.gcn`, and `self.hash_layer`
**And** It should implement `configure_optimizers` returning placeholders
**And** It should implement `configure_optimizers` returning placeholders
**And** It should pass a basic instantiation test

### Story 3.2: Implement Alternating Optimization Logic

As a Algorithm Engineer,
I want to implement the custom `training_step` with alternating optimization phases (Update F then Update B),
So that the model properly optimizes both the feature extraction network and the discrete hash codes.

**Acceptance Criteria:**

**Given** The `AGCHModule` with manual optimization
**When** `training_step` is executed
**Then** It should utilize `self.manual_backward()` for gradient calculation
**And** It should perform the "Fix B, Update F" step, ensuring the gradient of B is isolated (detached) from the encoder backprop
**And** It should perform the "Fix F, Update B" step (discrete optimization)
**And** It should log the loss for each phase using `self.log()`




### Epic 4: System Validation & Performance Evaluation
Implement strict evaluation metrics (mAP, Precision@k), visualization logging, and verification of non-functional requirements (reproducibility, numerical stability).
**FRs covered:** FR4

### Story 4.1: Implementation of Retrieval Metrics (mAP & P@k)

As a Data Scientist,
I want to implement mean Average Precision (mAP) and Precision@k metrics based on Hamming distance,
So that I can quantitatively evaluate the retrieval performance of the AGCH model.

**Acceptance Criteria:**

**Given** Query binary codes and Retrieval binary codes (plus ground truth labels)
**When** `calculate_mAP(query_code, retrieval_code, query_label, retrieval_label)` is called
**Then** It should compute the Hamming distance using **vectorized matrix operations** (e.g., XOR via Bitwise ops or Dot product approximation) for high performance
**And** It should sort results by distance (ascending)
**And** It should return the correct mAP value compared to a known reference (unit test)
**And** Precision@k should match expected values for top-k retrieved items

### Story 4.2: Reproducibility & Determinism Verification

As a Researcher,
I want to verify that the training process is fully deterministic under fixed random seeds,
So that I can results can be reliably reproduced as required by NFR-R1.

**Acceptance Criteria:**

**Given** Two identical training runs with the same fixed seed
**When** The system executes the first 100 iterations of training
**Then** The logged Loss values should be identical down to 6 decimal places
**And** The resulting model weights (state_dict) should be bit-for-bit identical
**And** The output mAP after the first evaluation should be exactly the same

### Epic 5: Final Integration & Experiments
Integrate all implemented components (Data, Model, Metrics) into the main training script, perform end-to-end training loops, and conduct hyperparameter tuning to achieve optimal retrieval performance.
**FRs covered:** FR2, FR3, FR4 (Integrated)

### Story 5.1: End-to-End System Integration

As a Machine Learning Engineer,
I want to implement the full training logic in `src/train.py` by instantiating the DataModule, Model, and Trainer with correct callbacks and loggers,
So that I can execute the complete training pipeline from a single command.

**Acceptance Criteria:**

**Given** The configured `src/train.py`
**When** I run `python src/train.py experiment=example`
**Then** The system should successfully initialize `AGCHDataModule` and `AGCHModule`
**And** It should execute `trainer.fit()` without errors
**And** It should save checkpoints and log metrics (Loss, mAP) to TensorBoard/WandB
**And** It should perform a final test evaluation
**And** It must pass `tests/test_integration_train.py` (Smoke Test) using `fast_dev_run=True` to verify crash-free execution
**And** Artifact Verification: The output directory must contain valid `.ckpt` and `config.yaml` files
**And** Override Testing: CLI overrides (e.g., `model.alpha=0.5`) must be correctly propagated to the saved config

### Story 5.2: Hyperparameter Tuning & Final Experiments

As a Researcher,
I want to run multiple experiments with varying hyperparameters (alpha, beta, gamma, hash code length),
So that I can find the optimal configuration that maximizes mean Average Precision (mAP).

**Acceptance Criteria:**

**Given** The integrated training pipeline
**When** I run a hyperparameter sweep (e.g., via Hydra multirun or manual scripts)
**Then** I should be able to override `model.alpha`, `model.beta`, and `model.gamma` via CLI
**And** I should obtain results for code lengths of 16, 32, and 64 bits
**And** The best model should achieve comparable mAP to the paper's reported results



