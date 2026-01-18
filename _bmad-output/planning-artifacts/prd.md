---
stepsCompleted:
  - step-01
  - step-02
  - step-03
  - step-04
  - step-05
  - step-06
  - step-07
  - step-08
  - step-09
  - step-10
  - step-11
inputDocuments:
  - docs/project-overview.md
  - _bmad-output/planning-artifacts/research/technical-AGCH-research-2026-01-18.md
classification:
  projectType: Research Implementation
  domain: Multi-modal AI
  complexity: High
  projectContext: Greenfield
workflowType: 'prd'
---

# Product Requirements Document - AGCH-Impl

**Author:** Ncu-caic
**Date:** 2026-01-18

<!-- PRD content will be added in subsequent steps -->

## Success Criteria

### User Success

- **Code Readability & Modularity**: The codebase must be modular (Encoder, GCN, Loss separated) to facilitate future research modification.
- **Reproducibility**: A single script `reproduce_paper.sh` should be able to run the full training and evaluation pipeline and yield results close to the paper.
- **Documentation**: Clear documentation mapping mathematical formulas from the paper to the code implementation.

### Business Success (Research Goals)

- **Performance targets (MIRFlickr-25K @ 16 bits)**:
    - Image-to-Text mAP: **~0.865**
    - Text-to-Image mAP: **~0.829**
- **Publication Readiness**: The implementation should be robust enough to serve as a baseline for future paper submissions.

### Technical Success

- **Environment Stability**: Error-free execution on **RTX 5080** with **PyTorch 2.9.1+cu128** and **CUDA 12.8**.
- **End-to-End Pipeline**: Fully functional pipeline from data loading -> model training -> hash code generation -> retrieval evaluation.
- **Implementation Correctness**: Correct implementation of core AGCH components:
    - Aggregation Similarity Matrix (Hadamard product of Cosine & Euclidean)
    - GCN Module structure
    - All 3 Loss functions (Reconstruction, Structure, Cross-modal)

### Measurable Outcomes

- `python train.py` runs without error on GPU.
- Evaluation script outputs mAP scores matching golden targets within reasonable margin (+/- 1-2%).

## Product Scope

### MVP - Minimum Viable Product

- **Core AGCH Model**: Full PyTorch implementation of the AGCH architecture.
- **Data Pipeline**: DataLoader for MIRFlickr-25K dataset with pre-extracted features (or raw image/text processing).
- **Training Loop**: Training script with alternating optimization strategy.
- **Evaluation**: Script to calculate mAP for I2T and T2I retrieval tasks.
- **Environment**: Compilable `environment.yml` and setup scripts.

### Growth Features (Post-MVP)

- **Additional Datasets**: Support for NUS-WIDE or MS-COCO.
- **Experiment Tracking**: Integration with WandB or MLflow.
- **Hyperparameter Optimization**: Automated tuning scripts (e.g., Optuna).
- **Model Checkpointing**: Save/Resume training functionality.

### Vision (Future)

- **AGCH+**: Enhanced version of AGCH with Transformer-based encoders (ViT, BERT) instead of AlexNet/MLP.
- **Real-time Retrieval**: Optimized inference pipeline for large-scale retrieval.

## User Journeys

### 1. Model Training (Primary - Reproduction)

**Persona**: AI Researcher attempting to reproduce AGCH paper results.

**Story**:
The researcher starts by setting up the environment using `setup-env.sh` to ensure all dependencies (PyTorch 2.9, CUDA 12.8) are correct. They then prepare the **MIRFlickr-25K** dataset.
They execute the training command `python train.py --config configs/mirflickr.yaml`.
During training, they monitor the **Loss curves** (Reconstruction, Structure, Cross-modal) via the terminal or a log file to ensure convergence.
Upon completion, the best model weights are automatically saved to the `checkpoints/` directory.

### 2. Model Evaluation (Primary - Validation)

**Persona**: AI Researcher verifying reproduction accuracy.

**Story**:
With a trained model, the researcher runs `python evaluate.py --checkpoint checkpoints/best_model.pth`.
The system loads the test set, generates **binary hash codes** for both queries and database items.
It calculates the **Hamming Distance** and computes **mAP@TopK** for both Image-to-Text and Text-to-Image tasks.
The researcher compares the output mAP (e.g., 0.865) against the paper's reported figures to validate technical success.

### 3. Model Extension (Secondary - Innovation)

**Persona**: AI Researcher improving the architecture.

**Story**:
The researcher decides to replace the `AlexNet` image encoder with `ResNet50`.
They navigate to `src/models/` and implement a new `ResNetEncoder` class following the project's modular interface.
They update the configuration file `configs/mirflickr.yaml` to use the new encoder.
They re-run the training pipeline to observe if the new architecture yields better mAP performance.

### Journey Requirements Summary

- **Configuration System**: YAML-based config for flexible switching of models and datasets.
- **Modular Interface**: Abstract base classes for Encoders and GCN modules.
- **Logging System**: Real-time training progress and metric logging.
- **Checkpointing**: Automatic saving of best models.
- **Evaluation Metrics**: Standardized mAP calculation implementation (Haming distance based).

## Domain-Specific Requirements

### Hardware Optimization (RTX 5080 Specific)

- **Mixed Precision Training**: Leverage **FP16/BF16** capabilities of the Blackwell architecture to accelerate training and reduce memory usage.
- **PTX JIT Compilation**: Ensure `pytorch-cuda` interaction via JIT compilation works seamlessly for capabilities > sm_90 until native support arrives.
- **Batch Size Optimization**: Utilize the **16GB VRAM** to maximize batch size (e.g., 64/128) to stabilize gradient estimation in AGCH alternate optimization.
- **DataLoader Efficiency**: Configure `num_workers` and `pin_memory=True` to prevent GPU starvation, matching the high throughput of the 5080.

### Academic Compliance

- **Dataset Citation**: Documentation must properly cite MIRFlickr-25K and AGCH paper.
- **Metric Standardization**: mAP calculation must strictly follow the standard definition (Hamming ranking) used in the hashing retrieval domain.
- **Seed Fixing**: Implement `seed_everything()` to ensure `torch`, `numpy`, and `python` random seeds are fixed for reproducible results.

### Technical Constraints & Stability

- **Numerical Stability**: Add safety clamps/epsilons to custom loss functions (especially logarithmic terms) and `tanh` gradients to prevent NaN during training.
- **Gradient Clipping**: Implement gradient clipping to handle potential exploding gradients from the alternating optimization process.
- **Data Normalization**: Standardization of AlexNet features and Text Bag-of-Words vectors is mandatory before feeding into the network.

## Innovation & Novel Patterns

### Detected Innovation Areas

- **Aggregation Similarity Matrix**: A novel approach combining Cosine similarity (direction) and Euclidean distance (magnitude) to construct a more robust affinity matrix for supervision.
- **GCN-enhanced Hashing**: Integrating Graph Convolutional Networks to verify and refine the structural relationships between data points during hash code generation.
- **Discrete Optimization Strategy**: A specialized alternating update rule to handle the binary constraints of hash codes without relaxing them to continuous values during early training.

### Market Context & Competitive Landscape

- **Status**: While many deep hashing methods exist (DCMH, CPAH), most rely on single-metric similarity. AGCH's dual-metric aggregation is a unique differentiator in the unsupervised cross-modal retrieval domain.
- **Gap**: No official implementation exists, making this reproduction valuable for the community.

### Validation Approach

- **A/B Testing**: Compare mAP scores with and without the "Aggregation" module (i.e., using only Cosine or only Euclidean) to validate the core contribution.
- **Baseline Comparison**: Direct comparison against published results of TEGAH, EGATH, and other state-of-the-art methods.

### Risk Mitigation

- **Replication Risk**: If results don't match the paper, we will verify the "Aggregation" formula implementation and the `tanh` approximation parameter $\gamma$ carefully.

## Research Implementation Specific Requirements

### Project-Type Overview

This project is a **Research Codebase**, designed for reproducibility, extensibility, and academic verification. It prioritizes clarity and correctness over production-grade latency.

### Technical Architecture Considerations

- **Configuration System**:
    - **YAML-driven**: All hyperparameters (LR, batch size, $\alpha$, $\delta$) must be defined in `configs/*.yaml`.
    - **No hardcoding**: Models and DataLoaders should read parameters from the config object.

- **Data Pipeline**:
    - **Pre-computed Features**: AlexNet image features and Text BOW vectors will be pre-extracted and stored in **HDF5 or NPY** format to speed up training I/O.
    - **Feature Extraction Script**: A dedicated `extract_features.py` script is required to process raw images into the required format.

- **Experiment Management**:
    - **Timestamped Logging**: Experiment results are saved in `runs/YYYY-MM-DD_HH-MM-SS/` to prevent overwriting.
    - **Log Content**: Each run directory must contain:
        - `config.yaml` (copy of the config used)
        - `best_model.pth`
        - `train.log` (text log of metrics)
        - `events.out.tfevents` (TensorBoard/WandB logs)

### Implementation Considerations

- **Python Version**: 3.11+ (Compatible with PyTorch 2.9).
- **Package Management**: Conda/Mamba for environment isolation.
- **Code Style**: Black/Flake8 compliant for readability.

## Project Scoping & Phased Development

### MVP Strategy & Philosophy

**MVP Approach:** **Research Reproduction**. The goal is to strictly replicate the results of the AGCH paper to establish a baseline. Complexity is managed by modular design, allowing components to be swapped later.
**Resource Requirements:** 1 Research Engineer (User) + AI Assistant.

### MVP Feature Set (Phase 1)

**Core User Journeys Supported:**
- Model Training (Reproduction)
- Model Evaluation (Validation)

**Must-Have Capabilities:**
1.  **Environment**: Full CUDA 12.8 + PyTorch 2.9 support.
2.  **Data**: MIRFlickr-25K pre-processing pipeline (Feature Extraction -> HDF5).
3.  **Model**: Complete AGCH architecture (Encoders, GCN, Hash Layer, Aggregation Matrix).
4.  **Training**: Alternating optimization loop with gradient clipping.
5.  **Evaluation**: Standard mAP calculation protocol.
6.  **Config**: YAML-based parameter management.

### Post-MVP Features

**Phase 2 (Growth):**
- Support for **NUS-WIDE** dataset.
- Integration with **WandB** for experiment tracking.
- **Checkpoint Resume** functionality.

**Phase 3 (Expansion/Vision):**
- **AGCH+**: Transformer-based backbones (ViT/BERT).
- **Optuna**: Automated hyperparameter tuning.

### Risk Mitigation Strategy

**Technical Risks:**
- **Discrete Optimization Instability**: Mitigate by implementing gradient clipping and strict numerical stability checks (NaN detection).
- **Hyperparameter Sensitivity**: Mitigate by using the paper's exact values as defaults in `config.yaml`.

**Resource Risks:**
- **VRAM Limitations**: Although RTX 5080 has 16GB, large batch sizes might OOM. Mitigate by supporting Gradient Accumulation if needed (though likely not needed for AlexNet).

## Functional Requirements

### Data Processing Capability

- **FR-D1**: The system must load MIRFlickr-25K raw images and associated multi-label text tags from disk.
- **FR-D2**: The system must extract 4096-dim image features using a pre-trained AlexNet (fc7 layer).
- **FR-D3**: The system must process text tags into a 1386-dim feature vector using PCA on binary tagging vectors.
- **FR-D4**: The system must implement the specific split strategy for MIRFlickr-25K:
    - **Test Set (Query/Probe)**: Randomly select 2,000 samples.
    - **Training Set**: Randomly select 5,000 samples from the retrieval database.
    - **Retrieval Database**: Remaining samples (or total samples minus query, depending on exact paper methodology).
- **FR-D5**: The system must package processed features into HDF5/NPY format for efficient I/O.

### Model Architecture Capability

- **FR-M1**: The system must instantiate a CNN-based Image Encoder (default: AlexNet).
- **FR-M2**: The system must instantiate an MLP-based Text Encoder (dim: 1386 -> 4096 -> c).
- **FR-M3**: The system must compute the Aggregation Similarity Matrix using Hadamard product of Cosine Similarity and Euclidean Distance.
- **FR-M4**: The system must implement a GCN module to aggregate neighborhood information for hash code generation.

### Training Pipeline Capability

- **FR-T1**: The system must execute an Alternating Optimization strategy (update B, then Theta_v, then Theta_t, etc.).
- **FR-T2**: The system must compute and optimize the total loss combining Reconstruction (L1), Structure (L2), and Cross-modal (L3) terms.
- **FR-T3**: The system must log training metrics (Total Loss, L1/L2/L3 Loss) to both console and log files.
- **FR-T4**: The system must automatically save model checkpoints (best metric and last epoch).

### Evaluation System Capability

- **FR-E1**: The system must load trained weights for inference.
- **FR-E2**: The system must generate binary hash codes for Query set and Retrieval Database.
- **FR-E3**: The system must calculate Hamming Distances between query and database codes.
- **FR-E4**: The system must compute mAP@TopK for Image-to-Text and Text-to-Image tasks.

### Infrastructure Capability

- **FR-I1**: The system must use YAML configuration files for all hyperparameter definitions.
- **FR-I2**: The system must allow overriding config parameters via CLI arguments.

- **FR-I3**: The system must support deterministic execution via `seed_everything`.

## Non-Functional Requirements

### Reproducibility

- **NFR-R1**: Variance in mAP between two independent runs with fixed seeds must be less than **0.1%**.
- **NFR-R2**: The repository must include a one-click reproduction script (`reproduce_paper.sh`) that runs successfully on a fresh environment.

### Numerical Precision & Stability

- **NFR-P1**: Hash codes must be strictly binary {-1, 1} during inference. Training must handle relaxation (tanh) effectively without divergence.
- **NFR-P2**: Loss calculation must remain stable (no NaN/Inf) throughout training (>=100 Epochs).

### Environment Compatibility

- **NFR-C1**: The system must run on **NVIDIA RTX 5080** (sm_120) without CUDA errors.
- **NFR-C2**: The system must support **Mixed Precision (AMP)** training with negligible accuracy loss (< 0.5% mAP drop) for efficiency.
