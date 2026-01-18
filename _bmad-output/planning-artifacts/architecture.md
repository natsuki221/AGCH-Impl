---
stepsCompleted:
  - step-01-init
  - step-02-context
  - step-03-starter
  - step-04-decisions
  - step-05-patterns
  - step-06-structure
  - step-07-validation
  - step-08-complete
workflowType: 'architecture'
lastStep: 8
status: 'complete'
completedAt: '2026-01-18'
inputDocuments:
  - _bmad-output/planning-artifacts/prd.md
  - _bmad-output/planning-artifacts/research/technical-AGCH-research-2026-01-18.md
  - docs/AGCH-Guide.md
  - docs/project-overview.md
workflowType: 'architecture'
project_name: 'AGCH-Impl'
user_name: 'Ncu-caic'
date: '2026-01-18'
---

# Architecture Decision Document

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Project Context Analysis

### Requirements Overview

**Functional Requirements:**
- **Data Pipeline**: High-throughput loading of MIRFlickr-25K images/tags. Pre-extraction to HDF5 required.
- **Model Architecture**: Modular Design (AlexNet/MLP Encoders + GCN + Hash Layer).
- **Training Strategy**: Alternating Optimization (Discrete constraints) with custom loss functions.
- **Evaluation**: Standard mAP metric with Hamming Distance calculation.

**Non-Functional Requirements:**
- **Reproducibility**: Strict seed fixing (NFR-R1) and one-click reproduction script (NFR-R2).
- **Precision**: Numerical stability in loss/gradients (NFR-P1, P2).
- **Compatibility**: Optimized for RTX 5080 (Blackwell) and Mixed Precision (NFR-C1, C2).

**Scale & Complexity:**
- **Primary Domain**: Research Implementation / Deep Learning (Computer Vision + NLP).
- **Complexity Level**: **High**. Involves optimization on manifolds/discrete spaces, not just standard backprop.
- **Estimated Components**: ~8 (Data, Models, Losses, Opt, Trainer, Eval, Config, Utils).

### Technical Constraints & Dependencies

- **GPU**: NVIDIA RTX 5080 (Blackwell architecture, sm_120).
- **CPU**: **Intel Core i9-14900KF** (24 Cores, 32 Threads, Max 6.0GHz) - Allows high parallelism for DataLoaders.
- **RAM**: **64GB Total** (62Gi Available) - Plenty of headroom for caching datasets in RAM.
- **Software Stack**: PyTorch 2.9+, CUDA 12.8, Python 3.11+.

### Cross-Cutting Concerns Identified

- **Configuration Management**: Centralized YAML config must drive ALL components (Model, Data, Training).
- **Experiment Tracking**: Consistent logging (loss, mAP) across all runs for reproducibility.
- **Numerical Stability**: Gradient clipping and NaN checks needed globally.

## Starter Template Evaluation

### Primary Technology Domain

**Research Implementation (Deep Learning / Python)** based on project requirements analysis.

### Starter Options Considered

- **Option A: Cookiecutter Data Science**: Good structure but lacks deep PyTorch integration.
- **Option B: PyTorch Lightning + Hydra (ashleve)**: **Selected**. Best for reproducibility, config management, and removing boilerplate suitable for complex research tasks.
- **Option C: Pure PyTorch**: Too much manual boilerplate risk for NFR-R1 (seed/device handling).

### Selected Starter: PyTorch Lightning + Hydra (ashleve)

**Rationale for Selection:**
1.  **Reproducibility (NFR-R1)**: Hydra ensures every run has a snapshot config. Lightning handles strict seeding and deterministic flags.
2.  **Productivity**: Removes `train_loop`, `cuda()` calls, and checkpoint management, allowing focus on AGCH logic.
3.  **Scalability**: Native support for Multi-GPU (if needed in future) and Mixed Precision (NFR-C2 for RTX 5080).
4.  **Config-Driven**: Perfectly matches the requirement for YAML-based experiment management (FR-I1).

**Initialization Command:**

```bash
# Recommended initialization (conceptual, as we might adapt manually for existing repo)
# git clone https://github.com/ashleve/lightning-hydra-template
# pip install -r requirements.txt
```

**Architectural Decisions Provided by Starter:**

**Language & Runtime:**
- Python 3.11+, PyTorch 2.9+, CUDA 12.8.

**Styling Solution:**
- **Black** (Code Formatter) + **Flake8** (Linter).

**Experience Features:**
- **Hydra 1.3**: Automatic CLI overrides (`python train.py experiment=agch_mirflickr`).
- **Lightning 2.6**: Automated mixed precision (`precision="16-mixed"`).
- **Rich Logging**: Console progress bars + file logs.

**Note:** Integrating this template structure will be the first step of Phase 4 Implementation.

## Core Architectural Decisions

### Decision Priority Analysis

**Critical Decisions (Block Implementation):**
- **Data Format**: HDF5 for pre-extracted features.
- **Optimization Strategy**: Manual Optimization Loop for Alternating Updates.
- **Experiment Logging**: TensorBoard for metrics.

**Important Decisions (Shape Architecture):**
- **Config Management**: Hydra Composition for modular experiments.
- **Reproducibility**: `seed_everything` + deterministic trainer flags.

### Data Architecture

- **Category**: Storage Format
- **Decision**: **HDF5 (Offline Extraction)**
- **Version**: `h5py` >= 3.10
- **Rationale**: Efficient chunking and unified storage for features + labels. Fits PRD requirements.
- **Provided by Starter**: No (Need to implement `AGCHDataModule`).

### Authentication & Security

- **Category**: N/A
- **Decision**: None required for Research Codebase.

### API & Communication Patterns

- **Category**: CLI Interface
- **Decision**: **Hydra CLI**
- **Rationale**: Allows complex config overrides (`python train.py experiment=agch_mirflickr model.hash_code_len=32`).
- **Provided by Starter**: Yes (Hydra integration).

### Training Architecture

- **Category**: Optimization Loop
- **Decision**: **PyTorch Lightning Manual Optimization**
- **Version**: Lightning 2.0+
- **Rationale**: AGCH requires updating B (Binary Codes) while holding F (Features) fixed, and vice-versa. Automatic optimization cannot handle this discrete/alternating logic.
- **Implementation**: Set `self.automatic_optimization = False` and use `self.manual_backward()`.

### Infrastructure & Deployment

- **Category**: Experiment Tracking
- **Decision**: **TensorBoard**
- **Rationale**: Built-in default for Lightning. Sufficient for local research on RTX 5080. No external cloud dependency required.
- **Provided by Starter**: Yes (Default Logger).

### Decision Impact Analysis

**Implementation Sequence:**
1.  **Project Init**: Clone template & setup Hydra.
2.  **Data Loop**: Implement `AGCHDataModule` to read HDF5.
3.  **Model Core**: Implement `AGCH` LightningModule with Manual Optimization.
4.  **Trainer**: Config for RTX 5080 (Mixed Precision).

**Cross-Component Dependencies:**
- `AGCHDataModule` must expose strict shapes for `AGCH` model input.
- `AGCHDataModule` must expose strict shapes for `AGCH` model input.
- `AGCH` model must expose `hash_code_len` to Hydra config.

## Implementation Patterns & Consistency Rules

### Implementation Strategy

**Critical Rule:** All code must be compatible with **PyTorch Lightning 2.0+** and **Hydra 1.3+**.

### Naming Patterns

**Variables & Tensors:**
- **Math-Notation Alignment**: Use `X` (Image), `T` (Text), `L` (Label), `B` (Binary Code) to match paper formulas.
- **Suffixes**: No hungarian notation, but use comments for shape info.
- **Example**: `F_I = self.img_enc(X) # [Batch, FeatureDim]`

**Module Attributes:**
- **Encoders**: `self.img_enc`, `self.txt_enc`
- **Core Layers**: `self.gcn_layers` (ModuleList), `self.hash_layer`
- **Losses**: `self.criterion_agch`, `self.criterion_quantization`

**Config Keys:**
- **Snake Case**: `learning_rate`, `hash_code_len`, `batch_size`.
- **Paths**: `data_dir`, `save_dir`.

### Structure Patterns

**Hydra Config Layout:**
- `configs/model/agch.yaml`: Model architecture params (alpha, beta, gamma).
- `configs/data/mirflickr.yaml`: Dataset paths and loader params.
- `configs/experiment/run_01.yaml`: Composition of model + data + overrides.

**LightningModule Order:**
1.  `__init__`: Define submodules and hyperparameters.
2.  `forward`: Inference logic (generation of hash codes).
3.  `training_step`: Loss calculation and manual backward pass.
4.  `configure_optimizers`: Define optimizers for Alternating Optimization.
5.  `on_train_epoch_end`: Logging or custom logic.

### Enforcement Guidelines

**All AI Agents MUST:**
- Run `black` formatter on all Python files.
- Add Type Hints to critical methods (`forward`, `__init__`) with Tensor shape comments.
- Use `self.log()` for all metrics (no `print` statements in training loop).

**Pattern Examples:**

**Good:**
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: Image features [Batch, 4096]
    Returns:
        h: Hash codes [Batch, hash_len]
    """
    feat = self.img_enc(x)
    return self.hash_layer(feat)
```

## Project Structure & Boundaries

### Complete Project Directory Structure

```bash
AGCH-Impl/
├── configs/                     # [Hydra] Centralized Configuration
│   ├── data/
│   │   └── mirflickr.yaml       # Dataset paths, batch size, num_workers
│   ├── model/
│   │   └── agch.yaml            # Hyperparameters (alpha, beta, gamma, hash_len)
│   ├── experiment/
│   │   └── example_run.yaml     # Composition: Model + Data + Overrides
│   ├── paths/
│   │   └── default.yaml         # Project paths
│   ├── trainer/
│   │   └── default.yaml         # Lightning Trainer args (epochs, precision)
│   └── train.yaml               # Main entry point config
├── src/
│   ├── data/
│   │   ├── agch_datamodule.py   # [Lightning] DataModule implementation
│   │   └── make_dataset.py      # [Script] Offline feature extraction to HDF5
│   ├── models/
│   │   ├── components/
│   │   │   ├── alexnet.py       # Image Encoder backbone
│   │   │   ├── gcn.py           # Graph Convolution Layers
│   │   │   └── hash.py          # Hashing Layer (Tanh/Discrete)
│   │   └── agch_module.py       # [Lightning] Training Loop, Validation, Manual Opt
│   ├── utils/
│   │   ├── metrics.py           # mAP, Precision@k calculation
│   │   └── instantiation.py     # Hydra instantiation helpers
│   └── train.py                 # Main training entry point
├── logs/                        # [Generated] Experiment logs & checkpoints
├── data/                        # [Ignored] Local data storage
│   ├── mirflickr25k/
│   └── features/                # HDF5 files location
├── tests/                       # [PyTest] Unit & Integration tests
│   ├── test_datamodule.py
│   └── test_model.py
├── .env                         # Environment variables (API keys, etc.)
├── .gitignore                   # Git ignore rules
├── pyproject.toml               # Project dependencies & tool config
└── README.md                    # Project documentation
```

### Architectural Boundaries

**API Boundaries (Configuration Interface):**
- **Input**: `configs/*.yaml` is the ONLY interface for changing behavior. No hardcoded constants in `src/`.
- **Overrides**: CLI arguments (`model.alpha=0.1`) take precedence over YAML files.

**Component Boundaries:**
- **Data ↔ Model**: Decoupled via `torch.Tensor`. `AGCHModule` implies no knowledge of file paths, only tensor shapes.
- **Model ↔ Trainer**: Decoupled via `LightningModule`. The model defines *what* to optimize, Trainer defines *how* (hardware, epochs).

**Data Boundaries:**
- **Raw vs Processed**: Raw images are read ONLY by `make_dataset.py`. `AGCHDataModule` reads ONLY processed HDF5 features.
- **Output**: All artifacts (checkpoints, logs) must go to `logs/`, never root.

### Requirements to Structure Mapping

- **PRD FR-D1 (Data Pipeline)** → `src/data/agch_datamodule.py` + `src/data/make_dataset.py`
- **PRD FR-M1 (Model Arch)** → `src/models/agch_module.py` + `src/models/components/`
- **PRD FR-T1 (Training)** → `src/train.py` + `configs/trainer/`
- **PRD FR-T1 (Training)** → `src/train.py` + `configs/trainer/`
- **PRD NFR-R1 (Reproducibility)** → `configs/` (Hydra snapshot) + `src/utils/instantiation.py` (Seeding)

## Architecture Validation Results

### Coherence Validation ✅

**Decision Compatibility:**
- **Lightning + Hydra**: Standard industry pattern. No conflicts.
- **HDF5 + DataModule**: HDF5 allows efficient, chunked usage fitting DataModule lifecycle.
- **Manual Optimization + AGCH**: Specifically addresses the "Alternating Optimization" requirement that standard trainers fail at.

**Pattern Consistency:**
- Naming patterns (Math-notation) align with the Research nature of the code.
- Structure patterns (Modular Configs) support the Reproducibility NFR.

### Requirements Coverage Validation ✅

**Functional Requirements Coverage:**
- **Data Pipeline (FR-D1)**: Covered by HDF5 Decision & `make_dataset.py`.
- **Model Arch (FR-M1)**: Covered by Modular Component Breakdown.
- **Training (FR-T1)**: Covered by Manual Optimization Strategy.
- **Evaluation (FR-E1)**: Covered by `metrics.py` and TensorBoard logging.

**Non-Functional Requirements Coverage:**
- **Reproducibility (NFR-R1)**: Strictly enforcing Hydra Configs & Seeding.
- **Precision (NFR-P1)**: Mixed Precision support via Lightning Trainer.
- **Performance (NFR-C1)**: HDF5 + RTX 5080 Optimization.

### Implementation Readiness Validation ✅

**Readiness Status:** **READY FOR IMPLEMENTATION**

- **Decisions**: All critical tech choices (Framework, Data, Config) made.
- **Structure**: Complete file tree defined.
- **Patterns**: Coding standards (Black, Type Hints) established.

### Gap Analysis Results

**Addressed Risks:**
- **Risk**: Debugging complex Hydra configs.
- **Mitigation**: Create `configs/debug.yaml` for simplified, single-process debugging.
- **Risk**: Numerical Instability (NaNs).
- **Mitigation**: Enforce naming patterns for tensors to track shapes, and use `self.log` to monitor loss spikes.

### Architecture Completeness Checklist

**✅ Requirements Analysis**
- [x] Project context thoroughly analyzed
- [x] Scale and complexity assessed

**✅ Architectural Decisions**
- [x] Critical decisions documented with versions
- [x] Technology stack fully specified
- [x] Integration patterns defined

**✅ Implementation Patterns**
- [x] Naming conventions established
- [x] Structure patterns defined

**✅ Project Structure**
- [x] Complete directory structure defined
- [x] Component boundaries established

### Implementation Handoff

**AI Agent Guidelines:**
1.  **Strict Config Separation**: NEVER hardcode hyperparameters. Always plumb through Hydra.
2.  **Type Safety**: Always annotate Tensor shapes in docstrings or type hints.
3.  **Manual Opt**: Respect the specific `training_step` logic for AGCH.

**First Implementation Priority:**
Initialize project skeleton using `ashleve/lightning-hydra-template` structure and setup `src/data` for MIRFlickr.

## Architecture Completion Summary

### Workflow Completion

**Architecture Decision Workflow:** COMPLETED ✅
**Total Steps Completed:** 8
**Date Completed:** 2026-01-18
**Document Location:** _bmad-output/planning-artifacts/architecture.md

### Final Architecture Deliverables

**📋 Complete Architecture Document**

- All architectural decisions documented with specific versions
- Implementation patterns ensuring AI agent consistency
- Complete project structure with all files and directories
- Requirements to architecture mapping
- Validation confirming coherence and completeness

**🏗️ Implementation Ready Foundation**

- **Critical Decisions**: Lightning+Hydra, HDF5, Manual Optimization.
- **Patterns**: Math-notation naming, Modular Configs.
- **Structure**: Full file tree for Research Codebase.
- **Requirements**: All FRs and NFRs supported (especially Reproducibility).

### Quality Assurance Checklist

**✅ Architecture Coherence**
- [x] All decisions work together without conflicts
- [x] Technology choices are compatible

**✅ Requirements Coverage**
- [x] All functional requirements are supported
- [x] All non-functional requirements are addressed

**✅ Implementation Readiness**
- [x] Decisions are specific and actionable
- [x] Patterns prevent agent conflicts

### Project Success Factors

**🎯 Clarity for Research**
Specific decisions (Manual Optimization, HDF5) address the unique challenges of Deep Learning Research, avoiding generic Web App patterns.

**🔧 Reproducibility First**
Hydra-centric design ensures every experiment is logging-ready and reproducible from Day 1.

---

**Architecture Status:** READY FOR IMPLEMENTATION ✅

**Next Phase:** Begin implementation using the architectural decisions and patterns documented herein.


