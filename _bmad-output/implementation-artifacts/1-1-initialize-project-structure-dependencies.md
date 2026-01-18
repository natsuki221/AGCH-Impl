# Story 1.1: Initialize Project Structure & Dependencies

**Status:** ready-for-dev

## Story

As a Developer,  
I want to initialize the project repository using the `lightning-hydra-template` and configure the development environment,  
So that I have a proven, reproducible foundation for deep learning research and development.

## Acceptance Criteria

1.  **Given** A clean working directory and necessary tools (git, python, conda/pip) / **When** I initialize the project / **Then** A complete directory structure following `ashleve/lightning-hydra-template` standard should be created (configs/, src/, tests/).
2.  **And** Explicit subdirectories in `src/` should be established: `src/data/`, `src/models/`, `src/utils/` (Architecture Consistency).
3.  **And** All dependencies listed in `requirements.txt` should be installable without conflict including PyTorch 2.9+, Lightning 2.6+, Hydra 1.3+.
4.  **And** The `black` formatter and `flake8` linter should be configured in `pyproject.toml` or `.pre-commit-config.yaml`.
5.  **And** A basic sanity check (e.g., `python -c "import lightning; import hydra"`) should pass.

## Technical Requirements (Dev Agent Guardrails)

### Library & Framework Requirements
-   **Template Source**: `ashleve/lightning-hydra-template` (Use latest stable or clone/adapt structure).
-   **Core Libs**:
    -   `torch` >= 2.9
    -   `lightning` >= 2.6
    -   `hydra-core` >= 1.3
    -   `torchmetrics`
    -   `rootutils` (Standard in the template for root path handling)
-   **Dev Tools**:
    -   `black` (Formatting)
    -   `flake8` (Linting)
    -   `pre-commit` (Optional but recommended)
    -   `pytest` (Testing)

### File Structure Requirements (Architecture Compliance)
The final structure MUST match the Architecture Plan:

```bash
AGCH-Impl/
├── configs/                     # [Hydra] Centralized Configuration
│   ├── data/
│   ├── model/
│   ├── experiment/
│   ├── paths/
│   ├── trainer/
│   └── train.yaml               # Main entry point config
├── src/
│   ├── data/                    # For AGCHDataModule later
│   ├── models/                  # For AGCHModule later
│   ├── utils/
│   └── train.py                 # Main training entry point
├── tests/
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

### Implementation Steps (Guide)
1.  **Clone/Init**: Since we are in an existing repo, do NOT clone the template into a subfolder. Instead, **copy/replicate** the structure into the root.
    -   *Constraint*: Do not overwrite existing `_bmad` or `docs` folders.
2.  **Cleanup**: Remove any example files from the template (e.g., MNIST example) unless used for sanity check.
3.  **Config**: Create/Update `pyproject.toml` with `black` and `flake8` settings.
    -   Black: `line-length = 88`
    -   Flake8: `max-line-length = 88`, `extend-ignore = E203`
4.  **Dependencies**: Create `requirements.txt` with pinned versions.
5.  **Verify**: Install requirements and run import check.

## Dev Notes

-   **Architecture Decisions**:
    -   We chose **PyTorch Lightning + Hydra** for reproducibility (NFR-R1).
    -   We require **strict separation** of config files.
-   **Existing Structure**: The `_bmad` folder is critical for the agent system. **DO NOT DELETE OR MOVE IT.**
-   **Git**: Ensure `.gitignore` covers python artifacts (`__pycache__`, `*.pyc`, `venv/`, `.env`, `logs/`, `data/`).

## Verification Plan

### Automated Tests
-   Run `pip install -r requirements.txt` and ensure success.
-   Run `pytest` (even if no tests yet, checking test discovery).
-   Run `black . --check` to verify formatting config.

### Manual Verification
-   Execute: `python -c "import lightning; import hydra; import torch; print(f'Lightning: {lightning.__version__}, Hydra: {hydra.__version__}')"`
-   Verify output contains expected versions.
