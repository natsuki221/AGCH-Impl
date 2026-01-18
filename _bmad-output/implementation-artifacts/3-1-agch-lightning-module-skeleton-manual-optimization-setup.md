# Story 3.1: AGCH Lightning Module Skeleton & Manual Optimization Setup

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a Algorithm Engineer,
I want to create the `AGCHModule` class inheriting from `LightningModule` with Manual Optimization enabled,
so that I can implement the alternating training logic for discrete hash codes.

## Acceptance Criteria

1. **Lightning Framework Compliance**: Must inherit from `L.LightningModule` and set `self.automatic_optimization = False` in `__init__`. [Source: architecture.md#Training Architecture]
2. **Modular Architecture**: Define placeholders for `img_enc`, `txt_enc`, `gcn`, and `hash_layer` as per design. [Source: epics.md#Story 3.1]
3. **Internal Math-Notation**: Use `X`, `T`, `L`, `B` naming in forward/step to match paper formulas. [Source: architecture.md#Naming Patterns]
4. **Manual Optimization API**: Implement `configure_optimizers` returning a list of optimizers (at least two for alternating update).
5. **Type Safety**: Critical methods must include Python Type Hints and Tensor shape documentation. [Source: architecture.md#Enforcement Guidelines]
6. **Hardware Forward-Compatibility**: Code must pass `torch.compile` compatibility check. [Source: architecture.md#PyTorch 2.6 關鍵優化]
7. **Verification**: A unit test must confirm that the module can be instantiated and moved to GPU without errors.

## Tasks / Subtasks

- [ ] **Task 1: Create Module Skeleton (AC: 1, 2, 3)**
  - [ ] Create `src/models/agch_module.py`.
  - [ ] Implement `__init__` with `automatic_optimization=False`.
  - [ ] Define placeholders for sub-modules using `torch.nn.Identity` or `nn.Module` stubs.
- [ ] **Task 2: Configure Manual Optimizers (AC: 4)**
  - [ ] Implement `configure_optimizers` returning multiple Adam optimizers (placeholders).
- [ ] **Task 3: Implement Basic Forward Logic (AC: 3, 5)**
  - [ ] Implement `forward` method with Type Hints and shape comments.
  - [ ] Support handling for both Image ($X$) and Text ($T$) inputs.
- [ ] **Task 4: Unit Test & Verification (AC: 6, 7)**
  - [ ] Create `tests/test_agch_module.py` for instantiation checks.
  - [ ] Verify `torch.compile` doesn't throw errors on the skeleton.

## Dev Notes

- **Manual Optimization**: This is a non-standard Lightning flow! Ensure the use of `self.manual_backward()` in the future Story 3.2.
- **Naming Patterns**: Strictly follow paper notation for Tensors in docstrings.
- **RTX 5080**: Keep Blackwell architecture in mind; use `bfloat16` if possible in tests.

### Project Structure Notes

- Module should be located at `src/models/agch_module.py`.
- Sub-components (backbones) will be implemented in subsequent stories inside `src/models/components/`.

### References

- [Architecture: Training Loop](file:///home/ncu-caic/Documents/Coding/github.com/natsuki221/AGCH-Impl/_bmad-output/planning-artifacts/architecture.md#L139-L146)
- [Architecture: Naming Patterns](file:///home/ncu-caic/Documents/Coding/github.com/natsuki221/AGCH-Impl/_bmad-output/planning-artifacts/architecture.md#L173-L188)
- [Epic 3 Details](file:///home/ncu-caic/Documents/Coding/github.com/natsuki221/AGCH-Impl/_bmad-output/planning-artifacts/epics.md#L110-L129)

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
