# Project: Metric-Aware Slot Abstraction NCO Method (`rl4co`)

## Architecture
The Metric-Aware Slot Abstraction NCO method integrates into `rl4co` via four modular layers:
1. **Data Engine (`rl4co/data/`)**: Vectorized $k$-NN sparsified $d_{\text{ins}}$ insertion cost matrix computation in `insertion_cost.py` and offline `.pt` dataset generator `generate_slot_dataset.py`.
2. **Neural Abstraction Modules (`rl4co/models/nn/`)**: Differentiable `SlotAttention` in `slot_attention.py` producing $K$ slot embeddings $z_k$ and soft assignment maps $A_{ik}$; METRA `MetricLoss` module in `metric_loss.py` implementing projection head $\phi(z_k)$, Lagrangian lower-bound dual ascent, and slot entropy regularization.
3. **POMO Model & Policy Wiring (`rl4co/models/zoo/pomo_slot/`)**: `POMOSlotPolicy` in `policy.py` conditioning POMO decoder queries with aggregated slot representations $\hat{z}_i = \sum_k A_{ik} z_k$; `POMOSlot` LightningModule in `model.py` integrating task & metric losses with clean toggles for Variants A through E.
4. **Hydra Configs & Benchmarks (`configs/model/` & `scripts/`)**: YAML configuration specs `pomo_slot_a.yaml` through `pomo_slot_e.yaml` (also linked in `conf/model/`); multi-seed evaluation script `scripts/eval_pomo_slot.py` for Decision Checkpoint M8.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | $k$-NN Sparsified $d_{\text{ins}}$ Operator | Vectorized cheapest insertion cost $d_{\text{ins}}(i,j)$ restricted to $k$-nearest neighbors (default $k=15$) in `rl4co/data/insertion_cost.py`. | M0 | R1 |
| 2 | Unit Test Insertion Cost | Comprehensive unit test suite in `tests/test_insertion_cost.py` for $d_{\text{ins}}$ and $k$-NN sparsification. | M0 | R1 |
| 3 | Offline Dataset Caching CLI | Precompute and cache Uniform & Clustered GMM instances + $d_{\text{ins}}$ matrices to disk (`.pt` format) for $N \in \{50, 100, 200, 500\}$ in `rl4co/data/generate_slot_dataset.py`. | M1 | R1 |
| 4 | Standalone `SlotAttention` Layer | Differentiable Slot Attention module in `rl4co/models/nn/slot_attention.py` with GRU refinement and softmax/L1 normalization. | M2 | R2 |
| 5 | Unit Test Slot Attention | Comprehensive unit tests in `tests/test_slot_attention.py` asserting shape $(B, K, d)$ and $\sum_k A_{ik} = 1.0$. | M2 | R2 |
| 6 | POMO Policy Wiring (Variant B) | Wire slot embeddings $z_k$ into POMO decoder conditioning in `rl4co/models/zoo/pomo_slot/policy.py` (Variant B task-only loss). | M3 | R2 |
| 7 | METRA Metric Loss & Dual Ascent | Dual ascent stability, projection head $\phi(z_k)$, Lagrangian lower-bound constraint, dual parameter $\lambda$ update, slot entropy in `rl4co/models/nn/metric_loss.py`. | M4 | R3 |
| 8 | Unit Test Metric Loss | Comprehensive unit test suite in `tests/test_metric_loss.py` for projection, dual ascent, and entropy. | M4 | R3 |
| 9 | Model Variant Toggles (A-E) | LightningModule variant toggles (A: Reconstruction, B: Task-Only, C: Euclidean, D: Insertion Cost, E: Future Regret) in `rl4co/models/zoo/pomo_slot/model.py`. | M5 | R3 |
| 10 | Model Variant CLI Execution | End-to-end single command execution and validation across all variants A through E. | M6 | R3 |
| 11 | Hydra Configurations (A-E) | Hydra YAML config files `pomo_slot_a.yaml` through `pomo_slot_e.yaml` under `conf/model/` and `configs/model/`. | M7 | R4 |
| 12 | M8 Decision Gate Benchmark | Evaluation script `scripts/eval_pomo_slot.py` comparing A-E on $N=50$ across multi-seed metrics (optimality gap, ARI stability, slot entropy). | M8 | R4 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 0 | E2E Testing Track | Requirement-driven test suite & infrastructure (`TEST_INFRA.md`, `TEST_READY.md`) | none | PLANNED |
| 1 | M0 & M1: Data Engine & Sparsified Cache | `insertion_cost.py`, `generate_slot_dataset.py`, `tests/test_insertion_cost.py` | none | PLANNED |
| 2 | M2 & M3: Slot Attention & POMO Policy Wiring | `slot_attention.py`, `pomo_slot/policy.py`, `tests/test_slot_attention.py` | M1 | PLANNED |
| 3 | M4, M5, M6: METRA Loss & Model Toggles | `metric_loss.py`, `pomo_slot/model.py`, `tests/test_metric_loss.py` | M2 | PLANNED |
| 4 | M7 & M8: Hydra Configs & Decision Checkpoint | `conf/model/pomo_slot_*.yaml`, `scripts/eval_pomo_slot.py` | M3 | PLANNED |

## Interface Contracts
### Data Engine ↔ Slot Attention / Metric Loss
- `compute_marginal_insertion_cost(locs, k_neighbors=15, depot_loc=None)` -> returns sparsified `(B, N, N)` tensor with `inf` non-neighbors.

### Slot Attention ↔ Decoder Conditioning
- `SlotAttention(inputs)` -> returns `slots` `(B, K, d)` and `attn` `(B, N, K)`.
- `POMOSlotDecoder` aggregates node slots $\hat{z}_i = \sum_k A_{ik} z_k$ and conditions decoder glimpse query.

### Metric Loss ↔ POMO Model
- `MetricLoss(slots, attn, target_dist)` -> computes projection $\phi(z_k)$, dual update for $\lambda$, slot entropy, and returns `loss_metric`, `loss_entropy`, `dual_penalty`.

## Code Layout
- `rl4co/data/insertion_cost.py`
- `rl4co/data/generate_slot_dataset.py`
- `rl4co/models/nn/slot_attention.py`
- `rl4co/models/nn/metric_loss.py`
- `rl4co/models/zoo/pomo_slot/policy.py`
- `rl4co/models/zoo/pomo_slot/model.py`
- `configs/model/pomo_slot_a.yaml` ... `configs/model/pomo_slot_e.yaml`
- `conf/model/pomo_slot_a.yaml` ... `conf/model/pomo_slot_e.yaml`
- `scripts/eval_pomo_slot.py`
- `tests/test_insertion_cost.py`
- `tests/test_slot_attention.py`
- `tests/test_metric_loss.py`
