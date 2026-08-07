# Scope: Milestone M0 & M1 (Requirement R1 — Data Engine & Sparsified d_ins Cache)

## Architecture
1. `rl4co/data/insertion_cost.py`: `compute_pairwise_distance_matrix`, `compute_marginal_insertion_cost` with $k$-NN sparsification masking non-neighbors to `inf`.
2. `tests/test_insertion_cost.py`: Unit tests for $d_{\text{ins}}$, $k$-NN sparsification, edge cases.
3. `rl4co/data/generate_slot_dataset.py`: CLI script for precomputing and saving Uniform & Clustered GMM datasets for $N \in \{50, 100, 200, 500\}$ in `.pt` format.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 0 | M0: $d_{\text{ins}}$ Operator & Unit Tests | `rl4co/data/insertion_cost.py`, `tests/test_insertion_cost.py` | none | DONE |
| 1 | M1: Offline Dataset Generator CLI | `rl4co/data/generate_slot_dataset.py` saving `.pt` datasets | M0 | IN_PROGRESS |


## Code Layout
- `rl4co/data/insertion_cost.py`
- `rl4co/data/generate_slot_dataset.py`
- `tests/test_insertion_cost.py`
