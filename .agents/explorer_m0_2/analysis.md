# Analysis and Comprehensive Test Specification: Milestone M0 ($d_{\text{ins}}$ Operator Unit Tests)

**Author:** `explorer_m0_2`  
**Date:** 2026-08-06  
**Target File under Test:** `rl4co/data/insertion_cost.py`  
**Target Test Suite:** `tests/test_insertion_cost.py`  
**Milestone:** M0 ($d_{\text{ins}}$ insertion cost operator unit tests)

---

## 1. Executive Summary

Milestone M0 specifies the design and implementation of the vectorized $k$-NN sparsified marginal insertion cost operator $d_{\text{ins}}(i, j)$ in `rl4co/data/insertion_cost.py`, along with a comprehensive, rigorous unit test suite in `tests/test_insertion_cost.py`.

The current test file `tests/test_insertion_cost.py` contains only 3 high-level basic tests (totaling 39 lines), which cover basic shape checks and simple sanity checks. It lacks coverage for:
- Mathematical correctness against analytical hand-calculated reference values.
- Default ($k=15$) sparsification verification and exact neighbor selection logic.
- Self-insertion cost $d_{\text{ins}}(i, i) = 0.0$ persistence under $k$-NN masking.
- Non-neighbor `inf` masking assertion.
- Edge cases ($N \le k$, $N=1, 2, 5$, unbatched 2D inputs vs batched 3D inputs, $B>1$, random seeds, custom depot positions).
- Mathematical invariants (symmetry of pairwise distances, non-negativity of insertion costs, triangle inequality).

This report presents a full gap analysis of existing tests and provides a complete, production-grade test specification containing **16 distinct test cases** organized into 5 test classes.

---

## 2. Analysis of Existing Test Coverage

### 2.1 Code under Test Analysis (`rl4co/data/insertion_cost.py`)

The module exposes two main functions:
1. `compute_pairwise_distance_matrix(coords: torch.Tensor) -> torch.Tensor`
   - Input: `(B, N, 2)` or `(N, 2)`
   - Computes Euclidean distance matrix using `torch.norm(coords.unsqueeze(2) - coords.unsqueeze(1), p=2, dim=-1)`.
   - Returns: `(B, N, N)` or `(N, N)`.
2. `compute_marginal_insertion_cost(locs: torch.Tensor, k_neighbors: Optional[int] = 15, depot_loc: Optional[torch.Tensor] = None) -> torch.Tensor`
   - Input: customer locations `(B, N, 2)` or `(N, 2)`, optional `k_neighbors` (default 15), optional `depot_loc`.
   - Default depot position: `(0.5, 0.5)` if `depot_loc is None`.
   - Formula: $d_{\text{ins}}(i, j) = \text{dist}(D, i) + \text{dist}(i, j) - \text{dist}(D, j)$.
   - Vectorized computation: `d_ins = dist_depot.unsqueeze(2) + dist_customers - dist_depot.unsqueeze(1)`.
   - Explicit zero-fill on diagonal: `d_ins[b, i, i] = 0.0`.
   - $k$-NN Sparsification: if `k_neighbors is not None and k_neighbors < N`:
     - Selects top $k+1$ smallest distance entries in `dist_customers` per row (node $i$ itself is smallest distance 0.0, plus $k$ nearest neighbors).
     - Sparsifies non-neighbors to `float("inf")`.

### 2.2 Existing Test Suite Evaluation (`tests/test_insertion_cost.py`)

| Existing Test Function | Tested Aspect | Code Covered | Critical Coverage Gaps |
|-----------------------|---------------|--------------|------------------------|
| `test_compute_pairwise_distance_matrix` | 3-4-5 right triangle points (unbatched) | `compute_pairwise_distance_matrix` | Batched inputs `(B, N, 2)`, symmetry $D_{ij} = D_{ji}$, zero diagonal $D_{ii} = 0$, triangle inequality $D_{ij} \le D_{ik} + D_{kj}$. |
| `test_marginal_insertion_cost_basic` | $B=2, N=5, k=\text{None}$ shape & self-insertion | `compute_marginal_insertion_cost` | Mathematical validation of $d_{\text{ins}}(i,j)$, custom depot coordinates, non-negativity, asymmetry of $d_{\text{ins}}$. |
| `test_knn_sparsification` | $B=1, N=10, k=3$ max count per row | $k$-NN masking condition | Default $k=15$, exact `inf` mask verification, neighbor identity correctness, $N \le k$ behavior, edge sizes $N \in \{1, 2, 5\}$, multi-batch $B>1$, random seed reproducibility. |

---

## 3. Mathematical & Algorithmic Specifications

### 3.1 Mathematical Formulation of Insertion Cost $d_{\text{ins}}(i, j)$

For customer node coordinates $x_1, \dots, x_N \in \mathbb{R}^2$ and depot location $x_D \in \mathbb{R}^2$:
- Pairwise customer distance: $d(i, j) = \|x_i - x_j\|_2$.
- Depot distance: $d(D, i) = \|x_D - x_i\|_2$.
- Marginal insertion cost:
  $$d_{\text{ins}}(i, j) = d(D, i) + d(i, j) - d(D, j)$$

#### Mathematical Properties to Test:
1. **Self-Insertion Identity:** $d_{\text{ins}}(i, i) = d(D, i) + 0 - d(D, i) = 0.0$.
2. **Asymmetry:** In general, $d_{\text{ins}}(i, j) \neq d_{\text{ins}}(j, i)$ when $d(D, i) \neq d(D, j)$. Specifically:
   $$d_{\text{ins}}(i, j) - d_{\text{ins}}(j, i) = 2 (d(D, i) - d(D, j))$$
3. **Non-Negativity:** By triangle inequality on $x_D, x_i, x_j$:
   $$d(D, i) + d(i, j) \ge d(D, j) \implies d_{\text{ins}}(i, j) \ge 0.0$$
4. **$k$-NN Sparsification Rule:**
   $$\tilde{d}_{\text{ins}}(i, j) = \begin{cases} 0.0 & \text{if } i = j \\ d_{\text{ins}}(i, j) & \text{if } j \in \text{kNN}(i) \\ \infty & \text{otherwise} \end{cases}$$
   where $\text{kNN}(i)$ are the $k$ closest nodes to node $i$ under Euclidean distance $d(i, j)$.
5. **No Sparsification Threshold:** When $N \le k$, `k_neighbors < N` is False, so $\tilde{d}_{\text{ins}}(i, j) = d_{\text{ins}}(i, j)$ for all $i, j$.

---

## 4. Comprehensive Test Suite Specification

The expanded test suite will consist of 5 specialized test classes containing 16 test functions:

```
tests/test_insertion_cost.py
├── TestComputePairwiseDistanceMatrix
│   ├── test_hand_crafted_triangle_unbatched
│   ├── test_batched_distance_matrix
│   ├── test_diagonal_zero_and_symmetry
│   └── test_triangle_inequality
├── TestComputeMarginalInsertionCostBasic
│   ├── test_analytical_insertion_cost_values
│   ├── test_asymmetry_property
│   ├── test_non_negativity_property
│   └── test_depot_location_variants
├── TestKNNSparsification
│   ├── test_default_k15_sparsification_n50
│   ├── test_exact_neighbor_selection_and_inf_masking
│   ├── test_k_zero_boundary
│   └── test_n_less_than_or_equal_k
├── TestInsertionCostEdgeCases
│   ├── test_single_customer_n1
│   ├── test_two_customers_n2
│   └── test_small_customer_n5
└── TestInsertionCostSystemIntegrity
    ├── test_unbatched_vs_batched_consistency
    ├── test_random_seed_reproducibility
    └── test_dtype_and_device_preservation
```

### 4.1 Detailed Test Scenario Specifications

#### Suite 1: `TestComputePairwiseDistanceMatrix`
- **`test_hand_crafted_triangle_unbatched`**:
  - Input: 2D coordinates `[(0, 0), (3, 0), (0, 4)]` (shape `(3, 2)`).
  - Expected Matrix: `[[0.0, 3.0, 4.0], [3.0, 0.0, 5.0], [4.0, 5.0, 0.0]]`.
- **`test_batched_distance_matrix`**:
  - Input: Shape `(2, 3, 2)` where batch 0 is 3-4-5 triangle and batch 1 is collinear points `[(0, 0), (1, 0), (3, 0)]`.
  - Batch 1 expected pairwise distances: `[[0, 1, 3], [1, 0, 2], [3, 2, 0]]`.
- **`test_diagonal_zero_and_symmetry`**:
  - Generate random `coords` shape `(4, 20, 2)`.
  - Assert `torch.allclose(torch.diagonal(dist, dim1=-2, dim2=-1), torch.zeros(4, 20))`.
  - Assert `torch.allclose(dist, dist.transpose(-1, -2))`.
- **`test_triangle_inequality`**:
  - For random coords `(2, 10, 2)`, assert for all $i, j, k$: `dist[b, i, j] <= dist[b, i, k] + dist[b, k, j] + 1e-6`.

#### Suite 2: `TestComputeMarginalInsertionCostBasic`
- **`test_analytical_insertion_cost_values`**:
  - Depot $D = (0, 0)$, Node 1 $A = (3, 0)$, Node 2 $B = (0, 4)$.
  - $d(D, A) = 3$, $d(D, B) = 4$, $d(A, B) = 5$.
  - Theoretical $d_{\text{ins}}(A, B) = 3 + 5 - 4 = 4.0$.
  - Theoretical $d_{\text{ins}}(B, A) = 4 + 5 - 3 = 6.0$.
  - Assert computed output tensor matches `[[0.0, 4.0], [6.0, 0.0]]` with `atol=1e-5`.
- **`test_asymmetry_property`**:
  - Verify $d_{\text{ins}}(A, B) - d_{\text{ins}}(B, A) = 4.0 - 6.0 = -2.0 \neq 0$.
- **`test_non_negativity_property`**:
  - For 100 random instances `(B=5, N=20, 2)`, assert `torch.all(d_ins[~torch.isinf(d_ins)] >= -1e-6)`.
- **`test_depot_location_variants`**:
  - Compare default `depot_loc=None` against `depot_loc=torch.tensor([0.5, 0.5])`.
  - Test custom depot at `(0.0, 0.0)` and `(1.0, 1.0)`.
  - Test depot tensor shapes: 1D `(2,)`, 2D `(1, 2)`, 3D `(B, 1, 2)`.

#### Suite 3: `TestKNNSparsification`
- **`test_default_k15_sparsification_n50`**:
  - Input: $B=2, N=50, k=15$.
  - Verify output shape `(2, 50, 50)`.
  - Count finite entries per row: `torch.sum(~torch.isinf(d_ins), dim=-1)`. Must be EXACTLY $k+1 = 16$ for every row.
  - Count `inf` entries per row: must be EXACTLY $N - (k+1) = 34$.
- **`test_exact_neighbor_selection_and_inf_masking`**:
  - Create 10 nodes along a straight line $x_i = (i, 0)$ for $i=0..9$. Depot $D = (-5, 0)$.
  - For node 0 $(0, 0)$, nearest neighbors under $k=2$ are node 0 (dist 0), node 1 (dist 1), node 2 (dist 2).
  - Verify `d_ins[0, 0, 0]`, `d_ins[0, 0, 1]`, `d_ins[0, 0, 2]` are finite, and `d_ins[0, 0, 3..9]` are `inf`.
- **`test_k_zero_boundary`**:
  - Input: $N=10, k=0$.
  - Exactly 1 non-inf entry per row (self-insertion $d_{\text{ins}}(i, i) = 0.0$).
  - All off-diagonal entries are `inf`.
- **`test_n_less_than_or_equal_k`**:
  - Test $N=5, k=15$ and $N=15, k=15$.
  - Assert zero `inf` entries (all $N \times N$ matrix elements are finite).

#### Suite 4: `TestInsertionCostEdgeCases`
- **`test_single_customer_n1`**:
  - Input: `locs` shape `(B=2, N=1, 2)`.
  - Output shape `(2, 1, 1)`.
  - Value: `d_ins[b, 0, 0] == 0.0`. Zero `inf` entries.
- **`test_two_customers_n2`**:
  - Input: `locs` shape `(B=3, N=2, 2)`, $k=15$.
  - Output shape `(3, 2, 2)`. Zero `inf` entries. Self-insertions are 0.0.
- **`test_small_customer_n5`**:
  - Input: `locs` shape `(B=2, N=5, 2)`.
  - Compare $k=2$ (sparsified, 3 non-inf per row) vs $k=15$ (dense, 5 non-inf per row).

#### Suite 5: `TestInsertionCostSystemIntegrity`
- **`test_unbatched_vs_batched_consistency`**:
  - `locs_unbatched` shape `(10, 2)`. `locs_batched` shape `(1, 10, 2)`.
  - Verify `compute_marginal_insertion_cost(locs_unbatched)` has shape `(10, 10)`.
  - Verify `torch.allclose(d_ins_unbatched, d_ins_batched.squeeze(0))`.
- **`test_random_seed_reproducibility`**:
  - Generate inputs under `torch.manual_seed(42)` in two separate calls. Assert outputs are identical.
- **`test_dtype_and_device_preservation`**:
  - Verify `torch.float32` -> output `torch.float32`.
  - Verify `torch.float64` -> output `torch.float64`.

---

## 5. Verification Plan & Test Commands

To verify the test suite:
1. Environment setup: execute using `conda run -n ec_nco python -m pytest tests/test_insertion_cost.py -v`.
2. All 16 tests must pass cleanly without warnings or errors.
3. Test execution time should be under 2.0 seconds.

