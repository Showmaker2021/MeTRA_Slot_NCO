# Analysis Report: Milestone M0 ($d_{\text{ins}}$ Insertion Cost Operator)

**Author**: Explorer `explorer_m0_3`  
**Date**: 2026-08-06  
**Target Module**: `rl4co/data/insertion_cost.py` & `tests/test_insertion_cost.py`  

---

## Executive Summary

This report provides a comprehensive mathematical, tensor vectorization, device compatibility, precision, and interface contract investigation of the $d_{\text{ins}}$ insertion cost operator (`compute_marginal_insertion_cost`) for Milestone M0.

---

## 1. Mathematical Analysis & Definition Verification

### 1.1 Mathematical Formulation
In Vehicle Routing Problems (VRP) and Traveling Salesperson Problem (TSP) insertion heuristics, inserting node $i$ into a sub-tour containing node $j$ and Depot $D$ ($D \to j \to D$) increases the tour length by the marginal insertion cost:

$$d_{\text{ins}}(i, j) = \|x_D - x_i\|_2 + \|x_i - x_j\|_2 - \|x_D - x_j\|_2$$

Where:
- $x_i \in \mathbb{R}^2$: location coordinates of inserted customer node $i$.
- $x_j \in \mathbb{R}^2$: location coordinates of reference host node $j$.
- $x_D \in \mathbb{R}^2$: depot location coordinates.
- $\|\cdot\|_2$: Euclidean $L_2$ norm.

### 1.2 Fundamental Mathematical Properties

1. **Non-negativity (Triangle Inequality)**:
   $$\|x_D - x_i\|_2 + \|x_i - x_j\|_2 \ge \|x_D - x_j\|_2 \implies d_{\text{ins}}(i, j) \ge 0 \quad \forall i, j$$
   Strict equality $d_{\text{ins}}(i, j) = 0$ holds if and only if $i = j$ or node $i$ lies on the line segment $\overline{D j}$.

2. **Self-Insertion Diagonal**:
   $$d_{\text{ins}}(i, i) = \|x_D - x_i\|_2 + \|x_i - x_i\|_2 - \|x_D - x_i\|_2 = 0.0$$

3. **Directional Asymmetry**:
   $$d_{\text{ins}}(i, j) - d_{\text{ins}}(j, i) = 2 \left( \|x_D - x_i\|_2 - \|x_D - x_j\|_2 \right)$$
   $d_{\text{ins}}(i, j)$ is **asymmetric** whenever $\|x_D - x_i\|_2 \ne \|x_D - x_j\|_2$.
   - **Row index $i$**: inserted node.
   - **Column index $j$**: reference host node.

4. **Analytical Validation**:
   - Depot $D = (0,0)$, Node $i = (2,0)$, Node $j = (0,1)$.
   - $\|x_D - x_i\|_2 = 2.0$, $\|x_D - x_j\|_2 = 1.0$, $\|x_i - x_j\|_2 = \sqrt{5} \approx 2.236068$.
   - $d_{\text{ins}}(i, j) = 2.0 + 2.236068 - 1.0 = 3.236068$.
   - $d_{\text{ins}}(j, i) = 1.0 + 2.236068 - 2.0 = 1.236068$.
   - Difference $d_{\text{ins}}(i, j) - d_{\text{ins}}(j, i) = 2.0 = 2 \cdot (2.0 - 1.0)$.

---

## 2. Tensor Vectorization & Memory Scaling

### 2.1 Implementation Tracing
In `rl4co/data/insertion_cost.py`:

```python
dist_customers = compute_pairwise_distance_matrix(locs)
dist_depot = torch.norm(locs - depot_loc, p=2, dim=-1)
d_ins = dist_depot.unsqueeze(2) + dist_customers - dist_depot.unsqueeze(1)
```

Broadcasting shape analysis:
- `dist_depot.unsqueeze(2)`: shape `(B, N, 1)` (node $i$).
- `dist_customers`: shape `(B, N, N)` (nodes $i, j$).
- `dist_depot.unsqueeze(1)`: shape `(B, 1, N)` (node $j$).
- Result `d_ins`: shape `(B, N, N)`.

### 2.2 Memory Allocation & Scaling Benchmark ($N \in \{50, 100, 200, 500\}$)

| $N$ | Batch Size ($B$) | `coords` $(B,N,2)$ | `dist_customers` $(B,N,N)$ | Intermediate `diff` $(B,N,N,2)$ | Output `d_ins` $(B,N,N)$ | Peak Temp Memory |
|---|---|---|---|---|---|---|
| 50 | 64 | 25.6 KB | 640 KB | 1.28 MB | 640 KB | ~3.3 MB |
| 100 | 64 | 51.2 KB | 2.56 MB | 5.12 MB | 2.56 MB | ~13.4 MB |
| 200 | 64 | 102.4 KB | 10.24 MB | 20.48 MB | 10.24 MB | ~53.6 MB |
| 500 | 64 | 256 KB | 64 MB | 128 MB | 64 MB | ~335 MB |
| 500 | 512 | 2.05 MB | 512 MB | 1.024 GB | 512 MB | ~2.68 GB |
| 500 | 1024 | 4.10 MB | 1.024 GB | 2.048 GB | 1.024 GB | ~5.36 GB |

### 2.3 Optimization Opportunity: `torch.cdist`
In `compute_pairwise_distance_matrix`:
```python
diff = coords.unsqueeze(2) - coords.unsqueeze(1)
dist_matrix = torch.norm(diff, p=2, dim=-1)
```
Creating `diff` allocates an explicit $(B, N, N, 2)$ tensor.
Using `torch.cdist(coords, coords, p=2.0)` avoids allocating the intermediate `diff` tensor entirely, cutting peak memory by $>50\%$ for $N=500, B=512/1024$ and speeding up execution on CUDA via optimized BLAS kernels.

---

## 3. Sparsification, Masking & Edge Cases Analysis

### 3.1 $k$-NN Sparsification Mechanics
```python
if k_neighbors is not None and k_neighbors < N:
    _, knn_indices = torch.topk(dist_customers, k=k_neighbors + 1, dim=-1, largest=False)
    knn_mask = torch.zeros((B, N, N), dtype=torch.bool, device=device)
    knn_mask.scatter_(2, knn_indices, True)
    d_ins = d_ins.masked_fill(~knn_mask, float("inf"))
```

1. **Top-$k$ Selection Metric**:
   - Uses spatial customer distances `dist_customers` to select the $k+1$ closest nodes $j$ for each node $i$.
   - Includes self-loop $i=i$ automatically because `dist_customers[i, i] = 0.0`.
   - Leaves non-neighbors masked with `float("inf")`.

2. **Edge Cases**:
   - `k_neighbors is None`: `if` condition is `False`, returns dense `(B, N, N)`. Correct.
   - `k_neighbors >= N`: `k_neighbors < N` is `False`, returns dense `(B, N, N)`. Correct.
   - `k_neighbors == N - 1`: `k_neighbors + 1 == N`, returns dense `(B, N, N)`. Correct.
   - `k_neighbors <= 0`: Could cause `torch.topk` runtime error. Assertion or bound check recommended.

3. **Numerical Precision Safeguard**:
   In float32, `dist_depot.unsqueeze(2) + dist_customers - dist_depot.unsqueeze(1)` can produce tiny negative numbers (e.g. `-1e-7`) due to cancellation rounding errors.
   - **Recommendation**: Add `d_ins = torch.clamp(d_ins, min=0.0)` before diagonal zeroing and masking.

---

## 4. Interface Contracts & Downstream Compatibility

1. **Data Engine API (`rl4co/data/insertion_cost.py`)**:
   - `compute_marginal_insertion_cost(locs, k_neighbors=15, depot_loc=None)`
   - Accepts 2D `(N, 2)` or 3D `(B, N, 2)` tensors.
   - Returns sparsified tensor `(B, N, N)` or `(N, N)`.

2. **Offline Dataset Caching (`generate_slot_dataset.py`, M1)**:
   - Evaluates `compute_marginal_insertion_cost` once offline for $N \in \{50, 100, 200, 500\}$ and caches `.pt` files.

3. **METRA Metric Loss (`metric_loss.py`, M4 / Variant D)**:
   - Uses cached `d_ins` as ground truth metric matrix, masking out `inf` values during loss computation.

---

## 5. Recommended Refinements for Implementation Team

1. **Use `torch.cdist` in `compute_pairwise_distance_matrix`**:
   ```python
   def compute_pairwise_distance_matrix(coords: torch.Tensor) -> torch.Tensor:
       if coords.dim() == 2:
           return torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0), p=2.0).squeeze(0)
       elif coords.dim() == 3:
           return torch.cdist(coords, coords, p=2.0)
       else:
           raise ValueError(f"Expected coords dimension 2 or 3, got {coords.dim()}")
   ```
2. **Add `clamp(min=0.0)` for Float32 Non-negativity**:
   ```python
   d_ins = torch.clamp(d_ins, min=0.0)
   ```
3. **Robust `depot_loc` Dimension Handling**:
   Support 1D `(2,)`, 2D `(B, 2)` / `(1, 2)`, and 3D `(B, 1, 2)` depot tensors cleanly.
4. **Expand Unit Test Coverage in `tests/test_insertion_cost.py`**:
   Add analytical test cases (known coordinates and exact values), asymmetry checks, non-negativity checks, 2D unbatched input checks, and device/dtype preservation checks.
