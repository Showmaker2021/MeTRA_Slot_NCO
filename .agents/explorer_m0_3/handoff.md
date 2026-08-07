# Handoff Report: Milestone M0 ($d_{\text{ins}}$ Insertion Cost Operator)

**Author**: Explorer `explorer_m0_3`  
**Working Directory**: `d:/NCO NEW/rl4co/.agents/explorer_m0_3`  
**Date**: 2026-08-06  

---

## 1. Observation

1. **Existing File Locations & Line Content**:
   - `rl4co/data/insertion_cost.py`:
     - Line 22: `diff = coords.unsqueeze(2) - coords.unsqueeze(1)`
     - Line 23: `dist_matrix = torch.norm(diff, p=2, dim=-1)`
     - Lines 79–81: `d_ins = (dist_depot.unsqueeze(2) + dist_customers - dist_depot.unsqueeze(1))`
     - Lines 84–85: `eye_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0); d_ins = d_ins.masked_fill(eye_mask, 0.0)`
     - Lines 88–92: `if k_neighbors is not None and k_neighbors < N: _, knn_indices = torch.topk(dist_customers, k=k_neighbors + 1, dim=-1, largest=False); knn_mask = torch.zeros((B, N, N), dtype=torch.bool, device=device); knn_mask.scatter_(2, knn_indices, True); d_ins = d_ins.masked_fill(~knn_mask, float("inf"))`
   - `tests/test_insertion_cost.py`:
     - Contains 3 basic tests: `test_compute_pairwise_distance_matrix`, `test_marginal_insertion_cost_basic`, and `test_knn_sparsification`.

2. **Project Specification Contracts**:
   - `PROJECT.md` line 37: `compute_marginal_insertion_cost(locs, k_neighbors=15, depot_loc=None)` -> returns sparsified `(B, N, N)` tensor with `inf` non-neighbors.
   - `ORIGINAL_REQUEST.md` line 15: "k-NN Sparsification: Vectorized cheapest insertion d_ins(i,j) restricted to k-nearest neighbors (default k=15) in `rl4co/data/insertion_cost.py`."

3. **Memory & Scaling Behavior**:
   - For $N=500, B=512$, allocating intermediate `diff` tensor `(B, N, N, 2)` consumes $512 \times 500 \times 500 \times 2 \times 4 \text{ bytes} = 1.024 \text{ GB}$ of RAM/VRAM.
   - Dense sparsified `d_ins` tensor `(B, N, N)` consumes $512 \text{ MB}$.

---

## 2. Logic Chain

1. **Step 1 (Mathematical Soundness)**:
   - From Observation 1 (lines 79–81), `d_ins[b, i, j] = dist_depot[b, i] + dist_customers[b, i, j] - dist_depot[b, j]`.
   - This matches $\|x_D - x_i\|_2 + \|x_i - x_j\|_2 - \|x_D - x_j\|_2$, which is the exact marginal insertion cost of adding node $i$ into a sub-tour containing node $j$ and depot $D$.
   - The formulation is directional and asymmetric ($d_{\text{ins}}(i, j) - d_{\text{ins}}(j, i) = 2(\|x_D - x_i\|_2 - \|x_D - x_j\|_2)$).
   - By the triangle inequality, $d_{\text{ins}}(i, j) \ge 0$.

2. **Step 2 (Tensor Vectorization & Memory Efficiency)**:
   - From Observation 1 (lines 22–23) and Observation 3, explicit calculation via `coords.unsqueeze(2) - coords.unsqueeze(1)` creates an intermediate tensor of shape `(B, N, N, 2)`.
   - Replacing this with `torch.cdist(coords, coords, p=2.0)` computes pairwise Euclidean distances natively in PyTorch C++/CUDA kernels without generating the intermediate `(B, N, N, 2)` tensor, reducing peak memory by over 50%.

3. **Step 3 (Float32 Precision & Masking Stability)**:
   - Floating-point subtraction in `a + b - c` can produce tiny negative numbers (e.g. `-1e-7`) due to float32 round-off.
   - Adding `d_ins = torch.clamp(d_ins, min=0.0)` guarantees mathematical non-negativity.
   - From Observation 1 (lines 88–92), $k$-NN sparsification correctly selects $k+1$ nearest spatial neighbors using `dist_customers` top-$k$ on `dim=-1`, keeping self-loop $i=i$ (0.0 cost) intact and setting non-neighbors to `float("inf")`.

4. **Step 4 (Interface Alignment)**:
   - The function signature and return types match the contract defined in `PROJECT.md` (Observation 2).

---

## 3. Caveats

- **Runtime Execution Permission**: CLI execution via `run_command` timed out waiting for user confirmation; static verification, mathematical proofs, and shape tensor derivations were performed.
- **Assumptions**: Assumes input coordinate tensors `locs` use standard Euclidean metric $L_2$ in 2D plane ($\mathbb{R}^2$), consistent with standard NCO problems (TSP/CVRP).

---

## 4. Conclusion

1. **Mathematical Accuracy**: The current $d_{\text{ins}}(i, j)$ implementation in `rl4co/data/insertion_cost.py` is mathematically correct and satisfies triangle inequality non-negativity, self-insertion diagonal zeroing, and directional asymmetry.
2. **Vectorization & Efficiency**: `compute_marginal_insertion_cost` is fully vectorized. Upgrading `compute_pairwise_distance_matrix` to use `torch.cdist` will optimize memory efficiency for $N=500$.
3. **Robustness Enhancements**: Adding `torch.clamp(d_ins, min=0.0)` prevents float32 numerical underflow below zero, and handling 1D/2D/3D `depot_loc` shapes ensures total API safety.
4. **Test Expansion**: Comprehensive analytical unit tests should be added to `tests/test_insertion_cost.py` for Milestone M0 completion.

---

## 5. Verification Method

To verify these findings and implementation recommendations independently:

1. **Inspect Source Files**:
   - `rl4co/data/insertion_cost.py`
   - `tests/test_insertion_cost.py`
   - `d:/NCO NEW/rl4co/.agents/explorer_m0_3/analysis.md`

2. **Run PyTorch Unit Tests**:
   - Command: `pytest tests/test_insertion_cost.py`
   - Invalidation conditions: Any test failure, shape mismatch, non-zero diagonal, or negative $d_{\text{ins}}$ values.

3. **Verify Tensor Memory & Output**:
   - Execute Python snippet:
     ```python
     import torch
     from rl4co.data.insertion_cost import compute_marginal_insertion_cost
     locs = torch.rand(64, 500, 2)
     d_ins = compute_marginal_insertion_cost(locs, k_neighbors=15)
     assert d_ins.shape == (64, 500, 500)
     assert torch.all(d_ins >= 0)
     ```
