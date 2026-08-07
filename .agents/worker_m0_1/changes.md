# Summary of Changes — Milestone M0

**Agent**: `worker_m0_1`  
**Date**: 2026-08-06  

## 1. Refined `rl4co/data/insertion_cost.py`
- **Pairwise Distance Matrix (`compute_pairwise_distance_matrix`)**:
  - Upgraded distance calculation from manual `(coords.unsqueeze(2) - coords.unsqueeze(1))` to `torch.cdist(coords, coords, p=2.0)`.
  - Preserved 2D squeezing logic to return `(N, N)` for unbatched 2D `(N, 2)` inputs and `(B, N, N)` for 3D `(B, N, 2)` batched inputs.
  - Achieved higher memory efficiency and speed by eliminating intermediate `(B, N, N, 2)` tensor allocation.
- **Marginal Insertion Cost (`compute_marginal_insertion_cost`)**:
  - Vectorized exact marginal insertion cost $d_{\text{ins}}(i, j) = c(D, i) + c(i, j) - c(D, j)$.
  - Added `torch.clamp(d_ins, min=0.0)` to eliminate float32 precision round-off underflow below zero.
  - Set self-insertion diagonal to 0.0 via `masked_fill(eye_mask, 0.0)`.
  - Sparsified to $k$-nearest neighbors (default $k=15$) using `torch.topk(dist_customers, k=k_neighbors + 1, dim=-1, largest=False)` and `masked_fill(~knn_mask, float('inf'))`.
  - Guarded $N \le k$ edge cases (`if k_neighbors is not None and k_neighbors < N:`) to skip sparsification and return a dense matrix when $N \le k$.
  - Expanded `depot_loc` parameter handling to support 1D `(2,)`, 2D `(1, 2)` or `(B, 2)`, and 3D `(B, 1, 2)` shapes gracefully.

## 2. Refined & Expanded `tests/test_insertion_cost.py`
- **`test_compute_pairwise_distance_matrix`**: Tested 2D `(N, 2)` and 3D `(B, N, 2)` coordinate tensors, verifying 3-4-5 right triangle distances (3.0, 4.0, 5.0) and batch scaling.
- **`test_marginal_insertion_cost_basic`**: Verified output shape `(B, N, N)`, zero diagonal self-insertion $d_{\text{ins}}[b, i, i] == 0.0$, non-negativity $d_{\text{ins}} \ge 0.0$, and formula accuracy against analytical distance calculations.
- **`test_knn_sparsification`**: Tested default $k=15$ on $N=20$ (verifying exactly 16 non-inf entries per node and 80 inf entries overall) and $k=3$ on $N=10$ (verifying exactly 4 non-inf entries per node and 60 inf entries overall).
- **`test_edge_cases`**: Tested $N \le k$ ($N=5, k=15$, verifying dense non-inf output), 2D unbatched input `(N, 2)` returning 2D matrix `(N, N)`, $k=\text{None}$ returning dense matrix, and custom `depot_loc` shapes (1D, 2D, 3D).
- **Additional Test Coverage**:
  - `test_customer_at_depot_and_colocation`: Tested co-located nodes and customer at depot without zero-division/NaN errors.
  - `test_gradient_flow_insertion_cost`: Verified PyTorch autograd backpropagation through `locs` with non-zero gradients.
  - `test_clustered_spatial_distribution`: Verified inter-cluster sparsification on Gaussian Mixture clusters.

## 3. Package Initialization (`rl4co/__init__.py`)
- Added robust try/except fallback for `PackageNotFoundError` when fetching `__version__`.

## 4. Test Verification
- Executed `D:\Miniconda\miniconda3\envs\ec_nco\python.exe -m pytest tests/test_insertion_cost.py -v` in `d:/NCO NEW/rl4co`.
- Result: All 7 unit tests passed cleanly (7/7 PASSED in 3.97s).
