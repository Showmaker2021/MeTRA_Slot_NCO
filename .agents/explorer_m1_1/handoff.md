# Handoff Report: Milestone M1 — Dataset Generator CLI & GMM Design

## 1. Observation

1. **Original Project Requirements (`d:/NCO NEW/rl4co/.agents/ORIGINAL_REQUEST.md`, lines 14-17)**:
   - "R1. Data Engine & Sparsified d_ins Cache (Milestones M0 & M1)"
   - "Offline Caching: Create `rl4co/data/generate_slot_dataset.py` to precompute and cache instances + sparsified d_ins matrices to disk (.pt format)."
   - "Data Distributions: Support both Uniform and Clustered (Gaussian Mixture) distributions for N in {50, 100, 200, 500}."

2. **Milestone Scope (`d:/NCO NEW/rl4co/.agents/sub_orch_m0_m1/SCOPE.md`, lines 6, 12)**:
   - "rl4co/data/generate_slot_dataset.py: CLI script for precomputing and saving Uniform & Clustered GMM datasets for $N \in \{50, 100, 200, 500\}$ in `.pt` format."
   - Status of M1: IN_PROGRESS.

3. **Insertion Cost API (`d:/NCO NEW/rl4co/rl4co/data/insertion_cost.py`, lines 36-40)**:
   - Function signature:
     ```python
     def compute_marginal_insertion_cost(
         locs: torch.Tensor,
         k_neighbors: Optional[int] = 15,
         depot_loc: Optional[torch.Tensor] = None,
     ) -> torch.Tensor:
     ```
   - Input `locs`: `(B, N, 2)`, `depot_loc`: `(B, 2)`, returns `d_ins`: `(B, N, N)` where non-$k$-NN neighbors are `inf`.

4. **Existing Distribution Utils (`d:/NCO NEW/rl4co/rl4co/envs/common/distribution_utils.py`, lines 6-50)**:
   - `Cluster` class defines cluster centers in `[0.2, 0.8]`, `std=0.07`, clamping to `[0, 1]`.

---

## 2. Logic Chain

1. **Observation 1 & 2** establish that `rl4co/data/generate_slot_dataset.py` must be a standalone CLI script generating `.pt` files containing precomputed `locs`, `depot`, and $k$-NN sparsified `d_ins` matrices.
2. **Observation 3** shows that `compute_marginal_insertion_cost` in `rl4co/data/insertion_cost.py` accepts `locs`, `depot_loc`, and `k_neighbors`, directly returning `d_ins` of shape `(B, N, N)` with non-$k$-NN elements set to `inf`. Thus, `generate_slot_dataset.py` can directly call `compute_marginal_insertion_cost` without reimplementing sparsification or distance calculations.
3. **Observation 4** provides the pattern for GMM cluster generation: sampling cluster centers in $[0.2, 0.8]^2$, standard deviations in $[0.05, 0.10]$, and clamping final coordinates to $[0.0, 1.0]$.
4. Combining these observations leads to the complete CLI specification (`--output_dir`, `--distributions`, `--graph_sizes`, `--num_samples`, `--k_neighbors`, `--seed`), GMM logic, and `.pt` PyTorch dictionary format (`{"locs": ..., "depot": ..., "d_ins": ...}`).

---

## 3. Caveats

- **RAM / Memory Limits for $N=500$**: A single batch of 10,000 instances with $N=500$ produces a `d_ins` tensor of size $10000 \times 500 \times 500 \times 4$ bytes $\approx 10$ GB. Generating in smaller sub-batches (e.g., 1000 samples at a time) is required to prevent CPU OOM during dataset creation.
- **Device Placement**: While dataset generation can utilize GPU (`cuda`) for fast top-$k$ and insertion cost computation, the final dictionary saved to disk via `torch.save` should ensure all tensors are moved to CPU (`.cpu()`).

---

## 4. Conclusion

The specification for `rl4co/data/generate_slot_dataset.py` is fully designed and ready for implementation by the implementer agent. It includes:
- Standard CLI interface supporting all required parameters (`output_dir`, `distributions`, `graph_sizes`, `num_samples`, `k_neighbors`, `seed`, `overwrite`, `device`).
- GMM clustered coordinate generator (3-5 clusters, std 0.05-0.10, centers $[0.2, 0.8]^2$, clamped to $[0, 1]^2$).
- Standard `.pt` file dictionary containing Float32 Tensors `locs`, `depot`, `d_ins`.

---

## 5. Verification Method

Once implemented by the implementer agent:
1. **Directory & Code Inspection**:
   - Verify `rl4co/data/generate_slot_dataset.py` exists and has `__name__ == "__main__"` CLI entrypoint.
2. **Execution Test**:
   ```bash
   python rl4co/data/generate_slot_dataset.py --output_dir data/slot_test --distributions uniform clustered --graph_sizes 50 --num_samples 10 --k_neighbors 15 --seed 1234 -f
   ```
3. **Data Integrity Inspection via Python**:
   ```python
   import torch
   data = torch.load("data/slot_test/uniform_n50_k15_seed1234.pt")
   assert data["locs"].shape == (10, 50, 2)
   assert data["depot"].shape == (10, 2)
   assert data["d_ins"].shape == (10, 50, 50)
   assert torch.isinf(data["d_ins"]).sum() > 0
   ```
