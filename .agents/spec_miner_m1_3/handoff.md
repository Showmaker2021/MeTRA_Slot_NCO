# Handoff Report: Specification Mining for Metric-Aware Slot Abstraction NCO (Milestone 1)

**Agent ID**: `spec_miner_m1_3`  
**Working Directory**: `d:/NCO NEW/rl4co/.agents/spec_miner_m1_3`  
**Parent Conversation ID**: `22b0ce59-1866-4433-a314-3dc905457e22`  
**Date**: 2026-08-06  

---

## Features Discovered

| # | Category | Feature | Description | Inputs | Outputs | Error Behavior | Discovered Via |
|---|----------|---------|-------------|--------|---------|----------------|----------------|
| 1 | Data Engine | Pairwise Distance Matrix Computation | Computes pairwise Euclidean distances between $N$ node coordinates. | `coords`: $(B, N, 2)$ or $(N, 2)$ tensor, `torch.float32` | `dist_matrix`: $(B, N, N)$ or $(N, N)$ tensor | Raises `ValueError` / `RuntimeError` if tensor dimension is not 2 or 3. | `rl4co/data/insertion_cost.py` |
| 2 | Data Engine | $d_{\text{ins}}$ Marginal Insertion Cost | Computes pairwise marginal insertion cost matrix $d_{\text{ins}}(i, j)$ with depot reference. | `locs`: $(B, N, 2)$, `k_neighbors`: `Optional[int]`, `depot_loc`: `Optional[Tensor]` | `d_ins`: $(B, N, N)$ tensor | Handles zero depot distance gracefully; sets diagonal $d_{\text{ins}}(i,i)=0$. | `ORIGINAL_REQUEST.md` R1, `PROJECT.md` § Feature 1 |
| 3 | Data Engine | $k$-NN Sparsification Masking | Restricts non-neighbor entries in $d_{\text{ins}}$ matrix to $+\infty$ based on $k$-nearest neighbors. | `k_neighbors`: int (default 15) | Masked `d_ins`: non-neighbors set to `float('inf')` | If $k \ge N$, no masking applied; returns dense $d_{\text{ins}}$. | `ORIGINAL_REQUEST.md` R1, `rl4co/data/insertion_cost.py` |
| 4 | Data Engine | Offline Dataset Caching CLI | Precomputes and caches Uniform & Clustered GMM instances + $d_{\text{ins}}$ matrices to disk (`.pt`). | CLI args: `--num_samples`, `--graph_sizes`, `--distributions`, `--output_dir` | Caching `.pt` files on disk | Raises `IOError` if output directory unwritable. | `ORIGINAL_REQUEST.md` R1, `PROJECT.md` § Feature 3 |
| 5 | Abstraction | Modular `SlotAttention` Layer | Differentiable Slot Attention module with GRU refinement and Softmax over slots $K$. | `inputs`: $(B, N, D_{\text{in}})$ tensor | `(slots, attn)`: slots $(B, K, D_{\text{slot}})$, attn $(B, N, K)$ | Raises error if `inputs` feature dim does not match layer configuration. | `ORIGINAL_REQUEST.md` R2, `PROJECT.md` § Feature 4 |
| 6 | Abstraction | Aggregated Slot Conditioning $\hat{z}_i$ | Aggregates node slots $\hat{z}_i = \sum_k A_{ik} z_k$ to condition decoder queries. | `slots`: $(B, K, D_{\text{slot}})$, `attn`: $(B, N, K)$ | `aggregated_slots`: $(B, N, D_{\text{slot}})$ tensor | Output shape guaranteed to match $(B, N, D_{\text{slot}})$. | `PROJECT.md` § Interface Contracts |
| 7 | Policy | POMO Policy Wiring (Variant B) | Integrates aggregated slot embeddings $\hat{z}_i$ into POMO decoder conditioning (task-only loss). | `embeddings`: $(B, N, D)$, `slots`: $(B, K, D_{\text{slot}})$ | Conditioned decoder query tensor | Validates slot dimension matches or is projected to decoder dimension. | `ORIGINAL_REQUEST.md` R2, `PROJECT.md` § Feature 6 |
| 8 | Metric Loss | METRA Projection Head $\phi(z_k)$ | Projection MLP mapping slot embeddings $z_k$ into metric space $\mathbb{R}^{D_{\text{metric}}}$. | `slots`: $(B, K, D_{\text{slot}})$ | `phi_slots`: $(B, K, D_{\text{metric}})$ tensor | Requires non-empty slot tensor. | `ORIGINAL_REQUEST.md` R3, `PROJECT.md` § Feature 7 |
| 9 | Metric Loss | Dual Ascent Lagrangian Penalty | Enforces lower-bound distance constraint $\|\phi(z_k) - \phi(z_j)\| \ge d_{ij}$ with dynamic $\lambda$. | `phi_slots`, `target_dist`: $(B, N, N)$ | `loss_metric`: scalar tensor, `dual_penalty`: scalar | Clamps $\lambda \ge 0.0$ to ensure dual stability; ignores `inf` non-neighbors. | `ORIGINAL_REQUEST.md` R3, `PROJECT.md` § Feature 7 |
| 10 | Metric Loss | Slot Entropy Regularization $H(A)$ | Computes slot assignment entropy $H(A) = - \frac{1}{N} \sum_{i,k} A_{ik} \log A_{ik}$ to prevent slot collapse. | `attn`: $(B, N, K)$ tensor | `loss_entropy`: scalar tensor | Adds $\epsilon = 10^{-8}$ before log to avoid $\log(0)$ NaNs. | `ORIGINAL_REQUEST.md` R3, `PROJECT.md` § Feature 7 |
| 11 | Model | Model Variant Toggles (A-E) | LightningModule configuring loss terms for Variants A (Reconstruction), B (Task-Only), C (Euclidean), D (Insertion Cost), E (Future Regret). | Config parameters / CLI flags | Configured model & forward step loss dict | Raises `ValueError` for unknown variant string. | `ORIGINAL_REQUEST.md` R3, `PROJECT.md` § Feature 9 |
| 12 | Benchmarks | Multi-Seed Decision Gate M8 | Evaluation script `scripts/eval_pomo_slot.py` computing optimality gap, ARI stability, and slot entropy. | Evaluated checkpoints, test datasets | Benchmark report / logs | Validates dataset format and checkpoint loading. | `ORIGINAL_REQUEST.md` R4, `PROJECT.md` § Feature 12 |

---

## Edge Cases

| # | Feature | Input | Observed / Expected Behavior |
|---|---------|-------|-----------------------------|
| 1 | $d_{\text{ins}}$ Operator | Single instance batch $B=1$ | Matrix dimensions must remain 3D $(1, N, N)$ without dropping batch dimension unless explicitly unbatched $(N, N)$. |
| 2 | $k$-NN Sparsification | $N \le k_{\text{neighbors}}$ (e.g. $N=10, k=15$) | No entries masked to $+\infty$; output matrix is dense $d_{\text{ins}}$ matrix with diagonal zeroed. |
| 3 | $d_{\text{ins}}$ Operator | Coincident node coordinates ($x_i = x_j$) | Distance $D_{ij} = 0$, $d_{\text{ins}}(i, j) = d_0(i) - d_0(j)$; numerically stable, no division by zero. |
| 4 | Slot Attention | $K=1$ slot | Softmax over slots $K=1$ yields $A_{ik} = 1.0$ for all nodes $i$; aggregated slot representation $\hat{z}_i = z_1$. |
| 5 | Slot Attention | All node embeddings $H$ set to zero | Attention logits $M_{ik} = 0$; softmax yields uniform attention $A_{ik} = 1/K$; GRU updates stably. |
| 6 | Slot Attention | Attention weight $A_{ik} = 0.0$ | Log calculation in entropy uses $A_{ik} \log(A_{ik} + \epsilon)$ to prevent $\log(0) \to -\infty$ or NaNs. |
| 7 | Metric Loss | $k$-NN sparsified `target_dist` with $+\infty$ values | Metric loss computation must filter/mask out $+\infty$ entries so only valid neighbor distance constraints are enforced. |
| 8 | Dual Ascent | Dual parameter $\lambda$ update step | $\lambda^{(t+1)} = \max(0.0, \lambda^{(t)} + \eta_\lambda \cdot \text{penalty})$; invariant $\lambda \ge 0.0$ enforced via clamping. |
| 9 | Slot Entropy | Perfect hard slot assignment (one-hot $A_{ik}$) | Entropy $H(A) = 0.0$; loss computation remains finite and stable. |
| 10 | Slot Entropy | Perfect uniform slot assignment ($A_{ik} = 1/K$) | Entropy $H(A) = \log(K)$; maximum entropy bound reached. |

---

## 5-Component Handoff Report

### 1. Observation
From direct inspection of `d:/NCO NEW/rl4co/PROJECT.md`, `d:/NCO NEW/rl4co/.agents/ORIGINAL_REQUEST.md`, `d:/NCO NEW/rl4co/.agents/sub_orch_e2e_testing/SCOPE.md`, `rl4co/data/insertion_cost.py`, and `tests/test_insertion_cost.py`:

1. **$d_{\text{ins}}$ Marginal Insertion Cost Module (`rl4co/data/insertion_cost.py`)**:
   - Implements `compute_pairwise_distance_matrix(coords)` returning $(B, N, N)$ Euclidean distance matrix $D_{ij} = \|x_i - x_j\|_2$.
   - Implements `compute_marginal_insertion_cost(locs, k_neighbors=15, depot_loc=None)` computing $d_{\text{ins}}(i, j) = d_0(i) + D_{ij} - d_0(j)$ where $d_0(i) = \|x_i - x_0\|_2$.
   - Self-insertion cost $d_{\text{ins}}(i, i) = 0.0$.
   - $k$-NN sparsification uses `torch.topk` on distance matrix to select $k_{\text{neighbors}} + 1$ indices per row and masks non-neighbors to `float('inf')`.

2. **Slot Attention Module (`rl4co/models/nn/slot_attention.py`)**:
   - Specified interface contract: `SlotAttention(inputs)` returning tuple `(slots, attn)` where `slots` has shape $(B, K, D_{\text{slot}})$ and `attn` has shape $(B, N, K)$.
   - Key mathematical invariant: Node attention normalization $\sum_{k=1}^K A_{ik} = 1.0$ enforced via `Softmax` over the slot dimension $K$ (dim=-1 of logits matrix $M \in \mathbb{R}^{B \times N \times K}$).
   - Aggregated slot representation for node conditioning: $\hat{z}_i = \sum_{k=1}^K A_{ik} z_k \in \mathbb{R}^{B \times N \times D_{\text{slot}}}$.

3. **METRA Metric Loss Module (`rl4co/models/nn/metric_loss.py`)**:
   - Projection head $\phi(z_k): \mathbb{R}^{D_{\text{slot}}} \to \mathbb{R}^{D_{\text{metric}}}$ via MLP.
   - Dual ascent penalty: Lagrangian constraint $\mathcal{C} = \max(0, d_{ij} - \|\phi(z_k) - \phi(z_j)\|)$ scaled by non-negative dual variable parameter $\lambda$.
   - Slot entropy: $H(A) = - \frac{1}{B \cdot N} \sum_{b, i, k} A_{ik} \log(A_{ik} + \epsilon)$.

### 2. Logic Chain
1. **Data Engine to Slot Attention Pipeline**:
   - $d_{\text{ins}}(i, j)$ provides ground-truth metric topology structure for customer locations, incorporating depot proximity and pairwise distance increment.
   - $k$-NN sparsification limits dense $O(N^2)$ interactions to top-$k$ local neighborhoods, setting non-neighbor entries to $+\infty$.
2. **Slot Attention Softmax Normalization**:
   - Normalizing over slots $K$ (instead of nodes $N$) ensures every node $i$ distributes 100% of its slot membership budget across the $K$ slots, satisfying $\sum_{k=1}^K A_{ik} = 1.0$.
   - Aggregating slots as $\hat{z}_i = \sum_k A_{ik} z_k$ projects slot-level abstract concepts back to individual nodes, allowing POMO decoder queries to be conditioned on node-specific sub-tour slot contexts.
3. **METRA Loss Regularization & Dual Ascent**:
   - Without metric loss, slot representations can suffer from slot collapse or uninformative cluster distributions.
   - METRA enforces lower-bound isometric constraints where distance in slot metric space $\|\phi(z_k) - \phi(z_j)\|$ preserves graph/routing distance $d_{ij}$ (Variant C Euclidean or Variant D $d_{\text{ins}}$).
   - Dynamic parameter $\lambda$ updated via dual ascent guarantees convergence under constraint satisfaction, while slot entropy $H(A)$ maximizes slot utilization diversity.

### 3. Caveats
- `rl4co/data/insertion_cost.py` currently exists and passes unit tests in `tests/test_insertion_cost.py`. `slot_attention.py` and `metric_loss.py` are fully specified in terms of interfaces, tensor shapes, and math, but pending full implementation in Milestones M2 and M4.
- In $k$-NN sparsified $d_{\text{ins}}$ matrices, $+\infty$ values must be handled carefully in metric loss functions to avoid propagating NaNs or $\infty \times 0$ zero-gradient artifacts. Masking invalid non-neighbor pairs prior to loss calculation is mandatory.

### 4. Conclusion
The specification for Metric-Aware Slot Abstraction NCO across $d_{\text{ins}}$ sparsification, Slot Attention, and METRA Metric Loss is complete, mathematically rigorous, and fully defined in terms of Python/PyTorch function signatures, matrix dimensions, normalization invariants, and edge case behaviors.

### 5. Verification Method
1. **Mathematical Invariant Verification**:
   - Execute `pytest tests/test_insertion_cost.py` to verify $d_{\text{ins}}$ matrix shapes $(B, N, N)$, self-insertion zeros, and $k$-NN bounds.
   - Verify `SlotAttention` outputs satisfy `attn.sum(dim=-1) == 1.0` within numerical tolerance `$10^{-6}$`.
   - Verify `MetricLoss` dual parameter $\lambda \ge 0.0$ and slot entropy $0.0 \le H(A) \le \log(K)$.
2. **Signature & Shape Inspection**:
   - Inspect signatures against contract definitions:
     - `compute_marginal_insertion_cost(locs, k_neighbors=15, depot_loc=None) -> Tensor`
     - `SlotAttention.forward(inputs) -> Tuple[Tensor, Tensor]`
     - `MetricLoss.forward(slots, attn, target_dist) -> Dict[str, Tensor]`
