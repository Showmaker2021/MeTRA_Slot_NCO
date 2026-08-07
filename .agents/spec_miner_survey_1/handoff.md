# Metric-Aware Slot Abstraction NCO: Specification Mining & Feature Inventory Report

**Agent:** `spec_miner_survey_1`  
**Working Directory:** `d:/NCO NEW/rl4co/.agents/spec_miner_survey_1`  
**Target Project:** `rl4co` (`d:/NCO NEW/rl4co`)  
**Date:** 2026-08-06  

---

## 1. Observation

Based on a thorough inspection of `d:/NCO NEW/rl4co/.agents/ORIGINAL_REQUEST.md`, `d:/NCO NEW/metric-aware-slot-abstraction-proposal.md`, `references/slot-attention/slot_attention/slot_attention.py`, `references/METRA/iod/metra.py`, and the `rl4co` codebase structure:

### Existing Infrastructure & File Status
1. **`ORIGINAL_REQUEST.md`** (`.agents/ORIGINAL_REQUEST.md:1-35`): Mandates the integration of Metric-Aware Slot Abstraction into `rl4co`, covering 4 key requirement areas (R1-R4) spanning 9 milestones (M0-M8).
2. **Proposal Specification** (`metric-aware-slot-abstraction-proposal.md:1-106`): Defines the mathematical formulation for $d_{\text{ins}}$, aggregated slot distance $D_{\text{ins}}$, METRA lower-bound metric loss $\mathcal{L}_{\text{metric}}$, dual ascent update for $\lambda$, slot entropy regularizer $\mathcal{L}_{\text{slot-entropy}}$, and model variants A through E.
3. **Reference Implementations**:
   - `references/slot-attention/slot_attention/slot_attention.py:21-86`: Provides PyTorch `SlotAttention` implementation with LayerNorm, linear queries/keys/values, GRU cell updates, and residual MLPs over softmax/L1-normalized cross-attention weights.
   - `references/METRA/iod/metra.py:194-300`: Provides the dual-ascent Lagrangian loss formulation with lower-bound inequality constraints.
4. **Target Files in `rl4co`**:
   - **R1 (Data)**: `rl4co/data/insertion_cost.py` (partially implemented in lines 1-98), `rl4co/data/generate_slot_dataset.py`, `tests/test_insertion_cost.py` (lines 1-39).
   - **R2 (Model Architecture & Policy)**: `rl4co/models/nn/slot_attention.py`, `rl4co/models/zoo/pomo_slot/policy.py`, `tests/test_slot_attention.py`.
   - **R3 (Metric Loss & Dual Ascent)**: `rl4co/models/nn/metric_loss.py`, `rl4co/models/zoo/pomo_slot/model.py`, `tests/test_metric_loss.py`.
   - **R4 (Configs & Decision Gate)**: `conf/model/pomo_slot_a.yaml` through `pomo_slot_e.yaml` (co-located at `rl4co/configs/model/pomo_slot_*.yaml`), evaluation benchmark script `scripts/eval_pomo_slot.py` (or `tests/test_pomo_slot_eval.py`).

---

## 2. Logic Chain

1. **R1 Data Pipeline (M0 & M1)**:
   - *Observation*: Neural combinatorial optimization models struggle with implicit distance representations for large graphs. Precomputing pairwise marginal insertion costs $d_{\text{ins}}(i, j)$ offline avoids bottlenecking online training loops.
   - *Mathematical Formulation*:
     - Node locations $\mathbf{x}_i \in [0, 1]^2$, Depot location $\mathbf{x}_{\text{depot}} \in [0, 1]^2$.
     - Depot distance: $d_{\text{depot}}(i) = \|\mathbf{x}_i - \mathbf{x}_{\text{depot}}\|_2$.
     - Pairwise node distance: $d_{\text{cust}}(i, j) = \|\mathbf{x}_i - \mathbf{x}_j\|_2$.
     - Marginal insertion cost: $d_{\text{ins}}(i, j) = d_{\text{depot}}(i) + d_{\text{cust}}(i, j) - d_{\text{depot}}(j)$.
     - Self insertion: $d_{\text{ins}}(i, i) = 0.0$.
     - $k$-NN sparsification: For node $j$, identify top-$(k+1)$ nearest neighbors based on $d_{\text{cust}}(\cdot, j)$ (including self). Mask all non-neighbors with `float('inf')`.
   - *Data Generation & Storage*:
     - Generate $N \in \{50, 100, 200, 500\}$ across **Uniform** (i.i.d. $U[0, 1]^2$) and **Clustered** (Gaussian Mixture Model with 3–5 cluster centers, $\sigma \in [0.05, 0.10]$, clipped to $[0, 1]^2$).
     - Serialize precomputed tensors (`locs`, `depot_loc`, `d_ins`) into `.pt` files.

2. **R2 Slot Attention & Policy Wiring (M2 & M3)**:
   - *Observation*: Bottlenecking high-dimensional graph node embeddings into $K \ll N$ region slots $z_1, \dots, z_K$ reduces computational overhead and exposes macro-level routing geometry.
   - *Slot Attention Computation*:
     - Input: Node embeddings $h \in \mathbb{R}^{B \times N \times d}$.
     - Learnable slot parameters: $\mu_{\text{slots}}, \sigma_{\text{slots}} \in \mathbb{R}^{1 \times 1 \times d}$. Sample initial slots $z^{(0)} \sim \mathcal{N}(\mu_{\text{slots}}, \text{diag}(\sigma_{\text{slots}}))$.
     - For iterations $t = 1 \dots T_{\text{iters}}$:
       $$q = W_q \cdot \text{LayerNorm}(z^{(t-1)}), \quad k = W_k \cdot \text{LayerNorm}(h), \quad v = W_v \cdot \text{LayerNorm}(h)$$
       $$A_{ik} = \text{Softmax}_k\left(\frac{q_k \cdot k_i}{\sqrt{d}}\right), \quad \tilde{A}_{ik} = \frac{A_{ik} + \epsilon}{\sum_{j} (A_{jk} + \epsilon)}$$
       $$U_k = \sum_{i} \tilde{A}_{ik} v_i, \quad z_k^{(t)} = \text{GRUCell}(U_k, z_k^{(t-1)}) + \text{MLP}(\text{LayerNorm}(\dots))$$
     - Return slot embeddings $z \in \mathbb{R}^{B \times K \times d}$ and soft assignment matrix $A \in \mathbb{R}^{B \times N \times K}$.
   - *POMO Policy Wiring*:
     - Modulate node embeddings $h_i$ with aggregated slot representation $\hat{z}_i = \sum_{k=1}^K A_{ik} z_k$.
     - Conditioned node embedding: $\tilde{h}_i = h_i + \hat{z}_i$ (or linear projection). Pass $\tilde{h}_i$ to POMO multi-start decoder.

3. **R3 METRA Metric Loss & Model Toggles (M4, M5, M6)**:
   - *Observation*: Unsupervised slot representations risk collapsing or smoothing out without explicit spatial metric targets.
   - *METRA Metric Loss & Dual Ascent*:
     - Projection head $\phi: \mathbb{R}^d \to \mathbb{R}^{d_{\text{proj}}}$. Latent slot distance: $d_{\text{latent}}(k, \ell) = \|\phi(z_k) - \phi(z_\ell)\|_2$.
     - Target region distance: $D_{\text{target}}(k, \ell)$.
       - Variant C (Euclidean): $D_{\text{target}}(k, \ell) = \|\bar{x}_k - \bar{x}_\ell\|_2$, where $\bar{x}_k = \frac{\sum_i A_{ik} \mathbf{x}_i}{\sum_i A_{ik} + \epsilon}$.
       - Variant D (Insertion Cost): $D_{\text{ins}}(k, \ell) = \sum_{i, j} A_{ik} A_{j\ell} d_{\text{ins}}(i, j)$ (ignoring `inf` values or masking).
     - Lower-bound constraint penalty: $g(k, \ell) = \max\left(0,\ d_{\text{latent}}(k, \ell) - D_{\text{target}}(k, \ell)\right)^2$.
     - Dual Ascent parameter update: $\lambda \leftarrow \max(0, \lambda + \eta_{\text{dual}} \cdot (\mathbb{E}[g(k, \ell)] - \epsilon_{\text{slack}}))$.
     - Metric Loss: $\mathcal{L}_{\text{metric}} = -\mathbb{E}_{k \neq \ell}[d_{\text{latent}}(k, \ell)] + \lambda \cdot \mathbb{E}_{k \neq \ell}[g(k, \ell)]$.
     - Slot Entropy Regularizer: $\mathcal{L}_{\text{slot-entropy}} = \mathbb{E}_i \left[ \sum_{k=1}^K A_{ik} \log (A_{ik} + \epsilon) \right]$.
     - Total Loss: $\mathcal{L} = \mathcal{L}_{\text{policy}} + \alpha \mathcal{L}_{\text{metric}} + \beta \mathcal{L}_{\text{slot-entropy}}$.
   - *Model Variant Matrix*:
     - **Variant A**: Coordinate Reconstruction Loss ($\alpha=0$, Autoencoder mode).
     - **Variant B**: Task-Only Loss ($\alpha=0$, DPN-style baseline).
     - **Variant C**: Euclidean Metric Loss ($\alpha > 0$, $D_{\text{target}} = D_{\text{Euclidean}}$).
     - **Variant D**: Insertion-Cost Metric Loss ($\alpha > 0$, $D_{\text{target}} = D_{\text{ins}}$) — **Proposed Core Method**.
     - **Variant E**: Future-Regret Metric Loss ($\alpha > 0$, $D_{\text{target}} = D_{\text{regret}}$).

4. **R4 Hydra Configs & Decision Checkpoint M8 (M7 & M8)**:
   - Config YAML files define hyperparameter choices (`variant`, `alpha`, `metric_type`, `num_slots`, `num_iters`, `dual_lr`).
   - M8 Decision Gate runs multi-seed comparisons on $N=50$ to evaluate Optimality Gap (%), ARI Stability, Slot Entropy, and Dual Parameter Convergence.

---

## 3. Caveats & Assumptions

1. **Numerical Stability in Sparsified Distances**:
   - Non-$k$-NN entries in $d_{\text{ins}}$ are set to `float('inf')`. Aggregating $D_{\text{ins}}(k, \ell) = \sum_{i,j} A_{ik} A_{j\ell} d_{\text{ins}}(i,j)$ directly with `inf` will produce `NaN`/`inf` loss values.
   - *Mitigation*: When computing $D_{\text{ins}}(k, \ell)$, mask out `inf` entries and normalize by $\sum_{(i,j) \in k\text{-NN}} A_{ik} A_{j\ell} + \epsilon$.
2. **Small Problem Sizes ($N \le k$)**:
   - If graph size $N \le k$, top-$k$ selection must clamp to $N$ to prevent PyTorch `topk` indexing errors.
3. **Dual Parameter Exponentiation vs Clamping**:
   - In METRA, dual multiplier $\lambda$ is parametrized as $\lambda = \exp(\log\text{\_lambda})$ to enforce non-negativity ($\lambda > 0$). Gradient descent is performed on $\log\text{\_lambda}$.
4. **Soft Assignment Collapse**:
   - Without entropy regularization ($\beta > 0$), slot attention may collapse to assigning all nodes to a single slot. The slot entropy term prevents uniform or degenerate single-slot collapse.

---

## 4. Conclusion: Detailed Feature Inventory & Edge Cases

### Features Discovered Matrix (R1 - R4, Milestones M0 - M8)

| # | Category | Feature / Requirement | Description | Inputs | Outputs | Error Behavior | Discovered Via | Milestone & Acceptance Criteria |
|---|----------|-----------------------|-------------|--------|---------|----------------|----------------|---------------------------------|
| 1 | R1: Data | Vectorized $d_{\text{ins}}$ Operator | Compute pairwise marginal insertion cost $d_{\text{ins}}(i,j) = d_{\text{depot}}(i) + d_{\text{cust}}(i,j) - d_{\text{depot}}(j)$ with self-insertion set to 0.0. | `locs`: $(B, N, 2)$, `depot_loc`: $(B, 1, 2)$ | `d_ins`: $(B, N, N)$ | Raise `ValueError` if `locs` dimension $< 2$ or channel $\neq 2$. | `rl4co/data/insertion_cost.py` | **M0**: Pass `tests/test_insertion_cost.py::test_compute_pairwise_distance_matrix` & `test_marginal_insertion_cost_basic`. |
| 2 | R1: Data | $k$-NN Sparsification | Restrict $d_{\text{ins}}$ entries per node $j$ to its $k$-nearest customer neighbors; set non-neighbors to `inf`. | `locs`, `k_neighbors` (int, default 15) | Sparsified `d_ins`: $(B, N, N)$ | If $k \le 0$, raise `ValueError`. If $k \ge N$, clamp $k = N-1$. | `rl4co/data/insertion_cost.py` | **M0**: Pass `tests/test_insertion_cost.py::test_knn_sparsification` (at most $k+1$ non-inf values per row). |
| 3 | R1: Data | Offline Dataset Generator CLI | Generate and save Uniform ($U[0,1]^2$) and Clustered (GMM) dataset instances with precomputed sparsified $d_{\text{ins}}$ to `.pt` disk files. | `--num_samples`, `--graph_sizes` (50, 100, 200, 500), `--distributions`, `--k_neighbors`, `--output_dir` | `.pt` dataset files containing `locs`, `depot`, `d_ins` | Validate file path creation and disk write access; raise error if output dir unwriteable. | `rl4co/data/generate_slot_dataset.py` | **M1**: Clean CLI execution creating `.pt` files for $N \in \{50, 100, 200, 500\}$ under `data/slots/`. |
| 4 | R2: Model | Standalone `SlotAttention` Layer | Differentiable iterative cross-attention mapping node embeddings $h \in \mathbb{R}^{B \times N \times d}$ to $K$ slot embeddings $z \in \mathbb{R}^{B \times K \times d}$ and soft assignment matrix $A \in \mathbb{R}^{B \times N \times K}$. | `inputs`: $(B, N, d)$, `num_slots` ($K$), `iters` (default 3), `hidden_dim` | `slots`: $(B, K, d)$, `attn`: $(B, N, K)$ | Raise `ValueError` if input feature dim $\neq \text{dim}$. | `rl4co/models/nn/slot_attention.py` | **M2**: Pass `tests/test_slot_attention.py` verifying output shapes $(B, K, d)$ and softmax sum $\sum_k A_{ik} = 1.0$. |
| 5 | R2: Model | Slot-Conditioned POMO Policy Wiring | Wire slot embeddings $z_k$ into node embeddings $h_i$ via soft assignment aggregation $\hat{z}_i = \sum_k A_{ik} z_k$, passing conditioned embeddings to POMO multi-start decoder. | `td`: TensorDict containing environment state, `env`: RL4CO environment instance | `out`: dict containing `reward`, `log_likelihood`, `actions` | Check tensor device mismatch between slots and node embeddings. | `rl4co/models/zoo/pomo_slot/policy.py` | **M3**: Successfully execute end-to-end forward pass for **Variant B** ($\alpha=0$, task-only loss) without metric loss. |
| 6 | R3: Loss | METRA Metric Loss & Projection Head | Project $z_k \to \phi(z_k)$, compute pairwise latent distance $d_{\text{latent}}(k,\ell)$, and enforce lower-bound constraint against target region distance $D_{\text{target}}(k,\ell)$. | `slots`: $(B, K, d)$, `attn`: $(B, N, K)$, `target_dist`: $(B, K, K)$ | `loss_metric`: scalar Tensor, `penalty`: scalar Tensor | Check for `NaN` propagation from zero-division in centroid/soft-assignment aggregation. | `rl4co/models/nn/metric_loss.py` | **M4**: Pass `tests/test_metric_loss.py` verifying non-negative penalty $g(k,\ell) \ge 0$ and gradient backpropagation to $\phi$. |
| 7 | R3: Loss | Dual Ascent Parameter Update | Maintain log dual multiplier $\log \lambda$, update $\lambda = \exp(\log \lambda)$ via Dual Ascent step based on constraint violation penalty $\mathbb{E}[g(k,\ell)] - \epsilon_{\text{slack}}$. | `penalty`: Tensor, `dual_lr`: float (default 1e-2), `slack`: float | Updated `log_lambda`, `dual_loss` | Clamp `log_lambda` to $[-\text{10}, \text{10}]$ to prevent numerical explosion/underflow. | `rl4co/models/nn/metric_loss.py` | **M4**: Verify $\lambda > 0$ strictly holds and converges dynamically during dual update steps. |
| 8 | R3: Loss | Slot Entropy Regularizer | Compute entropy $H(A) = -\frac{1}{N}\sum_{i,k} A_{ik} \log(A_{ik} + \epsilon)$ to penalize uniform or uninformative slot assignments. | `attn`: $(B, N, K)$, `eps`: float (1e-8) | `loss_entropy`: scalar Tensor | Add $\epsilon$ before $\log$ to prevent $\log(0) = -\infty$. | `rl4co/models/nn/metric_loss.py` | **M4**: Verify loss decreases when slot assignment matrix $A$ becomes sharper/crisper. |
| 9 | R3: Model | Model Variant Toggles (A, B, C, D, E) | Integrated model LightningModule supporting configuration toggles for Variants A (AE), B (Task-Only), C (Euclidean), D (Insertion Cost - Proposed), and E (Future Regret). | `variant`: str ("A", "B", "C", "D", "E"), `alpha`: float, `beta`: float | Combined model loss $\mathcal{L}_{\text{total}}$ and validation metrics | Raise `ValueError` if unknown variant string passed. | `rl4co/models/zoo/pomo_slot/model.py` | **M5 & M6**: Clean single-command execution of each model variant A through E on standard CVRP/TSP environment. |
| 10 | R4: Config | Hydra Configuration Specifications | Yaml configs for model variants A through E detailing hyperparameters, slot count, loss weights, and optimizer specs. | Config path: `conf/model/pomo_slot_[a-e].yaml` | Hydra config DictConfig object | Validate Hydra config parsing; fail on missing required keys (`variant`, `alpha`). | `conf/model/pomo_slot_a.yaml` ... `pomo_slot_e.yaml` | **M7**: Successful instantiation of model variants A through E via Hydra CLI overrides (`model=pomo_slot_d`). |
| 11 | R4: Eval | M8 Decision Gate Benchmark Suite | Multi-seed evaluation script comparing Variants A–E on $N=50$ instances across Optimality Gap (%), ARI Slot Stability, Slot Entropy, and Dual Parameter Convergence. | `--variants` (A,B,C,D,E), `--graph_size` (50), `--num_seeds` (3), `--dataset_path` | CSV/JSON summary report & logging metrics | Check dataset file existence before evaluation run. | `scripts/eval_pomo_slot.py` | **M8**: Complete multi-seed evaluation run generating comparative benchmark table for decision gate review. |

---

### Edge Cases Matrix

| # | Feature / Area | Edge Case Input | Expected / Observed Behavior | Handling / Safeguard |
|---|----------------|-----------------|------------------------------|----------------------|
| 1 | R1: $d_{\text{ins}}$ Operator | Graph size $N \le k$ (e.g. $N=10$, $k=15$). | $k$-NN top-$k$ selection would fail if requested $k \ge N$. | Automatically clamp $k_{\text{eff}} = \min(k, N-1)$. |
| 2 | R1: $d_{\text{ins}}$ Operator | Single instance unbatched input $(N, 2)$. | Tensor dimensions missing batch dimension. | Unsqueeze batch dim at entry ($(1, N, 2)$), compute, and squeeze at exit. |
| 3 | R1: $d_{\text{ins}}$ Operator | Customer coordinate identical to Depot coordinate. | Distance $d_{\text{depot}}(i) = 0$, leading to zero insertion cost. | Handled gracefully without division by zero; output cost is 0.0. |
| 4 | R2: Slot Attention | Single slot $K=1$. | Attention matrix $A_{ik} = 1.0$ everywhere. | GRU and MLP execute normally; slot embedding represents global graph centroid. |
| 5 | R2: Slot Attention | Zero-valued input node embeddings $h_i = 0$. | Dot product $q \cdot k = 0$, uniform initial softmax probabilities. | LayerNorm epsilon prevents zero variance normalization division error. |
| 6 | R3: Metric Loss | Non-neighbor $d_{\text{ins}}$ values equal to `float('inf')`. | Matrix multiplication with `inf` causes `NaN`/`inf` loss. | Mask out `inf` elements before computing soft region aggregation $D_{\text{ins}}(k, \ell)$. |
| 7 | R3: Dual Ascent | Dual parameter gradient explosion ($\lambda \to \infty$). | Infinite penalty forces latent space to contract to a single point. | Clamp $\log \lambda \in [-10.0, 10.0]$ and clip dual gradients. |
| 8 | R3: Slot Entropy | Soft assignment $A_{ik} = 0.0$. | $\log(0.0)$ yields $-\infty$ / `NaN`. | Add numerical stability epsilon: $A_{ik} \log(A_{ik} + 1e-8)$. |
| 9 | R4: Hydra Config | Missing variant toggle parameter in YAML config. | Model initialization fails silently or uses default. | Enforce strict schema validation in `POMOSlotModel.__init__`. |

---

## 5. Verification Method

To verify the implementation of Milestones M0 through M8 independently, execute the following commands in sequence:

### 1. Verification of Milestone M0 ($d_{\text{ins}}$ Sparsified Operator)
Run PyTorch unit test suite for insertion cost calculation:
```bash
pytest tests/test_insertion_cost.py -v
```
*Expected Result*: Tests `test_compute_pairwise_distance_matrix`, `test_marginal_insertion_cost_basic`, and `test_knn_sparsification` pass with zero failures.

### 2. Verification of Milestone M1 (Offline Dataset Caching CLI)
Generate sample cached datasets for Uniform and Clustered distributions:
```bash
python rl4co/data/generate_slot_dataset.py --num_samples 100 --graph_sizes 50 100 --distributions uniform clustered --k_neighbors 15 --output_dir data/slots_test/
```
*Expected Result*: Files `data/slots_test/slot_cvrp_u50_k15.pt`, `data/slots_test/slot_cvrp_c50_k15.pt`, `data/slots_test/slot_cvrp_u100_k15.pt`, and `data/slots_test/slot_cvrp_c100_k15.pt` are created and contain valid dictionary keys `locs`, `depot`, and `d_ins`.

### 3. Verification of Milestone M2 & M3 (Slot Attention & POMO Policy Wiring)
Run unit test for Slot Attention layer and Variant B policy forward pass:
```bash
pytest tests/test_slot_attention.py -v
```
*Expected Result*: Asserts output tensor shape $(B, K, d)$ and verifies softmax normalization across slots $\sum_k A_{ik} = 1.0$.

### 4. Verification of Milestone M4, M5, M6 (METRA Metric Loss & Variant Toggles)
Run metric loss unit tests and test forward pass for model variants A through E:
```bash
pytest tests/test_metric_loss.py -v
```
*Expected Result*: Verifies non-negativity of Lagrangian constraint penalties $g(k,\ell) \ge 0$, dual ascent update of $\lambda$, and loss computation across Variants A, B, C, D, E.

### 5. Verification of Milestone M7 & M8 (Hydra Configs & Comparative Decision Gate)
Execute single CLI run for Variant D training step and multi-seed evaluation:
```bash
python run.py model=pomo_slot_d env=cvrp trainer.max_epochs=1
python scripts/eval_pomo_slot.py --graph_size 50 --num_seeds 3 --output_report decision_checkpoint_m8.json
```
*Expected Result*: Evaluation script produces comparative metrics table logging Optimality Gap (%), ARI Slot Stability, Slot Entropy, and Dual Parameter Convergence for M8 decision gate approval.
