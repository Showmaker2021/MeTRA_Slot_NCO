# Handoff Report — Exploration & Survey of `rl4co` for POMO & Metric-Aware Slot Abstraction

## 1. Observation

Direct code examination of the `rl4co` codebase (`d:/NCO NEW/rl4co`) revealed the following exact structures and implementations across the 5 target domains:

### A. Data Engine & Cost Matrices (`rl4co/data/`)
- **`rl4co/data/insertion_cost.py`**:
  - Line 7-28: `compute_pairwise_distance_matrix(coords: torch.Tensor)` computes Euclidean distance matrix $(B, N, N)$ using `torch.norm(diff, p=2, dim=-1)`.
  - Line 31-97: `compute_marginal_insertion_cost(locs, k_neighbors=15, depot_loc=None)` computes pairwise marginal insertion cost matrix $d_{\text{ins}}(i,j) = d(D,i) + d(i,j) - d(D,j)$. It sets self-insertion $d_{\text{ins}}(i,i) = 0.0$ and applies k-NN sparsification using `torch.topk(dist_customers, k=k_neighbors + 1, dim=-1, largest=False)` filling non-neighbors with `float("inf")`.
- **`rl4co/data/dataset.py`**:
  - Lines 15-40: `FastTdDataset` wraps a `TensorDict` directly.
  - Lines 41-72: `TensorDictDataset` disassembles a `TensorDict` into list of dicts for low CPU usage during training.
  - Lines 74-91: `ExtraKeyDataset` decorates an existing dataset with an extra key (useful for precomputed cost matrices like $d_{\text{ins}}$ or baseline rewards).
- **`rl4co/data/generate_data.py`**:
  - Lines 37-39: `generate_tsp_data(dataset_size, tsp_size)` returns dictionary `{"locs": np.random.uniform(size=(dataset_size, tsp_size, 2))}`.
  - Lines 41-76: `generate_vrp_data(...)` returns depot, locs, demand, capacity dict.
  - Lines 214-313: `generate_dataset(...)` saves datasets as `.npz` files via `np.savez`.
- **`rl4co/data/transforms.py`**:
  - Lines 105-151: `StateAugmentation` applies 8x dihedral transformations (`dihedral_8_augmentation_wrapper`) or symmetric transformations to input `TensorDict` features.
- **`rl4co/data/utils.py`**:
  - Lines 11-19: `load_npz_to_tensordict(filename)` loads `.npz` into `TensorDict`.
  - Lines 22-30: `save_tensordict_to_npz(tensordict, filename)` saves `TensorDict` to `.npz`.

### B. Neural Network Modules & Context Embeddings (`rl4co/models/nn/`)
- **`rl4co/models/nn/attention.py`**:
  - Lines 19-51: `scaled_dot_product_attention_simple(q, k, v, attn_mask, dropout_p, is_causal)` exact SDPA implementation.
  - Lines 64-135: `MultiHeadAttention` (standard PyTorch native MHA with SDPA).
  - Lines 147-200: `MultiHeadCrossAttention`.
  - `PointerAttention` and `PointerAttnMoE` for pointer decoding logits calculation.
- **`rl4co/models/nn/env_embeddings/init.py`**:
  - `env_init_embedding(env_name, config)` creates initial node feature projection modules (e.g. `TSPInitEmbedding` mapping 2D coords to `embed_dim` $d=128$).
- **`rl4co/models/nn/env_embeddings/context.py`**:
  - Lines 105-134: `TSPContext` projects `[first_node_emb, current_node_emb]` via `nn.Linear(2 * embed_dim, embed_dim)`.
  - Uses `gather_by_index` from `rl4co/utils/ops.py` to extract embeddings of selected nodes.
- **`rl4co/models/nn/graph/attnnet.py`**:
  - `GraphAttentionNetwork`: Transformer encoder stacked layers with MHA and FeedForward blocks with configurable normalization (`"batch"`, `"instance"`, `"layer"`).

### C. POMO Model & Policy Chain (`rl4co/models/zoo/pomo/` & `rl4co/models/zoo/am/`)
- **`rl4co/models/zoo/pomo/model.py`**:
  - Lines 16-86: `POMO` inherits from `REINFORCE`. Default policy is `AttentionModelPolicy` with:
    - `num_encoder_layers=6`
    - `normalization="instance"`
    - `use_graph_context=False`
  - Lines 88-143: `shared_step` runs multi-start decoding (`n_start`), unbatches rewards to `[batch_size, num_augment, num_starts]`, calculates shared POMO baseline loss.
- **`rl4co/models/zoo/am/policy.py`**:
  - Lines 10-122: `AttentionModelPolicy` inherits from `AutoregressivePolicy` -> `ConstructivePolicy`.
  - Instantiates `AttentionModelEncoder` and `AttentionModelDecoder`.
- **`rl4co/models/common/constructive/base.py`**:
  - Lines 154-245: `ConstructivePolicy.forward`:
    1. Calls `hidden, init_embeds = self.encoder(td)`.
    2. Calls `td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_starts)`.
    3. Runs autoregressive decoding loop: `logits, mask = self.decoder(td, hidden, num_starts)`, steps environment until `td["done"].all()`.
    4. Computes log-likelihood and rewards.
- **`rl4co/models/zoo/am/encoder.py`**:
  - Lines 81-87: `AttentionModelEncoder.forward` computes `init_h = self.init_embedding(td)` and `h = self.net(init_h, mask)`, returning `(h, init_h)`.
- **`rl4co/models/zoo/am/decoder.py`**:
  - Lines 21-41: `PrecomputedCache` dataclass stores `node_embeddings`, `graph_context`, `glimpse_key`, `glimpse_val`, `logit_key`.
  - Lines 195-228: `_precompute_cache` projects `node_embeddings` to `glimpse_key`, `glimpse_val`, `logit_key`. If `use_graph_context=False` (POMO standard), `graph_context = 0`.
  - Lines 128-140: `_compute_q` computes `step_context = self.context_embedding(node_embeds_cache, td)` and `glimpse_q = step_context + graph_context_cache`.

### D. Existing Unit Tests (`tests/`)
- **`tests/test_insertion_cost.py`**:
  - Tests `compute_pairwise_distance_matrix` (Lines 6-15) and `compute_marginal_insertion_cost` (Lines 17-27) and `test_knn_sparsification` (Lines 29-38).
- **`tests/test_policy.py`**:
  - Tests `AttentionModelPolicy` with single-start, multi-start, and beam search decoding.
- **`tests/test_training.py`**:
  - Tests `REINFORCE`, `POMO`, `SymNCO` training loops using `RL4COTrainer`.

### E. Configurations (`configs/` / Hydra)
- **`configs/model/pomo.yaml`**: `_target_: rl4co.models.POMO`, `num_augment: 8`, `metrics` setup.
- **`configs/model/am.yaml`**: `_target_: rl4co.models.AttentionModel`.
- **`run.py`**: Entry point executing `from rl4co.tasks.train import train`.

---

## 2. Logic Chain

1. **Observation 1 (Encoder & Decoder Separation)**:
   - In `AttentionModelPolicy`, `ConstructivePolicy.forward` executes `hidden, init_embeds = self.encoder(td)` followed by `self.decoder.pre_decoder_hook(td, env, hidden, num_starts)`.
   - `AttentionModelEncoder` outputs node embeddings $Z \in \mathbb{R}^{B \times N \times d}$.
   - **Deduction**: Slot Attention can be inserted directly after the encoder pass inside `POMOSlotPolicy.forward` (or in a custom encoder / pre-decoder hook), consuming node embeddings $Z$ to produce slot representations $Z_{\text{slot}} \in \mathbb{R}^{B \times K \times d}$ and attention assignment matrix $A_{\text{slot}} \in \mathbb{R}^{B \times K \times N}$.

2. **Observation 2 (Decoder Query & Conditioning Injection)**:
   - `AttentionModelDecoder._compute_q` calculates query `glimpse_q = step_context + graph_context_cache`.
   - In standard POMO, `graph_context_cache` is `0` because `use_graph_context=False`.
   - **Deduction**: Slot embeddings $Z_{\text{slot}}$ can be injected into the POMO decoder conditioning in three modular ways:
     - *Option 1*: Setting `graph_context_cache = Linear_slot(Pool(Z_slot))` inside `_precompute_cache` in a customized `POMOSlotDecoder`.
     - *Option 2*: Passing slot conditioning into a custom `POMOSlotContext` module as part of `step_context`.
     - *Option 3*: Overriding `POMOSlotDecoder._compute_q` to add $\mathbf{c}_{\text{slot}} = \text{Linear}(\text{Pool}(Z_{\text{slot}}))$ to `glimpse_q`.
   - Option 1 is the cleanest and most modular because `graph_context_cache` is broadcasted and added to `step_context` at every decoding step without modifying the inner loop step logic.

3. **Observation 3 (Data Pipeline & Insertion Cost Reusability)**:
   - `rl4co/data/insertion_cost.py` already implements vectorized pairwise distance `compute_pairwise_distance_matrix` and k-NN sparsified marginal insertion cost matrix `compute_marginal_insertion_cost(locs, k_neighbors=15)`.
   - `rl4co/data/utils.py` contains `save_tensordict_to_npz` and `load_npz_to_tensordict`.
   - **Deduction**: Offline caching script `rl4co/data/generate_slot_dataset.py` (Required by R1) can directly call `generate_tsp_data` / `generate_vrp_data`, compute $d_{\text{ins}}$ via `compute_marginal_insertion_cost`, and save the combined `TensorDict` (or `.pt` / `.npz` file) containing coordinates and sparsified $d_{\text{ins}}$ matrices.

4. **Observation 4 (Metric Loss & Dual Ascent Integration)**:
   - `POMO.shared_step` computes model loss via `self.calculate_loss(td, batch, out, reward, log_likelihood)`.
   - **Deduction**: In `POMOSlot` (inheriting from `POMO`), `calculate_loss` can be overridden to compute:
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RL}} + \alpha \cdot \mathcal{L}_{\text{metric}} + \lambda \cdot (\text{Constraint}) + \gamma \cdot \mathcal{L}_{\text{entropy}}$$
     where `MetricLoss` module (`rl4co/models/nn/metric_loss.py`) computes projection $\phi(z_k)$, evaluates Euclidean distance or $d_{\text{ins}}$ distance between slots, updates dual multiplier $\lambda$, and regularizes slot entropy.

5. **Observation 5 (Hydra Config Integration & Model Toggles)**:
   - Model configs are located in `configs/model/`. `pomo.yaml` uses `_target_: rl4co.models.POMO`.
   - **Deduction**: Creating `configs/model/pomo_slot_a.yaml` through `configs/model/pomo_slot_e.yaml` setting `_target_: rl4co.models.POMOSlot` with variant toggles (`variant: "A"`, `"B"`, `"C"`, `"D"`, `"E"`) allows single CLI execution via `python run.py model=pomo_slot_a env=tsp`.

---

## 3. Caveats

1. **Hydra Config Directory Naming**: The user request referenced `conf/` for Hydra configurations, but the repository layout uses `configs/` (specifically `configs/model/`, `configs/env/`, etc.). All new configs must be placed in `configs/model/pomo_slot_*.yaml` to conform to `rl4co` conventions.
2. **TensorDict Serialization**: PyTorch `.pt` vs `.npz`: `rl4co/data/utils.py` uses `.npz` for fast numpy arrays in `TensorDict`. Requirement R1 requests `.pt` disk cache for precomputed instances + sparsified $d_{\text{ins}}$ matrices; using `torch.save` / `torch.load` for `.pt` files ensures full PyTorch Tensor type precision (including `float("inf")` masks for k-NN).
3. **Multi-start Decoding & Slot Embeddings**: Slot Attention is performed once per instance during the encoder pass before multi-start expansion ($N$ start nodes), ensuring low computational overhead during decoding.

---

## 4. Conclusion

The `rl4co` repository has a clean, highly modular architecture suitable for implementing Metric-Aware Slot Abstraction NCO.

### Summary of Component Placements:

| Component | Target File Location | Core Class / Functions | Wiring Point |
| :--- | :--- | :--- | :--- |
| **Data Engine & $d_{\text{ins}}$ Cost** | `rl4co/data/insertion_cost.py`<br>`rl4co/data/generate_slot_dataset.py` | `compute_marginal_insertion_cost`<br>`generate_slot_dataset` | Reuses existing `compute_marginal_insertion_cost`. Generates & caches Uniform & Clustered GMM instances + $d_{\text{ins}}$ matrices. |
| **Slot Attention** | `rl4co/models/nn/slot_attention.py` | `SlotAttention` | Receives encoder node embeddings $Z \in \mathbb{R}^{B \times N \times d}$, returns $Z_{\text{slot}} \in \mathbb{R}^{B \times K \times d}$ and attention maps $A_{\text{slot}}$. |
| **POMO Slot Policy & Decoder** | `rl4co/models/zoo/pomo_slot/policy.py`<br>`rl4co/models/zoo/pomo_slot/decoder.py` | `POMOSlotPolicy`<br>`POMOSlotDecoder` | Inject pooled slot embeddings into `PrecomputedCache.graph_context` or decoder query $\mathbf{q}_t$. |
| **METRA Metric Loss & Dual Ascent** | `rl4co/models/nn/metric_loss.py`<br>`rl4co/models/zoo/pomo_slot/model.py` | `MetricLoss`<br>`POMOSlot` | Called in `POMOSlot.calculate_loss`. Implements projection $\phi(z_k)$, Lagrangian dual ascent for $\lambda$, and slot entropy regularization. |
| **Hydra Configurations** | `configs/model/pomo_slot_a.yaml` ... `pomo_slot_e.yaml` | Hydra YAML specs | Model variant toggles A (Baseline POMO), B (Slot Task-only), C (Euclidean METRA), D (Insertion Cost METRA), E (Full Dual Ascent). |

---

## 5. Verification Method

To independently verify the codebase mapping and existing functionality:

1. **Run existing unit tests**:
   ```bash
   pytest tests/test_insertion_cost.py
   pytest tests/test_policy.py -k test_am_policy
   pytest tests/test_training.py -k test_reinforce
   ```
2. **Inspect data utilities**:
   - Confirm `rl4co/data/insertion_cost.py` functions:
     `python -c "import torch; from rl4co.data.insertion_cost import compute_marginal_insertion_cost; print(compute_marginal_insertion_cost(torch.rand(2, 10, 2), k_neighbors=3).shape)"`
3. **Verify Hydra integration**:
   - Check existing model configs under `configs/model/`.
