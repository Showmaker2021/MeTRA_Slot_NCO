# Dispatch Log

## 2026-08-06T07:39:38Z

Implement the Metric-Aware Slot Abstraction for Neural Vehicle Routing method integrated directly into the `rl4co` repository (`d:/NCO NEW/rl4co`), referencing `references/slot-attention` and `references/METRA`.

### Key Requirements (R1 - R4) & Milestones (M0 - M8)

#### R1. Data Engine & Sparsified d_ins Cache (Milestones M0 & M1)
- **k-NN Sparsification**: Vectorized cheapest insertion d_ins(i,j) restricted to k-nearest neighbors (default k=15) in `rl4co/data/insertion_cost.py`.
- **Offline Caching**: Create `rl4co/data/generate_slot_dataset.py` to precompute and cache instances + sparsified d_ins matrices to disk (.pt format).
- **Data Distributions**: Support both Uniform and Clustered (Gaussian Mixture) distributions for N in {50, 100, 200, 500}.

#### R2. Slot Attention & POMO Policy Wiring (Milestones M2 & M3)
- **Modular SlotAttention**: Implement standalone Slot Attention in `rl4co/models/nn/slot_attention.py`.
- **Policy Wiring**: Wire slot embeddings z_k into POMO decoder conditioning in `rl4co/models/zoo/pomo_slot/policy.py`. Test Variant B (alpha=0, task-only loss) first.

#### R3. METRA Metric Loss & Dual Ascent (Milestones M4, M5, M6)
- **Dual Ascent Stability**: Implement `rl4co/models/nn/metric_loss.py` with projection head phi(z_k), lower-bound Lagrangian constraint, dual parameter lambda update, and slot entropy regularization. Test Variant C (Euclidean) first before Variant D (Insertion Cost).
- **Config Toggles**: Integrate Variants A, B, C, D, E via clean model toggles in `rl4co/models/zoo/pomo_slot/model.py`.

#### R4. Hydra Configs & Decision Checkpoint M8 (Milestones M7 & M8)
- **Hydra Configurations**: Create YAML configs under `conf/model/pomo_slot_a.yaml` through `pomo_slot_e.yaml`.
- **Decision Gate (M8)**: Provide evaluation test script to compare A-E on N=50 with multi-seed logging (optimality gap, ARI stability, slot entropy).

### Acceptance Criteria
- Unit tests in `tests/test_insertion_cost.py`, `tests/test_slot_attention.py`, and `tests/test_metric_loss.py` pass cleanly.
- Dataset generator creates and caches Uniform & Clustered datasets cleanly.
- Model variant toggles A through E run via single CLI execution.
