## 2026-08-06T07:40:03Z
Extract detailed requirements and feature inventory for R1 - R4 and Milestones M0 - M8.
Your working directory is `d:/NCO NEW/rl4co/.agents/spec_miner_survey_1`.
You MUST read `d:/NCO NEW/rl4co/.agents/ORIGINAL_REQUEST.md` first.

Tasks:
1. Mine and extract every feature, input specification, mathematical constraint, CLI option, and verification requirement for:
   - R1: k-NN Sparsified d_ins Cache (M0 & M1) in `rl4co/data/insertion_cost.py` & `rl4co/data/generate_slot_dataset.py` (Uniform & Clustered GMM for N in {50, 100, 200, 500}, default k=15, .pt caching).
   - R2: Slot Attention & POMO Policy Wiring (M2 & M3) in `rl4co/models/nn/slot_attention.py` & `rl4co/models/zoo/pomo_slot/policy.py` (Variant B first).
   - R3: METRA Metric Loss & Dual Ascent (M4, M5, M6) in `rl4co/models/nn/metric_loss.py` & `rl4co/models/zoo/pomo_slot/model.py` (Variants A, B, C, D, E toggles).
   - R4: Hydra Configs & Decision Checkpoint M8 (M7 & M8) in `conf/model/pomo_slot_a.yaml` through `pomo_slot_e.yaml` & evaluation test script (multi-seed logging, optimality gap, ARI stability, slot entropy).
2. Format as a comprehensive Feature Inventory matrix with explicit acceptance criteria for each milestone.
3. Write your report to `d:/NCO NEW/rl4co/.agents/spec_miner_survey_1/handoff.md`.
