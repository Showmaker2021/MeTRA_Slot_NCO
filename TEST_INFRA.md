# Test Infrastructure Specification — Metric-Aware Slot Abstraction NCO (`rl4co`)

## 1. Overview & Test Philosophy

This document specifies the end-to-end testing infrastructure and test architecture for the **Metric-Aware Slot Abstraction NCO Method** integrated into `rl4co`. 

The test harness follows a **requirement-driven, opaque-box multi-tiered testing methodology** designed to ensure mathematical correctness, contract integrity, numerical stability, and regression safety across all 12 system features (R1–R4, Milestones M0–M8).

### Key Verification Goals
1. **Mathematical Rigor**: Rigorous assertion of theoretical invariants (e.g., softmax normalization $\sum_k A_{ik} = 1.0$, diagonal self-insertion cost $d_{\text{ins}}(i,i) = 0.0$, non-negative dual variable $\lambda \ge 0.0$, entropy bounds $0.0 \le H(A) \le \log K$).
2. **Numerical Stability**: Comprehensive testing under zero inputs, high-variance embeddings, sparsified infinite non-neighbor masking (`float('inf')`), and dual ascent clamping.
3. **Contract Adherence**: Strict validation of tensor shapes, batch dimension propagation, device compatibility, and module interface compliance.
4. **Progressive Testability**: Graceful import degradation with reference fallbacks during incremental milestone rollouts, turning into strict contract assertions once modules are present.

---

## 2. Multi-Tier Test Methodology

The testing architecture is organized into four distinct tiers, combining Category-Partitioning, Boundary Value Analysis (BVA), Pairwise Interactions, and Real-World Application Benchmarking.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Tier 4: Real-World Scenarios                       │
│      End-to-End CVRP & TSP Runs across Variants A–E (N in {50, 100})     │
├─────────────────────────────────────────────────────────────────────────┤
│                   Tier 3: Cross-Feature Integration                     │
│    d_ins + SlotAttention | SlotAttention + MetricLoss | METRA + Policy  │
├─────────────────────────────────────────────────────────────────────────┤
│                 Tier 2: Boundary & Corner Cases (BVA)                   │
│   N <= k | K=1 | Zero Embeddings | Inf Masking | Crisp/Uniform Entropy  │
├─────────────────────────────────────────────────────────────────────────┤
│                     Tier 1: Feature Coverage                            │
│   Analytical Right Triangle | Softmax Sum=1.0 | Dual Ascent | Projections │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tier 1: Feature Coverage (Category-Partitioning)
- Focuses on primary happy-path behavior, analytical edge solutions, formula verification, and shape contracts.
- Guarantees $\ge 5$ explicit unit test cases per core feature module.

### Tier 2: Boundary Value Analysis & Edge Cases
- Focuses on boundary conditions, extreme hyperparameter settings, zero/infinite values, and numerical stress points.
- Specific edge cases: $N \le k$, unbatched 2D inputs, single slot $K=1$, zero inputs $H=\mathbf{0}$, infinite non-neighbor soft-aggregation, crisp vs uniform slot assignments, dual multiplier non-negativity.

### Tier 3: Cross-Feature Integration
- Validates inter-module data pipelines and gradient flow across module boundaries:
  - Data Engine ($d_{\text{ins}}$) $\to$ Slot Attention (soft maps $A_{ik}$).
  - Slot Attention ($z_k, A_{ik}$) $\to$ Metric Loss ($\phi(z_k)$, dual ascent, entropy).
  - Metric Loss $\to$ POMO Slot Policy / LightningModule variant loss dictionary.

### Tier 4: Real-World Application Scenarios
- Validates full end-to-end execution of POMO Slot training and evaluation loops on standard CVRP and TSP benchmarks ($N=50, N=100$).
- Asserts multi-seed stability, Adjusted Rand Index (ARI) convergence, and optimality gap metrics for Milestone M8 decision gate evaluation.

---

## 3. Feature Mapping Across All 12 Features

Every feature specified in `PROJECT.md` maps to dedicated test modules and test tiers:

| Feature # | Feature Description | Core Implementation Module | Primary Test File | Applicable Tiers |
|---|---|---|---|---|
| **1** | $k$-NN Sparsified $d_{\text{ins}}$ Operator | `rl4co/data/insertion_cost.py` | `tests/test_insertion_cost.py` | Tier 1, 2, 3 |
| **2** | Unit Test Insertion Cost Suite | `tests/test_insertion_cost.py` | `tests/test_insertion_cost.py` | Tier 1, 2 |
| **3** | Offline Dataset Caching CLI | `rl4co/data/generate_slot_dataset.py` | `tests/test_insertion_cost.py` | Tier 2, 4 |
| **4** | Standalone `SlotAttention` Layer | `rl4co/models/nn/slot_attention.py` | `tests/test_slot_attention.py` | Tier 1, 2, 3 |
| **5** | Unit Test Slot Attention Suite | `tests/test_slot_attention.py` | `tests/test_slot_attention.py` | Tier 1, 2 |
| **6** | POMO Policy Wiring (Variant B) | `rl4co/models/zoo/pomo_slot/policy.py` | `tests/test_pomo_slot_eval.py` | Tier 3, 4 |
| **7** | METRA Metric Loss & Dual Ascent | `rl4co/models/nn/metric_loss.py` | `tests/test_metric_loss.py` | Tier 1, 2, 3 |
| **8** | Unit Test Metric Loss Suite | `tests/test_metric_loss.py` | `tests/test_metric_loss.py` | Tier 1, 2 |
| **9** | Model Variant Toggles (A–E) | `rl4co/models/zoo/pomo_slot/model.py` | `tests/test_pomo_slot_eval.py` | Tier 3, 4 |
| **10** | Model Variant CLI Execution | CLI / `pomo_slot/model.py` | `tests/test_pomo_slot_eval.py` | Tier 4 |
| **11** | Hydra Configurations (A–E) | `conf/model/pomo_slot_*.yaml` | `tests/test_pomo_slot_eval.py` | Tier 3, 4 |
| **12** | M8 Decision Gate Benchmark | `scripts/eval_pomo_slot.py` | `tests/test_pomo_slot_eval.py` | Tier 4 |

---

## 4. Test Directory Layout

```
rl4co/
├── TEST_INFRA.md                      # Project-wide test architecture & specs (this file)
├── TEST_READY.md                      # Final test readiness validation report
├── rl4co/                             # Source code
│   ├── data/
│   │   ├── insertion_cost.py          # Feature 1: d_ins and k-NN sparsification
│   │   └── generate_slot_dataset.py   # Feature 3: Offline caching CLI
│   ├── models/
│   │   ├── nn/
│   │   │   ├── slot_attention.py      # Feature 4: Differentiable Slot Attention
│   │   │   └── metric_loss.py         # Feature 7: METRA Loss & Dual Ascent
│   │   └── zoo/
│   │       └── pomo_slot/
│   │           ├── policy.py          # Feature 6: POMO slot conditioning
│   │           └── model.py           # Feature 9: Variant A-E toggles
└── tests/                             # Co-located unit and integration test suite
    ├── test_insertion_cost.py         # Features 1 & 2: Tier 1 & 2 insertion cost tests
    ├── test_slot_attention.py         # Features 4 & 5: Tier 1 & 2 slot attention tests
    ├── test_metric_loss.py            # Features 7 & 8: Tier 1 & 2 metric loss tests
    └── test_pomo_slot_eval.py         # Features 6, 9-12: Integration & benchmark tests
```

---

## 5. Test Harness & Pytest Commands

The test suite leverages `pytest` and `torch.testing` with seed control (`torch.manual_seed(42)`).

### Running Fast Unit Tests (Tier 1 & 2)
```bash
# Execute unit test suites for data engine, slot attention, and metric loss
pytest tests/test_insertion_cost.py tests/test_slot_attention.py tests/test_metric_loss.py -v
```

### Running Integration Tests (Tier 3 & 4)
```bash
# Execute integration and policy evaluation tests
pytest tests/test_pomo_slot_eval.py -v
```

### Executing Full Suite with Coverage
```bash
# Run all tests and generate terminal coverage report for rl4co
pytest --cov=rl4co/data --cov=rl4co/models/nn --cov-report=term-missing
```

---

## 6. Verification Criteria & Acceptance Gates

To consider the test infrastructure and feature implementation complete:

1. **Pass Rate**: 100% of test cases in `tests/test_insertion_cost.py`, `tests/test_slot_attention.py`, `tests/test_metric_loss.py`, and `tests/test_pomo_slot_eval.py` MUST pass cleanly.
2. **Zero NaN Propagation**: No operation under zero input, infinite masking, or dual ascent step may produce `NaN` or unhandled `inf` values.
3. **Contract Guarantees**:
   - `d_ins`: Shape $(B, N, N)$, $d_{\text{ins}}(i,i) = 0.0$, finite elements per node $\le k+1$.
   - `SlotAttention`: `slots` $(B, K, d)$, `attn` $(B, N, K)$, $\sum_k A_{ik} = 1.0 \pm 10^{-6}$.
   - `MetricLoss`: $\phi(z_k) \in (B, K, d_{\text{proj}})$, $\lambda \ge 0.0$, $0.0 \le H(A) \le \log K$.
4. **Independent Reproducibility**: Tests must be self-contained, isolated, deterministic across seeds, and execute cleanly on both CPU and CUDA hardware environments.
