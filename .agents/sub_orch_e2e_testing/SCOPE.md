# Scope: E2E Testing Track

## Architecture
Requirement-driven, opaque-box test suite for Metric-Aware Slot Abstraction NCO in `rl4co`.
1. Tier 1: Feature Coverage (>=5 test cases per feature across R1-R4)
2. Tier 2: Boundary & Corner Cases (edge cases: $N \le k$, single instance, $K=1$, zero embeddings, infinite non-neighbors, $A_{ik}=0.0$, numerical stability)
3. Tier 3: Cross-Feature Combinations (pairwise interactions: $d_{\text{ins}}$ + SlotAttention, SlotAttention + MetricLoss, METRA + POMOSlotPolicy)
4. Tier 4: Real-World Application Scenarios (end-to-end CVRP & TSP runs across variants A-E on $N \in \{50, 100\}$)

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Test Infra & Tier 1-2 Unit Tests | `TEST_INFRA.md`, `tests/test_insertion_cost.py`, `tests/test_slot_attention.py`, `tests/test_metric_loss.py` | none | DONE |
| 2 | Tier 3-4 Integration & Eval Tests | `tests/test_pomo_slot_eval.py`, multi-seed runner verification, publish `TEST_READY.md` | Milestone 1 | IN_PROGRESS |

## Code Layout
- `TEST_INFRA.md` (project root)
- `TEST_READY.md` (project root)
- `tests/test_insertion_cost.py`
- `tests/test_slot_attention.py`
- `tests/test_metric_loss.py`
- `tests/test_pomo_slot_eval.py`
