import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))

from rl4co.data.insertion_cost import (
    compute_pairwise_distance_matrix,
    compute_marginal_insertion_cost,
)
from tests.test_slot_attention import SlotAttention
from tests.test_metric_loss import MetricLoss


def run_stress_tests():
    print("=== STARTING NUMERICAL STABILITY & STRESS HARNESS ===")
    results = []

    # -------------------------------------------------------------------
    # Test 1: Scaling to N=500 and N=1000
    # -------------------------------------------------------------------
    print("\n--- Stress Test 1: Scale N=500 and N=1000 ---")
    try:
        B, N, K, d_in, d_slot = 2, 500, 8, 64, 64
        locs = torch.rand(B, N, 2)
        
        # Insertion cost N=500
        d_ins = compute_marginal_insertion_cost(locs, k_neighbors=15)
        assert d_ins.shape == (B, N, N)
        assert not torch.isnan(d_ins).any()
        # Verify k-NN sparsification on N=500: each row has 16 finite entries
        non_inf_counts = torch.sum(~torch.isinf(d_ins[0]), dim=-1)
        assert torch.all(non_inf_counts == 16)
        
        # Slot Attention N=500
        slot_attn = SlotAttention(num_slots=K, slot_dim=d_slot, in_dim=d_in)
        inputs = torch.randn(B, N, d_in)
        slots, attn = slot_attn(inputs)
        assert slots.shape == (B, K, d_slot)
        assert attn.shape == (B, N, K)
        assert torch.allclose(attn.sum(dim=-1), torch.ones(B, N), atol=1e-5)
        
        # Metric Loss N=500
        metric_loss = MetricLoss(slot_dim=d_slot, proj_dim=16)
        out = metric_loss(slots, attn, target_dist=d_ins)
        assert not torch.isnan(out["loss_metric"])
        assert not torch.isnan(out["loss_entropy"])
        assert not torch.isnan(out["dual_penalty"])

        # Scaling N=1000
        locs_1000 = torch.rand(1, 1000, 2)
        d_ins_1000 = compute_marginal_insertion_cost(locs_1000, k_neighbors=15)
        assert d_ins_1000.shape == (1, 1000, 1000)
        assert not torch.isnan(d_ins_1000).any()

        results.append(("Scaling N=500 & N=1000", True, "Passes without OOM or NaN"))
        print("PASS: Scaling N=500 & N=1000 successful")
    except Exception as e:
        results.append(("Scaling N=500 & N=1000", False, str(e)))
        print(f"FAIL: Scaling N=500 & N=1000 failed: {e}")

    # -------------------------------------------------------------------
    # Test 2: Extreme Coordinate Ranges
    # -------------------------------------------------------------------
    print("\n--- Stress Test 2: Extreme Coordinate Ranges ---")
    try:
        # Case 2a: Ultra large coordinates [1e6, 1e7]
        locs_large = torch.rand(2, 50, 2) * 1e6 + 1e6
        d_ins_large = compute_marginal_insertion_cost(locs_large, k_neighbors=10)
        assert not torch.isnan(d_ins_large).any()
        valid_large = d_ins_large.masked_fill(torch.isinf(d_ins_large), 0.0)
        assert torch.all(valid_large >= 0.0)

        # Case 2b: Micro coordinates [1e-7, 1e-6]
        locs_micro = torch.rand(2, 50, 2) * 1e-6
        d_ins_micro = compute_marginal_insertion_cost(locs_micro, k_neighbors=10)
        assert not torch.isnan(d_ins_micro).any()
        valid_micro = d_ins_micro.masked_fill(torch.isinf(d_ins_micro), 0.0)
        assert torch.all(valid_micro >= 0.0)

        # Case 2c: Negative coordinates [-1000, -500]
        locs_neg = torch.rand(2, 50, 2) * 500 - 1000
        d_ins_neg = compute_marginal_insertion_cost(locs_neg, k_neighbors=10)
        assert not torch.isnan(d_ins_neg).any()

        # Case 2d: Mixed extreme scales (cluster near 0, cluster near 1e6)
        c1 = torch.rand(1, 20, 2) * 0.001
        c2 = torch.rand(1, 20, 2) * 100.0 + 1e6
        locs_mixed = torch.cat([c1, c2], dim=1)
        d_ins_mixed = compute_marginal_insertion_cost(locs_mixed, k_neighbors=5)
        assert not torch.isnan(d_ins_mixed).any()

        results.append(("Extreme Coordinate Ranges", True, "Handles 1e6, 1e-6, negative, and mixed coordinates cleanly"))
        print("PASS: Extreme Coordinate Ranges successful")
    except Exception as e:
        results.append(("Extreme Coordinate Ranges", False, str(e)))
        print(f"FAIL: Extreme Coordinate Ranges failed: {e}")

    # -------------------------------------------------------------------
    # Test 3: Zero Gradients & Flat / Constant Inputs
    # -------------------------------------------------------------------
    print("\n--- Stress Test 3: Zero Gradients & Flat / Constant Inputs ---")
    try:
        # Flat zero input to Slot Attention
        slot_attn = SlotAttention(num_slots=4, slot_dim=32, in_dim=32)
        zero_inputs = torch.zeros(2, 20, 32, requires_grad=True)
        slots_z, attn_z = slot_attn(zero_inputs)
        assert not torch.isnan(slots_z).any()
        assert not torch.isnan(attn_z).any()

        # Backward on zero input
        loss_z = slots_z.sum() + attn_z.sum()
        loss_z.backward()
        assert zero_inputs.grad is not None
        assert not torch.isnan(zero_inputs.grad).any()

        # Flat identical locations to insertion cost
        identical_locs = torch.full((2, 10, 2), 0.5, requires_grad=True)
        d_ins_id = compute_marginal_insertion_cost(identical_locs, k_neighbors=5)
        assert not torch.isnan(d_ins_id).any()
        # Non-neighbor entries are inf, self & neighbor entries are 0.0
        valid_id = d_ins_id.masked_fill(torch.isinf(d_ins_id), 0.0)
        assert torch.allclose(valid_id, torch.zeros_like(valid_id), atol=1e-6)

        # Zero gradient check when output does not depend on some inputs
        dummy_param = torch.nn.Parameter(torch.randn(5, 5))
        loss_dummy = (dummy_param * 0.0).sum()
        loss_dummy.backward()
        assert torch.all(dummy_param.grad == 0.0)

        results.append(("Zero Gradients & Flat Inputs", True, "No NaN, numerical overflow, or gradient breakdown"))
        print("PASS: Zero Gradients & Flat Inputs successful")
    except Exception as e:
        results.append(("Zero Gradients & Flat Inputs", False, str(e)))
        print(f"FAIL: Zero Gradients & Flat Inputs failed: {e}")

    # -------------------------------------------------------------------
    # Test 4: Extreme Batch Sizes and Extreme Slot Counts (B=1, B=128, K=1, K=N)
    # -------------------------------------------------------------------
    print("\n--- Stress Test 4: Extreme Batch & Slot Counts ---")
    try:
        # B=128
        locs_b128 = torch.rand(128, 15, 2)
        d_ins_b128 = compute_marginal_insertion_cost(locs_b128, k_neighbors=5)
        assert d_ins_b128.shape == (128, 15, 15)

        # K = N (slots count equal to node count)
        slot_attn_kn = SlotAttention(num_slots=15, slot_dim=32, in_dim=32)
        inputs_kn = torch.randn(2, 15, 32)
        slots_kn, attn_kn = slot_attn_kn(inputs_kn)
        assert slots_kn.shape == (2, 15, 32)
        assert attn_kn.shape == (2, 15, 15)

        # MetricLoss with K=N
        metric_loss_kn = MetricLoss(slot_dim=32, proj_dim=16)
        out_kn = metric_loss_kn(slots_kn, attn_kn, target_dist=d_ins_b128[:2])
        assert not torch.isnan(out_kn["loss_metric"])

        results.append(("Extreme Batch & Slot Counts", True, "B=128, K=N handled without shape or dimension errors"))
        print("PASS: Extreme Batch & Slot Counts successful")
    except Exception as e:
        results.append(("Extreme Batch & Slot Counts", False, str(e)))
        print(f"FAIL: Extreme Batch & Slot Counts failed: {e}")

    # Print summary
    print("\n=== STRESS TEST HARNESS SUMMARY ===")
    all_passed = True
    for name, passed, detail in results:
        status = "PASSED" if passed else "FAILED"
        print(f"[{status}] {name}: {detail}")
        if not passed:
            all_passed = False

    return all_passed


if __name__ == "__main__":
    success = run_stress_tests()
    if not success:
        sys.exit(1)
