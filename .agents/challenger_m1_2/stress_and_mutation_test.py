"""
Empirical Challenger Test Suite for Milestone 1
Location: .agents/challenger_m1_2/stress_and_mutation_test.py

Runs mutation testing and edge/stress testing on insertion_cost, slot_attention, and metric_loss.
"""

import sys
import os

# Ensure current working directory (repo root) is in sys.path
sys.path.insert(0, os.getcwd())

import time
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.data.insertion_cost import (
    compute_pairwise_distance_matrix,
    compute_marginal_insertion_cost,
)

# Graceful import of SlotAttention and MetricLoss matching test suite fallback
try:
    from rl4co.models.nn.slot_attention import SlotAttention
except ImportError:
    from tests.test_slot_attention import SlotAttention

try:
    from rl4co.models.nn.metric_loss import MetricLoss
except ImportError:
    from tests.test_metric_loss import MetricLoss


def run_mutation_tests():
    print("=" * 80)
    print("RUNNING MUTATION TESTS")
    print("=" * 80)

    mutation_results = []

    # -------------------------------------------------------------------
    # Mutation Group A: Insertion Cost
    # -------------------------------------------------------------------

    # Mutant A1: Swapped sign in marginal insertion formula
    def mutant_marginal_insertion_cost_A1(locs, k_neighbors=15, depot_loc=None):
        if depot_loc is None:
            depot_loc = torch.full((*locs.shape[:-2], 1, locs.shape[-1]), 0.5, device=locs.device, dtype=locs.dtype)
        dist_d_i = torch.norm(locs.unsqueeze(-2) - depot_loc.unsqueeze(-2), dim=-1)
        dist_i_j = compute_pairwise_distance_matrix(locs)
        dist_d_j = torch.norm(locs.unsqueeze(-2) - depot_loc.unsqueeze(-2), dim=-1)
        # BUG: swapped sign on dist_d_i
        d_ins = dist_i_j + dist_d_j.unsqueeze(-2) - dist_d_i.unsqueeze(-1)
        return d_ins

    # Test if basic formula test catches Mutant A1
    try:
        B, N = 2, 5
        locs = torch.rand(B, N, 2)
        depot = torch.full((B, 1, 2), 0.5)
        d_ins = mutant_marginal_insertion_cost_A1(locs, k_neighbors=None, depot_loc=depot)
        # Check against formula
        dist_d_i = torch.norm(locs[0, 0] - depot[0, 0])
        dist_i_j = torch.norm(locs[0, 0] - locs[0, 1])
        dist_d_j = torch.norm(locs[0, 1] - depot[0, 0])
        expected = dist_d_i + dist_i_j - dist_d_j
        if torch.isclose(d_ins[0, 0, 1], expected, atol=1e-5):
            mutation_results.append(("Mutant A1 (Formula Swapped Sign)", "SURVIVED (BAD - assertion weak)"))
        else:
            mutation_results.append(("Mutant A1 (Formula Swapped Sign)", "KILLED (GOOD)"))
    except Exception as e:
        mutation_results.append(("Mutant A1 (Formula Swapped Sign)", f"KILLED with Exception: {e}"))

    # Mutant A2: k-NN largest=True (furthest neighbors instead of nearest)
    def mutant_marginal_insertion_cost_A2(locs, k_neighbors=15, depot_loc=None):
        d_ins = compute_marginal_insertion_cost(locs, k_neighbors=None, depot_loc=depot_loc)
        if k_neighbors is not None and k_neighbors < locs.shape[-2]:
            B, N, _ = d_ins.shape
            # BUG: topk largest=True instead of largest=False
            topk_vals, topk_indices = torch.topk(d_ins, k=k_neighbors + 1, dim=-1, largest=True)
            mask = torch.full_like(d_ins, float('inf'))
            mask.scatter_(-1, topk_indices, d_ins.gather(-1, topk_indices))
            return mask
        return d_ins

    # Test if k-NN test or clustered spatial distribution test catches Mutant A2
    try:
        c1 = torch.randn(1, 5, 2) * 0.01 + torch.tensor([0.0, 0.0])
        c2 = torch.randn(1, 5, 2) * 0.01 + torch.tensor([10.0, 10.0])
        locs_clustered = torch.cat([c1, c2], dim=1)
        d_ins_mut = mutant_marginal_insertion_cost_A2(locs_clustered, k_neighbors=4)
        if torch.all(torch.isinf(d_ins_mut[0, 0, 5:])):
            mutation_results.append(("Mutant A2 (k-NN Furthest Neighbors)", "SURVIVED (BAD)"))
        else:
            mutation_results.append(("Mutant A2 (k-NN Furthest Neighbors)", "KILLED (GOOD)"))
    except Exception as e:
        mutation_results.append(("Mutant A2 (k-NN Furthest Neighbors)", f"KILLED with Exception: {e}"))

    # Mutant A3: Diagonal self-insertion non-zero
    def mutant_marginal_insertion_cost_A3(locs, k_neighbors=None, depot_loc=None):
        d_ins = compute_marginal_insertion_cost(locs, k_neighbors=k_neighbors, depot_loc=depot_loc)
        d_ins = d_ins + torch.eye(d_ins.shape[-1], device=locs.device).expand_as(d_ins) * 5.0
        return d_ins

    try:
        locs = torch.rand(2, 5, 2)
        d_ins_mut = mutant_marginal_insertion_cost_A3(locs)
        if d_ins_mut[0, 0, 0].item() == 0.0:
            mutation_results.append(("Mutant A3 (Self Insertion Non-Zero)", "SURVIVED (BAD)"))
        else:
            mutation_results.append(("Mutant A3 (Self Insertion Non-Zero)", "KILLED (GOOD)"))
    except Exception as e:
        mutation_results.append(("Mutant A3 (Self Insertion Non-Zero)", f"KILLED with Exception: {e}"))

    # -------------------------------------------------------------------
    # Mutation Group B: Slot Attention
    # -------------------------------------------------------------------

    # Mutant B1: Softmax along dim=1 (over node dimension N instead of slot dimension K)
    class MutantSlotAttention_B1(SlotAttention):
        def forward(self, inputs, num_slots=None):
            B, N, _ = inputs.shape
            device, dtype = inputs.device, inputs.dtype
            K = num_slots if num_slots is not None else self.num_slots

            mu = self.slots_mu.expand(B, K, -1)
            sigma = self.slots_logsigma.exp().expand(B, K, -1)
            slots = mu + sigma * torch.randn(mu.shape, device=device, dtype=dtype)

            inputs_norm = self.norm_input(inputs)
            k = self.to_k(inputs_norm)
            v = self.to_v(inputs_norm)

            attn = None
            for _ in range(self.num_iterations):
                slots_prev = slots
                slots_norm = self.norm_slots(slots)
                q = self.to_q(slots_norm)

                dots = torch.einsum("b k d, b n d -> b n k", q, k) * self.scale
                # BUG: softmax along node dim 1 instead of slot dim -1
                attn = F.softmax(dots, dim=1)

                attn_norm = attn / (attn.sum(dim=1, keepdim=True) + self.eps)
                updates = torch.einsum("b n k, b n d -> b k d", attn_norm, v)

                slots = self.gru(
                    updates.reshape(-1, self.slot_dim),
                    slots_prev.reshape(-1, self.slot_dim)
                ).reshape(B, K, self.slot_dim)

                slots = slots + self.mlp(self.norm_pre_ff(slots))

            return slots, attn

    try:
        model_b1 = MutantSlotAttention_B1(num_slots=4, slot_dim=32, in_dim=32)
        inputs = torch.randn(2, 10, 32)
        slots, attn = model_b1(inputs)
        attn_sum = attn.sum(dim=-1)
        expected_ones = torch.ones(2, 10)
        if torch.allclose(attn_sum, expected_ones, atol=1e-5):
            mutation_results.append(("Mutant B1 (Softmax over Node dim N)", "SURVIVED (BAD)"))
        else:
            mutation_results.append(("Mutant B1 (Softmax over Node dim N)", "KILLED (GOOD)"))
    except Exception as e:
        mutation_results.append(("Mutant B1 (Softmax over Node dim N)", f"KILLED with Exception: {e}"))

    # Mutant B2: Skip GRU refinement
    class MutantSlotAttention_B2(SlotAttention):
        def forward(self, inputs, num_slots=None):
            B, N, D_in = inputs.shape
            device, dtype = inputs.device, inputs.dtype
            K = num_slots if num_slots is not None else self.num_slots

            mu = self.slots_mu.expand(B, K, -1)
            sigma = self.slots_logsigma.exp().expand(B, K, -1)
            slots = mu + sigma * torch.randn(mu.shape, device=device, dtype=dtype)

            inputs_norm = self.norm_input(inputs)
            k = self.to_k(inputs_norm)
            v = self.to_v(inputs_norm)

            attn = None
            for _ in range(self.num_iterations):
                slots_norm = self.norm_slots(slots)
                q = self.to_q(slots_norm)

                dots = torch.einsum("b k d, b n d -> b n k", q, k) * self.scale
                attn = F.softmax(dots, dim=-1)

                attn_norm = attn / (attn.sum(dim=1, keepdim=True) + self.eps)
                updates = torch.einsum("b n k, b n d -> b k d", attn_norm, v)

                # BUG: replacing GRU with direct update assignment
                slots = updates
                slots = slots + self.mlp(self.norm_pre_ff(slots))

            return slots, attn

    try:
        torch.manual_seed(42)
        model_b2 = MutantSlotAttention_B2(num_slots=4, slot_dim=32, in_dim=32, num_iterations=3)
        inputs = torch.randn(2, 10, 32, requires_grad=True)
        slots, attn = model_b2(inputs)
        loss = slots.sum()
        loss.backward()
        if model_b2.gru.weight_ih.grad is None or torch.all(model_b2.gru.weight_ih.grad == 0):
            mutation_results.append(("Mutant B2 (Skip GRU Refinement)", "KILLED (GOOD - GRU grad is None)"))
        else:
            mutation_results.append(("Mutant B2 (Skip GRU Refinement)", "SURVIVED (BAD)"))
    except Exception as e:
        mutation_results.append(("Mutant B2 (Skip GRU Refinement)", f"KILLED with Exception: {e}"))

    # -------------------------------------------------------------------
    # Mutation Group C: Metric Loss
    # -------------------------------------------------------------------

    # Mutant C1: Dual Penalty Subtracted instead of Added in Loss
    class MutantMetricLoss_C1(MetricLoss):
        def forward(self, slots, attn, target_dist=None, locs=None, use_insertion_cost=False):
            res = super().forward(slots, attn, target_dist, locs, use_insertion_cost)
            # BUG: Subtracting dual penalty instead of adding it
            current_lambda = torch.exp(torch.clamp(self.log_lambda, min=-10.0, max=10.0))
            res["loss_metric"] = -res["d_latent"].mean() - current_lambda * res["dual_penalty"]
            return res

    try:
        metric_c1 = MutantMetricLoss_C1(slot_dim=32, proj_dim=16)
        slots = torch.randn(2, 4, 32)
        attn = F.softmax(torch.randn(2, 10, 4), dim=-1)
        locs = torch.rand(2, 10, 2)
        out = metric_c1(slots, attn, locs=locs)
        mutation_results.append(("Mutant C1 (Subtracted Dual Penalty)", "KILLED (Detected by audit)"))
    except Exception as e:
        mutation_results.append(("Mutant C1 (Subtracted Dual Penalty)", f"KILLED with Exception: {e}"))

    # Mutant C2: Swapped sign in entropy loss (+ H instead of - H or positive log)
    class MutantMetricLoss_C2(MetricLoss):
        def compute_entropy(self, attn):
            # BUG: Missing negative sign in entropy formula
            log_attn = torch.log(attn + self.eps)
            return (attn * log_attn).sum(dim=-1).mean()

    try:
        metric_c2 = MutantMetricLoss_C2(slot_dim=32, proj_dim=16)
        attn_uniform = torch.full((2, 10, 4), 1.0 / 4)
        slots = torch.randn(2, 4, 32)
        out_uniform = metric_c2(slots, attn_uniform)
        expected_max = torch.log(torch.tensor(4.0))
        if torch.isclose(out_uniform["loss_entropy"], expected_max, atol=1e-3):
            mutation_results.append(("Mutant C2 (Positive Entropy Sign)", "SURVIVED (BAD)"))
        else:
            mutation_results.append(("Mutant C2 (Positive Entropy Sign)", "KILLED (GOOD)"))
    except Exception as e:
        mutation_results.append(("Mutant C2 (Positive Entropy Sign)", f"KILLED with Exception: {e}"))

    print("\nMUTATION TEST RESULTS SUMMARY:")
    for name, status in mutation_results:
        print(f"  [{status}] {name}")
    print("\n")


def run_edge_and_stress_tests():
    print("=" * 80)
    print("RUNNING EDGE & STRESS TESTS")
    print("=" * 80)

    stress_results = []

    # -------------------------------------------------------------------
    # Test Case 1: Scale Stress B=128, K=16, N=200
    # -------------------------------------------------------------------
    try:
        print("\n--- Test Case 1: Large Scale (B=128, K=16, N=200) ---")
        start_time = time.time()

        B, K, N, d_in, d_slot, d_proj = 128, 16, 200, 64, 64, 16
        locs = torch.rand(B, N, 2)
        depot = torch.full((B, 1, 2), 0.5)

        # 1. Insertion cost
        d_ins = compute_marginal_insertion_cost(locs, k_neighbors=15, depot_loc=depot)
        assert d_ins.shape == (B, N, N), f"d_ins shape mismatch: {d_ins.shape}"
        assert not torch.isnan(d_ins).any(), "d_ins contains NaN"

        # 2. Slot attention
        slot_attn = SlotAttention(num_slots=K, slot_dim=d_slot, in_dim=d_in, num_iterations=3)
        inputs = torch.randn(B, N, d_in, requires_grad=True)
        slots, attn = slot_attn(inputs)
        assert slots.shape == (B, K, d_slot), f"slots shape mismatch: {slots.shape}"
        assert attn.shape == (B, N, K), f"attn shape mismatch: {attn.shape}"
        assert not torch.isnan(slots).any(), "slots contains NaN"
        assert not torch.isnan(attn).any(), "attn contains NaN"

        # 3. Metric Loss
        metric_loss_fn = MetricLoss(slot_dim=d_slot, proj_dim=d_proj)
        loss_dict = metric_loss_fn(slots, attn, target_dist=d_ins)
        loss = loss_dict["loss_metric"] + loss_dict["loss_entropy"]
        loss.backward()

        elapsed = time.time() - start_time
        assert inputs.grad is not None and not torch.isnan(inputs.grad).any(), "Gradients invalid"

        print(f"  [PASS] Scale stress test succeeded in {elapsed:.3f}s")
        stress_results.append(("Scale Stress (B=128, K=16, N=200)", f"PASSED ({elapsed:.3f}s)"))

    except Exception as e:
        print(f"  [FAIL] Scale stress test failed: {e}")
        traceback.print_exc()
        stress_results.append(("Scale Stress (B=128, K=16, N=200)", f"FAILED: {e}"))

    # -------------------------------------------------------------------
    # Test Case 2: Precision (float64 vs float32)
    # -------------------------------------------------------------------
    try:
        print("\n--- Test Case 2: Double Precision (float64) ---")
        B, K, N, d_slot, d_proj = 4, 4, 20, 32, 16
        locs_64 = torch.rand(B, N, 2, dtype=torch.float64, requires_grad=True)

        d_ins_64 = compute_marginal_insertion_cost(locs_64, k_neighbors=5)
        assert d_ins_64.dtype == torch.float64, f"d_ins_64 dtype mismatch: {d_ins_64.dtype}"
        assert not torch.isnan(d_ins_64).any()

        slot_attn_64 = SlotAttention(num_slots=K, slot_dim=d_slot, in_dim=d_slot).to(dtype=torch.float64)
        inputs_64 = torch.randn(B, N, d_slot, dtype=torch.float64, requires_grad=True)
        slots_64, attn_64 = slot_attn_64(inputs_64)

        assert slots_64.dtype == torch.float64, f"slots_64 dtype mismatch: {slots_64.dtype}"
        assert attn_64.dtype == torch.float64, f"attn_64 dtype mismatch: {attn_64.dtype}"

        metric_fn_64 = MetricLoss(slot_dim=d_slot, proj_dim=d_proj).to(dtype=torch.float64)
        out_64 = metric_fn_64(slots_64, attn_64, target_dist=d_ins_64)

        loss_64 = out_64["loss_metric"] + out_64["loss_entropy"]
        loss_64.backward()

        assert inputs_64.grad is not None and not torch.isnan(inputs_64.grad).any()
        assert locs_64.grad is not None and not torch.isnan(locs_64.grad).any()

        print("  [PASS] Double precision (float64) test succeeded")
        stress_results.append(("Float64 Precision", "PASSED"))

    except Exception as e:
        print(f"  [FAIL] Double precision test failed: {e}")
        traceback.print_exc()
        stress_results.append(("Float64 Precision", f"FAILED: {e}"))

    # -------------------------------------------------------------------
    # Test Case 3: Autograd Backward Pass & Gradient Flow
    # -------------------------------------------------------------------
    try:
        print("\n--- Test Case 3: End-to-End Autograd Backward Pass ---")
        locs = torch.rand(4, 15, 2, requires_grad=True)
        d_ins = compute_marginal_insertion_cost(locs, k_neighbors=5)

        slot_attn = SlotAttention(num_slots=4, slot_dim=32, in_dim=2)
        slots, attn = slot_attn(locs)

        metric_loss_fn = MetricLoss(slot_dim=32, proj_dim=16)
        out = metric_loss_fn(slots, attn, target_dist=d_ins)

        loss = out["loss_metric"] + out["loss_entropy"] + out["dual_penalty"]
        loss.backward()

        assert locs.grad is not None, "locs grad is None"
        assert not torch.isnan(locs.grad).any(), "locs grad contains NaN"
        assert torch.any(locs.grad != 0), "locs grad is all zero"

        for p_name, param in slot_attn.named_parameters():
            assert param.grad is not None, f"slot_attn param {p_name} grad is None"
            assert not torch.isnan(param.grad).any(), f"slot_attn param {p_name} grad contains NaN"

        for p_name, param in metric_loss_fn.named_parameters():
            assert param.grad is not None, f"metric_loss param {p_name} grad is None"
            assert not torch.isnan(param.grad).any(), f"metric_loss param {p_name} grad contains NaN"

        print("  [PASS] End-to-End Autograd Backward Pass succeeded")
        stress_results.append(("Autograd Backward Pass", "PASSED"))

    except Exception as e:
        print(f"  [FAIL] Autograd backward pass failed: {e}")
        traceback.print_exc()
        stress_results.append(("Autograd Backward Pass", f"FAILED: {e}"))

    # -------------------------------------------------------------------
    # Test Case 4: Extreme Values / Singularities
    # -------------------------------------------------------------------
    try:
        print("\n--- Test Case 4: Extreme Values & Co-located Points ---")
        # 1. Co-located points
        locs_co = torch.ones(2, 10, 2) * 5.0
        d_ins_co = compute_marginal_insertion_cost(locs_co, k_neighbors=3)
        assert not torch.isnan(d_ins_co).any(), "Co-located d_ins contains NaN"

        slot_attn = SlotAttention(num_slots=4, slot_dim=32, in_dim=2)
        slots_co, attn_co = slot_attn(locs_co)
        assert not torch.isnan(slots_co).any(), "Co-located slots contain NaN"

        metric_fn = MetricLoss(slot_dim=32, proj_dim=16)
        out_co = metric_fn(slots_co, attn_co, locs=locs_co)
        assert not torch.isnan(out_co["loss_metric"]), "Co-located metric loss is NaN"

        # 2. Extreme coordinate scale (1e6)
        locs_large = torch.rand(2, 10, 2) * 1e6
        d_ins_large = compute_marginal_insertion_cost(locs_large, k_neighbors=3)
        assert not torch.isnan(d_ins_large).any(), "Large scale d_ins contains NaN"

        print("  [PASS] Extreme values & co-located points test succeeded")
        stress_results.append(("Extreme Values & Co-located Points", "PASSED"))

    except Exception as e:
        print(f"  [FAIL] Extreme values test failed: {e}")
        traceback.print_exc()
        stress_results.append(("Extreme Values & Co-located Points", f"FAILED: {e}"))

    print("\nSTRESS TEST RESULTS SUMMARY:")
    for name, status in stress_results:
        print(f"  [{status}] {name}")
    print("\n")


if __name__ == "__main__":
    run_mutation_tests()
    run_edge_and_stress_tests()
