import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

sys.path.insert(0, os.path.abspath("."))

from rl4co.data.insertion_cost import (
    compute_pairwise_distance_matrix,
    compute_marginal_insertion_cost,
)
from tests.test_slot_attention import SlotAttention
from tests.test_metric_loss import MetricLoss


def run_mutation_tests():
    print("=== STARTING MUTATION TESTING ===")
    results = []

    # -------------------------------------------------------------------
    # Mutation 1: Wrong d_ins formula (+ dist(D, j) instead of - dist(D, j))
    # -------------------------------------------------------------------
    print("\n--- Testing Mutation 1: Wrong d_ins formula ---")
    def buggy_compute_marginal_insertion_cost_1(locs, k_neighbors=15, depot_loc=None):
        if locs.dim() == 2:
            locs = locs.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        B, N, _ = locs.shape
        device = locs.device

        if depot_loc is None:
            depot_loc = torch.full((B, 1, 2), 0.5, device=device, dtype=locs.dtype)
        elif depot_loc.dim() == 1:
            depot_loc = depot_loc.unsqueeze(0).unsqueeze(0)
        elif depot_loc.dim() == 2:
            depot_loc = depot_loc.unsqueeze(1)

        dist_customers = compute_pairwise_distance_matrix(locs)
        dist_depot = torch.norm(locs - depot_loc, p=2, dim=-1)

        # BUGGY FORMULA: + dist_depot.unsqueeze(1) instead of - dist_depot.unsqueeze(1)
        d_ins = dist_depot.unsqueeze(2) + dist_customers + dist_depot.unsqueeze(1)

        d_ins = torch.clamp(d_ins, min=0.0)
        eye_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        d_ins = d_ins.masked_fill(eye_mask, 0.0)

        if k_neighbors is not None and k_neighbors < N:
            _, knn_indices = torch.topk(dist_customers, k=k_neighbors + 1, dim=-1, largest=False)
            knn_mask = torch.zeros((B, N, N), dtype=torch.bool, device=device)
            knn_mask.scatter_(2, knn_indices, True)
            d_ins = d_ins.masked_fill(~knn_mask, float("inf"))

        if squeeze_batch:
            d_ins = d_ins.squeeze(0)
        return d_ins

    # Verify that test_marginal_insertion_cost_basic FAILS under Mutation 1
    try:
        torch.manual_seed(42)
        B, N = 2, 5
        locs = torch.rand(B, N, 2)
        depot = torch.full((B, 1, 2), 0.5)
        d_ins = buggy_compute_marginal_insertion_cost_1(locs, k_neighbors=None, depot_loc=depot)
        # Test basic assertion
        dist_d_i = torch.norm(locs[0, 0] - depot[0, 0])
        dist_i_j = torch.norm(locs[0, 0] - locs[0, 1])
        dist_d_j = torch.norm(locs[0, 1] - depot[0, 0])
        expected = dist_d_i + dist_i_j - dist_d_j
        assert torch.isclose(d_ins[0, 0, 1], expected, atol=1e-5)
        print("FAIL: Mutation 1 was NOT caught by unit test!")
        results.append(("Mutation 1 (Wrong d_ins formula)", False, "Test did not detect bug"))
    except AssertionError as e:
        print(f"SUCCESS: Mutation 1 caught by unit test: {e}")
        results.append(("Mutation 1 (Wrong d_ins formula)", True, "Caught by formula assertion"))

    # -------------------------------------------------------------------
    # Mutation 2: Missing inf mask in k-NN sparsification
    # -------------------------------------------------------------------
    print("\n--- Testing Mutation 2: Missing inf mask in k-NN sparsification ---")
    def buggy_compute_marginal_insertion_cost_2(locs, k_neighbors=15, depot_loc=None):
        # BUG: ignores k_neighbors sparsification completely
        return compute_marginal_insertion_cost(locs, k_neighbors=None, depot_loc=depot_loc)

    try:
        locs_n20 = torch.rand(2, 20, 2)
        d_ins_k15 = buggy_compute_marginal_insertion_cost_2(locs_n20, k_neighbors=15)
        non_inf_counts = torch.sum(~torch.isinf(d_ins_k15[0]), dim=-1)
        assert torch.all(non_inf_counts == 16), f"Expected 16 non-inf entries, got {non_inf_counts}"
        print("FAIL: Mutation 2 was NOT caught by unit test!")
        results.append(("Mutation 2 (Missing k-NN inf mask)", False, "Test did not detect missing inf mask"))
    except AssertionError as e:
        print(f"SUCCESS: Mutation 2 caught by unit test: {e}")
        results.append(("Mutation 2 (Missing k-NN inf mask)", True, "Caught by non_inf_counts assertion"))

    # -------------------------------------------------------------------
    # Mutation 3: Unnormalized A_ik (Sigmoid instead of Softmax over K)
    # -------------------------------------------------------------------
    print("\n--- Testing Mutation 3: Unnormalized A_ik ---")
    class BuggySlotAttention(SlotAttention):
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
                # BUG: Sigmoid instead of Softmax
                attn = torch.sigmoid(dots)
                attn_norm = attn / (attn.sum(dim=1, keepdim=True) + self.eps)
                updates = torch.einsum("b n k, b n d -> b k d", attn_norm, v)
                slots = self.gru(
                    updates.reshape(-1, self.slot_dim),
                    slots_prev.reshape(-1, self.slot_dim)
                ).reshape(B, K, self.slot_dim)
                slots = slots + self.mlp(self.norm_pre_ff(slots))
            return slots, attn

    try:
        B, N, K, d = 8, 50, 6, 64
        model = BuggySlotAttention(num_slots=K, slot_dim=d, in_dim=d, num_iterations=3)
        inputs = torch.randn(B, N, d) * 10.0
        slots, attn = model(inputs)
        attn_sum = attn.sum(dim=-1)
        expected_ones = torch.ones(B, N, device=inputs.device)
        assert torch.allclose(attn_sum, expected_ones, atol=1e-5)
        print("FAIL: Mutation 3 was NOT caught by unit test!")
        results.append(("Mutation 3 (Unnormalized A_ik)", False, "Test did not detect unnormalized attn"))
    except AssertionError as e:
        print(f"SUCCESS: Mutation 3 caught by unit test: {e}")
        results.append(("Mutation 3 (Unnormalized A_ik)", True, "Caught by softmax sum=1 assertion"))

    # -------------------------------------------------------------------
    # Mutation 4: Broken dual ascent update (Subtract violation instead of add)
    # -------------------------------------------------------------------
    print("\n--- Testing Mutation 4: Broken dual ascent update ---")
    class BuggyMetricLoss4(MetricLoss):
        def update_lambda(self, violation: float, lr: float = None):
            lr = lr if lr is not None else self.lambda_lr
            with torch.no_grad():
                # BUG: subtract violation when constraint is violated
                updated = self.log_lambda.item() - lr * violation
                self.log_lambda.copy_(torch.tensor(updated).clamp(-10.0, 10.0))

    try:
        metric_fn = BuggyMetricLoss4(slot_dim=32, proj_dim=16, init_log_lambda=0.0)
        init_val = metric_fn.get_lambda()
        metric_fn.update_lambda(violation=2.0, lr=0.1)
        assert metric_fn.get_lambda() > init_val
        print("FAIL: Mutation 4 was NOT caught by unit test!")
        results.append(("Mutation 4 (Broken dual ascent)", False, "Test did not detect inverted dual update"))
    except AssertionError as e:
        print(f"SUCCESS: Mutation 4 caught by unit test: {e}")
        results.append(("Mutation 4 (Broken dual ascent)", True, "Caught by lambda increase assertion"))

    # -------------------------------------------------------------------
    # Mutation 5: Missing inf masking in MetricLoss target distance
    # -------------------------------------------------------------------
    print("\n--- Testing Mutation 5: Missing inf masking in MetricLoss target distance ---")
    class BuggyMetricLoss5(MetricLoss):
        def compute_target_dist_insertion(self, d_ins: torch.Tensor, attn: torch.Tensor) -> torch.Tensor:
            # BUG: ignore inf masking and use raw d_ins directly
            w_node = torch.einsum("b n k, b m l -> b k l n m", attn, attn)
            num = (w_node * d_ins.unsqueeze(1).unsqueeze(1)).sum(dim=(-2, -1))
            den = w_node.sum(dim=(-2, -1)) + self.eps
            return num / den

    try:
        B, N, K, d_slot = 2, 10, 4, 32
        metric_fn = BuggyMetricLoss5(slot_dim=d_slot, proj_dim=16)
        slots = torch.randn(B, K, d_slot)
        attn = F.softmax(torch.randn(B, N, K), dim=-1)
        d_ins_sparsified = torch.rand(B, N, N)
        d_ins_sparsified[:, 0, 5] = float("inf")
        d_ins_sparsified[:, 5, 0] = float("inf")
        
        out = metric_fn(slots, attn, target_dist=d_ins_sparsified)
        loss_metric = out["loss_metric"]
        assert not torch.isnan(loss_metric) and not torch.isinf(loss_metric)
        print("FAIL: Mutation 5 was NOT caught by unit test!")
        results.append(("Mutation 5 (Missing inf mask in MetricLoss)", False, "Test did not detect NaN/Inf loss"))
    except AssertionError as e:
        print(f"SUCCESS: Mutation 5 caught by unit test: {e}")
        results.append(("Mutation 5 (Missing inf mask in MetricLoss)", True, "Caught by NaN/Inf assertion"))

    # Print summary
    print("\n=== MUTATION TESTING SUMMARY ===")
    all_passed = True
    for name, passed, detail in results:
        status = "PASSED" if passed else "FAILED"
        print(f"[{status}] {name}: {detail}")
        if not passed:
            all_passed = False

    return all_passed


if __name__ == "__main__":
    success = run_mutation_tests()
    if not success:
        sys.exit(1)
