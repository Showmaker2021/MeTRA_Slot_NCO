import torch
import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.abspath("."))

from rl4co.data.insertion_cost import compute_pairwise_distance_matrix, compute_marginal_insertion_cost

def test_jit_trace_comprehensive():
    print("=== Testing torch.jit.trace comprehensively ===")
    
    # 1. Trace compute_pairwise_distance_matrix with 3D tensor
    sample_3d = torch.randn(2, 10, 2)
    traced_dist = torch.jit.trace(compute_pairwise_distance_matrix, (sample_3d,))
    res_eager = compute_pairwise_distance_matrix(sample_3d)
    res_traced = traced_dist(sample_3d)
    assert torch.allclose(res_eager, res_traced), "JIT trace pairwise distance 3D mismatch"
    assert res_traced.dtype == sample_3d.dtype, "JIT trace pairwise distance dtype mismatch"
    print("[PASS] JIT trace compute_pairwise_distance_matrix 3D")

    # 2. Trace compute_pairwise_distance_matrix with 2D tensor
    sample_2d = torch.randn(10, 2)
    traced_dist_2d = torch.jit.trace(compute_pairwise_distance_matrix, (sample_2d,))
    res_eager_2d = compute_pairwise_distance_matrix(sample_2d)
    res_traced_2d = traced_dist_2d(sample_2d)
    assert torch.allclose(res_eager_2d, res_traced_2d), "JIT trace pairwise distance 2D mismatch"
    print("[PASS] JIT trace compute_pairwise_distance_matrix 2D")

    # 3. Wrapper class for tracing compute_marginal_insertion_cost with fixed k_neighbors
    class InsertionCostWrapper(torch.nn.Module):
        def __init__(self, k_neighbors=5):
            super().__init__()
            self.k_neighbors = k_neighbors
        def forward(self, locs, depot_loc=None):
            return compute_marginal_insertion_cost(locs, k_neighbors=self.k_neighbors, depot_loc=depot_loc)

    wrapper = InsertionCostWrapper(k_neighbors=5)
    locs = torch.rand(4, 15, 2)
    depot = torch.rand(4, 1, 2)
    
    traced_cost = torch.jit.trace(wrapper, (locs, depot))
    res_eager_cost = wrapper(locs, depot)
    res_traced_cost = traced_cost(locs, depot)
    
    # Compare finite values
    finite_mask = torch.isfinite(res_eager_cost)
    assert torch.equal(finite_mask, torch.isfinite(res_traced_cost)), "Mask of inf mismatch in traced cost"
    assert torch.allclose(res_eager_cost[finite_mask], res_traced_cost[finite_mask]), "Traced cost values mismatch"
    print("[PASS] JIT trace wrapper compute_marginal_insertion_cost (3D locs + 3D depot)")

    # 4. Tracing with float16 & bfloat16
    for dtype in [torch.float16, torch.bfloat16]:
        locs_h = locs.to(dtype)
        depot_h = depot.to(dtype)
        traced_h = torch.jit.trace(wrapper, (locs_h, depot_h))
        out_traced = traced_h(locs_h, depot_h)
        out_eager = wrapper(locs_h, depot_h)
        assert out_traced.dtype == dtype, f"Traced output dtype mismatch for {dtype}"
        print(f"[PASS] JIT trace compute_marginal_insertion_cost for {dtype}")

def test_half_precision_comprehensive():
    print("\n=== Testing Half Precision (float16 / bfloat16) comprehensively ===")
    
    dtypes = [torch.float16, torch.bfloat16]
    
    for dtype in dtypes:
        # Test 1: Basic pairwise distance
        coords = torch.rand(3, 20, 2, dtype=dtype)
        dist = compute_pairwise_distance_matrix(coords)
        assert dist.dtype == dtype, f"Pairwise dist dtype failed for {dtype}"
        assert not torch.isnan(dist).any(), f"NaN in pairwise dist for {dtype}"
        assert (dist >= 0).all(), f"Negative dist for {dtype}"
        print(f"[PASS] compute_pairwise_distance_matrix for {dtype}")

        # Test 2: compute_marginal_insertion_cost (default depot)
        d_ins = compute_marginal_insertion_cost(coords, k_neighbors=5)
        assert d_ins.dtype == dtype, f"Insertion cost dtype failed for {dtype}"
        assert not torch.isnan(d_ins).any(), f"NaN in insertion cost for {dtype}"
        # Diagonal elements must be 0
        diag_val = torch.diagonal(d_ins, dim1=1, dim2=2)
        assert torch.all(diag_val == 0), f"Diagonal non-zero for {dtype}"
        print(f"[PASS] compute_marginal_insertion_cost (default depot) for {dtype}")

        # Test 3: custom depot (1D, 2D, 3D)
        depot_1d = torch.tensor([0.2, 0.8], dtype=dtype)
        d_ins_1d = compute_marginal_insertion_cost(coords, k_neighbors=5, depot_loc=depot_1d)
        assert d_ins_1d.dtype == dtype, f"1D depot insertion cost dtype failed for {dtype}"

        depot_2d = torch.tensor([[0.2, 0.8]], dtype=dtype)
        d_ins_2d = compute_marginal_insertion_cost(coords, k_neighbors=5, depot_loc=depot_2d)
        assert d_ins_2d.dtype == dtype, f"2D depot insertion cost dtype failed for {dtype}"

        depot_3d = torch.rand(3, 1, 2, dtype=dtype)
        d_ins_3d = compute_marginal_insertion_cost(coords, k_neighbors=5, depot_loc=depot_3d)
        assert d_ins_3d.dtype == dtype, f"3D depot insertion cost dtype failed for {dtype}"
        print(f"[PASS] compute_marginal_insertion_cost (custom depots 1D/2D/3D) for {dtype}")

        # Test 4: Gradient backward pass in float16/bfloat16
        locs_grad = torch.rand(2, 8, 2, dtype=dtype, requires_grad=True)
        d_ins_g = compute_marginal_insertion_cost(locs_grad, k_neighbors=3)
        loss = torch.where(torch.isfinite(d_ins_g), d_ins_g, torch.zeros_like(d_ins_g)).sum()
        loss.backward()
        assert locs_grad.grad is not None, f"Grad is None for {dtype}"
        assert not torch.isnan(locs_grad.grad).any(), f"Grad contains NaN for {dtype}"
        print(f"[PASS] Autograd backward pass for {dtype}")

if __name__ == "__main__":
    test_jit_trace_comprehensive()
    test_half_precision_comprehensive()
    print("\n>>> ALL STRESS TESTS PASSED SUCCESSFULLY! <<<")
