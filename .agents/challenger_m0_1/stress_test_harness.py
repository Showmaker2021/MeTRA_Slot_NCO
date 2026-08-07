import time
import torch
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.insert(0, r"d:\NCO NEW\rl4co")

from rl4co.data.insertion_cost import (
    compute_pairwise_distance_matrix,
    compute_marginal_insertion_cost,
)

def log_test(name, passed, details=""):
    status = "PASSED" if passed else "FAILED"
    print(f"[{status}] {name} {details}")
    return passed

def run_all_stress_tests():
    print("==================================================================")
    print("STARTING EMPIRICAL STRESS TEST HARNESS FOR MILESTONE M0")
    print("==================================================================")
    
    results = []

    # ---------------------------------------------------------
    # Test 1: Large Scale Benchmark (N=500, B=512)
    # ---------------------------------------------------------
    print("\n--- Test 1: Large Scale Stress & Performance (N=500, B=512) ---")
    try:
        B, N = 512, 500
        locs = torch.rand(B, N, 2, dtype=torch.float32)
        start_time = time.time()
        
        # Test pairwise distance
        dist_mat = compute_pairwise_distance_matrix(locs)
        dist_time = time.time() - start_time
        
        # Test marginal insertion cost with k=15
        start_ins = time.time()
        d_ins = compute_marginal_insertion_cost(locs, k_neighbors=15)
        ins_time = time.time() - start_ins
        
        total_time = time.time() - start_time
        
        # Shape check
        shape_ok = d_ins.shape == (B, N, N) and dist_mat.shape == (B, N, N)
        # Non-negativity check
        non_neg_ok = torch.all(d_ins[~torch.isinf(d_ins)] >= 0.0).item()
        # Non-inf count check: k=15 means 16 finite entries per row
        non_inf_counts = torch.sum(~torch.isinf(d_ins), dim=-1)
        k_count_ok = torch.all(non_inf_counts == 16).item()
        # Diagonal zero check
        diag_indices = torch.arange(N)
        diags = d_ins[:, diag_indices, diag_indices]
        diag_ok = torch.all(diags == 0.0).item()
        
        passed = shape_ok and non_neg_ok and k_count_ok and diag_ok
        details = (
            f"(Shape: {d_ins.shape}, Dist Time: {dist_time:.3f}s, "
            f"Ins Cost Time: {ins_time:.3f}s, Total: {total_time:.3f}s)"
        )
        results.append(log_test("Large Scale N=500, B=512", passed, details))
    except Exception as e:
        results.append(log_test("Large Scale N=500, B=512", False, f"Exception: {e}"))

    # ---------------------------------------------------------
    # Test 2: Collinear Coordinates Stress Test
    # ---------------------------------------------------------
    print("\n--- Test 2: Collinear Coordinates ---")
    try:
        B, N = 4, 50
        # Create collinear points along y = 2x
        t = torch.linspace(0, 10, N)
        collinear_locs = torch.stack([t, 2.0 * t], dim=-1).unsqueeze(0).repeat(B, 1, 1) # (B, N, 2)
        depot_collinear = torch.tensor([[0.0, 0.0]]).repeat(B, 1, 1)
        
        d_ins_collinear = compute_marginal_insertion_cost(collinear_locs, k_neighbors=None, depot_loc=depot_collinear)
        
        no_nan = not torch.isnan(d_ins_collinear).any().item()
        non_neg = torch.all(d_ins_collinear >= 0.0).item()
        
        # Verification: for collinear points D=(0,0), P_i=(x_i, y_i), P_j=(x_j, y_j) where 0 < x_i < x_j
        # dist(D, P_i) + dist(P_i, P_j) - dist(D, P_j) should be EXACTLY 0.0
        # i=5 (t=1.02), j=10 (t=2.04) -> P_i is between D and P_j
        exact_zero_val = d_ins_collinear[0, 5, 10].item()
        zero_exact_ok = abs(exact_zero_val) < 1e-5
        
        passed = no_nan and non_neg and zero_exact_ok
        details = f"(d_ins[0, 5, 10] = {exact_zero_val:.6f}, No NaN: {no_nan}, Non-neg: {non_neg})"
        results.append(log_test("Collinear Coordinates", passed, details))
    except Exception as e:
        results.append(log_test("Collinear Coordinates", False, f"Exception: {e}"))

    # ---------------------------------------------------------
    # Test 3: Co-located Nodes & Customer at Depot
    # ---------------------------------------------------------
    print("\n--- Test 3: Co-located Nodes & Customer at Depot ---")
    try:
        B, N = 2, 6
        locs = torch.tensor([
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.1, 0.1], [0.1, 0.1], [0.5, 0.5], [0.5, 0.5]]
        ])
        depot = torch.tensor([[[0.5, 0.5]], [[0.0, 0.0]]]) # Batch 0 depot at (0.5,0.5), Batch 1 depot at (0,0)
        
        d_ins = compute_marginal_insertion_cost(locs, k_neighbors=3, depot_loc=depot)
        
        no_nan = not torch.isnan(d_ins).any().item()
        # Co-located node 0 and 1 in batch 0
        co_located_ok = abs(d_ins[0, 0, 1].item()) < 1e-5 or torch.isinf(d_ins[0, 0, 1]).item()
        # Customer at depot (node 0 in batch 0 is at depot 0.5, 0.5)
        # d_ins(0, 3) = dist(D, 0) + dist(0, 3) - dist(D, 3) = 0 + dist(0, 3) - dist(D, 3) = 0
        depot_customer_ok = abs(d_ins[0, 0, 3].item()) < 1e-5 or torch.isinf(d_ins[0, 0, 3]).item()
        
        passed = no_nan and co_located_ok and depot_customer_ok
        details = f"(No NaN: {no_nan}, Co-located OK: {co_located_ok}, Customer at Depot OK: {depot_customer_ok})"
        results.append(log_test("Co-located & Customer at Depot", passed, details))
    except Exception as e:
        results.append(log_test("Co-located & Customer at Depot", False, f"Exception: {e}"))

    # ---------------------------------------------------------
    # Test 4: Zero Depot Stress Test
    # ---------------------------------------------------------
    print("\n--- Test 4: Zero Depot ---")
    try:
        B, N = 4, 20
        locs = torch.rand(B, N, 2)
        zero_depot_1d = torch.zeros(2)
        zero_depot_2d = torch.zeros(1, 2)
        zero_depot_3d = torch.zeros(B, 1, 2)
        
        d1 = compute_marginal_insertion_cost(locs, k_neighbors=5, depot_loc=zero_depot_1d)
        d2 = compute_marginal_insertion_cost(locs, k_neighbors=5, depot_loc=zero_depot_2d)
        d3 = compute_marginal_insertion_cost(locs, k_neighbors=5, depot_loc=zero_depot_3d)
        
        # Verify 1D, 2D, 3D zero depot produce identical outputs
        eq_1_2 = torch.allclose(d1.nan_to_num(posinf=1e9), d2.nan_to_num(posinf=1e9))
        eq_2_3 = torch.allclose(d2.nan_to_num(posinf=1e9), d3.nan_to_num(posinf=1e9))
        
        passed = eq_1_2 and eq_2_3
        details = f"(1D vs 2D equal: {eq_1_2}, 2D vs 3D equal: {eq_2_3})"
        results.append(log_test("Zero Depot Handling", passed, details))
    except Exception as e:
        results.append(log_test("Zero Depot Handling", False, f"Exception: {e}"))

    # ---------------------------------------------------------
    # Test 5: Random Seeds Stability (Multi-seed)
    # ---------------------------------------------------------
    print("\n--- Test 5: Multi-seed Determinism & Stability ---")
    try:
        seeds = [0, 42, 1234, 9999, 2026]
        all_passed = True
        for s in seeds:
            torch.manual_seed(s)
            locs = torch.rand(4, 30, 2)
            d_ins = compute_marginal_insertion_cost(locs, k_neighbors=10)
            if torch.isnan(d_ins).any().item():
                all_passed = False
                break
            if torch.any(d_ins[~torch.isinf(d_ins)] < 0.0).item():
                all_passed = False
                break
        
        results.append(log_test("Multi-seed Stability (5 seeds)", all_passed))
    except Exception as e:
        results.append(log_test("Multi-seed Stability", False, f"Exception: {e}"))

    # ---------------------------------------------------------
    # Test 6: Variations of k_neighbors (k=1, k=N, k>N, k=None)
    # ---------------------------------------------------------
    print("\n--- Test 6: Variations of k_neighbors (k=1, k=N, k>N, k=None) ---")
    try:
        B, N = 3, 10
        locs = torch.rand(B, N, 2)
        
        # k = 1 (1 nearest neighbor + self = 2 non-inf per row)
        d_k1 = compute_marginal_insertion_cost(locs, k_neighbors=1)
        count_k1 = torch.sum(~torch.isinf(d_k1), dim=-1)
        k1_ok = torch.all(count_k1 == 2).item()
        
        # k = N (k=10, N=10 -> N <= k -> dense matrix, 0 inf entries)
        d_kN = compute_marginal_insertion_cost(locs, k_neighbors=N)
        kN_ok = not torch.any(torch.isinf(d_kN)).item()
        
        # k > N (k=15, N=10 -> N <= k -> dense matrix, 0 inf entries)
        d_kgtN = compute_marginal_insertion_cost(locs, k_neighbors=15)
        kgtN_ok = not torch.any(torch.isinf(d_kgtN)).item()
        
        # k = None (dense matrix, 0 inf entries)
        d_kNone = compute_marginal_insertion_cost(locs, k_neighbors=None)
        kNone_ok = not torch.any(torch.isinf(d_kNone)).item()
        
        # Compare kN, kgtN, kNone (should be identical tensors)
        dense_match_1 = torch.allclose(d_kN, d_kNone)
        dense_match_2 = torch.allclose(d_kgtN, d_kNone)
        
        passed = k1_ok and kN_ok and kgtN_ok and kNone_ok and dense_match_1 and dense_match_2
        details = f"(k=1 count=2: {k1_ok}, k=N dense: {kN_ok}, k>N dense: {kgtN_ok}, k=None dense: {kNone_ok}, dense values match: {dense_match_1 and dense_match_2})"
        results.append(log_test("k_neighbors Variations (k=1, k=N, k>N, k=None)", passed, details))
    except Exception as e:
        results.append(log_test("k_neighbors Variations", False, f"Exception: {e}"))

    # ---------------------------------------------------------
    # Test 7: Mathematical Equivalence vs Reference Loop
    # ---------------------------------------------------------
    print("\n--- Test 7: Mathematical Equivalence vs Double-Loop Reference ---")
    try:
        torch.manual_seed(100)
        B, N = 4, 15
        locs = torch.rand(B, N, 2)
        depot = torch.tensor([[0.3, 0.7]]).repeat(B, 1, 1)
        
        d_ins_vec = compute_marginal_insertion_cost(locs, k_neighbors=None, depot_loc=depot)
        
        # Reference double loop computation
        d_ins_ref = torch.zeros(B, N, N)
        for b in range(B):
            d_loc = depot[b, 0]
            for i in range(N):
                for j in range(N):
                    if i == j:
                        d_ins_ref[b, i, j] = 0.0
                    else:
                        c_D_i = torch.norm(locs[b, i] - d_loc, p=2)
                        c_i_j = torch.norm(locs[b, i] - locs[b, j], p=2)
                        c_D_j = torch.norm(locs[b, j] - d_loc, p=2)
                        val = c_D_i + c_i_j - c_D_j
                        d_ins_ref[b, i, j] = max(0.0, float(val))
        
        max_diff = torch.max(torch.abs(d_ins_vec - d_ins_ref)).item()
        equiv_ok = max_diff < 1e-6
        
        results.append(log_test("Double-Loop Mathematical Equivalence", equiv_ok, f"(Max abs diff: {max_diff:.8f})"))
    except Exception as e:
        results.append(log_test("Double-Loop Mathematical Equivalence", False, f"Exception: {e}"))

    # ---------------------------------------------------------
    # Test 8: Dtype Stability (float32 vs float64)
    # ---------------------------------------------------------
    print("\n--- Test 8: Dtype Support (float32 & float64) ---")
    try:
        locs_f64 = torch.rand(2, 10, 2, dtype=torch.float64)
        depot_f64 = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        d_ins_f64 = compute_marginal_insertion_cost(locs_f64, k_neighbors=4, depot_loc=depot_f64)
        
        f64_ok = (d_ins_f64.dtype == torch.float64) and not torch.isnan(d_ins_f64).any()
        results.append(log_test("Float64 Precision Support", f64_ok, f"(Output dtype: {d_ins_f64.dtype})"))
    except Exception as e:
        results.append(log_test("Float64 Precision Support", False, f"Exception: {e}"))

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("\n==================================================================")
    all_pass = all(results)
    print(f"STRESS TEST SUMMARY: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print(f"Passed: {sum(results)} / {len(results)}")
    print("==================================================================")
    
    return all_pass

if __name__ == "__main__":
    success = run_all_stress_tests()
    sys.exit(0 if success else 1)
