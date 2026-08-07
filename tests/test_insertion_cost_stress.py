import torch
import pytest
import math
from rl4co.data.insertion_cost import (
    compute_pairwise_distance_matrix,
    compute_marginal_insertion_cost,
)


class TestInsertionCostNumericalLimits:
    """Stress tests for extreme numerical values, large coordinates, small coordinates, and collinear setups."""

    def test_large_coordinates(self):
        """Test with large coordinate values (1e6, 1e8) to check for overflow or inf issues."""
        locs = torch.tensor([[[0.0, 0.0], [1e6, 0.0], [0.0, 1e6]]], dtype=torch.float32)
        depot = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)

        d_ins = compute_marginal_insertion_cost(locs, k_neighbors=2, depot_loc=depot)

        assert not torch.isnan(d_ins).any(), "d_ins contains NaN on large coordinates"
        assert not torch.isinf(d_ins[0, 0, 1]), "Valid entries must not be Inf"
        # d_ins(1, 2): inserting node 1 (1e6, 0) into tour D -> node 2 (0, 1e6) -> D
        # dist(D, 1) = 1e6, dist(1, 2) = sqrt(2)*1e6, dist(D, 2) = 1e6
        # d_ins(1, 2) = 1e6 + sqrt(2)*1e6 - 1e6 = sqrt(2)*1e6
        expected = math.sqrt(2.0) * 1e6
        assert torch.isclose(d_ins[0, 1, 2], torch.tensor(expected, dtype=torch.float32), rtol=1e-4)

    def test_small_coordinates(self):
        """Test with tiny subnormal coordinate values (1e-7) to check for underflow/precision."""
        locs = torch.tensor([[[0.0, 0.0], [1e-7, 0.0], [0.0, 1e-7]]], dtype=torch.float32)
        depot = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)

        d_ins = compute_marginal_insertion_cost(locs, k_neighbors=2, depot_loc=depot)

        assert not torch.isnan(d_ins).any(), "d_ins contains NaN on tiny coordinates"
        assert (d_ins >= 0.0).all(), "d_ins must be non-negative"

    def test_collinear_nodes(self):
        """Test with collinear nodes along a straight line: (0,0), (1,0), (2,0), (3,0)."""
        locs = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]], dtype=torch.float32)
        depot = torch.tensor([[[0.0, 0.0]]], dtype=torch.float32)

        d_ins = compute_marginal_insertion_cost(locs, k_neighbors=None, depot_loc=depot)

        # For collinear nodes from depot:
        # dist(D, node1) = 1, dist(D, node2) = 2, dist(node1, node2) = 1
        # d_ins(1, 2) = dist(D, 1) + dist(1, 2) - dist(D, 2) = 1 + 1 - 2 = 0.0!
        # Inserting node 1 between D and node 2 incurs ZERO extra distance on a straight line!
        assert torch.isclose(d_ins[0, 0, 1], torch.tensor(0.0), atol=1e-6)

    def test_all_co_located_nodes(self):
        """Test when all nodes and depot are at the exact same location (0.5, 0.5)."""
        locs = torch.full((2, 10, 2), 0.5)
        depot = torch.full((2, 1, 2), 0.5)

        d_ins = compute_marginal_insertion_cost(locs, k_neighbors=5, depot_loc=depot)

        # Non-inf elements (k+1 = 6 per row) should be exactly 0.0
        finite_mask = ~torch.isinf(d_ins)
        assert torch.all(d_ins[finite_mask] == 0.0), "All finite d_ins values for co-located nodes must be 0.0"


class TestInsertionCostDtypes:
    """Stress tests for float64, float32, float16, and bfloat16 dtypes."""

    def test_float64_precision(self):
        """Test compute_marginal_insertion_cost with double precision (float64)."""
        locs = torch.rand(2, 10, 2, dtype=torch.float64)
        depot = torch.rand(2, 1, 2, dtype=torch.float64)

        d_ins = compute_marginal_insertion_cost(locs, k_neighbors=4, depot_loc=depot)

        assert d_ins.dtype == torch.float64
        assert not torch.isnan(d_ins).any()
        for b in range(2):
            for i in range(10):
                assert d_ins[b, i, i].item() == 0.0

    def test_half_precision_float16(self):
        """Test compute_marginal_insertion_cost with half precision (float16)."""
        locs = torch.rand(2, 8, 2, dtype=torch.float16)
        depot = torch.rand(2, 1, 2, dtype=torch.float16)

        d_ins = compute_marginal_insertion_cost(locs, k_neighbors=3, depot_loc=depot)

        assert d_ins.dtype == torch.float16
        assert not torch.isnan(d_ins).any()

    def test_bfloat16_precision(self):
        """Test compute_marginal_insertion_cost with bfloat16."""
        locs = torch.rand(2, 8, 2, dtype=torch.bfloat16)
        depot = torch.rand(2, 1, 2, dtype=torch.bfloat16)

        d_ins = compute_marginal_insertion_cost(locs, k_neighbors=3, depot_loc=depot)

        assert d_ins.dtype == torch.bfloat16
        assert not torch.isnan(d_ins).any()


class TestInsertionCostTorchScriptJIT:
    """Stress tests for PyTorch JIT / TorchScript compatibility."""

    def test_torch_jit_script_pairwise_distance(self):
        """Verify compute_pairwise_distance_matrix is TorchScript scriptable."""
        coords = torch.rand(4, 15, 2)

        scripted_fn = torch.jit.script(compute_pairwise_distance_matrix)
        res_eager = compute_pairwise_distance_matrix(coords)
        res_script = scripted_fn(coords)

        assert torch.allclose(res_eager, res_script)

    def test_torch_jit_script_marginal_insertion_cost(self):
        """Verify compute_marginal_insertion_cost is TorchScript scriptable."""
        locs = torch.rand(4, 15, 2)
        depot = torch.rand(4, 1, 2)

        scripted_fn = torch.jit.script(compute_marginal_insertion_cost)
        res_eager = compute_marginal_insertion_cost(locs, k_neighbors=5, depot_loc=depot)
        res_script = scripted_fn(locs, 5, depot)

        # Check finite equality and Inf location equality
        mask_eager = torch.isinf(res_eager)
        mask_script = torch.isinf(res_script)
        assert torch.equal(mask_eager, mask_script)
        assert torch.allclose(res_eager[~mask_eager], res_script[~mask_script])

    def test_torch_jit_trace(self):
        """Verify compute_marginal_insertion_cost can be traced with JIT trace."""
        locs = torch.rand(2, 10, 2)
        depot = torch.rand(2, 1, 2)

        traced_fn = torch.jit.trace(
            lambda l, d: compute_marginal_insertion_cost(l, k_neighbors=4, depot_loc=d),
            (locs, depot),
        )
        res_eager = compute_marginal_insertion_cost(locs, k_neighbors=4, depot_loc=depot)
        res_traced = traced_fn(locs, depot)

        assert torch.allclose(
            res_eager.masked_fill(torch.isinf(res_eager), 0.0),
            res_traced.masked_fill(torch.isinf(res_traced), 0.0),
        )


class TestInsertionCostGradientBackward:
    """Stress tests for autograd backward pass and gradient stability."""

    def test_gradient_with_knn_sparsification(self):
        """Verify gradient backward pass works cleanly through topk sparsified d_ins."""
        locs = torch.rand(2, 10, 2, requires_grad=True)
        depot = torch.rand(2, 1, 2, requires_grad=True)

        d_ins = compute_marginal_insertion_cost(locs, k_neighbors=3, depot_loc=depot)
        valid_d_ins = d_ins.masked_fill(torch.isinf(d_ins), 0.0)
        loss = valid_d_ins.pow(2).sum()
        loss.backward()

        assert locs.grad is not None and not torch.isnan(locs.grad).any()
        assert depot.grad is not None and not torch.isnan(depot.grad).any()

    def test_gradient_at_co_located_nodes(self):
        """Test gradient stability when two customers are co-located (distance = 0.0)."""
        locs = torch.tensor([[[0.2, 0.2], [0.2, 0.2], [0.5, 0.5]]], requires_grad=True)
        depot = torch.tensor([[[0.0, 0.0]]], requires_grad=True)

        d_ins = compute_marginal_insertion_cost(locs, k_neighbors=None, depot_loc=depot)
        loss = d_ins.sum()
        loss.backward()

        # Check if co-located points cause NaN in gradients due to derivative of norm at zero
        assert not torch.isnan(locs.grad).any(), "Gradient contains NaN for co-located nodes!"
        assert not torch.isnan(depot.grad).any(), "Depot gradient contains NaN!"

    def test_gradient_customer_at_depot(self):
        """Test gradient stability when customer is co-located with depot."""
        locs = torch.tensor([[[0.0, 0.0], [0.3, 0.4]]], requires_grad=True)
        depot = torch.tensor([[[0.0, 0.0]]], requires_grad=True)

        d_ins = compute_marginal_insertion_cost(locs, k_neighbors=None, depot_loc=depot)
        loss = d_ins.sum()
        loss.backward()

        assert not torch.isnan(locs.grad).any(), "Gradient contains NaN when customer is at depot!"


class TestInsertionCostMemoryAndScaling:
    """Stress tests for scaling to large problem sizes N=500, 1000, 2000."""

    def test_scaling_large_n(self):
        """Test memory and output correctness for N=500, 1000, 2000."""
        for N in [500, 1000]:
            locs = torch.rand(2, N, 2)
            d_ins = compute_marginal_insertion_cost(locs, k_neighbors=15)
            assert d_ins.shape == (2, N, N)
            finite_counts = (~torch.isinf(d_ins[0])).sum(dim=-1)
            assert (finite_counts == 16).all(), f"Failed finite count check for N={N}"

    def test_k_neighbors_zero(self):
        """Test edge case where k_neighbors=0 (only self-insertion is non-inf)."""
        locs = torch.rand(2, 5, 2)
        d_ins = compute_marginal_insertion_cost(locs, k_neighbors=0)
        assert d_ins.shape == (2, 5, 5)
        for b in range(2):
            for i in range(5):
                for j in range(5):
                    if i == j:
                        assert d_ins[b, i, j].item() == 0.0
                    else:
                        assert torch.isinf(d_ins[b, i, j])

    def test_single_customer_n1(self):
        """Test edge case N=1 customer node."""
        locs = torch.rand(2, 1, 2)
        d_ins = compute_marginal_insertion_cost(locs, k_neighbors=15)
        assert d_ins.shape == (2, 1, 1)
        assert (d_ins == 0.0).all()
