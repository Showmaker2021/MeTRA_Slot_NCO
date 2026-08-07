import torch
import pytest
from rl4co.data.insertion_cost import (
    compute_pairwise_distance_matrix,
    compute_marginal_insertion_cost,
)


def test_compute_pairwise_distance_matrix():
    """Verify pairwise distance matrix calculation on 2D and 3D tensors with a 3-4-5 right triangle."""
    # 2D tensor: (0,0), (3,0), (0,4)
    coords_2d = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    dist_2d = compute_pairwise_distance_matrix(coords_2d)

    assert dist_2d.shape == (3, 3)
    assert torch.isclose(dist_2d[0, 1], torch.tensor(3.0))
    assert torch.isclose(dist_2d[0, 2], torch.tensor(4.0))
    assert torch.isclose(dist_2d[1, 2], torch.tensor(5.0))  # 3-4-5 triangle
    assert torch.isclose(dist_2d[1, 0], torch.tensor(3.0))
    assert torch.isclose(dist_2d[2, 0], torch.tensor(4.0))
    assert torch.isclose(dist_2d[2, 1], torch.tensor(5.0))

    # 3D tensor: batch size B=2
    coords_3d = torch.stack([coords_2d, coords_2d * 2.0], dim=0)  # (2, 3, 2)
    dist_3d = compute_pairwise_distance_matrix(coords_3d)

    assert dist_3d.shape == (2, 3, 3)
    # Batch 0: 3-4-5
    assert torch.isclose(dist_3d[0, 0, 1], torch.tensor(3.0))
    assert torch.isclose(dist_3d[0, 0, 2], torch.tensor(4.0))
    assert torch.isclose(dist_3d[0, 1, 2], torch.tensor(5.0))
    # Batch 1: scaled by 2 -> 6-8-10
    assert torch.isclose(dist_3d[1, 0, 1], torch.tensor(6.0))
    assert torch.isclose(dist_3d[1, 0, 2], torch.tensor(8.0))
    assert torch.isclose(dist_3d[1, 1, 2], torch.tensor(10.0))


def test_marginal_insertion_cost_basic():
    """Test shape (B, N, N), diagonal self-insertion == 0.0, non-negativity d_ins >= 0.0, and formula correctness."""
    torch.manual_seed(42)
    B, N = 2, 5
    locs = torch.rand(B, N, 2)
    depot = torch.full((B, 1, 2), 0.5)

    d_ins = compute_marginal_insertion_cost(locs, k_neighbors=None, depot_loc=depot)
    assert d_ins.shape == (B, N, N)

    # Verify non-negativity across the entire tensor
    assert torch.all(d_ins >= 0.0), "d_ins must be non-negative (>= 0.0)"

    # Verify self-insertion diagonal == 0.0 and formula matching
    for b in range(B):
        for i in range(N):
            for j in range(N):
                if i == j:
                    assert d_ins[b, i, j].item() == 0.0
                else:
                    dist_d_i = torch.norm(locs[b, i] - depot[b, 0])
                    dist_i_j = torch.norm(locs[b, i] - locs[b, j])
                    dist_d_j = torch.norm(locs[b, j] - depot[b, 0])
                    expected = dist_d_i + dist_i_j - dist_d_j
                    assert torch.isclose(d_ins[b, i, j], expected, atol=1e-5)


def test_knn_sparsification():
    """Test default k=15 on N=20 and k=3 on N=10, verifying non-inf count per node is k+1 and non-neighbors are float('inf')."""
    torch.manual_seed(42)

    # 1. Test default k=15 on N=20
    locs_n20 = torch.rand(2, 20, 2)
    d_ins_k15 = compute_marginal_insertion_cost(locs_n20, k_neighbors=15)
    assert d_ins_k15.shape == (2, 20, 20)
    for b in range(2):
        non_inf_counts = torch.sum(~torch.isinf(d_ins_k15[b]), dim=-1)
        # Each node must have exactly k+1 = 16 finite entries (self + 15 neighbors)
        assert torch.all(non_inf_counts == 16), (
            f"Expected exactly 16 non-inf entries per row for k=15, N=20, got {non_inf_counts}"
        )
        # Verify non-neighbors are float('inf')
        inf_mask = torch.isinf(d_ins_k15[b])
        assert torch.sum(inf_mask) == 20 * (20 - 16)  # 20 rows * 4 inf per row

    # 2. Test k=3 on N=10
    locs_n10 = torch.rand(2, 10, 2)
    d_ins_k3 = compute_marginal_insertion_cost(locs_n10, k_neighbors=3)
    assert d_ins_k3.shape == (2, 10, 10)
    for b in range(2):
        non_inf_counts = torch.sum(~torch.isinf(d_ins_k3[b]), dim=-1)
        # Each node must have exactly k+1 = 4 finite entries (self + 3 neighbors)
        assert torch.all(non_inf_counts == 4), (
            f"Expected exactly 4 non-inf entries per row for k=3, N=10, got {non_inf_counts}"
        )
        inf_mask = torch.isinf(d_ins_k3[b])
        assert torch.sum(inf_mask) == 10 * (10 - 4)  # 10 rows * 6 inf per row


def test_edge_cases():
    """Test edge cases: N <= k (e.g. N=5, k=15), unbatched input (N, 2), k=None, and custom depot_loc shapes."""
    torch.manual_seed(42)

    # 1. N <= k (N=5, k=15): should skip sparsification, return dense matrix with no inf
    locs_small = torch.rand(2, 5, 2)
    d_ins_n5 = compute_marginal_insertion_cost(locs_small, k_neighbors=15)
    assert d_ins_n5.shape == (2, 5, 5)
    assert not torch.any(torch.isinf(d_ins_n5)), "Expected dense matrix (no inf) when N <= k"
    for b in range(2):
        for i in range(5):
            assert d_ins_n5[b, i, i].item() == 0.0

    # 2. Unbatched input (N, 2): should return 2D output (N, N)
    locs_2d = torch.rand(8, 2)
    d_ins_2d = compute_marginal_insertion_cost(locs_2d, k_neighbors=3)
    assert d_ins_2d.shape == (8, 8)
    assert d_ins_2d[0, 0].item() == 0.0

    # 3. k = None: should return dense matrix (N, N) with no inf
    d_ins_dense = compute_marginal_insertion_cost(locs_2d, k_neighbors=None)
    assert d_ins_dense.shape == (8, 8)
    assert not torch.any(torch.isinf(d_ins_dense)), "Expected dense matrix when k_neighbors is None"

    # 4. Custom depot_loc: test origin (0, 0) and various depot shapes (1D, 2D, 3D)
    depot_1d = torch.tensor([0.0, 0.0])  # (2,)
    d_ins_dep1 = compute_marginal_insertion_cost(locs_2d, k_neighbors=3, depot_loc=depot_1d)
    assert d_ins_dep1.shape == (8, 8)

    depot_2d = torch.tensor([[0.0, 0.0]])  # (1, 2)
    d_ins_dep2 = compute_marginal_insertion_cost(locs_2d, k_neighbors=3, depot_loc=depot_2d)
    assert d_ins_dep2.shape == (8, 8)

    locs_3d = torch.rand(2, 6, 2)
    depot_3d = torch.zeros(2, 1, 2)  # (2, 1, 2)
    d_ins_dep3 = compute_marginal_insertion_cost(locs_3d, k_neighbors=3, depot_loc=depot_3d)
    assert d_ins_dep3.shape == (2, 6, 6)


def test_customer_at_depot_and_colocation():
    """Test customer 0 at depot, customers 1 & 2 co-located. Verify zero insertion cost without zero-division error."""
    depot = torch.tensor([[[0.5, 0.5]]])
    # Customer 0 at depot; Customer 1 & 2 co-located at (0.8, 0.5)
    locs = torch.tensor([[[0.5, 0.5], [0.8, 0.5], [0.8, 0.5]]])

    d_ins = compute_marginal_insertion_cost(locs, k_neighbors=None, depot_loc=depot)
    assert not torch.isnan(d_ins).any(), "d_ins contains NaN"
    assert torch.isclose(d_ins[0, 1, 2], torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(d_ins[0, 2, 1], torch.tensor(0.0), atol=1e-5)


def test_gradient_flow_insertion_cost():
    """Verify autograd backpropagation through customer locations locs if requires_grad=True."""
    locs = torch.rand(2, 6, 2, requires_grad=True)
    d_ins = compute_marginal_insertion_cost(locs, k_neighbors=3)

    # Mask out inf before summing to enable gradient propagation
    valid_d_ins = d_ins.masked_fill(torch.isinf(d_ins), 0.0)
    loss = valid_d_ins.sum()
    loss.backward()

    assert locs.grad is not None, "Gradient should flow to locs"
    assert not torch.isnan(locs.grad).any(), "Gradients contain NaN"
    assert torch.any(locs.grad != 0.0), "Gradients are all zero"


def test_clustered_spatial_distribution():
    """Verify k-NN behavior on clustered locations (inter-cluster non-neighbors are inf)."""
    torch.manual_seed(42)
    c1 = torch.randn(1, 5, 2) * 0.01 + torch.tensor([0.0, 0.0])
    c2 = torch.randn(1, 5, 2) * 0.01 + torch.tensor([10.0, 10.0])
    locs = torch.cat([c1, c2], dim=1)  # (1, 10, 2)

    d_ins = compute_marginal_insertion_cost(locs, k_neighbors=4)
    # For node 0 in cluster 1 (indices 0..4), indices 5..9 in cluster 2 must be inf for small k=4
    assert torch.all(torch.isinf(d_ins[0, 0, 5:])), "Cross-cluster entries must be inf for k=4"
