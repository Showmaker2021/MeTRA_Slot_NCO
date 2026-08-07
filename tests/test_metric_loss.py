"""
Unit tests for MetricPreservationLoss, ProjectionHead, and SlotEntropyLoss (M3/M4).
"""

import torch
import pytest
from rl4co.models.nn.metric_loss import (
    ProjectionHead,
    MetricPreservationLoss,
    SlotEntropyLoss,
    _aggregate_d_ins,
    _euclidean_target,
)
from rl4co.data.insertion_cost import compute_marginal_insertion_cost


def make_slots_and_assignments(B=2, N=20, K=6, d=64):
    slots = torch.randn(B, K, d)
    A_ik = torch.randn(B, N, K).softmax(dim=-1)
    locs = torch.rand(B, N, 2)
    return slots, A_ik, locs


def make_d_ins(B=2, N=20, k_neighbors=5):
    locs = torch.rand(B, N, 2)
    depot = torch.full((B, 1, 2), 0.5)
    d_ins = compute_marginal_insertion_cost(locs, k_neighbors=k_neighbors, depot_loc=depot)
    return locs, d_ins


# ── ProjectionHead ────────────────────────────────────────────────────────────

def test_projection_head_shape():
    head = ProjectionHead(input_dim=64, proj_dim=32)
    z = torch.randn(3, 8, 64)
    out = head(z)
    assert out.shape == (3, 8, 32)


def test_projection_head_gradient():
    head = ProjectionHead(input_dim=64, proj_dim=32)
    z = torch.randn(2, 6, 64, requires_grad=True)
    head(z).sum().backward()
    assert z.grad is not None
    assert not torch.isnan(z.grad).any()


# ── SlotEntropyLoss ───────────────────────────────────────────────────────────

def test_entropy_loss_lower_for_uniform():
    B, N, K = 2, 20, 6
    A_uniform = torch.ones(B, N, K) / K
    A_collapsed = torch.zeros(B, N, K)
    A_collapsed[:, :, 0] = 1.0
    loss_fn = SlotEntropyLoss()
    assert loss_fn(A_uniform) < loss_fn(A_collapsed)


def test_entropy_loss_gradient():
    A_ik = torch.randn(2, 10, 4).softmax(dim=-1).requires_grad_(True)
    SlotEntropyLoss()(A_ik).backward()
    assert A_ik.grad is not None and not torch.isnan(A_ik.grad).any()


# ── Target aggregators ────────────────────────────────────────────────────────

def test_aggregate_d_ins_shape_and_no_nan():
    B, N, K = 2, 20, 6
    locs, d_ins = make_d_ins(B, N)
    A_ik = torch.randn(B, N, K).softmax(dim=-1)
    D = _aggregate_d_ins(d_ins, A_ik)
    assert D.shape == (B, K, K)
    assert not torch.isnan(D).any()
    assert not torch.isinf(D).any()


def test_euclidean_target_symmetric():
    B, N, K = 2, 20, 4
    locs = torch.rand(B, N, 2)
    A_ik = torch.randn(B, N, K).softmax(dim=-1)
    D = _euclidean_target(locs, A_ik)
    assert D.shape == (B, K, K)
    assert torch.allclose(D, D.transpose(1, 2), atol=1e-5)


# ── MetricPreservationLoss ────────────────────────────────────────────────────

@pytest.mark.parametrize("variant", ["C", "D"])
def test_metric_loss_scalar_no_nan(variant):
    B, N, K, d = 2, 20, 6, 64
    proj = ProjectionHead(d, proj_dim=32)
    loss_fn = MetricPreservationLoss(proj, variant=variant)
    slots, A_ik, locs = make_slots_and_assignments(B, N, K, d)
    _, d_ins = make_d_ins(B, N)
    loss, info = loss_fn(slots=slots, A_ik=A_ik, locs=locs, d_ins=d_ins)
    assert loss.shape == ()
    assert not torch.isnan(loss) and not torch.isinf(loss)


def test_lambda_always_positive():
    proj = ProjectionHead(64, proj_dim=32)
    loss_fn = MetricPreservationLoss(proj, variant="C", lambda_init=1.0)
    assert loss_fn.lmbda > 0


@pytest.mark.parametrize("variant", ["C", "D"])
def test_gradient_flows_to_slots(variant):
    B, N, K, d = 2, 20, 6, 64
    proj = ProjectionHead(d, proj_dim=32)
    loss_fn = MetricPreservationLoss(proj, variant=variant)
    slots = torch.randn(B, K, d, requires_grad=True)
    A_ik = torch.randn(B, N, K).softmax(dim=-1)
    locs, d_ins = make_d_ins(B, N)
    loss, _ = loss_fn(slots=slots, A_ik=A_ik, locs=locs, d_ins=d_ins)
    loss.backward()
    assert slots.grad is not None and not torch.isnan(slots.grad).any()


def test_lambda_stability_500_steps():
    """Lambda must remain positive and NaN-free over 500 gradient steps."""
    B, N, K, d = 2, 15, 4, 32
    proj = ProjectionHead(d, proj_dim=16)
    loss_fn = MetricPreservationLoss(proj, variant="C", lambda_init=1.0)
    optim = torch.optim.Adam(loss_fn.parameters(), lr=1e-3)
    for step in range(500):
        slots = torch.randn(B, K, d)
        A_ik = torch.randn(B, N, K).softmax(dim=-1)
        locs, d_ins = make_d_ins(B, N)
        optim.zero_grad()
        loss, _ = loss_fn(slots=slots, A_ik=A_ik, locs=locs, d_ins=d_ins)
        loss.backward()
        optim.step()
        lmbda = loss_fn.lmbda.item()
        assert lmbda == lmbda, f"Lambda is NaN at step {step}"
        assert lmbda > 0, f"Lambda non-positive at step {step}: {lmbda}"

    print(f"Lambda after 500 steps: {loss_fn.lmbda.item():.4f}")
