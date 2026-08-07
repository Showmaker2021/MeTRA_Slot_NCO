"""Quick integration test for all 3 core modules — bypasses pytest/torchrl DLL issue."""
import sys
sys.path.insert(0, ".")
import torch

print("=== TEST M0: Insertion Cost ===")
from rl4co.data.insertion_cost import compute_marginal_insertion_cost, compute_pairwise_distance_matrix

coords = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
dist = compute_pairwise_distance_matrix(coords)
assert dist.shape == (3, 3)
assert abs(dist[0, 1].item() - 3.0) < 1e-5
assert abs(dist[1, 2].item() - 5.0) < 1e-5

locs = torch.rand(2, 20, 2)
d = compute_marginal_insertion_cost(locs, k_neighbors=5)
assert d.shape == (2, 20, 20)
assert not torch.isnan(d[~torch.isinf(d)]).any()
non_inf = (~torch.isinf(d[0])).sum(dim=-1)
assert torch.all(non_inf == 6), f"Expected 6 non-inf per row (self+5 neighbors), got: {non_inf}"
print("PASS: insertion_cost M0")

print("\n=== TEST M2: SlotAttention ===")
from rl4co.models.nn.slot_attention import SlotAttention

# Standard interface
m = SlotAttention(num_slots=8, dim=128, iters=3)
inp = torch.randn(4, 50, 128)
slots, A = m(inp)
assert slots.shape == (4, 8, 128), f"Got {slots.shape}"
assert A.shape == (4, 50, 8), f"Got {A.shape}"
assert torch.allclose(A.sum(dim=-1), torch.ones(4, 50), atol=1e-5), f"Row sum max: {A.sum(dim=-1).max()}"

# Alias interface: slot_dim + in_dim (asymmetric in_dim != slot_dim)
m2 = SlotAttention(num_slots=5, slot_dim=32, in_dim=64, num_iterations=2)
inp2 = torch.randn(2, 10, 64)
s2, a2 = m2(inp2)
assert s2.shape == (2, 5, 32), f"Got {s2.shape}"
assert a2.shape == (2, 10, 5), f"Got {a2.shape}"
assert torch.allclose(a2.sum(dim=-1), torch.ones(2, 10), atol=1e-5)

# Gradient flow
inp3 = torch.randn(2, 10, 128, requires_grad=True)
s3, a3 = m(inp3)
(s3.sum() + a3.sum()).backward()
assert inp3.grad is not None and not torch.isnan(inp3.grad).any()

# K=1 single slot: attn should be all 1.0
m1 = SlotAttention(num_slots=1, dim=32, iters=2)
_, a1 = m1(torch.randn(2, 10, 32))
assert torch.allclose(a1, torch.ones(2, 10, 1), atol=1e-5)
print("PASS: slot_attention M2")

print("\n=== TEST M3-M4: MetricLoss (C, D) + Lambda Stability ===")
from rl4co.models.nn.metric_loss import (
    ProjectionHead, MetricPreservationLoss, SlotEntropyLoss, _aggregate_d_ins, _euclidean_target
)

# ProjectionHead
ph = ProjectionHead(64, proj_dim=32)
assert ph(torch.randn(2, 8, 64)).shape == (2, 8, 32)

# SlotEntropyLoss
ent = SlotEntropyLoss()
A_uniform = torch.ones(2, 20, 6) / 6
A_collapsed = torch.zeros(2, 20, 6)
A_collapsed[:, :, 0] = 1.0
assert ent(A_uniform) < ent(A_collapsed), "Entropy loss wrong direction"

# Euclidean target symmetric
locs_t = torch.rand(2, 20, 2)
A_t = torch.randn(2, 20, 4).softmax(dim=-1)
D_e = _euclidean_target(locs_t, A_t)
assert D_e.shape == (2, 4, 4)
assert torch.allclose(D_e, D_e.transpose(1, 2), atol=1e-5)

# D_ins aggregation
locs_d, d_ins = locs_t, compute_marginal_insertion_cost(locs_t, k_neighbors=5)
D_ins = _aggregate_d_ins(d_ins, A_t)
assert D_ins.shape == (2, 4, 4) and not torch.isnan(D_ins).any() and not torch.isinf(D_ins).any()

# Variants C + D: forward pass
for variant in ["C", "D"]:
    proj = ProjectionHead(64, proj_dim=32)
    fn = MetricPreservationLoss(proj, variant=variant, lambda_init=1.0)
    s = torch.randn(2, 6, 64, requires_grad=True)
    A_ik = torch.randn(2, 20, 6).softmax(dim=-1)
    locs_v = torch.rand(2, 20, 2)
    d_v = compute_marginal_insertion_cost(locs_v, k_neighbors=5)
    loss, info = fn(slots=s, A_ik=A_ik, locs=locs_v, d_ins=d_v)
    assert not torch.isnan(loss) and not torch.isinf(loss), f"Variant {variant}: bad loss"
    assert fn.lmbda > 0
    loss.backward()
    assert s.grad is not None and not torch.isnan(s.grad).any()
    print(f"  Variant {variant}: loss={loss.item():.4f} lambda={fn.lmbda.item():.4f}")

# Lambda stability over 200 steps (Variant C)
proj_s = ProjectionHead(32, proj_dim=16)
fn_s = MetricPreservationLoss(proj_s, variant="C", lambda_init=1.0)
opt = torch.optim.Adam(fn_s.parameters(), lr=1e-3)
for step in range(200):
    s_ = torch.randn(2, 4, 32)
    A_ = torch.randn(2, 15, 4).softmax(dim=-1)
    locs_ = torch.rand(2, 15, 2)
    d_ = compute_marginal_insertion_cost(locs_, k_neighbors=5)
    opt.zero_grad()
    l, _ = fn_s(s_, A_, locs_, d_)
    l.backward()
    opt.step()
    lv = fn_s.lmbda.item()
    assert lv == lv, f"Lambda NaN at step {step}"
    assert lv > 0, f"Lambda non-positive at step {step}: {lv}"
print(f"PASS: lambda stable 200 steps → {fn_s.lmbda.item():.4f}")

print("\n" + "=" * 50)
print("ALL TESTS PASSED ✓")
print("=" * 50)
