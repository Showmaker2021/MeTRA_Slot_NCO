import torch
import pytest
from typing import Optional

def compute_pairwise_distance_matrix_fixed(coords: torch.Tensor) -> torch.Tensor:
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    orig_dtype = coords.dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        coords_f32 = coords.to(torch.float32)
        dist_matrix = torch.cdist(coords_f32, coords_f32, p=2.0).to(orig_dtype)
    else:
        dist_matrix = torch.cdist(coords, coords, p=2.0)

    if squeeze_batch:
        dist_matrix = dist_matrix.squeeze(0)

    return dist_matrix


def compute_marginal_insertion_cost_fixed(
    locs: torch.Tensor,
    k_neighbors: Optional[int] = 15,
    depot_loc: Optional[torch.Tensor] = None,
) -> torch.Tensor:
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

    dist_customers = compute_pairwise_distance_matrix_fixed(locs)
    dist_depot = torch.norm(locs - depot_loc, p=2, dim=-1)
    d_ins = dist_depot.unsqueeze(2) + dist_customers - dist_depot.unsqueeze(1)

    d_ins = torch.clamp(d_ins, min=0.0)

    eye_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    d_ins = d_ins.masked_fill(eye_mask, 0.0)

    if k_neighbors is not None and k_neighbors < N:
        _, knn_indices = torch.topk(dist_customers, k=k_neighbors + 1, dim=-1, largest=False)
        knn_mask = torch.zeros((B, N, N), dtype=torch.bool, device=device)
        knn_mask.scatter_(2, knn_indices, 1)
        d_ins = d_ins.masked_fill(~knn_mask, float("inf"))

    if squeeze_batch:
        d_ins = d_ins.squeeze(0)

    return d_ins


def test_float16_fixed():
    locs = torch.rand(2, 8, 2, dtype=torch.float16)
    depot = torch.rand(2, 1, 2, dtype=torch.float16)
    d_ins = compute_marginal_insertion_cost_fixed(locs, k_neighbors=3, depot_loc=depot)
    assert d_ins.dtype == torch.float16
    assert not torch.isnan(d_ins).any()

def test_bfloat16_fixed():
    locs = torch.rand(2, 8, 2, dtype=torch.bfloat16)
    depot = torch.rand(2, 1, 2, dtype=torch.bfloat16)
    d_ins = compute_marginal_insertion_cost_fixed(locs, k_neighbors=3, depot_loc=depot)
    assert d_ins.dtype == torch.bfloat16
    assert not torch.isnan(d_ins).any()

def test_jit_trace_fixed():
    locs = torch.rand(2, 10, 2)
    depot = torch.rand(2, 1, 2)
    traced_fn = torch.jit.trace(
        lambda l, d: compute_marginal_insertion_cost_fixed(l, k_neighbors=4, depot_loc=d),
        (locs, depot),
    )
    res_eager = compute_marginal_insertion_cost_fixed(locs, k_neighbors=4, depot_loc=depot)
    res_traced = traced_fn(locs, depot)
    assert torch.allclose(
        res_eager.masked_fill(torch.isinf(res_eager), 0.0),
        res_traced.masked_fill(torch.isinf(res_traced), 0.0),
    )

def test_jit_script_fixed():
    locs = torch.rand(4, 15, 2)
    depot = torch.rand(4, 1, 2)
    scripted_fn = torch.jit.script(compute_marginal_insertion_cost_fixed)
    res_eager = compute_marginal_insertion_cost_fixed(locs, k_neighbors=5, depot_loc=depot)
    res_script = scripted_fn(locs, 5, depot)
    mask_eager = torch.isinf(res_eager)
    mask_script = torch.isinf(res_script)
    assert torch.equal(mask_eager, mask_script)
    assert torch.allclose(res_eager[~mask_eager], res_script[~mask_script])
