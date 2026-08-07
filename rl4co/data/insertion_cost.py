"""
Insertion cost utilities for Metric-Aware Slot NCO.

NOTE on d_ins formula:
    The formula used here is:
        d_ins(i, j) = dist(D, i) + dist(i, j) - dist(D, j)

    This is the "Clarke-Wright savings"-style marginal cost of visiting node i
    on a sub-tour (D -> j -> D), not a true cheapest-insertion into a built
    route. It ignores demand/capacity context and full route structure.
    It may correlate with Euclidean distance for some instances.

    TODO: If ablation results show D vs C are not clearly separated, replace
    with a proper nearest-neighbor route construction + marginal insertion cost
    computation that accounts for route structure.

Format:
    Dense (B, N, N): compute_marginal_insertion_cost() — used internally only
    Sparse (B, N, k): compute_sparse_insertion_cost() — used for dataset storage

Sparsification is done AFTER symmetrization to ensure d_ins(i,j) == d_ins(j,i)
before selecting top-k neighbors. This avoids the correctness issue that would
arise from trying to symmetrize an already-sparse tensor where j in kNN(i)
does not imply i in kNN(j).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def compute_pairwise_distance_matrix(coords: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise Euclidean distance matrix using torch.cdist.

    Args:
        coords: (batch_size, N, 2) or (N, 2) coordinate tensor.

    Returns:
        dist_matrix: (batch_size, N, N) or (N, N) pairwise distance matrix.
    """
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


def _compute_dense_d_ins(
    locs: torch.Tensor,
    depot_loc: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute dense (B, N, N) symmetrized marginal insertion cost matrix.

    d_ins(i, j) = dist(D, i) + dist(i, j) - dist(D, j)   [Clarke-Wright proxy]

    Symmetrized: d_ins_sym = (d_ins + d_ins^T) / 2
    Self-pairs set to 0.

    Args:
        locs:      (B, N, 2) customer coordinates (batch must already be 3D).
        depot_loc: (B, 1, 2) depot coordinates.

    Returns:
        d_ins_sym: (B, N, N) float32 — symmetric insertion cost matrix.
    """
    B, N, _ = locs.shape
    device = locs.device

    if depot_loc is None:
        depot_loc = torch.full((B, 1, 2), 0.5, device=device, dtype=locs.dtype)

    # Pairwise customer distances: (B, N, N)
    dist_customers = compute_pairwise_distance_matrix(locs)

    # Distance from depot to all customers: (B, N)
    dist_depot = torch.norm(locs - depot_loc, p=2, dim=-1)

    # Marginal insertion cost: d_ins(i, j) = dist(D,i) + dist(i,j) - dist(D,j)
    d_ins = dist_depot.unsqueeze(2) + dist_customers - dist_depot.unsqueeze(1)
    d_ins = torch.clamp(d_ins, min=0.0)

    # Symmetrize BEFORE sparsifying (avoids asymmetric kNN issues)
    d_ins = (d_ins + d_ins.transpose(-1, -2)) / 2.0

    # Zero out self-pairs
    eye_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    d_ins = d_ins.masked_fill(eye_mask, 0.0)

    return d_ins


def compute_marginal_insertion_cost(
    locs: torch.Tensor,
    k_neighbors: Optional[int] = 15,
    depot_loc: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute dense (B, N, N) insertion cost, sparsified with inf for non-kNN pairs.

    This is kept for backward compatibility and internal use.
    For dataset storage, prefer compute_sparse_insertion_cost().

    Args:
        locs:        (B, N, 2) or (N, 2) customer coordinates.
        k_neighbors: k for kNN sparsification. None = dense (no inf mask).
        depot_loc:   Optional depot coords.

    Returns:
        d_ins: (B, N, N) or (N, N) — inf for non-kNN pairs.
    """
    if locs.dim() == 2:
        locs = locs.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    B, N, _ = locs.shape
    device = locs.device

    if depot_loc is None:
        depot_loc_3d = torch.full((B, 1, 2), 0.5, device=device, dtype=locs.dtype)
    elif depot_loc.dim() == 1:
        depot_loc_3d = depot_loc.unsqueeze(0).unsqueeze(0).expand(B, 1, 2)
    elif depot_loc.dim() == 2:
        depot_loc_3d = depot_loc.unsqueeze(1)
    else:
        depot_loc_3d = depot_loc

    d_ins = _compute_dense_d_ins(locs, depot_loc_3d)  # (B, N, N) symmetric

    if k_neighbors is not None and k_neighbors < N:
        dist_customers = compute_pairwise_distance_matrix(locs)
        _, knn_indices = torch.topk(dist_customers, k=k_neighbors + 1, dim=-1, largest=False)
        knn_mask = torch.zeros((B, N, N), dtype=torch.bool, device=device)
        knn_mask.scatter_(2, knn_indices, torch.ones_like(knn_indices, dtype=torch.bool))
        d_ins = d_ins.masked_fill(~knn_mask, float("inf"))

    if squeeze_batch:
        d_ins = d_ins.squeeze(0)

    return d_ins


def compute_sparse_insertion_cost(
    locs: torch.Tensor,
    k_neighbors: int = 15,
    depot_loc: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute sparse (B, N, k) insertion cost for efficient storage and aggregation.

    Symmetrizes BEFORE selecting top-k to ensure d_ins(i,j) == d_ins(j,i)
    and consistent neighbor selection.

    Storage format:
        idx: (B, N, k) int16  — k-NN neighbor indices (cast to long before use)
        val: (B, N, k) float32 — insertion costs to those k neighbors

    Args:
        locs:        (B, N, 2) customer coordinates. Must be 3D.
        k_neighbors: Number of nearest neighbors per node.
        depot_loc:   (B, 1, 2) depot coordinates. Defaults to (0.5, 0.5).

    Returns:
        (idx, val):
            idx: (B, N, k) int16
            val: (B, N, k) float32
    """
    B, N, _ = locs.shape
    device = locs.device

    if depot_loc is None:
        depot_loc = torch.full((B, 1, 2), 0.5, device=device, dtype=locs.dtype)
    elif depot_loc.dim() == 2:
        depot_loc = depot_loc.unsqueeze(1)

    # Compute dense symmetric d_ins: (B, N, N)
    d_ins = _compute_dense_d_ins(locs, depot_loc)

    # Use Euclidean distance to determine kNN neighbours (not d_ins values)
    # so that we select the k geometrically closest nodes (consistent with training)
    dist_customers = compute_pairwise_distance_matrix(locs)  # (B, N, N)
    # Exclude self (distance 0) by adding large value on diagonal
    dist_customers = dist_customers + torch.eye(N, device=device).unsqueeze(0) * 1e9

    k_actual = min(k_neighbors, N - 1)
    _, knn_idx = torch.topk(dist_customers, k=k_actual, dim=-1, largest=False)
    # knn_idx: (B, N, k_actual)

    # Gather the d_ins values at kNN positions
    knn_val = torch.gather(d_ins, dim=2, index=knn_idx)  # (B, N, k_actual)

    # Store indices as int16 to save disk space (supports N up to 32767)
    # Must be cast to .long() before use in torch.gather during training
    assert N <= 32767, f"N={N} exceeds int16 range; use int32 for larger instances"
    knn_idx_i16 = knn_idx.to(torch.int16)

    return knn_idx_i16, knn_val
