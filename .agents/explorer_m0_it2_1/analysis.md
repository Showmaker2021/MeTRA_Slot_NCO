# Technical Analysis & Fix Strategy — Milestone M0 Iteration 2

**Agent**: `explorer_m0_it2_1`  
**Target File**: `rl4co/data/insertion_cost.py`  
**Date**: 2026-08-06  
**Status**: Read-only Analysis Complete  

---

## 1. Executive Summary

Challenger 2 identified 3 test failures out of 23 stress tests for the $d_{\text{ins}}$ insertion cost operator:
1. `RuntimeError: "cdist" not implemented for 'Half'` on CPU with `torch.float16`.
2. `RuntimeError: "cdist" not implemented for 'BFloat16'` on CPU with `torch.bfloat16`.
3. `RuntimeError: ... INTERNAL ASSERT FAILED ... We don't have an op for aten::scatter_ ... Argument types: Tensor, int, Tensor, bool` when tracing `compute_marginal_insertion_cost` using `torch.jit.trace`.

This analysis provides the exact root causes, technical rationale, and code modifications required for `rl4co/data/insertion_cost.py` to achieve 100% test pass rate across Eager execution, TorchScript (`torch.jit.script`), PyTorch JIT tracing (`torch.jit.trace`), autograd backpropagation, and all floating-point precisions (`float64`, `float32`, `float16`, `bfloat16`).

---

## 2. Failure 1: Half (`float16`) and BFloat16 CPU `cdist` Support

### 2.1 Observation & Root Cause
- **Location**: `rl4co/data/insertion_cost.py:23` in `compute_pairwise_distance_matrix`
- **Code snippet**:
  ```python
  dist_matrix = torch.cdist(coords, coords, p=2.0)
  ```
- **Root Cause**:
  PyTorch's CPU backend lacks native C++ kernel implementations for `torch.cdist` when operating on `torch.float16` (Half) or `torch.bfloat16` tensors. Calling `torch.cdist` on CPU with these dtypes throws an immediate `RuntimeError`.

### 2.2 Rationale & Fix Strategy
- Auto-detect if `coords.dtype` is `torch.float16` or `torch.bfloat16`.
- If true, temporarily cast `coords` to `torch.float32` before passing to `torch.cdist`, and cast the resulting distance matrix back to `coords.dtype`.
- In PyTorch, `.to(torch.float32)` and `.to(orig_dtype)` preserve the autograd computational graph via built-in VJPs.
- The condition `orig_dtype == torch.float16 or orig_dtype == torch.bfloat16` is fully compatible with TorchScript (`torch.jit.script`) and PyTorch JIT tracing (`torch.jit.trace`).

### 2.3 Proposed Code Modification
```python
def compute_pairwise_distance_matrix(coords: torch.Tensor) -> torch.Tensor:
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    orig_dtype = coords.dtype
    if orig_dtype == torch.float16 or orig_dtype == torch.bfloat16:
        coords_f32 = coords.to(torch.float32)
        dist_matrix = torch.cdist(coords_f32, coords_f32, p=2.0).to(orig_dtype)
    else:
        dist_matrix = torch.cdist(coords, coords, p=2.0)

    if squeeze_batch:
        dist_matrix = dist_matrix.squeeze(0)

    return dist_matrix
```

---

## 3. Failure 2: PyTorch JIT Tracing Assertion in `scatter_`

### 3.1 Observation & Root Cause
- **Location**: `rl4co/data/insertion_cost.py:95` in `compute_marginal_insertion_cost`
- **Code snippet**:
  ```python
  knn_mask.scatter_(2, knn_indices, True)
  ```
- **Root Cause**:
  In PyTorch's C++ JIT Tracer (`torch.jit.trace`), passing a Python primitive scalar `True` (bool) as the `src` parameter to `scatter_` attempts to resolve the overload `aten::scatter_(Tensor self, int dim, Tensor index, bool value)`. This overload is not registered in the JIT alias analyzer (`alias_analysis.cpp`), leading to `RuntimeError: 0 INTERNAL ASSERT FAILED`. In addition, converting scalar constants into tracer nodes emits TracerWarnings.

### 3.2 Rationale & Fix Strategy
- Replace the scalar boolean `True` with a tensor argument: `torch.ones_like(knn_indices, dtype=torch.bool)`.
- `torch.ones_like(knn_indices, dtype=torch.bool)` generates a boolean tensor matching `knn_indices`'s shape `(B, N, k+1)` and device.
- Passing a Tensor `src` routes `scatter_` to the standard C++ overload `aten::scatter_.src(Tensor self, int dim, Tensor index, Tensor src)`, which is fully registered and supported by `torch.jit.trace`, `torch.jit.script`, and PyTorch Eager mode.

### 3.3 Proposed Code Modification
```python
    # 4. k-NN Sparsification if requested and N > k
    if k_neighbors is not None and k_neighbors < N:
        _, knn_indices = torch.topk(dist_customers, k=k_neighbors + 1, dim=-1, largest=False)
        knn_mask = torch.zeros((B, N, N), dtype=torch.bool, device=device)
        knn_mask.scatter_(2, knn_indices, torch.ones_like(knn_indices, dtype=torch.bool))
        d_ins = d_ins.masked_fill(~knn_mask, float("inf"))
```

---

## 4. Complete Proposed Implementation for `rl4co/data/insertion_cost.py`

Below is the complete, drop-in replacement implementation for `rl4co/data/insertion_cost.py`:

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Tuple


def compute_pairwise_distance_matrix(coords: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise Euclidean distance matrix for node coordinates using torch.cdist.

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
    if orig_dtype == torch.float16 or orig_dtype == torch.bfloat16:
        coords_f32 = coords.to(torch.float32)
        dist_matrix = torch.cdist(coords_f32, coords_f32, p=2.0).to(orig_dtype)
    else:
        dist_matrix = torch.cdist(coords, coords, p=2.0)

    if squeeze_batch:
        dist_matrix = dist_matrix.squeeze(0)

    return dist_matrix


def compute_marginal_insertion_cost(
    locs: torch.Tensor,
    k_neighbors: Optional[int] = 15,
    depot_loc: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes pairwise marginal insertion cost matrix d_ins(i, j) sparsified by k-NN.

    d_ins(i, j) is defined as:
        Cost of inserting node i into the optimal position around node j
        minus cost of visiting node j alone (with depot).

    For a sub-tour containing Depot (D) and Node j: (D -> j -> D), cost is 2 * dist(D, j).
    Inserting node i between D and j or j and D yields:
        Insertion 1 (D -> i -> j -> D): dist(D, i) + dist(i, j) + dist(j, D)
        Delta cost = dist(D, i) + dist(i, j) - dist(D, j)

    Args:
        locs: (batch_size, N, 2) or (N, 2) customer location coordinates.
        k_neighbors: Optional int. If specified, restricts exact evaluation to k-NN of each node,
                      filling non-neighbors with inf.
        depot_loc: Optional depot coordinates. Can be (2,), (1, 2), (B, 2), or (B, 1, 2).
                   If None, defaults to (0.5, 0.5) for all batch items.

    Returns:
        d_ins: (batch_size, N, N) or (N, N) pairwise marginal insertion cost matrix.
    """
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

    # 1. Pairwise customer distances: (B, N, N)
    dist_customers = compute_pairwise_distance_matrix(locs)

    # 2. Distance from depot to all customers: (B, N)
    dist_depot = torch.norm(locs - depot_loc, p=2, dim=-1)

    # 3. Marginal insertion cost of i into sub-tour around j:
    # d_ins(i, j) = dist(D, i) + dist(i, j) - dist(D, j)
    d_ins = dist_depot.unsqueeze(2) + dist_customers - dist_depot.unsqueeze(1)

    # Clamp float32 numerical underflow below zero
    d_ins = torch.clamp(d_ins, min=0.0)

    # Self-insertion d_ins(i, i) = 0.0
    eye_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    d_ins = d_ins.masked_fill(eye_mask, 0.0)

    # 4. k-NN Sparsification if requested and N > k
    if k_neighbors is not None and k_neighbors < N:
        _, knn_indices = torch.topk(dist_customers, k=k_neighbors + 1, dim=-1, largest=False)
        knn_mask = torch.zeros((B, N, N), dtype=torch.bool, device=device)
        knn_mask.scatter_(2, knn_indices, torch.ones_like(knn_indices, dtype=torch.bool))
        d_ins = d_ins.masked_fill(~knn_mask, float("inf"))

    if squeeze_batch:
        d_ins = d_ins.squeeze(0)

    return d_ins
```

---

## 5. Verification Plan

Once the implementer applies these changes to `rl4co/data/insertion_cost.py`:

1. **Run Full Pytest Suite**:
   ```powershell
   D:\Miniconda\miniconda3\envs\ec_nco\python.exe -m pytest tests/test_insertion_cost.py tests/test_insertion_cost_stress.py -v
   ```
2. **Success Criteria**:
   - `23 PASSED` out of 23 total test cases.
   - 0 failures, 0 errors.
