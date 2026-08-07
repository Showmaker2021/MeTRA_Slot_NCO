"""
Unit test: verify that _aggregate_d_ins_sparse gives the same result as
the old dense aggregation (A^T @ d_ins_dense @ A) on a small instance.

Run with:
    python -m pytest tests/test_sparse_aggregate.py -v
or:
    python tests/test_sparse_aggregate.py
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl4co.models.nn.metric_loss import _aggregate_d_ins_sparse


def _dense_aggregate(d_ins_dense: torch.Tensor, A_ik: torch.Tensor) -> torch.Tensor:
    """Reference dense implementation: D_ins = A^T @ d_ins @ A."""
    A_T = A_ik.transpose(1, 2)  # (B, K, N)
    return torch.bmm(torch.bmm(A_T, d_ins_dense), A_ik)  # (B, K, K)


def _make_sparse_from_dense(
    d_ins_dense: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a dense (B, N, N) matrix to sparse (B, N, k) format by
    selecting the k entries with smallest VALUE per row (excluding self).
    Mimics the behavior of compute_sparse_insertion_cost (kNN by distance).
    """
    B, N, _ = d_ins_dense.shape
    # Exclude self-pairs (set diagonal to large value for topk selection)
    d_work = d_ins_dense.clone()
    eye = torch.eye(N, dtype=torch.bool).unsqueeze(0)
    d_work[eye.expand_as(d_work)] = float("inf")
    # Top-k smallest d_ins values as "neighbors"
    vals, idx = torch.topk(d_work, k=k, dim=-1, largest=False)
    return idx.to(torch.int16), vals


class TestSparseAggregate:

    @pytest.mark.parametrize("B,N,K,k", [
        (1, 20, 4, 5),
        (4, 20, 8, 5),
        (2, 30, 6, 10),
    ])
    def test_sparse_vs_dense(self, B: int, N: int, K: int, k: int):
        """
        Sparse aggregation must match dense aggregation within numerical tolerance.

        Note: sparse only covers k neighbors per row, so we create a dense
        d_ins where non-neighbor entries are zero (same as the masked sum in dense).
        """
        torch.manual_seed(42)

        # Create a symmetric dense d_ins (non-negative, zero diagonal)
        raw = torch.rand(B, N, N)
        d_dense = (raw + raw.transpose(-1, -2)) / 2
        eye = torch.eye(N, dtype=torch.bool).unsqueeze(0)
        d_dense[eye.expand_as(d_dense)] = 0.0

        # Soft assignment matrix (B, N, K), normalised
        A_ik = torch.softmax(torch.randn(B, N, K), dim=-1)

        # Build sparse representation selecting k nearest by d_ins value
        idx_i16, vals = _make_sparse_from_dense(d_dense, k)

        # Build a "zeroed-out" dense reference that only has values at kNN entries
        d_dense_masked = torch.zeros_like(d_dense)
        for b in range(B):
            for i in range(N):
                for ki in range(k):
                    j = idx_i16[b, i, ki].long().item()
                    d_dense_masked[b, i, j] = vals[b, i, ki].item()

        # Dense reference on masked version
        D_dense_ref = _dense_aggregate(d_dense_masked, A_ik)

        # Sparse implementation
        D_sparse = _aggregate_d_ins_sparse(idx_i16, vals, A_ik)

        assert D_sparse.shape == (B, K, K), f"Shape mismatch: {D_sparse.shape}"
        assert torch.allclose(D_sparse, D_dense_ref, rtol=1e-4, atol=1e-5), (
            f"Max diff: {(D_sparse - D_dense_ref).abs().max().item():.6f}"
        )

    def test_no_nan(self):
        """Sparse aggregation must not produce NaN or Inf."""
        torch.manual_seed(0)
        B, N, K, k = 2, 50, 8, 15
        d_dense = torch.rand(B, N, N)
        idx_i16, vals = _make_sparse_from_dense(d_dense, k)
        A_ik = torch.softmax(torch.randn(B, N, K), dim=-1)
        D = _aggregate_d_ins_sparse(idx_i16, vals, A_ik)
        assert not torch.isnan(D).any(), "NaN detected in sparse aggregation output"
        assert not torch.isinf(D).any(), "Inf detected in sparse aggregation output"

    def test_idx_cast_from_int16(self):
        """Must work correctly with int16 indices (cast to long internally)."""
        torch.manual_seed(7)
        B, N, K, k = 1, 20, 4, 5
        d_dense = torch.rand(B, N, N)
        idx_i16, vals = _make_sparse_from_dense(d_dense, k)
        assert idx_i16.dtype == torch.int16, "Expected int16 index tensor"
        A_ik = torch.softmax(torch.randn(B, N, K), dim=-1)
        # Should not raise
        D = _aggregate_d_ins_sparse(idx_i16, vals, A_ik)
        assert D.shape == (B, K, K)


if __name__ == "__main__":
    # Run directly without pytest
    suite = TestSparseAggregate()
    for params in [(1, 20, 4, 5), (4, 20, 8, 5), (2, 30, 6, 10)]:
        suite.test_sparse_vs_dense(*params)
        print(f"  test_sparse_vs_dense{params}: PASSED")
    suite.test_no_nan()
    print("  test_no_nan: PASSED")
    suite.test_idx_cast_from_int16()
    print("  test_idx_cast_from_int16: PASSED")
    print("\nAll tests passed!")
