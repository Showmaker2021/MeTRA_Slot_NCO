"""
Script to check correlation between d_ins (Clarke-Wright proxy) and raw Euclidean distance.

If correlation > 0.9, d_ins may not provide enough signal beyond Euclidean distance.
In that case, consider replacing with a proper nearest-neighbor route insertion cost.

Usage:
    python -X utf8 scripts/check_dins_correlation.py --n 100 --n_samples 50
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",          type=int, default=100, help="Number of customer nodes")
    parser.add_argument("--n_samples",  type=int, default=50,  help="Number of instances to average over")
    parser.add_argument("--k",          type=int, default=15,  help="k-NN neighbors used in d_ins")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    from rl4co.data.insertion_cost import compute_sparse_insertion_cost, compute_pairwise_distance_matrix

    all_corrs = []

    for _ in range(args.n_samples):
        locs  = torch.rand(1, args.n, 2)
        depot = torch.rand(1, 1, 2)

        # Sparse d_ins (symmetrized, kNN selected by Euclidean distance)
        idx_i16, val = compute_sparse_insertion_cost(locs, k_neighbors=args.k, depot_loc=depot)
        # (1, N, k)

        # Euclidean distances at same kNN positions
        dist_all = compute_pairwise_distance_matrix(locs)  # (1, N, N)
        idx_long = idx_i16.long()  # (1, N, k)
        euclid_knn = torch.gather(dist_all, dim=2, index=idx_long)  # (1, N, k)

        # Flatten and compute Pearson correlation
        d = val.flatten().float()
        e = euclid_knn.flatten().float()

        # Pearson r
        d_c = d - d.mean()
        e_c = e - e.mean()
        r = (d_c * e_c).sum() / (d_c.norm() * e_c.norm() + 1e-8)
        all_corrs.append(r.item())

    corrs = torch.tensor(all_corrs)
    mean_r = corrs.mean().item()
    std_r  = corrs.std().item()

    print("\n" + "=" * 60)
    print(f"d_ins vs Euclidean Correlation Check  (N={args.n}, k={args.k})")
    print("=" * 60)
    print(f"  Mean Pearson r : {mean_r:.4f}")
    print(f"  Std  Pearson r : {std_r:.4f}")
    print(f"  Min            : {corrs.min().item():.4f}")
    print(f"  Max            : {corrs.max().item():.4f}")

    if mean_r > 0.9:
        print("\n  [WARNING] High correlation (>0.9) detected!")
        print("  d_ins signal may not be sufficiently different from Euclidean.")
        print("  Consider implementing proper nearest-neighbor route insertion cost.")
        print("  For now: continue ablation but interpret D vs C results cautiously.")
    else:
        print(f"\n  [OK] Correlation {mean_r:.3f} < 0.9 — d_ins provides")
        print("  distinct signal from raw Euclidean distance.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
