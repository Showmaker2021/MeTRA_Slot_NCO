"""
Ablation Ladder Runner -- Runs Variants A->D sequentially.

Variants:
  A: Reconstruction loss
  B: Task loss only (DPN-style baseline)
  C: Euclidean centroid metric loss
  D: Insertion-cost metric loss (proposed method)
  # E: Future-regret target -- RESERVED, not yet implemented

Usage:
    # Run all variants at N=50 (fast ablation gate, 3 seeds each):
    conda run -n ec_nco python scripts/run_ablation.py --num_loc 50 --n_seeds 3

    # Full scale at N=100 (main paper results):
    conda run -n ec_nco python scripts/run_ablation.py --num_loc 100 --n_seeds 5

    # Only key variants (B vs D comparison):
    conda run -n ec_nco python scripts/run_ablation.py --variants B D --num_loc 100
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path


def run_variant_seed(variant: str, num_loc: int, dist: str, seed: int, extra_args: list[str]):
    cmd = [
        sys.executable, "scripts/train.py",
        "--variant", variant,
        "--num_loc", str(num_loc),
        "--dist", dist,
        "--seed", str(seed),
    ] + extra_args

    print(f"\n>>> Launching: Variant={variant} N={num_loc} dist={dist} seed={seed}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants",  nargs="+", default=list("ABCD"),
                        help="Ablation variants to run. E is reserved (not implemented).")
    parser.add_argument("--num_loc",   type=int,  default=50)
    parser.add_argument("--dists",     nargs="+", default=["uniform", "clustered"])
    parser.add_argument("--n_seeds",   type=int,  default=3)
    parser.add_argument("--devices",   type=int,  default=1)
    args, extra = parser.parse_known_args()

    extra_args = ["--devices", str(args.devices)] + extra

    total = len(args.variants) * len(args.dists) * args.n_seeds
    done, failed = 0, 0

    for variant in args.variants:
        for dist in args.dists:
            for seed in range(args.n_seeds):
                ok = run_variant_seed(variant, args.num_loc, dist, seed, extra_args)
                done += 1
                if not ok:
                    failed += 1
                print(f"Progress: {done}/{total}  (failed: {failed})")

    print(f"\nAblation ladder complete: {done - failed}/{total} succeeded.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
