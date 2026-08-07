"""
Utility script to export our cached PyTorch (.pt) datasets
to standard Pickle (.pkl) format used by classical baseline repositories
(such as Kool's Attention Model, original POMO, and RADAR).

Usage:
    python scripts/export_to_am_format.py --pt_file ./data/slot_datasets_v2/cvrp100_uniform_test.pt
"""

import argparse
import pickle
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description="Export .pt dataset to standard NCO baseline .pkl format")
    parser.add_argument("--pt_file", type=str, required=True, help="Path to .pt dataset file")
    parser.add_argument("--out_file", type=str, default=None, help="Output .pkl file path (defaults to same name)")
    args = parser.parse_args()

    pt_path = Path(args.pt_file)
    if not pt_path.exists():
        raise FileNotFoundError(f"Input file not found: {pt_path}")

    # Set default output name
    if args.out_file is None:
        out_path = pt_path.with_suffix(".pkl")
    else:
        out_path = Path(args.out_file)

    print(f"Loading PyTorch dataset: {pt_path}")
    data = torch.load(pt_path, map_location="cpu", weights_only=True)

    locs = data["locs"]       # (B, N, 2)
    depot = data["depot"]     # (B, 2)
    demand = data["demand"]   # (B, N)
    capacity = data.get("capacity", None)

    B = len(locs)
    print(f"Processing {B} instances...")

    # Convert to list of tuples: (depot_coords, customer_coords, demands, capacity)
    # matching the exact structure expected by Kool's AM and RADAR evaluation loaders.
    dataset_pkl = []
    for i in range(B):
        # Resolve capacity
        cap_val = 1.0
        if capacity is not None:
            if isinstance(capacity, torch.Tensor):
                cap_val = float(capacity[i].item())
            else:
                cap_val = float(capacity)

        # Convert tensors to list/numpy for pickle compatibility
        depot_coords = depot[i].tolist()        # [x, y]
        customer_coords = locs[i].tolist()     # [[x1, y1], [x2, y2], ...]
        instance_demands = demand[i].tolist()   # [d1, d2, ...]

        # Store as standard AM tuple
        dataset_pkl.append((depot_coords, customer_coords, instance_demands, cap_val))

    print(f"Writing to pickle file: {out_path}")
    with open(out_path, "wb") as f:
        pickle.dump(dataset_pkl, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Success! Export complete.\n")


if __name__ == "__main__":
    main()
