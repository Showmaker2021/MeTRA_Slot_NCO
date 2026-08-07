"""
Evaluation script using PyVRP (HGS solver) to compute Ground Truth solutions
and average tour lengths on our slot-NCO test datasets.

Usage:
    # Install pyvrp first:
    # pip install pyvrp

    # Run evaluation on CVRP-100 uniform test set:
    python scripts/eval_pyvrp.py --test_file ./data/slot_datasets_v2/cvrp100_uniform_test.pt --time_limit 1.0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import torch

try:
    import pyvrp
    from pyvrp import Model
    from pyvrp.stop import MaxRuntime
    PYVRP_AVAILABLE = True
except ImportError:
    PYVRP_AVAILABLE = False


def solve_instance_pyvrp(
    locs: torch.Tensor,
    depot: torch.Tensor,
    demands: torch.Tensor,
    capacity: float,
    time_limit: float = 1.0,
) -> float:
    """
    Solves a single CVRP instance using PyVRP (HGS solver).
    Since our datasets normalize demand by vehicle capacity (so capacity = 1.0),
    we scale up values to integer precision to avoid numeric issues in HGS.
    """
    model = Model()
    
    # Coordinates in PyVRP can be float, but HGS is internal integer.
    # PyVRP handles coordinates mapping automatically.
    depot_node = model.add_depot(x=float(depot[0]), y=float(depot[1]))
    
    # Scale demands to integers to avoid floating point issues
    scale = 10000
    int_capacity = int(capacity * scale)
    
    clients = []
    for i in range(len(locs)):
        int_demand = int(demands[i] * scale)
        clients.append(
            model.add_client(
                x=float(locs[i, 0]),
                y=float(locs[i, 1]),
                demand=int_demand,
            )
        )
        
    # Standard CVRP allows unlimited vehicles (we set a large count)
    model.add_vehicle_type(num_vehicles=len(locs), capacity=int_capacity)
    
    # Solve with time limit (seconds)
    result = model.solve(stop=MaxRuntime(time_limit))
    
    # cost is the total Euclidean tour distance
    return result.cost()


def main():
    if not PYVRP_AVAILABLE:
        print("\n[ERROR] pyvrp is not installed in this environment.")
        print("Please install it using: pip install pyvrp\n")
        return

    parser = argparse.ArgumentParser(description="Evaluate PyVRP (HGS) baseline on slot datasets")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test .pt dataset file")
    parser.add_argument("--time_limit", type=float, default=1.0, help="HGS solver runtime limit per instance (seconds)")
    parser.add_argument("--max_instances", type=int, default=None, help="Limit number of instances to evaluate")
    parser.add_argument("--out_dir", type=str, default="./results", help="Directory to save JSON results")
    args = parser.parse_args()

    test_path = Path(args.test_file)
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    print(f"Loading dataset: {test_path}")
    # Load test set on CPU
    data = torch.load(test_path, map_location="cpu", weights_only=True)
    
    locs = data["locs"]       # (B, N, 2)
    depot = data["depot"]     # (B, 2)
    demand = data["demand"]   # (B, N)
    
    # capacity is either a single float, a tensor of (B, 1), or defaults to 1.0 (since normalisation was applied)
    capacity = data.get("capacity", None)
    
    B = len(locs)
    if args.max_instances is not None:
        B = min(B, args.max_instances)
        
    print(f"Evaluating {B} instances using PyVRP (time_limit={args.time_limit}s per instance)...")
    
    costs = []
    t_start = time.time()
    
    for i in range(B):
        # Resolve capacity value
        cap_val = 1.0
        if capacity is not None:
            if isinstance(capacity, torch.Tensor):
                cap_val = float(capacity[i].item())
            else:
                cap_val = float(capacity)
                
        cost = solve_instance_pyvrp(
            locs=locs[i],
            depot=depot[i],
            demands=demand[i],
            capacity=cap_val,
            time_limit=args.time_limit,
        )
        costs.append(cost)
        
        if (i + 1) % 100 == 0 or (i + 1) == B:
            elapsed = time.time() - t_start
            avg_cost = sum(costs) / len(costs)
            print(f"  [{i+1}/{B}] running avg tour length: {avg_cost:.4f} | elapsed: {elapsed:.1f}s")

    t_total = time.time() - t_start
    final_avg = sum(costs) / len(costs)
    
    # Save results summary
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    
    result_summary = {
        "dataset": test_path.name,
        "num_instances": B,
        "time_limit_per_instance": args.time_limit,
        "avg_tour_length": final_avg,
        "total_elapsed_sec": t_total,
        "avg_sec_per_instance": t_total / B,
        "costs": costs
    }
    
    out_file = out_dir / f"pyvrp_{test_path.stem}.json"
    out_file.write_text(json.dumps(result_summary, indent=2))
    
    print("\n" + "=" * 60)
    print("PyVRP (HGS) Evaluation Summary")
    print("=" * 60)
    print(f"  Dataset:         {test_path.name}")
    print(f"  Instances:       {B}")
    print(f"  Avg Tour Length: {final_avg:.4f}")
    print(f"  Total Time:      {t_total:.1f}s ({t_total/B:.3f}s/instance)")
    print(f"  Results saved:   {out_file}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
