"""
Forward pass test: verify POMOSlot training step end-to-end
on a small batch (B=4, N=50) before launching full training.
"""
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import torch
from tensordict import TensorDict

from rl4co.envs import CVRPEnv
from rl4co.models.zoo.pomo_slot import POMOSlot

print("=== Forward Pass Test: POMOSlot Variant D ===\n")

env = CVRPEnv(generator_kwargs=dict(num_loc=50))
model = POMOSlot(
    env=env,
    num_slots=8,
    metric_variant="D",
    alpha_metric=0.1,
    beta_entropy=0.01,
    num_starts=10,   # small for speed
    num_augment=1,
)
model.train()

# Simulate a batch matching CVRP generator format exactly
# Ref: CVRPGenerator._generate() output format:
#   locs   : (B, N, 2)  customer coords
#   depot  : (B, 2)     depot coord (no middle dim!)
#   demand : (B, N)     integer demands normalized by vehicle_capacity (1..10/40)
#   capacity: (B, 1)    vehicle capacity (always 1.0 since demand is already normalized)
B = 4
N = 50
VEHICLE_CAPACITY = 40.0  # standard for N=50

locs = torch.rand(B, N, 2)
depot = torch.rand(B, 2)                                     # (B, 2) no middle dim
demand = (torch.randint(1, 10, (B, N)).float()) / VEHICLE_CAPACITY  # normalized like generator
capacity = torch.full((B, 1), 1.0)                          # always 1.0 after normalization
d_ins = torch.rand(B, N, N).abs()                           # proxy

batch = {
    "locs": locs,
    "depot": depot,
    "demand": demand,
    "capacity": capacity,
    "d_ins": d_ins,
}

print("Running shared_step (train)...")
try:
    out = model.shared_step(batch, batch_idx=0, phase="train")
    loss = out.get("loss")
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Output keys: {list(out.keys())}")
    assert loss is not None, "Loss is None!"
    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is Inf!"
    print("\nForward pass OK ✓")
except Exception as e:
    import traceback
    print(f"\nForward pass FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# Also test val step (should skip aux loss)
print("\nRunning shared_step (val)...")
try:
    with torch.no_grad():
        out_val = model.shared_step(batch, batch_idx=0, phase="val")
    print(f"  Val output keys: {list(out_val.keys())}")
    print("  Val step OK ✓")
except Exception as e:
    import traceback
    print(f"\nVal step FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== ALL FORWARD PASS TESTS PASSED ===")
