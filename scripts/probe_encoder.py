"""Probe encoder structure to fix _get_embed_dim and forward pass."""
import sys, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from rl4co.envs import CVRPEnv
from rl4co.models.zoo.pomo_slot import POMOSlot
import torch

env = CVRPEnv(generator_kwargs=dict(num_loc=50))
model = POMOSlot(env=env, num_slots=8, metric_variant="D")

# Print encoder structure
print("=== Policy encoder structure ===")
enc = model.policy.encoder
print(type(enc).__name__)
for name, mod in enc.named_children():
    print(f"  {name}: {type(mod).__name__}")

# Probe the actual embed_dim
print("\n=== Attempting embed_dim detection ===")
try:
    # Method 1: graph_network or net
    dim1 = model._get_embed_dim()
    print(f"  _get_embed_dim() returned: {dim1}")
except Exception as e:
    print(f"  _get_embed_dim() failed: {e}")

# Probe via a named linear
print("\n=== All linear layer dims ===")
for name, m in model.policy.encoder.named_modules():
    if isinstance(m, torch.nn.Linear):
        print(f"  {name}: in={m.in_features} out={m.out_features}")
        break  # just first one

# Test actual encoder forward pass
print("\n=== Test encoder forward pass ===")
td = env.reset(batch_size=[4])
try:
    h, _ = model.policy.encoder(td)
    print(f"  encoder output shape: {h.shape}")  # expect (4, 50, embed_dim)
except Exception as e:
    print(f"  encoder(td) failed: {e}")
    # Try without td
    try:
        out = model.policy.encoder(td)
        print(f"  type: {type(out)}")
    except Exception as e2:
        print(f"  second attempt failed: {e2}")
