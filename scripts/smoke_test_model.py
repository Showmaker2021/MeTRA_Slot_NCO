"""Smoke test: instantiate POMOSlot and verify model structure."""
import sys
sys.path.insert(0, ".")
import warnings
warnings.filterwarnings("ignore")

from rl4co.envs import CVRPEnv
from rl4co.models.zoo.pomo_slot import POMOSlot

env = CVRPEnv(num_loc=50)
model = POMOSlot(
    env=env,
    num_slots=8,
    metric_variant="D",
    alpha_metric=0.1,
    beta_entropy=0.01,
)
total_params = sum(p.numel() for p in model.parameters())
slot_params = sum(p.numel() for p in model.slot_attn.parameters())
metric_params = sum(p.numel() for p in model.metric_loss_fn.parameters())

print(f"POMOSlot OK")
print(f"  total_params     : {total_params:,}")
print(f"  slot_attn_params : {slot_params:,}")
print(f"  metric_loss_fn   : {type(model.metric_loss_fn).__name__}")
print(f"  metric_params    : {metric_params:,}")
print(f"  variant          : {model.metric_variant}")
print(f"  K slots          : {model.num_slots}")

# Verify all 5 variants can be instantiated
for v in ["A", "B", "C", "D", "E"]:
    m = POMOSlot(env=env, num_slots=8, metric_variant=v)
    print(f"  Variant {v}: OK  metric_loss_fn={type(m.metric_loss_fn).__name__ if m.metric_loss_fn else 'None'}")

print("\nALL VARIANTS INSTANTIATE OK")
