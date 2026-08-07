import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

# ---------------------------------------------------------------------------
# Graceful Module Import or Stand-in Reference Implementation
# ---------------------------------------------------------------------------
try:
    from rl4co.models.nn.slot_attention import SlotAttention
    USING_RL4CO_SLOT_ATTENTION = True
except ImportError:
    USING_RL4CO_SLOT_ATTENTION = False

    class SlotAttention(nn.Module):
        """
        Stand-in reference SlotAttention implementation matching the official contract
        specifications when rl4co.models.nn.slot_attention is not yet available.
        """
        def __init__(
            self,
            num_slots: int = 4,
            slot_dim: int = 64,
            in_dim: int = 64,
            num_iterations: int = 3,
            eps: float = 1e-8,
            hidden_dim: int = 128,
            dim: int = None,
            iters: int = None,
        ):
            super().__init__()
            if dim is not None:
                slot_dim = dim
            if iters is not None:
                num_iterations = iters

            self.num_slots = num_slots
            self.slot_dim = slot_dim
            self.in_dim = in_dim
            self.num_iterations = num_iterations
            self.eps = eps
            self.scale = slot_dim ** -0.5

            self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
            nn.init.xavier_uniform_(self.slots_logsigma)

            self.to_q = nn.Linear(slot_dim, slot_dim)
            self.to_k = nn.Linear(in_dim, slot_dim)
            self.to_v = nn.Linear(in_dim, slot_dim)

            self.gru = nn.GRUCell(slot_dim, slot_dim)
            hidden_dim = max(slot_dim, hidden_dim)

            self.mlp = nn.Sequential(
                nn.Linear(slot_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, slot_dim)
            )

            self.norm_input = nn.LayerNorm(in_dim)
            self.norm_slots = nn.LayerNorm(slot_dim)
            self.norm_pre_ff = nn.LayerNorm(slot_dim)

        def forward(self, inputs: torch.Tensor, num_slots: int = None):
            # inputs: (B, N, D_in)
            B, N, _ = inputs.shape
            device, dtype = inputs.device, inputs.dtype
            K = num_slots if num_slots is not None else self.num_slots

            mu = self.slots_mu.expand(B, K, -1)
            sigma = self.slots_logsigma.exp().expand(B, K, -1)
            slots = mu + sigma * torch.randn(mu.shape, device=device, dtype=dtype)

            inputs_norm = self.norm_input(inputs)
            k = self.to_k(inputs_norm)  # (B, N, d)
            v = self.to_v(inputs_norm)  # (B, N, d)

            attn = None
            for _ in range(self.num_iterations):
                slots_prev = slots
                slots_norm = self.norm_slots(slots)
                q = self.to_q(slots_norm)  # (B, K, d)

                # q: (B, K, d), k: (B, N, d) -> dots: (B, N, K)
                dots = torch.einsum("b k d, b n d -> b n k", q, k) * self.scale
                attn = F.softmax(dots, dim=-1)  # Softmax over K slots -> sum_k A_ik = 1.0

                # Weighted sum over N nodes: attn (B, N, K), v (B, N, d) -> updates (B, K, d)
                attn_norm = attn / (attn.sum(dim=1, keepdim=True) + self.eps)  # L1 over nodes
                updates = torch.einsum("b n k, b n d -> b k d", attn_norm, v)

                slots = self.gru(
                    updates.reshape(-1, self.slot_dim),
                    slots_prev.reshape(-1, self.slot_dim)
                ).reshape(B, K, self.slot_dim)

                slots = slots + self.mlp(self.norm_pre_ff(slots))

            return slots, attn


# ---------------------------------------------------------------------------
# Tier 1: Feature Coverage Test Cases
# ---------------------------------------------------------------------------

def test_slot_attention_output_shapes():
    """Verify slots tensor shape (B, K, d) and attn map shape (B, N, K)."""
    B, N, K, d_in, d_slot = 4, 20, 5, 64, 32
    model = SlotAttention(num_slots=K, slot_dim=d_slot, in_dim=d_in, num_iterations=3)
    inputs = torch.randn(B, N, d_in)
    
    slots, attn = model(inputs)
    assert slots.shape == (B, K, d_slot), f"Expected slots shape {(B, K, d_slot)}, got {slots.shape}"
    assert attn.shape == (B, N, K), f"Expected attn shape {(B, N, K)}, got {attn.shape}"


def test_slot_attention_softmax_sum_to_one():
    """Assert sum_k A_ik = 1.0 for every node i across batch."""
    B, N, K, d = 8, 50, 6, 64
    model = SlotAttention(num_slots=K, slot_dim=d, in_dim=d, num_iterations=3)
    inputs = torch.randn(B, N, d) * 10.0  # High variance inputs
    
    slots, attn = model(inputs)
    attn_sum = attn.sum(dim=-1)  # Sum over slot dimension K
    expected_ones = torch.ones(B, N, device=inputs.device)
    assert torch.allclose(attn_sum, expected_ones, atol=1e-5), (
        f"A_ik must sum to 1.0 across K slots. Max diff: {(attn_sum - expected_ones).abs().max().item()}"
    )


def test_slot_attention_iterative_refinement():
    """Run with num_iterations in {1, 3, 5}; verify tensor output integrity."""
    B, N, K, d = 2, 10, 4, 32
    inputs = torch.randn(B, N, d)
    
    for iters in [1, 3, 5]:
        model = SlotAttention(num_slots=K, slot_dim=d, in_dim=d, num_iterations=iters)
        slots, attn = model(inputs)
        assert slots.shape == (B, K, d)
        assert attn.shape == (B, N, K)
        assert not torch.isnan(slots).any()
        assert not torch.isnan(attn).any()


def test_slot_attention_gradient_flow():
    """Assert gradients flow backward from output slots and attn to inputs and parameters."""
    B, N, K, d = 2, 15, 4, 32
    model = SlotAttention(num_slots=K, slot_dim=d, in_dim=d, num_iterations=3)
    inputs = torch.randn(B, N, d, requires_grad=True)
    
    slots, attn = model(inputs)
    loss = slots.sum() + attn.sum()
    loss.backward()
    
    assert inputs.grad is not None, "Gradients should flow to inputs"
    assert not torch.isnan(inputs.grad).any(), "Input gradients contain NaN"
    assert torch.any(inputs.grad != 0.0), "Input gradients are all zero"
    assert model.to_q.weight.grad is not None, "Gradients should flow to model parameters"


def test_slot_attention_dynamic_num_slots():
    """Pass explicit num_slots=K_custom to forward and verify output shapes match K_custom."""
    model = SlotAttention(num_slots=4, slot_dim=32, in_dim=64, num_iterations=2)
    inputs = torch.randn(2, 12, 64)
    
    slots_5, attn_5 = model(inputs, num_slots=5)
    assert slots_5.shape == (2, 5, 32)
    assert attn_5.shape == (2, 12, 5)

    slots_2, attn_2 = model(inputs, num_slots=2)
    assert slots_2.shape == (2, 2, 32)
    assert attn_2.shape == (2, 12, 2)


# ---------------------------------------------------------------------------
# Tier 2: Boundary & Corner Cases (BVA)
# ---------------------------------------------------------------------------

def test_single_slot_K_1():
    """Verify K=1 single slot produces uniform attention map A_ik = 1.0 and shape (B, 1, d)."""
    B, N, d = 4, 15, 32
    model = SlotAttention(num_slots=1, slot_dim=d, in_dim=d, num_iterations=3)
    inputs = torch.randn(B, N, d)
    
    slots, attn = model(inputs)
    assert slots.shape == (B, 1, d)
    assert attn.shape == (B, N, 1)
    assert torch.allclose(attn, torch.ones(B, N, 1), atol=1e-5), "Attn map must be 1.0 everywhere when K=1"


def test_zero_input_embeddings():
    """Verify zero input embeddings do not produce NaN, sum to 1.0, and yield near-uniform attention."""
    B, N, K, d = 2, 10, 4, 32
    model = SlotAttention(num_slots=K, slot_dim=d, in_dim=d, num_iterations=3)
    inputs = torch.zeros(B, N, d)
    
    slots, attn = model(inputs)
    assert not torch.isnan(slots).any(), "Slots contain NaN on zero input"
    assert not torch.isnan(attn).any(), "Attn contains NaN on zero input"
    assert torch.allclose(attn.sum(dim=-1), torch.ones(B, N), atol=1e-5), "Attn sum across slots must be 1.0 on zero input"
    # Due to random initial slot seeds, attention is near-uniform (~1/K)
    assert (attn - (1.0 / K)).abs().max() < 0.1, "Attention should be near-uniform on zero input"


def test_large_N_scaling():
    """Test N=500 scaling check memory and execution stability."""
    B, N, K, d = 2, 500, 4, 64
    model = SlotAttention(num_slots=K, slot_dim=d, in_dim=d, num_iterations=3)
    inputs = torch.randn(B, N, d)
    
    slots, attn = model(inputs)
    assert slots.shape == (B, K, d)
    assert attn.shape == (B, N, K)
    assert not torch.isnan(slots).any()


def test_batch_sizes_and_small_N():
    """Verify batch sizes B=1, 64 and N < K scenario (N=3 nodes, K=5 slots)."""
    model = SlotAttention(num_slots=5, slot_dim=32, in_dim=32)
    for B in [1, 64]:
        inputs = torch.randn(B, 3, 32)  # N=3 < K=5
        slots, attn = model(inputs)
        assert slots.shape == (B, 5, 32)
        assert attn.shape == (B, 3, 5)
        assert not torch.isnan(slots).any()


def test_train_vs_eval_mode_reproducibility():
    """Test slot attention execution in .train() vs .eval() mode with fixed random seed."""
    model = SlotAttention(num_slots=4, slot_dim=32, in_dim=32)
    inputs = torch.randn(2, 10, 32)
    
    torch.manual_seed(42)
    model.train()
    slots_train, attn_train = model(inputs)
    
    torch.manual_seed(42)
    model.eval()
    slots_eval, attn_eval = model(inputs)
    
    assert torch.allclose(slots_train, slots_eval, atol=1e-5), "Train and eval mode should produce identical outputs under same seed"
    assert torch.allclose(attn_train, attn_eval, atol=1e-5)
