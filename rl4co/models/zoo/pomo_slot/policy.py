"""
SlotInjectingEncoder and SlotAwarePolicy for POMOSlot.

Instead of overriding ConstructivePolicy.forward() (which would duplicate
multistart/decode/log-likelihood logic), we wrap the base encoder with a
SlotInjectingEncoder that:

  1. Runs the original encoder to get node embeddings (B, N+1, d)
  2. Runs SlotAttention on customer nodes (B, N, d)
  3. Additively injects slot context back into node embeddings
  4. Exposes slots/A_ik via side-channel attributes for aux loss computation

NOTE on augmentation safety:
  POMO's shared_step sets n_aug=0 during training, so augmentation never
  runs before encoder during training. Auxiliary losses are only computed
  during training. Therefore slots and d_ins always share the same batch
  size B (not B*num_augment). If aux losses are ever extended to val/test,
  d_ins must be repeated by num_augment accordingly.

NOTE on injection strength:
  Current injection is additive (weak): hidden += [0_depot | slot_ctx].
  This acts as a routing-aware bias on node representations.
  A stronger alternative (context-embedding injection at each decode step)
  can be explored later if ablation results show insufficient signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SlotInjectingEncoder(nn.Module):
    """
    Encoder wrapper that injects slot context into node embeddings.

    Wraps any ConstructiveEncoder. After encoding, runs SlotAttention on
    customer node embeddings and additively injects the slot context back:

        hidden[:, 1:, :] += A_ik @ slots   (slot_ctx for each customer node)
        hidden[:, 0,  :] unchanged          (depot embedding unchanged)

    Side-channel attributes (populated after each forward pass):
        last_slots: (B, K, d)  — slot embeddings
        last_A_ik:  (B, N, K)  — soft assignment matrix

    Args:
        base_encoder: The original AttentionModelEncoder (or any encoder
            returning (hidden, init_embeds) tuple).
        slot_attn: SlotAttention module. Must accept (B, N, d) and return
            ((B, K, d), (B, N, K)).
    """

    def __init__(self, base_encoder: nn.Module, slot_attn: nn.Module) -> None:
        super().__init__()
        self.base_encoder = base_encoder
        self.slot_attn = slot_attn

        # Side-channel: populated after every forward, read by POMOSlot.shared_step
        self.last_slots: torch.Tensor | None = None
        self.last_A_ik: torch.Tensor | None = None

    def forward(self, td) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            td: TensorDict — environment state after reset (contains locs, depot, etc.)

        Returns:
            hidden:      (B, N+1, d) node embeddings with slot context injected
            init_embeds: (B, N+1, d) original embeddings before transformer layers
                         (unchanged — passed through from base encoder)
        """
        # ── Step 1: Run base encoder ───────────────────────────────────────
        hidden, init_embeds = self.base_encoder(td)  # (B, N+1, d)

        # ── Step 2: Slot Attention on customer nodes only ──────────────────
        node_embs = hidden[:, 1:, :]                  # (B, N, d) — exclude depot
        slots, A_ik = self.slot_attn(node_embs)       # (B, K, d), (B, N, K)

        # ── Step 3: Compute per-node slot context and inject ───────────────
        slot_ctx = torch.bmm(A_ik, slots)             # (B, N, d)
        pad_depot = torch.zeros_like(hidden[:, :1, :])  # (B, 1, d) — depot stays unchanged
        hidden = hidden + torch.cat([pad_depot, slot_ctx], dim=1)  # (B, N+1, d)

        # ── Step 4: Expose via side-channel for aux loss ───────────────────
        self.last_slots = slots   # (B, K, d)
        self.last_A_ik  = A_ik   # (B, N, K)

        return hidden, init_embeds
