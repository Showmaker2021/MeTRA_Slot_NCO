"""
M5 — POMOSlot: POMO + Slot Attention + Metric-Aware Loss

A drop-in extension of rl4co's POMO model that:
  1. Wraps the POMO encoder with SlotInjectingEncoder so that slot context
     is fed into the decoder on every forward pass (not just in a side branch).
  2. Reads slot embeddings from the encoder side-channel after each policy
     forward to compute auxiliary losses (Variants A, C, D).
  3. Adds a slot entropy regulariser to prevent slot collapse.

Ablation Variants (controlled via `metric_variant` arg):
  "none" / "B" : No metric loss — slot learns purely from task (REINFORCE) loss
  "A"           : Reconstruction loss (MSE on slot centroids vs node coords)
  "C"           : Metric loss with Euclidean centroid distance as target
  "D"           : Metric loss with insertion-cost distance (proposed method)
  # "E"         : Future-regret target — reserved, not implemented yet

Augmentation note:
  POMO disables augmentation during training (n_aug=0). Aux losses are only
  computed during training. Therefore slots and d_ins always share batch size B.
  If aux losses are ever extended to val/test, d_ins must be repeated by
  num_augment before passing to metric_loss_fn.

Usage example (Variant D):
    from rl4co.models.zoo.pomo_slot import POMOSlot

    model = POMOSlot(
        env,
        embed_dim=128,
        num_slots=8,
        metric_variant="D",
        alpha_metric=0.1,
        beta_entropy=0.01,
    )
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.zoo.pomo import POMO
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.zoo.am.encoder import AttentionModelEncoder
from rl4co.models.nn.slot_attention import SlotAttention
from rl4co.models.nn.metric_loss import (
    MetricPreservationLoss,
    ProjectionHead,
    SlotEntropyLoss,
)
from rl4co.models.zoo.pomo_slot.policy import SlotInjectingEncoder
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class POMOSlot(POMO):
    """
    POMO extended with Metric-Aware Slot Abstraction.

    Inherits the full POMO training loop (REINFORCE, multi-start).
    Replaces the POMO encoder with a SlotInjectingEncoder so that slot
    context is visible to the decoder on every step. Overrides shared_step
    only to extract d_ins from the batch and add auxiliary losses.

    Args:
        env: RL4CO environment (e.g. CVRP).
        embed_dim (int): Encoder/decoder embedding dimension. Must match
            the AttentionModelPolicy embed_dim. Default: 128.
        num_slots (int): K — number of slot/region embeddings.
        metric_variant (str): Ablation variant — one of "none"/"B", "A", "C", "D".
        alpha_metric (float): Weight for metric preservation/reconstruction loss.
        beta_entropy (float): Weight for slot entropy regulariser.
        slot_iters (int): Number of SlotAttention refinement iterations.
        proj_dim (int): Projection dimension for phi(z_k) in metric loss.
        lambda_init (float): Initial Lagrange multiplier for MetricPreservationLoss.
        lr_dual (float): Learning rate for dual ascent on lambda (log_lambda param group).
        **pomo_kwargs: All remaining kwargs forwarded to POMO base class.
    """

    # "E" excluded: future-regret target not yet implemented
    METRIC_VARIANTS = {"none", "B", "A", "C", "D"}

    def __init__(
        self,
        env: RL4COEnvBase,
        embed_dim: int = 128,
        num_slots: int = 8,
        metric_variant: str = "D",
        alpha_metric: float = 0.1,
        beta_entropy: float = 0.01,
        slot_iters: int = 3,
        proj_dim: int = 64,
        lambda_init: float = 1.0,
        lr_dual: float = 1e-3,
        **pomo_kwargs,
    ) -> None:
        assert metric_variant in self.METRIC_VARIANTS, (
            f"metric_variant must be one of {self.METRIC_VARIANTS}, got '{metric_variant}'"
        )

        # ── Build SlotAttention ───────────────────────────────────────────
        slot_attn = SlotAttention(
            num_slots=num_slots,
            dim=embed_dim,
            iters=slot_iters,
        )

        # ── Build base encoder, wrap it with slot injection ───────────────
        base_encoder = AttentionModelEncoder(
            embed_dim=embed_dim,
            num_heads=pomo_kwargs.pop("num_heads", 8),
            num_layers=pomo_kwargs.pop("num_encoder_layers", 6),
            env_name=env.name,
            normalization=pomo_kwargs.pop("normalization", "instance"),
            feedforward_hidden=pomo_kwargs.pop("feedforward_hidden", 512),
        )
        slot_encoder = SlotInjectingEncoder(base_encoder, slot_attn)

        # ── Build AttentionModelPolicy with our slot-injecting encoder ────
        policy = AttentionModelPolicy(
            encoder=slot_encoder,
            embed_dim=embed_dim,
            env_name=env.name,
            use_graph_context=pomo_kwargs.pop("use_graph_context", False),
        )

        # ── Init POMO (which sets up REINFORCE, shared baseline, etc.) ────
        super().__init__(env, policy=policy, **pomo_kwargs)
        self.save_hyperparameters(logger=False, ignore=["env", "policy"])

        self.embed_dim = embed_dim
        self.num_slots = num_slots
        self.metric_variant = metric_variant
        self.alpha_metric = alpha_metric
        self.beta_entropy = beta_entropy

        # Keep a reference to slot_attn for display in the model summary
        # (it's actually inside policy.encoder, but named here for clarity)
        self.slot_attn = slot_attn

        # ── Auxiliary losses ──────────────────────────────────────────────
        self.slot_entropy_loss = SlotEntropyLoss()

        self.metric_loss_fn: MetricPreservationLoss | None = None
        if metric_variant not in ("none", "B", "A"):
            proj_head = ProjectionHead(
                input_dim=embed_dim,
                proj_dim=proj_dim,
            )
            self.metric_loss_fn = MetricPreservationLoss(
                proj_head=proj_head,
                variant=metric_variant,
                lambda_init=lambda_init,
                lr_dual=lr_dual,
            )

        log.info(
            f"POMOSlot: embed_dim={embed_dim}, K={num_slots}, "
            f"variant={metric_variant}, alpha={alpha_metric}, beta={beta_entropy}"
        )

    # ────────────────────────────────────────────────────────────────────────
    # configure_optimizers: add separate param group for log_lambda (dual ascent)
    # Delegates back to base class to preserve the POMO LR scheduler.
    # ────────────────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        if self.metric_loss_fn is None:
            # No dual parameter — use base class as-is
            return super().configure_optimizers()

        lr_dual = self.hparams.get("lr_dual", 1e-3)
        log_lambda_id = id(self.metric_loss_fn.log_lambda)

        main_params = [p for p in self.parameters() if id(p) != log_lambda_id]
        dual_params = [self.metric_loss_fn.log_lambda]

        param_groups = [
            {"params": main_params},                      # uses base lr + scheduler
            {"params": dual_params, "lr": lr_dual},       # dual ascent, no scheduler
        ]
        # super().configure_optimizers(parameters) builds optimizer+scheduler using
        # RL4COLitModule machinery, so we preserve scheduler type and kwargs exactly.
        return super().configure_optimizers(parameters=param_groups)

    # ────────────────────────────────────────────────────────────────────────
    # shared_step: extract d_ins, run POMO, read slot side-channel, add aux loss
    # ────────────────────────────────────────────────────────────────────────

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        # ── Extract sparse d_ins keys from batch BEFORE passing to POMO ──
        # d_ins_idx / d_ins_val are slot-only keys not understood by CVRPEnv.
        d_ins_idx = None
        d_ins_val = None

        if isinstance(batch, dict):
            d_ins_idx = batch.pop("d_ins_idx", None)
            d_ins_val = batch.pop("d_ins_val", None)
            from tensordict import TensorDict as TD
            B = next(iter(batch.values())).shape[0]
            batch = TD(batch, batch_size=[B])
        elif hasattr(batch, "keys"):
            keys = list(batch.keys())
            if "d_ins_idx" in keys:
                d_ins_idx = batch.get("d_ins_idx")
                d_ins_val = batch.get("d_ins_val")
                batch = batch.exclude("d_ins_idx", "d_ins_val")

        # Move to model device
        device = self.device
        batch = batch.to(device)
        if d_ins_idx is not None:
            d_ins_idx = d_ins_idx.to(device)  # keep as int16 for transfer, cast in loss fn
        if d_ins_val is not None:
            d_ins_val = d_ins_val.to(device)

        # ── Read locs BEFORE super().shared_step() mutates the batch ──────
        # After super() calls env.reset(batch), batch gains 'visited' (shape B,N+1).
        # We cannot call env.reset(batch) again without a shape mismatch.
        # Instead, extract customer locs (excluding depot) directly from batch here.
        # CVRPEnv stores customers at locs[:, 0:N, :] and depot separately.
        # Our dataset stores them as locs (B,N,2) without depot prepended.
        # The encoder sees locs with depot prepended by the env's init_embedding.
        locs_customers: torch.Tensor | None = None
        if self.metric_variant not in ("none", "B"):
            raw_locs = batch.get("locs") if hasattr(batch, "get") else batch["locs"]
            locs_customers = raw_locs.to(device)  # (B, N, 2) — customers only, no depot

        # ── Run standard POMO step ────────────────────────────────────────
        # SlotInjectingEncoder runs inside policy.forward() here.
        # After this call, slots and A_ik are available via side-channel.
        out = super().shared_step(batch, batch_idx, phase, dataloader_idx)

        # ── Skip aux losses during val/test ──────────────────────────────
        if phase != "train" or self.metric_variant in ("none", "B"):
            return out

        # ── Read slot side-channel ────────────────────────────────────────
        # policy.encoder is our SlotInjectingEncoder
        slots = self.policy.encoder.last_slots  # (B, K, d)
        A_ik  = self.policy.encoder.last_A_ik   # (B, N, K)

        if slots is None or A_ik is None:
            log.warning("SlotInjectingEncoder side-channel is None — skipping aux loss.")
            return out

        # ── Compute auxiliary losses ──────────────────────────────────────
        aux_loss = torch.tensor(0.0, device=device)
        log_dict: dict = {}

        # Slot entropy regulariser (all variants except "none")
        ent_loss = self.slot_entropy_loss(A_ik)
        aux_loss = aux_loss + self.beta_entropy * ent_loss
        log_dict["slot_entropy_loss"] = ent_loss.detach()

        # Variant A: reconstruction loss (slot centroids vs node coords)
        if self.metric_variant == "A":
            locs = locs_customers  # (B, N, 2) customers only, read before super() above
            A_norm = A_ik / (A_ik.sum(dim=1, keepdim=True) + 1e-8)
            centroids = torch.einsum("bnk,bnc->bkc", A_norm, locs)
            recon = torch.einsum("bnk,bkc->bnc", A_ik, centroids)
            recon_loss = F.mse_loss(recon, locs)
            aux_loss = aux_loss + self.alpha_metric * recon_loss
            log_dict["recon_loss"] = recon_loss.detach()

        # Variants C / D: metric preservation loss
        elif self.metric_loss_fn is not None:
            locs = locs_customers  # (B, N, 2) customers only, read before super() above
            metric_loss, metric_info = self.metric_loss_fn(
                slots=slots,
                A_ik=A_ik,
                locs=locs,
                d_ins_idx=d_ins_idx,
                d_ins_val=d_ins_val,
            )
            aux_loss = aux_loss + self.alpha_metric * metric_loss
            log_dict.update(metric_info)

        # ── Merge auxiliary loss into policy loss ─────────────────────────
        policy_loss = out.get("loss", None)
        if policy_loss is not None and aux_loss.requires_grad:
            out["loss"] = policy_loss + aux_loss
            log_dict["aux_loss"]    = aux_loss.detach()
            log_dict["policy_loss"] = policy_loss.detach()

        # Log to Lightning
        for k, v in log_dict.items():
            self.log(f"train/{k}", v, prog_bar=False, on_step=True, on_epoch=True)

        return out
