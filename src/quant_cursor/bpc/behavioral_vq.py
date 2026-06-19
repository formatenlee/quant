"""
BehavioralVQ：EMA 码本量化编排层。

具体距离/量化/EMA 逻辑委托给 vq_backend（cosine_vq 或 l2_vq）。
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_cursor.bpc.vq_backend import VQMode, get_backend, normalize_vq_mode


class BehavioralVQ(nn.Module):
    """
    VQ-VAE 风格 EMA 码本（van den Oord et al., NeurIPS 2017）。
    vq_mode='cosine'：球面余弦量化 + 幅度分离（见 cosine_vq.py）
    vq_mode='l2'：欧氏 L2 量化（见 l2_vq.py）
    """

    def __init__(
        self,
        dim: int,
        num_coarse: int = 128,
        num_fine_per_coarse: int = 16,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        diversity_weight: float = 0.1,
        adapt_lr: float = 1e-5,
        dead_code_threshold: float = 0.0,
        vq_mode: VQMode | str | None = None,
        use_fine_vq: bool = False,
        *,
        use_cosine_vq: bool | None = None,
        use_normalized_vq: bool | None = None,
    ):
        super().__init__()
        self.vq_mode: VQMode = normalize_vq_mode(
            vq_mode,
            use_cosine_vq=use_cosine_vq,
            use_normalized_vq=use_normalized_vq,
        )
        self._backend = get_backend(self.vq_mode)
        self.use_cosine_vq = self.vq_mode == "cosine"
        self.use_normalized_vq = self.use_cosine_vq  # backward compat

        self.dim = dim
        self.num_coarse = num_coarse
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.diversity_weight = diversity_weight
        self.adapt_lr = adapt_lr
        self.dead_code_threshold = dead_code_threshold
        self.use_fine_vq = use_fine_vq

        self.coarse_embed = nn.Embedding(num_coarse, dim)
        self.coarse_embed.weight.data.uniform_(-1 / num_coarse, 1 / num_coarse)
        self._backend.init_codebook(self.coarse_embed.weight.data)

        self.num_fine_per_coarse = num_fine_per_coarse
        self.num_fine_total = num_coarse * num_fine_per_coarse
        if use_fine_vq:
            self.fine_embed = nn.Embedding(self.num_fine_total, dim)
            self.fine_embed.weight.data.uniform_(-1 / self.num_fine_total, 1 / self.num_fine_total)
            self.fine_mix_logit = nn.Parameter(torch.tensor(-1.5))
        else:
            self.fine_embed = None
            self.fine_mix_logit = None

        self.register_buffer("ema_cluster_size", torch.zeros(num_coarse))
        self.register_buffer("ema_w", self.coarse_embed.weight.data.clone())
        self.register_buffer("adaptation_steps", torch.tensor(0, dtype=torch.long))

    def _residual_metrics(
        self,
        z: torch.Tensor,
        z_q: torch.Tensor,
        min_dist: torch.Tensor,
        *,
        dist_is_squared: bool = False,
        z_scale: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        return self._backend.residual_metrics(
            z,
            z_q,
            min_dist,
            dist_is_squared=dist_is_squared,
            z_scale=z_scale,
        )

    def _adapt_codebook(self, z: torch.Tensor, coarse_idx: torch.Tensor) -> None:
        self._backend.adapt_codebook_step(
            z,
            coarse_idx,
            adapt_lr=self.adapt_lr,
            codebook_weight=self.coarse_embed.weight,
            ema_w=self.ema_w,
        )
        self.adaptation_steps += 1

    def _coarse_distances(
        self,
        z: torch.Tensor,
        codebook_gamma: Optional[torch.Tensor] = None,
        codebook_beta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self._backend.coarse_distances(
            z,
            self.coarse_embed.weight,
            codebook_gamma=codebook_gamma,
            codebook_beta=codebook_beta,
        )

    def _hierarchical_fine_quantize(
        self, z: torch.Tensor, z_q_coarse: torch.Tensor, coarse_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_residual = z - z_q_coarse.detach()
        fine_bank = self.fine_embed.weight.view(self.num_coarse, self.num_fine_per_coarse, self.dim)
        local_codes = fine_bank[coarse_idx]
        fine_dist = ((z_residual.unsqueeze(1) - local_codes) ** 2).sum(dim=-1)
        fine_local = fine_dist.argmin(dim=1)
        z_q_fine = local_codes[torch.arange(z.size(0), device=z.device), fine_local]
        fine_idx = coarse_idx * self.num_fine_per_coarse + fine_local
        z_q_total = z_q_coarse + z_q_fine
        return z_q_total, z_q_fine, fine_idx

    def _codebook_diversity_loss(self) -> torch.Tensor:
        return self._backend.codebook_diversity_min_distance(self.coarse_embed.weight)

    def forward(
        self,
        z: torch.Tensor,
        use_fine: bool = False,
        *,
        adapt: bool = False,
        codebook_gamma: Optional[torch.Tensor] = None,
        codebook_beta: Optional[torch.Tensor] = None,
        z_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, Dict[str, float]]:
        distances = self._coarse_distances(z, codebook_gamma, codebook_beta)
        min_dist, coarse_idx = distances.min(dim=1)
        z_q_coarse = self.coarse_embed(coarse_idx)

        if self.training:
            self._update_ema(z, coarse_idx)
        elif adapt:
            self._adapt_codebook(z, coarse_idx)

        z_q, commitment_raw = self._backend.straight_through_quantize(z, z_q_coarse)
        vq_loss = self.commitment_cost * commitment_raw

        if self.training:
            vq_loss = vq_loss + 0.05 * self._codebook_diversity_loss()

        with torch.no_grad():
            encodings = F.one_hot(coarse_idx, self.num_coarse).float()
            avg_probs = encodings.mean(dim=0)
            entropy = -(avg_probs * (avg_probs + self.epsilon).log()).sum()
            perplexity = torch.exp(entropy)
            unique_tokens = coarse_idx.unique().numel()
            usage_rate = unique_tokens / self.num_coarse

        diversity_loss = torch.tensor(0.0, device=z.device)
        if self.training and self.diversity_weight > 0:
            avg_probs = F.one_hot(coarse_idx, self.num_coarse).float().mean(dim=0)
            entropy = -(avg_probs * (avg_probs + self.epsilon).log()).sum()
            diversity_loss = (math.log(self.num_coarse) - entropy) * self.diversity_weight
            vq_loss = vq_loss + diversity_loss

        metrics = {
            "vq_usage_rate": float(usage_rate),
            "vq_unique_tokens": float(unique_tokens),
            "vq_perplexity": float(perplexity),
            "loss_diversity": float(diversity_loss.detach()) if isinstance(diversity_loss, torch.Tensor) else 0.0,
            "vq_commitment_raw": float(commitment_raw.detach()),
            **self._residual_metrics(
                z,
                z_q_coarse,
                min_dist,
                dist_is_squared=codebook_gamma is not None and not self.use_cosine_vq,
                z_scale=z_scale,
            ),
        }

        fine_idx = None
        if use_fine and self.fine_embed is not None:
            z_q_total, z_q_fine, fine_idx = self._hierarchical_fine_quantize(z, z_q_coarse, coarse_idx)
            fine_mix = torch.sigmoid(self.fine_mix_logit)
            z_q_mixed = z_q_coarse + fine_mix * z_q_fine
            z_q = z + (z_q_mixed - z).detach()
            z_residual = z - z_q_coarse.detach()
            vq_loss = vq_loss + fine_mix * self.commitment_cost * F.mse_loss(z_q_fine.detach(), z_residual)

            if self.training:
                with torch.no_grad():
                    fine_bank = self.fine_embed.weight.view(self.num_coarse, self.num_fine_per_coarse, self.dim)
                    local_codes_batch = fine_bank[coarse_idx]
                    fine_dist_batch = ((z_residual.unsqueeze(1) - local_codes_batch) ** 2).sum(dim=-1)
                    fine_probs = F.softmax(-fine_dist_batch, dim=1)
                    fine_entropy = -(fine_probs * (fine_probs + 1e-8).log()).sum(dim=1).mean()
                vq_loss = vq_loss - 0.01 * fine_entropy

            with torch.no_grad():
                fine_unique = fine_idx.unique().numel()
                metrics["vq_fine_usage_rate"] = fine_unique / self.num_fine_total
                metrics["vq_fine_unique_tokens"] = float(fine_unique)
                metrics["vq_fine_mix"] = float(fine_mix)

        return z_q, coarse_idx, fine_idx, vq_loss, metrics

    def _update_ema(self, z: torch.Tensor, indices: torch.Tensor) -> None:
        self._backend.ema_update(
            z,
            indices,
            num_codes=self.num_coarse,
            decay=self.decay,
            epsilon=self.epsilon,
            ema_cluster_size=self.ema_cluster_size,
            ema_w=self.ema_w,
            codebook_weight=self.coarse_embed.weight,
        )
        self._backend.renormalize_codebook(self.coarse_embed.weight.data, self.ema_w)
        self._backend.revitalize_dead_codes(
            z,
            dead_code_threshold=self.dead_code_threshold,
            epsilon=self.epsilon,
            ema_cluster_size=self.ema_cluster_size,
            codebook_weight=self.coarse_embed.weight,
            ema_w=self.ema_w,
        )
