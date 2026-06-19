"""
球面余弦向量量化（Cosine VQ）— 独立模块。

将方向量化与幅度分离：VQ 在单位球面上用 1-cosine 距离；幅度 z_scale=||z|| 单独保留。
EMA 更新后投影回单位球面。本模块为稳定基座，修改前请做 VQ 健康检查回归。

Public API
----------
- prepare_direction_scale
- init_unit_codebook / renormalize_codebook
- cosine_distance / coarse_distances
- straight_through_quantize / commitment_loss
- residual_metrics
- ema_update / adapt_codebook_step / revitalize_dead_codes
- codebook_diversity_min_distance
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def prepare_direction_scale(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Conditioned latent → unit direction for VQ, scalar scale, and original z."""
    z_scale = z.norm(dim=1, keepdim=True).clamp_min(1e-6)
    z_dir = F.normalize(z, dim=-1)
    return z_dir, z_scale, z


prepare_latent = prepare_direction_scale


def init_unit_codebook(weight: torch.Tensor) -> None:
    with torch.no_grad():
        weight.copy_(F.normalize(weight, dim=-1))


init_codebook = init_unit_codebook


def renormalize_codebook(weight: torch.Tensor, ema_w: torch.Tensor) -> None:
    with torch.no_grad():
        weight.copy_(F.normalize(weight, dim=-1))
        ema_w.copy_(weight)


def cosine_distance(
    z: torch.Tensor,
    codebook: torch.Tensor,
    *,
    beta: Optional[torch.Tensor] = None,
    gamma_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pairwise 1 - cosine_sim; returns [B, K]."""
    z_adj = z - beta if beta is not None else z
    z_u = F.normalize(z_adj, dim=-1)
    if gamma_scale is None:
        w_u = F.normalize(codebook, dim=-1)
        cos_sim = z_u @ w_u.T
    else:
        w_mod = codebook.unsqueeze(0) * gamma_scale.unsqueeze(1)
        w_u = F.normalize(w_mod, dim=-1)
        cos_sim = (z_u.unsqueeze(1) * w_u).sum(dim=-1)
    return (1.0 - cos_sim.clamp(-1.0, 1.0)).clamp_min(0.0)


def coarse_distances(
    z: torch.Tensor,
    codebook: torch.Tensor,
    *,
    codebook_gamma: Optional[torch.Tensor] = None,
    codebook_beta: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    z_adj = z - codebook_beta if codebook_beta is not None else z
    film_scale = film_scale_for_cosine(codebook_gamma) if codebook_gamma is not None else None
    return cosine_distance(z_adj, codebook, gamma_scale=film_scale)


def film_scale_for_cosine(gamma: torch.Tensor) -> torch.Tensor:
    """
    码本 FiLM 的 (1+gamma) 在 normalize 前使用。

    各维相同的缩放分量在 normalize 后抵消，故减去 per-sample 均值，仅保留
    逐维差异（与 bpc_v3.film 的逐维 bias 配合）。
    """
    centered = gamma - gamma.mean(dim=-1, keepdim=True)
    return 1.0 + centered


def remove_uniform_gamma_component(gamma: torch.Tensor) -> torch.Tensor:
    """Latent FiLM 乘性 gamma：去掉各向同性分量，避免 VQ 前 normalize 抹掉调制。"""
    return gamma - gamma.mean(dim=-1, keepdim=True)


def straight_through_quantize(
    z: torch.Tensor, z_q_coarse: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """STE: forward uses codebook vector; gradients flow through normalized z."""
    z_dir = F.normalize(z, dim=-1)
    z_q = z_dir + (z_q_coarse - z_dir).detach()
    commitment_raw = (1.0 - F.cosine_similarity(z_dir, z_q_coarse.detach(), dim=-1)).mean()
    return z_q, commitment_raw


def residual_metrics(
    z: torch.Tensor,
    z_q: torch.Tensor,
    min_dist: torch.Tensor,
    *,
    dist_is_squared: bool = False,
    z_scale: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    del dist_is_squared
    with torch.no_grad():
        z_u = F.normalize(z, dim=-1)
        q_u = F.normalize(z_q, dim=-1)
        dir_residual = (z_u - q_u).norm(dim=1)
        residual = dir_residual
        if z_scale is not None:
            residual = dir_residual * z_scale.squeeze(-1)
        dist = torch.nan_to_num(min_dist.clamp_min(0.0), nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "vq_residual_mean": float(residual.mean()),
            "vq_residual_p95": float(torch.quantile(residual, 0.95)),
            "vq_min_distance_mean": float(dist.mean()),
            "vq_dir_residual_mean": float(dir_residual.mean()),
        }


def codebook_diversity_min_distance(codebook: torch.Tensor) -> torch.Tensor:
    """Penalize codes closer than 0.5 angular distance on the sphere."""
    w_u = F.normalize(codebook, dim=-1)
    dists = 1.0 - (w_u @ w_u.T).clamp(-1.0, 1.0)
    k = dists.shape[0]
    mask = torch.eye(k, device=dists.device).bool()
    dists = dists.masked_fill(mask, float("inf"))
    return F.relu(0.5 - dists.min(dim=1).values).mean()


def ema_update(
    z: torch.Tensor,
    indices: torch.Tensor,
    *,
    num_codes: int,
    decay: float,
    epsilon: float,
    ema_cluster_size: torch.Tensor,
    ema_w: torch.Tensor,
    codebook_weight: torch.Tensor,
) -> None:
    with torch.no_grad():
        encodings = F.one_hot(indices, num_codes).float()
        new_cluster_size = encodings.sum(dim=0)
        ema_cluster_size.mul_(decay).add_(new_cluster_size, alpha=1 - decay)
        cluster_size = ema_cluster_size + epsilon
        z_ema = F.normalize(z, dim=-1)
        dw = encodings.t() @ z_ema
        ema_w.mul_(decay).add_(dw, alpha=1 - decay)
        codebook_weight.data.copy_(ema_w / cluster_size.unsqueeze(1))
        renormalize_codebook(codebook_weight.data, ema_w)


def adapt_codebook_step(
    z: torch.Tensor,
    coarse_idx: torch.Tensor,
    *,
    adapt_lr: float,
    codebook_weight: torch.Tensor,
    ema_w: torch.Tensor,
) -> None:
    with torch.no_grad():
        for k in coarse_idx.unique():
            mask = coarse_idx == k
            if not mask.any():
                continue
            kid = int(k.item())
            target = F.normalize(z[mask].mean(dim=0), dim=0)
            codebook_weight[kid].add_(adapt_lr * (target - codebook_weight[kid]))
            ema_w[kid].copy_(codebook_weight[kid])
        renormalize_codebook(codebook_weight.data, ema_w)


def revitalize_dead_codes(
    z: torch.Tensor,
    *,
    dead_code_threshold: float,
    epsilon: float,
    ema_cluster_size: torch.Tensor,
    codebook_weight: torch.Tensor,
    ema_w: torch.Tensor,
) -> None:
    if dead_code_threshold <= 0:
        return
    with torch.no_grad():
        total = ema_cluster_size.sum().clamp_min(epsilon)
        usage = ema_cluster_size / total
        dead = usage < dead_code_threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0 or z.size(0) == 0:
            return
        dead_ids = torch.where(dead)[0]
        pick = torch.randperm(z.size(0), device=z.device)[:n_dead]
        for i, code_id in enumerate(dead_ids):
            src = F.normalize(z[pick[i % pick.numel()]].detach(), dim=0)
            codebook_weight.data[code_id] = src
            ema_w[code_id] = src
            ema_cluster_size[code_id] = 1.0


class MagnitudeSplitVQ(nn.Module):
    """
    薄封装：BPC 在 encode 后调用 prepare_direction_scale，VQ 模块只消费 z_dir。
    解码端拼接 [z_q, z_scale]。本类不持有参数，仅文档化数据流契约。
    """

    @staticmethod
    def split(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_dir, z_scale, _ = prepare_direction_scale(z)
        return z_dir, z_scale

    @staticmethod
    def merge_for_decode(z_q: torch.Tensor, z_scale: torch.Tensor) -> torch.Tensor:
        return torch.cat([z_q, z_scale], dim=-1)


def merge_for_decode(z_q: torch.Tensor, z_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    if z_scale is None:
        return z_q
    return torch.cat([z_q, z_scale], dim=-1)
