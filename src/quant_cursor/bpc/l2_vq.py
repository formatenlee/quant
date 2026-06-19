"""
欧氏 L2 向量量化 — 独立模块。

标准 VQ-VAE EMA 码本：用 cdist / 加权 L2 距离分配，MSE commitment + STE。
本模块为稳定基座，修改前请做 VQ 健康检查回归。

Public API
----------
- prepare_latent
- coarse_distances
- straight_through_quantize
- residual_metrics
- ema_update / adapt_codebook_step / revitalize_dead_codes
- codebook_diversity_min_distance
- merge_for_decode
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def prepare_latent(z: torch.Tensor) -> Tuple[torch.Tensor, None, torch.Tensor]:
    """L2 VQ 不分离幅度；返回 (z, None, z)。"""
    return z, None, z


def init_codebook(_weight: torch.Tensor) -> None:
    """L2 码本保持均匀初始化，无需额外处理。"""
    return None


def renormalize_codebook(_weight: torch.Tensor, _ema_w: torch.Tensor) -> None:
    """L2 码本不做范数约束。"""
    return None


def coarse_distances(
    z: torch.Tensor,
    codebook: torch.Tensor,
    *,
    codebook_gamma: Optional[torch.Tensor] = None,
    codebook_beta: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Coarse assignment L2 distances; optional per-sample FiLM on codebook."""
    if codebook_gamma is None or codebook_beta is None:
        return torch.cdist(z, codebook)
    z_adj = z - codebook_beta
    scale = 1.0 + codebook_gamma
    zz = (z_adj * z_adj).sum(dim=1, keepdim=True)
    ww = torch.einsum("bd,kd->bk", scale * scale, codebook * codebook)
    cross = torch.einsum("bd,kd->bk", z_adj * scale, codebook)
    return (zz + ww - 2.0 * cross).clamp_min(0.0)


def straight_through_quantize(
    z: torch.Tensor, z_q_coarse: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """STE: forward uses codebook vector; gradients flow through z."""
    z_q = z + (z_q_coarse - z).detach()
    commitment_raw = F.mse_loss(z_q_coarse.detach(), z)
    return z_q, commitment_raw


def residual_metrics(
    z: torch.Tensor,
    z_q: torch.Tensor,
    min_dist: torch.Tensor,
    *,
    dist_is_squared: bool = False,
    z_scale: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    del z_scale
    with torch.no_grad():
        residual = (z - z_q).norm(dim=1)
        dist = min_dist.clamp_min(0.0)
        if dist_is_squared:
            dist = dist.sqrt()
        dist = torch.nan_to_num(dist, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "vq_residual_mean": float(residual.mean()),
            "vq_residual_p95": float(torch.quantile(residual, 0.95)),
            "vq_min_distance_mean": float(dist.mean()),
        }


def codebook_diversity_min_distance(codebook: torch.Tensor) -> torch.Tensor:
    """Penalize L2 codes closer than 0.5 Euclidean distance."""
    dists = torch.cdist(codebook, codebook)
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
        dw = encodings.t() @ z
        ema_w.mul_(decay).add_(dw, alpha=1 - decay)
        codebook_weight.data.copy_(ema_w / cluster_size.unsqueeze(1))


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
            target = z[mask].mean(dim=0)
            codebook_weight[kid].add_(adapt_lr * (target - codebook_weight[kid]))
            ema_w[kid].copy_(codebook_weight[kid])


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
            src = z[pick[i % pick.numel()]].detach()
            codebook_weight.data[code_id] = src
            ema_w[code_id] = src
            ema_cluster_size[code_id] = 1.0


def merge_for_decode(z_q: torch.Tensor, z_scale: Optional[torch.Tensor]) -> torch.Tensor:
    """L2 解码不拼接幅度。"""
    del z_scale
    return z_q
