"""
BPC-v2: 行为本体向量量化（接入预计算特征）。

多尺度：day + week（由预计算特征接入）。无分钟/Tick 数据时禁用 1min 尺度。
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from quant_cursor.bpc.behavior_features import (
    BEHAVIOR_AGENT_NAMES,
    BEHAVIOR_LOGITS_DIM,
    CORE_AGENTS,
    EXTENDED_AGENTS,
    agent_logit_slice,
    compute_behavior_proxies,
    compute_behavior_proxies_stacked,
    transform_proxies_for_labeling,
)

logger = logging.getLogger(__name__)


# =============================================================================
# 0. 配置与尺度注册中心
# =============================================================================
@dataclass
class ScaleConfig:
    name: str
    freq: str
    lookback_window: int
    feature_groups: List[str]
    encoder_dim: int = 128
    enabled: bool = True
    is_tick: bool = False


class ScaleRegistry:
    def __init__(self):
        self._scales: Dict[str, ScaleConfig] = {}

    def register(self, cfg: ScaleConfig):
        self._scales[cfg.name] = cfg

    def get_enabled(self) -> List[ScaleConfig]:
        return [c for c in self._scales.values() if c.enabled]

    def get(self, name: str) -> Optional[ScaleConfig]:
        return self._scales.get(name)


DEFAULT_DAY_LOOKBACK = 40
DEFAULT_WEEK_LOOKBACK = 24


def build_scale_registry(
    day_lookback: int = DEFAULT_DAY_LOOKBACK,
    week_lookback: int = DEFAULT_WEEK_LOOKBACK,
) -> ScaleRegistry:
    registry = ScaleRegistry()
    registry.register(
        ScaleConfig(
            name="day",
            freq="day",
            lookback_window=day_lookback,
            feature_groups=["precomputed_features"],
        )
    )
    registry.register(
        ScaleConfig(
            name="week",
            freq="week",
            lookback_window=week_lookback,
            feature_groups=["precomputed_features"],
        )
    )
    return registry


DEFAULT_REGISTRY = build_scale_registry()


def compute_raw_feat_dim(cfg: Optional[ScaleConfig]) -> int:
    if cfg is None:
        return 24
    # 依据尺度名称返回预计算特征维度
    if cfg.name == "day":
        return 24
    elif cfg.name == "week":
        return 7
    return 24


# =============================================================================
# 2. 精简特征提取器：只做归一化 + 投影
# =============================================================================
class PrecomputedFeatureComposer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, timestamps=None, return_raw: bool = False):
        norm = self.norm(x)
        proj = self.proj(norm)
        if return_raw:
            return proj, x
        return proj


# =============================================================================
# 3. 编码器、融合、VQ、解码器、损失
# =============================================================================
class ScaleEncoder(nn.Module):
    def __init__(self, cfg: ScaleConfig, feat_dim: int = 64):
        super().__init__()
        self.cfg = cfg
        in_dim = compute_raw_feat_dim(cfg)
        self.composer = PrecomputedFeatureComposer(in_dim, feat_dim)
        self.aggregator = nn.Sequential(
            nn.Linear(feat_dim, cfg.encoder_dim),
            nn.LayerNorm(cfg.encoder_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor, timestamps=None) -> Tuple[torch.Tensor, torch.Tensor]:
        proj, raw = self.composer(x, timestamps, return_raw=True)
        return self.aggregator(proj), raw


class MultiScaleFusion(nn.Module):
    def __init__(self, registry: ScaleRegistry, unified_dim: int = 128):
        super().__init__()
        self.registry = registry
        self.encoders = nn.ModuleDict()
        for cfg in registry.get_enabled():
            self.encoders[cfg.name] = ScaleEncoder(cfg)
            setattr(self, f"proj_{cfg.name}", nn.Linear(cfg.encoder_dim, unified_dim))
        self.scale_weights = nn.Parameter(torch.ones(len(registry.get_enabled())))

    def forward(self, batch: Dict[str, torch.Tensor]):
        scale_feats = {}
        raw_feats = {}
        for cfg in self.registry.get_enabled():
            name = cfg.name
            feat_key = f"{name}_features"
            if feat_key in batch:
                x = batch[feat_key]          # 预计算特征，形状 [B, in_dim]
            elif name in batch:
                x = batch[name]              # 回退到原始窗口（极少用）
            else:
                continue
            enc = self.encoders[name]
            h, raw = enc(x)
            proj = getattr(self, f"proj_{name}")
            scale_feats[name] = proj(h)
            raw_feats[name] = raw

        if not scale_feats:
            return None, {}, {}

        weights = F.softmax(self.scale_weights[:len(scale_feats)], dim=0)
        fused = sum(scale_feats[name] * w for name, w in zip(scale_feats.keys(), weights))
        return fused, scale_feats, raw_feats


class BehavioralVQ(nn.Module):
    """
    VQ-VAE 风格 EMA 码本（van den Oord et al., NeurIPS 2017）。
    仅 commitment loss 驱动 encoder；码本由 EMA 更新。
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
    ):
        super().__init__()
        self.dim = dim
        self.num_coarse = num_coarse
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.diversity_weight = diversity_weight
        self.adapt_lr = adapt_lr
        self.dead_code_threshold = dead_code_threshold

        self.coarse_embed = nn.Embedding(num_coarse, dim)
        self.coarse_embed.weight.data.uniform_(-1 / num_coarse, 1 / num_coarse)

        self.num_fine_per_coarse = num_fine_per_coarse
        self.num_fine_total = num_coarse * num_fine_per_coarse
        self.fine_embed = nn.Embedding(self.num_fine_total, dim)
        self.fine_embed.weight.data.uniform_(-1 / self.num_fine_total, 1 / self.num_fine_total)

        self.register_buffer("ema_cluster_size", torch.zeros(num_coarse))
        self.register_buffer("ema_w", self.coarse_embed.weight.data.clone())
        self.register_buffer("adaptation_steps", torch.tensor(0, dtype=torch.long))
        # Learnable fine residual mix; sigmoid init ~0.18 keeps coarse semantics primary.
        self.fine_mix_logit = nn.Parameter(torch.tensor(-1.5))

    def _residual_metrics(self, z: torch.Tensor, z_q: torch.Tensor, min_dist: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            residual = (z - z_q).norm(dim=1)
            return {
                "vq_residual_mean": float(residual.mean()),
                "vq_residual_p95": float(torch.quantile(residual, 0.95)),
                "vq_min_distance_mean": float(min_dist.mean()),
            }

    def _adapt_codebook(self, z: torch.Tensor, coarse_idx: torch.Tensor) -> None:
        """验证/推理期在线码本微调：编码器冻结，码本向 batch 质心缓慢漂移。"""
        with torch.no_grad():
            for k in coarse_idx.unique():
                mask = coarse_idx == k
                if not mask.any():
                    continue
                kid = int(k.item())
                target = z[mask].mean(dim=0)
                self.coarse_embed.weight[kid].add_(self.adapt_lr * (target - self.coarse_embed.weight[kid]))
                self.ema_w[kid].copy_(self.coarse_embed.weight[kid])
            self.adaptation_steps += 1

    def _coarse_distances(
        self,
        z: torch.Tensor,
        codebook_gamma: Optional[torch.Tensor] = None,
        codebook_beta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Coarse assignment distances; optional per-sample FiLM on shared codebook.

        Uses expanded L2 (no materialized [B, K, D] codebook) when FiLM is active.
        """
        W = self.coarse_embed.weight
        if codebook_gamma is None or codebook_beta is None:
            return torch.cdist(z, W)
        z_adj = z - codebook_beta
        scale = 1.0 + codebook_gamma
        z_scaled = z_adj * scale
        # 修正：zz 同步乘 scale^2，保持 L2 距离度量一致性
        zz = (z_scaled * z_scaled).sum(dim=1, keepdim=True)
        ww = torch.einsum("bd,kd->bk", scale * scale, W * W)
        cross = torch.einsum("bd,kd->bk", z_scaled, W)
        return zz + ww - 2.0 * cross

    def _hierarchical_fine_quantize(
        self, z: torch.Tensor, z_q_coarse: torch.Tensor, coarse_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Residual fine VQ within each coarse cell (true hierarchical VQ)."""
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
        """鼓励码本向量间保持最小距离，防止训练集上过度细分导致验证集泛化崩溃。"""
        W = self.coarse_embed.weight
        dists = torch.cdist(W, W)
        mask = torch.eye(self.num_coarse, device=W.device).bool()
        dists = dists.masked_fill(mask, float('inf'))
        min_dist = dists.min(dim=1).values
        return F.relu(0.5 - min_dist).mean()

    def forward(
        self,
        z: torch.Tensor,
        use_fine: bool = False,
        *,
        adapt: bool = False,
        codebook_gamma: Optional[torch.Tensor] = None,
        codebook_beta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, Dict[str, float]]:
        distances = self._coarse_distances(z, codebook_gamma, codebook_beta)
        min_dist, coarse_idx = distances.min(dim=1)
        z_q_coarse = self.coarse_embed(coarse_idx)

        if self.training:
            self._update_ema(z, coarse_idx)
        elif adapt:
            self._adapt_codebook(z, coarse_idx)

        z_q = z + (z_q_coarse - z).detach()
        vq_loss = self.commitment_cost * F.mse_loss(z_q_coarse.detach(), z)

        # 码本多样性约束：防止训练集上过度细分导致验证集泛化崩溃
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
            **self._residual_metrics(z, z_q_coarse, min_dist.sqrt() if codebook_gamma is not None else min_dist),
        }

        fine_idx = None
        if use_fine:
            z_q_total, z_q_fine, fine_idx = self._hierarchical_fine_quantize(z, z_q_coarse, coarse_idx)
            fine_mix = torch.sigmoid(self.fine_mix_logit)
            z_q_mixed = z_q_coarse + fine_mix * z_q_fine
            z_q = z + (z_q_mixed - z).detach()
            z_residual = z - z_q_coarse.detach()
            vq_loss = vq_loss + fine_mix * self.commitment_cost * F.mse_loss(z_q_fine.detach(), z_residual)
            
            # Fine 层熵正则化：鼓励跨样本泛化，防止验证集上使用率腰斩
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

    def _revitalize_dead_codes(self, z: torch.Tensor) -> None:
        """将长期未使用的码本向量重置为当前 batch 随机样本，缓解码本坍缩。"""
        if self.dead_code_threshold <= 0:
            return
        total = self.ema_cluster_size.sum().clamp_min(self.epsilon)
        usage = self.ema_cluster_size / total
        dead = usage < self.dead_code_threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0 or z.size(0) == 0:
            return
        dead_ids = torch.where(dead)[0]
        pick = torch.randperm(z.size(0), device=z.device)[:n_dead]
        for i, code_id in enumerate(dead_ids):
            src = z[pick[i % pick.numel()]].detach()
            self.coarse_embed.weight.data[code_id] = src
            self.ema_w[code_id] = src
            self.ema_cluster_size[code_id] = 1.0

    def _update_ema(self, z: torch.Tensor, indices: torch.Tensor):
        """标准 VQ-VAE EMA；死码复活由 dead_code_threshold>0 时可选启用。"""
        with torch.no_grad():
            encodings = F.one_hot(indices, self.num_coarse).float()
            new_cluster_size = encodings.sum(dim=0)
            self.ema_cluster_size.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)
            cluster_size = self.ema_cluster_size + self.epsilon
            dw = encodings.t() @ z
            self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            self.coarse_embed.weight.data.copy_(self.ema_w / cluster_size.unsqueeze(1))
            self._revitalize_dead_codes(z)


class CausalDecoder(nn.Module):
    def __init__(self, dim: int, out_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, out_features),
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.net(z_q)


class BehavioralPurityLoss(nn.Module):
    """
    行为纯度损失：8 个统计结构代理 × 3 档，共 24 维 logits。
    骨架：vol / attack / amount；扩展：路径、量价对称、波动结构、价格动力学、参与结构。
    仅描述可观测的市场动作结构，不做心理动因推断。
    """

    CORE_AGENTS = CORE_AGENTS  # from behavior_features
    EXTENDED_AGENTS = EXTENDED_AGENTS

    def __init__(
        self,
        unified_dim: int = 128,
        primary_scale: str = "day",
        purity_weight: float = 0.5,
        extended_purity_weight: float = 0.15,
        iso_weight: float = 0.05,
        labeling_mode: str = "global",
        num_symbols: int = 10000,
    ):
        super().__init__()
        self.primary_scale = primary_scale
        self.purity_weight = purity_weight
        self.extended_purity_weight = extended_purity_weight
        self.iso_weight = iso_weight
        self.labeling_mode = labeling_mode  # "global" | "per_stock" | "batch"
        self.num_symbols = num_symbols
        self.behavior_proj = nn.Sequential(
            nn.Linear(unified_dim, 96),
            nn.GELU(),
            nn.Linear(96, BEHAVIOR_LOGITS_DIM),
        )
        for name in BEHAVIOR_AGENT_NAMES:
            self.register_buffer(f"{name}_bounds", torch.zeros(2))
        self.register_buffer("thresholds_ready", torch.tensor(False))

        # Per-symbol precomputed quantiles (stable across epochs)
        # Shape: [num_symbols, 8 agents, 2 bounds (33%, 66%)]
        self.register_buffer(
            "per_symbol_bounds", torch.zeros(num_symbols, len(BEHAVIOR_AGENT_NAMES), 2)
        )
        self.register_buffer("per_symbol_ready", torch.zeros(num_symbols, dtype=torch.bool))
        self.register_buffer("per_symbol_counts", torch.zeros(num_symbols, dtype=torch.long))
        # Bayesian-style prior mass for per-stock vs global threshold blending (learnable).
        self.per_symbol_blend_log = nn.Parameter(torch.tensor(4.0))

    def set_thresholds(self, bounds: dict[str, torch.Tensor]) -> None:
        for name in BEHAVIOR_AGENT_NAMES:
            if name not in bounds:
                continue
            buf = getattr(self, f"{name}_bounds")
            buf.copy_(bounds[name].flatten()[:2].to(buf.device))
        self.thresholds_ready.fill_(True)

    def set_per_symbol_thresholds(
        self,
        symbol_bounds: dict[int, torch.Tensor],
        symbol_counts: Optional[dict[int, int]] = None,
    ) -> None:
        """Store precomputed per-symbol 33%/66% quantiles.

        symbol_bounds[sid] is a tensor of shape [8, 2] or [num_agents, 2].
        """
        for sid, b in symbol_bounds.items():
            if 0 <= sid < self.per_symbol_bounds.shape[0]:
                self.per_symbol_bounds[sid] = b.to(self.per_symbol_bounds.device)
                self.per_symbol_ready[sid] = True
        if symbol_counts:
            for sid, cnt in symbol_counts.items():
                if 0 <= sid < self.per_symbol_counts.shape[0]:
                    self.per_symbol_counts[sid] = int(cnt)

    def _agent_weight(self, name: str) -> float:
        if name in self.CORE_AGENTS:
            return self.purity_weight
        return self.extended_purity_weight

    def _bucketize_3class(self, values: torch.Tensor, inner: torch.Tensor) -> torch.Tensor:
        """Vectorized 3-bin labels. values [B], inner [B, 2] -> one_hot [B, 3]."""
        inner = torch.maximum(inner, inner.cummax(dim=-1).values)
        bins = (values.unsqueeze(-1) > inner).long().sum(dim=-1).clamp(max=2)
        return F.one_hot(bins, 3).float()

    def _labels_from_bounds(self, values: torch.Tensor, bounds: torch.Tensor, n_bins: int = 3) -> torch.Tensor:
        if not self.thresholds_ready.item():
            return self._labels_batch_quantile(values, n_bins)
        inner = bounds.to(values.device).unsqueeze(0).expand(values.shape[0], -1)
        # 若全局阈值过于接近（分布退化），回退到 batch 自适应
        if (inner[:, 1] - inner[:, 0]).mean() < 1e-6:
            return self._labels_batch_quantile(values, n_bins)
        return self._bucketize_3class(values, inner)

    def _labels_batch_quantile(self, values: torch.Tensor, n_bins: int = 3) -> torch.Tensor:
        if values.numel() == 0:
            return torch.zeros(0, n_bins, device=values.device)
        if (values.max() - values.min()).abs() < 1e-8:
            mid = torch.full((values.shape[0],), n_bins // 2, dtype=torch.long, device=values.device)
            return F.one_hot(mid, n_bins).float()
        boundaries = torch.quantile(values, torch.linspace(0, 1, n_bins + 1, device=values.device))
        inner = boundaries[1:-1]
        inner = torch.maximum(inner, inner.cummax(0).values)
        # 若 batch 内分位数过于接近，合并中间档避免标签噪声
        if inner.numel() > 1 and (inner[-1] - inner[0]).abs() < 1e-6:
            mid = torch.full((values.shape[0],), n_bins // 2, dtype=torch.long, device=values.device)
            return F.one_hot(mid, n_bins).float()
        inner = inner.unsqueeze(0).expand(values.shape[0], -1)
        return self._bucketize_3class(values, inner)

    def _labels_all_agents(
        self, proxy_mat: torch.Tensor, stock_ids: torch.Tensor, n_bins: int = 3
    ) -> torch.Tensor:
        """All 8 agents at once: proxy_mat [B, 8] -> labels [B, 8, 3]."""
        device = proxy_mat.device
        bsz, n_agents = proxy_mat.shape
        sids = stock_ids.to(device).long().clamp(0, self.per_symbol_bounds.shape[0] - 1)

        per_bounds = self.per_symbol_bounds[sids]  # [B, 8, 2]
        ready = self.per_symbol_ready[sids].float().unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

        global_bounds = torch.stack(
            [getattr(self, f"{name}_bounds") for name in BEHAVIOR_AGENT_NAMES], dim=0
        ).to(device)  # [8, 2]
        global_expanded = global_bounds.unsqueeze(0).expand(bsz, -1, -1)  # [B, 8, 2]
        counts = self.per_symbol_counts[sids].float().unsqueeze(-1).unsqueeze(-1)
        blend_mass = F.softplus(self.per_symbol_blend_log)
        conf = ready * (counts / (counts + blend_mass))
        bounds = conf * per_bounds + (1.0 - conf) * global_expanded

        # 对退化分布进行保护：若上下界过于接近，回退到 batch 内分位数
        degenerate = (bounds[:, :, 1] - bounds[:, :, 0]).abs() < 1e-6  # [B, 8]
        if degenerate.any():
            for j in range(n_agents):
                mask = degenerate[:, j]
                if mask.any():
                    vals = proxy_mat[mask, j]
                    bounds[mask, j] = torch.quantile(vals, torch.tensor([1/3, 2/3], device=device))

        inner = torch.maximum(bounds, bounds.cummax(dim=-1).values)
        bins = (proxy_mat.unsqueeze(-1) > inner).long().sum(dim=-1).clamp(max=2)  # [B, 8]
        return F.one_hot(bins, n_bins).float()

    def forward(
        self,
        z_q: torch.Tensor,
        scale_feats: Dict[str, torch.Tensor],
        raw_batch: Dict[str, torch.Tensor],
        stock_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}

        if len(scale_feats) >= 2:
            vals = list(scale_feats.values())
            iso_loss = 0.0
            iso_count = 0
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    iso_loss += F.mse_loss(vals[i], vals[j])
                    iso_count += 1
            losses["iso"] = (iso_loss / iso_count) * self.iso_weight

        ps = self.primary_scale
        if ps not in raw_batch:
            return losses

        token_logits = self.behavior_proj(z_q)

        if "behavior_proxies" in raw_batch:
            proxy_mat = raw_batch["behavior_proxies"]
            if proxy_mat.dim() == 1:
                proxy_mat = proxy_mat.unsqueeze(0)
        else:
            proxy_mat = compute_behavior_proxies_stacked(raw_batch[ps])

        proxy_mat = transform_proxies_for_labeling(proxy_mat)

        loss_key_map = {
            "vol": "purity_vol",
            "attack": "purity_attack",
            "amount": "purity_amount",
            "path_structure": "purity_path",
            "vp_symmetry": "purity_vp_sym",
            "vol_structure": "purity_vol_struct",
            "price_dynamics": "purity_price_dyn",
            "participation": "purity_participation",
        }

        if self.labeling_mode == "per_stock" and stock_ids is not None:
            all_labels = self._labels_all_agents(proxy_mat, stock_ids)
            for i, name in enumerate(BEHAVIOR_AGENT_NAMES):
                label = all_labels[:, i, :]
                sl = agent_logit_slice(i)
                w = self._agent_weight(name)
                losses[loss_key_map[name]] = (
                    F.kl_div(F.log_softmax(token_logits[:, sl], dim=1), label, reduction="batchmean") * w
                )
        else:
            for i, name in enumerate(BEHAVIOR_AGENT_NAMES):
                bounds = getattr(self, f"{name}_bounds")
                label = self._labels_from_bounds(proxy_mat[:, i], bounds)
                sl = agent_logit_slice(i)
                w = self._agent_weight(name)
                losses[loss_key_map[name]] = (
                    F.kl_div(F.log_softmax(token_logits[:, sl], dim=1), label, reduction="batchmean") * w
                )

        return losses


class SymbolTimeFiLM(nn.Module):
    """
    Symbol-specific + cyclical-time modulation on encoder latent and shared VQ codebook.

    Time features use only sin/cos of week-day and trading-year phase (no calendar month/
    quarter) to avoid binding tokens to macro regimes. Modulation strengths and codebook
    gate are learnable (softplus / sigmoid), not hand-tuned constants.
    """

    def __init__(
        self,
        dim: int,
        num_symbols: int = 10000,
        time_emb_dim: int = 16,
        enable_codebook_film: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.enable_codebook_film = enable_codebook_film
        self.symbol_embed = nn.Embedding(num_symbols, time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(4, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.film = nn.Sequential(
            nn.Linear(2 * time_emb_dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim * 2),
        )
        self.latent_scale_log = nn.Parameter(torch.tensor(0.0))
        if enable_codebook_film:
            self.codebook_film = nn.Sequential(
                nn.Linear(2 * time_emb_dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim * 2),
            )
            self.codebook_scale_log = nn.Parameter(torch.tensor(-1.0))
            # 调整初始化：从 -2.0 改为 0.0，使初始门控约 0.5，打破冻结陷阱
            self.codebook_gate_logit = nn.Parameter(torch.tensor(0.0))
        else:
            self.codebook_film = None
            self.codebook_scale_log = None
            self.codebook_gate_logit = None
        self._last_stats: dict[str, float] = {}

    @staticmethod
    def _cyclical_time_features(ts: torch.Tensor) -> torch.Tensor:
        """Pure periodic features: week cycle + trading-year cycle (no absolute calendar)."""
        dow_phase = 2 * math.pi * (ts % 7) / 7.0
        year_phase = 2 * math.pi * (ts % 252) / 252.0
        return torch.stack(
            [
                torch.sin(dow_phase),
                torch.cos(dow_phase),
                torch.sin(year_phase),
                torch.cos(year_phase),
            ],
            dim=-1,
        )

    def _condition_vector(
        self,
        stock_ids: Optional[torch.Tensor],
        timestamps: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if stock_ids is None and timestamps is None:
            return None
        sym_emb = torch.zeros(batch_size, self.symbol_embed.embedding_dim, device=device)
        if stock_ids is not None:
            sid = stock_ids.to(device).clamp(0, self.symbol_embed.num_embeddings - 1)
            sym_emb = self.symbol_embed(sid)
        time_emb = torch.zeros(batch_size, self.symbol_embed.embedding_dim, device=device)
        if timestamps is not None:
            ts = timestamps.to(device).float()
            time_emb = self.time_mlp(self._cyclical_time_features(ts))
        return torch.cat([sym_emb, time_emb], dim=-1)

    def _record_stats(self, prefix: str, gamma: torch.Tensor, scale: torch.Tensor, gate: float = 1.0) -> None:
        with torch.no_grad():
            self._last_stats[f"film_{prefix}_gamma_abs"] = float((gamma.abs() * scale * gate).mean())
            self._last_stats[f"film_{prefix}_scale"] = float(scale)
            if prefix == "codebook":
                self._last_stats["film_codebook_gate"] = gate

    def modulation_stats(self) -> dict[str, float]:
        return dict(self._last_stats)

    def forward(
        self, z: torch.Tensor, stock_ids: Optional[torch.Tensor], timestamps: Optional[torch.Tensor]
    ) -> torch.Tensor:
        cond = self._condition_vector(stock_ids, timestamps, z.shape[0], z.device)
        if cond is None:
            return z
        gamma_beta = self.film(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        latent_scale = F.softplus(self.latent_scale_log)
        gamma = torch.tanh(gamma) * latent_scale
        self._record_stats("latent", gamma, latent_scale)
        return z * (1 + gamma) + beta

    def codebook_modulation(
        self,
        stock_ids: Optional[torch.Tensor],
        timestamps: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Per-sample FiLM on shared coarse codebook vectors (gated, learnable strength)."""
        if not self.enable_codebook_film or self.codebook_film is None:
            return None, None
        cond = self._condition_vector(stock_ids, timestamps, batch_size, device)
        if cond is None:
            return None, None
        gamma_beta = self.codebook_film(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        cb_scale = F.softplus(self.codebook_scale_log)
        gate = float(torch.sigmoid(self.codebook_gate_logit).item())
        gamma = torch.tanh(gamma) * cb_scale * torch.sigmoid(self.codebook_gate_logit)
        beta = beta * torch.sigmoid(self.codebook_gate_logit)
        self._record_stats("codebook", gamma, cb_scale, gate)
        return gamma, beta


class AdaptiveTaskBalancer(nn.Module):
    """修正后的不确定性加权：只使用精度加权，不加对数方差项，防止负损失。"""

    def __init__(self) -> None:
        super().__init__()
        self.log_var_vq = nn.Parameter(torch.tensor(0.0))
        self.log_var_purity = nn.Parameter(torch.tensor(0.0))

    def forward(
        self, vq_loss: torch.Tensor, purity_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, dict[str, float]]:
        prec_vq = torch.exp(-self.log_var_vq)
        prec_pur = torch.exp(-self.log_var_purity)
        
        # 约束权重比：防止 VQ 任务碾压 Purity 任务导致语义监督失效
        max_ratio = 10.0
        ratio = prec_vq / prec_pur.clamp_min(1e-8)
        if ratio > max_ratio:
            prec_vq = prec_pur * max_ratio
        
        # 彻底删除 log_var 加法项，防止总损失变为负数
        total = prec_vq * vq_loss + prec_pur * purity_loss
        raw_weighted = prec_vq.detach() * vq_loss.detach() + prec_pur.detach() * purity_loss.detach()
        stats = {
            "balance_weight_vq": float(prec_vq.detach()),
            "balance_weight_purity": float(prec_pur.detach()),
            "loss_raw_weighted": float(raw_weighted),
        }
        return total, stats


class BPCv2(nn.Module):
    def __init__(
        self,
        registry: ScaleRegistry = DEFAULT_REGISTRY,
        unified_dim: int = 128,
        num_coarse: int = 128,
        commitment_cost: float = 0.25,
        primary_scale: str = "day",
        recon_weight: float = 1.0,
        purity_weight: float = 0.5,
        extended_purity_weight: float = 0.15,
        iso_weight: float = 0.05,
        diversity_weight: float = 0.1,
        vq_adapt_lr: float = 1e-5,
        vq_dead_code_threshold: float = 0.0,
        labeling_mode: str = "global",
        num_symbols: int = 10000,
        use_codebook_film: bool = False,   # 默认关闭，可按需开启
        use_fine_vq: bool = False,         # 默认关闭，简化训练
        num_fine_per_coarse: int = 16,
    ):
        super().__init__()
        self.registry = registry
        self.primary_scale = primary_scale
        self.recon_weight = recon_weight
        self.use_codebook_film = use_codebook_film
        self.use_fine_vq = use_fine_vq
        primary_cfg = registry.get(primary_scale)
        self.raw_feat_dim = compute_raw_feat_dim(primary_cfg)  # 24

        self.fusion = MultiScaleFusion(registry, unified_dim=unified_dim)
        self.pre_vq_norm = nn.LayerNorm(unified_dim)
        self.conditioner = SymbolTimeFiLM(
            unified_dim,
            num_symbols=num_symbols,
            enable_codebook_film=use_codebook_film,
        )
        self.vq = BehavioralVQ(
            dim=unified_dim,
            num_coarse=num_coarse,
            num_fine_per_coarse=num_fine_per_coarse,
            commitment_cost=commitment_cost,
            diversity_weight=diversity_weight,
            adapt_lr=vq_adapt_lr,
            dead_code_threshold=vq_dead_code_threshold,
        )
        self.decoder = CausalDecoder(unified_dim, out_features=self.raw_feat_dim)
        self.beh_loss_fn = BehavioralPurityLoss(
            unified_dim=unified_dim,
            primary_scale=primary_scale,
            purity_weight=purity_weight,
            extended_purity_weight=extended_purity_weight,
            iso_weight=iso_weight,
            labeling_mode=labeling_mode,
            num_symbols=num_symbols,
        )
        self.task_balancer = AdaptiveTaskBalancer()

    def encode(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        return self.fusion(batch)

    def quantize(
        self,
        z: torch.Tensor,
        use_fine: Optional[bool] = None,
        *,
        adapt: bool = False,
        stock_ids: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
    ):
        if use_fine is None:
            use_fine = self.use_fine_vq
        z = self.pre_vq_norm(z)
        cb_gamma, cb_beta = None, None
        if self.use_codebook_film:
            cb_gamma, cb_beta = self.conditioner.codebook_modulation(
                stock_ids, timestamps, z.shape[0], z.device
            )
        return self.vq(
            z,
            use_fine=use_fine,
            adapt=adapt,
            codebook_gamma=cb_gamma,
            codebook_beta=cb_beta,
        )

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        use_fine: Optional[bool] = None,
        return_loss: bool = True,
        adapt_vq: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # batch: {"day_features": [B, 24], "week": [B, 7], "stock_ids": [B], "timestamps": [B], "behavior_proxies": [B, 8]}
        z, scale_feats, raw_feats = self.encode(batch)
        if z is None:
            return {"loss": torch.tensor(0.0, requires_grad=True)}

        stock_ids = batch.get("stock_ids")
        timestamps = batch.get("timestamps")
        z = self.conditioner(z, stock_ids, timestamps)

        z_q, coarse_idx, fine_idx, vq_loss, vq_metrics = self.quantize(
            z,
            use_fine=use_fine,
            adapt=adapt_vq,
            stock_ids=stock_ids,
            timestamps=timestamps,
        )
        recon = self.decode(z_q)

        # 重构目标：预计算特征的归一化版本
        primary_raw = raw_feats.get(self.primary_scale)
        if primary_raw is not None:
            # raw_feats 现在是 PrecomputedFeatureComposer 的 raw 输出
            recon_target = primary_raw.detach()
        else:
            recon_target = z.detach()

        out = {
            "z_continuous": z,
            "z_quantized": z_q,
            "coarse_token": coarse_idx,
            "fine_token": fine_idx,
        }
        for mk, mv in vq_metrics.items():
            if mk.startswith("vq_") or mk == "loss_diversity":
                out[mk] = torch.tensor(mv, device=z.device)

        if return_loss:
            recon_loss = F.mse_loss(recon, recon_target.detach())
            stock_ids = batch.get("stock_ids")
            beh_losses = self.beh_loss_fn(z_q, scale_feats, batch, stock_ids=stock_ids)
            iso_loss = beh_losses.pop("iso", None)
            purity_loss = sum(beh_losses.values()) if beh_losses else torch.tensor(0.0, device=z.device)
            balanced, balance_stats = self.task_balancer(vq_loss, purity_loss)
            total_loss = self.recon_weight * recon_loss + balanced
            if iso_loss is not None:
                total_loss = total_loss + iso_loss
            out.update(
                {
                    "loss": total_loss,
                    "loss_vq": vq_loss,
                    "loss_recon": recon_loss,
                    "loss_purity_total": purity_loss,
                    **{f"loss_{k}": v for k, v in beh_losses.items()},
                }
            )
            if iso_loss is not None:
                out["loss_iso"] = iso_loss
            for mk, mv in balance_stats.items():
                out[mk] = torch.tensor(mv, device=z.device)
            for mk, mv in self.conditioner.modulation_stats().items():
                out[mk] = torch.tensor(mv, device=z.device)

        return out

    def save_behavioral_ontology(self, path: str):
        torch.save(
            {
                "coarse_embed": self.vq.coarse_embed.state_dict(),
                "ema_w": self.vq.ema_w.clone(),
                "ema_cluster_size": self.vq.ema_cluster_size.clone(),
                "num_coarse": self.vq.num_coarse,
                "dim": self.vq.dim,
            },
            path,
        )
        print(f"[BPCv2] Behavioral ontology saved to {path}")

    def load_behavioral_ontology(self, path: str, freeze: bool = True):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.vq.coarse_embed.load_state_dict(ckpt["coarse_embed"])
        self.vq.ema_w.copy_(ckpt["ema_w"])
        self.vq.ema_cluster_size.copy_(ckpt["ema_cluster_size"])
        if freeze:
            self.vq.coarse_embed.weight.requires_grad = False
            self.vq.ema_w.requires_grad = False
            print("[BPCv2] Coarse codebook frozen.")

    @torch.no_grad()
    def analyze_token_semantics(
        self,
        dataloader: DataLoader,
        device: str = "cpu",
        max_samples: int = 5000,
        primary_scale: str | None = None,
        compute_transitions: bool = True,
    ):
        """
        增强版 token 语义分析：
        - 返回每个 token 的行为代理均值/标准差/出现次数
        - 可选计算 token 转移熵（衡量从该 token 出发的市场状态不确定性）
        - 典型价格形态可通过反归一化重构（用户可进一步扩展）
        """
        import pandas as pd

        from .transition import TokenTransitionAnalyzer

        ps = primary_scale or self.primary_scale
        records = []
        token_sequences: List[List[int]] = []
        current_seq: List[int] = []

        for i, batch in enumerate(dataloader):
            if i * batch[list(batch.keys())[0]].shape[0] >= max_samples:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = self.forward(batch, return_loss=False)
            tokens = out["coarse_token"].cpu().numpy().tolist()

            for t_idx, tok in enumerate(tokens):
                rec = {"token": int(tok)}
                if ps in batch and t_idx < batch[ps].shape[0]:
                    x = batch[ps][t_idx : t_idx + 1]
                    proxies = compute_behavior_proxies(x)
                    for name in BEHAVIOR_AGENT_NAMES:
                        rec[name] = float(proxies[name][0])
                    close = batch[ps][t_idx, :, 3]
                    rec["net_return"] = float(close[-1] / close[0] - 1)
                records.append(rec)
                current_seq.append(int(tok))
            if current_seq:
                token_sequences.append(current_seq)
                current_seq = []

        df = pd.DataFrame(records)
        result = {"semantics": df.groupby("token").agg(["mean", "std", "count"]) if not df.empty else df}

        if compute_transitions and token_sequences:
            analyzer = TokenTransitionAnalyzer(num_tokens=self.vq.num_coarse)
            analyzer.update(token_sequences)
            result["transition_entropy"] = analyzer.transition_entropy().cpu().numpy()
            result["transition_matrix"] = analyzer.transition_matrix().cpu().numpy()

        return result


def precompute_purity_thresholds(
    model: BPCv2,
    loader: DataLoader,
    device: str = "cpu",
    max_batches: int = 500,
) -> None:
    """从预计算的 behavior_proxies 计算全局及 per-symbol 分位数阈值。"""
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in BEHAVIOR_AGENT_NAMES}
    per_symbol_collected: dict[int, list[torch.Tensor]] = defaultdict(list)
    ps = model.primary_scale
    use_per_symbol = model.beh_loss_fn.labeling_mode == "per_stock"  # ← 修复点

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            if "behavior_proxies" not in batch:
                continue
            stacked = batch["behavior_proxies"].to(device)
            stacked = transform_proxies_for_labeling(stacked)
            for j, name in enumerate(BEHAVIOR_AGENT_NAMES):
                collected[name].append(stacked[:, j].cpu())

            if use_per_symbol and "stock_ids" in batch:
                sids = batch["stock_ids"].to(device)
                for sid in torch.unique(sids):
                    mask = sids == sid
                    per_symbol_collected[int(sid)].append(stacked[mask].cpu())

    if not collected["vol"]:
        logger.warning("purity thresholds: no data, using batch quantile fallback.")
        return

    q = torch.tensor([1 / 3, 2 / 3])
    bounds: dict[str, torch.Tensor] = {}
    n_samples = 0
    for name in BEHAVIOR_AGENT_NAMES:
        vals = torch.cat(collected[name])
        bounds[name] = torch.quantile(vals, q)
        n_samples = max(n_samples, vals.shape[0])

    model.beh_loss_fn.set_thresholds(bounds)

    if use_per_symbol and per_symbol_collected:
        symbol_bounds: dict[int, torch.Tensor] = {}
        symbol_counts: dict[int, int] = {}
        sid_sizes = [len(torch.cat(pl)) for pl in per_symbol_collected.values() if pl]
        adaptive_min = max(8, int(torch.tensor(sid_sizes).median().item() * 0.1)) if sid_sizes else 8
        for sid, proxy_lists in per_symbol_collected.items():
            if not proxy_lists:
                continue
            all_proxies = torch.cat(proxy_lists, dim=0)
            if all_proxies.shape[0] < adaptive_min:
                continue
            per_agent_bounds = []
            for j in range(len(BEHAVIOR_AGENT_NAMES)):
                qs = torch.quantile(all_proxies[:, j], q)
                per_agent_bounds.append(qs)
            symbol_bounds[sid] = torch.stack(per_agent_bounds, dim=0)
            symbol_counts[sid] = all_proxies.shape[0]
        if symbol_bounds:
            model.beh_loss_fn.set_per_symbol_thresholds(symbol_bounds, symbol_counts=symbol_counts)
            logger.info("Per-symbol purity thresholds precomputed for %d symbols", len(symbol_bounds))

    core = {k: bounds[k].tolist() for k in ("vol", "attack", "amount")}
    ext = {k: bounds[k].tolist() for k in BehavioralPurityLoss.EXTENDED_AGENTS}
    logger.info("PurityThresholds fitted on %d samples | core=%s | extended=%s", n_samples, core, ext)


def precompute_normalizers(
    model: BPCv2,
    loader: DataLoader,
    device: str = "cpu",
    max_batches: int | None = None,
):
    """预计算归一化统计量（使用预计算特征时，LayerNorm 已替代手动归一化，此函数可跳过或用于兼容性）"""
    logger.info("Precomputed features use LayerNorm; skipping manual normalizer precomputation.")
    pass


def _aggregate_losses(out: dict, total: dict[str, float], count: int) -> int:
    if "loss" not in out:
        return count
    for k, v in out.items():
        if (
            k.startswith("loss")
            or k.startswith("vq_")
            or k.startswith("balance_weight_")
            or k.startswith("film_")
        ):
            if isinstance(v, torch.Tensor):
                total[k] += v.item()
            elif isinstance(v, (int, float)):
                total[k] += float(v)
    return count + 1


def _iter_training_batches(loader: DataLoader):
    ds = loader.dataset
    if hasattr(ds, "iter_batches"):
        ds.on_epoch_begin()
        yield from ds.iter_batches()
        return
    if hasattr(ds, "on_epoch_begin"):
        ds.on_epoch_begin()
    yield from loader


def _to_device(batch: dict, device: str, *, non_blocking: bool = False) -> dict:
    dev = torch.device(device)
    out: dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v if v.device == dev else v.to(dev, non_blocking=non_blocking)
        else:
            out[k] = v
    return out


@torch.no_grad()
def adapt_codebook_on_loader(
    model: BPCv2,
    loader: DataLoader,
    device: str = "cpu",
    *,
    max_batches: int | None = None,
    non_blocking: bool = False,
    max_distance_quantile: float | None = 0.9,
) -> int:
    """
    验证后单独执行：冻结 encoder，仅让码本向 val 分布缓慢漂移。
    新增 max_distance_quantile：仅适应与当前码本距离处于高分位数的样本（新模式/异常），
    避免对已充分覆盖的样本过度适应导致码本偏离训练分布。
    """
    model.eval()
    steps = 0
    all_distances: List[torch.Tensor] = []

    # 第一遍：收集距离统计（与 forward 一致：latent FiLM → pre_vq_norm → 可选码本 FiLM）
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = _to_device(batch, device, non_blocking=non_blocking)
        with torch.no_grad():
            z, _, _ = model.encode(batch)
            if z is None:
                continue
            stock_ids = batch.get("stock_ids")
            timestamps = batch.get("timestamps")
            z = model.conditioner(z, stock_ids, timestamps)
            z = model.pre_vq_norm(z)
            cb_gamma, cb_beta = None, None
            if model.use_codebook_film:
                cb_gamma, cb_beta = model.conditioner.codebook_modulation(
                    stock_ids, timestamps, z.shape[0], z.device
                )
            dist_mat = model.vq._coarse_distances(z, cb_gamma, cb_beta)
            distances = dist_mat.min(dim=1).values
            if cb_gamma is not None:
                distances = distances.sqrt()
            all_distances.append(distances.cpu())
    if not all_distances:
        return 0

    dist_cat = torch.cat(all_distances)
    if max_distance_quantile is not None:
        thresh = torch.quantile(dist_cat, max_distance_quantile)
    else:
        thresh = torch.inf

    # 第二遍：仅对高距离样本适应
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = _to_device(batch, device, non_blocking=non_blocking)
        with torch.no_grad():
            z, _, _ = model.encode(batch)
            if z is None:
                continue
            stock_ids = batch.get("stock_ids")
            timestamps = batch.get("timestamps")
            z = model.conditioner(z, stock_ids, timestamps)
            z = model.pre_vq_norm(z)
            cb_gamma, cb_beta = None, None
            if model.use_codebook_film:
                cb_gamma, cb_beta = model.conditioner.codebook_modulation(
                    stock_ids, timestamps, z.shape[0], z.device
                )
            dist_mat = model.vq._coarse_distances(z, cb_gamma, cb_beta)
            min_dist, coarse_idx = dist_mat.min(dim=1)
            if cb_gamma is not None:
                min_dist = min_dist.sqrt()
            mask = min_dist >= thresh
            if mask.any():
                model.vq._adapt_codebook(z[mask], coarse_idx[mask])
                steps += 1
    return steps


def eval_epoch(
    model: BPCv2,
    loader: DataLoader,
    device: str = "cpu",
    *,
    max_batches: int | None = None,
    amp: bool = False,
    non_blocking: bool = False,
) -> Dict[str, float]:
    model.eval()
    total_losses: dict[str, float] = defaultdict(float)
    count = 0
    use_amp = amp and device.startswith("cuda")
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = _to_device(batch, device, non_blocking=non_blocking)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            out = model(batch, return_loss=True, adapt_vq=False)
        count = _aggregate_losses(out, total_losses, count)
    if count == 0:
        return {}
    return {k: v / count for k, v in total_losses.items()}


def train_epoch(
    model: BPCv2,
    loader: DataLoader,
    optimizer,
    device: str = "cpu",
    max_grad_norm: float = 1.0,
    *,
    amp: bool = False,
    non_blocking: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> Dict[str, float]:
    model.train()
    total_losses: dict[str, float] = defaultdict(float)
    grad_norm_sum = 0.0
    count = 0
    use_amp = amp and device.startswith("cuda")
    if use_amp and scaler is None:
        scaler = torch.amp.GradScaler("cuda")

    for batch in _iter_training_batches(loader):
        if not hasattr(loader.dataset, "iter_batches"):
            batch = _to_device(batch, device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            out = model(batch)
        if "loss" not in out or not torch.isfinite(out["loss"]):
            continue
        if use_amp and scaler is not None:
            scaler.scale(out["loss"]).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            grad_norm_sum += float(grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            out["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                continue
            grad_norm_sum += float(grad_norm)
            optimizer.step()
        count = _aggregate_losses(out, total_losses, count)

    if count == 0:
        return {}
    metrics = {k: v / count for k, v in total_losses.items()}
    metrics["grad_norm"] = grad_norm_sum / count
    return metrics