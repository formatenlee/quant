"""
BPC-v2: 行为本体向量量化（接入 quant_cursor pyqlib 日线数据）。

多尺度：day + week（由日线重采样）。无分钟/Tick 数据时禁用 1min 尺度。
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


# 当前 pyqlib 仅有日线；week 由 dataset 重采样
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
            feature_groups=[
                "price_structure",
                "volume_structure",
                "attack_proxy",
                "micro_proxy",
            ],
        )
    )
    registry.register(
        ScaleConfig(
            name="week",
            freq="week",
            lookback_window=week_lookback,
            feature_groups=["price_structure"],
        )
    )
    return registry


DEFAULT_REGISTRY = build_scale_registry()


def compute_raw_feat_dim(cfg: Optional[ScaleConfig]) -> int:
    if cfg is None:
        return 16
    group_dim_map = {
        "price_structure": 7,
        "volume_structure": 4,
        "attack_proxy": 3,
        "micro_proxy": 2,
        "time_structure": 4,
    }
    return sum(group_dim_map.get(g, 0) for g in cfg.feature_groups)


# =============================================================================
# 2. 因果特征提取器
# =============================================================================
class CausalFeatureComposer(nn.Module):
    def __init__(self, feature_groups: List[str], out_dim: int):
        super().__init__()
        self.groups = feature_groups

        group_dim_map = {
            "price_structure": 7,
            "volume_structure": 4,
            "attack_proxy": 3,
            "micro_proxy": 2,
            "time_structure": 4,
        }
        self.group_dims = {k: group_dim_map[k] for k in feature_groups if k in group_dim_map}
        self.raw_feat_dim = sum(self.group_dims.values())
        total_feat_dim = self.raw_feat_dim

        self.proj = nn.Linear(total_feat_dim, out_dim)
        self.register_buffer("feature_mean", torch.zeros(total_feat_dim))
        self.register_buffer("feature_std", torch.ones(total_feat_dim))

    def _price_structure(self, x: torch.Tensor) -> torch.Tensor:
        close = x[..., 3]
        high = x[..., 1]
        low = x[..., 2]

        log_ret = torch.log(close[:, 1:] / close[:, :-1].clamp_min(1e-8))
        feats = []

        rv = log_ret.std(dim=1, keepdim=True)
        feats.append(rv)

        m = log_ret.mean(dim=1, keepdim=True)
        s = log_ret.std(dim=1, keepdim=True).clamp_min(1e-8)
        skew = ((log_ret - m) ** 3).mean(dim=1, keepdim=True) / (s**3).clamp_min(1e-4)
        skew = skew.clamp(-10.0, 10.0)
        feats.append(skew)

        net_move = (close[:, -1] - close[:, 0]).abs()
        path_len = (high - low).abs().sum(dim=1)
        efficiency = net_move / path_len.clamp_min(1e-8)
        feats.append(efficiency.unsqueeze(1))

        price_pos = (close[:, -1] - low.min(dim=1)[0]) / (
            high.max(dim=1)[0] - low.min(dim=1)[0]
        ).clamp_min(1e-8)
        feats.append(price_pos.unsqueeze(1))

        for lag in [1, 2, 5]:
            if log_ret.shape[1] > lag:
                x1 = log_ret[:, :-lag]
                x2 = log_ret[:, lag:]
                x1 = x1 - x1.mean(dim=1, keepdim=True)
                x2 = x2 - x2.mean(dim=1, keepdim=True)
                corr = (x1 * x2).sum(dim=1) / (x1.norm(dim=1) * x2.norm(dim=1)).clamp_min(1e-8)
                feats.append(corr.unsqueeze(1))
            else:
                feats.append(torch.zeros_like(rv))

        return torch.cat(feats, dim=1)

    def _volume_structure(self, x: torch.Tensor) -> torch.Tensor:
        volume = x[..., 4]
        close = x[..., 3]
        log_vol = torch.log(volume.clamp_min(1.0))
        feats = []

        feats.append(log_vol.mean(dim=1, keepdim=True))
        feats.append(
            log_vol.std(dim=1, keepdim=True) / log_vol.mean(dim=1, keepdim=True).clamp_min(1e-8)
        )

        ret = torch.log(close[:, 1:] / close[:, :-1].clamp_min(1e-8))
        vol_slice = volume[:, 1:]
        if ret.shape[1] > 1:
            vp_corr = self._pearson(ret, vol_slice).unsqueeze(1)
        else:
            vp_corr = torch.zeros_like(feats[0])
        feats.append(vp_corr)

        sorted_vol, _ = torch.sort(volume, dim=1, descending=True)
        top20 = sorted_vol[:, : max(1, volume.shape[1] // 5)].sum(dim=1) / volume.sum(
            dim=1
        ).clamp_min(1e-8)
        feats.append(top20.unsqueeze(1))

        return torch.cat(feats, dim=1)

    def _attack_proxy(self, x: torch.Tensor) -> torch.Tensor:
        open_p = x[..., 0]
        high = x[..., 1]
        low = x[..., 2]
        close = x[..., 3]
        volume = x[..., 4]
        feats = []

        bar_range = (high - low).clamp_min(1e-8)
        eff = (close - open_p) / bar_range
        vol_mean = volume.mean(dim=1, keepdim=True).clamp_min(1e-8)
        rel_vol = volume / vol_mean
        attack = (eff * rel_vol).mean(dim=1, keepdim=True)
        feats.append(attack)

        last_n = min(5, eff.shape[1])
        last_eff = eff[:, -last_n:]
        last_rel = rel_vol[:, -last_n:]
        last_attack = last_eff * last_rel
        if last_n > 1:
            decay = (last_attack[:, -1] - last_attack[:, 0]) / (last_n - 1)
        else:
            decay = torch.zeros_like(attack.squeeze(1))
        feats.append(decay.unsqueeze(1))

        pressure = (close[:, -1] - low.min(dim=1)[0]) / (
            high.max(dim=1)[0] - low.min(dim=1)[0]
        ).clamp_min(1e-8)
        feats.append(pressure.unsqueeze(1))

        return torch.cat(feats, dim=1)

    def _micro_proxy(self, x: torch.Tensor) -> torch.Tensor:
        high = x[..., 1]
        low = x[..., 2]
        volume = x[..., 4]
        feats = []

        parkinson = torch.sqrt(
            (torch.log(high / low.clamp_min(1e-8)) ** 2).mean(dim=1) / (4 * math.log(2))
        )
        feats.append(parkinson.unsqueeze(1))

        pulse = volume[:, -1] / volume.mean(dim=1).clamp_min(1e-8)
        feats.append(pulse.unsqueeze(1))

        return torch.cat(feats, dim=1)

    def _pearson(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=1, keepdim=True)
        y = y - y.mean(dim=1, keepdim=True)
        num = (x * y).sum(dim=1)
        den = (x.norm(dim=1) * y.norm(dim=1)).clamp_min(1e-8)
        return num / den

    def extract_raw_features(self, x: torch.Tensor) -> torch.Tensor:
        group_feats = []
        if "price_structure" in self.groups:
            group_feats.append(self._price_structure(x))
        if "volume_structure" in self.groups:
            group_feats.append(self._volume_structure(x))
        if "attack_proxy" in self.groups:
            group_feats.append(self._attack_proxy(x))
        if "micro_proxy" in self.groups:
            group_feats.append(self._micro_proxy(x))
        if not group_feats:
            return torch.zeros(x.shape[0], self.proj.in_features, device=x.device)
        feat = torch.cat(group_feats, dim=1)
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        return feat.clamp(-20.0, 20.0)

    def normalize_features(self, raw: torch.Tensor) -> torch.Tensor:
        """与 forward 中一致的因果归一化（用于重构目标）。"""
        norm = (raw - self.feature_mean) / self.feature_std.clamp_min(1e-6)
        return torch.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)

    def forward(self, x: torch.Tensor, timestamps=None, return_raw: bool = False):
        raw = self.extract_raw_features(x)
        feat = self.normalize_features(raw)
        proj = self.proj(feat)
        if return_raw:
            return proj, raw
        return proj


# =============================================================================
# 3. 因果归一化
# =============================================================================
class CausalNormalizer:
    def __init__(self, composer: CausalFeatureComposer, max_batches: int | None = None):
        self.composer = composer
        self.max_batches = max_batches

    @torch.no_grad()
    def fit(self, dataloader: DataLoader, device: str = "cpu", scale_name: str = "day"):
        feat_dim: int | None = None
        total_count = 0
        sum_x: torch.Tensor | None = None
        sum_x2: torch.Tensor | None = None

        for i, batch in enumerate(dataloader):
            if self.max_batches is not None and i >= self.max_batches:
                break
            if scale_name not in batch:
                continue
            x = batch[scale_name]
            if not isinstance(x, torch.Tensor):
                continue
            x = x.to(device)
            feat = self.composer.extract_raw_features(x)
            valid = torch.isfinite(feat).all(dim=1)
            feat = feat[valid]
            if feat.numel() == 0:
                continue

            if feat_dim is None:
                feat_dim = feat.shape[1]
                sum_x = torch.zeros(feat_dim, device=feat.device)
                sum_x2 = torch.zeros(feat_dim, device=feat.device)

            total_count += feat.shape[0]
            sum_x += feat.sum(dim=0)
            sum_x2 += (feat * feat).sum(dim=0)

        if not total_count or sum_x is None or sum_x2 is None:
            logger.warning("No valid data for scale %s during normalization.", scale_name)
            return

        mean = sum_x / total_count
        var = (sum_x2 / total_count - mean * mean).clamp_min(0.0)
        std = torch.sqrt(var).clamp_min(1e-6)
        self.composer.feature_mean.copy_(mean.cpu())
        self.composer.feature_std.copy_(std.cpu())
        logger.info(
            "CausalNormalizer scale '%s' fitted on %d samples (finite-only, online).",
            scale_name,
            total_count,
        )


# =============================================================================
# 4–8. 编码器、融合、VQ、解码器、损失
# =============================================================================
class ScaleEncoder(nn.Module):
    def __init__(self, cfg: ScaleConfig, feat_dim: int = 64):
        super().__init__()
        self.cfg = cfg
        self.composer = CausalFeatureComposer(feature_groups=cfg.feature_groups, out_dim=feat_dim)
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

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        scale_feats: Dict[str, torch.Tensor] = {}
        raw_feats: Dict[str, torch.Tensor] = {}
        for cfg in self.registry.get_enabled():
            name = cfg.name
            if name not in batch:
                continue
            x = batch[name]
            enc = self.encoders[name]
            h, raw = enc(x)
            proj = getattr(self, f"proj_{name}")
            scale_feats[name] = proj(h)
            raw_feats[name] = raw

        if not scale_feats:
            return None, {}, {}

        weights = F.softmax(self.scale_weights[: len(scale_feats)], dim=0)
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
        dead_code_threshold: float = 0.01,
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

        self.num_fine_total = num_coarse * num_fine_per_coarse
        self.fine_embed = nn.Embedding(self.num_fine_total, dim)
        self.fine_embed.weight.data.uniform_(-1 / self.num_fine_total, 1 / self.num_fine_total)

        self.register_buffer("ema_cluster_size", torch.zeros(num_coarse))
        self.register_buffer("ema_w", self.coarse_embed.weight.data.clone())
        self.register_buffer("adaptation_steps", torch.tensor(0, dtype=torch.long))

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

    def forward(
        self,
        z: torch.Tensor,
        use_fine: bool = False,
        *,
        adapt: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, Dict[str, float]]:
        distances = torch.cdist(z, self.coarse_embed.weight)
        min_dist, coarse_idx = distances.min(dim=1)
        z_q_coarse = self.coarse_embed(coarse_idx)

        if self.training:
            self._update_ema(z, coarse_idx)
        elif adapt:
            self._adapt_codebook(z, coarse_idx)

        z_q = z + (z_q_coarse - z).detach()
        vq_loss = self.commitment_cost * F.mse_loss(z_q_coarse.detach(), z)

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
            **self._residual_metrics(z, z_q_coarse, min_dist),
        }

        fine_idx = None
        if use_fine:
            fine_distances = torch.cdist(z, self.fine_embed.weight)
            fine_idx = torch.argmin(fine_distances, dim=1)
            z_q_fine = self.fine_embed(fine_idx)
            z_q = z + (z_q_fine - z).detach()
            vq_loss = vq_loss + self.commitment_cost * F.mse_loss(z_q_fine.detach(), z)

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
        """标准 VQ-VAE EMA + 死码复活。"""
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

    def set_thresholds(self, bounds: dict[str, torch.Tensor]) -> None:
        for name in BEHAVIOR_AGENT_NAMES:
            if name not in bounds:
                continue
            buf = getattr(self, f"{name}_bounds")
            buf.copy_(bounds[name].flatten()[:2].to(buf.device))
        self.thresholds_ready.fill_(True)

    def set_per_symbol_thresholds(self, symbol_bounds: dict[int, torch.Tensor]) -> None:
        """Store precomputed per-symbol 33%/66% quantiles.

        symbol_bounds[sid] is a tensor of shape [8, 2] or [num_agents, 2].
        """
        for sid, b in symbol_bounds.items():
            if 0 <= sid < self.per_symbol_bounds.shape[0]:
                self.per_symbol_bounds[sid] = b.to(self.per_symbol_bounds.device)
                self.per_symbol_ready[sid] = True

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
        ready = self.per_symbol_ready[sids].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

        global_bounds = torch.stack(
            [getattr(self, f"{name}_bounds") for name in BEHAVIOR_AGENT_NAMES], dim=0
        ).to(device)  # [8, 2]
        global_expanded = global_bounds.unsqueeze(0).expand(bsz, -1, -1)  # [B, 8, 2]
        bounds = torch.where(ready, per_bounds, global_expanded)

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
            count = 0
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    iso_loss += F.mse_loss(vals[i], vals[j])
                    count += 1
            losses["iso"] = (iso_loss / count) * self.iso_weight

        ps = self.primary_scale
        if ps not in raw_batch:
            return losses

        token_logits = self.behavior_proj(z_q)

        if "behavior_proxies" in raw_batch:
            proxy_mat = raw_batch["behavior_proxies"]
            if proxy_mat.dim() == 1:
                proxy_mat = proxy_mat.unsqueeze(0)
        else:
            x = raw_batch[ps]
            proxy_mat = compute_behavior_proxies_stacked(x)

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
    Symbol-specific + time-dependent modulation.

    Captures per-symbol stylistic offsets (e.g., ETF vs small-cap stock behavior)
    and seasonal / cyclical regime shifts while keeping the shared BehavioralVQ
    codebook focused on universal market primitives.

    Gamma is tanh-bounded and scaled by a learnable `gamma_scale` parameter
    (initialized at 0.6). The modulation strength is now determined during
    training rather than using a fixed hand-tuned constant.
    """

    def __init__(self, dim: int, num_symbols: int = 10000, time_emb_dim: int = 16):
        super().__init__()
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
        # Learnable modulation strength (replaces magic constant 0.6).
        # The model learns how aggressively to apply per-symbol / seasonal adjustments
        # instead of using a fixed hand-tuned scale.
        self.gamma_scale = nn.Parameter(torch.tensor(0.6))

    def forward(
        self, z: torch.Tensor, stock_ids: Optional[torch.Tensor], timestamps: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if stock_ids is None and timestamps is None:
            # Graceful fallback – conditioner becomes identity. This happens when the
            # DataLoader does not provide stock_ids / timestamps (common before dataset upgrade).
            return z
        B = z.shape[0]
        device = z.device
        sym_emb = torch.zeros(B, self.symbol_embed.embedding_dim, device=device)
        if stock_ids is not None:
            sid = stock_ids.to(device).clamp(0, self.symbol_embed.num_embeddings - 1)
            sym_emb = self.symbol_embed(sid)
        time_emb = torch.zeros(B, self.symbol_embed.embedding_dim, device=device)
        if timestamps is not None:
            ts = timestamps.to(device).float()
            # Purely seasonal + cyclical features (no absolute year to avoid binding to specific periods)
            month = ((ts // 30) % 12) / 12.0  # approximate month-of-year
            quarter = ((ts // 90) % 4) / 4.0
            dow = (ts % 7) / 7.0
            # Relative position within a typical trading year (smooth periodic signal)
            year_phase = torch.sin(2 * math.pi * (ts % 252) / 252.0)
            tfeat = torch.stack([month, quarter, dow, year_phase], dim=-1).view(B, 4)
            time_emb = self.time_mlp(tfeat)
        cond = torch.cat([sym_emb, time_emb], dim=-1)
        gamma_beta = self.film(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        # Learnable scale (data-driven instead of fixed 0.6).
        # Training will determine the appropriate strength of symbol/time modulation.
        gamma = torch.tanh(gamma) * self.gamma_scale
        return z * (1 + gamma) + beta


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
        vq_dead_code_threshold: float = 0.01,
        labeling_mode: str = "global",
        num_symbols: int = 10000,
    ):
        super().__init__()
        self.registry = registry
        self.primary_scale = primary_scale
        self.recon_weight = recon_weight
        primary_cfg = registry.get(primary_scale)
        self.raw_feat_dim = compute_raw_feat_dim(primary_cfg)

        self.fusion = MultiScaleFusion(registry, unified_dim=unified_dim)
        self.pre_vq_norm = nn.LayerNorm(unified_dim)
        self.conditioner = SymbolTimeFiLM(unified_dim, num_symbols=num_symbols)
        self.vq = BehavioralVQ(
            dim=unified_dim,
            num_coarse=num_coarse,
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

    def encode(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        return self.fusion(batch)

    def quantize(self, z: torch.Tensor, use_fine: bool = False, *, adapt: bool = False):
        z = self.pre_vq_norm(z)
        return self.vq(z, use_fine=use_fine, adapt=adapt)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        use_fine: bool = False,
        return_loss: bool = True,
        adapt_vq: bool = False,
    ) -> Dict[str, torch.Tensor]:
        z, scale_feats, raw_feats = self.encode(batch)
        if z is None:
            return {"loss": torch.tensor(0.0, requires_grad=True)}

        stock_ids = batch.get("stock_ids")
        timestamps = batch.get("timestamps")
        z = self.conditioner(z, stock_ids, timestamps)

        z_q, coarse_idx, fine_idx, vq_loss, vq_metrics = self.quantize(z, use_fine, adapt=adapt_vq)
        recon = self.decode(z_q)

        primary_raw = raw_feats.get(self.primary_scale)
        if primary_raw is not None:
            composer = self.fusion.encoders[self.primary_scale].composer
            recon_target = composer.normalize_features(primary_raw).detach()
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
            total_loss = self.recon_weight * recon_loss + vq_loss + sum(beh_losses.values())
            out.update(
                {
                    "loss": total_loss,
                    "loss_vq": vq_loss,
                    "loss_recon": recon_loss,
                    **{f"loss_{k}": v for k, v in beh_losses.items()},
                }
            )

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
    """在训练集上为全部 8 个行为代理固定 33%/66% 分位数阈值。

    当 labeling_mode == "per_stock" 时，同时预计算每只股票（symbol）的稳定分位数，
    避免训练过程中在线分位数波动导致的标签不稳定。
    """
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in BEHAVIOR_AGENT_NAMES}
    per_symbol_collected: dict[int, list[torch.Tensor]] = defaultdict(list)  # sid -> list of [8] proxy vectors
    ps = model.primary_scale
    use_per_symbol = model.beh_loss_fn.labeling_mode == "per_stock"

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            if ps not in batch:
                continue
            if "behavior_proxies" in batch:
                stacked = batch["behavior_proxies"].to(device)
                stacked = transform_proxies_for_labeling(stacked)
                for j, name in enumerate(BEHAVIOR_AGENT_NAMES):
                    collected[name].append(stacked[:, j].cpu())
            else:
                x = batch[ps].to(device)
                stacked = transform_proxies_for_labeling(compute_behavior_proxies_stacked(x))
                for j, name in enumerate(BEHAVIOR_AGENT_NAMES):
                    collected[name].append(stacked[:, j].cpu())

            if use_per_symbol and "stock_ids" in batch:
                sids = batch["stock_ids"].to(device)
                proxies = stacked if "behavior_proxies" in batch else transform_proxies_for_labeling(
                    compute_behavior_proxies_stacked(batch[ps].to(device))
                )
                for sid in torch.unique(sids):
                    mask = sids == sid
                    per_symbol_collected[int(sid)].append(proxies[mask].cpu())

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

    # Per-symbol stable thresholds (only when requested)
    if use_per_symbol and per_symbol_collected:
        symbol_bounds: dict[int, torch.Tensor] = {}
        agent_names = BEHAVIOR_AGENT_NAMES
        for sid, proxy_lists in per_symbol_collected.items():
            if len(proxy_lists) == 0:
                continue
            all_proxies = torch.cat(proxy_lists, dim=0)  # [n_samples_for_sid, 8]
            if all_proxies.shape[0] < 5:
                continue  # too few samples → keep global only
            per_agent_bounds = []
            for j in range(len(agent_names)):
                qs = torch.quantile(all_proxies[:, j], q)
                per_agent_bounds.append(qs)
            symbol_bounds[sid] = torch.stack(per_agent_bounds)  # [8, 2]
        if symbol_bounds:
            model.beh_loss_fn.set_per_symbol_thresholds(symbol_bounds)
            logger.info(
                "Per-symbol purity thresholds precomputed for %d symbols (stable labeling enabled)",
                len(symbol_bounds),
            )

    core = {k: bounds[k].tolist() for k in ("vol", "attack", "amount")}
    ext = {k: bounds[k].tolist() for k in BehavioralPurityLoss.EXTENDED_AGENTS}
    logger.info(
        "PurityThresholds fitted on %d samples | core=%s | extended=%s",
        n_samples,
        core,
        ext,
    )


def precompute_normalizers(
    model: BPCv2,
    loader: DataLoader,
    device: str = "cpu",
    max_batches: int | None = None,
):
    model.eval()
    for cfg in model.registry.get_enabled():
        name = cfg.name
        if name not in model.fusion.encoders:
            continue
        composer = model.fusion.encoders[name].composer
        normalizer = CausalNormalizer(composer, max_batches=max_batches)
        normalizer.fit(loader, device=device, scale_name=name)


def _aggregate_losses(out: dict, total: dict[str, float], count: int) -> int:
    if "loss" not in out:
        return count
    for k, v in out.items():
        if k.startswith("loss") or k.startswith("vq_"):
            if isinstance(v, torch.Tensor):
                total[k] += v.item()
            elif isinstance(v, (int, float)):
                total[k] += float(v)
    return count + 1


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

    # 第一遍：收集距离统计
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = _to_device(batch, device, non_blocking=non_blocking)
        with torch.no_grad():
            z, _, _ = model.encode(batch)
            if z is None:
                continue
            z = model.pre_vq_norm(z)
            distances = torch.cdist(z, model.vq.coarse_embed.weight).min(dim=1).values
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
            z = model.pre_vq_norm(z)
            min_dist, coarse_idx = torch.cdist(z, model.vq.coarse_embed.weight).min(dim=1)
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
            out = model(batch, use_fine=False, return_loss=True, adapt_vq=False)
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

    for batch in loader:
        batch = _to_device(batch, device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            out = model(batch, use_fine=False)
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
