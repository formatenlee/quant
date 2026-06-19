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
    PURITY_LOSS_KEYS,
    agent_logit_slice,
    compute_behavior_proxies,
    compute_behavior_proxies_stacked,
    transform_proxies_for_features,
    transform_proxies_for_labeling,
)
from quant_cursor.bpc.behavioral_vq import BehavioralVQ
from quant_cursor.bpc.feature_dims import (
    DAY_FULL_FEAT_DIM,
    DAY_STRUCT_FEAT_DIM,
    GROUP_DIM_MAP,
    WEEK_FEAT_DIM,
)
from quant_cursor.bpc.structure_features import volume_log_level_anomaly, volume_rel_cv
from quant_cursor.bpc.vq_backend import VQMode, get_backend, normalize_vq_mode, uses_magnitude_split

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
    *,
    precomputed: bool = True,
) -> ScaleRegistry:
    registry = ScaleRegistry()
    if precomputed:
        day_groups = ["precomputed_features"]
        week_groups = ["precomputed_features"]
    else:
        day_groups = [
            "price_structure",
            "volume_structure",
            "attack_proxy",
            "micro_proxy",
            "behavior_structure",
        ]
        week_groups = ["price_structure"]
    registry.register(
        ScaleConfig(name="day", freq="day", lookback_window=day_lookback, feature_groups=day_groups)
    )
    registry.register(
        ScaleConfig(name="week", freq="week", lookback_window=week_lookback, feature_groups=week_groups)
    )
    return registry


DEFAULT_REGISTRY = build_scale_registry()


def compute_encoder_in_dim(cfg: Optional[ScaleConfig]) -> int:
    if cfg is None:
        return DAY_FULL_FEAT_DIM
    if "precomputed_features" in cfg.feature_groups:
        return DAY_FULL_FEAT_DIM if cfg.name == "day" else WEEK_FEAT_DIM
    return sum(GROUP_DIM_MAP.get(g, 0) for g in cfg.feature_groups)


def compute_raw_feat_dim(cfg: Optional[ScaleConfig]) -> int:
    return compute_encoder_in_dim(cfg)


def compute_recon_feat_dim(cfg: Optional[ScaleConfig]) -> int:
    if cfg is None:
        return DAY_STRUCT_FEAT_DIM
    if cfg.name == "day":
        return DAY_STRUCT_FEAT_DIM
    if cfg.name == "week":
        return WEEK_FEAT_DIM
    return DAY_STRUCT_FEAT_DIM


# =============================================================================
# 2. 特征提取器（预计算向量 / 因果 OHLCV）
# =============================================================================
class PrecomputedFeatureComposer(nn.Module):
    """物化后的固定维特征：LayerNorm + Linear。"""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, timestamps=None, return_raw: bool = False):
        if x.dim() == 3:
            raise ValueError("PrecomputedFeatureComposer expects [B, D] features, not OHLCV windows")
        norm = self.norm(x)
        proj = self.proj(norm)
        if return_raw:
            return proj, x
        return proj


class CausalFeatureComposer(nn.Module):
    """因果特征组合器：绝对 OHLCV → bpc.features（v2 专用，与 v3 隔离）。"""

    def __init__(self, feature_groups: List[str], out_dim: int, *, scale_name: str = "day"):
        super().__init__()
        self.groups = feature_groups
        self.scale_name = scale_name
        self.group_dims = {k: GROUP_DIM_MAP[k] for k in feature_groups if k in GROUP_DIM_MAP}
        self.raw_feat_dim = sum(self.group_dims.values())
        self.proj = nn.Linear(self.raw_feat_dim, out_dim)
        self.register_buffer("feature_mean", torch.zeros(self.raw_feat_dim))
        self.register_buffer("feature_std", torch.ones(self.raw_feat_dim))

    def extract_raw_features(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        from quant_cursor.bpc.features import extract_group_features

        feat = extract_group_features(x, self.groups, scale_name=self.scale_name)
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        return feat.clamp(-20.0, 20.0)

    def normalize_features(self, raw: torch.Tensor) -> torch.Tensor:
        """与 forward 中一致的因果归一化（用于重构目标）。"""
        norm = (raw - self.feature_mean) / self.feature_std.clamp_min(1e-6)
        return torch.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)

    def forward(
        self,
        x: torch.Tensor,
        timestamps=None,
        return_raw: bool = False,
        **kwargs,
    ):
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
        sum_x: torch.Tensor | None = None
        sum_x2: torch.Tensor | None = None
        total_count: torch.Tensor | None = None

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

            if feat_dim is None:
                feat_dim = feat.shape[1]
                sum_x = torch.zeros(feat_dim, device=feat.device)
                sum_x2 = torch.zeros(feat_dim, device=feat.device)
                total_count = torch.zeros(feat_dim, device=feat.device)

            # 按特征维度独立计算有效样本，避免整行丢弃导致信息损失
            valid_mask = torch.isfinite(feat)
            for d in range(feat_dim):
                valid_d = valid_mask[:, d]
                if valid_d.any():
                    feat_d = feat[valid_d, d]
                    sum_x[d] += feat_d.sum()
                    sum_x2[d] += (feat_d * feat_d).sum()
                    total_count[d] += valid_d.sum()

        if total_count is None or (total_count == 0).all():
            logger.warning("No valid data for scale %s during normalization.", scale_name)
            return

        mean = sum_x / total_count.clamp_min(1.0)
        var = (sum_x2 / total_count.clamp_min(1.0) - mean * mean).clamp_min(0.0)
        std = torch.sqrt(var).clamp_min(1e-6)
        self.composer.feature_mean.copy_(mean.cpu())
        self.composer.feature_std.copy_(std.cpu())
        logger.info(
            "CausalNormalizer scale '%s' fitted per dim (min_samples=%d, max_samples=%d).",
            scale_name,
            int(total_count.min().item()),
            int(total_count.max().item()),
        )


# =============================================================================
# 4–8. 编码器、融合、VQ、解码器、损失
# =============================================================================
class ScaleEncoder(nn.Module):
    def __init__(self, cfg: ScaleConfig, feat_dim: int = 64):
        super().__init__()
        self.cfg = cfg
        self.use_precomputed = "precomputed_features" in cfg.feature_groups
        if self.use_precomputed:
            self.composer = PrecomputedFeatureComposer(compute_encoder_in_dim(cfg), feat_dim)
        else:
            self.composer = CausalFeatureComposer(
                feature_groups=cfg.feature_groups, out_dim=feat_dim, scale_name=cfg.name
            )
        self.aggregator = nn.Sequential(
            nn.Linear(feat_dim, cfg.encoder_dim),
            nn.LayerNorm(cfg.encoder_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        x: torch.Tensor,
        timestamps=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_precomputed:
            proj, raw = self.composer(x, timestamps, return_raw=True)
        else:
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
            feat_key = f"{name}_features"
            if feat_key in batch:
                x = batch[feat_key]
            elif name in batch:
                x = batch[name]
            else:
                continue
            enc = self.encoders[name]
            h, raw = enc(x, timestamps=batch.get("timestamps"))
            proj = getattr(self, f"proj_{name}")
            scale_feats[name] = proj(h)
            raw_feats[name] = raw

        if not scale_feats:
            return None, {}, {}

        weights = F.softmax(self.scale_weights[: len(scale_feats)], dim=0)
        fused = sum(scale_feats[name] * w for name, w in zip(scale_feats.keys(), weights))
        return fused, scale_feats, raw_feats


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
    行为纯度损失：5 个统计结构代理 × 3 档，共 15 维 logits。
    核心：vol / attack / path_structure / vol_structure；扩展：momentum。
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
        threshold_decay_half_life: float = 252.0,
        val_threshold_ema_decay: float = 0.95,
        use_magnitude_for_purity: bool = False,
        use_relative_z_scale_for_purity: bool = True,
        purity_latent: str = "continuous",
    ):
        super().__init__()
        self.primary_scale = primary_scale
        self.purity_weight = purity_weight
        self.extended_purity_weight = extended_purity_weight
        self.iso_weight = iso_weight
        self.labeling_mode = labeling_mode  # "global" | "per_stock" | "batch"
        self.num_symbols = num_symbols
        self.threshold_decay_half_life = threshold_decay_half_life
        self.val_threshold_ema_decay = val_threshold_ema_decay
        self.use_magnitude_for_purity = use_magnitude_for_purity
        self.use_relative_z_scale_for_purity = use_relative_z_scale_for_purity
        if purity_latent not in ("continuous", "quantized"):
            raise ValueError(f"purity_latent must be 'continuous' or 'quantized', got {purity_latent!r}")
        self.purity_latent = purity_latent
        self.proxies_label_ready = False
        self.register_buffer("threshold_fit_ordinal", torch.tensor(0, dtype=torch.long))
        self.register_buffer("thresholds_frozen", torch.tensor(False))
        self._val_proxy_accum: dict[int, list[torch.Tensor]] = {}
        purity_in = unified_dim + (1 if use_magnitude_for_purity else 0)
        self.purity_input_norm = nn.LayerNorm(purity_in)
        self.behavior_proj = nn.Sequential(
            nn.Linear(purity_in, 96),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(96, BEHAVIOR_LOGITS_DIM),
        )
        # vol 对宏观波动制度敏感，降低 per_stock 分位数权重、更多依赖全局阈值
        self._regime_sensitive_agent_idx = (BEHAVIOR_AGENT_NAMES.index("vol"),)
        self.register_buffer("regime_sensitive_blend", torch.tensor(0.5))
        for name in BEHAVIOR_AGENT_NAMES:
            self.register_buffer(f"{name}_bounds", torch.zeros(2))
        self.register_buffer("thresholds_ready", torch.tensor(False))

        # Per-symbol precomputed quantiles (stable across epochs)
        # Shape: [num_symbols, num_agents, 2 bounds (33%, 66%)]
        n_agents = len(BEHAVIOR_AGENT_NAMES)
        self.register_buffer("per_symbol_bounds", torch.zeros(num_symbols, n_agents, 2))
        self.register_buffer("per_symbol_bounds_frozen", torch.zeros(num_symbols, n_agents, 2))
        self.register_buffer("per_symbol_bounds_eval", torch.zeros(num_symbols, n_agents, 2))
        self.register_buffer("per_symbol_ready", torch.zeros(num_symbols, dtype=torch.bool))
        self.register_buffer("per_symbol_counts", torch.zeros(num_symbols, dtype=torch.long))
        self.register_buffer("per_symbol_z_scale_baseline", torch.ones(num_symbols))
        self.register_buffer("per_symbol_z_scale_frozen", torch.ones(num_symbols))
        self.register_buffer("per_symbol_z_scale_ready", torch.zeros(num_symbols, dtype=torch.bool))
        self.register_buffer("global_z_scale_baseline", torch.tensor(1.0))
        # Bayesian-style prior mass for per-stock vs global threshold blending (learnable).
        self.per_symbol_blend_log = nn.Parameter(torch.tensor(4.0))

    def set_z_scale_baselines(
        self,
        symbol_baselines: dict[int, float],
        *,
        global_baseline: float,
    ) -> None:
        for sid, val in symbol_baselines.items():
            if 0 <= sid < self.per_symbol_z_scale_baseline.shape[0]:
                v = float(max(val, 1e-6))
                self.per_symbol_z_scale_baseline[sid] = v
                self.per_symbol_z_scale_ready[sid] = True
        self.global_z_scale_baseline.fill_(float(max(global_baseline, 1e-6)))

    def freeze_z_scale_baselines(self) -> None:
        self.per_symbol_z_scale_frozen.copy_(self.per_symbol_z_scale_baseline)

    def _purity_scale_feature(
        self,
        z_scale: torch.Tensor,
        stock_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale = z_scale
        if scale.dim() == 1:
            scale = scale.unsqueeze(-1)
        if (
            self.use_relative_z_scale_for_purity
            and stock_ids is not None
            and self.labeling_mode == "per_stock"
        ):
            sids = stock_ids.to(scale.device).long().clamp(0, self.num_symbols - 1)
            ready = self.per_symbol_z_scale_ready[sids]
            per_b = self.per_symbol_z_scale_frozen[sids].unsqueeze(-1)
            global_b = self.global_z_scale_baseline.to(scale.device)
            baseline = torch.where(ready.unsqueeze(-1), per_b, global_b)
            scale = scale / baseline.clamp_min(1e-6)
        return torch.log1p(scale.clamp_min(0.0))

    def purity_features(
        self,
        z_latent: torch.Tensor,
        z_q: Optional[torch.Tensor] = None,
        z_scale: Optional[torch.Tensor] = None,
        stock_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Purity head input: continuous pre-VQ latent (default) or quantized token."""
        base = z_latent if self.purity_latent == "continuous" else z_q
        if base is None:
            raise ValueError("purity_latent='quantized' requires z_q")
        feat = base
        if self.use_magnitude_for_purity and z_scale is not None:
            feat = torch.cat([feat, self._purity_scale_feature(z_scale, stock_ids)], dim=-1)
        return self.purity_input_norm(feat)

    @staticmethod
    def purity_entropy_from_logits(token_logits: torch.Tensor) -> torch.Tensor:
        """Per-agent 3-way entropy averaged (max ≈ log(3) when uniform)."""
        n_agents = len(BEHAVIOR_AGENT_NAMES)
        ent = 0.0
        for i in range(n_agents):
            sl = agent_logit_slice(i)
            probs = F.softmax(token_logits[:, sl], dim=-1)
            ent = ent + -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
        return ent / n_agents

    def set_threshold_fit_ordinal(self, ordinal: int) -> None:
        self.threshold_fit_ordinal.fill_(int(ordinal))

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

        symbol_bounds[sid] is a tensor of shape [num_agents, 2].
        """
        for sid, b in symbol_bounds.items():
            if 0 <= sid < self.per_symbol_bounds.shape[0]:
                self.per_symbol_bounds[sid] = b.to(self.per_symbol_bounds.device)
                self.per_symbol_ready[sid] = True
        if symbol_counts:
            for sid, cnt in symbol_counts.items():
                if 0 <= sid < self.per_symbol_counts.shape[0]:
                    self.per_symbol_counts[sid] = int(cnt)

    def freeze_train_thresholds(self) -> None:
        """Snapshot train thresholds; training always uses frozen copy."""
        self.per_symbol_bounds_frozen.copy_(self.per_symbol_bounds)
        self.per_symbol_bounds_eval.copy_(self.per_symbol_bounds)
        self.thresholds_frozen.fill_(True)

    def begin_val_threshold_accum(self) -> None:
        self._val_proxy_accum.clear()

    def accumulate_val_proxies(
        self, stock_ids: torch.Tensor, proxy_mat: torch.Tensor
    ) -> None:
        """Collect validation proxies for epoch-end EMA threshold update."""
        if self.labeling_mode != "per_stock":
            return
        sids = stock_ids.detach().cpu().long()
        proxies = proxy_mat.detach().cpu()
        for i in range(sids.shape[0]):
            sid = int(sids[i].item())
            if 0 <= sid < self.per_symbol_bounds.shape[0]:
                self._val_proxy_accum.setdefault(sid, []).append(proxies[i])

    @torch.no_grad()
    def finalize_val_threshold_ema(self) -> int:
        """One EMA update per validation epoch (no per-batch noise)."""
        if self.labeling_mode != "per_stock" or not self._val_proxy_accum:
            return 0
        q = torch.tensor([1 / 3, 2 / 3])
        decay = self.val_threshold_ema_decay
        updated = 0
        for sid, chunks in self._val_proxy_accum.items():
            if not chunks:
                continue
            all_p = torch.stack(chunks, dim=0)
            if all_p.shape[0] < 8:
                continue
            per_agent_bounds = []
            for j in range(all_p.shape[1]):
                per_agent_bounds.append(torch.quantile(all_p[:, j], q))
            new_b = torch.stack(per_agent_bounds, dim=0).to(self.per_symbol_bounds_eval.device)
            old_b = self.per_symbol_bounds_eval[sid]
            self.per_symbol_bounds_eval[sid] = decay * old_b + (1.0 - decay) * new_b
            updated += 1
        self._val_proxy_accum.clear()
        return updated

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
        self,
        proxy_mat: torch.Tensor,
        stock_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        n_bins: int = 3,
        *,
        eval_thresholds: bool = False,
    ) -> torch.Tensor:
        """All agents at once: proxy_mat [B, A] -> labels [B, A, 3]."""
        device = proxy_mat.device
        bsz, n_agents = proxy_mat.shape
        sids = stock_ids.to(device).long().clamp(0, self.per_symbol_bounds.shape[0] - 1)

        if eval_thresholds and not self.training:
            bounds_bank = self.per_symbol_bounds_eval
        elif self.thresholds_frozen.item():
            bounds_bank = self.per_symbol_bounds_frozen
        else:
            bounds_bank = self.per_symbol_bounds
        per_bounds = bounds_bank[sids]  # [B, 8, 2]
        ready = self.per_symbol_ready[sids].float().unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

        global_bounds = torch.stack(
            [getattr(self, f"{name}_bounds") for name in BEHAVIOR_AGENT_NAMES], dim=0
        ).to(device)  # [8, 2]
        global_expanded = global_bounds.unsqueeze(0).expand(bsz, -1, -1)  # [B, 8, 2]
        counts = self.per_symbol_counts[sids].float().unsqueeze(-1).unsqueeze(-1)
        blend_mass = F.softplus(self.per_symbol_blend_log)
        conf = ready * (counts / (counts + blend_mass))
        if (
            not eval_thresholds
            and timestamps is not None
            and int(self.threshold_fit_ordinal.item()) > 0
        ):
            age = (
                timestamps.to(device).float() - float(self.threshold_fit_ordinal.item())
            ).clamp_min(0.0)
            half_life = max(float(self.threshold_decay_half_life), 1.0)
            time_decay = torch.exp(-age / half_life).view(-1, 1, 1)
            conf = conf * time_decay
        agent_conf = conf.expand(bsz, n_agents, 1)
        blend = self.regime_sensitive_blend.to(device)
        for j in self._regime_sensitive_agent_idx:
            agent_conf[:, j, :] = agent_conf[:, j, :] * blend
        bounds = agent_conf * per_bounds + (1.0 - agent_conf) * global_expanded

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
        timestamps: Optional[torch.Tensor] = None,
        z_scale: Optional[torch.Tensor] = None,
        z_latent: Optional[torch.Tensor] = None,
        *,
        eval_thresholds: bool = False,
        accumulate_val_thresholds: bool = False,
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

        has_proxies = "behavior_proxies" in raw_batch
        feat_key = f"{self.primary_scale}_features"
        has_ohlcv = self.primary_scale in raw_batch
        if not has_proxies and not has_ohlcv and feat_key not in raw_batch:
            return losses

        if z_latent is None:
            z_latent = z_q
        token_logits = self.behavior_proj(
            self.purity_features(z_latent, z_q=z_q, z_scale=z_scale, stock_ids=stock_ids)
        )

        if has_proxies:
            proxy_mat = raw_batch["behavior_proxies"]
            if proxy_mat.dim() == 1:
                proxy_mat = proxy_mat.unsqueeze(0)
        else:
            proxy_mat = compute_behavior_proxies_stacked(raw_batch[self.primary_scale])

        if not has_proxies or not self.proxies_label_ready:
            proxy_mat = transform_proxies_for_labeling(proxy_mat)

        if timestamps is None:
            timestamps = raw_batch.get("timestamps")
        if accumulate_val_thresholds and stock_ids is not None:
            self.accumulate_val_proxies(stock_ids, proxy_mat)

        if self.labeling_mode == "per_stock" and stock_ids is not None:
            all_labels = self._labels_all_agents(
                proxy_mat,
                stock_ids,
                timestamps=timestamps,
                eval_thresholds=eval_thresholds,
            )
            for i, name in enumerate(BEHAVIOR_AGENT_NAMES):
                label = all_labels[:, i, :]
                sl = agent_logit_slice(i)
                w = self._agent_weight(name)
                losses[PURITY_LOSS_KEYS[name]] = (
                    F.kl_div(F.log_softmax(token_logits[:, sl], dim=1), label, reduction="batchmean") * w
                )
        else:
            for i, name in enumerate(BEHAVIOR_AGENT_NAMES):
                bounds = getattr(self, f"{name}_bounds")
                label = self._labels_from_bounds(proxy_mat[:, i], bounds)
                sl = agent_logit_slice(i)
                w = self._agent_weight(name)
                losses[PURITY_LOSS_KEYS[name]] = (
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
        self.latent_scale_log = nn.Parameter(torch.tensor(-1.0))
        if enable_codebook_film:
            self.codebook_film = nn.Sequential(
                nn.Linear(2 * time_emb_dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim * 2),
            )
            self.codebook_scale_log = nn.Parameter(torch.tensor(-1.0))
            # sigmoid(0)=0.5；过小的初值（如 -2→0.12）会使码本调制长期近零
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
        latent_scale = F.softplus(self.latent_scale_log).clamp(max=0.5)
        gamma = torch.tanh(gamma) * latent_scale
        beta = torch.tanh(beta) * latent_scale
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
        use_codebook_film: bool = False,
        use_fine_vq: bool = False,
        num_fine_per_coarse: int = 16,
        use_adaptive_balance: bool = False,
        vq_mode: VQMode | str | None = None,
        use_cosine_vq: bool | None = None,
        use_normalized_vq: bool | None = None,
        threshold_decay_half_life: float = 252.0,
        val_threshold_ema_decay: float = 0.95,
        val_threshold_ema: bool = True,
        use_magnitude_for_purity: bool | None = None,
        use_relative_z_scale_for_purity: bool = True,
        purity_latent: str = "continuous",
    ):
        super().__init__()
        self.registry = registry
        self.primary_scale = primary_scale
        self.recon_weight = recon_weight
        self.use_codebook_film = use_codebook_film
        self.use_fine_vq = use_fine_vq
        self.use_adaptive_balance = use_adaptive_balance
        self.vq_mode: VQMode = normalize_vq_mode(
            vq_mode,
            use_cosine_vq=use_cosine_vq,
            use_normalized_vq=use_normalized_vq,
        )
        self.use_cosine_vq = self.vq_mode == "cosine"
        self._vq_backend = get_backend(self.vq_mode)
        self.val_threshold_ema = val_threshold_ema
        if use_magnitude_for_purity is None:
            use_magnitude_for_purity = self.use_cosine_vq
        self.use_magnitude_for_purity = use_magnitude_for_purity
        self.purity_latent = purity_latent
        primary_cfg = registry.get(primary_scale)
        self.raw_feat_dim = compute_raw_feat_dim(primary_cfg)
        self.recon_feat_dim = compute_recon_feat_dim(primary_cfg)

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
            vq_mode=self.vq_mode,
            use_fine_vq=use_fine_vq,
        )
        decoder_in = unified_dim + (1 if uses_magnitude_split(self.vq_mode) else 0)
        self.decoder = CausalDecoder(decoder_in, out_features=self.recon_feat_dim)
        self.z_norm_log_weight = nn.Parameter(torch.tensor(-3.0))
        self.beh_loss_fn = BehavioralPurityLoss(
            unified_dim=unified_dim,
            primary_scale=primary_scale,
            purity_weight=purity_weight,
            extended_purity_weight=extended_purity_weight,
            iso_weight=iso_weight,
            labeling_mode=labeling_mode,
            num_symbols=num_symbols,
            threshold_decay_half_life=threshold_decay_half_life,
            val_threshold_ema_decay=val_threshold_ema_decay,
            use_magnitude_for_purity=use_magnitude_for_purity,
            use_relative_z_scale_for_purity=use_relative_z_scale_for_purity,
            purity_latent=purity_latent,
        )
        self.task_balancer = AdaptiveTaskBalancer()

    def normalizer_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        """Export CausalFeatureComposer mean/std for non-precomputed inference."""
        out: dict[str, dict[str, torch.Tensor]] = {}
        for name, enc in self.fusion.encoders.items():
            comp = enc.composer
            if hasattr(comp, "feature_mean") and hasattr(comp, "feature_std"):
                out[name] = {
                    "feature_mean": comp.feature_mean.detach().cpu(),
                    "feature_std": comp.feature_std.detach().cpu(),
                }
        return out

    def load_normalizer_state_dict(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        for name, enc in self.fusion.encoders.items():
            if name not in state:
                continue
            comp = enc.composer
            if hasattr(comp, "feature_mean") and hasattr(comp, "feature_std"):
                comp.feature_mean.copy_(state[name]["feature_mean"].to(comp.feature_mean.device))
                comp.feature_std.copy_(state[name]["feature_std"].to(comp.feature_std.device))

    def encode(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        return self.fusion(batch)

    def _prepare_vq_inputs(
        self,
        z: torch.Tensor,
        stock_ids: Optional[torch.Tensor],
        timestamps: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Condition → LayerNorm → optional direction/scale split for cosine VQ."""
        z = self.conditioner(z, stock_ids, timestamps)
        z = self.pre_vq_norm(z)
        z_vq, z_scale, _ = self._vq_backend.prepare_latent(z)
        return z_vq, z_scale, z

    def quantize(
        self,
        z: torch.Tensor,
        use_fine: Optional[bool] = None,
        *,
        adapt: bool = False,
        stock_ids: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        skip_pre_vq_norm: bool = False,
        z_scale: Optional[torch.Tensor] = None,
    ):
        if use_fine is None:
            use_fine = self.use_fine_vq
        if not skip_pre_vq_norm and not self.use_cosine_vq:
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
            z_scale=z_scale,
        )

    def decode(self, z_q: torch.Tensor, z_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        z_q = self._vq_backend.merge_for_decode(z_q, z_scale)
        return self.decoder(z_q)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        use_fine: Optional[bool] = None,
        return_loss: bool = True,
        adapt_vq: bool = False,
        *,
        purity_eval_thresholds: bool = False,
        accumulate_val_thresholds: bool = False,
    ) -> Dict[str, torch.Tensor]:
        z, scale_feats, raw_feats = self.encode(batch)
        if z is None:
            return {"loss": torch.tensor(0.0, requires_grad=True)}

        stock_ids = batch.get("stock_ids")
        timestamps = batch.get("timestamps")
        z_vq, z_scale, z = self._prepare_vq_inputs(z, stock_ids, timestamps)

        z_q, coarse_idx, fine_idx, vq_loss, vq_metrics = self.quantize(
            z_vq,
            use_fine=use_fine,
            adapt=adapt_vq,
            stock_ids=stock_ids,
            timestamps=timestamps,
            skip_pre_vq_norm=True,
            z_scale=z_scale,
        )
        recon = self.decode(z_q, z_scale)

        primary_raw = raw_feats.get(self.primary_scale)
        if primary_raw is not None:
            recon_target = primary_raw[:, : self.recon_feat_dim].detach()
        else:
            recon_target = z.detach()

        out = {
            "z_continuous": z,
            "z_quantized": z_q,
            "coarse_token": coarse_idx,
            "fine_token": fine_idx,
        }
        for mk, mv in vq_metrics.items():
            if mk.startswith("vq_"):
                out[mk] = torch.tensor(mv, device=z.device)
            elif mk == "loss_diversity" and self.training:
                out[mk] = torch.tensor(mv, device=z.device)

        with torch.no_grad():
            out["z_norm_mean"] = z.norm(dim=1).mean()
            if z_scale is not None:
                out["z_scale_mean"] = z_scale.mean()
            out["codebook_norm_mean"] = self.vq.coarse_embed.weight.norm(dim=1).mean()
            r_flat = recon.reshape(recon.size(0), -1)
            t_flat = recon_target.reshape(recon_target.size(0), -1)
            if r_flat.size(1) > 0:
                out["recon_cosine"] = F.cosine_similarity(r_flat, t_flat, dim=1).mean()
            purity_feat = self.beh_loss_fn.purity_features(
                z, z_q=z_q, z_scale=z_scale, stock_ids=stock_ids
            )
            if z_scale is not None and self.beh_loss_fn.use_relative_z_scale_for_purity:
                out["z_scale_rel_mean"] = self.beh_loss_fn._purity_scale_feature(
                    z_scale, stock_ids
                ).exp().sub(1.0).mean()
            token_logits = self.beh_loss_fn.behavior_proj(purity_feat)
            out["purity_entropy"] = self.beh_loss_fn.purity_entropy_from_logits(token_logits)

        if return_loss:
            recon_loss = F.mse_loss(recon, recon_target.detach())
            stock_ids = batch.get("stock_ids")
            use_eval_thr = purity_eval_thresholds and not self.training
            beh_losses = self.beh_loss_fn(
                z_q,
                scale_feats,
                batch,
                stock_ids=stock_ids,
                timestamps=timestamps,
                z_scale=z_scale,
                z_latent=z,
                eval_thresholds=use_eval_thr,
                accumulate_val_thresholds=accumulate_val_thresholds and not self.training,
            )
            iso_loss = beh_losses.pop("iso", None)
            purity_loss = sum(beh_losses.values()) if beh_losses else torch.tensor(0.0, device=z.device)
            z_reg = F.softplus(self.z_norm_log_weight) * z.pow(2).mean()
            if self.use_adaptive_balance:
                balanced, balance_stats = self.task_balancer(vq_loss, purity_loss)
            else:
                balanced = vq_loss + purity_loss
                balance_stats = {}
            total_loss = self.recon_weight * recon_loss + balanced + z_reg
            if iso_loss is not None:
                total_loss = total_loss + iso_loss
            out.update(
                {
                    "loss": total_loss,
                    "loss_vq": vq_loss,
                    "loss_recon": recon_loss,
                    "loss_purity_total": purity_loss,
                    "loss_z_reg": z_reg,
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
        loader_or_ds,
        device: str = "cpu",
        max_samples: int = 5000,
        primary_scale: str | None = None,
        compute_transitions: bool = True,
    ):
        """返回 dict：semantics 为按 token 聚合的 DataFrame。"""
        import pandas as pd

        from .transition import TokenTransitionAnalyzer

        ps = primary_scale or self.primary_scale
        records = []
        token_sequences: List[List[int]] = []
        current_seq: List[int] = []
        seen = 0

        if hasattr(loader_or_ds, "iter_batches"):
            loader_or_ds.on_epoch_begin()
            batch_iter = loader_or_ds.iter_batches()
        elif isinstance(loader_or_ds, DataLoader):
            batch_iter = loader_or_ds
        else:
            raise TypeError("analyze_token_semantics expects DataLoader or iter_batches() dataset")

        for batch in batch_iter:
            if seen >= max_samples:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = self.forward(batch, return_loss=False)
            tokens = out["coarse_token"].cpu().numpy().tolist()

            for t_idx, tok in enumerate(tokens):
                rec = {"token": int(tok)}
                if "behavior_proxies" in batch:
                    proxies = batch["behavior_proxies"]
                    if proxies.dim() == 1:
                        proxies = proxies.unsqueeze(0)
                    for j, name in enumerate(BEHAVIOR_AGENT_NAMES):
                        rec[name] = float(proxies[t_idx, j].item())
                elif ps in batch and batch[ps].dim() == 3:
                    proxies = compute_behavior_proxies(batch[ps][t_idx : t_idx + 1])
                    for name in BEHAVIOR_AGENT_NAMES:
                        rec[name] = float(proxies[name][0])
                if ps in batch and batch[ps].dim() == 3 and batch[ps].shape[-1] >= 4:
                    close = batch[ps][t_idx, :, 3]
                    rec["net_return"] = float(close[-1] / close[0] - 1)
                records.append(rec)
                current_seq.append(int(tok))
            seen += len(tokens)
            if current_seq:
                token_sequences.append(current_seq)
                current_seq = []

        df = pd.DataFrame(records)
        result: dict = {
            "semantics": df.groupby("token").agg(["mean", "std", "count"]) if not df.empty else df,
            "n_records": len(records),
        }

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
    """在训练集上为全部行为代理固定 33%/66% 分位数阈值。

    当 labeling_mode == "per_stock" 时，同时预计算每只股票（symbol）的稳定分位数，
    避免训练过程中在线分位数波动导致的标签不稳定。
    """
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in BEHAVIOR_AGENT_NAMES}
    per_symbol_collected: dict[int, list[torch.Tensor]] = defaultdict(list)  # sid -> list of [8] proxy vectors
    ps = model.primary_scale
    use_per_symbol = model.beh_loss_fn.labeling_mode == "per_stock"

    label_ready = model.beh_loss_fn.proxies_label_ready
    gpu_batches = _loader_has_gpu_batches(loader)
    max_fit_ordinal = 0

    with torch.no_grad():
        for i, batch in enumerate(_iter_training_batches(loader)):
            if i >= max_batches:
                break
            if not gpu_batches:
                batch = _to_device(batch, device)
            if "behavior_proxies" not in batch:
                continue
            stacked = batch["behavior_proxies"]
            if not label_ready:
                stacked = transform_proxies_for_labeling(stacked)
            for j, name in enumerate(BEHAVIOR_AGENT_NAMES):
                collected[name].append(stacked[:, j].cpu())

            if use_per_symbol and "stock_ids" in batch:
                sids = batch["stock_ids"]
                proxies = stacked
                for sid in torch.unique(sids):
                    mask = sids == sid
                    per_symbol_collected[int(sid)].append(proxies[mask].cpu())
            if "timestamps" in batch:
                max_fit_ordinal = max(max_fit_ordinal, int(batch["timestamps"].max().item()))

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

    # 稳健性检查：若某代理的 33%/66% 分位数过于接近，发出警告
    for name, b in bounds.items():
        if (b[1] - b[0]).abs() < 1e-6:
            logger.warning("Purity threshold for '%s' is degenerate (%.6f/%.6f). Labels may be noisy.", name, b[0].item(), b[1].item())

    model.beh_loss_fn.set_thresholds(bounds)
    if max_fit_ordinal > 0:
        model.beh_loss_fn.set_threshold_fit_ordinal(max_fit_ordinal)
        logger.info(
            "Purity threshold fit ordinal=%d (per_stock decay half_life=%.0f trading days)",
            max_fit_ordinal,
            model.beh_loss_fn.threshold_decay_half_life,
        )

    # Per-symbol stable thresholds (only when requested)
    if use_per_symbol and per_symbol_collected:
        symbol_bounds: dict[int, torch.Tensor] = {}
        symbol_counts: dict[int, int] = {}
        agent_names = BEHAVIOR_AGENT_NAMES
        sid_sizes = [len(torch.cat(pl, dim=0)) for pl in per_symbol_collected.values() if pl]
        adaptive_min = max(8, int(torch.tensor(sid_sizes).median().item() * 0.1)) if sid_sizes else 8
        for sid, proxy_lists in per_symbol_collected.items():
            if len(proxy_lists) == 0:
                continue
            all_proxies = torch.cat(proxy_lists, dim=0)  # [n_samples_for_sid, 8]
            n_sid = int(all_proxies.shape[0])
            if n_sid < adaptive_min:
                continue
            per_agent_bounds = []
            for j in range(len(agent_names)):
                qs = torch.quantile(all_proxies[:, j], q)
                per_agent_bounds.append(qs)
            symbol_bounds[sid] = torch.stack(per_agent_bounds)  # [8, 2]
            symbol_counts[sid] = n_sid
        if symbol_bounds:
            model.beh_loss_fn.set_per_symbol_thresholds(symbol_bounds, symbol_counts=symbol_counts)
            logger.info(
                "Per-symbol purity thresholds precomputed for %d symbols (min_samples=%d, blend learns at train)",
                len(symbol_bounds),
                adaptive_min,
            )

    if model.beh_loss_fn.labeling_mode == "per_stock":
        model.beh_loss_fn.freeze_train_thresholds()
        logger.info("Per-symbol train thresholds frozen (val EMA updates eval copy only)")

    core = {k: bounds[k].tolist() for k in CORE_AGENTS}
    ext = {k: bounds[k].tolist() for k in BehavioralPurityLoss.EXTENDED_AGENTS}
    logger.info(
        "PurityThresholds fitted on %d samples | core=%s | extended=%s",
        n_samples,
        core,
        ext,
    )


def precompute_z_scale_baselines(
    model: BPCv2,
    loader: DataLoader,
    device: str = "cpu",
    max_batches: int = 500,
) -> None:
    """Per-symbol z_scale 中位数基线，供纯度头相对幅度输入。"""
    if not model.beh_loss_fn.use_magnitude_for_purity:
        return
    if not model.beh_loss_fn.use_relative_z_scale_for_purity:
        return
    model.eval()
    per_symbol: dict[int, list[float]] = defaultdict(list)
    all_scales: list[float] = []
    gpu_batches = _loader_has_gpu_batches(loader)

    with torch.no_grad():
        for i, batch in enumerate(_iter_training_batches(loader)):
            if i >= max_batches:
                break
            if not gpu_batches:
                batch = _to_device(batch, device)
            if "stock_ids" not in batch:
                continue
            z, _, _ = model.encode(batch)
            if z is None:
                continue
            z = model.conditioner(z, batch.get("stock_ids"), batch.get("timestamps"))
            z = model.pre_vq_norm(z)
            scales = z.norm(dim=1)
            sids = batch["stock_ids"].detach().cpu().long()
            for j in range(scales.shape[0]):
                sid = int(sids[j].item())
                val = float(scales[j].item())
                per_symbol[sid].append(val)
                all_scales.append(val)

    if not all_scales:
        logger.warning("z_scale baselines: no data, using global=1.0")
        return

    global_baseline = float(torch.tensor(all_scales).median().item())
    symbol_baselines = {
        sid: float(torch.tensor(vals).median().item()) for sid, vals in per_symbol.items() if vals
    }
    model.beh_loss_fn.set_z_scale_baselines(symbol_baselines, global_baseline=global_baseline)
    model.beh_loss_fn.freeze_z_scale_baselines()
    logger.info(
        "z_scale baselines: global_median=%.4f | per_symbol=%d symbols",
        global_baseline,
        len(symbol_baselines),
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
        if (
            k.startswith("loss")
            or k.startswith("vq_")
            or k.startswith("balance_weight_")
            or k.startswith("film_")
            or k in (
                "z_norm_mean",
                "z_scale_mean",
                "codebook_norm_mean",
                "recon_cosine",
                "purity_entropy",
                "vq_dir_residual_mean",
                "z_scale_rel_mean",
            )
        ):
            if isinstance(v, torch.Tensor):
                total[k] += v.item()
            elif isinstance(v, (int, float)):
                total[k] += float(v)
    return count + 1


def _codebook_usage_metrics(model: BPCv2, epoch_token_ids: set[int] | None = None) -> dict[str, float]:
    """EMA 全 epoch 码本使用率，补充 per-batch vq_usage_rate。"""
    num_coarse = model.vq.num_coarse
    out: dict[str, float] = {}
    with torch.no_grad():
        cs = model.vq.ema_cluster_size
        total = cs.sum().clamp_min(1e-8)
        frac = cs / total
        active = int((frac > 1e-4).sum().item())
        out["vq_ema_active_codes"] = float(active)
        out["vq_ema_usage_rate"] = active / num_coarse
        if epoch_token_ids is not None:
            out["vq_epoch_unique_tokens"] = float(len(epoch_token_ids))
            out["vq_epoch_usage_rate"] = len(epoch_token_ids) / num_coarse
    return out


def _loader_has_gpu_batches(loader) -> bool:
    if hasattr(loader, "iter_batches"):
        return True
    ds = getattr(loader, "dataset", None)
    return ds is not None and hasattr(ds, "iter_batches")


def _iter_training_batches(loader):
    if hasattr(loader, "iter_batches"):
        if hasattr(loader, "on_epoch_begin"):
            loader.on_epoch_begin()
        yield from loader.iter_batches()
        return
    ds = getattr(loader, "dataset", loader)
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

    gpu_batches = _loader_has_gpu_batches(loader)

    # 第一遍：收集距离统计（与 forward 一致：latent FiLM → pre_vq_norm → 可选码本 FiLM）
    for i, batch in enumerate(_iter_training_batches(loader)):
        if max_batches is not None and i >= max_batches:
            break
        if not gpu_batches:
            batch = _to_device(batch, device, non_blocking=non_blocking)
        with torch.no_grad():
            z, _, _ = model.encode(batch)
            if z is None:
                continue
            stock_ids = batch.get("stock_ids")
            timestamps = batch.get("timestamps")
            z_vq, _, _ = model._prepare_vq_inputs(z, stock_ids, timestamps)
            cb_gamma, cb_beta = None, None
            if model.use_codebook_film:
                cb_gamma, cb_beta = model.conditioner.codebook_modulation(
                    stock_ids, timestamps, z_vq.shape[0], z_vq.device
                )
            dist_mat = model.vq._coarse_distances(z_vq, cb_gamma, cb_beta)
            distances = dist_mat.min(dim=1).values
            if cb_gamma is not None and not model.vq.use_cosine_vq:
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
    for i, batch in enumerate(_iter_training_batches(loader)):
        if max_batches is not None and i >= max_batches:
            break
        if not gpu_batches:
            batch = _to_device(batch, device, non_blocking=non_blocking)
        with torch.no_grad():
            z, _, _ = model.encode(batch)
            if z is None:
                continue
            stock_ids = batch.get("stock_ids")
            timestamps = batch.get("timestamps")
            z_vq, _, _ = model._prepare_vq_inputs(z, stock_ids, timestamps)
            cb_gamma, cb_beta = None, None
            if model.use_codebook_film:
                cb_gamma, cb_beta = model.conditioner.codebook_modulation(
                    stock_ids, timestamps, z_vq.shape[0], z_vq.device
                )
            dist_mat = model.vq._coarse_distances(z_vq, cb_gamma, cb_beta)
            min_dist, coarse_idx = dist_mat.min(dim=1)
            if cb_gamma is not None and not model.vq.use_cosine_vq:
                min_dist = min_dist.sqrt()
            mask = min_dist >= thresh
            if mask.any():
                model.vq._adapt_codebook(z_vq[mask], coarse_idx[mask])
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
    skip_device_transfer: bool = False,
    val_threshold_ema: bool = False,
) -> Dict[str, float]:
    model.eval()
    total_losses: dict[str, float] = defaultdict(float)
    count = 0
    use_amp = amp and device.startswith("cuda")
    gpu_batches = skip_device_transfer or _loader_has_gpu_batches(loader)
    use_val_thr = val_threshold_ema and model.val_threshold_ema
    if use_val_thr:
        model.beh_loss_fn.begin_val_threshold_accum()
    for i, batch in enumerate(_iter_training_batches(loader)):
        if max_batches is not None and i >= max_batches:
            break
        if not gpu_batches:
            batch = _to_device(batch, device, non_blocking=non_blocking)
        with torch.autocast(device_type="cuda", enabled=use_amp):
            out = model(
                batch,
                return_loss=True,
                adapt_vq=False,
                purity_eval_thresholds=use_val_thr,
                accumulate_val_thresholds=use_val_thr,
            )
        count = _aggregate_losses(out, total_losses, count)
    if use_val_thr:
        n_updated = model.beh_loss_fn.finalize_val_threshold_ema()
        if n_updated:
            logger.info("Val threshold EMA updated for %d symbols", n_updated)
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
    profile_batches: int = 0,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.train()
    total_losses: dict[str, float] = defaultdict(float)
    grad_norm_sum = 0.0
    count = 0
    use_amp = amp and device.startswith("cuda")
    if use_amp and scaler is None:
        scaler = torch.amp.GradScaler("cuda")
    gpu_batches = _loader_has_gpu_batches(loader)
    profiled = 0
    epoch_token_ids: set[int] = set()

    for batch_i, batch in enumerate(_iter_training_batches(loader)):
        if max_batches is not None and batch_i >= max_batches:
            break
        if not gpu_batches:
            batch = _to_device(batch, device, non_blocking=non_blocking)
        if profile_batches > 0 and profiled < profile_batches:
            profiled += 1
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
        if "coarse_token" in out:
            epoch_token_ids.update(int(t) for t in out["coarse_token"].detach().cpu().unique().tolist())
        count = _aggregate_losses(out, total_losses, count)

    if count == 0:
        return {}
    metrics = {k: v / count for k, v in total_losses.items()}
    metrics["grad_norm"] = grad_norm_sum / count
    metrics.update(_codebook_usage_metrics(model, epoch_token_ids))
    return metrics