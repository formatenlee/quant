"""
BPC-v3 模型：复用 bpc.model 骨架，替换特征/行为/纯度路径（不修改 bpc 源码）。

相对 BPC-v2：
- 输入：相对化 OHLCV（bpc_v3/ohlcv_relative.py）
- 预计算 26 维 = struct(21) + behavior(5)，含 trend_structure
- 行为代理全符号化 + 固定阈值（无 CausalNormalizer / 标签 z-score）
- 默认跳过 precompute_normalizers；encoder 为 Linear+GELU（无 LayerNorm）
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from quant_cursor.bpc import model as base
from quant_cursor.bpc.behavioral_vq import BehavioralVQ
from quant_cursor.bpc_v3.features import extract_group_features
from quant_cursor.bpc_v3.behavior_features import (
    BEHAVIOR_AGENT_NAMES,
    BEHAVIOR_LOGITS_DIM,
    CORE_AGENTS,
    DEFAULT_SYMBOLIC_LABEL_TEMPERATURE,
    EXTENDED_AGENTS,
    PURITY_LOSS_KEYS,
    STRUCTURAL_PROXY_INDICES,
    SYMBOLIC_THRESHOLDS,
    agent_logit_slice,
    compute_behavior_proxies_stacked,
    symbolic_labels,
)
from quant_cursor.bpc_v3.feature_dims import (
    DAY_FEATURE_SLICES,
    DAY_FULL_FEAT_DIM,
    DAY_STRUCT_FEAT_DIM,
    GROUP_DIM_MAP,
    STRUCT_FEATURE_SCALE,
    WEEK_FEAT_DIM,
)
from quant_cursor.bpc_v3.vq_backend import VQMode, get_backend, normalize_vq_mode, uses_magnitude_split

logger = logging.getLogger(__name__)

ScaleConfig = base.ScaleConfig
ScaleRegistry = base.ScaleRegistry
PrecomputedFeatureComposer = base.PrecomputedFeatureComposer
CausalNormalizer = base.CausalNormalizer
CausalDecoder = base.CausalDecoder
from quant_cursor.bpc_v3.film import SymbolTimeFiLM
AdaptiveTaskBalancer = base.AdaptiveTaskBalancer

DEFAULT_DAY_LOOKBACK = base.DEFAULT_DAY_LOOKBACK
DEFAULT_WEEK_LOOKBACK = base.DEFAULT_WEEK_LOOKBACK


class V3PrecomputedFeatureComposer(nn.Module):
    """物化结构特征 → Linear 投影（无 LayerNorm；不含 behavior 维，防纯度标签泄漏）。"""

    def __init__(self, struct_dim: int, out_dim: int):
        super().__init__()
        self.struct_dim = struct_dim
        self.proj = nn.Linear(struct_dim, out_dim)

    def forward(self, x: torch.Tensor, timestamps=None, return_raw: bool = False):
        if x.dim() == 3:
            raise ValueError("V3PrecomputedFeatureComposer expects [B, D] features, not OHLCV windows")
        x_struct = x[..., : self.struct_dim]
        proj = self.proj(x_struct)
        if return_raw:
            return proj, x
        return proj


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
            "trend_structure",
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


class CausalFeatureComposer(nn.Module):
    """因果特征组合器：统一走相对化 OHLCV → extract_group_features。"""

    def __init__(self, feature_groups: List[str], out_dim: int, *, scale_name: str = "day"):
        super().__init__()
        self.groups = feature_groups
        self.scale_name = scale_name
        self.group_dims = {k: GROUP_DIM_MAP[k] for k in feature_groups if k in GROUP_DIM_MAP}
        self.raw_feat_dim = sum(self.group_dims.values())
        self.proj = nn.Linear(self.raw_feat_dim, out_dim)
        self.register_buffer("feature_mean", torch.zeros(self.raw_feat_dim))
        self.register_buffer("feature_std", torch.ones(self.raw_feat_dim))

    def extract_raw_features(
        self,
        x: torch.Tensor,
        *,
        prev_bar: Optional[torch.Tensor] = None,
        vol_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feat = extract_group_features(
            x,
            self.groups,
            prev_bar=prev_bar,
            vol_context=vol_context,
            scale_name=self.scale_name,
        )
        feat = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        return feat.clamp(-20.0, 20.0)

    def normalize_features(self, raw: torch.Tensor) -> torch.Tensor:
        """v3 跳过 CausalNormalizer 拟合；mean=0/std=1 时等价于恒等，不做 z-score。"""
        norm = (raw - self.feature_mean) / self.feature_std.clamp_min(1e-6)
        return torch.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)

    def forward(
        self,
        x: torch.Tensor,
        timestamps=None,
        return_raw: bool = False,
        *,
        prev_bar: Optional[torch.Tensor] = None,
        vol_context: Optional[torch.Tensor] = None,
    ):
        raw = self.extract_raw_features(x, prev_bar=prev_bar, vol_context=vol_context)
        proj = self.proj(raw)
        if return_raw:
            return proj, raw
        return proj


class ScaleEncoder(nn.Module):
    def __init__(self, cfg: ScaleConfig, feat_dim: int = 64):
        super().__init__()
        self.cfg = cfg
        self.use_precomputed = "precomputed_features" in cfg.feature_groups
        if self.use_precomputed:
            if self.cfg.name == "day":
                self.composer = V3PrecomputedFeatureComposer(DAY_STRUCT_FEAT_DIM, feat_dim)
            else:
                self.composer = V3PrecomputedFeatureComposer(WEEK_FEAT_DIM, feat_dim)
        else:
            self.composer = CausalFeatureComposer(
                feature_groups=cfg.feature_groups, out_dim=feat_dim, scale_name=cfg.name
            )
        self.aggregator = nn.Sequential(
            nn.Linear(feat_dim, cfg.encoder_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(
        self,
        x: torch.Tensor,
        timestamps=None,
        *,
        prev_bar: Optional[torch.Tensor] = None,
        vol_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_precomputed:
            proj, raw = self.composer(x, timestamps, return_raw=True)
        else:
            proj, raw = self.composer(
                x,
                timestamps,
                return_raw=True,
                prev_bar=prev_bar,
                vol_context=vol_context,
            )
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
            prev_bar = batch.get(f"{name}_prev_bar")
            vol_context = batch.get("vol_context") if name == "day" else None
            h, raw = enc(x, timestamps=batch.get("timestamps"), prev_bar=prev_bar, vol_context=vol_context)
            proj = getattr(self, f"proj_{name}")
            scale_feats[name] = proj(h)
            raw_feats[name] = raw
        if not scale_feats:
            return None, {}, {}
        weights = F.softmax(self.scale_weights[: len(scale_feats)], dim=0)
        fused = sum(scale_feats[name] * w for name, w in zip(scale_feats.keys(), weights))
        return fused, scale_feats, raw_feats


class BehavioralPurityLoss(nn.Module):
    """BPC-v3 纯度损失：固定符号阈值，无 proxy z-score / global 归一化。"""

    CORE_AGENTS = CORE_AGENTS
    EXTENDED_AGENTS = EXTENDED_AGENTS
    SYMBOLIC_AGENTS = frozenset(SYMBOLIC_THRESHOLDS.keys())

    @classmethod
    def _all_agents_symbolic(cls) -> bool:
        return len(cls.SYMBOLIC_AGENTS) >= len(BEHAVIOR_AGENT_NAMES)

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
        val_threshold_ema_decay: float = 0.90,
        use_magnitude_for_purity: bool = False,
        use_relative_z_scale_for_purity: bool = True,
        purity_latent: str = "quantized",
        behavior_dropout: float = 0.45,
        label_temperature: float = DEFAULT_SYMBOLIC_LABEL_TEMPERATURE,
    ):
        super().__init__()
        self.primary_scale = primary_scale
        self.purity_weight = purity_weight
        self.extended_purity_weight = extended_purity_weight
        self.iso_weight = iso_weight
        self.labeling_mode = labeling_mode
        self.num_symbols = num_symbols
        self.threshold_decay_half_life = threshold_decay_half_life
        self.val_threshold_ema_decay = val_threshold_ema_decay
        self.use_magnitude_for_purity = use_magnitude_for_purity
        self.use_relative_z_scale_for_purity = use_relative_z_scale_for_purity
        if purity_latent not in ("continuous", "quantized"):
            raise ValueError(f"purity_latent must be 'continuous' or 'quantized', got {purity_latent!r}")
        self.purity_latent = purity_latent
        self.label_temperature = float(label_temperature)
        self.proxies_label_ready = False
        self.register_buffer("threshold_fit_ordinal", torch.tensor(0, dtype=torch.long))
        self.register_buffer("thresholds_frozen", torch.tensor(False))
        self._val_proxy_accum: dict[int, list[torch.Tensor]] = {}
        purity_in = unified_dim + (1 if use_magnitude_for_purity else 0)
        self.behavior_proj = nn.Sequential(
            nn.Linear(purity_in, 96),
            nn.GELU(),
            nn.Dropout(behavior_dropout),
            nn.Linear(96, BEHAVIOR_LOGITS_DIM),
        )
        for name in BEHAVIOR_AGENT_NAMES:
            self.register_buffer(f"{name}_bounds", torch.zeros(2))
        self.register_buffer("thresholds_ready", torch.tensor(False))
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
        self.per_symbol_blend_log = nn.Parameter(torch.tensor(4.0))
        self.register_buffer("proxy_mean", torch.zeros(n_agents))
        self.register_buffer("proxy_std", torch.ones(n_agents))
        self.register_buffer("global_norm_ready", torch.tensor(False))

    def set_global_normalizers(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """仅 legacy structural 代理需要；全符号化时保持 identity。"""
        if not STRUCTURAL_PROXY_INDICES:
            self.global_norm_ready.fill_(False)
            return
        self.proxy_mean.copy_(mean.flatten()[: len(BEHAVIOR_AGENT_NAMES)].to(self.proxy_mean.device))
        self.proxy_std.copy_(std.flatten()[: len(BEHAVIOR_AGENT_NAMES)].clamp_min(1e-6).to(self.proxy_std.device))
        self.global_norm_ready.fill_(True)

    def _proxies_for_labeling(self, proxy_mat: torch.Tensor) -> torch.Tensor:
        if not STRUCTURAL_PROXY_INDICES:
            return proxy_mat
        if self.global_norm_ready.item():
            out = proxy_mat.clone()
            for j in STRUCTURAL_PROXY_INDICES:
                out[:, j] = (
                    (out[:, j] - self.proxy_mean[j]) / self.proxy_std[j].clamp_min(1e-6)
                )
            return out
        return proxy_mat

    def set_z_scale_baselines(self, symbol_baselines: dict[int, float], *, global_baseline: float) -> None:
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
        scale = z_scale.unsqueeze(-1) if z_scale.dim() == 1 else z_scale
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
        # 已是相对 baseline 的比率，不再 log1p 压缩
        return (scale - 1.0).clamp(-2.0, 3.0)

    def purity_features(
        self,
        z_latent: torch.Tensor,
        z_q: Optional[torch.Tensor] = None,
        z_scale: Optional[torch.Tensor] = None,
        stock_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base = z_latent if self.purity_latent == "continuous" else z_q
        if base is None:
            raise ValueError("purity_latent='quantized' requires z_q")
        feat = base
        if self.use_magnitude_for_purity and z_scale is not None:
            feat = torch.cat([feat, self._purity_scale_feature(z_scale, stock_ids)], dim=-1)
        return feat

    @staticmethod
    def purity_entropy_from_logits(token_logits: torch.Tensor) -> torch.Tensor:
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
        for sid, b in symbol_bounds.items():
            if 0 <= sid < self.per_symbol_bounds.shape[0]:
                self.per_symbol_bounds[sid] = b.to(self.per_symbol_bounds.device)
                self.per_symbol_ready[sid] = True
        if symbol_counts:
            for sid, cnt in symbol_counts.items():
                if 0 <= sid < self.per_symbol_counts.shape[0]:
                    self.per_symbol_counts[sid] = int(cnt)

    def freeze_train_thresholds(self) -> None:
        self.per_symbol_bounds_frozen.copy_(self.per_symbol_bounds)
        self.per_symbol_bounds_eval.copy_(self.per_symbol_bounds)
        self.thresholds_frozen.fill_(True)

    def begin_val_threshold_accum(self) -> None:
        self._val_proxy_accum.clear()

    def accumulate_val_proxies(self, stock_ids: torch.Tensor, proxy_mat: torch.Tensor) -> None:
        if self._all_agents_symbolic() or self.labeling_mode != "per_stock":
            return
        sids = stock_ids.detach().cpu().long()
        proxies = proxy_mat.detach().cpu()
        for i in range(sids.shape[0]):
            sid = int(sids[i].item())
            if 0 <= sid < self.per_symbol_bounds.shape[0]:
                self._val_proxy_accum.setdefault(sid, []).append(proxies[i])

    @torch.no_grad()
    def finalize_val_threshold_ema(self) -> int:
        if self._all_agents_symbolic() or self.labeling_mode != "per_stock" or not self._val_proxy_accum:
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
            for j, name in enumerate(BEHAVIOR_AGENT_NAMES):
                if name in self.SYMBOLIC_AGENTS:
                    lo, hi = SYMBOLIC_THRESHOLDS[name]
                    per_agent_bounds.append(torch.tensor([lo, hi], device=all_p.device))
                else:
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
        inner = torch.maximum(inner, inner.cummax(dim=-1).values)
        bins = (values.unsqueeze(-1) > inner).long().sum(dim=-1).clamp(max=2)
        return F.one_hot(bins, 3).float()

    def _labels_from_bounds(self, values: torch.Tensor, bounds: torch.Tensor, n_bins: int = 3) -> torch.Tensor:
        if not self.thresholds_ready.item():
            return self._labels_batch_quantile(values, n_bins)
        inner = bounds.to(values.device).unsqueeze(0).expand(values.shape[0], -1)
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
        if inner.numel() > 1 and (inner[-1] - inner[0]).abs() < 1e-6:
            mid = torch.full((values.shape[0],), n_bins // 2, dtype=torch.long, device=values.device)
            return F.one_hot(mid, n_bins).float()
        inner = inner.unsqueeze(0).expand(values.shape[0], -1)
        return self._bucketize_3class(values, inner)

    def _labels_for_agent(self, name: str, values: torch.Tensor, bounds: Optional[torch.Tensor] = None) -> torch.Tensor:
        if name in SYMBOLIC_THRESHOLDS:
            lo, hi = SYMBOLIC_THRESHOLDS[name]
            return symbolic_labels(values, lo, hi, temperature=self.label_temperature)
        if bounds is None:
            raise ValueError(f"structural agent {name!r} requires bounds")
        return self._labels_from_bounds(values, bounds)

    def _structural_bounds_for_agent(
        self,
        agent_index: int,
        name: str,
        stock_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor],
        *,
        eval_thresholds: bool,
    ) -> torch.Tensor:
        device = stock_ids.device
        bsz = stock_ids.shape[0]
        sids = stock_ids.to(device).long().clamp(0, self.per_symbol_bounds.shape[0] - 1)

        if eval_thresholds and not self.training:
            bounds_bank = self.per_symbol_bounds_eval
        elif self.thresholds_frozen.item():
            bounds_bank = self.per_symbol_bounds_frozen
        else:
            bounds_bank = self.per_symbol_bounds
        per_bounds = bounds_bank[sids, agent_index]
        ready = self.per_symbol_ready[sids].float().unsqueeze(-1)
        global_bounds = getattr(self, f"{name}_bounds").to(device)
        global_expanded = global_bounds.unsqueeze(0).expand(bsz, -1)
        counts = self.per_symbol_counts[sids].float().unsqueeze(-1)
        blend_mass = F.softplus(self.per_symbol_blend_log)
        conf = ready * (counts / (counts + blend_mass))
        if not eval_thresholds and timestamps is not None and int(self.threshold_fit_ordinal.item()) > 0:
            age = (timestamps.to(device).float() - float(self.threshold_fit_ordinal.item())).clamp_min(0.0)
            half_life = max(float(self.threshold_decay_half_life), 1.0)
            conf = conf * torch.exp(-age / half_life).unsqueeze(-1)
        return conf * per_bounds + (1.0 - conf) * global_expanded

    def _labels_all_agents(
        self,
        proxy_mat: torch.Tensor,
        stock_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        n_bins: int = 3,
        *,
        eval_thresholds: bool = False,
    ) -> torch.Tensor:
        device = proxy_mat.device
        labels = []
        for j, name in enumerate(BEHAVIOR_AGENT_NAMES):
            if name in self.SYMBOLIC_AGENTS:
                labels.append(self._labels_for_agent(name, proxy_mat[:, j]))
                continue
            # v8：全代理符号化；以下 per-stock 分位数分支当前不可达（保留供未来非符号代理）
            bounds = self._structural_bounds_for_agent(
                j,
                name,
                stock_ids,
                timestamps,
                eval_thresholds=eval_thresholds,
            )
            inner = torch.maximum(bounds, bounds.cummax(dim=-1).values)
            degenerate = (inner[:, 1] - inner[:, 0]).abs() < 1e-6
            if degenerate.any():
                for mask_idx in torch.where(degenerate)[0]:
                    vals = proxy_mat[mask_idx, j].unsqueeze(0)
                    inner[mask_idx] = torch.quantile(
                        vals, torch.tensor([1 / 3, 2 / 3], device=device)
                    )
            bins = (proxy_mat[:, j].unsqueeze(-1) > inner).long().sum(dim=-1).clamp(max=2)
            labels.append(F.one_hot(bins, n_bins).float())
        return torch.stack(labels, dim=1)

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
            vol_ctx = raw_batch.get("vol_context")
            prev_bar = raw_batch.get(f"{self.primary_scale}_prev_bar")
            proxy_mat = compute_behavior_proxies_stacked(
                raw_batch[self.primary_scale], vol_ctx, prev_bar=prev_bar
            )

        proxy_mat = self._proxies_for_labeling(proxy_mat)

        if timestamps is None:
            timestamps = raw_batch.get("timestamps")
        if accumulate_val_thresholds and stock_ids is not None:
            self.accumulate_val_proxies(stock_ids, proxy_mat)

        if self.labeling_mode == "per_stock" and stock_ids is not None and not self._all_agents_symbolic():
            all_labels = self._labels_all_agents(
                proxy_mat,
                stock_ids,
                timestamps=timestamps,
                eval_thresholds=eval_thresholds,
            )
            for i, name in enumerate(BEHAVIOR_AGENT_NAMES):
                sl = agent_logit_slice(i)
                w = self._agent_weight(name)
                losses[PURITY_LOSS_KEYS[name]] = (
                    F.kl_div(
                        F.log_softmax(token_logits[:, sl], dim=1),
                        all_labels[:, i, :],
                        reduction="batchmean",
                    )
                    * w
                )
        else:
            for i, name in enumerate(BEHAVIOR_AGENT_NAMES):
                bounds = getattr(self, f"{name}_bounds")
                label = self._labels_for_agent(name, proxy_mat[:, i], bounds)
                sl = agent_logit_slice(i)
                w = self._agent_weight(name)
                losses[PURITY_LOSS_KEYS[name]] = (
                    F.kl_div(F.log_softmax(token_logits[:, sl], dim=1), label, reduction="batchmean") * w
                )

        return losses


class BPCv3(nn.Module):
    """BPC-v3 主模型。"""

    def __init__(
        self,
        registry: ScaleRegistry = DEFAULT_REGISTRY,
        unified_dim: int = 96,
        num_coarse: int = 256,
        commitment_cost: float = 0.6,
        primary_scale: str = "day",
        recon_weight: float = 0.5,
        purity_weight: float = 0.40,
        extended_purity_weight: float = 0.15,
        iso_weight: float = 0.05,
        diversity_weight: float = 0.25,
        vq_adapt_lr: float = 1e-5,
        vq_dead_code_threshold: float = 0.01,
        labeling_mode: str = "global",
        num_symbols: int = 10000,
        use_codebook_film: bool = True,
        use_fine_vq: bool = False,
        num_fine_per_coarse: int = 16,
        use_adaptive_balance: bool = False,
        vq_mode: VQMode | str | None = None,
        use_cosine_vq: bool | None = None,
        use_normalized_vq: bool | None = None,
        threshold_decay_half_life: float = 252.0,
        val_threshold_ema_decay: float = 0.90,
        val_threshold_ema: bool = True,
        use_magnitude_for_purity: bool | None = None,
        use_relative_z_scale_for_purity: bool = True,
        purity_latent: str = "quantized",
        behavior_dropout: float = 0.45,
        label_temperature: float = DEFAULT_SYMBOLIC_LABEL_TEMPERATURE,
        recon_detach_scale: bool = True,
        z_reg_weight: float = 0.01,
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
        self.recon_detach_scale = recon_detach_scale
        self.z_reg_weight = float(z_reg_weight)
        primary_cfg = registry.get(primary_scale)
        self.raw_feat_dim = compute_raw_feat_dim(primary_cfg)
        self.recon_feat_dim = compute_recon_feat_dim(primary_cfg)

        self.fusion = MultiScaleFusion(registry, unified_dim=unified_dim)
        # v3：跳过 VQ 前 LayerNorm；cosine VQ 已单位化方向，FiLM 负责调制
        self.pre_vq_norm = nn.Identity()
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
            behavior_dropout=behavior_dropout,
            label_temperature=label_temperature,
        )
        self.task_balancer = AdaptiveTaskBalancer()

    def decode(self, z_q: torch.Tensor, z_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        if z_scale is not None and self.recon_detach_scale:
            z_scale = z_scale.detach()
        z_in = self._vq_backend.merge_for_decode(z_q, z_scale)
        return self.decoder(z_in)

    @staticmethod
    def _trend_sign_accuracy(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        trend_structure 第 0 维方向一致率（仅监控指标，不参与 loss）。

        torch.sign 梯度恒为 0，故不可作为训练损失；调用处必须在 no_grad 内。
        若需可微方向约束，请用 tanh(k·pred)·tanh(k·tgt) 等软符号替代。
        """
        sl = DAY_FEATURE_SLICES["trend_structure"]
        pred = recon[:, sl.start]
        tgt = target[:, sl.start]
        with torch.no_grad():
            return (torch.sign(pred) == torch.sign(tgt)).float().mean()

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

        out: Dict[str, torch.Tensor] = {
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
                sl = min(r_flat.size(1), len(STRUCT_FEATURE_SCALE))
                if sl > 0:
                    feat_w = torch.tensor(
                        STRUCT_FEATURE_SCALE[:sl],
                        device=z.device,
                        dtype=r_flat.dtype,
                    )
                    out["recon_cosine_balanced"] = F.cosine_similarity(
                        r_flat[:, :sl] * feat_w,
                        t_flat[:, :sl] * feat_w,
                        dim=1,
                    ).mean()
            purity_feat = self.beh_loss_fn.purity_features(
                z, z_q=z_q, z_scale=z_scale, stock_ids=stock_ids
            )
            if z_scale is not None and self.beh_loss_fn.use_relative_z_scale_for_purity:
                out["z_scale_rel_mean"] = self.beh_loss_fn._purity_scale_feature(
                    z_scale, stock_ids
                ).exp().sub(1.0).mean()
            token_logits = self.beh_loss_fn.behavior_proj(purity_feat)
            out["purity_entropy"] = self.beh_loss_fn.purity_entropy_from_logits(token_logits)
            sign_acc = self._trend_sign_accuracy(recon, recon_target)
            out["recon_trend_sign_acc"] = sign_acc  # 已在 _trend_sign_accuracy 内 no_grad

        if return_loss:
            recon_loss = F.mse_loss(recon, recon_target.detach())
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
            z_reg = self.z_reg_weight * z.pow(2).mean()
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

    normalizer_state_dict = base.BPCv2.normalizer_state_dict
    load_normalizer_state_dict = base.BPCv2.load_normalizer_state_dict
    encode = base.BPCv2.encode
    _prepare_vq_inputs = base.BPCv2._prepare_vq_inputs
    quantize = base.BPCv2.quantize
    save_behavioral_ontology = base.BPCv2.save_behavioral_ontology
    analyze_token_semantics = base.BPCv2.analyze_token_semantics


def precompute_purity_thresholds(
    model: BPCv3,
    loader: DataLoader,
    device: str = "cpu",
    max_batches: int = 500,
) -> None:
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in BEHAVIOR_AGENT_NAMES}
    collected_matrices: list[torch.Tensor] = []
    per_symbol_collected: dict[int, list[torch.Tensor]] = defaultdict(list)
    use_per_symbol = model.beh_loss_fn.labeling_mode == "per_stock"
    gpu_batches = base._loader_has_gpu_batches(loader)
    max_fit_ordinal = 0

    with torch.no_grad():
        for i, batch in enumerate(base._iter_training_batches(loader)):
            if i >= max_batches:
                break
            if not gpu_batches:
                batch = base._to_device(batch, device)
            if "behavior_proxies" not in batch:
                continue
            stacked = batch["behavior_proxies"]
            collected_matrices.append(stacked.cpu())
            for j, name in enumerate(BEHAVIOR_AGENT_NAMES):
                collected[name].append(stacked[:, j].cpu())
            if use_per_symbol and "stock_ids" in batch:
                sids = batch["stock_ids"]
                for sid in torch.unique(sids):
                    mask = sids == sid
                    per_symbol_collected[int(sid)].append(stacked[mask].cpu())
            if "timestamps" in batch:
                max_fit_ordinal = max(max_fit_ordinal, int(batch["timestamps"].max().item()))

    anchor = BEHAVIOR_AGENT_NAMES[0]
    if not collected[anchor]:
        logger.warning("purity thresholds: no data, using batch quantile fallback.")
        return

    q = torch.tensor([1 / 3, 2 / 3])
    bounds: dict[str, torch.Tensor] = {}
    n_samples = 0
    all_proxies = torch.cat(collected_matrices, dim=0)
    n_samples = int(all_proxies.shape[0])
    if STRUCTURAL_PROXY_INDICES:
        global_mean = all_proxies.mean(dim=0)
        global_std = all_proxies.std(dim=0).clamp_min(1e-6)
        norm_mean = torch.zeros(len(BEHAVIOR_AGENT_NAMES))
        norm_std = torch.ones(len(BEHAVIOR_AGENT_NAMES))
        for j in STRUCTURAL_PROXY_INDICES:
            norm_mean[j] = global_mean[j]
            norm_std[j] = global_std[j]
        model.beh_loss_fn.set_global_normalizers(norm_mean, norm_std)
        logger.info(
            "Global proxy normalizers: mean=%s std=%s (n=%d)",
            [round(norm_mean[j].item(), 4) for j in STRUCTURAL_PROXY_INDICES],
            [round(norm_std[j].item(), 4) for j in STRUCTURAL_PROXY_INDICES],
            n_samples,
        )
    else:
        model.beh_loss_fn.set_global_normalizers(
            torch.zeros(len(BEHAVIOR_AGENT_NAMES)),
            torch.ones(len(BEHAVIOR_AGENT_NAMES)),
        )
        logger.info(
            "Purity labels: raw symbolic proxies + fixed thresholds (no z-score, n=%d)",
            n_samples,
        )

    for name in BEHAVIOR_AGENT_NAMES:
        if name in SYMBOLIC_THRESHOLDS:
            lo, hi = SYMBOLIC_THRESHOLDS[name]
            bounds[name] = torch.tensor([lo, hi])
        else:
            vals = torch.cat(collected[name])
            bounds[name] = torch.quantile(vals, q)
            n_samples = max(n_samples, vals.shape[0])

    for name, b in bounds.items():
        if name in SYMBOLIC_THRESHOLDS:
            continue
        if (b[1] - b[0]).abs() < 1e-6:
            logger.warning(
                "Purity threshold for '%s' is degenerate (%.6f/%.6f). Labels may be noisy.",
                name,
                b[0].item(),
                b[1].item(),
            )

    model.beh_loss_fn.set_thresholds(bounds)
    if max_fit_ordinal > 0:
        model.beh_loss_fn.set_threshold_fit_ordinal(max_fit_ordinal)

    if use_per_symbol and per_symbol_collected:
        symbol_bounds: dict[int, torch.Tensor] = {}
        symbol_counts: dict[int, int] = {}
        sid_sizes = [len(torch.cat(pl, dim=0)) for pl in per_symbol_collected.values() if pl]
        adaptive_min = max(8, int(torch.tensor(sid_sizes).median().item() * 0.1)) if sid_sizes else 8
        for sid, proxy_lists in per_symbol_collected.items():
            if not proxy_lists:
                continue
            all_proxies = torch.cat(proxy_lists, dim=0)
            n_sid = int(all_proxies.shape[0])
            if n_sid < adaptive_min:
                continue
            per_agent_bounds = []
            for j, name in enumerate(BEHAVIOR_AGENT_NAMES):
                if name in SYMBOLIC_THRESHOLDS:
                    lo, hi = SYMBOLIC_THRESHOLDS[name]
                    per_agent_bounds.append(torch.tensor([lo, hi]))
                else:
                    per_agent_bounds.append(torch.quantile(all_proxies[:, j], q))
            symbol_bounds[sid] = torch.stack(per_agent_bounds)
            symbol_counts[sid] = n_sid
        if symbol_bounds:
            model.beh_loss_fn.set_per_symbol_thresholds(symbol_bounds, symbol_counts=symbol_counts)

    if model.beh_loss_fn.labeling_mode == "per_stock":
        model.beh_loss_fn.freeze_train_thresholds()

    core = {k: bounds[k].tolist() for k in CORE_AGENTS}
    ext = {k: bounds[k].tolist() for k in EXTENDED_AGENTS}
    logger.info(
        "PurityThresholds fitted on %d samples | core=%s | extended=%s",
        n_samples,
        core,
        ext,
    )


precompute_z_scale_baselines = base.precompute_z_scale_baselines
precompute_normalizers = base.precompute_normalizers
adapt_codebook_on_loader = base.adapt_codebook_on_loader
eval_epoch = base.eval_epoch
train_epoch = base.train_epoch

__all__ = [
    "BPCv3",
    "BehavioralPurityLoss",
    "CausalFeatureComposer",
    "CausalNormalizer",
    "DEFAULT_REGISTRY",
    "ScaleConfig",
    "ScaleRegistry",
    "adapt_codebook_on_loader",
    "build_scale_registry",
    "eval_epoch",
    "precompute_normalizers",
    "precompute_purity_thresholds",
    "precompute_z_scale_baselines",
    "train_epoch",
]
