"""BPC-v3 向量化特征（相对化 OHLCV 输入；与 bpc.features 绝对路径隔离）。

归一化约定：
- 输入：字段 Δ − 截面中值（L0，见 ohlcv_relative.py）
- 形态还原：prev_bar 绝对锚点 → levels_from_field_deltas
- 波动参照：vol_context 来自 VolatilityStats.lookup_at_anchor（非 close_Δ std）
- 成交量环比用 ratio-1，不对相对量再叠 log（避免 recon loss 虚低）
- 结构维：固定分组尺度 + vol_context 锚点动态缩放（非训练集 z-score）
- encoder：Linear+GELU（无 LayerNorm）；VQ 前 Identity（cosine 自行单位化方向）
"""

from __future__ import annotations

import math

import torch

from .ohlcv_relative import (
    levels_from_field_deltas,
    simple_returns_from_levels,
    volume_change_from_levels,
    volume_ratio_from_levels,
)
# volume_delta_* 来自 bpc/structure_features，v4 独立实现简化版
from .structure_helpers import volume_delta_level_anomaly, volume_delta_rel_cv
from .volatility_context import GLOBAL_BASELINE_VOL
from .behavior_features import compute_behavior_proxies_stacked
from .feature_dims import (
    DAY_FEATURE_SLICES,
    DAY_FULL_FEAT_DIM,
    DAY_STRUCT_FEAT_DIM,
    GROUP_DIM_MAP,
    STRUCT_FEATURE_SCALE,
    TREND_STRUCTURE_SCALE,
    WEEK_GROUP_SCALE,
)


def _levels_from_relative(
    ohlcv: torch.Tensor,
    prev_bar: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if prev_bar is not None:
        levels = levels_from_field_deltas(ohlcv, prev_bar)
        return levels[..., 0], levels[..., 1], levels[..., 2], levels[..., 3]
    return ohlcv[..., 0], ohlcv[..., 1], ohlcv[..., 2], ohlcv[..., 3]


def compute_trend_structure(
    close_delta: torch.Tensor,
    high_delta: torch.Tensor,
    low_delta: torch.Tensor,
    volume_delta: torch.Tensor,
    *,
    prev_bar: torch.Tensor | None = None,
) -> torch.Tensor:
    """趋势结构 [B, 5]：MA5/10/20 偏离 + 区间位置 + 成交量 Δ 趋势。"""
    feats: list[torch.Tensor] = []
    if prev_bar is not None:
        close_lvl = levels_from_field_deltas(
            torch.stack(
                [
                    torch.zeros_like(close_delta),
                    torch.zeros_like(high_delta),
                    torch.zeros_like(low_delta),
                    close_delta,
                    torch.zeros_like(volume_delta),
                ],
                dim=-1,
            ),
            prev_bar,
        )[..., 3]
        high_lvl = levels_from_field_deltas(
            torch.stack(
                [
                    torch.zeros_like(close_delta),
                    high_delta,
                    torch.zeros_like(low_delta),
                    torch.zeros_like(close_delta),
                    torch.zeros_like(volume_delta),
                ],
                dim=-1,
            ),
            prev_bar,
        )[..., 1]
        low_lvl = levels_from_field_deltas(
            torch.stack(
                [
                    torch.zeros_like(close_delta),
                    torch.zeros_like(high_delta),
                    low_delta,
                    torch.zeros_like(close_delta),
                    torch.zeros_like(volume_delta),
                ],
                dim=-1,
            ),
            prev_bar,
        )[..., 2]
    else:
        close_lvl = close_delta.cumsum(dim=1)
        high_lvl = high_delta.cumsum(dim=1)
        low_lvl = low_delta.cumsum(dim=1)

    for period in (5, 10, 20):
        if close_lvl.shape[1] >= period:
            ma = close_lvl[:, -period:].mean(dim=1, keepdim=True)
            deviation = (close_lvl[:, -1:] - ma) / ma.abs().clamp_min(1e-8)
            feats.append(deviation.clamp(-0.5, 0.5))
        else:
            feats.append(torch.zeros(close_lvl.shape[0], 1, device=close_lvl.device, dtype=close_lvl.dtype))

    recent_n = min(20, close_lvl.shape[1])
    recent_low = low_lvl[:, -recent_n:].min(dim=1, keepdim=True)[0]
    recent_high = high_lvl[:, -recent_n:].max(dim=1, keepdim=True)[0]
    position = (close_lvl[:, -1:] - recent_low) / (recent_high - recent_low).clamp_min(1e-8)
    feats.append(position.clamp(0.0, 1.0))

    if prev_bar is not None:
        vol_lvl = levels_from_field_deltas(
            torch.stack(
                [
                    torch.zeros_like(close_delta),
                    torch.zeros_like(high_delta),
                    torch.zeros_like(low_delta),
                    torch.zeros_like(close_delta),
                    volume_delta,
                ],
                dim=-1,
            ),
            prev_bar,
        )[..., 4]
        vol_chg = volume_change_from_levels(vol_lvl)
        vol_base_src = vol_chg
    else:
        vol_base_src = volume_delta
    vol_recent = vol_base_src[:, -5:].mean(dim=1, keepdim=True)
    if vol_base_src.shape[1] > 5:
        vol_baseline = vol_base_src[:, :-5].mean(dim=1, keepdim=True)
    else:
        vol_baseline = vol_recent
    vol_trend = (vol_recent - vol_baseline).clamp(-2.0, 2.0)
    feats.append(vol_trend)

    return torch.cat(feats, dim=1) * TREND_STRUCTURE_SCALE


# 21 维结构向量中波动敏感维索引
_IDX_RV = 0
_IDX_VOL_LEVEL = 7
_IDX_VOL_CV = 8
_IDX_RANGE_PROXY = 14


def _vol_regime_factor(
    vol_context: torch.Tensor | None,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """由 vol_context dim2 (= rv/baseline − 1) 得锚点 RV 相对基准倍数。"""
    if vol_context is None:
        return torch.ones(batch, 1, device=device, dtype=dtype)
    return (
        1.0 + vol_context[:, 2:3].to(device=device, dtype=dtype).clamp(-0.9, 4.0)
    ).clamp(0.2, 5.0)


def apply_struct_group_scales(
    struct: torch.Tensor,
    vol_context: torch.Tensor | None = None,
    *,
    base_scales: tuple[float, ...] = STRUCT_FEATURE_SCALE,
) -> torch.Tensor:
    """固定分组尺度 + 波动率锚点动态缩放（适应疫情等高波环境）。"""
    device, dtype = struct.device, struct.dtype
    scales = torch.tensor(base_scales, device=device, dtype=dtype)
    out = struct * scales
    regime = _vol_regime_factor(vol_context, struct.shape[0], device, dtype)
    anchor_rv = (GLOBAL_BASELINE_VOL * regime).clamp_min(1e-4)
    out[:, _IDX_RV : _IDX_RV + 1] = (
        struct[:, _IDX_RV : _IDX_RV + 1] * scales[_IDX_RV : _IDX_RV + 1] / anchor_rv
    )
    # vol_div 为 per-sample 标量，各 vol/range 维同除 — 非 normalize 场景，不会「同维缩放抵消」
    vol_div = regime.clamp_min(0.2)
    out[:, _IDX_VOL_LEVEL : _IDX_VOL_CV + 1] = (
        struct[:, _IDX_VOL_LEVEL : _IDX_VOL_CV + 1] * scales[_IDX_VOL_LEVEL : _IDX_VOL_CV + 1] / vol_div
    )
    out[:, _IDX_RANGE_PROXY : _IDX_RANGE_PROXY + 1] = (
        struct[:, _IDX_RANGE_PROXY : _IDX_RANGE_PROXY + 1]
        * scales[_IDX_RANGE_PROXY]
        / vol_div
    )
    return out


def apply_price_structure_scales(
    feats: torch.Tensor,
    vol_context: torch.Tensor | None = None,
    *,
    base_scales: tuple[float, ...] = WEEK_GROUP_SCALE,
) -> torch.Tensor:
    """周尺度 7 维 price_structure 的动态 rv 缩放。"""
    device, dtype = feats.device, feats.dtype
    scales = torch.tensor(base_scales, device=device, dtype=dtype)
    out = feats * scales
    regime = _vol_regime_factor(vol_context, feats.shape[0], device, dtype)
    anchor_rv = (GLOBAL_BASELINE_VOL * regime).clamp_min(1e-4)
    out[:, 0:1] = feats[:, 0:1] * scales[0:1] / anchor_rv
    return out
def compute_day_features_vectorized(
    ohlcv: torch.Tensor,
    vol_context: torch.Tensor | None = None,
    prev_bar: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    向量化计算 day 预计算特征。
    输入: [B, T, 5] 相对化 OHLCV（字段 Δ，已减截面中值）
    prev_bar: [B, 5] 窗口前一根绝对 bar，用于还原 K 线形态层级
    """
    device = ohlcv.device
    B, T, _ = ohlcv.shape

    open_d = ohlcv[..., 0]
    high_d = ohlcv[..., 1]
    low_d = ohlcv[..., 2]
    close_d = ohlcv[..., 3]
    volume_d = ohlcv[..., 4]

    open_p, high, low, close = _levels_from_relative(ohlcv, prev_bar)

    # 价格结构：用伪价格层级的简单收益率（非 Δclose 元）
    if close.shape[1] >= 2:
        ret = simple_returns_from_levels(close)
    else:
        ret = torch.zeros(B, 1, device=device, dtype=ohlcv.dtype)

    rv = ret.std(dim=1, keepdim=True)
    m = ret.mean(dim=1, keepdim=True)
    s = ret.std(dim=1, keepdim=True).clamp_min(1e-8)
    skew = ((ret - m) ** 3).mean(dim=1, keepdim=True) / (s**3).clamp_min(1e-4)
    skew = skew.clamp(-3, 3)

    anchor_close = close[:, :1].clamp_min(1e-8)
    net_move = ((close[:, -1:] - close[:, :1]) / anchor_close).abs()
    path_len = (high - low).abs().sum(dim=1, keepdim=True) / anchor_close
    efficiency = net_move / path_len.clamp_min(1e-8)

    span = (close.max(dim=1)[0] - close.min(dim=1)[0]).clamp_min(1e-8)
    price_pos = ((close[:, -1] - close.min(dim=1)[0]) / span).unsqueeze(1)

    autocorrs = []
    for lag in [1, 2, 5]:
        if ret.shape[1] > lag:
            x1 = ret[:, :-lag]
            x2 = ret[:, lag:]
            x1 = x1 - x1.mean(dim=1, keepdim=True)
            x2 = x2 - x2.mean(dim=1, keepdim=True)
            corr = (x1 * x2).sum(dim=1) / (x1.norm(dim=1) * x2.norm(dim=1)).clamp_min(1e-8)
            autocorrs.append(corr.unsqueeze(1))
        else:
            autocorrs.append(torch.zeros(B, 1, device=device))
    price_structure = torch.cat([rv, skew, efficiency, price_pos, torch.cat(autocorrs, dim=1)], dim=1)

    if prev_bar is not None:
        vol_levels = levels_from_field_deltas(ohlcv, prev_bar)[..., 4]
        vol_chg = volume_change_from_levels(vol_levels)
        vol_level = volume_delta_level_anomaly(vol_chg)
        vol_cv = volume_delta_rel_cv(vol_chg)
        vol_slice = vol_chg[:, 1:]
        ret_slice = ret
    else:
        vol_level = volume_delta_level_anomaly(volume_d)
        vol_cv = volume_delta_rel_cv(volume_d)
        vol_slice = volume_d[:, 1:]
        ret_slice = ret
    if ret_slice.shape[1] > 1 and vol_slice.shape[1] == ret_slice.shape[1]:
        ret_c = ret_slice - ret_slice.mean(dim=1, keepdim=True)
        vol_c = vol_slice - vol_slice.mean(dim=1, keepdim=True)
        vp_corr = (ret_c * vol_c).sum(dim=1) / (ret_c.norm(dim=1) * vol_c.norm(dim=1)).clamp_min(1e-8)
        vp_corr = vp_corr.unsqueeze(1)
    else:
        vp_corr = torch.zeros(B, 1, device=device)
    if prev_bar is not None:
        vol_ratio = volume_ratio_from_levels(vol_levels)
        sorted_ratio, _ = torch.sort(vol_ratio, dim=1, descending=True)
        top20 = (
            sorted_ratio[:, : max(1, T // 5)].sum(dim=1) / vol_ratio.sum(dim=1).clamp_min(1e-8)
        ).unsqueeze(1)
    else:
        sorted_vol, _ = torch.sort(volume_d.abs(), dim=1, descending=True)
        top20 = (
            sorted_vol[:, : max(1, T // 5)].sum(dim=1) / volume_d.abs().sum(dim=1).clamp_min(1e-8)
        ).unsqueeze(1)
    volume_structure = torch.cat([vol_level, vol_cv, vp_corr, top20], dim=1)

    bar_range = (high - low).clamp_min(1e-8)
    eff = (close - open_p) / bar_range
    if prev_bar is not None:
        vol_ratio = volume_ratio_from_levels(levels_from_field_deltas(ohlcv, prev_bar)[..., 4])
        rel_vol = vol_ratio / vol_ratio.mean(dim=1, keepdim=True).clamp_min(1e-8)
    else:
        vol_mean = volume_d.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        rel_vol = volume_d.abs() / vol_mean
    attack = (eff * rel_vol).mean(dim=1, keepdim=True)

    last_n = min(5, eff.shape[1])
    last_eff = eff[:, -last_n:]
    last_rel = rel_vol[:, -last_n:]
    last_attack = last_eff * last_rel
    if last_n > 1:
        decay = ((last_attack[:, -1] - last_attack[:, 0]) / (last_n - 1)).unsqueeze(1)
    else:
        decay = torch.zeros(B, 1, device=device)
    pressure = (
        (close[:, -1] - low.min(dim=1)[0])
        / (high.max(dim=1)[0] - low.min(dim=1)[0]).clamp_min(1e-8)
    ).unsqueeze(1)
    attack_proxy = torch.cat([attack, decay, pressure], dim=1)

    range_proxy = (high - low).abs().mean(dim=1).unsqueeze(1) / close[:, :1].clamp_min(1e-8)
    if prev_bar is not None:
        vol_ratio = volume_ratio_from_levels(levels_from_field_deltas(ohlcv, prev_bar)[..., 4])
        pulse = (vol_ratio[:, -1] / vol_ratio.mean(dim=1).clamp_min(1e-8)).unsqueeze(1)
    else:
        pulse = (volume_d[:, -1] / volume_d.abs().mean(dim=1).clamp_min(1e-8)).unsqueeze(1)
    micro_proxy = torch.cat([range_proxy, pulse], dim=1)

    trend_structure = compute_trend_structure(close_d, high_d, low_d, volume_d, prev_bar=prev_bar)
    behavior_structure = compute_behavior_proxies_stacked(ohlcv, vol_context, prev_bar=prev_bar)
    behavior_structure = torch.nan_to_num(behavior_structure, nan=0.0, posinf=0.0, neginf=0.0).clamp(
        -20.0, 20.0
    )

    features = torch.cat(
        [
            apply_struct_group_scales(
                torch.cat(
                    [
                        price_structure,
                        volume_structure,
                        attack_proxy,
                        micro_proxy,
                        trend_structure,
                    ],
                    dim=1,
                ),
                vol_context,
            ),
            behavior_structure,
        ],
        dim=1,
    )
    if features.shape[1] != DAY_FULL_FEAT_DIM:
        raise ValueError(
            f"feature dim {features.shape[1]} != DAY_FULL_FEAT_DIM ({DAY_FULL_FEAT_DIM})"
        )
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)


def compute_week_features_vectorized(
    ohlcv: torch.Tensor,
    prev_bar: torch.Tensor | None = None,
    vol_context: torch.Tensor | None = None,
) -> torch.Tensor:
    """周尺度 7 维 price_structure；收益率来自伪价格层级。"""
    device = ohlcv.device
    B, _T, _ = ohlcv.shape
    if prev_bar is not None:
        close = levels_from_field_deltas(ohlcv, prev_bar)[..., 3]
        high = levels_from_field_deltas(ohlcv, prev_bar)[..., 1]
        low = levels_from_field_deltas(ohlcv, prev_bar)[..., 2]
    else:
        close = ohlcv[..., 3].cumsum(dim=1)
        high = ohlcv[..., 1].cumsum(dim=1)
        low = ohlcv[..., 2].cumsum(dim=1)

    if close.shape[1] >= 2:
        ret = simple_returns_from_levels(close)
    else:
        ret = torch.zeros(B, 1, device=device, dtype=ohlcv.dtype)

    rv = ret.std(dim=1, keepdim=True)
    m = ret.mean(dim=1, keepdim=True)
    s = ret.std(dim=1, keepdim=True).clamp_min(1e-8)
    skew = ((ret - m) ** 3).mean(dim=1, keepdim=True) / (s**3).clamp_min(1e-4)
    skew = skew.clamp(-3, 3)

    anchor = close[:, :1].clamp_min(1e-8)
    net_move = ((close[:, -1:] - close[:, :1]) / anchor).abs()
    path_len = (high - low).abs().sum(dim=1, keepdim=True) / anchor
    efficiency = net_move / path_len.clamp_min(1e-8)

    span = (close.max(dim=1)[0] - close.min(dim=1)[0]).clamp_min(1e-8)
    price_pos = ((close[:, -1] - close.min(dim=1)[0]) / span).unsqueeze(1)

    autocorrs = []
    for lag in [1, 2, 5]:
        if ret.shape[1] > lag:
            x1 = ret[:, :-lag] - ret[:, :-lag].mean(dim=1, keepdim=True)
            x2 = ret[:, lag:] - ret[:, lag:].mean(dim=1, keepdim=True)
            corr = (x1 * x2).sum(dim=1) / (x1.norm(dim=1) * x2.norm(dim=1)).clamp_min(1e-8)
            autocorrs.append(corr.unsqueeze(1))
        else:
            autocorrs.append(torch.zeros(B, 1, device=device))
    return torch.nan_to_num(
        apply_price_structure_scales(
            torch.cat([rv, skew, efficiency, price_pos, torch.cat(autocorrs, dim=1)], dim=1),
            vol_context,
        ),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp(-20.0, 20.0)


def extract_group_features(
    ohlcv: torch.Tensor,
    groups: list[str],
    *,
    prev_bar: torch.Tensor | None = None,
    vol_context: torch.Tensor | None = None,
    scale_name: str = "day",
) -> torch.Tensor:
    """从相对化 OHLCV 提取指定特征组（Causal 路径与物化路径统一入口）。"""
    if scale_name == "week":
        return compute_week_features_vectorized(ohlcv, prev_bar, vol_context)
    full = compute_day_features_vectorized(ohlcv, vol_context, prev_bar)
    parts = [full[:, DAY_FEATURE_SLICES[g]] for g in groups if g in DAY_FEATURE_SLICES]
    if not parts:
        return torch.zeros(ohlcv.shape[0], 0, device=ohlcv.device, dtype=ohlcv.dtype)
    return torch.cat(parts, dim=1)


# ============================================================
# 以下为参考更优实现追加的三个函数（保持向后兼容）
# ============================================================

import math
from typing import Optional

from quant_cursor.bpc_v3.features import compute_day_features_vectorized as bpc_v3_feat


def compute_bpc_features(
    ohlcv: torch.Tensor,
    vol_context: torch.Tensor,
    prev_bar: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """复用 BPC-v3 的 26 维特征提取"""
    return bpc_v3_feat(ohlcv, vol_context, prev_bar)


def compute_context_features(
    ohlcv: torch.Tensor,
    vol_context: torch.Tensor,
    prev_bar: torch.Tensor,
) -> torch.Tensor:
    """计算 7 维上下文特征: vol_context(3) + 幅度代理(4)"""
    B, T, _ = ohlcv.shape
    device = ohlcv.device
    dtype = ohlcv.dtype

    ctx = vol_context.clone()

    close = ohlcv[:, :, 3]
    high = ohlcv[:, :, 1]
    low = ohlcv[:, :, 2]
    volume = ohlcv[:, :, 4]

    if prev_bar is not None:
        anchor_close = prev_bar[:, 3:4]
    else:
        anchor_close = close[:, 0:1]

    atr = (high - low).abs().mean(dim=1, keepdim=True) / anchor_close.clamp_min(1e-8)

    if prev_bar is not None:
        gap = (ohlcv[:, 0, 0:1] - prev_bar[:, 3:4]) / prev_bar[:, 3:4].clamp_min(1e-8)
    else:
        gap = torch.zeros(B, 1, device=device, dtype=dtype)

    vol_ma = volume.mean(dim=1, keepdim=True)
    vol_ratio = volume[:, -1:] / vol_ma.clamp_min(1e-8)
    amp_range = (high[:, -1:] - low[:, -1:]) / close[:, -1:].clamp_min(1e-8)

    amp_proxy = torch.cat([atr, gap, vol_ratio, amp_range], dim=1)
    ctx = torch.cat([ctx, amp_proxy], dim=1)
    return ctx


def compute_time_embedding(timestamps: torch.Tensor, raw_dim: int = 16) -> torch.Tensor:
    """从交易日序数生成 16 维周期编码（sin/cos）"""
    device = timestamps.device
    dtype = timestamps.dtype
    B = timestamps.shape[0]

    day_of_year = timestamps.float() % 365
    day_of_week = timestamps.float() % 7
    month = timestamps.float() % 12
    year_phase = timestamps.float() / 365.25

    periods = torch.stack([
        day_of_year / 365.25,
        day_of_week / 7.0,
        month / 12.0,
        year_phase,
    ], dim=1)

    emb = torch.zeros(B, raw_dim, device=device, dtype=dtype)
    for i in range(4):
        emb[:, 2 * i] = torch.sin(2 * math.pi * periods[:, i])
        emb[:, 2 * i + 1] = torch.cos(2 * math.pi * periods[:, i])
    return emb

