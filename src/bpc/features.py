"""
BPC-v2 向量化特征（绝对 OHLCV 输入）。

物化路径与 CausalFeatureComposer 共用本模块，保证 v2 与 v3 特征实现隔离。
"""

from __future__ import annotations

import torch

from quant_cursor.bpc.behavior_features import compute_behavior_proxies_stacked
from quant_cursor.bpc.feature_dims import DAY_FULL_FEAT_DIM, GROUP_DIM_MAP
from quant_cursor.bpc.structure_features import volume_log_level_anomaly, volume_rel_cv

_FEATURE_GROUP_ORDER: tuple[str, ...] = (
    "price_structure",
    "volume_structure",
    "attack_proxy",
    "micro_proxy",
    "behavior_structure",
)

DAY_FEATURE_SLICES: dict[str, slice] = {}
_offset = 0
for _g in _FEATURE_GROUP_ORDER:
    _d = GROUP_DIM_MAP[_g]
    DAY_FEATURE_SLICES[_g] = slice(_offset, _offset + _d)
    _offset += _d


def compute_day_features_vectorized(ohlcv: torch.Tensor) -> torch.Tensor:
    """
    绝对 OHLCV → day 全量预计算特征 [B, DAY_FULL_FEAT_DIM]。

    结构 16 维（price/volume/attack/micro）+ 行为代理 5 维。
    """
    device = ohlcv.device
    B, T, _ = ohlcv.shape
    eps = 1e-8

    open_p = ohlcv[..., 0]
    high = ohlcv[..., 1]
    low = ohlcv[..., 2]
    close = ohlcv[..., 3]
    volume = ohlcv[..., 4].clamp_min(0.0)

    log_ret = torch.log(close[:, 1:] / close[:, :-1].clamp_min(eps)) if T > 1 else close.new_zeros(B, 1)

    rv = log_ret.std(dim=1, keepdim=True)
    m = log_ret.mean(dim=1, keepdim=True)
    s = log_ret.std(dim=1, keepdim=True).clamp_min(eps)
    skew = ((log_ret - m) ** 3).mean(dim=1, keepdim=True) / (s**3).clamp_min(1e-4)
    skew = skew.clamp(-10, 10)

    net_move = (close[:, -1] - close[:, 0]).abs().unsqueeze(1)
    path_len = (high - low).abs().sum(dim=1, keepdim=True)
    efficiency = net_move / path_len.clamp_min(eps)

    span = (close.max(dim=1)[0] - close.min(dim=1)[0]).clamp_min(eps)
    price_pos = ((close[:, -1] - close.min(dim=1)[0]) / span).unsqueeze(1)

    autocorrs = []
    for lag in (1, 2, 5):
        if log_ret.shape[1] > lag:
            x1 = log_ret[:, :-lag] - log_ret[:, :-lag].mean(dim=1, keepdim=True)
            x2 = log_ret[:, lag:] - log_ret[:, lag:].mean(dim=1, keepdim=True)
            corr = (x1 * x2).sum(dim=1) / (x1.norm(dim=1) * x2.norm(dim=1)).clamp_min(eps)
            autocorrs.append(corr.unsqueeze(1))
        else:
            autocorrs.append(torch.zeros(B, 1, device=device))
    price_structure = torch.cat([rv, skew, efficiency, price_pos, torch.cat(autocorrs, dim=1)], dim=1)

    vol_level = volume_log_level_anomaly(volume)
    vol_cv = volume_rel_cv(volume)
    vol_slice = volume[:, 1:]
    ret_slice = log_ret
    if ret_slice.shape[1] > 1 and vol_slice.shape[1] == ret_slice.shape[1]:
        ret_c = ret_slice - ret_slice.mean(dim=1, keepdim=True)
        vol_c = vol_slice - vol_slice.mean(dim=1, keepdim=True)
        vp_corr = (ret_c * vol_c).sum(dim=1) / (ret_c.norm(dim=1) * vol_c.norm(dim=1)).clamp_min(eps)
        vp_corr = vp_corr.unsqueeze(1)
    else:
        vp_corr = torch.zeros(B, 1, device=device)
    sorted_vol, _ = torch.sort(volume, dim=1, descending=True)
    top20 = (sorted_vol[:, : max(1, T // 5)].sum(dim=1) / volume.sum(dim=1).clamp_min(eps)).unsqueeze(1)
    volume_structure = torch.cat([vol_level, vol_cv, vp_corr, top20], dim=1)

    bar_range = (high - low).clamp_min(eps)
    eff = (close - open_p) / bar_range
    vol_mean = volume.mean(dim=1, keepdim=True).clamp_min(eps)
    rel_vol = volume / vol_mean
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
        (close[:, -1] - low.min(dim=1)[0]) / (high.max(dim=1)[0] - low.min(dim=1)[0]).clamp_min(eps)
    ).unsqueeze(1)
    attack_proxy = torch.cat([attack, decay, pressure], dim=1)

    range_proxy = (high - low).mean(dim=1).unsqueeze(1)
    pulse = (volume[:, -1] / volume.mean(dim=1).clamp_min(eps)).unsqueeze(1)
    micro_proxy = torch.cat([range_proxy, pulse], dim=1)

    behavior_structure = compute_behavior_proxies_stacked(ohlcv)
    behavior_structure = torch.nan_to_num(behavior_structure, nan=0.0, posinf=0.0, neginf=0.0).clamp(
        -20.0, 20.0
    )

    features = torch.cat(
        [price_structure, volume_structure, attack_proxy, micro_proxy, behavior_structure],
        dim=1,
    )
    if features.shape[1] != DAY_FULL_FEAT_DIM:
        raise ValueError(f"feature dim {features.shape[1]} != DAY_FULL_FEAT_DIM ({DAY_FULL_FEAT_DIM})")
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).clamp(-20.0, 20.0)


def compute_week_features_vectorized(ohlcv: torch.Tensor) -> torch.Tensor:
    """周尺度 7 维 price_structure（绝对 OHLCV）。"""
    device = ohlcv.device
    B, _T, _ = ohlcv.shape
    eps = 1e-8
    close = ohlcv[..., 3]
    high = ohlcv[..., 1]
    low = ohlcv[..., 2]

    log_ret = torch.log(close[:, 1:] / close[:, :-1].clamp_min(eps)) if close.shape[1] > 1 else close.new_zeros(B, 1)

    rv = log_ret.std(dim=1, keepdim=True)
    m = log_ret.mean(dim=1, keepdim=True)
    s = log_ret.std(dim=1, keepdim=True).clamp_min(eps)
    skew = ((log_ret - m) ** 3).mean(dim=1, keepdim=True) / (s**3).clamp_min(1e-4)
    skew = skew.clamp(-10, 10)

    net_move = (close[:, -1] - close[:, 0]).abs().unsqueeze(1)
    path_len = (high - low).abs().sum(dim=1, keepdim=True)
    efficiency = net_move / path_len.clamp_min(eps)

    span = (close.max(dim=1)[0] - close.min(dim=1)[0]).clamp_min(eps)
    price_pos = ((close[:, -1] - close.min(dim=1)[0]) / span).unsqueeze(1)

    autocorrs = []
    for lag in (1, 2, 5):
        if log_ret.shape[1] > lag:
            x1 = log_ret[:, :-lag] - log_ret[:, :-lag].mean(dim=1, keepdim=True)
            x2 = log_ret[:, lag:] - log_ret[:, lag:].mean(dim=1, keepdim=True)
            corr = (x1 * x2).sum(dim=1) / (x1.norm(dim=1) * x2.norm(dim=1)).clamp_min(eps)
            autocorrs.append(corr.unsqueeze(1))
        else:
            autocorrs.append(torch.zeros(B, 1, device=device))
    return torch.nan_to_num(
        torch.cat([rv, skew, efficiency, price_pos, torch.cat(autocorrs, dim=1)], dim=1),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp(-20.0, 20.0)


def extract_group_features(
    ohlcv: torch.Tensor,
    groups: list[str],
    *,
    scale_name: str = "day",
) -> torch.Tensor:
    """Causal 路径：从绝对 OHLCV 提取指定特征组。"""
    if scale_name == "week":
        return compute_week_features_vectorized(ohlcv)
    full = compute_day_features_vectorized(ohlcv)
    parts = [full[:, DAY_FEATURE_SLICES[g]] for g in groups if g in DAY_FEATURE_SLICES]
    if not parts:
        return torch.zeros(ohlcv.shape[0], 0, device=ohlcv.device, dtype=ohlcv.dtype)
    return torch.cat(parts, dim=1)
