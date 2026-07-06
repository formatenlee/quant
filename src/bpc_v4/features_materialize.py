"""物化阶段特征：NumPy 数据流；torch 仅在 inference_mode 下作批量数学核。"""

from __future__ import annotations

import numpy as np
import torch

from .behavior_features import compute_behavior_proxies_stacked, compute_purity_targets_from_proxies
from .features import compute_bpc_features, compute_context_features, compute_time_embedding
from .ohlcv_relative import absolute_window_to_relative_batch
from .volatility_context import VolatilityStats


def _sanitize_np(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)


def relative_windows_batch(
    abs_windows: np.ndarray,
    prev_bars: np.ndarray,
    bar_ords: np.ndarray,
    cs_medians: np.ndarray,
) -> np.ndarray:
    """[B,T,5] 绝对窗口 → 截面中心化字段 Δ（纯 NumPy）。"""
    return absolute_window_to_relative_batch(abs_windows, prev_bars, bar_ords, cs_medians)


def compute_chunk_features_numpy(
    rel_ohlcv: np.ndarray,
    prev_bars: np.ndarray,
    stock_ids: np.ndarray,
    cal_ords: np.ndarray,
    vol_stats: VolatilityStats,
    *,
    label_temperature: float,
    time_raw_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    分片内批量 BPC / ctx / purity / time_emb。

    torch 仅在此函数内、无梯度地执行已有向量化算子。
    """
    vol_ctx = vol_stats.lookup_at_anchor_numpy(stock_ids, cal_ords)
    with torch.inference_mode():
        rel_t = torch.from_numpy(rel_ohlcv)
        prev_t = torch.from_numpy(prev_bars)
        vol_t = torch.from_numpy(vol_ctx)
        proxies = compute_behavior_proxies_stacked(rel_t, vol_t, prev_t)
        purity_tgt = compute_purity_targets_from_proxies(
            proxies, temperature=label_temperature
        )
        bpc_feat = compute_bpc_features(rel_t, vol_t, prev_t)
        ctx_feat = compute_context_features(rel_t, vol_t, prev_t)
        ord_t = torch.from_numpy(cal_ords.astype(np.int64, copy=False))
        time_emb = compute_time_embedding(ord_t, raw_dim=time_raw_dim)
        return (
            _sanitize_np(bpc_feat.cpu().numpy()),
            _sanitize_np(ctx_feat.cpu().numpy()),
            purity_tgt.cpu().numpy().astype(np.float32, copy=False),
            time_emb.cpu().numpy().astype(np.float32, copy=False),
        )
