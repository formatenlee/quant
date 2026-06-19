"""
BPC-v3 OHLCV 相对化（v2 不使用）。

规则（逐字段、逐交易步）：
  delta[field, t] = abs[field, t] - abs[field, t-1]
  rel[field, t] = delta[field, t] - median_instrument(delta[field, t])
窗口首根 K 线以窗口外上一根绝对 bar 为 t-1 参考。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from quant_cursor.bpc.dataset import QlibInstrumentStore

logger = logging.getLogger(__name__)

OHLCV_FIELDS = ("open", "high", "low", "close", "volume")
NUM_OHLCV_FIELDS = 5
OHLCV_RELATIVE_SCHEMA = "field_delta_cs_v1"


@dataclass
class CrossSectionDeltaMedians:
    """按交易日（或周结束日）索引的截面 Δ 中值表。"""

    day: np.ndarray
    week: np.ndarray

    @classmethod
    def from_store(
        cls,
        store: QlibInstrumentStore,
        calendar_ordinals: dict,
    ) -> CrossSectionDeltaMedians:
        n_cal = max(calendar_ordinals.values()) + 1 if calendar_ordinals else 0
        day_buckets: list[list[np.ndarray]] = [[] for _ in range(n_cal)]
        week_buckets: list[list[np.ndarray]] = [[] for _ in range(n_cal)]

        for series in store._cache.values():
            ohlcv = series.ohlcv
            for t in range(1, len(ohlcv)):
                d = series.dates[t]
                o = calendar_ordinals.get(d, -1)
                if o < 0:
                    continue
                day_buckets[o].append(ohlcv[t] - ohlcv[t - 1])

            weekly = series.weekly
            week_dates = series.weekly_end_dates
            for t in range(1, len(weekly)):
                d = week_dates[t]
                o = calendar_ordinals.get(d, -1)
                if o < 0:
                    continue
                week_buckets[o].append(weekly[t] - weekly[t - 1])

        day_med = np.zeros((n_cal, NUM_OHLCV_FIELDS), dtype=np.float32)
        week_med = np.zeros((n_cal, NUM_OHLCV_FIELDS), dtype=np.float32)
        for o in range(n_cal):
            if day_buckets[o]:
                day_med[o] = np.median(np.stack(day_buckets[o], axis=0), axis=0)
            if week_buckets[o]:
                week_med[o] = np.median(np.stack(week_buckets[o], axis=0), axis=0)

        logger.info(
            "CrossSectionDeltaMedians: calendar=%d, day_nonempty=%d, week_nonempty=%d",
            n_cal,
            sum(1 for b in day_buckets if b),
            sum(1 for b in week_buckets if b),
        )
        return cls(day=day_med, week=week_med)


def _bar_ordinals_for_window(
    dates,
    date_to_ordinal: dict,
    end_idx: int,
    lookback: int,
) -> np.ndarray:
    start = end_idx - lookback + 1
    ords = np.zeros(lookback, dtype=np.int64)
    for i, t in enumerate(range(start, end_idx + 1)):
        ords[i] = date_to_ordinal.get(dates[t], 0)
    return ords


def absolute_window_to_relative(
    abs_window: np.ndarray,
    prev_bar: np.ndarray,
    bar_ordinals: np.ndarray,
    cs_medians: np.ndarray,
) -> np.ndarray:
    """绝对 OHLCV 窗口 → 截面中心化字段 Δ。"""
    t_len = abs_window.shape[0]
    out = np.zeros_like(abs_window, dtype=np.float32)
    prev_abs = prev_bar.astype(np.float32, copy=False)
    for i in range(t_len):
        delta = abs_window[i] - prev_abs
        o = int(bar_ordinals[i])
        if 0 <= o < cs_medians.shape[0]:
            delta = delta - cs_medians[o]
        out[i] = delta
        prev_abs = abs_window[i]
    return out


def absolute_window_to_relative_torch(
    abs_window: torch.Tensor,
    prev_bar: torch.Tensor,
    bar_ordinals: torch.Tensor,
    cs_medians: torch.Tensor,
) -> torch.Tensor:
    """批量版：abs_window [B,T,5], prev_bar [B,5], bar_ordinals [B,T], cs_medians [n_cal,5]."""
    bsz, t_len, _ = abs_window.shape
    out = torch.zeros_like(abs_window)
    prev = prev_bar.unsqueeze(1)
    for i in range(t_len):
        delta = abs_window[:, i, :] - prev.squeeze(1)
        o = bar_ordinals[:, i].long().clamp(0, cs_medians.shape[0] - 1)
        delta = delta - cs_medians[o]
        out[:, i, :] = delta
        prev = abs_window[:, i : i + 1, :]
    return out


def levels_from_field_deltas(
    deltas: torch.Tensor,
    prev_bar: torch.Tensor,
) -> torch.Tensor:
    """由字段 Δ 还原窗口内伪价格层级（K 线形态 / 跳空）。"""
    if prev_bar.dim() == 1:
        prev_bar = prev_bar.unsqueeze(0)
    if prev_bar.dim() == 2 and prev_bar.shape[1] == NUM_OHLCV_FIELDS:
        base = prev_bar.unsqueeze(1)
    else:
        base = prev_bar
    return base + deltas.cumsum(dim=1)


def simple_returns_from_levels(close: torch.Tensor) -> torch.Tensor:
    """伪价格层级 → 简单收益率 (close_t - close_{t-1}) / close_{t-1}，无量纲。"""
    if close.shape[1] < 2:
        return torch.zeros(close.shape[0], max(1, close.shape[1] - 1), device=close.device, dtype=close.dtype)
    prev = close[:, :-1].clamp_min(1e-8)
    return (close[:, 1:] - prev) / prev


def volume_ratio_from_levels(vol: torch.Tensor) -> torch.Tensor:
    """绝对成交量层级 → 环比 V_t / V_{t-1}（首根为 1，因果）。"""
    vol = vol.clamp_min(1e-8)
    if vol.shape[1] < 2:
        return torch.ones_like(vol)
    ratio = vol[:, 1:] / vol[:, :-1]
    return torch.cat([torch.ones(vol.shape[0], 1, device=vol.device, dtype=vol.dtype), ratio], dim=1)


def volume_change_from_levels(vol: torch.Tensor) -> torch.Tensor:
    """绝对成交量环比变化率 V_t/V_{t-1} - 1（首根为 0）。

    相对量框架下不再对环比取 log，避免特征量级过小导致 recon loss 虚低。
    """
    return volume_ratio_from_levels(vol) - 1.0
