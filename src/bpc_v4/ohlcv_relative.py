"""
BPC-v4 OHLCV 相对化。

规则（逐字段、逐交易步）：
  delta[field, t] = abs[field, t] - abs[field, t-1]
  rel[field, t] = delta[field, t] - median_instrument(delta[field, *, t])_{t-1日}

截面中值使用 **上一交易日** 全市场 Δ 中值（lag=1），避免当日未收盘标的信息进入中值。
窗口首根 K 线以窗口外上一根绝对 bar 为 t-1 参考。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)

OHLCV_FIELDS = ("open", "high", "low", "close", "volume")
NUM_OHLCV_FIELDS = 5
OHLCV_RELATIVE_SCHEMA = "field_delta_cs_v2_lag1"


def _lag_trading_day_medians(raw: np.ndarray) -> np.ndarray:
    """raw[o] 为第 o 日收盘后可得的截面中值；返回 lag1[o]=raw[o-1]（o=0 为 0）。"""
    out = np.zeros_like(raw)
    if raw.shape[0] > 1:
        out[1:] = raw[:-1]
    return out


@dataclass
class CrossSectionDeltaMedians:
    """按交易日 ordinal 索引的截面 Δ 中值表（已 lag1，可直接用于物化）。"""

    day: np.ndarray

    @classmethod
    def from_v4_store(cls, store) -> CrossSectionDeltaMedians:
        """从 BPCV4InstrumentStore 构建 lag1 截面 Δ 中值。"""
        date_to_ordinal = {d: i for i, d in enumerate(store.calendar)}
        n_cal = len(store.calendar)
        day_buckets: list[list[np.ndarray]] = [[] for _ in range(n_cal)]

        for series in store._cache.values():
            ohlcv = series.ohlcva[:, :NUM_OHLCV_FIELDS]
            for t in range(1, len(ohlcv)):
                d = series.dates[t]
                o = date_to_ordinal.get(d, -1)
                if o < 0:
                    continue
                day_buckets[o].append(ohlcv[t] - ohlcv[t - 1])

        raw_med = np.zeros((n_cal, NUM_OHLCV_FIELDS), dtype=np.float32)
        for o in range(n_cal):
            if day_buckets[o]:
                raw_med[o] = np.median(np.stack(day_buckets[o], axis=0), axis=0)

        day_med = _lag_trading_day_medians(raw_med)

        logger.info(
            "CrossSectionDeltaMedians (v4, %s): calendar=%d, day_nonempty=%d, lag1 applied",
            OHLCV_RELATIVE_SCHEMA,
            n_cal,
            sum(1 for b in day_buckets if b),
        )
        return cls(day=day_med)


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
    """
    绝对 OHLCV 窗口 → 截面中心化字段 Δ。

    cs_medians 须为 lag1 表（见 CrossSectionDeltaMedians.from_v4_store）。
    """
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


def absolute_window_to_relative_batch(
    abs_windows: np.ndarray,
    prev_bars: np.ndarray,
    bar_ords: np.ndarray,
    cs_medians: np.ndarray,
) -> np.ndarray:
    """批量版 NumPy：abs_windows [B,T,5], prev_bars [B,5], bar_ords [B,T]。"""
    bsz, t_len, _ = abs_windows.shape
    out = np.empty_like(abs_windows, dtype=np.float32)
    prev = prev_bars.astype(np.float32, copy=False).copy()
    cs = cs_medians.astype(np.float32, copy=False)
    n_cal = cs.shape[0]
    for t in range(t_len):
        delta = abs_windows[:, t, :] - prev
        o = bar_ords[:, t].astype(np.int64, copy=False)
        o = np.clip(o, 0, max(n_cal - 1, 0))
        delta -= cs[o]
        out[:, t, :] = delta
        prev = abs_windows[:, t, :]
    return out


def absolute_window_to_relative_torch(
    abs_window: torch.Tensor,
    prev_bar: torch.Tensor,
    bar_ordinals: torch.Tensor,
    cs_medians: torch.Tensor,
) -> torch.Tensor:
    """批量版：abs_window [B,T,5], prev_bar [B,5], bar_ordinals [B,T], cs_medians [n_cal,5]（lag1）。"""
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
    prev_lvl = close[:, :-1]
    denom = torch.maximum(prev_lvl.abs() * 1e-6, torch.full_like(prev_lvl, 1e-8))
    return (close[:, 1:] - prev_lvl) / denom


def volume_ratio_from_levels(vol: torch.Tensor) -> torch.Tensor:
    """绝对成交量层级 → 环比 V_t / V_{t-1}（首根为 1，因果）。"""
    vol = vol.clamp_min(1e-8)
    if vol.shape[1] < 2:
        return torch.ones_like(vol)
    ratio = vol[:, 1:] / vol[:, :-1]
    return torch.cat([torch.ones(vol.shape[0], 1, device=vol.device, dtype=vol.dtype), ratio], dim=1)


def volume_change_from_levels(vol: torch.Tensor) -> torch.Tensor:
    """绝对成交量环比变化率 V_t/V_{t-1} - 1（首根为 0）。"""
    return volume_ratio_from_levels(vol) - 1.0
