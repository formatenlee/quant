"""三层波动率参照：标的自身历史、截面市场、全局基准。"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger(__name__)

# 全市场 log-return RV 中位基准（与 behavior 符号代理解耦，仅用于 vol_context dim2）
GLOBAL_BASELINE_VOL = 0.025
HISTORY_LOOKBACK_DAYS = 252
ROLLING_VOL_WINDOW = 20


def _rolling_realized_vol(close: np.ndarray, window: int = ROLLING_VOL_WINDOW) -> np.ndarray:
    """Daily rolling std of log returns; length = len(close)."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float32)
    if n < 2:
        return out
    log_ret = np.log(close[1:] / np.clip(close[:-1], 1e-8, None))
    for t in range(window, len(log_ret) + 1):
        out[t] = float(log_ret[t - window : t].std())
    # forward-fill leading NaNs with first valid
    valid = np.where(~np.isnan(out))[0]
    if valid.size > 0:
        out[: valid[0]] = out[valid[0]]
    return out


@dataclass
class VolatilityStats:
    """预计算的波动率统计，供物化样本 lookup。"""

    global_baseline: float = GLOBAL_BASELINE_VOL
    history_lookback: int = HISTORY_LOOKBACK_DAYS
    # symbol_id -> daily rolling vol aligned to instrument series
    symbol_daily_rv: dict[int, np.ndarray] = field(default_factory=dict)
    # calendar ordinal -> (mean, std) cross-section of daily rv
    cross_section: dict[int, tuple[float, float]] = field(default_factory=dict)
    # (symbol_id, calendar ordinal) -> historical percentile in [0, 1]
    own_percentile: dict[tuple[int, int], float] = field(default_factory=dict)
    # (symbol_id, calendar ordinal) -> 锚点日 log-return RV（绝对 close 上预计算，与 dim2 一致）
    anchor_rv: dict[tuple[int, int], float] = field(default_factory=dict)

    @classmethod
    def from_store(cls, store) -> VolatilityStats:
        """从 QlibInstrumentStore 构建历史 + 截面统计。"""
        date_to_ordinal = {d: i for i, d in enumerate(store.calendar)}
        symbol_daily_rv: dict[int, np.ndarray] = {}
        cross_lists: dict[int, list[float]] = defaultdict(list)

        for qlib_id, series in store._cache.items():
            sid = store.symbol_to_id[qlib_id]
            close = series.ohlcv[:, 3].astype(np.float64)
            daily_rv = _rolling_realized_vol(close)
            symbol_daily_rv[sid] = daily_rv

            for t, rv_val in enumerate(daily_rv):
                if np.isnan(rv_val):
                    continue
                ord_key = date_to_ordinal.get(series.dates[t])
                if ord_key is not None:
                    cross_lists[int(ord_key)].append(float(rv_val))

        cross_section = {
            k: (float(np.mean(v)), float(np.std(v)) + 1e-8)
            for k, v in cross_lists.items()
            if v
        }

        own_percentile: dict[tuple[int, int], float] = {}
        anchor_rv: dict[tuple[int, int], float] = {}
        lookback = cls.history_lookback
        for qlib_id, series in store._cache.items():
            sid = store.symbol_to_id[qlib_id]
            daily_rv = symbol_daily_rv[sid]
            for t in range(len(daily_rv)):
                if np.isnan(daily_rv[t]):
                    continue
                ord_key = date_to_ordinal.get(series.dates[t])
                if ord_key is None:
                    continue
                key = (sid, int(ord_key))
                anchor_rv[key] = float(daily_rv[t])
                start = max(0, t - lookback + 1)
                hist = daily_rv[start:t]
                hist = hist[~np.isnan(hist)]
                if hist.size < 10:
                    pct = 0.5
                else:
                    pct = float((hist <= daily_rv[t]).mean())
                own_percentile[(sid, int(ord_key))] = pct

        all_dates: set = set()
        for series in store._cache.values():
            all_dates.update(series.dates)
        missing = all_dates - set(date_to_ordinal.keys())
        if missing:
            logger.warning(
                "VolatilityStats: %d instrument dates missing from calendar; cross-section may be incomplete",
                len(missing),
            )

        stats = cls(
            symbol_daily_rv=symbol_daily_rv,
            cross_section=cross_section,
            own_percentile=own_percentile,
            anchor_rv=anchor_rv,
        )
        if own_percentile:
            pct_vals = np.array(list(own_percentile.values()), dtype=np.float64)
            p10, p50, p90 = np.percentile(pct_vals, [10, 50, 90])
            logger.info(
                "VolatilityStats own_percentile distribution: p10=%.3f p50=%.3f p90=%.3f (n=%d)",
                p10,
                p50,
                p90,
                len(pct_vals),
            )
            if p10 > 0.15 or p90 < 0.85:
                logger.warning(
                    "VolatilityStats own_percentile may be miscalibrated (expected ~uniform): p10=%.3f p90=%.3f",
                    p10,
                    p90,
                )
        logger.info(
            "VolatilityStats: %d symbols, %d cross-section dates, %d own-percentile keys",
            len(symbol_daily_rv),
            len(cross_section),
            len(own_percentile),
        )
        return stats

    def lookup(
        self,
        stock_ids: torch.Tensor,
        timestamps: torch.Tensor,
        window_rv: torch.Tensor,
    ) -> torch.Tensor:
        """
        返回 [B, 3]：
        - dim0: 自身历史分位数 [0, 1]
        - dim1: 截面 z-score（相对同期全市场）
        - dim2: 相对全局基准的对数比率
        """
        b = stock_ids.shape[0]
        device = window_rv.device
        dtype = window_rv.dtype
        out = torch.zeros(b, 3, device=device, dtype=dtype)
        out[:, 0] = 0.5
        sids = stock_ids.detach().cpu().numpy()
        ts = timestamps.detach().cpu().numpy()
        wrv = window_rv.detach().cpu().numpy()

        for i in range(b):
            sid = int(sids[i])
            ord_key = int(ts[i])
            rv = float(wrv[i])

            pct = self.own_percentile.get((sid, ord_key))
            if pct is not None:
                out[i, 0] = pct

            cs = self.cross_section.get(ord_key)
            if cs is not None:
                cs_mean, cs_std = cs
                out[i, 1] = (rv - cs_mean) / cs_std

            ratio = rv / self.global_baseline
            out[i, 2] = float(np.log(max(ratio, 1e-8)))

        return out

    def lookup_at_anchor(
        self,
        stock_ids: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """
        从绝对 close 预计算的锚点 RV 构建 vol_context [B, 3]。

        不依赖相对化 OHLCV 窗口，避免 close_Δ std 与 log-return RV 混用。
        """
        b = stock_ids.shape[0]
        device = stock_ids.device
        dtype = torch.float32
        out = torch.zeros(b, 3, device=device, dtype=dtype)
        out[:, 0] = 0.5
        sids = stock_ids.detach().cpu().numpy()
        ts = timestamps.detach().cpu().numpy()

        for i in range(b):
            sid = int(sids[i])
            ord_key = int(ts[i])
            key = (sid, ord_key)
            rv = self.anchor_rv.get(key)
            if rv is None:
                series = self.symbol_daily_rv.get(sid)
                if series is not None and len(series):
                    valid = series[~np.isnan(series)]
                    rv = float(valid[-1]) if valid.size else self.global_baseline
                else:
                    rv = self.global_baseline
            rv = float(rv)

            pct = self.own_percentile.get(key)
            if pct is not None:
                out[i, 0] = pct

            cs = self.cross_section.get(ord_key)
            if cs is not None:
                cs_mean, cs_std = cs
                out[i, 1] = (rv - cs_mean) / cs_std

            out[i, 2] = float(np.log(max(rv / self.global_baseline, 1e-8)))

        return out

    def lookup_from_ohlcv_batch(
        self,
        stock_ids: torch.Tensor,
        timestamps: torch.Tensor,
        ohlcv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del ohlcv
        return self.lookup_at_anchor(stock_ids, timestamps)
