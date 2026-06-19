"""CLI: python -m quant_cursor.bpc_v3.preflight [--qlib]"""

from __future__ import annotations

import argparse
import logging

import torch

from quant_cursor.bpc_v3.ohlcv_relative import absolute_window_to_relative_torch
from quant_cursor.bpc_v3.diagnostics import run_preflight_audit, run_qlib_structural_preflight
from quant_cursor.config import load_config

logger = logging.getLogger("quant_cursor.bpc_v3.preflight")


def _synthetic_relative_ohlcv(batch: int, lookback: int) -> tuple[torch.Tensor, torch.Tensor]:
    """生成相对化合成窗口（字段 Δ，无截面中值）供 smoke preflight。"""
    abs_win = torch.rand(batch, lookback, 5).abs() * 10 + 1.0
    abs_win[..., 4] = abs_win[..., 4].abs() * 1e6
    prev = abs_win[:, 0] * 0.98
    cs = torch.zeros(lookback, 5)
    ords = torch.arange(lookback)
    rel = absolute_window_to_relative_torch(abs_win, prev, ords.unsqueeze(0).expand(batch, -1), cs)
    return rel, prev


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="BPC-v3 preflight audit")
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lookback", type=int, default=40)
    p.add_argument("--qlib", action="store_true", help="在 qlib 真实数据上审查 path/vol 分布与 batch 漂移")
    p.add_argument("--start", type=str, default="2022-01-01")
    p.add_argument("--end", type=str, default="2026-01-01")
    p.add_argument("--max-instruments", type=int, default=300)
    p.add_argument("--max-samples-per-instrument", type=int, default=40)
    p.add_argument("--sample-size", type=int, default=50_000)
    p.add_argument("--n-batches", type=int, default=30, help="模拟训练 batch 数量")
    args = p.parse_args()

    load_config()

    if args.qlib:
        run_qlib_structural_preflight(
            start=args.start,
            end=args.end,
            max_instruments=args.max_instruments,
            max_samples_per_instrument=args.max_samples_per_instrument,
            sample_size=args.sample_size,
            batch_size=args.batch,
            n_random_batches=args.n_batches,
            day_lookback=args.lookback,
        )
        return 0

    torch.manual_seed(42)
    ohlcv, prev_bar = _synthetic_relative_ohlcv(args.batch, args.lookback)

    report = run_preflight_audit(ohlcv, prev_bar=prev_bar)
    for name, stats in report["symbolic"].items():
        logger.info(
            "%s bins low/mid/high=%.1f/%.1f/%.1f%% entropy=%.3f/%.3f",
            name,
            stats["low_pct"] * 100,
            stats["mid_pct"] * 100,
            stats["high_pct"] * 100,
            stats["label_entropy"],
            stats["max_entropy"],
        )
    logger.info(
        "trend variance ratio=%.1f%%",
        report["trend"]["trend_variance_ratio"] * 100,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
