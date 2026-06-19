"""BPC-v3 启动前 sanity checks（代理分布 / 标签熵 / trend 方差贡献）。"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quant_cursor.bpc_v3.behavior_features import (
    BEHAVIOR_AGENT_NAMES,
    SYMBOLIC_AUDIT_AGENTS,
    SYMBOLIC_THRESHOLDS,
    audit_symbolic_label_distribution,
    compute_behavior_proxies_stacked,
    symbolic_labels,
)
from quant_cursor.bpc_v3.feature_dims import DAY_STRUCT_FEAT_DIM, TREND_STRUCTURE_SLICE
from quant_cursor.bpc_v3.features import compute_day_features_vectorized
from quant_cursor.bpc_v3.volatility_context import VolatilityStats

logger = logging.getLogger(__name__)


def _proxies_for_labeling_audit(raw_proxies: torch.Tensor) -> torch.Tensor:
    """符号化代理：审计时直接用原始值，无 z-score。"""
    return raw_proxies


def audit_trend_structure_variance(day_features: torch.Tensor) -> dict[str, float]:
    """trend_structure 维度在总特征方差中的占比。"""
    if day_features.shape[1] < DAY_STRUCT_FEAT_DIM:
        raise ValueError(f"expected >= {DAY_STRUCT_FEAT_DIM} dims, got {day_features.shape[1]}")
    struct = day_features[:, :DAY_STRUCT_FEAT_DIM].float()
    total_var = struct.var(dim=0).sum().clamp_min(1e-8)
    trend_var = struct[:, TREND_STRUCTURE_SLICE].var(dim=0).sum()
    ratio = float((trend_var / total_var).item())
    return {
        "trend_variance_ratio": ratio,
        "trend_dim_start": TREND_STRUCTURE_SLICE.start,
        "trend_dim_end": TREND_STRUCTURE_SLICE.stop,
    }


def estimate_symbolic_thresholds(
    proxies: torch.Tensor,
    *,
    quantiles: tuple[float, float] = (0.33, 0.67),
) -> dict[str, tuple[float, float]]:
    """
    从代理矩阵估计三档固定阈值（p33, p67）。

    用于 preflight 复核 SYMBOLIC_THRESHOLDS；momentum 双峰分布时 mid 档可能仍稀少。
    """
    qs = torch.tensor(quantiles, dtype=torch.float32)
    out: dict[str, tuple[float, float]] = {}
    for j, name in enumerate(BEHAVIOR_AGENT_NAMES):
        col = proxies[:, j].float()
        q = torch.quantile(col, qs)
        out[name] = (float(q[0].item()), float(q[1].item()))
    return out


def run_preflight_audit(
    ohlcv: torch.Tensor,
    *,
    prev_bar: torch.Tensor | None = None,
    vol_context: torch.Tensor | None = None,
    warn_mid_pct: float = 0.80,
    warn_trend_ratio: float = 0.30,
    warn_entropy_ratio: float = 0.45,
) -> dict[str, Any]:
    """对一批相对化 OHLCV 窗口运行审查清单，返回指标并在异常时打 warning。"""
    proxies = compute_behavior_proxies_stacked(ohlcv, vol_context, prev_bar=prev_bar)
    label_proxies = _proxies_for_labeling_audit(proxies)
    features = compute_day_features_vectorized(ohlcv, vol_context, prev_bar)
    trend_stats = audit_trend_structure_variance(features)

    symbolic: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(BEHAVIOR_AGENT_NAMES):
        stats = audit_symbolic_label_distribution(label_proxies, agent_index=idx)
        symbolic[name] = stats
        if stats["mid_pct"] > warn_mid_pct:
            logger.warning(
                "Preflight %s: mid bin %.1f%% > %.0f%% — thresholds may be too wide",
                name,
                stats["mid_pct"] * 100,
                warn_mid_pct * 100,
            )
        if stats["label_entropy"] / stats["max_entropy"] < warn_entropy_ratio:
            logger.warning(
                "Preflight %s: label entropy %.3f / %.3f low — labels may be degenerate",
                name,
                stats["label_entropy"],
                stats["max_entropy"],
            )

    if trend_stats["trend_variance_ratio"] > warn_trend_ratio:
        logger.warning(
            "Preflight trend_structure variance ratio %.1f%% > %.0f%% — may dominate encoder",
            trend_stats["trend_variance_ratio"] * 100,
            warn_trend_ratio * 100,
        )

    return {"symbolic": symbolic, "trend": trend_stats}


def _tensor_stats(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "n": 0.0}
    v = values.detach().float().cpu()
    qs = torch.quantile(v, torch.tensor([0.1, 0.5, 0.9]))
    return {
        "mean": float(v.mean().item()),
        "std": float(v.std(unbiased=False).item()),
        "p10": float(qs[0].item()),
        "p50": float(qs[1].item()),
        "p90": float(qs[2].item()),
        "n": float(v.numel()),
    }


def _label_distribution(values: torch.Tensor, lo: float, hi: float) -> dict[str, float]:
    labels = symbolic_labels(values, lo, hi)
    probs = labels.mean(dim=0).clamp_min(1e-8)
    entropy = float((-(probs * probs.log()).sum()).item())
    return {
        "low_pct": float((labels[:, 0] > 0.5).float().mean().item()),
        "mid_pct": float((labels[:, 1] > 0.5).float().mean().item()),
        "high_pct": float((labels[:, 2] > 0.5).float().mean().item()),
        "label_entropy": entropy,
        "max_entropy": math.log(3),
    }


def _batch_label_drift(
    raw_proxies: torch.Tensor,
    *,
    batch_size: int,
    n_batches: int,
    seed: int,
    temporal: bool = False,
    timestamps: torch.Tensor | None = None,
) -> dict[str, Any]:
    """模拟训练 batch，统计符号化代理标签的 batch 间漂移。"""
    n = raw_proxies.shape[0]
    if n < batch_size:
        batch_size = max(2, n // 2)

    if temporal and timestamps is not None:
        order = torch.argsort(timestamps)
    else:
        gen = torch.Generator().manual_seed(seed)
        order = torch.randperm(n, generator=gen)

    per_agent: dict[str, dict[str, list[float]]] = {
        name: {
            "raw_mean": [],
            "raw_std": [],
            "z_mean": [],
            "z_std": [],
            "low_pct": [],
            "mid_pct": [],
            "high_pct": [],
            "entropy": [],
        }
        for name in SYMBOLIC_AUDIT_AGENTS
    }

    max_start = max(0, n - batch_size)
    step = max(1, max_start // max(n_batches - 1, 1))
    starts = list(range(0, max_start + 1, step))[:n_batches]

    for start in starts:
        idx = order[start : start + batch_size]
        batch_raw = raw_proxies[idx]
        batch_labeled = _proxies_for_labeling_audit(batch_raw)
        for name in SYMBOLIC_AUDIT_AGENTS:
            j = BEHAVIOR_AGENT_NAMES.index(name)
            lo, hi = SYMBOLIC_THRESHOLDS[name]
            raw_col = batch_raw[:, j]
            z_col = batch_labeled[:, j]
            lbl = _label_distribution(z_col, lo, hi)
            bucket = per_agent[name]
            bucket["raw_mean"].append(float(raw_col.mean().item()))
            bucket["raw_std"].append(float(raw_col.std(unbiased=False).item()))
            bucket["z_mean"].append(float(z_col.mean().item()))
            bucket["z_std"].append(float(z_col.std(unbiased=False).item()))
            bucket["low_pct"].append(lbl["low_pct"])
            bucket["mid_pct"].append(lbl["mid_pct"])
            bucket["high_pct"].append(lbl["high_pct"])
            bucket["entropy"].append(lbl["label_entropy"])

    summary: dict[str, dict[str, float]] = {}
    for name, bucket in per_agent.items():
        def _std(key: str) -> float:
            vals = bucket[key]
            return float(torch.tensor(vals).std(unbiased=False).item()) if len(vals) > 1 else 0.0

        summary[name] = {
            "n_batches": float(len(starts)),
            "raw_mean_drift_std": _std("raw_mean"),
            "raw_std_drift_std": _std("raw_std"),
            "z_mean_drift_std": _std("z_mean"),
            "z_std_drift_std": _std("z_std"),
            "mid_pct_drift_std": _std("mid_pct"),
            "entropy_drift_std": _std("entropy"),
            "mid_pct_mean": float(np.mean(bucket["mid_pct"])) if bucket["mid_pct"] else 0.0,
            "entropy_mean": float(np.mean(bucket["entropy"])) if bucket["entropy"] else 0.0,
        }
    mode = "temporal" if temporal else "random"
    return {"mode": mode, "batch_size": batch_size, "agents": summary}


def run_qlib_structural_preflight(
    *,
    start: str = "2022-01-01",
    end: str = "2026-01-01",
    max_instruments: int = 300,
    max_samples_per_instrument: int = 40,
    sample_size: int = 50_000,
    batch_size: int = 4096,
    n_random_batches: int = 30,
    day_lookback: int = 40,
    seed: int = 42,
    provider_uri: Path | str | None = None,
) -> dict[str, Any]:
    """
    在 qlib 真实数据上审查 path/vol 代理分布与 batch 标签漂移。

    重点判断：不收敛是否由跨时间/跨 batch 分布差异过大引起。
    """
    from quant_cursor.bpc.dataset import QlibMultiScaleDataset, TemporalSplit, load_qlib_instruments
    from quant_cursor.bpc_v3.dataset import build_datasets
    from quant_cursor.bpc_v3.model import build_scale_registry
    from quant_cursor.config import load_config

    config = load_config()
    qlib_uri = Path(provider_uri) if provider_uri else config.data_dir / "qlib_data"
    manifest = config.meta_dir / "qlib_manifest.parquet"

    instruments = load_qlib_instruments(
        manifest,
        instruments=None,
        asset_types=None,
        max_instruments=max_instruments,
        min_rows=500,
        prefer_minute=False,
        require_minute=False,
    )
    if not instruments:
        raise RuntimeError("qlib manifest 无可用标的")

    registry = build_scale_registry(day_lookback, week_lookback=0, precomputed=False)
    store, train_ds, _val_ds, split = build_datasets(
        instruments,
        start,
        end,
        qlib_uri,
        registry,
        val_ratio=0.15,
        max_samples_per_instrument=max_samples_per_instrument,
        seed=seed,
        materialize=False,
        precompute_features=False,
        precompute_proxies=False,
    )
    vol_stats = VolatilityStats.from_store(store)
    n_total = len(train_ds)
    take = min(sample_size, n_total)
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=gen)[:take].tolist()

    logger.info(
        "Qlib structural preflight: %d instruments, %d train samples, auditing %d windows (%s–%s)",
        len(store.instruments),
        n_total,
        take,
        start,
        end,
    )

    ohlcv_chunks: list[torch.Tensor] = []
    prev_bar_chunks: list[torch.Tensor] = []
    sid_chunks: list[torch.Tensor] = []
    ts_chunks: list[torch.Tensor] = []
    chunk = 2048
    for i in range(0, len(indices), chunk):
        batch_idx = indices[i : i + chunk]
        days: list[torch.Tensor] = []
        prev_bars: list[torch.Tensor] = []
        sids: list[torch.Tensor] = []
        tss: list[torch.Tensor] = []
        for idx in batch_idx:
            sample = train_ds[idx]
            days.append(sample["day"].float())
            sids.append(sample["stock_ids"])
            tss.append(sample["timestamps"])
            if "day_prev_bar" in sample:
                prev_bars.append(sample["day_prev_bar"].float())
        ohlcv_chunks.append(torch.stack(days))
        sid_chunks.append(torch.stack(sids))
        ts_chunks.append(torch.stack(tss))
        if prev_bars:
            prev_bar_chunks.append(torch.stack(prev_bars))

    ohlcv = torch.cat(ohlcv_chunks, dim=0)
    stock_ids = torch.cat(sid_chunks, dim=0)
    timestamps = torch.cat(ts_chunks, dim=0)
    prev_bar_batch = torch.cat(prev_bar_chunks, dim=0) if prev_bar_chunks else None

    ctx = vol_stats.lookup_from_ohlcv_batch(stock_ids, timestamps, ohlcv)
    proxies_with = compute_behavior_proxies_stacked(ohlcv, ctx, prev_bar=prev_bar_batch)
    proxies_without = compute_behavior_proxies_stacked(ohlcv, None, prev_bar=prev_bar_batch)
    labeled_with = _proxies_for_labeling_audit(proxies_with)

    calendar = store.calendar
    years = np.array([calendar[int(ts.item())].year for ts in timestamps])

    report: dict[str, Any] = {
        "meta": {
            "n_instruments": len(store.instruments),
            "n_samples": take,
            "train_end": str(split.train_end.date()),
            "batch_size": batch_size,
        },
        "agents": {},
    }

    for name in SYMBOLIC_AUDIT_AGENTS:
        j = BEHAVIOR_AGENT_NAMES.index(name)
        lo, hi = SYMBOLIC_THRESHOLDS[name]
        raw_with = proxies_with[:, j]
        raw_without = proxies_without[:, j]
        z_with = labeled_with[:, j]

        year_stats: dict[int, dict[str, Any]] = {}
        for year in sorted(set(years.tolist())):
            mask = torch.tensor(years == year)
            if mask.sum().item() < 50:
                continue
            year_raw = raw_with[mask]
            year_z = _proxies_for_labeling_audit(proxies_with[mask])[:, j]
            year_stats[int(year)] = {
                "raw": _tensor_stats(year_raw),
                "labels": _label_distribution(year_z, lo, hi),
            }

        raw_means = [v["raw"]["mean"] for v in year_stats.values()]
        raw_stds = [v["raw"]["std"] for v in year_stats.values()]
        year_mean_spread = float(max(raw_means) - min(raw_means)) if raw_means else 0.0
        year_std_spread = float(max(raw_stds) - min(raw_stds)) if raw_stds else 0.0

        agent_report = {
            "global_raw_with_vol_context": _tensor_stats(raw_with),
            "global_raw_without_vol_context": _tensor_stats(raw_without),
            "global_labels": _label_distribution(z_with, lo, hi),
            "vol_context_effect": {
                "mean_delta": float((raw_with.mean() - raw_without.mean()).item()),
                "std_ratio": float(
                    raw_with.std(unbiased=False).item()
                    / max(raw_without.std(unbiased=False).item(), 1e-6)
                ),
            },
            "by_year": year_stats,
            "year_raw_mean_spread": year_mean_spread,
            "year_raw_std_spread": year_std_spread,
            "random_batches": _batch_label_drift(
                proxies_with,
                batch_size=batch_size,
                n_batches=n_random_batches,
                seed=seed,
                temporal=False,
            ),
            "temporal_batches": _batch_label_drift(
                proxies_with,
                batch_size=batch_size,
                n_batches=n_random_batches,
                seed=seed + 1,
                temporal=True,
                timestamps=timestamps,
            ),
        }
        report["agents"][name] = agent_report

        g = agent_report["global_labels"]
        rb = agent_report["random_batches"]["agents"][name]
        tb = agent_report["temporal_batches"]["agents"][name]
        logger.info(
            "%s global raw mean=%.3f std=%.3f | labels low/mid/high=%.1f/%.1f/%.1f%% entropy=%.3f",
            name,
            agent_report["global_raw_with_vol_context"]["mean"],
            agent_report["global_raw_with_vol_context"]["std"],
            g["low_pct"] * 100,
            g["mid_pct"] * 100,
            g["high_pct"] * 100,
            g["label_entropy"],
        )
        logger.info(
            "%s batch drift (random): mid_pct_std=%.4f entropy_std=%.4f raw_mean_drift=%.4f | "
            "temporal: mid_pct_std=%.4f raw_mean_drift=%.4f | year mean spread=%.3f",
            name,
            rb["mid_pct_drift_std"],
            rb["entropy_drift_std"],
            rb["raw_mean_drift_std"],
            tb["mid_pct_drift_std"],
            tb["raw_mean_drift_std"],
            year_mean_spread,
        )

        if g["mid_pct"] > 0.80 or g["label_entropy"] / g["max_entropy"] < 0.45:
            logger.warning(
                "%s: 全局标签退化 mid=%.1f%% entropy=%.3f — 阈值或 vol_scale 可能不匹配 qlib 分布",
                name,
                g["mid_pct"] * 100,
                g["label_entropy"],
            )
        if rb["mid_pct_drift_std"] > 0.08 or tb["mid_pct_drift_std"] > 0.08:
            logger.warning(
                "%s: batch 间 mid 占比漂移较大 (random=%.3f temporal=%.3f) — 可能存在 batch 依赖标签噪声",
                name,
                rb["mid_pct_drift_std"],
                tb["mid_pct_drift_std"],
            )
        if year_mean_spread > 0.25:
            logger.warning(
                "%s: 跨年份 raw mean 漂移 %.3f — vol_scale 可能未完全消除时间 prior shift",
                name,
                year_mean_spread,
            )

    report["verdict"] = _structural_preflight_verdict(report)
    est = estimate_symbolic_thresholds(proxies_with)
    report["estimated_thresholds"] = est
    logger.info("Estimated thresholds (p33/p67): %s", est)
    logger.info("Verdict: %s", report["verdict"]["summary"])
    for line in report["verdict"].get("ok_points", []):
        logger.info("  [ok] %s", line)
    for line in report["verdict"].get("issues", []):
        logger.info("  [issue] %s", line)
    return report


def _structural_preflight_verdict(report: dict[str, Any]) -> dict[str, Any]:
    """根据指标给出是否需继续优化代码的结论。"""
    issues: list[str] = []
    ok_points: list[str] = []

    for name, agent in report["agents"].items():
        g = agent["global_labels"]
        rb = agent["random_batches"]["agents"][name]
        tb = agent["temporal_batches"]["agents"][name]

        balanced = 0.20 <= g["low_pct"] <= 0.45 and 0.20 <= g["high_pct"] <= 0.45 and g["mid_pct"] <= 0.45
        if name == "momentum":
            # 块级延续/反转评分双峰：接受 low/high 各 ~50%、mid 接近 0
            balanced = 0.35 <= g["low_pct"] <= 0.65 and 0.35 <= g["high_pct"] <= 0.65
        low_batch_drift = rb["mid_pct_drift_std"] <= 0.06 and tb["mid_pct_drift_std"] <= 0.06
        low_temporal_raw_drift = tb["raw_mean_drift_std"] <= 0.05
        moderate_year_shift = agent["year_raw_mean_spread"] <= 0.20

        if balanced:
            ok_points.append(f"{name}: 全局三档均衡 (entropy={g['label_entropy']:.3f})")
        else:
            issues.append(
                f"{name}: 全局标签不均衡 low/mid/high={g['low_pct']:.0%}/{g['mid_pct']:.0%}/{g['high_pct']:.0%}"
            )

        if not low_batch_drift:
            issues.append(
                f"{name}: batch 标签漂移偏大 random_mid_std={rb['mid_pct_drift_std']:.3f} "
                f"temporal_mid_std={tb['mid_pct_drift_std']:.3f}"
            )
        else:
            ok_points.append(f"{name}: batch 间标签稳定 (mid_std≤0.06)")

        if not moderate_year_shift:
            issues.append(f"{name}: 跨年份 raw 均值漂移 {agent['year_raw_mean_spread']:.3f}")
        elif not low_temporal_raw_drift:
            issues.append(f"{name}: 时间序 batch raw 均值漂移 {tb['raw_mean_drift_std']:.3f}")
        else:
            ok_points.append(f"{name}: 跨时间 raw 漂移可控")

        vol_eff = agent["vol_context_effect"]["std_ratio"]
        if abs(vol_eff - 1.0) > 0.15:
            ok_points.append(f"{name}: vol_context 改变了 raw 离散度 (std_ratio={vol_eff:.2f})")

    label_issues = [x for x in issues if "batch 标签" in x or "全局标签" in x]
    temporal_raw_issues = [x for x in issues if "跨年份" in x or "时间序" in x]

    if not label_issues and not temporal_raw_issues:
        summary = "分布与标签机制健康；path/vol 不收敛更可能来自模型/FiLM/优化器，而非标签分布问题"
        need_code_change = False
    elif label_issues:
        if any("batch 标签" in x for x in label_issues):
            summary = "存在 batch 依赖标签噪声；应调整 labeling 管线（非单纯调阈值）"
        else:
            summary = "全局标签不均衡；优先调 SYMBOLIC_THRESHOLDS 或 vol_scale 映射"
        need_code_change = True
    elif temporal_raw_issues and not label_issues:
        summary = (
            "跨时间 raw 漂移偏大，但 batch 内标签仍均衡稳定；"
            "这不解释 path/vol 纯度 loss 不收敛，优先查 FiLM/头容量/过拟合"
        )
        need_code_change = False
    else:
        summary = "指标边界；建议小范围调阈值后开短训验证"
        need_code_change = False


    return {
        "summary": summary,
        "need_code_change": need_code_change,
        "issues": issues,
        "ok_points": ok_points,
    }
