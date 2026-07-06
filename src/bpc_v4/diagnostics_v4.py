"""BPC-v4 训练过程中的可解析性与分布监控（z_q vs BPC 相关性、codebook 预测分布等）"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_zq_bpc_correlation(
    z_q: torch.Tensor,
    bpc_feat: torch.Tensor,
    *,
    sample: int = 2000,
) -> dict[str, float]:
    """
    计算 z_q 各方向与 BPC 26 维特征的平均绝对相关系数（可解析性指标）。

    返回：
        {"mean_abs_corr": float, "max_abs_corr": float, "n_pairs": int}
    """
    if z_q.dim() == 3:
        z_q = z_q.mean(dim=1)  # [B, 768]
    z = z_q.detach().float().cpu().numpy()
    b = bpc_feat.detach().float().cpu().numpy()

    if z.shape[0] > sample:
        idx = np.random.choice(z.shape[0], sample, replace=False)
        z = z[idx]
        b = b[idx]

    # 标准化
    z = (z - z.mean(axis=0, keepdims=True)) / (z.std(axis=0, keepdims=True) + 1e-8)
    b = (b - b.mean(axis=0, keepdims=True)) / (b.std(axis=0, keepdims=True) + 1e-8)

    corr = np.corrcoef(np.hstack([z, b]).T)
    n_z = z.shape[1]
    sub = corr[:n_z, n_z:]  # [768, 26]
    abs_sub = np.abs(sub)
    mean_corr = float(np.nanmean(abs_sub)) if abs_sub.size else 0.0
    if not math.isfinite(mean_corr):
        mean_corr = 0.0

    return {
        "mean_abs_corr": mean_corr,
        "max_abs_corr": float(np.nanmax(abs_sub)) if abs_sub.size else 0.0,
        "n_pairs": int(abs_sub.size),
    }


def save_zq_bpc_correlation_heatmap(
    z_q: torch.Tensor,
    bpc_feat: torch.Tensor,
    out_path: Path | str,
    *,
    sample: int = 3000,
    title: str = "z_q vs BPC Feature Correlation",
) -> None:
    """保存 z_q 方向与 BPC 特征的相关系数热力图（可解析性可视化）"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 未安装，跳过相关性热力图")
        return

    if z_q.dim() == 3:
        z_q = z_q.mean(dim=1)
    z = z_q.detach().float().cpu().numpy()
    b = bpc_feat.detach().float().cpu().numpy()

    if z.shape[0] > sample:
        idx = np.random.choice(z.shape[0], sample, replace=False)
        z = z[idx]
        b = b[idx]

    z = (z - z.mean(axis=0, keepdims=True)) / (z.std(axis=0, keepdims=True) + 1e-8)
    b = (b - b.mean(axis=0, keepdims=True)) / (b.std(axis=0, keepdims=True) + 1e-8)

    corr = np.corrcoef(np.hstack([z, b]).T)
    n_z = z.shape[1]
    sub = corr[:n_z, n_z:]  # [768, 26]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sub, aspect="auto", cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("BPC Feature Index (0-25)")
    ax.set_ylabel("z_q Dimension (0-767, sampled)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson corr")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("z_q-BPC correlation heatmap saved: %s", out_path)


def log_codebook_distribution(
    logits: torch.Tensor,
    targets: torch.Tensor,
    step: int | str = "",
) -> dict[str, float]:
    """记录 codebook 预测分布与目标分布的重叠度（调试用）"""
    with torch.no_grad():
        pred = logits.argmax(dim=-1).cpu().numpy()
        tgt = targets.cpu().numpy()
        pred_unique = len(np.unique(pred))
        tgt_unique = len(np.unique(tgt))
        overlap = len(set(pred) & set(tgt))
        logger.info(
            "[CodebookDist%s] pred_unique=%d, tgt_unique=%d, overlap=%d",
            f"@{step}" if step != "" else "",
            pred_unique,
            tgt_unique,
            overlap,
        )
        return {"pred_unique": pred_unique, "tgt_unique": tgt_unique, "overlap": overlap}


def audit_purity_targets(dataset, *, n_sample: int = 2000, seed: int = 42) -> dict[str, float]:
    """物化后纯度标签分布审计（与 v3 preflight 一致）。"""
    from .behavior_features import BEHAVIOR_AGENT_NAMES

    n = min(len(dataset), n_sample)
    if n == 0:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=n, replace=False)
    targets = torch.stack([dataset[int(i)]["purity_target"] for i in idx])
    # 从 15 维标签反推代理值不可行；直接统计每代理三档边际
    out: dict[str, float] = {}
    for j, name in enumerate(BEHAVIOR_AGENT_NAMES):
        sl = slice(j * 3, j * 3 + 3)
        probs = targets[:, sl].mean(dim=0).clamp_min(1e-8)
        entropy = float((-(probs * probs.log()).sum()).item())
        out[f"{name}_label_entropy"] = entropy
        out[f"{name}_high_pct"] = float((targets[:, j * 3 + 2] > 0.5).float().mean().item())
    logger.info("Purity target audit (n=%d): %s", n, out)
    return out


def audit_s1_token_diversity(
    dataset,
    *,
    vocab_size: int,
    n_sample: int = 5000,
    seed: int = 42,
    fail_if_degenerate: bool = True,
) -> dict[str, float | int]:
    """
    检查物化后 s1_ids 末 token 多样性。

    cb_target_unique=1 通常表示 Kronos 输入未做 per-window z-score，需重建 kronos_cache。
    """
    n = min(len(dataset), n_sample)
    if n == 0:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=n, replace=False)
    last_ids = [int(dataset[int(i)]["s1_ids"][-1].item()) for i in idx]
    unique = len(set(last_ids))
    out = {
        "s1_last_unique": unique,
        "s1_last_sampled": n,
        "s1_vocab": vocab_size,
        "s1_usage_rate": unique / max(vocab_size, 1),
    }
    logger.info(
        "s1_ids audit (n=%d): unique_last=%d vocab=%d usage=%.2f%%",
        n,
        unique,
        vocab_size,
        100.0 * out["s1_usage_rate"],
    )
    min_warn = max(20, min(100, vocab_size // 10))
    if unique <= 2:
        msg = (
            f"s1_ids 末 token 仅 {unique} 个唯一值（坍缩，vocab={vocab_size}）。"
            "请用 per-window z-score 后的 Kronos 重建: precompute_kronos --force-rebuild，"
            "再 train --force-rebuild-preprocessed。"
        )
        if fail_if_degenerate:
            raise RuntimeError(msg)
        logger.error(msg)
    elif unique < min_warn:
        logger.warning(
            "s1_ids 多样性偏低: unique=%d < %d（未达 fail 阈值 2，但建议检查缓存/标的数）",
            unique,
            min_warn,
        )
    return out