"""v4 独立实现的结构辅助函数（从 bpc.structure_features 简化复制）。"""

import torch


def volume_delta_level_anomaly(
    volume_delta: torch.Tensor,
    *,
    max_baseline_days: int = 20,
) -> torch.Tensor:
    """成交量字段 Δ → 窗口均 Δ 相对前半基线均 Δ 的差（因果）。"""
    _B, t = volume_delta.shape
    blen = min(max_baseline_days, max(1, t // 2))
    baseline = volume_delta[:, :blen].mean(dim=1, keepdim=True)
    window_mean = volume_delta.mean(dim=1, keepdim=True)
    return (window_mean - baseline).clamp(-5.0, 5.0)


def volume_delta_rel_cv(volume_delta: torch.Tensor) -> torch.Tensor:
    """成交量字段 Δ → 窗口内相对标准差（无量纲）。"""
    rel = volume_delta / volume_delta.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
    return rel.std(dim=1, keepdim=True)
