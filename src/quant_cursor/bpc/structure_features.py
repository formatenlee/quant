"""
市场结构特征子函数（因果）。

仅对从 Qlib 读取的**原始绝对量**（如成交量）做相对化；
已由收益率/波动率估计得到的相对量（如 rv、parkinson、vol 代理）在别处直接使用，不在此二次变换。
"""



from __future__ import annotations



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


def volume_log_level_anomaly(

    volume: torch.Tensor,

    *,

    max_baseline_days: int = 20,

) -> torch.Tensor:

    """Qlib 原始成交量 → 窗口均量相对前半基线的对数比（因果）。



    替代绝对 log(volume).mean()，对全市场成交量抬升更稳健。

    返回 [B, 1]。

    """

    _B, t = volume.shape

    blen = min(max_baseline_days, max(1, t // 2))

    baseline = volume[:, :blen].mean(dim=1, keepdim=True).clamp_min(1e-8)

    window_mean = volume.mean(dim=1, keepdim=True)

    return torch.log((window_mean / baseline).clamp_min(1e-8)).clamp(-5.0, 5.0)





def volume_rel_cv(volume: torch.Tensor) -> torch.Tensor:

    """Qlib 原始成交量 → 窗口内相对成交量标准差（无量纲）。



    rel_vol 均值为 1，故等价于 rel_vol.std()，不依赖 log 均量的绝对量级。

    返回 [B, 1]。

    """

    rel = volume / volume.mean(dim=1, keepdim=True).clamp_min(1e-8)

    return rel.std(dim=1, keepdim=True)


