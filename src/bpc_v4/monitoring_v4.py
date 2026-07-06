"""BPC-v4 训练监控指标（方向预测、纯度分类精度等，对齐 v3 recon_trend_sign_acc 角色）。"""

from __future__ import annotations

import torch

from .behavior_features import BEHAVIOR_AGENT_NAMES, NUM_BEHAVIOR_AGENTS, NUM_BEHAVIOR_CLASSES

MOMENTUM_AGENT_IDX = BEHAVIOR_AGENT_NAMES.index("momentum")
REGIME_AGENT_IDX = BEHAVIOR_AGENT_NAMES.index("regime")


def _agent_argmax_sign(pred_cls: torch.Tensor, agent_idx: int) -> torch.Tensor:
    """三档软标签 argmax → 方向：高档 +1、低档 -1、中档 0。"""
    cls = pred_cls[:, agent_idx]
    return torch.where(
        cls == 2,
        torch.ones_like(cls, dtype=torch.float32),
        torch.where(cls == 0, -torch.ones_like(cls, dtype=torch.float32), torch.zeros_like(cls, dtype=torch.float32)),
    )


def compute_step_monitoring(batch: dict, outputs: dict) -> dict[str, float]:
    """
    单 batch 监控量（不参与 loss）。

    - purity_acc / purity_acc_{agent}: 各行为代理硬标签准确率
    - next_day_sign_acc: momentum 代理预测方向 vs 锚点后一日涨跌（物化字段 next_day_sign）
    - next_day_sign_baseline: 多数类方向基线（随机/常数策略参照）
    """
    out: dict[str, float] = {}
    purity_logits = outputs.get("purity_logits")
    purity_target = batch.get("purity_target")
    if purity_logits is None or purity_target is None:
        return out

    logits = purity_logits.detach().float().view(-1, NUM_BEHAVIOR_AGENTS, NUM_BEHAVIOR_CLASSES)
    targets = purity_target.detach().float().view(-1, NUM_BEHAVIOR_AGENTS, NUM_BEHAVIOR_CLASSES)
    targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    pred_cls = logits.argmax(dim=-1)
    tgt_cls = targets.argmax(dim=-1)

    agent_accs = (pred_cls == tgt_cls).float().mean(dim=0)
    out["purity_acc"] = float(agent_accs.mean().item())
    for j, name in enumerate(BEHAVIOR_AGENT_NAMES):
        out[f"purity_acc_{name}"] = float(agent_accs[j].item())

    next_sign = batch.get("next_day_sign")
    if next_sign is None:
        return out

    sign = next_sign.detach().float().view(-1)
    valid = sign.abs() > 0
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        out["next_day_sign_n"] = 0.0
        return out

    sign_v = sign[valid]
    pred_mom = _agent_argmax_sign(pred_cls, MOMENTUM_AGENT_IDX)[valid]
    pred_reg = _agent_argmax_sign(pred_cls, REGIME_AGENT_IDX)[valid]

    out["next_day_sign_n"] = float(n_valid)
    out["next_day_sign_acc"] = float((pred_mom == sign_v).float().mean().item())
    out["next_day_sign_acc_regime"] = float((pred_reg == sign_v).float().mean().item())

    up_rate = float((sign_v > 0).float().mean().item())
    out["next_day_sign_baseline"] = max(up_rate, 1.0 - up_rate)

    return out


def accumulate_monitoring(total: dict[str, float], step: dict[str, float]) -> None:
    for k, v in step.items():
        if k == "next_day_sign_n":
            total[k] = total.get(k, 0.0) + v
            continue
        if k == "next_day_sign_baseline":
            # 加权平均：按有效样本数
            n = step.get("next_day_sign_n", 0.0)
            if n > 0:
                total[k] = total.get(k, 0.0) + v * n
            continue
        total[k] = total.get(k, 0.0) + v


def finalize_monitoring(total: dict[str, float], steps: int) -> dict[str, float]:
    if steps <= 0:
        return {}
    out: dict[str, float] = {}
    n_sign = total.get("next_day_sign_n", 0.0)
    for k, v in total.items():
        if k == "next_day_sign_n":
            out[k] = v
        elif k == "next_day_sign_baseline" and n_sign > 0:
            out[k] = v / n_sign
        else:
            out[k] = v / steps
    return out
