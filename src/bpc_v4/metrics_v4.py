"""BPC-v4 训练指标格式化（对齐 v3 日志风格）。"""

from __future__ import annotations

from .behavior_features import BEHAVIOR_AGENT_NAMES, NUM_BEHAVIOR_CLASSES, PURITY_LOSS_KEYS

PURITY_AGENT_LOSS_KEYS: tuple[str, ...] = tuple(
    f"loss_{PURITY_LOSS_KEYS[name]}" for name in BEHAVIOR_AGENT_NAMES
)

PURITY_AGENT_ACC_KEYS: tuple[str, ...] = tuple(f"purity_acc_{name}" for name in BEHAVIOR_AGENT_NAMES)

LOSS_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "loss",
        "purity_loss",
        "loss_purity_total",
        "weighted_purity",
        *PURITY_AGENT_LOSS_KEYS,
    }
)

HEALTH_METRIC_KEYS: frozenset[str] = frozenset(
    {"purity_entropy", "zq_bpc_corr_mean", "grad_norm"}
)

PREDICTION_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "purity_acc",
        *PURITY_AGENT_ACC_KEYS,
        "next_day_sign_acc",
        "next_day_sign_acc_regime",
        "next_day_sign_baseline",
        "next_day_sign_n",
    }
)

SUMMARY_HIGHLIGHT_KEYS: tuple[str, ...] = (
    "loss",
    "next_day_sign_acc",
    "purity_acc",
    "purity_entropy",
    "zq_bpc_corr_mean",
)


def purity_agent_loss_key(agent_name: str) -> str:
    return f"loss_{PURITY_LOSS_KEYS[agent_name]}"


def _format_pair(key: str, value: float) -> str | None:
    if key in LOSS_METRIC_KEYS or key.startswith("loss_purity_"):
        return f"{key}={value:.6f}"
    if key in HEALTH_METRIC_KEYS:
        if key == "grad_norm":
            return f"{key}={value:.4f}"
        return f"{key}={value:.4f}"
    if key in PREDICTION_METRIC_KEYS:
        if key.startswith("purity_acc") or key.startswith("next_day_sign_acc"):
            return f"{key}={value * 100:.1f}%"
        if key == "next_day_sign_baseline":
            return f"{key}={value * 100:.1f}% (maj)"
        if key == "next_day_sign_n":
            return f"{key}={round(value)}"
        return f"{key}={value:.4f}"
    if key == "weighted_purity":
        return f"{key}={value:.6f}"
    return None


def format_epoch_summary(
    train_metrics: dict[str, float],
    val_metrics: dict[str, float] | None = None,
) -> str:
    """单行摘要：loss / 次日方向 / 纯度。"""

    def _phase_line(prefix: str, m: dict[str, float]) -> str:
        parts: list[str] = []
        if "loss" in m:
            parts.append(f"loss={m['loss']:.4f}")
        if m.get("next_day_sign_n", 0) > 0 and "next_day_sign_acc" in m:
            base = m.get("next_day_sign_baseline", 0.5) * 100
            parts.append(
                f"next_day={m['next_day_sign_acc'] * 100:.1f}% (base {base:.0f}%)"
            )
        if "purity_acc" in m:
            parts.append(f"purity={m['purity_acc'] * 100:.1f}%")
        if "purity_entropy" in m:
            parts.append(f"p_ent={m['purity_entropy']:.2f}")
        if "zq_bpc_corr_mean" in m:
            parts.append(f"z_bpc={m['zq_bpc_corr_mean']:.3f}")
        return f"{prefix}[{' | '.join(parts)}]" if parts else ""

    lines = [_phase_line("train ", train_metrics)]
    if val_metrics:
        vl = _phase_line("val   ", val_metrics)
        if vl:
            lines.append(vl)
    return "  >> " + "  ".join(x for x in lines if x)


def format_metrics_lines(metrics: dict[str, float], *, indent: str = "") -> list[str]:
    """按 Loss / Purity-KL / Prediction / Health 分组输出。"""
    if not metrics:
        return [f"{indent}n/a"]

    loss_parts: list[str] = []
    purity_loss_parts: list[str] = []
    pred_parts: list[str] = []
    health_parts: list[str] = []

    ordered_loss = ["loss", "loss_purity_total", *PURITY_AGENT_LOSS_KEYS]
    seen: set[str] = set()
    for key in ordered_loss:
        if key not in metrics or key in seen:
            continue
        pair = _format_pair(key, float(metrics[key]))
        if pair is None:
            continue
        if key == "loss":
            loss_parts.append(pair)
        elif key.startswith("loss_purity_"):
            purity_loss_parts.append(pair)
        seen.add(key)

    pred_order = [
        "next_day_sign_acc",
        "next_day_sign_acc_regime",
        "next_day_sign_baseline",
        "next_day_sign_n",
        "purity_acc",
        *PURITY_AGENT_ACC_KEYS,
    ]
    for key in pred_order:
        if key not in metrics:
            continue
        pair = _format_pair(key, float(metrics[key]))
        if pair:
            pred_parts.append(pair)
            seen.add(key)

    for key in sorted(metrics):
        if key in seen or key == "weighted_purity":
            continue
        pair = _format_pair(key, float(metrics[key]))
        if pair is None:
            continue
        if key in HEALTH_METRIC_KEYS:
            health_parts.append(pair)
        elif key in LOSS_METRIC_KEYS:
            loss_parts.append(pair)
        elif key in PREDICTION_METRIC_KEYS:
            pred_parts.append(pair)

    lines: list[str] = []
    if loss_parts:
        w_p = metrics.get("weighted_purity")
        if w_p is not None:
            loss_parts.append(f"weighted_purity={w_p:.6f}")
        lines.append(f"{indent}Loss: {', '.join(loss_parts)}")
    if purity_loss_parts:
        lines.append(f"{indent}Purity-KL: {', '.join(purity_loss_parts)}")
    if pred_parts:
        lines.append(f"{indent}Prediction: {', '.join(pred_parts)}")
    if health_parts:
        lines.append(f"{indent}Health: {', '.join(health_parts)}")
    return lines or [f"{indent}n/a"]


def accumulate_tensor_metrics(total: dict[str, float], step_metrics: dict) -> None:
    """累加单 step 的标量 / tensor 指标。"""
    import torch

    for key, value in step_metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            val = float(value.detach().item())
        elif isinstance(value, (int, float)):
            val = float(value)
        else:
            continue
        total[key] = total.get(key, 0.0) + val


def finalize_averaged_metrics(total: dict[str, float], steps: int) -> dict[str, float]:
    if steps <= 0:
        return {}
    return {k: v / steps for k, v in total.items()}
