"""
VQ 后端选择：余弦（球面）vs L2（欧氏）。

通过 vq_mode 参数切换，BehavioralVQ / BPCv2 仅依赖本模块暴露的统一 API。
"""

from __future__ import annotations

from types import ModuleType
from typing import Literal

from quant_cursor.bpc import cosine_vq, l2_vq

VQMode = Literal["cosine", "l2"]
VQ_MODES: tuple[VQMode, ...] = ("cosine", "l2")


def normalize_vq_mode(
    vq_mode: str | None = None,
    *,
    use_cosine_vq: bool | None = None,
    use_normalized_vq: bool | None = None,
) -> VQMode:
    """解析 VQ 模式；use_cosine_vq / use_normalized_vq 为向后兼容布尔参数。"""
    if vq_mode is not None:
        mode = vq_mode.lower().strip()
        if mode not in VQ_MODES:
            raise ValueError(f"vq_mode must be one of {VQ_MODES}, got {vq_mode!r}")
        return mode  # type: ignore[return-value]
    if use_normalized_vq is not None:
        return "cosine" if use_normalized_vq else "l2"
    if use_cosine_vq is not None:
        return "cosine" if use_cosine_vq else "l2"
    return "cosine"


def get_backend(mode: VQMode) -> ModuleType:
    return cosine_vq if mode == "cosine" else l2_vq


def uses_magnitude_split(mode: VQMode) -> bool:
    return mode == "cosine"
