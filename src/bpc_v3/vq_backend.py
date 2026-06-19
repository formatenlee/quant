"""BPC-v3 VQ 后端：cosine 使用 log 压缩后的 merge_for_decode。"""

from __future__ import annotations

from types import ModuleType
from typing import Literal

from quant_cursor.bpc import l2_vq
from quant_cursor.bpc_v3 import cosine_vq

VQMode = Literal["cosine", "l2"]
VQ_MODES: tuple[VQMode, ...] = ("cosine", "l2")


def normalize_vq_mode(
    vq_mode: str | None = None,
    *,
    use_cosine_vq: bool | None = None,
    use_normalized_vq: bool | None = None,
) -> VQMode:
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
