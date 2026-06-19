"""
BPC-v3 余弦 VQ：解码端对 z_scale 做对数压缩，降低幅度噪声。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from quant_cursor.bpc.cosine_vq import MagnitudeSplitVQ, prepare_direction_scale

prepare_latent = prepare_direction_scale


def compress_z_scale_for_decode(z_scale: torch.Tensor) -> torch.Tensor:
    """小幅度线性保留，大幅度对数压缩。"""
    scale = z_scale.clamp_min(0.0)
    return torch.where(scale < 1.0, scale, torch.log(scale.clamp_min(1e-6)) + 1.0)


def merge_for_decode(z_q: torch.Tensor, z_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    if z_scale is None:
        return z_q
    return torch.cat([z_q, compress_z_scale_for_decode(z_scale)], dim=-1)


class DirectionScaleSplit:
    @staticmethod
    def split(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_dir, z_scale, _ = prepare_direction_scale(z)
        return z_dir, z_scale

    @staticmethod
    def merge_for_decode(z_q: torch.Tensor, z_scale: torch.Tensor) -> torch.Tensor:
        return merge_for_decode(z_q, z_scale)


__all__ = [
    "DirectionScaleSplit",
    "MagnitudeSplitVQ",
    "merge_for_decode",
    "prepare_direction_scale",
    "prepare_latent",
]
