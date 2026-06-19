"""
BPC-v3 FiLM：强化个股/时间调制，修复码本 FiLM 长期近零的问题。

Latent FiLM 在 VQ 前作用于 fusion 输出，随后 prepare_latent 会 normalize 方向。
Codebook FiLM 在 quantize 时调制 coarse 码本，cosine VQ 使用：
  z_adj = z - beta
  w' = normalize(codebook * film_scale_for_cosine(gamma))

共性：对向量做**各维相同**的乘性缩放，在 normalize 后几乎无效；需逐维
差异（可学习 bias）并在 cosine 层去掉 uniform 分量。
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_cursor.bpc.cosine_vq import remove_uniform_gamma_component
from quant_cursor.bpc.model import SymbolTimeFiLM as _BaseSymbolTimeFiLM

logger = logging.getLogger(__name__)


def _init_linear_orthogonal(linear: nn.Linear, gain: float = 1.0) -> None:
    nn.init.orthogonal_(linear.weight, gain=gain)
    nn.init.zeros_(linear.bias)


def _default_dim_bias(dim: int, half_range: float) -> torch.Tensor:
    """逐维不等偏置：即使网络输出近零，调制仍随维度变化。"""
    if dim <= 1:
        return torch.tensor([half_range])
    return torch.linspace(-half_range, half_range, dim)


class SymbolTimeFiLM(_BaseSymbolTimeFiLM):
    """增强版 Symbol+Time FiLM：可学习门控 + 逐维偏置 + 去 uniform gamma。"""

    def __init__(
        self,
        dim: int,
        num_symbols: int = 10000,
        time_emb_dim: int = 16,
        enable_codebook_film: bool = True,
    ):
        super().__init__(
            dim,
            num_symbols=num_symbols,
            time_emb_dim=time_emb_dim,
            enable_codebook_film=enable_codebook_film,
        )
        self.latent_scale_log = nn.Parameter(torch.tensor(0.5))
        self.latent_gate_logit = nn.Parameter(torch.tensor(1.0))
        self.latent_gamma_bias = nn.Parameter(_default_dim_bias(dim, 0.06))
        self.latent_beta_bias = nn.Parameter(torch.zeros(dim))
        self._film_warned = False

        if not self.enable_codebook_film or self.codebook_film is None:
            return

        self.codebook_scale_log = nn.Parameter(torch.tensor(0.55))
        self.codebook_gate_logit = nn.Parameter(torch.tensor(1.0))
        self.codebook_gamma_bias = nn.Parameter(_default_dim_bias(dim, 0.08))
        self.codebook_beta_bias = nn.Parameter(torch.zeros(dim))

        for i, layer in enumerate(self.codebook_film):
            if isinstance(layer, nn.Linear):
                _init_linear_orthogonal(layer, gain=1.0 if i == 0 else 1.2)

    def _record_stats(
        self,
        prefix: str,
        gamma: torch.Tensor,
        scale: torch.Tensor,
        gate: float = 1.0,
    ) -> None:
        """gamma 已含 scale/gate；勿再乘一次。"""
        with torch.no_grad():
            self._last_stats[f"film_{prefix}_gamma_abs"] = float(gamma.abs().mean())
            self._last_stats[f"film_{prefix}_gamma_std"] = float(
                gamma.std(dim=-1).mean()
            )
            self._last_stats[f"film_{prefix}_scale"] = float(scale)
            if prefix == "codebook":
                self._last_stats["film_codebook_gate"] = gate

    def _condition_vector(
        self,
        stock_ids: Optional[torch.Tensor],
        timestamps: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if stock_ids is None or timestamps is None:
            return None
        cond = super()._condition_vector(stock_ids, timestamps, batch_size, device)
        if cond is not None and cond.detach().std().item() < 1e-4:
            logger.warning(
                "FiLM condition vector has near-zero variance (check stock_ids/timestamps in batch)"
            )
        return cond

    def forward(
        self,
        z: torch.Tensor,
        stock_ids: Optional[torch.Tensor],
        timestamps: Optional[torch.Tensor],
    ) -> torch.Tensor:
        cond = self._condition_vector(stock_ids, timestamps, z.shape[0], z.device)
        if cond is None:
            return z
        gamma_beta = self.film(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        latent_scale = F.softplus(self.latent_scale_log).clamp(min=0.45, max=1.0)
        gate = torch.sigmoid(self.latent_gate_logit).clamp(min=0.55)
        gamma = (torch.tanh(gamma) + self.latent_gamma_bias) * latent_scale * gate
        gamma = remove_uniform_gamma_component(gamma)
        beta = (torch.tanh(beta) + self.latent_beta_bias) * latent_scale * gate
        self._record_stats("latent", gamma, latent_scale, float(gate.detach().item()))
        gamma_abs = gamma.detach().abs().mean().item()
        gamma_std = gamma.detach().std(dim=-1).mean().item()
        if not self._film_warned and gamma_abs < 0.02 and gamma_std < 0.01:
            logger.warning(
                "FiLM latent gamma abs mean %.4f std %.4f (gate=%.3f scale=%.3f)",
                gamma_abs,
                gamma_std,
                float(gate.detach().item()),
                float(latent_scale.detach().item()),
            )
            self._film_warned = True
        return z + gamma * z + beta

    def codebook_modulation(
        self,
        stock_ids: Optional[torch.Tensor],
        timestamps: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.enable_codebook_film or self.codebook_film is None:
            return None, None
        cond = self._condition_vector(stock_ids, timestamps, batch_size, device)
        if cond is None:
            return None, None
        gamma_beta = self.codebook_film(cond)
        gamma_net, beta_net = gamma_beta.chunk(2, dim=-1)
        cb_scale = F.softplus(self.codebook_scale_log).clamp(min=0.5, max=2.0)
        gate = torch.sigmoid(self.codebook_gate_logit).clamp(min=0.55)
        gamma = (torch.tanh(gamma_net) + self.codebook_gamma_bias) * cb_scale * gate
        beta = (torch.tanh(beta_net) + self.codebook_beta_bias) * cb_scale * gate
        self._record_stats("codebook", gamma, cb_scale, float(gate.detach().item()))
        return gamma, beta
