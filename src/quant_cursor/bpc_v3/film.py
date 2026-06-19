"""
BPC-v3 FiLM：强化个股/时间调制，修复码本 FiLM 长期近零的问题。

Latent FiLM 在 VQ 前作用于 fusion 输出（见 BPCv2._prepare_vq_inputs），
codebook FiLM 在 quantize 时调制 coarse 码本向量以影响 cosine 最近邻选择。
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_cursor.bpc.model import SymbolTimeFiLM as _BaseSymbolTimeFiLM

logger = logging.getLogger(__name__)


class SymbolTimeFiLM(_BaseSymbolTimeFiLM):
    """增强版 Symbol+Time FiLM：更强的 latent / codebook 调制初值与门控。"""

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
        self.latent_scale_log = nn.Parameter(torch.tensor(1.5))
        self.latent_gate_logit = nn.Parameter(torch.tensor(3.0))
        self._film_warned = False

        if not self.enable_codebook_film or self.codebook_film is None:
            return

        self.codebook_scale_log = nn.Parameter(torch.tensor(0.5))
        self.codebook_gate_logit = nn.Parameter(torch.tensor(1.0))

        for i, layer in enumerate(self.codebook_film):
            if isinstance(layer, nn.Linear):
                gain = 0.8 if i == 0 else 0.15
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)

        last = self.codebook_film[-1]
        if isinstance(last, nn.Linear):
            with torch.no_grad():
                half = last.out_features // 2
                last.bias[:half].fill_(0.08)

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
            logger.warning("FiLM condition vector has near-zero variance (check stock_ids/timestamps in batch)")
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
        latent_scale = F.softplus(self.latent_scale_log).clamp(max=2.0)
        gate = torch.sigmoid(self.latent_gate_logit)
        gamma = torch.tanh(gamma) * latent_scale * gate
        beta = torch.tanh(beta) * latent_scale * gate
        self._record_stats("latent", gamma, latent_scale, float(gate.detach().item()))
        gamma_abs = gamma.detach().abs().mean().item()
        if not self._film_warned and gamma_abs < 0.02:
            logger.warning(
                "FiLM latent gamma abs mean %.4f < 0.02 (gate=%.3f scale=%.3f)",
                gamma_abs,
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
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        cb_scale = F.softplus(self.codebook_scale_log).clamp(max=2.0)
        gate = torch.sigmoid(self.codebook_gate_logit)
        gamma = torch.tanh(gamma) * cb_scale * gate
        beta = torch.tanh(beta) * cb_scale * gate
        self._record_stats("codebook", gamma, cb_scale, float(gate.detach().item()))
        return gamma, beta
