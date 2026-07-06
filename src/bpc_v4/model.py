"""bpc_v4 核心模型：Kronos z_q 作市场状态上下文，BPC 作行为纯度解析（无 codebook 监督）。"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .behavior_features import BEHAVIOR_AGENT_NAMES, NUM_BEHAVIOR_CLASSES
from .config import GlobalConfig
from .metrics_v4 import purity_agent_loss_key

logger = logging.getLogger(__name__)


def _sanitize(x: torch.Tensor, *, clamp: float | None = None) -> torch.Tensor:
    out = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if clamp is not None:
        out = out.clamp(-clamp, clamp)
    return out


class TemporalAggregator(nn.Module):
    """时序聚合器：将 Kronos 时序 z_q 压缩为固定维度。"""

    def __init__(self, d_model: int, proj: bool = True):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model) if proj else nn.Identity()

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.proj(z_q.mean(dim=1))


class BPCV4Model(nn.Module):
    """
    BPC-v4：z_q + BPC + ctx + emb 融合 → purity_head。

    Kronos s1_ids 仅作数据审计/离线分析，不参与训练 loss。
    """

    def __init__(self, config: GlobalConfig):
        super().__init__()
        self.cfg = config

        self.temporal_agg = TemporalAggregator(d_model=config.kronos.d_model, proj=True)

        self.norm_z = nn.LayerNorm(config.kronos.d_model)
        self.norm_bpc = nn.LayerNorm(config.bpc.feat_dim)
        self.norm_ctx = nn.LayerNorm(config.context.total_ctx_dim)

        self.stock_embed = nn.Embedding(config.embedding.stock_vocab, config.embedding.stock_emb_dim)
        self.time_proj = nn.Linear(config.embedding.time_raw_dim, config.embedding.time_proj_dim)

        emb_dim = config.embedding.total_emb_dim
        full_fused_dim = config.kronos.d_model + config.bpc.feat_dim + config.context.total_ctx_dim + emb_dim

        self.fusion = nn.Sequential(
            nn.Linear(full_fused_dim, config.fusion.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.fusion.dropout),
        )

        self.purity_head = nn.Linear(config.fusion.hidden_dim, config.head.purity_output_dim)
        self.purity_weight = config.head.purity_weight

    def forward(self, batch: dict) -> dict:
        z_q = _sanitize(batch["z_q"].float(), clamp=10.0)
        bpc_feat = _sanitize(batch["bpc_feat"].float(), clamp=20.0)
        ctx_feat = _sanitize(batch["ctx_feat"].float(), clamp=20.0)

        z_agg = self.temporal_agg(z_q)
        z_norm = self.norm_z(z_agg)
        bpc_norm = self.norm_bpc(bpc_feat)
        ctx_norm = self.norm_ctx(ctx_feat)

        stock_emb = self.stock_embed(batch["stock_id"].long())
        time_emb = self.time_proj(batch["time_emb"].float())
        emb = torch.cat([stock_emb, time_emb], dim=-1)

        fused = torch.cat([z_norm, bpc_norm, ctx_norm, emb], dim=-1)
        h = self.fusion(fused)
        purity_logits = self.purity_head(h)

        return {
            "h": h,
            "z_agg": z_agg,
            "purity_logits": purity_logits,
        }

    def compute_loss(self, batch: dict, outputs: dict) -> dict:
        purity_logits = outputs["purity_logits"].float()

        purity_target = batch.get("purity_target")
        if purity_target is None:
            if not getattr(self, "_warned_missing_purity", False):
                logger.warning(
                    "batch 缺少 purity_target，使用均匀分布占位；"
                    "请 --force-rebuild-preprocessed 重新物化"
                )
                self._warned_missing_purity = True
            purity_target = torch.full(
                (purity_logits.shape[0], purity_logits.shape[1]),
                1.0 / 3.0,
                device=purity_logits.device,
                dtype=purity_logits.dtype,
            )
        else:
            purity_target = purity_target.float()

        n_agents = len(BEHAVIOR_AGENT_NAMES)
        logits_agents = purity_logits.view(-1, n_agents, NUM_BEHAVIOR_CLASSES)
        target_agents = purity_target.view(-1, n_agents, NUM_BEHAVIOR_CLASSES)
        target_agents = target_agents / target_agents.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        agent_losses: dict[str, torch.Tensor] = {}
        for j, name in enumerate(BEHAVIOR_AGENT_NAMES):
            log_p = F.log_softmax(logits_agents[:, j, :], dim=-1)
            tgt_p = target_agents[:, j, :]
            agent_losses[purity_agent_loss_key(name)] = F.kl_div(log_p, tgt_p, reduction="batchmean")

        purity_loss = sum(agent_losses.values()) / max(len(agent_losses), 1)

        probs = F.softmax(logits_agents, dim=-1)
        purity_entropy = (
            -(probs * probs.log().clamp_min(-20.0)).sum(dim=-1).mean()
        )

        total_loss = self.purity_weight * purity_loss

        return {
            "loss": total_loss,
            "purity_loss": purity_loss,
            "loss_purity_total": purity_loss,
            "purity_entropy": purity_entropy,
            "weighted_purity": self.purity_weight * purity_loss,
            **agent_losses,
        }
