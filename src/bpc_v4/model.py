"""bpc_v4 核心模型（参考更优实现）"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GlobalConfig


class TemporalAggregator(nn.Module):
    """时序聚合器（MeanPool + 可训练 Linear）"""

    def __init__(self, d_model: int = 768, proj: bool = True):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model) if proj else nn.Identity()

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        z_agg = z_q.mean(dim=1)
        return self.proj(z_agg)


class BPCV4Model(nn.Module):
    def __init__(self, config: GlobalConfig):
        super().__init__()
        self.cfg = config

        self.temporal_agg = TemporalAggregator(d_model=config.kronos.d_model, proj=True)

        self.norm_z = nn.LayerNorm(config.kronos.d_model)
        self.norm_bpc = nn.LayerNorm(config.bpc.feat_dim)
        self.norm_ctx = nn.LayerNorm(config.context.total_ctx_dim)

        self.stock_embed = nn.Embedding(config.embedding.stock_vocab, config.embedding.stock_emb_dim)
        self.time_proj = nn.Linear(config.embedding.time_raw_dim, config.embedding.time_proj_dim)

        fused_dim = config.kronos.d_model + config.bpc.feat_dim + config.context.total_ctx_dim + config.embedding.total_emb_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, config.fusion.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.fusion.dropout),
        )

        self.purity_head = nn.Linear(config.fusion.hidden_dim, config.head.purity_output_dim)
        self.codebook_head = nn.Linear(config.fusion.hidden_dim, config.head.codebook_output_dim)

        self.purity_weight = config.head.purity_weight
        self.codebook_weight = config.head.codebook_weight

    def forward(self, batch: dict) -> dict:
        z_q = batch["z_q"]
        z_agg = self.temporal_agg(z_q)

        z_norm = self.norm_z(z_agg)
        bpc_norm = self.norm_bpc(batch["bpc_feat"])
        ctx_norm = self.norm_ctx(batch["ctx_feat"])

        stock_emb = self.stock_embed(batch["stock_id"])
        time_emb = self.time_proj(batch["time_emb"])
        emb = torch.cat([stock_emb, time_emb], dim=-1)

        fused = torch.cat([z_norm, bpc_norm, ctx_norm, emb], dim=-1)
        h = self.fusion(fused)

        purity_logits = self.purity_head(h)
        codebook_logits = self.codebook_head(h)

        return {
            "h": h,
            "purity_logits": purity_logits,
            "codebook_logits": codebook_logits,
        }

    def compute_loss(self, batch: dict, outputs: dict) -> dict:
        purity_target = batch.get("purity_target")
        if purity_target is None:
            purity_target = torch.ones_like(outputs["purity_logits"]) / 3

        purity_loss = F.kl_div(
            F.log_softmax(outputs["purity_logits"].view(-1, 3), dim=-1),
            purity_target.view(-1, 3),
            reduction="batchmean",
        )

        s1_target = batch["s1_ids"][:, -1]
        codebook_loss = F.cross_entropy(outputs["codebook_logits"], s1_target)

        total_loss = self.purity_weight * purity_loss + self.codebook_weight * codebook_loss

        return {
            "loss": total_loss,
            "purity_loss": purity_loss,
            "codebook_loss": codebook_loss,
        }
