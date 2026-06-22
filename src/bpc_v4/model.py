"""bpc_v4 核心模型（参考更优实现）"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GlobalConfig


def _sanitize(x: torch.Tensor, *, clamp: float | None = None) -> torch.Tensor:
    out = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if clamp is not None:
        out = out.clamp(-clamp, clamp)
    return out


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
        self.num_codebook_classes = config.head.codebook_output_dim

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
        self.codebook_head = nn.Linear(config.fusion.hidden_dim, self.num_codebook_classes)

        self.purity_weight = config.head.purity_weight
        self.codebook_weight = config.head.codebook_weight

    def _codebook_targets(self, s1_ids: torch.Tensor) -> torch.Tensor:
        """Kronos s1 token id（末 bar），与 codebook_head 全词表一一对应。"""
        return s1_ids[:, -1].long()

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
        codebook_logits = self.codebook_head(h)

        return {
            "h": h,
            "purity_logits": purity_logits,
            "codebook_logits": codebook_logits,
        }

    def compute_loss(self, batch: dict, outputs: dict) -> dict:
        purity_logits = outputs["purity_logits"].float()
        codebook_logits = outputs["codebook_logits"].float()

        purity_target = batch.get("purity_target")
        if purity_target is None:
            purity_target = torch.full_like(purity_logits, 1.0 / 3.0)
        else:
            purity_target = purity_target.float()

        log_probs = F.log_softmax(purity_logits.view(-1, 3), dim=-1)
        target_probs = purity_target.view(-1, 3)
        target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        purity_loss = F.kl_div(log_probs, target_probs, reduction="batchmean")

        s1_target = self._codebook_targets(batch["s1_ids"])
        if s1_target.max().item() >= self.num_codebook_classes:
            raise ValueError(
                f"s1_id {s1_target.max().item()} >= codebook classes {self.num_codebook_classes}; "
                "请运行 sync_kronos_config 或 --force-rebuild-preprocessed"
            )
        codebook_loss = F.cross_entropy(codebook_logits, s1_target)

        total_loss = self.purity_weight * purity_loss + self.codebook_weight * codebook_loss

        return {
            "loss": total_loss,
            "purity_loss": purity_loss,
            "codebook_loss": codebook_loss,
            "codebook_target": s1_target,
        }
