"""VQ 码本跨时间泛化诊断。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch

if TYPE_CHECKING:
    from quant_cursor.bpc.model import BPCv2

logger = logging.getLogger(__name__)


@torch.no_grad()
def measure_vq_residual(
    model: BPCv2,
    loader,
    device: str,
    *,
    max_batches: int = 50,
) -> dict[str, float]:
    """统计 z 与 z_q 残差及最近码本距离。"""
    from quant_cursor.bpc.model import _iter_training_batches, _loader_has_gpu_batches, _to_device

    model.eval()
    residual_norms: list[torch.Tensor] = []
    min_dists: list[torch.Tensor] = []
    gpu_batches = _loader_has_gpu_batches(loader)

    for i, batch in enumerate(_iter_training_batches(loader)):
        if i >= max_batches:
            break
        if not gpu_batches:
            batch = _to_device(batch, device)
        z, _, _ = model.encode(batch)
        if z is None:
            continue
        stock_ids = batch.get("stock_ids")
        timestamps = batch.get("timestamps")
        z_vq, z_scale, _ = model._prepare_vq_inputs(z, stock_ids, timestamps)
        cb_gamma, cb_beta = None, None
        if model.use_codebook_film:
            cb_gamma, cb_beta = model.conditioner.codebook_modulation(
                stock_ids, timestamps, z_vq.shape[0], z_vq.device
            )
        dist_mat = model.vq._coarse_distances(z_vq, cb_gamma, cb_beta)
        min_dist, coarse_idx = dist_mat.min(dim=1)
        if cb_gamma is not None and not model.vq.use_cosine_vq:
            min_dist = min_dist.sqrt()
        z_q = model.vq.coarse_embed(coarse_idx)
        if model.use_cosine_vq:
            z_u = torch.nn.functional.normalize(z_vq, dim=-1)
            q_u = torch.nn.functional.normalize(z_q, dim=-1)
            dir_residual = (z_u - q_u).norm(dim=1)
            residual = dir_residual * z_scale.squeeze(-1) if z_scale is not None else dir_residual
        else:
            residual = (z_vq - z_q).norm(dim=1)
        residual_norms.append(residual.cpu())
        min_dists.append(min_dist.cpu())

    if not residual_norms:
        return {}

    r = torch.cat(residual_norms)
    d = torch.cat(min_dists)
    return {
        "residual_mean": float(r.mean()),
        "residual_std": float(r.std()),
        "residual_p95": float(torch.quantile(r, 0.95)),
        "min_distance_mean": float(d.mean()),
        "min_distance_p95": float(torch.quantile(d, 0.95)),
    }


def _token_index(semantics: pd.DataFrame) -> pd.Index:
    if isinstance(semantics.index, pd.MultiIndex):
        return semantics.index.get_level_values(0)
    return semantics.index


def diagnose_codebook_shift(
    model: BPCv2,
    train_loader,
    val_loader,
    device: str,
    *,
    max_samples: int = 8000,
    max_residual_batches: int = 50,
) -> dict:
    """
    对比 train/val 的 token 语义、重叠率与 VQ 残差，量化分布偏移。
    """
    train_result = model.analyze_token_semantics(train_loader, device=device, max_samples=max_samples)
    val_result = model.analyze_token_semantics(val_loader, device=device, max_samples=max_samples)
    train_sem = train_result.get("semantics")
    val_sem = val_result.get("semantics")

    train_res = measure_vq_residual(model, train_loader, device, max_batches=max_residual_batches)
    val_res = measure_vq_residual(model, val_loader, device, max_batches=max_residual_batches)

    report: dict = {
        "train_tokens": int(train_result.get("n_records", 0)),
        "val_tokens": int(val_result.get("n_records", 0)),
        "train_residual": train_res,
        "val_residual": val_res,
    }

    if train_res and val_res:
        report["residual_gap"] = val_res["residual_mean"] - train_res["residual_mean"]
        report["min_distance_gap"] = val_res["min_distance_mean"] - train_res["min_distance_mean"]

    train_df_ok = train_sem is not None and hasattr(train_sem, "empty") and not train_sem.empty
    val_df_ok = val_sem is not None and hasattr(val_sem, "empty") and not val_sem.empty
    if train_df_ok and val_df_ok:
        train_ids = set(_token_index(train_sem).astype(int))
        val_ids = set(_token_index(val_sem).astype(int))
        overlap = train_ids & val_ids
        report["token_overlap_count"] = len(overlap)
        report["token_overlap_rate_val"] = len(overlap) / max(len(val_ids), 1)
        report["token_overlap_rate_train"] = len(overlap) / max(len(train_ids), 1)

        # 验证集高频 token 在训练集上的语义漂移
        if ("vol", "mean") in val_sem.columns:
            val_counts = val_sem[("vol", "count")].sort_values(ascending=False)
            drifts: list[dict] = []
            for token in val_counts.head(10).index.astype(int):
                if token not in train_ids:
                    continue
                tr_vol = float(train_sem.loc[token, ("vol", "mean")])
                va_vol = float(val_sem.loc[token, ("vol", "mean")])
                tr_atk = float(train_sem.loc[token, ("attack", "mean")])
                va_atk = float(val_sem.loc[token, ("attack", "mean")])
                drifts.append(
                    {
                        "token": int(token),
                        "val_count": int(val_counts.loc[token]),
                        "vol_drift": va_vol - tr_vol,
                        "attack_drift": va_atk - tr_atk,
                    }
                )
            report["top_val_token_drifts"] = drifts

    return report


def save_diagnosis_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Codebook diagnosis saved: %s", path)

    if "residual_gap" in report:
        logger.info(
            "VQ residual gap (val-train)=%.4f | token overlap (val)=%.1f%%",
            report["residual_gap"],
            100.0 * report.get("token_overlap_rate_val", 0.0),
        )
