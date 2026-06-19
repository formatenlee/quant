"""
BPC-v2 全量训练 — pyqlib 日线，复权 + 时间切分 + 指标日志。

全量 1000 epoch 示例（80/20 切分，每 5 epoch 全量验证）:
  python -m quant_cursor.bpc.train --full --epochs 1000 --device cuda --seed 42 \\
    --day-lookback 40 --week-lookback 24 --val-every 5

监控指标:
  data/checkpoints/bpc/run_*/metrics.jsonl
  data/checkpoints/bpc/run_*/metrics.csv
  data/checkpoints/bpc/run_*/train.log
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from quant_cursor.bpc.dataset import (
    DEFAULT_TRAIN_INSTRUMENTS,
    GpuCachedDataset,
    TemporalSplit,
    build_datasets,
    load_materialized_dataset,
    load_qlib_instruments,
    save_materialized_dataset,
)
from quant_cursor.bpc.diagnostics import diagnose_codebook_shift, save_diagnosis_report
from quant_cursor.bpc.metrics import MetricsLogger
from quant_cursor.bpc.model import (
    BPCv2,
    adapt_codebook_on_loader,
    build_scale_registry,
    eval_epoch,
    precompute_normalizers,
    precompute_purity_thresholds,
    train_epoch,
)
from quant_cursor.bpc.seed import seed_worker, set_seed
from quant_cursor.config import load_config

logger = logging.getLogger("quant_cursor.bpc.train")


class _FlushingStreamHandler(logging.StreamHandler):
    """控制台 handler：每条日志立即 flush，便于实时调试。"""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()
        if self.stream and hasattr(self.stream, "flush"):
            self.stream.flush()


def setup_logging(log_dir: Path, *, console: bool = True) -> None:
    """同时写入 run 目录 train.log，并（默认）打印到启动终端。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # 统一挂载到 quant_cursor.bpc，train/dataset 子 logger 自动继承
    bpc_logger = logging.getLogger("quant_cursor.bpc")
    bpc_logger.setLevel(logging.INFO)
    bpc_logger.handlers.clear()
    bpc_logger.propagate = False

    fh = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
    fh.setFormatter(fmt)
    bpc_logger.addHandler(fh)

    if console:
        sh = _FlushingStreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        bpc_logger.addHandler(sh)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BPC-v2 on full Qlib daily data")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--start", type=str, default="1990-01-01")
    p.add_argument("--end", type=str, default="2026-12-31")
    p.add_argument("--full", action="store_true", help="使用 manifest 全量标的，不限制采样数")
    p.add_argument("--instruments", nargs="*", default=None)
    p.add_argument("--max-instruments", type=int, default=None)
    p.add_argument("--asset-types", nargs="*", default=None)
    p.add_argument("--val-ratio", type=float, default=0.20, help="验证集占交易日比例（默认 0.20 = 80/20 切分）")
    p.add_argument("--day-lookback", type=int, default=40, help="日尺度回看窗口（交易日数）")
    p.add_argument("--week-lookback", type=int, default=24, help="周尺度回看窗口（周数）")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-coarse", type=int, default=128, help="VQ 粗码本大小")
    p.add_argument("--purity-weight", type=float, default=0.5, help="核心行为代理(vol/attack/amount)纯度权重")
    p.add_argument(
        "--extended-purity-weight",
        type=float,
        default=0.15,
        help="扩展结构代理(path/vp_sym/vol_struct/price_dyn/participation)纯度权重",
    )
    p.add_argument("--recon-weight", type=float, default=1.0)
    p.add_argument("--diversity-weight", type=float, default=0.15, help="码本使用熵正则（略提高以抗坍缩）")
    p.add_argument("--vq-adapt-lr", type=float, default=1e-5, help="验证期在线码本适应学习率")
    p.add_argument(
        "--vq-adapt-on-val",
        action="store_true",
        help="验证时启用在线码本适应（encoder 仍 eval，仅码本漂移）",
    )
    p.add_argument(
        "--vq-dead-code-threshold",
        type=float,
        default=0.01,
        help="训练期 EMA 使用率低于此比例的码本将被复活；0=关闭",
    )
    p.add_argument(
        "--diagnose-on-val",
        action="store_true",
        help="验证后码本诊断（较重，建议配合 --diagnose-every）",
    )
    p.add_argument(
        "--diagnose-every",
        type=int,
        default=25,
        help="每 N 次验证才跑一次完整 diagnose（默认 25）",
    )
    p.add_argument(
        "--diagnose-max-samples",
        type=int,
        default=3000,
        help="diagnose 语义分析最大样本数",
    )
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader 预取 batch 数（num_workers>0）")
    p.add_argument("--no-materialize", action="store_true", help="禁用样本预物化（省内存但更慢）")
    p.add_argument(
        "--no-precompute-proxies",
        action="store_true",
        help="物化时不预计算 behavior_proxies（每 batch 现算，较慢）",
    )
    p.add_argument("--val-every", type=int, default=5, help="每 N epoch 做一次全量验证")
    p.add_argument(
        "--val-max-batches",
        type=int,
        default=0,
        help="验证最多 batch 数；0=全量验证集（默认）",
    )
    p.add_argument("--seed", type=int, default=42, help="随机种子（训练可复现）")
    p.add_argument(
        "--non-deterministic",
        action="store_true",
        help="关闭 cudnn 确定性（更快但不可完全复现）",
    )
    p.add_argument("--amp", action="store_true", help="CUDA 混合精度训练")
    p.add_argument("--compile", action="store_true", help="torch.compile 加速（PyTorch 2+）")
    p.add_argument(
        "--max-samples-per-instrument",
        type=int,
        default=None,
        help="每标的窗口上限；--full 时默认不限制",
    )
    p.add_argument("--norm-max-batches", type=int, default=2000, help="归一化统计最多 batch 数（在线累积）")
    p.add_argument("--save-every", type=int, default=50, help="每 N epoch 存 checkpoint")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--device", type=str, default=None, help="cuda / cuda:0 / cpu；默认自动检测")
    p.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复")
    p.add_argument("--no-console-log", action="store_true", help="不在终端打印，仅写 train.log")
    p.add_argument("--min-rows", type=int, default=60)
    p.add_argument(
        "--preprocessed-dir",
        type=str,
        default=None,
        help="使用预先物化并保存到磁盘的数据集（推荐 Linux 大规模训练）。首次运行后可直接加载，跳过 qlib + 窗口化。",
    )
    p.add_argument(
        "--save-preprocessed",
        type=str,
        default=None,
        help="训练前将物化后的数据集保存到指定目录，后续可通过 --preprocessed-dir 直接加载。",
    )
    p.add_argument(
        "--num-symbols",
        type=int,
        default=10000,
        help="Symbol embedding table size for SymbolTimeFiLM (per-symbol stylistic offset)",
    )
    p.add_argument(
        "--labeling-mode",
        type=str,
        default="per_stock",
        choices=["global", "per_stock", "batch"],
        help="Behavior proxy labeling strategy: per_stock recommended for heterogeneous instruments (ETF/index vs stock)",
    )
    p.add_argument(
        "--preprocess-only",
        action="store_true",
        help="仅构建/保存预处理数据后退出（配合 --save-preprocessed，跳过训练）",
    )
    p.add_argument(
        "--gpu-cache-data",
        action="store_true",
        help="将预处理数据一次性加载到 GPU（消除 H2D 传输，需配合 num_workers=0）",
    )
    return p.parse_args()


def _format_metrics(metrics: dict[str, float]) -> str:
    parts = []
    for k, v in sorted(metrics.items()):
        if k.startswith("loss") or k == "grad_norm":
            parts.append(f"{k}={v:.6f}")
        elif k.startswith("vq_"):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v:.4f}")
    return ", ".join(parts)


def _ensure_run_dir(log_dir: Path) -> Path:
    """Create run output directory (may be missing after long preprocessing on some filesystems)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _save_split_meta(
    log_dir: Path,
    split: TemporalSplit,
    instruments: list[str],
    *,
    val_ratio: float,
    seed: int,
    day_lookback: int,
    week_lookback: int,
) -> None:
    _ensure_run_dir(log_dir)
    meta = {
        "data_start": str(split.data_start.date()),
        "data_end": str(split.data_end.date()),
        "train_end": str(split.train_end.date()),
        "val_start": str(split.val_start.date()),
        "val_end": str(split.val_end.date()),
        "val_ratio": val_ratio,
        "train_ratio": round(1.0 - val_ratio, 4),
        "seed": seed,
        "day_lookback": day_lookback,
        "week_lookback": week_lookback,
        "n_instruments": len(instruments),
        "leakage_policy": (
            "train anchors <= train_end with full daily window <= train_end; "
            "val anchors >= val_start; normalizer fit on train only"
        ),
    }
    (log_dir / "split_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_run_config(log_dir: Path, args: argparse.Namespace) -> None:
    _ensure_run_dir(log_dir)
    cfg = {k: getattr(args, k) for k in vars(args)}
    (log_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_datasets(
    args: argparse.Namespace,
    *,
    instrument_list: list[str],
    qlib_uri: Path,
    registry,
) -> tuple[object, object, TemporalSplit | None, object | None]:
    """Build datasets from qlib, or load pre-materialized tensors from disk."""
    if args.preprocessed_dir:
        pre_dir = Path(args.preprocessed_dir)
        logger.info("Loading pre-materialized dataset from %s", pre_dir)
        train_ds = load_materialized_dataset(pre_dir / "train")
        val_ds = load_materialized_dataset(pre_dir / "val")
        return train_ds, val_ds, None, None

    use_full = args.full or (args.instruments is None and not args.max_instruments)
    max_spi = None if use_full else (args.max_samples_per_instrument or 200)
    store, train_ds, val_ds, split = build_datasets(
        instruments=instrument_list,
        start=args.start,
        end=args.end,
        provider_uri=qlib_uri,
        registry=registry,
        val_ratio=args.val_ratio,
        max_samples_per_instrument=max_spi,
        materialize=not args.no_materialize,
        share_memory=args.num_workers > 0,
        precompute_proxies=not args.no_precompute_proxies,
        seed=args.seed,
    )
    if args.save_preprocessed:
        save_dir = Path(args.save_preprocessed)
        logger.info("Saving preprocessed datasets to %s ...", save_dir)
        save_materialized_dataset(train_ds, save_dir / "train")
        save_materialized_dataset(val_ds, save_dir / "val")
        if split is not None and store is not None:
            _save_split_meta(
                save_dir,
                split,
                store.instruments,
                val_ratio=args.val_ratio,
                seed=args.seed,
                day_lookback=args.day_lookback,
                week_lookback=args.week_lookback,
            )
        logger.info("Preprocessed data saved. Next run: --preprocessed-dir %s", save_dir)
    return train_ds, val_ds, split, store


def main() -> int:
    args = parse_args()
    set_seed(args.seed, deterministic=not args.non_deterministic)

    config = load_config(Path(args.config) if args.config else None)
    qlib_uri = config.data_dir / "qlib_data"
    manifest = config.meta_dir / "qlib_manifest.parquet"

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else config.data_dir / "checkpoints" / "bpc" / run_name
    setup_logging(out_dir, console=not args.no_console_log)
    _save_run_config(out_dir, args)
    metrics_logger = MetricsLogger(out_dir)

    registry = build_scale_registry(args.day_lookback, args.week_lookback)

    use_full = args.full or (args.instruments is None and not args.max_instruments)
    instrument_list = load_qlib_instruments(
        manifest,
        instruments=None if use_full else (args.instruments or DEFAULT_TRAIN_INSTRUMENTS),
        asset_types=args.asset_types,
        max_instruments=args.max_instruments,
        min_rows=args.min_rows,
    )

    logger.info(
        "Run dir: %s | instruments=%d | full=%s | epochs=%d | val_ratio=%.2f | seed=%d | day_lb=%d | week_lb=%d",
        out_dir,
        len(instrument_list),
        use_full,
        args.epochs,
        args.val_ratio,
        args.seed,
        args.day_lookback,
        args.week_lookback,
    )

    train_ds, val_ds, split, store = _resolve_datasets(
        args,
        instrument_list=instrument_list,
        qlib_uri=qlib_uri,
        registry=registry,
    )

    _ensure_run_dir(out_dir)

    if args.preprocess_only:
        logger.info("Preprocess-only mode: datasets ready, exiting without training.")
        if split is not None and store is not None:
            _save_split_meta(
                out_dir,
                split,
                store.instruments,
                val_ratio=args.val_ratio,
                seed=args.seed,
                day_lookback=args.day_lookback,
                week_lookback=args.week_lookback,
            )
        return 0

    if split is not None and store is not None:
        _save_split_meta(
            out_dir,
            split,
            store.instruments,
            val_ratio=args.val_ratio,
            seed=args.seed,
            day_lookback=args.day_lookback,
            week_lookback=args.week_lookback,
        )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    if args.gpu_cache_data and use_cuda:
        if not isinstance(train_ds, GpuCachedDataset):
            logger.info("Enabling GPU data cache (train + val) on %s", device)
            train_ds = GpuCachedDataset(train_ds, device=device)  # type: ignore[arg-type]
            val_ds = GpuCachedDataset(val_ds, device=device)  # type: ignore[arg-type]
        if args.num_workers > 0:
            logger.warning(
                "--gpu-cache-data requires num_workers=0; overriding num_workers from %d to 0",
                args.num_workers,
            )
            args.num_workers = 0

    # Linux 优化：启用 cudnn benchmark + fork 上下文
    if use_cuda:
        torch.backends.cudnn.benchmark = True
    import platform

    mp_context = "fork" if platform.system() == "Linux" and args.num_workers > 0 else None

    if args.preprocessed_dir and not args.gpu_cache_data and args.num_workers > 8:
        logger.warning(
            "Preprocessed in-memory dataset with num_workers=%d may add IPC overhead; "
            "try --num-workers 0 or 4, or --gpu-cache-data if GPU memory allows",
            args.num_workers,
        )

    pin_memory = use_cuda and not args.gpu_cache_data
    loader_kwargs: dict = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        if mp_context:
            loader_kwargs["multiprocessing_context"] = mp_context

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=True,
        generator=train_generator,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    )

    model = BPCv2(
        registry=registry,
        unified_dim=128,
        num_coarse=args.num_coarse,
        primary_scale="day",
        recon_weight=args.recon_weight,
        purity_weight=args.purity_weight,
        extended_purity_weight=args.extended_purity_weight,
        diversity_weight=args.diversity_weight,
        vq_adapt_lr=args.vq_adapt_lr,
        vq_dead_code_threshold=args.vq_dead_code_threshold,
        num_symbols=args.num_symbols,
        labeling_mode=args.labeling_mode,
    )
    model.to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
        logger.info("torch.compile enabled")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    logger.info(
        "Device=%s | train_samples=%d | val_samples=%d | batch=%d | workers=%d | materialize=%s | amp=%s",
        device,
        len(train_ds),
        len(val_ds),
        args.batch_size,
        args.num_workers,
        not args.no_materialize,
        args.amp,
    )
    if split is not None:
        logger.info(
            "Split: train_end=%s | val_start=%s | val_every=%d | val_full=%s | seed=%d",
            split.train_end.date(),
            split.val_start.date(),
            args.val_every,
            args.val_max_batches <= 0,
            args.seed,
        )
    else:
        logger.info(
            "Split: preprocessed (no TemporalSplit metadata) | val_every=%d | seed=%d",
            args.val_every,
            args.seed,
        )

    precompute_normalizers(
        model,
        train_loader,
        device=device,
        max_batches=args.norm_max_batches,
    )
    precompute_purity_thresholds(
        model,
        train_loader,
        device=device,
        max_batches=args.norm_max_batches,
    )

    best_val = float("inf")
    non_blocking = pin_memory
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and args.amp)
    val_max = args.val_max_batches if args.val_max_batches > 0 else None

    val_pass = 0
    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            amp=args.amp,
            non_blocking=non_blocking,
            scaler=scaler,
        )
        val_metrics: dict[str, float] = {}
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val_pass += 1
            val_metrics = eval_epoch(
                model,
                val_loader,
                device=device,
                max_batches=val_max,
                amp=args.amp,
                non_blocking=non_blocking,
            )
            if args.vq_adapt_on_val:
                n_steps = adapt_codebook_on_loader(
                    model,
                    val_loader,
                    device=device,
                    max_batches=val_max,
                    non_blocking=non_blocking,
                )
                logger.info("VQ codebook adapted on val (%d batches, lr=%.1e)", n_steps, args.vq_adapt_lr)
            if args.diagnose_on_val and (
                val_pass % args.diagnose_every == 0 or epoch == args.epochs - 1
            ):
                diag = diagnose_codebook_shift(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    max_samples=args.diagnose_max_samples,
                    max_residual_batches=30,
                )
                save_diagnosis_report(diag, out_dir / f"codebook_diag_epoch_{epoch:04d}.json")
                if "residual_gap" in diag:
                    logger.info(
                        "Diag epoch %d | residual_gap=%.4f | val_usage=%.3f | overlap=%.1f%%",
                        epoch,
                        diag["residual_gap"],
                        val_metrics.get("vq_usage_rate", 0.0),
                        100.0 * diag.get("token_overlap_rate_val", 0.0),
                    )
        scheduler.step()

        record = {
            "epoch": epoch,
            "phase": "epoch",
            "lr": lr,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
        }
        for k, v in train_metrics.items():
            record[f"train_{k}"] = v
        for k, v in val_metrics.items():
            record[f"val_{k}"] = v

        metrics_logger.log(record)

        if train_metrics:
            val_str = _format_metrics(val_metrics) if val_metrics else "n/a"
            extra = ""
            if val_metrics:
                tr_res = train_metrics.get("vq_residual_mean")
                va_res = val_metrics.get("vq_residual_mean")
                if tr_res is not None and va_res is not None:
                    extra = f" | residual train/val={tr_res:.4f}/{va_res:.4f}"
            logger.info(
                "Epoch %d | lr=%.2e | train: %s | val: %s%s",
                epoch,
                lr,
                _format_metrics(train_metrics),
                val_str,
                extra,
            )
            usage = train_metrics.get("vq_usage_rate", 0.0)
            perplexity = train_metrics.get("vq_perplexity", 0.0)
            if usage < 0.05 and epoch > 2:
                logger.warning(
                    "Low codebook usage rate %.3f (perplexity %.1f) — check VQ health",
                    usage,
                    perplexity,
                )

        val_loss = val_metrics.get("loss", float("inf"))
        if val_metrics and val_loss < best_val:
            best_val = val_loss
            best_path = out_dir / "bpc_v2_best.pt"
            ckpt_payload: dict = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "val_loss": val_loss,
            }
            if split is not None:
                ckpt_payload["split"] = {
                    "train_end": str(split.train_end.date()),
                    "val_start": str(split.val_start.date()),
                }
            torch.save(ckpt_payload, best_path)

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = out_dir / f"bpc_v2_epoch_{epoch:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                ckpt_path,
            )
            logger.info("Checkpoint: %s", ckpt_path)

    model.save_behavioral_ontology(str(out_dir / "behavioral_ontology_v1.pt"))
    torch.save({"model": model.state_dict(), "epoch": args.epochs - 1}, out_dir / "bpc_v2_last.pt")

    semantics = model.analyze_token_semantics(val_loader, device=device, max_samples=5000)
    if not semantics.empty:
        semantics.to_csv(out_dir / "token_semantics_val.csv")

    metrics_logger.close()
    logger.info("Training complete. Logs: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
