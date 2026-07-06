"""
BPC-v4 训练入口（与 v3 相同：`src/` 即 `quant_cursor` 包）。

示例（在 quant/ 项目根目录）:
  python -m quant_cursor.bpc_v4.train --dev --device cuda

  python -m quant_cursor.bpc_v4.train \\
    --max-instruments 200 \\
    --start 2019-01-01 \\
    --gpu-cache-data \\
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

try:
    from torch.amp import GradScaler, autocast

    def _autocast(enabled: bool):
        return autocast("cuda", enabled=enabled)

    def _grad_scaler(enabled: bool) -> GradScaler:
        return GradScaler("cuda", enabled=enabled)

except ImportError:
    from torch.cuda.amp import GradScaler, autocast

    def _autocast(enabled: bool):
        return autocast(enabled=enabled)

    def _grad_scaler(enabled: bool) -> GradScaler:
        return GradScaler(enabled=enabled)

from quant_cursor.bpc.dataset import load_qlib_instruments
from quant_cursor.bpc.metrics import MetricsLogger
from quant_cursor.config import load_config

from .config import GlobalConfig
from .cpu_parallel import raise_nofile_soft_limit
from .dataset import (
    LoaderOptions,
    create_dataloaders,
    iter_training_batches,
    materialize_datasets,
    resolve_training_datasets,
    to_device,
    _loader_has_gpu_batches,
)
from .diagnostics_v4 import audit_purity_targets, audit_s1_token_diversity, compute_zq_bpc_correlation
from .kronos import sync_kronos_config
from .kronos_cache import resolve_kronos_cache_dir
from .loss_plots import LossCurveTrackerV4
from .metrics_v4 import (
    accumulate_tensor_metrics,
    finalize_averaged_metrics,
    format_epoch_summary,
    format_metrics_lines,
)
from .monitoring_v4 import accumulate_monitoring, compute_step_monitoring, finalize_monitoring
from .model import BPCV4Model

logger = logging.getLogger("quant_cursor.bpc_v4.train")

_V4_VALUE_DEFAULTS: dict[str, str] = {
    "--val-ratio": "0.15",
    "--batch-size": "256",
    "--epochs": "50",
    "--lr": "1e-3",
    "--num-workers": "8",
    "--prefetch-factor": "4",
    "--val-every": "5",
    "--save-every": "10",
    "--weight-decay": "1e-4",
}


def _log_epoch_metrics(
    epoch: int,
    lr: float,
    train_metrics: dict,
    val_metrics: dict | None = None,
    *,
    extra: str = "",
    total_epochs: int | None = None,
) -> None:
    """v3 风格分组日志：摘要行 + 分项纯度 / 方向预测。"""
    ep_label = f"{epoch}/{total_epochs}" if total_epochs else str(epoch)
    suffix = f" | {extra}" if extra else ""
    logger.info("Epoch %s | lr=%.2e%s", ep_label, lr, suffix)
    summary = format_epoch_summary(train_metrics, val_metrics)
    if "loss=" in summary:
        logger.info(summary)
    for line in format_metrics_lines(train_metrics, indent="  [train] "):
        logger.info(line)
    if val_metrics:
        for line in format_metrics_lines(val_metrics, indent="  [val]   "):
            logger.info(line)


def _metrics_to_record(prefix: str, metrics: dict) -> dict[str, float]:
    """将 epoch 指标写入 metrics.jsonl / loss_plots 的 train_/val_ 列。"""
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if key == "weighted_purity":
            continue
        if isinstance(value, (int, float)):
            out[f"{prefix}_{key}"] = float(value)
    if f"{prefix}_loss_purity_total" not in out and f"{prefix}_purity_loss" in out:
        out[f"{prefix}_loss_purity_total"] = out[f"{prefix}_purity_loss"]
    return out


def _audit_training_batch(train_ds, model_cfg) -> None:
    """启动前检查数据与标签分布，便于发现平凡解/缺字段。"""
    if len(train_ds) == 0:
        return
    sample = train_ds[0]
    has_purity = "purity_target" in sample
    if not has_purity:
        logger.warning("训练集缺少 purity_target，纯度 loss 将退化为均匀占位标签")
    audit_s1_token_diversity(
        train_ds,
        vocab_size=model_cfg.head.codebook_output_dim,
        n_sample=min(5000, len(train_ds)),
        fail_if_degenerate=True,
    )


def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    *,
    max_grad_norm: float = 1.0,
    non_blocking: bool = False,
    skip_device_transfer: bool = False,
    show_progress: bool = False,
):
    model.train()
    total: dict[str, float] = {}
    monitor_total: dict[str, float] = {}
    steps = 0
    gpu_batches = skip_device_transfer or _loader_has_gpu_batches(loader)

    batch_iter = iter_training_batches(loader)
    if show_progress:
        batch_iter = tqdm(batch_iter, desc="Training", leave=False)

    for batch in batch_iter:
        if not gpu_batches:
            batch = to_device(batch, device, non_blocking=non_blocking)
        optimizer.zero_grad()

        with _autocast(scaler.is_enabled()):
            outputs = model(batch)
        with _autocast(False):
            losses = model.compute_loss(batch, outputs)
            loss = losses["loss"]

        if not torch.isfinite(loss):
            logger.error(
                "Non-finite loss at step %d: loss=%s purity=%s",
                steps + 1,
                loss.item(),
                losses["purity_loss"].item(),
            )
            raise RuntimeError(
                "训练 loss 非有限值 (NaN/Inf)。常见原因：输入含 NaN 或 AMP 溢出。"
                "请 --force-rebuild-preprocessed 重试。"
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        accumulate_tensor_metrics(total, losses)
        accumulate_monitoring(monitor_total, compute_step_monitoring(batch, outputs))
        total["grad_norm"] = total.get("grad_norm", 0.0) + float(grad_norm.detach().item())
        steps += 1

    if steps == 0:
        raise RuntimeError("训练集 0 batch，请检查 batch_size / max_samples / 数据范围")

    metrics = finalize_averaged_metrics(total, steps)
    metrics.update(finalize_monitoring(monitor_total, steps))
    metrics["weighted_purity"] = model.purity_weight * metrics.get("purity_loss", 0.0)
    return metrics


@torch.no_grad()
def eval_epoch(
    model,
    loader,
    device,
    *,
    non_blocking: bool = False,
    skip_device_transfer: bool = False,
    max_batches: int = 0,
    compute_interpretability: bool = False,
    show_progress: bool = False,
):
    model.eval()
    total: dict[str, float] = {}
    monitor_total: dict[str, float] = {}
    steps = 0
    zq_bpc_corr: float | None = None
    gpu_batches = skip_device_transfer or _loader_has_gpu_batches(loader)

    batch_iter = iter_training_batches(loader)
    if show_progress:
        batch_iter = tqdm(batch_iter, desc="Evaluating", leave=False)

    for batch in batch_iter:
        if max_batches > 0 and steps >= max_batches:
            break
        if not gpu_batches:
            batch = to_device(batch, device, non_blocking=non_blocking)
        outputs = model(batch)
        losses = model.compute_loss(batch, outputs)

        accumulate_tensor_metrics(total, losses)
        accumulate_monitoring(monitor_total, compute_step_monitoring(batch, outputs))
        steps += 1

        if compute_interpretability and zq_bpc_corr is None:
            stats = compute_zq_bpc_correlation(batch["z_q"], batch["bpc_feat"])
            zq_bpc_corr = stats["mean_abs_corr"]

    if steps == 0:
        return {
            "loss": float("inf"),
            "purity_loss": 0.0,
            "zq_bpc_corr_mean": None,
        }

    metrics = finalize_averaged_metrics(total, steps)
    metrics.update(finalize_monitoring(monitor_total, steps))
    metrics["weighted_purity"] = model.purity_weight * metrics.get("purity_loss", 0.0)
    if zq_bpc_corr is not None:
        metrics["zq_bpc_corr_mean"] = zq_bpc_corr
    return metrics


def run_training(
    config: GlobalConfig,
    loader_opts: LoaderOptions,
    *,
    resume: str | None = None,
    preprocessed_dir: Path | None = None,
    save_preprocessed: Path | None = None,
    max_samples_per_instrument: int | None = None,
    val_every: int = 1,
    log_every: int = 1,
    save_every: int = 10,
    val_max_batches: int = 0,
    preprocess_only: bool = False,
    kronos_cache_dir: Path | None = None,
    allow_live_kronos: bool = False,
    cpu_threads: int = 4,
    show_progress: bool = False,
    materialize_fork: bool = False,
    force_rebuild_preprocessed: bool = False,
) -> None:
    device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    logger.info("Using device: %s", device)

    sync_kronos_config(config)

    if preprocess_only:
        if save_preprocessed is None:
            raise RuntimeError(
                "预处理物化须指定 --save-preprocessed <dir>，供后续训练 --preprocessed-dir 加载。"
            )
        materialize_datasets(
            config,
            max_samples_per_instrument=max_samples_per_instrument,
            save_preprocessed=save_preprocessed,
            seed=loader_opts.seed,
            kronos_cache_dir=kronos_cache_dir,
            allow_live_kronos=allow_live_kronos,
            cpu_threads=cpu_threads,
            materialize_fork=materialize_fork,
        )
        logger.info("preprocess-only: 物化完成，已保存至 %s。训练请: --preprocessed-dir %s", save_preprocessed, save_preprocessed)
        return

    share_memory = (
        loader_opts.num_workers > 0
        and not loader_opts.gpu_cache_data
        and not loader_opts.batched_gpu
    )
    train_ds, val_ds, _test_ds = resolve_training_datasets(
        config,
        preprocessed_dir=preprocessed_dir,
        save_preprocessed=save_preprocessed,
        force_rebuild=force_rebuild_preprocessed,
        share_memory=share_memory,
        max_samples_per_instrument=max_samples_per_instrument,
        seed=loader_opts.seed,
        kronos_cache_dir=kronos_cache_dir,
        allow_live_kronos=allow_live_kronos,
        cpu_threads=cpu_threads,
        materialize_fork=materialize_fork,
    )
    train_loader, val_loader, train_ds, _val_ds = create_dataloaders(
        config,
        loader_opts,
        train_ds,
        val_ds,
    )

    batched_gpu_resident = loader_opts.batched_gpu and use_cuda and not loader_opts.batched_gpu_cpu
    if batched_gpu_resident:
        logger.info("Training mode: GPU-resident — iter_batches() only slices preloaded tensors")
    elif loader_opts.gpu_cache_data:
        logger.info("Training mode: GpuCached — single-sample index on GPU (consider --batched-gpu)")
    else:
        logger.warning(
            "Training mode: DataLoader H2D each batch. With ample RAM use --batched-gpu to "
            "preload all features to GPU before epoch 1."
        )

    audit_purity_targets(train_ds)
    _audit_training_batch(train_ds, config)

    metrics_logger = MetricsLogger(config.train.save_dir)
    loss_tracker = LossCurveTrackerV4(config.train.save_dir)

    pin_memory = use_cuda and not batched_gpu_resident and not loader_opts.gpu_cache_data
    non_blocking = pin_memory and not batched_gpu_resident

    model = BPCV4Model(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epochs, eta_min=1e-6)
    scaler = _grad_scaler(config.train.amp and use_cuda)

    start_epoch = 0
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        logger.info("Resumed from %s at epoch %d", resume, start_epoch)

    best_val_loss = float("inf")
    log_every = max(1, int(log_every))
    for epoch in range(start_epoch, config.train.epochs):
        should_log = (
            (epoch + 1) % log_every == 0
            or epoch == start_epoch
            or epoch + 1 == config.train.epochs
        )
        if should_log:
            logger.info("%s", "=" * 50)

        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device,
            max_grad_norm=config.train.max_grad_norm,
            non_blocking=non_blocking,
            skip_device_transfer=batched_gpu_resident or loader_opts.gpu_cache_data,
            show_progress=show_progress,
        )

        val_metrics = None
        if (epoch + 1) % val_every == 0 or epoch + 1 == config.train.epochs:
            val_metrics = eval_epoch(
                model, val_loader, device,
                non_blocking=non_blocking,
                skip_device_transfer=batched_gpu_resident or loader_opts.gpu_cache_data,
                max_batches=val_max_batches,
                compute_interpretability=((epoch + 1) % 5 == 0 or epoch + 1 == config.train.epochs),
                show_progress=show_progress,
            )
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(
                    {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config},
                    config.train.save_dir / "best_model.pt",
                )
                logger.info("Best model saved (val_loss=%.4f)", best_val_loss)

        extra = ""
        if val_metrics and val_metrics.get("zq_bpc_corr_mean") is not None:
            extra = f"zq_bpc_corr={val_metrics['zq_bpc_corr_mean']:.4f}"
        if should_log:
            _log_epoch_metrics(
                epoch + 1,
                scheduler.get_last_lr()[0],
                train_metrics,
                val_metrics,
                extra=extra,
                total_epochs=config.train.epochs,
            )

        record = {
            "phase": "epoch",
            "epoch": epoch + 1,
            "lr": scheduler.get_last_lr()[0],
        }
        record.update(_metrics_to_record("train", train_metrics))
        if val_metrics is not None:
            record.update(_metrics_to_record("val", val_metrics))
            if val_metrics.get("zq_bpc_corr_mean") is not None:
                record["zq_bpc_corr_mean"] = val_metrics["zq_bpc_corr_mean"]
        metrics_logger.log(record)
        loss_tracker.update_from_record(record)

        # 每 5 个 epoch 或最后一个 epoch 渲染一次
        if (epoch + 1) % 5 == 0 or epoch + 1 == config.train.epochs:
            loss_tracker.render()

        scheduler.step()

        if save_every > 0 and (epoch + 1) % save_every == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                config.train.save_dir / f"checkpoint_epoch_{epoch+1}.pt",
            )

    # 最终渲染
    loss_tracker.render()
    metrics_logger.close()
    logger.info("Training complete!")


def _flag_present(argv: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(f"{flag}=") for a in argv)


def _inject_v4_cli_defaults(argv: list[str]) -> tuple[list[str], bool]:
    out: list[str] = []
    small_data = False
    for arg in argv:
        if arg == "--small-data":
            small_data = True
            continue
        out.append(arg)
    for flag, value in _V4_VALUE_DEFAULTS.items():
        if not _flag_present(out, flag):
            out.extend([flag, value])
    if not _flag_present(out, "--output-dir"):
        out.extend(["--output-dir", "__BPC_V4_AUTO__"])
    return out, small_data


def _resolve_output_dir(argv: list[str]) -> list[str]:
    if "__BPC_V4_AUTO__" not in argv:
        return argv
    resolved: list[str] = []
    for arg in argv:
        if arg == "__BPC_V4_AUTO__":
            cfg = load_config()
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            resolved.append(str(cfg.data_dir / "checkpoints" / "bpc_v4" / run_name))
        else:
            resolved.append(arg)
    return resolved


def _apply_dev_preset(args: argparse.Namespace) -> None:
    if not args.dev:
        return
    args.start = args.start or "2019-01-01"
    args.end = args.end or datetime.now().strftime("%Y-%m-%d")
    if args.max_instruments is None:
        args.max_instruments = 200
    if args.max_samples_per_instrument is None and args.max_samples is None:
        args.max_samples_per_instrument = 200
    if args.epochs == int(_V4_VALUE_DEFAULTS["--epochs"]):
        args.epochs = 3
    if args.batch_size == int(_V4_VALUE_DEFAULTS["--batch-size"]):
        args.batch_size = 64
    logger.info(
        "Dev preset: start=%s end=%s max_instruments=%s max_spi=%s max_samples=%s epochs=%s batch_size=%s",
        args.start, args.end, args.max_instruments, args.max_samples_per_instrument,
        args.max_samples, args.epochs, args.batch_size,
    )


def _apply_small_data_preset(args: argparse.Namespace) -> None:
    args.batch_size = min(int(args.batch_size), 128)
    args.num_workers = min(int(args.num_workers), 4)
    logger.info("Small-data preset: batch_size=%d num_workers=%d", args.batch_size, args.num_workers)


def _save_run_config(out_dir: Path, args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = {k: getattr(args, k) for k in vars(args)}
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, bool]:
    injected, small_data = _inject_v4_cli_defaults(list(argv) if argv is not None else sys.argv[1:])
    injected = _resolve_output_dir(injected)

    p = argparse.ArgumentParser(description="BPC-v4 训练（qlib + Kronos）")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--dev", action="store_true")
    p.add_argument("--small-data", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--instruments", nargs="*", default=None)
    p.add_argument("--max-instruments", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-samples-per-instrument", type=int, default=None)
    p.add_argument("--min-rows", type=int, default=60)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="每 N epoch 向控制台打印一次 train/val 指标摘要（metrics.jsonl 仍每 epoch 记录）",
    )
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--val-max-batches", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--gpu-cache-data", action="store_true")
    p.add_argument("--batched-gpu", action="store_true")
    p.add_argument("--batched-gpu-cpu", action="store_true")
    p.add_argument("--batched-gpu-resident", action="store_true")
    p.add_argument(
        "--preprocessed-dir",
        type=str,
        default=None,
        help="可选：已物化 train/val 目录；省略则在训练启动前现场物化（需 --kronos-cache-dir）",
    )
    p.add_argument(
        "--save-preprocessed",
        type=str,
        default=None,
        help="预处理用：物化完成后保存到此目录（须与 --preprocess-only 同用）",
    )
    p.add_argument("--force-rebuild-preprocessed", action="store_true", help="忽略 --preprocessed-dir，强制重新物化")
    p.add_argument("--preprocess-only", action="store_true")
    p.add_argument("--kronos-path", type=str, default=None)
    p.add_argument("--kronos-cache-dir", type=str, default=None, help="Kronos 预计算缓存目录")
    p.add_argument("--allow-live-kronos", action="store_true", help="无缓存时在线编码（慢，调试用）")
    p.add_argument("--cpu-threads", type=int, default=4, help="物化/Kronos 预计算 CPU 线程数（默认 4）")
    p.add_argument(
        "--materialize-fork",
        action="store_true",
        help="Linux：物化用 fork 进程池绕过 GIL（需足够 fd；失败则改线程池）",
    )
    p.add_argument(
        "--show-progress",
        action="store_true",
        help="显示 tqdm 训练/验证进度条（默认关闭，减少控制台噪音）",
    )
    return p.parse_args(injected), small_data


def _resolve_sample_cap(args: argparse.Namespace) -> int | None:
    """每标的样本上限；默认 None = 按日期窗口内全部有效 bar。"""
    if args.full:
        return None
    return args.max_samples_per_instrument


def main() -> int:
    raise_nofile_soft_limit()
    logger.info("BPC-v4 entry: %s", Path(__file__).resolve())
    args, small_data = parse_args()
    _apply_dev_preset(args)
    if small_data:
        _apply_small_data_preset(args)

    if args.batched_gpu_resident and not args.batched_gpu:
        logger.warning("--batched-gpu-resident is deprecated; treated as --batched-gpu")
        args.batched_gpu = True
    if args.batched_gpu_cpu and not args.batched_gpu:
        args.batched_gpu = True

    use_cuda_requested = args.device == "cuda" and torch.cuda.is_available()
    if use_cuda_requested and not args.batched_gpu and not args.gpu_cache_data:
        args.batched_gpu = True
        logger.info(
            "CUDA: auto-enabled --batched-gpu — all samples upload to GPU once before training "
            "(zero H2D / zero Kronos-BPC recompute per step)"
        )

    qc_cfg = load_config(Path(args.config) if args.config else None)
    start = args.start or "2019-01-01"
    end = args.end or datetime.now().strftime("%Y-%m-%d")

    if args.instruments:
        instruments = args.instruments
        logger.info("Instrument source: explicit list (%d)", len(instruments))
    else:
        manifest = qc_cfg.meta_dir / "qlib_manifest.parquet"
        instruments = load_qlib_instruments(
            manifest,
            max_instruments=None if args.full else args.max_instruments,
            min_rows=args.min_rows,
        )
        logger.info(
            "Instrument source: qlib manifest (full=%s max_instruments=%s loaded=%d)",
            args.full, args.max_instruments, len(instruments),
        )

    if not instruments:
        raise RuntimeError("无可用标的，请检查 data/meta/qlib_manifest.parquet")

    out_dir = Path(args.output_dir) if args.output_dir else qc_cfg.data_dir / "checkpoints" / "bpc_v4" / "run_manual"
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_run_config(out_dir, args)

    config = GlobalConfig()
    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.learning_rate = args.lr
    config.train.weight_decay = args.weight_decay
    config.train.device = args.device
    config.train.save_dir = out_dir
    config.train.amp = args.amp
    config.train.log_every = args.log_every
    config.qlib.provider_uri = qc_cfg.qlib_data_dir
    config.qlib.start_date = start
    config.qlib.end_date = end
    config.qlib.instruments = instruments
    config.qlib.max_samples = args.max_samples
    config.qlib.val_ratio = args.val_ratio
    config.preprocess.cpu_threads = args.cpu_threads
    if args.kronos_path:
        config.kronos.local_path = args.kronos_path
    else:
        from .kronos import resolve_kronos_local_path

        resolved = resolve_kronos_local_path(None)
        if resolved:
            config.kronos.local_path = resolved
        else:
            logger.warning(
                "未找到本地 Kronos 模型，将尝试 HuggingFace；"
                "可设置 --kronos-path 或 export KRONOS_PATH=/path/to/Kronos-Tokenizer-base"
            )

    max_spi = _resolve_sample_cap(args)
    if not args.full and args.max_instruments is None:
        logger.warning(
            "未指定 --max-instruments 且非 --full：将从 manifest 加载全部标的。"
            "建议加上 --max-instruments 200"
        )

    num_workers = args.num_workers
    if args.gpu_cache_data or (args.batched_gpu and not args.batched_gpu_cpu):
        if num_workers > 0:
            logger.info("GPU-resident data mode: overriding num_workers %d → 0", num_workers)
        num_workers = 0

    loader_opts = LoaderOptions(
        num_workers=num_workers,
        prefetch_factor=args.prefetch_factor,
        gpu_cache_data=args.gpu_cache_data,
        batched_gpu=args.batched_gpu,
        batched_gpu_cpu=args.batched_gpu_cpu,
        seed=args.seed,
    )

    kronos_cache_dir = resolve_kronos_cache_dir(
        explicit=Path(args.kronos_cache_dir) if args.kronos_cache_dir else None,
        default_dir=qc_cfg.data_dir / "kronos_cache",
        config=config,
        allow_live_kronos=args.allow_live_kronos,
    )

    kc_log = str(kronos_cache_dir) if kronos_cache_dir else ("live" if args.allow_live_kronos else "MISSING")
    logger.info(
        "BPC-v4: instruments=%d window=%s..%s max_spi=%s qlib=%s output=%s kronos_cache=%s",
        len(instruments),
        start,
        end,
        max_spi if max_spi is not None else "all",
        config.qlib.provider_uri,
        out_dir,
        kc_log,
    )

    if args.preprocess_only:
        if not args.save_preprocessed:
            raise SystemExit("error: --preprocess-only 须配合 --save-preprocessed <dir>")

    run_training(
        config,
        loader_opts,
        resume=args.resume,
        preprocessed_dir=Path(args.preprocessed_dir) if args.preprocessed_dir else None,
        save_preprocessed=Path(args.save_preprocessed) if args.save_preprocessed else None,
        max_samples_per_instrument=max_spi,
        val_every=args.val_every,
        log_every=args.log_every,
        save_every=args.save_every,
        val_max_batches=args.val_max_batches,
        preprocess_only=args.preprocess_only,
        kronos_cache_dir=kronos_cache_dir,
        allow_live_kronos=args.allow_live_kronos,
        cpu_threads=args.cpu_threads,
        show_progress=args.show_progress,
        materialize_fork=args.materialize_fork,
        force_rebuild_preprocessed=args.force_rebuild_preprocessed,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    raise SystemExit(main())
