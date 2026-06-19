"""bpc_v4 训练入口（amp、检查点、GPU 驻留数据、预处理缓存）"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .config import GlobalConfig
from .dataset import (
    LoaderOptions,
    create_dataloaders,
    iter_training_batches,
    resolve_qlib_instruments,
    to_device,
    _loader_has_gpu_batches,
)
from .model import BPCV4Model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
):
    model.train()
    total_loss = total_purity = total_codebook = steps = 0
    gpu_batches = skip_device_transfer or _loader_has_gpu_batches(loader)

    for batch in tqdm(iter_training_batches(loader), desc="Training"):
        if not gpu_batches:
            batch = to_device(batch, device, non_blocking=non_blocking)
        optimizer.zero_grad()

        with autocast(enabled=scaler.is_enabled()):
            outputs = model(batch)
            losses = model.compute_loss(batch, outputs)
            loss = losses["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_purity += losses["purity_loss"].item()
        total_codebook += losses["codebook_loss"].item()
        steps += 1

    if steps == 0:
        raise RuntimeError("训练集 0 batch，请检查 batch_size / max_samples / 数据范围")
    return {"loss": total_loss / steps, "purity_loss": total_purity / steps, "codebook_loss": total_codebook / steps}


@torch.no_grad()
def eval_epoch(
    model,
    loader,
    device,
    *,
    non_blocking: bool = False,
    skip_device_transfer: bool = False,
    max_batches: int = 0,
):
    model.eval()
    total_loss = total_purity = total_codebook = total_acc = steps = 0
    gpu_batches = skip_device_transfer or _loader_has_gpu_batches(loader)

    for batch in tqdm(iter_training_batches(loader), desc="Evaluating"):
        if max_batches > 0 and steps >= max_batches:
            break
        if not gpu_batches:
            batch = to_device(batch, device, non_blocking=non_blocking)
        outputs = model(batch)
        losses = model.compute_loss(batch, outputs)

        total_loss += losses["loss"].item()
        total_purity += losses["purity_loss"].item()
        total_codebook += losses["codebook_loss"].item()

        pred = outputs["codebook_logits"].argmax(dim=-1)
        target = batch["s1_ids"][:, -1]
        total_acc += (pred == target).float().mean().item()
        steps += 1

    if steps == 0:
        return {"loss": float("inf"), "purity_loss": 0.0, "codebook_loss": 0.0, "codebook_acc": 0.0}
    return {
        "loss": total_loss / steps,
        "purity_loss": total_purity / steps,
        "codebook_loss": total_codebook / steps,
        "codebook_acc": total_acc / steps,
    }


def run_training(
    config: GlobalConfig,
    loader_opts: LoaderOptions,
    *,
    resume: str | None = None,
    preprocessed_dir: Path | None = None,
    save_preprocessed: Path | None = None,
    force_rebuild_preprocessed: bool = False,
    max_samples_per_instrument: int | None = None,
    val_every: int = 1,
    save_every: int = 10,
    val_max_batches: int = 0,
    preprocess_only: bool = False,
) -> None:
    device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    logger.info("Using device: %s", device)

    train_loader, val_loader, _ = create_dataloaders(
        config,
        loader_opts,
        preprocessed_dir=preprocessed_dir,
        save_preprocessed=save_preprocessed,
        force_rebuild_preprocessed=force_rebuild_preprocessed,
        max_samples_per_instrument=max_samples_per_instrument,
    )

    if preprocess_only:
        logger.info("preprocess-only: 数据物化完成，跳过训练")
        return

    batched_gpu_resident = loader_opts.batched_gpu and use_cuda and not loader_opts.batched_gpu_cpu
    pin_memory = use_cuda and not batched_gpu_resident and not loader_opts.gpu_cache_data
    non_blocking = pin_memory and not batched_gpu_resident

    model = BPCV4Model(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=config.train.amp and use_cuda)

    start_epoch = 0
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        logger.info("Resumed from %s at epoch %d", resume, start_epoch)

    best_val_loss = float("inf")
    for epoch in range(start_epoch, config.train.epochs):
        logger.info("\n%s\nEpoch %d/%d", "=" * 50, epoch + 1, config.train.epochs)

        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device,
            max_grad_norm=config.train.max_grad_norm,
            non_blocking=non_blocking,
            skip_device_transfer=batched_gpu_resident or loader_opts.gpu_cache_data,
        )
        logger.info(
            "Train: loss=%.4f, purity=%.4f, cb=%.4f",
            train_metrics["loss"], train_metrics["purity_loss"], train_metrics["codebook_loss"],
        )

        if (epoch + 1) % val_every == 0 or epoch + 1 == config.train.epochs:
            val_metrics = eval_epoch(
                model, val_loader, device,
                non_blocking=non_blocking,
                skip_device_transfer=batched_gpu_resident or loader_opts.gpu_cache_data,
                max_batches=val_max_batches,
            )
            logger.info(
                "Val: loss=%.4f, purity=%.4f, cb=%.4f, acc=%.4f",
                val_metrics["loss"], val_metrics["purity_loss"],
                val_metrics["codebook_loss"], val_metrics["codebook_acc"],
            )
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(
                    {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config},
                    config.train.save_dir / "best_model.pt",
                )
                logger.info("Best model saved (val_loss=%.4f)", best_val_loss)

        scheduler.step()

        if save_every > 0 and (epoch + 1) % save_every == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                config.train.save_dir / f"checkpoint_epoch_{epoch+1}.pt",
            )

    logger.info("Training complete!")


def build_config_from_args(args) -> GlobalConfig:
    config = GlobalConfig()
    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.learning_rate = args.lr
    config.train.device = args.device
    config.train.save_dir = Path(args.save_dir)
    config.train.save_dir.mkdir(parents=True, exist_ok=True)
    config.train.amp = args.amp
    config.qlib.provider_uri = Path(args.qlib_uri)
    config.qlib.start_date = args.start_date
    config.qlib.end_date = args.end_date
    config.qlib.max_samples = args.max_samples
    config.qlib.val_ratio = args.val_ratio
    if args.kronos_path:
        config.kronos.local_path = args.kronos_path

    if args.instruments:
        config.qlib.instruments = args.instruments
    else:
        config.qlib.instruments = resolve_qlib_instruments(
            market=args.market,
            n_instruments=args.n_instruments,
            start=args.start_date,
            end=args.end_date,
            provider_uri=config.qlib.provider_uri,
        )
    return config


def main():
    parser = argparse.ArgumentParser(description="bpc_v4 训练（qlib 日线 + Kronos）")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/bpc_v4")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--qlib_uri", type=str, default="~/.qlib/qlib_data/cn_data")
    parser.add_argument("--market", type=str, default="csi300")
    parser.add_argument("--n_instruments", type=int, default=200)
    parser.add_argument("--start_date", type=str, default="2019-01-01")
    parser.add_argument("--end_date", type=str, default="2026-06-19")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--kronos_path", type=str, default=None)
    parser.add_argument("--instruments", nargs="*", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--gpu-cache-data", action="store_true")
    parser.add_argument("--batched-gpu", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    config = build_config_from_args(args)
    loader_opts = LoaderOptions(
        num_workers=0 if args.gpu_cache_data or args.batched_gpu else args.num_workers,
        gpu_cache_data=args.gpu_cache_data,
        batched_gpu=args.batched_gpu,
    )
    run_training(config, loader_opts, resume=args.resume)


if __name__ == "__main__":
    # 统一走 quant_cursor 入口（与 v3 一致，支持 --max-instruments 等参数）
    try:
        from quant_cursor.bpc_v4.train import main as _qc_main

        raise SystemExit(_qc_main())
    except ImportError:
        logger.warning(
            "quant_cursor.bpc_v4 未找到，回退到 bpc_v4.train 旧入口；"
            "请确认 PYTHONPATH 包含 src/ 且已 git pull 最新代码"
        )
        main()
