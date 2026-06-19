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
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from quant_cursor.bpc.dataset import load_qlib_instruments
from quant_cursor.config import load_config

from .config import GlobalConfig
from .dataset import (
    LoaderOptions,
    create_dataloaders,
    iter_training_batches,
    to_device,
    _loader_has_gpu_batches,
)
from .kronos import sync_kronos_config
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

    sync_kronos_config(config)

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
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--val-max-batches", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--gpu-cache-data", action="store_true")
    p.add_argument("--batched-gpu", action="store_true")
    p.add_argument("--batched-gpu-cpu", action="store_true")
    p.add_argument("--batched-gpu-resident", action="store_true")
    p.add_argument("--preprocessed-dir", type=str, default=None)
    p.add_argument("--save-preprocessed", type=str, default=None)
    p.add_argument("--force-rebuild-preprocessed", action="store_true")
    p.add_argument("--preprocess-only", action="store_true")
    p.add_argument("--kronos-path", type=str, default=None)
    return p.parse_args(injected), small_data


def _resolve_sample_cap(args: argparse.Namespace) -> int | None:
    if args.full:
        return None
    if args.max_samples_per_instrument is not None:
        return args.max_samples_per_instrument
    if args.max_instruments is not None:
        return 200
    return None


def main() -> int:
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
    config.qlib.provider_uri = qc_cfg.qlib_data_dir
    config.qlib.start_date = start
    config.qlib.end_date = end
    config.qlib.instruments = instruments
    config.qlib.max_samples = args.max_samples
    config.qlib.val_ratio = args.val_ratio
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

    logger.info(
        "BPC-v4: instruments=%d window=%s..%s qlib=%s output=%s",
        len(instruments), start, end, config.qlib.provider_uri, out_dir,
    )

    run_training(
        config,
        loader_opts,
        resume=args.resume,
        preprocessed_dir=Path(args.preprocessed_dir) if args.preprocessed_dir else None,
        save_preprocessed=Path(args.save_preprocessed) if args.save_preprocessed else None,
        force_rebuild_preprocessed=args.force_rebuild_preprocessed,
        max_samples_per_instrument=_resolve_sample_cap(args),
        val_every=args.val_every,
        save_every=args.save_every,
        val_max_batches=args.val_max_batches,
        preprocess_only=args.preprocess_only,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    raise SystemExit(main())
