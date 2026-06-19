"""bpc_v4 训练入口（参考更优实现，支持 amp、检查点、评估）"""

import argparse
import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .config import GlobalConfig
from .dataset import create_dataloaders, resolve_qlib_instruments
from .model import BPCV4Model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_epoch(model, loader, optimizer, scaler, device, max_grad_norm=1.0):
    model.train()
    total_loss = total_purity = total_codebook = steps = 0

    for batch in tqdm(loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
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

    return {"loss": total_loss / steps, "purity_loss": total_purity / steps, "codebook_loss": total_codebook / steps}


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = total_purity = total_codebook = total_acc = steps = 0

    for batch in tqdm(loader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        outputs = model(batch)
        losses = model.compute_loss(batch, outputs)

        total_loss += losses["loss"].item()
        total_purity += losses["purity_loss"].item()
        total_codebook += losses["codebook_loss"].item()

        pred = outputs["codebook_logits"].argmax(dim=-1)
        target = batch["s1_ids"][:, -1]
        total_acc += (pred == target).float().mean().item()
        steps += 1

    return {
        "loss": total_loss / steps,
        "purity_loss": total_purity / steps,
        "codebook_loss": total_codebook / steps,
        "codebook_acc": total_acc / steps,
    }


def run_training(config: GlobalConfig, resume: str | None = None) -> None:
    """执行完整训练流程（供 quant_cursor.bpc_v4.train 调用）。"""
    device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_loader, val_loader, _ = create_dataloaders(config)

    model = BPCV4Model(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=config.train.amp)

    start_epoch = 0
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from {resume} at epoch {start_epoch}")

    best_val_loss = float("inf")
    for epoch in range(start_epoch, config.train.epochs):
        logger.info(f"\n{'='*50}\nEpoch {epoch+1}/{config.train.epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device)
        logger.info(f"Train: loss={train_metrics['loss']:.4f}, purity={train_metrics['purity_loss']:.4f}, cb={train_metrics['codebook_loss']:.4f}")

        val_metrics = eval_epoch(model, val_loader, device)
        logger.info(f"Val: loss={val_metrics['loss']:.4f}, purity={val_metrics['purity_loss']:.4f}, cb={val_metrics['codebook_loss']:.4f}, acc={val_metrics['codebook_acc']:.4f}")

        scheduler.step()

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config},
                       config.train.save_dir / "best_model.pt")
            logger.info(f"Best model saved (val_loss={best_val_loss:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                       config.train.save_dir / f"checkpoint_epoch_{epoch+1}.pt")

    logger.info("Training complete!")


def build_config_from_args(args) -> GlobalConfig:
    config = GlobalConfig()
    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.learning_rate = args.lr
    config.train.device = args.device
    config.train.save_dir = Path(args.save_dir)
    config.train.save_dir.mkdir(parents=True, exist_ok=True)
    config.qlib.provider_uri = Path(args.qlib_uri)
    config.qlib.start_date = args.start_date
    config.qlib.end_date = args.end_date
    config.qlib.max_samples = args.max_samples
    if args.kronos_path:
        config.kronos.local_path = args.kronos_path

    if args.instruments:
        config.qlib.instruments = args.instruments
    else:
        instruments = resolve_qlib_instruments(
            market=args.market,
            n_instruments=args.n_instruments,
            start=args.start_date,
            end=args.end_date,
            provider_uri=config.qlib.provider_uri,
        )
        config.qlib.instruments = instruments
    return config


def main():
    parser = argparse.ArgumentParser(description="bpc_v4 训练（qlib 日线 + Kronos）")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/bpc_v4")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--qlib_uri", type=str, default="~/.qlib/qlib_data/cn_data", help="qlib 数据目录")
    parser.add_argument("--market", type=str, default="csi300", help="qlib 市场池，如 csi300/csi500")
    parser.add_argument("--n_instruments", type=int, default=200, help="从市场池截取的标的数量")
    parser.add_argument("--start_date", type=str, default="2019-01-01")
    parser.add_argument("--end_date", type=str, default="2026-06-19", help="结束日期，默认今天")
    parser.add_argument("--max_samples", type=int, default=None, help="小样本测试：限制 train+val 总窗口数")
    parser.add_argument("--kronos_path", type=str, default=None, help="Kronos 本地模型目录（可选）")
    parser.add_argument("--instruments", nargs="*", default=None, help="显式指定标的列表（跳过 market 池）")
    args = parser.parse_args()

    config = build_config_from_args(args)
    logger.info(
        f"Qlib: {len(config.qlib.instruments)} instruments, {args.start_date} ~ {args.end_date}, "
        f"uri={config.qlib.provider_uri}"
    )
    run_training(config, resume=args.resume)


if __name__ == "__main__":
    main()
