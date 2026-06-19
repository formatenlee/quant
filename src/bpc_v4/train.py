"""bpc_v4 训练入口（参考更优实现，支持 amp、检查点、评估）"""

import argparse
import logging
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .config import GlobalConfig
from .dataset import create_dataloaders
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/bpc_v4")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = GlobalConfig()
    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.learning_rate = args.lr
    config.train.device = args.device
    config.train.save_dir = Path(args.save_dir)
    config.train.save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_loader, val_loader, _ = create_dataloaders(config)

    model = BPCV4Model(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=config.train.amp)

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from {args.resume} at epoch {start_epoch}")

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


if __name__ == "__main__":
    main()
