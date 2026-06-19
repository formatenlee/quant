"""bpc_v4 训练 loss 曲线保存（简化适配版）"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class LossCurveTrackerV4:
    """
    v4 专用 loss 曲线跟踪器。
    
    记录：
    - total / purity / codebook
    - 每 N epoch 保存一次 PNG
    """
    
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.out_dir / "loss_curves_v4.png"
        
        self.epochs: list[int] = []
        self.train_total: list[float] = []
        self.val_total: list[float] = []
        self.train_purity: list[float] = []
        self.val_purity: list[float] = []
        self.train_codebook: list[float] = []
        self.val_codebook: list[float] = []
    
    def update(
        self,
        epoch: int,
        train_total: float,
        val_total: float,
        train_purity: float = 0.0,
        val_purity: float = 0.0,
        train_codebook: float = 0.0,
        val_codebook: float = 0.0,
    ):
        self.epochs.append(epoch)
        self.train_total.append(train_total)
        self.val_total.append(val_total)
        self.train_purity.append(train_purity)
        self.val_purity.append(val_purity)
        self.train_codebook.append(train_codebook)
        self.val_codebook.append(val_codebook)
    
    def render(self):
        """保存 loss 曲线图"""
        if not self.epochs:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Total Loss
        axes[0].plot(self.epochs, self.train_total, label="Train", marker="o", markersize=3)
        axes[0].plot(self.epochs, self.val_total, label="Val", marker="s", markersize=3)
        axes[0].set_title("Total Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Purity Loss
        axes[1].plot(self.epochs, self.train_purity, label="Train Purity", marker="o", markersize=3)
        axes[1].plot(self.epochs, self.val_purity, label="Val Purity", marker="s", markersize=3)
        axes[1].set_title("Purity KL Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Codebook Loss
        axes[2].plot(self.epochs, self.train_codebook, label="Train Codebook", marker="o", markersize=3)
        axes[2].plot(self.epochs, self.val_codebook, label="Val Codebook", marker="s", markersize=3)
        axes[2].set_title("Codebook CE Loss")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[LossPlot] Saved to {self.output_path}")
