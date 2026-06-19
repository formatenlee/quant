"""BPC-v3 训练 loss 曲线：train/val 分项叠加绘图。"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 按面板顺序展示的分项（metrics 字典中的 key，不含 train_/val_ 前缀）
LOSS_PANEL_KEYS: tuple[str, ...] = (
    "loss",
    "loss_purity_total",
    "loss_recon",
    "loss_vq",
    "loss_purity_regime",
    "loss_purity_attack",
    "loss_purity_path",
    "loss_purity_vol_struct",
    "loss_purity_momentum",
    "loss_iso",
    "loss_diversity",
    "loss_z_reg",
)

PANEL_TITLES: dict[str, str] = {
    "loss": "Total Loss",
    "loss_purity_total": "Purity Total",
    "loss_recon": "Reconstruction",
    "loss_vq": "VQ",
    "loss_purity_regime": "Purity Regime",
    "loss_purity_attack": "Purity Attack",
    "loss_purity_path": "Purity Path",
    "loss_purity_vol_struct": "Purity Vol Struct",
    "loss_purity_momentum": "Purity Momentum",
    "loss_iso": "Iso",
    "loss_diversity": "Diversity",
    "loss_z_reg": "Z Reg",
}


class LossCurveTracker:
    """从 metrics 记录中累积 train/val loss；render() 写入 PNG（由调用方控制频率）。"""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.out_dir / "loss_curves.png"
        self._epochs: list[int] = []
        self._train: dict[str, list[float | None]] = {k: [] for k in LOSS_PANEL_KEYS}
        self._val: dict[str, list[float | None]] = {k: [] for k in LOSS_PANEL_KEYS}
        self._mpl_warned = False

    def update_from_record(self, record: dict) -> None:
        epoch = int(record["epoch"])
        self._epochs.append(epoch)

        for key in LOSS_PANEL_KEYS:
            train_col = f"train_{key}"
            val_col = f"val_{key}"
            tr = record.get(train_col)
            va = record.get(val_col)
            self._train[key].append(float(tr) if tr is not None and tr != "" else None)
            self._val[key].append(float(va) if va is not None and va != "" else None)

    def render(self) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            if not self._mpl_warned:
                logger.warning(
                    "matplotlib 未安装，跳过 loss 曲线图。请 pip install matplotlib"
                )
                self._mpl_warned = True
            return

        if not self._epochs:
            return

        keys = [k for k in LOSS_PANEL_KEYS if any(v is not None for v in self._train[k])]
        if not keys:
            return

        ncols = 3
        nrows = (len(keys) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
        fig.suptitle("BPC Training Loss (train vs val)", fontsize=14, y=0.995)

        xs = self._epochs
        for idx, key in enumerate(keys):
            ax = axes[idx // ncols][idx % ncols]
            tr_y = self._train[key]
            va_y = self._val[key]

            ax.plot(xs, tr_y, color="#1f77b4", linewidth=1.6, label="train", alpha=0.9)
            val_x = [e for e, v in zip(xs, va_y) if v is not None]
            val_pts = [v for v in va_y if v is not None]
            if val_pts:
                ax.plot(
                    val_x,
                    val_pts,
                    color="#ff7f0e",
                    linewidth=1.6,
                    linestyle="--",
                    marker="o",
                    markersize=3,
                    label="val",
                    alpha=0.95,
                )
            ax.set_title(PANEL_TITLES.get(key, key), fontsize=10)
            ax.set_xlabel("epoch")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", fontsize=8)

        for j in range(len(keys), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(self.output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("Loss curves saved: %s", self.output_path)
