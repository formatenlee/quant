"""bpc_v4 训练 loss 曲线与指标监控（参考 v3 多子图风格 + v4 特有指标）"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

V4_PANEL_KEYS: tuple[str, ...] = (
    "loss",
    "loss_purity_total",
    "loss_purity_regime",
    "loss_purity_attack",
    "loss_purity_path",
    "loss_purity_vol_struct",
    "loss_purity_momentum",
    "purity_entropy",
    "next_day_sign_acc",
    "purity_acc",
    "grad_norm",
    "zq_bpc_corr_mean",
)

PANEL_TITLES_V4: dict[str, str] = {
    "loss": "Total Loss (Purity KL)",
    "loss_purity_total": "Purity Total (KL)",
    "loss_purity_regime": "Purity Regime",
    "loss_purity_attack": "Purity Attack",
    "loss_purity_path": "Purity Path",
    "loss_purity_vol_struct": "Purity Vol Struct",
    "loss_purity_momentum": "Purity Momentum",
    "purity_entropy": "Purity Prediction Entropy",
    "next_day_sign_acc": "Next-Day Sign Acc (momentum)",
    "purity_acc": "Purity Agent Acc (macro)",
    "grad_norm": "Gradient Norm",
    "zq_bpc_corr_mean": "z_q vs BPC |corr| (val)",
}


class LossCurveTrackerV4:
    """v4 loss / 纯度分项 / 可解释性曲线跟踪器。"""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.out_dir / "loss_curves_v4.png"
        self.correlation_path = self.out_dir / "zq_bpc_correlation.png"

        self._epochs: list[int] = []
        self._train: dict[str, list[float | None]] = {k: [] for k in V4_PANEL_KEYS}
        self._val: dict[str, list[float | None]] = {k: [] for k in V4_PANEL_KEYS}
        self._mpl_warned = False

    def update_from_record(self, record: dict) -> None:
        epoch = int(record["epoch"])
        self._epochs.append(epoch)

        for key in V4_PANEL_KEYS:
            train_col = f"train_{key}"
            val_col = f"val_{key}"
            tr = record.get(train_col)
            va = record.get(val_col)
            if key == "loss_purity_total" and tr is None:
                tr = record.get("train_purity_loss")
            if key == "loss_purity_total" and va is None:
                va = record.get("val_purity_loss")
            if key == "zq_bpc_corr_mean":
                tr = None
                va = record.get("zq_bpc_corr_mean")
            self._train[key].append(float(tr) if tr is not None and tr != "" else None)
            self._val[key].append(float(va) if va is not None and va != "" else None)

    def render(self) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            if not self._mpl_warned:
                logger.warning("matplotlib 未安装，跳过 loss 曲线图。请 pip install matplotlib")
                self._mpl_warned = True
            return

        if not self._epochs:
            return

        self._render_loss_panels(plt)
        self._render_correlation(plt)

    def _render_loss_panels(self, plt) -> None:
        keys = [k for k in V4_PANEL_KEYS if k != "zq_bpc_corr_mean"]
        keys = [k for k in keys if any(v is not None for v in self._train[k]) or any(v is not None for v in self._val[k])]
        if not keys:
            return

        ncols = 3
        nrows = (len(keys) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
        fig.suptitle("BPC-v4 Training Metrics (train vs val)", fontsize=14, y=0.995)

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
            ax.set_title(PANEL_TITLES_V4.get(key, key), fontsize=10)
            ax.set_xlabel("epoch")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", fontsize=8)

        for j in range(len(keys), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(self.output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Loss curves saved: %s", self.output_path)

    def _render_correlation(self, plt) -> None:
        ys = self._val["zq_bpc_corr_mean"]
        if not ys or all(v is None for v in ys):
            return

        fig, ax = plt.subplots(figsize=(8, 4))
        xs = self._epochs
        valid = [(e, v) for e, v in zip(xs, ys) if v is not None]
        if not valid:
            plt.close(fig)
            return
        ex, ey = zip(*valid)
        ax.plot(ex, ey, color="#2ca02c", linewidth=1.8, marker="o", markersize=4, label="mean |corr|")
        ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title("z_q vs BPC Feature Correlation (Interpretability)", fontsize=12)
        ax.set_xlabel("epoch")
        ax.set_ylabel("mean absolute correlation")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(self.correlation_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.debug("z_q-BPC correlation plot saved: %s", self.correlation_path)
