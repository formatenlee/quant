"""
bpc_v4 快速本地测试脚本（真实 qlib 模式）

支持两种模式：
1. 合成数据（use_synthetic=True）：快速验证流程
2. 真实 qlib 日线数据（use_qlib=True）：完整模拟训练

用户指定配置：
- instruments=200
- start=2020-01-01, end=2026-12-31
- seq_len=40

运行示例（真实 qlib，限制样本数用于快速测试）：
    python -m bpc_v4.quick_test --use_qlib --n_inst 200 --seq_len 40 \
        --start 2020-01-01 --end 2026-12-31 --epochs 5 --max_samples 2000
"""

import argparse
import warnings
from collections import Counter
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .config import BPCv4Config
from .dataset import BPCv4Dataset, collate_fn
from .model import BPCv4Model
from .kronos import KronosTokenizerPool
from .loss_plots import LossCurveTrackerV4


def compute_codebook_stats(logits: torch.Tensor) -> dict:
    """计算 codebook 使用情况，检测坍塌"""
    preds = logits.argmax(dim=-1).cpu().numpy().flatten()
    unique = len(np.unique(preds))
    counts = Counter(preds)
    # 熵（归一化到 [0,1]）
    total = len(preds)
    probs = np.array([c / total for c in counts.values()])
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    max_entropy = np.log(64)  # 64 classes
    norm_entropy = entropy / max_entropy
    return {
        "unique_classes": unique,
        "norm_entropy": float(norm_entropy),
        "most_common_top3": counts.most_common(3),
    }


def run_quick_test(
    cfg: BPCv4Config,
    epochs: int = 100,
    device=None,
    use_qlib: bool = False,
    max_samples: Optional[int] = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("bpc_v4 Quick Local Test (真实 qlib 模式)")
    print(f"Device: {device}")
    print(f"Instruments: {len(cfg.instruments)}")
    print(f"Date range: {cfg.start_date} ~ {cfg.end_date}")
    print(f"seq_len={cfg.seq_len}, epochs={epochs}")
    print("=" * 60)

    # 1. 初始化 Kronos（单例）
    KronosTokenizerPool.get_tokenizer(device)

    # 2. 数据集
    if use_qlib:
        from .qlib_loader import QlibDayDatasetV4
        full_ds = QlibDayDatasetV4(
            instruments=cfg.instruments,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            seq_len=cfg.seq_len,
            stride=5,
            use_kronos=True,
            amount_missing_log=True,
        )
    else:
        full_ds = BPCv4Dataset(
            instruments=cfg.instruments or ["000001.SZ", "000002.SZ", "510300.SH"],
            start_date=cfg.start_date or "2023-01-01",
            end_date=cfg.end_date or "2024-06-01",
            seq_len=cfg.seq_len,
            split="train",
            amount_missing_log=True,
            use_synthetic=True,
        )

    # 简单 train/val 切分（8:2）
    n = len(full_ds)
    if max_samples is not None and n > max_samples:
        # 随机子采样，加速测试
        indices = np.random.choice(n, max_samples, replace=False)
        full_ds = torch.utils.data.Subset(full_ds, indices)
        n = len(full_ds)
        print(f"[QuickTest] Subsampled to {max_samples} samples for faster testing")

    n_val = max(1, n // 5)
    n_train = n - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, collate_fn=collate_fn)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # 3. 模型
    model = BPCv4Model(
        d_model=cfg.d_model,
        bpc_dim=cfg.bpc_dim,
        ctx_dim=cfg.ctx_dim,
        emb_dim=cfg.emb_dim,
        fusion_dim=cfg.fusion_dim,
        n_codebook=cfg.n_codebook,
        purity_weight=cfg.purity_weight,
        codebook_weight=cfg.codebook_weight,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # 4. Loss 曲线跟踪器
    loss_tracker = LossCurveTrackerV4(Path("bpc_v4_outputs"))

    # 5. 训练循环
    history = []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_purity_losses = []
        train_codebook_losses = []
        codebook_stats_list = []

        for batch in train_loader:
            z_q = batch["z_q"].to(device)
            bpc = batch["bpc_feat"].to(device)
            ctx = batch["ctx_feat"].to(device)
            stock_ids = torch.zeros(z_q.size(0), dtype=torch.long, device=device)
            time_raw = torch.randn(z_q.size(0), 16, device=device)

            out = model(z_q, bpc, ctx, stock_ids, time_raw)

            proxies = batch["behavior_proxies"].to(device)
            purity_tgt = torch.softmax(proxies @ torch.randn(5, 15, device=device), dim=-1)
            code_tgt = batch["s1_ids"][:, 0].to(device).long()

            loss_dict = model.compute_loss(out, purity_tgt, code_tgt)

            optimizer.zero_grad()
            loss_dict["total"].backward()
            optimizer.step()

            train_losses.append(loss_dict["total"].item())
            train_purity_losses.append(loss_dict["purity"].item())
            train_codebook_losses.append(loss_dict["codebook"].item())

            cb_stats = compute_codebook_stats(out["codebook_logits"])
            codebook_stats_list.append(cb_stats)

        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                z_q = batch["z_q"].to(device)
                bpc = batch["bpc_feat"].to(device)
                ctx = batch["ctx_feat"].to(device)
                stock_ids = torch.zeros(z_q.size(0), dtype=torch.long, device=device)
                time_raw = torch.randn(z_q.size(0), 16, device=device)

                out = model(z_q, bpc, ctx, stock_ids, time_raw)
                proxies = batch["behavior_proxies"].to(device)
                purity_tgt = torch.softmax(proxies @ torch.randn(5, 15, device=device), dim=-1)
                code_tgt = batch["s1_ids"][:, 0].to(device).long()
                loss_dict = model.compute_loss(out, purity_tgt, code_tgt)
                val_losses.append(loss_dict["total"].item())

        # 聚合统计
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        train_purity = float(np.mean(train_purity_losses))
        train_codebook = float(np.mean(train_codebook_losses))
        avg_cb = {
            "unique": int(np.mean([s["unique_classes"] for s in codebook_stats_list])),
            "entropy": float(np.mean([s["norm_entropy"] for s in codebook_stats_list])),
        }

        # 更新 loss 曲线
        loss_tracker.update(
            epoch=epoch,
            train_total=train_loss,
            val_total=val_loss,
            train_purity=train_purity,
            val_purity=val_loss * 0.6,   # 简化
            train_codebook=train_codebook,
            val_codebook=val_loss * 0.4,
        )
        if epoch % 5 == 0 or epoch == epochs - 1:
            loss_tracker.render()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "codebook_unique": avg_cb["unique"],
            "codebook_entropy": avg_cb["entropy"],
        })

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
                f"Codebook unique={avg_cb['unique']}/64 entropy={avg_cb['entropy']:.3f}"
            )

    # 5. 最终诊断
    print("\n" + "=" * 60)
    print("训练完成 - 诊断报告")
    print("=" * 60)

    final = history[-1]
    print(f"最终 Train Loss: {final['train_loss']:.4f}")
    print(f"最终 Val   Loss: {final['val_loss']:.4f}")

    # 同步收敛判断
    if len(history) > 20:
        recent_train = [h["train_loss"] for h in history[-20:]]
        recent_val = [h["val_loss"] for h in history[-20:]]
        train_trend = np.polyfit(range(20), recent_train, 1)[0]
        val_trend = np.polyfit(range(20), recent_val, 1)[0]
        sync = "同步收敛" if abs(train_trend - val_trend) < 0.01 else "存在发散风险"
        print(f"最近20轮趋势: Train slope={train_trend:.5f}, Val slope={val_trend:.5f} → {sync}")

    # 码本坍塌判断
    if final["codebook_unique"] < 8 or final["codebook_entropy"] < 0.3:
        print("⚠️  警告：codebook 可能发生坍塌（使用类别过少或熵过低）")
    else:
        print(f"✅ codebook 使用正常（{final['codebook_unique']}/64 类，归一化熵 {final['codebook_entropy']:.3f}）")

    print("测试结束。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="bpc_v4 快速本地测试（支持真实 qlib）")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数（真实 qlib 建议先用 3-5）")
    parser.add_argument("--seq_len", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_qlib", action="store_true", help="使用真实 qlib 日线数据")
    parser.add_argument("--n_inst", type=int, default=200, help="仪器数量（真实 qlib 模式）")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2026-12-31")
    parser.add_argument("--max_samples", type=int, default=2000, help="最大样本数（限制用于快速测试）")
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    # 构造配置
    if args.use_qlib:
        # 真实 qlib 模式：需要用户提供 200 个 instruments
        # 这里用常见 ETF/指数作为示例，实际使用时应从 universe 加载
        instruments = [f"{i:06d}.SH" for i in range(1, min(args.n_inst, 200))] + \
                      ["510300.SH", "510500.SH", "159919.SZ"]  # 补充常见宽基
        instruments = instruments[:args.n_inst]
    else:
        instruments = ["000001.SZ", "000002.SZ", "510300.SH"]

    cfg = BPCv4Config(
        instruments=instruments,
        start_date=args.start,
        end_date=args.end,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        n_codebook=64,
    )
    run_quick_test(cfg, epochs=args.epochs, use_qlib=args.use_qlib, max_samples=args.max_samples)
