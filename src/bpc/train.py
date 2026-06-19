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
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from quant_cursor.bpc.dataset import (
    BatchedGpuDataset,
    ContiguousBatchDataset,
    DEFAULT_TRAIN_INSTRUMENTS,
    GpuCachedDataset,
    MaterializedMultiScaleDataset,
    TemporalSplit,
    build_datasets,
    load_materialized_dataset,
    load_trading_calendar,
    pin_dataset_share_memory,
    load_qlib_instruments,
    save_materialized_dataset,
)
from quant_cursor.bpc.behavior_features import BEHAVIOR_AGENT_NAMES, CORE_AGENTS, EXTENDED_AGENTS
from quant_cursor.bpc.vq_backend import normalize_vq_mode, uses_magnitude_split
from quant_cursor.bpc.diagnostics import diagnose_codebook_shift, save_diagnosis_report
from quant_cursor.bpc.metrics import MetricsLogger
from quant_cursor.bpc.model import (
    BPCv2,
    adapt_codebook_on_loader,
    build_scale_registry,
    eval_epoch,
    precompute_purity_thresholds,
    precompute_z_scale_baselines,
    train_epoch,
)
from quant_cursor.bpc.seed import seed_worker, set_seed
from quant_cursor.config import load_config

logger = logging.getLogger("quant_cursor.bpc.train")

# bpc_v3 等下游可替换为自定义构建函数 (model, vq_mode, use_cosine_vq)
build_model_fn = None


def _resolve_instrument_source(
    args: argparse.Namespace,
) -> tuple[list[str] | None, str]:
    """Return (manifest query list or None, source label)."""
    if args.instruments:
        return args.instruments, "explicit"
    if args.full or args.max_instruments is not None:
        return None, "manifest"
    return DEFAULT_TRAIN_INSTRUMENTS, "default_preset"


def _resolve_sample_cap(args: argparse.Namespace) -> int | None:
    """Per-instrument window cap; None = use all valid anchors in range."""
    if args.full:
        return None
    if args.max_samples_per_instrument is not None:
        return args.max_samples_per_instrument
    if args.max_instruments is not None:
        return 200
    return None


def _apply_dev_preset(args: argparse.Namespace) -> None:
    if not args.dev:
        return
    args.start = "2019-01-01"
    args.end = "2022-12-31"
    if args.max_instruments is None:
        args.max_instruments = 300
    if args.max_samples_per_instrument is None:
        args.max_samples_per_instrument = 200
    logger.info(
        "Dev preset: start=%s end=%s max_instruments=%s max_samples_per_instrument=%s",
        args.start,
        args.end,
        args.max_instruments,
        args.max_samples_per_instrument,
    )


def _load_split_meta(dir_path: Path) -> dict | None:
    p = dir_path / "split_meta.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _train_loader_batch_config(
    train_samples: int,
    batch_size: int,
) -> tuple[int, bool, int]:
    """Resolve effective batch size, drop_last, and batches/epoch for training."""
    if train_samples <= 0:
        raise ValueError(
            f"train_samples={train_samples}，无法训练。"
            "请检查 --max-instruments、--max-samples-per-instrument 或日期窗口。"
        )
    if train_samples <= batch_size:
        effective = train_samples
        drop_last = False
    else:
        effective = batch_size
        drop_last = True
    batches = train_samples // effective if drop_last else (train_samples + effective - 1) // effective
    if batches <= 0:
        raise ValueError(
            f"train_samples={train_samples}, batch_size={batch_size} → 0 batches/epoch；"
            "请减小 --batch-size 或增加训练样本。"
        )
    return effective, drop_last, batches


def _instruments_fingerprint(instruments: list[str]) -> str:
    payload = "\n".join(sorted(instruments))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _split_meta_compatible(
    meta: dict,
    *,
    eff_start: str,
    eff_end: str,
    args: argparse.Namespace,
    instrument_list: list[str],
    cache_dir: Path | None = None,
) -> tuple[bool, str]:
    n_instruments = len(instrument_list)
    checks: list[tuple[bool, str]] = [
        (meta.get("data_start") == eff_start, f"data_start {meta.get('data_start')!r} != {eff_start!r}"),
        (meta.get("data_end") == eff_end, f"data_end {meta.get('data_end')!r} != {eff_end!r}"),
        (
            abs(float(meta.get("val_ratio", -1.0)) - float(args.val_ratio)) < 1e-6,
            f"val_ratio {meta.get('val_ratio')} != {args.val_ratio}",
        ),
        (int(meta.get("seed", -1)) == int(args.seed), f"seed {meta.get('seed')} != {args.seed}"),
        (
            int(meta.get("day_lookback", -1)) == int(args.day_lookback),
            f"day_lookback {meta.get('day_lookback')} != {args.day_lookback}",
        ),
        (
            int(meta.get("week_lookback", -1)) == int(args.week_lookback),
            f"week_lookback {meta.get('week_lookback')} != {args.week_lookback}",
        ),
    ]
    for ok, reason in checks:
        if not ok:
            return False, reason

    saved_spi = meta.get("max_samples_per_instrument")
    current_spi = _resolve_sample_cap(args)
    if saved_spi is not None or current_spi is not None:
        if saved_spi != current_spi:
            return False, (
                f"max_samples_per_instrument {saved_spi!r} != {current_spi!r} "
                "(use --full for unlimited per-symbol windows)"
            )

    saved_max_inst = (meta.get("instrument_filter") or {}).get("max_instruments")
    if saved_max_inst != args.max_instruments:
        return False, (
            f"max_instruments {saved_max_inst!r} != {args.max_instruments!r}"
        )

    saved_fp = meta.get("instruments_fingerprint")
    current_fp = _instruments_fingerprint(instrument_list)
    if saved_fp and saved_fp != current_fp:
        saved_n = int(meta.get("n_instruments", 0))
        hint = (
            f"instrument set changed (cached {saved_n}, current {n_instruments}); "
            "manifest/filter differs from prior --save-preprocessed run"
        )
        filt = meta.get("instrument_filter") or {}
        if filt:
            hint += f"; cached_filter={filt}"
        return False, hint

    saved_n = meta.get("n_instruments")
    if saved_n is not None and int(saved_n) != int(n_instruments):
        return False, (
            f"n_instruments {saved_n} != {n_instruments} "
            "(manifest updated or instrument filter changed; one rebuild will refresh cache)"
        )

    if cache_dir is not None:
        inst_path = cache_dir / "instruments.json"
        if inst_path.exists():
            saved_ids = json.loads(inst_path.read_text(encoding="utf-8"))
            if sorted(saved_ids) != sorted(instrument_list):
                return False, (
                    f"instruments.json mismatch (cached {len(saved_ids)}, current {n_instruments})"
                )
    return True, "ok"


def _resolve_effective_dates(
    args: argparse.Namespace,
    calendar,
) -> tuple[str, str]:
    import pandas as pd

    cal = calendar.sort_values()
    cal = cal[(cal >= pd.Timestamp(args.start)) & (cal <= pd.Timestamp(args.end))]
    if args.max_calendar_days is not None and len(cal) > args.max_calendar_days:
        cal = cal[-args.max_calendar_days :]
    if len(cal) < 20:
        raise ValueError(
            f"交易日历过短（{len(cal)} 天），请放宽 --start/--end 或减小 --max-calendar-days"
        )
    return str(cal[0].date()), str(cal[-1].date())


def _resolve_vq_mode(args: argparse.Namespace) -> str:
    if args.vq_mode is not None:
        return normalize_vq_mode(args.vq_mode)
    if args.no_cosine_vq or args.no_normalized_vq:
        return "l2"
    return "cosine"


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
    p.add_argument(
        "--start",
        type=str,
        default="1990-01-01",
        help="数据起始交易日（含），缩小范围可显著加快验证迭代",
    )
    p.add_argument(
        "--end",
        type=str,
        default="2026-12-31",
        help="数据结束交易日（含）",
    )
    p.add_argument(
        "--max-calendar-days",
        type=int,
        default=None,
        help="仅使用 end 之前最近 N 个交易日（在 start/end 裁剪后再截断）",
    )
    p.add_argument(
        "--dev",
        action="store_true",
        help="快速验证预设：2019-01-01~2022-12-31 + max-instruments=300 + max-samples-per-instrument=200",
    )
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
    p.add_argument(
        "--num-fine-per-coarse",
        type=int,
        default=16,
        help="分层 VQ 每个粗码本下的细码本数量（--use-fine-vq 时生效）",
    )
    p.add_argument(
        "--use-fine-vq",
        action="store_true",
        help="启用分层残差细码本 VQ（默认关闭；开启后训练更慢且 fine_mix 需额外调参）",
    )
    p.add_argument(
        "--vq-mode",
        choices=["cosine", "l2"],
        default=None,
        help="VQ 距离：cosine=球面余弦+幅度分离（默认），l2=欧氏 L2",
    )
    p.add_argument(
        "--no-cosine-vq",
        action="store_true",
        help="等价于 --vq-mode l2（保留向后兼容）",
    )
    p.add_argument(
        "--no-purity-magnitude",
        action="store_true",
        help="纯度头不拼接 z_scale（默认 cosine VQ 下将 log1p(scale) 与潜变量拼接）",
    )
    p.add_argument(
        "--no-purity-scale-relative",
        action="store_true",
        help="纯度头用绝对 log1p(z_scale)；默认 per-stock 中位数归一化为相对强度",
    )
    p.add_argument(
        "--purity-from-quantized",
        action="store_true",
        help="纯度头使用量化 token z_q（BPC-v3 默认开启；v2 默认 continuous）",
    )
    p.add_argument(
        "--no-purity-from-quantized",
        action="store_true",
        help="纯度头改回 VQ 前连续潜变量 z（仅调试；易绕过码本）",
    )
    p.add_argument(
        "--label-temperature",
        type=float,
        default=None,
        help="符号化纯度标签软温度（>0 平滑三档；BPC-v3 默认 0.12，0=硬 one-hot）",
    )
    p.add_argument(
        "--no-normalized-vq",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--no-val-threshold-ema",
        action="store_true",
        help="禁用验证期 per_stock 阈值 EMA（默认开启，仅更新 eval 副本，不影响训练）",
    )
    p.add_argument(
        "--val-threshold-ema-decay",
        type=float,
        default=0.95,
        help="验证期 per_stock 阈值 EMA 衰减（越大越平滑，默认 0.95）",
    )
    p.add_argument(
        "--threshold-decay-half-life",
        type=float,
        default=252.0,
        help="per_stock 纯度阈值时间衰变半衰期（交易日数；验证期逐步回退到全局阈值）",
    )
    p.add_argument(
        "--no-codebook-film",
        action="store_true",
        help="禁用 SymbolTimeFiLM 时变码本调制（默认开启，缓解 val 分布偏移）",
    )
    p.add_argument(
        "--purity-weight",
        type=float,
        default=0.5,
        help="核心行为代理(vol/attack/path_structure/vol_structure)纯度权重",
    )
    p.add_argument(
        "--extended-purity-weight",
        type=float,
        default=0.15,
        help="扩展结构代理(momentum)纯度权重",
    )
    p.add_argument("--recon-weight", type=float, default=1.0)
    p.add_argument(
        "--commitment-cost",
        type=float,
        default=1.0,
        help="VQ commitment 系数（默认 1.0，抑制 z 远离码本）",
    )
    p.add_argument(
        "--adaptive-balance",
        action="store_true",
        help="启用 VQ/纯度自适应权重（默认关闭，使用固定 recon+vq+purity）",
    )
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
        default=0.0,
        help="训练期 EMA 使用率低于此比例的码本将被复活；默认 0=关闭",
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
    p.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader 预取 batch 数（仅 num_workers>0 时生效）")
    p.add_argument("--no-materialize", action="store_true", help="禁用样本预物化（省内存但更慢）")
    p.add_argument(
        "--no-precompute-proxies",
        action="store_true",
        help="物化时不预计算 behavior_proxies（每 batch 现算，较慢）",
    )
    p.add_argument("--val-every", type=int, default=5, help="每 N epoch 做一次全量验证")
    p.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="每 N epoch 向控制台/train.log 打印一次 train 指标摘要（metrics 文件仍每 epoch 记录）",
    )
    p.add_argument(
        "--val-max-batches",
        type=int,
        default=0,
        help="验证最多 batch 数；0=全量验证集（默认）",
    )
    p.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="每 epoch 训练最多 batch 数；0=全量（快速调试可设 50~200）",
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
    p.add_argument("--save-every", type=int, default=50, help="每 N epoch 存 checkpoint")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="训练产物目录：checkpoint (*.pt)、train.log、metrics.jsonl/csv、run_config.json",
    )
    p.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="同 --output-dir（更直观）；若两者都指定，--run-dir 优先",
    )
    p.add_argument("--device", type=str, default=None, help="cuda / cuda:0 / cpu；默认自动检测")
    p.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复")
    p.add_argument("--no-console-log", action="store_true", help="不在终端打印，仅写 train.log")
    p.add_argument("--min-rows", type=int, default=60)
    p.add_argument(
        "--prefer-minute",
        action="store_true",
        help="优先选取有分钟数据的标的（排序靠前，配合 --max-instruments 生效）",
    )
    p.add_argument(
        "--require-minute",
        action="store_true",
        help="仅使用有分钟数据的标的训练",
    )
    p.add_argument(
        "--minute-sample-boost",
        type=float,
        default=2.0,
        help="有分钟数据的标的每标的采样上限倍率（默认 2.0；1.0=不提升）",
    )
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
        help="物化后保存到目录；若 split_meta 与当前 --start/--end 等一致则自动复用，跳过 qlib 重建。",
    )
    p.add_argument(
        "--force-rebuild-preprocessed",
        action="store_true",
        help="忽略已有 --save-preprocessed 缓存，强制从 qlib 重新物化",
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
        "--ram-resident",
        action="store_true",
        help="物化数据全量载入 contiguous RAM；有 day_features 时丢弃冗余 raw OHLCV；建议 num_workers=0",
    )
    p.add_argument(
        "--no-ram-resident",
        action="store_true",
        help="禁用全内存驻留（BPC-v3 默认开启 --ram-resident）",
    )
    p.add_argument(
        "--gpu-cache-data",
        action="store_true",
        help="将预处理数据一次性加载到 GPU（消除 H2D 传输，需配合 num_workers=0）",
    )
    p.add_argument(
        "--batched-gpu",
        action="store_true",
        help="实验性：GPU 驻留 batch 切片（默认路径为普通 DataLoader，一般更稳）。",
    )
    p.add_argument(
        "--batched-gpu-cpu",
        action="store_true",
        help="特征留 CPU，batch 级 DataLoader + worker 预取（有 H2D 开销，显存紧张时用）。",
    )
    p.add_argument(
        "--batched-gpu-resident",
        action="store_true",
        help="已弃用：与 --batched-gpu 相同。",
    )
    p.add_argument(
        "--profile-batches",
        type=int,
        default=0,
        help="训练前 N 个 batch 打印 data/compute 耗时分解（0=关闭）",
    )
    # 新增：预计算特征物化选项
    p.add_argument(
        "--precompute-features",
        action="store_true",
        default=True,
        help="物化时预计算 day 特征向量（默认开启，维数见 feature_dims.DAY_FULL_FEAT_DIM）",
    )
    p.add_argument(
        "--no-precompute-features",
        action="store_true",
        dest="precompute_features",
        help="禁用预计算特征物化（使用原始 OHLCV，训练时实时计算特征）",
    )
    return p.parse_args()


def _checkpoint_extras(model: BPCv2) -> dict:
    norm = model.normalizer_state_dict()
    return {"normalizer_state": norm} if norm else {}


def _first_collate(batch: list) -> dict:
    """DataLoader collate when batch_size=1 and each item is already a full batch."""
    return batch[0]


_HEALTH_KEYS = frozenset(
    {
        "z_norm_mean",
        "z_scale_mean",
        "codebook_norm_mean",
        "recon_cosine",
        "purity_entropy",
        "vq_dir_residual_mean",
        "z_scale_rel_mean",
        "grad_norm",
    }
)
_VQ_BATCH_COUNT_KEYS = frozenset({"vq_unique_tokens", "vq_fine_unique_tokens"})
_VQ_EPOCH_COUNT_KEYS = frozenset({"vq_epoch_unique_tokens", "vq_ema_active_codes"})
_VAL_OMIT_KEYS = frozenset({"loss_diversity", "grad_norm"})


def _format_metric_pair(key: str, value: float, *, phase: str, codebook_film: bool) -> str | None:
    if phase == "val" and key in _VAL_OMIT_KEYS:
        return None
    if phase == "val" and key == "loss_diversity" and abs(value) < 1e-12:
        return None
    if key == "loss" or (key.startswith("loss_") and key != "loss_raw_weighted"):
        return f"{key}={value:.6f}"
    if key.startswith("vq_"):
        if key in _VQ_EPOCH_COUNT_KEYS:
            return f"{key}={round(value)} (epoch)"
        if key in _VQ_BATCH_COUNT_KEYS:
            return f"{key}={round(value)} (batch-avg)"
        if key in ("vq_usage_rate", "vq_epoch_usage_rate", "vq_ema_usage_rate"):
            return f"{key}={value * 100:.1f}%"
        return f"{key}={value:.4f}"
    if key in _HEALTH_KEYS:
        return f"{key}={value:.4f}"
    if key.startswith("film_"):
        if key.startswith("film_codebook") and not codebook_film:
            return None
        return f"{key}={value:.4f}"
    if key.startswith("balance_") or key == "loss_raw_weighted":
        return f"{key}={value:.4f}"
    return None


def _metric_groups(
    metrics: dict[str, float],
    *,
    phase: str = "train",
    codebook_film: bool = False,
) -> dict[str, list[str]]:
    if not metrics:
        return {}
    grouped: dict[str, list[str]] = {
        "Loss": [],
        "VQ": [],
        "Health": [],
        "Latent": [],
        "Film": [],
        "Balance": [],
    }
    for k, v in sorted(metrics.items()):
        pair = _format_metric_pair(k, v, phase=phase, codebook_film=codebook_film)
        if pair is None:
            continue
        if k == "loss" or (k.startswith("loss_") and k != "loss_raw_weighted"):
            grouped["Loss"].append(pair)
        elif k.startswith("vq_"):
            grouped["VQ"].append(pair)
        elif k in _HEALTH_KEYS:
            grouped["Health"].append(pair)
        elif k.startswith("film_"):
            bucket = "Film" if k.startswith("film_codebook") else "Latent"
            grouped[bucket].append(pair)
        elif k.startswith("balance_") or k == "loss_raw_weighted":
            grouped["Balance"].append(pair)
    return {title: parts for title, parts in grouped.items() if parts}


def _format_metrics_lines(
    metrics: dict[str, float],
    *,
    indent: str,
    phase: str = "train",
    codebook_film: bool = False,
) -> list[str]:
    lines: list[str] = []
    for title, parts in _metric_groups(
        metrics, phase=phase, codebook_film=codebook_film
    ).items():
        lines.append(f"{indent}{title}: {', '.join(parts)}")
    return lines or [f"{indent}n/a"]


def _log_epoch_metrics(
    epoch: int,
    lr: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    *,
    extra: str = "",
    codebook_film: bool = False,
) -> None:
    logger.info("Epoch %d | lr=%.2e%s", epoch, lr, extra)
    for line in _format_metrics_lines(
        train_metrics, indent="  [train] ", phase="train", codebook_film=codebook_film
    ):
        logger.info(line)

    if val_metrics:
        for line in _format_metrics_lines(
            val_metrics, indent="  [val]   ", phase="val", codebook_film=codebook_film
        ):
            logger.info(line)


def _ensure_run_dir(log_dir: Path) -> Path:
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
    train_samples: int | None = None,
    val_samples: int | None = None,
    requested_start: str | None = None,
    requested_end: str | None = None,
    instrument_filter: dict | None = None,
) -> None:
    _ensure_run_dir(log_dir)
    meta = {
        "data_start": str(split.data_start.date()),
        "data_end": str(split.data_end.date()),
        "train_end": str(split.train_end.date()),
        "val_start": str(split.val_start.date()),
        "val_end": str(split.val_end.date()),
        "requested_start": requested_start,
        "requested_end": requested_end,
        "val_ratio": val_ratio,
        "train_ratio": round(1.0 - val_ratio, 4),
        "seed": seed,
        "day_lookback": day_lookback,
        "week_lookback": week_lookback,
        "n_instruments": len(instruments),
        "max_samples_per_instrument": instrument_filter.get("max_samples_per_instrument")
        if instrument_filter
        else None,
        "instruments_fingerprint": _instruments_fingerprint(instruments),
        "instrument_filter": instrument_filter,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "leakage_policy": (
            "train anchors <= train_end with full daily window <= train_end; "
            "val anchors >= val_start; normalizer fit on train only"
        ),
    }
    (log_dir / "split_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (log_dir / "instruments.json").write_text(
        json.dumps(instruments, indent=2, ensure_ascii=False), encoding="utf-8"
    )


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
) -> tuple[MaterializedMultiScaleDataset | object, MaterializedMultiScaleDataset | object, TemporalSplit | None, object | None]:
    """Build datasets from qlib, or load pre-materialized tensors from disk."""
    import pandas as pd

    calendar = load_trading_calendar(qlib_uri)
    eff_start, eff_end = _resolve_effective_dates(args, calendar)
    n_eff_days = len(
        calendar[
            (calendar >= pd.Timestamp(eff_start)) & (calendar <= pd.Timestamp(eff_end))
        ]
    )
    n_instruments = len(instrument_list)
    max_spi = _resolve_sample_cap(args)

    logger.info(
        "Calendar window: %s .. %s (%d trading days) | instruments=%d | full=%s | max_samples_per_instrument=%s",
        eff_start,
        eff_end,
        n_eff_days,
        n_instruments,
        args.full,
        "all" if max_spi is None else max_spi,
    )
    if max_spi is None and n_eff_days > 500:
        est_train_days = max(1, int(n_eff_days * (1.0 - args.val_ratio)) - args.day_lookback)
        est_train_samples = n_instruments * est_train_days
        est_batches = est_train_samples // max(1, args.batch_size)
        logger.info(
            "Estimated train_samples≈%d, batches/epoch≈%d (--full 不截断每标的采样；缩小日期可线性减少 batch 数)",
            est_train_samples,
            est_batches,
        )

    load_dir: Path | None = None
    if args.preprocessed_dir:
        load_dir = Path(args.preprocessed_dir)
    elif (
        args.save_preprocessed
        and not args.force_rebuild_preprocessed
        and (Path(args.save_preprocessed) / "train").exists()
        and (Path(args.save_preprocessed) / "val").exists()
    ):
        cache_dir = Path(args.save_preprocessed)
        meta = _load_split_meta(cache_dir)
        if meta is not None:
            ok, reason = _split_meta_compatible(
                meta,
                eff_start=eff_start,
                eff_end=eff_end,
                args=args,
                instrument_list=instrument_list,
                cache_dir=cache_dir,
            )
            if ok:
                logger.info(
                    "Reusing preprocessed cache at %s (split_meta match; skip qlib + materialize)",
                    cache_dir,
                )
                load_dir = cache_dir
            else:
                logger.warning(
                    "Preprocessed cache at %s incompatible (%s); rebuilding from qlib. "
                    "本次物化完成后会自动覆盖缓存，下次同参数启动将直接复用。",
                    cache_dir,
                    reason,
                )
        else:
            logger.warning(
                "Preprocessed cache at %s missing split_meta.json; rebuilding from qlib",
                cache_dir,
            )

    if load_dir is not None:
        logger.info("Loading pre-materialized dataset from %s", load_dir)
        meta = _load_split_meta(load_dir)
        if meta is not None:
            ok, reason = _split_meta_compatible(
                meta,
                eff_start=eff_start,
                eff_end=eff_end,
                args=args,
                instrument_list=instrument_list,
                cache_dir=load_dir,
            )
            if not ok:
                logger.error(
                    "加载的预处理数据窗口为 %s..%s（train≈%s, val≈%s），"
                    "与请求的 %s..%s 不一致（%s）。"
                    "--start/--end 不会改变 epoch 耗时；请用相同参数重新 --save-preprocessed "
                    "或加 --force-rebuild-preprocessed",
                    meta.get("data_start"),
                    meta.get("data_end"),
                    meta.get("train_samples"),
                    meta.get("val_samples"),
                    eff_start,
                    eff_end,
                    reason,
                )
            else:
                logger.info(
                    "Preprocessed split_meta OK | train_samples=%s | val_samples=%s | train_end=%s | val_start=%s",
                    meta.get("train_samples"),
                    meta.get("val_samples"),
                    meta.get("train_end"),
                    meta.get("val_start"),
                )
        logger.warning(
            "If preprocessed data predates 5-agent schema, volume_level anomaly, or volume_rel_cv, "
            "re-run --save-preprocessed"
        )
        train_ds = load_materialized_dataset(load_dir / "train")
        val_ds = load_materialized_dataset(load_dir / "val")
        return train_ds, val_ds, None, None

    store, train_ds, val_ds, split = build_datasets(
        instruments=instrument_list,
        start=eff_start,
        end=eff_end,
        provider_uri=qlib_uri,
        registry=registry,
        val_ratio=args.val_ratio,
        max_samples_per_instrument=max_spi,
        minute_sample_boost=args.minute_sample_boost if args.prefer_minute else 1.0,
        materialize=not args.no_materialize,
        share_memory=args.num_workers > 0,
        precompute_features=args.precompute_features,
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
                train_samples=len(train_ds),
                val_samples=len(val_ds),
                requested_start=args.start,
                requested_end=args.end,
                instrument_filter={
                    "full": bool(args.full),
                    "max_instruments": args.max_instruments,
                    "max_samples_per_instrument": _resolve_sample_cap(args),
                    "min_rows": args.min_rows,
                    "asset_types": args.asset_types,
                    "source": _resolve_instrument_source(args)[1],
                },
            )
        logger.info(
            "Preprocessed data saved (%d train / %d val). Next run: drop --force-rebuild-preprocessed to reuse %s",
            len(train_ds),
            len(val_ds),
            save_dir,
        )
    return train_ds, val_ds, split, store


def main() -> int:
    args = parse_args()
    _apply_dev_preset(args)
    set_seed(args.seed, deterministic=not args.non_deterministic)

    config = load_config(Path(args.config) if args.config else None)
    qlib_uri = config.data_dir / "qlib_data"
    manifest = config.meta_dir / "qlib_manifest.parquet"

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if args.run_dir:
        args.output_dir = args.run_dir
    out_dir = Path(args.output_dir) if args.output_dir else config.data_dir / "checkpoints" / "bpc" / run_name
    setup_logging(out_dir, console=not args.no_console_log)
    logger.info("Run directory: %s (checkpoints, train.log, metrics)", out_dir.resolve())
    _save_run_config(out_dir, args)
    metrics_logger = MetricsLogger(out_dir)

    registry = build_scale_registry(
        args.day_lookback, args.week_lookback, precomputed=args.precompute_features
    )

    instrument_query, instrument_source = _resolve_instrument_source(args)
    instrument_list = load_qlib_instruments(
        manifest,
        instruments=instrument_query,
        asset_types=args.asset_types,
        max_instruments=args.max_instruments,
        min_rows=args.min_rows,
        prefer_minute=args.prefer_minute,
        require_minute=args.require_minute,
    )
    if instrument_source == "default_preset":
        logger.info(
            "Instrument source: DEFAULT_TRAIN_INSTRUMENTS (%d presets). "
            "Use --full or --max-instruments N to train on qlib manifest.",
            len(DEFAULT_TRAIN_INSTRUMENTS),
        )
    elif instrument_source == "manifest":
        logger.info(
            "Instrument source: qlib manifest (full=%s, max_instruments=%s, min_rows=%d)",
            args.full,
            args.max_instruments,
            args.min_rows,
        )
    if args.max_instruments and len(instrument_list) < args.max_instruments:
        logger.warning(
            "manifest 在 min_rows=%d 等过滤后仅 %d 只标的（请求 max_instruments=%d）",
            args.min_rows,
            len(instrument_list),
            args.max_instruments,
        )

    n_min = 0
    try:
        from quant_cursor.freq_registry import load_freq_coverage

        cov = load_freq_coverage(config)
        minute_ids = set(cov.loc[cov["has_1min"].fillna(False), "qlib_id"].astype(str))
        n_min = sum(1 for i in instrument_list if i in minute_ids)
    except Exception:  # noqa: BLE001
        pass

    logger.info(
        "Run dir: %s | instruments=%d (minute=%d) | source=%s | full=%s | prefer_minute=%s | minute_boost=%.1f",
        out_dir,
        len(instrument_list),
        n_min,
        instrument_source,
        args.full,
        args.prefer_minute,
        args.minute_sample_boost,
    )
    logger.info(
        "epochs=%d | val_ratio=%.2f | seed=%d | day_lb=%d | week_lb=%d | precompute_features=%s",
        args.epochs,
        args.val_ratio,
        args.seed,
        args.day_lookback,
        args.week_lookback,
        args.precompute_features,
    )

    train_ds, val_ds, split, store = _resolve_datasets(
        args,
        instrument_list=instrument_list,
        qlib_uri=qlib_uri,
        registry=registry,
    )

    if getattr(args, "ram_resident", False) and not getattr(args, "no_ram_resident", False):
        if isinstance(train_ds, MaterializedMultiScaleDataset):
            try:
                from quant_cursor.bpc_v3 import dataset as _bpc_v3_dataset

                if isinstance(train_ds, _bpc_v3_dataset.MaterializedMultiScaleDatasetV3):
                    _bpc_v3_dataset.assert_v3_training_cache_complete(train_ds)
                    train_ds = _bpc_v3_dataset.promote_dataset_ram_resident(train_ds)
                if isinstance(val_ds, _bpc_v3_dataset.MaterializedMultiScaleDatasetV3):
                    val_ds = _bpc_v3_dataset.promote_dataset_ram_resident(val_ds)
            except ImportError:
                logger.warning("ram-resident: bpc_v3.dataset unavailable, skip promote")
        if args.num_workers > 0:
            logger.info("--ram-resident: forcing num_workers=0 (dataset already in RAM)")
            args.num_workers = 0

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

    if use_cuda:
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    import platform

    mp_context = "fork" if platform.system() == "Linux" and args.num_workers > 0 else None

    if args.batched_gpu and not use_cuda:
        logger.warning("--batched-gpu requires CUDA; falling back to CPU DataLoader")
        args.batched_gpu = False
    if args.batched_gpu_resident and not args.batched_gpu:
        logger.warning("--batched-gpu-resident is deprecated; treated as --batched-gpu")
        args.batched_gpu = True
    if args.batched_gpu_cpu and not args.batched_gpu:
        args.batched_gpu = True

    train_sample_count = len(train_ds)
    val_sample_count = len(val_ds)
    train_batch_size, train_drop_last, train_batches_per_epoch = _train_loader_batch_config(
        train_sample_count, args.batch_size
    )
    if train_batch_size != args.batch_size or not train_drop_last:
        logger.warning(
            "train_samples=%d 小于或接近 batch_size=%d → effective_batch=%d, drop_last=%s, "
            "batches/epoch=%d（否则 DataLoader 在 drop_last=True 下会产出 0 batch，训练空转）",
            train_sample_count,
            args.batch_size,
            train_batch_size,
            train_drop_last,
            train_batches_per_epoch,
        )

    if (
        not args.batched_gpu
        and args.num_workers > 0
        and isinstance(train_ds, MaterializedMultiScaleDataset)
        and not getattr(args, "ram_resident", False)
    ):
        pin_dataset_share_memory(train_ds)
        if isinstance(val_ds, MaterializedMultiScaleDataset):
            pin_dataset_share_memory(val_ds)
        logger.info("share_memory enabled for DataLoader (num_workers=%d)", args.num_workers)

    batched_gpu_resident = args.batched_gpu and use_cuda and not args.batched_gpu_cpu

    if batched_gpu_resident:
        if args.num_workers > 0:
            logger.info(
                "--batched-gpu uses GPU-resident slices; ignoring num_workers=%d",
                args.num_workers,
            )
        logger.info(
            "BatchedGpuDataset: features on GPU, batch slice (batch_size=%d, num_workers=0).",
            train_batch_size,
        )
        if not isinstance(train_ds, MaterializedMultiScaleDataset):
            logger.error("BatchedGpuDataset requires MaterializedMultiScaleDataset as input")
            return 1
        train_ds = BatchedGpuDataset(
            train_ds,
            device=device,
            batch_size=train_batch_size,
            drop_last=train_drop_last,
            shuffle=True,
        )
        val_ds = BatchedGpuDataset(
            val_ds,
            device=device,
            batch_size=min(args.batch_size, max(1, val_sample_count)),
            drop_last=False,
            shuffle=False,
        )
        train_loader = train_ds
        val_loader = val_ds
        loader_kwargs = None
    elif args.batched_gpu and use_cuda and args.batched_gpu_cpu:
        if not isinstance(train_ds, MaterializedMultiScaleDataset):
            logger.error("--batched-gpu requires MaterializedMultiScaleDataset as input")
            return 1
        logger.info(
            "ContiguousBatchDataset (CPU): batch prefetch (batch_size=%d, workers=%d, prefetch=%d).",
            train_batch_size,
            args.num_workers,
            args.prefetch_factor if args.num_workers > 0 else 0,
        )
        train_batch_ds = ContiguousBatchDataset(
            train_ds, batch_size=train_batch_size, drop_last=train_drop_last
        )
        val_batch_ds = ContiguousBatchDataset(
            val_ds, batch_size=min(args.batch_size, max(1, val_sample_count)), drop_last=False
        )
        loader_kwargs = {
            "batch_size": 1,
            "num_workers": args.num_workers,
            "pin_memory": False,
            "collate_fn": _first_collate,
        }
        if args.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = args.prefetch_factor
            if mp_context:
                loader_kwargs["multiprocessing_context"] = mp_context
        train_generator = torch.Generator()
        train_generator.manual_seed(args.seed)
        train_loader = DataLoader(
            train_batch_ds,
            shuffle=True,
            drop_last=False,
            generator=train_generator,
            worker_init_fn=seed_worker if args.num_workers > 0 else None,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_batch_ds,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
    elif (args.gpu_cache_data or args.preprocessed_dir) and use_cuda and args.gpu_cache_data:
        if not isinstance(train_ds, GpuCachedDataset):
            logger.info("Enabling GPU data cache (train + val) on %s", device)
            train_ds = GpuCachedDataset(train_ds, device=device)
            val_ds = GpuCachedDataset(val_ds, device=device)
        if args.num_workers > 0:
            logger.warning(
                "--gpu-cache-data requires num_workers=0; overriding num_workers from %d to 0",
                args.num_workers,
            )
            args.num_workers = 0
        loader_kwargs = {
            "batch_size": train_batch_size,
            "num_workers": 0,
            "pin_memory": False,
        }
        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            drop_last=train_drop_last,
            generator=torch.Generator().manual_seed(args.seed),
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            shuffle=False,
            drop_last=False,
            batch_size=min(args.batch_size, max(1, val_sample_count)),
            num_workers=0,
            pin_memory=False,
        )
    else:
        logger.info(
            "Standard DataLoader (batch_size=%d, workers=%d, prefetch=%d, pin_memory=%s)",
            train_batch_size,
            args.num_workers,
            args.prefetch_factor if args.num_workers > 0 else 0,
            use_cuda,
        )
        pin_memory = use_cuda
        loader_kwargs = {
            "batch_size": train_batch_size,
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
            drop_last=train_drop_last,
            generator=train_generator,
            worker_init_fn=seed_worker if args.num_workers > 0 else None,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_ds,
            shuffle=False,
            drop_last=False,
            batch_size=min(args.batch_size, max(1, val_sample_count)),
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            **(
                {
                    "persistent_workers": True,
                    "prefetch_factor": args.prefetch_factor,
                    **({"multiprocessing_context": mp_context} if mp_context else {}),
                }
                if args.num_workers > 0
                else {}
            ),
        )

    if train_batches_per_epoch < 20:
        logger.warning(
            "train_batches/epoch=%d 偏少（samples=%d, batch=%d）：小样本下模型易快速过拟合；"
            "建议减小 --batch-size（如 512~1024）、--num-coarse（如 64）或增大正则（--weight-decay、dropout）",
            train_batches_per_epoch,
            train_sample_count,
            train_batch_size,
        )

    if build_model_fn is not None:
        model, vq_mode, use_cosine_vq = build_model_fn(args, registry)
    else:
        vq_mode = _resolve_vq_mode(args)
        use_cosine_vq = vq_mode == "cosine"
        model = BPCv2(
            registry=registry,
            unified_dim=128,
            num_coarse=args.num_coarse,
            commitment_cost=args.commitment_cost,
            primary_scale="day",
            recon_weight=args.recon_weight,
            purity_weight=args.purity_weight,
            extended_purity_weight=args.extended_purity_weight,
            diversity_weight=args.diversity_weight,
            vq_adapt_lr=args.vq_adapt_lr,
            vq_dead_code_threshold=args.vq_dead_code_threshold,
            num_symbols=args.num_symbols,
            labeling_mode=args.labeling_mode,
            use_codebook_film=not args.no_codebook_film,
            use_fine_vq=args.use_fine_vq,
            num_fine_per_coarse=args.num_fine_per_coarse,
            use_adaptive_balance=args.adaptive_balance,
            vq_mode=vq_mode,
            threshold_decay_half_life=args.threshold_decay_half_life,
            val_threshold_ema_decay=args.val_threshold_ema_decay,
            val_threshold_ema=not args.no_val_threshold_ema,
            use_magnitude_for_purity=use_cosine_vq and not args.no_purity_magnitude,
            use_relative_z_scale_for_purity=not args.no_purity_scale_relative,
            purity_latent="quantized" if args.purity_from_quantized else "continuous",
        )
    model.to(device)
    proxies_label_ready = getattr(train_ds, "_proxies_label_ready", False)
    model.beh_loss_fn.proxies_label_ready = proxies_label_ready
    if proxies_label_ready:
        logger.info("Behavior proxies pre-transformed for labeling (skip per-batch transform)")
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
        logger.info("torch.compile enabled")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "normalizer_state" in ckpt:
            model.load_normalizer_state_dict(ckpt["normalizer_state"])
            logger.info("Restored CausalNormalizer stats from checkpoint")
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    slim_batches = getattr(train_ds, "_slim_batches", False)
    effective_workers = 0 if batched_gpu_resident else (loader_kwargs or {}).get("num_workers", args.num_workers)
    logger.info(
        "Device=%s | train_samples=%d | val_samples=%d | train_batches/epoch=%d | "
        "batch=%d (requested=%d, drop_last=%s) | workers=%d | "
        "materialize=%s | precompute_features=%s | slim_batches=%s | amp=%s",
        device,
        train_sample_count,
        val_sample_count,
        train_batches_per_epoch,
        train_batch_size,
        args.batch_size,
        train_drop_last,
        effective_workers,
        not args.no_materialize,
        args.precompute_features,
        slim_batches,
        args.amp,
    )
    logger.info(
        "VQ: mode=%s | magnitude_split=%s | purity_latent=%s | purity_magnitude=%s | "
        "purity_scale_relative=%s | codebook_film=%s | fine_vq=%s | num_fine_per_coarse=%d | "
        "commitment=%.2f | adaptive_balance=%s",
        vq_mode,
        uses_magnitude_split(vq_mode),
        "quantized" if args.purity_from_quantized else "continuous",
        use_cosine_vq and not args.no_purity_magnitude,
        not args.no_purity_scale_relative,
        not args.no_codebook_film,
        args.use_fine_vq,
        args.num_fine_per_coarse,
        args.commitment_cost,
        args.adaptive_balance,
    )
    if args.labeling_mode == "per_stock":
        logger.info(
            "Purity: per_stock frozen train thresholds | val_ema=%s (decay=%.2f) | time_decay_half_life=%.0f",
            not args.no_val_threshold_ema,
            args.val_threshold_ema_decay,
            args.threshold_decay_half_life,
        )
    if not args.use_fine_vq:
        logger.info("Fine VQ disabled (default); coarse tokens only")
    logger.info(
        "Behavior proxies: %d agents (core=%s | extended=%s)",
        len(BEHAVIOR_AGENT_NAMES),
        ",".join(CORE_AGENTS),
        ",".join(EXTENDED_AGENTS),
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

    # 关键改动：移除 precompute_normalizers（LayerNorm 替代了手动归一化）
    # 仅预计算 purity thresholds
    precompute_purity_thresholds(
        model,
        train_loader,
        device=device,
        max_batches=500,
    )
    precompute_z_scale_baselines(
        model,
        train_loader,
        device=device,
        max_batches=500,
    )

    best_val = float("inf")
    non_blocking = (
        use_cuda
        and not batched_gpu_resident
        and (loader_kwargs.get("pin_memory", False) if loader_kwargs else False)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and args.amp)
    val_max = args.val_max_batches if args.val_max_batches > 0 else None
    train_max = args.max_train_batches if args.max_train_batches > 0 else None

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
            profile_batches=args.profile_batches if epoch == start_epoch else 0,
            max_batches=train_max,
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
                skip_device_transfer=batched_gpu_resident,
                val_threshold_ema=not args.no_val_threshold_ema,
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
        
        if train_metrics:
            scheduler.step()
        elif epoch == 0 or (epoch + 1) % args.log_every == 0:
            logger.error(
                "Epoch %d: 无训练 batch（train_samples=%d, batch=%d）；请减小 --batch-size 或增加样本",
                epoch,
                train_sample_count,
                train_batch_size,
            )

        # 记录指标
        record = {
            "epoch": epoch,
            "phase": "epoch",
            "lr": lr,
            "train_samples": train_sample_count,
            "val_samples": val_sample_count,
        }
        for k, v in train_metrics.items():
            record[f"train_{k}"] = v
        for k, v in val_metrics.items():
            record[f"val_{k}"] = v

        metrics_logger.log(record)

        # 日志输出
        should_log_train = (
            (epoch + 1) % args.log_every == 0 or epoch == 0 or epoch == args.epochs - 1
        )
        if should_log_train and not train_metrics:
            logger.warning(
                "Epoch %d: train_metrics 为空（batches/epoch=%d）；本 epoch 未发生梯度更新",
                epoch,
                train_batches_per_epoch,
            )
        if train_metrics and should_log_train:
            extra = ""
            if val_metrics:
                tr_res = train_metrics.get("vq_residual_mean")
                va_res = val_metrics.get("vq_residual_mean")
                tr_z = train_metrics.get("z_norm_mean")
                va_z = val_metrics.get("z_norm_mean")
                if tr_res is not None and va_res is not None:
                    extra = f" | residual {tr_res:.4f}/{va_res:.4f}"
                if tr_z is not None and va_z is not None:
                    extra += f" | z_norm {tr_z:.4f}/{va_z:.4f}"
            _log_epoch_metrics(
                epoch,
                lr,
                train_metrics,
                val_metrics,
                extra=extra,
                codebook_film=not args.no_codebook_film,
            )
            usage = train_metrics.get("vq_usage_rate")
            perplexity = train_metrics.get("vq_perplexity", 0.0)
            if usage is not None and usage < 0.05 and epoch > 2:
                logger.warning(
                    "Low codebook usage rate %.3f (perplexity %.1f) — check VQ health",
                    usage,
                    perplexity,
                )

        # 保存最佳模型
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
                **_checkpoint_extras(model),
            }
            if split is not None:
                ckpt_payload["split"] = {
                    "train_end": str(split.train_end.date()),
                    "val_start": str(split.val_start.date()),
                }
            torch.save(ckpt_payload, best_path)

        # 定期保存
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = out_dir / f"bpc_v2_epoch_{epoch:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    **_checkpoint_extras(model),
                },
                ckpt_path,
            )
            logger.info("Checkpoint: %s", ckpt_path)

    # 保存最终模型
    model.save_behavioral_ontology(str(out_dir / "behavioral_ontology_v1.pt"))
    torch.save({"model": model.state_dict(), "epoch": args.epochs - 1}, out_dir / "bpc_v2_last.pt")

    # Token 语义分析（analyze_token_semantics 返回 dict，semantics 键为 DataFrame）
    sem_loader = val_loader
    semantics_result = model.analyze_token_semantics(sem_loader, device=device, max_samples=5000)
    sem_df = semantics_result.get("semantics")
    if sem_df is not None and hasattr(sem_df, "empty") and not sem_df.empty:
        sem_df.to_csv(out_dir / "token_semantics_val.csv")
        logger.info("Token semantics saved (%d records)", semantics_result.get("n_records", 0))
    else:
        logger.warning("Token semantics empty — skipped CSV export")

    metrics_logger.close()
    logger.info("Training complete. Logs: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())