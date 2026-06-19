"""
BPC-v4 训练入口（仿 bpc_v3，从项目根目录启动）。

示例:
  python -m quant_cursor.bpc_v4.train --dev --device cuda

GPU 显存驻留（消除 H2D）:
  python -m quant_cursor.bpc_v4.train --dev --gpu-cache-data --device cuda
  python -m quant_cursor.bpc_v4.train --dev --batched-gpu --device cuda

预处理缓存:
  python -m quant_cursor.bpc_v4.train --dev --save-preprocessed data/preprocessed/bpc_v4_smoke
  python -m quant_cursor.bpc_v4.train --dev --preprocessed-dir data/preprocessed/bpc_v4_smoke --gpu-cache-data
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from quant_cursor.bpc.dataset import load_qlib_instruments
from quant_cursor.config import load_config

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
        args.start,
        args.end,
        args.max_instruments,
        args.max_samples_per_instrument,
        args.max_samples,
        args.epochs,
        args.batch_size,
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

    p = argparse.ArgumentParser(description="BPC-v4 训练（qlib + Kronos，接口对齐 v3）")
    p.add_argument("--config", type=str, default=None, help="quant_cursor config.yaml")
    p.add_argument("--dev", action="store_true", help="快速验证：2019~今天 + 200标的 + 每标的200窗")
    p.add_argument("--small-data", action="store_true", help="更小 batch/worker 预设")
    p.add_argument("--full", action="store_true", help="manifest 全量标的，不限制每标的采样")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--max-calendar-days", type=int, default=None)
    p.add_argument("--instruments", nargs="*", default=None)
    p.add_argument("--max-instruments", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None, help="限制 train+val 总窗口数")
    p.add_argument("--max-samples-per-instrument", type=int, default=None)
    p.add_argument("--min-rows", type=int, default=60)
    p.add_argument("--val-ratio", type=float, default=0.15)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", help="CUDA 混合精度")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--val-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--val-max-batches", type=int, default=0, help="0=全量验证")
    p.add_argument("--no-console-log", action="store_true")

    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--gpu-cache-data", action="store_true", help="全量样本上传 GPU，num_workers→0，零 H2D")
    p.add_argument("--batched-gpu", action="store_true", help="GPU batch 切片，num_workers→0")
    p.add_argument("--batched-gpu-cpu", action="store_true", help="CPU batch 预取 + 训练时 H2D")
    p.add_argument("--batched-gpu-resident", action="store_true", help="已废弃，等同 --batched-gpu")

    p.add_argument("--preprocessed-dir", type=str, default=None, help="加载磁盘物化缓存，跳过 qlib+Kronos")
    p.add_argument("--save-preprocessed", type=str, default=None, help="物化后保存；参数匹配时自动复用")
    p.add_argument("--force-rebuild-preprocessed", action="store_true")
    p.add_argument("--preprocess-only", action="store_true", help="仅物化数据后退出")

    p.add_argument("--kronos-path", type=str, default=None, help="Kronos 本地模型目录")

    args = p.parse_args(injected)
    return args, small_data


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
            args.full,
            args.max_instruments,
            len(instruments),
        )

    if not instruments:
        raise RuntimeError(f"无可用标的，请检查 manifest 或 --instruments")

    out_dir = Path(args.output_dir) if args.output_dir else qc_cfg.data_dir / "checkpoints" / "bpc_v4" / "run_manual"
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_run_config(out_dir, args)

    from bpc_v4.config import GlobalConfig
    from bpc_v4.dataset import LoaderOptions
    from bpc_v4.train import run_training

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

    max_spi = _resolve_sample_cap(args)
    logger.info(
        "BPC-v4 run: instruments=%d window=%s..%s qlib=%s output=%s | "
        "gpu_cache=%s batched_gpu=%s workers=%d prefetch=%d",
        len(instruments),
        start,
        end,
        config.qlib.provider_uri,
        out_dir,
        args.gpu_cache_data,
        args.batched_gpu,
        num_workers,
        args.prefetch_factor,
    )

    run_training(
        config,
        loader_opts,
        resume=args.resume,
        preprocessed_dir=Path(args.preprocessed_dir) if args.preprocessed_dir else None,
        save_preprocessed=Path(args.save_preprocessed) if args.save_preprocessed else None,
        force_rebuild_preprocessed=args.force_rebuild_preprocessed,
        max_samples_per_instrument=max_spi,
        val_every=args.val_every,
        save_every=args.save_every,
        val_max_batches=args.val_max_batches,
        preprocess_only=args.preprocess_only,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
