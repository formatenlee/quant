"""
BPC-v3 训练入口：复用 bpc.train 流程，注入 v3 模型/特征与推荐超参。

示例:
  python -m quant_cursor.bpc_v3.train --dev --epochs 100 --device cuda

预处理缓存（须用 bpc_v3 保存，26 维特征）:
  python -m quant_cursor.bpc_v3.train ... --save-preprocessed data/preprocessed/bpc_v3_run
  python -m quant_cursor.bpc_v3.train ... --preprocessed-dir data/preprocessed/bpc_v3_run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import quant_cursor.bpc.train as _train
from quant_cursor.bpc_v3 import behavior_features as _bf_v3
from quant_cursor.bpc_v3.behavior_features import BEHAVIOR_LABEL_SCHEMA
from quant_cursor.bpc_v3 import dataset as _dataset_v3
from quant_cursor.bpc_v3 import feature_dims as _fd_v3
from quant_cursor.bpc_v3.model import (
    BPCv3,
    adapt_codebook_on_loader,
    build_scale_registry,
    eval_epoch,
    precompute_normalizers,
    precompute_purity_thresholds,
    precompute_z_scale_baselines,
    train_epoch,
)
from quant_cursor.bpc_v3.loss_plots import LossCurveTracker
from quant_cursor.bpc_v3.vq_backend import normalize_vq_mode, uses_magnitude_split
from quant_cursor.bpc.metrics import MetricsLogger as _BaseMetricsLogger

logger = logging.getLogger("quant_cursor.bpc_v3.train")


class MetricsLoggerV3(_BaseMetricsLogger):
    """写入 metrics 并累积 loss 历史；绘图在 _log_epoch_metrics 回显时触发。"""

    def __init__(self, log_dir: Path):
        super().__init__(log_dir)
        self._loss_tracker = LossCurveTracker(log_dir)
        _train._bpc_v3_loss_tracker = self._loss_tracker

    def log(self, record: dict) -> None:
        super().log(record)
        if record.get("phase") == "epoch":
            self._loss_tracker.update_from_record(record)

# 注入 v3 实现（不修改 bpc 源码文件）
_train.BPCv2 = BPCv3
_train.build_scale_registry = build_scale_registry
_train.precompute_purity_thresholds = precompute_purity_thresholds
_train.precompute_z_scale_baselines = precompute_z_scale_baselines


def _precompute_normalizers_v3(*_args, **_kwargs) -> None:
    logger.info("BPC-v3: skip CausalNormalizer fit (precomputed features, no feature z-score)")


_train.precompute_normalizers = _precompute_normalizers_v3
_train.adapt_codebook_on_loader = adapt_codebook_on_loader
_train.eval_epoch = eval_epoch
_train.train_epoch = train_epoch
_train.normalize_vq_mode = normalize_vq_mode
_train.uses_magnitude_split = uses_magnitude_split
_train.build_datasets = _dataset_v3.build_datasets
_train.load_materialized_dataset = _dataset_v3.load_materialized_dataset
_train.save_materialized_dataset = _dataset_v3.save_materialized_dataset
_train.MaterializedMultiScaleDataset = _dataset_v3.MaterializedMultiScaleDataset
_train.pin_dataset_share_memory = _dataset_v3.pin_dataset_share_memory
_train.MetricsLogger = MetricsLoggerV3

_orig_save_split_meta = _train._save_split_meta
_orig_log_epoch_metrics = _train._log_epoch_metrics
_orig_split_meta_compatible = _train._split_meta_compatible
_orig_resolve_datasets = _train._resolve_datasets


def _v3_split_meta_fields() -> dict[str, object]:
    return {
        "schema_version": _dataset_v3.BPC_SCHEMA_VERSION,
        "day_feature_dim": _fd_v3.DAY_FULL_FEAT_DIM,
        "day_struct_feat_dim": _fd_v3.DAY_STRUCT_FEAT_DIM,
        "num_behavior_agents": _bf_v3.NUM_BEHAVIOR_AGENTS,
        "behavior_label_schema": BEHAVIOR_LABEL_SCHEMA,
    }


def _augment_split_meta_v3(meta: dict) -> dict:
    out = dict(meta)
    out.update(_v3_split_meta_fields())
    return out


def _split_meta_compatible_v3(
    meta: dict,
    *,
    eff_start: str,
    eff_end: str,
    args: argparse.Namespace,
    instrument_list: list[str],
    cache_dir: Path | None = None,
) -> tuple[bool, str]:
    ok, reason = _orig_split_meta_compatible(
        meta,
        eff_start=eff_start,
        eff_end=eff_end,
        args=args,
        instrument_list=instrument_list,
        cache_dir=cache_dir,
    )
    if not ok:
        return ok, reason

    saved_schema = meta.get("schema_version")
    saved_dim = meta.get("day_feature_dim")
    saved_label = meta.get("behavior_label_schema")
    expected = _v3_split_meta_fields()

    if saved_schema is not None and saved_schema != expected["schema_version"]:
        return False, f"schema_version {saved_schema!r} != {expected['schema_version']!r}"

    if saved_dim is not None and int(saved_dim) != int(expected["day_feature_dim"]):
        return False, (
            f"day_feature_dim {saved_dim} != {expected['day_feature_dim']} "
            "(v3 特征维变更，需 --force-rebuild-preprocessed)"
        )

    if saved_label is not None and saved_label != expected["behavior_label_schema"]:
        return False, (
            f"behavior_label_schema {saved_label!r} != {expected['behavior_label_schema']!r} "
            "(代理标签逻辑已更新，需 --force-rebuild-preprocessed)"
        )

    if cache_dir is not None and saved_schema is None:
        train_meta_path = cache_dir / "train" / "meta.json"
        day_npy = cache_dir / "train" / "day_features.npy"
        if day_npy.exists():
            import numpy as np

            actual_dim = int(np.load(day_npy, mmap_mode="r").shape[1])
            if actual_dim != int(expected["day_feature_dim"]):
                return False, (
                    f"train/day_features dim {actual_dim} != v3 {expected['day_feature_dim']} "
                    "(旧 v2 缓存，请用 bpc_v3.train 重建 --save-preprocessed)"
                )
        if train_meta_path.exists():
            train_meta = json.loads(train_meta_path.read_text(encoding="utf-8"))
            meta_dim = train_meta.get("day_feature_dim")
            if meta_dim is not None and int(meta_dim) != int(expected["day_feature_dim"]):
                return False, (
                    f"train/meta.json day_feature_dim={meta_dim} != v3 {expected['day_feature_dim']}"
                )
        logger.warning(
            "预处理 split_meta 缺少 schema_version/behavior_label_schema；"
            "将尝试加载，但若为 path/vol 分位数版缓存，请 --force-rebuild-preprocessed 刷新代理标签"
        )

    return True, "ok"


def _save_split_meta_v3(*args, **kwargs):
    _orig_save_split_meta(*args, **kwargs)
    log_dir = Path(args[0]) if args else Path(kwargs["log_dir"])
    meta_path = log_dir / "split_meta.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = _augment_split_meta_v3(meta)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_datasets_v3(
    args: argparse.Namespace,
    *,
    instrument_list: list[str],
    qlib_uri: Path,
    registry,
):
    calendar = _train.load_trading_calendar(qlib_uri)
    eff_start, eff_end = _train._resolve_effective_dates(args, calendar)
    cache_dir: Path | None = None
    if args.preprocessed_dir:
        cache_dir = Path(args.preprocessed_dir)
    elif (
        args.save_preprocessed
        and not args.force_rebuild_preprocessed
        and (Path(args.save_preprocessed) / "train").exists()
        and (Path(args.save_preprocessed) / "val").exists()
    ):
        cache_dir = Path(args.save_preprocessed)

    if cache_dir is not None:
        meta = _train._load_split_meta(cache_dir)
        if meta is None:
            logger.warning(
                "预处理目录 %s 缺少 split_meta.json，将走 qlib 物化（较慢）。"
                "请用相同参数 --save-preprocessed 生成完整缓存。",
                cache_dir,
            )
        else:
            ok, reason = _split_meta_compatible_v3(
                meta,
                eff_start=eff_start,
                eff_end=eff_end,
                args=args,
                instrument_list=instrument_list,
                cache_dir=cache_dir,
            )
            if not ok:
                hint = (
                    "常见原因：v3 默认 --val-ratio=0.15 与缓存不一致、"
                    "--max-instruments/--full/--max-samples-per-instrument 不同、"
                    "manifest 标的集变化、或 behavior_label_schema 过期。"
                )
                if args.preprocessed_dir:
                    raise RuntimeError(
                        f"无法加载 --preprocessed-dir={cache_dir}：{reason}。{hint}"
                        "请用与保存时相同的 CLI 参数重新 --save-preprocessed，"
                        "或加 --force-rebuild-preprocessed。"
                    )
                logger.warning(
                    "预处理缓存 %s 与当前参数不兼容（%s）；将重新从 qlib 物化（本次会很慢）。%s",
                    cache_dir,
                    reason,
                    hint,
                )
            else:
                logger.info(
                    "预处理缓存参数匹配 | schema=%s | day_feature_dim=%s | behavior_label_schema=%s",
                    meta.get("schema_version", "legacy"),
                    meta.get("day_feature_dim", "?"),
                    meta.get("behavior_label_schema", "legacy"),
                )

    return _orig_resolve_datasets(
        args,
        instrument_list=instrument_list,
        qlib_uri=qlib_uri,
        registry=registry,
    )


_train._split_meta_compatible = _split_meta_compatible_v3


def _log_epoch_metrics_v3(*args, **kwargs):
    _orig_log_epoch_metrics(*args, **kwargs)
    tracker = getattr(_train, "_bpc_v3_loss_tracker", None)
    if tracker is not None:
        tracker.render()


_train._log_epoch_metrics = _log_epoch_metrics_v3
_train._save_split_meta = _save_split_meta_v3
_train._resolve_datasets = _resolve_datasets_v3


def build_model_v3(args: argparse.Namespace, registry):
    """BPC-v3 默认更小容量 + 更强行为头 dropout。"""
    vq_mode = _train._resolve_vq_mode(args)
    use_cosine_vq = vq_mode == "cosine"
    unified_dim = int(getattr(args, "unified_dim", 96))
    behavior_dropout = float(getattr(args, "behavior_dropout", 0.45))
    model = BPCv3(
        registry=registry,
        unified_dim=unified_dim,
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
        use_magnitude_for_purity=use_cosine_vq and (
            getattr(args, "purity_magnitude", False) and not args.no_purity_magnitude
        ),
        use_relative_z_scale_for_purity=(
            getattr(args, "purity_scale_relative", False) and not args.no_purity_scale_relative
        ),
        purity_latent="quantized" if args.purity_from_quantized else "continuous",
        behavior_dropout=behavior_dropout,
    )
    logger.info(
        "BPC-v3 model: unified_dim=%d num_coarse=%d behavior_dropout=%.2f",
        unified_dim,
        args.num_coarse,
        behavior_dropout,
    )
    return model, vq_mode, use_cosine_vq


_train.build_model_fn = build_model_v3

_v3_small_data_flag = False
_v3_purity_magnitude = False
_v3_purity_scale_relative = False
_orig_parse_args = _train.parse_args


def _parse_args_v3():
    args = _orig_parse_args()
    _apply_v3_runtime_defaults(args, small_data=_v3_small_data_flag)
    return args


_train.parse_args = _parse_args_v3

_V3_VALUE_DEFAULTS: dict[str, str] = {
    "--val-ratio": "0.15",
    "--val-every": "20",
    "--val-threshold-ema-decay": "0.90",
    "--weight-decay": "0.08",
    "--diversity-weight": "0.15",
    "--batch-size": "1024",
    "--num-coarse": "64",
    "--purity-weight": "0.35",
    "--extended-purity-weight": "0.10",
}


def _flag_present(argv: list[str], flag: str) -> bool:
    return any(a == flag or a.startswith(f"{flag}=") for a in argv)


def _inject_v3_cli_defaults(argv: list[str]) -> tuple[list[str], bool]:
    global _v3_small_data_flag, _v3_purity_magnitude, _v3_purity_scale_relative
    out: list[str] = []
    small_data = False
    _v3_purity_magnitude = _flag_present(argv, "--purity-magnitude")
    _v3_purity_scale_relative = _flag_present(argv, "--purity-scale-relative")
    for arg in argv:
        if arg == "--small-data":
            small_data = True
            continue
        if arg in ("--purity-magnitude", "--purity-scale-relative"):
            continue
        out.append(arg)
    for flag, value in _V3_VALUE_DEFAULTS.items():
        if not _flag_present(out, flag):
            out.extend([flag, value])
    if not _v3_purity_magnitude and not _flag_present(out, "--no-purity-magnitude"):
        out.append("--no-purity-magnitude")
    if not _v3_purity_scale_relative and not _flag_present(out, "--no-purity-scale-relative"):
        out.append("--no-purity-scale-relative")
    if not _flag_present(out, "--output-dir"):
        out.extend(["--output-dir", "__BPC_V3_AUTO__"])
    return out, small_data


def _apply_v3_runtime_defaults(args: argparse.Namespace, *, small_data: bool) -> None:
    if not hasattr(args, "unified_dim"):
        args.unified_dim = 96
    if not hasattr(args, "behavior_dropout"):
        args.behavior_dropout = 0.45
    args.purity_magnitude = _v3_purity_magnitude
    args.purity_scale_relative = _v3_purity_scale_relative
    if small_data:
        args.batch_size = min(int(args.batch_size), 512)
        args.num_coarse = min(int(args.num_coarse), 48)
        args.unified_dim = min(int(args.unified_dim), 96)
        args.behavior_dropout = max(float(args.behavior_dropout), 0.50)
        args.weight_decay = max(float(args.weight_decay), 0.10)
        logger.info(
            "Small-data preset: batch=%d num_coarse=%d unified_dim=%d behavior_dropout=%.2f weight_decay=%.2f",
            args.batch_size,
            args.num_coarse,
            args.unified_dim,
            args.behavior_dropout,
            args.weight_decay,
        )


def _resolve_output_dir(argv: list[str]) -> list[str]:
    if "__BPC_V3_AUTO__" not in argv:
        return argv
    out: list[str] = []
    for arg in argv:
        if arg == "__BPC_V3_AUTO__":
            from quant_cursor.config import load_config

            config = load_config()
            run_name = f"run_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}"
            out.append(str(config.data_dir / "checkpoints" / "bpc_v3" / run_name))
        else:
            out.append(arg)
    return out


def parse_args():
    global _v3_small_data_flag
    argv, _v3_small_data_flag = _inject_v3_cli_defaults(sys.argv[1:])
    argv = _resolve_output_dir(argv)
    old_argv = sys.argv
    sys.argv = [old_argv[0], *argv]
    try:
        return _parse_args_v3()
    finally:
        sys.argv = old_argv


def main() -> int:
    global _v3_small_data_flag
    argv, _v3_small_data_flag = _inject_v3_cli_defaults(sys.argv[1:])
    argv = _resolve_output_dir(argv)
    old_argv = sys.argv
    sys.argv = [old_argv[0], *argv]
    try:
        logging.getLogger("quant_cursor.bpc").setLevel(logging.INFO)
        logger.info(
            "BPC-v3 defaults: val_ratio=%s batch_size=%s num_coarse=%s weight_decay=%s "
            "diversity_weight=%s purity_weight=%s behavior_dropout=%s unified_dim=%s",
            _V3_VALUE_DEFAULTS["--val-ratio"],
            _V3_VALUE_DEFAULTS["--batch-size"],
            _V3_VALUE_DEFAULTS["--num-coarse"],
            _V3_VALUE_DEFAULTS["--weight-decay"],
            _V3_VALUE_DEFAULTS["--diversity-weight"],
            _V3_VALUE_DEFAULTS["--purity-weight"],
            "0.45",
            "96",
        )
        if _v3_small_data_flag:
            logger.info("Small-data preset enabled (--small-data)")
        return _train.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
