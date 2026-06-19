"""bpc_v4 Qlib 数据集：单次物化 + 预处理缓存 + DataLoader 策略。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import qlib
    from qlib.data import D
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False

from .config import GlobalConfig
from .features import compute_bpc_features, compute_context_features, compute_time_embedding
from .kronos import KronosTokenizerPool
from .materialize import (
    BatchedGpuBPCV4Dataset,
    ContiguousBatchBPCV4Dataset,
    GpuCachedBPCV4Dataset,
    MaterializedBPCV4Dataset,
    load_materialized_dataset,
    pin_dataset_share_memory,
    save_materialized_dataset,
)

logger = logging.getLogger(__name__)


@dataclass
class LoaderOptions:
    num_workers: int = 8
    prefetch_factor: int = 4
    gpu_cache_data: bool = False
    batched_gpu: bool = False
    batched_gpu_cpu: bool = False
    seed: int = 42


def resolve_qlib_instruments(
    market: str = "csi300",
    n_instruments: Optional[int] = None,
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    provider_uri: Path = Path("~/.qlib/qlib_data/cn_data"),
) -> List[str]:
    uri = str(provider_uri.expanduser())
    qlib.init(provider_uri=uri, region="cn")
    all_inst = D.list_instruments(
        instruments=D.instruments(market),
        start_time=start,
        end_time=end,
        as_list=True,
    )
    codes = sorted(all_inst)
    if n_instruments is not None:
        codes = codes[:n_instruments]
    logger.info("Resolved %d instruments from qlib market=%s", len(codes), market)
    return codes


def _resolve_feature_codes(instruments: List[str]):
    if len(instruments) == 1 and "." not in instruments[0]:
        return D.instruments(instruments[0])
    return instruments


def _load_qlib_series(config: GlobalConfig) -> Tuple[Dict[str, np.ndarray], List[str]]:
    if not QLIB_AVAILABLE:
        raise RuntimeError("qlib 未安装")

    qlib.init(provider_uri=str(config.qlib.provider_uri.expanduser()), region="cn")
    codes = _resolve_feature_codes(config.qlib.instruments)
    fields = ["$open", "$high", "$low", "$close", "$volume", "$amount"]
    df = D.features(codes, fields, start_time=config.qlib.start_date, end_time=config.qlib.end_date, freq="day")

    symbols = df.index.get_level_values(1).unique().tolist()
    series: Dict[str, np.ndarray] = {}
    for sym in symbols:
        sym_df = df.xs(sym, level=1)
        series[sym] = sym_df[fields].values.astype(np.float32)

    logger.info("Loaded %d symbols from qlib", len(symbols))
    return series, symbols


def _build_sample_list(
    series: Dict[str, np.ndarray],
    seq_len: int,
    max_samples_per_instrument: Optional[int] = None,
) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    for sym, arr in series.items():
        T = arr.shape[0]
        sym_samples = [(sym, t) for t in range(seq_len, T)]
        if max_samples_per_instrument is not None:
            sym_samples = sym_samples[:max_samples_per_instrument]
        samples.extend(sym_samples)
    return samples


def _apply_split(
    samples: List[Tuple[str, int]],
    mode: str,
    val_ratio: float,
    test_ratio: float,
    max_samples: Optional[int],
) -> List[Tuple[str, int]]:
    split_idx = int(len(samples) * (1 - val_ratio - test_ratio))
    val_idx = int(len(samples) * (1 - test_ratio))

    if mode == "train":
        out = samples[:split_idx]
    elif mode == "val":
        out = samples[split_idx:val_idx]
    else:
        out = samples[val_idx:]

    if max_samples is not None and max_samples > 0:
        if mode == "train":
            cap = max(1, int(max_samples * 0.8))
        elif mode == "val":
            cap = max(1, max_samples - int(max_samples * 0.8))
        else:
            cap = max(1, max_samples // 10)
        out = out[:cap]
    return out


def materialize_samples(
    config: GlobalConfig,
    samples: List[Tuple[str, int]],
    series: Dict[str, np.ndarray],
    symbols: List[str],
    *,
    share_memory: bool = False,
) -> MaterializedBPCV4Dataset:
    seq_len = config.kronos.seq_len
    kronos = KronosTokenizerPool(
        model_name=config.kronos.model_name,
        local_path=config.kronos.local_path,
        device=config.train.device,
    )

    z_q_list, bpc_list, ctx_list, time_list, stock_list, s1_list = [], [], [], [], [], []

    for i, (sym, t_idx) in enumerate(samples):
        window = series[sym][t_idx - seq_len : t_idx]
        prev_bar = series[sym][t_idx - seq_len - 1]

        ohlcv = torch.from_numpy(window[:, :5]).float().unsqueeze(0)
        amount = torch.from_numpy(window[:, 5:6]).float().unsqueeze(0)
        ohlcva = torch.cat([ohlcv, amount], dim=-1)

        z_q, s1_ids, _ = kronos.encode(ohlcva)
        z_q_list.append(z_q.squeeze(0).cpu())
        s1_list.append(s1_ids.squeeze(0).cpu())

        vol_ctx = torch.zeros(1, 3)
        prev_bar_t = torch.from_numpy(prev_bar[:5]).float().unsqueeze(0)
        bpc_list.append(compute_bpc_features(ohlcv, vol_ctx, prev_bar_t).squeeze(0).cpu())
        ctx_list.append(compute_context_features(ohlcv, vol_ctx, prev_bar_t).squeeze(0).cpu())
        time_list.append(
            compute_time_embedding(
                torch.tensor([t_idx], dtype=torch.long),
                raw_dim=config.embedding.time_raw_dim,
            ).squeeze(0).cpu()
        )
        stock_list.append(symbols.index(sym) if sym in symbols else 0)

        if (i + 1) % 5000 == 0:
            logger.info("Materialized %d/%d samples", i + 1, len(samples))

    ds = MaterializedBPCV4Dataset(
        z_q=torch.stack(z_q_list),
        bpc_feat=torch.stack(bpc_list),
        ctx_feat=torch.stack(ctx_list),
        time_emb=torch.stack(time_list),
        stock_id=torch.tensor(stock_list, dtype=torch.long),
        s1_ids=torch.stack(s1_list),
    )
    if share_memory:
        ds.share_memory_()
    return ds


def build_datasets(
    config: GlobalConfig,
    *,
    share_memory: bool = False,
    max_samples_per_instrument: Optional[int] = None,
    preprocessed_dir: Optional[Path] = None,
    save_preprocessed: Optional[Path] = None,
    force_rebuild: bool = False,
) -> Tuple[MaterializedBPCV4Dataset, MaterializedBPCV4Dataset, MaterializedBPCV4Dataset]:
    cache_root = preprocessed_dir or save_preprocessed
    if cache_root and not force_rebuild:
        train_p = cache_root / "train"
        val_p = cache_root / "val"
        test_p = cache_root / "test"
        if train_p.exists() and val_p.exists():
            logger.info("Loading preprocessed cache from %s", cache_root)
            train_ds = load_materialized_dataset(train_p)
            val_ds = load_materialized_dataset(val_p)
            test_ds = load_materialized_dataset(test_p) if test_p.exists() else val_ds
            return train_ds, val_ds, test_ds

    series, symbols = _load_qlib_series(config)
    all_samples = _build_sample_list(series, config.kronos.seq_len, max_samples_per_instrument)

    train_samples = _apply_split(all_samples, "train", config.qlib.val_ratio, config.qlib.test_ratio, config.qlib.max_samples)
    val_samples = _apply_split(all_samples, "val", config.qlib.val_ratio, config.qlib.test_ratio, config.qlib.max_samples)
    test_samples = _apply_split(all_samples, "test", config.qlib.val_ratio, config.qlib.test_ratio, config.qlib.max_samples)

    logger.info("Materializing train=%d val=%d test=%d samples", len(train_samples), len(val_samples), len(test_samples))

    train_ds = materialize_samples(config, train_samples, series, symbols, share_memory=share_memory)
    val_ds = materialize_samples(config, val_samples, series, symbols, share_memory=share_memory)
    test_ds = materialize_samples(config, test_samples, series, symbols, share_memory=share_memory)

    if save_preprocessed:
        save_preprocessed.mkdir(parents=True, exist_ok=True)
        save_materialized_dataset(train_ds, save_preprocessed / "train")
        save_materialized_dataset(val_ds, save_preprocessed / "val")
        save_materialized_dataset(test_ds, save_preprocessed / "test")
        meta = {
            "schema_version": "bpc_v4",
            "start_date": config.qlib.start_date,
            "end_date": config.qlib.end_date,
            "n_instruments": len(config.qlib.instruments),
            "max_samples": config.qlib.max_samples,
            "max_samples_per_instrument": max_samples_per_instrument,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
        }
        (save_preprocessed / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Saved preprocessed cache to %s", save_preprocessed)

    return train_ds, val_ds, test_ds


def _train_loader_batch_config(n_samples: int, batch_size: int) -> Tuple[int, bool, int]:
    if n_samples <= 0:
        return 1, False, 0
    if n_samples >= batch_size:
        eff = batch_size
        drop_last = True
    else:
        eff = max(1, n_samples)
        drop_last = False
    batches = n_samples // eff if drop_last else (n_samples + eff - 1) // eff
    return eff, drop_last, batches


def _first_collate(batch: list) -> dict:
    return batch[0]


def _loader_has_gpu_batches(loader) -> bool:
    if hasattr(loader, "iter_batches"):
        return True
    ds = getattr(loader, "dataset", None)
    return ds is not None and hasattr(ds, "iter_batches")


def iter_training_batches(loader):
    if hasattr(loader, "iter_batches"):
        if hasattr(loader, "on_epoch_begin"):
            loader.on_epoch_begin()
        yield from loader.iter_batches()
        return
    ds = getattr(loader, "dataset", loader)
    if hasattr(ds, "iter_batches"):
        ds.on_epoch_begin()
        yield from ds.iter_batches()
        return
    if hasattr(ds, "on_epoch_begin"):
        ds.on_epoch_begin()
    yield from loader


def to_device(batch: dict, device: torch.device, *, non_blocking: bool = False) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v if v.device == device else v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out


def collate_fn(batch: list) -> dict:
    return {
        "z_q": torch.stack([b["z_q"] for b in batch]),
        "bpc_feat": torch.stack([b["bpc_feat"] for b in batch]),
        "ctx_feat": torch.stack([b["ctx_feat"] for b in batch]),
        "time_emb": torch.stack([b["time_emb"] for b in batch]),
        "stock_id": torch.stack([b["stock_id"] for b in batch]),
        "s1_ids": torch.stack([b["s1_ids"] for b in batch]),
    }


def create_dataloaders(
    config: GlobalConfig,
    opts: LoaderOptions,
    *,
    preprocessed_dir: Optional[Path] = None,
    save_preprocessed: Optional[Path] = None,
    force_rebuild_preprocessed: bool = False,
    max_samples_per_instrument: Optional[int] = None,
) -> Tuple[Union[DataLoader, BatchedGpuBPCV4Dataset], Union[DataLoader, BatchedGpuBPCV4Dataset], MaterializedBPCV4Dataset]:
    share_memory = opts.num_workers > 0 and not opts.gpu_cache_data and not opts.batched_gpu
    train_ds, val_ds, test_ds = build_datasets(
        config,
        share_memory=share_memory,
        max_samples_per_instrument=max_samples_per_instrument,
        preprocessed_dir=preprocessed_dir,
        save_preprocessed=save_preprocessed,
        force_rebuild=force_rebuild_preprocessed,
    )

    device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    train_bs, train_drop_last, _ = _train_loader_batch_config(len(train_ds), config.train.batch_size)
    val_bs = min(config.train.batch_size, max(1, len(val_ds)))

    batched_gpu_resident = opts.batched_gpu and use_cuda and not opts.batched_gpu_cpu

    if batched_gpu_resident:
        if opts.num_workers > 0:
            logger.info("--batched-gpu: ignoring num_workers=%d", opts.num_workers)
        logger.info("BatchedGpuBPCV4Dataset on %s (batch_size=%d)", device, train_bs)
        train_loader: Union[DataLoader, BatchedGpuBPCV4Dataset] = BatchedGpuBPCV4Dataset(
            train_ds, device=str(device), batch_size=train_bs, drop_last=train_drop_last, shuffle=True
        )
        val_loader = BatchedGpuBPCV4Dataset(
            val_ds, device=str(device), batch_size=val_bs, drop_last=False, shuffle=False
        )
        return train_loader, val_loader, test_ds

    if opts.batched_gpu and use_cuda and opts.batched_gpu_cpu:
        logger.info(
            "ContiguousBatchBPCV4Dataset (CPU prefetch, workers=%d, prefetch=%d)",
            opts.num_workers,
            opts.prefetch_factor if opts.num_workers > 0 else 0,
        )
        train_batch_ds = ContiguousBatchBPCV4Dataset(train_ds, batch_size=train_bs, drop_last=train_drop_last)
        val_batch_ds = ContiguousBatchBPCV4Dataset(val_ds, batch_size=val_bs, drop_last=False)
        loader_kwargs = {"batch_size": 1, "num_workers": opts.num_workers, "pin_memory": False, "collate_fn": _first_collate}
        if opts.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = opts.prefetch_factor
        gen = torch.Generator().manual_seed(opts.seed)
        return (
            DataLoader(train_batch_ds, shuffle=True, drop_last=False, generator=gen, **loader_kwargs),
            DataLoader(val_batch_ds, shuffle=False, drop_last=False, **loader_kwargs),
            test_ds,
        )

    if opts.gpu_cache_data and use_cuda:
        logger.info("GpuCachedBPCV4Dataset on %s (num_workers=0, zero H2D)", device)
        train_ds = GpuCachedBPCV4Dataset(train_ds, device=str(device))
        val_ds = GpuCachedBPCV4Dataset(val_ds, device=str(device))
        if opts.num_workers > 0:
            logger.warning("--gpu-cache-data forces num_workers=0 (was %d)", opts.num_workers)
        return (
            DataLoader(train_ds, batch_size=train_bs, shuffle=True, drop_last=train_drop_last, num_workers=0, pin_memory=False, collate_fn=collate_fn),
            DataLoader(val_ds, batch_size=val_bs, shuffle=False, drop_last=False, num_workers=0, pin_memory=False, collate_fn=collate_fn),
            test_ds,
        )

    if share_memory:
        pin_dataset_share_memory(train_ds)
        pin_dataset_share_memory(val_ds)
        logger.info("share_memory enabled (num_workers=%d)", opts.num_workers)

    logger.info(
        "Standard DataLoader (batch=%d, workers=%d, prefetch=%d, pin_memory=%s)",
        train_bs,
        opts.num_workers,
        opts.prefetch_factor if opts.num_workers > 0 else 0,
        use_cuda,
    )
    loader_kwargs = {"num_workers": opts.num_workers, "pin_memory": use_cuda, "collate_fn": collate_fn}
    if opts.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = opts.prefetch_factor
    gen = torch.Generator().manual_seed(opts.seed)
    return (
        DataLoader(train_ds, batch_size=train_bs, shuffle=True, drop_last=train_drop_last, generator=gen, **loader_kwargs),
        DataLoader(val_ds, batch_size=val_bs, shuffle=False, drop_last=False, **loader_kwargs),
        test_ds,
    )
