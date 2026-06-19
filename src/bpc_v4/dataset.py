"""bpc_v4 Qlib 数据集：逐标的加载 + 日历切分 + 预处理缓存 + DataLoader 策略。"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import qlib
    from qlib.data import D
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False

from quant_cursor.bpc.dataset import TemporalSplit, ensure_qlib, load_trading_calendar

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

OHLCVA_FIELDS = ["$open", "$high", "$low", "$close", "$volume", "$amount"]


@dataclass
class _V4InstrumentSeries:
    qlib_id: str
    dates: pd.DatetimeIndex
    ohlcva: np.ndarray


class BPCV4InstrumentStore:
    """按标的逐个从 qlib 加载 OHLCVA，仅拉取 config 指定日期窗口。"""

    def __init__(
        self,
        instruments: List[str],
        start: str,
        end: str,
        provider_uri: Path,
        *,
        calendar: pd.DatetimeIndex | None = None,
        pad_missing_amount: bool = True,
    ):
        if not QLIB_AVAILABLE:
            raise RuntimeError("qlib 未安装")

        ensure_qlib(provider_uri)
        self.provider_uri = Path(provider_uri)
        self.start = start
        self.end = end
        self.calendar = calendar if calendar is not None else load_trading_calendar(self.provider_uri)
        self.pad_missing_amount = pad_missing_amount

        self._cache: Dict[str, _V4InstrumentSeries] = {}
        total = len(instruments)
        for i, qlib_id in enumerate(instruments, 1):
            series = self._load_one(qlib_id)
            if series is not None:
                self._cache[qlib_id] = series
            if i % 50 == 0 or i == total:
                logger.info(
                    "BPC-v4 InstrumentStore: loaded %d/%d (cached=%d)",
                    i,
                    total,
                    len(self._cache),
                )

        if not self._cache:
            raise RuntimeError("BPC-v4 InstrumentStore: 无有效标的")

        self.symbol_to_id: Dict[str, int] = {qlib_id: i for i, qlib_id in enumerate(self._cache.keys())}
        logger.info(
            "BPC-v4 InstrumentStore ready: %d instruments, window=%s..%s",
            len(self._cache),
            start,
            end,
        )

    def _load_one(self, qlib_id: str) -> Optional[_V4InstrumentSeries]:
        try:
            panel = D.features([qlib_id], OHLCVA_FIELDS, start_time=self.start, end_time=self.end, freq="day")
        except Exception as exc:
            logger.warning("跳过 %s: %s", qlib_id, exc)
            return None

        if panel is None or panel.empty:
            return None
        if "instrument" in panel.index.names:
            panel = panel.droplevel("instrument")
        panel = panel.sort_index()

        for col in OHLCVA_FIELDS:
            if col not in panel.columns:
                if col == "$amount" and self.pad_missing_amount:
                    panel[col] = 0.0
                else:
                    logger.warning("跳过 %s: 缺少字段 %s", qlib_id, col)
                    return None

        values = panel[OHLCVA_FIELDS].apply(pd.to_numeric, errors="coerce")
        close = values["$close"]
        valid = close.notna() & close.gt(0)
        values = values[valid]
        if len(values) <= 0:
            return None

        return _V4InstrumentSeries(
            qlib_id=qlib_id,
            dates=values.index,
            ohlcva=values.to_numpy(dtype=np.float32),
        )


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


def _collect_valid_indices(
    series: _V4InstrumentSeries,
    split: TemporalSplit,
    mode: Literal["train", "val"],
    seq_len: int,
) -> List[int]:
    n = len(series.dates)
    if n <= seq_len:
        return []

    t_idx = np.arange(seq_len, n, dtype=np.int64)
    anchors = series.dates[t_idx]

    if mode == "train":
        mask = anchors <= split.train_end
        window_starts = series.dates[t_idx - seq_len]
        mask &= window_starts >= split.data_start
        mask &= window_starts <= split.train_end
    else:
        mask = (anchors >= split.val_start) & (anchors <= split.val_end)
        mask &= series.dates[t_idx - seq_len] >= split.data_start

    return t_idx[mask].tolist()


def _build_split_samples(
    store: BPCV4InstrumentStore,
    split: TemporalSplit,
    mode: Literal["train", "val"],
    seq_len: int,
    *,
    max_samples_per_instrument: Optional[int] = None,
    seed: int = 42,
) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    rng = np.random.default_rng(seed)
    n_inst = len(store._cache)

    for j, (qlib_id, series) in enumerate(store._cache.items(), 1):
        valid_ts = _collect_valid_indices(series, split, mode, seq_len)
        if max_samples_per_instrument is not None and len(valid_ts) > max_samples_per_instrument:
            picked = rng.choice(valid_ts, size=max_samples_per_instrument, replace=False).tolist()
            valid_ts = sorted(picked)
        samples.extend((qlib_id, t) for t in valid_ts)
        if j % 50 == 0 or j == n_inst:
            logger.info("[%s] indexed %d/%d instruments, samples=%d", mode, j, n_inst, len(samples))

    if not samples:
        raise RuntimeError(f"BPC-v4 {mode} 集未构建任何样本，请检查日期窗口或 max_samples_per_instrument")
    return samples


def materialize_samples(
    config: GlobalConfig,
    samples: List[Tuple[str, int]],
    store: BPCV4InstrumentStore,
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
    n_total = len(samples)

    for i, (sym, t_idx) in enumerate(samples):
        arr = store._cache[sym].ohlcva
        window = arr[t_idx - seq_len : t_idx]
        prev_bar = arr[t_idx - seq_len - 1]

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
        stock_list.append(store.symbol_to_id.get(sym, 0))

        if (i + 1) % 5000 == 0 or i + 1 == n_total:
            logger.info("Materialized %d/%d samples", i + 1, n_total)

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
    seed: int = 42,
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

    provider_uri = config.qlib.provider_uri.expanduser()
    calendar = load_trading_calendar(provider_uri)
    holdout_ratio = config.qlib.val_ratio + config.qlib.test_ratio
    split = TemporalSplit.from_calendar(
        calendar,
        holdout_ratio,
        data_start=config.qlib.start_date,
        data_end=config.qlib.end_date,
    )
    logger.info(
        "Calendar window: %s .. %s (%d trading days) | train_end=%s | val=%s..%s | "
        "instruments=%d | max_samples_per_instrument=%s",
        split.data_start.date(),
        split.data_end.date(),
        len(calendar[(calendar >= split.data_start) & (calendar <= split.data_end)]),
        split.train_end.date(),
        split.val_start.date(),
        split.val_end.date(),
        len(config.qlib.instruments),
        max_samples_per_instrument,
    )

    store = BPCV4InstrumentStore(
        instruments=list(config.qlib.instruments),
        start=config.qlib.start_date,
        end=config.qlib.end_date,
        provider_uri=provider_uri,
        calendar=calendar,
        pad_missing_amount=config.kronos.amount_pad_zero,
    )
    seq_len = config.kronos.seq_len
    train_samples = _build_split_samples(
        store, split, "train", seq_len,
        max_samples_per_instrument=max_samples_per_instrument,
        seed=seed,
    )
    val_samples = _build_split_samples(
        store, split, "val", seq_len,
        max_samples_per_instrument=max_samples_per_instrument,
        seed=seed + 1,
    )

    logger.info("Materializing train=%d val=%d samples (no separate test pass)", len(train_samples), len(val_samples))

    train_ds = materialize_samples(config, train_samples, store, share_memory=share_memory)
    val_ds = materialize_samples(config, val_samples, store, share_memory=share_memory)
    test_ds = val_ds

    if save_preprocessed:
        save_preprocessed.mkdir(parents=True, exist_ok=True)
        save_materialized_dataset(train_ds, save_preprocessed / "train")
        save_materialized_dataset(val_ds, save_preprocessed / "val")
        meta = {
            "schema_version": "bpc_v4",
            "start_date": config.qlib.start_date,
            "end_date": config.qlib.end_date,
            "train_end": str(split.train_end.date()),
            "val_start": str(split.val_start.date()),
            "val_end": str(split.val_end.date()),
            "n_instruments": len(store._cache),
            "instrument_ids": list(store._cache.keys()),
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
        seed=opts.seed,
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
