"""
BPC-v3 数据集：相对化 OHLCV + vol_context + 26 维特征。

与 bpc（v2）完全隔离：不修改 bpc 模块全局状态，不依赖 bpc.ohlcv_relative。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from quant_cursor.bpc_v3.behavior_features import (
    BEHAVIOR_LABEL_SCHEMA,
    compute_behavior_proxies_stacked,
    transform_proxies_for_labeling,
)
from quant_cursor.bpc.dataset import (
    BatchedGpuDataset,
    ContiguousBatchDataset,
    GpuCachedDataset,
    QlibInstrumentStore,
    QlibMultiScaleDataset,
    TemporalSplit,
    ensure_qlib,
    load_qlib_instruments,
    load_trading_calendar,
)
from quant_cursor.bpc.model import ScaleRegistry
from quant_cursor.bpc_v3 import feature_dims as _fd_v3
from quant_cursor.bpc_v3.features import (
    compute_day_features_vectorized,
    compute_week_features_vectorized,
)
from quant_cursor.bpc_v3.ohlcv_relative import (
    OHLCV_RELATIVE_SCHEMA,
    CrossSectionDeltaMedians,
    _bar_ordinals_for_window,
    absolute_window_to_relative,
)
from quant_cursor.bpc_v3.volatility_context import VolatilityStats

logger = logging.getLogger(__name__)

BPC_SCHEMA_VERSION = "bpc_v3"
DAY_FULL_FEAT_DIM = _fd_v3.DAY_FULL_FEAT_DIM
NUM_BEHAVIOR_AGENTS = _fd_v3.DAY_BEHAVIOR_FEAT_DIM
PROXY_CHUNK = 16_384


class QlibMultiScaleDatasetV3(QlibMultiScaleDataset):
    """采样时输出相对化 OHLCV 窗口与 prev_bar（v3 专用）。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cs_medians = CrossSectionDeltaMedians.from_store(self.store, self.date_to_ordinal)

    @staticmethod
    def _prev_bar(arr: np.ndarray, end_idx: int, lookback: int) -> np.ndarray:
        start = end_idx - lookback + 1
        if start > 0:
            return arr[start - 1]
        return arr[0]

    def _relative_day_window(self, series, t: int, lookback: int) -> tuple[np.ndarray, np.ndarray]:
        abs_win = self._window(series.ohlcv, t, lookback)
        prev_bar = self._prev_bar(series.ohlcv, t, lookback)
        bar_ords = _bar_ordinals_for_window(series.dates, self.date_to_ordinal, t, lookback)
        rel = absolute_window_to_relative(abs_win, prev_bar, bar_ords, self._cs_medians.day)
        return rel, prev_bar

    def _relative_week_window(self, series, w_idx: int, lookback: int) -> tuple[np.ndarray, np.ndarray]:
        abs_win = self._window(series.weekly, w_idx, lookback)
        prev_bar = self._prev_bar(series.weekly, w_idx, lookback)
        dates = series.weekly_end_dates
        bar_ords = _bar_ordinals_for_window(dates, self.date_to_ordinal, w_idx, lookback)
        rel = absolute_window_to_relative(abs_win, prev_bar, bar_ords, self._cs_medians.week)
        return rel, prev_bar

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        qlib_id, t = self._samples[idx]
        series = self.store._cache[qlib_id]
        sample: dict[str, torch.Tensor] = {}

        for cfg in self.scales:
            if cfg.name == "day":
                win, prev_bar = self._relative_day_window(series, t, cfg.lookback_window)
                sample["day_prev_bar"] = torch.from_numpy(prev_bar.astype(np.float32))
            elif cfg.name == "week":
                w_idx = int(series.daily_week_idx[t])
                win, prev_bar = self._relative_week_window(series, w_idx, cfg.lookback_window)
                sample["week_prev_bar"] = torch.from_numpy(prev_bar.astype(np.float32))
            else:
                raise NotImplementedError(f"尺度 '{cfg.name}' 尚未实现")
            sample[cfg.name] = torch.from_numpy(win.astype(np.float32))

        sid = self.symbol_to_id.get(qlib_id, 0)
        sample["stock_ids"] = torch.tensor(sid, dtype=torch.long)
        ts_val = self.date_to_ordinal.get(series.dates[t], 0)
        sample["timestamps"] = torch.tensor(ts_val, dtype=torch.long)
        return sample


class MaterializedMultiScaleDatasetV3(Dataset):
    """物化相对 OHLCV、vol_context、26 维特征与原始 behavior_proxies。"""

    def __init__(
        self,
        base: QlibMultiScaleDatasetV3,
        *,
        share_memory: bool = True,
        precompute_features: bool = True,
        precompute_proxies: bool = True,
    ):
        self.scales = base.scales
        self.primary_scale = "day"
        self._vol_stats = VolatilityStats.from_store(base.store)

        n = len(base)
        logger.info("Materializing v3 %d samples (relative OHLCV) ...", n)

        scale_arrays: dict[str, np.ndarray] = {}
        self._prev_bars: dict[str, np.ndarray] = {}
        for cfg in base.scales:
            lb = cfg.lookback_window
            scale_arrays[cfg.name] = np.empty((n, lb, 5), dtype=np.float32)
            self._prev_bars[cfg.name] = np.empty((n, 5), dtype=np.float32)

        for i, (qlib_id, t) in enumerate(base._samples):
            series = base.store._cache[qlib_id]
            for cfg in base.scales:
                if cfg.name == "day":
                    rel, prev = base._relative_day_window(series, t, cfg.lookback_window)
                    scale_arrays[cfg.name][i] = rel
                    self._prev_bars[cfg.name][i] = prev
                elif cfg.name == "week":
                    w_idx = int(series.daily_week_idx[t])
                    rel, prev = base._relative_week_window(series, w_idx, cfg.lookback_window)
                    scale_arrays[cfg.name][i] = rel
                    self._prev_bars[cfg.name][i] = prev
            if (i + 1) % 500_000 == 0 or i + 1 == n:
                logger.info("Materialized relative windows %d/%d", i + 1, n)

        self._tensors: dict[str, torch.Tensor] = {}
        for name, arr in scale_arrays.items():
            t = torch.from_numpy(arr)
            if share_memory:
                t = t.share_memory_()
            self._tensors[name] = t

        self._prev_bar_tensors: dict[str, torch.Tensor] = {}
        for name, arr in self._prev_bars.items():
            t = torch.from_numpy(arr.astype(np.float32))
            if share_memory:
                t = t.share_memory_()
            self._prev_bar_tensors[name] = t

        stock_ids_np = np.zeros(n, dtype=np.int64)
        timestamps_np = np.zeros(n, dtype=np.int64)
        for i, (qlib_id, t_idx) in enumerate(base._samples):
            stock_ids_np[i] = base.symbol_to_id.get(qlib_id, 0)
            series = base.store._cache[qlib_id]
            timestamps_np[i] = base.date_to_ordinal.get(series.dates[t_idx], 0)

        self._stock_ids = torch.from_numpy(stock_ids_np)
        self._timestamps = torch.from_numpy(timestamps_np)
        if share_memory:
            self._stock_ids = self._stock_ids.share_memory_()
            self._timestamps = self._timestamps.share_memory_()

        self._features: dict[str, torch.Tensor] = {}
        self._vol_context: torch.Tensor | None = None
        self._proxies: torch.Tensor | None = None
        self._proxies_label_ready = False

        if precompute_features or precompute_proxies:
            self._materialize_derived(share_memory, precompute_features, precompute_proxies)

    def _materialize_derived(
        self,
        share_memory: bool,
        precompute_features: bool,
        precompute_proxies: bool,
    ) -> None:
        if "day" not in self._tensors:
            return
        n = len(self)
        day = self._tensors["day"]
        day_prev = self._prev_bars.get("day")

        ctx_chunks: list[torch.Tensor] = []
        for start in range(0, n, PROXY_CHUNK):
            end = min(start + PROXY_CHUNK, n)
            ctx_chunks.append(
                self._vol_stats.lookup_at_anchor(self._stock_ids[start:end], self._timestamps[start:end])
            )
        vol_context = torch.cat(ctx_chunks, dim=0)
        if share_memory:
            vol_context = vol_context.share_memory_()
        self._vol_context = vol_context

        if precompute_features:
            logger.info("Precomputing %d-dim day features (%d samples) ...", DAY_FULL_FEAT_DIM, n)
            feat_chunks: list[torch.Tensor] = []
            for start in range(0, n, PROXY_CHUNK):
                end = min(start + PROXY_CHUNK, n)
                chunk = day[start:end].float()
                ctx = vol_context[start:end]
                prev_chunk = torch.from_numpy(day_prev[start:end]).float() if day_prev is not None else None
                if torch.cuda.is_available():
                    chunk = chunk.cuda()
                    ctx = ctx.cuda()
                    if prev_chunk is not None:
                        prev_chunk = prev_chunk.cuda()
                feat_chunks.append(compute_day_features_vectorized(chunk, ctx, prev_chunk).cpu())
            features = torch.cat(feat_chunks, dim=0)
            if share_memory:
                features = features.share_memory_()
            self._features["day_features"] = features

            if "week" in self._tensors:
                logger.info("Precomputing week features (%d samples) ...", n)
                week_chunks = []
                for start in range(0, n, PROXY_CHUNK):
                    end = min(start + PROXY_CHUNK, n)
                    chunk = self._tensors["week"][start:end].float()
                    ctx = vol_context[start:end]
                    if torch.cuda.is_available():
                        chunk = chunk.cuda()
                        ctx = ctx.cuda()
                    week_chunks.append(
                        compute_week_features_vectorized(chunk, vol_context=ctx).cpu()
                    )
                week_feats = torch.cat(week_chunks, dim=0)
                if share_memory:
                    week_feats = week_feats.share_memory_()
                self._features["week_features"] = week_feats

        if precompute_proxies:
            logger.info("Precomputing behavior proxies (%d samples) ...", n)
            proxy_chunks = []
            for start in range(0, n, PROXY_CHUNK):
                end = min(start + PROXY_CHUNK, n)
                chunk = day[start:end].float()
                ctx = vol_context[start:end]
                prev_chunk = torch.from_numpy(day_prev[start:end]).float() if day_prev is not None else None
                if torch.cuda.is_available():
                    chunk = chunk.cuda()
                    ctx = ctx.cuda()
                    if prev_chunk is not None:
                        prev_chunk = prev_chunk.cuda()
                proxy_chunks.append(compute_behavior_proxies_stacked(chunk, ctx, prev_chunk).cpu())
            proxies = torch.cat(proxy_chunks, dim=0)
            if share_memory:
                proxies = proxies.share_memory_()
            self._proxies = proxies

        logger.info(
            "Materialized v3 ready (vol_context=%s, day_feat=%s, proxies=%s)",
            self._vol_context is not None,
            "day_features" in self._features,
            self._proxies is not None,
        )

    def _sample_count(self) -> int:
        if "day_features" in self._features:
            return int(self._features["day_features"].shape[0])
        if self._tensors:
            return int(next(iter(self._tensors.values())).shape[0])
        return int(self._stock_ids.shape[0])

    def __len__(self) -> int:
        return self._sample_count()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample: dict[str, torch.Tensor] = {}
        if "day_features" in self._features:
            sample["day_features"] = self._features["day_features"][idx]
        else:
            sample["day"] = self._tensors["day"][idx]
        if "week_features" in self._features:
            sample["week_features"] = self._features["week_features"][idx]
        elif "week" in self._tensors:
            sample["week"] = self._tensors["week"][idx]
        for name, t in self._prev_bar_tensors.items():
            sample[f"{name}_prev_bar"] = t[idx]
        sample["stock_ids"] = self._stock_ids[idx]
        sample["timestamps"] = self._timestamps[idx]
        if self._vol_context is not None:
            sample["vol_context"] = self._vol_context[idx]
        if self._proxies is not None:
            sample["behavior_proxies"] = self._proxies[idx]
        return sample


MaterializedMultiScaleDataset = MaterializedMultiScaleDatasetV3


def estimate_materialized_bytes(ds: MaterializedMultiScaleDatasetV3) -> int:
    """估算物化数据集 RAM 占用（字节）。"""
    total = 0
    for t in ds._features.values():
        total += t.numel() * t.element_size()
    for t in ds._tensors.values():
        total += t.numel() * t.element_size()
    for t in ds._prev_bar_tensors.values():
        total += t.numel() * t.element_size()
    if ds._vol_context is not None:
        total += ds._vol_context.numel() * ds._vol_context.element_size()
    if ds._proxies is not None:
        total += ds._proxies.numel() * ds._proxies.element_size()
    total += ds._stock_ids.numel() * ds._stock_ids.element_size()
    total += ds._timestamps.numel() * ds._timestamps.element_size()
    return total


def assert_v3_training_cache_complete(ds: MaterializedMultiScaleDatasetV3) -> None:
    """确认训练无需实时重算特征/代理（需 precompute + 磁盘缓存完整）。"""
    missing: list[str] = []
    if "day_features" not in ds._features:
        missing.append("day_features")
    if "week_features" not in ds._features:
        missing.append("week_features")
    if ds._proxies is None:
        missing.append("behavior_proxies")
    if ds._vol_context is None:
        missing.append("vol_context")
    if missing:
        raise RuntimeError(
            "训练缓存不完整，缺少: "
            + ", ".join(missing)
            + "。请用 --save-preprocessed 且勿加 --no-precompute-features；"
            "或 --force-rebuild-preprocessed 重建。"
        )


def promote_dataset_ram_resident(
    ds: MaterializedMultiScaleDatasetV3,
    *,
    drop_raw_ohlcv: bool = True,
) -> MaterializedMultiScaleDatasetV3:
    """
    将物化张量固化为 contiguous RAM 副本；可选丢弃已冗余的 raw OHLCV 窗口。

    配合 --ram-resident 使用：消除 mmap/共享内存歧义，每 epoch 无磁盘 IO。
    """

    def _ram_copy(t: torch.Tensor) -> torch.Tensor:
        return t.contiguous().clone()

    for key in list(ds._features.keys()):
        ds._features[key] = _ram_copy(ds._features[key])
    if ds._vol_context is not None:
        ds._vol_context = _ram_copy(ds._vol_context)
    if ds._proxies is not None:
        ds._proxies = _ram_copy(ds._proxies)
    ds._stock_ids = _ram_copy(ds._stock_ids)
    ds._timestamps = _ram_copy(ds._timestamps)
    for name in list(ds._prev_bar_tensors.keys()):
        ds._prev_bar_tensors[name] = _ram_copy(ds._prev_bar_tensors[name])

    if drop_raw_ohlcv and "day_features" in ds._features and ds._tensors:
        raw_bytes = sum(t.numel() * t.element_size() for t in ds._tensors.values())
        ds._tensors.clear()
        ds._prev_bars.clear()
        ds._prev_bar_tensors.clear()
        logger.info(
            "ram-resident: dropped raw OHLCV windows (freed %.2f GiB)",
            raw_bytes / (1024**3),
        )
    elif ds._tensors:
        for name in list(ds._tensors.keys()):
            ds._tensors[name] = _ram_copy(ds._tensors[name])

    nbytes = estimate_materialized_bytes(ds)
    logger.info(
        "ram-resident: %d samples, %.2f GiB contiguous RAM",
        len(ds),
        nbytes / (1024**3),
    )
    return ds


def pin_dataset_share_memory(ds: MaterializedMultiScaleDatasetV3) -> None:
    def _pin(t: torch.Tensor) -> torch.Tensor:
        return t if t.is_shared() else t.share_memory_()

    for name, t in ds._tensors.items():
        ds._tensors[name] = _pin(t)
    for key, t in ds._features.items():
        ds._features[key] = _pin(t)
    ds._stock_ids = _pin(ds._stock_ids)
    ds._timestamps = _pin(ds._timestamps)
    if ds._proxies is not None:
        ds._proxies = _pin(ds._proxies)
    if ds._vol_context is not None:
        ds._vol_context = _pin(ds._vol_context)
    for name in ds._prev_bar_tensors:
        ds._prev_bar_tensors[name] = _pin(ds._prev_bar_tensors[name])


def _augment_meta_json(path: Path) -> None:
    meta_path = path / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["schema_version"] = BPC_SCHEMA_VERSION
    meta["day_feature_dim"] = _fd_v3.DAY_FULL_FEAT_DIM
    meta["day_struct_feat_dim"] = _fd_v3.DAY_STRUCT_FEAT_DIM
    meta["num_behavior_agents"] = NUM_BEHAVIOR_AGENTS
    meta["behavior_label_schema"] = BEHAVIOR_LABEL_SCHEMA
    meta["ohlcv_relative_schema"] = OHLCV_RELATIVE_SCHEMA
    meta["feature_scale_schema"] = _fd_v3.FEATURE_SCALE_SCHEMA
    meta["proxies_label_ready"] = False
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def save_materialized_dataset(ds: MaterializedMultiScaleDatasetV3, path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    meta = {
        "scales": [(c.name, c.freq, c.lookback_window) for c in ds.scales],
        "primary_scale": ds.primary_scale,
        "has_proxies": ds._proxies is not None,
        "has_day_features": "day_features" in ds._features,
        "has_week_features": "week_features" in ds._features,
        "has_features": "day_features" in ds._features,
        "proxies_label_ready": False,
        "n_samples": len(ds),
        "day_feature_dim": DAY_FULL_FEAT_DIM,
        "num_behavior_agents": NUM_BEHAVIOR_AGENTS,
        "schema_version": BPC_SCHEMA_VERSION,
        "behavior_label_schema": BEHAVIOR_LABEL_SCHEMA,
        "ohlcv_relative_schema": OHLCV_RELATIVE_SCHEMA,
        "feature_scale_schema": _fd_v3.FEATURE_SCALE_SCHEMA,
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _safe_save(t: torch.Tensor, stem: str) -> None:
        np.save(path / f"{stem}.npy", t.detach().cpu().numpy())

    for name, t in ds._tensors.items():
        _safe_save(t, f"{name}_raw")
    if "day_features" in ds._features:
        _safe_save(ds._features["day_features"], "day_features")
    if "week_features" in ds._features:
        _safe_save(ds._features["week_features"], "week_features")
    _safe_save(ds._stock_ids, "stock_ids")
    _safe_save(ds._timestamps, "timestamps")
    if ds._proxies is not None:
        _safe_save(ds._proxies, "behavior_proxies")
    if ds._vol_context is not None:
        np.save(path / "vol_context.npy", ds._vol_context.detach().cpu().numpy())
    logger.info("Saved v3 materialized dataset to %s (%d samples)", path, len(ds))


def load_materialized_dataset(path: str | Path) -> MaterializedMultiScaleDatasetV3:
    path = Path(path)
    meta_path = path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"预处理目录缺少 meta.json: {path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    day_npy = path / "day_features.npy"
    if day_npy.exists():
        actual_dim = int(np.load(day_npy, mmap_mode="r").shape[1])
        if actual_dim != DAY_FULL_FEAT_DIM:
            raise ValueError(
                f"{path}: day_features dim {actual_dim} != v3 expected {DAY_FULL_FEAT_DIM}; "
                "re-run bpc_v3.train --force-rebuild-preprocessed"
            )

    ds = MaterializedMultiScaleDatasetV3.__new__(MaterializedMultiScaleDatasetV3)
    ds.scales = []
    ds.primary_scale = meta.get("primary_scale", "day")
    ds._vol_stats = None
    ds._proxies_label_ready = False

    def _load(stem: str) -> torch.Tensor:
        p = path / f"{stem}.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing {stem}.npy in {path}")
        return torch.from_numpy(np.load(p))

    ds._tensors = {}
    for name, _, _ in meta.get("scales", []):
        try:
            ds._tensors[name] = _load(f"{name}_raw")
        except FileNotFoundError:
            pass

    ds._features = {}
    if meta.get("has_day_features") or meta.get("has_features"):
        ds._features["day_features"] = _load("day_features")
    if meta.get("has_week_features"):
        try:
            ds._features["week_features"] = _load("week_features")
        except FileNotFoundError:
            logger.warning("week_features missing in %s", path)

    ds._stock_ids = _load("stock_ids")
    ds._timestamps = _load("timestamps")
    ds._proxies = _load("behavior_proxies") if meta.get("has_proxies") else None
    vol_path = path / "vol_context.npy"
    if not vol_path.exists() and meta.get("schema_version") == BPC_SCHEMA_VERSION:
        raise FileNotFoundError(f"{path}: missing vol_context.npy for v3 cache")
    ds._vol_context = torch.from_numpy(np.load(vol_path)) if vol_path.exists() else None

    ds._prev_bars = {}
    ds._prev_bar_tensors = {}
    logger.info("Loaded v3 materialized dataset from %s (%d samples)", path, len(ds))
    return ds


def build_datasets(
    instruments: list[str],
    start: str,
    end: str,
    provider_uri: str | Path,
    registry: ScaleRegistry,
    **kwargs,
) -> tuple[QlibInstrumentStore, Dataset, Dataset, TemporalSplit]:
    """v3 构建：相对化采样 + 物化 vol_context。"""
    calendar = load_trading_calendar(provider_uri)
    val_ratio = kwargs.pop("val_ratio", 0.20)
    max_samples = kwargs.pop("max_samples_per_instrument", None)
    seed = kwargs.pop("seed", 42)
    materialize = kwargs.pop("materialize", True)
    share_memory = kwargs.pop("share_memory", True)
    precompute_features = kwargs.pop("precompute_features", True)
    precompute_proxies = kwargs.pop("precompute_proxies", True)
    split = TemporalSplit.from_calendar(calendar, val_ratio, data_start=start, data_end=end)

    store = QlibInstrumentStore(
        instruments=instruments,
        start=start,
        end=end,
        provider_uri=provider_uri,
        registry=registry,
        calendar=calendar,
    )
    train_base = QlibMultiScaleDatasetV3(
        store, split, mode="train", max_samples_per_instrument=max_samples, seed=seed
    )
    val_base = QlibMultiScaleDatasetV3(
        store, split, mode="val", max_samples_per_instrument=max_samples, seed=seed + 1
    )
    if materialize:
        train_ds: Dataset = MaterializedMultiScaleDatasetV3(
            train_base,
            share_memory=share_memory,
            precompute_features=precompute_features,
            precompute_proxies=precompute_proxies,
        )
        val_ds: Dataset = MaterializedMultiScaleDatasetV3(
            val_base,
            share_memory=share_memory,
            precompute_features=precompute_features,
            precompute_proxies=precompute_proxies,
        )
    else:
        train_ds = train_base
        val_ds = val_base
    return store, train_ds, val_ds, split


__all__ = [
    "BPC_SCHEMA_VERSION",
    "BEHAVIOR_LABEL_SCHEMA",
    "DAY_FULL_FEAT_DIM",
    "MaterializedMultiScaleDataset",
    "MaterializedMultiScaleDatasetV3",
    "QlibMultiScaleDatasetV3",
    "build_datasets",
    "load_materialized_dataset",
    "save_materialized_dataset",
    "pin_dataset_share_memory",
    "promote_dataset_ram_resident",
    "assert_v3_training_cache_complete",
    "estimate_materialized_bytes",
    "BatchedGpuDataset",
    "ContiguousBatchDataset",
    "GpuCachedDataset",
    "TemporalSplit",
    "ensure_qlib",
    "load_qlib_instruments",
    "load_trading_calendar",
]
