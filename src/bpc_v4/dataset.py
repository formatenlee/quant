"""bpc_v4 Qlib 数据集：逐标的加载 + 日历切分 + 预处理缓存 + DataLoader 策略。"""

from __future__ import annotations

import json
import logging
import gc
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
from .cpu_parallel import (
    DEFAULT_CPU_THREADS,
    can_use_process_pool,
    materialize_chunk_size,
    raise_nofile_soft_limit,
    resolve_cpu_threads,
    run_parallel_fork_chunks,
    run_parallel_thread_chunks,
)
from .features_materialize import compute_chunk_features_numpy, relative_windows_batch
from .materialize_bundle import MaterializeSpawnBundle
from .ohlcv_relative import CrossSectionDeltaMedians, NUM_OHLCV_FIELDS
from .kronos_cache import (
    KronosLookup,
    KronosPrecomputeStore,
    LiveKronosLookup,
    build_live_kronos_lookup,
    freeze_kronos_cache_for_spawn,
    freeze_live_lookup_for_samples,
    release_live_kronos_encoders,
)
from .diagnostics_v4 import audit_s1_token_diversity
from .volatility_context import VolatilityStats
from .materialize import (
    BPC_V4_SCHEMA,
    BatchedGpuBPCV4Dataset,
    ContiguousBatchBPCV4Dataset,
    GpuCachedBPCV4Dataset,
    MaterializedBPCV4Dataset,
    dataset_memory_bytes,
    load_materialized_dataset,
    log_dataset_memory,
    pin_dataset_share_memory,
    save_materialized_dataset,
)

logger = logging.getLogger(__name__)

OHLCVA_FIELDS = ["$open", "$high", "$low", "$close", "$volume", "$amount"]

_MATERIALIZE_CTX: dict | None = None


def _sanitize_np(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)


def _kronos_get_numpy(lookup: KronosLookup, sym: str, t_idx: int) -> tuple[np.ndarray, np.ndarray]:
    z_q, s1_ids = lookup.get(sym, t_idx)
    if isinstance(z_q, torch.Tensor):
        return z_q.detach().cpu().numpy().astype(np.float32, copy=False), s1_ids.detach().cpu().numpy()
    return np.asarray(z_q, dtype=np.float32), np.asarray(s1_ids)


def _compute_next_day_sign(arr: np.ndarray, t_idx: int) -> float:
    """锚点日 close → 次日 close 涨跌方向（+1/-1/0），用于方向预测监控。"""
    if t_idx >= len(arr):
        return 0.0
    c_prev = float(arr[t_idx - 1, 3])
    c_next = float(arr[t_idx, 3])
    if not (c_prev > 0 and c_next > 0):
        return 0.0
    delta = c_next - c_prev
    if delta > 0:
        return 1.0
    if delta < 0:
        return -1.0
    return 0.0


def _materialize_chunk(chunk_start: int) -> list[tuple[int, tuple]]:
    """
    分片物化（纯 NumPy 收集 + 分片内批量特征；torch 仅在 features_materialize 数学核内）。
    """
    ctx = _MATERIALIZE_CTX
    if ctx is None:
        raise RuntimeError("物化上下文未初始化")
    samples: list[tuple[str, int]] = ctx["samples"]
    bundle: MaterializeSpawnBundle = ctx["bundle"]
    chunk_size: int = ctx["chunk_size"]
    cs_medians: np.ndarray = ctx["cs_medians"]
    end = min(chunk_start + chunk_size, len(samples))
    B = end - chunk_start
    if B <= 0:
        return []

    seq_len = bundle.seq_len
    abs_windows = np.empty((B, seq_len, NUM_OHLCV_FIELDS), dtype=np.float32)
    prev_bars = np.empty((B, NUM_OHLCV_FIELDS), dtype=np.float32)
    bar_ords = np.empty((B, seq_len), dtype=np.int64)
    stock_ids = np.empty(B, dtype=np.int64)
    cal_ords = np.empty(B, dtype=np.int64)
    next_signs = np.zeros(B, dtype=np.float32)
    z_q_buf: list[np.ndarray] = []
    s1_buf: list[np.ndarray] = []

    for j, i in enumerate(range(chunk_start, end)):
        sym, t_idx = samples[i]
        inst = bundle.instruments[sym]
        arr = inst.ohlcva
        abs_windows[j] = arr[t_idx - seq_len : t_idx, :NUM_OHLCV_FIELDS]
        prev_bars[j] = arr[t_idx - seq_len - 1, :NUM_OHLCV_FIELDS]
        bar_ords[j] = inst.bar_ordinals[t_idx - seq_len : t_idx]
        stock_ids[j] = bundle.symbol_to_id.get(sym, 0)
        cal_ords[j] = int(inst.bar_ordinals[t_idx - 1])
        next_signs[j] = _compute_next_day_sign(arr, t_idx)
        z_np, s1_np = _kronos_get_numpy(bundle.kronos_lookup, sym, t_idx)
        z_q_buf.append(z_np)
        s1_buf.append(s1_np)

    rel_ohlcv = relative_windows_batch(abs_windows, prev_bars, bar_ords, cs_medians)
    bpc_feat, ctx_feat, purity_tgt, time_emb = compute_chunk_features_numpy(
        rel_ohlcv,
        prev_bars,
        stock_ids,
        cal_ords,
        bundle.vol_stats,
        label_temperature=bundle.config.bpc.label_temperature,
        time_raw_dim=bundle.config.embedding.time_raw_dim,
    )
    z_q_batch = _sanitize_np(np.stack(z_q_buf, axis=0))
    s1_batch = np.stack(s1_buf, axis=0)

    rows: list[tuple[int, tuple]] = []
    for j, i in enumerate(range(chunk_start, end)):
        rows.append(
            (
                i,
                (
                    z_q_batch[j],
                    bpc_feat[j],
                    ctx_feat[j],
                    time_emb[j],
                    int(stock_ids[j]),
                    s1_batch[j],
                    purity_tgt[j],
                    np.float32(next_signs[j]),
                ),
            )
        )
    return rows


def build_materialize_spawn_bundle(
    store: BPCV4InstrumentStore,
    *,
    cs_medians: CrossSectionDeltaMedians,
    vol_stats: VolatilityStats,
    kronos_lookup: KronosLookup,
    config: GlobalConfig,
    seq_len: int,
) -> MaterializeSpawnBundle:
    from .materialize_bundle import FrozenInstrument

    instruments: dict[str, FrozenInstrument] = {}
    for sym, series in store._cache.items():
        bar_ordinals = np.array(
            [store.date_to_ordinal.get(series.dates[t], 0) for t in range(len(series.dates))],
            dtype=np.int64,
        )
        instruments[sym] = FrozenInstrument(
            ohlcva=np.ascontiguousarray(series.ohlcva),
            bar_ordinals=bar_ordinals,
        )
    return MaterializeSpawnBundle(
        instruments=instruments,
        symbol_to_id=dict(store.symbol_to_id),
        cs_medians_day=np.ascontiguousarray(cs_medians.day),
        vol_stats=vol_stats,
        kronos_lookup=kronos_lookup,
        config=config,
        seq_len=seq_len,
    )


def _run_materialize_parallel(
    *,
    bundle: MaterializeSpawnBundle,
    samples: list[tuple[str, int]],
    chunk_size: int,
    workers: int,
    n_total: int,
    use_fork: bool = False,
) -> list[tuple]:
    """
    qlib/Kronos 已在主进程载入 RAM；此处仅按样本下标分片并行组装 BPC 张量。

    默认线程池（共享 bundle）；Linux 可选 fork 绕过 GIL（--materialize-fork）。
    """
    global _MATERIALIZE_CTX
    _MATERIALIZE_CTX = {
        "samples": samples,
        "chunk_size": chunk_size,
        "bundle": bundle,
        "cs_medians": bundle.cs_medians_day,
    }

    parallel_kwargs = {
        "chunk_fn": _materialize_chunk,
        "n_items": n_total,
        "chunk_size": chunk_size,
        "num_workers": workers,
        "desc": "BPC materialize",
        "progress_log_every": 5000,
    }

    if use_fork and workers > 1 and can_use_process_pool():
        raise_nofile_soft_limit()
        logger.info("物化并行: Linux fork 进程池（绕过 GIL，需足够 RLIMIT_NOFILE）")
        ordered = run_parallel_fork_chunks(**parallel_kwargs)
    else:
        if use_fork and workers > 1:
            logger.warning("--materialize-fork 不可用（非 Linux fork），回退线程池")
        ordered = run_parallel_thread_chunks(**parallel_kwargs)
    return ordered


def _ordered_rows_to_dataset(ordered: list, n_total: int) -> MaterializedBPCV4Dataset:
    """物化 NumPy 结果 → 训练用 torch Dataset（仅此一步接触 torch）。"""
    if n_total != len(ordered):
        raise RuntimeError(f"物化结果不完整: {len(ordered)}/{n_total}")
    z_q = np.stack([ordered[i][0] for i in range(n_total)], axis=0)
    bpc = np.stack([ordered[i][1] for i in range(n_total)], axis=0)
    ctx = np.stack([ordered[i][2] for i in range(n_total)], axis=0)
    time_emb = np.stack([ordered[i][3] for i in range(n_total)], axis=0)
    stock = np.array([ordered[i][4] for i in range(n_total)], dtype=np.int64)
    s1 = np.stack([ordered[i][5] for i in range(n_total)], axis=0)
    purity = np.stack([ordered[i][6] for i in range(n_total)], axis=0)
    sign = np.array([ordered[i][7] for i in range(n_total)], dtype=np.float32)
    return MaterializedBPCV4Dataset(
        z_q=torch.from_numpy(_sanitize_np(z_q)),
        bpc_feat=torch.from_numpy(_sanitize_np(bpc)),
        ctx_feat=torch.from_numpy(_sanitize_np(ctx)),
        time_emb=torch.from_numpy(time_emb.astype(np.float32, copy=False)),
        stock_id=torch.from_numpy(stock),
        s1_ids=torch.from_numpy(s1.astype(np.int64, copy=False)),
        purity_target=torch.from_numpy(purity.astype(np.float32, copy=False)),
        next_day_sign=torch.from_numpy(sign),
    )


@dataclass
class _V4InstrumentSeries:
    qlib_id: str
    dates: pd.DatetimeIndex
    ohlcva: np.ndarray


class BPCV4InstrumentStore:
    """从 qlib 批量加载 OHLCVA panel，再按标的切分为内存序列。"""

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
        self._load_batch(instruments)

        if not self._cache:
            raise RuntimeError("BPC-v4 InstrumentStore: 无有效标的")

        self.symbol_to_id: Dict[str, int] = {qlib_id: i for i, qlib_id in enumerate(self._cache.keys())}
        self.date_to_ordinal: Dict[pd.Timestamp, int] = {d: i for i, d in enumerate(self.calendar)}
        logger.info(
            "BPC-v4 InstrumentStore ready: %d instruments, window=%s..%s",
            len(self._cache),
            start,
            end,
        )

    def _load_batch(self, instruments: List[str]) -> None:
        total = len(instruments)
        if total == 0:
            return
        logger.info("BPC-v4 InstrumentStore: batch D.features %d instruments ...", total)
        try:
            panel = D.features(instruments, OHLCVA_FIELDS, start_time=self.start, end_time=self.end, freq="day")
        except Exception as exc:
            logger.warning("batch D.features 失败 (%s)，回退逐标的加载", exc)
            for i, qlib_id in enumerate(instruments, 1):
                series = self._load_one(qlib_id)
                if series is not None:
                    self._cache[qlib_id] = series
                if i % 50 == 0 or i == total:
                    logger.info("BPC-v4 InstrumentStore: loaded %d/%d (cached=%d)", i, total, len(self._cache))
            return

        if panel is None or panel.empty:
            logger.warning("batch D.features 返回空 panel")
            return

        loaded = 0
        for qlib_id in instruments:
            sub = self._extract_instrument_panel(panel, qlib_id)
            if sub is None:
                continue
            series = self._panel_to_series(qlib_id, sub)
            if series is not None:
                self._cache[qlib_id] = series
                loaded += 1
        logger.info("BPC-v4 InstrumentStore: batch loaded %d/%d instruments", loaded, total)

    def _extract_instrument_panel(self, panel: pd.DataFrame, qlib_id: str) -> pd.DataFrame | None:
        if not isinstance(panel.index, pd.MultiIndex):
            if len(panel.columns) and qlib_id in str(panel.columns):
                return panel
            return None
        names = list(panel.index.names)
        try:
            if "instrument" in names:
                return panel.xs(qlib_id, level="instrument")
            return panel.loc[(slice(None), qlib_id), :]
        except (KeyError, TypeError):
            try:
                return panel.loc[qlib_id]
            except KeyError:
                return None

    def _panel_to_series(self, qlib_id: str, panel: pd.DataFrame) -> Optional[_V4InstrumentSeries]:
        if panel is None or panel.empty:
            return None
        if "instrument" in panel.index.names:
            panel = panel.droplevel("instrument")
        panel = panel.sort_index()

        for col in OHLCVA_FIELDS:
            if col not in panel.columns:
                if col == "$amount" and self.pad_missing_amount:
                    panel = panel.copy()
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

        arr = values.to_numpy(dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if self.pad_missing_amount:
            amt = arr[:, 5]
            vol = arr[:, 4]
            price_mean = arr[:, :4].mean(axis=1)
            missing_amt = np.abs(amt) < 1e-12
            arr[missing_amt, 5] = vol[missing_amt] * price_mean[missing_amt]

        return _V4InstrumentSeries(
            qlib_id=qlib_id,
            dates=values.index,
            ohlcva=arr,
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
        return self._panel_to_series(qlib_id, panel.sort_index())


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


def _finalize_cpu_dataset(
    ds: MaterializedBPCV4Dataset,
    *,
    label: str,
    share_memory: bool,
) -> MaterializedBPCV4Dataset:
    ds.make_contiguous()
    log_dataset_memory(ds, label=label)
    if share_memory:
        pin_dataset_share_memory(ds)
    return ds


def materialize_samples(
    config: GlobalConfig,
    samples: List[Tuple[str, int]],
    store: BPCV4InstrumentStore,
    *,
    cs_medians: CrossSectionDeltaMedians,
    vol_stats: VolatilityStats,
    share_memory: bool = False,
    kronos_lookup: KronosLookup,
    cpu_threads: int = DEFAULT_CPU_THREADS,
    materialize_fork: bool = False,
) -> MaterializedBPCV4Dataset:
    """
    将样本列表物化为 CPU 驻留张量（训练前一次性完成，训练阶段只读内存）。

    Kronos z_q/s1_ids 须已由 kronos_lookup 预计算；本函数仅组装 BPC 特征。
    """
    seq_len = config.kronos.seq_len
    workers = resolve_cpu_threads(cpu_threads)

    bundle = build_materialize_spawn_bundle(
        store,
        cs_medians=cs_medians,
        vol_stats=vol_stats,
        kronos_lookup=kronos_lookup,
        config=config,
        seq_len=seq_len,
    )
    n_total = len(samples)
    chunk_size = materialize_chunk_size(n_total, workers) if workers > 1 else 1
    backend = "threads" if workers > 1 else "sequential"
    logger.info(
        "物化 BPC→CPU 张量: %d samples, 内存 bundle 已就绪, %d workers (%s, 分片并行)",
        n_total,
        workers,
        backend,
    )

    rows = _run_materialize_parallel(
        bundle=bundle,
        samples=samples,
        chunk_size=chunk_size,
        workers=workers,
        n_total=n_total,
        use_fork=materialize_fork,
    )

    ds = _ordered_rows_to_dataset(rows, n_total)
    audit_s1_token_diversity(
        ds,
        vocab_size=config.head.codebook_output_dim,
        n_sample=min(5000, len(ds)),
        fail_if_degenerate=True,
    )
    return ds


def load_preprocessed_datasets(
    preprocessed_dir: Path,
    *,
    share_memory: bool = False,
) -> Tuple[MaterializedBPCV4Dataset, MaterializedBPCV4Dataset, MaterializedBPCV4Dataset]:
    """从磁盘加载已物化的 train/val（训练阶段专用，不做任何特征计算）。"""
    root = Path(preprocessed_dir)
    train_p = root / "train"
    val_p = root / "val"
    test_p = root / "test"
    if not train_p.is_dir() or not val_p.is_dir():
        raise FileNotFoundError(
            f"预处理目录缺少 train/ 或 val/: {root}。"
            "请先 --preprocess-only --save-preprocessed <dir>。"
        )
    logger.info("加载已物化数据集: %s → CPU RAM", root)
    train_ds = _finalize_cpu_dataset(
        load_materialized_dataset(train_p), label="train(cached)", share_memory=share_memory
    )
    val_ds = _finalize_cpu_dataset(
        load_materialized_dataset(val_p), label="val(cached)", share_memory=share_memory
    )
    test_ds = load_materialized_dataset(test_p) if test_p.exists() else val_ds
    total_mib = (dataset_memory_bytes(train_ds) + dataset_memory_bytes(val_ds)) / (1024 * 1024)
    logger.info(
        "已物化数据驻留内存: %.1f MiB (train=%d, val=%d) — 训练仅 DataLoader 切片，无在线特征计算",
        total_mib,
        len(train_ds),
        len(val_ds),
    )
    return train_ds, val_ds, test_ds


def materialize_datasets(
    config: GlobalConfig,
    *,
    share_memory: bool = False,
    max_samples_per_instrument: Optional[int] = None,
    save_preprocessed: Optional[Path] = None,
    seed: int = 42,
    kronos_cache_dir: Optional[Path] = None,
    allow_live_kronos: bool = False,
    cpu_threads: int = DEFAULT_CPU_THREADS,
    materialize_fork: bool = False,
) -> Tuple[MaterializedBPCV4Dataset, MaterializedBPCV4Dataset, MaterializedBPCV4Dataset]:
    """
    预处理物化：qlib 批量读入 → Kronos → NumPy 分片物化 train/val；可选落盘。
    """
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
        "预处理物化: calendar %s..%s | train_end=%s | val=%s..%s | instruments=%d",
        split.data_start.date(),
        split.data_end.date(),
        split.train_end.date(),
        split.val_start.date(),
        split.val_end.date(),
        len(config.qlib.instruments),
    )

    store = BPCV4InstrumentStore(
        instruments=list(config.qlib.instruments),
        start=config.qlib.start_date,
        end=config.qlib.end_date,
        provider_uri=provider_uri,
        calendar=calendar,
        pad_missing_amount=config.kronos.amount_pad_zero,
    )
    cs_medians = CrossSectionDeltaMedians.from_v4_store(store)
    vol_stats = VolatilityStats.from_store(store)
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

    logger.info(
        "预处理: Kronos + BPC 物化 | train=%d val=%d 窗口",
        len(train_samples),
        len(val_samples),
    )

    kronos_cache: Optional[KronosPrecomputeStore] = None
    live_lookup: Optional[LiveKronosLookup] = None
    all_samples = train_samples + val_samples

    if kronos_cache_dir:
        kronos_cache = KronosPrecomputeStore.open(kronos_cache_dir)
        kronos_cache.validate_compatible(config)
        unique_syms = {sym for sym, _ in all_samples}
        kronos_cache.preload(unique_syms)
        logger.info("Kronos: 磁盘缓存 %d 标的已载入 RAM", len(unique_syms))
    elif allow_live_kronos:
        logger.info("Kronos: 在线编码 %d 窗口（建议改用 precompute_kronos + --kronos-cache-dir）", len(all_samples))
        live_lookup = build_live_kronos_lookup(
            all_samples,
            store,
            config,
            cpu_threads=resolve_cpu_threads(cpu_threads),
            batch_size=64,
        )
    else:
        raise RuntimeError(
            "物化需要 Kronos：指定 --kronos-cache-dir 或 --allow-live-kronos。"
            "推荐: python -m quant_cursor.bpc_v4.precompute_kronos --force-rebuild"
        )

    if live_lookup is not None:
        kronos_for_materialize: KronosLookup = freeze_live_lookup_for_samples(live_lookup, all_samples)
        del live_lookup
        release_live_kronos_encoders()
        gc.collect()
    else:
        assert kronos_cache is not None
        kronos_for_materialize = freeze_kronos_cache_for_spawn(kronos_cache, all_samples)

    logger.info("物化 train split（分片批量 + 并行）")
    train_ds = materialize_samples(
        config, train_samples, store, cs_medians=cs_medians, vol_stats=vol_stats,
        kronos_lookup=kronos_for_materialize, cpu_threads=cpu_threads,
        materialize_fork=materialize_fork,
    )
    logger.info("物化 val split（分片批量 + 并行）")
    val_ds = materialize_samples(
        config, val_samples, store, cs_medians=cs_medians, vol_stats=vol_stats,
        kronos_lookup=kronos_for_materialize, cpu_threads=cpu_threads,
        materialize_fork=materialize_fork,
    )
    train_ds = _finalize_cpu_dataset(train_ds, label="train", share_memory=share_memory)
    val_ds = _finalize_cpu_dataset(val_ds, label="val", share_memory=share_memory)
    test_ds = val_ds

    if save_preprocessed:
        save_preprocessed.mkdir(parents=True, exist_ok=True)
        save_materialized_dataset(train_ds, save_preprocessed / "train")
        save_materialized_dataset(val_ds, save_preprocessed / "val")
        meta = {
            "schema_version": BPC_V4_SCHEMA,
            "ohlcv_relative_schema": "field_delta_cs_v2_lag1",
            "start_date": config.qlib.start_date,
            "end_date": config.qlib.end_date,
            "train_end": str(split.train_end.date()),
            "val_start": str(split.val_start.date()),
            "val_end": str(split.val_end.date()),
            "n_instruments": len(store._cache),
            "instrument_ids": list(store._cache.keys()),
            "max_samples": config.qlib.max_samples,
            "max_samples_per_instrument": max_samples_per_instrument,
            "codebook_output_dim": config.head.codebook_output_dim,
            "kronos_s1_bits": config.kronos.s1_bits,
            "kronos_cache_dir": str(kronos_cache_dir) if kronos_cache_dir else None,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
        }
        (save_preprocessed / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("已保存物化数据集 → %s", save_preprocessed)

    total_mib = (dataset_memory_bytes(train_ds) + dataset_memory_bytes(val_ds)) / (1024 * 1024)
    logger.info(
        "预处理完成: %.1f MiB RAM (train=%d + val=%d)。训练请 --preprocessed-dir",
        total_mib,
        len(train_ds),
        len(val_ds),
    )
    return train_ds, val_ds, test_ds


def resolve_training_datasets(
    config: GlobalConfig,
    *,
    preprocessed_dir: Optional[Path] = None,
    save_preprocessed: Optional[Path] = None,
    force_rebuild: bool = False,
    share_memory: bool = False,
    max_samples_per_instrument: Optional[int] = None,
    seed: int = 42,
    kronos_cache_dir: Optional[Path] = None,
    allow_live_kronos: bool = False,
    cpu_threads: int = DEFAULT_CPU_THREADS,
    materialize_fork: bool = False,
) -> Tuple[MaterializedBPCV4Dataset, MaterializedBPCV4Dataset, MaterializedBPCV4Dataset]:
    """
    训练启动前一次性准备数据（不在 epoch 内在线计算）：

    - 有有效 ``--preprocessed-dir`` → 从磁盘加载
    - 否则 → 现场物化（类似 Kronos：训练前算好，再进 DataLoader）
    - 可选 ``--save-preprocessed`` 在物化后落盘
    """
    if preprocessed_dir and not force_rebuild:
        root = Path(preprocessed_dir)
        if (root / "train").is_dir() and (root / "val").is_dir():
            logger.info("数据: 加载已物化缓存 %s", root)
            return load_preprocessed_datasets(root, share_memory=share_memory)
        logger.warning("preprocessed-dir 无效 (%s)，改为训练前现场物化", root)

    logger.info(
        "数据: 训练前现场物化 (qlib batch → Kronos → NumPy BPC)；epoch 内仅 DataLoader 切片"
    )
    return materialize_datasets(
        config,
        share_memory=share_memory,
        max_samples_per_instrument=max_samples_per_instrument,
        save_preprocessed=save_preprocessed,
        seed=seed,
        kronos_cache_dir=kronos_cache_dir,
        allow_live_kronos=allow_live_kronos,
        cpu_threads=cpu_threads,
        materialize_fork=materialize_fork,
    )


def build_datasets(
    config: GlobalConfig,
    *,
    share_memory: bool = False,
    max_samples_per_instrument: Optional[int] = None,
    preprocessed_dir: Optional[Path] = None,
    save_preprocessed: Optional[Path] = None,
    force_rebuild: bool = False,
    seed: int = 42,
    kronos_cache_dir: Optional[Path] = None,
    allow_live_kronos: bool = False,
    cpu_threads: int = DEFAULT_CPU_THREADS,
    materialize_fork: bool = False,
) -> Tuple[MaterializedBPCV4Dataset, MaterializedBPCV4Dataset, MaterializedBPCV4Dataset]:
    """兼容入口 → resolve_training_datasets。"""
    return resolve_training_datasets(
        config,
        preprocessed_dir=preprocessed_dir,
        save_preprocessed=save_preprocessed,
        force_rebuild=force_rebuild,
        share_memory=share_memory,
        max_samples_per_instrument=max_samples_per_instrument,
        seed=seed,
        kronos_cache_dir=kronos_cache_dir,
        allow_live_kronos=allow_live_kronos,
        cpu_threads=cpu_threads,
        materialize_fork=materialize_fork,
    )


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
        "purity_target": torch.stack([b["purity_target"] for b in batch]),
        "next_day_sign": torch.stack([b["next_day_sign"] for b in batch]),
    }


def create_dataloaders(
    config: GlobalConfig,
    opts: LoaderOptions,
    train_ds: MaterializedBPCV4Dataset,
    val_ds: MaterializedBPCV4Dataset,
) -> Tuple[
    Union[DataLoader, BatchedGpuBPCV4Dataset],
    Union[DataLoader, BatchedGpuBPCV4Dataset],
    MaterializedBPCV4Dataset,
    MaterializedBPCV4Dataset,
]:
    """
    从已物化 CPU 数据集构建 DataLoader（或 GPU 驻留包装）。

    训练循环仅迭代 loader，不在线计算 Kronos/BPC 特征。
    """
    device = torch.device(config.train.device if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    train_bs, train_drop_last, _ = _train_loader_batch_config(len(train_ds), config.train.batch_size)
    val_bs = min(config.train.batch_size, max(1, len(val_ds)))

    batched_gpu_resident = opts.batched_gpu and use_cuda and not opts.batched_gpu_cpu
    share_memory = opts.num_workers > 0 and not opts.gpu_cache_data and not opts.batched_gpu

    if batched_gpu_resident:
        if opts.num_workers > 0:
            logger.info("--batched-gpu: ignoring num_workers=%d", opts.num_workers)
        logger.info(
            "训练: 一次性上传 GPU，iter_batches() 仅切片已物化张量"
        )
        train_loader: Union[DataLoader, BatchedGpuBPCV4Dataset] = BatchedGpuBPCV4Dataset(
            train_ds, device=str(device), batch_size=train_bs, drop_last=train_drop_last, shuffle=True
        )
        val_loader = BatchedGpuBPCV4Dataset(
            val_ds, device=str(device), batch_size=val_bs, drop_last=False, shuffle=False
        )
        train_mib = train_loader._gpu_mib() if hasattr(train_loader, "_gpu_mib") else 0.0
        val_mib = val_loader._gpu_mib() if hasattr(val_loader, "_gpu_mib") else 0.0
        logger.info(
            "GPU resident total: %.1f MiB (train %.1f + val %.1f) — zero H2D / zero feature compute per step",
            train_mib + val_mib,
            train_mib,
            val_mib,
        )
        return train_loader, val_loader, train_ds, val_ds

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
            train_ds,
            val_ds,
        )

    base_train_ds, base_val_ds = train_ds, val_ds
    if opts.gpu_cache_data and use_cuda:
        logger.info("GpuCachedBPCV4Dataset on %s (num_workers=0, zero H2D)", device)
        train_ds = GpuCachedBPCV4Dataset(train_ds, device=str(device))
        val_ds = GpuCachedBPCV4Dataset(val_ds, device=str(device))
        if opts.num_workers > 0:
            logger.warning("--gpu-cache-data forces num_workers=0 (was %d)", opts.num_workers)
        return (
            DataLoader(train_ds, batch_size=train_bs, shuffle=True, drop_last=train_drop_last, num_workers=0, pin_memory=False, collate_fn=collate_fn),
            DataLoader(val_ds, batch_size=val_bs, shuffle=False, drop_last=False, num_workers=0, pin_memory=False, collate_fn=collate_fn),
            base_train_ds,
            base_val_ds,
        )

    if share_memory:
        pin_dataset_share_memory(train_ds)
        pin_dataset_share_memory(val_ds)
        logger.info("share_memory enabled (num_workers=%d)", opts.num_workers)

    logger.info(
        "训练 DataLoader: batch=%d, workers=%d, prefetch=%d, pin_memory=%s",
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
        train_ds,
        val_ds,
    )
