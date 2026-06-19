from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from quant_cursor.bpc.behavior_features import compute_behavior_proxies_stacked

if TYPE_CHECKING:
    from quant_cursor.bpc.model import ScaleRegistry

logger = logging.getLogger(__name__)

OHLCV_FIELDS = ["$open", "$high", "$low", "$close", "$volume"]
QLIB_LOAD_FIELDS = OHLCV_FIELDS + ["$factor"]
_QLIB_INITIALIZED = False

DEFAULT_TRAIN_INSTRUMENTS = [
    "SH000001",
    "SH000016",
    "SH000300",
    "SH000905",
    "SZ399001",
    "SZ399006",
    "SH510050",
    "SH510300",
    "SZ159915",
]


@dataclass(frozen=True)
class TemporalSplit:
    """时间切分边界（均为交易日，含边界）。"""

    data_start: pd.Timestamp
    data_end: pd.Timestamp
    train_end: pd.Timestamp  # 训练锚点日 upper bound（含）
    val_start: pd.Timestamp  # 验证锚点日 lower bound（含）
    val_end: pd.Timestamp

    @classmethod
    def from_calendar(
        cls,
        calendar: pd.DatetimeIndex,
        val_ratio: float = 0.20,
        *,
        data_start: str | None = None,
        data_end: str | None = None,
    ) -> TemporalSplit:
        cal = calendar.sort_values()
        if data_start:
            cal = cal[cal >= pd.Timestamp(data_start)]
        if data_end:
            cal = cal[cal <= pd.Timestamp(data_end)]
        if len(cal) < 20:
            raise ValueError("交易日历过短，无法切分 train/val")

        n_val = max(1, int(len(cal) * val_ratio))
        n_train = len(cal) - n_val
        if n_train < 1:
            raise ValueError("val_ratio 过大，训练集为空")

        train_end = cal[n_train - 1]
        val_start = cal[n_train]
        return cls(
            data_start=cal[0],
            data_end=cal[-1],
            train_end=train_end,
            val_start=val_start,
            val_end=cal[-1],
        )


def ensure_qlib(provider_uri: str | Path) -> None:
    global _QLIB_INITIALIZED
    if _QLIB_INITIALIZED:
        return
    import qlib
    from qlib.constant import REG_CN

    uri = Path(provider_uri).resolve()
    qlib.init(provider_uri=str(uri), region=REG_CN)
    _QLIB_INITIALIZED = True
    logger.info("Qlib initialized: %s", uri)


def load_trading_calendar(provider_uri: str | Path) -> pd.DatetimeIndex:
    cal_path = Path(provider_uri) / "calendars" / "day.txt"
    if not cal_path.exists():
        raise FileNotFoundError(f"Qlib 日历不存在: {cal_path}")
    dates = pd.read_csv(cal_path, header=None, names=["date"])["date"]
    return pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()


def load_qlib_instruments(
    manifest_path: Path,
    *,
    instruments: list[str] | None = None,
    asset_types: list[str] | None = None,
    categories: list[str] | None = None,
    sw_l2_codes: list[str] | None = None,
    sw_l2_names: list[str] | None = None,
    groups: list[str] | None = None,
    min_rows: int = 60,
    max_instruments: int | None = None,
) -> list[str]:
    from quant_cursor.instruments import query_instruments

    return query_instruments(
        manifest_path=manifest_path,
        instruments=instruments,
        asset_types=asset_types,
        categories=categories,
        sw_l2_codes=sw_l2_codes,
        sw_l2_names=sw_l2_names,
        groups=groups,
        min_rows=min_rows,
        max_instruments=max_instruments,
    )


def _apply_factor_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """
    后复权 OHLCV（Qlib 约定：adj_price = raw_price * factor，adj_volume = volume / factor）。
    factor 缺失时视为 1.0（本项目指数/ETF 多为 1.0）。
    """
    factor = pd.to_numeric(df["$factor"], errors="coerce").ffill().bfill().fillna(1.0)
    factor = factor.replace(0.0, 1.0)

    out = pd.DataFrame(index=df.index)
    for col in ("$open", "$high", "$low", "$close"):
        out[col] = pd.to_numeric(df[col], errors="coerce") * factor
    vol = pd.to_numeric(df["$volume"], errors="coerce").fillna(0.0)
    out["$volume"] = vol / factor
    return out


def _align_to_trading_days(df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """将序列对齐到 Qlib 交易日历，缺失日用 NaN 占位（不向前看未来）。"""
    cal = calendar[(calendar >= df.index.min()) & (calendar <= df.index.max())]
    aligned = df.reindex(cal)
    return aligned


def _resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.resample("W-FRI")
        .agg(
            {
                "$open": "first",
                "$high": "max",
                "$low": "min",
                "$close": "last",
                "$volume": "sum",
            }
        )
        .dropna(subset=["$close"])
    )


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df[OHLCV_FIELDS].copy()
    for col in ("$open", "$high", "$low", "$close"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["$volume"] = pd.to_numeric(out["$volume"], errors="coerce").fillna(0.0)

    price = out[["$open", "$high", "$low", "$close"]]
    valid = price.notna().all(axis=1) & price.gt(0).all(axis=1) & (out["$high"] >= out["$low"])
    out = out[valid]
    if out.empty:
        return out

    first = out.index[0]
    out = out.loc[first:]
    return out


def _window_is_finite(arr: np.ndarray, end_idx: int, lookback: int) -> bool:
    start = end_idx - lookback + 1
    if start < 0:
        return False
    chunk = arr[start : end_idx + 1]
    return bool(np.isfinite(chunk).all() and (chunk[:, 3] > 0).all())


def _batch_window_valid(arr: np.ndarray, lookback: int) -> np.ndarray:
    """对每个窗口终点 t，返回 [t-lookback+1, t] 是否全为有限正值且 OHLC 合法。"""
    t_len, n_feat = arr.shape
    if t_len < lookback:
        return np.zeros(0, dtype=bool)
    from numpy.lib.stride_tricks import as_strided

    n = t_len - lookback + 1
    windows = as_strided(
        arr,
        shape=(n, lookback, n_feat),
        strides=(arr.strides[0], arr.strides[0], arr.strides[1]),
        writeable=False,
    )
    ok = np.isfinite(windows).all(axis=(1, 2))
    ok &= (windows[:, :, :4] > 0).all(axis=(1, 2))
    ok &= (windows[:, :, 1] >= windows[:, :, 2]).all(axis=1)
    return ok


@dataclass
class _InstrumentSeries:
    qlib_id: str
    dates: pd.DatetimeIndex
    ohlcv: np.ndarray
    weekly: np.ndarray
    weekly_end_dates: pd.DatetimeIndex
    daily_week_idx: np.ndarray


class QlibInstrumentStore:
    """一次性加载全量标的（复权 + 日历对齐），供 train/val 数据集共享。"""

    def __init__(
        self,
        instruments: list[str],
        start: str,
        end: str,
        provider_uri: str | Path,
        registry: ScaleRegistry,
        calendar: pd.DatetimeIndex | None = None,
    ):
        from quant_cursor.bpc.model import ScaleConfig

        ensure_qlib(provider_uri)
        from qlib.data import D

        self.provider_uri = Path(provider_uri)
        self.calendar = calendar if calendar is not None else load_trading_calendar(self.provider_uri)
        self.scales: list[ScaleConfig] = registry.get_enabled()
        self.start = start
        self.end = end

        day_cfg = next((c for c in self.scales if c.name == "day"), None)
        week_cfg = next((c for c in self.scales if c.name == "week"), None)
        self.day_lb = day_cfg.lookback_window if day_cfg else 0
        self.week_lb = week_cfg.lookback_window if week_cfg else 0

        self._cache: dict[str, _InstrumentSeries] = {}
        total = len(instruments)
        for i, qlib_id in enumerate(instruments, 1):
            series = self._load_one(D, qlib_id, start, end)
            if series is not None:
                self._cache[qlib_id] = series
            if i % 100 == 0 or i == total:
                logger.info("InstrumentStore: loaded %d/%d (cached=%d)", i, total, len(self._cache))

        if not self._cache:
            raise RuntimeError("InstrumentStore: 无有效标的")

        # Stable symbol → id mapping (order of insertion = order in instruments list)
        self.symbol_to_id: dict[str, int] = {qlib_id: i for i, qlib_id in enumerate(self._cache.keys())}
        self.id_to_symbol: dict[int, str] = {i: s for s, i in self.symbol_to_id.items()}

        logger.info("InstrumentStore ready: %d instruments", len(self._cache))

    def _load_one(self, D, qlib_id: str, start: str, end: str) -> _InstrumentSeries | None:
        try:
            panel = D.features([qlib_id], QLIB_LOAD_FIELDS, start_time=start, end_time=end, freq="day")
        except Exception as exc:  # noqa: BLE001
            logger.warning("跳过 %s: %s", qlib_id, exc)
            return None

        if panel is None or panel.empty:
            return None
        if "instrument" in panel.index.names:
            panel = panel.droplevel("instrument")
        panel = panel.sort_index()
        panel = _apply_factor_adjustment(panel)
        panel = _clean_ohlcv(panel)

        if len(panel) < max(self.day_lb, 1):
            return None

        weekly_df = _resample_weekly(panel)
        if self.week_lb and len(weekly_df) < self.week_lb:
            return None

        ohlcv = panel.values.astype(np.float32)
        weekly = weekly_df.values.astype(np.float32)
        week_ends = weekly_df.index
        daily_week_idx = np.searchsorted(week_ends.values, panel.index.values, side="right") - 1
        daily_week_idx = np.clip(daily_week_idx, 0, len(weekly) - 1)

        return _InstrumentSeries(
            qlib_id=qlib_id,
            dates=panel.index,
            ohlcv=ohlcv,
            weekly=weekly,
            weekly_end_dates=week_ends,
            daily_week_idx=daily_week_idx,
        )

    @property
    def instruments(self) -> list[str]:
        return list(self._cache.keys())


class QlibMultiScaleDataset(Dataset):
    """
    多尺度数据集，支持 train/val 时间切分。

    防泄漏规则：
    - train 样本锚点 t 满足 t <= train_end，且日窗口 [t-day_lb+1, t] 全部 <= train_end
    - val 样本锚点 t 满足 val_start <= t <= val_end（窗口可回看 train 期，但不看 t 之后）
    - train / val 锚点日期互不重叠
    """

    def __init__(
        self,
        store: QlibInstrumentStore,
        split: TemporalSplit,
        mode: Literal["train", "val"],
        *,
        max_samples_per_instrument: int | None = None,
        seed: int = 42,
    ):
        from quant_cursor.bpc.model import ScaleConfig

        self.store = store
        self.split = split
        self.mode = mode
        self.scales: list[ScaleConfig] = store.scales
        self.day_lb = store.day_lb
        self.week_lb = store.week_lb

        self._samples: list[tuple[str, int]] = []
        rng = np.random.default_rng(seed)
        n_inst = len(store._cache)

        for j, (qlib_id, series) in enumerate(store._cache.items(), 1):
            valid_ts = self._collect_valid_indices(series)
            if max_samples_per_instrument and len(valid_ts) > max_samples_per_instrument:
                valid_ts = rng.choice(valid_ts, size=max_samples_per_instrument, replace=False).tolist()
                valid_ts.sort()
            self._samples.extend((qlib_id, t) for t in valid_ts)
            if j % 500 == 0 or j == n_inst:
                logger.info("[%s] indexed %d/%d instruments, samples=%d", mode, j, n_inst, len(self._samples))

        # Pre-compute symbol ids and timestamps for the conditioner (shared across train/val via store)
        self.symbol_to_id = store.symbol_to_id
        # Use trading-day ordinal within the full calendar as a monotonic time signal
        self.calendar = store.calendar
        self.date_to_ordinal = {d: i for i, d in enumerate(self.calendar)}

        if not self._samples:
            raise RuntimeError(f"{mode} 集未构建任何样本")

        logger.info(
            "QlibMultiScaleDataset [%s]: %d samples (train_end=%s, val=[%s,%s])",
            mode,
            len(self._samples),
            split.train_end.date(),
            split.val_start.date(),
            split.val_end.date(),
        )

    def _collect_valid_indices(self, series: _InstrumentSeries) -> list[int]:
        n = len(series.dates)
        if n < self.day_lb:
            return []

        t_idx = np.arange(self.day_lb - 1, n, dtype=np.int64)
        anchors = series.dates[t_idx]

        if self.mode == "train":
            mask = anchors <= self.split.train_end
            window_starts = series.dates[t_idx - self.day_lb + 1]
            mask &= window_starts >= self.split.data_start
            mask &= window_starts <= self.split.train_end
        else:
            mask = (anchors >= self.split.val_start) & (anchors <= self.split.val_end)
            mask &= series.dates[t_idx - self.day_lb + 1] >= self.split.data_start

        if self.week_lb:
            w_idx = series.daily_week_idx[t_idx]
            mask &= w_idx >= self.week_lb - 1

        day_valid = _batch_window_valid(series.ohlcv, self.day_lb)
        day_pos = t_idx - self.day_lb + 1
        day_in_range = (day_pos >= 0) & (day_pos < len(day_valid))
        day_ok = np.zeros(len(t_idx), dtype=bool)
        if day_in_range.any():
            day_ok[day_in_range] = day_valid[day_pos[day_in_range]]
        mask &= day_ok

        if self.week_lb:
            week_valid = _batch_window_valid(series.weekly, self.week_lb)
            week_pos = series.daily_week_idx[t_idx] - self.week_lb + 1
            week_in_range = (week_pos >= 0) & (week_pos < len(week_valid))
            week_ok = np.zeros(len(t_idx), dtype=bool)
            if week_in_range.any():
                week_ok[week_in_range] = week_valid[week_pos[week_in_range]]
            mask &= week_ok

        return t_idx[mask].tolist()

    def __len__(self) -> int:
        return len(self._samples)

    def _window(self, arr: np.ndarray, end_idx: int, lookback: int) -> np.ndarray:
        start = end_idx - lookback + 1
        return arr[start : end_idx + 1]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        qlib_id, t = self._samples[idx]
        series = self.store._cache[qlib_id]
        sample: dict[str, torch.Tensor] = {}

        for cfg in self.scales:
            if cfg.name == "day":
                win = self._window(series.ohlcv, t, cfg.lookback_window)
            elif cfg.name == "week":
                w_idx = int(series.daily_week_idx[t])
                win = self._window(series.weekly, w_idx, cfg.lookback_window)
            else:
                raise NotImplementedError(f"尺度 '{cfg.name}' 尚未实现")
            sample[cfg.name] = torch.from_numpy(win)

        # Symbol id + timestamp for SymbolTimeFiLM conditioner
        sid = self.symbol_to_id.get(qlib_id, 0)
        sample["stock_ids"] = torch.tensor(sid, dtype=torch.long)
        # Trading-day ordinal (monotonic, stable across runs)
        ts_val = self.date_to_ordinal.get(series.dates[t], 0)
        sample["timestamps"] = torch.tensor(ts_val, dtype=torch.long)

        return sample


class MaterializedMultiScaleDataset(Dataset):
    """
    预物化全部窗口为连续 tensor，__getitem__ O(1)。
    share_memory_ 后多 worker 不再复制整库数据（Windows 友好）。
    可选预计算 behavior_proxies，避免训练每 batch 重复 CPU 特征工程。
    """

    PROXY_CHUNK = 16_384

    def __init__(self, base: QlibMultiScaleDataset, *, share_memory: bool = True, precompute_proxies: bool = True):
        self.scales = base.scales
        self.primary_scale = "day"
        for cfg in base.scales:
            if cfg.name == "day":
                self.primary_scale = "day"
                break
        n = len(base)
        logger.info("Materializing %d samples ...", n)

        scale_arrays: dict[str, np.ndarray] = {}
        for cfg in base.scales:
            lb = cfg.lookback_window
            scale_arrays[cfg.name] = np.empty((n, lb, 5), dtype=np.float32)

        for i, (qlib_id, t) in enumerate(base._samples):
            series = base.store._cache[qlib_id]
            for cfg in base.scales:
                if cfg.name == "day":
                    scale_arrays[cfg.name][i] = base._window(series.ohlcv, t, cfg.lookback_window)
                elif cfg.name == "week":
                    w_idx = int(series.daily_week_idx[t])
                    scale_arrays[cfg.name][i] = base._window(series.weekly, w_idx, cfg.lookback_window)
                else:
                    raise NotImplementedError(f"尺度 '{cfg.name}' 尚未实现")
            if (i + 1) % 500_000 == 0 or i + 1 == n:
                logger.info("Materialized windows %d/%d", i + 1, n)

        self._tensors: dict[str, torch.Tensor] = {}
        for name, arr in scale_arrays.items():
            t = torch.from_numpy(arr)
            if share_memory:
                t = t.share_memory_()
            self._tensors[name] = t

        # Materialize symbol ids and trading-day ordinals (needed by SymbolTimeFiLM)
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

        self._proxies: torch.Tensor | None = None
        if precompute_proxies and "day" in self._tensors:
            logger.info("Precomputing behavior proxies (%d samples) ...", n)
            day = self._tensors["day"]
            proxy_chunks: list[torch.Tensor] = []
            for start in range(0, n, self.PROXY_CHUNK):
                end = min(start + self.PROXY_CHUNK, n)
                proxy_chunks.append(compute_behavior_proxies_stacked(day[start:end].float()))
                if end % 500_000 == 0 or end == n:
                    logger.info("Precomputed proxies %d/%d", end, n)
            proxies = torch.cat(proxy_chunks, dim=0)
            if share_memory:
                proxies = proxies.share_memory_()
            self._proxies = proxies

        logger.info(
            "Materialized dataset ready (%d samples, share_memory=%s, proxies=%s, symbol_condition=%s)",
            n,
            share_memory,
            self._proxies is not None,
            True,
        )

    def __len__(self) -> int:
        return next(iter(self._tensors.values())).shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = {name: t[idx] for name, t in self._tensors.items()}
        sample["stock_ids"] = self._stock_ids[idx]
        sample["timestamps"] = self._timestamps[idx]
        if self._proxies is not None:
            sample["behavior_proxies"] = self._proxies[idx]
        return sample


# =============================================================================
# 磁盘持久化物化数据集（一次预处理，多次快速加载）
# =============================================================================

def save_materialized_dataset(ds: "MaterializedMultiScaleDataset", path: str | Path) -> None:
    """将已物化的数据集保存到磁盘，支持后续训练直接加载，无需重新走 qlib + 窗口化。

    保存内容：
      - 各尺度的 OHLCV 张量
      - stock_ids, timestamps
      - behavior_proxies（如有）
      - 元数据（scales, primary_scale）
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    meta = {
        "scales": [(c.name, c.freq, c.lookback_window) for c in ds.scales],
        "primary_scale": ds.primary_scale,
        "has_proxies": ds._proxies is not None,
        "n_samples": len(ds),
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2))

    # Use numpy .npy instead of torch.save to avoid PyTorch zip serialization bugs
    # with share_memory_() tensors on Linux ("unexpected pos" in inline_container.cc).
    def _safe_save(t: torch.Tensor, p: Path) -> None:
        np.save(p.with_suffix(".npy"), t.detach().cpu().numpy())

    for name, t in ds._tensors.items():
        _safe_save(t, path / f"{name}.pt")

    _safe_save(ds._stock_ids, path / "stock_ids.pt")
    _safe_save(ds._timestamps, path / "timestamps.pt")

    if ds._proxies is not None:
        _safe_save(ds._proxies, path / "behavior_proxies.pt")

    logger.info("Materialized dataset saved to %s (%d samples)", path, len(ds))


def load_materialized_dataset(path: str | Path) -> "MaterializedMultiScaleDataset":
    """从磁盘加载预物化数据集，跳过 qlib 加载与窗口化，实现秒级启动。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"预处理目录不存在: {path}")

    meta = json.loads((path / "meta.json").read_text())
    ds = MaterializedMultiScaleDataset.__new__(MaterializedMultiScaleDataset)
    ds.scales = []  # 占位，实际不使用
    ds.primary_scale = meta["primary_scale"]

    ds._tensors = {}
    for name, _, _ in meta["scales"]:
        npy_path = path / f"{name}.npy"
        pt_path = path / f"{name}.pt"
        if npy_path.exists():
            ds._tensors[name] = torch.from_numpy(np.load(npy_path))
        elif pt_path.exists():
            ds._tensors[name] = torch.load(pt_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"Missing tensor file for scale '{name}' in {path}")

    def _load_field(stem: str) -> torch.Tensor:
        npy_path = path / f"{stem}.npy"
        pt_path = path / f"{stem}.pt"
        if npy_path.exists():
            return torch.from_numpy(np.load(npy_path))
        if pt_path.exists():
            return torch.load(pt_path, map_location="cpu", weights_only=True)
        raise FileNotFoundError(f"Missing {stem} in {path}")

    ds._stock_ids = _load_field("stock_ids")
    ds._timestamps = _load_field("timestamps")

    if meta.get("has_proxies"):
        ds._proxies = _load_field("behavior_proxies")
    else:
        ds._proxies = None

    logger.info("Loaded pre-materialized dataset from %s (%d samples)", path, len(ds))
    return ds


class GpuCachedDataset(Dataset):
    """Move materialized tensors to GPU once — eliminates per-batch H2D transfer.

    Requires num_workers=0 in DataLoader (CUDA tensors cannot be loaded in workers).
    """

    def __init__(self, base: MaterializedMultiScaleDataset, device: str = "cuda"):
        self.scales = base.scales
        self.primary_scale = base.primary_scale
        dev = torch.device(device)
        logger.info("Moving %d samples to %s (one-time upload) ...", len(base), dev)
        self._tensors = {name: t.to(dev, non_blocking=True) for name, t in base._tensors.items()}
        self._stock_ids = base._stock_ids.to(dev, non_blocking=True)
        self._timestamps = base._timestamps.to(dev, non_blocking=True)
        self._proxies = base._proxies.to(dev, non_blocking=True) if base._proxies is not None else None
        if dev.type == "cuda":
            torch.cuda.synchronize()
        logger.info("GPU cache ready on %s", dev)

    def __len__(self) -> int:
        return next(iter(self._tensors.values())).shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = {name: t[idx] for name, t in self._tensors.items()}
        sample["stock_ids"] = self._stock_ids[idx]
        sample["timestamps"] = self._timestamps[idx]
        if self._proxies is not None:
            sample["behavior_proxies"] = self._proxies[idx]
        return sample


def build_datasets(
    instruments: list[str],
    start: str,
    end: str,
    provider_uri: str | Path,
    registry: ScaleRegistry,
    *,
    val_ratio: float = 0.20,
    max_samples_per_instrument: int | None = None,
    seed: int = 42,
    materialize: bool = True,
    share_memory: bool = True,
    precompute_proxies: bool = True,
) -> tuple[QlibInstrumentStore, Dataset, Dataset, TemporalSplit]:
    calendar = load_trading_calendar(provider_uri)
    split = TemporalSplit.from_calendar(calendar, val_ratio, data_start=start, data_end=end)

    store = QlibInstrumentStore(
        instruments=instruments,
        start=start,
        end=end,
        provider_uri=provider_uri,
        registry=registry,
        calendar=calendar,
    )
    train_base = QlibMultiScaleDataset(
        store, split, mode="train", max_samples_per_instrument=max_samples_per_instrument, seed=seed
    )
    val_base = QlibMultiScaleDataset(
        store, split, mode="val", max_samples_per_instrument=max_samples_per_instrument, seed=seed + 1
    )
    if materialize:
        train_ds: Dataset = MaterializedMultiScaleDataset(
            train_base, share_memory=share_memory, precompute_proxies=precompute_proxies
        )
        val_ds: Dataset = MaterializedMultiScaleDataset(
            val_base, share_memory=share_memory, precompute_proxies=precompute_proxies
        )
    else:
        train_ds = train_base
        val_ds = val_base
    return store, train_ds, val_ds, split
