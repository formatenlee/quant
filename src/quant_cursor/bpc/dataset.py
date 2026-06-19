from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from quant_cursor.bpc.behavior_features import (
    NUM_BEHAVIOR_AGENTS,
    compute_behavior_proxies_stacked,
    transform_proxies_for_labeling,
)
from quant_cursor.bpc.feature_dims import DAY_FULL_FEAT_DIM
from quant_cursor.bpc.structure_features import volume_log_level_anomaly, volume_rel_cv

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
    train_end: pd.Timestamp
    val_start: pd.Timestamp
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
    prefer_minute: bool = False,
    require_minute: bool = False,
    minute_min_rows: int = 1,
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
        prefer_minute=prefer_minute,
        require_minute=require_minute,
        minute_min_rows=minute_min_rows,
    )


def _apply_factor_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    factor = pd.to_numeric(df["$factor"], errors="coerce").ffill().bfill().fillna(1.0)
    factor = factor.replace(0.0, 1.0)

    out = pd.DataFrame(index=df.index)
    for col in ("$open", "$high", "$low", "$close"):
        out[col] = pd.to_numeric(df[col], errors="coerce") * factor
    vol = pd.to_numeric(df["$volume"], errors="coerce").fillna(0.0)
    out["$volume"] = vol / factor
    return out


def _align_to_trading_days(df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
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


def _batch_window_valid(arr: np.ndarray, lookback: int) -> np.ndarray:
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

        self.symbol_to_id: dict[str, int] = {qlib_id: i for i, qlib_id in enumerate(self._cache.keys())}
        self.id_to_symbol: dict[int, str] = {i: s for s, i in self.symbol_to_id.items()}

        logger.info("InstrumentStore ready: %d instruments", len(self._cache))

    def _load_one(self, D, qlib_id: str, start: str, end: str) -> _InstrumentSeries | None:
        try:
            panel = D.features([qlib_id], QLIB_LOAD_FIELDS, start_time=start, end_time=end, freq="day")
        except Exception as exc:
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

        self.symbol_to_id = store.symbol_to_id
        self.calendar = store.calendar
        self.date_to_ordinal = {d: i for i, d in enumerate(self.calendar)}

        if not self._samples:
            raise RuntimeError(f"{mode} 集未构建任何样本")

        logger.info(
            "QlibMultiScaleDataset [%s]: %d samples",
            mode,
            len(self._samples),
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
            sample[cfg.name] = torch.from_numpy(win.astype(np.float32))

        sid = self.symbol_to_id.get(qlib_id, 0)
        sample["stock_ids"] = torch.tensor(sid, dtype=torch.long)
        ts_val = self.date_to_ordinal.get(series.dates[t], 0)
        sample["timestamps"] = torch.tensor(ts_val, dtype=torch.long)

        return sample


# =============================================================================
# 核心优化：预计算 day_features 与 behavior_proxies，物化到磁盘
# =============================================================================
from quant_cursor.bpc.features import compute_day_features_vectorized, compute_week_features_vectorized


def _compute_features_vectorized(ohlcv: torch.Tensor, **kwargs) -> torch.Tensor:
    """绝对 OHLCV → day 全量特征（v2 物化路径）。"""
    return compute_day_features_vectorized(ohlcv)


def _compute_week_features_vectorized(ohlcv: torch.Tensor, **kwargs) -> torch.Tensor:
    return compute_week_features_vectorized(ohlcv)


class MaterializedMultiScaleDataset(Dataset):
    """
    物化数据集：预计算所有尺度的特征向量，__getitem__ 只做索引。
    支持 share_memory_ 多 worker 零拷贝共享。
    """

    PROXY_CHUNK = 16_384

    def __init__(
        self,
        base: QlibMultiScaleDataset,
        *,
        share_memory: bool = True,
        precompute_features: bool = True,
        precompute_proxies: bool = True,
    ):
        self.scales = base.scales
        self.primary_scale = "day"
        for cfg in base.scales:
            if cfg.name == "day":
                self.primary_scale = "day"
                break
        
        n = len(base)
        logger.info("Materializing %d samples ...", n)

        # 物化绝对 OHLCV 窗口
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
            if (i + 1) % 500_000 == 0 or i + 1 == n:
                logger.info("Materialized windows %d/%d", i + 1, n)

        self._tensors: dict[str, torch.Tensor] = {}
        for name, arr in scale_arrays.items():
            t = torch.from_numpy(arr)
            if share_memory:
                t = t.share_memory_()
            self._tensors[name] = t

        # 物化 symbol ids 和 timestamps
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

        # 核心优化：在物化阶段一次性计算全部 24 维特征
        self._features: dict[str, torch.Tensor] = {}
        if precompute_features and "day" in self._tensors:
            logger.info("Precomputing %d-dim day features (%d samples) ...", DAY_FULL_FEAT_DIM, n)
            day = self._tensors["day"]
            feat_chunks = []
            for start in range(0, n, self.PROXY_CHUNK):
                end = min(start + self.PROXY_CHUNK, n)
                chunk = day[start:end].float()
                if torch.cuda.is_available():
                    chunk = chunk.cuda()
                feats = _compute_features_vectorized(chunk)
                feat_chunks.append(feats.cpu())
                if end % 500_000 == 0 or end == n:
                    logger.info("Precomputed features %d/%d", end, n)
            
            features = torch.cat(feat_chunks, dim=0)
            if share_memory:
                features = features.share_memory_()
            self._features["day_features"] = features

            if "week" in self._tensors:
                logger.info("Precomputing week 7-dim features (%d samples) ...", n)
                week = self._tensors["week"]
                week_chunks = []
                for start in range(0, n, self.PROXY_CHUNK):
                    end = min(start + self.PROXY_CHUNK, n)
                    chunk = week[start:end].float()
                    if torch.cuda.is_available():
                        chunk = chunk.cuda()
                    week_chunks.append(_compute_week_features_vectorized(chunk).cpu())
                week_feats = torch.cat(week_chunks, dim=0)
                if share_memory:
                    week_feats = week_feats.share_memory_()
                self._features["week_features"] = week_feats

        self._proxies = None
        self._proxies_label_ready = False
        if precompute_proxies and "day" in self._tensors:
            logger.info("Precomputing behavior proxies (%d samples) ...", n)
            day = self._tensors["day"]
            proxy_chunks = []
            for start in range(0, n, self.PROXY_CHUNK):
                end = min(start + self.PROXY_CHUNK, n)
                chunk = day[start:end].float()
                if torch.cuda.is_available():
                    chunk = chunk.cuda()
                proxy_chunks.append(
                    transform_proxies_for_labeling(compute_behavior_proxies_stacked(chunk)).cpu()
                )
            proxies = torch.cat(proxy_chunks, dim=0)
            if share_memory:
                proxies = proxies.share_memory_()
            self._proxies = proxies
            self._proxies_label_ready = True

        logger.info(
            "Materialized dataset ready (%d samples, day_feat=%s, week_feat=%s, proxies=%s)",
            n,
            "day_features" in self._features,
            "week_features" in self._features,
            self._proxies is not None,
        )

    def __len__(self) -> int:
        return next(iter(self._tensors.values())).shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # 返回预计算特征，而非原始 OHLCV
        sample = {}
        if "day_features" in self._features:
            sample["day_features"] = self._features["day_features"][idx]
        else:
            sample["day"] = self._tensors["day"][idx]  # 回退到原始 OHLCV
        
        if "week_features" in self._features:
            sample["week_features"] = self._features["week_features"][idx]
        elif "week" in self._tensors:
            sample["week"] = self._tensors["week"][idx]

        sample["stock_ids"] = self._stock_ids[idx]
        sample["timestamps"] = self._timestamps[idx]
        if self._proxies is not None:
            sample["behavior_proxies"] = self._proxies[idx]
        return sample


# 磁盘持久化（扩展以支持预计算特征）
def save_materialized_dataset(ds: MaterializedMultiScaleDataset, path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    meta = {
        "scales": [(c.name, c.freq, c.lookback_window) for c in ds.scales],
        "primary_scale": ds.primary_scale,
        "has_proxies": ds._proxies is not None,
        "has_day_features": "day_features" in ds._features,
        "has_week_features": "week_features" in ds._features,
        "has_features": "day_features" in ds._features,
        "proxies_label_ready": getattr(ds, "_proxies_label_ready", False),
        "n_samples": len(ds),
        "day_feature_dim": DAY_FULL_FEAT_DIM,
        "num_behavior_agents": NUM_BEHAVIOR_AGENTS,
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2))

    def _safe_save(t: torch.Tensor, p: Path) -> None:
        np.save(p.with_suffix(".npy"), t.detach().cpu().numpy())

    for name, t in ds._tensors.items():
        _safe_save(t, path / f"{name}_raw.pt")
    
    if "day_features" in ds._features:
        _safe_save(ds._features["day_features"], path / "day_features.pt")
    if "week_features" in ds._features:
        _safe_save(ds._features["week_features"], path / "week_features.pt")

    _safe_save(ds._stock_ids, path / "stock_ids.pt")
    _safe_save(ds._timestamps, path / "timestamps.pt")

    if ds._proxies is not None:
        _safe_save(ds._proxies, path / "behavior_proxies.pt")

    logger.info("Materialized dataset saved to %s (%d samples)", path, len(ds))


def load_materialized_dataset(path: str | Path) -> MaterializedMultiScaleDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"预处理目录不存在: {path}")

    meta = json.loads((path / "meta.json").read_text())
    ds = MaterializedMultiScaleDataset.__new__(MaterializedMultiScaleDataset)
    ds.scales = []
    ds.primary_scale = meta["primary_scale"]

    def _load_field(stem: str) -> torch.Tensor:
        npy_path = path / f"{stem}.npy"
        if npy_path.exists():
            return torch.from_numpy(np.load(npy_path))
        raise FileNotFoundError(f"Missing {stem} in {path}")

    ds._tensors = {}
    for name, _, _ in meta["scales"]:
        try:
            ds._tensors[name] = _load_field(f"{name}_raw")
        except FileNotFoundError:
            pass  # 可能已预计算特征，无需原始数据

    ds._features = {}
    if meta.get("has_day_features") or meta.get("has_features"):
        ds._features["day_features"] = _load_field("day_features")
        feat_dim = ds._features["day_features"].shape[1]
        if feat_dim != DAY_FULL_FEAT_DIM:
            raise ValueError(
                f"day_features dim {feat_dim} != expected {DAY_FULL_FEAT_DIM} "
                f"(meta day_feature_dim={meta.get('day_feature_dim')}); "
                "re-run --save-preprocessed after behavior-proxy / feature schema update"
            )
    if meta.get("has_week_features"):
        try:
            ds._features["week_features"] = _load_field("week_features")
        except FileNotFoundError:
            logger.warning("week_features missing in %s — fusion will use day only", path)

    ds._stock_ids = _load_field("stock_ids")
    ds._timestamps = _load_field("timestamps")

    if meta.get("has_proxies"):
        ds._proxies = _load_field("behavior_proxies")
        if ds._proxies is not None and ds._proxies.shape[1] != NUM_BEHAVIOR_AGENTS:
            raise ValueError(
                f"behavior_proxies dim {ds._proxies.shape[1]} != {NUM_BEHAVIOR_AGENTS}; "
                "re-run --save-preprocessed after behavior-proxy schema update"
            )
    else:
        ds._proxies = None
    if meta.get("proxies_label_ready", False):
        ds._proxies_label_ready = True
    elif ds._proxies is not None:
        ds._proxies_label_ready = True
    else:
        ds._proxies_label_ready = False

    logger.info("Loaded pre-materialized dataset from %s (%d samples)", path, len(ds))
    return ds


class GpuCachedDataset(Dataset):
    """GPU 驻留数据集：一次性上传，训练时零拷贝切片。"""

    def __init__(self, base: MaterializedMultiScaleDataset, device: str = "cuda"):
        self.scales = base.scales
        self.primary_scale = base.primary_scale
        dev = torch.device(device)
        logger.info("Moving %d samples to %s (one-time upload) ...", len(base), dev)
        
        # 上传预计算特征而非原始 OHLCV
        self._features = {}
        self._tensors = {}
        if "day_features" in base._features:
            for key in ("day_features", "week_features"):
                if key in base._features:
                    self._features[key] = base._features[key].to(dev, non_blocking=True)
        else:
            self._tensors = {name: t.to(dev, non_blocking=True) for name, t in base._tensors.items()}

        self._stock_ids = base._stock_ids.to(dev, non_blocking=True)
        self._timestamps = base._timestamps.to(dev, non_blocking=True)
        self._proxies = base._proxies.to(dev, non_blocking=True) if base._proxies is not None else None
        self._proxies_label_ready = getattr(base, "_proxies_label_ready", False)
        
        if dev.type == "cuda":
            torch.cuda.synchronize()
        logger.info("GPU cache ready on %s", dev)

    def __len__(self) -> int:
        if "day_features" in self._features:
            return self._features["day_features"].shape[0]
        return next(iter(self._tensors.values())).shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = {}
        if "day_features" in self._features:
            sample["day_features"] = self._features["day_features"][idx]
            if "week_features" in self._features:
                sample["week_features"] = self._features["week_features"][idx]
        else:
            sample = {name: t[idx] for name, t in self._tensors.items()}

        sample["stock_ids"] = self._stock_ids[idx]
        sample["timestamps"] = self._timestamps[idx]
        if self._proxies is not None:
            sample["behavior_proxies"] = self._proxies[idx]
        return sample


def pin_dataset_share_memory(ds: MaterializedMultiScaleDataset) -> None:
    """Pin materialized tensors into shared memory for fork-based DataLoader workers."""

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


class ContiguousBatchDataset(Dataset):
    """
    CPU contiguous batch slices for multi-worker DataLoader prefetch.

    Each __getitem__ returns one full batch dict; train.py wraps this with
    DataLoader(batch_size=1, collate_fn=_first_collate).
    """

    def __init__(
        self,
        base: MaterializedMultiScaleDataset,
        *,
        batch_size: int,
        drop_last: bool = True,
    ):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.n = len(base)
        self._proxies_label_ready = getattr(base, "_proxies_label_ready", False)

        self._field_names: list[str] = []
        self._field_tensors: list[torch.Tensor] = []

        if "day_features" in base._features:
            self._field_names.append("day_features")
            self._field_tensors.append(base._features["day_features"])
        elif "day" in base._tensors:
            self._field_names.append("day")
            self._field_tensors.append(base._tensors["day"])

        if "week_features" in base._features:
            self._field_names.append("week_features")
            self._field_tensors.append(base._features["week_features"])
        elif "week" in base._tensors:
            self._field_names.append("week")
            self._field_tensors.append(base._tensors["week"])

        self._field_names.extend(["stock_ids", "timestamps"])
        self._field_tensors.extend([base._stock_ids, base._timestamps])

        if base._proxies is not None:
            self._field_names.append("behavior_proxies")
            self._field_tensors.append(base._proxies)

        self.num_batches = (
            self.n // batch_size if drop_last else (self.n + batch_size - 1) // batch_size
        )
        logger.info(
            "ContiguousBatchDataset: %d batches @ %d (%d samples, fields=%s)",
            self.num_batches,
            batch_size,
            self.n,
            self._field_names,
        )

    def __len__(self) -> int:
        return self.num_batches

    def __getitem__(self, batch_idx: int) -> dict[str, torch.Tensor]:
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, self.n)
        return {name: t[start:end] for name, t in zip(self._field_names, self._field_tensors)}


class BatchedGpuDataset(Dataset):
    """
    GPU-native contiguous batch dataset.
    训练循环直接调用 iter_batches()，绕过 DataLoader 的 per-sample 开销。
    """

    def __init__(
        self,
        base: MaterializedMultiScaleDataset | GpuCachedDataset,
        device: str = "cuda",
        batch_size: int = 4096,
        drop_last: bool = True,
        shuffle: bool = True,
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # 确定数据源
        if isinstance(base, GpuCachedDataset):
            self._features = base._features
            self._tensors = getattr(base, '_tensors', {})
            self._stock_ids = base._stock_ids
            self._timestamps = base._timestamps
            self._proxies = base._proxies
            self._proxies_label_ready = getattr(base, "_proxies_label_ready", False)
            self.n = len(base)
        else:
            self._features = {}
            for key in ("day_features", "week_features"):
                if key in base._features:
                    self._features[key] = base._features[key].to(device, non_blocking=True)
            self._tensors = {name: t.to(device, non_blocking=True) for name, t in base._tensors.items()}
            self._stock_ids = base._stock_ids.to(device, non_blocking=True)
            self._timestamps = base._timestamps.to(device, non_blocking=True)
            self._proxies = base._proxies.to(device, non_blocking=True) if base._proxies is not None else None
            self._proxies_label_ready = getattr(base, "_proxies_label_ready", False)
            self.n = len(base)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

        self.num_batches = self.n // batch_size if drop_last else (self.n + batch_size - 1) // batch_size
        
        # 构建字段名列表（用于 get_batch 的切片）
        self._field_names: list[str] = []
        self._field_tensors: list[torch.Tensor] = []
        
        if "day_features" in self._features:
            self._field_names.append("day_features")
            self._field_tensors.append(self._features["day_features"])
        elif "day" in self._tensors:
            self._field_names.append("day")
            self._field_tensors.append(self._tensors["day"])
        
        if "week_features" in self._features:
            self._field_names.append("week_features")
            self._field_tensors.append(self._features["week_features"])
        elif "week" in self._tensors:
            self._field_names.append("week")
            self._field_tensors.append(self._tensors["week"])
        
        self._field_names.extend(["stock_ids", "timestamps"])
        self._field_tensors.extend([self._stock_ids, self._timestamps])
        
        if self._proxies is not None:
            self._field_names.append("behavior_proxies")
            self._field_tensors.append(self._proxies)

        self._batch_order: list[int] = list(range(self.num_batches))
        if shuffle:
            self.on_epoch_begin()

        logger.info(
            "BatchedGpuDataset: %d batches @ %d on %s (fields=%s)",
            self.num_batches,
            batch_size,
            self.device,
            self._field_names,
        )

    def on_epoch_begin(self) -> None:
        if self.shuffle:
            self._batch_order = torch.randperm(self.num_batches).tolist()
        else:
            self._batch_order = list(range(self.num_batches))

    def __len__(self) -> int:
        return self.num_batches

    def get_batch(self, epoch_step: int) -> dict[str, torch.Tensor]:
        bi = self._batch_order[epoch_step]
        start = bi * self.batch_size
        end = min(start + self.batch_size, self.n)
        return {name: t[start:end] for name, t in zip(self._field_names, self._field_tensors)}

    def iter_batches(self):
        for i in range(self.num_batches):
            yield self.get_batch(i)

    def __getitem__(self, batch_idx: int) -> dict[str, torch.Tensor]:
        return self.get_batch(batch_idx)


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
    precompute_features: bool = True,
    precompute_proxies: bool = True,
    **kwargs,
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
            train_base,
            share_memory=share_memory,
            precompute_features=precompute_features,
            precompute_proxies=precompute_proxies,
        )
        val_ds: Dataset = MaterializedMultiScaleDataset(
            val_base,
            share_memory=share_memory,
            precompute_features=precompute_features,
            precompute_proxies=precompute_proxies,
        )
    else:
        train_ds = train_base
        val_ds = val_base
    return store, train_ds, val_ds, split