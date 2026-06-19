from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_cursor.config import Config, load_config
from quant_cursor.derivatives_universe import CALENDAR_QLIB_ID, EVENT_CALENDAR_FIELDS

_QLIB_INITIALIZED = False


def init_qlib(config: Config | None = None, provider_uri: str | Path | None = None) -> None:
    """初始化 Qlib（幂等）。"""
    global _QLIB_INITIALIZED
    if _QLIB_INITIALIZED:
        return

    import qlib
    from qlib.constant import REG_CN

    cfg = config or load_config()
    uri = str(provider_uri or cfg.qlib_data_dir.resolve())
    qlib.init(provider_uri=uri, region=REG_CN)
    _QLIB_INITIALIZED = True


def _ensure_fields(fields: list[str]) -> list[str]:
    return [f if f.startswith("$") else f"${f}" for f in fields]


def load_features(
    instruments: list[str],
    fields: list[str],
    start: str,
    end: str,
    *,
    config: Config | None = None,
    freq: str = "day",
) -> pd.DataFrame:
    """统一从 Qlib 加载行情特征。"""
    init_qlib(config)
    from qlib.data import D

    return D.features(instruments, _ensure_fields(fields), start, end, freq=freq)


def load_financial_futures(
    qlib_ids: list[str] | None = None,
    fields: list[str] | None = None,
    start: str = "2010-01-01",
    end: str = "2030-12-31",
    *,
    config: Config | None = None,
) -> pd.DataFrame:
    """加载金融期货主力连续（FF_IF 等）。"""
    from quant_cursor.derivatives_universe import FINANCIAL_FUTURES

    ids = qlib_ids or [s.qlib_id for s in FINANCIAL_FUTURES]
    flds = fields or ["close", "volume", "open", "high", "low"]
    return load_features(ids, flds, start, end, config=config)


def load_derivative_event_flags(
    start: str,
    end: str,
    *,
    fields: list[str] | None = None,
    config: Config | None = None,
) -> pd.DataFrame:
    """从 Qlib 读取衍生品事件日历标志（CAL_DERIV 伪标的）。"""
    flds = fields or list(EVENT_CALENDAR_FIELDS)
    df = load_features([CALENDAR_QLIB_ID], flds, start, end, config=config)
    # 便于 pandas 直接使用：同时保留 $ 前缀列与无前缀列
    rename = {c: c.lstrip("$") for c in df.columns if c.startswith("$")}
    return df.rename(columns=rename)


def load_derivative_events_detail(config: Config | None = None) -> pd.DataFrame:
    """读取完整事件明细表（含合约级字段，适合关联分析）。"""
    cfg = config or load_config()
    path = cfg.derivatives_events_path
    if not path.exists():
        raise FileNotFoundError(f"事件表不存在: {path}，请先 download-derivatives")
    return pd.read_parquet(path)


def join_event_flags_to_panel(
    panel: pd.DataFrame,
    *,
    start: str | None = None,
    end: str | None = None,
    event_fields: list[str] | None = None,
    config: Config | None = None,
) -> pd.DataFrame:
    """
    将 CAL_DERIV 事件标志合并到截面/面板数据。

    panel 需含 DatetimeIndex 或 `date` 列；其余列（如个股 close）保持不变。
    """
    if panel.empty:
        return panel

    out = panel.copy()
    if "date" in out.columns:
        dates = pd.to_datetime(out["date"])
    elif isinstance(out.index, pd.DatetimeIndex):
        dates = out.index
    else:
        raise ValueError("panel 需要 date 列或 DatetimeIndex")

    s = dates.min().strftime("%Y-%m-%d")
    e = dates.max().strftime("%Y-%m-%d")
    if start:
        s = start
    if end:
        e = end

    flags = load_derivative_event_flags(s, e, fields=event_fields, config=config)
    flags = flags.droplevel(level="instrument") if isinstance(flags.index, pd.MultiIndex) else flags
    flags.index = pd.to_datetime(flags.index)
    flags = flags.reset_index().rename(columns={"datetime": "date", "index": "date"})
    if "date" not in flags.columns and flags.index.name:
        flags = flags.reset_index().rename(columns={flags.columns[0]: "date"})

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
        return out.merge(flags.reset_index(), on="date", how="left")

    return out.join(flags, how="left")


def list_50etf_expiry_dates(config: Config | None = None) -> pd.Series:
    """50ETF 期权到期日列表（去重排序）。"""
    events = load_derivative_events_detail(config)
    mask = (events["product"] == "50ETF") & (events["exchange"] == "SSE") & (events["event_type"] == "expiry")
    return events.loc[mask, "event_date"].drop_duplicates().sort_values().reset_index(drop=True)
