from __future__ import annotations

"""Qlib / parquet 扩展字段定义（AKShare 常见列映射后）。"""

# 核心 OHLCV（Qlib 默认）
QLIB_CORE_FIELDS: tuple[str, ...] = ("open", "close", "high", "low", "volume", "factor")

# AKShare 可补充的扩展字段
QLIB_EXTRA_FIELDS: tuple[str, ...] = (
    "amount",
    "turnover_rate",
    "amplitude",
    "pct_change",
    "change",
    "prev_close",
    "sample_count",
    "pe_ttm",
)

QLIB_ALL_FIELDS: tuple[str, ...] = QLIB_CORE_FIELDS + QLIB_EXTRA_FIELDS

# normalize_ohlcv 列名映射
AKSHARE_RENAME_MAP: dict[str, str] = {
    "日期": "date",
    "prevclose": "prev_close",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交金额": "amount",
    "成交额": "amount",
    "amount": "amount",
    "开盘价": "open",
    "收盘价": "close",
    "最高价": "high",
    "最低价": "low",
    "振幅": "amplitude",
    "涨跌幅": "pct_change",
    "涨跌额": "change",
    "涨跌": "change",
    "换手率": "turnover_rate",
    "样本数量": "sample_count",
    "滚动市盈率": "pe_ttm",
    "持仓量": "open_interest",
    "动态结算价": "settle",
}

NUMERIC_OHLCV_COLS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "prev_close",
    "amplitude",
    "pct_change",
    "change",
    "turnover_rate",
    "sample_count",
    "pe_ttm",
    "factor",
)

PARQUET_BASE_COLS: tuple[str, ...] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover_rate",
    "amplitude",
    "pct_change",
    "change",
    "prev_close",
    "sample_count",
    "pe_ttm",
    "factor",
    "asset_type",
)

WEEKLY_AGG: dict[str, str] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "amount": "sum",
    "factor": "last",
    "turnover_rate": "mean",
    "amplitude": "max",
    "pct_change": "sum",
    "change": "sum",
    "prev_close": "first",
    "sample_count": "last",
    "pe_ttm": "last",
}
