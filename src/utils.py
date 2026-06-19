from __future__ import annotations

import random
import time

import pandas as pd

from quant_cursor.fields import (
    AKSHARE_RENAME_MAP,
    NUMERIC_OHLCV_COLS,
    PARQUET_BASE_COLS,
    QLIB_EXTRA_FIELDS,
)


def index_code_to_symbol(code: str) -> str:
    """将 6 位指数代码转为新浪/东财接口使用的带前缀 symbol。"""
    code = str(code).strip().zfill(6)
    if code.startswith(("399", "980")):
        return f"sz{code}"
    return f"sh{code}"


def etf_code_to_symbol(code: str) -> str:
    """将 6 位 ETF 代码转为新浪接口 symbol。"""
    code = str(code).strip().zfill(6)
    if code.startswith(("15", "16", "18")):
        return f"sz{code}"
    return f"sh{code}"


def stock_code_to_symbol(code: str) -> str:
    """A 股代码转 symbol（与 ETF 规则相同）。"""
    return etf_code_to_symbol(code)


def format_yyyymmdd(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d")


def _empty_ohlcv_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(PARQUET_BASE_COLS))


def normalize_ohlcv(df: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    """统一 OHLCV + AKShare 扩展列名与类型。"""
    if df is None or df.empty:
        return _empty_ohlcv_frame()

    out = df.rename(columns=AKSHARE_RENAME_MAP).copy()

    # 新浪期货主力：列顺序固定 date, open, high, low, close, volume, ...
    if asset_type in ("financial_futures", "futures") and "close" not in out.columns:
        cols = list(df.columns)
        if len(cols) >= 6:
            out = df.iloc[:, :6].copy()
            out.columns = ["date", "open", "high", "low", "close", "volume"]

    if "date" not in out.columns and out.index.name == "date":
        out = out.reset_index()

    for col in NUMERIC_OHLCV_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date"])
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    for col in ("open", "high", "low", "close", "volume", "amount", *QLIB_EXTRA_FIELDS):
        if col not in out.columns:
            out[col] = pd.NA

    out["asset_type"] = asset_type
    if "factor" not in out.columns:
        out["factor"] = pd.NA

    keep = [c for c in PARQUET_BASE_COLS if c in out.columns]
    return out[keep]


def normalize_minute_ohlcv(df: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    """统一分钟 OHLCV 列名；date 保留到分钟（不做 normalize）。"""
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume", "amount", "asset_type"]
        )

    rename_map = {
        **AKSHARE_RENAME_MAP,
        "时间": "date",
        "day": "date",
    }
    out = df.rename(columns=rename_map).copy()
    if "date" not in out.columns and out.index.name == "date":
        out = out.reset_index()

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = pd.NA

    out["asset_type"] = asset_type
    cols = ["date", "open", "high", "low", "close", "volume", "amount", "asset_type"]
    if "factor" in out.columns:
        out["factor"] = pd.to_numeric(out["factor"], errors="coerce")
        cols.append("factor")
    return out[cols]


def amount_coverage(df: pd.DataFrame) -> float:
    """返回 amount 非空比例（0~1）。"""
    if df is None or df.empty or "amount" not in df.columns:
        return 0.0
    amt = pd.to_numeric(df["amount"], errors="coerce")
    return float(amt.notna().mean())


def merge_minute_history(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return new.copy()
    if new is None or new.empty:
        return existing.copy()

    merged = pd.concat([existing, new], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return merged.reset_index(drop=True)


def merge_history(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return new.copy()
    if new is None or new.empty:
        return existing.copy()

    merged = pd.concat([existing, new], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"]).dt.normalize()
    merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return merged.reset_index(drop=True)


def sleep_with_jitter(base_delay: float, jitter: float) -> None:
    delay = max(0.0, base_delay + random.uniform(-jitter, jitter))
    time.sleep(delay)
