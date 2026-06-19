from __future__ import annotations

import logging
from typing import Literal

import akshare as ak
import pandas as pd

from quant_cursor.config import Config
from quant_cursor.rate_limit import RateLimitedClient
from quant_cursor.utils import index_code_to_symbol, etf_code_to_symbol, normalize_minute_ohlcv

logger = logging.getLogger(__name__)

MinuteSource = Literal["sina", "em"]


def symbol_for_minute(code: str, asset_type: str) -> str:
    c = str(code).zfill(6)
    if asset_type == "etf":
        return etf_code_to_symbol(c)
    if asset_type == "index":
        return index_code_to_symbol(c)
    if asset_type == "stock":
        return etf_code_to_symbol(c)
    raise ValueError(f"未知 asset_type: {asset_type}")


def resolve_minute_sources(config: Config) -> list[MinuteSource]:
    src = (config.minute_data_source or "auto").strip().lower()
    if src == "auto":
        return ["sina", "em"]
    if src in ("sina", "em"):
        return [src]  # type: ignore[list-item]
    raise ValueError(f"未知 minute_data_source: {config.minute_data_source}，可选 sina/em/auto")


def _filter_datetime_range(
    df: pd.DataFrame, start_date: str, end_date: str
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return out[(out["date"] >= start) & (out["date"] <= end)].reset_index(drop=True)


class SinaMinuteProvider:
    """新浪 CN_MarketDataService.getKLineData — 支持 ETF/指数 1/5/15/30/60 分钟。"""

    def __init__(self, client: RateLimitedClient, period: str) -> None:
        self.client = client
        self.period = period

    def fetch(
        self,
        code: str,
        asset_type: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        symbol = symbol_for_minute(code, asset_type)
        raw = self.client.call(
            ak.stock_zh_a_minute,
            symbol=symbol,
            period=self.period,
            adjust="",
        )
        frame = normalize_minute_ohlcv(raw, asset_type)
        return _filter_datetime_range(frame, start_date, end_date)


class EastMoneyMinuteProvider:
    """东财 push2his — fund_etf_hist_min_em / index_zh_a_hist_min_em。"""

    def __init__(self, client: RateLimitedClient, period: str) -> None:
        self.client = client
        self.period = period

    def fetch_etf(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        raw = self.client.call(
            ak.fund_etf_hist_min_em,
            symbol=code,
            period=self.period,
            start_date=start_date,
            end_date=end_date,
            adjust="",
        )
        return normalize_minute_ohlcv(raw, "etf")

    def fetch_index(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        raw = self.client.call(
            ak.index_zh_a_hist_min_em,
            symbol=code,
            period=self.period,
            start_date=start_date,
            end_date=end_date,
        )
        return normalize_minute_ohlcv(raw, "index")

    def fetch(
        self,
        code: str,
        asset_type: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        if asset_type == "etf":
            return self.fetch_etf(code, start_date, end_date)
        if asset_type == "index":
            return self.fetch_index(code, start_date, end_date)
        raise ValueError(f"东财分钟暂不支持 asset_type={asset_type}")


def fetch_minute_with_fallback(
    config: Config,
    client: RateLimitedClient,
    *,
    code: str,
    asset_type: str,
    start_date: str,
    end_date: str,
    fallback_client: RateLimitedClient | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """按配置数据源顺序拉取，返回 (df, source_name)。"""
    sources = resolve_minute_sources(config)
    sina = SinaMinuteProvider(client, config.minute_period)
    em_client = fallback_client or client
    em = EastMoneyMinuteProvider(em_client, config.minute_period)
    errors: list[str] = []

    for i, src in enumerate(sources):
        try:
            if src == "sina":
                df = sina.fetch(code, asset_type, start_date, end_date)
            else:
                df = em.fetch(code, asset_type, start_date, end_date)
            if df is not None and not df.empty:
                return df, src
            errors.append(f"{src}: 空数据")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{src}: {exc}")
            logger.debug("分钟源 %s 失败 %s %s: %s", src, asset_type, code, exc)

    if errors:
        raise RuntimeError(" / ".join(errors))
    return pd.DataFrame(), None
