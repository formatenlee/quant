from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd

from quant_cursor.config import Config
from quant_cursor.derivatives_events import build_derivatives_events, save_derivatives_events
from quant_cursor.derivatives_universe import futures_specs
from quant_cursor.rate_limit import RateLimitedClient
from quant_cursor.utils import format_yyyymmdd, merge_history, normalize_ohlcv

logger = logging.getLogger(__name__)


class DerivativesDownloader:
    """金融期货与期权事件数据下载（不含商品期货）。"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = RateLimitedClient(
            delay=config.request_delay,
            jitter=config.request_jitter,
            max_retries=config.max_retries,
            backoff=config.retry_backoff,
            batch_pause_every=config.batch_pause_every,
            batch_pause_seconds=config.batch_pause_seconds,
        )
        config.ensure_dirs()

    def _futures_path(self, code: str) -> Path:
        return self.config.derivatives_futures_dir / f"{code}.parquet"

    def _today_end(self) -> str:
        tz = ZoneInfo(self.config.timezone)
        return format_yyyymmdd(pd.Timestamp(datetime.now(tz).date()))

    def _fetch_range(self, existing: pd.DataFrame | None, mode: str) -> tuple[str, str]:
        end = self._today_end()
        if mode == "full" or existing is None or existing.empty:
            return "20100401", end
        last = pd.to_datetime(existing["date"]).max()
        start = format_yyyymmdd(last - pd.Timedelta(days=self.config.incremental_overlap_days))
        return start, end

    def _fetch_futures_main(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        def _pull():
            return ak.futures_main_sina(symbol=symbol, start_date=start, end_date=end)

        raw = self.client.call(_pull)
        frame = normalize_ohlcv(raw, "financial_futures")
        if "factor" not in frame.columns:
            frame["factor"] = 1.0
        return frame

    def download_futures(
        self,
        *,
        mode: str = "incremental",
        skip_existing: bool = False,
        codes: list[str] | None = None,
    ) -> pd.DataFrame:
        specs = futures_specs()
        if codes:
            codes_set = {c.upper() for c in codes}
            specs = [s for s in specs if s.code.upper() in codes_set or s.qlib_id.upper() in codes_set]

        records: list[dict] = []
        for spec in specs:
            path = self._futures_path(spec.code)
            if skip_existing and path.exists() and mode == "full":
                records.append(
                    {"code": spec.code, "qlib_id": spec.qlib_id, "status": "skip", "rows": 0, "message": "exists"}
                )
                continue

            existing = pd.read_parquet(path) if path.exists() else None
            start, end = self._fetch_range(existing, mode)
            try:
                chunk = self._fetch_futures_main(spec.ak_symbol or spec.code, start, end)
                if chunk.empty and existing is not None:
                    merged = existing
                elif existing is not None and not chunk.empty:
                    merged = merge_history(existing, chunk)
                else:
                    merged = chunk

                if merged.empty:
                    records.append(
                        {
                            "code": spec.code,
                            "qlib_id": spec.qlib_id,
                            "status": "fail",
                            "rows": 0,
                            "message": "empty",
                        }
                    )
                    continue

                out = merged.copy()
                out.to_parquet(path, index=False)

                records.append(
                    {
                        "code": spec.code,
                        "qlib_id": spec.qlib_id,
                        "status": "ok",
                        "rows": len(out),
                        "message": f"{out['date'].min()} ~ {out['date'].max()}",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("期货 %s 下载失败: %s", spec.code, exc)
                records.append(
                    {
                        "code": spec.code,
                        "qlib_id": spec.qlib_id,
                        "status": "fail",
                        "rows": 0,
                        "message": str(exc)[:200],
                    }
                )

        return pd.DataFrame(records)

    def download_events(self) -> pd.DataFrame:
        events = build_derivatives_events(self.config, client=self.client)
        save_derivatives_events(self.config, events)
        return events

    def download_all(
        self,
        *,
        mode: str = "incremental",
        skip_existing: bool = False,
        codes: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        futures_report = self.download_futures(mode=mode, skip_existing=skip_existing, codes=codes)
        events = self.download_events()
        return futures_report, events
