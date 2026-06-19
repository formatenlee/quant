from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd

from quant_cursor.config import Config
from quant_cursor.rate_limit import RateLimitedClient
from quant_cursor.utils import amount_coverage, format_yyyymmdd, merge_history, normalize_ohlcv

logger = logging.getLogger(__name__)

_NUMERIC_INDEX = re.compile(r"^\d{6}$")


def _is_sz_index(code: str) -> bool:
    return code.startswith(("399", "980"))


def _should_use_csindex(code: str) -> bool:
    if not _NUMERIC_INDEX.match(code):
        return True
    return not _is_sz_index(code)


class MarketDataDownloader:
    """指数、ETF、A 股行情下载器，支持全量与增量。"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = RateLimitedClient(
            delay=config.request_delay,
            jitter=config.request_jitter,
            max_retries=config.max_retries,
            backoff=config.retry_backoff,
            batch_pause_every=config.batch_pause_every,
            batch_pause_seconds=config.batch_pause_seconds,
            ban_consecutive_failures=config.ban_consecutive_failures,
            ban_cooldown_seconds=config.ban_cooldown_seconds,
        )
        self.stock_client = RateLimitedClient(
            delay=config.stock_request_delay,
            jitter=config.request_jitter,
            max_retries=config.max_retries,
            backoff=config.retry_backoff,
            batch_pause_every=config.stock_batch_pause_every,
            batch_pause_seconds=config.stock_batch_pause_seconds,
            ban_consecutive_failures=config.ban_consecutive_failures,
            ban_cooldown_seconds=config.ban_cooldown_seconds,
        )
        config.ensure_dirs()

    def _data_path(self, asset_type: str, code: str) -> Path:
        if asset_type == "index":
            base = self.config.indices_dir
        elif asset_type == "etf":
            base = self.config.etf_dir
        elif asset_type == "stock":
            base = self.config.stocks_dir
        else:
            raise ValueError(f"未知 asset_type: {asset_type}")
        return base / f"{code}.parquet"

    def _load_local(self, asset_type: str, code: str) -> pd.DataFrame | None:
        path = self._data_path(asset_type, code)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def _save(self, df: pd.DataFrame, asset_type: str, code: str) -> Path:
        path = self._data_path(asset_type, code)
        meta_cols = ["code", "name", "symbol", "category", "sw_l2_code", "sw_l2_name"]
        out = df.copy()
        for col in meta_cols:
            if col in out.columns:
                out = out.drop(columns=[col])
        if "factor" not in out.columns:
            out["factor"] = 1.0
        out.to_parquet(path, index=False)
        return path

    def _today_end(self) -> str:
        tz = ZoneInfo(self.config.timezone)
        return format_yyyymmdd(pd.Timestamp(datetime.now(tz).date()))

    def _fetch_range(self, existing: pd.DataFrame | None, mode: str) -> tuple[str, str]:
        end = self._today_end()
        if mode == "full" or existing is None or existing.empty:
            return "19900101", end
        last = pd.to_datetime(existing["date"]).max()
        overlap = self.config.incremental_overlap_days
        start = format_yyyymmdd(last - pd.Timedelta(days=overlap))
        return start, end

    def _fetch_csindex_history(
        self, code: str, start_date: str = "19900101", end_date: str = "20500101"
    ) -> pd.DataFrame:
        raw = self.client.call(
            ak.stock_zh_index_hist_csindex,
            symbol=code,
            start_date=start_date,
            end_date=end_date,
        )
        return normalize_ohlcv(raw, "index")

    def _fetch_sina_index_history(self, symbol: str) -> pd.DataFrame:
        raw = self.client.call(ak.stock_zh_index_daily, symbol=symbol)
        return normalize_ohlcv(raw, "index")

    def _fetch_em_index_history(
        self, symbol: str, start_date: str = "19900101", end_date: str = "20500101"
    ) -> pd.DataFrame:
        raw = self.client.call(
            ak.stock_zh_index_daily_em,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
        return normalize_ohlcv(raw, "index")

    def _fetch_index_history(
        self, code: str, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        errors: list[str] = []

        if _should_use_csindex(code):
            try:
                return self._fetch_csindex_history(code, start_date, end_date)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"csindex: {exc}")
                self.client.pause(
                    self.config.fallback_delay,
                    f"中证接口失败 {code}，切换东财",
                )

        if _NUMERIC_INDEX.match(code):
            try:
                return self._fetch_em_index_history(symbol, start_date, end_date)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"em: {exc}")
                self.client.pause(
                    self.config.fallback_delay,
                    f"东财指数失败 {code}，切换新浪",
                )
            if start_date <= "19900110":
                try:
                    return self._fetch_sina_index_history(symbol)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"sina: {exc}")

        raise RuntimeError(" / ".join(errors) if errors else f"无法下载指数 {code}")

    def _fetch_etf_history(
        self, symbol: str, code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        # 优先东财（含 amount / 换手率等扩展字段）
        try:
            raw = self.client.call(
                ak.fund_etf_hist_em,
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="",
            )
            return normalize_ohlcv(raw, "etf")
        except Exception as em_exc:  # noqa: BLE001
            logger.debug("东财 ETF %s 失败，尝试新浪: %s", code, em_exc)
            self.client.pause(self.config.fallback_delay, "东财 ETF 失败，切换新浪")

        raw = self.client.call(ak.fund_etf_hist_sina, symbol=symbol)
        return normalize_ohlcv(raw, "etf")

    def _fetch_stock_history(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """下载 A 股：存未复权 OHLCV + factor（前复权 close = close * factor）。"""
        adjust = self.config.adjust_mode or "qfq"
        client = self.stock_client

        raw = client.call(
            ak.stock_zh_a_hist,
            symbol=code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="",
        )
        raw_n = normalize_ohlcv(raw, "stock")
        if raw_n.empty:
            return raw_n

        if adjust == "qfq":
            qfq = client.call(
                ak.stock_zh_a_hist,
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",
            )
            qfq_n = normalize_ohlcv(qfq, "stock")
            merged = raw_n.merge(qfq_n[["date", "close"]], on="date", suffixes=("", "_qfq"))
            merged["factor"] = merged["close_qfq"] / merged["close"].replace(0, np.nan)
            merged["factor"] = merged["factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
            merged = merged.drop(columns=["close_qfq"])
        else:
            merged = raw_n
            merged["factor"] = 1.0

        return merged

    def download_symbol(
        self,
        row: pd.Series,
        mode: str = "full",
        skip_existing: bool = False,
    ) -> tuple[bool, str]:
        code = str(row["code"]).zfill(6)
        symbol = row["symbol"]
        asset_type = row["asset_type"]
        name = row["name"]

        existing = self._load_local(asset_type, code)
        if skip_existing and existing is not None and not existing.empty:
            last_date = existing["date"].max()
            return True, f"已跳过 {code} 最新={last_date.date()}"

        try:
            start_date, end_date = self._fetch_range(existing, mode)

            if asset_type == "index":
                fetched = self._fetch_index_history(code, symbol, start_date, end_date)
            elif asset_type == "etf":
                fetched = self._fetch_etf_history(symbol, code, start_date, end_date)
            elif asset_type == "stock":
                fetched = self._fetch_stock_history(code, start_date, end_date)
            else:
                return False, f"未知 asset_type: {asset_type}"

            if fetched.empty:
                if existing is not None and not existing.empty:
                    return True, f"无新数据 最新={existing['date'].max().date()}"
                return False, "空数据"

            if mode in ("incremental", "today") and existing is not None and not existing.empty:
                merged = merge_history(existing, fetched)
            else:
                merged = fetched.copy()

            if "factor" not in merged.columns:
                merged["factor"] = 1.0
            else:
                merged["factor"] = pd.to_numeric(merged["factor"], errors="coerce").fillna(1.0)
            merged["code"] = code
            merged["name"] = name
            merged["symbol"] = symbol
            merged["category"] = row.get("category", "")

            path = self._save(merged, asset_type, code)
            last_date = merged["date"].max()
            action = "增量" if mode in ("incremental", "today") and existing is not None else "全量"
            return True, f"{action} {path.name} 最新={last_date.date()} rows={len(merged)}"

        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    def download_universe(
        self,
        universe: pd.DataFrame,
        mode: str = "full",
        categories: list[str] | None = None,
        asset_types: list[str] | None = None,
        codes: list[str] | None = None,
        skip_existing: bool = False,
    ) -> pd.DataFrame:
        df = universe.copy()

        if asset_types:
            df = df[df["asset_type"].isin(asset_types)]
        if categories:
            df = df[df["category"].isin(categories)]
        if codes:
            norm_codes = {c.zfill(6) for c in codes}
            df = df[df["code"].isin(norm_codes)]

        if skip_existing and mode == "full":
            pending_rows = []
            skipped = 0
            for _, row in df.iterrows():
                path = self._data_path(row["asset_type"], row["code"])
                if path.exists():
                    skipped += 1
                    continue
                pending_rows.append(row)
            if skipped:
                logger.info("跳过已有数据 %s 个", skipped)
            df = (
                pd.DataFrame(pending_rows)
                if pending_rows
                else pd.DataFrame(columns=universe.columns)
            )

        results: list[dict] = []
        total = len(df)
        if total == 0:
            logger.info("无需下载")
            return pd.DataFrame(
                columns=["code", "name", "asset_type", "category", "status", "message"]
            )

        logger.info("开始下载 [%s] 模式, 共 %s 个标的", mode, total)

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            ok, message = self.download_symbol(row, mode=mode, skip_existing=skip_existing)
            status = "ok" if ok else "fail"
            results.append(
                {
                    "code": row["code"],
                    "name": row["name"],
                    "asset_type": row["asset_type"],
                    "category": row.get("category", ""),
                    "status": status,
                    "message": message,
                }
            )
            level = logging.INFO if ok else logging.WARNING
            logger.log(
                level,
                "[%s/%s] %s %s %s",
                i,
                total,
                row["code"],
                row["name"],
                message,
            )

        report = pd.DataFrame(results)
        ok_count = (report["status"] == "ok").sum()
        logger.info("下载完成: 成功 %s / 失败 %s", ok_count, total - ok_count)

        failed = report[report["status"] == "fail"]
        if not failed.empty:
            fail_path = self.config.meta_dir / f"download_failures_{mode}.parquet"
            failed.to_parquet(fail_path, index=False)
            logger.warning("失败列表已保存: %s", fail_path)

        return report

    def find_symbols_needing_field_repair(
        self,
        universe: pd.DataFrame,
        *,
        min_amount_coverage: float = 0.05,
        asset_types: list[str] | None = None,
        include_extra_redownload: bool = False,
    ) -> pd.DataFrame:
        """找出 amount 缺失或（可选）扩展字段未落盘的标的。"""
        from quant_cursor.fields import QLIB_EXTRA_FIELDS

        types = asset_types or ["index", "etf", "stock"]
        rows: list[dict] = []
        for _, row in universe.iterrows():
            if row["asset_type"] not in types:
                continue
            code = str(row["code"]).zfill(6)
            path = self._data_path(row["asset_type"], code)
            if not path.exists():
                rows.append(
                    {
                        "code": code,
                        "name": row["name"],
                        "asset_type": row["asset_type"],
                        "category": row.get("category", ""),
                        "reason": "missing_file",
                        "amount_coverage": 0.0,
                    }
                )
                continue
            df = pd.read_parquet(path)
            cov = amount_coverage(df)
            missing_extra = [c for c in QLIB_EXTRA_FIELDS if c not in df.columns]
            amount_sparse = cov < min_amount_coverage
            extra_missing = include_extra_redownload and bool(missing_extra)
            if not amount_sparse and not extra_missing:
                continue
            reason = []
            if amount_sparse:
                reason.append("amount_sparse")
            if extra_missing:
                reason.append(f"missing:{','.join(missing_extra)}")
            rows.append(
                {
                    "code": code,
                    "name": row["name"],
                    "asset_type": row["asset_type"],
                    "category": row.get("category", ""),
                    "reason": "|".join(reason),
                    "amount_coverage": round(cov, 4),
                }
            )
        return pd.DataFrame(rows)

    def repair_extra_fields(
        self,
        universe: pd.DataFrame,
        *,
        asset_types: list[str] | None = None,
        codes: list[str] | None = None,
        min_amount_coverage: float = 0.05,
        include_extra_redownload: bool = False,
    ) -> pd.DataFrame:
        """全量重下 amount 缺失或（可选）扩展字段未落盘的标的。"""
        pending = self.find_symbols_needing_field_repair(
            universe,
            min_amount_coverage=min_amount_coverage,
            asset_types=asset_types,
            include_extra_redownload=include_extra_redownload,
        )
        if codes:
            norm = {c.zfill(6) for c in codes}
            pending = pending[pending["code"].isin(norm)]
        if pending.empty:
            logger.info("无需 repair 的标的")
            return pd.DataFrame(columns=["code", "name", "asset_type", "category", "status", "message"])

        code_set = set(pending["code"])
        subset = universe[universe["code"].astype(str).str.zfill(6).isin(code_set)]
        logger.info("repair 扩展字段: %s 个标的", len(subset))
        return self.download_universe(subset, mode="full", skip_existing=False)
