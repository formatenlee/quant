from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from quant_cursor.config import Config
from quant_cursor.minute_providers import fetch_minute_with_fallback, resolve_minute_sources
from quant_cursor.qlib_export import code_to_qlib_id
from quant_cursor.rate_limit import RateLimitedClient
from quant_cursor.utils import merge_minute_history

logger = logging.getLogger(__name__)

_NUMERIC_INDEX = re.compile(r"^\d{6}$")
SKIP_CODES_NAME = "minute_skip_codes.parquet"


def load_minute_skip_codes(config: Config) -> set[str]:
    path = config.meta_dir / SKIP_CODES_NAME
    if not path.exists():
        return set()
    df = pd.read_parquet(path)
    return set(df["code"].astype(str).str.zfill(6))


def save_minute_skip_code(config: Config, code: str, reason: str) -> None:
    path = config.meta_dir / SKIP_CODES_NAME
    config.meta_dir.mkdir(parents=True, exist_ok=True)
    code = str(code).zfill(6)
    row = pd.DataFrame([{"code": code, "reason": reason}])
    if path.exists():
        existing = pd.read_parquet(path)
        if code in existing["code"].astype(str).str.zfill(6).values:
            return
        out = pd.concat([existing, row], ignore_index=True)
    else:
        out = row
    out.drop_duplicates(subset=["code"], keep="last").to_parquet(path, index=False)


def build_minute_skip_codes(config: Config) -> set[str]:
    """合并历史失败报告与字母码，生成快速跳过列表。"""
    skip = load_minute_skip_codes(config)
    for report_path in config.meta_dir.glob("minute_download_report_*.parquet"):
        report = pd.read_parquet(report_path)
        failed = report[report["status"] == "fail"]
        skip.update(failed["code"].astype(str).str.zfill(6))
    manifest = pd.read_parquet(config.meta_dir / "qlib_manifest.parquet")
    alpha = manifest[~manifest["code"].astype(str).str.match(_NUMERIC_INDEX)]
    skip.update(alpha["code"].astype(str).str.zfill(6))
    return skip


def can_fetch_minute(code: str, asset_type: str) -> bool:
    """分钟接口仅支持 6 位数字代码的 ETF / 指数。"""
    raw = str(code).strip()
    if asset_type in ("etf", "index", "stock"):
        return bool(_NUMERIC_INDEX.match(raw.zfill(6)))
    return False


class MinuteDataDownloader:
    """为已入库 Qlib 的标的补充分钟级行情（缺失则跳过，支持新浪/东财）。"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = RateLimitedClient(
            delay=config.effective_minute_delay,
            jitter=config.request_jitter,
            max_retries=config.minute_max_retries,
            backoff=config.retry_backoff,
            batch_pause_every=config.batch_pause_every,
            batch_pause_seconds=config.batch_pause_seconds,
            ban_consecutive_failures=config.ban_consecutive_failures,
            ban_cooldown_seconds=config.ban_cooldown_seconds,
        )
        self.fallback_client = RateLimitedClient(
            delay=config.effective_minute_delay,
            jitter=config.request_jitter,
            max_retries=1,
            backoff=15.0,
            batch_pause_every=config.batch_pause_every,
            batch_pause_seconds=config.batch_pause_seconds,
            ban_consecutive_failures=config.ban_consecutive_failures,
            ban_cooldown_seconds=config.ban_cooldown_seconds,
        )
        config.ensure_dirs()

    def _minute_path(self, asset_type: str, code: str) -> Path:
        c = str(code).zfill(6)
        if asset_type == "index":
            return self.config.indices_min_dir / f"{c}.parquet"
        if asset_type == "etf":
            return self.config.etf_min_dir / f"{c}.parquet"
        if asset_type == "stock":
            return self.config.data_dir / "stocks_min" / f"{c}.parquet"
        raise ValueError(f"未知 asset_type: {asset_type}")

    def _load_local(self, asset_type: str, code: str) -> pd.DataFrame | None:
        path = self._minute_path(asset_type, code)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def _save(self, df: pd.DataFrame, asset_type: str, code: str) -> Path:
        path = self._minute_path(asset_type, code)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = df.copy()
        for col in ("code", "name", "symbol", "category", "qlib_id"):
            if col in out.columns:
                out = out.drop(columns=[col])
        if "factor" not in out.columns:
            out["factor"] = 1.0
        out.to_parquet(path, index=False)
        return path

    def _datetime_bounds(self, existing: pd.DataFrame | None, mode: str) -> tuple[str, str]:
        tz = ZoneInfo(self.config.timezone)
        now = datetime.now(tz)
        end = now.strftime("%Y-%m-%d %H:%M:%S")
        if mode == "full" or existing is None or existing.empty:
            return "1990-01-01 09:30:00", end
        last = pd.to_datetime(existing["date"]).max()
        overlap = pd.Timedelta(days=max(1, self.config.incremental_overlap_days))
        start = (last - overlap).strftime("%Y-%m-%d %H:%M:%S")
        return start, end

    def download_symbol(
        self,
        row: pd.Series,
        mode: str = "full",
        skip_existing: bool = False,
    ) -> tuple[bool, str]:
        code = str(row["code"]).zfill(6)
        asset_type = row["asset_type"]

        if not can_fetch_minute(code, asset_type):
            return True, f"跳过 {code} 非数字代码或无分钟源"

        existing = self._load_local(asset_type, code)
        if skip_existing and existing is not None and not existing.empty:
            last = pd.to_datetime(existing["date"]).max()
            return True, f"已跳过 {code} 分钟最新={last}"

        try:
            start_date, end_date = self._datetime_bounds(existing, mode)
            fetched, source = fetch_minute_with_fallback(
                self.config,
                self.client,
                code=code,
                asset_type=asset_type,
                start_date=start_date,
                end_date=end_date,
                fallback_client=self.fallback_client,
            )

            if fetched.empty:
                if existing is not None and not existing.empty:
                    return True, f"无新分钟数据 最新={existing['date'].max()}"
                return False, "空数据"

            if mode in ("incremental", "today") and existing is not None and not existing.empty:
                merged = merge_minute_history(existing, fetched)
            else:
                merged = fetched.copy()

            if "factor" not in merged.columns:
                merged["factor"] = 1.0
            else:
                merged["factor"] = pd.to_numeric(merged["factor"], errors="coerce").fillna(1.0)
            path = self._save(merged, asset_type, code)
            last = pd.to_datetime(merged["date"]).max()
            action = "增量" if mode in ("incremental", "today") and existing is not None else "全量"
            src_tag = f"[{source}]" if source else ""
            return True, f"{action}{src_tag} {path.name} 最新={last} rows={len(merged)}"

        except Exception as exc:  # noqa: BLE001
            save_minute_skip_code(self.config, code, str(exc))
            return False, str(exc)

    def _load_qlib_targets(
        self,
        *,
        pending_only: bool = False,
        skip_codes: set[str] | None = None,
    ) -> pd.DataFrame:
        manifest_path = self.config.meta_dir / "qlib_manifest.parquet"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Qlib 日线 manifest 不存在: {manifest_path}，请先执行 qlib export"
            )
        manifest = pd.read_parquet(manifest_path)
        universe_path = self.config.universe_path
        if universe_path.exists():
            universe = pd.read_parquet(universe_path)
            extra = [c for c in ("name", "symbol", "category") if c in universe.columns]
            if extra:
                manifest = manifest.merge(
                    universe[["code", *extra]].drop_duplicates("code"),
                    on="code",
                    how="left",
                    suffixes=("", "_u"),
                )
                for col in extra:
                    ucol = f"{col}_u"
                    if ucol in manifest.columns:
                        manifest[col] = manifest[col].fillna(manifest[ucol])
                        manifest = manifest.drop(columns=[ucol])
        if skip_codes:
            manifest = manifest[
                ~manifest["code"].astype(str).str.zfill(6).isin(skip_codes)
            ]
        if pending_only:
            pending_rows = []
            for _, row in manifest.iterrows():
                code = str(row["code"]).zfill(6)
                if self._load_local(row["asset_type"], code) is None:
                    pending_rows.append(row)
            manifest = (
                pd.DataFrame(pending_rows)
                if pending_rows
                else pd.DataFrame(columns=manifest.columns)
            )
        return manifest

    def download_qlib_universe(
        self,
        mode: str = "full",
        *,
        asset_types: list[str] | None = None,
        categories: list[str] | None = None,
        codes: list[str] | None = None,
        skip_existing: bool = False,
        pending_only: bool = False,
    ) -> pd.DataFrame:
        skip_codes = build_minute_skip_codes(self.config)
        df = self._load_qlib_targets(pending_only=pending_only, skip_codes=skip_codes)

        if asset_types:
            df = df[df["asset_type"].isin(asset_types)]
        if categories:
            df = df[df["category"].isin(categories)]
        if codes:
            norm = {c.zfill(6) for c in codes}
            df = df[df["code"].astype(str).str.zfill(6).isin(norm)]

        sources = resolve_minute_sources(self.config)
        results: list[dict] = []
        total = len(df)
        logger.info(
            "分钟下载 [%s] period=%s source=%s, 待处理 %s 个 (skip_list=%s)",
            mode,
            self.config.minute_period,
            "+".join(sources),
            total,
            len(skip_codes),
        )

        if total == 0:
            logger.info("分钟数据已全部下载或无可下载标的，跳过")
            return pd.DataFrame(
                columns=["code", "qlib_id", "name", "asset_type", "category", "status", "message"]
            )

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            ok, message = self.download_symbol(row, mode=mode, skip_existing=skip_existing)
            status = "ok" if ok else "fail"
            if "跳过" in message and ok:
                status = "skip"
            results.append(
                {
                    "code": str(row["code"]).zfill(6),
                    "qlib_id": row.get("qlib_id") or code_to_qlib_id(str(row["code"]), row["asset_type"]),
                    "name": row.get("name", ""),
                    "asset_type": row["asset_type"],
                    "category": row.get("category", ""),
                    "status": status,
                    "message": message,
                }
            )
            level = logging.INFO if status != "fail" else logging.WARNING
            logger.log(level, "[%s/%s] %s %s", i, total, row["code"], message)

        report = pd.DataFrame(results)
        ok_count = (report["status"] == "ok").sum()
        skip_count = (report["status"] == "skip").sum()
        fail_count = (report["status"] == "fail").sum()
        logger.info(
            "分钟下载完成: 成功 %s / 跳过 %s / 失败 %s（失败已忽略）",
            ok_count,
            skip_count,
            fail_count,
        )

        out = self.config.meta_dir / f"minute_download_report_{mode}.parquet"
        report.to_parquet(out, index=False)
        return report
