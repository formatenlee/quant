from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant_cursor.config import Config
from quant_cursor.qlib_export import code_to_qlib_id
from quant_cursor.utils import normalize_ohlcv, stock_code_to_symbol

logger = logging.getLogger(__name__)

REQUIRED_COLS = {"date", "open", "high", "low", "close", "volume"}


@dataclass
class ValidationResult:
    ok: bool
    level: str  # ok | warn | error
    message: str


def _data_dir_for(config: Config, asset_type: str) -> Path:
    if asset_type == "index":
        return config.indices_dir
    if asset_type == "etf":
        return config.etf_dir
    if asset_type == "stock":
        return config.stocks_dir
    raise ValueError(f"未知 asset_type: {asset_type}")


def parquet_path(config: Config, asset_type: str, code: str) -> Path:
    return _data_dir_for(config, asset_type) / f"{code}.parquet"


def validate_parquet(
    path: Path,
    *,
    name: str = "",
    require_factor: bool = False,
) -> ValidationResult:
    if not path.exists():
        return ValidationResult(False, "error", "文件不存在")
    size = path.stat().st_size
    if size < 100:
        return ValidationResult(False, "error", f"文件过小 ({size} bytes)")

    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(False, "error", f"无法读取: {exc}")

    if df.empty:
        return ValidationResult(False, "error", "数据为空")

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        return ValidationResult(False, "error", f"缺少列: {missing}")
    if require_factor and "factor" not in df.columns:
        return ValidationResult(False, "error", "缺少 factor 列（A股需前复权因子）")

    if df["date"].isna().all():
        return ValidationResult(False, "error", "日期全为空")
    if df["close"].isna().all():
        return ValidationResult(False, "error", "收盘价全为空")

    price = df[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    bad = price.le(0).any(axis=1) | (df["high"] < df["low"])
    if bad.any():
        return ValidationResult(False, "error", f"OHLC 异常行数={int(bad.sum())}")

    if "factor" in df.columns:
        factor = pd.to_numeric(df["factor"], errors="coerce")
        if factor.isna().all():
            return ValidationResult(False, "error", "factor 全为空")
        if (factor <= 0).any():
            return ValidationResult(False, "error", "factor 存在非正值")

    rows = len(df)
    last = pd.to_datetime(df["date"]).max()
    if rows < 5:
        tag = "新上市" if name.startswith("N") or "新" in name else "历史较短"
        return ValidationResult(True, "warn", f"{tag} rows={rows} last={last.date()}")
    return ValidationResult(True, "ok", f"rows={rows} last={last.date()}")


def validate_universe(
    config: Config,
    universe: pd.DataFrame | None = None,
    *,
    require_stock_factor: bool = True,
) -> pd.DataFrame:
    """校验 universe 中全部本地 parquet，返回明细报告。"""
    if universe is None:
        if not config.universe_path.exists():
            raise FileNotFoundError(config.universe_path)
        universe = pd.read_parquet(config.universe_path)

    rows: list[dict] = []
    for _, item in universe.iterrows():
        asset_type = item["asset_type"]
        code = str(item["code"]).zfill(6)
        path = parquet_path(config, asset_type, code)
        need_factor = require_stock_factor and asset_type == "stock"
        result = validate_parquet(path, name=str(item["name"]), require_factor=need_factor)
        rows.append(
            {
                "code": code,
                "name": item["name"],
                "asset_type": asset_type,
                "category": item.get("category", ""),
                "path": str(path),
                "status": result.level,
                "message": result.message,
            }
        )

    report = pd.DataFrame(rows)
    config.meta_dir.mkdir(parents=True, exist_ok=True)

    invalid = report[report["status"] == "error"]
    warns = report[report["status"] == "warn"]
    if not invalid.empty:
        invalid.to_parquet(config.meta_dir / "invalid_data.parquet", index=False)
        logger.warning("无效数据 %s 条 -> invalid_data.parquet", len(invalid))
    else:
        invalid_path = config.meta_dir / "invalid_data.parquet"
        if invalid_path.exists():
            invalid_path.unlink()

    if not warns.empty:
        warns.to_parquet(config.meta_dir / "warn_data.parquet", index=False)

    ok_n = (report["status"] == "ok").sum()
    logger.info(
        "校验完成: ok=%s warn=%s error=%s / total=%s",
        ok_n,
        len(warns),
        len(invalid),
        len(report),
    )
    return report


def _sample_compare_dates(local: pd.DataFrame, n: int = 5) -> list[pd.Timestamp]:
    dates = pd.to_datetime(local["date"]).dropna().sort_values()
    if len(dates) <= n:
        return dates.tolist()
    picks = sorted(random.sample(range(len(dates)), n))
    return [dates[i] for i in picks]


def cross_check_qlib_vs_source(
    config: Config,
    *,
    sample_n: int = 20,
    seed: int = 42,
    tol: float = 1e-3,
) -> pd.DataFrame:
    """
    抽样对比 Qlib 读取值与本地 parquet（前复权 close = close * factor）。
    """
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    manifest_path = config.meta_dir / "qlib_manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = pd.read_parquet(manifest_path)
    qlib_dir = config.qlib_data_dir
    if not qlib_dir.exists():
        raise FileNotFoundError(f"Qlib 目录不存在: {qlib_dir}")

    qlib.init(provider_uri=str(qlib_dir.resolve()), region=REG_CN)

    rng = random.Random(seed)
    sample = manifest.sample(n=min(sample_n, len(manifest)), random_state=seed)
    rows: list[dict] = []

    for _, row in sample.iterrows():
        code = str(row["code"]).zfill(6)
        asset_type = row["asset_type"]
        qlib_id = row["qlib_id"]
        path = parquet_path(config, asset_type, code)
        if not path.exists():
            rows.append({"code": code, "qlib_id": qlib_id, "status": "error", "message": "本地文件缺失"})
            continue

        local = pd.read_parquet(path)
        local["date"] = pd.to_datetime(local["date"])
        factor = pd.to_numeric(local.get("factor", 1.0), errors="coerce").fillna(1.0)
        local_close = pd.to_numeric(local["close"], errors="coerce")
        local_adj = local_close * factor

        check_dates = _sample_compare_dates(local, n=5)
        if not check_dates:
            rows.append({"code": code, "qlib_id": qlib_id, "status": "error", "message": "无有效日期"})
            continue

        start = min(check_dates).strftime("%Y-%m-%d")
        end = max(check_dates).strftime("%Y-%m-%d")
        try:
            qdf = D.features(
                [qlib_id],
                ["$close", "$factor"],
                start_time=start,
                end_time=end,
                freq="day",
            )
            qdf = qdf.droplevel(0)
            qdf.index = pd.to_datetime(qdf.index)
            qdf["adj"] = qdf["$close"] * qdf["$factor"].fillna(1.0)
        except Exception as exc:  # noqa: BLE001
            rows.append({"code": code, "qlib_id": qlib_id, "status": "error", "message": f"Qlib 读取失败: {exc}"})
            continue

        max_diff = 0.0
        for dt in check_dates:
            if dt not in qdf.index:
                continue
            loc_row = local[local["date"] == dt]
            if loc_row.empty:
                continue
            lv = float(local_adj.iloc[loc_row.index[0]])
            qv = float(qdf.loc[dt, "adj"])
            if np.isfinite(lv) and np.isfinite(qv):
                max_diff = max(max_diff, abs(lv - qv) / max(abs(lv), 1e-6))

        status = "ok" if max_diff <= tol else "error"
        rows.append(
            {
                "code": code,
                "qlib_id": qlib_id,
                "asset_type": asset_type,
                "status": status,
                "max_rel_diff": max_diff,
                "message": f"max_rel_diff={max_diff:.6f}",
            }
        )

    report = pd.DataFrame(rows)
    out = config.meta_dir / "qlib_crosscheck_report.parquet"
    report.to_parquet(out, index=False)
    ok_n = (report["status"] == "ok").sum()
    logger.info("Qlib 抽样对账: ok=%s / %s -> %s", ok_n, len(report), out)
    return report


def verify_akshare_factor_sample(
    config: Config,
    *,
    sample_n: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """对 A 股抽样：本地 factor 是否与 akshare 前复权重算一致。"""
    import akshare as ak

    from quant_cursor.rate_limit import RateLimitedClient

    if not config.universe_path.exists():
        raise FileNotFoundError(config.universe_path)
    universe = pd.read_parquet(config.universe_path)
    stocks = universe[universe["asset_type"] == "stock"]
    if stocks.empty:
        logger.info("无 A 股标的，跳过 akshare factor 抽样")
        return pd.DataFrame()

    client = RateLimitedClient(delay=config.stock_request_delay, max_retries=2)
    sample = stocks.sample(n=min(sample_n, len(stocks)), random_state=seed)
    rows: list[dict] = []

    for _, row in sample.iterrows():
        code = str(row["code"]).zfill(6)
        path = parquet_path(config, "stock", code)
        if not path.exists():
            continue
        local = pd.read_parquet(path)
        local["date"] = pd.to_datetime(local["date"])
        tail = local.tail(30)
        start = tail["date"].min().strftime("%Y%m%d")
        end = tail["date"].max().strftime("%Y%m%d")

        try:
            raw = client.call(
                ak.stock_zh_a_hist,
                symbol=code,
                period="daily",
                start_date=start,
                end_date=end,
                adjust="",
            )
            qfq = client.call(
                ak.stock_zh_a_hist,
                symbol=code,
                period="daily",
                start_date=start,
                end_date=end,
                adjust="qfq",
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"code": code, "status": "error", "message": str(exc)})
            continue

        raw_n = normalize_ohlcv(raw, "stock")
        qfq_n = normalize_ohlcv(qfq, "stock")
        merged = raw_n.merge(qfq_n, on="date", suffixes=("", "_qfq"))
        merged["expected_factor"] = merged["close_qfq"] / merged["close"].replace(0, np.nan)
        merged = merged.dropna(subset=["expected_factor"])

        check = tail.merge(merged[["date", "expected_factor"]], on="date", how="inner")
        if check.empty:
            rows.append({"code": code, "status": "warn", "message": "日期无交集"})
            continue

        local_factor = pd.to_numeric(check["factor"], errors="coerce")
        diff = (local_factor - check["expected_factor"]).abs() / check["expected_factor"].clip(lower=1e-6)
        max_diff = float(diff.max())
        status = "ok" if max_diff < 0.02 else "error"
        rows.append(
            {
                "code": code,
                "status": status,
                "max_factor_rel_diff": max_diff,
                "message": f"factor diff={max_diff:.4f}",
            }
        )

    report = pd.DataFrame(rows)
    out = config.meta_dir / "akshare_factor_check.parquet"
    if not report.empty:
        report.to_parquet(out, index=False)
    logger.info("AkShare factor 抽样: %s 条 -> %s", len(report), out)
    return report
