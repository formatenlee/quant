from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from quant_cursor.config import Config
from quant_cursor.fields import QLIB_ALL_FIELDS, QLIB_CORE_FIELDS, WEEKLY_AGG

logger = logging.getLogger(__name__)

QLIB_FIELDS = QLIB_CORE_FIELDS
_NUMERIC = re.compile(r"^\d{6}$")


def qlib_dump_field_list() -> str:
    return ",".join(QLIB_ALL_FIELDS)


def code_to_qlib_id(code: str, asset_type: str) -> str:
    """将标的代码转为 Qlib instrument id（大写）。"""
    raw = str(code).strip()
    if asset_type in ("etf", "stock"):
        c = raw.zfill(6)
        if c.startswith(("4", "8")):
            return f"BJ{c}"
        if c.startswith(("15", "16", "18", "0", "3")):
            return f"SZ{c}"
        return f"SH{c}"
    # index
    if not _NUMERIC.match(raw):
        return f"IDX_{raw.upper()}"
    if raw.startswith(("399", "980")):
        return f"SZ{raw}"
    return f"SH{raw}"


def _asset_data_dir(config: Config, asset_type: str, *, freq: str = "day") -> Path:
    if freq == "1min":
        if asset_type == "index":
            return config.indices_min_dir
        if asset_type == "etf":
            return config.etf_min_dir
        if asset_type == "stock":
            return config.data_dir / "stocks_min"
        raise ValueError(asset_type)
    if asset_type == "index":
        return config.indices_dir
    if asset_type == "etf":
        return config.etf_dir
    if asset_type == "stock":
        return config.stocks_dir
    raise ValueError(asset_type)


def _load_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size < 100:
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return None
    if df.empty or "date" not in df.columns or "close" not in df.columns:
        return None
    return df


def _to_qlib_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    for col in QLIB_ALL_FIELDS:
        if col not in out.columns:
            out[col] = pd.NA
    if out["factor"].isna().all():
        out["factor"] = 1.0
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0)
    for col in QLIB_ALL_FIELDS:
        if col == "volume":
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[list(QLIB_ALL_FIELDS) + ["date"]].sort_values("date")


def _to_qlib_minute_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    minute_fields = ("open", "close", "high", "low", "volume", "factor", "amount")
    for col in minute_fields:
        if col not in out.columns:
            out[col] = pd.NA
    if out["factor"].isna().all():
        out["factor"] = 1.0
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0)
    for col in ("open", "close", "high", "low", "factor", "amount"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[list(minute_fields) + ["date"]].sort_values("date")


def daily_to_weekly(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    agg = {k: v for k, v in WEEKLY_AGG.items() if k in df.columns}
    weekly = df.resample("W-FRI").agg(agg).dropna(subset=["close"])
    weekly = weekly.reset_index()
    weekly["date"] = weekly["date"].dt.strftime("%Y-%m-%d")
    return weekly


def export_staging(
    config: Config,
    staging_dir: Path | None = None,
    skip_codes: set[str] | None = None,
    *,
    freq: str = "day",
) -> pd.DataFrame:
    """将 indices/etf/stocks parquet 转为 Qlib dump_bin 中间数据。"""
    if freq == "week":
        staging = staging_dir or config.qlib_week_staging_dir
    elif freq == "1min":
        staging = staging_dir or config.qlib_min_staging_dir
    else:
        staging = staging_dir or config.qlib_staging_dir
    staging.mkdir(parents=True, exist_ok=True)

    if freq == "1min":
        manifest_path = config.meta_dir / "qlib_manifest.parquet"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Qlib 日线 manifest 不存在: {manifest_path}")
        universe = pd.read_parquet(manifest_path)
    else:
        universe_path = config.universe_path
        if not universe_path.exists():
            raise FileNotFoundError(f"标的池不存在: {universe_path}")
        universe = pd.read_parquet(universe_path)

    skip = skip_codes or set()
    records: list[dict] = []
    exported = 0
    skipped = 0

    for _, row in universe.iterrows():
        code = str(row["code"]).zfill(6)
        if code in skip:
            skipped += 1
            continue

        base = _asset_data_dir(config, row["asset_type"], freq=freq)
        raw = _load_parquet(base / f"{code}.parquet")
        if raw is None:
            skipped += 1
            continue

        qlib_id = code_to_qlib_id(code, row["asset_type"])
        if freq == "1min":
            frame = _to_qlib_minute_frame(raw)
        else:
            frame = _to_qlib_frame(raw)
        if freq == "week":
            frame = daily_to_weekly(frame)
            if frame.empty:
                skipped += 1
                continue

        out_path = staging / f"{qlib_id.lower()}.parquet"
        frame.to_parquet(out_path, index=False)
        exported += 1
        records.append(
            {
                "code": code,
                "qlib_id": qlib_id,
                "name": row.get("name", ""),
                "asset_type": row["asset_type"],
                "category": row.get("category", ""),
                "sw_l2_code": row.get("sw_l2_code", ""),
                "sw_l2_name": row.get("sw_l2_name", ""),
                "freq": freq,
                "rows": len(frame),
                "start": frame["date"].iloc[0],
                "end": frame["date"].iloc[-1],
                "file": str(out_path),
            }
        )

    manifest_names = {
        "day": "qlib_manifest.parquet",
        "week": "qlib_manifest_week.parquet",
        "1min": "qlib_manifest_1min.parquet",
    }
    manifest_name = manifest_names.get(freq, f"qlib_manifest_{freq}.parquet")
    manifest = pd.DataFrame(records)
    manifest_path = config.meta_dir / manifest_name
    config.meta_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(manifest_path, index=False)

    if freq == "day":
        legacy = config.meta_dir / "qlib_manifest.parquet"
        manifest.to_parquet(legacy, index=False)

    logger.info(
        "Qlib 中间数据 [%s]: 导出 %s 个, 跳过 %s 个 -> %s",
        freq,
        exported,
        skipped,
        staging,
    )
    logger.info("清单已保存: %s", manifest_path)
    return manifest


def find_dump_bin_script() -> Path | None:
    from quant_cursor.config import PROJECT_ROOT

    bundled = PROJECT_ROOT / "scripts" / "qlib" / "dump_bin.py"
    if bundled.exists():
        return bundled

    try:
        import qlib  # noqa: F401
    except ImportError:
        return None

    qlib_root = Path(sys.modules["qlib"].__file__).resolve().parent.parent
    script = qlib_root / "scripts" / "dump_bin.py"
    if script.exists():
        return script
    return None


def _qlib_has_data(qlib_dir: Path) -> bool:
    return (qlib_dir / "calendars" / "day.txt").exists()


def dump_to_qlib(
    config: Config,
    staging_dir: Path | None = None,
    qlib_dir: Path | None = None,
    max_workers: int = 4,
    *,
    freq: str = "day",
    mode: str = "auto",
    include_fields: str | None = None,
) -> Path:
    """调用 dump_bin.py 将中间数据写入 Qlib .bin。"""
    if freq == "week":
        staging = staging_dir or config.qlib_week_staging_dir
        target = qlib_dir or config.qlib_data_dir
    elif freq == "1min":
        staging = staging_dir or config.qlib_min_staging_dir
        target = qlib_dir or config.qlib_data_dir
    else:
        staging = staging_dir or config.qlib_staging_dir
        target = qlib_dir or config.qlib_data_dir

    if not staging.exists() or not any(staging.glob("*.parquet")):
        raise FileNotFoundError(f"中间目录为空，请先 export [{freq}]: {staging}")

    script = find_dump_bin_script()
    if script is None:
        raise RuntimeError(
            "未找到 dump_bin.py。请执行: pip install -r requirements-qlib.txt"
        )

    try:
        import qlib  # noqa: F401
    except ImportError:
        raise RuntimeError("未安装 qlib。请执行: pip install -r requirements-qlib.txt") from None

    target.mkdir(parents=True, exist_ok=True)

    dump_cmd = "dump_update" if mode == "update" else "dump_all"
    if mode == "fix":
        dump_cmd = "dump_fix"
    if mode == "auto":
        cal = target / "calendars" / f"{freq}.txt"
        dump_cmd = "dump_update" if cal.exists() else "dump_all"

    fields = include_fields or qlib_dump_field_list()
    cmd = [
        sys.executable,
        str(script),
        dump_cmd,
        f"--data_path={staging}",
        f"--qlib_dir={target}",
        f"--freq={freq}",
        "--include_fields",
        fields,
        "--file_suffix",
        ".parquet",
        "--date_field_name",
        "date",
        f"--max_workers={max_workers}",
    ]
    logger.info("执行: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("Qlib 数据 [%s] 已写入: %s", freq, target.resolve())

    if freq == "1min":
        from quant_cursor.freq_registry import build_freq_coverage

        build_freq_coverage(config)

    return target


def write_qlib_init_snippet(qlib_dir: Path, out_path: Path) -> None:
    uri = qlib_dir.resolve().as_posix()
    snippet = f'''"""Qlib 初始化示例 — 指数 / ETF / A股。"""

import qlib
from qlib.constant import REG_CN
from qlib.data import D

qlib.init(provider_uri="{uri}", region=REG_CN)

# 日线（含 amount / 换手率等扩展字段）
df_day = D.features(["SH510050"], ["$close", "$amount", "$turnover_rate"], "2020-01-01", "2026-12-31", freq="day")
print("day", df_day.head())

# 周线（由日线聚合导出）
df_week = D.features(["SH600519"], ["$close"], "2020-01-01", "2026-12-31", freq="week")
print("week", df_week.head())

# 分钟（已补充下载的标的）
df_min = D.features(["SH510050"], ["$close"], "2025-06-01", "2025-06-10", freq="1min")
print("1min", df_min.head())

# 金融期货主力 + 衍生品事件日历（统一 Qlib 入口）
from quant_cursor.qlib_data import load_financial_futures, load_derivative_event_flags, join_event_flags_to_panel

df_ff = load_financial_futures(["FF_IF", "FF_IH"], start="2024-01-01", end="2025-12-31")
print("financial futures", df_ff.head())

df_evt = load_derivative_event_flags("2024-01-01", "2025-12-31", fields=["is_50etf_opt_expiry", "is_cffex_if_last_trade"])
print("50ETF expiry flags", df_evt[df_evt["is_50etf_opt_expiry"] > 0].head(10))

# 将事件标志合并到个股面板（示例）
stock = D.features(["SH600519"], ["$close"], "2024-01-01", "2024-12-31")
stock_panel = join_event_flags_to_panel(stock.reset_index(), event_fields=["is_50etf_opt_expiry"])
print("stock + expiry flag", stock_panel[stock_panel["is_50etf_opt_expiry"] > 0].head())

# 跨尺度覆盖（同一 qlib_id 关联 day/week/1min）
from quant_cursor.freq_registry import load_freq_coverage
cov = load_freq_coverage()
print("minute coverage", cov[cov["has_1min"]][["qlib_id", "day_start", "min_start"]].head())

# 按预设组筛选标的
from quant_cursor.instruments import query_instruments
from quant_cursor.config import load_config

cfg = load_config()
ids = query_instruments(cfg, groups=["large_cap"], max_instruments=10)
print("large_cap sample", ids)
'''
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(snippet, encoding="utf-8")
    logger.info("示例脚本: %s", out_path)
