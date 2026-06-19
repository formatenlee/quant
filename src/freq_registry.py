from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from quant_cursor.config import Config
from quant_cursor.qlib_export import code_to_qlib_id

logger = logging.getLogger(__name__)

FREQ_COVERAGE_NAME = "qlib_freq_coverage.parquet"


def _scan_qlib_minute_features(qlib_dir: Path) -> dict[str, dict]:
    """扫描 features 目录下 *.1min.bin 判断分钟是否已入库。"""
    feat_root = qlib_dir / "features"
    if not feat_root.exists():
        return {}
    found: dict[str, dict] = {}
    for inst_dir in feat_root.iterdir():
        if not inst_dir.is_dir():
            continue
        close_bin = inst_dir / "close.1min.bin"
        if not close_bin.exists():
            continue
        qlib_id = inst_dir.name.upper()
        if qlib_id.startswith("IDX_"):
            pass
        elif qlib_id.startswith(("SH", "SZ", "BJ")):
            pass
        else:
            qlib_id = inst_dir.name.upper()
        found[qlib_id] = {
            "has_1min_qlib": True,
            "min_qlib_bin": str(close_bin),
        }
    return found


def build_freq_coverage(config: Config) -> pd.DataFrame:
    """
    构建跨尺度关联表：同一 qlib_id 关联 day / week / 1min 覆盖情况。
    供训练时优先选取有分钟数据的标的。
    """
    meta = config.meta_dir
    qlib_dir = config.qlib_data_dir

    day_path = meta / "qlib_manifest.parquet"
    if not day_path.exists():
        raise FileNotFoundError(f"缺少日线 manifest: {day_path}")

    day = pd.read_parquet(day_path)
    day = day.rename(
        columns={
            "rows": "day_rows",
            "start": "day_start",
            "end": "day_end",
        }
    )
    for col in ("day_rows", "day_start", "day_end"):
        if col not in day.columns:
            day[col] = pd.NA
    day["has_day"] = True

    week_path = meta / "qlib_manifest_week.parquet"
    if week_path.exists():
        week = pd.read_parquet(week_path)[["qlib_id", "rows", "start", "end"]].rename(
            columns={"rows": "week_rows", "start": "week_start", "end": "week_end"}
        )
        week["has_week"] = True
        merged = day.merge(week, on="qlib_id", how="left")
    else:
        merged = day.copy()
        merged["has_week"] = False
        merged["week_rows"] = pd.NA
        merged["week_start"] = pd.NA
        merged["week_end"] = pd.NA

    min_manifest_path = meta / "qlib_manifest_1min.parquet"
    if min_manifest_path.exists():
        minute = pd.read_parquet(min_manifest_path)[["qlib_id", "rows", "start", "end"]].rename(
            columns={"rows": "min_rows", "start": "min_start", "end": "min_end"}
        )
        minute["has_1min"] = True
        merged = merged.merge(minute, on="qlib_id", how="left")
    else:
        merged["has_1min"] = False
        merged["min_rows"] = pd.NA
        merged["min_start"] = pd.NA
        merged["min_end"] = pd.NA

    qlib_scan = _scan_qlib_minute_features(qlib_dir)
    merged["has_1min_qlib"] = merged["qlib_id"].astype(str).map(
        lambda x: qlib_scan.get(x, {}).get("has_1min_qlib", False)
    )
    has_min = merged.get("has_1min", pd.Series(False, index=merged.index)).astype(bool)
    merged["has_1min"] = has_min | merged["has_1min_qlib"].astype(bool)

    for asset_type, base in (("index", config.indices_min_dir), ("etf", config.etf_min_dir)):
        if not base.exists():
            continue
        for path in base.glob("*.parquet"):
            code = path.stem.zfill(6)
            qlib_id = code_to_qlib_id(code, asset_type)
            mask = merged["qlib_id"] == qlib_id
            if not mask.any():
                continue
            try:
                df = pd.read_parquet(path, columns=["date"])
                if df.empty:
                    continue
                merged.loc[mask, "has_1min_parquet"] = True
                merged.loc[mask, "min_parquet_rows"] = len(df)
                merged.loc[mask, "min_parquet_start"] = str(pd.to_datetime(df["date"]).min())
                merged.loc[mask, "min_parquet_end"] = str(pd.to_datetime(df["date"]).max())
            except Exception:  # noqa: BLE001
                continue

    if "has_1min_parquet" not in merged.columns:
        merged["has_1min_parquet"] = False
    merged["has_1min"] = merged["has_1min"].astype(bool) | merged["has_1min_parquet"].astype(bool)

    merged["has_day"] = merged["has_day"].fillna(False)
    merged["has_week"] = merged["has_week"].fillna(False)

    out_path = meta / FREQ_COVERAGE_NAME
    meta.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    n_min = int(merged["has_1min"].sum())
    logger.info(
        "频率覆盖表: %s 标的, 含分钟 %s -> %s",
        len(merged),
        n_min,
        out_path,
    )
    return merged


def load_freq_coverage(config: Config | None = None, path: Path | None = None) -> pd.DataFrame:
    cfg = config or Config()
    cov_path = path or (cfg.meta_dir / FREQ_COVERAGE_NAME)
    if not cov_path.exists():
        return build_freq_coverage(cfg)
    return pd.read_parquet(cov_path)
