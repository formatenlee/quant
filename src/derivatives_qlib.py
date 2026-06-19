from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from quant_cursor.config import Config
from quant_cursor.derivatives_events import build_event_calendar_frame, save_derivatives_events
from quant_cursor.derivatives_universe import CALENDAR_QLIB_ID, EVENT_CALENDAR_FIELDS, futures_specs
from quant_cursor.qlib_export import _to_qlib_frame, dump_to_qlib

logger = logging.getLogger(__name__)


def export_derivatives_staging(config: Config) -> pd.DataFrame:
    """将金融期货 parquet 导出为 Qlib 中间文件，并写入事件日历。"""
    staging = config.qlib_deriv_staging_dir
    staging.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for spec in futures_specs():
        src = config.derivatives_futures_dir / f"{spec.code}.parquet"
        if not src.exists():
            logger.warning("跳过缺失期货文件: %s", src)
            continue
        raw = pd.read_parquet(src)
        if raw.empty or "close" not in raw.columns:
            continue
        frame = _to_qlib_frame(raw)
        out_path = staging / f"{spec.qlib_id.lower()}.parquet"
        frame.to_parquet(out_path, index=False)
        records.append(
            {
                "code": spec.code,
                "qlib_id": spec.qlib_id,
                "name": spec.name,
                "asset_type": "financial_futures",
                "product": spec.product,
                "freq": "day",
                "rows": len(frame),
                "start": frame["date"].iloc[0],
                "end": frame["date"].iloc[-1],
                "file": str(out_path),
            }
        )

    if config.derivatives_events_path.exists():
        cal_frame = build_event_calendar_frame(config)
        cal_path = staging / f"{CALENDAR_QLIB_ID.lower()}.parquet"
        cal_frame.to_parquet(cal_path, index=False)
        records.append(
            {
                "code": CALENDAR_QLIB_ID,
                "qlib_id": CALENDAR_QLIB_ID,
                "name": "衍生品事件日历",
                "asset_type": "calendar",
                "product": "EVENTS",
                "freq": "day",
                "rows": len(cal_frame),
                "start": cal_frame["date"].iloc[0],
                "end": cal_frame["date"].iloc[-1],
                "file": str(cal_path),
            }
        )

    manifest = pd.DataFrame(records)
    manifest_path = config.meta_dir / "qlib_manifest_derivatives.parquet"
    manifest.to_parquet(manifest_path, index=False)
    logger.info("衍生品 Qlib 中间数据: %s 个 -> %s", len(manifest), staging)
    return manifest


def dump_derivatives_to_qlib(
    config: Config,
    *,
    max_workers: int = 4,
    mode: str = "update",
) -> Path:
    """将衍生品 OHLCV 与事件日历写入 Qlib（始终 dump_update，避免覆盖 instruments/all.txt）。"""
    staging = config.qlib_deriv_staging_dir
    if not staging.exists() or not any(staging.glob("*.parquet")):
        raise FileNotFoundError(f"衍生品中间目录为空: {staging}，请先 export-derivatives")

    target = config.qlib_data_dir
    dump_mode = "update" if mode == "all" else mode
    if mode == "all":
        logger.warning("衍生品 dump 忽略 mode=all，强制使用 dump_update 以保护现有 instruments")

    # 期货 OHLCV（标准字段）
    futures_files = [p for p in staging.glob("*.parquet") if p.stem != CALENDAR_QLIB_ID.lower()]
    if futures_files:
        futures_staging = staging / "_futures_only"
        futures_staging.mkdir(exist_ok=True)
        for p in futures_files:
            dest = futures_staging / p.name
            if not dest.exists() or dest.stat().st_mtime < p.stat().st_mtime:
                pd.read_parquet(p).to_parquet(dest, index=False)
        dump_to_qlib(
            config,
            staging_dir=futures_staging,
            qlib_dir=target,
            max_workers=max_workers,
            freq="day",
            mode=dump_mode,
        )

    # 事件日历（自定义字段）
    cal_path = staging / f"{CALENDAR_QLIB_ID.lower()}.parquet"
    if cal_path.exists():
        cal_staging = staging / "_calendar_only"
        cal_staging.mkdir(exist_ok=True)
        cal_dest = cal_staging / cal_path.name
        pd.read_parquet(cal_path).to_parquet(cal_dest, index=False)
        dump_to_qlib(
            config,
            staging_dir=cal_staging,
            qlib_dir=target,
            max_workers=1,
            freq="day",
            mode=dump_mode,
            include_fields=",".join(EVENT_CALENDAR_FIELDS),
        )

    logger.info("衍生品 Qlib dump 完成: %s", target.resolve())
    return target


def refresh_derivatives_events_and_staging(config: Config) -> pd.DataFrame:
    """重建事件表并刷新 staging 中的日历文件。"""
    from quant_cursor.derivatives_events import build_derivatives_events

    events = build_derivatives_events(config)
    save_derivatives_events(config, events)
    return export_derivatives_staging(config)
