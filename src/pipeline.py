from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from quant_cursor.config import PROJECT_ROOT, load_config
from quant_cursor.downloader import MarketDataDownloader
from quant_cursor.qlib_export import (
    dump_to_qlib,
    export_staging,
    write_qlib_init_snippet,
)
from quant_cursor.freq_registry import build_freq_coverage
from quant_cursor.minute_downloader import MinuteDataDownloader
from quant_cursor.derivatives_qlib import dump_derivatives_to_qlib, export_derivatives_staging
from quant_cursor.derivatives_downloader import DerivativesDownloader
from quant_cursor.universe import build_universe, is_today_data_ready, load_universe, save_universe
from quant_cursor.validation import (
    cross_check_qlib_vs_source,
    validate_universe,
    verify_akshare_factor_sample,
)

logger = logging.getLogger("quant_cursor.pipeline")


def _load_skip_codes(config) -> set[str]:
    skip: set[str] = set()
    invalid = config.meta_dir / "invalid_data.parquet"
    if invalid.exists():
        skip.update(pd.read_parquet(invalid)["code"].astype(str).str.zfill(6))
    return skip


def step_universe(config, *, refresh_sw: bool = False) -> pd.DataFrame:
    from quant_cursor.rate_limit import RateLimitedClient

    if refresh_sw:
        from quant_cursor.universe import build_sw_l2_mapping

        client = RateLimitedClient(
            delay=config.stock_request_delay,
            max_retries=config.max_retries,
        )
        build_sw_l2_mapping(config, client, force=True)

    if config.universe_path.exists():
        existing = load_universe(config.universe_path)
        index_etf = existing[existing["asset_type"].isin(["index", "etf"])]
        if config.include_stocks:
            fresh = build_universe(config)
            universe = fresh
        else:
            universe = index_etf
    else:
        universe = build_universe(config)

    save_universe(universe, config.universe_path)
    logger.info("标的池: %s 条", len(universe))
    return universe


def step_validate(config, stage: str = "post") -> pd.DataFrame:
    universe = load_universe(config.universe_path)
    report = validate_universe(config, universe)
    out = config.meta_dir / f"validation_report_{stage}.parquet"
    report.to_parquet(out, index=False)
    return report


def step_download(
    config,
    *,
    mode: str = "incremental",
    asset_types: list[str] | None = None,
    skip_existing: bool = False,
    force: bool = False,
) -> pd.DataFrame:
    if mode in ("incremental", "today") and not force:
        if not is_today_data_ready(config):
            raise RuntimeError(
                f"当前时间早于 {config.today_update_hour}:00，请使用 --force 或稍后再试"
            )

    universe = load_universe(config.universe_path)
    downloader = MarketDataDownloader(config)
    report = downloader.download_universe(
        universe,
        mode=mode,
        asset_types=asset_types,
        skip_existing=skip_existing,
    )
    out = config.meta_dir / f"download_report_{mode}.parquet"
    report.to_parquet(out, index=False)
    return report


def step_export_qlib(config) -> None:
    skip = _load_skip_codes(config)
    export_staging(config, skip_codes=skip, freq="day")
    export_staging(config, skip_codes=skip, freq="week")
    build_freq_coverage(config)


def step_download_minute(
    config,
    *,
    mode: str = "incremental",
    asset_types: list[str] | None = None,
    skip_existing: bool = False,
    pending_only: bool = False,
) -> pd.DataFrame:
    downloader = MinuteDataDownloader(config)
    report = downloader.download_qlib_universe(
        mode=mode,
        asset_types=asset_types,
        skip_existing=skip_existing,
        pending_only=pending_only or skip_existing,
    )
    out = config.meta_dir / f"minute_download_report_{mode}.parquet"
    report.to_parquet(out, index=False)
    build_freq_coverage(config)
    return report


def step_export_minute_qlib(config) -> None:
    skip = _load_skip_codes(config)
    export_staging(config, skip_codes=skip, freq="1min")
    build_freq_coverage(config)


def step_dump_minute_qlib(config, *, workers: int = 4, dump_mode: str = "auto") -> None:
    dump_to_qlib(config, max_workers=workers, freq="1min", mode=dump_mode)
    build_freq_coverage(config)


def step_download_derivatives(config, *, mode: str = "incremental", skip_existing: bool = False) -> pd.DataFrame:
    downloader = DerivativesDownloader(config)
    futures_report, events = downloader.download_all(mode=mode, skip_existing=skip_existing)
    out = config.meta_dir / f"derivatives_download_{mode}.parquet"
    futures_report.to_parquet(out, index=False)
    logger.info("衍生品事件 %s 条", len(events))
    return futures_report


def step_export_derivatives_qlib(config) -> None:
    export_derivatives_staging(config)


def step_dump_derivatives_qlib(config, *, workers: int = 4, dump_mode: str = "auto") -> None:
    dump_derivatives_to_qlib(config, max_workers=workers, mode=dump_mode)


def step_dump_qlib(config, *, workers: int = 4, dump_mode: str = "auto") -> None:
    dump_to_qlib(config, max_workers=workers, freq="day", mode=dump_mode)
    dump_to_qlib(config, max_workers=workers, freq="week", mode=dump_mode)
    write_qlib_init_snippet(config.qlib_data_dir, config.meta_dir / "qlib_init_example.py")


def step_verify_qlib(config, *, sample_n: int | None = None) -> None:
    n = sample_n or config.sample_validation_count
    cross_check_qlib_vs_source(config, sample_n=n)
    if config.include_stocks:
        verify_akshare_factor_sample(config, sample_n=min(5, n))


def run_pipeline(
    config,
    *,
    steps: list[str] | None = None,
    download_mode: str = "incremental",
    asset_types: list[str] | None = None,
    skip_existing: bool = False,
    force: bool = False,
    workers: int = 4,
    dump_mode: str = "auto",
    sample_n: int | None = None,
) -> int:
    all_steps = steps or [
        "universe",
        "validate_pre",
        "download",
        "validate_post",
        "export",
        "dump",
        "verify_qlib",
    ]
    config.ensure_dirs()
    log_path = config.meta_dir / "pipeline.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    logger.info("Pipeline 开始 steps=%s mode=%s asset_types=%s", all_steps, download_mode, asset_types)
    exit_code = 0

    try:
        for step in all_steps:
            logger.info("=== step: %s ===", step)
            if step == "universe":
                step_universe(config)
            elif step == "validate_pre":
                if config.universe_path.exists():
                    step_validate(config, "pre")
            elif step == "download":
                report = step_download(
                    config,
                    mode=download_mode,
                    asset_types=asset_types,
                    skip_existing=skip_existing,
                    force=force,
                )
                fails = (report["status"] == "fail").sum() if not report.empty else 0
                if fails:
                    logger.warning("下载失败 %s 个，继续后续步骤", fails)
            elif step == "validate_post":
                step_validate(config, "post")
            elif step == "export":
                step_export_qlib(config)
            elif step == "dump":
                step_dump_qlib(config, workers=workers, dump_mode=dump_mode)
            elif step == "download_minute":
                report = step_download_minute(
                    config,
                    mode=download_mode,
                    asset_types=asset_types,
                    skip_existing=skip_existing,
                )
                fails = (report["status"] == "fail").sum() if not report.empty else 0
                if fails:
                    logger.warning("分钟下载失败 %s 个（已忽略）", fails)
            elif step == "export_minute":
                step_export_minute_qlib(config)
            elif step == "dump_minute":
                step_dump_minute_qlib(config, workers=workers, dump_mode=dump_mode)
            elif step == "download_derivatives":
                report = step_download_derivatives(
                    config,
                    mode=download_mode if download_mode != "today" else "incremental",
                    skip_existing=skip_existing,
                )
                fails = (report["status"] == "fail").sum() if not report.empty else 0
                if fails:
                    logger.warning("衍生品下载失败 %s 个", fails)
            elif step == "export_derivatives":
                step_export_derivatives_qlib(config)
            elif step == "dump_derivatives":
                step_dump_derivatives_qlib(config, workers=workers, dump_mode=dump_mode)
            elif step == "verify_qlib":
                step_verify_qlib(config, sample_n=sample_n)
            else:
                raise ValueError(f"未知 step: {step}")
        logger.info("Pipeline 完成")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline 失败: %s", exc)
        exit_code = 1

    logger.removeHandler(file_handler)
    file_handler.close()
    return exit_code


def spawn_background(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    config.meta_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(ZoneInfo(config.timezone)).strftime("%Y%m%d_%H%M%S")
    log_path = config.meta_dir / f"pipeline_bg_{ts}.log"
    pid_path = config.meta_dir / "pipeline.pid"

    cmd = [
        sys.executable,
        "-m",
        "quant_cursor",
        "pipeline",
        "run",
        "--foreground",
        "--download-mode",
        args.download_mode,
        "--workers",
        str(args.workers),
    ]
    if args.config:
        cmd.extend(["--config", args.config])
    if args.asset_type:
        cmd.extend(["--asset-type", args.asset_type])
    if args.steps:
        cmd.extend(["--steps", args.steps])
    if args.force:
        cmd.append("--force")
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.dump_mode:
        cmd.extend(["--dump-mode", args.dump_mode])

    log_f = open(log_path, "a", encoding="utf-8")
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        creationflags=creationflags,
    )
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    print(f"后台任务已启动 pid={proc.pid}")
    print(f"日志: {log_path}")
    print(f"PID 文件: {pid_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="数据采集 + 校验 + Qlib 入库流水线")
    parser.add_argument("--config", help="配置文件路径")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="执行完整流水线")
    p_run.add_argument(
        "--steps",
        help="逗号分隔: universe,validate_pre,download,validate_post,export,dump,verify_qlib,download_minute,export_minute,dump_minute,download_derivatives,export_derivatives,dump_derivatives",
    )
    p_run.add_argument(
        "--download-mode",
        choices=["full", "incremental", "today"],
        default="incremental",
        help="下载模式，默认 incremental",
    )
    p_run.add_argument("--asset-type", help="仅下载 index,etf,stock（逗号分隔）")
    p_run.add_argument("--force", action="store_true", help="忽略当日更新时间检查")
    p_run.add_argument("--skip-existing", action="store_true", help="full 模式下跳过已有文件")
    p_run.add_argument("--workers", type=int, default=4)
    p_run.add_argument(
        "--dump-mode",
        choices=["auto", "all", "update"],
        default="auto",
        help="Qlib dump 模式",
    )
    p_run.add_argument("--foreground", action="store_true", help=argparse.SUPPRESS)
    p_run.add_argument("--background", action="store_true", help="后台 detached 运行")
    p_run.set_defaults(func=cmd_run)

    p_status = sub.add_parser("status", help="查看后台任务 PID")
    p_status.set_defaults(func=cmd_status)

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    if args.background and not args.foreground:
        return spawn_background(args)

    config = load_config(Path(args.config) if args.config else None)
    steps = args.steps.split(",") if args.steps else None
    asset_types = args.asset_type.split(",") if args.asset_type else None
    return run_pipeline(
        config,
        steps=steps,
        download_mode=args.download_mode,
        asset_types=asset_types,
        skip_existing=args.skip_existing,
        force=args.force,
        workers=args.workers,
        dump_mode=args.dump_mode,
    )


def cmd_status(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    pid_path = config.meta_dir / "pipeline.pid"
    if not pid_path.exists():
        print("无 pipeline.pid，可能未在后台运行")
        return 1
    print(f"PID: {pid_path.read_text(encoding='utf-8').strip()}")
    logs = sorted(config.meta_dir.glob("pipeline_bg_*.log"))
    if logs:
        print(f"最新日志: {logs[-1]}")
    if (config.meta_dir / "pipeline.log").exists():
        print(f"前台日志: {config.meta_dir / 'pipeline.log'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
