from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from quant_cursor.config import load_config
from quant_cursor.downloader import MarketDataDownloader
from quant_cursor.minute_downloader import MinuteDataDownloader
from quant_cursor.universe import (
    build_universe,
    is_today_data_ready,
    load_universe,
    save_universe,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("quant_cursor")


def cmd_universe(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    universe = build_universe(config)
    save_universe(universe, config.universe_path)

    summary = universe.groupby(["asset_type", "category"]).size().reset_index(name="count")
    print("\n标的池统计:")
    print(summary.to_string(index=False))
    print(f"\n已保存至 {config.universe_path}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    universe = load_universe(config.universe_path)

    mode = args.mode
    if mode in ("today", "incremental") and not args.force:
        if not is_today_data_ready(config):
            logger.warning(
                "当前时间早于 %s:00（%s），当日行情可能尚未更新。可使用 --force 强制执行。",
                config.today_update_hour,
                config.timezone,
            )
            return 1

    categories = args.category.split(",") if args.category else None
    asset_types = args.asset_type.split(",") if args.asset_type else None
    codes = args.codes.split(",") if args.codes else None

    downloader = MarketDataDownloader(config)
    report = downloader.download_universe(
        universe,
        mode=mode,
        categories=categories,
        asset_types=asset_types,
        codes=codes,
        skip_existing=args.skip_existing,
    )

    config.meta_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.meta_dir / f"download_report_{mode}.parquet"
    report.to_parquet(report_path, index=False)

    failed = report[report["status"] == "fail"]
    if not failed.empty:
        print(f"\n失败 {len(failed)} 个，详情见 {report_path}")
        return 2

    print(f"\n全部成功，报告: {report_path}")
    return 0


def cmd_download_minute(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    from quant_cursor.freq_registry import build_freq_coverage

    if getattr(args, "minute_source", None):
        config.minute_data_source = args.minute_source

    categories = args.category.split(",") if args.category else None
    asset_types = args.asset_type.split(",") if args.asset_type else None
    codes = args.codes.split(",") if args.codes else None

    downloader = MinuteDataDownloader(config)
    report = downloader.download_qlib_universe(
        mode=args.mode,
        asset_types=asset_types,
        categories=categories,
        codes=codes,
        skip_existing=args.skip_existing,
    )

    build_freq_coverage(config)

    failed = report[report["status"] == "fail"]
    ok = report[report["status"] == "ok"]
    print(f"\n分钟下载: 成功 {len(ok)} / 跳过 {(report['status']=='skip').sum()} / 失败 {len(failed)}（失败已忽略）")
    return 0 if len(failed) == len(report) and len(ok) == 0 else 0


def cmd_download_derivatives(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    from quant_cursor.derivatives_downloader import DerivativesDownloader

    codes = args.codes.split(",") if args.codes else None
    downloader = DerivativesDownloader(config)
    futures_report, events = downloader.download_all(
        mode=args.mode,
        skip_existing=args.skip_existing,
        codes=codes,
    )

    config.meta_dir.mkdir(parents=True, exist_ok=True)
    futures_report.to_parquet(config.meta_dir / f"derivatives_download_{args.mode}.parquet", index=False)

    ok = futures_report[futures_report["status"] == "ok"]
    fail = futures_report[futures_report["status"] == "fail"]
    print(f"\n金融期货: 成功 {len(ok)} / 失败 {len(fail)}")
    print(f"衍生品事件: {len(events)} 条 -> {config.derivatives_events_path}")
    if not fail.empty:
        print(fail.to_string(index=False))
    return 0 if len(fail) == 0 else 2


def cmd_repair_fields(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    universe = load_universe(config.universe_path)
    asset_types = args.asset_type.split(",") if args.asset_type else None
    codes = args.codes.split(",") if args.codes else None

    downloader = MarketDataDownloader(config)
    pending = downloader.find_symbols_needing_field_repair(
        universe,
        min_amount_coverage=args.min_amount_coverage,
        asset_types=asset_types,
        include_extra_redownload=args.with_extra_fields,
    )
    pending_path = config.meta_dir / "repair_fields_pending.parquet"
    config.meta_dir.mkdir(parents=True, exist_ok=True)
    pending.to_parquet(pending_path, index=False)
    print(f"待 repair: {len(pending)} 个 -> {pending_path}")
    if not pending.empty and args.list_only:
        print(pending.head(30).to_string(index=False))
        if len(pending) > 30:
            print(f"... 共 {len(pending)} 条")
        return 0

    report = downloader.repair_extra_fields(
        universe,
        asset_types=asset_types,
        codes=codes,
        min_amount_coverage=args.min_amount_coverage,
        include_extra_redownload=args.with_extra_fields,
    )
    report_path = config.meta_dir / "repair_fields_report.parquet"
    report.to_parquet(report_path, index=False)
    ok = (report["status"] == "ok").sum() if not report.empty else 0
    fail = (report["status"] == "fail").sum() if not report.empty else 0
    print(f"\nrepair 完成: 成功 {ok} / 失败 {fail}，报告 {report_path}")
    return 0 if fail == 0 else 2


def cmd_list(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    universe = load_universe(config.universe_path)

    if args.asset_type:
        universe = universe[universe["asset_type"] == args.asset_type]
    if args.category:
        universe = universe[universe["category"] == args.category]

    print(universe.to_string(index=False))
    print(f"\n共 {len(universe)} 条")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="中国指数与宽基 ETF 行情数据采集（基于 AKShare）"
    )
    parser.add_argument(
        "--config",
        help="配置文件路径，默认使用项目根目录 config.yaml",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_universe = sub.add_parser("universe", help="构建并保存标的池")
    p_universe.set_defaults(func=cmd_universe)

    p_download = sub.add_parser("download", help="下载行情数据")
    p_download.add_argument(
        "--mode",
        choices=["full", "incremental", "today"],
        default="incremental",
        help="full=全量; incremental/today=增量至最新（today 检查 21:00）",
    )
    p_download.add_argument(
        "--force",
        action="store_true",
        help="忽略当日更新时间检查",
    )
    p_download.add_argument(
        "--asset-type",
        help="过滤资产类型: index, etf, stock（逗号分隔）",
    )
    p_download.add_argument(
        "--category",
        help="过滤分类: major, broad, industry, broad_etf 等（逗号分隔）",
    )
    p_download.add_argument(
        "--codes",
        help="仅下载指定代码（逗号分隔，如 000001,399006,510050）",
    )
    p_download.add_argument(
        "--skip-existing",
        action="store_true",
        help="跳过本地已有 parquet 文件，不重复下载",
    )
    p_download.set_defaults(func=cmd_download)

    p_min = sub.add_parser("download-minute", help="为已入库 Qlib 标的补充分钟数据（缺失跳过）")
    p_min.add_argument(
        "--mode",
        choices=["full", "incremental", "today"],
        default="incremental",
    )
    p_min.add_argument("--asset-type", help="过滤 index, etf（逗号分隔）")
    p_min.add_argument("--category", help="过滤 major, broad, industry, broad_etf 等")
    p_min.add_argument("--codes", help="仅下载指定代码")
    p_min.add_argument("--skip-existing", action="store_true")
    p_min.add_argument(
        "--minute-source",
        choices=["sina", "em", "auto"],
        help="分钟数据源：sina / em / auto（默认读 config.yaml）",
    )
    p_min.set_defaults(func=cmd_download_minute)

    p_deriv = sub.add_parser("download-derivatives", help="下载金融期货 + 期权交割/到期事件（不含商品）")
    p_deriv.add_argument("--mode", choices=["full", "incremental"], default="incremental")
    p_deriv.add_argument("--skip-existing", action="store_true")
    p_deriv.add_argument("--codes", help="仅下载指定期货代码，如 IF0,IH0")
    p_deriv.set_defaults(func=cmd_download_derivatives)

    p_repair = sub.add_parser(
        "repair-fields",
        help="重下 amount/扩展字段缺失的标的，并写入本地 parquet",
    )
    p_repair.add_argument("--asset-type", help="index, etf, stock（逗号分隔）")
    p_repair.add_argument("--codes", help="仅 repair 指定代码")
    p_repair.add_argument(
        "--min-amount-coverage",
        type=float,
        default=0.05,
        help="amount 非空比例低于该值则重下，默认 0.05",
    )
    p_repair.add_argument(
        "--with-extra-fields",
        action="store_true",
        help="同时全量重下缺少 turnover_rate 等扩展列的标的（ETF/股票，耗时较长）",
    )
    p_repair.add_argument(
        "--list-only",
        action="store_true",
        help="仅列出待 repair 标的，不下载",
    )
    p_repair.set_defaults(func=cmd_repair_fields)

    p_list = sub.add_parser("list", help="查看标的池")
    p_list.add_argument("--asset-type", choices=["index", "etf", "stock"])
    p_list.add_argument("--category")
    p_list.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    if argv and argv[0] == "qlib":
        from quant_cursor.qlib_cli import main as qlib_main

        return qlib_main(argv[1:])
    if argv and argv[0] == "pipeline":
        from quant_cursor.pipeline import main as pipeline_main

        return pipeline_main(argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
