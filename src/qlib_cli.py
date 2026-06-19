from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from quant_cursor.config import load_config
from quant_cursor.qlib_export import (
    dump_to_qlib,
    export_staging,
    write_qlib_init_snippet,
)
from quant_cursor.derivatives_qlib import dump_derivatives_to_qlib, export_derivatives_staging
from quant_cursor.freq_registry import build_freq_coverage

logger = logging.getLogger("quant_cursor")


def _load_skip_codes(config) -> set[str]:
    skip: set[str] = set()
    invalid = config.meta_dir / "invalid_data.parquet"
    if invalid.exists():
        skip.update(pd.read_parquet(invalid)["code"].astype(str))
    return skip


def cmd_export(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    staging = Path(args.staging) if args.staging else None
    skip = _load_skip_codes(config)
    manifest = export_staging(config, staging_dir=staging, skip_codes=skip, freq="day")
    week_manifest = export_staging(config, skip_codes=skip, freq="week")
    build_freq_coverage(config)
    print(f"\n已导出 day={len(manifest)} week={len(week_manifest)} 个标的")
    print(manifest.head(10).to_string(index=False))
    return 0


def cmd_export_minute(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    skip = _load_skip_codes(config)
    manifest = export_staging(config, skip_codes=skip, freq="1min")
    build_freq_coverage(config)
    print(f"\n已导出 1min={len(manifest)} 个标的")
    if not manifest.empty:
        print(manifest.head(10).to_string(index=False))
    return 0


def cmd_dump(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    staging = Path(args.staging) if args.staging else None
    qlib_dir = Path(args.qlib_dir) if args.qlib_dir else config.qlib_data_dir

    if args.export_first:
        skip = _load_skip_codes(config)
        export_staging(config, staging_dir=staging, skip_codes=skip, freq="day")
        export_staging(config, skip_codes=skip, freq="week")

    dump_mode = getattr(args, "dump_mode", "auto")
    target = dump_to_qlib(
        config,
        staging_dir=staging,
        qlib_dir=qlib_dir,
        max_workers=args.workers,
        freq="day",
        mode=dump_mode,
    )
    dump_to_qlib(
        config,
        staging_dir=None,
        qlib_dir=qlib_dir,
        max_workers=args.workers,
        freq="week",
        mode=dump_mode,
    )
    write_qlib_init_snippet(target, config.meta_dir / "qlib_init_example.py")
    build_freq_coverage(config)
    print(f"\nQlib 数据目录: {target.resolve()}")
    print(f"初始化示例: {config.meta_dir / 'qlib_init_example.py'}")
    print('\n在代码中使用: qlib.init(provider_uri="...", region=REG_CN)')
    return 0


def cmd_dump_minute(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    qlib_dir = Path(args.qlib_dir) if args.qlib_dir else config.qlib_data_dir

    if args.export_first:
        skip = _load_skip_codes(config)
        export_staging(config, skip_codes=skip, freq="1min")

    dump_mode = getattr(args, "dump_mode", "auto")
    target = dump_to_qlib(
        config,
        qlib_dir=qlib_dir,
        max_workers=args.workers,
        freq="1min",
        mode=dump_mode,
    )
    print(f"\nQlib 分钟数据已写入: {target.resolve()}")
    return 0


def cmd_all_minute(args: argparse.Namespace) -> int:
    if cmd_export_minute(args) != 0:
        return 1
    args.export_first = False
    return cmd_dump_minute(args)


def cmd_export_derivatives(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    manifest = export_derivatives_staging(config)
    print(f"\n已导出衍生品 staging={len(manifest)} 个")
    if not manifest.empty:
        print(manifest.to_string(index=False))
    return 0


def cmd_dump_derivatives(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config) if args.config else None)
    if args.export_first:
        export_derivatives_staging(config)
    dump_mode = getattr(args, "dump_mode", "auto")
    target = dump_derivatives_to_qlib(config, max_workers=args.workers, mode=dump_mode)
    print(f"\n衍生品 Qlib 数据已写入: {target.resolve()}")
    return 0


def cmd_all_derivatives(args: argparse.Namespace) -> int:
    if cmd_export_derivatives(args) != 0:
        return 1
    args.export_first = False
    return cmd_dump_derivatives(args)


def cmd_all(args: argparse.Namespace) -> int:
    if cmd_export(args) != 0:
        return 1
    args.export_first = False
    return cmd_dump(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导出数据至 Microsoft Qlib 格式")
    parser.add_argument("--config", help="配置文件路径")

    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser("export", help="导出为 Qlib 中间 parquet（按标的分文件）")
    p_export.add_argument("--staging", help="中间目录，默认 data/qlib_staging")
    p_export.set_defaults(func=cmd_export)

    p_export_min = sub.add_parser("export-minute", help="导出分钟 parquet 至 qlib_staging_1min")
    p_export_min.set_defaults(func=cmd_export_minute)

    p_dump = sub.add_parser("dump", help="将中间数据转为 Qlib .bin（需安装 pyqlib）")
    p_dump.add_argument("--staging", help="中间目录")
    p_dump.add_argument("--qlib-dir", help="Qlib 数据目录，默认 data/qlib_data")
    p_dump.add_argument("--workers", type=int, default=4)
    p_dump.add_argument(
        "--export-first",
        action="store_true",
        help="dump 前先执行 export",
    )
    p_dump.add_argument(
        "--dump-mode",
        choices=["auto", "all", "update"],
        default="auto",
        help="Qlib dump 模式",
    )
    p_dump.set_defaults(func=cmd_dump)

    p_dump_min = sub.add_parser("dump-minute", help="将分钟中间数据写入 Qlib .bin")
    p_dump_min.add_argument("--qlib-dir", help="Qlib 数据目录")
    p_dump_min.add_argument("--workers", type=int, default=4)
    p_dump_min.add_argument("--export-first", action="store_true")
    p_dump_min.add_argument("--dump-mode", choices=["auto", "all", "update"], default="auto")
    p_dump_min.set_defaults(func=cmd_dump_minute)

    p_all = sub.add_parser("all", help="export + dump 一步完成")
    p_all.add_argument("--staging", help="中间目录")
    p_all.add_argument("--qlib-dir", help="Qlib 数据目录")
    p_all.add_argument("--workers", type=int, default=4)
    p_all.set_defaults(func=cmd_all)

    p_all_min = sub.add_parser("all-minute", help="export-minute + dump-minute 一步完成")
    p_all_min.add_argument("--qlib-dir", help="Qlib 数据目录")
    p_all_min.add_argument("--workers", type=int, default=4)
    p_all_min.set_defaults(func=cmd_all_minute)

    p_export_deriv = sub.add_parser("export-derivatives", help="导出金融期货 + 事件日历至 qlib_staging_deriv")
    p_export_deriv.set_defaults(func=cmd_export_derivatives)

    p_dump_deriv = sub.add_parser("dump-derivatives", help="将衍生品中间数据写入 Qlib .bin")
    p_dump_deriv.add_argument("--workers", type=int, default=4)
    p_dump_deriv.add_argument("--export-first", action="store_true")
    p_dump_deriv.add_argument("--dump-mode", choices=["auto", "all", "update"], default="auto")
    p_dump_deriv.set_defaults(func=cmd_dump_derivatives)

    p_all_deriv = sub.add_parser("all-derivatives", help="export-derivatives + dump-derivatives")
    p_all_deriv.add_argument("--workers", type=int, default=4)
    p_all_deriv.set_defaults(func=cmd_all_derivatives)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
