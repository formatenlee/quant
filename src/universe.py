from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import pandas as pd

from quant_cursor.config import Config
from quant_cursor.rate_limit import RateLimitedClient
from quant_cursor.utils import etf_code_to_symbol, index_code_to_symbol, stock_code_to_symbol

logger = logging.getLogger(__name__)

# 大盘/宽基核心指数（用于 category=major / broad 标记）
MAJOR_INDEX_CODES = {
    "000001",  # 上证指数
    "399001",  # 深证成指
    "399006",  # 创业板指
    "399102",  # 创业板综
    "000688",  # 科创50
    "000016",  # 上证50
    "000300",  # 沪深300
    "000905",  # 中证500
    "000852",  # 中证1000
    "932000",  # 中证2000
    "000510",  # 中证A500
    "930050",  # 中证A50
    "000903",  # 中证A100
    "399330",  # 深证100
    "399673",  # 创业板50
}

BOND_CATEGORIES = {"利率债", "信用债", "综合债", "可转债", "综合"}

CATEGORY_MAP = {
    "规模": "broad",
    "行业": "industry",
    "风格": "style",
    "主题": "theme",
    "策略": "strategy",
    "综合": "composite",
}

BROAD_ETF_KEYWORDS = (
    "A50",
    "A500",
    "2000",
    "1000",
    "500",
    "300",
    "沪深",
    "创业板",
    "科创",
    "上证50",
    "深证100",
    "MSCI",
    "红利",
    "宽基",
)


def _classify_index(code: str, name: str, csi_category: str | None) -> str:
    if code in MAJOR_INDEX_CODES:
        return "major"
    if csi_category == "规模":
        return "broad"
    if csi_category == "行业":
        return "industry"
    if csi_category and csi_category in CATEGORY_MAP:
        return CATEGORY_MAP[csi_category]
    if any(kw in name for kw in ("银行", "白酒", "医药", "消费", "能源", "证券", "地产")):
        return "industry"
    return "other"


def _classify_etf(name: str) -> str:
    if any(kw in name for kw in BROAD_ETF_KEYWORDS):
        return "broad_etf"
    return "etf"


def _classify_industry_etf(name: str) -> str:
    industry_kw = (
        "银行",
        "白酒",
        "医药",
        "消费",
        "能源",
        "证券",
        "地产",
        "军工",
        "芯片",
        "半导体",
        "新能源",
        "光伏",
        "钢铁",
        "煤炭",
        "有色",
        "化工",
        "农业",
        "食品",
        "汽车",
        "电子",
    )
    if any(kw in name for kw in industry_kw):
        return "industry_etf"
    return _classify_etf(name)


def build_sw_l2_mapping(config: Config, client: RateLimitedClient, *, force: bool = False) -> pd.DataFrame:
    """构建申万二级行业 -> 成分股映射，缓存至 meta。"""
    cache_path = config.meta_dir / "sw_l2_constituents.parquet"
    config.meta_dir.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force:
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime, tz=ZoneInfo(config.timezone))
        age_days = (datetime.now(ZoneInfo(config.timezone)) - mtime).days
        if age_days < config.sw_l2_mapping_refresh_days:
            logger.info("使用缓存申万二级映射: %s", cache_path)
            return pd.read_parquet(cache_path)

    logger.info("拉取申万二级行业列表 sw_index_second_info ...")
    l2_df = client.call(ak.sw_index_second_info)
    records: list[dict] = []

    for _, row in l2_df.iterrows():
        raw_code = str(row.get("行业代码", row.iloc[0])).strip()
        sw_code = raw_code.split(".")[0]
        sw_name = str(row.get("行业名称", row.iloc[1])).strip()
        try:
            cons = client.call(ak.index_component_sw, symbol=sw_code)
        except Exception as exc:  # noqa: BLE001
            logger.warning("申万二级 %s %s 成分拉取失败: %s", sw_code, sw_name, exc)
            continue
        for _, s in cons.iterrows():
            records.append(
                {
                    "code": str(s["证券代码"]).strip().zfill(6),
                    "sw_l2_code": sw_code,
                    "sw_l2_name": sw_name,
                }
            )

    if not records:
        if cache_path.exists():
            logger.warning("申万映射拉取失败，回退缓存")
            return pd.read_parquet(cache_path)
        raise RuntimeError("申万二级成分映射为空")

    mapping = pd.DataFrame(records).drop_duplicates(["code", "sw_l2_code"])
    mapping.to_parquet(cache_path, index=False)
    logger.info("申万二级映射已保存: %s (%s 条)", cache_path, len(mapping))
    return mapping


def build_stock_universe(config: Config, client: RateLimitedClient) -> pd.DataFrame:
    """拉取沪深京 A 股列表并标注申万二级 / 大盘股。"""
    logger.info("拉取 A 股列表 stock_zh_a_spot_em ...")
    spot = client.call(ak.stock_zh_a_spot_em)
    sw_map = build_sw_l2_mapping(config, client)
    sw_primary = sw_map.drop_duplicates("code").set_index("code")

    records: list[dict] = []
    seen: set[str] = set()
    for _, row in spot.iterrows():
        code = str(row["代码"]).strip().zfill(6)
        if code in seen:
            continue
        seen.add(code)
        name = str(row["名称"]).strip()
        total_mv = pd.to_numeric(row.get("总市值"), errors="coerce")
        category = "stock"
        if pd.notna(total_mv) and total_mv >= config.large_cap_min_mv:
            category = "large_cap"

        sw_l2_code = ""
        sw_l2_name = ""
        if code in sw_primary.index:
            sw_l2_code = str(sw_primary.loc[code, "sw_l2_code"])
            sw_l2_name = str(sw_primary.loc[code, "sw_l2_name"])

        records.append(
            {
                "code": code,
                "name": name,
                "symbol": stock_code_to_symbol(code),
                "asset_type": "stock",
                "category": category,
                "source": "stock_em",
                "sw_l2_code": sw_l2_code,
                "sw_l2_name": sw_l2_name,
                "total_mv": float(total_mv) if pd.notna(total_mv) else None,
            }
        )

    stocks = pd.DataFrame(records)
    logger.info(
        "A 股标的池: %s 个 (large_cap=%s, 有申万二级=%s)",
        len(stocks),
        (stocks["category"] == "large_cap").sum(),
        (stocks["sw_l2_code"] != "").sum(),
    )
    return stocks


def merge_universe_parts(*parts: pd.DataFrame) -> pd.DataFrame:
    if not parts:
        raise ValueError("merge_universe_parts 需要至少一个 DataFrame")
    merged = pd.concat(parts, ignore_index=True)
    merged["code"] = merged["code"].astype(str).str.zfill(6)
    merged = merged.sort_values(["asset_type", "category", "code"])
    merged = merged.drop_duplicates(subset=["asset_type", "code"], keep="last")
    return merged.reset_index(drop=True)


def _append_records(
    records: list[dict],
    seen: set[tuple[str, str]],
    code: str,
    name: str,
    asset_type: str,
    category: str,
    source: str,
) -> None:
    code = str(code).strip().zfill(6)
    key = (asset_type, code)
    if key in seen:
        return
    seen.add(key)

    if asset_type == "index":
        symbol = index_code_to_symbol(code)
    elif asset_type == "stock":
        symbol = stock_code_to_symbol(code)
    else:
        symbol = etf_code_to_symbol(code)

    records.append(
        {
            "code": code,
            "name": name,
            "symbol": symbol,
            "asset_type": asset_type,
            "category": category,
            "source": source,
            "sw_l2_code": "",
            "sw_l2_name": "",
            "total_mv": None,
        }
    )


def build_universe(config: Config) -> pd.DataFrame:
    """汇总 akshare 多数据源，构建指数 + 宽基 ETF 标的池。"""
    client = RateLimitedClient(
        delay=config.request_delay,
        jitter=config.request_jitter,
        max_retries=config.max_retries,
        backoff=config.retry_backoff,
        batch_pause_every=config.batch_pause_every,
        batch_pause_seconds=config.batch_pause_seconds,
    )

    records: list[dict] = []
    seen: set[tuple[str, str]] = set()

    logger.info("拉取中证指数全量列表 index_csindex_all ...")
    csi_df = client.call(ak.index_csindex_all)
    for _, row in csi_df.iterrows():
        code = str(row["指数代码"]).strip()
        name = str(row["指数简称"]).strip()
        csi_category = str(row.get("指数类别", "")).strip()
        if not config.include_bond_indices and csi_category in BOND_CATEGORIES:
            continue
        category = _classify_index(code, name, csi_category or None)
        _append_records(records, seen, code, name, "index", category, "csindex")

    logger.info("拉取新浪指数列表 index_stock_info ...")
    sina_df = client.call(ak.index_stock_info)
    for _, row in sina_df.iterrows():
        code = str(row["index_code"]).strip()
        name = str(row["display_name"]).strip()
        category = _classify_index(code, name, None)
        _append_records(records, seen, code, name, "index", category, "sina")

    for cat in config.em_index_categories:
        logger.info("拉取东财指数分类 stock_zh_index_spot_em: %s", cat)
        try:
            em_df = client.call(ak.stock_zh_index_spot_em, symbol=cat)
        except Exception as exc:  # noqa: BLE001
            logger.warning("东财分类 %s 拉取失败，已跳过: %s", cat, exc)
            continue
        for _, row in em_df.iterrows():
            code = str(row["代码"]).strip()
            name = str(row["名称"]).strip()
            category = _classify_index(code, name, None)
            _append_records(records, seen, code, name, "index", category, f"em:{cat}")

    if config.include_etf:
        logger.info("拉取 ETF 列表 fund_etf_spot_em ...")
        etf_df = client.call(ak.fund_etf_spot_em)
        for _, row in etf_df.iterrows():
            code = str(row["代码"]).strip()
            name = str(row["名称"]).strip()
            category = _classify_industry_etf(name)
            _append_records(records, seen, code, name, "etf", category, "etf_em")

    index_etf = pd.DataFrame(records)
    if index_etf.empty:
        raise RuntimeError("未获取到任何指数/ETF 标的")

    parts = [index_etf]
    if config.include_stocks:
        stock_client = RateLimitedClient(
            delay=config.stock_request_delay,
            jitter=config.request_jitter,
            max_retries=config.max_retries,
            backoff=config.retry_backoff,
            batch_pause_every=config.stock_batch_pause_every,
            batch_pause_seconds=config.stock_batch_pause_seconds,
            ban_consecutive_failures=config.ban_consecutive_failures,
            ban_cooldown_seconds=config.ban_cooldown_seconds,
        )
        stocks = build_stock_universe(config, stock_client)
        parts.append(stocks)

    universe = merge_universe_parts(*parts)
    logger.info(
        "标的池构建完成: 指数 %s 个, ETF %s 个, A股 %s 个",
        (universe["asset_type"] == "index").sum(),
        (universe["asset_type"] == "etf").sum(),
        (universe["asset_type"] == "stock").sum(),
    )
    return universe


def save_universe(universe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(path, index=False)
    logger.info("标的池已保存: %s (%s 条)", path, len(universe))


def load_universe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"标的池不存在: {path}，请先运行 `python -m quant_cursor universe`"
        )
    return pd.read_parquet(path)


def is_today_data_ready(config: Config, now: datetime | None = None) -> bool:
    tz = ZoneInfo(config.timezone)
    current = now or datetime.now(tz)
    if current.weekday() >= 5:
        return True
    return current.hour >= config.today_update_hour
