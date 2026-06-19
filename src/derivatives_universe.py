from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DerivativeSpec:
    code: str
    qlib_id: str
    name: str
    product: str
    asset_class: str  # financial_futures | index_option
    exchange: str
    underlying_qlib_id: str | None = None
    ak_symbol: str | None = None


# 中金所金融期货主力连续（新浪 IF0 / IH0 …）
FINANCIAL_FUTURES: tuple[DerivativeSpec, ...] = (
    DerivativeSpec("IF0", "FF_IF", "沪深300股指期货主力", "IF", "financial_futures", "CFFEX", "SH000300", "IF0"),
    DerivativeSpec("IH0", "FF_IH", "上证50股指期货主力", "IH", "financial_futures", "CFFEX", "SH000016", "IH0"),
    DerivativeSpec("IC0", "FF_IC", "中证500股指期货主力", "IC", "financial_futures", "CFFEX", "SH000905", "IC0"),
    DerivativeSpec("IM0", "FF_IM", "中证1000股指期货主力", "IM", "financial_futures", "CFFEX", "SH000852", "IM0"),
    DerivativeSpec("T0", "FF_T", "10年期国债期货主力", "T", "financial_futures", "CFFEX", None, "T0"),
    DerivativeSpec("TF0", "FF_TF", "5年期国债期货主力", "TF", "financial_futures", "CFFEX", None, "TF0"),
    DerivativeSpec("TS0", "FF_TS", "2年期国债期货主力", "TS", "financial_futures", "CFFEX", None, "TS0"),
    DerivativeSpec("TL0", "FF_TL", "30年期国债期货主力", "TL", "financial_futures", "CFFEX", None, "TL0"),
)

# 股指/ETF 期权品种（事件日历用；日线由底层 ETF/指数已在 universe 中）
INDEX_ETF_OPTIONS: tuple[DerivativeSpec, ...] = (
    DerivativeSpec("50ETF", "OPT_50ETF", "上证50ETF期权", "50ETF", "index_option", "SSE", "SH510050"),
    DerivativeSpec("300ETF", "OPT_300ETF", "沪深300ETF期权", "300ETF", "index_option", "SSE", "SH510300"),
    DerivativeSpec("500ETF", "OPT_500ETF", "中证500ETF期权", "500ETF", "index_option", "SSE", "SH510500"),
    DerivativeSpec("科创50ETF", "OPT_KC50ETF", "科创50ETF期权", "KC50ETF", "index_option", "SSE", "SH588000"),
    DerivativeSpec("创业板ETF", "OPT_CYETF", "创业板ETF期权", "CYETF", "index_option", "SSE", "SH159915"),
    DerivativeSpec("SZ50ETF", "OPT_SZ50ETF", "深证50ETF期权", "50ETF", "index_option", "SZSE", "SZ159901"),
    DerivativeSpec("SZ300ETF", "OPT_SZ300ETF", "沪深300ETF期权(深)", "300ETF", "index_option", "SZSE", "SZ159919"),
    DerivativeSpec("HO", "OPT_HO", "上证50股指期权", "HO", "index_option", "CFFEX", "SH000016"),
    DerivativeSpec("IO", "OPT_IO", "沪深300股指期权", "IO", "index_option", "CFFEX", "SH000300"),
    DerivativeSpec("MO", "OPT_MO", "中证1000股指期权", "MO", "index_option", "CFFEX", "SH000852"),
)

CALENDAR_QLIB_ID = "CAL_DERIV"

# 写入 Qlib 的日历特征列（0/1 标志）
EVENT_CALENDAR_FIELDS: tuple[str, ...] = (
    "is_50etf_opt_expiry",
    "is_50etf_opt_exercise",
    "is_50etf_opt_settle",
    "is_300etf_opt_expiry",
    "is_500etf_opt_expiry",
    "is_cffex_if_last_trade",
    "is_cffex_ih_last_trade",
    "is_cffex_ic_last_trade",
    "is_cffex_im_last_trade",
    "is_cffex_io_last_trade",
    "is_cffex_ho_last_trade",
    "is_cffex_mo_last_trade",
    "is_treasury_fut_last_trade",
    "n_deriv_events",
)


def all_derivative_specs() -> list[DerivativeSpec]:
    return list(FINANCIAL_FUTURES) + list(INDEX_ETF_OPTIONS)


def futures_specs() -> list[DerivativeSpec]:
    return list(FINANCIAL_FUTURES)


def option_specs() -> list[DerivativeSpec]:
    return list(INDEX_ETF_OPTIONS)
