from __future__ import annotations

import calendar
import logging
from datetime import date
from pathlib import Path

import akshare as ak
import pandas as pd

from quant_cursor.config import Config
from quant_cursor.derivatives_universe import EVENT_CALENDAR_FIELDS, INDEX_ETF_OPTIONS
from quant_cursor.rate_limit import RateLimitedClient

logger = logging.getLogger(__name__)

EVENT_COLUMNS = [
    "event_date",
    "event_type",
    "product",
    "exchange",
    "qlib_id",
    "underlying_qlib_id",
    "contract_code",
    "delivery_month",
    "source",
    "meta",
]


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    cal = calendar.Calendar(firstweekday=calendar.MONDAY)
    count = 0
    for day in cal.itermonthdates(year, month):
        if day.month != month:
            continue
        if day.weekday() == weekday:
            count += 1
            if count == n:
                return day
    raise ValueError(f"no {n}th weekday={weekday} in {year}-{month:02d}")


def _adjust_to_trading_day(d: date, trading_days: set[date]) -> date:
    cur = d
    for _ in range(14):
        if cur in trading_days:
            return cur
        cur += pd.Timedelta(days=1)
    return d


def _load_trading_calendar(config: Config) -> list[date]:
    cal_path = config.qlib_data_dir / "calendars" / "day.txt"
    if cal_path.exists():
        days = pd.read_csv(cal_path, header=None)[0]
        return [pd.Timestamp(x).date() for x in days.tolist()]

    idx = pd.bdate_range("2010-01-01", pd.Timestamp.today() + pd.DateOffset(years=1))
    return [x.date() for x in idx]


def cffex_index_last_trade(year: int, month: int, trading_days: set[date]) -> date:
    raw = _nth_weekday(year, month, calendar.FRIDAY, 3)
    return _adjust_to_trading_day(raw, trading_days)


def cffex_treasury_last_trade(year: int, month: int, trading_days: set[date]) -> date:
    raw = _nth_weekday(year, month, calendar.FRIDAY, 2)
    return _adjust_to_trading_day(raw, trading_days)


def sse_etf_option_expiry(year: int, month: int, trading_days: set[date]) -> date:
    raw = _nth_weekday(year, month, calendar.WEDNESDAY, 4)
    return _adjust_to_trading_day(raw, trading_days)


def _month_range(start: str, end: str) -> list[tuple[int, int]]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    out: list[tuple[int, int]] = []
    cur = pd.Timestamp(year=s.year, month=s.month, day=1)
    while cur <= e:
        out.append((cur.year, cur.month))
        cur += pd.DateOffset(months=1)
    return out


def _append_event(
    rows: list[dict],
    *,
    event_date: date | str,
    event_type: str,
    product: str,
    exchange: str,
    qlib_id: str,
    underlying_qlib_id: str | None,
    delivery_month: str,
    source: str,
    contract_code: str = "",
    meta: str = "",
) -> None:
    rows.append(
        {
            "event_date": pd.Timestamp(event_date).normalize(),
            "event_type": event_type,
            "product": product,
            "exchange": exchange,
            "qlib_id": qlib_id,
            "underlying_qlib_id": underlying_qlib_id or "",
            "contract_code": contract_code,
            "delivery_month": delivery_month,
            "source": source,
            "meta": meta,
        }
    )


def _build_rule_based_events(
    trading_days: set[date],
    start: str = "2010-04-01",
    end: str | None = None,
) -> list[dict]:
    end = end or str(pd.Timestamp.today().date() + pd.DateOffset(months=18))[:10]
    rows: list[dict] = []

    for year, month in _month_range(start, end):
        ym = f"{year}{month:02d}"
        lt_if = cffex_index_last_trade(year, month, trading_days)
        for product, qlib_id, und in (
            ("IF", "FF_IF", "SH000300"),
            ("IH", "FF_IH", "SH000016"),
            ("IC", "FF_IC", "SH000905"),
            ("IM", "FF_IM", "SH000852"),
            ("IO", "OPT_IO", "SH000300"),
            ("HO", "OPT_HO", "SH000016"),
            ("MO", "OPT_MO", "SH000852"),
        ):
            _append_event(
                rows,
                event_date=lt_if,
                event_type="last_trade",
                product=product,
                exchange="CFFEX",
                qlib_id=qlib_id,
                underlying_qlib_id=und,
                delivery_month=ym,
                source="rule_cffex_3rd_fri",
            )

        lt_t = cffex_treasury_last_trade(year, month, trading_days)
        for product, qlib_id in (("T", "FF_T"), ("TF", "FF_TF"), ("TS", "FF_TS"), ("TL", "FF_TL")):
            _append_event(
                rows,
                event_date=lt_t,
                event_type="last_trade",
                product=product,
                exchange="CFFEX",
                qlib_id=qlib_id,
                underlying_qlib_id=None,
                delivery_month=ym,
                source="rule_cffex_2nd_fri",
            )

        if pd.Timestamp(year, month, 1) >= pd.Timestamp("2015-02-01"):
            exp = sse_etf_option_expiry(year, month, trading_days)
            settle = _adjust_to_trading_day(exp + pd.Timedelta(days=1), trading_days)
            for spec in INDEX_ETF_OPTIONS:
                if spec.exchange != "SSE" or spec.product in ("KC50ETF", "CYETF"):
                    continue
                if spec.product == "50ETF" and pd.Timestamp(exp) < pd.Timestamp("2015-02-09"):
                    continue
                if spec.product == "300ETF" and pd.Timestamp(exp) < pd.Timestamp("2019-12-23"):
                    continue
                if spec.product == "500ETF" and pd.Timestamp(exp) < pd.Timestamp("2022-09-19"):
                    continue
                _append_event(
                    rows,
                    event_date=exp,
                    event_type="expiry",
                    product=spec.product,
                    exchange="SSE",
                    qlib_id=spec.qlib_id,
                    underlying_qlib_id=spec.underlying_qlib_id,
                    delivery_month=ym,
                    source="rule_sse_4th_wed",
                )
                _append_event(
                    rows,
                    event_date=exp,
                    event_type="exercise",
                    product=spec.product,
                    exchange="SSE",
                    qlib_id=spec.qlib_id,
                    underlying_qlib_id=spec.underlying_qlib_id,
                    delivery_month=ym,
                    source="rule_sse_4th_wed",
                )
                _append_event(
                    rows,
                    event_date=settle,
                    event_type="settlement",
                    product=spec.product,
                    exchange="SSE",
                    qlib_id=spec.qlib_id,
                    underlying_qlib_id=spec.underlying_qlib_id,
                    delivery_month=ym,
                    source="rule_sse_4th_wed",
                )

    return rows


def _fetch_sse_live_events(client: RateLimitedClient) -> list[dict]:
    rows: list[dict] = []

    def _pull():
        return ak.option_current_day_sse()

    try:
        df = client.call(_pull)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SSE 期权当日列表不可用: %s", exc)
        return rows

    for _, r in df.iterrows():
        name = str(r.get("合约简称", ""))
        underlying = str(r.get("标的券名称及代码", ""))
        product = "50ETF"
        qlib_id = "OPT_50ETF"
        und = "SH510050"
        if "300ETF" in name or "300ETF" in underlying:
            product, qlib_id, und = "300ETF", "OPT_300ETF", "SH510300"
        elif "500ETF" in name or "500ETF" in underlying:
            product, qlib_id, und = "500ETF", "OPT_500ETF", "SH510500"
        elif "科创" in name:
            product, qlib_id, und = "KC50ETF", "OPT_KC50ETF", "SH588000"
        elif "创业板" in name:
            product, qlib_id, und = "CYETF", "OPT_CYETF", "SH159915"

        expiry = pd.to_datetime(str(r.get("到期日", "")), errors="coerce")
        exercise = pd.to_datetime(str(r.get("期权行权日", "")), errors="coerce")
        settle = pd.to_datetime(str(r.get("行权交收日", "")), errors="coerce")
        code = str(r.get("合约交易代码", ""))
        if pd.notna(expiry):
            _append_event(
                rows,
                event_date=expiry.date(),
                event_type="expiry",
                product=product,
                exchange="SSE",
                qlib_id=qlib_id,
                underlying_qlib_id=und,
                delivery_month=expiry.strftime("%Y%m"),
                source="akshare_sse_live",
                contract_code=code,
            )
        if pd.notna(exercise):
            _append_event(
                rows,
                event_date=exercise.date(),
                event_type="exercise",
                product=product,
                exchange="SSE",
                qlib_id=qlib_id,
                underlying_qlib_id=und,
                delivery_month=exercise.strftime("%Y%m"),
                source="akshare_sse_live",
                contract_code=code,
            )
        if pd.notna(settle):
            _append_event(
                rows,
                event_date=settle.date(),
                event_type="settlement",
                product=product,
                exchange="SSE",
                qlib_id=qlib_id,
                underlying_qlib_id=und,
                delivery_month=settle.strftime("%Y%m"),
                source="akshare_sse_live",
                contract_code=code,
            )
    return rows


def _fetch_szse_live_events(client: RateLimitedClient) -> list[dict]:
    rows: list[dict] = []

    def _pull():
        return ak.option_current_day_szse()

    try:
        df = client.call(_pull)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SZSE 期权当日列表不可用: %s", exc)
        return rows

    for _, r in df.iterrows():
        name = str(r.get("合约简称", ""))
        product = "50ETF"
        qlib_id = "OPT_SZ50ETF"
        und = "SZ159901"
        if "300" in name:
            product, qlib_id, und = "300ETF", "OPT_SZ300ETF", "SZ159919"

        code = str(r.get("合约编码", ""))
        for col, etype in (
            ("最后交易日", "last_trade"),
            ("行权日", "exercise"),
            ("到期日", "expiry"),
            ("交收日", "settlement"),
        ):
            if col not in df.columns:
                continue
            dt = pd.to_datetime(r.get(col), errors="coerce")
            if pd.isna(dt):
                continue
            _append_event(
                rows,
                event_date=dt.date(),
                event_type=etype,
                product=product,
                exchange="SZSE",
                qlib_id=qlib_id,
                underlying_qlib_id=und,
                delivery_month=dt.strftime("%Y%m"),
                source="akshare_szse_live",
                contract_code=code,
            )
    return rows


def _fetch_sse_sina_expire(
    client: RateLimitedClient,
    symbol: str,
    qlib_id: str,
    und: str,
    *,
    max_months: int = 3,
) -> list[dict]:
    rows: list[dict] = []
    fast = RateLimitedClient(delay=client.delay, max_retries=1, backoff=client.backoff)

    def _list_months():
        return ak.option_sse_list_sina(symbol=symbol)

    try:
        months = fast.call(_list_months)
    except Exception as exc:  # noqa: BLE001
        logger.debug("SSE 月份列表 %s 失败（已用规则日历兜底）: %s", symbol, exc)
        return rows

    for ym in months[-max_months:]:
        if len(ym) != 6 or not ym.isdigit():
            continue

        def _expire(m=ym):
            return ak.option_sse_expire_day_sina(trade_date=m, symbol=symbol)

        try:
            expire_day, _ = fast.call(_expire)
        except Exception:
            continue
        exp = pd.to_datetime(expire_day, errors="coerce")
        if pd.isna(exp):
            continue
        _append_event(
            rows,
            event_date=exp.date(),
            event_type="expiry",
            product=symbol,
            exchange="SSE",
            qlib_id=qlib_id,
            underlying_qlib_id=und,
            delivery_month=ym,
            source="akshare_sse_sina",
        )
        _append_event(
            rows,
            event_date=exp.date(),
            event_type="exercise",
            product=symbol,
            exchange="SSE",
            qlib_id=qlib_id,
            underlying_qlib_id=und,
            delivery_month=ym,
            source="akshare_sse_sina",
        )
    return rows


def build_derivatives_events(config: Config, client: RateLimitedClient | None = None) -> pd.DataFrame:
    client = client or RateLimitedClient(delay=config.request_delay, max_retries=config.max_retries)
    trading_list = _load_trading_calendar(config)
    trading_days = set(trading_list)

    rows = _build_rule_based_events(trading_days)
    rows.extend(_fetch_sse_live_events(client))
    rows.extend(_fetch_szse_live_events(client))

    for symbol, qlib_id, und in (
        ("50ETF", "OPT_50ETF", "SH510050"),
        ("300ETF", "OPT_300ETF", "SH510300"),
        ("500ETF", "OPT_500ETF", "SH510500"),
    ):
        rows.extend(_fetch_sse_sina_expire(client, symbol, qlib_id, und, max_months=3))

    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    df = pd.DataFrame(rows)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.normalize()
    df = df.drop_duplicates(
        subset=["event_date", "event_type", "product", "exchange", "qlib_id", "contract_code"],
        keep="first",
    )
    df = df.sort_values(["event_date", "product", "event_type"]).reset_index(drop=True)
    return df[EVENT_COLUMNS]


def _event_to_flag_column(row: pd.Series) -> str | None:
    p = row["product"]
    t = row["event_type"]
    ex = row["exchange"]

    if p == "50ETF" and ex == "SSE":
        if t == "expiry":
            return "is_50etf_opt_expiry"
        if t == "exercise":
            return "is_50etf_opt_exercise"
        if t == "settlement":
            return "is_50etf_opt_settle"
    if p == "300ETF" and ex == "SSE" and t == "expiry":
        return "is_300etf_opt_expiry"
    if p == "500ETF" and ex == "SSE" and t == "expiry":
        return "is_500etf_opt_expiry"

    if t in ("last_trade", "expiry"):
        mapping = {
            "IF": "is_cffex_if_last_trade",
            "IH": "is_cffex_ih_last_trade",
            "IC": "is_cffex_ic_last_trade",
            "IM": "is_cffex_im_last_trade",
            "IO": "is_cffex_io_last_trade",
            "HO": "is_cffex_ho_last_trade",
            "MO": "is_cffex_mo_last_trade",
        }
        if p in mapping:
            return mapping[p]
        if p in ("T", "TF", "TS", "TL"):
            return "is_treasury_fut_last_trade"
    return None


def build_event_calendar_frame(config: Config, events: pd.DataFrame | None = None) -> pd.DataFrame:
    events = events if events is not None else pd.read_parquet(config.derivatives_events_path)
    trading_list = _load_trading_calendar(config)
    if not trading_list:
        raise FileNotFoundError("无法获取交易日历，请先 dump 股票/指数日线至 Qlib")

    cal = pd.DataFrame({"date": pd.to_datetime(trading_list)})
    for col in EVENT_CALENDAR_FIELDS:
        if col != "n_deriv_events":
            cal[col] = 0.0

    flag_rows: list[dict] = []
    for _, row in events.iterrows():
        col = _event_to_flag_column(row)
        if col is None:
            continue
        flag_rows.append({"date": row["event_date"], "col": col})

    if flag_rows:
        flags = pd.DataFrame(flag_rows)
        for col in EVENT_CALENDAR_FIELDS:
            if col == "n_deriv_events":
                continue
            dates = flags.loc[flags["col"] == col, "date"].drop_duplicates()
            if dates.empty:
                continue
            mask = cal["date"].isin(dates)
            cal.loc[mask, col] = 1.0

    ev = events.copy()
    ev["flag_col"] = ev.apply(_event_to_flag_column, axis=1)
    ev = ev.dropna(subset=["flag_col"])
    counts = ev.groupby("event_date").size().rename("n_deriv_events").reset_index()
    cal = cal.merge(counts, left_on="date", right_on="event_date", how="left")
    cal = cal.drop(columns=["event_date"], errors="ignore")
    cal["n_deriv_events"] = cal["n_deriv_events"].fillna(0.0)

    cal["date"] = cal["date"].dt.strftime("%Y-%m-%d")
    return cal


def save_derivatives_events(config: Config, events: pd.DataFrame) -> Path:
    path: Path = config.derivatives_events_path
    path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(path, index=False)
    logger.info("衍生品事件表: %s 行 -> %s", len(events), path)
    return path
