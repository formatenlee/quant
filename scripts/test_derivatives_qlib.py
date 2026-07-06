import sys

sys.stdout.reconfigure(encoding="utf-8")

from quant_cursor.qlib_data import (
    init_qlib,
    load_derivative_event_flags,
    load_financial_futures,
    list_50etf_expiry_dates,
)

init_qlib()

ff = load_financial_futures(["FF_IF"], start="2024-06-01", end="2024-06-30")
print("FF_IF sample\n", ff.dropna(subset=["$close"]).tail(5))

evt = load_derivative_event_flags(
    "2024-01-01",
    "2024-12-31",
    fields=["is_50etf_opt_expiry", "is_cffex_if_last_trade"],
)
print("evt cols", evt.columns.tolist())
print("50ETF expiry hits", evt[evt["is_50etf_opt_expiry"] > 0].head())

ex = list_50etf_expiry_dates()
print("50etf expiry count", len(ex), "last", ex.tail(3).tolist())
