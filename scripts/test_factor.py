import qlib
from qlib.constant import REG_CN
from qlib.data import D

if __name__ == "__main__":
    qlib.init(provider_uri="e:/quant_cursor/data/qlib_data", region=REG_CN)
    fields = ["$open", "$close", "$factor"]
    for inst in ["SH510050", "SH000016"]:
        df = D.features([inst], fields, "2015-01-01", "2024-12-31", "day")
        df = df.droplevel(0)
        print(inst, "factor nunique", df["$factor"].nunique(), "sample", df["$factor"].dropna().head(3).tolist())
