import qlib
from qlib.constant import REG_CN
from qlib.data import D

if __name__ == "__main__":
    qlib.init(provider_uri="e:/quant_cursor/data/qlib_data", region=REG_CN)
    fields = ["$open", "$high", "$low", "$close", "$volume"]
    df = D.features(
        ["SH000016"],
        fields,
        start_time="2020-01-01",
        end_time="2020-06-01",
        freq="day",
    )
    print("shape:", df.shape)
    print("index names:", df.index.names)
    print(df.head())
