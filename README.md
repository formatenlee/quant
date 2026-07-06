# Quant Cursor

面向中国有价证券（指数 + 宽基 ETF）的分析预测项目。第一步实现基于 [AKShare](https://github.com/akfamily/akshare) 的行情数据采集，支持全量与当日增量下载，并内置请求延迟与批量暂停以降低封 IP 风险。

## 环境

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## 快速开始

### 1. 构建标的池

从 AKShare 汇总中证指数、新浪指数、东财分类指数及 ETF 列表：

```bash
python -m quant_cursor universe
```

输出：`data/meta/universe.parquet`

标的分类说明：

| category | 含义 |
|----------|------|
| major | 大盘核心指数（上证、深成指、创业板、沪深300 等） |
| broad | 宽基指数（中证500/1000/2000、A50、A500 等） |
| industry | 行业指数（银行、白酒、医药等） |
| broad_etf | 宽基 ETF |
| etf | 其他 ETF |

### 2. 全量下载历史行情

```bash
python -m quant_cursor download --mode full
```

数据保存至：

- 指数：`data/indices/{code}.parquet`
- ETF：`data/etf/{code}.parquet`

### 3. 当日增量更新

指数/ETF 日线一般在 **21:00 后** 更新（可在 `config.yaml` 调整 `today_update_hour`）：

```bash
python -m quant_cursor download --mode today
```

若需在 21:00 前强制拉取：

```bash
python -m quant_cursor download --mode today --force
```

### 4. 按类型/代码过滤

```bash
# 仅大盘与宽基指数
python -m quant_cursor download --mode full --asset-type index --category major,broad

# 仅宽基 ETF
python -m quant_cursor download --mode full --asset-type etf --category broad_etf

# 指定代码
python -m quant_cursor download --mode today --codes 000001,399006,510050
```

### 5. 查看标的池

```bash
python -m quant_cursor list --asset-type index --category industry
```

## 配置

编辑 `config.yaml`：

```yaml
request_delay: 2.0        # 每次请求间隔（秒）
request_jitter: 0.8       # 随机抖动
batch_pause_every: 40     # 每 N 次请求长暂停
batch_pause_seconds: 25   # 长暂停时长
today_update_hour: 21     # 当日数据可用时间
include_etf: true
include_bond_indices: false
```

若频繁出现连接中断，可适当 **增大** `request_delay` 与 `batch_pause_seconds`。

## 数据源

| 用途 | AKShare 接口 |
|------|-------------|
| 中证指数列表 | `index_csindex_all` |
| 新浪指数列表 | `index_stock_info` |
| 东财指数分类 | `stock_zh_index_spot_em` |
| ETF 列表 | `fund_etf_spot_em` |
| 指数历史 | `stock_zh_index_daily`（主）/ `stock_zh_index_daily_em`（备） |
| ETF 历史 | `fund_etf_hist_sina`（主）/ `fund_etf_hist_em`（备） |

## 导入 Microsoft Qlib

将已下载的 parquet 转为 Qlib 二进制格式，供后续因子研究与模型训练使用。

```bash
pip install -r requirements-qlib.txt

# 一步完成：导出中间格式 + 转 .bin
python -m quant_cursor qlib all

# 或分步执行
python -m quant_cursor qlib export    # -> data/qlib_staging/
python -m quant_cursor qlib dump      # -> data/qlib_data/
```

在 Python 中使用（路径按实际修改）：

```python
import qlib
from qlib.constant import REG_CN
from qlib.data import D

qlib.init(provider_uri="e:/quant_cursor/data/qlib_data", region=REG_CN)

# 上证50指数 SH000016，沪深300 ETF 等
df = D.features(
    ["SH000016", "SH510050"],
    ["$open", "$close", "$high", "$low", "$volume"],
    start_time="2020-01-01",
    end_time="2026-12-31",
)
```

标的代码映射规则（Qlib instrument id）：

| 类型 | 原始代码 | Qlib ID 示例 |
|------|----------|--------------|
| 上交所指数/ETF | 000016, 510050 | SH000016, SH510050 |
| 深交所指数/ETF | 399006, 159915 | SZ399006, SZ159915 |
| 中证字母代码 | H30590 | IDX_H30590 |

清单见 `data/meta/qlib_manifest.parquet`，初始化示例见 `data/meta/qlib_init_example.py`。

## 项目结构

```
quant_cursor/
├── config.yaml
├── requirements.txt
├── requirements-qlib.txt
├── quant_cursor/
│   ├── cli.py           # 命令行入口
│   ├── config.py        # 配置加载
│   ├── universe.py      # 标的池构建
│   ├── downloader.py    # 行情下载
│   ├── qlib_export.py   # Qlib 导出
│   ├── qlib_cli.py      # Qlib 子命令
│   ├── rate_limit.py    # 限速与重试
│   └── utils.py         # 工具函数
├── scripts/
│   ├── qlib/dump_bin.py # Qlib 官方转换脚本
│   └── verify_data.py   # 数据完整性核查
└── data/
    ├── meta/            # 标的池、下载报告
    ├── indices/         # 指数 parquet
    ├── etf/             # ETF parquet
    ├── qlib_staging/    # Qlib 中间 parquet
    └── qlib_data/       # Qlib .bin 数据
```
