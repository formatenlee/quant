"""
BPC-v3: 方向/结构层级行为本体（与 bpc v2 源码隔离）。

- 数据：bpc_v3/dataset.py（相对 OHLCV + vol_context）
- 特征：bpc_v3/features.py（26 维）
- 行为：bpc_v3/behavior_features.py（符号代理 + 固定阈值）
- 模型：复用 bpc.train 流程，运行时注入 BPCv3
"""

from quant_cursor.bpc_v3.behavior_features import (
    BEHAVIOR_AGENT_NAMES,
    BEHAVIOR_LOGITS_DIM,
    CORE_AGENTS,
    EXTENDED_AGENTS,
    NUM_BEHAVIOR_AGENTS,
)
from quant_cursor.bpc_v3.dataset import (
    MaterializedMultiScaleDataset,
    TemporalSplit,
    build_datasets,
    ensure_qlib,
    load_qlib_instruments,
    load_trading_calendar,
)
from quant_cursor.bpc_v3.feature_dims import DAY_FULL_FEAT_DIM, DAY_STRUCT_FEAT_DIM, GROUP_DIM_MAP
from quant_cursor.bpc_v3.model import BPCv3, DEFAULT_REGISTRY, ScaleRegistry, build_scale_registry

__all__ = [
    "BEHAVIOR_AGENT_NAMES",
    "BEHAVIOR_LOGITS_DIM",
    "BPCv3",
    "CORE_AGENTS",
    "DAY_FULL_FEAT_DIM",
    "DAY_STRUCT_FEAT_DIM",
    "DEFAULT_REGISTRY",
    "EXTENDED_AGENTS",
    "GROUP_DIM_MAP",
    "MaterializedMultiScaleDataset",
    "NUM_BEHAVIOR_AGENTS",
    "ScaleRegistry",
    "TemporalSplit",
    "build_datasets",
    "build_scale_registry",
    "ensure_qlib",
    "load_qlib_instruments",
    "load_trading_calendar",
]
