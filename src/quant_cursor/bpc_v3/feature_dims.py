"""BPC-v3 预计算特征维度（含 trend_structure）。"""

from __future__ import annotations

from quant_cursor.bpc_v3.behavior_features import NUM_BEHAVIOR_AGENTS

GROUP_DIM_MAP: dict[str, int] = {
    "price_structure": 7,
    "volume_structure": 4,
    "attack_proxy": 3,
    "micro_proxy": 2,
    "trend_structure": 5,
    "time_structure": 4,
    "behavior_structure": NUM_BEHAVIOR_AGENTS,
}

# price(7) + volume(4) + attack(3) + micro(2) + trend(5)
DAY_STRUCT_FEAT_DIM = 21
DAY_BEHAVIOR_FEAT_DIM = NUM_BEHAVIOR_AGENTS
DAY_FULL_FEAT_DIM = DAY_STRUCT_FEAT_DIM + DAY_BEHAVIOR_FEAT_DIM
WEEK_FEAT_DIM = 7

# 预计算路径固定缩放，避免 trend_structure 淹没 price/volume 结构特征
TREND_STRUCTURE_SCALE = 0.5
TREND_STRUCTURE_SLICE = slice(16, 16 + GROUP_DIM_MAP["trend_structure"])
