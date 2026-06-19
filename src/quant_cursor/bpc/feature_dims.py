"""预计算特征维度（与 behavior_features 代理数同步）。

volume_structure[0] = 原始成交量相对前半基线的 log 比；[1] = rel_vol.std（非 log 均量）。
vol 代理 / price_structure.rv = 收益率 std，已是相对量，不做二次相对化。
"""

from __future__ import annotations

from quant_cursor.bpc.behavior_features import NUM_BEHAVIOR_AGENTS

# price_structure(7) + volume_structure(4) + attack_proxy(3) + micro_proxy(2)
DAY_STRUCT_FEAT_DIM = 16
DAY_BEHAVIOR_FEAT_DIM = NUM_BEHAVIOR_AGENTS
DAY_FULL_FEAT_DIM = DAY_STRUCT_FEAT_DIM + DAY_BEHAVIOR_FEAT_DIM
WEEK_FEAT_DIM = 7

GROUP_DIM_MAP: dict[str, int] = {
    "price_structure": 7,
    "volume_structure": 4,
    "attack_proxy": 3,
    "micro_proxy": 2,
    "time_structure": 4,
    "behavior_structure": NUM_BEHAVIOR_AGENTS,
}
