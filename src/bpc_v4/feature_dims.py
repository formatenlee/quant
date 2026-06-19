"""BPC-v3 预计算特征维度（含 trend_structure）。"""

from __future__ import annotations

from .behavior_features import NUM_BEHAVIOR_AGENTS

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

# rv 典型值 ~0.01–0.03；静态放大后再 /anchor_rv，与 attack/micro 等同量级（MSE 重构）
RV_FEATURE_SCALE = 20.0

# 结构特征：rv/range 等波动敏感维由 vol_context 动态缩放（见 features.apply_struct_group_scales）
STRUCT_GROUP_SCALE: dict[str, tuple[float, ...]] = {
    "price_structure": (RV_FEATURE_SCALE, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    "volume_structure": (4.0, 4.0, 1.0, 1.0),
    "attack_proxy": (3.0, 5.0, 1.0),
    "micro_proxy": (12.0, 1.0),
    "trend_structure": (1.0, 1.0, 1.0, 1.0, 2.0),
}

WEEK_GROUP_SCALE: tuple[float, ...] = (
    RV_FEATURE_SCALE,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
)

ENCODER_INPUT_SCHEMA = "struct_only_v1"

FEATURE_SCALE_SCHEMA = "vol_adaptive_v2"

_STRUCT_GROUP_ORDER: tuple[str, ...] = (
    "price_structure",
    "volume_structure",
    "attack_proxy",
    "micro_proxy",
    "trend_structure",
)

STRUCT_FEATURE_SCALE: tuple[float, ...] = tuple(
    s for g in _STRUCT_GROUP_ORDER for s in STRUCT_GROUP_SCALE[g]
)

DAY_FEATURE_SLICES: dict[str, slice] = {}
_offset = 0
for _g in (*_STRUCT_GROUP_ORDER, "behavior_structure"):
    _d = GROUP_DIM_MAP[_g]
    DAY_FEATURE_SLICES[_g] = slice(_offset, _offset + _d)
    _offset += _d
BEHAVIOR_FEATURE_SLICE = DAY_FEATURE_SLICES["behavior_structure"]
