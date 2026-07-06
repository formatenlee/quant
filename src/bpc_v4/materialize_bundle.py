"""物化阶段只读 bundle（纯 numpy，无 qlib）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GlobalConfig
from .kronos_cache import KronosLookup
from .volatility_context import VolatilityStats


@dataclass
class FrozenInstrument:
    ohlcva: np.ndarray
    bar_ordinals: np.ndarray


@dataclass
class MaterializeSpawnBundle:
    """物化专用只读上下文。"""

    instruments: dict[str, FrozenInstrument]
    symbol_to_id: dict[str, int]
    cs_medians_day: np.ndarray
    vol_stats: VolatilityStats
    kronos_lookup: KronosLookup
    config: GlobalConfig
    seq_len: int
