"""
BPC-v3 行为代理（符号化方向/结构，与 bpc v2 完全隔离）。

数据流（归一化分层，避免重复）：
  L0 绝对 OHLCV (store) → 字段 Δ − 截面中值 → 模型唯一 OHLCV 输入
  L1 窗口内无量纲比（收益率、vol 环比-1、efficiency、rel_vol）→ 结构特征
  L2 vol_context：绝对 close RV 锚点；dim1 截面 z-score（仅 RV，非 close_Δ）；dim2=rv/baseline-1
  L3 纯度标签：固定 SYMBOLIC_THRESHOLDS，无 batch/global z-score
  L4 结构维：固定分组尺度 + vol_context 锚点动态缩放（FEATURE_SCALE_SCHEMA）
  L5 V3PrecomputedFeatureComposer：Linear 投影（无 LayerNorm）

禁用：CausalNormalizer、相对量上叠 log、LayerNorm（特征/VQ/纯度路径）、训练集 z-score。

5 代理：regime / attack / path_structure / vol_structure / momentum
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .ohlcv_relative import (
    levels_from_field_deltas,
    simple_returns_from_levels,
    volume_ratio_from_levels,
)

BEHAVIOR_AGENT_NAMES: tuple[str, ...] = (
    "regime",
    "attack",
    "path_structure",
    "vol_structure",
    "momentum",
)

CORE_AGENTS = ("regime", "attack", "path_structure", "vol_structure")
EXTENDED_AGENTS = ("momentum",)

NUM_BEHAVIOR_AGENTS = len(BEHAVIOR_AGENT_NAMES)
NUM_BEHAVIOR_CLASSES = 3
BEHAVIOR_LOGITS_DIM = NUM_BEHAVIOR_AGENTS * NUM_BEHAVIOR_CLASSES

PURITY_LOSS_KEYS: dict[str, str] = {
    "regime": "purity_regime",
    "attack": "purity_attack",
    "path_structure": "purity_path",
    "vol_structure": "purity_vol_struct",
    "momentum": "purity_momentum",
}

# 纯度标签 schema；v14 修正 vol_bucket（环比）与收益率代理
# 软标签温度：>0 时 symbolic_labels 输出平滑三档概率，缓解主导类记忆
DEFAULT_SYMBOLIC_LABEL_TEMPERATURE = 0.12
BEHAVIOR_LABEL_SCHEMA = "symbolic_agents_v14_soft_t012"

STRUCTURAL_PROXY_INDICES: tuple[int, ...] = ()
STRUCTURAL_ZSCORE_AGENTS: tuple[str, ...] = ()
SYMBOLIC_AUDIT_AGENTS: tuple[str, ...] = (
    "attack",
    "path_structure",
    "vol_structure",
    "momentum",
)

# (low_hi, mid_hi)：symbolic_labels 三档分界；v14 qlib preflight 复核（2022–2026）
SYMBOLIC_THRESHOLDS: dict[str, tuple[float, float]] = {
    "regime": (-0.05, 0.05),
    "attack": (-0.10, 0.15),         # qlib p33/p67≈(-0.10,0.15)
    "path_structure": (0.41, 0.84),  # qlib p33/p67≈(0.41,0.84)
    "vol_structure": (-0.10, 0.10),  # 分布近零中心，p33≈p67≈0 故保留对称窗
    "momentum": (-0.35, 0.35),       # 块级惯性天然双峰，mid 档稀少属预期
}

REGIME_MA_WEIGHTS: dict[int, float] = {5: 1.0, 10: 2.0, 20: 4.0}

# |实体|/振幅 分档（无量纲，典型范围 [0, 1]）
KLINE_MAG_SMALL = 0.25
KLINE_MAG_LARGE = 0.65
KLINE_SHADOW_DOMINANT = 0.6
KLINE_VOL_SHRINK = 0.7
KLINE_VOL_EXPAND = 1.4
PATH_BLOCK_LEN = 5


def _kline_tokenize_daily(
    open_p: torch.Tensor,
    high: torch.Tensor,
    low: torch.Tensor,
    close: torch.Tensor,
    volume: torch.Tensor,
) -> torch.Tensor:
    """
    每日 K-line 离散符号 [B, T, 4]：
    - dim0 方向 {-1, 0, +1}（实体）
    - dim1 幅度档 {0, 1, 2}（|实体|/振幅，无量纲）
    - dim2 实体位置 {0, 1, 2}（下影线长 / 均衡 / 上影线长）
    - dim3 成交量档 {0, 1, 2}（缩量 / 均量 / 放量），基于绝对成交量环比 V_t/V_{t-1}
    """
    body = close - open_p
    direction = torch.where(body > 0, 1.0, torch.where(body < 0, -1.0, 0.0))

    range_ = (high - low).clamp_min(1e-8)
    # 日内实体占振幅比例（无量纲），避免伪价格层级与 close_Δ RV 混用
    body_z = body / range_
    abs_z = body_z.abs()
    magnitude = torch.where(
        abs_z < KLINE_MAG_SMALL,
        torch.zeros_like(abs_z),
        torch.where(abs_z < KLINE_MAG_LARGE, torch.ones_like(abs_z), torch.full_like(abs_z, 2.0)),
    )

    upper_shadow = (high - torch.max(open_p, close)) / range_
    lower_shadow = (torch.min(open_p, close) - low) / range_
    body_pos = torch.where(
        (upper_shadow > KLINE_SHADOW_DOMINANT) & (lower_shadow < 0.2),
        torch.full_like(body, 2.0),
        torch.where(
            (lower_shadow > KLINE_SHADOW_DOMINANT) & (upper_shadow < 0.2),
            torch.zeros_like(body),
            torch.ones_like(body),
        ),
    )

    # volume 须为绝对成交量层级（非 volume_Δ）
    vol_ratio = volume_ratio_from_levels(volume.clamp_min(1e-8))
    cum_ratio = vol_ratio.cumsum(dim=1)
    count = torch.arange(1, volume.shape[1] + 1, device=volume.device, dtype=volume.dtype).view(
        1, -1
    )
    vol_ma_ratio = cum_ratio / count
    vol_rel = vol_ratio / vol_ma_ratio.clamp_min(1e-8)
    vol_bucket = torch.where(
        vol_rel < KLINE_VOL_SHRINK,
        torch.zeros_like(vol_rel),
        torch.where(vol_rel < KLINE_VOL_EXPAND, torch.ones_like(vol_rel), torch.full_like(vol_rel, 2.0)),
    )

    return torch.stack([direction, magnitude, body_pos, vol_bucket], dim=-1)


def _block_direction_score(
    recent_dir: torch.Tensor,
    prev_dir: torch.Tensor,
) -> torch.Tensor:
    """无足够转移统计时的离散回退：{-1, 0, +1}。"""
    out = torch.zeros_like(recent_dir)
    continuation = (recent_dir == prev_dir) & (prev_dir != 0)
    reversal = (recent_dir != prev_dir) & (prev_dir != 0) & (recent_dir != 0)
    out = torch.where(continuation, torch.ones_like(out), out)
    out = torch.where(reversal, -torch.ones_like(out), out)
    return out


def _state_transition_score(
    symbols: torch.Tensor,
    *,
    block_len: int = PATH_BLOCK_LEN,
) -> torch.Tensor:
    """
    基于窗口内因果前缀的状态转移评分（Symbolic Dynamics）。

    每个样本仅用 recent block 之前的日方向序列估计 3×3 转移矩阵，
    再对 prev→recent 主导方向转移给出确定性分数 ∈ [-1, 1]。
    """
    _B, T, _D = symbols.shape
    device = symbols.device
    dtype = torch.float32
    out = torch.zeros(_B, device=device, dtype=dtype)

    if T < 2 * block_len + 2:
        return out

    recent_dir = torch.sign(symbols[:, -block_len:, 0].mean(dim=1)).clamp(-1.0, 1.0)
    prev_dir = torch.sign(symbols[:, -2 * block_len : -block_len, 0].mean(dim=1)).clamp(-1.0, 1.0)
    prev_idx = (prev_dir + 1).long().clamp(0, 2)
    recent_idx = (recent_dir + 1).long().clamp(0, 2)

    prefix_end = T - block_len
    if prefix_end < 2:
        return _block_direction_score(recent_dir, prev_dir)

    daily_idx = (symbols[:, :, 0] + 1).long().clamp(0, 2)
    left = daily_idx[:, : prefix_end - 1]
    right = daily_idx[:, 1:prefix_end]
    flat = left * 3 + right
    ones = torch.ones_like(flat, dtype=dtype)
    trans_count = torch.zeros(_B, 9, device=device, dtype=dtype)
    trans_count.scatter_add_(1, flat, ones)
    trans_count = trans_count.view(_B, 3, 3)

    row_sum = trans_count.sum(dim=2, keepdim=True)
    sparse = trans_count.sum(dim=(1, 2)) < 3.0
    smooth_alpha = max(0.1, 1.0 / float(max(prefix_end - 1, 1)))
    trans_prob = (trans_count + smooth_alpha) / (
        row_sum + 3.0 * smooth_alpha
    ).clamp_min(1e-6)

    batch_idx = torch.arange(_B, device=device)
    p_rows = trans_prob[batch_idx, prev_idx]
    p_same = p_rows[batch_idx, recent_idx]
    opposite_idx = (2 - prev_idx).clamp(0, 2)
    p_reverse = p_rows[batch_idx, opposite_idx]

    uniform = 1.0 / 3.0
    cont_score = ((p_same - uniform) / (1.0 - uniform)).clamp(0.0, 1.0)
    rev_score = -((p_reverse - uniform) / (1.0 - uniform)).clamp(0.0, 1.0)

    same_dir = (recent_dir == prev_dir) & (prev_dir != 0)
    reverse_dir = (recent_dir != prev_dir) & (prev_dir != 0) & (recent_dir != 0)
    fallback = _block_direction_score(recent_dir, prev_dir)

    out = torch.where(same_dir, cont_score, out)
    out = torch.where(reverse_dir, rev_score, out)
    out = torch.where(sparse, fallback, out)
    return out.clamp(-1.0, 1.0)


def symbolic_path_structure_score(
    open_p: torch.Tensor,
    high: torch.Tensor,
    low: torch.Tensor,
    close: torch.Tensor,
    volume: torch.Tensor,
    log_ret: torch.Tensor,
    *,
    vol_context: torch.Tensor | None = None,
    block_len: int = PATH_BLOCK_LEN,
) -> torch.Tensor:
    """K-line 符号化 + 因果状态转移 → path_structure ∈ [-1, 1]。"""
    symbols = _kline_tokenize_daily(open_p, high, low, close, volume)
    return _state_transition_score(symbols, block_len=block_len)


def symbolic_attack_score(
    symbols: torch.Tensor,
    *,
    window: int = PATH_BLOCK_LEN * 2,
) -> torch.Tensor:
    """
    量价攻击：实体方向与成交量档一致率 ∈ [-1, 1]。
    放量阳线 / 缩量阴线 → 正；反向 → 负。
    """
    if symbols.shape[1] > window:
        symbols = symbols[:, -window:]
    direction = symbols[..., 0]
    vol_bucket = symbols[..., 3]
    aligned = ((direction > 0) & (vol_bucket >= 2)) | ((direction < 0) & (vol_bucket == 0))
    misaligned = ((direction > 0) & (vol_bucket == 0)) | ((direction < 0) & (vol_bucket >= 2))
    mild_pos = ((direction > 0) & (vol_bucket == 1)).float() * 0.25
    mild_neg = ((direction < 0) & (vol_bucket == 1)).float() * -0.25
    return (
        aligned.float().mean(dim=1)
        - misaligned.float().mean(dim=1)
        + mild_pos.mean(dim=1)
        + mild_neg.mean(dim=1)
    ).clamp(-1.0, 1.0)


def symbolic_vol_structure_score(
    high: torch.Tensor,
    low: torch.Tensor,
    symbols: torch.Tensor,
    day_ret: torch.Tensor,
    *,
    window: int = PATH_BLOCK_LEN * 2,
) -> torch.Tensor:
    """
    波动结构：HL 拓扑跳空方向偏置 + 涨跌日影线位置偏度 ∈ [-1, 1]。

    使用 low > prev_high / high < prev_low 的纯方向事件，避免 v8 相对开盘跳空
    （gap_ret vs recent_rv）引入按标的波动率校准的可记忆模式。
    """
    _B, T = high.shape
    if T > window + 1:
        high = high[:, -window:]
        low = low[:, -window:]
        symbols = symbols[:, -window:]
        day_ret = day_ret[:, -window:]

    n_gap = max(high.shape[1] - 1, 1)
    up_gap = (low[:, 1:] > high[:, :-1]).float()
    down_gap = (high[:, 1:] < low[:, :-1]).float()
    gap_bias = (up_gap.sum(dim=1) - down_gap.sum(dim=1)) / n_gap

    body_pos = symbols[:, 1:, 2]
    ret_aligned = day_ret[:, 1:] if day_ret.shape[1] == high.shape[1] else day_ret
    up_day = ret_aligned > 0
    down_day = ret_aligned < 0
    up_count = up_day.float().sum(dim=1).clamp_min(1.0)
    down_count = down_day.float().sum(dim=1).clamp_min(1.0)
    up_upper = ((body_pos > 1.5) & up_day).float().sum(dim=1) / up_count
    up_lower = ((body_pos < 0.5) & up_day).float().sum(dim=1) / up_count
    down_upper = ((body_pos > 1.5) & down_day).float().sum(dim=1) / down_count
    down_lower = ((body_pos < 0.5) & down_day).float().sum(dim=1) / down_count
    rally_weak = up_upper - up_lower
    decline_weak = down_upper - down_lower
    shadow_skew = ((down_lower - down_upper) - rally_weak).clamp(-1.0, 1.0)

    gap_weight = (up_gap.sum(dim=1) + down_gap.sum(dim=1)) / n_gap
    gap_weight = gap_weight.clamp(0.0, 1.0)
    gap_s = gap_bias.clamp(-1.0, 1.0)
    mix = 0.35 * gap_weight + 0.65
    return (gap_s * (1.0 - mix) + shadow_skew * mix).clamp(-1.0, 1.0)


def _resolve_block_direction(sign_block: torch.Tensor) -> torch.Tensor:
    """块主导方向；均值平局时回退到末日系。"""
    block_dir = torch.sign(sign_block.mean(dim=1))
    block_dir = torch.where(block_dir == 0, sign_block[:, -1], block_dir)
    return block_dir.clamp(-1.0, 1.0)


def symbolic_momentum_score(
    log_ret: torch.Tensor,
    *,
    block_len: int = PATH_BLOCK_LEN,
) -> torch.Tensor:
    """
    动量：块级方向惯性 + 延续/反转 ∈ [-1, 1]。
    替代全局 pos_ratio 以提升标签分散度。
    """
    sign = torch.sign(log_ret)
    T = sign.shape[1]
    out = torch.zeros(sign.shape[0], device=sign.device, dtype=torch.float32)
    if T < block_len:
        return out

    recent = sign[:, -block_len:]
    recent_dir = _resolve_block_direction(recent)
    inertia = (recent == recent_dir.unsqueeze(1)).float().mean(dim=1)

    if T < 2 * block_len:
        return (recent_dir * inertia).clamp(-1.0, 1.0)

    prev = sign[:, -2 * block_len : -block_len]
    prev_dir = _resolve_block_direction(prev)
    continuation = (recent_dir == prev_dir) & (recent_dir != 0)
    reversal = (recent_dir != prev_dir) & (prev_dir != 0) & (recent_dir != 0)

    score = recent_dir * inertia * 0.5
    boost = 0.5 + 0.5 * inertia
    score = torch.where(continuation, recent_dir * boost, score)
    score = torch.where(reversal, -prev_dir * boost, score)
    return score.clamp(-1.0, 1.0)


def transform_proxies_for_labeling(proxies: torch.Tensor) -> torch.Tensor:
    """符号化代理：标签直接用原始值 + 固定阈值，不做 z-score。"""
    return proxies


def normalize_proxies_for_labeling(
    proxies: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Legacy API：v6+ 全符号代理，忽略 global mean/std。"""
    del mean, std
    return proxies


def transform_proxies_for_features(proxies: torch.Tensor) -> torch.Tensor:
    return proxies



def _regime_proxy(
    close: torch.Tensor,
    ma_periods: tuple[int, ...] = (5, 10, 20),
    tolerance: float = 0.02,
) -> torch.Tensor:
    """多周期均线位置聚合：[-1, 1] 连续值，后续由纯度头分 3 档。"""
    positions: list[torch.Tensor] = []
    weights: list[float] = []
    for period in ma_periods:
        if close.shape[1] < period:
            continue
        ma = close[:, -period:].mean(dim=1)
        price = close[:, -1]
        upper = ma * (1.0 + tolerance)
        lower = ma * (1.0 - tolerance)
        pos = torch.where(price > upper, 1.0, torch.where(price < lower, -1.0, 0.0))
        positions.append(pos)
        weights.append(REGIME_MA_WEIGHTS.get(period, 1.0))
    if not positions:
        return torch.zeros(close.shape[0], device=close.device, dtype=close.dtype)
    stack = torch.stack(positions, dim=1)
    w = torch.tensor(weights, device=close.device, dtype=close.dtype)
    return (stack * w.unsqueeze(0)).sum(dim=1) / w.sum().clamp_min(1e-8)


def compute_behavior_proxies(x: torch.Tensor) -> dict[str, torch.Tensor]:
    stacked = compute_behavior_proxies_stacked(x)
    return {name: stacked[:, i] for i, name in enumerate(BEHAVIOR_AGENT_NAMES)}


def compute_behavior_proxies_stacked(
    x: torch.Tensor,
    vol_context: torch.Tensor | None = None,
    prev_bar: torch.Tensor | None = None,
) -> torch.Tensor:
    """返回 [B, 5] float32 行为结构标量。x 为相对化 OHLCV（字段 Δ）。"""
    if prev_bar is not None:
        levels = levels_from_field_deltas(x, prev_bar)
        open_p = levels[..., 0]
        high = levels[..., 1]
        low = levels[..., 2]
        close = levels[..., 3]
        vol_levels = levels[..., 4]
    else:
        open_p = x[..., 0]
        high = x[..., 1]
        low = x[..., 2]
        close = x[..., 3]
        vol_levels = x[..., 4].abs().clamp_min(1e-8)

    if close.shape[1] >= 2:
        day_ret = simple_returns_from_levels(close)
    else:
        day_ret = torch.zeros(close.shape[0], 1, device=close.device, dtype=close.dtype)

    symbols = _kline_tokenize_daily(open_p, high, low, close, vol_levels)

    regime = _regime_proxy(close)
    attack = symbolic_attack_score(symbols)
    path_structure = _state_transition_score(symbols, block_len=PATH_BLOCK_LEN)
    vol_structure = symbolic_vol_structure_score(high, low, symbols, day_ret)
    momentum = symbolic_momentum_score(day_ret, block_len=PATH_BLOCK_LEN)

    return torch.stack(
        [regime, attack, path_structure, vol_structure, momentum],
        dim=1,
    ).to(dtype=torch.float32)


def compute_purity_targets_from_proxies(
    proxy_mat: torch.Tensor,
    *,
    temperature: float = DEFAULT_SYMBOLIC_LABEL_TEMPERATURE,
) -> torch.Tensor:
    """5 代理 × 3 档软标签 → [B, 15]，与 v3 symbolic_labels 一致。"""
    labels: list[torch.Tensor] = []
    for j, name in enumerate(BEHAVIOR_AGENT_NAMES):
        lo, hi = SYMBOLIC_THRESHOLDS[name]
        labels.append(symbolic_labels(proxy_mat[:, j], lo, hi, temperature=temperature))
    return torch.cat(labels, dim=-1)


def symbolic_labels(
    values: torch.Tensor,
    low: float,
    high: float,
    *,
    temperature: float = DEFAULT_SYMBOLIC_LABEL_TEMPERATURE,
) -> torch.Tensor:
    """固定阈值 3 档标签；temperature>0 时为平滑概率，否则 one-hot。"""
    if temperature <= 0.0:
        bins = (values > high).long() + (values > low).long()
        return F.one_hot(bins.clamp(max=2), 3).float()
    temp = max(float(temperature), 1e-4)
    p_above_low = torch.sigmoid((values - low) / temp)
    p_above_high = torch.sigmoid((values - high) / temp)
    p_low = 1.0 - p_above_low
    p_mid = (p_above_low - p_above_high).clamp_min(0.0)
    p_high = p_above_high
    probs = torch.stack([p_low, p_mid, p_high], dim=-1).clamp_min(0.0)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def audit_symbolic_label_distribution(
    proxy_mat: torch.Tensor,
    *,
    agent_index: int = 0,
) -> dict[str, float]:
    """统计符号化代理三档占比与标签熵（启动训练前 sanity check）。"""
    import math

    name = BEHAVIOR_AGENT_NAMES[agent_index]
    lo, hi = SYMBOLIC_THRESHOLDS[name]
    labels = symbolic_labels(proxy_mat[:, agent_index], lo, hi)
    low_pct = float((labels[:, 0] > 0.5).float().mean().item())
    mid_pct = float((labels[:, 1] > 0.5).float().mean().item())
    high_pct = float((labels[:, 2] > 0.5).float().mean().item())
    probs = labels.mean(dim=0).clamp_min(1e-8)
    entropy = float((-(probs * probs.log()).sum()).item())
    max_entropy = math.log(3)
    return {
        "agent": name,
        "low_pct": low_pct,
        "mid_pct": mid_pct,
        "high_pct": high_pct,
        "label_entropy": entropy,
        "max_entropy": max_entropy,
    }


def agent_logit_slice(agent_index: int) -> slice:
    start = agent_index * NUM_BEHAVIOR_CLASSES
    return slice(start, start + NUM_BEHAVIOR_CLASSES)
