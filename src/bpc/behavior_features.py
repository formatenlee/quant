"""

因果行为代理：纯统计学/结构描述（非心理标签）。



输入 x: [B, T, 5] — open, high, low, close, volume。

每个标量对应一种可复现的市场动作结构，可用 33/66% 分位数分档。



5 代理体系（4 核心 + 1 扩展）：

- 核心：vol, attack, path_structure, vol_structure

- 扩展：momentum

已移除 liquidity（rel_vol/vol 与 vol 制度冗余）；成交量异常见 volume_structure。

"""



from __future__ import annotations



import torch

import torch.nn.functional as F



# 4 核心 + 1 扩展 = 5 代理 × 3 档 = 15 logits

BEHAVIOR_AGENT_NAMES: tuple[str, ...] = (

    "vol",  # 已实现波动率

    "attack",  # 量价共振攻势

    "path_structure",  # 路径效率与缺口连贯性

    "vol_structure",  # 跳空/波动不对称结构

    "momentum",  # 正收益日占比 + 最大回撤

)



CORE_AGENTS = ("vol", "attack", "path_structure", "vol_structure")

EXTENDED_AGENTS = ("momentum",)



NUM_BEHAVIOR_AGENTS = len(BEHAVIOR_AGENT_NAMES)

NUM_BEHAVIOR_CLASSES = 3

BEHAVIOR_LOGITS_DIM = NUM_BEHAVIOR_AGENTS * NUM_BEHAVIOR_CLASSES



# 日志键名（与历史字段尽量兼容）

PURITY_LOSS_KEYS: dict[str, str] = {

    "vol": "purity_vol",

    "attack": "purity_attack",

    "path_structure": "purity_path",

    "vol_structure": "purity_vol_struct",

    "momentum": "purity_momentum",

}





def transform_proxies_for_labeling(proxies: torch.Tensor) -> torch.Tensor:

    """纯度分箱：代理已在 compute 阶段完成单调变换，此处不再二次处理。"""

    return proxies





def transform_proxies_for_features(proxies: torch.Tensor) -> torch.Tensor:

    """编码器输入：代理本身已是相对/结构量，原样使用。"""

    return proxies





def _masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:

    m = mask.float()

    return (x * m).sum(dim=dim) / m.sum(dim=dim).clamp_min(1.0)





def compute_behavior_proxies(x: torch.Tensor) -> dict[str, torch.Tensor]:

    stacked = compute_behavior_proxies_stacked(x)

    return {name: stacked[:, i] for i, name in enumerate(BEHAVIOR_AGENT_NAMES)}





def compute_behavior_proxies_stacked(x: torch.Tensor) -> torch.Tensor:

    """返回 [B, 5] float32 行为结构标量。"""

    open_p = x[..., 0]

    high = x[..., 1]

    low = x[..., 2]

    close = x[..., 3]

    volume = x[..., 4].clamp_min(0.0)



    _B, T = close.shape

    eps = 1e-8

    log_ret = torch.log(close[:, 1:] / close[:, :-1].clamp_min(eps))



    # --- 核心 ---

    vol = log_ret.std(dim=1)



    bar_range = (high - low).clamp_min(eps)

    eff = (close - open_p) / bar_range

    rel_vol = volume / volume.mean(dim=1, keepdim=True).clamp_min(eps)

    attack = (eff * rel_vol).mean(dim=1)



    prev_high = high[:, :-1]

    prev_low = low[:, :-1]

    up_gap = (low[:, 1:] > prev_high).float()

    down_gap = (high[:, 1:] < prev_low).float()

    net_up = (close[:, -1] > close[:, 0]).unsqueeze(1)

    aligned_gap = torch.where(net_up, up_gap, down_gap)

    gap_coherence = aligned_gap.sum(dim=1) / max(T - 1, 1)



    abs_ret = (close[:, 1:] / close[:, :-1] - 1.0).abs()

    energy_conc = abs_ret.max(dim=1).values / abs_ret.sum(dim=1).clamp_min(eps)

    path_structure = gap_coherence + energy_conc



    prev_close = close[:, :-1]

    open_next = open_p[:, 1:]

    gap_move = (open_next / prev_close - 1.0).abs()

    intraday = ((high[:, 1:] - low[:, 1:]) / prev_close.clamp_min(eps)).abs()

    jump_vol_ratio = gap_move.sum(dim=1) / (gap_move.sum(dim=1) + intraday.sum(dim=1)).clamp_min(eps)



    up_day = close[:, 1:] > close[:, :-1]

    down_day = ~up_day

    rv_up = _masked_mean(log_ret.abs(), up_day)

    rv_down = _masked_mean(log_ret.abs(), down_day)

    vol_skew = torch.log((rv_up / rv_down.clamp_min(eps)).clamp_min(eps))

    vol_structure = jump_vol_ratio + vol_skew



    # --- 扩展：动量强度 ---

    pos_ratio = (log_ret > 0).float().mean(dim=1)

    running_max = close.cummax(dim=1).values

    max_drawdown = ((running_max - close) / running_max.clamp_min(eps)).max(dim=1).values

    momentum = pos_ratio - max_drawdown



    return torch.stack(

        [vol, attack, path_structure, vol_structure, momentum],

        dim=1,

    ).to(dtype=torch.float32)





def agent_logit_slice(agent_index: int) -> slice:

    start = agent_index * NUM_BEHAVIOR_CLASSES

    return slice(start, start + NUM_BEHAVIOR_CLASSES)


