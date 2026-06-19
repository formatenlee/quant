# bpc_v4

Kronos BSQ Tokenizer + BPC-v3 行为特征的融合版本。

## 核心设计

- **Kronos 路径**（冻结）：OHLCVA → z_q [B, T, 768] → Mean Pool → [B, 768]
- **BPC 路径**（纯函数）：相对化 OHLCV → compute_day_features_vectorized → [B, 26]
- **Context**：vol_context 3维（历史分位/截面Z/全局偏移）+ 幅度代理 4维（ATR/Close、隔夜跳空、量比、振幅率）→ [B, 7]
- **Embedding**：Stock + 时间周期 → [B, 32]
- **融合**：分组 LayerNorm + concat(833) → Linear(256) → 双 Head
- **输出**：
  - 行为纯度 Logits [B, 15] + KL Loss (weight=1.0)
  - 离散码本 Logits [B, 64] + CE Loss (weight=0.5)  ← 适配 Kronos s1_ids（vocab=64）

## Amount 缺失处理

训练时会自动统计并打印：
```
[Data] Amount field missing in X/Y instruments (Z%). Padded with zeros.
```
Kronos 对全 0 amount 的注意力自然衰减，无需额外 mask。

## 运行

```bash
cd quant_cursor
python -m bpc_v4.train
```

## 可解析性 (v4.0)

重点输出 `z_q` 方向与 BPC 26 维特征的相关系数矩阵，验证 Kronos 方向是否与结构因子对齐。

## 后续演进

- TemporalAggregator 可替换为 Conv1dAggregator / CrossAttentionAggregator
- 可选离线缓存 z_q.npy 加速训练

## Codebook 伪标签策略（方案 A）

- **短期（v4.0）**：`codebook_head` 输出维度 = **64**，直接使用 Kronos `s1_ids`（vocab=64）作为静态伪标签。
  - `CrossEntropyLoss` 要求 `num_classes` 与标签最大值+1 严格相等，此方案完全匹配。
  - 训练稳定，可验证融合特征对 Kronos 原始粒度的保留能力。
- **长期（方案 B）**：若需对齐 BPC-v3 的 256 码本，只需将 `codebook_head` 改为 `Linear(256→256)`，并切换伪标签源为 K-Means（K=256）聚类中心即可，改动极小。
