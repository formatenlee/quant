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

在 **quant/** 项目根目录（`src/` 即 `quant_cursor` Python 包）：

### 1. Kronos 预计算（一次性，与 BPC 物化解耦）

```bash
python -m quant_cursor.bpc_v4.precompute_kronos \
  --full --start 2015-01-01 --end 2024-12-31 \
  --cpu-threads 4 \
  --kronos-path /path/to/Kronos-Tokenizer-base \
  --output-dir data/kronos_cache
```

- 按标的分片缓存 `z_q` / `s1_ids` 到 `data/kronos_cache/instruments/`
- **输入**：每个窗口做 per-window z-score + clip(±5)（与 Kronos 官方 `KronosPredictor` 一致）；未归一化会导致 `s1_ids` 坍缩为常数
- 缓存 schema：`kronos_cache_v2`（含 `ohlcva_norm=per_window_zscore_clip5`）
- 增量：已有分片跳过；`--force-rebuild` 覆盖单标的分片
- 物化/训练变更 BPC 特征时**无需重跑 Kronos**

### 2. 训练

```bash
python -m quant_cursor.bpc_v4.train --dev --device cuda \
  --kronos-cache-dir data/kronos_cache
```

- 默认自动探测 `data/kronos_cache/meta.json`
- `--force-rebuild-preprocessed` 仅重建 BPC 路径（快）
- 调试可加 `--allow-live-kronos` 在线编码（慢）

```bash
python -m quant_cursor.bpc_v4.train --dev --device cuda
```

## 可解析性 (v4.0)

重点输出 `z_q` 方向与 BPC 26 维特征的相关系数矩阵，验证 Kronos 方向是否与结构因子对齐。

## 后续演进

- TemporalAggregator 可替换为 Conv1dAggregator / CrossAttentionAggregator
- 可选离线缓存 z_q.npy 加速训练

## Codebook 伪标签策略（方案 A）

- `codebook_head` 输入为 **BPC + ctx + emb**（**不含 z_q**），避免从 Kronos 潜变量直接读出 `s1_ids` 的平凡解。
- 监督目标：Kronos `s1_ids`（vocab=2^s1_bits），验证 BPC 行为特征能否对齐 Kronos 离散粒度。
- `purity_head` 仍使用 **z_q + BPC + ctx + emb** 全路径融合。
