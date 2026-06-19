"""
Token 转移分析工具：支持模式转移矩阵、高胜率路径挖掘、转移熵计算。

符合“人为规律可重复”哲学：将离散 token 视为行为原语，统计其转移规律。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class TokenTransitionAnalyzer:
    """
    分析 token 序列的转移模式。

    用法示例：
        analyzer = TokenTransitionAnalyzer(num_tokens=128)
        analyzer.update(coarse_tokens)          # [B, T] 或 list of sequences
        matrix = analyzer.transition_matrix()   # [K, K]
        paths = analyzer.top_paths(start=42, k=5)
    """

    def __init__(self, num_tokens: int, order: int = 1):
        self.num_tokens = num_tokens
        self.order = order  # 当前仅支持 1 阶（Markov）
        self.counts = torch.zeros(num_tokens, num_tokens, dtype=torch.long)
        self.start_counts = torch.zeros(num_tokens, dtype=torch.long)

    def update(self, token_seq: torch.Tensor | List[List[int]]) -> None:
        """接受 [B, T] tensor 或 list of token lists。"""
        if isinstance(token_seq, torch.Tensor):
            seqs = token_seq.cpu().tolist()
        else:
            seqs = token_seq
        for seq in seqs:
            if len(seq) < 2:
                continue
            self.start_counts[seq[0]] += 1
            for t in range(len(seq) - 1):
                prev, curr = seq[t], seq[t + 1]
                if 0 <= prev < self.num_tokens and 0 <= curr < self.num_tokens:
                    self.counts[prev, curr] += 1

    def transition_matrix(self, normalize: bool = True, eps: float = 1e-8) -> torch.Tensor:
        """返回转移概率矩阵 P(next | prev)。"""
        mat = self.counts.float()
        if normalize:
            row_sum = mat.sum(dim=1, keepdim=True).clamp_min(eps)
            mat = mat / row_sum
        return mat

    def transition_entropy(self) -> torch.Tensor:
        """每行的熵值，衡量从该 token 出发的后续状态不确定性。"""
        p = self.transition_matrix(normalize=True)
        p = p.clamp_min(1e-12)
        entropy = -(p * p.log()).sum(dim=1)
        return entropy

    def top_paths(
        self, start: int, k: int = 5, max_len: int = 4
    ) -> List[Tuple[List[int], float]]:
        """
        从 start token 出发，找概率最高的 k 条路径（贪心搜索）。
        返回 [(path, prob), ...]
        """
        mat = self.transition_matrix(normalize=True)
        results: List[Tuple[List[int], float]] = []

        def dfs(path: List[int], prob: float):
            if len(path) > max_len:
                return
            if len(path) >= 2:
                results.append((path[:], prob))
            if len(results) >= k * 3:  # 宽松截断
                return
            next_probs = mat[path[-1]]
            topk = torch.topk(next_probs, min(3, self.num_tokens))
            for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
                if p < 1e-6:
                    continue
                dfs(path + [idx], prob * p)

        dfs([start], 1.0)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def high_probability_transitions(self, min_prob: float = 0.15) -> List[Tuple[int, int, float]]:
        """返回所有 P(j|i) >= min_prob 的转移三元组 (i, j, prob)。"""
        mat = self.transition_matrix(normalize=True)
        i_idx, j_idx = torch.where(mat >= min_prob)
        return [(int(i), int(j), float(mat[i, j])) for i, j in zip(i_idx.tolist(), j_idx.tolist())]

    def reset(self) -> None:
        self.counts.zero_()
        self.start_counts.zero_()
