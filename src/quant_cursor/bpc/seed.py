from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """固定 Python / NumPy / PyTorch 随机源，便于可复现训练。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 部分 CUDA 算子需此环境变量才严格确定性
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
