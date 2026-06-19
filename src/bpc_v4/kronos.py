"""Kronos Tokenizer 单例封装（完全冻结，参考更优实现）"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_KRONOS_SUBPATH = Path("NeoQuasar/Kronos-Tokenizer-base")


def resolve_kronos_local_path(explicit: Optional[str] = None) -> Optional[str]:
    """解析 Kronos 本地目录，避免在无网环境误走 HuggingFace。"""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    for env_key in ("KRONOS_PATH", "BPC_V4_KRONOS_PATH"):
        env_val = os.environ.get(env_key)
        if env_val:
            candidates.append(Path(env_val).expanduser())

    candidates.extend([
        Path("/home/user/pdl/models") / DEFAULT_KRONOS_SUBPATH,
        Path.home() / "pdl/models" / DEFAULT_KRONOS_SUBPATH,
        Path.home() / "models" / DEFAULT_KRONOS_SUBPATH,
    ])

    try:
        from quant_cursor.config import load_config

        root = load_config().data_dir.parent  # quant/
        candidates.append(root.parent / "pdl/models" / DEFAULT_KRONOS_SUBPATH)
        candidates.append(root / "models" / DEFAULT_KRONOS_SUBPATH)
    except Exception:
        pass

    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_dir() and (path / "config.json").exists():
            resolved = str(path.resolve())
            logger.info("Resolved Kronos local path: %s", resolved)
            return resolved
    return None


class KronosTokenizerPool:
    """全局单例，避免重复加载 102M 参数模型"""

    _instance: Optional["KronosTokenizerPool"] = None
    _tokenizer: Optional[nn.Module] = None
    _device: Optional[torch.device] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str = "NeoQuasar/Kronos-Tokenizer-base",
        local_path: Optional[str] = None,
        device: str = "cuda",
    ):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._model_name = model_name
        self._local_path = local_path
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")

        self._load_tokenizer()

    def _load_tokenizer(self):
        """加载预训练 Kronos Tokenizer 并冻结"""
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError("请安装 transformers: pip install transformers")

        if self._local_path and Path(self._local_path).exists():
            logger.info("正在从本地路径加载 Kronos Tokenizer: %s", self._local_path)
            self._tokenizer = AutoModel.from_pretrained(self._local_path, local_files_only=True)
        else:
            resolved = resolve_kronos_local_path(self._local_path)
            if resolved:
                self._local_path = resolved
                logger.info("正在从本地路径加载 Kronos Tokenizer: %s", resolved)
                self._tokenizer = AutoModel.from_pretrained(resolved, local_files_only=True)
            else:
                logger.info("正在从 HuggingFace 加载 Kronos Tokenizer: %s", self._model_name)
                self._tokenizer = AutoModel.from_pretrained(self._model_name)

        self._tokenizer.eval()
        self._tokenizer.to(self._device)

        for param in self._tokenizer.parameters():
            param.requires_grad = False

        self._d_model = getattr(self._tokenizer.config, "hidden_size", 768)
        logger.info(f"Kronos loaded: d_model={self._d_model}, device={self._device}")

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def device(self) -> torch.device:
        return self._device

    @torch.no_grad()
    def encode(self, ohlcva: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        输入: ohlcva [B, T, 6]
        返回: (z_q [B, T, d_model], s1_ids [B, T], s2_ids [B, T])
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        ohlcva = ohlcva.to(self._device)
        (_, _), _, z_q, _ = self._tokenizer(ohlcva)
        s1_ids, s2_ids = self._tokenizer.encode(ohlcva, half=True)
        return z_q, s1_ids, s2_ids