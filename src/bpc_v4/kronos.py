"""Kronos Tokenizer 单例封装（完全冻结，仅加载 tokenizer 权重）"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .kronos_model import KronosTokenizer

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


def read_kronos_z_q_dim(local_path: Optional[str] = None) -> int:
    """从 tokenizer config.json 读取 z_q 维度（s1_bits + s2_bits）。"""
    path = local_path or resolve_kronos_local_path()
    if not path:
        return 20
    cfg_path = Path(path) / "config.json"
    if not cfg_path.exists():
        return 20
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return int(cfg.get("s1_bits", 10) + cfg.get("s2_bits", 10))


def read_kronos_s1_bits(local_path: Optional[str] = None) -> int:
    """从 tokenizer config.json 读取 s1_bits。"""
    path = local_path or resolve_kronos_local_path()
    if not path:
        return 10
    cfg_path = Path(path) / "config.json"
    if not cfg_path.exists():
        return 10
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    return int(cfg.get("s1_bits", 10))


def read_kronos_s1_vocab_size(local_path: Optional[str] = None) -> int:
    return 2 ** read_kronos_s1_bits(local_path)


def sync_kronos_config(config) -> None:
    """将 kronos / head 维度与 tokenizer 配置对齐。"""
    local_path = config.kronos.local_path or None
    z_q_dim = read_kronos_z_q_dim(local_path)
    if config.kronos.d_model != z_q_dim:
        logger.info("Sync kronos.d_model: %d -> %d (z_q codebook dim)", config.kronos.d_model, z_q_dim)
        config.kronos.d_model = z_q_dim

    s1_bits = read_kronos_s1_bits(local_path)
    s1_vocab = 2 ** s1_bits
    if config.kronos.s1_bits != s1_bits:
        logger.info("Sync kronos.s1_bits: %d -> %d", config.kronos.s1_bits, s1_bits)
        config.kronos.s1_bits = s1_bits
    if config.head.codebook_output_dim != s1_vocab:
        logger.info(
            "Sync head.codebook_output_dim: %d -> %d (full Kronos s1 vocabulary)",
            config.head.codebook_output_dim,
            s1_vocab,
        )
        config.head.codebook_output_dim = s1_vocab


class KronosTokenizerPool:
    """全局单例，避免重复加载 tokenizer 权重。"""

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
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._model_name = model_name
        self._local_path = local_path or resolve_kronos_local_path()
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")

        self._load_tokenizer()

    def _load_tokenizer(self):
        """加载预训练 KronosTokenizer 并冻结。"""
        local_files_only = False
        load_path = self._local_path

        if load_path and Path(load_path).is_dir():
            logger.info("正在从本地路径加载 Kronos Tokenizer: %s", load_path)
            local_files_only = True
        else:
            load_path = self._model_name
            logger.info("正在从 HuggingFace 加载 Kronos Tokenizer: %s", load_path)

        self._tokenizer = KronosTokenizer.from_pretrained(
            load_path,
            local_files_only=local_files_only,
        )
        self._tokenizer.eval()
        self._tokenizer.to(self._device)

        for param in self._tokenizer.parameters():
            param.requires_grad = False

        self._z_q_dim = self._tokenizer.codebook_dim
        logger.info(
            "Kronos tokenizer loaded: z_q_dim=%d, d_model=%d, device=%s",
            self._z_q_dim,
            self._tokenizer.d_model,
            self._device,
        )

    @property
    def d_model(self) -> int:
        """BPC 融合使用的 z_q 维度（codebook bits，非 transformer hidden size）。"""
        return self._z_q_dim

    @property
    def device(self) -> torch.device:
        return self._device

    @torch.no_grad()
    def encode(self, ohlcva: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        输入: ohlcva [B, T, 6]
        返回: (z_q [B, T, codebook_dim], s1_ids [B, T], s2_ids [B, T])
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        ohlcva = ohlcva.to(self._device)
        _, _, z_q, _ = self._tokenizer(ohlcva)
        z_indices = self._tokenizer.encode(ohlcva, half=True)
        s1_ids, s2_ids = z_indices[0], z_indices[1]
        return z_q, s1_ids, s2_ids
