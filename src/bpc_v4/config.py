"""bpc_v4 全局配置（采用分层设计，参考更优实现）"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class KronosConfig:
    model_name: str = "NeoQuasar/Kronos-Tokenizer-base"
    revision: str = "main"
    seq_len: int = 40
    d_model: int = 768
    amount_pad_zero: bool = True
    local_path: str = r"E:\quant_cursor\models\NeoQuasar\Kronos-Tokenizer-base"


@dataclass
class BPCConfig:
    feat_dim: int = 26
    struct_dim: int = 21
    behavior_dim: int = 5
    num_agents: int = 5
    num_classes_per_agent: int = 3


@dataclass
class ContextConfig:
    vol_context_dim: int = 3
    amp_proxy_dim: int = 4
    total_ctx_dim: int = 7


@dataclass
class EmbeddingConfig:
    stock_vocab: int = 5000
    stock_emb_dim: int = 16
    time_raw_dim: int = 16
    time_proj_dim: int = 16
    total_emb_dim: int = 32


@dataclass
class FusionConfig:
    fused_dim: int = 768 + 26 + 7 + 32  # 833
    hidden_dim: int = 256
    dropout: float = 0.2


@dataclass
class HeadConfig:
    purity_output_dim: int = 15
    purity_weight: float = 1.0
    codebook_output_dim: int = 64  # 适配 s1_ids
    codebook_weight: float = 0.5


@dataclass
class TrainingConfig:
    batch_size: int = 256
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    amp: bool = True
    log_interval: int = 50
    eval_interval: int = 5
    save_dir: Path = Path("./checkpoints/bpc_v4")
    device: str = "cuda"


@dataclass
class QlibConfig:
    provider_uri: Path = Path("~/.qlib/qlib_data/cn_data")
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    instruments: list = field(default_factory=lambda: ["CSI300"])
    max_samples: Optional[int] = None


@dataclass
class GlobalConfig:
    kronos: KronosConfig = field(default_factory=KronosConfig)
    bpc: BPCConfig = field(default_factory=BPCConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    qlib: QlibConfig = field(default_factory=QlibConfig)