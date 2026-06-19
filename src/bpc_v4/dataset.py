"""bpc_v4 Qlib 数据集（参考更优实现，完整预计算 + 缓存 + 划分）"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import qlib
    from qlib.data import D
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False

from .config import GlobalConfig
from .features import compute_bpc_features, compute_context_features, compute_time_embedding
from .kronos import KronosTokenizerPool

logger = logging.getLogger(__name__)


def resolve_qlib_instruments(
    market: str = "csi300",
    n_instruments: Optional[int] = None,
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    provider_uri: Path = Path("~/.qlib/qlib_data/cn_data"),
) -> List[str]:
    """从 qlib 市场池取标的列表，可按数量截断（用于小样本测试）。"""
    uri = str(provider_uri.expanduser())
    qlib.init(provider_uri=uri, region="cn")
    all_inst = D.list_instruments(
        instruments=D.instruments(market),
        start_time=start,
        end_time=end,
        as_list=True,
    )
    codes = sorted(all_inst)
    if n_instruments is not None:
        codes = codes[:n_instruments]
    logger.info(f"Resolved {len(codes)} instruments from qlib market={market}")
    return codes


def _resolve_feature_codes(instruments: List[str]):
    """支持市场名（如 csi300）或具体 ticker 列表。"""
    if len(instruments) == 1 and "." not in instruments[0]:
        return D.instruments(instruments[0])
    return instruments


class QlibBPCV4Dataset(Dataset):
    """Qlib 数据加载 + 特征预计算 + Kronos z_q 缓存"""

    def __init__(
        self,
        config: GlobalConfig,
        mode: str = "train",
        cache_dir: Optional[Path] = None,
        precompute_zq: bool = True,
    ):
        self.cfg = config
        self.mode = mode
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.precompute_zq = precompute_zq

        if not QLIB_AVAILABLE:
            raise RuntimeError("qlib 未安装")

        qlib.init(provider_uri=str(config.qlib.provider_uri.expanduser()), region="cn")

        self.instruments = config.qlib.instruments
        self.start = config.qlib.start_date
        self.end = config.qlib.end_date

        self._load_data()
        self._kronos = KronosTokenizerPool(
            model_name=config.kronos.model_name,
            local_path=config.kronos.local_path,
            device=config.train.device,
        )
        self._precompute_all()

    def _load_data(self):
        codes = _resolve_feature_codes(self.instruments)
        logger.info(f"Loading instruments from {self.start} to {self.end}")

        fields = ["$open", "$high", "$low", "$close", "$volume", "$amount"]
        df = D.features(codes, fields, start_time=self.start, end_time=self.end, freq="day")

        self._raw_data = df
        self.dates = df.index.get_level_values(0).unique().tolist()
        self.symbols = df.index.get_level_values(1).unique().tolist()
        self.date_to_idx = {d: i for i, d in enumerate(self.dates)}

        self._series: Dict[str, np.ndarray] = {}
        for sym in self.symbols:
            sym_df = df.xs(sym, level=1)
            arr = sym_df[fields].values
            self._series[sym] = arr.astype(np.float32)

        logger.info(f"Loaded {len(self.symbols)} symbols, {len(self.dates)} dates")

    def _precompute_all(self):
        self._samples: List[Tuple[str, int]] = []
        for sym, series in self._series.items():
            T = series.shape[0]
            for t in range(self.cfg.kronos.seq_len, T):
                self._samples.append((sym, t))

        split_idx = int(len(self._samples) * (1 - self.cfg.qlib.val_ratio - self.cfg.qlib.test_ratio))
        val_idx = int(len(self._samples) * (1 - self.cfg.qlib.test_ratio))

        if self.mode == "train":
            self._samples = self._samples[:split_idx]
        elif self.mode == "val":
            self._samples = self._samples[split_idx:val_idx]
        else:
            self._samples = self._samples[val_idx:]

        max_samples = getattr(self.cfg.qlib, "max_samples", None)
        if max_samples is not None and max_samples > 0:
            if self.mode == "train":
                cap = max(1, int(max_samples * 0.8))
            elif self.mode == "val":
                cap = max(1, max_samples - int(max_samples * 0.8))
            else:
                cap = max(1, max_samples // 10)
            self._samples = self._samples[:cap]

        logger.info(f"{self.mode} samples: {len(self._samples)}")

        self._z_q, self._bpc_feat, self._ctx_feat, self._time_emb, self._stock_ids, self._s1_ids = [], [], [], [], [], []

        for i, (sym, t_idx) in enumerate(self._samples):
            series = self._series[sym]
            start = t_idx - self.cfg.kronos.seq_len
            window = series[start:t_idx]
            prev_bar = series[t_idx - self.cfg.kronos.seq_len - 1]

            ohlcv = torch.from_numpy(window[:, :5]).float().unsqueeze(0)
            amount = torch.from_numpy(window[:, 5:6]).float().unsqueeze(0)
            ohlcva = torch.cat([ohlcv, amount], dim=-1)

            if self.precompute_zq:
                z_q, s1_ids, _ = self._kronos.encode(ohlcva)
                self._z_q.append(z_q.squeeze(0).cpu())
                self._s1_ids.append(s1_ids.squeeze(0).cpu())

            vol_ctx = torch.zeros(1, 3)
            prev_bar_t = torch.from_numpy(prev_bar[:5]).float().unsqueeze(0)

            bpc_feat = compute_bpc_features(ohlcv, vol_ctx, prev_bar_t)
            self._bpc_feat.append(bpc_feat.squeeze(0).cpu())

            ctx_feat = compute_context_features(ohlcv, vol_ctx, prev_bar_t)
            self._ctx_feat.append(ctx_feat.squeeze(0).cpu())

            time_raw = compute_time_embedding(torch.tensor([t_idx], dtype=torch.long), raw_dim=self.cfg.embedding.time_raw_dim)
            self._time_emb.append(time_raw.squeeze(0).cpu())

            sym_id = self.symbols.index(sym) if sym in self.symbols else 0
            self._stock_ids.append(sym_id)

            if (i + 1) % 10000 == 0:
                logger.info(f"Precomputed {i+1}/{len(self._samples)}")

        self._z_q = torch.stack(self._z_q) if self._z_q else torch.zeros(0, self.cfg.kronos.seq_len, 768)
        self._bpc_feat = torch.stack(self._bpc_feat)
        self._ctx_feat = torch.stack(self._ctx_feat)
        self._time_emb = torch.stack(self._time_emb)
        self._stock_ids = torch.tensor(self._stock_ids, dtype=torch.long)
        self._s1_ids = torch.stack(self._s1_ids) if self._s1_ids else torch.zeros(0, self.cfg.kronos.seq_len, dtype=torch.long)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "z_q": self._z_q[idx],
            "bpc_feat": self._bpc_feat[idx],
            "ctx_feat": self._ctx_feat[idx],
            "time_emb": self._time_emb[idx],
            "stock_id": self._stock_ids[idx],
            "s1_ids": self._s1_ids[idx],
        }


def create_dataloaders(
    config: GlobalConfig,
    cache_dir: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    def collate_fn(batch):
        return {
            "z_q": torch.stack([b["z_q"] for b in batch]),
            "bpc_feat": torch.stack([b["bpc_feat"] for b in batch]),
            "ctx_feat": torch.stack([b["ctx_feat"] for b in batch]),
            "time_emb": torch.stack([b["time_emb"] for b in batch]),
            "stock_id": torch.stack([b["stock_id"] for b in batch]),
            "s1_ids": torch.stack([b["s1_ids"] for b in batch]),
        }

    train_ds = QlibBPCV4Dataset(config, mode="train", cache_dir=cache_dir)
    val_ds = QlibBPCV4Dataset(config, mode="val", cache_dir=cache_dir)
    test_ds = QlibBPCV4Dataset(config, mode="test", cache_dir=cache_dir)

    train_loader = DataLoader(train_ds, batch_size=config.train.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.train.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config.train.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)

    return train_loader, val_loader, test_loader
