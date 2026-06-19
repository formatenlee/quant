"""bpc_v4 物化数据集：磁盘缓存 / share_memory / GPU 驻留。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

BPC_V4_SCHEMA = "bpc_v4"
FIELD_NAMES = ("z_q", "bpc_feat", "ctx_feat", "time_emb", "stock_id", "s1_ids")


class MaterializedBPCV4Dataset(Dataset):
    """CPU 驻留物化张量；支持 share_memory / GPU 包装。"""

    def __init__(
        self,
        *,
        z_q: torch.Tensor,
        bpc_feat: torch.Tensor,
        ctx_feat: torch.Tensor,
        time_emb: torch.Tensor,
        stock_id: torch.Tensor,
        s1_ids: torch.Tensor,
    ):
        self._z_q = z_q
        self._bpc_feat = bpc_feat
        self._ctx_feat = ctx_feat
        self._time_emb = time_emb
        self._stock_id = stock_id
        self._s1_ids = s1_ids
        n = z_q.shape[0]
        for name, t in self._fields().items():
            if t.shape[0] != n:
                raise ValueError(f"{name} batch dim {t.shape[0]} != {n}")

    def _fields(self) -> Dict[str, torch.Tensor]:
        return {
            "z_q": self._z_q,
            "bpc_feat": self._bpc_feat,
            "ctx_feat": self._ctx_feat,
            "time_emb": self._time_emb,
            "stock_id": self._stock_id,
            "s1_ids": self._s1_ids,
        }

    def share_memory_(self) -> MaterializedBPCV4Dataset:
        for name in FIELD_NAMES:
            t = getattr(self, f"_{name}")
            if not t.is_shared():
                setattr(self, f"_{name}", t.share_memory_())
        return self

    def __len__(self) -> int:
        return self._z_q.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {name: getattr(self, f"_{name}")[idx] for name in FIELD_NAMES}


def pin_dataset_share_memory(ds: MaterializedBPCV4Dataset) -> None:
    ds.share_memory_()


def save_materialized_dataset(ds: MaterializedBPCV4Dataset, path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_version": BPC_V4_SCHEMA,
        "n_samples": len(ds),
        "z_q_shape": list(ds._z_q.shape),
        "bpc_feat_dim": ds._bpc_feat.shape[-1],
        "ctx_feat_dim": ds._ctx_feat.shape[-1],
        "time_emb_dim": ds._time_emb.shape[-1],
        "s1_ids_shape": list(ds._s1_ids.shape),
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for name in FIELD_NAMES:
        t = getattr(ds, f"_{name}")
        np.save(path / f"{name}.npy", t.detach().cpu().numpy())

    logger.info("Saved materialized bpc_v4 dataset to %s (%d samples)", path, len(ds))


def load_materialized_dataset(path: str | Path) -> MaterializedBPCV4Dataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"预处理目录不存在: {path}")

    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if meta.get("schema_version") != BPC_V4_SCHEMA:
        raise ValueError(
            f"schema_version {meta.get('schema_version')!r} != {BPC_V4_SCHEMA!r}; "
            "请用 bpc_v4 --save-preprocessed 重新物化"
        )

    def _load(name: str) -> torch.Tensor:
        npy = path / f"{name}.npy"
        if not npy.exists():
            raise FileNotFoundError(f"Missing {name}.npy in {path}")
        return torch.from_numpy(np.load(npy))

    ds = MaterializedBPCV4Dataset(
        z_q=_load("z_q"),
        bpc_feat=_load("bpc_feat"),
        ctx_feat=_load("ctx_feat"),
        time_emb=_load("time_emb"),
        stock_id=_load("stock_id"),
        s1_ids=_load("s1_ids"),
    )
    logger.info("Loaded pre-materialized bpc_v4 dataset from %s (%d samples)", path, len(ds))
    return ds


class GpuCachedBPCV4Dataset(Dataset):
    """一次性上传全量样本到 GPU，训练时零 H2D。"""

    def __init__(self, base: MaterializedBPCV4Dataset, device: str = "cuda"):
        dev = torch.device(device)
        logger.info("Moving %d bpc_v4 samples to %s (one-time upload)...", len(base), dev)
        self._fields_gpu: Dict[str, torch.Tensor] = {}
        for name in FIELD_NAMES:
            self._fields_gpu[name] = getattr(base, f"_{name}").to(dev, non_blocking=True)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        logger.info("GPU cache ready on %s", dev)

    def __len__(self) -> int:
        return self._fields_gpu["z_q"].shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {name: t[idx] for name, t in self._fields_gpu.items()}


class ContiguousBatchBPCV4Dataset(Dataset):
    """CPU 连续 batch 切片，配合多 worker prefetch。"""

    def __init__(
        self,
        base: MaterializedBPCV4Dataset,
        *,
        batch_size: int,
        drop_last: bool = True,
    ):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.n = len(base)
        self._field_names = list(FIELD_NAMES)
        self._field_tensors = [getattr(base, f"_{n}") for n in FIELD_NAMES]
        self.num_batches = (
            self.n // batch_size if drop_last else (self.n + batch_size - 1) // batch_size
        )
        logger.info(
            "ContiguousBatchBPCV4Dataset: %d batches @ %d (%d samples)",
            self.num_batches,
            batch_size,
            self.n,
        )

    def __len__(self) -> int:
        return self.num_batches

    def __getitem__(self, batch_idx: int) -> dict[str, torch.Tensor]:
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, self.n)
        return {name: t[start:end] for name, t in zip(self._field_names, self._field_tensors)}


class BatchedGpuBPCV4Dataset(Dataset):
    """GPU 驻留 batch 切片；训练循环用 iter_batches()。"""

    def __init__(
        self,
        base: MaterializedBPCV4Dataset | GpuCachedBPCV4Dataset,
        device: str = "cuda",
        batch_size: int = 256,
        drop_last: bool = True,
        shuffle: bool = True,
    ):
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        if isinstance(base, GpuCachedBPCV4Dataset):
            self._field_tensors = [base._fields_gpu[n] for n in FIELD_NAMES]
        else:
            self._field_tensors = [
                getattr(base, f"_{n}").to(self.device, non_blocking=True) for n in FIELD_NAMES
            ]
            if self.device.type == "cuda":
                torch.cuda.synchronize()

        self._field_names = list(FIELD_NAMES)
        self.n = self._field_tensors[0].shape[0]
        self.num_batches = (
            self.n // batch_size if drop_last else (self.n + batch_size - 1) // batch_size
        )
        self._batch_order: list[int] = list(range(self.num_batches))
        if shuffle:
            self.on_epoch_begin()

        logger.info(
            "BatchedGpuBPCV4Dataset: %d batches @ %d on %s",
            self.num_batches,
            batch_size,
            self.device,
        )

    def on_epoch_begin(self) -> None:
        if self.shuffle:
            self._batch_order = torch.randperm(self.num_batches).tolist()
        else:
            self._batch_order = list(range(self.num_batches))

    def __len__(self) -> int:
        return self.num_batches

    def get_batch(self, epoch_step: int) -> dict[str, torch.Tensor]:
        bi = self._batch_order[epoch_step]
        start = bi * self.batch_size
        end = min(start + self.batch_size, self.n)
        return {name: t[start:end] for name, t in zip(self._field_names, self._field_tensors)}

    def iter_batches(self) -> Iterable[dict[str, torch.Tensor]]:
        for i in range(self.num_batches):
            yield self.get_batch(i)

    def __getitem__(self, batch_idx: int) -> dict[str, torch.Tensor]:
        return self.get_batch(batch_idx)
