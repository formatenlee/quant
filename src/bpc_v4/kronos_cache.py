"""Kronos 预计算缓存：按标的分片存储 z_q / s1_ids，与 BPC 物化解耦。"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import GlobalConfig
from .cpu_parallel import DEFAULT_CPU_THREADS, configure_torch_cpu_parallelism, parallel_map_ordered, resolve_cpu_threads
from .kronos import KronosTokenizerEncoder, sync_kronos_config

logger = logging.getLogger(__name__)

KRONOS_CACHE_SCHEMA = "kronos_cache_v2"
KRONOS_CACHE_OHLCVA_NORM = "per_window_zscore_clip5_sanitize"

_tls = threading.local()


def _shard_path(cache_dir: Path, qlib_id: str) -> Path:
    safe = qlib_id.replace("/", "_").replace("\\", "_")
    return cache_dir / "instruments" / f"{safe}.npz"


def iter_valid_t_indices(
    series,
    *,
    seq_len: int,
    start_date: str,
    end_date: str,
) -> list[int]:
    """标的在日期窗口内所有可用窗口末端索引 t_idx（与 dataset 物化一致）。"""
    n = len(series.dates)
    if n <= seq_len:
        return []
    start_ts = np.datetime64(start_date)
    end_ts = np.datetime64(end_date)
    out: list[int] = []
    for t_idx in range(seq_len, n):
        anchor = series.dates[t_idx - 1]
        w_start = series.dates[t_idx - seq_len]
        if w_start >= start_ts and anchor <= end_ts:
            out.append(t_idx)
    return out


@dataclass
class KronosCacheMeta:
    schema_version: str
    seq_len: int
    z_q_dim: int
    s1_bits: int
    kronos_local_path: str
    kronos_model_name: str
    provider_uri: str
    start_date: str
    end_date: str
    n_instruments: int
    n_windows: int
    ohlcva_norm: str = ""
    s1_last_unique_sampled: int = 0

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> KronosCacheMeta:
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def cache_meta_issues(cache_dir: Path | str, *, config: GlobalConfig | None = None) -> list[str]:
    """检查缓存目录是否可被当前代码使用；空列表表示兼容。"""
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "meta.json"
    if not meta_path.is_file():
        return [f"meta.json 不存在: {meta_path}"]
    meta = KronosCacheMeta.load(meta_path)
    issues: list[str] = []
    if meta.schema_version != KRONOS_CACHE_SCHEMA:
        issues.append(
            f"schema={meta.schema_version!r}（需要 {KRONOS_CACHE_SCHEMA!r}；"
            "v1 未做 per-window z-score，s1_ids 会坍缩）"
        )
    norm = meta.ohlcva_norm or ""
    if norm != KRONOS_CACHE_OHLCVA_NORM:
        issues.append(
            f"ohlcva_norm={norm!r}（需要 {KRONOS_CACHE_OHLCVA_NORM!r}）"
        )
    if config is not None:
        sync_kronos_config(config)
        if meta.seq_len != config.kronos.seq_len:
            issues.append(f"seq_len cache={meta.seq_len} config={config.kronos.seq_len}")
        if meta.z_q_dim != config.kronos.d_model:
            issues.append(f"z_q_dim cache={meta.z_q_dim} config={config.kronos.d_model}")
        if meta.s1_bits != config.kronos.s1_bits:
            issues.append(f"s1_bits cache={meta.s1_bits} config={config.kronos.s1_bits}")
    return issues


def resolve_kronos_cache_dir(
    *,
    explicit: Path | str | None,
    default_dir: Path,
    config: GlobalConfig,
    allow_live_kronos: bool,
) -> Path | None:
    """
    解析训练用 Kronos 缓存目录。

  - 显式 --kronos-cache-dir：必须兼容，否则报错并提示重建。
  - 默认 data/kronos_cache：仅当 meta 为 v2+sanitize 时自动启用；否则跳过
    （若 allow_live_kronos 则回退在线编码）。
    """
    rebuild_hint = (
        "请运行: python -m quant_cursor.bpc_v4.precompute_kronos "
        f"--kronos-path <path> --output-dir {default_dir} --force-rebuild"
    )
    if explicit is not None:
        path = Path(explicit)
        issues = cache_meta_issues(path, config=config)
        if issues:
            raise ValueError(
                f"Kronos 缓存不兼容 ({path}): " + "; ".join(issues) + "。" + rebuild_hint
            )
        return path

    if not (default_dir / "meta.json").is_file():
        return None

    issues = cache_meta_issues(default_dir, config=config)
    if not issues:
        logger.info("Auto-using Kronos cache: %s", default_dir)
        return default_dir

    logger.warning(
        "跳过默认 Kronos 缓存 %s（%s）。%s",
        default_dir,
        "; ".join(issues),
        rebuild_hint,
    )
    if allow_live_kronos:
        logger.warning(
            "已启用 --allow-live-kronos：物化阶段将在线 CPU 编码 Kronos（慢，仅调试/过渡）。"
            "正式训练请先 precompute_kronos 生成 v2 缓存。"
        )
    return None


class KronosPrecomputeStore:
    """只读 Kronos 缓存；lookup(qlib_id, t_idx) -> (z_q, s1_ids)。"""

    def __init__(self, cache_dir: Path, meta: KronosCacheMeta):
        self.cache_dir = Path(cache_dir)
        self.meta = meta
        self._shards: dict[str, dict] = {}
        self._shard_lock = threading.Lock()

    @classmethod
    def open(cls, cache_dir: Path | str) -> KronosPrecomputeStore:
        cache_dir = Path(cache_dir)
        meta_path = cache_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Kronos 缓存不存在: {meta_path}")
        meta = KronosCacheMeta.load(meta_path)
        issues = cache_meta_issues(cache_dir)
        if issues:
            raise ValueError(
                f"不支持的 Kronos 缓存 ({cache_dir}): " + "; ".join(issues) + "。"
                "请 precompute_kronos --force-rebuild 重建。"
            )
        return cls(cache_dir, meta)

    def validate_compatible(self, config: GlobalConfig) -> None:
        sync_kronos_config(config)
        issues = cache_meta_issues(self.cache_dir, config=config)
        if issues:
            raise ValueError(
                "Kronos 缓存与当前配置不兼容: " + "; ".join(issues) + "。"
                "请用 precompute_kronos 重新生成或调整 config。"
            )
        logger.info(
            "Kronos cache OK: %s (%d instruments, %d windows, seq_len=%d)",
            self.cache_dir,
            self.meta.n_instruments,
            self.meta.n_windows,
            self.meta.seq_len,
        )

    def _load_shard(self, qlib_id: str) -> dict:
        cached = self._shards.get(qlib_id)
        if cached is not None:
            return cached
        with self._shard_lock:
            if qlib_id in self._shards:
                return self._shards[qlib_id]
            path = _shard_path(self.cache_dir, qlib_id)
            if not path.is_file():
                raise KeyError(f"Kronos 缓存缺少标的分片: {qlib_id} ({path})")
            data = np.load(path)
            t_indices = data["t_indices"].astype(np.int64)
            shard = {
                "t_indices": t_indices,
                "z_q": torch.from_numpy(data["z_q"]),
                "s1_ids": torch.from_numpy(data["s1_ids"]),
                "index": {int(t): i for i, t in enumerate(t_indices)},
            }
            self._shards[qlib_id] = shard
            return shard

    def preload(self, qlib_ids: set[str] | list[str]) -> int:
        """并行物化前预加载分片，避免热路径抢锁 + 重复磁盘读。"""
        unique = sorted(set(qlib_ids))
        loaded = 0
        for qlib_id in unique:
            if qlib_id not in self._shards and self.has_instrument(qlib_id):
                self._load_shard(qlib_id)
                loaded += 1
        if loaded:
            logger.info("Kronos cache preloaded %d instrument shards (%d total in memory)", loaded, len(self._shards))
        return loaded

    def has_instrument(self, qlib_id: str) -> bool:
        return _shard_path(self.cache_dir, qlib_id).is_file()

    def get(self, qlib_id: str, t_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard = self._load_shard(qlib_id)
        row = shard["index"].get(int(t_idx))
        if row is None:
            raise KeyError(f"Kronos 缓存无窗口: {qlib_id} t_idx={t_idx}")
        return shard["z_q"][row], shard["s1_ids"][row]


def release_live_kronos_encoders() -> None:
    """Phase 1 live 编码后释放线程内 Kronos 编码器，减轻 fork 子进程负担。"""
    enc = getattr(_tls, "kronos_encoder", None)
    if enc is not None:
        try:
            del _tls.kronos_encoder
        except AttributeError:
            pass
        del enc


def _thread_kronos_encoder(config: GlobalConfig) -> KronosTokenizerEncoder:
    enc = getattr(_tls, "kronos_encoder", None)
    if enc is None:
        enc = KronosTokenizerEncoder(
            model_name=config.kronos.model_name,
            local_path=config.kronos.local_path or None,
            device="cpu",
        )
        _tls.kronos_encoder = enc
    return enc


def _batch_encode_t_indices(
    arr: np.ndarray,
    t_list: list[int],
    *,
    seq_len: int,
    enc: KronosTokenizerEncoder,
    batch_size: int,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """按 batch_size 批量 Kronos 编码，返回 t_idx -> (z_q[T,D], s1_ids[T])。"""
    if not t_list:
        return {}
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    sorted_t = sorted(t_list)
    for batch_start in range(0, len(sorted_t), batch_size):
        batch_t = sorted_t[batch_start : batch_start + batch_size]
        windows: list[torch.Tensor] = []
        for t_idx in batch_t:
            window = arr[t_idx - seq_len : t_idx]
            ohlcv_abs = torch.from_numpy(window[:, :5]).float()
            amount = torch.from_numpy(window[:, 5:6]).float()
            windows.append(torch.cat([ohlcv_abs, amount], dim=-1))
        ohlcva_batch = torch.stack(windows, dim=0)
        z_q, s1_ids, _ = enc.encode(ohlcva_batch)
        z_np = z_q.detach().cpu().numpy().astype(np.float32)
        s1_np = s1_ids.detach().cpu().numpy().astype(np.int32)
        for j, t_idx in enumerate(batch_t):
            out[t_idx] = (z_np[j], s1_np[j])
    return out


class LiveKronosLookup:
    """在线 Kronos 批量编码结果（物化前按标的预计算，接口同 KronosPrecomputeStore.get）。"""

    def __init__(self) -> None:
        self._z: dict[tuple[str, int], np.ndarray] = {}
        self._s1: dict[tuple[str, int], np.ndarray] = {}

    def put(self, qlib_id: str, t_idx: int, z_q: np.ndarray, s1_ids: np.ndarray) -> None:
        key = (qlib_id, int(t_idx))
        self._z[key] = z_q
        self._s1[key] = s1_ids

    def __len__(self) -> int:
        return len(self._z)

    def get(self, qlib_id: str, t_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (qlib_id, int(t_idx))
        if key not in self._z:
            raise KeyError(f"Live Kronos 无窗口: {qlib_id} t_idx={t_idx}")
        return torch.from_numpy(self._z[key]), torch.from_numpy(self._s1[key])


class FrozenKronosShardLookup:
    """从预加载 cache 分片构建的只读 lookup；无 threading.Lock，可 pickle 供 spawn 物化。"""

    def __init__(self, shards: dict[str, dict]) -> None:
        self._shards = shards

    def get(self, qlib_id: str, t_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard = self._shards[qlib_id]
        row = shard["index"].get(int(t_idx))
        if row is None:
            raise KeyError(f"Kronos 缓存无窗口: {qlib_id} t_idx={t_idx}")
        return torch.from_numpy(shard["z_q"][row]), torch.from_numpy(shard["s1_ids"][row])


def freeze_kronos_cache_for_spawn(
    kronos_cache: KronosPrecomputeStore,
    samples: list[tuple[str, int]],
) -> FrozenKronosShardLookup:
    """将已预加载分片冻结为 numpy 只读结构，供 spawn 物化（避免 fork 继承 qlib FD）。"""
    needed_syms = {sym for sym, _ in samples}
    shards: dict[str, dict] = {}
    for sym in sorted(needed_syms):
        if not kronos_cache.has_instrument(sym):
            raise KeyError(f"Kronos 缓存缺少标的分片: {sym}")
        raw = kronos_cache._load_shard(sym)
        shards[sym] = {
            "index": raw["index"],
            "z_q": raw["z_q"].detach().cpu().numpy(),
            "s1_ids": raw["s1_ids"].detach().cpu().numpy(),
        }
    n_windows = sum(len(s["index"]) for s in shards.values())
    logger.info(
        "Kronos cache frozen for spawn materialize: %d instruments, %d windows",
        len(shards),
        n_windows,
    )
    return FrozenKronosShardLookup(shards)


def freeze_live_lookup_for_samples(
    lookup: LiveKronosLookup,
    samples: list[tuple[str, int]],
) -> FrozenKronosShardLookup:
    """将 LiveKronosLookup 中本批样本所需窗口压成 per-symbol 分片（可 pickle / fork COW）。"""
    by_sym: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for sym, t_idx in samples:
        key = (sym, int(t_idx))
        if key not in lookup._z:
            raise KeyError(f"Live Kronos 无窗口: {sym} t_idx={t_idx}")
        by_sym.setdefault(sym, []).append((int(t_idx), lookup._z[key], lookup._s1[key]))
    shards: dict[str, dict] = {}
    for sym, entries in by_sym.items():
        entries.sort(key=lambda x: x[0])
        t_indices = np.array([e[0] for e in entries], dtype=np.int64)
        z_q = np.stack([e[1] for e in entries], axis=0).astype(np.float32, copy=False)
        s1_ids = np.stack([e[2] for e in entries], axis=0).astype(np.int32, copy=False)
        shards[sym] = {
            "index": {int(t): i for i, t in enumerate(t_indices)},
            "z_q": np.ascontiguousarray(z_q),
            "s1_ids": np.ascontiguousarray(s1_ids),
        }
    logger.info(
        "Live Kronos lookup frozen for materialize: %d instruments, %d windows",
        len(shards),
        len(samples),
    )
    return FrozenKronosShardLookup(shards)


KronosLookup = LiveKronosLookup | FrozenKronosShardLookup


def build_live_kronos_lookup(
    samples: list[tuple[str, int]],
    store,
    config: GlobalConfig,
    *,
    cpu_threads: int = DEFAULT_CPU_THREADS,
    batch_size: int = 64,
) -> LiveKronosLookup:
    """
    训练物化用：与 precompute_kronos 相同——按标的并行 + batch 编码。

    避免逐样本 batch=1 在线编码（GIL 下多线程仅 ~200% CPU）。
    """
    threads = resolve_cpu_threads(cpu_threads)
    configure_torch_cpu_parallelism(threads)
    seq_len = config.kronos.seq_len

    by_sym: dict[str, set[int]] = {}
    for sym, t_idx in samples:
        by_sym.setdefault(sym, set()).add(int(t_idx))
    instruments = sorted(by_sym.keys())
    lookup = LiveKronosLookup()

    logger.info(
        "[live-kronos] batch encode: %d instruments, %d sample windows "
        "(%d threads, batch_size=%d)",
        len(instruments),
        len(samples),
        threads,
        batch_size,
    )

    def _worker(_i: int, qlib_id: str) -> tuple[int, int]:
        series = store._cache[qlib_id]
        t_list = sorted(by_sym[qlib_id])
        enc = _thread_kronos_encoder(config)
        encoded = _batch_encode_t_indices(
            series.ohlcva,
            t_list,
            seq_len=seq_len,
            enc=enc,
            batch_size=batch_size,
        )
        for t_idx, (z_q, s1_ids) in encoded.items():
            lookup.put(qlib_id, t_idx, z_q, s1_ids)
        return _i, len(encoded)

    counts = parallel_map_ordered(
        _worker,
        instruments,
        num_threads=threads,
        desc="Live Kronos encode",
    )
    logger.info("[live-kronos] encoded %d windows", sum(counts))
    return lookup


def _encode_instrument_shard(
    qlib_id: str,
    *,
    store,
    config: GlobalConfig,
    out_dir: Path,
    batch_size: int,
) -> int:
    """编码单标的并写入 NPZ；返回窗口数。"""
    series = store._cache[qlib_id]
    seq_len = config.kronos.seq_len
    t_list = iter_valid_t_indices(
        series,
        seq_len=seq_len,
        start_date=config.qlib.start_date,
        end_date=config.qlib.end_date,
    )
    if not t_list:
        logger.warning("标的无有效窗口: %s", qlib_id)
        return 0

    kronos = _thread_kronos_encoder(config)
    arr = series.ohlcva
    encoded = _batch_encode_t_indices(
        arr, t_list, seq_len=seq_len, enc=kronos, batch_size=batch_size
    )
    z_all = np.stack([encoded[t][0] for t in t_list], axis=0)
    s1_all = np.stack([encoded[t][1] for t in t_list], axis=0)
    s1_last_u = len(np.unique(s1_all[:, -1]))
    if s1_last_u <= 2 and len(t_list) > 2:
        logger.warning(
            "Kronos shard %s: s1_last unique=%d for %d windows (check OHLCVA normalization)",
            qlib_id,
            s1_last_u,
            len(t_list),
        )
    shard_path = _shard_path(out_dir, qlib_id)
    np.savez_compressed(
        shard_path,
        t_indices=np.array(t_list, dtype=np.int32),
        z_q=z_all,
        s1_ids=s1_all,
    )
    return len(t_list)


def sample_s1_last_unique(
    cache_dir: Path,
    *,
    max_shards: int = 64,
    max_windows: int = 50_000,
) -> tuple[int, int]:
    """抽样统计缓存中 s1 末 token 多样性。"""
    inst_dir = cache_dir / "instruments"
    paths = sorted(inst_dir.glob("*.npz"))
    if not paths:
        return 0, 0
    if len(paths) > max_shards:
        rng = np.random.default_rng(42)
        paths = list(rng.choice(paths, size=max_shards, replace=False))
    tokens: list[int] = []
    for p in paths:
        s1 = np.load(p)["s1_ids"]
        tokens.extend(int(x) for x in s1[:, -1].tolist())
        if len(tokens) >= max_windows:
            tokens = tokens[:max_windows]
            break
    return len(set(tokens)), len(tokens)


def build_kronos_cache(
    store,
    config: GlobalConfig,
    out_dir: Path,
    *,
    batch_size: int = 64,
    device: str = "cpu",
    force: bool = False,
    instruments: Optional[list[str]] = None,
    cpu_threads: int = DEFAULT_CPU_THREADS,
) -> KronosCacheMeta:
    """
    为 store 内所有标的预计算 Kronos z_q / s1_ids，写入 out_dir。

    纯 CPU 计算；多线程按标的并行（默认 4 线程）。
    每个标的一个 NPZ 分片：t_indices, z_q [N,T,D], s1_ids [N,T]。
    """
    if device != "cpu":
        logger.info("Kronos 预计算固定使用 CPU（忽略 device=%s）", device)

    out_dir = Path(out_dir)
    inst_dir = out_dir / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)

    sync_kronos_config(config)
    seq_len = config.kronos.seq_len
    threads = resolve_cpu_threads(cpu_threads)
    configure_torch_cpu_parallelism(threads)

    targets = instruments or list(store._cache.keys())
    total_windows = 0
    n_done = 0
    to_encode: list[str] = []

    for j, qlib_id in enumerate(targets, 1):
        if qlib_id not in store._cache:
            logger.warning("跳过未加载标的: %s", qlib_id)
            continue

        shard_path = _shard_path(out_dir, qlib_id)
        if shard_path.is_file() and not force:
            data = np.load(shard_path)
            total_windows += int(data["t_indices"].shape[0])
            n_done += 1
            if j % 50 == 0 or j == len(targets):
                logger.info("[kronos-cache] skip existing %d/%d, windows=%d", j, len(targets), total_windows)
            continue

        to_encode.append(qlib_id)

    if to_encode:
        logger.info(
            "[kronos-cache] encoding %d instruments on CPU (%d threads, batch_size=%d)",
            len(to_encode),
            threads,
            batch_size,
        )

        def _worker(_i: int, qlib_id: str) -> tuple[int, int]:
            n_win = _encode_instrument_shard(
                qlib_id,
                store=store,
                config=config,
                out_dir=out_dir,
                batch_size=batch_size,
            )
            return _i, n_win

        win_counts = parallel_map_ordered(
            _worker,
            to_encode,
            num_threads=threads,
            desc="Kronos precompute",
        )
        for n_win in win_counts:
            if n_win > 0:
                n_done += 1
                total_windows += n_win
        logger.info(
            "[kronos-cache] encoded %d instruments, total_windows=%d",
            len(to_encode),
            total_windows,
        )

    s1_unique, s1_sampled = sample_s1_last_unique(out_dir)
    logger.info(
        "Kronos cache s1_last diversity (sampled): unique=%d / windows=%d",
        s1_unique,
        s1_sampled,
    )
    if s1_sampled > 0 and s1_unique <= 2:
        raise RuntimeError(
            f"Kronos 缓存 s1_last unique={s1_unique} 过低（sampled={s1_sampled}，"
            f"需要 >2）。请确认 encode 前已做 per-window z-score + clip，"
            "并 --force-rebuild 重建缓存。"
        )
    if s1_sampled > 0 and s1_unique < max(20, min(100, total_windows // 1000)):
        logger.warning(
            "Kronos cache s1_last diversity 偏低: unique=%d sampled=%d（未 fail，但建议抽查）",
            s1_unique,
            s1_sampled,
        )

    meta = KronosCacheMeta(
        schema_version=KRONOS_CACHE_SCHEMA,
        seq_len=seq_len,
        z_q_dim=config.kronos.d_model,
        s1_bits=config.kronos.s1_bits,
        kronos_local_path=config.kronos.local_path or "",
        kronos_model_name=config.kronos.model_name,
        provider_uri=str(config.qlib.provider_uri),
        start_date=config.qlib.start_date,
        end_date=config.qlib.end_date,
        n_instruments=n_done,
        n_windows=total_windows,
        ohlcva_norm=KRONOS_CACHE_OHLCVA_NORM,
        s1_last_unique_sampled=s1_unique,
    )
    meta.save(out_dir / "meta.json")
    logger.info("Kronos cache saved: %s (%d instruments, %d windows)", out_dir, n_done, total_windows)
    return meta
