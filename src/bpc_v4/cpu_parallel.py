"""CPU 预处理并行工具（物化 / Kronos 预计算；优先 joblib 跨平台）。"""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

DEFAULT_CPU_THREADS = 4
DEFAULT_MATERIALIZE_CHUNK_SIZE = 256

T = TypeVar("T")
R = TypeVar("R")

try:
    from joblib import Parallel, delayed

    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False


def resolve_cpu_threads(num_threads: int | None) -> int:
    if num_threads is None:
        return DEFAULT_CPU_THREADS
    return max(1, int(num_threads))


def raise_nofile_soft_limit(min_soft: int = 65536) -> None:
    """提高进程可打开文件数软上限（物化多进程前调用，减轻 EMFILE）。"""
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(max(min_soft, soft), hard)
        if target > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            logger.info("Raised RLIMIT_NOFILE soft %d -> %d (hard=%d)", soft, target, hard)
    except Exception as exc:
        logger.debug("RLIMIT_NOFILE unchanged: %s", exc)


def can_use_process_pool() -> bool:
    """Linux fork：子进程 COW 继承 store / kronos 预加载，可绕过 GIL。"""
    if sys.platform == "win32":
        return False
    try:
        return mp.get_start_method(allow_none=True) == "fork"
    except RuntimeError:
        return False


def parallel_backend_label(*, use_processes: bool, mp_ctx: dict | None = None) -> str:
    if use_processes and mp_ctx is not None:
        return "joblib-loky (Windows, ctx per worker)"
    if use_processes and can_use_process_pool():
        return "fork-processes (Linux)"
    if _JOBLIB_AVAILABLE:
        return "joblib-threading"
    return "thread-pool"


def materialize_chunk_size(n_items: int, num_workers: int) -> int:
    """物化分块大小：控制任务数，减少调度开销。"""
    if n_items <= 0:
        return DEFAULT_MATERIALIZE_CHUNK_SIZE
    workers = max(1, num_workers)
    target_tasks = max(workers * 8, 32)
    chunk = max(64, (n_items + target_tasks - 1) // target_tasks)
    return min(chunk, 2048)


def configure_torch_cpu_parallelism(num_workers: int) -> None:
    """
    每个 worker 进程/线程内限制 PyTorch/MKL intra-op=1，避免 OpenMP 过度订阅。
    """
    if num_workers <= 1:
        return
    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception as exc:
        logger.warning("无法限制 torch 线程数: %s", exc)


def _process_pool_torch_init() -> None:
    configure_torch_cpu_parallelism(1)


def _materialize_loky_worker(chunk_start: int, ctx: dict) -> list:
    """joblib loky 回退：每任务注入物化上下文。"""
    from . import dataset as _ds_mod

    _ds_mod._init_spawn_materialize_worker(ctx)
    return _ds_mod._materialize_chunk_spawn(chunk_start)


def _run_joblib_chunks(
    chunk_fn: Callable[[int], list[tuple[int, R]]],
    *,
    chunk_starts: list[int],
    num_workers: int,
    use_processes: bool,
    mp_ctx: dict | None,
) -> list[R | None]:
    backend = "loky" if use_processes else "threading"
    if use_processes and mp_ctx is not None:
        partials = Parallel(n_jobs=num_workers, backend=backend, prefer="processes")(
            delayed(_materialize_loky_worker)(start, mp_ctx) for start in chunk_starts
        )
    else:
        partials = Parallel(n_jobs=num_workers, backend=backend, prefer="threads")(
            delayed(chunk_fn)(start) for start in chunk_starts
        )
    return partials  # type: ignore[return-value]


def run_parallel_chunks(
    chunk_fn: Callable[[int], list[tuple[int, R]]],
    *,
    n_items: int,
    chunk_size: int,
    num_workers: int,
    desc: str = "",
    use_processes: bool = False,
    progress_log_every: int = 0,
    mp_ctx: dict | None = None,
) -> list[R]:
    """
    按 chunk_start 并行执行 chunk_fn，返回与 n_items 等长的有序结果。

    use_processes=True：Linux 用 fork ProcessPool；Windows 物化可传 mp_ctx 走 joblib loky。
    use_processes=False：joblib threading（跨平台）或 ThreadPoolExecutor 回退。
    """
    n = n_items
    if n == 0:
        return []
    chunk = max(1, min(chunk_size, n))
    chunk_starts = list(range(0, n, chunk))

    if num_workers <= 1 or len(chunk_starts) == 1:
        results: list[R | None] = [None] * n
        for start in chunk_starts:
            for idx, val in chunk_fn(start):
                results[idx] = val
        return results  # type: ignore[return-value]

    workers = min(num_workers, len(chunk_starts))
    backend = parallel_backend_label(use_processes=use_processes, mp_ctx=mp_ctx)
    if desc:
        logger.info(
            "%s: %d items, %d workers, %s, chunk_size=%d (%d tasks)",
            desc,
            n,
            workers,
            backend,
            chunk,
            len(chunk_starts),
        )

    if use_processes:
        configure_torch_cpu_parallelism(1)
        if mp_ctx is None:
            logger.info("CPU preprocess: fork multiprocessing + torch intra_op=1 per worker")
        else:
            logger.info("CPU preprocess: joblib loky + torch intra_op=1 per worker (Windows)")
    else:
        configure_torch_cpu_parallelism(workers)
        logger.info(
            "CPU preprocess: %s + torch intra_op=1",
            backend,
        )

    results: list[R | None] = [None] * n

    if _JOBLIB_AVAILABLE and (not use_processes or mp_ctx is not None):
        partials = _run_joblib_chunks(
            chunk_fn,
            chunk_starts=chunk_starts,
            num_workers=workers,
            use_processes=use_processes,
            mp_ctx=mp_ctx,
        )
        done = 0
        for partial in partials:
            for idx, val in partial:
                results[idx] = val
            if partial:
                done += len(partial)
                if progress_log_every > 0 and (
                    done % progress_log_every < len(partial) or done >= n
                ):
                    logger.info("Materialized %d/%d samples", min(done, n), n)
        return results  # type: ignore[return-value]

    executor_cls = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    pool_kwargs: dict[str, Any] = {"max_workers": workers}
    if use_processes:
        pool_kwargs["initializer"] = _process_pool_torch_init

    done = 0
    with executor_cls(**pool_kwargs) as executor:
        for partial in executor.map(chunk_fn, chunk_starts):
            for idx, val in partial:
                results[idx] = val
            if partial:
                done += len(partial)
                if progress_log_every > 0 and (
                    done % progress_log_every < len(partial) or done >= n
                ):
                    logger.info("Materialized %d/%d samples", min(done, n), n)
    return results  # type: ignore[return-value]


def run_parallel_spawn_chunks(
    chunk_fn: Callable[[int], list[tuple[int, R]]],
    *,
    init_fn: Callable[[dict], None],
    spawn_ctx: dict,
    n_items: int,
    chunk_size: int,
    num_workers: int,
    desc: str = "",
    progress_log_every: int = 0,
) -> list[R]:
    """
    spawn 多进程分块并行（不 fork，避免继承 qlib 大量 FD 触发 EMFILE）。

    子进程通过 initializer 注入 spawn_ctx（store / kronos lookup / config 等须可 pickle）。
    """
    n = n_items
    if n == 0:
        return []
    chunk = max(1, min(chunk_size, n))
    chunk_starts = list(range(0, n, chunk))

    if num_workers <= 1 or len(chunk_starts) == 1:
        init_fn(spawn_ctx)
        results: list[R | None] = [None] * n
        for start in chunk_starts:
            for idx, val in chunk_fn(start):
                results[idx] = val
        return results  # type: ignore[return-value]

    workers = min(num_workers, len(chunk_starts))
    if desc:
        logger.info(
            "%s: %d items, %d spawn workers, chunk_size=%d (%d tasks)",
            desc,
            n,
            workers,
            chunk,
            len(chunk_starts),
        )
    configure_torch_cpu_parallelism(1)
    logger.info("CPU preprocess: spawn multiprocessing + torch intra_op=1 per worker (no fork)")

    mp_ctx = mp.get_context("spawn")
    results: list[R | None] = [None] * n
    done = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp_ctx,
        initializer=init_fn,
        initargs=(spawn_ctx,),
    ) as executor:
        for partial in executor.map(chunk_fn, chunk_starts):
            for idx, val in partial:
                results[idx] = val
            if partial:
                done += len(partial)
                if progress_log_every > 0 and (
                    done % progress_log_every < len(partial) or done >= n
                ):
                    logger.info("Materialized %d/%d samples", min(done, n), n)
    return results  # type: ignore[return-value]


def run_parallel_thread_chunks(
    chunk_fn: Callable[[int], list[tuple[int, R]]],
    *,
    n_items: int,
    chunk_size: int,
    num_workers: int,
    desc: str = "",
    progress_log_every: int = 0,
) -> list[R]:
    """线程池分块并行：与父进程共享 slim bundle，无 pickle；Phase 1 已用 torch 时比 fork 安全。"""
    n = n_items
    if n == 0:
        return []
    chunk = max(1, min(chunk_size, n))
    chunk_starts = list(range(0, n, chunk))

    if num_workers <= 1 or len(chunk_starts) == 1:
        results: list[R | None] = [None] * n
        for start in chunk_starts:
            for idx, val in chunk_fn(start):
                results[idx] = val
        return results  # type: ignore[return-value]

    workers = min(num_workers, len(chunk_starts))
    if desc:
        logger.info(
            "%s: %d items, %d thread workers, chunk_size=%d (%d tasks)",
            desc,
            n,
            workers,
            chunk,
            len(chunk_starts),
        )
    configure_torch_cpu_parallelism(1)
    logger.info("CPU preprocess: thread-pool + torch intra_op=1 per worker")

    results: list[R | None] = [None] * n
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for partial in executor.map(chunk_fn, chunk_starts):
            for idx, val in partial:
                results[idx] = val
            if partial:
                done += len(partial)
                if progress_log_every > 0 and (
                    done % progress_log_every < len(partial) or done >= n
                ):
                    logger.info("Materialized %d/%d samples", min(done, n), n)
    return results  # type: ignore[return-value]


def run_parallel_fork_chunks(
    chunk_fn: Callable[[int], list[tuple[int, R]]],
    *,
    n_items: int,
    chunk_size: int,
    num_workers: int,
    desc: str = "",
    progress_log_every: int = 0,
) -> list[R]:
    """
    Linux fork 分块并行：依赖父进程已设置的全局物化上下文（COW，无 pickle）。

    调用前须 release qlib FD 且上下文仅为 numpy/纯 Python 结构。
    """
    n = n_items
    if n == 0:
        return []
    chunk = max(1, min(chunk_size, n))
    chunk_starts = list(range(0, n, chunk))

    if num_workers <= 1 or len(chunk_starts) == 1:
        results: list[R | None] = [None] * n
        for start in chunk_starts:
            for idx, val in chunk_fn(start):
                results[idx] = val
        return results  # type: ignore[return-value]

    workers = min(num_workers, len(chunk_starts))
    if desc:
        logger.info(
            "%s: %d items, %d fork workers, chunk_size=%d (%d tasks)",
            desc,
            n,
            workers,
            chunk,
            len(chunk_starts),
        )
    configure_torch_cpu_parallelism(1)
    logger.info("CPU preprocess: fork multiprocessing + torch intra_op=1 per worker (COW bundle)")

    results: list[R | None] = [None] * n
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_process_pool_torch_init) as executor:
        for partial in executor.map(chunk_fn, chunk_starts):
            for idx, val in partial:
                results[idx] = val
            if partial:
                done += len(partial)
                if progress_log_every > 0 and (
                    done % progress_log_every < len(partial) or done >= n
                ):
                    logger.info("Materialized %d/%d samples", min(done, n), n)
    return results  # type: ignore[return-value]


def parallel_map_ordered(
    func: Callable[[int, T], tuple[int, R]],
    items: list[T],
    *,
    num_threads: int,
    desc: str = "",
    chunk_size: int | None = None,
    use_processes: bool = False,
) -> list[R]:
    """兼容旧接口：包装 run_parallel_chunks。"""
    n = len(items)
    if n == 0:
        return []
    chunk = chunk_size if chunk_size is not None else 1
    if chunk <= 1 and not use_processes:
        if num_threads <= 1 or n == 1:
            return [func(i, item)[1] for i, item in enumerate(items)]
        return _parallel_per_item(func, items, workers=min(num_threads, n), desc=desc)

    chunk = max(1, min(chunk, n))

    def _chunk_fn(start: int) -> list[tuple[int, R]]:
        end = min(start + chunk, n)
        return [func(i, items[i]) for i in range(start, end)]

    return run_parallel_chunks(
        _chunk_fn,
        n_items=n,
        chunk_size=chunk,
        num_workers=num_threads,
        desc=desc,
        use_processes=use_processes,
    )


def _parallel_per_item(
    func: Callable[[int, T], tuple[int, R]],
    items: list[T],
    *,
    workers: int,
    desc: str,
) -> list[R]:
    n = len(items)
    if desc:
        logger.info(
            "%s: %d items, %d workers, %s",
            desc,
            n,
            workers,
            parallel_backend_label(use_processes=False),
        )
    results: list[R | None] = [None] * n

    if _JOBLIB_AVAILABLE:

        def _one(i: int) -> tuple[int, R]:
            return func(i, items[i])

        pairs = Parallel(n_jobs=workers, backend="threading", prefer="threads")(
            delayed(_one)(i) for i in range(n)
        )
        for idx, value in pairs:
            results[idx] = value
        return results  # type: ignore[return-value]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(func, i, item) for i, item in enumerate(items)]
        for fut in futures:
            idx, value = fut.result()
            results[idx] = value
    return results  # type: ignore[return-value]
