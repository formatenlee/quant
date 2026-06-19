from __future__ import annotations

import logging
import random
import re
import time
from collections.abc import Callable
from typing import TypeVar

from quant_cursor.utils import sleep_with_jitter

logger = logging.getLogger(__name__)
T = TypeVar("T")


class RateLimitedClient:
    """带延迟、抖动、指数退避与熔断的请求封装。"""

    def __init__(
        self,
        delay: float = 2.0,
        jitter: float = 0.8,
        max_retries: int = 3,
        backoff: float = 8.0,
        batch_pause_every: int = 40,
        batch_pause_seconds: float = 25.0,
        ban_consecutive_failures: int = 5,
        ban_cooldown_seconds: float = 900.0,
    ) -> None:
        self.delay = delay
        self.jitter = jitter
        self.max_retries = max_retries
        self.backoff = backoff
        self.batch_pause_every = batch_pause_every
        self.batch_pause_seconds = batch_pause_seconds
        self.ban_consecutive_failures = ban_consecutive_failures
        self.ban_cooldown_seconds = ban_cooldown_seconds
        self._request_count = 0
        self._consecutive_failures = 0

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                self._consecutive_failures = 0
                self._after_success()
                return result
            except Exception as exc:  # noqa: BLE001 - 网络异常类型多样
                last_error = exc
                self._consecutive_failures += 1
                wait = self.backoff * attempt + random.uniform(0, self.jitter)
                logger.warning(
                    "请求失败 (%s/%s): %s，%.1fs 后重试",
                    attempt,
                    self.max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
                self._maybe_cooldown()

        assert last_error is not None
        raise last_error

    def _maybe_cooldown(self) -> None:
        if (
            self.ban_consecutive_failures > 0
            and self._consecutive_failures >= self.ban_consecutive_failures
            and self._consecutive_failures % self.ban_consecutive_failures == 0
        ):
            logger.error(
                "连续失败 %s 次，疑似被封 IP，熔断暂停 %.0fs ...",
                self._consecutive_failures,
                self.ban_cooldown_seconds,
            )
            time.sleep(self.ban_cooldown_seconds)

    def _after_success(self) -> None:
        self._request_count += 1
        sleep_with_jitter(self.delay, self.jitter)
        if (
            self.batch_pause_every > 0
            and self._request_count % self.batch_pause_every == 0
        ):
            logger.info(
                "已完成 %s 次请求，暂停 %.0fs 以降低封禁风险",
                self._request_count,
                self.batch_pause_seconds,
            )
            time.sleep(self.batch_pause_seconds)

    def pause(self, seconds: float, reason: str) -> None:
        logger.info("%s，额外暂停 %.0fs", reason, seconds)
        time.sleep(seconds)
